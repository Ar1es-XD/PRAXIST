"""Product-usage integration contracts for consent and lifecycle projection."""

from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from praxist.cli import product_usage as product_usage_cli
from praxist.infrastructure import product_usage as product_usage_infrastructure
from praxist.infrastructure.product_usage import (
    ProductUsageObserver,
    _prune_run_states,
    _release_run_state_lock,
    _try_acquire_run_state_lock,
    product_usage_notice,
)
from praxist.plugins.workflow_stages.research_loop.backend import generation_boundary
from praxist.plugins.workflow_stages.research_loop.lifecycle import (
    PeerLifecycleSummary,
    ResearchRunLifecycleObserver,
    record_generation_finished_safely,
    summarize_generation_results,
)
from praxist.product_usage.batching import parse_batch_bytes
from praxist.product_usage.client import UploadCoordinator, UsageSdk
from praxist.product_usage.consent import ConsentDecision, ConsentStatus, ConsentStore
from praxist.product_usage.identity import EnvironmentIdentityStore
from praxist.product_usage.notice import consent_notice_v2
from praxist.product_usage.outbox import Outbox
from praxist.product_usage.transport import (
    DEV_COLLECTOR_ENDPOINT,
    PRODUCTION_COLLECTOR_ENDPOINT,
    DevHttpBatchSender,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_COLLECTOR = REPO_ROOT / "scripts" / "dev" / "run_product_usage_collector.py"


class ProductUsageNoticeContract(unittest.TestCase):
    def test_client_uses_the_canonical_v2_notice(self) -> None:
        notice = product_usage_notice()
        normalized_notice = " ".join(notice.split())

        self.assertEqual(notice, consent_notice_v2())
        self.assertIn("help improve Praxist", notice)
        self.assertIn("environment_id", notice)
        self.assertIn("180 days", notice)
        self.assertNotIn("development placeholder", notice.lower())
        self.assertIn(PRODUCTION_COLLECTOR_ENDPOINT, notice)
        self.assertIn("does not delete already delivered events", normalized_notice)
        for reply in ("Yes", "No", "Agree", "Disagree"):
            self.assertIn(reply, notice)

    def test_local_consent_selector_has_no_preselected_answer(self) -> None:
        with patch.object(product_usage_cli, "select_choice", return_value="No") as select:
            self.assertEqual(product_usage_cli._select_consent_choice(), "No")
        self.assertIsNone(select.call_args.kwargs["default"])


class ProductUsageArchitectureContract(unittest.TestCase):
    def test_adapter_satisfies_generic_lifecycle_port(self) -> None:
        self.assertTrue(issubclass(ProductUsageObserver, ResearchRunLifecycleObserver))

    def test_research_loop_internals_do_not_name_product_usage_adapter(self) -> None:
        paths = (
            REPO_ROOT / "praxist/plugins/workflow_stages/research_loop/stage.py",
            REPO_ROOT / "praxist/plugins/workflow_stages/research_loop/backend/generation_loop.py",
            REPO_ROOT / "praxist/plugins/workflow_stages/research_loop/backend/cohort_runner.py",
        )
        for path in paths:
            self.assertNotIn("product_usage", path.read_text(encoding="utf-8"), path)

    def test_observer_does_not_add_a_peer_scheduler_yield(self) -> None:
        source = (
            REPO_ROOT / "praxist/plugins/workflow_stages/research_loop/backend/cohort_runner.py"
        ).read_text(encoding="utf-8")
        peer_creation = source.index("peer_tasks = [asyncio.create_task(peer.run())")
        following = source[peer_creation : peer_creation + 500]
        self.assertNotIn("asyncio.sleep(0)", following)
        self.assertNotIn("run_lifecycle_observer", source)

    def test_result_projection_ignores_late_duplicates_and_open_payloads(self) -> None:
        summary = summarize_generation_results(
            generation_ordinal=2,
            planned_peer_count=4,
            results=[
                {
                    "peer_id": "gen2_peer0",
                    "stop_reason": "synthesis_closing",
                    "prompt": "private",
                },
                {"peer_id": "gen2_peer1", "success": False, "path": "/private"},
                {"peer_id": "gen2_peer2", "stop_reason": "user_interrupt"},
                {
                    "peer_id": "gen2_peer0",
                    "success": False,
                    "late_result_policy": "quarantined",
                },
                {"peer_id": "gen2_protected_jobs", "success": True},
            ],
        )

        self.assertEqual(summary.peer_completed_count, 1)
        self.assertEqual(summary.peer_failed_count, 1)
        self.assertEqual(summary.peer_cancelled_count, 1)
        self.assertEqual(summary.peer_unknown_count, 1)
        self.assertFalse(hasattr(summary, "results"))

    def test_clean_canonical_peer_termination_is_completed_without_domain_inference(self) -> None:
        summary = summarize_generation_results(
            generation_ordinal=0,
            planned_peer_count=2,
            results=[
                {"peer_id": "gen0_peer0", "stop_reason": "timeout"},
                {"peer_id": "gen0_peer1", "stop_reason": "unknown"},
            ],
        )

        self.assertEqual(summary.peer_completed_count, 2)
        self.assertEqual(summary.peer_unknown_count, 0)

    def test_operator_control_exception_escapes_lifecycle_boundary(self) -> None:
        class BrokenObserver:
            def record_generation_finished(self, _summary: PeerLifecycleSummary) -> None:
                raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            record_generation_finished_safely(
                BrokenObserver(),  # type: ignore[arg-type]
                generation_ordinal=0,
                planned_peer_count=1,
                results=[],
            )

    def test_projection_failure_is_contained_even_without_an_observer(self) -> None:
        record_generation_finished_safely(
            None,
            generation_ordinal=0,
            planned_peer_count=-1,
            results=[object()],  # type: ignore[list-item]
        )

    def test_generation_observation_runs_only_after_a_successful_durable_boundary(self) -> None:
        observed: list[PeerLifecycleSummary] = []

        class Observer:
            def record_generation_finished(self, summary: PeerLifecycleSummary) -> None:
                observed.append(summary)

        loop = SimpleNamespace(
            run_lifecycle_observer=Observer(),
            task_spec=SimpleNamespace(
                generation_policy=SimpleNamespace(cohort_size=1),
            ),
        )

        async def successful_boundary(*_args, **_kwargs) -> None:
            return None

        with patch.object(
            generation_boundary,
            "_complete_generation_boundary",
            new=successful_boundary,
        ):
            asyncio.run(
                generation_boundary.complete_generation_boundary(
                    loop,
                    gen_id=0,
                    pi_agent=None,
                    pi_cfg=None,
                    generation_results=[{"peer_id": "gen0_peer0", "success": True}],
                )
            )

        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0].peer_completed_count, 1)

        async def failed_boundary(*_args, **_kwargs) -> None:
            raise RuntimeError("boundary failed")

        with (
            patch.object(
                generation_boundary,
                "_complete_generation_boundary",
                new=failed_boundary,
            ),
            self.assertRaisesRegex(RuntimeError, "boundary failed"),
        ):
            asyncio.run(
                generation_boundary.complete_generation_boundary(
                    loop,
                    gen_id=1,
                    pi_agent=None,
                    pi_cfg=None,
                    generation_results=[{"peer_id": "gen1_peer0", "success": True}],
                )
            )
        self.assertEqual(len(observed), 1)

    def test_generation_observation_projection_failure_cannot_fail_boundary(self) -> None:
        class BrokenLoop:
            run_lifecycle_observer = object()

            @property
            def task_spec(self):
                raise RuntimeError("observer projection failed")

        generation_boundary.record_completed_generation_observation(
            BrokenLoop(),
            gen_id=0,
            generation_results=[],
        )


class ProductUsageObserverContract(unittest.TestCase):
    def _granted_observer(
        self,
        root: Path,
        *,
        run_name: str = "run",
    ) -> tuple[ProductUsageObserver, list[bytes], Outbox]:
        consent = ConsentStore._at_path_for_tests(root / "consent.json")
        consent.write(ConsentDecision.GRANTED)
        identity = EnvironmentIdentityStore._at_path_for_tests(root / "environment.json")
        outbox = Outbox._at_path_for_tests(root / f"{run_name}.sqlite3")
        upload_outbox = Outbox._at_path_for_tests(outbox.path)
        sdk = UsageSdk(consent, identity_store=identity, _outbox_factory=lambda: outbox)
        sent: list[bytes] = []

        class Sender:
            def send(self, body: bytes) -> set[str]:
                sent.append(body)
                return {str(event.event_id) for event in parse_batch_bytes(body).events}

        observer = ProductUsageObserver(
            run_dir=root / run_name,
            praxist_version="0.2.0",
            sdk=sdk,
            upload_once=UploadCoordinator(consent, upload_outbox, Sender()).flush_once,
            _state_path=root / f"{run_name}-state.json",
        )
        return observer, sent, outbox

    @staticmethod
    def _events(sent: list[bytes]):
        return [event for body in sent for event in parse_batch_bytes(body).events]

    def _close_and_wait(self, observer: ProductUsageObserver) -> None:
        observer.close()
        self.assertTrue(observer._wait_for_idle_for_tests())

    def test_observer_requires_environment_and_grant_identities(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_identity_") as raw:
            root = Path(raw)
            with self.assertRaisesRegex(ValueError, "Environment ID"):
                ProductUsageObserver(
                    run_dir=root / "run-a",
                    praxist_version="0.2.0",
                    sdk=SimpleNamespace(environment_id=None),
                    _state_path=root / "state-a.json",
                )
            with self.assertRaisesRegex(ValueError, "consent grant"):
                ProductUsageObserver(
                    run_dir=root / "run-b",
                    praxist_version="0.2.0",
                    sdk=SimpleNamespace(environment_id=uuid4(), consent_grant_id=None),
                    _state_path=root / "state-b.json",
                )

    def test_lifecycle_boundaries_reject_duplicates_and_post_close_work(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_duplicates_") as raw:
            observer, _, _ = self._granted_observer(Path(raw))
            started = PeerLifecycleSummary.planned(
                generation_ordinal=0,
                planned_peer_count=1,
            )
            finished = PeerLifecycleSummary(
                generation_ordinal=0,
                planned_peer_count=1,
                peer_completed_count=1,
            )
            self.assertTrue(observer.record_run_started(started))
            self.assertFalse(observer.record_run_started(started))
            self.assertTrue(observer.record_generation_finished(finished))
            self.assertFalse(observer.record_generation_finished(finished))
            self.assertTrue(observer.record_run_finished(active_duration_seconds=1))
            self.assertFalse(observer.record_run_finished(active_duration_seconds=1))
            self._close_and_wait(observer)
            self.assertFalse(observer.record_run_started(started))
            self.assertFalse(
                observer.record_generation_finished(
                    PeerLifecycleSummary(
                        generation_ordinal=1,
                        planned_peer_count=1,
                        peer_completed_count=1,
                    )
                )
            )

    def test_terminal_duration_is_capped_on_the_wire(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_duration_") as raw:
            observer, sent, _ = self._granted_observer(Path(raw))
            self.assertTrue(
                observer.record_run_finished(
                    active_duration_seconds=(43_201 * 60),
                    failed=False,
                )
            )
            self._close_and_wait(observer)

            event = self._events(sent)[0]
            self.assertEqual(event.active_duration_minutes, 43_200)
            self.assertTrue(event.duration_capped)

    def test_upload_scheduler_and_upload_failure_remain_observer_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_upload_") as raw:
            root = Path(raw)
            scheduled: list[object] = []

            class Sdk:
                environment_id = uuid4()
                consent_grant_id = "00000000-0000-4000-8000-000000000001"

                @staticmethod
                def capture(_event: object, *, expected_grant_id: str) -> bool:
                    return bool(expected_grant_id)

                @staticmethod
                def close() -> None:
                    return None

            observer = ProductUsageObserver(
                run_dir=root / "run-a",
                praxist_version="0.2.0",
                sdk=Sdk(),
                upload_once=lambda: 1,
                schedule_upload=lambda upload: scheduled.append(upload),
                _state_path=root / "state-a.json",
            )
            observer.record_run_started(
                PeerLifecycleSummary.planned(generation_ordinal=0, planned_peer_count=1)
            )
            self._close_and_wait(observer)
            self.assertEqual(len(scheduled), 1)

            def fail_upload() -> int:
                raise OSError("offline")

            observer = ProductUsageObserver(
                run_dir=root / "run-b",
                praxist_version="0.2.0",
                sdk=Sdk(),
                upload_once=fail_upload,
                _state_path=root / "state-b.json",
            )
            observer.record_run_started(
                PeerLifecycleSummary.planned(generation_ordinal=0, planned_peer_count=1)
            )
            self._close_and_wait(observer)

    def test_private_observer_failures_remain_bounded(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_private_failures_") as raw:
            root = Path(raw)

            class Sdk:
                environment_id = uuid4()
                consent_grant_id = "00000000-0000-4000-8000-000000000001"

                @staticmethod
                def close() -> None:
                    return None

            observer = ProductUsageObserver(
                run_dir=root / "run",
                praxist_version="0.2.0",
                sdk=Sdk(),
                _state_path=root / "state.json",
            )
            observer._request_upload()
            observer._upload_worker()
            with patch.object(product_usage_infrastructure, "_MAX_RUN_STATE_FILE_BYTES", 1):
                self.assertFalse(observer._persist_state())
            self._close_and_wait(observer)

            (root / "invalid.json").write_text("{", encoding="utf-8")
            (root / "dangling.json").symlink_to(root / "missing.json")
            self.assertTrue(_prune_run_states(root, keep=root / "keep.json"))

    def test_incompatible_resume_states_start_a_fresh_private_context(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_invalid_resume_") as raw:
            root = Path(raw)
            environment_id = uuid4()

            class Sdk:
                consent_grant_id = "00000000-0000-4000-8000-000000000001"

                def __init__(self) -> None:
                    self.environment_id = environment_id

                @staticmethod
                def close() -> None:
                    return None

            payloads = (
                {"schema_version": 1},
                {"schema_version": 2, "environment_id": str(uuid4())},
                {
                    "schema_version": 2,
                    "environment_id": str(environment_id),
                    "telemetry_run_id": str(uuid4()),
                    "next_sequence": 1,
                    "run_started_emitted": False,
                    "finished_generations": list(
                        range(product_usage_infrastructure._MAX_TRACKED_GENERATIONS + 1)
                    ),
                },
            )
            for index, payload in enumerate(payloads):
                with self.subTest(index=index):
                    state_path = root / f"state-{index}.json"
                    state_path.write_text(json.dumps(payload), encoding="utf-8")
                    observer = ProductUsageObserver(
                        run_dir=root / f"run-{index}",
                        praxist_version="0.2.0",
                        sdk=Sdk(),
                        _state_path=state_path,
                    )
                    self.assertEqual(observer._context.next_sequence, 1)
                    self._close_and_wait(observer)

    def test_projects_only_aggregate_lifecycle_boundaries(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_observer_") as raw:
            observer, sent, outbox = self._granted_observer(Path(raw))
            observer.record_run_started(
                PeerLifecycleSummary.planned(generation_ordinal=0, planned_peer_count=3)
            )
            observer.record_generation_finished(
                PeerLifecycleSummary(
                    generation_ordinal=0,
                    planned_peer_count=3,
                    peer_completed_count=1,
                    peer_cancelled_count=1,
                    peer_failed_count=1,
                )
            )
            observer.record_run_finished(active_duration_seconds=125, failed=True)
            self._close_and_wait(observer)

            events = self._events(sent)
            self.assertEqual(
                [event.event_type for event in events],
                ["run_started", "generation_finished", "run_finished"],
            )
            self.assertEqual([event.event_sequence for event in events], [1, 2, 3])
            self.assertEqual(len({event.environment_id for event in events}), 1)
            self.assertEqual(events[0].peer_planned_count, 3)
            self.assertEqual(events[1].peer_completed_count, 1)
            self.assertEqual(events[1].peer_cancelled_count, 1)
            self.assertEqual(events[1].peer_failed_count, 1)
            self.assertEqual(events[1].error_summaries[0].error_code, "PRX-PEER-RUNTIME")
            self.assertEqual(events[2].active_duration_minutes, 2)
            self.assertEqual(events[2].error_summaries[0].error_code, "PRX-RUN-FAILED")
            self.assertEqual(outbox.count(), 0)

    def test_close_allows_a_bounded_terminal_upload_to_finish(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_close_") as raw:
            root = Path(raw)
            consent = ConsentStore._at_path_for_tests(root / "consent.json")
            consent.write(ConsentDecision.GRANTED)
            identity = EnvironmentIdentityStore._at_path_for_tests(root / "environment.json")
            outbox = Outbox._at_path_for_tests(root / "outbox.sqlite3")
            upload_outbox = Outbox._at_path_for_tests(outbox.path)
            sdk = UsageSdk(
                consent,
                identity_store=identity,
                _outbox_factory=lambda: outbox,
            )
            sent: list[bytes] = []
            send_started = threading.Event()
            allow_send_to_finish = threading.Event()

            class Sender:
                def send(self, body: bytes) -> set[str]:
                    send_started.set()
                    self.assert_send_released()
                    sent.append(body)
                    return {str(event.event_id) for event in parse_batch_bytes(body).events}

                @staticmethod
                def assert_send_released() -> None:
                    if not allow_send_to_finish.wait(timeout=2):
                        raise TimeoutError("test sender was not released")

            observer = ProductUsageObserver(
                run_dir=root / "run",
                praxist_version="0.2.0",
                sdk=sdk,
                upload_once=UploadCoordinator(
                    consent,
                    upload_outbox,
                    Sender(),
                ).flush_once,
                _state_path=root / "state.json",
            )
            observer.record_run_started(
                PeerLifecycleSummary.planned(
                    generation_ordinal=0,
                    planned_peer_count=1,
                )
            )
            self.assertTrue(send_started.wait(timeout=1))
            observer.record_generation_finished(
                PeerLifecycleSummary(
                    generation_ordinal=0,
                    planned_peer_count=1,
                    peer_completed_count=1,
                )
            )
            observer.record_run_finished(active_duration_seconds=1)
            release = threading.Timer(0.35, allow_send_to_finish.set)
            release.start()
            try:
                observer.close()
                canonical = Outbox._at_path_for_tests(outbox.path)
                self.assertEqual(canonical.count(), 0)
                canonical.close()
                self.assertEqual(
                    [event.event_type for event in self._events(sent)],
                    ["run_started", "generation_finished", "run_finished"],
                )
            finally:
                allow_send_to_finish.set()
                release.cancel()
                observer._wait_for_idle_for_tests()

    def test_research_path_only_enqueues_when_capture_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_nonblocking_") as raw:
            capture_started = threading.Event()
            release_capture = threading.Event()

            class BlockingSdk:
                environment_id = uuid4()

                def capture(self, _event: object) -> bool:
                    capture_started.set()
                    release_capture.wait(timeout=2)
                    return True

                def close(self) -> None:
                    return None

            observer = ProductUsageObserver(
                run_dir=Path(raw) / "run",
                praxist_version="0.2.0",
                sdk=BlockingSdk(),
                _state_path=Path(raw) / "state.json",
            )
            started_at = time.monotonic()
            self.assertTrue(
                observer.record_run_started(
                    PeerLifecycleSummary.planned(
                        generation_ordinal=0,
                        planned_peer_count=1,
                    )
                )
            )
            self.assertLess(time.monotonic() - started_at, 0.1)
            self.assertTrue(capture_started.wait(timeout=1))
            release_capture.set()
            self._close_and_wait(observer)

    def test_lifecycle_callbacks_use_only_the_cached_grant_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_cached_grant_") as raw:

            class Sdk:
                environment_id = uuid4()
                grant_reads = 0
                asserted_grant_id = ""

                @property
                def consent_grant_id(self) -> str:
                    self.grant_reads += 1
                    return "00000000-0000-4000-8000-000000000001"

                def capture(self, _event: object, *, expected_grant_id: str) -> bool:
                    self.asserted_grant_id = expected_grant_id
                    return True

                def close(self) -> None:
                    return None

            sdk = Sdk()
            observer = ProductUsageObserver(
                run_dir=Path(raw) / "run",
                praxist_version="0.2.0",
                sdk=sdk,
                _state_path=Path(raw) / "state.json",
            )
            observer.record_run_started(
                PeerLifecycleSummary.planned(generation_ordinal=0, planned_peer_count=1)
            )
            observer.record_run_finished(active_duration_seconds=1)
            self._close_and_wait(observer)

            self.assertEqual(sdk.grant_reads, 1)
            self.assertEqual(
                sdk.asserted_grant_id,
                "00000000-0000-4000-8000-000000000001",
            )

    def test_queued_observation_cannot_cross_withdrawal_and_regrant(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_grant_epoch_") as raw:
            root = Path(raw)
            consent = ConsentStore._at_path_for_tests(root / "consent.json")
            consent.write(ConsentDecision.GRANTED)
            identity = EnvironmentIdentityStore._at_path_for_tests(root / "environment.json")
            outbox = Outbox._at_path_for_tests(root / "outbox.sqlite3")
            sdk = UsageSdk(consent, identity_store=identity, _outbox_factory=lambda: outbox)
            observer = ProductUsageObserver(
                run_dir=root / "run",
                praxist_version="0.2.0",
                sdk=sdk,
                _state_path=root / "state.json",
            )
            persistence_started = threading.Event()
            allow_persistence = threading.Event()
            original_persist = observer._persist_state

            def blocked_persist() -> bool:
                persistence_started.set()
                self.assertTrue(allow_persistence.wait(timeout=2))
                return original_persist()

            with patch.object(observer, "_persist_state", side_effect=blocked_persist):
                self.assertTrue(
                    observer.record_run_started(
                        PeerLifecycleSummary.planned(
                            generation_ordinal=0,
                            planned_peer_count=1,
                        )
                    )
                )
                self.assertTrue(persistence_started.wait(timeout=2))
                withdrawing_sdk = UsageSdk(
                    ConsentStore._at_path_for_tests(consent.path),
                    _outbox_factory=lambda: Outbox._at_path_for_tests(outbox.path),
                )
                self.assertTrue(withdrawing_sdk.withdraw())
                consent.write(ConsentDecision.GRANTED)
                allow_persistence.set()
                self._close_and_wait(observer)

            canonical = Outbox._at_path_for_tests(outbox.path)
            self.assertEqual(canonical.count(), 0)
            canonical.close()

    def test_unset_consent_creates_no_observer_or_run_artifact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_unset_") as raw:
            root = Path(raw)
            run_dir = root / "run"
            consent = ConsentStore._at_path_for_tests(root / "consent.json")
            outbox = Outbox._at_path_for_tests(root / "outbox.sqlite3")

            observer = ProductUsageObserver.create(
                run_dir=run_dir,
                praxist_version="0.2.0",
                consent_store=consent,
                outbox=outbox,
                _state_path=root / "private-state.json",
            )

            self.assertIsNone(observer)
            self.assertFalse(run_dir.exists())
            self.assertFalse((root / "private-state.json").exists())
            self.assertFalse(outbox.path.exists())

    def test_private_run_state_is_bounded_and_prunes_terminal_files_first(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_state_cap_") as raw:
            directory = Path(raw)
            keep = directory / "keep.json"
            keep.write_text('{"run_finished_emitted":false}\n', encoding="utf-8")
            unfinished: list[Path] = []
            for index in range(520):
                terminal = index % 2 == 0
                path = directory / f"state-{index:04d}.json"
                path.write_text(
                    json.dumps({"run_finished_emitted": terminal}) + "\n",
                    encoding="utf-8",
                )
                if not terminal:
                    unfinished.append(path)

            _prune_run_states(directory, keep=keep)

            self.assertTrue(keep.exists())
            self.assertLessEqual(len(list(directory.glob("*.json"))), 512)
            self.assertTrue(all(path.exists() for path in unfinished))

            stale_temporary = directory / ".orphan.json.tmp"
            stale_temporary.write_text("partial", encoding="utf-8")
            old = time.time() - 7200
            os.utime(stale_temporary, (old, old))
            _prune_run_states(directory, keep=keep)
            self.assertFalse(stale_temporary.exists())

    def test_generation_tracking_has_a_fixed_observer_only_bound(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_generation_cap_") as raw:

            class Sdk:
                environment_id = uuid4()

                def capture(self, _event: object) -> bool:
                    return False

                def close(self) -> None:
                    return None

            observer = ProductUsageObserver(
                run_dir=Path(raw) / "run",
                praxist_version="0.2.0",
                sdk=Sdk(),
                _state_path=Path(raw) / "state.json",
            )
            with patch(
                "praxist.infrastructure.product_usage._MAX_TRACKED_GENERATIONS",
                2,
            ):
                for generation in range(2):
                    self.assertTrue(
                        observer.record_generation_finished(
                            PeerLifecycleSummary(
                                generation_ordinal=generation,
                                planned_peer_count=1,
                                peer_completed_count=1,
                            )
                        )
                    )
                self.assertFalse(
                    observer.record_generation_finished(
                        PeerLifecycleSummary(
                            generation_ordinal=2,
                            planned_peer_count=1,
                            peer_completed_count=1,
                        )
                    )
                )
            self._close_and_wait(observer)

    def test_private_run_state_pruning_never_deletes_an_owned_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_owned_state_") as raw:
            directory = Path(raw)
            keep = directory / "keep.json"
            keep.write_text('{"run_finished_emitted":false}\n', encoding="utf-8")
            states = []
            for index in range(512):
                path = directory / f"state-{index:04d}.json"
                path.write_text('{"run_finished_emitted":true}\n', encoding="utf-8")
                states.append(path)
            owned = states[0]
            owned_lock = _try_acquire_run_state_lock(owned)
            self.assertIsNotNone(owned_lock)
            try:
                self.assertTrue(_prune_run_states(directory, keep=keep))
                self.assertTrue(owned.exists())
                self.assertLessEqual(len(list(directory.glob("*.json"))), 512)
            finally:
                assert owned_lock is not None
                _release_run_state_lock(owned_lock)

    def test_oversized_private_state_is_not_loaded(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_state_size_") as raw:
            root = Path(raw)
            state_path = root / "run-state.json"
            state_path.write_bytes(b"{" + b"x" * (600 * 1024) + b"}")

            observer, sent, _ = self._granted_observer(root)
            observer.record_run_started(
                PeerLifecycleSummary.planned(generation_ordinal=0, planned_peer_count=1)
            )
            self._close_and_wait(observer)

            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(self._events(sent)[0].event_sequence, 1)

    def test_only_one_observer_can_own_a_run_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_state_lock_") as raw:
            root = Path(raw)
            environment_id = uuid4()

            class Sdk:
                def __init__(self) -> None:
                    self.environment_id = environment_id

                def capture(self, _event: object) -> bool:
                    return False

                def close(self) -> None:
                    return None

            first = ProductUsageObserver(
                run_dir=root / "run",
                praxist_version="0.2.0",
                sdk=Sdk(),
                _state_path=root / "state.json",
            )
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        "from praxist.infrastructure.product_usage import "
                        "_try_acquire_run_state_lock; "
                        f"print(_try_acquire_run_state_lock(Path({str(root / 'state.json')!r})) "
                        "is None)"
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )
            self.assertEqual(probe.stdout.strip(), "True")
            self.assertEqual((root / ".observer-locks").stat().st_size, 4096)
            self.assertEqual(list(root.glob("*.lock")), [])
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                ProductUsageObserver(
                    run_dir=root / "run",
                    praxist_version="0.2.0",
                    sdk=Sdk(),
                    _state_path=root / "state.json",
                )
            self._close_and_wait(first)

            replacement = ProductUsageObserver(
                run_dir=root / "run",
                praxist_version="0.2.0",
                sdk=Sdk(),
                _state_path=root / "state.json",
            )
            self._close_and_wait(replacement)

    def test_state_is_committed_before_outbox_capture(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_write_ahead_") as raw:
            root = Path(raw)
            state_path = root / "state.json"
            environment_id = uuid4()
            observed_states: list[dict[str, object]] = []

            class FailingSdk:
                def __init__(self) -> None:
                    self.environment_id = environment_id

                def capture(self, _event: object) -> bool:
                    observed_states.append(json.loads(state_path.read_text(encoding="utf-8")))
                    raise RuntimeError("simulated crash window")

                def close(self) -> None:
                    return None

            first = ProductUsageObserver(
                run_dir=root / "run",
                praxist_version="0.2.0",
                sdk=FailingSdk(),
                _state_path=state_path,
            )
            self.assertTrue(
                first.record_run_started(
                    PeerLifecycleSummary.planned(generation_ordinal=0, planned_peer_count=1)
                )
            )
            self._close_and_wait(first)

            self.assertTrue(observed_states[0]["run_started_emitted"])
            self.assertEqual(observed_states[0]["next_sequence"], 2)

            resumed = ProductUsageObserver(
                run_dir=root / "run",
                praxist_version="0.2.0",
                sdk=FailingSdk(),
                _state_path=state_path,
            )
            self.assertFalse(
                resumed.record_run_started(
                    PeerLifecycleSummary.planned(generation_ordinal=0, planned_peer_count=1)
                )
            )
            self._close_and_wait(resumed)

    def test_directory_sync_failure_prevents_outbox_capture(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_state_sync_") as raw:
            root = Path(raw)
            captured: list[object] = []

            class Sdk:
                environment_id = uuid4()

                def capture(self, event: object) -> bool:
                    captured.append(event)
                    return True

                def close(self) -> None:
                    return None

            observer = ProductUsageObserver(
                run_dir=root / "run",
                praxist_version="0.2.0",
                sdk=Sdk(),
                _state_path=root / "state.json",
            )
            with patch(
                "praxist.infrastructure.product_usage._fsync_directory",
                side_effect=OSError("directory sync unavailable"),
            ):
                observer.record_run_started(
                    PeerLifecycleSummary.planned(generation_ordinal=0, planned_peer_count=1)
                )
                self._close_and_wait(observer)

            self.assertEqual(captured, [])

    def test_resume_preserves_sequence_and_generation_idempotence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_resume_") as raw:
            root = Path(raw)
            first, first_sent, _ = self._granted_observer(root)
            first.record_run_started(
                PeerLifecycleSummary.planned(generation_ordinal=0, planned_peer_count=1)
            )
            first.record_generation_finished(
                PeerLifecycleSummary(
                    generation_ordinal=0,
                    planned_peer_count=1,
                    peer_completed_count=1,
                )
            )
            self._close_and_wait(first)

            second, second_sent, _ = self._granted_observer(root)
            duplicate = second.record_generation_finished(
                PeerLifecycleSummary(
                    generation_ordinal=0,
                    planned_peer_count=1,
                    peer_completed_count=1,
                )
            )
            accepted = second.record_generation_finished(
                PeerLifecycleSummary(
                    generation_ordinal=1,
                    planned_peer_count=1,
                    peer_completed_count=1,
                )
            )
            self._close_and_wait(second)

            first_events = self._events(first_sent)
            second_events = self._events(second_sent)
            self.assertFalse(duplicate)
            self.assertTrue(accepted)
            self.assertEqual([event.event_sequence for event in second_events], [3])
            self.assertEqual(second_events[0].generation_ordinal, 1)
            self.assertEqual(second_events[0].telemetry_run_id, first_events[0].telemetry_run_id)

    def test_run_finished_is_idempotent_across_resume(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_finished_") as raw:
            root = Path(raw)
            first, first_sent, _ = self._granted_observer(root)
            self.assertTrue(first.record_run_finished(active_duration_seconds=60, failed=False))
            self._close_and_wait(first)

            second, second_sent, _ = self._granted_observer(root)
            self.assertFalse(second.record_run_finished(active_duration_seconds=120, failed=True))
            self._close_and_wait(second)

            self.assertEqual(
                [event.event_type for event in self._events(first_sent)], ["run_finished"]
            )
            self.assertEqual(second_sent, [])

    def test_resume_reconciles_a_previous_unfinished_process(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_reconcile_") as raw:
            root = Path(raw)
            first, first_sent, _ = self._granted_observer(root)
            first.record_run_started(
                PeerLifecycleSummary.planned(generation_ordinal=0, planned_peer_count=1)
            )
            self._close_and_wait(first)

            second, second_sent, _ = self._granted_observer(root)
            self.assertTrue(second.record_run_finished(active_duration_seconds=90, failed=False))
            self._close_and_wait(second)

            self.assertEqual(
                [event.event_type for event in self._events(first_sent)], ["run_started"]
            )
            self.assertEqual(
                [event.event_type for event in self._events(second_sent)],
                ["run_reconciled"],
            )

    def test_distinct_runs_share_environment_but_not_telemetry_run_id(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_environment_") as raw:
            root = Path(raw)
            events = []
            for run_name in ("run-a", "run-b"):
                observer, sent, _ = self._granted_observer(root, run_name=run_name)
                observer.record_run_started(
                    PeerLifecycleSummary.planned(generation_ordinal=0, planned_peer_count=1)
                )
                observer.close()
                self.assertTrue(observer._wait_for_idle_for_tests())
                events.extend(self._events(sent))

            self.assertEqual(events[0].environment_id, events[1].environment_id)
            self.assertNotEqual(events[0].telemetry_run_id, events[1].telemetry_run_id)

    def test_create_fails_closed_when_sender_selection_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_sender_") as raw:
            root = Path(raw)
            consent = ConsentStore._at_path_for_tests(root / "consent.json")
            consent.write(ConsentDecision.GRANTED)
            with patch(
                "praxist.infrastructure.product_usage.default_batch_sender",
                side_effect=RuntimeError("release gate closed"),
            ):
                observer = ProductUsageObserver.create(
                    run_dir=root / "run",
                    praxist_version="0.2.0",
                    consent_store=consent,
                    _state_path=root / "state.json",
                )
            self.assertIsNone(observer)


class ProductUsageCliContract(unittest.TestCase):
    def test_help_and_notice_commands_cover_available_and_unavailable_states(self) -> None:
        with patch.object(product_usage_cli.argparse.ArgumentParser, "print_help") as help_output:
            self.assertEqual(product_usage_cli.main([]), 0)
        help_output.assert_called_once()

        with (
            patch.object(product_usage_cli, "_product_usage_notice", return_value="NOTICE"),
            patch("builtins.print") as printed,
        ):
            self.assertEqual(product_usage_cli.main(["notice"]), 0)
        printed.assert_called_once_with("NOTICE")

        unavailable = product_usage_cli.ProductUsageUnavailableError("unavailable")
        with (
            patch.object(product_usage_cli, "_product_usage_notice", side_effect=unavailable),
            patch("builtins.print") as printed,
        ):
            self.assertEqual(product_usage_cli.main(["notice"]), 1)
        self.assertIn("unavailable", printed.call_args.args[0])

    def test_consent_command_handles_loading_and_notice_failures(self) -> None:
        unavailable = product_usage_cli.ProductUsageUnavailableError("unavailable")
        args = SimpleNamespace(agent_reply="Yes")
        with (
            patch.object(product_usage_cli, "_usage_sdk", side_effect=unavailable),
            patch("builtins.print") as printed,
        ):
            self.assertEqual(product_usage_cli.cmd_consent(args), 1)
        self.assertIn("unavailable", printed.call_args.args[0])

        sdk = SimpleNamespace(consent_status=ConsentStatus.GRANTED)
        with (
            patch.object(product_usage_cli, "_usage_sdk", return_value=sdk),
            patch.object(product_usage_cli, "_collection_transport_available", return_value=True),
            patch("builtins.print") as printed,
        ):
            self.assertEqual(product_usage_cli.cmd_consent(args), 0)
        self.assertIn("already granted", printed.call_args.args[0])

        sdk = SimpleNamespace(consent_status=ConsentStatus.UNSET)
        with (
            patch.object(product_usage_cli, "_usage_sdk", return_value=sdk),
            patch.object(product_usage_cli, "_collection_transport_available", return_value=True),
            patch.object(product_usage_cli, "_product_usage_notice", side_effect=unavailable),
            patch("builtins.print") as printed,
        ):
            self.assertEqual(product_usage_cli.cmd_consent(args), 1)
        self.assertIn("unavailable", printed.call_args.args[0])

    def test_interactive_consent_handles_cancel_and_supported_choice(self) -> None:
        class FakeSdk:
            consent_status = ConsentStatus.UNSET

            @staticmethod
            def record_direct_choice(_choice: str) -> ConsentStatus:
                return ConsentStatus.UNSET

        args = SimpleNamespace(agent_reply=None)
        with (
            patch.object(product_usage_cli, "_usage_sdk", return_value=FakeSdk()),
            patch.object(product_usage_cli, "_collection_transport_available", return_value=True),
            patch.object(product_usage_cli, "_product_usage_notice", return_value="NOTICE"),
            patch.object(product_usage_cli.sys.stdin, "isatty", return_value=True),
            patch.object(
                product_usage_cli,
                "_select_consent_choice",
                side_effect=product_usage_cli.TerminalInteractionCancelled("cancelled"),
            ),
            patch("builtins.print"),
        ):
            self.assertEqual(product_usage_cli.cmd_consent(args), 0)

        recorded: list[str] = []

        class AcceptedSdk(FakeSdk):
            @staticmethod
            def record_direct_choice(choice: str) -> ConsentStatus:
                recorded.append(choice)
                return ConsentStatus.DENIED

        with (
            patch.object(product_usage_cli, "_usage_sdk", return_value=AcceptedSdk()),
            patch.object(product_usage_cli, "_collection_transport_available", return_value=True),
            patch.object(product_usage_cli, "_product_usage_notice", return_value="NOTICE"),
            patch.object(product_usage_cli.sys.stdin, "isatty", return_value=True),
            patch.object(product_usage_cli, "_select_consent_choice", return_value="No"),
            patch("builtins.print"),
        ):
            self.assertEqual(product_usage_cli.cmd_consent(args), 0)
        self.assertEqual(recorded, ["No"])

    def test_status_and_withdraw_commands_cover_failure_and_text_paths(self) -> None:
        unavailable = product_usage_cli.ProductUsageUnavailableError("unavailable")
        with (
            patch.object(product_usage_cli, "_usage_sdk", side_effect=unavailable),
            patch("builtins.print"),
        ):
            self.assertEqual(product_usage_cli.cmd_status(SimpleNamespace(json_output=False)), 1)
            self.assertEqual(product_usage_cli.cmd_withdraw(SimpleNamespace()), 1)

        with (
            patch.object(
                product_usage_cli,
                "_usage_sdk",
                return_value=SimpleNamespace(consent_status=ConsentStatus.DENIED),
            ),
            patch("builtins.print") as printed,
        ):
            self.assertEqual(product_usage_cli.main(["status"]), 0)
        printed.assert_called_once_with("denied")

        for withdrawn, expected in ((False, 1), (True, 0)):
            with (
                self.subTest(withdrawn=withdrawn),
                patch.object(
                    product_usage_cli,
                    "_usage_sdk",
                    return_value=SimpleNamespace(withdraw=lambda result=withdrawn: result),
                ),
                patch("builtins.print"),
            ):
                self.assertEqual(product_usage_cli.main(["withdraw"]), expected)

    def test_prompt_is_fail_closed_at_each_interactive_boundary(self) -> None:
        unavailable_output = io.StringIO()
        with patch.object(product_usage_cli, "_collection_transport_available", return_value=False):
            self.assertTrue(
                product_usage_cli.prompt_for_consent_if_unset(output_stream=unavailable_output)
            )
        self.assertIn("no usage data will be collected", unavailable_output.getvalue())

        unavailable = product_usage_cli.ProductUsageUnavailableError("unavailable")
        with (
            patch.object(product_usage_cli, "_collection_transport_available", return_value=True),
            patch.object(product_usage_cli, "_usage_sdk", side_effect=unavailable),
        ):
            self.assertTrue(product_usage_cli.prompt_for_consent_if_unset())

        sdk = SimpleNamespace(consent_status=ConsentStatus.UNSET)
        with (
            patch.object(product_usage_cli, "_collection_transport_available", return_value=True),
            patch.object(product_usage_cli, "_usage_sdk", return_value=sdk),
            patch.object(product_usage_cli.sys.stdin, "isatty", return_value=True),
            patch.object(product_usage_cli, "_product_usage_notice", side_effect=unavailable),
        ):
            self.assertTrue(product_usage_cli.prompt_for_consent_if_unset())

        with (
            patch.object(product_usage_cli, "_collection_transport_available", return_value=True),
            patch.object(product_usage_cli, "_usage_sdk", return_value=sdk),
            patch.object(product_usage_cli.sys.stdin, "isatty", return_value=True),
            patch.object(product_usage_cli, "_product_usage_notice", return_value="NOTICE"),
            patch.object(
                product_usage_cli,
                "_select_consent_choice",
                side_effect=product_usage_cli.TerminalInteractionCancelled("cancelled"),
            ),
            patch("builtins.print"),
        ):
            self.assertFalse(product_usage_cli.prompt_for_consent_if_unset())

    def test_lazy_component_loaders_fail_closed(self) -> None:
        with patch.object(product_usage_cli.importlib, "import_module", side_effect=ImportError):
            with self.assertRaises(product_usage_cli.ProductUsageUnavailableError):
                product_usage_cli._usage_sdk()
            with self.assertRaises(product_usage_cli.ProductUsageUnavailableError):
                product_usage_cli._product_usage_notice()

    def test_first_interactive_run_prompts_when_consent_is_unset(self) -> None:
        class FakeSdk:
            consent_status = ConsentStatus.UNSET

            def record_direct_choice(self, choice: str) -> ConsentStatus:
                self.choice = choice
                return ConsentStatus.GRANTED

        sdk = FakeSdk()
        with (
            patch.object(product_usage_cli, "_collection_transport_available", return_value=True),
            patch.object(product_usage_cli, "_usage_sdk", return_value=sdk),
            patch.object(product_usage_cli.sys.stdin, "isatty", return_value=True),
            patch.object(product_usage_cli, "_select_consent_choice", return_value="Yes"),
            patch("builtins.print"),
        ):
            self.assertTrue(product_usage_cli.prompt_for_consent_if_unset())
        self.assertEqual(sdk.choice, "Yes")

    def test_first_interactive_prompt_can_keep_machine_stdout_clean(self) -> None:
        sdk = SimpleNamespace(
            consent_status=ConsentStatus.UNSET,
            record_direct_choice=lambda _choice: ConsentStatus.DENIED,
        )
        output = io.StringIO()
        with (
            patch.object(product_usage_cli, "_collection_transport_available", return_value=True),
            patch.object(product_usage_cli, "_usage_sdk", return_value=sdk),
            patch.object(product_usage_cli.sys.stdin, "isatty", return_value=True),
            patch.object(product_usage_cli, "_product_usage_notice", return_value="NOTICE"),
            patch.object(product_usage_cli, "_select_consent_choice", return_value="No"),
        ):
            self.assertTrue(product_usage_cli.prompt_for_consent_if_unset(output_stream=output))
        self.assertNotIn("NOTICE", output.getvalue())
        self.assertIn("denied", output.getvalue())

    def test_agent_reply_records_only_explicit_keywords(self) -> None:
        class FakeSdk:
            consent_status = ConsentStatus.UNSET

            def __init__(self) -> None:
                self.replies: list[str] = []

            def record_agent_reply(self, reply: str) -> ConsentStatus:
                self.replies.append(reply)
                return ConsentStatus.GRANTED if reply == "Yes" else ConsentStatus.UNSET

        sdk = FakeSdk()
        with (
            patch.object(product_usage_cli, "_collection_transport_available", return_value=True),
            patch.object(product_usage_cli, "_usage_sdk", return_value=sdk),
            patch("builtins.print"),
        ):
            exit_code = product_usage_cli.main(["consent", "--agent-reply", "Yes"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(sdk.replies, ["Yes"])

    def test_denied_user_can_explicitly_regrant_after_full_notice(self) -> None:
        class FakeSdk:
            consent_status = ConsentStatus.DENIED

            def __init__(self) -> None:
                self.replies: list[str] = []

            def record_agent_reply(self, reply: str) -> ConsentStatus:
                self.replies.append(reply)
                return ConsentStatus.GRANTED

        sdk = FakeSdk()
        with (
            patch.object(product_usage_cli, "_collection_transport_available", return_value=True),
            patch.object(product_usage_cli, "_usage_sdk", return_value=sdk),
            patch.object(product_usage_cli, "_product_usage_notice", return_value="FULL NOTICE"),
            patch("builtins.print") as printed,
        ):
            exit_code = product_usage_cli.main(["consent", "--agent-reply", "Yes"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(sdk.replies, ["Yes"])
        self.assertNotIn("FULL NOTICE", [call.args[0] for call in printed.call_args_list])

    def test_noninteractive_consent_leaves_state_unset(self) -> None:
        class FakeSdk:
            consent_status = ConsentStatus.UNSET

        with (
            patch.object(product_usage_cli, "_collection_transport_available", return_value=True),
            patch.object(product_usage_cli, "_usage_sdk", return_value=FakeSdk()),
            patch.object(product_usage_cli.sys.stdin, "isatty", return_value=False),
            patch("builtins.print") as printed,
        ):
            exit_code = product_usage_cli.main(["consent"])
        self.assertEqual(exit_code, 0)
        output = "\n".join(
            " ".join(str(item) for item in call.args) for call in printed.call_args_list
        )
        self.assertIn("Consent remains unset", output)

    def test_unavailable_transport_reports_existing_choice(self) -> None:
        class FakeSdk:
            consent_status = ConsentStatus.DENIED

        with (
            patch.object(product_usage_cli, "_collection_transport_available", return_value=False),
            patch.object(product_usage_cli, "_usage_sdk", return_value=FakeSdk()),
            patch("builtins.print") as printed,
        ):
            exit_code = product_usage_cli.main(["consent"])

        self.assertEqual(exit_code, 0)
        self.assertIn("stored consent is denied", printed.call_args.args[0])

    def test_status_json_reports_client_state(self) -> None:
        class FakeSdk:
            consent_status = ConsentStatus.DENIED

        with (
            patch.object(product_usage_cli, "_usage_sdk", return_value=FakeSdk()),
            patch("builtins.print") as printed,
        ):
            exit_code = product_usage_cli.main(["status", "--json"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            json.loads(printed.call_args.args[0]),
            {"collection_available": True, "status": "denied"},
        )


class LocalCollectorContract(unittest.TestCase):
    def test_injected_sender_and_collector_persist_and_deduplicate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist.product_usage_collector_") as raw:
            root = Path(raw)
            output = root / "events.jsonl"
            process = subprocess.Popen(
                [sys.executable, str(LOCAL_COLLECTOR), "--port", "0", "--output", str(output)],
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                assert process.stdout is not None
                endpoint_line = process.stdout.readline().strip()
                self.assertTrue(endpoint_line.startswith("collector_url="), endpoint_line)
                endpoint = endpoint_line.split("=", 1)[1]
                process.stdout.readline()

                consent = ConsentStore._at_path_for_tests(root / "consent.json")
                consent.write(ConsentDecision.GRANTED)
                identity = EnvironmentIdentityStore._at_path_for_tests(root / "environment.json")
                outbox = Outbox._at_path_for_tests(root / "outbox.sqlite3")

                class TestSender:
                    def send(self, body: bytes) -> set[str]:
                        request = urllib.request.Request(
                            endpoint,
                            data=body,
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        with urllib.request.urlopen(request, timeout=2.0) as response:
                            payload = json.loads(response.read())
                        events = parse_batch_bytes(body).events
                        if payload.get("accepted", 0) + payload.get("duplicates", 0) != len(events):
                            return set()
                        return {str(event.event_id) for event in events}

                observer = ProductUsageObserver.create(
                    run_dir=root / "run",
                    praxist_version="0.2.0",
                    consent_store=consent,
                    identity_store=identity,
                    outbox=outbox,
                    sender=TestSender(),
                    schedule_upload=lambda upload: upload(),
                    _state_path=root / "state.json",
                )
                assert observer is not None
                self.assertTrue(
                    observer.record_run_started(
                        PeerLifecycleSummary.planned(
                            generation_ordinal=0,
                            planned_peer_count=1,
                        )
                    )
                )
                observer.close()
                self.assertTrue(observer._wait_for_idle_for_tests())

                first_record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
                event_id = str(first_record["event"]["event_id"])
                body = json.dumps(
                    {"events": [first_record["event"]]},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                self.assertEqual(TestSender().send(body), {event_id})
            finally:
                process.terminate()
                process.wait(timeout=5)

            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["event"]["event_id"], event_id)
            self.assertIn("received_at", records[0])

    def test_default_development_sender_is_fixed(self) -> None:
        self.assertEqual(DevHttpBatchSender().endpoint, DEV_COLLECTOR_ENDPOINT)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
