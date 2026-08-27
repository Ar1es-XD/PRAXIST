from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class FindingsSyncContractsTest(unittest.TestCase):
    def test_findings_sync_blocking_sync_waits_briefly_or_times_out(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import findings_sync

        with tempfile.TemporaryDirectory() as tmp:
            sync = findings_sync.FindingsSync(Path(tmp) / "findings", poll_interval=0)

            with patch.object(sync, "_sync_once_locked", return_value=7) as locked:
                self.assertEqual(sync.sync_once_blocking(timeout=0.01), 7)
                locked.assert_called_once()

            self.assertTrue(sync._sync_mutex.acquire(blocking=False))
            try:
                with patch.object(sync, "_sync_once_locked", return_value=9) as locked:
                    self.assertEqual(sync.sync_once_blocking(timeout=0.0), 0)
                    locked.assert_not_called()
            finally:
                sync._sync_mutex.release()

    def test_findings_sync_lifecycle_fetch_modes_and_event_loop_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            findings_sync,
            local_store,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            findings_dir = root / "findings"
            sync = findings_sync.FindingsSync(findings_dir, poll_interval=0, local_mode=True)

            sync._run = Mock()
            sync.start()
            sync.start()
            sync.stop()
            self.assertIsNone(sync._thread)
            self.assertTrue(findings_dir.exists())

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root / "store")}, clear=False):
                local_store.init_db()
                local_store.insert_finding(
                    {
                        "id": "db_finding",
                        "finding_type": "result",
                        "title": "DB",
                        "content": "content",
                        "metrics": {},
                        "variant_name": "V",
                        "peer_id": "p",
                        "generation_id": 0,
                        "timestamp": "2026-05-12T00:00:00",
                    }
                )
                self.assertEqual(sync._fetch_from_sqlite()[0]["id"], "db_finding")

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.tools.local_store.init_db",
                side_effect=RuntimeError("db"),
            ):
                self.assertEqual(sync._fetch_from_sqlite(), [])

            self.assertEqual(
                findings_sync.finding_filename({"id": "f", "title": "A/B C"}), "f_A_B_C.json"
            )
            self.assertEqual(findings_sync._sanitize_filename("!!!"), "")
            existing = findings_dir / "exists.json"
            existing.write_text("{}", encoding="utf-8")
            self.assertIsNone(
                findings_sync.save_finding_to_dir(
                    {"id": "exists", "source_filename": existing.name},
                    findings_dir,
                )
            )
            with patch.object(findings_sync, "atomic_write_json", side_effect=RuntimeError("disk")):
                self.assertIsNone(
                    findings_sync.save_finding_to_dir({"id": "bad", "title": "Bad"}, findings_dir)
                )

            http_sync = findings_sync.FindingsSync(findings_dir, poll_interval=0, local_mode=False)
            with patch.object(findings_sync, "get_server_url", side_effect=ValueError("unset")):
                self.assertEqual(http_sync._fetch_from_http(), [])

            fake_response = SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: [{"id": "remote-list"}],
            )
            with (
                patch.object(findings_sync, "HAS_HTTPX", False),
                patch.object(findings_sync, "HAS_REQUESTS", True),
                patch.object(
                    findings_sync,
                    "requests",
                    SimpleNamespace(get=lambda *args, **kwargs: fake_response),
                    create=True,
                ),
                patch.object(findings_sync, "get_server_url", return_value="http://server"),
            ):
                self.assertEqual(http_sync._fetch_from_http(), [{"id": "remote-list"}])

            fake_dict_response = SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"findings": [{"id": "remote-dict"}]},
            )
            with (
                patch.object(findings_sync, "HAS_HTTPX", True),
                patch.object(findings_sync, "HAS_REQUESTS", False),
                patch.object(
                    findings_sync,
                    "httpx",
                    SimpleNamespace(get=lambda *args, **kwargs: fake_dict_response),
                    create=True,
                ),
                patch.object(findings_sync, "get_server_url", return_value="http://server"),
            ):
                self.assertEqual(http_sync._fetch_from_http(), [{"id": "remote-dict"}])

            with (
                patch.object(findings_sync, "HAS_HTTPX", False),
                patch.object(findings_sync, "HAS_REQUESTS", False),
                patch.object(findings_sync, "get_server_url", return_value="http://server"),
            ):
                self.assertEqual(http_sync._fetch_from_http(), [])

            with (
                patch.object(findings_sync, "HAS_HTTPX", True),
                patch.object(
                    findings_sync,
                    "httpx",
                    SimpleNamespace(get=Mock(side_effect=RuntimeError("network"))),
                    create=True,
                ),
                patch.object(findings_sync, "get_server_url", return_value="http://server"),
            ):
                self.assertEqual(http_sync._fetch_from_http(), [])

            local_loop = findings_sync.FindingsSync(findings_dir, poll_interval=0, local_mode=True)
            calls = {"sync": 0}

            def sync_once() -> int:
                calls["sync"] += 1
                if calls["sync"] >= 1:
                    local_loop._stop_event.set()
                return 0

            async def fake_wait(*_args, **_kwargs):
                return SimpleNamespace(reason="filesystem_event")

            local_loop.sync_once = sync_once
            with patch.object(findings_sync, "wait_for_filesystem_event", fake_wait):
                local_loop._run()
            self.assertEqual(calls["sync"], 1)

            local_error = findings_sync.FindingsSync(findings_dir, poll_interval=0, local_mode=True)
            local_error.sync_once = Mock(side_effect=[RuntimeError("sync"), 0])

            async def failing_wait(*_args, **_kwargs):
                local_error._stop_event.set()
                raise RuntimeError("wait")

            with (
                patch.object(findings_sync, "wait_for_filesystem_event", failing_wait),
                patch.object(
                    local_error._stop_event, "wait", side_effect=lambda timeout=None: None
                ),
            ):
                local_error._run()
            self.assertGreaterEqual(local_error.sync_once.call_count, 1)

            server_loop = findings_sync.FindingsSync(
                findings_dir, poll_interval=0, local_mode=False
            )
            server_loop.sync_once = Mock(side_effect=lambda: server_loop._stop_event.set() or 0)
            server_loop._run()
            server_loop.sync_once.assert_called_once()

    def test_findings_sync_materializes_result_artifacts_before_local_ingest(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            findings_sync,
            local_store,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            findings_dir = run_dir / "shared_findings"
            result_dir = run_dir / "results" / "gen1_peer0_late_signal"
            result_dir.mkdir(parents=True)
            (run_dir / "orchestrator_status.json").write_text(
                json.dumps({"current_generation": 2}),
                encoding="utf-8",
            )
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "gen1_peer0_late_signal",
                        "generation_id": 1,
                        "current_aggregate": {"score": 1.25, "scored_complete": True},
                        "n_eval_cells": 4,
                        "scored_complete": True,
                    }
                ),
                encoding="utf-8",
            )
            sync = findings_sync.FindingsSync(
                findings_dir,
                poll_interval=0,
                local_mode=True,
                run_dir=run_dir,
                materialize_result_artifacts=True,
                result_scoring_metric_keys=("score",),
            )

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(Path(tmp) / "db")}):
                touched = sync.sync_once()
                rows = local_store.get_all_findings()

            self.assertGreaterEqual(touched, 1)
            self.assertEqual([row["variant_name"] for row in rows], ["gen1_peer0_late_signal"])
            self.assertEqual(rows[0]["generation_id"], 1)
            self.assertEqual(rows[0]["metrics"]["score"], 1.25)
            self.assertTrue(
                (findings_dir / f"{rows[0]['id']}_gen1_peer0_late_signal.json").exists()
            )

    def test_findings_sync_does_not_restore_auto_result_after_source_disappears(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            findings_sync,
            local_store,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            findings_dir = run_dir / "shared_findings"
            result_dir = run_dir / "results" / "candidate"
            result_dir.mkdir(parents=True)
            summary = result_dir / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "variant_name": "candidate",
                        "generation_id": 0,
                        "current_aggregate": {
                            "score": 1.0,
                            "scored_complete": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            sync = findings_sync.FindingsSync(
                findings_dir,
                poll_interval=0,
                local_mode=True,
                run_dir=run_dir,
                materialize_result_artifacts=True,
                result_scoring_metric_keys=("score",),
            )
            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root / "db")}):
                sync.sync_once()
                [row] = local_store.get_all_findings()
                finding_path = findings_dir / findings_sync.finding_filename(row)
                self.assertTrue(finding_path.exists())

                summary.write_text("{", encoding="utf-8")
                sync.sync_once()
                self.assertEqual(
                    [item["id"] for item in local_store.get_all_findings()], [row["id"]]
                )
                self.assertTrue(finding_path.exists())

                summary.unlink()
                sync.sync_once()
                rows = local_store.get_all_findings()

            self.assertEqual(rows, [])
            self.assertFalse(finding_path.exists())

    def test_findings_sync_generation_hint_and_materializer_error_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import findings_sync

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            findings_dir = run_dir / "shared_findings"
            sync = findings_sync.FindingsSync(findings_dir, run_dir=run_dir)
            self.assertEqual(sync._current_generation_hint(), 0)

            status = run_dir / "orchestrator_status.json"
            status.write_text("[]", encoding="utf-8")
            self.assertEqual(sync._current_generation_hint(), 0)
            status.write_text(
                json.dumps(
                    {
                        "current_generation": "bad",
                        "active_generation": "bad",
                        "generation": "4",
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(sync._current_generation_hint(), 4)
            status.write_text(json.dumps({"current_generation": -2}), encoding="utf-8")
            self.assertEqual(sync._current_generation_hint(), 0)
            status.write_text(json.dumps({"completed_generations": 5}), encoding="utf-8")
            self.assertEqual(sync._current_generation_hint(), 5)
            status.write_text(json.dumps({"completed_generations": "bad"}), encoding="utf-8")
            self.assertEqual(sync._current_generation_hint(), 0)

            sync.materialize_result_artifacts = True
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.findings_collection._materialize_result_artifacts",
                side_effect=RuntimeError("materializer"),
            ):
                self.assertEqual(sync._materialize_result_artifacts_once(), 0)

    def test_findings_sync_applies_active_boundary_cutoff_until_commit(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import findings_sync

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            findings_dir = run_dir / "shared_findings"
            sync = findings_sync.FindingsSync(
                findings_dir,
                run_dir=run_dir,
                materialize_result_artifacts=True,
            )
            (run_dir / "orchestrator_status.json").write_text(
                json.dumps({"current_generation": 3}),
                encoding="utf-8",
            )
            cutoff = datetime.now(UTC)
            source_snapshot = {"results/candidate/summary.json": "target:1:2"}
            sync.begin_boundary_evidence_cutoff(4, cutoff, source_snapshot)

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.findings_collection._materialize_result_artifacts",
                return_value=[],
            ) as materialize:
                sync._materialize_result_artifacts_once()

            self.assertIs(materialize.call_args.kwargs["evidence_cutoff"], cutoff)
            self.assertEqual(
                materialize.call_args.kwargs["evidence_source_snapshot"],
                source_snapshot,
            )
            self.assertEqual(materialize.call_args.kwargs["gen_id"], 4)
            sync.clear_boundary_evidence_cutoff(3)
            self.assertIs(sync._evidence_cutoff_for_generation(4), cutoff)
            sync.clear_boundary_evidence_cutoff(4)
            self.assertIsNone(sync._evidence_cutoff_for_generation(4))

    def test_findings_sync_quarantines_result_arriving_during_pi_boundary_gap(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            findings_sync,
            local_store,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            findings_dir = run_dir / "shared_findings"
            run_dir.mkdir()
            (run_dir / "orchestrator_status.json").write_text(
                json.dumps({"current_generation": 0}),
                encoding="utf-8",
            )
            sync = findings_sync.FindingsSync(
                findings_dir,
                run_dir=run_dir,
                local_mode=True,
                materialize_result_artifacts=True,
                result_scoring_metric_keys=("score",),
            )
            cutoff = datetime.now(UTC)
            sync.begin_boundary_evidence_cutoff(0, cutoff, {})
            result_dir = run_dir / "results" / "gen0_peer0_during_pi"
            result_dir.mkdir(parents=True)
            (result_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "during_pi",
                        "generation_id": 0,
                        "current_aggregate": {"score": 0.9, "scored_complete": True},
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root / "store")}):
                sync.sync_once()
                [finding] = local_store.get_all_findings()

            metrics = finding["metrics"]
            self.assertTrue(metrics["late_after_generation_boundary"])
            self.assertTrue(metrics["generation_boundary_pending_commit"])
            self.assertTrue(metrics["excluded_from_durable_frontier"])
            self.assertFalse(metrics["promotion_eligible"])

    def test_findings_sync_reapplies_cutoff_after_task_finding_update(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            findings_sync,
            local_store,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            findings_dir = run_dir / "shared_findings"
            sync = findings_sync.FindingsSync(
                findings_dir,
                run_dir=run_dir,
                local_mode=True,
                primary_metric="score",
            )
            cutoff = datetime.now(UTC)
            sync.begin_boundary_evidence_cutoff(0, cutoff, {})
            finding_path = findings_dir / "candidate.json"
            finding_path.write_text(
                json.dumps(
                    {
                        "id": "candidate",
                        "generation_id": 0,
                        "peer_id": "gen0_peer0",
                        "metrics": {"score": 1.0},
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root / "store")}):
                sync.sync_once()
                [first] = local_store.get_all_findings()
                finding_path.write_text(
                    json.dumps(
                        {
                            "id": "candidate",
                            "generation_id": 0,
                            "peer_id": "gen0_peer0",
                            "metrics": {"score": 2.0, "promotion_eligible": True},
                        }
                    ),
                    encoding="utf-8",
                )
                sync.sync_once()
                [updated] = local_store.get_all_findings()

            self.assertTrue(first["metrics"]["late_after_generation_boundary"])
            self.assertEqual(updated["metrics"]["score"], 2.0)
            self.assertTrue(updated["metrics"]["late_after_generation_boundary"])
            self.assertTrue(updated["metrics"]["promotion_eligible"])

    def test_boundary_status_update_does_not_replace_newer_metrics(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"LOCAL_STORE_DIR": tmp}):
                local_store.init_db()
                local_store.insert_finding(
                    {
                        "id": "candidate",
                        "generation_id": 0,
                        "metrics": {"score": 2.0, "promotion_eligible": True},
                    }
                )
                updated = local_store.mark_finding_boundary_validation(
                    "candidate",
                    {
                        "score": 1.0,
                        "late_after_generation_boundary": True,
                        "promotion_eligible": False,
                    },
                )
                [finding] = local_store.get_all_findings()

            self.assertTrue(updated)
            self.assertEqual(finding["metrics"]["score"], 2.0)
            self.assertTrue(finding["metrics"]["late_after_generation_boundary"])
            self.assertTrue(finding["metrics"]["promotion_eligible"])

    def test_boundary_status_update_recovers_invalid_legacy_metrics(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"LOCAL_STORE_DIR": tmp}):
                local_store.init_db()
                self.assertFalse(
                    local_store.mark_finding_boundary_validation(
                        "", {"late_after_generation_boundary": True}
                    )
                )
                self.assertFalse(
                    local_store.mark_finding_boundary_validation("missing", {"score": 1.0})
                )
                self.assertFalse(
                    local_store.mark_finding_boundary_validation(
                        "missing", {"late_after_generation_boundary": True}
                    )
                )
                local_store.insert_finding(
                    {"id": "candidate", "generation_id": 0, "metrics": {"score": 1.0}}
                )
                with local_store._get_conn() as conn:
                    conn.execute(
                        "UPDATE findings SET metrics = ? WHERE id = ?",
                        ("not-json", "candidate"),
                    )
                self.assertTrue(
                    local_store.mark_finding_boundary_validation(
                        "candidate", {"late_after_generation_boundary": True}
                    )
                )
                with local_store._get_conn() as conn:
                    conn.execute(
                        "UPDATE findings SET metrics = ? WHERE id = ?", ("[]", "candidate")
                    )
                self.assertTrue(
                    local_store.mark_finding_boundary_validation(
                        "candidate", {"artifact_signal_status": "late_signal"}
                    )
                )
                [finding] = local_store.get_all_findings()

            self.assertEqual(finding["metrics"], {"artifact_signal_status": "late_signal"})

    def test_abandoned_boundary_clear_tolerates_legacy_row_shapes(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        legacy_metrics = {
            "score": 2.0,
            "generation_boundary_pending_commit": True,
            "late_after_generation_boundary": True,
            "late_observed_generation_id": 0,
            "validation_only_result": True,
            "promotion_eligible": False,
            "clean_promotion_eligible": False,
            "excluded_from_durable_frontier": True,
            "exclusion_reason": "late_after_generation_boundary",
            "recommended_next_step": "revalidate",
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"LOCAL_STORE_DIR": tmp}):
                local_store.init_db()
                for finding_id in ("malformed", "scalar", "other-generation", "bad-generation"):
                    local_store.insert_finding(
                        {"id": finding_id, "generation_id": 0, "metrics": {"score": 1.0}}
                    )
                local_store.insert_finding(
                    {"id": "legacy", "generation_id": 0, "metrics": legacy_metrics}
                )
                raw_metrics = {
                    "malformed": "not-json",
                    "scalar": "[]",
                    "other-generation": json.dumps(
                        {
                            "generation_boundary_pending_commit": True,
                            "late_observed_generation_id": 1,
                        }
                    ),
                    "bad-generation": json.dumps(
                        {
                            "generation_boundary_pending_commit": True,
                            "late_observed_generation_id": "invalid",
                        }
                    ),
                }
                with local_store._get_conn() as conn:
                    for finding_id, metrics in raw_metrics.items():
                        conn.execute(
                            "UPDATE findings SET metrics = ? WHERE id = ?",
                            (metrics, finding_id),
                        )

                self.assertEqual(local_store.clear_pending_boundary_validation(0), 1)
                with local_store._get_conn(readonly=True) as conn:
                    stored_metrics = {
                        row["id"]: row["metrics"]
                        for row in conn.execute(
                            "SELECT id, metrics FROM findings WHERE id IN (?, ?, ?, ?)",
                            ("legacy", "malformed", "other-generation", "bad-generation"),
                        ).fetchall()
                    }

            self.assertEqual(json.loads(stored_metrics["legacy"]), {"score": 2.0})
            self.assertEqual(stored_metrics["malformed"], "not-json")
            self.assertEqual(
                json.loads(stored_metrics["other-generation"])["late_observed_generation_id"], 1
            )
            self.assertEqual(
                json.loads(stored_metrics["bad-generation"])["late_observed_generation_id"],
                "invalid",
            )

    def test_abandoned_boundary_clear_restores_canonical_routing_state(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"LOCAL_STORE_DIR": tmp}):
                local_store.init_db()
                local_store.insert_finding(
                    {
                        "id": "candidate",
                        "generation_id": 0,
                        "metrics": {"score": 2.0, "promotion_eligible": True},
                    }
                )
                local_store.mark_finding_boundary_validation(
                    "candidate",
                    {
                        "generation_boundary_pending_commit": True,
                        "generation_boundary_evidence_cutoff_at": ("2027-01-01T00:00:00+00:00"),
                        "late_after_generation_boundary": True,
                        "late_observed_generation_id": 0,
                        "promotion_eligible": False,
                    },
                )
                cleared = local_store.clear_pending_boundary_validation(0)
                [finding] = local_store.get_all_findings()

            self.assertEqual(cleared, 1)
            self.assertEqual(finding["metrics"], {"score": 2.0, "promotion_eligible": True})

    def test_canonical_finding_cutoff_snapshot_returns_existing_rows(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"LOCAL_STORE_DIR": tmp}):
                local_store.init_db()
                local_store.insert_finding(
                    {"id": "before-cutoff", "generation_id": 2, "metrics": {"score": 1.0}}
                )
                cutoff, findings = local_store.snapshot_findings_at_cutoff(2)

            self.assertIsNotNone(cutoff.tzinfo)
            self.assertEqual([finding["id"] for finding in findings], ["before-cutoff"])

    def test_boundary_sync_keeps_running_when_validation_annotation_fails(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import findings_collection
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            findings_ingest,
            findings_sync,
        )

        self.assertEqual(
            findings_sync._finding_generation_id({"metrics": {"source_generation_id": "4"}}),
            4,
        )
        self.assertIsNone(findings_sync._finding_generation_id({"generation_id": "bad"}))

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            findings_dir = run_dir / "shared_findings"
            (run_dir / "gen_0" / "shared_findings").mkdir(parents=True)
            sync = findings_sync.FindingsSync(
                findings_dir,
                run_dir=run_dir,
                local_mode=True,
            )
            sync.begin_boundary_evidence_cutoff(0, datetime.now(UTC), {})
            finding = {"id": "candidate", "generation_id": 0, "metrics": {}}

            with (
                patch.object(sync, "_materialize_result_artifacts_once", return_value=0),
                patch.object(
                    findings_ingest,
                    "ingest_findings_directory",
                    return_value=1,
                ) as ingest,
                patch.object(sync, "_fetch_all_findings", return_value=[finding]),
                patch.object(
                    findings_collection,
                    "annotate_late_boundary_findings",
                    side_effect=OSError("validation unavailable"),
                ),
                patch.object(findings_sync, "save_finding_to_dir", return_value=None),
            ):
                touched = sync._sync_once_locked()

        self.assertEqual(touched, 2)
        self.assertEqual(ingest.call_count, 2)

    def test_findings_sync_event_loop_watches_results_when_materializing(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import findings_sync

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            findings_dir = run_dir / "shared_findings"
            results_dir = run_dir / "results"
            results_dir.mkdir(parents=True)
            loop = findings_sync.FindingsSync(
                findings_dir,
                run_dir=run_dir,
                poll_interval=0,
                local_mode=True,
                materialize_result_artifacts=True,
            )
            seen: dict[str, object] = {}

            async def fake_wait(paths, **kwargs):
                seen["paths"] = list(paths)
                seen["recursive"] = kwargs.get("recursive")
                loop._stop_event.set()
                return SimpleNamespace(reason="filesystem_event")

            loop.sync_once = Mock(return_value=0)
            with patch.object(findings_sync, "wait_for_filesystem_event", fake_wait):
                loop._run()

            self.assertIn(findings_dir, seen["paths"])
            self.assertIn(results_dir, seen["paths"])
            self.assertTrue(seen["recursive"])


if __name__ == "__main__":
    unittest.main()
