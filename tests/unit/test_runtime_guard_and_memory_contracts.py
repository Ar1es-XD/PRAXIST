from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class RuntimeGuardAndMemoryContractsTest(unittest.TestCase):
    def test_sidecar_recovery_scope_only_targets_rerun_generation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import sidecars

        with tempfile.TemporaryDirectory() as tmp:
            captured: list[int | None] = []

            class FakeScheduler:
                def __init__(self, **kwargs):
                    captured.append(kwargs.get("recovery_rerun_generation"))

                def start(self):
                    pass

            def make_loop(name: str):
                return SimpleNamespace(
                    local_mode=False,
                    run_dir=Path(tmp) / name,
                    task_spec=SimpleNamespace(
                        compute_budget=SimpleNamespace(
                            resource_scheduler={
                                "mode": "central",
                                "profiles": {"cpu": {"accelerator": "cpu"}},
                                "default_profile": "cpu",
                            },
                            max_parallel_runs_per_peer=1,
                        )
                    ),
                    _findings_sync=None,
                    _graph_maintainer=None,
                    _status_writer=None,
                    _build_status_snapshot=lambda: {},
                )

            status = SimpleNamespace(start=Mock())
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler.ExperimentSchedulerService",
                    FakeScheduler,
                ),
                patch.object(sidecars, "OrchestratorStatusWriter", return_value=status),
            ):
                sidecars.start_sidecars(
                    make_loop("rerun"),
                    resume_plan=SimpleNamespace(
                        start_generation=3,
                        has_pending_boundary=False,
                    ),
                )
                sidecars.start_sidecars(
                    make_loop("pending-boundary"),
                    resume_plan=SimpleNamespace(
                        start_generation=3,
                        has_pending_boundary=True,
                    ),
                )

            self.assertEqual(captured, [3, None])

    def test_sidecar_initial_sync_inherits_resume_boundary_cutoff(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import sidecars

        with tempfile.TemporaryDirectory() as tmp:
            cutoff = object()
            source_snapshot = {"results/candidate/summary.json": "target:1:2"}
            calls: list[object] = []

            class FakeFindingsSync:
                def __init__(self, **_kwargs):
                    pass

                def begin_boundary_evidence_cutoff(self, *boundary):
                    calls.append(("begin", boundary))

                def sync_once(self):
                    calls.append("sync")

                def start(self):
                    calls.append("start")

            loop = SimpleNamespace(
                local_mode=True,
                findings_dir=Path(tmp) / "findings",
                run_dir=Path(tmp),
                _boundary_evidence_cutoff=(2, cutoff, source_snapshot),
                _findings_sync=None,
                _graph_maintainer=None,
                _status_writer=None,
                _build_status_snapshot=lambda: {},
            )
            status = SimpleNamespace(start=Mock())
            graph = SimpleNamespace(sync_once=Mock(), start=Mock())
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.tools.findings_sync.FindingsSync",
                    FakeFindingsSync,
                ),
                patch(
                    "praxist.plugins.graph_maintainers.finding_graph_mvp.engine.FindingGraphMaintainer",
                    return_value=graph,
                ),
                patch.object(sidecars, "OrchestratorStatusWriter", return_value=status),
            ):
                sidecars.start_sidecars(loop)

            self.assertEqual(
                calls,
                [("begin", (2, cutoff, source_snapshot)), "sync", "start"],
            )

    def test_sidecar_start_stop_is_best_effort_and_status_writer_always_starts(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import sidecars

        with tempfile.TemporaryDirectory() as tmp:
            loop = SimpleNamespace(
                local_mode=True,
                findings_dir=Path(tmp) / "findings",
                run_dir=Path(tmp),
                _findings_sync=None,
                _graph_maintainer=None,
                _status_writer=None,
                _build_status_snapshot=lambda: {"ok": True},
            )

            class FakeFindingsSync:
                def __init__(self, **_kwargs):
                    self.calls: list[str] = []

                def sync_once(self):
                    self.calls.append("sync")
                    raise RuntimeError("initial sync failed")

                def start(self):
                    self.calls.append("start")

                def stop(self):
                    self.calls.append("stop")
                    raise RuntimeError("stop failed")

            class FakeGraphMaintainer:
                def __init__(self, **_kwargs):
                    self.calls: list[str] = []

                def sync_once(self):
                    self.calls.append("sync")
                    raise RuntimeError("initial graph failed")

                def start(self):
                    self.calls.append("start")

                def stop(self):
                    self.calls.append("stop")
                    raise RuntimeError("stop failed")

            status = SimpleNamespace(
                start=Mock(), stop=Mock(side_effect=RuntimeError("status stop"))
            )
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.tools.findings_sync.FindingsSync",
                    FakeFindingsSync,
                ),
                patch(
                    "praxist.plugins.graph_maintainers.finding_graph_mvp.engine.FindingGraphMaintainer",
                    FakeGraphMaintainer,
                ),
                patch.object(sidecars, "OrchestratorStatusWriter", return_value=status),
            ):
                sidecars.start_sidecars(loop)
                self.assertEqual(loop._findings_sync.calls, ["sync", "start"])
                self.assertEqual(loop._graph_maintainer.calls, ["sync", "start"])
                status.start.assert_called_once()
                sidecars.stop_sidecars(loop, exit_condition="done")
                status.stop.assert_called_once_with(exit_condition="done")

            nonlocal_loop = SimpleNamespace(
                local_mode=False,
                run_dir=Path(tmp),
                _findings_sync=None,
                _graph_maintainer=None,
                _status_writer=None,
                _build_status_snapshot=lambda: {},
            )
            status2 = SimpleNamespace(start=Mock(), stop=Mock())
            with patch.object(sidecars, "OrchestratorStatusWriter", return_value=status2):
                sidecars.start_sidecars(nonlocal_loop)
            self.assertIsNone(nonlocal_loop._findings_sync)
            self.assertIsNone(nonlocal_loop._graph_maintainer)
            status2.start.assert_called_once()

            failing_loop = SimpleNamespace(
                local_mode=True,
                findings_dir=Path(tmp) / "findings",
                run_dir=Path(tmp),
                _findings_sync=None,
                _graph_maintainer=None,
                _status_writer=None,
                _build_status_snapshot=lambda: {},
            )
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.tools.findings_sync.FindingsSync",
                    side_effect=RuntimeError("sync init"),
                ),
                patch(
                    "praxist.plugins.graph_maintainers.finding_graph_mvp.engine.FindingGraphMaintainer",
                    side_effect=RuntimeError("graph init"),
                ),
                patch.object(sidecars, "OrchestratorStatusWriter", return_value=status2),
            ):
                sidecars.start_sidecars(failing_loop)
            self.assertIsNone(failing_loop._findings_sync)
            self.assertIsNone(failing_loop._graph_maintainer)

            invalid_central = SimpleNamespace(
                local_mode=False,
                run_dir=Path(tmp),
                task_spec=SimpleNamespace(
                    compute_budget=SimpleNamespace(
                        resource_scheduler={
                            "mode": "central",
                            "profiles": {"cpu": {"accelerator": "cpu"}},
                            "default_profile": "missing",
                        }
                    )
                ),
                _findings_sync=None,
                _graph_maintainer=None,
                _status_writer=None,
                _build_status_snapshot=lambda: {},
            )
            with self.assertRaisesRegex(RuntimeError, "could not start"):
                sidecars.start_sidecars(invalid_central)
            invalid_central.task_spec.compute_budget.resource_scheduler = {"mode": "  "}
            with self.assertRaisesRegex(RuntimeError, "could not start"):
                sidecars.start_sidecars(invalid_central)
            invalid_central.task_spec.compute_budget.resource_scheduler = {"mode": "centrla"}
            with self.assertRaisesRegex(RuntimeError, "could not start"):
                sidecars.start_sidecars(invalid_central)

            valid_central = SimpleNamespace(
                local_mode=False,
                run_dir=Path(tmp) / "central-platform",
                task_spec=SimpleNamespace(
                    compute_budget=SimpleNamespace(
                        resource_scheduler={
                            "mode": "central",
                            "profiles": {"cpu": {"accelerator": "cpu"}},
                            "default_profile": "cpu",
                        },
                        max_parallel_runs_per_peer=1,
                    )
                ),
                _findings_sync=None,
                _graph_maintainer=None,
                _status_writer=None,
                _build_status_snapshot=lambda: {},
            )
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler._posix_file_locking",
                    side_effect=RuntimeError("requires POSIX file locking"),
                ),
                self.assertRaisesRegex(RuntimeError, "requires POSIX file locking"),
            ):
                sidecars.start_sidecars(valid_central)
            self.assertIsNotNone(valid_central._experiment_scheduler)

    def test_sidecar_stop_retries_transient_scheduler_cleanup_failure(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import sidecars

        scheduler = Mock()
        scheduler.stop.side_effect = [RuntimeError("transient cleanup failure"), None]
        loop = SimpleNamespace(
            _experiment_scheduler=scheduler,
            _findings_sync=None,
            _graph_maintainer=None,
            _status_writer=None,
        )

        sidecars.stop_sidecars(loop, exit_condition="error")

        self.assertEqual(scheduler.stop.call_count, 2)

    def test_protected_pid_manifest_and_cli_contracts(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {protected_pids.ENV_PROTECTED_DIR: ""}, clear=False),
        ):
            run_dir = Path(tmp)
            protected_dir = run_dir / "protected_pids"
            self.assertEqual(protected_pids._protected_dir(run_dir), protected_dir)
            with patch.dict(
                os.environ, {protected_pids.ENV_PROTECTED_DIR: str(run_dir / "override")}
            ):
                self.assertEqual(protected_pids._protected_dir(None), run_dir / "override")
            with self.assertRaises(ValueError):
                protected_pids._protected_dir(None)

            with patch.object(protected_pids, "_is_pid_alive", return_value=True):
                entry = protected_pids.register_pid(
                    123,
                    peer_id="gen/0 peer",
                    tag="long",
                    eta_seconds=60,
                    run_dir=run_dir,
                )
                self.assertEqual(entry.pid, 123)
                updated = protected_pids.register_pid(
                    123,
                    peer_id="gen/0 peer",
                    tag="longer",
                    eta_seconds=0,
                    run_dir=run_dir,
                )
                self.assertEqual(updated.tag, "longer")
                self.assertEqual(updated.eta_seconds, 60)
                self.assertIn(123, protected_pids.get_protected_pids_set(run_dir))

            with patch.object(protected_pids, "_is_pid_alive", side_effect=lambda pid: pid == 123):
                self.assertEqual([e.pid for e in protected_pids.list_all_protected(run_dir)], [123])
            self.assertFalse(
                protected_pids.unregister_pid(999, peer_id="gen/0 peer", run_dir=run_dir)
            )
            self.assertTrue(
                protected_pids.unregister_pid(123, peer_id="gen/0 peer", run_dir=run_dir)
            )

            manifest = protected_dir / "manual.json"
            manifest.write_text(
                json.dumps(
                    [
                        {"pid": 456, "tag": "live", "extra": "ignored"},
                        {"pid": -1, "tag": "dead"},
                    ]
                ),
                encoding="utf-8",
            )
            with patch.object(protected_pids, "_is_pid_alive", side_effect=lambda pid: pid == 456):
                listed = protected_pids.list_all_protected(run_dir, prune_dead=True)
            self.assertEqual([entry.pid for entry in listed], [456])
            self.assertNotIn("-1", manifest.read_text(encoding="utf-8"))

            with (
                patch.object(protected_pids, "_is_pid_alive", return_value=True),
                patch(
                    "sys.argv",
                    ["protected_pids", "list", "--run-dir", str(run_dir), "--format", "pids"],
                ),
                self.assertRaises(SystemExit) as cm,
                redirect_stdout(io.StringIO()) as stdout,
            ):
                protected_pids._cli()
            self.assertEqual(cm.exception.code, 0)
            self.assertIn("456", stdout.getvalue())

            with (
                patch.object(protected_pids, "unregister_pid", return_value=False),
                patch("sys.argv", ["protected_pids", "unregister", "--pid", "456", "--peer", "p"]),
                self.assertRaises(SystemExit) as cm,
            ):
                protected_pids._cli()
            self.assertEqual(cm.exception.code, 1)

        with patch.object(protected_pids.os, "kill", side_effect=PermissionError):
            self.assertTrue(protected_pids._is_pid_alive(999999))
        with patch.object(protected_pids.os, "kill", side_effect=ProcessLookupError):
            self.assertFalse(protected_pids._is_pid_alive(999999))
        self.assertFalse(protected_pids._is_pid_alive(0))

    def test_protected_pid_close_signal_lookup_is_fail_open_for_telemetry(self) -> None:
        """Read-side signal lookup must never make process telemetry unavailable."""

        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with patch.dict(os.environ, {protected_pids.ENV_PROTECTED_DIR: ""}, clear=False):
                self.assertIsNone(protected_pids._run_dir_from_protected_env())
            self.assertIsNone(protected_pids._generation_closing_signal(run_dir, "peer_unknown"))
            with patch.object(Path, "exists", side_effect=OSError("transient filesystem")):
                self.assertIsNone(protected_pids._generation_closing_signal(run_dir, "gen4_peer2"))

    def test_protected_pid_telemetry_tolerates_missing_manifest_and_permissions(self) -> None:
        """Status collection remains available without a manifest or killpg permission."""

        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        with patch.dict(os.environ, {protected_pids.ENV_PROTECTED_DIR: ""}, clear=False):
            self.assertEqual(protected_pids.list_all_protected(), [])
        with patch.object(protected_pids.os, "killpg", side_effect=PermissionError):
            self.assertTrue(protected_pids._is_process_group_alive(12345))

    def test_context_firewall_budgeting_and_auditor_semantics(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory import (
            context_auditor,
            context_firewall,
        )

        large = "x" * 3000
        nested = {
            "keep": "small",
            "negative_evidence_digest": large,
            "role_performance": list(range(60)),
            "findings_summary": large,
            "coverage_matrix_digest": "must remain",
        }
        shrunk = context_firewall.shrink_dict(nested, budget_tokens=20)
        self.assertEqual(shrunk["coverage_matrix_digest"], "must remain")
        self.assertIn("budget truncated", shrunk["negative_evidence_digest"])
        small = {"a": 1}
        self.assertIs(context_firewall.shrink_dict(small, budget_tokens=1000), small)

        pack = SimpleNamespace(
            pack_id="pack",
            built_at="now",
            panel_mode="full",
            target_decisions=["d"],
            shared_core=nested,
            private_packs={
                "builder": [{"interpretation": {"short": large}, "n": i} for i in range(4)]
            },
            audit={"raw_history": "not allowed"},
        )
        fitted = context_firewall.fit_pack_to_budget(pack, "unknown-mode")
        self.assertEqual(fitted["pack_id"], "pack")
        self.assertLessEqual(len(fitted["private_packs"]["builder"]), 4)
        fitted_mini = context_firewall.fit_pack_to_budget(pack, "mini")
        self.assertLessEqual(len(fitted_mini["private_packs"]["builder"]), 2)
        self.assertTrue(context_firewall.forbid_raw_history(fitted))

        self.assertFalse(context_auditor._has_source_id({"source_findings": ["", None]}))
        self.assertTrue(context_auditor._has_source_id({"source_evidence": [{"id": "f1"}]}))
        self.assertEqual(context_auditor._has_overclaim_language(123), [])
        self.assertIn(
            "generally dominant",
            context_auditor._has_overclaim_language("This is generally dominant"),
        )
        self.assertFalse(context_auditor._has_source_id("bad"))
        self.assertTrue(context_auditor._has_source_id({"supports": "f1"}))
        self.assertEqual(context_auditor._negative_digest_entries("bad"), [])
        digest_entries = context_auditor._negative_digest_entries(
            {
                "shared_core": {
                    "negative_evidence_digest": {
                        "items": [
                            {"id": "neg1", "title": "specific drawdown regression"},
                            {"id": "neg1", "title": "duplicate"},
                        ]
                    }
                },
                "negative_evidence_digest": {
                    "summary": "unique underinvestment failure",
                },
            }
        )
        self.assertEqual(len(digest_entries), 2)
        self.assertIn("id:neg1", context_auditor._negative_digest_identity(digest_entries[0]))
        odd_key = object()
        self.assertTrue(
            context_auditor._negative_digest_identity({odd_key: "value"}).startswith("dict:")
        )
        circular: list[object] = []
        circular.append(circular)
        self.assertFalse(context_auditor._mentions_negative_digest(circular, ["cycle token"]))
        tokens = context_auditor._negative_digest_reference_tokens(
            [
                {
                    "finding_id": "neg-2",
                    "claim_id": "claim_x",
                    "variant_name": "variant_y",
                    "title": "rare drawdown regression",
                    "summary": "specific active alpha failure",
                },
                "underinvestment regression",
            ]
        )
        self.assertIn("neg 2", tokens)
        self.assertIn("rare drawdown", tokens)
        self.assertFalse(context_auditor._mentions_negative_digest({"text": "none"}, []))

        agenda = {
            "consensus_actions": [
                {"id": "c1", "claim_or_hypothesis": "generally dominant"},
                "ignored",
            ],
            "cross_peer_hypotheses": [{"id": "h1", "claim": "supported", "source_id": "f1"}],
            "retired_claims": [{"claim_id": "old", "boundary": "", "revive_if": ""}],
            "peer_contracts": {
                "peer0": {"role": "Bridge", "instructions": "use query_coverage_matrix"},
                "peer1": {"role": "bridge", "instructions": "no coverage"},
            },
        }
        weak_pack = {"private_packs": {"skeptic": [{"quality": {"is_negative": False}}]}}
        memos = {"builder": {}, "skeptic": "ignored"}
        first = context_auditor.audit_agenda(
            agenda,
            weak_pack,
            memos,
            audit_id="first",
            completed_gen_id=0,
        )
        self.assertFalse(first.pass_)
        self.assertTrue(any("lacks source_id" in warning for warning in first.warnings))
        self.assertTrue(
            any("tolerated for first synthesis" in warning for warning in first.warnings)
        )
        self.assertEqual(first.metrics["bridge_contracts"], 2)
        self.assertEqual(first.metrics["bridge_with_coverage_check"], 1)
        self.assertGreaterEqual(first.metrics["citation_coverage"], 0.5)

        later = context_auditor.audit_agenda(
            {**agenda, "retired_claims": [], "peer_contracts": {"a": {}, "b": {}}},
            weak_pack,
            {"builder": {"private_knowledge_used": []}},
            audit_id="later",
            completed_gen_id=2,
            expected_peer_contract_count=3,
        )
        self.assertFalse(later.pass_)
        self.assertTrue(any("below 15%" in issue for issue in later.blocking_issues))
        self.assertTrue(any("peer_contracts count" in issue for issue in later.blocking_issues))

        ledger_digest_report = context_auditor.audit_agenda(
            {
                "consensus_actions": [
                    {
                        "id": "c",
                        "claim_or_hypothesis": "bounded by failed scout family",
                        "source_id": "neg1",
                    }
                ],
                "peer_contracts": {"a": {}, "b": {}},
            },
            {
                "shared_core": {
                    "negative_evidence_digest": [
                        {
                            "id": "neg1",
                            "title": "failed scout family",
                            "category": "failed_lineage",
                            "summary": "scout evidence contradicted the mechanism",
                        }
                    ]
                },
                "private_packs": {
                    "skeptic": [{"quality": {"is_negative": False}}],
                    "builder": [{"quality": {"is_negative": False}}],
                },
            },
            {"builder": {"private_knowledge_used": []}},
            audit_id="ledger_digest",
            completed_gen_id=2,
            expected_peer_contract_count=2,
        )
        self.assertTrue(ledger_digest_report.pass_)
        self.assertEqual(ledger_digest_report.metrics["negative_evidence_cards_in_pack"], 0)
        self.assertEqual(ledger_digest_report.metrics["negative_evidence_digest_entries"], 1)
        self.assertEqual(ledger_digest_report.metrics["negative_evidence_in_pack"], 1)
        self.assertTrue(ledger_digest_report.metrics["negative_evidence_digest_referenced"])

        fallback_digest_report = context_auditor.audit_agenda(
            {
                "consensus_actions": [
                    {
                        "id": "c",
                        "claim_or_hypothesis": "bounded by negative evidence",
                        "source_id": "neg2",
                    }
                ],
                "peer_contracts": {"a": {}, "b": {}},
            },
            {
                "shared_core": {"negative_evidence_digest": []},
                "negative_evidence_digest": [{"id": "neg2"}],
                "private_packs": {
                    "skeptic": [{"quality": {"is_negative": False}}],
                    "builder": [{"quality": {"is_negative": False}}],
                },
            },
            {"builder": {"private_knowledge_used": []}},
            audit_id="fallback_digest",
            completed_gen_id=2,
            expected_peer_contract_count=2,
        )
        self.assertTrue(fallback_digest_report.pass_)
        self.assertEqual(fallback_digest_report.metrics["negative_evidence_digest_entries"], 1)

        digest_level_reference_report = context_auditor.audit_agenda(
            {
                "consensus_actions": [
                    {
                        "id": "c",
                        "claim_or_hypothesis": "bounded by mechanism failed on scout replay",
                        "source_id": "f",
                    }
                ],
                "peer_contracts": {"a": {}, "b": {}},
            },
            {
                "shared_core": {
                    "negative_evidence_digest": [
                        {
                            "id": "neg_digest_level",
                            "title": "mechanism failed on scout replay",
                            "summary": "digest entry was available",
                        }
                    ]
                },
                "private_packs": {
                    "skeptic": [{"quality": {"is_negative": False}}],
                    "builder": [{"quality": {"is_negative": False}}],
                },
            },
            {"builder": {"private_knowledge_used": []}},
            audit_id="digest_level_reference",
            completed_gen_id=2,
            expected_peer_contract_count=2,
        )
        self.assertTrue(digest_level_reference_report.pass_)
        self.assertTrue(
            digest_level_reference_report.metrics["negative_evidence_digest_referenced"]
        )

        digest_category_only_report = context_auditor.audit_agenda(
            {
                "consensus_actions": [
                    {
                        "id": "c",
                        "claim_or_hypothesis": "failed lineage evidence bounds the next repair",
                        "source_id": "f",
                    }
                ],
                "peer_contracts": {"a": {}, "b": {}},
            },
            {
                "shared_core": {
                    "negative_evidence_digest": [
                        {
                            "id": "neg_category",
                            "title": "specific scout failure",
                            "category": "failed_lineage",
                            "summary": "digest entry was available",
                        }
                    ]
                },
                "private_packs": {
                    "skeptic": [{"quality": {"is_negative": False}}],
                    "builder": [{"quality": {"is_negative": False}}],
                },
            },
            {"builder": {"private_knowledge_used": []}},
            audit_id="digest_category_only",
            completed_gen_id=2,
            expected_peer_contract_count=2,
        )
        self.assertFalse(digest_category_only_report.pass_)
        self.assertFalse(digest_category_only_report.metrics["negative_evidence_digest_referenced"])
        self.assertTrue(
            any(
                "negative evidence digest is present" in issue
                for issue in digest_category_only_report.blocking_issues
            )
        )

        digest_source_reference_report = context_auditor.audit_agenda(
            {
                "consensus_actions": [
                    {
                        "id": "c",
                        "claim_or_hypothesis": (
                            "source artifact::results/neg_child/tiered_eval_summary.json "
                            "bounds the repair"
                        ),
                        "source_id": "f",
                    }
                ],
                "peer_contracts": {"a": {}, "b": {}},
            },
            {
                "shared_core": {
                    "negative_evidence_digest": [
                        {
                            "id": "neg_source",
                            "source": "artifact::results/neg_child/tiered_eval_summary.json",
                            "title": "specific source failure",
                            "summary": "digest entry was available",
                        }
                    ]
                },
                "private_packs": {
                    "skeptic": [{"quality": {"is_negative": False}}],
                    "builder": [{"quality": {"is_negative": False}}],
                },
            },
            {"builder": {"private_knowledge_used": []}},
            audit_id="digest_source_reference",
            completed_gen_id=2,
            expected_peer_contract_count=2,
        )
        self.assertTrue(digest_source_reference_report.pass_)
        self.assertTrue(
            digest_source_reference_report.metrics["negative_evidence_digest_referenced"]
        )

        digest_summary_reference_report = context_auditor.audit_agenda(
            {
                "consensus_actions": [
                    {
                        "id": "c",
                        "claim_or_hypothesis": (
                            "the scout evidence contradicted this parent mechanism"
                        ),
                        "source_id": "f",
                    }
                ],
                "peer_contracts": {"a": {}, "b": {}},
            },
            {
                "shared_core": {
                    "negative_evidence_digest": [
                        {
                            "id": "neg_summary",
                            "title": "specific scout boundary",
                            "summary": "scout evidence contradicted the mechanism",
                        }
                    ]
                },
                "private_packs": {
                    "skeptic": [{"quality": {"is_negative": False}}],
                    "builder": [{"quality": {"is_negative": False}}],
                },
            },
            {"builder": {"private_knowledge_used": []}},
            audit_id="digest_summary_reference",
            completed_gen_id=2,
            expected_peer_contract_count=2,
        )
        self.assertTrue(digest_summary_reference_report.pass_)
        self.assertTrue(
            digest_summary_reference_report.metrics["negative_evidence_digest_referenced"]
        )

        mirrored_digest_report = context_auditor.audit_agenda(
            {
                "consensus_actions": [
                    {
                        "id": "c",
                        "claim_or_hypothesis": "same failed scout limits the claim",
                        "source_id": "neg3",
                    }
                ],
                "peer_contracts": {"a": {}, "b": {}},
            },
            {
                "shared_core": {
                    "negative_evidence_digest": [
                        {
                            "id": "neg3",
                            "title": "same failed scout",
                            "summary": "same contradiction",
                        }
                    ]
                },
                "negative_evidence_digest": [
                    {
                        "id": "neg3",
                        "title": "same failed scout",
                        "summary": "same contradiction",
                    }
                ],
                "private_packs": {
                    "skeptic": [{"quality": {"is_negative": False}}],
                    "builder": [{"quality": {"is_negative": False}}],
                },
            },
            {"builder": {"private_knowledge_used": []}},
            audit_id="mirrored_digest",
            completed_gen_id=2,
            expected_peer_contract_count=2,
        )
        self.assertTrue(mirrored_digest_report.pass_)
        self.assertEqual(mirrored_digest_report.metrics["negative_evidence_digest_entries"], 1)
        self.assertEqual(mirrored_digest_report.metrics["negative_evidence_in_pack"], 1)

        generic_negative_phrase_report = context_auditor.audit_agenda(
            {
                "consensus_actions": [
                    {
                        "id": "c",
                        "claim_or_hypothesis": "bounded by the negative evidence digest",
                        "source_id": "f",
                    }
                ],
                "peer_contracts": {"a": {}, "b": {}},
            },
            {
                "shared_core": {
                    "negative_evidence_digest": [
                        {
                            "id": "neg_generic_unseen",
                            "title": "failed lineage not cited",
                            "summary": "contradiction was available",
                        }
                    ]
                },
                "private_packs": {
                    "skeptic": [{"quality": {"is_negative": False}}],
                    "builder": [{"quality": {"is_negative": False}}],
                },
            },
            {"builder": {"private_knowledge_used": []}},
            audit_id="generic_negative_phrase",
            completed_gen_id=2,
            expected_peer_contract_count=2,
        )
        self.assertFalse(generic_negative_phrase_report.pass_)
        self.assertFalse(
            generic_negative_phrase_report.metrics["negative_evidence_digest_referenced"]
        )

        generic_category_report = context_auditor.audit_agenda(
            {
                "consensus_actions": [
                    {
                        "id": "c",
                        "claim_or_hypothesis": "bounded by the negative evidence digest",
                        "source_id": "f",
                    }
                ],
                "peer_contracts": {"a": {}, "b": {}},
            },
            {
                "shared_core": {
                    "negative_evidence_digest": [
                        {
                            "id": "neg_generic_category_unseen",
                            "category": "negative",
                            "title": "failed lineage not cited",
                            "summary": "contradiction was available",
                        }
                    ]
                },
                "private_packs": {
                    "skeptic": [{"quality": {"is_negative": False}}],
                    "builder": [{"quality": {"is_negative": False}}],
                },
            },
            {"builder": {"private_knowledge_used": []}},
            audit_id="generic_negative_category",
            completed_gen_id=2,
            expected_peer_contract_count=2,
        )
        self.assertFalse(generic_category_report.pass_)
        self.assertFalse(generic_category_report.metrics["negative_evidence_digest_referenced"])

        unreferenced_digest_report = context_auditor.audit_agenda(
            {
                "consensus_actions": [
                    {"id": "c", "claim_or_hypothesis": "supportive only", "source_id": "f"}
                ],
                "peer_contracts": {"a": {}, "b": {}},
            },
            {
                "shared_core": {
                    "negative_evidence_digest": [
                        {
                            "id": "neg_unseen",
                            "title": "failed lineage unseen",
                            "summary": "contradiction was available",
                        }
                    ]
                },
                "private_packs": {
                    "skeptic": [{"quality": {"is_negative": False}}],
                    "builder": [{"quality": {"is_negative": False}}],
                },
            },
            {"builder": {"private_knowledge_used": []}},
            audit_id="unreferenced_digest",
            completed_gen_id=2,
            expected_peer_contract_count=2,
        )
        self.assertFalse(unreferenced_digest_report.pass_)
        self.assertFalse(unreferenced_digest_report.metrics["negative_evidence_digest_referenced"])
        self.assertTrue(
            any(
                "negative evidence digest is present" in issue
                for issue in unreferenced_digest_report.blocking_issues
            )
        )

        eight_peer_agenda = {
            "consensus_actions": [{"id": "c", "claim_or_hypothesis": "bounded", "source_id": "f"}],
            "peer_contracts": {f"gen1_peer{i}": {"role": "exploit"} for i in range(8)},
        }
        eight_peer_report = context_auditor.audit_agenda(
            eight_peer_agenda,
            {
                "private_packs": {
                    "skeptic": [{"quality": {"is_negative": True}}],
                    "builder": [{"quality": {"is_negative": False}}],
                }
            },
            {"builder": {"private_knowledge_used": []}},
            audit_id="eight_peer",
            completed_gen_id=2,
            expected_peer_contract_count=8,
        )
        self.assertTrue(eight_peer_report.pass_)
        self.assertFalse(
            any("peer_contracts count" in issue for issue in eight_peer_report.blocking_issues)
        )

        eight_peer_fallback_report = context_auditor.audit_agenda(
            eight_peer_agenda,
            {
                "private_packs": {
                    "skeptic": [{"quality": {"is_negative": True}}],
                    "builder": [{"quality": {"is_negative": False}}],
                }
            },
            {"builder": {"private_knowledge_used": []}},
            audit_id="eight_peer_fallback",
            completed_gen_id=2,
        )
        self.assertTrue(eight_peer_fallback_report.pass_)
        self.assertFalse(
            any(
                "peer_contracts count" in issue
                for issue in eight_peer_fallback_report.blocking_issues
            )
        )

        healthy = context_auditor.audit_agenda(
            {
                "consensus_actions": [
                    {"id": "c", "claim_or_hypothesis": "bounded", "source_id": "f"}
                ],
                "peer_contracts": {"a": {}, "b": {}, "c": {}},
            },
            {
                "private_packs": {
                    "skeptic": [{"quality": {"is_negative": True}}],
                    "builder": [{"quality": {"is_negative": False}}],
                }
            },
            {"builder": {"private_knowledge_used": []}},
            audit_id="healthy",
            completed_gen_id=2,
        )
        self.assertTrue(healthy.pass_)

    def test_audit_agenda_peer_contracts_bound_scales_with_cohort_size(self) -> None:
        """Issue #100: when ``cohort_size`` is supplied the peer_contracts

        count check is ``max(1, cohort-2) <= len <= cohort+1`` rather than
        the bundled-default ``3-6`` bound. Tasks with cohort_size > 6
        (e.g. 8-GPU rocket panel) need this to pass audit.
        """
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory import (
            context_auditor,
        )

        agenda_8 = {
            "consensus_actions": [{"id": "c", "claim_or_hypothesis": "bounded", "source_id": "f"}],
            "peer_contracts": {f"peer{i}": {} for i in range(8)},  # exact cohort
        }
        pack = {
            "private_packs": {
                "skeptic": [{"quality": {"is_negative": True}}],
                "builder": [{"quality": {"is_negative": False}}],
            }
        }
        memos = {"builder": {"private_knowledge_used": []}}

        # Without cohort_size, the merged architecture intentionally avoids
        # falling back to the historical 3-6 small-cohort cap. Some callers
        # still lack task_spec context, and large-cohort runs must not be
        # blocked merely because the caller omitted cohort_size.
        legacy_report = context_auditor.audit_agenda(
            agenda_8, pack, memos, audit_id="legacy", completed_gen_id=2
        )
        self.assertTrue(legacy_report.pass_)
        self.assertFalse(
            any("peer_contracts count = 8" in issue for issue in legacy_report.blocking_issues)
        )

        # With cohort_size=8 the same agenda passes (8 ∈ [6, 9]).
        ok_report = context_auditor.audit_agenda(
            agenda_8, pack, memos, audit_id="ok8", completed_gen_id=2, cohort_size=8
        )
        self.assertTrue(ok_report.pass_, msg=str(ok_report.blocking_issues))

        # Chair-side drop: 6 contracts under cohort_size=8 is still in
        # the slack window (max(1, 8-2) = 6) so audit passes.
        agenda_6 = {**agenda_8, "peer_contracts": {f"peer{i}": {} for i in range(6)}}
        drop_report = context_auditor.audit_agenda(
            agenda_6, pack, memos, audit_id="drop", completed_gen_id=2, cohort_size=8
        )
        self.assertTrue(drop_report.pass_, msg=str(drop_report.blocking_issues))

        # 4 contracts under cohort=8 is outside the slack — blocks.
        agenda_4 = {**agenda_8, "peer_contracts": {f"peer{i}": {} for i in range(4)}}
        too_few = context_auditor.audit_agenda(
            agenda_4, pack, memos, audit_id="few", completed_gen_id=2, cohort_size=8
        )
        self.assertFalse(too_few.pass_)
        self.assertTrue(
            any(
                "peer_contracts count = 4" in issue and "6-9" in issue
                for issue in too_few.blocking_issues
            )
        )

    def test_audit_agenda_cohort_size_one_keeps_at_least_one_lower_bound(self) -> None:
        """``cohort_size=1`` is a degenerate single-peer cohort. The lower

        bound floors at 1 (never 0) so an empty peer_contracts on a
        non-first-synthesis still blocks rather than silently passing.
        """
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory import (
            context_auditor,
        )

        empty = context_auditor.audit_agenda(
            {"peer_contracts": {}},
            {"private_packs": {}},
            {},
            audit_id="empty",
            completed_gen_id=2,
            cohort_size=1,
        )
        self.assertFalse(empty.pass_)
        self.assertTrue(any("peer_contracts count = 0" in issue for issue in empty.blocking_issues))


if __name__ == "__main__":
    unittest.main()
