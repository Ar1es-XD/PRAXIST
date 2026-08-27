from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import time
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class SynthesisTriggerContractsTest(unittest.TestCase):
    def test_current_evidence_fields_and_promotion_rejection_control_maturity(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        merged = synthesis_trigger._merged_evidence_payload(
            {
                "effort_ratio": 1.0,
                "coverage_ratio": 1.0,
                "current_aggregate": {
                    "effort_ratio": 0.1,
                    "coverage_ratio": 0.1,
                },
            },
            {},
        )
        self.assertEqual(merged["effort_ratio"], 1.0)
        self.assertEqual(merged["coverage_ratio"], 1.0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                maturity_policy={
                    "min_effort_ratio": 0.75,
                    "min_coverage_ratio": 0.80,
                    "require_ratio_gate": True,
                },
            )
            rejected = {
                "effort_ratio": 1.0,
                "coverage_ratio": 1.0,
                "promotion_eligible": False,
            }
            self.assertTrue(trigger._payload_is_mature(rejected, "ratio mature"))
            self.assertTrue(trigger._payload_is_explicitly_non_durable(rejected))

    def test_deadline_fire_is_idempotent_across_watchdog_and_event_loop_threads(self) -> None:
        import threading

        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen_2"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=2,
                gen_start_time=time.time() - 60,
            )
            workers = [threading.Thread(target=trigger.fire_deadline) for _ in range(8)]
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler_client.freeze_generation"
            ) as freeze:
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join(timeout=1)

            self.assertTrue(trigger.fired)
            self.assertTrue((gen_dir / "STOP_SIGNAL").exists())
            self.assertEqual(freeze.call_count, 1)
            payload = (gen_dir / "STOP_SIGNAL").read_text(encoding="utf-8")
            self.assertIn("trigger_reason=generation_wall_timeout", payload)
            self.assertIn("findings_count=-1", payload)
            self.assertIn("mature_result_count=-1", payload)
            self.assertIn("active_generation_work=-1", payload)

    def test_assessment_fence_failure_remains_retryable(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
            )
            snapshot = synthesis_trigger.TriggerSnapshot(
                fired=False,
                reason="assessment",
                findings_count=1,
                minutes_since_start=1,
                contributing_peers=1,
            )
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler_client.begin_assessment",
                side_effect=[OSError("temporary RPC failure"), True],
            ) as fence:
                self.assertIsNone(trigger.begin_assessment(snapshot))
                self.assertFalse(trigger._assessment_started)
                self.assertTrue(trigger.begin_assessment(snapshot))
            self.assertEqual(fence.call_count, 2)

    def test_mature_result_count_does_not_double_count_legacy_peer_record(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
            )
            with patch.object(
                trigger,
                "_query_mature_evidence_details",
                return_value=(
                    {"gen0_peer0", "gen0_peer1"},
                    {("gen0_peer0", "variant_a"), ("gen0_peer0", "variant_b")},
                    3,
                ),
            ):
                self.assertEqual(trigger.mature_result_count(synchronize=False), 3)

    def test_variantless_mature_findings_count_once_per_peer(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE findings (
                        id TEXT PRIMARY KEY, finding_type TEXT NOT NULL,
                        title TEXT NOT NULL, metrics TEXT NOT NULL DEFAULT '{}',
                        extra TEXT NOT NULL DEFAULT '{}', variant_name TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '', generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL, variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                conn.executemany(
                    "INSERT INTO findings(id, finding_type, title, metrics, peer_id, generation_id) "
                    "VALUES (?, 'result', ?, ?, 'gen0_peer0', 0)",
                    [
                        ("a", "full result", '{"scored_complete": true, "tier_reached": "T1"}'),
                        ("b", "result summary", '{"scored_complete": true, "tier_reached": "T1"}'),
                    ],
                )
                conn.commit()
            finally:
                conn.close()
            self.assertEqual(trigger.mature_result_count(), 1)

    def test_protocol_violation_count_is_not_mature_evidence(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        payload = {
            "finding_type": "result",
            "scored_complete": True,
            "effort_ratio": 1.0,
            "coverage_ratio": 1.0,
            "protocol_integrity_violation_count": 1,
        }

        self.assertTrue(synthesis_trigger._payload_has_hard_non_mature_status(payload))

    def test_inferred_false_completion_is_not_a_hard_synthesis_veto(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        inferred = {
            "finding_type": "result",
            "scored_complete": False,
            "_inferred_scored_complete": True,
            "effort_ratio": 1.0,
            "coverage_ratio": 1.0,
        }
        explicit = {**inferred, "_inferred_scored_complete": False}

        self.assertFalse(synthesis_trigger._payload_has_hard_non_mature_status(inferred))
        self.assertTrue(synthesis_trigger._payload_has_hard_non_mature_status(explicit))

    def test_nonfinite_timing_inputs_fall_back_to_safe_defaults(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=float("nan"),
                min_findings=float("nan"),
                min_interval_minutes=float("inf"),
                max_interval_minutes=float("-inf"),
                min_contributing_peers=float("inf"),
                poll_interval_seconds=float("nan"),
                mature_quorum_fraction=float("nan"),
                cohort_size=float("nan"),
                adaptive_policy={"enabled": "false"},
            )

            self.assertEqual(trigger.min_findings, 30)
            self.assertEqual(trigger.min_interval_minutes, 120.0)
            self.assertEqual(trigger.max_interval_minutes, 240.0)
            self.assertEqual(trigger.min_contributing_peers, 3)
            self.assertEqual(trigger.poll_interval_seconds, 900.0)
            self.assertEqual(trigger.mature_quorum_fraction, 0.0)
            self.assertEqual(trigger.cohort_size, 0)
            self.assertEqual(trigger.required_mature_result_peers, 0)
            self.assertFalse(trigger.adaptive_policy.enabled)

            clamped = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                mature_quorum_fraction=1.5,
                cohort_size=4,
                adaptive_policy={"enabled": "false"},
            )
            self.assertEqual(clamped.mature_quorum_fraction, 1.0)
            self.assertEqual(clamped.required_mature_result_peers, 4)

    def test_query_evaluate_and_marker_paths_are_result_preserving(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                min_findings=3,
                min_interval_minutes=10,
                max_interval_minutes=60,
                min_contributing_peers=2,
                pre_eval_sync_callback=lambda: (_ for _ in ()).throw(RuntimeError("sync")),
            )
            self.assertEqual(
                trigger._watch_paths(),
                [
                    root / "shared_findings",
                    root / "protected_pids",
                    root / "shared_store.db",
                ],
            )
            self.assertTrue(trigger._is_trigger_event(Path("finding.JSON")))
            self.assertTrue(trigger._is_trigger_event(Path("shared_store.db")))
            self.assertTrue(trigger._is_trigger_event(Path("shared_store.db-wal")))
            self.assertFalse(trigger._is_trigger_event(Path("finding.json.tmp")))
            self.assertEqual(trigger._query_gen_state(), (0, 0))

            custom_store = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                store_db_filename="findings.sqlite",
            )
            self.assertEqual(custom_store._watch_paths()[-1], root / "findings.sqlite")
            self.assertTrue(custom_store._is_trigger_event(Path("findings.sqlite")))
            self.assertTrue(custom_store._is_trigger_event(Path("findings.sqlite-wal")))
            self.assertTrue(custom_store._is_trigger_event(Path("findings.sqlite-shm")))
            self.assertFalse(custom_store._is_trigger_event(Path("shared_store.db-wal")))

            legacy_positional_store = synthesis_trigger.SynthesisTrigger(
                root,
                gen_dir,
                0,
                time.time(),
                30,
                120.0,
                240.0,
                3,
                30,
                None,
                None,
                0.0,
                0,
                "legacy.sqlite",
            )
            self.assertEqual(legacy_positional_store.db_path, root / "legacy.sqlite")
            self.assertIsNone(legacy_positional_store.started_peer_ids)

            trigger.db_path.write_text("", encoding="utf-8")

            class FakeConn:
                def __init__(self, rows):
                    self.rows = rows
                    self.closed = False

                def execute(self, *_args, **_kwargs):
                    return iter(self.rows)

                def close(self):
                    self.closed = True

            with patch.object(synthesis_trigger.sqlite3, "connect", return_value=FakeConn([])):
                self.assertEqual(trigger._query_gen_state(), (0, 0))
            with patch.object(
                synthesis_trigger.sqlite3,
                "connect",
                return_value=FakeConn([("gen0_peer0", 3), ("legacy_peer", 1)]),
            ):
                self.assertEqual(trigger._query_gen_state(), (4, 1))
            with patch.object(
                synthesis_trigger.sqlite3,
                "connect",
                side_effect=sqlite3.Error("db"),
            ):
                self.assertEqual(trigger._query_gen_state(), (0, 0))
            with patch.object(
                synthesis_trigger.sqlite3,
                "connect",
                side_effect=RuntimeError("unexpected"),
            ):
                self.assertEqual(trigger._query_gen_state(), (0, 0))

            with patch.object(trigger, "_query_gen_state", return_value=(1, 1)):
                snap = trigger.evaluate()
            self.assertFalse(snap.fired)
            self.assertEqual(snap.reason, "not_yet")
            self.assertGreaterEqual(
                trigger._seconds_until_next_timer_check(
                    synthesis_trigger.TriggerSnapshot(False, "not_yet", 3, 0.0, 2)
                ),
                1.0,
            )

            marker_snap = synthesis_trigger.TriggerSnapshot(False, "not_yet", 1, 2.5, 1)
            marker_snap.mature_result_count = 2
            trigger.write_postgen_marker(marker_snap)
            marker_path = gen_dir / synthesis_trigger.STOP_SIGNAL_POSTGEN_FILENAME
            self.assertTrue(marker_path.exists())
            self.assertIn("mature_result_count=2", marker_path.read_text())
            with patch.object(Path, "write_text", side_effect=OSError("disk")):
                trigger.write_postgen_marker(marker_snap)

            fire_snap = synthesis_trigger.TriggerSnapshot(True, "safety_cap", 2, 3.0, 1)
            with (
                patch.object(synthesis_trigger.os, "open", side_effect=OSError("no fsync")),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.experiment_scheduler_client.freeze_generation"
                ) as freeze,
            ):
                trigger.fire(fire_snap)
            freeze.assert_called_once_with(0, "safety_cap")
            self.assertTrue(trigger.fired)
            self.assertTrue(trigger.stop_signal_path.exists())
            trigger.write_postgen_marker(marker_snap)

            with patch.object(Path, "exists", side_effect=OSError("bad path")):
                self.assertFalse(synthesis_trigger.stop_signal_present(gen_dir))

    def test_synthetic_producers_do_not_contribute_or_repeat_malformed_warnings(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                cohort_size=2,
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.execute(
                    "CREATE TABLE findings (peer_id TEXT NOT NULL, generation_id INTEGER NOT NULL)"
                )
                conn.executemany(
                    "INSERT INTO findings(peer_id, generation_id) VALUES (?, 0)",
                    [
                        ("gen0_peer0",),
                        ("gen0_result_artifact",),
                        ("gen0_late_signal",),
                        ("gen0_protected_jobs",),
                        ("gen0_unknown_peer",),
                        ("gen1_result_artifact",),
                        ("gems_agent",),
                        ("tiered_eval_auto",),
                        ("gen0-peer1",),
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            with patch.object(synthesis_trigger.logger, "warning") as warning:
                self.assertEqual(trigger._query_gen_state(), (9, 1))
                self.assertEqual(trigger._query_gen_state(), (9, 1))

        warning.assert_called_once()
        rendered_warning = warning.call_args.args[0] % warning.call_args.args[1:]
        self.assertIn("gen0-peer1", rendered_warning)
        self.assertIn("gen1_result_artifact", rendered_warning)
        self.assertNotIn("gen0_result_artifact", rendered_warning)
        self.assertNotIn("tiered_eval_auto", rendered_warning)

    def test_result_evidence_peers_uses_canonical_materialized_owner(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                cohort_size=4,
                adaptive_policy={"enabled": True},
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL,
                        variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}',
                        extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                owned_metrics = {
                    "scored_complete": True,
                    "auto_materialized_from_result_artifact": True,
                    "source_result_path": "results/owned/summary.json",
                    "source_result_sha256": "owned-sha",
                }
                unknown_metrics = {
                    **owned_metrics,
                    "source_result_path": "results/ambiguous/summary.json",
                    "source_result_sha256": "ambiguous-sha",
                }
                conn.executemany(
                    "INSERT INTO findings(finding_type, title, metrics, variant_name, peer_id, "
                    "generation_id) VALUES ('result', ?, ?, ?, ?, 0)",
                    [
                        (
                            "owned result",
                            json.dumps(owned_metrics),
                            "owned_result",
                            "gen0_peer2",
                        ),
                        (
                            "ambiguous result",
                            json.dumps(unknown_metrics),
                            "ambiguous_result",
                            "gen0_unknown_peer",
                        ),
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            with patch.object(
                trigger,
                "_infer_auto_result_peer",
                return_value="gen0_peer3",
            ) as infer_peer:
                evidence_units, result_evidence_peers = trigger._query_adaptive_state()
                mature_peers = trigger.mature_peer_ids(synchronize=False)

            self.assertEqual(evidence_units, 2.0)
            self.assertEqual(result_evidence_peers, 1)
            self.assertEqual(mature_peers, {"gen0_peer2"})
            infer_peer.assert_not_called()

    def test_adaptive_policy_waits_for_formal_evidence_and_drains_active_evals(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time() - 180,
                min_findings=999,
                min_interval_minutes=1,
                max_interval_minutes=20,
                min_contributing_peers=2,
                adaptive_policy={
                    "enabled": True,
                    "min_evidence_units": 3.0,
                    "min_formal_result_peers": 2,
                    "min_interval_floor_minutes": 2,
                    "max_interval_ceiling_minutes": 30,
                    "drain_grace_minutes": 5,
                    "evidence_weights": {"T1": 1.0, "T2": 2.0, "T3": 4.0},
                },
            )
            self.assertEqual(trigger.min_interval_minutes, 2)
            self.assertEqual(trigger.max_interval_minutes, 30)

            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE findings (
                        id TEXT PRIMARY KEY,
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        notes TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0,
                        timestamp TEXT NOT NULL,
                        extra TEXT NOT NULL DEFAULT '{}'
                    );
                    CREATE TABLE metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        notes TEXT NOT NULL DEFAULT '',
                        step INTEGER NOT NULL DEFAULT 0,
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0,
                        timestamp TEXT NOT NULL
                    );
                """)
                rows = [
                    (
                        "m1",
                        "variant_a",
                        '{"tier_reached": "T1", "epochs": 500}',
                        "gen0_peer0",
                    ),
                    (
                        "m2",
                        "variant_b",
                        '{"tier_reached": "T2", "epochs": 500}',
                        "gen0_peer1",
                    ),
                    (
                        "m3_smoke",
                        "variant_c",
                        '{"tier_reached": "T1", "epochs": 1}',
                        "gen0_peer2",
                    ),
                ]
                conn.executemany(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, "
                    "generation_id, timestamp) VALUES (?, ?, ?, ?, 0, 'now')",
                    rows,
                )
                findings = [
                    (
                        "f1",
                        "result",
                        "formal a",
                        "{}",
                        "variant_a",
                        "gen0_peer0",
                    ),
                    (
                        "f2",
                        "result",
                        "formal b",
                        "{}",
                        "variant_b",
                        "gen0_peer1",
                    ),
                ]
                conn.executemany(
                    "INSERT INTO findings(id, finding_type, title, metrics, "
                    "variant_name, peer_id, generation_id, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, 0, 'now')",
                    findings,
                )
                conn.commit()
            finally:
                conn.close()

            evidence_units, formal_peers = trigger._query_adaptive_state()
            self.assertEqual(evidence_units, 3.25)
            self.assertEqual(formal_peers, 2)

            with patch.object(trigger, "_active_protected_pid_count", return_value=1):
                snap = trigger.evaluate()
            self.assertFalse(snap.fired)
            self.assertEqual(snap.reason, "draining_active_evals")
            self.assertEqual(snap.formal_result_peers, 2)
            self.assertEqual(snap.active_protected_pids, 1)
            self.assertTrue(trigger.closing)
            self.assertTrue((gen_dir / synthesis_trigger.CLOSING_SIGNAL_FILENAME).exists())
            self.assertLessEqual(trigger._seconds_until_next_timer_check(snap), 60.0)

            with patch.object(trigger, "_active_protected_pid_count", return_value=0):
                snap = trigger.evaluate()
            self.assertTrue(snap.fired)
            self.assertEqual(snap.reason, "adaptive_evidence")

    def test_adaptive_state_counts_one_artifact_once_across_store_aliases(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                cohort_size=2,
                adaptive_policy={"enabled": True},
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL,
                        variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}',
                        extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                artifact = (
                    '{"scored_complete": true, "source_result_path": '
                    '"results/shared/summary.json", "source_result_sha256": "shared-sha"}'
                )
                conn.execute(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, generation_id) "
                    "VALUES ('metric_alias', 'metric_alias', ?, 'gen0_peer0', 0)",
                    (artifact,),
                )
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, variant_name, peer_id, "
                    "generation_id) VALUES ('result', 'finding alias', ?, 'finding_alias', "
                    "'gen0_peer0', 0)",
                    (artifact,),
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger._query_adaptive_state(), (1.0, 1))

    def test_adaptive_ready_cannot_bypass_configured_mature_quorum(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            active_work = {"count": 1}
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time() - 120,
                min_findings=1,
                min_interval_minutes=1,
                max_interval_minutes=60,
                min_contributing_peers=1,
                adaptive_policy={
                    "enabled": True,
                    "min_evidence_units": 1,
                    "min_formal_result_peers": 1,
                },
                mature_quorum_fraction=0.5,
                cohort_size=4,
                cohort_active_peers_callback=lambda: active_work["count"],
            )
            with (
                patch.object(trigger, "_query_gen_state", return_value=(1, 1)),
                patch.object(trigger, "_query_adaptive_state", return_value=(1.0, 1)),
                patch.object(trigger, "_query_mature_state", return_value=1),
                patch.object(trigger, "mature_result_count", return_value=2),
                patch.object(trigger, "_active_protected_pid_count", return_value=0),
                patch.object(trigger, "begin_assessment", return_value=True),
            ):
                draining = trigger.evaluate()
            self.assertFalse(draining.fired)
            self.assertEqual(draining.reason, "assessment_mature_topup")
            self.assertFalse(trigger.closing)

            active_work["count"] = 0
            with (
                patch.object(trigger, "_query_gen_state", return_value=(1, 1)),
                patch.object(trigger, "_query_adaptive_state", return_value=(1.0, 1)),
                patch.object(trigger, "_query_mature_state", return_value=1),
                patch.object(trigger, "mature_result_count", return_value=2),
                patch.object(trigger, "_active_protected_pid_count", return_value=0),
            ):
                insufficient = trigger.evaluate()
            self.assertTrue(insufficient.fired)
            self.assertEqual(insufficient.reason, "cohort_drained_insufficient_mature")

    def test_boundary_refreshes_lightweight_integration_without_canonical_cutoff(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import generation_boundary

        class Frontier:
            def __init__(self) -> None:
                self.calls: list[list[str]] = []

            def get_summary(self):
                return []

            def promote(self, _gen_id, findings):
                self.calls.append([str(item.get("id")) for item in findings])
                return findings

        async def successful_pi(*, completed_gen_id):
            return SimpleNamespace(
                success=True,
                next_gen_id=completed_gen_id + 1,
                agenda_path="agenda.yaml",
                duration_seconds=0,
            )

        with tempfile.TemporaryDirectory() as tmp:
            findings = iter(
                [
                    [{"id": "peer6", "variant_name": "a", "metrics": {}}],
                    [
                        {"id": "peer6", "variant_name": "a", "metrics": {}},
                        {"id": "peer5", "variant_name": "b", "metrics": {}},
                    ],
                ]
            )
            frontier = Frontier()
            loop = SimpleNamespace(
                run_dir=Path(tmp),
                _collect_findings_for_generation=lambda _gen: next(findings),
                _strategy_for_gen=lambda _gen: "explore",
                frontier=frontier,
                task_spec=SimpleNamespace(
                    generation_policy=SimpleNamespace(max_generations=3),
                    research_memory=SimpleNamespace(enabled=False),
                    evaluation=SimpleNamespace(diversity_dimensions=[]),
                ),
                _graph_maintainer=None,
            )
            asyncio.run(
                generation_boundary.complete_generation_boundary(
                    loop,
                    gen_id=0,
                    pi_agent=SimpleNamespace(run=successful_pi),
                    pi_cfg=SimpleNamespace(strict=False),
                )
            )

        self.assertEqual(frontier.calls, [["peer6", "peer5"]])

    def test_server_boundary_excludes_source_less_finding_added_after_collection(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import generation_boundary

        class Frontier:
            def __init__(self) -> None:
                self.findings: list[dict] = []

            def get_summary(self):
                return []

            def promote(self, _gen_id, findings):
                self.findings = list(findings)
                return self.findings

        findings = iter(
            [
                [],
                [{"id": "server-late", "variant_name": "late", "metrics": {"score": 1.0}}],
            ]
        )
        frontier = Frontier()
        with tempfile.TemporaryDirectory() as tmp:
            loop = SimpleNamespace(
                run_dir=Path(tmp),
                findings_dir=Path(tmp) / "shared_findings",
                local_mode=False,
                _collect_findings_for_generation=lambda _gen: next(findings),
                _strategy_for_gen=lambda _gen: "explore",
                frontier=frontier,
                task_spec=SimpleNamespace(
                    generation_policy=SimpleNamespace(max_generations=1),
                    research_memory=SimpleNamespace(enabled=False),
                    evaluation=SimpleNamespace(diversity_dimensions=[]),
                ),
                _graph_maintainer=None,
                _findings_sync=None,
                gems=None,
            )

            asyncio.run(
                generation_boundary.complete_generation_boundary(
                    loop,
                    gen_id=0,
                    pi_agent=None,
                    pi_cfg=SimpleNamespace(strict=False),
                )
            )

        self.assertEqual(frontier.findings, [])

    def test_boundary_promotes_finding_present_in_atomic_canonical_cutoff(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import generation_boundary

        class Frontier:
            def __init__(self) -> None:
                self.calls: list[list[str]] = []

            def get_summary(self):
                return []

            def promote(self, _gen_id, findings):
                self.calls.append([str(item.get("id")) for item in findings])
                return findings

        async def successful_pi(*, completed_gen_id):
            return SimpleNamespace(
                success=True,
                next_gen_id=completed_gen_id + 1,
                agenda_path="agenda.yaml",
                duration_seconds=0,
            )

        first = {"id": "peer6", "variant_name": "a", "metrics": {}}
        arrived_before_cutoff = {"id": "peer5", "variant_name": "b", "metrics": {}}
        findings = iter([[first], [first, arrived_before_cutoff]])
        frontier = Frontier()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend.tools.local_store."
                "snapshot_findings_at_cutoff",
                return_value=(datetime.now(UTC), [first, arrived_before_cutoff]),
            ),
        ):
            run_dir = Path(tmp)
            loop = SimpleNamespace(
                run_dir=run_dir,
                findings_dir=run_dir / "shared_findings",
                local_mode=True,
                _collect_findings_for_generation=lambda _gen: next(findings),
                _strategy_for_gen=lambda _gen: "explore",
                frontier=frontier,
                task_spec=SimpleNamespace(
                    generation_policy=SimpleNamespace(max_generations=3),
                    research_memory=SimpleNamespace(enabled=False),
                    evaluation=SimpleNamespace(diversity_dimensions=[]),
                ),
                _graph_maintainer=None,
            )
            asyncio.run(
                generation_boundary.complete_generation_boundary(
                    loop,
                    gen_id=0,
                    pi_agent=SimpleNamespace(run=successful_pi),
                    pi_cfg=SimpleNamespace(strict=False),
                )
            )

        self.assertEqual(frontier.calls, [["peer6", "peer5"]])

    def test_boundary_promotes_pre_cutoff_value_and_retains_late_update_as_signal(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import generation_boundary

        class Frontier:
            def __init__(self) -> None:
                self.findings: list[dict] = []

            def get_summary(self):
                return []

            def promote(self, _gen_id, findings):
                self.findings = list(findings)
                return self.findings

        at_cutoff = {
            "id": "same-finding",
            "variant_name": "candidate",
            "metrics": {"score": 1.0},
        }
        after_cutoff = {
            **at_cutoff,
            "metrics": {
                "score": 2.0,
                "late_after_generation_boundary": True,
                "generation_boundary_pending_commit": True,
            },
        }
        memory_findings: list[dict] = []
        frontier = Frontier()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend.tools.local_store."
                "snapshot_findings_at_cutoff",
                return_value=(datetime.now(UTC), [at_cutoff]),
            ),
        ):
            run_dir = Path(tmp)
            loop = SimpleNamespace(
                run_dir=run_dir,
                findings_dir=run_dir / "shared_findings",
                local_mode=True,
                _collect_findings_for_generation=lambda _gen: [at_cutoff],
                _collect_findings_for_boundary=lambda _gen, **_kwargs: [after_cutoff],
                _strategy_for_gen=lambda _gen: "explore",
                _update_research_memory_post_gen=lambda **kwargs: memory_findings.extend(
                    kwargs["findings"]
                ),
                frontier=frontier,
                task_spec=SimpleNamespace(
                    generation_policy=SimpleNamespace(max_generations=1),
                    research_memory=SimpleNamespace(enabled=True),
                    evaluation=SimpleNamespace(
                        diversity_dimensions=[],
                        constructive_peer_mix_enabled=False,
                    ),
                ),
                _graph_maintainer=None,
                _findings_sync=None,
                gems=None,
            )

            asyncio.run(
                generation_boundary.complete_generation_boundary(
                    loop,
                    gen_id=0,
                    pi_agent=None,
                    pi_cfg=SimpleNamespace(strict=False),
                )
            )
            marker = json.loads(
                (run_dir / "gen_0" / "generation_boundary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(frontier.findings[0]["metrics"]["score"], 1.0)
        self.assertEqual(memory_findings[0]["metrics"]["score"], 2.0)
        self.assertTrue(memory_findings[0]["metrics"]["late_after_generation_boundary"])
        snapshot_value = marker["evidence_source_snapshot_at_cutoff"][
            "canonical-finding:same-finding"
        ]
        self.assertEqual(len(snapshot_value), 64)
        self.assertNotIn("canonical-finding-payload", snapshot_value)

    def test_resumed_boundary_promotes_value_saved_in_transient_cutoff(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
            generation_boundary,
            resume_state,
        )

        class Frontier:
            def __init__(self) -> None:
                self.findings: list[dict] = []

            def get_summary(self):
                return []

            def promote(self, _gen_id, findings):
                self.findings = list(findings)
                return self.findings

        at_cutoff = {
            "id": "resume-same-finding",
            "variant_name": "candidate",
            "metrics": {"score": 1.0},
        }
        after_cutoff = {
            **at_cutoff,
            "metrics": {
                "score": 9.0,
                "late_after_generation_boundary": True,
                "generation_boundary_pending_commit": True,
            },
        }
        frontier = Frontier()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen_dir = run_dir / "gen_0"
            gen_dir.mkdir()
            (gen_dir / "generation_results.json").write_text("[]", encoding="utf-8")
            cutoff = datetime.now(UTC)
            snapshot = findings_collection.include_finding_sources_in_snapshot(
                {},
                [at_cutoff],
                run_dir=run_dir,
                findings_dir=run_dir / "shared_findings",
                gen_id=0,
                cutoff=cutoff,
            )
            self.assertTrue(
                resume_state.write_boundary_evidence_checkpoint(
                    run_dir,
                    gen_id=0,
                    cutoff=cutoff,
                    evidence_source_snapshot=snapshot,
                )
            )
            loop = SimpleNamespace(
                run_dir=run_dir,
                findings_dir=run_dir / "shared_findings",
                local_mode=True,
                _collect_findings_for_generation=lambda _gen: [after_cutoff],
                _collect_findings_for_boundary=lambda _gen, **_kwargs: [after_cutoff],
                _strategy_for_gen=lambda _gen: "explore",
                frontier=frontier,
                task_spec=SimpleNamespace(
                    generation_policy=SimpleNamespace(max_generations=1),
                    research_memory=SimpleNamespace(enabled=False),
                    evaluation=SimpleNamespace(
                        diversity_dimensions=[],
                        constructive_peer_mix_enabled=False,
                    ),
                ),
                _graph_maintainer=None,
                _findings_sync=None,
                gems=None,
            )

            asyncio.run(
                generation_boundary.complete_generation_boundary(
                    loop,
                    gen_id=0,
                    pi_agent=None,
                    pi_cfg=SimpleNamespace(strict=False),
                )
            )

            marker = json.loads((gen_dir / "generation_boundary.json").read_text(encoding="utf-8"))

        self.assertEqual(frontier.findings[0]["metrics"]["score"], 1.0)
        self.assertEqual(
            len(
                marker["evidence_source_snapshot_at_cutoff"][
                    "canonical-finding:resume-same-finding"
                ]
            ),
            64,
        )

    def test_boundary_refresh_failure_does_not_mutate_frontier(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import generation_boundary

        class Frontier:
            def __init__(self) -> None:
                self.calls = 0

            def get_summary(self):
                return []

            def promote(self, _gen_id, _findings):
                self.calls += 1
                return []

        frontier = Frontier()
        calls = 0

        def collect(_gen_id):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("materialization unavailable")
            return [{"id": "candidate", "variant_name": "candidate", "metrics": {}}]

        with tempfile.TemporaryDirectory() as tmp:
            loop = SimpleNamespace(
                run_dir=Path(tmp),
                _collect_findings_for_generation=collect,
                _strategy_for_gen=lambda _gen: "explore",
                frontier=frontier,
                task_spec=SimpleNamespace(
                    generation_policy=SimpleNamespace(max_generations=2),
                    research_memory=SimpleNamespace(enabled=False),
                    evaluation=SimpleNamespace(diversity_dimensions=[]),
                ),
                _graph_maintainer=None,
                _findings_sync=None,
                gems=None,
            )
            with self.assertRaisesRegex(OSError, "materialization unavailable"):
                asyncio.run(
                    generation_boundary.complete_generation_boundary(
                        loop,
                        gen_id=0,
                        pi_agent=None,
                        pi_cfg=SimpleNamespace(strict=False),
                    )
                )

        self.assertEqual(frontier.calls, 0)

    def test_boundary_checkpoint_failure_prevents_frontier_mutation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import generation_boundary

        class Frontier:
            def __init__(self) -> None:
                self.calls = 0

            def get_summary(self):
                return []

            def promote(self, _gen_id, _findings):
                self.calls += 1
                return []

        frontier = Frontier()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen_dir = run_dir / "gen_0"
            gen_dir.mkdir()
            (gen_dir / "generation_results.json").write_text("[]", encoding="utf-8")
            loop = SimpleNamespace(
                run_dir=run_dir,
                findings_dir=run_dir / "shared_findings",
                local_mode=False,
                _collect_findings_for_generation=lambda _gen: [],
                _strategy_for_gen=lambda _gen: "explore",
                frontier=frontier,
                task_spec=SimpleNamespace(
                    generation_policy=SimpleNamespace(max_generations=1),
                    research_memory=SimpleNamespace(enabled=False),
                    evaluation=SimpleNamespace(diversity_dimensions=[]),
                ),
                _graph_maintainer=None,
                _findings_sync=None,
                gems=None,
            )
            with (
                patch.object(
                    generation_boundary,
                    "write_boundary_evidence_checkpoint",
                    side_effect=OSError("checkpoint unavailable"),
                ),
                self.assertRaisesRegex(OSError, "checkpoint unavailable"),
            ):
                asyncio.run(
                    generation_boundary.complete_generation_boundary(
                        loop,
                        gen_id=0,
                        pi_agent=None,
                        pi_cfg=SimpleNamespace(strict=False),
                    )
                )

        self.assertEqual(frontier.calls, 0)
        self.assertEqual(loop._boundary_evidence_cutoff[0], 0)

    def test_boundary_retry_reuses_first_cutoff_after_checkpoint_write_failure(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import generation_boundary

        class Frontier:
            def __init__(self) -> None:
                self.findings: list[dict] = []

            def get_summary(self):
                return []

            def promote(self, _gen_id, findings):
                self.findings = list(findings)
                return self.findings

        first = {"id": "at-cutoff", "variant_name": "at-cutoff", "metrics": {}}
        late = {"id": "after-cutoff", "variant_name": "after-cutoff", "metrics": {}}
        visible = [first]
        frontier = Frontier()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen_dir = run_dir / "gen_0"
            gen_dir.mkdir()
            (gen_dir / "generation_results.json").write_text("[]", encoding="utf-8")
            loop = SimpleNamespace(
                run_dir=run_dir,
                findings_dir=run_dir / "shared_findings",
                local_mode=False,
                _boundary_evidence_cutoff=None,
                _collect_findings_for_generation=lambda _gen: list(visible),
                _strategy_for_gen=lambda _gen: "explore",
                frontier=frontier,
                task_spec=SimpleNamespace(
                    generation_policy=SimpleNamespace(max_generations=1),
                    research_memory=SimpleNamespace(enabled=False),
                    evaluation=SimpleNamespace(diversity_dimensions=[]),
                ),
                _graph_maintainer=None,
                _findings_sync=None,
                gems=None,
            )
            with patch.object(
                generation_boundary,
                "write_boundary_evidence_checkpoint",
                side_effect=[OSError("checkpoint unavailable"), True],
            ) as write_checkpoint:
                with self.assertRaisesRegex(OSError, "checkpoint unavailable"):
                    asyncio.run(
                        generation_boundary.complete_generation_boundary(
                            loop,
                            gen_id=0,
                            pi_agent=None,
                            pi_cfg=SimpleNamespace(strict=False),
                        )
                    )
                first_cutoff = loop._boundary_evidence_cutoff
                visible.append(late)
                asyncio.run(
                    generation_boundary.complete_generation_boundary(
                        loop,
                        gen_id=0,
                        pi_agent=None,
                        pi_cfg=SimpleNamespace(strict=False),
                    )
                )

        self.assertEqual(write_checkpoint.call_count, 2)
        self.assertEqual(write_checkpoint.call_args.kwargs["cutoff"], first_cutoff[1])
        self.assertEqual(
            write_checkpoint.call_args.kwargs["evidence_source_snapshot"],
            first_cutoff[2],
        )
        self.assertEqual([finding["id"] for finding in frontier.findings], ["at-cutoff"])

    def test_boundary_refresh_defers_results_published_after_cutoff(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import generation_boundary

        class Frontier:
            def __init__(self) -> None:
                self.findings = []

            def get_summary(self):
                return []

            def promote(self, _gen_id, findings):
                self.findings = list(findings)
                return self.findings

        frontier = Frontier()
        calls = 0

        def collect(_gen_id):
            nonlocal calls
            calls += 1
            early = {"id": "early", "variant_name": "early", "metrics": {}}
            late = {
                "id": "late",
                "variant_name": "late",
                "source_result_path": "results/late/summary.json",
                "metrics": {},
            }
            return [early] if calls == 1 else [early, late]

        with tempfile.TemporaryDirectory() as tmp:
            loop = SimpleNamespace(
                run_dir=Path(tmp),
                _collect_findings_for_generation=collect,
                _strategy_for_gen=lambda _gen: "explore",
                frontier=frontier,
                task_spec=SimpleNamespace(
                    generation_policy=SimpleNamespace(max_generations=1),
                    research_memory=SimpleNamespace(enabled=False),
                    evaluation=SimpleNamespace(diversity_dimensions=[]),
                ),
                _graph_maintainer=None,
                _findings_sync=None,
                gems=None,
            )
            with patch.object(
                generation_boundary,
                "finding_source_published_after",
                side_effect=lambda finding, **_kwargs: finding.get("id") == "late",
            ):
                asyncio.run(
                    generation_boundary.complete_generation_boundary(
                        loop,
                        gen_id=0,
                        pi_agent=None,
                        pi_cfg=SimpleNamespace(strict=False),
                    )
                )

        self.assertEqual([finding["id"] for finding in frontier.findings], ["early"])

    def test_boundary_uses_reconciled_source_snapshot_and_cutoff(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import generation_boundary

        expected_cutoff = datetime.now(UTC)
        observed_cutoff = 0.0

        def collect_boundary(_gen_id, *, evidence_cutoff, evidence_source_snapshot):
            nonlocal observed_cutoff
            observed_cutoff = evidence_cutoff.timestamp()
            self.assertEqual(
                evidence_source_snapshot["canonical-finding-snapshot:v1"],
                "captured",
            )
            self.assertTrue(evidence_source_snapshot["result-source-root:v1"].startswith("root:"))
            return []

        class Frontier:
            def get_summary(self):
                return []

            def promote(self, _gen_id, findings):
                return findings

        with tempfile.TemporaryDirectory() as tmp:
            loop = SimpleNamespace(
                run_dir=Path(tmp),
                findings_dir=Path(tmp) / "shared_findings",
                local_mode=True,
                _collect_findings_for_generation=lambda _gen: [],
                _collect_findings_for_boundary=collect_boundary,
                _strategy_for_gen=lambda _gen: "explore",
                frontier=Frontier(),
                task_spec=SimpleNamespace(
                    generation_policy=SimpleNamespace(max_generations=1),
                    research_memory=SimpleNamespace(enabled=False),
                    evaluation=SimpleNamespace(diversity_dimensions=[]),
                ),
                _graph_maintainer=None,
                _findings_sync=None,
                gems=None,
            )
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.tools.local_store."
                "snapshot_findings_at_cutoff",
                return_value=(expected_cutoff, []),
            ):
                asyncio.run(
                    generation_boundary.complete_generation_boundary(
                        loop,
                        gen_id=0,
                        pi_agent=None,
                        pi_cfg=SimpleNamespace(strict=False),
                    )
                )

        self.assertEqual(observed_cutoff, expected_cutoff.timestamp())

    def test_boundary_marker_failure_prevents_successful_boundary_return(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import generation_boundary

        class Frontier:
            def get_summary(self):
                return []

            def promote(self, _gen_id, findings):
                return findings

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            loop = SimpleNamespace(
                run_dir=run_dir,
                _collect_findings_for_generation=lambda _gen: [],
                _strategy_for_gen=lambda _gen: "explore",
                frontier=Frontier(),
                task_spec=SimpleNamespace(
                    generation_policy=SimpleNamespace(max_generations=1),
                    research_memory=SimpleNamespace(enabled=False),
                    evaluation=SimpleNamespace(diversity_dimensions=[]),
                ),
                _graph_maintainer=None,
                _findings_sync=None,
                gems=None,
            )
            with (
                patch.object(
                    generation_boundary,
                    "write_boundary_marker",
                    side_effect=OSError("disk unavailable"),
                ),
                self.assertRaisesRegex(RuntimeError, "boundary could not be committed"),
            ):
                asyncio.run(
                    generation_boundary.complete_generation_boundary(
                        loop,
                        gen_id=0,
                        pi_agent=None,
                        pi_cfg=SimpleNamespace(strict=False),
                    )
                )
            self.assertFalse((run_dir / "gen_0" / "generation_boundary.json").exists())

    def test_pi_success_does_not_reclassify_boundary_marker_failure(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import generation_boundary

        class Frontier:
            def get_summary(self):
                return []

            def promote(self, _gen_id, findings):
                return findings

        async def successful_pi(*, completed_gen_id):
            return SimpleNamespace(
                success=True,
                next_gen_id=completed_gen_id + 1,
                agenda_path=Path("agenda.yaml"),
                duration_seconds=1.0,
            )

        with tempfile.TemporaryDirectory() as tmp:
            loop = SimpleNamespace(
                run_dir=Path(tmp),
                _collect_findings_for_generation=lambda _gen: [],
                _strategy_for_gen=lambda _gen: "explore",
                frontier=Frontier(),
                task_spec=SimpleNamespace(
                    generation_policy=SimpleNamespace(max_generations=2),
                    research_memory=SimpleNamespace(enabled=False),
                    evaluation=SimpleNamespace(diversity_dimensions=[]),
                ),
                _graph_maintainer=None,
                _findings_sync=None,
                gems=None,
            )
            with (
                patch.object(
                    generation_boundary,
                    "write_boundary_marker",
                    side_effect=OSError("disk unavailable"),
                ) as marker_write,
                self.assertRaisesRegex(RuntimeError, "boundary could not be committed"),
            ):
                asyncio.run(
                    generation_boundary.complete_generation_boundary(
                        loop,
                        gen_id=0,
                        pi_agent=SimpleNamespace(run=successful_pi),
                        pi_cfg=SimpleNamespace(strict=False),
                    )
                )
            self.assertEqual(marker_write.call_count, 1)

    def test_adaptive_policy_retains_validation_weight_without_formalizing_signal(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time() - 60,
                min_findings=999,
                min_interval_minutes=1,
                max_interval_minutes=20,
                min_contributing_peers=2,
                adaptive_policy={
                    "enabled": True,
                    "min_evidence_units": 1.0,
                    "min_formal_result_peers": 1,
                    "evidence_weights": {"T1": 1.0},
                },
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE findings (
                        id TEXT PRIMARY KEY,
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        notes TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0,
                        timestamp TEXT NOT NULL,
                        extra TEXT NOT NULL DEFAULT '{}'
                    );
                    CREATE TABLE metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        notes TEXT NOT NULL DEFAULT '',
                        step INTEGER NOT NULL DEFAULT 0,
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0,
                        timestamp TEXT NOT NULL
                    );
                """)
                conn.executemany(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, "
                    "generation_id, timestamp) VALUES (?, ?, ?, ?, 0, 'now')",
                    [
                        (
                            "m_validation",
                            "candidate_validation",
                            '{"tier_reached": "T1", "validation_only_result": true, '
                            '"scored_complete": true, "source_result_path": '
                            '"results/candidate_validation/summary.json", '
                            '"source_result_sha256": "validation-sha"}',
                            "gen0_peer0",
                        ),
                        (
                            "m_late",
                            "candidate_late",
                            '{"tier_reached": "T1", "artifact_signal_status": "late_after_generation_boundary"}',
                            "gen0_peer1",
                        ),
                    ],
                )
                conn.executemany(
                    "INSERT INTO findings(id, finding_type, title, metrics, "
                    "variant_name, peer_id, generation_id, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, 0, 'now')",
                    [
                        (
                            "f_quarantine",
                            "result",
                            "candidate quarantine",
                            '{"late_result_policy": "quarantined_signal"}',
                            "candidate_quarantine",
                            "gen0_peer2",
                        ),
                        (
                            "f_scope",
                            "result",
                            "candidate scope",
                            '{"durability_scope": "validation_signal_only"}',
                            "candidate_scope",
                            "gen0_peer3",
                        ),
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            evidence_units, formal_peers = trigger._query_adaptive_state()
            snap = trigger.evaluate()

        self.assertEqual(evidence_units, 2.0)
        self.assertEqual(formal_peers, 0)
        self.assertFalse(snap.fired)
        self.assertNotEqual(snap.reason, "adaptive_evidence")

    def test_adaptive_policy_excludes_digest_confirmed_non_durable_alias(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                cohort_size=1,
                adaptive_policy={
                    "enabled": True,
                    "min_evidence_units": 1.0,
                    "min_formal_result_peers": 1,
                },
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL,
                        variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}',
                        extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                artifact = (
                    '{"child_variant_id": "candidate", "scored_complete": true, '
                    '"source_result_path": '
                    '"results/shared.json", "source_result_sha256": "shared-sha"}'
                )
                conn.execute(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, generation_id) "
                    "VALUES ('run', 'candidate', ?, 'gen0_peer0', 0)",
                    (artifact,),
                )
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, variant_name, peer_id, "
                    "generation_id) VALUES ('result', 'validation alias', ?, 'alias', "
                    "'gen0_peer0', 0)",
                    (
                        '{"child_variant_id": "candidate", '
                        '"validation_only_result": true, "source_result_path": '
                        '"results/shared.json", "source_result_sha256": "shared-sha"}',
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger._query_adaptive_state(), (0.0, 0))

    def test_adaptive_policy_retains_preliminary_units_without_formalizing_alias(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                cohort_size=1,
                adaptive_policy={
                    "enabled": True,
                    "evidence_weights": {"preliminary": 0.25},
                },
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL, variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL, title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}', extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                preliminary = (
                    '{"child_variant_id": "candidate", "evidence_stage": "preliminary", '
                    '"scored_complete": false, '
                    '"effort_ratio": 0.2, '
                    '"coverage_ratio": 0.2, "source_result_path": '
                    '"results/candidate.json", "source_result_sha256": "same-sha"}'
                )
                mature_alias = (
                    '{"child_variant_id": "candidate", "scored_complete": true, '
                    '"effort_ratio": 1.0, '
                    '"coverage_ratio": 1.0, "source_result_path": '
                    '"results/candidate.json", "source_result_sha256": "same-sha"}'
                )
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, variant_name, peer_id, "
                    "generation_id) VALUES ('result', 'preliminary', ?, 'candidate', "
                    "'gen0_peer0', 0)",
                    (preliminary,),
                )
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, variant_name, peer_id, "
                    "generation_id) VALUES ('result', 'mature alias', ?, 'candidate_alias', "
                    "'gen0_peer0', 0)",
                    (mature_alias,),
                )
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, variant_name, peer_id, "
                    "generation_id) VALUES ('result', 'label-only preliminary', ?, "
                    "'label_candidate', 'gen0_peer0', 0)",
                    (
                        '{"child_variant_id": "label", "evidence_stage": "preliminary", '
                        '"scored_complete": false, '
                        '"validation_only_result": true, "source_result_path": '
                        '"results/label.json", "source_result_sha256": "label-sha"}',
                    ),
                )
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, variant_name, peer_id, "
                    "generation_id) VALUES ('result', 'label mature alias', ?, "
                    "'label_alias', 'gen0_peer0', 0)",
                    (
                        '{"child_variant_id": "label", "scored_complete": true, '
                        '"source_result_path": '
                        '"results/label.json", "source_result_sha256": "label-sha"}',
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger._query_adaptive_state(), (0.5, 0))
            self.assertEqual(trigger.mature_result_count(synchronize=False), 0)

    def test_soft_non_mature_marker_never_contributes_formal_peer(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                cohort_size=1,
                adaptive_policy={
                    "enabled": True,
                    "evidence_weights": {"T1": 1.0},
                },
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL, variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL, title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}', extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, variant_name, peer_id, "
                    "generation_id) VALUES ('result', 'scout signal', ?, 'candidate', "
                    "'gen0_peer0', 0)",
                    ('{"tier": "T1", "scout_only": true}',),
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger._query_adaptive_state(), (1.0, 0))

    def test_multistage_preliminary_retains_selected_stage_weight(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                cohort_size=1,
                adaptive_policy={
                    "enabled": True,
                    "evidence_weights": {"T1": 0.4},
                },
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL, variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL, title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}', extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                conn.execute(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, generation_id) "
                    "VALUES ('run', 'candidate', ?, 'gen0_peer0', 0)",
                    ('{"tier": "T1", "evidence_stage": "preliminary", "scored_complete": false}',),
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger._query_adaptive_state(), (0.4, 0))

    def test_shared_artifact_does_not_cross_producer_identity_boundary(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL, variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL, title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}', extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                shared = (
                    '"canonical_variant_id": "shared-family", '
                    '"source_result_path": "results/shared.json", '
                    '"source_result_sha256": "shared-sha"'
                )
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, variant_name, peer_id, "
                    "generation_id) VALUES ('result', 'preliminary child', ?, 'child-a', "
                    "'gen0_peer0', 0)",
                    (
                        '{"child_variant_id": "child-a", "evidence_stage": "preliminary", '
                        f'"scored_complete": false, {shared}}}',
                    ),
                )
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, variant_name, peer_id, "
                    "generation_id) VALUES ('result', 'complete child', ?, 'child-b', "
                    "'gen0_peer1', 0)",
                    (f'{{"child_variant_id": "child-b", "scored_complete": true, {shared}}}',),
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger.mature_result_count(synchronize=False), 1)
            self.assertEqual(trigger.mature_peer_ids(synchronize=False), {"gen0_peer1"})

    def test_producer_identity_preserves_distinct_separators(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL, variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL, title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}', extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                shared = (
                    '"source_result_path": "results/shared.json", '
                    '"source_result_sha256": "shared-sha"'
                )
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, variant_name, peer_id, "
                    "generation_id) VALUES ('result', 'preliminary', ?, 'candidate-a', "
                    "'gen0_peer0', 0)",
                    (
                        '{"child_variant_id": "child/a", "evidence_stage": "preliminary", '
                        f'"scored_complete": false, {shared}}}',
                    ),
                )
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, variant_name, peer_id, "
                    "generation_id) VALUES ('result', 'complete', ?, 'candidate-b', "
                    "'gen0_peer1', 0)",
                    (f'{{"child_variant_id": "child-a", "scored_complete": true, {shared}}}',),
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger.mature_result_count(synchronize=False), 1)
            self.assertEqual(trigger.mature_peer_ids(synchronize=False), {"gen0_peer1"})

    def test_nested_child_identity_takes_precedence_over_root_parent_identity(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL, variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL, title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}', extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                shared = (
                    '"variant_id": "parent", "source_result_path": "results/shared.json", '
                    '"source_result_sha256": "shared-sha"'
                )
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, extra, variant_name, "
                    "peer_id, generation_id) VALUES ('result', 'preliminary child a', ?, ?, "
                    "'parent', 'gen0_peer0', 0)",
                    (
                        f'{{"evidence_stage": "preliminary", "scored_complete": false, {shared}}}',
                        '{"child_variant_id": "child-a"}',
                    ),
                )
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, extra, variant_name, "
                    "peer_id, generation_id) VALUES ('result', 'complete child b', ?, ?, "
                    "'parent', 'gen0_peer1', 0)",
                    (
                        f'{{"scored_complete": true, {shared}}}',
                        '{"child_variant_id": "child-b"}',
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger.mature_result_count(synchronize=False), 1)
            self.assertEqual(trigger.mature_peer_ids(synchronize=False), {"gen0_peer1"})

    def test_missing_producer_alias_deduplicates_same_immutable_artifact(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL, variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL, title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}', extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                shared = (
                    '{"scored_complete": true, "source_result_path": '
                    '"results/candidate.json", "source_result_sha256": "same-sha"}'
                )
                conn.execute(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, generation_id) "
                    "VALUES ('experiment-a', 'candidate', ?, 'gen0_peer0', 0)",
                    (shared,),
                )
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, extra, variant_name, "
                    "peer_id, generation_id) VALUES ('result', 'canonical', ?, "
                    "'{\"canonical_variant_name\": \"candidate\"}', 'candidate', "
                    "'gen0_peer0', 0)",
                    (shared,),
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger.mature_result_count(synchronize=False), 1)

    def test_producerless_alias_does_not_bridge_two_explicit_producers(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL, variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL, title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}', extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                shared = (
                    '"scored_complete": true, "source_result_path": "results/shared.json", '
                    '"source_result_sha256": "shared-sha"'
                )
                conn.executemany(
                    "INSERT INTO findings(finding_type, title, metrics, variant_name, peer_id, "
                    "generation_id) VALUES ('result', ?, ?, ?, ?, 0)",
                    [
                        ("producerless alias", f"{{{shared}}}", "child/a", "gen0_peer0"),
                        (
                            "explicit producer a",
                            f'{{"child_variant_id": "child/a", {shared}}}',
                            "child/a",
                            "gen0_peer0",
                        ),
                        (
                            "explicit producer b",
                            f'{{"child_variant_id": "child-b", {shared}}}',
                            "child-b",
                            "gen0_peer1",
                        ),
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger.mature_result_count(synchronize=False), 2)
            self.assertEqual(
                trigger.mature_peer_ids(synchronize=False),
                {"gen0_peer0", "gen0_peer1"},
            )

    def test_task_complete_stage_overrides_generic_soft_stage_word(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                maturity_policy={"complete_stage_labels": ["scout"]},
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL, variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL, title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}', extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                conn.execute(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, generation_id) "
                    "VALUES ('run', 'candidate', ?, 'gen0_peer0', 0)",
                    (
                        '{"evidence_stage": "scout", "result_status": "preliminary", '
                        '"scout_only": true, "partial_eval": true, "capped": true}',
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger.mature_result_count(synchronize=False), 1)
            self.assertEqual(trigger.mature_peer_count(synchronize=False), 1)

    def test_mature_quorum_same_artifact_validation_signal_suppresses_metric_row(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time() - 180,
                min_findings=999,
                min_interval_minutes=1,
                max_interval_minutes=60,
                min_contributing_peers=1,
                maturity_policy={"min_effort_ratio": 0.75, "min_coverage_ratio": 0.80},
                mature_quorum_fraction=1.0,
                cohort_size=1,
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        notes TEXT NOT NULL DEFAULT '',
                        step INTEGER NOT NULL DEFAULT 0,
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0,
                        timestamp TEXT NOT NULL
                    );
                    CREATE TABLE findings (
                        id TEXT PRIMARY KEY,
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        notes TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0,
                        timestamp TEXT NOT NULL,
                        extra TEXT NOT NULL DEFAULT '{}'
                    );
                """)
                conn.execute(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, "
                    "generation_id, timestamp) VALUES (?, ?, ?, ?, 0, 'now')",
                    (
                        "r0",
                        "candidate_a",
                        '{"child_variant_id": "candidate_a", "effort_ratio": 0.95, '
                        '"coverage_ratio": 0.95, '
                        '"scored_complete": true, "source_result_path": '
                        '"results/candidate_a/summary.json", "source_result_sha256": "abc123"}',
                        "gen0_peer0",
                    ),
                )
                conn.execute(
                    "INSERT INTO findings(id, finding_type, title, content, metrics, "
                    "variant_name, peer_id, generation_id, timestamp, extra) "
                    "VALUES (?, ?, ?, '', ?, ?, ?, 0, 'now', '{}')",
                    (
                        "f0",
                        "result",
                        "candidate_a",
                        '{"child_variant_id": "candidate_a", '
                        '"validation_only_result": true, "source_result_path": '
                        '"results/candidate_a/summary.json", "source_result_sha256": "abc123", '
                        '"auto_materialized_from_result_artifact": true}',
                        "candidate_a",
                        "gen0_result_artifact",
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger._query_mature_state(), 0)

    def test_result_artifact_identity_requires_non_conflicting_coordinates(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            evidence_maturity,
            synthesis_trigger,
        )

        self.assertFalse(
            evidence_maturity.same_result_artifact(
                ("results/preliminary.json", "same-sha"),
                ("results/complete.json", "same-sha"),
            )
        )
        self.assertFalse(
            evidence_maturity.same_result_artifact(
                ("results/candidate.json", "old-sha"),
                ("results/candidate.json", ""),
            )
        )
        self.assertFalse(
            evidence_maturity.same_result_artifact(
                ("results/candidate.json", ""),
                ("results/candidate.json", ""),
            )
        )
        self.assertFalse(
            evidence_maturity.same_result_artifact(
                ("", "shared-sha"),
                ("results/candidate.json", "shared-sha"),
            )
        )
        self.assertEqual(
            synthesis_trigger._result_artifact_key(
                {"result_artifact_path": "./results/candidate.json"}
            ),
            ("results/candidate.json", ""),
        )
        self.assertIsNone(
            synthesis_trigger._result_artifact_key({"source_path": "checkpoints/shared.pt"})
        )
        self.assertIsNone(
            synthesis_trigger._result_artifact_key(
                {
                    "source_result_path": "results/a.json",
                    "extra": {"source_result_sha256": "unrelated-sha"},
                }
            )
        )
        self.assertEqual(
            synthesis_trigger._result_artifact_key(
                {
                    "extra": {
                        "extra": {
                            "source_result_path": "results/nested.json",
                            "source_result_sha256": "nested-sha",
                        }
                    }
                }
            ),
            ("results/nested.json", "nested-sha"),
        )
        self.assertIsNone(
            evidence_maturity.result_snapshot_key(
                {
                    "child_id": "child-a",
                    "result_variant_id": "child-b",
                    "source_result_path": "results/shared.json",
                    "source_result_sha256": "shared-sha",
                }
            )
        )
        self.assertEqual(
            evidence_maturity.result_snapshot_key(
                {
                    "source_result_path": "results/producerless.json",
                    "source_result_sha256": "producerless-sha",
                }
            ),
            ("", "results/producerless.json", "producerless-sha"),
        )
        self.assertTrue(
            evidence_maturity.same_result_snapshot(
                ("", "results/producerless.json", "producerless-sha"),
                ("", "results/producerless.json", "producerless-sha"),
            )
        )
        self.assertFalse(
            evidence_maturity.same_result_snapshot(
                ("", "results/producerless.json", "producerless-sha"),
                ("explicit", "results/producerless.json", "producerless-sha"),
            )
        )
        self.assertEqual(
            evidence_maturity.result_snapshot_key(
                {
                    "variant_id": "shared-parent",
                    "source_result_path": "results/shared.json",
                    "source_result_sha256": "shared-sha",
                }
            ),
            ("", "results/shared.json", "shared-sha"),
        )

    def test_mature_quorum_same_artifact_metric_quarantine_suppresses_metric_row(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                maturity_policy={"min_effort_ratio": 0.75, "min_coverage_ratio": 0.80},
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL,
                        variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}',
                        extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                conn.executemany(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, generation_id) "
                    "VALUES (?, 'candidate_a', ?, 'gen0_peer0', 0)",
                    [
                        (
                            "complete",
                            '{"child_variant_id": "candidate_a", "effort_ratio": 1.0, '
                            '"coverage_ratio": 1.0, '
                            '"scored_complete": true, "source_result_path": '
                            '"results/candidate_a/summary.json", '
                            '"source_result_sha256": "same-sha"}',
                        ),
                        (
                            "quarantine",
                            '{"child_variant_id": "candidate_a", '
                            '"validation_only_result": true, "source_result_path": '
                            '"results/candidate_a/summary.json", '
                            '"source_result_sha256": "same-sha"}',
                        ),
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger._query_mature_state(), 0)

    def test_mature_quorum_does_not_join_validation_coordinates_across_containers(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                cohort_size=1,
                maturity_policy={"min_effort_ratio": 0.75, "min_coverage_ratio": 0.8},
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL,
                        variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}',
                        extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                conn.execute(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, generation_id) "
                    "VALUES ('run', 'candidate', ?, 'gen0_peer0', 0)",
                    (
                        '{"effort_ratio": 1.0, "coverage_ratio": 1.0, '
                        '"scored_complete": true, "source_result_path": "results/a.json", '
                        '"source_result_sha256": "same-sha"}',
                    ),
                )
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, extra, variant_name, "
                    "peer_id, generation_id) VALUES ('result', 'preliminary', ?, ?, "
                    "'candidate', 'gen0_peer0', 0)",
                    (
                        '{"validation_only_result": true, "source_result_path": "results/a.json"}',
                        '{"source_result_sha256": "same-sha"}',
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger.mature_result_count(synchronize=False), 1)

    def test_mature_quorum_quarantines_only_immutable_preliminary_snapshot(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                cohort_size=1,
                maturity_policy={
                    "min_effort_ratio": 0.75,
                    "min_coverage_ratio": 0.8,
                    "preliminary_stage_labels": ["preliminary"],
                },
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL,
                        variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}',
                        extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                mature = (
                    '{"child_variant_id": "candidate", "effort_ratio": 1.0, '
                    '"coverage_ratio": 1.0, '
                    '"scored_complete": true, "source_result_path": "results/a.json", '
                    '"source_result_sha256": "same-sha"}'
                )
                conn.execute(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, generation_id) "
                    "VALUES ('run', 'candidate', ?, 'gen0_peer0', 0)",
                    (mature,),
                )
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, variant_name, peer_id, "
                    "generation_id) VALUES ('result', 'preliminary', ?, 'candidate', "
                    "'gen0_peer0', 0)",
                    (
                        '{"child_variant_id": "candidate", '
                        '"evidence_stage": "preliminary", "effort_ratio": 0.2, '
                        '"coverage_ratio": 0.2, "source_result_path": "results/a.json", '
                        '"source_result_sha256": "same-sha"}',
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger.mature_result_count(synchronize=False), 0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                cohort_size=1,
                maturity_policy={
                    "min_effort_ratio": 0.75,
                    "min_coverage_ratio": 0.8,
                    "preliminary_stage_labels": ["preliminary"],
                },
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL,
                        variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}',
                        extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                conn.execute(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, generation_id) "
                    "VALUES ('run', 'candidate', ?, 'gen0_peer0', 0)",
                    (
                        '{"effort_ratio": 1.0, "coverage_ratio": 1.0, '
                        '"scored_complete": true, '
                        '"source_result_path": "results/rewritten.json"}',
                    ),
                )
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, variant_name, peer_id, "
                    "generation_id) VALUES ('result', 'preliminary', ?, 'candidate', "
                    "'gen0_peer0', 0)",
                    (
                        '{"evidence_stage": "preliminary", "effort_ratio": 0.2, '
                        '"coverage_ratio": 0.2, '
                        '"source_result_path": "results/rewritten.json"}',
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger.mature_result_count(synchronize=False), 1)

    def test_unidentified_validation_does_not_tombstone_separate_mature_row(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                maturity_policy={"min_effort_ratio": 0.75, "min_coverage_ratio": 0.80},
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL,
                        variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}',
                        extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                conn.execute(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, generation_id) "
                    "VALUES ('complete', 'candidate_a', ?, 'gen0_peer0', 0)",
                    ('{"effort_ratio": 1.0, "coverage_ratio": 1.0, "scored_complete": true}',),
                )
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, variant_name, "
                    "peer_id, generation_id) VALUES ('result', 'preliminary', ?, "
                    "'candidate_a', 'gen0_peer0', 0)",
                    ('{"validation_only_result": true, "evidence_stage": "preliminary"}',),
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger._query_mature_state(), 1)

    def test_unknown_outer_stage_does_not_hide_configured_complete_aggregate(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
            compact_result_identity_container,
            evidence_maturity_snapshot,
            has_result_artifact_coordinates,
            same_result_artifact,
        )

        snapshot = evidence_maturity_snapshot(
            {
                "stage": "evaluation",
                "current_aggregate": {"evidence_stage": "complete_study"},
            },
            {"complete_stage_labels": ["complete_study"]},
        )

        self.assertTrue(snapshot["mature_enough"])
        self.assertEqual(snapshot["maturity_basis"], "task_configured_stage")
        identity = {
            "metrics": {
                "current_aggregate": {
                    "child_id": "child-a",
                    "source_result_path": "results/child-a/summary.json",
                    "source_result_sha256": "sha-a",
                    "ignored_measurement": 9.0,
                }
            }
        }
        compact = compact_result_identity_container(identity)
        self.assertEqual(
            compact["metrics"]["current_aggregate"],
            {
                "child_id": "child-a",
                "source_result_path": "results/child-a/summary.json",
                "source_result_sha256": "sha-a",
            },
        )
        self.assertTrue(has_result_artifact_coordinates(identity))
        self.assertFalse(same_result_artifact(None, ("results/child-a/summary.json", "sha-a")))
        self.assertFalse(evidence_maturity_snapshot({"complete_eval": 0})["mature_enough"])
        self.assertFalse(evidence_maturity_snapshot({"complete_eval": "failed"})["mature_enough"])

    def test_unattributed_quarantine_does_not_suppress_mature_artifact(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                maturity_policy={"min_effort_ratio": 0.75, "min_coverage_ratio": 0.80},
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL,
                        variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}',
                        extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                conn.execute(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, generation_id) "
                    "VALUES ('', '', ?, 'gen0_peer0', 0)",
                    (
                        '{"effort_ratio": 1.0, "coverage_ratio": 1.0, '
                        '"scored_complete": true, "source_result_path": '
                        '"results/anonymous/summary.json", "source_result_sha256": "same-sha"}',
                    ),
                )
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, peer_id, generation_id) "
                    "VALUES ('result', 'quarantine', ?, 'gen0_result_artifact', 0)",
                    (
                        '{"validation_only_result": true, "source_result_path": '
                        '"results/anonymous/summary.json", "source_result_sha256": "same-sha"}',
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger._query_mature_state(), 1)

    def test_variantless_same_artifact_is_not_counted_for_multiple_peers(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                maturity_policy={"min_effort_ratio": 0.75, "min_coverage_ratio": 0.80},
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL,
                        variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}',
                        extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                mature = (
                    '{"effort_ratio": 1.0, "coverage_ratio": 1.0, '
                    '"scored_complete": true, "source_result_path": '
                    '"results/shared/summary.json", "source_result_sha256": "shared-sha"}'
                )
                conn.executemany(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, generation_id) "
                    "VALUES ('', '', ?, ?, 0)",
                    [(mature, "gen0_peer0"), (mature, "gen0_peer1")],
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger.mature_result_count(synchronize=False), 1)
            self.assertEqual(len(trigger.mature_peer_ids(synchronize=False)), 1)

    def test_auto_result_name_inference_stays_within_known_cohort(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                cohort_size=2,
                maturity_policy={"min_effort_ratio": 0.75, "min_coverage_ratio": 0.80},
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL,
                        variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}',
                        extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                mature = (
                    '{"effort_ratio": 1.0, "coverage_ratio": 1.0, '
                    '"scored_complete": true, '
                    '"auto_materialized_from_result_artifact": true}'
                )
                conn.executemany(
                    "INSERT INTO findings(finding_type, title, metrics, variant_name, peer_id, "
                    "generation_id) VALUES ('result', ?, ?, ?, 'gen0_result_artifact', 0)",
                    [
                        ("gen0_peer1_candidate", mature, "gen0_peer1_candidate"),
                        ("gen0_peer9_candidate", mature, "gen0_peer9_candidate"),
                    ],
                )
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, variant_name, peer_id, "
                    "generation_id) VALUES ('insight', 'shared_alias', '{}', 'shared_alias', "
                    "'gen0_peer9', 0)"
                )
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, variant_name, peer_id, "
                    "generation_id) VALUES ('result', 'shared_alias', ?, 'shared_alias', "
                    "'gen0_result_artifact', 0)",
                    (mature,),
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger.mature_peer_ids(synchronize=False), {"gen0_peer1"})

    def test_unattributed_auto_result_counts_complete_immutable_result_not_peer(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL, variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL, title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}', extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, variant_name, peer_id, "
                    "generation_id) VALUES ('result', 'complete artifact', ?, 'candidate', "
                    "'gen0_result_artifact', 0)",
                    (
                        '{"scored_complete": true, "effort_ratio": 1.0, '
                        '"coverage_ratio": 1.0, '
                        '"auto_materialized_from_result_artifact": true, '
                        '"source_result_path": "results/candidate/summary.json", '
                        '"source_result_sha256": "complete-sha"}',
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger.mature_result_count(synchronize=False), 1)
            self.assertEqual(trigger.mature_peer_count(synchronize=False), 0)

    def test_current_aggregate_validation_marker_excludes_synthesis_maturity(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL, variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL, title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}', extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, variant_name, peer_id, "
                    "generation_id) VALUES ('result', 'validation signal', ?, 'candidate', "
                    "'gen0_peer0', 0)",
                    (
                        '{"current_aggregate": {"scored_complete": true, '
                        '"effort_ratio": 1.0, "coverage_ratio": 1.0, '
                        '"validation_only_result": true}}',
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger.mature_result_count(synchronize=False), 0)
            self.assertEqual(trigger.mature_peer_count(synchronize=False), 0)

    def test_started_peer_membership_preserves_noncontiguous_launched_ids(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                cohort_size=3,
                started_peer_ids=("gen0_peer0", "gen0_peer2", "gen0_peer3"),
                maturity_policy={"min_effort_ratio": 0.75, "min_coverage_ratio": 0.8},
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL,
                        variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}',
                        extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                mature = '{"effort_ratio": 1.0, "coverage_ratio": 1.0, "scored_complete": true}'
                conn.executemany(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, generation_id) "
                    "VALUES (?, ?, ?, ?, 0)",
                    [
                        ("r0", "a", mature, "gen0_peer0"),
                        ("r2", "b", mature, "gen0_peer2"),
                        ("r3", "c", mature, "gen0_peer3"),
                        ("r1", "not-started", mature, "gen0_peer1"),
                    ],
                )
                conn.executemany(
                    "INSERT INTO findings(finding_type, title, metrics, peer_id, generation_id) "
                    "VALUES ('result', ?, '{}', ?, 0)",
                    [
                        ("started", "gen0_peer3"),
                        ("not-started", "gen0_peer1"),
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(
                trigger.mature_peer_ids(synchronize=False),
                {"gen0_peer0", "gen0_peer2", "gen0_peer3"},
            )
            self.assertEqual(trigger._query_gen_state(), (2, 1))

            empty_membership = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                cohort_size=3,
                started_peer_ids=(),
                maturity_policy={"min_effort_ratio": 0.75, "min_coverage_ratio": 0.8},
            )
            self.assertEqual(empty_membership.mature_peer_ids(synchronize=False), set())
            self.assertEqual(empty_membership._query_gen_state(), (2, 0))

    def test_mature_quorum_staged_validation_does_not_tombstone_complete_artifact(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time() - 180,
                maturity_policy={
                    "min_effort_ratio": 0.75,
                    "min_coverage_ratio": 0.80,
                    "require_ratio_gate": True,
                },
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        id TEXT PRIMARY KEY,
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}',
                        extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                conn.executemany(
                    "INSERT INTO findings(id, finding_type, title, metrics, "
                    "variant_name, peer_id, generation_id) VALUES (?, 'result', ?, ?, ?, ?, 0)",
                    [
                        (
                            "preliminary",
                            "candidate_a preliminary",
                            '{"evidence_stage": "preliminary", "validation_only_result": true, '
                            '"source_result_path": "results/candidate_a/preliminary/summary.json", '
                            '"source_result_sha256": "preliminary-sha"}',
                            "candidate_a",
                            "gen0_peer0",
                        ),
                        (
                            "aligned",
                            "candidate_a aligned",
                            '{"evidence_stage": "aligned", "validation_only_result": true, '
                            '"source_result_path": "results/candidate_a/aligned/summary.json", '
                            '"source_result_sha256": "aligned-sha"}',
                            "candidate_a",
                            "gen0_peer0",
                        ),
                        (
                            "complete",
                            "candidate_a complete",
                            '{"evidence_stage": "complete", "effort_ratio": 1.0, '
                            '"coverage_ratio": 1.0, "scored_complete": true, '
                            '"auto_materialized_from_result_artifact": true, '
                            '"source_result_path": "results/candidate_a/complete/summary.json", '
                            '"source_result_sha256": "complete-sha"}',
                            "candidate_a",
                            "gen0_result_artifact",
                        ),
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger._query_mature_state(), 1)
            self.assertEqual(trigger.mature_result_count(synchronize=False), 1)

    def test_mature_result_count_deduplicates_same_artifact_variant_aliases(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                maturity_policy={"min_effort_ratio": 0.75, "min_coverage_ratio": 0.80},
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL,
                        variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}',
                        extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                mature = (
                    '{"effort_ratio": 1.0, "coverage_ratio": 1.0, '
                    '"scored_complete": true, "source_result_path": '
                    '"results/candidate/summary.json", "source_result_sha256": "one-sha"}'
                )
                conn.execute(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, generation_id) "
                    "VALUES ('run-a', 'canonical_candidate', ?, 'gen0_peer0', 0)",
                    (mature,),
                )
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, variant_name, peer_id, "
                    "generation_id) VALUES ('result', 'display alias', ?, 'display_alias', "
                    "'gen0_peer0', 0)",
                    (mature,),
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger.mature_result_count(synchronize=False), 1)
            self.assertEqual(trigger.mature_peer_count(synchronize=False), 1)

    def test_mature_result_count_preserves_independent_results_for_same_variant(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL, variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL, title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}', extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                conn.executemany(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, generation_id) "
                    "VALUES (?, 'candidate', ?, 'gen0_peer0', 0)",
                    [
                        (
                            "run-a",
                            '{"scored_complete": true, "source_result_path": '
                            '"results/a.json", "source_result_sha256": "sha-a"}',
                        ),
                        (
                            "run-b",
                            '{"scored_complete": true, "source_result_path": '
                            '"results/b.json", "source_result_sha256": "sha-b"}',
                        ),
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger.mature_result_count(synchronize=False), 2)
            self.assertEqual(trigger.mature_peer_count(synchronize=False), 1)

    def test_path_only_results_are_not_folded_before_identity_comparison(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL, variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL, title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}', extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                result = '{"scored_complete": true, "source_result_path": "results/reused.json"}'
                conn.executemany(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, generation_id) "
                    "VALUES (?, 'candidate', ?, 'gen0_peer0', 0)",
                    [("run-a", result), ("run-b", result)],
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger.mature_result_count(synchronize=False), 2)

    def test_coordinate_free_metric_runs_remain_distinct_results(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL, variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL, title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}', extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                conn.executemany(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, generation_id) "
                    "VALUES (?, ?, '{\"scored_complete\": true}', ?, 0)",
                    [
                        ("shared-run", "candidate-a", "gen0_peer0"),
                        ("shared-run", "candidate-a", "gen0_peer0"),
                        ("shared-run", "candidate-b", "gen0_peer1"),
                        ("other-run", "candidate-a", "gen0_peer0"),
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger.mature_result_count(synchronize=False), 3)
            self.assertEqual(trigger.mature_peer_count(synchronize=False), 2)

    def test_coordinate_free_preliminary_metric_does_not_tombstone_later_complete(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL, variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL, title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}', extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                conn.executemany(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, generation_id) "
                    "VALUES ('run-a', 'candidate', ?, 'gen0_peer0', 0)",
                    [
                        ('{"validation_only_result": true}',),
                        ('{"scored_complete": true, "effort_ratio": 1.0, "coverage_ratio": 1.0}',),
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger.mature_result_count(synchronize=False), 1)
            self.assertEqual(trigger.mature_peer_count(synchronize=False), 1)

    def test_coordinate_free_metric_does_not_duplicate_canonical_artifact(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL, variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL, title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}', extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                conn.execute(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, generation_id) "
                    "VALUES (?, 'candidate', '{\"scored_complete\": true}', "
                    "'gen0_peer0', 0)",
                    (root.name,),
                )
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, extra, variant_name, "
                    "peer_id, generation_id) VALUES ('result', 'canonical', "
                    "'{\"scored_complete\": true}', "
                    '\'{"canonical_variant_name": "candidate", '
                    '"source_result_path": "results/candidate/complete/summary.json", '
                    '"source_result_sha256": "sha-complete"}\', '
                    "'candidate', 'gen0_peer0', 0)"
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger.mature_result_count(synchronize=False), 1)
            self.assertEqual(trigger.mature_peer_count(synchronize=False), 1)

    def test_explicit_coordinate_free_run_remains_distinct_from_canonical_artifact(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        run_id TEXT NOT NULL, variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE findings (
                        finding_type TEXT NOT NULL, title TEXT NOT NULL,
                        metrics TEXT NOT NULL DEFAULT '{}', extra TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '', peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0
                    );
                """)
                conn.execute(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, generation_id) "
                    "VALUES ('experiment-a', 'candidate', '{\"scored_complete\": true}', "
                    "'gen0_peer0', 0)"
                )
                conn.execute(
                    "INSERT INTO findings(finding_type, title, metrics, extra, variant_name, "
                    "peer_id, generation_id) VALUES ('result', 'canonical', "
                    "'{\"scored_complete\": true}', "
                    '\'{"canonical_variant_name": "candidate", '
                    '"source_result_path": "results/candidate/complete/summary.json", '
                    '"source_result_sha256": "sha-complete"}\', '
                    "'candidate', 'gen0_peer0', 0)"
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger.mature_result_count(synchronize=False), 2)

    def test_local_store_nested_extra_artifact_aliases_deduplicate(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            with patch.dict("os.environ", {"LOCAL_STORE_DIR": str(root)}):
                local_store.init_db()
                for finding_id, variant_name in (
                    ("canonical", "candidate"),
                    ("alias", "candidate_alias"),
                ):
                    local_store.insert_finding(
                        {
                            "id": finding_id,
                            "finding_type": "result",
                            "title": finding_id,
                            "variant_name": variant_name,
                            "peer_id": "gen0_peer0",
                            "generation_id": 0,
                            "metrics": {
                                "scored_complete": True,
                                "effort_ratio": 1.0,
                                "coverage_ratio": 1.0,
                            },
                            "extra": {
                                "source_result_path": "results/candidate.json",
                                "source_result_sha256": "same-sha",
                            },
                        }
                    )
                trigger = synthesis_trigger.SynthesisTrigger(
                    run_dir=root,
                    gen_dir=gen_dir,
                    gen_id=0,
                    gen_start_time=time.time(),
                    local_store_dir=root,
                )

                self.assertEqual(trigger.mature_result_count(synchronize=False), 1)
                self.assertEqual(trigger.mature_peer_count(synchronize=False), 1)

    def test_local_store_nested_validation_marker_is_non_durable_but_weighted(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            with patch.dict("os.environ", {"LOCAL_STORE_DIR": str(root)}):
                local_store.init_db()
                local_store.insert_finding(
                    {
                        "id": "validation",
                        "finding_type": "result",
                        "title": "validation",
                        "variant_name": "candidate",
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "metrics": {"scored_complete": True, "tier": "T1"},
                        "extra": {
                            "validation_only_result": True,
                            "source_result_path": "results/candidate.json",
                            "source_result_sha256": "same-sha",
                        },
                    }
                )
                trigger = synthesis_trigger.SynthesisTrigger(
                    run_dir=root,
                    gen_dir=gen_dir,
                    gen_id=0,
                    gen_start_time=time.time(),
                    local_store_dir=root,
                    adaptive_policy={
                        "enabled": True,
                        "evidence_weights": {"T1": 0.25},
                    },
                )

                self.assertEqual(trigger.mature_result_count(synchronize=False), 0)
                self.assertEqual(trigger._query_adaptive_state(), (0.25, 0))

    def test_mature_quorum_starts_assessment_then_fires_after_work_drains(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            active_work = {"count": 1}
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time() - 180,
                min_findings=999,
                min_interval_minutes=1,
                max_interval_minutes=60,
                min_contributing_peers=4,
                maturity_policy={"min_effort_ratio": 0.75, "min_coverage_ratio": 0.80},
                mature_quorum_fraction=0.125,
                cohort_size=16,
                cohort_active_peers_callback=lambda: active_work["count"],
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        notes TEXT NOT NULL DEFAULT '',
                        step INTEGER NOT NULL DEFAULT 0,
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0,
                        timestamp TEXT NOT NULL
                    );
                    CREATE TABLE findings (
                        id TEXT PRIMARY KEY,
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        notes TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0,
                        timestamp TEXT NOT NULL,
                        extra TEXT NOT NULL DEFAULT '{}'
                    );
                """)
                conn.executemany(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, "
                    "generation_id, timestamp) VALUES (?, ?, ?, ?, 0, 'now')",
                    [
                        (
                            "r0",
                            "candidate_a",
                            '{"effort_ratio": 0.9, "coverage_ratio": 0.9, "scored_complete": true}',
                            "gen0_peer0",
                        ),
                        (
                            "r1",
                            "candidate_b",
                            '{"effort_ratio": 0.8, "coverage_ratio": 0.85, "scored_complete": true}',
                            "gen0_peer1",
                        ),
                        (
                            "r_bare",
                            "candidate_bare",
                            '{"effort_ratio": 0.95, "coverage_ratio": 0.95}',
                            "gen0_peer5",
                        ),
                        (
                            "r_not_scored",
                            "candidate_not_scored",
                            '{"effort_ratio": 0.95, "coverage_ratio": 0.95, "result_status": "not_scored_complete"}',
                            "gen0_peer6",
                        ),
                        (
                            "r_false_complete",
                            "candidate_false_complete",
                            '{"effort_ratio": 0.95, "coverage_ratio": 0.95, "tier_status": "scored_complete=false"}',
                            "gen0_peer7",
                        ),
                        (
                            "r_explicit_false",
                            "candidate_explicit_false",
                            '{"effort_ratio": 0.95, "coverage_ratio": 0.95, "scored_complete": false}',
                            "gen0_peer8",
                        ),
                        (
                            "r_summary_only",
                            "candidate_summary_only",
                            '{"effort_ratio": 0.95, "coverage_ratio": 0.95, "summary_only": true}',
                            "gen0_peer9",
                        ),
                        (
                            "r_unscored",
                            "candidate_unscored",
                            '{"effort_ratio": 0.95, "coverage_ratio": 0.95, "unscored_artifact": true}',
                            "gen0_peer10",
                        ),
                        (
                            "r_legacy_suspect",
                            "candidate_legacy_suspect",
                            '{"effort_ratio": 0.95, "coverage_ratio": 0.95, "suspect_fixed_weight_eval": true}',
                            "gen0_peer11",
                        ),
                        (
                            "r_scout_label_with_mature_ratios",
                            "candidate_scout_label_with_mature_ratios",
                            '{"effort_ratio": 0.95, "coverage_ratio": 0.95, "scout_only": true}',
                            "gen0_peer12",
                        ),
                        (
                            "r_validation_only_result",
                            "candidate_validation_only_result",
                            '{"effort_ratio": 0.95, "coverage_ratio": 0.95, "validation_only_result": true}',
                            "gen0_peer13",
                        ),
                        (
                            "r_late_boundary",
                            "candidate_late_boundary",
                            '{"effort_ratio": 0.95, "coverage_ratio": 0.95, "artifact_signal_status": "late_after_generation_boundary"}',
                            "gen0_peer14",
                        ),
                        (
                            "r_quarantined_signal",
                            "candidate_quarantined_signal",
                            '{"effort_ratio": 0.95, "coverage_ratio": 0.95, "late_result_policy": "quarantined_signal"}',
                            "gen0_peer15",
                        ),
                        (
                            "r2",
                            "candidate_c",
                            '{"effort_ratio": 0.95, "coverage_ratio": 0.95, "final_status": "failed"}',
                            "gen0_peer2",
                        ),
                        (
                            "r3",
                            "candidate_d",
                            '{"effort_ratio": 0.95, "coverage_ratio": 0.95, "final_status": "running"}',
                            "gen0_peer3",
                        ),
                        (
                            "r4",
                            "candidate_e",
                            '{"effort_ratio": 0.95, "coverage_ratio": 0.95, "incomplete_eval": true}',
                            "gen0_peer4",
                        ),
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            snap = trigger.evaluate()
            self.assertFalse(snap.fired)
            self.assertEqual(snap.reason, "assessment_draining")
            self.assertEqual(snap.mature_result_peers, 4)
            self.assertEqual(snap.required_mature_result_peers, 2)
            self.assertTrue((gen_dir / synthesis_trigger.CLOSING_SIGNAL_FILENAME).exists())
            self.assertLessEqual(trigger._seconds_until_next_timer_check(snap), 60.0)

            active_work["count"] = 0
            fired = trigger.evaluate()
            self.assertTrue(fired.fired)
            self.assertEqual(fired.reason, "mature_quorum")

    def test_mature_quorum_falls_back_to_findings_when_metrics_table_missing(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time() - 180,
                min_findings=999,
                min_interval_minutes=1,
                max_interval_minutes=60,
                min_contributing_peers=4,
                maturity_policy={"min_effort_ratio": 0.75, "min_coverage_ratio": 0.80},
                mature_quorum_fraction=0.5,
                cohort_size=2,
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE findings (
                        id TEXT PRIMARY KEY,
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        notes TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0,
                        timestamp TEXT NOT NULL,
                        extra TEXT NOT NULL DEFAULT '{}'
                    );
                """)
                conn.execute(
                    "INSERT INTO findings(id, finding_type, title, metrics, variant_name, "
                    "peer_id, generation_id, timestamp, extra) VALUES "
                    "(?, 'result', ?, ?, ?, ?, 0, 'now', ?)",
                    (
                        "f0",
                        "mature finding",
                        '{"score": 1.0}',
                        "candidate_a",
                        "gen0_peer0",
                        '{"effort_ratio": 0.9, "coverage_ratio": 0.9, "scored_complete": true}',
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger._query_mature_state(), 1)

    def test_required_ratio_gap_warns_once_without_discarding_result_signal(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time() - 180,
                maturity_policy={"require_ratio_gate": True},
                mature_quorum_fraction=0.5,
                cohort_size=2,
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE findings (
                        id TEXT PRIMARY KEY,
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        notes TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0,
                        timestamp TEXT NOT NULL,
                        extra TEXT NOT NULL DEFAULT '{}'
                    );
                """)
                conn.execute(
                    "INSERT INTO findings(id, finding_type, title, metrics, variant_name, "
                    "peer_id, generation_id, timestamp, extra) VALUES "
                    "(?, 'result', ?, ?, ?, ?, 0, 'now', ?)",
                    (
                        "f0",
                        "complete-looking result",
                        '{"score": 1.0, "scored_complete": true, "effort_ratio": 0.5}',
                        "candidate_a",
                        "gen0_peer0",
                        "{}",
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            with self.assertLogs(synthesis_trigger.logger, level="WARNING") as captured:
                self.assertEqual(trigger._query_mature_state(), 0)
                self.assertEqual(trigger._query_mature_state(), 0)

        warnings = [line for line in captured.output if "without required finite" in line]
        self.assertEqual(len(warnings), 1)
        self.assertIn("coverage_ratio", warnings[0])
        self.assertNotIn("coverage_ratio, effort_ratio", warnings[0])

    def test_required_ratio_warning_ignores_step_metrics_and_explicit_partial_results(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time() - 180,
                maturity_policy={"require_ratio_gate": True},
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        notes TEXT NOT NULL DEFAULT '',
                        step INTEGER NOT NULL DEFAULT 0,
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0,
                        timestamp TEXT NOT NULL
                    );
                    CREATE TABLE findings (
                        id TEXT PRIMARY KEY,
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        notes TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0,
                        timestamp TEXT NOT NULL,
                        extra TEXT NOT NULL DEFAULT '{}'
                    );
                """)
                conn.execute(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, "
                    "generation_id, timestamp) VALUES (?, ?, ?, ?, 0, 'now')",
                    ("r0", "candidate_a", '{"loss": 0.5}', "gen0_peer0"),
                )
                conn.execute(
                    "INSERT INTO findings(id, finding_type, title, metrics, variant_name, "
                    "peer_id, generation_id, timestamp, extra) VALUES "
                    "(?, 'result', ?, ?, ?, ?, 0, 'now', ?)",
                    (
                        "f0",
                        "partial result",
                        '{"score": 1.0, "scored_complete": false}',
                        "candidate_a",
                        "gen0_peer0",
                        "{}",
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(trigger._query_mature_state(), 0)

        self.assertFalse(trigger._warned_missing_required_ratios)

    def test_mature_quorum_draining_does_not_bypass_safety_cap(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time() - 360,
                min_findings=999,
                min_interval_minutes=1,
                max_interval_minutes=1,
                min_contributing_peers=4,
                maturity_policy={"min_effort_ratio": 0.75, "min_coverage_ratio": 0.80},
                mature_quorum_fraction=0.5,
                cohort_size=4,
                cohort_active_peers_callback=lambda: 1,
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        notes TEXT NOT NULL DEFAULT '',
                        step INTEGER NOT NULL DEFAULT 0,
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0,
                        timestamp TEXT NOT NULL
                    );
                    CREATE TABLE findings (
                        id TEXT PRIMARY KEY,
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        notes TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0,
                        timestamp TEXT NOT NULL,
                        extra TEXT NOT NULL DEFAULT '{}'
                    );
                """)
                conn.executemany(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, "
                    "generation_id, timestamp) VALUES (?, ?, ?, ?, 0, 'now')",
                    [
                        (
                            "r0",
                            "candidate_a",
                            '{"effort_ratio": 0.9, "coverage_ratio": 0.9, "scored_complete": true}',
                            "gen0_peer0",
                        ),
                        (
                            "r1",
                            "candidate_b",
                            '{"effort_ratio": 0.8, "coverage_ratio": 0.85, "scored_complete": true}',
                            "gen0_peer1",
                        ),
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            snap = trigger.evaluate()

        self.assertTrue(snap.fired)
        self.assertEqual(snap.reason, "safety_cap")
        self.assertEqual(snap.mature_result_peers, 2)
        self.assertEqual(snap.active_generation_work, 1)

    def test_closing_agent_drain_deadline_waits_for_protected_work_then_fires(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            active_work = {"count": 2}
            protected = {"count": 1}
            base_time = time.time()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=base_time - 600,
                min_findings=999,
                min_interval_minutes=1,
                max_interval_minutes=60,
                mature_quorum_fraction=0.5,
                cohort_size=2,
                cohort_active_peers_callback=lambda: active_work["count"],
                adaptive_policy={"drain_grace_minutes": 5},
            )

            query_patches = (
                patch.object(trigger, "_query_gen_state", return_value=(2, 2)),
                patch.object(trigger, "_query_adaptive_state", return_value=(0.0, 0)),
                patch.object(trigger, "_query_mature_state", return_value=1),
                patch.object(trigger, "mature_result_count", return_value=1),
                patch.object(
                    trigger,
                    "_active_protected_pid_count",
                    side_effect=lambda: protected["count"],
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler_client.freeze_generation"
                ),
            )
            with (
                query_patches[0],
                query_patches[1],
                query_patches[2],
                query_patches[3],
                query_patches[4],
                query_patches[5],
            ):
                with patch.object(synthesis_trigger.time, "time", return_value=base_time):
                    closing = trigger.evaluate()
                self.assertFalse(closing.fired)
                self.assertEqual(closing.reason, "draining_active_evals")
                self.assertIsNone(trigger._drain_started_at)

                protected["count"] = 0
                active_work["count"] = 1
                with patch.object(synthesis_trigger.time, "time", return_value=base_time + 60):
                    agent_drain = trigger.evaluate()
                self.assertFalse(agent_drain.fired)
                self.assertEqual(agent_drain.reason, "assessment_draining")

                with patch.object(synthesis_trigger.time, "time", return_value=base_time + 359):
                    still_draining = trigger.evaluate()
                self.assertFalse(still_draining.fired)

                with patch.object(synthesis_trigger.time, "time", return_value=base_time + 361):
                    fired = trigger.evaluate()

        self.assertTrue(fired.fired)
        self.assertEqual(fired.reason, "closing_agent_drain_deadline")
        self.assertEqual(fired.active_protected_pids, 0)
        self.assertEqual(fired.active_generation_work, 1)

    def test_mature_quorum_accepts_legacy_metrics_only_tier_evidence(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time() - 180,
                min_findings=999,
                min_interval_minutes=1,
                max_interval_minutes=60,
                min_contributing_peers=99,
                mature_quorum_fraction=0.5,
                cohort_size=4,
                cohort_active_peers_callback=lambda: 0,
                maturity_policy={"complete_stage_labels": ["T1", "T2"]},
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE findings (
                        id TEXT PRIMARY KEY,
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        notes TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0,
                        timestamp TEXT NOT NULL,
                        extra TEXT NOT NULL DEFAULT '{}'
                    );
                    CREATE TABLE metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        notes TEXT NOT NULL DEFAULT '',
                        step INTEGER NOT NULL DEFAULT 0,
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0,
                        timestamp TEXT NOT NULL
                    );
                """)
                conn.executemany(
                    "INSERT INTO metrics(run_id, variant_name, metrics, peer_id, "
                    "generation_id, timestamp) VALUES (?, ?, ?, ?, 0, 'now')",
                    [
                        (
                            "r0",
                            "candidate_a",
                            '{"tier_reached": "T1", "epochs": 500}',
                            "gen0_peer0",
                        ),
                        (
                            "r1",
                            "candidate_b",
                            '{"completed_tier": "T2", "epochs": 500}',
                            "gen0_peer1",
                        ),
                        (
                            "r2",
                            "candidate_failed",
                            '{"tier_reached": "T1", "final_status": "failed", "epochs": 500}',
                            "gen0_peer2",
                        ),
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            snap = trigger.evaluate()

        self.assertTrue(snap.fired)
        self.assertEqual(snap.reason, "mature_quorum")
        self.assertEqual(snap.mature_result_peers, 2)

    def test_info_density_label_wins_when_safety_cap_ties_success_condition(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time() - 180,
                min_findings=1,
                min_interval_minutes=1,
                max_interval_minutes=1,
                min_contributing_peers=1,
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE findings (
                        id TEXT PRIMARY KEY,
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        notes TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0,
                        timestamp TEXT NOT NULL,
                        extra TEXT NOT NULL DEFAULT '{}'
                    );
                """)
                conn.execute(
                    "INSERT INTO findings(id, finding_type, title, metrics, "
                    "variant_name, peer_id, generation_id, timestamp) "
                    "VALUES ('f1', 'result', 'result', '{}', 'candidate', 'gen0_peer0', 0, 'now')"
                )
                conn.commit()
            finally:
                conn.close()

            snap = trigger.evaluate()

        self.assertTrue(snap.fired)
        self.assertEqual(snap.reason, "info_density")

    def test_mature_quorum_enters_topup_assessment_instead_of_closing_early(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time() - 180,
                min_findings=2,
                min_interval_minutes=1,
                max_interval_minutes=60,
                min_contributing_peers=2,
                mature_quorum_fraction=0.5,
                cohort_size=4,
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE findings (
                        id TEXT PRIMARY KEY,
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        notes TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0,
                        timestamp TEXT NOT NULL,
                        extra TEXT NOT NULL DEFAULT '{}'
                    );
                """)
                conn.executemany(
                    "INSERT INTO findings(id, finding_type, title, metrics, "
                    "variant_name, peer_id, generation_id, timestamp) "
                    "VALUES (?, 'result', ?, '{}', ?, ?, 0, 'now')",
                    [
                        ("f1", "score-only result", "candidate_a", "gen0_peer0"),
                        ("f2", "score-only result", "candidate_b", "gen0_peer1"),
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            with patch.object(trigger, "begin_assessment", return_value=True):
                snap = trigger.evaluate()

        self.assertFalse(snap.fired)
        self.assertEqual(snap.reason, "assessment_mature_topup")
        self.assertEqual(snap.required_mature_result_peers, 2)
        self.assertEqual(snap.mature_result_peers, 0)

    def test_adaptive_policy_uses_explicit_smoke_marker_not_epoch_count(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),
                adaptive_policy={
                    "enabled": True,
                    "smoke_weight": 0.25,
                    "evidence_weights": {"T1": 1.0, "T2": 2.0, "T3": 4.0},
                },
            )

            self.assertEqual(
                trigger._evidence_units_from_payload(
                    {"tier": "T1", "requested_epochs": 5, "is_smoke_eval": True},
                    "normal title",
                ),
                0.25,
            )
            self.assertEqual(
                trigger._evidence_units_from_payload(
                    {"tier": "T1", "epochs": 500, "is_smoke_eval": "true"},
                    "normal title",
                ),
                0.25,
            )
            self.assertEqual(
                trigger._evidence_units_from_payload(
                    {"tier": "T1", "requested_epochs": 500, "is_smoke_eval": False},
                    "normal title",
                ),
                1.0,
            )

    def test_auto_evaluator_results_can_be_attributed_to_unique_peer_variant(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time() - 180,
                min_interval_minutes=1,
                adaptive_policy={
                    "enabled": True,
                    "min_evidence_units": 4.0,
                    "min_formal_result_peers": 4,
                    "evidence_weights": {"T1": 1.0, "T2": 2.0, "T3": 4.0},
                },
            )
            conn = sqlite3.connect(trigger.db_path)
            try:
                conn.executescript("""
                    CREATE TABLE metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        variant_name TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        notes TEXT NOT NULL DEFAULT '',
                        step INTEGER NOT NULL DEFAULT 0,
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0,
                        timestamp TEXT NOT NULL
                    );
                    CREATE TABLE findings (
                        id TEXT PRIMARY KEY,
                        finding_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL DEFAULT '',
                        metrics TEXT NOT NULL DEFAULT '{}',
                        variant_name TEXT NOT NULL DEFAULT '',
                        notes TEXT NOT NULL DEFAULT '',
                        peer_id TEXT NOT NULL DEFAULT '',
                        generation_id INTEGER NOT NULL DEFAULT 0,
                        timestamp TEXT NOT NULL,
                        extra TEXT NOT NULL DEFAULT '{}'
                    );
                """)
                rows = [
                    (
                        "h0",
                        "hypothesis",
                        "alpha_shape initial hypothesis",
                        "{}",
                        "alpha_shape",
                        "gen0_peer0",
                    ),
                    (
                        "h1",
                        "hypothesis",
                        "beta_shape initial hypothesis",
                        "{}",
                        "beta_shape",
                        "gen0_peer1",
                    ),
                    (
                        "a0",
                        "result",
                        "alpha_shape T1 on Alpaca US trading task",
                        '{"scored_complete": true, "auto_materialized_from_result_artifact": true}',
                        "alpha_shape",
                        "tiered_eval_auto",
                    ),
                    (
                        "a1",
                        "result",
                        "gen0_peer7_sparse T1 on Alpaca US trading task",
                        '{"scored_complete": true, "auto_materialized_from_result_artifact": true}',
                        "gen0_peer7_sparse",
                        "tiered_eval_auto",
                    ),
                    (
                        "amb",
                        "result",
                        "shared_variant T1 on Alpaca US trading task",
                        '{"scored_complete": true, "auto_materialized_from_result_artifact": true}',
                        "shared_variant",
                        "tiered_eval_auto",
                    ),
                    (
                        "amb_h0",
                        "hypothesis",
                        "shared_variant path a",
                        "{}",
                        "shared_variant",
                        "gen0_peer2",
                    ),
                    (
                        "amb_h1",
                        "hypothesis",
                        "shared_variant path b",
                        "{}",
                        "shared_variant",
                        "gen0_peer3",
                    ),
                ]
                conn.executemany(
                    "INSERT INTO findings(id, finding_type, title, metrics, "
                    "variant_name, peer_id, generation_id, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, 0, 'now')",
                    rows,
                )
                conn.commit()
            finally:
                conn.close()

            evidence_units, formal_peers = trigger._query_adaptive_state()
            self.assertEqual(evidence_units, 3.0)
            # alpha_shape is uniquely mapped to peer0; gen0_peer7_sparse is
            # self-identifying; shared_variant is ambiguous and is not counted.
            self.assertEqual(formal_peers, 2)

    def test_wait_until_fire_paths_are_event_driven_and_abortable(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=4,
                gen_start_time=time.time(),
                poll_interval_seconds=60,
            )

            abort = asyncio.Event()
            abort.set()
            aborted = asyncio.run(trigger.wait_until_fire(abort_event=abort))
            self.assertEqual(aborted.reason, "not_yet")

            fired_snap = synthesis_trigger.TriggerSnapshot(True, "info_density", 10, 30.0, 3)
            fired: list[synthesis_trigger.TriggerSnapshot] = []

            async def evaluate_fired():
                return fired_snap

            trigger.evaluate_async = evaluate_fired
            trigger.fire = lambda snap: fired.append(snap)
            self.assertIs(asyncio.run(trigger.wait_until_fire()), fired_snap)
            self.assertEqual(fired, [fired_snap])

            stop_trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=5,
                gen_start_time=time.time(),
                poll_interval_seconds=60,
            )

            async def evaluate_raises():
                raise RuntimeError("transient")

            async def fake_wait(*_args, **_kwargs):
                return SimpleNamespace(reason="stop")

            stop_trigger.evaluate_async = evaluate_raises
            with patch.object(synthesis_trigger, "wait_for_filesystem_event", fake_wait):
                stopped = asyncio.run(stop_trigger.wait_until_fire())
            self.assertEqual(stopped.reason, "not_yet")

            poll_trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=6,
                gen_start_time=time.time(),
            )

            async def fake_wait_until_fire(*, abort_event=None):
                return synthesis_trigger.TriggerSnapshot(False, "not_yet", 0, 0.0, 0)

            poll_trigger.wait_until_fire = fake_wait_until_fire
            self.assertEqual(asyncio.run(poll_trigger.poll_until_fire()).reason, "not_yet")

    def test_cohort_drained_fires_once_active_peers_reach_zero(self) -> None:
        """#148: when the callback reports 0 live peers after the warmup
        window, ``evaluate()`` fires with ``reason='cohort_drained'``.
        Without this exit, a fully crashed cohort sits idle until the
        240-min safety cap expires (the original orchestrator deadlock).
        """
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            # gen_start_time is in the past so the 60s warmup is satisfied
            # without sleeping in the test.
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time() - 120.0,
                min_findings=999,  # info_density unreachable
                min_interval_minutes=1.0,
                max_interval_minutes=999.0,  # safety_cap unreachable
                min_contributing_peers=999,
                cohort_active_peers_callback=lambda: 0,
                cohort_drain_warmup_seconds=60.0,
            )
            with patch.object(trigger, "_query_gen_state", return_value=(0, 0)):
                snap = trigger.evaluate()
            self.assertTrue(snap.fired)
            self.assertEqual(snap.reason, "cohort_drained")

    def test_cohort_drained_does_not_fire_while_peers_still_active(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time() - 120.0,
                min_findings=999,
                min_interval_minutes=1.0,
                max_interval_minutes=999.0,
                min_contributing_peers=999,
                cohort_active_peers_callback=lambda: 3,
                cohort_drain_warmup_seconds=60.0,
            )
            with patch.object(trigger, "_query_gen_state", return_value=(0, 0)):
                snap = trigger.evaluate()
            self.assertFalse(snap.fired)
            self.assertEqual(snap.reason, "not_yet")

    def test_cohort_drained_respects_warmup_window(self) -> None:
        """Before the warmup deadline, a zero-active-peers reading is
        ignored — guards against the cohort being marked drained in the
        race window where peers haven't yet picked up their work.
        """
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time(),  # just started → warmup unsatisfied
                min_findings=999,
                min_interval_minutes=1.0,
                max_interval_minutes=999.0,
                min_contributing_peers=999,
                cohort_active_peers_callback=lambda: 0,
                cohort_drain_warmup_seconds=60.0,
            )
            with patch.object(trigger, "_query_gen_state", return_value=(0, 0)):
                snap = trigger.evaluate()
            self.assertFalse(snap.fired)
            self.assertEqual(snap.reason, "not_yet")

    def test_cohort_drained_callback_exception_is_non_fatal(self) -> None:
        """A raising callback must not crash ``evaluate()`` — the trigger
        falls back to the legacy info_density / safety_cap path.
        """
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()

            def boom() -> int:
                raise RuntimeError("peer task list went away")

            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time() - 120.0,
                min_findings=999,
                min_interval_minutes=1.0,
                max_interval_minutes=999.0,
                min_contributing_peers=999,
                cohort_active_peers_callback=boom,
                cohort_drain_warmup_seconds=60.0,
            )
            with patch.object(trigger, "_query_gen_state", return_value=(0, 0)):
                snap = trigger.evaluate()
            self.assertFalse(snap.fired)
            self.assertEqual(snap.reason, "not_yet")

    def test_cohort_drained_callback_absent_keeps_legacy_behaviour(self) -> None:
        """Callers that don't pass the callback see exactly the old
        info_density / safety_cap behaviour — opt-in only.
        """
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time() - 120.0,
                min_findings=999,
                min_interval_minutes=1.0,
                max_interval_minutes=999.0,
                min_contributing_peers=999,
            )
            with patch.object(trigger, "_query_gen_state", return_value=(0, 0)):
                snap = trigger.evaluate()
            self.assertFalse(snap.fired)
            self.assertEqual(snap.reason, "not_yet")

    def test_helper_and_defensive_edges_are_result_preserving(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger

        self.assertIs(
            synthesis_trigger.AdaptiveSynthesisPolicy.from_raw(
                synthesis_trigger.AdaptiveSynthesisPolicy(enabled=True)
            ).enabled,
            True,
        )
        self.assertFalse(synthesis_trigger.AdaptiveSynthesisPolicy.from_raw("bad").enabled)
        policy = synthesis_trigger.AdaptiveSynthesisPolicy.from_raw(
            {
                "enabled": True,
                "min_evidence_units": "bad",
                "min_formal_result_peers": "2",
                "evidence_weights": {"T1": "1.5", "bad": object()},
                "smoke_weight": "0.1",
                "result_finding_weight": "2.0",
            }
        )
        self.assertTrue(policy.enabled)
        self.assertEqual(policy.min_evidence_units, 0.0)
        self.assertEqual(policy.min_formal_result_peers, 2)
        self.assertEqual(policy.evidence_weights, {"T1": 1.5})
        self.assertEqual(synthesis_trigger._float_or_default([], 3.0), 3.0)
        self.assertEqual(synthesis_trigger._float_or_default("bad", 4.0), 4.0)
        self.assertTrue(synthesis_trigger._truthy(" ON "))
        self.assertFalse(synthesis_trigger._truthy("off"))
        self.assertEqual(synthesis_trigger._json_object({"a": 1}), {"a": 1})
        self.assertEqual(synthesis_trigger._json_object("[]"), {})
        self.assertEqual(synthesis_trigger._json_object("{"), {})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            array_path = root / "array.json"
            array_path.write_text("[1, 2]", encoding="utf-8")
            self.assertEqual(synthesis_trigger._json_array_from_path(array_path), [1, 2])
            array_path.write_text("{}", encoding="utf-8")
            self.assertEqual(synthesis_trigger._json_array_from_path(array_path), [])
            self.assertEqual(synthesis_trigger._json_array_from_path(root / "missing.json"), [])

            trigger = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time() - 30.0,
                min_findings=2,
                min_interval_minutes=1.0,
                max_interval_minutes=5.0,
                min_contributing_peers=1,
                adaptive_policy=policy,
                poll_interval_seconds=5.0,
            )
            with patch.object(synthesis_trigger.Path, "exists", side_effect=OSError("stat")):
                self.assertEqual(trigger._query_gen_state(), (0, 0))
                self.assertEqual(trigger._query_adaptive_state(), (0.0, 0))
            with patch.object(synthesis_trigger.sqlite3, "connect", side_effect=MemoryError):
                self.assertEqual(trigger._query_gen_state(), (0, 0))
                self.assertEqual(trigger._query_adaptive_state(), (0.0, 0))

            self.assertEqual(
                trigger._variant_aliases(
                    "Alpha T1 on benchmark",
                    "baseline",
                    "child-smoke",
                    "",
                ),
                {"alpha_t1_on_benchmark", "child_smoke"},
            )
            self.assertEqual(
                trigger._evidence_units_from_payload(
                    {"tier": "T1", "is_smoke_eval": True},
                    "smoke run",
                ),
                0.1,
            )
            self.assertEqual(
                trigger._evidence_units_from_payload(
                    {"final_status": "passed_t1", "requested_epochs": 5},
                    "plain",
                ),
                0.0,
            )
            for generic_status in ("passed", "ok", "success", "completed", "passed_preview"):
                payload = {
                    "finding_type": "result",
                    "status": generic_status,
                    "score": 1.0,
                }
                self.assertEqual(trigger._evidence_units_from_payload(payload, "plain"), 0.0)
                self.assertFalse(trigger._payload_is_mature(payload, "plain"))
            self.assertEqual(
                trigger._evidence_units_from_payload(
                    {"finding_type": "result", "scored_complete": True},
                    "explicit completion",
                ),
                2.0,
            )
            self.assertEqual(
                trigger._evidence_units_from_payload(
                    {"finding_type": "result", "tier": "T1"},
                    "configured stage",
                ),
                1.5,
            )
            self.assertEqual(
                trigger._evidence_units_from_payload(
                    {"scored_complete": True, "partial_eval": True},
                    "explicitly incomplete",
                ),
                0.0,
            )

            protected_dir = root / "protected_pids"
            protected_dir.mkdir()
            (protected_dir / "bad.json").write_text("{", encoding="utf-8")
            (protected_dir / "entries.json").write_text(
                '[{"pid": "bad"}, {"pid": 1}, {"pid": 999999999}]',
                encoding="utf-8",
            )
            self.assertEqual(trigger._active_protected_pid_count(), 0)
            (protected_dir / "active.json").write_text(
                '[{"pid": 101, "peer_id": "gen1_peer0"}, {"pid": 102, "peer_id": "gen0-peer0"}]',
                encoding="utf-8",
            )
            from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

            with patch.object(protected_pids, "_is_pid_alive", return_value=True):
                self.assertEqual(trigger._active_protected_pid_count(), 1)
            with patch.object(synthesis_trigger.Path, "glob", side_effect=OSError("glob")):
                self.assertEqual(trigger._active_protected_pid_count(), 0)

            snap = synthesis_trigger.TriggerSnapshot(
                fired=False,
                reason="not_yet",
                findings_count=2,
                minutes_since_start=0.5,
                contributing_peers=1,
                evidence_units=2.0,
                formal_result_peers=1,
            )
            self.assertLessEqual(trigger._seconds_until_next_timer_check(snap), 30.0)
            mature_timer = synthesis_trigger.SynthesisTrigger(
                run_dir=root,
                gen_dir=gen_dir,
                gen_id=0,
                gen_start_time=time.time() - 30.0,
                min_findings=999,
                min_interval_minutes=1.0,
                max_interval_minutes=5.0,
                min_contributing_peers=99,
                mature_quorum_fraction=0.5,
                cohort_size=4,
                poll_interval_seconds=5.0,
            )
            mature_snap = synthesis_trigger.TriggerSnapshot(
                fired=False,
                reason="not_yet",
                findings_count=0,
                minutes_since_start=0.5,
                contributing_peers=0,
                mature_result_peers=2,
                mature_result_count=2,
                required_mature_result_peers=2,
            )
            self.assertLessEqual(mature_timer._seconds_until_next_timer_check(mature_snap), 30.0)
            draining = synthesis_trigger.TriggerSnapshot(
                fired=False,
                reason="draining_active_evals",
                findings_count=2,
                minutes_since_start=0.5,
                contributing_peers=1,
            )
            self.assertEqual(trigger._seconds_until_next_timer_check(draining), 60.0)
            self.assertFalse(trigger.closing)
            trigger.begin_closing(snap)
            self.assertTrue(trigger.closing)
            trigger.begin_closing(snap)
            trigger.fire(snap)
            trigger.fire(snap)
            trigger.write_postgen_marker(snap)
            self.assertTrue(synthesis_trigger.stop_signal_present(gen_dir))
            self.assertTrue(synthesis_trigger.closing_signal_present(gen_dir))

        with patch.object(Path, "exists", side_effect=OSError("stat")):
            self.assertFalse(synthesis_trigger.stop_signal_present(Path("x")))
            self.assertFalse(synthesis_trigger.closing_signal_present(Path("x")))


if __name__ == "__main__":
    unittest.main()


class SynthesisTriggerTimeoutClampingTest(unittest.TestCase):
    """Issue #179: synthesis_trigger max_interval clamped to per_generation_hours."""

    def _make_spec(self, per_gen_hours: int, max_interval_min: int) -> object:
        import os
        import tempfile

        import yaml

        spec_yaml = {
            "task_id": "test_clamp",
            "task_name": "Clamp Test",
            "description_file": "desc.md",
            "compute_budget": {},
            "generation_policy": {
                "per_generation_hours": per_gen_hours,
                "max_generations": 2,
                "cohort_size": 4,
            },
            "synthesis_trigger": {
                "max_interval_minutes": max_interval_min,
            },
        }
        d = tempfile.mkdtemp()
        spec_path = os.path.join(d, "task.yaml")
        desc_path = os.path.join(d, "desc.md")
        with open(spec_path, "w") as f:
            yaml.dump(spec_yaml, f)
        with open(desc_path, "w") as f:
            f.write("test")
        from praxist.task_spec import load_task_spec

        return load_task_spec(spec_path)

    def test_max_interval_exceeds_cap_is_clamped(self) -> None:
        """max_interval=240, per_gen=1h → clamped to 30 min."""
        ts = self._make_spec(per_gen_hours=1, max_interval_min=240)
        self.assertEqual(ts.synthesis_trigger.max_interval_minutes, 30)

    def test_max_interval_with_slack_not_clamped(self) -> None:
        """max_interval=60, per_gen=2h → untouched (slack > 30 min)."""
        ts = self._make_spec(per_gen_hours=2, max_interval_min=60)
        self.assertEqual(ts.synthesis_trigger.max_interval_minutes, 60)

    def test_max_interval_tight_slack_warns_only(self) -> None:
        """max_interval=50, per_gen=1h → tight but not clamped (< safety_cap)."""
        ts = self._make_spec(per_gen_hours=1, max_interval_min=50)
        self.assertEqual(ts.synthesis_trigger.max_interval_minutes, 50)

    def test_max_interval_equal_to_cap_warns_only(self) -> None:
        """max_interval=240, per_gen=4h → intentionally aligned and not clamped."""
        ts = self._make_spec(per_gen_hours=4, max_interval_min=240)
        self.assertEqual(ts.synthesis_trigger.max_interval_minutes, 240)

    def test_absurd_max_interval_with_long_peer_cap_stays_bounded(self) -> None:
        """per_gen=720h with max_interval=100000 must not become a 30-day cap."""
        ts = self._make_spec(per_gen_hours=720, max_interval_min=100000)
        self.assertEqual(ts.synthesis_trigger.max_interval_minutes, 240)

    def test_too_small_per_gen_raises(self) -> None:
        """per_gen=0h makes safety_cap=0 → ValueError (below min_usable_cap=15)."""
        with self.assertRaises(ValueError):
            self._make_spec(per_gen_hours=0, max_interval_min=240)


if __name__ == "__main__":
    unittest.main()
