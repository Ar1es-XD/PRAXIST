from __future__ import annotations

import json
import os
import signal
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from praxist.core.replay import verify_run
from praxist.core.storage import output_ledger_hashes, read_jsonl, write_json
from praxist.plugins.workflow_stages.research_loop.startup import (
    finalize_research_loop_plugin_run,
    prepare_research_loop_plugin_run,
)
from praxist.run import _install_research_loop_signal_finalizer


class Step14C5MaterializerTest(unittest.TestCase):
    def test_legacy_memory_and_graph_outputs_are_materialized_to_c5_ledgers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_step14"
            prepared = _prepare(run_dir, root)
            _write_legacy_outputs(run_dir)

            finalize_research_loop_plugin_run(
                prepared,
                success=True,
                result={
                    "generations_completed": 1,
                    "run_dir": str(run_dir),
                    "exit_condition": "test",
                    "frontier_summary": [
                        {
                            "finding_id": "legacy_finding_1",
                            "variant_name": "step14 variant",
                            "metric_name": "mean_test_accuracy",
                            "metric_value": 0.91,
                        }
                    ],
                },
            )

            memory, memory_errors = read_jsonl(run_dir / "memory" / "research_memory.jsonl")
            graph_edges, graph_errors = read_jsonl(run_dir / "memory" / "graph_edges.jsonl")
            self.assertEqual(memory_errors, [])
            self.assertEqual(graph_errors, [])
            self.assertEqual(len(memory), 1)
            self.assertEqual(len(graph_edges), 1)
            self.assertEqual(memory[0]["source_finding_ids"], ["legacy_finding_1"])
            self.assertTrue(memory[0]["source_event_ids"])
            self.assertTrue(memory[0]["artifact_refs"])
            self.assertEqual(graph_edges[0]["src_finding_id"], "legacy_finding_1")
            self.assertEqual(graph_edges[0]["dst_finding_id"], "legacy_finding_2")
            self.assertTrue(graph_edges[0]["advisory"])

            artifact_index, artifact_errors = read_jsonl(run_dir / "artifact_index.jsonl")
            self.assertEqual(artifact_errors, [])
            artifact_types = {record["artifact_type"] for record in artifact_index}
            self.assertIn("research_memory_ledger", artifact_types)
            self.assertIn("graph_edges_snapshot", artifact_types)
            self.assertIn("graph_materialized_artifact", artifact_types)

            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["research_memory_records"], 1)
            self.assertEqual(summary["graph_edges"], 1)
            self.assertEqual(summary["graph_artifacts"], 3)
            self.assertIn("memory/research_memory.jsonl", summary["output_hashes"])
            self.assertIn("memory/graph_edges.jsonl", summary["output_hashes"])

            report = verify_run(run_dir)
            self.assertTrue(report["success"], report)
            self.assertEqual(report["summary"]["research_memory_records"], 1)
            self.assertEqual(report["summary"]["graph_edges"], 1)

    def test_failed_research_loop_still_materializes_available_legacy_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_step14_failed"
            prepared = _prepare(run_dir, root)
            _write_legacy_outputs(run_dir)

            finalize_research_loop_plugin_run(
                prepared,
                success=False,
                result={
                    "generations_completed": 1,
                    "run_dir": str(run_dir),
                    "exit_condition": "signal_sigterm",
                    "frontier_summary": [
                        {
                            "finding_id": "legacy_finding_1",
                            "variant_name": "step14 variant",
                            "metric_name": "mean_test_accuracy",
                            "metric_value": 0.91,
                        }
                    ],
                },
                error="terminated by SIGTERM",
                exit_code=143,
            )

            findings, finding_errors = read_jsonl(run_dir / "findings" / "findings.jsonl")
            frontier, frontier_errors = read_jsonl(run_dir / "findings" / "frontier.jsonl")
            memory, memory_errors = read_jsonl(run_dir / "memory" / "research_memory.jsonl")
            graph_edges, graph_errors = read_jsonl(run_dir / "memory" / "graph_edges.jsonl")
            self.assertEqual(finding_errors + frontier_errors + memory_errors + graph_errors, [])
            self.assertEqual(len(findings), 1)
            self.assertEqual(len(frontier), 1)
            self.assertEqual(len(memory), 1)
            self.assertEqual(len(graph_edges), 1)

            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["exit_code"], 143)
            self.assertEqual(summary["frontier_records"], 1)
            self.assertEqual(summary["research_memory_records"], 1)
            self.assertEqual(summary["graph_edges"], 1)
            self.assertIsNone(summary["materialization_error"])

    def test_failed_finalization_preserves_summary_when_materialization_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_step14_materializer_error"
            prepared = _prepare(run_dir, root)

            with patch(
                "praxist.plugins.workflow_stages.research_loop.startup._materialize_legacy_outputs",
                side_effect=RuntimeError("materializer boom"),
            ):
                finalize_research_loop_plugin_run(
                    prepared,
                    success=False,
                    result={
                        "generations_completed": 0,
                        "run_dir": str(run_dir),
                        "exit_condition": "signal_sigterm",
                    },
                    error="terminated by SIGTERM",
                    exit_code=143,
                )

            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["exit_code"], 143)
            self.assertEqual(summary["frontier_records"], 0)
            self.assertEqual(summary["research_memory_records"], 0)
            self.assertEqual(summary["graph_edges"], 0)
            self.assertEqual(summary["materialization_error"], "materializer boom")

            trajectory, errors = read_jsonl(run_dir / "trajectory.jsonl")
            self.assertEqual(errors, [])
            stage_events = [
                event
                for event in trajectory
                if event.get("kind") == "workflow.stage_failed"
                and event.get("scope", {}).get("stage_id") == "research_loop"
            ]
            self.assertEqual(len(stage_events), 1)
            self.assertEqual(
                stage_events[0]["payload"]["materialization_error"],
                "materializer boom",
            )

    def test_signal_finalizer_materializes_failed_run_before_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_step14_signal"
            prepared = _prepare(run_dir, root)
            _write_legacy_outputs(run_dir)

            handlers = {}
            signal_calls = []

            def fake_getsignal(sig):
                return f"previous-{sig.name}"

            def fake_signal(sig, handler):
                signal_calls.append((sig, handler))
                handlers[sig] = handler

            with (
                patch("signal.getsignal", side_effect=fake_getsignal),
                patch("signal.signal", side_effect=fake_signal),
                patch("os._exit", side_effect=SystemExit) as exit_mock,
            ):
                restore = _install_research_loop_signal_finalizer(
                    prepared,
                    finalize_research_loop_plugin_run,
                )
                sigterm_handler = handlers[signal.SIGTERM]
                restore()

                with self.assertRaises(SystemExit):
                    sigterm_handler(signal.SIGTERM, None)

            exit_mock.assert_called_once_with(143)
            self.assertIn((signal.SIGTERM, "previous-SIGTERM"), signal_calls)
            self.assertIn((signal.SIGINT, "previous-SIGINT"), signal_calls)

            memory, memory_errors = read_jsonl(run_dir / "memory" / "research_memory.jsonl")
            graph_edges, graph_errors = read_jsonl(run_dir / "memory" / "graph_edges.jsonl")
            self.assertEqual(memory_errors + graph_errors, [])
            self.assertEqual(len(memory), 1)
            self.assertEqual(len(graph_edges), 1)

            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["exit_code"], 143)
            self.assertEqual(summary["exit_condition"], "signal_sigterm")
            self.assertEqual(summary["error"], "terminated by SIGTERM")
            self.assertEqual(summary["research_memory_records"], 1)
            self.assertEqual(summary["graph_edges"], 1)
            self.assertIsNone(summary["materialization_error"])

    def test_replay_verifies_memory_and_graph_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_step14_bad_source"
            prepared = _prepare(run_dir, root)
            _write_legacy_outputs(run_dir)
            finalize_research_loop_plugin_run(
                prepared,
                success=True,
                result={
                    "generations_completed": 1,
                    "run_dir": str(run_dir),
                    "exit_condition": "test",
                    "frontier_summary": [{"finding_id": "legacy_finding_1"}],
                },
            )

            graph_edges, _ = read_jsonl(run_dir / "memory" / "graph_edges.jsonl")
            graph_edges[0]["source_event_ids"] = ["evt_missing"]
            (run_dir / "memory" / "graph_edges.jsonl").write_text(
                json.dumps(graph_edges[0], sort_keys=True) + "\n",
                encoding="utf-8",
            )
            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            summary["output_hashes"] = output_ledger_hashes(run_dir)
            write_json(run_dir / "run_summary.json", summary)
            trajectory, _ = read_jsonl(run_dir / "trajectory.jsonl")
            for event in trajectory:
                if event.get("kind") == "run.finalized":
                    event["payload"] = summary
            (run_dir / "trajectory.jsonl").write_text(
                "".join(json.dumps(event, sort_keys=True) + "\n" for event in trajectory),
                encoding="utf-8",
            )

            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(
                any(
                    "graph_edges:1 references unknown source_event_id: evt_missing" in error
                    for error in report["errors"]
                ),
                report,
            )


def _prepare(run_dir: Path, workspace: Path):
    with patch.dict(os.environ, {}, clear=False):
        return prepare_research_loop_plugin_run(
            task_project_path=Path.cwd() / "templates" / "tasks" / "toy_math",
            workspace=workspace,
            run_dir=run_dir,
            runtime_ref="agent_runtime:fake_runtime",
            model_provider_ref="model_provider:fake_provider",
            budget_policy_ref="budget_policy:fake_tiered",
            model="fake-deterministic",
            local_mode=True,
            frontier_strategy="auto",
            credential_profile="fake_multi_key",
            command="step14 test",
        )


def _write_legacy_outputs(run_dir: Path) -> None:
    shared_findings = run_dir / "shared_findings"
    shared_findings.mkdir(parents=True, exist_ok=True)
    (shared_findings / "legacy_finding_1.json").write_text(
        json.dumps(
            {
                "id": "legacy_finding_1",
                "title": "legacy result",
                "peer_id": "peer_1",
                "metrics": {"mean_test_accuracy": 0.91},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    ledger_dir = run_dir / "research_memory" / "ledgers"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    (ledger_dir / "claim_ledger.yaml").write_text(
        """
ledger_name: claim_ledger
entries:
  - id: claim_1
    created_at: "2026-05-10T00:00:00Z"
    data:
      confidence: 0.82
      source_ref:
        finding_id: legacy_finding_1
        kind: finding
""".lstrip(),
        encoding="utf-8",
    )

    conn = sqlite3.connect(run_dir / "shared_store.db")
    try:
        conn.execute(
            """CREATE TABLE finding_edges (
                   edge_id TEXT PRIMARY KEY,
                   src_finding_id TEXT,
                   dst_finding_id TEXT,
                   edge_type TEXT,
                   confidence REAL,
                   created_by TEXT,
                   created_at TEXT,
                   rationale TEXT,
                   provenance TEXT
               )"""
        )
        conn.execute(
            """INSERT INTO finding_edges
               (edge_id, src_finding_id, dst_finding_id, edge_type, confidence,
                created_by, created_at, rationale, provenance)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "edge_1",
                "legacy_finding_1",
                "legacy_finding_2",
                "supports",
                0.7,
                "legacy_graph",
                "2026-05-10T00:00:01Z",
                "shared baseline",
                json.dumps({"tool": "share_finding_link"}),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    graph_dir = run_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    (graph_dir / "graph_health.json").write_text(
        '{"node_count": 2, "edge_count": 1}\n', encoding="utf-8"
    )
    (graph_dir / "unlinked_recent_findings.json").write_text("[]\n", encoding="utf-8")
    (graph_dir / "graph.html").write_text("<html><body>graph</body></html>\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
