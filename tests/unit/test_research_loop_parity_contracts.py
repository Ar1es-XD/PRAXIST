from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


class ResearchLoopParityContractsTest(unittest.TestCase):
    def test_parity_verifier_accepts_materialized_legacy_and_canonical_surfaces(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import parity

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            deliverables = Path(tmp) / "deliverables"
            _write_successful_parity_run(run_dir)
            for rel in (
                "README.md",
                "executive_summary.md",
                "data/run_summary.json",
                "data/frontier_manifest.json",
            ):
                path = deliverables / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("ok", encoding="utf-8")

            with patch.object(parity, "verify_run", return_value={"success": True}):
                report = parity.verify_research_loop_parity(
                    run_dir,
                    deliverables_dir=deliverables,
                    strict=True,
                    write_report=True,
                )

            self.assertTrue(report["success"], report)
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["warnings"], [])
            self.assertEqual(report["summary"]["legacy_findings"], 2)
            self.assertEqual(report["summary"]["legacy_graph_edges"], 1)
            self.assertTrue((run_dir / "research_loop_parity_report.json").exists())

    def test_parity_helpers_report_recoverable_warnings_and_structural_failures(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import parity

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "run.json").write_text("{}", encoding="utf-8")
            (run_dir / "bad.json").write_text("{bad", encoding="utf-8")
            (run_dir / "list.json").write_text("[]", encoding="utf-8")
            (run_dir / "gen_bad").mkdir()

            legacy = {
                "finding_ids": set(),
                "frontier_ids": set(),
                "research_memory_entry_count": 0,
                "graph_edge_count": 0,
                "graph_artifact_names": set(),
                "postgen_prompt_paths": [],
                "prompt_texts": {},
                "agenda_paths": [],
            }
            canonical = {
                "finding_ids": set(),
                "frontier_ids": set(),
                "research_memory": [],
                "graph_edges": [],
                "artifact_types": set(),
                "artifact_logical_paths": set(),
                "budget": [],
            }

            self.assertTrue(parity.ParityCheck("c", "fail", "m").failed)
            self.assertTrue(parity.ParityCheck("c", "warn", "m", severity="warning").warning)
            self.assertEqual(
                parity._check_replay({"success": False, "errors": ["e"]}).status, "fail"
            )
            self.assertEqual(parity._check_task_ref(run_dir).status, "warn")
            self.assertEqual(
                parity._check_legacy_findings_materialized(legacy, canonical).status,
                "warn",
            )
            self.assertEqual(parity._check_frontier_materialized(legacy, canonical).status, "warn")
            self.assertEqual(
                parity._check_research_memory_materialized(legacy, canonical).status,
                "warn",
            )
            self.assertEqual(
                parity._check_graph_edges_materialized(legacy, canonical).status, "warn"
            )
            self.assertEqual(
                parity._check_graph_artifacts_materialized(legacy, canonical).status,
                "warn",
            )
            self.assertEqual(
                parity._check_prompt_guidance_surfaces(legacy, canonical, strict=False).severity,
                "warning",
            )
            self.assertEqual(parity._check_panel_agenda_surface(legacy, strict=True).status, "fail")
            self.assertEqual(
                parity._check_operator_status_surface(
                    run_dir, legacy, canonical, strict=False
                ).status,
                "warn",
            )
            self.assertEqual(
                parity._check_resource_guard_usage(canonical, strict=True).status, "fail"
            )
            self.assertEqual(parity._check_deliverables(None, strict=False).status, "warn")
            self.assertEqual(
                parity._check_deliverables(Path(tmp) / "missing", strict=True).status, "fail"
            )
            self.assertEqual(parity._read_json(run_dir / "bad.json"), {})
            self.assertEqual(parity._read_json(run_dir / "list.json"), {})
            self.assertEqual(parity._read_text(run_dir / "missing.txt"), "")
            self.assertIsNone(parity._prompt_generation(run_dir / "gen_bad" / "p_prompt.md"))
            self.assertEqual(
                parity._frontier_entries(
                    {
                        "cumulative_top": [{"finding_id": "f1"}, "bad"],
                        "generations": {"0": [{"id": "f2"}, 1], "1": "bad"},
                    }
                ),
                [{"finding_id": "f1"}, {"id": "f2"}],
            )


def _write_successful_parity_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"task_ref": "task:toy"}), encoding="utf-8")
    (run_dir / "shared_findings").mkdir()
    (run_dir / "shared_findings" / "f1.json").write_text(json.dumps({"id": "f1"}), encoding="utf-8")

    conn = sqlite3.connect(run_dir / "shared_store.db")
    try:
        conn.execute("CREATE TABLE findings (id TEXT PRIMARY KEY)")
        conn.execute(
            "CREATE TABLE finding_edges (edge_id TEXT, src_finding_id TEXT, dst_finding_id TEXT, "
            "edge_type TEXT, confidence REAL, created_by TEXT, created_at TEXT, rationale TEXT, provenance TEXT)"
        )
        conn.execute("INSERT INTO findings VALUES ('f2')")
        conn.execute(
            "INSERT INTO finding_edges VALUES ('e1', 'f1', 'f2', 'supports', 0.7, 'graph', 'now', 'r', '{}')"
        )
        conn.commit()
    finally:
        conn.close()

    (run_dir / "frontier").mkdir()
    (run_dir / "frontier" / "frontier_manifest.json").write_text(
        json.dumps(
            {"cumulative_top": [{"finding_id": "f1"}], "generations": {"0": [{"id": "f2"}]}}
        ),
        encoding="utf-8",
    )
    prompt_dir = run_dir / "gen_1"
    prompt_dir.mkdir()
    (prompt_dir / "peer_prompt.md").write_text(
        "Graph-surfaced context mentions frontier f1 f2 and "
        "mcp__finding-graph-query__get_unlinked_recent_findings",
        encoding="utf-8",
    )

    ledgers = run_dir / "research_memory" / "ledgers"
    ledgers.mkdir(parents=True)
    (ledgers / "coverage.yaml").write_text(
        yaml.safe_dump({"entries": [{"id": "m1"}, {"id": "m2"}]}),
        encoding="utf-8",
    )
    graph_dir = run_dir / "graph"
    graph_dir.mkdir()
    for name in parity_graph_artifact_names():
        (graph_dir / name).write_text("graph", encoding="utf-8")

    (run_dir / "findings").mkdir()
    _write_jsonl(
        run_dir / "findings" / "findings.jsonl",
        [{"finding_id": "f1"}, {"finding_id": "f2"}],
    )
    _write_jsonl(
        run_dir / "findings" / "frontier.jsonl",
        [{"finding_id": "f1"}, {"finding_id": "f2"}],
    )
    (run_dir / "memory").mkdir()
    _write_jsonl(run_dir / "memory" / "research_memory.jsonl", [{"id": "m1"}, {"id": "m2"}])
    _write_jsonl(run_dir / "memory" / "graph_edges.jsonl", [{"graph_edge_id": "e1"}])
    _write_jsonl(
        run_dir / "budget_ledger.jsonl",
        [{"kind": "usage", "action_type": "gpu_slot"}],
    )
    _write_jsonl(
        run_dir / "artifact_index.jsonl",
        [
            {
                "artifact_type": "graph_materialized_artifact",
                "logical_path": f"graph/legacy/{name}",
            }
            for name in parity_graph_artifact_names()
        ],
    )

    (run_dir / "agendas").mkdir()
    (run_dir / "agendas" / "research_agenda_gen1.yaml").write_text(
        yaml.safe_dump({"peer_contracts": {"peer0": {"role": "explore"}}}),
        encoding="utf-8",
    )
    (run_dir / "orchestrator_status.final.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "task_id": "toy",
                "task_name": "Toy",
                "current_generation": 1,
                "generations_completed": 1,
                "findings_total": 2,
                "frontier_candidates": 2,
                "exit_condition": "completed",
            }
        ),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def parity_graph_artifact_names() -> tuple[str, ...]:
    return (
        "graph_health.json",
        "unlinked_recent_findings.json",
        "graph.html",
        "graph_live.html",
    )
