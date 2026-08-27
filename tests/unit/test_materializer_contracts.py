from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class LegacyMaterializerContractsTest(unittest.TestCase):
    def test_legacy_output_materializer_preserves_agent_and_imported_provenance(self) -> None:
        from praxist.plugins.workflow_stages.research_loop import (
            legacy_output_materializer as materializer,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "shared_findings").mkdir()
            finding = {
                "id": "f1",
                "finding_type": "result",
                "title": "Finding One",
                "content": "agent content",
                "metrics": {"score": 0.8, "note": "ignored"},
                "variant_name": "V1",
                "peer_id": "gen0_peer0",
                "generation_id": 0,
                "timestamp": "2026-05-12T00:00:00",
                "supersedes": ["old"],
            }
            (run_dir / "shared_findings" / "f1.json").write_text(
                json.dumps(finding),
                encoding="utf-8",
            )
            (run_dir / "shared_findings" / "bad.json").write_text("{bad", encoding="utf-8")
            (run_dir / "shared_findings" / "list.json").write_text("[1]", encoding="utf-8")
            (run_dir / "frontier").mkdir()
            (run_dir / "frontier" / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {
                            "bad": [{"finding_id": "bad"}],
                            "0": [
                                {
                                    "finding_id": "f1",
                                    "variant_name": "V1",
                                    "metric_name": "score",
                                    "metric_value": None,
                                    "metrics": {"score": 0.8},
                                    "promoted_at": "2026-05-12T00:01:00",
                                },
                                {
                                    "finding_id": "frontier_only",
                                    "variant_name": "V2",
                                    "metric_name": "score",
                                    "metric_value": 0.9,
                                    "peer_id": "gen0_peer1",
                                },
                                ["skip"],
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "trajectory.jsonl").write_text(
                json.dumps(
                    {
                        "event_id": "evt_agent",
                        "kind": "agent.run_finished",
                        "scope": {"agent_name": "gen0_peer0-session"},
                        "payload": {
                            "output_summary": {
                                "tool_uses": [
                                    {
                                        "tool": "mcp__evaluation-tools__share_finding",
                                        "input": {
                                            "peer_id": "gen0_peer0",
                                            "finding_type": "result",
                                            "title": "Finding One",
                                            "content": "agent content",
                                            "variant_name": "V1",
                                            "metrics": {"score": 0.8, "note": "ignored"},
                                        },
                                    }
                                ]
                            }
                        },
                    }
                )
                + "\n{bad\n",
                encoding="utf-8",
            )
            prepared = SimpleNamespace(
                run_dir=run_dir,
                run_id="run1",
                task_ref="task:demo",
                peer_role_ref="task_role:peer",
                task_spec=SimpleNamespace(
                    evaluation=SimpleNamespace(primary_metric="score"),
                ),
            )
            with patch.object(
                materializer,
                "materialize_legacy_c5_views",
                return_value={
                    "research_memory_record_count": 0,
                    "graph_edge_count": 0,
                    "graph_artifact_count": 0,
                },
            ):
                counts = materializer._materialize_legacy_outputs(prepared, {})
            self.assertEqual(counts["finding_count"], 3)
            self.assertEqual(counts["frontier_count"], 3)
            findings = [
                json.loads(line)
                for line in (run_dir / "findings" / "findings.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            frontier = [
                json.loads(line)
                for line in (run_dir / "findings" / "frontier.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            by_id = {item["finding_id"]: item for item in findings}
            self.assertEqual(by_id["f1"]["source_event_ids"], ["evt_agent"])
            self.assertEqual(by_id["f1"]["scores"], {"score": 0.8})
            self.assertEqual(by_id["f1"]["producer_ref"], "task_role:peer/gen0_peer0")
            self.assertEqual(by_id["frontier_only"]["provenance_quality"], "legacy_weak")
            self.assertEqual(frontier[0]["metric_value"], 0.8)
            self.assertTrue((run_dir / "artifact_index.jsonl").exists())

            self.assertEqual(
                materializer._collect_legacy_frontier_summary(
                    run_dir, {"frontier_summary": ["x", {"id": "direct"}]}
                ),
                [{"id": "direct"}],
            )
            self.assertEqual(
                materializer._collect_legacy_frontier_summary(run_dir / "missing", {}), []
            )
            self.assertEqual(materializer._finding_id({"finding_id": "x"}), "x")
            self.assertEqual(materializer._frontier_finding_id({"id": "x"}), "x")
            self.assertEqual(
                materializer._producer_ref(prepared, {"peer_role": "builder"}),
                "task_role:peer/builder",
            )
            self.assertEqual(materializer._producer_ref(prepared, {}), "task_role:peer/legacy")
            self.assertEqual(materializer._supersedes({"retry_of": "old"}), ["old"])
            self.assertFalse(
                materializer._share_finding_tool_input_matches(
                    {
                        "tool": "share_finding",
                        "input": {"peer_id": "gen0_peer0", "metrics": {"score": 0.1}},
                    },
                    finding,
                )
            )

    def test_legacy_output_materializer_writes_gems_jsonl(self) -> None:
        from praxist.plugins.workflow_stages.research_loop import (
            legacy_output_materializer as materializer,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "shared_findings").mkdir()
            (run_dir / "frontier").mkdir()
            (run_dir / "frontier" / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {},
                        "cumulative_top": [],
                        "lane_frontiers": {},
                        "gems": {
                            "entries": [
                                {
                                    "gem_finding_id": "gem_1",
                                    "variant_name": "alpha_gem",
                                    "frontier_lane": "alpha_incubator",
                                    "metric_name": "future_fitness",
                                    "metric_value": 1.2,
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            prepared = SimpleNamespace(
                run_dir=run_dir,
                run_id="run1",
                task_ref="task:demo",
                peer_role_ref="task_role:peer",
                task_spec=SimpleNamespace(
                    evaluation=SimpleNamespace(primary_metric="score"),
                ),
            )
            with patch.object(
                materializer,
                "materialize_legacy_c5_views",
                return_value={
                    "research_memory_record_count": 0,
                    "graph_edge_count": 0,
                    "graph_artifact_count": 0,
                },
            ):
                counts = materializer._materialize_legacy_outputs(prepared, {})

            self.assertEqual(counts["gems_count"], 1)
            gems = [
                json.loads(line)
                for line in (run_dir / "findings" / "gems.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(gems[0]["gem_finding_id"], "gem_1")
            self.assertEqual(gems[0]["variant_name"], "alpha_gem")


if __name__ == "__main__":
    unittest.main()
