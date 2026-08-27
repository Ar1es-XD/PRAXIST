from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from praxist.plugins.tools.evaluation_tools import adapter as evaluation_adapter
from praxist.plugins.tools.finding_graph_query import adapter as graph_query_adapter
from praxist.plugins.tools.frontier_tools import adapter as frontier_adapter
from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store


def _payload(result: dict) -> dict:
    return json.loads(result["content"][0]["text"])


def _read_full_tool_result(ref: str) -> dict:
    offset = 0
    chunks: list[str] = []
    while True:
        result = asyncio.run(
            evaluation_adapter._handle_read_tool_result(
                {"ref": ref, "offset": offset, "max_chars": 512}
            )
        )
        chunk = _payload(result)
        chunks.append(chunk["text"])
        next_offset = chunk["next_offset"]
        if next_offset is None:
            break
        offset = next_offset
    return json.loads("".join(chunks))


class CostOptimizationIntegrationTest(unittest.TestCase):
    def test_tool_outputs_are_bounded_and_full_results_are_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            root = Path(tmp_raw)
            run_dir = root / "run"
            frontier_dir = root / "frontier"
            run_dir.mkdir()
            frontier_dir.mkdir()
            env = {
                "PRAXIST_RUN_DIR": str(run_dir),
                "LOCAL_STORE_DIR": str(run_dir),
                "LOCAL_MODE": "true",
                "PRIMARY_METRIC": "mean_test_accuracy",
                "METRIC_DIRECTION": "maximize",
                "FRONTIER_DIR": str(frontier_dir),
            }
            with patch.dict(os.environ, env, clear=True):
                self._seed_local_store()
                self._seed_frontier_manifest(frontier_dir)

                leaderboard = _payload(
                    asyncio.run(
                        evaluation_adapter._handle_get_leaderboard({"top_k": 8, "inline_limit": 3})
                    )
                )
                self.assertEqual(len(leaderboard["entries"]), 3)
                self.assertTrue(leaderboard["_tool_output"]["truncated"])
                self.assertEqual(
                    leaderboard["_tool_output"]["truncated_lists"]["entries"]["total"],
                    8,
                )
                full_leaderboard = _read_full_tool_result(
                    leaderboard["_tool_output"]["full_result_ref"]
                )
                self.assertEqual(len(full_leaderboard["payload"]["entries"]), 8)
                self.assertEqual(full_leaderboard["payload"]["entries"][0]["variant_name"], "v7")

                frontier = _payload(
                    asyncio.run(
                        frontier_adapter._handle_get_frontier({"top_k": 6, "inline_limit": 2})
                    )
                )
                self.assertEqual(len(frontier["entries"]), 2)
                self.assertTrue(frontier["_tool_output"]["truncated"])
                full_frontier = _read_full_tool_result(frontier["_tool_output"]["full_result_ref"])
                self.assertEqual(len(full_frontier["payload"]["entries"]), 6)
                self.assertEqual(
                    full_frontier["payload"]["entries"][0]["variant_name"], "frontier_5"
                )

                neighbors = _payload(
                    asyncio.run(
                        graph_query_adapter._handle_get_finding_neighbors(
                            {"finding_id": "f0", "limit": 8, "inline_limit": 2}
                        )
                    )
                )
                self.assertEqual(len(neighbors["outgoing_edges"]), 2)
                self.assertEqual(len(neighbors["neighbor_findings"]), 2)
                self.assertTrue(neighbors["_tool_output"]["truncated"])
                full_neighbors = _read_full_tool_result(
                    neighbors["_tool_output"]["full_result_ref"]
                )
                self.assertEqual(len(full_neighbors["payload"]["outgoing_edges"]), 5)
                self.assertEqual(len(full_neighbors["payload"]["neighbor_findings"]), 5)
                self.assertIn(
                    "untrimmed neighbor content 4",
                    json.dumps(full_neighbors["payload"], default=str),
                )

                tool_result_files = sorted((run_dir / "tool_results").glob("*.json"))
                self.assertGreaterEqual(len(tool_result_files), 3)
                for path in tool_result_files:
                    path.resolve().relative_to((run_dir / "tool_results").resolve())

    @staticmethod
    def _seed_local_store() -> None:
        local_store.init_db()
        for i in range(8):
            local_store.insert_finding(
                {
                    "id": f"f{i}",
                    "finding_type": "result",
                    "title": f"Finding {i}",
                    "content": f"untrimmed neighbor content {i} " + ("x" * 1200),
                    "variant_name": f"v{i}",
                    "peer_id": f"gen0_peer{i % 2}",
                    "generation_id": 0,
                    "metrics": {
                        "mean_test_accuracy": 0.70 + (i / 1000),
                        "compute_overhead_ratio": 1.0 + (i / 10),
                        "tier": "T3",
                        "promotion_eligible": True,
                    },
                }
            )
        for i in range(1, 6):
            local_store.insert_edge(
                {
                    "src_finding_id": "f0",
                    "dst_finding_id": f"f{i}",
                    "edge_type": "supports",
                    "confidence": 0.95 - (i / 100),
                    "created_by": "integration_test",
                    "rationale": f"f0 supports f{i}",
                }
            )

    @staticmethod
    def _seed_frontier_manifest(frontier_dir: Path) -> None:
        entries = [
            {
                "generation_id": 0,
                "rank": i,
                "variant_name": f"frontier_{i}",
                "metric_value": 0.80 + (i / 100),
                "metrics": {
                    "mean_test_accuracy": 0.80 + (i / 100),
                    "complete_eval": True,
                    "evidence_stage": "full_T1",
                    "tier": "T1",
                },
                "finding_id": f"frontier-f{i}",
                "evidence_stage": "full_T1",
            }
            for i in range(6)
        ]
        (frontier_dir / "frontier_manifest.json").write_text(
            json.dumps({"metric_direction": "maximize", "generations": {"0": entries}}),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
