from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from praxist.plugins.tools.evaluation_tools.adapter import _handle_wait_for_file_impl
from praxist.plugins.tools.frontier_tools.adapter import _handle_get_frontier
from praxist.plugins.tools.memory_tools.adapter import get_ledger_entry


class ToolBoundaryAdversarialContracts(unittest.TestCase):
    def test_wait_for_file_rejects_dangerous_roots_before_reading(self) -> None:
        async def _run() -> None:
            with patch.dict("os.environ", {"LOCAL_STORE_DIR": "/"}, clear=False):
                result = await _handle_wait_for_file_impl(
                    {
                        "path": "/etc/passwd",
                        "timeout_seconds": 1,
                        "poll_interval_seconds": 2,
                        "min_bytes": 1,
                    }
                )
            self.assertTrue(result.get("is_error"))
            payload = json.loads(result["content"][0]["text"])
            self.assertIn("outside allowed roots", payload["error"])

        asyncio.run(_run())

    def test_memory_ledger_name_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.joinpath("research_memory", "ledgers").mkdir(parents=True)
            outside_ledger = run_dir / "research_memory" / "outside.yaml"
            outside_ledger.write_text(
                yaml.safe_dump(
                    {
                        "ledger_name": "outside",
                        "schema_version": "1.0",
                        "entries": [
                            {
                                "id": "secret",
                                "created_by": "test",
                                "data": {"value": "should_not_be_read"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = get_ledger_entry(run_dir, "../outside", "secret")
            self.assertIn(
                "error",
                result,
                "memory ledger tool must whitelist ledger names instead of following path segments",
            )

    def test_frontier_tool_handles_malformed_manifest_without_exception(self) -> None:
        async def _run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                frontier_dir = Path(tmp) / "frontier"
                frontier_dir.mkdir()
                frontier_dir.joinpath("frontier_manifest.json").write_text(
                    json.dumps(
                        {
                            "metric_direction": "maximize",
                            "generations": {
                                "not_a_generation": [
                                    {"metric_value": "not-a-number", "variant_name": "bad"}
                                ],
                                "1": [{"metric_value": None, "variant_name": "missing_metric"}],
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                with patch.dict(os.environ, {"FRONTIER_DIR": str(frontier_dir)}, clear=False):
                    try:
                        result = await _handle_get_frontier(
                            {"up_to_generation": "bad", "top_k": "-1"}
                        )
                    except Exception as exc:  # noqa: BLE001
                        self.fail(
                            f"frontier tool should return a structured response, not raise {exc!r}"
                        )

            payload = json.loads(result["content"][0]["text"])
            self.assertIn("entries", payload)

        asyncio.run(_run())
