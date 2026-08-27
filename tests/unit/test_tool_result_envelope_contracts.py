from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from praxist.plugins.tools import result_envelope


def _payload(result: dict) -> dict:
    return json.loads(result["content"][0]["text"])


class ToolResultEnvelopeContractsTest(unittest.TestCase):
    def test_envelope_truncates_inline_lists_and_preserves_full_result_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            payload = {"entries": [{"i": i} for i in range(5)], "note": "ok"}
            with patch.dict(os.environ, {"PRAXIST_RUN_DIR": str(run_dir)}, clear=True):
                wrapped = result_envelope.with_tool_output_envelope(
                    payload,
                    tool_name="demo_tool",
                    list_fields=("entries",),
                    inline_limit=2,
                )

                self.assertEqual([row["i"] for row in wrapped["entries"]], [0, 1])
                meta = wrapped["_tool_output"]
                self.assertTrue(meta["truncated"])
                self.assertEqual(meta["truncated_lists"]["entries"]["total"], 5)
                ref = meta["full_result_ref"]
                self.assertIsInstance(ref, str)
                first_chunk = result_envelope.read_tool_result_ref(ref, max_chars=80)
                self.assertEqual(first_chunk["offset"], 0)
                self.assertIsNotNone(first_chunk["next_offset"])
                second_chunk = result_envelope.read_tool_result_ref(
                    ref,
                    offset=first_chunk["next_offset"],
                    max_chars=50000,
                )
                self.assertIn('"i": 4', first_chunk["text"] + second_chunk["text"])

    def test_envelope_keeps_complete_inline_payload_when_full_artifact_unavailable(self) -> None:
        payload = {"entries": [{"i": i} for i in range(5)], "note": "ok"}
        with patch.dict(os.environ, {}, clear=True):
            wrapped = result_envelope.with_tool_output_envelope(
                payload,
                tool_name="demo_tool",
                list_fields=("entries",),
                inline_limit=2,
            )

        self.assertEqual([row["i"] for row in wrapped["entries"]], [0, 1, 2, 3, 4])
        meta = wrapped["_tool_output"]
        self.assertFalse(meta["truncated"])
        self.assertEqual(meta["truncated_lists"], {})
        self.assertIsNone(meta["full_result_ref"])
        self.assertEqual(meta["view"], "complete_inline")

    def test_read_tool_result_rejects_non_refs_and_path_traversal(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"PRAXIST_RUN_DIR": tmp}, clear=True),
        ):
            with self.assertRaises(ValueError):
                result_envelope.read_tool_result_ref("not-a-ref")
            with self.assertRaises(ValueError):
                result_envelope.read_tool_result_ref("tool_result:../escape.json")

        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaises(ValueError),
        ):
            result_envelope.read_tool_result_ref("tool_result:missing.json")

        with patch.dict(os.environ, {"PRAXIST_RUN_DIR": "/"}, clear=True):
            self.assertIsNone(result_envelope.active_run_dir())

        with (
            patch.dict(os.environ, {"PRAXIST_RUN_DIR": "/tmp/anywhere"}, clear=True),
            patch.object(result_envelope.Path, "resolve", side_effect=OSError("resolve")),
        ):
            self.assertIsNone(result_envelope.active_run_dir())

        self.assertEqual(result_envelope.coerce_inline_limit("bad", default=7), 7)
        self.assertEqual(result_envelope.coerce_inline_limit(None, default=3), 3)

    def test_evaluation_tool_reads_stored_result_chunks(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"PRAXIST_RUN_DIR": tmp}, clear=True),
        ):
            ref = result_envelope.store_tool_result("demo", {"value": "x" * 200})
            self.assertIsNotNone(ref)
            chunk = _payload(
                asyncio.run(
                    adapter._handle_read_tool_result({"ref": ref, "offset": 0, "max_chars": 40})
                )
            )
            self.assertEqual(chunk["returned_chars"], 40)
            self.assertIsNotNone(chunk["next_offset"])
            bad = asyncio.run(adapter._handle_read_tool_result({"ref": "tool_result:missing"}))
            self.assertTrue(bad["is_error"])

    def test_frontier_tool_keeps_business_fields_and_adds_full_lookup_metadata(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontier_dir = root / "frontier"
            frontier_dir.mkdir()
            generations = {
                "0": [
                    {
                        "generation_id": 0,
                        "rank": i,
                        "variant_name": f"v{i}",
                        "metric_value": i,
                        "evidence_stage": "full_T1",
                    }
                    for i in range(5)
                ]
            }
            frontier_dir.joinpath("frontier_manifest.json").write_text(
                json.dumps({"metric_direction": "maximize", "generations": generations}),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"FRONTIER_DIR": str(frontier_dir), "PRAXIST_RUN_DIR": str(root / "run")},
                clear=True,
            ):
                payload = _payload(
                    asyncio.run(adapter._handle_get_frontier({"top_k": 5, "inline_limit": 2}))
                )

            self.assertEqual(len(payload["entries"]), 2)
            self.assertEqual(payload["entries"][0]["variant_name"], "v4")
            self.assertTrue(payload["_tool_output"]["truncated"])
            self.assertIsInstance(payload["_tool_output"]["full_result_ref"], str)


if __name__ == "__main__":
    unittest.main()
