from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class FindingGraphVisualizationContractsTest(unittest.TestCase):
    def test_viz_metric_extraction_payload_and_html_rendering(self) -> None:
        from praxist.plugins.graph_maintainers.finding_graph_mvp import viz
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        self.assertEqual(viz._truncate("abcdef", 3), "abc…")
        self.assertEqual(viz._short_title("", 5), "(untitled)")
        self.assertGreater(viz._node_size(10), viz._node_size(0))
        self.assertGreater(viz._edge_width(1.0), viz._edge_width(0.55))
        self.assertTrue(viz._hex_to_rgba("#009E73", 0.5).startswith("rgba("))
        self.assertEqual(viz._truncate(None, 3), "")

        task_spec = {
            "evaluation": {
                "primary_metric": "loss",
                "direction": "minimize",
                "aux_metrics": ["gap"],
                "anchor_metrics": [{"name": "gap", "direction": "minimize"}],
            },
            "baselines": [{"name": "base", "expected_acc": 0.5}],
        }
        self.assertEqual(viz._task_primary_metric(task_spec), "loss")
        self.assertEqual(viz._task_direction(task_spec), "minimize")
        self.assertEqual(viz._task_secondary_metric(task_spec), "gap")
        self.assertEqual(viz._task_metric_direction(task_spec, "loss"), "minimize")
        self.assertEqual(viz._task_metric_direction(task_spec, "gap"), "minimize")
        self.assertIsNone(viz._task_metric_direction(task_spec, "undeclared"))
        self.assertIsNone(viz._task_secondary_metric({"evaluation": {"aux_metrics": []}}))
        self.assertEqual(viz._extract_primary_secondary({"metrics": []}), (None, None, None, None))
        self.assertEqual(
            viz._extract_primary_secondary(
                {"metrics": {"loss": 0.3, "gap": 0.1}},
                task_spec=task_spec,
            )[:2],
            (0.3, 0.1),
        )
        self.assertEqual(
            viz._extract_primary_secondary(
                {"metrics": {"test_accuracy": 0.77, "train_test_gap": 0.09}},
            ),
            (None, None, None, None),
        )
        fallback = viz._extract_primary_secondary(
            {
                "variant_name": "candidate",
                "metrics": {
                    "baseline_accuracy": 0.91,
                    "baseline_gap": 0.3,
                    "candidate_final_acc": 0.82,
                    "candidate_gap": 0.05,
                    "seed42_acc": 0.99,
                    "train_acc": 0.88,
                    "projected_accuracy": 0.95,
                    "ep150_acc": 0.97,
                    "candidate_acc_std": 0.01,
                },
            }
        )
        self.assertEqual(fallback, (None, None, None, None))
        ranks, pareto = viz._compute_top_and_pareto(
            [
                {"id": "a", "finding_type": "result", "metrics": {"loss": 0.4, "gap": 0.2}},
                {"id": "b", "finding_type": "result", "metrics": {"loss": 0.3, "gap": 0.3}},
                {"id": "c", "finding_type": "insight", "metrics": {"loss": 0.1, "gap": 0.1}},
                {"finding_type": "result", "metrics": {"loss": 0.1}},
            ],
            top_k=2,
            task_spec=task_spec,
        )
        self.assertEqual(ranks, {"b": 1, "a": 2})
        self.assertEqual(pareto, {"a", "b"})
        _, no_direction_pareto = viz._compute_top_and_pareto(
            [
                {"id": "a", "finding_type": "result", "metrics": {"loss": 0.4, "gap": 0.2}},
                {"id": "b", "finding_type": "result", "metrics": {"loss": 0.3, "gap": 0.3}},
            ],
            task_spec={
                "evaluation": {
                    "primary_metric": "loss",
                    "direction": "minimize",
                    "aux_metrics": ["gap"],
                }
            },
        )
        self.assertEqual(no_direction_pareto, set())

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"LOCAL_STORE_DIR": tmp}, clear=False),
        ):
            run_dir = Path(tmp)
            (run_dir / "task_spec.yaml").write_text(
                """
evaluation:
  primary_metric: loss
  direction: minimize
  aux_metrics: [gap]
baselines:
  - name: base
    expected_acc: 0.5
""",
                encoding="utf-8",
            )
            local_store.init_db()
            local_store.insert_finding(
                {
                    "id": "a",
                    "finding_type": "result",
                    "title": "A",
                    "content": "A <tag>",
                    "metrics": {"loss": 0.3, "gap": 0.1},
                    "variant_name": "A",
                    "peer_id": "p1",
                    "generation_id": 0,
                    "timestamp": "2026-05-12T00:00:00",
                }
            )
            local_store.insert_finding(
                {
                    "id": "b",
                    "finding_type": "hypothesis",
                    "title": "B",
                    "content": "B",
                    "metrics": {},
                    "variant_name": "B",
                    "peer_id": "p2",
                    "generation_id": 0,
                    "timestamp": "2026-05-12T00:00:01",
                }
            )
            local_store.insert_finding(
                {
                    "id": "c",
                    "finding_type": "error",
                    "title": "C",
                    "content": "C",
                    "metrics": {},
                    "variant_name": "C",
                    "peer_id": "p3",
                    "generation_id": 0,
                    "timestamp": "2026-05-12T00:00:02",
                }
            )
            local_store.insert_edges_batch(
                [
                    {
                        "edge_id": "e1",
                        "src_finding_id": "a",
                        "dst_finding_id": "b",
                        "edge_type": "supports",
                        "confidence": 0.9,
                        "created_by": "test",
                        "rationale": "<support>",
                    },
                    {
                        "edge_id": "e_orphan",
                        "src_finding_id": "a",
                        "dst_finding_id": "missing",
                        "edge_type": "untyped",
                        "confidence": 0.2,
                        "created_by": "test",
                        "rationale": "orphan",
                    },
                ]
            )
            payload = viz.build_viz_payload()
            self.assertEqual(payload["meta"]["primary_key"], "loss")
            self.assertEqual(payload["meta"]["num_findings"], 3)
            self.assertEqual(len(payload["anchors"]), 3)
            self.assertTrue(payload["leash_edges"])
            self.assertFalse(any(edge["id"] == "e_orphan" for edge in payload["edges"]))
            self.assertEqual(payload["edges"][0]["edge_type"], "supports")
            out = run_dir / "graph" / "graph.html"
            with patch.object(viz, "_load_vis_assets", return_value={"js": None, "css": None}):
                rendered = viz.render_graph_html(out, payload)
            self.assertEqual(rendered, out)
            html = out.read_text(encoding="utf-8")
            self.assertIn("Finding Graph", html)
            self.assertIn("<\\/script>", json.dumps({"x": "</script>"}).replace("</", "<\\/"))

    def test_viz_asset_cache_contracts_and_atomic_render_failures(self) -> None:
        from praxist.plugins.graph_maintainers.finding_graph_mvp import viz

        class FakeResponse:
            def __init__(self, data: bytes) -> None:
                self.data = data

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self) -> bytes:
                return self.data

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / "asset.js"
            with (
                patch.dict(viz._MIN_SIZES, {dest.name: 10}, clear=False),
                patch.object(viz.urllib.request, "urlopen", return_value=FakeResponse(b"x")),
                self.assertRaises(OSError),
            ):
                viz._download("https://example.invalid/asset.js", dest)

            with (
                patch.dict(viz._MIN_SIZES, {dest.name: 2}, clear=False),
                patch.object(viz.urllib.request, "urlopen", return_value=FakeResponse(b"okay")),
            ):
                viz._download("https://example.invalid/asset.js", dest)
            self.assertEqual(dest.read_bytes(), b"okay")
            self.assertTrue(viz._asset_is_healthy(dest))
            self.assertFalse(viz._asset_is_healthy(root / "missing.js"))

            failing_dest = root / "replace_fails.js"
            with (
                patch.dict(viz._MIN_SIZES, {failing_dest.name: 2}, clear=False),
                patch.object(viz.urllib.request, "urlopen", return_value=FakeResponse(b"okay")),
                patch.object(viz.os, "replace", side_effect=OSError("replace failed")),
                self.assertRaises(OSError),
            ):
                viz._download("https://example.invalid/asset.js", failing_dest)
            self.assertFalse(any(path.name.endswith(".tmp") for path in root.iterdir()))

            with patch.object(Path, "stat", side_effect=OSError("stat failed")):
                self.assertFalse(viz._asset_is_healthy(dest))

            cache = root / "cache"
            cache.mkdir()
            js_path = cache / "vis-network-9.1.9.min.js"
            css_path = cache / "vis-network-9.1.9.min.css"
            js_path.write_text("stale", encoding="utf-8")
            css_path.write_text("stale", encoding="utf-8")

            def fake_download(_url: str, path: Path) -> None:
                if path.suffix == ".css":
                    raise RuntimeError("network")
                path.write_text("cached-js", encoding="utf-8")

            with (
                patch.object(viz, "_ASSET_CACHE_DIR", cache),
                patch.dict(viz._MIN_SIZES, {js_path.name: 100, css_path.name: 100}, clear=False),
                patch.object(viz, "_download", side_effect=fake_download),
            ):
                assets = viz._load_vis_assets()
            self.assertEqual(assets["js"], "cached-js")
            self.assertIsNone(assets["css"])

            js_path.write_text("x", encoding="utf-8")
            css_path.write_text("x", encoding="utf-8")
            with (
                patch.object(viz, "_ASSET_CACHE_DIR", cache),
                patch.object(viz, "_asset_is_healthy", return_value=True),
                patch.object(Path, "read_text", side_effect=[OSError("read failed"), "css-body"]),
            ):
                assets = viz._load_vis_assets()
            self.assertIsNone(assets["js"])
            self.assertEqual(assets["css"], "css-body")

            with patch.dict(os.environ, {}, clear=True):
                self.assertIsNone(viz._load_run_task_spec())
            with (
                patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root)}, clear=False),
                patch("builtins.open", side_effect=OSError("open failed")),
            ):
                (root / "task_spec.yaml").write_text("evaluation: {}", encoding="utf-8")
                self.assertIsNone(viz._load_run_task_spec())

            out = root / "graph.html"
            payload = {
                "nodes": [],
                "edges": [],
                "anchors": [],
                "leash_edges": [],
                "leaderboard": [],
                "meta": {"num_findings": 0, "num_edges": 0, "linked_finding_ratio": 0.0},
            }
            with (
                patch.object(
                    viz, "_load_vis_assets", return_value={"js": "var vis = {};", "css": ""}
                ),
                patch.object(viz.os, "replace", side_effect=OSError("replace failed")),
                self.assertRaises(OSError),
            ):
                viz.render_graph_html(out, payload)


if __name__ == "__main__":
    unittest.main()
