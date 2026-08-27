from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class FindingGraphEngineContractsTest(unittest.TestCase):
    def test_rule_engine_preserves_explicit_intent_and_conservative_edges(self) -> None:
        from praxist.plugins.graph_maintainers.finding_graph_mvp import engine

        old_uuid = "11111111-1111-4111-8111-111111111111"
        old_fs_id = "fs_" + "a" * 32
        findings = [
            {
                "id": old_uuid,
                "finding_type": "result",
                "title": "ALPHA-X baseline",
                "content": "initial result",
                "variant_name": "ALPHA-X alpha=0.1",
                "peer_id": "gen0_peer1",
                "timestamp": "2026-05-12T00:00:01",
            },
            {
                "id": old_fs_id,
                "finding_type": "result",
                "title": "ALPHA-X sibling",
                "content": "sibling result",
                "variant_name": "ALPHA-X",
                "peer_id": "gen0_peer2",
                "timestamp": "2026-05-12T00:00:02",
            },
            {
                "id": "plain-target",
                "finding_type": "insight",
                "title": "OMEGA-Y",
                "content": "unrelated but linkable",
                "variant_name": "OMEGA-Y",
                "peer_id": "gen0_peer3",
                "timestamp": "2026-05-12T00:00:03",
            },
            {
                "id": "new",
                "finding_type": "result",
                "title": "ALPHA-X followup",
                "content": f"failed to reproduce {old_uuid}; also see {old_fs_id}",
                "notes": "不一致",
                "extra": {"nested": ["失败", old_uuid]},
                "variant_name": "ALPHA-X alpha=0.3",
                "peer_id": "gen0_peer1",
                "timestamp": "2026-05-12T00:00:04",
                "links": [
                    {
                        "target_finding_id": old_fs_id,
                        "edge_type": "supports",
                        "rationale": "agent explicitly links support",
                    },
                    {
                        "target_finding_id": "plain-target",
                        "edge_type": "not-a-real-type",
                    },
                    "ignored",
                ],
            },
        ]

        self.assertEqual(
            engine._extract_referenced_ids(f"{old_uuid} {old_uuid.upper()} {old_fs_id}"),
            [old_uuid, old_fs_id],
        )
        self.assertFalse(
            engine._has_any_non_negated("not consistent with alpha-x", ("consistent with",))
        )
        self.assertTrue(engine._has_any_non_negated("validated alpha-x", ("validated",)))
        self.assertEqual(
            engine._normalize_title_tokens("try ALPHA-X and BETA-Z2"), {"ALPHA-X", "BETA-Z2"}
        )

        builder = engine.FindingGraphBuilder(findings)
        self.assertEqual(builder._norm_variant("ALPHA-X alpha=0.3"), "alpha-x")
        self.assertEqual([row["id"] for row in builder.chronological()][0], old_uuid)

        edges = builder.propose_edges_for(findings[-1])
        by_pair = {(edge["dst_finding_id"], edge["edge_type"]): edge for edge in edges}
        self.assertIn((old_uuid, "challenges"), by_pair)
        self.assertIn((old_fs_id, "supports"), by_pair)
        self.assertEqual(by_pair[(old_fs_id, "supports")]["created_by"], "agent_declared")
        self.assertIn((old_fs_id, "related_to"), by_pair)
        self.assertIn(("plain-target", "related_to"), by_pair)
        self.assertGreaterEqual(len(builder.build_all_edges()), len(edges))

        self.assertEqual(builder.propose_edges_for({"title": "no id"}), [])
        capped_links = [
            {"target_finding_id": old_uuid, "edge_type": "derived_from"}
            for _ in range(builder.MAX_LINKS_PER_FINDING + 5)
        ]
        capped = builder._rule2_explicit_links(
            {
                "id": "new",
                "links": capped_links,
                "timestamp": "2026-05-12T00:00:04",
            }
        )
        self.assertLessEqual(len(capped), builder.MAX_LINKS_PER_FINDING)

    def test_health_context_and_maintainer_surface_are_best_effort(self) -> None:
        from praxist.plugins.graph_maintainers.finding_graph_mvp import engine
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"LOCAL_STORE_DIR": tmp}, clear=False),
        ):
            run_dir = Path(tmp)
            local_store.init_db()
            local_store.insert_finding(
                {
                    "id": "anchor",
                    "finding_type": "result",
                    "title": "ANCHOR-X ``` fenced",
                    "content": "line one\nline two ~~~",
                    "variant_name": "ANCHOR-X",
                    "peer_id": "gen0_peer1",
                    "generation_id": 0,
                    "timestamp": "2026-05-12T00:00:01",
                }
            )
            local_store.insert_finding(
                {
                    "id": "neighbor",
                    "finding_type": "insight",
                    "title": "Neighbor",
                    "content": "useful sibling context",
                    "variant_name": "ANCHOR-X",
                    "peer_id": "gen0_peer2",
                    "generation_id": 0,
                    "timestamp": "2026-05-12T00:00:02",
                }
            )
            local_store.insert_edges_batch(
                [
                    {
                        "edge_id": "edge1",
                        "src_finding_id": "anchor",
                        "dst_finding_id": "neighbor",
                        "edge_type": "supports",
                        "confidence": 0.9,
                        "created_by": "rule_engine",
                        "rationale": "strong edge",
                    }
                ]
            )

            self.assertGreater(
                engine._score_edge_pair({"edge_type": "supports", "confidence": 0.9}, "a", "b"), 0.9
            )
            self.assertNotIn("\n", engine._snippet("a\n```b~~~c", 100))
            self.assertEqual(engine._previous_generation_peer_id("gen3_peer7"), "gen2_peer7")
            self.assertIsNone(engine._previous_generation_peer_id("gen0_peer7"))

            health = engine.compute_graph_health()
            self.assertEqual(health["num_findings"], 2)
            self.assertEqual(health["num_edges"], 1)
            self.assertEqual(health["edge_type_distribution"]["supports"], 1)

            health_path = run_dir / "graph"
            written = engine.write_graph_health(health_path)
            self.assertEqual(written["num_edges"], 1)
            self.assertTrue((health_path / "graph_health.json").exists())

            lineage_context = engine.build_session_start_graph_context("gen1_peer1")
            self.assertIn("lineage predecessor", lineage_context)
            self.assertIn("neighbor", lineage_context.lower())

            orientation_context = engine.build_session_start_graph_context("fresh_peer")
            self.assertIn("most-connected current findings", orientation_context)

            engine._record_session_failure("custom")
            self.assertGreaterEqual(
                engine.compute_graph_health()["session_context_failures"]["custom"], 1
            )
            engine.reset_graph_observability_state()
            self.assertEqual(
                engine.compute_graph_health()["maintainer"]["last_cycle_status"], "never"
            )

            maintainer = engine.FindingGraphMaintainer(run_dir)
            maintainer._cycle_lock.acquire()
            try:
                self.assertEqual(maintainer.sync_once(), {"status": "busy"})
                self.assertEqual(maintainer.sync_once_blocking(timeout=0.01), {"status": "timeout"})
            finally:
                maintainer._cycle_lock.release()

            with patch(
                "praxist.plugins.graph_maintainers.finding_graph_mvp.viz.render_graph_html",
                return_value=run_dir / "graph" / "graph.html",
            ):
                result = maintainer.sync_once()
            self.assertEqual(result["status"], "ok")
            self.assertGreaterEqual(result["proposed"], 1)

            maintainer.start()
            self.assertIsNotNone(maintainer._thread)
            maintainer.stop()
            self.assertIsNone(maintainer._thread)

    def test_graph_engine_error_edges_and_empty_contexts_are_best_effort(self) -> None:
        from praxist.plugins.graph_maintainers.finding_graph_mvp import engine
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        self.assertEqual(engine._extract_referenced_ids(""), [])
        self.assertFalse(engine._has_any_non_negated("unconfirmed alpha-x", ("confirmed",)))
        self.assertEqual(engine._normalize_title_tokens("lowercase"), set())

        old_uuid = "22222222-2222-4222-8222-222222222222"
        old = {
            "id": old_uuid,
            "finding_type": "result",
            "title": "BETA-X",
            "content": "old",
            "variant_name": "BETA-X",
            "peer_id": "gen0_peer0",
            "timestamp": "2026-05-12T00:00:01",
        }
        new = {
            "id": "newer",
            "finding_type": "result",
            "title": "BETA-X",
            "content": f"derived notes cite {old_uuid}",
            "variant_name": "BETA-X",
            "peer_id": "gen0_peer0",
            "timestamp": "2026-05-12T00:00:02",
            "extra": '{"bad"',
        }
        builder = engine.FindingGraphBuilder([old, new])
        derived = builder._rule1_explicit_id_ref(new)
        self.assertEqual(derived[0]["edge_type"], "derived_from")
        self.assertEqual(builder._rule1_explicit_id_ref({**new, "id": old_uuid}), [])
        self.assertEqual(
            builder._rule1_explicit_id_ref({**new, "timestamp": "2026-05-12T00:00:00"}), []
        )
        self.assertEqual(builder._rule2_explicit_links({**new, "links": "{bad"}), [])
        self.assertEqual(builder._rule2_explicit_links({**new, "links": {"target": old_uuid}}), [])
        self.assertEqual(
            builder._rule2_explicit_links(
                {
                    **new,
                    "links": [
                        {"target_finding_id": "missing", "edge_type": "supports"},
                        {"target_finding_id": "newer", "edge_type": "supports"},
                    ],
                }
            ),
            [],
        )
        self.assertEqual(builder._rule3_same_variant_time_series({**new, "variant_name": ""}), [])
        self.assertEqual(
            builder._rule3_same_variant_time_series({**new, "variant_name": "UNKNOWN"}),
            [],
        )
        updates = builder._rule3_same_variant_time_series(new)
        self.assertTrue(any(edge["edge_type"] == "updates" for edge in updates))
        self.assertEqual(builder._rule4_supports({**new, "content": "nothing useful"}), [])
        self.assertEqual(builder._rule5_challenges({**new, "content": "neutral"}), [])
        weak_only = builder._resolve(
            [
                {
                    "src_finding_id": "a",
                    "dst_finding_id": "a",
                    "edge_type": "related_to",
                    "confidence": 0.9,
                },
                {
                    "src_finding_id": "a",
                    "dst_finding_id": "b",
                    "edge_type": "related_to",
                    "confidence": 0.57,
                },
                {
                    "src_finding_id": "a",
                    "dst_finding_id": "b",
                    "edge_type": "related_to",
                    "confidence": 0.62,
                },
            ],
            "a",
        )
        self.assertEqual(weak_only[0]["confidence"], 0.62)

        with (
            patch.object(local_store, "count_findings", return_value=2),
            patch.object(local_store, "count_edges", return_value=1),
            patch.object(local_store, "edge_count_by_type", return_value={"related_to": 1}),
            patch.object(local_store, "get_unlinked_recent_findings", return_value=[]),
            patch.object(local_store, "_get_conn", side_effect=RuntimeError("db")),
        ):
            health = engine.compute_graph_health()
        self.assertEqual(health["linked_finding_ratio"], 0.0)

        with patch.object(local_store, "init_db", side_effect=RuntimeError("init")):
            self.assertEqual(engine.build_session_start_graph_context("gen1_peer0"), "")
        self.assertGreaterEqual(
            engine.compute_graph_health()["session_context_failures"]["init_db"], 1
        )

        with (
            patch.object(local_store, "init_db", return_value=None),
            patch.object(local_store, "get_findings", side_effect=RuntimeError("fetch")),
        ):
            self.assertEqual(engine.build_session_start_graph_context("gen1_peer0"), "")

        with (
            patch.object(local_store, "init_db", return_value=None),
            patch.object(
                local_store,
                "get_findings",
                side_effect=[[], RuntimeError("lineage")],
            ),
        ):
            self.assertEqual(engine.build_session_start_graph_context("gen1_peer0"), "")

        with (
            patch.object(local_store, "init_db", return_value=None),
            patch.object(local_store, "get_findings", return_value=[]),
            patch.object(local_store, "_get_conn", side_effect=RuntimeError("orientation")),
        ):
            self.assertEqual(engine.build_session_start_graph_context("fresh"), "")

        engine.reset_graph_observability_state()
        self.assertEqual(engine._report_maintainer_status(None)["last_cycle_status"], "never")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            maintainer = engine.FindingGraphMaintainer(run_dir)
            with patch.object(local_store, "init_db", side_effect=RuntimeError("init")):
                self.assertEqual(maintainer.sync_once()["status"], "error")
            with (
                patch.object(local_store, "init_db", return_value=None),
                patch.object(local_store, "get_all_findings", side_effect=RuntimeError("read")),
            ):
                self.assertEqual(maintainer.sync_once()["status"], "error")
            with (
                patch.object(local_store, "init_db", return_value=None),
                patch.object(local_store, "get_all_findings", return_value=[]),
            ):
                self.assertEqual(maintainer.sync_once()["status"], "empty")

            calls = 0

            def fake_sync_once():
                nonlocal calls
                calls += 1
                maintainer._stop_event.set()

            with (
                patch.object(maintainer, "sync_once", side_effect=fake_sync_once),
                patch.object(engine, "wait_for_filesystem_event", side_effect=RuntimeError("wait")),
            ):
                maintainer._run()
            self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
