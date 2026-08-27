from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch


class LocalStoreContractsTest(unittest.TestCase):
    def test_delete_findings_supports_legacy_store_without_edge_table(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"LOCAL_STORE_DIR": tmp}, clear=False),
        ):
            db_path = os.path.join(tmp, "shared_store.db")
            with sqlite3.connect(db_path) as conn:
                conn.execute("CREATE TABLE findings (id TEXT PRIMARY KEY)")
                conn.execute("INSERT INTO findings(id) VALUES ('stale-auto-result')")

            self.assertEqual(local_store.delete_findings_by_ids(["stale-auto-result"]), 1)
            with sqlite3.connect(db_path) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0], 0)

    def test_db_path_falls_back_to_praxist_run_dir_when_store_env_is_missing(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"PRAXIST_RUN_DIR": tmp}, clear=True),
        ):
            self.assertEqual(local_store._get_db_path(), os.path.join(tmp, "shared_store.db"))

    def test_findings_metrics_edges_subgraph_and_pareto_contracts(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"LOCAL_STORE_DIR": tmp}, clear=False),
        ):
            status_path = os.path.join(tmp, "orchestrator_status.json")
            with open(status_path, "w", encoding="utf-8") as handle:
                json.dump({"updated_at": "old", "findings_total": 0}, handle)
            local_store.init_db()
            now = datetime.now()
            old = (now - timedelta(hours=12)).isoformat()
            recent = now.isoformat()
            rows = [
                {
                    "id": "a",
                    "finding_type": "result",
                    "title": "A",
                    "content": "A",
                    "metrics": {
                        "score": 0.8,
                        "cost": 2.0,
                        "tier": "T3",
                        "promotion_eligible": True,
                        "clean_promotion_eligible": True,
                    },
                    "variant_name": "A",
                    "peer_id": "p1",
                    "generation_id": 0,
                    "timestamp": old,
                    "details": {"note": "extra"},
                },
                {
                    "id": "b",
                    "finding_type": "result",
                    "title": "B",
                    "content": "B",
                    "metrics": {
                        "score": 0.7,
                        "cost": 1.0,
                        "tier": "T3",
                        "promotion_eligible": "yes",
                        "clean_promotion_eligible": True,
                    },
                    "variant_name": "B",
                    "peer_id": "p2",
                    "generation_id": 0,
                    "timestamp": recent,
                },
                {
                    "id": "c",
                    "finding_type": "insight",
                    "title": "C",
                    "content": "C",
                    "metrics": {
                        "score": 0.9,
                        "cost": 4.0,
                        "tier": "T3",
                        "promotion_eligible": 1,
                        "clean_promotion_eligible": True,
                    },
                    "variant_name": "C",
                    "peer_id": "p3",
                    "generation_id": 1,
                    "timestamp": recent,
                },
                {
                    "id": "d",
                    "finding_type": "result",
                    "title": "D",
                    "content": "D",
                    "metrics": {
                        "score": 0.1,
                        "cost": 0.1,
                        "tier": "T1",
                        "promotion_eligible": True,
                    },
                    "variant_name": "D",
                    "peer_id": "p4",
                    "generation_id": 1,
                    "timestamp": recent,
                },
                {
                    "id": "missing-axis",
                    "finding_type": "result",
                    "title": "Missing",
                    "content": "M",
                    "metrics": {
                        "score": 0.5,
                        "tier": "T3",
                        "promotion_eligible": True,
                        "clean_promotion_eligible": True,
                    },
                    "variant_name": "M",
                    "peer_id": "p5",
                    "generation_id": 1,
                    "timestamp": recent,
                },
            ]
            for row in rows:
                self.assertEqual(local_store.insert_finding(row), row["id"])
            self.assertEqual(local_store.count_findings(), 5)
            self.assertEqual(
                local_store.get_findings(generation_id=0, peer_id="p1")[0]["details"]["note"],
                "extra",
            )
            self.assertEqual(len(local_store.get_findings(finding_type="result", limit=2)), 2)

            metric_id = local_store.insert_metric(
                {
                    "run_id": "run",
                    "variant_name": "A",
                    "metrics": {"score": 0.8},
                    "step": 1,
                    "peer_id": "p1",
                    "generation_id": 0,
                }
            )
            self.assertIsInstance(metric_id, int)
            with open(status_path, encoding="utf-8") as handle:
                status = json.load(handle)
            self.assertEqual(status["findings_total"], 5)
            self.assertEqual(status["metrics_total"], 1)
            self.assertEqual(status["last_metric_row_id"], metric_id)
            self.assertEqual(
                local_store.get_leaderboard("score", "maximize", top_k=1)[0]["id"], "a"
            )
            self.assertEqual(
                local_store.get_leaderboard("score", "minimize", top_k=1)[0]["id"], "d"
            )

            edge = {
                "edge_id": "e1",
                "src_finding_id": "a",
                "dst_finding_id": "b",
                "edge_type": "supports",
                "confidence": 0.8,
                "created_by": "rule_engine",
                "rationale": "r",
                "provenance": {"p": 1},
            }
            self.assertEqual(local_store.insert_edge(edge), "e1")
            self.assertIsNone(local_store.insert_edge({**edge, "edge_id": "duplicate"}))
            with self.assertRaises(ValueError):
                local_store.insert_edge({**edge, "edge_type": "bad"})
            with self.assertRaises(ValueError):
                local_store.insert_edge({**edge, "confidence": 2.0})
            inserted = local_store.insert_edges_batch(
                [
                    {
                        "edge_id": "weak",
                        "src_finding_id": "a",
                        "dst_finding_id": "c",
                        "edge_type": "supports",
                        "confidence": 0.5,
                        "created_by": "rule_engine",
                    },
                    {
                        "edge_id": "strong",
                        "src_finding_id": "a",
                        "dst_finding_id": "c",
                        "edge_type": "derived_from",
                        "confidence": 0.4,
                        "created_by": "rule_engine",
                    },
                    {
                        "edge_id": "agent",
                        "src_finding_id": "a",
                        "dst_finding_id": "c",
                        "edge_type": "challenges",
                        "confidence": 0.1,
                        "created_by": "agent_declared",
                    },
                    {"edge_type": "bad", "confidence": 0.1},
                    {
                        "src_finding_id": "b",
                        "dst_finding_id": "c",
                        "edge_type": "related_to",
                        "confidence": 2.0,
                        "created_by": "rule_engine",
                    },
                ]
            )
            self.assertGreaterEqual(inserted, 2)
            self.assertGreaterEqual(local_store.count_edges(), 2)
            self.assertEqual(local_store.edge_count_by_type()["supports"], 1)
            self.assertTrue(
                local_store.get_edges_for_finding("a", direction="out", min_confidence=0.7)
            )
            self.assertTrue(
                local_store.get_edges_for_finding("b", direction="in", edge_types=["supports"])
            )
            subgraph = local_store.get_subgraph("a", max_depth=2, max_nodes=2)
            self.assertTrue(subgraph["truncated"])
            self.assertTrue(subgraph["nodes"])
            unlinked_ids = {row["id"] for row in local_store.get_unlinked_recent_findings(hours=24)}
            self.assertIn("missing-axis", unlinked_ids)

            self.assertTrue(
                local_store._pareto_dominates(
                    {"score": 2, "cost": 1},
                    {"score": 1, "cost": 2},
                    [
                        {"name": "score", "direction": "maximize"},
                        {"name": "cost", "direction": "minimize"},
                    ],
                )
            )
            pareto = local_store.get_pareto_leaderboard(
                "score",
                "maximize",
                [{"name": "cost", "direction": "minimize"}, {"name": "score"}],
                top_k_dominated=-1,
                requires_tier=True,
            )
            self.assertEqual({axis["name"] for axis in pareto["axes"]}, {"score", "cost"})
            self.assertEqual(pareto["n_total"], 3)
            self.assertEqual(pareto["dominated_top"], [])
            self.assertEqual(pareto["n_excluded_missing_axis"], {"cost": 1})
            self.assertIn("C", pareto["best_in"]["score"])
            self.assertIn("B", pareto["best_in"]["cost"])

            permissive = local_store.get_pareto_leaderboard(
                "score",
                "maximize",
                [],
                generation_id=1,
                requires_tier=False,
            )
            # In non-strict mode tier labels are opaque metadata, so T1/T2-style
            # names are not filtered by core. Only generic promotion flags apply.
            self.assertEqual(permissive["n_total"], 3)

    def test_pareto_keeps_sweep_children_and_excludes_scout_partial_rows(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"LOCAL_STORE_DIR": tmp}, clear=False),
        ):
            local_store.init_db()
            base = {
                "finding_type": "result",
                "title": "candidate",
                "content": "content",
                "variant_name": "bridge_l1_eff_n_sweep",
                "peer_id": "gen0_peer0",
                "generation_id": 0,
                "timestamp": datetime.now().isoformat(),
            }
            for finding in (
                {
                    **base,
                    "id": "child005",
                    "metrics": {
                        "score": 9.0,
                        "cost": 2.0,
                        "tier": "T3",
                        "promotion_eligible": True,
                        "clean_promotion_eligible": True,
                        "source_result_path": "results/bridge_l1_c005/tiered_eval_summary.json",
                    },
                },
                {
                    **base,
                    "id": "child025",
                    "metrics": {
                        "score": 8.0,
                        "cost": 1.0,
                        "tier": "T3",
                        "promotion_eligible": True,
                        "clean_promotion_eligible": True,
                        "source_result_path": "results/bridge_l1_c025/tiered_eval_summary.json",
                    },
                },
                {
                    **base,
                    "id": "scout-high",
                    "variant_name": "scout_high",
                    "metrics": {
                        "score": 99.0,
                        "cost": 0.1,
                        "tier": "T3",
                        "promotion_eligible": True,
                        "clean_promotion_eligible": True,
                    },
                    "extra": {"evidence_stage": "scout", "scout_only": True},
                },
                {
                    **base,
                    "id": "child005-frontier-key-duplicate",
                    "metrics": {
                        "score": 7.0,
                        "cost": 3.0,
                        "tier": "T3",
                        "promotion_eligible": True,
                        "clean_promotion_eligible": True,
                        "frontier_entity_key": "variant::bridge_l1_c005",
                    },
                },
            ):
                local_store.insert_finding(finding)

            pareto = local_store.get_pareto_leaderboard(
                "score",
                "maximize",
                [{"name": "cost", "direction": "minimize"}],
                requires_tier=True,
            )
            self.assertEqual(pareto["n_total"], 2)
            self.assertEqual(
                {row["metrics"]["source_result_path"] for row in pareto["pareto_front"]},
                {
                    "results/bridge_l1_c005/tiered_eval_summary.json",
                    "results/bridge_l1_c025/tiered_eval_summary.json",
                },
            )
            self.assertNotIn(
                "scout_high",
                {row["variant_name"] for row in pareto["pareto_front"]},
            )

    def test_local_store_conflict_resolution_and_pareto_edge_filters(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"LOCAL_STORE_DIR": tmp}, clear=False),
        ):
            local_store.init_db()
            with self.assertRaises(RuntimeError), local_store._get_conn() as conn:
                conn.execute(
                    "INSERT INTO findings (id, finding_type, title, timestamp) VALUES (?, ?, ?, ?)",
                    ("rollback", "result", "Rollback", datetime.now().isoformat()),
                )
                raise RuntimeError("rollback")
            self.assertEqual(local_store.count_findings(), 0)

            base_time = datetime.now().isoformat()
            for row in (
                {
                    "id": "a",
                    "finding_type": "result",
                    "title": "A",
                    "metrics": {
                        "score": 0.8,
                        "cost": 2.0,
                        "promotion_eligible": True,
                        "clean_promotion_eligible": True,
                    },
                    "variant_name": "A",
                    "generation_id": 0,
                    "timestamp": base_time,
                    "details": {"tier": "T3"},
                },
                {
                    "id": "b",
                    "finding_type": "result",
                    "title": "B",
                    "metrics": {"score": 0.9, "cost": 1.5},
                    "variant_name": "B",
                    "generation_id": 0,
                    "timestamp": base_time,
                    "tier": "T3",
                    "promotion_eligible": "promotable",
                    "clean_promotion_eligible": True,
                },
                {
                    "id": "c",
                    "finding_type": "insight",
                    "title": "C",
                    "metrics": {
                        "score": 0.9,
                        "cost": 1.5,
                        "tier": "T3",
                        "promotion_eligible": True,
                        "clean_promotion_eligible": True,
                    },
                    "variant_name": "C",
                    "generation_id": 0,
                    "timestamp": base_time,
                },
                {
                    "id": "c2",
                    "finding_type": "result",
                    "title": "C2",
                    "metrics": {
                        "score": 0.9,
                        "cost": 1.5,
                        "tier": "T3",
                        "promotion_eligible": True,
                        "clean_promotion_eligible": True,
                    },
                    "variant_name": "C",
                    "generation_id": 0,
                    "timestamp": base_time,
                },
                {
                    "id": "bad-tier",
                    "finding_type": "result",
                    "title": "Bad tier",
                    "metrics": {"score": 0.7, "cost": 1.0, "tier": 3, "promotion_eligible": True},
                    "variant_name": "BadTier",
                    "generation_id": 0,
                    "timestamp": base_time,
                },
                {
                    "id": "bad-promo",
                    "finding_type": "result",
                    "title": "Bad promo",
                    "metrics": {
                        "score": 0.7,
                        "cost": 1.0,
                        "tier": "T3",
                        "promotion_eligible": "maybe",
                    },
                    "variant_name": "BadPromo",
                    "generation_id": 0,
                    "timestamp": base_time,
                },
                {
                    "id": "anon",
                    "finding_type": "result",
                    "title": "Anon",
                    "metrics": {
                        "score": 0.6,
                        "cost": 0.5,
                        "tier": "T3",
                        "promotion_eligible": True,
                        "clean_promotion_eligible": True,
                    },
                    "variant_name": "",
                    "generation_id": 0,
                    "timestamp": base_time,
                },
                {
                    "id": "nan",
                    "finding_type": "result",
                    "title": "NaN",
                    "metrics": {
                        "score": math.nan,
                        "cost": 0.5,
                        "tier": "T3",
                        "promotion_eligible": True,
                        "clean_promotion_eligible": True,
                    },
                    "variant_name": "NaN",
                    "generation_id": 0,
                    "timestamp": base_time,
                },
                {
                    "id": "false-string",
                    "finding_type": "result",
                    "title": "False String",
                    "metrics": {
                        "score": 0.5,
                        "cost": 0.2,
                        "tier": "T3",
                        "promotion_eligible": "no",
                    },
                    "variant_name": "FalseString",
                    "generation_id": 0,
                    "timestamp": base_time,
                },
                {
                    "id": "false-int",
                    "finding_type": "result",
                    "title": "False Int",
                    "metrics": {"score": 0.5, "cost": 0.2, "tier": "T3", "promotion_eligible": 0},
                    "variant_name": "FalseInt",
                    "generation_id": 0,
                    "timestamp": base_time,
                },
            ):
                local_store.insert_finding(row)

            inserted = local_store.insert_edges_batch(
                [
                    {
                        "edge_id": "existing_strong",
                        "src_finding_id": "a",
                        "dst_finding_id": "b",
                        "edge_type": "derived_from",
                        "confidence": 0.6,
                        "created_by": "rule_engine",
                    },
                    {
                        "edge_id": "weaker_skip",
                        "src_finding_id": "a",
                        "dst_finding_id": "b",
                        "edge_type": "supports",
                        "confidence": 0.9,
                        "created_by": "rule_engine",
                    },
                    {
                        "edge_id": "supports_first",
                        "src_finding_id": "b",
                        "dst_finding_id": "c",
                        "edge_type": "supports",
                        "confidence": 0.4,
                        "created_by": "rule_engine",
                    },
                    {
                        "edge_id": "stronger_replaces",
                        "src_finding_id": "b",
                        "dst_finding_id": "c",
                        "edge_type": "derived_from",
                        "confidence": 0.3,
                        "created_by": "rule_engine",
                    },
                    {
                        "edge_id": "agent_first",
                        "src_finding_id": "c",
                        "dst_finding_id": "a",
                        "edge_type": "supports",
                        "confidence": 0.4,
                        "created_by": "agent_declared",
                    },
                    {
                        "edge_id": "rule_loses_to_agent",
                        "src_finding_id": "c",
                        "dst_finding_id": "a",
                        "edge_type": "derived_from",
                        "confidence": 0.9,
                        "created_by": "rule_engine",
                    },
                    {
                        "edge_id": "rule_first",
                        "src_finding_id": "c2",
                        "dst_finding_id": "a",
                        "edge_type": "supports",
                        "confidence": 0.4,
                        "created_by": "rule_engine",
                    },
                    {
                        "edge_id": "agent_replaces_rule",
                        "src_finding_id": "c2",
                        "dst_finding_id": "a",
                        "edge_type": "challenges",
                        "confidence": 0.1,
                        "created_by": "agent_declared",
                    },
                    {
                        "edge_id": "bad_conf",
                        "src_finding_id": "x",
                        "dst_finding_id": "y",
                        "edge_type": "supports",
                        "confidence": "bad",
                        "created_by": "rule_engine",
                    },
                ]
            )
            self.assertGreaterEqual(inserted, 5)
            edge_ids = {edge["edge_id"] for edge in local_store.get_edges_for_finding("b")}
            self.assertIn("existing_strong", edge_ids)
            self.assertIn("stronger_replaces", edge_ids)
            self.assertNotIn("weaker_skip", edge_ids)
            self.assertNotIn("supports_first", edge_ids)

            with local_store._get_conn() as conn:
                conn.execute(
                    "UPDATE finding_edges SET provenance = ? WHERE edge_id = ?",
                    ("{bad", "existing_strong"),
                )
            both_edges = local_store.get_edges_for_finding(
                "a", direction="both", edge_types=["derived_from"]
            )
            self.assertEqual(
                next(edge for edge in both_edges if edge["edge_id"] == "existing_strong")[
                    "provenance"
                ],
                {},
            )
            typed_subgraph = local_store.get_subgraph(
                "a",
                max_depth=2,
                min_confidence=0.0,
                edge_types=["derived_from", "challenges"],
                max_nodes=10,
            )
            self.assertFalse(typed_subgraph["truncated"])
            self.assertTrue(typed_subgraph["edges"])

            strict = local_store.get_pareto_leaderboard(
                "score",
                "maximize",
                [{"name": "cost", "direction": "minimize"}],
                requires_tier=True,
                top_k_dominated=5,
            )
            self.assertEqual(strict["best_in"]["score"], ["B", "C"])
            self.assertIn("C", strict["best_in"]["cost"])
            self.assertGreaterEqual(strict["n_total"], 3)
            self.assertGreaterEqual(strict["n_excluded_missing_axis"].get("score", 0), 1)

            permissive = local_store.get_pareto_leaderboard(
                "score",
                "maximize",
                [{"name": "cost", "direction": "minimize"}],
                requires_tier=False,
            )
            names = {row["variant_name"] for row in permissive["pareto_front"]}
            self.assertNotIn("FalseString", names)
            self.assertNotIn("FalseInt", names)

    def test_local_store_helper_edge_branches_are_stable(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        parsed_extra = {"extra": '{"peer_role": "builder"}'}
        self.assertEqual(local_store._finding_field(parsed_extra, "peer_role"), "builder")
        self.assertIsNone(local_store._finding_field({"extra": "{bad"}, "peer_role"))
        self.assertTrue(local_store._boolish_finding_field({"flag": 1}, "flag"))
        self.assertFalse(local_store._boolish_finding_field({"flag": 0}, "flag"))
        self.assertTrue(local_store._boolish_finding_field({"flag": "on"}, "flag"))
        self.assertFalse(local_store._boolish_finding_field({"flag": "off"}, "flag"))
        self.assertIsNone(local_store._boolish_finding_field({"flag": "maybe"}, "flag"))

        self.assertEqual(local_store._explicit_entity_key("bad::x"), "")
        self.assertEqual(local_store._explicit_entity_key("variant::"), "")
        self.assertEqual(
            local_store._pareto_entity_key(
                {"metrics": {"source_result_path": "artifacts/result.json"}}
            ),
            "artifact::artifacts/result.json",
        )
        self.assertEqual(
            local_store._pareto_entity_key({"metrics": {"frontier_entity_key": "variant::X"}}),
            "variant::x",
        )
        self.assertTrue(local_store._pareto_entity_key({"id": ""}).startswith("object::"))

        self.assertTrue(local_store._is_scout_or_partial_finding({"complete_eval": 0}))
        self.assertTrue(local_store._is_scout_or_partial_finding({"stage": "cheap-probe"}))
        self.assertTrue(
            local_store._is_scout_or_partial_finding({"status": "scored_complete_false"})
        )
        self.assertFalse(local_store._is_scout_or_partial_finding({"status": "fully scored"}))

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"LOCAL_STORE_DIR": tmp}, clear=False),
        ):
            local_store.init_db()
            status_path = os.path.join(tmp, "orchestrator_status.json")
            with open(status_path, "w", encoding="utf-8") as handle:
                json.dump([], handle)
            local_store._touch_operator_status({"ignored": True})
            with open(status_path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), [])

            with open(status_path, "w", encoding="utf-8") as handle:
                json.dump({"updated_at": "old"}, handle)
            with patch.object(local_store, "count_findings", side_effect=RuntimeError("boom")):
                local_store._touch_operator_status({"ignored": True})
            with open(status_path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), {"updated_at": "old"})

    def test_pareto_tie_breaker_keeps_larger_finding_id(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"LOCAL_STORE_DIR": tmp}, clear=False),
        ):
            local_store.init_db()
            base = {
                "finding_type": "result",
                "title": "candidate",
                "content": "content",
                "variant_name": "same",
                "generation_id": 0,
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "score": 1.0,
                    "cost": 1.0,
                    "tier": "T3",
                    "promotion_eligible": True,
                    "clean_promotion_eligible": True,
                },
            }
            local_store.insert_finding({**base, "id": "a"})
            local_store.insert_finding({**base, "id": "z", "title": "candidate z"})

            pareto = local_store.get_pareto_leaderboard(
                "score",
                "maximize",
                [{"name": "cost", "direction": "minimize"}],
                requires_tier=True,
            )

            self.assertEqual(pareto["n_total"], 1)
            self.assertEqual(pareto["pareto_front"][0]["title"], "candidate z")


if __name__ == "__main__":
    unittest.main()
