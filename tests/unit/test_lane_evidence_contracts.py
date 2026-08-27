from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class LaneEvidenceContractsTest(unittest.TestCase):
    def test_card_preserves_task_defined_numeric_and_categorical_metrics(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.card_builder import (
            build_card_from_finding,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            finding = {
                "id": "finding-a",
                "finding_type": "result",
                "title": "Active-alpha result",
                "metrics": {
                    "mean_active_alpha_vs_benchmark_pct": 1.25,
                    "active_ir": 0.31,
                    "frontier_lane": "alpha_incubator",
                    "source_frontier_lane": "alpha",
                    "strategy_family": "learned_alpha",
                    "tier": "T2",
                    "promotion_eligible": "false",
                    "clean_promotion_eligible": False,
                    "risk_violating_frontier_candidate": True,
                    "nan_metric": float("nan"),
                },
                "extra": {"peer_role": "exploit", "target_hypothesis": "H_g1_01"},
                "variant_name": "variant_a",
                "peer_id": "gen1_peer0",
                "generation_id": 1,
            }
            card = build_card_from_finding(finding, run_dir)

            self.assertEqual(card["metrics"]["mean_active_alpha_vs_benchmark_pct"], 1.25)
            self.assertEqual(card["metrics"]["active_ir"], 0.31)
            self.assertEqual(card["metrics"]["frontier_lane"], "alpha_incubator")
            self.assertEqual(card["metrics"]["source_frontier_lane"], "alpha")
            self.assertEqual(card["metrics"]["strategy_family"], "learned_alpha")
            self.assertEqual(card["metrics"]["peer_role"], "exploit")
            self.assertEqual(card["metrics"]["target_hypothesis"], "H_g1_01")
            self.assertFalse(card["metrics"]["promotion_eligible"])
            self.assertFalse(card["metrics"]["clean_promotion_eligible"])
            self.assertTrue(card["metrics"]["risk_violating_frontier_candidate"])
            self.assertNotIn("nan_metric", card["metrics"])

    def test_card_uses_top_level_promotion_fields_as_fallback(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.card_builder import (
            build_card_from_finding,
        )

        with tempfile.TemporaryDirectory() as tmp:
            finding = {
                "id": "finding-b",
                "finding_type": "result",
                "title": "Top-level metadata result",
                "metrics": {"mean_active_alpha_vs_benchmark_pct": 0.5},
                "promotion_eligible": True,
                "tier": "T3",
                "variant_name": "variant_b",
            }
            card = build_card_from_finding(finding, Path(tmp))

            self.assertTrue(card["metrics"]["promotion_eligible"])
            self.assertEqual(card["metrics"]["tier"], "T3")

    def test_evidence_pack_digest_preserves_lane_frontiers(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory import (
            evidence_pack_builder,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir()
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "lane_frontiers": {
                            "alpha_incubator": [
                                {
                                    "finding_id": "f1",
                                    "variant_name": "v1",
                                    "lane_metric_name": "mean_active_alpha",
                                    "lane_metric_value": -1.0,
                                    "promoted_for_lane": "alpha_incubator",
                                    "source_frontier_lane": "alpha",
                                    "generation_id": 0,
                                    "metrics": {
                                        "tier": "T1",
                                        "tier_status": "stop_after_T1",
                                        "promotion_eligible": False,
                                        "clean_promotion_eligible": False,
                                        "scored_complete": True,
                                        "risk_violating_frontier_candidate": True,
                                        "risk_repair_required": True,
                                        "risk_violation_reason": "drawdown",
                                        "lane_lower_tier_candidate": True,
                                        "lane_non_promotable_candidate": True,
                                    },
                                },
                                {
                                    "finding_id": "prelim-lane",
                                    "variant_name": "prelim_lane",
                                    "lane_metric_name": "mean_active_alpha",
                                    "lane_metric_value": 99.0,
                                    "promoted_for_lane": "alpha_incubator",
                                    "generation_id": 0,
                                    "evidence_stage": "preliminary",
                                },
                                {
                                    "finding_id": "unknown-lane",
                                    "variant_name": "unknown_lane",
                                    "lane_metric_name": "mean_active_alpha",
                                    "lane_metric_value": 98.0,
                                    "promoted_for_lane": "alpha_incubator",
                                    "generation_id": 0,
                                },
                                {
                                    "finding_id": "legacy-committed",
                                    "variant_name": "legacy_committed",
                                    "generation_id": 0,
                                    "tier": "historical_complete",
                                    "lane_metric_name": "score",
                                    "lane_metric_value": 1.0,
                                },
                                {
                                    "finding_id": "known-ratio-failure",
                                    "variant_name": "known_ratio_failure",
                                    "generation_id": 0,
                                    "scored_complete": True,
                                    "effort_ratio": 0.5,
                                    "lane_metric_name": "score",
                                    "lane_metric_value": 2.0,
                                },
                                {
                                    "finding_id": "future-legacy",
                                    "variant_name": "future_legacy",
                                    "generation_id": 2,
                                    "tier": "historical_complete",
                                    "lane_metric_name": "score",
                                    "lane_metric_value": 3.0,
                                },
                            ]
                        },
                        "frontier_lanes": [
                            {
                                "name": "alpha_incubator",
                                "description": "repair candidates",
                                "include_lanes": ["alpha", "alpha_incubator"],
                                "k": 10,
                                "cumulative_cap": 50,
                                "require_truthy_metrics": ["promotion_eligible"],
                                "require_falsey_metrics": ["is_smoke_eval"],
                                "min_metrics": {"mean_active_share": 0.005},
                                "exclude_roles": ["theorist"],
                                "axes": [["mean_active_alpha_vs_benchmark_pct", "maximize"]],
                                "allow_lower_tier": True,
                                "allow_non_promotable": True,
                                "allow_risk_violating": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            digest = evidence_pack_builder._digest_lane_frontiers(run_dir, current_gen_id=0)
            self.assertEqual(list(digest), ["alpha_incubator"])
            self.assertEqual(digest["alpha_incubator"][0]["finding_id"], "f1")
            self.assertNotIn(
                "prelim-lane",
                {entry["finding_id"] for entry in digest["alpha_incubator"]},
            )
            self.assertNotIn(
                "unknown-lane",
                {entry["finding_id"] for entry in digest["alpha_incubator"]},
            )
            self.assertIn(
                "legacy-committed",
                {entry["finding_id"] for entry in digest["alpha_incubator"]},
            )
            self.assertNotIn(
                "known-ratio-failure",
                {entry["finding_id"] for entry in digest["alpha_incubator"]},
            )
            self.assertNotIn(
                "future-legacy",
                {entry["finding_id"] for entry in digest["alpha_incubator"]},
            )
            self.assertEqual(digest["alpha_incubator"][0]["frontier_lane"], "alpha_incubator")
            self.assertEqual(digest["alpha_incubator"][0]["source_frontier_lane"], "alpha")
            self.assertEqual(digest["alpha_incubator"][0]["tier"], "T1")
            self.assertEqual(digest["alpha_incubator"][0]["tier_status"], "stop_after_T1")
            self.assertTrue(digest["alpha_incubator"][0]["risk_violating_frontier_candidate"])
            self.assertTrue(digest["alpha_incubator"][0]["risk_repair_required"])
            self.assertEqual(digest["alpha_incubator"][0]["risk_violation_reason"], "drawdown")
            self.assertTrue(digest["alpha_incubator"][0]["lane_lower_tier_candidate"])
            self.assertTrue(digest["alpha_incubator"][0]["lane_non_promotable_candidate"])

            metadata = evidence_pack_builder._digest_frontier_lane_metadata(run_dir)
            self.assertEqual(metadata[0]["name"], "alpha_incubator")
            self.assertEqual(metadata[0]["include_lanes"], ["alpha", "alpha_incubator"])
            self.assertEqual(metadata[0]["require_truthy_metrics"], ["promotion_eligible"])
            self.assertEqual(metadata[0]["require_falsey_metrics"], ["is_smoke_eval"])
            self.assertEqual(metadata[0]["min_metrics"]["mean_active_share"], 0.005)
            self.assertEqual(metadata[0]["exclude_roles"], ["theorist"])
            self.assertTrue(metadata[0]["allow_lower_tier"])
            self.assertTrue(metadata[0]["allow_non_promotable"])
            self.assertTrue(metadata[0]["allow_risk_violating"])

    def test_evidence_pack_digest_preserves_validation_candidates(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory import (
            evidence_pack_builder,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir()
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "validation_candidates": {
                            "cumulative": [
                                {
                                    "finding_id": "scout-now",
                                    "variant_name": "scout_now",
                                    "generation_id": 1,
                                    "metric_name": "score",
                                    "metric_value": 10.0,
                                    "metric_direction": "maximize",
                                    "submitted_frontier_lane": "alpha",
                                    "matched_frontier_lanes": ["alpha_incubator"],
                                    "evidence_stage": "scout",
                                    "evidence_maturity_rank": 1,
                                    "scout_only": True,
                                    "scored_cell_count": 6,
                                    "frontier_entity_key": "variant::scout_now",
                                    "exclusion_reason": "preliminary_or_incomplete_evidence",
                                    "recommended_next_step": "complete_scored_validation_before_frontier_or_gems",
                                },
                                {
                                    "finding_id": "scout-future",
                                    "variant_name": "scout_future",
                                    "generation_id": 3,
                                    "metric_name": "score",
                                    "metric_value": 99.0,
                                    "metric_direction": "maximize",
                                },
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            digest = evidence_pack_builder._digest_validation_candidates(
                run_dir,
                current_gen_id=1,
            )

            self.assertEqual([entry["finding_id"] for entry in digest], ["scout-now"])
            self.assertEqual(digest[0]["variant_name"], "scout_now")
            self.assertEqual(digest[0]["metric_value"], 10.0)
            self.assertEqual(digest[0]["submitted_frontier_lane"], "alpha")
            self.assertEqual(digest[0]["matched_frontier_lanes"], ["alpha_incubator"])
            self.assertEqual(digest[0]["evidence_stage"], "scout")
            self.assertEqual(
                digest[0]["recommended_next_step"],
                "complete_scored_validation_before_frontier_or_gems",
            )

    def test_evidence_pack_keeps_sibling_validation_candidate_with_shared_family_alias(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory import (
            evidence_pack_builder,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir()
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {
                            "1": [
                                {
                                    "finding_id": "durable-child-c025",
                                    "variant_name": "bridge_l1_eff_n_sweep",
                                    "generation_id": 1,
                                    "metric_name": "score",
                                    "metric_value": 10.0,
                                    "metrics": {
                                        "score": 10.0,
                                        "evidence_stage": "full_T1",
                                        "scored_complete": True,
                                        "frontier_entity_key": ("variant::bridge_l1_eff_n_sweep"),
                                        "source_result_path": (
                                            "results/c025/tiered_eval_summary.json"
                                        ),
                                    },
                                }
                            ]
                        },
                        "validation_candidates": {
                            "cumulative": [
                                {
                                    "finding_id": "scout-child-c005",
                                    "variant_name": "bridge_l1_eff_n_sweep",
                                    "generation_id": 0,
                                    "metric_name": "score",
                                    "metric_value": 9.0,
                                    "metric_direction": "maximize",
                                    "evidence_stage": "scout",
                                    "scout_only": True,
                                    "frontier_entity_key": ("variant::bridge_l1_eff_n_sweep"),
                                    "source_result_path": ("results/c005/tiered_eval_summary.json"),
                                    "identity_aliases": [
                                        "bridge_l1_eff_n_sweep",
                                        "results/c005/tiered_eval_summary.json",
                                    ],
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            digest = evidence_pack_builder._digest_validation_candidates(
                run_dir,
                current_gen_id=1,
            )

            self.assertEqual([entry["finding_id"] for entry in digest], ["scout-child-c005"])

    def test_evidence_pack_validation_candidates_prefer_generation_history(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory import (
            evidence_pack_builder,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir()
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "validation_candidates": {
                            "generations": {
                                "0": [
                                    {
                                        "finding_id": "old",
                                        "variant_name": "gen9_named_old_signal",
                                        "generation_id": 0,
                                        "metric_name": "score",
                                        "metric_value": 5.0,
                                        "metric_direction": "maximize",
                                        "signal_source_priority": 3,
                                        "evidence_maturity_rank": 1,
                                        "frontier_entity_key": "variant::old",
                                        "metrics": {"large": "x" * 100},
                                    }
                                ],
                                "2": [
                                    {
                                        "finding_id": "future",
                                        "variant_name": "future_signal",
                                        "generation_id": 2,
                                        "metric_name": "score",
                                        "metric_value": 99.0,
                                        "metric_direction": "maximize",
                                        "frontier_entity_key": "variant::future",
                                    }
                                ],
                            },
                            "cumulative": [
                                {
                                    "finding_id": "old-stale",
                                    "variant_name": "stale_cumulative_copy",
                                    "generation_id": 0,
                                    "metric_name": "score",
                                    "metric_value": 500.0,
                                    "metric_direction": "maximize",
                                    "signal_source_priority": 9,
                                    "evidence_maturity_rank": 9,
                                    "frontier_entity_key": "variant:old",
                                },
                                {
                                    "finding_id": "retired",
                                    "variant_name": "retired_signal",
                                    "generation_id": 0,
                                    "metric_name": "score",
                                    "metric_value": 777.0,
                                    "metric_direction": "maximize",
                                    "frontier_entity_key": "variant::retired",
                                },
                                {
                                    "finding_id": "bad",
                                    "variant_name": "bad_metric",
                                    "generation_id": 0,
                                    "metric_name": "score",
                                    "metric_value": "inf",
                                    "metric_direction": "sideways",
                                },
                                {
                                    "finding_id": "cum",
                                    "variant_name": "cumulative_only",
                                    "generation_id": 1,
                                    "metric_name": "score",
                                    "metric_value": 6.0,
                                    "metric_direction": "maximize",
                                    "frontier_entity_key": "variant::cum",
                                },
                            ],
                        },
                        "cumulative_top": [
                            {
                                "finding_id": "retired-full",
                                "variant_name": "retired_signal",
                                "generation_id": 0,
                                "metric_value": 1.0,
                                "frontier_entity_key": "variant:retired",
                                "scored_complete": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            digest = evidence_pack_builder._digest_validation_candidates(
                run_dir,
                current_gen_id=0,
            )

            self.assertEqual([entry["finding_id"] for entry in digest], ["old", "bad"])
            self.assertEqual(digest[0]["variant_name"], "gen9_named_old_signal")
            self.assertEqual(digest[0]["metric_value"], 5.0)
            self.assertNotIn("metric_value", digest[1])

    def test_evidence_pack_validation_candidates_keep_diverse_low_rank_representative(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory import (
            evidence_pack_builder,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir()
            candidates = [
                {
                    "finding_id": "high-a0",
                    "variant_name": "high_a0",
                    "generation_id": 0,
                    "metric_name": "score",
                    "metric_value": 100.0,
                    "metric_direction": "maximize",
                    "mechanism_family": "family_a",
                    "novelty_axis": "axis_a",
                },
                {
                    "finding_id": "high-a1",
                    "variant_name": "high_a1",
                    "generation_id": 0,
                    "metric_name": "score",
                    "metric_value": 99.0,
                    "metric_direction": "maximize",
                    "mechanism_family": "family_a",
                    "novelty_axis": "axis_a",
                },
                {
                    "finding_id": "high-a2",
                    "variant_name": "high_a2",
                    "generation_id": 0,
                    "metric_name": "score",
                    "metric_value": 98.0,
                    "metric_direction": "maximize",
                    "mechanism_family": "family_a",
                    "novelty_axis": "axis_a",
                },
                {
                    "finding_id": "low-bridge",
                    "variant_name": "low_bridge",
                    "generation_id": 0,
                    "metric_name": "score",
                    "metric_value": 1.0,
                    "metric_direction": "maximize",
                    "mechanism_family": "bridge_family",
                    "novelty_axis": "bridge_axis",
                    "next_step_intent": "bridge_validation",
                },
            ]
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps({"validation_candidates": {"cumulative": candidates}}),
                encoding="utf-8",
            )

            digest = evidence_pack_builder._digest_validation_candidates(
                run_dir,
                current_gen_id=0,
                max_entries=3,
            )

            self.assertEqual(
                [entry["finding_id"] for entry in digest],
                ["high-a0", "low-bridge", "high-a1"],
            )

    def test_evidence_pack_validation_candidates_ignore_future_or_preliminary_durable_surfaces(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory import (
            evidence_pack_builder,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir()
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "validation_candidates": {
                            "cumulative": [
                                {
                                    "finding_id": "future-current",
                                    "variant_name": "future_durable_later",
                                    "generation_id": 0,
                                    "metric_name": "score",
                                    "metric_value": 5.0,
                                    "metric_direction": "maximize",
                                    "frontier_entity_key": "variant::future_durable_later",
                                },
                                {
                                    "finding_id": "legacy-prelim",
                                    "variant_name": "legacy_prelim_surface",
                                    "generation_id": 0,
                                    "metric_name": "score",
                                    "metric_value": 4.0,
                                    "metric_direction": "maximize",
                                    "frontier_entity_key": "variant::legacy_prelim_surface",
                                },
                                {
                                    "finding_id": "alias-prelim",
                                    "variant_name": "alias_prelim_surface",
                                    "generation_id": 0,
                                    "metric_name": "score",
                                    "metric_value": 2.0,
                                    "metric_direction": "maximize",
                                    "frontier_entity_key": "variant::alias_prelim_surface",
                                },
                                {
                                    "finding_id": "bucket-now-current",
                                    "variant_name": "bucket_durable_now",
                                    "generation_id": 0,
                                    "metric_name": "score",
                                    "metric_value": 3.0,
                                    "metric_direction": "maximize",
                                    "frontier_entity_key": "variant::bucket_durable_now",
                                },
                                {
                                    "finding_id": "retired-by-variant-id",
                                    "variant_name": "gen0_peer0",
                                    "variant_id": "actual_child",
                                    "generation_id": 0,
                                    "metric_name": "score",
                                    "metric_value": 1.0,
                                    "metric_direction": "maximize",
                                },
                            ]
                        },
                        "generations": {
                            "0": [
                                {
                                    "finding_id": "bucket-now-full",
                                    "variant_name": "bucket_durable_now",
                                    "metric_value": 8.0,
                                    "frontier_entity_key": "variant::bucket_durable_now",
                                    "scored_complete": True,
                                }
                            ],
                            "2": [
                                {
                                    "finding_id": "future-full-bucket",
                                    "variant_name": "future_durable_later",
                                    "metric_value": 9.0,
                                    "frontier_entity_key": "variant::future_durable_later",
                                }
                            ],
                        },
                        "cumulative_top": [
                            {
                                "finding_id": "future-full",
                                "variant_name": "future_durable_later",
                                "generation_id": 2,
                                "metric_value": 10.0,
                                "frontier_entity_key": "variant::future_durable_later",
                            },
                            {
                                "finding_id": "legacy-prelim-copy",
                                "variant_name": "legacy_prelim_surface",
                                "generation_id": 0,
                                "metric_value": 99.0,
                                "frontier_entity_key": "variant::legacy_prelim_surface",
                                "excluded_from_durable_frontier": True,
                            },
                            {
                                "finding_id": "alias-prelim-copy",
                                "variant_name": "alias_prelim_surface",
                                "generation_id": 0,
                                "metric_value": 98.0,
                                "frontier_entity_key": "variant::alias_prelim_surface",
                                "status": "unscored_artifact",
                                "metrics": {"complete_eval": False},
                            },
                            {
                                "finding_id": "variant-id-full",
                                "variant_name": "gen0_peer0",
                                "variant_id": "actual_child",
                                "generation_id": 0,
                                "metric_value": 97.0,
                                "scored_complete": True,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            digest = evidence_pack_builder._digest_validation_candidates(
                run_dir,
                current_gen_id=0,
            )

            self.assertEqual(
                {entry["finding_id"] for entry in digest},
                {"future-current", "legacy-prelim", "alias-prelim"},
            )
            self.assertNotIn("retired-by-variant-id", {entry["finding_id"] for entry in digest})

    def test_legacy_committed_frontier_retires_only_matching_validation_signal(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory import (
            evidence_pack_builder,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir()
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "validation_candidates": {
                            "cumulative": [
                                {
                                    "finding_id": "legacy-signal",
                                    "variant_name": "legacy_committed",
                                    "generation_id": 0,
                                    "metric_name": "score",
                                    "metric_value": 2.0,
                                    "evidence_stage": "scout",
                                },
                                {
                                    "finding_id": "ratio-signal",
                                    "variant_name": "ratio_failed",
                                    "generation_id": 0,
                                    "metric_name": "score",
                                    "metric_value": 3.0,
                                    "evidence_stage": "scout",
                                },
                            ]
                        },
                        "lane_frontiers": {
                            "candidate_library": [
                                {
                                    "finding_id": "legacy-frontier",
                                    "variant_name": "legacy_committed",
                                    "generation_id": 0,
                                    "evidence_stage": "T1",
                                    "metric_value": 2.0,
                                },
                                {
                                    "finding_id": "ratio-frontier",
                                    "variant_name": "ratio_failed",
                                    "generation_id": 0,
                                    "evidence_stage": "T1",
                                    "metric_value": 3.0,
                                    "effort_ratio": 0.5,
                                    "coverage_ratio": 1.0,
                                },
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            digest = evidence_pack_builder._digest_validation_candidates(
                run_dir,
                current_gen_id=0,
            )

        self.assertEqual([entry["finding_id"] for entry in digest], ["ratio-signal"])

    def test_evidence_pack_validation_candidates_cover_fallback_alias_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory import (
            evidence_pack_builder,
        )

        real_import = __import__

        def force_local_fallbacks(name, globals=None, locals=None, fromlist=(), level=0):
            if name in {
                "praxist.plugins.workflow_stages.research_loop.backend.frontier",
                "praxist.plugins.workflow_stages.research_loop.backend.gems",
            }:
                raise ImportError("forced fallback")
            return real_import(name, globals, locals, fromlist, level)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir()
            shared_findings = run_dir / "shared_findings"
            shared_findings.mkdir()
            (shared_findings / "legacy-gem.json").write_text(
                json.dumps(
                    {
                        "id": "legacy-gem",
                        "source_frontier_entry": {"generation_id": 0},
                    }
                ),
                encoding="utf-8",
            )
            manifest_path = frontier_dir / "frontier_manifest.json"
            manifest_path.write_text("{bad", encoding="utf-8")

            self.assertEqual(evidence_pack_builder._digest_validation_candidates(run_dir), [])
            self.assertEqual(
                evidence_pack_builder._validation_candidate_aliases_from_manifest(run_dir),
                set(),
            )

            manifest_path.write_text(
                json.dumps(
                    {
                        "validation_candidates": {
                            "generations": {
                                "bad": "not-list",
                                "0": [
                                    "bad-entry",
                                    {
                                        "finding_id": "gen-candidate",
                                        "variant_name": "fallback_candidate",
                                        "metric_name": "score",
                                        "metric_value": "4.0",
                                        "metric_direction": "minimize",
                                        "signal_source_priority": 2,
                                        "evidence_maturity_rank": 1,
                                        "frontier_entity_key": "variant:fallback_candidate",
                                        "identity_aliases": ["fallback-alias"],
                                        "metrics": {
                                            "child_variant_id": ["child-a", "child-b"],
                                            "next_step_intent": "complete_validation",
                                        },
                                    },
                                ],
                                "2": [
                                    {
                                        "finding_id": "future-candidate",
                                        "variant_name": "future_candidate",
                                        "metric_value": 99.0,
                                    }
                                ],
                            },
                            "cumulative": [
                                {
                                    "finding_id": "retired-by-durable",
                                    "variant_name": "durable_candidate",
                                    "generation_id": 0,
                                    "metric_value": 10.0,
                                    "frontier_entity_key": "variant::durable_candidate",
                                    "identity_aliases": ["durable-alias"],
                                },
                                {
                                    "finding_id": "cum-candidate",
                                    "variant_name": "fallback_candidate",
                                    "generation_id": 0,
                                    "metric_value": 5.0,
                                    "metric_direction": "maximize",
                                    "frontier_entity_key": "variant::fallback_candidate",
                                    "identity_aliases": ["cum-alias"],
                                    "source_result_path": "results/cum/summary.json",
                                },
                            ],
                            "validator_identity_aliases_by_generation": {
                                "0": [
                                    "manual-active",
                                    "durable-alias",
                                    "gem-alias",
                                    "unknown-gem-alias",
                                ],
                                "2": ["future-alias"],
                                "bad": ["bad-alias"],
                            },
                            "validator_identity_aliases": ["only-without-cutoff"],
                        },
                        "cumulative_top": [
                            {
                                "finding_id": "durable-full",
                                "variant_name": "durable_candidate",
                                "generation_id": 0,
                                "metric_value": 1.0,
                                "scored_complete": True,
                                "frontier_entity_key": "variant::durable_candidate",
                                "identity_aliases": ["durable-alias"],
                            },
                            {
                                "finding_id": "prelim-copy",
                                "variant_name": "prelim_surface",
                                "generation_id": 0,
                                "metric_value": 9.0,
                                "excluded_from_durable_frontier": True,
                            },
                        ],
                        "generations": {
                            "0": [
                                {
                                    "finding_id": "generation-full",
                                    "variant_name": "generation_full",
                                    "metric_value": 2.0,
                                    "complete_eval": "yes",
                                    "identity_aliases": ["generation-alias"],
                                }
                            ],
                            "2": [
                                {
                                    "finding_id": "future-full",
                                    "variant_name": "future_full",
                                    "metric_value": 3.0,
                                    "complete_eval": True,
                                }
                            ],
                        },
                        "lane_frontiers": {
                            "control": [
                                {
                                    "finding_id": "lane-full",
                                    "variant_name": "lane_full",
                                    "generation_id": 0,
                                    "scored_complete": True,
                                }
                            ],
                            "bad": "not-list",
                        },
                        "gems": {
                            "entries": [
                                "bad-gem",
                                {
                                    "gem_finding_id": "legacy-gem",
                                    "admission_metrics": {
                                        "metric_value": 0.3,
                                        "frontier_entity_key": "variant::gem_candidate",
                                        "identity_aliases": ["gem-alias"],
                                    },
                                },
                                {
                                    "gem_finding_id": "future-gem",
                                    "admission_metrics": {
                                        "source_generation_id": 2,
                                        "metric_value": 0.4,
                                        "frontier_entity_key": "variant::future_gem",
                                    },
                                },
                                {
                                    "gem_finding_id": "unknown-gem",
                                    "admission_metrics": {
                                        "metric_value": 0.45,
                                        "frontier_entity_key": "variant::unknown_gem",
                                        "identity_aliases": ["unknown-gem-alias"],
                                    },
                                },
                                {
                                    "gem_finding_id": "nonclean-gem",
                                    "admission_metrics": {
                                        "metric_value": 0.5,
                                        "clean_promotion_eligible": False,
                                        "frontier_entity_key": "variant::bad_gem",
                                    },
                                },
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch("builtins.__import__", side_effect=force_local_fallbacks):
                digest = evidence_pack_builder._digest_validation_candidates(
                    run_dir,
                    current_gen_id=0,
                    max_entries=4,
                )
                aliases = evidence_pack_builder._validation_candidate_aliases_from_manifest(
                    run_dir,
                    current_gen_id=0,
                )
                aliases_without_cutoff = (
                    evidence_pack_builder._validation_candidate_aliases_from_manifest(
                        run_dir,
                        current_gen_id=None,
                    )
                )

            finding_ids = {entry["finding_id"] for entry in digest}
            self.assertIn("gen-candidate", finding_ids)
            self.assertIn("cum-candidate", finding_ids)
            self.assertNotIn("retired-by-durable", finding_ids)
            self.assertNotIn("future-candidate", finding_ids)
            self.assertIn("manual-active", aliases)
            self.assertIn("fallback-alias", aliases)
            self.assertNotIn("durable-alias", aliases)
            self.assertNotIn("gem-alias", aliases)
            self.assertIn("unknown-gem-alias", aliases)
            self.assertNotIn("future-alias", aliases)
            self.assertNotIn("only-without-cutoff", aliases)
            self.assertNotIn("unknown-gem-alias", aliases_without_cutoff)
            self.assertIn("only-without-cutoff", aliases_without_cutoff)


if __name__ == "__main__":
    unittest.main()
