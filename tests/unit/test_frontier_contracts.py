from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class FrontierStoreContractsTest(unittest.TestCase):
    def test_candidate_helper_contracts_cover_identity_and_evidence_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        cyclic: dict[str, object] = {}
        cyclic["self"] = cyclic
        self.assertIsNone(frontier._walk_for_metric(cyclic, "score"))
        self.assertIsNone(frontier._walk_for_metric({"score": True}, "score"))
        self.assertIsNone(frontier._walk_for_metric({"score": float("inf")}, "score"))
        self.assertEqual(
            frontier._walk_for_metric(
                {"notes": {"score": 99.0}, "metrics": {"score": 2.5}},
                "score",
                _strict_canonical=True,
            ),
            2.5,
        )
        self.assertEqual(frontier._norm_token_set({"Alpha": 1, "Beta": 2}), {"alpha", "beta"})
        self.assertEqual(frontier._norm_token_set([" Alpha ", None, ""]), {"alpha"})

        finding = {
            "variant_name": "Alpha_Sweep",
            "metrics": {
                "score": 1.0,
                "frontier_lane": "alpha",
                "strategy_family": "learned_alpha",
                "tags": ["Momentum", "Carry"],
                "risk_violating_frontier_candidate": True,
            },
            "details": {"role": "peer_lead", "evidence_stage": "full-t1"},
            "extra": {
                "extra": {
                    "bottleneck_target": "drawdown",
                    "source_result_path": "results/child/tiered_eval_summary.json",
                }
            },
        }
        self.assertEqual(frontier._merged_extra(finding)["bottleneck_target"], "drawdown")
        self.assertEqual(
            frontier._research_metadata_from_finding(finding)["bottleneck_target"],
            "drawdown",
        )
        self.assertEqual(frontier._metric_value(finding, "score"), 1.0)
        self.assertIsNone(frontier._metric_value({"metrics": {"score": True}}, "score"))
        self.assertEqual(frontier._candidate_lane(finding), "alpha")
        self.assertEqual(frontier._candidate_family(finding), "learned_alpha")
        self.assertEqual(frontier._candidate_role(finding), "peer_lead")
        self.assertIn("risk_violating_frontier_candidate", frontier._candidate_tags(finding))
        self.assertEqual(
            frontier._raw_candidate_field(finding, "source_result_path"),
            "results/child/tiered_eval_summary.json",
        )
        self.assertTrue(frontier._boolish_candidate_field({"x": "confirmed"}, "x"))
        self.assertFalse(frontier._boolish_candidate_field({"x": "partial"}, "x"))
        self.assertIsNone(frontier._boolish_candidate_field({"x": "maybe"}, "x"))
        aggregate_rejection = {
            "metrics": {
                "current_aggregate": {
                    "protocol_integrity_failed": True,
                    "validation_only_result": True,
                }
            }
        }
        self.assertTrue(frontier._candidate_protocol_integrity_failed(aggregate_rejection))
        self.assertTrue(
            frontier._candidate_has_validation_only_durability_marker(aggregate_rejection)
        )

        self.assertEqual(
            frontier._result_artifact_variant_token("results/child/tiered_eval_summary.json"),
            "child",
        )
        self.assertEqual(
            frontier._result_artifact_variant_token("results/child/summary.json"),
            "child",
        )
        self.assertEqual(
            frontier._result_artifact_variant_token("results/child/result_summary.json"),
            "child",
        )
        self.assertEqual(frontier._result_artifact_variant_token("results/summary.json"), "")
        self.assertEqual(
            frontier._result_artifact_variant_token("results/custom_root_eval_summary.json"),
            "root",
        )
        self.assertEqual(
            frontier._candidate_entity_key(
                {
                    "variant_id": "producer_owned_root_id",
                    "metrics": {"source_result_path": "results/summary.json"},
                }
            ),
            "variant::producer_owned_root_id",
        )
        self.assertEqual(
            frontier._candidate_entity_key(
                {
                    "variant_id": "stable_candidate_id",
                    "metrics": {"source_result_path": "results/moved/result_summary.json"},
                }
            ),
            "variant::stable_candidate_id",
        )
        self.assertEqual(
            frontier._candidate_entity_key(
                {
                    "variant_id": "producer_owned_candidate",
                    "metrics": {
                        "child_id": "legacy_shared_parent",
                        "source_result_path": "results/candidate/result_summary.json",
                    },
                }
            ),
            "variant::producer_owned_candidate",
        )
        self.assertEqual(
            frontier._candidate_entity_key(
                {
                    "variant_name": "shared_display_label",
                    "metrics": {
                        "child_id": "legacy_child",
                        "source_result_path": "results/path/result_summary.json",
                    },
                }
            ),
            "variant::legacy_child",
        )
        self.assertEqual(
            frontier._candidate_entity_key(
                {"child_variant_id": "first_child", "result_variant_id": "second_child"}
            ),
            "variant::first_child",
        )
        self.assertEqual(frontier._identity_token("unknown"), "")
        self.assertEqual(frontier._identity_variant_token("variant::Child A"), "child a")
        self.assertEqual(
            frontier._identity_variant_token("artifact::results/child/summary.json"),
            "child",
        )
        self.assertFalse(frontier._looks_like_broad_result_family("alpha_sweep_family"))
        self.assertEqual(
            frontier._candidate_child_identity_token({"child_variant_name": "child_1"}),
            "child_1",
        )
        self.assertEqual(
            frontier._candidate_child_identity_token({"child_variant_id": "child_2"}),
            "child_2",
        )
        self.assertEqual(
            frontier._candidate_child_identity_token({"child_variant_name": "alpha_sweep"}),
            "alpha_sweep",
        )
        self.assertEqual(
            frontier._candidate_result_path_identity(
                {
                    "variant_id": "alpha_sweep",
                    "metrics": {"source_result_path": "results/child/tiered_eval_summary.json"},
                }
            ),
            ("results/child/tiered_eval_summary.json", "child"),
        )
        self.assertEqual(frontier._candidate_entity_key(finding), "variant::child")
        self.assertEqual(
            frontier._candidate_entity_key({"frontier_entity_key": "variant::child"}),
            "variant::child",
        )
        self.assertEqual(
            frontier._candidate_entity_key(
                {
                    "frontier_entity_key": "variant::child",
                    "metrics": {"frontier_entity_key": "variant::broad_sweep"},
                }
            ),
            "variant::child",
        )
        self.assertTrue(frontier._entity_key_matches_variant_name("variant::child", "child"))
        self.assertFalse(frontier._entity_key_matches_variant_name("variant::child", "other"))

        self.assertEqual(
            frontier._candidate_status_text({"final_status": "Scored Complete"}).strip(),
            "scored complete",
        )
        self.assertTrue(frontier._status_has_any("scored-complete=false", "scored_complete_false"))
        self.assertEqual(frontier._normalized_evidence_stage({"is_smoke_eval": True}), "smoke")
        self.assertEqual(frontier._normalized_evidence_stage({"scout_only": True}), "scout")
        self.assertEqual(frontier._normalized_evidence_stage({"partial_cohort": True}), "scout")
        self.assertEqual(
            frontier._normalized_evidence_stage({"evidence_stage": "partial"}), "scout"
        )
        self.assertEqual(
            frontier._normalized_evidence_stage({"evidence_stage": "preliminary"}), "scout"
        )
        self.assertEqual(frontier._normalized_evidence_stage({"status": "prelim"}), "scout")
        self.assertEqual(frontier._normalized_evidence_stage({"status": "un-scored"}), "smoke")
        self.assertEqual(
            frontier._normalized_evidence_stage({"evidence_stage": "full_eval"}),
            "unknown",
        )
        self.assertEqual(
            frontier._normalized_evidence_stage({"evidence_stage": "promotion_attempt"}),
            "unknown",
        )
        self.assertEqual(
            frontier._normalized_evidence_stage({"evidence_stage": "replication"}),
            "unknown",
        )
        self.assertFalse(frontier._has_mature_frontier_evidence({"evidence_stage": "full_eval"}))
        self.assertEqual(
            frontier._normalized_evidence_stage(
                {"metrics": {"evidence_stage": "full_T1"}, "evidence_stage": "scout"}
            ),
            "scout",
        )
        self.assertEqual(
            frontier._normalized_evidence_stage({"status": "scout", "completion_status": False}),
            "scout",
        )
        self.assertTrue(
            frontier._is_preliminary_or_incomplete_evidence({"evidence_stage": "incomplete"})
        )
        incomplete_scored = {
            "variant_name": "incomplete_scored",
            "metrics": {
                "score": 9.0,
                "result_status": "failed_or_unscored",
                "exclusion_reason": "preliminary_or_incomplete_evidence",
                "excluded_from_durable_frontier": True,
            },
        }
        self.assertTrue(frontier._is_preliminary_or_incomplete_evidence(incomplete_scored))
        self.assertTrue(frontier._is_retainable_validation_candidate(incomplete_scored))
        self.assertEqual(
            frontier._normalized_evidence_stage({"scored_complete": False}),
            "scout",
        )
        self.assertEqual(
            frontier._normalized_evidence_stage(
                {"metrics": {"scored_complete": True}, "scored_complete": False}
            ),
            "scout",
        )
        self.assertTrue(
            frontier._is_preliminary_or_incomplete_evidence(
                {"metrics": {"scored_complete": True}, "scored_complete": False}
            )
        )
        self.assertEqual(
            frontier._normalized_evidence_stage({"evidence_stage": "forced-t3"}), "unknown"
        )
        self.assertEqual(frontier._normalized_evidence_stage({"tier": "T1"}), "unknown")
        self.assertEqual(
            frontier._normalized_evidence_stage({"scored_complete": True}), "scored_complete"
        )
        self.assertEqual(frontier._evidence_maturity_rank({"tier": "T3"}), 0)
        metadata = frontier._evidence_metadata_from_candidate(
            {
                "tier": "T2",
                "evidence_stage": "T2",
                "scored_complete": True,
                "scout_only": "no",
            }
        )
        self.assertEqual(metadata["evidence_stage"], "scored_complete")
        self.assertEqual(metadata["evidence_maturity_rank"], 2)
        self.assertTrue(metadata["scored_complete"])
        current_completion = frontier._evidence_metadata_from_candidate(
            {
                "scored_complete": True,
                "metrics": {"scored_complete": False},
            }
        )
        self.assertTrue(current_completion["scored_complete"])
        current_incompletion = frontier._evidence_metadata_from_candidate(
            {
                "scored_complete": False,
                "metrics": {"scored_complete": True},
            }
        )
        self.assertFalse(current_incompletion["scored_complete"])

    def test_frontier_filter_risk_and_diversity_helper_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier
        from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
            FrontierStore,
        )

        finding = {
            "variant_name": "alpha",
            "frontier_lane": "alpha_incubator",
            "strategy_family": "fam-a",
            "tags": ["momentum", "repair"],
            "role": "peer",
            "metrics": {
                "score": 1.2,
                "drawdown": 0.1,
                "clean_promotion_eligible": "false",
                "risk_flag": "clean",
            },
        }
        self.assertFalse(frontier._entity_key_matches_variant_name("finding::alpha", "alpha"))
        self.assertTrue(frontier._entity_key_matches_variant_name("variant::alpha", "Alpha"))
        self.assertFalse(frontier._matches_lane_filters(finding, {"include_lanes": ["benchmark"]}))
        self.assertFalse(
            frontier._matches_lane_filters(finding, {"exclude_lanes": ["alpha_incubator"]})
        )
        self.assertFalse(frontier._matches_lane_filters(finding, {"include_families": ["other"]}))
        self.assertFalse(frontier._matches_lane_filters(finding, {"exclude_families": ["fam-a"]}))
        self.assertFalse(frontier._matches_lane_filters(finding, {"include_tags": ["missing"]}))
        self.assertFalse(frontier._matches_lane_filters(finding, {"exclude_tags": ["repair"]}))
        self.assertFalse(frontier._matches_lane_filters(finding, {"include_roles": ["chair"]}))
        self.assertFalse(frontier._matches_lane_filters(finding, {"exclude_roles": ["peer"]}))
        self.assertFalse(
            frontier._matches_lane_filters(finding, {"require_metrics": ["missing_metric"]})
        )
        self.assertFalse(
            frontier._matches_lane_filters(
                finding, {"require_truthy_metrics": ["clean_promotion_eligible"]}
            )
        )
        self.assertFalse(
            frontier._matches_lane_filters(finding, {"require_falsey_metrics": ["risk_flag"]})
        )
        self.assertFalse(frontier._matches_lane_filters(finding, {"min_metrics": {"score": 2.0}}))
        self.assertFalse(
            frontier._matches_lane_filters(finding, {"max_metrics": {"drawdown": 0.05}})
        )
        self.assertTrue(
            frontier._matches_lane_filters(
                finding,
                {
                    "include_lanes": ["alpha_incubator"],
                    "include_families": ["fam-a"],
                    "include_tags": ["momentum"],
                    "include_roles": ["peer"],
                    "require_metrics": ["score"],
                    "require_falsey_metrics": ["clean_promotion_eligible"],
                    "min_metrics": {"score": 1.0},
                    "max_metrics": {"drawdown": 0.2},
                },
            )
        )
        self.assertIsNone(frontier._lane_values(finding, [("missing", "maximize")]))
        self.assertEqual(
            frontier._lane_values(
                finding,
                [("score", "maximize")],
                [("drawdown", "minimize"), ("missing", "maximize")],
            ),
            {"score": 1.2, "drawdown": 0.1},
        )
        self.assertTrue(
            frontier._pareto_dominates(
                {"score": 2.0, "drawdown": 0.1},
                {"score": 1.0, "drawdown": 0.2},
                [("score", "maximize"), ("drawdown", "minimize")],
            )
        )
        self.assertFalse(
            frontier._pareto_dominates(
                {"score": 1.0, "drawdown": 0.3},
                {"score": 1.0, "drawdown": 0.2},
                [("score", "maximize"), ("drawdown", "minimize")],
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            store = FrontierStore(
                Path(tmp) / "frontier",
                primary_metric="score",
                metric_direction="minimize",
                risk_violating_frontier_enabled=True,
                risk_violating_primary_threshold=0.5,
            )
            self.assertTrue(store._beats_risk_frontier_threshold(0.4))
            self.assertFalse(store._beats_risk_frontier_threshold(0.7))
            self.assertEqual(
                store._hard_constraint_count(
                    {"metrics": {"hard_constraint_violations": ["a", "b"]}}
                ),
                2,
            )
            self.assertIsNone(
                store._risk_violating_reason(
                    finding={"metrics": {}},
                    tier="T3",
                    promotion_rejected=False,
                    metric_value=0.4,
                )
            )
            reason = store._risk_violating_reason(
                finding={"metrics": {"n_hard_constraint_violations": 2}},
                tier="T1",
                promotion_rejected=True,
                metric_value=0.4,
            )
            self.assertIsNotNone(reason)
            self.assertIn("risk issues", reason or "")

        self.assertIsNone(frontier._extract_design_dimensions({"design_dimensions": []}))
        self.assertEqual(
            frontier._extract_design_dimensions(
                {
                    "design_dimensions": {"axis": "top"},
                    "metrics": {"design_dimensions": {"axis": " Nested ", "drop": None}},
                }
            ),
            {"axis": "nested"},
        )
        self.assertIsNone(
            frontier.compute_dimension_overlap(
                {"design_dimensions": {"a": "x"}},
                {"design_dimensions": {"b": "x"}},
            )
        )
        self.assertEqual(
            frontier.compute_dimension_overlap(
                {"design_dimensions": {"a": "x", "b": "y"}},
                {"design_dimensions": {"a": "x", "b": "z"}},
            )["overlap_count"],
            1,
        )
        annotated = frontier.annotate_findings_with_diversity_overlap(
            [
                {"variant_name": "clone", "design_dimensions": {"a": "x", "b": "y"}},
                {"variant_name": "clean", "design_dimensions": {"a": "z", "b": "w"}},
                {"variant_name": "missing"},
            ],
            [{"variant_name": "anchor", "design_dimensions": {"a": "x", "b": "y"}}],
            expected_dim_count=2,
        )
        statuses = [item["metrics"]["diversity_overlap_status"] for item in annotated]
        self.assertEqual(statuses, ["clone", "clean", "no_data"])
        self.assertEqual(
            annotated[2]["metrics"]["diversity_overlap_no_data_reason"],
            "finding_dimensions_missing",
        )
        missing_anchor_dimensions = frontier.annotate_findings_with_diversity_overlap(
            [{"variant_name": "candidate", "design_dimensions": {"a": "x"}}],
            [{"variant_name": "anchor"}],
            expected_dim_count=1,
        )
        self.assertEqual(
            missing_anchor_dimensions[0]["metrics"]["diversity_overlap_no_data_reason"],
            "anchor_dimensions_missing",
        )
        no_common_dimensions = frontier.annotate_findings_with_diversity_overlap(
            [{"variant_name": "candidate", "design_dimensions": {"a": "x"}}],
            [{"variant_name": "anchor", "design_dimensions": {"b": "x"}}],
            expected_dim_count=1,
        )
        self.assertEqual(
            no_common_dimensions[0]["metrics"]["diversity_overlap_no_data_reason"],
            "no_common_dimensions",
        )
        no_anchor = frontier.annotate_findings_with_diversity_overlap(
            [{"variant_name": "lonely", "design_dimensions": {"a": "x"}}],
            [],
            expected_dim_count=1,
        )
        self.assertEqual(no_anchor[0]["metrics"]["diversity_overlap_status"], "no_anchors")

    def test_promoted_design_dimensions_remain_available_to_next_generation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "anchor",
                        "finding_type": "result",
                        "variant_name": "anchor",
                        "metrics": {"score": 1.0, "scored_complete": True},
                        "design_dimensions": {"mechanism": "a", "schedule": "cosine"},
                    }
                ],
            )
            summary = store.get_summary()
            annotated = frontier.annotate_findings_with_diversity_overlap(
                [
                    {
                        "variant_name": "next",
                        "design_dimensions": {"mechanism": "a", "schedule": "linear"},
                    }
                ],
                summary,
                expected_dim_count=2,
            )

        self.assertEqual(promoted[0]["design_dimensions"], {"mechanism": "a", "schedule": "cosine"})
        self.assertEqual(summary[0]["design_dimensions"], promoted[0]["design_dimensions"])
        self.assertEqual(annotated[0]["metrics"]["diversity_overlap_status"], "clean")

    def test_pi_lane_digest_trusts_committed_canonical_membership(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier, prompt_context
        from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import PIAgent
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
            _digest_frontier_lane_metadata,
            _digest_lane_frontiers,
            _digest_validation_candidates,
            _validation_candidate_aliases_from_manifest,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            store = frontier.FrontierStore(
                run_dir / "frontier",
                primary_metric="quality_score",
                frontier_lanes=[
                    {
                        "name": "durable_candidates",
                        "k": 1,
                        "include_lanes": ["candidate"],
                        "axes": [("quality_score", "maximize")],
                    }
                ],
            )
            store.promote(
                0,
                [
                    {
                        "id": "finding-complete",
                        "finding_type": "result",
                        "variant_name": "nextgen99_method",
                        "metrics": {
                            "quality_score": 1.0,
                            "frontier_lane": "candidate",
                            "scored_complete": True,
                        },
                    }
                ],
            )
            manifest_path = run_dir / "frontier" / "frontier_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = manifest["lane_frontiers"]["durable_candidates"][0]
            entry.update(
                {
                    "tier_status": "capped_at_authorized_stage",
                    "mature_enough": True,
                    "effort_ratio": 1.0,
                    "coverage_ratio": 1.0,
                }
            )
            entry["metrics"].update(
                {
                    "result_capped": True,
                    "mature_enough": True,
                    "effort_ratio": 1.0,
                    "coverage_ratio": 1.0,
                }
            )
            future_entry = json.loads(json.dumps(entry))
            future_entry.update(
                {
                    "finding_id": "finding-future",
                    "variant_name": "future_candidate",
                    "generation_id": 0,
                    "source_generation_id": 2,
                }
            )
            manifest["lane_frontiers"]["durable_candidates"].append(future_entry)
            manifest["cumulative_top"].append(future_entry)
            manifest["generations"]["0"].append(future_entry)
            bucket_future_entry = json.loads(json.dumps(entry))
            bucket_future_entry.update(
                {
                    "finding_id": "finding-bucket-future",
                    "variant_name": "bucket_future_candidate",
                    "generation_id": 0,
                    "source_generation_id": 0,
                }
            )
            manifest["generations"]["2"] = [bucket_future_entry]
            manifest["validation_candidates"] = {
                "generations": {
                    "0": [
                        {
                            "finding_id": "validation-duplicate",
                            "variant_name": "nextgen99_method",
                            "generation_id": 0,
                            "metric_value": 0.5,
                        },
                        {
                            "finding_id": "validation-future",
                            "variant_name": "future_candidate",
                            "generation_id": 0,
                            "metric_value": 0.6,
                        },
                        {
                            "finding_id": "validation-bucket-future",
                            "variant_name": "bucket_future_candidate",
                            "generation_id": 0,
                            "metric_value": 0.7,
                        },
                    ]
                },
                "cumulative": [],
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.frontier."
                "_is_committed_frontier_entry",
                return_value=False,
            ) as legacy_filter:
                digest = _digest_lane_frontiers(run_dir, current_gen_id=0)
            self.assertEqual(
                [item["finding_id"] for item in digest["durable_candidates"]],
                ["finding-complete"],
            )
            legacy_filter.assert_not_called()
            validation = _digest_validation_candidates(run_dir, current_gen_id=0)
            self.assertEqual(
                {item["finding_id"] for item in validation},
                {"validation-future", "validation-bucket-future"},
            )
            self.assertNotIn(
                "validation-duplicate",
                _validation_candidate_aliases_from_manifest(run_dir, current_gen_id=0),
            )
            self.assertIn(
                "validation-future",
                _validation_candidate_aliases_from_manifest(run_dir, current_gen_id=0),
            )
            self.assertIn(
                "validation-bucket-future",
                _validation_candidate_aliases_from_manifest(run_dir, current_gen_id=0),
            )
            agent = PIAgent(
                run_dir=run_dir,
                workspace=run_dir,
                cohort_size=1,
                model="test-model",
            )
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.pi_agent."
                "_is_committed_frontier_entry",
                return_value=False,
            ) as fallback_filter:
                summary = agent._load_frontier_summary(completed_gen_id=0)
            self.assertEqual(
                [item["finding_id"] for item in summary],
                ["finding-complete"],
            )
            fallback_filter.assert_not_called()

            fallback_manifest = json.loads(json.dumps(manifest))
            fallback_manifest["cumulative_top"] = []
            manifest_path.write_text(json.dumps(fallback_manifest), encoding="utf-8")
            fallback_summary = agent._load_frontier_summary(completed_gen_id=0)
            self.assertEqual(
                [item["finding_id"] for item in fallback_summary],
                ["finding-complete"],
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            lane_counts: dict[str, int] = {}
            bounded_lanes = _digest_lane_frontiers(
                run_dir,
                max_entries_per_lane=1,
                total_entries_by_lane=lane_counts,
            )
            lane_metadata = {
                item["name"]: item
                for item in _digest_frontier_lane_metadata(
                    run_dir,
                    total_entries_by_lane=lane_counts,
                    returned_entries_by_lane=bounded_lanes,
                )
            }
            self.assertEqual(lane_metadata["durable_candidates"]["available_entry_count"], 2)
            self.assertEqual(lane_metadata["durable_candidates"]["returned_entry_count"], 1)
            self.assertTrue(lane_metadata["durable_candidates"]["entries_truncated"])

            store._manifest = manifest
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.frontier."
                "_is_committed_frontier_entry",
                return_value=False,
            ) as runtime_filter:
                self.assertEqual(
                    [item["finding_id"] for item in store.get_summary_up_to_generation(0)],
                    ["finding-complete"],
                )
                self.assertEqual(
                    [item["finding_id"] for item in store.get_parent_summary_up_to_generation(0)],
                    ["finding-complete"],
                )
                self.assertEqual(
                    [item["finding_id"] for item in store.get_summary_for_generation(0)],
                    ["finding-complete"],
                )
            runtime_filter.assert_not_called()
            task_spec = SimpleNamespace(
                evaluation=SimpleNamespace(
                    primary_metric="quality_score",
                    maturity_policy={},
                    aux_metrics=[],
                    anchor_metrics=[],
                    frontier_lanes=[],
                ),
                gems=None,
            )
            with patch.object(prompt_context, "_is_committed_frontier_entry", return_value=False):
                lane_prompt = prompt_context._lane_entries_for_prompt(
                    manifest,
                    parent_eligible=True,
                    task_spec=task_spec,
                    completed_gen_id=0,
                )
            self.assertEqual(
                [item["finding_id"] for item in lane_prompt],
                ["finding-complete"],
            )

            manifest.pop("artifact_semantics")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.frontier."
                "_is_committed_frontier_entry",
                return_value=False,
            ):
                self.assertEqual(
                    _digest_lane_frontiers(run_dir, current_gen_id=0)["durable_candidates"],
                    [],
                )

    def test_lane_frontier_preserves_research_metadata_from_extra(self) -> None:
        from types import SimpleNamespace

        from praxist.plugins.workflow_stages.research_loop.backend import (
            frontier,
            prompt_context,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
            _digest_lane_frontiers,
        )

        metadata = {
            "bottleneck_target": "drawdown_regression",
            "evidence_stage": "full_T1",
            "scored_complete": True,
            "tradeoff_class": "high_return_drawdown_repair_target",
            "primary_tradeoff": "return_vs_mdd",
            "next_step_intent": "repair_failure_mode",
            "parent_candidate": "parent_alpha",
            "parent_usage": "repair",
            "source_lane": "alpha_incubator",
            "target_lane": "confirmed_alpha",
            "coverage_check": "span alpha and risk axes",
            "mechanism_hypothesis_deliverable": "write mechanism note",
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            store = frontier.FrontierStore(
                run_dir / "frontier",
                promote_top_k=1,
                primary_metric="future_fitness",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 1,
                        "include_lanes": ["alpha_incubator"],
                        "allow_risk_violating": True,
                        "axes": [("future_fitness", "maximize")],
                    },
                ],
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "f_extra",
                        "finding_type": "result",
                        "variant_name": "repair_parent",
                        "metrics": {
                            "future_fitness": 1.25,
                            "tier": "T3",
                            "promotion_eligible": True,
                            "frontier_lane": "alpha_incubator",
                        },
                        "extra": metadata,
                    }
                ],
            )

            self.assertEqual(len(promoted), 1)
            for key, value in metadata.items():
                self.assertEqual(promoted[0][key], value)
                self.assertEqual(promoted[0]["metrics"][key], value)

            manifest_entry = store.get_manifest()["lane_frontiers"]["alpha_incubator"][0]
            digest_entry = _digest_lane_frontiers(run_dir)["alpha_incubator"][0]
            for key, value in metadata.items():
                self.assertEqual(manifest_entry[key], value)
                self.assertEqual(manifest_entry["metrics"][key], value)
                self.assertEqual(digest_entry[key], value)
            frontier_entry_for_prompt = dict(manifest_entry)
            frontier_entry_for_prompt["generation_id"] = 0
            frontier_for_prompt = SimpleNamespace(get_summary=lambda: [frontier_entry_for_prompt])
            task_spec = SimpleNamespace(
                evaluation=SimpleNamespace(
                    primary_metric="future_fitness",
                    diversity_dimensions=[],
                    must_explore_axes=[],
                )
            )
            with (
                patch(
                    "praxist.plugins.graph_maintainers.finding_graph_mvp.engine.build_session_start_graph_context",
                    return_value="",
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.pi_agent.load_agenda_for_gen",
                    return_value={},
                ),
            ):
                context = prompt_context.build_prompt_context(
                    task_spec=task_spec,
                    workspace=run_dir,
                    run_dir=run_dir,
                    results_dir=run_dir / "results",
                    variants_dir=run_dir / "variants",
                    findings_dir=run_dir / "shared_findings",
                    frontier=frontier_for_prompt,
                    local_mode=True,
                    gen_id=1,
                    peer_index=0,
                    cohort_size=1,
                    strategy="explore",
                )
            frontier_entry = context["frontier_summary"][0]
            self.assertEqual(frontier_entry["source_lane"], "alpha_incubator")
            self.assertEqual(frontier_entry["target_lane"], "confirmed_alpha")
            self.assertEqual(frontier_entry["coverage_check"], "span alpha and risk axes")
            self.assertEqual(
                frontier_entry["mechanism_hypothesis_deliverable"],
                "write mechanism note",
            )

    def test_frontier_repairs_stale_source_result_to_best_canonical_summary(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier
        from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
            _json_digest,
            normalized_result_summary,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
            _digest_lane_frontiers,
        )

        def write_summary(
            run_dir: Path,
            child: str,
            score: float,
            stage: str,
            generation_id: int = 0,
            variant_name: str = "shared_parent",
        ) -> str:
            summary_path = run_dir / "results" / child / "tiered_eval_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "variant_name": variant_name,
                        "result_variant_id": variant_name,
                        "generation_id": generation_id,
                        "tier_reached": "T1",
                        "tier_status": "complete",
                        "evidence_stage": stage,
                        "current_aggregate": {
                            "score": score,
                            "scored_complete": stage != "scout",
                            "frontier_lane": "alpha_incubator",
                        },
                    }
                ),
                encoding="utf-8",
            )
            return str(summary_path.relative_to(run_dir)).replace("\\", "/")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            stale_path = write_summary(run_dir, "shared_parent_scout", 0.40, "scout")
            full_path = write_summary(run_dir, "shared_parent_full_t1", 1.30, "full_T1")
            store = frontier.FrontierStore(
                run_dir / "frontier",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 1,
                        "include_lanes": ["alpha_incubator"],
                        "axes": [("score", "maximize")],
                    }
                ],
            )

            promoted = store.promote(
                0,
                [
                    {
                        "id": "finding_stale_source",
                        "finding_type": "result",
                        "variant_name": "shared_parent",
                        "metrics": {
                            "score": 0.40,
                            "tier": "T1",
                            "promotion_eligible": True,
                            "frontier_lane": "alpha_incubator",
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                            "source_result_path": stale_path,
                            "source_result_sha256": "stale-sha",
                        },
                        "source_result_path": stale_path,
                        "source_result_sha256": "stale-sha",
                        "extra": {
                            "source_result_path": stale_path,
                            "source_result_sha256": "stale-sha",
                        },
                    }
                ],
            )

            full_summary_path = run_dir / full_path
            expected_digest = _json_digest(
                normalized_result_summary(
                    json.loads(full_summary_path.read_text(encoding="utf-8")),
                    summary_path=full_summary_path,
                )
            )

            self.assertEqual(promoted[0]["source_result_path"], full_path)
            self.assertEqual(promoted[0]["source_result_sha256"], expected_digest)
            self.assertNotIn("source_result_sha256", promoted[0].get("extra", {}))
            self.assertEqual(promoted[0]["selected_source_result_path"], stale_path)
            self.assertEqual(promoted[0]["canonical_source_result_path"], full_path)
            self.assertEqual(promoted[0]["best_available_summary_path"], full_path)
            self.assertEqual(promoted[0]["metric_value"], 1.30)
            self.assertEqual(promoted[0]["lane_metric_value"], 1.30)
            top_finding = json.loads(
                (run_dir / "frontier" / "gen_0" / "top_1_finding.json").read_text(encoding="utf-8")
            )
            self.assertEqual(top_finding["source_result_path"], full_path)
            self.assertEqual(top_finding["metrics"]["score"], 1.30)

            manifest_entry = store.get_manifest()["lane_frontiers"]["alpha_incubator"][0]
            self.assertEqual(manifest_entry["source_result_path"], full_path)
            self.assertEqual(manifest_entry["source_result_sha256"], expected_digest)
            self.assertEqual(manifest_entry["selected_source_result_path"], stale_path)
            self.assertEqual(
                manifest_entry["source_selection_warning"],
                "better_canonical_result_source_available",
            )
            digest_entry = _digest_lane_frontiers(run_dir)["alpha_incubator"][0]
            self.assertEqual(digest_entry["source_result_path"], full_path)
            self.assertEqual(digest_entry["best_available_summary_path"], full_path)

            rewritten_summary = json.loads(full_summary_path.read_text(encoding="utf-8"))
            rewritten_summary["revision"] = 2
            full_summary_path.write_text(json.dumps(rewritten_summary), encoding="utf-8")
            rewritten_digest = _json_digest(
                normalized_result_summary(rewritten_summary, summary_path=full_summary_path)
            )
            manifest_entry["extra"] = {
                "source_result_path": full_path,
                "source_result_sha256": expected_digest,
            }
            store._save_manifest()

            reloaded = frontier.FrontierStore(
                run_dir / "frontier",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 1,
                        "include_lanes": ["alpha_incubator"],
                        "axes": [("score", "maximize")],
                    }
                ],
            )
            reloaded_entry = reloaded.get_manifest()["lane_frontiers"]["alpha_incubator"][0]
            self.assertEqual(reloaded_entry["source_result_path"], full_path)
            self.assertEqual(reloaded_entry["source_result_sha256"], rewritten_digest)
            self.assertNotIn("source_result_sha256", reloaded_entry.get("extra", {}))
            self.assertEqual(reloaded_entry["selected_source_result_path"], stale_path)
            self.assertEqual(
                reloaded_entry["source_selection_warning"],
                "better_canonical_result_source_available",
            )

    def test_source_repair_clears_nested_current_aggregate_coordinates(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        payload = {
            "current_aggregate": {
                "source_result_path": "results/old.json",
                "source_result_sha256": "old-sha",
                "unrelated_measurement": 7.0,
            }
        }

        frontier._clear_result_artifact_coordinates(payload)

        self.assertEqual(payload["current_aggregate"], {"unrelated_measurement": 7.0})

    def test_frontier_source_repair_preserves_distinct_sweep_children(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        def write_summary(
            run_dir: Path,
            child: str,
            score: float,
            summary_name: str = "tiered_eval_summary.json",
        ) -> str:
            summary_path = run_dir / "results" / child / summary_name
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "variant_name": "shared_parent",
                        "generation_id": 0,
                        "tier_reached": "T1",
                        "tier_status": "complete",
                        "evidence_stage": "full_T1",
                        "scored_complete": True,
                        "current_aggregate": {
                            "score": score,
                            "scored_complete": True,
                            "frontier_lane": "alpha_incubator",
                        },
                    }
                ),
                encoding="utf-8",
            )
            return str(summary_path.relative_to(run_dir)).replace("\\", "/")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            child_a_path = write_summary(run_dir, "child_a", 0.70, "result_summary.json")
            child_b_path = write_summary(run_dir, "child_b", 1.20, "result_summary.json")
            store = frontier.FrontierStore(
                run_dir / "frontier",
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 2,
                        "cumulative_cap": 2,
                        "include_lanes": ["alpha_incubator"],
                        "axes": [("score", "maximize")],
                    }
                ],
            )

            store.promote(
                0,
                [
                    {
                        "id": "finding_a",
                        "finding_type": "result",
                        "variant_name": "shared_parent",
                        "metrics": {
                            "score": 0.70,
                            "tier": "T1",
                            "promotion_eligible": True,
                            "frontier_lane": "alpha_incubator",
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                            "source_result_path": child_a_path,
                        },
                    },
                    {
                        "id": "finding_b",
                        "finding_type": "result",
                        "variant_name": "shared_parent",
                        "metrics": {
                            "score": 1.20,
                            "tier": "T1",
                            "promotion_eligible": True,
                            "frontier_lane": "alpha_incubator",
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                            "source_result_path": child_b_path,
                        },
                    },
                ],
            )

            manifest = store.get_manifest()
            paths = {
                entry["finding_id"]: entry["source_result_path"]
                for entry in manifest["generations"]["0"]
            }
            self.assertEqual(paths["finding_a"], child_a_path)
            self.assertEqual(paths["finding_b"], child_b_path)
            self.assertEqual(len(manifest["lane_frontiers"]["alpha_incubator"]), 2)

    def test_frontier_source_repair_does_not_pull_future_unpromoted_results(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        def write_summary(run_dir: Path, child: str, score: float, gen_id: int) -> str:
            summary_path = run_dir / "results" / child / "tiered_eval_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "variant_name": "shared_parent",
                        "generation_id": gen_id,
                        "tier_reached": "T1",
                        "tier_status": "complete",
                        "evidence_stage": "full_T1",
                        "scored_complete": True,
                        "current_aggregate": {
                            "score": score,
                            "scored_complete": True,
                            "frontier_lane": "alpha_incubator",
                        },
                    }
                ),
                encoding="utf-8",
            )
            return str(summary_path.relative_to(run_dir)).replace("\\", "/")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen0_path = write_summary(run_dir, "shared_parent", 1.00, 0)
            store = frontier.FrontierStore(
                run_dir / "frontier",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 1,
                        "include_lanes": ["alpha_incubator"],
                        "axes": [("score", "maximize")],
                    }
                ],
            )
            store.promote(
                0,
                [
                    {
                        "id": "finding_gen0",
                        "finding_type": "result",
                        "variant_name": "shared_parent",
                        "metrics": {
                            "score": 1.00,
                            "tier": "T1",
                            "promotion_eligible": True,
                            "frontier_lane": "alpha_incubator",
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                            "source_result_path": gen0_path,
                        },
                    }
                ],
            )
            write_summary(run_dir, "shared_parent_t1", 2.00, 1)

            reloaded = frontier.FrontierStore(
                run_dir / "frontier",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 1,
                        "include_lanes": ["alpha_incubator"],
                        "axes": [("score", "maximize")],
                    }
                ],
            )
            entry = reloaded.get_manifest()["generations"]["0"][0]
            self.assertEqual(entry["source_result_path"], gen0_path)
            self.assertEqual(entry["metric_value"], 1.00)

    def test_frontier_source_repair_uses_metrics_generation_for_legacy_entries(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        def write_summary(run_dir: Path, child: str, score: float, gen_id: int) -> str:
            summary_path = run_dir / "results" / child / "tiered_eval_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "variant_name": "shared_parent",
                        "generation_id": gen_id,
                        "tier_reached": "T1",
                        "tier_status": "complete",
                        "evidence_stage": "full_T1",
                        "scored_complete": True,
                        "current_aggregate": {
                            "score": score,
                            "scored_complete": True,
                            "frontier_lane": "alpha_incubator",
                        },
                    }
                ),
                encoding="utf-8",
            )
            return str(summary_path.relative_to(run_dir)).replace("\\", "/")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen0_path = write_summary(run_dir, "shared_parent", 1.00, 0)
            write_summary(run_dir, "shared_parent_t1", 2.00, 1)
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir(parents=True)
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {
                            "0": [
                                {
                                    "finding_id": "legacy_finding",
                                    "variant_name": "shared_parent",
                                    "metric_name": "score",
                                    "metric_value": 1.00,
                                    "metric_direction": "maximize",
                                    "source_result_path": gen0_path,
                                    "metrics": {
                                        "score": 1.00,
                                        "generation_id": 0,
                                        "tier": "T1",
                                        "promotion_eligible": True,
                                        "frontier_lane": "alpha_incubator",
                                        "evidence_stage": "full_T1",
                                        "scored_complete": True,
                                        "source_result_path": gen0_path,
                                    },
                                }
                            ]
                        },
                        "frontier_lanes": [
                            {
                                "name": "alpha_incubator",
                                "k": 1,
                                "include_lanes": ["alpha_incubator"],
                                "axes": [["score", "maximize"]],
                            }
                        ],
                        "lane_frontiers": {},
                        "cumulative_top": [],
                        "metric_direction": "maximize",
                    }
                ),
                encoding="utf-8",
            )

            reloaded = frontier.FrontierStore(
                frontier_dir,
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 1,
                        "include_lanes": ["alpha_incubator"],
                        "axes": [("score", "maximize")],
                    }
                ],
            )
            entry = reloaded.get_manifest()["generations"]["0"][0]
            self.assertEqual(entry["source_result_path"], gen0_path)
            self.assertEqual(entry["metric_value"], 1.00)

    def test_frontier_source_repair_corrects_existing_future_source_leak(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        def write_summary(run_dir: Path, child: str, score: float, gen_id: int) -> str:
            summary_path = run_dir / "results" / child / "tiered_eval_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "variant_name": "shared_parent",
                        "result_variant_id": "shared_parent",
                        "generation_id": gen_id,
                        "tier_reached": "T1",
                        "tier_status": "complete",
                        "evidence_stage": "full_T1",
                        "scored_complete": True,
                        "current_aggregate": {
                            "score": score,
                            "scored_complete": True,
                            "frontier_lane": "alpha_incubator",
                        },
                    }
                ),
                encoding="utf-8",
            )
            return str(summary_path.relative_to(run_dir)).replace("\\", "/")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen0_path = write_summary(run_dir, "shared_parent", 1.00, 0)
            future_path = write_summary(run_dir, "shared_parent_full_t1", 2.00, 1)
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir(parents=True)
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {
                            "0": [
                                {
                                    "generation_id": 0,
                                    "finding_id": "legacy_future_leak",
                                    "variant_name": "shared_parent",
                                    "metric_name": "score",
                                    "metric_value": 2.00,
                                    "metric_direction": "maximize",
                                    "source_result_path": future_path,
                                    "metrics": {
                                        "score": 2.00,
                                        "generation_id": 1,
                                        "tier": "T1",
                                        "promotion_eligible": True,
                                        "frontier_lane": "alpha_incubator",
                                        "evidence_stage": "full_T1",
                                        "scored_complete": True,
                                        "source_result_path": future_path,
                                    },
                                }
                            ]
                        },
                        "frontier_lanes": [
                            {
                                "name": "alpha_incubator",
                                "k": 1,
                                "include_lanes": ["alpha_incubator"],
                                "axes": [["score", "maximize"]],
                            }
                        ],
                        "lane_frontiers": {},
                        "cumulative_top": [],
                        "metric_direction": "maximize",
                    }
                ),
                encoding="utf-8",
            )

            reloaded = frontier.FrontierStore(
                frontier_dir,
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 1,
                        "include_lanes": ["alpha_incubator"],
                        "axes": [("score", "maximize")],
                    }
                ],
            )
            entry = reloaded.get_manifest()["generations"]["0"][0]
            self.assertEqual(entry["source_result_path"], gen0_path)
            self.assertEqual(entry["selected_source_result_path"], future_path)
            self.assertEqual(entry["metric_value"], 1.00)

    def test_frontier_source_repair_reads_source_generation_from_summary_metrics(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        def write_summary(
            run_dir: Path,
            child: str,
            score: float,
            gen_id: int,
            *,
            top_level_generation: bool,
        ) -> str:
            summary_path = run_dir / "results" / child / "tiered_eval_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "variant_name": "shared_parent",
                "result_variant_id": "shared_parent",
                "tier_reached": "T1",
                "tier_status": "complete",
                "evidence_stage": "full_T1",
                "scored_complete": True,
                "metrics": {"generation_id": gen_id},
                "current_aggregate": {
                    "score": score,
                    "scored_complete": True,
                    "frontier_lane": "alpha_incubator",
                },
            }
            if top_level_generation:
                payload["generation_id"] = gen_id
            summary_path.write_text(json.dumps(payload), encoding="utf-8")
            return str(summary_path.relative_to(run_dir)).replace("\\", "/")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen0_path = write_summary(run_dir, "shared_parent", 1.00, 0, top_level_generation=True)
            future_path = write_summary(
                run_dir, "shared_parent_full_t1", 2.00, 1, top_level_generation=False
            )
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir(parents=True)
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {
                            "0": [
                                {
                                    "generation_id": 0,
                                    "finding_id": "legacy_future_metrics_generation",
                                    "variant_name": "shared_parent",
                                    "metric_name": "score",
                                    "metric_value": 2.00,
                                    "metric_direction": "maximize",
                                    "source_result_path": future_path,
                                    "metrics": {
                                        "score": 2.00,
                                        "generation_id": 1,
                                        "tier": "T1",
                                        "promotion_eligible": True,
                                        "frontier_lane": "alpha_incubator",
                                        "evidence_stage": "full_T1",
                                        "scored_complete": True,
                                        "source_result_path": future_path,
                                    },
                                }
                            ]
                        },
                        "frontier_lanes": [
                            {
                                "name": "alpha_incubator",
                                "k": 1,
                                "include_lanes": ["alpha_incubator"],
                                "axes": [["score", "maximize"]],
                            }
                        ],
                        "lane_frontiers": {},
                        "cumulative_top": [],
                        "metric_direction": "maximize",
                    }
                ),
                encoding="utf-8",
            )

            reloaded = frontier.FrontierStore(
                frontier_dir,
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 1,
                        "include_lanes": ["alpha_incubator"],
                        "axes": [("score", "maximize")],
                    }
                ],
            )
            entry = reloaded.get_manifest()["generations"]["0"][0]
            self.assertEqual(entry["source_result_path"], gen0_path)
            self.assertEqual(entry["selected_source_result_path"], future_path)
            self.assertEqual(entry["metric_value"], 1.00)

    def test_frontier_source_repair_uses_manifest_generation_key_for_legacy_entries(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        def write_summary(run_dir: Path, child: str, score: float, gen_id: int) -> str:
            summary_path = run_dir / "results" / child / "tiered_eval_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "variant_name": "shared_parent",
                        "result_variant_id": "shared_parent",
                        "generation_id": gen_id,
                        "tier_reached": "T1",
                        "tier_status": "complete",
                        "evidence_stage": "full_T1",
                        "scored_complete": True,
                        "current_aggregate": {
                            "score": score,
                            "scored_complete": True,
                            "frontier_lane": "alpha_incubator",
                        },
                    }
                ),
                encoding="utf-8",
            )
            return str(summary_path.relative_to(run_dir)).replace("\\", "/")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen0_path = write_summary(run_dir, "shared_parent", 1.00, 0)
            future_path = write_summary(run_dir, "shared_parent_full_t1", 2.00, 1)
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir(parents=True)
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {
                            "0": [
                                {
                                    "finding_id": "legacy_no_entry_generation",
                                    "variant_name": "shared_parent",
                                    "metric_name": "score",
                                    "metric_value": 2.00,
                                    "metric_direction": "maximize",
                                    "source_result_path": future_path,
                                    "metrics": {
                                        "score": 2.00,
                                        "tier": "T1",
                                        "promotion_eligible": True,
                                        "frontier_lane": "alpha_incubator",
                                        "evidence_stage": "full_T1",
                                        "scored_complete": True,
                                        "source_result_path": future_path,
                                    },
                                }
                            ]
                        },
                        "frontier_lanes": [
                            {
                                "name": "alpha_incubator",
                                "k": 1,
                                "include_lanes": ["alpha_incubator"],
                                "axes": [["score", "maximize"]],
                            }
                        ],
                        "lane_frontiers": {},
                        "cumulative_top": [],
                        "metric_direction": "maximize",
                    }
                ),
                encoding="utf-8",
            )

            reloaded = frontier.FrontierStore(
                frontier_dir,
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 1,
                        "include_lanes": ["alpha_incubator"],
                        "axes": [("score", "maximize")],
                    }
                ],
            )
            entry = reloaded.get_manifest()["generations"]["0"][0]
            self.assertEqual(entry["source_result_path"], gen0_path)
            self.assertEqual(entry["metric_value"], 1.00)

    def test_frontier_source_repair_ranks_by_lane_metric_not_global_primary(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        def write_summary(run_dir: Path, child: str, score: float, risk: float) -> str:
            summary_path = run_dir / "results" / child / "tiered_eval_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "variant_name": "risk_parent",
                        "generation_id": 0,
                        "tier_reached": "T1",
                        "tier_status": "complete",
                        "evidence_stage": "full_T1",
                        "scored_complete": True,
                        "current_aggregate": {
                            "score": score,
                            "risk": risk,
                            "scored_complete": True,
                            "frontier_lane": "risk_lane",
                        },
                    }
                ),
                encoding="utf-8",
            )
            return str(summary_path.relative_to(run_dir)).replace("\\", "/")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            low_risk_path = write_summary(run_dir, "risk_parent", 0.60, 0.10)
            high_score_path = write_summary(run_dir, "risk_parent_full_t1", 1.40, 0.80)
            store = frontier.FrontierStore(
                run_dir / "frontier",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "risk_lane",
                        "k": 1,
                        "include_lanes": ["risk_lane"],
                        "axes": [("risk", "minimize")],
                    }
                ],
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "risk_parent",
                        "finding_type": "result",
                        "variant_name": "risk_parent",
                        "metrics": {
                            "score": 0.60,
                            "risk": 0.10,
                            "tier": "T1",
                            "promotion_eligible": True,
                            "frontier_lane": "risk_lane",
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                            "source_result_path": low_risk_path,
                        },
                    }
                ],
            )
            self.assertEqual(promoted[0]["source_result_path"], low_risk_path)
            self.assertEqual(promoted[0]["lane_metric_value"], 0.10)
            self.assertNotEqual(promoted[0]["source_result_path"], high_score_path)

    def test_frontier_source_repair_does_not_merge_real_t_suffix_variants(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        def write_summary(
            run_dir: Path,
            child: str,
            score: float,
            variant_name: str | None = None,
        ) -> str:
            summary_path = run_dir / "results" / child / "tiered_eval_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "variant_name": variant_name or child,
                        "generation_id": 0,
                        "tier_reached": "T1",
                        "tier_status": "complete",
                        "evidence_stage": "full_T1",
                        "scored_complete": True,
                        "current_aggregate": {
                            "score": score,
                            "scored_complete": True,
                            "frontier_lane": "alpha_incubator",
                        },
                    }
                ),
                encoding="utf-8",
            )
            return str(summary_path.relative_to(run_dir)).replace("\\", "/")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            adapter_t1_path = write_summary(
                run_dir, "gen0_peer0_adapter_t1", 0.80, "gen0_peer0_adapter_t1"
            )
            adapter_t2_path = write_summary(
                run_dir, "gen0_peer1_adapter_t2", 1.40, "gen0_peer1_adapter_t2"
            )
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir(parents=True)
            entries = [
                {
                    "generation_id": 0,
                    "finding_id": "adapter_t1",
                    "variant_name": "gen0_peer0_adapter_t1",
                    "metric_name": "score",
                    "metric_value": 0.80,
                    "metric_direction": "maximize",
                    "source_result_path": adapter_t1_path,
                    "metrics": {
                        "score": 0.80,
                        "tier": "T1",
                        "promotion_eligible": True,
                        "frontier_lane": "alpha_incubator",
                        "evidence_stage": "full_T1",
                        "scored_complete": True,
                        "source_result_path": adapter_t1_path,
                    },
                },
                {
                    "generation_id": 0,
                    "finding_id": "adapter_t2",
                    "variant_name": "gen0_peer1_adapter_t2",
                    "metric_name": "score",
                    "metric_value": 1.40,
                    "metric_direction": "maximize",
                    "source_result_path": adapter_t2_path,
                    "metrics": {
                        "score": 1.40,
                        "tier": "T1",
                        "promotion_eligible": True,
                        "frontier_lane": "alpha_incubator",
                        "evidence_stage": "full_T1",
                        "scored_complete": True,
                        "source_result_path": adapter_t2_path,
                    },
                },
            ]
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {"0": entries},
                        "frontier_lanes": [
                            {
                                "name": "alpha_incubator",
                                "k": 2,
                                "cumulative_cap": 2,
                                "include_lanes": ["alpha_incubator"],
                                "axes": [["score", "maximize"]],
                            }
                        ],
                        "lane_frontiers": {"alpha_incubator": entries},
                        "cumulative_top": entries,
                        "metric_direction": "maximize",
                    }
                ),
                encoding="utf-8",
            )
            store = frontier.FrontierStore(
                frontier_dir,
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 2,
                        "cumulative_cap": 2,
                        "include_lanes": ["alpha_incubator"],
                        "axes": [("score", "maximize")],
                    }
                ],
            )
            paths = {
                entry["finding_id"]: entry["source_result_path"]
                for entry in store.get_manifest()["generations"]["0"]
            }
            self.assertEqual(paths["adapter_t1"], adapter_t1_path)
            self.assertEqual(paths["adapter_t2"], adapter_t2_path)

    def test_frontier_source_repair_does_not_pool_shared_canonical_family_children(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        def write_summary(run_dir: Path, child: str, score: float) -> str:
            summary_path = run_dir / "results" / child / "tiered_eval_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "variant_name": child,
                        "canonical_variant_id": "shared_family",
                        "generation_id": 0,
                        "tier_reached": "T1",
                        "tier_status": "complete",
                        "evidence_stage": "full_T1",
                        "scored_complete": True,
                        "current_aggregate": {
                            "score": score,
                            "scored_complete": True,
                            "frontier_lane": "alpha_incubator",
                        },
                    }
                ),
                encoding="utf-8",
            )
            return str(summary_path.relative_to(run_dir)).replace("\\", "/")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            child_a_path = write_summary(run_dir, "child_a", 0.70)
            child_b_path = write_summary(run_dir, "child_b", 1.20)
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir(parents=True)
            entries = [
                {
                    "generation_id": 0,
                    "finding_id": "child_a",
                    "variant_name": "child_a",
                    "canonical_variant_id": "shared_family",
                    "metric_name": "score",
                    "metric_value": 0.70,
                    "metric_direction": "maximize",
                    "source_result_path": child_a_path,
                    "metrics": {
                        "score": 0.70,
                        "canonical_variant_id": "shared_family",
                        "tier": "T1",
                        "promotion_eligible": True,
                        "frontier_lane": "alpha_incubator",
                        "evidence_stage": "full_T1",
                        "scored_complete": True,
                        "source_result_path": child_a_path,
                    },
                },
                {
                    "generation_id": 0,
                    "finding_id": "child_b",
                    "variant_name": "child_b",
                    "canonical_variant_id": "shared_family",
                    "metric_name": "score",
                    "metric_value": 1.20,
                    "metric_direction": "maximize",
                    "source_result_path": child_b_path,
                    "metrics": {
                        "score": 1.20,
                        "canonical_variant_id": "shared_family",
                        "tier": "T1",
                        "promotion_eligible": True,
                        "frontier_lane": "alpha_incubator",
                        "evidence_stage": "full_T1",
                        "scored_complete": True,
                        "source_result_path": child_b_path,
                    },
                },
            ]
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {"0": entries},
                        "frontier_lanes": [
                            {
                                "name": "alpha_incubator",
                                "k": 2,
                                "cumulative_cap": 2,
                                "include_lanes": ["alpha_incubator"],
                                "axes": [["score", "maximize"]],
                            }
                        ],
                        "lane_frontiers": {"alpha_incubator": entries},
                        "cumulative_top": entries,
                        "metric_direction": "maximize",
                    }
                ),
                encoding="utf-8",
            )

            store = frontier.FrontierStore(
                frontier_dir,
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 2,
                        "cumulative_cap": 2,
                        "include_lanes": ["alpha_incubator"],
                        "axes": [("score", "maximize")],
                    }
                ],
            )
            paths = {
                entry["finding_id"]: entry["source_result_path"]
                for entry in store.get_manifest()["generations"]["0"]
            }
            self.assertEqual(paths["child_a"], child_a_path)
            self.assertEqual(paths["child_b"], child_b_path)

    def test_frontier_source_repair_prefers_top_level_source_path_over_nested_metrics(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        def write_summary(run_dir: Path, child: str, score: float) -> str:
            summary_path = run_dir / "results" / child / "tiered_eval_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "variant_name": child,
                        "generation_id": 0,
                        "tier_reached": "T1",
                        "tier_status": "complete",
                        "evidence_stage": "full_T1",
                        "scored_complete": True,
                        "current_aggregate": {
                            "score": score,
                            "scored_complete": True,
                            "frontier_lane": "alpha_incubator",
                        },
                    }
                ),
                encoding="utf-8",
            )
            return str(summary_path.relative_to(run_dir)).replace("\\", "/")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            child_a_path = write_summary(run_dir, "child_a", 0.70)
            child_b_path = write_summary(run_dir, "child_b", 1.20)
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir(parents=True)
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {
                            "0": [
                                {
                                    "generation_id": 0,
                                    "finding_id": "child_a",
                                    "variant_name": "child_a",
                                    "metric_name": "score",
                                    "metric_value": 0.70,
                                    "metric_direction": "maximize",
                                    "source_result_path": child_a_path,
                                    "metrics": {
                                        "score": 0.70,
                                        "tier": "T1",
                                        "promotion_eligible": True,
                                        "frontier_lane": "alpha_incubator",
                                        "evidence_stage": "full_T1",
                                        "scored_complete": True,
                                        "source_result_path": child_b_path,
                                    },
                                }
                            ]
                        },
                        "frontier_lanes": [
                            {
                                "name": "alpha_incubator",
                                "k": 1,
                                "include_lanes": ["alpha_incubator"],
                                "axes": [["score", "maximize"]],
                            }
                        ],
                        "lane_frontiers": {},
                        "cumulative_top": [],
                        "metric_direction": "maximize",
                    }
                ),
                encoding="utf-8",
            )

            store = frontier.FrontierStore(
                frontier_dir,
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 1,
                        "include_lanes": ["alpha_incubator"],
                        "axes": [("score", "maximize")],
                    }
                ],
            )
            entry = store.get_manifest()["generations"]["0"][0]
            self.assertEqual(entry["source_result_path"], child_a_path)
            self.assertEqual(entry["metric_value"], 0.70)

    def test_frontier_source_repair_ignores_selected_source_path_for_grouping(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        def write_summary(run_dir: Path, child: str, score: float) -> str:
            summary_path = run_dir / "results" / child / "tiered_eval_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "variant_name": child,
                        "generation_id": 0,
                        "tier_reached": "T1",
                        "tier_status": "complete",
                        "evidence_stage": "full_T1",
                        "scored_complete": True,
                        "current_aggregate": {
                            "score": score,
                            "scored_complete": True,
                            "frontier_lane": "alpha_incubator",
                        },
                    }
                ),
                encoding="utf-8",
            )
            return str(summary_path.relative_to(run_dir)).replace("\\", "/")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            child_a_path = write_summary(run_dir, "child_a", 0.70)
            child_b_path = write_summary(run_dir, "child_b", 1.20)
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir(parents=True)
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {
                            "0": [
                                {
                                    "generation_id": 0,
                                    "finding_id": "child_a",
                                    "variant_name": "child_a",
                                    "metric_name": "score",
                                    "metric_value": 0.70,
                                    "metric_direction": "maximize",
                                    "source_result_path": child_a_path,
                                    "selected_source_result_path": child_b_path,
                                    "metrics": {
                                        "score": 0.70,
                                        "tier": "T1",
                                        "promotion_eligible": True,
                                        "frontier_lane": "alpha_incubator",
                                        "evidence_stage": "full_T1",
                                        "scored_complete": True,
                                        "source_result_path": child_a_path,
                                        "selected_source_result_path": child_b_path,
                                    },
                                }
                            ]
                        },
                        "frontier_lanes": [
                            {
                                "name": "alpha_incubator",
                                "k": 1,
                                "include_lanes": ["alpha_incubator"],
                                "axes": [["score", "maximize"]],
                            }
                        ],
                        "lane_frontiers": {},
                        "cumulative_top": [],
                        "metric_direction": "maximize",
                    }
                ),
                encoding="utf-8",
            )

            store = frontier.FrontierStore(
                frontier_dir,
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 1,
                        "include_lanes": ["alpha_incubator"],
                        "axes": [("score", "maximize")],
                    }
                ],
            )
            entry = store.get_manifest()["generations"]["0"][0]
            self.assertEqual(entry["source_result_path"], child_a_path)
            self.assertEqual(entry["selected_source_result_path"], child_b_path)
            self.assertEqual(entry["metric_value"], 0.70)

    def test_frontier_source_repair_scores_flat_result_summary_files(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            scout_path = run_dir / "results" / "flat_parent_scout" / "result_summary.json"
            scout_path.parent.mkdir(parents=True, exist_ok=True)
            scout_path.write_text(
                json.dumps(
                    {
                        "variant_name": "flat_parent",
                        "result_variant_id": "flat_parent",
                        "generation_id": 0,
                        "score": 0.20,
                        "tier": "T1",
                        "tier_status": "complete",
                        "evidence_stage": "scout",
                        "scored_complete": False,
                        "frontier_lane": "alpha_incubator",
                    }
                ),
                encoding="utf-8",
            )
            full_path = run_dir / "results" / "flat_parent_full_t1" / "result_summary.json"
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(
                json.dumps(
                    {
                        "variant_name": "flat_parent",
                        "result_variant_id": "flat_parent",
                        "generation_id": 0,
                        "score": 0.90,
                        "tier": "T1",
                        "tier_status": "complete",
                        "evidence_stage": "full_T1",
                        "scored_complete": True,
                        "frontier_lane": "alpha_incubator",
                    }
                ),
                encoding="utf-8",
            )
            stale_rel = str(scout_path.relative_to(run_dir)).replace("\\", "/")
            full_rel = str(full_path.relative_to(run_dir)).replace("\\", "/")
            store = frontier.FrontierStore(
                run_dir / "frontier",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 1,
                        "include_lanes": ["alpha_incubator"],
                        "axes": [("score", "maximize")],
                    }
                ],
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "flat_parent",
                        "finding_type": "result",
                        "variant_name": "flat_parent",
                        "metrics": {
                            "score": 0.20,
                            "tier": "T1",
                            "promotion_eligible": True,
                            "frontier_lane": "alpha_incubator",
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                            "source_result_path": stale_rel,
                        },
                    }
                ],
            )
            self.assertEqual(promoted[0]["source_result_path"], full_rel)
            self.assertEqual(promoted[0]["metric_value"], 0.90)

    def test_lane_based_frontier_separates_alpha_benchmark_and_controls(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "frontier_lanes"
            findings = [
                {
                    "id": "alpha_good",
                    "finding_type": "result",
                    "variant_name": "alpha_good",
                    "metrics": {
                        "score": 0.1,
                        "tier": "T3",
                        "promotion_eligible": True,
                        "frontier_lane": "alpha",
                        "strategy_family": "learned_alpha",
                        "mean_active_alpha_vs_benchmark_pct": 1.2,
                        "q25_active_alpha_vs_benchmark_pct": 0.2,
                        "active_ir": 0.4,
                        "mean_active_share": 0.08,
                        "evidence_stage": "T3",
                        "scored_complete": True,
                    },
                },
                {
                    "id": "alpha_near_ew",
                    "finding_type": "result",
                    "variant_name": "alpha_near_ew",
                    "metrics": {
                        "score": 5.0,
                        "tier": "T3",
                        "promotion_eligible": True,
                        "frontier_lane": "alpha",
                        "strategy_family": "learned_alpha",
                        "mean_active_alpha_vs_benchmark_pct": 2.0,
                        "q25_active_alpha_vs_benchmark_pct": 1.0,
                        "active_ir": 0.8,
                        "mean_active_share": 0.005,
                        "evidence_stage": "T3",
                        "scored_complete": True,
                    },
                },
                {
                    "id": "benchmark_floor",
                    "finding_type": "result",
                    "variant_name": "benchmark_floor",
                    "metrics": {
                        "score": 10.0,
                        "tier": "T3",
                        "promotion_eligible": True,
                        "frontier_lane": "benchmark_floor",
                        "strategy_family": "benchmark",
                        "evidence_stage": "T3",
                        "scored_complete": True,
                    },
                },
                {
                    "id": "control",
                    "finding_type": "result",
                    "variant_name": "control",
                    "metrics": {
                        "score": 9.0,
                        "tier": "T3",
                        "promotion_eligible": True,
                        "frontier_lane": "diagnostic_control",
                        "strategy_family": "control",
                        "diagnostic_value": 7.0,
                        "evidence_stage": "T3",
                        "scored_complete": True,
                    },
                },
            ]
            store = frontier.FrontierStore(
                base,
                promote_top_k=4,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha",
                        "k": 2,
                        "include_lanes": ["alpha"],
                        "exclude_families": ["benchmark", "control"],
                        "require_metrics": [
                            "mean_active_alpha_vs_benchmark_pct",
                            "active_ir",
                            "mean_active_share",
                        ],
                        "min_metrics": {
                            "mean_active_alpha_vs_benchmark_pct": 0.0,
                            "mean_active_share": 0.03,
                        },
                        "axes": [
                            ("mean_active_alpha_vs_benchmark_pct", "maximize"),
                            ("active_ir", "maximize"),
                        ],
                    },
                    {
                        "name": "benchmark_floor",
                        "k": 1,
                        "include_lanes": ["benchmark_floor"],
                        "axes": [("score", "maximize")],
                    },
                    {
                        "name": "diagnostic_control",
                        "k": 1,
                        "include_lanes": ["diagnostic_control"],
                        "axes": [("diagnostic_value", "maximize")],
                    },
                ],
            )
            promoted = store.promote(0, findings)

            self.assertEqual(
                [(p["finding_id"], p.get("frontier_lane")) for p in promoted],
                [
                    ("alpha_good", "alpha"),
                    ("benchmark_floor", "benchmark_floor"),
                    ("control", "diagnostic_control"),
                ],
            )
            self.assertEqual(promoted[0]["lane_metric_name"], "mean_active_alpha_vs_benchmark_pct")
            self.assertEqual(promoted[0]["lane_metric_value"], 1.2)
            manifest = store.get_manifest()
            self.assertEqual(
                [p["finding_id"] for p in manifest["lane_frontiers"]["alpha"]],
                ["alpha_good"],
            )
            self.assertEqual(
                [p["frontier_lane"] for p in manifest["cumulative_top"]],
                ["alpha", "benchmark_floor", "diagnostic_control"],
            )

    def test_incubator_lane_can_opt_into_non_promotable_candidates(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        lower_tier = {
            "id": "early_alpha",
            "finding_type": "result",
            "variant_name": "early_alpha",
            "metrics": {
                "score": 0.2,
                "tier": "T1",
                "promotion_eligible": False,
                "frontier_lane": "alpha",
                "strategy_family": "learned_alpha",
                "mean_active_alpha_vs_benchmark_pct": -0.4,
                "mean_active_share": 0.02,
                "active_ir": 0.1,
                "evidence_stage": "full_T1",
                "scored_complete": True,
                "full_t1_confirmed": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            strict_store = frontier.FrontierStore(
                Path(tmp) / "strict",
                promote_top_k=4,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "confirmed_alpha",
                        "k": 2,
                        "include_lanes": ["alpha"],
                        "min_metrics": {"mean_active_alpha_vs_benchmark_pct": 0.0},
                        "axes": [("mean_active_alpha_vs_benchmark_pct", "maximize")],
                    }
                ],
            )
            self.assertEqual(strict_store.promote(0, [lower_tier]), [])

        with tempfile.TemporaryDirectory() as tmp:
            incubator_store = frontier.FrontierStore(
                Path(tmp) / "incubator",
                promote_top_k=4,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "confirmed_alpha",
                        "k": 2,
                        "include_lanes": ["alpha"],
                        "min_metrics": {"mean_active_alpha_vs_benchmark_pct": 0.0},
                        "axes": [("mean_active_alpha_vs_benchmark_pct", "maximize")],
                    },
                    {
                        "name": "alpha_incubator",
                        "k": 10,
                        "include_lanes": ["alpha"],
                        "allow_lower_tier": True,
                        "allow_non_promotable": True,
                        "require_metrics": [
                            "mean_active_alpha_vs_benchmark_pct",
                            "mean_active_share",
                        ],
                        "min_metrics": {
                            "mean_active_alpha_vs_benchmark_pct": -5.0,
                            "mean_active_share": 0.005,
                        },
                        "axes": [
                            ("mean_active_alpha_vs_benchmark_pct", "maximize"),
                            ("active_ir", "maximize"),
                            ("score", "maximize"),
                        ],
                    },
                ],
            )
            promoted = incubator_store.promote(0, [lower_tier])

            self.assertEqual(len(promoted), 1)
            self.assertEqual(promoted[0]["finding_id"], "early_alpha")
            self.assertEqual(promoted[0]["frontier_lane"], "alpha_incubator")
            self.assertEqual(promoted[0]["promoted_for_lane"], "alpha_incubator")
            self.assertEqual(promoted[0]["source_frontier_lane"], "alpha")
            metrics = promoted[0]["metrics"]
            self.assertEqual(metrics["source_frontier_lane"], "alpha")
            self.assertNotIn("lane_lower_tier_candidate", metrics)
            self.assertTrue(metrics["lane_non_promotable_candidate"])
            self.assertNotIn("candidate_tier", metrics)

            immature = {
                **lower_tier,
                "id": "immature_alpha",
                "variant_name": "immature_alpha",
                "metrics": {
                    **lower_tier["metrics"],
                    "score": 0.1,
                    "tier": "",
                    "evidence_stage": "",
                    "full_t1_confirmed": "",
                },
            }
            self.assertEqual(incubator_store.promote(1, [immature]), [])
            manifest = incubator_store.get_manifest()
            self.assertNotIn(
                "immature_alpha",
                [entry["variant_name"] for entry in manifest["cumulative_top"]],
            )
            self.assertIn(
                "immature_alpha",
                [
                    entry["variant_name"]
                    for entry in manifest["validation_candidates"]["cumulative"]
                ],
            )

    def test_confirmed_lane_can_require_clean_promotion_truthy_metrics(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        dirty = {
            "id": "dirty_alpha",
            "finding_type": "result",
            "variant_name": "dirty_alpha",
            "metrics": {
                "score": 0.9,
                "tier": "T3",
                "promotion_eligible": True,
                "clean_promotion_eligible": False,
                "frontier_lane": "alpha",
                "strategy_family": "learned_alpha",
                "mean_active_alpha_vs_benchmark_pct": 2.0,
                "mean_active_share": 0.1,
                "n_hard_constraint_violations": 1,
                "evidence_stage": "T3",
                "scored_complete": True,
            },
        }
        clean = {
            "id": "clean_alpha",
            "finding_type": "result",
            "variant_name": "clean_alpha",
            "metrics": {
                "score": 1.0,
                "tier": "T3",
                "promotion_eligible": True,
                "clean_promotion_eligible": True,
                "frontier_lane": "alpha",
                "strategy_family": "learned_alpha",
                "mean_active_alpha_vs_benchmark_pct": 1.0,
                "mean_active_share": 0.08,
                "n_hard_constraint_violations": 0,
                "evidence_stage": "T3",
                "scored_complete": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "confirmed_clean",
                promote_top_k=4,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "confirmed_alpha",
                        "k": 2,
                        "include_lanes": ["alpha"],
                        "require_truthy_metrics": [
                            "promotion_eligible",
                            "clean_promotion_eligible",
                        ],
                        "require_metrics": [
                            "mean_active_alpha_vs_benchmark_pct",
                            "mean_active_share",
                            "n_hard_constraint_violations",
                        ],
                        "max_metrics": {"n_hard_constraint_violations": 0.0},
                        "axes": [("mean_active_alpha_vs_benchmark_pct", "maximize")],
                    }
                ],
            )
            promoted = store.promote(0, [dirty, clean])

            self.assertEqual([p["finding_id"] for p in promoted], ["clean_alpha"])
            self.assertEqual(promoted[0]["frontier_lane"], "confirmed_alpha")

    def test_lane_mode_accepts_intermediate_result_findings_for_incubation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "intermediate_result",
                primary_metric="future_fitness",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 10,
                        "allow_lower_tier": True,
                        "allow_non_promotable": True,
                        "require_metrics": [
                            "mean_active_alpha_vs_benchmark_pct",
                            "mean_active_share",
                        ],
                        "min_metrics": {
                            "mean_active_alpha_vs_benchmark_pct": -10.0,
                            "mean_active_share": 0.005,
                        },
                        "axes": [
                            ("mean_active_alpha_vs_benchmark_pct", "maximize"),
                            ("mean_active_share", "maximize"),
                        ],
                    }
                ],
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "t1_intermediate",
                        "finding_type": "intermediate_result",
                        "variant_name": "t1_intermediate",
                        "metrics": {
                            "tier": "T1",
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                            "promotion_eligible": False,
                            "mean_active_alpha_vs_benchmark_pct": -0.5,
                            "mean_active_share": 0.03,
                        },
                    }
                ],
            )

            self.assertEqual(len(promoted), 1)
            self.assertEqual(promoted[0]["finding_id"], "t1_intermediate")
            self.assertEqual(promoted[0]["frontier_lane"], "alpha_incubator")
            self.assertEqual(promoted[0]["evidence_maturity_rank"], 2)

    def test_incubator_deduplicates_summary_and_result_by_variant_and_prefers_full_t1(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "incubator_dedup",
                primary_metric="future_fitness",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 3,
                        "include_lanes": ["alpha"],
                        "allow_lower_tier": True,
                        "allow_non_promotable": True,
                        "require_metrics": [
                            "mean_active_alpha_vs_benchmark_pct",
                            "mean_active_share",
                        ],
                        "min_metrics": {
                            "mean_active_alpha_vs_benchmark_pct": -10.0,
                            "mean_active_share": 0.005,
                        },
                        "axes": [("mean_active_alpha_vs_benchmark_pct", "maximize")],
                    }
                ],
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "summary_row",
                        "finding_type": "intermediate_result",
                        "variant_name": "bridge_l1_c005",
                        "metrics": {
                            "tier": "T1",
                            "promotion_eligible": False,
                            "frontier_lane": "alpha",
                            "mean_active_alpha_vs_benchmark_pct": 9.9,
                            "mean_active_share": 0.05,
                            "evidence_stage": "scout",
                            "scout_only": True,
                        },
                    },
                    {
                        "id": "canonical_result",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_c005",
                        "source_result_path": "results/bridge_l1_c005/tiered_eval_summary.json",
                        "metrics": {
                            "tier": "T1",
                            "promotion_eligible": False,
                            "frontier_lane": "alpha",
                            "source_result_path": "results/bridge_l1_c005/tiered_eval_summary.json",
                            "mean_active_alpha_vs_benchmark_pct": 6.0,
                            "mean_active_share": 0.04,
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                            "full_t1_confirmed": True,
                        },
                    },
                    {
                        "id": "distinct_variant",
                        "finding_type": "result",
                        "variant_name": "distinct_variant",
                        "metrics": {
                            "tier": "T1",
                            "promotion_eligible": False,
                            "frontier_lane": "alpha",
                            "mean_active_alpha_vs_benchmark_pct": 1.0,
                            "mean_active_share": 0.04,
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                        },
                    },
                ],
            )

            self.assertEqual(
                [p["finding_id"] for p in promoted],
                ["canonical_result", "distinct_variant"],
            )
            self.assertEqual(promoted[0]["evidence_stage"], "full_T1")
            self.assertFalse(promoted[0].get("scout_only", False))
            manifest = store.get_manifest()
            self.assertEqual(
                [p["finding_id"] for p in manifest["lane_frontiers"]["alpha_incubator"]],
                ["canonical_result", "distinct_variant"],
            )

    def test_incubator_deduplicates_family_summary_and_canonical_sweep_child(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "incubator_family_child_dedup",
                primary_metric="future_fitness",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 3,
                        "include_lanes": ["alpha"],
                        "allow_lower_tier": True,
                        "allow_non_promotable": True,
                        "require_metrics": [
                            "mean_active_alpha_vs_benchmark_pct",
                            "mean_active_share",
                        ],
                        "min_metrics": {
                            "mean_active_alpha_vs_benchmark_pct": -10.0,
                            "mean_active_share": 0.005,
                        },
                        "axes": [("mean_active_alpha_vs_benchmark_pct", "maximize")],
                    }
                ],
            )
            result_path = "results/bridge_l1_c005/tiered_eval_summary.json"
            promoted = store.promote(
                0,
                [
                    {
                        "id": "family_summary_row",
                        "finding_type": "intermediate_result",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "metrics": {
                            "tier": "T1",
                            "promotion_eligible": False,
                            "frontier_lane": "alpha",
                            "source_result_path": result_path,
                            "mean_active_alpha_vs_benchmark_pct": 9.9,
                            "mean_active_share": 0.05,
                            "evidence_stage": "scout",
                            "scout_only": True,
                        },
                    },
                    {
                        "id": "canonical_child_result",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_c005",
                        "source_result_path": result_path,
                        "metrics": {
                            "tier": "T1",
                            "promotion_eligible": False,
                            "frontier_lane": "alpha",
                            "mean_active_alpha_vs_benchmark_pct": 6.0,
                            "mean_active_share": 0.04,
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                            "full_t1_confirmed": True,
                        },
                    },
                ],
            )

            self.assertEqual([p["finding_id"] for p in promoted], ["canonical_child_result"])
            self.assertEqual(promoted[0]["frontier_entity_key"], "variant::bridge_l1_c005")
            self.assertEqual(promoted[0]["evidence_stage"], "full_T1")

    def test_incubator_preserves_distinct_sweep_children_with_shared_family_label(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "incubator_sweep_children",
                primary_metric="future_fitness",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 3,
                        "include_lanes": ["alpha"],
                        "allow_lower_tier": True,
                        "allow_non_promotable": True,
                        "require_metrics": [
                            "mean_active_alpha_vs_benchmark_pct",
                            "mean_active_share",
                        ],
                        "min_metrics": {
                            "mean_active_alpha_vs_benchmark_pct": -10.0,
                            "mean_active_share": 0.005,
                        },
                        "axes": [("mean_active_alpha_vs_benchmark_pct", "maximize")],
                    }
                ],
            )
            base_metrics = {
                "tier": "T1",
                "promotion_eligible": False,
                "frontier_lane": "alpha",
                "mean_active_share": 0.04,
                "evidence_stage": "full_T1",
                "scored_complete": True,
            }
            promoted = store.promote(
                0,
                [
                    {
                        "id": "child_c005",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "metrics": {
                            **base_metrics,
                            "mean_active_alpha_vs_benchmark_pct": 6.0,
                            "source_result_path": "results/bridge_l1_c005/tiered_eval_summary.json",
                        },
                    },
                    {
                        "id": "child_c025",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "metrics": {
                            **base_metrics,
                            "mean_active_alpha_vs_benchmark_pct": 5.0,
                            "source_result_path": "results/bridge_l1_c025/tiered_eval_summary.json",
                        },
                    },
                ],
            )

            self.assertEqual(
                [p["finding_id"] for p in promoted],
                ["child_c005", "child_c025"],
            )
            self.assertEqual(
                {p["frontier_entity_key"] for p in promoted},
                {
                    "variant::bridge_l1_c005",
                    "variant::bridge_l1_c025",
                },
            )

    def test_explicit_child_identity_overrides_parent_variant_id(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "incubator_variant_id_sweep_children",
                primary_metric="future_fitness",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 3,
                        "include_lanes": ["alpha"],
                        "allow_lower_tier": True,
                        "allow_non_promotable": True,
                        "require_metrics": [
                            "mean_active_alpha_vs_benchmark_pct",
                            "mean_active_share",
                        ],
                        "min_metrics": {
                            "mean_active_alpha_vs_benchmark_pct": -10.0,
                            "mean_active_share": 0.005,
                        },
                        "axes": [("mean_active_alpha_vs_benchmark_pct", "maximize")],
                    }
                ],
            )
            base_metrics = {
                "tier": "T1",
                "promotion_eligible": False,
                "frontier_lane": "alpha",
                "variant_id": "bridge_l1_eff_n_sweep",
                "mean_active_share": 0.04,
                "evidence_stage": "full_T1",
                "scored_complete": True,
            }
            promoted = store.promote(
                0,
                [
                    {
                        "id": "child_c005",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "metrics": {
                            **base_metrics,
                            "child_variant_id": "bridge_l1_c005",
                            "mean_active_alpha_vs_benchmark_pct": 6.0,
                            "source_result_path": "results/bridge_l1_c005/tiered_eval_summary.json",
                        },
                    },
                    {
                        "id": "child_c025",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "metrics": {
                            **base_metrics,
                            "child_variant_id": "bridge_l1_c025",
                            "mean_active_alpha_vs_benchmark_pct": 5.0,
                            "source_result_path": "results/bridge_l1_c025/tiered_eval_summary.json",
                        },
                    },
                ],
            )

            self.assertEqual(
                [p["finding_id"] for p in promoted],
                ["child_c005", "child_c025"],
            )
            self.assertEqual(
                {p["frontier_entity_key"] for p in promoted},
                {
                    "variant::bridge_l1_c005",
                    "variant::bridge_l1_c025",
                },
            )

    def test_result_artifact_child_identity_overrides_stale_persisted_entity_key(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "incubator_stale_entity_sweep_children",
                primary_metric="future_fitness",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 3,
                        "include_lanes": ["alpha"],
                        "allow_lower_tier": True,
                        "allow_non_promotable": True,
                        "require_metrics": [
                            "mean_active_alpha_vs_benchmark_pct",
                            "mean_active_share",
                        ],
                        "min_metrics": {
                            "mean_active_alpha_vs_benchmark_pct": -10.0,
                            "mean_active_share": 0.005,
                        },
                        "axes": [("mean_active_alpha_vs_benchmark_pct", "maximize")],
                    }
                ],
            )
            base_metrics = {
                "tier": "T1",
                "promotion_eligible": False,
                "frontier_lane": "alpha",
                "frontier_entity_key": "variant::bridge_l1_eff_n_sweep",
                "mean_active_share": 0.04,
                "evidence_stage": "full_T1",
                "scored_complete": True,
            }
            promoted = store.promote(
                0,
                [
                    {
                        "id": "child_c005",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "metrics": {
                            **base_metrics,
                            "mean_active_alpha_vs_benchmark_pct": 6.0,
                            "source_result_path": "results/bridge_l1_c005/tiered_eval_summary.json",
                        },
                    },
                    {
                        "id": "child_c025",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "metrics": {
                            **base_metrics,
                            "mean_active_alpha_vs_benchmark_pct": 5.0,
                            "source_result_path": "results/bridge_l1_c025/tiered_eval_summary.json",
                        },
                    },
                ],
            )

            self.assertEqual(
                [p["finding_id"] for p in promoted],
                ["child_c005", "child_c025"],
            )
            self.assertEqual(
                {p["frontier_entity_key"] for p in promoted},
                {
                    "variant::bridge_l1_c005",
                    "variant::bridge_l1_c025",
                },
            )

    def test_concrete_result_path_overrides_stale_metrics_path_and_entity_key(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "incubator_stale_metrics_path",
                primary_metric="future_fitness",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 3,
                        "include_lanes": ["alpha"],
                        "allow_lower_tier": True,
                        "allow_non_promotable": True,
                        "require_metrics": [
                            "mean_active_alpha_vs_benchmark_pct",
                            "mean_active_share",
                        ],
                        "min_metrics": {
                            "mean_active_alpha_vs_benchmark_pct": -10.0,
                            "mean_active_share": 0.005,
                        },
                        "axes": [("mean_active_alpha_vs_benchmark_pct", "maximize")],
                    }
                ],
            )
            base_metrics = {
                "tier": "T1",
                "promotion_eligible": False,
                "frontier_lane": "alpha",
                "frontier_entity_key": "variant::bridge_l1_eff_n_sweep",
                "source_result_path": ("results/bridge_l1_eff_n_sweep/tiered_eval_summary.json"),
                "mean_active_share": 0.04,
                "evidence_stage": "full_T1",
                "scored_complete": True,
            }
            promoted = store.promote(
                0,
                [
                    {
                        "id": "child_c005",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "result_path": "results/bridge_l1_c005/tiered_eval_summary.json",
                        "metrics": {
                            **base_metrics,
                            "mean_active_alpha_vs_benchmark_pct": 6.0,
                        },
                    },
                    {
                        "id": "child_c025",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "result_path": "results/bridge_l1_c025/tiered_eval_summary.json",
                        "metrics": {
                            **base_metrics,
                            "mean_active_alpha_vs_benchmark_pct": 5.0,
                        },
                    },
                ],
            )

            self.assertEqual(
                [p["finding_id"] for p in promoted],
                ["child_c005", "child_c025"],
            )
            self.assertEqual(
                {p["frontier_entity_key"] for p in promoted},
                {
                    "variant::bridge_l1_c005",
                    "variant::bridge_l1_c025",
                },
            )

    def test_result_path_child_overrides_stale_top_level_source_path(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "incubator_stale_top_level_source_path",
                primary_metric="future_fitness",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 3,
                        "include_lanes": ["alpha"],
                        "allow_lower_tier": True,
                        "allow_non_promotable": True,
                        "require_metrics": [
                            "mean_active_alpha_vs_benchmark_pct",
                            "mean_active_share",
                        ],
                        "min_metrics": {
                            "mean_active_alpha_vs_benchmark_pct": -10.0,
                            "mean_active_share": 0.005,
                        },
                        "axes": [("mean_active_alpha_vs_benchmark_pct", "maximize")],
                    }
                ],
            )
            base = {
                "finding_type": "result",
                "variant_name": "bridge_l1_eff_n_sweep",
                "source_result_path": ("results/bridge_l1_eff_n_sweep/tiered_eval_summary.json"),
                "metrics": {
                    "tier": "T1",
                    "promotion_eligible": False,
                    "frontier_lane": "alpha",
                    "frontier_entity_key": "variant::bridge_l1_eff_n_sweep",
                    "variant_id": "bridge_l1_eff_n_sweep",
                    "mean_active_share": 0.04,
                    "evidence_stage": "full_T1",
                    "scored_complete": True,
                },
            }
            promoted = store.promote(
                0,
                [
                    {
                        **base,
                        "id": "child_c005",
                        "result_path": "results/bridge_l1_c005/tiered_eval_summary.json",
                        "metrics": {
                            **base["metrics"],
                            "mean_active_alpha_vs_benchmark_pct": 6.0,
                        },
                    },
                    {
                        **base,
                        "id": "child_c025",
                        "result_path": "results/bridge_l1_c025/tiered_eval_summary.json",
                        "metrics": {
                            **base["metrics"],
                            "mean_active_alpha_vs_benchmark_pct": 5.0,
                        },
                    },
                ],
            )

            self.assertEqual([p["finding_id"] for p in promoted], ["child_c005", "child_c025"])
            self.assertEqual(
                {p["frontier_entity_key"] for p in promoted},
                {"variant::bridge_l1_c005", "variant::bridge_l1_c025"},
            )

    def test_child_id_overrides_stale_persisted_entity_key_without_result_path(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "incubator_stale_entity_child_id",
                primary_metric="future_fitness",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 3,
                        "include_lanes": ["alpha"],
                        "allow_lower_tier": True,
                        "allow_non_promotable": True,
                        "require_metrics": [
                            "mean_active_alpha_vs_benchmark_pct",
                            "mean_active_share",
                        ],
                        "min_metrics": {
                            "mean_active_alpha_vs_benchmark_pct": -10.0,
                            "mean_active_share": 0.005,
                        },
                        "axes": [("mean_active_alpha_vs_benchmark_pct", "maximize")],
                    }
                ],
            )
            base_metrics = {
                "tier": "T1",
                "promotion_eligible": False,
                "frontier_lane": "alpha",
                "frontier_entity_key": "variant::bridge_l1_eff_n_sweep",
                "candidate_entity_key": "variant::bridge_l1_eff_n_sweep",
                "variant_id": "bridge_l1_eff_n_sweep",
                "mean_active_share": 0.04,
                "evidence_stage": "full_T1",
                "scored_complete": True,
            }
            promoted = store.promote(
                0,
                [
                    {
                        "id": "child_c005",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "metrics": {
                            **base_metrics,
                            "child_id": "bridge_l1_c005",
                            "mean_active_alpha_vs_benchmark_pct": 6.0,
                        },
                    },
                    {
                        "id": "child_c025",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "metrics": {
                            **base_metrics,
                            "result_variant_id": "bridge_l1_c025",
                            "mean_active_alpha_vs_benchmark_pct": 5.0,
                        },
                    },
                ],
            )

            self.assertEqual([p["finding_id"] for p in promoted], ["child_c005", "child_c025"])
            self.assertEqual(
                {p["frontier_entity_key"] for p in promoted},
                {"variant::bridge_l1_c005", "variant::bridge_l1_c025"},
            )

    def test_trial_id_does_not_collapse_distinct_named_variants(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "incubator_trial_id_not_identity",
                primary_metric="future_fitness",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 3,
                        "include_lanes": ["alpha"],
                        "allow_lower_tier": True,
                        "allow_non_promotable": True,
                        "require_metrics": [
                            "mean_active_alpha_vs_benchmark_pct",
                            "mean_active_share",
                        ],
                        "min_metrics": {
                            "mean_active_alpha_vs_benchmark_pct": -10.0,
                            "mean_active_share": 0.005,
                        },
                        "axes": [("mean_active_alpha_vs_benchmark_pct", "maximize")],
                    }
                ],
            )
            base_metrics = {
                "tier": "T1",
                "promotion_eligible": False,
                "frontier_lane": "alpha",
                "trial_id": "7",
                "mean_active_share": 0.04,
                "evidence_stage": "full_T1",
                "scored_complete": True,
            }
            promoted = store.promote(
                0,
                [
                    {
                        "id": "variant_a",
                        "finding_type": "result",
                        "variant_name": "trial_shared_a",
                        "metrics": {
                            **base_metrics,
                            "mean_active_alpha_vs_benchmark_pct": 6.0,
                        },
                    },
                    {
                        "id": "variant_b",
                        "finding_type": "result",
                        "variant_name": "trial_shared_b",
                        "metrics": {
                            **base_metrics,
                            "mean_active_alpha_vs_benchmark_pct": 5.0,
                        },
                    },
                ],
            )

            self.assertEqual(
                [p["finding_id"] for p in promoted],
                ["variant_a", "variant_b"],
            )
            self.assertEqual(
                {p["frontier_entity_key"] for p in promoted},
                {"variant::trial_shared_a", "variant::trial_shared_b"},
            )

    def test_frontier_entity_key_and_incomplete_maturity_are_stable(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        self.assertEqual(
            frontier._candidate_entity_key(
                {
                    "variant_name": "bridge_l1_eff_n_sweep",
                    "frontier_entity_key": (
                        "artifact::results/bridge_l1_c005/tiered_eval_summary.json"
                    ),
                }
            ),
            "variant::bridge_l1_c005",
        )
        self.assertEqual(
            frontier._candidate_entity_key(
                {
                    "variant_name": "bridge_l1_eff_n_sweep",
                    "metrics": {
                        "frontier_entity_key": "variant::bridge_l1_c005",
                        "sweep_child_id": "bridge_l1_eff_n_sweep",
                    },
                }
            ),
            "variant::bridge_l1_eff_n_sweep",
        )
        self.assertEqual(
            frontier._candidate_entity_key(
                {
                    "id": "malformed_without_variant",
                    "finding_type": "result",
                    "metrics": {"score": 1.0},
                }
            ),
            "finding::malformed_without_variant",
        )
        self.assertEqual(
            frontier._candidate_entity_key(
                {
                    "id": "malformed_string_none",
                    "finding_type": "result",
                    "variant_name": "None",
                    "result_path": "results/None/tiered_eval_summary.json",
                    "metrics": {
                        "variant_id": "null",
                        "frontier_entity_key": "variant::None",
                    },
                }
            ),
            "finding::malformed_string_none",
        )
        self.assertTrue(
            frontier._candidate_entity_key(
                {
                    "id": "None",
                    "finding_id": "null",
                    "finding_type": "result",
                    "variant_name": "unknown",
                    "metrics": {"variant_id": "n/a"},
                }
            ).startswith("object::")
        )
        self.assertEqual(
            frontier._candidate_entity_key(
                {
                    "id": "child_id_result",
                    "finding_type": "result",
                    "variant_name": "bridge_l1_eff_n_sweep",
                    "metrics": {
                        "frontier_entity_key": "variant::bridge_l1_eff_n_sweep",
                        "variant_id": "bridge_l1_eff_n_sweep",
                        "child_id": "bridge_l1_c005",
                    },
                }
            ),
            "variant::bridge_l1_c005",
        )

        partial = frontier._evidence_metadata_from_candidate(
            {
                "variant_name": "partial_child",
                "metrics": {
                    "tier": "T1",
                    "scored_complete": False,
                    "partial_cohort": True,
                },
            }
        )
        self.assertEqual(partial["evidence_stage"], "scout")
        self.assertEqual(partial["evidence_maturity_rank"], 1)

        unscored = frontier._evidence_metadata_from_candidate(
            {
                "variant_name": "unscored_child",
                "metrics": {
                    "tier": "T1",
                    "scored_complete": False,
                    "unscored_artifact": True,
                },
            }
        )
        self.assertEqual(unscored["evidence_stage"], "smoke")
        self.assertEqual(unscored["evidence_maturity_rank"], 0)

        explicit_incomplete = frontier._evidence_metadata_from_candidate(
            {
                "variant_name": "explicit_incomplete",
                "metrics": {
                    "tier": "T1",
                    "evidence_stage": "incomplete",
                },
            }
        )
        self.assertEqual(explicit_incomplete["evidence_stage"], "scout")
        self.assertEqual(explicit_incomplete["evidence_maturity_rank"], 1)

        explicit_unscored = frontier._evidence_metadata_from_candidate(
            {
                "variant_name": "explicit_unscored",
                "metrics": {
                    "tier": "T1",
                    "evidence_stage": "unscored",
                },
            }
        )
        self.assertEqual(explicit_unscored["evidence_stage"], "smoke")
        self.assertEqual(explicit_unscored["evidence_maturity_rank"], 0)

        contradictory_partial = frontier._evidence_metadata_from_candidate(
            {
                "variant_name": "contradictory_partial",
                "metrics": {
                    "tier": "T1",
                    "evidence_stage": "full_T1",
                    "scored_complete": False,
                    "partial_cohort": True,
                },
            }
        )
        self.assertEqual(contradictory_partial["evidence_stage"], "scout")
        self.assertEqual(contradictory_partial["evidence_maturity_rank"], 1)

        partial_without_scored_complete = frontier._evidence_metadata_from_candidate(
            {
                "variant_name": "partial_without_scored_complete",
                "metrics": {
                    "tier": "T1",
                    "evidence_stage": "full_T1",
                    "scored_complete": True,
                    "partial_cohort": True,
                },
            }
        )
        self.assertEqual(partial_without_scored_complete["evidence_stage"], "scout")
        self.assertEqual(partial_without_scored_complete["evidence_maturity_rank"], 1)

        contradictory_unscored = frontier._evidence_metadata_from_candidate(
            {
                "variant_name": "contradictory_unscored",
                "metrics": {
                    "tier": "T1",
                    "evidence_stage": "full_T1",
                    "scored_complete": True,
                    "unscored_artifact": True,
                },
            }
        )
        self.assertEqual(contradictory_unscored["evidence_stage"], "smoke")
        self.assertEqual(contradictory_unscored["evidence_maturity_rank"], 0)

        real_t1 = frontier._evidence_metadata_from_candidate(
            {
                "variant_name": "real_t1",
                "metrics": {
                    "tier": "T1",
                    "evidence_stage": "full_T1",
                    "scored_complete": True,
                },
            }
        )
        self.assertEqual(real_t1["evidence_stage"], "scored_complete")
        self.assertEqual(real_t1["evidence_maturity_rank"], 2)

        capped_status_text = frontier._evidence_metadata_from_candidate(
            {
                "variant_name": "capped_status_text",
                "metrics": {
                    "tier": "T1",
                    "tier_status": "capped_at_T1",
                },
            }
        )
        self.assertEqual(capped_status_text["evidence_stage"], "scout")
        self.assertEqual(capped_status_text["evidence_maturity_rank"], 1)

        summary_status_text = frontier._evidence_metadata_from_candidate(
            {
                "variant_name": "summary_status_text",
                "metrics": {
                    "tier": "T1",
                    "tier_status": "summary-only",
                },
            }
        )
        self.assertEqual(summary_status_text["evidence_stage"], "scout")
        self.assertEqual(summary_status_text["evidence_maturity_rank"], 1)

        not_scored_status_text = frontier._evidence_metadata_from_candidate(
            {
                "variant_name": "not_scored_status_text",
                "metrics": {
                    "tier": "T1",
                    "tier_status": "scored_complete=false",
                },
            }
        )
        self.assertEqual(not_scored_status_text["evidence_stage"], "scout")
        self.assertEqual(not_scored_status_text["evidence_maturity_rank"], 1)

        for payload in (
            {"status": "unscored_artifact"},
            {"completion_status": "incomplete"},
            {"metrics": {"is_incomplete_eval": True}},
            {"metrics": {"complete_eval": False}},
            {"metrics": {"partial_eval": True}},
            {"metrics": {"scout_only": False, "is_incomplete_eval": True}},
            {"metrics": {"scored_complete": True, "complete_eval": False}},
            {"metrics": {"is_smoke_eval": False, "unscored_artifact": True}},
            {"metrics": {"scout_only": False}, "scout_only": True},
            {"metrics": {"scored_complete": True}, "scored_complete": False},
            {
                "details": {"unscored_artifact": False},
                "metrics": {"unscored_artifact": True},
            },
        ):
            self.assertTrue(frontier._is_preliminary_or_incomplete_evidence(payload))
            self.assertIn(frontier._normalized_evidence_stage(payload), {"smoke", "scout"})

    def test_legacy_frontier_does_not_collapse_fake_none_identity(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "legacy_fake_none_identity",
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "malformed_a",
                        "finding_type": "result",
                        "variant_name": "None",
                        "result_path": "results/None/tiered_eval_summary.json",
                        "metrics": {
                            "score": 10.0,
                            "variant_id": "null",
                            "frontier_entity_key": "variant::None",
                            "scored_complete": True,
                        },
                    },
                    {
                        "id": "malformed_b",
                        "finding_type": "result",
                        "variant_name": "n/a",
                        "summary_path": "results/null/tiered_eval_summary.json",
                        "metrics": {
                            "score": 9.0,
                            "variant_id": "undefined",
                            "candidate_entity_key": "artifact::results/None/tiered_eval_summary.json",
                            "scored_complete": True,
                        },
                    },
                ],
            )

            self.assertEqual([p["finding_id"] for p in promoted], ["malformed_a", "malformed_b"])
            self.assertEqual(
                [p["frontier_entity_key"] for p in promoted],
                ["finding::malformed_a", "finding::malformed_b"],
            )

    def test_lower_maturity_lane_candidate_cannot_dominate_full_t1(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "maturity_pareto",
                primary_metric="future_fitness",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 1,
                        "include_lanes": ["alpha"],
                        "allow_lower_tier": True,
                        "allow_non_promotable": True,
                        "require_metrics": [
                            "mean_active_alpha_vs_benchmark_pct",
                            "mean_active_share",
                        ],
                        "min_metrics": {
                            "mean_active_alpha_vs_benchmark_pct": -10.0,
                            "mean_active_share": 0.005,
                        },
                        "axes": [("mean_active_alpha_vs_benchmark_pct", "maximize")],
                    }
                ],
            )
            base_metrics = {
                "tier": "T1",
                "promotion_eligible": False,
                "frontier_lane": "alpha",
                "mean_active_share": 0.04,
            }
            promoted = store.promote(
                0,
                [
                    {
                        "id": "scout_high_metric",
                        "finding_type": "intermediate_result",
                        "variant_name": "scout_high_metric",
                        "metrics": {
                            **base_metrics,
                            "mean_active_alpha_vs_benchmark_pct": 99.0,
                            "evidence_stage": "scout",
                            "scout_only": True,
                        },
                    },
                    {
                        "id": "full_t1_lower_metric",
                        "finding_type": "result",
                        "variant_name": "full_t1_lower_metric",
                        "metrics": {
                            **base_metrics,
                            "mean_active_alpha_vs_benchmark_pct": 1.0,
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                            "full_t1_confirmed": True,
                        },
                    },
                ],
            )

            self.assertEqual([p["finding_id"] for p in promoted], ["full_t1_lower_metric"])
            self.assertEqual(promoted[0]["evidence_stage"], "full_T1")
            self.assertEqual(promoted[0]["evidence_maturity_rank"], 2)

    def test_require_falsey_metrics_rejects_truthy_but_allows_missing(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        base_metrics = {
            "tier": "T1",
            "promotion_eligible": False,
            "frontier_lane": "alpha",
            "strategy_family": "learned_alpha",
            "mean_active_alpha_vs_benchmark_pct": -0.5,
            "mean_active_share": 0.03,
            "evidence_stage": "full_T1",
            "scored_complete": True,
            "full_t1_confirmed": True,
        }

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "falsey_optional_guard",
                primary_metric="future_fitness",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 10,
                        "include_lanes": ["alpha"],
                        "allow_lower_tier": True,
                        "allow_non_promotable": True,
                        "require_metrics": [
                            "mean_active_alpha_vs_benchmark_pct",
                            "mean_active_share",
                        ],
                        "require_falsey_metrics": ["is_smoke_eval"],
                        "min_metrics": {
                            "mean_active_alpha_vs_benchmark_pct": -10.0,
                            "mean_active_share": 0.005,
                        },
                        "axes": [("mean_active_alpha_vs_benchmark_pct", "maximize")],
                    }
                ],
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "missing_smoke_flag",
                        "finding_type": "result",
                        "variant_name": "missing_smoke_flag",
                        "metrics": dict(base_metrics),
                    },
                    {
                        "id": "explicit_smoke",
                        "finding_type": "result",
                        "variant_name": "explicit_smoke",
                        "metrics": {**base_metrics, "is_smoke_eval": True},
                    },
                ],
            )

            self.assertEqual([p["finding_id"] for p in promoted], ["missing_smoke_flag"])

    def test_cumulative_lane_frontier_deduplicates_same_variant_across_generations(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "cumulative_variant_dedup",
                primary_metric="future_fitness",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 2,
                        "cumulative_cap": 4,
                        "include_lanes": ["alpha"],
                        "allow_lower_tier": True,
                        "allow_non_promotable": True,
                        "require_metrics": [
                            "mean_active_alpha_vs_benchmark_pct",
                            "mean_active_share",
                        ],
                        "min_metrics": {
                            "mean_active_alpha_vs_benchmark_pct": -10.0,
                            "mean_active_share": 0.005,
                        },
                        "axes": [("mean_active_alpha_vs_benchmark_pct", "maximize")],
                    }
                ],
            )
            base_metrics = {
                "tier": "T1",
                "promotion_eligible": False,
                "frontier_lane": "alpha",
                "mean_active_alpha_vs_benchmark_pct": 2.0,
                "mean_active_share": 0.04,
            }
            store.promote(
                0,
                [
                    {
                        "id": "same_variant_scout",
                        "finding_type": "intermediate_result",
                        "variant_name": "same_variant",
                        "metrics": {**base_metrics, "evidence_stage": "scout", "scout_only": True},
                    }
                ],
            )
            store.promote(
                1,
                [
                    {
                        "id": "same_variant_full",
                        "finding_type": "result",
                        "variant_name": "same_variant",
                        "metrics": {
                            **base_metrics,
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                        },
                    }
                ],
            )

            manifest = store.get_manifest()
            self.assertEqual(
                [p["finding_id"] for p in manifest["lane_frontiers"]["alpha_incubator"]],
                ["same_variant_full"],
            )
            self.assertEqual(
                [p["finding_id"] for p in manifest["cumulative_top"]],
                ["same_variant_full"],
            )

    def test_lane_frontier_deduplicates_gen_peer_aliases_before_topk(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "lane_alias_dedup",
                primary_metric="future_fitness",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 3,
                        "cumulative_cap": 3,
                        "include_lanes": ["alpha"],
                        "allow_lower_tier": True,
                        "allow_non_promotable": True,
                        "allow_risk_violating": True,
                        "require_metrics": [
                            "mean_active_alpha_vs_benchmark_pct",
                            "mean_active_share",
                        ],
                        "axes": [("mean_active_alpha_vs_benchmark_pct", "maximize")],
                    }
                ],
            )

            def row(finding_id: str, variant: str, alpha: float) -> dict:
                return {
                    "id": finding_id,
                    "finding_id": finding_id,
                    "finding_type": "result",
                    "variant_name": variant,
                    "metrics": {
                        "tier": "T1",
                        "evidence_stage": "full_T1",
                        "scored_complete": True,
                        "promotion_eligible": False,
                        "frontier_lane": "alpha",
                        "future_fitness": -10.0 + alpha,
                        "mean_active_alpha_vs_benchmark_pct": alpha,
                        "mean_active_share": 0.04,
                        "n_hard_constraint_violations": 1,
                    },
                }

            promoted = store.promote(
                0,
                [
                    row("root_a", "risk_adjusted_listwise_bc_target", 10.0),
                    row("alias_a", "gen0_peer5_risk_adjusted_listwise_bc_target", 9.9),
                    row("root_b", "gen0_peer2_aux_vr", 9.0),
                    row("alias_b", "gen0_peer2_aux_vr", 8.9),
                    row("drop_topk", "bc_curriculum_drop_topk", 8.0),
                ],
            )

            self.assertEqual(
                [p["finding_id"] for p in promoted],
                ["root_a", "root_b", "drop_topk"],
            )
            self.assertEqual(
                [
                    p["finding_id"]
                    for p in store.get_manifest()["lane_frontiers"]["alpha_incubator"]
                ],
                ["root_a", "root_b", "drop_topk"],
            )
            self.assertEqual(
                [
                    p["frontier_entity_key"]
                    for p in store.get_manifest()["lane_frontiers"]["alpha_incubator"]
                ],
                [
                    "variant::risk_adjusted_listwise_bc_target",
                    "variant::aux_vr",
                    "variant::bc_curriculum_drop_topk",
                ],
            )

    def test_lane_frontier_preserves_distinct_sweep_child_aliases(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "lane_sweep_aliases",
                primary_metric="future_fitness",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 3,
                        "cumulative_cap": 3,
                        "include_lanes": ["alpha"],
                        "allow_lower_tier": True,
                        "allow_non_promotable": True,
                        "allow_risk_violating": True,
                        "require_metrics": ["mean_active_alpha_vs_benchmark_pct"],
                        "axes": [("mean_active_alpha_vs_benchmark_pct", "maximize")],
                    }
                ],
            )
            base_metrics = {
                "tier": "T1",
                "evidence_stage": "full_T1",
                "scored_complete": True,
                "promotion_eligible": False,
                "frontier_lane": "alpha",
                "future_fitness": -2.0,
                "mean_active_alpha_vs_benchmark_pct": 5.0,
                "n_hard_constraint_violations": 1,
            }

            promoted = store.promote(
                0,
                [
                    {
                        "id": "c005",
                        "finding_type": "result",
                        "variant_name": "gen0_peer0_bridge_l1_c005_t1",
                        "variant_id": "bridge_l1_c005",
                        "metrics": dict(base_metrics),
                    },
                    {
                        "id": "c025",
                        "finding_type": "result",
                        "variant_name": "gen0_peer1_bridge_l1_c025_t1",
                        "variant_id": "bridge_l1_c025",
                        "metrics": {
                            **base_metrics,
                            "mean_active_alpha_vs_benchmark_pct": 4.5,
                        },
                    },
                ],
            )

            self.assertEqual(
                {p["frontier_entity_key"] for p in promoted},
                {"variant::bridge_l1_c005", "variant::bridge_l1_c025"},
            )

    def test_frontier_entity_alias_does_not_strip_unwrapped_variant_suffixes(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        self.assertEqual(
            frontier._candidate_entity_key({"variant_name": "real_t1"}),
            "variant::real_t1",
        )
        self.assertEqual(
            frontier._candidate_entity_key({"variant_name": "real"}),
            "variant::real",
        )
        self.assertEqual(
            frontier._candidate_entity_key({"variant_name": "strategy_seed1"}),
            "variant::strategy_seed1",
        )
        self.assertEqual(
            frontier._candidate_entity_key({"variant_name": "strategy_scout"}),
            "variant::strategy_scout",
        )
        self.assertEqual(
            frontier._candidate_entity_key({"variant_name": "strategy_smoke"}),
            "variant::strategy_smoke",
        )
        self.assertEqual(
            frontier._candidate_entity_key({"variant_name": "gen0_peer2_real_t1"}),
            "variant::real_t1",
        )
        self.assertEqual(
            frontier._candidate_entity_key({"variant_name": "gen0_peer2_strategy_seed1"}),
            "variant::strategy_seed1",
        )

    def test_legacy_frontier_uses_result_path_identity_for_sweep_children(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "legacy_sweep_identity",
                promote_top_k=3,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            base_metrics = {
                "variant_id": "bridge_l1_eff_n_sweep",
            }
            promoted = store.promote(
                0,
                [
                    {
                        "id": "family_summary_c005",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "metrics": {
                            **base_metrics,
                            "score": 10.0,
                            "source_result_path": (
                                "results/bridge_l1_c005/tiered_eval_summary.json"
                            ),
                            "evidence_stage": "scout",
                            "scout_only": True,
                        },
                    },
                    {
                        "id": "canonical_c005",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_c005",
                        "source_result_path": "results/bridge_l1_c005/tiered_eval_summary.json",
                        "metrics": {
                            **base_metrics,
                            "score": 6.0,
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                            "full_t1_confirmed": True,
                        },
                    },
                    {
                        "id": "canonical_c025",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "metrics": {
                            **base_metrics,
                            "score": 5.0,
                            "source_result_path": (
                                "results/bridge_l1_c025/tiered_eval_summary.json"
                            ),
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                        },
                    },
                ],
            )

            self.assertEqual(
                [p["finding_id"] for p in promoted],
                ["canonical_c005", "canonical_c025"],
            )
            self.assertEqual(
                {p["frontier_entity_key"] for p in promoted},
                {"variant::bridge_l1_c005", "variant::bridge_l1_c025"},
            )

    def test_legacy_frontier_excludes_scout_from_durable_frontier(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "legacy_maturity_order",
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "scout_high",
                        "finding_type": "result",
                        "variant_name": "scout_high",
                        "metrics": {
                            "score": "100.0",
                            "evidence_stage": "scout",
                            "scout_only": True,
                        },
                    },
                    {
                        "id": "status_unscored",
                        "finding_type": "result",
                        "variant_name": "status_unscored",
                        "metrics": {"score": 90.0},
                        "status": "unscored_artifact",
                    },
                    {
                        "id": "completion_incomplete",
                        "finding_type": "result",
                        "variant_name": "completion_incomplete",
                        "metrics": {"score": 80.0},
                        "completion_status": "incomplete",
                    },
                    {
                        "id": "incomplete_alias",
                        "finding_type": "result",
                        "variant_name": "incomplete_alias",
                        "metrics": {"score": 70.0, "is_incomplete_eval": True},
                    },
                    {
                        "id": "complete_false_alias",
                        "finding_type": "result",
                        "variant_name": "complete_false_alias",
                        "metrics": {"score": 60.0, "complete_eval": False},
                    },
                    {
                        "id": "full_low",
                        "finding_type": "result",
                        "variant_name": "full_low",
                        "metrics": {
                            "score": 1.0,
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                            "full_t1_confirmed": True,
                        },
                    },
                ],
            )

            self.assertEqual([p["finding_id"] for p in promoted], ["full_low"])
            self.assertEqual([p["evidence_maturity_rank"] for p in promoted], [2])
            self.assertEqual(promoted[0]["metrics"]["frontier_entity_key"], "variant::full_low")
            manifest = store.get_manifest()
            validation = manifest["validation_candidates"]["generations"]["0"]
            validation_ids = {p["finding_id"] for p in validation}
            self.assertEqual(
                validation_ids,
                {
                    "scout_high",
                    "status_unscored",
                    "completion_incomplete",
                    "incomplete_alias",
                    "complete_false_alias",
                },
            )
            self.assertEqual(
                validation[0]["recommended_next_step"],
                "complete_scored_validation_before_frontier_or_gems",
            )
            self.assertTrue(validation[0]["excluded_from_durable_frontier"])
            self.assertEqual(validation[0]["metric_name"], "score")
            self.assertEqual(validation[0]["metric_value"], 100.0)
            self.assertEqual(
                manifest["validation_candidates"]["cumulative"][0]["finding_id"],
                "scout_high",
            )

    def test_loaded_manifest_migrates_preliminary_durable_entries_to_validation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "frontier"
            frontier_dir.mkdir()
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {
                            "0": [
                                {
                                    "generation_id": 0,
                                    "finding_id": "old-scout",
                                    "variant_name": "old_scout",
                                    "metric_name": "score",
                                    "metric_value": 100.0,
                                    "metrics": {"score": 100.0, "evidence_stage": "scout"},
                                    "evidence_stage": "scout",
                                },
                                {
                                    "generation_id": 0,
                                    "finding_id": "old-full",
                                    "variant_name": "old_full",
                                    "metric_name": "score",
                                    "metric_value": 1.0,
                                    "metrics": {"score": 1.0, "evidence_stage": "full_T1"},
                                    "evidence_stage": "full_T1",
                                    "scored_complete": True,
                                },
                                {
                                    "generation_id": 0,
                                    "finding_id": "old-unknown",
                                    "variant_name": "old_unknown",
                                    "metric_name": "score",
                                    "metric_value": 90.0,
                                    "metrics": {"score": 90.0},
                                },
                                {
                                    "generation_id": 0,
                                    "finding_id": "uncapped-full",
                                    "variant_name": "uncapped_full",
                                    "metric_name": "score",
                                    "metric_value": 2.0,
                                    "metrics": {
                                        "score": 2.0,
                                        "evidence_stage": "full_T1",
                                        "scored_complete": True,
                                        "status": "uncapped",
                                    },
                                    "evidence_stage": "full_T1",
                                    "scored_complete": True,
                                },
                                {
                                    "generation_id": 0,
                                    "finding_id": "old-running",
                                    "variant_name": "old_running",
                                    "metric_name": "score",
                                    "metric_value": 200.0,
                                    "metrics": {
                                        "score": 200.0,
                                        "scored_complete": True,
                                        "result_status": "running",
                                    },
                                },
                                {
                                    "generation_id": 0,
                                    "finding_id": "old-summary-only",
                                    "variant_name": "old_summary_only",
                                    "metric_name": "score",
                                    "metric_value": 150.0,
                                    "metrics": {
                                        "score": 150.0,
                                        "summary_only": True,
                                        "result_status": "summary_only",
                                    },
                                },
                                {
                                    "generation_id": 0,
                                    "finding_id": "old-protocol-invalid",
                                    "variant_name": "old_protocol_invalid",
                                    "metric_name": "score",
                                    "metric_value": 250.0,
                                    "metrics": {
                                        "score": 250.0,
                                        "evidence_stage": "scout",
                                        "result_status": "protocol-invalid",
                                    },
                                },
                            ]
                        },
                        "cumulative_top": [
                            {
                                "generation_id": 0,
                                "finding_id": "old-scout",
                                "variant_name": "old_scout",
                                "metric_name": "score",
                                "metric_value": 100.0,
                                "evidence_stage": "scout",
                            },
                            {
                                "finding_id": "old-cumulative-only",
                                "variant_name": "old_cumulative_only",
                                "metric_value": 50.0,
                                "evidence_stage": "scout",
                            },
                            {
                                "generation_id": 0,
                                "finding_id": "old-full",
                                "variant_name": "old_full",
                                "metric_name": "score",
                                "metric_value": 1.0,
                                "evidence_stage": "full_T1",
                                "scored_complete": True,
                            },
                            {
                                "generation_id": 0,
                                "finding_id": "old-unknown",
                                "variant_name": "old_unknown",
                                "metric_name": "score",
                                "metric_value": 90.0,
                            },
                            {
                                "generation_id": 0,
                                "finding_id": "uncapped-full",
                                "variant_name": "uncapped_full",
                                "metric_name": "score",
                                "metric_value": 2.0,
                                "evidence_stage": "full_T1",
                                "scored_complete": True,
                                "status": "uncapped",
                            },
                            {
                                "generation_id": 0,
                                "finding_id": "old-running",
                                "variant_name": "old_running",
                                "metric_name": "score",
                                "metric_value": 200.0,
                                "metrics": {
                                    "score": 200.0,
                                    "scored_complete": True,
                                    "result_status": "running",
                                },
                            },
                            {
                                "generation_id": 0,
                                "finding_id": "old-summary-only",
                                "variant_name": "old_summary_only",
                                "metric_name": "score",
                                "metric_value": 150.0,
                                "metrics": {
                                    "score": 150.0,
                                    "summary_only": True,
                                    "result_status": "summary_only",
                                },
                            },
                            {
                                "generation_id": 0,
                                "finding_id": "old-protocol-invalid",
                                "variant_name": "old_protocol_invalid",
                                "metric_name": "score",
                                "metric_value": 250.0,
                                "metrics": {
                                    "score": 250.0,
                                    "evidence_stage": "scout",
                                    "result_status": "protocol-invalid",
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            store = frontier.FrontierStore(
                frontier_dir,
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )

            self.assertEqual(
                [entry["finding_id"] for entry in store.get_summary()],
                ["uncapped-full", "old-full"],
            )
            manifest = json.loads((frontier_dir / "frontier_manifest.json").read_text())
            self.assertEqual(
                [entry["finding_id"] for entry in manifest["generations"]["0"]],
                ["old-full", "uncapped-full"],
            )
            validation_ids = {
                entry["finding_id"] for entry in manifest["validation_candidates"]["cumulative"]
            }
            self.assertEqual(
                manifest["validation_candidates"]["cumulative"][0]["finding_id"],
                "old-scout",
            )
            self.assertIn("old-cumulative-only", validation_ids)
            self.assertIn("old-unknown", validation_ids)
            self.assertNotIn("old-running", validation_ids)
            self.assertNotIn("old-summary-only", validation_ids)
            self.assertIn("old-protocol-invalid", validation_ids)

    def test_loaded_manifest_preserves_opaque_tier_commitment_without_new_admission(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "frontier"
            frontier_dir.mkdir()
            legacy_entry = {
                "generation_id": 0,
                "finding_id": "legacy-tier-only",
                "variant_name": "legacy_tier_only",
                "metric_name": "score",
                "metric_value": 1.0,
                "metrics": {"score": 1.0, "tier": "task_stage_3"},
            }
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {"0": [legacy_entry]},
                        "cumulative_top": [legacy_entry],
                    }
                ),
                encoding="utf-8",
            )

            store = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                metric_direction="maximize",
            )
            promoted = store.promote(
                1,
                [
                    {
                        "id": "new-tier-only",
                        "finding_type": "result",
                        "variant_name": "new_tier_only",
                        "metrics": {"score": 2.0, "tier": "task_stage_3"},
                    }
                ],
            )
            manifest = store.get_manifest()

        self.assertEqual(
            [entry["finding_id"] for entry in manifest["generations"]["0"]],
            ["legacy-tier-only"],
        )
        self.assertEqual(
            [entry["finding_id"] for entry in store.get_summary()],
            ["legacy-tier-only"],
        )
        self.assertEqual(
            [entry["finding_id"] for entry in store.get_summary_for_generation(0)],
            ["legacy-tier-only"],
        )
        self.assertEqual(
            [entry["finding_id"] for entry in store.get_summary_up_to_generation(0)],
            ["legacy-tier-only"],
        )
        self.assertEqual(promoted, [])
        self.assertFalse(frontier._is_durable_frontier_entry(legacy_entry))

    def test_only_preliminary_signal_is_persisted_for_validation_not_promotion(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "only_prelim"
            store = frontier.FrontierStore(
                frontier_dir,
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "scout_high",
                        "finding_type": "result",
                        "variant_name": "scout_high",
                        "metrics": {
                            "score": 100.0,
                            "aux_bad": float("nan"),
                            "aux_bad_string": "Infinity",
                            "evidence_stage": "scout",
                            "scout_only": True,
                            "bottleneck_target": float("nan"),
                        },
                    }
                ],
            )

            self.assertEqual(promoted, [])
            manifest = json.loads((frontier_dir / "frontier_manifest.json").read_text())
            self.assertEqual(manifest["generations"], {})
            self.assertEqual(
                manifest["validation_candidates"]["generations"]["0"][0]["finding_id"],
                "scout_high",
            )
            self.assertEqual(
                manifest["validation_candidates"]["cumulative"][0]["metric_value"], 100.0
            )
            self.assertIsNone(
                manifest["validation_candidates"]["cumulative"][0]["metrics"]["aux_bad"]
            )
            self.assertIsNone(
                manifest["validation_candidates"]["cumulative"][0]["metrics"]["aux_bad_string"]
            )
            self.assertIsNone(
                manifest["validation_candidates"]["cumulative"][0]["bottleneck_target"]
            )
            self.assertNotIn("NaN", (frontier_dir / "frontier_manifest.json").read_text())
            self.assertNotIn("Infinity", (frontier_dir / "frontier_manifest.json").read_text())

    def test_task_configured_validation_signal_retains_incomplete_candidate(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "configured_validation_signal"
            store = frontier.FrontierStore(
                frontier_dir,
                promote_top_k=2,
                primary_metric="primary_score",
                metric_direction="maximize",
                validation_signal_metrics=["configured_signal"],
                require_tier=False,
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "configured_only_incomplete",
                        "finding_type": "result",
                        "variant_name": "configured_only_incomplete",
                        "metrics": {
                            "configured_signal": 7.5,
                            "result_status": "not_scored_complete",
                            "excluded_from_durable_frontier": True,
                            "exclusion_reason": "preliminary_or_incomplete_evidence",
                        },
                    }
                ],
            )

            self.assertEqual(promoted, [])
            validation = store.get_manifest()["validation_candidates"]["cumulative"]
            self.assertEqual(validation[0]["finding_id"], "configured_only_incomplete")
            self.assertEqual(validation[0]["metric_name"], "configured_signal")
            self.assertEqual(validation[0]["metric_value"], 7.5)

    def test_validation_candidate_store_preserves_full_generation_without_caps(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "wide_validation"
            store = frontier.FrontierStore(
                frontier_dir,
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": f"scout_{idx}",
                        "finding_type": "result",
                        "variant_name": f"scout_{idx}",
                        "metrics": {
                            "score": float(idx),
                            "evidence_stage": "scout",
                            "scout_only": True,
                        },
                    }
                    for idx in range(30)
                ],
            )

            self.assertEqual(promoted, [])
            manifest = json.loads((frontier_dir / "frontier_manifest.json").read_text())
            self.assertEqual(len(manifest["validation_candidates"]["generations"]["0"]), 30)
            self.assertEqual(len(manifest["validation_candidates"]["cumulative"]), 30)
            self.assertEqual(
                manifest["validation_candidates"]["generations"]["0"][0]["finding_id"],
                "scout_29",
            )

    def test_mature_nonclean_rejections_are_retained_for_validation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "mature_nonclean"
            store = frontier.FrontierStore(
                frontier_dir,
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=True,
            )

            promoted = store.promote(
                0,
                [
                    {
                        "id": "mature_missing_tier",
                        "finding_type": "result",
                        "variant_name": "mature_missing_tier",
                        "metrics": {
                            "score": 10.0,
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                            "full_t1_confirmed": True,
                        },
                    },
                    {
                        "id": "mature_non_promotable",
                        "finding_type": "result",
                        "variant_name": "mature_non_promotable",
                        "metrics": {
                            "score": 9.0,
                            "tier": "T1",
                            "promotion_eligible": False,
                            "result_status": "promotion_failed",
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                            "full_t1_confirmed": True,
                        },
                    },
                    {
                        "id": "legacy_suspect_protocol",
                        "finding_type": "result",
                        "variant_name": "legacy_suspect_protocol",
                        "metrics": {
                            "score": 8.0,
                            "tier": "T1",
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                            "full_t1_confirmed": True,
                            "suspect_fixed_weight_eval": True,
                        },
                    },
                ],
            )

            self.assertEqual(promoted, [])
            validation = store.get_manifest()["validation_candidates"]["cumulative"]
            by_id = {entry["finding_id"]: entry for entry in validation}
            self.assertEqual(
                by_id["mature_missing_tier"]["exclusion_reason"],
                "missing_required_tier_metadata",
            )
            self.assertEqual(
                by_id["mature_non_promotable"]["exclusion_reason"],
                "promotion_eligible_false",
            )
            self.assertEqual(
                by_id["legacy_suspect_protocol"]["exclusion_reason"],
                "protocol_integrity_failed",
            )
            self.assertEqual(by_id["mature_missing_tier"]["metric_value"], 10.0)
            self.assertEqual(by_id["mature_non_promotable"]["metric_value"], 9.0)

    def test_unknown_maturity_result_is_validation_candidate_not_durable_frontier(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "unknown_maturity"
            store = frontier.FrontierStore(
                frontier_dir,
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "unknown_high",
                        "finding_type": "result",
                        "variant_name": "unknown_high",
                        "metrics": {"score": 100.0},
                    }
                ],
            )

            self.assertEqual(promoted, [])
            manifest = json.loads((frontier_dir / "frontier_manifest.json").read_text())
            self.assertEqual(manifest["generations"], {})
            self.assertEqual(
                manifest["validation_candidates"]["cumulative"][0]["finding_id"],
                "unknown_high",
            )

    def test_durable_frontier_honors_persisted_maturity_when_policy_absent(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        persisted_mature = {
            "finding_type": "result",
            "variant_name": "persisted_mature",
            "metric_value": 1.0,
            "mature_enough": True,
            "maturity_basis": "effort_coverage_ratio",
            "effort_ratio": 0.2,
            "coverage_ratio": 0.2,
        }
        self.assertTrue(frontier._is_durable_frontier_entry(persisted_mature))
        self.assertTrue(
            frontier._is_durable_frontier_entry(
                {
                    "finding_type": "result",
                    "variant_name": "persisted_flag_only",
                    "metric_value": 1.0,
                    "mature_enough": True,
                    "maturity_basis": "effort_coverage_ratio",
                }
            )
        )
        self.assertFalse(
            frontier._is_durable_frontier_entry(
                {
                    "finding_type": "result",
                    "variant_name": "strict_ratio_gate_needs_ratios",
                    "metric_value": 1.0,
                    "mature_enough": True,
                    "maturity_basis": "effort_coverage_ratio",
                },
                maturity_policy={
                    "min_effort_ratio": 0.75,
                    "min_coverage_ratio": 0.80,
                    "require_ratio_gate": True,
                },
            )
        )
        self.assertFalse(
            frontier._is_durable_frontier_entry(
                persisted_mature,
                maturity_policy={
                    "min_effort_ratio": 0.75,
                    "min_coverage_ratio": 0.80,
                    "require_ratio_gate": True,
                },
            )
        )
        self.assertFalse(
            frontier._is_durable_frontier_entry(
                {
                    **persisted_mature,
                    "variant_name": "persisted_immature",
                    "mature_enough": False,
                    "effort_ratio": 1.0,
                    "coverage_ratio": 1.0,
                }
            )
        )
        self.assertFalse(
            frontier._is_durable_frontier_entry(
                {
                    **persisted_mature,
                    "variant_name": "persisted_but_partial",
                    "partial_eval": True,
                }
            )
        )

    def test_bad_runtime_status_overrides_scored_complete_for_durable_frontier(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "bad_runtime_status"
            store = frontier.FrontierStore(
                frontier_dir,
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "running_high",
                        "finding_type": "result",
                        "variant_name": "running_high",
                        "metrics": {
                            "score": 100.0,
                            "scored_complete": True,
                            "result_status": "running",
                        },
                    },
                    {
                        "id": "crashed_high",
                        "finding_type": "result",
                        "variant_name": "crashed_high",
                        "metrics": {
                            "score": 99.0,
                            "scored_complete": True,
                            "result_status": "crashed",
                        },
                    },
                ],
            )

            self.assertEqual(promoted, [])
            manifest = store.get_manifest()
            self.assertEqual(manifest["generations"], {})
            self.assertEqual(manifest["validation_candidates"]["cumulative"], [])

    def test_scored_bad_runtime_nonpromotable_signal_is_retained_for_validation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "bad_runtime_retained_validation"
            store = frontier.FrontierStore(
                frontier_dir,
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "timeout_scored_partial",
                        "finding_type": "result",
                        "variant_name": "timeout_scored_partial",
                        "metrics": {
                            "score": 100.0,
                            "scored_complete": False,
                            "result_status": "timeout",
                            "excluded_from_durable_frontier": True,
                            "exclusion_reason": "preliminary_or_incomplete_evidence",
                        },
                    }
                ],
            )

            self.assertEqual(promoted, [])
            manifest = store.get_manifest()
            self.assertEqual(manifest["generations"], {})
            validation = manifest["validation_candidates"]["cumulative"]
            self.assertEqual(validation[0]["finding_id"], "timeout_scored_partial")
            self.assertEqual(validation[0]["metric_value"], 100.0)

    def test_summary_only_result_is_not_retained_as_validation_candidate(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "summary_only"
            store = frontier.FrontierStore(
                frontier_dir,
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "summary_only_high",
                        "finding_type": "result",
                        "variant_name": "summary_only_high",
                        "metrics": {
                            "score": 100.0,
                            "summary_only": True,
                            "effort_ratio": 0.95,
                            "coverage_ratio": 0.95,
                            "result_status": "summary_only",
                        },
                    },
                    {
                        "id": "bool_summary_only_high",
                        "finding_type": "result",
                        "variant_name": "bool_summary_only_high",
                        "metrics": {
                            "score": 99.0,
                            "summary_only": True,
                            "effort_ratio": 0.95,
                            "coverage_ratio": 0.95,
                        },
                    },
                    {
                        "id": "unscored_high",
                        "finding_type": "result",
                        "variant_name": "unscored_high",
                        "metrics": {
                            "score": 98.0,
                            "unscored_artifact": True,
                            "effort_ratio": 0.95,
                            "coverage_ratio": 0.95,
                        },
                    },
                    {
                        "id": "validation_only_high",
                        "finding_type": "result",
                        "variant_name": "validation_only_high",
                        "metrics": {
                            "score": 97.0,
                            "validation_only": True,
                            "effort_ratio": 0.95,
                            "coverage_ratio": 0.95,
                        },
                    },
                ],
            )

            self.assertEqual(promoted, [])
            manifest = store.get_manifest()
            self.assertEqual(manifest["generations"], {})
            validation_names = {
                entry.get("variant_name")
                for entry in manifest["validation_candidates"]["cumulative"]
            }
            self.assertNotIn("summary_only_high", validation_names)
            self.assertNotIn("bool_summary_only_high", validation_names)
            self.assertIn("unscored_high", validation_names)
            self.assertIn("validation_only_high", validation_names)

    def test_protocol_invalid_result_is_visible_validation_candidate_not_durable_frontier(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "protocol_invalid"
            store = frontier.FrontierStore(
                frontier_dir,
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "protocol_invalid_high",
                        "finding_type": "result",
                        "variant_name": "protocol_invalid_high",
                        "metrics": {
                            "score": 100.0,
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                            "full_t1_confirmed": True,
                            "result_status": "protocol-invalid",
                        },
                    }
                ],
            )

            self.assertEqual(promoted, [])
            manifest = store.get_manifest()
            self.assertEqual(manifest["generations"], {})
            validation = manifest["validation_candidates"]["cumulative"]
            self.assertEqual(len(validation), 1)
            self.assertEqual(validation[0]["finding_id"], "protocol_invalid_high")
            self.assertTrue(validation[0]["excluded_from_durable_frontier"])
            self.assertEqual(validation[0]["exclusion_reason"], "protocol_integrity_failed")
            self.assertEqual(
                validation[0]["recommended_next_step"],
                "rerun_with_valid_evaluator_protocol",
            )
            self.assertEqual(validation[0]["metrics"]["result_status"], "protocol-invalid")

    def test_low_confidence_scored_result_is_validation_candidate_not_durable_frontier(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "low_confidence_scored",
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "low_confidence_scored",
                        "finding_type": "result",
                        "variant_name": "low_confidence_scored",
                        "metrics": {
                            "score": 100.0,
                            "scored_complete": True,
                            "source_generation_low_confidence": True,
                            "excluded_from_durable_frontier": True,
                            "exclusion_reason": "source_generation_low_confidence",
                        },
                    }
                ],
            )

            self.assertEqual(promoted, [])
            manifest = store.get_manifest()
            self.assertEqual(manifest["generations"], {})
            validation = manifest["validation_candidates"]["cumulative"]
            self.assertEqual(len(validation), 1)
            self.assertEqual(validation[0]["finding_id"], "low_confidence_scored")
            self.assertEqual(validation[0]["metric_value"], 100.0)
            self.assertEqual(
                validation[0]["exclusion_reason"],
                "source_generation_low_confidence",
            )

    def test_bad_runtime_hypothesis_is_not_retained_as_validation_candidate(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "bad_hypothesis"
            store = frontier.FrontierStore(
                frontier_dir,
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "running_hypothesis",
                        "finding_type": "hypothesis",
                        "variant_name": "running_hypothesis",
                        "metrics": {
                            "score": 100.0,
                            "result_status": "running",
                        },
                    },
                    {
                        "id": "summary_hypothesis",
                        "finding_type": "hypothesis",
                        "variant_name": "summary_hypothesis",
                        "metrics": {
                            "score": 99.0,
                            "summary_only": True,
                            "result_status": "summary_only",
                        },
                    },
                ],
            )

            self.assertEqual(promoted, [])
            manifest = store.get_manifest()
            self.assertEqual(manifest["validation_candidates"]["cumulative"], [])

    def test_durable_child_does_not_retire_sibling_validation_candidate_by_family_alias(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "broad_alias_retirement"
            store = frontier.FrontierStore(
                frontier_dir,
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            scout_finding = {
                "id": "scout-child-c005",
                "finding_type": "result",
                "variant_name": "bridge_l1_eff_n_sweep",
                "metrics": {
                    "child_variant_id": "c005",
                    "score": 9.0,
                    "evidence_stage": "scout",
                    "scout_only": True,
                    "frontier_entity_key": "variant::bridge_l1_eff_n_sweep",
                    "source_result_path": "results/c005/tiered_eval_summary.json",
                    "source_result_sha256": "c005-sha",
                },
            }
            scout_entry = store._validation_candidate_entry(gen_id=0, finding=scout_finding)
            assert scout_entry is not None
            durable_sibling = {
                "finding_id": "durable-child-c025",
                "variant_name": "bridge_l1_eff_n_sweep",
                "metric_name": "score",
                "metric_value": 10.0,
                "generation_id": 1,
                "metrics": {
                    "child_variant_id": "c025",
                    "score": 10.0,
                    "evidence_stage": "full_T1",
                    "scored_complete": True,
                    "frontier_entity_key": "variant::bridge_l1_eff_n_sweep",
                    "source_result_path": "results/c025/tiered_eval_summary.json",
                    "source_result_sha256": "c025-sha",
                },
            }
            store._manifest["validation_candidates"] = {
                "generations": {"0": [scout_entry]},
                "cumulative": [scout_entry],
                "validator_identity_aliases_by_generation": {
                    "0": scout_entry["identity_aliases"],
                },
                "validator_identity_aliases": scout_entry["identity_aliases"],
            }
            store._manifest["generations"] = {"1": [durable_sibling]}

            changed = store._retire_validation_candidates_for_durable_entities()

            self.assertFalse(changed)
            validation = store.get_manifest()["validation_candidates"]
            self.assertEqual(validation["cumulative"][0]["finding_id"], "scout-child-c005")

            durable_same_child = {
                **durable_sibling,
                "finding_id": "durable-child-c005",
                "metrics": {
                    **durable_sibling["metrics"],
                    "child_variant_id": "c005",
                    "source_result_path": "results/c005/tiered_eval_summary.json",
                    "source_result_sha256": "c005-sha",
                },
            }
            store._manifest["generations"] = {"1": [durable_same_child]}

            changed = store._retire_validation_candidates_for_durable_entities()

            self.assertTrue(changed)
            self.assertEqual(store.get_manifest()["validation_candidates"]["cumulative"], [])

    def test_durable_snapshot_does_not_retire_distinct_same_path_validation_signal(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "snapshot_scoped_retirement",
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            store.promote(
                0,
                [
                    {
                        "id": "preliminary-child-a",
                        "finding_type": "result",
                        "variant_name": "child-a",
                        "metrics": {
                            "child_variant_id": "child-a",
                            "score": 1.0,
                            "evidence_stage": "preliminary",
                            "scored_complete": False,
                            "source_result_path": "results/shared.json",
                            "source_result_sha256": "sha-a",
                        },
                    }
                ],
            )
            promoted = store.promote(
                1,
                [
                    {
                        "id": "complete-child-a-new-snapshot",
                        "finding_type": "result",
                        "variant_name": "child-a",
                        "metrics": {
                            "child_variant_id": "child-a",
                            "score": 2.0,
                            "scored_complete": True,
                            "source_result_path": "results/shared.json",
                            "source_result_sha256": "sha-b",
                        },
                    }
                ],
            )

            self.assertEqual(
                [entry["finding_id"] for entry in promoted],
                ["complete-child-a-new-snapshot"],
            )
            validation = store.get_manifest()["validation_candidates"]
            retained_ids = {entry["finding_id"] for entry in validation["generations"].get("0", [])}
            self.assertEqual(retained_ids, {"preliminary-child-a"})

            store.promote(
                2,
                [
                    {
                        "id": "complete-child-a-exact-snapshot",
                        "finding_type": "result",
                        "variant_name": "child-a",
                        "metrics": {
                            "child_variant_id": "child-a",
                            "score": 3.0,
                            "scored_complete": True,
                            "source_result_path": "results/shared.json",
                            "source_result_sha256": "sha-a",
                        },
                    }
                ],
            )
            validation = store.get_manifest()["validation_candidates"]
            self.assertEqual(validation["generations"].get("0", []), [])

    def test_durable_producer_keeps_producerless_same_snapshot_signal(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "producerless_snapshot_retirement",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            store.promote(
                0,
                [
                    {
                        "id": "producerless-preliminary",
                        "finding_type": "result",
                        "variant_name": "display-alias",
                        "metrics": {
                            "score": 1.0,
                            "evidence_stage": "preliminary",
                            "scored_complete": False,
                            "source_result_path": "results/shared.json",
                            "source_result_sha256": "shared-sha",
                        },
                    }
                ],
            )
            store.promote(
                1,
                [
                    {
                        "id": "explicit-complete",
                        "finding_type": "result",
                        "variant_name": "producer",
                        "metrics": {
                            "child_variant_id": "producer",
                            "score": 2.0,
                            "scored_complete": True,
                            "source_result_path": "results/shared.json",
                            "source_result_sha256": "shared-sha",
                        },
                    }
                ],
            )

            validation = store.get_manifest()["validation_candidates"]
            retained = validation["generations"].get("0", [])
            self.assertEqual(
                [entry["finding_id"] for entry in retained],
                ["producerless-preliminary"],
            )

    def test_current_aggregate_snapshot_survives_validation_and_exact_retirement(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        identity = {
            "child_variant_id": "child-a",
            "source_result_path": "results/child-a.json",
            "source_result_sha256": "child-a-sha",
        }
        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "current_aggregate_retirement",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            store.promote(
                0,
                [
                    {
                        "id": "preliminary-child-a",
                        "finding_type": "result",
                        "variant_name": "child-a",
                        "metrics": {"score": 1.0, "evidence_stage": "preliminary"},
                        "current_aggregate": dict(identity),
                    }
                ],
            )
            [validation] = store.get_manifest()["validation_candidates"]["cumulative"]
            self.assertEqual(validation["current_aggregate"], identity)

            store.promote(
                1,
                [
                    {
                        "id": "complete-child-a",
                        "finding_type": "result",
                        "variant_name": "child-a",
                        "metrics": {"score": 2.0, "scored_complete": True},
                        "current_aggregate": dict(identity),
                    }
                ],
            )

            self.assertEqual(store.get_manifest()["validation_candidates"]["cumulative"], [])

    def test_validation_compaction_preserves_conflicting_nested_producer_identity(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier
        from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
            result_snapshot_key,
        )

        with tempfile.TemporaryDirectory() as tmp:
            producer_a = {
                "child_id": "child-a",
                "sweep_child_id": "child-a",
                "child_variant_id": "child-a",
                "result_variant_id": "child-a",
                "canonical_variant_id": "child-a",
                "variant_id": "child-a",
                "child_variant_name": "child-a",
                "result_variant_name": "child-a",
                "canonical_variant_name": "child-a",
            }
            producer_b = {key: "child-b" for key in producer_a}
            store = frontier.FrontierStore(
                Path(tmp) / "conflicting_identity_retirement",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            store.promote(
                0,
                [
                    {
                        "id": "ambiguous-preliminary",
                        "finding_type": "result",
                        "variant_name": "parent",
                        "metrics": {
                            **producer_a,
                            "score": 1.0,
                            "evidence_stage": "preliminary",
                            "scored_complete": False,
                            "source_result_path": "results/shared.json",
                            "source_result_sha256": "shared-sha",
                        },
                        "extra": {
                            **producer_b,
                            "large_note": "not needed in compact identity context",
                        },
                    }
                ],
            )
            validation = store.get_manifest()["validation_candidates"]
            [stored] = validation["generations"]["0"]
            self.assertIsNone(result_snapshot_key(stored))
            self.assertEqual(stored["extra"], producer_b)

            store.promote(
                1,
                [
                    {
                        "id": "complete-child-a",
                        "finding_type": "result",
                        "variant_name": "child-a",
                        "metrics": {
                            **producer_a,
                            "score": 2.0,
                            "scored_complete": True,
                            "source_result_path": "results/shared.json",
                            "source_result_sha256": "shared-sha",
                        },
                    }
                ],
            )

            validation = store.get_manifest()["validation_candidates"]
            retained_ids = {entry["finding_id"] for entry in validation["generations"].get("0", [])}
            self.assertEqual(retained_ids, {"ambiguous-preliminary"})

    def test_validation_compaction_preserves_root_producer_identity(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "root_identity_retirement",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            store.promote(
                0,
                [
                    {
                        "id": "preliminary-child-a",
                        "finding_type": "result",
                        "variant_name": "shared-display",
                        "child_id": "child-a",
                        "metrics": {
                            "score": 1.0,
                            "evidence_stage": "preliminary",
                            "scored_complete": False,
                            "source_result_path": "results/shared.json",
                            "source_result_sha256": "shared-sha",
                        },
                    }
                ],
            )
            store.promote(
                1,
                [
                    {
                        "id": "complete-child-b",
                        "finding_type": "result",
                        "variant_name": "shared-display",
                        "child_id": "child-b",
                        "metrics": {
                            "score": 2.0,
                            "scored_complete": True,
                            "source_result_path": "results/shared.json",
                            "source_result_sha256": "shared-sha",
                        },
                    }
                ],
            )

            validation = store.get_manifest()["validation_candidates"]
            retained = validation["generations"]["0"]
            self.assertEqual([entry["finding_id"] for entry in retained], ["preliminary-child-a"])
            self.assertEqual(retained[0]["child_id"], "child-a")

    def test_conflicting_artifact_coordinates_do_not_fall_back_to_alias_retirement(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "conflicting_coordinates",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            store.promote(
                0,
                [
                    {
                        "id": "ambiguous-preliminary",
                        "finding_type": "result",
                        "variant_name": "candidate",
                        "child_id": "candidate",
                        "metrics": {
                            "score": 1.0,
                            "evidence_stage": "preliminary",
                            "scored_complete": False,
                            "source_result_path": "results/first.json",
                            "source_result_sha256": "first-sha",
                        },
                        "extra": {
                            "source_result_path": "results/second.json",
                            "source_result_sha256": "second-sha",
                        },
                    }
                ],
            )
            store.promote(
                1,
                [
                    {
                        "id": "complete-candidate",
                        "finding_type": "result",
                        "variant_name": "candidate",
                        "child_id": "candidate",
                        "metrics": {
                            "score": 2.0,
                            "scored_complete": True,
                            "source_result_path": "results/first.json",
                            "source_result_sha256": "first-sha",
                        },
                    }
                ],
            )

            validation = store.get_manifest()["validation_candidates"]
            retained = validation["generations"]["0"]

        self.assertEqual([entry["finding_id"] for entry in retained], ["ambiguous-preliminary"])

    def test_partial_artifact_coordinates_do_not_fall_back_to_alias_retirement(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "partial_coordinates",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            store.promote(
                0,
                [
                    {
                        "id": "path-only-preliminary",
                        "finding_type": "result",
                        "variant_name": "candidate",
                        "child_id": "candidate",
                        "metrics": {
                            "score": 1.0,
                            "evidence_stage": "preliminary",
                            "scored_complete": False,
                            "source_result_path": "results/candidate.json",
                        },
                    }
                ],
            )
            store.promote(
                1,
                [
                    {
                        "id": "complete-candidate",
                        "finding_type": "result",
                        "variant_name": "candidate",
                        "child_id": "candidate",
                        "metrics": {
                            "score": 2.0,
                            "scored_complete": True,
                            "source_result_path": "results/candidate.json",
                            "source_result_sha256": "complete-sha",
                        },
                    }
                ],
            )

            retained = store.get_manifest()["validation_candidates"]["generations"]["0"]

        self.assertEqual([entry["finding_id"] for entry in retained], ["path-only-preliminary"])

    def test_frontier_rejects_nonfinite_primary_metric_and_normalizes_direction(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "finite_primary_metric"
            store = frontier.FrontierStore(
                frontier_dir,
                promote_top_k=2,
                primary_metric="score",
                metric_direction="sideways",
                require_tier=False,
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "bad_nan",
                        "finding_type": "result",
                        "variant_name": "bad_nan",
                        "metrics": {"score": float("nan")},
                    },
                    {
                        "id": "bad_pos_inf",
                        "finding_type": "result",
                        "variant_name": "bad_pos_inf",
                        "metrics": {"score": "Infinity"},
                    },
                    {
                        "id": "bad_neg_inf",
                        "finding_type": "result",
                        "variant_name": "bad_neg_inf",
                        "metrics": {"score": float("-inf")},
                    },
                    {
                        "id": "low",
                        "finding_type": "result",
                        "variant_name": "low",
                        "metrics": {"score": 1.0, "aux_bad": float("nan"), "scored_complete": True},
                    },
                    {
                        "id": "high_string",
                        "finding_type": "result",
                        "variant_name": "high_string",
                        "metrics": {"score": "100.0", "scored_complete": True},
                    },
                ],
            )

            self.assertEqual([entry["finding_id"] for entry in promoted], ["high_string", "low"])
            self.assertEqual([entry["metric_value"] for entry in promoted], [100.0, 1.0])
            manifest = json.loads((frontier_dir / "frontier_manifest.json").read_text())
            self.assertEqual(manifest["metric_direction"], "maximize")
            self.assertEqual(
                [entry["finding_id"] for entry in manifest["generations"]["0"]],
                ["high_string", "low"],
            )
            manifest_text = (frontier_dir / "frontier_manifest.json").read_text()
            self.assertNotIn("NaN", manifest_text)
            self.assertNotIn("Infinity", manifest_text)
            finding_text = (frontier_dir / "gen_0" / "top_2_finding.json").read_text()
            self.assertNotIn("NaN", finding_text)
            self.assertIn('"aux_bad": null', finding_text)

    def test_lane_frontier_uses_lane_metric_when_primary_metric_is_nonfinite(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "lane_nonfinite_primary",
                primary_metric="future_fitness",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 1,
                        "include_lanes": ["alpha"],
                        "allow_non_promotable": True,
                        "require_metrics": [
                            "mean_active_alpha_vs_benchmark_pct",
                            "mean_active_share",
                        ],
                        "min_metrics": {
                            "mean_active_alpha_vs_benchmark_pct": -10.0,
                            "mean_active_share": 0.005,
                        },
                        "axes": [("mean_active_alpha_vs_benchmark_pct", "maximize")],
                    }
                ],
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "lane_candidate",
                        "finding_type": "result",
                        "variant_name": "lane_candidate",
                        "metrics": {
                            "future_fitness": "NaN",
                            "tier": "T1",
                            "promotion_eligible": False,
                            "frontier_lane": "alpha",
                            "mean_active_alpha_vs_benchmark_pct": 3.0,
                            "mean_active_share": 0.02,
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                        },
                    }
                ],
            )

            self.assertEqual(promoted[0]["metric_value"], 3.0)
            self.assertIsNone(promoted[0]["metrics"]["future_fitness"])
            self.assertEqual(promoted[0]["lane_metric_value"], 3.0)

    def test_validation_candidates_preserve_old_cumulative_and_lane_semantics(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "validation_candidate_lane_semantics",
                promote_top_k=2,
                primary_metric="future_fitness",
                metric_direction="maximize",
                anchor_metrics=[("anchor_score", "sideways")],
                require_tier=False,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 2,
                        "include_lanes": ["alpha"],
                        "require_metrics": [
                            "mean_active_alpha_vs_benchmark_pct",
                            "mean_active_share",
                        ],
                        "min_metrics": {
                            "mean_active_alpha_vs_benchmark_pct": -10.0,
                            "mean_active_share": 0.005,
                        },
                        "axes": [("mean_active_alpha_vs_benchmark_pct", "maximize")],
                    }
                ],
            )
            store._manifest["validation_candidates"] = {
                "generations": {},
                "cumulative": [
                    {
                        "generation_id": 0,
                        "finding_id": "old-scout",
                        "variant_name": "old_scout",
                        "metric_name": "future_fitness",
                        "metric_value": 7.0,
                        "metric_direction": "maximize",
                        "signal_source_priority": "bad-int",
                        "evidence_maturity_rank": "bad-rank",
                        "frontier_entity_key": "variant::old_scout",
                    }
                ],
            }

            promoted = store.promote(
                1,
                [
                    {
                        "id": "alpha-scout",
                        "finding_type": "result",
                        "variant_name": "alpha_scout",
                        "metrics": {
                            "future_fitness": 1.0,
                            "frontier_lane": "alpha",
                            "mean_active_alpha_vs_benchmark_pct": 9.0,
                            "mean_active_share": 0.02,
                            "evidence_stage": "scout",
                            "scout_only": True,
                        },
                    },
                    {
                        "id": "stray-scout",
                        "finding_type": "result",
                        "notes": {"display": "non-string notes must not crash"},
                        "metrics": {
                            "future_fitness": 50.0,
                            "frontier_lane": "benchmark",
                            "mean_active_alpha_vs_benchmark_pct": 99.0,
                            "mean_active_share": 0.02,
                            "evidence_stage": "scout",
                            "scout_only": True,
                        },
                    },
                    {
                        "id": "hypothesis-scout",
                        "finding_type": "hypothesis",
                        "variant_name": "hypothesis_scout",
                        "metrics": {
                            "anchor_score": "42.0",
                            "evidence_stage": "scout",
                            "scout_only": True,
                        },
                    },
                    {
                        "id": "lane-compact-scout",
                        "finding_type": "result",
                        "variant_name": "lane_compact_scout",
                        "metric_name": "mean_active_alpha_vs_benchmark_pct",
                        "metric_value": "12.0",
                        "metric_direction": "maximize",
                        "evidence_stage": "scout",
                        "scout_only": True,
                    },
                    {
                        "id": "hypothesis-unmarked",
                        "finding_type": "hypothesis",
                        "variant_name": "hypothesis_unmarked",
                        "metrics": {
                            "future_fitness": "12.0",
                        },
                    },
                    {
                        "id": "hypothesis-final-metrics",
                        "finding_type": "hypothesis",
                        "variant_name": "hypothesis_final_metrics",
                        "final_metrics": {
                            "future_fitness": "13.0",
                        },
                    },
                    {
                        "id": "scout-final-metrics",
                        "finding_type": "result",
                        "variant_name": "scout_final_metrics",
                        "metrics": {
                            "evidence_stage": "scout",
                        },
                        "final_metrics": {
                            "aggregated": {"future_fitness": "14.0"},
                        },
                    },
                    {
                        "id": "result-partial",
                        "finding_type": "result",
                        "variant_name": "result_partial",
                        "metrics": {
                            "future_fitness": 8.0,
                            "evidence_stage": "partial",
                        },
                    },
                ],
            )

            self.assertEqual(promoted, [])
            manifest = store.get_manifest()
            self.assertEqual(manifest["generations"], {})
            self.assertEqual(manifest["lane_frontiers"].get("alpha_incubator", []), [])
            validation = manifest["validation_candidates"]
            by_id = {
                entry["finding_id"]: entry
                for entry in validation["cumulative"]
                if entry.get("finding_id")
            }
            self.assertIn("old-scout", by_id)
            self.assertIn("alpha-scout", by_id)
            self.assertIn("stray-scout", by_id)
            self.assertIn("hypothesis-scout", by_id)
            self.assertIn("lane-compact-scout", by_id)
            self.assertIn("hypothesis-unmarked", by_id)
            self.assertIn("hypothesis-final-metrics", by_id)
            self.assertIn("scout-final-metrics", by_id)
            self.assertIn("result-partial", by_id)
            self.assertEqual(
                by_id["alpha-scout"]["metric_name"], "mean_active_alpha_vs_benchmark_pct"
            )
            self.assertEqual(by_id["alpha-scout"]["matched_frontier_lanes"], ["alpha_incubator"])
            self.assertEqual(by_id["stray-scout"]["matched_frontier_lanes"], [])
            self.assertEqual(by_id["stray-scout"]["signal_axis_lanes"], ["alpha_incubator"])
            self.assertIn("non-string notes", by_id["stray-scout"]["variant_name"])
            self.assertEqual(by_id["hypothesis-scout"]["metric_name"], "anchor_score")
            self.assertEqual(by_id["hypothesis-scout"]["metric_value"], 42.0)
            self.assertEqual(
                by_id["lane-compact-scout"]["metric_name"],
                "mean_active_alpha_vs_benchmark_pct",
            )
            self.assertEqual(by_id["lane-compact-scout"]["metric_value"], 12.0)
            self.assertEqual(
                by_id["lane-compact-scout"]["signal_axis_lanes"],
                ["alpha_incubator"],
            )
            self.assertEqual(by_id["hypothesis-unmarked"]["metric_name"], "future_fitness")
            self.assertEqual(by_id["hypothesis-unmarked"]["metric_value"], 12.0)
            self.assertEqual(by_id["hypothesis-final-metrics"]["metric_value"], 13.0)
            self.assertEqual(by_id["scout-final-metrics"]["metric_value"], 14.0)
            self.assertEqual(by_id["result-partial"]["evidence_stage"], "scout")

    def test_legacy_cumulative_deduplicates_same_entity_across_generations(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "legacy_cumulative_entity_dedup",
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            store.promote(
                0,
                [
                    {
                        "id": "same_variant_scout",
                        "finding_type": "result",
                        "variant_name": "same_variant",
                        "metrics": {
                            "score": 100.0,
                            "evidence_stage": "scout",
                            "scout_only": True,
                        },
                    }
                ],
            )
            store.promote(
                1,
                [
                    {
                        "id": "same_variant_full",
                        "finding_type": "result",
                        "variant_name": "same_variant",
                        "metrics": {
                            "score": 1.0,
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                            "full_t1_confirmed": True,
                        },
                    }
                ],
            )

            manifest = store.get_manifest()
            self.assertEqual(
                [p["finding_id"] for p in manifest["cumulative_top"]],
                ["same_variant_full"],
            )
            self.assertEqual(manifest["cumulative_top"][0]["evidence_maturity_rank"], 2)
            self.assertEqual(
                [entry["finding_id"] for entry in manifest["validation_candidates"]["cumulative"]],
                ["same_variant_scout"],
            )

    def test_strict_primary_metric_does_not_read_list_final_metrics(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "strict_list_final_metrics",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "list-final-metrics",
                        "finding_type": "result",
                        "variant_name": "list_final_metrics",
                        "final_metrics": [{"score": 100.0}],
                    }
                ],
            )

            self.assertEqual(promoted, [])

    def test_legacy_cumulative_preserves_top_level_result_path_identity(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "legacy_cumulative_top_level_path",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            base_metrics = {
                "frontier_entity_key": "variant::bridge_l1_eff_n_sweep",
                "evidence_stage": "full_T1",
                "scored_complete": True,
            }
            store.promote(
                0,
                [
                    {
                        "id": "child_c005",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "result_path": "results/bridge_l1_c005/tiered_eval_summary.json",
                        "metrics": {**base_metrics, "score": 10.0},
                    }
                ],
            )
            store.promote(
                1,
                [
                    {
                        "id": "child_c025",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "result_path": "results/bridge_l1_c025/tiered_eval_summary.json",
                        "metrics": {**base_metrics, "score": 9.0},
                    }
                ],
            )

            manifest = store.get_manifest()
            self.assertEqual(
                {p["frontier_entity_key"] for p in manifest["cumulative_top"]},
                {"variant::bridge_l1_c005", "variant::bridge_l1_c025"},
            )
            self.assertEqual(
                {p["metrics"]["frontier_entity_key"] for p in manifest["cumulative_top"]},
                {"variant::bridge_l1_c005", "variant::bridge_l1_c025"},
            )

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "legacy_cumulative_result_artifact_path",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            base_metrics = {
                "frontier_entity_key": "variant::bridge_l1_eff_n_sweep",
                "evidence_stage": "full_T1",
                "scored_complete": True,
            }
            store.promote(
                0,
                [
                    {
                        "id": "artifact_c005",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "result_artifact_path": ("results/bridge_l1_c005/tiered_eval_summary.json"),
                        "metrics": {**base_metrics, "score": 10.0},
                    }
                ],
            )
            store.promote(
                1,
                [
                    {
                        "id": "artifact_c025",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "result_artifact_path": ("results/bridge_l1_c025/tiered_eval_summary.json"),
                        "metrics": {**base_metrics, "score": 9.0},
                    }
                ],
            )

            manifest = store.get_manifest()
            self.assertEqual(
                {p["frontier_entity_key"] for p in manifest["cumulative_top"]},
                {"variant::bridge_l1_c005", "variant::bridge_l1_c025"},
            )
            self.assertEqual(
                {p["metrics"]["frontier_entity_key"] for p in manifest["cumulative_top"]},
                {"variant::bridge_l1_c005", "variant::bridge_l1_c025"},
            )

    def test_validation_candidate_entity_key_does_not_replace_exact_snapshot_retirement(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "validation_candidate_retire_canonical_child",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            store._manifest["validation_candidates"] = {
                "generations": {
                    "0": [
                        {
                            "generation_id": 0,
                            "finding_id": "scout-child",
                            "variant_name": "bridge_l1_eff_n_sweep",
                            "result_artifact_path": (
                                "results/bridge_l1_c005/tiered_eval_summary.json"
                            ),
                            "source_result_sha256": "c005-sha",
                            "metric_name": "score",
                            "metric_value": 5.0,
                            "metric_direction": "maximize",
                            "frontier_entity_key": "variant::bridge_l1_c005",
                            "metrics": {
                                "score": 5.0,
                                "frontier_entity_key": "variant::bridge_l1_eff_n_sweep",
                            },
                        }
                    ]
                },
                "cumulative": [],
            }
            store.promote(
                1,
                [
                    {
                        "id": "full-child",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "result_artifact_path": ("results/bridge_l1_c005/tiered_eval_summary.json"),
                        "source_result_sha256": "c005-sha",
                        "metrics": {
                            "score": 10.0,
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                            "source_result_sha256": "c005-sha",
                        },
                    }
                ],
            )

            validation = store.get_manifest().get("validation_candidates", {})
            self.assertEqual(
                [entry["finding_id"] for entry in validation.get("generations", {}).get("0", [])],
                ["scout-child"],
            )
            self.assertEqual(
                [entry["finding_id"] for entry in validation.get("cumulative", [])],
                ["scout-child"],
            )

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "validation_candidate_retire_top_level_key",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            store._manifest["validation_candidates"] = {
                "generations": {
                    "0": [
                        {
                            "generation_id": 0,
                            "finding_id": "scout-child",
                            "variant_name": "bridge_l1_eff_n_sweep",
                            "metric_name": "score",
                            "metric_value": 5.0,
                            "metric_direction": "maximize",
                            "frontier_entity_key": "variant::child_c005",
                            "metrics": {
                                "score": 5.0,
                                "frontier_entity_key": "variant::bridge_l1_eff_n_sweep",
                            },
                        }
                    ]
                },
                "cumulative": [],
            }
            store.promote(
                1,
                [
                    {
                        "id": "full-child",
                        "finding_type": "result",
                        "variant_name": "child_c005",
                        "metrics": {
                            "score": 10.0,
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                        },
                    }
                ],
            )

            validation = store.get_manifest().get("validation_candidates", {})
            self.assertEqual(
                [entry["finding_id"] for entry in validation.get("generations", {}).get("0", [])],
                ["scout-child"],
            )
            self.assertEqual(
                [entry["finding_id"] for entry in validation.get("cumulative", [])],
                ["scout-child"],
            )

    def test_unknown_aggregate_string_metric_is_retained_for_validation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "aggregate_string_signal",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "aggregate_signal",
                        "finding_type": "result",
                        "variant_name": "aggregate_signal",
                        "metrics": {"final_metrics": {"aggregated": {"score": "13.0"}}},
                    }
                ],
            )

            self.assertEqual(promoted, [])
            validation = store.get_manifest()["validation_candidates"]["cumulative"]
            self.assertEqual(validation[0]["finding_id"], "aggregate_signal")
            self.assertEqual(validation[0]["metric_value"], 13.0)

    def test_validation_candidate_runtime_dedup_merges_identity_aliases(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "validation_candidate_alias_merge",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            store.promote(
                0,
                [
                    {
                        "id": "winner_alias",
                        "finding_type": "result",
                        "variant_name": "same_candidate",
                        "metrics": {"score": 9.0, "evidence_stage": "scout"},
                    },
                    {
                        "id": "losing_alias",
                        "finding_type": "result",
                        "variant_name": "same_candidate",
                        "source_path": "results/losing_alias/summary.json",
                        "metrics": {"score": 1.0, "evidence_stage": "scout"},
                    },
                ],
            )

            validation = store.get_manifest()["validation_candidates"]
            kept_ids = [entry["finding_id"] for entry in validation["generations"]["0"]]
            self.assertEqual(kept_ids, ["winner_alias", "losing_alias"])
            self.assertIn(
                "results/losing_alias/summary.json",
                validation["generations"]["0"][1]["identity_aliases"],
            )
            self.assertIn(
                "results/losing_alias/summary.json",
                validation["validator_identity_aliases_by_generation"]["0"],
            )

    def test_legacy_result_path_child_overrides_stale_top_level_source_path(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "legacy_stale_top_level_source_path",
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            base = {
                "finding_type": "result",
                "variant_name": "bridge_l1_eff_n_sweep",
                "source_result_path": ("results/bridge_l1_eff_n_sweep/tiered_eval_summary.json"),
                "metrics": {
                    "frontier_entity_key": "variant::bridge_l1_eff_n_sweep",
                    "variant_id": "bridge_l1_eff_n_sweep",
                    "evidence_stage": "full_T1",
                    "scored_complete": True,
                },
            }
            promoted = store.promote(
                0,
                [
                    {
                        **base,
                        "id": "child_c005",
                        "result_path": "results/bridge_l1_c005/tiered_eval_summary.json",
                        "metrics": {**base["metrics"], "score": 10.0},
                    },
                    {
                        **base,
                        "id": "child_c025",
                        "result_path": "results/bridge_l1_c025/tiered_eval_summary.json",
                        "metrics": {**base["metrics"], "score": 9.0},
                    },
                ],
            )

            self.assertEqual([p["finding_id"] for p in promoted], ["child_c005", "child_c025"])
            self.assertEqual(
                [p["frontier_entity_key"] for p in promoted],
                ["variant::bridge_l1_c005", "variant::bridge_l1_c025"],
            )

    def test_legacy_child_id_overrides_stale_persisted_entity_key(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "legacy_stale_entity_child_id",
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            base_metrics = {
                "frontier_entity_key": "variant::bridge_l1_eff_n_sweep",
                "candidate_entity_key": "variant::bridge_l1_eff_n_sweep",
                "variant_id": "bridge_l1_eff_n_sweep",
                "evidence_stage": "full_T1",
                "scored_complete": True,
            }
            promoted = store.promote(
                0,
                [
                    {
                        "id": "child_c005",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "metrics": {
                            **base_metrics,
                            "score": 10.0,
                            "child_id": "bridge_l1_c005",
                        },
                    },
                    {
                        "id": "child_c025",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "metrics": {
                            **base_metrics,
                            "score": 9.0,
                            "result_variant_id": "bridge_l1_c025",
                        },
                    },
                ],
            )

            self.assertEqual([p["finding_id"] for p in promoted], ["child_c005", "child_c025"])
            self.assertEqual(
                [p["frontier_entity_key"] for p in promoted],
                ["variant::bridge_l1_c005", "variant::bridge_l1_c025"],
            )

    def test_legacy_anchor_selection_keeps_distinct_result_path_children(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "legacy_anchor_sweep_identity",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                anchor_metrics=[("cost", "minimize")],
                require_tier=False,
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "child_high_score",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "metrics": {
                            "score": 10.0,
                            "cost": 10.0,
                            "variant_id": "bridge_l1_eff_n_sweep",
                            "scored_complete": True,
                            "source_result_path": (
                                "results/bridge_l1_c005/tiered_eval_summary.json"
                            ),
                        },
                    },
                    {
                        "id": "child_low_cost",
                        "finding_type": "result",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "metrics": {
                            "score": 9.0,
                            "cost": 1.0,
                            "variant_id": "bridge_l1_eff_n_sweep",
                            "scored_complete": True,
                            "source_result_path": (
                                "results/bridge_l1_c025/tiered_eval_summary.json"
                            ),
                        },
                    },
                ],
            )

            self.assertEqual(
                [p["finding_id"] for p in promoted],
                ["child_high_score", "child_low_cost"],
            )
            self.assertEqual(promoted[1]["promoted_for_anchor"], "cost")
            self.assertEqual(
                [p["frontier_entity_key"] for p in promoted],
                ["variant::bridge_l1_c005", "variant::bridge_l1_c025"],
            )

    def test_legacy_anchor_selection_prefers_mature_anchor_evidence(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "legacy_anchor_maturity",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                anchor_metrics=[("cost", "minimize")],
                require_tier=False,
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "primary",
                        "finding_type": "result",
                        "variant_name": "primary",
                        "metrics": {
                            "score": 10.0,
                            "cost": 10.0,
                            "evidence_stage": "T3",
                            "scored_complete": True,
                        },
                    },
                    {
                        "id": "scout_low_cost",
                        "finding_type": "result",
                        "variant_name": "scout_low_cost",
                        "metrics": {
                            "score": 1.0,
                            "cost": 0.1,
                            "evidence_stage": "scout",
                            "scout_only": True,
                        },
                    },
                    {
                        "id": "full_higher_cost",
                        "finding_type": "result",
                        "variant_name": "full_higher_cost",
                        "metrics": {
                            "score": 1.0,
                            "cost": 1.0,
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                            "full_t1_confirmed": True,
                        },
                    },
                ],
            )

            self.assertEqual(
                [p["finding_id"] for p in promoted],
                ["primary", "full_higher_cost"],
            )
            self.assertEqual(promoted[1]["promoted_for_anchor"], "cost")
            self.assertEqual(promoted[1]["evidence_maturity_rank"], 2)

    def test_legacy_frontier_does_not_promote_intermediate_results(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "legacy_intermediate_result",
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "legacy_intermediate",
                        "finding_type": "intermediate_result",
                        "variant_name": "legacy_intermediate",
                        "metrics": {"score": 10.0},
                    }
                ],
            )

            self.assertEqual(promoted, [])

    def test_lane_mode_accepts_missing_primary_metric_and_optional_axes(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "optional_axes",
                promote_top_k=4,
                primary_metric="future_fitness",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 10,
                        "exclude_roles": ["theorist", "falsifier"],
                        "allow_lower_tier": True,
                        "allow_non_promotable": True,
                        "require_metrics": [
                            "mean_active_alpha_vs_benchmark_pct",
                            "mean_active_share",
                        ],
                        "min_metrics": {
                            "mean_active_alpha_vs_benchmark_pct": -10.0,
                            "mean_active_share": 0.005,
                        },
                        "axes": [
                            ("mean_active_alpha_vs_benchmark_pct", "maximize"),
                            ("mean_active_share", "maximize"),
                        ],
                        "optional_axes": [
                            ("active_ir", "maximize"),
                            ("future_fitness", "maximize"),
                        ],
                    }
                ],
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "incubate_without_ff",
                        "finding_type": "result",
                        "variant_name": "incubate_without_ff",
                        "metrics": {
                            "tier": "T1",
                            "promotion_eligible": False,
                            "strategy_family": "learned_alpha",
                            "mean_active_alpha_vs_benchmark_pct": -1.0,
                            "mean_active_share": 0.02,
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                            "full_t1_confirmed": True,
                        },
                        "extra": {"peer_role": "exploit"},
                    },
                    {
                        "id": "theory_same_metrics",
                        "finding_type": "result",
                        "variant_name": "theory_same_metrics",
                        "metrics": {
                            "tier": "T1",
                            "promotion_eligible": False,
                            "strategy_family": "learned_alpha",
                            "mean_active_alpha_vs_benchmark_pct": 9.0,
                            "mean_active_share": 0.2,
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                            "full_t1_confirmed": True,
                        },
                        "extra": {"peer_role": "theorist"},
                    },
                ],
            )

            self.assertEqual([p["finding_id"] for p in promoted], ["incubate_without_ff"])
            self.assertEqual(promoted[0]["frontier_lane"], "alpha_incubator")
            self.assertNotIn("future_fitness", promoted[0]["metrics"])

    def test_lane_metric_direction_is_persisted_for_minimize_lane_picks(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "minimize_lane",
                promote_top_k=4,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "performance",
                        "k": 1,
                        "axes": [("loss", "minimize")],
                    }
                ],
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "low_loss",
                        "finding_type": "result",
                        "variant_name": "low_loss",
                        "metrics": {
                            "tier": "T1",
                            "loss": 0.2,
                            "promotion_eligible": True,
                            "scored_complete": True,
                        },
                    }
                ],
            )

        self.assertEqual(len(promoted), 1)
        self.assertEqual(promoted[0]["lane_metric_direction"], "minimize")
        self.assertEqual(promoted[0]["metric_direction"], "minimize")
        self.assertEqual(promoted[0]["metrics"]["lane_metric_direction"], "minimize")

    def test_lane_metrics_do_not_recurse_into_per_task_values(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "no_nested_lane_metrics",
                primary_metric="future_fitness",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 10,
                        "include_lanes": ["alpha"],
                        "allow_lower_tier": True,
                        "allow_non_promotable": True,
                        "require_metrics": [
                            "mean_active_alpha_vs_benchmark_pct",
                            "mean_active_share",
                        ],
                        "min_metrics": {
                            "mean_active_alpha_vs_benchmark_pct": -10.0,
                            "mean_active_share": 0.005,
                        },
                        "axes": [("mean_active_alpha_vs_benchmark_pct", "maximize")],
                    }
                ],
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "nested_only",
                        "finding_type": "result",
                        "variant_name": "nested_only",
                        "metrics": {
                            "tier": "T1",
                            "promotion_eligible": False,
                            "frontier_lane": "alpha",
                            "strategy_family": "learned_alpha",
                            "per_task": {
                                "P01": {
                                    "mean_active_alpha_vs_benchmark_pct": 99.0,
                                    "mean_active_share": 0.2,
                                }
                            },
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                            "full_t1_confirmed": True,
                        },
                    },
                    {
                        "id": "aggregate",
                        "finding_type": "result",
                        "variant_name": "aggregate",
                        "metrics": {
                            "tier": "T1",
                            "promotion_eligible": False,
                            "frontier_lane": "alpha",
                            "strategy_family": "learned_alpha",
                            "mean_active_alpha_vs_benchmark_pct": -0.5,
                            "mean_active_share": 0.03,
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                            "full_t1_confirmed": True,
                        },
                    },
                ],
            )

            self.assertEqual([p["finding_id"] for p in promoted], ["aggregate"])

    def test_lane_mode_can_route_falsifier_to_diagnostic_lane(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "diagnostic_roles",
                primary_metric="score",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "diagnostic_control",
                        "k": 1,
                        "include_roles": ["falsifier"],
                        "include_lanes": ["diagnostic_control"],
                        "axes": [("diagnostic_value", "maximize")],
                    }
                ],
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "falsifier_control",
                        "finding_type": "result",
                        "variant_name": "falsifier_control",
                        "metrics": {
                            "tier": "T3",
                            "promotion_eligible": True,
                            "frontier_lane": "diagnostic_control",
                            "diagnostic_value": 3.5,
                            "evidence_stage": "T3",
                            "scored_complete": True,
                        },
                        "extra": {"peer_role": "falsifier"},
                    }
                ],
            )

            self.assertEqual(len(promoted), 1)
            self.assertEqual(promoted[0]["frontier_lane"], "diagnostic_control")
            self.assertEqual(promoted[0]["metrics"]["peer_role"], "falsifier")

    def test_lane_cumulative_sort_respects_minimize_axis_direction(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "minimize_lane",
                primary_metric="score",
                metric_direction="maximize",
                frontier_lanes=[
                    {
                        "name": "risk_control",
                        "k": 1,
                        "cumulative_cap": 1,
                        "include_lanes": ["risk_control"],
                        "axes": [("drawdown_delta", "minimize")],
                    }
                ],
            )
            store.promote(
                0,
                [
                    {
                        "id": "bad_drawdown",
                        "finding_type": "result",
                        "variant_name": "bad_drawdown",
                        "metrics": {
                            "frontier_lane": "risk_control",
                            "drawdown_delta": 9.0,
                            "scored_complete": True,
                        },
                    }
                ],
            )
            store.promote(
                1,
                [
                    {
                        "id": "good_drawdown",
                        "finding_type": "result",
                        "variant_name": "good_drawdown",
                        "metrics": {
                            "frontier_lane": "risk_control",
                            "drawdown_delta": 1.0,
                            "scored_complete": True,
                        },
                    }
                ],
            )

            manifest = store.get_manifest()
            self.assertEqual(
                [row["finding_id"] for row in manifest["lane_frontiers"]["risk_control"]],
                ["good_drawdown"],
            )

    def test_metric_lookup_promotion_filtering_and_diversity_annotations(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        shared = {"score": 0.3}
        nested = {"summary": shared, "metrics": shared, "details": {"score": 9.9}}
        self.assertEqual(frontier._walk_for_metric(nested, "score", _strict_canonical=True), 0.3)
        self.assertEqual(frontier._walk_for_metric({"score": True}, "score"), None)
        self.assertEqual(frontier._walk_for_metric({"score": math.inf}, "score"), None)
        self.assertEqual(frontier._walk_for_metric({"score": "0.9"}, "score"), None)
        self.assertIsNone(frontier._walk_for_metric({"details": {"score": 0.9}}, "score", depth=7))
        self.assertIsNone(
            frontier._walk_for_metric(
                {"metrics": {"other": 1.0}, "details": {"score": 0.9}},
                "score",
                _strict_canonical=True,
            )
        )
        self.assertEqual(frontier._walk_for_metric({"a": [{"score": 0.4}]}, "score"), 0.4)
        cycle: dict = {}
        cycle["self"] = cycle
        self.assertIsNone(frontier._walk_for_metric(cycle, "score"))

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "frontier"
            snapshot_src = Path(tmp) / "snapshot"
            snapshot_src.mkdir()
            (snapshot_src / "optimizer.py").write_text("x", encoding="utf-8")
            (snapshot_src / "__pycache__").mkdir()
            (snapshot_src / "__pycache__" / "skip.pyc").write_text("cache", encoding="utf-8")
            findings = [
                {
                    "id": "low",
                    "finding_type": "result",
                    "metrics": {"score": 0.1, "tier": "T2", "promotion_eligible": True},
                    "variant_name": "low",
                },
                {
                    "id": "role",
                    "finding_type": "result",
                    "metrics": {
                        "score": 0.95,
                        "tier": "T3",
                        "evidence_stage": "T3",
                        "scored_complete": True,
                    },
                    "variant_name": "role",
                },
                {
                    "id": "bad-tier",
                    "finding_type": "result",
                    "metrics": {"score": 0.99, "tier": 3},
                    "variant_name": "bad-tier",
                },
                {
                    "id": "unknown-tier",
                    "finding_type": "result",
                    "metrics": {"score": 0.97, "tier": "T9", "promotion_eligible": "no"},
                    "variant_name": "unknown-tier",
                },
                {
                    "id": "no-promo",
                    "finding_type": "result",
                    "metrics": {"score": 0.98, "tier": "T3", "promotion_eligible": "no"},
                    "variant_name": "no-promo",
                },
                {
                    "id": "weird-promo",
                    "finding_type": "result",
                    "metrics": {"score": 0.97, "tier": "T3", "promotion_eligible": "maybe"},
                    "variant_name": "weird-promo",
                },
                {
                    "id": "numeric-promo",
                    "finding_type": "result",
                    "metrics": {"score": 0.96, "tier": "T3", "promotion_eligible": 0},
                    "variant_name": "numeric-promo",
                },
                {
                    "id": "best",
                    "finding_type": "insight",
                    "metrics": {"score": 0.8, "tier": "T3", "cost": 10},
                    "variant_name": "dup",
                    "snapshot_local_path": str(snapshot_src),
                    "design_dimensions": {"mechanism": "a", "schedule": "cosine"},
                },
                {
                    "id": "dup2",
                    "finding_type": "result",
                    "metrics": {"score": 0.7, "tier": "T3", "cost": 1},
                    "variant_name": "dup",
                },
                {
                    "id": "anchor",
                    "finding_type": "result",
                    "metrics": {
                        "score": 0.6,
                        "tier": "T3",
                        "cost": 0.2,
                        "evidence_stage": "T3",
                        "scored_complete": True,
                    },
                    "variant_name": "anchor",
                    "details": {"promotion_eligible": "yes"},
                },
                {
                    "id": "missing",
                    "finding_type": "note",
                    "metrics": {"score": 1.0},
                    "variant_name": "missing",
                },
                {
                    "id": "bad-score",
                    "finding_type": "result",
                    "metrics": {"score": "not-float", "tier": "T3"},
                    "final_metrics": {"score": 0.55},
                    "variant_name": "bad-score",
                },
                {
                    "id": "no-score",
                    "finding_type": "result",
                    "metrics": {"tier": "T3"},
                    "variant_name": "no-score",
                },
            ]
            store = frontier.FrontierStore(
                base,
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                anchor_metrics=[("cost", "minimize")],
                require_tier=True,
            )
            promoted = store.promote(0, findings)
            self.assertEqual([p["finding_id"] for p in promoted], ["role", "anchor"])
            self.assertEqual(promoted[1]["promoted_for_anchor"], "cost")
            self.assertTrue((base / "gen_0" / "top_1_finding.json").exists())
            self.assertFalse(
                (base / "gen_0" / "top_1_snapshot.tar.gz").exists()
            )  # "role" has no snapshot
            # tarfile assertions removed — "role" finding has no snapshot
            manifest = store.get_manifest()
            self.assertEqual(manifest["cumulative_top"][0]["finding_id"], "role")
            self.assertEqual(store.get_generation_top_metrics()[0], 0.95)
            summary = store.get_summary()
            summary.clear()
            self.assertTrue(store.get_summary())

            store._manifest["generations"]["bad"] = [{"finding_id": "bad", "metric_value": 100}]
            store._manifest["generations"]["3"] = []
            store._update_cumulative_top()
            self.assertTrue(store.get_summary())
            store._manifest["generations"].pop("bad")
            self.assertIsNone(store.get_generation_top_metrics()[3])

            empty = frontier.FrontierStore(
                Path(tmp) / "empty",
                promote_top_k=0,
                primary_metric="score",
                require_tier=False,
            )
            self.assertEqual(
                empty.promote(1, [{"finding_type": "result", "metrics": {"score": 1}}]), []
            )
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.tools.atomic_io.atomic_write_json",
                side_effect=RuntimeError("atomic unavailable"),
            ):
                empty._save_manifest()
            self.assertTrue(empty.manifest_path.exists())
            self.assertEqual(
                empty.promote(
                    2,
                    [
                        {
                            "id": "theory",
                            "finding_type": "result",
                            "metrics": {"score": 1.0},
                            "extra": {"peer_role": "theorist"},
                        }
                    ],
                ),
                [],
            )
            self.assertEqual(
                empty.promote(
                    3,
                    [{"id": "nom", "finding_type": "result", "metrics": {"other": 1.0}}],
                ),
                [],
            )
            with patch.object(frontier.tarfile, "open", side_effect=RuntimeError("tar")):
                self.assertIsNone(
                    store._freeze_snapshot(
                        {"snapshot_local_path": str(snapshot_src)},
                        base / "bad_snapshot.tar.gz",
                    )
                )

            class FakeTarInfo:
                def __init__(
                    self, name: str, size: int = 1, *, sym: bool = False, chr_: bool = False
                ):
                    self.name = name
                    self.size = size
                    self._sym = sym
                    self._chr = chr_

                def issym(self):
                    return self._sym

                def islnk(self):
                    return False

                def ischr(self):
                    return self._chr

                def isblk(self):
                    return False

                def isfifo(self):
                    return False

                def isfile(self):
                    return True

            self.assertIsNone(frontier.FrontierStore._tar_filter(FakeTarInfo("x/.git/config")))
            self.assertIsNone(frontier.FrontierStore._tar_filter(FakeTarInfo("x", sym=True)))
            self.assertIsNone(frontier.FrontierStore._tar_filter(FakeTarInfo("x", chr_=True)))
            self.assertIsNone(
                frontier.FrontierStore._tar_filter(FakeTarInfo("x", size=300 * 1024 * 1024))
            )
            self.assertIsNotNone(frontier.FrontierStore._tar_filter(FakeTarInfo("x.py")))

            manifest_dir = Path(tmp) / "manifest_existing"
            manifest_dir.mkdir()
            (manifest_dir / "frontier_manifest.json").write_text(
                json.dumps({"generations": {}, "cumulative_top": [{"finding_id": "old"}]}),
                encoding="utf-8",
            )
            self.assertEqual(frontier.FrontierStore(manifest_dir).get_summary(), [])

        self.assertEqual(
            frontier._extract_design_dimensions({"metrics": {"design_dimensions": {"A": " X "}}}),
            {"A": "x"},
        )
        self.assertIsNone(frontier._extract_design_dimensions({"design_dimensions": {"a": None}}))
        self.assertIsNone(frontier.compute_dimension_overlap({}, {}))
        self.assertIsNone(
            frontier.compute_dimension_overlap(
                {"design_dimensions": {"a": "x"}},
                {"design_dimensions": {"b": "x"}},
            )
        )
        overlap = frontier.compute_dimension_overlap(
            {"design_dimensions": {"a": "x", "b": "y"}},
            {"design_dimensions": {"a": "x", "b": "z"}},
        )
        self.assertEqual(overlap["overlap_count"], 1)
        no_anchors = frontier.annotate_findings_with_diversity_overlap(
            [{"metrics": {"score": 1}}],
            [],
            expected_dim_count=2,
        )
        self.assertEqual(no_anchors[0]["metrics"]["diversity_overlap_status"], "no_anchors")
        annotated = frontier.annotate_findings_with_diversity_overlap(
            [
                {"variant_name": "clone", "design_dimensions": {"a": "x", "b": "y"}},
                {"variant_name": "narrow", "design_dimensions": {"a": "x", "b": "z", "c": "q"}},
                {"variant_name": "clean", "design_dimensions": {"a": "q", "b": "z"}},
                {"variant_name": "missing"},
            ],
            [{"variant_name": "anchor", "design_dimensions": {"a": "x", "b": "y", "c": "q"}}],
            expected_dim_count=3,
        )
        self.assertEqual(
            [row["metrics"]["diversity_overlap_status"] for row in annotated],
            ["clone", "narrow", "clean", "no_data"],
        )

    def test_frontier_identity_and_evidence_helpers_cover_fallback_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        class BadStr:
            def __str__(self) -> str:
                raise ValueError("cannot stringify")

        self.assertEqual(frontier._norm_token(BadStr()), "")
        self.assertEqual(frontier._norm_token_set(7), {"7"})
        self.assertEqual(frontier._merged_extra({"extra": "bad"}), {})
        self.assertEqual(frontier._merged_extra({"extra": {"plain": "value"}}), {"plain": "value"})
        self.assertEqual(
            frontier._research_metadata_from_finding(
                {
                    "metrics": {"bottleneck_target": "  "},
                    "details": {"bottleneck_target": "drawdown"},
                    "extra": {"next_step_intent": "repair"},
                }
            ),
            {"bottleneck_target": "drawdown", "next_step_intent": "repair"},
        )
        self.assertEqual(frontier._metric_value({"details": {"score": 2}}, "score"), 2.0)
        self.assertIsNone(frontier._metric_value({"metrics": {"score": float("nan")}}, "score"))
        self.assertEqual(
            frontier._raw_candidate_field(
                {
                    "metrics": {"value": ""},
                    "details": {"value": "from-details"},
                    "extra": {"value": "from-extra"},
                },
                "value",
            ),
            "from-details",
        )
        self.assertTrue(frontier._boolish_candidate_field({"x": 1}, "x"))
        self.assertFalse(frontier._boolish_candidate_field({"x": 0}, "x"))
        self.assertEqual(
            frontier._identity_variant_token("artifact::results/unknown/tiered_eval_summary.json"),
            "",
        )
        self.assertEqual(
            frontier._identity_variant_token("artifact::results/unknown/summary.json"),
            "",
        )
        self.assertEqual(
            frontier._candidate_result_path_identity(
                {
                    "variant_id": "alpha_sweep",
                    "metrics": {
                        "source_result_path": "results/alpha_sweep/tiered_eval_summary.json"
                    },
                    "extra": {"source_path": "results/alpha_child/tiered_eval_summary.json"},
                }
            ),
            ("results/alpha_sweep/tiered_eval_summary.json", "alpha_sweep"),
        )
        self.assertEqual(
            frontier._candidate_entity_key(
                {"frontier_entity_key": "artifact::results/family_sweep/tiered_eval_summary.json"}
            ),
            "variant::family_sweep",
        )
        self.assertEqual(
            frontier._candidate_entity_key({"source_path": "artifacts/no_child.json"}),
            "artifact::artifacts/no_child.json",
        )
        self.assertEqual(
            frontier._candidate_entity_key(
                {
                    "variant_name": "shared_display_label",
                    "metrics": {
                        "variant_id": "legacy_shared_parent",
                        "source_path": "artifacts/no_child.json",
                    },
                }
            ),
            "artifact::artifacts/no_child.json",
        )
        self.assertEqual(
            frontier._candidate_entity_key({"gem_variant_ref": "gem-alpha"}),
            "variant::gem-alpha",
        )
        self.assertEqual(frontier._candidate_entity_key({"id": "row-1"}), "finding::row-1")
        self.assertTrue(frontier._candidate_entity_key({}).startswith("object::"))

        for candidate, expected in [
            ({"tier_status": "smoke pending"}, "smoke"),
            ({"result_status": "cheap-probe"}, "scout"),
            ({"final_status": "capped_at budget"}, "scout"),
            ({"stage": "fullt1"}, "unknown"),
            ({"stage": "full-t2"}, "unknown"),
            ({"stage": "full t3"}, "unknown"),
            ({"stage": "sanity"}, "smoke"),
            ({"stage": "failed_or_unscored"}, "smoke"),
            ({"full_t1_confirmed": "yes"}, "unknown"),
            ({"scored_complete": "yes"}, "scored_complete"),
        ]:
            self.assertEqual(frontier._normalized_evidence_stage(candidate), expected)

        metadata = frontier._evidence_metadata_from_candidate(
            {
                "cell_count": "12",
                "scored_cell_count": True,
                "n_scored_cells": float("nan"),
                "n_eval_cells": "Infinity",
                "scout_only": "no",
            }
        )
        self.assertEqual(metadata["scored_cell_count"], 12)
        self.assertFalse(metadata["scout_only"])

    def test_frontier_store_lane_and_cumulative_helpers_cover_edge_paths(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "lane_edges",
                primary_metric="score",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "incubator",
                        "k": 2,
                        "allow_missing_tier": True,
                        "allow_non_promotable": True,
                        "axes": [
                            "bad-axis",
                            ("cost", "minimize"),
                            {"name": "score"},
                            {"name": "ignored", "direction": "sideways"},
                        ],
                        "optional_axes": [
                            "bad-axis",
                            {"name": "novelty", "direction": "maximize"},
                            ("ignored", "sideways"),
                        ],
                    }
                ],
            )

            self.assertEqual(store._lane_axes({"axes": []}), [("score", "maximize")])
            self.assertEqual(
                store._lane_axes(store.frontier_lanes[0]),
                [("cost", "minimize"), ("score", "maximize")],
            )
            self.assertEqual(
                store._lane_optional_axes(store.frontier_lanes[0]),
                [("novelty", "maximize")],
            )
            self.assertEqual(
                frontier.FrontierStore._lane_sort_key(
                    {"variant_name": "alpha", "scored_complete": True},
                    {"score": 4.0},
                    [("score", "minimize")],
                    [("missing", "maximize")],
                ),
                (2, -4.0, float("-inf"), "alpha"),
            )

            promoted = store.promote(
                0,
                [
                    {
                        "id": "missing-tier",
                        "finding_type": "result",
                        "variant_name": "missing-tier",
                        "tier": 3,
                        "promotion_eligible": "maybe",
                        "metrics": {"score": 4.0, "cost": 1.0, "scored_complete": True},
                    }
                ],
            )
            self.assertEqual([p["finding_id"] for p in promoted], ["missing-tier"])
            self.assertTrue(promoted[0]["metrics"]["lane_missing_tier_candidate"])
            self.assertTrue(promoted[0]["metrics"]["lane_non_promotable_candidate"])

            store._manifest["generations"] = {
                "bad": [
                    {
                        "finding_id": "dup-old",
                        "variant_name": "same",
                        "frontier_lane": "incubator",
                        "promoted_for_lane": "incubator",
                        "lane_metric_value": "bad",
                        "generation_id": "bad",
                    }
                ],
                "1": [
                    {
                        "finding_id": "dup-new",
                        "variant_name": "same",
                        "frontier_lane": "incubator",
                        "promoted_for_lane": "incubator",
                        "lane_metric_value": 2.0,
                        "generation_id": 1,
                        "scored_complete": True,
                    },
                    {
                        "finding_id": "loose",
                        "variant_name": "loose",
                        "metric_value": 5.0,
                    },
                ],
            }
            store._update_cumulative_top()
            self.assertEqual(
                [entry["finding_id"] for entry in store.get_manifest()["cumulative_top"]],
                ["dup-new"],
            )
            self.assertEqual(
                [
                    entry["finding_id"]
                    for entry in store.get_manifest()["lane_frontiers"]["incubator"]
                ],
                ["dup-new"],
            )

        with tempfile.TemporaryDirectory() as tmp:
            legacy = frontier.FrontierStore(
                Path(tmp) / "legacy_edges",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="minimize",
                anchor_metrics=[("novelty", "maximize")],
            )
            legacy._manifest["generations"] = {
                "bad": [
                    {
                        "finding_id": "primary-bad",
                        "variant_name": "primary-bad",
                        "metric_value": "bad",
                        "generation_id": "bad",
                    }
                ],
                "1": [
                    {
                        "finding_id": "primary-good",
                        "variant_name": "primary-good",
                        "metric_value": 0.1,
                        "generation_id": 1,
                        "tier": "T3",
                        "scored_complete": True,
                    },
                    {
                        "finding_id": "anchor-old",
                        "variant_name": "anchor",
                        "metric_value": 9,
                        "generation_id": "bad",
                        "promoted_for_anchor": "novelty",
                    },
                    {
                        "finding_id": "anchor-new",
                        "variant_name": "anchor",
                        "metric_value": 8,
                        "generation_id": 2,
                        "promoted_for_anchor": "novelty",
                        "tier": "T3",
                        "scored_complete": True,
                    },
                ],
            }
            legacy._update_cumulative_top()
            self.assertEqual(
                [entry["finding_id"] for entry in legacy.get_summary()],
                ["primary-good", "anchor-new"],
            )

    def test_legacy_cumulative_preserves_all_distinct_anchor_entities(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            legacy = frontier.FrontierStore(
                Path(tmp) / "legacy_anchor_preserve",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                anchor_metrics=[("novelty", "maximize")],
            )
            legacy._manifest["generations"] = {
                str(i): [
                    {
                        "finding_id": f"anchor-{i}",
                        "variant_name": f"anchor-{i}",
                        "metric_value": float(i),
                        "generation_id": i,
                        "tier": "T3",
                        "scored_complete": True,
                        "promoted_for_anchor": "novelty",
                    }
                ]
                for i in range(8)
            }
            legacy._manifest["generations"]["8"] = [
                {
                    "finding_id": "primary",
                    "variant_name": "primary",
                    "metric_value": 100.0,
                    "generation_id": 8,
                    "tier": "T3",
                    "scored_complete": True,
                }
            ]

            legacy._update_cumulative_top()

            anchor_ids = [
                entry["finding_id"]
                for entry in legacy.get_summary()
                if entry.get("promoted_for_anchor")
            ]
            self.assertEqual(set(anchor_ids), {f"anchor-{i}" for i in range(8)})
            self.assertEqual(len(anchor_ids), 8)

    def test_frontier_promote_anchor_and_snapshot_recovery_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot_src = root / "snapshot"
            snapshot_src.mkdir()
            (snapshot_src / "result.txt").write_text("ok", encoding="utf-8")
            store = frontier.FrontierStore(
                root / "frontier",
                promote_top_k=0,
                primary_metric="score",
                metric_direction="maximize",
                anchor_metrics=[("novelty", "maximize"), ("cost", "minimize")],
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "anchor-a",
                        "finding_type": "result",
                        "variant_name": "anchor-a",
                        "metrics": {
                            "score": 1.0,
                            "novelty": 5.0,
                            "cost": 10.0,
                            "scored_complete": True,
                        },
                        "snapshot_local_path": str(snapshot_src),
                    },
                    {
                        "id": "anchor-b",
                        "finding_type": "result",
                        "variant_name": "anchor-b",
                        "metrics": {
                            "score": 1.0,
                            "novelty": 4.0,
                            "cost": 1.0,
                            "scored_complete": True,
                        },
                    },
                    {
                        "id": "anchor-missing",
                        "finding_type": "result",
                        "variant_name": "anchor-missing",
                        "metrics": {
                            "score": 1.0,
                            "novelty": "not numeric",
                            "scored_complete": True,
                        },
                    },
                ],
            )

            self.assertEqual(
                [(p["finding_id"], p.get("promoted_for_anchor")) for p in promoted],
                [("anchor-a", "novelty"), ("anchor-b", "cost")],
            )
            self.assertTrue(Path(promoted[0]["snapshot_path"]).exists())
            self.assertIsNone(promoted[1]["snapshot_path"])
            self.assertEqual(store.get_summary_for_generation(0), promoted)

            store._manifest["generations"] = []
            self.assertEqual(store.get_summary_for_generation(0), [])

    def test_retiring_validation_candidate_removes_validator_aliases(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )
            store.promote(
                0,
                [
                    {
                        "id": "scout_parent",
                        "finding_type": "result",
                        "variant_name": "repair_candidate",
                        "child_variant_id": "repair_candidate",
                        "source_result_path": "results/repair_candidate/summary.json",
                        "source_result_sha256": "repair-sha",
                        "metrics": {
                            "child_variant_id": "repair_candidate",
                            "source_result_path": "results/repair_candidate/summary.json",
                            "source_result_sha256": "repair-sha",
                            "score": 9.0,
                            "evidence_stage": "scout",
                        },
                    }
                ],
            )
            before = store.get_manifest()["validation_candidates"]
            self.assertIn("scout_parent", before["validator_identity_aliases_by_generation"]["0"])

            store.promote(
                1,
                [
                    {
                        "id": "full_parent",
                        "finding_type": "result",
                        "variant_name": "repair_candidate",
                        "child_variant_id": "repair_candidate",
                        "source_result_path": "results/repair_candidate/summary.json",
                        "source_result_sha256": "repair-sha",
                        "metrics": {
                            "child_variant_id": "repair_candidate",
                            "source_result_path": "results/repair_candidate/summary.json",
                            "source_result_sha256": "repair-sha",
                            "score": 8.0,
                            "evidence_stage": "full_T1",
                            "scored_complete": True,
                        },
                    }
                ],
            )

            validation = store.get_manifest()["validation_candidates"]
            self.assertEqual(validation["generations"]["0"], [])
            self.assertEqual(validation["cumulative"], [])
            self.assertNotIn("0", validation.get("validator_identity_aliases_by_generation", {}))

    def test_maturity_ratios_keep_useful_signals_without_hard_stage_names(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        lane = {
            "name": "incubator",
            "k": 5,
            "include_lanes": ["incubator"],
            "axes": [{"metric": "score", "direction": "maximize"}],
            "admit_new_high": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
                frontier_lanes=[lane],
                maturity_policy={
                    "min_effort_ratio": 0.85,
                    "min_coverage_ratio": 0.80,
                    "require_ratio_gate": True,
                },
            )

            promoted = store.promote(
                0,
                [
                    {
                        "id": "ratio-mature",
                        "finding_type": "result",
                        "variant_name": "ratio_mature",
                        "metrics": {
                            "score": 1.0,
                            "frontier_lane": "incubator",
                            "evidence_stage": "scout",
                            "scout_only": True,
                            "partial_eval": True,
                            "capped": True,
                            "effort_ratio": 0.9,
                            "coverage_ratio": 0.9,
                            "source_result_path": "results/ratio_mature.json",
                            "source_result_sha256": "ratio-mature-sha",
                        },
                    }
                ],
            )

            self.assertEqual(len(promoted), 1)
            self.assertEqual(promoted[0]["finding_id"], "ratio-mature")
            self.assertTrue(promoted[0]["mature_enough"])
            self.assertEqual(promoted[0]["min_effort_ratio"], 0.85)
            self.assertEqual(promoted[0]["min_coverage_ratio"], 0.8)
            self.assertEqual(promoted[0]["frontier_lane"], "incubator")

            rejected = store.promote(
                1,
                [
                    {
                        "id": "ratio-immature",
                        "finding_type": "result",
                        "variant_name": "ratio_immature",
                        "metrics": {
                            "score": 2.0,
                            "frontier_lane": "incubator",
                            "effort_ratio": 0.5,
                            "coverage_ratio": 0.9,
                        },
                    }
                ],
            )

            self.assertEqual(rejected, [])
            manifest = store.get_manifest()
            validation = manifest["validation_candidates"]["generations"]["1"]
            self.assertEqual(validation[0]["finding_id"], "ratio-immature")
            self.assertEqual(
                validation[0]["exclusion_reason"], "insufficient_mature_evidence_ratio"
            )
            self.assertFalse(validation[0]["mature_enough"])
            self.assertEqual(
                manifest["promotion_rejections"]["1"]["counts"][
                    "insufficient_mature_evidence_ratio"
                ],
                1,
            )

            incomplete = store.promote(
                2,
                [
                    {
                        "id": "ratio-incomplete",
                        "finding_type": "result",
                        "variant_name": "ratio_incomplete",
                        "metrics": {
                            "score": 3.0,
                            "frontier_lane": "incubator",
                            "effort_ratio": 0.95,
                            "coverage_ratio": 0.95,
                            "incomplete_eval": True,
                        },
                    }
                ],
            )

            self.assertEqual(incomplete, [])
            manifest = store.get_manifest()
            validation = manifest["validation_candidates"]["generations"]["2"]
            self.assertEqual(validation[0]["finding_id"], "ratio-incomplete")
            self.assertEqual(
                validation[0]["exclusion_reason"],
                "preliminary_or_incomplete_evidence",
            )

    def test_task_configured_stage_is_authoritative_for_durable_evidence(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        candidate = {
            "variant_name": "task_named_stage",
            "metrics": {"score": 1.0, "evidence_stage": "scout"},
        }
        policy = {
            "complete_stage_labels": ["scout"],
            "preliminary_stage_labels": [],
            "require_ratio_gate": False,
        }

        self.assertTrue(frontier._is_durable_frontier_entry(candidate, policy))
        self.assertFalse(frontier._is_durable_frontier_entry(candidate))
        self.assertFalse(
            frontier._is_durable_frontier_entry(
                {
                    "variant_name": "opaque_stage",
                    "metrics": {"score": 1.0, "evidence_stage": "full_eval"},
                }
            )
        )

    def test_task_authorized_reduced_protocol_can_promote_with_truthful_partial_metadata(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        policy = {
            "complete_stage_labels": ["reduced"],
            "preliminary_stage_labels": ["diagnostic"],
            "require_ratio_gate": False,
        }
        finding = {
            "id": "approved-reduced",
            "finding_type": "result",
            "variant_name": "approved_reduced",
            "metrics": {
                "score": 1.0,
                "evidence_stage": "reduced",
                "scored_complete": True,
                "partial": True,
                "promotion_eligible": True,
                "source_result_path": "results/approved_reduced/summary.json",
                "source_result_sha256": "approved-reduced-sha",
            },
        }

        self.assertTrue(frontier._has_mature_durable_evidence(finding, policy))
        self.assertTrue(frontier._is_durable_frontier_entry(finding, policy))

        for marker in ("summary_only", "unscored_artifact"):
            unusable = {
                **finding,
                "id": f"unusable-{marker}",
                "metrics": {**finding["metrics"], marker: True},
            }
            self.assertFalse(frontier._has_mature_durable_evidence(unusable, policy))
            self.assertFalse(frontier._is_durable_frontier_entry(unusable, policy))

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                maturity_policy=policy,
            )
            promoted = store.promote(0, [finding])

        self.assertEqual([entry["finding_id"] for entry in promoted], ["approved-reduced"])
        self.assertTrue(promoted[0]["mature_enough"])

        unfinished = {
            **finding,
            "id": "unfinished-reduced",
            "metrics": {**finding["metrics"], "scored_complete": False},
        }
        self.assertFalse(frontier._has_mature_durable_evidence(unfinished, policy))

    def test_raw_result_index_uses_task_authorized_maturity_policy(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result_dir = run_dir / "results" / "approved_reduced"
            result_dir.mkdir(parents=True)
            (result_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "approved_reduced",
                        "score": 1.0,
                        "evidence_stage": "reduced",
                        "tier_status": "partial",
                        "scored_complete": True,
                        "partial": True,
                        "promotion_eligible": True,
                    }
                ),
                encoding="utf-8",
            )
            store = frontier.FrontierStore(
                run_dir / "frontier",
                primary_metric="score",
                maturity_policy={
                    "complete_stage_labels": ["reduced"],
                    "preliminary_stage_labels": ["diagnostic"],
                },
            )

            by_path, _by_variant = store._canonical_result_source_index()
            metrics = by_path["results/approved_reduced/summary.json"]["candidate"]["metrics"]

            self.assertTrue(metrics["scored_complete"])
            self.assertEqual(metrics["result_status"], "scored_complete")
            self.assertTrue(metrics["partial"])

    def test_nonpromotable_lane_signal_is_never_exposed_as_parent(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        policy = {
            "complete_stage_labels": ["reduced"],
            "preliminary_stage_labels": ["diagnostic"],
        }
        finding = {
            "id": "validation-signal",
            "finding_type": "result",
            "variant_name": "validation_signal",
            "metrics": {
                "score": 1.0,
                "frontier_lane": "incubator",
                "evidence_stage": "reduced",
                "scored_complete": True,
                "partial": True,
                "promotion_eligible": False,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                primary_metric="score",
                maturity_policy=policy,
                frontier_lanes=[
                    {
                        "name": "incubator",
                        "include_lanes": ["incubator"],
                        "allow_non_promotable": True,
                        "parent_eligible": True,
                        "axes": [{"metric": "score", "direction": "maximize"}],
                    }
                ],
            )
            promoted = store.promote(0, [finding])
            parents = store.get_parent_summary_up_to_generation(0)

        self.assertEqual([entry["finding_id"] for entry in promoted], ["validation-signal"])
        self.assertFalse(promoted[0]["parent_eligible"])
        self.assertFalse(promoted[0]["metrics"]["parent_eligible"])
        self.assertEqual(parents, [])

    def test_explicit_nonparent_entry_is_not_exposed_without_frontier_lanes(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        finding = {
            "id": "durable-nonparent",
            "finding_type": "result",
            "variant_name": "durable_nonparent",
            "metrics": {
                "score": 1.0,
                "evidence_stage": "full_eval",
                "scored_complete": True,
                "promotion_eligible": True,
                "parent_eligible": False,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                primary_metric="score",
            )
            promoted = store.promote(0, [finding])
            parents = store.get_parent_summary_up_to_generation(0)

        self.assertEqual([entry["finding_id"] for entry in promoted], ["durable-nonparent"])
        self.assertEqual(parents, [])

    def test_ratio_immature_manifest_prune_migrates_to_validation_candidates(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                maturity_policy={
                    "min_effort_ratio": 0.60,
                    "min_coverage_ratio": 0.90,
                    "require_ratio_gate": True,
                },
            )
            store._manifest["generations"] = {
                "0": [
                    {
                        "finding_id": "old_ratio_immature",
                        "finding_type": "result",
                        "variant_name": "old_ratio_immature",
                        "metric_name": "score",
                        "metric_value": 2.0,
                        "evidence_stage": "full_eval",
                        "effort_ratio": 0.55,
                        "coverage_ratio": 0.85,
                        "metrics": {
                            "score": 2.0,
                            "effort_ratio": 0.55,
                            "coverage_ratio": 0.85,
                        },
                    }
                ]
            }

            changed = store._prune_durable_frontier_entries(migrate=True)
            manifest = store.get_manifest()

        self.assertTrue(changed)
        self.assertEqual(manifest["generations"]["0"], [])
        migrated = manifest["validation_candidates"]["generations"]["0"]
        self.assertEqual(migrated[0]["finding_id"], "old_ratio_immature")
        self.assertEqual(migrated[0]["min_effort_ratio"], 0.6)
        self.assertEqual(migrated[0]["min_coverage_ratio"], 0.9)
        self.assertFalse(migrated[0]["mature_enough"])

    def test_canonical_source_repair_replaces_score_and_maturity_as_one_fact_tuple(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        policy = {
            "min_effort_ratio": 0.75,
            "min_coverage_ratio": 0.8,
            "require_ratio_gate": True,
            "complete_stage_labels": ["complete"],
            "preliminary_stage_labels": ["aligned"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frontier_dir = run_dir / "frontier"
            aligned_path = run_dir / "results" / "candidate" / "aligned" / "summary.json"
            complete_path = (
                run_dir / "results" / "candidate" / "complete" / "evaluation_summary.json"
            )
            aligned_path.parent.mkdir(parents=True)
            complete_path.parent.mkdir(parents=True)
            aligned_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "candidate",
                        "variant_name": "candidate",
                        "score": 0.95,
                        "evidence_stage": "aligned",
                        "effort_ratio": 0.4117647,
                        "coverage_ratio": 1.0,
                        "scored_complete": True,
                        "promotion_eligible": False,
                        "is_negative": True,
                        "failure_mode": "preliminary_failure",
                        "diagnostic_role": "preliminary_diagnostic",
                    }
                ),
                encoding="utf-8",
            )
            complete_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "candidate",
                        "variant_name": "candidate",
                        "score": 0.8,
                        "evidence_stage": "complete",
                        "effort_ratio": 1.0,
                        "coverage_ratio": 1.0,
                        "scored_complete": True,
                        "promotion_eligible": True,
                        "parent_eligible": False,
                        "is_negative": False,
                        "diagnostic_role": "canonical_complete",
                        "current_aggregate": {
                            "score": 0.8,
                            "extra": {
                                "source_lane": "candidate_pool",
                                "target_lane": "durable_frontier",
                                "coverage_check": "required_units_complete",
                                "mechanism_hypothesis_deliverable": "compare mechanism",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            store = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                maturity_policy=policy,
            )
            stale = {
                "generation_id": 0,
                "finding_id": "candidate-finding",
                "variant_name": "candidate",
                "metric_name": "score",
                "metric_value": 0.95,
                "source_result_path": str(aligned_path.relative_to(run_dir)),
                "evidence_stage": "complete",
                "effort_ratio": 1.0,
                "coverage_ratio": 1.0,
                "mature_enough": True,
                "promotion_eligible": True,
                "result_status": "protocol_invalid",
                "summary_only": True,
                "validation_only_result": True,
                "protocol_integrity_failed": True,
                "is_negative": True,
                "failure_mode": "stale_manifest_failure",
                "diagnostic_role": "stale_manifest_diagnostic",
                "metrics": {
                    "score": 0.95,
                    "evidence_stage": "complete",
                    "effort_ratio": 1.0,
                    "coverage_ratio": 1.0,
                    "mature_enough": True,
                    "promotion_eligible": True,
                    "result_status": "protocol_invalid",
                    "summary_only": True,
                    "validation_only_result": True,
                    "protocol_integrity_failed": True,
                    "is_negative": True,
                    "failure_mode": "stale_manifest_failure",
                    "diagnostic_role": "stale_manifest_diagnostic",
                    "source_result_path": str(aligned_path.relative_to(run_dir)),
                },
            }
            store._manifest["generations"] = {"0": [stale]}
            store._save_manifest()

            repaired_manifest = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                maturity_policy=policy,
            ).get_manifest()
            repaired = repaired_manifest["generations"]["0"][0]

        self.assertEqual(
            repaired["source_result_path"],
            "results/candidate/complete/evaluation_summary.json",
        )
        self.assertEqual(repaired["metric_value"], 0.8)
        self.assertEqual(repaired["metrics"]["score"], 0.8)
        self.assertEqual(repaired["evidence_stage"], "complete")
        self.assertEqual(repaired["effort_ratio"], 1.0)
        self.assertTrue(repaired["promotion_eligible"])
        self.assertFalse(repaired["parent_eligible"])
        self.assertTrue(repaired["mature_enough"])
        self.assertTrue(repaired["source_result_sha256"])
        self.assertEqual(repaired["result_status"], "scored_complete")
        self.assertNotIn("summary_only", repaired)
        self.assertFalse(repaired["validation_only_result"])
        self.assertNotIn("protocol_integrity_failed", repaired)
        self.assertFalse(repaired["is_negative"])
        self.assertEqual(repaired["diagnostic_role"], "canonical_complete")
        self.assertEqual(repaired["source_lane"], "candidate_pool")
        self.assertEqual(repaired["target_lane"], "durable_frontier")
        self.assertEqual(repaired["coverage_check"], "required_units_complete")
        self.assertEqual(
            repaired["mechanism_hypothesis_deliverable"],
            "compare mechanism",
        )
        self.assertNotIn("failure_mode", repaired)
        self.assertNotIn("failure_mode", repaired["metrics"])
        aligned_signals = repaired_manifest["validation_candidates"]["generations"]["0"]
        self.assertTrue(
            any(
                frontier._raw_candidate_field(entry, "source_result_path")
                == "results/candidate/aligned/summary.json"
                and entry.get("metric_value") == 0.95
                for entry in aligned_signals
            )
        )
        self.assertTrue(
            any(
                frontier._raw_candidate_field(entry, "is_negative") is True
                and frontier._raw_candidate_field(entry, "failure_mode") == "preliminary_failure"
                for entry in aligned_signals
            )
        )

    def test_canonical_source_repair_does_not_cross_independent_result_families(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frontier_dir = run_dir / "frontier"
            source_a = run_dir / "results" / "implementation_a" / "summary.json"
            source_b = run_dir / "results" / "implementation_b" / "summary.json"
            source_a.parent.mkdir(parents=True)
            source_b.parent.mkdir(parents=True)
            source_a.write_text(
                json.dumps(
                    {
                        "result_variant_id": "reused-reported-id",
                        "variant_name": "candidate",
                        "generation_id": 0,
                        "score": 0.6,
                        "evidence_stage": "complete",
                        "scored_complete": True,
                        "promotion_eligible": True,
                    }
                ),
                encoding="utf-8",
            )
            source_b.write_text(
                json.dumps(
                    {
                        "result_variant_id": "reused-reported-id",
                        "variant_name": "candidate",
                        "generation_id": 0,
                        "score": 0.99,
                        "evidence_stage": "complete",
                        "scored_complete": True,
                        "promotion_eligible": True,
                    }
                ),
                encoding="utf-8",
            )
            store = frontier.FrontierStore(frontier_dir, primary_metric="score")
            store._manifest["generations"] = {
                "0": [
                    {
                        "generation_id": 0,
                        "finding_id": "implementation-a",
                        "snapshot_path": "snapshots/implementation-a.tar.gz",
                        "variant_name": "candidate",
                        "result_variant_id": "reused-reported-id",
                        "metric_name": "score",
                        "metric_value": 0.6,
                        "source_result_path": str(source_a.relative_to(run_dir)),
                        "evidence_stage": "complete",
                        "scored_complete": True,
                        "promotion_eligible": True,
                        "metrics": {"score": 0.6},
                    }
                ]
            }
            store._save_manifest()

            manifest = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
            ).get_manifest()

        [entry] = manifest["generations"]["0"]
        self.assertEqual(entry["source_result_path"], "results/implementation_a/summary.json")
        self.assertEqual(entry["metric_value"], 0.6)
        self.assertEqual(entry["finding_id"], "implementation-a")
        self.assertEqual(entry["snapshot_path"], "snapshots/implementation-a.tar.gz")

    def test_preliminary_source_does_not_upgrade_from_independent_result_family(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        policy = {
            "complete_stage_labels": ["complete"],
            "preliminary_stage_labels": ["aligned"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frontier_dir = run_dir / "frontier"
            source_a = run_dir / "results" / "implementation_a" / "summary.json"
            source_b = run_dir / "results" / "implementation_b" / "summary.json"
            source_a.parent.mkdir(parents=True)
            source_b.parent.mkdir(parents=True)
            source_a.write_text(
                json.dumps(
                    {
                        "result_variant_id": "reused-reported-id",
                        "variant_name": "candidate",
                        "generation_id": 0,
                        "score": 0.6,
                        "evidence_stage": "aligned",
                        "scored_complete": False,
                        "promotion_eligible": False,
                    }
                ),
                encoding="utf-8",
            )
            source_b.write_text(
                json.dumps(
                    {
                        "result_variant_id": "reused-reported-id",
                        "variant_name": "candidate",
                        "generation_id": 0,
                        "score": 0.99,
                        "evidence_stage": "complete",
                        "scored_complete": True,
                        "promotion_eligible": True,
                    }
                ),
                encoding="utf-8",
            )
            store = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                maturity_policy=policy,
            )
            source_path = str(source_a.relative_to(run_dir))
            store._manifest["generations"] = {
                "0": [
                    {
                        "generation_id": 0,
                        "finding_id": "implementation-a",
                        "variant_name": "candidate",
                        "result_variant_id": "reused-reported-id",
                        "metric_name": "score",
                        "metric_value": 0.6,
                        "source_result_path": source_path,
                        "evidence_stage": "complete",
                        "scored_complete": True,
                        "promotion_eligible": True,
                        "metrics": {"score": 0.6},
                    }
                ]
            }
            store._save_manifest()

            manifest = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                maturity_policy=policy,
            ).get_manifest()

        self.assertEqual(manifest["generations"]["0"], [])
        retained = manifest["validation_candidates"]["generations"]["0"]
        retained_paths = {
            frontier._raw_candidate_field(entry, "source_result_path") for entry in retained
        }
        self.assertIn("results/implementation_a/summary.json", retained_paths)
        self.assertNotIn("results/implementation_b/summary.json", retained_paths)

    def test_same_path_source_rewrite_retains_previous_signal_as_validation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        policy = {
            "complete_stage_labels": ["complete"],
            "preliminary_stage_labels": ["aligned"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frontier_dir = run_dir / "frontier"
            summary_path = run_dir / "results" / "candidate" / "summary.json"
            summary_path.parent.mkdir(parents=True)
            preliminary = {
                "result_variant_id": "candidate",
                "variant_name": "candidate",
                "score": 0.95,
                "evidence_stage": "aligned",
                "scored_complete": True,
                "promotion_eligible": False,
            }
            summary_path.write_text(json.dumps(preliminary), encoding="utf-8")
            old_digest = frontier._json_digest(preliminary)
            store = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                maturity_policy=policy,
            )
            source_path = str(summary_path.relative_to(run_dir))
            store._manifest["generations"] = {
                "0": [
                    {
                        "generation_id": 0,
                        "finding_id": "candidate-finding",
                        "variant_name": "candidate",
                        "result_variant_id": "candidate",
                        "metric_name": "score",
                        "metric_value": 0.95,
                        "source_result_path": source_path,
                        "source_result_sha256": old_digest,
                        "evidence_stage": "aligned",
                        "promotion_eligible": False,
                        "metrics": {
                            "score": 0.95,
                            "result_variant_id": "candidate",
                            "source_result_path": source_path,
                            "source_result_sha256": old_digest,
                            "evidence_stage": "aligned",
                            "promotion_eligible": False,
                        },
                    }
                ]
            }
            store._save_manifest()
            summary_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "candidate",
                        "variant_name": "candidate",
                        "score": 0.8,
                        "evidence_stage": "complete",
                        "scored_complete": True,
                        "promotion_eligible": True,
                    }
                ),
                encoding="utf-8",
            )

            manifest = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                maturity_policy=policy,
            ).get_manifest()

        repaired = manifest["generations"]["0"][0]
        self.assertEqual(repaired["metric_value"], 0.8)
        retained = manifest["validation_candidates"]["generations"]["0"]
        self.assertTrue(any(entry.get("metric_value") == 0.95 for entry in retained))

    def test_post_boundary_result_updates_remain_validation_signals(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
            frontier,
            resume_state,
        )

        boundary_cases = (
            ("committed", "same_path"),
            ("committed", "alternate_path"),
            ("pending", "same_path"),
            ("pending", "alternate_path"),
        )
        for boundary_mode, source_mode in boundary_cases:
            with (
                self.subTest(boundary_mode=boundary_mode, source_mode=source_mode),
                tempfile.TemporaryDirectory() as tmp,
            ):
                run_dir = Path(tmp) / "run"
                frontier_dir = run_dir / "frontier"
                original_path = run_dir / "results" / "candidate" / "summary.json"
                original_path.parent.mkdir(parents=True)
                original_path.write_text(
                    json.dumps(
                        {
                            "generation_id": 0,
                            "result_variant_id": "candidate",
                            "variant_name": "candidate",
                            "score": 0.8,
                            "evidence_stage": "full_eval",
                            "scored_complete": True,
                            "promotion_eligible": True,
                        }
                    ),
                    encoding="utf-8",
                )
                store = frontier.FrontierStore(frontier_dir, primary_metric="score")
                original_rel = str(original_path.relative_to(run_dir))
                by_path, _by_variant = store._canonical_result_source_index()
                original_digest = by_path[original_rel]["candidate"]["metrics"][
                    "source_result_sha256"
                ]
                store._manifest["generations"] = {
                    "0": [
                        {
                            "generation_id": 0,
                            "finding_id": "candidate-at-boundary",
                            "result_variant_id": "candidate",
                            "variant_name": "candidate",
                            "metric_name": "score",
                            "metric_value": 0.8,
                            "source_result_path": original_rel,
                            "source_result_sha256": original_digest,
                            "evidence_stage": "full_eval",
                            "scored_complete": True,
                            "promotion_eligible": True,
                            "metrics": {
                                "score": 0.8,
                                "result_variant_id": "candidate",
                                "source_result_path": original_rel,
                                "source_result_sha256": original_digest,
                                "evidence_stage": "full_eval",
                                "scored_complete": True,
                                "promotion_eligible": True,
                            },
                        }
                    ]
                }
                store._save_manifest()
                cutoff, snapshot = findings_collection.result_source_snapshot_with_cutoff(run_dir)
                if boundary_mode == "committed":
                    resume_state.write_boundary_marker(
                        run_dir,
                        gen_id=0,
                        promoted_count=1,
                        pi_status="succeeded",
                        evidence_cutoff_at=cutoff.isoformat(),
                        evidence_source_snapshot_at_cutoff=snapshot,
                    )
                else:
                    gen_dir = run_dir / "gen_0"
                    gen_dir.mkdir(parents=True)
                    (gen_dir / "generation_results.json").write_text("{}", encoding="utf-8")
                    self.assertTrue(
                        resume_state.write_boundary_evidence_checkpoint(
                            run_dir,
                            gen_id=0,
                            cutoff=cutoff,
                            evidence_source_snapshot=snapshot,
                        )
                    )
                updated_path = original_path
                if source_mode == "alternate_path":
                    updated_path = run_dir / "results" / "candidate" / "complete" / "summary.json"
                    updated_path.parent.mkdir(parents=True)
                updated_path.write_text(
                    json.dumps(
                        {
                            "generation_id": 0,
                            "result_variant_id": "candidate",
                            "variant_name": "candidate",
                            "score": 0.95,
                            "evidence_stage": "full_eval",
                            "scored_complete": True,
                            "promotion_eligible": True,
                        }
                    ),
                    encoding="utf-8",
                )

                manifest = frontier.FrontierStore(
                    frontier_dir,
                    primary_metric="score",
                ).get_manifest()

                durable = manifest["generations"]["0"][0]
                self.assertEqual(durable["metric_value"], 0.8)
                self.assertEqual(durable["source_result_path"], original_rel)
                self.assertEqual(durable["source_result_sha256"], original_digest)
                retained = manifest["validation_candidates"]["generations"]["0"]
                late = [entry for entry in retained if entry.get("metric_value") == 0.95]
                self.assertEqual(len(late), 1)
                self.assertTrue(late[0]["metrics"]["late_after_generation_boundary"])
                self.assertFalse(late[0]["metrics"]["promotion_eligible"])
                self.assertEqual(
                    late[0]["metrics"].get("generation_boundary_pending_commit", False),
                    boundary_mode == "pending",
                )

    def test_validation_candidates_deduplicate_same_immutable_result_snapshot(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(Path(tmp) / "frontier", primary_metric="score")
            common = {
                "generation_id": 0,
                "variant_name": "candidate",
                "metric_name": "score",
                "metric_value": 0.9,
                "metric_direction": "maximize",
                "source_result_path": "results/candidate/summary.json",
                "source_result_sha256": "a" * 64,
                "metrics": {
                    "score": 0.9,
                    "source_result_path": "results/candidate/summary.json",
                    "source_result_sha256": "a" * 64,
                },
            }
            store._record_validation_candidates(
                gen_id=0,
                entries=[
                    {
                        **common,
                        "finding_id": "materialized-finding",
                        "signal_source": "finding_metric",
                        "signal_source_priority": 1,
                    },
                    {
                        **common,
                        "metrics": dict(common["metrics"]),
                        "finding_id": "source-repair-observation",
                        "signal_source": "canonical_result_summary",
                        "signal_source_priority": 2,
                    },
                    {
                        **common,
                        "metrics": {
                            **common["metrics"],
                            "protocol_integrity_failed": True,
                            "promotion_eligible": False,
                            "exclusion_reason": "protocol_integrity_failed",
                        },
                        "finding_id": "protocol-failure",
                        "signal_source": "canonical_result_summary",
                        "signal_source_priority": 2,
                    },
                    {
                        **common,
                        "metrics": {
                            **common["metrics"],
                            "late_after_generation_boundary": True,
                            "validation_only_result": True,
                            "promotion_eligible": False,
                            "artifact_signal_status": "late_after_generation_boundary",
                            "exclusion_reason": "late_after_generation_boundary",
                        },
                        "finding_id": "late-result",
                        "signal_source": "canonical_result_summary",
                        "signal_source_priority": 2,
                    },
                ],
            )
            manifest = store.get_manifest()

        retained = manifest["validation_candidates"]["generations"]["0"]
        self.assertEqual(len(retained), 3)
        neutral = next(
            entry
            for entry in retained
            if not frontier._raw_candidate_field(entry, "exclusion_reason")
        )
        self.assertEqual(neutral["signal_source"], "canonical_result_summary")
        self.assertIn("materialized-finding", neutral["identity_aliases"])
        self.assertIn("source-repair-observation", neutral["identity_aliases"])
        self.assertTrue(
            any(frontier._candidate_protocol_integrity_failed(entry) for entry in retained)
        )
        self.assertTrue(
            any(
                frontier._any_boolish_candidate_field_true(
                    entry,
                    "late_after_generation_boundary",
                )
                for entry in retained
            )
        )
        self.assertEqual(len(manifest["validation_candidates"]["cumulative"]), 3)

    def test_canonical_source_repair_preserves_lane_only_metric(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            summary_path = run_dir / "results" / "candidate" / "summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "generation_id": 0,
                        "result_variant_id": "candidate",
                        "variant_name": "candidate",
                        "promotion_eligible": True,
                        "current_aggregate": {
                            "robustness": 0.9,
                            "frontier_lane": "robustness_lane",
                            "evidence_stage": "full_eval",
                            "scored_complete": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            source_path = str(summary_path.relative_to(run_dir))
            store = frontier.FrontierStore(
                run_dir / "frontier",
                primary_metric="score",
                frontier_lanes=[
                    {
                        "name": "robustness_lane",
                        "include_lanes": ["robustness_lane"],
                        "axes": [{"metric": "robustness", "direction": "maximize"}],
                    }
                ],
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "lane-only",
                        "finding_type": "result",
                        "variant_name": "candidate",
                        "metrics": {
                            "result_variant_id": "candidate",
                            "robustness": 0.95,
                            "frontier_lane": "robustness_lane",
                            "evidence_stage": "full_eval",
                            "scored_complete": True,
                            "promotion_eligible": True,
                            "source_result_path": source_path,
                        },
                    }
                ],
            )
            committed = store.get_manifest()["generations"]["0"]

        self.assertEqual(len(promoted), 1)
        self.assertEqual(promoted, committed)
        self.assertEqual(promoted[0]["metric_name"], "robustness")
        self.assertEqual(promoted[0]["metric_value"], 0.9)
        self.assertEqual(promoted[0]["lane_metric_name"], "robustness")
        self.assertEqual(promoted[0]["lane_metric_value"], 0.9)
        self.assertFalse(promoted[0].get("excluded_from_durable_frontier", False))

    def test_promote_returns_only_entries_committed_after_source_repair(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            summary_path = run_dir / "results" / "candidate" / "summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "candidate",
                        "variant_name": "candidate",
                        "score": 0.9,
                        "evidence_stage": "full_eval",
                        "scored_complete": True,
                        "promotion_eligible": False,
                    }
                ),
                encoding="utf-8",
            )
            source_path = str(summary_path.relative_to(run_dir))
            store = frontier.FrontierStore(
                run_dir / "frontier",
                promote_top_k=1,
                primary_metric="score",
            )

            promoted = store.promote(
                0,
                [
                    {
                        "id": "candidate-finding",
                        "finding_type": "result",
                        "variant_name": "candidate",
                        "metrics": {
                            "result_variant_id": "candidate",
                            "score": 0.95,
                            "evidence_stage": "full_eval",
                            "scored_complete": True,
                            "promotion_eligible": True,
                            "source_result_path": source_path,
                        },
                    }
                ],
            )
            manifest = store.get_manifest()

        self.assertEqual(promoted, [])
        self.assertEqual(manifest["generations"]["0"], [])
        retained = manifest["validation_candidates"]["generations"]["0"]
        self.assertTrue(any(entry.get("promotion_eligible") is False for entry in retained))

    def test_tiered_summary_top_level_parent_decision_survives_source_repair(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frontier_dir = run_dir / "frontier"
            summary_path = run_dir / "results" / "candidate" / "summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "candidate",
                        "variant_name": "candidate",
                        "parent_eligible": False,
                        "promotion_eligible": True,
                        "current_aggregate": {
                            "score": 0.8,
                            "evidence_stage": "full_eval",
                            "scored_complete": True,
                            "result_status": "scored_complete",
                        },
                    }
                ),
                encoding="utf-8",
            )
            store = frontier.FrontierStore(frontier_dir, primary_metric="score")
            source_path = str(summary_path.relative_to(run_dir))
            store._manifest["generations"] = {
                "0": [
                    {
                        "generation_id": 0,
                        "finding_id": "candidate-finding",
                        "variant_name": "candidate",
                        "metric_name": "score",
                        "metric_value": 0.8,
                        "source_result_path": source_path,
                        "parent_eligible": True,
                        "metrics": {
                            "score": 0.8,
                            "source_result_path": source_path,
                            "parent_eligible": True,
                        },
                    }
                ]
            }
            store._save_manifest()

            reloaded = frontier.FrontierStore(frontier_dir, primary_metric="score")
            repaired = reloaded.get_manifest()["generations"]["0"][0]
            parents = reloaded.get_parent_summary_up_to_generation(0)

        self.assertFalse(repaired["parent_eligible"])
        self.assertFalse(repaired["metrics"]["parent_eligible"])
        self.assertEqual(parents, [])

    def test_same_source_repair_preserves_manifest_nonparent_decision(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frontier_dir = run_dir / "frontier"
            summary_path = run_dir / "results" / "candidate" / "summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "candidate",
                        "variant_name": "candidate",
                        "score": 0.8,
                        "evidence_stage": "full_eval",
                        "scored_complete": True,
                        "promotion_eligible": True,
                    }
                ),
                encoding="utf-8",
            )
            store = frontier.FrontierStore(frontier_dir, primary_metric="score")
            source_path = str(summary_path.relative_to(run_dir))
            store._manifest["generations"] = {
                "0": [
                    {
                        "generation_id": 0,
                        "finding_id": "candidate-finding",
                        "variant_name": "candidate",
                        "metric_name": "score",
                        "metric_value": 0.8,
                        "source_result_path": source_path,
                        "parent_eligible": False,
                        "metrics": {
                            "score": 0.8,
                            "source_result_path": source_path,
                            "parent_eligible": False,
                        },
                    }
                ]
            }
            store._save_manifest()

            reloaded = frontier.FrontierStore(frontier_dir, primary_metric="score")
            repaired = reloaded.get_manifest()["generations"]["0"][0]
            parents = reloaded.get_parent_summary_up_to_generation(0)

        self.assertFalse(repaired["parent_eligible"])
        self.assertFalse(repaired["metrics"]["parent_eligible"])
        self.assertEqual(parents, [])

    def test_frontier_store_preserves_legacy_positional_require_tier_argument(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                2,
                "score",
                "maximize",
                [],
                [],
                [],
                True,
            )

        self.assertTrue(store._require_tier)

    def test_source_repair_demotes_nonpromotable_result_without_lane_exception(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frontier_dir = run_dir / "frontier"
            summary_path = run_dir / "results" / "candidate" / "summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "candidate",
                        "variant_name": "candidate",
                        "promotion_eligible": False,
                        "current_aggregate": {
                            "score": 0.8,
                            "evidence_stage": "full_eval",
                            "scored_complete": True,
                            "result_status": "scored_complete",
                        },
                    }
                ),
                encoding="utf-8",
            )
            store = frontier.FrontierStore(frontier_dir, primary_metric="score")
            source_path = str(summary_path.relative_to(run_dir))
            store._manifest["generations"] = {
                "0": [
                    {
                        "generation_id": 0,
                        "finding_id": "candidate-finding",
                        "variant_name": "candidate",
                        "metric_name": "score",
                        "metric_value": 0.8,
                        "source_result_path": source_path,
                        "promotion_eligible": True,
                        "metrics": {
                            "score": 0.8,
                            "source_result_path": source_path,
                            "promotion_eligible": True,
                        },
                    }
                ]
            }
            store._save_manifest()

            manifest = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
            ).get_manifest()

        self.assertEqual(manifest["generations"]["0"], [])
        retained = manifest["validation_candidates"]["generations"]["0"]
        self.assertTrue(any(entry.get("promotion_eligible") is False for entry in retained))

    def test_complete_source_with_same_explicit_result_id_can_cross_stage_layouts(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        policy = {
            "complete_stage_labels": ["complete"],
            "preliminary_stage_labels": ["aligned"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frontier_dir = run_dir / "frontier"
            preliminary_path = run_dir / "results" / "layout_a" / "aligned" / "summary.json"
            complete_path = run_dir / "results" / "layout_b" / "custom" / "summary.json"
            preliminary_path.parent.mkdir(parents=True)
            complete_path.parent.mkdir(parents=True)
            preliminary_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "stable-candidate-id",
                        "variant_name": "candidate",
                        "generation_id": 0,
                        "score": 0.95,
                        "evidence_stage": "aligned",
                        "promotion_eligible": False,
                    }
                ),
                encoding="utf-8",
            )
            complete_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "stable-candidate-id",
                        "variant_name": "candidate",
                        "generation_id": 0,
                        "score": 0.8,
                        "evidence_stage": "complete",
                        "scored_complete": True,
                        "promotion_eligible": True,
                    }
                ),
                encoding="utf-8",
            )
            store = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                maturity_policy=policy,
            )
            source_path = str(preliminary_path.relative_to(run_dir))
            store._manifest["generations"] = {
                "0": [
                    {
                        "generation_id": 0,
                        "finding_id": "candidate-finding",
                        "variant_name": "candidate",
                        "result_variant_id": "stable-candidate-id",
                        "metric_name": "score",
                        "metric_value": 0.95,
                        "source_result_path": source_path,
                        "evidence_stage": "aligned",
                        "metrics": {
                            "score": 0.95,
                            "result_variant_id": "stable-candidate-id",
                            "source_result_path": source_path,
                            "evidence_stage": "aligned",
                        },
                    }
                ]
            }
            store._save_manifest()
            preliminary_path.unlink()

            manifest = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                maturity_policy=policy,
            ).get_manifest()

        repaired = manifest["generations"]["0"][0]
        self.assertEqual(repaired["metric_value"], 0.8)
        self.assertEqual(
            repaired["source_result_path"],
            str(complete_path.relative_to(run_dir)),
        )

    def test_unknown_generation_reused_result_id_cannot_replace_durable_source(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frontier_dir = run_dir / "frontier"
            original_path = run_dir / "results" / "layout_a" / "summary.json"
            later_path = run_dir / "results" / "layout_b" / "summary.json"
            original_path.parent.mkdir(parents=True)
            later_path.parent.mkdir(parents=True)
            original_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "reused-id",
                        "variant_name": "candidate",
                        "score": 0.8,
                        "scored_complete": True,
                        "promotion_eligible": True,
                    }
                ),
                encoding="utf-8",
            )
            later_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "reused-id",
                        "variant_name": "candidate",
                        "score": 0.99,
                        "scored_complete": True,
                        "promotion_eligible": True,
                    }
                ),
                encoding="utf-8",
            )
            store = frontier.FrontierStore(frontier_dir, primary_metric="score")
            source_path = str(original_path.relative_to(run_dir))
            store._manifest["generations"] = {
                "0": [
                    {
                        "generation_id": 0,
                        "finding_id": "candidate-finding",
                        "variant_name": "candidate",
                        "result_variant_id": "reused-id",
                        "metric_name": "score",
                        "metric_value": 0.8,
                        "source_result_path": source_path,
                        "scored_complete": True,
                        "promotion_eligible": True,
                        "metrics": {
                            "score": 0.8,
                            "result_variant_id": "reused-id",
                            "source_result_path": source_path,
                            "scored_complete": True,
                            "promotion_eligible": True,
                        },
                    }
                ]
            }
            store._save_manifest()

            manifest = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
            ).get_manifest()

        repaired = manifest["generations"]["0"][0]
        self.assertEqual(repaired["metric_value"], 0.8)
        self.assertEqual(repaired["source_result_path"], source_path)

    def test_unknown_generation_same_path_rewrite_is_retained_only_as_signal(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frontier_dir = run_dir / "frontier"
            summary_path = run_dir / "results" / "summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "fixed-path-id",
                        "variant_name": "candidate",
                        "score": 0.8,
                        "scored_complete": True,
                        "promotion_eligible": True,
                    }
                ),
                encoding="utf-8",
            )
            store = frontier.FrontierStore(frontier_dir, primary_metric="score")
            source_path = str(summary_path.relative_to(run_dir))
            source_record = store._canonical_result_source_index()[0][source_path]
            old_digest = source_record["candidate"]["metrics"]["source_result_sha256"]
            store._manifest["generations"] = {
                "0": [
                    {
                        "generation_id": 0,
                        "finding_id": "candidate-finding",
                        "variant_name": "candidate",
                        "result_variant_id": "fixed-path-id",
                        "metric_name": "score",
                        "metric_value": 0.8,
                        "source_result_path": source_path,
                        "source_result_sha256": old_digest,
                        "scored_complete": True,
                        "promotion_eligible": True,
                        "metrics": {
                            "score": 0.8,
                            "source_result_path": source_path,
                            "source_result_sha256": old_digest,
                            "scored_complete": True,
                            "promotion_eligible": True,
                        },
                    }
                ]
            }
            store._save_manifest()
            summary_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "fixed-path-id",
                        "variant_name": "candidate",
                        "score": 0.99,
                        "scored_complete": True,
                        "promotion_eligible": True,
                    }
                ),
                encoding="utf-8",
            )

            manifest = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
            ).get_manifest()

        self.assertEqual(manifest["generations"]["0"], [])
        signals = manifest["validation_candidates"]["generations"]["0"]
        self.assertTrue(any(entry.get("metric_value") == 0.8 for entry in signals))
        self.assertTrue(any(entry.get("metric_value") == 0.99 for entry in signals))

    def test_unscoped_same_path_failed_rewrite_is_retained_as_artifact_signal(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frontier_dir = run_dir / "frontier"
            summary_path = run_dir / "results" / "summary.json"
            summary_path.parent.mkdir(parents=True)
            original = {
                "result_variant_id": "fixed-path-id",
                "variant_name": "candidate",
                "score": 0.8,
                "scored_complete": True,
                "promotion_eligible": True,
            }
            summary_path.write_text(json.dumps(original), encoding="utf-8")
            store = frontier.FrontierStore(frontier_dir, primary_metric="score")
            source_path = str(summary_path.relative_to(run_dir))
            old_digest = store._canonical_result_source_index()[0][source_path]["candidate"][
                "metrics"
            ]["source_result_sha256"]
            store._manifest["generations"] = {
                "0": [
                    {
                        "generation_id": 0,
                        "finding_id": "candidate-finding",
                        "variant_name": "candidate",
                        "result_variant_id": "fixed-path-id",
                        "metric_name": "score",
                        "metric_value": 0.8,
                        "source_result_path": source_path,
                        "source_result_sha256": old_digest,
                        "scored_complete": True,
                        "promotion_eligible": True,
                        "metrics": {
                            "score": 0.8,
                            "source_result_path": source_path,
                            "source_result_sha256": old_digest,
                            "scored_complete": True,
                            "promotion_eligible": True,
                        },
                    }
                ]
            }
            store._save_manifest()
            summary_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "fixed-path-id",
                        "variant_name": "candidate",
                        "result_status": "failed",
                        "promotion_eligible": False,
                    }
                ),
                encoding="utf-8",
            )

            manifest = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
            ).get_manifest()

        self.assertEqual(manifest["generations"]["0"], [])
        signals = manifest["validation_candidates"]["generations"]["0"]
        self.assertTrue(any(entry.get("metric_value") == 0.8 for entry in signals))
        artifact_signals = [
            entry for entry in signals if entry.get("signal_source") == "artifact_state"
        ]
        self.assertEqual(len(artifact_signals), 1)
        self.assertIsNone(artifact_signals[0]["metric_value"])
        self.assertEqual(
            frontier._raw_candidate_field(artifact_signals[0], "result_status"),
            "failed_or_unscored",
        )

    def test_unknown_generation_same_path_without_digest_cannot_become_durable(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frontier_dir = run_dir / "frontier"
            summary_path = run_dir / "results" / "summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "fixed-path-id",
                        "variant_name": "candidate",
                        "score": 0.8,
                        "scored_complete": True,
                        "promotion_eligible": True,
                    }
                ),
                encoding="utf-8",
            )
            store = frontier.FrontierStore(frontier_dir, primary_metric="score")
            source_path = str(summary_path.relative_to(run_dir))
            store._manifest["generations"] = {
                "0": [
                    {
                        "generation_id": 0,
                        "finding_id": "candidate-finding",
                        "variant_name": "candidate",
                        "result_variant_id": "fixed-path-id",
                        "metric_name": "score",
                        "metric_value": 0.8,
                        "source_result_path": source_path,
                        "scored_complete": True,
                        "promotion_eligible": True,
                        "metrics": {
                            "score": 0.8,
                            "source_result_path": source_path,
                            "scored_complete": True,
                            "promotion_eligible": True,
                        },
                    }
                ]
            }
            store._save_manifest()
            summary_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "fixed-path-id",
                        "variant_name": "candidate",
                        "score": 0.99,
                        "scored_complete": True,
                        "promotion_eligible": True,
                    }
                ),
                encoding="utf-8",
            )

            manifest = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
            ).get_manifest()

        self.assertEqual(manifest["generations"]["0"], [])
        signals = manifest["validation_candidates"]["generations"]["0"]
        self.assertTrue(any(entry.get("metric_value") == 0.8 for entry in signals))
        self.assertTrue(any(entry.get("metric_value") == 0.99 for entry in signals))

    def test_source_repair_uses_task_metric_aliases(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frontier_dir = run_dir / "frontier"
            summary_path = run_dir / "results" / "candidate" / "summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "candidate",
                        "variant_name": "candidate",
                        "generation_id": 0,
                        "raw_score": 0.8,
                        "scored_complete": True,
                        "promotion_eligible": True,
                    }
                ),
                encoding="utf-8",
            )
            store = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                result_metric_aliases={"score": "raw_score"},
            )
            source_path = str(summary_path.relative_to(run_dir))
            store._manifest["generations"] = {
                "0": [
                    {
                        "generation_id": 0,
                        "finding_id": "candidate-finding",
                        "variant_name": "candidate",
                        "result_variant_id": "candidate",
                        "metric_name": "score",
                        "metric_value": 0.8,
                        "source_result_path": source_path,
                        "scored_complete": True,
                        "promotion_eligible": True,
                        "metrics": {
                            "score": 0.8,
                            "source_result_path": source_path,
                            "scored_complete": True,
                            "promotion_eligible": True,
                        },
                    }
                ]
            }
            store._save_manifest()

            manifest = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                result_metric_aliases={"score": "raw_score"},
            ).get_manifest()

        repaired = manifest["generations"]["0"][0]
        self.assertEqual(repaired["metric_value"], 0.8)
        self.assertEqual(repaired["metrics"]["score"], 0.8)

    def test_durable_complete_source_outranks_late_validation_only_source(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frontier_dir = run_dir / "frontier"
            complete_path = run_dir / "results" / "candidate" / "complete" / "summary.json"
            late_path = run_dir / "results" / "candidate" / "late" / "summary.json"
            complete_path.parent.mkdir(parents=True)
            late_path.parent.mkdir(parents=True)
            complete_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "candidate",
                        "variant_name": "candidate",
                        "generation_id": 0,
                        "score": 0.8,
                        "scored_complete": True,
                        "promotion_eligible": True,
                    }
                ),
                encoding="utf-8",
            )
            late_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "candidate",
                        "variant_name": "candidate",
                        "generation_id": 0,
                        "score": 0.99,
                        "scored_complete": True,
                        "promotion_eligible": True,
                        "validation_only": True,
                        "late_after_generation_boundary": True,
                    }
                ),
                encoding="utf-8",
            )
            store = frontier.FrontierStore(frontier_dir, primary_metric="score")
            source_path = str(complete_path.relative_to(run_dir))
            store._manifest["generations"] = {
                "0": [
                    {
                        "generation_id": 0,
                        "finding_id": "candidate-finding",
                        "variant_name": "candidate",
                        "result_variant_id": "candidate",
                        "metric_name": "score",
                        "metric_value": 0.8,
                        "source_result_path": source_path,
                        "scored_complete": True,
                        "promotion_eligible": True,
                        "metrics": {
                            "score": 0.8,
                            "source_result_path": source_path,
                            "scored_complete": True,
                            "promotion_eligible": True,
                        },
                    }
                ]
            }
            store._save_manifest()

            manifest = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
            ).get_manifest()

        repaired = manifest["generations"]["0"][0]
        self.assertEqual(repaired["metric_value"], 0.8)
        self.assertEqual(repaired["source_result_path"], source_path)

    def test_legacy_same_path_rewrite_keeps_old_score_as_unverified_signal(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        policy = {
            "complete_stage_labels": ["complete"],
            "preliminary_stage_labels": ["aligned"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frontier_dir = run_dir / "frontier"
            summary_path = run_dir / "results" / "candidate" / "summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "candidate",
                        "variant_name": "candidate",
                        "score": 0.95,
                        "evidence_stage": "complete",
                        "scored_complete": True,
                        "promotion_eligible": True,
                    }
                ),
                encoding="utf-8",
            )
            store = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                maturity_policy=policy,
            )
            source_path = str(summary_path.relative_to(run_dir))
            store._manifest["generations"] = {
                "0": [
                    {
                        "generation_id": 0,
                        "finding_id": "candidate-finding",
                        "variant_name": "candidate",
                        "metric_name": "score",
                        "metric_value": 0.95,
                        "source_result_path": source_path,
                        "evidence_stage": "complete",
                        "promotion_eligible": True,
                        "metrics": {
                            "score": 0.95,
                            "source_result_path": source_path,
                            "evidence_stage": "complete",
                            "promotion_eligible": True,
                        },
                    }
                ]
            }
            store._save_manifest()
            summary_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "candidate",
                        "variant_name": "candidate",
                        "score": 0.2,
                        "evidence_stage": "aligned",
                        "promotion_eligible": False,
                    }
                ),
                encoding="utf-8",
            )

            manifest = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                maturity_policy=policy,
            ).get_manifest()

        self.assertEqual(manifest["generations"]["0"], [])
        retained = manifest["validation_candidates"]["generations"]["0"]
        self.assertTrue(
            any(
                entry.get("signal_source") == "manifest_snapshot"
                and entry.get("metric_value") == 0.95
                and entry.get("mature_enough") is not True
                and entry.get("metrics", {}).get("mature_enough") is False
                for entry in retained
            ),
            retained,
        )

    def test_canonical_source_must_still_satisfy_complete_lane_contract(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        lanes = [
            {
                "name": "incubator",
                "include_lanes": ["alpha"],
                "axes": [
                    {"metric": "score", "direction": "maximize"},
                    {"metric": "robustness", "direction": "maximize"},
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frontier_dir = run_dir / "frontier"
            summary_path = run_dir / "results" / "candidate" / "summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "candidate",
                        "variant_name": "candidate",
                        "score": 0.8,
                        "frontier_lane": "alpha",
                        "evidence_stage": "full_eval",
                        "scored_complete": True,
                        "promotion_eligible": True,
                    }
                ),
                encoding="utf-8",
            )
            store = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                frontier_lanes=lanes,
            )
            source_path = str(summary_path.relative_to(run_dir))
            store._manifest["generations"] = {
                "0": [
                    {
                        "generation_id": 0,
                        "finding_id": "candidate-finding",
                        "variant_name": "candidate",
                        "metric_name": "score",
                        "metric_value": 0.8,
                        "source_result_path": source_path,
                        "promoted_for_lane": "incubator",
                        "frontier_lane": "incubator",
                        "source_frontier_lane": "alpha",
                        "evidence_stage": "full_eval",
                        "scored_complete": True,
                        "promotion_eligible": True,
                        "metrics": {
                            "score": 0.8,
                            "robustness": 0.9,
                            "source_result_path": source_path,
                            "frontier_lane": "incubator",
                            "source_frontier_lane": "alpha",
                            "evidence_stage": "full_eval",
                            "scored_complete": True,
                            "promotion_eligible": True,
                        },
                    }
                ]
            }
            store._save_manifest()

            manifest = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                frontier_lanes=lanes,
            ).get_manifest()

        self.assertEqual(manifest["generations"]["0"], [])
        retained = manifest["validation_candidates"]["generations"]["0"]
        self.assertTrue(
            any(
                entry.get("finding_id") == "candidate-finding"
                and entry.get("exclusion_reason") == "preliminary_or_incomplete_evidence"
                for entry in retained
            )
        )

    def test_source_repair_never_carries_preliminary_score_into_scoreless_complete_result(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        policy = {
            "min_effort_ratio": 0.75,
            "min_coverage_ratio": 0.8,
            "require_ratio_gate": True,
            "complete_stage_labels": ["complete"],
            "preliminary_stage_labels": ["aligned"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frontier_dir = run_dir / "frontier"
            aligned_path = run_dir / "results" / "candidate" / "aligned" / "summary.json"
            complete_path = (
                run_dir / "results" / "candidate" / "complete" / "evaluation_summary.json"
            )
            aligned_path.parent.mkdir(parents=True)
            complete_path.parent.mkdir(parents=True)
            aligned_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "candidate",
                        "variant_name": "candidate",
                        "score": 0.95,
                        "evidence_stage": "aligned",
                        "effort_ratio": 0.4,
                        "coverage_ratio": 1.0,
                        "promotion_eligible": False,
                    }
                ),
                encoding="utf-8",
            )
            complete_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "candidate",
                        "variant_name": "candidate",
                        "evidence_stage": "complete",
                        "effort_ratio": 1.0,
                        "coverage_ratio": 1.0,
                        "scored_complete": True,
                        "promotion_eligible": True,
                    }
                ),
                encoding="utf-8",
            )
            store = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                maturity_policy=policy,
            )
            store._manifest["generations"] = {
                "0": [
                    {
                        "generation_id": 0,
                        "finding_id": "candidate-finding",
                        "variant_name": "candidate",
                        "metric_name": "score",
                        "metric_value": 0.95,
                        "source_result_path": str(aligned_path.relative_to(run_dir)),
                        "evidence_stage": "complete",
                        "effort_ratio": 1.0,
                        "coverage_ratio": 1.0,
                        "mature_enough": True,
                        "promotion_eligible": True,
                        "metrics": {"score": 0.95},
                    }
                ]
            }
            store._save_manifest()

            manifest = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                maturity_policy=policy,
            ).get_manifest()

        self.assertEqual(manifest["generations"]["0"], [])
        validation = manifest["validation_candidates"]["generations"]["0"]
        self.assertTrue(any(entry.get("metric_value") == 0.95 for entry in validation))
        self.assertFalse(
            any(
                frontier._raw_candidate_field(entry, "evidence_stage") == "complete"
                and frontier._metric_value(entry, "score") == 0.95
                for entry in validation
            )
        )

    def test_source_repair_excludes_canonical_summary_missing_required_lane_metric(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        policy = {
            "min_effort_ratio": 0.75,
            "min_coverage_ratio": 0.8,
            "require_ratio_gate": True,
            "complete_stage_labels": ["complete"],
            "preliminary_stage_labels": ["aligned"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frontier_dir = run_dir / "frontier"
            aligned_path = run_dir / "results" / "candidate" / "aligned" / "summary.json"
            complete_path = (
                run_dir / "results" / "candidate" / "complete" / "evaluation_summary.json"
            )
            aligned_path.parent.mkdir(parents=True)
            complete_path.parent.mkdir(parents=True)
            aligned_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "candidate",
                        "variant_name": "candidate",
                        "score": 0.95,
                        "robustness": 0.9,
                        "evidence_stage": "aligned",
                        "effort_ratio": 0.4,
                        "coverage_ratio": 1.0,
                        "promotion_eligible": False,
                    }
                ),
                encoding="utf-8",
            )
            complete_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "candidate",
                        "variant_name": "candidate",
                        "score": 0.8,
                        "evidence_stage": "complete",
                        "effort_ratio": 1.0,
                        "coverage_ratio": 1.0,
                        "scored_complete": True,
                        "promotion_eligible": True,
                    }
                ),
                encoding="utf-8",
            )
            store = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                maturity_policy=policy,
            )
            store._manifest["generations"] = {
                "0": [
                    {
                        "generation_id": 0,
                        "finding_id": "candidate-finding",
                        "variant_name": "candidate",
                        "metric_name": "score",
                        "metric_value": 0.95,
                        "lane_metric_name": "robustness",
                        "lane_metric_value": 0.9,
                        "source_result_path": str(aligned_path.relative_to(run_dir)),
                        "evidence_stage": "complete",
                        "mature_enough": True,
                        "promotion_eligible": True,
                        "metrics": {"score": 0.95, "robustness": 0.9},
                    }
                ]
            }
            store._save_manifest()

            manifest = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                maturity_policy=policy,
            ).get_manifest()

        self.assertEqual(manifest["generations"]["0"], [])
        validation = manifest["validation_candidates"]["generations"]["0"]
        self.assertTrue(validation)
        self.assertFalse(
            any(
                entry.get("evidence_stage") == "complete"
                and frontier._metric_value(entry, "robustness") == 0.9
                for entry in validation
            )
        )

    def test_same_source_preliminary_repair_survives_nonmigrating_prune(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        policy = {
            "min_effort_ratio": 0.75,
            "min_coverage_ratio": 0.8,
            "require_ratio_gate": True,
            "complete_stage_labels": ["complete"],
            "preliminary_stage_labels": ["aligned"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frontier_dir = run_dir / "frontier"
            aligned_path = run_dir / "results" / "candidate" / "aligned" / "summary.json"
            aligned_path.parent.mkdir(parents=True)
            aligned_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "candidate",
                        "variant_name": "candidate",
                        "score": 0.95,
                        "evidence_stage": "aligned",
                        "effort_ratio": 0.4,
                        "coverage_ratio": 1.0,
                        "promotion_eligible": False,
                    }
                ),
                encoding="utf-8",
            )
            store = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                maturity_policy=policy,
            )
            store._manifest["generations"] = {
                "0": [
                    {
                        "generation_id": 0,
                        "finding_id": "candidate-finding",
                        "variant_name": "candidate",
                        "metric_name": "score",
                        "metric_value": 0.95,
                        "source_result_path": str(aligned_path.relative_to(run_dir)),
                        "evidence_stage": "complete",
                        "mature_enough": True,
                        "promotion_eligible": True,
                        "metrics": {"score": 0.95},
                    }
                ]
            }

            self.assertTrue(store._repair_manifest_canonical_result_sources())
            self.assertTrue(store._prune_durable_frontier_entries(migrate=False))
            manifest = store.get_manifest()

        self.assertEqual(manifest["generations"]["0"], [])
        validation = manifest["validation_candidates"]["generations"]["0"]
        self.assertTrue(any(entry.get("metric_value") == 0.95 for entry in validation))

    def test_unknown_generation_source_repair_stays_within_result_family(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        policy = {
            "min_effort_ratio": 0.75,
            "min_coverage_ratio": 0.8,
            "require_ratio_gate": True,
            "complete_stage_labels": ["complete"],
            "preliminary_stage_labels": ["aligned"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frontier_dir = run_dir / "frontier"
            aligned_path = run_dir / "results" / "family_a" / "aligned" / "summary.json"
            unrelated_path = (
                run_dir / "results" / "family_b" / "complete" / "evaluation_summary.json"
            )
            aligned_path.parent.mkdir(parents=True)
            unrelated_path.parent.mkdir(parents=True)
            aligned_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "shared-reported-id",
                        "variant_name": "candidate",
                        "score": 0.9,
                        "evidence_stage": "aligned",
                        "effort_ratio": 0.4,
                        "coverage_ratio": 1.0,
                        "promotion_eligible": False,
                    }
                ),
                encoding="utf-8",
            )
            unrelated_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "shared-reported-id",
                        "variant_name": "candidate",
                        "score": 0.8,
                        "evidence_stage": "complete",
                        "effort_ratio": 1.0,
                        "coverage_ratio": 1.0,
                        "scored_complete": True,
                        "promotion_eligible": True,
                    }
                ),
                encoding="utf-8",
            )
            store = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                maturity_policy=policy,
            )
            store._manifest["generations"] = {
                "0": [
                    {
                        "generation_id": 0,
                        "finding_id": "candidate-finding",
                        "variant_name": "candidate",
                        "metric_name": "score",
                        "metric_value": 0.9,
                        "source_result_path": str(aligned_path.relative_to(run_dir)),
                        "evidence_stage": "complete",
                        "mature_enough": True,
                        "metrics": {"score": 0.9},
                    }
                ]
            }
            store._save_manifest()

            manifest = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                maturity_policy=policy,
            ).get_manifest()

        self.assertEqual(manifest["generations"]["0"], [])
        validation = manifest["validation_candidates"]["generations"]["0"]
        paths = {frontier._raw_candidate_field(entry, "source_result_path") for entry in validation}
        self.assertIn("results/family_a/aligned/summary.json", paths)
        self.assertNotIn("results/family_b/complete/evaluation_summary.json", paths)

    def test_source_repair_prunes_nonpromotable_aligned_result_claiming_complete(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        policy = {
            "min_effort_ratio": 0.75,
            "min_coverage_ratio": 0.8,
            "require_ratio_gate": True,
            "complete_stage_labels": ["complete"],
            "preliminary_stage_labels": ["aligned"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frontier_dir = run_dir / "frontier"
            aligned_path = run_dir / "results" / "candidate" / "aligned" / "summary.json"
            aligned_path.parent.mkdir(parents=True)
            aligned_path.write_text(
                json.dumps(
                    {
                        "result_variant_id": "candidate",
                        "variant_name": "candidate",
                        "score": 0.95,
                        "evidence_stage": "aligned",
                        "effort_ratio": 0.4117647,
                        "coverage_ratio": 1.0,
                        "scored_complete": True,
                        "promotion_eligible": False,
                    }
                ),
                encoding="utf-8",
            )
            store = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                maturity_policy=policy,
            )
            store._manifest["generations"] = {
                "0": [
                    {
                        "generation_id": 0,
                        "finding_id": "candidate-finding",
                        "variant_name": "candidate",
                        "metric_name": "score",
                        "metric_value": 0.95,
                        "source_result_path": str(aligned_path.relative_to(run_dir)),
                        "evidence_stage": "complete",
                        "effort_ratio": 1.0,
                        "coverage_ratio": 1.0,
                        "mature_enough": True,
                        "promotion_eligible": True,
                        "metrics": {
                            "score": 0.95,
                            "evidence_stage": "complete",
                            "effort_ratio": 1.0,
                            "coverage_ratio": 1.0,
                            "mature_enough": True,
                            "promotion_eligible": True,
                            "source_result_path": str(aligned_path.relative_to(run_dir)),
                        },
                    }
                ]
            }
            store._save_manifest()

            manifest = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                maturity_policy=policy,
            ).get_manifest()

        self.assertEqual(manifest["generations"]["0"], [])
        validation = manifest["validation_candidates"]["generations"]["0"][0]
        self.assertEqual(validation["evidence_stage"], "aligned")
        self.assertFalse(validation["promotion_eligible"])
        self.assertEqual(validation["effort_ratio"], 0.4117647)

    def test_required_ratio_manifest_prune_retains_legacy_result_for_validation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp) / "frontier"
            base_dir.mkdir()
            legacy_entry = {
                "finding_id": "legacy-full-result",
                "finding_type": "result",
                "variant_name": "legacy-full-result",
                "metric_name": "score",
                "metric_value": 2.0,
                "generation_id": 0,
                "evidence_stage": "full_T1",
                "scored_complete": True,
                "metrics": {"score": 2.0},
            }
            known_failed_entry = {
                **legacy_entry,
                "finding_id": "known-ratio-failure",
                "variant_name": "known-ratio-failure",
                "effort_ratio": 0.5,
            }
            (base_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {"0": [legacy_entry, known_failed_entry]},
                        "cumulative_top": [legacy_entry, known_failed_entry],
                        "lane_frontiers": {"candidate": [legacy_entry, known_failed_entry]},
                        "primary_metric": "score",
                        "metric_direction": "maximize",
                    }
                ),
                encoding="utf-8",
            )
            store = frontier.FrontierStore(
                base_dir,
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                maturity_policy={"require_ratio_gate": True},
            )

            self.assertEqual(store.get_summary()[0]["finding_id"], "legacy-full-result")
            self.assertEqual(
                store.get_summary_for_generation(0)[0]["finding_id"],
                "legacy-full-result",
            )
            persisted = json.loads((base_dir / "frontier_manifest.json").read_text())

        self.assertEqual(
            persisted["generations"]["0"][0]["finding_id"],
            "legacy-full-result",
        )
        self.assertEqual(len(persisted["generations"]["0"]), 1)
        self.assertFalse(
            frontier._is_durable_frontier_entry(
                legacy_entry,
                {"require_ratio_gate": True},
            )
        )
        self.assertFalse(
            frontier._is_committed_frontier_entry(
                known_failed_entry,
                {
                    "min_effort_ratio": 0.75,
                    "require_ratio_gate": True,
                },
            )
        )

    def test_positive_protocol_violation_count_is_not_durable(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        candidate = {
            "finding_type": "result",
            "metrics": {
                "score": 1.0,
                "effort_ratio": 1.0,
                "coverage_ratio": 1.0,
                "protocol_integrity_violation_count": 1,
            },
        }

        self.assertTrue(frontier._candidate_protocol_integrity_failed(candidate))
        self.assertFalse(
            frontier._is_durable_frontier_entry(candidate, {"require_ratio_gate": True})
        )
        protocol_failed = {
            "finding_type": "result",
            "metrics": {
                "score": 1.0,
                "effort_ratio": 1.0,
                "coverage_ratio": 1.0,
                "protocol_integrity_passed": False,
            },
        }
        self.assertTrue(frontier._candidate_protocol_integrity_failed(protocol_failed))
        self.assertFalse(
            frontier._is_durable_frontier_entry(
                protocol_failed,
                {"require_ratio_gate": True},
            )
        )

    def test_required_eval_coverage_ratio_is_not_actual_maturity_coverage(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
            evidence_maturity_snapshot,
        )

        snapshot = evidence_maturity_snapshot(
            {
                "score": 1.0,
                "effort_ratio": 1.0,
                "required_eval_coverage_ratio": 0.8,
            },
            {
                "min_effort_ratio": 0.75,
                "min_coverage_ratio": 0.80,
                "require_ratio_gate": True,
            },
        )

        self.assertIsNone(snapshot["coverage_ratio"])
        self.assertIsNone(snapshot["mature_enough"])
        persisted_gem_snapshot = evidence_maturity_snapshot(
            {
                "score": 1.0,
                "admission_metrics": {
                    "effort_ratio": 0.9,
                    "coverage_ratio": 0.85,
                },
            },
            {
                "min_effort_ratio": 0.75,
                "min_coverage_ratio": 0.80,
                "require_ratio_gate": True,
            },
        )
        self.assertTrue(persisted_gem_snapshot["mature_enough"])

    def test_late_quarantined_marker_blocks_durable_frontier_but_keeps_validation_signal(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        policy = {
            "min_effort_ratio": 0.75,
            "min_coverage_ratio": 0.80,
            "require_ratio_gate": True,
        }
        candidate = {
            "id": "late-ratio",
            "finding_type": "result",
            "variant_name": "late_ratio",
            "metrics": {
                "score": 10.0,
                "effort_ratio": 0.9,
                "coverage_ratio": 0.9,
                "late_after_generation_boundary": True,
                "late_result_policy": "quarantined_signal",
            },
        }

        self.assertFalse(frontier._is_durable_frontier_entry(candidate, maturity_policy=policy))
        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
                maturity_policy=policy,
            )

            promoted = store.promote(0, [candidate])
            manifest = store.get_manifest()

        self.assertEqual(promoted, [])
        validation = manifest["validation_candidates"]["generations"]["0"]
        self.assertEqual(validation[0]["finding_id"], "late-ratio")
        self.assertEqual(validation[0]["recommended_next_step"], "rerun_or_revalidate_late_signal")
        self.assertTrue(validation[0]["mature_enough"])

    def test_ingested_validation_only_marker_blocks_durable_frontier(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            findings_ingest,
        )

        policy = {
            "min_effort_ratio": 0.75,
            "min_coverage_ratio": 0.80,
            "require_ratio_gate": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            finding_path = root / "gen0_peer0_validation_only.json"
            finding_path.write_text(
                json.dumps(
                    {
                        "id": "validation-only-ratio",
                        "finding_type": "result",
                        "variant_name": "validation_only_ratio",
                        "metrics": {
                            "score": 10.0,
                            "effort_ratio": 0.9,
                            "coverage_ratio": 0.9,
                            "scored_complete": True,
                            "validation_only": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            parsed = findings_ingest.parse_finding_file(finding_path, primary_metric="score")
            self.assertIsNotNone(parsed)
            assert parsed is not None

            store = frontier.FrontierStore(
                root / "frontier",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
                maturity_policy=policy,
            )

            promoted = store.promote(0, [parsed])
            manifest = store.get_manifest()

        self.assertEqual(promoted, [])
        validation = manifest["validation_candidates"]["generations"]["0"]
        self.assertEqual(validation[0]["finding_id"], parsed["id"])
        self.assertEqual(validation[0]["variant_name"], "validation_only_ratio")
        self.assertEqual(
            validation[0]["recommended_next_step"],
            "rerun_or_revalidate_late_signal",
        )
        self.assertTrue(validation[0]["metrics"]["validation_only"])
        self.assertTrue(validation[0]["mature_enough"])

    def test_validation_artifact_blocks_same_source_variant_alias_only(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        policy = {
            "min_effort_ratio": 0.75,
            "min_coverage_ratio": 0.80,
            "require_ratio_gate": True,
        }
        shared_source = {
            "child_variant_id": "candidate",
            "source_result_path": "results/candidate/summary.json",
            "source_result_sha256": "same-sha",
        }
        findings = [
            {
                "id": "quarantine",
                "finding_type": "result",
                "variant_name": "preliminary_alias",
                "metrics": {
                    **shared_source,
                    "validation_only_result": True,
                    "score": 1.0,
                },
            },
            {
                "id": "mature-alias",
                "finding_type": "result",
                "variant_name": "complete_alias",
                "metrics": {
                    **shared_source,
                    "score": 2.0,
                    "effort_ratio": 1.0,
                    "coverage_ratio": 1.0,
                    "scored_complete": True,
                },
            },
            {
                "id": "independent",
                "finding_type": "result",
                "variant_name": "independent",
                "metrics": {
                    "source_result_path": "results/independent/summary.json",
                    "source_result_sha256": "independent-sha",
                    "score": 1.5,
                    "effort_ratio": 1.0,
                    "coverage_ratio": 1.0,
                    "scored_complete": True,
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                promote_top_k=3,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
                maturity_policy=policy,
            )
            promoted = store.promote(0, findings)
            manifest = store.get_manifest()

        self.assertEqual([entry["finding_id"] for entry in promoted], ["independent"])
        validation_ids = {
            entry["finding_id"] for entry in manifest["validation_candidates"]["generations"]["0"]
        }
        self.assertEqual(validation_ids, {"quarantine", "mature-alias"})

    def test_preliminary_snapshot_quarantines_only_same_immutable_artifact(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        policy = {
            "min_effort_ratio": 0.75,
            "min_coverage_ratio": 0.80,
            "preliminary_stage_labels": ["preliminary"],
        }
        findings = [
            {
                "id": "preliminary",
                "finding_type": "result",
                "variant_name": "candidate",
                "metrics": {
                    "child_variant_id": "candidate",
                    "evidence_stage": "preliminary",
                    "effort_ratio": 0.2,
                    "coverage_ratio": 0.2,
                    "source_result_path": "results/candidate.json",
                    "source_result_sha256": "same-sha",
                    "score": 1.0,
                },
            },
            {
                "id": "same-snapshot-alias",
                "finding_type": "result",
                "variant_name": "candidate_alias",
                "metrics": {
                    "child_variant_id": "candidate",
                    "effort_ratio": 1.0,
                    "coverage_ratio": 1.0,
                    "scored_complete": True,
                    "source_result_path": "results/candidate.json",
                    "source_result_sha256": "same-sha",
                    "score": 2.0,
                },
            },
            {
                "id": "rewritten-path-only",
                "finding_type": "result",
                "variant_name": "rewritten",
                "metrics": {
                    "effort_ratio": 1.0,
                    "coverage_ratio": 1.0,
                    "scored_complete": True,
                    "source_result_path": "results/candidate.json",
                    "score": 1.5,
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                promote_top_k=3,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
                maturity_policy=policy,
            )
            promoted = store.promote(0, findings)

        self.assertEqual([entry["finding_id"] for entry in promoted], ["rewritten-path-only"])

    def test_shared_parent_variant_id_does_not_quarantine_distinct_complete_result(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        findings = [
            {
                "id": "preliminary-child-a",
                "finding_type": "result",
                "variant_name": "child-a",
                "metrics": {
                    "variant_id": "shared-parent",
                    "source_result_path": "results/shared.json",
                    "source_result_sha256": "shared-sha",
                    "evidence_stage": "preliminary",
                    "scored_complete": False,
                    "score": 1.0,
                },
            },
            {
                "id": "complete-child-b",
                "finding_type": "result",
                "variant_name": "child-b",
                "metrics": {
                    "variant_id": "shared-parent",
                    "source_result_path": "results/shared.json",
                    "source_result_sha256": "shared-sha",
                    "scored_complete": True,
                    "score": 2.0,
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
            )

            promoted = store.promote(0, findings)

        self.assertEqual([entry["finding_id"] for entry in promoted], ["complete-child-b"])

    def test_ratio_immature_snapshot_quarantines_same_artifact_alias(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        policy = {
            "min_effort_ratio": 0.75,
            "min_coverage_ratio": 0.80,
        }
        shared_source = {
            "child_variant_id": "candidate",
            "source_result_path": "results/candidate.json",
            "source_result_sha256": "same-sha",
        }
        findings = [
            {
                "id": "ratio-immature",
                "finding_type": "result",
                "variant_name": "candidate",
                "metrics": {
                    **shared_source,
                    "effort_ratio": 0.2,
                    "coverage_ratio": 0.2,
                    "score": 1.0,
                },
            },
            {
                "id": "same-snapshot-alias",
                "finding_type": "result",
                "variant_name": "candidate_alias",
                "metrics": {
                    **shared_source,
                    "effort_ratio": 1.0,
                    "coverage_ratio": 1.0,
                    "scored_complete": True,
                    "score": 2.0,
                },
            },
            {
                "id": "independent",
                "finding_type": "result",
                "variant_name": "independent",
                "metrics": {
                    "source_result_path": "results/independent.json",
                    "source_result_sha256": "independent-sha",
                    "effort_ratio": 1.0,
                    "coverage_ratio": 1.0,
                    "scored_complete": True,
                    "score": 1.5,
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                promote_top_k=3,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
                maturity_policy=policy,
            )
            promoted = store.promote(0, findings)
            manifest = store.get_manifest()

        self.assertEqual([entry["finding_id"] for entry in promoted], ["independent"])
        validation_ids = {
            entry["finding_id"] for entry in manifest["validation_candidates"]["generations"]["0"]
        }
        self.assertEqual(validation_ids, {"ratio-immature", "same-snapshot-alias"})

    def test_same_path_different_digests_remain_distinct_frontier_entities(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        findings = [
            {
                "id": "first",
                "finding_type": "result",
                "variant_name": "candidate_a",
                "metrics": {
                    "source_result_path": "results/shared.json",
                    "source_result_sha256": "sha-a",
                    "effort_ratio": 1.0,
                    "coverage_ratio": 1.0,
                    "scored_complete": True,
                    "score": 2.0,
                },
            },
            {
                "id": "second",
                "finding_type": "result",
                "variant_name": "candidate_b",
                "metrics": {
                    "source_result_path": "results/shared.json",
                    "source_result_sha256": "sha-b",
                    "effort_ratio": 1.0,
                    "coverage_ratio": 1.0,
                    "scored_complete": True,
                    "score": 1.5,
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
                maturity_policy={},
            )
            promoted = store.promote(0, findings)

        self.assertEqual([entry["finding_id"] for entry in promoted], ["first", "second"])

    def test_shared_artifact_quarantine_is_scoped_to_producer_identity(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        shared = {
            "source_result_path": "results/shared.json",
            "source_result_sha256": "shared-sha",
        }
        findings = [
            {
                "id": "preliminary-child",
                "finding_type": "result",
                "variant_name": "child-a",
                "metrics": {
                    **shared,
                    "child_variant_id": "child-a",
                    "evidence_stage": "preliminary",
                    "scored_complete": False,
                    "score": 1.0,
                },
            },
            {
                "id": "complete-child",
                "finding_type": "result",
                "variant_name": "child-b",
                "metrics": {
                    **shared,
                    "child_variant_id": "child-b",
                    "scored_complete": True,
                    "score": 2.0,
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                promote_top_k=2,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
                maturity_policy={},
            )
            promoted = store.promote(0, findings)

        self.assertEqual([entry["finding_id"] for entry in promoted], ["complete-child"])

    def test_legacy_suspect_alias_true_overrides_false_protocol_flag(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        candidate = {
            "finding_type": "result",
            "variant_name": "legacy_suspect_conflict",
            "metric_value": 1.0,
            "metrics": {
                "score": 1.0,
                "suspect_protocol": False,
                "suspect_fixed_weight_eval": True,
            },
        }

        self.assertTrue(frontier._candidate_protocol_integrity_failed(candidate))
        self.assertFalse(frontier._is_durable_frontier_entry(candidate))

    def test_incubator_lane_without_new_high_keeps_existing_top_k_behavior(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        lane = {
            "name": "incubator",
            "k": 5,
            "include_lanes": ["incubator"],
            "axes": [{"metric": "score", "direction": "maximize"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
                frontier_lanes=[lane],
            )
            store.promote(
                0,
                [
                    {
                        "id": "first",
                        "finding_type": "result",
                        "variant_name": "first",
                        "metrics": {
                            "score": 1.0,
                            "frontier_lane": "incubator",
                            "evidence_stage": "full_eval",
                            "scored_complete": True,
                        },
                    }
                ],
            )
            promoted = store.promote(
                1,
                [
                    {
                        "id": "non_new_high",
                        "finding_type": "result",
                        "variant_name": "non_new_high",
                        "metrics": {
                            "score": 0.9,
                            "frontier_lane": "incubator",
                            "evidence_stage": "full_eval",
                            "scored_complete": True,
                        },
                    }
                ],
            )

        self.assertEqual([entry["finding_id"] for entry in promoted], ["non_new_high"])

    def test_incubator_lane_admits_only_new_pareto_points_when_enabled(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        lane = {
            "name": "incubator",
            "k": 5,
            "include_lanes": ["incubator"],
            "axes": [{"metric": "score", "direction": "maximize"}],
            "admit_new_high": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
                frontier_lanes=[lane],
            )
            self.assertEqual(
                len(
                    store.promote(
                        0,
                        [
                            {
                                "id": "first",
                                "finding_type": "result",
                                "variant_name": "first",
                                "metrics": {
                                    "score": 1.0,
                                    "frontier_lane": "incubator",
                                    "evidence_stage": "full_eval",
                                    "scored_complete": True,
                                },
                            }
                        ],
                    )
                ),
                1,
            )
            self.assertEqual(
                store.promote(
                    1,
                    [
                        {
                            "id": "dominated",
                            "finding_type": "result",
                            "variant_name": "dominated",
                            "metrics": {
                                "score": 0.9,
                                "frontier_lane": "incubator",
                                "evidence_stage": "full_eval",
                                "scored_complete": True,
                            },
                        }
                    ],
                ),
                [],
            )
            manifest = store.get_manifest()
            self.assertEqual(
                [entry["finding_id"] for entry in manifest["lane_frontiers"]["incubator"]],
                ["first"],
            )

    def test_incubator_lane_does_not_readmit_an_equal_pareto_point(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        lane = {
            "name": "incubator",
            "k": 5,
            "cumulative_cap": 1,
            "include_lanes": ["incubator"],
            "axes": [{"metric": "score", "direction": "maximize"}],
            "admit_new_high": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
                frontier_lanes=[lane],
            )
            store.promote(
                0,
                [
                    {
                        "id": "first",
                        "finding_type": "result",
                        "variant_name": "first",
                        "metrics": {
                            "score": 1.0,
                            "frontier_lane": "incubator",
                            "evidence_stage": "full_eval",
                            "scored_complete": True,
                        },
                    }
                ],
            )
            promoted = store.promote(
                1,
                [
                    {
                        "id": "equal",
                        "finding_type": "result",
                        "variant_name": "equal",
                        "metrics": {
                            "score": 1.0,
                            "frontier_lane": "incubator",
                            "evidence_stage": "full_eval",
                            "scored_complete": True,
                        },
                    }
                ],
            )

            lane_entries = store.get_manifest()["lane_frontiers"]["incubator"]

        self.assertEqual(promoted, [])
        self.assertEqual([entry["finding_id"] for entry in lane_entries], ["first"])

    def test_lane_cumulative_cap_bounds_distinct_pareto_points(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        lane = {
            "name": "incubator",
            "k": 3,
            "cumulative_cap": 2,
            "include_lanes": ["incubator"],
            "axes": [
                {"metric": "quality", "direction": "maximize"},
                {"metric": "cost", "direction": "minimize"},
            ],
            "admit_new_high": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                promote_top_k=1,
                primary_metric="quality",
                metric_direction="maximize",
                require_tier=False,
                frontier_lanes=[lane],
            )
            store.promote(
                0,
                [
                    {
                        "id": f"point-{idx}",
                        "finding_type": "result",
                        "variant_name": f"point-{idx}",
                        "metrics": {
                            "quality": float(idx),
                            "cost": float(idx),
                            "frontier_lane": "incubator",
                            "evidence_stage": "full_eval",
                            "scored_complete": True,
                        },
                    }
                    for idx in range(1, 4)
                ],
            )
            lane_entries = store.get_manifest()["lane_frontiers"]["incubator"]

        self.assertEqual(len(lane_entries), 2)

    def test_same_generation_lane_repromotion_is_idempotent(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        lanes = [
            {
                "name": "confirmed",
                "k": 4,
                "include_lanes": ["performance"],
                "axes": [
                    {"metric": "quality", "direction": "maximize"},
                    {"metric": "cost", "direction": "minimize"},
                ],
                "admit_new_high": True,
            },
            {
                "name": "incubator",
                "k": 4,
                "include_lanes": ["performance"],
                "axes": [
                    {"metric": "quality", "direction": "maximize"},
                    {"metric": "cost", "direction": "minimize"},
                ],
                "admit_new_high": True,
            },
        ]

        def finding(finding_id: str, quality: float, cost: float) -> dict[str, object]:
            return {
                "id": finding_id,
                "finding_type": "result",
                "variant_name": finding_id,
                "metrics": {
                    "quality": quality,
                    "cost": cost,
                    "frontier_lane": "performance",
                    "evidence_stage": "full_eval",
                    "scored_complete": True,
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                promote_top_k=1,
                primary_metric="quality",
                metric_direction="maximize",
                require_tier=False,
                frontier_lanes=lanes,
            )
            store.promote(0, [finding("anchor", 5.0, 5.0)])
            generation_findings = [
                finding("high_quality", 10.0, 4.0),
                finding("low_cost", 8.0, 1.0),
            ]
            first = store.promote(1, generation_findings)
            first_manifest = json.dumps(store.get_manifest(), sort_keys=True)
            first_local = {
                path.name: path.read_bytes()
                for path in sorted((Path(tmp) / "frontier" / "gen_1").glob("top_*_finding.json"))
            }

            second = store.promote(1, generation_findings)
            second_manifest = json.dumps(store.get_manifest(), sort_keys=True)
            second_local = {
                path.name: path.read_bytes()
                for path in sorted((Path(tmp) / "frontier" / "gen_1").glob("top_*_finding.json"))
            }

        self.assertEqual(first, second)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_local, second_local)
        self.assertEqual(
            {(entry["finding_id"], entry["frontier_lane"]) for entry in second},
            {("high_quality", "confirmed"), ("low_cost", "confirmed")},
        )

    def test_repromotion_removes_stale_generation_rank_artifacts(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        lane = {
            "name": "incubator",
            "k": 3,
            "include_lanes": ["candidate"],
            "axes": [{"metric": "score", "direction": "maximize"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots = []
            for index in range(3):
                snapshot = root / f"snapshot_{index}"
                snapshot.mkdir()
                (snapshot / "state.txt").write_text(str(index), encoding="utf-8")
                snapshots.append(snapshot)

            store = frontier.FrontierStore(
                root / "frontier",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                require_tier=False,
                frontier_lanes=[lane],
            )
            findings = [
                {
                    "id": f"candidate-{index}",
                    "finding_type": "result",
                    "variant_name": f"candidate-{index}",
                    "snapshot_local_path": str(snapshots[index]),
                    "metrics": {
                        "score": float(3 - index),
                        "frontier_lane": "candidate",
                        "evidence_stage": "full_eval",
                        "scored_complete": True,
                    },
                }
                for index in range(3)
            ]
            store.promote(1, findings)
            gen_dir = root / "frontier" / "gen_1"
            (gen_dir / "operator_note.txt").write_text("keep", encoding="utf-8")
            self.assertTrue((gen_dir / "top_3_finding.json").exists())
            self.assertTrue((gen_dir / "top_3_snapshot.tar.gz").exists())

            promoted = store.promote(1, findings[:1])

            remaining = sorted(path.name for path in gen_dir.iterdir())
            manifest_entries = store.get_manifest()["generations"]["1"]

        self.assertEqual([entry["finding_id"] for entry in promoted], ["candidate-0"])
        self.assertEqual([entry["finding_id"] for entry in manifest_entries], ["candidate-0"])
        self.assertEqual(
            remaining,
            [
                "operator_note.txt",
                "top_1_finding.json",
                "top_1_snapshot.tar.gz",
            ],
        )

    def test_same_generation_refresh_keeps_other_current_points_as_new_high_anchors(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        lane = {
            "name": "incubator",
            "k": 5,
            "cumulative_cap": 1,
            "include_lanes": ["candidate"],
            "axes": [{"metric": "score", "direction": "maximize"}],
            "admit_new_high": True,
        }

        def finding(finding_id: str, score: float) -> dict[str, object]:
            return {
                "id": finding_id,
                "finding_type": "result",
                "variant_name": finding_id,
                "metrics": {
                    "score": score,
                    "frontier_lane": "candidate",
                    "evidence_stage": "full_eval",
                    "scored_complete": True,
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                primary_metric="score",
                frontier_lanes=[lane],
            )
            store.promote(0, [finding("old_anchor", 10.0)])
            first = store.promote(1, [finding("current_high", 11.0)])
            refreshed = store.promote(
                1,
                [
                    finding("current_high", 11.0),
                    finding("late_but_dominated", 9.0),
                ],
            )

        self.assertEqual([entry["finding_id"] for entry in first], ["current_high"])
        self.assertEqual([entry["finding_id"] for entry in refreshed], ["current_high"])

    def test_boundary_evidence_signature_ignores_observation_telemetry(self) -> None:
        import asyncio
        import copy

        from praxist.plugins.workflow_stages.research_loop.backend import generation_boundary

        raw_finding = {
            "id": "stable-result",
            "finding_type": "result",
            "variant_name": "stable-result",
            "timestamp": "2026-07-25T00:00:00+00:00",
            "metrics": {
                "score": 1.0,
                "source_result_path": "results/stable/summary.json",
                "source_result_sha256": "sha-stable",
            },
            "design_dimensions": {"mechanism": "a"},
        }
        annotated = copy.deepcopy(raw_finding)
        annotated["timestamp"] = "2026-07-25T00:00:01+00:00"
        annotated["metrics"].update(
            {
                "diversity_overlap_status": "clone",
                "diversity_overlap_no_data_reason": "no_common_dimensions",
                "diversity_violated": True,
                "diversity_narrow_variation": False,
                "diversity_overlap_count": 1,
                "diversity_overlap_total": 1,
                "diversity_overlap_fraction": 1.0,
                "diversity_most_similar_anchor": "anchor",
            }
        )
        self.assertEqual(
            generation_boundary._canonical_evidence_signature([raw_finding]),
            generation_boundary._canonical_evidence_signature([annotated]),
        )
        changed = copy.deepcopy(raw_finding)
        changed["metrics"]["score"] = 2.0
        self.assertNotEqual(
            generation_boundary._canonical_evidence_signature([raw_finding]),
            generation_boundary._canonical_evidence_signature([changed]),
        )

        class CountingFrontier:
            def __init__(self) -> None:
                self.calls = 0

            def get_summary(self):
                return [
                    {
                        "finding_id": "anchor",
                        "variant_name": "anchor",
                        "design_dimensions": {"mechanism": "a"},
                    }
                ]

            def promote(self, gen_id, findings):
                self.calls += 1
                return [
                    {
                        "generation_id": gen_id,
                        "finding_id": findings[0]["id"],
                        "frontier_lane": "incubator",
                    }
                ]

        with tempfile.TemporaryDirectory() as tmp:
            counting_frontier = CountingFrontier()
            loop = SimpleNamespace(
                run_dir=Path(tmp),
                _strategy_for_gen=lambda gen_id: "pi_directed",
                _collect_findings_for_generation=lambda gen_id: [copy.deepcopy(raw_finding)],
                frontier=counting_frontier,
                task_spec=SimpleNamespace(
                    evaluation=SimpleNamespace(diversity_dimensions=[{"name": "mechanism"}]),
                    research_memory=SimpleNamespace(enabled=False),
                    generation_policy=SimpleNamespace(max_generations=2),
                ),
                _graph_maintainer=None,
                _findings_sync=None,
                gems=None,
            )
            asyncio.run(
                generation_boundary.complete_generation_boundary(
                    loop,
                    gen_id=1,
                    pi_agent=None,
                    pi_cfg=SimpleNamespace(strict=False),
                )
            )

        self.assertEqual(counting_frontier.calls, 1)

    def test_boundary_late_evidence_refresh_preserves_existing_pareto_pick(self) -> None:
        import asyncio
        import copy

        from praxist.plugins.workflow_stages.research_loop.backend import (
            frontier,
            generation_boundary,
        )

        lane = {
            "name": "incubator",
            "k": 4,
            "include_lanes": ["candidate"],
            "axes": [
                {"metric": "quality", "direction": "maximize"},
                {"metric": "cost", "direction": "minimize"},
            ],
            "admit_new_high": True,
        }

        def finding(finding_id: str, quality: float, cost: float) -> dict[str, object]:
            return {
                "id": finding_id,
                "finding_type": "result",
                "variant_name": finding_id,
                "source_result_path": f"results/{finding_id}/summary.json",
                "metrics": {
                    "quality": quality,
                    "cost": cost,
                    "source_result_path": f"results/{finding_id}/summary.json",
                    "source_result_sha256": f"sha-{finding_id}",
                    "frontier_lane": "candidate",
                    "evidence_stage": "full_eval",
                    "scored_complete": True,
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            store = frontier.FrontierStore(
                run_dir / "frontier",
                promote_top_k=1,
                primary_metric="quality",
                metric_direction="maximize",
                require_tier=False,
                frontier_lanes=[lane],
            )
            store.promote(0, [finding("anchor", 5.0, 5.0)])
            initial = finding("high_quality", 10.0, 4.0)
            late = finding("low_cost", 8.0, 1.0)
            collections = [[initial], [initial, late]]

            def collect(_gen_id):
                return copy.deepcopy(collections.pop(0))

            loop = SimpleNamespace(
                run_dir=run_dir,
                _strategy_for_gen=lambda gen_id: "pi_directed",
                _collect_findings_for_generation=collect,
                frontier=store,
                task_spec=SimpleNamespace(
                    evaluation=SimpleNamespace(diversity_dimensions=[]),
                    research_memory=SimpleNamespace(enabled=False),
                    generation_policy=SimpleNamespace(max_generations=2),
                ),
                _graph_maintainer=None,
                _findings_sync=None,
                gems=None,
            )
            asyncio.run(
                generation_boundary.complete_generation_boundary(
                    loop,
                    gen_id=1,
                    pi_agent=None,
                    pi_cfg=SimpleNamespace(strict=False),
                )
            )

            generation_entries = store.get_manifest()["generations"]["1"]
            local_files = sorted(
                path.name for path in (run_dir / "frontier" / "gen_1").glob("top_*_finding.json")
            )
            marker = json.loads(
                (run_dir / "gen_1" / "generation_boundary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(
            {entry["finding_id"] for entry in generation_entries},
            {"high_quality", "low_cost"},
        )
        self.assertEqual(local_files, ["top_1_finding.json", "top_2_finding.json"])
        self.assertEqual(marker["promoted_count"], 2)

    def test_promoted_maturity_metadata_overrides_stale_candidate_claim(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                promote_top_k=1,
                primary_metric="score",
                maturity_policy={"require_ratio_gate": True},
            )
            store.promote(
                0,
                [
                    {
                        "id": "fresh-maturity",
                        "finding_type": "result",
                        "variant_name": "fresh-maturity",
                        "metrics": {
                            "score": 1.0,
                            "effort_ratio": 1.0,
                            "coverage_ratio": 1.0,
                            "mature_enough": False,
                            "maturity_basis": "stale",
                        },
                    }
                ],
            )
            entry = store.get_manifest()["generations"]["0"][0]

        self.assertTrue(entry["mature_enough"])
        self.assertTrue(entry["metrics"]["mature_enough"])
        self.assertEqual(entry["metrics"]["maturity_basis"], "effort_coverage_ratio")
        self.assertTrue(frontier._is_durable_frontier_entry(entry))

    def test_incubator_lane_retains_multi_axis_pareto_points_cumulatively(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        lane = {
            "name": "incubator",
            "k": 1,
            "include_lanes": ["incubator"],
            "axes": [
                {"metric": "quality", "direction": "maximize"},
                {"metric": "cost", "direction": "minimize"},
            ],
            "admit_new_high": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                promote_top_k=1,
                primary_metric="quality",
                metric_direction="maximize",
                require_tier=False,
                frontier_lanes=[lane],
            )
            store.promote(
                0,
                [
                    {
                        "id": "high_quality",
                        "finding_type": "result",
                        "variant_name": "high_quality",
                        "metrics": {
                            "quality": 10.0,
                            "cost": 5.0,
                            "frontier_lane": "incubator",
                            "evidence_stage": "full_eval",
                            "scored_complete": True,
                        },
                    },
                    {
                        "id": "low_cost",
                        "finding_type": "result",
                        "variant_name": "low_cost",
                        "metrics": {
                            "quality": 8.0,
                            "cost": 1.0,
                            "frontier_lane": "incubator",
                            "evidence_stage": "full_eval",
                            "scored_complete": True,
                        },
                    },
                ],
            )
            lane_ids = [
                entry["finding_id"] for entry in store.get_manifest()["lane_frontiers"]["incubator"]
            ]

        self.assertEqual(set(lane_ids), {"high_quality", "low_cost"})

    def test_allow_lower_tier_retains_signal_on_validation_surface(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
            _digest_validation_candidates,
        )

        lane = {
            "name": "follow_up",
            "k": 2,
            "include_lanes": ["candidate"],
            "axes": [{"metric": "score", "direction": "maximize"}],
            "allow_lower_tier": True,
            "parent_eligible": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                primary_metric="score",
                frontier_lanes=[lane],
            )
            promoted = store.promote(
                0,
                [
                    {
                        "id": "promising_partial",
                        "finding_type": "result",
                        "variant_name": "promising_partial",
                        "metrics": {
                            "score": 0.8,
                            "frontier_lane": "candidate",
                            "scout_only": True,
                            "scored_complete": False,
                        },
                    }
                ],
            )
            manifest = store.get_manifest()
            prompt_visible = _digest_validation_candidates(Path(tmp), current_gen_id=0)

        self.assertEqual(promoted, [])
        self.assertEqual(manifest.get("lane_frontiers", {}).get("follow_up", []), [])
        self.assertEqual(
            manifest["validation_candidates"]["cumulative"][0]["retained_validation_lanes"],
            ["follow_up"],
        )
        self.assertEqual(manifest["cumulative_top"], [])
        self.assertEqual(
            manifest["validation_candidates"]["cumulative"][0]["finding_id"],
            "promising_partial",
        )
        self.assertEqual(prompt_visible[0]["finding_id"], "promising_partial")
        self.assertEqual(prompt_visible[0]["retained_validation_lanes"], ["follow_up"])

    def test_same_mature_entity_does_not_consume_multiple_lane_slots(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        lanes = [
            {
                "name": name,
                "k": 1,
                "include_lanes": ["performance"],
                "axes": [{"metric": "score", "direction": "maximize"}],
            }
            for name in ("leader", "long_term_candidates")
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                primary_metric="score",
                frontier_lanes=lanes,
            )
            store.promote(
                0,
                [
                    {
                        "id": "shared_best",
                        "finding_type": "result",
                        "variant_name": "shared_best",
                        "metrics": {
                            "score": 1.0,
                            "frontier_lane": "performance",
                            "scored_complete": True,
                        },
                    }
                ],
            )
            lane_frontiers = store.get_manifest()["lane_frontiers"]

        self.assertEqual(lane_frontiers["leader"][0]["finding_id"], "shared_best")
        self.assertEqual(lane_frontiers["long_term_candidates"], [])

    def test_lane_capacity_deduplicates_same_immutable_artifact_aliases(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        lane = {
            "name": "incubator",
            "k": 2,
            "cumulative_cap": 2,
            "include_lanes": ["incubator"],
            "axes": [{"metric": "score", "direction": "maximize"}],
        }
        shared_coordinates = {
            "source_result_path": "results/shared/result_summary.json",
            "source_result_sha256": "shared-sha",
        }
        findings = [
            {
                "id": "artifact_alias_low",
                "finding_type": "result",
                "variant_name": "artifact_alias_low",
                "variant_id": "lineage_alias_low",
                **shared_coordinates,
                "metrics": {
                    "score": 2.0,
                    "frontier_lane": "incubator",
                    "scored_complete": True,
                },
            },
            {
                "id": "artifact_alias_high",
                "finding_type": "result",
                "variant_name": "artifact_alias_high",
                "variant_id": "lineage_alias_high",
                **shared_coordinates,
                "metrics": {
                    "score": 3.0,
                    "frontier_lane": "incubator",
                    "scored_complete": True,
                },
            },
            {
                "id": "independent",
                "finding_type": "result",
                "variant_name": "independent",
                "variant_id": "independent",
                "source_result_path": "results/independent/result_summary.json",
                "source_result_sha256": "independent-sha",
                "metrics": {
                    "score": 1.0,
                    "frontier_lane": "incubator",
                    "scored_complete": True,
                },
            },
        ]

        self.assertNotEqual(
            frontier._candidate_entity_key(findings[0]),
            frontier._candidate_entity_key(findings[1]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                primary_metric="score",
                require_tier=False,
                frontier_lanes=[lane],
            )
            promoted = store.promote(0, findings)
            lane_entries = store.get_manifest()["lane_frontiers"]["incubator"]

        self.assertEqual(
            [entry["finding_id"] for entry in promoted],
            ["artifact_alias_high", "independent"],
        )
        self.assertEqual(
            [entry["finding_id"] for entry in lane_entries],
            ["artifact_alias_high", "independent"],
        )
        self.assertEqual(lane_entries[0]["frontier_entity_key"], "variant::lineage_alias_high")
        self.assertEqual(lane_entries[0]["variant_id"], "lineage_alias_high")

    def test_lane_cumulative_deduplicates_artifact_aliases_across_generations(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        lane = {
            "name": "incubator",
            "k": 1,
            "cumulative_cap": 2,
            "include_lanes": ["incubator"],
            "axes": [{"metric": "score", "direction": "maximize"}],
        }

        def alias(finding_id: str, variant_id: str, score: float) -> dict[str, object]:
            return {
                "id": finding_id,
                "finding_type": "result",
                "variant_name": finding_id,
                "variant_id": variant_id,
                "source_result_path": "results/shared/result_summary.json",
                "source_result_sha256": "shared-sha",
                "metrics": {
                    "score": score,
                    "frontier_lane": "incubator",
                    "scored_complete": True,
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                primary_metric="score",
                require_tier=False,
                frontier_lanes=[lane],
            )
            store.promote(0, [alias("old_alias", "old_lineage_alias", 1.0)])
            store.promote(1, [alias("new_alias", "new_lineage_alias", 2.0)])
            manifest = store.get_manifest()

            # Simulate a pre-upgrade manifest whose derived lane views still
            # contain aliases of one immutable result artifact.
            old_entry = manifest["generations"]["0"][0]
            new_entry = manifest["generations"]["1"][0]
            store._manifest["lane_frontiers"]["incubator"] = [old_entry, new_entry]
            store._manifest["cumulative_top"] = [old_entry, new_entry]
            store._save_manifest()
            resumed = frontier.FrontierStore(
                Path(tmp) / "frontier",
                primary_metric="score",
                require_tier=False,
                frontier_lanes=[lane],
            )
            manifest = resumed.get_manifest()

        self.assertEqual(len(manifest["generations"]["0"]), 1)
        self.assertEqual(len(manifest["generations"]["1"]), 1)
        self.assertEqual(
            [entry["finding_id"] for entry in manifest["lane_frontiers"]["incubator"]],
            ["new_alias"],
        )
        self.assertEqual(
            [entry["finding_id"] for entry in manifest["cumulative_top"]],
            ["new_alias"],
        )
        self.assertEqual(
            manifest["cumulative_top"][0]["frontier_entity_key"],
            "variant::new_lineage_alias",
        )

    def test_lane_cumulative_deduplicates_artifact_aliases_across_lanes(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        lanes = [
            {
                "name": name,
                "k": 1,
                "cumulative_cap": 1,
                "include_lanes": [name],
                "axes": [{"metric": "score", "direction": "maximize"}],
            }
            for name in ("confirmed", "incubator")
        ]

        def alias(finding_id: str, source_lane: str) -> dict[str, object]:
            return {
                "id": finding_id,
                "finding_type": "result",
                "variant_name": finding_id,
                "variant_id": f"{finding_id}_lineage_alias",
                "source_result_path": "results/shared/result_summary.json",
                "source_result_sha256": "shared-sha",
                "metrics": {
                    "score": 1.0,
                    "frontier_lane": source_lane,
                    "scored_complete": True,
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                primary_metric="score",
                require_tier=False,
                frontier_lanes=lanes,
            )
            store.promote(0, [alias("confirmed_alias", "confirmed")])
            store.promote(1, [alias("incubator_alias", "incubator")])
            manifest = store.get_manifest()

        self.assertEqual(
            [entry["finding_id"] for entry in manifest["lane_frontiers"]["confirmed"]],
            ["confirmed_alias"],
        )
        self.assertEqual(manifest["lane_frontiers"]["incubator"], [])
        self.assertEqual(
            [entry["finding_id"] for entry in manifest["cumulative_top"]],
            ["confirmed_alias"],
        )

    def test_lane_capacity_retains_independent_artifacts_from_one_lineage(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        lane = {
            "name": "incubator",
            "k": 3,
            "cumulative_cap": 3,
            "include_lanes": ["incubator"],
            "axes": [{"metric": "score", "direction": "maximize"}],
        }
        coordinates = [
            ("results/replication-a/result_summary.json", "sha-a"),
            ("results/replication-a/result_summary.json", "sha-b"),
            ("results/replication-b/result_summary.json", "sha-a"),
        ]
        findings = [
            {
                "id": "shared_finding",
                "finding_type": "result",
                "variant_name": f"replication_{idx}",
                "variant_id": "shared_lineage",
                "source_result_path": path,
                "source_result_sha256": sha256,
                "metrics": {
                    "score": float(4 - idx),
                    "frontier_lane": "incubator",
                    "scored_complete": True,
                },
            }
            for idx, (path, sha256) in enumerate(coordinates, start=1)
        ]

        self.assertEqual(
            {frontier._candidate_entity_key(finding) for finding in findings},
            {"variant::shared_lineage"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                primary_metric="score",
                require_tier=False,
                frontier_lanes=[lane],
            )
            promoted = store.promote(0, findings)
            manifest = store.get_manifest()

        self.assertEqual(len(promoted), 3)
        self.assertEqual(len(manifest["lane_frontiers"]["incubator"]), 3)
        self.assertEqual(len(manifest["cumulative_top"]), 3)
        self.assertEqual(
            {frontier.result_artifact_key(entry) for entry in manifest["cumulative_top"]},
            set(coordinates),
        )
        self.assertEqual(
            {entry["frontier_entity_key"] for entry in manifest["cumulative_top"]},
            {"variant::shared_lineage"},
        )
        self.assertEqual(
            {entry["variant_id"] for entry in manifest["cumulative_top"]},
            {"shared_lineage"},
        )

    def test_shared_source_preserves_secondary_pareto_candidate_after_strict_top_k(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        lanes = [
            {
                "name": "confirmed",
                "k": 2,
                "include_lanes": ["performance"],
                "axes": [("primary_score", "maximize")],
                "parent_eligible": True,
            },
            {
                "name": "incubator",
                "k": 1,
                "include_lanes": ["performance"],
                "axes": [
                    ("primary_score", "maximize"),
                    ("secondary_score", "maximize"),
                ],
                "parent_eligible": True,
            },
        ]
        points = [
            ("primary_best", 10.0, 1.0),
            ("primary_second", 9.0, 2.0),
            ("secondary_best", 8.0, 100.0),
            ("dominated", 7.0, 0.0),
        ]
        findings = [
            {
                "id": variant,
                "finding_type": "result",
                "variant_name": variant,
                "metrics": {
                    "score": primary,
                    "primary_score": primary,
                    "secondary_score": secondary,
                    "frontier_lane": "performance",
                    "promotion_lane": "performance",
                    "tier": "T3",
                    "evidence_stage": "T3",
                    "scored_complete": True,
                    "promotion_eligible": True,
                    "protocol_integrity": "passed",
                },
            }
            for variant, primary, secondary in points
        ]

        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                primary_metric="score",
                frontier_lanes=lanes,
            )
            store.promote(0, findings)
            lane_frontiers = store.get_manifest()["lane_frontiers"]

        self.assertEqual(
            [entry["finding_id"] for entry in lane_frontiers["confirmed"]],
            ["primary_best", "primary_second"],
        )
        self.assertEqual(
            [entry["finding_id"] for entry in lane_frontiers["incubator"]],
            ["secondary_best"],
        )

    def test_supported_result_summary_names_share_frontier_identity_parser(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        for name in (
            "tiered_eval_summary.json",
            "result_summary.json",
            "evaluation_summary.json",
            "eval_summary.json",
            "final_summary.json",
            "summary.json",
            "custom_candidate_eval_summary.json",
        ):
            path = f"run/results/candidate/protocol/{name}"
            self.assertTrue(frontier._is_result_summary_path(path), name)
            self.assertEqual(
                frontier._result_artifact_variant_token(path),
                "candidate__protocol",
            )

    def test_inferred_false_completion_does_not_override_task_maturity(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        inferred = {
            "variant_name": "legacy_result",
            "metrics": {
                "score": 1.0,
                "effort_ratio": 1.0,
                "coverage_ratio": 1.0,
                "scored_complete": False,
                "_inferred_scored_complete": True,
            },
        }
        explicit = {
            **inferred,
            "metrics": {
                **inferred["metrics"],
                "_inferred_scored_complete": False,
            },
        }
        mixed = {
            **inferred,
            "metrics": {
                **inferred["metrics"],
                "complete_eval": False,
            },
        }

        self.assertFalse(frontier._candidate_has_hard_incomplete_marker(inferred))
        self.assertTrue(frontier._has_mature_durable_evidence(inferred))
        self.assertTrue(frontier._candidate_has_hard_incomplete_marker(explicit))
        self.assertTrue(frontier._candidate_has_hard_incomplete_marker(mixed))

    def test_exploit_parent_summary_excludes_non_parentable_lanes(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            frontier,
            prompt_context,
            prompt_strategy,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
            _digest_frontier_lane_metadata,
            _digest_lane_frontiers,
        )

        lanes = [
            {
                "name": "diagnostic",
                "k": 2,
                "include_lanes": ["diagnostic"],
                "axes": [{"metric": "score", "direction": "maximize"}],
                "parent_eligible": False,
            },
            {
                "name": "incubator",
                "k": 2,
                "include_lanes": ["incubator"],
                "axes": [{"metric": "score", "direction": "maximize"}],
                "parent_eligible": True,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = frontier.FrontierStore(
                Path(tmp) / "frontier",
                primary_metric="score",
                frontier_lanes=lanes,
            )
            store.promote(
                0,
                [
                    {
                        "id": "diagnostic-best",
                        "finding_type": "result",
                        "variant_name": "diagnostic_best",
                        "metrics": {
                            "score": 9.0,
                            "frontier_lane": "diagnostic",
                            "scored_complete": True,
                        },
                    },
                    {
                        "id": "parent-best",
                        "finding_type": "result",
                        "variant_name": "parent_best",
                        "metrics": {
                            "score": 1.0,
                            "frontier_lane": "incubator",
                            "scored_complete": True,
                        },
                    },
                ],
            )
            all_entries = store.get_summary_up_to_generation(0)
            parent_entries = store.get_parent_summary_up_to_generation(0)
            context_parent_entries = prompt_context._parent_frontier_summary_up_to_generation(
                store,
                0,
            )
            lane_digest = _digest_lane_frontiers(Path(tmp))
            lane_metadata = {
                lane["name"]: lane for lane in _digest_frontier_lane_metadata(Path(tmp))
            }
            hint = prompt_strategy._generate_variant_hint(
                1,
                0,
                1,
                "exploit",
                store,
                frontier_summary=parent_entries,
            )

        self.assertEqual(
            {entry["variant_name"] for entry in all_entries},
            {"diagnostic_best", "parent_best"},
        )
        self.assertEqual(
            [entry["variant_name"] for entry in parent_entries],
            ["parent_best"],
        )
        self.assertEqual(context_parent_entries, parent_entries)
        self.assertFalse(lane_digest["diagnostic"][0]["parent_eligible"])
        self.assertTrue(lane_digest["incubator"][0]["parent_eligible"])
        self.assertFalse(lane_metadata["diagnostic"]["parent_eligible"])
        self.assertTrue(lane_metadata["incubator"]["parent_eligible"])
        self.assertIn("parent_best", hint)
        self.assertNotIn("diagnostic_best", hint)

    def test_resume_replaces_stale_lane_parent_policy_in_all_views(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            frontier,
            prompt_context,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
            _digest_lane_frontiers,
        )

        parentable_lane = {
            "name": "candidate_library",
            "k": 2,
            "include_lanes": ["candidate"],
            "axes": [{"metric": "score", "direction": "maximize"}],
            "parent_eligible": True,
        }
        diagnostic_lane = {**parentable_lane, "parent_eligible": False}
        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "frontier"
            store = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                frontier_lanes=[parentable_lane],
            )
            store.promote(
                0,
                [
                    {
                        "id": "candidate-a",
                        "finding_type": "result",
                        "variant_name": "candidate_a",
                        "metrics": {
                            "score": 1.0,
                            "frontier_lane": "candidate",
                            "scored_complete": True,
                        },
                    }
                ],
            )
            self.assertEqual(len(store.get_parent_summary_up_to_generation(0)), 1)

            resumed = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                frontier_lanes=[diagnostic_lane],
            )
            manifest = resumed.get_manifest()
            lane_digest = _digest_lane_frontiers(Path(tmp), current_gen_id=0)
            task_spec = SimpleNamespace(
                evaluation=SimpleNamespace(maturity_policy={}, frontier_lanes=[diagnostic_lane]),
                gems=None,
            )
            views = prompt_context._strong_parent_views_for_prompt(
                frontier=resumed,
                validation_candidates=[],
                task_spec=task_spec,
                completed_gen_id=0,
            )

        self.assertFalse(manifest["frontier_lanes"][0]["parent_eligible"])
        self.assertEqual(resumed.get_parent_summary_up_to_generation(0), [])
        self.assertFalse(lane_digest["candidate_library"][0]["parent_eligible"])
        self.assertEqual(views["incubator_top_k"], [])
        self.assertEqual(
            [entry["variant_name"] for entry in views["diagnostic_control_top_k"]],
            ["candidate_a"],
        )

    def test_resume_rebuilds_active_lanes_for_filter_cap_and_removed_lane_changes(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        base_lane = {
            "name": "candidate_library",
            "k": 2,
            "cumulative_cap": 2,
            "include_lanes": ["candidate"],
            "axes": [{"metric": "score", "direction": "maximize"}],
            "parent_eligible": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "frontier"
            store = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                frontier_lanes=[base_lane],
            )
            store.promote(
                0,
                [
                    {
                        "id": f"candidate-{index}",
                        "finding_type": "result",
                        "variant_name": f"candidate_{index}",
                        "metrics": {
                            "score": score,
                            "frontier_lane": "candidate",
                            "scored_complete": True,
                        },
                    }
                    for index, score in enumerate((2.0, 1.0))
                ],
            )

            capped = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                frontier_lanes=[{**base_lane, "cumulative_cap": 1}],
            )
            self.assertEqual(
                [entry["variant_name"] for entry in capped.get_parent_summary_up_to_generation(0)],
                ["candidate_0"],
            )

            filtered = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                frontier_lanes=[{**base_lane, "include_lanes": ["confirmed"]}],
            )
            self.assertEqual(filtered.get_parent_summary_up_to_generation(0), [])
            self.assertEqual(filtered.get_manifest()["lane_frontiers"]["candidate_library"], [])
            self.assertEqual(len(filtered.get_manifest()["generations"]["0"]), 2)

            removed = frontier.FrontierStore(
                frontier_dir,
                primary_metric="score",
                frontier_lanes=[
                    {
                        **base_lane,
                        "name": "other_library",
                        "include_lanes": ["other"],
                    }
                ],
            )

        self.assertEqual(removed.get_parent_summary_up_to_generation(0), [])
        self.assertEqual(removed.get_manifest()["lane_frontiers"]["other_library"], [])
        self.assertEqual(len(removed.get_manifest()["generations"]["0"]), 2)

    def test_lane_view_builder_preserves_filter_dedup_pareto_and_cross_lane_contract(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import frontier

        lanes = [
            {
                "name": "confirmed",
                "include_lanes": ["performance"],
                "axes": [["score", "maximize"], ["cost", "minimize"]],
                "cumulative_cap": 1,
                "admit_new_high": True,
            },
            {
                "name": "incubator",
                "include_lanes": ["performance"],
                "axes": [["score", "maximize"], ["cost", "minimize"]],
                "cumulative_cap": 2,
                "admit_new_high": True,
            },
        ]

        def entry(
            name: str,
            generation: int,
            target: str,
            score: float,
            cost: float,
            source: str = "performance",
        ) -> dict[str, object]:
            return {
                "generation_id": generation,
                "finding_id": f"{name}-{generation}-{target}",
                "variant_name": name,
                "frontier_lane": target,
                "promoted_for_lane": target,
                "source_frontier_lane": source,
                "lane_metric_name": "score",
                "lane_metric_value": score,
                "metrics": {"score": score, "cost": cost},
            }

        shared_confirmed = entry("shared", 0, "confirmed", 10.0, 10.0)
        shared_incubator = entry("shared", 0, "incubator", 10.0, 10.0)
        a_old = entry("a", 0, "incubator", 8.0, 1.0)
        a_better = entry("a", 1, "incubator", 8.5, 1.0)
        b = entry("b", 0, "incubator", 9.0, 5.0)
        dominated = entry("dominated", 0, "incubator", 7.0, 6.0)
        wrong_source = entry("wrong_source", 0, "incubator", 100.0, 0.0, "diagnostic")

        lane_views, cumulative = frontier._build_cumulative_lane_views(
            {
                "0": [
                    shared_confirmed,
                    shared_incubator,
                    a_old,
                    b,
                    dominated,
                    wrong_source,
                ],
                "1": [a_better],
            },
            lanes,
            maturity_policy=None,
            primary_metric="score",
            metric_direction="maximize",
            promote_top_k=2,
            entry_is_committed=lambda _entry: True,
        )

        self.assertEqual(
            [item["variant_name"] for item in lane_views["confirmed"]],
            ["shared"],
        )
        self.assertEqual(
            [item["variant_name"] for item in lane_views["incubator"]],
            ["b", "a"],
        )
        self.assertEqual(
            [item["variant_name"] for item in cumulative],
            ["shared", "b", "a", "wrong_source"],
        )


if __name__ == "__main__":
    unittest.main()
