from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class FindingsCollectionContractsTest(unittest.TestCase):
    def test_current_aggregate_completion_precedes_stale_outer_flag(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        for aggregate_flag, outer_flag in ((True, False), (False, True)):
            with self.subTest(aggregate_flag=aggregate_flag, outer_flag=outer_flag):
                normalized = fc.normalized_result_summary(
                    {
                        "scored_complete": outer_flag,
                        "current_aggregate": {
                            "score": 1.0,
                            "scored_complete": aggregate_flag,
                        },
                    }
                )
                metrics = fc._result_summary_metrics(normalized)
                self.assertIs(metrics["scored_complete"], aggregate_flag)

    def test_summary_helper_contracts_cover_status_and_metric_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        self.assertEqual(fc._slug(" Alpha / Beta! "), "alpha_beta")
        self.assertIsNone(fc._load_json(Path("/definitely/missing.json")))
        self.assertIsNone(fc._as_float(True))
        self.assertIsNone(fc._as_float("nan"))
        self.assertEqual(fc._as_float("3.5"), 3.5)
        self.assertEqual(fc._mean([1.0, 3.0]), 2.0)
        self.assertIsNone(fc._mean([]))
        self.assertEqual(fc._q25([10.0, 2.0, 6.0, 4.0]), 2.0)
        self.assertIsNone(fc._q25([]))
        self.assertEqual(fc._cell_float({"a": None, "b": "2"}, "a", "b"), 2.0)

        self.assertEqual(
            fc._infer_strategy_family("benchclone_alpha", {}),
            "task_candidate",
        )
        self.assertEqual(
            fc._infer_strategy_family(
                "random_control",
                {"current_aggregate": {"strategy_family": "diagnostic_control"}},
            ),
            "diagnostic_control",
        )
        self.assertEqual(
            fc._infer_strategy_family(
                "learned_alpha", {"current_aggregate": {"strategy_family": "custom"}}
            ),
            "custom",
        )
        text = fc._status_text("Scored Complete", None, "T1")
        self.assertTrue(fc._status_has_any(text, "scored_complete"))
        self.assertFalse(fc._is_capped_status("uncapped_full"))
        self.assertFalse(fc._is_capped_status("alpha_not_capped"))
        marker = fc._status_marker_text(
            {"variant_name": "v", "final_status": "summary only"},
            Path("results/v/tiered_eval_summary.json"),
        )
        self.assertTrue(fc._is_summary_only_status(marker))
        self.assertTrue(fc._is_scout_or_smoke_status("cheap_probe"))
        self.assertTrue(fc._is_capped_status("capped_at_epoch"))
        self.assertTrue(fc._is_not_scored_status("complete_eval_false"))
        self.assertFalse(fc._is_bad_result_status("not_failed"))
        self.assertFalse(fc._is_bad_result_status("no_timeout"))
        marked = fc._apply_path_status_markers(
            {"variant_name": "smoke_probe"},
            Path("results/smoke_probe/smoke_tiered_eval_summary.json"),
        )
        self.assertTrue(marked["scout_only"])
        self.assertTrue(marked["is_smoke_eval"])

        aggregate = {"x": "yes", "y": "off", "score": 1.0}
        self.assertTrue(fc._summary_flag({}, aggregate, "x"))
        self.assertTrue(fc._summary_explicit_false({}, aggregate, "y"))
        self.assertTrue(fc._has_scored_metrics(aggregate))
        self.assertFalse(fc._has_scored_metrics(aggregate, ("task_metric",)))
        self.assertTrue(fc._is_bad_result_status("timeout failure"))
        metrics: dict[str, object] = {"bottleneck_target": ""}
        fc._copy_research_metadata(
            metrics,
            {"bottleneck_target": "drawdown", "parent_usage": "repair"},
            {"ignored": object()},
        )
        self.assertEqual(metrics["bottleneck_target"], "drawdown")
        self.assertEqual(metrics["parent_usage"], "repair")
        self.assertEqual(fc._scalar_dict({"a": object(), "b": 1, "c": None}), {"b": 1, "c": None})

    def test_normalized_result_summary_and_metric_contracts(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "alpha_child"
            result_dir.mkdir(parents=True)
            result_payload = result_dir / "result.json"
            result_payload.write_text(
                json.dumps(
                    {
                        "mean_test_taskscore": 4.0,
                        "score": 4.0,
                        "mean_active_alpha_vs_benchmark_pct": 2.0,
                        "strategy_family": "learned_alpha",
                    }
                ),
                encoding="utf-8",
            )
            summary_path = result_dir / "tiered_eval_summary.json"
            summary = {
                "variant_name": "alpha_reported",
                "final_status": "ok",
                "tiers": [
                    {
                        "tier": "T1",
                        "status": "scored_complete",
                        "gate": {"passed": True},
                        "result_path": "result.json",
                        "metrics_summary": {
                            "scored_cell_count": 29,
                            "q25_active_alpha_vs_benchmark_pct": 1.0,
                        },
                    }
                ],
                "all_eval_cells": [
                    {"active_alpha_vs_benchmark_pct": 1.0, "return_pct": 2.0},
                    {"active_alpha_vs_benchmark_pct": 3.0, "return_pct": 4.0},
                ],
                "metrics": {"parent_candidate": "parent_a"},
            }
            normalized = fc.normalized_result_summary(summary, summary_path=summary_path)
            self.assertEqual(normalized["current_aggregate"]["promotion_eligible"], True)
            self.assertEqual(normalized["current_aggregate"]["mean_test_taskscore"], 4.0)
            self.assertEqual(normalized["tier_reached"], "T1")
            self.assertEqual(normalized["evaluation_units"], 29)
            self.assertNotIn("n_eval_cells", normalized)
            self.assertTrue(normalized["scored_complete"])
            self.assertEqual(
                fc.result_summary_variant_name(summary_path, normalized, run_dir),
                "alpha_child",
            )
            custom_path = run_dir / "results" / "custom_alpha_tiered_eval_summary.json"
            self.assertEqual(
                fc.result_summary_variant_name(custom_path, {}, run_dir),
                "alpha",
            )

            metrics = fc._result_summary_metrics(normalized)
            self.assertEqual(metrics["mean_test_taskscore"], 4.0)
            self.assertEqual(metrics["parent_candidate"], "parent_a")
            self.assertEqual(metrics["evaluation_units"], 29)
            self.assertNotIn("scored_cell_count", metrics)
            self.assertEqual(metrics["q25_active_alpha_vs_benchmark_pct"], 1.0)
            self.assertEqual(metrics["mean_active_alpha_vs_benchmark_pct"], 2.0)
            self.assertEqual(metrics["strategy_family"], "learned_alpha")

            summary_only = fc.normalized_result_summary(
                {
                    "tiers": [
                        {
                            "tier": "T1",
                            "status": "summary_only",
                            "metrics_summary": {"scored_cell_count": 29},
                        }
                    ]
                },
                summary_path=result_dir / "summary_only_tiered_eval_summary.json",
            )
            self.assertTrue(summary_only["summary_only"])
            self.assertFalse(summary_only["scored_complete"])
            self.assertEqual(summary_only["result_status"], "summary_only")

    def test_task_configured_cell_derivation_can_score_tier_summary(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        summary = {
            "variant_name": "derived_candidate",
            "tiers": [{"tier": "T1", "status": "ok", "metrics_summary": {"scored_cell_count": 2}}],
            "all_eval_cells": [
                {"task_return": 1.0},
                {"task_return": 3.0},
            ],
        }
        normalized = fc.normalized_result_summary(
            summary,
            summary_path=Path("results/derived_candidate/tiered_eval_summary.json"),
        )
        self.assertNotIn("scored_complete", normalized)
        self.assertEqual(normalized["result_status"], "unscored_artifact")

        normalized["complete_eval"] = True
        metrics = fc._result_summary_metrics(
            normalized,
            cell_metric_derivations=[
                {
                    "name": "mean_task_return",
                    "source_keys": ["task_return"],
                    "aggregate": "mean",
                }
            ],
            scoring_metric_keys=["mean_task_return"],
        )
        self.assertEqual(metrics["mean_task_return"], 2.0)
        self.assertTrue(metrics["scored_complete"])
        self.assertEqual(metrics["result_status"], "scored_complete")

    def test_generic_passed_status_does_not_claim_mature_completion(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        summary = {
            "variant_name": "candidate",
            "tiers": [
                {
                    "tier": "T1",
                    "status": "passed",
                    "metrics_summary": {"score": 1.0},
                }
            ],
        }
        normalized = fc.normalized_result_summary(
            summary,
            summary_path=Path("results/candidate/tiered_eval_summary.json"),
        )
        metrics = fc._result_summary_metrics(normalized, scoring_metric_keys=("score",))
        self.assertEqual(normalized["result_status"], "unknown_maturity")
        self.assertNotIn("scored_complete", normalized)
        self.assertEqual(metrics["result_status"], "unknown_maturity")
        self.assertNotIn("scored_complete", metrics)

        explicit = {
            **summary,
            "tiers": [
                {
                    "tier": "T1",
                    "status": "scored_complete",
                    "metrics_summary": {"score": 1.0},
                }
            ],
        }
        explicit_normalized = fc.normalized_result_summary(
            explicit,
            summary_path=Path("results/candidate/tiered_eval_summary.json"),
        )
        explicit_metrics = fc._result_summary_metrics(
            explicit_normalized,
            scoring_metric_keys=("score",),
        )
        self.assertTrue(explicit_normalized["scored_complete"])
        self.assertTrue(explicit_metrics["scored_complete"])
        self.assertEqual(explicit_metrics["result_status"], "scored_complete")

    def test_top_level_completion_fact_overrides_stale_nested_copy(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        normalized = fc.normalized_result_summary(
            {
                "complete_eval": True,
                "metrics": {"score": 1.0, "scored_complete": False},
            }
        )
        self.assertTrue(normalized["current_aggregate"]["scored_complete"])
        self.assertNotIn("complete_eval", normalized["current_aggregate"])
        metrics = fc._result_summary_metrics(normalized, scoring_metric_keys=("score",))
        self.assertTrue(metrics["scored_complete"])
        self.assertEqual(metrics["result_status"], "scored_complete")

        incomplete = fc.normalized_result_summary(
            {
                "complete_eval": False,
                "metrics": {"score": 1.0, "scored_complete": True},
            }
        )
        incomplete_metrics = fc._result_summary_metrics(
            incomplete,
            scoring_metric_keys=("score",),
        )
        self.assertFalse(incomplete_metrics["scored_complete"])
        self.assertEqual(incomplete_metrics["result_status"], "not_scored_complete")

    def test_nested_inferred_false_defers_to_ratio_maturity_for_frontier_and_gems(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )
        from praxist.plugins.workflow_stages.research_loop.backend import (
            frontier,
            gems,
        )

        maturity_policy = {
            "require_ratio_gate": True,
            "min_effort_ratio": 0.8,
            "min_coverage_ratio": 0.8,
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "legacy_candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "legacy_candidate",
                        "generation_id": 0,
                        "metrics": {
                            "score": 1.0,
                            "frontier_lane": "candidate",
                            "effort_ratio": 1.0,
                            "coverage_ratio": 1.0,
                            "scored_complete": False,
                            "_inferred_scored_complete": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=0,
                scoring_metric_keys=("score",),
            )
            store = frontier.FrontierStore(
                run_dir / "frontier",
                primary_metric="score",
                maturity_policy=maturity_policy,
                frontier_lanes=[
                    {
                        "name": "candidate_library",
                        "include_lanes": ["candidate"],
                        "axes": [{"metric": "score", "direction": "maximize"}],
                        "parent_eligible": True,
                    }
                ],
            )
            promoted = store.promote(0, [finding])

        self.assertNotIn("scored_complete", finding["metrics"])
        self.assertEqual(finding["metrics"]["result_status"], "unknown_maturity")
        self.assertFalse(gems._entry_has_hard_gem_rejection_marker(finding))
        self.assertTrue(gems._entry_has_generic_mature_evidence(finding, maturity_policy))
        self.assertEqual([entry["variant_name"] for entry in promoted], ["legacy_candidate"])

    def test_realized_dimensions_round_trip_from_result_to_frontier(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )
        from praxist.plugins.workflow_stages.research_loop.backend import frontier
        from praxist.plugins.workflow_stages.research_loop.backend.tools import findings_ingest

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "gen1_peer2" / "candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "candidate",
                        "generation_id": 1,
                        "scored_complete": True,
                        "design_dimensions": {
                            "architecture_family": "residual",
                            "regularization": "adaptive",
                        },
                        "current_aggregate": {"score": 1.25},
                        "metrics": {"frontier_lane": "performance"},
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=1,
                scoring_metric_keys=("score",),
            )
            finding_path = next((run_dir / "shared_findings").glob("*.json"))
            stale = json.loads(finding_path.read_text(encoding="utf-8"))
            stale["metrics"].pop("design_dimensions")
            finding_path.write_text(json.dumps(stale), encoding="utf-8")
            [finding] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=1,
                scoring_metric_keys=("score",),
            )
            finding["metrics"]["realized_dimensions"] = {"architecture_family": "conflicting_alias"}
            ingested = findings_ingest.parse_finding_file(
                finding_path,
                primary_metric="score",
            )
            assert ingested is not None
            ingested["metrics"]["realized_dimensions"] = {
                "architecture_family": "conflicting_alias"
            }
            store = frontier.FrontierStore(
                run_dir / "frontier",
                primary_metric="score",
                maturity_policy={"require_ratio_gate": False},
                frontier_lanes=[
                    {
                        "name": "performance",
                        "include_lanes": ["performance"],
                        "axes": [{"metric": "score", "direction": "maximize"}],
                        "parent_eligible": True,
                    }
                ],
            )
            promoted = store.promote(1, [ingested])

        expected = {
            "architecture_family": "residual",
            "regularization": "adaptive",
        }
        self.assertEqual(finding["metrics"]["design_dimensions"], expected)
        self.assertEqual(ingested["design_dimensions"], expected)
        self.assertNotIn("design_dimensions", ingested["metrics"])
        self.assertEqual(frontier._extract_design_dimensions(finding), expected)
        self.assertEqual(frontier._extract_design_dimensions(promoted[0]), expected)
        self.assertNotIn("design_dimensions", promoted[0]["metrics"])
        self.assertNotIn("realized_dimensions", promoted[0]["metrics"])

    def test_tiered_result_dimensions_survive_normalization_with_one_canonical_name(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "score": 2.0,
                        "design_dimensions": {
                            "mechanism_family": "canonical",
                            "depth": 3,
                        },
                    }
                ),
                encoding="utf-8",
            )
            normalized = fc.normalized_result_summary(
                {
                    "variant_name": "tiered",
                    "tiers": [
                        {
                            "tier": "full",
                            "status": "complete",
                            "result_path": "result.json",
                            "metrics_summary": {
                                "score": 2.0,
                                "realized_dimensions": {"mechanism_family": "stale_alias"},
                            },
                        }
                    ],
                },
                summary_path=root / "summary.json",
            )
            metrics = fc._result_summary_metrics(
                normalized,
                scoring_metric_keys=("score",),
            )

        self.assertEqual(
            metrics["design_dimensions"],
            {"mechanism_family": "canonical", "depth": 3},
        )
        self.assertNotIn("realized_dimensions", metrics)

    def test_result_artifact_options_use_task_scoring_metrics_not_cost_keys(self) -> None:
        from types import SimpleNamespace

        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        spec = SimpleNamespace(
            evaluation=SimpleNamespace(
                primary_metric="objective",
                aux_metrics=["auxiliary"],
                anchor_metrics=[{"name": "anchor"}],
                maturity_policy={
                    "complete_stage_labels": ["approved_reduced"],
                    "preliminary_stage_labels": ["diagnostic"],
                },
                frontier_lanes=[
                    {
                        "require_metrics": ["required_score"],
                        "require_truthy_metrics": ["protocol_clean"],
                        "require_falsey_metrics": ["has_violation"],
                        "axes": ["lane_axis"],
                        "min_metrics": {"min_required": 1.0},
                    }
                ],
            ),
            gems=SimpleNamespace(
                primary_metric_keys=["configured_primary"],
                secondary_metric_keys=["configured_secondary"],
                lower_tail_metric_keys=["configured_tail"],
                validation_metric_keys=["configured_validation"],
                cost_metric_keys=["risk_only"],
            ),
        )
        options = fc.result_artifact_options_from_task_spec(spec)
        self.assertEqual(options["result_scoring_metric_keys"][0], "objective")
        scoring = set(options["result_scoring_metric_keys"])
        self.assertTrue(
            {
                "objective",
                "auxiliary",
                "anchor",
                "required_score",
                "lane_axis",
                "min_required",
                "configured_primary",
                "configured_secondary",
                "configured_tail",
                "configured_validation",
            }.issubset(scoring)
        )
        self.assertNotIn("risk_only", scoring)
        self.assertNotIn("protocol_clean", scoring)
        self.assertNotIn("has_violation", scoring)
        self.assertEqual(
            options["result_maturity_policy"]["complete_stage_labels"],
            ["approved_reduced"],
        )

    def test_task_authorized_reduced_result_survives_materialization_and_durable_routing(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )
        from praxist.plugins.workflow_stages.research_loop.backend import (
            gems,
            generation_boundary,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
            FrontierStore,
        )

        maturity_policy = {
            "complete_stage_labels": ["approved_reduced"],
            "preliminary_stage_labels": ["diagnostic"],
            "require_ratio_gate": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "approved_reduced"
            result_dir.mkdir(parents=True)
            (result_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "approved_reduced",
                        "generation_id": 0,
                        "evidence_stage": "approved_reduced",
                        "scored_complete": True,
                        "scout_only": True,
                        "partial_eval": True,
                        "promotion_eligible": True,
                        "current_aggregate": {
                            "score": 1.25,
                            "evaluation_units": 1,
                        },
                        "metrics": {"frontier_lane": "performance"},
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=0,
                scoring_metric_keys=("score",),
                result_maturity_policy=maturity_policy,
            )
            store = FrontierStore(
                run_dir / "frontier",
                primary_metric="score",
                maturity_policy=maturity_policy,
                frontier_lanes=[
                    {
                        "name": "performance",
                        "include_lanes": ["performance"],
                        "axes": [{"metric": "score", "direction": "maximize"}],
                        "parent_eligible": True,
                    }
                ],
            )
            promoted = store.promote(0, [finding])

        metrics = finding["metrics"]
        self.assertIs(metrics["scored_complete"], True)
        self.assertEqual(metrics["result_status"], "scored_complete")
        self.assertIs(metrics["scout_only"], True)
        self.assertIs(metrics["partial_eval"], True)
        self.assertNotIn("excluded_from_durable_frontier", metrics)
        self.assertEqual([entry["finding_id"] for entry in promoted], [finding["id"]])
        self.assertTrue(
            gems._entry_is_clean_gem_admission_candidate(
                finding,
                maturity_policy=maturity_policy,
            )
        )
        self.assertTrue(
            gems._entry_is_mature_gem_admission_candidate(
                finding,
                min_mature_eval_units=1,
                maturity_policy=maturity_policy,
            )
        )
        self.assertTrue(generation_boundary._is_mature_result_payload(metrics, maturity_policy))

    def test_task_authorized_tiered_scout_result_is_not_rewritten_as_failure(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        maturity_policy = {
            "complete_stage_labels": ["scout"],
            "preliminary_stage_labels": ["diagnostic"],
            "require_ratio_gate": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "authorized_scout"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "authorized_scout",
                        "generation_id": 0,
                        "evidence_stage": "scout",
                        "tiers": [
                            {
                                "tier": "scout",
                                "status": "scout",
                                "evidence_stage": "scout",
                                "metrics_summary": {
                                    "score": 1.25,
                                    "evaluation_units": 1,
                                },
                                "gate": {"passed": True},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=0,
                scoring_metric_keys=("score",),
                result_maturity_policy=maturity_policy,
            )

        metrics = finding["metrics"]
        self.assertTrue(metrics["scored_complete"])
        self.assertEqual(metrics["result_status"], "scored_complete")
        self.assertTrue(metrics["scout_only"])
        self.assertTrue(metrics["promotion_eligible"])
        self.assertNotIn("excluded_from_durable_frontier", metrics)

    def test_ratio_authorized_partial_result_routes_consistently(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )
        from praxist.plugins.workflow_stages.research_loop.backend import (
            gems,
            generation_boundary,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
            FrontierStore,
        )

        maturity_policy = {
            "min_effort_ratio": 0.75,
            "min_coverage_ratio": 0.80,
            "require_ratio_gate": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "ratio_reduced"
            result_dir.mkdir(parents=True)
            (result_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "ratio_reduced",
                        "generation_id": 0,
                        "evidence_stage": "reduced",
                        "scout_only": True,
                        "partial_eval": True,
                        "capped": True,
                        "promotion_eligible": True,
                        "effort_ratio": 0.90,
                        "coverage_ratio": 0.90,
                        "current_aggregate": {"score": 1.25},
                        "metrics": {"frontier_lane": "performance"},
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=0,
                scoring_metric_keys=("score",),
                result_maturity_policy=maturity_policy,
            )
            store = FrontierStore(
                run_dir / "frontier",
                primary_metric="score",
                maturity_policy=maturity_policy,
                frontier_lanes=[
                    {
                        "name": "performance",
                        "include_lanes": ["performance"],
                        "axes": [{"metric": "score", "direction": "maximize"}],
                        "parent_eligible": True,
                    }
                ],
            )
            promoted = store.promote(0, [finding])

        metrics = finding["metrics"]
        self.assertNotIn("scored_complete", metrics)
        self.assertEqual(metrics["result_status"], "unknown_maturity")
        self.assertTrue(metrics["partial_eval"])
        self.assertTrue(metrics["capped"])
        self.assertEqual([entry["finding_id"] for entry in promoted], [finding["id"]])
        self.assertTrue(
            gems._entry_is_mature_gem_admission_candidate(
                finding,
                min_mature_eval_units=100,
                maturity_policy=maturity_policy,
            )
        )
        self.assertTrue(generation_boundary._is_mature_result_payload(metrics, maturity_policy))

    def test_task_authorized_stage_does_not_override_real_protocol_failure(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )
        from praxist.plugins.workflow_stages.research_loop.backend import (
            gems,
            generation_boundary,
        )

        maturity_policy = {
            "complete_stage_labels": ["approved_reduced"],
            "require_ratio_gate": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "protocol_failed"
            result_dir.mkdir(parents=True)
            (result_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "protocol_failed",
                        "generation_id": 0,
                        "evidence_stage": "approved_reduced",
                        "scored_complete": True,
                        "protocol_integrity_passed": False,
                        "current_aggregate": {"score": 9.0},
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=0,
                scoring_metric_keys=("score",),
                result_maturity_policy=maturity_policy,
            )

        self.assertEqual(finding["metrics"]["result_status"], "protocol_invalid")
        self.assertFalse(
            gems._entry_is_clean_gem_admission_candidate(
                finding,
                maturity_policy=maturity_policy,
            )
        )
        self.assertFalse(
            generation_boundary._is_mature_result_payload(finding["metrics"], maturity_policy)
        )

    def test_result_summary_projects_canonical_maturity_from_supported_sources(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        cases = {
            "top_level_ratios": {
                "effort_ratio": 0.9,
                "coverage_ratio": 0.85,
            },
            "metrics_aliases": {
                "metrics": {
                    "training_effort_ratio": 0.8,
                    "evaluation_coverage_ratio": 0.95,
                }
            },
            "extra_counts": {
                "extra": {
                    "completed_steps": 9,
                    "planned_steps": 10,
                    "completed_eval_units": 8,
                    "total_eval_units": 10,
                }
            },
            "ratio_maps": {
                "metrics": {
                    "effort_ratio_by_dimension": {"training": 0.9, "search": 0.8},
                    "coverage_ratios": {"required_cases": 0.85, "replicates": 0.95},
                }
            },
        }
        expected = {
            "top_level_ratios": (0.9, 0.85),
            "metrics_aliases": (0.8, 0.95),
            "extra_counts": (0.9, 0.8),
            "ratio_maps": (0.8, 0.85),
        }

        for name, source in cases.items():
            with self.subTest(name=name):
                summary = {"current_aggregate": {"score": 1.0}, **source}
                metrics = fc._result_summary_metrics(
                    summary,
                    scoring_metric_keys=("score",),
                )
                self.assertEqual(
                    (metrics["effort_ratio"], metrics["coverage_ratio"]),
                    expected[name],
                )

    def test_result_summary_does_not_invent_missing_maturity_or_policy_decisions(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        metrics = fc._result_summary_metrics(
            {
                "effort_ratio": 0.5,
                "current_aggregate": {"score": 1.0},
            },
            scoring_metric_keys=("score",),
        )

        self.assertEqual(metrics["effort_ratio"], 0.5)
        self.assertNotIn("coverage_ratio", metrics)
        self.assertNotIn("mature_enough", metrics)
        self.assertNotIn("maturity_basis", metrics)
        self.assertNotIn("min_effort_ratio", metrics)
        self.assertNotIn("min_coverage_ratio", metrics)

    def test_result_materialization_preserves_maturity_and_primary_metric_end_to_end(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
            evidence_maturity_snapshot,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
            FrontierStore,
        )

        maturity_policy = {
            "min_effort_ratio": 0.75,
            "min_coverage_ratio": 0.80,
            "require_ratio_gate": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "candidate" / "complete_protocol"
            result_dir.mkdir(parents=True)
            (result_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "candidate",
                        "generation_id": 0,
                        "scored_complete": True,
                        "effort_ratio": 1.0,
                        "coverage_ratio": 1.0,
                        "current_aggregate": {
                            "primary_objective": 0.25,
                            "auxiliary_objective": 9.0,
                        },
                        "metrics": {
                            "effort_ratio": 1.0,
                            "coverage_ratio": 1.0,
                            "frontier_lane": "confirmed",
                        },
                    }
                ),
                encoding="utf-8",
            )
            [finding] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=0,
                scoring_metric_keys=("primary_objective", "auxiliary_objective"),
            )

            finding_path = next((run_dir / "shared_findings").glob("*.json"))
            stale = json.loads(finding_path.read_text(encoding="utf-8"))
            stale["metrics"].pop("effort_ratio")
            stale["metrics"].pop("coverage_ratio")
            finding_path.write_text(json.dumps(stale), encoding="utf-8")
            [refreshed] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=0,
                scoring_metric_keys=("primary_objective", "auxiliary_objective"),
            )

            store = FrontierStore(
                run_dir / "frontier",
                primary_metric="primary_objective",
                metric_direction="minimize",
                maturity_policy=maturity_policy,
                frontier_lanes=[
                    {
                        "name": "confirmed",
                        "include_lanes": ["confirmed"],
                        "axes": [{"metric": "primary_objective", "direction": "minimize"}],
                    }
                ],
            )
            promoted = store.promote(0, [refreshed])

        self.assertEqual(finding["metrics"]["effort_ratio"], 1.0)
        self.assertEqual(finding["metrics"]["coverage_ratio"], 1.0)
        self.assertIn("primary_score=0.25", finding["content"])
        self.assertEqual(refreshed["metrics"]["effort_ratio"], 1.0)
        self.assertEqual(refreshed["metrics"]["coverage_ratio"], 1.0)
        self.assertTrue(evidence_maturity_snapshot(refreshed, maturity_policy)["mature_enough"])
        self.assertEqual([entry["finding_id"] for entry in promoted], [refreshed["id"]])

    def test_result_materialization_removes_stale_cached_maturity(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "candidate",
                        "generation_id": 0,
                        "scored_complete": True,
                        "current_aggregate": {"score": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=0,
                scoring_metric_keys=("score",),
            )
            finding_path = next((run_dir / "shared_findings").glob("*.json"))
            stale = json.loads(finding_path.read_text(encoding="utf-8"))
            stale["metrics"]["effort_ratio"] = 1.0
            stale["metrics"]["coverage_ratio"] = 1.0
            finding_path.write_text(json.dumps(stale), encoding="utf-8")

            [refreshed] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=0,
                scoring_metric_keys=("score",),
            )

        self.assertNotIn("effort_ratio", refreshed["metrics"])
        self.assertNotIn("coverage_ratio", refreshed["metrics"])

    def test_generation_local_sync_and_result_path_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            results_dir = run_dir / "results"
            results_dir.mkdir()
            root_custom = results_dir / "custom_root_tiered_eval_summary.json"
            root_custom.write_text("{}", encoding="utf-8")
            nested_dir = results_dir / "nested"
            nested_dir.mkdir()
            nested_custom = nested_dir / "custom_nested_tiered_eval_summary.json"
            nested_custom.write_text("{}", encoding="utf-8")
            legacy = nested_dir / "tiered_eval_summary.json"
            legacy.write_text("{}", encoding="utf-8")
            generic_summary = nested_dir / "summary.json"
            generic_summary.write_text("{}", encoding="utf-8")
            eval_summary = nested_dir / "eval_summary.json"
            eval_summary.write_text("{}", encoding="utf-8")
            final_summary = nested_dir / "final_summary.json"
            final_summary.write_text("{}", encoding="utf-8")
            self.assertEqual(
                {path.name for path in fc.iter_result_summary_paths(run_dir)},
                {
                    "custom_root_tiered_eval_summary.json",
                    "custom_nested_tiered_eval_summary.json",
                    "tiered_eval_summary.json",
                    "summary.json",
                    "eval_summary.json",
                    "final_summary.json",
                },
            )
            self.assertTrue(fc.is_supported_result_summary_filename("eval_summary.json"))
            self.assertTrue(fc.is_supported_result_summary_filename("final_summary.json"))

            protocol_dir = results_dir / "candidate_nested" / "protocol_a"
            protocol_dir.mkdir(parents=True)
            protocol_summary = protocol_dir / "summary.json"
            protocol_summary.write_text(
                json.dumps(
                    {
                        "variant_name": "stable_candidate_id",
                        "metrics": {
                            "score": 0.75,
                            "scored_complete": True,
                            "effort_ratio": 1.0,
                            "coverage_ratio": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertIn(protocol_summary, fc.iter_result_summary_paths(run_dir))
            payload = json.loads(protocol_summary.read_text(encoding="utf-8"))
            self.assertEqual(
                fc.result_summary_variant_name(protocol_summary, payload, run_dir),
                "candidate_nested/protocol_a",
            )
            self.assertEqual(
                fc.result_summary_variant_name(protocol_summary, {}, run_dir),
                "candidate_nested/protocol_a",
            )

            legacy_nested_id = {
                "current_aggregate": {"variant_id": "shared_parent"},
                "metrics": {"variant_id": "shared_parent"},
            }
            self.assertEqual(
                fc.result_summary_variant_name(protocol_summary, legacy_nested_id, run_dir),
                "candidate_nested/protocol_a",
            )
            self.assertEqual(
                fc.result_summary_variant_name(
                    protocol_summary,
                    {
                        "variant_name": "shared_display_label",
                        "current_aggregate": {"child_id": "legacy_child"},
                    },
                    run_dir,
                ),
                "legacy_child",
            )
            self.assertEqual(
                fc.result_summary_variant_name(
                    protocol_summary,
                    {**legacy_nested_id, "variant_id": "producer_owned_candidate"},
                    run_dir,
                ),
                "producer_owned_candidate",
            )
            self.assertEqual(
                fc.result_summary_variant_name(
                    protocol_summary,
                    {"child_variant_id": "producer_owned_child"},
                    run_dir,
                ),
                "producer_owned_child",
            )
            canonical_id_summary = {
                "canonical_variant_id": "canonical_producer_id",
                "metrics": {"score": 1.0},
            }
            self.assertEqual(
                fc._result_summary_metrics(canonical_id_summary)["canonical_variant_id"],
                "canonical_producer_id",
            )
            self.assertEqual(
                fc.result_summary_variant_name(protocol_summary, canonical_id_summary, run_dir),
                "candidate_nested/protocol_a",
            )
            self.assertEqual(
                fc.result_summary_variant_name(
                    protocol_summary,
                    {"child_variant_id": "first_child", "result_variant_id": "second_child"},
                    run_dir,
                ),
                "first_child",
            )

            gen_dir = run_dir / "gen_3" / "shared_findings"
            gen_dir.mkdir(parents=True)
            (gen_dir / "bad.json").write_text("{bad", encoding="utf-8")
            local_payload = {
                "id": "local",
                "finding_type": "result",
                "variant_name": "v",
                "metrics": {"score": 1},
            }
            (gen_dir / "gen3_peer7_local.json").write_text(
                json.dumps(local_payload),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}, clear=False):
                ingested = fc._sync_generation_local_findings(run_dir=run_dir, gen_id=3)
                self.assertEqual(ingested, 1)
                self.assertFalse((run_dir / "shared_findings").exists())
                self.assertEqual(fc._sync_generation_local_findings(run_dir=run_dir, gen_id=3), 0)

                changed_payload = dict(local_payload, metrics={"score": 2})
                (gen_dir / "gen3_peer7_local.json").write_text(
                    json.dumps(changed_payload), encoding="utf-8"
                )
                self.assertEqual(fc._sync_generation_local_findings(run_dir=run_dir, gen_id=3), 1)
                with sqlite3.connect(run_dir / "shared_store.db") as conn:
                    generation_id, peer_id, metrics_json = conn.execute(
                        "SELECT generation_id, peer_id, metrics FROM findings "
                        "WHERE variant_name = 'v' ORDER BY rowid DESC LIMIT 1"
                    ).fetchone()
                self.assertEqual(generation_id, 3)
                self.assertEqual(peer_id, "gen3_peer7")
                self.assertEqual(json.loads(metrics_json)["score"], 2)

    def test_result_summary_discovery_preserves_distinct_symlink_paths(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            payload = run_dir / "shared_summary_payload.json"
            payload.write_text("{}", encoding="utf-8")
            for name in ("variant_a", "variant_b"):
                result_dir = run_dir / "results" / name
                result_dir.mkdir(parents=True)
                (result_dir / "summary.json").symlink_to(payload)

            paths = fc.iter_result_summary_paths(run_dir)

        self.assertEqual(
            [str(path.relative_to(run_dir)) for path in paths],
            ["results/variant_a/summary.json", "results/variant_b/summary.json"],
        )

    def test_result_summary_discovery_follows_symlink_directories_without_cycles(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            results_dir = run_dir / "results"
            results_dir.mkdir()
            external = run_dir / "external_producer"
            external.mkdir()
            (external / "summary.json").write_text("{}", encoding="utf-8")
            (external / "cycle").symlink_to(results_dir, target_is_directory=True)
            for name in ("producer_a", "producer_b"):
                (results_dir / name).symlink_to(external, target_is_directory=True)

            paths = fc.iter_result_summary_paths(run_dir)

        self.assertEqual(
            [str(path.relative_to(run_dir)) for path in paths],
            ["results/producer_a/summary.json", "results/producer_b/summary.json"],
        )

    def test_filesystem_fallback_reads_generation_local_findings_without_root_dir(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            generation_findings = run_dir / "gen_4" / "shared_findings"
            generation_findings.mkdir(parents=True)
            (generation_findings / "result.json").write_text(
                json.dumps(
                    {
                        "id": "generation-local-result",
                        "finding_type": "result",
                        "generation_id": 4,
                        "peer_id": "gen4_peer1",
                        "variant_name": "candidate",
                    }
                ),
                encoding="utf-8",
            )

            findings = fc.collect_findings_for_generation(
                findings_dir=run_dir / "shared_findings",
                gen_id=4,
                local_mode=False,
                materialize_result_artifacts=False,
            )

        self.assertEqual([row["id"] for row in findings], ["generation-local-result"])

    def test_metrics_only_summary_json_materializes_before_frontier_promotion(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
            FrontierStore,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            summary_dir = run_dir / "results" / "gen4_peer0_c04_1e9_full"
            summary_dir.mkdir(parents=True)
            summary_path = summary_dir / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "mode": "full",
                        "cell_count": 10,
                        "frontier_lane": "confirmed",
                        "evidence_stage": "full_eval",
                        "status": "ok",
                        "primary_metric": "task_score",
                        "metrics": {
                            "invalid_runs": 0,
                            "scored_complete": True,
                            "task_score": 61.45295685700597,
                            "success_rate": 0.5,
                        },
                    }
                ),
                encoding="utf-8",
            )

            materialized = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=4,
                scoring_metric_keys=("task_score",),
                default_lane="task_candidate",
            )
            store = FrontierStore(
                run_dir / "frontier",
                primary_metric="task_score",
                metric_direction="maximize",
                frontier_lanes=[
                    {
                        "name": "confirmed",
                        "k": 4,
                        "include_lanes": ["confirmed"],
                        "require_metrics": ["task_score"],
                        "axes": [{"name": "task_score", "direction": "maximize"}],
                    }
                ],
            )
            promoted = store.promote(4, materialized)

            self.assertEqual(len(materialized), 1)
            finding = materialized[0]
            self.assertEqual(finding["variant_name"], "gen4_peer0_c04_1e9_full")
            self.assertEqual(finding["generation_id"], 4)
            self.assertEqual(
                finding["metrics"]["source_result_path"],
                str(summary_path.relative_to(run_dir)),
            )
            self.assertEqual(finding["metrics"]["source_result_kind"], "summary.json")
            self.assertEqual(finding["metrics"]["frontier_lane"], "confirmed")
            self.assertEqual(finding["metrics"]["evidence_stage"], "full_eval")
            self.assertTrue(finding["metrics"]["scored_complete"])
            self.assertEqual(finding["metrics"]["result_status"], "scored_complete")
            self.assertEqual(finding["metrics"]["task_score"], 61.45295685700597)
            self.assertTrue((run_dir / "shared_findings").exists())
            self.assertEqual(len(promoted), 1)
            self.assertEqual(promoted[0]["variant_name"], "gen4_peer0_c04_1e9_full")
            self.assertEqual(promoted[0]["metric_value"], 61.45295685700597)
            self.assertEqual(promoted[0]["frontier_lane"], "confirmed")

    def test_result_materialization_refreshes_low_confidence_provenance(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "Alpha Child"
            result_dir.mkdir(parents=True)
            summary_path = result_dir / "tiered_eval_summary.json"
            summary = {
                "variant_name": "family",
                "generation_id": "gen_4",
                "tiers": [
                    {
                        "tier": "T3",
                        "status": "passed_t3",
                        "metrics_summary": {
                            "mean_test_taskscore": 9.0,
                            "hard_constraint_violations": [],
                            "max_drawdown_pct": 4.0,
                        },
                    }
                ],
                "all_eval_cells": [
                    {"return_pct": 1.0, "mdd_pct": 2.0, "validation_only": "false"},
                    {"return_pct": 3.0, "mdd_pct": 4.0, "validation_only": True},
                ],
            }
            normalized = fc.normalized_result_summary(summary, summary_path=summary_path)
            digest = fc._json_digest(normalized)
            rel = str(summary_path.relative_to(run_dir))
            findings_dir = run_dir / "shared_findings"
            findings_dir.mkdir()
            stale = findings_dir / "stale.json"
            stale.write_text(
                json.dumps(
                    {
                        "id": "stale",
                        "metrics": {
                            "auto_materialized_from_result_artifact": True,
                            "source_result_path": rel,
                            "source_result_sha256": digest,
                            "source_generation_inference": "boundary_fallback",
                            "source_generation_low_confidence": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            materialized = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=0,
                scoring_metric_keys=("mean_test_taskscore",),
                cell_metric_derivations=[
                    {
                        "name": "mean_mdd_pct",
                        "source_keys": ["mdd_pct"],
                        "aggregate": "mean",
                        "validation_only": "false",
                    },
                    {
                        "name": "validation_2026_return_pct",
                        "source_keys": ["return_pct"],
                        "aggregate": "mean",
                        "validation_only": True,
                    },
                ],
                metric_aliases={"promotion_mean_mdd_pct": "mean_mdd_pct"},
            )
            self.assertEqual(len(materialized), 1)
            finding = materialized[0]
            self.assertEqual(finding["variant_name"], "Alpha Child")
            self.assertEqual(finding["generation_id"], 4)
            self.assertEqual(
                finding["metrics"]["source_generation_inference"], "summary_generation_id"
            )
            self.assertEqual(finding["metrics"]["promotion_mean_mdd_pct"], 2.0)
            self.assertEqual(finding["metrics"]["validation_2026_return_pct"], 3.0)
            self.assertFalse(stale.exists())

            invalid_dir = run_dir / "results" / "invalid"
            invalid_dir.mkdir()
            (invalid_dir / "tiered_eval_summary.json").write_text("[]", encoding="utf-8")
            self.assertEqual(
                fc._materialize_result_artifacts(
                    run_dir=run_dir,
                    gen_id=0,
                    cell_metric_derivations=[
                        {
                            "name": "mean_mdd_pct",
                            "source_keys": ["mdd_pct"],
                            "aggregate": "mean",
                        },
                        {
                            "name": "validation_2026_return_pct",
                            "source_keys": ["return_pct"],
                            "aggregate": "mean",
                            "validation_only": True,
                        },
                    ],
                    metric_aliases={"promotion_mean_mdd_pct": "mean_mdd_pct"},
                ),
                [],
            )

    def test_recursive_result_paths_share_explicit_producer_identity(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
            FrontierStore,
            _candidate_entity_key,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)

            def write_summary(relative_dir: str, *, identity_key: str, identity: str) -> None:
                summary_dir = run_dir / "results" / relative_dir
                summary_dir.mkdir(parents=True)
                (summary_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            identity_key: identity,
                            "generation_id": 2,
                            "frontier_lane": "candidate",
                            "metrics": {
                                "score": 1.0,
                                "scored_complete": True,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            write_summary("candidate/protocol_a", identity_key="variant_id", identity="stable-a")
            write_summary("candidate/protocol_b", identity_key="variant_id", identity="stable-a")
            write_summary(
                "candidate/protocol_c",
                identity_key="child_variant_id",
                identity="stable-b",
            )

            materialized = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=2,
                scoring_metric_keys=("score",),
                default_lane="candidate",
            )
            stable_a = [
                finding
                for finding in materialized
                if finding["metrics"].get("canonical_variant_name") == "stable-a"
            ]
            stable_b = [
                finding
                for finding in materialized
                if finding["metrics"].get("canonical_variant_name") == "stable-b"
            ]
            store = FrontierStore(
                run_dir / "frontier",
                primary_metric="score",
                promote_top_k=5,
                frontier_lanes=[
                    {
                        "name": "candidate",
                        "k": 5,
                        "include_lanes": ["candidate"],
                        "axes": [{"metric": "score", "direction": "maximize"}],
                    }
                ],
            )
            promoted = store.promote(2, materialized)

        self.assertEqual(len(materialized), 3)
        self.assertEqual(len(stable_a), 2)
        self.assertEqual(len(stable_b), 1)
        self.assertEqual(
            {_candidate_entity_key(finding) for finding in stable_a}, {"variant::stable-a"}
        )
        self.assertEqual(_candidate_entity_key(stable_b[0]), "variant::stable-b")
        self.assertEqual(
            {_candidate_entity_key(finding) for finding in promoted},
            {"variant::stable-a", "variant::stable-b"},
        )

    def test_result_materialization_preserves_root_nested_producer_conflict(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
            result_snapshot_key,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "conflicting_candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "child_id": "root-child",
                        "generation_id": 0,
                        "metrics": {
                            "child_id": "nested-child",
                            "score": 1.0,
                            "scored_complete": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=0,
                scoring_metric_keys=("score",),
            )

        self.assertEqual(finding["child_id"], "root-child")
        self.assertEqual(finding["metrics"]["child_id"], "nested-child")
        self.assertIsNone(result_snapshot_key(finding))

    def test_result_materialization_retains_boundary_fallback_with_low_confidence_marker(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "boundaryless_candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "boundaryless_candidate",
                        "current_aggregate": {
                            "score": 0.81,
                            "scored_complete": True,
                        },
                        "n_eval_cells": 3,
                        "scored_complete": True,
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(run_dir=run_dir, gen_id=7)

            self.assertEqual(finding["generation_id"], 7)
            self.assertEqual(finding["metrics"]["source_generation_id"], 7)
            self.assertEqual(
                finding["metrics"]["source_generation_inference"],
                "boundary_fallback",
            )
            self.assertTrue(finding["metrics"]["source_generation_low_confidence"])
            self.assertTrue(finding["metrics"]["excluded_from_durable_frontier"])
            self.assertEqual(
                finding["metrics"]["exclusion_reason"],
                "source_generation_low_confidence",
            )
            self.assertFalse(finding["metrics"]["promotion_eligible"])

    def test_late_result_artifact_after_boundary_is_retained_as_validation_signal(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            boundary = run_dir / "gen_1" / "generation_boundary.json"
            boundary.parent.mkdir(parents=True)
            boundary.write_text(json.dumps({"generation_id": 1}), encoding="utf-8")
            old = 1_800_000_000
            os.utime(boundary, (old, old))
            result_dir = run_dir / "results" / "gen1_peer3_late_candidate"
            result_dir.mkdir(parents=True)
            summary_path = result_dir / "tiered_eval_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "variant_name": "gen1_peer3_late_candidate",
                        "generation_id": 1,
                        "current_aggregate": {
                            "score": 0.91,
                            "scored_complete": True,
                        },
                        "n_eval_cells": 3,
                        "scored_complete": True,
                    }
                ),
                encoding="utf-8",
            )
            late = old + 120
            os.utime(summary_path, (late, late))

            [finding] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=2,
                scoring_metric_keys=("score",),
            )

        metrics = finding["metrics"]
        self.assertEqual(finding["generation_id"], 1)
        self.assertEqual(finding["variant_name"], "gen1_peer3_late_candidate")
        self.assertEqual(metrics["source_generation_inference"], "summary_generation_id")
        self.assertTrue(metrics["late_after_generation_boundary"])
        self.assertEqual(metrics["artifact_signal_status"], "late_after_generation_boundary")
        self.assertEqual(metrics["generation_boundary_path"], "gen_1/generation_boundary.json")
        self.assertEqual(metrics["score"], 0.91)
        self.assertTrue(metrics["excluded_from_durable_frontier"])
        self.assertFalse(metrics["promotion_eligible"])
        self.assertFalse(metrics["clean_promotion_eligible"])
        self.assertEqual(metrics["exclusion_reason"], "late_after_generation_boundary")
        self.assertIn("late result validation signal", finding["title"])

    def test_boundary_evidence_cutoff_closes_final_refresh_to_marker_gap(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            old = 1_800_000_000
            cutoff_at = fc._timestamp_from_mtime(old + 60)
            boundary = run_dir / "gen_1" / "generation_boundary.json"
            boundary.parent.mkdir(parents=True)
            boundary.write_text(
                json.dumps(
                    {
                        "generation_id": 1,
                        "evidence_cutoff_at": cutoff_at,
                    }
                ),
                encoding="utf-8",
            )
            os.utime(boundary, (old + 180, old + 180))
            result_dir = run_dir / "results" / "gen1_peer3_gap_candidate"
            result_dir.mkdir(parents=True)
            summary_path = result_dir / "tiered_eval_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "variant_name": "gen1_peer3_gap_candidate",
                        "generation_id": 1,
                        "current_aggregate": {"score": 0.91, "scored_complete": True},
                        "n_eval_cells": 3,
                    }
                ),
                encoding="utf-8",
            )
            # The result completed after the final evidence cutoff but before
            # the later PI/Gems boundary marker write.
            os.utime(summary_path, (old + 120, old + 120))

            [finding] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=2,
                scoring_metric_keys=("score",),
            )

        metrics = finding["metrics"]
        self.assertTrue(metrics["late_after_generation_boundary"])
        self.assertTrue(metrics["excluded_from_durable_frontier"])
        self.assertEqual(metrics["generation_boundary_evidence_cutoff_at"], cutoff_at)

    def test_boundary_collector_quarantines_only_post_cutoff_nonstandard_sources(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            results_dir = run_dir / "results"
            findings_dir = run_dir / "shared_findings"
            results_dir.mkdir()
            findings_dir.mkdir()
            early_source = results_dir / "early-candidate.json"
            early_source.write_text("{}", encoding="utf-8")
            (findings_dir / "early.json").write_text(
                json.dumps(
                    {
                        "id": "early",
                        "generation_id": 0,
                        "peer_id": "gen0_peer0",
                        "source_result_path": "results/early-candidate.json",
                        "metrics": {"score": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            time.sleep(0.02)
            cutoff = datetime.now(UTC)
            evidence_snapshot = fc.include_finding_sources_in_snapshot(
                {},
                [
                    {
                        "id": "early",
                        "generation_id": 0,
                        "peer_id": "gen0_peer0",
                        "source_result_path": "results/early-candidate.json",
                        "metrics": {"score": 1.0},
                    }
                ],
                run_dir=run_dir,
                findings_dir=findings_dir,
                gen_id=0,
                cutoff=cutoff,
                canonical_findings=False,
            )
            time.sleep(0.02)
            late_source = results_dir / "late-candidate.json"
            late_source.write_text("{}", encoding="utf-8")
            late_finding_path = findings_dir / "late.json"
            late_finding_path.write_text(
                json.dumps(
                    {
                        "id": "late",
                        "generation_id": 0,
                        "peer_id": "gen0_peer1",
                        "source_result_path": "results/late-candidate.json",
                        "metrics": {"score": 2.0},
                        "task_evidence": {"trace": ["kept"]},
                    }
                ),
                encoding="utf-8",
            )
            late_finding_bytes = late_finding_path.read_bytes()
            loop = SimpleNamespace(
                findings_dir=findings_dir,
                local_mode=False,
                task_spec=SimpleNamespace(
                    evaluation=SimpleNamespace(primary_metric="score"),
                    gems=SimpleNamespace(result_artifact_materialization=False),
                ),
                _boundary_evidence_cutoff=(0, cutoff, evidence_snapshot),
            )

            findings = fc.collect_loop_findings(loop, 0)

            by_id = {finding["id"]: finding for finding in findings}
            self.assertNotIn("late_after_generation_boundary", by_id["early"]["metrics"])
            self.assertTrue(by_id["late"]["metrics"]["late_after_generation_boundary"])
            self.assertFalse(by_id["late"]["metrics"]["promotion_eligible"])
            persisted = json.loads(late_finding_path.read_text(encoding="utf-8"))
            self.assertEqual(late_finding_path.read_bytes(), late_finding_bytes)
            self.assertNotIn("late_after_generation_boundary", persisted["metrics"])
            self.assertEqual(persisted["task_evidence"], {"trace": ["kept"]})

    def test_boundary_collector_quarantines_post_cutoff_source_less_finding(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            findings_dir = run_dir / "shared_findings"
            findings_dir.mkdir()
            (findings_dir / "early.json").write_text(
                json.dumps(
                    {
                        "id": "early-inline",
                        "generation_id": 0,
                        "peer_id": "gen0_peer0",
                        "metrics": {"score": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            time.sleep(0.02)
            cutoff = datetime.now(UTC)
            evidence_snapshot = fc.include_finding_sources_in_snapshot(
                {},
                [
                    {
                        "id": "early-inline",
                        "generation_id": 0,
                        "peer_id": "gen0_peer0",
                        "metrics": {"score": 1.0},
                    }
                ],
                run_dir=run_dir,
                findings_dir=findings_dir,
                gen_id=0,
                cutoff=cutoff,
                canonical_findings=False,
            )
            time.sleep(0.02)
            late_path = findings_dir / "late.json"
            late_path.write_text(
                json.dumps(
                    {
                        "id": "late-inline",
                        "generation_id": 0,
                        "peer_id": "gen0_peer1",
                        "metrics": {"score": 2.0},
                    }
                ),
                encoding="utf-8",
            )
            loop = SimpleNamespace(
                findings_dir=findings_dir,
                local_mode=False,
                task_spec=SimpleNamespace(
                    evaluation=SimpleNamespace(primary_metric="score"),
                    gems=SimpleNamespace(result_artifact_materialization=False),
                ),
                _boundary_evidence_cutoff=(0, cutoff, evidence_snapshot),
            )

            findings = fc.collect_loop_findings(loop, 0)

            by_id = {finding["id"]: finding for finding in findings}
            self.assertNotIn("late_after_generation_boundary", by_id["early-inline"]["metrics"])
            self.assertTrue(by_id["late-inline"]["metrics"]["late_after_generation_boundary"])
            persisted = json.loads(late_path.read_text(encoding="utf-8"))
            self.assertNotIn("late_after_generation_boundary", persisted["metrics"])

    def test_boundary_collector_persists_quarantine_in_canonical_store_only(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            findings_dir = run_dir / "shared_findings"
            findings_dir.mkdir()
            cutoff = datetime.now(UTC)
            time.sleep(0.02)
            late_path = findings_dir / "late.json"
            late_path.write_text(
                json.dumps(
                    {
                        "id": "late-inline",
                        "generation_id": 0,
                        "peer_id": "gen0_peer0",
                        "metrics": {"score": 2.0},
                        "task_evidence": {"trace": ["untouched"]},
                    }
                ),
                encoding="utf-8",
            )
            original_bytes = late_path.read_bytes()
            loop = SimpleNamespace(
                findings_dir=findings_dir,
                local_mode=True,
                task_spec=SimpleNamespace(
                    evaluation=SimpleNamespace(primary_metric="score"),
                    gems=SimpleNamespace(result_artifact_materialization=False),
                ),
                _boundary_evidence_cutoff=(0, cutoff, {}),
            )

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}):
                findings = fc.collect_loop_findings(loop, 0)
                with sqlite3.connect(run_dir / "shared_store.db") as conn:
                    [metrics_json] = conn.execute("SELECT metrics FROM findings").fetchone()

            self.assertTrue(findings[0]["metrics"]["late_after_generation_boundary"])
            self.assertTrue(json.loads(metrics_json)["late_after_generation_boundary"])
            self.assertEqual(late_path.read_bytes(), original_bytes)

    def test_boundary_collector_quarantines_sqlite_only_finding_added_after_cutoff(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.tools.local_store import (
            get_findings,
            init_db,
            insert_finding,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            findings_dir = run_dir / "shared_findings"
            findings_dir.mkdir()
            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}):
                init_db()
                insert_finding(
                    {
                        "id": "sqlite-early",
                        "generation_id": 0,
                        "peer_id": "gen0_peer0",
                        "metrics": {"score": 1.0},
                    }
                )
                cutoff = datetime.now(UTC)
                evidence_snapshot = fc.include_finding_sources_in_snapshot(
                    {},
                    get_findings(generation_id=0),
                    run_dir=run_dir,
                    findings_dir=findings_dir,
                    gen_id=0,
                    cutoff=cutoff,
                )
                insert_finding(
                    {
                        "id": "sqlite-late",
                        "generation_id": 0,
                        "peer_id": "gen0_peer1",
                        "metrics": {"score": 2.0},
                    }
                )
                loop = SimpleNamespace(
                    findings_dir=findings_dir,
                    local_mode=True,
                    task_spec=SimpleNamespace(
                        evaluation=SimpleNamespace(primary_metric="score"),
                        gems=SimpleNamespace(result_artifact_materialization=False),
                    ),
                    _boundary_evidence_cutoff=(0, cutoff, evidence_snapshot),
                )

                findings = fc.collect_loop_findings(loop, 0)

            by_id = {finding["id"]: finding for finding in findings}
            self.assertNotIn("late_after_generation_boundary", by_id["sqlite-early"]["metrics"])
            self.assertTrue(by_id["sqlite-late"]["metrics"]["late_after_generation_boundary"])

    def test_boundary_collector_quarantines_sqlite_only_finding_changed_after_cutoff(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            finding = {
                "id": "sqlite-updated",
                "generation_id": 0,
                "peer_id": "gen0_peer0",
                "metrics": {"score": 1.0},
            }
            cutoff = datetime.now(UTC)
            snapshot = fc.include_finding_sources_in_snapshot(
                {},
                [finding],
                run_dir=run_dir,
                findings_dir=run_dir / "shared_findings",
                gen_id=0,
                cutoff=cutoff,
            )
            updated = {**finding, "metrics": {"score": 2.0}}

            self.assertTrue(
                fc.finding_source_published_after(
                    updated,
                    run_dir=run_dir,
                    cutoff=cutoff,
                    evidence_source_snapshot=snapshot,
                )
            )

    def test_canonical_snapshot_rejects_new_finding_that_reuses_old_result_source(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            source = run_dir / "results" / "shared" / "summary.json"
            source.parent.mkdir(parents=True)
            source.write_text("{}", encoding="utf-8")
            cutoff = datetime.now(UTC)
            snapshot = fc.include_finding_sources_in_snapshot(
                {},
                [
                    {
                        "id": "known-at-cutoff",
                        "generation_id": 0,
                        "source_result_path": "results/shared/summary.json",
                        "metrics": {"score": 1.0},
                    }
                ],
                run_dir=run_dir,
                findings_dir=run_dir / "shared_findings",
                gen_id=0,
                cutoff=cutoff,
            )

            self.assertTrue(
                fc.finding_source_published_after(
                    {
                        "id": "published-after-cutoff",
                        "generation_id": 0,
                        "source_result_path": "results/shared/summary.json",
                        "metrics": {"score": 2.0},
                    },
                    run_dir=run_dir,
                    cutoff=cutoff,
                    evidence_source_snapshot=snapshot,
                )
            )

    def test_boundary_cutoff_restores_exact_canonical_value_and_compacts_it(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            cutoff = datetime.now(UTC)
            original = {
                "id": "updated-after-cutoff",
                "generation_id": 0,
                "metrics": {"score": 1.0},
            }
            snapshot = fc.include_finding_sources_in_snapshot(
                {},
                [original],
                run_dir=run_dir,
                findings_dir=run_dir / "shared_findings",
                gen_id=0,
                cutoff=cutoff,
            )
            updated = {
                **original,
                "metrics": {
                    "score": 2.0,
                    "late_after_generation_boundary": True,
                },
            }

            selected = fc.findings_at_boundary_cutoff(snapshot, [updated])
            compacted = fc.compact_boundary_source_snapshot(snapshot)

            self.assertEqual(selected, [original])
            finding_ref = "canonical-finding:updated-after-cutoff"
            self.assertTrue(snapshot[finding_ref].startswith("canonical-finding-payload:v1:"))
            self.assertEqual(len(compacted[finding_ref]), 64)
            self.assertNotIn("canonical-finding-payload", compacted[finding_ref])

    def test_empty_canonical_cutoff_still_excludes_late_findings(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        accepted = {"id": "accepted-filesystem", "metrics": {"score": 1.0}}
        late = {
            "id": "late-canonical-row",
            "metrics": {
                "score": 2.0,
                "late_after_generation_boundary": True,
            },
        }

        selected = fc.findings_at_boundary_cutoff(
            {"canonical-finding-snapshot:v1": "captured"},
            [accepted, late],
        )

        self.assertEqual(selected, [accepted])

    def test_boundary_fingerprint_ignores_regenerated_ingest_provenance(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            cutoff = datetime.now(UTC)
            finding = {
                "id": "relocated-finding",
                "generation_id": 0,
                "metrics": {"score": 1.0, "promotion_eligible": True},
                "source_filename": "candidate.json",
                "source_filepath": "/old/run/shared_findings/candidate.json",
                "source_mtime_ns": 100,
                "ingest_schema_version": "v1",
                "ingested_at": "2026-08-01T00:00:00+00:00",
            }
            snapshot = fc.include_finding_sources_in_snapshot(
                {},
                [finding],
                run_dir=run_dir,
                findings_dir=run_dir / "shared_findings",
                gen_id=0,
                cutoff=cutoff,
            )
            relocated = {
                **finding,
                "source_filepath": "/new/run/shared_findings/candidate.json",
                "source_mtime_ns": 999,
                "ingest_schema_version": "v2",
                "ingested_at": "2026-08-04T00:00:00+00:00",
            }

            self.assertFalse(
                fc.finding_source_published_after(
                    relocated,
                    run_dir=run_dir,
                    cutoff=cutoff,
                    evidence_source_snapshot=snapshot,
                )
            )

    def test_boundary_fingerprint_tracks_promotion_routing_changes(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            finding = {
                "id": "routing-updated",
                "generation_id": 0,
                "metrics": {
                    "score": 1.0,
                    "promotion_eligible": False,
                    "validation_only_result": True,
                },
            }
            cutoff = datetime.now(UTC)
            snapshot = fc.include_finding_sources_in_snapshot(
                {},
                [finding],
                run_dir=run_dir,
                findings_dir=run_dir / "shared_findings",
                gen_id=0,
                cutoff=cutoff,
            )
            changed = {
                **finding,
                "metrics": {
                    "score": 1.0,
                    "promotion_eligible": True,
                    "validation_only_result": False,
                },
            }

            self.assertTrue(
                fc.finding_source_published_after(
                    changed,
                    run_dir=run_dir,
                    cutoff=cutoff,
                    evidence_source_snapshot=snapshot,
                )
            )

    def test_boundary_fingerprint_tracks_canonical_update_with_unchanged_source(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            source = run_dir / "results" / "candidate.json"
            source.parent.mkdir()
            source.write_text("{}", encoding="utf-8")
            finding = {
                "id": "source-backed-update",
                "generation_id": 0,
                "source_result_path": "results/candidate.json",
                "metrics": {"score": 1.0},
            }
            cutoff = datetime.now(UTC)
            snapshot = fc.include_finding_sources_in_snapshot(
                {},
                [finding],
                run_dir=run_dir,
                findings_dir=run_dir / "shared_findings",
                gen_id=0,
                cutoff=cutoff,
            )
            changed = {**finding, "metrics": {"score": 99.0}}

            self.assertTrue(
                fc.finding_source_published_after(
                    changed,
                    run_dir=run_dir,
                    cutoff=cutoff,
                    evidence_source_snapshot=snapshot,
                )
            )

    def test_boundary_source_resolution_prefers_sqlite_source_path_for_duplicate_ids(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            findings_dir = run_dir / "shared_findings"
            findings_dir.mkdir()
            duplicate_id = "16f12a16-c6f0-4b36-ae15-1efb11b04f95"
            early_path = findings_dir / "early.json"
            early_path.write_text(json.dumps({"id": duplicate_id}), encoding="utf-8")
            time.sleep(0.02)
            cutoff = datetime.now(UTC)
            evidence_snapshot = fc.include_finding_sources_in_snapshot(
                {},
                [
                    {
                        "id": duplicate_id,
                        "source_filepath": str(early_path),
                        "metrics": {"score": 1.0},
                    }
                ],
                run_dir=run_dir,
                findings_dir=findings_dir,
                gen_id=0,
                cutoff=cutoff,
            )
            time.sleep(0.02)
            late_path = findings_dir / "late.json"
            late_path.write_text(json.dumps({"id": duplicate_id}), encoding="utf-8")
            source_index = fc._finding_source_index(
                findings_dir=findings_dir,
                run_dir=run_dir,
                gen_id=0,
            )
            early = {
                "id": duplicate_id,
                "source_filepath": str(early_path),
                "metrics": {"score": 1.0},
            }
            late = {
                "id": "source-scoped-id",
                "source_filepath": str(late_path),
                "metrics": {"score": 2.0},
            }

            self.assertEqual(fc._finding_source_paths(early, source_index), [early_path])
            self.assertEqual(fc._finding_source_paths(late, source_index), [late_path])
            annotated = fc.annotate_late_boundary_findings(
                [early, late],
                run_dir=run_dir,
                findings_dir=findings_dir,
                gen_id=0,
                cutoff=cutoff,
                evidence_source_snapshot=evidence_snapshot,
            )

            self.assertNotIn("late_after_generation_boundary", annotated[0]["metrics"])
            self.assertTrue(annotated[1]["metrics"]["late_after_generation_boundary"])

    def test_filesystem_fallback_preserves_duplicate_ids_as_distinct_evidence(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            findings_dir = run_dir / "shared_findings"
            findings_dir.mkdir()
            duplicate_id = "16f12a16-c6f0-4b36-ae15-1efb11b04f95"
            early_path = findings_dir / "early.json"
            early_path.write_text(
                json.dumps(
                    {
                        "id": duplicate_id,
                        "generation_id": 0,
                        "metrics": {"score": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            early_findings = fc.collect_findings_for_generation(
                findings_dir=findings_dir,
                gen_id=0,
                local_mode=False,
                materialize_result_artifacts=False,
            )
            cutoff = datetime.now(UTC)
            evidence_snapshot = fc.include_finding_sources_in_snapshot(
                {},
                early_findings,
                run_dir=run_dir,
                findings_dir=findings_dir,
                gen_id=0,
                cutoff=cutoff,
            )
            time.sleep(0.02)
            generation_findings = run_dir / "gen_0" / "shared_findings"
            generation_findings.mkdir(parents=True)
            late_path = generation_findings / "late.json"
            late_path.write_text(
                json.dumps(
                    {
                        "id": duplicate_id,
                        "generation_id": 0,
                        "metrics": {"score": 2.0},
                    }
                ),
                encoding="utf-8",
            )
            loop = SimpleNamespace(
                findings_dir=findings_dir,
                local_mode=False,
                task_spec=SimpleNamespace(
                    evaluation=SimpleNamespace(primary_metric="score"),
                    gems=SimpleNamespace(result_artifact_materialization=False),
                ),
                _boundary_evidence_cutoff=(0, cutoff, evidence_snapshot),
            )

            findings = fc.collect_loop_findings(loop, 0)

            self.assertEqual(len(findings), 2)
            self.assertEqual(len({finding["id"] for finding in findings}), 2)
            by_score = {finding["metrics"]["score"]: finding for finding in findings}
            self.assertNotIn("late_after_generation_boundary", by_score[1.0]["metrics"])
            self.assertTrue(by_score[2.0]["metrics"]["late_after_generation_boundary"])
            self.assertEqual(Path(by_score[1.0]["source_filepath"]), early_path)
            self.assertEqual(Path(by_score[2.0]["source_filepath"]), late_path)

    def test_boundary_collector_detects_atomic_generation_finding_directory(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            findings_dir = run_dir / "shared_findings"
            findings_dir.mkdir()
            staged = run_dir / "staged_findings"
            staged.mkdir()
            finding_path = staged / "late.json"
            finding_path.write_text(
                json.dumps(
                    {
                        "id": "atomic-late",
                        "generation_id": 0,
                        "metrics": {"score": 2.0},
                    }
                ),
                encoding="utf-8",
            )
            time.sleep(0.02)
            cutoff = datetime.now(UTC)
            time.sleep(0.02)
            generation_findings = run_dir / "gen_0" / "shared_findings"
            generation_findings.parent.mkdir()
            staged.rename(generation_findings)
            loop = SimpleNamespace(
                findings_dir=findings_dir,
                local_mode=False,
                task_spec=SimpleNamespace(
                    evaluation=SimpleNamespace(primary_metric="score"),
                    gems=SimpleNamespace(result_artifact_materialization=False),
                ),
                _boundary_evidence_cutoff=(0, cutoff, {}),
            )

            findings = fc.collect_loop_findings(loop, 0)

            by_id = {finding["id"]: finding for finding in findings}
            self.assertTrue(by_id["atomic-late"]["metrics"]["late_after_generation_boundary"])

    def test_boundary_snapshot_includes_pre_cutoff_task_defined_result_source(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            findings_dir = run_dir / "shared_findings"
            findings_dir.mkdir()
            finding_path = findings_dir / "candidate.json"
            finding_path.write_text(
                json.dumps({"id": "candidate", "metrics": {"score": 1.0}}),
                encoding="utf-8",
            )
            source = run_dir / "task_outputs" / "candidate.json"
            source.parent.mkdir()
            source.write_text("{}", encoding="utf-8")
            cutoff = datetime.now(UTC)

            snapshot = fc.include_finding_sources_in_snapshot(
                {},
                [
                    {
                        "id": "candidate",
                        "source_result_path": "task_outputs/candidate.json",
                        "metrics": {"score": 1.0},
                    }
                ],
                run_dir=run_dir,
                findings_dir=findings_dir,
                gen_id=0,
                cutoff=cutoff,
            )

            self.assertIn("task_outputs/candidate.json", snapshot)
            self.assertIn("shared_findings/candidate.json", snapshot)
            self.assertFalse(
                fc.finding_source_published_after(
                    {
                        "source_result_path": "task_outputs/candidate.json",
                        "metrics": {"score": 1.0},
                    },
                    run_dir=run_dir,
                    cutoff=cutoff,
                    evidence_source_snapshot=snapshot,
                )
            )

    def test_snapshotted_result_directory_ignores_sibling_child_publication(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            findings_dir = run_dir / "shared_findings"
            findings_dir.mkdir()
            source_dir = run_dir / "task_outputs" / "candidate"
            source_dir.mkdir(parents=True)
            (source_dir / "summary.json").write_text("{}", encoding="utf-8")
            finding = {
                "id": "directory-backed",
                "source_result_path": "task_outputs/candidate",
                "metrics": {"score": 1.0},
            }
            cutoff = datetime.now(UTC)
            snapshot = fc.include_finding_sources_in_snapshot(
                {},
                [finding],
                run_dir=run_dir,
                findings_dir=findings_dir,
                gen_id=0,
                cutoff=cutoff,
            )
            time.sleep(0.02)
            (source_dir / "worker.log").write_text("finished\n", encoding="utf-8")

            self.assertFalse(
                fc.finding_source_published_after(
                    finding,
                    run_dir=run_dir,
                    cutoff=cutoff,
                    evidence_source_snapshot=snapshot,
                )
            )
            archived = run_dir / "archived-candidate"
            source_dir.rename(archived)
            source_dir.mkdir()
            (source_dir / "summary.json").write_text("{}", encoding="utf-8")
            self.assertTrue(
                fc.finding_source_published_after(
                    finding,
                    run_dir=run_dir,
                    cutoff=cutoff,
                    evidence_source_snapshot=snapshot,
                )
            )

    def test_next_generation_revalidation_does_not_inherit_old_pending_boundary(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            cutoff_gen0 = datetime.now(UTC)
            time.sleep(0.02)
            result_dir = run_dir / "results" / "revalidated"
            result_dir.mkdir(parents=True)
            summary_path = result_dir / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "variant_name": "revalidated",
                        "generation_id": 0,
                        "current_aggregate": {"score": 0.5, "scored_complete": True},
                    }
                ),
                encoding="utf-8",
            )
            [late_gen0] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=0,
                scoring_metric_keys=("score",),
                evidence_cutoff=cutoff_gen0,
                evidence_source_snapshot={},
            )
            self.assertTrue(late_gen0["metrics"]["generation_boundary_pending_commit"])
            boundary = run_dir / "gen_0" / "generation_boundary.json"
            boundary.parent.mkdir()
            boundary.write_text(json.dumps({"generation_id": 0}), encoding="utf-8")
            summary_path.write_text(
                json.dumps(
                    {
                        "variant_name": "revalidated",
                        "generation_id": 1,
                        "current_aggregate": {"score": 0.8, "scored_complete": True},
                    }
                ),
                encoding="utf-8",
            )
            cutoff_gen1, snapshot_gen1 = fc.result_source_snapshot_with_cutoff(run_dir)

            [revalidated] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=1,
                scoring_metric_keys=("score",),
                evidence_cutoff=cutoff_gen1,
                evidence_source_snapshot=snapshot_gen1,
            )

            metrics = revalidated["metrics"]
            self.assertEqual(revalidated["generation_id"], 1)
            self.assertNotIn("late_after_generation_boundary", metrics)
            self.assertNotIn("generation_boundary_pending_commit", metrics)
            self.assertNotIn("generation_boundary_path", metrics)

    def test_post_boundary_symlink_publication_is_retained_as_validation_signal(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            old = 1_800_000_000
            boundary = run_dir / "gen_1" / "generation_boundary.json"
            boundary.parent.mkdir(parents=True)
            boundary.write_text(json.dumps({"generation_id": 1}), encoding="utf-8")
            os.utime(boundary, (old + 60, old + 60))

            external = run_dir / "external_result"
            external.mkdir()
            target = external / "summary.json"
            target.write_text(
                json.dumps(
                    {
                        "variant_name": "linked_late_candidate",
                        "generation_id": 1,
                        "current_aggregate": {"score": 0.9, "scored_complete": True},
                        "scored_complete": True,
                    }
                ),
                encoding="utf-8",
            )
            os.utime(target, (old, old))
            logical_dir = run_dir / "results" / "linked_late_candidate"
            logical_dir.parent.mkdir()
            logical_dir.symlink_to(external, target_is_directory=True)
            os.utime(logical_dir, (old + 120, old + 120), follow_symlinks=False)

            [finding] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=2,
                scoring_metric_keys=("score",),
            )

        self.assertTrue(finding["metrics"]["late_after_generation_boundary"])
        self.assertTrue(finding["metrics"]["excluded_from_durable_frontier"])
        self.assertFalse(finding["metrics"]["promotion_eligible"])

    def test_atomic_result_directory_publication_is_detected_after_cutoff(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            staged = run_dir / "staging" / "gen0_peer0_atomic_candidate"
            staged.mkdir(parents=True)
            summary = staged / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "variant_name": "atomic_candidate",
                        "generation_id": 0,
                        "current_aggregate": {"score": 1.0, "scored_complete": True},
                    }
                ),
                encoding="utf-8",
            )
            old = time.time() - 3600
            os.utime(summary, (old, old))
            os.utime(staged, (old, old))
            cutoff = datetime.now(UTC)
            source_snapshot = fc.result_source_snapshot_at_cutoff(run_dir)
            time.sleep(0.02)
            destination = run_dir / "results" / staged.name
            destination.parent.mkdir()
            staged.rename(destination)

            [finding] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=0,
                scoring_metric_keys=("score",),
                evidence_cutoff=cutoff,
                evidence_source_snapshot=source_snapshot,
            )

            metrics = finding["metrics"]
            self.assertTrue(
                fc.finding_source_published_after(
                    finding,
                    run_dir=run_dir,
                    cutoff=cutoff,
                    evidence_source_snapshot=source_snapshot,
                )
            )
            self.assertTrue(metrics["late_after_generation_boundary"])
            self.assertTrue(metrics["generation_boundary_pending_commit"])
            self.assertTrue(metrics["excluded_from_durable_frontier"])

            boundary = run_dir / "gen_0" / "generation_boundary.json"
            boundary.parent.mkdir()
            boundary.write_text(
                json.dumps(
                    {
                        "generation_id": 0,
                        "evidence_cutoff_at": cutoff.isoformat(),
                        "evidence_source_snapshot_at_cutoff": {},
                    }
                ),
                encoding="utf-8",
            )
            [after_commit] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=1,
                scoring_metric_keys=("score",),
            )
            self.assertTrue(after_commit["metrics"]["late_after_generation_boundary"])
            self.assertNotIn(
                "generation_boundary_pending_commit",
                after_commit["metrics"],
            )

    def test_reconciled_cutoff_snapshot_admits_only_pre_cutoff_publications(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        source = "results/candidate/summary.json"
        for published_at, expected in ((0.0, True), (time.time() + 3600, False)):
            with (
                self.subTest(published_at=published_at),
                tempfile.TemporaryDirectory() as tmp,
                patch.object(
                    fc,
                    "result_source_snapshot_at_cutoff",
                    side_effect=[{}, {source: "target:1:2"}],
                ),
                patch.object(
                    fc,
                    "_result_discovery_mtime",
                    return_value=published_at,
                ),
            ):
                _cutoff, snapshot = fc.result_source_snapshot_with_cutoff(Path(tmp))

            self.assertEqual(source in snapshot, expected)

    def test_missing_result_reference_is_not_classified_as_new_publication(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            finding = {
                "source_result_path": "results/missing/summary.json",
                "metrics": {},
            }
            self.assertFalse(
                fc.finding_source_published_after(
                    finding,
                    run_dir=run_dir,
                    cutoff=datetime.now(UTC),
                    evidence_source_snapshot={},
                )
            )

    def test_reconciled_snapshot_rejects_nested_atomic_directory_after_cutoff(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            staged = run_dir / "staging" / "candidate" / "complete"
            staged.mkdir(parents=True)
            summary = staged / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "variant_name": "nested_atomic",
                        "generation_id": 0,
                        "current_aggregate": {"score": 1.0, "scored_complete": True},
                    }
                ),
                encoding="utf-8",
            )
            old = time.time() - 3600
            for path in (summary, staged, staged.parent):
                os.utime(path, (old, old))
            destination = run_dir / "results" / "candidate"
            destination.parent.mkdir()
            original_snapshot = fc.result_source_snapshot_at_cutoff
            scans = 0

            def scan(root):
                nonlocal scans
                scans += 1
                if scans == 1:
                    return {}
                time.sleep(0.02)
                staged.parent.rename(destination)
                return original_snapshot(root)

            with patch.object(fc, "result_source_snapshot_at_cutoff", side_effect=scan):
                cutoff, snapshot = fc.result_source_snapshot_with_cutoff(run_dir)

            source = "results/candidate/complete/summary.json"
            self.assertNotIn(source, snapshot)
            self.assertGreater(
                fc._result_discovery_mtime(run_dir / source, run_dir=run_dir),
                cutoff.timestamp(),
            )

    def test_sibling_result_publication_does_not_make_existing_summary_late(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            protocol_dir = run_dir / "results" / "full_protocol"
            first_dir = protocol_dir / "gen0_peer0_first"
            first_dir.mkdir(parents=True)
            (first_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "first",
                        "generation_id": 0,
                        "current_aggregate": {"score": 1.0, "scored_complete": True},
                    }
                ),
                encoding="utf-8",
            )
            cutoff = datetime.now(UTC)
            source_snapshot = fc.result_source_snapshot_at_cutoff(run_dir)
            time.sleep(0.02)
            (first_dir / "worker.log").write_text("finished\n", encoding="utf-8")
            second_dir = protocol_dir / "gen0_peer1_second"
            second_dir.mkdir()
            (second_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "second",
                        "generation_id": 0,
                        "current_aggregate": {"score": 2.0, "scored_complete": True},
                    }
                ),
                encoding="utf-8",
            )

            findings = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=0,
                scoring_metric_keys=("score",),
                evidence_cutoff=cutoff,
                evidence_source_snapshot=source_snapshot,
            )

            by_source = {finding["metrics"]["source_result_path"]: finding for finding in findings}
            self.assertNotIn(
                "late_after_generation_boundary",
                by_source["results/full_protocol/gen0_peer0_first/summary.json"]["metrics"],
            )
            self.assertTrue(
                by_source["results/full_protocol/gen0_peer1_second/summary.json"]["metrics"][
                    "late_after_generation_boundary"
                ]
            )

    def test_atomic_replacement_of_snapshotted_result_directory_is_late(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "gen0_peer0_replaced"
            result_dir.mkdir(parents=True)
            original = result_dir / "summary.json"
            original.write_text(
                json.dumps(
                    {
                        "variant_name": "original",
                        "generation_id": 0,
                        "current_aggregate": {"score": 1.0, "scored_complete": True},
                    }
                ),
                encoding="utf-8",
            )
            source_snapshot = fc.result_source_snapshot_at_cutoff(run_dir)
            cutoff = datetime.now(UTC)

            staged = run_dir / "staged_replacement"
            staged.mkdir()
            replacement = staged / "summary.json"
            replacement.write_text(
                json.dumps(
                    {
                        "variant_name": "replacement",
                        "generation_id": 0,
                        "current_aggregate": {"score": 2.0, "scored_complete": True},
                    }
                ),
                encoding="utf-8",
            )
            old = cutoff.timestamp() - 3600
            os.utime(replacement, (old, old))
            archived = run_dir / "archived_original"
            result_dir.rename(archived)
            staged.rename(result_dir)

            [finding] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=0,
                scoring_metric_keys=("score",),
                evidence_cutoff=cutoff,
                evidence_source_snapshot=source_snapshot,
            )

            self.assertTrue(finding["metrics"]["late_after_generation_boundary"])
            self.assertTrue(finding["metrics"]["generation_boundary_pending_commit"])
            self.assertFalse(finding["metrics"]["promotion_eligible"])

    def test_relocated_pending_run_accepts_unchanged_source_content_only(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_run = root / "original"
            restored_run = root / "restored"
            relative_source = Path("results/candidate/summary.json")
            original_source = original_run / relative_source
            restored_source = restored_run / relative_source
            original_source.parent.mkdir(parents=True)
            payload = json.dumps({"variant_name": "candidate", "score": 1.0})
            original_source.write_text(payload, encoding="utf-8")
            finding = {
                "id": "relocated-source",
                "generation_id": 0,
                "source_result_path": str(relative_source),
                "metrics": {"score": 1.0},
            }
            cutoff = datetime.now(UTC)
            snapshot = fc.include_finding_sources_in_snapshot(
                fc.result_source_snapshot_at_cutoff(original_run),
                [finding],
                run_dir=original_run,
                findings_dir=original_run / "shared_findings",
                gen_id=0,
                cutoff=cutoff,
            )
            time.sleep(0.02)
            restored_source.parent.mkdir(parents=True)
            restored_source.write_text(payload, encoding="utf-8")

            self.assertFalse(
                fc.finding_source_published_after(
                    finding,
                    run_dir=restored_run,
                    cutoff=cutoff,
                    evidence_source_snapshot=snapshot,
                )
            )

            restored_source.write_text(
                json.dumps({"variant_name": "candidate", "score": 2.0}),
                encoding="utf-8",
            )
            self.assertTrue(
                fc.finding_source_published_after(
                    finding,
                    run_dir=restored_run,
                    cutoff=cutoff,
                    evidence_source_snapshot=snapshot,
                )
            )

    def test_relocated_pending_run_materializes_unchanged_result_as_current(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_run = root / "original"
            restored_run = root / "restored"
            relative_source = Path("results/candidate/summary.json")
            original_source = original_run / relative_source
            restored_source = restored_run / relative_source
            original_source.parent.mkdir(parents=True)
            payload = json.dumps(
                {
                    "variant_name": "candidate",
                    "generation_id": 0,
                    "current_aggregate": {"score": 1.0, "scored_complete": True},
                }
            )
            original_source.write_text(payload, encoding="utf-8")
            cutoff = datetime.now(UTC)
            snapshot = fc.include_finding_sources_in_snapshot(
                fc.result_source_snapshot_at_cutoff(original_run),
                [],
                run_dir=original_run,
                findings_dir=original_run / "shared_findings",
                gen_id=0,
                cutoff=cutoff,
            )
            time.sleep(0.02)
            restored_source.parent.mkdir(parents=True)
            restored_source.write_text(payload, encoding="utf-8")

            [finding] = fc._materialize_result_artifacts(
                run_dir=restored_run,
                gen_id=0,
                scoring_metric_keys=("score",),
                evidence_cutoff=cutoff,
                evidence_source_snapshot=snapshot,
            )

            self.assertNotIn("late_after_generation_boundary", finding["metrics"])
            self.assertNotIn("generation_boundary_pending_commit", finding["metrics"])
            self.assertEqual(finding["metrics"]["score"], 1.0)
            self.assertEqual(
                finding["metrics"]["source_result_path"],
                relative_source.as_posix(),
            )

    def test_provisional_late_finding_stays_validation_only_through_boundary_retry(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )
        from praxist.plugins.workflow_stages.research_loop.backend import (
            resume_state,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "gen0_peer0_cutoff_candidate"
            result_dir.mkdir(parents=True)
            gen_dir = run_dir / "gen_0"
            gen_dir.mkdir()
            (gen_dir / "generation_results.json").write_text("[]", encoding="utf-8")
            summary = result_dir / "summary.json"
            cutoff = datetime.now(UTC)
            resume_state.write_boundary_evidence_checkpoint(
                run_dir,
                gen_id=0,
                cutoff=cutoff,
                evidence_source_snapshot={},
            )
            time.sleep(0.02)
            summary.write_text(
                json.dumps(
                    {
                        "variant_name": "cutoff_candidate",
                        "generation_id": 0,
                        "current_aggregate": {"score": 0.8, "scored_complete": True},
                    }
                ),
                encoding="utf-8",
            )

            [first] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=0,
                scoring_metric_keys=("score",),
                evidence_cutoff=cutoff,
            )
            [during_pi] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=0,
                scoring_metric_keys=("score",),
            )
            retry_cutoff = datetime.now(UTC)
            resume_state.write_boundary_evidence_checkpoint(
                run_dir,
                gen_id=0,
                cutoff=retry_cutoff,
                evidence_source_snapshot={},
            )
            [after_retry] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=0,
                scoring_metric_keys=("score",),
                evidence_cutoff=retry_cutoff,
            )
            boundary = gen_dir / "generation_boundary.json"
            boundary.write_text(
                json.dumps(
                    {
                        "generation_id": 0,
                        "evidence_cutoff_at": retry_cutoff.isoformat(),
                        "evidence_source_snapshot_at_cutoff": (
                            fc.result_source_snapshot_at_cutoff(run_dir)
                        ),
                    }
                ),
                encoding="utf-8",
            )
            [after_commit] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=1,
                scoring_metric_keys=("score",),
            )

            self.assertTrue(first["metrics"]["late_after_generation_boundary"])
            self.assertTrue(during_pi["metrics"]["late_after_generation_boundary"])
            self.assertTrue(during_pi["metrics"]["generation_boundary_pending_commit"])
            self.assertTrue(after_retry["metrics"]["late_after_generation_boundary"])
            self.assertTrue(after_retry["metrics"]["generation_boundary_pending_commit"])
            self.assertFalse(after_retry["metrics"]["promotion_eligible"])
            self.assertTrue(after_commit["metrics"]["late_after_generation_boundary"])
            self.assertNotIn(
                "generation_boundary_pending_commit",
                after_commit["metrics"],
            )
            self.assertFalse(after_commit["metrics"]["promotion_eligible"])

    def test_unobserved_post_cutoff_result_is_late_after_boundary_restart(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )
        from praxist.plugins.workflow_stages.research_loop.backend import (
            generation_boundary,
            resume_state,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen_dir = run_dir / "gen_0"
            gen_dir.mkdir()
            (gen_dir / "generation_results.json").write_text("[]", encoding="utf-8")
            (gen_dir / "CLOSING_SIGNAL").write_text(
                "trigger_reason=mature_quorum\ngen_id=0\n",
                encoding="utf-8",
            )
            source_snapshot = fc.result_source_snapshot_at_cutoff(run_dir)
            cutoff = datetime.now(UTC)
            self.assertTrue(
                resume_state.write_boundary_evidence_checkpoint(
                    run_dir,
                    gen_id=0,
                    cutoff=cutoff,
                    evidence_source_snapshot=source_snapshot,
                )
            )

            result_dir = run_dir / "results" / "gen0_peer0_after_crash"
            result_dir.mkdir(parents=True)
            summary = result_dir / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "variant_name": "after_crash",
                        "generation_id": 0,
                        "current_aggregate": {"score": 0.9, "scored_complete": True},
                    }
                ),
                encoding="utf-8",
            )
            old = cutoff.timestamp() - 3600
            os.utime(summary, (old, old))

            recovered = resume_state.read_boundary_evidence_checkpoint(run_dir, 0)
            self.assertIsNotNone(recovered)
            recovered_cutoff, recovered_snapshot = recovered
            stop_audit = generation_boundary._generation_stop_audit(
                SimpleNamespace(run_dir=run_dir),
                gen_id=0,
            )
            [finding] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=0,
                scoring_metric_keys=("score",),
                evidence_cutoff=recovered_cutoff,
                evidence_source_snapshot=recovered_snapshot,
            )
            resume_state.clear_boundary_evidence_checkpoint(run_dir, 0)

            self.assertTrue(finding["metrics"]["late_after_generation_boundary"])
            self.assertTrue(finding["metrics"]["generation_boundary_pending_commit"])
            self.assertFalse(finding["metrics"]["promotion_eligible"])
            self.assertNotIn("boundary_evidence_cutoff_at", stop_audit)
            self.assertNotIn("boundary_evidence_source_snapshot", stop_audit)
            self.assertIsNone(resume_state.read_boundary_evidence_checkpoint(run_dir, 0))
            closing = (gen_dir / "CLOSING_SIGNAL").read_text(encoding="utf-8")
            self.assertIn("trigger_reason=mature_quorum", closing)
            self.assertNotIn("boundary_evidence_cutoff_at", closing)

    def test_boundary_cutoff_persists_late_signal_for_pi_database_read(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import PIAgent

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "gen0_peer0_pi_visible_candidate"
            result_dir.mkdir(parents=True)
            cutoff = datetime.now(UTC)
            time.sleep(0.02)
            (result_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "pi_visible_candidate",
                        "generation_id": 0,
                        "current_aggregate": {"score": 0.7, "scored_complete": True},
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}):
                findings = fc.collect_findings_for_generation(
                    findings_dir=run_dir / "shared_findings",
                    gen_id=0,
                    local_mode=True,
                    result_scoring_metric_keys=("score",),
                    evidence_cutoff=cutoff,
                )
                pi_agent = object.__new__(PIAgent)
                pi_agent.db_path = run_dir / "shared_store.db"
                pi_findings = pi_agent._load_gen_findings(0)

            self.assertEqual(len(findings), 1)
            self.assertEqual(len(pi_findings), 1)
            metrics = pi_findings[0]["metrics"]
            self.assertTrue(metrics["late_after_generation_boundary"])
            self.assertTrue(metrics["generation_boundary_pending_commit"])
            self.assertTrue(metrics["excluded_from_durable_frontier"])
            self.assertFalse(metrics["promotion_eligible"])

    def test_newly_observed_late_result_replays_until_current_boundary_commits(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        for local_mode in (False, True):
            with self.subTest(local_mode=local_mode), tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp)
                boundary = run_dir / "gen_0" / "generation_boundary.json"
                boundary.parent.mkdir(parents=True)
                boundary.write_text(json.dumps({"generation_id": 0}), encoding="utf-8")
                old = 1_800_000_000
                os.utime(boundary, (old, old))
                result_dir = run_dir / "results" / "gen0_peer1_late_candidate"
                result_dir.mkdir(parents=True)
                summary_path = result_dir / "summary.json"
                summary_path.write_text(
                    json.dumps(
                        {
                            "variant_id": "late-candidate",
                            "generation_id": 0,
                            "current_aggregate": {
                                "score": 0.9,
                                "scored_complete": True,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                os.utime(summary_path, (old + 30, old + 30))

                first = fc.collect_findings_for_generation(
                    findings_dir=run_dir / "shared_findings",
                    gen_id=1,
                    local_mode=local_mode,
                    materialize_result_artifacts=True,
                )
                second = fc.collect_findings_for_generation(
                    findings_dir=run_dir / "shared_findings",
                    gen_id=1,
                    local_mode=local_mode,
                    materialize_result_artifacts=True,
                )
                current_boundary = run_dir / "gen_1" / "generation_boundary.json"
                current_boundary.parent.mkdir(exist_ok=True)
                current_boundary.write_text(
                    json.dumps({"generation_id": 1, "status": "complete"}),
                    encoding="utf-8",
                )
                after_commit = fc.collect_findings_for_generation(
                    findings_dir=run_dir / "shared_findings",
                    gen_id=1,
                    local_mode=local_mode,
                    materialize_result_artifacts=True,
                )

            self.assertEqual([finding["variant_name"] for finding in first], ["late-candidate"])
            self.assertEqual(first[0]["generation_id"], 0)
            self.assertTrue(first[0]["metrics"]["late_after_generation_boundary"])
            self.assertTrue(first[0]["metrics"]["excluded_from_durable_frontier"])
            self.assertEqual([finding["id"] for finding in second], [first[0]["id"]])
            self.assertEqual(first[0]["metrics"]["late_observed_generation_id"], 1)
            self.assertEqual(after_commit, [])

    def test_low_confidence_boundary_fallback_is_collectable_as_validation_candidate(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "boundaryless_candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "boundaryless_candidate",
                        "current_aggregate": {"score": 0.81, "scored_complete": True},
                        "n_eval_cells": 3,
                        "scored_complete": True,
                    }
                ),
                encoding="utf-8",
            )

            rows = fc.collect_findings_for_generation(
                findings_dir=run_dir / "shared_findings",
                gen_id=7,
                local_mode=False,
                materialize_result_artifacts=True,
            )

        self.assertEqual([row["variant_name"] for row in rows], ["boundaryless_candidate"])
        self.assertTrue(rows[0]["metrics"]["excluded_from_durable_frontier"])

    def test_task_configured_scoring_keys_do_not_promote_generic_score_only_result(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "generic_score_only"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "generic_score_only",
                        "generation_id": 4,
                        "tier_reached": "T1",
                        "tier_status": "stop_after_T1",
                        "current_aggregate": {
                            "score": 0.99,
                            "scored_cell_count": 29,
                        },
                    }
                ),
                encoding="utf-8",
            )

            materialized = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=4,
                scoring_metric_keys=("task_configured_metric",),
            )

        self.assertEqual(materialized, [])

    def test_protocol_invalid_result_artifact_is_materialized_as_excluded_finding(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "fixed_weight_candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "fixed_weight_candidate",
                        "generation_id": 4,
                        "result_status": "protocol_invalid",
                        "failed_cells": [{"id": "cell_0", "reason": "protocol mismatch"}],
                        "current_aggregate": {
                            "score": 99.0,
                            "scored_cell_count": 29,
                        },
                        "n_eval_cells": 29,
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(run_dir=run_dir, gen_id=5)

        self.assertEqual(finding["variant_name"], "fixed_weight_candidate")
        self.assertEqual(finding["metrics"]["result_status"], "protocol_invalid")
        self.assertFalse(finding["metrics"]["scored_complete"])
        self.assertTrue(finding["metrics"]["excluded_from_durable_frontier"])
        self.assertFalse(finding["metrics"]["promotion_eligible"])
        self.assertEqual(finding["metrics"]["exclusion_reason"], "protocol_integrity_failed")
        self.assertEqual(
            finding["metrics"]["recommended_next_step"],
            "rerun_with_valid_evaluator_protocol",
        )

    def test_protocol_invalid_result_materialization_is_idempotent(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "invalid_candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "invalid_candidate",
                        "generation_id": 2,
                        "result_status": "protocol_invalid",
                        "current_aggregate": {"score": 2.0, "scored_cell_count": 1},
                    }
                ),
                encoding="utf-8",
            )

            [first] = fc._materialize_result_artifacts(run_dir=run_dir, gen_id=2)
            output = next((run_dir / "shared_findings").glob("*.json"))
            before = output.stat().st_mtime_ns
            time.sleep(0.002)
            second = fc._materialize_result_artifacts(run_dir=run_dir, gen_id=2)

            self.assertEqual(second, [])
            self.assertEqual(output.stat().st_mtime_ns, before)
            self.assertEqual(first["metrics"]["exclusion_reason"], "protocol_integrity_failed")

    def test_atomic_finding_write_uses_unique_temporary_files_across_threads(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shared_findings" / "same.json"
            barrier = threading.Barrier(3)
            errors: list[BaseException] = []

            def writer(value: int) -> None:
                try:
                    barrier.wait(timeout=2)
                    fc._atomic_write_json(target, {"value": value})
                except BaseException as exc:  # noqa: BLE001 - preserve thread failures.
                    errors.append(exc)

            threads = [threading.Thread(target=writer, args=(value,)) for value in (1, 2)]
            for thread in threads:
                thread.start()
            barrier.wait(timeout=2)
            for thread in threads:
                thread.join(timeout=2)

            self.assertEqual(errors, [])
            self.assertIn(json.loads(target.read_text(encoding="utf-8"))["value"], {1, 2})
            self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_concurrent_result_materialization_has_one_idempotent_writer(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "candidate",
                        "generation_id": 0,
                        "result_status": "protocol_invalid",
                        "current_aggregate": {"score": 1.0, "scored_cell_count": 1},
                    }
                ),
                encoding="utf-8",
            )
            barrier = threading.Barrier(3)
            results: list[list[dict[str, object]]] = []
            errors: list[BaseException] = []

            def materialize() -> None:
                try:
                    barrier.wait(timeout=2)
                    results.append(fc._materialize_result_artifacts(run_dir=run_dir, gen_id=0))
                except BaseException as exc:  # noqa: BLE001 - preserve thread failures.
                    errors.append(exc)

            threads = [threading.Thread(target=materialize) for _ in range(2)]
            for thread in threads:
                thread.start()
            barrier.wait(timeout=2)
            for thread in threads:
                thread.join(timeout=2)

            self.assertEqual(errors, [])
            self.assertEqual(sorted(len(result) for result in results), [0, 1])
            self.assertEqual(len(list((run_dir / "shared_findings").glob("*.json"))), 1)

    def test_findings_sync_event_filter_ignores_materializer_owned_output(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools.findings_sync import (
            _is_sync_source_event,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            findings_dir = run_dir / "shared_findings"
            results_dir = run_dir / "results"
            findings_dir.mkdir()
            results_dir.mkdir()
            generated = findings_dir / "generated.json"
            generated.write_text(
                json.dumps({"metrics": {"auto_materialized_from_result_artifact": True}}),
                encoding="utf-8",
            )
            external = findings_dir / "peer.json"
            external.write_text(json.dumps({"id": "peer"}), encoding="utf-8")
            result = results_dir / "summary.json"
            result.write_text(json.dumps({"score": 1}), encoding="utf-8")
            non_json = findings_dir / "notes.txt"
            non_json.write_text("notes", encoding="utf-8")
            outside = run_dir / "outside.json"
            outside.write_text("{}", encoding="utf-8")
            malformed = findings_dir / "malformed.json"
            malformed.write_text("{", encoding="utf-8")
            list_payload = findings_dir / "list.json"
            list_payload.write_text("[]", encoding="utf-8")

            self.assertFalse(
                _is_sync_source_event(
                    generated,
                    findings_dir=findings_dir,
                    results_dir=results_dir,
                )
            )
            self.assertTrue(
                _is_sync_source_event(
                    external,
                    findings_dir=findings_dir,
                    results_dir=results_dir,
                )
            )
            self.assertTrue(
                _is_sync_source_event(
                    result,
                    findings_dir=findings_dir,
                    results_dir=results_dir,
                )
            )
            self.assertFalse(
                _is_sync_source_event(
                    non_json,
                    findings_dir=findings_dir,
                    results_dir=results_dir,
                )
            )
            self.assertFalse(
                _is_sync_source_event(
                    outside,
                    findings_dir=findings_dir,
                    results_dir=results_dir,
                )
            )
            self.assertTrue(
                _is_sync_source_event(
                    malformed,
                    findings_dir=findings_dir,
                    results_dir=results_dir,
                )
            )
            self.assertTrue(
                _is_sync_source_event(
                    list_payload,
                    findings_dir=findings_dir,
                    results_dir=results_dir,
                )
            )

    def test_protocol_integrity_passed_false_survives_summary_normalization(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        for field_container in ("top_level", "metrics"):
            with (
                self.subTest(field_container=field_container),
                tempfile.TemporaryDirectory() as tmp,
            ):
                run_dir = Path(tmp)
                result_dir = run_dir / "results" / field_container
                result_dir.mkdir(parents=True)
                summary = {
                    "variant_name": field_container,
                    "generation_id": 4,
                    "current_aggregate": {
                        "score": 99.0,
                        "scored_cell_count": 29,
                        "complete_eval": True,
                    },
                    "n_eval_cells": 29,
                }
                if field_container == "top_level":
                    summary["protocol_integrity_passed"] = False
                else:
                    summary["metrics"] = {"protocol_integrity_passed": False}
                (result_dir / "tiered_eval_summary.json").write_text(
                    json.dumps(summary),
                    encoding="utf-8",
                )

                [finding] = fc._materialize_result_artifacts(run_dir=run_dir, gen_id=5)

                self.assertIs(finding["metrics"]["protocol_integrity_passed"], False)
                self.assertEqual(finding["metrics"]["result_status"], "protocol_invalid")
                self.assertFalse(finding["metrics"]["scored_complete"])
                self.assertTrue(finding["metrics"]["excluded_from_durable_frontier"])

    def test_nested_protocol_integrity_failure_status_excludes_result(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "nested_protocol_failure"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "nested_protocol_failure",
                        "generation_id": 4,
                        "current_aggregate": {
                            "score": 99.0,
                            "scored_cell_count": 29,
                            "complete_eval": True,
                        },
                        "metrics": {"protocol_integrity_status": "failed"},
                        "n_eval_cells": 29,
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(run_dir=run_dir, gen_id=5)

        self.assertEqual(finding["metrics"]["protocol_integrity_status"], "failed")
        self.assertEqual(finding["metrics"]["result_status"], "protocol_invalid")
        self.assertFalse(finding["metrics"]["scored_complete"])
        self.assertTrue(finding["metrics"]["excluded_from_durable_frontier"])

    def test_current_protocol_pass_overrides_stale_nested_failure(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "current_protocol_pass"
            result_dir.mkdir(parents=True)
            (result_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "current_protocol_pass",
                        "generation_id": 4,
                        "protocol_integrity_passed": True,
                        "scored_complete": True,
                        "current_aggregate": {"score": 1.0},
                        "metrics": {
                            "protocol_integrity_passed": False,
                            "protocol_integrity_status": "failed",
                        },
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(run_dir=run_dir, gen_id=4)

        metrics = finding["metrics"]
        self.assertTrue(metrics["protocol_integrity_passed"])
        self.assertEqual(metrics["result_status"], "scored_complete")
        self.assertTrue(metrics["scored_complete"])
        self.assertNotIn("excluded_from_durable_frontier", metrics)
        self.assertNotIn("suspect_protocol", metrics)

    def test_nested_protocol_flags_exclude_scored_result(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        for nested_metrics in (
            {"protocol_integrity_failed": True},
            {"suspect_protocol": True},
        ):
            with self.subTest(nested_metrics=nested_metrics), tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp)
                result_dir = run_dir / "results" / "nested_protocol_flag"
                result_dir.mkdir(parents=True)
                (result_dir / "tiered_eval_summary.json").write_text(
                    json.dumps(
                        {
                            "variant_name": "nested_protocol_flag",
                            "generation_id": 4,
                            "current_aggregate": {"score": 99.0},
                            "metrics": nested_metrics,
                            "n_eval_cells": 29,
                        }
                    ),
                    encoding="utf-8",
                )

                [finding] = fc._materialize_result_artifacts(run_dir=run_dir, gen_id=5)

                self.assertEqual(finding["metrics"]["result_status"], "protocol_invalid")
                self.assertFalse(finding["metrics"]["scored_complete"])
                self.assertTrue(finding["metrics"]["excluded_from_durable_frontier"])

    def test_protocol_violation_count_is_preserved_and_excludes_result(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "count_invalid_candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "count_invalid_candidate",
                        "generation_id": 4,
                        "protocol_integrity_violation_count": 1,
                        "current_aggregate": {"score": 99.0, "scored_cell_count": 29},
                        "n_eval_cells": 29,
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(run_dir=run_dir, gen_id=5)

        self.assertEqual(finding["metrics"]["protocol_integrity_violation_count"], 1)
        self.assertEqual(finding["metrics"]["result_status"], "protocol_invalid")
        self.assertFalse(finding["metrics"]["promotion_eligible"])

    def test_unscored_structured_failures_are_retained_as_validation_signal(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "failed_candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "failed_candidate",
                        "generation_id": 2,
                        "current_aggregate": {},
                        "failed_units": [
                            {"unit_id": "unit_7", "reason": "solver did not converge"}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(run_dir=run_dir, gen_id=3)

        self.assertEqual(finding["metrics"]["result_status"], "partial_cohort")
        self.assertTrue(finding["metrics"]["validation_only_result"])
        self.assertTrue(finding["metrics"]["excluded_from_durable_frontier"])
        self.assertFalse(finding["metrics"]["promotion_eligible"])
        self.assertIn("Structured failure details", finding["content"])

    def test_nested_unscored_failures_are_retained_as_validation_signal(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "nested_failed_candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "nested_failed_candidate",
                        "generation_id": 2,
                        "current_aggregate": {},
                        "metrics": {
                            "failed_units": [
                                {"unit_id": "unit_7", "reason": "solver did not converge"}
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(run_dir=run_dir, gen_id=3)

        self.assertEqual(finding["metrics"]["result_status"], "partial_cohort")
        self.assertTrue(finding["metrics"]["validation_only_result"])
        self.assertTrue(finding["metrics"]["excluded_from_durable_frontier"])
        self.assertIn("Structured failure details", finding["content"])

    def test_late_generation_results_become_canonical_validation_signals(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen_dir = run_dir / "gen_3"
            gen_dir.mkdir(parents=True)
            (gen_dir / "generation_results.json").write_text(
                json.dumps(
                    [
                        {
                            "peer_id": "gen3_peer1",
                            "generation_id": 3,
                            "status": "late_quarantined_protected_job",
                            "late_result_policy": "quarantined_signal",
                            "message": "background evaluation remained active",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            findings = fc._materialize_late_generation_signals(run_dir=run_dir, gen_id=3)

        self.assertEqual(len(findings), 1)
        self.assertTrue(findings[0]["metrics"]["validation_only_result"])
        self.assertFalse(findings[0]["metrics"]["promotion_eligible"])
        self.assertEqual(
            findings[0]["metrics"]["artifact_signal_status"], "late_after_generation_boundary"
        )

    def test_partial_scored_result_artifact_is_materialized_as_validation_candidate(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "partial_candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "partial_candidate",
                        "generation_id": 3,
                        "current_aggregate": {
                            "score": 12.0,
                            "scored_cell_count": 11,
                        },
                        "failed_cells": [{"id": "unit_7"}],
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(run_dir=run_dir, gen_id=4)

        self.assertEqual(finding["variant_name"], "partial_candidate")
        self.assertEqual(finding["metrics"]["result_status"], "partial_cohort")
        self.assertFalse(finding["metrics"]["scored_complete"])
        self.assertEqual(finding["metrics"]["score"], 12.0)
        self.assertTrue(finding["metrics"]["excluded_from_durable_frontier"])
        self.assertFalse(finding["metrics"]["promotion_eligible"])
        self.assertEqual(
            finding["metrics"]["exclusion_reason"],
            "preliminary_or_incomplete_evidence",
        )
        self.assertEqual(
            finding["metrics"]["recommended_next_step"],
            "complete_standard_evaluation_protocol",
        )

    def test_explicit_partial_status_without_failed_cells_is_materialized_for_validation(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "explicit_partial_candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "explicit_partial_candidate",
                        "generation_id": 3,
                        "result_status": "partial_cohort",
                        "current_aggregate": {"score": 12.0, "scored_cell_count": 11},
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(run_dir=run_dir, gen_id=4)

        self.assertEqual(finding["metrics"]["result_status"], "partial_cohort")
        self.assertTrue(finding["metrics"]["excluded_from_durable_frontier"])
        self.assertEqual(finding["metrics"]["score"], 12.0)

    def test_scored_incomplete_result_artifact_is_materialized_for_validation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "incomplete_candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "incomplete_candidate",
                        "generation_id": 3,
                        "result_status": "incomplete",
                        "current_aggregate": {"score": 9.0, "scored_cell_count": 9},
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(run_dir=run_dir, gen_id=4)

        self.assertEqual(finding["variant_name"], "incomplete_candidate")
        self.assertEqual(finding["metrics"]["result_status"], "failed_or_unscored")
        self.assertFalse(finding["metrics"]["scored_complete"])
        self.assertEqual(finding["metrics"]["score"], 9.0)
        self.assertTrue(finding["metrics"]["excluded_from_durable_frontier"])
        self.assertEqual(
            finding["metrics"]["exclusion_reason"],
            "preliminary_or_incomplete_evidence",
        )

    def test_scored_timeout_result_artifact_is_materialized_for_validation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "timeout_candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "timeout_candidate",
                        "generation_id": 3,
                        "result_status": "timeout",
                        "current_aggregate": {"score": 11.0, "scored_cell_count": 11},
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(run_dir=run_dir, gen_id=4)

        self.assertEqual(finding["variant_name"], "timeout_candidate")
        self.assertEqual(finding["metrics"]["result_status"], "failed_or_unscored")
        self.assertFalse(finding["metrics"]["scored_complete"])
        self.assertEqual(finding["metrics"]["score"], 11.0)
        self.assertTrue(finding["metrics"]["excluded_from_durable_frontier"])
        self.assertEqual(
            finding["metrics"]["exclusion_reason"],
            "preliminary_or_incomplete_evidence",
        )

    def test_scored_failed_diagnostic_is_retained_as_validation_signal(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "failed_diagnostic"
            result_dir.mkdir(parents=True)
            (result_dir / "evaluation_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "failed_diagnostic",
                        "generation_id": 3,
                        "result_status": "failed",
                        "current_aggregate": {
                            "score": 7.0,
                            "extra": {
                                "is_negative": True,
                                "failure_mode": "solver_did_not_converge",
                                "diagnostic_role": "failure_analysis",
                                "next_step_intent": "adjust_solver_initialization",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(run_dir=run_dir, gen_id=4)

        metrics = finding["metrics"]
        self.assertEqual(metrics["score"], 7.0)
        self.assertTrue(metrics["is_negative"])
        self.assertEqual(metrics["failure_mode"], "solver_did_not_converge")
        self.assertEqual(metrics["diagnostic_role"], "failure_analysis")
        self.assertEqual(metrics["next_step_intent"], "adjust_solver_initialization")
        self.assertTrue(metrics["validation_only_result"])
        self.assertFalse(metrics["promotion_eligible"])
        self.assertTrue(metrics["excluded_from_durable_frontier"])

    def test_scored_failed_result_with_failure_mode_is_retained_as_negative_signal(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "failed_with_reason"
            result_dir.mkdir(parents=True)
            (result_dir / "evaluation_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "failed_with_reason",
                        "generation_id": 3,
                        "result_status": "failed",
                        "current_aggregate": {
                            "score": 7.0,
                            "extra": {
                                "is_negative": True,
                                "failure_mode": "solver_did_not_converge",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(run_dir=run_dir, gen_id=4)

        metrics = finding["metrics"]
        self.assertEqual(metrics["score"], 7.0)
        self.assertTrue(metrics["is_negative"])
        self.assertEqual(metrics["failure_mode"], "solver_did_not_converge")
        self.assertTrue(metrics["validation_only_result"])
        self.assertFalse(metrics["promotion_eligible"])
        self.assertTrue(metrics["excluded_from_durable_frontier"])

    def test_unscored_actionable_negative_result_is_retained_as_validation_signal(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "unscored_diagnostic"
            result_dir.mkdir(parents=True)
            (result_dir / "evaluation_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "unscored_diagnostic",
                        "generation_id": 3,
                        "current_aggregate": {
                            "extra": {
                                "is_negative": True,
                                "failure_mode": "resource_unavailable",
                                "diagnostic_role": "environment_diagnosis",
                                "next_step_intent": "retry_when_resource_is_available",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(run_dir=run_dir, gen_id=4)

        metrics = finding["metrics"]
        self.assertTrue(metrics["is_negative"])
        self.assertEqual(metrics["failure_mode"], "resource_unavailable")
        self.assertTrue(metrics["validation_only_result"])
        self.assertFalse(metrics["promotion_eligible"])
        self.assertTrue(metrics["excluded_from_durable_frontier"])

    def test_scored_complete_false_result_artifact_is_materialized_for_validation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "boolean_incomplete_candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "boolean_incomplete_candidate",
                        "generation_id": 3,
                        "scored_complete": False,
                        "current_aggregate": {"score": 9.0, "scored_cell_count": 9},
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(run_dir=run_dir, gen_id=4)

        self.assertEqual(finding["variant_name"], "boolean_incomplete_candidate")
        self.assertEqual(finding["metrics"]["result_status"], "not_scored_complete")
        self.assertEqual(finding["metrics"]["score"], 9.0)
        self.assertTrue(finding["metrics"]["excluded_from_durable_frontier"])
        self.assertEqual(
            finding["metrics"]["exclusion_reason"],
            "preliminary_or_incomplete_evidence",
        )

    def test_complete_eval_false_result_artifact_is_materialized_for_validation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "complete_eval_false_candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "complete_eval_false_candidate",
                        "generation_id": 3,
                        "complete_eval": False,
                        "current_aggregate": {"score": 9.0, "scored_cell_count": 9},
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(run_dir=run_dir, gen_id=4)

        metrics = finding["metrics"]
        self.assertEqual(metrics["result_status"], "not_scored_complete")
        self.assertFalse(metrics["scored_complete"])
        self.assertEqual(metrics["score"], 9.0)
        self.assertTrue(metrics["excluded_from_durable_frontier"])
        self.assertEqual(metrics["exclusion_reason"], "preliminary_or_incomplete_evidence")
        self.assertEqual(metrics["recommended_next_step"], "complete_standard_evaluation_protocol")

    def test_all_validation_only_result_artifact_is_not_scored_complete(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "validation_only_candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "validation_only_candidate",
                        "generation_id": 3,
                        "tier_reached": "T1",
                        "tier_status": "stop_after_T1",
                        "current_aggregate": {
                            "quality_score": 12.0,
                            "scored_cell_count": 1,
                        },
                        "all_eval_cells": [
                            {"unit_id": "training_placeholder"},
                            {
                                "validation_only": True,
                                "quality_score": 12.0,
                                "error_rate": 3.0,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=4,
                scoring_metric_keys=("quality_score",),
            )

            finding_path = next((run_dir / "shared_findings").glob("*.json"))
            stale = json.loads(finding_path.read_text(encoding="utf-8"))
            stale["metrics"]["validation_only_result"] = False
            finding_path.write_text(json.dumps(stale), encoding="utf-8")
            [refreshed] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=4,
                scoring_metric_keys=("quality_score",),
            )

        metrics = finding["metrics"]
        self.assertEqual(metrics["result_status"], "not_scored_complete")
        self.assertFalse(metrics["scored_complete"])
        self.assertTrue(metrics["validation_only_result"])
        self.assertTrue(metrics["excluded_from_durable_frontier"])
        self.assertEqual(metrics["exclusion_reason"], "preliminary_or_incomplete_evidence")
        self.assertTrue(refreshed["metrics"]["validation_only_result"])

    def test_result_artifact_preserves_top_level_durability_routing_markers(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "marker_candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "marker_candidate",
                        "generation_id": 3,
                        "tier_reached": "T1",
                        "tier_status": "stop_after_T1",
                        "validation_only_result": True,
                        "durability_scope": "validation_signal_only",
                        "late_result_policy": "quarantined_signal",
                        "artifact_signal_status": "late_after_generation_boundary",
                        "current_aggregate": {
                            "score": 9.0,
                            "effort_ratio": 0.95,
                            "coverage_ratio": 0.95,
                            "scored_complete": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=4,
                scoring_metric_keys=("score",),
            )

        metrics = finding["metrics"]
        self.assertTrue(metrics["validation_only_result"])
        self.assertEqual(metrics["durability_scope"], "validation_signal_only")
        self.assertEqual(metrics["late_result_policy"], "quarantined_signal")
        self.assertEqual(metrics["artifact_signal_status"], "late_after_generation_boundary")

    def test_top_level_routing_markers_override_aggregate_false_defaults(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "marker_candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "marker_candidate",
                        "generation_id": 3,
                        "tier_reached": "T1",
                        "tier_status": "stop_after_T1",
                        "validation_only_result": True,
                        "durability_scope": "validation_signal_only",
                        "late_result_policy": "quarantined_signal",
                        "artifact_signal_status": "late_after_generation_boundary",
                        "current_aggregate": {
                            "score": 9.0,
                            "effort_ratio": 0.95,
                            "coverage_ratio": 0.95,
                            "scored_complete": True,
                            "validation_only_result": False,
                            "durability_scope": False,
                            "late_result_policy": False,
                            "artifact_signal_status": False,
                        },
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=4,
                scoring_metric_keys=("score",),
            )

        metrics = finding["metrics"]
        self.assertTrue(metrics["validation_only_result"])
        self.assertEqual(metrics["durability_scope"], "validation_signal_only")
        self.assertEqual(metrics["late_result_policy"], "quarantined_signal")
        self.assertEqual(metrics["artifact_signal_status"], "late_after_generation_boundary")

    def test_partial_cell_only_scored_result_artifact_is_materialized_for_validation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "cell_only_partial"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "cell_only_partial",
                        "generation_id": 3,
                        "result_status": "partial_cohort",
                        "current_aggregate": {"scored_cell_count": 1},
                        "all_eval_cells": [{"score": 12.0}],
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=4,
                scoring_metric_keys=("score",),
            )

        self.assertEqual(finding["metrics"]["result_status"], "partial_cohort")
        self.assertTrue(finding["metrics"]["excluded_from_durable_frontier"])

    def test_explicit_unknown_maturity_result_is_materialized_for_validation_only(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        summary = {
            "tier_reached": "T1",
            "evidence_stage": "unknown",
            "current_aggregate": {"score": 12.0},
        }
        metrics = fc._result_summary_metrics(summary, scoring_metric_keys=("score",))

        self.assertEqual(metrics["result_status"], "unknown_maturity")
        self.assertNotIn("scored_complete", metrics)

    def test_ratio_maturity_is_not_vetoed_by_inferred_completion_status(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )
        from praxist.plugins.workflow_stages.research_loop.backend import frontier, gems

        summary = {
            "variant_id": "ratio_complete_candidate",
            "generation_id": 0,
            "current_aggregate": {
                "score": 1.25,
                "effort_ratio": 1.0,
                "coverage_ratio": 1.0,
                "frontier_lane": "incubator",
            },
        }
        policy = {
            "min_effort_ratio": 0.75,
            "min_coverage_ratio": 0.80,
            "require_ratio_gate": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "candidate" / "complete_protocol"
            result_dir.mkdir(parents=True)
            (result_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            [finding] = fc._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=0,
                scoring_metric_keys=("score",),
            )
            store = frontier.FrontierStore(
                run_dir / "frontier",
                primary_metric="score",
                metric_direction="maximize",
                maturity_policy=policy,
                frontier_lanes=[
                    {
                        "name": "incubator",
                        "k": 2,
                        "include_lanes": ["incubator"],
                        "axes": [{"metric": "score", "direction": "maximize"}],
                    }
                ],
            )
            promoted = store.promote(0, [finding])

        self.assertNotIn("scored_complete", finding["metrics"])
        self.assertNotIn("excluded_from_durable_frontier", finding["metrics"])
        self.assertEqual([entry["finding_id"] for entry in promoted], [finding["id"]])
        self.assertTrue(
            gems._entry_is_clean_gem_admission_candidate(
                finding,
                maturity_policy=policy,
            )
        )

    def test_rewritten_result_snapshot_clears_preliminary_routing_before_promotion(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )
        from praxist.plugins.workflow_stages.research_loop.backend import frontier
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            findings_ingest,
            local_store,
        )

        policy = {
            "min_effort_ratio": 0.75,
            "min_coverage_ratio": 0.80,
            "require_ratio_gate": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "candidate"
            result_dir.mkdir(parents=True)
            summary_path = result_dir / "summary.json"
            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}):
                summary_path.write_text(
                    json.dumps(
                        {
                            "variant_id": "candidate",
                            "generation_id": 0,
                            "evidence_stage": "preliminary",
                            "current_aggregate": {
                                "score": 1.0,
                                "effort_ratio": 0.25,
                                "coverage_ratio": 0.25,
                                "scored_complete": False,
                                "frontier_lane": "incubator",
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                [preliminary] = fc._materialize_result_artifacts(
                    run_dir=run_dir,
                    gen_id=0,
                    scoring_metric_keys=("score",),
                )
                findings_ingest.ingest_findings_directory(
                    run_dir / "shared_findings",
                    primary_metric="score",
                )
                self.assertTrue(preliminary["metrics"]["excluded_from_durable_frontier"])

                summary_path.write_text(
                    json.dumps(
                        {
                            "variant_id": "candidate",
                            "generation_id": 0,
                            "evidence_stage": "complete",
                            "current_aggregate": {
                                "score": 2.0,
                                "effort_ratio": 1.0,
                                "coverage_ratio": 1.0,
                                "scored_complete": True,
                                "frontier_lane": "incubator",
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                [complete] = fc._materialize_result_artifacts(
                    run_dir=run_dir,
                    gen_id=0,
                    scoring_metric_keys=("score",),
                )
                findings_ingest.ingest_findings_directory(
                    run_dir / "shared_findings",
                    primary_metric="score",
                )
                [row] = local_store.get_findings(generation_id=0)

            self.assertEqual(
                row["metrics"]["source_result_sha256"],
                complete["metrics"]["source_result_sha256"],
            )
            self.assertIs(row["metrics"]["validation_only_result"], False)
            self.assertNotIn("excluded_from_durable_frontier", row["metrics"])
            self.assertNotIn("exclusion_reason", row["metrics"])
            store = frontier.FrontierStore(
                run_dir / "frontier",
                primary_metric="score",
                metric_direction="maximize",
                maturity_policy=policy,
                frontier_lanes=[
                    {
                        "name": "incubator",
                        "k": 2,
                        "include_lanes": ["incubator"],
                        "axes": [{"metric": "score", "direction": "maximize"}],
                    }
                ],
            )
            promoted = store.promote(0, [row])

        self.assertEqual([entry["finding_id"] for entry in promoted], [complete["id"]])

    def test_protocol_invalid_materialized_finding_refreshes_existing_exclusion_schema(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "fixed_weight_candidate"
            result_dir.mkdir(parents=True)
            summary_path = result_dir / "tiered_eval_summary.json"
            summary = {
                "variant_name": "fixed_weight_candidate",
                "generation_id": 4,
                "result_status": "protocol_invalid",
                "current_aggregate": {"score": 99.0, "scored_cell_count": 29},
                "n_eval_cells": 29,
            }
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            normalized = fc.normalized_result_summary(summary, summary_path=summary_path)
            digest = fc._json_digest(normalized)
            findings_dir = run_dir / "shared_findings"
            findings_dir.mkdir()
            stale = findings_dir / "stale_protocol.json"
            stale.write_text(
                json.dumps(
                    {
                        "id": "stale",
                        "variant_name": "fixed_weight_candidate",
                        "metrics": {
                            "auto_materialized_from_result_artifact": True,
                            "source_result_path": str(summary_path.relative_to(run_dir)),
                            "source_result_sha256": digest,
                            "result_status": "scored_complete",
                            "scored_complete": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            [finding] = fc._materialize_result_artifacts(run_dir=run_dir, gen_id=5)

        self.assertEqual(finding["metrics"]["result_status"], "protocol_invalid")
        self.assertTrue(finding["metrics"]["excluded_from_durable_frontier"])
        self.assertFalse(stale.exists())

    def test_result_metric_status_edges_are_preserved(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        base = {
            "tier_reached": "T2",
            "tier_status": "scored_complete",
            "current_aggregate": {"score": 1.0},
        }
        opaque_stage_pass = {
            "tier_reached": "preview",
            "tier_status": "passed_preview",
            "current_aggregate": {"score": 1.0},
        }
        self.assertEqual(
            fc._result_summary_metrics(opaque_stage_pass)["result_status"],
            "unknown_maturity",
        )
        partial = dict(base, failed_cells=[{"id": 1}])
        self.assertEqual(fc._result_summary_metrics(partial)["result_status"], "partial_cohort")
        nested_partial = dict(base, metrics={"failed_units": [{"unit_id": "unit_1"}]})
        self.assertEqual(
            fc._result_summary_metrics(nested_partial)["result_status"],
            "partial_cohort",
        )

        scout = dict(base, result_status="cheap_probe")
        self.assertEqual(fc._result_summary_metrics(scout)["result_status"], "scout_or_smoke")

        capped = dict(base, final_status="capped_at_epoch")
        self.assertEqual(fc._result_summary_metrics(capped)["result_status"], "capped")
        uncapped = dict(base, variant_name="uncapped_full", final_status="uncapped_full")
        self.assertEqual(fc._result_summary_metrics(uncapped)["result_status"], "scored_complete")

        not_scored = dict(base, current_aggregate={"scored_complete": "false"})
        self.assertEqual(
            fc._result_summary_metrics(not_scored)["result_status"],
            "not_scored_complete",
        )

        incomplete_eval = dict(base, complete_eval=False)
        self.assertEqual(
            fc._result_summary_metrics(incomplete_eval)["result_status"],
            "not_scored_complete",
        )
        nested_incomplete_eval = dict(base, metrics={"complete_eval": False})
        self.assertEqual(
            fc._result_summary_metrics(nested_incomplete_eval)["result_status"],
            "not_scored_complete",
        )
        nested_not_scored = dict(base, metrics={"scored_complete": False})
        self.assertEqual(
            fc._result_summary_metrics(nested_not_scored)["result_status"],
            "not_scored_complete",
        )

        failed = dict(base, final_status="timeout failure")
        self.assertEqual(fc._result_summary_metrics(failed)["result_status"], "failed_or_unscored")

        repair_failure = dict(base, final_status="failed_hard_constraints")
        repair_metrics = fc._result_summary_metrics(repair_failure)
        self.assertEqual(repair_metrics["result_status"], "scored_complete")
        self.assertTrue(repair_metrics["scored_complete"])

        protocol_invalid = dict(
            base,
            protocol_integrity_status="protocol-invalid",
            suspect_fixed_weight_eval=True,
            current_aggregate={"score": 99.0},
        )
        protocol_metrics = fc._result_summary_metrics(protocol_invalid)
        self.assertEqual(protocol_metrics["result_status"], "protocol_invalid")
        self.assertFalse(protocol_metrics["scored_complete"])
        self.assertTrue(protocol_metrics["unscored_artifact"])
        self.assertTrue(protocol_metrics["suspect_protocol"])
        self.assertNotIn("suspect_fixed_weight_eval", protocol_metrics)

        unscored = {"tier_reached": "T1", "current_aggregate": {}}
        metrics = fc._result_summary_metrics(unscored)
        self.assertEqual(metrics["result_status"], "unscored_artifact")
        self.assertEqual(metrics["n_hard_constraint_violations"], 0)

        blocked = dict(
            base,
            current_aggregate={
                "score": 1.0,
                "promotion_blocking_hard_constraint_violations": ["risk", "leak"],
            },
        )
        self.assertEqual(fc._result_summary_metrics(blocked)["n_hard_constraint_violations"], 2)

    def test_generation_inference_fallbacks_scan_canonical_findings(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection as fc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen_dir = run_dir / "gen_7" / "shared_findings"
            gen_dir.mkdir(parents=True)
            (gen_dir / "bad.json").write_text("{bad", encoding="utf-8")
            (gen_dir / "parent.json").write_text(
                json.dumps(
                    {
                        "variant_name": "parent",
                        "content": "completed scored strongest result for child_a",
                        "metrics": {"child_variants": ["child_b"]},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                fc._infer_result_generation(
                    run_dir=run_dir,
                    summary={},
                    variant="child_a",
                    boundary_gen_id=0,
                ),
                (0, "boundary_fallback"),
            )
            self.assertEqual(
                fc._infer_result_generation(
                    run_dir=run_dir,
                    summary={},
                    variant="child_b",
                    boundary_gen_id=0,
                ),
                (7, "generation_local_finding_reference"),
            )

            root = run_dir / "shared_findings"
            root.mkdir()
            (root / "auto.json").write_text(
                json.dumps(
                    {
                        "variant_name": "root_child",
                        "generation_id": 9,
                        "metrics": {"auto_materialized_from_result_artifact": True},
                    }
                ),
                encoding="utf-8",
            )
            (root / "manual.json").write_text(
                json.dumps(
                    {
                        "variant_name": "root_child",
                        "generation_id": "gen_8",
                        "metrics": {},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                fc._infer_result_generation(
                    run_dir=run_dir,
                    summary={},
                    variant="root_child",
                    boundary_gen_id=0,
                ),
                (8, "root_finding_reference"),
            )
            self.assertEqual(
                fc._infer_result_generation(
                    run_dir=run_dir,
                    summary={},
                    variant="missing",
                    boundary_gen_id=2,
                ),
                (2, "boundary_fallback"),
            )


if __name__ == "__main__":
    unittest.main()
