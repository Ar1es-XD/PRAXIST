from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

AIST_RESULT_SCORING_KEYS = (
    "future_fitness",
    "mean_test_taskscore",
    "mean_test_accuracy",
    "test_accuracy",
)


class ResearchLoopSupportContractsTest(unittest.TestCase):
    def test_prompt_context_does_not_replay_disabled_constructive_mix(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import prompt_context

        class FakeFrontier:
            def get_summary(self):
                return []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen0 = run_dir / "gen_0"
            gen0.mkdir(parents=True)
            (gen0 / "generation_boundary.json").write_text(
                json.dumps(
                    {
                        "peer_mix": {
                            "mature_constructive_ratio": 0.25,
                            "target_constructive_ratio": 0.75,
                        }
                    }
                ),
                encoding="utf-8",
            )
            task_spec = SimpleNamespace(
                agent=SimpleNamespace(runtime="agent_runtime:claude_sdk"),
                evaluation=SimpleNamespace(
                    diversity_dimensions=[],
                    constructive_peer_mix_enabled=False,
                ),
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
                    workspace=root,
                    run_dir=run_dir,
                    results_dir=root / "results",
                    variants_dir=root / "variants",
                    findings_dir=root / "findings",
                    frontier=FakeFrontier(),
                    local_mode=True,
                    gen_id=1,
                    peer_index=0,
                    cohort_size=1,
                    strategy="explore",
                )

        self.assertNotIn("peer_mix", context["research_loop_control"])

    def test_pi_directed_variant_hint_uses_role_aligned_diversity_wording(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import prompt_strategy

        class FakeFrontier:
            def get_summary(self):
                return [
                    {
                        "finding_id": "f1",
                        "variant_name": "anchor",
                        "metric_value": 1.0,
                    }
                ]

        hint = prompt_strategy._generate_variant_hint(
            gen_id=2,
            peer_index=0,
            strategy="pi_directed",
            frontier=FakeFrontier(),
            frontier_summary=None,
            cohort_size=1,
            diversity_dimensions=[{"name": "mechanism", "description": "mechanism axis"}],
        )

        self.assertIn("PI-directed phase", hint)
        self.assertIn("design_dimensions", hint)
        self.assertNotIn("explore phase", hint)
        self.assertNotIn("Exploration Mandate", hint)

    def test_prompt_context_filters_future_frontier_entries(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import prompt_context

        class FakeFrontier:
            def get_summary(self):
                return [
                    {"variant_name": "current_candidate", "generation_id": 0},
                    {"variant_name": "unknown_generation_candidate"},
                    {"variant_name": "future_candidate", "generation_id": 3},
                ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_spec = SimpleNamespace(
                agent=SimpleNamespace(runtime="agent_runtime:claude_sdk"),
                evaluation=SimpleNamespace(diversity_dimensions=[]),
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
                    workspace=root,
                    run_dir=root / "run",
                    results_dir=root / "results",
                    variants_dir=root / "variants",
                    findings_dir=root / "findings",
                    frontier=FakeFrontier(),
                    local_mode=True,
                    gen_id=1,
                    peer_index=0,
                    cohort_size=1,
                    strategy="explore",
                )

        names = [entry["variant_name"] for entry in context["frontier_summary"]]
        self.assertEqual(names, ["current_candidate"])
        self.assertNotIn("future_candidate", context["variant_hint"])
        self.assertNotIn("unknown_generation_candidate", context["variant_hint"])

    def test_stop_audit_does_not_mark_safety_cap_as_sufficient_without_quorum(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import generation_boundary

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen_dir = run_dir / "gen_0"
            gen_dir.mkdir()
            (gen_dir / "STOP_SIGNAL").write_text(
                "trigger_reason=safety_cap\nrequired_mature_result_peers=0\nmature_result_peers=0\n",
                encoding="utf-8",
            )

            audit = generation_boundary._generation_stop_audit(
                SimpleNamespace(run_dir=run_dir),
                gen_id=0,
            )

        self.assertFalse(audit["evidence_sufficient"])

    def test_stop_audit_preserves_postgen_mature_quorum_sufficiency(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import generation_boundary

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen_dir = run_dir / "gen_0"
            gen_dir.mkdir()
            (gen_dir / "STOP_SIGNAL_POSTGEN").write_text(
                "trigger_reason=mature_quorum\n"
                "required_mature_result_peers=2\n"
                "mature_result_peers=2\n",
                encoding="utf-8",
            )

            audit = generation_boundary._generation_stop_audit(
                SimpleNamespace(run_dir=run_dir),
                gen_id=0,
            )

        self.assertTrue(audit["evidence_sufficient"])
        self.assertEqual(audit["signal_file"], "gen_0/STOP_SIGNAL_POSTGEN")

    def test_evidence_maturity_treats_string_false_as_false(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
            normalize_maturity_policy,
        )

        self.assertFalse(
            normalize_maturity_policy({"require_ratio_gate": "false"})["require_ratio_gate"]
        )
        self.assertTrue(
            normalize_maturity_policy({"require_ratio_gate": "true"})["require_ratio_gate"]
        )

    def test_evidence_maturity_never_divides_unlike_effort_units(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
            evidence_maturity_snapshot,
        )

        snapshot = evidence_maturity_snapshot(
            {
                "actual_steps": 1000,
                "actual_epochs": 1,
                "planned_epochs": 10,
                "completed_eval_units": 8,
                "total_eval_units": 10,
            }
        )

        self.assertEqual(snapshot["effort_ratio"], 0.1)
        self.assertFalse(snapshot["mature_enough"])

    def test_evidence_maturity_handles_ratio_maps_and_numeric_flags(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
            evidence_maturity_snapshot,
            normalize_maturity_policy,
        )

        snapshot = evidence_maturity_snapshot(
            {
                "effort_ratio": -1,
                "coverage_ratios": {"condition_a": 0.8, "condition_b": 0.6},
            }
        )

        self.assertEqual(snapshot["effort_ratio"], 0.0)
        self.assertEqual(snapshot["coverage_ratio"], 0.6)
        self.assertTrue(normalize_maturity_policy({"require_ratio_gate": 1})["require_ratio_gate"])

    def test_evidence_maturity_explicit_incomplete_overrides_high_ratios(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
            evidence_maturity_snapshot,
        )

        snapshot = evidence_maturity_snapshot(
            {
                "effort_ratio": 1.0,
                "coverage_ratio": 1.0,
                "complete_eval": False,
            }
        )

        self.assertFalse(snapshot["mature_enough"])
        self.assertEqual(snapshot["maturity_basis"], "explicit_completion_flag")

    def test_evidence_maturity_protocol_failure_overrides_stage_and_ratios(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            frontier,
            gems,
            status_snapshot,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
            durable_promotion_exclusion,
            evidence_maturity_snapshot,
            protocol_integrity_failed,
        )

        policy = {
            "complete_stage_labels": ["approved_reduced"],
            "require_ratio_gate": False,
        }
        failed = {
            "evidence_stage": "approved_reduced",
            "scored_complete": True,
            "effort_ratio": 1.0,
            "coverage_ratio": 1.0,
            "protocol_integrity_passed": False,
        }
        snapshot = evidence_maturity_snapshot(failed, policy)

        self.assertTrue(protocol_integrity_failed(failed))
        self.assertFalse(snapshot["mature_enough"])
        self.assertEqual(snapshot["maturity_basis"], "protocol_integrity")

        current_pass = {
            "protocol_integrity_passed": True,
            "promotion_eligible": True,
            "validation_only": False,
            "effort_ratio": 1.0,
            "coverage_ratio": 1.0,
            "metrics": {
                "protocol_integrity_passed": False,
                "promotion_eligible": False,
                "validation_only": True,
            },
        }
        self.assertFalse(protocol_integrity_failed(current_pass))
        self.assertIsNone(durable_promotion_exclusion(current_pass))
        self.assertFalse(frontier._candidate_protocol_integrity_failed(current_pass))
        self.assertFalse(gems._entry_has_gem_integrity_rejection_marker(current_pass))
        self.assertFalse(
            status_snapshot._result_view_is_restricted(
                current_pass,
                {"require_ratio_gate": True},
            )
        )

    def test_evidence_maturity_current_facts_override_stale_nested_copies(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
            evidence_maturity_snapshot,
            has_explicit_false_completion,
        )

        policy = {
            "complete_stage_labels": ["complete"],
            "preliminary_stage_labels": ["preliminary"],
        }
        candidate = {
            "evidence_stage": "complete",
            "scored_complete": True,
            "metrics": {
                "evidence_stage": "preliminary",
                "scored_complete": False,
            },
        }

        snapshot = evidence_maturity_snapshot(candidate, policy)

        self.assertTrue(snapshot["mature_enough"])
        self.assertEqual(snapshot["maturity_basis"], "task_configured_stage")
        self.assertFalse(has_explicit_false_completion(candidate))
        ratio_snapshot = evidence_maturity_snapshot(
            {
                "effort_ratio": 1.0,
                "coverage_ratio": 1.0,
                "metrics": {
                    "effort_ratio": 0.2,
                    "coverage_ratio": 0.2,
                },
            }
        )
        self.assertEqual(ratio_snapshot["effort_ratio"], 1.0)
        self.assertEqual(ratio_snapshot["coverage_ratio"], 1.0)
        self.assertTrue(ratio_snapshot["mature_enough"])
        computed_snapshot = evidence_maturity_snapshot(
            {
                "actual_steps": 9,
                "planned_steps": 10,
                "completed_eval_units": 9,
                "total_eval_units": 10,
                "metrics": {
                    "effort_ratio": 0.1,
                    "coverage_ratio": 0.1,
                },
            }
        )
        self.assertEqual(computed_snapshot["effort_ratio"], 0.9)
        self.assertEqual(computed_snapshot["coverage_ratio"], 0.9)
        self.assertTrue(computed_snapshot["mature_enough"])
        self.assertFalse(
            evidence_maturity_snapshot(
                {"metrics": {"evidence_stage": "preliminary"}},
                policy,
            )["mature_enough"]
        )

    def test_prompt_context_exposes_non_durable_validation_signals(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import prompt_context
        from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
            CANONICAL_STATE,
            FAILED,
            attach_artifact_semantics,
        )

        class FakeFrontier:
            def get_summary(self):
                return []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir(parents=True)
            manifest = attach_artifact_semantics(
                {
                    "generations": {
                        "0": [
                            {
                                "generation_id": 0,
                                "variant_name": "failed_signal",
                                "metric_value": 2.5,
                                "evidence_stage": "failed_probe",
                                "mechanism_family": "repairable_mechanism",
                            }
                        ]
                    }
                },
                role=CANONICAL_STATE,
                status=FAILED,
                stage="frontier_manifest",
                runtime_fact_source=False,
            )
            (frontier_dir / "frontier_manifest.json").write_text(json.dumps(manifest))
            task_spec = SimpleNamespace(
                agent=SimpleNamespace(runtime="agent_runtime:claude_sdk"),
                evaluation=SimpleNamespace(diversity_dimensions=[]),
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
                    workspace=root,
                    run_dir=run_dir,
                    results_dir=root / "results",
                    variants_dir=root / "variants",
                    findings_dir=root / "findings",
                    frontier=FakeFrontier(),
                    local_mode=True,
                    gen_id=1,
                    peer_index=0,
                    cohort_size=1,
                    strategy="explore",
                )

        self.assertEqual(context["frontier_summary"], [])
        self.assertEqual(context["validation_candidates"][0]["variant_name"], "failed_signal")
        self.assertEqual(context["validation_candidates"][0]["artifact_signal_status"], FAILED)
        self.assertEqual(
            context["validation_candidates"][0]["durability_scope"],
            "validation_signal_only",
        )
        self.assertFalse(context["validation_candidates_meta"]["truncated"])

    def test_prompt_context_exposes_generic_incubator_signals(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import prompt_context

        class FakeFrontier:
            def get_summary(self):
                return []

            def get_manifest(self):
                return {
                    "frontier_lanes": [
                        {"name": "candidate_library", "parent_eligible": True},
                        {"name": "incubator", "parent_eligible": True},
                        {"name": "controller_improvements", "parent_eligible": True},
                        {"name": "process_optimizer", "parent_eligible": True},
                        {"name": "control", "parent_eligible": True},
                        {"name": "negative_control", "parent_eligible": False},
                    ],
                    "lane_frontiers": {
                        "candidate_library": [
                            {
                                "generation_id": 0,
                                "variant_name": "durable_candidate",
                                "metric_name": "score",
                                "metric_value": 4.0,
                                "frontier_lane": "candidate_library",
                                "effort_ratio": 0.8,
                                "coverage_ratio": 0.9,
                                "mature_enough": True,
                                "metrics": {"score": 4.0},
                            },
                            {
                                "generation_id": 2,
                                "variant_name": "future_candidate",
                                "metric_name": "score",
                                "metric_value": 9.0,
                                "metrics": {"score": 9.0},
                            },
                            {
                                "variant_name": "unknown_generation_candidate",
                                "metric_name": "score",
                                "metric_value": 8.0,
                                "metrics": {"score": 8.0},
                            },
                        ],
                        "incubator": [
                            {
                                "generation_id": 0,
                                "variant_name": "incubator_candidate",
                                "metric_name": "score",
                                "metric_value": 3.0,
                                "frontier_lane": "incubator",
                                "scored_complete": True,
                                "metrics": {"score": 3.0},
                            },
                            {
                                "generation_id": 0,
                                "variant_name": "ratio_failed_candidate",
                                "frontier_lane": "incubator",
                                "scored_complete": True,
                                "effort_ratio": 0.5,
                                "coverage_ratio": 1.0,
                                "metrics": {"score": 30.0},
                            },
                            {
                                "generation_id": 0,
                                "variant_name": "entry_parent_ineligible",
                                "frontier_lane": "incubator",
                                "parent_eligible": False,
                                "scored_complete": True,
                                "metrics": {"score": 20.0},
                            },
                        ],
                        "controller_improvements": [
                            {
                                "generation_id": 0,
                                "variant_name": "valid_parent_lane",
                                "metric_name": "score",
                                "metric_value": 2.0,
                                "frontier_lane": "controller_improvements",
                                "parent_eligible": True,
                                "scored_complete": True,
                                "metrics": {"score": 2.0},
                            }
                        ],
                        "process_optimizer": [
                            {
                                "generation_id": 0,
                                "variant_name": "process_parent_lane",
                                "metric_name": "score",
                                "metric_value": 1.5,
                                "frontier_lane": "process_optimizer",
                                "parent_eligible": True,
                                "scored_complete": True,
                                "metrics": {"score": 1.5},
                            }
                        ],
                        "control": [
                            {
                                "generation_id": 0,
                                "variant_name": "control_domain_lane",
                                "metric_name": "score",
                                "metric_value": 1.2,
                                "frontier_lane": "control",
                                "parent_eligible": True,
                                "scored_complete": True,
                                "metrics": {"score": 1.2},
                            }
                        ],
                        "negative_control": [
                            {
                                "generation_id": 0,
                                "variant_name": "negative_control_lane",
                                "metric_name": "score",
                                "metric_value": 0.1,
                                "frontier_lane": "negative_control",
                                "parent_eligible": False,
                                "scored_complete": True,
                                "metrics": {"score": 0.1},
                            }
                        ],
                    },
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_spec = SimpleNamespace(
                agent=SimpleNamespace(runtime="agent_runtime:claude_sdk"),
                evaluation=SimpleNamespace(
                    primary_metric="score",
                    direction="maximize",
                    aux_metrics=[],
                    anchor_metrics=[],
                    diversity_dimensions=[],
                ),
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
                    workspace=root,
                    run_dir=root / "run",
                    results_dir=root / "results",
                    variants_dir=root / "variants",
                    findings_dir=root / "findings",
                    frontier=FakeFrontier(),
                    local_mode=True,
                    gen_id=1,
                    peer_index=0,
                    cohort_size=1,
                    strategy="explore",
                )

        self.assertIn("incubator_top_k", context)
        self.assertEqual(
            [entry["variant_name"] for entry in context["incubator_top_k"]],
            [
                "durable_candidate",
                "incubator_candidate",
                "valid_parent_lane",
                "process_parent_lane",
                "control_domain_lane",
            ],
        )
        self.assertEqual(context["incubator_top_k"][0]["metric_name"], "score")
        self.assertEqual(context["incubator_top_k"][0]["effort_ratio"], 0.8)
        self.assertEqual(context["incubator_top_k"][0]["coverage_ratio"], 0.9)
        self.assertNotIn(
            "future_candidate",
            [entry["variant_name"] for entry in context["incubator_top_k"]],
        )
        self.assertNotIn(
            "unknown_generation_candidate",
            [entry["variant_name"] for entry in context["incubator_top_k"]],
        )
        self.assertNotIn(
            "ratio_failed_candidate",
            [entry["variant_name"] for entry in context["incubator_top_k"]],
        )
        self.assertNotIn(
            "entry_parent_ineligible",
            [entry["variant_name"] for entry in context["incubator_top_k"]],
        )
        self.assertNotIn(
            "valid_parent_lane",
            [entry["variant_name"] for entry in context["diagnostic_control_top_k"]],
        )
        self.assertNotIn(
            "process_parent_lane",
            [entry["variant_name"] for entry in context["diagnostic_control_top_k"]],
        )
        self.assertNotIn(
            "control_domain_lane",
            [entry["variant_name"] for entry in context["diagnostic_control_top_k"]],
        )
        self.assertIn(
            "negative_control_lane",
            [entry["variant_name"] for entry in context["diagnostic_control_top_k"]],
        )
        self.assertEqual(
            context["strong_parent_visibility_policy"]["incubator"],
            "parentable_when_task_protocol_allows",
        )

    def test_prompt_context_compacts_frontier_metrics_for_prompt(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import prompt_context

        class FakeFrontier:
            def get_summary(self):
                metrics = {f"unused_metric_{i}": i for i in range(50)}
                metrics.update(
                    {
                        "future_fitness": 1.5,
                        "validation_2026_active_alpha_pct": 7.25,
                        "mean_return_pct": 11.5,
                        "mean_mdd_pct": 8.75,
                        "promotion_mean_mdd_pct": 7.5,
                        "mean_effective_n": 18.0,
                        "mechanism_family": "objective",
                    }
                )
                return [
                    {
                        "variant_name": "compact_me",
                        "generation_id": 0,
                        "rank": 1,
                        "metric_value": 1.5,
                        "metrics": metrics,
                    }
                ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_spec = SimpleNamespace(
                evaluation=SimpleNamespace(
                    primary_metric="future_fitness",
                    diversity_dimensions=[],
                    must_explore_axes=[],
                ),
                gems=SimpleNamespace(
                    primary_metric_keys=[],
                    secondary_metric_keys=["mean_return_pct", "mean_effective_n"],
                    lower_tail_metric_keys=[],
                    validation_metric_keys=["validation_2026_active_alpha_pct"],
                    cost_metric_keys=["mean_mdd_pct", "promotion_mean_mdd_pct"],
                    result_cell_metric_derivations=[],
                    result_metric_aliases={},
                ),
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
                    workspace=root,
                    run_dir=root / "run",
                    results_dir=root / "results",
                    variants_dir=root / "variants",
                    findings_dir=root / "findings",
                    frontier=FakeFrontier(),
                    local_mode=True,
                    gen_id=1,
                    peer_index=0,
                    cohort_size=1,
                    strategy="explore",
                )

        metrics = context["frontier_summary"][0]["metrics"]
        self.assertEqual(metrics["future_fitness"], 1.5)
        self.assertEqual(metrics["validation_2026_active_alpha_pct"], 7.25)
        self.assertEqual(metrics["mean_return_pct"], 11.5)
        self.assertEqual(metrics["mean_mdd_pct"], 8.75)
        self.assertEqual(metrics["promotion_mean_mdd_pct"], 7.5)
        self.assertEqual(metrics["mean_effective_n"], 18.0)
        self.assertEqual(metrics["mechanism_family"], "objective")
        self.assertNotIn("unused_metric_0", metrics)
        self.assertGreater(metrics["_omitted_metric_count"], 0)

    def test_prompt_context_preserves_task_defined_frontier_metrics(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import prompt_context

        class FakeFrontier:
            def get_summary(self):
                return [
                    {
                        "variant_name": "anchor_variant",
                        "generation_id": 0,
                        "rank": 1,
                        "metric_value": 1.5,
                        "promoted_for_anchor": "sharpness_top_eigen",
                        "anchor_metric_value": 0.03,
                        "metrics": {
                            "future_fitness": 1.5,
                            "mean_train_test_gap": 0.12,
                            "sharpness_top_eigen": 0.03,
                            "compute_overhead_ratio": 1.08,
                            "lane_required_metric": 0.9,
                            "lane_required_flag": True,
                            "lane_min_metric": 0.3,
                            "lane_max_metric": 0.7,
                            "lane_optional_score": 0.7,
                            "test_accuracy_cifar100": 0.71,
                            "test_accuracy_cifar10": 0.94,
                            "test_accuracy_tiny_imagenet": 0.59,
                            "train_test_gap_cifar100": 0.11,
                            "train_test_gap_cifar10": 0.04,
                            "train_test_gap_tiny_imagenet": 0.22,
                            "unused_metric": "drop",
                        },
                    }
                ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_spec = SimpleNamespace(
                evaluation=SimpleNamespace(
                    primary_metric="future_fitness",
                    aux_metrics=[
                        "mean_train_test_gap",
                        "test_accuracy_cifar100",
                        "test_accuracy_cifar10",
                        "test_accuracy_tiny_imagenet",
                        "train_test_gap_cifar100",
                        "train_test_gap_cifar10",
                        "train_test_gap_tiny_imagenet",
                    ],
                    anchor_metrics=[("sharpness_top_eigen", "minimize")],
                    frontier_lanes=[
                        {
                            "axes": [("compute_overhead_ratio", "minimize")],
                            "optional_axes": [("lane_optional_score", "maximize")],
                            "require_metrics": ["lane_required_metric"],
                            "require_truthy_metrics": ["lane_required_flag"],
                            "min_metrics": {"lane_min_metric": 0.25},
                            "max_metrics": {"lane_max_metric": 0.75},
                        }
                    ],
                    diversity_dimensions=[],
                    must_explore_axes=[],
                ),
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
                    workspace=root,
                    run_dir=root / "run",
                    results_dir=root / "results",
                    variants_dir=root / "variants",
                    findings_dir=root / "findings",
                    frontier=FakeFrontier(),
                    local_mode=True,
                    gen_id=1,
                    peer_index=0,
                    cohort_size=1,
                    strategy="explore",
                )

        entry = context["frontier_summary"][0]
        metrics = entry["metrics"]
        self.assertEqual(entry["promoted_for_anchor"], "sharpness_top_eigen")
        self.assertEqual(entry["anchor_metric_value"], 0.03)
        self.assertEqual(metrics["mean_train_test_gap"], 0.12)
        self.assertEqual(metrics["sharpness_top_eigen"], 0.03)
        self.assertEqual(metrics["compute_overhead_ratio"], 1.08)
        self.assertEqual(metrics["lane_required_metric"], 0.9)
        self.assertTrue(metrics["lane_required_flag"])
        self.assertEqual(metrics["lane_min_metric"], 0.3)
        self.assertEqual(metrics["lane_max_metric"], 0.7)
        self.assertEqual(metrics["lane_optional_score"], 0.7)
        self.assertEqual(metrics["test_accuracy_cifar100"], 0.71)
        self.assertEqual(metrics["test_accuracy_cifar10"], 0.94)
        self.assertEqual(metrics["test_accuracy_tiny_imagenet"], 0.59)
        self.assertEqual(metrics["train_test_gap_cifar100"], 0.11)
        self.assertEqual(metrics["train_test_gap_cifar10"], 0.04)
        self.assertEqual(metrics["train_test_gap_tiny_imagenet"], 0.22)
        self.assertNotIn("unused_metric", metrics)

    def test_prompt_context_preserves_compact_dig_provenance(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import prompt_context

        class FakeFrontier:
            def get_summary(self):
                return [
                    {
                        "variant_name": "dig_materialized_variant",
                        "generation_id": 0,
                        "rank": 1,
                        "metric_value": 1.5,
                        "metrics": {
                            "future_fitness": 1.5,
                            "dig_selected_contract_path": (
                                "gen_0/peers/gen0_peer0/dig/selected_contract.yaml"
                            ),
                            "dig_provenance": {
                                "selected_contract_path": (
                                    "gen_0/peers/gen0_peer0/dig/selected_contract.yaml"
                                ),
                                "selected_candidate_id": "local_candidate",
                                "final_selected_candidate_id": "final_candidate",
                                "selected_contract_source": "cohort_qd_override",
                                "semantic_family": "temporal_gating",
                                "parent_lineage": "dual_critic_repair",
                                "novelty_axis": "temperature_gate",
                                "diversity_cell": {
                                    "mechanism_family": "gating",
                                    "intervention_surface": "ppo_loss",
                                },
                                "canonical_labels": {
                                    "canonical_semantic_family": "temporal_gating",
                                    "canonical_novelty_axis": "temperature_gate",
                                },
                                "expected_metric_signature": {
                                    "primary": "long diagnostic block should not re-enter prompt"
                                },
                            },
                            "unused_metric": "drop",
                        },
                    }
                ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_spec = SimpleNamespace(
                evaluation=SimpleNamespace(
                    primary_metric="future_fitness",
                    diversity_dimensions=[],
                    must_explore_axes=[],
                ),
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
                    workspace=root,
                    run_dir=root / "run",
                    results_dir=root / "results",
                    variants_dir=root / "variants",
                    findings_dir=root / "findings",
                    frontier=FakeFrontier(),
                    local_mode=True,
                    gen_id=1,
                    peer_index=0,
                    cohort_size=1,
                    strategy="explore",
                )

        metrics = context["frontier_summary"][0]["metrics"]
        provenance = metrics["dig_provenance"]
        self.assertEqual(
            metrics["dig_selected_contract_path"],
            "gen_0/peers/gen0_peer0/dig/selected_contract.yaml",
        )
        self.assertEqual(provenance["selected_candidate_id"], "local_candidate")
        self.assertEqual(provenance["final_selected_candidate_id"], "final_candidate")
        self.assertEqual(provenance["selected_contract_source"], "cohort_qd_override")
        self.assertEqual(provenance["semantic_family"], "temporal_gating")
        self.assertEqual(provenance["parent_lineage"], "dual_critic_repair")
        self.assertEqual(provenance["novelty_axis"], "temperature_gate")
        self.assertEqual(provenance["diversity_cell"]["mechanism_family"], "gating")
        self.assertEqual(
            provenance["canonical_labels"]["canonical_novelty_axis"],
            "temperature_gate",
        )
        self.assertNotIn("expected_metric_signature", provenance)
        self.assertNotIn("unused_metric", metrics)

    def test_prompt_context_slices_research_agenda_for_current_peer(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import prompt_context

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_spec = SimpleNamespace(
                evaluation=SimpleNamespace(
                    primary_metric="future_fitness",
                    diversity_dimensions=[],
                    must_explore_axes=[],
                ),
            )
            frontier = SimpleNamespace(get_summary=lambda: [])
            agenda = {
                "generation": 2,
                "synthesized_from_gen": 1,
                "panel_summary": {"large": "x" * 5000},
                "consensus_actions": [{"large": "y" * 5000}],
                "peer_contracts": {
                    "gen2_peer0": {
                        "role": "exploit",
                        "target_hypothesis": "current target",
                        "success_signal": "current success",
                        "forbidden_actions": [f"do not copy {i}" for i in range(12)],
                        "source_lane": "alpha_incubator",
                        "target_lane": "confirmed_alpha",
                        "coverage_check": "current coverage",
                        "required_controls": [f"control {i}" for i in range(12)],
                        "mechanism_hypothesis_deliverable": "deliver mechanism note",
                        "custom_binding_field": "must survive",
                    },
                    "gen2_peer1": {
                        "role": "bridge",
                        "target_hypothesis": "other target",
                        "success_signal": "other success",
                        "coverage_check": "other coverage",
                        "source_lane": "diagnostic_control",
                        "target_lane": "alpha_incubator",
                        "private_kb_source": "drop me",
                    },
                },
                "cross_peer_hypotheses": [
                    {
                        "id": "H1",
                        "claim": "short claim",
                        "minimal_test": "test",
                        "kill_condition": "kill",
                        "promote_condition": "promote",
                    }
                ],
            }
            with (
                patch(
                    "praxist.plugins.graph_maintainers.finding_graph_mvp.engine.build_session_start_graph_context",
                    return_value="",
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.pi_agent.load_agenda_for_gen",
                    return_value=agenda,
                ),
            ):
                context = prompt_context.build_prompt_context(
                    task_spec=task_spec,
                    workspace=root,
                    run_dir=root / "run",
                    results_dir=root / "results",
                    variants_dir=root / "variants",
                    findings_dir=root / "findings",
                    frontier=frontier,
                    local_mode=True,
                    gen_id=2,
                    peer_index=0,
                    cohort_size=2,
                    strategy="pi_directed",
                )

        compact = context["research_agenda"]
        self.assertEqual(compact["peer_contracts"]["gen2_peer0"]["role"], "exploit")
        self.assertEqual(len(compact["peer_contracts"]["gen2_peer0"]["forbidden_actions"]), 12)
        self.assertEqual(len(compact["peer_contracts"]["gen2_peer0"]["required_controls"]), 12)
        self.assertEqual(compact["peer_contracts"]["gen2_peer0"]["source_lane"], "alpha_incubator")
        self.assertEqual(compact["peer_contracts"]["gen2_peer0"]["target_lane"], "confirmed_alpha")
        self.assertEqual(
            compact["peer_contracts"]["gen2_peer0"]["coverage_check"], "current coverage"
        )
        self.assertEqual(
            compact["peer_contracts"]["gen2_peer0"]["custom_binding_field"], "must survive"
        )
        self.assertNotIn("gen2_peer1", compact["peer_contracts"])
        sibling = compact["sibling_roster"][0]
        self.assertEqual(sibling["peer_id"], "gen2_peer1")
        self.assertEqual(sibling["target_hypothesis"], "other target")
        self.assertEqual(sibling["coverage_check"], "other coverage")
        self.assertEqual(sibling["source_lane"], "diagnostic_control")
        self.assertEqual(sibling["target_lane"], "alpha_incubator")
        self.assertEqual(compact["sibling_roster_total_count"], 1)
        self.assertEqual(compact["sibling_contracts_omitted_count"], 0)
        self.assertEqual(compact["sibling_coordination_summary"]["role_counts"], {"bridge": 1})
        self.assertIn("cross_peer_hypotheses", compact)
        self.assertEqual(compact["cross_peer_hypotheses"][0]["id"], "H1")
        self.assertNotIn("panel_summary", compact)
        self.assertNotIn("consensus_actions", compact)
        self.assertNotIn("private_kb_source", sibling)
        self.assertIn("panel_summary", compact["_prompt_slicing"]["omitted_top_level_keys"])

    def test_prompt_context_caps_sibling_roster_with_coordination_summary(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import prompt_context

        contracts = {
            "gen2_peer0": {
                "role": "exploit",
                "target_hypothesis": "H0",
                "source_lane": "alpha",
                "target_lane": "confirmed_alpha",
            }
        }
        for index in range(1, 14):
            contracts[f"gen2_peer{index}"] = {
                "role": "bridge" if index % 2 else "skeptic",
                "target_hypothesis": "H0" if index in {1, 2} else f"H{index}",
                "source_lane": "alpha" if index in {1, 3} else "diagnostic_control",
                "target_lane": "alpha_incubator",
            }

        compact = prompt_context._compact_research_agenda_for_prompt(
            {"peer_contracts": contracts},
            "gen2_peer0",
        )

        self.assertIsNotNone(compact)
        assert compact is not None
        self.assertEqual(compact["sibling_roster_total_count"], 13)
        self.assertEqual(len(compact["sibling_roster"]), 8)
        self.assertEqual(compact["sibling_contracts_omitted_count"], 5)
        visible_ids = {item["peer_id"] for item in compact["sibling_roster"]}
        self.assertIn("gen2_peer1", visible_ids)
        self.assertIn("gen2_peer2", visible_ids)
        self.assertEqual(
            compact["sibling_coordination_summary"]["role_counts"],
            {"bridge": 7, "skeptic": 6},
        )

    def test_agenda_metadata_does_not_backfill_empty_coverage_check(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.agenda_metadata import (
            normalize_agenda_research_metadata,
        )

        agenda = {
            "cross_peer_hypotheses": [
                {
                    "id": "H1",
                    "claim": "claim",
                    "minimal_test": "test",
                    "kill_condition": "kill",
                    "promote_condition": "promote",
                }
            ],
            "peer_contracts": {
                "gen2_peer0": {
                    "role": "bridge",
                    "target_hypothesis": "H1",
                    "success_signal": "bridge succeeds",
                }
            },
        }

        changed = normalize_agenda_research_metadata(agenda)

        self.assertNotIn("coverage_check", agenda["peer_contracts"]["gen2_peer0"])
        self.assertNotIn("mechanism_hypothesis_deliverable", agenda["peer_contracts"]["gen2_peer0"])
        self.assertFalse(any(item.endswith(".coverage_check") for item in changed))

    def test_prompt_context_keeps_current_peer_source_context(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import prompt_context

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_spec = SimpleNamespace(
                evaluation=SimpleNamespace(
                    primary_metric="future_fitness",
                    diversity_dimensions=[],
                    must_explore_axes=[],
                ),
            )
            frontier = SimpleNamespace(get_summary=lambda: [])
            agenda = {
                "generation": 2,
                "synthesized_from_gen": 1,
                "consensus_actions": [
                    {
                        "action_id": "A1",
                        "claim_or_hypothesis": "H1",
                        "minimal_experiment": "run compact consensus experiment",
                        "large_notes": "x" * 5000,
                    },
                    {"action_id": "A2", "claim_or_hypothesis": "unrelated"},
                ],
                "DISSENT_TO_EXPERIMENT": [
                    {
                        "dissent_id": "D1",
                        "disputed_claim": "H1",
                        "resolving_experiment": "decide the disputed claim",
                    }
                ],
                "minority_high_upside": [
                    {
                        "idea_id": "M1",
                        "rationale": "rare upside rationale",
                        "success_condition": "rare signal succeeds",
                    }
                ],
                "claim_boundary_updates": [
                    {
                        "claim_id": "H1",
                        "new_language": "bounded claim",
                        "required_validation_before_upgrade": [
                            "validate the bounded claim before promotion"
                        ],
                    }
                ],
                "peer_contracts": {
                    "gen2_peer0": {
                        "role": "falsifier",
                        "target_hypothesis": "H1",
                        "source": "DISSENT_TO_EXPERIMENT",
                        "parent_candidate": "M1",
                        "success_signal": "resolve H1",
                    }
                },
                "cross_peer_hypotheses": [],
            }
            with (
                patch(
                    "praxist.plugins.graph_maintainers.finding_graph_mvp.engine.build_session_start_graph_context",
                    return_value="",
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.pi_agent.load_agenda_for_gen",
                    return_value=agenda,
                ),
            ):
                context = prompt_context.build_prompt_context(
                    task_spec=task_spec,
                    workspace=root,
                    run_dir=root / "run",
                    results_dir=root / "results",
                    variants_dir=root / "variants",
                    findings_dir=root / "findings",
                    frontier=frontier,
                    local_mode=True,
                    gen_id=2,
                    peer_index=0,
                    cohort_size=1,
                    strategy="pi_directed",
                )

        compact = context["research_agenda"]
        source_context = compact["current_peer_source_context"]
        self.assertNotIn("consensus_actions", compact)
        self.assertEqual(
            source_context["consensus_actions"][0]["minimal_experiment"],
            "run compact consensus experiment",
        )
        self.assertNotIn("A2", str(source_context))
        self.assertEqual(len(source_context["consensus_actions"][0]["large_notes"]), 5000)
        self.assertEqual(
            source_context["DISSENT_TO_EXPERIMENT"][0]["resolving_experiment"],
            "decide the disputed claim",
        )
        self.assertEqual(
            source_context["minority_high_upside"][0]["rationale"],
            "rare upside rationale",
        )
        self.assertEqual(
            source_context["claim_boundary_updates"][0]["required_validation_before_upgrade"][0],
            "validate the bounded claim before promotion",
        )

    def test_prompt_context_keeps_current_target_hypothesis_when_late(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import prompt_context

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_spec = SimpleNamespace(
                evaluation=SimpleNamespace(
                    primary_metric="future_fitness",
                    diversity_dimensions=[],
                    must_explore_axes=[],
                ),
            )
            frontier = SimpleNamespace(get_summary=lambda: [])
            agenda = {
                "generation": 2,
                "peer_contracts": {
                    "gen2_peer0": {
                        "role": "exploit",
                        "target_hypothesis": "H10",
                        "success_signal": "test late target",
                    }
                },
                "cross_peer_hypotheses": [
                    {
                        "id": f"H{i}",
                        "claim": f"claim {i}",
                        "minimal_test": "test",
                        "kill_condition": "kill",
                        "promote_condition": "promote",
                    }
                    for i in range(12)
                ],
            }
            with (
                patch(
                    "praxist.plugins.graph_maintainers.finding_graph_mvp.engine.build_session_start_graph_context",
                    return_value="",
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.pi_agent.load_agenda_for_gen",
                    return_value=agenda,
                ),
            ):
                context = prompt_context.build_prompt_context(
                    task_spec=task_spec,
                    workspace=root,
                    run_dir=root / "run",
                    results_dir=root / "results",
                    variants_dir=root / "variants",
                    findings_dir=root / "findings",
                    frontier=frontier,
                    local_mode=True,
                    gen_id=2,
                    peer_index=0,
                    cohort_size=1,
                    strategy="pi_directed",
                )

        hypothesis_ids = {
            item["id"] for item in context["research_agenda"]["cross_peer_hypotheses"]
        }
        self.assertIn("H10", hypothesis_ids)
        self.assertIn("cross_peer_hypotheses_omitted_count", context["research_agenda"])

    def test_prompt_context_keeps_falsification_target_hypothesis_when_late(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import prompt_context

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_spec = SimpleNamespace(
                evaluation=SimpleNamespace(
                    primary_metric="future_fitness",
                    diversity_dimensions=[],
                    must_explore_axes=[],
                ),
            )
            frontier = SimpleNamespace(get_summary=lambda: [])
            agenda = {
                "generation": 2,
                "peer_contracts": {
                    "gen2_peer0": {
                        "role": "falsifier",
                        "target_hypothesis": "falsification_contract",
                        "success_signal": "falsify late target",
                    }
                },
                "falsification_contract": {
                    "target_hypothesis": "H10",
                    "required_controls": ["late target control"],
                    "decision_rule": "kill if control fails",
                },
                "cross_peer_hypotheses": [
                    {
                        "id": f"H{i}",
                        "claim": f"claim {i}",
                        "minimal_test": f"test {i}",
                        "kill_condition": "kill",
                        "promote_condition": "promote",
                    }
                    for i in range(12)
                ],
            }
            with (
                patch(
                    "praxist.plugins.graph_maintainers.finding_graph_mvp.engine.build_session_start_graph_context",
                    return_value="",
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.pi_agent.load_agenda_for_gen",
                    return_value=agenda,
                ),
            ):
                context = prompt_context.build_prompt_context(
                    task_spec=task_spec,
                    workspace=root,
                    run_dir=root / "run",
                    results_dir=root / "results",
                    variants_dir=root / "variants",
                    findings_dir=root / "findings",
                    frontier=frontier,
                    local_mode=True,
                    gen_id=2,
                    peer_index=0,
                    cohort_size=1,
                    strategy="pi_directed",
                )

        hypotheses = {
            item["id"]: item for item in context["research_agenda"]["cross_peer_hypotheses"]
        }
        self.assertEqual(hypotheses["H10"]["minimal_test"], "test 10")
        self.assertIn("cross_peer_hypotheses_omitted_count", context["research_agenda"])

    def test_prompt_context_keeps_bridge_anchor_hypotheses_when_late(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import prompt_context

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_spec = SimpleNamespace(
                evaluation=SimpleNamespace(
                    primary_metric="future_fitness",
                    diversity_dimensions=[],
                    must_explore_axes=[],
                ),
            )
            frontier = SimpleNamespace(get_summary=lambda: [])
            agenda = {
                "generation": 2,
                "peer_contracts": {
                    "gen2_peer0": {
                        "role": "bridge",
                        "target_hypothesis": "B_g2_01",
                        "success_signal": "bridge late anchors",
                    }
                },
                "bridge_hypothesis": {
                    "id": "B_g2_01",
                    "source_anchor_A": {
                        "variant": "H10",
                        "extracted_mechanism": "H10",
                    },
                    "source_anchor_B": {
                        "variant": "H11",
                        "extracted_mechanism": "H11",
                    },
                    "expected_pareto_movement": "combine late anchors",
                },
                "cross_peer_hypotheses": [
                    {
                        "id": f"H{i}",
                        "claim": f"claim {i}",
                        "minimal_test": f"test {i}",
                        "kill_condition": "kill",
                        "promote_condition": "promote",
                    }
                    for i in range(12)
                ],
            }
            with (
                patch(
                    "praxist.plugins.graph_maintainers.finding_graph_mvp.engine.build_session_start_graph_context",
                    return_value="",
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.pi_agent.load_agenda_for_gen",
                    return_value=agenda,
                ),
            ):
                context = prompt_context.build_prompt_context(
                    task_spec=task_spec,
                    workspace=root,
                    run_dir=root / "run",
                    results_dir=root / "results",
                    variants_dir=root / "variants",
                    findings_dir=root / "findings",
                    frontier=frontier,
                    local_mode=True,
                    gen_id=2,
                    peer_index=0,
                    cohort_size=1,
                    strategy="pi_directed",
                )

        hypothesis_ids = {
            item["id"] for item in context["research_agenda"]["cross_peer_hypotheses"]
        }
        self.assertIn("H10", hypothesis_ids)
        self.assertIn("H11", hypothesis_ids)
        self.assertIn("cross_peer_hypotheses_omitted_count", context["research_agenda"])

    def test_prompt_context_slices_top_level_role_contracts_to_current_peer(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import prompt_context

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_spec = SimpleNamespace(
                evaluation=SimpleNamespace(
                    primary_metric="future_fitness",
                    diversity_dimensions=[],
                    must_explore_axes=[],
                ),
            )
            frontier = SimpleNamespace(get_summary=lambda: [])
            agenda = {
                "generation": 2,
                "peer_contracts": {
                    "gen2_peer0": {
                        "role": "bridge",
                        "target_hypothesis": "bridge_hypothesis",
                        "success_signal": "test bridge",
                    }
                },
                "bridge_hypothesis": {
                    **{f"metadata_{i}": f"value_{i}" for i in range(12)},
                    "id": "bridge_hypothesis",
                    "source_anchor_A": {
                        "variant": "A",
                        "extracted_mechanism": "mechanism A",
                    },
                    "source_anchor_B": {
                        "variant": "B",
                        "extracted_mechanism": "mechanism B",
                    },
                    "expected_pareto_movement": "improve return without worsening drawdown",
                },
                "falsification_contract": {
                    **{f"metadata_{i}": f"value_{i}" for i in range(12)},
                    "target_hypothesis": "target",
                    "required_controls": [f"control {i}" for i in range(12)],
                    "decision_rule": "reject if control fails",
                },
                "anti_mainline_contract": {
                    **{f"metadata_{i}": f"value_{i}" for i in range(12)},
                    "forbidden_mechanisms": [f"mechanism {i}" for i in range(12)],
                    "target_axes": [f"axis {i}" for i in range(12)],
                },
            }
            with (
                patch(
                    "praxist.plugins.graph_maintainers.finding_graph_mvp.engine.build_session_start_graph_context",
                    return_value="",
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.pi_agent.load_agenda_for_gen",
                    return_value=agenda,
                ),
            ):
                context = prompt_context.build_prompt_context(
                    task_spec=task_spec,
                    workspace=root,
                    run_dir=root / "run",
                    results_dir=root / "results",
                    variants_dir=root / "variants",
                    findings_dir=root / "findings",
                    frontier=frontier,
                    local_mode=True,
                    gen_id=2,
                    peer_index=0,
                    cohort_size=1,
                    strategy="pi_directed",
                )

        compact = context["research_agenda"]
        self.assertEqual(
            compact["bridge_hypothesis"]["expected_pareto_movement"],
            "improve return without worsening drawdown",
        )
        self.assertEqual(
            compact["bridge_hypothesis"]["source_anchor_A"]["extracted_mechanism"],
            "mechanism A",
        )
        self.assertNotIn("falsification_contract", compact)
        self.assertNotIn("anti_mainline_contract", compact)

    def test_prompt_context_keeps_role_specific_top_level_contract(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import prompt_context

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_spec = SimpleNamespace(
                evaluation=SimpleNamespace(
                    primary_metric="future_fitness",
                    diversity_dimensions=[],
                    must_explore_axes=[],
                ),
            )
            frontier = SimpleNamespace(get_summary=lambda: [])
            agenda = {
                "generation": 2,
                "peer_contracts": {
                    "gen2_peer0": {
                        "role": "falsifier",
                        "target_hypothesis": "falsification_contract",
                        "success_signal": "test falsification",
                    }
                },
                "falsification_contract": {
                    "target_hypothesis": "target",
                    "required_controls": [f"control {i}" for i in range(12)],
                    "decision_rule": "reject if control fails",
                },
                "anti_mainline_contract": {
                    "forbidden_mechanisms": ["unrelated"],
                    "target_axes": ["unrelated"],
                },
            }
            with (
                patch(
                    "praxist.plugins.graph_maintainers.finding_graph_mvp.engine.build_session_start_graph_context",
                    return_value="",
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.pi_agent.load_agenda_for_gen",
                    return_value=agenda,
                ),
            ):
                context = prompt_context.build_prompt_context(
                    task_spec=task_spec,
                    workspace=root,
                    run_dir=root / "run",
                    results_dir=root / "results",
                    variants_dir=root / "variants",
                    findings_dir=root / "findings",
                    frontier=frontier,
                    local_mode=True,
                    gen_id=2,
                    peer_index=0,
                    cohort_size=1,
                    strategy="pi_directed",
                )

        compact = context["research_agenda"]
        self.assertEqual(len(compact["falsification_contract"]["required_controls"]), 12)
        self.assertEqual(
            compact["falsification_contract"]["decision_rule"], "reject if control fails"
        )
        self.assertNotIn("anti_mainline_contract", compact)

    def test_extract_generation_id_accepts_peer_and_path_forms(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
        )

        self.assertEqual(findings_collection._extract_generation_id("gen6_peer1_alpha"), 6)
        self.assertEqual(findings_collection._extract_generation_id("gen_6/gen6_peer1"), 6)
        self.assertEqual(findings_collection._extract_generation_id("gen-6-peer1"), 6)
        self.assertEqual(findings_collection._extract_generation_id("results/gen12_peer3_x"), 12)

    def test_result_summary_metrics_preserve_distinct_mdd_rollups(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
        )

        metrics = findings_collection._result_summary_metrics(
            {
                "variant_name": "paired_cell_only_variant",
                "tier_reached": "T1",
                "tier_status": "stop_after_T1",
                "final_status": "stop_after_T1",
                "metrics": {
                    "bottleneck_target": "drawdown_regression",
                    "evidence_stage": "full_T1",
                    "tradeoff_class": "high_return_drawdown_repair_target",
                    "primary_tradeoff": "return_vs_mdd",
                    "next_step_intent": "repair_failure_mode",
                    "parent_candidate": "parent_alpha",
                    "parent_usage": "repair",
                },
                "failed_cells": [],
                "current_aggregate": {
                    "future_fitness": -1.0,
                    "mean_active_alpha_vs_benchmark_pct": 1.0,
                    "mean_active_share": 0.4,
                    "max_drawdown_pct": 21.0,
                    "max_drawdown_delta_pct": 2.5,
                },
                "all_paired_cells": [
                    {
                        "validation_only": False,
                        "variant_return": 3.0,
                        "variant_mdd": 8.0,
                        "mean_effective_n": 12.0,
                        "mean_active_share": 0.4,
                    },
                    {
                        "validation_only": False,
                        "variant_return": 5.0,
                        "variant_mdd": 10.0,
                        "mean_effective_n": 10.0,
                        "mean_active_share": 0.6,
                    },
                    {
                        "validation_only": True,
                        "variant_return": 7.0,
                        "variant_mdd": 6.0,
                        "mean_effective_n": 9.0,
                    },
                ],
            },
            cell_metric_derivations=[
                {
                    "name": "mean_return_pct",
                    "source_keys": ["variant_return"],
                    "aggregate": "mean",
                },
                {
                    "name": "mean_mdd_pct",
                    "source_keys": ["variant_mdd"],
                    "aggregate": "mean",
                },
                {
                    "name": "validation_2026_return_pct",
                    "source_keys": ["variant_return"],
                    "aggregate": "mean",
                    "validation_only": True,
                },
                {
                    "name": "validation_2026_mdd_pct",
                    "source_keys": ["variant_mdd"],
                    "aggregate": "mean",
                    "validation_only": True,
                },
            ],
            metric_aliases={
                "promotion_mean_mdd_pct": "mean_mdd_pct",
                "promotion_worst_window_mdd_pct": "max_drawdown_pct",
            },
        )

        self.assertEqual(metrics["mean_return_pct"], 4.0)
        self.assertEqual(metrics["mean_mdd_pct"], 9.0)
        self.assertEqual(metrics["promotion_mean_mdd_pct"], 9.0)
        self.assertEqual(metrics["promotion_worst_window_mdd_pct"], 21.0)
        self.assertEqual(metrics["validation_2026_return_pct"], 7.0)
        self.assertEqual(metrics["validation_2026_mdd_pct"], 6.0)
        self.assertEqual(metrics["bottleneck_target"], "drawdown_regression")
        self.assertEqual(metrics["evidence_stage"], "full_T1")
        self.assertEqual(metrics["tradeoff_class"], "high_return_drawdown_repair_target")
        self.assertEqual(metrics["primary_tradeoff"], "return_vs_mdd")
        self.assertEqual(metrics["next_step_intent"], "repair_failure_mode")
        self.assertEqual(metrics["parent_candidate"], "parent_alpha")
        self.assertEqual(metrics["parent_usage"], "repair")

    def test_pi_prior_metrics_keep_distinct_mdd_rollups(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import (
            PIAgent,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_spec.yaml").write_text(
                json.dumps(
                    {
                        "evaluation": {"primary_metric": "future_fitness"},
                        "gems": {
                            "cost_metric_keys": [
                                "mean_mdd_pct",
                                "promotion_mean_mdd_pct",
                                "promotion_worst_window_mdd_pct",
                                "validation_2026_mdd_pct",
                                "max_drawdown_pct",
                                "max_drawdown_delta_pct",
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            pi = PIAgent(run_dir=root, workspace=root, cohort_size=1, model="dummy")
            trimmed = pi._trim_prior_metrics(
                {
                    "future_fitness": 1.0,
                    "mean_mdd_pct": 9.0,
                    "promotion_mean_mdd_pct": 9.0,
                    "promotion_worst_window_mdd_pct": 21.0,
                    "validation_2026_mdd_pct": 6.0,
                    "max_drawdown_pct": 21.0,
                    "max_drawdown_delta_pct": 2.5,
                    "large_unused_blob": "x" * 1000,
                }
            )

        self.assertEqual(trimmed["mean_mdd_pct"], 9.0)
        self.assertEqual(trimmed["promotion_mean_mdd_pct"], 9.0)
        self.assertEqual(trimmed["promotion_worst_window_mdd_pct"], 21.0)
        self.assertEqual(trimmed["validation_2026_mdd_pct"], 6.0)
        self.assertNotIn("large_unused_blob", trimmed)

    def test_task_prior_metric_names_collects_all_spec_sources(self) -> None:
        """_task_prior_metric_names harvests metric names from every
        evaluation and gems source declared in task_spec.yaml, then caches
        the result. Metric names are intentionally domain-neutral here so the
        contract stays task-agnostic.
        """
        from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import (
            PIAgent,
        )

        spec = {
            "evaluation": {
                "primary_metric": "metric_primary",
                "aux_metrics": ["metric_aux", "  ", "metric_aux2"],
                "anchor_metrics": [
                    {"name": "metric_anchor_named"},
                    {"metric": "metric_anchor_metric"},
                    ["metric_anchor_seq", "ignored"],
                    "metric_anchor_scalar",
                ],
                "frontier_lanes": [
                    {
                        "require_metrics": ["metric_require"],
                        "require_truthy_metrics": ["metric_truthy"],
                        "require_falsey_metrics": ["metric_falsey"],
                        "axes": [
                            {"name": "metric_axis_named"},
                            ["metric_axis_seq"],
                            "metric_axis_scalar",
                        ],
                        "optional_axes": [{"metric": "metric_optional_axis"}],
                        "min_metrics": {"metric_min": 0.0},
                        "max_metrics": {"metric_max": 1.0},
                    },
                    "not_a_lane_dict",
                ],
            },
            "gems": {
                "primary_metric_keys": ["metric_gem_primary"],
                "secondary_metric_keys": ["metric_gem_secondary"],
                "lower_tail_metric_keys": ["metric_gem_lower"],
                "validation_metric_keys": ["metric_gem_validation"],
                "cost_metric_keys": ["metric_gem_cost"],
                "result_cell_metric_derivations": [
                    {"name": "metric_deriv_name"},
                    {"output": "metric_deriv_output"},
                    "not_a_dict",
                ],
                "result_metric_aliases": {
                    "metric_alias_key": "metric_alias_value",
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_spec.yaml").write_text(json.dumps(spec), encoding="utf-8")
            pi = PIAgent(run_dir=root, workspace=root, cohort_size=1, model="dummy")
            names = pi._task_prior_metric_names()

        expected = {
            "metric_primary",
            "metric_aux",
            "metric_aux2",
            "metric_anchor_named",
            "metric_anchor_metric",
            "metric_anchor_seq",
            "metric_anchor_scalar",
            "metric_require",
            "metric_truthy",
            "metric_falsey",
            "metric_axis_named",
            "metric_axis_seq",
            "metric_axis_scalar",
            "metric_optional_axis",
            "metric_min",
            "metric_max",
            "metric_gem_primary",
            "metric_gem_secondary",
            "metric_gem_lower",
            "metric_gem_validation",
            "metric_gem_cost",
            "metric_deriv_name",
            "metric_deriv_output",
            "metric_alias_key",
            "metric_alias_value",
        }
        self.assertEqual(names, expected)
        # Second call returns a cached copy (distinct object, equal value).
        cached = pi._task_prior_metric_names()
        self.assertEqual(cached, expected)
        self.assertIsNot(cached, names)

    def test_load_prior_findings_summary_reads_shared_store(self) -> None:
        """_load_prior_findings_summary returns compact per-gen summaries from
        the shared_store findings table, decoding JSON metrics/extra, merging
        nested extra, tolerating malformed JSON, and skipping non result/
        insight rows.
        """
        import sqlite3

        from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import (
            PIAgent,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "shared_store.db"
            conn = sqlite3.connect(db)
            conn.execute(
                "CREATE TABLE findings ("
                "id TEXT, finding_type TEXT, peer_id TEXT, variant_name TEXT, "
                "metrics TEXT, extra TEXT, title TEXT, generation_id INTEGER, "
                "timestamp TEXT)"
            )
            conn.executemany(
                "INSERT INTO findings VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (
                        "f0",
                        "result",
                        "peer-a",
                        "variant-a",
                        json.dumps({"metric_primary": 1.0}),
                        json.dumps({"extra": {"metric_primary": 2.0}}),
                        "Title A",
                        0,
                        "2026-01-01T00:00:00",
                    ),
                    (
                        "f1",
                        "insight",
                        "peer-b",
                        "variant-b",
                        "{not valid json",
                        json.dumps({"metric_primary": 3.0}),
                        "Title B",
                        1,
                        "2026-01-01T00:01:00",
                    ),
                    (
                        "f2",
                        "note",
                        "peer-c",
                        "variant-c",
                        None,
                        None,
                        "Filtered",
                        0,
                        "2026-01-01T00:02:00",
                    ),
                ],
            )
            conn.commit()
            conn.close()

            # Declare the metric so _trim_prior_metrics keeps it.
            (root / "task_spec.yaml").write_text(
                json.dumps({"evaluation": {"primary_metric": "metric_primary"}}),
                encoding="utf-8",
            )
            pi = PIAgent(run_dir=root, workspace=root, cohort_size=1, model="dummy")
            summary = pi._load_prior_findings_summary(2)

        ids = {row["id"] for row in summary}
        self.assertEqual(ids, {"f0", "f1"})
        by_id = {row["id"]: row for row in summary}
        # Nested extra merged, then real metrics override.
        self.assertEqual(by_id["f0"]["metrics"]["metric_primary"], 1.0)
        # Malformed metrics fall back to the extra dict.
        self.assertEqual(by_id["f1"]["metrics"]["metric_primary"], 3.0)

    def test_load_prior_agenda_reads_fenced_agenda_file(self) -> None:
        """_load_prior_agenda returns None for gen < 1 and for a missing
        file, and parses a markdown-fenced agenda YAML for a real gen.
        """
        from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import (
            AGENDA_FILE_PATTERN,
            PIAgent,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = PIAgent(run_dir=root, workspace=root, cohort_size=1, model="dummy")
            self.assertIsNone(pi._load_prior_agenda(0))
            self.assertIsNone(pi._load_prior_agenda(1))

            agendas_dir = root / "agendas"
            agendas_dir.mkdir(parents=True, exist_ok=True)
            (agendas_dir / AGENDA_FILE_PATTERN.format(1)).write_text(
                "```yaml\nconsensus_actions: []\nnote: ok\n```\n",
                encoding="utf-8",
            )
            loaded = pi._load_prior_agenda(1)

        self.assertIsInstance(loaded, dict)
        self.assertEqual(loaded["note"], "ok")

    def test_validation_candidate_parent_ids_collects_all_token_sources(self) -> None:
        """_validation_candidate_parent_ids harvests normalized parent tokens
        from identity keys, identity_aliases, and nested metrics (both keys
        and aliases), while skipping non-dict entries.
        """
        from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import (
            _validation_candidate_parent_ids,
        )

        entries = [
            "not_a_dict",
            {
                "finding_id": "Top Level ID",
                "identity_aliases": ["Alias One", "  ", "Alias Two"],
                "metrics": {
                    "variant_name": "Nested Variant",
                    "identity_aliases": ["Nested Alias"],
                },
            },
        ]

        ids = _validation_candidate_parent_ids(entries)

        # Tokens are whitespace-stripped and lowercased.
        self.assertIn("toplevelid", ids)
        self.assertIn("aliasone", ids)
        self.assertIn("aliastwo", ids)
        self.assertIn("nestedvariant", ids)
        self.assertIn("nestedalias", ids)
        # None input degrades to an empty set.
        self.assertEqual(_validation_candidate_parent_ids(None), set())

    def test_task_prior_metric_names_handles_missing_and_malformed_spec(self) -> None:
        """A missing task_spec.yaml yields an empty set; a malformed YAML
        file degrades to an empty set instead of raising.
        """
        from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import (
            PIAgent,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = PIAgent(run_dir=root, workspace=root, cohort_size=1, model="dummy")
            self.assertEqual(pi._task_prior_metric_names(), set())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_spec.yaml").write_text("[: not valid", encoding="utf-8")
            pi = PIAgent(run_dir=root, workspace=root, cohort_size=1, model="dummy")
            self.assertEqual(pi._task_prior_metric_names(), set())

    def test_baseline_prompt_context_and_findings_collection_fallbacks(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            baseline_runtime,
            findings_collection,
            prompt_context,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            task_dir.mkdir()
            curated = task_dir / "baselines.jsonl"
            curated.write_text(
                json.dumps({"optimizer": "base", "accuracy": 0.5}) + "\n", encoding="utf-8"
            )
            task_spec = SimpleNamespace(
                task_id="task",
                task_dir=task_dir,
                baselines=[SimpleNamespace(name="base")],
                _raw={"task_assets": {"baselines": {"curated_results": "baselines.jsonl"}}},
                evaluation=SimpleNamespace(
                    diversity_dimensions=[{"name": "mechanism"}],
                    must_explore_axes=[{"name": "mechanism"}],
                ),
            )
            report = SimpleNamespace(
                stale=0,
                missing_baselines=[],
                fresh=1,
                curated_baseline_names=["base"],
            )
            with (
                patch.object(
                    baseline_runtime.baseline_cache_mod,
                    "load_curated_baseline_entries",
                    return_value=[{"name": "base"}],
                ) as load_curated,
                patch.object(
                    baseline_runtime.baseline_cache_mod, "validate_cache", return_value=report
                ) as validate,
                patch.object(
                    baseline_runtime.baseline_cache_mod, "write_report_for_peers"
                ) as write_report,
            ):
                baseline_runtime.validate_baseline_cache_for_run(
                    task_spec=task_spec,
                    workspace=root,
                    run_dir=root / "run",
                )
            load_curated.assert_called_once_with(curated)
            validate.assert_called_once()
            write_report.assert_called_once()
            with patch.object(
                baseline_runtime.baseline_cache_mod,
                "load_curated_baseline_entries",
                side_effect=RuntimeError("non fatal"),
            ):
                baseline_runtime.validate_baseline_cache_for_run(
                    task_spec=task_spec,
                    workspace=root,
                    run_dir=root / "run",
                )

            frontier = SimpleNamespace(get_summary=lambda: [{"variant": "v", "generation_id": 0}])
            with (
                patch(
                    "praxist.plugins.graph_maintainers.finding_graph_mvp.engine.build_session_start_graph_context",
                    return_value="graph context",
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.pi_agent.load_agenda_for_gen",
                    return_value={"peer_contracts": {}},
                ),
            ):
                context = prompt_context.build_prompt_context(
                    task_spec=task_spec,
                    workspace=root,
                    run_dir=root / "run",
                    results_dir=root / "results",
                    variants_dir=root / "variants",
                    findings_dir=root / "findings",
                    frontier=frontier,
                    local_mode=True,
                    gen_id=1,
                    peer_index=2,
                    cohort_size=4,
                    strategy="explore",
                )
            self.assertEqual(context["peer_id"], "gen1_peer2")
            self.assertEqual(context["frontier_summary"], [{"variant": "v", "generation_id": 0}])
            self.assertEqual(context["graph_session_context"], "graph context")
            self.assertEqual(context["research_agenda"]["peer_contracts"], {})
            self.assertEqual(
                context["research_agenda"]["full_agenda_path"],
                str(root / "run" / "agendas" / "research_agenda_gen1.yaml"),
            )
            with (
                patch(
                    "praxist.plugins.graph_maintainers.finding_graph_mvp.engine.build_session_start_graph_context",
                    side_effect=RuntimeError("graph"),
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.pi_agent.load_agenda_for_gen",
                    side_effect=RuntimeError("agenda"),
                ),
            ):
                context = prompt_context.build_prompt_context(
                    task_spec=task_spec,
                    workspace=root,
                    run_dir=root / "run",
                    results_dir=root / "results",
                    variants_dir=root / "variants",
                    findings_dir=root / "findings",
                    frontier=frontier,
                    local_mode=False,
                    gen_id=0,
                    peer_index=0,
                    cohort_size=1,
                    strategy="explore",
                )
            self.assertEqual(context["graph_session_context"], "")
            self.assertIsNone(context["research_agenda"])

            findings_dir = root / "findings"
            findings_dir.mkdir()
            (findings_dir / "a.json").write_text(
                json.dumps({"id": "a", "peer_id": "gen2_peer0", "generation_id": 99}),
                encoding="utf-8",
            )
            (findings_dir / "b.json").write_text(
                json.dumps({"id": "b", "generation_id": 2}),
                encoding="utf-8",
            )
            (findings_dir / "bad.json").write_text("{bad", encoding="utf-8")
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.tools.findings_ingest.ingest_findings_directory",
                    side_effect=RuntimeError("ingest"),
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.tools.local_store.get_findings",
                    side_effect=RuntimeError("sqlite"),
                ),
            ):
                found = findings_collection.collect_findings_for_generation(
                    findings_dir=findings_dir,
                    gen_id=2,
                    local_mode=True,
                )
            self.assertEqual({row["id"] for row in found}, {"a", "b"})
            self.assertEqual(
                findings_collection.collect_findings_for_generation(
                    findings_dir=root / "missing",
                    gen_id=1,
                    local_mode=False,
                ),
                [],
            )
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.tools.findings_ingest.ingest_findings_directory",
                    return_value=0,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.tools.local_store.get_findings",
                    return_value=[{"id": "sqlite"}],
                ),
            ):
                self.assertEqual(
                    findings_collection.collect_findings_for_generation(
                        findings_dir=findings_dir,
                        gen_id=2,
                        local_mode=True,
                    ),
                    [{"id": "sqlite"}],
                )

    def test_generation_local_findings_and_result_artifacts_reach_incubator(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
            FrontierStore,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            shared = run_dir / "shared_findings"
            gen_shared = run_dir / "gen_2" / "shared_findings"
            gen_shared.mkdir(parents=True)
            shared.mkdir(parents=True)
            (gen_shared / "bridge_l1_eff_n_sweep.json").write_text(
                json.dumps(
                    {
                        "id": "local_sweep",
                        "finding_type": "insight",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "peer_id": "gen2_peer4",
                        "generation_id": 2,
                        "content": "bridge_l1_c005 looked strongest inside the sweep",
                        "metrics": {
                            "mean_active_alpha_vs_benchmark_pct": 4.0,
                            "mean_active_share": 0.5,
                            "tier": "T1",
                            "promotion_eligible": False,
                            "source_result_path": "results/bridge_l1_c005/tiered_eval_summary.json",
                        },
                    }
                ),
                encoding="utf-8",
            )
            result_dir = run_dir / "results" / "bridge_l1_c005"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "complete_eval": True,
                        "effort_ratio": 1.0,
                        "coverage_ratio": 1.0,
                        "tier_reached": "T1",
                        "tier_status": "stop_after_T1",
                        "final_status": "stop_after_T1",
                        "tier_history": [
                            {
                                "tier": "T1",
                                "status": "stop_after_T1",
                                "n_hard_constraint_violations": 3,
                            }
                        ],
                        "failed_cells": [],
                        "current_aggregate": {
                            "future_fitness": -7.73,
                            "mean_active_alpha_vs_benchmark_pct": 6.19,
                            "q25_active_alpha_vs_benchmark_pct": 1.2,
                            "active_ir": 0.31,
                            "mean_active_share": 0.62,
                            "mean_effective_n": 1.46,
                            "max_drawdown_delta_pct": 8.0,
                        },
                        "all_eval_cells": [
                            {
                                "validation_only": False,
                                "return_pct": 15.14,
                                "mdd_pct": 9.0,
                                "mean_effective_n": 1.46,
                                "mean_active_share": 0.62,
                            },
                            {
                                "validation_only": True,
                                "return_pct": 17.02,
                                "mdd_pct": 7.0,
                                "mean_effective_n": 1.4,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            crashed_dir = run_dir / "results" / "crashed_alpha_child"
            crashed_dir.mkdir(parents=True)
            (crashed_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "crashed_alpha_child",
                        "generation_id": 2,
                        "tier_reached": "T3",
                        "tier_status": "crashed",
                        "final_status": "crashed",
                        "current_aggregate": {
                            "future_fitness": 9.0,
                            "mean_active_alpha_vs_benchmark_pct": 9.0,
                            "mean_active_share": 0.5,
                        },
                    }
                ),
                encoding="utf-8",
            )
            incomplete_dir = run_dir / "results" / "pending_alpha_child"
            incomplete_dir.mkdir(parents=True)
            (incomplete_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "pending_alpha_child",
                        "generation_id": 2,
                        "tier_reached": "T2",
                        "tier_status": "running",
                        "final_status": "incomplete",
                        "current_aggregate": {
                            "future_fitness": 8.0,
                            "mean_active_alpha_vs_benchmark_pct": 8.0,
                            "mean_active_share": 0.5,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root / "db")}):
                found = findings_collection.collect_findings_for_generation(
                    findings_dir=shared,
                    gen_id=2,
                    local_mode=True,
                    result_scoring_metric_keys=AIST_RESULT_SCORING_KEYS,
                    result_artifact_default_lane="alpha_incubator",
                    result_artifact_default_family="task_candidate",
                    result_cell_metric_derivations=[
                        {
                            "name": "mean_return_pct",
                            "source_keys": ["return_pct"],
                            "aggregate": "mean",
                        }
                    ],
                )
                variants = {row.get("variant_name") for row in found}
                self.assertIn("bridge_l1_eff_n_sweep", variants)
                self.assertIn("bridge_l1_c005", variants)
                self.assertNotIn("crashed_alpha_child", variants)
                self.assertNotIn("pending_alpha_child", variants)
                self.assertTrue(
                    (run_dir / "gen_2" / "shared_findings" / "bridge_l1_eff_n_sweep.json").exists()
                )
                self.assertFalse((shared / "gen2_bridge_l1_eff_n_sweep.json").exists())
                bridge_row = next(
                    row for row in found if row.get("variant_name") == "bridge_l1_c005"
                )
                self.assertEqual(bridge_row["generation_id"], 2)
                self.assertEqual(
                    bridge_row["metrics"]["source_generation_inference"],
                    "generation_local_finding_reference",
                )

                store = FrontierStore(
                    root / "frontier",
                    primary_metric="future_fitness",
                    metric_direction="maximize",
                    require_tier=True,
                    frontier_lanes=[
                        {
                            "name": "alpha_incubator",
                            "k": 10,
                            "exclude_families": [
                                "benchmark_floor",
                                "benchmark",
                                "passive",
                                "nonrl_floor",
                                "diagnostic_control",
                                "process_audit",
                            ],
                            "allow_lower_tier": True,
                            "allow_non_promotable": True,
                            "allow_risk_violating": True,
                            "require_metrics": [
                                "mean_active_alpha_vs_benchmark_pct",
                                "mean_active_share",
                            ],
                            "min_metrics": {
                                "mean_active_alpha_vs_benchmark_pct": -10.0,
                                "mean_active_share": 0.005,
                            },
                            "axes": [
                                {
                                    "name": "mean_active_alpha_vs_benchmark_pct",
                                    "direction": "maximize",
                                },
                                {"name": "mean_active_share", "direction": "maximize"},
                            ],
                            "optional_axes": [
                                {"name": "active_ir", "direction": "maximize"},
                                {"name": "future_fitness", "direction": "maximize"},
                            ],
                        }
                    ],
                )
                promoted = store.promote(2, found)

            promoted_variants = {row.get("variant_name") for row in promoted}
            self.assertIn("bridge_l1_c005", promoted_variants)
            self.assertNotIn("crashed_alpha_child", promoted_variants)
            self.assertNotIn("pending_alpha_child", promoted_variants)
            bridge = next(row for row in promoted if row.get("variant_name") == "bridge_l1_c005")
            self.assertEqual(bridge["frontier_lane"], "alpha_incubator")
            self.assertEqual(bridge["metrics"]["strategy_family"], "task_candidate")

            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "generation_id": 2,
                        "tier_reached": "T3",
                        "tier_status": "crashed",
                        "final_status": "crashed",
                        "current_aggregate": {
                            "future_fitness": 99.0,
                            "mean_active_alpha_vs_benchmark_pct": 99.0,
                            "mean_active_share": 0.99,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root / "db")}):
                refreshed = findings_collection.collect_findings_for_generation(
                    findings_dir=shared,
                    gen_id=2,
                    local_mode=True,
                    result_scoring_metric_keys=AIST_RESULT_SCORING_KEYS,
                )
                refreshed_variants = {row.get("variant_name") for row in refreshed}
                self.assertNotIn("bridge_l1_c005", refreshed_variants)

    def test_result_artifact_uses_unique_longest_scheduler_output_owner(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result_dir = run_dir / "results" / "family" / "candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "candidate",
                        "generation_id": 2,
                        "complete_eval": True,
                        "current_aggregate": {"score": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            scheduler_dir = run_dir / "resource_scheduler"
            scheduler_dir.mkdir()
            events = [
                {
                    "event": "submitted",
                    "peer_id": "gen2_peer0",
                    "generation_id": 2,
                    "cwd": str(run_dir),
                    "command": ["python", "evaluate.py", "--output-dir", "results/family"],
                },
                {
                    "event": "submitted",
                    "peer_id": "gen2_peer1",
                    "generation_id": 2,
                    "cwd": str(run_dir),
                    "command": [
                        "python",
                        "evaluate.py",
                        "--output-dir=results/family/candidate",
                    ],
                },
                {
                    "event": "completed",
                    "peer_id": "gen2_peer3",
                    "generation_id": 2,
                    "cwd": str(run_dir),
                    "command": ["python", "evaluate.py", str(result_dir)],
                },
            ]
            (scheduler_dir / "events.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )

            findings = findings_collection._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=2,
                scoring_metric_keys=("score",),
            )

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["peer_id"], "gen2_peer1")

    def test_result_artifact_accepts_out_dir_over_broad_results_cwd(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result_dirs = {
                "candidate_a": run_dir / "results" / "candidate_a",
                "candidate_b": run_dir / "results" / "candidate_b",
            }
            for variant_name, result_dir in result_dirs.items():
                result_dir.mkdir(parents=True)
                (result_dir / "evaluation_summary.json").write_text(
                    json.dumps(
                        {
                            "variant_name": variant_name,
                            "generation_id": 2,
                            "complete_eval": True,
                            "current_aggregate": {"score": 1.0},
                        }
                    ),
                    encoding="utf-8",
                )

            scheduler_dir = run_dir / "resource_scheduler"
            scheduler_dir.mkdir()
            events = [
                {
                    "event": "submitted",
                    "peer_id": "gen2_peer6",
                    "generation_id": 2,
                    "cwd": str(run_dir / "results"),
                    "command": [
                        "python",
                        "evaluate.py",
                        "--out-dir",
                        str(result_dirs["candidate_a"]),
                    ],
                },
                {
                    "event": "submitted",
                    "peer_id": "gen2_peer8",
                    "generation_id": 2,
                    "cwd": str(run_dir),
                    "command": [
                        "python",
                        "evaluate.py",
                        f"--out-dir={result_dirs['candidate_b']}",
                    ],
                },
            ]
            (scheduler_dir / "events.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )

            findings = findings_collection._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=2,
                scoring_metric_keys=("score",),
            )

        owners = {finding["variant_name"]: finding["peer_id"] for finding in findings}
        self.assertEqual(
            owners,
            {"candidate_a": "gen2_peer6", "candidate_b": "gen2_peer8"},
        )

    def test_result_artifact_refreshes_scheduler_owner_and_preserves_ambiguous_unknown(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result_dir = run_dir / "results" / "candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "candidate",
                        "generation_id": 0,
                        "complete_eval": True,
                        "current_aggregate": {"score": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            first = findings_collection._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=0,
                scoring_metric_keys=("score",),
            )
            self.assertEqual(first[0]["peer_id"], "gen0_result_artifact")

            scheduler_dir = run_dir / "resource_scheduler"
            scheduler_dir.mkdir()
            first_event = {
                "event": "submitted",
                "peer_id": "gen0_peer2",
                "generation_id": 0,
                "cwd": str(run_dir),
                "command": ["python", "evaluate.py", "--output", "results/candidate"],
            }
            events_path = scheduler_dir / "events.jsonl"
            events_path.write_text(json.dumps(first_event) + "\n", encoding="utf-8")

            refreshed = findings_collection._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=0,
                scoring_metric_keys=("score",),
            )
            self.assertEqual(refreshed[0]["peer_id"], "gen0_peer2")

            summary_path = result_dir / "tiered_eval_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["peer_id"] = "gen0_peer9"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            conflicting_event = {**first_event, "peer_id": "gen0_peer3"}
            events_path.write_text(
                json.dumps(first_event) + "\n" + json.dumps(conflicting_event) + "\n",
                encoding="utf-8",
            )
            ambiguous = findings_collection._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=0,
                scoring_metric_keys=("score",),
            )
            persisted = next((run_dir / "shared_findings").glob("*.json"))

            self.assertEqual(ambiguous[0]["peer_id"], "gen0_unknown_peer")
            self.assertEqual(json.loads(persisted.read_text())["peer_id"], "gen0_unknown_peer")

    def test_result_artifact_owner_ignores_inputs_and_resolves_symlinked_outputs(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            physical_results = root / "physical-results"
            result_dir = physical_results / "candidate"
            result_dir.mkdir(parents=True)
            run_dir.mkdir()
            (run_dir / "results").symlink_to(physical_results, target_is_directory=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "candidate",
                        "generation_id": 1,
                        "complete_eval": True,
                        "current_aggregate": {"score": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            scheduler_dir = run_dir / "resource_scheduler"
            scheduler_dir.mkdir()
            events = [
                {
                    "event": "submitted",
                    "peer_id": "gen1_peer0",
                    "generation_id": 1,
                    "cwd": str(run_dir),
                    "command": [
                        "python",
                        "evaluate.py",
                        "--config",
                        "results/candidate/input.json",
                    ],
                },
                {
                    "event": "submitted",
                    "peer_id": "gen1_peer1",
                    "generation_id": 1,
                    "cwd": str(run_dir),
                    "command": [
                        "python",
                        "evaluate.py",
                        "--output-dir",
                        "results/candidate",
                    ],
                },
            ]
            (scheduler_dir / "events.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )

            findings = findings_collection._materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=1,
                scoring_metric_keys=("score",),
            )

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["peer_id"], "gen1_peer1")

    def test_result_artifact_materialization_disabled_removes_stale_auto_findings(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            local_store,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            shared = run_dir / "shared_findings"
            result_dir = run_dir / "results" / "candidate_a"
            shared.mkdir(parents=True)
            result_dir.mkdir(parents=True)
            stale = shared / "stale_auto.json"
            stale.write_text(
                json.dumps(
                    {
                        "id": "stale_auto",
                        "finding_type": "result",
                        "variant_name": "candidate_a",
                        "peer_id": "gen0_result_artifact",
                        "generation_id": 0,
                        "metrics": {
                            "score": 0.9,
                            "auto_materialized_from_result_artifact": True,
                            "source_result_path": "results/candidate_a/result_summary.json",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (shared / "manual.json").write_text(
                json.dumps(
                    {
                        "id": "manual",
                        "finding_type": "result",
                        "variant_name": "manual_candidate",
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "metrics": {"score": 0.5},
                    }
                ),
                encoding="utf-8",
            )
            (result_dir / "result_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "candidate_a",
                        "current_aggregate": {"score": 0.9, "scored_cell_count": 3},
                        "n_eval_cells": 3,
                    }
                ),
                encoding="utf-8",
            )
            noise_dir = run_dir / "results" / "gen2_peer5_failed_noise"
            noise_dir.mkdir(parents=True)
            (noise_dir / "result_summary.json").write_text(
                json.dumps(
                    {
                        "variant_id": "gen2_peer5_failed_noise",
                        "generation_id": 2,
                        "result_status": "failed",
                        "scored_complete": False,
                        "current_aggregate": {
                            "result_status": "failed",
                            "scored_complete": False,
                            "failure_mode": "process_error",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root / "db")}):
                local_store.init_db()
                local_store.insert_finding(
                    {
                        "id": "stale_auto_db",
                        "finding_type": "result",
                        "variant_name": "candidate_a_db",
                        "peer_id": "gen0_result_artifact",
                        "generation_id": 0,
                        "metrics": {
                            "score": 0.95,
                            "auto_materialized_from_result_artifact": True,
                            "source_result_path": "results/candidate_a/result_summary.json",
                        },
                    }
                )
                found = findings_collection.collect_findings_for_generation(
                    findings_dir=shared,
                    gen_id=0,
                    local_mode=True,
                    materialize_result_artifacts=False,
                )
                stored_variants = {
                    row.get("variant_name") for row in local_store.get_all_findings()
                }

            self.assertFalse(stale.exists())
            self.assertEqual({row.get("variant_name") for row in found}, {"manual_candidate"})
            self.assertEqual(stored_variants, {"manual_candidate"})

    def test_filesystem_fallback_filters_disabled_auto_materialized_findings(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            shared = run_dir / "shared_findings"
            shared.mkdir(parents=True)
            stale = shared / "stale_auto.json"
            stale.write_text(
                json.dumps(
                    {
                        "id": "stale_auto",
                        "finding_type": "result",
                        "variant_name": "stale_candidate",
                        "peer_id": "gen0_result_artifact",
                        "generation_id": 0,
                        "metrics": {
                            "score": 0.9,
                            "auto_materialized_from_result_artifact": True,
                            "source_result_path": "results/stale/result_summary.json",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (shared / "manual.json").write_text(
                json.dumps(
                    {
                        "id": "manual",
                        "finding_type": "result",
                        "variant_name": "manual_candidate",
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "metrics": {"score": 0.5},
                    }
                ),
                encoding="utf-8",
            )

            found = findings_collection.collect_findings_for_generation(
                findings_dir=shared,
                gen_id=0,
                local_mode=False,
                materialize_result_artifacts=False,
            )

        self.assertFalse(stale.exists())
        self.assertEqual({row.get("variant_name") for row in found}, {"manual_candidate"})

    def test_filesystem_fallback_filters_stale_auto_materialized_missing_source(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            shared = run_dir / "shared_findings"
            shared.mkdir(parents=True)
            stale = shared / "stale_auto.json"
            stale.write_text(
                json.dumps(
                    {
                        "id": "stale_auto",
                        "finding_type": "result",
                        "variant_name": "stale_candidate",
                        "peer_id": "gen0_result_artifact",
                        "generation_id": 0,
                        "metrics": {
                            "score": 0.9,
                            "auto_materialized_from_result_artifact": True,
                            "source_result_path": "results/stale/result_summary.json",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (shared / "manual.json").write_text(
                json.dumps(
                    {
                        "id": "manual",
                        "finding_type": "result",
                        "variant_name": "manual_candidate",
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "metrics": {"score": 0.5},
                    }
                ),
                encoding="utf-8",
            )

            found = findings_collection.collect_findings_for_generation(
                findings_dir=shared,
                gen_id=0,
                local_mode=False,
                materialize_result_artifacts=True,
            )

        self.assertFalse(stale.exists())
        self.assertEqual({row.get("variant_name") for row in found}, {"manual_candidate"})

    def test_missing_results_dir_removes_stale_auto_materialized_files(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
            _materialize_result_artifacts,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            shared.mkdir(parents=True)
            stale = shared / "stale_auto.json"
            stale.write_text(
                json.dumps(
                    {
                        "id": "stale_auto",
                        "finding_type": "result",
                        "variant_name": "stale_candidate",
                        "generation_id": 0,
                        "metrics": {
                            "auto_materialized_from_result_artifact": True,
                            "source_result_path": "results/stale/result_summary.json",
                        },
                    }
                ),
                encoding="utf-8",
            )

            findings = _materialize_result_artifacts(run_dir=run_dir, gen_id=0)

        self.assertEqual(findings, [])
        self.assertFalse(stale.exists())

    def test_generation_local_same_name_findings_are_synced_once_without_overwrite(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            shared = run_dir / "shared_findings"
            gen_shared = run_dir / "gen_2" / "shared_findings"
            shared.mkdir(parents=True)
            gen_shared.mkdir(parents=True)
            root_payload = {
                "id": "result",
                "finding_type": "result",
                "variant_name": "root_alpha",
                "peer_id": "gen2_peer0",
                "generation_id": 2,
                "title": "root result",
                "content": "root channel result",
                "metrics": {
                    "mean_active_alpha_vs_benchmark_pct": 1.0,
                    "mean_active_share": 0.2,
                    "tier": "T1",
                },
            }
            local_payload = {
                "id": "result",
                "finding_type": "result",
                "variant_name": "local_alpha",
                "peer_id": "gen2_peer1",
                "generation_id": 2,
                "title": "local result",
                "content": "generation local result",
                "metrics": {
                    "mean_active_alpha_vs_benchmark_pct": 2.0,
                    "mean_active_share": 0.3,
                    "tier": "T1",
                },
            }
            (shared / "result.json").write_text(json.dumps(root_payload), encoding="utf-8")
            (gen_shared / "result.json").write_text(json.dumps(local_payload), encoding="utf-8")

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root / "db")}):
                first = findings_collection.collect_findings_for_generation(
                    findings_dir=shared,
                    gen_id=2,
                    local_mode=True,
                    result_scoring_metric_keys=AIST_RESULT_SCORING_KEYS,
                )
                second = findings_collection.collect_findings_for_generation(
                    findings_dir=shared,
                    gen_id=2,
                    local_mode=True,
                    result_scoring_metric_keys=AIST_RESULT_SCORING_KEYS,
                )

            variants = [row.get("variant_name") for row in first]
            self.assertEqual(variants.count("root_alpha"), 1)
            self.assertEqual(variants.count("local_alpha"), 1)
            self.assertTrue((shared / "result.json").exists())
            self.assertTrue((gen_shared / "result.json").exists())
            self.assertFalse((shared / "gen2_result.json").exists())
            self.assertEqual(
                {row.get("variant_name") for row in second},
                {row.get("variant_name") for row in first},
            )
            self.assertEqual(
                [row.get("variant_name") for row in second].count("local_alpha"),
                1,
            )

    def test_generation_local_finding_without_metadata_inherits_generation_id(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            shared = run_dir / "shared_findings"
            gen_shared = run_dir / "gen_2" / "shared_findings"
            shared.mkdir(parents=True)
            gen_shared.mkdir(parents=True)
            (gen_shared / "finding.json").write_text(
                json.dumps(
                    {
                        "finding_type": "result",
                        "variant_name": "metadata_missing_alpha",
                        "title": "strong local result",
                        "content": "full T1 result written without peer metadata",
                        "metrics": {
                            "mean_active_alpha_vs_benchmark_pct": 3.0,
                            "mean_active_share": 0.4,
                            "tier": "T1",
                            "evidence_stage": "full_T1",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root / "db")}):
                found = findings_collection.collect_findings_for_generation(
                    findings_dir=shared,
                    gen_id=2,
                    local_mode=True,
                    result_scoring_metric_keys=AIST_RESULT_SCORING_KEYS,
                )

            row = next(row for row in found if row.get("variant_name") == "metadata_missing_alpha")
            self.assertEqual(row.get("generation_id"), 2)
            self.assertEqual(row["metrics"]["tier"], "T1")
            self.assertTrue((gen_shared / "finding.json").exists())
            self.assertFalse((shared / "gen2_finding.json").exists())

    def test_result_artifact_requires_reliable_generation_provenance(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            local_store,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            shared = run_dir / "shared_findings"
            result_dir = run_dir / "results" / "late_bridge_child"
            shared.mkdir(parents=True)
            result_dir.mkdir(parents=True)
            summary = {
                "variant_name": "late_bridge_family",
                "tier_reached": "T1",
                "tier_status": "stop_after_T1",
                "final_status": "stop_after_T1",
                "current_aggregate": {
                    "future_fitness": -1.0,
                    "mean_active_alpha_vs_benchmark_pct": 2.0,
                    "mean_active_share": 0.3,
                },
            }
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root / "db")}):
                first = findings_collection.collect_findings_for_generation(
                    findings_dir=shared,
                    gen_id=5,
                    local_mode=True,
                    result_scoring_metric_keys=AIST_RESULT_SCORING_KEYS,
                )
                first_child = [
                    row for row in first if row.get("variant_name") == "late_bridge_child"
                ]
                self.assertEqual(len(first_child), 1)
                self.assertTrue(first_child[0]["metrics"]["source_generation_low_confidence"])
                self.assertTrue(first_child[0]["metrics"]["excluded_from_durable_frontier"])
                self.assertEqual(
                    first_child[0]["metrics"]["exclusion_reason"],
                    "source_generation_low_confidence",
                )

                (shared / "manual_reference.json").write_text(
                    json.dumps(
                        {
                            "id": "manual_reference",
                            "finding_type": "insight",
                            "variant_name": "manual_note",
                            "generation_id": 4,
                            "content": "manual note that points to the result path",
                            "metrics": {
                                "source_result_path": "results/late_bridge_child/tiered_eval_summary.json",
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                unchanged = findings_collection.collect_findings_for_generation(
                    findings_dir=shared,
                    gen_id=5,
                    local_mode=True,
                    result_scoring_metric_keys=AIST_RESULT_SCORING_KEYS,
                )
                self.assertTrue((shared / "manual_reference.json").exists())
                self.assertFalse(
                    any(row.get("variant_name") == "late_bridge_child" for row in unchanged)
                )
                gen4_child = [
                    row
                    for row in local_store.get_findings(generation_id=4)
                    if row.get("variant_name") == "late_bridge_child"
                ]
                self.assertEqual(len(gen4_child), 1)
                self.assertEqual(
                    gen4_child[0]["metrics"]["source_generation_inference"],
                    "root_finding_reference",
                )

                older_gen_shared = run_dir / "gen_1" / "shared_findings"
                older_gen_shared.mkdir(parents=True)
                (older_gen_shared / "old_mention.json").write_text(
                    json.dumps(
                        {
                            "id": "old_mention",
                            "finding_type": "insight",
                            "variant_name": "unrelated_old_note",
                            "generation_id": 1,
                            "peer_id": "gen1_peer0",
                            "content": "late_bridge_child was mentioned in passing",
                            "metrics": {},
                        }
                    ),
                    encoding="utf-8",
                )
                gen_shared = run_dir / "gen_2" / "shared_findings"
                gen_shared.mkdir(parents=True)
                (gen_shared / "late_bridge_child.json").write_text(
                    json.dumps(
                        {
                            "id": "late_child_local",
                            "finding_type": "insight",
                            "variant_name": "late_bridge_family",
                            "generation_id": 2,
                            "peer_id": "gen2_peer3",
                            "content": "late_bridge_child completed inside this generation",
                            "metrics": {
                                "source_result_path": "results/late_bridge_child/tiered_eval_summary.json"
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                gen9_shared = run_dir / "gen_9" / "shared_findings"
                gen9_shared.mkdir(parents=True)
                (gen9_shared / "stale_scored_reference.json").write_text(
                    json.dumps(
                        {
                            "id": "gen9_ref",
                            "finding_type": "insight",
                            "variant_name": "late_bridge_family",
                            "generation_id": 9,
                            "peer_id": "gen9_peer0",
                            "content": "late_bridge_child scored in this older generation",
                            "metrics": {
                                "source_result_path": "results/late_bridge_child/tiered_eval_summary.json"
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                gen10_shared = run_dir / "gen_10" / "shared_findings"
                gen10_shared.mkdir(parents=True)
                (gen10_shared / "newer_scored_reference.json").write_text(
                    json.dumps(
                        {
                            "id": "gen10_ref",
                            "finding_type": "insight",
                            "variant_name": "late_bridge_family",
                            "generation_id": 10,
                            "peer_id": "gen10_peer0",
                            "content": "late_bridge_child scored in the newer generation",
                            "metrics": {
                                "source_result_path": "results/late_bridge_child/tiered_eval_summary.json"
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                second = findings_collection.collect_findings_for_generation(
                    findings_dir=shared,
                    gen_id=5,
                    local_mode=True,
                    result_scoring_metric_keys=AIST_RESULT_SCORING_KEYS,
                )
                self.assertFalse(
                    any(row.get("variant_name") == "late_bridge_child" for row in second)
                )
                corrected = [
                    row
                    for row in local_store.get_findings(generation_id=10)
                    if row.get("variant_name") == "late_bridge_child"
                ]
                self.assertEqual(len(corrected), 1)
                self.assertEqual(
                    corrected[0]["metrics"]["source_generation_inference"],
                    "generation_local_finding_reference",
                )

                gen10_rows = findings_collection.collect_findings_for_generation(
                    findings_dir=shared,
                    gen_id=10,
                    local_mode=True,
                    result_scoring_metric_keys=AIST_RESULT_SCORING_KEYS,
                )
                self.assertTrue(
                    any(row.get("variant_name") == "late_bridge_child" for row in gen10_rows)
                )

    def test_custom_tiered_eval_summary_materializes_canonical_finding(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            shared = run_dir / "shared_findings"
            results_dir = run_dir / "results" / "gen2_peer4_custom_alpha"
            shared.mkdir(parents=True)
            results_dir.mkdir(parents=True)
            result_path = results_dir / "custom_gen2_peer4_custom_alpha_T1_multi_benchmark.json"
            result_path.write_text(
                json.dumps(
                    {
                        "tier": "T1",
                        "promotion_eligible": False,
                        "mean_test_taskscore": 12.0,
                        "mean_active_alpha_vs_benchmark_pct": 4.5,
                    }
                ),
                encoding="utf-8",
            )
            summary_path = results_dir / "custom_gen2_peer4_custom_alpha_tiered_eval_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "schema_version": "sam_optimizer.tiered_eval_summary.v1",
                        "variant_name": "generic_family_name",
                        "final_status": "stopped_at_T1",
                        "tiers": [
                            {
                                "tier": "T1",
                                "returncode": 0,
                                "result_path": str(result_path),
                                "metrics_summary": {
                                    "tier": "T1",
                                    "promotion_eligible": False,
                                    "mean_test_taskscore": 12.0,
                                    "mean_active_alpha_vs_benchmark_pct": 4.5,
                                    "mean_active_share": 0.41,
                                    "scored_cell_count": 29,
                                    "per_dataset": {
                                        "cifar100": {
                                            "test_accuracy": {"mean": 0.71, "std": 0.02},
                                            "train_test_gap": {"mean": 0.09, "std": 0.01},
                                            "num_seeds_ok": 3,
                                        },
                                        "tiny-imagenet": {
                                            "test_accuracy": {"mean": 0.52, "std": 0.03},
                                            "train_test_gap": {"mean": 0.15, "std": 0.02},
                                            "num_seeds_ok": 3,
                                        },
                                    },
                                },
                                "gate": {"passed": False, "reason": "T1 diagnostic"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root / "db")}):
                found = findings_collection.collect_findings_for_generation(
                    findings_dir=shared,
                    gen_id=2,
                    local_mode=True,
                    result_scoring_metric_keys=AIST_RESULT_SCORING_KEYS,
                )

            child = next(
                row for row in found if row.get("variant_name") == "gen2_peer4_custom_alpha"
            )
            self.assertEqual(child["finding_type"], "result")
            self.assertEqual(child["metrics"]["source_result_kind"], summary_path.name)
            self.assertEqual(child["metrics"]["reported_variant_name"], "generic_family_name")
            self.assertRegex(child["metrics"]["source_result_config_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(child["metrics"]["mean_active_alpha_vs_benchmark_pct"], 4.5)
            self.assertEqual(child["metrics"]["evaluation_units"], 29)
            self.assertEqual(child["metrics"]["test_accuracy_cifar100"], 0.71)
            self.assertEqual(child["metrics"]["train_test_gap_cifar100"], 0.09)
            self.assertEqual(child["metrics"]["test_accuracy_tiny_imagenet"], 0.52)
            self.assertEqual(child["metrics"]["train_test_gap_tiny_imagenet"], 0.15)
            self.assertNotIn("per_dataset", child["metrics"])
            self.assertEqual(child["metrics"]["num_seeds_ok_cifar100"], 3)
            self.assertEqual(child["metrics"]["num_seeds_ok_tiny_imagenet"], 3)

    def test_canonical_result_summary_flattens_per_dataset_prompt_metrics(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
        )

        normalized = findings_collection.normalized_result_summary(
            {
                "variant_name": "canonical_sam_variant",
                "current_aggregate": {
                    "mean_test_accuracy": 0.63,
                    "per_dataset": {
                        "cifar100": {
                            "test_accuracy": {"mean": 0.71, "std": 0.02},
                            "train_test_gap": {"mean": 0.09, "std": 0.01},
                            "num_seeds_ok": 3,
                        },
                        "tiny-imagenet": {
                            "test_accuracy": {"mean": 0.52, "std": 0.03},
                            "train_test_gap": {"mean": 0.15, "std": 0.02},
                            "num_seeds_ok": 3,
                        },
                    },
                },
            }
        )
        metrics = findings_collection._result_summary_metrics(normalized)

        self.assertEqual(metrics["mean_test_accuracy"], 0.63)
        self.assertEqual(metrics["test_accuracy_cifar100"], 0.71)
        self.assertEqual(metrics["train_test_gap_cifar100"], 0.09)
        self.assertEqual(metrics["test_accuracy_tiny_imagenet"], 0.52)
        self.assertEqual(metrics["train_test_gap_tiny_imagenet"], 0.15)
        self.assertNotIn("per_dataset", metrics)
        self.assertEqual(metrics["num_seeds_ok_cifar100"], 3)
        self.assertEqual(metrics["num_seeds_ok_tiny_imagenet"], 3)

    def test_task_local_stage_summary_preserves_scores_without_inventing_completion(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            shared = run_dir / "shared_findings"
            results_dir = run_dir / "results" / "gen2_peer4_sam_accuracy"
            shared.mkdir(parents=True)
            results_dir.mkdir(parents=True)
            (results_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "schema_version": "sam_optimizer.tiered_eval_summary.v1",
                        "variant_name": "sam_accuracy",
                        "final_status": "stopped_at_T1",
                        "tiers": [
                            {
                                "tier": "T1",
                                "returncode": 0,
                                "metrics_summary": {
                                    "tier": "T1",
                                    "promotion_eligible": False,
                                    "mean_test_accuracy": 0.63,
                                    "per_dataset": {
                                        "cifar100": {
                                            "test_accuracy": {"mean": 0.71, "std": 0.02},
                                            "train_test_gap": {"mean": 0.09, "std": 0.01},
                                        }
                                    },
                                },
                                "gate": {"passed": False, "reason": "T1 diagnostic"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root / "db")}):
                found = findings_collection.collect_findings_for_generation(
                    findings_dir=shared,
                    gen_id=2,
                    local_mode=True,
                    result_scoring_metric_keys=AIST_RESULT_SCORING_KEYS,
                )

            child = next(
                row for row in found if row.get("variant_name") == "gen2_peer4_sam_accuracy"
            )
            self.assertNotIn("scored_complete", child["metrics"])
            self.assertEqual(child["metrics"]["result_status"], "unknown_maturity")
            self.assertEqual(child["metrics"]["mean_test_accuracy"], 0.63)
            self.assertEqual(child["metrics"]["test_accuracy_cifar100"], 0.71)
            self.assertEqual(child["metrics"]["train_test_gap_cifar100"], 0.09)

    def test_per_dataset_only_summary_is_not_scored_complete(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            shared = run_dir / "shared_findings"
            results_dir = run_dir / "results" / "gen2_peer4_partial_dataset"
            shared.mkdir(parents=True)
            results_dir.mkdir(parents=True)
            (results_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "schema_version": "sam_optimizer.tiered_eval_summary.v1",
                        "variant_name": "partial_dataset",
                        "final_status": "stopped_at_T1",
                        "tiers": [
                            {
                                "tier": "T1",
                                "returncode": 0,
                                "metrics_summary": {
                                    "tier": "T1",
                                    "promotion_eligible": False,
                                    "per_dataset": {
                                        "cifar100": {
                                            "test_accuracy": {"mean": 0.71, "std": 0.02},
                                            "train_test_gap": {"mean": 0.09, "std": 0.01},
                                        }
                                    },
                                },
                                "gate": {"passed": False, "reason": "partial diagnostic"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root / "db")}):
                found = findings_collection.collect_findings_for_generation(
                    findings_dir=shared,
                    gen_id=2,
                    local_mode=True,
                    result_scoring_metric_keys=AIST_RESULT_SCORING_KEYS,
                )

            self.assertFalse(
                any(row.get("variant_name") == "gen2_peer4_partial_dataset" for row in found)
            )

    def test_unscored_structured_negative_result_is_retained_as_validation_signal(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            shared = run_dir / "shared_findings"
            result_dir = run_dir / "results" / "gen2_peer4_failure_signal"
            shared.mkdir(parents=True)
            result_dir.mkdir(parents=True)
            (result_dir / "result_summary.json").write_text(
                json.dumps(
                    {
                        "variant_id": "gen2_peer4_failure_signal",
                        "generation_id": 2,
                        "result_status": "failed",
                        "scored_complete": False,
                        "failure_mode": "solver_did_not_converge",
                        "diagnostic_role": "counterexample",
                        "next_step_intent": "revise_update_rule",
                        "failed_units": [{"unit_id": "case-a", "reason": "nonconvergence"}],
                        "current_aggregate": {
                            "result_status": "failed",
                            "scored_complete": False,
                            "is_negative": True,
                            "failure_mode": "solver_did_not_converge",
                            "diagnostic_role": "counterexample",
                            "next_step_intent": "revise_update_rule",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root / "db")}):
                found = findings_collection.collect_findings_for_generation(
                    findings_dir=shared,
                    gen_id=2,
                    local_mode=True,
                    result_scoring_metric_keys=["score"],
                )

        signal = next(
            row for row in found if row.get("variant_name") == "gen2_peer4_failure_signal"
        )
        self.assertEqual(signal["metrics"]["failure_mode"], "solver_did_not_converge")
        self.assertEqual(signal["metrics"]["diagnostic_role"], "counterexample")
        self.assertTrue(signal["metrics"]["excluded_from_durable_frontier"])
        self.assertFalse(signal["metrics"]["promotion_eligible"])
        self.assertTrue(signal["metrics"]["validation_only_result"])
        self.assertEqual(
            signal["metrics"]["exclusion_reason"],
            "preliminary_or_incomplete_evidence",
        )
        self.assertFalse(any(row.get("variant_name") == "gen2_peer5_failed_noise" for row in found))

    def test_scout_named_tiered_eval_summary_is_not_auto_materialized(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            shared = run_dir / "shared_findings"
            results_dir = run_dir / "results" / "gen2_peer4_scout"
            shared.mkdir(parents=True)
            results_dir.mkdir(parents=True)
            (results_dir / "scout_tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "schema_version": "sam_optimizer.tiered_eval_summary.v1",
                        "variant_name": "scout_alpha",
                        "tiers": [
                            {
                                "tier": "T1",
                                "returncode": 0,
                                "metrics_summary": {
                                    "mean_test_taskscore": 99.0,
                                    "mean_active_alpha_vs_benchmark_pct": 99.0,
                                    "scored_cell_count": 29,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root / "db")}):
                found = findings_collection.collect_findings_for_generation(
                    findings_dir=shared,
                    gen_id=2,
                    local_mode=True,
                    result_scoring_metric_keys=AIST_RESULT_SCORING_KEYS,
                )

            self.assertFalse(any(row.get("variant_name") == "gen2_peer4_scout" for row in found))

    def test_standard_summary_in_scout_result_dir_is_materialized_as_scout_only(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
            FrontierStore,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            shared = run_dir / "shared_findings"
            results_dir = run_dir / "results" / "gen2_peer4_scout"
            shared.mkdir(parents=True)
            results_dir.mkdir(parents=True)
            (results_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "scout_alpha",
                        "tier_reached": "T1",
                        "completed_tier": "T1",
                        "tier_status": "stop_after_T1",
                        "current_aggregate": {
                            "mean_test_taskscore": 99.0,
                            "mean_active_alpha_vs_benchmark_pct": 99.0,
                            "mean_active_share": 0.9,
                            "scored_cell_count": 29,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root / "db")}):
                found = findings_collection.collect_findings_for_generation(
                    findings_dir=shared,
                    gen_id=2,
                    local_mode=True,
                    result_scoring_metric_keys=AIST_RESULT_SCORING_KEYS,
                )

            variants = {row.get("variant_name") for row in found}
            self.assertIn("gen2_peer4_scout", variants)
            scout_row = next(row for row in found if row.get("variant_name") == "gen2_peer4_scout")
            self.assertEqual(scout_row["metrics"]["result_status"], "scout_or_smoke")
            self.assertTrue(scout_row["metrics"]["excluded_from_durable_frontier"])
            self.assertEqual(
                scout_row["metrics"]["exclusion_reason"],
                "preliminary_or_incomplete_evidence",
            )

            store = FrontierStore(
                root / "frontier",
                primary_metric="future_fitness",
                metric_direction="maximize",
                require_tier=True,
                frontier_lanes=[
                    {
                        "name": "alpha_incubator",
                        "k": 10,
                        "exclude_families": ["process_audit"],
                        "allow_lower_tier": True,
                        "allow_non_promotable": True,
                        "allow_risk_violating": True,
                        "require_metrics": [
                            "mean_active_alpha_vs_benchmark_pct",
                            "mean_active_share",
                        ],
                        "axes": [
                            {
                                "name": "mean_active_alpha_vs_benchmark_pct",
                                "direction": "maximize",
                            },
                            {"name": "mean_active_share", "direction": "maximize"},
                        ],
                    }
                ],
            )
            self.assertEqual(store.promote(2, found), [])

    def test_result_artifact_preserves_summary_only_and_not_scored_flags(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            shared = run_dir / "shared_findings"
            results_dir = run_dir / "results" / "gen2_peer4_summary_only"
            shared.mkdir(parents=True)
            results_dir.mkdir(parents=True)
            (results_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "summary_family",
                        "tier_reached": "T1",
                        "completed_tier": "T1",
                        "tier_status": "stop_after_T1",
                        "summary_only": True,
                        "scored_complete": False,
                        "n_eval_cells": 29,
                        "current_aggregate": {
                            "mean_test_taskscore": 99.0,
                            "mean_active_alpha_vs_benchmark_pct": 99.0,
                            "mean_active_share": 0.8,
                            "scored_cell_count": 29,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root / "db")}):
                found = findings_collection.collect_findings_for_generation(
                    findings_dir=shared,
                    gen_id=2,
                    local_mode=True,
                    result_scoring_metric_keys=AIST_RESULT_SCORING_KEYS,
                )

            variants = {row.get("variant_name") for row in found}
            self.assertNotIn("gen2_peer4_summary_only", variants)

    def test_result_artifact_parses_negative_status_text_flags(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            shared = run_dir / "shared_findings"
            shared.mkdir(parents=True)

            capped_dir = run_dir / "results" / "capped_child"
            capped_dir.mkdir(parents=True)
            (capped_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "capped_family",
                        "tier_reached": "T1",
                        "completed_tier": "T1",
                        "tier_status": "capped_at_T1",
                        "current_aggregate": {
                            "mean_test_taskscore": 88.0,
                            "mean_active_alpha_vs_benchmark_pct": 4.0,
                            "mean_active_share": 0.4,
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary_dir = run_dir / "results" / "summary_text_child"
            summary_dir.mkdir(parents=True)
            (summary_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "summary_text_family",
                        "tier_reached": "T1",
                        "completed_tier": "T1",
                        "tier_status": "summary-only",
                        "current_aggregate": {
                            "mean_test_taskscore": 99.0,
                            "mean_active_alpha_vs_benchmark_pct": 5.0,
                            "mean_active_share": 0.5,
                        },
                    }
                ),
                encoding="utf-8",
            )

            not_scored_dir = run_dir / "results" / "not_scored_text_child"
            not_scored_dir.mkdir(parents=True)
            (not_scored_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "not_scored_text_family",
                        "tier_reached": "T1",
                        "completed_tier": "T1",
                        "tier_status": "scored_complete=false",
                        "current_aggregate": {
                            "mean_test_taskscore": 77.0,
                            "mean_active_alpha_vs_benchmark_pct": 3.0,
                            "mean_active_share": 0.3,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root / "db")}):
                found = findings_collection.collect_findings_for_generation(
                    findings_dir=shared,
                    gen_id=2,
                    local_mode=True,
                    result_scoring_metric_keys=AIST_RESULT_SCORING_KEYS,
                )
            variants = {row.get("variant_name") for row in found}
            self.assertIn("capped_child", variants)
            capped = next(row for row in found if row.get("variant_name") == "capped_child")
            self.assertTrue(capped["metrics"]["excluded_from_durable_frontier"])
            self.assertTrue(capped["metrics"]["source_generation_low_confidence"])
            self.assertEqual(
                capped["metrics"]["exclusion_reason"],
                "preliminary_or_incomplete_evidence",
            )
            self.assertNotIn("summary_text_child", variants)
            self.assertIn("not_scored_text_child", variants)
            not_scored = next(
                row for row in found if row.get("variant_name") == "not_scored_text_child"
            )
            self.assertTrue(not_scored["metrics"]["excluded_from_durable_frontier"])
            self.assertEqual(
                not_scored["metrics"]["exclusion_reason"],
                "preliminary_or_incomplete_evidence",
            )


class PromptContextGen0RoleRotationTest(unittest.TestCase):
    def test_build_prompt_context_gen0_role_rotation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            prompt_context,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_spec = SimpleNamespace(
                evaluation=SimpleNamespace(
                    diversity_dimensions=[{"name": "mechanism"}],
                    must_explore_axes=[{"name": "mechanism"}],
                ),
            )
            frontier = SimpleNamespace(get_summary=lambda: [])
            rotation = ("specialist_a", "specialist_b", "specialist_c")
            captured: list[dict] = []
            with (
                patch(
                    "praxist.plugins.graph_maintainers.finding_graph_mvp.engine.build_session_start_graph_context",
                    return_value="",
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.pi_agent.load_agenda_for_gen",
                    return_value=None,
                ),
            ):
                for peer_index in range(4):
                    ctx = prompt_context.build_prompt_context(
                        task_spec=task_spec,
                        workspace=root,
                        run_dir=root / "run",
                        results_dir=root / "results",
                        variants_dir=root / "variants",
                        findings_dir=root / "findings",
                        frontier=frontier,
                        local_mode=True,
                        gen_id=0,
                        peer_index=peer_index,
                        cohort_size=4,
                        strategy="explore",
                        peer_role_rotation=rotation,
                    )
                    captured.append(ctx)

        roles = [
            ctx["research_agenda"]["peer_contracts"][ctx["peer_id"]]["role"] for ctx in captured
        ]
        self.assertEqual(roles, ["specialist_a", "specialist_b", "specialist_c", "specialist_a"])
        sources = {
            ctx["research_agenda"]["peer_contracts"][ctx["peer_id"]]["source"] for ctx in captured
        }
        self.assertEqual(sources, {"gen0_role_rotation"})
        self.assertTrue(
            all(ctx["research_agenda"]["synthesized_from_gen"] == -1 for ctx in captured)
        )

    def test_build_prompt_context_gen0_without_rotation_keeps_free_explore(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            prompt_context,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_spec = SimpleNamespace(
                evaluation=SimpleNamespace(
                    diversity_dimensions=[{"name": "mechanism"}],
                    must_explore_axes=[{"name": "mechanism"}],
                ),
            )
            frontier = SimpleNamespace(get_summary=lambda: [])
            with (
                patch(
                    "praxist.plugins.graph_maintainers.finding_graph_mvp.engine.build_session_start_graph_context",
                    return_value="",
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.pi_agent.load_agenda_for_gen",
                    return_value=None,
                ),
            ):
                ctx = prompt_context.build_prompt_context(
                    task_spec=task_spec,
                    workspace=root,
                    run_dir=root / "run",
                    results_dir=root / "results",
                    variants_dir=root / "variants",
                    findings_dir=root / "findings",
                    frontier=frontier,
                    local_mode=True,
                    gen_id=0,
                    peer_index=0,
                    cohort_size=4,
                    strategy="explore",
                )
        self.assertIsNone(ctx["research_agenda"])

    def test_build_prompt_context_gen1_ignores_role_rotation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            prompt_context,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_spec = SimpleNamespace(
                evaluation=SimpleNamespace(
                    diversity_dimensions=[{"name": "mechanism"}],
                    must_explore_axes=[{"name": "mechanism"}],
                ),
            )
            frontier = SimpleNamespace(get_summary=lambda: [])
            chair_agenda = {
                "peer_contracts": {"gen1_peer0": {"role": "exploit"}},
                "synthesized_from_gen": 0,
            }
            with (
                patch(
                    "praxist.plugins.graph_maintainers.finding_graph_mvp.engine.build_session_start_graph_context",
                    return_value="",
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.pi_agent.load_agenda_for_gen",
                    return_value=chair_agenda,
                ),
            ):
                ctx = prompt_context.build_prompt_context(
                    task_spec=task_spec,
                    workspace=root,
                    run_dir=root / "run",
                    results_dir=root / "results",
                    variants_dir=root / "variants",
                    findings_dir=root / "findings",
                    frontier=frontier,
                    local_mode=True,
                    gen_id=1,
                    peer_index=0,
                    cohort_size=4,
                    strategy="pi_directed",
                    peer_role_rotation=("specialist_a",),
                )
        self.assertEqual(
            ctx["research_agenda"]["peer_contracts"],
            {"gen1_peer0": {"role": "exploit"}},
        )
        self.assertEqual(ctx["research_agenda"]["synthesized_from_gen"], 0)

    def test_build_prompt_context_includes_indirect_action_hypothesis(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            prompt_context,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_spec = SimpleNamespace(
                evaluation=SimpleNamespace(
                    diversity_dimensions=[{"name": "mechanism"}],
                    must_explore_axes=[{"name": "mechanism"}],
                ),
            )
            frontier = SimpleNamespace(get_summary=lambda: [])
            chair_agenda = {
                "synthesized_from_gen": 0,
                "cross_peer_hypotheses": [
                    {
                        "id": f"H{i}",
                        "claim": f"claim {i}",
                        "minimal_test": f"test {i}",
                        "kill_condition": f"kill {i}",
                        "promote_condition": f"promote {i}",
                    }
                    for i in range(1, 11)
                ],
                "consensus_actions": [
                    {
                        "action_id": "A10",
                        "claim_or_hypothesis": "H10",
                        "minimal_experiment": "test the late hypothesis",
                    }
                ],
                "peer_contracts": {
                    "gen1_peer0": {
                        "role": "specialist",
                        "target_hypothesis": "A10",
                        "source": "consensus_actions",
                    }
                },
            }
            with (
                patch(
                    "praxist.plugins.graph_maintainers.finding_graph_mvp.engine.build_session_start_graph_context",
                    return_value="",
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.pi_agent.load_agenda_for_gen",
                    return_value=chair_agenda,
                ),
            ):
                ctx = prompt_context.build_prompt_context(
                    task_spec=task_spec,
                    workspace=root,
                    run_dir=root / "run",
                    results_dir=root / "results",
                    variants_dir=root / "variants",
                    findings_dir=root / "findings",
                    frontier=frontier,
                    local_mode=True,
                    gen_id=1,
                    peer_index=0,
                    cohort_size=1,
                    strategy="pi_directed",
                )

        sliced = ctx["research_agenda"]
        hypothesis_ids = [item["id"] for item in sliced["cross_peer_hypotheses"]]
        self.assertIn("H10", hypothesis_ids)
        self.assertLess(len(hypothesis_ids), len(chair_agenda["cross_peer_hypotheses"]))
        self.assertEqual(
            sliced["current_peer_source_context"]["consensus_actions"][0]["claim_or_hypothesis"],
            "H10",
        )


class PromptContextRuntimeNeutralityTest(unittest.TestCase):
    def _ctx(self, *, runtime_ref: str = "") -> dict[str, object]:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            prompt_context,
        )

        task_spec = SimpleNamespace(
            evaluation=SimpleNamespace(diversity_dimensions=None, must_explore_axes=None),
        )
        frontier = SimpleNamespace(get_summary=lambda: [])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch(
                    "praxist.plugins.graph_maintainers.finding_graph_mvp.engine.build_session_start_graph_context",
                    return_value="",
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.pi_agent.load_agenda_for_gen",
                    return_value=None,
                ),
            ):
                return prompt_context.build_prompt_context(
                    task_spec=task_spec,
                    workspace=root,
                    run_dir=root / "run",
                    results_dir=root / "results",
                    variants_dir=root / "variants",
                    findings_dir=root / "findings",
                    frontier=frontier,
                    local_mode=True,
                    gen_id=0,
                    peer_index=0,
                    cohort_size=1,
                    strategy="explore",
                    runtime_ref=runtime_ref,
                )

    def test_supported_runtimes_only_change_runtime_metadata(self) -> None:
        claude_ctx = self._ctx(runtime_ref="agent_runtime:claude_sdk")
        codex_ctx = self._ctx(runtime_ref="agent_runtime:codex_sdk")

        self.assertEqual(claude_ctx["agent_system"], "claude_sdk")
        self.assertEqual(codex_ctx["agent_system"], "codex_sdk")
        runtime_specific_keys = {
            "agent_system",
            "workspace_dir",
            "run_dir",
            "results_dir",
            "variants_dir",
            "notebook_path",
            "findings_dir",
            "logs_dir",
        }
        self.assertEqual(
            {key: value for key, value in claude_ctx.items() if key not in runtime_specific_keys},
            {key: value for key, value in codex_ctx.items() if key not in runtime_specific_keys},
        )

    def test_default_empty_runtime_ref_falls_back_to_mcp_path(self) -> None:
        ctx = self._ctx()
        self.assertEqual(ctx["agent_system"], "")

    def test_malformed_runtime_ref_falls_back_to_mcp_path(self) -> None:
        ctx = self._ctx(runtime_ref="not_a_proper_ref")
        self.assertEqual(ctx["agent_system"], "")


class PromptBaseTemplateRuntimeNeutralityTest(unittest.TestCase):
    def _render(self, *, runtime_name: str, literature_lookup_enabled: bool = False) -> str:
        import jinja2

        repo_root = (
            Path(__file__).resolve().parents[2]
            / "praxist"
            / "plugins"
            / "workflow_stages"
            / "research_loop"
            / "backend"
        )
        template = (repo_root / "prompt_base.jinja2").read_text(encoding="utf-8")
        env = jinja2.Environment(undefined=jinja2.StrictUndefined)
        ctx = {
            "peer_id": "gen0_peer0",
            "gen_id": 0,
            "logical_gen_id": 0,
            "cohort_size": 1,
            "workspace_dir": "/ws",
            "run_dir": "/run",
            "results_dir": "/run/results",
            "variants_dir": "/run/variants",
            "notebook_path": "/run/notebook.json",
            "findings_dir": "/run/findings",
            "logs_dir": "/run/logs",
            "local_mode": True,
            "graph_session_context": "",
            "research_agenda": None,
            "frontier_summary": [],
            "variant_hint": "",
            "task_spec": SimpleNamespace(),
            "peer_role_descriptions": {},
            "agent_system": runtime_name,
            "literature_lookup_enabled": literature_lookup_enabled,
            "gems_context": {"enabled": False, "gems_count": 0, "bottleneck_reports": []},
        }
        return env.from_string(template).render(**ctx)

    def test_prompt_generation_renders_extra_current_peer_contract_fields(self) -> None:
        import jinja2

        repo_root = (
            Path(__file__).resolve().parents[2]
            / "praxist"
            / "plugins"
            / "workflow_stages"
            / "research_loop"
            / "backend"
        )
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(repo_root)),
            undefined=jinja2.StrictUndefined,
        )
        out = env.get_template("prompt_generation.jinja2").render(
            peer_id="gen2_peer0",
            gen_id=2,
            logical_gen_id=2,
            cohort_size=1,
            local_mode=True,
            peer_role_descriptions={},
            gems_context={"enabled": False, "gem_seeded_baseline_mode": False},
            frontier_summary=[],
            research_agenda={
                "synthesized_from_gen": 1,
                "mainline_observation": {},
                "peer_contracts": {
                    "gen2_peer0": {
                        "role": "bridge",
                        "target_hypothesis": "H1",
                        "success_signal": "finish exact assigned test",
                        "research_plan": "train compact ablation before full run",
                        "minimal_experiment": {
                            "window_count": 3,
                            "protocol": "cheap screening",
                        },
                    }
                },
                "cross_peer_hypotheses": [],
            },
        )

        self.assertIn("Additional peer-specific contract fields", out)
        self.assertIn("research_plan", out)
        self.assertIn("train compact ablation before full run", out)
        self.assertIn("minimal_experiment", out)
        self.assertIn("cheap screening", out)

    def test_supported_runtimes_render_the_same_direct_mcp_guidance(self) -> None:
        outputs = {
            runtime: self._render(runtime_name=runtime) for runtime in ("claude_sdk", "codex_sdk")
        }
        self.assertEqual(outputs["claude_sdk"], outputs["codex_sdk"])
        for out in outputs.values():
            self.assertIn("## MCP Tools Available", out)
            self.assertIn("`mcp__evaluation-tools__share_finding`", out)
            self.assertIn("`mcp__frontier-tools__get_frontier`", out)

    def test_literature_lookup_guidance_renders_when_runtime_tool_set_includes_it(self) -> None:
        rendered: dict[str, tuple[str, str]] = {}
        for runtime in ("claude_sdk", "codex_sdk"):
            rendered[runtime] = (
                self._render(runtime_name=runtime, literature_lookup_enabled=False),
                self._render(runtime_name=runtime, literature_lookup_enabled=True),
            )

        self.assertEqual(rendered["claude_sdk"], rendered["codex_sdk"])
        for without_lookup, with_lookup in rendered.values():
            self.assertNotIn("literature-lookup", without_lookup)
            self.assertIn("mcp__literature-lookup__literature_search", with_lookup)
            self.assertIn("literature_source_guide", with_lookup)
            self.assertIn("bounded lookup", with_lookup)
            self.assertIn("1-3 focused", with_lookup)
            self.assertIn("do not download", with_lookup)
            self.assertIn("current local", with_lookup)

    def test_literature_lookup_prompt_context_uses_effective_default_refs(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.prompt_context import (
            _literature_lookup_enabled,
        )

        self.assertTrue(
            _literature_lookup_enabled(SimpleNamespace(_raw={"praxist_plugins": {"tools": []}}))
        )
        self.assertFalse(
            _literature_lookup_enabled(
                SimpleNamespace(
                    _raw={"praxist_plugins": {"tools": ["tool_server:evaluation_tools"]}}
                )
            )
        )
        self.assertFalse(
            _literature_lookup_enabled(
                SimpleNamespace(_raw={"praxist_plugins": {"tools": []}}),
                available_tool_server_names=[],
            )
        )
        self.assertTrue(
            _literature_lookup_enabled(
                SimpleNamespace(
                    _raw={"praxist_plugins": {"tools": ["tool_server:evaluation_tools"]}}
                ),
                available_tool_server_names=["literature-lookup"],
            )
        )

    def test_supported_runtimes_keep_the_same_finding_write_discipline(self) -> None:
        for runtime in ("claude_sdk", "codex_sdk"):
            out = self._render(runtime_name=runtime)
            self.assertIn("always use `mcp__evaluation-tools__share_finding`", out)
            self.assertIn("Writing a JSON", out)
            self.assertIn("fallback", out)

    def test_stop_signal_section_renders(self) -> None:
        for runtime in ("claude_sdk", "codex_sdk"):
            out = self._render(runtime_name=runtime)
            self.assertIn("Generation Stop Signal", out)
            self.assertIn("STOP_SIGNAL", out)


if __name__ == "__main__":
    unittest.main()
