from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from praxist.plugins.workflow_stages.research_loop.backend.dig.config import (
    DIGLiteConfig,
    QualityDiversityConfig,
)
from praxist.plugins.workflow_stages.research_loop.backend.dig.prompts import (
    _compact_context,
    build_baseline_map_prompt,
    build_candidate_generation_prompt,
)
from praxist.plugins.workflow_stages.research_loop.backend.dig.runner import (
    DIGLiteResult,
    _clear_final_artifacts,
    _known_signatures_from_context,
    _known_texts_from_context,
    _parse_yaml_mapping,
    _planner_call,
    _record_checkpoint,
    run_dig_lite,
)
from praxist.plugins.workflow_stages.research_loop.backend.dig.schema import (
    BaselineMechanismMap,
    CandidatePool,
    CandidateReviews,
    DIGSchemaError,
    SelectedContract,
)
from praxist.plugins.workflow_stages.research_loop.backend.dig.selection import (
    is_near_duplicate_candidate,
    select_quality_diverse_candidate,
)
from praxist.plugins.workflow_stages.research_loop.backend.dig.validator import (
    DIGValidationContext,
    DIGValidationError,
    validate_candidate_pool,
    validate_reviews,
    validate_selected_contract,
    validate_selected_contract_matches_candidate,
)
from praxist.plugins.workflow_stages.research_loop.backend.dig.write_gate import (
    DIGWriteGate,
    PeerPhase,
)


def _candidate(candidate_id: str, family: str, surface: str, intent: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "name": f"{candidate_id}_{family}",
        "mechanism_family": family,
        "intervention_surface": surface,
        "intent": intent,
        "hypothesis": f"{family} mechanism improves the target.",
        "expected_gain_path": "Cleaner implementation signal with explicit diagnostics.",
        "implementation_sketch": {
            "files_to_modify": ["train.py" if surface != "model_def" else "model_def.py"],
            "changes": [f"add {family} change"],
        },
        "diagnostic_prediction": {
            "primary_metric": "primary should improve",
            "secondary_or_safety_metric": "safety should remain stable",
            "internal_signal": "diagnostic should move",
        },
        "risk": {
            "implementation": "low",
            "metric_gaming": "low",
            "silent_bug": "low",
            "compute": "low",
        },
        "ablation_hooks": [f"disable_{candidate_id}"],
        "diversity_signature": {
            "mechanism_family": family,
            "intervention_surface": surface,
            "intent": intent,
        },
    }


def _candidate_pool_dict() -> dict:
    return {
        "candidates": [
            _candidate("C01", "calibration", "train_loop", "repair"),
            _candidate("C02", "architecture", "model_def", "explore"),
            _candidate("C03", "loss_objective", "loss", "exploit"),
            _candidate("C04", "diagnostic_falsifier", "logging", "falsify"),
        ]
    }


def _reviews_dict() -> dict:
    return {
        "reviews": [
            {
                "candidate_id": "C01",
                "scores": {
                    "mechanism_plausibility": 5,
                    "implementability": 5,
                    "diagnostic_clarity": 4,
                    "diversity_value": 4,
                    "shortcut_risk": 1,
                    "silent_bug_risk": 1,
                    "compute_risk": 1,
                },
                "fatal_flaws": [],
                "repair_suggestion": "keep it small",
                "reasoned_summary": "best lane fit",
            },
            {
                "candidate_id": "C02",
                "scores": {
                    "mechanism_plausibility": 4,
                    "implementability": 4,
                    "diagnostic_clarity": 4,
                    "diversity_value": 5,
                    "shortcut_risk": 1,
                    "silent_bug_risk": 1,
                    "compute_risk": 2,
                },
                "fatal_flaws": [],
                "repair_suggestion": "keep shape stable",
                "reasoned_summary": "good adjacent candidate",
            },
            {
                "candidate_id": "C03",
                "scores": {
                    "mechanism_plausibility": 3,
                    "implementability": 4,
                    "diagnostic_clarity": 4,
                    "diversity_value": 3,
                    "shortcut_risk": 2,
                    "silent_bug_risk": 1,
                    "compute_risk": 1,
                },
                "fatal_flaws": [],
                "repair_suggestion": "small coefficient",
                "reasoned_summary": "reasonable exploit",
            },
            {
                "candidate_id": "C04",
                "scores": {
                    "mechanism_plausibility": 3,
                    "implementability": 5,
                    "diagnostic_clarity": 5,
                    "diversity_value": 5,
                    "shortcut_risk": 1,
                    "silent_bug_risk": 1,
                    "compute_risk": 1,
                },
                "fatal_flaws": [],
                "repair_suggestion": "bound scope",
                "reasoned_summary": "diagnostic value",
            },
        ]
    }


def _contract_dict(files_to_modify: list[str] | None = None) -> dict:
    return {
        "selected_candidate_id": "C01",
        "variant_name": "calibration_repair_v1",
        "diversity_cell": {
            "mechanism_family": "calibration",
            "intervention_surface": "train_loop",
            "intent": "repair",
        },
        "mechanism_hypothesis": "A calibrated training signal improves the target.",
        "why_selected": "It is implementable, diagnostic, and lane-compatible.",
        "rejected_alternatives": [
            {"candidate_id": "C02", "reason": "adjacent but larger"},
            {"candidate_id": "C03", "reason": "less diagnostic"},
            {"candidate_id": "C04", "reason": "falsifier, not repair"},
        ],
        "files_to_modify": files_to_modify or ["train.py"],
        "allowed_changes": ["add calibration coefficient", "log diagnostics"],
        "forbidden_changes": [
            "do not modify evaluator",
            "do not change data split",
            "do not change metric calculation",
        ],
        "implementation_plan": [{"step": 1, "action": "edit train.py locally"}],
        "expected_metric_signature": {
            "primary": "primary metric should improve",
            "secondary_or_safety": "safety should not regress",
            "diagnostic": "calibration diagnostic should change",
        },
        "ablation_hooks": ["disable_calibration"],
        "fail_fast_checks": ["loss finite", "schema stable"],
        "contract_amendment_policy": {
            "allowed_reasons": [
                "baseline assumption was wrong",
                "shape or API mismatch makes original implementation impossible",
                "contract would require touching a forbidden path",
            ],
            "required_artifact": "contract_amendment.yaml",
        },
    }


def _validation_context(**overrides) -> DIGValidationContext:
    data = {
        "peer_lane": {
            "mechanism_family_preferences": ["calibration"],
            "intervention_surface_preferences": ["train_loop"],
            "intent_preference": "repair",
        },
        "disallowed_file_rules": ["evaluator.py", "data/split.py"],
        "known_diversity_signatures": set(),
        "known_mechanism_texts": [],
        "duplicate_threshold": 0.82,
    }
    data.update(overrides)
    return DIGValidationContext(**data)


class DIGLiteSchemaValidatorTests(unittest.TestCase):
    def test_dig_is_opt_in_when_task_spec_omits_config(self) -> None:
        self.assertFalse(DIGLiteConfig.from_raw(None).enabled)
        self.assertFalse(DIGLiteConfig.from_raw("bad").enabled)
        self.assertTrue(DIGLiteConfig.from_raw({}).enabled)
        self.assertEqual(DIGLiteConfig().max_total_runtime_minutes, 40)
        self.assertTrue(DIGLiteConfig().enabled_for_generation(0))
        self.assertFalse(DIGLiteConfig().enabled_for_generation(1))

    def test_dig_generation_scope_can_restore_all_generation_planning(self) -> None:
        config = DIGLiteConfig.from_raw({"generation_scope": "all_generations"})

        self.assertEqual(config.generation_scope, "all")
        self.assertTrue(config.enabled_for_generation(0))
        self.assertTrue(config.enabled_for_generation(8))

    def test_quality_diversity_uses_independent_generation_switches(self) -> None:
        dig = DIGLiteConfig.from_raw(
            {"cohort_qd": {"enabled": False, "max_same_intent_fraction": 0.1}}
        )
        config = QualityDiversityConfig.from_task_spec(
            {
                "dig_lite": {"cohort_qd": {"enabled": False}},
                "quality_diversity": {
                    "enabled": True,
                    "initial_generation_enabled": False,
                    "later_generations_enabled": True,
                    "max_same_intent_fraction": 0.55,
                },
            },
            dig_config=dig,
        )

        self.assertFalse(config.enabled_for_generation(0))
        self.assertTrue(config.enabled_for_generation(1))
        self.assertEqual(config.cohort.max_same_intent_fraction, 0.55)
        self.assertEqual(
            config.pi_planning_policy(1)["candidate_source"],
            "existing_pi_synthesis",
        )

    def test_quality_diversity_legacy_nested_config_remains_supported(self) -> None:
        dig = DIGLiteConfig.from_raw(
            {"cohort_qd": {"enabled": True, "max_same_mechanism_family_peers": 2}}
        )
        config = QualityDiversityConfig.from_task_spec(
            {"dig_lite": {"enabled": True}},
            dig_config=dig,
        )

        self.assertTrue(config.enabled_for_generation(0))
        self.assertTrue(config.enabled_for_generation(5))
        self.assertEqual(config.cohort.max_same_mechanism_family_peers, 2)

    def test_legacy_nested_switch_only_disables_cross_peer_allocator(self) -> None:
        dig = DIGLiteConfig.from_raw({"cohort_qd": {"enabled": False}})
        config = QualityDiversityConfig.from_task_spec(
            {"dig_lite": {"enabled": True, "cohort_qd": {"enabled": False}}},
            dig_config=dig,
        )

        self.assertTrue(config.enabled_for_generation(0))
        self.assertTrue(config.enabled_for_generation(1))
        self.assertFalse(config.cohort.enabled)
        self.assertEqual(config.pi_planning_policy(1), {})

    def test_disabled_legacy_dig_does_not_silently_enable_undeclared_qd(self) -> None:
        dig = DIGLiteConfig.from_raw({"enabled": False})
        config = QualityDiversityConfig.from_task_spec(
            {"dig_lite": {"enabled": False}},
            dig_config=dig,
        )

        self.assertFalse(config.enabled_for_generation(0))
        self.assertFalse(config.enabled_for_generation(1))

    def test_disabled_legacy_dig_ignores_nested_cohort_qd(self) -> None:
        dig = DIGLiteConfig.from_raw({"enabled": False, "cohort_qd": {"enabled": True}})
        config = QualityDiversityConfig.from_task_spec(
            {"dig_lite": {"enabled": False, "cohort_qd": {"enabled": True}}},
            dig_config=dig,
        )

        self.assertFalse(config.enabled_for_generation(0))
        self.assertFalse(config.enabled_for_generation(4))
        self.assertEqual(config.pi_planning_policy(4), {})

    def test_dig_innovation_config_parses_task_overrides(self) -> None:
        config = DIGLiteConfig.from_raw(
            {
                "innovation": {
                    "max_diagnostic_fraction": 0.15,
                    "max_diagnostic_peers": 1,
                    "diagnostic_score_penalty": 7,
                    "diagnostic_intents": ["falsify", "control"],
                }
            }
        )

        self.assertTrue(config.innovation.enabled)
        self.assertEqual(config.innovation.max_diagnostic_fraction, 0.15)
        self.assertEqual(config.innovation.max_diagnostic_peers, 1)
        self.assertEqual(config.innovation.diagnostic_score_penalty, 7.0)
        self.assertEqual(config.innovation.diagnostic_intents, ["falsify", "control"])

    def test_dig_timeout_config_separates_phase_attempt_and_total_budget(self) -> None:
        config = DIGLiteConfig.from_raw(
            {
                "planner_max_runtime_minutes": 7,
                "attempt_max_runtime_minutes": 31,
                "max_total_runtime_minutes": 40,
            }
        )

        self.assertEqual(config.planner_max_runtime_minutes, 7)
        self.assertEqual(config.attempt_max_runtime_minutes, 31)
        self.assertEqual(config.max_total_runtime_minutes, 40)

    def test_dig_compact_context_includes_anchor_and_frontier_lane_schema(self) -> None:
        task_spec = SimpleNamespace(
            task_id="task",
            task_name="Task",
            research_direction="direction",
            evaluation=SimpleNamespace(
                primary_metric="future_fitness",
                direction="maximize",
                aux_metrics=["mean_train_test_gap"],
                anchor_metrics=[("sharpness_top_eigen", "minimize")],
                frontier_lanes=[
                    {
                        "name": "alpha",
                        "axes": [("future_fitness", "maximize")],
                        "require_metrics": ["mean_train_test_gap"],
                        "min_metrics": {"future_fitness": 0.0},
                    }
                ],
                diversity_dimensions=[],
                must_explore_axes=[],
            ),
            toolchain=SimpleNamespace(),
        )

        compact = _compact_context({"task_spec": task_spec}, DIGLiteConfig())

        task = compact["task"]
        self.assertEqual(task["anchor_metrics"], [("sharpness_top_eigen", "minimize")])
        self.assertEqual(task["frontier_lanes"][0]["name"], "alpha")
        self.assertEqual(task["frontier_lanes"][0]["require_metrics"], ["mean_train_test_gap"])
        self.assertEqual(task["frontier_lanes"][0]["min_metrics"], {"future_fitness": 0.0})

    def test_dig_known_context_uses_sibling_roster_without_full_contracts(self) -> None:
        ctx = {
            "peer_id": "gen2_peer0",
            "research_agenda": {
                "peer_contracts": {
                    "gen2_peer0": {
                        "target_hypothesis": "own target",
                        "success_signal": "own success",
                    }
                },
                "sibling_roster": [
                    {
                        "peer_id": "gen2_peer1",
                        "target_hypothesis": "sibling target",
                        "mechanism_family": "temporal_credit",
                        "intervention_surface": "loss",
                        "intent": "repair",
                    }
                ],
            },
        }

        self.assertIn(
            ("temporal_credit", "loss", "repair"),
            _known_signatures_from_context(ctx),
        )
        texts = _known_texts_from_context(ctx)
        self.assertIn("sibling target", texts)
        self.assertIn("temporal_credit", texts)

    def test_dig_cohort_qd_config_parses_task_overrides(self) -> None:
        config = DIGLiteConfig.from_raw(
            {
                "cohort_qd": {
                    "enabled": True,
                    "max_same_mechanism_family_peers": 2,
                    "max_same_semantic_family_peers": 1,
                    "max_same_parent_lineage_fraction": 0.25,
                    "max_same_intervention_surface_fraction": 0.25,
                    "target_keyword_groups": [
                        {
                            "name": "architecture_or_representation",
                            "min_peers": 2,
                            "fields": ["mechanism_family", "hypothesis"],
                            "keywords": ["architecture", "encoder"],
                        }
                    ],
                    "semantic_label_groups": [
                        {
                            "name": "score_calibration_family",
                            "fields": ["hypothesis", "changes"],
                            "keywords": ["temperature", "calibration"],
                        }
                    ],
                }
            }
        )

        self.assertTrue(config.cohort_qd.enabled)
        self.assertEqual(config.cohort_qd.max_same_mechanism_family_peers, 2)
        self.assertEqual(config.cohort_qd.max_same_semantic_family_peers, 1)
        self.assertEqual(config.cohort_qd.max_same_parent_lineage_fraction, 0.25)
        self.assertEqual(config.cohort_qd.max_same_intervention_surface_fraction, 0.25)
        self.assertEqual(len(config.cohort_qd.target_keyword_groups), 1)
        self.assertEqual(config.cohort_qd.target_keyword_groups[0].min_peers, 2)
        self.assertEqual(len(config.cohort_qd.semantic_label_groups), 1)

    def test_dig_read_only_prompt_rules_are_task_generic(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.dig.prompts import (
            READ_ONLY_RULES,
        )

        for term in ("scout", "T1", "T2", "T3", "trading", "alpha", "drawdown"):
            self.assertNotIn(term, READ_ONLY_RULES)

    def test_planner_parser_accepts_strict_json_and_json_fence(self) -> None:
        self.assertEqual(
            _parse_yaml_mapping('{"task_objective": {"primary_metric": "score"}}', label="x"),
            {"task_objective": {"primary_metric": "score"}},
        )
        self.assertEqual(
            _parse_yaml_mapping('```json\n{"reviews": []}\n```', label="x"),
            {"reviews": []},
        )

    def test_planner_parser_extracts_final_json_from_noisy_text(self) -> None:
        noisy_output = """
I will inspect the task first.

{"not_the_payload": true}

Here is the final object:
{
  "task_objective": {"primary_metric": "score"},
  "baseline_core_path": [{"file": "train.py", "role": "training"}],
  "intervention_surfaces": [{"name": "train", "files": ["train.py"], "allowed": true}]
}
"""
        parsed = _parse_yaml_mapping(noisy_output, label="baseline_map")

        self.assertEqual(parsed["task_objective"]["primary_metric"], "score")
        self.assertIn("baseline_core_path", parsed)

    def test_planner_parser_uses_labeled_payload_not_first_json(self) -> None:
        noisy_output = """
{"reviews": []}
{"candidates": [{"candidate_id": "C01"}]}
"""
        parsed = _parse_yaml_mapping(noisy_output, label="candidate_pool")

        self.assertEqual(parsed, {"candidates": [{"candidate_id": "C01"}]})

    def test_planner_call_repairs_unparseable_payload_once(self) -> None:
        class FakeAgent:
            calls: list[str] = []
            outputs = ["not a mapping", '{"candidates": []}']

            def __init__(self, label: str):
                self.label = label

            async def execute(self, prompt: str):
                FakeAgent.calls.append(self.label)
                if self.label.endswith("_repair"):
                    assert "Repair the serialization only" in prompt
                return SimpleNamespace(
                    success=True,
                    output={"text": FakeAgent.outputs.pop(0)},
                    duration=0.0,
                    iteration_count=1,
                    error=None,
                )

        parsed = asyncio.run(
            _planner_call(
                prompt="make candidates",
                label="candidate_pool",
                agent_factory=lambda label: FakeAgent(label),
                timeout_seconds=5,
            )
        )

        self.assertEqual(parsed, {"candidates": []})
        self.assertEqual(FakeAgent.calls, ["candidate_pool", "candidate_pool_repair"])

    def test_planner_call_raises_when_repair_is_still_unparseable(self) -> None:
        class FakeAgent:
            calls: list[str] = []
            outputs = ["not a mapping", "still not a mapping"]

            def __init__(self, label: str):
                self.label = label

            async def execute(self, _prompt: str):
                FakeAgent.calls.append(self.label)
                return SimpleNamespace(
                    success=True,
                    output={"text": FakeAgent.outputs.pop(0)},
                    duration=0.0,
                    iteration_count=1,
                    error=None,
                )

        with self.assertRaises(ValueError):
            asyncio.run(
                _planner_call(
                    prompt="make candidates",
                    label="candidate_pool",
                    agent_factory=lambda label: FakeAgent(label),
                    timeout_seconds=5,
                )
            )

        self.assertEqual(FakeAgent.calls, ["candidate_pool", "candidate_pool_repair"])

    def test_candidate_pool_and_reviews_validate_diversity(self) -> None:
        config = DIGLiteConfig(candidate_count=4, min_mechanism_families=4)
        pool = CandidatePool.from_dict(_candidate_pool_dict())
        reviews = CandidateReviews.from_dict(_reviews_dict())

        validate_candidate_pool(pool, config)
        validate_reviews(pool, reviews)

    def test_review_score_outside_range_is_rejected(self) -> None:
        data = _reviews_dict()
        data["reviews"][0]["scores"]["mechanism_plausibility"] = 9
        with self.assertRaises(DIGSchemaError):
            CandidateReviews.from_dict(data)

    def test_selected_contract_rejects_forbidden_paths(self) -> None:
        contract = SelectedContract.from_dict(_contract_dict(["assets/evaluator.py"]))
        with self.assertRaises(DIGValidationError):
            validate_selected_contract(contract, _validation_context(), DIGLiteConfig())

    def test_selected_contract_rejects_absolute_or_traversal_paths(self) -> None:
        for unsafe_path in ("/tmp/train.py", "../other/train.py", "C:/tmp/train.py"):
            with self.subTest(unsafe_path=unsafe_path):
                contract = SelectedContract.from_dict(_contract_dict([unsafe_path]))
                with self.assertRaises(DIGValidationError):
                    validate_selected_contract(contract, _validation_context(), DIGLiteConfig())

    def test_selected_contract_rejects_canonical_task_paths_by_default(self) -> None:
        unsafe_paths = (
            "assets/harness/baseline/model_def.py",
            "assets/harness/env/market_env.py",
            "assets/harness/eval/tiered_eval.py",
            "data/processed/features_all_expanded.parquet",
            "evaluations/trading_pareto/run.py",
            "task.yaml",
        )
        for unsafe_path in unsafe_paths:
            with self.subTest(unsafe_path=unsafe_path):
                contract = SelectedContract.from_dict(_contract_dict([unsafe_path]))
                with self.assertRaises(DIGValidationError):
                    validate_selected_contract(contract, _validation_context(), DIGLiteConfig())

    def test_selected_contract_rejects_lane_mismatch(self) -> None:
        contract_data = _contract_dict()
        contract_data["diversity_cell"]["mechanism_family"] = "architecture"
        contract = SelectedContract.from_dict(contract_data)
        with self.assertRaises(DIGValidationError):
            validate_selected_contract(contract, _validation_context(), DIGLiteConfig())

    def test_selected_contract_allows_qd_adjacent_fallback(self) -> None:
        contract_data = _contract_dict()
        contract_data["selected_candidate_id"] = "C02"
        contract_data["diversity_cell"] = {
            "mechanism_family": "architecture",
            "intervention_surface": "model_def",
            "intent": "explore",
        }
        contract = SelectedContract.from_dict(contract_data)
        validate_selected_contract(
            contract,
            _validation_context(allow_adjacent_lane_selected=True),
            DIGLiteConfig(),
        )

    def test_selected_contract_rejects_near_duplicate_signature(self) -> None:
        contract = SelectedContract.from_dict(_contract_dict())
        ctx = _validation_context(
            known_diversity_signatures={("calibration", "train_loop", "repair")}
        )
        with self.assertRaises(DIGValidationError):
            validate_selected_contract(contract, ctx, DIGLiteConfig())

    def test_selected_contract_must_match_qd_selected_candidate(self) -> None:
        pool = CandidatePool.from_dict(_candidate_pool_dict())
        contract_data = _contract_dict()
        contract_data["selected_candidate_id"] = "C02"
        contract = SelectedContract.from_dict(contract_data)
        with self.assertRaises(DIGValidationError):
            validate_selected_contract_matches_candidate(contract, pool.candidates[0])

    def test_selected_contract_files_must_match_selected_candidate_sketch(self) -> None:
        pool = CandidatePool.from_dict(_candidate_pool_dict())
        contract = SelectedContract.from_dict(_contract_dict(["model_def.py"]))
        with self.assertRaises(DIGValidationError):
            validate_selected_contract_matches_candidate(contract, pool.candidates[0])


class DIGLiteSelectionAndGateTests(unittest.TestCase):
    def test_selection_prefers_best_candidate_inside_peer_lane(self) -> None:
        pool = CandidatePool.from_dict(_candidate_pool_dict())
        reviews = CandidateReviews.from_dict(_reviews_dict())
        selected, _review, qd = select_quality_diverse_candidate(
            pool,
            reviews,
            _validation_context(),
            DIGLiteConfig(candidate_count=4, min_mechanism_families=4),
        )

        self.assertEqual(selected.candidate_id, "C01")
        self.assertEqual(qd.selected_candidate_id, "C01")
        self.assertTrue(qd.eligible_candidates)

    def test_disabled_qd_keeps_quality_selection_without_duplicate_filter(self) -> None:
        pool = CandidatePool.from_dict(_candidate_pool_dict())
        reviews = CandidateReviews.from_dict(_reviews_dict())
        selected, _review, selection = select_quality_diverse_candidate(
            pool,
            reviews,
            _validation_context(
                known_diversity_signatures={
                    ("architecture", "model_def", "explore"),
                }
            ),
            DIGLiteConfig(candidate_count=4, min_mechanism_families=4),
            quality_diversity_enabled=False,
        )

        self.assertEqual(selected.candidate_id, "C01")
        self.assertFalse(selection.quality_diversity_enabled)
        self.assertEqual(selection.cell_elites, [])
        self.assertIn("disabled", selection.selection_reason)

    def test_near_duplicate_filter_uses_task_label_synonyms(self) -> None:
        candidate = CandidatePool.from_dict(
            {"candidates": [_candidate("C99", "network", "model_def", "explore")]}
        ).candidates[0]
        config = DIGLiteConfig.from_raw(
            {"cohort_qd": {"label_synonyms": {"network": "architecture"}}}
        )
        ctx = _validation_context(
            known_diversity_signatures={("architecture", "model_def", "explore")}
        )

        self.assertTrue(is_near_duplicate_candidate(candidate, ctx, config))

    def test_cohort_target_matching_uses_task_label_synonyms(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.dig.cohort_allocator import (
            _matches_target_group,
        )

        candidate = CandidatePool.from_dict(
            {"candidates": [_candidate("C99", "network", "model_def", "explore")]}
        ).candidates[0]
        group = SimpleNamespace(fields=["mechanism_family"], keywords=["architecture"])
        config = DIGLiteConfig.from_raw(
            {"cohort_qd": {"label_synonyms": {"network": "architecture"}}}
        )

        self.assertFalse(_matches_target_group(candidate, group))
        self.assertTrue(_matches_target_group(candidate, group, config))

        canonical_candidate = CandidatePool.from_dict(
            {"candidates": [_candidate("C100", "architecture", "model_def", "explore")]}
        ).candidates[0]
        synonym_group = SimpleNamespace(fields=["mechanism_family"], keywords=["model"])
        reverse_config = DIGLiteConfig.from_raw(
            {"cohort_qd": {"label_synonyms": {"model": "architecture"}}}
        )
        self.assertTrue(_matches_target_group(canonical_candidate, synonym_group, reverse_config))

    def test_forward_innovation_slot_does_not_select_diagnostic_when_forward_survives(self) -> None:
        pool = CandidatePool.from_dict(_candidate_pool_dict())
        reviews_data = _reviews_dict()
        for review in reviews_data["reviews"]:
            if review["candidate_id"] == "C04":
                review["scores"] = {
                    "mechanism_plausibility": 5,
                    "implementability": 5,
                    "diagnostic_clarity": 5,
                    "diversity_value": 5,
                    "shortcut_risk": 1,
                    "silent_bug_risk": 1,
                    "compute_risk": 1,
                }
        reviews = CandidateReviews.from_dict(reviews_data)

        selected, _review, qd = select_quality_diverse_candidate(
            pool,
            reviews,
            _validation_context(
                peer_lane={},
                selection_policy={"intent_slot": "forward_innovation"},
            ),
            DIGLiteConfig(candidate_count=4, min_mechanism_families=4),
        )

        self.assertNotEqual(selected.candidate_id, "C04")
        selected_trace = {
            item["candidate_id"]: item for item in qd.to_dict()["eligible_candidates"]
        }
        self.assertTrue(selected_trace["C04"]["diagnostic_like"])
        self.assertIn("forward-innovation", qd.selection_reason)

    def test_diagnostic_slot_can_select_diagnostic_candidate(self) -> None:
        pool = CandidatePool.from_dict(_candidate_pool_dict())
        reviews_data = _reviews_dict()
        for review in reviews_data["reviews"]:
            if review["candidate_id"] == "C04":
                review["scores"] = {
                    "mechanism_plausibility": 5,
                    "implementability": 5,
                    "diagnostic_clarity": 5,
                    "diversity_value": 5,
                    "shortcut_risk": 1,
                    "silent_bug_risk": 1,
                    "compute_risk": 1,
                }
        reviews = CandidateReviews.from_dict(reviews_data)

        selected, _review, qd = select_quality_diverse_candidate(
            pool,
            reviews,
            _validation_context(
                peer_lane={},
                selection_policy={"intent_slot": "diagnostic"},
            ),
            DIGLiteConfig(candidate_count=4, min_mechanism_families=4),
        )

        self.assertEqual(selected.candidate_id, "C04")
        self.assertIn("diagnostic/control", qd.selection_reason)

    def test_selection_filters_candidates_that_modify_canonical_task_paths(self) -> None:
        pool_data = _candidate_pool_dict()
        pool_data["candidates"][0]["implementation_sketch"]["files_to_modify"] = [
            "assets/harness/baseline/model_def.py"
        ]
        pool = CandidatePool.from_dict(pool_data)
        reviews = CandidateReviews.from_dict(_reviews_dict())
        selected, _review, qd = select_quality_diverse_candidate(
            pool,
            reviews,
            _validation_context(allow_adjacent_lane_selected=True),
            DIGLiteConfig(candidate_count=4, min_mechanism_families=4),
        )

        self.assertNotEqual(selected.candidate_id, "C01")
        rejected = {item["candidate_id"]: item["reason"] for item in qd.rejected_close_alternatives}
        self.assertEqual(rejected.get("C01"), "violates file rules")

    def test_canonical_label_normalizer_uses_task_local_semantic_groups(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.dig.labels import (
            canonical_labels_for_candidate,
        )

        pool = CandidatePool.from_dict(_candidate_pool_dict())
        config = DIGLiteConfig.from_raw(
            {
                "cohort_qd": {
                    "label_synonyms": {
                        "train_loop": "optimization",
                    },
                    "semantic_label_groups": [
                        {
                            "name": "score_calibration_family",
                            "fields": ["hypothesis", "changes"],
                            "keywords": ["calibration"],
                        }
                    ],
                    "parent_lineage_label_groups": [
                        {
                            "name": "baseline_lineage",
                            "fields": ["hypothesis"],
                            "keywords": ["target"],
                        }
                    ],
                }
            }
        )

        labels = canonical_labels_for_candidate(pool.candidates[0], config)

        self.assertEqual(labels.canonical_semantic_family, "score_calibration_family")
        self.assertEqual(labels.canonical_parent_lineage, "baseline_lineage")
        self.assertEqual(labels.formal_cell(), ("calibration", "optimization", "repair"))

    def test_canonical_label_normalizer_does_not_embed_ml_synonyms_by_default(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.dig.labels import (
            canonical_labels_for_candidate,
        )

        pool = CandidatePool.from_dict(_candidate_pool_dict())

        labels = canonical_labels_for_candidate(pool.candidates[0], DIGLiteConfig())

        self.assertEqual(labels.formal_cell(), ("calibration", "train_loop", "repair"))

    def test_write_gate_blocks_code_and_shell_until_contract_unlock(self) -> None:
        config = DIGLiteConfig()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = DIGWriteGate(root / "run/gen_0/peers/gen0_peer0/dig", config)
            self.assertTrue(gate.can_write(root / "run/gen_0/peers/gen0_peer0/dig/x.yaml"))
            self.assertFalse(
                gate.can_write(root / "run/gen_0/peers/gen0_peer0/dig/../../escape.yaml")
            )
            self.assertFalse(gate.can_write(root / "run/variants/v/train.py"))
            self.assertFalse(gate.can_write(root / "run/results/v/result.json"))
            self.assertFalse(gate.can_run_command("python train.py"))

            contract = SelectedContract.from_dict(_contract_dict())
            gate.unlock(contract, _validation_context())
            self.assertEqual(gate.phase, PeerPhase.IMPLEMENTATION)
            self.assertTrue(gate.can_write(root / "run/variants/v/train.py"))
            self.assertTrue(gate.can_run_command("python train.py"))

    def test_cohort_qd_allocator_spreads_same_family_local_selections(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.dig.cohort_allocator import (
            allocate_cohort_qd_contracts,
        )

        pool = CandidatePool.from_dict(_candidate_pool_dict())
        reviews = CandidateReviews.from_dict(_reviews_dict())
        config = DIGLiteConfig.from_raw(
            {
                "candidate_count": 4,
                "min_mechanism_families": 4,
                "cohort_qd": {
                    "enabled": True,
                    "max_same_mechanism_family_peers": 1,
                    "max_same_diversity_cell_peers": 1,
                },
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = {}
            for index in range(2):
                dig_dir = root / "run" / "gen_0" / "peers" / f"gen0_peer{index}" / "dig"
                dig_dir.mkdir(parents=True)
                path = dig_dir / "selected_contract.yaml"
                contract = SelectedContract.from_dict(_contract_dict())
                path.write_text(yaml.safe_dump(contract.to_dict()), encoding="utf-8")
                results[index] = DIGLiteResult(
                    dig_dir=dig_dir,
                    selected_contract=contract,
                    selected_contract_path=path,
                    qd_selection={"selected_candidate_id": "C01"},
                    candidate_pool=pool,
                    candidate_reviews=reviews,
                    validation_context=_validation_context(),
                )

            updated = allocate_cohort_qd_contracts(
                dig_results=results,
                config=config,
                gen_dir=root / "run" / "gen_0",
            )

            selected_families = {
                result.selected_contract.diversity_cell.mechanism_family
                for result in updated.values()
            }
            self.assertIn("calibration", selected_families)
            self.assertIn("architecture", selected_families)
            self.assertTrue((root / "run" / "gen_0" / "dig_cohort_allocation.yaml").exists())
            self.assertEqual(
                yaml.safe_load(results[1].selected_contract_path.read_text(encoding="utf-8"))[
                    "selected_candidate_id"
                ],
                "C02",
            )
            persisted = yaml.safe_load(
                results[1].selected_contract_path.read_text(encoding="utf-8")
            )
            self.assertTrue(persisted["dig_provenance"]["cohort_qd_changed"])
            self.assertEqual(persisted["dig_provenance"]["local_selected_candidate_id"], "C01")
            self.assertIn("canonical_labels", persisted)

    def test_cohort_qd_caps_semantic_family_not_only_raw_family(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.dig.cohort_allocator import (
            allocate_cohort_qd_contracts,
        )

        pool_data = {
            "candidates": [
                {
                    **_candidate("C01", "calibration", "train_loop", "repair"),
                    "semantic_family": "shared_parent",
                },
                {
                    **_candidate("C02", "architecture", "model_def", "explore"),
                    "semantic_family": "shared_parent",
                },
                {
                    **_candidate("C03", "loss_objective", "loss", "exploit"),
                    "semantic_family": "distinct_objective",
                },
                {
                    **_candidate("C04", "diagnostic_falsifier", "logging", "falsify"),
                    "semantic_family": "diagnostic",
                },
            ]
        }
        reviews_data = _reviews_dict()
        for review in reviews_data["reviews"]:
            if review["candidate_id"] == "C02":
                review["scores"]["mechanism_plausibility"] = 5
                review["scores"]["implementability"] = 5
            if review["candidate_id"] == "C03":
                review["scores"]["mechanism_plausibility"] = 4
        pool = CandidatePool.from_dict(pool_data)
        reviews = CandidateReviews.from_dict(reviews_data)
        config = DIGLiteConfig.from_raw(
            {
                "candidate_count": 4,
                "min_mechanism_families": 4,
                "cohort_qd": {
                    "enabled": True,
                    "max_same_diversity_cell_peers": 2,
                    "max_same_semantic_family_peers": 1,
                },
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = {}
            for index in range(2):
                dig_dir = root / "run" / "gen_0" / "peers" / f"gen0_peer{index}" / "dig"
                dig_dir.mkdir(parents=True)
                path = dig_dir / "selected_contract.yaml"
                contract = SelectedContract.from_dict(_contract_dict())
                path.write_text(yaml.safe_dump(contract.to_dict()), encoding="utf-8")
                results[index] = DIGLiteResult(
                    dig_dir=dig_dir,
                    selected_contract=contract,
                    selected_contract_path=path,
                    qd_selection={"selected_candidate_id": "C01"},
                    candidate_pool=pool,
                    candidate_reviews=reviews,
                    validation_context=_validation_context(allow_adjacent_lane_selected=True),
                )

            updated = allocate_cohort_qd_contracts(
                dig_results=results,
                config=config,
                gen_dir=root / "run" / "gen_0",
            )

            semantic_families = {
                result.selected_contract.canonical_labels["canonical_semantic_family"]
                for result in updated.values()
            }
            self.assertIn("shared_parent", semantic_families)
            self.assertIn("distinct_objective", semantic_families)


class DIGLiteRunnerTests(unittest.TestCase):
    def test_fresh_attempt_clears_stale_final_qd_views(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dig_dir = Path(tmp)
            for name in ("qd_selection.yaml", "selected_contract.yaml", "dig_summary.md"):
                (dig_dir / name).write_text("stale\n", encoding="utf-8")
            (dig_dir / "candidate_pool.yaml").write_text("keep\n", encoding="utf-8")

            _clear_final_artifacts(dig_dir)

            self.assertFalse((dig_dir / "qd_selection.yaml").exists())
            self.assertFalse((dig_dir / "selected_contract.yaml").exists())
            self.assertFalse((dig_dir / "dig_summary.md").exists())
            self.assertTrue((dig_dir / "candidate_pool.yaml").exists())

    def test_dig_context_includes_validation_signals_for_prompt_and_dedup(self) -> None:
        ctx = {
            "peer_id": "gen1_peer0",
            "gen_id": 1,
            "frontier_summary": [],
            "validation_candidates": [
                {
                    "variant_name": "partial_signal",
                    "mechanism_family": "low_rank_adapter",
                    "intervention_surface": "optimizer",
                    "intent": "repair",
                    "recommended_next_step": "complete_scored_validation",
                    "metrics": {"score": 1.2},
                }
            ],
            "validation_candidates_meta": {"returned": 1, "truncated": False, "cap": 16},
        }

        compact = _compact_context(ctx, DIGLiteConfig())
        signatures = _known_signatures_from_context(ctx)
        texts = _known_texts_from_context(ctx)

        self.assertEqual(compact["validation_candidates"][0]["variant_name"], "partial_signal")
        self.assertIn(
            ("low_rank_adapter", "optimizer", "repair"),
            signatures,
        )
        self.assertIn("partial_signal", texts)
        self.assertIn("complete_scored_validation", texts)

    def test_claude_sdk_no_shell_runtime_restricts_available_tools(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import adapter as claude_adapter
        from praxist.plugins.agent_runtimes.claude_sdk.adapter import (
            ClaudeSdkAgentRuntime,
            LegacyClaudeRuntimeOptions,
        )

        captured_options: dict = {}

        class FakeResultMessage:
            pass

        class FakeClaudeAgentOptions:
            def __init__(self, **kwargs):
                captured_options.update(kwargs)

        class FakeHookMatcher:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        async def fake_query(prompt, options):
            yield FakeResultMessage()

        fake_sdk = {
            "ClaudeAgentOptions": FakeClaudeAgentOptions,
            "HookMatcher": FakeHookMatcher,
            "query": fake_query,
            "AssistantMessage": None,
            "ResultMessage": FakeResultMessage,
            "ToolUseBlock": None,
        }

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(claude_adapter, "_load_claude_sdk", return_value=fake_sdk),
        ):
            workspace = Path(tmp)
            result = asyncio.run(
                ClaudeSdkAgentRuntime().execute_legacy(
                    "plan",
                    LegacyClaudeRuntimeOptions(
                        name="dig",
                        allowed_tools=["Read", "Grep", "Glob"],
                        workspace=workspace,
                        mcp_servers={"frontier-tools": object()},
                        model="fake",
                        permission_mode="acceptEdits",
                        require_no_shell_runtime=True,
                    ),
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(captured_options["tools"], ["Read", "Grep", "Glob"])
        self.assertEqual(captured_options["mcp_servers"], {})
        self.assertEqual(captured_options["permission_mode"], "default")
        self.assertIn("Never request Bash", captured_options["system_prompt"])
        self.assertNotIn("BASH_ENV", captured_options["env"])
        self.assertFalse((workspace / ".runtime_guards").exists())
        self.assertFalse((workspace / "peer_workspaces").exists())
        self.assertIn("Bash", captured_options["disallowed_tools"])
        self.assertIn("Write", captured_options["disallowed_tools"])
        self.assertIn("Edit", captured_options["disallowed_tools"])
        bash_hook = captured_options["hooks"]["PreToolUse"][0].kwargs["hooks"][0]
        denied = asyncio.run(
            bash_hook(
                {
                    "tool_name": "Bash",
                    "tool_input": {"command": "echo should-not-run"},
                },
                None,
                None,
            )
        )
        hook_output = denied["hookSpecificOutput"]
        self.assertEqual(hook_output["permissionDecision"], "deny")
        self.assertIn("read-only planner denied", hook_output["permissionDecisionReason"])

    def test_base_agent_can_carry_dig_read_only_sandbox_intent(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.agent import BaseAgent

        with tempfile.TemporaryDirectory() as tmp:
            agent = BaseAgent(
                name="dig-planner",
                allowed_tools=["Read", "Grep", "Glob"],
                workspace=Path(tmp),
                mcp_servers={},
                model="fake",
                runtime_sandbox_intent={
                    "filesystem": "read_only",
                    "network": "on",
                    "approval": "auto",
                },
                runtime_timeout_seconds=123,
                runtime_output_schema={"type": "object"},
                require_no_shell_runtime=True,
                require_read_only_runtime=True,
            )
            request = agent._build_agent_run_request("plan only", {})  # noqa: SLF001

        self.assertEqual(
            request.runtime_options["sandbox_intent"],
            {"filesystem": "read_only", "network": "on", "approval": "auto"},
        )
        self.assertEqual(request.timeout_seconds, 123)
        self.assertEqual(request.runtime_options["output_schema"], {"type": "object"})
        self.assertIs(request.runtime_options["require_no_shell_runtime"], True)
        self.assertIs(request.runtime_options["require_read_only_runtime"], True)

    def test_runner_hardens_planner_against_permissive_config(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.agent import BaseAgent

        captured_requests = []
        baseline_map = {
            "task_objective": {"primary_metric": "score"},
            "baseline_core_path": [{"file": "train.py", "role": "training"}],
            "intervention_surfaces": [
                {"name": "train", "files": ["train.py"], "allowed": True},
                {"name": "model", "files": ["model_def.py"], "allowed": True},
                {"name": "logging", "files": ["train.py"], "allowed": True},
            ],
            "forbidden_surfaces": [
                {"name": "evaluator", "files": ["evaluator.py"], "allowed": False}
            ],
        }
        outputs = [
            yaml.safe_dump(baseline_map),
            yaml.safe_dump(_candidate_pool_dict()),
            yaml.safe_dump(_reviews_dict()),
            yaml.safe_dump(_contract_dict()),
        ]

        async def fake_execute(self, _prompt: str):
            captured_requests.append(
                self._build_agent_run_request("plan only", {})  # noqa: SLF001
            )
            return SimpleNamespace(
                success=True,
                output={"text": outputs.pop(0)},
                duration=0.0,
                iteration_count=1,
                error=None,
            )

        config = DIGLiteConfig(
            candidate_count=4,
            min_mechanism_families=4,
            shell_allowed=True,
            planner_allowed_tools=["Read", "Bash", "Write", "Glob", "Read"],
            planner_permission_mode="acceptEdits",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(BaseAgent, "execute", fake_execute):
                asyncio.run(
                    run_dig_lite(
                        ctx={"peer_id": "gen0_peer0", "gen_id": 0},
                        config=config,
                        dig_dir=root / "run/gen_0/peers/gen0_peer0/dig",
                        workspace=root,
                        model="fake",
                        mcp_servers={"unsafe-shell-server": object()},
                        plugin_registry=None,
                        reasoning_effort="high",
                    )
                )

        self.assertEqual(len(captured_requests), 4)
        for request in captured_requests:
            self.assertEqual(request.tool_permissions.allowed_tools, ["Read", "Glob"])
            self.assertEqual(request.tool_servers, [])
            self.assertEqual(request.runtime_options["permission_mode"], "default")
            self.assertEqual(request.runtime_options["output_schema"]["type"], "object")
            self.assertEqual(request.runtime_options["reasoning_effort"], "high")
            self.assertIs(request.runtime_options["require_no_shell_runtime"], True)
            self.assertNotIn("require_read_only_runtime", request.runtime_options)

    def test_runner_uses_read_only_runtime_contract_for_codex_planner(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.agent import BaseAgent

        captured_requests = []
        baseline_map = {
            "task_objective": {"primary_metric": "score"},
            "baseline_core_path": [{"file": "train.py", "role": "training"}],
            "intervention_surfaces": [
                {"name": "train", "files": ["train.py"], "allowed": True},
                {"name": "model", "files": ["model_def.py"], "allowed": True},
                {"name": "logging", "files": ["train.py"], "allowed": True},
            ],
            "forbidden_surfaces": [
                {"name": "evaluator", "files": ["evaluator.py"], "allowed": False}
            ],
        }
        outputs = [
            yaml.safe_dump(baseline_map),
            yaml.safe_dump(_candidate_pool_dict()),
            yaml.safe_dump(_reviews_dict()),
            yaml.safe_dump(_contract_dict()),
        ]

        async def fake_execute(self, _prompt: str):
            captured_requests.append(
                self._build_agent_run_request("plan only", {})  # noqa: SLF001
            )
            return SimpleNamespace(
                success=True,
                output={"text": outputs.pop(0)},
                duration=0.0,
                iteration_count=1,
                error=None,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(BaseAgent, "execute", fake_execute):
                asyncio.run(
                    run_dig_lite(
                        ctx={
                            "peer_id": "gen0_peer0",
                            "gen_id": 0,
                            "agent_runtime_ref": "agent_runtime:codex_sdk",
                        },
                        config=DIGLiteConfig(candidate_count=4, min_mechanism_families=4),
                        dig_dir=root / "run/gen_0/peers/gen0_peer0/dig",
                        workspace=root,
                        model="fake",
                        mcp_servers={},
                        plugin_registry=None,
                    )
                )

        self.assertEqual(len(captured_requests), 4)
        for request in captured_requests:
            self.assertEqual(request.tool_permissions.allowed_tools, ["Read", "Grep", "Glob"])
            self.assertEqual(request.runtime_options["sandbox_intent"]["filesystem"], "read_only")
            self.assertEqual(request.runtime_options["output_schema"]["type"], "object")
            self.assertNotIn("require_no_shell_runtime", request.runtime_options)
            self.assertIs(request.runtime_options["require_read_only_runtime"], True)

    def test_runner_request_timeout_is_capped_by_remaining_total_budget(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.agent import BaseAgent

        captured_timeouts: list[int] = []
        baseline_map = {
            "task_objective": {"primary_metric": "score"},
            "baseline_core_path": [{"file": "train.py", "role": "training"}],
            "intervention_surfaces": [
                {"name": "train", "files": ["train.py"], "allowed": True},
                {"name": "model", "files": ["model_def.py"], "allowed": True},
                {"name": "logging", "files": ["train.py"], "allowed": True},
            ],
            "forbidden_surfaces": [
                {"name": "evaluator", "files": ["evaluator.py"], "allowed": False}
            ],
        }
        outputs = [
            yaml.safe_dump(baseline_map),
            yaml.safe_dump(_candidate_pool_dict()),
            yaml.safe_dump(_reviews_dict()),
            yaml.safe_dump(_contract_dict()),
        ]

        async def fake_execute(self, _prompt: str):
            request = self._build_agent_run_request("plan only", {})  # noqa: SLF001
            captured_timeouts.append(request.timeout_seconds)
            return SimpleNamespace(
                success=True,
                output={"text": outputs.pop(0)},
                duration=0.0,
                iteration_count=1,
                error=None,
            )

        config = DIGLiteConfig(
            candidate_count=4,
            min_mechanism_families=4,
            planner_max_runtime_minutes=10,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(BaseAgent, "execute", fake_execute):
                asyncio.run(
                    run_dig_lite(
                        ctx={
                            "peer_id": "gen0_peer0",
                            "gen_id": 0,
                            "dig_remaining_budget_seconds": 2,
                            "peer_lane": {
                                "mechanism_family_preferences": ["calibration"],
                                "intervention_surface_preferences": ["train_loop"],
                                "intent_preference": "repair",
                            },
                        },
                        config=config,
                        dig_dir=root / "run/gen_0/peers/gen0_peer0/dig",
                        workspace=root,
                        model="fake",
                        mcp_servers={},
                        plugin_registry=None,
                    )
                )

        self.assertEqual(len(captured_timeouts), 4)
        self.assertTrue(all(0 < value <= 2 for value in captured_timeouts))

    def test_runner_aligns_contract_identity_to_qd_selected_candidate(self) -> None:
        class FakeAgent:
            outputs: list[str] = []

            def __init__(self, label: str):
                self.label = label

            async def execute(self, _prompt: str):
                return SimpleNamespace(
                    success=True,
                    output={"text": FakeAgent.outputs.pop(0)},
                    duration=0.0,
                    iteration_count=1,
                    error=None,
                )

        baseline_map = {
            "task_objective": {"primary_metric": "score"},
            "baseline_core_path": [{"file": "train.py", "role": "training"}],
            "intervention_surfaces": [
                {"name": "train", "files": ["train.py"], "allowed": True},
                {"name": "model", "files": ["model_def.py"], "allowed": True},
                {"name": "logging", "files": ["train.py"], "allowed": True},
            ],
        }
        contract = _contract_dict()
        contract["selected_candidate_id"] = "C99"
        contract["diversity_cell"] = {
            "mechanism_family": "wrong",
            "intervention_surface": "wrong",
            "intent": "explore",
        }
        FakeAgent.outputs = [
            yaml.safe_dump(baseline_map),
            yaml.safe_dump(_candidate_pool_dict()),
            yaml.safe_dump(_reviews_dict()),
            yaml.safe_dump(contract),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = asyncio.run(
                run_dig_lite(
                    ctx={
                        "peer_id": "gen0_peer0",
                        "gen_id": 0,
                        "peer_lane": {
                            "mechanism_family_preferences": ["calibration"],
                            "intervention_surface_preferences": ["train_loop"],
                            "intent_preference": "repair",
                        },
                    },
                    config=DIGLiteConfig(candidate_count=4, min_mechanism_families=4),
                    dig_dir=root / "run/gen_0/peers/gen0_peer0/dig",
                    workspace=root,
                    model="fake",
                    mcp_servers={},
                    plugin_registry=None,
                    agent_factory=lambda label: FakeAgent(label),
                )
            )

        self.assertEqual(result.selected_contract.selected_candidate_id, "C01")
        self.assertEqual(
            result.selected_contract.diversity_cell.as_tuple(),
            ("calibration", "train_loop", "repair"),
        )

    def test_runner_checkpoints_phases_but_not_final_contract_before_validation(self) -> None:
        class FakeAgent:
            outputs: list[str] = []

            def __init__(self, label: str):
                self.label = label

            async def execute(self, _prompt: str):
                return SimpleNamespace(
                    success=True,
                    output={"text": FakeAgent.outputs.pop(0)},
                    duration=0.0,
                    iteration_count=1,
                    error=None,
                )

        baseline_map = {
            "task_objective": {"primary_metric": "score"},
            "baseline_core_path": [{"file": "train.py", "role": "training"}],
            "intervention_surfaces": [
                {"name": "train", "files": ["train.py"], "allowed": True},
                {"name": "model", "files": ["model_def.py"], "allowed": True},
                {"name": "logging", "files": ["train.py"], "allowed": True},
            ],
            "forbidden_surfaces": [
                {"name": "evaluator", "files": ["evaluator.py"], "allowed": False}
            ],
        }
        invalid_contract = _contract_dict(["assets/evaluator.py"])
        FakeAgent.outputs = [
            yaml.safe_dump(baseline_map),
            yaml.safe_dump(_candidate_pool_dict()),
            yaml.safe_dump(_reviews_dict()),
            yaml.safe_dump(invalid_contract),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dig_dir = root / "run/gen_0/peers/gen0_peer0/dig"
            dig_dir.mkdir(parents=True)
            (dig_dir / "selected_contract.yaml").write_text("stale\n", encoding="utf-8")
            (dig_dir / "dig_summary.md").write_text("stale\n", encoding="utf-8")

            with self.assertRaises(DIGValidationError):
                asyncio.run(
                    run_dig_lite(
                        ctx={
                            "peer_id": "gen0_peer0",
                            "gen_id": 0,
                            "peer_lane": {
                                "mechanism_family_preferences": ["calibration"],
                                "intervention_surface_preferences": ["train_loop"],
                                "intent_preference": "repair",
                            },
                        },
                        config=DIGLiteConfig(candidate_count=4, min_mechanism_families=4),
                        dig_dir=dig_dir,
                        workspace=root,
                        model="fake",
                        mcp_servers={},
                        plugin_registry=None,
                        agent_factory=lambda label: FakeAgent(label),
                    )
                )

            for name in (
                "baseline_mechanism_map.yaml",
                "candidate_pool.yaml",
                "candidate_reviews.yaml",
                "qd_selection.yaml",
            ):
                self.assertTrue((dig_dir / name).exists(), name)
            self.assertFalse((dig_dir / "selected_contract.yaml").exists())
            self.assertFalse((dig_dir / "dig_summary.md").exists())
            status = yaml.safe_load((dig_dir / "dig_stage_status.json").read_text())
            self.assertEqual(status["last_phase"], "selected_contract")

    def test_runner_resumes_from_valid_checkpoint_artifacts(self) -> None:
        class FakeAgent:
            outputs: list[str] = []
            calls: list[str] = []

            def __init__(self, label: str):
                self.label = label

            async def execute(self, _prompt: str):
                FakeAgent.calls.append(self.label)
                return SimpleNamespace(
                    success=True,
                    output={"text": FakeAgent.outputs.pop(0)},
                    duration=0.0,
                    iteration_count=1,
                    error=None,
                )

        baseline_map = {
            "task_objective": {"primary_metric": "score"},
            "baseline_core_path": [{"file": "train.py", "role": "training"}],
            "intervention_surfaces": [
                {"name": "train", "files": ["train.py"], "allowed": True},
                {"name": "model", "files": ["model_def.py"], "allowed": True},
                {"name": "logging", "files": ["train.py"], "allowed": True},
            ],
            "forbidden_surfaces": [
                {"name": "evaluator", "files": ["evaluator.py"], "allowed": False}
            ],
        }
        FakeAgent.outputs = [
            yaml.safe_dump(_reviews_dict()),
            yaml.safe_dump(_contract_dict()),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dig_dir = root / "run/gen_0/peers/gen0_peer0/dig"
            dig_dir.mkdir(parents=True)
            config = DIGLiteConfig(candidate_count=4, min_mechanism_families=4)
            ctx = {
                "peer_id": "gen0_peer0",
                "gen_id": 0,
                "peer_lane": {
                    "mechanism_family_preferences": ["calibration"],
                    "intervention_surface_preferences": ["train_loop"],
                    "intent_preference": "repair",
                },
            }
            (dig_dir / "baseline_mechanism_map.yaml").write_text(
                yaml.safe_dump(baseline_map),
                encoding="utf-8",
            )
            (dig_dir / "candidate_pool.yaml").write_text(
                yaml.safe_dump(_candidate_pool_dict()),
                encoding="utf-8",
            )
            baseline_prompt = build_baseline_map_prompt(ctx, config)
            _record_checkpoint(
                dig_dir,
                phase="baseline_map",
                prompt=baseline_prompt,
                artifact=dig_dir / "baseline_mechanism_map.yaml",
            )
            normalized_baseline_map = BaselineMechanismMap.from_dict(baseline_map).to_dict()
            candidate_prompt = build_candidate_generation_prompt(
                ctx,
                normalized_baseline_map,
                config,
            )
            _record_checkpoint(
                dig_dir,
                phase="candidate_pool",
                prompt=candidate_prompt,
                artifact=dig_dir / "candidate_pool.yaml",
            )
            result = asyncio.run(
                run_dig_lite(
                    ctx=ctx,
                    config=config,
                    dig_dir=dig_dir,
                    workspace=root,
                    model="fake",
                    mcp_servers={},
                    plugin_registry=None,
                    agent_factory=lambda label: FakeAgent(label),
                )
            )

            self.assertEqual(FakeAgent.calls, ["candidate_reviews", "selected_contract"])
            self.assertEqual(result.selected_contract.variant_name, "calibration_repair_v1")
            status = yaml.safe_load((result.dig_dir / "dig_stage_status.json").read_text())
            reused = [event for event in status["events"] if event["status"] == "reused"]
            self.assertEqual(
                [event["phase"] for event in reused],
                ["baseline_map", "candidate_pool"],
            )

    def test_runner_does_not_reuse_checkpoint_without_matching_manifest(self) -> None:
        class FakeAgent:
            outputs: list[str] = []
            calls: list[str] = []

            def __init__(self, label: str):
                self.label = label

            async def execute(self, _prompt: str):
                FakeAgent.calls.append(self.label)
                return SimpleNamespace(
                    success=True,
                    output={"text": FakeAgent.outputs.pop(0)},
                    duration=0.0,
                    iteration_count=1,
                    error=None,
                )

        stale_baseline = {
            "task_objective": {"primary_metric": "old"},
            "baseline_core_path": [{"file": "old.py", "role": "old"}],
            "intervention_surfaces": [{"name": "old", "files": ["old.py"], "allowed": True}],
        }
        fresh_baseline = {
            "task_objective": {"primary_metric": "score"},
            "baseline_core_path": [{"file": "train.py", "role": "training"}],
            "intervention_surfaces": [
                {"name": "train", "files": ["train.py"], "allowed": True},
                {"name": "model", "files": ["model_def.py"], "allowed": True},
                {"name": "logging", "files": ["train.py"], "allowed": True},
            ],
            "forbidden_surfaces": [
                {"name": "evaluator", "files": ["evaluator.py"], "allowed": False}
            ],
        }
        FakeAgent.outputs = [
            yaml.safe_dump(fresh_baseline),
            yaml.safe_dump(_candidate_pool_dict()),
            yaml.safe_dump(_reviews_dict()),
            yaml.safe_dump(_contract_dict()),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dig_dir = root / "run/gen_0/peers/gen0_peer0/dig"
            dig_dir.mkdir(parents=True)
            (dig_dir / "baseline_mechanism_map.yaml").write_text(
                yaml.safe_dump(stale_baseline),
                encoding="utf-8",
            )
            asyncio.run(
                run_dig_lite(
                    ctx={
                        "peer_id": "gen0_peer0",
                        "gen_id": 0,
                        "peer_lane": {
                            "mechanism_family_preferences": ["calibration"],
                            "intervention_surface_preferences": ["train_loop"],
                            "intent_preference": "repair",
                        },
                    },
                    config=DIGLiteConfig(candidate_count=4, min_mechanism_families=4),
                    dig_dir=dig_dir,
                    workspace=root,
                    model="fake",
                    mcp_servers={},
                    plugin_registry=None,
                    agent_factory=lambda label: FakeAgent(label),
                )
            )

        self.assertEqual(
            FakeAgent.calls,
            ["baseline_map", "candidate_pool", "candidate_reviews", "selected_contract"],
        )

    def test_runner_does_not_reuse_checkpoint_when_artifact_hash_changes(self) -> None:
        class FakeAgent:
            outputs: list[str] = []
            calls: list[str] = []

            def __init__(self, label: str):
                self.label = label

            async def execute(self, _prompt: str):
                FakeAgent.calls.append(self.label)
                return SimpleNamespace(
                    success=True,
                    output={"text": FakeAgent.outputs.pop(0)},
                    duration=0.0,
                    iteration_count=1,
                    error=None,
                )

        ctx = {
            "peer_id": "gen0_peer0",
            "gen_id": 0,
            "peer_lane": {
                "mechanism_family_preferences": ["calibration"],
                "intervention_surface_preferences": ["train_loop"],
                "intent_preference": "repair",
            },
        }
        config = DIGLiteConfig(candidate_count=4, min_mechanism_families=4)
        valid_baseline = {
            "task_objective": {"primary_metric": "score"},
            "baseline_core_path": [{"file": "train.py", "role": "training"}],
            "intervention_surfaces": [
                {"name": "train", "files": ["train.py"], "allowed": True},
                {"name": "model", "files": ["model_def.py"], "allowed": True},
                {"name": "logging", "files": ["train.py"], "allowed": True},
            ],
            "forbidden_surfaces": [
                {"name": "evaluator", "files": ["evaluator.py"], "allowed": False}
            ],
        }
        FakeAgent.outputs = [
            yaml.safe_dump(valid_baseline),
            yaml.safe_dump(_candidate_pool_dict()),
            yaml.safe_dump(_reviews_dict()),
            yaml.safe_dump(_contract_dict()),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dig_dir = root / "run/gen_0/peers/gen0_peer0/dig"
            dig_dir.mkdir(parents=True)
            baseline_path = dig_dir / "baseline_mechanism_map.yaml"
            baseline_path.write_text(yaml.safe_dump(valid_baseline), encoding="utf-8")
            _record_checkpoint(
                dig_dir,
                phase="baseline_map",
                prompt=build_baseline_map_prompt(ctx, config),
                artifact=baseline_path,
            )
            baseline_path.write_text(
                yaml.safe_dump(
                    {
                        "task_objective": {"primary_metric": "tampered"},
                        "baseline_core_path": [{"file": "bad.py", "role": "bad"}],
                        "intervention_surfaces": [
                            {"name": "bad", "files": ["bad.py"], "allowed": True}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            asyncio.run(
                run_dig_lite(
                    ctx=ctx,
                    config=config,
                    dig_dir=dig_dir,
                    workspace=root,
                    model="fake",
                    mcp_servers={},
                    plugin_registry=None,
                    agent_factory=lambda label: FakeAgent(label),
                )
            )

        self.assertEqual(
            FakeAgent.calls,
            ["baseline_map", "candidate_pool", "candidate_reviews", "selected_contract"],
        )

    def test_runner_keeps_completed_checkpoint_when_later_phase_times_out(self) -> None:
        class FakeAgent:
            calls: list[str] = []

            def __init__(self, label: str):
                self.label = label

            async def execute(self, _prompt: str):
                FakeAgent.calls.append(self.label)
                if self.label == "candidate_pool":
                    raise TimeoutError
                baseline_map = {
                    "task_objective": {"primary_metric": "score"},
                    "baseline_core_path": [{"file": "train.py", "role": "training"}],
                    "intervention_surfaces": [
                        {"name": "train", "files": ["train.py"], "allowed": True},
                        {"name": "model", "files": ["model_def.py"], "allowed": True},
                        {"name": "logging", "files": ["train.py"], "allowed": True},
                    ],
                    "forbidden_surfaces": [
                        {"name": "evaluator", "files": ["evaluator.py"], "allowed": False}
                    ],
                }
                return SimpleNamespace(
                    success=True,
                    output={"text": yaml.safe_dump(baseline_map)},
                    duration=0.0,
                    iteration_count=1,
                    error=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dig_dir = root / "run/gen_0/peers/gen0_peer0/dig"
            with self.assertRaises(TimeoutError):
                asyncio.run(
                    run_dig_lite(
                        ctx={"peer_id": "gen0_peer0", "gen_id": 0},
                        config=DIGLiteConfig(candidate_count=4, min_mechanism_families=4),
                        dig_dir=dig_dir,
                        workspace=root,
                        model="fake",
                        mcp_servers={},
                        plugin_registry=None,
                        agent_factory=lambda label: FakeAgent(label),
                    )
                )

            self.assertEqual(FakeAgent.calls, ["baseline_map", "candidate_pool"])
            self.assertTrue((dig_dir / "baseline_mechanism_map.yaml").exists())
            self.assertFalse((dig_dir / "candidate_pool.yaml").exists())
            self.assertFalse((dig_dir / "selected_contract.yaml").exists())

    def test_runner_writes_required_artifacts_with_mock_agent(self) -> None:
        class FakeAgent:
            outputs: list[str] = []

            def __init__(self, label: str):
                self.label = label

            async def execute(self, _prompt: str):
                return SimpleNamespace(
                    success=True,
                    output={"text": FakeAgent.outputs.pop(0)},
                    duration=0.0,
                    iteration_count=1,
                    error=None,
                )

        baseline_map = {
            "task_objective": {"primary_metric": "score"},
            "baseline_core_path": [{"file": "train.py", "role": "training"}],
            "data_and_training_flow": {"input_flow": ["data"]},
            "intervention_surfaces": [
                {"name": "train", "files": ["train.py"], "allowed": True},
                {"name": "model", "files": ["model_def.py"], "allowed": True},
                {"name": "logging", "files": ["train.py"], "allowed": True},
            ],
            "forbidden_surfaces": [
                {"name": "evaluator", "files": ["evaluator.py"], "allowed": False}
            ],
        }
        FakeAgent.outputs = [
            yaml.safe_dump(baseline_map),
            yaml.safe_dump(_candidate_pool_dict()),
            yaml.safe_dump(_reviews_dict()),
            yaml.safe_dump(_contract_dict()),
        ]

        config = DIGLiteConfig(candidate_count=4, min_mechanism_families=4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = asyncio.run(
                run_dig_lite(
                    ctx={
                        "peer_id": "gen0_peer0",
                        "gen_id": 0,
                        "task_prompt": "Task",
                        "peer_lane": {
                            "mechanism_family_preferences": ["calibration"],
                            "intervention_surface_preferences": ["train_loop"],
                            "intent_preference": "repair",
                        },
                    },
                    config=config,
                    dig_dir=root / "run/gen_0/peers/gen0_peer0/dig",
                    workspace=root,
                    model="fake",
                    mcp_servers={},
                    plugin_registry=None,
                    agent_factory=lambda label: FakeAgent(label),
                )
            )

            self.assertEqual(result.selected_contract.variant_name, "calibration_repair_v1")
            for name in (
                "baseline_mechanism_map.yaml",
                "candidate_pool.yaml",
                "candidate_reviews.yaml",
                "qd_selection.yaml",
                "selected_contract.yaml",
                "dig_summary.md",
            ):
                self.assertTrue((result.dig_dir / name).exists(), name)

    def test_runner_qd_off_allows_known_similar_contract(self) -> None:
        class FakeAgent:
            outputs: list[str] = []

            def __init__(self, label: str):
                self.label = label

            async def execute(self, _prompt: str):
                return SimpleNamespace(
                    success=True,
                    output={"text": FakeAgent.outputs.pop(0)},
                    duration=0.0,
                    iteration_count=1,
                    error=None,
                )

        baseline_map = {
            "task_objective": {"primary_metric": "score"},
            "baseline_core_path": [{"file": "train.py", "role": "training"}],
            "data_and_training_flow": {"input_flow": ["data"]},
            "intervention_surfaces": [
                {"name": "train", "files": ["train.py"], "allowed": True},
                {"name": "model", "files": ["model_def.py"], "allowed": True},
                {"name": "logging", "files": ["train.py"], "allowed": True},
            ],
            "forbidden_surfaces": [
                {"name": "evaluator", "files": ["evaluator.py"], "allowed": False}
            ],
        }
        FakeAgent.outputs = [
            yaml.safe_dump(baseline_map),
            yaml.safe_dump(_candidate_pool_dict()),
            yaml.safe_dump(_reviews_dict()),
            yaml.safe_dump(_contract_dict()),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = asyncio.run(
                run_dig_lite(
                    ctx={
                        "peer_id": "gen0_peer0",
                        "gen_id": 0,
                        "frontier_summary": [
                            {
                                "variant_name": "known_similar",
                                "mechanism_family": "calibration",
                                "intervention_surface": "train_loop",
                                "intent": "repair",
                            }
                        ],
                        "peer_lane": {
                            "mechanism_family_preferences": ["calibration"],
                            "intervention_surface_preferences": ["train_loop"],
                            "intent_preference": "repair",
                        },
                    },
                    config=DIGLiteConfig(candidate_count=4, min_mechanism_families=4),
                    dig_dir=root / "run/gen_0/peers/gen0_peer0/dig",
                    workspace=root,
                    model="fake",
                    mcp_servers={},
                    plugin_registry=None,
                    agent_factory=lambda label: FakeAgent(label),
                    quality_diversity_enabled=False,
                )
            )

        self.assertEqual(result.selected_contract.selected_candidate_id, "C01")
        self.assertFalse(result.qd_selection["quality_diversity_enabled"])


class DIGLiteCohortIntegrationTests(unittest.TestCase):
    def test_cohort_dig_policy_caps_diagnostic_slots(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.cohort_runner import (
            _assign_dig_selection_policies,
        )

        contexts = {i: {"peer_id": f"gen0_peer{i}"} for i in range(12)}
        config = DIGLiteConfig.from_raw(
            {
                "candidate_count": 4,
                "min_mechanism_families": 4,
                "innovation": {
                    "max_diagnostic_fraction": 0.20,
                    "max_diagnostic_peers": 2,
                },
            }
        )

        policies = _assign_dig_selection_policies(
            contexts,
            dig_config=config,
            cohort_size=12,
        )

        diagnostic = [
            index for index, policy in policies.items() if policy["intent_slot"] == "diagnostic"
        ]
        forward = [
            index
            for index, policy in policies.items()
            if policy["intent_slot"] == "forward_innovation"
        ]
        self.assertEqual(diagnostic, [10, 11])
        self.assertEqual(len(forward), 10)

    def test_cohort_injects_selected_contract_prompt_block(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import cohort_runner

        created: list[dict] = []

        class FakeAgentLoop:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                created.append(kwargs)

            async def run(self):
                return {"peer_id": self.kwargs["peer_id"], "success": True}

        class FakeTrigger:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.fired = False
                self.closing = False

            async def wait_until_fire(self, abort_event):
                return None

            async def evaluate_async(self):
                return SimpleNamespace(fired=False, reason="postgen")

            def fire(self, _snapshot):
                self.fired = True

            def write_postgen_marker(self, snapshot):
                (self.kwargs["gen_dir"] / "STOP_SIGNAL_POSTGEN").write_text(
                    snapshot.reason,
                    encoding="utf-8",
                )

        async def fake_run_dig_lite(**kwargs):
            contract = SelectedContract.from_dict(_contract_dict())
            path = kwargs["dig_dir"] / "selected_contract.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(contract.to_dict()), encoding="utf-8")
            return DIGLiteResult(
                dig_dir=kwargs["dig_dir"],
                selected_contract=contract,
                selected_contract_path=path,
                qd_selection={"selected_candidate_id": "C01"},
            )

        def fake_resolve_prompt_with_layout(**kwargs):
            extra = kwargs.get("extra_dynamic_blocks") or []
            text = "base prompt\n" + "\n".join(block["text"] for block in extra)
            return text, {"layout_hash": "h"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = _fake_loop(root, cohort_size=2)
            role_skills = {}
            for index, role_name in enumerate(("starter", "analyst")):
                role_dir = root / "roles" / role_name
                role_dir.mkdir(parents=True)
                role_text = f"Follow the {role_name} contract."
                (role_dir / "skill.md").write_text(role_text, encoding="utf-8")
                role_skills[f"gen0_peer{index}"] = SimpleNamespace(
                    role_ref=f"task_role:{role_name}",
                    plugin_path=role_dir,
                    skill_markdown=role_text,
                    content_hash=f"{role_name}-hash",
                )
            loop.peer_role_ref = "task_role:peer"
            loop.peer_role_skill = None
            loop.peer_role_skill_for_context = lambda ctx: role_skills[ctx["peer_id"]]
            with (
                patch.object(cohort_runner, "AutonomousAgentLoop", FakeAgentLoop),
                patch.object(cohort_runner, "run_dig_lite", fake_run_dig_lite),
                patch.object(
                    cohort_runner,
                    "resolve_prompt_with_layout",
                    side_effect=fake_resolve_prompt_with_layout,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.synthesis_trigger.SynthesisTrigger",
                    FakeTrigger,
                ),
            ):
                results = asyncio.run(cohort_runner.run_generation_cohort(loop, 0))

            self.assertEqual(
                results,
                [
                    {"peer_id": "gen0_peer0", "success": True},
                    {"peer_id": "gen0_peer1", "success": True},
                ],
            )
            created_by_peer = {item["peer_id"]: item for item in created}
            for index, role_name in enumerate(("starter", "analyst")):
                peer = created_by_peer[f"gen0_peer{index}"]
                self.assertIn("Selected peer RoleSkill contract", peer["task_prompt"])
                self.assertIn(f"Follow the {role_name} contract.", peer["task_prompt"])
                self.assertIn("DIG-Lite Selected Contract", peer["task_prompt"])
                self.assertIn("calibration_repair_v1", peer["task_prompt"])
                self.assertEqual(peer["role_ref"], f"task_role:{role_name}")
                self.assertEqual(peer["role_skill_sha256"], f"{role_name}-hash")

    def test_cohort_passes_codex_runtime_ref_to_dig_planner(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import cohort_runner

        captured_ctxs: list[dict[str, object]] = []
        created: list[dict] = []

        class FakeAgentLoop:
            def __init__(self, **kwargs):
                created.append(kwargs)

            async def run(self):
                return {"peer_id": created[-1]["peer_id"], "success": True}

        class FakeTrigger:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.fired = False
                self.closing = False

            async def wait_until_fire(self, abort_event):
                return None

            async def evaluate_async(self):
                return SimpleNamespace(fired=False, reason="postgen")

            def fire(self, _snapshot):
                return None

            def write_postgen_marker(self, snapshot):
                return None

        async def fake_run_dig_lite(**kwargs):
            captured_ctxs.append(dict(kwargs["ctx"]))
            contract = SelectedContract.from_dict(_contract_dict())
            path = kwargs["dig_dir"] / "selected_contract.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(contract.to_dict()), encoding="utf-8")
            return DIGLiteResult(
                dig_dir=kwargs["dig_dir"],
                selected_contract=contract,
                selected_contract_path=path,
                qd_selection={"selected_candidate_id": "C01"},
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = _fake_loop(root, cohort_size=1)
            loop.runtime_ref = "agent_runtime:codex_sdk"
            with (
                patch.object(cohort_runner, "AutonomousAgentLoop", FakeAgentLoop),
                patch.object(cohort_runner, "run_dig_lite", fake_run_dig_lite),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.synthesis_trigger.SynthesisTrigger",
                    FakeTrigger,
                ),
            ):
                asyncio.run(cohort_runner.run_generation_cohort(loop, 0))

        self.assertEqual(captured_ctxs[0]["agent_runtime_ref"], "agent_runtime:codex_sdk")

    def test_cohort_runs_dig_prelaunch_for_peers_concurrently(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import cohort_runner

        active = 0
        max_active = 0
        created: list[dict] = []
        initialized_peers: list[str] = []

        class FakeAgentLoop:
            def __init__(self, **kwargs):
                created.append(kwargs)

            async def run(self):
                return {"peer_id": created[-1]["peer_id"], "success": True}

        class FakeTrigger:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.fired = False
                self.closing = False

            async def wait_until_fire(self, abort_event):
                return None

            async def evaluate_async(self):
                return SimpleNamespace(fired=False, reason="postgen")

            def fire(self, _snapshot):
                self.fired = True

            def write_postgen_marker(self, snapshot):
                (self.kwargs["gen_dir"] / "STOP_SIGNAL_POSTGEN").write_text(
                    snapshot.reason,
                    encoding="utf-8",
                )

        async def fake_run_dig_lite(**kwargs):
            nonlocal active, max_active
            state_path = kwargs["dig_dir"].parent / "memory" / "peer_state.yaml"
            state = yaml.safe_load(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["research_state"], "initializing")
            initialized_peers.append(str(state["peer_id"]))
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            active -= 1
            contract = SelectedContract.from_dict(_contract_dict())
            path = kwargs["dig_dir"] / "selected_contract.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(contract.to_dict()), encoding="utf-8")
            return DIGLiteResult(
                dig_dir=kwargs["dig_dir"],
                selected_contract=contract,
                selected_contract_path=path,
                qd_selection={"selected_candidate_id": "C01"},
            )

        def fake_resolve_prompt_with_layout(**kwargs):
            extra = kwargs.get("extra_dynamic_blocks") or []
            text = "base prompt\n" + "\n".join(block["text"] for block in extra)
            return text, {"layout_hash": "h"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = _fake_loop(root, cohort_size=3)
            with (
                patch.object(cohort_runner, "AutonomousAgentLoop", FakeAgentLoop),
                patch.object(cohort_runner, "run_dig_lite", fake_run_dig_lite),
                patch.object(
                    cohort_runner,
                    "resolve_prompt_with_layout",
                    side_effect=fake_resolve_prompt_with_layout,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.synthesis_trigger.SynthesisTrigger",
                    FakeTrigger,
                ),
            ):
                results = asyncio.run(cohort_runner.run_generation_cohort(loop, 0))

            self.assertEqual(len(results), 3)
            self.assertEqual(len(created), 3)
            self.assertGreater(max_active, 1)
            self.assertCountEqual(
                initialized_peers,
                ["gen0_peer0", "gen0_peer1", "gen0_peer2"],
            )
            for item in created:
                self.assertIn("DIG-Lite Selected Contract", item["task_prompt"])

    def test_cohort_qd_allocator_updates_prompt_contracts_before_launch(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import cohort_runner

        created: list[dict] = []

        class FakeAgentLoop:
            def __init__(self, **kwargs):
                created.append(kwargs)

            async def run(self):
                return {"peer_id": created[-1]["peer_id"], "success": True}

        class FakeTrigger:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.fired = False
                self.closing = False

            async def wait_until_fire(self, abort_event):
                return None

            async def evaluate_async(self):
                return SimpleNamespace(fired=False, reason="postgen")

            def fire(self, _snapshot):
                self.fired = True

            def write_postgen_marker(self, snapshot):
                (self.kwargs["gen_dir"] / "STOP_SIGNAL_POSTGEN").write_text(
                    snapshot.reason,
                    encoding="utf-8",
                )

        async def fake_run_dig_lite(**kwargs):
            contract = SelectedContract.from_dict(_contract_dict())
            path = kwargs["dig_dir"] / "selected_contract.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(contract.to_dict()), encoding="utf-8")
            return DIGLiteResult(
                dig_dir=kwargs["dig_dir"],
                selected_contract=contract,
                selected_contract_path=path,
                qd_selection={"selected_candidate_id": "C01"},
                candidate_pool=CandidatePool.from_dict(_candidate_pool_dict()),
                candidate_reviews=CandidateReviews.from_dict(_reviews_dict()),
                validation_context=_validation_context(),
            )

        def fake_resolve_prompt_with_layout(**kwargs):
            extra = kwargs.get("extra_dynamic_blocks") or []
            text = "base prompt\n" + "\n".join(block["text"] for block in extra)
            return text, {"layout_hash": "h"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = _fake_loop(root, cohort_size=2)
            loop.dig_lite_config.cohort_qd.max_same_mechanism_family_peers = 1
            loop.dig_lite_config.cohort_qd.max_same_diversity_cell_peers = 1
            with (
                patch.object(cohort_runner, "AutonomousAgentLoop", FakeAgentLoop),
                patch.object(cohort_runner, "run_dig_lite", fake_run_dig_lite),
                patch.object(
                    cohort_runner,
                    "resolve_prompt_with_layout",
                    side_effect=fake_resolve_prompt_with_layout,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.synthesis_trigger.SynthesisTrigger",
                    FakeTrigger,
                ),
            ):
                results = asyncio.run(cohort_runner.run_generation_cohort(loop, 0))

            self.assertEqual(len(results), 2)
            prompts = [item["task_prompt"] for item in created]
            self.assertTrue(any("calibration_repair_v1" in prompt for prompt in prompts))
            self.assertTrue(any("c02_architecture" in prompt for prompt in prompts))
            self.assertTrue((root / "run" / "gen_0" / "dig_cohort_allocation.yaml").exists())

    def test_cohort_generation_scope_defaults_initial_and_all_override(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import cohort_runner

        dig_calls: list[tuple[int, str]] = []
        created: list[dict] = []

        class FakeAgentLoop:
            def __init__(self, **kwargs):
                created.append(kwargs)

            async def run(self):
                return {"peer_id": self.kwargs["peer_id"], "success": True}

            @property
            def kwargs(self):
                return created[-1]

        class FakeTrigger:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.fired = False
                self.closing = False

            async def wait_until_fire(self, abort_event):
                return None

            async def evaluate_async(self):
                return SimpleNamespace(fired=False, reason="postgen")

            def fire(self, _snapshot):
                self.fired = True

            def write_postgen_marker(self, snapshot):
                (self.kwargs["gen_dir"] / "STOP_SIGNAL_POSTGEN").write_text(
                    snapshot.reason,
                    encoding="utf-8",
                )

        async def fake_run_dig_lite(**kwargs):
            ctx = kwargs["ctx"]
            dig_calls.append((ctx["gen_id"], ctx["peer_id"]))
            contract = SelectedContract.from_dict(_contract_dict())
            path = kwargs["dig_dir"] / "selected_contract.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(contract.to_dict()), encoding="utf-8")
            return DIGLiteResult(
                dig_dir=kwargs["dig_dir"],
                selected_contract=contract,
                selected_contract_path=path,
                qd_selection={"selected_candidate_id": "C01"},
            )

        def fake_resolve_prompt_with_layout(**kwargs):
            extra = kwargs.get("extra_dynamic_blocks") or []
            text = "base prompt\n" + "\n".join(block["text"] for block in extra)
            return text, {"layout_hash": "h"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = _fake_loop(root, cohort_size=2)
            # Gems can reset its logical cycle number, but DIG scope is tied to
            # the absolute generation id and must not reactivate at gen1.
            loop.gems = SimpleNamespace(
                logical_generation=lambda _gen_id: 0,
                prompt_context=lambda _gen_id: {"cycle_index": 1},
            )
            with (
                patch.object(cohort_runner, "AutonomousAgentLoop", FakeAgentLoop),
                patch.object(cohort_runner, "run_dig_lite", fake_run_dig_lite),
                patch.object(
                    cohort_runner,
                    "resolve_prompt_with_layout",
                    side_effect=fake_resolve_prompt_with_layout,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.synthesis_trigger.SynthesisTrigger",
                    FakeTrigger,
                ),
            ):
                gen0_results = asyncio.run(cohort_runner.run_generation_cohort(loop, 0))
                gen1_results = asyncio.run(cohort_runner.run_generation_cohort(loop, 1))
                loop.dig_lite_config.generation_scope = "all"
                gen2_results = asyncio.run(cohort_runner.run_generation_cohort(loop, 2))

            self.assertEqual(len(gen0_results), 2)
            self.assertEqual(len(gen1_results), 2)
            self.assertEqual(len(gen2_results), 2)
            self.assertEqual(
                dig_calls,
                [
                    (0, "gen0_peer0"),
                    (0, "gen0_peer1"),
                    (2, "gen2_peer0"),
                    (2, "gen2_peer1"),
                ],
            )
            self.assertEqual(len(created), 6)
            for item in created[:2]:
                self.assertIn("DIG-Lite Selected Contract", item["task_prompt"])
            for item in created[2:]:
                expected = "DIG-Lite Selected Contract" if "gen2_" in item["peer_id"] else ""
                if expected:
                    self.assertIn(expected, item["task_prompt"])
                else:
                    self.assertNotIn("DIG-Lite Selected Contract", item["task_prompt"])

    def test_initial_qd_can_be_disabled_without_disabling_gen0_dig(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import cohort_runner

        qd_flags: list[bool] = []
        created: list[dict] = []

        class FakeAgentLoop:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                created.append(kwargs)

            async def run(self):
                return {"peer_id": self.kwargs["peer_id"], "success": True}

        class FakeTrigger:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.fired = False
                self.closing = False

            async def wait_until_fire(self, abort_event):
                return None

            async def evaluate_async(self):
                return SimpleNamespace(fired=False, reason="postgen")

            def fire(self, _snapshot):
                return None

            def write_postgen_marker(self, snapshot):
                return None

        async def fake_run_dig_lite(**kwargs):
            qd_flags.append(kwargs["quality_diversity_enabled"])
            contract = SelectedContract.from_dict(_contract_dict())
            path = kwargs["dig_dir"] / "selected_contract.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(contract.to_dict()), encoding="utf-8")
            return DIGLiteResult(
                dig_dir=kwargs["dig_dir"],
                selected_contract=contract,
                selected_contract_path=path,
                qd_selection={"selected_candidate_id": "C01"},
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = _fake_loop(root, cohort_size=1)
            loop.quality_diversity_config = QualityDiversityConfig.from_task_spec(
                {
                    "quality_diversity": {
                        "enabled": True,
                        "initial_generation_enabled": False,
                        "later_generations_enabled": True,
                    }
                },
                dig_config=loop.dig_lite_config,
            )
            with (
                patch.object(cohort_runner, "AutonomousAgentLoop", FakeAgentLoop),
                patch.object(cohort_runner, "run_dig_lite", fake_run_dig_lite),
                patch.object(
                    cohort_runner,
                    "allocate_cohort_qd_contracts",
                    side_effect=AssertionError("cohort QD must be skipped"),
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.synthesis_trigger.SynthesisTrigger",
                    FakeTrigger,
                ),
            ):
                results = asyncio.run(cohort_runner.run_generation_cohort(loop, 0))

        self.assertEqual(results, [{"peer_id": "gen0_peer0", "success": True}])
        self.assertEqual(qd_flags, [False])
        self.assertEqual(len(created), 1)

    def test_legacy_allocator_off_keeps_local_qd_and_clears_stale_views(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import cohort_runner

        qd_flags: list[bool] = []

        class FakeAgentLoop:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def run(self):
                return {"peer_id": self.kwargs["peer_id"], "success": True}

        class FakeTrigger:
            def __init__(self, **kwargs):
                self.fired = False
                self.closing = False

            async def wait_until_fire(self, abort_event):
                return None

            async def evaluate_async(self):
                return SimpleNamespace(fired=False, reason="postgen")

            def fire(self, _snapshot):
                return None

            def write_postgen_marker(self, snapshot):
                return None

        async def fake_run_dig_lite(**kwargs):
            qd_flags.append(kwargs["quality_diversity_enabled"])
            contract = SelectedContract.from_dict(_contract_dict())
            path = kwargs["dig_dir"] / "selected_contract.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(contract.to_dict()), encoding="utf-8")
            return DIGLiteResult(
                dig_dir=kwargs["dig_dir"],
                selected_contract=contract,
                selected_contract_path=path,
                qd_selection={"selected_candidate_id": "C01"},
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = _fake_loop(root, cohort_size=1)
            loop.dig_lite_config.generation_scope = "all"
            loop.dig_lite_config.cohort_qd.enabled = False
            loop.quality_diversity_config = QualityDiversityConfig.from_task_spec(
                {"dig_lite": {"cohort_qd": {"enabled": False}}},
                dig_config=loop.dig_lite_config,
            )
            gen_dir = root / "run" / "gen_1"
            override = gen_dir / "peers" / "gen1_peer0" / "dig" / "cohort_qd_override.yaml"
            override.parent.mkdir(parents=True, exist_ok=True)
            override.write_text("stale: true\n", encoding="utf-8")
            allocation = gen_dir / "dig_cohort_allocation.yaml"
            allocation.write_text("stale: true\n", encoding="utf-8")
            with (
                patch.dict(os.environ, {}, clear=False),
                patch.object(cohort_runner, "AutonomousAgentLoop", FakeAgentLoop),
                patch.object(cohort_runner, "run_dig_lite", fake_run_dig_lite),
                patch.object(
                    cohort_runner,
                    "allocate_cohort_qd_contracts",
                    side_effect=AssertionError("legacy allocator switch must remain off"),
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.synthesis_trigger.SynthesisTrigger",
                    FakeTrigger,
                ),
            ):
                results = asyncio.run(cohort_runner.run_generation_cohort(loop, 1))

            self.assertEqual(results, [{"peer_id": "gen1_peer0", "success": True}])
            self.assertEqual(qd_flags, [True])
            self.assertFalse(override.exists())
            self.assertFalse(allocation.exists())

    def test_cohort_retries_dig_then_falls_back_to_direct_prompt(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import cohort_runner

        attempts = 0
        created: list[dict] = []

        class FakeAgentLoop:
            def __init__(self, **kwargs):
                created.append(kwargs)

            async def run(self):
                return {"peer_id": self.kwargs["peer_id"], "success": True}

            @property
            def kwargs(self):
                return created[-1]

        class FakeTrigger:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.fired = False
                self.closing = False

            async def wait_until_fire(self, abort_event):
                return None

            async def evaluate_async(self):
                return SimpleNamespace(fired=False, reason="postgen")

            def fire(self, _snapshot):
                self.fired = True

            def write_postgen_marker(self, snapshot):
                (self.kwargs["gen_dir"] / "STOP_SIGNAL_POSTGEN").write_text(
                    snapshot.reason,
                    encoding="utf-8",
                )

        async def failing_dig(**_kwargs):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("bad yaml")

        def fake_resolve_prompt_with_layout(**kwargs):
            _ = kwargs
            return "base prompt", {"layout_hash": "h"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = _fake_loop(root, cohort_size=1)
            loop.dig_lite_config.max_attempts = 2
            with (
                patch.object(cohort_runner, "AutonomousAgentLoop", FakeAgentLoop),
                patch.object(cohort_runner, "run_dig_lite", failing_dig),
                patch.object(
                    cohort_runner,
                    "resolve_prompt_with_layout",
                    side_effect=fake_resolve_prompt_with_layout,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.synthesis_trigger.SynthesisTrigger",
                    FakeTrigger,
                ),
            ):
                results = asyncio.run(cohort_runner.run_generation_cohort(loop, 0))

            self.assertEqual(attempts, 2)
            self.assertEqual(results, [{"peer_id": "gen0_peer0", "success": True}])
            self.assertEqual(created[0]["task_prompt"], "base prompt")
            summary = root / "run/gen_0/peers/gen0_peer0/dig/dig_failure_summary.json"
            self.assertTrue(summary.exists())

    def test_cohort_caps_dig_retries_at_ten(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import cohort_runner

        attempts = 0
        created: list[dict] = []

        class FakeAgentLoop:
            def __init__(self, **kwargs):
                created.append(kwargs)

            async def run(self):
                return {"peer_id": self.kwargs["peer_id"], "success": True}

            @property
            def kwargs(self):
                return created[-1]

        class FakeTrigger:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.fired = False
                self.closing = False

            async def wait_until_fire(self, abort_event):
                return None

            async def evaluate_async(self):
                return SimpleNamespace(fired=False, reason="postgen")

            def fire(self, _snapshot):
                self.fired = True

            def write_postgen_marker(self, snapshot):
                (self.kwargs["gen_dir"] / "STOP_SIGNAL_POSTGEN").write_text(
                    snapshot.reason,
                    encoding="utf-8",
                )

        async def failing_dig(**_kwargs):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("bad yaml")

        def fake_resolve_prompt_with_layout(**_kwargs):
            return "base prompt", {"layout_hash": "h"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = _fake_loop(root, cohort_size=1)
            loop.dig_lite_config.max_attempts = 25
            with (
                patch.object(cohort_runner, "AutonomousAgentLoop", FakeAgentLoop),
                patch.object(cohort_runner, "run_dig_lite", failing_dig),
                patch.object(
                    cohort_runner,
                    "resolve_prompt_with_layout",
                    side_effect=fake_resolve_prompt_with_layout,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.synthesis_trigger.SynthesisTrigger",
                    FakeTrigger,
                ),
            ):
                results = asyncio.run(cohort_runner.run_generation_cohort(loop, 0))

            self.assertEqual(attempts, 10)
            self.assertEqual(results, [{"peer_id": "gen0_peer0", "success": True}])

    def test_cohort_suppresses_peer_when_dig_fallback_disabled(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import cohort_runner

        attempts = 0
        created: list[dict] = []

        class FakeAgentLoop:
            def __init__(self, **kwargs):
                created.append(kwargs)

            async def run(self):
                return {"peer_id": self.kwargs["peer_id"], "success": True}

            @property
            def kwargs(self):
                return created[-1]

        class FakeTrigger:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.fired = False
                self.closing = False

            async def wait_until_fire(self, abort_event):
                return None

            async def evaluate_async(self):
                return SimpleNamespace(fired=False, reason="postgen")

            def fire(self, _snapshot):
                self.fired = True

            def write_postgen_marker(self, snapshot):
                (self.kwargs["gen_dir"] / "STOP_SIGNAL_POSTGEN").write_text(
                    snapshot.reason,
                    encoding="utf-8",
                )

        async def failing_dig(**_kwargs):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("bad yaml")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = _fake_loop(root, cohort_size=1)
            loop.dig_lite_config.max_attempts = 2
            loop.dig_lite_config.strict = False
            loop.dig_lite_config.fallback_to_direct_on_failure = False
            with (
                patch.object(cohort_runner, "AutonomousAgentLoop", FakeAgentLoop),
                patch.object(cohort_runner, "run_dig_lite", failing_dig),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.synthesis_trigger.SynthesisTrigger",
                    FakeTrigger,
                ),
            ):
                results = asyncio.run(cohort_runner.run_generation_cohort(loop, 0))

            self.assertEqual(attempts, 2)
            self.assertEqual(created, [])
            self.assertEqual(len(results), 1)
            self.assertFalse(results[0]["success"])
            self.assertEqual(results[0]["phase"], "dig_lite")


def _fake_loop(root: Path, *, cohort_size: int) -> SimpleNamespace:
    run_dir = root / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        task_spec=SimpleNamespace(
            generation_policy=SimpleNamespace(cohort_size=cohort_size, per_generation_hours=0.001),
            synthesis_trigger=SimpleNamespace(
                enabled=False,
                min_findings=1,
                min_interval_minutes=1,
                max_interval_minutes=2,
                min_contributing_peers=1,
                poll_interval_seconds=1,
            ),
            agent=SimpleNamespace(premium_mode=False),
        ),
        run_dir=run_dir,
        workspace=root,
        base_template=root / "base.jinja2",
        task_prompt_path=root / "task.jinja2",
        gen_template=root / "gen.jinja2",
        findings_dir=root / "findings",
        model="fake",
        local_mode=True,
        mcp_servers={},
        _peer_allowed_tools=["Read"],
        plugin_registry=None,
        _findings_sync=SimpleNamespace(sync_once=lambda: None),
        _build_prompt_context=lambda gen_id, peer, cohort: {"peer": peer, "gen_id": gen_id},
        _persist_prompt_layout_artifacts=lambda **kwargs: kwargs["manifest"],
        dig_lite_config=DIGLiteConfig(candidate_count=4, min_mechanism_families=4),
    )


if __name__ == "__main__":
    unittest.main()
