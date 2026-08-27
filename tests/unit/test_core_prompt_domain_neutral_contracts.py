from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def test_bundled_research_loop_prompts_do_not_embed_trading_schema_terms() -> None:
    """Core prompts must stay task-generic.

    Trading/AIST terms belong in task-local prompts and task configuration.
    Bundled prompts are used by MLE and other tasks that do not speak in
    returns, drawdowns, L1 quote features, or alpha/incubator lanes.
    """

    root = Path("praxist/plugins/workflow_stages/research_loop/backend")
    prompt_paths = [
        root / "prompt_base.jinja2",
        root / "prompt_generation.jinja2",
        root / "synthesis_prompt.jinja2",
        root / "multi_pi" / "prompts" / "base.jinja2",
        root / "multi_pi" / "prompts" / "chair.jinja2",
        root / "dig" / "prompts.py",
        root / "multi_pi" / "pi_roles" / "skeptic_pi.py",
        root / "multi_pi" / "pi_roles" / "external_validity_pi.py",
    ]
    forbidden = [
        "High-Return",
        "high-return",
        "candidate_alpha",
        "high_return_clean_candidate",
        "high_return_drawdown_repair_target",
        "diversified_low_alpha_candidate",
        "L1_behavior_candidate",
        "active_alpha_vs_drawdown",
        "return_vs_mdd",
        "return_vs_effN",
        "mean_active_alpha_vs_benchmark_pct",
        "q25_active_alpha_vs_benchmark_pct",
        "validation_2026_active_alpha_pct",
        "cash_drag_or_underinvestment",
        "drawdown_regression",
        "max_drawdown_pct",
        "mean_mdd_pct",
        "learned_alpha",
        "confirmed_alpha",
        "high_turnover_or_cost_fragility",
        "alpha-incubator candidates",
        "alpha_incubator",
        "alpha_incubator protection",
        "data_and_training_flow",
        "model_forward_flow",
        "loss_flow",
        "candidate_model",
        "<training command>",
        "training outputs",
        "sentinel architecture",
        "cross-arch",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in prompt_paths)
    for term in forbidden:
        assert term not in combined


def test_default_prompt_strategy_examples_do_not_embed_trading_domain_terms() -> None:
    path = Path("praxist/plugins/workflow_stages/research_loop/backend/prompt_strategy.py")
    text = path.read_text(encoding="utf-8").lower()

    for term in ("cross-stock", "bull market", "bull markets"):
        assert term not in text


def test_generic_surfaces_do_not_use_task_local_evaluation_window_terms() -> None:
    roots = (
        Path("praxist/plugins/workflow_stages/research_loop"),
        Path("skills"),
        Path("templates/tasks/template"),
        Path("templates/tasks/machine_learning_template"),
        Path("templates/tasks/toy_math"),
    )
    suffixes = {".py", ".md", ".yaml", ".yml", ".jinja2"}

    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            text = path.read_text(encoding="utf-8").lower()
            for term in ("full-window", "full_window", "full window", "windowed"):
                assert term not in text, f"{path} contains task-local term {term!r}"

    # Historical spellings live only in the one-way migration boundary. They
    # must never leak back into the generic task model or runtime consumers.
    for path in (
        Path("praxist/task_spec.py"),
        Path("praxist/plugins/tools/frontier_tools/adapter.py"),
    ):
        text = path.read_text(encoding="utf-8").lower()
        for term in ("full-window", "full_window", "full window", "windowed"):
            assert term not in text, f"{path} contains task-local term {term!r}"


def test_tiered_stage_names_remain_opaque_without_explicit_maturity_policy() -> None:
    from praxist.task_spec import load_task_spec

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "description.md").write_text("x", encoding="utf-8")
        (root / "task.yaml").write_text(
            """
task_id: legacy_staged_task
task_name: Legacy staged task
description_file: description.md
evaluation:
  primary_metric: score
  maturity_policy:
    require_ratio_gate: false
tiered_eval:
  coarse_check: {budget: 1}
  complete_study: {budget: 2}
""",
            encoding="utf-8",
        )

        with unittest.TestCase().assertLogs("praxist.task_spec", level="WARNING") as captured:
            spec = load_task_spec(root / "task.yaml")

    assert spec.evaluation.maturity_policy["complete_stage_labels"] == []
    assert spec.evaluation.maturity_policy["preliminary_stage_labels"] == []
    assert "stage names remain advisory" in "\n".join(captured.output)


def test_bundled_sam_task_declares_its_task_local_maturity_stages() -> None:
    from praxist.task_spec import load_task_spec

    task_path = (
        Path(__file__).resolve().parents[2] / "templates" / "tasks" / "sam_optimizer" / "task.yaml"
    )
    spec = load_task_spec(task_path)

    assert spec.evaluation.maturity_policy["complete_stage_labels"] == ["T3"]
    assert spec.evaluation.maturity_policy["preliminary_stage_labels"] == ["T1", "T2"]


def test_explicit_task_stage_maturity_is_not_overridden_by_legacy_adapter() -> None:
    from praxist.task_spec import load_task_spec

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "description.md").write_text("x", encoding="utf-8")
        (root / "task.yaml").write_text(
            """
task_id: explicit_staged_task
task_name: Explicit staged task
description_file: description.md
evaluation:
  primary_metric: score
  maturity_policy:
    require_ratio_gate: false
    complete_stage_labels: [complete_study]
    preliminary_stage_labels: [coarse_check]
tiered_eval:
  coarse_check: {budget: 1}
  complete_study: {budget: 2}
""",
            encoding="utf-8",
        )

        spec = load_task_spec(root / "task.yaml")

    assert spec.evaluation.maturity_policy["complete_stage_labels"] == ["complete_study"]
    assert spec.evaluation.maturity_policy["preliminary_stage_labels"] == ["coarse_check"]


def test_default_gems_config_uses_generic_result_artifact_lane_and_detector() -> None:
    from praxist.task_spec import GemsConfig

    cfg = GemsConfig()

    assert cfg.bottleneck_detector_mode == "generic"
    assert cfg.result_artifact_default_lane == "performance"
    assert cfg.result_artifact_default_family == "task_candidate"
    assert cfg.performance_lanes == []
    assert cfg.gem_seeded_independent_peers == 0


def test_generic_bottleneck_detector_does_not_emit_l1_priors_by_default() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.exploration_bottleneck_detector import (
        ExplorationBottleneckDetector,
    )

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        findings = run_dir / "shared_findings"
        findings.mkdir()
        (findings / "gen0_peer0.json").write_text(
            json.dumps(
                {
                    "id": "f1",
                    "generation_id": 0,
                    "variant_name": "plain_candidate",
                    "finding_type": "result",
                    "metrics": {"score": 0.7},
                }
            ),
            encoding="utf-8",
        )

        report = ExplorationBottleneckDetector(run_dir=run_dir).analyze(
            completed_gen_id=0,
            manifest={"cumulative_top": []},
        )

    serialized = json.dumps(report)
    assert "l1_opportunity_gap" not in serialized
    assert "increase_l1_aware_contract_probability" not in serialized
    assert report["metrics"]["detector_mode"] == "generic"


def test_result_artifact_materialization_defaults_to_generic_performance_lane() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
        _materialize_result_artifacts,
    )

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        result_dir = run_dir / "results" / "candidate_a"
        result_dir.mkdir(parents=True)
        (result_dir / "tiered_eval_summary.json").write_text(
            json.dumps(
                {
                    "variant_name": "candidate_a",
                    "generation_id": 0,
                    "current_aggregate": {
                        "score": 0.81,
                        "completed_required_eval_units": 3,
                    },
                    "tier_reached": "full_eval",
                    "tier_status": "passed",
                    "evaluation_units": 3,
                }
            ),
            encoding="utf-8",
        )

        [finding] = _materialize_result_artifacts(run_dir=run_dir, gen_id=0)

    assert finding["metrics"]["frontier_lane"] == "performance"
    assert finding["metrics"]["strategy_family"] == "task_candidate"
    assert "alpha" not in finding["content"].lower()


def test_generic_result_summary_artifact_is_discovered() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
        _materialize_result_artifacts,
        iter_result_summary_paths,
    )

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        result_dir = run_dir / "results" / "candidate_a"
        result_dir.mkdir(parents=True)
        summary_path = result_dir / "result_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "variant_name": "candidate_a",
                    "generation_id": 0,
                    "scored_complete": True,
                    "current_aggregate": {"score": 0.81, "completed_required_eval_units": 3},
                    "evaluation_units": 3,
                }
            ),
            encoding="utf-8",
        )

        paths = iter_result_summary_paths(run_dir)
        [finding] = _materialize_result_artifacts(run_dir=run_dir, gen_id=0)

    assert paths == [summary_path]
    assert finding["metrics"]["frontier_lane"] == "performance"
    assert finding["metrics"]["strategy_family"] == "task_candidate"


def test_unscored_result_summary_artifact_is_not_materialized() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
        _materialize_result_artifacts,
    )

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        result_dir = run_dir / "results" / "candidate_a"
        result_dir.mkdir(parents=True)
        (result_dir / "result_summary.json").write_text(
            json.dumps(
                {
                    "variant_name": "candidate_a",
                    "current_aggregate": {"completed_required_eval_units": 3},
                    "evaluation_units": 3,
                }
            ),
            encoding="utf-8",
        )

        findings = _materialize_result_artifacts(run_dir=run_dir, gen_id=0)

    assert findings == []


def test_task_specific_metric_names_only_score_when_task_config_declares_them() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
        _materialize_result_artifacts,
    )

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        result_dir = run_dir / "results" / "candidate_a"
        result_dir.mkdir(parents=True)
        (result_dir / "result_summary.json").write_text(
            json.dumps(
                {
                    "variant_name": "candidate_a",
                    "generation_id": 0,
                    "current_aggregate": {
                        "future_fitness": 99.0,
                        "test_accuracy": 0.99,
                        "completed_required_eval_units": 3,
                    },
                    "evaluation_units": 3,
                }
            ),
            encoding="utf-8",
        )

        generic_findings = _materialize_result_artifacts(run_dir=run_dir, gen_id=0)
        task_configured_findings = _materialize_result_artifacts(
            run_dir=run_dir,
            gen_id=0,
            scoring_metric_keys=("future_fitness",),
        )

    assert generic_findings == []
    assert len(task_configured_findings) == 1
    assert task_configured_findings[0]["metrics"]["future_fitness"] == 99.0
    assert "primary_score=99.0" in task_configured_findings[0]["content"]


def test_unreadable_result_summary_removes_stale_materialized_finding() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
        _materialize_result_artifacts,
    )

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        result_dir = run_dir / "results" / "candidate_a"
        result_dir.mkdir(parents=True)
        summary_path = result_dir / "result_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "variant_name": "candidate_a",
                    "generation_id": 0,
                    "scored_complete": True,
                    "current_aggregate": {"score": 0.81, "completed_required_eval_units": 3},
                    "evaluation_units": 3,
                }
            ),
            encoding="utf-8",
        )
        [finding] = _materialize_result_artifacts(run_dir=run_dir, gen_id=0)
        materialized_path = run_dir / "shared_findings" / (f"{finding['id']}_candidate_a.json")
        assert materialized_path.exists()

        summary_path.write_text("{not valid json", encoding="utf-8")
        findings = _materialize_result_artifacts(run_dir=run_dir, gen_id=0)

    assert findings == []
    assert not materialized_path.exists()


def test_deleted_result_summary_removes_stale_materialized_finding() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
        _materialize_result_artifacts,
    )

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        result_dir = run_dir / "results" / "candidate_a"
        result_dir.mkdir(parents=True)
        summary_path = result_dir / "result_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "variant_name": "candidate_a",
                    "generation_id": 0,
                    "current_aggregate": {"score": 0.81, "completed_required_eval_units": 3},
                    "evaluation_units": 3,
                }
            ),
            encoding="utf-8",
        )
        [finding] = _materialize_result_artifacts(run_dir=run_dir, gen_id=0)
        materialized_path = run_dir / "shared_findings" / (f"{finding['id']}_candidate_a.json")
        assert materialized_path.exists()

        summary_path.unlink()
        findings = _materialize_result_artifacts(run_dir=run_dir, gen_id=0)

    assert findings == []
    assert not materialized_path.exists()


def test_unsupported_result_summary_shape_removes_stale_materialized_finding() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
        _materialize_result_artifacts,
    )

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        result_dir = run_dir / "results" / "candidate_a"
        result_dir.mkdir(parents=True)
        summary_path = result_dir / "result_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "variant_name": "candidate_a",
                    "generation_id": 0,
                    "current_aggregate": {"score": 0.81, "completed_required_eval_units": 3},
                    "evaluation_units": 3,
                }
            ),
            encoding="utf-8",
        )
        [finding] = _materialize_result_artifacts(run_dir=run_dir, gen_id=0)
        materialized_path = run_dir / "shared_findings" / (f"{finding['id']}_candidate_a.json")
        assert materialized_path.exists()

        summary_path.write_text(
            json.dumps({"variant_name": "candidate_a", "metadata": {"status": "pending"}}),
            encoding="utf-8",
        )
        findings = _materialize_result_artifacts(run_dir=run_dir, gen_id=0)

    assert findings == []
    assert not materialized_path.exists()


def test_failed_cells_dict_result_summary_is_materialized_as_validation_candidate() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
        _materialize_result_artifacts,
    )

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        result_dir = run_dir / "results" / "candidate_a"
        result_dir.mkdir(parents=True)
        (result_dir / "result_summary.json").write_text(
            json.dumps(
                {
                    "variant_name": "candidate_a",
                    "failed_cells": {"fold_2": "runtime error"},
                    "current_aggregate": {"score": 0.81, "completed_required_eval_units": 3},
                    "evaluation_units": 3,
                }
            ),
            encoding="utf-8",
        )

        [finding] = _materialize_result_artifacts(run_dir=run_dir, gen_id=0)

    metrics = finding["metrics"]
    assert metrics["result_status"] == "partial_cohort"
    assert metrics["excluded_from_durable_frontier"] is True
    assert metrics["exclusion_reason"] == "preliminary_or_incomplete_evidence"


def test_failed_cells_count_alias_result_summary_is_materialized_as_validation_candidate() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
        _materialize_result_artifacts,
    )

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        result_dir = run_dir / "results" / "candidate_a"
        result_dir.mkdir(parents=True)
        (result_dir / "result_summary.json").write_text(
            json.dumps(
                {
                    "variant_name": "candidate_a",
                    "generation_id": 0,
                    "current_aggregate": {
                        "score": 0.81,
                        "completed_required_eval_units": 3,
                        "n_failed_cells": 1,
                    },
                    "evaluation_units": 3,
                }
            ),
            encoding="utf-8",
        )

        findings = _materialize_result_artifacts(run_dir=run_dir, gen_id=0)

    assert len(findings) == 1
    metrics = findings[0]["metrics"]
    assert metrics["result_status"] == "partial_cohort"
    assert metrics["excluded_from_durable_frontier"] is True
    assert metrics["exclusion_reason"] == "preliminary_or_incomplete_evidence"


def test_materialized_generic_result_summary_does_not_invent_null_tier() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
        _materialize_result_artifacts,
    )

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        result_dir = run_dir / "results" / "candidate_a"
        result_dir.mkdir(parents=True)
        (result_dir / "result_summary.json").write_text(
            json.dumps(
                {
                    "variant_name": "candidate_a",
                    "generation_id": 0,
                    "scored_complete": True,
                    "current_aggregate": {"score": 0.81, "completed_required_eval_units": 3},
                    "evaluation_units": 3,
                }
            ),
            encoding="utf-8",
        )

        [finding] = _materialize_result_artifacts(run_dir=run_dir, gen_id=0)

    assert "tier" not in finding["metrics"]
    assert finding["metrics"]["scored_complete"] is True


def test_generic_result_summary_preserves_explicit_promotion_eligible() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
        _materialize_result_artifacts,
    )

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        result_dir = run_dir / "results" / "candidate_a"
        result_dir.mkdir(parents=True)
        (result_dir / "result_summary.json").write_text(
            json.dumps(
                {
                    "variant_name": "candidate_a",
                    "promotion_eligible": True,
                    "scored_complete": True,
                    "generation_id": 0,
                    "current_aggregate": {"score": 0.81, "completed_required_eval_units": 3},
                    "evaluation_units": 3,
                }
            ),
            encoding="utf-8",
        )

        [finding] = _materialize_result_artifacts(run_dir=run_dir, gen_id=0)

    assert finding["metrics"]["promotion_eligible"] is True


def test_equalized_variant_name_is_not_default_benchmark_floor() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
        _infer_strategy_family,
    )

    assert (
        _infer_strategy_family(
            "equalized_attention_candidate",
            {"current_aggregate": {"score": 0.7}},
        )
        == "task_candidate"
    )


def test_generic_gem_record_does_not_invent_trading_admission_metrics() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager
    from praxist.task_spec import GemsConfig

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        mgr = GemsManager(
            run_dir=run_dir,
            task_spec=SimpleNamespace(gems=GemsConfig(enabled=True)),
            frontier=SimpleNamespace(),
        )
        record = mgr._write_gem_finding(
            entry={
                "finding_id": "f_score",
                "variant_name": "candidate_model",
                "frontier_lane": "performance",
                "metric_value": 0.81,
                "metrics": {
                    "strategy_family": "learned_candidate",
                    "completed_required_eval_units": 3,
                    "tier": "full_eval",
                },
            },
            rank=1,
            reset_count=1,
            next_cycle_index=1,
            completed_gen_id=0,
            reason="test",
        )
        finding_text = (run_dir / record["finding_path"]).read_text(encoding="utf-8")

    admission = record["admission_metrics"]
    for key in (
        "mean_active_alpha_vs_benchmark_pct",
        "q25_active_alpha_vs_benchmark_pct",
        "validation_2026_active_alpha_pct",
        "max_drawdown_pct",
        "mean_mdd_pct",
    ):
        assert key not in admission
    assert admission["primary_score"] == 0.81
    for term in ("full-window T1", "T1-or-better", "tiered protocol"):
        assert term not in finding_text


def test_result_artifact_gem_candidate_does_not_invent_missing_evidence_stage() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager
    from praxist.task_spec import GemsConfig

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        result_dir = run_dir / "results" / "candidate_a"
        result_dir.mkdir(parents=True)
        (result_dir / "tiered_eval_summary.json").write_text(
            json.dumps(
                {
                    "variant_name": "candidate_a",
                    "generation_id": 0,
                    "scored_complete": True,
                    "current_aggregate": {
                        "score": 0.81,
                        "completed_required_eval_units": 3,
                    },
                    "evaluation_units": 3,
                }
            ),
            encoding="utf-8",
        )
        mgr = GemsManager(
            run_dir=run_dir,
            task_spec=SimpleNamespace(gems=GemsConfig(enabled=True)),
            frontier=SimpleNamespace(),
        )

        [candidate] = mgr._result_artifact_gem_candidates()

    assert "evidence_stage" not in candidate


def test_failed_result_artifact_is_not_mature_evidence_gem_candidate() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager
    from praxist.task_spec import GemsConfig

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        for variant, score, status in (
            ("failed_candidate", 99.0, "failed"),
            ("protocol_invalid_candidate", 98.0, "protocol_invalid"),
            ("good_candidate_a", 3.0, "passed"),
            ("good_candidate_b", 2.0, "passed"),
            ("good_candidate_c", 1.0, "passed"),
        ):
            result_dir = run_dir / "results" / variant
            result_dir.mkdir(parents=True)
            (result_dir / "result_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": variant,
                        "generation_id": 0,
                        "result_status": status,
                        "scored_complete": status == "passed",
                        "protocol_integrity_status": (
                            "failed" if variant == "protocol_invalid_candidate" else "passed"
                        ),
                        "suspect_protocol": variant == "protocol_invalid_candidate",
                        "current_aggregate": {"score": score, "completed_required_eval_units": 3},
                        "evaluation_units": 3,
                    }
                ),
                encoding="utf-8",
            )
        mgr = GemsManager(
            run_dir=run_dir,
            task_spec=SimpleNamespace(
                gems=GemsConfig(
                    enabled=True,
                    selection_policy="mature_evidence_top_k",
                    min_mature_eval_units=3,
                )
            ),
            frontier=SimpleNamespace(),
        )

        selected = mgr._select_mature_evidence_topk_entries(
            {"lane_frontiers": {}, "cumulative_top": []}
        )

    assert [entry["variant_name"] for entry in selected] == [
        "good_candidate_a",
        "good_candidate_b",
        "good_candidate_c",
    ]


def test_result_artifact_with_failed_units_is_not_mature_evidence_gem_candidate() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager
    from praxist.task_spec import GemsConfig

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        bad_dir = run_dir / "results" / "high_score_partial_candidate"
        bad_dir.mkdir(parents=True)
        (bad_dir / "result_summary.json").write_text(
            json.dumps(
                {
                    "variant_name": "high_score_partial_candidate",
                    "generation_id": 0,
                    "failed_units": [{"unit_id": "case_2"}],
                    "scored_complete": True,
                    "current_aggregate": {"score": 99.0, "completed_required_eval_units": 3},
                    "evaluation_units": 3,
                }
            ),
            encoding="utf-8",
        )
        good_dir = run_dir / "results" / "good_candidate"
        good_dir.mkdir(parents=True)
        (good_dir / "result_summary.json").write_text(
            json.dumps(
                {
                    "variant_name": "good_candidate",
                    "generation_id": 0,
                    "scored_complete": True,
                    "current_aggregate": {"score": 1.0, "completed_required_eval_units": 3},
                    "evaluation_units": 3,
                }
            ),
            encoding="utf-8",
        )
        mgr = GemsManager(
            run_dir=run_dir,
            task_spec=SimpleNamespace(
                gems=GemsConfig(
                    enabled=True,
                    selection_policy="mature_evidence_top_k",
                    min_mature_eval_units=3,
                )
            ),
            frontier=SimpleNamespace(),
        )

        selected = mgr._select_mature_evidence_topk_entries(
            {"lane_frontiers": {}, "cumulative_top": []}
        )

    assert [entry["variant_name"] for entry in selected] == ["good_candidate"]


def test_unscored_evaluation_units_do_not_make_result_artifact_gem_candidate() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager
    from praxist.task_spec import GemsConfig

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        bad_dir = run_dir / "results" / "unscored_cells_candidate"
        bad_dir.mkdir(parents=True)
        (bad_dir / "result_summary.json").write_text(
            json.dumps(
                {
                    "variant_name": "unscored_cells_candidate",
                    "generation_id": 0,
                    "evaluation_unit_records": [{"unit_id": "case_1"}, {"unit_id": "case_2"}],
                    "evaluation_units": 3,
                }
            ),
            encoding="utf-8",
        )
        good_dir = run_dir / "results" / "good_candidate"
        good_dir.mkdir(parents=True)
        (good_dir / "result_summary.json").write_text(
            json.dumps(
                {
                    "variant_name": "good_candidate",
                    "generation_id": 0,
                    "complete_eval": True,
                    "current_aggregate": {"score": 1.5},
                    "evaluation_unit_records": [
                        {"unit_id": "case_1", "score": 1.0},
                        {"unit_id": "case_2", "score": 2.0},
                    ],
                    "evaluation_units": 3,
                }
            ),
            encoding="utf-8",
        )
        mgr = GemsManager(
            run_dir=run_dir,
            task_spec=SimpleNamespace(
                gems=GemsConfig(
                    enabled=True,
                    selection_policy="mature_evidence_top_k",
                    min_mature_eval_units=3,
                )
            ),
            frontier=SimpleNamespace(),
        )

        selected = mgr._select_mature_evidence_topk_entries(
            {"lane_frontiers": {}, "cumulative_top": []}
        )

    assert [entry["variant_name"] for entry in selected] == ["good_candidate"]


def test_result_artifact_without_source_generation_is_not_gem_candidate() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager
    from praxist.task_spec import GemsConfig

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        stale_dir = run_dir / "results" / "stale_unknown_candidate"
        stale_dir.mkdir(parents=True)
        (stale_dir / "result_summary.json").write_text(
            json.dumps(
                {
                    "variant_name": "stale_unknown_candidate",
                    "current_aggregate": {"score": 99.0, "completed_required_eval_units": 3},
                    "evaluation_units": 3,
                }
            ),
            encoding="utf-8",
        )
        good_dir = run_dir / "results" / "good_candidate"
        good_dir.mkdir(parents=True)
        (good_dir / "result_summary.json").write_text(
            json.dumps(
                {
                    "variant_name": "good_candidate",
                    "generation_id": 0,
                    "scored_complete": True,
                    "current_aggregate": {"score": 1.0, "completed_required_eval_units": 3},
                    "evaluation_units": 3,
                }
            ),
            encoding="utf-8",
        )
        mgr = GemsManager(
            run_dir=run_dir,
            task_spec=SimpleNamespace(
                gems=GemsConfig(
                    enabled=True,
                    selection_policy="mature_evidence_top_k",
                    min_mature_eval_units=3,
                )
            ),
            frontier=SimpleNamespace(),
        )

        selected = mgr._select_mature_evidence_topk_entries(
            {"lane_frontiers": {}, "cumulative_top": []}
        )

    assert [entry["variant_name"] for entry in selected] == ["good_candidate"]


def test_evidence_card_does_not_invent_missing_tier_or_promotion_fields() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.research_memory.card_builder import (
        build_card_from_finding,
    )

    with tempfile.TemporaryDirectory() as tmp:
        card = build_card_from_finding(
            {
                "id": "f1",
                "finding_type": "result",
                "title": "score improved",
                "metrics": {"score": 0.7},
                "variant_name": "candidate_model",
                "generation_id": 0,
            },
            run_dir=Path(tmp),
        )

    assert "promotion_eligible" not in card["metrics"]
    assert "tier" not in card["metrics"]
    assert "seed_count" not in card["metrics"]


def test_evidence_card_preserves_result_risk_and_provenance_markers() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.research_memory.card_builder import (
        build_card_from_finding,
    )

    with tempfile.TemporaryDirectory() as tmp:
        card = build_card_from_finding(
            {
                "id": "f1",
                "finding_type": "result",
                "title": "score improved but provenance is weak",
                "metrics": {
                    "score": 0.7,
                    "scored_complete": True,
                    "source_generation_low_confidence": True,
                    "source_generation_inference": "boundary_fallback",
                    "provenance_warning": "source_generation_boundary_fallback",
                    "suspect_protocol": True,
                    "excluded_from_durable_frontier": True,
                    "exclusion_reason": "protocol_integrity_failed",
                },
                "variant_name": "candidate_model",
                "generation_id": 0,
            },
            run_dir=Path(tmp),
        )

    metrics = card["metrics"]
    assert metrics["score"] == 0.7
    assert metrics["scored_complete"] is True
    assert metrics["source_generation_low_confidence"] is True
    assert metrics["source_generation_inference"] == "boundary_fallback"
    assert metrics["provenance_warning"] == "source_generation_boundary_fallback"
    assert metrics["suspect_protocol"] is True
    assert metrics["excluded_from_durable_frontier"] is True
    assert metrics["exclusion_reason"] == "protocol_integrity_failed"


def test_evidence_pack_lane_digest_does_not_invent_missing_tier_booleans() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
        _digest_lane_frontiers,
    )

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        frontier = run_dir / "frontier"
        frontier.mkdir()
        (frontier / "frontier_manifest.json").write_text(
            json.dumps(
                {
                    "lane_frontiers": {
                        "performance": [
                            {
                                "finding_id": "f1",
                                "variant_name": "candidate_model",
                                "lane_metric_name": "score",
                                "lane_metric_value": 0.7,
                                "scored_complete": True,
                                "mature_enough": True,
                                "metrics": {"score": 0.7},
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )

        digest = _digest_lane_frontiers(run_dir)

    [entry] = digest["performance"]
    for key in (
        "promotion_eligible",
        "clean_promotion_eligible",
        "scout_only",
        "tier",
        "candidate_tier",
        "tier_status",
    ):
        assert key not in entry


def test_evidence_pack_helpers_cover_cutoff_sanitize_and_role_edges() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.research_memory import (
        evidence_pack_builder as builder,
    )

    class Entry:
        def __init__(self, entry_id: str, data: dict, updated: int = 0) -> None:
            self.id = entry_id
            self.data = data
            self.last_updated_at = updated

    class ClaimLedger:
        def list_active(self):
            return [
                Entry(
                    "C1",
                    {
                        "generation_id": 1,
                        "title": "current",
                        "status": "active",
                        "confidence": 0.8,
                        "boundary": "scope",
                        "supports": ["E1"],
                        "challenges": ["E2"],
                        "missing_tests": ["m"] * 8,
                    },
                    2,
                ),
                Entry("C2", {"generation_id": 3, "title": "future"}, 3),
            ]

        def list_recently_killed(self, n=30):
            return [
                Entry("K1", {"source_evidence_id": "finding_gen1_peer0", "title": "killed"}, 1),
                Entry("K2", {"generation_id": 3, "title": "future killed"}, 3),
            ][:n]

    class FrontierLedger:
        def latest_per_axis(self):
            return {
                "axis": Entry(
                    "F1",
                    {
                        "current_anchor": {"id": "now"},
                        "previous_anchor": {"id": "old"},
                        "raw_delta": 1,
                        "generation_id": 1,
                    },
                )
            }

        def all(self):
            return [
                Entry("bad", {"axis": 5, "generation_id": 1}),
                Entry("future", {"axis": "a", "generation_id": 3}),
                Entry("current", {"axis": "a", "generation_id": 1, "raw_delta": 2}),
                Entry("newer", {"axis": "a", "generation_id": 2, "raw_delta": 3}),
                Entry("invalid", {"axis": "b", "generation_id": "bad"}),
            ]

    class RoleRoi:
        def __init__(self, entries):
            self._entries = entries

        def all(self):
            return self._entries

    assert builder._ledger_entry_generation(Entry("x", {})) is None
    assert builder._ledger_entry_generation(Entry(2.0, "bad")) == 2
    assert builder._ledger_entry_generation(Entry("x", {"sources": ["peer_gen4"]})) == 4
    assert builder._within_generation_cutoff(Entry("x", {"generation_id": 99}), None)
    assert builder._within_generation_cutoff(Entry("x", {"generation_id": 2}), 2)
    assert not builder._within_generation_cutoff(Entry("x", {"generation_id": 3}), 2)

    sanitized = builder._sanitize_value(
        {"bad": float("inf"), "tmpl": "{{ value }} {% raw %}", "list": [float("nan"), 1.0]}
    )
    assert sanitized["bad"] is None
    assert sanitized["list"] == [None, 1.0]
    assert "{\u200b{" in sanitized["tmpl"]

    claims = builder._digest_claims(ClaimLedger(), current_gen_id=1)
    assert [item["id"] for item in claims["active"]] == ["C1"]
    assert [item["id"] for item in claims["recently_killed"]] == ["K1"]

    assert builder._digest_frontier(FrontierLedger())["axis"]["raw_delta"] == 1
    assert builder._digest_frontier(FrontierLedger(), current_gen_id=2)["a"]["raw_delta"] == 3

    empty_roi = builder._digest_role_roi(RoleRoi([]), current_gen_id=1)
    assert "no role_roi entries" in empty_roi["note"]
    future_roi = builder._digest_role_roi(
        RoleRoi([Entry("roi", {"generation_id": 3, "per_role": {"builder": {}}})]),
        current_gen_id=1,
    )
    assert "cutoff" in future_roi["note"]
    current_roi = builder._digest_role_roi(
        RoleRoi(
            [
                Entry("old", {"generation_id": 1, "per_role": {"builder": {"wins": 1}}}),
                Entry("new", {"generation_id": 2, "per_role": {"skeptic": {"wins": 1}}}),
            ]
        ),
        current_gen_id=2,
    )
    assert current_roi["latest_recorded_gen"] == 2

    cards = [
        {
            "evidence_id": "builder",
            "quality": {"is_negative": False},
            "metrics": {"promotion_eligible": True},
            "interpretation": {"short": "synergy scaling champion"},
        },
        {
            "evidence_id": "skeptic",
            "quality": {"is_negative": True},
            "metrics": {"promotion_eligible": True},
            "interpretation": {"short": "baseline fairness control"},
        },
        {
            "evidence_id": "portfolio",
            "quality": {"is_negative": True, "is_retired": True},
            "interpretation": {"short": "anti_mainline online random bridge"},
        },
        {
            "evidence_id": "external",
            "quality": {"is_negative": True},
            "interpretation": {"short": "cross sentinel long"},
        },
        {"evidence_id": "bad_interp", "interpretation": "not-a-dict"},
    ]
    assert builder._role_filter("builder", cards[0]) > builder._role_filter("builder", cards[1])
    assert builder._role_filter("skeptic", cards[1]) > 0.9
    assert builder._role_filter("portfolio", cards[2]) > 0.9
    assert builder._role_filter("external_validity", cards[3]) > 0.9
    assert builder.build_role_private_pack("builder", [], {}, "mini", 3) == []
    assert builder.build_role_private_pack("builder", cards[:-1], {}, "high_stakes", 2)

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        frontier_dir = run_dir / "frontier"
        frontier_dir.mkdir()
        manifest_path = frontier_dir / "frontier_manifest.json"
        manifest_path.write_text("{bad", encoding="utf-8")
        assert builder._digest_lane_frontiers(run_dir) == {}
        assert builder._digest_frontier_lane_metadata(run_dir) == []
        manifest_path.write_text(
            json.dumps(
                {
                    "lane_frontiers": {
                        "performance": [
                            "bad",
                            {
                                "generation_id": 2,
                                "variant_name": "future",
                            },
                            {
                                "generation_id": 1,
                                "finding_id": "f1",
                                "variant_name": "v1",
                                "lane_metric_name": "score",
                                "lane_metric_value": 1.2,
                                "scored_complete": True,
                                "mature_enough": True,
                                "promotion_eligible": "promotable",
                                "clean_promotion_eligible": "failed",
                                "lane_lower_tier_candidate": 1,
                                "lane_non_promotable_candidate": 0,
                            },
                        ],
                        "bad": "not-list",
                    },
                    "frontier_lanes": [
                        "bad",
                        {
                            "name": "performance",
                            "description": "primary",
                            "k": 2,
                            "allow_lower_tier": True,
                            "filters": {"tier": "T1"},
                        },
                    ],
                    "validation_candidates": {
                        "cumulative": [
                            {
                                "generation_id": 1,
                                "finding_id": f"scout_{idx}",
                                "variant_name": f"scout_{idx}",
                                "metric_name": "score",
                                "metric_value": idx,
                                "metric_direction": "maximize",
                                "metrics": {
                                    "score": idx,
                                    "risk_metric": 100 - idx,
                                },
                                "frontier_entity_key": f"variant::scout_{idx}",
                                "source_result_path": f"results/scout_{idx}/summary.json",
                            }
                            for idx in range(18)
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )
        lane_digest = builder._digest_lane_frontiers(run_dir, current_gen_id=1)
        assert lane_digest["performance"][0]["variant_name"] == "v1"
        assert lane_digest["performance"][0]["promotion_eligible"] == "promotable"
        assert lane_digest["performance"][0]["clean_promotion_eligible"] == "failed"
        assert lane_digest["performance"][0]["lane_non_promotable_candidate"] is False
        lane_metadata = builder._digest_frontier_lane_metadata(run_dir)
        assert lane_metadata[0]["name"] == "performance"
        assert lane_metadata[0]["allow_lower_tier"] is True

        class ListLedger:
            def __init__(self, entries):
                self._entries = entries

            def all(self):
                return self._entries

            def list_recent(self, n=60):
                return self._entries[:n]

            def list_open(self):
                return self._entries

        shared = builder.build_shared_core(
            run_dir,
            panel_mode="mini",
            current_gen_id=1,
            target_decisions=["choose"],
            claim_ledger=ClaimLedger(),
            frontier_delta_ledger=FrontierLedger(),
            coverage_matrix=ListLedger(
                [
                    Entry(
                        "bridge",
                        {
                            "generation_id": 1,
                            "relation": "bridge",
                            "variant_pair": ["a", "b"],
                            "grid_dimension": "rho",
                            "bridge_points_tested": [0.1],
                        },
                    ),
                    Entry(
                        "grid",
                        {
                            "generation_id": 1,
                            "variant_family": "fam",
                            "parameter": "rho",
                            "values_tested": [1],
                        },
                    ),
                    Entry("future_grid", {"generation_id": 3, "variant_family": "future"}),
                ]
            ),
            negative_evidence_ledger=ListLedger(
                [Entry("neg", {"generation_id": 1, "title": "n", "summary": "s" * 300})]
            ),
            retired_claim_ledger=ListLedger(
                [Entry("ret", {"generation_id": 1, "title": "r", "revive_if": ["x"]})]
            ),
            dissent_ledger=ListLedger(
                [Entry("d", {"generation_id": 1, "disputed_claim_id": "C1", "status": "open"})]
            ),
            role_roi_ledger=RoleRoi(
                [Entry("roi", {"generation_id": 1, "per_role": {"builder": {"wins": 1}}})]
            ),
            findings_summary={"ok": True},
        )
        assert shared["coverage_matrix_digest"]["bridge_grids"][0]["pair"] == ["a", "b"]
        assert shared["coverage_matrix_digest"]["single_family_grids"][0]["variant_family"] == "fam"
        assert shared["negative_evidence_digest"][0]["summary"] == "s" * 200
        assert shared["retired_claims"][0]["id"] == "ret"
        assert shared["open_objections"][0]["id"] == "d"
        assert len(shared["validation_candidates"]) == 16
        assert shared["validation_candidates_meta"]["total"] == 18
        assert shared["validation_candidates_meta"]["returned"] == 16
        assert shared["validation_candidates_meta"]["truncated"] is True
        assert shared["validation_candidates_meta"]["cap"] == 16
        assert shared["validation_candidates_meta"]["validator_id_count"] == 54
        assert shared["validation_candidates"][0]["metrics"]["score"] == 17
        assert shared["validation_candidates"][0]["metrics"]["risk_metric"] == 83
        assert shared["validation_candidates_meta"]["full_source_ref"] == {
            "result_path": "research_memory/validation_candidates/gen_1_full.json",
            "generation_id": 1,
            "total": 18,
        }
        assert "validation_candidate_ids" not in shared


def test_evidence_pack_respects_current_generation_cutoff_across_sources() -> None:
    from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
        build_evidence_pack,
    )
    from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.claim_ledger import (
        ClaimLedger,
    )
    from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.coverage_matrix import (
        CoverageMatrix,
    )
    from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.dissent_ledger import (
        DissentLedger,
    )
    from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.frontier_delta_ledger import (
        FrontierDeltaLedger,
    )
    from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.negative_evidence_ledger import (
        NegativeEvidenceLedger,
    )
    from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.retired_claim_ledger import (
        RetiredClaimLedger,
    )
    from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.role_roi_ledger import (
        RoleROILedger,
    )
    from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        frontier = run_dir / "frontier"
        frontier.mkdir()
        (frontier / "frontier_manifest.json").write_text(
            json.dumps(
                {
                    "lane_frontiers": {
                        "performance": [
                            {
                                "finding_id": "current_lane",
                                "variant_name": "current_candidate",
                                "generation_id": 0,
                                "lane_metric_name": "score",
                                "lane_metric_value": 0.5,
                                "scored_complete": True,
                                "mature_enough": True,
                                "metrics": {"score": 0.5},
                            },
                            {
                                "finding_id": "future_lane",
                                "variant_name": "future_candidate",
                                "generation_id": 2,
                                "lane_metric_name": "score",
                                "lane_metric_value": 99.0,
                                "scored_complete": True,
                                "mature_enough": True,
                                "metrics": {"score": 99.0},
                            },
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        gems_dir = run_dir / "gems"
        gems_dir.mkdir()
        (gems_dir / "gems_state.json").write_text(
            json.dumps(
                {
                    "cycle_index": 1,
                    "reset_count": 1,
                    "cycle_start_generation": 1,
                    "gems": [
                        {
                            "gem_finding_id": "gem_current",
                            "variant_name": "current_gem",
                            "source_generation_id": 0,
                            "scored_complete": True,
                            "admission_metrics": {"score": 0.5},
                        },
                        {
                            "gem_finding_id": "gem_future",
                            "variant_name": "future_gem",
                            "source_generation_id": 2,
                            "scored_complete": True,
                            "admission_metrics": {"score": 99.0},
                        },
                    ],
                    "active_bottleneck_reports": [
                        {
                            "completed_generation": 0,
                            "records": [{"gem_type": "current_gap"}],
                            "soft_agenda_priors": {"current_prior": 0.1},
                        },
                        {
                            "completed_generation": 2,
                            "records": [{"gem_type": "future_gap"}],
                            "soft_agenda_priors": {"future_prior": 0.9},
                        },
                    ],
                    "latest_soft_agenda_priors": {"future_prior": 0.9},
                }
            ),
            encoding="utf-8",
        )
        fd = FrontierDeltaLedger(run_dir)
        fd.record_promote(
            generation_id=0,
            axis="mean_test_accuracy",
            previous_anchor=None,
            current_anchor={"variant_name": "current_frontier", "value": 0.5},
        )
        fd.record_promote(
            generation_id=2,
            axis="mean_test_accuracy",
            previous_anchor={"variant_name": "current_frontier", "value": 0.5},
            current_anchor={"variant_name": "future_frontier", "value": 99.0},
        )
        claims = ClaimLedger(run_dir)
        claims.upsert_claim(
            "claim_gen0",
            "current claim",
            "active",
            0.5,
            created_by="gen0_pi",
        )
        claims.upsert_claim(
            "claim_gen2",
            "future claim",
            "active",
            0.9,
            created_by="gen2_pi",
        )
        retired = RetiredClaimLedger(run_dir)
        retired.retire(
            "retired_gen2",
            "future retired claim",
            "future reason",
            "future boundary",
            ["future revive"],
            created_by="gen2_pi",
        )
        dissent = DissentLedger(run_dir)
        dissent.add(
            "dissent_gen2",
            "claim_gen2",
            {"skeptic": "future objection"},
            status="open",
            created_by="gen2_pi",
        )
        coverage = CoverageMatrix(run_dir)
        coverage.record_grid_point(
            "current_family",
            "lr",
            0.1,
            source_evidence_id="gen0_current_grid",
        )
        coverage.record_grid_point(
            "future_family",
            "lr",
            0.2,
            source_evidence_id="gen2_future_grid",
        )
        neg = NegativeEvidenceLedger(run_dir)
        neg.add(
            "neg_gen2",
            "future negative evidence",
            "failed_lineage",
            evidence_id="gen2_future_negative",
            summary="future negative summary",
            created_by="gen2_pi",
        )
        RoleROILedger(run_dir).record_gen_summary(
            2,
            {"future_role": {"future_signal": 1}},
            created_by="gen2_pi",
        )

        with patch.dict("os.environ", {"LOCAL_STORE_DIR": str(run_dir)}, clear=False):
            local_store.init_db()
            local_store.insert_finding(
                {
                    "id": "finding_current",
                    "finding_type": "result",
                    "title": "current result",
                    "content": "current",
                    "metrics": {"score": 0.5},
                    "variant_name": "current_db_variant",
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                }
            )
            local_store.insert_finding(
                {
                    "id": "finding_future",
                    "finding_type": "result",
                    "title": "future result",
                    "content": "future",
                    "metrics": {"score": 99.0},
                    "variant_name": "future_db_variant",
                    "peer_id": "gen2_peer0",
                    "generation_id": 2,
                }
            )

            pack = build_evidence_pack(
                run_dir=run_dir,
                panel_mode="mini",
                current_gen_id=0,
                target_decisions=["agenda"],
                pi_roles=["builder"],
            )

    payload = json.dumps(
        {
            "shared_core": pack.shared_core,
            "private_packs": pack.private_packs,
            "all_cards": pack.all_cards,
        },
        sort_keys=True,
        default=str,
    )
    for visible in (
        "current_candidate",
        "current_gem",
        "current_db_variant",
        "current_frontier",
        "current claim",
        "current_family",
    ):
        assert visible in payload
    for leaked in (
        "future_candidate",
        "future_gem",
        "future_db_variant",
        "future_frontier",
        "future_gap",
        "future_prior",
        "future claim",
        "future retired claim",
        "future objection",
        "future_family",
        "future negative evidence",
        "future_role",
    ):
        assert leaked not in payload


class EvidencePackBuilderUnittestCoverage(unittest.TestCase):
    def test_evidence_pack_helpers_cover_unittest_gate(self) -> None:
        test_evidence_pack_helpers_cover_cutoff_sanitize_and_role_edges()
