from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import tempfile
import threading
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _task_with_gems(**overrides):
    defaults = {
        "enabled": True,
        "reset_interval_generations": 6,
        "max_resets": 3,
        "max_gems_per_reset": 10,
        "max_gems_total": 4,
        "max_gems_per_family": 2,
        "min_frontier_entries": 1,
        "archive_ordinary_findings": True,
        "signature_top_k": 16,
        "signature_entries_per_lane": 8,
        "prompt_max_gems": 12,
        "include_lanes": [],
        "selection_policy": "frontier_lane_balanced",
        "min_mature_eval_units": 29,
        "gem_seeded_independent_peers": 4,
        "performance_lanes": ["confirmed_alpha", "alpha", "alpha_incubator"],
        "control_lanes": ["benchmark_floor", "diagnostic_control", "process_audit"],
        "bottleneck_detector_mode": "generic",
        "result_artifact_materialization": True,
        "result_artifact_default_lane": "alpha_incubator",
        "result_artifact_default_family": "learned_alpha",
        "evidence_stage_min_units": {
            "T1": 29,
            "T2": 29,
            "T3": 29,
            "full_T1": 29,
        },
        "primary_metric_keys": ["mean_test_taskscore", "future_fitness", "score"],
        "secondary_metric_keys": ["mean_active_alpha_vs_benchmark_pct"],
        "lower_tail_metric_keys": ["q25_active_alpha_vs_benchmark_pct"],
        "validation_metric_keys": ["validation_2026_active_alpha_pct"],
        "cost_metric_keys": ["max_drawdown_pct", "mean_mdd_pct"],
    }
    defaults.update(overrides)
    return SimpleNamespace(
        gems=SimpleNamespace(**defaults),
        evaluation=SimpleNamespace(
            maturity_policy={
                "complete_stage_labels": ["T1", "T2", "T3", "full_T1"],
                "preliminary_stage_labels": ["smoke", "scout", "partial"],
            }
        ),
    )


def _write_surface_narrowing_findings(shared: Path, *, generation_id: int = 0) -> None:
    shared.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        (shared / f"gen{generation_id}_peer{i}_surface.json").write_text(
            json.dumps(
                {
                    "id": f"surface_{generation_id}_{i}",
                    "generation_id": generation_id,
                    "finding_type": "result",
                    "variant_name": f"surface_probe_{generation_id}_{i}",
                    "title": "PPO objective probe" if i < 5 else "attention probe",
                    "metrics": {
                        "primary_metric_delta": -1.0,
                        "strategy_family": (
                            "task_defined_surface_probe"
                            if i < 5
                            else "task_defined_alternate_probe"
                        ),
                    },
                }
            ),
            encoding="utf-8",
        )


class _FakeFrontier:
    def __init__(self, run_dir: Path, manifest: dict):
        self.base_dir = run_dir / "frontier"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.base_dir / "frontier_manifest.json"
        self._manifest = manifest
        self._save_manifest()

    def get_manifest(self):
        return dict(self._manifest)

    def get_summary(self):
        return list(self._manifest.get("cumulative_top", []))

    def _save_manifest(self):
        self.manifest_path.write_text(json.dumps(self._manifest, indent=2), encoding="utf-8")


class GemsContractsTest(unittest.TestCase):
    def test_current_aggregate_rejection_markers_remain_visible_to_gems(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import gems

        entry = {
            "metrics": {
                "current_aggregate": {
                    "protocol_integrity_failed": True,
                    "validation_only_result": True,
                }
            }
        }

        self.assertTrue(gems._entry_has_validation_only_durability_marker(entry))
        self.assertTrue(gems._entry_has_hard_gem_rejection_marker(entry))

    def test_legacy_gems_compat_migrates_config_and_nested_entry_fields(self) -> None:
        from praxist.task_spec_compat import (
            migrate_legacy_gems_config,
            migrate_legacy_gems_entry,
        )

        raw_config = {
            "selection_policy": "full_window_top4",
            "min_full_t1_eval_cells": 29,
            "evidence_stage_min_cells": 3,
        }
        config, used = migrate_legacy_gems_config(raw_config)
        self.assertEqual(config["selection_policy"], "mature_evidence_top_k")
        self.assertEqual(config["min_mature_eval_units"], 29)
        self.assertEqual(config["evidence_stage_min_units"], 3)
        self.assertEqual(
            set(used),
            {"selection_policy", "min_full_t1_eval_cells", "evidence_stage_min_cells"},
        )
        self.assertNotIn("min_mature_eval_units", raw_config)

        raw_entry = {
            "gem_finding_id": "legacy-gem",
            "selection_policy": "full_window_performance_top_k",
            "_gems_min_full_t1_eval_cells": 29,
            "metrics": {
                "selection_policy": "full_window_top_k",
                "_gems_evidence_stage_min_cells": 3,
            },
            "admission_metrics": {
                "tier": "T1",
                "mean_test_taskscore": 1.0,
            },
        }
        migrated = migrate_legacy_gems_entry(raw_entry)
        self.assertEqual(migrated["selection_policy"], "mature_evidence_top_k")
        self.assertEqual(migrated["_gems_min_mature_eval_units"], 29)
        self.assertEqual(migrated["metrics"]["selection_policy"], "mature_evidence_top_k")
        self.assertEqual(migrated["metrics"]["_gems_evidence_stage_min_units"], 3)
        self.assertTrue(migrated["_legacy_committed_complete_evidence"])
        self.assertNotIn("_gems_min_mature_eval_units", raw_entry)

    def test_task_configured_complete_stage_overrides_stage_name_heuristics(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import gems

        entry = {
            "variant_name": "task_named_stage",
            "metrics": {
                "score": 1.0,
                "frontier_lane": "candidate",
                "evidence_stage": "scout",
                "evaluation_units": 3,
            },
        }
        policy = {
            "complete_stage_labels": ["scout"],
            "preliminary_stage_labels": [],
            "require_ratio_gate": False,
        }

        self.assertTrue(
            gems._entry_is_mature_gem_admission_candidate(
                entry,
                min_mature_eval_units=3,
                maturity_policy=policy,
            )
        )
        self.assertFalse(
            gems._entry_is_mature_gem_admission_candidate(
                entry,
                min_mature_eval_units=3,
            )
        )

        authorized_reduced = {
            **entry,
            "metrics": {
                **entry["metrics"],
                "scout_only": True,
                "partial_eval": True,
                "scored_complete": True,
                "promotion_eligible": True,
            },
        }
        self.assertTrue(
            gems._entry_is_clean_gem_admission_candidate(
                authorized_reduced,
                maturity_policy=policy,
            )
        )
        self.assertTrue(
            gems._entry_is_mature_gem_admission_candidate(
                authorized_reduced,
                min_mature_eval_units=3,
                maturity_policy=policy,
            )
        )

        for blocking_update in (
            {"scored_complete": False},
            {"promotion_eligible": False},
            {"protocol_integrity_failed": True},
        ):
            blocked = {
                **authorized_reduced,
                "metrics": {**authorized_reduced["metrics"], **blocking_update},
            }
            self.assertFalse(
                gems._entry_is_clean_gem_admission_candidate(
                    blocked,
                    maturity_policy=policy,
                )
            )

    def test_ratio_gate_accepts_truthful_partial_stage_without_stage_whitelist(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import gems

        entry = {
            "variant_name": "authorized_reduced",
            "metrics": {
                "score": 1.0,
                "evidence_stage": "reduced",
                "effort_ratio": 1.0,
                "coverage_ratio": 1.0,
                "partial_eval": True,
                "promotion_eligible": True,
            },
        }
        policy = {
            "complete_stage_labels": ["reduced"],
            "preliminary_stage_labels": ["diagnostic"],
            "require_ratio_gate": True,
        }

        self.assertTrue(
            gems._entry_is_mature_gem_admission_candidate(
                entry,
                min_mature_eval_units=100,
                maturity_policy=policy,
            )
        )
        self.assertTrue(
            gems._entry_is_mature_gem_admission_candidate(
                entry,
                min_mature_eval_units=100,
                maturity_policy={"require_ratio_gate": True},
            )
        )

    def test_raw_result_gems_candidate_uses_task_authorized_maturity_policy(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

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
                        "evaluation_units": 1,
                    }
                ),
                encoding="utf-8",
            )
            task = _task_with_gems(primary_metric_keys=["score"], min_mature_eval_units=1)
            task.evaluation.primary_metric = "score"
            task.evaluation.maturity_policy = {
                "complete_stage_labels": ["reduced"],
                "preliminary_stage_labels": ["diagnostic"],
            }
            manager = GemsManager(
                run_dir=run_dir,
                task_spec=task,
                frontier=_FakeFrontier(run_dir, {}),
            )

            candidates = manager._result_artifact_gem_candidates()

            self.assertEqual(len(candidates), 1)
            self.assertTrue(candidates[0]["metrics"]["scored_complete"])
            self.assertEqual(candidates[0]["metrics"]["result_status"], "scored_complete")
            self.assertTrue(candidates[0]["metrics"]["partial"])

    def test_raw_result_gems_recomputes_effective_config_provenance(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result_dir = run_dir / "results" / "candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "candidate",
                        "score": 1.0,
                        "scored_complete": True,
                        "promotion_eligible": True,
                        "evaluation_units": 1,
                        "effective_config": {"seed": 7},
                        "effective_config_complete": True,
                        "replication_of_effective_config_sha256": "a" * 64,
                        "current_aggregate": {
                            "score": 1.0,
                            "source_result_effective_config_sha256": "b" * 64,
                            "replication_effective_config_status": "matched",
                        },
                    }
                ),
                encoding="utf-8",
            )
            task = _task_with_gems(primary_metric_keys=["score"], min_mature_eval_units=1)
            task.evaluation.primary_metric = "score"
            manager = GemsManager(
                run_dir=run_dir,
                task_spec=task,
                frontier=_FakeFrontier(run_dir, {}),
            )

            candidates = manager._result_artifact_gem_candidates()

        self.assertEqual(len(candidates), 1)
        metrics = candidates[0]["metrics"]
        self.assertNotEqual(metrics["source_result_effective_config_sha256"], "b" * 64)
        self.assertEqual(metrics["replication_effective_config_status"], "mismatch")

    def test_ratio_mature_evidence_can_enter_gems_without_legacy_stage_markers(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import gems

        mature_entry = {
            "variant_name": "ratio_mature",
            "metrics": {
                "score": 1.0,
                "mature_enough": True,
                "maturity_basis": "effort_coverage_ratio",
                "effort_ratio": 0.9,
                "coverage_ratio": 0.9,
            },
        }
        policy = {
            "min_effort_ratio": 0.75,
            "min_coverage_ratio": 0.80,
            "require_ratio_gate": True,
        }

        self.assertTrue(
            gems._entry_is_clean_gem_admission_candidate(
                mature_entry,
                maturity_policy=policy,
            )
        )
        self.assertTrue(
            gems._entry_is_mature_gem_admission_candidate(
                mature_entry,
                min_mature_eval_units=29,
                evidence_stage_min_units={"T1": 29},
                maturity_policy=policy,
            )
        )
        stale_scout_label = {
            **mature_entry,
            "metrics": {
                **mature_entry["metrics"],
                "scout_only": True,
            },
        }
        self.assertTrue(
            gems._entry_is_clean_gem_admission_candidate(
                stale_scout_label,
                maturity_policy=policy,
            )
        )
        self.assertTrue(
            gems._entry_is_mature_gem_admission_candidate(
                stale_scout_label,
                min_mature_eval_units=29,
                evidence_stage_min_units={"T1": 29},
                maturity_policy=policy,
            )
        )
        strict_flag_without_ratios = {
            "variant_name": "flag_without_ratios",
            "metrics": {
                "score": 1.0,
                "mature_enough": True,
                "maturity_basis": "effort_coverage_ratio",
            },
        }
        legacy_complete_without_ratios = {
            "variant_name": "legacy_complete_without_ratios",
            "metrics": {
                "score": 1.0,
                "scored_complete": True,
                "complete_eval": True,
                "tier": "T1",
                "n_eval_cells": 29,
            },
        }
        self.assertFalse(
            gems._entry_is_clean_gem_admission_candidate(
                strict_flag_without_ratios,
                maturity_policy=policy,
            )
        )
        self.assertTrue(gems._entry_is_clean_gem_admission_candidate(strict_flag_without_ratios))
        self.assertFalse(
            gems._entry_is_clean_gem_admission_candidate(
                legacy_complete_without_ratios,
                maturity_policy=policy,
            )
        )
        self.assertFalse(
            gems._entry_is_mature_gem_admission_candidate(
                legacy_complete_without_ratios,
                min_mature_eval_units=29,
                evidence_stage_min_units={"T1": 29},
                maturity_policy=policy,
            )
        )

        for blocked in (
            {
                **mature_entry,
                "metrics": {
                    **mature_entry["metrics"],
                    "effort_ratio": 0.5,
                    "mature_enough": False,
                },
            },
            {
                **mature_entry,
                "metrics": {
                    **mature_entry["metrics"],
                    "suspect_protocol": True,
                },
            },
            {
                **mature_entry,
                "metrics": {
                    **mature_entry["metrics"],
                    "incomplete_eval": True,
                },
            },
        ):
            self.assertFalse(
                gems._entry_is_clean_gem_admission_candidate(
                    blocked,
                    maturity_policy=policy,
                )
            )
            self.assertFalse(
                gems._entry_is_mature_gem_admission_candidate(
                    blocked,
                    min_mature_eval_units=29,
                    evidence_stage_min_units={"T1": 29},
                    maturity_policy=policy,
                )
            )

    def test_gems_ratio_policy_applies_to_selection_and_compaction(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        policy = {
            "min_effort_ratio": 0.75,
            "min_coverage_ratio": 0.80,
            "require_ratio_gate": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier = _FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []})
            task = _task_with_gems(max_gems_total=4, max_gems_per_reset=4)
            task.evaluation = SimpleNamespace(maturity_policy=policy)
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            stale_scout_mature = {
                "finding_id": "f_stale",
                "variant_name": "stale_scout_mature",
                "generation_id": 0,
                "frontier_lane": "incubator",
                "metrics": {
                    "score": 2.0,
                    "scout_only": True,
                    "effort_ratio": 0.9,
                    "coverage_ratio": 0.9,
                },
            }
            no_ratio_complete = {
                "finding_id": "f_no_ratio",
                "variant_name": "no_ratio_complete",
                "generation_id": 0,
                "frontier_lane": "incubator",
                "metrics": {
                    "score": 3.0,
                    "scored_complete": True,
                    "complete_eval": True,
                    "tier": "T1",
                    "n_eval_cells": 29,
                },
            }

            selected = mgr._select_gem_entries(
                {
                    "lane_frontiers": {
                        "incubator": [stale_scout_mature, no_ratio_complete],
                    }
                },
                completed_gen_id=0,
            )
            self.assertEqual([entry["variant_name"] for entry in selected], ["stale_scout_mature"])

            compacted = mgr._compact_gems([stale_scout_mature, no_ratio_complete])
            self.assertEqual([entry["variant_name"] for entry in compacted], ["stale_scout_mature"])

    def test_ratio_mature_written_gem_survives_strict_compaction(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        policy = {
            "min_effort_ratio": 0.75,
            "min_coverage_ratio": 0.80,
            "require_ratio_gate": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            task = _task_with_gems(max_gems_total=4, max_gems_per_reset=4)
            task.evaluation = SimpleNamespace(maturity_policy=policy)
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=task,
                frontier=_FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []}),
            )
            source_entry = {
                "finding_id": "ratio_source",
                "variant_name": "ratio_source",
                "generation_id": 0,
                "frontier_lane": "alpha_incubator",
                "metrics": {
                    "score": 2.0,
                    "mean_test_taskscore": 2.0,
                    "effort_ratio": 0.9,
                    "coverage_ratio": 0.9,
                    "scout_only": True,
                },
                "current_aggregate": {
                    "child_variant_id": "ratio_source",
                    "source_result_path": "results/ratio_source.json",
                    "source_result_sha256": "ratio-source-sha",
                },
            }

            record = mgr._write_gem_finding(
                entry=source_entry,
                rank=1,
                reset_count=0,
                next_cycle_index=1,
                completed_gen_id=0,
                reason="test",
            )
            compacted = mgr._compact_gems(
                [record],
                sort_by_performance=True,
                preserve_lane_reserves=True,
                max_generation_id=0,
                allow_legacy_unknown_source=True,
            )

        self.assertEqual(len(compacted), 1)
        self.assertEqual(compacted[0]["gem_finding_id"], record["gem_finding_id"])
        self.assertIn("effort_ratio", compacted[0]["admission_metrics"])
        self.assertIn("coverage_ratio", compacted[0]["admission_metrics"])
        self.assertEqual(record["source_result_sha256"], "ratio-source-sha")
        self.assertEqual(
            record["current_aggregate"],
            {
                "child_variant_id": "ratio_source",
                "source_result_path": "results/ratio_source.json",
                "source_result_sha256": "ratio-source-sha",
            },
        )

    def test_late_quarantined_marker_blocks_gems_parenting_even_with_ratios(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import gems
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        policy = {
            "min_effort_ratio": 0.75,
            "min_coverage_ratio": 0.80,
            "require_ratio_gate": True,
        }
        entry = {
            "finding_id": "late_signal",
            "variant_name": "late_signal",
            "frontier_lane": "alpha_incubator",
            "metrics": {
                "score": 9.0,
                "effort_ratio": 0.9,
                "coverage_ratio": 0.9,
                "late_after_generation_boundary": True,
                "late_result_policy": "quarantined_signal",
            },
        }

        self.assertFalse(
            gems._entry_is_clean_gem_admission_candidate(entry, maturity_policy=policy)
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            task = _task_with_gems(max_gems_total=4, max_gems_per_reset=4)
            task.evaluation = SimpleNamespace(maturity_policy=policy)
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=task,
                frontier=_FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []}),
            )

            compacted = mgr._compact_gems([entry], allow_legacy_unknown_source=True)

        self.assertEqual(compacted, [])

    def test_mature_evidence_gem_admission_rejects_hyphenated_protocol_invalid(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import gems

        entry = {
            "variant_name": "fixed_weight_protocol_invalid",
            "metrics": {
                "mean_test_taskscore": 100.0,
                "evidence_stage": "full_T1",
                "scored_cell_count": 29,
                "complete_eval": True,
                "protocol_integrity_status": "protocol-invalid",
            },
        }

        self.assertTrue(gems._entry_has_explicit_gem_rejection_marker(entry))
        self.assertFalse(
            gems._entry_is_mature_gem_admission_candidate(
                entry,
                min_mature_eval_units=29,
                evidence_stage_min_units={"T1": 29},
            )
        )

    def test_gem_admission_metrics_do_not_invent_legacy_taskscore_alias(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import (
            GemsManager,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier = _FakeFrontier(run_dir, {"cumulative_top": []})
            task = _task_with_gems(
                primary_metric_keys=["score"],
                secondary_metric_keys=[],
                lower_tail_metric_keys=[],
                validation_metric_keys=[],
                cost_metric_keys=[],
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            generic = mgr._write_gem_finding(
                entry={
                    "finding_id": "generic",
                    "variant_name": "generic_score",
                    "metric_name": "score",
                    "metric_value": 12.0,
                    "metrics": {"score": 12.0},
                },
                rank=1,
                reset_count=1,
                next_cycle_index=1,
                completed_gen_id=0,
                reason="test",
            )
            legacy = mgr._write_gem_finding(
                entry={
                    "finding_id": "legacy",
                    "variant_name": "legacy_score",
                    "metric_name": "mean_test_taskscore",
                    "metric_value": 13.0,
                    "metrics": {"mean_test_taskscore": 13.0},
                },
                rank=2,
                reset_count=1,
                next_cycle_index=1,
                completed_gen_id=0,
                reason="test",
            )

        self.assertNotIn("mean_test_taskscore", generic["admission_metrics"])
        self.assertEqual(legacy["admission_metrics"]["mean_test_taskscore"], 13.0)

    def test_entry_primitive_helpers_cover_identity_metric_and_tier_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import gems

        self.assertIs(gems._safe_metric(True), True)
        self.assertIsNone(gems._safe_metric(float("nan")))
        self.assertIsNone(gems._safe_metric(float("inf")))
        self.assertEqual(gems._safe_metric("  " + "x" * 150), "x" * 120)
        self.assertEqual(
            gems._entry_extra({"extra": {"peer_role": "builder"}}), {"peer_role": "builder"}
        )
        self.assertEqual(
            gems._entry_extra({"extra": '{"peer_role": "builder"}'}), {"peer_role": "builder"}
        )
        self.assertEqual(gems._entry_extra({"extra": "{bad"}), {})
        self.assertEqual(
            gems._entry_field({"metrics": {"score": 1}, "details": {"score": 2}}, "score"),
            1,
        )
        self.assertTrue(gems._any_boolish_entry_field_true({"metrics": {"flag": "yes"}}, "flag"))
        self.assertTrue(gems._any_boolish_entry_field_false({"metrics": {"flag": "no"}}, "flag"))
        self.assertEqual(gems._gem_identity_token(" Bad/Name! "), "bad/name")
        self.assertEqual(gems._gem_explicit_entity_key("bad::value"), "")
        self.assertEqual(gems._gem_explicit_entity_key("variant::"), "")
        self.assertEqual(
            gems._variant_key({"metrics": {"result_path": "artifacts/result.json"}}),
            "artifact:artifacts/result.json",
        )
        self.assertEqual(
            gems._variant_key({"variant_name": "gen0_peer5_strategy_t1"}),
            "variant:strategy_t1",
        )
        self.assertEqual(
            gems._variant_key(
                {
                    "variant_name": "gen0_peer5_strategy_t1",
                    "evidence_stage": "T1",
                }
            ),
            "variant:strategy_t1",
        )
        self.assertEqual(
            gems._variant_key({"variant_name": "strategy_t1"}),
            "variant:strategy_t1",
        )
        self.assertEqual(
            gems._variant_key({"variant_name": "strategy_scout"}),
            "variant:strategy_scout",
        )
        self.assertEqual(
            gems._variant_key({"variant_name": "strategy_smoke"}),
            "variant:strategy_smoke",
        )
        with patch.object(gems.re, "findall", return_value=["bad"]):
            self.assertIsNone(gems._infer_generation_id_from_text("gen_bad"))

        metric_entry = {
            "metric_name": "score",
            "metric_value": "3.5",
            "lane_metric_name": "lane_score",
            "lane_metric_value": "4.5",
            "metrics": {"bad": "nan?", "count": "5.9", "flag": "yes", "off": "0"},
        }
        self.assertEqual(gems._metric_float(metric_entry, "score"), 3.5)
        self.assertEqual(gems._metric_float(metric_entry, "lane_score"), 4.5)
        self.assertEqual(gems._metric_float(metric_entry, "bad", default=7.0), 7.0)
        self.assertEqual(gems._metric_int(metric_entry, "count"), 5)
        self.assertEqual(gems._metric_int({"count": 2.9}, "count"), 2)
        self.assertEqual(gems._metric_int({"count": "bad"}, "count", default=4), 4)
        self.assertEqual(
            gems._entry_failure_evidence_count(
                {
                    "failed_cells": ["a", "b"],
                    "error_cells": {"x": 1},
                    "missing_cells": 2.7,
                    "incomplete_cells": "bad",
                    "n_failed_cells": "",
                    "n_error_cells": True,
                }
            ),
            6,
        )
        self.assertTrue(gems._boolish_entry_field(metric_entry, "flag"))
        self.assertFalse(gems._boolish_entry_field(metric_entry, "off"))
        self.assertIsNone(gems._boolish_entry_field({"flag": "maybe"}, "flag"))

        self.assertEqual(
            gems._entry_source_generation_id(
                {
                    "generation_id": "bad",
                    "metrics": {"source_generation_id": "2"},
                    "variant_name": "run_gen4_variant",
                }
            ),
            4,
        )
        complete_eval_cases = [
            (
                {
                    "variant_name": "alpha",
                    "metrics": {
                        "frontier_lane": "confirmed",
                        "tier": "T1",
                        "scored_cell_count": 29,
                        "score": 1.0,
                        "scored_complete": True,
                    },
                },
                True,
            ),
            (
                {
                    "variant_name": "alpha",
                    "metrics": {
                        "frontier_lane": "confirmed",
                        "scored_cell_count": 29,
                        "score": 1.0,
                        "complete_eval": True,
                    },
                },
                True,
            ),
            (
                {
                    "variant_name": "alpha",
                    "metrics": {"frontier_lane": "alpha", "tier": "T2", "scored_cell_count": 86},
                },
                False,
            ),
            (
                {
                    "variant_name": "alpha",
                    "metrics": {
                        "frontier_lane": "confirmed",
                        "tier": "T2",
                        "scored_cell_count": 87,
                        "score": 1.0,
                        "scored_complete": True,
                    },
                },
                True,
            ),
            (
                {
                    "variant_name": "alpha",
                    "metrics": {
                        "frontier_lane": "confirmed",
                        "tier": "T3",
                        "scored_cell_count": 144,
                        "score": 1.0,
                    },
                },
                False,
            ),
            (
                {
                    "variant_name": "alpha",
                    "metrics": {
                        "frontier_lane": "confirmed",
                        "tier": "T3",
                        "scored_cell_count": 145,
                        "score": 1.0,
                        "scored_complete": True,
                    },
                },
                True,
            ),
            (
                {
                    "variant_name": "alpha",
                    "metrics": {"frontier_lane": "alpha", "tier": "T1", "scored_cell_count": 28},
                },
                False,
            ),
            (
                {
                    "variant_name": "alpha",
                    "metrics": {
                        "frontier_lane": "confirmed",
                        "scored_cell_count": 29,
                        "score": 1.0,
                    },
                },
                False,
            ),
            (
                {
                    "variant_name": "alpha",
                    "metrics": {
                        "frontier_lane": "confirmed",
                        "evidence_stage": "full_eval",
                        "scored_cell_count": 29,
                        "score": 1.0,
                        "scored_complete": True,
                    },
                },
                True,
            ),
            (
                {
                    "variant_name": "alpha",
                    "metrics": {
                        "frontier_lane": "confirmed",
                        "evidence_stage": "promotion_attempt",
                        "scored_cell_count": 29,
                        "score": 1.0,
                        "scored_complete": True,
                    },
                },
                True,
            ),
            (
                {
                    "variant_name": "alpha",
                    "metrics": {
                        "frontier_lane": "confirmed",
                        "evidence_stage": "replication",
                        "scored_cell_count": 87,
                        "score": 1.0,
                        "scored_complete": True,
                    },
                },
                True,
            ),
            (
                {
                    "variant_name": "alpha",
                    "metrics": {
                        "frontier_lane": "confirmed",
                        "evidence_stage": "unknown",
                        "scored_cell_count": 29,
                        "score": 1.0,
                    },
                },
                False,
            ),
            (
                {
                    "variant_name": "alpha",
                    "metrics": {
                        "frontier_lane": "confirmed",
                        "evidence_stage": "non-tier",
                        "scored_cell_count": 29,
                        "score": 1.0,
                    },
                },
                False,
            ),
        ]
        for entry, expected in complete_eval_cases:
            self.assertIs(
                gems._is_mature_evaluation_or_better(
                    entry,
                    min_mature_eval_units=29,
                    evidence_stage_min_units={
                        "T1": 29,
                        "T2": 87,
                        "T3": 145,
                        "promotion_attempt": 29,
                        "replication": 87,
                    },
                ),
                expected,
            )
        self.assertFalse(
            gems._is_performance_entry(
                {"variant_name": "alpha", "metrics": {"mechanism_family": "control_family"}}
            )
        )
        self.assertTrue(
            gems._is_performance_entry(
                {
                    "variant_name": "candidate",
                    "strategy_family": "candidate",
                    "score": 0.1,
                }
            )
        )
        self.assertFalse(
            gems._is_performance_entry(
                {
                    "variant_name": "candidate",
                    "strategy_family": "candidate",
                    "score": 0,
                }
            )
        )
        self.assertEqual(gems._entry_family({"variant_name": "plain_ppo_v2"}), "")
        self.assertEqual(gems._entry_family({"variant_name": "offpolicy_replay_v2"}), "")
        self.assertEqual(gems._entry_family({"variant_name": "bconly_seed"}), "")
        self.assertEqual(gems._entry_family({"variant_name": "bc40_seed"}), "")
        self.assertEqual(gems._entry_family({"variant_name": "ua_ppo_v1"}), "")
        self.assertEqual(gems._entry_family({"variant_name": "dual_critic_v1"}), "")
        self.assertEqual(gems._entry_family({"variant_name": "cvar_risk_v1"}), "")
        self.assertEqual(gems._entry_family({"variant_name": "film_regime_v1"}), "")
        self.assertEqual(gems._entry_family({"variant_name": "score_calibration_v1"}), "")
        self.assertEqual(gems._entry_family({"variant_name": "sparse_score_v1"}), "")
        self.assertEqual(gems._entry_family({"variant_name": "ppo_v1"}), "")
        self.assertEqual(
            gems._entry_family({"metrics": {"strategy_family": "novel_family"}}), "novel_family"
        )
        reports = [
            "bad",
            {"completed_generation": "bad"},
            {"completed_generation": 1, "soft_agenda_priors": {"old": True}},
            {"completed_generation": 3, "soft_agenda_priors": {"future": True}},
        ]
        self.assertEqual(gems._bottleneck_report_generation("bad"), None)
        self.assertEqual(
            gems._filter_bottleneck_reports_for_generation(reports, 2),
            [reports[2]],
        )
        self.assertEqual(
            gems._latest_soft_priors_for_generation(reports, 2, {"fallback": True}),
            {"old": True},
        )
        self.assertEqual(
            gems._latest_soft_priors_for_generation([], 2, {"fallback": True}),
            {},
        )
        self.assertEqual(
            gems._latest_soft_priors_for_generation(reports, None, {"fallback": True}),
            {"fallback": True},
        )
        self.assertEqual(
            gems._state_bottleneck_reports({"active_bottleneck_reports": [reports[2], "bad"]}),
            [reports[2]],
        )
        self.assertEqual(gems._state_bottleneck_reports({"bottleneck_history": "bad"}), [])
        self.assertEqual(gems._evidence_rank({"metrics": {"tier_reached": "forced_t3"}}), 0)
        self.assertEqual(gems._evidence_rank({"metrics": {"tier_reached": "scout"}}), 1)
        self.assertEqual(gems._evidence_rank({"metrics": {"evidence_rank": 2.8}}), 2)
        self.assertEqual(gems._performance_lane_priority("incubator"), 0)
        self.assertEqual(gems._performance_lane_priority("confirmed"), 2)
        self.assertEqual(gems._performance_lane_priority("performance"), 1)
        self.assertEqual(gems._performance_lane_priority("alpha_incubator"), 0)

    def test_variant_key_preserves_sweep_child_identity(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import gems

        from_path = {
            "variant_name": "bridge_l1_eff_n_sweep",
            "admission_metrics": {
                "source_result_path": "results/bridge_l1_c005/tiered_eval_summary.json"
            },
        }
        from_frontier = {
            "variant_name": "bridge_l1_eff_n_sweep",
            "metrics": {"frontier_entity_key": "variant::bridge_l1_c005"},
        }
        from_child_id = {
            "variant_name": "bridge_l1_eff_n_sweep",
            "metrics": {
                "frontier_entity_key": "variant::bridge_l1_eff_n_sweep",
                "child_id": "bridge_l1_c015",
            },
        }
        sibling = {
            "variant_name": "bridge_l1_eff_n_sweep",
            "metrics": {"source_result_path": "results/bridge_l1_c025/tiered_eval_summary.json"},
        }
        shared_parent_a = {
            "variant_id": "shared_parent",
            "metrics": {"source_result_path": "results/child_a/final_summary.json"},
        }
        shared_parent_b = {
            "variant_id": "shared_parent",
            "metrics": {"source_result_path": "results/child_b/final_summary.json"},
        }

        self.assertEqual(gems._variant_key(from_path), "variant:bridge_l1_c005")
        self.assertEqual(gems._variant_key(from_frontier), "variant:bridge_l1_c005")
        self.assertEqual(gems._variant_key(from_child_id), "variant:bridge_l1_c015")
        self.assertEqual(gems._variant_key(sibling), "variant:bridge_l1_c025")
        self.assertEqual(gems._variant_key(shared_parent_a), "variant:shared_parent")
        self.assertEqual(gems._variant_key(shared_parent_b), "variant:shared_parent")

    def test_task_spec_parses_gems_config(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.yaml"
            path.write_text(
                """
task_id: t
task_name: T
description_file: description.md
gems:
  enabled: true
  reset_interval_generations: 6
  max_resets: 2
  max_gems_per_reset: 11
  max_gems_total: 4
  max_gems_per_family: 2
  selection_policy: mature_evidence_top_k
  min_mature_eval_units: 29
  gem_seeded_independent_peers: 4
  include_lanes: [confirmed, incubator]
""",
                encoding="utf-8",
            )
            (Path(tmp) / "description.md").write_text("x", encoding="utf-8")

            spec = load_task_spec(path)

            self.assertTrue(spec.gems.enabled)
            self.assertEqual(spec.gems.reset_interval_generations, 6)
            self.assertEqual(spec.gems.max_resets, 2)
            self.assertEqual(spec.gems.max_gems_per_reset, 11)
            self.assertEqual(spec.gems.max_gems_total, 4)
            self.assertEqual(spec.gems.max_gems_per_family, 2)
            self.assertEqual(spec.gems.selection_policy, "mature_evidence_top_k")
            self.assertEqual(spec.gems.min_mature_eval_units, 29)
            self.assertEqual(spec.gems.gem_seeded_independent_peers, 4)
            self.assertEqual(spec.gems.include_lanes, ["confirmed", "incubator"])

    def test_task_spec_translates_legacy_gems_inputs_at_load_boundary(self) -> None:
        from praxist import task_spec

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "task.yaml"
            path.write_text(
                json.dumps(
                    {
                        "task_id": "legacy",
                        "task_name": "Legacy",
                        "description_file": "description.md",
                        "gems": {
                            "enabled": True,
                            "selection_policy": "full_window_top_k",
                            "min_full_t1_eval_cells": 7,
                            "evidence_stage_min_cells": {"complete": 7},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "description.md").write_text("x", encoding="utf-8")

            with self.assertLogs("praxist.task_spec", level="WARNING"):
                spec = task_spec.load_task_spec(path)

        self.assertEqual(spec.gems.selection_policy, task_spec.GEMS_MATURE_EVIDENCE_TOP_K)
        self.assertEqual(spec.gems.min_mature_eval_units, 7)
        self.assertEqual(spec.gems.evidence_stage_min_units, {"complete": 7})

    def test_inferred_false_completion_does_not_reject_ratio_mature_gem(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import gems

        inferred = {
            "metrics": {
                "score": 1.0,
                "effort_ratio": 1.0,
                "coverage_ratio": 1.0,
                "scored_complete": False,
                "_inferred_scored_complete": True,
            }
        }
        explicit = {
            "metrics": {
                **inferred["metrics"],
                "_inferred_scored_complete": False,
            }
        }

        self.assertFalse(gems._entry_has_hard_gem_rejection_marker(inferred))
        self.assertTrue(gems._entry_has_generic_mature_evidence(inferred))
        self.assertTrue(gems._entry_has_hard_gem_rejection_marker(explicit))

    def test_current_lane_policy_overrides_stale_gem_parent_metadata(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        task = _task_with_gems()
        task.evaluation.frontier_lanes = [
            {"name": "diagnostic", "parent_eligible": False},
            {"name": "candidate_library", "parent_eligible": True},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            manager = GemsManager(
                run_dir=Path(tmp),
                task_spec=task,
                frontier=SimpleNamespace(),
            )

            stale_diagnostic = {
                "frontier_lane": "diagnostic",
                "parent_eligible": True,
                "admission_metrics": {"parent_eligible": True},
            }
            stale_unknown = {
                "frontier_lane": "removed_lane",
                "parent_eligible": True,
            }
            eligible = {
                "frontier_lane": "candidate_library",
                "parent_eligible": True,
            }
            vetoed = {
                "frontier_lane": "candidate_library",
                "parent_eligible": False,
            }

            self.assertFalse(manager._entry_parent_eligible(stale_diagnostic))
            self.assertFalse(manager._entry_parent_eligible(stale_unknown))
            self.assertTrue(manager._entry_parent_eligible(eligible))
            self.assertFalse(manager._entry_parent_eligible(vetoed))
            self.assertEqual(
                manager._compact_gems(
                    [
                        {
                            **stale_diagnostic,
                            "variant_name": "stale_diagnostic",
                            "generation_id": 0,
                            "score": 1.0,
                            "scored_complete": True,
                        }
                    ],
                    preserve_committed_gems=True,
                ),
                [],
            )

    def test_historical_committed_tier_gem_survives_resume_migration_only(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        task = _task_with_gems(selection_policy="frontier_lane_balanced")
        task.evaluation.frontier_lanes = [
            {"name": "candidate_library", "parent_eligible": True},
            {"name": "diagnostic", "parent_eligible": False},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            manager = GemsManager(
                run_dir=Path(tmp),
                task_spec=task,
                frontier=SimpleNamespace(),
            )
            legacy = {
                "gem_finding_id": "legacy-gem",
                "variant_name": "legacy-candidate",
                "source_generation_id": 2,
                "frontier_lane": "candidate_library",
                "admission_metrics": {"score": 1.0, "tier": "T1"},
            }
            explicit_incomplete = {
                **legacy,
                "gem_finding_id": "legacy-incomplete",
                "variant_name": "legacy-incomplete",
                "admission_metrics": {
                    "score": 2.0,
                    "tier": "T1",
                    "scored_complete": False,
                },
            }
            stale_lane = {
                **legacy,
                "gem_finding_id": "legacy-stale-lane",
                "variant_name": "legacy-stale-lane",
                "frontier_lane": "diagnostic",
            }

            active = manager.active_gems_from_state(
                {
                    "last_completed_generation": 2,
                    "gems": [legacy, explicit_incomplete, stale_lane],
                },
                max_generation_id=2,
            )

        self.assertEqual([entry["variant_name"] for entry in active], ["legacy-candidate"])
        self.assertTrue(active[0]["_legacy_committed_complete_evidence"])

    def test_task_spec_warns_for_missing_mature_evidence_threshold(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.yaml"
            path.write_text(
                """
task_id: t
task_name: T
description_file: description.md
gems:
  enabled: true
  selection_policy: mature_evidence_top_k
""",
                encoding="utf-8",
            )
            (Path(tmp) / "description.md").write_text("x", encoding="utf-8")

            with self.assertLogs("praxist.task_spec", level="WARNING"):
                spec = load_task_spec(path)

            self.assertEqual(spec.gems.min_mature_eval_units, 1)

    def test_task_spec_falls_back_from_task_specific_bottleneck_detector_mode(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.yaml"
            path.write_text(
                """
task_id: t
task_name: T
description_file: description.md
gems:
  enabled: true
  bottleneck_detector_mode: task_specific_detector
""",
                encoding="utf-8",
            )
            (Path(tmp) / "description.md").write_text("x", encoding="utf-8")

            with self.assertLogs("praxist.task_spec", level="WARNING"):
                spec = load_task_spec(path)

            self.assertEqual(spec.gems.bottleneck_detector_mode, "generic")

    def test_task_spec_ignores_gems_specific_thresholds_when_gems_disabled(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.yaml"
            path.write_text(
                """
task_id: t
task_name: T
description_file: description.md
gems:
  enabled: false
  selection_policy: mature_evidence_top_k
  bottleneck_detector_mode: task_specific_detector
""",
                encoding="utf-8",
            )
            (Path(tmp) / "description.md").write_text("x", encoding="utf-8")

            spec = load_task_spec(path)

            self.assertFalse(spec.gems.enabled)

    def test_task_spec_parses_quoted_gems_booleans(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.yaml"
            path.write_text(
                """
task_id: t
task_name: T
description_file: description.md
gems:
  enabled: "false"
  archive_ordinary_findings: "false"
  result_artifact_materialization: "false"
""",
                encoding="utf-8",
            )
            (Path(tmp) / "description.md").write_text("x", encoding="utf-8")

            spec = load_task_spec(path)

        self.assertFalse(spec.gems.enabled)
        self.assertFalse(spec.gems.archive_ordinary_findings)
        self.assertFalse(spec.gems.result_artifact_materialization)

    def test_task_spec_defaults_to_four_total_gems(self) -> None:
        from praxist.task_spec import GemsConfig, load_task_spec

        self.assertEqual(GemsConfig().max_gems_total, 4)
        self.assertEqual(GemsConfig().max_gems_per_reset, 4)
        self.assertEqual(GemsConfig().prompt_max_gems, 4)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.yaml"
            path.write_text(
                """
task_id: t
task_name: T
description_file: description.md
gems:
  enabled: true
""",
                encoding="utf-8",
            )
            (Path(tmp) / "description.md").write_text("x", encoding="utf-8")

            spec = load_task_spec(path)

            self.assertEqual(spec.gems.max_gems_total, 4)
            self.assertEqual(spec.gems.max_gems_per_reset, 4)
            self.assertEqual(spec.gems.prompt_max_gems, 4)
            self.assertEqual(spec.gems.gem_seeded_independent_peers, 0)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.yaml"
            path.write_text(
                """
task_id: t
task_name: T
description_file: description.md
gems:
  enabled: true
  max_gems_total: 12
""",
                encoding="utf-8",
            )
            (Path(tmp) / "description.md").write_text("x", encoding="utf-8")

            spec = load_task_spec(path)

            self.assertEqual(spec.gems.max_gems_total, 12)

    def test_gems_cycle_properties_and_prompt_context_bad_inputs_are_stable(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier = _FakeFrontier(run_dir, {"cumulative_top": []})
            mgr = GemsManager(run_dir=run_dir, task_spec=_task_with_gems(), frontier=frontier)
            mgr.save_state(
                {
                    "cycle_index": 2,
                    "cycle_start_generation": 6,
                    "reset_count": 1,
                    "gems": [
                        {
                            "finding_id": "g1",
                            "variant_name": "v1",
                            "source_generation_id": 5,
                            "metrics": {
                                "frontier_lane": "confirmed",
                                "score": 1.0,
                                "evidence_stage": "full_T1",
                            },
                        },
                        {
                            "finding_id": "g2",
                            "variant_name": "v2",
                            "source_generation_id": 5,
                            "metrics": {
                                "frontier_lane": "performance",
                                "score": 0.8,
                                "evidence_stage": "full_T1",
                            },
                        },
                    ],
                }
            )

            self.assertEqual(mgr.cycle_index, 2)
            self.assertEqual(mgr.cycle_start_generation, 6)

            context = mgr.prompt_context(absolute_gen_id=6, peer_index="bad", cohort_size="bad")

            self.assertEqual(context["logical_generation"], 0)
            self.assertEqual(context["gem_anchor_assignment_mode"], "gem_inheritance")
            self.assertEqual(context["primary_gem_anchor"]["finding_id"], "g1")
            self.assertEqual(context["gem_anchor_roster"], [])

    def test_gem_seeded_context_reserves_independent_exploration_slots(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier = _FakeFrontier(run_dir, {"cumulative_top": []})
            task = _task_with_gems(
                max_gems_per_reset=4,
                max_gems_total=4,
                gem_seeded_independent_peers=4,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            mgr.save_state(
                {
                    "enabled": True,
                    "cycle_index": 1,
                    "cycle_start_generation": 6,
                    "reset_count": 1,
                    "max_resets": 3,
                    "gems": [
                        {
                            "gem_finding_id": f"g{i}",
                            "variant_name": f"gem_variant_{i}",
                            "source_generation_id": 0,
                            "evidence_stage": "full_T1",
                            "admission_metrics": {"score": float(i + 1)},
                        }
                        for i in range(4)
                    ],
                }
            )

            context = mgr.prompt_context(6, peer_index=10, cohort_size=12)
            roster = context["gem_anchor_roster"]

            self.assertEqual(context["gem_inheritance_slots"], 8)
            self.assertEqual(context["independent_exploration_slots"], 4)
            self.assertIn("honor those slots", context["baseline_code_policy"])
            self.assertIn(
                "independent exploration or recombination", context["baseline_code_policy"]
            )
            self.assertEqual(
                context["gem_anchor_assignment_mode"],
                "independent_exploration_or_recombination",
            )
            self.assertEqual(
                sum(1 for item in roster if item["assignment_type"] == "gem_inheritance"),
                8,
            )
            self.assertEqual(
                sum(
                    1
                    for item in roster
                    if item["assignment_type"] == "independent_exploration_or_recombination"
                ),
                4,
            )
            independent = [
                item
                for item in roster
                if item["assignment_type"] == "independent_exploration_or_recombination"
            ]
            self.assertTrue(all(not item["primary_variant_name"] for item in independent))
            self.assertTrue(
                all("protected independent slot" in item["instruction"] for item in independent)
            )
            self.assertTrue(all("code parent" in item["instruction"] for item in independent))

    def test_gem_seeded_context_reserves_one_independent_slot_for_default_cohort(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier = _FakeFrontier(run_dir, {"cumulative_top": []})
            task = _task_with_gems(
                max_gems_per_reset=4,
                max_gems_total=4,
                gem_seeded_independent_peers=4,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            mgr.save_state(
                {
                    "enabled": True,
                    "cycle_index": 1,
                    "cycle_start_generation": 6,
                    "reset_count": 1,
                    "max_resets": 3,
                    "gems": [
                        {
                            "gem_finding_id": f"g{i}",
                            "variant_name": f"gem_variant_{i}",
                            "source_generation_id": 0,
                            "evidence_stage": "full_T1",
                            "admission_metrics": {"score": float(i + 1)},
                        }
                        for i in range(4)
                    ],
                }
            )

            inheritance = mgr.prompt_context(6, peer_index=3, cohort_size=5)
            independent = mgr.prompt_context(6, peer_index=4, cohort_size=5)

            self.assertEqual(inheritance["gem_inheritance_slots"], 4)
            self.assertEqual(inheritance["independent_exploration_slots"], 1)
            self.assertEqual(inheritance["gem_anchor_assignment_mode"], "gem_inheritance")
            self.assertEqual(
                independent["gem_anchor_assignment_mode"],
                "independent_exploration_or_recombination",
            )
            self.assertFalse(independent["primary_gem_anchor"])

    def test_gem_seeded_context_reserves_independent_slot_when_cohort_equals_gem_cap(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier = _FakeFrontier(run_dir, {"cumulative_top": []})
            task = _task_with_gems(
                max_gems_per_reset=4,
                max_gems_total=4,
                gem_seeded_independent_peers=4,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            mgr.save_state(
                {
                    "enabled": True,
                    "cycle_index": 1,
                    "cycle_start_generation": 6,
                    "reset_count": 1,
                    "max_resets": 3,
                    "gems": [
                        {
                            "gem_finding_id": f"g{i}",
                            "variant_name": f"gem_variant_{i}",
                            "source_generation_id": 0,
                            "evidence_stage": "full_T1",
                            "admission_metrics": {"score": float(i + 1)},
                        }
                        for i in range(4)
                    ],
                }
            )

            inheritance = mgr.prompt_context(6, peer_index=2, cohort_size=4)
            independent = mgr.prompt_context(6, peer_index=3, cohort_size=4)
            roster = independent["gem_anchor_roster"]

            self.assertEqual(inheritance["gem_inheritance_slots"], 3)
            self.assertEqual(independent["independent_exploration_slots"], 1)
            self.assertEqual(
                independent["gem_anchor_assignment_mode"],
                "independent_exploration_or_recombination",
            )
            self.assertFalse(independent["primary_gem_anchor"])
            self.assertEqual(
                sum(
                    1
                    for item in roster
                    if item["assignment_type"] == "independent_exploration_or_recombination"
                ),
                1,
            )

    def test_mature_evidence_topk_keeps_top_three_mature_variants_with_hard_violations(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        def candidate(
            name: str,
            score: float,
            *,
            hard_violations: int = 0,
            n_eval_cells: int = 29,
            scout_only: bool = False,
        ) -> dict:
            return {
                "finding_id": f"finding_{name}",
                "variant_name": f"gen5_{name}",
                "generation_id": 5,
                "frontier_lane": "alpha_incubator",
                "strategy_family": "learned_alpha",
                "metrics": {
                    "mean_test_taskscore": score,
                    "mean_active_alpha_vs_benchmark_pct": score,
                    "n_eval_cells": n_eval_cells,
                    "scored_cell_count": n_eval_cells,
                    "complete_eval": True,
                    "tier": "T1",
                    "n_hard_constraint_violations": hard_violations,
                    "scout_only": scout_only,
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "cumulative_top": [
                    candidate("partial_but_high", 99.0, n_eval_cells=6),
                    candidate("scout_but_high", 98.0, scout_only=True),
                    candidate("best_hard", 50.0, hard_violations=8),
                    candidate("second_hard", 40.0, hard_violations=4),
                    candidate("third_hard", 30.0, hard_violations=2),
                    candidate("clean_fourth", 20.0),
                ]
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(
                selection_policy="mature_evidence_top_k",
                max_gems_per_reset=4,
                max_gems_total=4,
                max_gems_per_family=1,
                min_mature_eval_units=29,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(  # noqa: SLF001 - contract regression test.
                manifest,
                existing_gems=[],
                completed_gen_id=5,
            )
            compact = mgr._compact_gems(  # noqa: SLF001 - contract regression test.
                selected,
                sort_by_performance=True,
                max_generation_id=5,
            )

            self.assertEqual(
                [entry["variant_name"] for entry in selected[:3]],
                ["gen5_best_hard", "gen5_second_hard", "gen5_third_hard"],
            )
            self.assertEqual(
                [entry["variant_name"] for entry in compact[:3]],
                ["gen5_best_hard", "gen5_second_hard", "gen5_third_hard"],
            )
            self.assertNotIn(
                "gen5_partial_but_high",
                {entry["variant_name"] for entry in compact},
            )
            self.assertNotIn(
                "gen5_scout_but_high",
                {entry["variant_name"] for entry in compact},
            )

    def test_mature_evidence_reset_final_state_keeps_top_hard_t1_over_rejected_legacy(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        def candidate(name: str, score: float) -> dict:
            return {
                "finding_id": f"finding_{name}",
                "variant_name": f"gen5_{name}",
                "generation_id": 5,
                "frontier_lane": "alpha_incubator",
                "strategy_family": "learned_alpha",
                "metrics": {
                    "mean_test_taskscore": score,
                    "mean_active_alpha_vs_benchmark_pct": score,
                    "n_eval_cells": 29,
                    "scored_cell_count": 29,
                    "complete_eval": True,
                    "tier": "T1",
                    "n_hard_constraint_violations": 3,
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "cumulative_top": [
                    candidate("best_hard", 50.0),
                    candidate("second_hard", 40.0),
                    candidate("third_hard", 30.0),
                    candidate("clean_fourth", 20.0),
                ]
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(
                selection_policy="mature_evidence_top_k",
                max_gems_per_reset=4,
                max_gems_total=4,
                max_gems_per_family=1,
                min_mature_eval_units=29,
                archive_ordinary_findings=False,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            state = {
                "enabled": True,
                "cycle_index": 0,
                "cycle_start_generation": 0,
                "reset_count": 0,
                "max_resets": 3,
                "gems": [
                    {
                        "gem_finding_id": "rejected_legacy_a",
                        "variant_name": "legacy_rejected_a",
                        "admission_metrics": {
                            "mean_test_taskscore": 999.0,
                            "promotion_eligible": False,
                        },
                    },
                    {
                        "gem_finding_id": "rejected_legacy_b",
                        "variant_name": "legacy_rejected_b",
                        "admission_metrics": {
                            "mean_test_taskscore": 998.0,
                            "promotion_eligible": False,
                        },
                    },
                    {
                        "gem_finding_id": "legacy_control_a",
                        "variant_name": "legacy_control_a",
                        "frontier_lane": "benchmark_floor",
                        "admission_metrics": {"mean_test_taskscore": 997.0},
                    },
                    {
                        "gem_finding_id": "legacy_control_b",
                        "variant_name": "legacy_control_b",
                        "frontier_lane": "diagnostic_control",
                        "admission_metrics": {"mean_test_taskscore": 996.0},
                    },
                ],
            }

            result = mgr._admit_gems_and_reset(  # noqa: SLF001 - reset contract test.
                state=state,
                manifest=manifest,
                signature_hash="sig",
                completed_gen_id=5,
                reason="test_top_hard_t1",
            )
            final_names = [gem["variant_name"] for gem in mgr.load_state()["gems"]]

            self.assertTrue(result.triggered)
            self.assertEqual(
                final_names[:3],
                ["gen5_best_hard", "gen5_second_hard", "gen5_third_hard"],
            )
            self.assertNotIn("legacy_rejected_a", final_names)
            self.assertNotIn("legacy_rejected_b", final_names)
            self.assertNotIn("legacy_control_a", final_names)
            self.assertNotIn("legacy_control_b", final_names)

    def test_mature_evidence_pending_recovery_keeps_hard_t1_and_rejects_control(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        hard_t1 = {
            "finding_id": "hard_t1",
            "variant_name": "gen5_hard_t1",
            "generation_id": 5,
            "frontier_lane": "alpha_incubator",
            "strategy_family": "learned_alpha",
            "metrics": {
                "mean_test_taskscore": 20.0,
                "mean_active_alpha_vs_benchmark_pct": 20.0,
                "complete_eval": True,
                "n_eval_cells": 29,
                "scored_cell_count": 29,
                "tier": "T1",
                "hard_constraint_violations": ["drawdown"],
            },
        }
        rejected = {
            "finding_id": "explicit_reject",
            "variant_name": "gen5_explicit_reject",
            "generation_id": 5,
            "frontier_lane": "alpha_incubator",
            "strategy_family": "learned_alpha",
            "metrics": {
                "mean_test_taskscore": 99.0,
                "mean_active_alpha_vs_benchmark_pct": 99.0,
                "complete_eval": True,
                "n_eval_cells": 29,
                "scored_cell_count": 29,
                "tier": "T1",
                "promotion_eligible": False,
            },
        }
        control = {
            "finding_id": "control_anchor",
            "variant_name": "benchmark_floor_anchor",
            "generation_id": 5,
            "frontier_lane": "benchmark_floor",
            "strategy_family": "diagnostic_control",
            "metrics": {
                "mean_test_taskscore": 98.0,
                "complete_eval": True,
                "n_eval_cells": 29,
                "scored_cell_count": 29,
                "tier": "T1",
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            selected = [hard_t1, rejected, control]
            frontier = _FakeFrontier(run_dir, {"lane_frontiers": {"alpha_incubator": selected}})
            task = _task_with_gems(
                selection_policy="mature_evidence_top_k",
                max_gems_per_reset=4,
                max_gems_total=4,
                archive_ordinary_findings=False,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            mgr.save_state(
                {
                    "enabled": True,
                    "cycle_index": 1,
                    "cycle_start_generation": 6,
                    "reset_count": 1,
                    "gems": [],
                    "pending_reset": {
                        "status": "pending",
                        "reset_count": 2,
                        "cycle_index": 2,
                        "completed_gen_id": 5,
                        "next_absolute_generation": 6,
                        "signature_hash": "sig",
                        "reason": "test_recovery",
                        "archive_dir": str(run_dir / "archive" / "gems_cycle_2_recovery"),
                        "selected_entries": selected,
                    },
                }
            )

            result = mgr.recover_pending_reset(completed_gen_id=5)
            final_names = [gem["variant_name"] for gem in mgr.load_state()["gems"]]

            self.assertTrue(result.triggered)
            self.assertEqual(final_names, ["gen5_hard_t1"])

    def test_mature_evidence_prompt_context_filters_stale_gems_before_assignment(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier = _FakeFrontier(run_dir, {"cumulative_top": []})
            task = _task_with_gems(
                selection_policy="mature_evidence_top_k",
                max_gems_per_reset=4,
                max_gems_total=4,
                min_mature_eval_units=29,
                gem_seeded_independent_peers=4,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            mgr.save_state(
                {
                    "enabled": True,
                    "cycle_index": 1,
                    "cycle_start_generation": 7,
                    "reset_count": 1,
                    "max_resets": 3,
                    "gems": [
                        {
                            "gem_finding_id": "stale_summary",
                            "variant_name": "gen2_stale_summary",
                            "source_generation_id": 2,
                            "summary_only": True,
                            "frontier_lane": "alpha_incubator",
                            "strategy_family": "learned_alpha",
                            "admission_metrics": {
                                "mean_test_taskscore": 100.0,
                                "mean_active_alpha_vs_benchmark_pct": 100.0,
                                "n_eval_cells": 29,
                                "scored_cell_count": 29,
                                "complete_eval": True,
                            },
                        },
                        {
                            "gem_finding_id": "future_gen7",
                            "variant_name": "gen7_future_stale",
                            "source_generation_id": 7,
                            "frontier_lane": "alpha_incubator",
                            "strategy_family": "learned_alpha",
                            "admission_metrics": {
                                "mean_test_taskscore": 99.0,
                                "mean_active_alpha_vs_benchmark_pct": 99.0,
                                "n_eval_cells": 29,
                                "scored_cell_count": 29,
                                "complete_eval": True,
                            },
                        },
                        {
                            "gem_finding_id": "eligible_gen6",
                            "variant_name": "gen6_valid_mature",
                            "source_generation_id": 6,
                            "frontier_lane": "alpha_incubator",
                            "strategy_family": "learned_alpha",
                            "admission_metrics": {
                                "mean_test_taskscore": 7.0,
                                "mean_active_alpha_vs_benchmark_pct": 7.0,
                                "n_eval_cells": 29,
                                "scored_cell_count": 29,
                                "complete_eval": True,
                            },
                        },
                    ],
                }
            )

            context = mgr.prompt_context(7, peer_index=0, cohort_size=12)

            self.assertEqual(context["gems_count"], 1)
            self.assertEqual(
                context["primary_gem_anchor"]["variant_name"],
                "gen6_valid_mature",
            )

    def test_prompt_context_allows_pruned_restart_gems_from_same_absolute_generation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier = _FakeFrontier(run_dir, {"cumulative_top": []})
            task = _task_with_gems(
                selection_policy="mature_evidence_top_k",
                max_gems_per_reset=4,
                max_gems_total=4,
                min_mature_eval_units=29,
                gem_seeded_independent_peers=4,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            mature = {
                "frontier_lane": "alpha_incubator",
                "strategy_family": "learned_alpha",
                "admission_metrics": {
                    "mean_test_taskscore": 10.0,
                    "mean_active_alpha_vs_benchmark_pct": 10.0,
                    "n_eval_cells": 29,
                    "scored_cell_count": 29,
                    "complete_eval": True,
                },
            }
            mgr.save_state(
                {
                    "enabled": True,
                    "cycle_index": 1,
                    "cycle_start_generation": 6,
                    "reset_count": 1,
                    "max_resets": 3,
                    "reset_events": [
                        {
                            "completed_gen_id": 6,
                            "next_absolute_generation": 6,
                            "operator_pruned_restart_generation": 6,
                            "committed": True,
                        }
                    ],
                    "gems": [
                        {
                            **mature,
                            "gem_finding_id": "same_gen",
                            "variant_name": "gen6_valid_before_prune",
                            "source_generation_id": 6,
                        },
                        {
                            **mature,
                            "gem_finding_id": "future_gen7",
                            "variant_name": "gen7_future_after_restart",
                            "source_generation_id": 7,
                        },
                    ],
                }
            )

            context = mgr.prompt_context(6, peer_index=0, cohort_size=12)

            self.assertEqual(context["gems_count"], 1)
            self.assertEqual(
                context["primary_gem_anchor"]["variant_name"],
                "gen6_valid_before_prune",
            )

    def test_prompt_gems_loader_filters_state_for_pi_surfaces(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import (
            load_active_gems_for_prompt,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "task_spec.yaml").write_text(
                """
task_id: t
task_name: T
description_file: description.md
gems:
  enabled: true
  selection_policy: mature_evidence_top_k
  max_gems_total: 4
  max_gems_per_reset: 4
  min_mature_eval_units: 29
  performance_lanes: [confirmed_alpha, alpha, alpha_incubator]
  control_lanes: [benchmark_floor, diagnostic_control, process_audit]
  bottleneck_detector_mode: generic
  result_artifact_default_lane: alpha_incubator
  result_artifact_default_family: learned_alpha
  primary_metric_keys: [mean_test_taskscore]
  secondary_metric_keys: [mean_active_alpha_vs_benchmark_pct]
  cost_metric_keys: [max_drawdown_pct]
""",
                encoding="utf-8",
            )
            (run_dir / "description.md").write_text("x", encoding="utf-8")
            state_dir = run_dir / "gems"
            state_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / "gems_state.json").write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "cycle_index": 1,
                        "cycle_start_generation": 7,
                        "reset_count": 1,
                        "max_resets": 3,
                        "gems": [
                            {
                                "gem_finding_id": "bad",
                                "variant_name": "gen2_summary_only",
                                "source_generation_id": 2,
                                "summary_only": True,
                                "frontier_lane": "alpha_incubator",
                                "strategy_family": "learned_alpha",
                                "admission_metrics": {
                                    "mean_test_taskscore": 100.0,
                                    "mean_active_alpha_vs_benchmark_pct": 100.0,
                                    "n_eval_cells": 29,
                                    "scored_cell_count": 29,
                                },
                            },
                            {
                                "gem_finding_id": "future",
                                "variant_name": "gen7_future",
                                "source_generation_id": 7,
                                "frontier_lane": "alpha_incubator",
                                "strategy_family": "learned_alpha",
                                "admission_metrics": {
                                    "mean_test_taskscore": 99.0,
                                    "mean_active_alpha_vs_benchmark_pct": 99.0,
                                    "n_eval_cells": 29,
                                    "scored_cell_count": 29,
                                },
                            },
                            {
                                "gem_finding_id": "good",
                                "variant_name": "gen6_good",
                                "source_generation_id": 6,
                                "frontier_lane": "alpha_incubator",
                                "strategy_family": "learned_alpha",
                                "admission_metrics": {
                                    "mean_test_taskscore": 8.0,
                                    "mean_active_alpha_vs_benchmark_pct": 8.0,
                                    "n_eval_cells": 29,
                                    "scored_cell_count": 29,
                                    "complete_eval": True,
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            filtered = load_active_gems_for_prompt(run_dir)

            self.assertEqual([item["variant_name"] for item in filtered["entries"]], ["gen6_good"])

    def test_prompt_gems_loader_keeps_mature_evidence_hard_violation_gems(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import (
            load_active_gems_for_prompt,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "task_spec.yaml").write_text(
                """
task_id: t
task_name: T
description_file: description.md
gems:
  enabled: true
  selection_policy: mature_evidence_top_k
  max_gems_total: 4
  max_gems_per_reset: 4
  min_mature_eval_units: 29
  performance_lanes: [confirmed_alpha, alpha, alpha_incubator]
  control_lanes: [benchmark_floor, diagnostic_control, process_audit]
  bottleneck_detector_mode: generic
  result_artifact_default_lane: alpha_incubator
  result_artifact_default_family: learned_alpha
  primary_metric_keys: [mean_test_taskscore]
  secondary_metric_keys: [mean_active_alpha_vs_benchmark_pct]
  cost_metric_keys: [max_drawdown_pct]
""",
                encoding="utf-8",
            )
            (run_dir / "description.md").write_text("x", encoding="utf-8")
            state_dir = run_dir / "gems"
            state_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / "gems_state.json").write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "cycle_index": 1,
                        "cycle_start_generation": 6,
                        "reset_count": 1,
                        "gems": [
                            {
                                "gem_finding_id": "hard",
                                "variant_name": "gen5_hard_mature",
                                "source_generation_id": 5,
                                "frontier_lane": "alpha_incubator",
                                "strategy_family": "learned_alpha",
                                "admission_metrics": {
                                    "mean_test_taskscore": 30.0,
                                    "mean_active_alpha_vs_benchmark_pct": 30.0,
                                    "complete_eval": True,
                                    "tier": "T1",
                                    "n_eval_cells": 29,
                                    "scored_cell_count": 29,
                                    "promotion_eligible": True,
                                    "clean_promotion_eligible": False,
                                    "n_hard_constraint_violations": 3,
                                },
                            },
                            {
                                "gem_finding_id": "reject",
                                "variant_name": "gen5_rejected_mature",
                                "source_generation_id": 5,
                                "frontier_lane": "alpha_incubator",
                                "strategy_family": "learned_alpha",
                                "admission_metrics": {
                                    "mean_test_taskscore": 99.0,
                                    "complete_eval": True,
                                    "tier": "T1",
                                    "n_eval_cells": 29,
                                    "scored_cell_count": 29,
                                    "promotion_eligible": False,
                                },
                            },
                            {
                                "gem_finding_id": "control",
                                "variant_name": "benchmark_floor_anchor",
                                "source_generation_id": 5,
                                "frontier_lane": "benchmark_floor",
                                "strategy_family": "diagnostic_control",
                                "admission_metrics": {
                                    "mean_test_taskscore": 98.0,
                                    "complete_eval": True,
                                    "tier": "T1",
                                    "n_eval_cells": 29,
                                    "scored_cell_count": 29,
                                },
                            },
                            {
                                "gem_finding_id": "partial",
                                "variant_name": "gen5_partial",
                                "source_generation_id": 5,
                                "frontier_lane": "alpha_incubator",
                                "strategy_family": "learned_alpha",
                                "admission_metrics": {
                                    "mean_test_taskscore": 97.0,
                                    "tier": "T1",
                                    "n_eval_cells": 6,
                                    "scored_cell_count": 6,
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            filtered = load_active_gems_for_prompt(run_dir)

            self.assertEqual(
                [item["variant_name"] for item in filtered["entries"]],
                ["gen5_hard_mature"],
            )

    def test_prompt_gems_loader_raw_fallback_rejects_modern_low_evidence_rows(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import (
            load_active_gems_for_prompt,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            state_dir = run_dir / "gems"
            state_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / "gems_state.json").write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "gems": [
                            {
                                "gem_finding_id": "modern_low",
                                "variant_name": "modern_low",
                                "source_generation_id": 0,
                                "frontier_lane": "alpha_incubator",
                                "admission_metrics": {
                                    "score": 99.0,
                                    "n_eval_cells": 6,
                                },
                            },
                            {
                                "gem_finding_id": "legacy",
                                "variant_name": "legacy_anchor",
                                "source_generation_id": 0,
                                "admission_metrics": {
                                    "score": 1.0,
                                    "scored_complete": True,
                                },
                            },
                            {
                                "gem_finding_id": "legacy_unknown",
                                "variant_name": "legacy_unknown_anchor",
                                "admission_metrics": {
                                    "score": 0.5,
                                    "scored_complete": True,
                                },
                            },
                            {
                                "gem_finding_id": "control",
                                "variant_name": "benchmark_floor_anchor",
                                "source_generation_id": 0,
                                "frontier_lane": "benchmark_floor",
                                "admission_metrics": {"score": 0.0},
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            filtered = load_active_gems_for_prompt(run_dir)
            bounded = load_active_gems_for_prompt(run_dir, max_generation_id=0)

            self.assertEqual(
                [item["variant_name"] for item in filtered["entries"]],
                ["legacy_anchor", "legacy_unknown_anchor"],
            )
            self.assertEqual(
                [item["variant_name"] for item in bounded["entries"]],
                ["legacy_anchor"],
            )

    def test_prompt_gems_loader_trusts_committed_membership_and_explicit_provenance(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
            CANONICAL_STATE,
            attach_artifact_semantics,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.gems import (
            load_active_gems_for_prompt,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared_dir = run_dir / "shared_findings"
            shared_dir.mkdir(parents=True, exist_ok=True)
            (shared_dir / "committed.json").write_text(
                json.dumps(
                    {
                        "id": "committed",
                        "source_frontier_entry": {"generation_id": 0},
                    }
                ),
                encoding="utf-8",
            )
            state_dir = run_dir / "gems"
            state_dir.mkdir(parents=True, exist_ok=True)
            state = attach_artifact_semantics(
                {
                    "enabled": True,
                    "cycle_start_generation": 1,
                    "gems": [
                        {
                            "gem_finding_id": "committed",
                            "variant_name": "nextgen99_is_just_a_name",
                            "finding_path": "shared_findings/committed.json",
                            "admission_metrics": {
                                "generic_score": 1.0,
                                "result_capped": True,
                            },
                        },
                        {
                            "gem_finding_id": "state-bounded",
                            "variant_name": "legacy_state_bounded_candidate",
                            "admission_metrics": {"generic_score": 0.5},
                        },
                        {
                            "gem_finding_id": "future",
                            "variant_name": "future_candidate",
                            "source_generation_id": 2,
                            "admission_metrics": {"generic_score": 2.0},
                        },
                    ],
                },
                role=CANONICAL_STATE,
                stage="gems_state",
                runtime_fact_source=True,
            )
            (state_dir / "gems_state.json").write_text(
                json.dumps(state),
                encoding="utf-8",
            )

            filtered = load_active_gems_for_prompt(run_dir, max_generation_id=0)

        self.assertEqual(
            [entry["gem_finding_id"] for entry in filtered["entries"]],
            ["committed", "state-bounded"],
        )
        self.assertEqual(filtered["entries"][0]["source_generation_id"], 0)
        self.assertEqual(filtered["entries"][1]["source_generation_id"], 0)

    def test_prompt_gems_loader_uses_sidecar_generation_for_cutoff(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import (
            load_active_gems_for_prompt,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "task_spec.yaml").write_text(
                """
task_id: t
task_name: T
description_file: description.md
gems:
  enabled: true
  selection_policy: mature_evidence_top_k
  max_gems_total: 4
  max_gems_per_reset: 4
  min_mature_eval_units: 29
  performance_lanes: [confirmed_alpha, alpha, alpha_incubator]
  control_lanes: [benchmark_floor, diagnostic_control, process_audit]
  bottleneck_detector_mode: generic
  result_artifact_default_lane: alpha_incubator
  result_artifact_default_family: learned_alpha
  primary_metric_keys: [mean_test_taskscore]
  secondary_metric_keys: [mean_active_alpha_vs_benchmark_pct]
  cost_metric_keys: [max_drawdown_pct]
""",
                encoding="utf-8",
            )
            (run_dir / "description.md").write_text("x", encoding="utf-8")
            shared = run_dir / "shared_findings"
            shared.mkdir(parents=True)
            (shared / "legacy_current.json").write_text(
                json.dumps(
                    {
                        "id": "legacy_current",
                        "variant_name": "legacy_current_variant",
                        "generation_id": 0,
                        "source_frontier_entry": {"generation_id": 5},
                    }
                ),
                encoding="utf-8",
            )
            (shared / "legacy_future.json").write_text(
                json.dumps(
                    {
                        "id": "legacy_future",
                        "variant_name": "legacy_future_variant",
                        "generation_id": 0,
                        "source_frontier_entry": {"generation_id": 7},
                    }
                ),
                encoding="utf-8",
            )
            (shared / "legacy_unknown.json").write_text(
                json.dumps(
                    {
                        "id": "legacy_unknown",
                        "variant_name": "legacy_unknown_variant",
                        "generation_id": 0,
                    }
                ),
                encoding="utf-8",
            )
            gems_dir = run_dir / "gems"
            gems_dir.mkdir(parents=True)
            (gems_dir / "gems_state.json").write_text(
                json.dumps(
                    {
                        "cycle_index": 1,
                        "cycle_start_generation": 6,
                        "reset_count": 1,
                        "gems": [
                            {
                                "gem_finding_id": "legacy_current",
                                "variant_name": "legacy_current_variant",
                                "frontier_lane": "alpha_incubator",
                                "strategy_family": "learned_alpha",
                                "admission_metrics": {
                                    "mean_test_taskscore": 8.0,
                                    "mean_active_alpha_vs_benchmark_pct": 8.0,
                                    "n_eval_cells": 29,
                                    "scored_cell_count": 29,
                                    "complete_eval": True,
                                },
                            },
                            {
                                "gem_finding_id": "legacy_future",
                                "variant_name": "gen1_legacy_future_variant",
                                "frontier_lane": "alpha_incubator",
                                "strategy_family": "learned_alpha",
                                "admission_metrics": {
                                    "mean_test_taskscore": 99.0,
                                    "mean_active_alpha_vs_benchmark_pct": 99.0,
                                    "n_eval_cells": 29,
                                    "scored_cell_count": 29,
                                    "complete_eval": True,
                                },
                            },
                            {
                                "gem_finding_id": "legacy_unknown",
                                "variant_name": "legacy_unknown_variant",
                                "frontier_lane": "alpha_incubator",
                                "strategy_family": "learned_alpha",
                                "admission_metrics": {
                                    "mean_test_taskscore": 77.0,
                                    "mean_active_alpha_vs_benchmark_pct": 77.0,
                                    "n_eval_cells": 29,
                                    "scored_cell_count": 29,
                                    "complete_eval": True,
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            filtered = load_active_gems_for_prompt(run_dir, max_generation_id=5)
            unbounded = load_active_gems_for_prompt(run_dir)

            self.assertEqual(
                [item["variant_name"] for item in filtered["entries"]],
                ["legacy_current_variant"],
            )
            self.assertIn(
                "legacy_unknown_variant",
                [item["variant_name"] for item in unbounded["entries"]],
            )

    def test_mature_evidence_topk_gems_requires_actual_eval_cells(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "alpha_incubator": [
                        {
                            "finding_id": "capped_high",
                            "variant_name": "capped_t1_high_score",
                            "generation_id": 0,
                            "metrics": {
                                "tier": "T1",
                                "final_status": "capped_at_T1",
                                "strategy_family": "learned_alpha",
                                "mean_test_taskscore": 99.0,
                                "mean_active_alpha_vs_benchmark_pct": 99.0,
                                "n_eval_cells": 29,
                                "scored_cell_count": 29,
                            },
                        },
                        {
                            "finding_id": "summary_bool_high",
                            "variant_name": "summary_bool_high_score",
                            "generation_id": 0,
                            "summary_only": True,
                            "metrics": {
                                "tier": "T1",
                                "strategy_family": "learned_alpha",
                                "mean_test_taskscore": 120.0,
                                "mean_active_alpha_vs_benchmark_pct": 120.0,
                                "n_eval_cells": 29,
                                "scored_cell_count": 29,
                            },
                        },
                        {
                            "finding_id": "not_complete_high",
                            "variant_name": "not_complete_high_score",
                            "generation_id": 0,
                            "scored_complete": False,
                            "metrics": {
                                "tier": "T1",
                                "strategy_family": "learned_alpha",
                                "mean_test_taskscore": 110.0,
                                "mean_active_alpha_vs_benchmark_pct": 110.0,
                                "n_eval_cells": 29,
                                "scored_cell_count": 29,
                            },
                        },
                        {
                            "finding_id": "full_lower",
                            "variant_name": "mature_lower_score",
                            "generation_id": 0,
                            "metrics": {
                                "tier": "T1",
                                "strategy_family": "learned_alpha",
                                "mean_test_taskscore": 9.0,
                                "mean_active_alpha_vs_benchmark_pct": 9.0,
                                "n_eval_cells": 29,
                                "scored_cell_count": 29,
                            },
                        },
                        {
                            "finding_id": "promotion_false_full",
                            "variant_name": "promotion_false_full_high",
                            "generation_id": 0,
                            "metrics": {
                                "tier": "T1",
                                "strategy_family": "learned_alpha",
                                "mean_test_taskscore": 130.0,
                                "mean_active_alpha_vs_benchmark_pct": 130.0,
                                "n_eval_cells": 29,
                                "scored_cell_count": 29,
                                "complete_eval": True,
                                "promotion_eligible": False,
                            },
                        },
                        {
                            "finding_id": "hard_constraint_full",
                            "variant_name": "hard_constraint_full_high",
                            "generation_id": 0,
                            "metrics": {
                                "tier": "T1",
                                "strategy_family": "learned_alpha",
                                "mean_test_taskscore": 125.0,
                                "mean_active_alpha_vs_benchmark_pct": 125.0,
                                "n_eval_cells": 29,
                                "scored_cell_count": 29,
                                "complete_eval": True,
                                "hard_constraint_violations": ["max_weight"],
                            },
                        },
                    ]
                },
                "cumulative_top": [],
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(
                max_gems_per_reset=4,
                max_gems_total=4,
                selection_policy="mature_evidence_top_k",
                min_mature_eval_units=29,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(manifest)
            names = [entry["variant_name"] for entry in selected]

            self.assertEqual(names, ["hard_constraint_full_high", "mature_lower_score"])

    def test_default_gem_selection_filters_partial_frontier_entries(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "performance": [
                        {
                            "finding_id": "partial_high",
                            "variant_name": "partial_high_score",
                            "generation_id": 0,
                            "scored_complete": False,
                            "partial_cohort": True,
                            "metrics": {
                                "strategy_family": "learned_candidate",
                                "score": 99.0,
                                "n_eval_cells": 3,
                                "scored_cell_count": 3,
                            },
                        },
                        {
                            "finding_id": "complete_lower",
                            "variant_name": "complete_lower_score",
                            "generation_id": 0,
                            "metrics": {
                                "strategy_family": "learned_candidate",
                                "score": 1.0,
                                "n_eval_cells": 3,
                                "scored_cell_count": 3,
                                "scored_complete": True,
                            },
                        },
                    ]
                },
                "cumulative_top": [],
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(
                max_gems_per_reset=4,
                max_gems_total=4,
                selection_policy="frontier_lane_balanced",
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(manifest)
            names = [entry["variant_name"] for entry in selected]

            self.assertEqual(names, ["complete_lower_score"])

    def test_default_gem_selection_filters_mature_nonclean_entries(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "performance": [
                        {
                            "finding_id": "promotion_false",
                            "variant_name": "promotion_false_high",
                            "source_generation_id": 0,
                            "metrics": {
                                "strategy_family": "learned_candidate",
                                "score": 99.0,
                                "n_eval_cells": 29,
                                "scored_cell_count": 29,
                                "complete_eval": True,
                                "promotion_eligible": False,
                            },
                        },
                        {
                            "finding_id": "hard_constraint",
                            "variant_name": "hard_constraint_high",
                            "source_generation_id": 0,
                            "metrics": {
                                "strategy_family": "learned_candidate",
                                "score": 98.0,
                                "n_eval_cells": 29,
                                "scored_cell_count": 29,
                                "complete_eval": True,
                                "n_hard_constraint_violations": 1,
                            },
                        },
                        {
                            "finding_id": "clean_low",
                            "variant_name": "clean_low",
                            "source_generation_id": 0,
                            "metrics": {
                                "strategy_family": "learned_candidate",
                                "score": 1.0,
                                "n_eval_cells": 29,
                                "scored_cell_count": 29,
                                "complete_eval": True,
                            },
                        },
                    ]
                },
                "cumulative_top": [],
            }
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=_task_with_gems(
                    max_gems_per_reset=3,
                    max_gems_total=3,
                    selection_policy="frontier_lane_balanced",
                ),
                frontier=_FakeFrontier(run_dir, manifest),
            )

            selected = mgr._select_gem_entries(manifest, completed_gen_id=0)

        self.assertEqual([entry["variant_name"] for entry in selected], ["clean_low"])

    def test_default_gem_selection_filters_unscored_frontier_entries(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "performance": [
                        {
                            "finding_id": "unscored",
                            "variant_name": "unscored_frontier_entry",
                            "generation_id": 0,
                            "metrics": {
                                "strategy_family": "learned_candidate",
                                "n_eval_cells": 3,
                                "scored_cell_count": 3,
                                "scored_complete": True,
                            },
                        },
                        {
                            "finding_id": "scored",
                            "variant_name": "scored_frontier_entry",
                            "generation_id": 0,
                            "metrics": {
                                "strategy_family": "learned_candidate",
                                "score": 1.0,
                                "n_eval_cells": 3,
                                "scored_cell_count": 3,
                                "scored_complete": True,
                            },
                        },
                    ]
                },
                "cumulative_top": [],
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(selection_policy="frontier_lane_balanced")
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(manifest)
            names = [entry["variant_name"] for entry in selected]

            self.assertEqual(names, ["scored_frontier_entry"])

    def test_default_gem_selection_filters_unknown_generation_frontier_entries(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "performance": [
                        {
                            "finding_id": "unknown",
                            "variant_name": "unknown_generation_high_score",
                            "metrics": {
                                "strategy_family": "learned_candidate",
                                "score": 99.0,
                                "n_eval_cells": 3,
                                "scored_cell_count": 3,
                                "scored_complete": True,
                            },
                        },
                        {
                            "finding_id": "complete",
                            "variant_name": "complete_lower_score",
                            "generation_id": 0,
                            "metrics": {
                                "strategy_family": "learned_candidate",
                                "score": 1.0,
                                "n_eval_cells": 3,
                                "scored_cell_count": 3,
                                "scored_complete": True,
                            },
                        },
                    ]
                },
                "cumulative_top": [],
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(selection_policy="frontier_lane_balanced")
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(manifest)
            names = [entry["variant_name"] for entry in selected]

            self.assertEqual(names, ["complete_lower_score"])

    def test_default_gem_selection_filters_running_frontier_entries(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "performance": [
                        {
                            "finding_id": "running",
                            "variant_name": "running_high_score",
                            "generation_id": 0,
                            "result_status": "running",
                            "metrics": {
                                "strategy_family": "learned_candidate",
                                "score": 99.0,
                                "n_eval_cells": 3,
                                "scored_cell_count": 3,
                                "scored_complete": True,
                            },
                        },
                        {
                            "finding_id": "complete",
                            "variant_name": "complete_lower_score",
                            "generation_id": 0,
                            "metrics": {
                                "strategy_family": "learned_candidate",
                                "score": 1.0,
                                "n_eval_cells": 3,
                                "scored_cell_count": 3,
                                "scored_complete": True,
                            },
                        },
                    ]
                },
                "cumulative_top": [],
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(
                selection_policy="frontier_lane_balanced",
                performance_lanes=["performance"],
                bottleneck_detector_mode="generic",
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(manifest)
            names = [entry["variant_name"] for entry in selected]

            self.assertEqual(names, ["complete_lower_score"])

    def test_mature_evidence_topk_filters_failed_cells_manifest_entries(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "performance": [
                        {
                            "finding_id": "failed_cells",
                            "variant_name": "failed_cells_high_score",
                            "generation_id": 0,
                            "failed_cells": [{"cell_id": "fold_2"}],
                            "metrics": {
                                "strategy_family": "learned_candidate",
                                "score": 99.0,
                                "n_eval_cells": 3,
                                "scored_cell_count": 3,
                                "scored_complete": True,
                            },
                        },
                        {
                            "finding_id": "complete",
                            "variant_name": "complete_lower_score",
                            "generation_id": 0,
                            "metrics": {
                                "strategy_family": "learned_candidate",
                                "score": 1.0,
                                "n_eval_cells": 3,
                                "scored_cell_count": 3,
                                "scored_complete": True,
                            },
                        },
                    ]
                },
                "cumulative_top": [],
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(
                selection_policy="mature_evidence_top_k",
                min_mature_eval_units=3,
                performance_lanes=["performance"],
                bottleneck_detector_mode="generic",
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(manifest)
            names = [entry["variant_name"] for entry in selected]

            self.assertEqual(names, ["complete_lower_score"])

    def test_default_gem_selection_filters_failed_count_manifest_entries(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "performance": [
                        {
                            "finding_id": "failed_count",
                            "variant_name": "failed_count_high_score",
                            "generation_id": 0,
                            "metrics": {
                                "strategy_family": "learned_candidate",
                                "score": 99.0,
                                "n_eval_cells": 3,
                                "scored_cell_count": 3,
                                "scored_complete": True,
                                "failed_cell_count": 1,
                            },
                        },
                        {
                            "finding_id": "complete",
                            "variant_name": "complete_lower_score",
                            "generation_id": 0,
                            "metrics": {
                                "strategy_family": "learned_candidate",
                                "score": 1.0,
                                "n_eval_cells": 3,
                                "scored_cell_count": 3,
                                "scored_complete": True,
                            },
                        },
                    ]
                },
                "cumulative_top": [],
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(
                selection_policy="frontier_lane_balanced",
                performance_lanes=["performance"],
                bottleneck_detector_mode="generic",
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(manifest)
            names = [entry["variant_name"] for entry in selected]

            self.assertEqual(names, ["complete_lower_score"])

    def test_mature_evidence_topk_filters_unscored_manifest_entries(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "performance": [
                        {
                            "finding_id": "unscored",
                            "variant_name": "unscored_mature_evidence_entry",
                            "generation_id": 0,
                            "metrics": {
                                "strategy_family": "learned_candidate",
                                "n_eval_cells": 3,
                                "scored_cell_count": 3,
                                "scored_complete": True,
                            },
                        },
                        {
                            "finding_id": "scored",
                            "variant_name": "scored_mature_evidence_entry",
                            "generation_id": 0,
                            "metrics": {
                                "strategy_family": "learned_candidate",
                                "score": 1.0,
                                "n_eval_cells": 3,
                                "scored_cell_count": 3,
                                "scored_complete": True,
                            },
                        },
                    ]
                },
                "cumulative_top": [],
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(
                selection_policy="mature_evidence_top_k",
                min_mature_eval_units=3,
                performance_lanes=["performance"],
                result_artifact_default_lane="performance",
                result_artifact_default_family="learned_candidate",
                bottleneck_detector_mode="generic",
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(manifest)
            names = [entry["variant_name"] for entry in selected]

            self.assertEqual(names, ["scored_mature_evidence_entry"])

    def test_mature_evidence_topk_filters_future_manifest_entries_by_variant_name(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "alpha_incubator": [
                        {
                            "finding_id": "future_without_generation_field",
                            "variant_name": "gen5_repair_from_gen7_future",
                            "generation_id": 6,
                            "metrics": {
                                "tier": "T1",
                                "strategy_family": "learned_alpha",
                                "mean_test_taskscore": 99.0,
                                "mean_active_alpha_vs_benchmark_pct": 99.0,
                                "n_eval_cells": 29,
                                "scored_cell_count": 29,
                            },
                        },
                        {
                            "finding_id": "eligible_gen6",
                            "variant_name": "gen6_clean_mature",
                            "metrics": {
                                "tier": "T1",
                                "strategy_family": "learned_alpha",
                                "mean_test_taskscore": 8.0,
                                "mean_active_alpha_vs_benchmark_pct": 8.0,
                                "n_eval_cells": 29,
                                "scored_cell_count": 29,
                            },
                        },
                        {
                            "finding_id": "diagnostic_control_lane",
                            "variant_name": "clean_named_diag",
                            "frontier_lane": "alpha_incubator",
                            "metrics": {
                                "tier": "T1",
                                "strategy_family": "diagnostic_control",
                                "mean_test_taskscore": 98.0,
                                "mean_active_alpha_vs_benchmark_pct": 98.0,
                                "n_eval_cells": 29,
                                "scored_cell_count": 29,
                            },
                        },
                    ]
                },
                "cumulative_top": [],
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(
                max_gems_per_reset=4,
                max_gems_total=4,
                selection_policy="mature_evidence_top_k",
                min_mature_eval_units=29,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(manifest, completed_gen_id=6)
            names = [entry["variant_name"] for entry in selected]

            self.assertEqual(names, ["gen6_clean_mature"])

    def test_default_gem_compaction_filters_invalid_existing_gems(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier = _FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []})
            task = _task_with_gems(
                max_gems_per_reset=4,
                max_gems_total=4,
                selection_policy="frontier_lane_balanced",
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            active = mgr.active_gems_from_state(
                {
                    "gems": [
                        {
                            "gem_finding_id": "old_partial",
                            "variant_name": "old_partial_gem",
                            "frontier_lane": "performance",
                            "partial_cohort": True,
                            "scored_complete": False,
                            "admission_metrics": {"score": 99.0},
                        },
                        {
                            "gem_finding_id": "old_complete",
                            "variant_name": "old_complete_gem",
                            "frontier_lane": "performance",
                            "source_generation_id": 0,
                            "admission_metrics": {
                                "score": 1.0,
                                "scored_complete": True,
                            },
                        },
                    ]
                }
            )

            names = [entry["variant_name"] for entry in active]

            self.assertEqual(names, ["old_complete_gem"])

    def test_active_gems_preserve_committed_complete_entry_when_policy_adds_ratio_gate(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier = _FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []})
            task = _task_with_gems(selection_policy="frontier_lane_balanced")
            task.evaluation.maturity_policy = {
                "min_effort_ratio": 0.75,
                "min_coverage_ratio": 0.80,
                "require_ratio_gate": True,
            }
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            committed = {
                "gem_finding_id": "committed-complete",
                "variant_name": "committed-complete",
                "source_generation_id": 2,
                "frontier_lane": "performance",
                "scored_complete": True,
                "evidence_stage": "full_eval",
                "admission_metrics": {"score": 1.0},
            }

            known_ratio_failure = {
                **committed,
                "gem_finding_id": "known-ratio-failure",
                "variant_name": "known-ratio-failure",
                "admission_metrics": {
                    "score": 2.0,
                    "effort_ratio": 0.5,
                    "coverage_ratio": 0.9,
                },
            }
            non_parent = {
                **committed,
                "gem_finding_id": "non-parent",
                "variant_name": "non-parent",
                "parent_eligible": False,
            }
            complete_control = {
                **committed,
                "gem_finding_id": "complete-control",
                "variant_name": "complete-control",
                "frontier_lane": "diagnostic_control",
            }
            failed_control = {
                **complete_control,
                "gem_finding_id": "failed-control",
                "variant_name": "failed-control",
                "admission_metrics": {
                    "score": 3.0,
                    "effort_ratio": 0.5,
                    "coverage_ratio": 0.9,
                },
            }
            incomplete_control = {
                "gem_finding_id": "incomplete-control",
                "variant_name": "incomplete-control",
                "source_generation_id": 2,
                "frontier_lane": "diagnostic_control",
                "admission_metrics": {"score": 4.0},
            }

            active = mgr.active_gems_from_state(
                {
                    "gems": [
                        committed,
                        known_ratio_failure,
                        non_parent,
                        complete_control,
                        failed_control,
                        incomplete_control,
                    ]
                }
            )

        self.assertEqual(
            [entry["gem_finding_id"] for entry in active],
            ["committed-complete", "complete-control"],
        )

    def test_prompt_context_filters_existing_unknown_maturity_gems(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier = _FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []})
            task = _task_with_gems(
                max_gems_per_reset=4,
                max_gems_total=4,
                selection_policy="frontier_lane_balanced",
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            mgr.save_state(
                {
                    "enabled": True,
                    "cycle_index": 1,
                    "cycle_start_generation": 6,
                    "reset_count": 1,
                    "max_resets": 3,
                    "gems": [
                        {
                            "gem_finding_id": "unknown_existing",
                            "variant_name": "unknown_existing_gem",
                            "frontier_lane": "performance",
                            "source_generation_id": 0,
                            "admission_metrics": {"mean_test_taskscore": 99.0},
                        },
                        {
                            "gem_finding_id": "mature_existing",
                            "variant_name": "mature_existing_gem",
                            "frontier_lane": "performance",
                            "source_generation_id": 0,
                            "evidence_stage": "full_T1",
                            "admission_metrics": {"mean_test_taskscore": 1.0},
                        },
                    ],
                }
            )

            context = mgr.prompt_context(absolute_gen_id=6, peer_index=0, cohort_size=1)

            self.assertEqual(
                [entry["variant_name"] for entry in context["gems"]],
                ["mature_existing_gem"],
            )
            self.assertEqual(context["primary_gem_anchor"]["variant_name"], "mature_existing_gem")

    def test_mature_evidence_topk_compaction_rechecks_existing_gem_eligibility(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier = _FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []})
            task = _task_with_gems(
                max_gems_per_reset=4,
                max_gems_total=4,
                selection_policy="mature_evidence_top_k",
                min_mature_eval_units=29,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            compact = mgr._compact_gems(
                [
                    {
                        "gem_finding_id": "old_summary_gem",
                        "variant_name": "gen2_old_summary_only",
                        "source_generation_id": 2,
                        "summary_only": True,
                        "frontier_lane": "alpha_incubator",
                        "strategy_family": "learned_alpha",
                        "admission_metrics": {
                            "mean_test_taskscore": 99.0,
                            "mean_active_alpha_vs_benchmark_pct": 99.0,
                            "n_eval_cells": 29,
                            "scored_cell_count": 29,
                            "complete_eval": True,
                        },
                    },
                    {
                        "gem_finding_id": "future_gem",
                        "variant_name": "gen7_future_gem",
                        "source_generation_id": 7,
                        "frontier_lane": "alpha_incubator",
                        "strategy_family": "learned_alpha",
                        "admission_metrics": {
                            "mean_test_taskscore": 98.0,
                            "mean_active_alpha_vs_benchmark_pct": 98.0,
                            "n_eval_cells": 29,
                            "scored_cell_count": 29,
                            "complete_eval": True,
                        },
                    },
                    {
                        "gem_finding_id": "eligible_gem",
                        "variant_name": "gen6_eligible_gem",
                        "source_generation_id": 6,
                        "frontier_lane": "alpha_incubator",
                        "strategy_family": "learned_alpha",
                        "admission_metrics": {
                            "mean_test_taskscore": 7.0,
                            "mean_active_alpha_vs_benchmark_pct": 7.0,
                            "n_eval_cells": 29,
                            "scored_cell_count": 29,
                            "complete_eval": True,
                        },
                    },
                ],
                sort_by_performance=True,
                max_generation_id=6,
            )

            self.assertEqual([gem["variant_name"] for gem in compact], ["gen6_eligible_gem"])

    def test_mature_evidence_topk_gems_reads_result_artifacts_and_ranks_top4(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            results_dir = run_dir / "results"
            specs = [
                ("gen0_same_family_a", 10.0, 29),
                ("gen1_same_family_b", 13.0, 29),
                ("gen2_same_family_c", 11.0, 29),
                ("gen3_same_family_d", 12.0, 29),
                ("gen4_same_family_e", 9.0, 29),
                ("gen5_capped_high", 99.0, 29),
                ("gen7_future_leak_high", 98.0, 29),
                ("gen6_summary_only_high", 97.0, 29),
                ("gen6_not_scored_complete_high", 96.0, 29),
                ("control_baseline_high", 95.0, 29),
                ("clean_name_with_explicit_future_generation", 94.0, 29),
                ("gen6_scored_complete_false_status", 93.0, 29),
                ("gen6_not_scored_status", 92.0, 29),
                ("gen6_complete_eval_false_high", 94.5, 29),
                ("gen6_aggregate_complete_eval_false_high", 94.25, 29),
                ("gen6_promotion_false_high", 94.75, 29),
                ("gen6_hard_constraint_high", 94.6, 29),
                ("gen6_legacy_suspect_protocol_high", 101.0, 29),
                ("gen5_benchmark_relative_reward_high", 94.8, 29),
                ("gen5_benchmark_floor_anchor_high", 100.0, 29),
                ("gen6_control_shuffled_high", 91.0, 29),
                ("candidategen7future_explicit_name", 90.0, 29),
                ("gen6_validation_only_result_high", 102.0, 29),
                ("gen6_durability_scope_validation_high", 103.0, 29),
                ("gen6_late_policy_quarantined_high", 104.0, 29),
            ]
            for name, score, cells in specs:
                out = results_dir / name
                out.mkdir(parents=True, exist_ok=True)
                summary_only = "summary_only" in name
                not_scored_complete = "not_scored_complete" in name
                summary = {
                    "variant_name": "generic_family_name",
                    "tier_reached": "T1",
                    "completed_tier": "T1",
                    "tier_status": (
                        "capped_at_T1"
                        if "capped" in name
                        else (
                            "summary-only"
                            if "summary_only" in name
                            else (
                                "scored_complete=false"
                                if "scored_complete_false_status" in name
                                else (
                                    "not-scored" if "not_scored_status" in name else "stop_after_T1"
                                )
                            )
                        )
                    ),
                    "summary_only": summary_only,
                    "scored_complete": not not_scored_complete,
                    "scout_only": False,
                    "generation_id": (7 if "explicit_future_generation" in name else None),
                    "n_eval_cells": cells,
                    "current_aggregate": {
                        "mean_test_taskscore": score,
                        "mean_active_alpha_vs_benchmark_pct": score,
                        "max_drawdown_pct": 10.0,
                        "strategy_family": (
                            "diagnostic_control" if "control_shuffled" in name else "learned_alpha"
                        ),
                    },
                }
                if "aggregate_complete_eval_false" in name:
                    summary.pop("scored_complete", None)
                    summary["current_aggregate"]["complete_eval"] = False
                else:
                    summary["complete_eval"] = "complete_eval_false" not in name
                if "promotion_false" in name:
                    summary["promotion_eligible"] = False
                if "hard_constraint" in name:
                    summary["hard_constraint_violations"] = ["max_weight"]
                if "legacy_suspect_protocol" in name:
                    summary["suspect_fixed_weight_eval"] = True
                if "validation_only_result" in name:
                    summary["validation_only_result"] = True
                if "durability_scope_validation" in name:
                    summary["durability_scope"] = "validation_signal_only"
                if "late_policy_quarantined" in name:
                    summary["late_result_policy"] = "quarantined_signal"
                (out / "tiered_eval_summary.json").write_text(
                    json.dumps(summary),
                    encoding="utf-8",
                )
            manifest = {"lane_frontiers": {}, "cumulative_top": []}
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(
                max_gems_per_reset=4,
                max_gems_total=4,
                max_gems_per_family=1,
                selection_policy="mature_evidence_top_k",
                min_mature_eval_units=29,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(manifest, completed_gen_id=6)
            names = [entry["variant_name"] for entry in selected]

            self.assertEqual(
                names,
                [
                    "gen5_benchmark_floor_anchor_high",
                    "gen5_benchmark_relative_reward_high",
                    "gen6_hard_constraint_high",
                    "gen1_same_family_b",
                ],
            )
            self.assertNotIn("gen5_capped_high", names)
            self.assertNotIn("gen7_future_leak_high", names)
            self.assertNotIn("gen6_complete_eval_false_high", names)
            self.assertNotIn("gen6_aggregate_complete_eval_false_high", names)
            self.assertNotIn("gen6_promotion_false_high", names)
            self.assertNotIn("gen6_summary_only_high", names)
            self.assertNotIn("gen6_not_scored_complete_high", names)
            self.assertNotIn("control_baseline_high", names)
            self.assertNotIn("clean_name_with_explicit_future_generation", names)
            self.assertNotIn("gen6_scored_complete_false_status", names)
            self.assertNotIn("gen6_not_scored_status", names)
            self.assertNotIn("gen6_control_shuffled_high", names)
            self.assertNotIn("candidategen7future_explicit_name", names)
            self.assertNotIn("gen6_legacy_suspect_protocol_high", names)
            self.assertNotIn("gen6_validation_only_result_high", names)
            self.assertNotIn("gen6_durability_scope_validation_high", names)
            self.assertNotIn("gen6_late_policy_quarantined_high", names)

    def test_result_artifact_gems_reject_late_after_boundary_summary(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            boundary = run_dir / "gen_1" / "generation_boundary.json"
            boundary.parent.mkdir(parents=True)
            boundary.write_text(json.dumps({"generation_id": 1}), encoding="utf-8")
            old = 1_800_000_000
            os.utime(boundary, (old, old))
            result_dir = run_dir / "results" / "gen1_peer0_late_strong"
            result_dir.mkdir(parents=True)
            summary_path = result_dir / "tiered_eval_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "variant_name": "late_strong",
                        "tier_reached": "T1",
                        "completed_tier": "T1",
                        "tier_status": "stop_after_T1",
                        "scout_only": False,
                        "scored_complete": True,
                        "generation_id": 1,
                        "n_eval_cells": 29,
                        "effort_ratio": 0.95,
                        "coverage_ratio": 0.95,
                        "current_aggregate": {
                            "mean_test_taskscore": 120.0,
                            "strategy_family": "learned_alpha",
                        },
                    }
                ),
                encoding="utf-8",
            )
            late = old + 120
            os.utime(summary_path, (late, late))
            task = _task_with_gems(
                max_gems_per_reset=4,
                max_gems_total=4,
                selection_policy="mature_evidence_top_k",
                min_mature_eval_units=29,
            )
            task.evaluation = SimpleNamespace(
                maturity_policy={
                    "min_effort_ratio": 0.75,
                    "min_coverage_ratio": 0.80,
                    "require_ratio_gate": True,
                }
            )
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=task,
                frontier=_FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []}),
            )

            direct = mgr._result_artifact_gem_candidates(max_generation_id=2)
            selected = mgr._select_mature_evidence_topk_entries(
                {"lane_frontiers": {}, "cumulative_top": []},
                max_generation_id=2,
            )

        self.assertEqual(direct, [])
        self.assertEqual(selected, [])

    def test_result_artifact_gems_reject_summary_after_pending_boundary_cutoff(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import resume_state
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen_dir = run_dir / "gen_1"
            gen_dir.mkdir()
            (gen_dir / "generation_results.json").write_text("[]", encoding="utf-8")
            cutoff = datetime.now(UTC)
            resume_state.write_boundary_evidence_checkpoint(
                run_dir,
                gen_id=1,
                cutoff=cutoff,
                evidence_source_snapshot={},
            )
            result_dir = run_dir / "results" / "gen1_peer0_late_strong"
            result_dir.mkdir(parents=True)
            summary_path = result_dir / "tiered_eval_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "variant_name": "late_strong",
                        "tier_reached": "T1",
                        "completed_tier": "T1",
                        "tier_status": "stop_after_T1",
                        "scout_only": False,
                        "scored_complete": True,
                        "generation_id": 1,
                        "n_eval_cells": 29,
                        "effort_ratio": 0.95,
                        "coverage_ratio": 0.95,
                        "current_aggregate": {
                            "mean_test_taskscore": 120.0,
                            "strategy_family": "candidate_family",
                        },
                    }
                ),
                encoding="utf-8",
            )
            task = _task_with_gems(
                selection_policy="mature_evidence_top_k",
                min_mature_eval_units=29,
            )
            task.evaluation = SimpleNamespace(
                maturity_policy={
                    "min_effort_ratio": 0.75,
                    "min_coverage_ratio": 0.80,
                    "require_ratio_gate": True,
                }
            )
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=task,
                frontier=_FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []}),
            )

            direct = mgr._result_artifact_gem_candidates(max_generation_id=1)
            selected = mgr._select_mature_evidence_topk_entries(
                {"lane_frontiers": {}, "cumulative_top": []},
                max_generation_id=1,
            )

        self.assertEqual(direct, [])
        self.assertEqual(selected, [])

    def test_result_artifact_gems_reject_top_level_quarantine_when_aggregate_false(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "gen1_peer0_quarantined_strong"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "quarantined_strong",
                        "tier_reached": "T1",
                        "completed_tier": "T1",
                        "tier_status": "stop_after_T1",
                        "scout_only": False,
                        "scored_complete": True,
                        "generation_id": 1,
                        "n_eval_cells": 29,
                        "effort_ratio": 0.95,
                        "coverage_ratio": 0.95,
                        "late_result_policy": "quarantined_signal",
                        "durability_scope": "validation_signal_only",
                        "current_aggregate": {
                            "mean_test_taskscore": 120.0,
                            "strategy_family": "learned_alpha",
                            "late_result_policy": False,
                            "durability_scope": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            task = _task_with_gems(
                max_gems_per_reset=4,
                max_gems_total=4,
                selection_policy="mature_evidence_top_k",
                min_mature_eval_units=29,
            )
            task.evaluation = SimpleNamespace(
                maturity_policy={
                    "min_effort_ratio": 0.75,
                    "min_coverage_ratio": 0.80,
                    "require_ratio_gate": True,
                }
            )
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=task,
                frontier=_FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []}),
            )

            direct = mgr._result_artifact_gem_candidates(max_generation_id=2)
            selected = mgr._select_mature_evidence_topk_entries(
                {"lane_frontiers": {}, "cumulative_top": []},
                max_generation_id=2,
            )

        self.assertEqual(direct, [])
        self.assertEqual(selected, [])

    def test_result_artifact_gems_accept_task_primary_metric_with_ratio_gate(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            clean_dir = run_dir / "results" / "gen1_peer0_custom_metric_clean"
            clean_dir.mkdir(parents=True)
            (clean_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "custom_metric_clean",
                        "tier_reached": "T1",
                        "completed_tier": "T1",
                        "tier_status": "stop_after_T1",
                        "scout_only": False,
                        "scored_complete": True,
                        "generation_id": 1,
                        "n_eval_cells": 29,
                        "effort_ratio": 0.95,
                        "coverage_ratio": 0.95,
                        "current_aggregate": {
                            "accuracy": 0.92,
                            "strategy_family": "task_defined_family",
                        },
                    }
                ),
                encoding="utf-8",
            )
            quarantined_dir = run_dir / "results" / "gen1_peer1_custom_metric_quarantined"
            quarantined_dir.mkdir(parents=True)
            (quarantined_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "custom_metric_quarantined",
                        "tier_reached": "T1",
                        "completed_tier": "T1",
                        "tier_status": "stop_after_T1",
                        "scout_only": False,
                        "scored_complete": True,
                        "generation_id": 1,
                        "n_eval_cells": 29,
                        "effort_ratio": 0.95,
                        "coverage_ratio": 0.95,
                        "late_result_policy": "quarantined_signal",
                        "current_aggregate": {
                            "accuracy": 0.99,
                            "strategy_family": "task_defined_family",
                        },
                    }
                ),
                encoding="utf-8",
            )
            task = _task_with_gems(
                max_gems_per_reset=4,
                max_gems_total=4,
                selection_policy="mature_evidence_top_k",
                min_mature_eval_units=29,
                primary_metric_keys=[],
            )
            task.evaluation = SimpleNamespace(
                primary_metric="accuracy",
                direction="maximize",
                maturity_policy={
                    "min_effort_ratio": 0.75,
                    "min_coverage_ratio": 0.80,
                    "require_ratio_gate": True,
                },
            )
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=task,
                frontier=_FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []}),
            )

            direct = mgr._result_artifact_gem_candidates(max_generation_id=2)
            selected = mgr._select_mature_evidence_topk_entries(
                {"lane_frontiers": {}, "cumulative_top": []},
                max_generation_id=2,
            )

        direct_names = {entry["variant_name"] for entry in direct}
        selected_names = {entry["variant_name"] for entry in selected}
        self.assertIn("gen1_peer0_custom_metric_clean", direct_names)
        self.assertIn("gen1_peer0_custom_metric_clean", selected_names)
        self.assertNotIn("gen1_peer1_custom_metric_quarantined", direct_names)
        self.assertNotIn("gen1_peer1_custom_metric_quarantined", selected_names)

    def test_mature_evidence_topk_gems_deduplicates_manifest_and_result_aliases(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "gen0_peer5_risk_adjusted_listwise_bc_target_t1"
            result_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "variant_name": "risk_adjusted_listwise_bc_target",
                "tier_reached": "T1",
                "completed_tier": "T1",
                "tier_status": "stop_after_T1",
                "scout_only": False,
                "scored_complete": True,
                "generation_id": 0,
                "n_eval_cells": 29,
                "current_aggregate": {
                    "hist_return_pct": 13.5,
                    "mean_active_alpha_vs_benchmark_pct": 9.9,
                    "max_drawdown_pct": 12.0,
                    "strategy_family": "learned_alpha",
                },
            }
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            manifest = {
                "lane_frontiers": {
                    "alpha_incubator": [
                        {
                            "finding_id": "manifest_alias",
                            "variant_name": "risk_adjusted_listwise_bc_target",
                            "frontier_lane": "alpha_incubator",
                            "promoted_for_lane": "alpha_incubator",
                            "generation_id": 0,
                            "tier": "T1",
                            "evidence_stage": "full_T1",
                            "n_eval_cells": 29,
                            "scored_cell_count": 29,
                            "metric_value": 13.5,
                            "lane_metric_value": 9.9,
                            "metrics": {
                                "hist_return_pct": 13.5,
                                "mean_active_alpha_vs_benchmark_pct": 9.9,
                                "max_drawdown_pct": 12.0,
                                "strategy_family": "learned_alpha",
                                "frontier_lane": "alpha_incubator",
                                "evidence_stage": "full_T1",
                                "tier": "T1",
                                "n_eval_cells": 29,
                                "scored_cell_count": 29,
                            },
                        },
                        {
                            "finding_id": "other_candidate",
                            "variant_name": "bc_curriculum_drop_topk",
                            "frontier_lane": "alpha_incubator",
                            "promoted_for_lane": "alpha_incubator",
                            "generation_id": 0,
                            "tier": "T1",
                            "evidence_stage": "full_T1",
                            "n_eval_cells": 29,
                            "scored_cell_count": 29,
                            "metric_value": 12.0,
                            "lane_metric_value": 8.0,
                            "metrics": {
                                "hist_return_pct": 12.0,
                                "mean_active_alpha_vs_benchmark_pct": 8.0,
                                "max_drawdown_pct": 10.0,
                                "strategy_family": "learned_alpha",
                                "frontier_lane": "alpha_incubator",
                                "evidence_stage": "full_T1",
                                "tier": "T1",
                                "n_eval_cells": 29,
                                "scored_cell_count": 29,
                            },
                        },
                    ]
                },
                "cumulative_top": [],
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(
                max_gems_per_reset=4,
                max_gems_total=4,
                max_gems_per_family=4,
                selection_policy="mature_evidence_top_k",
                min_mature_eval_units=29,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_mature_evidence_topk_entries(manifest, max_generation_id=0)
            names = [entry["variant_name"] for entry in selected]

            self.assertEqual(names.count("risk_adjusted_listwise_bc_target"), 1)
            self.assertIn("bc_curriculum_drop_topk", names)

    def test_mature_evidence_topk_gems_reads_custom_tiered_eval_summary(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "gen2_peer4_custom_alpha"
            result_dir.mkdir(parents=True)
            (result_dir / "custom_gen2_peer4_custom_alpha_tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "schema_version": "sam_optimizer.tiered_eval_summary.v1",
                        "variant_name": "family_name",
                        "final_status": "stopped_at_T1",
                        "tiers": [
                            {
                                "tier": "T1",
                                "returncode": 0,
                                "metrics_summary": {
                                    "tier": "T1",
                                    "mean_test_taskscore": 14.0,
                                    "mean_active_alpha_vs_benchmark_pct": 14.0,
                                    "max_drawdown_pct": 9.0,
                                    "strategy_family": "learned_alpha",
                                    "scored_cell_count": 29,
                                },
                                "gate": {"passed": False, "reason": "T1 diagnostic"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            clean_result_dir = run_dir / "results" / "gen2_peer5_custom_alpha_clean"
            clean_result_dir.mkdir(parents=True)
            (
                clean_result_dir / "custom_gen2_peer5_custom_alpha_clean_tiered_eval_summary.json"
            ).write_text(
                json.dumps(
                    {
                        "schema_version": "sam_optimizer.tiered_eval_summary.v1",
                        "variant_name": "family_name",
                        "final_status": "stopped_at_T1",
                        "scored_complete": True,
                        "tiers": [
                            {
                                "tier": "T1",
                                "returncode": 0,
                                "metrics_summary": {
                                    "tier": "T1",
                                    "mean_test_taskscore": 13.0,
                                    "mean_active_alpha_vs_benchmark_pct": 13.0,
                                    "max_drawdown_pct": 9.0,
                                    "strategy_family": "learned_alpha",
                                    "scored_cell_count": 29,
                                },
                                "gate": {"passed": True, "reason": "T1 clean"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            frontier = _FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []})
            task = _task_with_gems(
                max_gems_per_reset=4,
                max_gems_total=4,
                max_gems_per_family=1,
                selection_policy="mature_evidence_top_k",
                min_mature_eval_units=29,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(
                {"lane_frontiers": {}, "cumulative_top": []}, completed_gen_id=6
            )

            self.assertEqual(
                [entry["variant_name"] for entry in selected],
                ["gen2_peer5_custom_alpha_clean"],
            )
            self.assertEqual(
                selected[0]["metrics"]["source_result_kind"],
                "custom_gen2_peer5_custom_alpha_clean_tiered_eval_summary.json",
            )

    def test_sam_tier_summary_reports_scored_cells_for_gems(self) -> None:
        from templates.tasks.sam_optimizer.evaluations.pareto_tiered.run import (
            summarize_metrics,
        )

        summary = summarize_metrics(
            {
                "tier": "T3",
                "promotion_eligible": True,
                "mean_test_accuracy": {"mean": 0.78},
                "mean_train_test_gap": {"mean": 0.04},
                "sharpness_top_eigen": {"mean": 8.5},
                "per_dataset": {
                    "cifar10": {
                        "status": "ok",
                        "test_accuracy": {"mean": 0.95},
                        "train_test_gap": {"mean": 0.02},
                        "num_seeds_ok": 5,
                        "num_seeds_total": 5,
                    },
                    "cifar100": {
                        "status": "ok",
                        "test_accuracy": {"mean": 0.78},
                        "train_test_gap": {"mean": 0.05},
                        "num_seeds_ok": 5,
                        "num_seeds_total": 5,
                    },
                    "tiny-imagenet": {
                        "status": "ok",
                        "test_accuracy": {"mean": 0.52},
                        "train_test_gap": {"mean": 0.08},
                        "num_seeds_ok": 5,
                        "num_seeds_total": 5,
                    },
                },
            }
        )

        self.assertEqual(summary["tier_reached"], "T3")
        self.assertEqual(summary["evidence_stage"], "T3")
        self.assertEqual(summary["scored_cell_count"], 15)
        self.assertEqual(summary["n_eval_cells"], 15)

    def test_mature_evidence_topk_gems_ignores_scout_named_summary_artifact(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "gen2_peer4_scout"
            result_dir.mkdir(parents=True)
            (result_dir / "scout_tiered_eval_summary.json").write_text(
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
                                    "max_drawdown_pct": 9.0,
                                    "strategy_family": "learned_alpha",
                                    "scored_cell_count": 29,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            frontier = _FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []})
            task = _task_with_gems(
                max_gems_per_reset=4,
                max_gems_total=4,
                max_gems_per_family=1,
                selection_policy="mature_evidence_top_k",
                min_mature_eval_units=29,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(
                {"lane_frontiers": {}, "cumulative_top": []}, completed_gen_id=6
            )

            self.assertEqual(selected, [])

    def test_mature_evidence_topk_gems_ignores_standard_summary_in_scout_dir(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "gen2_peer4_scout"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "scout_alpha",
                        "tier_reached": "T1",
                        "completed_tier": "T1",
                        "tier_status": "stop_after_T1",
                        "current_aggregate": {
                            "mean_test_taskscore": 99.0,
                            "mean_active_alpha_vs_benchmark_pct": 99.0,
                            "max_drawdown_pct": 9.0,
                            "strategy_family": "learned_alpha",
                            "scored_cell_count": 29,
                        },
                    }
                ),
                encoding="utf-8",
            )
            frontier = _FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []})
            task = _task_with_gems(
                max_gems_per_reset=4,
                max_gems_total=4,
                max_gems_per_family=1,
                selection_policy="mature_evidence_top_k",
                min_mature_eval_units=29,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(
                {"lane_frontiers": {}, "cumulative_top": []}, completed_gen_id=6
            )

            self.assertEqual(selected, [])

    def test_mature_evidence_t1_predicate_rejects_scout_result_path(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import gems

        self.assertFalse(
            gems._is_mature_evaluation_or_better(
                {
                    "variant_name": "scout_alpha",
                    "source_result_path": "results/gen2_peer4_scout/tiered_eval_summary.json",
                    "metrics": {
                        "tier": "T1",
                        "scored_cell_count": 29,
                        "mean_test_taskscore": 99.0,
                    },
                },
                min_mature_eval_units=29,
            )
        )

    def test_gems_admission_rescues_strong_performance_candidate_after_mature_artifacts(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "alpha_incubator": [
                        {
                            "finding_id": "weak_t2_a",
                            "variant_name": "gen2_peer4_ua_ppo_t2",
                            "metrics": {
                                "tier": "T2",
                                "strategy_family": "learned_alpha",
                                "mean_test_taskscore": 4.9,
                                "mean_active_alpha_vs_benchmark_pct": 6.0,
                                "max_drawdown_pct": 41.0,
                            },
                        },
                        {
                            "finding_id": "weak_t2_b",
                            "variant_name": "gen1_peer5_ua_ppo",
                            "metrics": {
                                "tier": "T2",
                                "strategy_family": "learned_alpha",
                                "mean_test_taskscore": 4.9,
                                "mean_active_alpha_vs_benchmark_pct": 6.0,
                                "max_drawdown_pct": 41.0,
                            },
                        },
                        {
                            "finding_id": "strong_t1",
                            "variant_name": "gen1_peer1_offpolicy_replay_ppo_bc40",
                            "metrics": {
                                "tier": "T1",
                                "strategy_family": "learned_alpha",
                                "mean_test_taskscore": 13.6,
                                "mean_active_alpha_vs_benchmark_pct": 17.5,
                                "max_drawdown_pct": 9.5,
                            },
                        },
                    ]
                },
                "cumulative_top": [],
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(max_gems_per_reset=4, max_gems_total=4)
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(manifest)
            selected_names = [entry["variant_name"] for entry in selected]

            self.assertIn("gen1_peer1_offpolicy_replay_ppo_bc40", selected_names)
            self.assertLess(
                selected_names.index("gen1_peer1_offpolicy_replay_ppo_bc40"),
                2,
            )

    def test_gems_family_cap_uses_mechanism_specific_family_not_generic_learned_alpha(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            variants = [
                ("gen1_peer1_offpolicy_replay_ppo_bc40", 13.6),
                ("gen1_peer6_awr_bconly", 11.3),
                ("gen2_peer4_ua_ppo_t2", 9.0),
                ("gen0_peer1_dual_critic_ppo_full", 8.0),
            ]
            manifest = {
                "lane_frontiers": {
                    "alpha_incubator": [
                        {
                            "finding_id": name,
                            "variant_name": name,
                            "metrics": {
                                "tier": "T1",
                                "strategy_family": "learned_alpha",
                                "mean_test_taskscore": score,
                                "mean_active_alpha_vs_benchmark_pct": score + 2.0,
                                "max_drawdown_pct": 10.0,
                            },
                        }
                        for name, score in variants
                    ]
                },
                "cumulative_top": [],
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(max_gems_per_reset=4, max_gems_total=4, max_gems_per_family=2)
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(manifest)
            selected_names = [entry["variant_name"] for entry in selected]

            self.assertEqual(len(selected_names), 4)
            self.assertEqual(selected_names[0], "gen1_peer1_offpolicy_replay_ppo_bc40")

    def test_generic_learned_alpha_without_marker_does_not_consume_family_cap(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "alpha_incubator": [
                        {
                            "finding_id": f"f{i}",
                            "variant_name": f"strong_custom_{i}",
                            "generation_id": 0,
                            "metrics": {
                                "tier": "T1",
                                "strategy_family": "learned_alpha",
                                "mean_test_taskscore": 10.0 + i,
                                "mean_active_alpha_vs_benchmark_pct": 12.0 + i,
                                "max_drawdown_pct": 9.0,
                            },
                        }
                        for i in range(4)
                    ]
                },
                "cumulative_top": [],
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(max_gems_per_reset=4, max_gems_total=4, max_gems_per_family=2)
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(manifest)

            self.assertEqual(len(selected), 4)

    def test_gems_rank_lane_metric_value_with_lane_direction(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "performance": [
                        {
                            "finding_id": "low_loss",
                            "variant_name": "low_loss",
                            "source_generation_id": 0,
                            "frontier_lane": "performance",
                            "lane_metric_value": 0.2,
                            "lane_metric_direction": "minimize",
                            "metrics": {
                                "scored_complete": True,
                                "complete_eval": True,
                                "source_generation_id": 0,
                            },
                        },
                        {
                            "finding_id": "high_loss",
                            "variant_name": "high_loss",
                            "source_generation_id": 0,
                            "frontier_lane": "performance",
                            "lane_metric_value": 0.8,
                            "lane_metric_direction": "minimize",
                            "metrics": {
                                "scored_complete": True,
                                "complete_eval": True,
                                "source_generation_id": 0,
                            },
                        },
                    ]
                },
                "cumulative_top": [],
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(
                max_gems_per_reset=1,
                max_gems_total=1,
                include_lanes=["performance"],
                performance_lanes=["performance"],
            )
            task.evaluation = SimpleNamespace(direction="maximize", frontier_lanes=[])
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(manifest)

        self.assertEqual([entry["variant_name"] for entry in selected], ["low_loss"])

    def test_entry_family_marker_precedence_and_generic_fallbacks(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import _entry_family

        cases = [
            (
                {"variant_name": "gen1_peer1_offpolicy_replay_ppo_bc40"},
                "",
            ),
            ({"variant_name": "gen2_peer3_ua_ppo_bc40"}, ""),
            ({"variant_name": "gen1_peer6_awr_bconly"}, ""),
            (
                {
                    "variant_name": "gen3_peer0_custom",
                    "metrics": {"strategy_family": "learned_alpha"},
                },
                "learned_alpha",
            ),
            (
                {
                    "variant_name": "gen1_peer1_offpolicy_replay_ppo_bc40",
                    "metrics": {"mechanism_family": "explicit_override"},
                },
                "explicit_override",
            ),
        ]
        for entry, expected in cases:
            self.assertEqual(_entry_family(entry), expected, entry)

    def test_result_artifact_gem_candidates_use_task_configured_derivations(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_dir = run_dir / "results" / "derived_gen3"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "reported_family",
                        "generation_id": 3,
                        "complete_eval": True,
                        "tiers": [
                            {
                                "tier": "T1",
                                "status": "ok",
                                "metrics_summary": {"scored_cell_count": 2},
                            }
                        ],
                        "all_eval_cells": [
                            {"task_return": 1.0},
                            {"task_return": 3.0},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            task = _task_with_gems(
                selection_policy="mature_evidence_top_k",
                min_mature_eval_units=2,
                evidence_stage_min_units={"T1": 2},
                performance_lanes=["performance"],
                result_artifact_default_lane="performance",
                result_artifact_default_family="learned_candidate",
                primary_metric_keys=["derived_score"],
                result_cell_metric_derivations=[
                    {
                        "name": "derived_score",
                        "source_keys": ["task_return"],
                        "aggregate": "mean",
                    }
                ],
            )
            task.evaluation = SimpleNamespace(direction="maximize", frontier_lanes=[])
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=SimpleNamespace())

            candidates = mgr._result_artifact_gem_candidates(max_generation_id=3)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["variant_name"], "derived_gen3")
        self.assertEqual(candidates[0]["metrics"]["derived_score"], 2.0)
        self.assertTrue(candidates[0]["metrics"]["scored_complete"])

    def test_stronger_new_candidate_can_replace_full_existing_gems(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "alpha_incubator": [
                        {
                            "finding_id": "new_strong",
                            "variant_name": "new_strong_offpolicy_replay",
                            "frontier_lane": "alpha_incubator",
                            "generation_id": 0,
                            "metrics": {
                                "tier": "T1",
                                "complete_eval": True,
                                "scored_cell_count": 29,
                                "strategy_family": "learned_alpha",
                                "mean_test_taskscore": 20.0,
                                "mean_active_alpha_vs_benchmark_pct": 22.0,
                                "max_drawdown_pct": 8.0,
                            },
                        }
                    ]
                },
                "cumulative_top": [],
            }
            existing = [
                {
                    "gem_finding_id": f"old_{i}",
                    "variant_name": f"old_weak_{i}",
                    "source_generation_id": 0,
                    "frontier_lane": "alpha_incubator",
                    "evidence_stage": "full_T1",
                    "admission_metrics": {
                        "mean_test_taskscore": float(i),
                        "mean_active_alpha_vs_benchmark_pct": float(i),
                        "max_drawdown_pct": 20.0,
                    },
                }
                for i in range(4)
            ]
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(max_gems_per_reset=4, max_gems_total=4)
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(manifest, existing_gems=existing)
            combined = mgr._compact_gems([*existing, *selected], sort_by_performance=True)
            names = [entry["variant_name"] for entry in combined]

            self.assertIn("new_strong_offpolicy_replay", names)
            self.assertEqual(names[0], "new_strong_offpolicy_replay")
            self.assertEqual(len(names), 4)

    def test_lane_metric_value_participates_in_gem_performance_replacement(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "alpha_incubator": [
                        {
                            "finding_id": "lane_only_strong",
                            "variant_name": "lane_only_strong_offpolicy_replay",
                            "generation_id": 0,
                            "tier": "T1",
                            "strategy_family": "learned_alpha",
                            "lane_metric_name": "mean_active_alpha_vs_benchmark_pct",
                            "lane_metric_value": 99.0,
                            "metrics": {"max_drawdown_pct": 8.0},
                        }
                    ]
                },
                "cumulative_top": [],
            }
            existing = [
                {
                    "gem_finding_id": f"old_{i}",
                    "variant_name": f"old_weak_{i}",
                    "source_generation_id": 0,
                    "frontier_lane": "alpha_incubator",
                    "evidence_stage": "full_T1",
                    "admission_metrics": {
                        "mean_test_taskscore": 0.0,
                        "mean_active_alpha_vs_benchmark_pct": float(i),
                        "max_drawdown_pct": 20.0,
                    },
                }
                for i in range(4)
            ]
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(max_gems_per_reset=4, max_gems_total=4)
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(manifest, existing_gems=existing)
            combined = mgr._compact_gems([*existing, *selected], sort_by_performance=True)
            names = [entry["variant_name"] for entry in combined]

            self.assertIn("lane_only_strong_offpolicy_replay", names)
            self.assertEqual(names[0], "lane_only_strong_offpolicy_replay")

    def test_existing_gem_admission_evidence_rank_round_trips_for_tiebreak(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier = _FakeFrontier(run_dir, {"cumulative_top": []})
            task = _task_with_gems(max_gems_per_reset=4, max_gems_total=4)
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            existing = {
                "gem_finding_id": "existing_t3",
                "variant_name": "existing_t3_alpha",
                "frontier_lane": "alpha_incubator",
                "source_generation_id": 0,
                "evidence_stage": "T3",
                "admission_metrics": {
                    "mean_test_taskscore": 10.0,
                    "mean_active_alpha_vs_benchmark_pct": 10.0,
                    "q25_active_alpha_vs_benchmark_pct": 1.0,
                    "validation_2026_active_alpha_pct": 2.0,
                    "max_drawdown_pct": 12.0,
                    "evidence_rank": 4,
                },
            }
            new = {
                "finding_id": "new_t1",
                "variant_name": "new_t1_alpha",
                "frontier_lane": "alpha_incubator",
                "generation_id": 0,
                "metrics": {
                    "tier": "T1",
                    "mean_test_taskscore": 10.0,
                    "mean_active_alpha_vs_benchmark_pct": 10.0,
                    "q25_active_alpha_vs_benchmark_pct": 1.0,
                    "validation_2026_active_alpha_pct": 2.0,
                    "max_drawdown_pct": 12.0,
                },
            }

            compact = mgr._compact_gems([new, existing], sort_by_performance=True)

            self.assertEqual(compact[0]["variant_name"], "existing_t3_alpha")

    def test_final_gem_compaction_preserves_reserved_non_performance_lanes(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            selected = [
                {
                    "finding_id": "bench",
                    "variant_name": "benchmark_floor_anchor",
                    "frontier_lane": "benchmark_floor",
                    "promoted_for_lane": "benchmark_floor",
                    "generation_id": 0,
                    "metrics": {
                        "mean_test_taskscore": 0.0,
                        "mean_active_alpha_vs_benchmark_pct": 0.0,
                        "max_drawdown_pct": 1.0,
                    },
                },
                {
                    "finding_id": "diag",
                    "variant_name": "diagnostic_control_anchor",
                    "frontier_lane": "diagnostic_control",
                    "promoted_for_lane": "diagnostic_control",
                    "generation_id": 0,
                    "metrics": {
                        "mean_test_taskscore": -1.0,
                        "mean_active_alpha_vs_benchmark_pct": -1.0,
                        "max_drawdown_pct": 1.0,
                    },
                },
            ]
            existing = [
                {
                    "gem_finding_id": f"old_alpha_{i}",
                    "variant_name": f"old_high_alpha_{i}",
                    "frontier_lane": "alpha_incubator",
                    "source_generation_id": 0,
                    "admission_metrics": {
                        "mean_test_taskscore": 20.0 - i,
                        "mean_active_alpha_vs_benchmark_pct": 20.0 - i,
                        "max_drawdown_pct": 8.0,
                    },
                }
                for i in range(4)
            ]
            manifest = {
                "lane_frontiers": {
                    "benchmark_floor": selected[:1],
                    "diagnostic_control": selected[1:],
                }
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(max_gems_per_reset=4, max_gems_total=4)
            state = {
                "enabled": True,
                "cycle_index": 1,
                "cycle_start_generation": 6,
                "reset_count": 1,
                "gems": existing,
                "pending_reset": {
                    "status": "pending",
                    "reset_count": 2,
                    "cycle_index": 2,
                    "completed_gen_id": 11,
                    "next_absolute_generation": 12,
                    "signature_hash": "sig",
                    "reason": "test_lane_reserve",
                    "archive_dir": str(run_dir / "archive" / "gems_cycle_2_lane_reserve"),
                    "selected_entries": selected,
                },
            }
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            mgr.save_state(state)

            result = mgr.recover_pending_reset(completed_gen_id=11)

            self.assertTrue(result.triggered)
            state_after = json.loads((run_dir / "gems" / "gems_state.json").read_text())
            lanes = {gem.get("frontier_lane") for gem in state_after["gems"]}
            self.assertIn("benchmark_floor", lanes)
            self.assertIn("diagnostic_control", lanes)

    def test_performance_lanes_compete_by_performance_before_family_cap(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier = _FakeFrontier(run_dir, {"cumulative_top": []})
            task = _task_with_gems(max_gems_per_reset=4, max_gems_total=4, max_gems_per_family=2)
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            gems = [
                {
                    "gem_finding_id": "weak_confirmed",
                    "variant_name": "weak_confirmed_offpolicy_replay",
                    "frontier_lane": "confirmed_alpha",
                    "source_generation_id": 0,
                    "evidence_stage": "full_T1",
                    "admission_metrics": {
                        "mean_test_taskscore": 1.0,
                        "mean_active_alpha_vs_benchmark_pct": 1.0,
                    },
                },
                {
                    "gem_finding_id": "weak_alpha",
                    "variant_name": "weak_alpha_offpolicy_replay",
                    "frontier_lane": "alpha",
                    "source_generation_id": 0,
                    "evidence_stage": "full_T1",
                    "admission_metrics": {
                        "mean_test_taskscore": 2.0,
                        "mean_active_alpha_vs_benchmark_pct": 2.0,
                    },
                },
                {
                    "gem_finding_id": "strong_incubator",
                    "variant_name": "strong_incubator_offpolicy_replay",
                    "frontier_lane": "alpha_incubator",
                    "source_generation_id": 0,
                    "evidence_stage": "full_T1",
                    "admission_metrics": {
                        "mean_test_taskscore": 20.0,
                        "mean_active_alpha_vs_benchmark_pct": 20.0,
                    },
                },
                {
                    "gem_finding_id": "bench",
                    "variant_name": "benchmark_floor_anchor",
                    "frontier_lane": "benchmark_floor",
                    "source_generation_id": 0,
                    "evidence_stage": "full_T1",
                    "admission_metrics": {"mean_test_taskscore": 0.0},
                },
                {
                    "gem_finding_id": "diag",
                    "variant_name": "diagnostic_control_anchor",
                    "frontier_lane": "diagnostic_control",
                    "source_generation_id": 0,
                    "evidence_stage": "full_T1",
                    "admission_metrics": {"mean_test_taskscore": -1.0},
                },
            ]

            compact = mgr._compact_gems(
                gems,
                sort_by_performance=True,
                preserve_lane_reserves=True,
            )
            names = [gem["variant_name"] for gem in compact]

            self.assertIn("strong_incubator_offpolicy_replay", names)
            self.assertIn("benchmark_floor_anchor", names)
            self.assertIn("diagnostic_control_anchor", names)
            self.assertEqual(len(names), 4)

    def test_aist_like_gems_selection_preserves_multiple_strong_incubator_candidates(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            strong_incubator = [
                {
                    "finding_id": f"inc_{i}",
                    "variant_name": f"strong_incubator_{i}_offpolicy_replay",
                    "generation_id": 0,
                    "frontier_lane": "alpha_incubator",
                    "metrics": {
                        "strategy_family": "learned_alpha",
                        "mean_test_taskscore": 30.0 - i,
                        "mean_active_alpha_vs_benchmark_pct": 30.0 - i,
                        "complete_eval": True,
                    },
                }
                for i in range(4)
            ]
            manifest = {
                "lane_frontiers": {
                    "confirmed_alpha": [
                        {
                            "finding_id": "weak_confirmed",
                            "variant_name": "weak_confirmed_offpolicy_replay",
                            "generation_id": 0,
                            "metrics": {
                                "strategy_family": "learned_alpha",
                                "mean_test_taskscore": 1.0,
                            },
                        }
                    ],
                    "alpha_incubator": strong_incubator,
                    "benchmark_floor": [
                        {
                            "finding_id": "bench",
                            "variant_name": "benchmark_floor_anchor",
                            "generation_id": 0,
                            "metrics": {"mean_test_taskscore": 0.0, "complete_eval": True},
                        }
                    ],
                    "diagnostic_control": [
                        {
                            "finding_id": "diag",
                            "variant_name": "diagnostic_control_anchor",
                            "generation_id": 0,
                            "metrics": {"mean_test_taskscore": -1.0, "complete_eval": True},
                        }
                    ],
                },
                "cumulative_top": [],
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(max_gems_per_reset=4, max_gems_total=4, max_gems_per_family=2)
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(manifest)
            names = [entry["variant_name"] for entry in selected]
            incubator_count = sum(name.startswith("strong_incubator_") for name in names)

            self.assertGreaterEqual(incubator_count, 2)
            self.assertNotIn("weak_confirmed_offpolicy_replay", names)

    def test_final_gem_compaction_keeps_strong_alpha_when_controls_share_family(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import (
            GemsManager,
            _is_performance_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier = _FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []})
            task = _task_with_gems(max_gems_per_reset=4, max_gems_total=4, max_gems_per_family=2)
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            gems = [
                {
                    "gem_finding_id": "alpha_high_1",
                    "variant_name": "alpha_high_1",
                    "frontier_lane": "alpha_incubator",
                    "mechanism_family": "offpolicy_replay",
                    "source_generation_id": 0,
                    "evidence_stage": "full_T1",
                    "admission_metrics": {
                        "mean_test_taskscore": 20.0,
                        "mean_active_alpha_vs_benchmark_pct": 25.0,
                    },
                },
                {
                    "gem_finding_id": "alpha_high_2",
                    "variant_name": "alpha_high_2",
                    "frontier_lane": "alpha_incubator",
                    "mechanism_family": "offpolicy_replay",
                    "source_generation_id": 0,
                    "evidence_stage": "full_T1",
                    "admission_metrics": {
                        "mean_test_taskscore": 19.0,
                        "mean_active_alpha_vs_benchmark_pct": 24.0,
                    },
                },
                {
                    "gem_finding_id": "bench_low",
                    "variant_name": "bench_low",
                    "frontier_lane": "benchmark_floor",
                    "mechanism_family": "offpolicy_replay",
                    "source_generation_id": 0,
                    "evidence_stage": "full_T1",
                    "admission_metrics": {
                        "strategy_family": "learned_alpha",
                        "mean_test_taskscore": -5.0,
                    },
                },
                {
                    "gem_finding_id": "diag_low",
                    "variant_name": "diag_low",
                    "frontier_lane": "diagnostic_control",
                    "mechanism_family": "offpolicy_replay",
                    "source_generation_id": 0,
                    "evidence_stage": "full_T1",
                    "admission_metrics": {
                        "strategy_family": "learned_alpha",
                        "mean_test_taskscore": -6.0,
                    },
                },
            ]

            compact = mgr._compact_gems(
                gems,
                sort_by_performance=True,
                preserve_lane_reserves=True,
            )

            ids = [gem["gem_finding_id"] for gem in compact]
            self.assertIn("alpha_high_1", ids)
            self.assertIn("alpha_high_2", ids)
            self.assertLess(ids.index("alpha_high_1"), ids.index("bench_low"))
            self.assertLess(ids.index("alpha_high_2"), ids.index("diag_low"))
            self.assertFalse(
                _is_performance_entry(
                    {
                        "variant_name": "benchmark_template_control",
                        "metrics": {
                            "strategy_family": "learned_alpha",
                            "mean_test_taskscore": -1.0,
                        },
                    }
                )
            )
            self.assertFalse(
                _is_performance_entry(
                    {
                        "variant_name": "target_allocation_reference",
                        "metrics": {
                            "strategy_family": "learned_alpha",
                            "mean_test_taskscore": 15.0,
                        },
                    }
                )
            )
            legacy_strong_alpha = {
                "variant_name": "legacy_strong_alpha",
                "frontier_lane": "alpha_incubator",
                "metrics": {
                    "strategy_family": "learned_alpha",
                    "mean_test_taskscore": 10.0,
                },
            }
            self.assertFalse(_is_performance_entry(legacy_strong_alpha))
            self.assertTrue(mgr._is_task_performance_entry(legacy_strong_alpha))

    def test_cumulative_top_fallback_preserves_lane_anchors_when_no_lane_frontiers(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            alpha_entries = [
                {
                    "finding_id": f"alpha_{i}",
                    "variant_name": f"alpha_variant_{i}",
                    "generation_id": 0,
                    "frontier_lane": "alpha_incubator",
                    "metrics": {
                        "mean_test_taskscore": 20.0 - i,
                        "mean_active_alpha_vs_benchmark_pct": 20.0 - i,
                        "complete_eval": True,
                    },
                }
                for i in range(4)
            ]
            anchors = [
                {
                    "finding_id": "bench",
                    "variant_name": "benchmark_floor_anchor",
                    "generation_id": 0,
                    "frontier_lane": "benchmark_floor",
                    "metrics": {"mean_test_taskscore": 0.0, "complete_eval": True},
                },
                {
                    "finding_id": "diag",
                    "variant_name": "diagnostic_control_anchor",
                    "generation_id": 0,
                    "frontier_lane": "diagnostic_control",
                    "metrics": {"mean_test_taskscore": -1.0, "complete_eval": True},
                },
            ]
            manifest = {"cumulative_top": [*alpha_entries, *anchors]}
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(max_gems_per_reset=4, max_gems_total=4)
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(manifest)
            lanes = {entry.get("frontier_lane") for entry in selected}

            self.assertIn("benchmark_floor", lanes)
            self.assertIn("diagnostic_control", lanes)

    def test_superseded_gem_findings_are_pruned_before_resume_sync(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store
        from praxist.plugins.workflow_stages.research_loop.backend.tools.findings_sync import (
            FindingsSync,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            shared.mkdir(parents=True)
            manifest = {
                "lane_frontiers": {
                    "alpha_incubator": [
                        {
                            "finding_id": "new_strong",
                            "variant_name": "new_strong_offpolicy_replay",
                            "frontier_lane": "alpha_incubator",
                            "generation_id": 0,
                            "metrics": {
                                "tier": "T1",
                                "complete_eval": True,
                                "scored_cell_count": 29,
                                "strategy_family": "learned_alpha",
                                "mean_test_taskscore": 20.0,
                                "mean_active_alpha_vs_benchmark_pct": 22.0,
                                "max_drawdown_pct": 8.0,
                            },
                        }
                    ]
                },
                "cumulative_top": [],
            }
            old_gems = [
                {
                    "gem_finding_id": f"old_{i}",
                    "variant_name": f"old_weak_{i}",
                    "frontier_lane": "alpha_incubator",
                    "source_generation_id": 0,
                    "admission_metrics": {
                        "mean_test_taskscore": float(i),
                        "mean_active_alpha_vs_benchmark_pct": float(i),
                        "max_drawdown_pct": 20.0,
                    },
                }
                for i in range(4)
            ]
            state = {
                "enabled": True,
                "cycle_index": 1,
                "cycle_start_generation": 6,
                "reset_count": 1,
                "gems": old_gems,
                "pending_reset": {
                    "status": "pending",
                    "reset_count": 2,
                    "cycle_index": 2,
                    "completed_gen_id": 11,
                    "next_absolute_generation": 12,
                    "signature_hash": "sig",
                    "reason": "test_replacement",
                    "archive_dir": str(run_dir / "archive" / "gems_cycle_2_test"),
                    "selected_entries": manifest["lane_frontiers"]["alpha_incubator"],
                },
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(max_gems_per_reset=4, max_gems_total=4)

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}, clear=False):
                local_store.init_db()
                for gem in old_gems:
                    finding = {
                        "id": gem["gem_finding_id"],
                        "finding_type": "result",
                        "title": f"old {gem['variant_name']}",
                        "content": "old gem",
                        "metrics": {
                            "is_gem_finding": True,
                            **gem["admission_metrics"],
                        },
                        "variant_name": gem["variant_name"],
                        "peer_id": "gems_agent",
                        "generation_id": 0,
                    }
                    local_store.insert_finding(finding)
                    (shared / f"{finding['id']}_{finding['variant_name']}.json").write_text(
                        json.dumps(finding),
                        encoding="utf-8",
                    )

                mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
                mgr.save_state(state)
                result = mgr.recover_pending_reset(completed_gen_id=11)

                self.assertTrue(result.triggered)
                state_after = json.loads((run_dir / "gems" / "gems_state.json").read_text())
                final_ids = {gem["gem_finding_id"] for gem in state_after["gems"]}
                self.assertIn("gem_r02_01_new_strong_offpolicy_replay", final_ids)
                self.assertNotIn("old_0", final_ids)
                active_ids = {
                    json.loads(path.read_text(encoding="utf-8"))["id"]
                    for path in shared.glob("*.json")
                }
                self.assertEqual(active_ids, final_ids)

                FindingsSync(shared, local_mode=True).sync_once()
                db_ids = {finding["id"] for finding in local_store.get_all_findings()}
                self.assertEqual(db_ids, final_ids)

    def test_superseded_gem_prune_uses_known_ids_even_without_gem_marker(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store
        from praxist.plugins.workflow_stages.research_loop.backend.tools.findings_sync import (
            FindingsSync,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            shared.mkdir(parents=True)
            selected = [
                {
                    "finding_id": f"new_{i}",
                    "variant_name": f"new_strong_{i}",
                    "frontier_lane": "alpha_incubator",
                    "generation_id": 0,
                    "metrics": {
                        "mechanism_family": f"family_{i}",
                        "tier": "T1",
                        "complete_eval": True,
                        "scored_cell_count": 29,
                        "strategy_family": "learned_alpha",
                        "mean_test_taskscore": 30.0 - i,
                        "mean_active_alpha_vs_benchmark_pct": 35.0 - i,
                    },
                }
                for i in range(4)
            ]
            manifest = {"lane_frontiers": {"alpha_incubator": selected}, "cumulative_top": []}
            old_gem_id = "gem_r01_01_old_alpha"
            state = {
                "enabled": True,
                "cycle_index": 1,
                "cycle_start_generation": 6,
                "reset_count": 1,
                "gems": [
                    {
                        "gem_finding_id": old_gem_id,
                        "variant_name": "old_alpha",
                        "frontier_lane": "alpha_incubator",
                        "admission_metrics": {"mean_test_taskscore": -10.0},
                    }
                ],
                "pending_reset": {
                    "status": "pending",
                    "reset_count": 2,
                    "cycle_index": 2,
                    "completed_gen_id": 11,
                    "next_absolute_generation": 12,
                    "signature_hash": "sig",
                    "reason": "test_missing_marker_prune",
                    "archive_dir": str(run_dir / "archive" / "gems_cycle_2_marker"),
                    "selected_entries": selected,
                },
            }
            old_finding = {
                "id": old_gem_id,
                "finding_type": "result",
                "title": "old alpha",
                "content": "old gem without marker",
                "metrics": {"mean_test_taskscore": -10.0},
                "variant_name": "old_alpha",
                "peer_id": "gems_agent",
                "generation_id": 0,
            }
            (shared / "old_alpha.json").write_text(json.dumps(old_finding), encoding="utf-8")
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(
                max_gems_per_reset=4,
                max_gems_total=4,
                max_gems_per_family=4,
                archive_ordinary_findings=False,
            )

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}, clear=False):
                local_store.init_db()
                local_store.insert_finding(old_finding)
                mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
                mgr.save_state(state)
                result = mgr.recover_pending_reset(completed_gen_id=11)

                self.assertTrue(result.triggered)
                active_ids = {
                    json.loads(path.read_text(encoding="utf-8"))["id"]
                    for path in shared.glob("*.json")
                }
                self.assertNotIn(old_gem_id, active_ids)
                FindingsSync(shared, local_mode=True).sync_once()
                db_ids = {finding["id"] for finding in local_store.get_all_findings()}
                self.assertNotIn(old_gem_id, db_ids)

    def test_forged_gem_sidecar_does_not_replace_existing_gem_on_ingest(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store
        from praxist.plugins.workflow_stages.research_loop.backend.tools.findings_sync import (
            FindingsSync,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            shared.mkdir(parents=True)
            gem_id = "gem_r01_01_real_alpha"
            (run_dir / "gems").mkdir()
            (run_dir / "gems" / "gems_state.json").write_text(
                json.dumps(
                    {
                        "gems": [{"gem_finding_id": gem_id, "variant_name": "real_alpha"}],
                        "reset_count": 1,
                        "cycle_start_generation": 6,
                    }
                ),
                encoding="utf-8",
            )
            real = {
                "id": gem_id,
                "finding_type": "result",
                "title": "real gem",
                "content": "real content",
                "metrics": {"is_gem_finding": True, "mean_test_taskscore": 10.0},
                "variant_name": "real_alpha",
                "peer_id": "gems_agent",
                "generation_id": 0,
            }
            fake = {
                "id": gem_id,
                "finding_type": "result",
                "title": "fake gem",
                "content": "fake content",
                "metrics": {"is_gem_finding": True, "mean_test_taskscore": 999.0},
                "variant_name": "fake_alpha",
                "peer_id": "gen6_peer0",
                "generation_id": 6,
            }

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}, clear=False):
                local_store.init_db()
                local_store.insert_finding(real)
                (shared / "forged_gem.json").write_text(json.dumps(fake), encoding="utf-8")

                FindingsSync(shared, local_mode=True).sync_once()
                findings = local_store.get_all_findings()

                self.assertEqual(len(findings), 1)
                self.assertEqual(findings[0]["id"], gem_id)
                self.assertEqual(findings[0]["variant_name"], "real_alpha")
                self.assertEqual(findings[0]["content"], "real content")

    def test_noncanonical_gem_sidecar_is_not_trusted_or_kept_active(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager
        from praxist.plugins.workflow_stages.research_loop.backend.tools.findings_ingest import (
            parse_finding_file,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            shared.mkdir(parents=True)
            gem_id = "gem_r01_01_real_alpha"
            (run_dir / "gems").mkdir()
            (run_dir / "gems" / "gems_state.json").write_text(
                json.dumps(
                    {
                        "gems": [
                            {
                                "gem_finding_id": gem_id,
                                "variant_name": "real_alpha",
                                "finding_path": "shared_findings/canonical_real_alpha.json",
                            }
                        ],
                        "reset_count": 1,
                        "cycle_start_generation": 6,
                    }
                ),
                encoding="utf-8",
            )
            canonical = {
                "id": gem_id,
                "finding_type": "result",
                "title": "real gem",
                "content": "real content",
                "metrics": {"is_gem_finding": True, "mean_test_taskscore": 10.0},
                "variant_name": "real_alpha",
                "peer_id": "gems_agent",
                "generation_id": 0,
            }
            fake = {
                "id": gem_id,
                "finding_type": "result",
                "title": "fake gem",
                "content": "fake content",
                "metrics": {"is_gem_finding": True, "mean_test_taskscore": 999.0},
                "variant_name": "fake_alpha",
                "peer_id": "gen6_peer0",
                "generation_id": 6,
            }
            canonical_path = shared / "canonical_real_alpha.json"
            fake_path = shared / "forged_gem.json"
            canonical_path.write_text(json.dumps(canonical), encoding="utf-8")
            fake_path.write_text(json.dumps(fake), encoding="utf-8")

            fake_row = parse_finding_file(fake_path)
            self.assertIsNotNone(fake_row)
            self.assertNotIn("is_gem_finding", fake_row["metrics"])
            alias_path = shared / "alias_to_canonical.json"
            alias_path.symlink_to(canonical_path)
            alias_row = parse_finding_file(alias_path)
            self.assertIsNotNone(alias_row)
            self.assertNotIn("is_gem_finding", alias_row["metrics"])

            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=_task_with_gems(),
                frontier=_FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []}),
            )
            archive_dir = run_dir / "archive" / "fake_sidecar"
            mgr._archive_shared_findings(
                archive_dir=archive_dir,
                keep_finding_ids={gem_id},
                keep_finding_paths_by_id={gem_id: {"shared_findings/canonical_real_alpha.json"}},
            )

            self.assertTrue(canonical_path.exists())
            self.assertFalse(fake_path.exists())
            self.assertTrue((archive_dir / "shared_findings" / "forged_gem.json").exists())

    def test_ambiguous_legacy_gem_without_finding_path_archives_all_same_id_sidecars(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            shared.mkdir(parents=True)
            old_gem_id = "gem_r01_01_old_alpha"
            sidecar_a = {
                "id": old_gem_id,
                "finding_type": "result",
                "title": "GEM 1.1: old_alpha",
                "content": "legacy gem a",
                "metrics": {"is_gem_finding": True, "mean_test_taskscore": 10.0},
                "variant_name": "old_alpha",
                "peer_id": "gems_agent",
                "generation_id": 0,
            }
            sidecar_b = dict(sidecar_a)
            sidecar_b["content"] = "legacy gem b with same trust score"
            path_a = shared / "old_alpha_a.json"
            path_b = shared / "old_alpha_b.json"
            path_a.write_text(json.dumps(sidecar_a), encoding="utf-8")
            path_b.write_text(json.dumps(sidecar_b), encoding="utf-8")
            state = {
                "enabled": True,
                "cycle_index": 1,
                "cycle_start_generation": 6,
                "reset_count": 1,
                "gems": [
                    {
                        "gem_finding_id": old_gem_id,
                        "variant_name": "old_alpha",
                        "frontier_lane": "alpha_incubator",
                        "admission_metrics": {"mean_test_taskscore": 10.0},
                    }
                ],
                "pending_reset": {
                    "status": "pending",
                    "reset_count": 2,
                    "cycle_index": 2,
                    "completed_gen_id": 11,
                    "next_absolute_generation": 12,
                    "signature_hash": "sig",
                    "reason": "legacy_ambiguous_path",
                    "archive_dir": str(run_dir / "archive" / "gems_cycle_2_ambiguous"),
                    "selected_entries": [],
                },
            }
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=_task_with_gems(),
                frontier=_FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []}),
            )
            mgr.save_state(state)
            result = mgr.recover_pending_reset(completed_gen_id=11)

            self.assertTrue(result.triggered)
            self.assertFalse(path_a.exists())
            self.assertFalse(path_b.exists())
            archived = run_dir / "archive" / "gems_cycle_2_ambiguous" / "shared_findings"
            self.assertTrue((archived / "old_alpha_a.json").exists())
            self.assertTrue((archived / "old_alpha_b.json").exists())

    def test_legacy_gem_without_finding_path_keeps_best_sidecar_and_archives_duplicate(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            shared.mkdir(parents=True)
            old_gem_id = "gem_r01_01_old_alpha"
            true_sidecar = {
                "id": old_gem_id,
                "finding_type": "result",
                "title": "GEM 1.1: old_alpha",
                "content": "legacy true gem",
                "metrics": {"is_gem_finding": True, "mean_test_taskscore": 10.0},
                "variant_name": "old_alpha",
                "peer_id": "gems_agent",
                "generation_id": 0,
            }
            fake_sidecar = {
                "id": old_gem_id,
                "finding_type": "result",
                "title": "fake duplicate",
                "content": "legacy fake",
                "metrics": {"is_gem_finding": True, "mean_test_taskscore": 999.0},
                "variant_name": "fake_alpha",
                "peer_id": "gen6_peer0",
                "generation_id": 6,
            }
            true_path = shared / "old_alpha_true.json"
            fake_path = shared / "old_alpha_fake.json"
            true_path.write_text(json.dumps(true_sidecar), encoding="utf-8")
            fake_path.write_text(json.dumps(fake_sidecar), encoding="utf-8")
            manifest = {"lane_frontiers": {}, "cumulative_top": []}
            state = {
                "enabled": True,
                "cycle_index": 1,
                "cycle_start_generation": 6,
                "reset_count": 1,
                "gems": [
                    {
                        "gem_finding_id": old_gem_id,
                        "variant_name": "old_alpha",
                        "frontier_lane": "alpha_incubator",
                        "admission_metrics": {"mean_test_taskscore": 10.0},
                    }
                ],
                "pending_reset": {
                    "status": "pending",
                    "reset_count": 2,
                    "cycle_index": 2,
                    "completed_gen_id": 11,
                    "next_absolute_generation": 12,
                    "signature_hash": "sig",
                    "reason": "legacy_path_backfill",
                    "archive_dir": str(run_dir / "archive" / "gems_cycle_2_legacy"),
                    "selected_entries": [],
                },
            }
            frontier = _FakeFrontier(run_dir, manifest)
            mgr = GemsManager(run_dir=run_dir, task_spec=_task_with_gems(), frontier=frontier)
            mgr.save_state(state)
            result = mgr.recover_pending_reset(completed_gen_id=11)

            self.assertTrue(result.triggered)
            self.assertTrue(true_path.exists())
            self.assertFalse(fake_path.exists())
            self.assertTrue(
                (
                    run_dir
                    / "archive"
                    / "gems_cycle_2_legacy"
                    / "shared_findings"
                    / "old_alpha_fake.json"
                ).exists()
            )

    def test_superseded_prune_ignores_unknown_ordinary_is_gem_markers(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            shared.mkdir(parents=True)
            ordinary = {
                "id": "ordinary_fake_gem_marker",
                "finding_type": "result",
                "title": "ordinary",
                "content": "ordinary content",
                "metrics": {"is_gem_finding": True, "mean_test_taskscore": -1.0},
                "variant_name": "ordinary",
                "peer_id": "gen5_peer0",
                "generation_id": 5,
            }
            (shared / "ordinary_fake_gem_marker.json").write_text(
                json.dumps(ordinary),
                encoding="utf-8",
            )
            manifest = {
                "lane_frontiers": {
                    "alpha_incubator": [
                        {
                            "finding_id": "new_strong",
                            "variant_name": "new_strong_offpolicy_replay",
                            "generation_id": 0,
                            "metrics": {
                                "mean_test_taskscore": 20.0,
                                "mean_active_alpha_vs_benchmark_pct": 22.0,
                                "complete_eval": True,
                            },
                        }
                    ]
                },
                "cumulative_top": [],
            }
            state = {
                "enabled": True,
                "cycle_index": 1,
                "cycle_start_generation": 6,
                "reset_count": 1,
                "gems": [],
                "pending_reset": {
                    "status": "pending",
                    "reset_count": 2,
                    "cycle_index": 2,
                    "completed_gen_id": 11,
                    "next_absolute_generation": 12,
                    "signature_hash": "sig",
                    "reason": "test_prune_scope",
                    "archive_dir": str(run_dir / "archive" / "gems_cycle_2_prune_scope"),
                    "selected_entries": manifest["lane_frontiers"]["alpha_incubator"],
                },
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(
                max_gems_per_reset=4,
                max_gems_total=4,
                archive_ordinary_findings=False,
            )

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}, clear=False):
                local_store.init_db()
                local_store.insert_finding(ordinary)
                mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
                mgr.save_state(state)
                result = mgr.recover_pending_reset(completed_gen_id=11)

                self.assertTrue(result.triggered)
                self.assertTrue((shared / "ordinary_fake_gem_marker.json").exists())
                db_ids = {finding["id"] for finding in local_store.get_all_findings()}
                self.assertIn("ordinary_fake_gem_marker", db_ids)

    def test_pending_reset_recovery_rejects_unproven_mature_evidence_rows(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            selected = [
                {
                    "finding_id": "unproven",
                    "variant_name": "unproven_mature_evidence",
                    "generation_id": 0,
                    "frontier_lane": "alpha_incubator",
                    "metrics": {
                        "mean_test_taskscore": 20.0,
                        "mean_active_alpha_vs_benchmark_pct": 22.0,
                        "n_eval_cells": 29,
                        "scored_cell_count": 29,
                    },
                }
            ]
            state = {
                "enabled": True,
                "cycle_index": 1,
                "cycle_start_generation": 6,
                "reset_count": 1,
                "gems": [],
                "pending_reset": {
                    "status": "pending",
                    "reset_count": 2,
                    "cycle_index": 2,
                    "completed_gen_id": 1,
                    "next_absolute_generation": 2,
                    "signature_hash": "sig",
                    "reason": "test_unproven",
                    "archive_dir": str(run_dir / "archive" / "gems_cycle_2_unproven"),
                    "selected_entries": selected,
                },
            }
            frontier = _FakeFrontier(run_dir, {"lane_frontiers": {"alpha_incubator": selected}})
            task = _task_with_gems(
                max_gems_per_reset=4,
                max_gems_total=4,
                archive_ordinary_findings=False,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            mgr.save_state(state)

            result = mgr.recover_pending_reset(completed_gen_id=1)

            self.assertFalse(result.triggered)
            self.assertEqual(mgr.load_state()["gems"], [])

    def test_pending_reset_recovery_rejects_mature_evidence_legacy_control_anchor(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            selected = [
                {
                    "gem_finding_id": "legacy_anchor",
                    "variant_name": "benchmark_floor_anchor",
                    "frontier_lane": "benchmark_floor",
                    "metric_name": "mean_test_taskscore",
                    "metric_value": 1.0,
                    "admission_metrics": {"mean_test_taskscore": 1.0},
                }
            ]
            state = {
                "enabled": True,
                "cycle_index": 1,
                "cycle_start_generation": 6,
                "reset_count": 1,
                "gems": [],
                "pending_reset": {
                    "status": "pending",
                    "reset_count": 2,
                    "cycle_index": 2,
                    "completed_gen_id": 1,
                    "next_absolute_generation": 2,
                    "signature_hash": "sig",
                    "reason": "test_legacy_anchor",
                    "archive_dir": str(run_dir / "archive" / "gems_cycle_2_legacy"),
                    "selected_entries": selected,
                },
            }
            frontier = _FakeFrontier(run_dir, {"lane_frontiers": {"benchmark_floor": selected}})
            task = _task_with_gems(
                selection_policy="mature_evidence_top_k",
                max_gems_per_reset=4,
                max_gems_total=4,
                archive_ordinary_findings=False,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            mgr.save_state(state)

            result = mgr.recover_pending_reset(completed_gen_id=1)
            state_after = mgr.load_state()

            self.assertFalse(result.triggered)
            self.assertEqual(state_after["gems"], [])

    def test_pending_reset_recovery_rejects_nonclean_selected_entries(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            selected = [
                {
                    "finding_id": "nonclean",
                    "variant_name": "nonclean_mature_evidence",
                    "generation_id": 0,
                    "frontier_lane": "alpha_incubator",
                    "metrics": {
                        "mean_test_taskscore": 20.0,
                        "mean_active_alpha_vs_benchmark_pct": 22.0,
                        "complete_eval": True,
                        "n_eval_cells": 29,
                        "scored_cell_count": 29,
                        "clean_promotion_eligible": False,
                    },
                }
            ]
            state = {
                "enabled": True,
                "cycle_index": 1,
                "cycle_start_generation": 6,
                "reset_count": 1,
                "gems": [],
                "pending_reset": {
                    "status": "pending",
                    "reset_count": 2,
                    "cycle_index": 2,
                    "completed_gen_id": 1,
                    "next_absolute_generation": 2,
                    "signature_hash": "sig",
                    "reason": "test_nonclean",
                    "archive_dir": str(run_dir / "archive" / "gems_cycle_2_nonclean"),
                    "selected_entries": selected,
                },
            }
            frontier = _FakeFrontier(run_dir, {"lane_frontiers": {"alpha_incubator": selected}})
            task = _task_with_gems(
                max_gems_per_reset=4,
                max_gems_total=4,
                archive_ordinary_findings=False,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            mgr.save_state(state)

            result = mgr.recover_pending_reset(completed_gen_id=1)

            self.assertFalse(result.triggered)
            self.assertEqual(mgr.load_state()["gems"], [])

    def test_pending_reset_recovery_rejects_low_cell_rows_without_complete_evidence(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            selected = [
                {
                    "finding_id": "low_cell_unproven",
                    "variant_name": "low_cell_unproven",
                    "generation_id": 0,
                    "frontier_lane": "alpha_incubator",
                    "metrics": {
                        "mean_test_taskscore": 20.0,
                        "mean_active_alpha_vs_benchmark_pct": 22.0,
                        "n_eval_cells": 6,
                        "scored_cell_count": 6,
                    },
                }
            ]
            state = {
                "enabled": True,
                "cycle_index": 1,
                "cycle_start_generation": 6,
                "reset_count": 1,
                "gems": [],
                "pending_reset": {
                    "status": "pending",
                    "reset_count": 2,
                    "cycle_index": 2,
                    "completed_gen_id": 1,
                    "next_absolute_generation": 2,
                    "signature_hash": "sig",
                    "reason": "test_low_cell_unproven",
                    "archive_dir": str(run_dir / "archive" / "gems_cycle_2_low_cell"),
                    "selected_entries": selected,
                },
            }
            frontier = _FakeFrontier(run_dir, {"lane_frontiers": {"alpha_incubator": selected}})
            task = _task_with_gems(
                max_gems_per_reset=4,
                max_gems_total=4,
                archive_ordinary_findings=False,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            mgr.save_state(state)

            result = mgr.recover_pending_reset(completed_gen_id=1)

            self.assertFalse(result.triggered)
            self.assertEqual(mgr.load_state()["gems"], [])

    def test_pending_reset_recovery_rejects_modern_scored_rows_without_maturity_marker(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            selected = [
                {
                    "finding_id": "modern_unmarked",
                    "variant_name": "modern_unmarked",
                    "generation_id": 0,
                    "frontier_lane": "alpha_incubator",
                    "metric_name": "mean_test_taskscore",
                    "metric_value": 20.0,
                    "frontier_entity_key": "variant::modern_unmarked",
                    "metrics": {
                        "mean_test_taskscore": 20.0,
                        "mean_active_alpha_vs_benchmark_pct": 22.0,
                    },
                }
            ]
            state = {
                "enabled": True,
                "cycle_index": 1,
                "cycle_start_generation": 6,
                "reset_count": 1,
                "gems": [],
                "pending_reset": {
                    "status": "pending",
                    "reset_count": 2,
                    "cycle_index": 2,
                    "completed_gen_id": 1,
                    "next_absolute_generation": 2,
                    "signature_hash": "sig",
                    "reason": "test_modern_unmarked",
                    "archive_dir": str(run_dir / "archive" / "gems_cycle_2_modern_unmarked"),
                    "selected_entries": selected,
                },
            }
            frontier = _FakeFrontier(run_dir, {"lane_frontiers": {"alpha_incubator": selected}})
            task = _task_with_gems(
                max_gems_per_reset=4,
                max_gems_total=4,
                archive_ordinary_findings=False,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            mgr.save_state(state)

            result = mgr.recover_pending_reset(completed_gen_id=1)

            self.assertFalse(result.triggered)
            self.assertEqual(mgr.load_state()["gems"], [])

    def test_gems_archive_low_level_helpers_cover_guard_and_fallback_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import (
            GemsManager,
            load_active_gems_for_prompt,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            frontier = _FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []})
            mgr = GemsManager(run_dir=run_dir, task_spec=_task_with_gems(), frontier=frontier)

            self.assertEqual(mgr._copy_snapshot_if_available(snapshot_path="", gem_id="g"), "")
            self.assertEqual(
                mgr._copy_snapshot_if_available(
                    snapshot_path=root / "missing.snapshot", gem_id="g"
                ),
                str(root / "missing.snapshot"),
            )
            snapshot = run_dir / "variant.snapshot"
            snapshot.write_text("payload", encoding="utf-8")
            copied_rel = mgr._copy_snapshot_if_available(snapshot_path=snapshot, gem_id="gem_x")
            self.assertEqual(copied_rel, "gems/variants/gem_x.snapshot")
            self.assertTrue((run_dir / copied_rel).exists())

            archive_dir = run_dir / "archive" / "helpers"
            archive_dir.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "escapes run_dir"):
                mgr._require_inside_run_dir(
                    root / "outside.txt", label="outside", allow_missing=True
                )
            with self.assertRaisesRegex(RuntimeError, "archive_dir escapes"):
                mgr._safe_archive_dir(root / "archive" / "helpers")
            with self.assertRaisesRegex(RuntimeError, "archive child"):
                mgr._safe_archive_child_dir(archive_dir, "../escape")

            fd = mgr._open_archive_output_once(archive_dir / "once.jsonl")
            self.assertIsNotNone(fd)
            if fd is not None:
                os.close(fd)
            self.assertIsNone(mgr._open_archive_output_once(archive_dir / "once.jsonl"))
            mgr._write_jsonl_once(archive_dir / "rows.jsonl", [{"id": "a"}])
            self.assertEqual(
                (archive_dir / "rows.jsonl").read_text(encoding="utf-8"),
                '{"id": "a"}\n',
            )
            self.assertEqual(GemsManager._slug(" Bad/Name! "), "bad_name")

            src_dir = run_dir / "research_memory"
            dst_dir = archive_dir / "research_memory"
            (src_dir / "nested").mkdir(parents=True)
            (src_dir / "nested" / "move.txt").write_text("move", encoding="utf-8")
            dst_dir.mkdir()
            (dst_dir / "nested").mkdir()
            (dst_dir / "nested" / "move.txt").write_text("existing", encoding="utf-8")
            (src_dir / "new.txt").write_text("new", encoding="utf-8")
            mgr._merge_directory_into_archive(src=src_dir, dst=dst_dir)
            self.assertFalse(src_dir.exists())
            self.assertEqual(
                (dst_dir / "nested" / "move.txt").read_text(encoding="utf-8"), "existing"
            )
            self.assertTrue((dst_dir / "new.txt").exists())

            shared = run_dir / "shared_findings"
            shared.mkdir()
            legacy_id = "gem_r01_01_alpha"
            (shared / "bad.json").write_text("{bad", encoding="utf-8")
            (shared / "ordinary.json").write_text(
                json.dumps({"id": "ordinary", "title": "ordinary"}),
                encoding="utf-8",
            )
            (shared / "alpha.json").write_text(
                json.dumps(
                    {
                        "id": legacy_id,
                        "title": "GEM 1.1: alpha",
                        "variant_name": "alpha",
                        "peer_id": "gems_agent",
                        "metrics": {"is_gem_finding": True},
                    }
                ),
                encoding="utf-8",
            )
            inferred = mgr._infer_legacy_gem_paths_by_id({legacy_id: {"variant_name": "alpha"}})
            self.assertEqual(inferred[legacy_id], {"shared_findings/alpha.json"})

            db_path = run_dir / "shared_store.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute("CREATE TABLE findings (id TEXT PRIMARY KEY, title TEXT)")
                conn.execute("CREATE TABLE metrics (finding_id TEXT, metric_name TEXT)")
                conn.execute(
                    "CREATE TABLE finding_edges (src_finding_id TEXT, dst_finding_id TEXT)"
                )
                conn.execute("INSERT INTO findings VALUES ('keep', 'Keep')")
                conn.execute("INSERT INTO findings VALUES ('drop', 'Drop')")
                conn.execute("INSERT INTO metrics VALUES ('drop', 'score')")
                conn.execute("INSERT INTO finding_edges VALUES ('drop', 'keep')")
            mgr._archive_sqlite_rows(archive_dir=archive_dir, keep_finding_ids={"keep"})
            with sqlite3.connect(db_path) as conn:
                remaining = [row[0] for row in conn.execute("SELECT id FROM findings").fetchall()]
                self.assertEqual(remaining, ["keep"])
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0], 0)
                self.assertEqual(
                    conn.execute("SELECT COUNT(*) FROM finding_edges").fetchone()[0], 0
                )

            state_path = run_dir / "gems" / "gems_state.json"
            state_path.parent.mkdir(exist_ok=True)
            state_path.write_text(
                json.dumps(
                    {
                        "cycle_index": 3,
                        "reset_count": 2,
                        "cycle_start_generation": 12,
                        "gems": [
                            {"variant_name": "summary", "summary_only": True},
                            {
                                "variant_name": "unknown",
                                "hist_return_pct": 99.0,
                                "n_eval_cells": 99,
                            },
                            {
                                "variant_name": "alpha",
                                "gem_finding_id": "alpha",
                                "hist_return_pct": 1.0,
                                "n_eval_cells": 29,
                                "complete_eval": True,
                                "admission_metrics": {
                                    "score": 1.0,
                                    "complete_eval": True,
                                },
                            },
                            "bad",
                        ],
                        "active_bottleneck_reports": [{"r": i} for i in range(7)],
                        "latest_soft_agenda_priors": {"flag": True},
                    }
                ),
                encoding="utf-8",
            )
            prompt_gems = load_active_gems_for_prompt(run_dir, max_entries=1)
            self.assertEqual([entry["variant_name"] for entry in prompt_gems["entries"]], ["alpha"])
            self.assertTrue(prompt_gems["entries"][0]["complete_eval"])
            self.assertEqual(
                prompt_gems["bottleneck_reports"],
                [{"r": 2}, {"r": 3}, {"r": 4}, {"r": 5}, {"r": 6}],
            )
            self.assertEqual(load_active_gems_for_prompt(run_dir, max_entries=0)["entries"], [])

    def test_gems_trigger_archives_ordinary_findings_and_resets_logical_generation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "primary_metric": "future_fitness",
                "metric_direction": "maximize",
                "generations": {},
                "cumulative_top": [
                    {
                        "generation_id": 0,
                        "rank": 1,
                        "finding_id": "f_alpha",
                        "variant_name": "alpha_v1",
                        "metric_value": 0.42,
                        "metrics": {
                            "future_fitness": 0.42,
                            "aux_bad": float("nan"),
                            "tier": "T3",
                            "frontier_lane": "alpha",
                            "strategy_family": "learned_alpha",
                            "bottleneck_target": "drawdown_regression",
                            "evidence_stage": "full_T1",
                            "tradeoff_class": "high_return_drawdown_repair_target",
                            "primary_tradeoff": "return_vs_mdd",
                            "next_step_intent": "repair_failure_mode",
                            "parent_candidate": "parent_alpha",
                            "parent_usage": "repair",
                            "diversity_overlap_status": "clean",
                            "diversity_overlap_fraction": 0.25,
                            "diversity_overlap_count": 1,
                            "diversity_overlap_total": 4,
                            "diversity_narrow_variation": False,
                            "semantic_family": "temporal_gate",
                            "parent_lineage": "parent_alpha",
                            "novelty_axis": "drawdown_gate",
                        },
                        "promoted_for_lane": "alpha_incubator",
                        "frontier_lane": "alpha_incubator",
                        "lane_metric_name": "mean_active_alpha_vs_benchmark_pct",
                        "lane_metric_value": 1.7,
                    }
                ],
                "lane_frontiers": {
                    "alpha_incubator": [
                        {
                            "generation_id": 0,
                            "rank": 1,
                            "finding_id": "f_alpha",
                            "variant_name": "alpha_v1",
                            "metric_value": 0.42,
                            "metrics": {
                                "future_fitness": 0.42,
                                "aux_bad": float("nan"),
                                "tier": "T3",
                                "frontier_lane": "alpha",
                                "strategy_family": "learned_alpha",
                                "bottleneck_target": "drawdown_regression",
                                "evidence_stage": "full_T1",
                                "tradeoff_class": "high_return_drawdown_repair_target",
                                "primary_tradeoff": "return_vs_mdd",
                                "next_step_intent": "repair_failure_mode",
                                "parent_candidate": "parent_alpha",
                                "parent_usage": "repair",
                                "diversity_overlap_status": "clean",
                                "diversity_overlap_fraction": 0.25,
                                "diversity_overlap_count": 1,
                                "diversity_overlap_total": 4,
                                "diversity_narrow_variation": False,
                                "semantic_family": "temporal_gate",
                                "parent_lineage": "parent_alpha",
                                "novelty_axis": "drawdown_gate",
                            },
                            "promoted_for_lane": "alpha_incubator",
                            "frontier_lane": "alpha_incubator",
                            "lane_metric_name": "mean_active_alpha_vs_benchmark_pct",
                            "lane_metric_value": 1.7,
                        }
                    ]
                },
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(reset_interval_generations=3)
            shared = run_dir / "shared_findings"
            shared.mkdir()
            ordinary_path = shared / "gen0_peer0_alpha.json"
            ordinary_path.write_text(
                json.dumps(
                    {
                        "id": "f_alpha",
                        "finding_type": "result",
                        "title": "ordinary alpha",
                        "content": "short ordinary finding",
                        "metrics": {"future_fitness": 0.42, "tier": "T3"},
                        "variant_name": "alpha_v1",
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}, clear=False):
                local_store.init_db()
                local_store.insert_finding(
                    {
                        "id": "f_alpha",
                        "finding_type": "result",
                        "title": "ordinary alpha",
                        "content": "short ordinary finding",
                        "metrics": {"future_fitness": 0.42, "tier": "T3"},
                        "variant_name": "alpha_v1",
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                    }
                )

                mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
                self.assertFalse(mgr.maybe_trigger_after_boundary(completed_gen_id=0).triggered)
                self.assertFalse(mgr.maybe_trigger_after_boundary(completed_gen_id=1).triggered)
                result = mgr.maybe_trigger_after_boundary(completed_gen_id=2)

                self.assertTrue(result.triggered)
                self.assertEqual(result.admitted_count, 1)
                self.assertEqual(mgr.logical_generation(3), 0)

                findings = local_store.get_all_findings()
                self.assertEqual(len(findings), 1)
                self.assertTrue(findings[0]["metrics"]["is_gem_finding"])
                self.assertIsNone(findings[0]["metrics"]["aux_bad"])
                self.assertEqual(findings[0]["variant_name"], "alpha_v1")
                gem_file_text = next(shared.glob("gem_r01_*.json")).read_text()
                self.assertNotIn("NaN", gem_file_text)
                self.assertFalse(ordinary_path.exists())
                archived = list((run_dir / "archive").glob("gems_cycle_1_*/shared_findings/*.json"))
                self.assertEqual(len(archived), 1)

                state = json.loads((run_dir / "gems" / "gems_state.json").read_text())
                self.assertEqual(state["reset_count"], 1)
                self.assertEqual(state["cycle_start_generation"], 3)
                self.assertEqual(len(state["gems"]), 1)
                self.assertEqual(state["gems"][0]["bottleneck_target"], "drawdown_regression")
                self.assertEqual(
                    state["gems"][0]["tradeoff_class"],
                    "high_return_drawdown_repair_target",
                )
                self.assertEqual(state["gems"][0]["next_step_intent"], "repair_failure_mode")
                self.assertEqual(state["gems"][0]["parent_candidate"], "parent_alpha")
                self.assertEqual(state["gems"][0]["parent_usage"], "repair")
                self.assertEqual(state["gems"][0]["diversity_overlap_status"], "clean")
                self.assertEqual(state["gems"][0]["diversity_overlap_fraction"], 0.25)
                self.assertEqual(state["gems"][0]["diversity_overlap_count"], 1)
                self.assertEqual(state["gems"][0]["diversity_overlap_total"], 4)
                self.assertFalse(state["gems"][0]["diversity_narrow_variation"])
                self.assertEqual(state["gems"][0]["semantic_family"], "temporal_gate")
                self.assertEqual(state["gems"][0]["parent_lineage"], "parent_alpha")
                self.assertEqual(state["gems"][0]["novelty_axis"], "drawdown_gate")
                self.assertEqual(
                    state["gems"][0]["admission_metrics"]["diversity_overlap_fraction"],
                    0.25,
                )
                manifest_after = json.loads(
                    (run_dir / "frontier" / "frontier_manifest.json").read_text()
                )
                self.assertEqual(manifest_after["gems"]["reset_count"], 1)
                self.assertEqual(
                    manifest_after["gems"]["entries"][0]["bottleneck_target"],
                    "drawdown_regression",
                )
                self.assertEqual(
                    manifest_after["gems"]["entries"][0]["diversity_overlap_status"],
                    "clean",
                )

    def test_pending_gems_reset_can_be_recovered_transactionally(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "primary_metric": "future_fitness",
                "metric_direction": "maximize",
                "generations": {"2": [{"variant_name": "stale"}]},
                "cumulative_top": [
                    {
                        "generation_id": 2,
                        "finding_id": "f_alpha",
                        "variant_name": "alpha_v1",
                        "metric_value": 0.42,
                        "metrics": {"future_fitness": 0.42, "tier": "T3"},
                        "frontier_lane": "alpha_incubator",
                    }
                ],
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems()
            archive_dir = run_dir / "archive" / "gems_cycle_1_recovery"
            state = {
                "enabled": True,
                "cycle_index": 0,
                "cycle_start_generation": 0,
                "reset_count": 0,
                "gems": [],
                "pending_reset": {
                    "status": "pending",
                    "reset_count": 1,
                    "cycle_index": 1,
                    "completed_gen_id": 2,
                    "next_absolute_generation": 3,
                    "signature_hash": "sig",
                    "reason": "test_recovery",
                    "archive_dir": str(archive_dir),
                    "selected_entries": manifest["cumulative_top"],
                },
            }
            shared = run_dir / "shared_findings"
            shared.mkdir(parents=True)
            ordinary_path = shared / "ordinary.json"
            ordinary_path.write_text(
                json.dumps(
                    {
                        "id": "f_alpha",
                        "finding_type": "result",
                        "title": "ordinary alpha",
                        "content": "ordinary",
                        "metrics": {"future_fitness": 0.42},
                        "variant_name": "alpha_v1",
                        "peer_id": "gen2_peer0",
                        "generation_id": 2,
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}, clear=False):
                local_store.init_db()
                local_store.insert_finding(
                    {
                        "id": "f_alpha",
                        "finding_type": "result",
                        "title": "ordinary alpha",
                        "content": "ordinary",
                        "metrics": {"future_fitness": 0.42},
                        "variant_name": "alpha_v1",
                        "peer_id": "gen2_peer0",
                        "generation_id": 2,
                    }
                )
                mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
                mgr.save_state(state)

                result = mgr.recover_pending_reset(completed_gen_id=2)

                self.assertTrue(result.triggered)
                state_after = json.loads((run_dir / "gems" / "gems_state.json").read_text())
                self.assertNotIn("pending_reset", state_after)
                self.assertEqual(state_after["cycle_start_generation"], 3)
                self.assertEqual(state_after["reset_events"][0]["committed"], True)
                self.assertEqual(state_after["reset_events"][0]["recovered"], True)
                self.assertFalse(ordinary_path.exists())
                manifest_after = json.loads(
                    (run_dir / "frontier" / "frontier_manifest.json").read_text()
                )
                self.assertEqual(manifest_after["gems"]["reset_count"], 1)
                self.assertEqual(manifest_after["generations"], {})

    def test_pending_gems_reset_rejects_archive_dir_outside_run_archive(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            outside = root / "outside_archive"
            run_dir.mkdir()
            outside.mkdir()
            manifest = {
                "lane_frontiers": {},
                "cumulative_top": [
                    {
                        "generation_id": 5,
                        "finding_id": "f_alpha",
                        "variant_name": "alpha_v1",
                        "metrics": {"mean_test_taskscore": 1.0, "complete_eval": True},
                    }
                ],
            }
            state = {
                "enabled": True,
                "cycle_index": 0,
                "cycle_start_generation": 0,
                "reset_count": 0,
                "gems": [],
                "pending_reset": {
                    "status": "pending",
                    "reset_count": 1,
                    "cycle_index": 1,
                    "completed_gen_id": 5,
                    "next_absolute_generation": 6,
                    "signature_hash": "sig",
                    "reason": "bad_archive",
                    "archive_dir": str(outside),
                    "selected_entries": manifest["cumulative_top"],
                },
            }
            frontier = _FakeFrontier(run_dir, manifest)
            mgr = GemsManager(run_dir=run_dir, task_spec=_task_with_gems(), frontier=frontier)
            mgr.save_state(state)

            with self.assertRaisesRegex(RuntimeError, "archive_dir escapes run archive"):
                mgr.recover_pending_reset(completed_gen_id=5)

            self.assertFalse(any(outside.iterdir()))
            state_after = json.loads((run_dir / "gems" / "gems_state.json").read_text())
            self.assertIn("pending_reset", state_after)

    def test_gems_reset_rejects_symlinked_shared_findings_before_writing(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            outside_shared = root / "outside_shared"
            run_dir.mkdir()
            outside_shared.mkdir()
            victim = outside_shared / "victim.json"
            victim.write_text(json.dumps({"id": "victim"}), encoding="utf-8")
            (run_dir / "shared_findings").symlink_to(outside_shared, target_is_directory=True)
            manifest = {
                "lane_frontiers": {},
                "cumulative_top": [
                    {
                        "generation_id": 5,
                        "finding_id": "f_alpha",
                        "variant_name": "alpha_v1",
                        "metrics": {"mean_test_taskscore": 1.0, "complete_eval": True},
                    }
                ],
            }
            state = {
                "enabled": True,
                "cycle_index": 0,
                "cycle_start_generation": 0,
                "reset_count": 0,
                "gems": [],
                "pending_reset": {
                    "status": "pending",
                    "reset_count": 1,
                    "cycle_index": 1,
                    "completed_gen_id": 5,
                    "next_absolute_generation": 6,
                    "signature_hash": "sig",
                    "reason": "bad_shared",
                    "archive_dir": str(run_dir / "archive" / "gems_cycle_1"),
                    "selected_entries": manifest["cumulative_top"],
                },
            }
            frontier = _FakeFrontier(run_dir, manifest)
            mgr = GemsManager(run_dir=run_dir, task_spec=_task_with_gems(), frontier=frontier)
            mgr.save_state(state)

            with self.assertRaisesRegex(
                RuntimeError, "shared_findings .*run_dir|symlinked shared_findings"
            ):
                mgr.recover_pending_reset(completed_gen_id=5)

            self.assertEqual(json.loads(victim.read_text(encoding="utf-8"))["id"], "victim")
            self.assertEqual(list(outside_shared.glob("*.json")), [victim])

    def test_gems_reset_rejects_symlinked_archive_shared_findings_child(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            outside_archive_child = root / "outside_archive_child"
            run_dir.mkdir()
            outside_archive_child.mkdir()
            shared = run_dir / "shared_findings"
            shared.mkdir()
            ordinary = shared / "ordinary.json"
            ordinary.write_text(json.dumps({"id": "ordinary"}), encoding="utf-8")
            archive_dir = run_dir / "archive" / "gems_cycle_1"
            archive_dir.mkdir(parents=True)
            (archive_dir / "shared_findings").symlink_to(
                outside_archive_child,
                target_is_directory=True,
            )
            manifest = {
                "lane_frontiers": {},
                "cumulative_top": [
                    {
                        "generation_id": 5,
                        "finding_id": "f_alpha",
                        "variant_name": "alpha_v1",
                        "metrics": {"mean_test_taskscore": 1.0, "complete_eval": True},
                    }
                ],
            }
            state = {
                "enabled": True,
                "cycle_index": 0,
                "cycle_start_generation": 0,
                "reset_count": 0,
                "gems": [],
                "pending_reset": {
                    "status": "pending",
                    "reset_count": 1,
                    "cycle_index": 1,
                    "completed_gen_id": 5,
                    "next_absolute_generation": 6,
                    "signature_hash": "sig",
                    "reason": "bad_archive_child",
                    "archive_dir": str(archive_dir),
                    "selected_entries": manifest["cumulative_top"],
                },
            }
            frontier = _FakeFrontier(run_dir, manifest)
            mgr = GemsManager(run_dir=run_dir, task_spec=_task_with_gems(), frontier=frontier)
            mgr.save_state(state)

            with self.assertRaisesRegex(
                RuntimeError, "archive child shared_findings|symlinked archive child"
            ):
                mgr.recover_pending_reset(completed_gen_id=5)

            self.assertTrue(ordinary.exists())
            self.assertFalse(any(outside_archive_child.iterdir()))

    def test_gems_archive_rejects_archive_child_symlink_created_after_check(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            outside_archive_child = root / "outside_archive_child"
            run_dir.mkdir()
            outside_archive_child.mkdir()
            shared = run_dir / "shared_findings"
            shared.mkdir()
            ordinary = shared / "ordinary.json"
            ordinary.write_text(json.dumps({"id": "ordinary"}), encoding="utf-8")
            archive_dir = run_dir / "archive" / "gems_cycle_1"
            archive_dir.mkdir(parents=True)
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=_task_with_gems(),
                frontier=_FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []}),
            )
            original_safe = mgr._safe_archive_child_dir
            armed = {"value": True}

            def race_once(path: Path, name: str) -> Path:
                child = original_safe(path, name)
                if armed["value"] and name == "shared_findings":
                    armed["value"] = False
                    child.symlink_to(outside_archive_child, target_is_directory=True)
                return child

            with (
                patch.object(mgr, "_safe_archive_child_dir", side_effect=race_once),
                self.assertRaisesRegex(RuntimeError, "archive child|symlinked"),
            ):
                mgr._archive_shared_findings(
                    archive_dir=archive_dir,
                    keep_finding_ids=set(),
                    keep_finding_paths_by_id={},
                )

            self.assertTrue(ordinary.exists())
            self.assertFalse(any(outside_archive_child.iterdir()))

    def test_gems_archive_rejects_dangling_symlink_jsonl_leaf(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            outside = root / "outside"
            run_dir.mkdir()
            outside.mkdir()
            archive_dir = run_dir / "archive" / "gems_cycle_1"
            archive_dir.mkdir(parents=True)
            outside_target = outside / "created_by_archive.jsonl"
            (archive_dir / "metrics_before_archive.jsonl").symlink_to(outside_target)
            frontier = _FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []})
            mgr = GemsManager(run_dir=run_dir, task_spec=_task_with_gems(), frontier=frontier)

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}, clear=False):
                local_store.init_db()
                local_store.insert_metric(
                    {
                        "run_id": "run",
                        "variant_name": "v",
                        "metrics": {"score": 1.0},
                    }
                )
                with self.assertRaisesRegex(RuntimeError, "symlinked archive file"):
                    mgr._archive_sqlite_rows(archive_dir=archive_dir, keep_finding_ids=set())

            self.assertFalse(outside_target.exists())

    def test_gems_archive_rejects_dangling_symlink_frontier_manifest_leaf(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            outside = root / "outside"
            run_dir.mkdir()
            outside.mkdir()
            shared = run_dir / "shared_findings"
            shared.mkdir()
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir()
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps({"lane_frontiers": {}, "cumulative_top": []}),
                encoding="utf-8",
            )
            archive_dir = run_dir / "archive" / "gems_cycle_1"
            archive_dir.mkdir(parents=True)
            outside_target = outside / "frontier_copy.json"
            (archive_dir / "frontier_manifest_before_gems_reset.json").symlink_to(outside_target)
            manifest = {
                "lane_frontiers": {},
                "cumulative_top": [
                    {
                        "finding_id": "f_alpha",
                        "variant_name": "alpha_v1",
                        "metrics": {"mean_test_taskscore": 1.0},
                    }
                ],
            }
            frontier = _FakeFrontier(run_dir, manifest)
            mgr = GemsManager(run_dir=run_dir, task_spec=_task_with_gems(), frontier=frontier)

            with self.assertRaisesRegex(RuntimeError, "symlinked archive file"):
                mgr._archive_and_prune_active_context(
                    archive_dir=archive_dir,
                    keep_finding_ids=set(),
                    keep_finding_paths_by_id={},
                )

            self.assertFalse(outside_target.exists())

    def test_gems_archive_copy_refuses_leaf_symlink_created_after_path_check(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            outside = root / "outside"
            run_dir.mkdir()
            outside.mkdir()
            source = run_dir / "frontier" / "frontier_manifest.json"
            source.parent.mkdir()
            source.write_text(json.dumps({"ok": True}), encoding="utf-8")
            archive_dir = run_dir / "archive" / "gems_cycle_1"
            archive_dir.mkdir(parents=True)
            dest = archive_dir / "frontier_manifest_before_gems_reset.json"
            outside_target = outside / "race_written.json"
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=_task_with_gems(),
                frontier=_FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []}),
            )
            original_safe = mgr._safe_archive_file_path
            armed = {"value": True}

            def race_once(path: Path) -> Path:
                result = original_safe(path)
                if armed["value"] and path == dest:
                    armed["value"] = False
                    path.symlink_to(outside_target)
                return result

            with (
                patch.object(mgr, "_safe_archive_file_path", side_effect=race_once),
                self.assertRaisesRegex(RuntimeError, "symlinked archive file"),
            ):
                mgr._copy_archive_file_once(source, dest)

            self.assertFalse(outside_target.exists())

    def test_gems_archive_copy_opens_source_before_creating_destination(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            source = run_dir / "frontier" / "frontier_manifest.json"
            source.parent.mkdir()
            source.write_text(json.dumps({"ok": True}), encoding="utf-8")
            archive_dir = run_dir / "archive" / "gems_cycle_1"
            archive_dir.mkdir(parents=True)
            dest = archive_dir / "frontier_manifest_before_gems_reset.json"
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=_task_with_gems(),
                frontier=_FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []}),
            )
            original_require = mgr._require_inside_run_dir
            armed = {"value": True}

            def remove_source_once(path: Path, *, label: str, allow_missing: bool = False) -> Path:
                result = original_require(path, label=label, allow_missing=allow_missing)
                if armed["value"] and label == "archive source file":
                    armed["value"] = False
                    source.unlink()
                return result

            with (
                patch.object(mgr, "_require_inside_run_dir", side_effect=remove_source_once),
                self.assertRaisesRegex(RuntimeError, "archive source file"),
            ):
                mgr._copy_archive_file_once(source, dest)

            self.assertFalse(dest.exists())

    def test_gems_reset_rejects_symlinked_shared_store_before_pruning(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            outside_store_dir = root / "outside_store"
            run_dir.mkdir()
            outside_store_dir.mkdir()
            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(outside_store_dir)}, clear=False):
                local_store.init_db()
                local_store.insert_finding(
                    {
                        "id": "outside_finding",
                        "finding_type": "result",
                        "title": "outside",
                        "content": "outside",
                        "metrics": {"mean_test_taskscore": 1.0},
                        "variant_name": "outside",
                        "peer_id": "peer",
                        "generation_id": 0,
                    }
                )
            outside_db = outside_store_dir / "shared_store.db"
            (run_dir / "shared_store.db").symlink_to(outside_db)
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=_task_with_gems(),
                frontier=_FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []}),
            )

            with self.assertRaisesRegex(
                RuntimeError, "shared_store.db .*run_dir|symlinked shared_store.db"
            ):
                mgr._archive_sqlite_rows(
                    archive_dir=run_dir / "archive" / "safe",
                    keep_finding_ids=set(),
                )

            with sqlite3.connect(outside_db) as conn:
                count = conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0]
            self.assertEqual(count, 1)

    def test_gems_archive_backs_up_metrics_before_active_context_clear(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            archive_dir = run_dir / "archive" / "metrics_backup"
            archive_dir.mkdir(parents=True)
            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}, clear=False):
                local_store.init_db()
                local_store.insert_finding(
                    {
                        "id": "keep",
                        "finding_type": "result",
                        "title": "keep",
                        "content": "keep",
                        "metrics": {"mean_test_taskscore": 1.0},
                        "variant_name": "keep",
                        "peer_id": "peer",
                        "generation_id": 0,
                    }
                )
                local_store.insert_metric(
                    {
                        "run_id": "run",
                        "variant_name": "keep",
                        "metrics": {"score": 1.23},
                        "peer_id": "peer",
                        "generation_id": 0,
                    }
                )
                mgr = GemsManager(
                    run_dir=run_dir,
                    task_spec=_task_with_gems(),
                    frontier=_FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []}),
                )

                mgr._archive_sqlite_rows(archive_dir=archive_dir, keep_finding_ids={"keep"})

                backup_rows = [
                    json.loads(line)
                    for line in (archive_dir / "metrics_before_archive.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                self.assertEqual(len(backup_rows), 1)
                self.assertEqual(json.loads(backup_rows[0]["metrics"])["score"], 1.23)
                with sqlite3.connect(run_dir / "shared_store.db") as conn:
                    metric_count = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
                    finding_count = conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0]
                self.assertEqual(metric_count, 0)
                self.assertEqual(finding_count, 1)

    def test_frontier_tool_exposes_gems(self) -> None:
        from praxist.plugins.tools.frontier_tools.adapter import _handle_get_frontier

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "frontier"
            frontier_dir.mkdir()
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {},
                        "cumulative_top": [],
                        "lane_frontiers": {},
                        "gems": {
                            "cycle_index": 1,
                            "reset_count": 1,
                            "cycle_start_generation": 3,
                            "entries": [
                                {
                                    "gem_finding_id": f"gem_r01_{i:02d}_alpha",
                                    "variant_name": f"alpha_v{i}",
                                    "frontier_lane": "alpha_incubator",
                                    "metric_name": "active_alpha",
                                    "metric_value": 1.7 + i,
                                    "complete_eval": True,
                                }
                                for i in range(16)
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"FRONTIER_DIR": str(frontier_dir), "PRAXIST_FRONTIER_ALLOW_UNBOUNDED": "1"},
                clear=False,
            ):
                out = asyncio.run(_handle_get_frontier({"top_k": 5, "up_to_generation": -1}))
            payload = json.loads(out["content"][0]["text"])
            self.assertIn("gems", payload)
            self.assertEqual(payload["gems"]["reset_count"], 1)
            self.assertEqual(len(payload["gems"]["entries"]), 16)
            self.assertEqual(payload["gems"]["entries"][0]["variant_name"], "alpha_v0")
            self.assertEqual(payload["gems"]["entries"][-1]["variant_name"], "alpha_v15")

    def test_frontier_store_exposes_generation_summary_for_run_summary(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
            FrontierStore,
        )

        with tempfile.TemporaryDirectory() as tmp:
            frontier = FrontierStore(
                Path(tmp) / "frontier",
                promote_top_k=1,
                primary_metric="score",
            )
            promoted = frontier.promote(
                4,
                [
                    {
                        "id": "f",
                        "finding_type": "result",
                        "title": "result",
                        "variant_name": "v",
                        "metrics": {"score": 0.7},
                    }
                ],
            )

            self.assertEqual(frontier.get_summary_for_generation(4), promoted)
            self.assertEqual(frontier.get_summary_for_generation(9), [])

    def test_prompt_context_keeps_all_cumulative_gems(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier = _FakeFrontier(run_dir, {"cumulative_top": []})
            task = _task_with_gems(prompt_max_gems=1)
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            state = mgr.load_state()
            state["gems"] = [
                {
                    "gem_finding_id": f"gem_{i}",
                    "variant_name": f"alpha_{i}",
                    "frontier_lane": "alpha_incubator",
                    "source_generation_id": 0,
                    "evidence_stage": "full_T1",
                    "admission_metrics": {"mean_test_taskscore": float(i + 1)},
                    "bottleneck_target": "L1_underutilization" if i == 3 else "",
                    "tradeoff_class": "L1_behavior_candidate" if i == 3 else "",
                    "primary_tradeoff": "L1_evidence_vs_stress" if i == 3 else "",
                    "next_step_intent": "preserve_and_validate" if i == 3 else "",
                    "parent_candidate": "l1_parent" if i == 3 else "",
                    "parent_usage": "stress_validate" if i == 3 else "",
                }
                for i in range(4)
            ]
            mgr.save_state(state)

            context = mgr.prompt_context(absolute_gen_id=7)

            self.assertEqual(context["gems_count"], 4)
            self.assertEqual(len(context["gems"]), 4)
            self.assertEqual(context["gems"][-1]["variant_name"], "alpha_3")
            self.assertEqual(context["gems"][-1]["bottleneck_target"], "L1_underutilization")
            self.assertEqual(context["gems"][-1]["evidence_stage"], "full_T1")
            self.assertEqual(context["gems"][-1]["tradeoff_class"], "L1_behavior_candidate")
            self.assertEqual(context["gems"][-1]["primary_tradeoff"], "L1_evidence_vs_stress")
            self.assertEqual(context["gems"][-1]["next_step_intent"], "preserve_and_validate")
            self.assertEqual(context["gems"][-1]["parent_candidate"], "l1_parent")
            self.assertEqual(context["gems"][-1]["parent_usage"], "stress_validate")

    def test_prompt_context_respects_task_configured_gem_capacity(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier = _FakeFrontier(run_dir, {"cumulative_top": []})
            task = _task_with_gems(max_gems_total=12, max_gems_per_family=12)
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            state = mgr.load_state()
            state["gems"] = [
                {
                    "gem_finding_id": f"gem_{i}",
                    "variant_name": f"alpha_{i}",
                    "frontier_lane": "alpha_incubator",
                    "strategy_family": f"family_{i}",
                    "source_generation_id": 0,
                    "evidence_stage": "full_T1",
                    "admission_metrics": {"mean_test_taskscore": float(i + 1)},
                }
                for i in range(12)
            ]
            mgr.save_state(state)

            context = mgr.prompt_context(absolute_gen_id=7)

            self.assertEqual(context["gems_count"], 12)
            self.assertEqual(len(context["gems"]), 12)
            self.assertEqual(context["gems"][-1]["variant_name"], "alpha_11")

    def test_prompt_context_assigns_balanced_gem_seeded_anchors(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier = _FakeFrontier(run_dir, {"cumulative_top": []})
            task = _task_with_gems(max_gems_total=4, max_gems_per_family=4)
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            state = mgr.load_state()
            state["reset_count"] = 1
            state["cycle_index"] = 1
            state["cycle_start_generation"] = 6
            state["gems"] = [
                {
                    "gem_finding_id": f"gem_{i}",
                    "variant_name": f"alpha_{i}",
                    "frontier_lane": "alpha_incubator",
                    "strategy_family": f"family_{i}",
                    "source_generation_id": 0,
                    "evidence_stage": "full_T1",
                    "admission_metrics": {"mean_test_taskscore": float(i + 1)},
                }
                for i in range(4)
            ]
            mgr.save_state(state)

            context = mgr.prompt_context(absolute_gen_id=6, peer_index=1, cohort_size=8)

            self.assertTrue(context["gem_seeded_baseline_mode"])
            self.assertIn("implementation or mechanism parents", context["baseline_code_policy"])
            self.assertIn("performance references", context["official_baseline_performance_policy"])
            self.assertEqual(context["primary_gem_anchor"]["variant_name"], "alpha_1")
            self.assertEqual(context["secondary_gem_anchor"]["variant_name"], "alpha_2")
            self.assertEqual(len(context["gem_anchor_roster"]), 8)
            self.assertEqual(
                [row["primary_variant_name"] for row in context["gem_anchor_roster"]],
                [
                    "alpha_0",
                    "alpha_1",
                    "alpha_2",
                    "alpha_3",
                    "",
                    "",
                    "",
                    "",
                ],
            )
            self.assertEqual(
                [row["assignment_type"] for row in context["gem_anchor_roster"]],
                [
                    "gem_inheritance",
                    "gem_inheritance",
                    "gem_inheritance",
                    "gem_inheritance",
                    "independent_exploration_or_recombination",
                    "independent_exploration_or_recombination",
                    "independent_exploration_or_recombination",
                    "independent_exploration_or_recombination",
                ],
            )

            independent_context = mgr.prompt_context(
                absolute_gen_id=6,
                peer_index=5,
                cohort_size=8,
            )
            self.assertEqual(
                independent_context["gem_anchor_assignment_mode"],
                "independent_exploration_or_recombination",
            )
            self.assertEqual(independent_context["primary_gem_anchor"], {})

    def test_prompt_context_and_single_pi_gems_apply_generation_cutoff(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager
        from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import PIAgent

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier = _FakeFrontier(run_dir, {"cumulative_top": []})
            task = _task_with_gems(max_gems_total=4, max_gems_per_family=4)
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            state = mgr.load_state()
            state["reset_count"] = 1
            state["cycle_index"] = 1
            state["cycle_start_generation"] = 1
            state["gems"] = [
                {
                    "gem_finding_id": "gem_current",
                    "variant_name": "current_gem",
                    "frontier_lane": "performance",
                    "source_generation_id": 0,
                    "evidence_stage": "full_T1",
                    "admission_metrics": {"score": 1.0, "scored_complete": True},
                },
                {
                    "gem_finding_id": "gem_future",
                    "variant_name": "same_generation_gem",
                    "frontier_lane": "performance",
                    "source_generation_id": 1,
                    "evidence_stage": "full_T1",
                    "admission_metrics": {"score": 99.0, "scored_complete": True},
                },
            ]
            state["active_bottleneck_reports"] = [
                {
                    "completed_generation": 0,
                    "records": [{"gem_type": "current_gap"}],
                    "soft_agenda_priors": {"current_prior": 0.1},
                },
                {
                    "completed_generation": 1,
                    "records": [{"gem_type": "same_generation_gap"}],
                    "soft_agenda_priors": {"same_generation_prior": 0.9},
                },
            ]
            state["latest_soft_agenda_priors"] = {"same_generation_prior": 0.9}
            mgr.save_state(state)

            peer_context = mgr.prompt_context(absolute_gen_id=1, peer_index=0, cohort_size=1)
            pi_context = PIAgent(
                run_dir=run_dir,
                workspace=run_dir,
                cohort_size=1,
                model="noop",
            )._load_gems_context(completed_gen_id=0)

            for context in (peer_context, pi_context):
                payload = json.dumps(context, sort_keys=True, default=str)
                self.assertIn("current_gem", payload)
                self.assertIn("current_gap", payload)
                self.assertIn("current_prior", payload)
                self.assertNotIn("same_generation_gem", payload)
                self.assertNotIn("same_generation_gap", payload)
                self.assertNotIn("same_generation_prior", payload)

    def test_prompt_context_does_not_seed_later_logical_generations(self) -> None:
        from jinja2 import Environment, FileSystemLoader

        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier = _FakeFrontier(run_dir, {"cumulative_top": []})
            task = _task_with_gems(max_gems_total=4, max_gems_per_family=4)
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            state = mgr.load_state()
            state["reset_count"] = 1
            state["cycle_index"] = 1
            state["cycle_start_generation"] = 6
            state["gems"] = [
                {
                    "gem_finding_id": f"gem_{i}",
                    "variant_name": f"alpha_{i}",
                    "frontier_lane": "alpha_incubator",
                    "strategy_family": f"family_{i}",
                    "source_generation_id": 0,
                    "evidence_stage": "full_T1",
                    "admission_metrics": {"mean_test_taskscore": float(i + 1)},
                }
                for i in range(4)
            ]
            mgr.save_state(state)

            context = mgr.prompt_context(absolute_gen_id=7, peer_index=1, cohort_size=8)

            self.assertFalse(context["gem_seeded_baseline_mode"])
            self.assertEqual(context["primary_gem_anchor"], {})
            self.assertEqual(context["secondary_gem_anchor"], {})
            self.assertEqual(context["gem_anchor_roster"], [])

            backend_dir = (
                Path(__file__).resolve().parents[2]
                / "praxist"
                / "plugins"
                / "workflow_stages"
                / "research_loop"
                / "backend"
            )
            template = Environment(loader=FileSystemLoader(str(backend_dir))).get_template(
                "prompt_base.jinja2",
            )
            rendered = template.render(
                peer_id="gen7_peer1",
                gen_id=7,
                logical_gen_id=1,
                cohort_size=8,
                workspace_dir="/workspace",
                variants_dir="/workspace/variants",
                results_dir="/workspace/results",
                findings_dir="/workspace/shared_findings",
                notebook_path="/workspace/notebook.json",
                logs_dir="/workspace/logs",
                graph_session_context="",
                gems_context=context,
            )
            self.assertNotIn("Gem-Seeded Baseline Mode", rendered)

    def test_server_mode_does_not_archive_or_reset(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_surface_narrowing_findings(run_dir / "shared_findings", generation_id=2)
            manifest = {
                "cumulative_top": [
                    {
                        "generation_id": 0,
                        "finding_id": "f_alpha",
                        "variant_name": "alpha_v1",
                        "metric_value": 0.42,
                        "metrics": {"future_fitness": 0.42},
                    }
                ]
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems()
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=task,
                frontier=frontier,
                local_mode=False,
            )

            result = mgr.maybe_trigger_after_boundary(completed_gen_id=2)

            self.assertFalse(result.triggered)
            self.assertEqual(result.reason, "server_mode_not_supported")
            self.assertFalse((run_dir / "archive").exists())
            state = json.loads((run_dir / "gems" / "gems_state.json").read_text())
            self.assertEqual(state["reset_count"], 0)
            self.assertTrue(state["bottleneck_history"])
            manifest_after = json.loads(
                (run_dir / "frontier" / "frontier_manifest.json").read_text()
            )
            self.assertIn("bottleneck_reports", manifest_after["gems"])

    def test_evidence_pack_digest_caps_cumulative_gems_context(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
            _digest_gems,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gems_dir = run_dir / "gems"
            gems_dir.mkdir()
            (gems_dir / "gems_state.json").write_text(
                json.dumps(
                    {
                        "cycle_index": 3,
                        "reset_count": 3,
                        "cycle_start_generation": 12,
                        "gems": [
                            {
                                "gem_finding_id": f"gem_{i}",
                                "variant_name": f"alpha_{i}",
                                "frontier_lane": "alpha_incubator",
                                "source_generation_id": 0,
                                "bottleneck_target": "drawdown_regression" if i == 0 else "",
                                "evidence_stage": "full_T1" if i < 4 else "",
                                "tradeoff_class": (
                                    "high_return_drawdown_repair_target" if i == 0 else ""
                                ),
                                "primary_tradeoff": "return_vs_mdd" if i == 0 else "",
                                "next_step_intent": "repair_failure_mode" if i == 0 else "",
                                "parent_candidate": "parent_alpha" if i == 0 else "",
                                "parent_usage": "repair" if i == 0 else "",
                                "admission_metrics": {
                                    "mean_test_taskscore": 12.5 - i,
                                    "mean_active_alpha_vs_benchmark_pct": 4.5 - i,
                                    "complete_eval": True,
                                }
                                if i < 4
                                else {},
                            }
                            for i in range(30)
                        ],
                    }
                ),
                encoding="utf-8",
            )

            digest = _digest_gems(run_dir)

            self.assertEqual(digest["reset_count"], 3)
            self.assertEqual(len(digest["entries"]), 4)
            self.assertEqual(digest["entries"][-1]["variant_name"], "alpha_3")
            self.assertNotIn("alpha_4", {entry["variant_name"] for entry in digest["entries"]})
            self.assertEqual(digest["entries"][0]["bottleneck_target"], "drawdown_regression")
            self.assertEqual(digest["entries"][0]["evidence_stage"], "full_T1")
            self.assertEqual(
                digest["entries"][0]["tradeoff_class"],
                "high_return_drawdown_repair_target",
            )
            self.assertEqual(digest["entries"][0]["primary_tradeoff"], "return_vs_mdd")
            self.assertEqual(
                digest["entries"][0]["admission_metrics"]["mean_test_taskscore"],
                12.5,
            )
            self.assertEqual(digest["entries"][0]["next_step_intent"], "repair_failure_mode")
            self.assertEqual(digest["entries"][0]["parent_candidate"], "parent_alpha")
            self.assertEqual(digest["entries"][0]["parent_usage"], "repair")

    def test_evidence_cards_and_lane_digest_preserve_generic_and_configured_fields(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.card_builder import (
            build_card_from_finding,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
            _digest_lane_frontiers,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            metrics = {
                "future_fitness": 1.0,
                "custom_task_metric": 5.5,
                "custom_surface_label": "surface_a",
                "custom_surface_tags": ["mask", "stress"],
            }
            card = build_card_from_finding(
                {
                    "id": "f_custom",
                    "finding_type": "result",
                    "title": "Configured task metric result",
                    "metrics": metrics,
                    "extra": {
                        "extra": {
                            "bottleneck_target": "underused_surface",
                            "evidence_stage": "full_T1",
                            "tradeoff_class": "surface_vs_stability",
                            "primary_tradeoff": "custom_metric_vs_stress",
                            "next_step_intent": "preserve_and_validate",
                            "parent_candidate": "parent_candidate",
                            "parent_usage": "stress_validate",
                        },
                    },
                    "variant_name": "custom_policy",
                    "generation_id": 1,
                    "peer_id": "gen1_peer0",
                },
                run_dir,
            )

            self.assertEqual(card["metrics"]["custom_task_metric"], 5.5)
            self.assertNotIn("custom_surface_label", card["metrics"])
            self.assertEqual(card["metrics"]["evidence_stage"], "full_T1")
            self.assertEqual(card["metrics"]["bottleneck_target"], "underused_surface")
            self.assertEqual(card["metrics"]["tradeoff_class"], "surface_vs_stability")
            self.assertEqual(card["metrics"]["primary_tradeoff"], "custom_metric_vs_stress")
            self.assertEqual(card["metrics"]["next_step_intent"], "preserve_and_validate")
            self.assertEqual(card["metrics"]["parent_candidate"], "parent_candidate")
            self.assertEqual(card["metrics"]["parent_usage"], "stress_validate")

            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir()
            lane_metrics = dict(
                metrics,
                mature_enough=True,
                bottleneck_target="underused_surface",
                evidence_stage="full_T1",
                tradeoff_class="surface_vs_stability",
                primary_tradeoff="custom_metric_vs_stress",
                next_step_intent="preserve_and_validate",
                parent_candidate="parent_candidate",
                parent_usage="stress_validate",
            )
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "gems": {
                            "primary_metric_keys": ["custom_task_metric"],
                            "secondary_metric_keys": [
                                "custom_surface_label",
                                "custom_surface_tags",
                            ],
                        },
                        "lane_frontiers": {
                            "performance": [
                                {
                                    "finding_id": "f_custom",
                                    "variant_name": "custom_policy",
                                    "frontier_lane": "performance",
                                    "metrics": lane_metrics,
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            digest = _digest_lane_frontiers(run_dir)
            entry = digest["performance"][0]
            self.assertEqual(entry["custom_task_metric"], 5.5)
            self.assertEqual(entry["custom_surface_label"], "surface_a")
            self.assertEqual(entry["custom_surface_tags"], ["mask", "stress"])
            self.assertEqual(entry["bottleneck_target"], "underused_surface")
            self.assertEqual(entry["evidence_stage"], "full_T1")
            self.assertEqual(entry["tradeoff_class"], "surface_vs_stability")
            self.assertEqual(entry["primary_tradeoff"], "custom_metric_vs_stress")
            self.assertEqual(entry["next_step_intent"], "preserve_and_validate")
            self.assertEqual(entry["parent_candidate"], "parent_candidate")
            self.assertEqual(entry["parent_usage"], "stress_validate")

    def test_evidence_card_preserves_generic_metadata_list(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.card_builder import (
            build_card_from_finding,
        )

        with tempfile.TemporaryDirectory() as tmp:
            card = build_card_from_finding(
                {
                    "id": "f_l1_list",
                    "finding_type": "result",
                    "title": "Generic evidence list",
                    "variant_name": "generic_list",
                    "metrics": {
                        "tradeoff_class": ["mask", "policy_stress"],
                    },
                },
                Path(tmp),
            )

            self.assertEqual(card["metrics"]["tradeoff_class"], ["mask", "policy_stress"])

    def test_lane_digest_accepts_legacy_mean_top_weight_aliases(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
            _digest_lane_frontiers,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir()
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "gems": {
                            "secondary_metric_keys": [
                                "mean_top1_weight",
                                "mean_top5_weight",
                                "mean_top10_weight",
                            ],
                            "result_metric_aliases": {
                                "top1_weight_mean": "mean_top1_weight",
                                "top5_weight_mean": "mean_top5_weight",
                                "top10_weight_mean": "mean_top10_weight",
                                "max_single_name_weight": "mean_top1_weight",
                            },
                        },
                        "lane_frontiers": {
                            "alpha_incubator": [
                                {
                                    "finding_id": "f_conc",
                                    "variant_name": "conc_probe",
                                    "evidence_stage": "full_T1",
                                    "mature_enough": True,
                                    "metrics": {
                                        "mean_top1_weight": 0.44,
                                        "mean_top5_weight": 0.70,
                                        "mean_top10_weight": 0.86,
                                    },
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            entry = _digest_lane_frontiers(run_dir)["alpha_incubator"][0]
            self.assertEqual(entry["top1_weight_mean"], 0.44)
            self.assertEqual(entry["top5_weight_mean"], 0.70)
            self.assertEqual(entry["top10_weight_mean"], 0.86)
            self.assertEqual(entry["max_single_name_weight"], 0.44)

    def test_context_firewall_budgeting_caps_gem_entries(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory import (
            context_firewall,
        )

        payload = {
            "gems": {
                "entries": [
                    {
                        "gem_finding_id": f"gem_{i}",
                        "variant_name": f"alpha_{i}",
                        "frontier_lane": "alpha_incubator",
                        "bottleneck_target": "drawdown_regression" if i == 3 else "",
                        "evidence_stage": "full_T1" if i == 3 else "",
                        "tradeoff_class": ("high_return_drawdown_repair_target" if i == 3 else ""),
                        "primary_tradeoff": "return_vs_mdd" if i == 3 else "",
                        "next_step_intent": "repair_failure_mode" if i == 3 else "",
                        "parent_candidate": "parent_alpha" if i == 3 else "",
                        "parent_usage": "repair" if i == 3 else "",
                        "admission_metrics": {
                            "mean_test_taskscore": 13.5,
                            "mean_active_alpha_vs_benchmark_pct": 7.5,
                        }
                        if i == 3
                        else {},
                    }
                    for i in range(30)
                ]
            },
            "findings_summary": ["x" * 100 for _ in range(100)],
        }

        shrunk = context_firewall.shrink_dict(payload, budget_tokens=1)

        self.assertEqual(len(shrunk["gems"]["entries"]), 4)
        self.assertEqual(shrunk["gems"]["entries"][-1]["variant_name"], "alpha_3")
        self.assertEqual(shrunk["gems"]["entries"][-1]["bottleneck_target"], "drawdown_regression")
        self.assertEqual(shrunk["gems"]["entries"][-1]["evidence_stage"], "full_T1")
        self.assertEqual(
            shrunk["gems"]["entries"][-1]["tradeoff_class"],
            "high_return_drawdown_repair_target",
        )
        self.assertEqual(shrunk["gems"]["entries"][-1]["primary_tradeoff"], "return_vs_mdd")
        self.assertEqual(shrunk["gems"]["entries"][-1]["next_step_intent"], "repair_failure_mode")
        self.assertEqual(shrunk["gems"]["entries"][-1]["parent_candidate"], "parent_alpha")
        self.assertEqual(shrunk["gems"]["entries"][-1]["parent_usage"], "repair")
        self.assertEqual(
            shrunk["gems"]["entries"][-1]["admission_metrics"]["mean_test_taskscore"],
            13.5,
        )

    def test_gem_selection_reserves_included_lane_quotas(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "confirmed_alpha": [
                        {
                            "finding_id": f"c{i}",
                            "variant_name": f"confirmed_{i}",
                            "generation_id": 0,
                            "metrics": {
                                "mean_test_taskscore": float(10 - i),
                                "complete_eval": True,
                            },
                        }
                        for i in range(8)
                    ],
                    "alpha_incubator": [
                        {
                            "finding_id": f"a{i}",
                            "variant_name": f"incubator_{i}",
                            "generation_id": 0,
                            "metrics": {
                                "mean_test_taskscore": float(30 - i),
                                "complete_eval": True,
                            },
                        }
                        for i in range(30)
                    ],
                    "benchmark_floor": [
                        {
                            "finding_id": "b0",
                            "variant_name": "benchmark_0",
                            "generation_id": 0,
                            "metrics": {"mean_test_taskscore": 0.0, "complete_eval": True},
                        }
                    ],
                    "diagnostic_control": [
                        {
                            "finding_id": "d0",
                            "variant_name": "diagnostic_0",
                            "generation_id": 0,
                            "metrics": {"mean_test_taskscore": -1.0, "complete_eval": True},
                        }
                    ],
                }
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(
                max_gems_per_reset=16,
                max_gems_total=16,
                include_lanes=[
                    "confirmed_alpha",
                    "alpha_incubator",
                    "benchmark_floor",
                    "diagnostic_control",
                ],
            )
            task.evaluation = SimpleNamespace(
                maturity_policy={
                    "complete_stage_labels": ["T1", "T2", "T3", "full_T1"],
                    "preliminary_stage_labels": ["smoke", "scout", "partial"],
                },
                frontier_lanes=[
                    {"name": "confirmed_alpha", "k": 2},
                    {"name": "alpha_incubator", "k": 10},
                    {"name": "benchmark_floor", "k": 1},
                    {"name": "diagnostic_control", "k": 1},
                ],
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(manifest)
            lanes = [entry.get("frontier_lane") for entry in selected]

            self.assertLessEqual(len(selected), 16)
            self.assertEqual(len(selected), 16)
            self.assertGreaterEqual(lanes.count("alpha_incubator"), 2)
            self.assertGreaterEqual(lanes.count("benchmark_floor"), 1)
            self.assertGreaterEqual(lanes.count("diagnostic_control"), 1)

    def test_gem_selection_dedupes_variants_and_respects_global_cap(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "alpha_incubator": [
                        {
                            "finding_id": "f0",
                            "variant_name": "bridge_l1_c005",
                            "generation_id": 0,
                            "metrics": {
                                "strategy_family": "bridge",
                                "mean_test_taskscore": 10.0,
                                "complete_eval": True,
                            },
                        },
                        {
                            "finding_id": "f0_duplicate",
                            "variant_name": "bridge_l1_c005",
                            "generation_id": 0,
                            "metrics": {
                                "strategy_family": "bridge",
                                "mean_test_taskscore": 9.0,
                                "complete_eval": True,
                            },
                        },
                        {
                            "finding_id": "f1",
                            "variant_name": "lstm_l1_c002",
                            "generation_id": 0,
                            "metrics": {
                                "strategy_family": "lstm",
                                "mean_test_taskscore": 8.0,
                                "complete_eval": True,
                            },
                        },
                        {
                            "finding_id": "f2",
                            "variant_name": "vtrace_l1_probe",
                            "generation_id": 0,
                            "metrics": {
                                "strategy_family": "vtrace",
                                "mean_test_taskscore": 7.0,
                                "complete_eval": True,
                            },
                        },
                        {
                            "finding_id": "f3",
                            "variant_name": "extra_l1_probe",
                            "generation_id": 0,
                            "metrics": {
                                "strategy_family": "extra",
                                "mean_test_taskscore": 6.0,
                                "complete_eval": True,
                            },
                        },
                    ]
                }
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(
                max_gems_per_reset=10,
                max_gems_total=3,
                max_gems_per_family=10,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            selected = mgr._select_gem_entries(manifest)
            variants = [entry.get("variant_name") for entry in selected]

            self.assertEqual(len(selected), 3)
            self.assertEqual(variants, ["bridge_l1_c005", "lstm_l1_c002", "vtrace_l1_probe"])

    def test_periodic_gems_reset_does_not_defer_for_current_generation_positive_alpha(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "alpha_incubator": [
                        {
                            "generation_id": 1,
                            "rank": 1,
                            "finding_id": "f_new",
                            "variant_name": "new_alpha",
                            "frontier_lane": "alpha_incubator",
                            "lane_metric_name": "mean_active_alpha_vs_benchmark_pct",
                            "lane_metric_value": 0.9,
                            "metrics": {
                                "mean_active_alpha_vs_benchmark_pct": 0.9,
                                "strategy_family": "learned_alpha",
                                "complete_eval": True,
                            },
                        }
                    ]
                }
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(reset_interval_generations=2, max_gems_total=8)
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            result = mgr.maybe_trigger_after_boundary(completed_gen_id=1)
            updated = mgr.load_state()

            self.assertTrue(result.triggered)
            self.assertIn("periodic_reset_every_2_generations", result.reason)
            self.assertEqual(mgr.load_state().get("reset_count"), 1)
            self.assertNotIn("reset_defer_history", updated)

    def test_validation_candidates_are_not_gems_parents(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {},
                "cumulative_top": [],
                "validation_candidates": {
                    "cumulative": [
                        {
                            "generation_id": 0,
                            "finding_id": "scout-high",
                            "variant_name": "scout_high",
                            "metric_name": "mean_active_alpha_vs_benchmark_pct",
                            "metric_value": 999.0,
                            "metric_direction": "maximize",
                            "frontier_entity_key": "variant::scout_high",
                            "excluded_from_durable_frontier": True,
                        }
                    ]
                },
            }
            frontier = _FakeFrontier(run_dir, manifest)
            mgr = GemsManager(run_dir=run_dir, task_spec=_task_with_gems(), frontier=frontier)

            self.assertEqual(mgr._manifest_entries(manifest), [])
            self.assertEqual(mgr._select_gem_entries(manifest), [])

    def test_validation_candidate_copied_to_durable_manifest_is_not_gem_parent(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            copied_candidate = {
                "generation_id": 0,
                "finding_id": "scout-high",
                "variant_name": "scout_high",
                "metric_name": "mean_active_alpha_vs_benchmark_pct",
                "metric_value": 999.0,
                "metric_direction": "maximize",
                "frontier_entity_key": "variant::scout_high",
                "excluded_from_durable_frontier": True,
                "exclusion_reason": "preliminary_or_incomplete_evidence",
                "metrics": {"mean_active_alpha_vs_benchmark_pct": 999.0},
            }
            string_flag_candidate = {
                **copied_candidate,
                "finding_id": "scout-string-flag",
                "variant_name": "scout_string_flag",
                "frontier_entity_key": "variant::scout_string_flag",
                "excluded_from_durable_frontier": "true",
                "exclusion_reason": "",
            }
            manifest = {
                "lane_frontiers": {},
                "cumulative_top": [dict(copied_candidate), dict(string_flag_candidate)],
                "validation_candidates": {
                    "cumulative": [dict(copied_candidate), dict(string_flag_candidate)]
                },
            }
            frontier = _FakeFrontier(run_dir, manifest)
            mgr = GemsManager(run_dir=run_dir, task_spec=_task_with_gems(), frontier=frontier)

            self.assertEqual(mgr._select_gem_entries(manifest), [])

    def test_pending_reset_recovery_filters_validation_candidate_selected_entries(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            validation_candidate = {
                "generation_id": 0,
                "finding_id": "hypothesis-unmarked",
                "variant_name": "hypothesis_unmarked",
                "source_is_existing_gem": "false",
                "frontier_lane": "alpha_incubator",
                "lane_metric_name": "mean_active_alpha_vs_benchmark_pct",
                "lane_metric_value": 999.0,
                "metrics": {"mean_active_alpha_vs_benchmark_pct": 999.0},
                "excluded_from_durable_frontier": True,
                "exclusion_reason": "preliminary_or_incomplete_evidence",
            }
            frontier = _FakeFrontier(
                run_dir,
                {
                    "lane_frontiers": {},
                    "cumulative_top": [],
                    "validation_candidates": {"cumulative": [dict(validation_candidate)]},
                },
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=_task_with_gems(), frontier=frontier)
            mgr.save_state(
                {
                    "enabled": True,
                    "cycle_index": 0,
                    "cycle_start_generation": 0,
                    "reset_count": 0,
                    "gems": [],
                    "pending_reset": {
                        "status": "pending",
                        "reset_count": 1,
                        "cycle_index": 1,
                        "completed_gen_id": 0,
                        "next_absolute_generation": 1,
                        "signature_hash": "sig",
                        "reason": "bad_history",
                        "archive_dir": str(run_dir / "archive" / "bad_history"),
                        "selected_entries": [dict(validation_candidate)],
                    },
                }
            )

            result = mgr.recover_pending_reset(completed_gen_id=0)

            self.assertFalse(result.triggered)
            self.assertEqual(result.reason, "pending_reset_without_entries")
            state_after = json.loads((run_dir / "gems" / "gems_state.json").read_text())
            self.assertEqual(state_after.get("gems"), [])
            self.assertNotIn("pending_reset", state_after)

    def test_pending_reset_recovery_filters_gem_records_without_selected_entries(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            bad_record = {
                "gem_finding_id": "gem_bad_validation_candidate",
                "source_finding_id": "hypothesis-unmarked",
                "source_generation_id": 0,
                "variant_name": "hypothesis_unmarked",
                "frontier_lane": "alpha_incubator",
                "metric_name": "mean_active_alpha_vs_benchmark_pct",
                "metric_value": 999.0,
                "admission_metrics": {
                    "mean_active_alpha_vs_benchmark_pct": 999.0,
                    "primary_score": 999.0,
                },
            }
            frontier = _FakeFrontier(
                run_dir,
                {
                    "lane_frontiers": {},
                    "cumulative_top": [],
                    "validation_candidates": {
                        "cumulative": [
                            {
                                "generation_id": 0,
                                "finding_id": "hypothesis-unmarked",
                                "variant_name": "hypothesis_unmarked",
                                "metric_value": 999.0,
                            }
                        ]
                    },
                },
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=_task_with_gems(), frontier=frontier)
            mgr.save_state(
                {
                    "enabled": True,
                    "cycle_index": 0,
                    "cycle_start_generation": 0,
                    "reset_count": 0,
                    "gems": [],
                    "pending_reset": {
                        "status": "pending",
                        "reset_count": 1,
                        "cycle_index": 1,
                        "completed_gen_id": 0,
                        "next_absolute_generation": 1,
                        "signature_hash": "sig",
                        "reason": "bad_history",
                        "archive_dir": str(run_dir / "archive" / "bad_history"),
                        "selected_entries": [],
                        "gem_records": [bad_record],
                    },
                }
            )

            result = mgr.recover_pending_reset(completed_gen_id=0)

            self.assertFalse(result.triggered)
            self.assertEqual(result.reason, "pending_reset_without_entries")
            state_after = json.loads((run_dir / "gems" / "gems_state.json").read_text())
            self.assertEqual(state_after.get("gems"), [])
            self.assertNotIn("pending_reset", state_after)

    def test_pending_reset_recovery_filters_unknown_generation_selected_entries(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            unknown_generation_entry = {
                "finding_id": "unknown-gen",
                "variant_name": "unknown_gen",
                "frontier_lane": "alpha_incubator",
                "lane_metric_name": "mean_active_alpha_vs_benchmark_pct",
                "lane_metric_value": 999.0,
                "metrics": {"mean_active_alpha_vs_benchmark_pct": 999.0},
            }
            frontier = _FakeFrontier(
                run_dir,
                {"lane_frontiers": {}, "cumulative_top": [dict(unknown_generation_entry)]},
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=_task_with_gems(), frontier=frontier)
            mgr.save_state(
                {
                    "enabled": True,
                    "cycle_index": 0,
                    "cycle_start_generation": 0,
                    "reset_count": 0,
                    "gems": [],
                    "pending_reset": {
                        "status": "pending",
                        "reset_count": 1,
                        "cycle_index": 1,
                        "completed_gen_id": 0,
                        "next_absolute_generation": 1,
                        "signature_hash": "sig",
                        "reason": "bad_history",
                        "archive_dir": str(run_dir / "archive" / "bad_history"),
                        "selected_entries": [unknown_generation_entry],
                    },
                }
            )

            result = mgr.recover_pending_reset(completed_gen_id=0)

            self.assertFalse(result.triggered)
            self.assertEqual(result.reason, "pending_reset_without_entries")
            state_after = json.loads((run_dir / "gems" / "gems_state.json").read_text())
            self.assertEqual(state_after.get("gems"), [])
            self.assertNotIn("pending_reset", state_after)

    def test_gems_reset_preserves_validation_candidates_without_promoting_them(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "generations": {"0": [{"finding_id": "durable"}]},
                "cumulative_top": [{"finding_id": "durable"}],
                "lane_frontiers": {
                    "alpha_incubator": [
                        {
                            "generation_id": 0,
                            "finding_id": "durable",
                            "variant_name": "durable",
                            "frontier_lane": "alpha_incubator",
                            "metrics": {"mean_test_taskscore": 1.0},
                        }
                    ]
                },
                "validation_candidates": {
                    "generations": {
                        "0": [
                            {
                                "generation_id": 0,
                                "finding_id": "scout-high",
                                "variant_name": "scout_high",
                                "metric_value": 999.0,
                                "frontier_entity_key": "variant::scout_high",
                            }
                        ]
                    },
                    "cumulative": [
                        {
                            "generation_id": 0,
                            "finding_id": "scout-high",
                            "variant_name": "scout_high",
                            "metric_value": 999.0,
                            "frontier_entity_key": "variant::scout_high",
                        }
                    ],
                },
            }
            frontier = _FakeFrontier(run_dir, manifest)
            mgr = GemsManager(run_dir=run_dir, task_spec=_task_with_gems(), frontier=frontier)
            mgr._merge_gems_into_frontier_manifest(
                gems=[],
                state={
                    "cycle_index": 1,
                    "reset_count": 1,
                    "cycle_start_generation": 1,
                    "active_bottleneck_reports": [],
                },
            )

            updated = frontier.get_manifest()
            self.assertEqual(updated["generations"], {})
            self.assertEqual(updated["cumulative_top"], [])
            self.assertEqual(updated["lane_frontiers"], {})
            self.assertEqual(
                updated["validation_candidates"]["cumulative"][0]["finding_id"],
                "scout-high",
            )
            self.assertEqual(mgr._manifest_entries(updated), [])
            self.assertEqual(mgr._select_gem_entries(updated), [])

    def test_gems_reset_retires_validation_candidate_that_became_gem(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            validation_candidate = {
                "generation_id": 0,
                "finding_id": "scout-parent",
                "variant_name": "artifact_parent",
                "metric_value": 999.0,
                "frontier_entity_key": "variant::artifact_parent",
                "identity_aliases": ["scout-parent", "variant::artifact_parent"],
                "metrics": {
                    "child_variant_id": "artifact_parent",
                    "source_result_path": "results/artifact_parent.json",
                    "source_result_sha256": "artifact-parent-sha",
                },
            }
            manifest = {
                "validation_candidates": {
                    "generations": {"0": [dict(validation_candidate)]},
                    "cumulative": [dict(validation_candidate)],
                    "validator_identity_aliases_by_generation": {
                        "0": ["scout-parent", "variant::artifact_parent"]
                    },
                }
            }
            frontier = _FakeFrontier(run_dir, manifest)
            mgr = GemsManager(run_dir=run_dir, task_spec=_task_with_gems(), frontier=frontier)
            mgr._merge_gems_into_frontier_manifest(
                gems=[
                    {
                        "gem_finding_id": "gem-parent",
                        "variant_name": "artifact_parent",
                        "source_generation_id": 0,
                        "frontier_lane": "alpha_incubator",
                        "admission_metrics": {
                            "mean_test_taskscore": 10.0,
                            "complete_eval": True,
                            "child_variant_id": "artifact_parent",
                            "source_result_path": "results/artifact_parent.json",
                            "source_result_sha256": "artifact-parent-sha",
                        },
                    }
                ],
                state={
                    "cycle_index": 1,
                    "reset_count": 1,
                    "cycle_start_generation": 1,
                    "active_bottleneck_reports": [],
                },
            )

            updated = frontier.get_manifest()["validation_candidates"]

        self.assertEqual(updated["generations"]["0"], [])
        self.assertEqual(updated["cumulative"], [])
        self.assertNotIn("0", updated.get("validator_identity_aliases_by_generation", {}))

    def test_gems_reset_keeps_sibling_validation_candidate_with_shared_family_alias(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            validation_candidate = {
                "generation_id": 0,
                "finding_id": "scout-child-c005",
                "variant_name": "bridge_l1_eff_n_sweep",
                "metric_value": 9.0,
                "frontier_entity_key": "variant::bridge_l1_eff_n_sweep",
                "source_result_path": "results/c005/tiered_eval_summary.json",
                "identity_aliases": [
                    "bridge_l1_eff_n_sweep",
                    "results/c005/tiered_eval_summary.json",
                ],
            }
            manifest = {
                "validation_candidates": {
                    "generations": {"0": [dict(validation_candidate)]},
                    "cumulative": [dict(validation_candidate)],
                    "validator_identity_aliases_by_generation": {
                        "0": list(validation_candidate["identity_aliases"])
                    },
                }
            }
            frontier = _FakeFrontier(run_dir, manifest)
            mgr = GemsManager(run_dir=run_dir, task_spec=_task_with_gems(), frontier=frontier)
            mgr._merge_gems_into_frontier_manifest(
                gems=[
                    {
                        "gem_finding_id": "gem-child-c025",
                        "variant_name": "bridge_l1_eff_n_sweep",
                        "source_generation_id": 1,
                        "frontier_lane": "alpha_incubator",
                        "admission_metrics": {
                            "mean_test_taskscore": 10.0,
                            "complete_eval": True,
                            "frontier_entity_key": "variant::bridge_l1_eff_n_sweep",
                            "source_result_path": "results/c025/tiered_eval_summary.json",
                        },
                    }
                ],
                state={
                    "cycle_index": 1,
                    "reset_count": 1,
                    "cycle_start_generation": 1,
                    "active_bottleneck_reports": [],
                },
            )

            updated = frontier.get_manifest()["validation_candidates"]

        self.assertEqual(updated["cumulative"][0]["finding_id"], "scout-child-c005")

    def test_gems_reset_keeps_same_producer_validation_from_different_snapshot(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            validation_candidate = {
                "generation_id": 0,
                "finding_id": "preliminary-snapshot",
                "variant_name": "candidate",
                "metrics": {
                    "child_variant_id": "candidate",
                    "source_result_path": "results/candidate.json",
                    "source_result_sha256": "preliminary-sha",
                    "score": 1.0,
                },
            }
            path_only_candidate = {
                "generation_id": 0,
                "finding_id": "path-only-snapshot",
                "variant_name": "candidate",
                "metrics": {
                    "child_variant_id": "candidate",
                    "source_result_path": "results/candidate.json",
                    "score": 0.5,
                },
            }
            producerless_candidate = {
                "generation_id": 0,
                "finding_id": "producerless-snapshot",
                "variant_name": "candidate",
                "metrics": {
                    "source_result_path": "results/candidate.json",
                    "source_result_sha256": "complete-sha",
                    "score": 0.25,
                },
            }
            manifest = {
                "validation_candidates": {
                    "generations": {
                        "0": [
                            dict(validation_candidate),
                            dict(path_only_candidate),
                            dict(producerless_candidate),
                        ]
                    },
                    "cumulative": [
                        dict(validation_candidate),
                        dict(path_only_candidate),
                        dict(producerless_candidate),
                    ],
                }
            }
            frontier = _FakeFrontier(run_dir, manifest)
            mgr = GemsManager(run_dir=run_dir, task_spec=_task_with_gems(), frontier=frontier)
            mgr._merge_gems_into_frontier_manifest(
                gems=[
                    {
                        "gem_finding_id": "complete-snapshot",
                        "variant_name": "candidate",
                        "source_generation_id": 1,
                        "frontier_lane": "alpha_incubator",
                        "admission_metrics": {
                            "child_variant_id": "candidate",
                            "source_result_path": "results/candidate.json",
                            "source_result_sha256": "complete-sha",
                            "score": 2.0,
                            "scored_complete": True,
                        },
                    }
                ],
                state={
                    "cycle_index": 1,
                    "reset_count": 1,
                    "cycle_start_generation": 1,
                    "active_bottleneck_reports": [],
                },
            )

            updated = frontier.get_manifest()["validation_candidates"]

        self.assertEqual(
            {entry["finding_id"] for entry in updated["cumulative"]},
            {"preliminary-snapshot", "path-only-snapshot", "producerless-snapshot"},
        )

    def test_pending_reset_recovery_preserves_validation_candidates(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            durable = {
                "generation_id": 0,
                "finding_id": "durable",
                "variant_name": "durable",
                "frontier_lane": "alpha_incubator",
                "lane_metric_name": "mean_active_alpha_vs_benchmark_pct",
                "lane_metric_value": 1.0,
                "metrics": {
                    "mean_active_alpha_vs_benchmark_pct": 1.0,
                    "complete_eval": True,
                },
            }
            manifest = {
                "generations": {"0": [dict(durable)]},
                "cumulative_top": [dict(durable)],
                "lane_frontiers": {"alpha_incubator": [dict(durable)]},
                "validation_candidates": {
                    "cumulative": [
                        {
                            "generation_id": 0,
                            "finding_id": "scout-high",
                            "variant_name": "scout_high",
                            "metric_value": 999.0,
                            "frontier_entity_key": "variant::scout_high",
                            "excluded_from_durable_frontier": True,
                        }
                    ]
                },
            }
            frontier = _FakeFrontier(run_dir, manifest)
            mgr = GemsManager(run_dir=run_dir, task_spec=_task_with_gems(), frontier=frontier)
            mgr.save_state(
                {
                    "enabled": True,
                    "cycle_index": 0,
                    "cycle_start_generation": 0,
                    "reset_count": 0,
                    "gems": [],
                    "pending_reset": {
                        "status": "pending",
                        "reset_count": 1,
                        "cycle_index": 1,
                        "completed_gen_id": 0,
                        "next_absolute_generation": 1,
                        "signature_hash": "sig",
                        "reason": "test_recovery",
                        "archive_dir": str(run_dir / "archive" / "gems_cycle_1_recovery"),
                        "selected_entries": [dict(durable)],
                    },
                }
            )

            result = mgr.recover_pending_reset(completed_gen_id=0)

            self.assertTrue(result.triggered)
            updated = json.loads((run_dir / "frontier" / "frontier_manifest.json").read_text())
            self.assertEqual(
                updated["validation_candidates"]["cumulative"][0]["finding_id"],
                "scout-high",
            )
            state_after = json.loads((run_dir / "gems" / "gems_state.json").read_text())
            self.assertNotIn("pending_reset", state_after)
            self.assertNotIn(
                "scout-high",
                {gem.get("source_finding_id") for gem in state_after.get("gems", [])},
            )

    def test_periodic_gems_reset_ignores_frontier_rank_guard(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "alpha_incubator": [
                        {
                            "generation_id": 1,
                            "rank": 4,
                            "finding_id": "f_rank4",
                            "variant_name": "rank4_positive_alpha",
                            "frontier_lane": "alpha_incubator",
                            "lane_metric_name": "mean_active_alpha_vs_benchmark_pct",
                            "lane_metric_value": 2.0,
                            "metrics": {
                                "mean_active_alpha_vs_benchmark_pct": 2.0,
                                "complete_eval": True,
                            },
                        }
                    ]
                }
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(
                reset_interval_generations=2,
                max_gems_total=8,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            result = mgr.maybe_trigger_after_boundary(completed_gen_id=1)

            self.assertTrue(result.triggered)
            self.assertIn("periodic_reset_every_2_generations", result.reason)

    def test_top_rank_alpha_incubator_without_positive_metric_resets_on_period(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "alpha_incubator": [
                        {
                            "generation_id": 1,
                            "rank": 1,
                            "finding_id": "f_weak",
                            "variant_name": "weak_new_family",
                            "frontier_lane": "alpha_incubator",
                            "lane_metric_name": "mean_active_alpha_vs_benchmark_pct",
                            "lane_metric_value": -0.4,
                            "metrics": {
                                "mean_active_alpha_vs_benchmark_pct": -0.4,
                                "mechanism_family": "new_family_name",
                                "complete_eval": True,
                            },
                        }
                    ]
                }
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(reset_interval_generations=2, max_gems_total=8)
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            result = mgr.maybe_trigger_after_boundary(completed_gen_id=1)

            self.assertTrue(result.triggered)
            self.assertIn("periodic_reset_every_2_generations", result.reason)

    def test_new_mechanism_family_alone_does_not_affect_periodic_gems_reset(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "diagnostic_control": [
                        {
                            "generation_id": 1,
                            "rank": 1,
                            "finding_id": "f_new_family",
                            "variant_name": "novel_family_probe",
                            "frontier_lane": "diagnostic_control",
                            "metrics": {
                                "mechanism_family": "new_family_name",
                                "future_fitness": -10.0,
                                "complete_eval": True,
                            },
                        }
                    ]
                }
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(reset_interval_generations=2, max_gems_total=8)
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            result = mgr.maybe_trigger_after_boundary(completed_gen_id=1)

            self.assertTrue(result.triggered)
            self.assertIn("periodic_reset_every_2_generations", result.reason)

    def test_gems_reset_still_runs_when_global_cap_is_full(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "alpha_incubator": [
                        {
                            "generation_id": 1,
                            "rank": 1,
                            "finding_id": "f_new",
                            "variant_name": "new_but_cap_full",
                            "frontier_lane": "alpha_incubator",
                            "lane_metric_value": -1.0,
                            "metrics": {"strategy_family": "new"},
                        }
                    ]
                }
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(
                reset_interval_generations=2,
                max_gems_per_reset=4,
                max_gems_total=4,
                max_gems_per_family=4,
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            state = mgr.load_state()
            state["gems"] = [
                {
                    "gem_finding_id": f"gem_{i}",
                    "variant_name": f"alpha_{i}",
                    "frontier_lane": "alpha_incubator",
                    "strategy_family": f"family_{i}",
                    "source_generation_id": 0,
                    "evidence_stage": "full_T1",
                    "admission_metrics": {"mean_test_taskscore": float(i + 1)},
                }
                for i in range(4)
            ]
            payload = mgr._frontier_signature_payload(manifest)
            state["last_signature_hash"] = mgr._signature_hash(payload)
            mgr.save_state(state)

            result = mgr.maybe_trigger_after_boundary(completed_gen_id=1)
            updated = mgr.load_state()

            self.assertTrue(result.triggered)
            self.assertEqual(result.admitted_count, 0)
            self.assertEqual(updated["reset_count"], 1)
            self.assertEqual(len(updated["gems"]), 4)
            self.assertEqual(mgr.logical_generation(2), 0)

    def test_single_pi_prompt_includes_compact_gems_context_and_aist_metrics(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import PIAgent

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "description.md").write_text("x", encoding="utf-8")
            (run_dir / "task_spec.yaml").write_text(
                """
task_id: t
task_name: T
description_file: description.md
gems:
  enabled: true
  primary_metric_keys: [mean_test_taskscore, future_fitness]
  secondary_metric_keys: [mean_active_alpha_vs_benchmark_pct]
  performance_lanes: [confirmed_alpha, alpha, alpha_incubator]
  result_artifact_default_lane: alpha_incubator
  result_artifact_default_family: learned_alpha
""",
                encoding="utf-8",
            )
            gems_dir = run_dir / "gems"
            gems_dir.mkdir(parents=True)
            (gems_dir / "gems_state.json").write_text(
                json.dumps(
                    {
                        "cycle_index": 2,
                        "reset_count": 2,
                        "cycle_start_generation": 9,
                        "gems": [
                            {
                                "gem_finding_id": f"gem_{i}",
                                "variant_name": f"alpha_{i}",
                                "frontier_lane": "alpha_incubator",
                                "source_generation_id": 0,
                                "metric_name": "future_fitness",
                                "metric_value": i,
                                "bottleneck_target": "drawdown_regression" if i == 3 else "",
                                "evidence_stage": "full_T1" if i == 3 else "",
                                "tradeoff_class": (
                                    "high_return_drawdown_repair_target" if i == 3 else ""
                                ),
                                "primary_tradeoff": "return_vs_mdd" if i == 3 else "",
                                "next_step_intent": "repair_failure_mode" if i == 3 else "",
                                "parent_candidate": "parent_alpha" if i == 3 else "",
                                "parent_usage": "repair" if i == 3 else "",
                                "admission_metrics": {
                                    "mean_test_taskscore": 13.0 + i,
                                    "mean_active_alpha_vs_benchmark_pct": 17.0 + i,
                                    "complete_eval": True,
                                },
                            }
                            for i in range(16)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            agent = PIAgent(
                run_dir=run_dir,
                workspace=run_dir,
                cohort_size=5,
                model="dummy",
                max_runtime_minutes=1,
                local_mode=True,
            )
            gems_context = agent._load_gems_context()
            self.assertEqual(
                gems_context["entries"][3]["admission_metrics"]["mean_test_taskscore"],
                16.0,
            )
            prompt = agent._build_synthesis_prompt(
                completed_gen_id=8,
                findings=[],
                edges=[],
                frontier=[],
                prior_agenda=None,
                prior_agendas_summary=[],
                prior_findings_summary=[
                    {
                        "gen": 7,
                        "id": "f",
                        "type": "result",
                        "peer": "p",
                        "variant": "v",
                        "title": "t",
                        "metrics": agent._trim_prior_metrics(
                            {
                                "future_fitness": 1.2,
                                "mean_active_alpha_vs_benchmark_pct": 3.4,
                                "active_ir": 0.5,
                                "mean_active_share": 0.06,
                            }
                        ),
                    }
                ],
                gems_context=gems_context,
                agenda_output_path=run_dir / "agenda.yaml",
            )

            self.assertIn("Durable Gems context", prompt)
            self.assertIn("alpha_3", prompt)
            self.assertNotIn("alpha_4", prompt)
            self.assertNotIn("alpha_5", prompt)
            self.assertNotIn("alpha_15", prompt)
            self.assertIn("future_fitness", prompt)
            self.assertIn("mean_active_alpha_vs_benchmark_pct", prompt)
            self.assertIn("drawdown_regression", prompt)
            self.assertIn("high_return_drawdown_repair_target", prompt)
            self.assertIn("repair_failure_mode", prompt)
            self.assertIn("parent_alpha", prompt)

    def test_single_pi_prompt_uses_dynamic_cohort_size_for_aist(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import PIAgent

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            agent = PIAgent(
                run_dir=run_dir,
                workspace=run_dir,
                cohort_size=8,
                model="dummy",
                max_runtime_minutes=1,
                local_mode=True,
            )
            prompt = agent._build_synthesis_prompt(
                completed_gen_id=0,
                findings=[],
                edges=[],
                frontier=[],
                prior_agenda=None,
                prior_agendas_summary=[],
                prior_findings_summary=[],
                gems_context=agent._load_gems_context(),
                agenda_output_path=run_dir / "agenda.yaml",
            )

            self.assertIn("`8` peers", prompt)
            self.assertNotIn("5 peers", prompt)
            self.assertNotIn("across the 5 peers", prompt)
            self.assertIn("gen1_peer7", prompt)
            self.assertNotIn("gen1_peer8", prompt)

    def test_generation_boundary_holds_findings_sync_mutex_during_gems_reset(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import generation_boundary

        class FakeSync:
            def __init__(self):
                self._sync_mutex = threading.Lock()
                self.calls: list[str] = []

            def _sync_once_locked(self):
                self.calls.append("locked_sync")

        class FakeGems:
            enabled = True

            def __init__(self, sync: FakeSync):
                self.sync = sync
                self.saw_mutex_locked = False

            def maybe_trigger_after_boundary(self, *, completed_gen_id: int):
                self.saw_mutex_locked = self.sync._sync_mutex.locked()
                return SimpleNamespace(
                    triggered=True,
                    reset_count=1,
                    admitted_count=1,
                    archive_dir="/tmp/archive",
                )

        class FakeFrontier:
            def get_summary(self):
                return []

            def promote(self, gen_id, findings):
                return [{"id": "f"}]

        with tempfile.TemporaryDirectory() as tmp:
            sync = FakeSync()
            gems = FakeGems(sync)
            loop = SimpleNamespace(
                run_dir=Path(tmp),
                _strategy_for_gen=lambda gen_id: "explore",
                _collect_findings_for_generation=lambda gen_id: [],
                frontier=FakeFrontier(),
                task_spec=SimpleNamespace(
                    evaluation=SimpleNamespace(diversity_dimensions=[]),
                    research_memory=SimpleNamespace(enabled=False),
                    generation_policy=SimpleNamespace(max_generations=3),
                ),
                _graph_maintainer=None,
                _findings_sync=sync,
                gems=gems,
            )

            asyncio.run(
                generation_boundary.complete_generation_boundary(
                    loop,
                    gen_id=1,
                    pi_agent=object(),
                    pi_cfg=SimpleNamespace(strict=False),
                )
            )

            self.assertTrue(gems.saw_mutex_locked)
            self.assertEqual(sync.calls, ["locked_sync", "locked_sync"])
            marker = json.loads((Path(tmp) / "gen_1" / "generation_boundary.json").read_text())
            self.assertEqual(marker["pi_status"], "skipped_gems_reset")

    def test_pending_gems_resume_recovery_holds_findings_sync_mutex(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            generation_loop,
            resume_state,
        )

        class FakeSync:
            def __init__(self):
                self._sync_mutex = threading.Lock()
                self.calls: list[str] = []
                self.boundary = None
                self.sync_boundaries: list[object] = []

            def begin_boundary_evidence_cutoff(self, gen_id, cutoff, source_snapshot):
                self.boundary = (gen_id, cutoff, source_snapshot)

            def clear_boundary_evidence_cutoff(self, gen_id):
                if self.boundary is not None and self.boundary[0] == gen_id:
                    self.boundary = None

            def _sync_once_locked(self):
                self.calls.append("locked_sync")
                self.sync_boundaries.append(self.boundary)

        class FakeGraph:
            def __init__(self):
                self.calls = 0

            def sync_once_blocking(self, timeout: float):
                self.calls += 1
                return {"status": "ok", "timeout": timeout}

        class FakeGems:
            def __init__(self, sync: FakeSync):
                self.sync = sync
                self.saw_mutex_locked = False

            def recover_pending_reset(self, *, completed_gen_id: int):
                self.saw_mutex_locked = self.sync._sync_mutex.locked()
                return SimpleNamespace(
                    triggered=True,
                    reset_count=1,
                    admitted_count=2,
                    archive_dir="/tmp/archive",
                )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen_dir = run_dir / "gen_2"
            gen_dir.mkdir()
            (gen_dir / "generation_results.json").write_text("[]", encoding="utf-8")
            cutoff = datetime.now(UTC)
            source_snapshot = {"results/gen2_peer0/summary.json": "target:1:2"}
            self.assertTrue(
                resume_state.write_boundary_evidence_checkpoint(
                    run_dir,
                    gen_id=2,
                    cutoff=cutoff,
                    evidence_source_snapshot=source_snapshot,
                )
            )
            sync = FakeSync()
            graph = FakeGraph()
            gems = FakeGems(sync)
            loop = SimpleNamespace(
                run_dir=run_dir,
                _findings_sync=sync,
                _graph_maintainer=graph,
                gems=gems,
            )

            result = generation_loop._recover_pending_gems_reset_for_resume(
                loop,
                pending_gen=2,
            )

            self.assertTrue(result.triggered)
            self.assertTrue(gems.saw_mutex_locked)
            self.assertEqual(sync.calls, ["locked_sync", "locked_sync"])
            self.assertEqual(
                sync.sync_boundaries,
                [(2, cutoff, source_snapshot), (2, cutoff, source_snapshot)],
            )
            self.assertIsNone(sync.boundary)
            self.assertIsNone(loop._boundary_evidence_cutoff)
            self.assertEqual(graph.calls, 1)
            marker = json.loads((Path(tmp) / "gen_2" / "generation_boundary.json").read_text())
            self.assertEqual(marker["pi_status"], "skipped_gems_reset_recovered")
            self.assertEqual(marker["evidence_cutoff_at"], cutoff.isoformat())
            self.assertEqual(
                marker["evidence_source_snapshot_at_cutoff"],
                source_snapshot,
            )
            self.assertIsNone(resume_state.read_boundary_evidence_checkpoint(run_dir, 2))

    def test_generation_boundary_skips_gems_reset_on_last_generation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import generation_boundary

        class FakeGems:
            enabled = True
            called = False
            completed_gen_id = None

            def maybe_trigger_after_boundary(self, *, completed_gen_id: int):
                self.called = True
                self.completed_gen_id = completed_gen_id
                return SimpleNamespace(triggered=False)

        class FakeFrontier:
            def get_summary(self):
                return []

            def promote(self, gen_id, findings):
                return [{"id": "f"}]

        with tempfile.TemporaryDirectory() as tmp:
            loop = SimpleNamespace(
                run_dir=Path(tmp),
                _strategy_for_gen=lambda gen_id: "pi_directed",
                _collect_findings_for_generation=lambda gen_id: [],
                frontier=FakeFrontier(),
                task_spec=SimpleNamespace(
                    evaluation=SimpleNamespace(diversity_dimensions=[]),
                    research_memory=SimpleNamespace(enabled=False),
                    generation_policy=SimpleNamespace(max_generations=2),
                ),
                _graph_maintainer=None,
                _findings_sync=None,
                gems=FakeGems(),
            )

            asyncio.run(
                generation_boundary.complete_generation_boundary(
                    loop,
                    gen_id=1,
                    pi_agent=object(),
                    pi_cfg=SimpleNamespace(strict=False),
                )
            )

            marker = json.loads((Path(tmp) / "gen_1" / "generation_boundary.json").read_text())
            self.assertFalse(loop.gems.called)
            self.assertIsNone(loop.gems.completed_gen_id)
            self.assertEqual(marker["pi_status"], "skipped_last_generation")

    def test_generation_boundary_preserves_frontier_on_last_generation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import generation_boundary

        class FakeGems:
            enabled = True

            def maybe_trigger_after_boundary(self, *, completed_gen_id: int):
                return SimpleNamespace(
                    triggered=True,
                    reset_count=1,
                    admitted_count=2,
                    archive_dir="/tmp/gems_archive",
                )

        class FakeFrontier:
            def get_summary(self):
                return []

            def promote(self, gen_id, findings):
                return [{"id": "f"}]

        with tempfile.TemporaryDirectory() as tmp:
            loop = SimpleNamespace(
                run_dir=Path(tmp),
                _strategy_for_gen=lambda gen_id: "pi_directed",
                _collect_findings_for_generation=lambda gen_id: [],
                frontier=FakeFrontier(),
                task_spec=SimpleNamespace(
                    evaluation=SimpleNamespace(diversity_dimensions=[]),
                    research_memory=SimpleNamespace(enabled=False),
                    generation_policy=SimpleNamespace(max_generations=2),
                ),
                _graph_maintainer=None,
                _findings_sync=None,
                gems=FakeGems(),
            )

            asyncio.run(
                generation_boundary.complete_generation_boundary(
                    loop,
                    gen_id=1,
                    pi_agent=object(),
                    pi_cfg=SimpleNamespace(strict=False),
                )
            )

            marker = json.loads((Path(tmp) / "gen_1" / "generation_boundary.json").read_text())
            self.assertEqual(marker["pi_status"], "skipped_last_generation")
            self.assertIsNone(marker["error"])

    def test_surface_narrowing_trigger(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.exploration_bottleneck_detector import (
            ExplorationBottleneckDetector,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            shared.mkdir()
            for i in range(6):
                (shared / f"gen2_peer{i}.json").write_text(
                    json.dumps(
                        {
                            "id": f"f{i}",
                            "generation_id": 2,
                            "finding_type": "result",
                            "variant_name": f"surface_probe_{i}",
                            "title": "PPO objective probe" if i < 5 else "attention probe",
                            "metrics": {
                                "primary_metric_delta": -1.0,
                                "strategy_family": (
                                    "task_defined_surface_probe"
                                    if i < 5
                                    else "task_defined_alternate_probe"
                                ),
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            detector = ExplorationBottleneckDetector(run_dir=run_dir, mode="generic")
            report = detector.analyze(
                completed_gen_id=2,
                manifest={"lane_frontiers": {"performance": []}},
            )

            records = report["records"]
            self.assertTrue(any(r["gem_type"] == "surface_narrowing" for r in records))
            surface_record = next(r for r in records if r["gem_type"] == "surface_narrowing")
            self.assertEqual(surface_record["hard_constraints"], [])
            self.assertIn(
                "increase_underused_surface_probability",
                surface_record["soft_agenda_priors"],
            )

    def test_generic_surface_narrowing_uses_explicit_task_family_not_ml_text(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.exploration_bottleneck_detector import (
            ExplorationBottleneckDetector,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            shared.mkdir()
            for i in range(6):
                (shared / f"gen2_peer{i}.json").write_text(
                    json.dumps(
                        {
                            "id": f"text_only_{i}",
                            "generation_id": 2,
                            "finding_type": "result",
                            "variant_name": f"text_only_{i}",
                            "title": "PPO reward objective" if i < 5 else "attention probe",
                            "metrics": {"primary_metric_delta": -1.0},
                        }
                    ),
                    encoding="utf-8",
                )

            report = ExplorationBottleneckDetector(run_dir=run_dir, mode="generic").analyze(
                completed_gen_id=2,
                manifest={"lane_frontiers": {}},
            )

        self.assertEqual(report["metrics"]["top_mechanism_family"], "other")
        self.assertEqual(report["records"], [])

    def test_exploration_bottleneck_primitive_helpers_cover_classifier_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            exploration_bottleneck_detector as detector,
        )

        self.assertIsNone(detector._as_float(True))
        self.assertIsNone(detector._as_float(float("inf")))
        self.assertEqual(detector._as_float("3.5"), 3.5)
        self.assertIsNone(detector._as_float(object()))
        self.assertEqual(detector._metric({"score": 1}, "score"), 1)
        self.assertEqual(detector._generation_id({"gen": "2"}), 2)
        self.assertEqual(detector._generation_id({"gen": "bad", "source_generation_id": "4"}), 4)
        self.assertEqual(detector._generation_id({"peer_id": "gen7_peer3"}), 7)
        self.assertIsNone(detector._generation_id({"peer_id": "genbad_peer3"}))
        classifier = detector.ExplorationBottleneckDetector(run_dir=Path("."))
        self.assertEqual(
            classifier._family_for_mode({"title": "passive benchmark floor"}),
            "reference_or_floor",
        )
        self.assertEqual(
            classifier._family_for_mode({"title": "ablation control"}),
            "diagnostic_or_control",
        )
        self.assertEqual(classifier._family_for_mode({"title": "offline replay AWR"}), "other")
        self.assertEqual(classifier._family_for_mode({"title": "reward objective PPO"}), "other")
        self.assertEqual(classifier._family_for_mode({"title": "regime MoE gate"}), "other")
        self.assertEqual(classifier._family_for_mode({"title": "attention transformer"}), "other")
        self.assertEqual(classifier._family_for_mode({"title": "drawdown risk"}), "robustness")
        self.assertEqual(classifier._family_for_mode({"title": "self_supervised aux"}), "other")
        self.assertEqual(classifier._family_for_mode({"title": "unknown"}), "other")
        self.assertEqual(detector._frontier_entries({"lane_frontiers": {"alpha": "bad"}}), [])
        self.assertEqual(detector._entropy([]), 0.0)
        self.assertEqual(
            detector.ExplorationBottleneckDetector._combine_soft_priors(
                [
                    {
                        "soft_agenda_priors": {
                            "flag": True,
                            "weight": 0.1,
                            "label": "first",
                        }
                    },
                    {
                        "soft_agenda_priors": {
                            "flag": False,
                            "weight": 0.2,
                            "label": "second",
                        }
                    },
                    {"soft_agenda_priors": "bad"},
                ]
            ),
            {"flag": True, "weight": 0.3, "label": "second"},
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good.json"
            bad = root / "bad.json"
            good.write_text('{"ok": true}', encoding="utf-8")
            bad.write_text("{bad", encoding="utf-8")
            self.assertEqual(detector._read_json_files([bad, good]), [{"ok": True}])

    def test_generic_exploration_bottleneck_covers_surface_narrowing_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            exploration_bottleneck_detector as detector,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            shared.mkdir()
            shared.joinpath("bad.json").write_text("{bad", encoding="utf-8")
            for i in range(3):
                shared.joinpath(f"gen2_peer{i}.json").write_text(
                    json.dumps(
                        {
                            "id": f"f{i}",
                            "generation_id": 2,
                            "title": f"reward loss failed attempt {i}",
                            "metrics": {
                                "score_delta": -0.1,
                                "strategy_family": "learning_update",
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            analyzer = detector.ExplorationBottleneckDetector(
                run_dir=run_dir,
                mode="generic",
                performance_lanes={"candidate", "  ", "performance"},
            )
            self.assertEqual(
                analyzer._family_for_mode({"metrics": {"strategy_family": " Novel "}}), "novel"
            )
            self.assertEqual(
                analyzer._family_for_mode({"metrics": {"strategy_family": "unknown"}}), "other"
            )
            self.assertEqual(
                analyzer._family_for_mode({"title": "benchmark baseline floor"}),
                "reference_or_floor",
            )
            self.assertEqual(
                analyzer._family_for_mode({"title": "ablation control falsification"}),
                "diagnostic_or_control",
            )
            self.assertEqual(analyzer._family_for_mode({"title": "offline replay vtrace"}), "other")
            self.assertEqual(
                analyzer._family_for_mode({"title": "reward objective PPO loss"}), "other"
            )
            self.assertEqual(analyzer._family_for_mode({"title": "regime MoE gate"}), "other")
            self.assertEqual(
                analyzer._family_for_mode({"title": "attention transformer representation"}),
                "other",
            )
            self.assertEqual(
                analyzer._family_for_mode({"title": "risk robust stability"}),
                "robustness",
            )
            self.assertEqual(analyzer._family_for_mode({"title": "self_supervised aux"}), "other")

            report = analyzer.analyze(
                completed_gen_id=2,
                manifest={
                    "lane_frontiers": {
                        "candidate": [{"generation_id": 2, "title": "promoted candidate"}],
                        "performance": "bad",
                    }
                },
            )
            self.assertEqual(report["records"], [])
            self.assertEqual(report["metrics"]["confirmed_candidate_count"], 1)
            self.assertEqual(report["metrics"]["repeated_failed_family_count"], 1)

            trigger_records = analyzer._trigger_records(
                {
                    "detector_mode": "generic",
                    "confirmed_candidate_count": 0,
                    "top_mechanism_family": "learning_update",
                    "top_mechanism_family_share": 0.8,
                    "surface_entropy_low": True,
                    "repeated_failed_family_count": 3,
                }
            )
            self.assertEqual(trigger_records[0]["gem_type"], "surface_narrowing")
            self.assertEqual(trigger_records[0]["hard_constraints"], [])

    def test_surface_narrowing_trigger_preserves_candidates_softly(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.exploration_bottleneck_detector import (
            ExplorationBottleneckDetector,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_surface_narrowing_findings(run_dir / "shared_findings", generation_id=3)
            manifest = {"lane_frontiers": {}}
            report = ExplorationBottleneckDetector(run_dir=run_dir, mode="generic").analyze(
                completed_gen_id=3,
                manifest=manifest,
            )

            record = next(r for r in report["records"] if r["gem_type"] == "surface_narrowing")
            self.assertIn(
                "increase_underused_surface_probability",
                record["soft_agenda_priors"],
            )
            self.assertEqual(record["hard_constraints"], [])

    def test_no_trigger_when_l1_level2_evidence_exists(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.exploration_bottleneck_detector import (
            ExplorationBottleneckDetector,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            shared.mkdir()
            (shared / "gen4_peer0.json").write_text(
                json.dumps(
                    {
                        "id": "l1_mask",
                        "generation_id": 4,
                        "finding_type": "result",
                        "variant_name": "l1_mask_ablation",
                        "title": "L1 mask ablation changes behavior",
                        "metrics": {"l1_evidence_level": 2, "mean_effective_n": 10},
                    }
                ),
                encoding="utf-8",
            )

            report = ExplorationBottleneckDetector(run_dir=run_dir, mode="generic").analyze(
                completed_gen_id=4,
                manifest={"lane_frontiers": {}},
            )

            self.assertFalse(any(r["gem_type"] == "l1_opportunity_gap" for r in report["records"]))

    def test_empty_bottleneck_report_does_not_pollute_prompt_context(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            shared.mkdir()
            (shared / "gen0_peer0.json").write_text(
                json.dumps(
                    {
                        "id": "l1_mask",
                        "generation_id": 0,
                        "finding_type": "result",
                        "variant_name": "l1_mask_ablation",
                        "title": "L1 mask ablation changes behavior",
                        "metrics": {"l1_evidence_level": 2, "uses_l1_features": True},
                    }
                ),
                encoding="utf-8",
            )
            frontier = _FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []})
            mgr = GemsManager(run_dir=run_dir, task_spec=_task_with_gems(), frontier=frontier)

            result = mgr.maybe_trigger_after_boundary(completed_gen_id=0)
            state = mgr.load_state()

            self.assertFalse(result.triggered)
            self.assertEqual(state.get("bottleneck_history", []), [])
            self.assertEqual(state.get("latest_soft_agenda_priors", {}), {})
            self.assertEqual(mgr.prompt_context(absolute_gen_id=1)["bottleneck_reports"], [])

    def test_clean_detector_pass_clears_active_bottleneck_context_only(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager
        from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import PIAgent
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
            _digest_gems,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            shared.mkdir()
            frontier = _FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []})
            mgr = GemsManager(run_dir=run_dir, task_spec=_task_with_gems(), frontier=frontier)
            state = mgr.load_state()
            stale_report = {
                "completed_generation": 0,
                "records": [{"gem_type": "l1_opportunity_gap"}],
                "soft_agenda_priors": {"increase_l1_aware_contract_probability": 0.25},
            }
            state["bottleneck_history"] = [stale_report]
            state["active_bottleneck_reports"] = [stale_report]
            state["latest_soft_agenda_priors"] = {"increase_l1_aware_contract_probability": 0.25}
            mgr.save_state(state)

            (shared / "gen1_peer0.json").write_text(
                json.dumps(
                    {
                        "id": "l1_mask",
                        "generation_id": 1,
                        "finding_type": "result",
                        "variant_name": "l1_mask_ablation",
                        "title": "L1 mask ablation changes behavior",
                        "metrics": {"l1_evidence_level": 2, "uses_l1_features": True},
                    }
                ),
                encoding="utf-8",
            )

            result = mgr.maybe_trigger_after_boundary(completed_gen_id=1)
            state_after = mgr.load_state()

            self.assertFalse(result.triggered)
            self.assertEqual(state_after.get("active_bottleneck_reports", []), [])
            self.assertEqual(state_after.get("latest_soft_agenda_priors", {}), {})
            self.assertEqual(len(state_after.get("bottleneck_history", [])), 1)
            self.assertEqual(mgr.prompt_context(absolute_gen_id=2)["bottleneck_reports"], [])
            self.assertEqual(_digest_gems(run_dir)["bottleneck_reports"], [])
            agent = PIAgent(
                run_dir=run_dir,
                workspace=run_dir,
                cohort_size=8,
                model="dummy",
                max_runtime_minutes=1,
                local_mode=True,
            )
            self.assertEqual(agent._load_gems_context()["bottleneck_reports"], [])
            manifest_after = json.loads(
                (run_dir / "frontier" / "frontier_manifest.json").read_text()
            )
            self.assertEqual(manifest_after["gems"]["bottleneck_reports"], [])

    def test_task_specific_l1_terms_do_not_create_core_l1_metrics(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.exploration_bottleneck_detector import (
            ExplorationBottleneckDetector,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            shared.mkdir()
            for i in range(4):
                (shared / f"gen2_peer{i}.json").write_text(
                    json.dumps(
                        {
                            "id": f"non_l1_{i}",
                            "generation_id": 2,
                            "finding_type": "result",
                            "variant_name": f"non_l1_ablation_control_{i}",
                            "title": "non_l1_distinct_mechanism ablation control",
                            "metrics": {
                                "l1_utilization_mode": "non_l1_distinct_mechanism",
                                "uses_l1_features": False,
                                "mean_effective_n": 12,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            report = ExplorationBottleneckDetector(run_dir=run_dir, mode="generic").analyze(
                completed_gen_id=2,
                manifest={"lane_frontiers": {}},
            )

            self.assertFalse(any("l1" in key for key in report["metrics"]))
            self.assertFalse(
                any(r.get("gem_type") == "l1_opportunity_gap" for r in report["records"])
            )

    def test_spaced_task_specific_l1_text_does_not_create_core_l1_metrics(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.exploration_bottleneck_detector import (
            ExplorationBottleneckDetector,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            shared.mkdir()
            for i in range(4):
                (shared / f"gen2_peer{i}.json").write_text(
                    json.dumps(
                        {
                            "id": f"non_l1_spaced_{i}",
                            "generation_id": 2,
                            "finding_type": "result",
                            "variant_name": f"non_l1_probe_{i}",
                            "title": "non L1 ablation control for reward mechanics",
                            "metrics": {"mean_effective_n": 12},
                        }
                    ),
                    encoding="utf-8",
                )

            report = ExplorationBottleneckDetector(run_dir=run_dir, mode="generic").analyze(
                completed_gen_id=2,
                manifest={"lane_frontiers": {}},
            )

            self.assertFalse(any("l1" in key for key in report["metrics"]))
            self.assertFalse(
                any(r.get("gem_type") == "l1_opportunity_gap" for r in report["records"])
            )

    def test_non_l1_distinct_mode_does_not_create_l1_family_attractor(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.exploration_bottleneck_detector import (
            ExplorationBottleneckDetector,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            shared.mkdir()
            for i in range(4):
                (shared / f"gen2_peer{i}.json").write_text(
                    json.dumps(
                        {
                            "id": f"non_l1_{i}",
                            "generation_id": 2,
                            "finding_type": "result",
                            "variant_name": f"ppo_reward_probe_{i}",
                            "title": "distinct reward objective probe",
                            "metrics": {
                                "l1_utilization_mode": "non_l1_distinct_mechanism",
                                "uses_l1_features": False,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            report = ExplorationBottleneckDetector(run_dir=run_dir, mode="generic").analyze(
                completed_gen_id=2,
                manifest={"lane_frontiers": {}},
            )

            self.assertNotEqual(report["metrics"]["top_mechanism_family"], "l1_liquidity")

    def test_non_l1_distinct_spread_text_does_not_create_l1_family_attractor(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.exploration_bottleneck_detector import (
            ExplorationBottleneckDetector,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            shared.mkdir()
            for i in range(4):
                (shared / f"gen2_peer{i}.json").write_text(
                    json.dumps(
                        {
                            "id": f"non_l1_spread_{i}",
                            "generation_id": 2,
                            "finding_type": "result",
                            "variant_name": f"reward_spread_probe_{i}",
                            "title": "non L1 reward probe mentioning spread robustness",
                            "metrics": {
                                "l1_utilization_mode": "non_l1_distinct_mechanism",
                                "uses_l1_features": False,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            report = ExplorationBottleneckDetector(run_dir=run_dir, mode="generic").analyze(
                completed_gen_id=2,
                manifest={"lane_frontiers": {}},
            )

            self.assertNotEqual(report["metrics"]["top_mechanism_family"], "l1_liquidity")

    def test_old_confirmed_candidate_does_not_disable_recent_bottleneck_priors(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.exploration_bottleneck_detector import (
            ExplorationBottleneckDetector,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_surface_narrowing_findings(run_dir / "shared_findings", generation_id=5)

            report = ExplorationBottleneckDetector(
                run_dir=run_dir,
                mode="generic",
                performance_lanes={"confirmed"},
            ).analyze(
                completed_gen_id=5,
                manifest={
                    "lane_frontiers": {
                        "confirmed": [
                            {
                                "generation_id": 0,
                                "finding_id": "old_confirmed",
                                "variant_name": "old_confirmed_candidate",
                                "metrics": {"score": 10},
                            }
                        ]
                    }
                },
            )

            self.assertEqual(report["metrics"]["confirmed_candidate_count"], 0)
            self.assertEqual(report["metrics"]["total_confirmed_candidate_count"], 1)
            self.assertTrue(any(r["gem_type"] == "surface_narrowing" for r in report["records"]))

    def test_no_fixed_l1_quota_generated(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.exploration_bottleneck_detector import (
            ExplorationBottleneckDetector,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "shared_findings").mkdir()
            report = ExplorationBottleneckDetector(run_dir=run_dir, mode="generic").analyze(
                completed_gen_id=0,
                manifest={"lane_frontiers": {}},
            )

            for record in report["records"]:
                rendered = json.dumps(record)
                self.assertNotIn("peer_count", rendered)
                self.assertNotIn("fixed", rendered.lower())
                self.assertEqual(record["hard_constraints"], [])

    def test_soft_priors_persist_to_gems_prompt_context(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier = _FakeFrontier(run_dir, {"lane_frontiers": {}})
            task = _task_with_gems()
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)
            state = mgr.load_state()
            state["bottleneck_history"] = [
                {
                    "completed_generation": 2,
                    "records": [
                        {
                            "gem_type": "l1_opportunity_gap",
                            "soft_agenda_priors": {"increase_l1_aware_contract_probability": 0.25},
                            "hard_constraints": [],
                        }
                    ],
                    "soft_agenda_priors": {"increase_l1_aware_contract_probability": 0.25},
                }
            ]
            state["latest_soft_agenda_priors"] = {"increase_l1_aware_contract_probability": 0.25}
            mgr.save_state(state)

            context = mgr.prompt_context(absolute_gen_id=3)

            self.assertIn("bottleneck_reports", context)
            self.assertEqual(
                context["latest_soft_agenda_priors"]["increase_l1_aware_contract_probability"],
                0.25,
            )

    def test_gems_boundary_records_soft_priors_without_triggering_reset(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            _write_surface_narrowing_findings(shared, generation_id=0)
            frontier = _FakeFrontier(
                run_dir,
                {
                    "lane_frontiers": {},
                    "cumulative_top": [
                        {
                            "generation_id": 0,
                            "finding_id": "f0",
                            "variant_name": "surface_anchor",
                            "metric_name": "future_fitness",
                            "metric_value": 1.0,
                        }
                    ],
                },
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=_task_with_gems(), frontier=frontier)

            result = mgr.maybe_trigger_after_boundary(completed_gen_id=0)
            state = mgr.load_state()

            self.assertFalse(result.triggered)
            self.assertIn("bottleneck_history", state)
            self.assertTrue(state["bottleneck_history"])
            manifest_after = json.loads(
                (run_dir / "frontier" / "frontier_manifest.json").read_text()
            )
            self.assertIn("latest_soft_agenda_priors", manifest_after["gems"])
            self.assertFalse((run_dir / "archive").exists())

    def test_bottleneck_report_records_even_without_frontier_entries(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            _write_surface_narrowing_findings(shared, generation_id=0)
            frontier = _FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []})
            mgr = GemsManager(run_dir=run_dir, task_spec=_task_with_gems(), frontier=frontier)

            result = mgr.maybe_trigger_after_boundary(completed_gen_id=0)
            state = mgr.load_state()

            self.assertFalse(result.triggered)
            self.assertEqual(result.reason, "insufficient_frontier_entries")
            self.assertTrue(state["bottleneck_history"])
            self.assertIn("surface_narrowing", json.dumps(state["bottleneck_history"]))
            manifest_after = json.loads(
                (run_dir / "frontier" / "frontier_manifest.json").read_text()
            )
            self.assertIn("bottleneck_reports", manifest_after["gems"])

    def test_bottleneck_report_still_updates_after_max_gems_resets(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            _write_surface_narrowing_findings(shared, generation_id=4)
            frontier = _FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []})
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=_task_with_gems(max_resets=0),
                frontier=frontier,
            )

            result = mgr.maybe_trigger_after_boundary(completed_gen_id=4)
            state = mgr.load_state()

            self.assertFalse(result.triggered)
            self.assertEqual(result.reason, "max_resets_reached")
            self.assertTrue(state["bottleneck_history"])
            self.assertIn("surface_narrowing", json.dumps(state["bottleneck_history"]))
            manifest_after = json.loads(
                (run_dir / "frontier" / "frontier_manifest.json").read_text()
            )
            self.assertIn("bottleneck_reports", manifest_after["gems"])

    def test_persistent_bottleneck_records_soft_priors_without_triggering_reset(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            for gen in (0, 1):
                _write_surface_narrowing_findings(shared, generation_id=gen)
            manifest = {
                "cumulative_top": [
                    {
                        "finding_id": "old_anchor",
                        "variant_name": "old_anchor",
                        "generation_id": 0,
                        "metric_name": "future_fitness",
                        "metric_value": -1.0,
                        "metrics": {"frontier_lane": "alpha_incubator"},
                    }
                ],
                "lane_frontiers": {},
            }
            frontier = _FakeFrontier(run_dir, manifest)
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=_task_with_gems(reset_interval_generations=6),
                frontier=frontier,
            )

            first = mgr.maybe_trigger_after_boundary(completed_gen_id=0)
            self.assertFalse(first.triggered)
            repeated = mgr.maybe_trigger_after_boundary(completed_gen_id=0)
            self.assertFalse(repeated.triggered)
            frontier._manifest["cumulative_top"][0]["variant_name"] = "different_anchor"
            frontier._save_manifest()
            second = mgr.maybe_trigger_after_boundary(completed_gen_id=1)

            self.assertFalse(second.triggered)
            self.assertIn("periodic_reset_waiting", second.reason)
            state = json.loads((run_dir / "gems" / "gems_state.json").read_text())
            self.assertEqual(state["reset_count"], 0)
            self.assertTrue(state["bottleneck_history"])
            self.assertIn("latest_soft_agenda_priors", state)

    def test_failed_bottleneck_detector_does_not_reuse_stale_active_report(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import gems as gems_mod
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "cumulative_top": [
                    {
                        "finding_id": "anchor",
                        "variant_name": "anchor",
                        "generation_id": 0,
                        "metric_name": "future_fitness",
                        "metric_value": 1.0,
                        "metrics": {"frontier_lane": "alpha_incubator"},
                    }
                ],
                "lane_frontiers": {
                    "alpha_incubator": [
                        {
                            "finding_id": "anchor",
                            "variant_name": "anchor",
                            "generation_id": 0,
                            "lane_metric_name": "mean_active_alpha_vs_benchmark_pct",
                            "lane_metric_value": 1.0,
                            "metrics": {
                                "frontier_lane": "alpha_incubator",
                                "mean_active_alpha_vs_benchmark_pct": 1.0,
                            },
                        }
                    ]
                },
            }
            frontier = _FakeFrontier(run_dir, manifest)
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=_task_with_gems(reset_interval_generations=6),
                frontier=frontier,
            )
            state = mgr.load_state()
            state["active_bottleneck_reports"] = [
                {
                    "metrics": {"finding_count": 8, "evidence_item_count": 8},
                    "records": [{"gem_type": "l1_opportunity_gap"}],
                }
            ]
            state["last_bottleneck_signature"] = "l1_opportunity_gap"
            mgr.save_state(state)

            with patch.object(
                gems_mod.ExplorationBottleneckDetector,
                "analyze",
                side_effect=RuntimeError("detector failed"),
            ):
                result = mgr.maybe_trigger_after_boundary(completed_gen_id=1)

            self.assertFalse(result.triggered)
            state_after = mgr.load_state()
            self.assertEqual(state_after["active_bottleneck_reports"], [])
            self.assertEqual(state_after.get("latest_soft_agenda_priors"), {})

    def test_frontier_tool_exposes_gems_bottleneck_priors(self) -> None:
        from praxist.plugins.tools.frontier_tools.adapter import _handle_get_frontier

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "frontier"
            frontier_dir.mkdir()
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {},
                        "lane_frontiers": {},
                        "gems": {
                            "entries": [
                                {
                                    "gem_finding_id": "gem_r01_01_alpha",
                                    "variant_name": "alpha_gem",
                                    "frontier_lane": "alpha_incubator",
                                    "admission_metrics": {
                                        "mean_test_taskscore": 11.0,
                                        "mean_active_alpha_vs_benchmark_pct": 5.0,
                                        "complete_eval": True,
                                    },
                                }
                            ],
                            "bottleneck_reports": [
                                {
                                    "completed_generation": 0,
                                    "records": [{"gem_type": "l1_opportunity_gap"}],
                                }
                            ],
                            "latest_soft_agenda_priors": {
                                "increase_l1_aware_contract_probability": 0.25
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"FRONTIER_DIR": str(frontier_dir), "PRAXIST_FRONTIER_ALLOW_UNBOUNDED": "1"},
                clear=False,
            ):
                out = asyncio.run(_handle_get_frontier({"top_k": 5, "up_to_generation": -1}))
            payload = json.loads(out["content"][0]["text"])

            self.assertEqual(
                payload["gems"]["latest_soft_agenda_priors"][
                    "increase_l1_aware_contract_probability"
                ],
                0.25,
            )
            self.assertEqual(
                payload["gems"]["bottleneck_reports"][0]["records"][0]["gem_type"],
                "l1_opportunity_gap",
            )
            self.assertEqual(
                payload["gems"]["entries"][0]["admission_metrics"]["mean_test_taskscore"],
                11.0,
            )

    def test_frontier_tool_filters_future_gems_context_by_generation_cutoff(self) -> None:
        from praxist.plugins.tools.frontier_tools.adapter import _handle_get_frontier

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
                                    "variant_name": "past_candidate",
                                    "metric_value": 1.0,
                                    "evidence_stage": "full_T1",
                                    "mature_enough": True,
                                }
                            ],
                            "2": [
                                {
                                    "generation_id": 2,
                                    "variant_name": "future_candidate",
                                    "metric_value": 99.0,
                                    "evidence_stage": "full_T1",
                                }
                            ],
                        },
                        "lane_frontiers": {},
                        "gems": {
                            "entries": [
                                {
                                    "gem_finding_id": "gem_future",
                                    "variant_name": "future_gem",
                                    "source_generation_id": 2,
                                    "admission_metrics": {"score": 99.0, "complete_eval": True},
                                },
                                {
                                    "gem_finding_id": "gem_unknown",
                                    "variant_name": "unknown_generation_gem",
                                    "admission_metrics": {"score": 50.0},
                                },
                                {
                                    "gem_finding_id": "gem_past",
                                    "variant_name": "past_gem",
                                    "source_generation_id": 0,
                                    "admission_metrics": {"score": 1.0, "complete_eval": True},
                                },
                            ],
                            "bottleneck_reports": [
                                {
                                    "records": [{"gem_type": "unknown_generation_gap"}],
                                    "soft_agenda_priors": {"unknown_prior": 0.5},
                                },
                                {
                                    "completed_generation": 0,
                                    "records": [{"gem_type": "past_gap"}],
                                    "soft_agenda_priors": {"past_prior": 0.1},
                                },
                                {
                                    "completed_generation": 2,
                                    "records": [{"gem_type": "future_gap"}],
                                    "soft_agenda_priors": {"future_prior": 0.9},
                                },
                            ],
                            "latest_soft_agenda_priors": {"future_prior": 0.9},
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"FRONTIER_DIR": str(frontier_dir)}, clear=False):
                payload = json.loads(
                    asyncio.run(_handle_get_frontier({"top_k": 5, "up_to_generation": "0"}))[
                        "content"
                    ][0]["text"]
                )

        self.assertEqual(
            [entry["variant_name"] for entry in payload["entries"]], ["past_candidate"]
        )
        self.assertEqual(payload["gems"]["entries"], [])
        self.assertFalse(payload["gems"]["historical_entries_complete"])
        self.assertIsNone(payload["gems"]["cycle_index"])
        self.assertIsNone(payload["gems"]["reset_count"])
        self.assertIsNone(payload["gems"]["cycle_start_generation"])
        self.assertEqual(
            [report["records"][0]["gem_type"] for report in payload["gems"]["bottleneck_reports"]],
            ["past_gap"],
        )
        self.assertEqual(payload["gems"]["latest_soft_agenda_priors"], {"past_prior": 0.1})

    def test_peer_prompt_checklist_mentions_bottleneck_only_gems_context(self) -> None:
        from jinja2 import Environment, FileSystemLoader

        backend_dir = (
            Path(__file__).resolve().parents[2]
            / "praxist"
            / "plugins"
            / "workflow_stages"
            / "research_loop"
            / "backend"
        )
        prompt = (
            Environment(loader=FileSystemLoader(str(backend_dir)))
            .get_template("prompt_base.jinja2")
            .render(
                peer_id="gen1_peer0",
                gen_id=1,
                logical_gen_id=1,
                cohort_size=8,
                workspace_dir="/workspace",
                variants_dir="/workspace/variants",
                results_dir="/workspace/results",
                findings_dir="/workspace/shared_findings",
                notebook_path="/workspace/notebook.json",
                logs_dir="/workspace/logs",
                graph_session_context="",
                gems_context={
                    "enabled": True,
                    "cycle_index": 0,
                    "gems_count": 0,
                    "gems": [],
                    "bottleneck_reports": [
                        {
                            "completed_generation": 0,
                            "metrics": {"l1_contract_ratio": 0.0},
                            "soft_agenda_priors": {"increase_l1_aware_contract_probability": 0.25},
                            "records": [
                                {
                                    "gem_type": "l1_opportunity_gap",
                                    "severity": "medium",
                                    "evidence": {},
                                    "soft_agenda_priors": {},
                                    "hard_constraints": [],
                                }
                            ],
                        }
                    ],
                },
            )
        )

        self.assertIn("Read the Gems context below before proposing work", prompt)
        self.assertIn("Gems Soft Agenda Priors", prompt)

    def test_peer_prompt_marks_gem_seeded_baseline_mode(self) -> None:
        from jinja2 import Environment, FileSystemLoader

        backend_dir = (
            Path(__file__).resolve().parents[2]
            / "praxist"
            / "plugins"
            / "workflow_stages"
            / "research_loop"
            / "backend"
        )
        gem = {
            "gem_finding_id": "gem_1",
            "variant_name": "alpha_repair_parent",
            "frontier_lane": "alpha_incubator",
            "metric_name": "hist_return_pct",
            "metric_value": 12.3,
            "source_generation_id": 5,
            "source_finding_id": "finding_1",
            "gem_variant_ref": "variants/alpha_repair_parent",
            "finding_path": "shared_findings/gem_1.json",
        }
        prompt = (
            Environment(loader=FileSystemLoader(str(backend_dir)))
            .get_template("prompt_base.jinja2")
            .render(
                peer_id="gen6_peer0",
                gen_id=6,
                logical_gen_id=0,
                cohort_size=8,
                workspace_dir="/workspace",
                variants_dir="/workspace/variants",
                results_dir="/workspace/results",
                findings_dir="/workspace/shared_findings",
                notebook_path="/workspace/notebook.json",
                logs_dir="/workspace/logs",
                graph_session_context="",
                local_mode=True,
                gems_context={
                    "enabled": True,
                    "cycle_index": 1,
                    "gems_count": 1,
                    "gems": [gem],
                    "bottleneck_reports": [],
                    "gem_seeded_baseline_mode": True,
                    "primary_gem_anchor": gem,
                    "secondary_gem_anchor": {},
                    "gem_anchor_roster": [
                        {
                            "peer_index": 0,
                            "primary_variant_name": "alpha_repair_parent",
                            "secondary_variant_name": "",
                        }
                    ],
                    "baseline_code_policy": "Gems are implementation parents.",
                    "official_baseline_performance_policy": (
                        "Official baseline and benchmark records remain performance references."
                    ),
                },
            )
        )

        self.assertIn("Gem-Seeded Baseline Mode", prompt)
        self.assertIn("implementation baseline", prompt)
        self.assertIn("alpha_repair_parent", prompt)
        self.assertIn("performance references", prompt)

    def test_generation_prompt_gates_gem_seeded_restart_mode(self) -> None:
        from jinja2 import Environment, FileSystemLoader

        backend_dir = (
            Path(__file__).resolve().parents[2]
            / "praxist"
            / "plugins"
            / "workflow_stages"
            / "research_loop"
            / "backend"
        )
        template = Environment(loader=FileSystemLoader(str(backend_dir))).get_template(
            "prompt_generation.jinja2",
        )
        base_context = {
            "gen_id": 0,
            "logical_gen_id": 0,
            "research_agenda": None,
            "peer_id": "gen0_peer0",
            "gems_context": {
                "enabled": True,
                "cycle_index": 0,
                "gems_count": 1,
                "gems": [{"variant_name": "alpha_parent"}],
                "gem_seeded_baseline_mode": False,
            },
        }

        rendered_not_seeded = template.render(**base_context)
        rendered_seeded = template.render(
            **{
                **base_context,
                "gems_context": {
                    **base_context["gems_context"],
                    "gem_seeded_baseline_mode": True,
                },
            }
        )

        self.assertNotIn("Gem-Seeded Baseline Mode", rendered_not_seeded)
        self.assertNotIn("Gems Restart Mode", rendered_not_seeded)
        self.assertIn("Gem-Seeded Baseline Mode", rendered_seeded)
        self.assertIn("Gems Restart Mode", rendered_seeded)

    def test_entry_helper_contracts_cover_metric_and_evidence_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import gems

        malformed_extra = {"extra": "{not json", "finding_id": "F1"}
        self.assertEqual(gems._entry_extra(malformed_extra), {})
        self.assertEqual(gems._entry_key({"variant_name": "v1"}), "v1")
        self.assertEqual(gems._entry_field({"details": {"tier": "T1"}}, "tier"), "T1")
        self.assertEqual(gems._gem_identity_token("Alpha / Beta++"), "alpha_/_beta")
        self.assertEqual(gems._gem_explicit_entity_key("variant::Alpha One"), "variant:alpha_one")
        self.assertEqual(gems._gem_explicit_entity_key("bad::Alpha"), "")
        self.assertEqual(
            gems._gem_source_result_child_token("results/child/tiered_eval_summary.json"),
            ("results/child/tiered_eval_summary.json", "child"),
        )
        self.assertEqual(gems._variant_key({"variant_id": "Variant A"}), "variant:variant_a")
        self.assertEqual(gems._variant_key({"source_finding_id": "Finding A"}), "finding:finding a")

        entry = {
            "metric_name": "hist_return_pct",
            "metric_value": "12.5",
            "lane_metric_name": "mean_active_alpha_vs_benchmark_pct",
            "lane_metric_value": "3.25",
            "metrics": {
                "n_eval_cells": "29.0",
                "frontier_lane": "alpha_incubator",
                "tier_reached": "T2",
                "scored_complete": True,
                "strategy_family": "learned_alpha",
            },
            "admission_metrics": {"hist_return_pct": 1.0, "q25_active_alpha_vs_benchmark_pct": 2.5},
        }
        self.assertEqual(gems._entry_metrics(entry)["hist_return_pct"], 1.0)
        self.assertEqual(gems._metric_float(entry, "hist_return_pct"), 1.0)
        self.assertEqual(
            gems._metric_float(
                {"metric_name": "hist_return_pct", "metric_value": "12.5"},
                "hist_return_pct",
            ),
            12.5,
        )
        self.assertEqual(gems._metric_float(entry, "mean_active_alpha_vs_benchmark_pct"), 3.25)
        self.assertEqual(gems._metric_float({"x": "nan"}, "x", default=7.0), 7.0)
        self.assertEqual(gems._metric_int(entry, "n_eval_cells"), 29)
        self.assertEqual(gems._metric_int({"x": True}, "x", default=4), 4)
        self.assertTrue(gems._boolish_entry_field({"x": "yes"}, "x"))
        self.assertFalse(gems._boolish_entry_field({"x": 0}, "x"))
        self.assertIsNone(gems._boolish_entry_field({"x": "maybe"}, "x"))
        self.assertTrue(gems._any_boolish_entry_field_true({"a": "1"}, "a", "b"))
        self.assertTrue(gems._any_boolish_entry_field_false({"b": "no"}, "a", "b"))

        self.assertEqual(gems._infer_generation_id_from_text("gen_2 later gen-7"), 7)
        self.assertIsNone(gems._infer_generation_id_from_text("no generation"))
        self.assertEqual(gems._entry_eval_unit_count(entry), 29)
        self.assertEqual(gems._entry_tier_text(entry), "t2")
        status_texts = gems._entry_status_texts(
            {"final_status": "Scored Complete", "source_result_path": "results/v1/out.json"}
        )
        self.assertIn("scored_complete", status_texts)
        self.assertTrue(gems._entry_is_scout_or_partial({"scored_complete": "false"}))
        self.assertFalse(gems._entry_is_scout_or_partial({"score": 10.0, "n_eval_cells": 99}))
        self.assertFalse(
            gems._entry_is_scout_or_partial(
                {
                    "hist_return_pct": 10.0,
                    "n_eval_cells": 99,
                    "complete_eval": True,
                    "_gems_primary_metric_keys": ["hist_return_pct"],
                }
            )
        )
        self.assertTrue(
            gems._entry_is_scout_or_partial(
                {"hist_return_pct": 10.0, "n_eval_cells": 99, "complete_eval": "false"}
            )
        )
        self.assertTrue(
            gems._entry_is_scout_or_partial(
                {
                    "hist_return_pct": 10.0,
                    "n_eval_cells": 99,
                    "details": {"complete_eval": False},
                }
            )
        )
        self.assertTrue(
            gems._entry_is_scout_or_partial(
                {
                    "hist_return_pct": 10.0,
                    "n_eval_cells": 99,
                    "extra": {"partial_eval": True},
                }
            )
        )
        self.assertTrue(
            gems._entry_is_scout_or_partial(
                {
                    "hist_return_pct": 10.0,
                    "n_eval_cells": 99,
                    "status": "complete_eval=false",
                }
            )
        )
        self.assertTrue(
            gems._entry_is_scout_or_partial(
                {
                    "metrics": {
                        "hist_return_pct": 10.0,
                        "n_eval_cells": 99,
                        "status": "scored_complete",
                    },
                    "admission_metrics": {"status": "complete_eval=false"},
                }
            )
        )
        self.assertTrue(
            gems._entry_is_scout_or_partial(
                {
                    "variant_name": "full_name",
                    "hist_return_pct": 10.0,
                    "n_eval_cells": 99,
                    "evidence_stage": "preliminary",
                }
            )
        )
        self.assertTrue(
            gems._entry_is_scout_or_partial(
                {"hist_return_pct": 10.0, "n_eval_cells": 99, "status": "prelim"}
            )
        )
        self.assertFalse(
            gems._entry_is_scout_or_partial(
                {
                    "variant_name": "scout_to_mature",
                    "hist_return_pct": 10.0,
                    "n_eval_cells": 99,
                    "complete_eval": True,
                    "status": "uncapped",
                    "_gems_primary_metric_keys": ["hist_return_pct"],
                }
            )
        )
        self.assertFalse(
            gems._entry_is_scout_or_partial(
                {
                    "variant_name": "alpha_not_capped",
                    "hist_return_pct": 10.0,
                    "n_eval_cells": 99,
                    "complete_eval": True,
                    "status": "not_capped",
                    "_gems_primary_metric_keys": ["hist_return_pct"],
                }
            )
        )
        self.assertFalse(
            gems._entry_is_scout_or_partial(
                {
                    "variant_name": "not_failed_full",
                    "hist_return_pct": 10.0,
                    "n_eval_cells": 99,
                    "complete_eval": True,
                    "status": "not_failed",
                    "_gems_primary_metric_keys": ["hist_return_pct"],
                }
            )
        )
        self.assertFalse(
            gems._entry_is_scout_or_partial(
                {
                    "variant_name": "no_timeout_full",
                    "hist_return_pct": 10.0,
                    "n_eval_cells": 99,
                    "complete_eval": True,
                    "status": "no_timeout",
                    "_gems_primary_metric_keys": ["hist_return_pct"],
                }
            )
        )
        self.assertTrue(gems._entry_is_scout_or_partial({"variant_name": "smoke_probe"}))
        self.assertEqual(gems._entry_source_generation_id({"variant_name": "gen3_alpha"}), 3)
        self.assertTrue(
            gems._is_mature_evaluation_or_better(
                {
                    **entry,
                    "metrics": {
                        **entry["metrics"],
                        "frontier_lane": "confirmed",
                        "tier_reached": "T1",
                        "scored_complete": True,
                    },
                },
                min_mature_eval_units=29,
            )
        )
        self.assertFalse(
            gems._is_mature_evaluation_or_better(
                {"variant_name": "smoke_probe", "n_eval_cells": 99, "frontier_lane": "alpha"},
                min_mature_eval_units=29,
            )
        )
        self.assertFalse(
            gems._is_mature_evaluation_or_better(
                {
                    "variant_name": "full_name",
                    "hist_return_pct": 10.0,
                    "n_eval_cells": 99,
                    "frontier_lane": "alpha",
                    "metrics": {"complete_eval": False},
                },
                min_mature_eval_units=29,
            )
        )

        self.assertEqual(gems._evidence_rank({"tier": "forced_t3"}), 0)
        self.assertEqual(gems._evidence_rank({"tier": "scout"}), 1)
        self.assertEqual(gems._evidence_rank({"metrics": {"evidence_rank": "2"}}), 2)
        key = gems._gem_performance_key(
            {
                **entry,
                "mean_test_taskscore": 4.0,
                "validation_2026_active_alpha_pct": 1.25,
                "max_drawdown_pct": 8.0,
                "_gems_secondary_metric_keys": ["mean_active_alpha_vs_benchmark_pct"],
                "_gems_lower_tail_metric_keys": ["q25_active_alpha_vs_benchmark_pct"],
                "_gems_validation_metric_keys": ["validation_2026_active_alpha_pct"],
                "_gems_cost_metric_keys": ["max_drawdown_pct"],
            }
        )
        self.assertEqual(key, (12.5, 3.25, 2.5, 1.25, 2, -8.0))
        better_loss = gems._gem_performance_key(
            {
                "loss": 0.2,
                "_gems_primary_metric_keys": ["loss"],
                "_gems_metric_direction": "minimize",
            }
        )
        worse_loss = gems._gem_performance_key(
            {
                "loss": 0.8,
                "_gems_primary_metric_keys": ["loss"],
                "_gems_metric_direction": "minimize",
            }
        )
        self.assertGreater(better_loss, worse_loss)
        self.assertEqual(gems._performance_lane_priority("incubator"), 0)
        self.assertEqual(gems._performance_lane_priority("other"), 0)
        self.assertFalse(gems._is_performance_entry(entry))
        self.assertTrue(gems._is_performance_entry({"frontier_lane": "confirmed", "score": 1}))
        self.assertFalse(gems._is_performance_entry({"variant_name": "benchmark_floor"}))
        self.assertFalse(gems._is_performance_entry({"strategy_family": "diagnostic_control"}))
        self.assertEqual(gems._entry_lane(entry), "alpha_incubator")
        self.assertEqual(gems._entry_family({"mechanism_family": "film"}), "film")
        self.assertEqual(gems._entry_family({"variant_name": "bc40_parent"}), "")

    def test_manager_helper_contracts_cover_fallbacks_and_compaction_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import gems
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            disabled = GemsManager(
                run_dir=run_dir,
                task_spec=_task_with_gems(enabled=False),
                frontier=SimpleNamespace(),
            )
            self.assertEqual(disabled.active_gems(), [])

            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir()
            manifest_path = frontier_dir / "frontier_manifest.json"
            file_manifest = {
                "lane_frontiers": {
                    "alpha_incubator": [
                        {
                            "finding_id": "f-alpha",
                            "variant_name": "alpha",
                            "lane_metric_name": "score",
                            "lane_metric_value": 0.4,
                            "metrics": {"tier": "T1", "strategy_family": "fam-alpha"},
                        }
                    ],
                    "bad": "not-a-list",
                },
                "cumulative_top": [
                    {
                        "finding_id": "f-beta",
                        "variant_name": "beta",
                        "metric_value": float("inf"),
                        "metrics": {"tier_reached": "T1"},
                    }
                ],
            }
            manifest_path.write_text(json.dumps(file_manifest), encoding="utf-8")
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=_task_with_gems(
                    selection_policy="frontier_lane_balanced",
                    max_gems_total=3,
                    max_gems_per_family=1,
                    include_lanes=["alpha_incubator", "diagnostic_controls"],
                ),
                frontier=SimpleNamespace(),
            )

            self.assertEqual(
                mgr.frontier_manifest()["lane_frontiers"], file_manifest["lane_frontiers"]
            )
            signature = mgr._frontier_signature_payload(mgr.frontier_manifest())
            self.assertEqual([item["variant_name"] for item in signature], ["alpha"])
            entries = mgr._manifest_entries(mgr.frontier_manifest())
            self.assertEqual(entries[0]["promoted_for_lane"], "alpha_incubator")
            self.assertEqual(mgr._signature_hash(signature), mgr._signature_hash(list(signature)))

            raw_gems = [
                "bad",
                {
                    "variant_name": "alpha",
                    "source_generation_id": 1,
                    "frontier_lane": "confirmed_alpha",
                    "metrics": {
                        "frontier_entity_key": "variant::alpha",
                        "strategy_family": "fam-a",
                        "score": 0.9,
                        "tier": "T1",
                        "n_eval_cells": 30,
                        "complete_eval": True,
                    },
                },
                {
                    "variant_name": "alpha-dup",
                    "source_generation_id": 1,
                    "frontier_lane": "confirmed_alpha",
                    "metrics": {
                        "frontier_entity_key": "variant::alpha",
                        "strategy_family": "fam-a",
                        "score": 0.8,
                    },
                },
                {
                    "variant_name": "beta",
                    "source_generation_id": 1,
                    "frontier_lane": "benchmark_shadow",
                    "metrics": {"strategy_family": "fam-a", "score": 0.7},
                },
                {
                    "variant_name": "gamma",
                    "source_generation_id": 1,
                    "frontier_lane": "diagnostic_controls",
                    "metrics": {
                        "strategy_family": "fam-c",
                        "score": 0.1,
                        "complete_eval": True,
                    },
                },
            ]
            compact = mgr._compact_gems(raw_gems, cap=3, sort_by_performance=True)
            self.assertEqual([item["variant_name"] for item in compact], ["alpha", "gamma"])
            self.assertEqual(
                mgr._ordered_lanes_for_gems(compact),
                ["diagnostic_controls", "confirmed_alpha"],
            )
            lane_reserved = mgr._compact_gems(
                raw_gems,
                cap=2,
                sort_by_performance=True,
                preserve_lane_reserves=True,
            )
            self.assertIn("gamma", {item["variant_name"] for item in lane_reserved})

            mature_evidence = GemsManager(
                run_dir=run_dir,
                task_spec=_task_with_gems(
                    selection_policy="mature_evidence_top_k",
                    max_gems_per_reset=0,
                    max_gems_total=4,
                    include_lanes=["alpha_incubator"],
                    performance_lanes=[],
                ),
                frontier=SimpleNamespace(),
            )
            self.assertEqual(mature_evidence._select_mature_evidence_topk_entries({}), [])
            self.assertIsNone(mature_evidence._state_source_generation_limit({}))
            generic_defaults = GemsManager(
                run_dir=run_dir,
                task_spec=_task_with_gems(
                    include_lanes=["alpha_incubator"],
                    performance_lanes=[],
                    control_lanes=[],
                    result_artifact_default_family="learned_candidate",
                    bottleneck_detector_mode="generic",
                ),
                frontier=SimpleNamespace(),
            )
            self.assertIn("performance", generic_defaults._performance_lanes())
            self.assertNotIn("alpha_incubator", generic_defaults._performance_lanes())
            self.assertIn("diagnostic", generic_defaults._control_lanes())
            self.assertNotIn("control", generic_defaults._control_lanes())
            self.assertNotIn("process", generic_defaults._control_lanes())
            self.assertNotIn("diagnostic_control", generic_defaults._control_lanes())
            self.assertFalse(
                gems._entry_is_recoverable_legacy_or_control_gem_source(
                    {
                        "variant_name": "controller_candidate",
                        "strategy_family": "process_optimizer",
                        "score": 1.0,
                    }
                )
            )
            self.assertNotIn("alpha_incubator", mature_evidence._performance_lanes())
            custom_family = GemsManager(
                run_dir=run_dir,
                task_spec=_task_with_gems(
                    performance_lanes=[],
                    result_artifact_default_family="custom_family",
                ),
                frontier=SimpleNamespace(),
            )
            self.assertTrue(
                custom_family._is_task_performance_entry(
                    {"strategy_family": "custom_family", "score": 1.0}
                )
            )
            self.assertTrue(
                custom_family._is_task_performance_entry(
                    {
                        "frontier_lane": "performance",
                        "variant_name": "baseline_reference_candidate",
                        "strategy_family": "learned_candidate",
                        "score": 1.0,
                    }
                )
            )
            self.assertTrue(
                custom_family._is_task_performance_entry(
                    {
                        "frontier_lane": "performance",
                        "variant_name": "benchmark_floor_anchor",
                        "strategy_family": "learned_candidate",
                        "score": 99.0,
                    }
                )
            )
            self.assertFalse(
                custom_family._is_task_performance_entry(
                    {
                        "frontier_lane": "performance",
                        "variant_name": "explicit_control_anchor",
                        "strategy_family": "benchmark_floor",
                        "score": 99.0,
                    }
                )
            )
            self.assertFalse(
                custom_family._is_task_performance_entry(
                    {"strategy_family": "other_family", "score": 1.0}
                )
            )
            self.assertEqual(
                mature_evidence._state_source_generation_limit({"source_generation_limit": "2"}),
                2,
            )
            self.assertIsNone(
                mature_evidence._state_source_generation_limit(
                    {
                        "source_generation_limit": "bad",
                        "cycle_start_generation": "bad",
                    }
                )
            )
            self.assertEqual(
                mature_evidence._state_source_generation_limit(
                    {
                        "cycle_start_generation": 5,
                        "reset_events": [
                            "bad",
                            {
                                "next_absolute_generation": None,
                                "completed_gen_id": 3,
                            },
                            {
                                "next_absolute_generation": "bad",
                                "completed_gen_id": 3,
                            },
                            {
                                "next_absolute_generation": 5,
                                "completed_gen_id": 4,
                            },
                        ],
                    }
                ),
                4,
            )
            self.assertIsNone(mature_evidence._operator_pruned_restart_cutoff({}, 5))
            self.assertIsNone(
                mature_evidence._operator_pruned_restart_cutoff({"reset_events": ["bad"]}, 5)
            )
            self.assertIsNone(
                mature_evidence._operator_pruned_restart_cutoff(
                    {
                        "reset_events": [
                            {
                                "next_absolute_generation": 5,
                                "completed_gen_id": 4,
                                "operator_pruned_restart_generation": "bad",
                            },
                            {
                                "next_absolute_generation": 5,
                                "completed_gen_id": 4,
                                "operator_pruned_restart_generation": 5,
                                "committed": "false",
                            },
                        ]
                    },
                    5,
                )
            )
            self.assertEqual(
                mature_evidence._operator_pruned_restart_cutoff(
                    {
                        "reset_events": [
                            {
                                "next_absolute_generation": 5,
                                "completed_gen_id": 4,
                                "operator_pruned_restart_generation": 5,
                            }
                        ]
                    },
                    5,
                ),
                4,
            )
            self.assertEqual(
                mature_evidence._gem_lane_quotas(ordered_lanes=["a", "b"], cap=0),
                {"a": 0, "b": 0},
            )
            mature_evidence.task_spec.evaluation = SimpleNamespace(
                frontier_lanes=[
                    "bad",
                    {"name": "", "k": 2},
                    {"name": "diagnostic_control", "k": "bad", "parent_eligible": False},
                    {"name": "performance", "k": 3, "parent_eligible": True},
                    {"name": "extra_control", "k": 3, "parent_eligible": False},
                ]
            )
            generic_defaults.task_spec.evaluation = mature_evidence.task_spec.evaluation
            self.assertEqual(
                mature_evidence._gem_lane_quotas(
                    ordered_lanes=["diagnostic_control", "performance", "alpha_incubator"],
                    cap=2,
                ),
                {"diagnostic_control": 1, "performance": 1, "alpha_incubator": 0},
            )
            self.assertEqual(
                generic_defaults._gem_lane_quotas(
                    ordered_lanes=["diagnostic_control", "extra_control"],
                    cap=3,
                ),
                {"diagnostic_control": 1, "extra_control": 2},
            )
            fallback_selected = generic_defaults._select_gem_entries(
                {
                    "cumulative_top": [
                        "bad",
                        {
                            "variant_name": "fallback_a",
                            "source_generation_id": 1,
                            "frontier_lane": "performance",
                            "score": 0.6,
                            "strategy_family": "candidate",
                            "complete_eval": True,
                        },
                        {
                            "variant_name": "fallback_b",
                            "source_generation_id": 1,
                            "frontier_lane": "performance",
                            "score": 0.8,
                            "strategy_family": "candidate",
                            "complete_eval": True,
                        },
                    ]
                },
                completed_gen_id=1,
            )
            self.assertEqual(
                {entry["variant_name"] for entry in fallback_selected},
                {"fallback_a", "fallback_b"},
            )

            results_dir = run_dir / "results"
            bad_variant_dir = results_dir / "bad"
            bad_variant_dir.mkdir(parents=True)
            (bad_variant_dir / "tiered_eval_summary.json").write_text("{", encoding="utf-8")
            list_variant_dir = results_dir / "list"
            list_variant_dir.mkdir(parents=True)
            (list_variant_dir / "tiered_eval_summary.json").write_text("[]", encoding="utf-8")
            good_variant_dir = results_dir / "good_gen3"
            good_variant_dir.mkdir(parents=True)
            (good_variant_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "reported",
                        "scored_complete": True,
                        "evaluation_units": 30,
                        "generation_id": 3,
                        "current_aggregate": {
                            "score": 0.8,
                            "frontier_lane": "confirmed_alpha",
                        },
                    }
                ),
                encoding="utf-8",
            )
            candidates = mature_evidence._result_artifact_gem_candidates(max_generation_id=3)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["variant_name"], "good_gen3")
            self.assertEqual(candidates[0]["metrics"]["reported_variant_name"], "reported")
            self.assertEqual(
                mature_evidence._result_artifact_gem_candidates(max_generation_id=2),
                [],
            )
            boundaryless_dir = results_dir / "boundaryless_candidate"
            boundaryless_dir.mkdir(parents=True)
            (boundaryless_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "boundaryless_candidate",
                        "evaluation_units": 30,
                        "scored_complete": True,
                        "current_aggregate": {
                            "score": 0.7,
                            "scored_complete": True,
                            "frontier_lane": "confirmed_alpha",
                        },
                    }
                ),
                encoding="utf-8",
            )
            fallback_candidates = mature_evidence._result_artifact_gem_candidates(
                max_generation_id=3
            )
            boundaryless = next(
                c for c in fallback_candidates if c["variant_name"] == "boundaryless_candidate"
            )
            self.assertEqual(boundaryless["generation_id"], 3)
            self.assertEqual(boundaryless["metrics"]["source_generation_id"], 3)
            self.assertEqual(
                boundaryless["metrics"]["source_generation_inference"],
                "boundary_fallback",
            )
            self.assertTrue(boundaryless["metrics"]["source_generation_low_confidence"])

            manifest_path.write_text("{", encoding="utf-8")
            self.assertEqual(mgr.frontier_manifest(), {})

        self.assertTrue(gems.utc_stamp().endswith("Z"))

    def test_gems_manager_records_bottleneck_report_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier = _FakeFrontier(run_dir, {"lane_frontiers": {"performance": []}})
            disabled = GemsManager(
                run_dir=run_dir,
                task_spec=_task_with_gems(bottleneck_detector_mode="disabled"),
                frontier=frontier,
            )
            disabled_state = {"cycle_index": 1, "reset_count": 0, "cycle_start_generation": 0}
            disabled._record_bottleneck_report(
                state=disabled_state,
                completed_gen_id=1,
                manifest=frontier.get_manifest(),
            )
            self.assertEqual(disabled_state["latest_soft_agenda_priors"], {})
            self.assertEqual(frontier.get_manifest()["gems"]["bottleneck_reports"], [])

            failing = GemsManager(
                run_dir=run_dir,
                task_spec=_task_with_gems(bottleneck_detector_mode="generic"),
                frontier=frontier,
            )
            failing_state = {"cycle_index": 1, "reset_count": 0, "cycle_start_generation": 0}
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.gems.ExplorationBottleneckDetector",
                side_effect=RuntimeError("detector"),
            ):
                failing._record_bottleneck_report(
                    state=failing_state,
                    completed_gen_id=1,
                    manifest=frontier.get_manifest(),
                )
            self.assertEqual(failing_state["active_bottleneck_reports"], [])

            empty_state = {"cycle_index": 1, "reset_count": 0, "cycle_start_generation": 0}

            class EmptyDetector:
                def __init__(self, **_kwargs) -> None:
                    pass

                def analyze(self, **_kwargs) -> dict:
                    return {"records": [], "soft_agenda_priors": {}}

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.gems.ExplorationBottleneckDetector",
                EmptyDetector,
            ):
                failing._record_bottleneck_report(
                    state=empty_state,
                    completed_gen_id=1,
                    manifest=frontier.get_manifest(),
                )
            self.assertEqual(empty_state["latest_soft_agenda_priors"], {})

            reported_state = {"cycle_index": 1, "reset_count": 0, "cycle_start_generation": 0}

            class ReportingDetector:
                def __init__(self, **_kwargs) -> None:
                    pass

                def analyze(self, **_kwargs) -> dict:
                    return {
                        "records": [{"gem_type": "surface_narrowing"}],
                        "soft_agenda_priors": {"increase_underused_surface_probability": 0.2},
                    }

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.gems.ExplorationBottleneckDetector",
                ReportingDetector,
            ):
                failing._record_bottleneck_report(
                    state=reported_state,
                    completed_gen_id=1,
                    manifest=frontier.get_manifest(),
                )
            self.assertEqual(len(reported_state["active_bottleneck_reports"]), 1)
            self.assertEqual(
                frontier.get_manifest()["gems"]["latest_soft_agenda_priors"],
                {"increase_underused_surface_probability": 0.2},
            )

    def test_non_mature_evidence_gem_selection_rejects_low_cell_unproven_rows(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "alpha_incubator": [
                        {
                            "finding_id": "unproven_high",
                            "variant_name": "unproven_high",
                            "source_generation_id": 1,
                            "frontier_lane": "alpha_incubator",
                            "metrics": {
                                "strategy_family": "learned_alpha",
                                "mean_test_taskscore": 99.0,
                                "n_eval_cells": 3,
                            },
                        },
                        {
                            "finding_id": "complete_low",
                            "variant_name": "complete_low",
                            "source_generation_id": 1,
                            "frontier_lane": "alpha_incubator",
                            "metrics": {
                                "strategy_family": "learned_alpha",
                                "mean_test_taskscore": 1.0,
                                "complete_eval": True,
                                "n_eval_cells": 3,
                            },
                        },
                    ]
                }
            }
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=_task_with_gems(
                    selection_policy="frontier_lane_balanced",
                    include_lanes=["alpha_incubator"],
                    max_gems_per_reset=2,
                    max_gems_total=2,
                ),
                frontier=_FakeFrontier(run_dir, manifest),
            )

            selected = mgr._select_gem_entries(manifest, completed_gen_id=1)

        self.assertEqual([entry["variant_name"] for entry in selected], ["complete_low"])

    def test_gems_selection_compaction_and_trigger_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import gems
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        self.assertEqual(gems._entry_key({"variant_name": "v"}), "v")
        self.assertEqual(gems._entry_field({"details": {"x": 1}}, "x"), 1)
        self.assertEqual(gems._gem_identity_token(" Variant::A/B "), "variant_a/b")
        self.assertEqual(
            gems._gem_source_result_child_token("results/Child/tiered_eval_summary.json"),
            ("results/child/tiered_eval_summary.json", "child"),
        )
        self.assertEqual(gems._variant_key({"candidate_entity_key": "finding::F1"}), "finding:f1")
        self.assertEqual(gems._variant_key({"child_id": "Child A"}), "variant:child_a")
        self.assertEqual(gems._variant_key({"source_finding_id": "S1"}), "finding:s1")
        self.assertTrue(gems._any_boolish_entry_field_true({"x": "y"}, "x"))
        self.assertTrue(gems._any_boolish_entry_field_false({"x": "n"}, "x"))
        self.assertIsNone(gems._infer_generation_id_from_text("no generation here"))
        self.assertEqual(gems._entry_eval_unit_count({"admission_metrics": {"n_cells": "7"}}), 7)
        self.assertIn("capped", gems._entry_status_texts({"final_status": "capped"}))
        self.assertTrue(gems._entry_is_scout_or_partial({"scout_only": True}))
        self.assertTrue(
            gems._is_performance_entry(
                {"variant_name": "alpha", "frontier_lane": "confirmed", "mean_test_taskscore": 1}
            )
        )
        self.assertFalse(gems._is_performance_entry({"variant_name": "diagnostic_control"}))

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "diagnostic_control": [
                        {
                            "finding_id": "diag",
                            "variant_name": "diag",
                            "source_generation_id": 1,
                            "frontier_lane": "diagnostic_control",
                            "metrics": {
                                "diagnostic_value": 10,
                                "score": 0.1,
                                "complete_eval": True,
                            },
                        }
                    ],
                    "alpha_incubator": [
                        {
                            "finding_id": "perf_a",
                            "variant_name": "offpolicy_replay_a",
                            "source_generation_id": 1,
                            "frontier_lane": "alpha_incubator",
                            "metrics": {
                                "strategy_family": "learned_alpha",
                                "tier": "T1",
                                "scored_cell_count": 29,
                                "mean_test_taskscore": 5,
                                "mean_active_alpha_vs_benchmark_pct": 6,
                            },
                        },
                        {
                            "finding_id": "perf_b",
                            "variant_name": "offpolicy_replay_b",
                            "source_generation_id": 1,
                            "frontier_lane": "alpha_incubator",
                            "metrics": {
                                "strategy_family": "learned_alpha",
                                "tier": "T1",
                                "scored_cell_count": 29,
                                "mean_test_taskscore": 4,
                                "mean_active_alpha_vs_benchmark_pct": 5,
                            },
                        },
                    ],
                },
                "cumulative_top": [{"finding_id": "cum", "variant_name": "cum"}],
            }
            frontier = _FakeFrontier(run_dir, manifest)
            task = _task_with_gems(
                max_gems_per_reset=2,
                max_gems_total=2,
                max_gems_per_family=1,
                include_lanes=["diagnostic_control", "alpha_incubator"],
            )
            task.evaluation = SimpleNamespace(
                maturity_policy={
                    "complete_stage_labels": ["T1", "T2", "T3", "full_T1"],
                    "preliminary_stage_labels": ["smoke", "scout", "partial"],
                },
                frontier_lanes=[
                    {"name": "diagnostic_control", "k": "bad"},
                    {"name": "alpha_incubator", "k": 3},
                    "bad",
                    {"name": ""},
                ],
            )
            mgr = GemsManager(run_dir=run_dir, task_spec=task, frontier=frontier)

            self.assertEqual(mgr.logical_generation(4), 4)
            mgr.save_state(
                {"enabled": True, "cycle_start_generation": 2, "cycle_index": 1, "gems": []}
            )
            self.assertEqual(mgr.logical_generation(4), 2)
            (mgr.state_path).write_text("{", encoding="utf-8")
            self.assertEqual(mgr.load_state()["cycle_index"], 0)
            mgr.save_state(
                {"enabled": True, "cycle_start_generation": 2, "cycle_index": 1, "gems": []}
            )

            payload = mgr._frontier_signature_payload(manifest)
            self.assertEqual(
                [item["lane"] for item in payload],
                ["alpha_incubator", "alpha_incubator", "diagnostic_control"],
            )
            self.assertEqual(len(mgr._signature_hash(payload)), 16)
            self.assertEqual(mgr._max_gems_total(), 2)
            self.assertEqual(mgr._max_gems_per_family(), 1)
            self.assertFalse(mgr._mature_evidence_topk_policy_enabled())
            self.assertEqual(mgr._min_mature_eval_units(), 29)
            self.assertEqual(
                [entry["frontier_lane"] for entry in mgr._manifest_entries(manifest)[:3]],
                ["diagnostic_control", "alpha_incubator", "alpha_incubator"],
            )
            self.assertEqual(
                mgr._ordered_lanes_for_gems(
                    [
                        {"frontier_lane": "z_lane"},
                        {"frontier_lane": "diagnostic_control"},
                        {"frontier_lane": "alpha_incubator"},
                    ]
                ),
                ["diagnostic_control", "alpha_incubator", "z_lane"],
            )
            self.assertEqual(
                mgr._ordered_lanes_for_gems([{"variant_name": "no_lane"}]),
                [""],
            )
            self.assertEqual(
                mgr._gem_lane_quotas(
                    ordered_lanes=["diagnostic_control", "alpha_incubator", "other"],
                    cap=2,
                ),
                {"diagnostic_control": 1, "alpha_incubator": 1, "other": 0},
            )
            self.assertEqual(
                mgr._gem_lane_quotas(ordered_lanes=["a", "b"], cap=0), {"a": 0, "b": 0}
            )

            selected = mgr._select_gem_entries(manifest)
            self.assertEqual(
                {entry["variant_name"] for entry in selected},
                {"offpolicy_replay_a", "diag"},
            )
            compact = mgr._compact_gems(
                [
                    {
                        "variant_name": "offpolicy_replay_a",
                        "source_generation_id": 1,
                        "frontier_lane": "alpha_incubator",
                        "metrics": {"mean_test_taskscore": 1, "complete_eval": True},
                    },
                    {
                        "variant_name": "offpolicy_replay_a",
                        "source_generation_id": 1,
                        "frontier_lane": "alpha_incubator",
                        "metrics": {"mean_test_taskscore": 2, "complete_eval": True},
                    },
                    {
                        "variant_name": "diag",
                        "source_generation_id": 1,
                        "frontier_lane": "diagnostic_control",
                        "metrics": {"score": 0.1, "complete_eval": True},
                    },
                ],
                cap=2,
                sort_by_performance=True,
                preserve_lane_reserves=True,
            )
            self.assertEqual(len(compact), 2)

            disabled = GemsManager(
                run_dir=run_dir / "disabled",
                task_spec=_task_with_gems(enabled=False),
                frontier=_FakeFrontier(run_dir / "disabled", manifest),
            )
            self.assertEqual(disabled.active_gems(), [])
            self.assertEqual(
                disabled.maybe_trigger_after_boundary(completed_gen_id=0).reason, "disabled"
            )
            self.assertEqual(disabled.recover_pending_reset(completed_gen_id=0).reason, "disabled")

            server = GemsManager(
                run_dir=run_dir / "server",
                task_spec=_task_with_gems(reset_interval_generations=1, min_frontier_entries=1),
                frontier=_FakeFrontier(run_dir / "server", manifest),
                local_mode=False,
            )
            self.assertEqual(
                server.maybe_trigger_after_boundary(completed_gen_id=0).reason,
                "server_mode_not_supported",
            )

            limited = GemsManager(
                run_dir=run_dir / "limited",
                task_spec=_task_with_gems(reset_interval_generations=1, max_resets=0),
                frontier=_FakeFrontier(run_dir / "limited", manifest),
            )
            self.assertEqual(
                limited.maybe_trigger_after_boundary(completed_gen_id=0).reason,
                "max_resets_reached",
            )

            waiting = GemsManager(
                run_dir=run_dir / "waiting",
                task_spec=_task_with_gems(reset_interval_generations=3, min_frontier_entries=1),
                frontier=_FakeFrontier(run_dir / "waiting", manifest),
            )
            self.assertIn(
                "periodic_reset_waiting",
                waiting.maybe_trigger_after_boundary(completed_gen_id=0).reason,
            )

    def test_gem_record_round_trips_concrete_result_identity(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import (
            GemsManager,
            _variant_key,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=_task_with_gems(),
                frontier=_FakeFrontier(run_dir, {"cumulative_top": []}),
            )

            def write(rank: int, child: str) -> dict:
                return mgr._write_gem_finding(
                    entry={
                        "finding_id": f"source-{rank}",
                        "variant_name": "family_sweep",
                        "generation_id": 0,
                        "source_result_path": f"results/{child}/tiered_eval_summary.json",
                        "metrics": {"score": float(rank), "complete_eval": True},
                    },
                    rank=rank,
                    reset_count=1,
                    next_cycle_index=1,
                    completed_gen_id=0,
                    reason="test",
                )

            first = write(1, "child_a")
            second = write(2, "child_b")

        self.assertEqual(first["source_result_path"], "results/child_a/tiered_eval_summary.json")
        self.assertNotEqual(_variant_key(first), _variant_key(second))

    def test_configured_primary_metrics_fall_back_to_lane_metric_direction(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "performance": [
                        {
                            "finding_id": "high-loss",
                            "variant_name": "high-loss",
                            "generation_id": 0,
                            "frontier_lane": "performance",
                            "lane_metric_value": 0.8,
                            "lane_metric_direction": "minimize",
                            "metrics": {"complete_eval": True},
                        },
                        {
                            "finding_id": "low-loss",
                            "variant_name": "low-loss",
                            "generation_id": 0,
                            "frontier_lane": "performance",
                            "lane_metric_value": 0.2,
                            "lane_metric_direction": "minimize",
                            "metrics": {"complete_eval": True},
                        },
                    ]
                }
            }
            task = _task_with_gems(
                max_gems_per_reset=1,
                max_gems_total=1,
                primary_metric_keys=["task_score"],
                performance_lanes=["performance"],
            )
            task.evaluation = SimpleNamespace(
                primary_metric="task_score", direction="maximize", frontier_lanes=[]
            )
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=task,
                frontier=_FakeFrontier(run_dir, manifest),
            )

            selected = mgr._select_gem_entries(manifest, completed_gen_id=0)

        self.assertEqual([entry["variant_name"] for entry in selected], ["low-loss"])

    def test_new_evidence_can_replace_an_existing_gem_with_same_identity(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "performance": [
                        {
                            "finding_id": "new-v",
                            "variant_name": "v",
                            "generation_id": 1,
                            "frontier_lane": "performance",
                            "metrics": {"score": 100.0, "complete_eval": True},
                        }
                    ]
                }
            }
            task = _task_with_gems(performance_lanes=["performance"])
            task.evaluation = SimpleNamespace(
                primary_metric="score", direction="maximize", frontier_lanes=[]
            )
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=task,
                frontier=_FakeFrontier(run_dir, manifest),
            )
            existing = {
                "gem_finding_id": "old-v",
                "variant_name": "v",
                "source_generation_id": 0,
                "frontier_lane": "performance",
                "admission_metrics": {"score": 1.0, "complete_eval": True},
            }

            selected = mgr._select_gem_entries(
                manifest, existing_gems=[existing], completed_gen_id=1
            )

        self.assertEqual([entry["finding_id"] for entry in selected], ["new-v"])

    def test_generation_manifest_rows_remain_gem_candidates_outside_compact_views(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {"performance": []},
                "cumulative_top": [],
                "generations": {
                    "2": [
                        {
                            "finding_id": "generation-only",
                            "variant_name": "generation-only",
                            "frontier_lane": "performance",
                            "metrics": {"score": 2.0, "complete_eval": True},
                        }
                    ]
                },
            }
            task = _task_with_gems(performance_lanes=["performance"])
            task.evaluation = SimpleNamespace(
                primary_metric="score", direction="maximize", frontier_lanes=[]
            )
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=task,
                frontier=_FakeFrontier(run_dir, manifest),
            )

            selected = mgr._select_gem_entries(manifest, completed_gen_id=2)

        self.assertIn("generation-only", [entry["variant_name"] for entry in selected])

    def test_direct_result_candidates_can_satisfy_reset_entry_gate(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import (
            GemsManager,
            GemsTriggerResult,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            task = _task_with_gems(
                reset_interval_generations=1,
                min_frontier_entries=1,
                selection_policy="mature_evidence_top_k",
            )
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=task,
                frontier=_FakeFrontier(run_dir, {"lane_frontiers": {}, "cumulative_top": []}),
            )
            candidate = {
                "finding_id": "direct-result",
                "variant_name": "direct-result",
                "generation_id": 0,
            }
            expected = GemsTriggerResult(True, reason="test")
            with (
                patch.object(mgr, "_select_gem_entries", return_value=[candidate]),
                patch.object(mgr, "_admit_gems_and_reset", return_value=expected) as admit,
            ):
                result = mgr.maybe_trigger_after_boundary(completed_gen_id=0)

        self.assertTrue(result.triggered)
        self.assertEqual(admit.call_args.kwargs["entries"], [candidate])

    def test_positive_protocol_violation_count_rejects_gem_candidate(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import (
            _entry_has_hard_gem_rejection_marker,
        )

        self.assertTrue(
            _entry_has_hard_gem_rejection_marker(
                {
                    "metrics": {
                        "score": 1.0,
                        "effort_ratio": 1.0,
                        "coverage_ratio": 1.0,
                        "protocol_integrity_violation_count": 1,
                    }
                }
            )
        )
        self.assertTrue(
            _entry_has_hard_gem_rejection_marker(
                {
                    "metrics": {
                        "score": 1.0,
                        "effort_ratio": 1.0,
                        "coverage_ratio": 1.0,
                        "protocol_integrity_passed": False,
                    }
                }
            )
        )

    def test_mixed_lane_manifest_keeps_stronger_unlaned_candidate(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "lane_frontiers": {
                    "performance": [
                        {
                            "finding_id": "laned",
                            "variant_name": "laned",
                            "generation_id": 0,
                            "frontier_lane": "performance",
                            "metrics": {"score": 0.6, "complete_eval": True},
                        }
                    ]
                },
                "cumulative_top": [
                    {
                        "finding_id": "unlaned",
                        "variant_name": "unlaned",
                        "generation_id": 0,
                        "metrics": {"score": 0.8, "complete_eval": True},
                    }
                ],
            }
            task = _task_with_gems(
                max_gems_per_reset=1,
                max_gems_total=1,
                performance_lanes=["performance"],
                primary_metric_keys=["score"],
            )
            task.evaluation = SimpleNamespace(
                primary_metric="score", direction="maximize", frontier_lanes=[]
            )
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=task,
                frontier=_FakeFrontier(run_dir, manifest),
            )

            selected = mgr._select_gem_entries(manifest, completed_gen_id=0)

        self.assertEqual([entry["variant_name"] for entry in selected], ["unlaned"])

    def test_missing_primary_metric_never_outranks_real_negative_or_minimize_score(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import (
            _gem_performance_key,
        )

        missing = {
            "_gems_primary_metric_keys": ("score",),
            "metrics": {"validation_score": 100.0},
        }
        negative = {
            "_gems_primary_metric_keys": ("score",),
            "_gems_metric_direction": "maximize",
            "metrics": {"score": -1.0},
        }
        minimizing = {
            "_gems_primary_metric_keys": ("loss",),
            "_gems_metric_direction": "minimize",
            "metrics": {"loss": 0.2},
        }

        self.assertGreater(_gem_performance_key(negative), _gem_performance_key(missing))
        self.assertGreater(_gem_performance_key(minimizing), _gem_performance_key(missing))

    def test_known_ratio_failure_is_not_restored_by_complete_eval_fallback(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import (
            GemsManager,
            _entry_is_clean_gem_admission_candidate,
        )

        candidate = {
            "gem_finding_id": "ratio-failed",
            "variant_name": "ratio_failed",
            "source_generation_id": 0,
            "frontier_lane": "alpha_incubator",
            "admission_metrics": {
                "score": 2.0,
                "complete_eval": True,
                "effort_ratio": 0.5,
                "coverage_ratio": 1.0,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            task = _task_with_gems()
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=task,
                frontier=_FakeFrontier(run_dir, {}),
            )
            self.assertFalse(
                _entry_is_clean_gem_admission_candidate(
                    candidate,
                    task.evaluation.maturity_policy,
                )
            )
            self.assertEqual(
                mgr._compact_gems([candidate], max_generation_id=0),
                [],
            )

    def test_pending_reset_preserves_parent_ineligible_control_anchor_only_for_recovery(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager

        anchor = {
            "gem_finding_id": "control-anchor",
            "variant_name": "control_anchor",
            "source_generation_id": 0,
            "frontier_lane": "diagnostic_control",
            "parent_eligible": False,
            "admission_metrics": {"score": 1.0, "complete_eval": True},
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            task = _task_with_gems()
            task.evaluation.frontier_lanes = [
                {"name": "alpha_incubator", "parent_eligible": True},
                {"name": "diagnostic_control", "parent_eligible": False},
            ]
            mgr = GemsManager(
                run_dir=run_dir,
                task_spec=task,
                frontier=_FakeFrontier(run_dir, {}),
            )
            self.assertEqual(mgr._compact_gems([anchor], max_generation_id=0), [])
            recovered = mgr._compact_gems(
                [anchor],
                max_generation_id=0,
                allow_legacy_unknown_source=True,
            )

        self.assertEqual([entry["gem_finding_id"] for entry in recovered], ["control-anchor"])


if __name__ == "__main__":
    unittest.main()
