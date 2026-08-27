from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _text_payload(result: dict) -> dict:
    return json.loads(result["content"][0]["text"])


class PluginToolAdapterContractsTest(unittest.TestCase):
    def test_frontier_tool_defaults_to_environment_generation_cutoff(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "frontier"
            frontier_dir.mkdir()
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "metric_direction": "maximize",
                        "generations": {
                            "0": [
                                {
                                    "generation_id": 0,
                                    "variant_name": "current_candidate",
                                    "metric_value": 1.0,
                                    "evidence_stage": "full_T1",
                                },
                                {
                                    "generation_id": 0,
                                    "variant_name": "legacy_preliminary",
                                    "metric_value": 99.0,
                                    "evidence_stage": "preliminary",
                                },
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
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"FRONTIER_DIR": str(frontier_dir), "GENERATION_ID": "1"},
                clear=False,
            ):
                payload = _text_payload(asyncio.run(frontier_tools._handle_get_frontier({})))
                explicit_future = _text_payload(
                    asyncio.run(
                        frontier_tools._handle_get_frontier({"top_k": 10, "up_to_generation": 2})
                    )
                )
                explicit_unbounded = _text_payload(
                    asyncio.run(
                        frontier_tools._handle_get_frontier({"top_k": 10, "up_to_generation": -1})
                    )
                )

        names = [entry["variant_name"] for entry in payload["entries"]]
        self.assertEqual(payload["artifact_semantics"]["role"], "derived_view")
        self.assertFalse(payload["artifact_semantics"]["runtime_fact_source"])
        self.assertEqual(payload["artifact_semantics"]["stage"], "frontier_tool_response")
        self.assertIn("current_candidate", names)
        self.assertNotIn("legacy_preliminary", names)
        self.assertNotIn("future_candidate", names)
        self.assertNotIn(
            "future_candidate",
            [entry["variant_name"] for entry in explicit_future["entries"]],
        )
        self.assertNotIn(
            "future_candidate",
            [entry["variant_name"] for entry in explicit_unbounded["entries"]],
        )

    def test_frontier_tool_prioritizes_generic_performance_lane(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "frontier"
            frontier_dir.mkdir()
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "lane_frontiers": {
                            "benchmark_floor": [
                                {
                                    "generation_id": 0,
                                    "variant_name": "benchmark_anchor",
                                    "lane_metric_value": 100.0,
                                    "evidence_stage": "full_T1",
                                }
                            ],
                            "diagnostic_control": [
                                {
                                    "generation_id": 0,
                                    "variant_name": "diagnostic_anchor",
                                    "lane_metric_value": 90.0,
                                    "evidence_stage": "full_T1",
                                }
                            ],
                            "performance": [
                                {
                                    "generation_id": 0,
                                    "variant_name": "performance_candidate",
                                    "lane_metric_value": 1.0,
                                    "evidence_stage": "full_T1",
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"FRONTIER_DIR": str(frontier_dir), "GENERATION_ID": "1"},
                clear=False,
            ):
                payload = _text_payload(
                    asyncio.run(frontier_tools._handle_get_frontier({"top_k": 1}))
                )

        self.assertEqual(payload["entries"][0]["variant_name"], "performance_candidate")

    def test_evaluation_leaderboard_server_pareto_path_is_derived_view(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter as evaluation_tools

        with (
            patch.object(evaluation_tools, "_is_local_mode", return_value=False),
            patch.object(
                evaluation_tools, "_parse_anchor_metrics_env", return_value=[("score", "maximize")]
            ),
            patch.object(
                evaluation_tools,
                "_sqlite_leaderboard",
                return_value=json.dumps(
                    {
                        "mode": "pareto",
                        "entries": [{"variant_name": "v0", "score": 1.0}],
                    }
                ),
            ),
        ):
            payload = _text_payload(
                asyncio.run(evaluation_tools._handle_get_leaderboard({"top_k": 1, "generation": 2}))
            )

        self.assertEqual(payload["artifact_semantics"]["role"], "derived_view")
        self.assertEqual(payload["artifact_semantics"]["generation_id"], 2)
        self.assertFalse(payload["artifact_semantics"]["runtime_fact_source"])
        self.assertEqual(payload["_tool_output"]["tool_name"], "get_leaderboard")

    def test_frontier_tool_round_robins_configured_lane_entries(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "frontier"
            frontier_dir.mkdir()
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "frontier_lanes": [
                            {"name": "risk_control"},
                            {"name": "novelty"},
                            {"name": "performance"},
                        ],
                        "lane_frontiers": {
                            "performance": [
                                {
                                    "generation_id": 0,
                                    "variant_name": "perf_a",
                                    "lane_metric_value": 5.0,
                                    "evidence_stage": "full_T1",
                                    "scored_complete": True,
                                },
                                {
                                    "generation_id": 0,
                                    "variant_name": "perf_b",
                                    "lane_metric_value": 4.0,
                                    "evidence_stage": "full_T1",
                                    "scored_complete": True,
                                },
                            ],
                            "novelty": [
                                {
                                    "generation_id": 0,
                                    "variant_name": "novel_a",
                                    "lane_metric_value": 2.0,
                                    "evidence_stage": "full_T1",
                                    "scored_complete": True,
                                }
                            ],
                            "risk_control": [
                                {
                                    "generation_id": 0,
                                    "variant_name": "risk_a",
                                    "lane_metric_value": 1.0,
                                    "evidence_stage": "full_T1",
                                    "scored_complete": True,
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"FRONTIER_DIR": str(frontier_dir), "GENERATION_ID": "1"},
                clear=False,
            ):
                payload = _text_payload(
                    asyncio.run(frontier_tools._handle_get_frontier({"top_k": 3}))
                )

        self.assertEqual(
            [entry["variant_name"] for entry in payload["entries"]],
            ["risk_a", "novel_a", "perf_a"],
        )

    def test_frontier_tool_compacts_gems_lanes_and_generation_edges(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontier_dir = root / "frontier"
            frontier_dir.mkdir()
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "metric_direction": "maximize",
                        "generations": {
                            "bad": [],
                            "0": [
                                {"variant_name": "missing_metric", "generation_id": 0},
                                ["not-an-entry"],
                            ],
                            "1": "not-a-list",
                            "2": [{"variant_name": "future", "metric_value": 99}],
                        },
                        "lane_frontiers": {
                            "z_lane": [
                                {
                                    "generation_id": 0,
                                    "variant_name": "z",
                                    "lane_metric_name": "score",
                                    "lane_metric_value": "bad",
                                    "finding_id": "fz",
                                },
                                {"generation_id": 2, "variant_name": "future_lane"},
                            ],
                            "performance": [
                                {
                                    "generation_id": 0,
                                    "variant_name": "perf",
                                    "metric_value": 1,
                                    "promoted_for_lane": "performance",
                                    "evidence_stage": "full_T1",
                                }
                            ],
                            "bad_lane": "not-a-list",
                        },
                        "gems": {
                            "cycle_index": 2,
                            "reset_count": 1,
                            "cycle_start_generation": 3,
                            "entries": [
                                "bad",
                                {
                                    "gem_finding_id": "g0",
                                    "variant_name": "gem0",
                                    "source_generation_id": 0,
                                    "admission_metrics": {"score": 1, "complete_eval": True},
                                },
                                {
                                    "gem_finding_id": "g2",
                                    "variant_name": "future_gem",
                                    "source_generation_id": 2,
                                },
                            ],
                            "bottleneck_reports": [
                                "bad",
                                {
                                    "completed_generation": 0,
                                    "soft_agenda_priors": {"old": True},
                                },
                                {
                                    "completed_generation": 2,
                                    "soft_agenda_priors": {"future": True},
                                },
                            ],
                            "latest_soft_agenda_priors": {"unbounded": True},
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "FRONTIER_DIR": str(frontier_dir),
                    "LAST_COMPLETED_GENERATION_ID": "0",
                    "PRAXIST_FRONTIER_ALLOW_UNBOUNDED": "0",
                },
                clear=False,
            ):
                payload = _text_payload(
                    asyncio.run(
                        frontier_tools._handle_get_frontier(
                            {"top_k": "bad", "inline_limit": "bad", "up_to_generation": "9"}
                        )
                    )
                )

        self.assertEqual(payload["entries"][0]["variant_name"], "perf")
        self.assertEqual(payload["skipped_generations"], 1)
        self.assertEqual(payload["skipped_entries"], 2)
        self.assertEqual(payload["gems"]["entries"], [])
        self.assertIsNone(payload["gems"]["cycle_index"])
        self.assertIsNone(payload["gems"]["reset_count"])
        self.assertIsNone(payload["gems"]["cycle_start_generation"])
        self.assertFalse(payload["gems"]["historical_entries_complete"])
        self.assertFalse(payload["historical_validation_view_complete"])
        self.assertEqual(payload["gems"]["latest_soft_agenda_priors"], {"old": True})
        self.assertNotIn("future_gem", json.dumps(payload))

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "frontier"
            frontier_dir.mkdir()
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "metric_direction": "maximize",
                        "generations": "not-a-dict",
                        "lane_frontiers": {
                            "performance": [
                                "bad",
                                {
                                    "generation_id": "bad",
                                    "variant_name": "unknown",
                                },
                                {
                                    "generation_id": 1,
                                    "variant_name": "visible",
                                    "lane_metric_value": 1,
                                    "evidence_stage": "full_T1",
                                },
                            ]
                        },
                        "gems": {
                            "entries": "bad",
                            "bottleneck_reports": [
                                {"completed_generation": "bad"},
                                {"completed_generation": 1, "soft_agenda_priors": {"ok": True}},
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "FRONTIER_DIR": str(frontier_dir),
                    "PRAXIST_FRONTIER_ALLOW_UNBOUNDED": "1",
                },
                clear=False,
            ):
                unbounded_payload = _text_payload(
                    asyncio.run(frontier_tools._handle_get_frontier({"up_to_generation": "1"}))
                )

        self.assertEqual(unbounded_payload["total_generations"], 0)
        self.assertEqual(unbounded_payload["entries"][0]["variant_name"], "visible")
        self.assertEqual(unbounded_payload["gems"]["entries"], [])
        self.assertEqual(unbounded_payload["gems"]["latest_soft_agenda_priors"], {"ok": True})

    def test_frontier_tool_revalidates_gems_against_current_parent_lane_policy(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        def gem(name: str, lane: str, *, parent_eligible=True) -> dict:
            return {
                "gem_finding_id": name,
                "variant_name": name,
                "source_generation_id": 0,
                "frontier_lane": lane,
                "parent_eligible": parent_eligible,
                "admission_metrics": {"score": 1.0, "complete_eval": True},
            }

        manifest = {
            "frontier_lanes": [
                {"name": "candidate_library", "parent_eligible": True},
                {"name": "diagnostic", "parent_eligible": False},
            ],
            "gems": {
                "entries": [
                    gem("eligible", "candidate_library"),
                    gem("explicitly_ineligible", "candidate_library", parent_eligible=False),
                    gem("stale_diagnostic", "diagnostic"),
                    gem("removed_lane", "retired_lane"),
                ]
            },
        }

        compact = frontier_tools._compact_gems(manifest)

        self.assertEqual(
            [entry["variant_name"] for entry in compact["entries"]],
            ["eligible"],
        )

    def test_frontier_tool_exposes_migrated_historical_committed_gem(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        manifest = {
            "frontier_lanes": [
                {"name": "candidate_library", "parent_eligible": True},
                {"name": "diagnostic", "parent_eligible": False},
            ],
            "gems": {
                "min_mature_eval_units": 8,
                "entries": [
                    {
                        "gem_finding_id": "legacy-gem",
                        "variant_name": "legacy-candidate",
                        "source_generation_id": 2,
                        "frontier_lane": "candidate_library",
                        "admission_metrics": {"score": 1.0, "tier": "T1"},
                    },
                    {
                        "gem_finding_id": "legacy-partial",
                        "variant_name": "legacy-partial",
                        "source_generation_id": 2,
                        "frontier_lane": "candidate_library",
                        "admission_metrics": {
                            "score": 2.0,
                            "tier": "T1",
                            "evaluation_units": 2,
                        },
                    },
                ],
            },
        }

        compact = frontier_tools._compact_gems(manifest, up_to_generation=2)
        durable_keys = frontier_tools._durable_validation_entity_keys(
            manifest,
            up_to_generation=2,
        )

        self.assertEqual(
            [entry["variant_name"] for entry in compact["entries"]],
            ["legacy-candidate"],
        )
        self.assertTrue(any("legacy-candidate" in key for key in durable_keys))
        self.assertFalse(any("legacy-partial" in key for key in durable_keys))

    def test_frontier_tool_uses_gem_sidecar_source_generation_for_bounded_views(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        def gem(gem_id: str) -> dict:
            return {
                "gem_finding_id": gem_id,
                "variant_name": gem_id,
                "generation_id": 0,
                "admission_metrics": {"score": 1.0, "complete_eval": True},
            }

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            shared.mkdir()
            for gem_id, source_generation in (("current-gem", 1), ("future-gem", 3)):
                (shared / f"{gem_id}.json").write_text(
                    json.dumps(
                        {
                            "id": gem_id,
                            "source_frontier_entry": {"generation_id": source_generation},
                        }
                    ),
                    encoding="utf-8",
                )
            manifest = {
                "gems": {"entries": [gem("current-gem"), gem("future-gem"), gem("unknown-gem")]}
            }

            bounded = frontier_tools._compact_gems(
                manifest,
                up_to_generation=1,
                run_dir=run_dir,
            )
            unbounded = frontier_tools._compact_gems(manifest)
            bounded_keys = frontier_tools._durable_validation_entity_keys(
                manifest,
                up_to_generation=1,
                run_dir=run_dir,
            )

        self.assertEqual(
            [entry["variant_name"] for entry in bounded["entries"]],
            ["current-gem"],
        )
        self.assertEqual(
            [entry["variant_name"] for entry in unbounded["entries"]],
            ["current-gem", "future-gem", "unknown-gem"],
        )
        self.assertTrue(any("current-gem" in key for key in bounded_keys))
        self.assertFalse(any("future-gem" in key for key in bounded_keys))
        self.assertFalse(any("unknown-gem" in key for key in bounded_keys))

    def test_frontier_tool_keeps_mature_hard_violation_gems(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "frontier"
            frontier_dir.mkdir()
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "metric_direction": "maximize",
                        "generations": {},
                        "lane_frontiers": {},
                        "validation_candidates": {
                            "cumulative": [
                                {
                                    "generation_id": 5,
                                    "variant_name": "gen5_hard_full_t1",
                                    "finding_id": "retire_same_gem",
                                    "metric_name": "score",
                                    "metric_value": 1.0,
                                }
                            ],
                        },
                        "gems": {
                            "selection_policy": "mature_evidence_top_k",
                            "min_mature_eval_units": 29,
                            "primary_metric_keys": ["custom_task_metric"],
                            "performance_lanes": ["performance"],
                            "control_lanes": ["diagnostic_control"],
                            "entries": [
                                {
                                    "gem_finding_id": "hard",
                                    "variant_name": "gen5_hard_full_t1",
                                    "source_generation_id": 5,
                                    "frontier_lane": "performance",
                                    "admission_metrics": {
                                        "strategy_family": "learned_candidate",
                                        "custom_task_metric": 30.0,
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
                                    "gem_finding_id": "custom",
                                    "variant_name": "gen5_custom_full_t1",
                                    "source_generation_id": 5,
                                    "frontier_lane": "performance",
                                    "admission_metrics": {
                                        "strategy_family": "learned_candidate",
                                        "custom_task_metric": 42.0,
                                        "complete_eval": True,
                                        "tier": "T1",
                                        "n_eval_cells": 29,
                                        "scored_cell_count": 29,
                                        "promotion_eligible": True,
                                    },
                                },
                                {
                                    "gem_finding_id": "reject",
                                    "variant_name": "gen5_rejected_full_t1",
                                    "source_generation_id": 5,
                                    "frontier_lane": "performance",
                                    "admission_metrics": {
                                        "strategy_family": "learned_candidate",
                                        "custom_task_metric": 99.0,
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
                                    "admission_metrics": {
                                        "strategy_family": "diagnostic_control",
                                        "mean_test_taskscore": 98.0,
                                        "complete_eval": True,
                                        "tier": "T1",
                                        "n_eval_cells": 29,
                                        "scored_cell_count": 29,
                                    },
                                },
                                {
                                    "gem_finding_id": "control_task_candidate",
                                    "variant_name": "control_task_candidate_anchor",
                                    "source_generation_id": 5,
                                    "frontier_lane": "diagnostic_control",
                                    "admission_metrics": {
                                        "strategy_family": "task_candidate",
                                        "custom_task_metric": 101.0,
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
                                    "frontier_lane": "performance",
                                    "admission_metrics": {
                                        "strategy_family": "learned_candidate",
                                        "custom_task_metric": 97.0,
                                        "tier": "T1",
                                        "n_eval_cells": 6,
                                        "scored_cell_count": 6,
                                    },
                                },
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "FRONTIER_DIR": str(frontier_dir),
                    "PRAXIST_FRONTIER_ALLOW_UNBOUNDED": "1",
                },
                clear=False,
            ):
                payload = _text_payload(asyncio.run(frontier_tools._handle_get_frontier({})))

        self.assertEqual(
            [entry["variant_name"] for entry in payload["gems"]["entries"]],
            ["gen5_custom_full_t1"],
        )
        self.assertEqual(
            [entry["finding_id"] for entry in payload["validation_candidates"]],
            ["retire_same_gem"],
        )

    def test_frontier_tool_does_not_let_unknown_rows_hide_validation_candidates(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

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
                                    "variant_name": "candidate_x",
                                    "metric_value": 99.0,
                                }
                            ]
                        },
                        "lane_frontiers": {
                            "alpha_incubator": [
                                {
                                    "generation_id": 0,
                                    "variant_name": "lane_unknown",
                                    "lane_metric_value": 10.0,
                                }
                            ]
                        },
                        "validation_candidates": {
                            "generations": {
                                "0": [
                                    {
                                        "generation_id": 0,
                                        "finding_id": "scout_x",
                                        "variant_name": "candidate_x",
                                        "metric_name": "score",
                                        "metric_value": 7.0,
                                        "metric_direction": "maximize",
                                        "evidence_stage": "scout",
                                    }
                                ]
                            },
                            "cumulative": [],
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"FRONTIER_DIR": str(frontier_dir)}, clear=True):
                payload = _text_payload(
                    asyncio.run(frontier_tools._handle_get_frontier({"up_to_generation": 0}))
                )

        self.assertEqual(payload["entries"], [])
        self.assertEqual(
            payload["validation_candidates"][0]["variant_name"],
            "candidate_x",
        )
        self.assertEqual(payload["lane_frontiers"]["alpha_incubator"], [])
        self.assertEqual(payload["validation_candidates"][0]["finding_id"], "scout_x")

    def test_frontier_and_graph_query_handlers_cover_empty_error_and_success_paths(self) -> None:
        from praxist.plugins.tools.finding_graph_query import adapter as graph_tools
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontier_dir = root / "frontier"
            frontier_dir.mkdir()
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(
                    _text_payload(asyncio.run(frontier_tools._handle_get_frontier({})))["note"],
                    "FRONTIER_DIR not set",
                )
            with patch.dict(os.environ, {"FRONTIER_DIR": str(frontier_dir)}, clear=True):
                self.assertIn(
                    "No frontier manifest",
                    _text_payload(asyncio.run(frontier_tools._handle_get_frontier({})))["note"],
                )
                (frontier_dir / "frontier_manifest.json").write_text("{bad", encoding="utf-8")
                self.assertIn(
                    "could not be read",
                    _text_payload(asyncio.run(frontier_tools._handle_get_frontier({})))["error"],
                )
                (frontier_dir / "frontier_manifest.json").write_text("[]", encoding="utf-8")
                self.assertIn(
                    "JSON object",
                    _text_payload(asyncio.run(frontier_tools._handle_get_frontier({})))["error"],
                )
                (frontier_dir / "frontier_manifest.json").write_text(
                    json.dumps(
                        {
                            "metric_direction": "minimize",
                            "generations": {
                                "bad": [{"metric_value": 9}],
                                "0": [
                                    {
                                        "generation_id": 0,
                                        "rank": 1,
                                        "variant_name": "slow",
                                        "metric_value": 5,
                                        "evidence_stage": "full_T1",
                                    },
                                    {
                                        "generation_id": 0,
                                        "rank": 2,
                                        "variant_name": "fast",
                                        "metric_value": "1.5",
                                        "evidence_stage": "full_T1",
                                    },
                                    ["bad"],
                                ],
                                "1": "bad",
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                payload = _text_payload(
                    asyncio.run(
                        frontier_tools._handle_get_frontier(
                            {"top_k": "500", "up_to_generation": "0"}
                        )
                    )
                )
                self.assertEqual(payload["entries"][0]["variant_name"], "fast")
                self.assertEqual(payload["skipped_generations"], 1)
                self.assertEqual(payload["skipped_entries"], 1)
                self.assertEqual(frontier_tools._coerce_int("-1", default=3, minimum=1), 3)
                self.assertEqual(frontier_tools._coerce_int("999", default=3, maximum=10), 10)
                self.assertIsNone(frontier_tools._coerce_float("not-a-float"))

                (frontier_dir / "frontier_manifest.json").write_text(
                    json.dumps(
                        {
                            "metric_direction": "maximize",
                            "generations": {
                                "0": [
                                    {
                                        "generation_id": 0,
                                        "rank": 1,
                                        "variant_name": "benchmark_high",
                                        "metric_value": 100,
                                        "evidence_stage": "full_T1",
                                    }
                                ]
                            },
                            "lane_frontiers": {
                                "benchmark_floor": [
                                    {
                                        "generation_id": 0,
                                        "variant_name": "benchmark_high",
                                        "lane_metric_value": 100,
                                        "lane_metric_name": "future_fitness",
                                        "finding_id": "bench",
                                        "evidence_stage": "full_T1",
                                    }
                                ],
                                "alpha_incubator": [
                                    {
                                        "generation_id": 0,
                                        "variant_name": "legacy_preliminary_lane",
                                        "lane_metric_value": 999,
                                        "lane_metric_name": "mean_active_alpha",
                                        "finding_id": "legacy-prelim-lane",
                                        "source_frontier_lane": "alpha",
                                        "evidence_stage": "preliminary",
                                    },
                                    {
                                        "generation_id": 0,
                                        "variant_name": "alpha_low",
                                        "lane_metric_value": -1,
                                        "lane_metric_name": "mean_active_alpha",
                                        "finding_id": "alpha",
                                        "source_frontier_lane": "alpha",
                                        "evidence_stage": "full_T1",
                                    },
                                    {
                                        "generation_id": 2,
                                        "variant_name": "future_alpha",
                                        "lane_metric_value": 99,
                                        "lane_metric_name": "mean_active_alpha",
                                        "finding_id": "future-alpha",
                                        "source_frontier_lane": "alpha",
                                    },
                                    {
                                        "variant_name": "unknown_gen_alpha",
                                        "lane_metric_value": 50,
                                        "lane_metric_name": "mean_active_alpha",
                                        "finding_id": "unknown-alpha",
                                        "source_frontier_lane": "alpha",
                                    },
                                ],
                            },
                            "validation_candidates": {
                                "cumulative": [
                                    {
                                        "generation_id": 0,
                                        "variant_name": "scout_now",
                                        "metric_name": "score",
                                        "metric_value": 10,
                                        "metric_direction": "maximize",
                                        "finding_id": "scout-now",
                                        "evidence_stage": "scout",
                                        "scout_only": True,
                                        "metrics": {
                                            "bottleneck_target": "drawdown_regression",
                                            "tradeoff_class": "return_vs_mdd",
                                            "next_step_intent": "complete_validation",
                                        },
                                        "recommended_next_step": "complete_scored_validation_before_frontier_or_gems",
                                    },
                                    {
                                        "generation_id": 2,
                                        "variant_name": "scout_future",
                                        "metric_name": "score",
                                        "metric_value": 99,
                                        "metric_direction": "maximize",
                                        "finding_id": "scout-future",
                                        "evidence_stage": "scout",
                                    },
                                ]
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                lane_payload = _text_payload(
                    asyncio.run(
                        frontier_tools._handle_get_frontier({"top_k": 2, "up_to_generation": "0"})
                    )
                )
                self.assertTrue(lane_payload["lane_mode"])
                self.assertIn("alpha_incubator", lane_payload["lane_frontiers"])
                self.assertNotIn(
                    "legacy_preliminary_lane",
                    [entry["variant_name"] for entry in lane_payload["entries"]],
                )
                self.assertEqual(lane_payload["entries"][0]["variant_name"], "alpha_low")
                self.assertEqual(lane_payload["entries"][1]["variant_name"], "benchmark_high")
                self.assertEqual(
                    [entry["variant_name"] for entry in lane_payload["validation_candidates"]],
                    ["scout_now"],
                )
                self.assertEqual(
                    lane_payload["validation_candidates"][0]["bottleneck_target"],
                    "drawdown_regression",
                )
                self.assertEqual(
                    lane_payload["validation_candidates"][0]["next_step_intent"],
                    "complete_validation",
                )
                self.assertEqual(lane_payload["total_validation_candidates"], 1)
                past_lane_payload = _text_payload(
                    asyncio.run(
                        frontier_tools._handle_get_frontier({"top_k": 3, "up_to_generation": "0"})
                    )
                )
                names = [entry["variant_name"] for entry in past_lane_payload["entries"]]
                self.assertNotIn("future_alpha", names)
                self.assertNotIn("unknown_gen_alpha", names)
                validation_names = [
                    entry["variant_name"] for entry in past_lane_payload["validation_candidates"]
                ]
                self.assertNotIn("scout_future", validation_names)

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root / "store")}, clear=True):
                local_store.init_db()
                now = "2099-01-01T00:00:00"
                local_store.insert_finding(
                    {
                        "id": "a",
                        "finding_type": "result",
                        "title": "A",
                        "content": "x" * 900,
                        "metrics": {"score": 1.0},
                        "variant_name": "VA",
                        "peer_id": "p0",
                        "generation_id": 0,
                        "timestamp": now,
                    }
                )
                local_store.insert_finding(
                    {
                        "id": "b",
                        "finding_type": "insight",
                        "title": "B",
                        "content": "B",
                        "metrics": {},
                        "variant_name": "VB",
                        "peer_id": "p1",
                        "generation_id": 0,
                        "timestamp": now,
                    }
                )
                local_store.insert_edge(
                    {
                        "edge_id": "e1",
                        "src_finding_id": "a",
                        "dst_finding_id": "b",
                        "edge_type": "supports",
                        "confidence": 0.9,
                        "created_by": "test",
                    }
                )
                self.assertTrue(
                    asyncio.run(graph_tools._handle_get_finding_neighbors({})).get("is_error")
                )
                neighbors = _text_payload(
                    asyncio.run(
                        graph_tools._handle_get_finding_neighbors(
                            {"finding_id": "a", "edge_types": '["supports"]', "limit": 5}
                        )
                    )
                )
                self.assertEqual(neighbors["finding"]["id"], "a")
                self.assertEqual(neighbors["neighbor_findings"][0]["id"], "b")
                self.assertIn("truncated", neighbors["finding"]["content"])
                missing = asyncio.run(
                    graph_tools._handle_get_finding_neighbors({"finding_id": "missing"})
                )
                self.assertTrue(missing["is_error"])
                subgraph = _text_payload(
                    asyncio.run(
                        graph_tools._handle_get_finding_subgraph(
                            {"finding_id": "a", "max_depth": 1, "edge_types": '["supports"]'}
                        )
                    )
                )
                self.assertEqual({node["id"] for node in subgraph["nodes"]}, {"a", "b"})
                self.assertEqual(
                    _text_payload(
                        asyncio.run(
                            graph_tools._handle_get_finding_subgraph(
                                {"finding_id": "a", "max_depth": 1, "edge_types": "bad-json["}
                            )
                        )
                    )["edges"],
                    [],
                )
                unlinked = _text_payload(
                    asyncio.run(
                        graph_tools._handle_get_unlinked_recent_findings({"hours": 1000000})
                    )
                )
                self.assertEqual(unlinked["unlinked_findings"], [])
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.tools.local_store.init_db",
                side_effect=RuntimeError("db"),
            ):
                self.assertTrue(
                    asyncio.run(graph_tools._handle_get_finding_subgraph({"finding_id": "a"}))[
                        "is_error"
                    ]
                )
            self.assertEqual(
                graph_tools.create_tool_plugin()["tool_server_ref"],
                "tool_server:finding_graph_query",
            )
            with (
                patch.object(frontier_tools, "create_sdk_mcp_server", None),
                patch.object(frontier_tools, "tool", None),
                self.assertRaises(ImportError),
            ):
                frontier_tools.create_frontier_tools_server()
            with (
                patch.object(graph_tools, "create_sdk_mcp_server", None),
                patch.object(graph_tools, "tool", None),
                self.assertRaises(ImportError),
            ):
                graph_tools.create_finding_graph_query_server()

    def test_validation_candidates_use_generation_buckets_and_explicit_cutoff(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        manifest = {
            "validation_candidates": {
                "generations": {
                    "0": [
                        {
                            "generation_id": 0,
                            "variant_name": "gen9_named_old_signal",
                            "finding_id": "old",
                            "metric_name": "score",
                            "metric_value": 5,
                            "metric_direction": "maximize",
                            "signal_source_priority": 3,
                            "evidence_maturity_rank": 1,
                            "frontier_entity_key": "variant::old",
                            "metrics": {
                                "large": "x" * 100,
                                "bottleneck_target": "cash_drag",
                                "next_step_intent": "stress_validate",
                                "diversity_overlap_status": "narrow",
                                "diversity_most_similar_anchor": "anchor_a",
                                "diversity_overlap_fraction": 0.75,
                                "diversity_overlap_count": 3,
                                "diversity_overlap_total": 4,
                                "diversity_narrow_variation": True,
                            },
                            "identity_aliases": ["raw-alias"],
                        }
                    ],
                    "2": [
                        {
                            "generation_id": 2,
                            "variant_name": "future_signal",
                            "finding_id": "future",
                            "metric_name": "score",
                            "metric_value": 99,
                            "metric_direction": "maximize",
                            "frontier_entity_key": "variant::future",
                        }
                    ],
                },
                "cumulative": [
                    {
                        "generation_id": 0,
                        "variant_name": "stale_cumulative_copy",
                        "finding_id": "old-stale",
                        "metric_name": "score",
                        "metric_value": 500,
                        "metric_direction": "maximize",
                        "signal_source_priority": 9,
                        "evidence_maturity_rank": 9,
                        "frontier_entity_key": "variant:old",
                        "result_artifact_path": ("results/old/tiered_eval_summary.json"),
                    },
                    {
                        "generation_id": 0,
                        "variant_name": "retired_signal",
                        "finding_id": "retired",
                        "metric_name": "score",
                        "metric_value": 777,
                        "metric_direction": "maximize",
                        "frontier_entity_key": "variant::retired",
                    },
                    {
                        "generation_id": 0,
                        "variant_name": "bad_metric",
                        "finding_id": "bad",
                        "metric_name": "score",
                        "metric_value": "NaN",
                        "metric_direction": "sideways",
                    },
                    {
                        "generation_id": 1,
                        "variant_name": "cumulative_only",
                        "finding_id": "cum",
                        "metric_name": "score",
                        "metric_value": 6,
                        "metric_direction": "maximize",
                        "frontier_entity_key": "variant::cum",
                    },
                    {
                        "generation_id": 0,
                        "variant_name": "running_dirty",
                        "finding_id": "running-dirty",
                        "metric_name": "score",
                        "metric_value": 999,
                        "metric_direction": "maximize",
                        "metrics": {"result_status": "running", "score": 999},
                    },
                    {
                        "generation_id": 0,
                        "variant_name": "summary_dirty",
                        "finding_id": "summary-dirty",
                        "metric_name": "score",
                        "metric_value": 998,
                        "metric_direction": "maximize",
                        "metrics": {"summary_only": True, "result_status": "summary_only"},
                    },
                    {
                        "generation_id": 0,
                        "variant_name": "bool_summary_dirty",
                        "finding_id": "bool-summary-dirty",
                        "metric_name": "score",
                        "metric_value": 998,
                        "metric_direction": "maximize",
                        "metrics": {"summary_only": True},
                    },
                    {
                        "generation_id": 0,
                        "variant_name": "protocol_dirty",
                        "finding_id": "protocol-dirty",
                        "metric_name": "score",
                        "metric_value": 997,
                        "metric_direction": "maximize",
                        "metrics": {
                            "result_status": "protocol_invalid",
                            "protocol_integrity_status": "failed",
                            "excluded_from_durable_frontier": True,
                            "exclusion_reason": "protocol_integrity_failed",
                            "recommended_next_step": "rerun_with_valid_evaluator_protocol",
                        },
                    },
                    {
                        "generation_id": 0,
                        "variant_name": "mature_repair",
                        "finding_id": "mature-repair",
                        "metric_name": "score",
                        "metric_value": 7,
                        "metric_direction": "maximize",
                        "evidence_stage": "full_T1",
                        "excluded_from_durable_frontier": True,
                        "exclusion_reason": "promotion_eligible_false",
                        "metrics": {
                            "score": 7,
                            "evidence_stage": "full_T1",
                            "promotion_eligible": False,
                            "result_status": "promotion_failed",
                        },
                    },
                ],
            },
            "cumulative_top": [
                {
                    "generation_id": 0,
                    "variant_name": "retired_signal",
                    "finding_id": "retired-full",
                    "metric_value": 1.0,
                    "frontier_entity_key": "variant:retired",
                    "scored_complete": True,
                }
            ],
        }

        compact = frontier_tools._compact_validation_candidates(
            manifest,
            up_to_generation=0,
        )

        self.assertEqual(
            [entry["finding_id"] for entry in compact],
            ["old", "protocol-dirty", "mature-repair", "bad"],
        )
        self.assertEqual(compact[0]["variant_name"], "gen9_named_old_signal")
        self.assertEqual(compact[0]["bottleneck_target"], "cash_drag")
        self.assertEqual(compact[0]["next_step_intent"], "stress_validate")
        self.assertEqual(compact[0]["diversity_overlap_status"], "narrow")
        self.assertEqual(compact[0]["diversity_most_similar_anchor"], "anchor_a")
        self.assertEqual(compact[0]["diversity_overlap_fraction"], 0.75)
        self.assertEqual(compact[0]["diversity_overlap_count"], 3)
        self.assertEqual(compact[0]["diversity_overlap_total"], 4)
        self.assertIs(compact[0]["diversity_narrow_variation"], True)
        self.assertEqual(compact[0]["metric_value"], 5.0)
        self.assertIn(
            "results/old/tiered_eval_summary.json",
            compact[0]["identity_aliases"],
        )
        self.assertIn("raw-alias", compact[0]["identity_aliases"])
        self.assertEqual(compact[1]["result_status"], "protocol_invalid")
        self.assertEqual(compact[1]["protocol_integrity_status"], "failed")
        self.assertIs(compact[1]["excluded_from_durable_frontier"], True)
        self.assertEqual(compact[1]["exclusion_reason"], "protocol_integrity_failed")
        self.assertEqual(
            compact[1]["recommended_next_step"],
            "rerun_with_valid_evaluator_protocol",
        )
        self.assertEqual(compact[2]["metric_value"], 7.0)
        self.assertEqual(compact[2]["exclusion_reason"], "promotion_eligible_false")
        self.assertNotIn("metric_value", compact[3])
        self.assertNotIn("metrics", compact[0])

    def test_lane_frontier_cutoff_uses_source_generation_fields(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        compact = frontier_tools._compact_lane_frontiers(
            {
                "lane_frontiers": {
                    "alpha_incubator": [
                        {
                            "source_generation_id": 0,
                            "variant_name": "source_gen_alpha",
                            "lane_metric_value": 1.0,
                            "lane_metric_name": "score",
                            "finding_id": "source-gen-alpha",
                            "evidence_stage": "full_T1",
                        },
                        {
                            "metrics": {"generation_id": 0},
                            "variant_name": "metrics_gen_alpha",
                            "lane_metric_value": 0.5,
                            "lane_metric_name": "score",
                            "finding_id": "metrics-gen-alpha",
                            "evidence_stage": "full_T1",
                        },
                        {
                            "source_generation_id": 2,
                            "variant_name": "future_alpha",
                            "lane_metric_value": 99.0,
                            "lane_metric_name": "score",
                            "finding_id": "future-alpha",
                            "evidence_stage": "full_T1",
                        },
                    ]
                }
            },
            up_to_generation=0,
        )

        self.assertEqual(
            [entry["variant_name"] for entry in compact["alpha_incubator"]],
            ["source_gen_alpha", "metrics_gen_alpha"],
        )

    def test_frontier_tool_uses_task_policy_for_committed_capped_terminal_result(
        self,
    ) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "task"
            run_dir = task_dir / "experiments" / "run"
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir(parents=True)
            (run_dir / "effective_task_spec.yaml").write_text(
                """
task_id: policy_view
evaluation:
  primary_metric: score
  direction: maximize
  maturity_policy:
    min_effort_ratio: 1.0
    min_coverage_ratio: 1.0
    require_ratio_gate: true
    complete_stage_labels: [complete]
    preliminary_stage_labels: [preliminary, partial]
""".strip(),
                encoding="utf-8",
            )
            committed = {
                "generation_id": 0,
                "variant_name": "terminal_protocol_result",
                "finding_id": "terminal-result",
                "frontier_entity_key": "variant::terminal_protocol_result",
                "lane_metric_name": "score",
                "lane_metric_value": 1.0,
                "metric_name": "score",
                "metric_value": 1.0,
                "evidence_stage": "complete",
                "tier_status": "capped_at_T1",
                "final_status": "capped_at_T1",
                "result_status": "scored_complete",
                "capped": True,
                "result_capped": True,
                "scored_complete": True,
                "mature_enough": True,
                "maturity_basis": "effort_coverage_ratio",
                "effort_ratio": 1.0,
                "coverage_ratio": 1.0,
                "protocol_integrity_passed": True,
                "promotion_eligible": True,
                "parent_eligible": True,
            }
            manifest = {
                "artifact_semantics": {
                    "role": "canonical_state",
                    "status": "committed",
                    "runtime_fact_source": True,
                    "derived": False,
                    "audit_only": False,
                },
                "generations": {"0": [committed]},
                "lane_frontiers": {"alpha_incubator": [committed]},
                "validation_candidates": {"cumulative": [committed]},
                "gems": {
                    "entries": [
                        {
                            **committed,
                            "gem_finding_id": "gem-terminal-result",
                            "source_generation_id": 0,
                            "frontier_lane": "alpha_incubator",
                            "admission_metrics": {"score": 1.0},
                        }
                    ]
                },
            }
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "FRONTIER_DIR": str(frontier_dir),
                    "GENERATION_ID": "1",
                },
                clear=True,
            ):
                payload = _text_payload(
                    asyncio.run(
                        frontier_tools._handle_get_frontier({"top_k": 10, "up_to_generation": 0})
                    )
                )
                legacy_manifest = dict(manifest)
                legacy_manifest.pop("artifact_semantics")
                (frontier_dir / "frontier_manifest.json").write_text(
                    json.dumps(legacy_manifest),
                    encoding="utf-8",
                )
                legacy_payload = _text_payload(
                    asyncio.run(
                        frontier_tools._handle_get_frontier({"top_k": 10, "up_to_generation": 0})
                    )
                )

        self.assertTrue(payload["maturity_policy_loaded"])
        self.assertEqual(payload["maturity_policy_source"], "effective_task_spec.yaml")
        self.assertEqual(payload["canonical_frontier_entry_count"], 1)
        self.assertEqual(payload["returned_frontier_entry_count"], 1)
        self.assertEqual(payload["frontier_view_integrity_status"], "ok")
        self.assertEqual(
            [entry["variant_name"] for entry in payload["lane_frontiers"]["alpha_incubator"]],
            ["terminal_protocol_result"],
        )
        self.assertEqual(
            [entry["variant_name"] for entry in payload["gems"]["entries"]],
            ["terminal_protocol_result"],
        )
        self.assertEqual(payload["validation_candidates"], [])
        self.assertFalse(legacy_payload["committed_membership_trusted"])
        self.assertEqual(legacy_payload["returned_frontier_entry_count"], 1)
        self.assertEqual(
            legacy_payload["lane_frontiers"]["alpha_incubator"][0]["variant_name"],
            "terminal_protocol_result",
        )

    def test_committed_lane_is_stable_when_complete_entry_gains_capped_descriptors(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir(parents=True)
            (run_dir / "effective_task_spec.yaml").write_text(
                """
task_id: cross_task_entry_shape
evaluation:
  maturity_policy:
    min_effort_ratio: 1.0
    min_coverage_ratio: 1.0
    require_ratio_gate: true
    complete_stage_labels: [complete]
    preliminary_stage_labels: [preliminary, partial]
""".strip(),
                encoding="utf-8",
            )
            complete = {
                "generation_id": 0,
                "variant_name": "direct_complete",
                "finding_id": "direct-complete",
                "lane_metric_name": "score",
                "lane_metric_value": 1.0,
                "status": "ok",
                "evidence_stage": "complete",
                "scored_complete": True,
                "effort_ratio": 1.0,
                "coverage_ratio": 1.0,
                "promotion_eligible": True,
            }
            complete_with_descriptors = {
                **complete,
                "variant_name": "complete_with_descriptors",
                "finding_id": "complete-with-descriptors",
                "capped": True,
                "result_capped": True,
                "tier_status": "capped_at_task_terminal",
                "final_status": "capped_at_task_terminal",
            }
            manifest = {
                "artifact_semantics": {
                    "role": "canonical_state",
                    "status": "committed",
                    "runtime_fact_source": True,
                    "derived": False,
                    "audit_only": False,
                },
                "generations": {"0": [complete, complete_with_descriptors]},
                "lane_frontiers": {
                    "performance": [complete, complete_with_descriptors],
                },
            }
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"FRONTIER_DIR": str(frontier_dir), "GENERATION_ID": "1"},
                clear=True,
            ):
                payload = _text_payload(
                    asyncio.run(frontier_tools._handle_get_frontier({"up_to_generation": 0}))
                )

        self.assertEqual(payload["canonical_frontier_entry_count"], 2)
        self.assertEqual(payload["returned_frontier_entry_count"], 2)
        self.assertEqual(
            {entry["variant_name"] for entry in payload["lane_frontiers"]["performance"]},
            {"direct_complete", "complete_with_descriptors"},
        )
        self.assertEqual(payload["frontier_view_integrity_status"], "ok")

    def test_frontier_tool_does_not_reinterpret_explicit_committed_lane_membership(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "task"
            frontier_dir = task_dir / "experiments" / "run" / "frontier"
            frontier_dir.mkdir(parents=True)
            (task_dir / "task.yaml").write_text(
                """
task_id: policy_view_negative_controls
evaluation:
  primary_metric: score
  direction: maximize
  maturity_policy:
    min_effort_ratio: 1.0
    min_coverage_ratio: 1.0
    require_ratio_gate: true
    complete_stage_labels: [complete]
    preliminary_stage_labels: [preliminary, partial]
""".strip(),
                encoding="utf-8",
            )
            base = {
                "generation_id": 0,
                "lane_metric_name": "score",
                "lane_metric_value": 1.0,
                "evidence_stage": "complete",
                "scored_complete": True,
                "effort_ratio": 1.0,
                "coverage_ratio": 1.0,
                "promotion_eligible": True,
            }
            accepted = {**base, "variant_name": "accepted"}
            contradictory_descriptions = [
                {**base, "variant_name": "protocol_failed", "protocol_integrity_failed": True},
                {**base, "variant_name": "runtime_failed", "result_status": "timed_out"},
                {**base, "variant_name": "ratio_failed", "effort_ratio": 0.5},
                {
                    **base,
                    "variant_name": "explicitly_excluded",
                    "excluded_from_durable_frontier": True,
                },
            ]
            manifest = {
                "artifact_semantics": {
                    "role": "canonical_state",
                    "status": "committed",
                    "runtime_fact_source": True,
                    "derived": False,
                    "audit_only": False,
                },
                "lane_frontiers": {"performance": [accepted, *contradictory_descriptions]},
            }
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "FRONTIER_DIR": str(frontier_dir),
                    "PRAXIST_TASK_PROJECT_PATH": str(task_dir),
                    "GENERATION_ID": "1",
                },
                clear=True,
            ):
                payload = _text_payload(
                    asyncio.run(frontier_tools._handle_get_frontier({"up_to_generation": 0}))
                )
                manifest["lane_frontiers"] = {"performance": contradictory_descriptions}
                (frontier_dir / "frontier_manifest.json").write_text(
                    json.dumps(manifest),
                    encoding="utf-8",
                )
                contradictory_only_payload = _text_payload(
                    asyncio.run(frontier_tools._handle_get_frontier({"up_to_generation": 0}))
                )

        self.assertEqual(payload["canonical_frontier_entry_count"], 5)
        self.assertEqual(payload["returned_frontier_entry_count"], 5)
        self.assertTrue(payload["committed_membership_trusted"])
        self.assertEqual(payload["frontier_view_integrity_status"], "ok")
        self.assertEqual(
            payload["skipped_by_reason"]["lane_entries"],
            {},
        )
        self.assertEqual(
            {entry["variant_name"] for entry in payload["lane_frontiers"]["performance"]},
            {
                "accepted",
                "protocol_failed",
                "runtime_failed",
                "ratio_failed",
                "explicitly_excluded",
            },
        )
        self.assertEqual(contradictory_only_payload["returned_frontier_entry_count"], 4)
        self.assertEqual(contradictory_only_payload["frontier_view_integrity_status"], "ok")

    def test_frontier_tool_does_not_apply_live_policy_to_historical_run(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "task"
            frontier_dir = task_dir / "experiments" / "historical" / "frontier"
            frontier_dir.mkdir(parents=True)
            (task_dir / "task.yaml").write_text(
                """
task_id: changed_live_task
evaluation:
  primary_metric: score
  direction: maximize
  maturity_policy:
    min_effort_ratio: 1.0
    min_coverage_ratio: 1.0
    require_ratio_gate: true
""".strip(),
                encoding="utf-8",
            )
            committed = {
                "generation_id": 0,
                "source_generation_id": 0,
                "variant_name": "historically_authorized",
                "finding_id": "historically-authorized",
                "lane_metric_name": "score",
                "lane_metric_value": 1.0,
                "evidence_stage": "terminal",
                "tier_status": "capped_at_terminal",
                "capped": True,
                "scored_complete": True,
                "mature_enough": True,
                "maturity_basis": "effort_coverage_ratio",
                "effort_ratio": 0.75,
                "coverage_ratio": 0.75,
                "promotion_eligible": True,
                "parent_eligible": True,
            }
            manifest = {
                "artifact_semantics": {
                    "role": "canonical_state",
                    "status": "committed",
                    "runtime_fact_source": True,
                    "derived": False,
                    "audit_only": False,
                },
                "lane_frontiers": {"performance": [committed]},
                "gems": {
                    "entries": [
                        {
                            **committed,
                            "gem_finding_id": "historical-gem",
                            "frontier_lane": "performance",
                            "admission_metrics": {"score": 1.0},
                        }
                    ]
                },
            }
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "FRONTIER_DIR": str(frontier_dir),
                    "PRAXIST_TASK_PROJECT_PATH": str(task_dir),
                    "GENERATION_ID": "1",
                },
                clear=True,
            ):
                payload = _text_payload(
                    asyncio.run(frontier_tools._handle_get_frontier({"up_to_generation": 0}))
                )

        self.assertFalse(payload["maturity_policy_loaded"])
        self.assertEqual(payload["maturity_policy_source"], "unavailable")
        self.assertEqual(payload["frontier_view_integrity_status"], "ok")
        self.assertEqual(
            [entry["variant_name"] for entry in payload["lane_frontiers"]["performance"]],
            ["historically_authorized"],
        )
        self.assertEqual(
            [entry["variant_name"] for entry in payload["gems"]["entries"]],
            ["historically_authorized"],
        )

    def test_future_source_durable_entry_does_not_retire_current_validation_signal(
        self,
    ) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        durable = {
            "generation_id": 0,
            "source_generation_id": 2,
            "variant_name": "future_result",
            "finding_id": "future-result-full",
            "frontier_entity_key": "variant::future_result",
            "lane_metric_name": "score",
            "lane_metric_value": 2.0,
            "evidence_stage": "full_T1",
        }
        signal = {
            "generation_id": 0,
            "variant_name": "future_result",
            "finding_id": "future-result-signal",
            "frontier_entity_key": "variant::future_result",
            "metric_name": "score",
            "metric_value": 1.0,
        }
        manifest = {
            "lane_frontiers": {"performance": [durable]},
            "generations": {"0": [durable]},
            "validation_candidates": {"cumulative": [signal]},
        }

        lanes = frontier_tools._compact_lane_frontiers(manifest, up_to_generation=0)
        candidates = frontier_tools._compact_validation_candidates(
            manifest,
            up_to_generation=0,
        )

        self.assertEqual(lanes, {"performance": []})
        self.assertEqual(
            [entry["finding_id"] for entry in candidates],
            ["future-result-signal"],
        )

    def test_historical_cutoff_reconstructs_evicted_lane_member(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        def entry(generation: int, name: str, score: float) -> dict:
            return {
                "generation_id": generation,
                "variant_name": name,
                "finding_id": name,
                "frontier_entity_key": f"variant::{name}",
                "frontier_lane": "performance",
                "promoted_for_lane": "performance",
                "lane_metric_name": "score",
                "lane_metric_value": score,
                "evidence_stage": "full_T1",
            }

        old = entry(0, "old_member", 1.0)
        old["promotion_eligible"] = False
        current = entry(1, "current_member", 2.0)
        late_commit_with_old_source = {
            **entry(0, "late_commit", 100.0),
            "source_generation_id": 0,
        }
        manifest = {
            "frontier_lanes": [
                {
                    "name": "performance",
                    "k": 1,
                    "cumulative_cap": 1,
                    "axes": [["score", "maximize"]],
                    "require_truthy_metrics": ["promotion_eligible"],
                }
            ],
            "generations": {
                "0": [old],
                "1": [current],
                "2": [late_commit_with_old_source],
            },
            "lane_frontiers": {"performance": [late_commit_with_old_source]},
        }

        historical = frontier_tools._compact_historical_lane_frontiers(
            manifest,
            up_to_generation=0,
            maturity_policy=None,
            trust_committed_membership=True,
        )
        generation_one = frontier_tools._compact_historical_lane_frontiers(
            manifest,
            up_to_generation=1,
            maturity_policy=None,
            trust_committed_membership=True,
        )

        self.assertEqual(
            [item["variant_name"] for item in historical["performance"]],
            ["old_member"],
        )
        self.assertEqual(
            [item["variant_name"] for item in generation_one["performance"]],
            ["current_member"],
        )
        latest = frontier_tools._compact_historical_lane_frontiers(
            manifest,
            up_to_generation=2,
            maturity_policy=None,
            trust_committed_membership=True,
        )
        self.assertIsNone(latest)

    def test_historical_non_promotable_lane_membership_is_preserved(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        repair = {
            "generation_id": 0,
            "variant_name": "repair_signal",
            "finding_id": "repair-signal",
            "frontier_entity_key": "variant::repair_signal",
            "frontier_lane": "repair",
            "promoted_for_lane": "repair",
            "lane_metric_name": "score",
            "lane_metric_value": 1.0,
            "promotion_eligible": False,
            "parent_eligible": False,
        }
        excluded = {
            **repair,
            "variant_name": "excluded_signal",
            "finding_id": "excluded-signal",
            "frontier_entity_key": "variant::excluded_signal",
            "excluded_from_durable_frontier": True,
        }
        manifest = {
            "frontier_lanes": [
                {
                    "name": "repair",
                    "allow_non_promotable": True,
                    "parent_eligible": False,
                }
            ],
            "lane_frontiers": {"repair": [repair, excluded]},
            "validation_candidates": {
                "cumulative": [
                    {
                        "generation_id": 0,
                        "variant_name": "repair_signal",
                        "finding_id": "repair-signal-validation",
                        "frontier_entity_key": "variant::repair_signal",
                        "metric_name": "score",
                        "metric_value": 0.5,
                    }
                ]
            },
        }

        lanes = frontier_tools._compact_lane_frontiers(manifest, up_to_generation=0)
        candidates = frontier_tools._compact_validation_candidates(
            manifest,
            up_to_generation=0,
        )

        self.assertEqual(
            [item["variant_name"] for item in lanes["repair"]],
            ["repair_signal"],
        )
        self.assertEqual(candidates, [])

    def test_legacy_shell_row_cannot_retire_measured_validation_signal(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        shell = {
            "generation_id": 0,
            "variant_name": "candidate_x",
            "scored_complete": True,
        }
        signal = {
            "generation_id": 0,
            "variant_name": "candidate_x",
            "finding_id": "candidate-x-signal",
            "frontier_entity_key": "variant::candidate_x",
            "metric_name": "score",
            "metric_value": 1.0,
        }
        manifest = {
            "lane_frontiers": {"performance": [shell]},
            "validation_candidates": {"cumulative": [signal]},
        }

        lanes = frontier_tools._compact_lane_frontiers(manifest, up_to_generation=0)
        candidates = frontier_tools._compact_validation_candidates(
            manifest,
            up_to_generation=0,
        )

        self.assertEqual(lanes, {"performance": []})
        self.assertEqual(
            [entry["finding_id"] for entry in candidates],
            ["candidate-x-signal"],
        )

    def test_historical_replay_tolerates_malformed_persisted_lane_capacity(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        entry = {
            "generation_id": 0,
            "variant_name": "kept",
            "finding_id": "kept",
            "frontier_lane": "performance",
            "promoted_for_lane": "performance",
            "lane_metric_name": "score",
            "lane_metric_value": 1.0,
        }
        manifest = {
            "frontier_lanes": [
                {
                    "name": "performance",
                    "k": "bad",
                    "cumulative_cap": "bad",
                    "axes": [["score", "maximize"]],
                }
            ],
            "generations": {"0": [entry], "1": []},
            "lane_frontiers": {"performance": []},
        }

        historical = frontier_tools._compact_historical_lane_frontiers(
            manifest,
            up_to_generation=0,
            maturity_policy=None,
            trust_committed_membership=True,
        )

        self.assertEqual(
            [item["variant_name"] for item in historical["performance"]],
            ["kept"],
        )

    def test_pre_reset_cutoff_hides_current_gems_when_generation_ledger_is_empty(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "frontier"
            frontier_dir.mkdir()
            manifest = {
                "artifact_semantics": {
                    "role": "canonical_state",
                    "status": "committed",
                    "runtime_fact_source": True,
                    "derived": False,
                    "audit_only": False,
                },
                "generations": {},
                "lane_frontiers": {},
                "gems": {
                    "cycle_index": 1,
                    "reset_count": 1,
                    "cycle_start_generation": 3,
                    "entries": [
                        {
                            "gem_finding_id": "gem-after-reset",
                            "variant_name": "old_source_new_admission",
                            "source_generation_id": 0,
                            "admission_metrics": {
                                "score": 1.0,
                                "complete_eval": True,
                            },
                        }
                    ],
                },
            }
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"FRONTIER_DIR": str(frontier_dir), "GENERATION_ID": "3"},
                clear=True,
            ):
                before_reset = _text_payload(
                    asyncio.run(frontier_tools._handle_get_frontier({"up_to_generation": 0}))
                )
                reset_boundary = _text_payload(
                    asyncio.run(frontier_tools._handle_get_frontier({"up_to_generation": 2}))
                )

        self.assertEqual(before_reset["gems"]["entries"], [])
        self.assertIsNone(before_reset["gems"]["cycle_index"])
        self.assertFalse(before_reset["historical_validation_view_complete"])
        self.assertEqual(
            [entry["variant_name"] for entry in reset_boundary["gems"]["entries"]],
            ["old_source_new_admission"],
        )
        self.assertEqual(reset_boundary["gems"]["cycle_index"], 1)

    def test_legacy_cumulative_only_manifest_remains_readable(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "frontier"
            frontier_dir.mkdir()
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {},
                        "cumulative_top": [
                            {
                                "generation_id": 0,
                                "variant_name": "legacy_committed",
                                "finding_id": "legacy-committed",
                                "metric_value": 1.0,
                                "evidence_stage": "full_T1",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"FRONTIER_DIR": str(frontier_dir), "GENERATION_ID": "1"},
                clear=True,
            ):
                payload = _text_payload(
                    asyncio.run(frontier_tools._handle_get_frontier({"up_to_generation": 0}))
                )

        self.assertEqual(payload["canonical_frontier_entry_count"], 1)
        self.assertEqual(payload["returned_frontier_entry_count"], 1)
        self.assertEqual(payload["frontier_view_integrity_status"], "ok")
        self.assertEqual(payload["entries"][0]["variant_name"], "legacy_committed")

    def test_frontier_tool_reports_validation_candidate_hard_cap(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools
        from praxist.plugins.tools.result_envelope import read_tool_result_ref

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            frontier_dir = Path(tmp) / "frontier"
            frontier_dir.mkdir()
            candidates = [
                {
                    "generation_id": 0,
                    "variant_name": f"candidate_{idx}",
                    "finding_id": f"candidate-{idx}",
                    "metric_name": "score",
                    "metric_value": idx,
                    "metric_direction": "maximize",
                    "frontier_entity_key": f"variant::candidate_{idx}",
                }
                for idx in range(60)
            ]
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps({"validation_candidates": {"cumulative": candidates}}),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {"FRONTIER_DIR": str(frontier_dir), "PRAXIST_RUN_DIR": str(run_dir)},
                clear=True,
            ):
                payload = _text_payload(
                    asyncio.run(frontier_tools._handle_get_frontier({"inline_limit": 2}))
                )
                full_ref = payload["_tool_output"]["full_result_ref"]
                chunks = []
                offset = 0
                while True:
                    chunk = read_tool_result_ref(full_ref, offset=offset, max_chars=50000)
                    chunks.append(chunk["text"])
                    next_offset = chunk["next_offset"]
                    if next_offset is None:
                        break
                    offset = next_offset

            self.assertEqual(payload["total_validation_candidates"], 60)
            self.assertEqual(payload["returned_validation_candidates"], 48)
            self.assertTrue(payload["validation_candidates_truncated"])
            self.assertEqual(len(payload["validation_candidates"]), 48)
            self.assertEqual(
                payload["returned_validation_candidates"],
                len(payload["validation_candidates"]),
            )
            self.assertTrue(payload["_tool_output"]["truncated"])
            self.assertEqual(
                payload["_tool_output"]["truncated_lists"]["validation_candidates"],
                {"returned": 48, "total": 60},
            )
            full_payload = json.loads("".join(chunks))["payload"]
            self.assertEqual(full_payload["returned_validation_candidates"], 60)
            self.assertFalse(full_payload["validation_candidates_truncated"])
            self.assertEqual(len(full_payload["validation_candidates"]), 60)

    def test_gem_prompt_eligibility_fallback_rejects_nonclean_entries(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        real_import = __import__

        def fail_gems_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "praxist.plugins.workflow_stages.research_loop.backend.gems":
                raise ImportError("forced fallback")
            return real_import(name, globals, locals, fromlist, level)

        entry = {
            "variant_name": "candidate",
            "metrics": {
                "complete_eval": True,
                "clean_promotion_eligible": "non_promotable",
            },
        }
        with patch("builtins.__import__", side_effect=fail_gems_import):
            self.assertFalse(frontier_tools._gem_entry_is_prompt_eligible(entry))

        violated = {
            "variant_name": "candidate",
            "metrics": {
                "complete_eval": True,
                "hard_constraint_violations": ["missing_required_control"],
            },
        }
        with patch("builtins.__import__", side_effect=fail_gems_import):
            self.assertFalse(frontier_tools._gem_entry_is_prompt_eligible(violated))

        protocol_invalid = {
            "variant_name": "fixed_weight_candidate",
            "metrics": {
                "complete_eval": True,
                "protocol_integrity_status": "failed",
                "suspect_protocol": True,
            },
        }
        with patch("builtins.__import__", side_effect=fail_gems_import):
            self.assertFalse(frontier_tools._gem_entry_is_prompt_eligible(protocol_invalid))

        protocol_invalid_hyphen = {
            "variant_name": "fixed_weight_candidate",
            "metrics": {
                "complete_eval": True,
                "protocol_integrity_status": "protocol-invalid",
            },
        }
        with patch("builtins.__import__", side_effect=fail_gems_import):
            self.assertFalse(frontier_tools._gem_entry_is_prompt_eligible(protocol_invalid_hyphen))

        unknown_maturity = {
            "variant_name": "candidate",
            "metrics": {
                "score": 1.0,
            },
        }
        with patch("builtins.__import__", side_effect=fail_gems_import):
            self.assertFalse(frontier_tools._gem_entry_is_prompt_eligible(unknown_maturity))

        legacy = {
            "gem_finding_id": "legacy",
            "variant_name": "legacy",
            "admission_metrics": {
                "mean_test_taskscore": 1.0,
            },
        }
        with patch("builtins.__import__", side_effect=fail_gems_import):
            self.assertFalse(frontier_tools._gem_entry_is_prompt_eligible(legacy))

        parent_ineligible = {
            "gem_finding_id": "parent-false",
            "parent_eligible": False,
            "admission_metrics": {"score": 1.0, "complete_eval": True},
        }
        self.assertFalse(frontier_tools._gem_entry_is_prompt_eligible(parent_ineligible))

        ratio_failed = {
            "gem_finding_id": "ratio-failed",
            "admission_metrics": {
                "score": 1.0,
                "complete_eval": True,
                "effort_ratio": 0.5,
                "coverage_ratio": 1.0,
            },
        }
        self.assertFalse(frontier_tools._gem_entry_is_prompt_eligible(ratio_failed))

        modern_low_cell = {
            "gem_finding_id": "modern_low",
            "variant_name": "modern_low",
            "admission_metrics": {
                "mean_test_taskscore": 1.0,
                "n_eval_cells": 6,
            },
        }
        with patch("builtins.__import__", side_effect=fail_gems_import):
            self.assertFalse(frontier_tools._gem_entry_is_prompt_eligible(modern_low_cell))

    def test_frontier_durable_fallback_requires_complete_evidence(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        real_import = __import__

        def fail_frontier_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "praxist.plugins.workflow_stages.research_loop.backend.frontier":
                raise ImportError("forced fallback")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fail_frontier_import):
            self.assertFalse(
                frontier_tools._frontier_entry_is_durable(
                    {"variant_name": "unknown", "metric_value": 1.0}
                )
            )
            self.assertTrue(
                frontier_tools._frontier_entry_is_durable(
                    {
                        "variant_name": "mature",
                        "metric_value": 1.0,
                        "metrics": {"scored_complete": True},
                    }
                )
            )

    def test_frontier_tool_does_not_drop_validation_candidates_without_result_artifact(
        self,
    ) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "frontier"
            frontier_dir.mkdir()
            candidates = [
                {
                    "generation_id": 0,
                    "variant_name": f"candidate_{idx}",
                    "finding_id": f"candidate-{idx}",
                    "metric_name": "score",
                    "metric_value": idx,
                    "metric_direction": "maximize",
                    "frontier_entity_key": f"variant::candidate_{idx}",
                }
                for idx in range(60)
            ]
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps({"validation_candidates": {"cumulative": candidates}}),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"FRONTIER_DIR": str(frontier_dir)}, clear=True):
                payload = _text_payload(
                    asyncio.run(frontier_tools._handle_get_frontier({"inline_limit": 2}))
                )

            self.assertIsNone(payload["_tool_output"]["full_result_ref"])
            self.assertFalse(payload["_tool_output"]["truncated"])
            self.assertEqual(payload["total_validation_candidates"], 60)
            self.assertEqual(payload["returned_validation_candidates"], 60)
            self.assertFalse(payload["validation_candidates_truncated"])
            self.assertEqual(len(payload["validation_candidates"]), 60)

    def test_validation_candidates_ignore_future_or_preliminary_durable_surfaces(self) -> None:
        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        manifest = {
            "validation_candidates": {
                "cumulative": [
                    {
                        "generation_id": 0,
                        "variant_name": "future_durable_later",
                        "finding_id": "future-current",
                        "metric_name": "score",
                        "metric_value": 5,
                        "metric_direction": "maximize",
                        "frontier_entity_key": "variant::future_durable_later",
                    },
                    {
                        "generation_id": 0,
                        "variant_name": "legacy_prelim_surface",
                        "finding_id": "legacy-prelim",
                        "metric_name": "score",
                        "metric_value": 4,
                        "metric_direction": "maximize",
                        "frontier_entity_key": "variant::legacy_prelim_surface",
                    },
                    {
                        "generation_id": 0,
                        "variant_name": "alias_prelim_surface",
                        "finding_id": "alias-prelim",
                        "metric_name": "score",
                        "metric_value": 2,
                        "metric_direction": "maximize",
                        "frontier_entity_key": "variant::alias_prelim_surface",
                    },
                    {
                        "generation_id": 0,
                        "variant_name": "bucket_durable_now",
                        "finding_id": "bucket-now-current",
                        "metric_name": "score",
                        "metric_value": 3,
                        "metric_direction": "maximize",
                        "frontier_entity_key": "variant::bucket_durable_now",
                    },
                    {
                        "generation_id": 0,
                        "variant_name": "gen0_peer0",
                        "variant_id": "actual_child",
                        "finding_id": "retired-by-variant-id",
                        "metric_name": "score",
                        "metric_value": 1,
                        "metric_direction": "maximize",
                    },
                ]
            },
            "generations": {
                "0": [
                    {
                        "variant_name": "bucket_durable_now",
                        "finding_id": "bucket-now-full",
                        "metric_value": 8,
                        "frontier_entity_key": "variant::bucket_durable_now",
                        "scored_complete": True,
                    }
                ],
                "2": [
                    {
                        "variant_name": "future_durable_later",
                        "finding_id": "future-full-bucket",
                        "metric_value": 9,
                        "frontier_entity_key": "variant::future_durable_later",
                    }
                ],
            },
            "cumulative_top": [
                {
                    "generation_id": 2,
                    "variant_name": "future_durable_later",
                    "finding_id": "future-full",
                    "metric_value": 10,
                    "frontier_entity_key": "variant::future_durable_later",
                },
                {
                    "generation_id": 0,
                    "variant_name": "legacy_prelim_surface",
                    "finding_id": "legacy-prelim-copy",
                    "metric_value": 99,
                    "frontier_entity_key": "variant::legacy_prelim_surface",
                    "excluded_from_durable_frontier": True,
                },
                {
                    "generation_id": 0,
                    "variant_name": "alias_prelim_surface",
                    "finding_id": "alias-prelim-copy",
                    "metric_value": 98,
                    "frontier_entity_key": "variant::alias_prelim_surface",
                    "status": "unscored_artifact",
                    "metrics": {"complete_eval": False},
                },
                {
                    "generation_id": 0,
                    "variant_name": "gen0_peer0",
                    "variant_id": "actual_child",
                    "finding_id": "variant-id-full",
                    "metric_value": 97,
                    "scored_complete": True,
                },
            ],
        }

        compact = frontier_tools._compact_validation_candidates(
            manifest,
            up_to_generation=0,
        )

        self.assertEqual(
            {entry["finding_id"] for entry in compact},
            {"future-current", "legacy-prelim", "alias-prelim"},
        )
        self.assertNotIn("retired-by-variant-id", {entry["finding_id"] for entry in compact})

    def test_memory_and_prior_work_tools_preserve_read_only_boundaries(self) -> None:
        from praxist.plugins.tools.memory_tools import adapter as memory_tools
        from praxist.plugins.tools.prior_work_tools import adapter as prior_tools

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self.assertEqual(
                memory_tools.get_evidence_card(run_dir, "missing")["error"],
                "evidence_id not found: missing",
            )
            with self.assertRaises(ValueError):
                memory_tools.get_evidence_card(run_dir / "missing", "x")

            cards = [
                {
                    "evidence_id": "e1",
                    "source_type": "finding",
                    "source_ref": {"peer_id": "p0", "generation_id": 1, "variant_name": "V"},
                    "claim_relevance": {"supports": ["c1"], "challenges": [], "informs": []},
                    "interpretation": {"short": "Mechanism improves stability"},
                    "quality": {"is_negative": False},
                    "metrics": {"score": 1},
                },
                "bad",
                {
                    "evidence_id": "e2",
                    "source_ref": {"peer_id": "p1", "generation_id": 1},
                    "claim_relevance": {"supports": []},
                    "interpretation": {"short": "negative"},
                    "quality": {
                        "is_negative": True,
                        "evidence_valence": "negative",
                        "failure_mode": "generalization_failure",
                        "disconfirming_claim_ids": ["c2"],
                    },
                },
            ]
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.research_memory.card_builder.build_cards_from_db",
                return_value=cards,
            ):
                self.assertEqual(memory_tools.get_evidence_card(run_dir, "e1")["evidence_id"], "e1")
                filtered = memory_tools.query_evidence_cards(
                    run_dir,
                    claim_id="c1",
                    mechanism="stability",
                    peer_id="p0",
                    generation_id=1,
                    is_negative=False,
                    limit=1,
                )
                self.assertEqual(filtered[0]["evidence_id"], "e1")
                self.assertEqual(memory_tools.query_evidence_cards(run_dir, peer_id="nope"), [])
                self.assertEqual(
                    memory_tools.query_evidence_cards(run_dir, generation_id=99),
                    [],
                )
                self.assertEqual(
                    memory_tools.query_evidence_cards(run_dir, is_negative=True)[0]["evidence_id"],
                    "e2",
                )
                negative_summary = memory_tools.query_evidence_cards(run_dir, is_negative=True)[0]
                self.assertEqual(negative_summary["evidence_valence"], "negative")
                self.assertEqual(negative_summary["failure_mode"], "generalization_failure")
                self.assertEqual(negative_summary["disconfirming_claim_ids"], ["c2"])
                self.assertEqual(
                    memory_tools.query_evidence_cards(run_dir, claim_id="missing"),
                    [],
                )
                self.assertEqual(
                    memory_tools.query_evidence_cards(run_dir, mechanism="absent"),
                    [],
                )

            class FakeCoverageMatrix:
                def __init__(self, _run_dir: Path) -> None:
                    pass

                def query_bridge(self, left: str, right: str, dim: str):
                    return {
                        "variant_pair": [left, right],
                        "grid_dimension": dim,
                        "bridge_points_tested": [0.1],
                        "sources": ["s"],
                    }

                def query_grid(self, family: str, parameter: str):
                    if family == "missing":
                        return None
                    return {
                        "variant_family": family,
                        "parameter": parameter,
                        "values_tested": [1, 2],
                        "seed_counts": {"1": 3},
                        "sources": ["s"],
                    }

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.coverage_matrix.CoverageMatrix",
                FakeCoverageMatrix,
            ):
                self.assertTrue(
                    memory_tools.query_coverage_matrix(
                        run_dir,
                        bridge_pair=["A", "B"],
                        bridge_dimension="rho",
                    )["covered"]
                )
                self.assertFalse(
                    memory_tools.query_coverage_matrix(run_dir, variant_family="missing")["covered"]
                )
                self.assertEqual(
                    memory_tools.query_coverage_matrix(run_dir)["error"],
                    "must specify either variant_family or bridge_pair+bridge_dimension",
                )
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.coverage_matrix.CoverageMatrix",
                    FakeCoverageMatrix,
                ),
                patch.object(memory_tools, "_card_generation_map", return_value={"s": 2}),
            ):
                self.assertFalse(
                    memory_tools.query_coverage_matrix(
                        run_dir,
                        bridge_pair=["A", "B"],
                        bridge_dimension="rho",
                        max_generation_id=1,
                    )["covered"]
                )
                self.assertFalse(
                    memory_tools.query_coverage_matrix(
                        run_dir,
                        variant_family="family",
                        max_generation_id=1,
                    )["covered"]
                )

            class Entry:
                def __init__(self, entry_id: str, data: dict) -> None:
                    self.id = entry_id
                    self.data = data

                def to_dict(self) -> dict:
                    return {"id": self.id, "data": self.data}

            class FakeClaimLedger:
                def __init__(self, _run_dir: Path) -> None:
                    pass

                def list_active(self):
                    return [
                        Entry("c1", {"status": "active", "supports": [1], "challenges": [2, 3]})
                    ]

            class FakeDissentLedger:
                def __init__(self, _run_dir: Path) -> None:
                    pass

                def list_open(self):
                    return [Entry("d1", {"disputed_claim_id": "c1", "status": "open"})]

            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.claim_ledger.ClaimLedger",
                    FakeClaimLedger,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.dissent_ledger.DissentLedger",
                    FakeDissentLedger,
                ),
            ):
                self.assertEqual(memory_tools.list_active_claims(run_dir)[0]["supports_count"], 1)
                self.assertEqual(
                    memory_tools.list_open_objections(run_dir)[0]["disputed_claim_id"], "c1"
                )

            self.assertIn(
                "unsupported", memory_tools.get_ledger_entry(run_dir, "bad", "id")["error"]
            )
            self.assertIn(
                "not initialized",
                memory_tools.get_ledger_entry(run_dir, "claim_ledger", "missing")["error"],
            )
            ledgers = run_dir / "research_memory" / "ledgers"
            ledgers.mkdir(parents=True)
            (ledgers / "claim_ledger.yaml").write_text(
                "schema_version: v1\nledger_name: claim_ledger\nentries:\n  - id: c1\n    data:\n      x: 1\n",
                encoding="utf-8",
            )
            self.assertEqual(
                memory_tools.get_ledger_entry(run_dir, "claim_ledger", "c1")["id"], "c1"
            )
            self.assertIn(
                "entry not found",
                memory_tools.get_ledger_entry(run_dir, "claim_ledger", "missing")["error"],
            )
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.research_memory.source_resolver.SourceResolver.resolve",
                return_value={"source": "ok"},
            ):
                self.assertEqual(
                    memory_tools.resolve_source_ref(run_dir, {"artifact": "a"}), {"source": "ok"}
                )
            self.assertEqual(memory_tools.create_tool_plugin()["server_name"], "memory-tools")
            with (
                patch.object(memory_tools, "create_sdk_mcp_server", None),
                patch.object(memory_tools, "tool", None),
                self.assertRaises(ImportError),
            ):
                memory_tools.create_memory_tools_server(run_dir)

            self.assertTrue(
                asyncio.run(prior_tools._handle_download_snapshot({"snapshot_id": "../bad"}))[
                    "is_error"
                ]
            )
            with patch(
                "praxist.plugins.tools.prior_work_tools.adapter.get_server_url",
                side_effect=RuntimeError("offline"),
            ):
                self.assertTrue(
                    asyncio.run(prior_tools._handle_download_snapshot({"snapshot_id": "snap1"}))[
                        "is_error"
                    ]
                )
            with (
                patch(
                    "praxist.plugins.tools.prior_work_tools.adapter.get_server_url",
                    return_value="http://server",
                ),
                patch(
                    "praxist.plugins.tools.prior_work_tools.adapter.async_http_get",
                    return_value={"error": "not found"},
                ),
            ):
                self.assertEqual(
                    _text_payload(
                        asyncio.run(prior_tools._handle_download_snapshot({"snapshot_id": "snap1"}))
                    )["error"],
                    "not found",
                )
            with (
                patch.dict(os.environ, {"WORKSPACE_DIR": str(run_dir)}, clear=True),
                patch(
                    "praxist.plugins.tools.prior_work_tools.adapter.get_server_url",
                    return_value="http://server",
                ),
                patch(
                    "praxist.plugins.tools.prior_work_tools.adapter.async_http_get",
                    return_value={"snapshot_s3_key": "key", "variant_name": "V"},
                ),
                patch(
                    "praxist.infrastructure.s3_utils.download_snapshot_from_s3",
                    return_value=["a.py", "b.py"],
                ),
            ):
                out = _text_payload(
                    asyncio.run(prior_tools._handle_download_snapshot({"snapshot_id": "snap1"}))
                )
                self.assertEqual(out["status"], "downloaded")
                self.assertEqual(out["files_count"], 2)
            with (
                patch.object(prior_tools, "create_sdk_mcp_server", None),
                patch.object(prior_tools, "tool", None),
                self.assertRaises(ImportError),
            ):
                prior_tools.create_prior_work_tools_server()

    def test_memory_tools_apply_runtime_generation_cutoff(self) -> None:
        from praxist.plugins.tools.memory_tools import adapter as memory_tools
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.claim_ledger import (
            ClaimLedger,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            cards = [
                {
                    "evidence_id": "visible",
                    "source_ref": {
                        "generation_id": 1,
                        "peer_id": "gen1_peer0",
                        "variant_name": "visible_variant",
                    },
                    "claim_relevance": {"supports": ["C_visible"], "challenges": [], "informs": []},
                    "interpretation": {"short": "visible evidence"},
                    "quality": {"is_negative": False},
                    "metrics": {"score": 1.0},
                },
                {
                    "evidence_id": "future",
                    "source_ref": {
                        "generation_id": 2,
                        "peer_id": "gen2_peer0",
                        "variant_name": "future_variant",
                    },
                    "claim_relevance": {"supports": ["C_future"], "challenges": [], "informs": []},
                    "interpretation": {"short": "future evidence"},
                    "quality": {"is_negative": False},
                    "metrics": {"score": 99.0},
                },
            ]

            def fake_cards(_run_dir, only_gen=None, max_gen=None):
                out = []
                for card in cards:
                    generation = card["source_ref"]["generation_id"]
                    if only_gen is not None and generation != only_gen:
                        continue
                    if max_gen is not None and generation > max_gen:
                        continue
                    out.append(card)
                return out

            claims = ClaimLedger(run_dir)
            claims.upsert_claim(
                "C_visible",
                "Visible claim",
                "active",
                0.8,
                supports=["visible"],
            )
            claims.upsert_claim(
                "C_future",
                "Future claim",
                "active",
                0.9,
                supports=["future"],
            )

            ledgers = run_dir / "research_memory" / "ledgers"
            (ledgers / "mechanism_ledger.yaml").write_text(
                """
schema_version: v1
ledger_name: mechanism_ledger
entries:
  - id: raw_visible
    data:
      generation_id: 1
      value: ok
  - id: raw_future
    data:
      generation_id: 2
      value: leak
""",
                encoding="utf-8",
            )

            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.research_memory.card_builder.build_cards_from_db",
                    side_effect=fake_cards,
                ),
                patch.dict(os.environ, {"CURRENT_GEN_ID": "2"}, clear=False),
            ):
                queried = memory_tools.query_evidence_cards(run_dir, limit=10)
                self.assertEqual([card["evidence_id"] for card in queried], ["visible"])
                self.assertEqual(
                    memory_tools.get_evidence_card(run_dir, "visible")["evidence_id"],
                    "visible",
                )
                self.assertIn("error", memory_tools.get_evidence_card(run_dir, "future"))

                active_claim_ids = [
                    claim["id"] for claim in memory_tools.list_active_claims(run_dir)
                ]
                self.assertIn("C_visible", active_claim_ids)
                self.assertNotIn("C_future", active_claim_ids)

                self.assertEqual(
                    memory_tools.get_ledger_entry(run_dir, "mechanism_ledger", "raw_visible")["id"],
                    "raw_visible",
                )
                self.assertIn(
                    "error",
                    memory_tools.get_ledger_entry(run_dir, "mechanism_ledger", "raw_future"),
                )
                self.assertIn(
                    "error",
                    memory_tools.resolve_source_ref(
                        run_dir,
                        {"finding_path": "shared_findings/future.json", "generation_id": 2},
                    ),
                )

                shared = run_dir / "shared_findings"
                shared.mkdir(exist_ok=True)
                (shared / "future.json").write_text(
                    json.dumps({"id": "future_file", "generation_id": 2, "content": "leak"}),
                    encoding="utf-8",
                )
                self.assertIn(
                    "error",
                    memory_tools.resolve_source_ref(
                        run_dir,
                        {"finding_path": "shared_findings/future.json", "generation_id": 1},
                    ),
                )

    def test_memory_tools_generation_cutoff_helpers_cover_nested_sources(self) -> None:
        from praxist.plugins.tools.memory_tools import adapter as memory_tools

        self.assertEqual(memory_tools._coerce_int("bad", default=7), 7)
        self.assertEqual(
            memory_tools._generation_from_obj({"data": {"source_ref": {"gen_id": "3"}}}),
            3,
        )
        self.assertEqual(
            memory_tools._source_ids_from_obj(
                {"nested": [{"supports": "E1"}, {"sources": ["E2"]}]}
            ),
            ["E1", "E2"],
        )
        self.assertFalse(
            memory_tools._visible_at_generation(
                {"supports": ["missing"]},
                1,
                evidence_generations={},
            )
        )
        self.assertTrue(
            memory_tools._visible_at_generation(
                {"supports": ["E1"]},
                1,
                evidence_generations={"E1": 1},
            )
        )
        self.assertFalse(memory_tools._resolved_source_visible({}, 1))
        self.assertTrue(memory_tools._resolved_source_visible({"error": "missing"}, 1))
        self.assertFalse(
            memory_tools._resolved_source_visible({"content": {"generation_id": 3}}, 1)
        )
        with patch.dict(os.environ, {"PRAXIST_MEMORY_TOOLS_ALLOW_UNBOUNDED": "yes"}, clear=True):
            self.assertEqual(memory_tools._runtime_generation_limit(9), 9)
        with patch.dict(os.environ, {"COMPLETED_GEN_ID": "2"}, clear=True):
            self.assertEqual(memory_tools._runtime_generation_limit(None), 2)
            self.assertEqual(memory_tools._runtime_generation_limit(1), 1)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            run_dir.mkdir(exist_ok=True)
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.research_memory.card_builder.build_cards_from_db",
                side_effect=RuntimeError("db"),
            ):
                self.assertEqual(memory_tools._card_generation_map(run_dir), {})
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.research_memory.card_builder.build_cards_from_db",
                return_value=[
                    "bad",
                    {"evidence_id": "E2", "source_ref": {"generation_id": 2}},
                ],
            ):
                self.assertEqual(memory_tools._card_generation_map(run_dir), {"E2": 2})


if __name__ == "__main__":
    unittest.main()
