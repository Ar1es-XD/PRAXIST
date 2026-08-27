"""Regression tests for #150: primary_metric hoisting in findings_ingest.

When a peer writes a finding to ``shared_findings/`` directly (because
the MCP ``share_finding`` tool is unavailable — see #149), the pipeline
ingests the JSON into SQLite and end-of-gen ``frontier.promote`` reads
the row. Promotion silently drops findings whose primary metric is
nested under ``final_results`` / ``aggregated`` / ``details`` etc.
instead of the canonical flat ``metrics[primary_metric]`` location — so
the operator sees ``findings_total: 15, variants_total: 0`` despite
the peer producing real measurements.

These tests pin the hoisting contract: when the task's
``primary_metric`` is forwarded into ingest, a nested value gets
lifted to ``metrics[primary_metric]`` and downstream filters see the
canonical shape. When ``primary_metric`` is absent, ingest keeps the source
fields but does not invent domain-specific metric semantics.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from praxist.plugins.workflow_stages.research_loop.backend.tools import findings_ingest


class ExtractMetricsPrimaryMetricHoistTest(unittest.TestCase):
    def test_primary_metric_hoisted_from_nested_parent(self) -> None:
        """Peer nested the primary value under a non-canonical parent
        (the propcalc-migration footgun from #150). With
        ``primary_metric`` set, ingest hoists the discovered numeric to
        ``metrics[primary_metric]`` so frontier.promote can find it.
        """
        content = {
            "final_results": {"casam": {"verification_pass_rate": 0.85}},
        }
        out = findings_ingest.extract_metrics(content, primary_metric="verification_pass_rate")
        self.assertEqual(out.get("verification_pass_rate"), 0.85)

    def test_primary_metric_in_flat_metrics_dict_passes_through_unchanged(self) -> None:
        """The flat-dict path (step 1 of ``extract_metrics``) already
        preserves the primary value — the hoist must not double-write.
        """
        content = {"metrics": {"verification_pass_rate": 0.9}}
        out = findings_ingest.extract_metrics(content, primary_metric="verification_pass_rate")
        self.assertEqual(out["verification_pass_rate"], 0.9)

    def test_boolean_primary_metric_is_not_coerced_to_a_measurement(self) -> None:
        content = {"metrics": {"verification_pass_rate": False}}

        out = findings_ingest.extract_metrics(content, primary_metric="verification_pass_rate")

        self.assertNotIn("verification_pass_rate", out)

    def test_negative_completion_facts_survive_filesystem_ingest(self) -> None:
        content = {
            "metrics": {
                "score": 9.0,
                "complete_eval": False,
                "partial": True,
                "protocol_integrity_failed": True,
                "protocol_integrity_passed": False,
                "effort_ratio": 1.0,
                "coverage_ratio": 1.0,
            }
        }

        out = findings_ingest.extract_metrics(content, primary_metric="score")

        self.assertIs(out["complete_eval"], False)
        self.assertIs(out["partial"], True)
        self.assertIs(out["protocol_integrity_failed"], True)
        self.assertIs(out["protocol_integrity_passed"], False)
        from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
            evidence_maturity_snapshot,
        )

        self.assertIs(evidence_maturity_snapshot({"metrics": out})["mature_enough"], False)

    def test_canonical_variant_id_survives_filesystem_ingest(self) -> None:
        out = findings_ingest.extract_metrics(
            {"metrics": {"canonical_variant_id": "candidate-a", "score": 1.0}},
            primary_metric="score",
        )

        self.assertEqual(out["canonical_variant_id"], "candidate-a")

    def test_provenance_hashes_remain_exact_strings(self) -> None:
        hashes = {
            "candidate_sha256": "8bb00d" + "1" * 58,
            "artifact_digest": "abcdef" + "2" * 58,
            "content_checksum": "5" * 64,
        }

        out = findings_ingest.extract_metrics(
            {"metrics": {**hashes, "score": 1.0}},
            primary_metric="score",
        )

        self.assertEqual({key: out[key] for key in hashes}, hashes)
        self.assertEqual(out["score"], 1.0)

    def test_primary_metric_none_does_not_guess_domain_metric(self) -> None:
        content = {
            "final_results": {
                "candidate": {
                    "verification_pass_rate": 0.85,
                    "accuracy": 0.99,
                    "gap": 0.03,
                }
            }
        }
        out = findings_ingest.extract_metrics(content)
        self.assertNotIn("verification_pass_rate", out)
        self.assertNotIn("test_accuracy", out)
        self.assertNotIn("train_test_gap", out)

    def test_primary_metric_accuracy_alias_not_double_walked(self) -> None:
        """When ``primary_metric`` is one of the accuracy aliases
        (e.g. ``test_accuracy``), step 2 already handled it. The new
        step 4 must skip aliases so the legacy normalisation
        (percentage → fraction, dataset suffixing) still wins.
        """
        content = {"nested": {"test_acc": 82}}  # 82 → 0.82 (alias-walked)
        out = findings_ingest.extract_metrics(content, primary_metric="test_accuracy")
        self.assertEqual(out.get("test_accuracy"), 0.82)

    def test_declared_metric_aliases_preserve_the_exact_task_key(self) -> None:
        for alias in findings_ingest._ACC_KEY_ALIASES:
            with self.subTest(alias=alias):
                out = findings_ingest.extract_metrics(
                    {"nested": {alias: 82}},
                    primary_metric=alias,
                )
                self.assertEqual(out[alias], 0.82)
        for alias in findings_ingest._GAP_KEY_ALIASES:
            with self.subTest(alias=alias):
                out = findings_ingest.extract_metrics(
                    {"nested": {alias: 3}},
                    primary_metric=alias,
                )
                self.assertEqual(out[alias], 0.03)

    def test_declared_alias_wins_when_multiple_compatible_keys_exist(self) -> None:
        content = {"nested": {"test_accuracy": 82, "accuracy": 91}}

        out = findings_ingest.extract_metrics(content, primary_metric="accuracy")

        self.assertEqual(out["accuracy"], 0.91)
        self.assertEqual(out["test_accuracy"], 0.91)

    def test_accuracy_primary_does_not_infer_undeclared_gap(self) -> None:
        content = {"nested": {"test_acc": 82, "gap": 0.03}}

        out = findings_ingest.extract_metrics(content, primary_metric="test_accuracy")

        self.assertEqual(out.get("test_accuracy"), 0.82)
        self.assertNotIn("train_test_gap", out)

    def test_gap_alias_is_hoisted_only_when_declared_primary(self) -> None:
        content = {"nested": {"gap": 3.0}}

        out = findings_ingest.extract_metrics(content, primary_metric="train_test_gap")

        self.assertEqual(out.get("train_test_gap"), 0.03)

    def test_primary_metric_missing_nested_value_is_no_op(self) -> None:
        """If the primary metric is not findable anywhere in the
        content, the hoist is a silent no-op — never inserts a fake.
        """
        content = {"summary": "no metrics here"}
        out = findings_ingest.extract_metrics(content, primary_metric="verification_pass_rate")
        self.assertNotIn("verification_pass_rate", out)

    def test_non_accuracy_primary_metric_does_not_invent_accuracy_alias(self) -> None:
        content = {
            "final_results": {
                "candidate": {
                    "verification_pass_rate": 0.85,
                    "accuracy": 0.99,
                    "gap": 0.03,
                }
            }
        }
        out = findings_ingest.extract_metrics(content, primary_metric="verification_pass_rate")
        self.assertEqual(out.get("verification_pass_rate"), 0.85)
        self.assertNotIn("test_accuracy", out)
        self.assertNotIn("train_test_gap", out)

    def test_optimizer_is_not_generic_variant_name_alias(self) -> None:
        self.assertEqual(
            findings_ingest.extract_variant_name({"optimizer": "adamw"}),
            "",
        )

    def test_explicit_generation_zero_is_not_treated_as_unresolved(self) -> None:
        nested_later = Path("/tmp/run/gen_1/gen1_peer4_result.json")

        peer_id, generation_id = findings_ingest._infer_peer_and_gen(
            nested_later,
            {"peer_id": "gen0_peer2", "generation_id": 0},
        )

        self.assertEqual(peer_id, "gen0_peer2")
        self.assertEqual(generation_id, 0)
        self.assertEqual(
            findings_ingest._infer_peer_and_gen(nested_later, {"peer_id": "gen1_peer4"})[1],
            1,
        )


class ParseFindingFilePrimaryMetricTest(unittest.TestCase):
    def test_parse_finding_file_forwards_primary_metric_to_extract(self) -> None:
        """The orchestrator-side wrapper threads ``primary_metric`` so
        a peer-written file lands in SQLite with the canonical key
        populated even when the peer nested the value.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "gen0_peer0_result.json"
            file.write_text(
                json.dumps(
                    {
                        "finding_type": "result",
                        "title": "T",
                        "content": "C",
                        "final_results": {"casam": {"verification_pass_rate": 0.77}},
                    }
                ),
                encoding="utf-8",
            )
            parsed = findings_ingest.parse_finding_file(
                file, primary_metric="verification_pass_rate"
            )
            self.assertIsNotNone(parsed)
            assert parsed is not None  # narrow for type checker
            self.assertEqual(parsed["metrics"].get("verification_pass_rate"), 0.77)

    def test_parse_finding_file_preserves_top_level_diversity_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "gen0_peer0_diversity.json"
            file.write_text(
                json.dumps(
                    {
                        "finding_type": "result",
                        "title": "Diversity candidate",
                        "content": "C",
                        "variant_name": "candidate_a",
                        "metrics": {"score": 1.0},
                        "source_lane": "validation_candidates",
                        "target_lane": "mechanism_bridge",
                        "coverage_check": "touches sparse mechanism family",
                        "mechanism_hypothesis_deliverable": "explain gate interaction",
                        "identity_aliases": ["candidate-a", "variant::candidate_a"],
                        "diversity_overlap_status": "narrow",
                        "diversity_most_similar_anchor": "gen1_peer2_anchor",
                        "diversity_overlap_score": 0.87,
                        "diversity_overlap_fraction": 0.75,
                        "diversity_overlap_count": 3,
                        "diversity_overlap_total": 4,
                        "diversity_violated": False,
                        "diversity_violation": "semantic_family_cap",
                        "diversity_narrow_variation": True,
                        "mechanism_family": "temporal_credit",
                        "intervention_surface": "loss",
                        "intent": "repair_failure_mode",
                        "semantic_family": "drawdown_repair",
                        "parent_lineage": "gen1_peer2_anchor",
                        "novelty_axis": "critic_temperature",
                    }
                ),
                encoding="utf-8",
            )

            parsed = findings_ingest.parse_finding_file(file, primary_metric="score")

            self.assertIsNotNone(parsed)
            assert parsed is not None
            self.assertEqual(parsed["metrics"]["source_lane"], "validation_candidates")
            self.assertEqual(parsed["metrics"]["target_lane"], "mechanism_bridge")
            self.assertEqual(parsed["metrics"]["coverage_check"], "touches sparse mechanism family")
            self.assertEqual(
                parsed["metrics"]["mechanism_hypothesis_deliverable"],
                "explain gate interaction",
            )
            self.assertEqual(
                parsed["metrics"]["identity_aliases"],
                ["candidate-a", "variant::candidate_a"],
            )
            self.assertEqual(parsed["metrics"]["diversity_overlap_status"], "narrow")
            self.assertEqual(
                parsed["metrics"]["diversity_most_similar_anchor"],
                "gen1_peer2_anchor",
            )
            self.assertEqual(parsed["metrics"]["diversity_overlap_score"], 0.87)
            self.assertEqual(parsed["metrics"]["diversity_overlap_fraction"], 0.75)
            self.assertEqual(parsed["metrics"]["diversity_overlap_count"], 3)
            self.assertEqual(parsed["metrics"]["diversity_overlap_total"], 4)
            self.assertIs(parsed["metrics"]["diversity_violated"], False)
            self.assertEqual(parsed["metrics"]["diversity_violation"], "semantic_family_cap")
            self.assertIs(parsed["metrics"]["diversity_narrow_variation"], True)
            self.assertEqual(parsed["metrics"]["mechanism_family"], "temporal_credit")
            self.assertEqual(parsed["metrics"]["intervention_surface"], "loss")
            self.assertEqual(parsed["metrics"]["intent"], "repair_failure_mode")
            self.assertEqual(parsed["metrics"]["semantic_family"], "drawdown_repair")
            self.assertEqual(parsed["metrics"]["parent_lineage"], "gen1_peer2_anchor")
            self.assertEqual(parsed["metrics"]["novelty_axis"], "critic_temperature")
            self.assertEqual(parsed["extra"]["mechanism_family"], "temporal_credit")
            self.assertEqual(parsed["extra"]["diversity_overlap_score"], 0.87)
            self.assertEqual(parsed["extra"]["diversity_overlap_fraction"], 0.75)

    def test_parse_finding_file_canonicalizes_realized_dimension_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gen0_peer0_realized.json"
            path.write_text(
                json.dumps(
                    {
                        "finding_type": "result",
                        "title": "Realized design",
                        "metrics": {"score": 1.0},
                        "realized_dimensions": {"mechanism_family": "measured implementation"},
                    }
                ),
                encoding="utf-8",
            )

            parsed = findings_ingest.parse_finding_file(path, primary_metric="score")

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(
            parsed["design_dimensions"],
            {"mechanism_family": "measured implementation"},
        )
        self.assertNotIn("planned_dimensions", parsed)

    def test_parse_finding_file_preserves_maturity_ratios_from_supported_containers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "gen0_peer0_maturity.json"
            file.write_text(
                json.dumps(
                    {
                        "finding_type": "result",
                        "title": "Maturity candidate",
                        "content": "C",
                        "variant_name": "candidate_a",
                        "metrics": {"score": 1.0},
                        "effort_ratio": 0.9,
                        "details": {"coverage_ratio": 0.85},
                    }
                ),
                encoding="utf-8",
            )

            parsed = findings_ingest.parse_finding_file(file, primary_metric="score")

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["metrics"]["effort_ratio"], 0.9)
        self.assertEqual(parsed["metrics"]["coverage_ratio"], 0.85)

    def test_parse_finding_file_preserves_metrics_identity_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "gen0_peer0_aliases.json"
            file.write_text(
                json.dumps(
                    {
                        "finding_type": "result",
                        "title": "Alias candidate",
                        "content": "C",
                        "variant_name": "candidate_b",
                        "metrics": {
                            "score": 2.0,
                            "identity_aliases": ["candidate-b", "variant::candidate_b"],
                        },
                    }
                ),
                encoding="utf-8",
            )

            parsed = findings_ingest.parse_finding_file(file, primary_metric="score")

            self.assertIsNotNone(parsed)
            assert parsed is not None
            self.assertEqual(
                parsed["metrics"]["identity_aliases"],
                ["candidate-b", "variant::candidate_b"],
            )

    def test_parse_finding_file_backfills_empty_metrics_identity_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "gen0_peer0_alias_backfill.json"
            file.write_text(
                json.dumps(
                    {
                        "finding_type": "result",
                        "title": "Alias backfill candidate",
                        "content": "C",
                        "variant_name": "candidate_c",
                        "metrics": {"score": 3.0, "identity_aliases": []},
                        "extra": {"identity_aliases": ["candidate-c", "variant::candidate_c"]},
                    }
                ),
                encoding="utf-8",
            )

            parsed = findings_ingest.parse_finding_file(file, primary_metric="score")

            self.assertIsNotNone(parsed)
            assert parsed is not None
            self.assertEqual(
                parsed["metrics"]["identity_aliases"],
                ["candidate-c", "variant::candidate_c"],
            )

    def test_parse_finding_file_preserves_root_nested_identity_conflicts(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
            result_artifact_key,
            result_snapshot_key,
        )

        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "gen0_peer0_result.json"
            file.write_text(
                json.dumps(
                    {
                        "finding_type": "result",
                        "child_id": "root-child",
                        "source_result_path": "results/root.json",
                        "source_result_sha256": "root-sha",
                        "metrics": {
                            "child_id": "nested-child",
                            "source_result_path": "results/nested.json",
                            "source_result_sha256": "nested-sha",
                            "score": 1.0,
                            "current_aggregate": {
                                "child_id": "aggregate-child",
                                "source_result_path": "results/aggregate.json",
                                "source_result_sha256": "aggregate-sha",
                            },
                        },
                        "current_aggregate": {
                            "child_id": "top-aggregate-child",
                            "source_result_path": "results/top-aggregate.json",
                            "source_result_sha256": "top-aggregate-sha",
                        },
                    }
                ),
                encoding="utf-8",
            )

            parsed = findings_ingest.parse_finding_file(file, primary_metric="score")

        assert parsed is not None
        self.assertEqual(parsed["child_id"], "root-child")
        self.assertEqual(parsed["metrics"]["child_id"], "nested-child")
        self.assertEqual(
            parsed["metrics"]["current_aggregate"]["child_id"],
            "aggregate-child",
        )
        self.assertEqual(parsed["current_aggregate"]["child_id"], "top-aggregate-child")
        self.assertIsNone(result_artifact_key(parsed))
        self.assertIsNone(result_snapshot_key(parsed))

    def test_parse_finding_preserves_current_aggregate_maturity_facts(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
            evidence_maturity_snapshot,
        )

        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "gen0_peer0_result.json"
            file.write_text(
                json.dumps(
                    {
                        "finding_type": "result",
                        "metrics": {
                            "current_aggregate": {
                                "child_id": "child-a",
                                "source_result_path": "results/child-a.json",
                                "source_result_sha256": "child-a-sha",
                                "effort_ratio": 1.0,
                                "coverage_ratio": 1.0,
                                "scored_complete": True,
                                "score": 3.0,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            parsed = findings_ingest.parse_finding_file(file, primary_metric="score")

        assert parsed is not None
        aggregate = parsed["metrics"]["current_aggregate"]
        self.assertEqual(aggregate["score"], 3.0)
        self.assertTrue(aggregate["scored_complete"])
        snapshot = evidence_maturity_snapshot(parsed, {"require_ratio_gate": True})
        self.assertIs(snapshot["mature_enough"], True)
        self.assertEqual(snapshot["effort_ratio"], 1.0)
        self.assertEqual(snapshot["coverage_ratio"], 1.0)

    def test_parse_finding_file_preserves_promotion_maturity_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "gen0_peer2_protocol.json"
            file.write_text(
                json.dumps(
                    {
                        "finding_type": "result",
                        "title": "Protocol-passed candidate",
                        "content": "C",
                        "variant_name": "candidate_protocol",
                        "metrics": {
                            "score": 2.5,
                            "frontier_lane": "alpha_incubator",
                            "scored_complete": True,
                            "protocol_integrity_status": "passed",
                            "tier_reached": "T1",
                            "validation_only_result": False,
                            "scout_only": False,
                            "is_smoke_eval": False,
                            "suspect_protocol": False,
                        },
                    }
                ),
                encoding="utf-8",
            )

            parsed = findings_ingest.parse_finding_file(file, primary_metric="score")

            self.assertIsNotNone(parsed)
            assert parsed is not None
            metrics = parsed["metrics"]
            self.assertEqual(metrics["frontier_lane"], "alpha_incubator")
            self.assertTrue(metrics["scored_complete"])
            self.assertEqual(metrics["protocol_integrity_status"], "passed")
            self.assertEqual(metrics["tier_reached"], "T1")
            self.assertIs(metrics["validation_only_result"], False)
            self.assertIs(metrics["scout_only"], False)
            self.assertIs(metrics["is_smoke_eval"], False)
            self.assertIs(metrics["suspect_protocol"], False)

    def test_parse_finding_file_preserves_validation_signal_routing_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "gen0_peer2_validation_signal.json"
            file.write_text(
                json.dumps(
                    {
                        "finding_type": "result",
                        "title": "Validation-only signal",
                        "content": "C",
                        "variant_name": "candidate_signal",
                        "metrics": {
                            "score": 2.5,
                            "scored_complete": True,
                            "validation_only": True,
                        },
                        "validation_only": True,
                        "validation_only_result": True,
                        "artifact_signal_status": "late_after_generation_boundary",
                        "late_result_policy": "quarantined_signal",
                        "durability_scope": "validation_signal_only",
                    }
                ),
                encoding="utf-8",
            )

            parsed = findings_ingest.parse_finding_file(file, primary_metric="score")

            self.assertIsNotNone(parsed)
            assert parsed is not None
            metrics = parsed["metrics"]
            self.assertIs(metrics["validation_only"], True)
            self.assertIs(metrics["validation_only_result"], True)
            self.assertEqual(metrics["artifact_signal_status"], "late_after_generation_boundary")
            self.assertEqual(metrics["late_result_policy"], "quarantined_signal")
            self.assertEqual(metrics["durability_scope"], "validation_signal_only")

    def test_parse_finding_does_not_join_artifact_coordinates_across_containers(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            findings_ingest,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gen0_peer0_validation.json"
            path.write_text(
                json.dumps(
                    {
                        "finding_type": "result",
                        "variant_name": "candidate",
                        "source_result_path": "results/candidate/preliminary.json",
                        "metrics": {
                            "score": 1.0,
                            "validation_only_result": True,
                            "source_result_sha256": "unrelated-sha",
                        },
                    }
                ),
                encoding="utf-8",
            )

            parsed = findings_ingest.parse_finding_file(path, primary_metric="score")

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(
            parsed["metrics"]["source_result_path"],
            "results/candidate/preliminary.json",
        )
        self.assertNotIn("source_result_sha256", parsed["metrics"])

    def test_parse_finding_file_maps_legacy_suspect_alias_to_protocol_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "gen0_peer2_legacy_protocol.json"
            file.write_text(
                json.dumps(
                    {
                        "finding_type": "result",
                        "title": "Legacy suspect candidate",
                        "content": "C",
                        "variant_name": "legacy_candidate",
                        "metrics": {
                            "score": 2.5,
                            "scored_complete": True,
                            "suspect_fixed_weight_eval": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            parsed = findings_ingest.parse_finding_file(file, primary_metric="score")

            self.assertIsNotNone(parsed)
            assert parsed is not None
            metrics = parsed["metrics"]
            self.assertIs(metrics["suspect_protocol"], True)
            self.assertNotIn("suspect_fixed_weight_eval", metrics)

    def test_parse_finding_file_legacy_suspect_true_overrides_false_protocol_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "gen0_peer2_legacy_protocol_conflict.json"
            file.write_text(
                json.dumps(
                    {
                        "finding_type": "result",
                        "title": "Conflicting legacy suspect candidate",
                        "content": "C",
                        "variant_name": "legacy_conflict",
                        "metrics": {
                            "score": 2.5,
                            "scored_complete": True,
                            "suspect_protocol": False,
                            "suspect_fixed_weight_eval": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            parsed = findings_ingest.parse_finding_file(file, primary_metric="score")

            self.assertIsNotNone(parsed)
            assert parsed is not None
            metrics = parsed["metrics"]
            self.assertIs(metrics["suspect_protocol"], True)
            self.assertNotIn("suspect_fixed_weight_eval", metrics)


class IngestFindingsDirectoryPrimaryMetricTest(unittest.TestCase):
    """End-to-end: scan a directory with a nested-metric finding, verify
    the SQLite row carries ``metrics[primary_metric]`` so the variant
    filter and frontier.promote both accept it.
    """

    def setUp(self) -> None:
        # Each ingest test gets its own SQLite file so we don't bleed
        # rows across cases (init_db reads $LOCAL_STORE_DIR).
        import os
        from unittest.mock import patch

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._env = patch.dict(
            os.environ,
            {"LOCAL_STORE_DIR": self._tmp.name},
            clear=False,
        )
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_ingested_row_carries_primary_metric_after_hoist(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        root = Path(self._tmp.name)
        findings_dir = root / "findings"
        findings_dir.mkdir()
        (findings_dir / "gen0_peer0_result.json").write_text(
            json.dumps(
                {
                    "finding_type": "result",
                    "title": "Variant A",
                    "content": "experimental result",
                    "variant_name": "variant_A",
                    "final_results": {"casam": {"verification_pass_rate": 0.91}},
                }
            ),
            encoding="utf-8",
        )

        touched = findings_ingest.ingest_findings_directory(
            findings_dir, primary_metric="verification_pass_rate"
        )
        self.assertEqual(touched, 1)

        # The row must round-trip through SQLite with the canonical key
        # populated — that's the contract the variant filter +
        # frontier.promote both depend on.
        rows = local_store.get_findings(generation_id=0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["metrics"].get("verification_pass_rate"), 0.91)

    def test_ingest_without_primary_metric_drops_nested_value(self) -> None:
        """Legacy contract: when the orchestrator hasn't forwarded a
        primary_metric, the row is still ingested but the nested value
        isn't hoisted — confirms the new path is strictly opt-in.
        """
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        root = Path(self._tmp.name)
        findings_dir = root / "findings"
        findings_dir.mkdir()
        (findings_dir / "gen0_peer0_result.json").write_text(
            json.dumps(
                {
                    "finding_type": "result",
                    "title": "Variant A",
                    "content": "experimental result",
                    "final_results": {"casam": {"verification_pass_rate": 0.91}},
                }
            ),
            encoding="utf-8",
        )

        touched = findings_ingest.ingest_findings_directory(findings_dir)
        self.assertEqual(touched, 1)

        rows = local_store.get_findings(generation_id=0)
        self.assertEqual(len(rows), 1)
        self.assertNotIn("verification_pass_rate", rows[0]["metrics"])

    def test_ingest_refreshes_cached_row_when_schema_preserves_promotion_metadata(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        root = Path(self._tmp.name)
        findings_dir = root / "findings"
        findings_dir.mkdir()
        finding_id = "88344bd4-643a-53a2-a219-407561513b4e"
        finding_path = findings_dir / f"{finding_id}_candidate_protocol.json"
        finding_path.write_text(
            json.dumps(
                {
                    "id": finding_id,
                    "finding_type": "result",
                    "title": "Protocol-passed candidate",
                    "content": "complete result",
                    "variant_name": "candidate_protocol",
                    "peer_id": "gen0_peer2",
                    "generation_id": 0,
                    "metrics": {
                        "score": 2.5,
                        "frontier_lane": "alpha_incubator",
                        "scored_complete": True,
                        "protocol_integrity_status": "passed",
                        "tier_reached": "T1",
                    },
                }
            ),
            encoding="utf-8",
        )
        mtime_ns = finding_path.stat().st_mtime_ns
        local_store.init_db()
        local_store.insert_finding(
            {
                "id": finding_id,
                "finding_type": "result",
                "title": "Stale protocol candidate",
                "content": "old row",
                "variant_name": "candidate_protocol",
                "peer_id": "gen0_peer2",
                "generation_id": 0,
                "metrics": {
                    "score": 2.5,
                    "frontier_lane": "alpha_incubator",
                    "scored_complete": True,
                },
                "source_filepath": str(finding_path),
                "source_filename": finding_path.name,
                "source_mtime_ns": mtime_ns,
            }
        )

        touched = findings_ingest.ingest_findings_directory(findings_dir, primary_metric="score")

        self.assertEqual(touched, 1)
        rows = local_store.get_findings(generation_id=0)
        self.assertEqual(len(rows), 1)
        metrics = rows[0]["metrics"]
        self.assertEqual(metrics["protocol_integrity_status"], "passed")
        self.assertEqual(metrics["tier_reached"], "T1")
        self.assertTrue(metrics["scored_complete"])
        self.assertEqual(
            findings_ingest.ingest_findings_directory(findings_dir, primary_metric="score"),
            0,
        )

    def test_ingest_updates_non_uuid_declared_existing_id_with_promotion_metadata(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        root = Path(self._tmp.name)
        findings_dir = root / "findings"
        findings_dir.mkdir()
        finding_id = "candidate-protocol"
        finding_path = findings_dir / "gen0_peer2_candidate_protocol.json"
        finding_path.write_text(
            json.dumps(
                {
                    "id": finding_id,
                    "finding_type": "result",
                    "title": "Protocol-passed candidate",
                    "content": "complete result",
                    "variant_name": "candidate_protocol",
                    "peer_id": "gen0_peer2",
                    "generation_id": 0,
                    "metrics": {
                        "score": 2.5,
                        "frontier_lane": "alpha_incubator",
                        "scored_complete": True,
                        "protocol_integrity_status": "passed",
                        "tier_reached": "T1",
                    },
                }
            ),
            encoding="utf-8",
        )
        local_store.init_db()
        local_store.insert_finding(
            {
                "id": finding_id,
                "finding_type": "result",
                "title": "Stale protocol candidate",
                "content": "old row",
                "variant_name": "candidate_protocol",
                "peer_id": "gen0_peer2",
                "generation_id": 0,
                "metrics": {
                    "score": 2.5,
                    "frontier_lane": "alpha_incubator",
                    "scored_complete": True,
                },
            }
        )

        touched = findings_ingest.ingest_findings_directory(findings_dir, primary_metric="score")

        self.assertEqual(touched, 1)
        rows = local_store.get_findings(generation_id=0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], finding_id)
        self.assertFalse(rows[0]["id"].startswith("fs_"))
        metrics = rows[0]["metrics"]
        self.assertEqual(metrics["protocol_integrity_status"], "passed")
        self.assertEqual(metrics["tier_reached"], "T1")
        self.assertEqual(
            findings_ingest.ingest_findings_directory(findings_dir, primary_metric="score"),
            0,
        )

    def test_same_id_merge_preserves_missing_metadata_but_honors_false_and_zero(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        local_store.init_db()
        local_store.insert_finding(
            {
                "id": "same-id",
                "finding_type": "result",
                "title": "Initial",
                "content": "initial row",
                "variant_name": "variant",
                "peer_id": "gen0_peer1",
                "generation_id": 0,
                "metrics": {
                    "score": 1.0,
                    "scored_complete": True,
                    "protocol_integrity_status": "passed",
                    "tier_reached": "T1",
                    "old_signal": 3.0,
                },
            }
        )
        local_store.insert_finding(
            {
                "id": "same-id",
                "finding_type": "result",
                "title": "Updated",
                "content": "updated row",
                "variant_name": "variant",
                "peer_id": "gen0_peer1",
                "generation_id": 0,
                "metrics": {
                    "score": 0,
                    "scored_complete": False,
                },
            }
        )

        rows = local_store.get_findings(generation_id=0)
        self.assertEqual(len(rows), 1)
        metrics = rows[0]["metrics"]
        self.assertEqual(metrics["score"], 0)
        self.assertIs(metrics["scored_complete"], False)
        self.assertEqual(metrics["protocol_integrity_status"], "passed")
        self.assertEqual(metrics["tier_reached"], "T1")
        self.assertEqual(metrics["old_signal"], 3.0)

    def test_schema_upgrade_refreshes_stale_snapshot_state_for_same_source_file(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        local_store.init_db()
        source_path = "/tmp/findings/gen0_peer1_candidate.json"
        local_store.insert_finding(
            {
                "id": "schema-upgrade-id",
                "finding_type": "result",
                "title": "Old snapshot",
                "variant_name": "candidate",
                "peer_id": "gen0_peer1",
                "generation_id": 0,
                "source_filepath": source_path,
                "ingest_schema_version": 3,
                "metrics": {
                    "score": 1.0,
                    "source_result_sha256": "stale-cross-container-sha",
                    "protocol_integrity_status": "passed",
                    "validation_only_result": True,
                    "excluded_from_durable_frontier": True,
                    "exclusion_reason": "preliminary_or_incomplete_evidence",
                    "promotion_eligible": False,
                    "evidence_stage": "preliminary",
                    "partial_cohort": True,
                    "source_generation_low_confidence": True,
                    "effort_ratio": 0.2,
                    "coverage_ratio": 0.2,
                    "actual_epochs": 1,
                    "required_epochs": 10,
                    "completed_eval_units": 1,
                    "total_eval_units": 10,
                    "lane": "validation",
                    "failed_unit_count": 1,
                    "n_hard_constraint_violations": 1,
                },
            }
        )
        local_store.insert_finding(
            {
                "id": "schema-upgrade-id",
                "finding_type": "result",
                "title": "Current snapshot",
                "variant_name": "candidate",
                "peer_id": "gen0_peer1",
                "generation_id": 0,
                "source_filepath": source_path,
                "ingest_schema_version": 4,
                "metrics": {
                    "score": 2.0,
                    "scored_complete": True,
                    "effort_ratio": 1.0,
                    "coverage_ratio": 1.0,
                },
            }
        )

        [row] = local_store.get_findings(generation_id=0)
        self.assertEqual(row["metrics"]["score"], 2.0)
        self.assertTrue(row["metrics"]["scored_complete"])
        self.assertEqual(row["metrics"]["effort_ratio"], 1.0)
        self.assertEqual(row["metrics"]["coverage_ratio"], 1.0)
        self.assertNotIn("source_result_sha256", row["metrics"])
        for stale_key in (
            "protocol_integrity_status",
            "validation_only_result",
            "excluded_from_durable_frontier",
            "exclusion_reason",
            "promotion_eligible",
            "evidence_stage",
            "partial_cohort",
            "source_generation_low_confidence",
            "actual_epochs",
            "required_epochs",
            "completed_eval_units",
            "total_eval_units",
            "lane",
            "failed_unit_count",
            "n_hard_constraint_violations",
        ):
            self.assertNotIn(stale_key, row["metrics"])

    def test_schema_upgrade_refreshes_relocated_same_source_file(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        local_store.init_db()
        local_store.insert_finding(
            {
                "id": "relocated-schema-upgrade-id",
                "finding_type": "result",
                "title": "Old snapshot",
                "variant_name": "candidate",
                "peer_id": "gen0_peer1",
                "generation_id": 0,
                "source_filepath": "/old/run/shared_findings/gen0_peer1_candidate.json",
                "ingest_schema_version": 2,
                "metrics": {
                    "score": 1.0,
                    "scored_complete": False,
                    "validation_only_result": True,
                    "partial": True,
                    "unrelated_measurement": 7.0,
                },
                "current_aggregate": {
                    "coverage_ratio": 0.2,
                    "partial_cohort": True,
                },
            }
        )
        local_store.insert_finding(
            {
                "id": "relocated-schema-upgrade-id",
                "finding_type": "result",
                "title": "Current snapshot",
                "variant_name": "candidate",
                "peer_id": "gen0_peer1",
                "generation_id": 0,
                "source_filepath": "/new/run/shared_findings/gen0_peer1_candidate.json",
                "ingest_schema_version": 4,
                "metrics": {
                    "score": 2.0,
                    "scored_complete": True,
                    "effort_ratio": 1.0,
                    "coverage_ratio": 1.0,
                },
                "current_aggregate": {"coverage_ratio": 1.0},
            }
        )

        [row] = local_store.get_findings(generation_id=0)
        self.assertEqual(
            row["source_filepath"],
            "/new/run/shared_findings/gen0_peer1_candidate.json",
        )
        self.assertEqual(row["metrics"]["score"], 2.0)
        self.assertTrue(row["metrics"]["scored_complete"])
        self.assertEqual(row["metrics"]["unrelated_measurement"], 7.0)
        self.assertNotIn("validation_only_result", row["metrics"])
        self.assertNotIn("partial", row["metrics"])
        self.assertEqual(row["current_aggregate"]["coverage_ratio"], 1.0)
        self.assertNotIn("partial_cohort", row["current_aggregate"])

    def test_schema_upgrade_stores_a_different_source_file_separately(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        local_store.init_db()
        local_store.insert_finding(
            {
                "id": "different-source-id",
                "finding_type": "result",
                "source_filepath": "/old/run/shared_findings/candidate-a.json",
                "metrics": {"score": 1.0, "validation_only_result": True},
            }
        )
        local_store.insert_finding(
            {
                "id": "different-source-id",
                "finding_type": "result",
                "source_filepath": "/new/run/shared_findings/candidate-b.json",
                "ingest_schema_version": 4,
                "metrics": {"score": 2.0, "scored_complete": True},
            }
        )

        rows = local_store.get_findings()
        self.assertEqual(len(rows), 2)
        by_source = {row["source_filepath"]: row for row in rows}
        old = by_source["/old/run/shared_findings/candidate-a.json"]
        new = by_source["/new/run/shared_findings/candidate-b.json"]
        self.assertEqual(old["metrics"]["score"], 1.0)
        self.assertTrue(old["metrics"]["validation_only_result"])
        self.assertEqual(new["metrics"]["score"], 2.0)
        self.assertTrue(new["metrics"]["scored_complete"])
        self.assertNotIn("validation_only_result", new["metrics"])

        local_store.insert_finding(
            {
                "id": "different-source-id",
                "finding_type": "result",
                "source_filepath": "/new/run/shared_findings/candidate-b.json",
                "ingest_schema_version": 4,
                "metrics": {"score": 3.0, "scored_complete": True},
            }
        )
        rows = local_store.get_findings()
        self.assertEqual(len(rows), 2)
        by_source = {row["source_filepath"]: row for row in rows}
        self.assertEqual(
            by_source["/new/run/shared_findings/candidate-b.json"]["metrics"]["score"],
            3.0,
        )

    def test_schema_upgrade_refreshes_stale_snapshot_on_first_source_attribution(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        local_store.init_db()
        local_store.insert_finding(
            {
                "id": "declared-id",
                "finding_type": "result",
                "title": "Legacy direct row",
                "variant_name": "candidate",
                "peer_id": "gen0_peer1",
                "generation_id": 0,
                "metrics": {
                    "score": 1.0,
                    "child_id": "stale-child",
                    "canonical_variant_id": "shared-family",
                    "validation_only_result": True,
                    "excluded_from_durable_frontier": True,
                    "unrelated_measurement": 7.0,
                },
            }
        )
        local_store.insert_finding(
            {
                "id": "declared-id",
                "finding_type": "result",
                "title": "Filesystem complete row",
                "variant_name": "candidate",
                "peer_id": "gen0_peer1",
                "generation_id": 0,
                "source_filepath": "/tmp/findings/candidate.json",
                "ingest_schema_version": 4,
                "metrics": {
                    "score": 2.0,
                    "scored_complete": True,
                    "effort_ratio": 1.0,
                    "coverage_ratio": 1.0,
                },
            }
        )

        [row] = local_store.get_findings(generation_id=0)
        self.assertEqual(row["source_filepath"], "/tmp/findings/candidate.json")
        self.assertTrue(row["metrics"]["scored_complete"])
        self.assertEqual(row["metrics"]["unrelated_measurement"], 7.0)
        self.assertEqual(row["metrics"]["canonical_variant_id"], "shared-family")
        self.assertNotIn("child_id", row["metrics"])
        self.assertNotIn("validation_only_result", row["metrics"])
        self.assertNotIn("excluded_from_durable_frontier", row["metrics"])

    def test_schema_upgrade_replaces_artifact_coordinates_atomically(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        local_store.init_db()
        source_path = "/tmp/findings/gen0_peer1_candidate.json"
        local_store.insert_finding(
            {
                "id": "coordinate-upgrade-id",
                "finding_type": "result",
                "source_filepath": source_path,
                "metrics": {
                    "source_result_path": "results/old.json",
                    "source_result_sha256": "old-sha",
                    "score": 1.0,
                },
                "details": {"result_path": "results/details-old.json"},
                "extra": {
                    "summary_path": "results/extra-old.json",
                    "extra": {"source_result_sha256": "nested-old-sha"},
                },
                "current_aggregate": {
                    "child_id": "old-child",
                    "canonical_variant_id": "shared-family",
                    "source_result_path": "results/aggregate-old.json",
                    "source_result_sha256": "aggregate-old-sha",
                    "unrelated_measurement": 7.0,
                },
            }
        )
        local_store.insert_finding(
            {
                "id": "coordinate-upgrade-id",
                "finding_type": "result",
                "source_filepath": source_path,
                "ingest_schema_version": 4,
                "metrics": {"source_result_sha256": "new-sha", "score": 2.0},
                "current_aggregate": {"child_id": "new-child"},
            }
        )

        [row] = local_store.get_all_findings()
        self.assertEqual(row["metrics"]["source_result_sha256"], "new-sha")
        self.assertNotIn("source_result_path", row["metrics"])
        self.assertNotIn("result_path", row["details"])
        self.assertNotIn("summary_path", row["extra"])
        self.assertNotIn("source_result_sha256", row["extra"]["extra"])
        self.assertEqual(row["current_aggregate"]["child_id"], "new-child")
        self.assertEqual(row["current_aggregate"]["canonical_variant_id"], "shared-family")
        self.assertEqual(row["current_aggregate"]["unrelated_measurement"], 7.0)
        self.assertNotIn("source_result_path", row["current_aggregate"])
        self.assertNotIn("source_result_sha256", row["current_aggregate"])

    def test_auto_materialized_same_id_replaces_previous_derived_snapshot(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        local_store.init_db()
        local_store.insert_finding(
            {
                "id": "auto-result-id",
                "finding_type": "result",
                "title": "Preliminary",
                "variant_name": "candidate",
                "peer_id": "gen0_result_artifact",
                "generation_id": 0,
                "metrics": {
                    "auto_materialized_from_result_artifact": True,
                    "source_result_path": "results/candidate/summary.json",
                    "source_result_sha256": "preliminary-sha",
                    "validation_only_result": True,
                    "excluded_from_durable_frontier": True,
                    "exclusion_reason": "preliminary_or_incomplete_evidence",
                    "promotion_eligible": False,
                },
            }
        )
        local_store.insert_finding(
            {
                "id": "auto-result-id",
                "finding_type": "result",
                "title": "Complete",
                "variant_name": "candidate",
                "peer_id": "gen0_result_artifact",
                "generation_id": 0,
                "metrics": {
                    "auto_materialized_from_result_artifact": True,
                    "source_result_path": "results/candidate/summary.json",
                    "source_result_sha256": "complete-sha",
                    "effort_ratio": 1.0,
                    "coverage_ratio": 1.0,
                    "scored_complete": True,
                },
            }
        )

        [row] = local_store.get_findings(generation_id=0)
        metrics = row["metrics"]
        self.assertEqual(metrics["source_result_sha256"], "complete-sha")
        self.assertTrue(metrics["scored_complete"])
        self.assertNotIn("validation_only_result", metrics)
        self.assertNotIn("excluded_from_durable_frontier", metrics)
        self.assertNotIn("exclusion_reason", metrics)
        self.assertNotIn("promotion_eligible", metrics)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
