from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from praxist.plugins.workflow_stages.research_loop.backend.effective_config import (
    has_effective_config_metadata,
    result_effective_config_metadata,
)
from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
    _existing_materialized_results,
    _json_digest,
    _late_generation_boundary_info,
    _materialize_result_artifacts,
    _result_summary_metrics,
    collect_findings_for_generation,
    normalized_result_summary,
    result_source_snapshot_with_cutoff,
    result_summary_control_digest,
)
from praxist.plugins.workflow_stages.research_loop.backend.prompt_context import (
    _compact_frontier_entry,
)
from praxist.plugins.workflow_stages.research_loop.backend.research_memory.card_builder import (
    _safe_categorical_map,
    build_card_from_finding,
)
from praxist.plugins.workflow_stages.research_loop.backend.run_report import (
    _collect_shared_finding_entries,
    _lineage_row,
)
from praxist.plugins.workflow_stages.research_loop.backend.tools.findings_ingest import (
    extract_metrics,
    parse_finding_file,
)


def _digest(value: dict[str, object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class EffectiveConfigMetadataTest(unittest.TestCase):
    def test_compact_metadata_detection_is_opt_in(self) -> None:
        self.assertFalse(has_effective_config_metadata({"score": 1.0}))
        self.assertTrue(
            has_effective_config_metadata(
                [{"metrics": {"source_result_effective_config_status": "complete"}}]
            )
        )

    def test_legacy_summary_adds_no_metadata(self) -> None:
        self.assertEqual(result_effective_config_metadata({"score": 1.0}), {})

        metrics = _result_summary_metrics(
            {"current_aggregate": {"score": 1.0, "scored_complete": True}}
        )
        self.assertFalse(any("effective_config" in key for key in metrics))

    def test_digest_is_order_independent_and_treatment_sensitive(self) -> None:
        first = {"optimizer": {"weight_decay": 0.1}, "seed": 7}
        reordered = {"seed": 7, "optimizer": {"weight_decay": 0.1}}
        changed = {"optimizer": {"weight_decay": 0.2}, "seed": 7}

        first_metadata = result_effective_config_metadata(
            {"effective_config": first, "effective_config_complete": True}
        )
        reordered_metadata = result_effective_config_metadata(
            {"effective_config": reordered, "effective_config_complete": True}
        )
        changed_metadata = result_effective_config_metadata(
            {"effective_config": changed, "effective_config_complete": True}
        )

        self.assertEqual(
            first_metadata["source_result_effective_config_sha256"],
            reordered_metadata["source_result_effective_config_sha256"],
        )
        self.assertNotEqual(
            first_metadata["source_result_effective_config_sha256"],
            changed_metadata["source_result_effective_config_sha256"],
        )
        self.assertEqual(first_metadata["source_result_effective_config_sha256"], _digest(first))
        self.assertTrue(first_metadata["source_result_effective_config_complete"])
        self.assertEqual(first_metadata["source_result_effective_config_status"], "complete")

    def test_incomplete_and_invalid_declarations_remain_advisory(self) -> None:
        incomplete = result_effective_config_metadata(
            {
                "effective_config": {"seed": 7},
                "effective_config_complete": False,
                "effective_config_incomplete_reasons": ["runtime override not captured"],
            }
        )
        self.assertFalse(incomplete["source_result_effective_config_complete"])
        self.assertEqual(incomplete["source_result_effective_config_status"], "declared_incomplete")

        missing = result_effective_config_metadata(
            {"replication_of_effective_config_sha256": "a" * 64}
        )
        self.assertEqual(missing["source_result_effective_config_status"], "missing")
        self.assertEqual(
            missing["replication_effective_config_status"], "current_config_unverified"
        )

        invalid = result_effective_config_metadata(
            {"effective_config": {"loss": float("nan")}, "effective_config_complete": True}
        )
        self.assertEqual(invalid["source_result_effective_config_status"], "invalid")
        self.assertFalse(invalid["source_result_effective_config_complete"])

    def test_exact_replication_requires_complete_digest_match(self) -> None:
        config = {"budget": {"epochs": 20}, "seed": 3}
        parent_digest = _digest(config)
        matched = result_effective_config_metadata(
            {
                "effective_config": config,
                "effective_config_complete": True,
                "replication_of_effective_config_sha256": parent_digest,
            }
        )
        self.assertTrue(matched["replication_effective_config_match"])
        self.assertEqual(matched["replication_effective_config_status"], "matched")

        mismatch = result_effective_config_metadata(
            {
                "effective_config": {"budget": {"epochs": 10}, "seed": 3},
                "effective_config_complete": True,
                "replication_of_effective_config_sha256": parent_digest,
            }
        )
        self.assertFalse(mismatch["replication_effective_config_match"])
        self.assertEqual(mismatch["replication_effective_config_status"], "mismatch")

        invalid_parent = result_effective_config_metadata(
            {
                "effective_config": config,
                "effective_config_complete": True,
                "replication_of_effective_config_sha256": "not-a-digest",
            }
        )
        self.assertNotIn("replication_effective_config_match", invalid_parent)
        self.assertEqual(
            invalid_parent["replication_effective_config_status"], "invalid_parent_digest"
        )


class EffectiveConfigProjectionTest(unittest.TestCase):
    def setUp(self) -> None:
        config = {"protocol": {"epochs": 20}, "seed": 11}
        self.parent_digest = _digest(config)
        self.summary = {
            "variant_name": "candidate",
            "current_aggregate": {"score": 0.8, "scored_complete": True},
            "effective_config": config,
            "effective_config_complete": True,
            "replication_of_effective_config_sha256": self.parent_digest,
        }

    def test_summary_projection_keeps_digest_not_full_config(self) -> None:
        metrics = _result_summary_metrics(self.summary)
        metadata = result_effective_config_metadata(self.summary)
        self.assertNotIn("source_result_effective_config_sha256", metrics)
        self.assertEqual(metadata["source_result_effective_config_sha256"], self.parent_digest)
        self.assertEqual(metadata["replication_effective_config_status"], "matched")
        self.assertNotIn("effective_config", metrics)

    def test_replication_mismatch_does_not_change_existing_routing_facts(self) -> None:
        base_summary = {
            "current_aggregate": {"score": 0.8, "scored_complete": True},
            "promotion_eligible": True,
        }
        legacy_metrics = _result_summary_metrics(base_summary)
        mismatch_metrics = _result_summary_metrics(
            {
                **base_summary,
                "effective_config": {"protocol": {"epochs": 10}, "seed": 11},
                "effective_config_complete": True,
                "replication_of_effective_config_sha256": self.parent_digest,
            }
        )

        for key, value in legacy_metrics.items():
            self.assertEqual(mismatch_metrics[key], value)
        self.assertTrue(mismatch_metrics["promotion_eligible"])
        self.assertNotIn("replication_effective_config_match", mismatch_metrics)
        mismatch_metadata = result_effective_config_metadata(
            {
                **base_summary,
                "effective_config": {"protocol": {"epochs": 10}, "seed": 11},
                "effective_config_complete": True,
                "replication_of_effective_config_sha256": self.parent_digest,
            }
        )
        self.assertFalse(mismatch_metadata["replication_effective_config_match"])
        self.assertEqual(mismatch_metadata["replication_effective_config_status"], "mismatch")

    def test_existing_ingest_prompt_card_and_report_paths_preserve_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            summary_path = run_dir / "results/candidate/result_summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(json.dumps(self.summary), encoding="utf-8")
            materialized = _materialize_result_artifacts(run_dir=run_dir, gen_id=1)
            self.assertEqual(len(materialized), 1)
            finding_path = next((run_dir / "shared_findings").glob("*.json"))
            parsed = parse_finding_file(finding_path)

        self.assertIsNotNone(parsed)
        ingested = parsed["metrics"]
        self.assertEqual(ingested["source_result_effective_config_sha256"], self.parent_digest)
        self.assertTrue(ingested["source_result_effective_config_complete"])
        self.assertEqual(ingested["replication_effective_config_status"], "matched")
        metrics = ingested

        compact = _compact_frontier_entry({"variant_name": "candidate", "metrics": metrics}, None)
        self.assertEqual(
            compact["metrics"]["source_result_effective_config_sha256"], self.parent_digest
        )

        categories = _safe_categorical_map({}, metrics, {})
        self.assertEqual(categories["source_result_effective_config_sha256"], self.parent_digest)
        self.assertTrue(categories["source_result_effective_config_complete"])

        card = build_card_from_finding(
            {
                "id": "finding-1",
                "variant_name": "candidate",
                "metrics": metrics,
                "generation_id": 1,
            },
            run_dir=Path("/tmp/nonexistent-run"),
        )
        self.assertEqual(card["source_ref"]["source_result_path"], metrics["source_result_path"])

        legacy_card = build_card_from_finding(
            {
                "id": "legacy-finding",
                "variant_name": "candidate",
                "metrics": {"score": 0.8, "source_result_path": "results/legacy.json"},
            },
            run_dir=Path("/tmp/nonexistent-run"),
        )
        self.assertNotIn("source_result_path", legacy_card["source_ref"])

        lineage = _lineage_row({"variant_name": "candidate", "metrics": metrics})
        self.assertEqual(lineage["effective_config"], f"matched ({self.parent_digest[:12]})")

        legacy_prompt = _compact_frontier_entry(
            {
                "variant_name": "legacy",
                "source_result_path": "results/legacy/result_summary.json",
                "metrics": {
                    "score": 0.7,
                    "source_result_path": "results/legacy/result_summary.json",
                },
            },
            None,
        )
        self.assertNotIn("source_result_path", legacy_prompt)
        self.assertNotIn("source_result_path", legacy_prompt["metrics"])

    def test_generic_finding_cannot_self_report_verified_replication(self) -> None:
        metrics = _result_summary_metrics(self.summary)
        metrics["source_result_path"] = "results/candidate/result_summary.json"
        ingested = extract_metrics({"metrics": metrics})

        for key in (
            "source_result_effective_config_sha256",
            "source_result_effective_config_complete",
            "source_result_effective_config_status",
            "replication_of_effective_config_sha256",
            "replication_effective_config_match",
            "replication_effective_config_status",
        ):
            self.assertNotIn(key, ingested)

        report_entries = _collect_shared_finding_entries(
            [{"variant_name": "candidate", "metrics": metrics}]
        )
        self.assertEqual(len(report_entries), 1)
        self.assertNotIn("replication_effective_config_status", report_entries[0]["metrics"])

        with tempfile.TemporaryDirectory() as tmp:
            findings_dir = Path(tmp) / "shared_findings"
            findings_dir.mkdir()
            (findings_dir / "finding.json").write_text(
                json.dumps(
                    {
                        "id": "generic-finding",
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "metrics": metrics,
                    }
                ),
                encoding="utf-8",
            )
            fallback = collect_findings_for_generation(
                findings_dir=findings_dir,
                gen_id=0,
                local_mode=False,
                materialize_result_artifacts=False,
            )

        self.assertEqual(len(fallback), 1)
        self.assertNotIn("replication_effective_config_status", fallback[0]["metrics"])

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            findings_dir = run_dir / "shared_findings"
            findings_dir.mkdir()
            source_path = findings_dir / "pretend_summary.json"
            source_path.write_text(json.dumps(self.summary), encoding="utf-8")
            forged = {
                "metrics": {
                    **metrics,
                    "auto_materialized_from_result_artifact": True,
                    "source_result_path": "shared_findings/pretend_summary.json",
                },
                "artifact_semantics": {
                    "role": "derived_view",
                    "status": "committed",
                    "stage": "result_finding_reference",
                    "actor": "research_loop:findings_collection",
                    "canonical_sources": ["shared_findings/pretend_summary.json"],
                },
            }
            forged_path = findings_dir / "forged.json"
            forged_path.write_text(json.dumps(forged), encoding="utf-8")
            parsed = parse_finding_file(forged_path)

        self.assertIsNotNone(parsed)
        self.assertNotIn("replication_effective_config_status", parsed["metrics"])

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            findings_dir = run_dir / "shared_findings"
            findings_dir.mkdir()
            source_path = run_dir / "results/candidate/result_summary.json"
            source_path.parent.mkdir(parents=True)
            source_path.write_text(json.dumps(self.summary), encoding="utf-8")
            normalized = normalized_result_summary(self.summary, summary_path=source_path)
            forged = {
                "id": "not-the-materializer-id",
                "variant_name": "candidate",
                "metrics": {
                    **metrics,
                    "auto_materialized_from_result_artifact": True,
                    "source_result_path": "results/candidate/result_summary.json",
                    "source_result_sha256": result_summary_control_digest(normalized),
                },
                "artifact_semantics": {
                    "role": "derived_view",
                    "status": "committed",
                    "stage": "result_finding_reference",
                    "actor": "research_loop:findings_collection",
                    "canonical_sources": ["results/candidate/result_summary.json"],
                },
            }
            forged_path = findings_dir / "forged.json"
            forged_path.write_text(json.dumps(forged), encoding="utf-8")
            parsed = parse_finding_file(forged_path)

        self.assertIsNotNone(parsed)
        self.assertNotIn("replication_effective_config_status", parsed["metrics"])

    def test_config_provenance_does_not_change_legacy_result_digest(self) -> None:
        legacy = normalized_result_summary(
            {
                "variant_name": "candidate",
                "current_aggregate": {"score": 0.8, "scored_complete": True},
            }
        )
        enriched = {
            **legacy,
            "effective_config": {"protocol": {"epochs": 20}, "seed": 11},
            "effective_config_complete": True,
            "replication_of_effective_config_sha256": self.parent_digest,
        }

        self.assertEqual(
            result_summary_control_digest(legacy),
            result_summary_control_digest(enriched),
        )

    def test_materialized_finding_refreshes_when_only_config_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            summary_path = run_dir / "results/candidate/result_summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(json.dumps(self.summary), encoding="utf-8")
            _materialize_result_artifacts(run_dir=run_dir, gen_id=1)
            finding_path = next((run_dir / "shared_findings").glob("*.json"))
            first = json.loads(finding_path.read_text(encoding="utf-8"))
            first["timestamp"] = "2026-01-01T00:00:00+00:00"
            finding_path.write_text(json.dumps(first), encoding="utf-8")

            changed = copy.deepcopy(self.summary)
            changed["effective_config"]["protocol"]["epochs"] = 10
            summary_path.write_text(json.dumps(changed), encoding="utf-8")
            _materialize_result_artifacts(run_dir=run_dir, gen_id=1)
            second = json.loads(finding_path.read_text(encoding="utf-8"))

        self.assertEqual(
            first["metrics"]["source_result_sha256"],
            second["metrics"]["source_result_sha256"],
        )
        self.assertEqual(first["metrics"]["replication_effective_config_status"], "matched")
        self.assertEqual(second["metrics"]["replication_effective_config_status"], "mismatch")
        self.assertEqual(second["timestamp"], first["timestamp"])

    def test_pre_feature_digest_migrates_without_reordering_finding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            summary_path = run_dir / "results/candidate/result_summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(json.dumps(self.summary), encoding="utf-8")
            _materialize_result_artifacts(run_dir=run_dir, gen_id=1)
            finding_path = next((run_dir / "shared_findings").glob("*.json"))
            finding = json.loads(finding_path.read_text(encoding="utf-8"))
            finding["timestamp"] = "2026-01-01T00:00:00+00:00"
            finding["metrics"]["source_result_sha256"] = _json_digest(
                normalized_result_summary(self.summary, summary_path=summary_path)
            )
            for key in tuple(finding["metrics"]):
                if "effective_config" in key:
                    finding["metrics"].pop(key)
            finding_path.write_text(json.dumps(finding), encoding="utf-8")

            _materialize_result_artifacts(run_dir=run_dir, gen_id=1)
            migrated = json.loads(finding_path.read_text(encoding="utf-8"))

        self.assertEqual(migrated["timestamp"], finding["timestamp"])
        self.assertEqual(
            migrated["metrics"]["source_result_sha256"],
            result_summary_control_digest(
                normalized_result_summary(self.summary, summary_path=summary_path)
            ),
        )
        self.assertEqual(migrated["metrics"]["replication_effective_config_status"], "matched")

    def test_frontier_canonical_source_rebuild_keeps_config_provenance(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.frontier import FrontierStore

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            summary_path = run_dir / "results/candidate/summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(json.dumps(self.summary), encoding="utf-8")
            store = FrontierStore(run_dir / "frontier", primary_metric="score")

            by_path, _by_variant = store._canonical_result_source_index()
            metrics = by_path["results/candidate/summary.json"]["candidate"]["metrics"]

        self.assertEqual(metrics["source_result_effective_config_sha256"], self.parent_digest)
        self.assertEqual(metrics["replication_effective_config_status"], "matched")

    def test_post_boundary_config_only_update_is_not_a_late_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            summary_path = run_dir / "results/candidate/result_summary.json"
            summary_path.parent.mkdir(parents=True)
            fieldless = {
                key: value
                for key, value in self.summary.items()
                if key
                not in {
                    "effective_config",
                    "effective_config_complete",
                    "effective_config_incomplete_reasons",
                    "replication_of_effective_config_sha256",
                }
            }
            summary_path.write_text(json.dumps(fieldless), encoding="utf-8")
            cutoff, snapshot = result_source_snapshot_with_cutoff(run_dir)
            prior_digest = result_summary_control_digest(
                normalized_result_summary(fieldless, summary_path=summary_path)
            )

            changed = copy.deepcopy(self.summary)
            changed["effective_config"]["protocol"]["epochs"] = 10
            summary_path.write_text(json.dumps(changed), encoding="utf-8")
            current_digest = result_summary_control_digest(
                normalized_result_summary(changed, summary_path=summary_path)
            )
            config_only = _late_generation_boundary_info(
                run_dir=run_dir,
                summary_path=summary_path,
                source_gen_id=1,
                evidence_cutoff=cutoff,
                evidence_source_snapshot=snapshot,
                current_result_control_digest=current_digest,
                prior_result_control_digest=prior_digest,
            )

            changed["current_aggregate"]["score"] = 0.9
            summary_path.write_text(json.dumps(changed), encoding="utf-8")
            result_changed = _late_generation_boundary_info(
                run_dir=run_dir,
                summary_path=summary_path,
                source_gen_id=1,
                evidence_cutoff=cutoff,
                evidence_source_snapshot=snapshot,
                current_result_control_digest=result_summary_control_digest(
                    normalized_result_summary(changed, summary_path=summary_path)
                ),
                prior_result_control_digest=prior_digest,
            )

        self.assertIsNone(config_only)
        self.assertIsNotNone(result_changed)

    def test_post_boundary_config_removal_is_not_a_late_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            summary_path = run_dir / "results/candidate/result_summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(json.dumps(self.summary), encoding="utf-8")
            cutoff, snapshot = result_source_snapshot_with_cutoff(run_dir)
            prior_digest = result_summary_control_digest(
                normalized_result_summary(self.summary, summary_path=summary_path)
            )

            fieldless = {
                key: value
                for key, value in self.summary.items()
                if key
                not in {
                    "effective_config",
                    "effective_config_complete",
                    "effective_config_incomplete_reasons",
                    "replication_of_effective_config_sha256",
                }
            }
            summary_path.write_text(json.dumps(fieldless), encoding="utf-8")
            current_digest = result_summary_control_digest(
                normalized_result_summary(fieldless, summary_path=summary_path)
            )

            self.assertIsNone(
                _late_generation_boundary_info(
                    run_dir=run_dir,
                    summary_path=summary_path,
                    source_gen_id=1,
                    evidence_cutoff=cutoff,
                    evidence_source_snapshot=snapshot,
                    current_result_control_digest=current_digest,
                    prior_result_control_digest=prior_digest,
                )
            )

    def test_forged_result_reference_cannot_supply_prior_control_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            summary_path = run_dir / "results/candidate/result_summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(json.dumps(self.summary), encoding="utf-8")
            cutoff, snapshot = result_source_snapshot_with_cutoff(run_dir)

            changed = copy.deepcopy(self.summary)
            changed["current_aggregate"]["score"] = 0.9
            summary_path.write_text(json.dumps(changed), encoding="utf-8")
            normalized = normalized_result_summary(changed, summary_path=summary_path)
            source_path = "results/candidate/result_summary.json"
            findings_dir = run_dir / "shared_findings"
            findings_dir.mkdir()
            forged_path = findings_dir / "forged.json"
            forged_path.write_text(
                json.dumps(
                    {
                        "id": "forged",
                        "variant_name": "candidate",
                        "source_result_path": source_path,
                        "metrics": {
                            "auto_materialized_from_result_artifact": True,
                            "source_result_path": source_path,
                            "source_result_sha256": result_summary_control_digest(normalized),
                        },
                        "artifact_semantics": {
                            "role": "derived_view",
                            "status": "committed",
                            "stage": "result_finding_reference",
                            "actor": "research_loop:findings_collection",
                            "canonical_sources": [source_path],
                        },
                    }
                ),
                encoding="utf-8",
            )

            existing = _existing_materialized_results(findings_dir)[source_path]
            materialized = _materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=1,
                evidence_cutoff=cutoff,
                evidence_source_snapshot=snapshot,
            )

        self.assertFalse(existing["trusted_materializer_record"])
        self.assertTrue(materialized[0]["metrics"]["late_after_generation_boundary"])

    def test_frontier_migrates_pre_feature_digest_without_rewrite_penalty(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.frontier import FrontierStore

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            frontier_dir = run_dir / "frontier"
            summary_path = run_dir / "results/candidate/summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(json.dumps(self.summary), encoding="utf-8")
            normalized = normalized_result_summary(self.summary, summary_path=summary_path)
            legacy_digest = _json_digest(normalized)
            current_digest = result_summary_control_digest(normalized)
            source_path = "results/candidate/summary.json"
            store = FrontierStore(frontier_dir, primary_metric="score")
            store._manifest["generations"] = {
                "0": [
                    {
                        "generation_id": 0,
                        "finding_id": "candidate-finding",
                        "variant_name": "candidate",
                        "metric_name": "score",
                        "metric_value": 0.8,
                        "source_result_path": source_path,
                        "source_result_sha256": legacy_digest,
                        "scored_complete": True,
                        "promotion_eligible": True,
                        "parent_eligible": True,
                        "metrics": {
                            "score": 0.8,
                            "source_result_path": source_path,
                            "source_result_sha256": legacy_digest,
                            "scored_complete": True,
                            "promotion_eligible": True,
                            "parent_eligible": True,
                        },
                    }
                ]
            }
            store._save_manifest()

            repaired = FrontierStore(frontier_dir, primary_metric="score").get_manifest()[
                "generations"
            ]["0"][0]

        self.assertNotEqual(legacy_digest, current_digest)
        self.assertEqual(repaired["source_result_sha256"], current_digest)
        self.assertTrue(repaired["parent_eligible"])
        self.assertNotEqual(repaired.get("exclusion_reason"), "source_rewritten_without_generation")

    def test_ingest_uses_materializer_maturity_policy_for_digest_validation(self) -> None:
        policy = {"complete_stage_labels": ["full"]}
        summary = {
            "variant_name": "candidate",
            "tiers": [
                {
                    "tier": "full",
                    "status": "ok",
                    "evidence_stage": "full",
                    "metrics_summary": {"score": 0.8},
                }
            ],
            "effective_config": {"protocol": "full", "seed": 11},
            "effective_config_complete": True,
        }
        summary["replication_of_effective_config_sha256"] = _digest(summary["effective_config"])

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            summary_path = run_dir / "results/candidate/result_summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            _materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=1,
                result_maturity_policy=policy,
            )
            finding_path = next((run_dir / "shared_findings").glob("*.json"))
            without_policy = parse_finding_file(finding_path)
            with_policy = parse_finding_file(
                finding_path,
                result_maturity_policy=policy,
            )

        self.assertIsNotNone(without_policy)
        self.assertNotIn("replication_effective_config_status", without_policy["metrics"])
        self.assertIsNotNone(with_policy)
        self.assertEqual(with_policy["metrics"]["replication_effective_config_status"], "matched")

    def test_share_finding_strips_reserved_fields_before_dual_write(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter

        saved: list[dict[str, object]] = []
        post = AsyncMock()
        reserved = {
            "replication_effective_config_status": "matched",
            "nested": {
                "effective_config": {"seed": 11},
                "source_result_effective_config_complete": True,
            },
        }
        with (
            patch.dict(
                os.environ,
                {"LOCAL_MODE": "false", "LOCAL_FINDINGS_DIR": "/tmp/findings"},
                clear=False,
            ),
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend.tools.findings_sync.save_finding_to_dir",
                side_effect=lambda finding, _directory: saved.append(copy.deepcopy(finding)),
            ),
            patch.object(adapter, "get_server_url", return_value="http://example.invalid"),
            patch.object(adapter, "async_http_post", post),
        ):
            asyncio.run(
                adapter._handle_share_finding(
                    {
                        "finding_type": "result",
                        "title": "candidate",
                        "content": "result",
                        "metrics": copy.deepcopy(reserved),
                        "extra": copy.deepcopy(reserved),
                        "peer_id": "gen0_peer0",
                    }
                )
            )

        self.assertEqual(len(saved), 1)
        posted = post.await_args.kwargs["json_data"]
        for finding in (saved[0], posted):
            serialized = json.dumps(finding)
            self.assertNotIn("effective_config", serialized)
            self.assertNotIn("replication_effective_config_status", serialized)

    def test_nested_result_fields_cannot_override_computed_verification(self) -> None:
        metrics = _result_summary_metrics(
            {
                "current_aggregate": {
                    "score": 0.8,
                    "replication_effective_config_status": "matched",
                    "replication_effective_config_match": True,
                }
            }
        )

        self.assertNotIn("replication_effective_config_status", metrics)
        self.assertNotIn("replication_effective_config_match", metrics)

        with tempfile.TemporaryDirectory() as tmp:
            finding_path = Path(tmp) / "shared_findings" / "finding.json"
            finding_path.parent.mkdir()
            finding_path.write_text(
                json.dumps(
                    {
                        "metrics": {
                            "score": 0.8,
                            "current_aggregate": {
                                "replication_effective_config_status": "matched",
                                "nested": {
                                    "replication_effective_config_match": True,
                                },
                            },
                        },
                        "extra": {"replication_effective_config_status": "matched"},
                        "details": {"source_result_effective_config_complete": True},
                        "current_aggregate": {
                            "source_result_effective_config_sha256": self.parent_digest
                        },
                    }
                ),
                encoding="utf-8",
            )
            parsed = parse_finding_file(finding_path)

        self.assertIsNotNone(parsed)
        nested = parsed["metrics"]["current_aggregate"]
        self.assertNotIn("replication_effective_config_status", nested)
        self.assertNotIn("replication_effective_config_match", nested["nested"])
        self.assertNotIn("replication_effective_config_status", parsed["extra"])
        self.assertNotIn("source_result_effective_config_complete", parsed["details"])
        self.assertNotIn("source_result_effective_config_sha256", parsed["current_aggregate"])


if __name__ == "__main__":
    unittest.main()
