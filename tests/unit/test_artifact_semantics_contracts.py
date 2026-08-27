from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from praxist.core.storage import ArtifactWriter
from praxist.plugins.tools.evaluation_tools.adapter import _envelope_leaderboard
from praxist.plugins.workflow_stages.research_loop.backend import pi_agent
from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
    CANONICAL_STATE,
    COMMITTED,
    DERIVED_AUDIT_SNAPSHOT,
    DERIVED_VIEW,
    FAILED,
    PARTIAL,
    SUPERSEDED,
    artifact_semantics,
    attach_artifact_semantics,
    explicit_entry_generation_id,
    has_explicit_artifact_semantics,
    is_audit_or_derived,
    is_committed_runtime_fact_source,
    is_readable_signal_source,
    is_runtime_fact_source,
)
from praxist.plugins.workflow_stages.research_loop.backend.frontier import FrontierStore
from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager
from praxist.plugins.workflow_stages.research_loop.backend.prompt_artifacts import (
    compact_artifact_ref,
    persist_prompt_layout_artifacts,
)
from praxist.plugins.workflow_stages.research_loop.backend.research_memory.context_firewall import (
    fit_pack_to_budget,
)
from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
    EvidencePack,
    _digest_lane_frontiers,
    _digest_validation_candidates,
    build_evidence_pack,
)
from praxist.plugins.workflow_stages.research_loop.backend.research_topology.api import (
    ResearchLoopModuleAPI,
)
from praxist.plugins.workflow_stages.research_loop.backend.resume_state import (
    _frontier_generations,
    inspect_resume_plan,
    write_boundary_marker,
)
from praxist.plugins.workflow_stages.research_loop.backend.tools.findings_ingest import (
    _trusted_declared_gem_id,
)


class ArtifactSemanticsContractsTest(unittest.TestCase):
    def test_explicit_entry_generation_uses_latest_recorded_provenance_only(self) -> None:
        entry = {
            "variant_name": "nextgen99_is_just_a_name",
            "generation_id": 0,
            "metrics": {
                "generation_id": 1,
                "source_generation_id": 2,
            },
        }

        self.assertEqual(explicit_entry_generation_id(entry), 2)
        self.assertEqual(explicit_entry_generation_id(entry, generation_hint=3), 3)
        self.assertIsNone(
            explicit_entry_generation_id({"variant_name": "nextgen99_is_just_a_name"})
        )

    def test_semantics_distinguish_canonical_state_from_derived_views(self) -> None:
        canonical = attach_artifact_semantics(
            {"value": 1},
            role=CANONICAL_STATE,
            stage="generation_boundary",
            generation_id=2,
        )
        derived = attach_artifact_semantics(
            {"value": 2},
            role=DERIVED_VIEW,
            stage="leaderboard_tool_response",
            runtime_fact_source=False,
        )

        self.assertTrue(is_runtime_fact_source(canonical))
        self.assertFalse(is_audit_or_derived(canonical))
        self.assertFalse(is_runtime_fact_source(derived))
        self.assertTrue(is_audit_or_derived(derived))
        self.assertEqual(derived["artifact_semantics"]["status"], COMMITTED)

    def test_semantics_helpers_reject_non_dicts_and_malformed_metadata(self) -> None:
        malformed = {"artifact_semantics": "not-a-dict"}
        self.assertFalse(is_runtime_fact_source(None))
        self.assertFalse(is_runtime_fact_source({}))
        self.assertFalse(is_runtime_fact_source(malformed))
        self.assertFalse(has_explicit_artifact_semantics(None))
        self.assertFalse(has_explicit_artifact_semantics(malformed))
        self.assertFalse(is_committed_runtime_fact_source(None, legacy_ok=True))
        self.assertTrue(is_committed_runtime_fact_source({}, legacy_ok=True))
        self.assertFalse(is_committed_runtime_fact_source({}, legacy_ok=False))
        self.assertFalse(is_readable_signal_source(None, legacy_ok=True))
        self.assertTrue(is_readable_signal_source({}, legacy_ok=True))
        self.assertFalse(is_readable_signal_source({}, legacy_ok=False))
        self.assertFalse(is_audit_or_derived(None))
        self.assertFalse(is_audit_or_derived(malformed))

    def test_artifact_writer_indexes_role_status_and_source_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            writer = ArtifactWriter(run_dir)

            artifact = writer.persist_json(
                "prompt.layout_manifest",
                "prompts/gen_0/gen0_peer0_layout.json",
                {"schema_version": "x"},
                schema_ref="praxist.prompt_layout.v1",
                producer={"stage_id": "research_loop", "role_ref": "peer:gen0_peer0"},
                artifact_role=DERIVED_AUDIT_SNAPSHOT,
                artifact_status=COMMITTED,
                runtime_fact_source=False,
                derived_from=["art_000001"],
            )

            metadata = json.loads(
                (run_dir / artifact["payload_path"]).parent.joinpath("metadata.json").read_text()
            )
            index_record = json.loads((run_dir / "artifact_index.jsonl").read_text())
            for record in (metadata, index_record, artifact):
                self.assertEqual(record["artifact_role"], DERIVED_AUDIT_SNAPSHOT)
                self.assertEqual(record["artifact_status"], COMMITTED)
                self.assertFalse(record["runtime_fact_source"])
                self.assertEqual(record["derived_from"], ["art_000001"])

            compact = compact_artifact_ref({**artifact, "secret": "drop"})
            self.assertEqual(compact["artifact_role"], DERIVED_AUDIT_SNAPSHOT)
            self.assertNotIn("secret", compact)

    def test_prompt_layout_artifacts_are_audit_snapshots_not_runtime_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest_path = run_dir / "prompts" / "gen_0" / "gen0_peer0_layout.json"
            prompt_path = run_dir / "prompts" / "gen_0" / "gen0_peer0.md"

            manifest = persist_prompt_layout_artifacts(
                run_dir=run_dir,
                prompt_text="prompt",
                prompt_path=prompt_path,
                manifest={"schema_version": "layout"},
                manifest_path=manifest_path,
                peer_id="gen0_peer0",
                gen_id=0,
            )

            self.assertEqual(
                manifest["artifact_semantics"]["role"],
                DERIVED_AUDIT_SNAPSHOT,
            )
            self.assertIn("created_at", manifest["artifact_semantics"])
            self.assertFalse(manifest["artifact_semantics"]["runtime_fact_source"])
            self.assertFalse(manifest["rendered_prompt_ref"]["runtime_fact_source"])
            records = [
                json.loads(line)
                for line in (run_dir / "artifact_index.jsonl").read_text().splitlines()
            ]
            self.assertEqual(records[0]["artifact_role"], "audit_snapshot")
            self.assertEqual(records[1]["artifact_role"], DERIVED_AUDIT_SNAPSHOT)
            self.assertFalse(records[0]["runtime_fact_source"])
            self.assertFalse(records[1]["runtime_fact_source"])

    def test_pi_agenda_loader_ignores_explicit_failed_agendas_but_keeps_legacy_compat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            agendas = run_dir / "agendas"
            agendas.mkdir()
            legacy = {
                "generation": 1,
                "peer_contracts": {},
                "mainline_observation": {},
                "cross_peer_hypotheses": [],
            }
            (agendas / "research_agenda_gen1.yaml").write_text(
                yaml.safe_dump(legacy),
                encoding="utf-8",
            )
            failed = attach_artifact_semantics(
                {
                    "generation": 2,
                    "peer_contracts": {},
                    "mainline_observation": {},
                    "cross_peer_hypotheses": [],
                },
                role=DERIVED_VIEW,
                status=FAILED,
                stage="pi_agenda",
                runtime_fact_source=False,
            )
            (agendas / "research_agenda_gen2.yaml").write_text(
                yaml.safe_dump(failed),
                encoding="utf-8",
            )

            self.assertEqual(pi_agent.load_agenda_for_gen(run_dir, 1)["generation"], 1)
            self.assertIsNone(pi_agent.load_agenda_for_gen(run_dir, 2))

    def test_resume_requires_committed_next_agenda_when_inferring_from_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_generation_results(run_dir, 0)
            self._write_frontier_manifest(run_dir, {"0": [{"variant_name": "v0"}]})
            agenda = attach_artifact_semantics(
                {
                    "generation": 1,
                    "peer_contracts": {},
                    "mainline_observation": {},
                    "cross_peer_hypotheses": [],
                },
                role=DERIVED_VIEW,
                status=FAILED,
                stage="pi_agenda",
                runtime_fact_source=False,
            )
            agenda_path = run_dir / "agendas" / "research_agenda_gen1.yaml"
            agenda_path.parent.mkdir()
            agenda_path.write_text(yaml.safe_dump(agenda), encoding="utf-8")

            plan = inspect_resume_plan(run_dir, max_generations=3, pi_enabled=True)

            self.assertEqual(plan.start_generation, 0)
            self.assertEqual(plan.pending_boundary_generation, 0)

    def test_boundary_marker_is_canonical_runtime_fact_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_boundary_marker(run_dir, gen_id=3, promoted_count=2, pi_status="succeeded")

            marker = json.loads(
                (run_dir / "gen_3" / "generation_boundary.json").read_text(encoding="utf-8")
            )
            semantics = marker["artifact_semantics"]
            self.assertEqual(semantics["role"], CANONICAL_STATE)
            self.assertTrue(semantics["runtime_fact_source"])
            self.assertEqual(semantics["generation_id"], 3)
            self.assertNotIn("gems/gems_state.json", semantics["canonical_sources"])

            gems_dir = run_dir / "gems"
            gems_dir.mkdir()
            (gems_dir / "gems_state.json").write_text(
                json.dumps(
                    attach_artifact_semantics(
                        {"enabled": True},
                        role=CANONICAL_STATE,
                        status=PARTIAL,
                        stage="gems_state",
                        runtime_fact_source=False,
                    )
                ),
                encoding="utf-8",
            )
            write_boundary_marker(run_dir, gen_id=4, promoted_count=0, pi_status="succeeded")
            partial_marker = json.loads(
                (run_dir / "gen_4" / "generation_boundary.json").read_text(encoding="utf-8")
            )
            self.assertNotIn(
                "gems/gems_state.json",
                partial_marker["artifact_semantics"]["canonical_sources"],
            )

            (gems_dir / "gems_state.json").write_text(
                json.dumps(
                    attach_artifact_semantics(
                        {"enabled": True},
                        role=CANONICAL_STATE,
                        status=COMMITTED,
                        stage="gems_state",
                        runtime_fact_source=True,
                    )
                ),
                encoding="utf-8",
            )
            write_boundary_marker(run_dir, gen_id=5, promoted_count=1, pi_status="succeeded")
            committed_marker = json.loads(
                (run_dir / "gen_5" / "generation_boundary.json").read_text(encoding="utf-8")
            )
            self.assertIn(
                "gems/gems_state.json",
                committed_marker["artifact_semantics"]["canonical_sources"],
            )

    def test_frontier_manifest_save_marks_existing_owner_as_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frontier_dir = Path(tmp) / "frontier"
            store = FrontierStore(frontier_dir)
            store._save_manifest()

            manifest = json.loads((frontier_dir / "frontier_manifest.json").read_text())
            semantics = manifest["artifact_semantics"]
            self.assertEqual(semantics["role"], CANONICAL_STATE)
            self.assertTrue(semantics["runtime_fact_source"])
            self.assertEqual(semantics["stage"], "frontier_manifest")
            self.assertNotIn("gems/gems_state.json", semantics["canonical_sources"])

            gems_dir = Path(tmp) / "gems"
            gems_dir.mkdir()
            (gems_dir / "gems_state.json").write_text(
                json.dumps(
                    attach_artifact_semantics(
                        {"enabled": True},
                        role=CANONICAL_STATE,
                        status=COMMITTED,
                        stage="gems_state",
                        runtime_fact_source=True,
                    )
                ),
                encoding="utf-8",
            )
            store._save_manifest()
            manifest = json.loads((frontier_dir / "frontier_manifest.json").read_text())
            self.assertIn(
                "gems/gems_state.json",
                manifest["artifact_semantics"]["canonical_sources"],
            )

    def test_gems_frontier_writer_only_cites_usable_gems_state(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import (
            _with_frontier_manifest_semantics,
        )

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "gems_state.json"
            without_state = _with_frontier_manifest_semantics(
                {},
                gems_state_path=state_path,
            )
            self.assertNotIn(
                "gems/gems_state.json",
                without_state["artifact_semantics"]["canonical_sources"],
            )

            state_path.write_text(
                json.dumps(
                    attach_artifact_semantics(
                        {"enabled": True},
                        role=CANONICAL_STATE,
                        status=PARTIAL,
                        stage="gems_state",
                        runtime_fact_source=False,
                    )
                ),
                encoding="utf-8",
            )
            partial_state = _with_frontier_manifest_semantics(
                {},
                gems_state_path=state_path,
            )
            self.assertNotIn(
                "gems/gems_state.json",
                partial_state["artifact_semantics"]["canonical_sources"],
            )

            state_path.write_text(json.dumps({"enabled": True}), encoding="utf-8")
            legacy_state = _with_frontier_manifest_semantics(
                {},
                gems_state_path=state_path,
            )
            self.assertIn(
                "gems/gems_state.json",
                legacy_state["artifact_semantics"]["canonical_sources"],
            )

    def test_gems_state_save_marks_existing_owner_as_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            task_spec = SimpleNamespace(
                gems=SimpleNamespace(enabled=True, max_resets=2),
            )
            manager = GemsManager(
                run_dir=run_dir,
                task_spec=task_spec,
                frontier=SimpleNamespace(),
            )
            state = manager.load_state()
            manager.save_state(state)

            persisted = json.loads((run_dir / "gems" / "gems_state.json").read_text())
            semantics = persisted["artifact_semantics"]
            self.assertEqual(semantics["role"], CANONICAL_STATE)
            self.assertTrue(semantics["runtime_fact_source"])
            self.assertEqual(semantics["stage"], "gems_state")

    def test_gems_pending_reset_state_is_partial_not_runtime_fact_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            task_spec = SimpleNamespace(
                gems=SimpleNamespace(enabled=True, max_resets=2),
            )
            manager = GemsManager(
                run_dir=run_dir,
                task_spec=task_spec,
                frontier=SimpleNamespace(),
            )
            state = manager.load_state()
            state["pending_reset"] = {"completed_gen_id": 2}
            manager.save_state(state)

            persisted = json.loads((run_dir / "gems" / "gems_state.json").read_text())
            semantics = persisted["artifact_semantics"]
            self.assertEqual(semantics["role"], CANONICAL_STATE)
            self.assertEqual(semantics["status"], PARTIAL)
            self.assertFalse(semantics["runtime_fact_source"])

    def test_committed_runtime_helper_keeps_legacy_but_rejects_derived_or_partial(self) -> None:
        legacy = {"value": 1}
        committed = attach_artifact_semantics(
            {"value": 2},
            role=CANONICAL_STATE,
            stage="frontier_manifest",
        )
        derived = attach_artifact_semantics(
            {"value": 3},
            role=DERIVED_VIEW,
            stage="frontier_manifest",
            runtime_fact_source=False,
        )
        partial = attach_artifact_semantics(
            {"value": 4},
            role=CANONICAL_STATE,
            status=PARTIAL,
            stage="gems_state",
            runtime_fact_source=False,
        )

        self.assertTrue(is_committed_runtime_fact_source(legacy, legacy_ok=True))
        self.assertFalse(is_committed_runtime_fact_source(legacy, legacy_ok=False))
        self.assertTrue(is_committed_runtime_fact_source(committed, legacy_ok=True))
        self.assertFalse(is_committed_runtime_fact_source(derived, legacy_ok=True))
        self.assertFalse(is_committed_runtime_fact_source(partial, legacy_ok=True))
        self.assertTrue(is_readable_signal_source(derived, legacy_ok=True))
        self.assertTrue(is_readable_signal_source(partial, legacy_ok=True))
        self.assertTrue(
            is_readable_signal_source(
                attach_artifact_semantics(
                    {"value": 5},
                    role=CANONICAL_STATE,
                    status=FAILED,
                    stage="frontier_manifest",
                    runtime_fact_source=False,
                ),
                legacy_ok=True,
            )
        )
        self.assertFalse(
            is_readable_signal_source(
                attach_artifact_semantics(
                    {"value": 6},
                    role=CANONICAL_STATE,
                    status=SUPERSEDED,
                    stage="frontier_manifest",
                    runtime_fact_source=False,
                ),
                legacy_ok=True,
            )
        )

    def test_runtime_frontier_readers_keep_signals_from_explicit_derived_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = attach_artifact_semantics(
                {
                    "generations": {
                        "0": [
                            {
                                "generation_id": 0,
                                "variant_name": "should_not_propagate",
                                "metric_value": 99.0,
                                "evidence_stage": "full_T1",
                            }
                        ]
                    },
                    "cumulative_top": [
                        {
                            "generation_id": 0,
                            "variant_name": "should_not_propagate",
                            "metric_value": 99.0,
                            "evidence_stage": "full_T1",
                        }
                    ],
                    "lane_frontiers": {
                        "performance": [
                            {
                                "generation_id": 0,
                                "variant_name": "should_not_propagate",
                                "lane_metric_value": 99.0,
                                "evidence_stage": "full_T1",
                            }
                        ]
                    },
                    "validation_candidates": {
                        "cumulative": [
                            {
                                "generation_id": 0,
                                "variant_name": "partial_signal",
                                "metric_value": 1.0,
                            }
                        ]
                    },
                },
                role=DERIVED_VIEW,
                stage="frontier_manifest",
                runtime_fact_source=False,
            )
            self._write_raw_frontier_manifest(run_dir, manifest)

            agent = pi_agent.PIAgent(
                run_dir=run_dir,
                workspace=run_dir,
                cohort_size=1,
                model="test-model",
            )
            store = FrontierStore(run_dir / "frontier")
            api = ResearchLoopModuleAPI(run_dir)

            self.assertEqual(agent._load_frontier_summary(completed_gen_id=0), [])
            self.assertEqual(store.get_manifest()["generations"], {})
            self.assertEqual(_digest_lane_frontiers(run_dir, current_gen_id=0), {})
            validation = _digest_validation_candidates(run_dir, current_gen_id=0)
            validation_names = {entry.get("variant_name") for entry in validation}
            self.assertIn("partial_signal", validation_names)
            self.assertIn("should_not_propagate", validation_names)
            self.assertTrue(
                all(
                    entry.get("durability_scope") == "validation_signal_only"
                    for entry in validation
                )
            )
            self.assertEqual(_frontier_generations(run_dir), set())
            self.assertEqual(api.get_frontier_summary(), [])
            self.assertEqual(
                api.get_artifact_warnings()[0]["warning_type"], "non_runtime_fact_source"
            )

    def test_evidence_pack_mines_signal_only_manifest_without_validation_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = attach_artifact_semantics(
                {
                    "generations": {
                        "2": [
                            {
                                "generation_id": 2,
                                "variant_name": "late_partial_signal",
                                "metric_value": 4.2,
                                "evidence_stage": "partial",
                            }
                        ]
                    }
                },
                role=DERIVED_VIEW,
                status=PARTIAL,
                stage="frontier_manifest",
                runtime_fact_source=False,
            )
            self._write_raw_frontier_manifest(run_dir, manifest)

            validation = _digest_validation_candidates(run_dir, current_gen_id=3)

            self.assertEqual(len(validation), 1)
            self.assertEqual(validation[0]["variant_name"], "late_partial_signal")
            self.assertEqual(validation[0]["durability_scope"], "validation_signal_only")
            self.assertEqual(validation[0]["artifact_signal_status"], PARTIAL)

    def test_failed_frontier_manifest_is_signal_only_not_runtime_fact(self) -> None:
        import asyncio
        import os
        from unittest.mock import patch

        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier_dir = run_dir / "frontier"
            manifest = attach_artifact_semantics(
                {
                    "generations": {
                        "1": [
                            {
                                "generation_id": 1,
                                "variant_name": "failed_but_interesting",
                                "metric_value": 7.5,
                                "evidence_stage": "failed_probe",
                            }
                        ]
                    }
                },
                role=CANONICAL_STATE,
                status=FAILED,
                stage="frontier_manifest",
                runtime_fact_source=False,
            )
            self._write_raw_frontier_manifest(run_dir, manifest)

            validation = _digest_validation_candidates(run_dir, current_gen_id=2)
            self.assertEqual(len(validation), 1)
            self.assertEqual(validation[0]["variant_name"], "failed_but_interesting")
            self.assertEqual(validation[0]["artifact_signal_status"], FAILED)
            self.assertEqual(validation[0]["durability_scope"], "validation_signal_only")
            self.assertEqual(_frontier_generations(run_dir), set())

            with patch.dict(
                os.environ,
                {"FRONTIER_DIR": str(frontier_dir), "GENERATION_ID": "2"},
                clear=False,
            ):
                result = asyncio.run(frontier_tools._handle_get_frontier({}))

        payload = json.loads(result["content"][0]["text"])
        self.assertEqual(payload["entries"], [])
        self.assertEqual(payload["manifest_scope"], "signal_only")
        self.assertEqual(payload["manifest_signal_status"], FAILED)
        self.assertEqual(
            payload["validation_candidates"][0]["variant_name"],
            "failed_but_interesting",
        )
        self.assertEqual(payload["validation_candidates"][0]["artifact_signal_status"], FAILED)

    def test_frontier_tool_keeps_non_runtime_manifest_validation_signals(self) -> None:
        import asyncio
        import os
        from unittest.mock import patch

        from praxist.plugins.tools.frontier_tools import adapter as frontier_tools

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frontier_dir = run_dir / "frontier"
            manifest = attach_artifact_semantics(
                {
                    "generations": {
                        "0": [
                            {
                                "generation_id": 0,
                                "variant_name": "bad",
                                "metric_value": 10.0,
                                "evidence_stage": "full_T1",
                            }
                        ]
                    },
                    "validation_candidates": {
                        "cumulative": [
                            {
                                "generation_id": 0,
                                "variant_name": "partial_signal",
                                "metric_value": 3.0,
                            }
                        ]
                    },
                },
                role=CANONICAL_STATE,
                status=PARTIAL,
                stage="frontier_manifest",
                runtime_fact_source=False,
            )
            self._write_raw_frontier_manifest(run_dir, manifest)
            with patch.dict(
                os.environ,
                {"FRONTIER_DIR": str(frontier_dir), "GENERATION_ID": "1"},
                clear=False,
            ):
                result = asyncio.run(frontier_tools._handle_get_frontier({}))

        payload = json.loads(result["content"][0]["text"])
        self.assertEqual(payload["entries"], [])
        self.assertEqual(payload["manifest_scope"], "signal_only")
        self.assertFalse(payload["manifest_runtime_fact_source"])
        names = {entry.get("variant_name") for entry in payload["validation_candidates"]}
        self.assertIn("bad", names)
        self.assertIn("partial_signal", names)
        self.assertEqual(payload["artifact_semantics"]["role"], DERIVED_VIEW)

    def test_deliver_loader_ignores_non_runtime_frontier_manifest(self) -> None:
        from praxist import deliver

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_raw_frontier_manifest(
                run_dir,
                attach_artifact_semantics(
                    {"cumulative_top": [{"variant_name": "bad"}]},
                    role=DERIVED_VIEW,
                    stage="frontier_manifest",
                    runtime_fact_source=False,
                ),
            )

            self.assertIsNone(deliver.load_frontier_manifest(run_dir))

    def test_system_frontier_snapshot_returns_non_runtime_manifest_as_signal_only(self) -> None:
        from praxist.plugins.tools.system.adapter import system_frontier_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_raw_frontier_manifest(
                run_dir,
                attach_artifact_semantics(
                    {"cumulative_top": [{"variant_name": "signal"}]},
                    role=DERIVED_VIEW,
                    stage="frontier_manifest",
                    runtime_fact_source=False,
                ),
            )

            snapshot = system_frontier_snapshot(run_dir)

            self.assertIsNone(snapshot["frontier"])
            self.assertEqual(
                snapshot["frontier_signal"]["cumulative_top"][0]["variant_name"], "signal"
            )
            self.assertEqual(snapshot["frontier_signal_status"], COMMITTED)
            self.assertIn("frontier_signal", snapshot["warnings"][0])

    def test_gems_prompt_and_module_api_ignore_partial_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            task_spec = SimpleNamespace(
                gems=SimpleNamespace(enabled=True, max_resets=2),
            )
            manager = GemsManager(
                run_dir=run_dir,
                task_spec=task_spec,
                frontier=SimpleNamespace(),
            )
            state = manager.load_state()
            state["gems"] = [
                {
                    "gem_finding_id": "gem_r00_01_x",
                    "variant_name": "should_not_propagate",
                    "admission_metrics": {"score": 1.0},
                }
            ]
            state["pending_reset"] = {"completed_gen_id": 3}
            manager.save_state(state)

            api = ResearchLoopModuleAPI(run_dir)

            self.assertEqual(manager.active_gems(), [])
            self.assertEqual(manager.prompt_context(absolute_gen_id=4)["entries"], [])
            self.assertEqual(api.get_gems_summary(), {})
            self.assertEqual(
                api.get_artifact_warnings()[0]["warning_type"], "non_runtime_fact_source"
            )

    def test_gem_id_trust_rejects_failed_state_but_allows_pending_repair_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            shared.mkdir(parents=True)
            finding_path = shared / "gem.json"
            finding_path.write_text(
                json.dumps({"id": "gem_r00_01_x", "finding_type": "gem"}),
                encoding="utf-8",
            )
            failed_state = attach_artifact_semantics(
                {
                    "gems": [
                        {
                            "gem_finding_id": "gem_r00_01_x",
                            "finding_path": "shared_findings/gem.json",
                        }
                    ]
                },
                role=CANONICAL_STATE,
                status=FAILED,
                stage="gems_state",
                runtime_fact_source=False,
            )
            self._write_raw_gems_state(run_dir, failed_state)

            self.assertEqual(_trusted_declared_gem_id(finding_path, "gem_r00_01_x"), "")

            pending_state = attach_artifact_semantics(
                {
                    "pending_reset": {
                        "gem_records": [
                            {
                                "gem_finding_id": "gem_r00_01_x",
                                "finding_path": "shared_findings/gem.json",
                            }
                        ]
                    }
                },
                role=CANONICAL_STATE,
                status=PARTIAL,
                stage="gems_state",
                runtime_fact_source=False,
            )
            self._write_raw_gems_state(run_dir, pending_state)

            self.assertEqual(
                _trusted_declared_gem_id(finding_path, "gem_r00_01_x"),
                "gem_r00_01_x",
            )

    def test_leaderboard_tool_response_is_a_bounded_derived_view(self) -> None:
        payload = {"mode": "server", "entries": [{"variant_name": "v0", "score": 1.0}]}

        out = _envelope_leaderboard(payload, inline_limit=4, generation=5)

        self.assertEqual(out["artifact_semantics"]["role"], DERIVED_VIEW)
        self.assertEqual(out["artifact_semantics"]["generation_id"], 5)
        self.assertFalse(out["artifact_semantics"]["runtime_fact_source"])
        self.assertEqual(out["_tool_output"]["tool_name"], "get_leaderboard")

    def test_evidence_pack_and_budgeted_pack_are_derived_audit_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pack = build_evidence_pack(
                Path(tmp),
                panel_mode="mini",
                current_gen_id=0,
                target_decisions=["agenda"],
                pi_roles=["builder"],
            )

            self.assertEqual(
                pack.shared_core["artifact_semantics"]["role"],
                DERIVED_AUDIT_SNAPSHOT,
            )
            self.assertFalse(pack.shared_core["artifact_semantics"]["runtime_fact_source"])
            self.assertEqual(
                pack.shared_core["current_frontier_scope"],
                "latest_per_axis_generation_delta_anchors",
            )
            self.assertEqual(pack.audit["artifact_semantics"]["role"], DERIVED_AUDIT_SNAPSHOT)
            self.assertIn("created_at", pack.audit["artifact_semantics"])

            budgeted = fit_pack_to_budget(
                EvidencePack(
                    pack_id=pack.pack_id,
                    built_at=pack.built_at,
                    panel_mode=pack.panel_mode,
                    target_decisions=pack.target_decisions,
                    shared_core=pack.shared_core,
                    private_packs=pack.private_packs,
                    all_cards=pack.all_cards,
                    audit=pack.audit,
                ),
                mode="mini",
            )
            self.assertEqual(budgeted["artifact_semantics"]["role"], DERIVED_AUDIT_SNAPSHOT)
            self.assertFalse(budgeted["artifact_semantics"]["runtime_fact_source"])

    @staticmethod
    def _write_generation_results(run_dir: Path, gen_id: int) -> None:
        gen_dir = run_dir / f"gen_{gen_id}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        (gen_dir / "generation_results.json").write_text(
            json.dumps([{"peer_id": f"gen{gen_id}_peer0", "success": True}]),
            encoding="utf-8",
        )

    @staticmethod
    def _write_frontier_manifest(run_dir: Path, generations: dict[str, list[dict]]) -> None:
        frontier_dir = run_dir / "frontier"
        frontier_dir.mkdir(parents=True, exist_ok=True)
        (frontier_dir / "frontier_manifest.json").write_text(
            json.dumps({"generations": generations, "cumulative_top": []}),
            encoding="utf-8",
        )

    @staticmethod
    def _write_raw_frontier_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
        frontier_dir = run_dir / "frontier"
        frontier_dir.mkdir(parents=True, exist_ok=True)
        (frontier_dir / "frontier_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

    @staticmethod
    def _write_raw_gems_state(run_dir: Path, state: dict[str, Any]) -> None:
        gems_dir = run_dir / "gems"
        gems_dir.mkdir(parents=True, exist_ok=True)
        (gems_dir / "gems_state.json").write_text(
            json.dumps(state),
            encoding="utf-8",
        )


class ArtifactSemanticsHelperCoverageTest(unittest.TestCase):
    def test_semantics_helper_accepts_explicit_created_at(self) -> None:
        semantics = artifact_semantics(
            role=DERIVED_VIEW,
            status=COMMITTED,
            stage="test",
            created_at="2026-01-01T00:00:00Z",
            canonical_sources=["source"],
        )

        self.assertEqual(semantics["created_at"], "2026-01-01T00:00:00Z")
        self.assertEqual(semantics["canonical_sources"], ["source"])


if __name__ == "__main__":
    unittest.main()
