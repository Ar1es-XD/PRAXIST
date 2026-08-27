from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from praxist.plugins.workflow_stages.research_loop.backend.research_topology import (
    LegacyResearchTopologyExecutor,
    ResearchCommand,
    ResearchLoopModuleAPI,
    ResearchTopologySpec,
    TopologyChangeRequest,
    TopologyEdge,
    WorkerCapabilitySet,
    WorkerSpec,
    build_legacy_generation_topology,
)


class ResearchTopologyContractTest(unittest.TestCase):
    def test_topology_rejects_edges_to_unknown_workers(self) -> None:
        topology = ResearchTopologySpec(
            topology_id="bad",
            nodes=[WorkerSpec(worker_id="peer0", worker_type="experiment_peer")],
            edges=[TopologyEdge(source="peer0", target="missing")],
        )

        with self.assertRaisesRegex(ValueError, "unknown target"):
            topology.to_dict()

    def test_worker_spec_defaults_experiment_capabilities(self) -> None:
        worker = WorkerSpec(worker_id="peer0", worker_type="experiment_peer").to_dict()

        self.assertTrue(worker["capabilities"]["can_write_code"])
        self.assertTrue(worker["capabilities"]["can_run_training"])
        self.assertTrue(worker["capabilities"]["can_run_eval"])

    def test_worker_spec_accepts_custom_capabilities(self) -> None:
        worker = WorkerSpec(
            worker_id="audit0",
            worker_type="data_engineering_peer",
            capabilities=WorkerCapabilitySet(can_modify_data_pipeline=True),
        ).to_dict()

        self.assertTrue(worker["capabilities"]["can_modify_data_pipeline"])
        self.assertFalse(worker["capabilities"]["can_write_code"])

    def test_legacy_generation_topology_matches_existing_cohort_shape(self) -> None:
        loop = SimpleNamespace(
            task_spec=SimpleNamespace(
                task_id="toy",
                generation_policy=SimpleNamespace(cohort_size=3),
            ),
            _panel_topology_ref="panel_topology:test",
            peer_role_ref="task_role:task_peer",
            peer_role_ref_for=lambda _gen, index: (
                "task_role:starter" if index == 0 else "task_role:solver"
            ),
        )

        topology = build_legacy_generation_topology(loop, 2)
        data = topology.to_dict()

        self.assertEqual(data["topology_id"], "legacy_generation_cohort_v1_gen_2")
        self.assertEqual(data["generation_id"], 2)
        self.assertEqual(
            [node["worker_id"] for node in data["nodes"]],
            [
                "gen2_peer0",
                "gen2_peer1",
                "gen2_peer2",
            ],
        )
        self.assertTrue(data["policy"]["metadata"]["compatibility_adapter"])
        self.assertTrue(data["metadata"]["preserves_legacy_behavior"])
        self.assertEqual(
            [node["role_ref"] for node in data["nodes"]],
            ["task_role:starter", "task_role:solver", "task_role:solver"],
        )

    def test_legacy_generation_topology_does_not_fabricate_missing_role(self) -> None:
        loop = SimpleNamespace(
            task_spec=SimpleNamespace(
                task_id="toy",
                generation_policy=SimpleNamespace(cohort_size=1),
            ),
            _panel_topology_ref="panel_topology:test",
        )

        topology = build_legacy_generation_topology(loop, 0).to_dict()

        self.assertIsNone(topology["nodes"][0]["role_ref"])

    def test_generation_loop_resolves_rotated_peer_roles_independently(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.generation_loop import (
            GenerationLoop,
            _index_peer_role_skills,
        )
        from praxist.plugins.workflow_stages.research_loop.peer_roles import PeerRoleSelector

        starter = SimpleNamespace(
            role_ref="task_role:starter",
            role_id="starter",
            legacy_role_id="initial_builder",
        )
        solver = SimpleNamespace(
            role_ref="task_role:solver",
            role_id="solver",
            legacy_role_id="problem_solver",
        )
        indexed = _index_peer_role_skills((starter, solver))
        self.assertIs(indexed["initial_builder"], starter)
        self.assertIs(indexed["problem_solver"], solver)
        self.assertIs(indexed["task_role:solver"], solver)
        loop = object.__new__(GenerationLoop)
        loop.run_dir = Path("/unused")
        loop.task_spec = SimpleNamespace(generation_policy=SimpleNamespace(cohort_size=2))
        loop._peer_role_rotation = ("task_role:starter", "solver")
        loop.peer_role_skills = (starter, solver)
        loop._peer_role_skills_by_id = indexed
        loop.peer_role_skill = None
        loop.peer_role_ref = "task_role:peer_generalist"
        loop._peer_role_selector = PeerRoleSelector(
            run_dir=loop.run_dir,
            task_spec=loop.task_spec,
            role_rotation=loop._peer_role_rotation,
            default_role_skill=None,
            role_skills=(starter, solver),
        )

        with patch(
            "praxist.plugins.workflow_stages.research_loop.backend.pi_agent.load_agenda_for_gen",
            return_value=None,
        ):
            self.assertEqual(loop.peer_role_ref_for(0, 0), "task_role:starter")
            self.assertEqual(loop.peer_role_ref_for(0, 1), "task_role:solver")

        with patch(
            "praxist.plugins.workflow_stages.research_loop.backend.pi_agent.load_agenda_for_gen",
            return_value={"peer_contracts": {"gen1_peer0": {"role": "task_role:solver"}}},
        ):
            self.assertEqual(loop.peer_role_ref_for(1, 0), "task_role:solver")

        with patch(
            "praxist.plugins.workflow_stages.research_loop.backend.pi_agent.load_agenda_for_gen",
            return_value={"peer_contracts": {"gen1_peer0": {"role": "Problem-Solver"}}},
        ):
            self.assertEqual(loop.peer_role_ref_for(1, 0), "task_role:solver")

        unresolved = {
            "peer_id": "gen1_peer0",
            "research_agenda": {
                "peer_contracts": {"gen1_peer0": {"role": "undeclared specialist"}}
            },
        }
        self.assertIsNone(loop.peer_role_skill_for_context(unresolved))
        self.assertIsNone(loop.peer_role_ref_for_context(unresolved))

        shared = SimpleNamespace(
            role_ref="task_role:peer",
            role_id="peer",
            legacy_role_id="peer",
        )
        loop.peer_role_skills = (shared,)
        loop.peer_role_skill = shared
        loop._peer_role_skills_by_id = {"peer": shared}
        loop._peer_role_selector = PeerRoleSelector(
            run_dir=loop.run_dir,
            task_spec=loop.task_spec,
            role_rotation=loop._peer_role_rotation,
            default_role_skill=shared,
            role_skills=(shared,),
        )
        self.assertIs(loop.peer_role_skill_for_context(unresolved), shared)
        self.assertEqual(loop.peer_role_ref_for_context(unresolved), "task_role:peer")

    def test_module_api_keeps_frontier_facts_and_validation_signals_separate(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
            CANONICAL_STATE,
            PARTIAL,
            attach_artifact_semantics,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontier_dir = root / "frontier"
            frontier_dir.mkdir()
            manifest = attach_artifact_semantics(
                {
                    "generations": {
                        "0": [
                            {
                                "generation_id": 0,
                                "variant_name": "partial_signal",
                                "metric_value": 3.0,
                                "evidence_stage": "partial",
                            }
                        ]
                    }
                },
                role=CANONICAL_STATE,
                status=PARTIAL,
                stage="frontier_manifest",
                runtime_fact_source=False,
            )
            (frontier_dir / "frontier_manifest.json").write_text(json.dumps(manifest))

            api = ResearchLoopModuleAPI(root)

            self.assertEqual(api.get_frontier_summary(), [])
            signals = api.get_validation_signals(current_gen_id=0)
            self.assertEqual(signals[0]["variant_name"], "partial_signal")
            self.assertEqual(signals[0]["durability_scope"], "validation_signal_only")

    def test_executor_persists_topology_and_delegates_to_cohort_runner(self) -> None:
        calls: list[tuple[object, int]] = []

        async def fake_cohort_runner(loop, gen_id: int):
            calls.append((loop, gen_id))
            return [{"peer_id": f"gen{gen_id}_peer0", "success": True}]

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            loop = SimpleNamespace(
                run_dir=run_dir,
                task_spec=SimpleNamespace(
                    task_id="toy",
                    generation_policy=SimpleNamespace(cohort_size=1),
                ),
                _panel_topology_ref="panel_topology:test",
            )
            executor = LegacyResearchTopologyExecutor(fake_cohort_runner)
            result = asyncio.run(executor.execute_generation(loop, 4))

            self.assertEqual(result, [{"peer_id": "gen4_peer0", "success": True}])
            self.assertEqual(calls, [(loop, 4)])
            topology_path = run_dir / "gen_4/research_topology.json"
            self.assertTrue(topology_path.exists())
            topology = json.loads(topology_path.read_text(encoding="utf-8"))
            self.assertEqual(topology["nodes"][0]["worker_id"], "gen4_peer0")

    def test_executor_sidecar_failure_does_not_block_generation(self) -> None:
        async def fake_cohort_runner(loop, gen_id: int):
            return [{"peer_id": f"gen{gen_id}_peer0", "success": True}]

        loop = SimpleNamespace(
            run_dir=Path("/proc/definitely-not-writable"),
            task_spec=SimpleNamespace(
                task_id="toy",
                generation_policy=SimpleNamespace(cohort_size=1),
            ),
            _panel_topology_ref="panel_topology:test",
        )
        executor = LegacyResearchTopologyExecutor(fake_cohort_runner)

        result = asyncio.run(executor.execute_generation(loop, 4))

        self.assertEqual(result, [{"peer_id": "gen4_peer0", "success": True}])


class ResearchLoopModuleAPITest(unittest.TestCase):
    def test_submit_recommendation_writes_structured_command_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            api = ResearchLoopModuleAPI(tmp)
            command = api.submit_recommendation(
                recommendation="Add one replication worker next generation.",
                reason="External reviewer flagged weak reproducibility.",
            )

            records = api.list_commands()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["command_id"], command.command_id)
            self.assertEqual(records[0]["command_type"], "recommendation")
            self.assertEqual(records[0]["scope"], "next_generation")
            self.assertIn("replication worker", records[0]["payload"]["recommendation"])
            self.assertEqual(records[0]["status"], "queued")

    def test_append_command_forces_queued_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            api = ResearchLoopModuleAPI(tmp)
            command = ResearchCommand.create(
                command_type="recommendation",
                payload={"recommendation": "Use a safer topology."},
            )
            object.__setattr__(command, "status", "accepted")

            api.append_command(command)

            self.assertEqual(api.list_commands()[0]["status"], "queued")

    def test_list_commands_skips_malformed_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            api = ResearchLoopModuleAPI(tmp)
            api.requests_dir.mkdir(parents=True)
            api.commands_path.write_text(
                "{bad\n"
                + json.dumps(
                    {
                        "command_id": "cmd_good",
                        "command_type": "recommendation",
                        "scope": "next_generation",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            records = api.list_commands()

            self.assertEqual(
                records,
                [
                    {
                        "command_id": "cmd_good",
                        "command_type": "recommendation",
                        "scope": "next_generation",
                    }
                ],
            )

    def test_list_commands_limit_returns_latest_well_formed_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            api = ResearchLoopModuleAPI(tmp)
            api.requests_dir.mkdir(parents=True)
            api.commands_path.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "command_id": f"cmd_{index}",
                            "command_type": "recommendation",
                            "scope": "next_generation",
                        }
                    )
                    for index in range(5)
                ),
                encoding="utf-8",
            )

            records = api.list_commands(limit=2)

            self.assertEqual([record["command_id"] for record in records], ["cmd_3", "cmd_4"])

    def test_request_topology_change_records_request_without_mutating_loop_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            api = ResearchLoopModuleAPI(tmp)
            command = api.request_topology_change(
                requested_changes=[
                    {"op": "add_worker", "worker_type": "replication_peer", "count": 1},
                    {"op": "reduce_worker", "worker_type": "experiment_peer", "count": 1},
                ],
                reason="Need independent replication before promotion.",
                safety_constraints=["apply after generation boundary"],
            )

            records = api.list_commands()
            self.assertEqual(command.command_type, "topology_change_request")
            payload = records[0]["payload"]
            self.assertEqual(payload["reason"], "Need independent replication before promotion.")
            self.assertEqual(payload["scope"], "next_generation")
            self.assertEqual(payload["requested_changes"][0]["worker_type"], "replication_peer")

    def test_topology_change_request_can_be_serialized_as_command(self) -> None:
        request = TopologyChangeRequest.create(
            requested_changes=[{"op": "add_worker", "worker_type": "falsifier_peer"}],
            reason="Need a falsifier.",
        )

        command = request.to_command(submitted_by="paper_writer")

        self.assertEqual(command.command_type, "topology_change_request")
        self.assertEqual(command.submitted_by, "paper_writer")
        self.assertEqual(command.payload["request_id"], request.request_id)

    def test_artifact_api_reads_findings_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            findings = root / "findings"
            findings.mkdir()
            (findings / "findings.jsonl").write_text(
                json.dumps({"variant_name": "a", "metric": 1.0})
                + "\n"
                + json.dumps({"variant_name": "b", "metric": 2.0})
                + "\n",
                encoding="utf-8",
            )

            records = ResearchLoopModuleAPI(root).list_findings()

            self.assertEqual([record["variant_name"] for record in records], ["a", "b"])

    def test_artifact_api_excludes_frontier_from_findings_and_reads_shared_findings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            findings = root / "findings"
            findings.mkdir()
            (findings / "findings.jsonl").write_text(
                json.dumps({"finding_id": "canonical", "variant_name": "a"}) + "\n",
                encoding="utf-8",
            )
            (findings / "frontier.jsonl").write_text(
                json.dumps({"finding_id": "frontier_only", "variant_name": "frontier"}) + "\n",
                encoding="utf-8",
            )
            shared = root / "shared_findings"
            shared.mkdir()
            (shared / "shared.json").write_text(
                json.dumps({"finding_id": "shared", "variant_name": "b"}),
                encoding="utf-8",
            )

            records = ResearchLoopModuleAPI(root).list_findings()

            self.assertEqual(
                [record["finding_id"] for record in records],
                ["canonical", "shared"],
            )

    def test_artifact_api_dedupes_finding_id_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            findings = root / "findings"
            findings.mkdir()
            (findings / "findings.jsonl").write_text(
                json.dumps({"finding_id": "same", "variant_name": "a"}) + "\n",
                encoding="utf-8",
            )
            shared = root / "shared_findings"
            shared.mkdir()
            (shared / "same.json").write_text(
                json.dumps({"id": "same", "variant_name": "a"}),
                encoding="utf-8",
            )

            records = ResearchLoopModuleAPI(root).list_findings()

            self.assertEqual(
                [record.get("finding_id") or record.get("id") for record in records], ["same"]
            )

    def test_artifact_api_skips_malformed_shared_finding_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = root / "shared_findings"
            shared.mkdir()
            (shared / "bad.json").write_text("{bad", encoding="utf-8")
            (shared / "good.json").write_text(
                json.dumps({"finding_id": "good", "variant_name": "v"}),
                encoding="utf-8",
            )

            api = ResearchLoopModuleAPI(root)
            records = api.list_findings()

            self.assertEqual([record["finding_id"] for record in records], ["good"])
            warnings = api.get_artifact_warnings()
            self.assertEqual(warnings[0]["warning_type"], "json_decode_error")
            self.assertIn("bad.json", warnings[0]["path"])

    def test_artifact_api_reads_real_frontier_manifest_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontier = root / "frontier"
            frontier.mkdir()
            (frontier / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "cumulative_top": [{"finding_id": "top", "variant_name": "a"}],
                        "lane_frontiers": {
                            "incubator": [{"finding_id": "lane", "variant_name": "b"}]
                        },
                        "generations": {
                            "1": [{"finding_id": "gen", "variant_name": "c"}],
                        },
                    }
                ),
                encoding="utf-8",
            )

            records = ResearchLoopModuleAPI(root).get_frontier_summary()

            self.assertEqual([record["finding_id"] for record in records], ["top", "lane", "gen"])
            self.assertEqual(records[1]["frontier_lane"], "incubator")
            self.assertEqual(records[2]["generation_id"], "1")

    def test_artifact_api_frontier_summary_deduplicates_variant_entities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontier = root / "frontier"
            frontier.mkdir()
            (frontier / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "cumulative_top": [
                            {
                                "finding_id": "root",
                                "variant_name": "risk_adjusted_listwise_bc_target",
                                "frontier_entity_key": "variant::risk_adjusted_listwise_bc_target",
                            }
                        ],
                        "lane_frontiers": {
                            "alpha_incubator": [
                                {
                                    "finding_id": "alias",
                                    "variant_name": "gen0_peer5_risk_adjusted_listwise_bc_target_t1",
                                    "frontier_entity_key": "variant::risk_adjusted_listwise_bc_target",
                                },
                                {
                                    "finding_id": "other",
                                    "variant_name": "bc_curriculum_drop_topk",
                                },
                            ]
                        },
                        "generations": {
                            "0": [
                                {
                                    "finding_id": "generation_alias",
                                    "variant_name": "gen0_peer5_risk_adjusted_listwise_bc_target_t1",
                                    "frontier_entity_key": "variant::risk_adjusted_listwise_bc_target",
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            records = ResearchLoopModuleAPI(root).get_frontier_summary()

            self.assertEqual(
                [record["finding_id"] for record in records],
                ["root", "other"],
            )

    def test_artifact_api_reads_frontier_generation_entry_wrappers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontier = root / "frontier"
            frontier.mkdir()
            (frontier / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {
                            "2": {
                                "entries": [
                                    {"finding_id": "entry", "variant_name": "wrapped"},
                                ]
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            records = ResearchLoopModuleAPI(root).get_frontier_summary()

            self.assertEqual(records[0]["finding_id"], "entry")
            self.assertEqual(records[0]["generation_id"], "2")

    def test_artifact_api_falls_back_when_frontier_manifest_is_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontier = root / "frontier"
            frontier.mkdir()
            (frontier / "frontier_manifest.json").write_text("{bad", encoding="utf-8")
            findings = root / "findings"
            findings.mkdir()
            (findings / "frontier.jsonl").write_text(
                json.dumps({"finding_id": "frontier", "variant_name": "a"}) + "\n",
                encoding="utf-8",
            )

            api = ResearchLoopModuleAPI(root)
            records = api.get_frontier_summary()

            self.assertEqual([record["finding_id"] for record in records], ["frontier"])
            warnings = api.get_artifact_warnings()
            self.assertEqual(warnings[0]["warning_type"], "json_decode_error")
            self.assertIn("frontier_manifest.json", warnings[0]["path"])

    def test_artifact_api_reads_materialized_frontier_when_manifest_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            findings = root / "findings"
            findings.mkdir()
            (findings / "frontier.jsonl").write_text(
                json.dumps({"finding_id": "frontier", "variant_name": "a"}) + "\n",
                encoding="utf-8",
            )

            records = ResearchLoopModuleAPI(root).get_frontier_summary()

            self.assertEqual([record["finding_id"] for record in records], ["frontier"])

    def test_artifact_api_deduplicates_frontier_json_fallback_by_entity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontier = root / "frontier"
            frontier.mkdir()
            (frontier / "frontier.json").write_text(
                json.dumps(
                    [
                        {
                            "finding_id": "root",
                            "variant_name": "risk_adjusted_listwise_bc_target",
                        },
                        {
                            "finding_id": "alias",
                            "variant_name": "gen0_peer5_risk_adjusted_listwise_bc_target_t1",
                            "frontier_entity_key": "variant::risk_adjusted_listwise_bc_target",
                        },
                        {"finding_id": "other", "variant_name": "bc_curriculum_drop_topk"},
                    ]
                ),
                encoding="utf-8",
            )

            records = ResearchLoopModuleAPI(root).get_frontier_summary()

            self.assertEqual([record["finding_id"] for record in records], ["root", "other"])

    def test_artifact_api_deduplicates_materialized_frontier_jsonl_by_entity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            findings = root / "findings"
            findings.mkdir()
            (findings / "frontier.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "finding_id": "root",
                                "variant_name": "risk_adjusted_listwise_bc_target",
                            }
                        ),
                        json.dumps(
                            {
                                "finding_id": "alias",
                                "variant_name": "gen0_peer5_risk_adjusted_listwise_bc_target_t1",
                                "frontier_entity_key": "variant::risk_adjusted_listwise_bc_target",
                            }
                        ),
                        json.dumps(
                            {
                                "finding_id": "other",
                                "variant_name": "bc_curriculum_drop_topk",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            records = ResearchLoopModuleAPI(root).get_frontier_summary()

            self.assertEqual([record["finding_id"] for record in records], ["root", "other"])

    def test_artifact_api_prefers_gems_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gems = root / "gems"
            gems.mkdir()
            (gems / "gems.json").write_text(json.dumps({"version": "old"}), encoding="utf-8")
            (gems / "gems_state.json").write_text(
                json.dumps({"version": "state", "gems": [{"variant_name": "v"}]}),
                encoding="utf-8",
            )

            summary = ResearchLoopModuleAPI(root).get_gems_summary()

            self.assertEqual(summary["version"], "state")

    def test_artifact_api_skips_malformed_gems_state_and_reads_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gems = root / "gems"
            gems.mkdir()
            (gems / "gems_state.json").write_text("{bad", encoding="utf-8")
            (gems / "gems.json").write_text(json.dumps({"version": "fallback"}), encoding="utf-8")

            api = ResearchLoopModuleAPI(root)
            summary = api.get_gems_summary()

            self.assertEqual(summary["version"], "fallback")
            warnings = api.get_artifact_warnings()
            self.assertEqual(warnings[0]["warning_type"], "json_decode_error")
            self.assertIn("gems_state.json", warnings[0]["path"])

    def test_artifact_api_reads_canonical_memory_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = root / "memory"
            memory.mkdir()
            (memory / "research_memory.jsonl").write_text(
                json.dumps({"memory_record_id": "m1", "summary": "kept"}) + "\n",
                encoding="utf-8",
            )

            summary = ResearchLoopModuleAPI(root).get_memory_summary()

            self.assertEqual(summary["records"][0]["memory_record_id"], "m1")

    def test_artifact_api_skips_malformed_run_status_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "orchestrator_status.json").write_text("{bad", encoding="utf-8")
            (root / "run_summary.json").write_text(
                json.dumps({"status": "running"}),
                encoding="utf-8",
            )

            api = ResearchLoopModuleAPI(root)
            status = api.get_run_status()

            self.assertEqual(status["status"], "running")
            warnings = api.get_artifact_warnings()
            self.assertEqual(warnings[0]["warning_type"], "json_decode_error")
            self.assertIn("orchestrator_status.json", warnings[0]["path"])

    def test_artifact_api_prefers_final_run_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "orchestrator_status.json").write_text(
                json.dumps({"status": "in_progress"}),
                encoding="utf-8",
            )
            (root / "orchestrator_status.final.json").write_text(
                json.dumps({"status": "complete", "exit_condition": "completed"}),
                encoding="utf-8",
            )

            status = ResearchLoopModuleAPI(root).get_run_status()

            self.assertEqual(status["status"], "complete")
            self.assertEqual(status["exit_condition"], "completed")

    def test_prompt_context_does_not_inject_external_commands_into_peer_prompt(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import prompt_context

        class FakeFrontier:
            def get_summary(self):
                return []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            api = ResearchLoopModuleAPI(root)
            for index in range(7):
                api.submit_recommendation(
                    recommendation=f"recommendation {index}",
                    reason=f"reason {index}",
                    submitted_by="paper_writer",
                )
            task_spec = SimpleNamespace(
                evaluation=SimpleNamespace(
                    primary_metric="score",
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
                    run_dir=root,
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

            self.assertNotIn("external_research_commands", context)


if __name__ == "__main__":
    unittest.main()
