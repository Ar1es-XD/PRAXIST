from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class TaskProjectAndPanelContractsTest(unittest.TestCase):
    def test_task_project_loader_manifest_refs_and_runner_edges(self) -> None:
        from praxist.core import task_project

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            task = base / "task"
            task.mkdir()
            task.joinpath("task.yaml").write_text(
                """
task_id: demo
capabilities: [cap.a]
runner:
  entrypoint: runner:create_runner
praxist_plugins:
  task_ref: task:demo
  capabilities: [cap.b]
  workflow_stage: workflow_stage:research_loop
  workflow: {ref: workflow_stage:research_loop}
  agent_runtime: agent_runtime:fake_runtime
  model_provider: model_provider:fake_provider
  budget_policy: budget_policy:default_basic
  graph_maintainer: graph_maintainer:finding_graph_mvp
  panel:
    topology: panel_topology:fake_two_round
    roles:
      - role:generic_pi
      - task_role:local_pi
    audit_rules: [audit_rule:generic, task_audit:local]
    evaluations: [evaluation:generic, task_evaluation:local]
  tools: [tool_server:evaluation_tools]
  graph_maintainers: [graph_maintainer:finding_graph_mvp]
  optional_workflow_stages:
    enabled: workflow_stage:ideation_stub
    disabled: {enabled: false, ref: workflow_stage:paper_writing_stub}
    dict: {ref: workflow_stage:reviewer_stub}
  tool_servers:
    a: tool_server:frontier_tools
    b: {enabled: false, ref: tool_server:memory_tools}
    c: {ref: tool_server:prior_work_tools}
""",
                encoding="utf-8",
            )
            task.joinpath("runner.py").write_text(
                "def create_runner(project):\n    return {'task_id': project.task_id}\n",
                encoding="utf-8",
            )
            task.joinpath("description.md").write_text("demo", encoding="utf-8")
            task.joinpath("experiments").mkdir()
            task.joinpath("experiments", "run.json").write_text("ignored", encoding="utf-8")
            task.joinpath("data", "raw").mkdir(parents=True)
            task.joinpath("data", "raw", "huge.bin").write_text("ignored", encoding="utf-8")
            task.joinpath("__pycache__").mkdir()
            task.joinpath("__pycache__", "x.py").write_text("ignored", encoding="utf-8")
            task.joinpath("compiled.pyc").write_bytes(b"ignored")

            project = task_project.resolve_task_project("task", workspace=base)
            self.assertEqual(project.task_id, "demo")
            self.assertEqual(project.task_ref, "task:demo")
            self.assertTrue(task_project.task_project_has_capability(project, "cap.a"))
            self.assertTrue(task_project.task_project_has_capability(project, "cap.b"))
            self.assertFalse(task_project.task_project_has_capability(project, "missing"))
            self.assertEqual(task_project.load_task_project_runner(project), {"task_id": "demo"})
            manifest_paths = {entry["path"] for entry in project.manifest["files"]}
            self.assertIn("task.yaml", manifest_paths)
            self.assertIn("description.md", manifest_paths)
            self.assertNotIn("experiments/run.json", manifest_paths)
            self.assertNotIn("data/raw/huge.bin", manifest_paths)
            self.assertNotIn("__pycache__/x.py", manifest_paths)
            self.assertNotIn("compiled.pyc", manifest_paths)
            written = task_project.write_task_project_manifest(base / "run", project)
            self.assertEqual(json.loads(written.read_text(encoding="utf-8"))["task_id"], "demo")

            refs = {
                ref.as_string()
                for ref in task_project.task_project_global_plugin_refs(project.descriptor)
            }
            self.assertIn("workflow_stage:research_loop", refs)
            self.assertIn("agent_runtime:fake_runtime", refs)
            self.assertIn("panel_topology:fake_two_round", refs)
            self.assertIn("tool_server:frontier_tools", refs)
            self.assertNotIn("task_role:local_pi", refs)
            self.assertNotIn("tool_server:memory_tools", refs)
            self.assertTrue(task_project.is_task_local_ref("task_role:x"))
            self.assertFalse(task_project.is_task_local_ref("role:x"))

            by_file = task_project.resolve_task_project(task / "task.yaml")
            self.assertEqual(by_file.path.resolve(), task.resolve())
            bad_file = task / "not_task.yaml"
            bad_file.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be task.yaml"):
                task_project.resolve_task_project(bad_file)
            with self.assertRaises(FileNotFoundError):
                task_project.resolve_task_project(task / "missing")

            bad_ref_task = base / "bad_ref_task"
            bad_ref_task.mkdir()
            bad_ref_task.joinpath("task.yaml").write_text(
                "task_id: demo\npraxist_plugins:\n  task_ref: role:bad\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "task_ref"):
                task_project.resolve_task_project(bad_ref_task)

            no_runner = task_project.TaskProject(
                path=task,
                task_id="demo",
                task_ref="task:demo",
                descriptor_path=task / "task.yaml",
                descriptor={},
                manifest={},
            )
            with self.assertRaisesRegex(ValueError, "no runner.entrypoint"):
                task_project.load_task_project_runner(no_runner)
            bad_entry = task_project.TaskProject(
                path=task,
                task_id="demo",
                task_ref="task:demo",
                descriptor_path=task / "task.yaml",
                descriptor={"runner": {"entrypoint": "runner"}},
                manifest={},
            )
            with self.assertRaisesRegex(ValueError, "module:function"):
                task_project.load_task_project_runner(bad_entry)

            with self.assertRaises(FileNotFoundError):
                task_project._load_project_module(task, "missing")
            outside = base / "escape.py"
            outside.write_text("x = 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes"):
                task_project._load_project_module(task, str(outside.with_suffix("")))
            with (
                patch.object(
                    task_project.importlib.util, "spec_from_file_location", return_value=None
                ),
                self.assertRaises(ImportError),
            ):
                task_project._load_project_module(task, "runner")
            non_mapping = task / "list.yaml"
            non_mapping.write_text("bad\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mapping"):
                task_project._read_yaml(non_mapping)

            self.assertEqual(
                task_project._task_id_from_descriptor(
                    task, {"praxist_plugins": {"task_ref": "task:from_ref"}}
                ),
                "from_ref",
            )
            self.assertEqual(task_project._task_id_from_descriptor(task, {}), "task")
            deduped = task_project._dedupe_refs(
                [
                    task_project.PluginRef("agent_runtime", "x"),
                    task_project.PluginRef("agent_runtime", "x"),
                    task_project.PluginRef("model_provider", "p"),
                ]
            )
            self.assertEqual(
                [ref.as_string() for ref in deduped], ["agent_runtime:x", "model_provider:p"]
            )

    def test_panel_topology_manifest_roles_and_error_edges(self) -> None:
        from praxist.core import panel_topology
        from praxist.core.registry import PluginMetadata, SelectedPlugin

        manifest = {
            "topology": {
                "topology_ref": "panel_topology:test",
                "roles": [
                    {
                        "role_ref": "role:builder",
                        "role_id": "builder",
                        "legacy_role_id": "builder_legacy",
                        "role_kind": "pi",
                        "model_profile_ref": "profile:builder",
                    },
                    {"bad": "ignored"},
                    "ignored",
                ],
                "modes": {
                    "mini": ["builder", "role:missing_role", "literal"],
                    "full": ["role:builder"],
                    "bad": "ignored",
                },
                "rounds": [
                    {
                        "round_id": "round1",
                        "role_refs": ["role:builder"],
                        "parallelism": 2,
                        "timeout_seconds": 30,
                    }
                ],
                "high_stakes_triggers": ["Generally Dominant"],
                "chair_role_ref": "role:chair",
                "peer_role_rotation": [" builder ", "skeptic"],
                "peer_role_descriptions": {"builder": " Build candidates. "},
            }
        }
        spec = panel_topology.panel_topology_from_manifest("panel_topology:test", manifest)
        self.assertEqual(spec.roles_for_mode("full"), ["builder"])
        self.assertEqual(spec.roles_for_mode("unknown"), ["builder", "missing_role", "literal"])
        self.assertEqual(spec.role_refs_for_mode("full"), ["role:builder"])
        self.assertEqual(spec.role_spec("builder").legacy_role_id, "builder_legacy")
        with self.assertRaises(KeyError):
            spec.role_spec("missing")
        self.assertTrue(
            spec.has_high_stakes_signal(
                {
                    "claim_ledger_digest": {
                        "active": [{"title": "Generally dominant claim", "boundary": ""}]
                    },
                    "retired_claims": [{"title": "old", "boundary": ""}],
                }
            )
        )
        self.assertFalse(spec.has_high_stakes_signal("not a dict"))
        self.assertIn("rounds", spec.to_dict())
        fallback_spec = spec.role_specs_for_mode("mini")[-1]
        self.assertEqual(fallback_spec.role_ref, "role:literal")
        self.assertEqual(spec.peer_role_rotation, ("builder", "skeptic"))
        self.assertEqual(spec.peer_role_descriptions, {"builder": "Build candidates."})

        for bad_manifest, message in (
            ({}, "missing a topology contract"),
            ({"topology": {"modes": {}, "rounds": [{"round_id": "r"}]}}, "non-empty modes"),
            ({"topology": {"modes": {"mini": "bad"}, "rounds": [{"round_id": "r"}]}}, "valid mode"),
            ({"topology": {"modes": {"mini": []}, "rounds": []}}, "non-empty rounds"),
            ({"topology": {"modes": {"mini": []}, "rounds": ["bad"]}}, "rounds must be objects"),
            ({"topology": {"modes": {"mini": []}, "rounds": [{}]}}, "missing round_id"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                panel_topology.panel_topology_from_manifest("panel_topology:bad", bad_manifest)

        valid_shape = {
            "topology": {
                "modes": {"mini": []},
                "rounds": [{"round_id": "r"}],
            }
        }
        for extra, message in (
            ({"peer_role_descriptions": "bad"}, "peer_role_descriptions must be a mapping"),
            ({"peer_role_descriptions": {"": "desc"}}, "keys must be non-empty"),
            ({"peer_role_descriptions": {"builder": ""}}, "must be a non-empty string"),
            ({"peer_role_rotation": "bad"}, "peer_role_rotation must be a list"),
            ({"peer_role_rotation": ["builder", ""]}, "entries must be non-empty strings"),
            ({"prompts_dir": 3}, "prompts_dir must be a string"),
            ({"prompts_dir": "missing"}, "plugin_path is unknown"),
            ({"prompts_dir": "/definitely/not/a/panel/prompts"}, "prompts_dir does not exist"),
        ):
            bad_manifest = {"topology": {**valid_shape["topology"], **extra}}
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                panel_topology.panel_topology_from_manifest("panel_topology:bad", bad_manifest)

        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = Path(tmp)
            prompts = plugin_root / "prompts"
            prompts.mkdir()
            prompt_manifest = {
                "topology": {
                    **valid_shape["topology"],
                    "prompts_dir": "prompts",
                }
            }
            prompt_spec = panel_topology.panel_topology_from_manifest(
                "panel_topology:prompted",
                prompt_manifest,
                plugin_path=plugin_root,
            )
            self.assertEqual(prompt_spec.prompts_dir, prompts.resolve())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "bad.yaml"
            manifest_path.write_text("bad\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest must be an object"):
                panel_topology._read_manifest(manifest_path)

        selected = SelectedPlugin(
            metadata=PluginMetadata(
                schema_version=1,
                name="bad",
                kind="agent_runtime",
                version="0.1.0",
                protocol_version=1,
                stability="v1_stable",
                description="bad",
                compatibility={},
                dependencies=[],
                capabilities=[],
                entrypoint=None,
                code=[],
                assets=[],
            ),
            source="bundled",
            path="/tmp",
            content_hash="sha256:" + "0" * 64,
            selected_by=["test"],
        )
        registry = SimpleNamespace(descriptor=lambda _kind, _name: selected)
        with self.assertRaisesRegex(ValueError, "kind panel_topology"):
            panel_topology._selected_topology_plugin(
                "agent_runtime:bad",
                registry=registry,
                workspace=None,
            )
        with self.assertRaisesRegex(ValueError, "resolved as"):
            panel_topology._selected_topology_plugin(
                "panel_topology:bad",
                registry=registry,
                workspace=None,
            )


if __name__ == "__main__":
    unittest.main()
