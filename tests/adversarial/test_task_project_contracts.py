from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from praxist.core.role_skills import load_role_skill
from praxist.core.task_project import (
    resolve_task_project,
    task_project_global_plugin_refs,
)
from praxist.core.tool_servers import tool_server_refs_from_task_descriptor


class TaskProjectAdversarialContracts(unittest.TestCase):
    def test_task_project_manifest_does_not_hash_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            task_root = base / "task"
            (task_root / "assets").mkdir(parents=True)
            task_root.joinpath("task.yaml").write_text(
                "\n".join(
                    [
                        "task_id: adversarial_task",
                        "praxist_plugins:",
                        "  workflow_stage: workflow_stage:research_loop",
                    ]
                ),
                encoding="utf-8",
            )
            outside = base / "outside_secret.txt"
            outside.write_text("outside-secret-material", encoding="utf-8")
            task_root.joinpath("assets", "leak.txt").symlink_to(outside)

            try:
                project = resolve_task_project(task_root, workspace=base)
            except ValueError:
                return

            outside_hash = hashlib.sha256(outside.read_bytes()).hexdigest()
            leaked = [
                item
                for item in project.manifest["files"]
                if item["sha256"] == outside_hash or item["path"] == "assets/leak.txt"
            ]
            self.assertEqual(
                leaked,
                [],
                "task manifests must reject or skip symlink targets outside the task root",
            )

    def test_task_role_ref_cannot_escape_roles_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "task"
            task_root.joinpath("roles").mkdir(parents=True)
            outside_role = task_root / "outside_role"
            outside_role.mkdir()
            outside_role.joinpath("role.yaml").write_text(
                "role_id: outside_role\nrole_kind: adversarial\n",
                encoding="utf-8",
            )
            outside_role.joinpath("skill.md").write_text(
                "This role is outside the task roles directory.\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_role_skill(
                    "task_role:../outside_role",
                    task_project_path=task_root,
                )

    def test_task_descriptor_tool_servers_alias_is_runtime_visible(self) -> None:
        descriptor = {
            "task_id": "adversarial_task",
            "praxist_plugins": {
                "workflow_stage": "workflow_stage:research_loop",
                "tool_servers": {
                    "memory": {
                        "ref": "tool_server:memory_tools",
                        "enabled": True,
                    }
                },
                "tools": [],
            },
        }

        global_refs = {ref.as_string() for ref in task_project_global_plugin_refs(descriptor)}
        runtime_refs = set(tool_server_refs_from_task_descriptor(descriptor))
        self.assertIn("tool_server:memory_tools", global_refs)
        self.assertIn(
            "tool_server:memory_tools",
            runtime_refs,
            "any tool server accepted by task-project plugin resolution must also be visible to runtime tool activation",
        )
