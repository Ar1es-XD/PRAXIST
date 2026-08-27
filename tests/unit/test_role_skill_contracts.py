from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class RoleSkillContractsTest(unittest.TestCase):
    def test_task_role_skill_loading_hashing_and_boundary_guards(self) -> None:
        from praxist.core import role_skills

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            role_dir = root / "roles" / "builder"
            role_dir.mkdir(parents=True)
            (role_dir / "skill.md").write_text(
                "# Builder\n\nFixed review questions:\n- What worked?\n- What failed?\n\nText",
                encoding="utf-8",
            )
            (role_dir / "kb.md").write_text("private", encoding="utf-8")
            (role_dir / "role.yaml").write_text(
                """
role:
  role_id: builder
  legacy_role_id: builder_legacy
  display_name: Builder PI
  role_kind: panel
  output_schema_ref: schema:v1
  default_model_profile_ref: pi_model
  tool_scope: [mcp__memory-tools__query]
  private_kb: kb.md
""",
                encoding="utf-8",
            )
            skill = role_skills.load_role_skill(
                "task_role:builder",
                task_project_path=root,
            )
            context = skill.to_prompt_context()
            self.assertEqual(skill.role_id, "builder")
            self.assertEqual(skill.legacy_role_id, "builder_legacy")
            self.assertEqual(context["legacy_role_id"], "builder_legacy")
            self.assertEqual(skill.display_name, "Builder PI")
            self.assertEqual(skill.fixed_questions, ("What worked?", "What failed?"))
            self.assertEqual(
                [Path(p).resolve() for p in context["private_kb_paths"]],
                [(role_dir / "kb.md").resolve()],
            )
            self.assertRegex(skill.content_hash, r"^sha256:[0-9a-f]{64}$")

            with patch.dict(os.environ, {"PRAXIST_TASK_PROJECT_PATH": str(root)}, clear=False):
                self.assertEqual(
                    role_skills.load_role_skill("task_role:builder").role_id, "builder"
                )
            with self.assertRaises(ValueError):
                role_skills.load_role_skill("task_role:../escape", task_project_path=root)
            with self.assertRaises(ValueError):
                role_skills.load_role_skill("task_role:missing", task_project_path=root)
            (root / "roles" / "no_skill").mkdir()
            with self.assertRaises(ValueError):
                role_skills.load_role_skill("task_role:no_skill", task_project_path=root)
            with self.assertRaises(ValueError):
                role_skills.load_role_skill("task_role:builder")

            self.assertEqual(
                role_skills._title_from_role_id("external_validity"), "External Validity"
            )
            self.assertEqual(
                role_skills._fixed_questions({"fixed_questions": [" Q ", ""]}, ""), [" Q "]
            )
            self.assertEqual(role_skills._extract_fixed_questions("no block"), [])
            with self.assertRaises(ValueError):
                role_skills._safe_plugin_file(role_dir, "../bad")

    def test_generic_role_plugin_loading_uses_registry_descriptor_and_assets(self) -> None:
        from praxist.core import role_skills

        with tempfile.TemporaryDirectory() as tmp:
            plugin = Path(tmp) / "role_plugin"
            (plugin / "private_kb").mkdir(parents=True)
            (plugin / "private_kb" / "a.md").write_text("A", encoding="utf-8")
            (plugin / "skill.md").write_text("Skill body", encoding="utf-8")
            (plugin / "plugin.yaml").write_text(
                """
role:
  legacy_role_id: skeptic
  private_kb: []
assets:
  - private_kb/*.md
""",
                encoding="utf-8",
            )
            selected = SimpleNamespace(
                path=str(plugin),
                content_hash="hash",
                metadata=SimpleNamespace(
                    kind="role",
                    name="skeptic",
                    capabilities=["role.skill"],
                ),
            )
            registry = SimpleNamespace(descriptor=lambda kind, name: selected)
            skill = role_skills.load_role_skill("role:skeptic", registry=registry)
            self.assertEqual(skill.role_id, "skeptic")
            self.assertEqual(skill.legacy_role_id, "skeptic")
            self.assertEqual(skill.display_name, "Skeptic")
            self.assertEqual(skill.private_kb_paths, ((plugin / "private_kb" / "a.md").resolve(),))

            selected.metadata.kind = "tool_server"
            with self.assertRaises(ValueError):
                role_skills.load_role_skill("role:skeptic", registry=registry)
            selected.metadata.kind = "role"
            selected.metadata.capabilities = []
            with self.assertRaises(ValueError):
                role_skills.load_role_skill("role:skeptic", registry=registry)
            selected.metadata.capabilities = ["role.skill"]
            with self.assertRaises(ValueError):
                role_skills.load_role_skill("tool:x", registry=registry)


if __name__ == "__main__":
    unittest.main()
