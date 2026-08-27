from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from praxist.core.panel_topology import panel_topology_for_ref
from praxist.core.role_skills import load_role_skill
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import BuilderPI
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.role_bindings import (
    instantiate_pi_roles,
)


class Step15PanelRoleMigrationTest(unittest.TestCase):
    def test_role_skill_loader_reads_manifest_skill_and_private_kb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = _write_task_role_fixture(Path(tmp))
            builder = load_role_skill("task_role:builder_pi", task_project_path=task_root)

            self.assertEqual(builder.role_id, "builder")
            self.assertEqual(builder.role_kind, "pi")
            self.assertEqual(builder.output_schema_ref, "core:pi_memo.v1")
            self.assertEqual(builder.default_model_profile_ref, "strong_reasoner")
            self.assertIn("memory_tools.panel_read", builder.tool_scope)
            self.assertIn(
                "What is the strongest mechanism explanation right now?", builder.fixed_questions
            )
            self.assertEqual(len(builder.private_kb_paths), 1)
            self.assertTrue(builder.private_kb_paths[0].exists())
            self.assertIn("roles/builder_pi/private_kb", builder.private_kb_paths[0].as_posix())

            chair = load_role_skill("task_role:chair", task_project_path=task_root)
            self.assertEqual(chair.role_kind, "chair")
            self.assertEqual(chair.output_schema_ref, "core:research_agenda.v1")
            self.assertIn("Do not introduce new scientific claims", chair.skill_markdown)

    def test_panel_topology_exports_role_refs_and_visibility_plan(self) -> None:
        topology = panel_topology_for_ref("panel_topology:legacy_multi_pi_two_round")

        self.assertEqual(topology.roles_for_mode("full"), ["builder", "skeptic", "portfolio"])
        self.assertEqual(
            topology.role_refs_for_mode("full"),
            ["task_role:builder_pi", "task_role:skeptic_pi", "task_role:portfolio_pi"],
        )
        self.assertEqual(topology.role_spec("builder").private_pack_key, "builder")
        self.assertEqual(
            topology.role_spec("external_validity").role_ref, "task_role:external_validity_pi"
        )
        rounds = {round_spec.round_id: round_spec for round_spec in topology.rounds}
        self.assertEqual(rounds["independent_memos"].visibility, "private_pack")
        self.assertEqual(rounds["cross_review"].visibility, "anonymized_peer_memos")
        self.assertEqual(rounds["chair_synthesis"].role_refs, ["task_role:chair"])
        self.assertIn("task_role:external_validity_pi", rounds["independent_memos"].role_refs)

    def test_legacy_pi_uses_role_skill_prompt_and_plugin_private_kb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            task_root = _write_task_role_fixture(tmp_path / "task")
            with patch.dict(
                "os.environ", {"PRAXIST_TASK_PROJECT_PATH": str(task_root)}, clear=False
            ):
                pi = BuilderPI(
                    run_dir=tmp_path,
                    workspace=Path.cwd(),
                    model="fake-model",
                    max_runtime_minutes=1,
                )

                entries = pi.load_private_kb(top_k=3, query_blob="mechanism")
                self.assertTrue(entries)
                self.assertIn("roles/builder_pi/private_kb", entries[0]["source_relative_path"])
                self.assertEqual(
                    pi.fixed_questions()[0],
                    "What is the strongest mechanism explanation right now?",
                )

                prompt = pi.render_prompt(
                    shared_core={"claim_ledger_digest": {"active": []}},
                    private_pack=[],
                    private_kb_entries=[],
                    target_decisions=["allocate gen1 peer contracts"],
                )
            self.assertIn("## RoleSkill contract", prompt)
            self.assertIn("# Builder PI", prompt)
            self.assertIn("What is the strongest mechanism explanation right now?", prompt)

    def test_role_bindings_instantiate_from_topology_role_specs(self) -> None:
        topology = panel_topology_for_ref("panel_topology:legacy_multi_pi_two_round")
        specs = topology.role_specs_for_mode("high_stakes")
        with tempfile.TemporaryDirectory() as tmp:
            pis = instantiate_pi_roles(
                specs,
                run_dir=Path(tmp),
                workspace=Path.cwd(),
                model="fake-model",
                max_runtime_minutes=1,
                mcp_servers=None,
                stop_check_fn=None,
                premium_mode=False,
            )

        self.assertEqual(
            [pi.role_name for pi in pis],
            ["builder", "skeptic", "portfolio", "external_validity"],
        )
        self.assertEqual(
            [pi.role_ref for pi in pis],
            [
                "task_role:builder_pi",
                "task_role:skeptic_pi",
                "task_role:portfolio_pi",
                "task_role:external_validity_pi",
            ],
        )


def _write_task_role_fixture(root: Path) -> Path:
    for role_name, role_kind, schema in (
        ("builder_pi", "pi", "core:pi_memo.v1"),
        ("chair", "chair", "core:research_agenda.v1"),
    ):
        role_dir = root / "roles" / role_name
        (role_dir / "private_kb").mkdir(parents=True, exist_ok=True)
        role_dir.joinpath("role.yaml").write_text(
            "\n".join(
                [
                    "role:",
                    f"  role_id: {'builder' if role_name == 'builder_pi' else 'chair'}",
                    f"  display_name: {'Builder PI' if role_name == 'builder_pi' else 'Chair'}",
                    f"  role_kind: {role_kind}",
                    "  default_model_profile_ref: strong_reasoner",
                    f"  output_schema_ref: {schema}",
                    "  tool_scope:",
                    "    - memory_tools.panel_read",
                    "  private_kb:",
                    "    - private_kb/notes.md",
                    "  fixed_questions:",
                    "    - What is the strongest mechanism explanation right now?",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        role_dir.joinpath("skill.md").write_text(
            "# Builder PI\nWhat is the strongest mechanism explanation right now?\n"
            if role_name == "builder_pi"
            else "# Chair\nDo not introduce new scientific claims.\n",
            encoding="utf-8",
        )
        role_dir.joinpath("private_kb/notes.md").write_text(
            "mechanism notes\n",
            encoding="utf-8",
        )
    return root


if __name__ == "__main__":
    unittest.main()
