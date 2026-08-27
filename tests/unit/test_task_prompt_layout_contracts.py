"""Contracts for task-local prompt layout overrides."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from praxist.plugins.workflow_stages.research_loop.backend.generation_loop import (
    GenerationLoop,
)
from praxist.task_spec import load_task_spec


class TaskPromptLayoutContractsTest(unittest.TestCase):
    def test_task_spec_resolves_base_prompt_override_relative_to_task_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "prompts" / "mle_base.jinja2"
            base.parent.mkdir()
            base.write_text("Task base for {{ peer_id }}\n", encoding="utf-8")
            self._write_task_yaml(root, base_template="prompts/mle_base.jinja2")

            spec = load_task_spec(str(root / "task.yaml"))

            self.assertEqual(spec.get_prompt_base_path(Path("/default/base.jinja2")), base)

    def test_task_spec_falls_back_to_default_base_prompt_when_override_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_task_yaml(root, base_template=None)

            spec = load_task_spec(str(root / "task.yaml"))
            default = root / "default_base.jinja2"

            self.assertEqual(spec.get_prompt_base_path(default), default)

    def test_task_spec_rejects_missing_base_prompt_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_task_yaml(root, base_template="missing_base.jinja2")

            spec = load_task_spec(str(root / "task.yaml"))

            with self.assertRaisesRegex(FileNotFoundError, "prompt_layout.base_template"):
                spec.get_prompt_base_path(root / "default_base.jinja2")

    def test_task_spec_resolves_generation_prompt_override_relative_to_task_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generation = root / "prompts" / "mle_generation.jinja2"
            generation.parent.mkdir()
            generation.write_text("Task generation for {{ peer_id }}\n", encoding="utf-8")
            self._write_task_yaml(root, generation_template="prompts/mle_generation.jinja2")

            spec = load_task_spec(str(root / "task.yaml"))

            self.assertEqual(
                spec.get_prompt_generation_path(Path("/default/prompt_generation.jinja2")),
                generation,
            )

    def test_task_spec_rejects_missing_generation_prompt_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_task_yaml(root, generation_template="missing_generation.jinja2")

            spec = load_task_spec(str(root / "task.yaml"))

            with self.assertRaisesRegex(
                FileNotFoundError,
                "prompt_layout.generation_template",
            ):
                spec.get_prompt_generation_path(root / "default_generation.jinja2")

    def test_generation_loop_uses_task_local_base_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "prompts" / "mle_base.jinja2"
            base.parent.mkdir()
            base.write_text("Task-local base\n", encoding="utf-8")
            generation = root / "prompts" / "mle_generation.jinja2"
            generation.write_text("Task-local generation\n", encoding="utf-8")
            self._write_task_yaml(
                root,
                base_template="prompts/mle_base.jinja2",
                generation_template="prompts/mle_generation.jinja2",
            )
            spec = load_task_spec(str(root / "task.yaml"))

            loop = GenerationLoop(
                task_spec=spec,
                workspace=root / "workspace",
                run_dir=root / "run",
                local_mode=True,
                tool_server_refs=[],
            )

            self.assertEqual(loop.base_template, base)
            self.assertEqual(loop.gen_template, generation)

    @staticmethod
    def _write_task_yaml(
        root: Path,
        *,
        base_template: str | None = None,
        generation_template: str | None = None,
    ) -> None:
        prompt_layout = ""
        if base_template is not None or generation_template is not None:
            lines = ["", "prompt_layout:"]
            if base_template is not None:
                lines.append(f"  base_template: {base_template}")
            if generation_template is not None:
                lines.append(f"  generation_template: {generation_template}")
            prompt_layout = "\n".join(lines) + "\n"
        (root / "description.md").write_text("Description\n", encoding="utf-8")
        (root / "prompt_task.jinja2").write_text("Task prompt\n", encoding="utf-8")
        (root / "task.yaml").write_text(
            f"""
task_id: prompt_layout_contract
task_name: Prompt Layout Contract
description_file: description.md
research_direction: Generic prompt layout contract.
evaluation:
  primary_metric: score
  direction: maximize
generation_policy:
  max_generations: 1
  cohort_size: 1
  per_generation_hours: 1
  promote_top_k: 1
pi_agent:
  enabled: false
multi_pi:
  enabled: false
praxist_plugins:
  tools: []
{prompt_layout}
""".lstrip(),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
