"""Lock the CI workflow contract.

Every pull request to the canonical branch must run GitHub checks. Guardrail
commands stay in sync with ``AGENTS.md`` so local and hosted validation agree.
"""

from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github/workflows/ci.yml"
AGENTS_MD = REPO_ROOT / "AGENTS.md"
PYTHON_VERSION = REPO_ROOT / ".python-version"


def _load_workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _all_run_blocks(workflow: dict) -> list[str]:
    blocks: list[str] = []
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            run = step.get("run")
            if isinstance(run, str):
                blocks.append(run)
    return blocks


class WorkflowFileExists(unittest.TestCase):
    def test_workflow_file_present(self) -> None:
        self.assertTrue(WORKFLOW.exists(), f"{WORKFLOW.relative_to(REPO_ROOT)} is missing")

    def test_workflow_yaml_is_parseable(self) -> None:
        _load_workflow()  # raises if malformed


class WorkflowTriggers(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = _load_workflow()
        # PyYAML parses YAML's `on:` as Python `True` because it's the YAML
        # boolean. Workflows in the wild use `on: ...` so we accept either key.
        self.triggers = self.workflow.get("on") or self.workflow.get(True) or {}

    def test_triggers_on_pull_request(self) -> None:
        self.assertIn("pull_request", self.triggers)

    def test_pull_request_triggers_include_main(self) -> None:
        pull_request = self.triggers.get("pull_request") or {}
        branches = set(pull_request.get("branches") or [])
        self.assertIn("main", branches)

    def test_triggers_on_push_to_core_arch_or_main(self) -> None:
        push = self.triggers.get("push") or {}
        branches = push.get("branches") or []
        self.assertTrue(
            any(b in branches for b in ("core_arch", "main")),
            f"push trigger must include core_arch or main, got {branches}",
        )


class WorkflowCommandsMatchGuardrails(unittest.TestCase):
    """The CI commands must match ``scripts/dev/run_guardrails.py`` exactly,
    so dev-machine pre-commit and CI never disagree."""

    def setUp(self) -> None:
        self.workflow = _load_workflow()
        self.runs = "\n".join(_all_run_blocks(self.workflow))

    def test_runs_ruff_check(self) -> None:
        self.assertRegex(self.runs, r"ruff check\b")

    def test_runs_ruff_format_check(self) -> None:
        self.assertRegex(self.runs, r"ruff format\s+--check\b")

    def test_runs_pyrefly(self) -> None:
        self.assertRegex(self.runs, r"\bpyrefly\s+check\b")

    def test_runs_pytest(self) -> None:
        self.assertRegex(self.runs, r"\bpytest\b")

    def test_runs_unit_coverage(self) -> None:
        self.assertRegex(self.runs, r"run_test_coverage\.py\s+unit\b")
        self.assertRegex(self.runs, r"--fail-under-statements\s+95\b")

    def test_runs_integration_coverage(self) -> None:
        self.assertRegex(self.runs, r"run_test_coverage\.py\s+integration\b")


class WorkflowPythonMatrix(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = _load_workflow()

    def test_python_matrix_covers_311_and_312(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("3.11", text)
        self.assertIn("3.12", text)


class PythonVersionFile(unittest.TestCase):
    def test_python_version_file_exists(self) -> None:
        self.assertTrue(PYTHON_VERSION.exists(), ".python-version must be tracked")

    def test_python_version_aligns_with_pyproject(self) -> None:
        declared = PYTHON_VERSION.read_text(encoding="utf-8").strip()
        self.assertRegex(declared, r"^3\.1[1-9](\.\d+)?$")
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        requires = pyproject.get("project", {}).get("requires-python", "")
        # Just sanity: declared minor must satisfy requires-python lower bound.
        m = re.search(r">=\s*3\.(\d+)", requires)
        self.assertIsNotNone(m)
        assert m is not None  # for type-checkers
        self.assertGreaterEqual(int(declared.split(".")[1]), int(m.group(1)))


class AgentsMdPrecommitChecklist(unittest.TestCase):
    """AGENTS.md must spell out the same four guardrail commands so that a
    contributor reading the contract doesn't need to discover the CI YAML."""

    def setUp(self) -> None:
        self.text = AGENTS_MD.read_text(encoding="utf-8")

    def test_contains_pre_commit_section_header(self) -> None:
        self.assertRegex(self.text, r"(?im)^##\s+\d+\.\s+Pre-?commit\b")

    def test_lists_ruff_check(self) -> None:
        self.assertRegex(self.text, r"ruff check\b")

    def test_lists_ruff_format_check(self) -> None:
        self.assertRegex(self.text, r"ruff format\s+--check\b")

    def test_lists_pyrefly(self) -> None:
        self.assertRegex(self.text, r"\bpyrefly\s+check\b")

    def test_lists_pytest(self) -> None:
        self.assertRegex(self.text, r"\bpytest\b")

    def test_lists_unit_and_integration_coverage(self) -> None:
        self.assertRegex(self.text, r"run_test_coverage\.py\s+unit\s+integration\b")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
