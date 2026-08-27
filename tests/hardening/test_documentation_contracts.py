from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DOCUMENTED_PATHS = (
    REPO_ROOT / "praxist",
    REPO_ROOT / "templates" / "tasks",
    REPO_ROOT / "scripts",
)

PATCH_NOTE_PATTERN = re.compile(
    r"\\b(?:R\\d+(?:[-#]\\w+)?|Round\\s+\\d+|20\\d{2}-\\d{2}-\\d{2}\\s+run)\\b",
    re.IGNORECASE,
)


def _python_files() -> list[Path]:
    files: list[Path] = []
    for path in DOCUMENTED_PATHS:
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(child for child in path.rglob("*.py") if "__pycache__" not in child.parts)
    return sorted(set(files))


class DocumentationContracts(unittest.TestCase):
    def test_public_top_level_python_api_has_docstrings(self) -> None:
        missing: list[str] = []
        for path in _python_files():
            module = ast.parse(path.read_text(encoding="utf-8"))
            for node in module.body:
                if not isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    continue
                if node.name.startswith("_"):
                    continue
                if not ast.get_docstring(node):
                    rel = path.relative_to(REPO_ROOT)
                    missing.append(f"{rel}:{node.lineno}:{node.name}")
        self.assertEqual(missing, [])

    def test_public_docstrings_are_stable_contract_text(self) -> None:
        offenders: list[str] = []
        for path in _python_files():
            module = ast.parse(path.read_text(encoding="utf-8"))
            for node in module.body:
                if not isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    continue
                if node.name.startswith("_"):
                    continue
                docstring = ast.get_docstring(node) or ""
                if PATCH_NOTE_PATTERN.search(docstring):
                    rel = path.relative_to(REPO_ROOT)
                    offenders.append(f"{rel}:{node.lineno}:{node.name}")
        self.assertEqual(offenders, [])

    def test_contributor_docs_cover_current_boundaries(self) -> None:
        required_docs = [
            REPO_ROOT / "AGENTS.md",
            REPO_ROOT / "docs" / "guides" / "contributing.md",
            REPO_ROOT / "docs" / "guides" / "operators.md",
            REPO_ROOT / "docs" / "guides" / "task-projects.md",
            REPO_ROOT / "docs" / "guides" / "plugins.md",
            REPO_ROOT / "docs" / "guides" / "agent-runtimes.md",
            REPO_ROOT / "docs" / "guides" / "model-providers.md",
            REPO_ROOT / "docs" / "guides" / "budget-policies.md",
            REPO_ROOT / "docs" / "guides" / "credentials.md",
            REPO_ROOT / "docs" / "guides" / "workflow-stages.md",
            REPO_ROOT / "docs" / "guides" / "tool-servers.md",
            REPO_ROOT / "docs" / "guides" / "legacy-migration.md",
            REPO_ROOT / "docs" / "guides" / "costs.md",
        ]
        missing = [
            path.relative_to(REPO_ROOT).as_posix() for path in required_docs if not path.exists()
        ]
        self.assertEqual(missing, [])

        agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        for phrase in (
            "praxist",
            "python -m praxist.run",
            "--task-path",
            "praxist/plugins",
            "Result Preservation Principle",
            "PromptLayout V1",
            "python scripts/build_docs_site.py",
        ):
            self.assertIn(phrase, agents)

        mkdocs = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        for guide in required_docs[1:]:
            self.assertIn(guide.relative_to(REPO_ROOT / "docs").as_posix(), mkdocs)
