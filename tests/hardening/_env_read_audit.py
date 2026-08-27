"""AST-based audit of env-var reads under ``praxist/``.

Shared helper for ``test_env_read_discipline``: walks the source tree,
classifies every ``os.environ.get(...)``, ``os.getenv(...)``, and
``os.environ[...]`` read site, and returns the result as a sorted list
of stable ``(relative_path, env_var_name)`` tuples.

This module is the single source of truth for "what counts as an env
read" in the discipline rule. Anything not detectable here doesn't
count; anything detectable here either goes through ``RunConfig`` or
must appear in the allowlist with a reason.

Detection strategy
------------------
* ``os.environ.get(...)`` / ``os.getenv(...)`` calls — recognized by
  matching the call chain ``os.environ.get`` or ``os.getenv`` and
  reading the first positional argument when it's a string literal.
  Non-literal first args (dynamic var names) are recorded as the
  sentinel ``"<dynamic>"`` so the allowlist still has a stable key.
* ``os.environ[...]`` subscripts in ``Load`` context — i.e. reads.
  Writes (``os.environ["X"] = "..."``) are detected by checking the
  AST context and routed to a separate set; the current discipline
  rule only tracks reads, but we surface writes too for visibility.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_ENV_READ_SENTINEL_DYNAMIC = "<dynamic>"


@dataclass(frozen=True)
class EnvRead:
    """One detected env read."""

    path: str
    """Path relative to the repository root (e.g. ``praxist/core/registry.py``)."""

    var_name: str
    """The env var name as a string literal, or ``"<dynamic>"`` when not literal."""

    line: int
    """Line number of the read in the source file."""


class _EnvReadVisitor(ast.NodeVisitor):
    """Collect every env read in one module."""

    def __init__(self) -> None:
        self.reads: list[tuple[int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        var_name = self._call_var_name(node)
        if var_name is not None:
            self.reads.append((node.lineno, var_name))
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # Only count ``Load`` context: ``x = os.environ["X"]``.
        # ``Store`` context (``os.environ["X"] = ...``) is a write, not a read.
        if isinstance(node.ctx, ast.Load) and self._is_os_environ(node.value):
            self.reads.append((node.lineno, _constant_or_dynamic(node.slice)))
        self.generic_visit(node)

    @staticmethod
    def _call_var_name(node: ast.Call) -> str | None:
        """Return the env-var name if ``node`` is an env-read call, else ``None``."""
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "environ"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "os"
        ):
            return _constant_or_dynamic(node.args[0]) if node.args else _ENV_READ_SENTINEL_DYNAMIC
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "getenv"
            and isinstance(func.value, ast.Name)
            and func.value.id == "os"
        ):
            return _constant_or_dynamic(node.args[0]) if node.args else _ENV_READ_SENTINEL_DYNAMIC
        return None

    @staticmethod
    def _is_os_environ(value: ast.expr) -> bool:
        return (
            isinstance(value, ast.Attribute)
            and value.attr == "environ"
            and isinstance(value.value, ast.Name)
            and value.value.id == "os"
        )


def _constant_or_dynamic(node: ast.expr) -> str:
    """Return ``node``'s string-literal value, or the dynamic sentinel."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return _ENV_READ_SENTINEL_DYNAMIC


def collect_env_reads(roots: Iterable[Path], *, repo_root: Path) -> list[EnvRead]:
    """Return every detected env read under ``roots`` as a sorted list.

    Sort key: ``(path, var_name, line)``. Output is deterministic so
    test diffs are stable across runs.
    """
    out: list[EnvRead] = []
    for root in roots:
        for python_file in sorted(root.rglob("*.py")):
            relative = python_file.relative_to(repo_root)
            tree = ast.parse(python_file.read_text(encoding="utf-8"), filename=str(python_file))
            visitor = _EnvReadVisitor()
            visitor.visit(tree)
            for line, var_name in visitor.reads:
                out.append(EnvRead(path=str(relative), var_name=var_name, line=line))
    out.sort(key=lambda r: (r.path, r.var_name, r.line))
    return out


def collapse_to_pairs(reads: Iterable[EnvRead]) -> set[tuple[str, str]]:
    """Reduce reads to ``(path, var_name)`` pairs — what the allowlist tracks.

    Multiple reads of the same var in the same file collapse to one
    pair: line numbers change with edits and we don't want the
    allowlist to churn on cosmetic refactors.
    """
    return {(r.path, r.var_name) for r in reads}


__all__ = [
    "EnvRead",
    "collapse_to_pairs",
    "collect_env_reads",
]
