"""Source snapshot helpers for replay drift detection."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

SOURCE_PATTERNS = (
    "praxist/core/**/*.py",
    "praxist/infrastructure/**/*.py",
    "praxist/testing/**/*.py",
    "praxist/plugins/**/*.py",
    "praxist/plugins/workflow_stages/research_loop/backend/**/*.jinja2",
    "praxist/plugins/workflow_stages/research_loop/backend/**/*.md",
    "praxist/run.py",
    "praxist/deliver.py",
    "praxist/task_spec.py",
    "praxist/config.py",
)


def build_core_source_snapshot(repo_root: Path | None = None) -> dict[str, Any]:
    """Hash execution-critical Praxist source files for replay drift detection."""
    root = repo_root or Path(__file__).resolve().parents[2]
    paths: set[Path] = set()
    for pattern in SOURCE_PATTERNS:
        paths.update(path for path in root.glob(pattern) if path.is_file())

    hasher = hashlib.sha256()
    files = []
    for path in sorted(paths, key=lambda item: str(item.relative_to(root))):
        rel = str(path.relative_to(root)).replace("\\", "/")
        files.append(rel)
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")

    return {
        "workspace_hash": "sha256:" + hasher.hexdigest(),
        "git_commit": _git_commit(root),
        "source_hash_algorithm": "sha256",
        "source_file_count": len(files),
        "source_patterns": list(SOURCE_PATTERNS),
    }


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None
