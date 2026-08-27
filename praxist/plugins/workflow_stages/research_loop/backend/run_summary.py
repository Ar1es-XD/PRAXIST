"""Helpers for writing run_summary.json without discarding task annotations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from praxist.core.storage import write_json


def preserve_existing_summary_extensions(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Merge a new core summary over existing task-supplied extension fields."""

    existing = _read_existing_summary(path)
    if not existing:
        return dict(summary)
    merged = dict(existing)
    merged.update(summary)
    return merged


def write_run_summary(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Write ``run_summary.json`` while preserving non-core task summary fields."""

    merged = preserve_existing_summary_extensions(path, summary)
    write_json(path, merged)
    return merged


def _read_existing_summary(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return raw if isinstance(raw, dict) else {}
