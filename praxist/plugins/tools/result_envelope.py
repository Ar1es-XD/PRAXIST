"""Shared helpers for bounded MCP tool responses.

Tool calls are expensive because their returned text is fed back into the
agent context. These helpers keep the inline response small while preserving a
complete, replayable JSON artifact under the active run directory.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

TOOL_OUTPUT_SCHEMA_VERSION = "praxist.tool_output_envelope.v1"
TOOL_RESULT_SCHEMA_VERSION = "praxist.tool_result_artifact.v1"
TOOL_RESULT_CHUNK_SCHEMA_VERSION = "praxist.tool_result_chunk.v1"
TOOL_RESULT_REF_PREFIX = "tool_result:"
DEFAULT_CHUNK_CHARS = 4000
MAX_CHUNK_CHARS = 50000
DEFAULT_INLINE_LIST_LIMIT = 20
MAX_INLINE_LIST_LIMIT = 100

_DANGEROUS_RUN_DIRS = {
    Path("/"),
    Path("/etc"),
    Path("/proc"),
    Path("/sys"),
    Path("/dev"),
    Path("/run"),
    Path("/usr"),
    Path("/var"),
    Path("/root"),
    Path("/home"),
    Path("/boot"),
}


def active_run_dir() -> Path | None:
    """Return the current run directory if one is safely declared.

    Praxist stages set ``PRAXIST_RUN_DIR`` and ``LOCAL_STORE_DIR`` before launching
    tools. Tests and standalone utilities may omit both; in that case helpers
    still return bounded inline data but skip the full-result artifact.
    """

    for env_name in ("PRAXIST_RUN_DIR", "LOCAL_STORE_DIR"):
        raw = os.environ.get(env_name, "").strip()
        if not raw:
            continue
        try:
            resolved = Path(raw).expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if resolved in _DANGEROUS_RUN_DIRS:
            continue
        return resolved
    return None


def _tool_results_dir() -> Path | None:
    run_dir = active_run_dir()
    if run_dir is None:
        return None
    return run_dir / "tool_results"


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _safe_tool_slug(tool_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", tool_name).strip("._-")
    return slug or "tool"


def store_tool_result(tool_name: str, payload: Any) -> str | None:
    """Persist a full tool payload and return a stable local result ref.

    The persisted document is intentionally JSON-only so replay, docs, and
    humans can inspect it without importing tool code.
    """

    results_dir = _tool_results_dir()
    if results_dir is None:
        return None
    results_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_"
        f"{_safe_tool_slug(tool_name)}_{uuid.uuid4().hex[:12]}.json"
    )
    path = results_dir / filename
    document = {
        "schema_version": TOOL_RESULT_SCHEMA_VERSION,
        "tool_name": tool_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "payload": _json_safe(payload),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return f"{TOOL_RESULT_REF_PREFIX}{filename}"


def read_tool_result_ref(
    ref: str,
    *,
    offset: int = 0,
    max_chars: int = DEFAULT_CHUNK_CHARS,
) -> dict[str, Any]:
    """Read a stored tool-result artifact by bounded character window."""

    if not ref.startswith(TOOL_RESULT_REF_PREFIX):
        raise ValueError(f"unsupported tool result ref: {ref!r}")
    filename = ref.removeprefix(TOOL_RESULT_REF_PREFIX)
    if not filename or Path(filename).name != filename:
        raise ValueError("tool result refs may not contain path separators")
    results_dir = _tool_results_dir()
    if results_dir is None:
        raise ValueError("no active Praxist run directory is available")
    path = results_dir / filename
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(results_dir.resolve())
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"tool result ref not found: {ref}") from exc

    text = resolved.read_text(encoding="utf-8")
    safe_offset = max(0, int(offset))
    safe_max_chars = max(1, min(int(max_chars), MAX_CHUNK_CHARS))
    chunk = text[safe_offset : safe_offset + safe_max_chars]
    next_offset = safe_offset + len(chunk)
    return {
        "schema_version": TOOL_RESULT_CHUNK_SCHEMA_VERSION,
        "ref": ref,
        "offset": safe_offset,
        "max_chars": safe_max_chars,
        "returned_chars": len(chunk),
        "total_chars": len(text),
        "next_offset": next_offset if next_offset < len(text) else None,
        "text": chunk,
    }


def coerce_inline_limit(
    value: Any,
    *,
    default: int = DEFAULT_INLINE_LIST_LIMIT,
    maximum: int = MAX_INLINE_LIST_LIMIT,
) -> int:
    """Coerce a user-supplied inline list limit into a bounded integer."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0, min(parsed, maximum))


def with_tool_output_envelope(
    payload: dict[str, Any],
    *,
    tool_name: str,
    list_fields: tuple[str, ...] = (),
    inline_limit: int | None = None,
    full_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a bounded payload with metadata pointing to the full artifact."""

    complete_payload = _json_safe(full_payload if full_payload is not None else payload)
    output = _json_safe(payload)
    safe_inline_limit = (
        DEFAULT_INLINE_LIST_LIMIT
        if inline_limit is None
        else max(0, min(int(inline_limit), MAX_INLINE_LIST_LIMIT))
    )
    truncated_lists: dict[str, dict[str, int]] = {}
    full_ref = store_tool_result(tool_name, complete_payload)
    if full_ref is not None:
        for field in list_fields:
            value = output.get(field)
            if isinstance(value, list) and len(value) > safe_inline_limit:
                output[field] = value[:safe_inline_limit]
                truncated_lists[field] = {
                    "returned": safe_inline_limit,
                    "total": len(value),
                }
    output["_tool_output"] = {
        "schema_version": TOOL_OUTPUT_SCHEMA_VERSION,
        "tool_name": tool_name,
        "view": "summary" if full_ref is not None else "complete_inline",
        "full_result_ref": full_ref,
        "truncated": bool(truncated_lists),
        "truncated_lists": truncated_lists,
        "read_full_result_tool": "mcp__evaluation-tools__read_tool_result",
        "pagination": {
            "kind": "char_offset",
            "default_max_chars": DEFAULT_CHUNK_CHARS,
            "max_chars": MAX_CHUNK_CHARS,
        },
    }
    return output
