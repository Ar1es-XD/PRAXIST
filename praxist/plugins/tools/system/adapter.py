"""System tool server adapter — read-only surfacing of Praxist run state.

Provides 5 MCP tools any role can use to see what is happening across the
workspace. A Peer might inspect sibling activity, a PI might request a
frontier glance, and a Chair might scan recent errors.

Tools are read-only and side-effect free:

* ``system_active_runs(workspace=None)`` — list runs whose status file
  has been touched recently.
* ``system_run_summary(run_dir)`` — assemble a compact summary of one
  run from ``run_summary.json`` and budget ledger.
* ``system_recent_findings(run_dir, n=10)`` — last N finding records.
* ``system_frontier_snapshot(run_dir)`` — current top-K frontier.
* ``system_recent_errors(run_dir, n=10)`` — last N error/warn events
  from the trajectory.

Each handler returns a JSON-serializable dict.  Errors are surfaced as
``{"error": "..."}`` rather than raising so the LLM agent can recover.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
    is_committed_runtime_fact_source,
    is_readable_signal_source,
)

try:  # claude_agent_sdk is runtime-required for the MCP server factory
    from claude_agent_sdk import (  # type: ignore[import-not-found]
        create_sdk_mcp_server,
        tool,
    )
except ImportError:  # pragma: no cover - SDK missing in some test envs
    tool = None
    create_sdk_mcp_server = None


_ACTIVE_RUN_WINDOW_SECONDS = 24 * 60 * 60
"""A run counts as 'active' if its status file was touched within this window."""

_DEFAULT_RECENT_LIMIT = 10
_MAX_RECENT_LIMIT = 100


__all__ = [
    "create_tool_plugin",
    "create_system_tools_server",
    "system_active_runs",
    "system_run_summary",
    "system_recent_findings",
    "system_frontier_snapshot",
    "system_recent_errors",
]


# --------------------------------------------------------------------------- #
# Handlers (pure functions, callable from tests without claude_agent_sdk)
# --------------------------------------------------------------------------- #


def system_active_runs(workspace: str | Path | None = None) -> dict[str, Any]:
    """Return runs whose status file changed within the active-run window.

    Args:
        workspace: Directory containing per-run subdirectories.  Defaults
            to ``<cwd>/runs``.  A workspace is recognized by containing
            child directories that look like run dirs (have a
            ``run_summary.json``, ``trajectory.jsonl``, or
            ``orchestrator_status*.json``).

    Returns:
        ``{"runs": [...], "workspace": "<resolved path>"}`` where each
        entry has ``run_dir``, ``status_path``, ``last_active_ms``,
        ``age_seconds``.
    """
    ws = _resolve_workspace(workspace)
    if not ws.is_dir():
        return {"workspace": str(ws), "runs": [], "error": f"workspace not found: {ws}"}

    now = time.time()
    entries: list[dict[str, Any]] = []
    for child in sorted(ws.iterdir()):
        if not child.is_dir():
            continue
        status_path = _newest_status_path(child)
        if status_path is None:
            continue
        try:
            mtime = status_path.stat().st_mtime
        except OSError:  # pragma: no cover - race: file vanished between glob and stat
            continue
        age = now - mtime
        if age > _ACTIVE_RUN_WINDOW_SECONDS:
            continue
        entries.append(
            {
                "run_dir": str(child),
                "status_path": str(status_path),
                "last_active_ms": int(mtime * 1000),
                "age_seconds": int(age),
            }
        )

    entries.sort(key=lambda item: -int(item["last_active_ms"]))
    return {"workspace": str(ws), "runs": entries}


def system_run_summary(run_dir: str | Path) -> dict[str, Any]:
    """Assemble a compact summary of one run.

    Reads ``<run_dir>/run_summary.json`` and supplements with budget
    totals from ``budget_ledger.jsonl`` when present.

    Args:
        run_dir: Run directory path.

    Returns:
        Dict with ``run_dir``, ``summary`` (the JSON), ``budget_totals``,
        and ``warnings`` for any partial-state issues.
    """
    rd = Path(run_dir)
    out: dict[str, Any] = {"run_dir": str(rd), "warnings": []}

    summary_path = rd / "run_summary.json"
    if summary_path.is_file():
        try:
            out["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            out["summary"] = None
            out["warnings"].append(f"run_summary.json: {exc}")
    else:
        out["summary"] = None
        out["warnings"].append("run_summary.json not found")

    out["budget_totals"] = _budget_totals(rd / "budget_ledger.jsonl", out["warnings"])
    return out


def system_recent_findings(run_dir: str | Path, n: int = _DEFAULT_RECENT_LIMIT) -> dict[str, Any]:
    """Return the last N finding records, newest first.

    Args:
        run_dir: Run directory path.
        n: Maximum number of records to return (capped at 100).

    Returns:
        Dict with ``run_dir``, ``findings``, ``source`` (which file the
        records came from), and ``warnings``.
    """
    rd = Path(run_dir)
    limit = max(1, min(int(n), _MAX_RECENT_LIMIT))
    out: dict[str, Any] = {"run_dir": str(rd), "findings": [], "source": None, "warnings": []}

    findings_dir = rd / "findings"
    if findings_dir.is_dir():
        records = _tail_jsonl_dir(findings_dir, limit, out["warnings"])
        if records is not None:
            out["findings"] = records
            out["source"] = str(findings_dir)
            return out

    shared = rd / "shared_findings"
    if shared.is_dir():
        out["findings"] = _tail_shared_findings(shared, limit, out["warnings"])
        out["source"] = str(shared)
        return out

    out["warnings"].append(
        "no findings directory found (looked for findings/ and shared_findings/)"
    )
    return out


def system_frontier_snapshot(run_dir: str | Path) -> dict[str, Any]:
    """Return the current frontier manifest for one run.

    Args:
        run_dir: Run directory path.

    Returns:
        Dict with ``run_dir``, ``frontier`` (manifest contents or
        ``None``), and ``warnings``.
    """
    rd = Path(run_dir)
    out: dict[str, Any] = {
        "run_dir": str(rd),
        "frontier": None,
        "frontier_signal": None,
        "frontier_signal_status": "",
        "warnings": [],
    }

    manifest = rd / "frontier" / "frontier_manifest.json"
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and not is_committed_runtime_fact_source(
                payload,
                legacy_ok=True,
            ):
                if is_readable_signal_source(payload, legacy_ok=True):
                    semantics = payload.get("artifact_semantics")
                    out["frontier_signal_status"] = (
                        str(semantics.get("status") or "").strip().lower()
                        if isinstance(semantics, dict)
                        else "legacy"
                    )
                    out["frontier_signal"] = payload
                    out["warnings"].append(
                        "frontier_manifest.json is not committed runtime state; "
                        "returned only as frontier_signal"
                    )
                else:
                    out["warnings"].append("frontier_manifest.json is superseded; ignored")
            else:
                out["frontier"] = payload
        except (json.JSONDecodeError, OSError) as exc:
            out["warnings"].append(f"frontier_manifest.json: {exc}")
    else:
        out["warnings"].append("frontier/frontier_manifest.json not found")
    return out


def system_recent_errors(run_dir: str | Path, n: int = _DEFAULT_RECENT_LIMIT) -> dict[str, Any]:
    """Return the last N error/warning events from the trajectory.

    Scans ``<run_dir>/trajectory.jsonl`` from the tail and keeps records
    whose ``type`` looks like an error or warning, or whose payload
    contains an ``error`` field.

    Args:
        run_dir: Run directory path.
        n: Maximum number of records to return (capped at 100).

    Returns:
        Dict with ``run_dir``, ``errors``, and ``warnings``.
    """
    rd = Path(run_dir)
    limit = max(1, min(int(n), _MAX_RECENT_LIMIT))
    out: dict[str, Any] = {"run_dir": str(rd), "errors": [], "warnings": []}

    trajectory = rd / "trajectory.jsonl"
    if not trajectory.is_file():
        out["warnings"].append("trajectory.jsonl not found")
        return out

    try:
        records = _read_jsonl_lines(trajectory)
    except OSError as exc:  # pragma: no cover - race: file vanished between is_file and open
        out["warnings"].append(f"trajectory.jsonl: {exc}")
        return out

    matched: list[dict[str, Any]] = []
    for record in reversed(records):
        if not isinstance(record, dict):
            continue
        if _looks_like_error_event(record):
            matched.append(record)
            if len(matched) >= limit:
                break

    out["errors"] = matched
    return out


# --------------------------------------------------------------------------- #
# MCP server wiring
# --------------------------------------------------------------------------- #


def _text_result(data: Any) -> dict[str, Any]:
    """Wrap a JSON-serializable value as an MCP text content response."""
    text = json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data
    return {"content": [{"type": "text", "text": text}]}


def create_system_tools_server() -> Any:  # pragma: no cover - requires claude_agent_sdk
    """Create the MCP server exposing the five system tools.

    Uses the ``claude_agent_sdk.tool(name, description, input_schema)``
    positional form expected by SDK 0.1.x: the third positional argument
    is the ``input_schema`` (a Python type dict like
    ``{"run_dir": str, "n": int}``), and the resulting wrapper is applied
    to an ``async`` handler that receives a single ``args: dict`` payload.

    The handler functions :func:`system_active_runs` and friends are
    unit-tested directly; this factory only wires them through the SDK
    and is exercised by integration spawn tests.

    Raises:
        ImportError: If ``claude_agent_sdk`` is unavailable.
    """
    if create_sdk_mcp_server is None or tool is None:
        raise ImportError("claude_agent_sdk is required for MCP tools")

    async def _handle_active_runs(args: dict[str, Any]) -> dict[str, Any]:
        return _text_result(system_active_runs(args.get("workspace")))

    async def _handle_run_summary(args: dict[str, Any]) -> dict[str, Any]:
        return _text_result(system_run_summary(args["run_dir"]))

    async def _handle_recent_findings(args: dict[str, Any]) -> dict[str, Any]:
        return _text_result(
            system_recent_findings(args["run_dir"], int(args.get("n", _DEFAULT_RECENT_LIMIT)))
        )

    async def _handle_frontier(args: dict[str, Any]) -> dict[str, Any]:
        return _text_result(system_frontier_snapshot(args["run_dir"]))

    async def _handle_recent_errors(args: dict[str, Any]) -> dict[str, Any]:
        return _text_result(
            system_recent_errors(args["run_dir"], int(args.get("n", _DEFAULT_RECENT_LIMIT)))
        )

    active_runs = tool(
        "system_active_runs",
        "List Praxist runs whose status was updated in the last 24 hours.",
        {"workspace": str},
    )(_handle_active_runs)
    run_summary = tool(
        "system_run_summary",
        "Return a compact summary of one run (goals, generations, budget).",
        {"run_dir": str},
    )(_handle_run_summary)
    recent_findings = tool(
        "system_recent_findings",
        "Return the N most recent finding records for a run, newest first.",
        {"run_dir": str, "n": int},
    )(_handle_recent_findings)
    frontier_snapshot = tool(
        "system_frontier_snapshot",
        "Return the current top-K frontier manifest for a run.",
        {"run_dir": str},
    )(_handle_frontier)
    recent_errors = tool(
        "system_recent_errors",
        "Return the N most recent error or warning events from the run trajectory.",
        {"run_dir": str, "n": int},
    )(_handle_recent_errors)

    return create_sdk_mcp_server(
        "system-tools",
        tools=[active_runs, run_summary, recent_findings, frontier_snapshot, recent_errors],
    )


def create_tool_plugin() -> dict[str, object]:
    """Manifest entrypoint exposing the system tool server descriptor."""
    return {
        "tool_server_ref": "tool_server:system",
        "server_name": "system-tools",
        "factory": "praxist.plugins.tools.system.adapter:create_system_tools_server",
        "tool_names": [
            "system_active_runs",
            "system_run_summary",
            "system_recent_findings",
            "system_frontier_snapshot",
            "system_recent_errors",
        ],
        "visibility": ["peer", "panel"],
        "required_capability": "tool_server.system",
        "handlers": {
            "system_active_runs": "praxist.plugins.tools.system.adapter:system_active_runs",
            "system_run_summary": "praxist.plugins.tools.system.adapter:system_run_summary",
            "system_recent_findings": "praxist.plugins.tools.system.adapter:system_recent_findings",
            "system_frontier_snapshot": "praxist.plugins.tools.system.adapter:system_frontier_snapshot",
            "system_recent_errors": "praxist.plugins.tools.system.adapter:system_recent_errors",
        },
    }


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _resolve_workspace(workspace: str | Path | None) -> Path:
    """Resolve the workspace directory for run discovery."""
    if workspace is None:
        env = os.environ.get("PRAXIST_WORKSPACE")
        if env:
            return Path(env).expanduser()
        return Path.cwd() / "runs"
    return Path(workspace).expanduser()


def _newest_status_path(run_dir: Path) -> Path | None:
    """Return the most recently modified status-like file in ``run_dir``."""
    candidates: list[Path] = []
    for pattern in ("orchestrator_status*.json", "run_summary.json", "trajectory.jsonl"):
        candidates.extend(run_dir.glob(pattern))
    if not candidates:
        return None
    try:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    except OSError:  # pragma: no cover - race: file vanished between glob and stat
        return None


def _budget_totals(ledger: Path, warnings: list[str]) -> dict[str, Any]:
    """Sum approved / used / denied amounts from a budget ledger."""
    if not ledger.is_file():
        return {}
    try:
        records = _read_jsonl_lines(ledger)
    except OSError as exc:  # pragma: no cover - race: file vanished between is_file and open
        warnings.append(f"budget_ledger.jsonl: {exc}")
        return {}

    totals: dict[str, dict[str, float]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        unit = str(record.get("unit") or "unknown")
        bucket = totals.setdefault(unit, {"granted": 0.0, "used": 0.0, "denied": 0.0})
        for field in ("granted", "used", "denied"):
            value = record.get(field)
            if isinstance(value, (int, float)):
                bucket[field] += float(value)
    return totals


def _tail_jsonl_dir(
    findings_dir: Path, limit: int, warnings: list[str]
) -> list[dict[str, Any]] | None:
    """Tail JSONL files inside a findings directory, newest first."""
    files = sorted(findings_dir.glob("*.jsonl"), key=lambda p: -p.stat().st_mtime)
    if not files:
        return None
    records: list[dict[str, Any]] = []
    for path in files:
        try:
            lines = _read_jsonl_lines(path)
        except OSError as exc:  # pragma: no cover - race: file vanished mid-scan
            warnings.append(f"{path.name}: {exc}")
            continue
        for record in reversed(lines):
            if isinstance(record, dict):
                records.append(record)
                if len(records) >= limit:
                    return records
    return records


def _tail_shared_findings(shared: Path, limit: int, warnings: list[str]) -> list[dict[str, Any]]:
    """Tail JSON files inside a shared_findings directory, newest first."""
    files = sorted(shared.glob("*.json"), key=lambda p: -p.stat().st_mtime)
    records: list[dict[str, Any]] = []
    for path in files[:limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            warnings.append(f"{path.name}: {exc}")
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _read_jsonl_lines(path: Path) -> list[Any]:
    """Read a JSONL file, skipping blank lines and malformed records."""
    records: list[Any] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
    return records


def _looks_like_error_event(record: dict[str, Any]) -> bool:
    """Heuristic: trajectory record looks like an error or warning."""
    type_field = str(record.get("type") or "").lower()
    if any(needle in type_field for needle in ("error", "warn", "fail", "exception")):
        return True
    payload = record.get("payload")
    if isinstance(payload, dict):
        if any(key in payload for key in ("error", "exception", "traceback")):
            return True
        level = str(payload.get("level") or "").lower()
        if level in {"error", "warning", "warn"}:
            return True
    return False
