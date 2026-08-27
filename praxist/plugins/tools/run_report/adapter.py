"""Run-report tool server adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from praxist.plugins.tools.result_envelope import active_run_dir
from praxist.plugins.workflow_stages.research_loop.backend.run_report import (
    generate_run_report as _generate_run_report,
)

try:
    from claude_agent_sdk import create_sdk_mcp_server, tool  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - SDK optional in unit tests
    create_sdk_mcp_server = None
    tool = None


def generate_run_report(
    run_dir: str | None = None,
    task_dir: str | None = None,
    trigger: str = "manual_tool_call",
    generation_id: int | None = None,
) -> dict[str, Any]:
    """Generate a Markdown report and return its path."""

    resolved_run_dir = Path(run_dir).expanduser().resolve() if run_dir else active_run_dir()
    if resolved_run_dir is None:
        return {
            "ok": False,
            "error": "run_dir is required when PRAXIST_RUN_DIR or LOCAL_STORE_DIR is not set",
        }
    resolved_task_dir = Path(task_dir).expanduser().resolve() if task_dir else None
    result = _generate_run_report(
        run_dir=resolved_run_dir,
        task_dir=resolved_task_dir,
        trigger=trigger,
        generation_id=generation_id,
    )
    return {
        "ok": True,
        "report_path": str(result.path),
        "pdf_path": str(result.pdf_path) if result.pdf_path is not None else None,
        "trigger": result.trigger,
        "generation_id": result.generation_id,
        "report_kind": result.report_kind,
    }


def create_run_report_server() -> Any:  # pragma: no cover - requires SDK
    """Create the SDK MCP server for manual Praxist run-report generation."""

    if create_sdk_mcp_server is None or tool is None:
        raise RuntimeError("claude_agent_sdk is required for run_report tool server")

    @tool(
        "generate_run_report",
        "Generate a human-readable Markdown report for the active Praxist run.",
        {
            "run_dir": str,
            "task_dir": str,
            "trigger": str,
            "generation_id": int,
        },
    )
    async def _generate(args: dict[str, Any]) -> dict[str, Any]:
        out = generate_run_report(
            run_dir=args.get("run_dir") or None,
            task_dir=args.get("task_dir") or None,
            trigger=args.get("trigger") or "manual_tool_call",
            generation_id=args.get("generation_id"),
        )
        return {"content": [{"type": "text", "text": json.dumps(out, default=str)}]}

    return create_sdk_mcp_server("run-report", tools=[_generate])


def create_tool_plugin() -> dict[str, object]:
    """Return the plugin descriptor consumed by the Praxist tool-server registry."""

    return {
        "kind": "tool_server",
        "tool_server_ref": "tool_server:run_report",
        "server_name": "run-report",
        "factory": "praxist.plugins.tools.run_report.adapter:create_run_report_server",
        "tool_names": ["generate_run_report"],
        "visibility": ["panel"],
        "required_capability": "tool_server.run_report",
    }
