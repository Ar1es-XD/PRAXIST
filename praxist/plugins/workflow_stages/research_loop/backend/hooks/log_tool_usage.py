"""
PostToolUse hook — log tool usage to tracker.

Records duration, success/failure, and metadata for skills and MCP tools.
"""

import json
import sys


def read_hook_input():
    """Read and parse JSON from stdin (Claude Code hook protocol)."""
    try:
        data = sys.stdin.read()
        if not data:
            return None
        return json.loads(data)
    except (OSError, json.JSONDecodeError):
        return None


def parse_mcp_tool_name(tool_name: str):
    """Parse mcp__server__tool format. Returns (server, tool) or (None, None)."""
    if not tool_name.startswith("mcp__"):
        return None, None
    parts = tool_name[5:].split("__", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None


def main():
    """Record completed legacy Claude Code tool usage for run diagnostics."""
    hook_input = read_hook_input()
    if not hook_input:
        return

    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})
    tool_response = hook_input.get("tool_response", "")
    session_id = hook_input.get("session_id", "unknown")

    # Only track Skill and MCP tools
    if not (tool_name.startswith("Skill") or tool_name.startswith("mcp__")):
        return

    try:
        from praxist.plugins.workflow_stages.research_loop.backend.telemetry.tool_timing import (
            get_duration_ms,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.telemetry.usage_tracker import (
            get_tracker,
        )

        duration = get_duration_ms(
            session_id=session_id,
            tool_name=tool_name,
            tool_input=json.dumps(tool_input, sort_keys=True, default=str),
        )

        # Detect success/failure
        success = True
        if isinstance(tool_response, str):
            try:
                resp_data = json.loads(tool_response)
                if "error" in resp_data:
                    success = False
            except json.JSONDecodeError:
                pass

        tracker = get_tracker()

        if tool_name.startswith("mcp__"):
            server, tool_fn = parse_mcp_tool_name(tool_name)
            tracker.record_mcp_tool(
                name=tool_name,
                duration_ms=duration,
                success=success,
                metadata={"server": server, "tool": tool_fn},
            )
        else:
            tracker.record_skill(
                name=tool_name,
                duration_ms=duration,
                success=success,
            )

    except Exception:
        pass  # Hooks must not crash the agent


if __name__ == "__main__":
    main()
