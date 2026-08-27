"""
PreToolUse hook — record start time for tool invocations.
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


def main():
    """Record the start of a legacy Claude Code tool invocation."""
    hook_input = read_hook_input()
    if not hook_input:
        return

    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})
    session_id = hook_input.get("session_id", "unknown")

    # Only track Skill and MCP tools
    if not (tool_name.startswith("Skill") or tool_name.startswith("mcp__")):
        return

    try:
        from praxist.plugins.workflow_stages.research_loop.backend.telemetry.tool_timing import (
            record_start_time,
        )

        record_start_time(
            session_id=session_id,
            tool_name=tool_name,
            tool_input=json.dumps(tool_input, sort_keys=True, default=str),
        )
    except Exception:
        pass  # Hooks must not crash the agent


if __name__ == "__main__":
    main()
