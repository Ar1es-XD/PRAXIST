"""
File-based timing tracker for tool invocations.

PreToolUse hooks call record_start_time(); PostToolUse hooks call
get_duration_ms() to calculate how long a tool took.
"""

import hashlib
import json
import os
import time
from pathlib import Path

from praxist.core.run_config import DEFAULT_LOGS_DIR


def _get_timing_file() -> Path:
    logs = Path(os.environ.get("LOGS_DIR", DEFAULT_LOGS_DIR))
    logs.mkdir(parents=True, exist_ok=True)
    return logs / ".tool_timing.json"


def _get_tool_key(session_id: str, tool_name: str, tool_input: str) -> str:
    input_hash = hashlib.md5(tool_input.encode()).hexdigest()[:8]
    return f"{session_id}:{tool_name}:{input_hash}"


def _load_timing_state(path: Path) -> dict:
    if path.exists():
        try:
            with open(path) as f:
                state = json.load(f)
            # Auto-cleanup entries older than 1 hour
            now = time.time()
            state = {k: v for k, v in state.items() if now - v.get("start_time", 0) < 3600}
            return state
        except (json.JSONDecodeError, KeyError):
            return {}
    return {}


def _save_timing_state(path: Path, state: dict):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f)
    tmp.rename(path)


def record_start_time(session_id: str, tool_name: str, tool_input: str = ""):
    """Record the start time for a tool invocation."""
    path = _get_timing_file()
    state = _load_timing_state(path)
    key = _get_tool_key(session_id, tool_name, tool_input)
    state[key] = {"start_time": time.time(), "tool_name": tool_name}
    _save_timing_state(path, state)


def get_duration_ms(session_id: str, tool_name: str, tool_input: str = "") -> float | None:
    """Get duration in milliseconds since start was recorded.

    Returns None if no matching start was found. Cleans up the entry.
    """
    path = _get_timing_file()
    state = _load_timing_state(path)
    key = _get_tool_key(session_id, tool_name, tool_input)

    entry = state.pop(key, None)
    _save_timing_state(path, state)

    if entry is None:
        return None

    return (time.time() - entry["start_time"]) * 1000
