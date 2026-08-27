"""
Thread-safe usage tracker for skills and MCP tools.

Tracks frequency, duration, success/failure. Persists to usage_stats.json.
"""

import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from praxist.core.run_config import DEFAULT_LOGS_DIR

_tracker_lock = threading.Lock()
_global_tracker: Optional["UsageTracker"] = None


class UsageTracker:
    """Track tool/skill usage with thread-safe JSON persistence."""

    def __init__(self, stats_dir: Path | None = None):
        self._lock = threading.Lock()
        self._stats_dir = Path(stats_dir or os.environ.get("LOGS_DIR", DEFAULT_LOGS_DIR))
        self._stats_dir.mkdir(parents=True, exist_ok=True)
        self._stats_file = self._stats_dir / "usage_stats.json"
        self._stats = self._load_or_create()

    def _load_or_create(self) -> dict:
        if self._stats_file.exists():
            try:
                with open(self._stats_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, KeyError):
                pass
        return {
            "session_id": str(uuid.uuid4()),
            "created_at": datetime.now().isoformat(),
            "tools": {},
            "skills": {},
        }

    def _save(self):
        tmp = self._stats_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._stats, f, indent=2, default=str)
        tmp.rename(self._stats_file)

    def record(
        self,
        category: str,
        name: str,
        duration_ms: float | None = None,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ):
        """Record a tool or skill invocation."""
        with self._lock:
            bucket = self._stats.setdefault(category, {})
            entry = bucket.setdefault(
                name,
                {
                    "total_calls": 0,
                    "successes": 0,
                    "failures": 0,
                    "total_duration_ms": 0,
                    "calls": [],
                },
            )

            entry["total_calls"] += 1
            if success:
                entry["successes"] += 1
            else:
                entry["failures"] += 1
            if duration_ms is not None:
                entry["total_duration_ms"] += duration_ms

            call_record: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "duration_ms": duration_ms,
                "success": success,
            }
            if metadata:
                call_record["metadata"] = metadata

            entry["calls"].append(call_record)
            # Keep last 100 calls
            if len(entry["calls"]) > 100:
                entry["calls"] = entry["calls"][-100:]

            self._save()

    def record_skill(self, name: str, **kwargs):
        self.record("skills", name, **kwargs)

    def record_mcp_tool(self, name: str, **kwargs):
        self.record("tools", name, **kwargs)

    def get_stats(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._stats, default=str))


def get_tracker() -> UsageTracker:
    """Get or create global UsageTracker instance."""
    global _global_tracker
    with _tracker_lock:
        if _global_tracker is None:
            _global_tracker = UsageTracker()
        return _global_tracker


def reset_tracker():
    """Reset global tracker."""
    global _global_tracker
    with _tracker_lock:
        _global_tracker = None
