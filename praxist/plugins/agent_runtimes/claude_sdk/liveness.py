"""Thread-safe liveness state for one isolated Claude SDK session."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

TERMINAL_BACKGROUND_TASK_STATUSES = frozenset({"completed", "failed", "killed", "stopped"})


def _safe_tool_name(value: str) -> str:
    return " ".join(str(value or "unknown").split())[:80] or "unknown"


@dataclass
class ClaudeSessionLiveness:
    """Own progress clocks and work state shared by parent and worker loops."""

    sdk_complete_message_progress_at: float = field(default_factory=time.monotonic)
    model_stream_progress_at: float = field(default_factory=time.monotonic)
    tool_progress_at: float = field(default_factory=time.monotonic)
    _foreground_by_id: dict[str, tuple[str, float]] = field(default_factory=dict, repr=False)
    _foreground_without_id: list[tuple[str, float]] = field(default_factory=list, repr=False)
    _background_by_id: dict[str, tuple[str, float]] = field(default_factory=dict, repr=False)
    _closing: bool = field(default=False, repr=False)
    _terminal: bool = field(default=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_complete_message(self) -> None:
        with self._lock:
            self.sdk_complete_message_progress_at = time.monotonic()

    def record_model_stream_progress(self) -> None:
        with self._lock:
            self.model_stream_progress_at = time.monotonic()

    def start_foreground_tool(self, tool_name: str, tool_id: str = "") -> None:
        now = time.monotonic()
        tool = (_safe_tool_name(tool_name), now)
        with self._lock:
            if tool_id:
                self._foreground_by_id.setdefault(tool_id, tool)
            else:
                self._foreground_without_id.append(tool)
            self.tool_progress_at = now

    def finish_foreground_tool(self, tool_id: str = "") -> None:
        now = time.monotonic()
        with self._lock:
            removed = self._foreground_by_id.pop(tool_id, None) if tool_id else None
            if removed is None and self._foreground_without_id:
                self._foreground_without_id.pop(0)
            elif removed is None and len(self._foreground_by_id) == 1:
                self._foreground_by_id.pop(next(iter(self._foreground_by_id)))
            self.tool_progress_at = now

    def record_active_work_heartbeat(self) -> None:
        """Record that the SDK worker is still polling active foreground/background work."""

        with self._lock:
            background_active = any(
                status not in TERMINAL_BACKGROUND_TASK_STATUSES
                for status, _started_at in self._background_by_id.values()
            )
            if (
                not self._foreground_by_id
                and not self._foreground_without_id
                and not background_active
            ):
                return
            self.tool_progress_at = time.monotonic()

    def record_background_status(self, task_id: str, status: str) -> None:
        now = time.monotonic()
        normalized = str(status or "running").strip().lower() or "running"
        with self._lock:
            previous = self._background_by_id.get(task_id)
            started_at = previous[1] if previous is not None else now
            self._background_by_id[task_id] = (normalized, started_at)
            self.tool_progress_at = now

    def foreground_work_active(self) -> bool:
        with self._lock:
            return bool(self._foreground_by_id or self._foreground_without_id)

    def all_background_terminal(self) -> bool:
        with self._lock:
            return bool(self._background_by_id) and all(
                status in TERMINAL_BACKGROUND_TASK_STATUSES
                for status, _started_at in self._background_by_id.values()
            )

    def any_background_failed(self) -> bool:
        with self._lock:
            return any(
                status in {"failed", "killed", "stopped"}
                for status, _started_at in self._background_by_id.values()
            )

    def mark_closing(self) -> None:
        with self._lock:
            self._closing = True

    def mark_terminal(self) -> None:
        with self._lock:
            self._terminal = True

    def observation(self) -> dict[str, Any]:
        """Return one coherent snapshot without exposing tool inputs or task IDs."""

        with self._lock:
            foreground = [*self._foreground_by_id.values(), *self._foreground_without_id]
            active_background = [
                started_at
                for status, started_at in self._background_by_id.values()
                if status not in TERMINAL_BACKGROUND_TASK_STATUSES
            ]
            if self._terminal:
                state = "idle"
                state_started_at = self.tool_progress_at
            elif self._closing:
                state = "closing"
                state_started_at = self.tool_progress_at
            elif foreground:
                state = "foreground_tool_running"
                state_started_at = min(started_at for _name, started_at in foreground)
            elif active_background:
                state = "background_work_running"
                state_started_at = min(active_background)
            else:
                state = "model_waiting"
                state_started_at = max(
                    self.sdk_complete_message_progress_at,
                    self.model_stream_progress_at,
                    self.tool_progress_at,
                )
            return {
                "session_state": state,
                "sdk_complete_message_progress_at": self.sdk_complete_message_progress_at,
                "model_stream_progress_at": self.model_stream_progress_at,
                "tool_progress_at": self.tool_progress_at,
                "latest_progress_at": max(
                    self.sdk_complete_message_progress_at,
                    self.model_stream_progress_at,
                    self.tool_progress_at,
                ),
                "state_started_at": state_started_at,
                "active_foreground_tools": tuple(sorted({name for name, _started in foreground})),
                "active_background_tasks": len(active_background),
            }
