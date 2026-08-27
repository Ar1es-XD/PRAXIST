"""Append-only trajectory writer for Gate A."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from praxist.core.redaction import redact_json
from praxist.core.storage import utc_now

_TRAJECTORY_THREAD_LOCK = threading.Lock()


class TrajectoryWriter:
    """Append-only trajectory writer that assigns stable event ids and redacts persisted fields."""

    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.path = run_dir / "trajectory.jsonl"
        self._seq = _existing_trajectory_seq(self.path)

    def emit(
        self,
        kind: str,
        *,
        severity: str = "info",
        scope: dict[str, str] | None = None,
        actor: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
        artifact_refs: list[dict[str, Any]] | None = None,
        parent_event_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        redacted_payload, payload_hits = redact_json(payload or {})
        redacted_scope, scope_hits = redact_json(scope or {})
        redacted_actor, actor_hits = redact_json(actor or {"type": "core", "id": "unknown"})
        redacted_artifact_refs, artifact_hits = redact_json(artifact_refs or [])
        hits = [*payload_hits, *scope_hits, *actor_hits, *artifact_hits]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _TRAJECTORY_THREAD_LOCK, self.path.open("a+b") as handle:
            _lock_file(handle)
            try:
                self._seq = max(self._seq, _latest_trajectory_seq(handle)) + 1
                event = {
                    "schema_version": "praxist.trajectory.v1",
                    "run_id": self.run_id,
                    "seq": self._seq,
                    "event_id": f"evt_{self._seq:06d}",
                    "parent_event_ids": parent_event_ids or [],
                    "timestamp": utc_now(),
                    "kind": kind,
                    "severity": severity,
                    "scope": redacted_scope if isinstance(redacted_scope, dict) else {},
                    "actor": redacted_actor
                    if isinstance(redacted_actor, dict)
                    else {"type": "core", "id": "unknown"},
                    "payload": redacted_payload,
                    "artifact_refs": redacted_artifact_refs
                    if isinstance(redacted_artifact_refs, list)
                    else [],
                    "redaction": {
                        "applied": bool(hits),
                        "match_classes": sorted(set(hits)),
                    },
                }
                handle.seek(0, os.SEEK_END)
                serialized = json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n"
                handle.write(serialized.encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
                return event
            finally:
                _unlock_file(handle)


def _existing_trajectory_seq(path: Path) -> int:
    if not path.exists():
        return 0
    highest = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            try:
                highest = max(highest, int(record.get("seq", 0)))
            except (TypeError, ValueError):
                continue
    return highest


def _latest_trajectory_seq(handle: Any) -> int:
    """Read the last valid JSONL record without rescanning the full trajectory."""

    handle.seek(0, os.SEEK_END)
    end = handle.tell()
    if end <= 0:
        return 0
    chunk_size = 8192
    suffix = b""
    position = end
    while position > 0:
        size = min(chunk_size, position)
        position -= size
        handle.seek(position)
        suffix = handle.read(size) + suffix
        lines = suffix.splitlines()
        complete = lines if position == 0 else lines[1:]
        for line in reversed(complete):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                return int(record.get("seq", 0)) if isinstance(record, dict) else 0
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        if position > 0:
            suffix = suffix[: suffix.find(b"\n") + 1] if b"\n" in suffix else suffix
    return 0


def _lock_file(handle: Any) -> None:
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Praxist production hosts are POSIX.
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle: Any) -> None:
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Praxist production hosts are POSIX.
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
