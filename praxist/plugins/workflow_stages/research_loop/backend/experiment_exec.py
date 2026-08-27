"""Tiny launch barrier used by the central experiment scheduler."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    """Wait for scheduler commit, then replace this barrier with the task command."""

    if len(sys.argv) < 5:
        return 75
    ready_path = Path(sys.argv[1])
    go_path = Path(sys.argv[2])
    attempt_id = sys.argv[3]
    command = sys.argv[4:]
    try:
        suffix = Path(f"/proc/{os.getpid()}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1]
        pid_start_time = int(suffix.split()[19])
    except (OSError, ValueError, IndexError):
        pid_start_time = None
    temporary = ready_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "pgid": os.getpgrp(),
                "attempt_id": attempt_id,
                "pid_start_time": pid_start_time,
            }
        ),
        encoding="utf-8",
    )
    os.replace(temporary, ready_path)
    deadline = time.monotonic() + 300.0
    while not go_path.exists():
        if time.monotonic() >= deadline:
            return 75
        time.sleep(0.05)
    try:
        os.execvpe(command[0], command, os.environ)
    except OSError:
        return 75
    return 75  # pragma: no cover - exec never returns on success.


if __name__ == "__main__":
    raise SystemExit(main())
