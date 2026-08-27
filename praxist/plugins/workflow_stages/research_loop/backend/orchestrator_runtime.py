"""Runtime guardrails for the research-loop orchestrator process."""

from __future__ import annotations

import contextlib
import os
import signal
import time
from pathlib import Path
from typing import Any

from praxist.plugins.workflow_stages.research_loop.backend.resume_state import (
    append_resume_event,
    lock_pid,
    pid_is_alive,
)


class OrchestratorRuntimeScope:
    """Own the lock file and signal handlers for one orchestrator run."""

    def __init__(self, *, run_dir: Path, resume: bool, logger: Any) -> None:
        self.run_dir = Path(run_dir)
        self.resume = bool(resume)
        self.logger = logger
        self.lock_path = self.run_dir / "orchestrator.lock"
        self.shutdown_path = self.run_dir / "ORCHESTRATOR_SHUTDOWN"
        self._prev_sigint: Any = None
        self._prev_sigterm: Any = None
        self._signals_installed = False

    def enter(self) -> OrchestratorRuntimeScope:
        """Create the run lock and install shutdown signal handlers."""

        self._prepare_lock()
        self._install_signal_handlers()
        return self

    def close(self) -> None:
        """Restore signal handlers and remove the run lock."""

        if self._signals_installed:
            try:
                signal.signal(signal.SIGINT, self._prev_sigint)
                signal.signal(signal.SIGTERM, self._prev_sigterm)
            except (ValueError, OSError):
                pass
        with contextlib.suppress(OSError):
            self.lock_path.unlink(missing_ok=True)

    def _prepare_lock(self) -> None:
        if self.lock_path.exists():
            try:
                prev = self.lock_path.read_text(encoding="utf-8").strip()
                prev_pid = lock_pid(prev)
                if self.resume and prev_pid is not None and pid_is_alive(prev_pid):
                    raise RuntimeError(
                        f"cannot resume {self.run_dir}: orchestrator.lock belongs "
                        f"to live pid {prev_pid}"
                    )
                if self.resume:
                    self.logger.warning(
                        "orchestrator: removing stale lock file at %s during resume (content: %r)",
                        self.lock_path,
                        prev[:120],
                    )
                    append_resume_event(
                        self.run_dir,
                        {
                            "event": "stale_lock_removed",
                            "lock_path": str(self.lock_path),
                            "lock_content": prev[:500],
                        },
                    )
                    with contextlib.suppress(OSError):
                        self.lock_path.unlink(missing_ok=True)
                else:
                    self.logger.warning(
                        "orchestrator: existing lock file at %s — content: %r. "
                        "Another orchestrator may be running on this run_dir. "
                        "STOP_SIGNAL files may race between processes. "
                        "If the previous run is dead, manually delete the lock.",
                        self.lock_path,
                        prev[:120],
                    )
            except OSError:
                pass
        with contextlib.suppress(OSError):
            self.lock_path.write_text(
                f"pid={os.getpid()}\nstarted={time.time():.0f}\nrun_dir={self.run_dir}\n",
                encoding="utf-8",
            )

    def _install_signal_handlers(self) -> None:
        def _signal_handler(signum: int, _frame: Any) -> None:
            self.logger.warning(
                "orchestrator: received signal %d — touching shutdown sentinel at %s",
                signum,
                self.shutdown_path,
            )
            with contextlib.suppress(OSError):
                self.shutdown_path.write_text(
                    f"signal={signum}\nat={time.time():.0f}\n",
                    encoding="utf-8",
                )
            previous = self._prev_sigint if signum == signal.SIGINT else self._prev_sigterm
            if callable(previous):
                previous(signum, _frame)
                return
            if previous == signal.SIG_IGN:
                return
            raise SystemExit(128 + int(signum))

        self._prev_sigint = signal.getsignal(signal.SIGINT)
        self._prev_sigterm = signal.getsignal(signal.SIGTERM)
        try:
            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)
            self._signals_installed = True
        except (ValueError, OSError):
            self.logger.debug("orchestrator: could not install signal handlers")
            try:
                signal.signal(signal.SIGINT, self._prev_sigint)
                signal.signal(signal.SIGTERM, self._prev_sigterm)
            except (ValueError, OSError):
                pass


def enter_orchestrator_runtime_scope(
    *, run_dir: Path, resume: bool, logger: Any
) -> OrchestratorRuntimeScope:
    """Return an entered runtime scope for lock and signal cleanup."""

    return OrchestratorRuntimeScope(run_dir=run_dir, resume=resume, logger=logger).enter()
