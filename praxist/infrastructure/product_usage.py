"""Failure-isolated projection into Praxist's built-in product-usage client."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID

from praxist.plugins.workflow_stages.research_loop.lifecycle import PeerLifecycleSummary
from praxist.product_usage.client import UploadCoordinator, UsageSdk
from praxist.product_usage.consent import ConsentStatus, ConsentStore
from praxist.product_usage.lifecycle import PeerStatusSummary, RunTelemetryContext
from praxist.product_usage.notice import consent_notice_v2
from praxist.product_usage.outbox import Outbox
from praxist.product_usage.paths import run_state_path
from praxist.product_usage.protocol import (
    MAX_ERROR_COUNT,
    ErrorScope,
    ErrorStage,
    ErrorSummary,
    ErrorType,
    PraxistErrorCode,
    ReasonCode,
)
from praxist.product_usage.transport import default_batch_sender

_MAX_RUN_STATE_FILES = 512
_MAX_RUN_STATE_BYTES = 4 * 1024 * 1024
_MAX_RUN_STATE_FILE_BYTES = 512 * 1024
_RUN_STATE_RETENTION_SECONDS = 180 * 24 * 60 * 60
_RUN_STATE_TEMP_RETENTION_SECONDS = 60 * 60
_RUN_STATE_LOCK_SLOTS = 4096
_RUN_STATE_LOCK_FILE = ".observer-locks"
_MAX_TRACKED_GENERATIONS = 16_384
_UPLOAD_CLOSE_DRAIN_SECONDS = 2.0
_RUN_STATE_LOCK_REGISTRY_GUARD = threading.Lock()
_RUN_STATE_LOCK_REGISTRY: set[tuple[Path, int]] = set()


def product_usage_notice() -> str:
    """Return the canonical consent notice bundled with Praxist."""

    return consent_notice_v2()


class ProductUsageObserver:
    """Project privacy-bounded lifecycle summaries into the local outbox."""

    def __init__(
        self,
        *,
        run_dir: Path,
        praxist_version: str,
        sdk: Any,
        upload_once: Callable[[], int] | None = None,
        schedule_upload: Callable[[Callable[[], int]], object] | None = None,
        _state_path: Path | None = None,
    ) -> None:
        self._state_path = _state_path or run_state_path(run_dir)
        self._sdk = sdk
        self._upload_once = upload_once
        self._external_scheduler = schedule_upload
        self._upload_lock = threading.Lock()
        self._upload_thread: threading.Thread | None = None
        self._upload_pending = False
        self._commands: queue.Queue[tuple[str, object, str]] = queue.Queue(maxsize=64)
        self._queue_lock = threading.Lock()
        self._closed = threading.Event()
        self._worker_stopped = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._run_started_emitted = False
        self._run_finished_emitted = False
        self._resumed_unfinished = False
        self._finished_generations: set[int] = set()
        environment_id = getattr(self._sdk, "environment_id", None)
        if environment_id is None:
            raise ValueError("granted product usage requires an Environment ID")
        self._environment_id = UUID(str(environment_id))
        grant_id = getattr(self._sdk, "consent_grant_id", "")
        if grant_id is None:
            raise ValueError("granted product usage requires a consent grant identity")
        self._grant_id = str(grant_id)
        self._state_lock = _try_acquire_run_state_lock(self._state_path)
        if self._state_lock is None:
            raise RuntimeError("product-usage observer already owns this run state")
        try:
            self._context = self._restore_or_create_context(praxist_version)
            self._run_started_queued = self._run_started_emitted
            self._run_finished_queued = self._run_finished_emitted
            self._queued_generations = set(self._finished_generations)
            self._worker = threading.Thread(
                target=self._run_worker,
                name="praxist-product-usage",
                daemon=True,
            )
            self._worker.start()
        except BaseException:
            _release_run_state_lock(self._state_lock)
            self._state_lock = None
            raise

    @classmethod
    def create(
        cls,
        *,
        run_dir: Path,
        praxist_version: str,
        consent_store: Any | None = None,
        identity_store: Any | None = None,
        outbox: Any | None = None,
        sender: Any | None = None,
        schedule_upload: Callable[[Callable[[], int]], object] | None = None,
        _state_path: Path | None = None,
    ) -> ProductUsageObserver | None:
        """Create an observer only after explicit consent; otherwise return ``None``."""

        try:
            consent = consent_store or ConsentStore()
            status = consent.status()
            if getattr(status, "value", status) != ConsentStatus.GRANTED.value:
                return None
            capture_outbox = outbox or Outbox()
            sdk = UsageSdk(
                consent,
                identity_store=identity_store,
                _outbox_factory=lambda: capture_outbox,
            )
            if sdk.environment_id is None:
                return None
            batch_sender = sender if sender is not None else default_batch_sender(praxist_version)

            def flush_once() -> int:
                upload_outbox = (
                    Outbox(outbox.path) if isinstance(outbox, Outbox) else outbox or Outbox()
                )
                try:
                    return UploadCoordinator(consent, upload_outbox, batch_sender).flush_once()
                finally:
                    if outbox is None or isinstance(outbox, Outbox):
                        upload_outbox.close()

            return cls(
                run_dir=run_dir,
                praxist_version=praxist_version,
                sdk=sdk,
                upload_once=flush_once,
                schedule_upload=schedule_upload,
                _state_path=_state_path,
            )
        except Exception:
            return None

    def record_run_started(self, summary: PeerLifecycleSummary) -> bool:
        """Queue a non-invasive planned cohort snapshot at most once."""

        with self._queue_lock:
            if self._run_started_queued:
                return False
            if not self._enqueue_locked("run_started", summary):
                return False
            self._run_started_queued = True
            return True

    def record_generation_finished(self, summary: PeerLifecycleSummary) -> bool:
        """Capture one aggregate snapshot after a durable generation boundary."""

        with self._queue_lock:
            if summary.generation_ordinal in self._queued_generations:
                return False
            if len(self._queued_generations) >= _MAX_TRACKED_GENERATIONS:
                return False
            if not self._enqueue_locked("generation_finished", summary):
                return False
            self._queued_generations.add(summary.generation_ordinal)
            self._run_started_queued = True
            return True

    def record_run_finished(
        self,
        *,
        active_duration_seconds: float | None,
        failed: bool = False,
    ) -> bool:
        """Capture a terminal duration once without result or end-reason fields."""

        with self._queue_lock:
            if self._run_finished_queued:
                return False
            if not self._enqueue_locked("run_finished", (active_duration_seconds, failed)):
                return False
            self._run_finished_queued = True
            return True

    def close(self) -> None:
        """Persist queued observations and allow one bounded final upload drain."""

        deadline = time.monotonic() + _UPLOAD_CLOSE_DRAIN_SECONDS
        with self._queue_lock:
            self._closed.set()
        if not self._worker_stopped.wait(timeout=_remaining_seconds(deadline)):
            return
        with self._upload_lock:
            upload_thread = self._upload_thread
        if upload_thread is not None and upload_thread.is_alive():
            upload_thread.join(timeout=_remaining_seconds(deadline))

    def _wait_for_idle_for_tests(self, timeout: float = 2.0) -> bool:
        """Wait for queued observations in deterministic tests only."""

        deadline = time.monotonic() + max(0.0, timeout)
        event = self._worker_stopped if self._closed.is_set() else self._idle
        if not event.wait(timeout=max(0.0, deadline - time.monotonic())):
            return False
        while True:
            with self._upload_lock:
                upload_thread = self._upload_thread
            if upload_thread is None or not upload_thread.is_alive():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            upload_thread.join(timeout=remaining)

    def _enqueue_locked(self, kind: str, payload: object) -> bool:
        if self._closed.is_set():
            return False
        self._idle.clear()
        try:
            self._commands.put_nowait((kind, payload, self._grant_id))
        except queue.Full:
            return False
        return True

    def _run_worker(self) -> None:
        try:
            while True:
                try:
                    kind, payload, grant_id = self._commands.get(timeout=0.1)
                except queue.Empty:
                    with self._queue_lock:
                        if self._closed.is_set() and self._commands.empty():
                            return
                    continue
                try:
                    self._dispatch(kind, payload, grant_id)
                except Exception:
                    pass
                finally:
                    self._commands.task_done()
                with self._queue_lock:
                    if self._commands.empty():
                        self._idle.set()
                        if self._closed.is_set():
                            return
        finally:
            try:
                with contextlib.suppress(Exception):
                    self._sdk.close()
            finally:
                if self._state_lock is not None:
                    _release_run_state_lock(self._state_lock)
                    self._state_lock = None
                self._worker_stopped.set()

    def _dispatch(self, kind: str, payload: object, grant_id: str) -> None:
        if kind == "run_started" and isinstance(payload, PeerLifecycleSummary):
            self._record_run_started(payload, grant_id=grant_id)
        elif kind == "generation_finished" and isinstance(payload, PeerLifecycleSummary):
            self._record_generation_finished(payload, grant_id=grant_id)
        elif kind == "run_finished" and isinstance(payload, tuple) and len(payload) == 2:
            duration, failed = payload
            self._record_run_finished(
                duration if isinstance(duration, int | float) else None,
                failed=bool(failed),
                grant_id=grant_id,
            )

    def _record_run_started(self, summary: PeerLifecycleSummary, *, grant_id: str) -> None:
        if self._run_started_emitted:
            return
        event = self._context.run_started(
            self._peer_summary(summary),
            error_summaries=self._bounded_error_summary(
                scope="peer",
                stage="launch",
                error_code="PRX-PEER-LAUNCH",
                reason_code="process_start_failed",
                count=summary.peer_failed_count,
            ),
        )
        self._run_started_emitted = True
        self._capture(event, grant_id=grant_id)

    def _record_generation_finished(
        self,
        summary: PeerLifecycleSummary,
        *,
        grant_id: str,
    ) -> None:
        if summary.generation_ordinal in self._finished_generations:
            return
        if not self._run_started_emitted:
            self._record_run_started(
                PeerLifecycleSummary.planned(
                    generation_ordinal=summary.generation_ordinal,
                    planned_peer_count=summary.planned_peer_count,
                ),
                grant_id=grant_id,
            )
        event = self._context.generation_finished(
            self._peer_summary(summary),
            error_summaries=self._bounded_error_summary(
                scope="peer",
                stage="execution",
                error_code="PRX-PEER-RUNTIME",
                reason_code="runtime_error",
                count=summary.peer_failed_count,
            ),
        )
        self._finished_generations.add(summary.generation_ordinal)
        self._capture(event, grant_id=grant_id)

    def _record_run_finished(
        self,
        active_duration_seconds: int | float | None,
        *,
        failed: bool,
        grant_id: str,
    ) -> None:
        if self._run_finished_emitted:
            return
        minutes: int | None = None
        capped = False
        if active_duration_seconds is not None:
            minutes = max(0, int(active_duration_seconds // 60))
            if minutes > 43_200:
                minutes = 43_200
                capped = True
        stage = "reconciliation" if self._resumed_unfinished else "finalization"
        error_summaries = self._bounded_error_summary(
            scope="run",
            stage=stage,
            error_code="PRX-RUN-FAILED",
            reason_code="unexpected_termination",
            count=int(failed),
        )
        if self._resumed_unfinished:
            event = self._context.run_reconciled(
                minutes,
                duration_capped=capped,
                error_summaries=error_summaries,
            )
        else:
            event = self._context.run_finished(
                minutes,
                duration_capped=capped,
                error_summaries=error_summaries,
            )
        self._run_finished_emitted = True
        self._capture(event, grant_id=grant_id)

    def _restore_or_create_context(self, praxist_version: str) -> RunTelemetryContext:
        try:
            with self._state_path.open("rb") as handle:
                encoded = handle.read(_MAX_RUN_STATE_FILE_BYTES + 1)
            if len(encoded) > _MAX_RUN_STATE_FILE_BYTES:
                raise ValueError("product-usage run state is oversized")
            payload = json.loads(encoded)
            if int(payload["schema_version"]) != 2:
                raise ValueError("incompatible product-usage run state")
            stored_environment_id = UUID(str(payload["environment_id"]))
            if stored_environment_id != self._environment_id:
                raise ValueError("run state belongs to another Environment ID")
            run_started_emitted = bool(payload["run_started_emitted"])
            finished_generations = {int(value) for value in payload["finished_generations"]}
            if len(finished_generations) > _MAX_TRACKED_GENERATIONS:
                raise ValueError("product-usage generation state exceeds its fixed bound")
            context = RunTelemetryContext.resume(
                praxist_version,
                environment_id=self._environment_id,
                telemetry_run_id=UUID(str(payload["telemetry_run_id"])),
                next_sequence=int(payload["next_sequence"]),
                run_started_emitted=run_started_emitted,
                finished_generations=finished_generations,
            )
            self._run_started_emitted = run_started_emitted
            self._run_finished_emitted = bool(payload.get("run_finished_emitted", False))
            self._resumed_unfinished = self._run_started_emitted and not self._run_finished_emitted
            self._finished_generations = finished_generations
            return context
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._run_started_emitted = False
            self._run_finished_emitted = False
            self._resumed_unfinished = False
            self._finished_generations = set()
            return RunTelemetryContext(
                praxist_version,
                environment_id=self._environment_id,
            )

    @staticmethod
    def _peer_summary(summary: PeerLifecycleSummary) -> PeerStatusSummary:
        return PeerStatusSummary(
            generation_ordinal=summary.generation_ordinal,
            planned_peer_count=summary.planned_peer_count,
            peer_planned_count=summary.peer_planned_count,
            peer_running_count=summary.peer_running_count,
            peer_completed_count=summary.peer_completed_count,
            peer_cancelled_count=summary.peer_cancelled_count,
            peer_failed_count=summary.peer_failed_count,
            peer_unknown_count=summary.peer_unknown_count,
        )

    @staticmethod
    def _bounded_error_summary(
        *,
        scope: str,
        stage: str,
        error_code: PraxistErrorCode,
        reason_code: str,
        count: int,
    ) -> tuple[ErrorSummary, ...]:
        if count < 1:
            return ()
        return (
            ErrorSummary(
                scope=ErrorScope(scope),
                stage=ErrorStage(stage),
                error_type=ErrorType.RUNTIME,
                error_code=error_code,
                reason_code=ReasonCode(reason_code),
                count=min(count, MAX_ERROR_COUNT),
                count_capped=count > MAX_ERROR_COUNT,
            ),
        )

    def _capture(self, event: Any, *, grant_id: str) -> bool:
        # Commit the idempotence/sequence state before the outbox. A crash may
        # omit an optional usage event, but can never duplicate a boundary.
        if not self._persist_state():
            return False
        try:
            if grant_id:
                captured = bool(self._sdk.capture(event, expected_grant_id=grant_id))
            else:
                captured = bool(self._sdk.capture(event))
        except Exception:
            captured = False
        if captured and self._upload_once is not None:
            self._request_upload()
        return captured

    def _persist_state(self) -> bool:
        try:
            self._state_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                os.chmod(self._state_path.parent, 0o700)
            payload = {
                "schema_version": 2,
                "environment_id": str(self._context.environment_id),
                "telemetry_run_id": str(self._context.telemetry_run_id),
                "next_sequence": self._context.next_sequence,
                "run_started_emitted": self._run_started_emitted,
                "run_finished_emitted": self._run_finished_emitted,
                "finished_generations": sorted(self._finished_generations),
            }
            temporary = self._state_path.with_name(f".{self._state_path.name}.tmp")
            descriptor = os.open(
                temporary,
                os.O_CREAT | os.O_TRUNC | os.O_WRONLY,
                0o600,
            )
            encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
            if len(encoded) > _MAX_RUN_STATE_FILE_BYTES:
                os.close(descriptor)
                temporary.unlink(missing_ok=True)
                return False
            try:
                os.chmod(temporary, 0o600)
                handle = os.fdopen(descriptor, "wb")
                descriptor = -1
                with handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self._state_path)
                _fsync_directory(self._state_path.parent)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                temporary.unlink(missing_ok=True)
            return _prune_run_states(self._state_path.parent, keep=self._state_path)
        except Exception:
            return False

    def _request_upload(self) -> None:
        if self._upload_once is None:
            return
        with self._upload_lock:
            if self._upload_thread is not None and self._upload_thread.is_alive():
                self._upload_pending = True
                return
            self._upload_pending = False
            self._upload_thread = threading.Thread(
                target=self._upload_worker,
                name="praxist-product-usage-upload",
                daemon=True,
            )
            self._upload_thread.start()

    def _upload_worker(self) -> None:
        upload_once = self._upload_once
        if upload_once is None:
            with self._upload_lock:
                self._upload_thread = None
            return
        while True:
            try:
                if self._external_scheduler is None:
                    uploaded = int(upload_once())
                else:
                    self._external_scheduler(upload_once)
                    uploaded = 0
            except Exception:
                uploaded = 0
            with self._upload_lock:
                if uploaded > 0 or self._upload_pending:
                    self._upload_pending = False
                    continue
                self._upload_thread = None
                return


def _fsync_directory(directory: Path) -> None:
    """Commit an atomic rename before the corresponding outbox write."""

    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remaining_seconds(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _prune_run_states(directory: Path, *, keep: Path) -> bool:
    """Bound private observer state without touching Research Run artifacts."""

    now = time.time()
    for temporary in directory.glob(".*.json.tmp"):
        try:
            if now - temporary.stat().st_mtime > _RUN_STATE_TEMP_RETENTION_SECONDS:
                temporary.unlink()
        except OSError:
            continue
    entries: list[tuple[Path, int, float, bool]] = []
    for path in directory.glob("*.json"):
        try:
            file_stat = path.stat()
        except OSError:
            continue
        if (
            path != keep
            and now - file_stat.st_mtime > _RUN_STATE_RETENTION_SECONDS
            and _unlink_run_state_if_unowned(path)
        ):
            continue
        terminal = False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            terminal = bool(payload.get("run_finished_emitted", False))
        except (OSError, AttributeError, json.JSONDecodeError):
            terminal = True
        entries.append((path, file_stat.st_size, file_stat.st_mtime, terminal))

    total_bytes = sum(entry[1] for entry in entries)
    candidates = sorted(
        (entry for entry in entries if entry[0] != keep),
        key=lambda entry: (not entry[3], entry[2]),
    )
    while (
        len(entries) > _MAX_RUN_STATE_FILES or total_bytes > _MAX_RUN_STATE_BYTES
    ) and candidates:
        candidate = candidates.pop(0)
        if not _unlink_run_state_if_unowned(candidate[0]):
            continue
        entries.remove(candidate)
        total_bytes -= candidate[1]
    return len(entries) <= _MAX_RUN_STATE_FILES and total_bytes <= _MAX_RUN_STATE_BYTES


def _unlink_run_state_if_unowned(path: Path) -> bool:
    """Delete a state only while holding the same slot used by its observer."""

    state_lock = _try_acquire_run_state_lock(path)
    if state_lock is None:
        return False
    try:
        path.unlink()
    except OSError:
        return False
    finally:
        _release_run_state_lock(state_lock)
    return True


def _try_acquire_run_state_lock(state_path: Path) -> tuple[int, int, Path] | None:
    state_path = Path(os.path.abspath(state_path))
    state_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = state_path.parent / _RUN_STATE_LOCK_FILE
    lock_slot = (
        int.from_bytes(
            hashlib.sha256(str(state_path).encode("utf-8")).digest()[:8],
            "big",
        )
        % _RUN_STATE_LOCK_SLOTS
    )
    lock_path = Path(os.path.abspath(lock_path))
    registry_key = (lock_path, lock_slot)
    with _RUN_STATE_LOCK_REGISTRY_GUARD:
        if registry_key in _RUN_STATE_LOCK_REGISTRY:
            return None
        _RUN_STATE_LOCK_REGISTRY.add(registry_key)
    descriptor: int | None = None
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        with contextlib.suppress(OSError):
            os.chmod(lock_path, 0o600)
        if os.fstat(descriptor).st_size < _RUN_STATE_LOCK_SLOTS:
            os.ftruncate(descriptor, _RUN_STATE_LOCK_SLOTS)
        if os.name == "nt":  # pragma: no cover - exercised by Windows CI.
            import msvcrt

            os.lseek(descriptor, lock_slot, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.lockf(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, lock_slot)
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
        with _RUN_STATE_LOCK_REGISTRY_GUARD:
            _RUN_STATE_LOCK_REGISTRY.discard(registry_key)
        return None
    assert descriptor is not None
    return descriptor, lock_slot, lock_path


def _release_run_state_lock(state_lock: tuple[int, int, Path]) -> None:
    descriptor, lock_slot, lock_path = state_lock
    try:
        if os.name == "nt":  # pragma: no cover - exercised by Windows CI.
            import msvcrt

            os.lseek(descriptor, lock_slot, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.lockf(descriptor, fcntl.LOCK_UN, 1, lock_slot)
    finally:
        try:
            os.close(descriptor)
        finally:
            with _RUN_STATE_LOCK_REGISTRY_GUARD:
                _RUN_STATE_LOCK_REGISTRY.discard((lock_path, lock_slot))
