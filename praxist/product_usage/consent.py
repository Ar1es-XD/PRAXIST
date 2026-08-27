"""Minimal, explicit, fail-closed Consent State storage."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import UUID4, ValidationError

from .paths import consent_path
from .protocol import CONSENT_NOTICE_VERSION, StrictModel, utc_now_seconds

_LOCK_REGISTRY_GUARD = threading.Lock()
_LOCK_REGISTRY: dict[Path, threading.RLock] = {}
_FORCED_DENIALS: dict[Path, str | None] = {}


class ConsentDecision(StrEnum):
    """The two explicit decisions accepted by the consent store."""

    GRANTED = "granted"
    DENIED = "denied"


class ConsentStatus(StrEnum):
    """The effective local collection state, including an unset state."""

    UNSET = "unset"
    GRANTED = "granted"
    DENIED = "denied"


class ConsentRecord(StrictModel):
    """Durable record of one explicit decision for the current notice."""

    decision: ConsentDecision
    decision_id: UUID4 | None = None
    consent_notice_version: int
    decided_at: str
    source: Literal["direct", "agent", "withdrawal"]
    language: str


def parse_agent_reply(reply: str) -> ConsentDecision | None:
    """Parse only the four explicitly approved Agent interaction keywords."""

    normalized = reply.strip()
    if normalized in {"Yes", "Agree"}:
        return ConsentDecision.GRANTED
    if normalized in {"No", "Disagree"}:
        return ConsentDecision.DENIED
    return None


class ConsentStore:
    """Read and atomically write the fixed current-user consent record."""

    def __init__(self) -> None:
        self._path = consent_path()

    @classmethod
    def _at_path_for_tests(cls, path: Path) -> ConsentStore:
        store = cls.__new__(cls)
        store._path = path
        return store

    @property
    def path(self) -> Path:
        return self._path

    def read(self) -> ConsentRecord | None:
        try:
            payload = self._path.read_bytes()
            record = ConsentRecord.model_validate_json(payload)
        except (OSError, ValidationError):
            return None
        if record.consent_notice_version != CONSENT_NOTICE_VERSION:
            return None
        return record

    def status(self) -> ConsentStatus:
        forced, prior_grant_id = self._forced_denial()
        if forced:
            record = self.read()
            if (
                record is not None
                and record.decision is ConsentDecision.GRANTED
                and _grant_id_for_record(record) != prior_grant_id
            ):
                self._clear_forced_denial()
                return ConsentStatus.GRANTED
            return ConsentStatus.DENIED
        record = self.read()
        if record is None:
            return ConsentStatus.UNSET
        return ConsentStatus(record.decision.value)

    def grant_id(self) -> str | None:
        """Return the exact durable grant identity, or ``None`` when not granted."""

        if self.status() is not ConsentStatus.GRANTED:
            return None
        record = self.read()
        if record is None or record.decision is not ConsentDecision.GRANTED:
            return None
        return _grant_id_for_record(record)

    def write(
        self,
        decision: ConsentDecision,
        *,
        source: Literal["direct", "agent", "withdrawal"] = "direct",
        language: str = "en",
        _while_locked: Callable[[], None] | None = None,
    ) -> None:
        # Test and plugin reloads may retain an equivalent StrEnum instance from
        # the prior module object. Revalidate by its closed value set before
        # constructing the strict persisted model.
        decision = ConsentDecision(getattr(decision, "value", decision))
        if decision is ConsentDecision.DENIED:
            self._force_denied()
        quarantined: Path | None = None
        with self.exclusive_access(), self.capture_access():
            if decision is ConsentDecision.DENIED:
                quarantined = self._quarantine_existing_grant_unlocked()
            try:
                self._write_unlocked(decision, source=source, language=language)
                if _while_locked is not None:
                    _while_locked()
            finally:
                if quarantined is not None:
                    with suppress(OSError):
                        quarantined.unlink()
        if decision is ConsentDecision.GRANTED:
            with suppress(OSError):
                self._revoked_path().unlink()
            self._clear_forced_denial()

    def _write_unlocked(
        self,
        decision: ConsentDecision,
        *,
        source: Literal["direct", "agent", "withdrawal"],
        language: str,
    ) -> None:
        record = ConsentRecord(
            decision=decision,
            decision_id=uuid4(),
            consent_notice_version=CONSENT_NOTICE_VERSION,
            decided_at=utc_now_seconds(),
            source=source,
            language=language,
        )
        encoded = (
            json.dumps(
                record.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        directory = self._path.parent
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        _chmod_user_only(directory, 0o700)

        temporary: Path | None = None
        try:
            descriptor, name = tempfile.mkstemp(prefix=".consent-", dir=directory)
            temporary = Path(name)
            os.chmod(temporary, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._path)
            temporary = None
            _chmod_user_only(self._path, 0o600)
            _fsync_directory(directory)
        finally:
            if temporary is not None:
                with suppress(FileNotFoundError):
                    temporary.unlink()

    def reset(self) -> None:
        """Remove the record so subsequent reads fail closed to unset."""

        with self.exclusive_access(), self.capture_access():
            with suppress(FileNotFoundError):
                self._path.unlink()
            with suppress(OSError):
                self._revoked_path().unlink()
        self._clear_forced_denial()

    @contextmanager
    def exclusive_access(self) -> Iterator[None]:
        """Serialize upload and consent changes across threads and processes."""

        lock_path = self._path.with_name(f"{self._path.name}.lock")
        with self._serialized_access(lock_path):
            yield

    @contextmanager
    def capture_access(self) -> Iterator[None]:
        """Serialize local capture with consent changes, independent of upload."""

        lock_path = self._path.with_name(f"{self._path.name}.capture.lock")
        with self._serialized_access(lock_path):
            yield

    @contextmanager
    def _serialized_access(self, lock_path: Path) -> Iterator[None]:
        """Hold one named consent lock across threads and POSIX processes."""

        directory = self._path.parent
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        _chmod_user_only(directory, 0o700)
        registry_key = _path_key(lock_path)
        with _LOCK_REGISTRY_GUARD:
            thread_lock = _LOCK_REGISTRY.setdefault(registry_key, threading.RLock())
        with thread_lock:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            locked = False
            try:
                _chmod_user_only(lock_path, 0o600)
                _lock_descriptor(descriptor)
                locked = True
                yield
            finally:
                try:
                    if locked:
                        _unlock_descriptor(descriptor)
                finally:
                    os.close(descriptor)

    def _forced_denial(self) -> tuple[bool, str | None]:
        with _LOCK_REGISTRY_GUARD:
            key = _path_key(self._path)
            return key in _FORCED_DENIALS, _FORCED_DENIALS.get(key)

    def _force_denied(self) -> None:
        record = self.read()
        prior_grant_id = (
            _grant_id_for_record(record)
            if record is not None and record.decision is ConsentDecision.GRANTED
            else None
        )
        with _LOCK_REGISTRY_GUARD:
            _FORCED_DENIALS[_path_key(self._path)] = prior_grant_id

    def _clear_forced_denial(self) -> None:
        with _LOCK_REGISTRY_GUARD:
            _FORCED_DENIALS.pop(_path_key(self._path), None)

    def _quarantine_existing_grant_unlocked(self) -> Path | None:
        if not self._path.exists():
            return None
        revoked = self._revoked_path()
        os.replace(self._path, revoked)
        _chmod_user_only(revoked, 0o600)
        _fsync_directory(self._path.parent)
        return revoked

    def _revoked_path(self) -> Path:
        return self._path.with_name(f".{self._path.name}.revoked")


def _path_key(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _grant_id_for_record(record: ConsentRecord) -> str:
    if record.decision_id is not None:
        return str(record.decision_id)
    # Records written before grant identities were introduced remain usable.
    # Any subsequent decision gets a UUID, so this stable legacy identity still
    # cannot match a withdrawal/regrant cycle.
    payload = json.dumps(
        record.model_dump(mode="json", exclude={"decision_id"}),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"legacy-{hashlib.sha256(payload).hexdigest()}"


def _lock_descriptor(descriptor: int) -> None:
    if os.name == "nt":  # pragma: no cover - exercised by Windows CI.
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        return
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_EX)


def _unlock_descriptor(descriptor: int) -> None:
    if os.name == "nt":  # pragma: no cover - exercised by Windows CI.
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_UN)


def _chmod_user_only(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        # Windows ACLs are not represented by POSIX mode bits.
        if os.name != "nt":
            raise


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
