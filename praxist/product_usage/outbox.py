"""Permission-protected, bounded SQLite outbox."""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from .paths import outbox_path
from .protocol import MAX_REQUEST_BYTES, UsageEvent, canonical_event_json

MAX_OUTBOX_EVENTS = 1_000
MAX_OUTBOX_BYTES = 1 * 1024 * 1024
OUTBOX_RETENTION_SECONDS = 7 * 24 * 60 * 60
_BATCH_ENVELOPE_BYTES = len(b'{"events":[') + len(b"]}")
_OUTBOX_SCHEMA_VERSION = 3


@dataclass(frozen=True, slots=True)
class QueuedEvent:
    """One validated outbox row scoped to a durable consent grant."""

    event_id: str
    grant_id: str
    payload: str
    payload_size: int
    created_at_epoch: int
    expires_at_epoch: int


class Outbox:
    """A small FIFO store containing only already-validated safe projections."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or outbox_path()
        self._connection: sqlite3.Connection | None = None
        self._connection_identity: tuple[int, int, int] | None = None
        self._clock: Callable[[], float] = time.time

    @classmethod
    def _at_path_for_tests(
        cls,
        path: Path,
        *,
        clock: Callable[[], float] = time.time,
    ) -> Outbox:
        outbox = cls(path)
        outbox._clock = clock
        return outbox

    @property
    def path(self) -> Path:
        return self._path

    def enqueue(self, event: UsageEvent, *, grant_id: str) -> bool:
        """Insert once by Event ID, then enforce expiry and capacity."""

        _validate_grant_id(grant_id)
        payload = canonical_event_json(event)
        payload_size = len(payload.encode("utf-8"))
        if payload_size + _BATCH_ENVELOPE_BYTES > MAX_REQUEST_BYTES:
            raise ValueError("one event cannot fit inside the maximum request body")

        now = int(self._clock())
        connection = self._connect()
        with connection:
            self._purge_expired(connection, now)
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO usage_outbox (
                    event_id,
                    grant_id,
                    payload,
                    payload_size,
                    created_at_epoch,
                    expires_at_epoch
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(event.event_id),
                    grant_id,
                    payload,
                    payload_size,
                    now,
                    now + OUTBOX_RETENTION_SECONDS,
                ),
            )
            inserted = cursor.rowcount == 1
            self._enforce_capacity(connection)
        self._protect_sqlite_files()
        return inserted

    def fetch_oldest(
        self,
        *,
        grant_id: str,
        limit: int = MAX_OUTBOX_EVENTS,
    ) -> list[QueuedEvent]:
        _validate_grant_id(grant_id)
        if limit < 1:
            return []
        now = int(self._clock())
        connection = self._connect()
        with connection:
            self._purge_expired(connection, now)
            rows = connection.execute(
                """
                SELECT event_id, grant_id, payload, payload_size,
                       created_at_epoch, expires_at_epoch
                FROM usage_outbox
                WHERE grant_id = ?
                ORDER BY created_at_epoch ASC, rowid ASC
                LIMIT ?
                """,
                (grant_id, min(limit, MAX_OUTBOX_EVENTS)),
            ).fetchall()
        return [QueuedEvent(*row) for row in rows]

    def acknowledge(self, event_ids: set[str], *, grant_id: str) -> int:
        _validate_grant_id(grant_id)
        if not event_ids:
            return 0
        connection = self._connect()
        deleted = 0
        with connection:
            for event_id in event_ids:
                cursor = connection.execute(
                    "DELETE FROM usage_outbox WHERE event_id = ? AND grant_id = ?",
                    (event_id, grant_id),
                )
                deleted += cursor.rowcount
        return deleted

    def discard_other_grants(self, grant_id: str) -> int:
        """Remove rows that no longer belong to the current explicit grant."""

        _validate_grant_id(grant_id)
        connection = self._connect()
        with connection:
            cursor = connection.execute(
                "DELETE FROM usage_outbox WHERE grant_id IS NULL OR grant_id != ?",
                (grant_id,),
            )
        return cursor.rowcount

    def count(self) -> int:
        connection = self._connect()
        with connection:
            self._purge_expired(connection, int(self._clock()))
            value = connection.execute("SELECT COUNT(*) FROM usage_outbox").fetchone()
        assert value is not None
        return int(value[0])

    def logical_payload_bytes(self) -> int:
        connection = self._connect()
        with connection:
            self._purge_expired(connection, int(self._clock()))
            value = connection.execute(
                "SELECT COALESCE(SUM(payload_size), 0) FROM usage_outbox"
            ).fetchone()
        assert value is not None
        return int(value[0])

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._connection_identity = None

    def close_and_delete(self) -> None:
        """Secure-delete rows, close SQLite, then unlink DB/WAL/SHM files."""

        try:
            if self._connection is None and self._path.exists():
                self._connect()
            if self._connection is not None:
                try:
                    self._connection.execute("DELETE FROM usage_outbox")
                    self._connection.commit()
                    self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.Error:
                    # Unlinking all SQLite files remains the fail-closed fallback.
                    pass
                finally:
                    self.close()
        finally:
            for candidate in (
                self._path,
                Path(f"{self._path}-wal"),
                Path(f"{self._path}-shm"),
            ):
                with suppress(FileNotFoundError):
                    candidate.unlink()

    def _connect(self) -> sqlite3.Connection:
        if self._connection is not None:
            try:
                current_identity = _path_identity(self._path)
            except OSError:
                current_identity = None
            if current_identity == self._connection_identity:
                return self._connection
            self.close()
        directory = self._path.parent
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        _chmod_user_only(directory, 0o700)
        connection = sqlite3.connect(self._path)
        connection.execute("PRAGMA secure_delete = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_outbox (
                event_id TEXT PRIMARY KEY,
                grant_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                payload_size INTEGER NOT NULL CHECK (payload_size >= 0),
                created_at_epoch INTEGER NOT NULL,
                expires_at_epoch INTEGER NOT NULL
            )
            """
        )
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(usage_outbox)")}
        if "grant_id" not in columns:
            connection.execute("ALTER TABLE usage_outbox ADD COLUMN grant_id TEXT")
            connection.execute("DELETE FROM usage_outbox")
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS usage_outbox_expiry
            ON usage_outbox (expires_at_epoch)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_outbox_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        stored_version = connection.execute(
            "SELECT value FROM usage_outbox_metadata WHERE key = ?",
            ("schema_version",),
        ).fetchone()
        if stored_version is None or stored_version[0] != str(_OUTBOX_SCHEMA_VERSION):
            connection.execute("DELETE FROM usage_outbox")
            connection.execute(
                """
                INSERT INTO usage_outbox_metadata (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                ("schema_version", str(_OUTBOX_SCHEMA_VERSION)),
            )
        connection.commit()
        _chmod_user_only(self._path, 0o600)
        self._connection = connection
        self._connection_identity = _path_identity(self._path)
        return connection

    def _protect_sqlite_files(self) -> None:
        for candidate in (
            self._path,
            Path(f"{self._path}-wal"),
            Path(f"{self._path}-shm"),
        ):
            if candidate.exists():
                _chmod_user_only(candidate, 0o600)

    @staticmethod
    def _purge_expired(connection: sqlite3.Connection, now: int) -> None:
        connection.execute(
            "DELETE FROM usage_outbox WHERE expires_at_epoch <= ?",
            (now,),
        )

    @staticmethod
    def _enforce_capacity(connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT event_id, payload_size
            FROM usage_outbox
            ORDER BY created_at_epoch ASC, rowid ASC
            """
        ).fetchall()
        total_bytes = sum(int(row[1]) for row in rows)
        remove_count = 0
        while len(rows) - remove_count > MAX_OUTBOX_EVENTS or total_bytes > MAX_OUTBOX_BYTES:
            total_bytes -= int(rows[remove_count][1])
            remove_count += 1
        if remove_count:
            connection.executemany(
                "DELETE FROM usage_outbox WHERE event_id = ?",
                ((str(row[0]),) for row in rows[:remove_count]),
            )


def _chmod_user_only(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        if os.name != "nt":
            raise


def _validate_grant_id(grant_id: str) -> None:
    if not grant_id or len(grant_id) > 80 or not grant_id.isascii():
        raise ValueError("grant_id must be a nonempty bounded ASCII identifier")


def _path_identity(path: Path) -> tuple[int, int, int]:
    file_stat = path.stat()
    return file_stat.st_dev, file_stat.st_ino, file_stat.st_ctime_ns
