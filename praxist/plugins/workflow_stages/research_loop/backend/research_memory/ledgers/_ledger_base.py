"""Append-only YAML-backed ledger store.

A ledger is a flat list of entries with stable IDs, audit trail, and
last-write-wins update semantics on per-entry fields. Concurrent writers
serialize through a per-file lock (fcntl.flock).

Why YAML, not SQLite?
  - Human-readable diffs (git-friendly post-mortem)
  - Schema evolves frequently in early phases
  - Per-run scope; size <10 MB even at 8 gens × 100 findings/gen
  - SQLite shared_store.db remains the source of truth for raw findings;
    ledgers are derived structured views.
"""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class LedgerEntry:
    """Base entry shape; subclasses add ledger-specific fields via `data`."""

    id: str
    created_at: str = field(default_factory=_utcnow_iso)
    created_by: str = "bootstrap"
    last_updated_at: str = field(default_factory=_utcnow_iso)
    update_trail: list[dict[str, Any]] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LedgerEntry:
        return cls(
            id=d["id"],
            created_at=d.get("created_at", _utcnow_iso()),
            created_by=d.get("created_by", "unknown"),
            last_updated_at=str(d.get("last_updated_at", d.get("created_at", _utcnow_iso()))),
            update_trail=list(d.get("update_trail", [])),
            data=dict(d.get("data", {})),
        )


class LedgerStore:
    """File-backed ledger with simple append + upsert semantics.

    Locking strategy:
      - Use fcntl.flock(LOCK_EX) on the file during read-modify-write.
      - On platforms without fcntl (Windows), fall back to a sidecar .lock
        sentinel file with O_CREAT|O_EXCL retry loop.
    """

    schema_version = "1.0"

    def __init__(self, ledger_path: Path, ledger_name: str):
        self.ledger_path = Path(ledger_path)
        self.ledger_name = ledger_name
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ I/O

    def _read_all(self) -> dict[str, Any]:
        if not self.ledger_path.exists():
            return {
                "ledger_name": self.ledger_name,
                "schema_version": self.schema_version,
                "entries": [],
            }
        try:
            import yaml

            with open(self.ledger_path, encoding="utf-8") as f:
                d = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(
                "ledger %s: read failed (%s); treating as empty",
                self.ledger_name,
                e,
            )
            return {
                "ledger_name": self.ledger_name,
                "schema_version": self.schema_version,
                "entries": [],
            }
        # Defensive shape check
        if not isinstance(d, dict):
            logger.warning("ledger %s: top-level not dict, resetting", self.ledger_name)
            return {
                "ledger_name": self.ledger_name,
                "schema_version": self.schema_version,
                "entries": [],
            }
        d.setdefault("ledger_name", self.ledger_name)
        d.setdefault("schema_version", self.schema_version)
        if not isinstance(d.get("entries"), list):
            d["entries"] = []
        return d

    def _atomic_write(self, payload: dict[str, Any]) -> None:
        """Write via tmp + rename + dir fsync (NFS-safe; mirrors atomic_io).

        R3#10 fix: catch os.replace failure, clean up the orphan .tmp,
        and re-raise so caller knows the write failed.
        """
        import yaml

        tmp_path = self.ledger_path.with_suffix(self.ledger_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
            f.flush()
            with contextlib.suppress(OSError):
                os.fsync(f.fileno())
        try:
            os.replace(tmp_path, self.ledger_path)
        except OSError as e:
            logger.error(
                "ledger %s: atomic replace failed (%s); cleaning up orphan tmp %s",
                self.ledger_name,
                e,
                tmp_path,
            )
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)
            raise
        # parent dir fsync (R3-N6 pattern)
        try:
            dirfd = os.open(str(self.ledger_path.parent), os.O_RDONLY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
        except OSError:
            pass

    def _with_lock(self, fn):
        """Run fn() with exclusive lock on the ledger file."""
        try:
            import fcntl
        except ImportError:
            fcntl = None  # Windows
        # Use a sidecar lock so we don't lock the file we're about to replace.
        lock_path = self.ledger_path.with_suffix(self.ledger_path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        # touch lock file
        with open(lock_path, "a", encoding="utf-8") as lf:
            if fcntl is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                return fn()
            finally:
                if fcntl is not None:
                    with contextlib.suppress(OSError):
                        fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    # ------------------------------------------------------------------ public

    def list_entries(self) -> list[LedgerEntry]:
        d = self._read_all()
        return [LedgerEntry.from_dict(e) for e in d["entries"]]

    def get(self, entry_id: str) -> LedgerEntry | None:
        for e in self.list_entries():
            if e.id == entry_id:
                return e
        return None

    def upsert(
        self,
        entry_id: str,
        data: dict[str, Any],
        created_by: str = "unknown",
        action: str = "upsert",
    ) -> LedgerEntry:
        """Create or update an entry. Records update_trail."""

        def _do():
            d = self._read_all()
            now = _utcnow_iso()
            existing_idx = None
            for i, e in enumerate(d["entries"]):
                if e.get("id") == entry_id:
                    existing_idx = i
                    break
            if existing_idx is None:
                entry = LedgerEntry(
                    id=entry_id,
                    created_at=now,
                    created_by=created_by,
                    last_updated_at=now,
                    update_trail=[{"at": now, "by": created_by, "action": "create"}],
                    data=dict(data),
                )
                d["entries"].append(entry.to_dict())
            else:
                old = d["entries"][existing_idx]
                trail = list(old.get("update_trail", []))
                trail.append(
                    {
                        "at": now,
                        "by": created_by,
                        "action": action,
                        "diff_keys": sorted(
                            k for k, v in data.items() if old.get("data", {}).get(k) != v
                        ),
                    }
                )
                merged_data = dict(old.get("data", {}))
                merged_data.update(data)
                old["data"] = merged_data
                old["last_updated_at"] = now
                old["update_trail"] = trail[-50:]  # cap trail
                entry = LedgerEntry.from_dict(old)
                d["entries"][existing_idx] = old
            self._atomic_write(d)
            return entry

        return self._with_lock(_do)

    def append_only(
        self,
        entry_id: str,
        data: dict[str, Any],
        created_by: str = "unknown",
    ) -> LedgerEntry:
        """Create a new entry; refuse if id already exists."""

        def _do():
            d = self._read_all()
            for e in d["entries"]:
                if e.get("id") == entry_id:
                    raise ValueError(f"ledger {self.ledger_name}: id '{entry_id}' already exists")
            entry = LedgerEntry(
                id=entry_id,
                created_by=created_by,
                data=dict(data),
            )
            d["entries"].append(entry.to_dict())
            self._atomic_write(d)
            return entry

        return self._with_lock(_do)

    def filter(self, predicate) -> list[LedgerEntry]:
        return [e for e in self.list_entries() if predicate(e)]

    def __len__(self) -> int:
        return len(self._read_all().get("entries", []))
