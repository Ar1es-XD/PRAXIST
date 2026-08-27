"""
Atomic file I/O helpers with optimistic concurrency control.

Provides two families of helpers:

1. `atomic_write_json(path, data)` — tmp + fsync + rename. POSIX-atomic on the
   same filesystem. Use for shared-state writes where last-writer-wins is
   acceptable (e.g. shared_findings JSON files published to a uniquely-named path).

2. `atomic_write_json_cas(path, data, expected_version, ...)` — read-modify-write
   with a `_version` integer field. Aborts if the file's current version differs
   from `expected_version`. Use for shared mutable state where two writers might
   race (e.g. a shared notebook across peers).

On CAS conflict, the writer escalates by writing to
`PATH.conflict.<peer>.<nonce>.json` so no data is lost, and returns a structured
result the caller can surface in logs.

Example failure mode this avoids:
    L:25695 peer2: "My notebook was overwritten by Peer3!"
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# The key used to track version on shared JSON state.
VERSION_FIELD = "_version"

# Exit codes surfaced by the CAS helper so callers can distinguish cases.
CAS_OK = "ok"
CAS_CONFLICT = "conflict"
CAS_CONFLICT_DUMPED = "conflict_dumped"
CAS_CREATED = "created"


@dataclass
class CasResult:
    """Compare-and-swap write result for atomic JSON file updates."""

    status: str  # one of CAS_OK, CAS_CONFLICT, CAS_CONFLICT_DUMPED, CAS_CREATED
    new_version: int
    actual_version: int | None = None
    conflict_path: Path | None = None


def _fsync_file(f):
    """Force buffered writes to disk. Best-effort on platforms where fsync fails."""
    try:
        f.flush()
        os.fsync(f.fileno())
    except (OSError, AttributeError):
        # fsync is not available on every FS (e.g. some network mounts).
        # Rename atomicity still holds; durability is best-effort.
        pass


def atomic_write_json(
    path: os.PathLike,
    data: Any,
    *,
    indent: int = 2,
) -> Path:
    """Write `data` as JSON to `path` atomically via tmp + rename.

    Same-directory rename is POSIX-atomic (within a single filesystem). Readers
    will see either the old file or the new file, never a partially-written one.

    Parameters
    ----------
    path : path-like
        Destination. Parent directory is created if missing.
    data : any JSON-serializable
        Payload.
    indent : int
        json.dump indent (default 2).

    Returns
    -------
    Path
        The final written path.

    Raises
    ------
    OSError, json.JSONEncodeError
        Propagated so callers can detect failure. On failure the tmp file is
        cleaned up best-effort.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # NamedTemporaryFile uses mkstemp under the hood — unique name per call.
    # delete=False so rename can retain the file.
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dest.name}.",
        suffix=".tmp",
        dir=str(dest.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=indent, default=str)
            _fsync_file(f)
        # mkstemp creates files with mode 0600 (owner-only). Chmod the TMP
        # file to 0644 BEFORE the rename, so the final dest exists atomically
        # with 0644 from the moment it becomes visible. If we chmod after
        # replace, there's a microsecond window during which a concurrent
        # reader opens dest at 0600 and hits PermissionError.
        # 0o644 is the right default for auto-research coordination JSON
        # (world-readable, owner-writable). We avoid the os.umask(0) +
        # restore idiom because os.umask is process-global and setting it
        # to 0 would race with concurrent file creation in other threads.
        # chmod failures are not worth erroring the write over; the file is
        # still written correctly, just with restrictive perms.
        with contextlib.suppress(OSError):
            os.chmod(tmp_path, 0o644)
        # os.replace is atomic on POSIX and on Windows since 3.3 — unlike
        # Path.rename which raises on Windows if dest exists.
        os.replace(tmp_path, dest)
        # R3-N6 fix: fsync the parent directory after rename so the
        # rename is durable on NFS / shared FS. Best-effort: some FS
        # don't support dir fsync (Windows, certain network FS).
        try:
            parent_fd = os.open(str(dest.parent), os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except OSError:
            pass  # not supported on this FS
        return dest
    except BaseException:
        # BaseException (not just Exception) so KeyboardInterrupt,
        # asyncio.CancelledError, and SystemExit also trigger tmp cleanup.
        # Before this fix a SIGINT between mkstemp and os.replace leaked
        # `.<dest>.<uuid>.tmp` files into the destination directory, which
        # accumulated across restarts.
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise


def read_json_with_version(
    path: os.PathLike,
) -> tuple[dict[str, Any] | None, int]:
    """Read a JSON file and extract its `_version` field.

    Returns (data, version). If the file does not exist, returns (None, 0).
    If the file exists but has no _version, returns (data, 0) — treated as
    unversioned legacy content which a subsequent CAS write will overwrite.
    """
    p = Path(path)
    if not p.exists():
        return None, 0
    try:
        with open(p) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("read_json_with_version: failed to read %s: %s", p, e)
        return None, 0
    if not isinstance(data, dict):
        # Not a dict — cannot carry version. Treat as legacy; version 0.
        return data, 0
    version = data.get(VERSION_FIELD, 0)
    if not isinstance(version, int):
        version = 0
    return data, version


def atomic_write_json_cas(
    path: os.PathLike,
    data: dict[str, Any],
    *,
    expected_version: int,
    peer_id: str = "",
    dump_on_conflict: bool = True,
    indent: int = 2,
) -> CasResult:
    """Atomic write with optimistic version-CAS.

    Semantics:
      1. Read current file; extract current _version (0 if missing or malformed).
      2. If `expected_version` matches, write data (with _version = current+1) atomically.
      3. If it does not match, do NOT overwrite. Optionally dump to
         `<path>.conflict.<peer_id>.<nonce>.json` so nothing is lost.

    This is NOT a strict lock — two near-simultaneous writers can both read
    the same version then both decide their write is valid. The tmp+rename
    step serializes at the filesystem level, but only one's write lands; the
    other writer must call `read_json_with_version` again to detect that its
    expected_version was stale.

    For stronger semantics, the caller should:
        for _ in range(3):
            data, v = read_json_with_version(path)
            new_data = {**data, ...}  # merge
            result = atomic_write_json_cas(path, new_data, expected_version=v)
            if result.status in (CAS_OK, CAS_CREATED):
                break

    Parameters
    ----------
    path : path-like
        Destination file.
    data : dict
        New payload. `_version` will be set by this helper — do not pre-set it.
    expected_version : int
        What version the caller last observed. Pass 0 if the file did not
        exist or was unversioned legacy content.
    peer_id : str
        Used in conflict-dump filename for traceability.
    dump_on_conflict : bool
        If True (default) and conflict is detected, data is written to a
        uniquely-named `.conflict.*.json` sidecar so no write is lost.
    indent : int
        json.dump indent.

    Returns
    -------
    CasResult
    """
    if not isinstance(data, dict):
        raise TypeError("atomic_write_json_cas requires a dict payload")

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    _, actual_version = read_json_with_version(dest)
    file_exists = dest.exists()

    if actual_version != expected_version:
        if dump_on_conflict:
            conflict_path = _conflict_dump_path(dest, peer_id)
            try:
                atomic_write_json(conflict_path, data, indent=indent)
            except OSError as e:
                logger.error("CAS conflict AND conflict-dump failed for %s: %s", dest, e)
                return CasResult(
                    status=CAS_CONFLICT,
                    new_version=expected_version,
                    actual_version=actual_version,
                )
            logger.warning(
                "CAS conflict writing %s (expected v%d, actual v%d); dumped to %s",
                dest,
                expected_version,
                actual_version,
                conflict_path,
            )
            return CasResult(
                status=CAS_CONFLICT_DUMPED,
                new_version=expected_version,
                actual_version=actual_version,
                conflict_path=conflict_path,
            )
        return CasResult(
            status=CAS_CONFLICT,
            new_version=expected_version,
            actual_version=actual_version,
        )

    # Version matches — write with incremented version.
    new_payload = dict(data)
    new_payload[VERSION_FIELD] = actual_version + 1
    atomic_write_json(dest, new_payload, indent=indent)
    return CasResult(
        status=CAS_CREATED if not file_exists else CAS_OK,
        new_version=actual_version + 1,
    )


def _conflict_dump_path(dest: Path, peer_id: str) -> Path:
    """Construct a uniquely-named sidecar path for a CAS-conflict dump."""
    nonce = uuid.uuid4().hex[:8]
    safe_peer = "".join(c if (c.isalnum() or c in "-_") else "_" for c in (peer_id or "unknown"))
    return dest.with_name(f"{dest.name}.conflict.{safe_peer}.{nonce}.json")
