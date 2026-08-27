"""Filesystem-event waits used by the research loop.

The production path runs on Linux, so this module uses inotify through the
standard library's ``ctypes``/``select`` stack instead of adding a dependency.
Callers get real file-change wakeups when available and a deliberately
low-frequency fallback on platforms where inotify is unavailable.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import errno
import logging
import os
import select
import struct
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_EVENT_STRUCT = struct.Struct("iIII")
_IN_ACCESS = 0x00000001
_IN_MODIFY = 0x00000002
_IN_ATTRIB = 0x00000004
_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_FROM = 0x00000040
_IN_MOVED_TO = 0x00000080
_IN_CREATE = 0x00000100
_IN_DELETE = 0x00000200
_IN_DELETE_SELF = 0x00000400
_IN_MOVE_SELF = 0x00000800
_IN_UNMOUNT = 0x00002000
_IN_Q_OVERFLOW = 0x00004000
_IN_IGNORED = 0x00008000
_IN_ISDIR = 0x40000000

_WATCH_MASK = (
    _IN_MODIFY
    | _IN_ATTRIB
    | _IN_CLOSE_WRITE
    | _IN_MOVED_FROM
    | _IN_MOVED_TO
    | _IN_CREATE
    | _IN_DELETE
    | _IN_DELETE_SELF
    | _IN_MOVE_SELF
    | _IN_UNMOUNT
    | _IN_Q_OVERFLOW
)


@dataclass(frozen=True)
class FileEventWaitResult:
    """Result of waiting for filesystem events, stop signals, or heartbeat timeouts."""

    reason: str
    elapsed_seconds: float
    paths: tuple[str, ...] = ()
    used_inotify: bool = False


class _InotifyWaiter:
    def __init__(
        self,
        roots: Iterable[Path],
        *,
        recursive: bool,
        max_dirs: int,
        event_filter: Callable[[Path], bool] | None = None,
    ):
        self._libc = ctypes.CDLL("libc.so.6", use_errno=True)
        flags = getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
        fd = self._libc.inotify_init1(flags)
        if fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        self.fd = int(fd)
        self.recursive = bool(recursive)
        self.max_dirs = int(max_dirs)
        self.event_filter = event_filter
        self._wd_to_path: dict[int, Path] = {}
        self._path_to_wd: dict[Path, int] = {}
        self._closed = False
        for root in roots:
            self._add_tree(Path(root))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(OSError):
            os.close(self.fd)

    def _add_watch(self, path: Path) -> None:
        try:
            rp = path.resolve()
        except OSError:
            return
        if rp in self._path_to_wd or not rp.is_dir():
            return
        if len(self._wd_to_path) >= self.max_dirs:
            logger.debug(
                "event_wait: watch dir cap reached (%d); skipping %s",
                self.max_dirs,
                rp,
            )
            return
        wd = self._libc.inotify_add_watch(
            self.fd,
            os.fsencode(str(rp)),
            ctypes.c_uint32(_WATCH_MASK),
        )
        if wd < 0:
            err = ctypes.get_errno()
            if err not in (errno.ENOENT, errno.ENOTDIR, errno.EACCES, errno.EPERM):
                logger.debug("event_wait: inotify_add_watch(%s) failed: %s", rp, err)
            return
        self._wd_to_path[int(wd)] = rp
        self._path_to_wd[rp] = int(wd)

    def _add_tree(self, root: Path) -> None:
        try:
            root = root.resolve()
        except OSError:
            return
        if not root.exists():
            root = root.parent
        if not root.exists():
            return
        if root.is_file():
            root = root.parent
        self._add_watch(root)
        if not self.recursive:
            return
        try:
            for child in root.rglob("*"):
                if len(self._wd_to_path) >= self.max_dirs:
                    break
                if child.is_dir():
                    self._add_watch(child)
        except OSError:
            return

    def _read_events(self) -> tuple[str, ...]:
        changed: list[str] = []
        while True:
            try:
                data = os.read(self.fd, 65536)
            except BlockingIOError:
                break
            except OSError as e:
                if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    break
                raise
            if not data:
                break
            offset = 0
            while offset + _EVENT_STRUCT.size <= len(data):
                wd, mask, _cookie, name_len = _EVENT_STRUCT.unpack_from(data, offset)
                offset += _EVENT_STRUCT.size
                raw_name = data[offset : offset + name_len].split(b"\0", 1)[0]
                offset += name_len
                base = self._wd_to_path.get(int(wd))
                if base is None:
                    continue
                path = base / os.fsdecode(raw_name) if raw_name else base
                if self.recursive and (mask & _IN_ISDIR) and (mask & (_IN_CREATE | _IN_MOVED_TO)):
                    self._add_tree(path)
                if self.event_filter is not None:
                    try:
                        if not self.event_filter(path):
                            continue
                    except Exception as e:
                        logger.debug("event_wait: event_filter failed for %s: %s", path, e)
                        continue
                changed.append(str(path))
                if mask & (_IN_IGNORED | _IN_DELETE_SELF | _IN_MOVE_SELF):
                    try:
                        rp = base.resolve()
                    except OSError:
                        rp = base
                    self._path_to_wd.pop(rp, None)
                    self._wd_to_path.pop(int(wd), None)
        return tuple(changed)

    def wait(
        self,
        timeout_seconds: float,
        *,
        stop_check: Callable[[], bool] | None,
        stop_check_interval_seconds: float,
    ) -> FileEventWaitResult:
        start = time.monotonic()
        deadline = start + max(0.0, float(timeout_seconds))
        stop_interval = max(1.0, float(stop_check_interval_seconds))
        while True:
            if stop_check is not None:
                try:
                    if stop_check():
                        return FileEventWaitResult(
                            reason="stop",
                            elapsed_seconds=time.monotonic() - start,
                            used_inotify=True,
                        )
                except Exception as e:
                    logger.debug("event_wait: stop_check failed: %s", e)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return FileEventWaitResult(
                    reason="timeout",
                    elapsed_seconds=time.monotonic() - start,
                    used_inotify=True,
                )
            wait_s = min(remaining, stop_interval)
            readable, _, _ = select.select([self.fd], [], [], wait_s)
            if not readable:
                continue
            paths = self._read_events()
            if paths:
                return FileEventWaitResult(
                    reason="filesystem_event",
                    elapsed_seconds=time.monotonic() - start,
                    paths=paths,
                    used_inotify=True,
                )


def _candidate_watch_roots(paths: Iterable[Path]) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        try:
            path = path.resolve()
        except OSError:
            path = path.absolute()
        root = path if path.exists() and path.is_dir() else path.parent
        try:
            root = root.resolve()
        except OSError:
            root = root.absolute()
        if root in seen:
            continue
        seen.add(root)
        roots.append(root)
    return roots


async def wait_for_filesystem_event(
    paths: Iterable[Path | str],
    *,
    timeout_seconds: float,
    stop_check: Callable[[], bool] | None = None,
    recursive: bool = False,
    max_dirs: int = 512,
    fallback_interval_seconds: float = 900.0,
    stop_check_interval_seconds: float = 30.0,
    event_filter: Callable[[Path], bool] | None = None,
) -> FileEventWaitResult:
    """Wait for a filesystem event under ``paths``.

    ``reason`` is one of:
    - ``filesystem_event``: inotify observed a change.
    - ``timeout``: no event before the requested timeout.
    - ``stop``: ``stop_check`` asked the caller to exit.
    - ``no_watch_paths``: no parent directory existed to watch.
    - ``fallback_elapsed``: inotify unavailable; low-frequency fallback elapsed.
    """
    roots = _candidate_watch_roots(Path(p) for p in paths)
    roots = [p for p in roots if p.exists() and p.is_dir()]
    if not roots:
        await _sleep_with_stop_checks(
            min(float(timeout_seconds), float(fallback_interval_seconds)),
            stop_check=stop_check,
            stop_check_interval_seconds=stop_check_interval_seconds,
        )
        return FileEventWaitResult(
            reason="no_watch_paths",
            elapsed_seconds=min(float(timeout_seconds), float(fallback_interval_seconds)),
            used_inotify=False,
        )

    try:
        waiter = _InotifyWaiter(
            roots,
            recursive=recursive,
            max_dirs=max_dirs,
            event_filter=event_filter,
        )
    except Exception as e:
        logger.debug("event_wait: inotify unavailable, using fallback sleep: %s", e)
        start = time.monotonic()
        stopped = await _sleep_with_stop_checks(
            min(float(timeout_seconds), float(fallback_interval_seconds)),
            stop_check=stop_check,
            stop_check_interval_seconds=stop_check_interval_seconds,
        )
        return FileEventWaitResult(
            reason="stop" if stopped else "fallback_elapsed",
            elapsed_seconds=time.monotonic() - start,
            used_inotify=False,
        )

    try:
        return await asyncio.to_thread(
            waiter.wait,
            timeout_seconds,
            stop_check=stop_check,
            stop_check_interval_seconds=stop_check_interval_seconds,
        )
    finally:
        waiter.close()


async def _sleep_with_stop_checks(
    seconds: float,
    *,
    stop_check: Callable[[], bool] | None,
    stop_check_interval_seconds: float,
) -> bool:
    start = time.monotonic()
    deadline = start + max(0.0, float(seconds))
    interval = max(1.0, float(stop_check_interval_seconds))
    while True:
        if stop_check is not None:
            try:
                if stop_check():
                    return True
            except Exception as e:
                logger.debug("event_wait: fallback stop_check failed: %s", e)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(interval, remaining))
