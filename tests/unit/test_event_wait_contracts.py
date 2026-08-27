from __future__ import annotations

import asyncio
import errno
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class EventWaitContractsTest(unittest.TestCase):
    def test_candidate_roots_fallback_and_inotify_lifecycle_contracts(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import event_wait

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            watched = root / "watched"
            watched.mkdir()
            target = watched / "file.json"
            roots = event_wait._candidate_watch_roots([target, watched, root / "missing" / "x"])
            self.assertIn(watched.resolve(), roots)

            async def no_roots() -> event_wait.FileEventWaitResult:
                with patch.object(
                    event_wait, "_sleep_with_stop_checks", return_value=False
                ) as sleep:
                    result = await event_wait.wait_for_filesystem_event(
                        [root / "does-not-exist" / "x"],
                        timeout_seconds=10,
                        fallback_interval_seconds=2,
                    )
                sleep.assert_called_once()
                return result

            self.assertEqual(asyncio.run(no_roots()).reason, "no_watch_paths")

            async def fallback_stop() -> event_wait.FileEventWaitResult:
                with (
                    patch.object(event_wait, "_InotifyWaiter", side_effect=OSError("no inotify")),
                    patch.object(event_wait, "_sleep_with_stop_checks", return_value=True),
                ):
                    return await event_wait.wait_for_filesystem_event(
                        [watched],
                        timeout_seconds=1,
                        stop_check=lambda: True,
                    )

            stopped = asyncio.run(fallback_stop())
            self.assertEqual(stopped.reason, "stop")
            self.assertFalse(stopped.used_inotify)

            class FakeWaiter:
                def __init__(self, roots, **kwargs):
                    self.roots = tuple(roots)
                    self.kwargs = kwargs
                    self.closed = False

                def wait(self, *_args, **_kwargs):
                    return event_wait.FileEventWaitResult(
                        reason="filesystem_event",
                        elapsed_seconds=0.0,
                        paths=(str(watched / "a.json"),),
                        used_inotify=True,
                    )

                def close(self):
                    self.closed = True

            with patch.object(event_wait, "_InotifyWaiter", FakeWaiter):
                result = asyncio.run(
                    event_wait.wait_for_filesystem_event(
                        [watched],
                        timeout_seconds=1,
                        recursive=True,
                        max_dirs=4,
                        event_filter=lambda path: path.suffix == ".json",
                    )
                )
            self.assertEqual(result.reason, "filesystem_event")
            self.assertTrue(result.used_inotify)

            calls = {"n": 0}

            async def fast_sleep(_seconds: float) -> None:
                calls["n"] += 1

            checks = iter([False, RuntimeError("ignored"), True])

            def stop_check() -> bool:
                value = next(checks)
                if isinstance(value, Exception):
                    raise value
                return value

            with patch.object(event_wait.asyncio, "sleep", fast_sleep):
                self.assertTrue(
                    asyncio.run(
                        event_wait._sleep_with_stop_checks(
                            10,
                            stop_check=stop_check,
                            stop_check_interval_seconds=0.01,
                        )
                    )
                )
            self.assertGreaterEqual(calls["n"], 1)

    def test_inotify_waiter_read_event_parsing_is_filter_safe(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import event_wait

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            waiter = object.__new__(event_wait._InotifyWaiter)
            waiter.fd = 99
            waiter.recursive = True
            waiter.max_dirs = 4
            waiter._closed = False
            waiter._wd_to_path = {1: root}
            waiter._path_to_wd = {root.resolve(): 1}
            waiter._add_tree = lambda _path: None
            waiter.event_filter = lambda path: path.suffix == ".json"

            name = b"keep.json\0"
            event = (
                struct.pack(
                    "iIII",
                    1,
                    event_wait._IN_CREATE | event_wait._IN_ISDIR | event_wait._IN_IGNORED,
                    0,
                    len(name),
                )
                + name
            )
            with patch.object(event_wait.os, "read", side_effect=[event, BlockingIOError()]):
                paths = waiter._read_events()
            self.assertEqual(paths, (str(root / "keep.json"),))
            self.assertNotIn(1, waiter._wd_to_path)

            waiter._wd_to_path = {2: root}
            waiter._path_to_wd = {root.resolve(): 2}
            waiter.event_filter = lambda _path: (_ for _ in ()).throw(RuntimeError("filter"))
            event = struct.pack("iIII", 2, event_wait._IN_MODIFY, 0, 0)
            with patch.object(event_wait.os, "read", side_effect=[event, BlockingIOError()]):
                self.assertEqual(waiter._read_events(), ())

            with patch.object(event_wait.os, "read", side_effect=OSError(errno.EAGAIN, "again")):
                self.assertEqual(waiter._read_events(), ())

            waiter._closed = False
            with patch.object(event_wait.os, "close") as close:
                waiter.close()
                waiter.close()
            close.assert_called_once_with(99)

    def test_inotify_waiter_watch_lifecycle_edges_are_best_effort(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import event_wait

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = root / "child"
            child.mkdir()
            file_path = root / "file.txt"
            file_path.write_text("x", encoding="utf-8")

            class FakeLibC:
                def __init__(self, *, init_fd: int = 55, watch_returns: list[int] | None = None):
                    self.init_fd = init_fd
                    self.watch_returns = list(watch_returns or [])
                    self.watch_calls: list[bytes] = []

                def inotify_init1(self, _flags):
                    return self.init_fd

                def inotify_add_watch(self, _fd, encoded_path, _mask):
                    self.watch_calls.append(encoded_path)
                    if self.watch_returns:
                        return self.watch_returns.pop(0)
                    return len(self.watch_calls)

            with (
                patch.object(event_wait.ctypes, "CDLL", return_value=FakeLibC(init_fd=-1)),
                patch.object(event_wait.ctypes, "get_errno", return_value=errno.EPERM),
                self.assertRaises(OSError),
            ):
                event_wait._InotifyWaiter([root], recursive=False, max_dirs=4)

            fake_libc = FakeLibC(watch_returns=[10, -1])
            with patch.object(event_wait.ctypes, "CDLL", return_value=fake_libc):
                waiter = event_wait._InotifyWaiter([root], recursive=False, max_dirs=4)
            self.assertEqual(waiter.fd, 55)
            self.assertIn(10, waiter._wd_to_path)
            waiter._add_watch(root)
            waiter._add_watch(file_path)
            with patch.object(event_wait.ctypes, "get_errno", return_value=errno.EIO):
                waiter._add_watch(child)
            self.assertNotIn(child.resolve(), waiter._path_to_wd)

            waiter.max_dirs = len(waiter._wd_to_path)
            cap_dir = root / "cap"
            cap_dir.mkdir()
            waiter._add_watch(cap_dir)
            self.assertNotIn(cap_dir.resolve(), waiter._path_to_wd)

            with patch.object(Path, "resolve", side_effect=OSError("resolve failed")):
                waiter._add_watch(root)

            added: list[Path] = []
            tree_waiter = object.__new__(event_wait._InotifyWaiter)
            tree_waiter.recursive = False
            tree_waiter.max_dirs = 10
            tree_waiter._wd_to_path = {}
            tree_waiter._add_watch = lambda path: added.append(Path(path))
            tree_waiter._add_tree(file_path)
            self.assertEqual(added[-1], root.resolve())
            tree_waiter._add_tree(root / "missing" / "leaf")

            tree_waiter.recursive = True
            tree_waiter._wd_to_path = {}

            def add_and_mark(path):
                added.append(Path(path))
                tree_waiter._wd_to_path[len(added)] = Path(path)

            tree_waiter._add_watch = add_and_mark
            tree_waiter.max_dirs = 10
            tree_waiter._add_tree(root)
            self.assertIn(child.resolve(), {p.resolve() for p in added if p.exists()})
            tree_waiter.max_dirs = len(tree_waiter._wd_to_path)
            tree_waiter._add_tree(root)
            with patch.object(Path, "rglob", side_effect=OSError("walk failed")):
                tree_waiter.max_dirs = 10
                tree_waiter._wd_to_path = {}
                tree_waiter._add_tree(root)
            with patch.object(Path, "resolve", side_effect=OSError("resolve failed")):
                tree_waiter._add_tree(root)

    def test_inotify_read_and_wait_edges_preserve_event_driven_contract(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import event_wait

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            waiter = object.__new__(event_wait._InotifyWaiter)
            waiter.fd = 101
            waiter.recursive = False
            waiter.max_dirs = 4
            waiter._closed = False
            waiter._wd_to_path = {1: root}
            waiter._path_to_wd = {root.resolve(): 1}
            waiter.event_filter = None
            waiter._add_tree = lambda _path: None

            with (
                patch.object(event_wait.os, "read", side_effect=OSError(errno.EINVAL, "bad")),
                self.assertRaises(OSError),
            ):
                waiter._read_events()
            with patch.object(event_wait.os, "read", return_value=b""):
                self.assertEqual(waiter._read_events(), ())

            unknown_event = struct.pack("iIII", 999, event_wait._IN_MODIFY, 0, 0)
            with patch.object(
                event_wait.os, "read", side_effect=[unknown_event, BlockingIOError()]
            ):
                self.assertEqual(waiter._read_events(), ())

            waiter._wd_to_path = {2: root}
            waiter._path_to_wd = {root.resolve(): 2}
            waiter.event_filter = lambda _path: False
            filtered_event = struct.pack("iIII", 2, event_wait._IN_MODIFY, 0, 0)
            with patch.object(
                event_wait.os, "read", side_effect=[filtered_event, BlockingIOError()]
            ):
                self.assertEqual(waiter._read_events(), ())

            waiter.event_filter = None
            ignored_event = struct.pack("iIII", 2, event_wait._IN_IGNORED, 0, 0)
            with (
                patch.object(event_wait.os, "read", side_effect=[ignored_event, BlockingIOError()]),
                patch.object(Path, "resolve", side_effect=OSError("gone")),
            ):
                self.assertEqual(waiter._read_events(), (str(root),))
            self.assertNotIn(2, waiter._wd_to_path)

            wait_target = SimpleNamespace(
                fd=7,
                _read_events=lambda: (str(root / "changed.json"),),
            )
            wait_target.wait = event_wait._InotifyWaiter.wait.__get__(wait_target)
            with (
                patch.object(event_wait.select, "select", return_value=([7], [], [])),
                patch.object(event_wait.time, "monotonic", side_effect=[0.0, 0.0, 0.1]),
            ):
                result = wait_target.wait(1, stop_check=None, stop_check_interval_seconds=1)
            self.assertEqual(result.reason, "filesystem_event")
            self.assertTrue(result.used_inotify)

            wait_target._read_events = lambda: ()
            with patch.object(event_wait.time, "monotonic", side_effect=[0.0, 0.2]):
                result = wait_target.wait(
                    10, stop_check=lambda: True, stop_check_interval_seconds=1
                )
            self.assertEqual(result.reason, "stop")

            with patch.object(event_wait.time, "monotonic", side_effect=[0.0, 2.0, 2.5]):
                result = wait_target.wait(1, stop_check=None, stop_check_interval_seconds=1)
            self.assertEqual(result.reason, "timeout")

            stop_checks = iter([False, True])
            with (
                patch.object(event_wait.select, "select", return_value=([], [], [])),
                patch.object(event_wait.time, "monotonic", return_value=0.0),
            ):
                result = wait_target.wait(
                    10,
                    stop_check=lambda: next(stop_checks),
                    stop_check_interval_seconds=1,
                )
            self.assertEqual(result.reason, "stop")

            checks = iter([RuntimeError("transient"), False])

            def stop_check():
                value = next(checks)
                if isinstance(value, Exception):
                    raise value
                return bool(value)

            wait_target._read_events = lambda: (str(root / "after_error.json"),)
            with (
                patch.object(event_wait.select, "select", return_value=([7], [], [])),
                patch.object(event_wait.time, "monotonic", side_effect=[0.0, 0.0, 0.1]),
            ):
                result = wait_target.wait(1, stop_check=stop_check, stop_check_interval_seconds=1)
            self.assertEqual(result.reason, "filesystem_event")


if __name__ == "__main__":
    unittest.main()
