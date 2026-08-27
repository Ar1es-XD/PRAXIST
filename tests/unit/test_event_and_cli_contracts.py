from __future__ import annotations

import asyncio
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class EventWaitContractsTest(unittest.TestCase):
    def test_event_wait_roots_fallback_stop_and_fake_inotify_paths(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import event_wait

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            watched = root / "watched"
            watched.mkdir()
            file_path = watched / "file.txt"
            file_path.write_text("x", encoding="utf-8")
            roots = event_wait._candidate_watch_roots([file_path, watched, watched / "missing"])
            self.assertEqual(roots[0], watched.resolve())
            self.assertEqual(len(roots), 1)

            closed: list[bool] = []

            class FakeWaiter:
                def __init__(self, roots, **kwargs):
                    self.roots = list(roots)
                    self.kwargs = kwargs

                def wait(self, timeout_seconds, *, stop_check, stop_check_interval_seconds):
                    return event_wait.FileEventWaitResult(
                        reason="filesystem_event",
                        elapsed_seconds=0.1,
                        paths=(str(file_path),),
                        used_inotify=True,
                    )

                def close(self):
                    closed.append(True)

            with patch.object(event_wait, "_InotifyWaiter", FakeWaiter):
                result = asyncio.run(
                    event_wait.wait_for_filesystem_event(
                        [watched],
                        timeout_seconds=1,
                        recursive=True,
                        event_filter=lambda p: p.suffix == ".txt",
                    )
                )
            self.assertEqual(result.reason, "filesystem_event")
            self.assertTrue(closed)

            with patch.object(event_wait, "_InotifyWaiter", side_effect=OSError("no inotify")):
                stopped = asyncio.run(
                    event_wait.wait_for_filesystem_event(
                        [watched],
                        timeout_seconds=1,
                        fallback_interval_seconds=1,
                        stop_check=lambda: True,
                    )
                )
            self.assertEqual(stopped.reason, "stop")
            self.assertFalse(stopped.used_inotify)

            with patch.object(event_wait, "_InotifyWaiter", side_effect=OSError("no inotify")):
                fallback = asyncio.run(
                    event_wait.wait_for_filesystem_event(
                        [watched],
                        timeout_seconds=0,
                        fallback_interval_seconds=0,
                        stop_check=lambda: False,
                    )
                )
            self.assertEqual(fallback.reason, "fallback_elapsed")

            no_paths = asyncio.run(
                event_wait.wait_for_filesystem_event(
                    [root / "missing" / "file.txt"],
                    timeout_seconds=0,
                    fallback_interval_seconds=0,
                )
            )
            self.assertEqual(no_paths.reason, "no_watch_paths")

        calls = {"n": 0}

        def noisy_stop() -> bool:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")
            return True

        self.assertTrue(
            asyncio.run(
                event_wait._sleep_with_stop_checks(
                    2,
                    stop_check=noisy_stop,
                    stop_check_interval_seconds=1,
                )
            )
        )


class CliParityAndReplayContractsTest(unittest.TestCase):
    def test_cli_replay_parity_and_error_paths_are_operator_visible(self) -> None:
        from praxist import run as cli

        out = io.StringIO()
        with (
            patch(
                "praxist.core.replay.inspect_run",
                return_value={"success": True, "mode": "inspect"},
            ) as inspect_run,
            patch.object(
                __import__("sys"),
                "argv",
                ["praxist", "replay", "/tmp/run", "--mode", "inspect"],
            ),
            contextlib.redirect_stdout(out),
        ):
            cli.main()
        self.assertIn('"mode": "inspect"', out.getvalue())
        inspect_run.assert_called_once()

        with (
            patch(
                "praxist.core.replay.verify_run",
                return_value={"success": False, "errors": ["bad"]},
            ),
            patch.object(
                __import__("sys"),
                "argv",
                ["praxist", "replay", "/tmp/run", "--mode", "verify", "--strict-tail", "--locked"],
            ),
            self.assertRaises(SystemExit) as cm,
        ):
            cli.main()
        self.assertEqual(cm.exception.code, 1)

        out = io.StringIO()
        with (
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend.parity.verify_research_loop_parity",
                return_value={"success": True},
            ) as parity,
            patch.object(
                __import__("sys"),
                "argv",
                [
                    "praxist",
                    "parity",
                    "/tmp/run",
                    "--deliverables-dir",
                    "/tmp/deliverables",
                    "--strict",
                    "--write-report",
                ],
            ),
            contextlib.redirect_stdout(out),
        ):
            cli.main()
        self.assertIn('"success": true', out.getvalue())
        parity.assert_called_once()

        with (
            patch.object(__import__("sys"), "argv", ["praxist"]),
            self.assertRaises(SystemExit) as cm,
        ):
            cli.main()
        self.assertEqual(cm.exception.code, 1)

        with (
            patch.object(__import__("sys"), "argv", ["praxist", "run", "--task", "task:x"]),
            self.assertRaises(SystemExit) as cm,
        ):
            cli.main()
        self.assertEqual(cm.exception.code, 2)

        with (
            patch.object(__import__("sys"), "argv", ["praxist", "run", "--task-spec", "spec.yaml"]),
            self.assertRaises(SystemExit) as cm,
        ):
            cli.main()
        self.assertEqual(cm.exception.code, 2)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs"
            task_project = SimpleNamespace(
                path=Path(tmp) / "task",
                descriptor={"runtime_outputs": {"root": "experiments"}},
            )
            selected = cli._default_run_dir_for_task_project(
                task_project,
                Path(tmp),
                "task:demo",
            )
            self.assertEqual(selected.parent, task_project.path / "experiments")
            task_project.descriptor = {}
            selected = cli._default_run_dir_for_task_project(task_project, Path(tmp), "task:demo")
            self.assertEqual(selected.parent, task_project.path / "experiments")
            fake_default = cli._default_run_dir_for_fake_fixture("task:fake_panel")
            self.assertIn("praxist", fake_default.parts)
            self.assertIn("fake_runs", fake_default.parts)
            with self.assertRaises(ValueError):
                cli._ensure_run_dir_not_in_source_checkout(Path.cwd() / "bad_run")

            with (
                patch(
                    "praxist.testing.fake_workflow_fixture.run_fake_workflow_fixture",
                    return_value={"ok": True},
                ),
                patch.object(
                    __import__("sys"),
                    "argv",
                    ["praxist", "run", "--fake", "--workspace", tmp, "--run-dir", str(run_dir)],
                ),
                contextlib.redirect_stdout(io.StringIO()) as fake_out,
            ):
                cli.main()
            self.assertIn('"ok": true', fake_out.getvalue())


if __name__ == "__main__":
    unittest.main()
