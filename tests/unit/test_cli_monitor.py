"""Tests for the direct ``praxist --monitor`` read-only TUI."""

from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


def _row(**overrides: object):
    from praxist.cli.status import SOURCE_REGISTRY, StatusRow

    base = {
        "pid": 1234,
        "ppid": 1,
        "etime": "00:10",
        "command": "python -m praxist.run run --task-path /task --run-dir /run",
        "run_dir": "/run",
        "source": SOURCE_REGISTRY,
        "state": "running",
        "run_id": "run_demo",
        "task_path": "/task",
        "model": "deepseek-v4-pro[1m]",
        "model_provider_ref": "model_provider:deepseek_alias",
        "generation": 2,
        "findings_total": 5,
        "updated_at": "2026-07-07T00:00:00+00:00",
        "peer_health_summary": {"red": 1, "yellow": 1, "green": 2},
        "peers": [
            {
                "peer_id": "gen2_peer0",
                "health": "green",
                "research_state": "evaluating",
                "active_variant": "variant_a",
                "best_metric_value": 1.25,
                "baseline_status": "above",
                "health_reason": "recent progress",
                "last_updated_utc": "2026-07-07T00:00:00+00:00",
            }
        ],
    }
    base.update(overrides)
    return StatusRow(**base)


class MonitorRenderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._term_patch = patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=False)
        self._term_patch.start()
        self.addCleanup(self._term_patch.stop)

    def test_snapshot_and_text_renderer_include_run_peer_phase_and_logs(self) -> None:
        from praxist.cli import monitor

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            logs = run_dir / "logs"
            logs.mkdir(parents=True)
            (run_dir / "gen_2").mkdir()
            (logs / "launcher.nohup.log").write_text("line1\nline2\n", encoding="utf-8")
            (run_dir / "orchestrator_status.json").write_text(
                json.dumps(
                    {
                        "current_generation": 2,
                        "generations_completed": 1,
                        "cohort_size": 4,
                        "findings_total": 8,
                        "frontier_candidates": 3,
                        "gems_count": 2,
                        "exit_condition": "in_progress",
                        "updated_at": "2026-07-07T00:00:01+00:00",
                        "last_peer_mix": {
                            "mature_constructive_ratio": 0.75,
                            "target_constructive_ratio": 0.8,
                        },
                        "last_stop_audit": {
                            "trigger_reason": "mature_quorum",
                            "mature_result_peers": 3,
                            "required_mature_result_peers": 3,
                        },
                        "best_mature_result": {
                            "variant_name": "variant_complete",
                            "metric_name": "score",
                            "metric_value": 1.5,
                            "evidence_stage": "complete",
                            "baseline_relation": "above_baseline",
                        },
                        "best_validation_signal": {
                            "variant_name": "variant_preview",
                            "metric_name": "score",
                            "metric_value": 2.0,
                            "evidence_stage": "preview",
                            "baseline_relation": "above_baseline",
                            "validation_reason": "not_scored_complete",
                        },
                    }
                ),
                encoding="utf-8",
            )
            row = _row(run_dir=str(run_dir))
            with (
                patch("praxist.cli.monitor.status.collect_status_rows", return_value=[row]),
                patch(
                    "praxist.cli.monitor.collect_hardware_snapshot",
                    return_value=monitor.HardwareSnapshot(
                        loadavg="1.00 2.00 3.00",
                        memory="1.0 GiB / 4.0 GiB",
                        gpus=["GPU 0: util 10%, mem 100/1000 MiB"],
                    ),
                ),
            ):
                snapshot = monitor.collect_monitor_snapshot(
                    target=monitor.MonitorTarget(run_id="run_demo"),
                    log_lines=5,
                )

        output = monitor.TextMonitorRenderer().render(snapshot)
        self.assertIn("Praxist Monitor", output)
        self.assertIn("run_demo", output)
        self.assertIn("gen2:running", output)
        self.assertIn("gen2_peer0", output)
        self.assertIn("variant_a", output)
        self.assertIn("last_peer_mix: constructive=0.75 target=0.8", output)
        self.assertIn("last_stop: mature_quorum mature=3/3", output)
        self.assertIn("best_mature_result: variant_complete", output)
        self.assertIn("best_validation_signal: variant_preview", output)
        self.assertIn("health_reason[gen2_peer0]: recent progress", output)
        self.assertIn("line2", output)
        self.assertIn("GPU 0", output)

    def test_snapshot_prefers_final_orchestrator_status(self) -> None:
        from praxist.cli import monitor

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "orchestrator_status.json").write_text(
                json.dumps(
                    {
                        "current_generation": 1,
                        "generations_completed": 1,
                        "exit_condition": "in_progress",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "orchestrator_status.final.json").write_text(
                json.dumps(
                    {
                        "current_generation": 3,
                        "generations_completed": 4,
                        "exit_condition": "completed",
                    }
                ),
                encoding="utf-8",
            )
            row = _row(run_dir=str(run_dir), state="completed")
            with (
                patch("praxist.cli.monitor.status.collect_status_rows", return_value=[row]),
                patch(
                    "praxist.cli.monitor.collect_hardware_snapshot",
                    return_value=monitor.HardwareSnapshot(),
                ),
            ):
                snapshot = monitor.collect_monitor_snapshot(
                    target=monitor.MonitorTarget(run_id="run_demo"),
                    log_lines=5,
                )

        self.assertEqual(snapshot.orchestrator_status["generations_completed"], 4)
        self.assertEqual(snapshot.phase, "finished:completed")

    def test_select_status_row_refuses_ambiguous_active_without_latest(self) -> None:
        from praxist.cli import monitor

        old = _row(pid=1, run_id="old", updated_at="2026-07-06T00:00:00+00:00")
        new = _row(pid=2, run_id="new", updated_at="2026-07-07T00:00:00+00:00")
        selected, warnings = monitor.select_status_row([old, new], monitor.MonitorTarget())
        self.assertIsNone(selected)
        self.assertTrue(any("pass --latest" in warning for warning in warnings))

        selected_latest, latest_warnings = monitor.select_status_row(
            [old, new],
            monitor.MonitorTarget(latest=True),
        )
        self.assertIs(selected_latest, new)
        self.assertEqual(latest_warnings, [])

    def test_cli_once_renders_one_frame(self) -> None:
        from praxist.cli import main, monitor

        row = _row()
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with (
                redirect_stdout(stdout),
                redirect_stderr(stderr),
                patch("praxist.cli.monitor.status.collect_status_rows", return_value=[row]),
                patch(
                    "praxist.cli.monitor.collect_hardware_snapshot",
                    return_value=monitor.HardwareSnapshot(loadavg="-", memory="-"),
                ),
            ):
                main(["--monitor", "--once", "--run-id", "run_demo", "--no-clear"])
        except SystemExit as exc:
            self.assertEqual(int(exc.code or 0), 0)
        self.assertIn("Praxist Monitor", stdout.getvalue())
        self.assertIn("run_demo", stdout.getvalue())

    def test_legacy_monitor_subcommand_uses_the_same_foreground_handler(self) -> None:
        from praxist.cli import main, monitor

        row = _row()
        stdout = io.StringIO()
        try:
            with (
                redirect_stdout(stdout),
                patch("praxist.cli.monitor.status.collect_status_rows", return_value=[row]),
                patch(
                    "praxist.cli.monitor.collect_hardware_snapshot",
                    return_value=monitor.HardwareSnapshot(loadavg="-", memory="-"),
                ),
            ):
                main(["monitor", "--once", "--run-id", "run_demo", "--no-clear"])
        except SystemExit as exc:
            self.assertEqual(int(exc.code or 0), 0)
        self.assertIn("run_demo", stdout.getvalue())

    def test_plain_mode_is_append_friendly_without_extra_flag(self) -> None:
        from praxist.cli import main, monitor

        try:
            with (
                patch(
                    "praxist.cli.monitor.resolve_monitor_target",
                    return_value=monitor.MonitorTarget(run_id="run_demo"),
                ),
                patch("praxist.cli.monitor.run_foreground_monitor", return_value=0) as run,
            ):
                main(["--monitor", "--run-id", "run_demo", "--plain"])
        except SystemExit as exc:
            self.assertEqual(int(exc.code or 0), 0)
        self.assertFalse(run.call_args.kwargs["clear"])
        self.assertEqual(run.call_args.kwargs["interval"], 1.0)

    def test_non_fullscreen_defaults_to_plain_cadence_without_terminal_controls(self) -> None:
        from praxist.cli import main, monitor

        class TtyBuffer(io.StringIO):
            def isatty(self) -> bool:
                return True

        cases = [
            (
                ["--monitor", "--run-id", "run_demo", "--follow"],
                io.StringIO(),
                {"TERM": "xterm-256color"},
            ),
            (["--monitor", "--run-id", "run_demo"], TtyBuffer(), {"TERM": "dumb"}),
            (
                ["--monitor", "--run-id", "run_demo", "--no-clear"],
                TtyBuffer(),
                {"TERM": "xterm-256color"},
            ),
        ]
        for argv, stdout, environment in cases:
            with self.subTest(argv=argv, environment=environment):
                with (
                    redirect_stdout(stdout),
                    patch.dict(os.environ, environment, clear=False),
                    patch(
                        "praxist.cli.monitor.resolve_monitor_target",
                        return_value=monitor.MonitorTarget(run_id="run_demo"),
                    ),
                    patch("praxist.cli.monitor.run_foreground_monitor", return_value=0) as run,
                    self.assertRaises(SystemExit) as raised,
                ):
                    main(argv)
                self.assertEqual(int(raised.exception.code or 0), 0)
                self.assertFalse(run.call_args.kwargs["clear"])
                self.assertEqual(run.call_args.kwargs["interval"], 1.0)

    def test_cli_rejects_non_finite_or_non_positive_refresh_interval(self) -> None:
        from praxist.cli import main, monitor

        for interval in ("nan", "0", "-1"):
            stderr = io.StringIO()
            with (
                redirect_stderr(stderr),
                patch(
                    "praxist.cli.monitor.resolve_monitor_target",
                    return_value=monitor.MonitorTarget(run_id="run_demo"),
                ),
                self.assertRaises(SystemExit) as raised,
            ):
                main(["--monitor", "--once", "--interval", interval])
            self.assertEqual(int(raised.exception.code or 0), 2)
            self.assertIn("--interval must be a finite positive number", stderr.getvalue())

    def test_tui_renderer_fits_terminal_frame_and_sections(self) -> None:
        from praxist.cli import monitor

        snapshot = monitor.MonitorSnapshot(
            rows=[_row()],
            selected=_row(),
            target=monitor.MonitorTarget(run_id="run_demo"),
            generated_at="2026-07-07T00:00:00+00:00",
            orchestrator_status={
                "current_generation": 2,
                "generations_completed": 1,
                "cohort_size": 4,
                "findings_total": 8,
                "frontier_candidates": 3,
                "gems_count": 2,
                "best_mature_result": {
                    "variant_name": "variant_complete",
                    "metric_name": "score",
                    "metric_value": 1.5,
                },
                "best_validation_signal": {
                    "variant_name": "variant_preview",
                    "metric_name": "score",
                    "metric_value": 2.0,
                    "validation_reason": "preview_only",
                },
            },
            phase="gen2:running",
            recent_logs=["peer launched", "evaluation finished"],
            hardware=monitor.HardwareSnapshot(
                loadavg="1.00 2.00 3.00",
                memory="1.0 GiB / 4.0 GiB",
                gpus=["GPU 0: util 10%, mem 100/1000 MiB"],
            ),
        )
        frame = monitor.TuiMonitorRenderer(color=False).render(snapshot, width=128, height=32)
        lines = frame.splitlines()
        self.assertEqual(len(lines), 32)
        self.assertFalse(frame.endswith("\n"))
        self.assertTrue(all(len(line) == 128 for line in lines))
        self.assertIn("PRAXIST  RESEARCH MONITOR", frame)
        self.assertIn("Selected Run", frame)
        self.assertIn("Peers", frame)
        self.assertIn("Recent Logs", frame)
        self.assertIn("Hardware / Warnings", frame)
        self.assertIn("Live log stream", frame)
        self.assertIn("best_mature: variant_complete", frame)
        self.assertIn("validation_signal: variant_preview", frame)
        self.assertIn("loadavg: 1.00 2.00 3.00", frame)
        self.assertIn("evaluation finished", frame)

    def test_tui_renderer_uses_plain_title_and_fixed_log_viewport(self) -> None:
        from praxist.cli import monitor

        snapshot = monitor.MonitorSnapshot(
            rows=[_row()],
            selected=_row(),
            target=monitor.MonitorTarget(run_id="run_demo"),
            generated_at="2026-07-07T00:00:00+00:00",
            phase="gen2:running",
            recent_logs=[f"log line {idx}" for idx in range(20)],
            hardware=monitor.HardwareSnapshot(loadavg="1.00 2.00 3.00", memory="1/4 GiB"),
        )
        frame = monitor.TuiMonitorRenderer(color=True).render(snapshot, width=100, height=25)
        plain = monitor._strip_ansi(frame)
        self.assertEqual(len(plain.splitlines()), 25)
        self.assertNotIn("\033[1;37;44m", frame)
        self.assertNotIn("\033[37;44m", frame)
        self.assertNotIn("\033[30;47m", frame)
        self.assertIn("PRAXIST  RESEARCH MONITOR", plain)
        self.assertIn("Recent Logs", plain)
        self.assertIn("Live log stream", plain)
        self.assertIn("log line 19", plain)
        self.assertNotIn("log line 0", plain)

        start, end = monitor.TuiMonitorRenderer(color=False).stream_scroll_region(
            width=100,
            height=25,
        )
        self.assertLess(start, end)
        self.assertEqual(end, 24)

    def test_tui_log_stream_soft_wraps_instead_of_truncating(self) -> None:
        from praxist.cli import monitor

        long_token = "payload=" + "x" * 90 + " important_tail=true"
        snapshot = monitor.MonitorSnapshot(
            rows=[_row()],
            selected=_row(),
            target=monitor.MonitorTarget(run_id="run_demo"),
            generated_at="2026-07-07T00:00:00+00:00",
            recent_logs=[long_token],
        )
        frame = monitor.TuiMonitorRenderer(color=False).render(snapshot, width=64, height=24)
        self.assertIn("important_tail=true", frame)

    def test_missing_selected_run_directory_is_explicitly_reported(self) -> None:
        from praxist.cli import monitor

        row = _row(run_dir="/definitely/missing/praxist-run")
        with (
            patch("praxist.cli.monitor.status.collect_status_rows", return_value=[row]),
            patch(
                "praxist.cli.monitor.collect_hardware_snapshot",
                return_value=monitor.HardwareSnapshot(),
            ),
        ):
            snapshot = monitor.collect_monitor_snapshot(
                target=monitor.MonitorTarget(run_id="run_demo"),
                log_lines=5,
            )
        self.assertEqual(snapshot.phase, "missing-run-directory")
        self.assertTrue(any("run directory is missing" in item for item in snapshot.warnings))

    def test_tui_renderer_handles_small_window_and_wide_text(self) -> None:
        from praxist.cli import monitor

        row = _row(
            run_id="实验_run_with_a_very_long_identifier",
            task_path="/tmp/研究项目/with/a/very/long/path",
            peers=[
                {
                    "peer_id": "gen2_peer_宽字符",
                    "health": "green",
                    "research_state": "评估中-with-long-state",
                    "active_variant": "变体_with_a_very_long_identifier",
                    "best_metric_value": 1.25,
                }
            ],
        )
        snapshot = monitor.MonitorSnapshot(
            rows=[row],
            selected=row,
            target=monitor.MonitorTarget(run_id=row.run_id),
            generated_at="2026-07-07T00:00:00+00:00",
            orchestrator_status={"current_generation": 2, "findings_total": 8},
            phase="gen2:running",
            hardware=monitor.HardwareSnapshot(loadavg="1.00 2.00 3.00", memory="1/4 GiB"),
            warnings=["窗口很小但不应溢出"],
        )
        frame = monitor.TuiMonitorRenderer(color=False).render(snapshot, width=32, height=8)
        lines = frame.splitlines()
        self.assertEqual(len(lines), 8)
        self.assertTrue(all(monitor._visible_len(line) <= 32 for line in lines))
        self.assertIn("Praxist", frame)
        self.assertIn("gen:", frame)

        tiny = monitor.TuiMonitorRenderer(color=False).render(snapshot, width=10, height=3)
        tiny_lines = tiny.splitlines()
        self.assertEqual(len(tiny_lines), 3)
        self.assertTrue(all(monitor._visible_len(line) <= 10 for line in tiny_lines))
        self.assertEqual(
            monitor.TuiMonitorRenderer(color=False).stream_scroll_region(width=10, height=3),
            (3, 3),
        )

    def test_renderers_remove_untrusted_terminal_control_sequences(self) -> None:
        from praxist.cli import monitor

        control_text = "unsafe\x1b]52;c;clipboard\x07\x1b[2J\u202estill-visible"
        row = _row(run_id=control_text)
        snapshot = monitor.MonitorSnapshot(
            rows=[row],
            selected=row,
            target=monitor.MonitorTarget(run_id=control_text),
            generated_at="2026-07-07T00:00:00+00:00",
            recent_logs=[control_text],
            warnings=[control_text],
        )

        tui = monitor.TuiMonitorRenderer(color=False).render(snapshot, width=100, height=25)
        plain = monitor.TextMonitorRenderer().render(snapshot)
        for rendered in (tui, plain):
            self.assertNotIn("\x1b", rendered)
            self.assertNotIn("\x07", rendered)
            self.assertNotIn("\u202e", rendered)
            self.assertIn("still-visible", rendered)

        wide = monitor.TuiMonitorRenderer(color=False).render(
            snapshot,
            width=128,
            height=32,
        )
        self.assertNotIn("\x1b", wide)
        self.assertNotIn("\x07", wide)
        self.assertNotIn("\u202e", wide)
        self.assertIn("still-visible", wide)

    def test_foreground_monitor_uses_tui_for_interactive_terminal(self) -> None:
        from praxist.cli import monitor

        class TtyBuffer(io.StringIO):
            def isatty(self) -> bool:
                return True

        snapshot = monitor.MonitorSnapshot(
            rows=[_row()],
            selected=_row(),
            target=monitor.MonitorTarget(run_id="run_demo"),
            generated_at="2026-07-07T00:00:00+00:00",
            phase="gen2:running",
            hardware=monitor.HardwareSnapshot(loadavg="-", memory="-"),
        )
        stdout, stderr = TtyBuffer(), io.StringIO()
        with (
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            patch(
                "praxist.cli.monitor.collect_monitor_snapshot",
                return_value=snapshot,
            ) as collect,
            patch(
                "praxist.cli.monitor.shutil.get_terminal_size",
                return_value=os.terminal_size((90, 20)),
            ),
            patch("praxist.cli.monitor.time.sleep", side_effect=KeyboardInterrupt),
        ):
            code = monitor.run_foreground_monitor(
                target=monitor.MonitorTarget(run_id="run_demo"),
                interval=1,
                once=False,
                clear=True,
                plain=False,
                log_lines=5,
                peer_limit=4,
            )
        output = stdout.getvalue()
        self.assertEqual(code, 130)
        self.assertIn("\033[?1049h", output)
        self.assertIn("\033[?7l", output)
        self.assertIn("\033[14;19r", output)
        self.assertIn("\033[?7h", output)
        self.assertIn("\033[?1049l", output)
        self.assertIn("PRAXIST  RESEARCH MONITOR", output)
        self.assertEqual(collect.call_count, 1)
        self.assertEqual(stderr.getvalue(), "")

    def test_foreground_monitor_uses_ascii_logo_when_terminal_cannot_encode_blocks(self) -> None:
        from praxist.cli import monitor

        class AsciiTty(io.StringIO):
            encoding = "ascii"

            def isatty(self) -> bool:
                return True

            def write(self, value: str) -> int:
                value.encode(self.encoding)
                return super().write(value)

        unicode_row = _row(run_id="运行_demo")
        snapshot = monitor.MonitorSnapshot(
            rows=[unicode_row],
            selected=unicode_row,
            target=monitor.MonitorTarget(run_id="run_demo"),
            generated_at="2026-07-07T00:00:00+00:00",
            phase="gen2:运行",
            recent_logs=["实验 completed"],
        )
        stdout = AsciiTty()
        with (
            redirect_stdout(stdout),
            patch("praxist.cli.monitor.collect_monitor_snapshot", return_value=snapshot),
            patch(
                "praxist.cli.monitor.shutil.get_terminal_size",
                return_value=os.terminal_size((120, 32)),
            ),
            patch("praxist.cli.monitor.time.sleep", side_effect=KeyboardInterrupt),
        ):
            code = monitor.run_foreground_monitor(
                target=monitor.MonitorTarget(run_id="run_demo"),
                interval=0.2,
                once=False,
                clear=True,
                plain=False,
                log_lines=5,
                peer_limit=4,
            )

        self.assertEqual(code, 130)
        self.assertIn("PRAXIST  RESEARCH MONITOR", stdout.getvalue())
        self.assertNotIn("█", stdout.getvalue())

    def test_foreground_monitor_restores_terminal_when_entry_is_interrupted(self) -> None:
        from praxist.cli import monitor

        class InterruptedTty(io.StringIO):
            def __init__(self) -> None:
                super().__init__()
                self.write_count = 0

            def isatty(self) -> bool:
                return True

            def write(self, value: str) -> int:
                self.write_count += 1
                if self.write_count == 1:
                    raise KeyboardInterrupt
                return super().write(value)

        stdout = InterruptedTty()
        with redirect_stdout(stdout):
            code = monitor.run_foreground_monitor(
                target=monitor.MonitorTarget(run_id="run_demo"),
                interval=1,
                once=False,
                clear=True,
                plain=False,
                log_lines=1,
                peer_limit=1,
            )
        self.assertEqual(code, 130)
        self.assertIn("\033[?25h", stdout.getvalue())
        self.assertIn("\033[?1049l", stdout.getvalue())

    def test_foreground_monitor_restores_terminal_on_termination_signal(self) -> None:
        from praxist.cli import monitor

        class TtyBuffer(io.StringIO):
            def isatty(self) -> bool:
                return True

        previous = object()
        installed: dict[int, object] = {}
        restored: list[int] = []

        def fake_signal(signum: int, handler: object) -> object:
            if handler is previous:
                restored.append(signum)
            else:
                installed[signum] = handler
            return previous

        def interrupt_with_term(_seconds: float) -> None:
            installed[monitor.signal.SIGTERM](monitor.signal.SIGTERM, None)  # type: ignore[operator]

        snapshot = monitor.MonitorSnapshot(
            rows=[_row()],
            selected=_row(),
            target=monitor.MonitorTarget(run_id="run_demo"),
            generated_at="2026-07-07T00:00:00+00:00",
        )
        stdout, stderr = TtyBuffer(), io.StringIO()
        with (
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            patch(
                "praxist.cli.monitor.collect_monitor_snapshot",
                return_value=snapshot,
            ),
            patch(
                "praxist.cli.monitor.shutil.get_terminal_size",
                return_value=os.terminal_size((90, 20)),
            ),
            patch("praxist.cli.monitor.signal.getsignal", return_value=previous),
            patch("praxist.cli.monitor.signal.signal", side_effect=fake_signal),
            patch("praxist.cli.monitor.time.sleep", side_effect=interrupt_with_term),
        ):
            code = monitor.run_foreground_monitor(
                target=monitor.MonitorTarget(run_id="run_demo"),
                interval=1,
                once=False,
                clear=True,
                plain=False,
                log_lines=1,
                peer_limit=1,
            )

        self.assertEqual(code, 128 + monitor.signal.SIGTERM)
        self.assertIn("\033[?25h", stdout.getvalue())
        self.assertIn("\033[?1049l", stdout.getvalue())
        self.assertCountEqual(
            restored,
            [
                getattr(monitor.signal, name)
                for name in ("SIGHUP", "SIGTERM", "SIGTSTP", "SIGQUIT")
                if hasattr(monitor.signal, name)
            ],
        )

    def test_foreground_monitor_restores_terminal_while_suspended_and_reenters(self) -> None:
        from praxist.cli import monitor

        if not hasattr(monitor.signal, "SIGTSTP"):
            self.skipTest("terminal job control is unavailable")

        class TtyBuffer(io.StringIO):
            def isatty(self) -> bool:
                return True

        previous = object()
        installed: dict[int, object] = {}

        def fake_signal(signum: int, handler: object) -> object:
            installed[signum] = handler
            return previous

        def suspend_then_interrupt(_seconds: float) -> None:
            handler = installed[monitor.signal.SIGTSTP]
            self.assertTrue(callable(handler))
            handler(monitor.signal.SIGTSTP, None)  # type: ignore[operator]
            raise KeyboardInterrupt

        snapshot = monitor.MonitorSnapshot(
            rows=[_row()],
            selected=_row(),
            target=monitor.MonitorTarget(run_id="run_demo"),
            generated_at="2026-07-07T00:00:00+00:00",
        )
        stdout = TtyBuffer()
        with (
            redirect_stdout(stdout),
            patch("praxist.cli.monitor.collect_monitor_snapshot", return_value=snapshot),
            patch(
                "praxist.cli.monitor.shutil.get_terminal_size",
                return_value=os.terminal_size((90, 20)),
            ),
            patch("praxist.cli.monitor.signal.getsignal", return_value=previous),
            patch("praxist.cli.monitor.signal.signal", side_effect=fake_signal),
            patch("praxist.cli.monitor.os.kill") as kill,
            patch("praxist.cli.monitor.time.sleep", side_effect=suspend_then_interrupt),
        ):
            code = monitor.run_foreground_monitor(
                target=monitor.MonitorTarget(run_id="run_demo"),
                interval=1,
                once=False,
                clear=True,
                plain=False,
                log_lines=1,
                peer_limit=1,
            )

        self.assertEqual(code, 130)
        kill.assert_called_once_with(os.getpid(), monitor.signal.SIGTSTP)
        self.assertEqual(stdout.getvalue().count("\033[?1049h"), 2)
        self.assertEqual(stdout.getvalue().count("\033[?1049l"), 2)

    def test_foreground_monitor_tolerates_non_main_thread_signal_limits(self) -> None:
        from praxist.cli import monitor

        class TtyBuffer(io.StringIO):
            def isatty(self) -> bool:
                return True

        snapshot = monitor.MonitorSnapshot(
            rows=[_row()],
            selected=_row(),
            target=monitor.MonitorTarget(run_id="run_demo"),
            generated_at="2026-07-07T00:00:00+00:00",
        )
        stdout, stderr = TtyBuffer(), io.StringIO()
        with (
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            patch("praxist.cli.monitor.collect_monitor_snapshot", return_value=snapshot),
            patch(
                "praxist.cli.monitor.shutil.get_terminal_size",
                return_value=os.terminal_size((90, 20)),
            ),
            patch(
                "praxist.cli.monitor.signal.signal",
                side_effect=ValueError("signal only works in main thread"),
            ),
            patch("praxist.cli.monitor.time.sleep", side_effect=KeyboardInterrupt),
        ):
            code = monitor.run_foreground_monitor(
                target=monitor.MonitorTarget(run_id="run_demo"),
                interval=1,
                once=False,
                clear=True,
                plain=False,
                log_lines=1,
                peer_limit=1,
            )

        self.assertEqual(code, 130)
        self.assertIn("Praxist Monitor", stdout.getvalue())
        self.assertNotIn("\033[?1049h", stdout.getvalue())
        self.assertNotIn("\033[?1049l", stdout.getvalue())
        self.assertIn("display interrupted", stderr.getvalue())

    def test_foreground_monitor_rechecks_terminal_size_each_frame(self) -> None:
        from praxist.cli import monitor

        class TtyBuffer(io.StringIO):
            def isatty(self) -> bool:
                return True

        snapshot = monitor.MonitorSnapshot(
            rows=[_row()],
            selected=_row(),
            target=monitor.MonitorTarget(run_id="run_demo"),
            generated_at="2026-07-07T00:00:00+00:00",
            phase="gen2:running",
            hardware=monitor.HardwareSnapshot(loadavg="-", memory="-"),
        )
        stdout, stderr = TtyBuffer(), io.StringIO()
        with (
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            patch(
                "praxist.cli.monitor.collect_monitor_snapshot",
                return_value=snapshot,
            ) as collect,
            patch(
                "praxist.cli.monitor.shutil.get_terminal_size",
                side_effect=[os.terminal_size((72, 18)), os.terminal_size((120, 28))],
            ) as terminal_size,
            patch(
                "praxist.cli.monitor.time.sleep",
                side_effect=[None, KeyboardInterrupt],
            ),
        ):
            code = monitor.run_foreground_monitor(
                target=monitor.MonitorTarget(run_id="run_demo"),
                interval=1,
                once=False,
                clear=True,
                plain=False,
                log_lines=5,
                peer_limit=4,
            )
        self.assertEqual(code, 130)
        self.assertEqual(collect.call_count, 1)
        collect.assert_called_once_with(
            target=monitor.MonitorTarget(run_id="run_demo"),
            log_lines=5,
            scan_peer_result_artifacts=False,
        )
        self.assertEqual(terminal_size.call_count, 2)
        self.assertGreaterEqual(stdout.getvalue().count("\033[H\033[2J"), 3)

    def test_monitor_default_refresh_interval_is_fast(self) -> None:
        from praxist.cli import main, monitor

        self.assertEqual(monitor.DEFAULT_INTERVAL_SECONDS, 0.2)
        self.assertEqual(1.0 / monitor.DEFAULT_INTERVAL_SECONDS, 5.0)
        self.assertEqual(monitor.DEFAULT_SAMPLE_INTERVAL_SECONDS, 1.0)
        stdout = io.StringIO()
        with redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            main(["monitor", "--help"])
        self.assertEqual(int(raised.exception.code or 0), 0)
        help_text = stdout.getvalue()
        self.assertIn("default: 0.2 for the", help_text)
        self.assertIn("fullscreen TUI, 1 for plain text", help_text)

    def test_tui_brand_mark_uses_canonical_truecolor_and_live_animation(self) -> None:
        from praxist.cli import monitor

        snapshot = monitor.MonitorSnapshot(
            rows=[_row()],
            selected=_row(),
            target=monitor.MonitorTarget(run_id="run_demo"),
            generated_at="2026-07-07T00:00:00+00:00",
            phase="gen2:running",
        )
        renderer = monitor.TuiMonitorRenderer(color=True)
        first = renderer.render(snapshot, width=128, height=32, frame_index=0)
        second = renderer.render(snapshot, width=128, height=32, frame_index=1)

        self.assertIn("\033[38;2;60;85;255m", first)
        self.assertIn("\033[38;2;61;214;140mLIVE ●", first)
        self.assertIn("\033[38;2;61;214;140mLIVE ◉", second)
        self.assertNotIn("FPS interface", first)
        self.assertNotIn("bounded sampling", first)
        self.assertNotEqual(first, second)

        plain_mark = monitor._render_brand_mark(monitor._BRAND_MARK_PIXELS, color=False)
        self.assertEqual(len(plain_mark), 6)
        self.assertTrue(all(monitor._visible_len(line) == 14 for line in plain_mark))
        self.assertLess(sum(row.count("1") for row in monitor._BRAND_MARK_PIXELS), 50)
        self.assertFalse(any("\033[" in line for line in plain_mark))

        uncolored = monitor.TuiMonitorRenderer(color=False).render(
            snapshot,
            width=128,
            height=32,
        )
        self.assertNotIn("FPS interface", uncolored)
        self.assertNotIn("bounded sampling", uncolored)

    def test_peer_health_lights_are_colored_and_animate_at_the_frame_cadence(self) -> None:
        from praxist.cli import monitor

        peers = [
            {
                "peer_id": f"gen2_peer{index}",
                "health": health,
                "health_reason": reason,
                "research_state": "evaluating",
                "active_variant": f"variant_{index}",
            }
            for index, (health, reason) in enumerate(
                (
                    ("green", "recent progress"),
                    ("yellow", "result pending"),
                    ("red", "session failed"),
                    ("unknown", "not sampled"),
                )
            )
        ]
        row = _row(peers=peers)
        snapshot = monitor.MonitorSnapshot(
            rows=[row],
            selected=row,
            target=monitor.MonitorTarget(run_id="run_demo"),
            generated_at="2026-07-07T00:00:00+00:00",
            phase="gen2:running",
        )
        renderer = monitor.TuiMonitorRenderer(color=True)
        first = "\n".join(renderer._peer_lines(row, frame_index=0))
        second = "\n".join(renderer._peer_lines(row, frame_index=1))

        self.assertIn("\033[38;2;61;214;140m●", first)
        self.assertIn("\033[38;2;245;184;65m●", first)
        self.assertIn("\033[38;2;255;91;111m●", first)
        self.assertIn("\033[38;2;148;163;184m●", first)
        self.assertIn("\033[38;2;61;214;140m◉", second)
        self.assertIn("green: recent pro~", monitor._strip_ansi(first))

        dashboard = renderer.render(snapshot, width=160, height=40, frame_index=0)
        self.assertIn("\033[38;2;61;214;140m●", dashboard)
        self.assertIn("\033[38;2;245;184;65m●", dashboard)
        self.assertIn("\033[38;2;255;91;111m●", dashboard)
        self.assertIn("\033[38;2;148;163;184m●", dashboard)
        self.assertTrue(all(monitor._visible_len(line) == 160 for line in dashboard.splitlines()))

    def test_snapshot_sampler_refreshes_off_render_path_and_retains_failures(self) -> None:
        from praxist.cli import monitor

        initial = monitor.MonitorSnapshot(
            rows=[],
            selected=None,
            target=monitor.MonitorTarget(),
            generated_at="initial",
        )
        updated = replace(initial, generated_at="updated")
        refreshed = threading.Event()

        def collect_updated() -> monitor.MonitorSnapshot:
            refreshed.set()
            return updated

        with patch("praxist.cli.monitor.DEFAULT_SAMPLE_INTERVAL_SECONDS", 0.01):
            sampler = monitor._MonitorSnapshotSampler(
                initial=initial,
                collector=collect_updated,
                interval=0.01,
            )
            sampler.start()
            self.assertTrue(refreshed.wait(timeout=1.0))
            self.assertEqual(sampler.latest().generated_at, "updated")
            sampler.close()
            self.assertIsNotNone(sampler._thread)
            assert sampler._thread is not None
            self.assertFalse(sampler._thread.is_alive())

        failed = threading.Event()

        def fail_collection() -> monitor.MonitorSnapshot:
            failed.set()
            raise RuntimeError("probe unavailable")

        with patch("praxist.cli.monitor.DEFAULT_SAMPLE_INTERVAL_SECONDS", 0.01):
            sampler = monitor._MonitorSnapshotSampler(
                initial=initial,
                collector=fail_collection,
                interval=0.01,
            )
            sampler.start()
            self.assertTrue(failed.wait(timeout=1.0))
            for _ in range(100):
                if sampler.latest().warnings:
                    break
                threading.Event().wait(0.001)
            self.assertIn("RuntimeError: probe unavailable", sampler.latest().warnings[0])
            sampler.close()
            self.assertIsNotNone(sampler._thread)
            assert sampler._thread is not None
            self.assertFalse(sampler._thread.is_alive())

    def test_foreground_monitor_plain_forces_text_renderer(self) -> None:
        from praxist.cli import monitor

        class TtyBuffer(io.StringIO):
            def isatty(self) -> bool:
                return True

        snapshot = monitor.MonitorSnapshot(
            rows=[_row()],
            selected=_row(),
            target=monitor.MonitorTarget(run_id="run_demo"),
            generated_at="2026-07-07T00:00:00+00:00",
            phase="gen2:running",
            hardware=monitor.HardwareSnapshot(loadavg="-", memory="-"),
        )
        stdout = TtyBuffer()
        with (
            redirect_stdout(stdout),
            patch(
                "praxist.cli.monitor.collect_monitor_snapshot",
                return_value=snapshot,
            ) as collect,
        ):
            code = monitor.run_foreground_monitor(
                target=monitor.MonitorTarget(run_id="run_demo"),
                interval=1,
                once=True,
                clear=True,
                plain=True,
                log_lines=5,
                peer_limit=4,
            )
        output = stdout.getvalue()
        self.assertEqual(code, 0)
        self.assertNotIn("\033[?1049h", output)
        self.assertIn("Mode: read-only", output)
        collect.assert_called_once_with(
            target=monitor.MonitorTarget(run_id="run_demo"),
            log_lines=5,
            scan_peer_result_artifacts=True,
        )

    def test_foreground_monitor_treats_closed_pipe_as_clean_exit(self) -> None:
        from praxist.cli import monitor

        class ClosedPipe(io.StringIO):
            def write(self, value: str) -> int:
                raise BrokenPipeError

        snapshot = monitor.MonitorSnapshot(
            rows=[_row()],
            selected=_row(),
            target=monitor.MonitorTarget(run_id="run_demo"),
            generated_at="2026-07-07T00:00:00+00:00",
        )
        with (
            redirect_stdout(ClosedPipe()),
            patch("praxist.cli.monitor.collect_monitor_snapshot", return_value=snapshot),
        ):
            code = monitor.run_foreground_monitor(
                target=monitor.MonitorTarget(run_id="run_demo"),
                interval=1,
                once=True,
                clear=False,
                plain=True,
                log_lines=1,
                peer_limit=1,
            )
        self.assertEqual(code, 0)

    def test_cli_once_refuses_ambiguous_active_runs(self) -> None:
        from praxist.cli import main, monitor

        rows = [
            _row(pid=1, run_id="old", updated_at="2026-07-06T00:00:00+00:00"),
            _row(pid=2, run_id="new", updated_at="2026-07-07T00:00:00+00:00"),
        ]
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            patch("praxist.cli.monitor.status.collect_status_rows", return_value=rows),
            patch(
                "praxist.cli.monitor.collect_hardware_snapshot",
                return_value=monitor.HardwareSnapshot(loadavg="-", memory="-"),
            ),
            self.assertRaises(SystemExit) as raised,
        ):
            main(["monitor", "--once", "--no-clear"])
        self.assertEqual(int(raised.exception.code or 0), 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("multiple active rows detected", stderr.getvalue())
        self.assertIn("run_id=old", stderr.getvalue())

    def test_resolve_monitor_target_pins_latest_to_selected_run(self) -> None:
        from praxist.cli import monitor

        rows = [
            _row(pid=1, run_id="old", updated_at="2026-07-06T00:00:00+00:00"),
            _row(pid=2, run_id="new", updated_at="2026-07-07T00:00:00+00:00"),
        ]
        with patch("praxist.cli.monitor.status.collect_status_rows", return_value=rows):
            resolved = monitor.resolve_monitor_target(monitor.MonitorTarget(latest=True))
        self.assertEqual(resolved, monitor.MonitorTarget(run_id="new"))

    def test_resolve_monitor_target_refuses_latest_without_stable_identity(self) -> None:
        from praxist.cli import monitor

        row = _row(run_id=None, run_dir=None)
        stderr = io.StringIO()
        with (
            redirect_stderr(stderr),
            patch("praxist.cli.monitor.status.collect_status_rows", return_value=[row]),
        ):
            resolved = monitor.resolve_monitor_target(monitor.MonitorTarget(latest=True))
        self.assertIsNone(resolved)
        self.assertIn("no stable run_id or run_dir", stderr.getvalue())

    def test_resolve_monitor_target_sanitizes_error_and_candidate_rows(self) -> None:
        from praxist.cli import monitor

        unsafe = "unsafe\x1b[2J\x07\u202evalue"
        row = _row(run_id=unsafe, task_path=unsafe, run_dir=f"/tmp/{unsafe}")
        stderr = io.StringIO()
        with (
            redirect_stderr(stderr),
            patch("praxist.cli.monitor.status.collect_status_rows", return_value=[row]),
        ):
            resolved = monitor.resolve_monitor_target(
                monitor.MonitorTarget(run_id=unsafe + "-missing")
            )

        self.assertIsNone(resolved)
        rendered = stderr.getvalue()
        self.assertNotIn("\x1b", rendered)
        self.assertNotIn("\x07", rendered)
        self.assertNotIn("\u202e", rendered)
        self.assertIn("value", rendered)


class MonitorSelectionAndPhaseTest(unittest.TestCase):
    def test_target_peer_enrichment_preserves_all_run_rows(self) -> None:
        from praxist.cli import monitor

        target_row = _row(run_id="target", peers=[], peer_health_summary=None)
        other_row = _row(pid=4321, run_id="other", peers=[], peer_health_summary=None)

        class Peer:
            def to_dict(self) -> dict[str, object]:
                return {"peer_id": "gen2_peer0", "health": "green"}

        class Health:
            summary = {"red": 0, "yellow": 0, "green": 1}
            peers = [Peer()]

        with (
            patch(
                "praxist.cli.monitor.status.collect_status_rows",
                return_value=[target_row, other_row],
            ),
            patch("praxist.cli.monitor.status._read_peer_health", return_value=Health()) as read,
        ):
            rows = monitor._status_rows_for_target(monitor.MonitorTarget(run_id="target"))

        self.assertEqual({row.run_id for row in rows}, {"target", "other"})
        selected, warnings = monitor.select_status_row(rows, monitor.MonitorTarget(run_id="target"))
        self.assertEqual(warnings, [])
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.peers, [{"peer_id": "gen2_peer0", "health": "green"}])
        read.assert_called_once_with(
            target_row.run_dir,
            target_row.task_path,
            target_row.generation,
            scan_result_artifacts=False,
        )

    def test_explicit_offline_run_directory_is_materialized_for_monitoring(self) -> None:
        from praxist.cli import monitor

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_offline"
            run_dir.mkdir()
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "run_id": "run_offline",
                        "status": "running",
                        "started_at": "2026-07-07T00:00:00+00:00",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "run_summary.json").write_text(
                json.dumps({"status": "completed"}),
                encoding="utf-8",
            )
            (run_dir / "startup_config.json").write_text(
                json.dumps(
                    {
                        "canonical_args": {
                            "task_path": "/task/from-canonical",
                            "model": "model-a",
                            "model_provider": "model_provider:test",
                        },
                        "task_project": {"path": "/task/from-project"},
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "praxist.cli.monitor.status.collect_status_rows", return_value=[]
            ) as collect_status:
                rows = monitor._status_rows_for_target(monitor.MonitorTarget(run_dir=str(run_dir)))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.source, "offline")
        self.assertEqual(row.state, "completed")
        self.assertEqual(row.run_id, "run_offline")
        self.assertEqual(row.task_path, "/task/from-canonical")
        self.assertEqual(row.model, "model-a")
        self.assertEqual(row.model_provider_ref, "model_provider:test")
        self.assertEqual(row.started_at, "2026-07-07T00:00:00+00:00")
        collect_status.assert_called_once_with(
            include_peer_health=False,
            process_probe_timeout=1.0,
        )

    def test_renderer_empty_and_limited_sections(self) -> None:
        from praxist.cli import monitor

        empty = monitor.MonitorSnapshot(
            rows=[],
            selected=None,
            target=monitor.MonitorTarget(),
            generated_at="2026-07-07T00:00:00+00:00",
            hardware=monitor.HardwareSnapshot(warnings=["gpu probe skipped"]),
            warnings=["multiple rows"],
        )
        renderer = monitor.TextMonitorRenderer(peer_limit=1)
        rendered = renderer.render(empty)
        self.assertIn("No Praxist run rows detected", rendered)
        self.assertIn("Warnings", rendered)
        self.assertIn("gpu probe skipped", rendered)
        self.assertIn("No single selected run", "\n".join(renderer._render_selected(empty)))

        peers = [
            {
                "peer_id": f"gen0_peer{i}",
                "health": "green",
                "research_state": "a very long active research state",
                "active_variant": "a very long variant identifier",
                "best_metric_value": "not-a-number",
            }
            for i in range(3)
        ]
        peer_text = "\n".join(renderer._render_peers(_row(peers=peers)))
        self.assertIn("2 more peers hidden", peer_text)
        self.assertIn("a very long act", peer_text)
        self.assertIn("No peer health rows", "\n".join(renderer._render_peers(_row(peers=[]))))
        self.assertIn("No recent run logs", "\n".join(renderer._render_logs([])))
        blocker_text = "\n".join(
            renderer._render_selected(
                monitor.MonitorSnapshot(
                    rows=[],
                    selected=_row(),
                    target=monitor.MonitorTarget(),
                    generated_at="now",
                    orchestrator_status={"gen_promotion_blocker": "waiting for evidence"},
                )
            )
        )
        self.assertIn("promotion_blocker", blocker_text)

    def test_select_status_row_covers_explicit_and_task_filters(self) -> None:
        from praxist.cli import monitor

        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp) / "task"
            task.mkdir()
            run_a = task / "experiments" / "run_a"
            run_b = task / "experiments" / "run_b"
            run_a.mkdir(parents=True)
            run_b.mkdir(parents=True)
            stale = _row(
                pid=10,
                run_id="run_a",
                run_dir=str(run_a),
                task_path=str(task),
                state="stale",
                updated_at="2026-07-06T00:00:00+00:00",
            )
            active = _row(
                pid=11,
                run_id="run_b",
                run_dir=str(run_b),
                task_path=str(task),
                state="running",
                updated_at="2026-07-07T00:00:00+00:00",
            )

            selected, warnings = monitor.select_status_row(
                [stale, active],
                monitor.MonitorTarget(run_id="missing"),
            )
            self.assertIsNone(selected)
            self.assertIn("run id not found", warnings[0])

            selected, warnings = monitor.select_status_row(
                [stale, active],
                monitor.MonitorTarget(run_dir=str(run_a)),
            )
            self.assertIs(selected, stale)
            self.assertEqual(warnings, [])

            selected, warnings = monitor.select_status_row(
                [stale, active],
                monitor.MonitorTarget(run_dir=str(task / "nope")),
            )
            self.assertIsNone(selected)
            self.assertIn("run dir not found", warnings[0])

            selected, warnings = monitor.select_status_row(
                [stale, active],
                monitor.MonitorTarget(task_path=str(task)),
            )
            self.assertIs(selected, active)
            self.assertEqual(warnings, [])

            selected, warnings = monitor.select_status_row(
                [stale, active],
                monitor.MonitorTarget(task_path=str(task), latest=True),
            )
            self.assertIs(selected, active)
            self.assertEqual(warnings, [])

            selected, warnings = monitor.select_status_row(
                [stale],
                monitor.MonitorTarget(task_path=str(task / "missing")),
            )
            self.assertIsNone(selected)
            self.assertIn("no rows match task path", warnings[0])

        selected, warnings = monitor.select_status_row([], monitor.MonitorTarget(latest=True))
        self.assertIsNone(selected)
        self.assertEqual(warnings, [])

        selected, warnings = monitor.select_status_row([], monitor.MonitorTarget())
        self.assertIsNone(selected)
        self.assertEqual(warnings, [])

        selected, warnings = monitor.select_status_row(
            [_row(state="stale")], monitor.MonitorTarget()
        )
        self.assertIsNotNone(selected)
        self.assertEqual(warnings, [])

        selected, warnings = monitor.select_status_row(
            [_row(state="running")], monitor.MonitorTarget()
        )
        self.assertIsNotNone(selected)
        self.assertEqual(warnings, [])

        selected, warnings = monitor.select_status_row(
            [_row(pid=1, run_id="a", state="stale"), _row(pid=2, run_id="b", state="stale")],
            monitor.MonitorTarget(),
        )
        self.assertIsNone(selected)
        self.assertIn("multiple non-running rows", warnings[0])

        rows = [
            _row(pid=1, run_id="a", task_path="/same", state="running"),
            _row(pid=2, run_id="b", task_path="/same", state="running"),
        ]
        selected, warnings = monitor.select_status_row(
            rows, monitor.MonitorTarget(task_path="/same")
        )
        self.assertIsNone(selected)
        self.assertIn("multiple rows match task path", warnings[0])
        selected, warnings = monitor.select_status_row(
            rows,
            monitor.MonitorTarget(task_path="/same", latest=True),
        )
        self.assertIs(selected, rows[1])
        self.assertEqual(warnings, [])

    def test_infer_run_phase_covers_generation_states(self) -> None:
        from praxist.cli import monitor

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            row = _row(run_dir=str(run_dir), generation=3)
            self.assertEqual(
                monitor.infer_run_phase(
                    run_dir=run_dir,
                    row=row,
                    orchestrator_status={"exit_condition": "completed"},
                ),
                "finished:completed",
            )
            self.assertEqual(
                monitor.infer_run_phase(
                    run_dir=run_dir, row=_row(generation=None), orchestrator_status={}
                ),
                "starting/no-orchestrator-status",
            )
            self.assertEqual(
                monitor.infer_run_phase(run_dir=run_dir, row=row, orchestrator_status={}),
                "gen3:initializing",
            )
            gen3 = run_dir / "gen_3"
            gen3.mkdir()
            (gen3 / "generation_results.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                monitor.infer_run_phase(run_dir=run_dir, row=row, orchestrator_status={}),
                "gen3:boundary-pending",
            )
            (gen3 / "generation_boundary.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                monitor.infer_run_phase(run_dir=run_dir, row=row, orchestrator_status={}),
                "gen3:boundary-committed",
            )
            (gen3 / "generation_boundary.json").unlink()
            (gen3 / "generation_results.json").unlink()
            (gen3 / "STOP_SIGNAL").write_text("stop", encoding="utf-8")
            self.assertEqual(
                monitor.infer_run_phase(run_dir=run_dir, row=row, orchestrator_status={}),
                "gen3:closing",
            )
            (gen3 / "STOP_SIGNAL").unlink()
            dig = gen3 / "dig"
            dig.mkdir()
            (dig / "dig_stage_status.json").write_text(
                json.dumps({"last_phase": "qd", "last_status": "ok"}),
                encoding="utf-8",
            )
            self.assertEqual(
                monitor.infer_run_phase(run_dir=run_dir, row=row, orchestrator_status={}),
                "gen3:dig:qd:ok",
            )
            (dig / "dig_stage_status.json").unlink()
            self.assertEqual(
                monitor.infer_run_phase(run_dir=run_dir, row=row, orchestrator_status={}),
                "gen3:running",
            )

    def test_log_json_and_hardware_helpers(self) -> None:
        from praxist.cli import monitor

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self.assertEqual(monitor.tail_recent_logs(run_dir, max_lines=0), [])
            self.assertEqual(monitor.tail_recent_logs(run_dir, max_lines=5), [])
            logs = run_dir / "logs"
            logs.mkdir()
            (logs / "old.log").write_text("old1\nold2\n", encoding="utf-8")
            (logs / "new.log").write_text("new1\nnew2\nnew3\n", encoding="utf-8")
            (logs / "link.log").symlink_to(logs / "new.log")
            tailed = monitor.tail_recent_logs(run_dir, max_lines=4)
            self.assertTrue(any("[new.log]" in line for line in tailed))
            self.assertFalse(any("[link.log]" in line for line in tailed))

            payload = run_dir / "status.json"
            payload.write_text(json.dumps({"ok": True}), encoding="utf-8")
            self.assertEqual(monitor._read_json_object(payload), {"ok": True})
            payload.write_text("[]", encoding="utf-8")
            self.assertEqual(monitor._read_json_object(payload), {})
            payload.write_text("{bad", encoding="utf-8")
            self.assertEqual(monitor._read_json_object(payload), {})
            link = run_dir / "link.json"
            link.symlink_to(payload)
            self.assertEqual(monitor._read_json_object(link), {})

        self.assertEqual(monitor._format_kib(512), "0 MiB")
        self.assertEqual(monitor._format_kib(2 * 1024 * 1024), "2.0 GiB")
        self.assertEqual(monitor._format_peer_health(None), "-")
        self.assertEqual(monitor._format_peer_health({"red": 0, "yellow": 0, "green": 0}), "-")
        self.assertEqual(monitor._format_float("bad"), "-")
        self.assertEqual(monitor._safe_int("7", None), 7)
        self.assertEqual(monitor._safe_int("bad", 3), 3)
        self.assertEqual(monitor._display("", None), "-")
        self.assertEqual(monitor._truncate("abcdef", 4), "a...")

        with patch(
            "pathlib.Path.read_text", return_value="MemTotal: 2048 kB\nMemAvailable: 1024 kB\n"
        ):
            self.assertEqual(monitor._read_meminfo(), "1 MiB / 2 MiB")
        with (
            patch("pathlib.Path.read_text", side_effect=OSError),
            patch("praxist.cli.monitor.os.sysconf", side_effect=OSError),
        ):
            self.assertEqual(monitor._read_meminfo(), "-")
        sysconf_values = {
            "SC_PHYS_PAGES": 1_048_576,
            "SC_PAGE_SIZE": 4096,
            "SC_AVPHYS_PAGES": 262_144,
        }
        with (
            patch("pathlib.Path.read_text", side_effect=OSError),
            patch(
                "praxist.cli.monitor.os.sysconf",
                side_effect=lambda key: sysconf_values[key],
            ),
        ):
            self.assertEqual(monitor._read_meminfo(), "3.0 GiB / 4.0 GiB")
        with (
            patch("praxist.cli.monitor.os.getloadavg", return_value=(1.0, 2.0, 3.0)),
            patch("praxist.cli.monitor._read_meminfo", return_value="1 / 2"),
            patch("praxist.cli.monitor._read_nvidia_smi", return_value=["GPU 0"]),
        ):
            snapshot = monitor.collect_hardware_snapshot()
        self.assertEqual(snapshot.loadavg, "1.00 2.00 3.00")
        self.assertEqual(snapshot.memory, "1 / 2")
        self.assertEqual(snapshot.gpus, ["GPU 0"])

        warnings: list[str] = []
        with patch("praxist.cli.monitor.shutil.which", return_value=None):
            self.assertEqual(monitor._read_nvidia_smi(warnings), [])
        with (
            patch("praxist.cli.monitor.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch(
                "praxist.cli.monitor.subprocess.run",
                side_effect=subprocess.TimeoutExpired("nvidia-smi", 3),
            ),
        ):
            self.assertEqual(monitor._read_nvidia_smi(warnings), [])
        with (
            patch("praxist.cli.monitor.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch(
                "praxist.cli.monitor.subprocess.run",
                return_value=subprocess.CompletedProcess(["nvidia-smi"], 1, stdout=""),
            ),
        ):
            self.assertEqual(monitor._read_nvidia_smi(warnings), [])
        with (
            patch("praxist.cli.monitor.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch(
                "praxist.cli.monitor.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    ["nvidia-smi"],
                    0,
                    stdout="0, 11, 100, 1000\nbad\n",
                ),
            ),
        ):
            self.assertEqual(
                monitor._read_nvidia_smi(warnings), ["GPU 0: util 11%, mem 100/1000 MiB"]
            )

    def test_file_helper_exception_branches(self) -> None:
        from praxist.cli import monitor

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            logs = run_dir / "logs"
            logs.mkdir()
            with patch("pathlib.Path.glob", side_effect=OSError):
                self.assertEqual(monitor.tail_recent_logs(run_dir, max_lines=3), [])
            (logs / "empty.log").write_text("", encoding="utf-8")
            with patch("praxist.cli.monitor._tail_file", return_value=[]):
                self.assertEqual(monitor.tail_recent_logs(run_dir, max_lines=3), [])
            path = logs / "x.log"
            path.write_text("x", encoding="utf-8")
            with patch("pathlib.Path.stat", side_effect=OSError):
                self.assertEqual(monitor._tail_file(path, 1), [])
                self.assertEqual(monitor._mtime(path), 0.0)
            with patch("pathlib.Path.is_dir", side_effect=OSError):
                self.assertIsNone(monitor._generation_dir(run_dir, 9))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
