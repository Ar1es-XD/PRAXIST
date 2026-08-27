from __future__ import annotations

import io
import json
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class RunCliContractsTest(unittest.TestCase):
    def test_fake_fixture_claims_only_a_fresh_or_launcher_owned_run_shell(self) -> None:
        from praxist.testing.fake_workflow_fixture import _claim_run_dir

        with tempfile.TemporaryDirectory() as tmp_raw:
            root = Path(tmp_raw)
            fresh = root / "fresh"
            _claim_run_dir(fresh)
            self.assertTrue(fresh.is_dir())

            launcher_shell = root / "launcher-shell"
            logs = launcher_shell / "logs"
            logs.mkdir(parents=True)
            (launcher_shell / ".DS_Store").write_text("", encoding="utf-8")
            (launcher_shell / ".gitkeep").write_text("", encoding="utf-8")
            (logs / ".gitkeep").write_text("", encoding="utf-8")
            (logs / "launcher.nohup.log").write_text("ready\n", encoding="utf-8")
            _claim_run_dir(launcher_shell)

            (launcher_shell / "run_summary.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, r"run_summary\.json"):
                _claim_run_dir(launcher_shell)

            (launcher_shell / "run_summary.json").unlink()
            (logs / "unexpected.log").write_text("unexpected\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, r"\(logs\)"):
                _claim_run_dir(launcher_shell)

    def _run_args(self, **overrides):
        defaults = {
            "workspace": "",
            "resume_from": "",
            "run_dir": "",
            "resume": False,
            "resume_policy": "completed_generation",
            "task_path": "",
            "fake": False,
            "task": "",
            "runtime": "",
            "model_provider": "",
            "budget_policy": "",
            "credential_profile": "",
            "resolve_only": False,
            "task_spec": "",
            "model": "",
            "local": True,
            "frontier_strategy": "auto",
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_cmd_run_fake_fixture_and_argument_error_edges(self) -> None:
        from praxist import run

        observed = {}

        def fake_fixture(**kwargs):
            observed.update(kwargs)
            return {"success": True, "run_dir": str(kwargs["run_dir"])}

        with (
            patch(
                "praxist.testing.fake_workflow_fixture.run_fake_workflow_fixture",
                side_effect=fake_fixture,
            ),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            run.cmd_run(self._run_args(fake=True, task="task:fake_panel"))
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["success"])
        self.assertIn("fake_panel", payload["run_dir"])
        self.assertEqual(observed["runtime_ref"], "agent_runtime:fake_runtime")
        self.assertEqual(observed["model_provider_ref"], "model_provider:fake_provider")
        self.assertEqual(observed["budget_policy_ref"], "budget_policy:fake_tiered")

        with (
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
            self.assertRaises(SystemExit) as raised,
        ):
            run.cmd_run(
                self._run_args(
                    resume_from="/tmp/resume-a",
                    run_dir="/tmp/resume-b",
                )
            )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--resume-from and --run-dir", stderr.getvalue())

        with (
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
            self.assertRaises(SystemExit) as raised,
        ):
            run.cmd_run(self._run_args(task="task:legacy"))
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("task refs are no longer discovered", stderr.getvalue())

        with (
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
            self.assertRaises(SystemExit) as raised,
        ):
            run.cmd_run(self._run_args())
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--task-path is required", stderr.getvalue())

    def test_task_runner_lifecycle_observer_keyword_is_capability_checked(self) -> None:
        from praxist.run import (
            _runner_accepts_lifecycle_observer,
            _task_project_planned_peer_count,
        )

        def legacy_runner(
            *,
            workspace,
            task_ref,
            task_project,
            run_dir,
            runtime_ref,
            model_provider_ref,
            budget_policy_ref,
            credential_profile,
            resolve_only,
        ):
            return None

        def generic_forwarder(**_kwargs):
            return None

        def telemetry_aware_runner(*, run_lifecycle_observer=None, **_kwargs):
            return None

        self.assertFalse(_runner_accepts_lifecycle_observer(legacy_runner))
        self.assertFalse(_runner_accepts_lifecycle_observer(generic_forwarder))
        self.assertTrue(_runner_accepts_lifecycle_observer(telemetry_aware_runner))
        self.assertEqual(
            _task_project_planned_peer_count(
                SimpleNamespace(descriptor={"generation_policy": {"cohort_size": 8}})
            ),
            8,
        )
        self.assertIsNone(_task_project_planned_peer_count(SimpleNamespace(descriptor={})))

    def test_cmd_run_task_project_dispatch_edges(self) -> None:
        from praxist import run

        task_project = SimpleNamespace(
            path=Path("/tmp/task-project"),
            task_ref="task:project",
            descriptor={"generation_policy": {"cohort_size": 3}},
        )

        with (
            patch(
                "praxist.core.task_project.resolve_task_project",
                return_value=task_project,
            ),
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
            self.assertRaises(SystemExit) as raised,
        ):
            run.cmd_run(
                self._run_args(
                    task_path="/tmp/task-project",
                    task="task:other",
                )
            )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("task ref mismatch", stderr.getvalue())

        observed = {}
        lifecycle_observer = object()

        def fake_runner(*, run_lifecycle_observer=None, **kwargs):
            observed.update(kwargs)
            observed["run_lifecycle_observer"] = run_lifecycle_observer
            return {"success": True, "run_dir": str(kwargs["run_dir"])}

        with (
            patch(
                "praxist.core.task_project.resolve_task_project",
                return_value=task_project,
            ),
            patch(
                "praxist.core.task_project.task_project_has_capability",
                return_value=True,
            ),
            patch(
                "praxist.core.task_project.load_task_project_runner",
                return_value=fake_runner,
            ),
            patch(
                "praxist.infrastructure.product_usage.ProductUsageObserver.create",
                return_value=lifecycle_observer,
            ),
            patch("praxist.cli.product_usage.prompt_for_consent_if_unset") as consent_prompt,
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            run.cmd_run(self._run_args(task_path="/tmp/task-project"))
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["success"])
        self.assertEqual(observed["task_project"], task_project)
        self.assertEqual(observed["task_ref"], "task:project")
        self.assertEqual(observed["runtime_ref"], "agent_runtime:fake_runtime")
        self.assertIs(observed["run_lifecycle_observer"], lifecycle_observer)
        consent_prompt.assert_called_once_with()

        with (
            patch(
                "praxist.core.task_project.resolve_task_project",
                return_value=task_project,
            ),
            patch(
                "praxist.core.task_project.task_project_has_capability",
                return_value=False,
            ),
            patch(
                "praxist.plugins.workflow_stages.research_loop.startup.is_research_loop_task_project",
                return_value=False,
            ),
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
            self.assertRaises(SystemExit) as raised,
        ):
            run.cmd_run(self._run_args(task_path="/tmp/task-project"))
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("Unsupported task project workflow", stderr.getvalue())

        with (
            patch(
                "praxist.core.task_project.resolve_task_project",
                return_value=task_project,
            ),
            patch(
                "praxist.core.task_project.task_project_has_capability",
                return_value=False,
            ),
            patch(
                "praxist.plugins.workflow_stages.research_loop.startup.is_research_loop_task_project",
                return_value=True,
            ),
            patch(
                "praxist.plugins.workflow_stages.research_loop.startup.default_runtime_for_task",
                return_value="agent_runtime:fake",
            ),
            patch(
                "praxist.plugins.workflow_stages.research_loop.startup.default_model_provider_for_task",
                return_value="model_provider:fake",
            ),
            patch(
                "praxist.plugins.workflow_stages.research_loop.startup.default_budget_policy_for_task",
                return_value="budget_policy:fake",
            ),
            patch(
                "praxist.plugins.workflow_stages.research_loop.startup.prepare_research_loop_plugin_run",
                side_effect=RuntimeError("startup"),
            ),
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
            self.assertRaises(SystemExit) as raised,
        ):
            run.cmd_run(self._run_args(task_path="/tmp/task-project"))
        self.assertEqual(raised.exception.code, 3)
        self.assertIn("startup failed: startup", stderr.getvalue())

        prepared = SimpleNamespace(
            stage_budget_grant_id="grant",
            task_spec=SimpleNamespace(),
            task_execution_cwd=Path("/tmp/task-project"),
            run_dir=Path("/tmp/task-project/experiments/run_resolved"),
            startup_config={"canonical_args": {"model": "model"}},
            runtime_ref="agent_runtime:fake",
            model_provider_ref="model_provider:fake",
            budget_policy_ref="budget_policy:fake",
            model_provider_credential_key_id="key",
            provider_env={},
            tool_server_refs=[],
            registry=None,
            task_project_path=Path("/tmp/task-project"),
            run_id="run-resolved",
        )
        stage_result = SimpleNamespace(
            success=True,
            summary={"generations_completed": 0, "run_dir": str(prepared.run_dir)},
            error="",
        )
        finalized: list[dict[str, object]] = []

        with (
            patch(
                "praxist.core.task_project.resolve_task_project",
                return_value=task_project,
            ),
            patch(
                "praxist.core.task_project.task_project_has_capability",
                return_value=False,
            ),
            patch(
                "praxist.plugins.workflow_stages.research_loop.startup.is_research_loop_task_project",
                return_value=True,
            ),
            patch(
                "praxist.plugins.workflow_stages.research_loop.startup.default_runtime_for_task",
                return_value="agent_runtime:fake",
            ),
            patch(
                "praxist.plugins.workflow_stages.research_loop.startup.default_model_provider_for_task",
                return_value="model_provider:fake",
            ),
            patch(
                "praxist.plugins.workflow_stages.research_loop.startup.default_budget_policy_for_task",
                return_value="budget_policy:fake",
            ),
            patch(
                "praxist.plugins.workflow_stages.research_loop.startup.prepare_research_loop_plugin_run",
                return_value=prepared,
            ),
            patch(
                "praxist.plugins.workflow_stages.research_loop.startup.finalize_research_loop_plugin_run",
                side_effect=lambda _prepared, **kwargs: finalized.append(kwargs),
            ),
            patch(
                "praxist.plugins.workflow_stages.research_loop.stage.run_research_loop_stage",
                return_value=stage_result,
            ),
            patch.object(run, "_install_research_loop_signal_finalizer", return_value=lambda: None),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            run.cmd_run(self._run_args(task_path="/tmp/task-project", resolve_only=True))
        resolved = json.loads(stdout.getvalue())
        self.assertEqual(resolved["run_id"], "run-resolved")
        self.assertEqual(resolved["status"], "resolved")
        self.assertTrue(finalized[-1]["success"])

    def test_cmd_run_research_loop_failure_and_completion_edges(self) -> None:
        from praxist import run

        task_project = SimpleNamespace(
            path=Path("/tmp/task-project"),
            task_ref="task:project",
            descriptor={},
        )

        def base_prepared(**overrides):
            prepared = SimpleNamespace(
                stage_budget_grant_id="grant",
                task_spec=SimpleNamespace(),
                task_execution_cwd=Path("/tmp/task-project"),
                run_dir=Path("/tmp/task-project/experiments/run"),
                startup_config={"canonical_args": {"model": "model"}},
                runtime_ref="agent_runtime:fake",
                model_provider_ref="model_provider:fake",
                budget_policy_ref="budget_policy:fake",
                model_provider_credential_key_id="key",
                provider_env={},
                tool_server_refs=[],
                registry=None,
                task_project_path=Path("/tmp/task-project"),
                run_id="run-id",
            )
            for key, value in overrides.items():
                setattr(prepared, key, value)
            return prepared

        def run_with(
            prepared, stage_result=None, stage_side_effect=None, finalize_side_effect=None
        ):
            with (
                patch(
                    "praxist.core.task_project.resolve_task_project",
                    return_value=task_project,
                ),
                patch(
                    "praxist.core.task_project.task_project_has_capability",
                    return_value=False,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.startup.is_research_loop_task_project",
                    return_value=True,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.startup.default_runtime_for_task",
                    return_value="agent_runtime:fake",
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.startup.default_model_provider_for_task",
                    return_value="model_provider:fake",
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.startup.default_budget_policy_for_task",
                    return_value="budget_policy:fake",
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.startup.prepare_research_loop_plugin_run",
                    return_value=prepared,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.startup.finalize_research_loop_plugin_run",
                    side_effect=finalize_side_effect,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.stage.run_research_loop_stage",
                    return_value=stage_result,
                    side_effect=stage_side_effect,
                ),
                patch.object(
                    run, "_install_research_loop_signal_finalizer", return_value=lambda: None
                ),
                patch("sys.stdout", new_callable=io.StringIO) as stdout,
                patch("sys.stderr", new_callable=io.StringIO) as stderr,
            ):
                run.cmd_run(self._run_args(task_path="/tmp/task-project"))
            return stdout.getvalue(), stderr.getvalue()

        with self.assertRaises(SystemExit) as raised:
            run_with(base_prepared(stage_budget_grant_id=""))
        self.assertEqual(raised.exception.code, 5)

        with self.assertRaises(RuntimeError):
            run_with(base_prepared(), stage_side_effect=RuntimeError("stage"))

        ok_result = SimpleNamespace(
            success=True,
            summary={"generations_completed": 2, "run_dir": "/tmp/run"},
            error="",
        )
        with self.assertRaises(SystemExit) as raised:
            run_with(
                base_prepared(),
                stage_result=ok_result,
                finalize_side_effect=[RuntimeError("fin"), None],
            )
        self.assertEqual(raised.exception.code, 1)

        failed_result = SimpleNamespace(
            success=False,
            summary={"generations_completed": 0, "run_dir": "/tmp/run"},
            error="stage failed",
        )
        with self.assertRaises(SystemExit) as raised:
            run_with(base_prepared(), stage_result=failed_result)
        self.assertEqual(raised.exception.code, 1)

        stdout, _stderr = run_with(base_prepared(), stage_result=ok_result)
        self.assertIn("Run complete: 2 generations", stdout)
        self.assertIn("Run directory: /tmp/run", stdout)

    def test_run_dir_helpers_runner_loader_and_signal_finalizer_edges(self) -> None:
        from praxist import run

        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "task"
            task_root.mkdir()
            task_project = SimpleNamespace(
                path=task_root,
                task_ref="task:demo",
                descriptor={"runtime_outputs": {"root": "runs"}},
            )
            default_run_dir = run._default_run_dir_for_task_project(
                task_project, Path(tmp), "task:demo"
            )
            self.assertEqual(default_run_dir.parent, task_root / "runs")
            self.assertIn("demo", default_run_dir.name)
            self.assertIn(
                "demo",
                run._default_run_dir_for_fake_fixture("task:demo").name,
            )
            with self.assertRaises(ValueError):
                run._ensure_run_dir_not_in_source_checkout(Path.cwd() / "inside")

        class RunnerObject:
            def __init__(self):
                self.called = False

            def run(self):
                self.called = True
                return {"ok": True}

        runner_object = RunnerObject()
        self.assertEqual(
            run._task_runner_for_capability("task", "cap", lambda _task: runner_object)(),
            {"ok": True},
        )
        self.assertTrue(runner_object.called)
        self.assertEqual(
            run._task_runner_for_capability("task", "cap", lambda _task: lambda: "ok")(),
            "ok",
        )
        with self.assertRaises(TypeError):
            run._task_runner_for_capability(
                SimpleNamespace(task_ref="task:bad"),
                "cap",
                lambda _task: object(),
            )

        prepared = SimpleNamespace(run_dir=Path("/tmp/run"))
        finalized = []
        exits = []

        def finalize(*args, **kwargs):
            finalized.append((args, kwargs))

        def fake_exit(code):
            exits.append(code)
            raise SystemExit(code)

        with (
            patch.object(run.os, "_exit", side_effect=fake_exit),
            patch.object(signal, "getsignal", return_value=None),
            patch.object(signal, "signal") as signal_mock,
        ):
            restore = run._install_research_loop_signal_finalizer(prepared, finalize)
            handler = signal_mock.call_args_list[0].args[1]
            with self.assertRaises(SystemExit):
                handler(15, None)
            with self.assertRaises(SystemExit):
                handler(15, None)
            restore()

        self.assertEqual(exits, [143, 143])
        self.assertEqual(finalized[0][1]["exit_code"], 143)

        failing_finalized = []

        def failing_finalize(*args, **kwargs):
            failing_finalized.append((args, kwargs))
            raise RuntimeError("finalize")

        with (
            patch.object(run.os, "_exit", side_effect=fake_exit),
            patch.object(signal, "getsignal", return_value=None),
            patch.object(signal, "signal") as signal_mock,
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            restore = run._install_research_loop_signal_finalizer(prepared, failing_finalize)
            handler = signal_mock.call_args_list[0].args[1]
            with self.assertRaises(SystemExit):
                handler(-999, None)
            restore()
        self.assertIn("signal finalization failed: finalize", stderr.getvalue())
        self.assertEqual(failing_finalized[0][1]["exit_code"], -871)

        signal_calls = []

        def flaky_signal(_sig, _handler):
            signal_calls.append(_sig)
            if len(signal_calls) in {1, 3}:
                raise OSError("signal")
            return None

        with (
            patch.object(signal, "getsignal", return_value=None),
            patch.object(signal, "signal", side_effect=flaky_signal),
        ):
            restore = run._install_research_loop_signal_finalizer(prepared, finalize)
            restore()
        self.assertGreaterEqual(len(signal_calls), 3)

        with patch(
            "praxist.testing.fake_workflow_fixture.FakeWorkflowFixtureTaskRunner",
            side_effect=lambda task: lambda: {"task_ref": task.task_ref},
        ):
            fallback_runner = run._task_runner_for_capability(
                SimpleNamespace(task_ref="task:fallback"),
                "testing.fake_workflow_fixture",
                lambda _task: (_ for _ in ()).throw(ValueError("no entrypoint")),
            )
        self.assertEqual(fallback_runner(), {"task_ref": "task:fallback"})

    def test_peer_server_replay_parity_and_main_edges(self) -> None:
        from praxist import run

        peer_calls = []
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "praxist.infrastructure.execute_autonomous.main",
                side_effect=lambda: peer_calls.append(True),
            ),
        ):
            run.cmd_peer(
                SimpleNamespace(
                    peer_id="peer1",
                    generation_id=2,
                    max_runtime=9,
                    prompt_file="prompt.md",
                    model="model",
                    local=True,
                )
            )
            self.assertEqual(peer_calls, [True])
            self.assertEqual(os.environ["PEER_ID"], "peer1")
            self.assertEqual(os.environ["GENERATION_ID"], "2")
            self.assertEqual(os.environ["MAX_RUNTIME_SECONDS"], "9")
            self.assertEqual(os.environ["TASK_PROMPT_FILE"], "prompt.md")
            self.assertEqual(os.environ["AGENT_MODEL"], "model")
            self.assertEqual(os.environ["LOCAL_MODE"], "true")

        with self.assertRaises(SystemExit) as server_exit:
            run.cmd_server(SimpleNamespace())
        self.assertEqual(server_exit.exception.code, 1)

        replay_args = SimpleNamespace(
            run_dir="/tmp/run",
            mode="verify",
            strict_tail=True,
            allow_plugin_drift=True,
            locked=True,
        )
        with (
            patch("praxist.core.replay.verify_run", return_value={"success": False}),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
            self.assertRaises(SystemExit) as replay_exit,
        ):
            run.cmd_replay(replay_args)
        self.assertEqual(replay_exit.exception.code, 1)
        self.assertEqual(json.loads(stdout.getvalue())["success"], False)

        dry_args = SimpleNamespace(
            run_dir="/tmp/run",
            mode="dry-run",
            strict_tail=False,
            allow_plugin_drift=False,
            locked=False,
        )
        with patch("praxist.core.replay.dry_run", return_value={"success": True}):
            run.cmd_replay(dry_args)

        with (
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend.parity.verify_research_loop_parity",
                return_value={"success": False},
            ),
            self.assertRaises(SystemExit) as parity_exit,
        ):
            run.cmd_parity(
                SimpleNamespace(
                    run_dir="/tmp/run",
                    deliverables_dir="",
                    strict=True,
                    write_report=False,
                )
            )
        self.assertEqual(parity_exit.exception.code, 1)

        with (
            patch.object(sys, "argv", ["praxist.run"]),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
            self.assertRaises(SystemExit) as main_exit,
        ):
            run.main()
        self.assertEqual(main_exit.exception.code, 1)
        self.assertIn("usage:", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
