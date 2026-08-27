from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from praxist.core.replay import inspect_run, verify_run
from praxist.core.storage import read_jsonl
from praxist.core.task_project import resolve_task_project
from praxist.testing.fake_workflow_fixture import run_fake_workflow_fixture
from tests.helpers.paths import REPO_ROOT

FIXTURE_PLUGIN_ROOT = REPO_ROOT / "tests" / "fixtures" / "plugins"
API_KEY_ENV_NAMES = (
    "ANTHROPIC_API_KEY",
    "CLAUDE_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
    "CROSSREF_API_KEY",
)


class OfflineResearchRunIntegrationTest(unittest.TestCase):
    def test_launcher_owned_run_shell_is_claimed_without_reusing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            tmp = Path(tmp_raw)
            task_path = _copy_example_task(tmp)
            run_dir = tmp / "runs" / "launcher_shell"
            logs_dir = run_dir / "logs"
            logs_dir.mkdir(parents=True)
            (logs_dir / "launcher.nohup.log").write_text("launcher ready\n", encoding="utf-8")
            env_overrides = {"PRAXIST_BUNDLED_PLUGIN_ROOTS": str(FIXTURE_PLUGIN_ROOT)}
            env_overrides.update({key: "" for key in API_KEY_ENV_NAMES})

            with patch.dict(os.environ, env_overrides, clear=False):
                task_project = resolve_task_project(task_path, workspace=tmp)
                result = run_fake_workflow_fixture(
                    workspace=tmp,
                    task_project=task_project,
                    run_dir=run_dir,
                    credential_profile="fake_multi_key",
                )

            self.assertEqual(result["status"], "succeeded")
            self.assertTrue((run_dir / "run_summary.json").is_file())

    def test_fake_workflow_rejects_preexisting_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            tmp = Path(tmp_raw)
            task_path = _copy_example_task(tmp)
            run_dir = tmp / "runs" / "old_run"
            run_dir.mkdir(parents=True)
            (run_dir / "run_summary.json").write_text("{}\n", encoding="utf-8")
            env_overrides = {"PRAXIST_BUNDLED_PLUGIN_ROOTS": str(FIXTURE_PLUGIN_ROOT)}
            env_overrides.update({key: "" for key in API_KEY_ENV_NAMES})

            with patch.dict(os.environ, env_overrides, clear=False):
                task_project = resolve_task_project(task_path, workspace=tmp)
                with self.assertRaisesRegex(FileExistsError, "run artifacts"):
                    run_fake_workflow_fixture(
                        workspace=tmp,
                        task_project=task_project,
                        run_dir=run_dir,
                        credential_profile="fake_multi_key",
                    )

    def test_external_task_fake_workflow_round_trip_produces_replayable_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            tmp = Path(tmp_raw)
            task_path = _copy_example_task(tmp)
            env_overrides = {"PRAXIST_BUNDLED_PLUGIN_ROOTS": str(FIXTURE_PLUGIN_ROOT)}
            env_overrides.update({key: "" for key in API_KEY_ENV_NAMES})
            with patch.dict(os.environ, env_overrides, clear=False):
                task_project = resolve_task_project(task_path, workspace=tmp)
                run_dir = tmp / "runs" / "offline_integration"
                result = run_fake_workflow_fixture(
                    workspace=tmp,
                    task_project=task_project,
                    run_dir=run_dir,
                    credential_profile="fake_multi_key",
                )

                self.assertEqual(result["status"], "succeeded")
                self.assertEqual(Path(result["run_dir"]), run_dir)
                self.assertFalse(task_project.path.is_relative_to(REPO_ROOT))

                replay_report = verify_run(run_dir)
                self.assertTrue(replay_report["success"], replay_report["errors"])
                summary = inspect_run(run_dir)
                self.assertEqual(summary["stages"]["research_loop"], "succeeded")
                self.assertEqual(summary["stages"]["ideation"], "skipped")
                self.assertEqual(summary["stages"]["paper_writing"], "skipped")
                self.assertEqual(summary["stages"]["reviewer"], "skipped")
                self.assertEqual(summary["findings"], 4)
                self.assertEqual(summary["frontier_records"], 1)

                _assert_required_run_files(self, run_dir)
                _assert_external_task_manifest(self, run_dir, task_path)
                _assert_plugin_resolution(self, run_dir)
                _assert_ledgers(self, run_dir)
                _assert_no_real_secret_material(self, run_dir)

    def test_python_entrypoint_runs_external_fixture_task_and_cli_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            tmp = Path(tmp_raw)
            task_path = _copy_example_task(tmp)
            run_dir = tmp / "runs" / "cli_integration"
            env = _offline_subprocess_env()

            run_command = [
                sys.executable,
                "-m",
                "praxist.run",
                "run",
                "--task-path",
                str(task_path),
                "--workspace",
                str(tmp),
                "--run-dir",
                str(run_dir),
                "--local",
                "--credential-profile",
                "fake_multi_key",
            ]
            completed = subprocess.run(
                run_command,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            result = json.loads(completed.stdout)
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(Path(result["run_dir"]).resolve(), run_dir.resolve())

            replay_command = [
                sys.executable,
                "-m",
                "praxist.run",
                "replay",
                str(run_dir),
                "--mode",
                "verify",
            ]
            replay = subprocess.run(
                replay_command,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(replay.returncode, 0, replay.stderr + replay.stdout)
            replay_report = json.loads(replay.stdout)
            self.assertTrue(replay_report["success"], replay_report["errors"])

    def test_python_entrypoint_defaults_run_dir_to_external_task_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            tmp = Path(tmp_raw)
            task_path = _copy_example_task(tmp)
            env = _offline_subprocess_env()
            root_logs_before = {path.name for path in REPO_ROOT.glob("praxist_*.log")}

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "praxist.run",
                    "run",
                    "--task-path",
                    str(task_path),
                    "--workspace",
                    str(REPO_ROOT),
                    "--local",
                    "--credential-profile",
                    "fake_multi_key",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            result = json.loads(completed.stdout)
            run_dir = Path(result["run_dir"]).resolve()
            self.assertTrue(run_dir.is_relative_to((task_path / "experiments").resolve()))
            self.assertTrue((run_dir / "run.json").exists())
            self.assertEqual(
                root_logs_before, {path.name for path in REPO_ROOT.glob("praxist_*.log")}
            )


def _copy_example_task(tmp: Path) -> Path:
    source = REPO_ROOT / "templates" / "tasks" / "toy_math"
    destination = tmp / "external_tasks" / "toy_math"
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            ".pytest_cache",
            "experiments",
            "runs",
            "outputs",
        ),
    )
    return destination.resolve()


def _offline_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PRAXIST_BUNDLED_PLUGIN_ROOTS"] = str(FIXTURE_PLUGIN_ROOT)
    pythonpath = [str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    for key in API_KEY_ENV_NAMES:
        env.pop(key, None)
    return env


def _assert_required_run_files(testcase: unittest.TestCase, run_dir: Path) -> None:
    for rel in (
        "run.json",
        "startup_config.json",
        "task_project_manifest.json",
        "plugin_resolution.json",
        "credentials_redacted.json",
        "model_profiles.json",
        "cache_policy.json",
        "effective_task_spec.yaml",
        "trajectory.jsonl",
        "budget_ledger.jsonl",
        "artifact_index.jsonl",
        "findings/findings.jsonl",
        "findings/frontier.jsonl",
        "run_summary.json",
    ):
        testcase.assertTrue((run_dir / rel).exists(), rel)


def _assert_external_task_manifest(
    testcase: unittest.TestCase, run_dir: Path, task_path: Path
) -> None:
    manifest = json.loads((run_dir / "task_project_manifest.json").read_text(encoding="utf-8"))
    testcase.assertEqual(manifest["source"], "external_task_project")
    testcase.assertEqual(manifest["task_ref"], "task:toy_math")
    testcase.assertEqual(Path(manifest["path"]).resolve(), task_path.resolve())
    testcase.assertGreater(len(manifest["files"]), 0)
    testcase.assertTrue(any(item["path"] == "task.yaml" for item in manifest["files"]))

    startup = json.loads((run_dir / "startup_config.json").read_text(encoding="utf-8"))
    testcase.assertEqual(startup["canonical_args"]["task"], "task:toy_math")
    testcase.assertEqual(
        Path(startup["canonical_args"]["task_path"]).resolve(),
        task_path.resolve(),
    )


def _assert_plugin_resolution(testcase: unittest.TestCase, run_dir: Path) -> None:
    manifest = json.loads((run_dir / "plugin_resolution.json").read_text(encoding="utf-8"))
    selected_refs = {
        f"{item['metadata']['kind']}:{item['metadata']['name']}" for item in manifest["selected"]
    }
    for expected in (
        "workflow_stage:research_loop",
        "agent_runtime:fake_runtime",
        "model_provider:fake_provider",
        "budget_policy:fake_tiered",
        "panel_topology:fake_two_round",
        "role:fake_peer",
        "role:fake_pi",
        "role:fake_chair",
        "audit_rule:fake_panel_audit",
        "evaluation:fake_pareto",
    ):
        testcase.assertIn(expected, selected_refs)
    testcase.assertNotIn("task:toy_math", selected_refs)
    testcase.assertEqual(manifest["root_task_ref"], "task:toy_math")
    testcase.assertEqual(manifest["errors"], [])


def _assert_ledgers(testcase: unittest.TestCase, run_dir: Path) -> None:
    trajectory, trajectory_errors = read_jsonl(run_dir / "trajectory.jsonl")
    testcase.assertEqual(trajectory_errors, [])
    kinds = [event["kind"] for event in trajectory]
    for expected in (
        "run.started",
        "task.resolved",
        "plugins.resolved",
        "workflow.stage_started",
        "credential.failover",
        "finding.created",
        "frontier.promoted",
        "workflow.stage_succeeded",
        "run.finalized",
    ):
        testcase.assertIn(expected, kinds)

    findings, finding_errors = read_jsonl(run_dir / "findings" / "findings.jsonl")
    frontier, frontier_errors = read_jsonl(run_dir / "findings" / "frontier.jsonl")
    budget, budget_errors = read_jsonl(run_dir / "budget_ledger.jsonl")
    artifacts, artifact_errors = read_jsonl(run_dir / "artifact_index.jsonl")
    testcase.assertEqual(finding_errors + frontier_errors + budget_errors + artifact_errors, [])
    testcase.assertEqual(len(findings), 4)
    testcase.assertEqual(len(frontier), 1)
    testcase.assertTrue(
        any(record["kind"] == "decision" and record["decision"] == "grant" for record in budget)
    )
    testcase.assertTrue(any(record["kind"] == "usage" for record in budget))
    testcase.assertGreaterEqual(len(artifacts), 8)

    run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    testcase.assertEqual(run_summary["status"], "succeeded")
    testcase.assertEqual(run_summary["finding_summary"]["accepted"], 1)
    testcase.assertEqual(run_summary["finding_summary"]["retry_corrections"], 1)
    testcase.assertEqual(run_summary["frontier_records"], 1)
    testcase.assertEqual(run_summary["credential_mode"], "robust")


def _assert_no_real_secret_material(testcase: unittest.TestCase, run_dir: Path) -> None:
    scanned = [
        "run.json",
        "startup_config.json",
        "task_project_manifest.json",
        "credentials_redacted.json",
        "model_profiles.json",
        "trajectory.jsonl",
        "budget_ledger.jsonl",
        "artifact_index.jsonl",
        "findings/findings.jsonl",
        "findings/frontier.jsonl",
        "run_summary.json",
    ]
    joined = "\n".join((run_dir / rel).read_text(encoding="utf-8") for rel in scanned)
    for forbidden in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "sk-or-v1-", "sk-ant-"):
        testcase.assertNotIn(forbidden, joined)
