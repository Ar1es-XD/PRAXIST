"""Pin the pure builder extracted from ``configure_runtime_environment``.

#75 B-class consolidation: the per-run env contract is now produced by
``build_runtime_env_overrides`` (a pure function), and
``configure_runtime_environment`` is the thin writer over it. These
tests pin the dict shape directly so the contract is testable without
touching ``os.environ``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from praxist.plugins.workflow_stages.research_loop.backend import runtime_environment


def _task_spec(
    *,
    primary="score",
    direction="maximize",
    anchors=None,
    requires_tier=False,
    max_parallel_runs_per_peer=2,
    protected_child_paths=None,
    task_dir=None,
    launch_guard=None,
    eval_entrypoint="",
):
    spec = SimpleNamespace(
        evaluation=SimpleNamespace(
            primary_metric=primary,
            direction=direction,
            anchor_metrics=anchors or [],
            requires_tier=requires_tier,
            launch_guard=launch_guard or {},
        ),
        compute_budget=SimpleNamespace(max_parallel_runs_per_peer=max_parallel_runs_per_peer),
        toolchain=SimpleNamespace(eval_entrypoint=eval_entrypoint),
        runtime_environment=SimpleNamespace(
            protected_child_paths=protected_child_paths or [],
        ),
    )
    if task_dir is not None:
        spec._task_dir = Path(task_dir)
    return spec


class BuildRuntimeEnvOverridesTest(unittest.TestCase):
    def test_unspecified_peer_concurrency_does_not_invent_a_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, _ = runtime_environment.build_runtime_env_overrides(
                task_spec=_task_spec(max_parallel_runs_per_peer=None),
                run_dir=Path(tmp) / "run",
                findings_dir=Path(tmp) / "findings",
                local_mode=False,
            )

        self.assertNotIn("PRAXIST_MAX_PARALLEL_RUNS_PER_PEER", env)

    def test_dict_contains_every_subprocess_env_var_the_writer_used_to_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            findings_dir = Path(tmp) / "findings"
            env, anchors = runtime_environment.build_runtime_env_overrides(
                task_spec=_task_spec(),
                run_dir=run_dir,
                findings_dir=findings_dir,
                local_mode=True,
            )
        expected_keys = {
            "LOCAL_MODE",
            "PRIMARY_METRIC",
            "METRIC_DIRECTION",
            "ANCHOR_METRICS",
            "REQUIRES_TIER",
            "PRAXIST_RUN_DIR",
            "PRAXIST_RUN_ID",
            "AUTO_RESEARCH_RUN_DIR",
            "FRONTIER_DIR",
            "LOCAL_STORE_DIR",
            "LOCAL_FINDINGS_DIR",
            "PROTECTED_PIDS_DIR",
            "PRAXIST_MAX_PARALLEL_RUNS_PER_PEER",
            "PRAXIST_EXPERIMENT_SCHEDULER_CONFIG",
            "GPU_GOVERNOR_DIR",
            "PRAXIST_BASELINE_CACHE_DIR",
            "PRAXIST_LAUNCH_GUARD_ENABLED",
            "LOGS_DIR",
        }
        self.assertEqual(set(env), expected_keys)
        self.assertEqual(env["PRIMARY_METRIC"], "score")
        self.assertEqual(env["METRIC_DIRECTION"], "maximize")
        self.assertEqual(env["LOCAL_MODE"], "true")
        self.assertEqual(env["PRAXIST_RUN_DIR"], str(run_dir))
        self.assertEqual(env["PRAXIST_RUN_ID"], run_dir.name)
        self.assertEqual(env["AUTO_RESEARCH_RUN_DIR"], str(run_dir))
        self.assertEqual(env["FRONTIER_DIR"], str(run_dir / "frontier"))
        self.assertEqual(env["LOCAL_STORE_DIR"], str(run_dir))
        self.assertEqual(env["LOCAL_FINDINGS_DIR"], str(findings_dir))
        self.assertEqual(env["PRAXIST_MAX_PARALLEL_RUNS_PER_PEER"], "2")
        self.assertEqual(env["PRAXIST_BASELINE_CACHE_DIR"], str(run_dir / "baseline_cache"))
        self.assertEqual(env["PRAXIST_LAUNCH_GUARD_ENABLED"], "1")
        self.assertEqual(env["LOGS_DIR"], str(run_dir / "logs"))
        self.assertEqual(anchors, [])

    def test_launch_guard_freezes_new_work_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, _ = runtime_environment.build_runtime_env_overrides(
                task_spec=_task_spec(launch_guard={"enabled": True}),
                run_dir=Path(tmp) / "run",
                findings_dir=Path(tmp) / "f",
                local_mode=False,
            )
            env_legacy, _ = runtime_environment.build_runtime_env_overrides(
                task_spec=_task_spec(launch_guard={"enabled": True, "hard_enforcement": "false"}),
                run_dir=Path(tmp) / "run2",
                findings_dir=Path(tmp) / "f2",
                local_mode=False,
            )
            env_disabled, _ = runtime_environment.build_runtime_env_overrides(
                task_spec=_task_spec(launch_guard={"enabled": False}),
                run_dir=Path(tmp) / "run3",
                findings_dir=Path(tmp) / "f3",
                local_mode=False,
            )

        self.assertEqual(env["PRAXIST_LAUNCH_GUARD_ENABLED"], "1")
        self.assertEqual(env_legacy["PRAXIST_LAUNCH_GUARD_ENABLED"], "1")
        self.assertEqual(env_disabled["PRAXIST_LAUNCH_GUARD_ENABLED"], "0")

    def test_launch_guard_accepts_numeric_and_string_boolean_settings(self) -> None:
        """Task YAML may express launch_guard.enabled in legacy scalar forms."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            disabled_numeric, _ = runtime_environment.build_runtime_env_overrides(
                task_spec=_task_spec(launch_guard={"enabled": 0}),
                run_dir=root / "run-numeric",
                findings_dir=root / "findings-numeric",
                local_mode=False,
            )
            enabled_string, _ = runtime_environment.build_runtime_env_overrides(
                task_spec=_task_spec(launch_guard={"enabled": "yes"}),
                run_dir=root / "run-string",
                findings_dir=root / "findings-string",
                local_mode=False,
            )
            disabled_string, _ = runtime_environment.build_runtime_env_overrides(
                task_spec=_task_spec(launch_guard={"enabled": "off"}),
                run_dir=root / "run-string-disabled",
                findings_dir=root / "findings-string-disabled",
                local_mode=False,
            )

        self.assertEqual(disabled_numeric["PRAXIST_LAUNCH_GUARD_ENABLED"], "0")
        self.assertEqual(enabled_string["PRAXIST_LAUNCH_GUARD_ENABLED"], "1")
        self.assertEqual(disabled_string["PRAXIST_LAUNCH_GUARD_ENABLED"], "0")

    def test_evaluation_entrypoint_is_forwarded_to_peer_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, _ = runtime_environment.build_runtime_env_overrides(
                task_spec=_task_spec(eval_entrypoint="evaluations/generic/run.py"),
                run_dir=Path(tmp) / "run",
                findings_dir=Path(tmp) / "findings",
                local_mode=False,
            )
        self.assertEqual(env["PRAXIST_EVALUATION_ENTRYPOINT"], "evaluations/generic/run.py")

    def test_blank_protected_child_paths_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "task"
            task_dir.mkdir()
            env, _ = runtime_environment.build_runtime_env_overrides(
                task_spec=_task_spec(
                    protected_child_paths=["  ", "child.py"],
                    task_dir=task_dir,
                ),
                run_dir=Path(tmp) / "run",
                findings_dir=Path(tmp) / "findings",
                local_mode=False,
            )

        self.assertEqual(env["PRAXIST_PROTECTED_CHILD_PATHS"], str(task_dir / "child.py"))

    def test_local_mode_off_omits_LOCAL_MODE_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, _ = runtime_environment.build_runtime_env_overrides(
                task_spec=_task_spec(),
                run_dir=Path(tmp) / "run",
                findings_dir=Path(tmp) / "f",
                local_mode=False,
            )
        self.assertNotIn("LOCAL_MODE", env)

    def test_anchor_metrics_payload_round_trips_through_env_string(self) -> None:
        anchors_in = [
            {"name": "score", "direction": "maximize"},
            ("loss", "minimize"),
            {"name": "bad", "direction": "invalid"},  # direction falls back to maximize
        ]
        with tempfile.TemporaryDirectory() as tmp:
            env, anchors = runtime_environment.build_runtime_env_overrides(
                task_spec=_task_spec(anchors=anchors_in),
                run_dir=Path(tmp) / "run",
                findings_dir=Path(tmp) / "f",
                local_mode=True,
            )
        self.assertEqual(
            anchors,
            [
                {"name": "score", "direction": "maximize"},
                {"name": "loss", "direction": "minimize"},
                {"name": "bad", "direction": "maximize"},
            ],
        )
        self.assertEqual(json.loads(env["ANCHOR_METRICS"]), anchors)

    def test_requires_tier_serialized_as_lower_case_bool_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_t, _ = runtime_environment.build_runtime_env_overrides(
                task_spec=_task_spec(requires_tier=True),
                run_dir=Path(tmp) / "run",
                findings_dir=Path(tmp) / "f",
                local_mode=False,
            )
            env_f, _ = runtime_environment.build_runtime_env_overrides(
                task_spec=_task_spec(requires_tier=False),
                run_dir=Path(tmp) / "run",
                findings_dir=Path(tmp) / "f",
                local_mode=False,
            )
        self.assertEqual(env_t["REQUIRES_TIER"], "true")
        self.assertEqual(env_f["REQUIRES_TIER"], "false")

    def test_protected_child_paths_are_serialized_under_task_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            task_root = Path(tmp) / "task"
            (task_root / "child").mkdir(parents=True)
            env, _ = runtime_environment.build_runtime_env_overrides(
                task_spec=_task_spec(
                    protected_child_paths=["child", "../escape"],
                    task_dir=task_root,
                ),
                run_dir=run_dir,
                findings_dir=Path(tmp) / "f",
                local_mode=False,
            )
        self.assertEqual(env["PRAXIST_PROTECTED_CHILD_PATHS"], str(task_root / "child"))

    def test_protected_child_paths_without_task_root_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, _ = runtime_environment.build_runtime_env_overrides(
                task_spec=_task_spec(protected_child_paths=["child"]),
                run_dir=Path(tmp) / "run",
                findings_dir=Path(tmp) / "f",
                local_mode=False,
            )

        self.assertNotIn("PRAXIST_PROTECTED_CHILD_PATHS", env)

    def test_configure_runtime_environment_clears_stale_protected_paths(self) -> None:
        import os

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            run_dir = Path(tmp) / "run"
            findings_dir = Path(tmp) / "f"
            os.environ["PRAXIST_PROTECTED_CHILD_PATHS"] = "/stale"
            runtime_environment.configure_runtime_environment(
                task_spec=_task_spec(),
                run_dir=run_dir,
                findings_dir=findings_dir,
                local_mode=False,
            )
        self.assertNotIn("PRAXIST_PROTECTED_CHILD_PATHS", os.environ)

    def test_legacy_claude_bridge_forwards_protected_paths(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.agent import (
            _legacy_runtime_env_keys,
        )

        self.assertIn("PRAXIST_PROTECTED_CHILD_PATHS", _legacy_runtime_env_keys())
        self.assertIn("PRAXIST_TASK_WRITABLE_ROOTS", _legacy_runtime_env_keys())

    def test_pure_builder_does_not_touch_os_environ(self) -> None:
        """Sanity: calling the builder must not pollute the parent env."""
        import os

        marker = "PRAXIST_BUILDER_PURITY_PROBE_DO_NOT_SET"
        snapshot_before = dict(os.environ)
        snapshot_before.pop(marker, None)
        with tempfile.TemporaryDirectory() as tmp:
            runtime_environment.build_runtime_env_overrides(
                task_spec=_task_spec(),
                run_dir=Path(tmp) / "run",
                findings_dir=Path(tmp) / "f",
                local_mode=True,
            )
        snapshot_after = dict(os.environ)
        snapshot_after.pop(marker, None)
        self.assertEqual(snapshot_before, snapshot_after)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
