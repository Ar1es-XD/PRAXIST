"""Runtime environment setup for research-loop child tools and peers."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from praxist.core.runtime_guard_policy import (
    OPERATOR_ONLY_ENV_KEYS,
    RESOURCE_STATE_DIR_NAMES,
)
from praxist.task_spec import resolve_declared_evaluation_entrypoint

logger = logging.getLogger(__name__)


def task_process_identity_env(
    *, evaluation_entrypoint: str, task_dir: Path, runtime_cwd: object = None
) -> dict[str, str]:
    """Return runner and statically resolved evaluator process identities."""

    env = {"PRAXIST_RUNNER_PYTHON": os.path.abspath(sys.executable)}
    evaluator = resolve_declared_evaluation_entrypoint(
        evaluation_entrypoint,
        task_dir=task_dir,
        runtime_cwd=runtime_cwd,
    )
    if evaluator is not None:
        env["PRAXIST_EVALUATION_ENTRYPOINT_PATH"] = str(evaluator)
    return env


def _truthy(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off", ""}:
            return False
    return default


def _anchor_metrics_payload(task_spec: Any) -> list[dict[str, str]]:
    anchors = getattr(task_spec.evaluation, "anchor_metrics", None) or []
    payload: list[dict[str, str]] = []
    for anchor in anchors:
        name = None
        direction = "maximize"
        if isinstance(anchor, dict):
            name = anchor.get("name")
            raw_direction = anchor.get("direction", "maximize")
            direction = raw_direction if raw_direction in ("maximize", "minimize") else "maximize"
        elif isinstance(anchor, (list, tuple)) and len(anchor) >= 1:
            name = anchor[0]
            if len(anchor) >= 2 and anchor[1] in ("maximize", "minimize"):
                direction = anchor[1]
        if name:
            payload.append({"name": str(name), "direction": direction})
    return payload


def _protected_child_paths_payload(task_spec: Any) -> str:
    runtime_cfg = getattr(task_spec, "runtime_environment", None)
    raw_paths = getattr(runtime_cfg, "protected_child_paths", None) or []
    raw_task_dir = getattr(task_spec, "_task_dir", None) or getattr(task_spec, "task_dir", None)
    if raw_paths and not raw_task_dir:
        logger.warning(
            "runtime_environment.protected_child_paths ignored because task root is unavailable."
        )
        return ""
    task_dir = Path(raw_task_dir).expanduser() if raw_task_dir else None
    resolved_task_dir = task_dir.resolve(strict=False) if task_dir is not None else None
    paths: list[str] = []
    for raw in raw_paths:
        text = str(raw).strip()
        if not text:
            continue
        path = Path(text).expanduser()
        if not path.is_absolute():
            if task_dir is None:
                continue
            path = task_dir / path
        try:
            resolved = path.resolve(strict=False)
            if resolved_task_dir is not None:
                resolved.relative_to(resolved_task_dir)
        except (OSError, ValueError):
            logger.warning(
                "runtime_environment.protected_child_paths entry escapes task root: %s",
                raw,
            )
            continue
        paths.append(str(path))
    return os.pathsep.join(dict.fromkeys(paths))


def build_runtime_env_overrides(
    *,
    task_spec: Any,
    run_dir: Path,
    findings_dir: Path,
    local_mode: bool,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Compute the per-run env overrides without touching ``os.environ``.

    Returns ``(env_overrides, anchor_payload)``.

    Separating the dict-build step from the os.environ-mutation step
    (#75 B-class consolidation) lets callers that *don't* want to
    pollute the parent process env (tests, future hermetic subprocess
    spawners) materialize the contract as a plain dict. ``configure_runtime_environment``
    is now a thin writer over this builder.
    """
    env: dict[str, str] = {}
    if local_mode:
        env["LOCAL_MODE"] = "true"
    env["PRIMARY_METRIC"] = task_spec.evaluation.primary_metric
    env["METRIC_DIRECTION"] = task_spec.evaluation.direction

    anchor_payload = _anchor_metrics_payload(task_spec)
    env["ANCHOR_METRICS"] = json.dumps(anchor_payload)
    env["REQUIRES_TIER"] = (
        "true" if getattr(task_spec.evaluation, "requires_tier", False) else "false"
    )

    env["PRAXIST_RUN_DIR"] = str(run_dir)
    env["PRAXIST_RUN_ID"] = run_dir.name
    env["AUTO_RESEARCH_RUN_DIR"] = str(run_dir)
    env["FRONTIER_DIR"] = str(run_dir / "frontier")
    env["LOCAL_STORE_DIR"] = str(run_dir)
    env["LOCAL_FINDINGS_DIR"] = str(findings_dir)
    process_governor_dir, protected_pids_dir = RESOURCE_STATE_DIR_NAMES
    env["PROTECTED_PIDS_DIR"] = str(run_dir / protected_pids_dir)
    launch_guard = getattr(getattr(task_spec, "evaluation", None), "launch_guard", {})
    launch_guard_enabled = True
    if isinstance(launch_guard, dict):
        launch_guard_enabled = _truthy(launch_guard.get("enabled"), True)
    env["PRAXIST_LAUNCH_GUARD_ENABLED"] = "1" if launch_guard_enabled else "0"
    toolchain = getattr(task_spec, "toolchain", None)
    evaluation_entrypoint = str(getattr(toolchain, "eval_entrypoint", "") or "").strip()
    if evaluation_entrypoint:
        env["PRAXIST_EVALUATION_ENTRYPOINT"] = evaluation_entrypoint
    compute_budget = getattr(task_spec, "compute_budget", None)
    max_parallel = getattr(compute_budget, "max_parallel_runs_per_peer", None)
    if max_parallel is not None:
        env["PRAXIST_MAX_PARALLEL_RUNS_PER_PEER"] = str(max(1, int(max_parallel)))
    scheduler_config = getattr(compute_budget, "resource_scheduler", {}) or {}
    if isinstance(scheduler_config, dict):
        env["PRAXIST_EXPERIMENT_SCHEDULER_CONFIG"] = json.dumps(scheduler_config, sort_keys=True)
    env["GPU_GOVERNOR_DIR"] = str(run_dir / process_governor_dir)
    env["PRAXIST_BASELINE_CACHE_DIR"] = str(run_dir / "baseline_cache")
    env["LOGS_DIR"] = str(run_dir / "logs")
    protected_child_paths = _protected_child_paths_payload(task_spec)
    if protected_child_paths:
        env["PRAXIST_PROTECTED_CHILD_PATHS"] = protected_child_paths
    return env, anchor_payload


def configure_runtime_environment(
    *,
    task_spec: Any,
    run_dir: Path,
    findings_dir: Path,
    local_mode: bool,
) -> list[dict[str, str]]:
    """Populate non-secret environment variables consumed by backend tools.

    Thin writer over :func:`build_runtime_env_overrides` (#75 B-class).
    Side effects beyond ``os.environ`` writes — the GPU-governor pointer
    file and the ``BYPASS_GPU_GOVERNOR`` clear — stay here because they
    are state the parent process owns, not subprocess-env contract.
    """
    env_overrides, anchor_payload = build_runtime_env_overrides(
        task_spec=task_spec,
        run_dir=run_dir,
        findings_dir=findings_dir,
        local_mode=local_mode,
    )
    # ``pop`` first so previous-run values do not survive when the new
    # task omits an optional runtime payload.
    os.environ.pop("ANCHOR_METRICS", None)
    os.environ.pop("PRAXIST_PROTECTED_CHILD_PATHS", None)
    os.environ.pop("PRAXIST_EXPERIMENT_SCHEDULER_ENDPOINT", None)
    for key, value in env_overrides.items():
        os.environ[key] = value
    if anchor_payload:
        logger.info(
            "generation_loop: ANCHOR_METRICS env set with %d axes (Pareto leaderboard active): %s",
            len(anchor_payload),
            [a["name"] for a in anchor_payload],
        )

    os.environ["FRONTIER_DIR"] = str(run_dir / "frontier")
    os.environ["LOCAL_STORE_DIR"] = str(run_dir)
    os.environ["LOCAL_FINDINGS_DIR"] = str(findings_dir)
    os.environ["AUTO_RESEARCH_RUN_DIR"] = str(run_dir)
    process_governor_dir, protected_pids_dir = RESOURCE_STATE_DIR_NAMES
    os.environ["PROTECTED_PIDS_DIR"] = str(run_dir / protected_pids_dir)
    os.environ["GPU_GOVERNOR_DIR"] = str(run_dir / process_governor_dir)
    os.environ["PRAXIST_BASELINE_CACHE_DIR"] = str(run_dir / "baseline_cache")
    for key in OPERATOR_ONLY_ENV_KEYS:
        if os.environ.pop(key, None) is not None:
            logger.warning(
                "generation_loop: cleared operator-only runtime env var %s "
                "inherited from outer shell — peers will now respect run guards.",
                key,
            )

    try:
        uid = os.getuid()
    except AttributeError:  # pragma: no cover - non-Unix fallback.
        uid = 0
    pointer_path = Path(f"/tmp/praxist_active_governor_uid{uid}")
    try:
        tmp = pointer_path.with_suffix(pointer_path.suffix + ".tmp")
        tmp.write_text(str(run_dir / process_governor_dir) + "\n", encoding="utf-8")
        os.replace(tmp, pointer_path)
        os.environ["GPU_GOVERNOR_POINTER_FILE"] = str(pointer_path)
        logger.info("generation_loop: wrote governor pointer file %s", pointer_path)
    except OSError as e:
        logger.warning("generation_loop: could not write governor pointer file: %s", e)

    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return anchor_payload


def initialize_local_store_if_needed(*, local_mode: bool) -> None:
    """Initialize the local SQLite store when the workflow runs in local mode."""
    if not local_mode:
        return
    try:
        from praxist.plugins.workflow_stages.research_loop.backend.tools.local_store import (
            init_db,
        )

        init_db()
    except Exception as e:  # noqa: BLE001 - local store failures degrade to warnings.
        logger.warning("Could not init local store: %s", e)
