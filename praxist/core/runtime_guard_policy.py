"""Shared runtime guard policy for Praxist peer execution.

This module is the single source of truth for guard-related environment keys,
run-local resource state names, and trusted resource guard module identities.
Task-specific guard rules belong in generated task helpers, not here.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_GUARD_ENV_KEYS: tuple[str, ...] = (
    "PRAXIST_SAFE_DELETE_ROOTS",
    "PRAXIST_PEER_WORKSPACE",
    "PRAXIST_DELETE_GUARD_AGENT",
    "PRAXIST_DELETE_GUARD_ACTIVE",
    "PRAXIST_DELETE_GUARD_RUN_DIR",
    "PRAXIST_GUARD_WARNINGS_PATH",
    "PRAXIST_PROTECTED_CHILD_PATHS",
    "PRAXIST_TASK_WRITABLE_ROOTS",
    "PRAXIST_TASK_VENV",
    "BASH_ENV",
    "PYTHONPATH",
    "LD_PRELOAD",
    "LD_AUDIT",
    "LD_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
)

SHELL_WRAPPER_ENV_KEYS: tuple[str, ...] = ()

RESOURCE_GUARD_ENV_KEYS: tuple[str, ...] = (
    "PROTECTED_PIDS_DIR",
    "PRAXIST_MAX_PARALLEL_RUNS_PER_PEER",
    "GPU_GOVERNOR_DIR",
    "GPU_GOVERNOR_MAX_PER_GPU",
)

OPERATOR_ONLY_ENV_KEYS: tuple[str, ...] = ("BYPASS_GPU_GOVERNOR",)

TRUSTED_PROJECT_ENV_KEYS: tuple[str, ...] = (
    "PRAXIST_TASK_PROJECT_PATH",
    "PRAXIST_WORKSPACE_ROOT",
    "PRAXIST_TASK_PATH",
    "AUTO_RESEARCH_TASK_PATH",
    "TASK_PATH",
)

TRUSTED_PROJECT_EXTRA_ROOTS_ENV = "PRAXIST_TRUSTED_PROJECT_EXTRA_ROOTS"

PROTECTED_ROOT_ENV_KEYS: tuple[str, ...] = (
    "PRAXIST_DELETE_GUARD_RUN_DIR",
    "PRAXIST_RUN_DIR",
    "PRAXIST_TASK_PROJECT_PATH",
    "PRAXIST_WORKSPACE_ROOT",
)

RESOURCE_STATE_DIR_NAMES: tuple[str, ...] = ("process_governor", "protected_pids")

GUARD_WARNING_ENV_KEY = "PRAXIST_GUARD_WARNINGS_PATH"

TRUSTED_RESOURCE_GUARD_MODULES: tuple[str, ...] = (
    "praxist.plugins.workflow_stages.research_loop.backend.gpu_governor",
    "praxist.plugins.workflow_stages.research_loop.backend.protected_pids",
    "praxist.plugins.workflow_stages.research_loop.backend.run_lifecycle",
)

TRUSTED_RESOURCE_GUARD_MODULE_SUFFIXES: tuple[str, ...] = tuple(
    f"{module.replace('.', '/')}.py" for module in TRUSTED_RESOURCE_GUARD_MODULES
)

PYTHON_GUARD_ENV_KEYS: tuple[str, ...] = (
    *BASE_GUARD_ENV_KEYS,
    *RESOURCE_GUARD_ENV_KEYS,
    *OPERATOR_ONLY_ENV_KEYS,
    TRUSTED_PROJECT_EXTRA_ROOTS_ENV,
)

SHELL_GUARD_ENV_KEYS: tuple[str, ...] = (
    *PYTHON_GUARD_ENV_KEYS,
    *SHELL_WRAPPER_ENV_KEYS,
)

IMMUTABLE_GUARD_ENV_KEYS: tuple[str, ...] = (
    "PRAXIST_SAFE_DELETE_ROOTS",
    "PRAXIST_PEER_WORKSPACE",
    "PRAXIST_DELETE_GUARD_AGENT",
    "PRAXIST_DELETE_GUARD_RUN_DIR",
    "PRAXIST_PROTECTED_CHILD_PATHS",
    "PRAXIST_TASK_WRITABLE_ROOTS",
    "PRAXIST_TASK_VENV",
    "BASH_ENV",
    *RESOURCE_GUARD_ENV_KEYS,
    *OPERATOR_ONLY_ENV_KEYS,
    TRUSTED_PROJECT_EXTRA_ROOTS_ENV,
    "PYTHONPATH",
    "PATH",
    "LD_PRELOAD",
    "LD_AUDIT",
    "LD_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
)

HOST_LAUNCH_DROP_ENV_KEYS: tuple[str, ...] = (
    *OPERATOR_ONLY_ENV_KEYS,
    "CUDA_VISIBLE_DEVICES",
    "PROTECTED_PIDS_DIR",
    "PRAXIST_MAX_PARALLEL_RUNS_PER_PEER",
    "PRAXIST_RUN_DIR",
    "AUTO_RESEARCH_RUN_DIR",
    "LOCAL_STORE_DIR",
    "LOCAL_FINDINGS_DIR",
    "FRONTIER_DIR",
    "LOGS_DIR",
    "PEER_ID",
    "PRAXIST_PEER_ID",
    "GENERATION_ID",
)

HOST_LAUNCH_DROP_ENV_PREFIXES: tuple[str, ...] = ("GPU_GOVERNOR_",)


def split_path_list(raw: str | None) -> tuple[Path, ...]:
    """Parse an ``os.pathsep`` separated path list into expanded paths."""

    if not raw:
        return ()
    return tuple(Path(item).expanduser() for item in raw.split(os.pathsep) if item.strip())
