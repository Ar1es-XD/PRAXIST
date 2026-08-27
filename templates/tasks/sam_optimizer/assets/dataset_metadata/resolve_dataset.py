"""Dataset root resolver for the SAM optimizer task project.

The task project carries only lightweight metadata. CIFAR and Tiny-ImageNet
data stay outside the task source tree and are located explicitly at run
time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


ENV_KEYS = (
    "PRAXIST_SAM_DATA_DIR",
    "SAM_DATA_DIR",
    "PRAXIST_DATA_DIR",
)

ROOT_ENV_KEYS = (
    "PRAXIST_DATASETS_DIR",
    "PRAXIST_DATA_ROOT",
)


def resolve_dataset_root(cli_value: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the dataset root without silently importing legacy task code.

    Priority:
      1. Explicit CLI value.
      2. Task-scoped or generic data env vars that point directly at the
         SAM dataset root.
      3. Generic dataset-root env vars, under ``sam_optimizer/``.
      4. ``./data/sam_optimizer`` for standalone use; downstream loaders fail
         clearly if required datasets are absent.
    """

    explicit = _clean(cli_value)
    if explicit:
        return Path(explicit).expanduser()

    for key in ENV_KEYS:
        value = _clean(os.environ.get(key))
        if value:
            return Path(value).expanduser()

    for key in ROOT_ENV_KEYS:
        value = _clean(os.environ.get(key))
        if value:
            return Path(value).expanduser() / "sam_optimizer"

    return Path("./data/sam_optimizer")


def resolver_metadata(root: Path | str | None = None) -> dict[str, object]:
    """Return a small provenance payload suitable for logs or artifacts."""

    resolved = Path(root) if root is not None else resolve_dataset_root()
    return {
        "schema_version": "praxist.sam_dataset_resolver.v1",
        "resolved_root": str(resolved),
        "exists": resolved.exists(),
        "env_priority": list(ENV_KEYS),
        "dataset_root_env_priority": list(ROOT_ENV_KEYS),
        "raw_dataset_packaged_with_task": False,
        "legacy_task_data_fallback": False,
    }


def write_metadata(path: Path | str, root: Path | str | None = None) -> None:
    """Write resolved dataset metadata for this task template."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(resolver_metadata(root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


__all__ = ["ENV_KEYS", "ROOT_ENV_KEYS", "resolve_dataset_root", "resolver_metadata", "write_metadata"]
