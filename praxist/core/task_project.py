"""Task project loading for explicit, external research tasks.

Task projects are not bundled Praxist plugins.  They are user-owned project
directories selected with ``--task-path`` and may contain task-local roles,
audits, evaluations, harness assets, and optional reference implementations.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from praxist.core.registry import ALL_KINDS, PluginRef

TASK_LOCAL_KINDS = frozenset(
    {
        "task_role",
        "task_audit",
        "task_evaluation",
        "task_budget_profile",
    }
)

_SKIPPED_MANIFEST_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
}

_SKIPPED_MANIFEST_TOP_LEVEL_DIRS = {
    "data",
    "datasets",
    "experiments",
    "experiments_tracking",
    "frontier",
    "logs",
    "research_memory",
    "results",
    "runs",
    "outputs",
    "shared_findings",
    "variants",
}

_SKIPPED_MANIFEST_RELATIVE_DIRS = {
    ("docs", "praxist_reports"),
}


@dataclass(frozen=True)
class TaskProject:
    """Resolved task project selected by an explicit filesystem path."""

    path: Path
    task_id: str
    task_ref: str
    descriptor_path: Path
    descriptor: dict[str, Any]
    manifest: dict[str, Any]


def resolve_task_project(task_path: str | Path, workspace: str | Path | None = None) -> TaskProject:
    """Load and fingerprint a task project directory.

    ``task_path`` may point at either the project directory or its ``task.yaml``.
    Relative paths are resolved against ``workspace`` when provided.
    """

    root = Path(task_path).expanduser()
    if not root.is_absolute() and workspace is not None:
        root = Path(workspace).expanduser().resolve() / root
    root = root.resolve()
    if root.is_file():
        if root.name != "task.yaml":
            raise ValueError(f"task_path file must be task.yaml: {root}")
        descriptor_path = root
        root = root.parent
    else:
        descriptor_path = root / "task.yaml"
    if not descriptor_path.exists():
        raise FileNotFoundError(f"task project descriptor not found: {descriptor_path}")
    descriptor = _read_yaml(descriptor_path)
    task_id = _task_id_from_descriptor(root, descriptor)
    task_ref = str(descriptor.get("praxist_plugins", {}).get("task_ref") or f"task:{task_id}")
    if not task_ref.startswith("task:"):
        raise ValueError(f"task_ref must use task:<name>: {task_ref}")
    manifest = build_task_project_manifest(
        root, descriptor_path=descriptor_path, descriptor=descriptor
    )
    return TaskProject(
        path=root,
        task_id=task_id,
        task_ref=task_ref,
        descriptor_path=descriptor_path,
        descriptor=descriptor,
        manifest=manifest,
    )


def task_project_has_capability(project: TaskProject, capability: str) -> bool:
    """Return whether a resolved task project declares a given capability."""
    capabilities = set(project.descriptor.get("capabilities") or [])
    capabilities.update(project.descriptor.get("praxist_plugins", {}).get("capabilities") or [])
    return capability in capabilities


def task_project_global_plugin_refs(descriptor: dict[str, Any]) -> list[PluginRef]:
    """Return bundled plugin refs required by a task descriptor.

    Task-local refs such as ``task_role:peer`` remain inside the task project and
    are intentionally omitted from bundled plugin resolution.
    """

    refs: list[PluginRef] = []
    praxist_plugins = descriptor.get("praxist_plugins", {}) or {}
    panel = praxist_plugins.get("panel", {}) or {}
    optional = praxist_plugins.get("optional_workflow_stages", {}) or {}
    tool_servers = praxist_plugins.get("tool_servers", {}) or {}

    workflow = praxist_plugins.get("workflow") or {}
    _append_ref(refs, praxist_plugins.get("workflow_stage"))
    if isinstance(workflow, dict):
        _append_ref(
            refs, workflow.get("stage") or workflow.get("ref") or workflow.get("workflow_stage")
        )
    _append_ref(refs, praxist_plugins.get("agent_runtime"))
    _append_ref(refs, praxist_plugins.get("model_provider"))
    _append_ref(refs, praxist_plugins.get("budget_policy"))
    _append_ref(refs, praxist_plugins.get("graph_maintainer"))
    _append_ref(refs, panel.get("topology"))

    for role in panel.get("roles", []) or []:
        _append_ref(
            refs,
            (role.get("role_ref") or role.get("role")) if isinstance(role, dict) else role,
        )
    for rule in list(panel.get("audit_rules", []) or []) + list(
        praxist_plugins.get("audit_rules", []) or []
    ):
        _append_ref(refs, rule)
    for evaluation in list(panel.get("evaluations", []) or []) + list(
        praxist_plugins.get("evaluations", []) or []
    ):
        _append_ref(refs, evaluation)
    for tool in praxist_plugins.get("tools", []) or []:
        _append_ref(refs, tool)
    for graph_maintainer in praxist_plugins.get("graph_maintainers", []) or []:
        _append_ref(refs, graph_maintainer)

    for spec in optional.values():
        if isinstance(spec, str):
            _append_ref(refs, spec)
        elif isinstance(spec, dict) and spec.get("enabled", True):
            _append_ref(refs, spec.get("ref"))

    for spec in tool_servers.values():
        if isinstance(spec, str):
            _append_ref(refs, spec)
        elif isinstance(spec, dict) and spec.get("enabled", True):
            _append_ref(refs, spec.get("ref"))

    return _dedupe_refs(refs)


def is_task_local_ref(ref: str | None) -> bool:
    """Return whether a reference is scoped to the explicit task project rather than global plugins."""
    if not ref or ":" not in ref:
        return False
    kind, _ = ref.split(":", 1)
    return kind in TASK_LOCAL_KINDS


def build_task_project_manifest(
    root: Path,
    *,
    descriptor_path: Path | None = None,
    descriptor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Hash a task project descriptor and source files into a replayable manifest."""
    root = Path(root).resolve()
    descriptor_path = (descriptor_path or root / "task.yaml").resolve()
    descriptor = descriptor if descriptor is not None else _read_yaml(descriptor_path)
    files = []
    digest = hashlib.sha256()
    for path in _iter_project_files(root):
        rel = path.relative_to(root).as_posix()
        content = path.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()
        files.append({"path": rel, "sha256": file_hash, "bytes": len(content)})
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    task_id = _task_id_from_descriptor(root, descriptor)
    return {
        "schema_version": "task_project_manifest.v1",
        "source": "external_task_project",
        "task_id": task_id,
        "task_ref": str(descriptor.get("praxist_plugins", {}).get("task_ref") or f"task:{task_id}"),
        "path": str(root),
        "descriptor_path": str(descriptor_path),
        "sha256": digest.hexdigest(),
        "files": files,
    }


def load_task_project_runner(project: TaskProject) -> Any:
    """Import the task-local runner factory declared by a task project descriptor."""
    runner_spec = project.descriptor.get("runner", {}) or {}
    entrypoint = runner_spec.get("entrypoint")
    if not entrypoint:
        raise ValueError(
            f"task project {project.path} declares no runner.entrypoint for fixture execution"
        )
    module_name, _, factory_name = entrypoint.partition(":")
    if not module_name or not factory_name:
        raise ValueError(f"runner.entrypoint must use module:function syntax: {entrypoint}")
    module = _load_project_module(project.path, module_name)
    factory = getattr(module, factory_name)
    return factory(project)


def write_task_project_manifest(run_dir: str | Path, project: TaskProject) -> Path:
    """Persist a resolved task project manifest into the run directory."""
    import json

    path = Path(run_dir) / "task_project_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(project.manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _load_project_module(root: Path, module_name: str) -> ModuleType:
    root = Path(root).resolve()
    rel = Path(*module_name.split(".")).with_suffix(".py")
    module_path = (root / rel).resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"task runner module not found: {module_path}")
    try:
        module_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"task runner module escapes task project: {module_path}") from exc
    synthetic_name = (
        f"_praxist_task_project_{hashlib.sha1(str(root).encode()).hexdigest()}_{module_name}"
    )
    spec = importlib.util.spec_from_file_location(synthetic_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import task runner module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[synthetic_name] = module
    old_path = list(sys.path)
    sys.path.insert(0, str(root))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML document must be a mapping: {path}")
    return data


def _task_id_from_descriptor(root: Path, descriptor: dict[str, Any]) -> str:
    explicit = descriptor.get("task_id") or descriptor.get("id")
    if explicit:
        return str(explicit)
    task_ref = descriptor.get("praxist_plugins", {}).get("task_ref")
    if isinstance(task_ref, str) and task_ref.startswith("task:"):
        return task_ref.split(":", 1)[1]
    return root.name


def _append_ref(refs: list[PluginRef], ref: Any) -> None:
    if not isinstance(ref, str) or ":" not in ref:
        return
    kind, name = ref.split(":", 1)
    if kind in TASK_LOCAL_KINDS:
        return
    if kind not in ALL_KINDS:
        return
    refs.append(PluginRef(kind=kind, name=name))


def _dedupe_refs(refs: list[PluginRef]) -> list[PluginRef]:
    seen: set[tuple[str, str]] = set()
    out: list[PluginRef] = []
    for ref in refs:
        key = (ref.kind, ref.name)
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def _iter_project_files(root: Path):
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current = Path(dirpath)
        rel_dir = current.relative_to(root)
        rel_dir_parts = () if rel_dir == Path(".") else rel_dir.parts

        pruned_dirs: list[str] = []
        for dirname in sorted(dirnames):
            child_parts = (*rel_dir_parts, dirname)
            if dirname in _SKIPPED_MANIFEST_DIR_NAMES:
                continue
            if len(child_parts) == 1 and dirname in _SKIPPED_MANIFEST_TOP_LEVEL_DIRS:
                continue
            if child_parts in _SKIPPED_MANIFEST_RELATIVE_DIRS:
                continue
            child_path = current / dirname
            if child_path.is_symlink():
                continue
            pruned_dirs.append(dirname)
        dirnames[:] = pruned_dirs

        for filename in sorted(filenames):
            path = current / filename
            if path.is_symlink() or not path.is_file():
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                continue
            yield path
