"""RoleSkill loading for declarative role plugins."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from praxist.core.registry import (
    PluginLoader,
    PluginRef,
    PluginRegistry,
    PluginRoots,
    SelectedPlugin,
)


@dataclass(frozen=True)
class RoleSkill:
    """Loaded role prompt contract, including skill markdown, tool scope, private KB, and content hash."""

    role_ref: str
    role_id: str
    display_name: str
    role_kind: str
    skill_markdown: str
    output_schema_ref: str
    default_model_profile_ref: str | None
    tool_scope: tuple[str, ...]
    fixed_questions: tuple[str, ...]
    private_kb_paths: tuple[Path, ...]
    plugin_path: Path
    content_hash: str
    legacy_role_id: str | None = None

    def to_prompt_context(self) -> dict[str, Any]:
        return {
            "role_ref": self.role_ref,
            "role_id": self.role_id,
            "legacy_role_id": self.legacy_role_id,
            "display_name": self.display_name,
            "role_kind": self.role_kind,
            "skill_markdown": self.skill_markdown,
            "output_schema_ref": self.output_schema_ref,
            "default_model_profile_ref": self.default_model_profile_ref,
            "tool_scope": list(self.tool_scope),
            "fixed_questions": list(self.fixed_questions),
            "private_kb_paths": [str(path) for path in self.private_kb_paths],
            "plugin_path": str(self.plugin_path),
            "content_hash": self.content_hash,
        }


def load_role_skill(
    role_ref: str,
    *,
    registry: PluginRegistry | None = None,
    workspace: Path | None = None,
    task_project_path: Path | None = None,
) -> RoleSkill:
    """Load a bundled role plugin or task-local role skill for prompt assembly."""
    if role_ref.startswith("task_role:"):
        return _load_task_role_skill(
            role_ref,
            workspace=workspace,
            task_project_path=task_project_path,
        )
    selected = _selected_role_plugin(role_ref, registry=registry, workspace=workspace)
    plugin_path = Path(selected.path)
    manifest = _read_manifest(plugin_path / "plugin.yaml")
    role_config = manifest.get("role") if isinstance(manifest.get("role"), dict) else {}
    skill_path = plugin_path / "skill.md"
    if not skill_path.exists():
        raise ValueError(f"{role_ref} is missing skill.md")
    skill_markdown = skill_path.read_text(encoding="utf-8")
    private_kb_paths = _private_kb_paths(plugin_path, manifest, role_config)
    fixed_questions = _fixed_questions(role_config, skill_markdown)
    role_id = str(
        role_config.get("role_id") or role_config.get("legacy_role_id") or selected.metadata.name
    )
    return RoleSkill(
        role_ref=role_ref,
        role_id=role_id,
        display_name=str(role_config.get("display_name") or _title_from_role_id(role_id)),
        role_kind=str(role_config.get("role_kind") or "generic"),
        skill_markdown=skill_markdown,
        output_schema_ref=str(role_config.get("output_schema_ref") or "core:role_output.v1"),
        default_model_profile_ref=role_config.get("default_model_profile_ref"),
        tool_scope=tuple(str(item) for item in role_config.get("tool_scope") or []),
        fixed_questions=tuple(fixed_questions),
        private_kb_paths=tuple(private_kb_paths),
        plugin_path=plugin_path,
        content_hash=selected.content_hash,
        legacy_role_id=str(role_config.get("legacy_role_id") or role_id),
    )


def _load_task_role_skill(
    role_ref: str,
    *,
    workspace: Path | None,
    task_project_path: Path | None,
) -> RoleSkill:
    _, role_name = role_ref.split(":", 1)
    _validate_task_role_name(role_name)
    task_root = _resolve_task_project_path(workspace=workspace, task_project_path=task_project_path)
    roles_root = (task_root / "roles").resolve()
    role_path = (roles_root / role_name).resolve()
    if not role_path.is_relative_to(roles_root):
        raise ValueError(f"{role_ref} escapes task project roles directory")
    if not role_path.is_dir():
        raise ValueError(f"{role_ref} not found in task project roles: {role_path}")
    config_path = role_path / "role.yaml"
    if not config_path.exists():
        config_path = role_path / "plugin.yaml"
    manifest = _read_manifest(config_path) if config_path.exists() else {}
    role_config = manifest.get("role") if isinstance(manifest.get("role"), dict) else manifest
    skill_path = role_path / "skill.md"
    if not skill_path.exists():
        raise ValueError(f"{role_ref} is missing skill.md")
    skill_markdown = skill_path.read_text(encoding="utf-8")
    private_kb_paths = _private_kb_paths(role_path, manifest, role_config)
    fixed_questions = _fixed_questions(role_config, skill_markdown)
    role_id = str(role_config.get("role_id") or role_config.get("legacy_role_id") or role_name)
    return RoleSkill(
        role_ref=role_ref,
        role_id=role_id,
        display_name=str(role_config.get("display_name") or _title_from_role_id(role_id)),
        role_kind=str(role_config.get("role_kind") or "generic"),
        skill_markdown=skill_markdown,
        output_schema_ref=str(role_config.get("output_schema_ref") or "core:role_output.v1"),
        default_model_profile_ref=role_config.get("default_model_profile_ref"),
        tool_scope=tuple(str(item) for item in role_config.get("tool_scope") or []),
        fixed_questions=tuple(fixed_questions),
        private_kb_paths=tuple(private_kb_paths),
        plugin_path=role_path,
        content_hash=_content_hash([config_path, skill_path, *private_kb_paths]),
        legacy_role_id=str(role_config.get("legacy_role_id") or role_id),
    )


def _validate_task_role_name(role_name: str) -> None:
    path = Path(role_name)
    if (
        not role_name
        or path.is_absolute()
        or len(path.parts) != 1
        or any(part in {"", ".."} for part in path.parts)
        or "/" in role_name
        or "\\" in role_name
    ):
        raise ValueError(f"task_role name must be a single role directory name: {role_name!r}")


def _resolve_task_project_path(
    *,
    workspace: Path | None,
    task_project_path: Path | None,
) -> Path:
    configured = task_project_path or os.environ.get("PRAXIST_TASK_PROJECT_PATH")
    if configured is None:
        raise ValueError("task_role refs require task_project_path or PRAXIST_TASK_PROJECT_PATH")
    root = Path(configured).expanduser()
    if not root.is_absolute() and workspace is not None:
        root = Path(workspace).expanduser().resolve() / root
    return root.resolve()


def _content_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({p.resolve() for p in paths if p.exists()}):
        digest.update(str(path.name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _selected_role_plugin(
    role_ref: str,
    *,
    registry: PluginRegistry | None,
    workspace: Path | None,
) -> SelectedPlugin:
    parsed = PluginRef.parse(role_ref)
    if parsed.kind != "role":
        raise ValueError(f"RoleSkill requires role:<name>, got {role_ref}")
    if registry is not None:
        selected = registry.descriptor(parsed.kind, parsed.name)
    else:
        loader = PluginLoader(PluginRoots.defaults(workspace))
        manifest = loader.resolve(
            [role_ref], root_task_ref=role_ref, enforce_bundled_execution=True
        )
        selected = loader.load(manifest).descriptor(parsed.kind, parsed.name)
    if selected.metadata.kind != "role":
        raise ValueError(f"{role_ref} resolved as {selected.metadata.kind}, expected role")
    if "role.skill" not in selected.metadata.capabilities:
        raise ValueError(f"{role_ref} missing role.skill capability")
    return selected


def _read_manifest(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"role manifest must be an object: {path}")
    return value


def _private_kb_paths(
    plugin_path: Path, manifest: dict[str, Any], role_config: dict[str, Any]
) -> list[Path]:
    configured_value = role_config.get("private_kb") or []
    configured = [configured_value] if isinstance(configured_value, str) else configured_value
    paths: list[Path] = []
    for item in configured:
        path = _safe_plugin_file(plugin_path, str(item))
        if path.is_file():
            paths.append(path)
    if paths:
        return sorted(paths)
    plugin_root = plugin_path.resolve()
    for pattern in manifest.get("assets") or []:
        pattern_text = str(pattern)
        if not pattern_text.startswith("private_kb/"):
            continue
        pattern_path = Path(pattern_text)
        if pattern_path.is_absolute() or ".." in pattern_path.parts:
            raise ValueError(
                f"role private_kb asset path must stay inside plugin root: {pattern_text}"
            )
        for path in plugin_path.glob(pattern_text):
            if path.is_symlink() or not path.is_file():
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(plugin_root):
                raise ValueError(f"role private_kb asset path escapes plugin root: {pattern_text}")
            paths.append(resolved)
    return sorted(set(paths))


def _safe_plugin_file(plugin_path: Path, rel: str) -> Path:
    path = Path(rel)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"role private_kb path must stay inside plugin root: {rel}")
    resolved = (plugin_path / path).resolve()
    root = plugin_path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"role private_kb path escapes plugin root: {rel}")
    return resolved


def _fixed_questions(role_config: dict[str, Any], skill_markdown: str) -> list[str]:
    configured = role_config.get("fixed_questions") or []
    if isinstance(configured, list) and configured:
        return [str(item) for item in configured if str(item).strip()]
    return _extract_fixed_questions(skill_markdown)


def _extract_fixed_questions(skill_markdown: str) -> list[str]:
    lines = skill_markdown.splitlines()
    questions: list[str] = []
    in_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("fixed review questions:"):
            in_block = True
            continue
        if not in_block:
            continue
        if not stripped:
            if questions:
                break
            continue
        match = re.match(r"^[-*]\s+(.*)$", stripped)
        if not match:
            if questions:
                break
            continue
        questions.append(match.group(1).strip())
    return questions


def _title_from_role_id(role_id: str) -> str:
    return role_id.replace("_", " ").title()
