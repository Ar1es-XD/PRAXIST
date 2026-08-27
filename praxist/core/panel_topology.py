"""Panel topology descriptors and deterministic planning helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
class PanelRoundSpec:
    """Declarative description of one PI panel round and its participating role references."""

    round_id: str
    role_refs: list[str]
    parallelism: int
    timeout_seconds: int
    model_profile_ref: str | None = None
    visibility: str = "shared"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PanelRoleSpec:
    """Declarative role binding used by a panel topology to instantiate a PI or Chair role."""

    role_ref: str
    role_id: str
    legacy_role_id: str
    role_kind: str
    model_profile_ref: str | None = None
    private_pack_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PanelTopologySpec:
    """Resolved panel topology contract consumed by the research_loop workflow stage."""

    topology_ref: str
    modes: dict[str, list[str]]
    rounds: list[PanelRoundSpec]
    role_specs: dict[str, PanelRoleSpec] = field(default_factory=dict)
    high_stakes_triggers: list[str] = field(default_factory=list)
    chair_role_ref: str | None = None
    peer_memo_visibility: str = "anonymized"
    # Optional plugin-supplied prompts directory. When set, PI roles and the
    # Chair load Jinja templates from this directory first, falling back to
    # the framework's bundled prompts for any template the plugin omits.
    # See ``docs/concepts/panel_topology_prompts.md`` for the override contract.
    prompts_dir: Path | None = None
    # Optional plugin-supplied peer-role rotation used by the Chair's
    # deterministic fallback agenda (when the Chair LLM emits non-parseable
    # YAML). Custom panel topologies whose peers use task-local role
    # vocabulary set this so the fallback emits agenda contracts with role
    # names the task project actually defines, rather than the bundled
    # ``exploit / falsifier / bridge / anti_mainline / theorist`` set.
    # See issue #84 for the rationale.
    peer_role_rotation: tuple[str, ...] = ()
    # Optional plugin-supplied descriptions for the peer roles a topology
    # binds. Keyed by role name; values are the prose the peer prompt
    # template (``prompt_generation.jinja2``) renders for
    # ``my_contract.role`` in place of the bundled five-bullet vocabulary
    # block. Topologies that don't declare this field keep the bundled
    # bullets so existing tasks render unchanged. See issue #85.
    peer_role_descriptions: dict[str, str] = field(default_factory=dict)

    def roles_for_mode(self, panel_mode: str) -> list[str]:
        if panel_mode in self.modes:
            return list(self.modes[panel_mode])
        return list(self.modes.get("mini", []))

    def role_specs_for_mode(self, panel_mode: str) -> list[PanelRoleSpec]:
        specs = []
        for role_id in self.roles_for_mode(panel_mode):
            spec = self.role_specs.get(role_id)
            if spec is None:
                spec = PanelRoleSpec(
                    role_ref=f"role:{role_id}",
                    role_id=role_id,
                    legacy_role_id=role_id,
                    role_kind="pi",
                )
            specs.append(spec)
        return specs

    def role_refs_for_mode(self, panel_mode: str) -> list[str]:
        return [spec.role_ref for spec in self.role_specs_for_mode(panel_mode)]

    def role_spec(self, role_id: str) -> PanelRoleSpec:
        if role_id not in self.role_specs:
            raise KeyError(f"Unknown panel role: {role_id}")
        return self.role_specs[role_id]

    def has_high_stakes_signal(self, shared_core: dict[str, Any]) -> bool:
        if not isinstance(shared_core, dict):
            return False
        claim_blob_parts: list[str] = []
        claim_ledger = shared_core.get("claim_ledger_digest", {})
        for entry in claim_ledger.get("active") or []:
            if isinstance(entry, dict):
                claim_blob_parts.append(str(entry.get("title", "")))
                claim_blob_parts.append(str(entry.get("boundary", "")))
        for entry in shared_core.get("retired_claims") or []:
            if isinstance(entry, dict):
                claim_blob_parts.append(str(entry.get("title", "")))
                claim_blob_parts.append(str(entry.get("boundary", "")))
        blob = " ".join(claim_blob_parts).lower()
        return any(trigger in blob for trigger in self.high_stakes_triggers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topology_ref": self.topology_ref,
            "modes": self.modes,
            "rounds": [round_spec.to_dict() for round_spec in self.rounds],
            "role_specs": {key: value.to_dict() for key, value in self.role_specs.items()},
            "high_stakes_triggers": self.high_stakes_triggers,
            "chair_role_ref": self.chair_role_ref,
            "peer_memo_visibility": self.peer_memo_visibility,
            "prompts_dir": str(self.prompts_dir) if self.prompts_dir is not None else None,
            "peer_role_rotation": list(self.peer_role_rotation),
            "peer_role_descriptions": dict(self.peer_role_descriptions),
        }


def panel_topology_for_ref(
    topology_ref: str,
    *,
    registry: PluginRegistry | None = None,
    workspace: Path | None = None,
) -> PanelTopologySpec:
    """Resolve a panel topology plugin reference into an executable topology specification."""
    selected = _selected_topology_plugin(topology_ref, registry=registry, workspace=workspace)
    manifest = _read_manifest(Path(selected.path) / "plugin.yaml")
    return panel_topology_from_manifest(topology_ref, manifest, plugin_path=Path(selected.path))


def panel_topology_from_manifest(
    topology_ref: str,
    manifest: dict[str, Any],
    *,
    plugin_path: Path | None = None,
) -> PanelTopologySpec:
    """Build a PanelTopologySpec from a plugin manifest and selected plugin descriptor.

    When ``plugin_path`` is provided, a relative ``topology.prompts_dir`` entry
    in the manifest is resolved against it. Absolute ``prompts_dir`` entries
    are accepted regardless. When ``plugin_path`` is ``None`` (e.g. from unit
    tests that exercise manifest parsing in isolation), a manifest that
    declares a relative ``prompts_dir`` raises ``ValueError`` rather than
    silently producing a topology that would later miss its templates.
    """
    topology = manifest.get("topology") if isinstance(manifest.get("topology"), dict) else None
    if topology is None:
        raise ValueError(f"{topology_ref} is missing a topology contract")
    role_specs = _role_specs_from_manifest(topology)
    role_modes = _mode_roles_from_manifest(topology, role_specs)
    return PanelTopologySpec(
        topology_ref=str(topology.get("topology_ref") or topology_ref),
        modes=role_modes,
        rounds=_round_specs_from_manifest(topology),
        role_specs=role_specs,
        high_stakes_triggers=[
            str(item).lower() for item in topology.get("high_stakes_triggers") or []
        ],
        chair_role_ref=str(topology["chair_role_ref"]) if topology.get("chair_role_ref") else None,
        peer_memo_visibility=str(topology.get("peer_memo_visibility") or "anonymized"),
        prompts_dir=_resolve_prompts_dir(topology, plugin_path),
        peer_role_rotation=_peer_role_rotation_from_manifest(topology),
        peer_role_descriptions=_peer_role_descriptions_from_manifest(topology),
    )


def _peer_role_descriptions_from_manifest(topology: dict[str, Any]) -> dict[str, str]:
    """Parse the optional ``peer_role_descriptions`` mapping from a manifest.

    Returns an empty dict when the field is absent so the peer prompt
    template renders the bundled five-bullet vocabulary unchanged.
    Non-mapping shapes and non-string entries raise so manifest typos
    surface at topology resolution rather than at first render.
    """
    raw = topology.get("peer_role_descriptions")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"panel topology peer_role_descriptions must be a mapping, got {type(raw).__name__}"
        )
    out: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(
                f"panel topology peer_role_descriptions keys must be non-empty strings: {key!r}"
            )
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"panel topology peer_role_descriptions[{key!r}] must be a non-empty string"
            )
        out[key.strip()] = value.strip()
    return out


def _peer_role_rotation_from_manifest(topology: dict[str, Any]) -> tuple[str, ...]:
    """Parse the optional ``peer_role_rotation`` list from a topology manifest.

    Returns an empty tuple when the field is absent or empty so the Chair
    fallback retains its bundled rotation. Non-list values, non-string
    entries, and blank strings raise so manifest typos surface at topology
    resolution rather than at first fallback firing.
    """
    raw = topology.get("peer_role_rotation")
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(
            f"panel topology peer_role_rotation must be a list, got {type(raw).__name__}"
        )
    rotation: list[str] = []
    for entry in raw:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(
                f"panel topology peer_role_rotation entries must be non-empty strings: {entry!r}"
            )
        rotation.append(entry.strip())
    return tuple(rotation)


def _resolve_prompts_dir(topology: dict[str, Any], plugin_path: Path | None) -> Path | None:
    """Resolve ``topology.prompts_dir`` from a manifest into an absolute Path.

    Returns ``None`` when the manifest does not declare a custom prompts
    directory. Raises ``ValueError`` for malformed entries or paths that do
    not point at an existing directory so manifest authoring bugs surface
    at topology resolution rather than at first template lookup.
    """
    raw = topology.get("prompts_dir")
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str):
        raise ValueError(f"panel topology prompts_dir must be a string, got {type(raw).__name__}")
    candidate = Path(raw)
    if candidate.is_absolute():
        if not candidate.is_dir():
            raise ValueError(f"panel topology prompts_dir does not exist: {candidate}")
        return candidate
    if plugin_path is None:
        raise ValueError(
            f"panel topology declared prompts_dir={raw!r} but plugin_path is unknown; "
            f"cannot resolve relative path"
        )
    resolved = (plugin_path / candidate).resolve()
    if not resolved.is_dir():
        raise ValueError(f"panel topology prompts_dir does not exist: {resolved}")
    return resolved


def _selected_topology_plugin(
    topology_ref: str,
    *,
    registry: PluginRegistry | None,
    workspace: Path | None,
) -> SelectedPlugin:
    parsed = PluginRef.parse(topology_ref)
    if parsed.kind != "panel_topology":
        raise ValueError(f"Panel topology ref must use kind panel_topology: {topology_ref}")
    if registry is not None:
        selected = registry.descriptor(parsed.kind, parsed.name)
    else:
        loader = PluginLoader(PluginRoots.defaults(workspace))
        manifest = loader.resolve(
            [topology_ref], root_task_ref=topology_ref, enforce_bundled_execution=True
        )
        selected = loader.load(manifest).descriptor(parsed.kind, parsed.name)
    if selected.metadata.kind != "panel_topology":
        raise ValueError(
            f"{topology_ref} resolved as {selected.metadata.kind}, expected panel_topology"
        )
    return selected


def _read_manifest(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"panel topology manifest must be an object: {path}")
    return value


def _role_specs_from_manifest(topology: dict[str, Any]) -> dict[str, PanelRoleSpec]:
    raw_roles = topology.get("roles") if isinstance(topology, dict) else None
    role_specs: dict[str, PanelRoleSpec] = {}
    if isinstance(raw_roles, list):
        for raw in raw_roles:
            if not isinstance(raw, dict):
                continue
            role_id = str(raw.get("role_id") or raw.get("legacy_role_id") or "")
            role_ref = str(raw.get("role_ref") or "")
            if not role_id or not role_ref:
                continue
            role_specs[role_id] = PanelRoleSpec(
                role_ref=role_ref,
                role_id=role_id,
                legacy_role_id=str(raw.get("legacy_role_id") or role_id),
                role_kind=str(raw.get("role_kind") or "pi"),
                model_profile_ref=raw.get("model_profile_ref"),
                private_pack_key=raw.get("private_pack_key") or role_id,
            )
    return role_specs


def _mode_roles_from_manifest(
    topology: dict[str, Any], role_specs: dict[str, PanelRoleSpec]
) -> dict[str, list[str]]:
    raw_modes = topology.get("modes") if isinstance(topology, dict) else None
    if not isinstance(raw_modes, dict) or not raw_modes:
        raise ValueError("panel topology contract must define non-empty modes")
    modes = {
        str(mode): [
            _role_id_for_ref_or_id(str(item), role_specs)
            for item in (items or [])
            if _role_id_for_ref_or_id(str(item), role_specs)
        ]
        for mode, items in raw_modes.items()
        if isinstance(items, list)
    }
    if not modes:
        raise ValueError("panel topology contract must define at least one valid mode")
    return modes


def _role_id_for_ref_or_id(value: str, role_specs: dict[str, PanelRoleSpec]) -> str:
    if value in role_specs:
        return value
    for role_id, spec in role_specs.items():
        if value == spec.role_ref:
            return role_id
    try:
        parsed = PluginRef.parse(value)
        if parsed.kind == "role":
            return parsed.name
    except ValueError:
        pass
    return value


def _round_specs_from_manifest(topology: dict[str, Any]) -> list[PanelRoundSpec]:
    raw_rounds = topology.get("rounds")
    if not isinstance(raw_rounds, list) or not raw_rounds:
        raise ValueError("panel topology contract must define non-empty rounds")
    rounds: list[PanelRoundSpec] = []
    for raw in raw_rounds:
        if not isinstance(raw, dict):
            raise ValueError("panel topology rounds must be objects")
        round_id = str(raw.get("round_id") or "")
        if not round_id:
            raise ValueError("panel topology round is missing round_id")
        rounds.append(
            PanelRoundSpec(
                round_id=round_id,
                role_refs=[str(item) for item in raw.get("role_refs") or []],
                parallelism=int(raw.get("parallelism", 1)),
                timeout_seconds=int(raw.get("timeout_seconds", 0)),
                model_profile_ref=raw.get("model_profile_ref"),
                visibility=str(raw.get("visibility") or "shared"),
            )
        )
    return rounds
