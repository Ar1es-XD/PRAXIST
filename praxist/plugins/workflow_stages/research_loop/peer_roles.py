"""Task-declared peer RoleSkill discovery and per-peer selection."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from praxist.core.role_skills import RoleSkill, load_role_skill

logger = logging.getLogger(__name__)

DEFAULT_TOPOLOGY_REF = "panel_topology:legacy_multi_pi_two_round"


def role_refs_from_task_descriptor(descriptor: dict[str, Any]) -> list[str]:
    """Return ordered role refs from supported task descriptor forms."""

    plugins = descriptor.get("praxist_plugins") or {}
    panel = plugins.get("panel") or {} if isinstance(plugins, dict) else {}
    refs: list[str] = []
    roles = panel.get("roles") or [] if isinstance(panel, dict) else []
    for item in roles if isinstance(roles, list) else []:
        if isinstance(item, str):
            refs.append(item)
        elif isinstance(item, dict):
            ref = item.get("role_ref") or item.get("role")
            if ref:
                refs.append(str(ref))
    optional = panel.get("optional_roles") or {} if isinstance(panel, dict) else {}
    for value in optional.values() if isinstance(optional, dict) else ():
        if not isinstance(value, dict) or not bool(value.get("enabled", False)):
            continue
        ref = value.get("role") or value.get("role_ref")
        if ref:
            refs.append(str(ref))
    return list(dict.fromkeys(refs))


def model_profile_defaults_from_task_descriptor(
    descriptor: dict[str, Any],
    registry: Any | None,
    task_project_path: Path | None = None,
) -> dict[str, str]:
    """Return model-profile defaults contributed by loadable task roles."""

    defaults = {"research_loop": "cheap_peer"}
    for role_ref in role_refs_from_task_descriptor(descriptor):
        try:
            skill = load_role_skill(
                role_ref,
                registry=registry,
                task_project_path=task_project_path,
            )
        except Exception:
            continue
        if skill.default_model_profile_ref:
            defaults[role_ref] = skill.default_model_profile_ref
    return defaults


def peer_role_refs_from_task_descriptor(
    descriptor: dict[str, Any],
    registry: Any | None,
    task_project_path: Path | None = None,
) -> tuple[str, ...]:
    """Return all loadable task roles that declare peer behavior."""

    refs: list[str] = []
    for role_ref in role_refs_from_task_descriptor(descriptor):
        try:
            skill = load_role_skill(
                role_ref,
                registry=registry,
                task_project_path=task_project_path,
            )
        except Exception as exc:
            if role_ref.startswith("task_role:"):
                raise ValueError(f"cannot load declared task role {role_ref}: {exc}") from exc
            continue
        if skill.role_kind == "peer":
            refs.append(role_ref)
    return tuple(dict.fromkeys(refs))


def peer_role_ref_from_task_descriptor(
    descriptor: dict[str, Any],
    registry: Any | None,
    task_project_path: Path | None = None,
) -> str:
    """Return the first declared peer role for legacy single-role callers."""

    refs = peer_role_refs_from_task_descriptor(descriptor, registry, task_project_path)
    return refs[0] if refs else "role:peer"


def normalize_peer_role_id(value: Any) -> str:
    """Match the role-label normalization used by PI agenda validation."""

    if not isinstance(value, str):
        return ""
    normalized = re.sub(r"[\s./-]+", "_", value.strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")


def index_peer_role_skills(skills: tuple[RoleSkill, ...]) -> dict[str, RoleSkill]:
    """Index canonical, legacy, short, and fully qualified role aliases."""

    indexed: dict[str, RoleSkill] = {}
    for skill in skills:
        keys = {
            skill.role_id,
            skill.legacy_role_id,
            skill.role_ref,
            skill.role_ref.partition(":")[2],
        }
        for key in keys:
            normalized = normalize_peer_role_id(key)
            existing = indexed.get(normalized)
            if existing is not None and existing.role_ref != skill.role_ref:
                raise ValueError(
                    f"ambiguous peer role alias {normalized!r}: "
                    f"{existing.role_ref} and {skill.role_ref}"
                )
            if normalized:
                indexed[normalized] = skill
    return indexed


def resolve_topology_peer_info(
    plugin_registry: Any | None,
    *,
    topology_ref: str = DEFAULT_TOPOLOGY_REF,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Resolve advisory topology role rotation and descriptions."""

    try:
        from praxist.core.panel_topology import panel_topology_for_ref

        spec = panel_topology_for_ref(topology_ref, registry=plugin_registry)
        return tuple(spec.peer_role_rotation), dict(spec.peer_role_descriptions)
    except Exception as exc:  # noqa: BLE001 - advisory metadata must not block startup.
        logger.debug("could not resolve topology peer info: %s", exc)
        return (), {}


def resolve_peer_role_rotation(
    plugin_registry: Any | None,
    *,
    topology_ref: str = DEFAULT_TOPOLOGY_REF,
) -> tuple[str, ...]:
    """Return only the topology's advisory initial role rotation."""

    rotation, _ = resolve_topology_peer_info(plugin_registry, topology_ref=topology_ref)
    return rotation


class PeerRoleSelector:
    """Resolve one peer's effective task RoleSkill without inventing provenance."""

    def __init__(
        self,
        *,
        run_dir: Path,
        task_spec: Any,
        role_rotation: tuple[str, ...],
        default_role_skill: RoleSkill | None,
        role_skills: tuple[RoleSkill, ...],
    ) -> None:
        self.run_dir = run_dir
        self.task_spec = task_spec
        self.role_rotation = role_rotation
        self.default_role_skill = default_role_skill
        self.role_skills = role_skills
        self.skills_by_id = index_peer_role_skills(role_skills)

    def skill_for_context(self, context: dict[str, Any]) -> RoleSkill | None:
        return self._skill_for_id(self._role_id_from_context(context))

    def ref_for_context(self, context: dict[str, Any]) -> str | None:
        role_id = self._role_id_from_context(context)
        skill = self._skill_for_id(role_id)
        if skill is not None:
            return skill.role_ref
        return None if normalize_peer_role_id(role_id) else self._default_role_ref

    def ref_for_peer(self, gen_id: int, peer_index: int) -> str | None:
        peer_id = f"gen{gen_id}_peer{peer_index}"
        role_id = ""
        try:
            from .backend.pi_agent import load_agenda_for_gen

            agenda = load_agenda_for_gen(
                self.run_dir,
                gen_id,
                cohort_size=self.task_spec.generation_policy.cohort_size,
            )
            contracts = agenda.get("peer_contracts") if isinstance(agenda, dict) else None
            contract = contracts.get(peer_id) if isinstance(contracts, dict) else None
            if isinstance(contract, dict):
                role_id = str(contract.get("role") or "")
        except Exception as exc:  # noqa: BLE001 - topology provenance remains advisory.
            logger.debug("could not resolve peer role for %s: %s", peer_id, exc)
        if not role_id and gen_id == 0 and self.role_rotation:
            role_id = self.role_rotation[peer_index % len(self.role_rotation)]
        skill = self._skill_for_id(role_id)
        if skill is not None:
            return skill.role_ref
        return None if normalize_peer_role_id(role_id) else self._default_role_ref

    @property
    def _default_role_ref(self) -> str | None:
        return self.default_role_skill.role_ref if self.default_role_skill is not None else None

    def _skill_for_id(self, role_id: str) -> RoleSkill | None:
        normalized = normalize_peer_role_id(role_id)
        if not normalized:
            return self.default_role_skill
        skill = self.skills_by_id.get(normalized)
        if skill is not None:
            return skill
        return self.role_skills[0] if len(self.role_skills) == 1 else None

    @staticmethod
    def _role_id_from_context(context: dict[str, Any]) -> str:
        peer_id = str(context.get("peer_id") or "")
        agenda = context.get("research_agenda")
        contracts = agenda.get("peer_contracts") if isinstance(agenda, dict) else None
        contract = contracts.get(peer_id) if isinstance(contracts, dict) else None
        return str(contract.get("role") or "") if isinstance(contract, dict) else ""
