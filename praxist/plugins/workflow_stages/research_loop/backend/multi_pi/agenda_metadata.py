"""Research-metadata preservation for PI/Chair agenda surfaces."""

from __future__ import annotations

from typing import Any

RESEARCH_METADATA_KEYS = (
    "bottleneck_target",
    "evidence_stage",
    "tradeoff_class",
    "primary_tradeoff",
    "next_step_intent",
    "parent_candidate",
    "parent_usage",
)


def fallback_research_metadata(role: str = "") -> dict[str, str]:
    """Conservative labels used when an agenda surface lacks explicit metadata."""

    norm = str(role or "").strip().lower().replace("-", "_").replace(" ", "_")
    if norm == "falsifier":
        return {
            "bottleneck_target": "negative_control_or_falsification",
            "evidence_stage": "scout",
            "tradeoff_class": "negative_falsifier",
            "primary_tradeoff": "custom",
            "next_step_intent": "ablate_or_falsify",
            "parent_candidate": "",
            "parent_usage": "falsify",
        }
    if norm == "bridge":
        return {
            "bottleneck_target": "process_or_evidence_gap",
            "evidence_stage": "scout",
            "tradeoff_class": "incomplete_evidence",
            "primary_tradeoff": "custom",
            "next_step_intent": "combine_with_other_mechanism",
            "parent_candidate": "",
            "parent_usage": "compare",
        }
    if norm == "anti_mainline":
        return {
            "bottleneck_target": "process_or_evidence_gap",
            "evidence_stage": "scout",
            "tradeoff_class": "incomplete_evidence",
            "primary_tradeoff": "custom",
            "next_step_intent": "pivot_to_distinct_surface",
            "parent_candidate": "",
            "parent_usage": "none",
        }
    if norm == "theorist":
        return {
            "bottleneck_target": "process_or_evidence_gap",
            "evidence_stage": "smoke",
            "tradeoff_class": "incomplete_evidence",
            "primary_tradeoff": "custom",
            "next_step_intent": "ablate_or_falsify",
            "parent_candidate": "",
            "parent_usage": "none",
        }
    return {
        "bottleneck_target": "process_or_evidence_gap",
        "evidence_stage": "scout",
        "tradeoff_class": "incomplete_evidence",
        "primary_tradeoff": "custom",
        "next_step_intent": "preserve_and_validate",
        "parent_candidate": "",
        "parent_usage": "none",
    }


def research_metadata_overrides(source: Any) -> dict[str, str]:
    """Extract explicit metadata from top-level, metrics, extra, or nested extra."""

    if not isinstance(source, dict):
        return {}
    containers: list[dict[str, Any]] = [source]
    for key in ("metrics", "extra"):
        value = source.get(key)
        if not isinstance(value, dict):
            continue
        containers.append(value)
        nested_extra = value.get("extra")
        if isinstance(nested_extra, dict):
            containers.append(nested_extra)

    out: dict[str, str] = {}
    for key in RESEARCH_METADATA_KEYS:
        for container in containers:
            value = container.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                out[key] = text
                break
    return out


def research_metadata_from_sources(role: str, sources: list[Any]) -> dict[str, str]:
    """Return role defaults with explicit source metadata layered on top."""

    metadata = fallback_research_metadata(role)
    for source in sources:
        metadata.update(research_metadata_overrides(source))
    return metadata


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _set_missing_metadata(
    surface: dict[str, Any],
    metadata: dict[str, str],
    path: str,
    changed: list[str],
) -> None:
    for key in RESEARCH_METADATA_KEYS:
        if _nonempty(surface.get(key)):
            continue
        surface[key] = metadata.get(key, "")
        changed.append(f"{path}.{key}")


def _hypotheses(agenda: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        h
        for h in (agenda.get("cross_peer_hypotheses") or [])
        if isinstance(h, dict) and h.get("id")
    ]


def agenda_metadata_sources_for_target(
    agenda: dict[str, Any],
    target: Any,
    _seen: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Resolve an agenda target token to metadata-bearing surfaces."""

    if not isinstance(agenda, dict) or target is None:
        return []
    token = str(target).strip()
    if not token:
        return []
    seen = set() if _seen is None else _seen
    if token in seen:
        return []
    seen.add(token)

    sources: list[dict[str, Any]] = []
    for hyp in _hypotheses(agenda):
        if str(hyp.get("id")) == token:
            sources.append(hyp)

    bridge = agenda.get("bridge_hypothesis")
    if (
        isinstance(bridge, dict)
        and bridge
        and (token in {"bridge_hypothesis", "bridge_contract"} or str(bridge.get("id")) == token)
    ):
        for anchor_key in ("source_anchor_A", "source_anchor_B"):
            anchor = bridge.get(anchor_key)
            if not isinstance(anchor, dict):
                continue
            sources.extend(
                agenda_metadata_sources_for_target(
                    agenda,
                    anchor.get("extracted_mechanism") or anchor.get("variant"),
                    seen,
                )
            )
            sources.append(bridge)

    anti = agenda.get("anti_mainline_contract")
    if token == "anti_mainline_contract" and isinstance(anti, dict):
        sources.append(anti)

    falsification = agenda.get("falsification_contract")
    if token == "falsification_contract" and isinstance(falsification, dict):
        sources.extend(
            agenda_metadata_sources_for_target(
                agenda,
                falsification.get("target_hypothesis"),
                seen,
            )
        )
        sources.append(falsification)

    for action in agenda.get("consensus_actions") or []:
        if not isinstance(action, dict):
            continue
        if token in {str(action.get("action_id")), str(action.get("claim_or_hypothesis"))}:
            sources.append(action)

    for dissent in agenda.get("DISSENT_TO_EXPERIMENT") or []:
        if not isinstance(dissent, dict):
            continue
        if token in {str(dissent.get("dissent_id")), str(dissent.get("disputed_claim"))}:
            sources.append(dissent)

    for idea in agenda.get("minority_high_upside") or []:
        if isinstance(idea, dict) and str(idea.get("idea_id")) == token:
            sources.append(idea)
    return sources


def normalize_agenda_research_metadata(agenda: dict[str, Any]) -> list[str]:
    """Fill missing metadata on agenda surfaces without overwriting explicit labels."""

    if not isinstance(agenda, dict):
        return []
    changed: list[str] = []
    hyps = _hypotheses(agenda)
    explicit_sources: list[dict[str, Any]] = list(hyps)
    for key in ("bridge_hypothesis", "anti_mainline_contract", "falsification_contract"):
        value = agenda.get(key)
        if isinstance(value, dict):
            explicit_sources.append(value)
    for key in ("consensus_actions", "DISSENT_TO_EXPERIMENT", "minority_high_upside"):
        for item in agenda.get(key) or []:
            if isinstance(item, dict):
                explicit_sources.append(item)
    contracts = agenda.get("peer_contracts")
    if isinstance(contracts, dict):
        explicit_sources.extend(c for c in contracts.values() if isinstance(c, dict))
    explicit_metadata_by_id = {
        id(source): research_metadata_overrides(source) for source in explicit_sources
    }

    def _metadata(role: str, sources: list[Any]) -> dict[str, str]:
        metadata = fallback_research_metadata(role)
        for source in sources:
            if not isinstance(source, dict):
                continue
            metadata.update(
                explicit_metadata_by_id.get(id(source), research_metadata_overrides(source))
            )
        return metadata

    for idx, hyp in enumerate(hyps):
        metadata = _metadata("hypothesis", [hyp])
        _set_missing_metadata(hyp, metadata, f"cross_peer_hypotheses[{idx}]", changed)

    bridge = agenda.get("bridge_hypothesis")
    if isinstance(bridge, dict) and bridge:
        bridge_sources: list[Any] = []
        for anchor_key in ("source_anchor_A", "source_anchor_B"):
            anchor = bridge.get(anchor_key)
            if not isinstance(anchor, dict):
                continue
            bridge_sources.extend(
                agenda_metadata_sources_for_target(
                    agenda,
                    anchor.get("extracted_mechanism") or anchor.get("variant"),
                )
            )
        if not bridge_sources and len(hyps) > 1:
            bridge_sources.append(hyps[1])
        metadata = _metadata("bridge", bridge_sources + [bridge])
        metadata["parent_usage"] = "compare"
        _set_missing_metadata(bridge, metadata, "bridge_hypothesis", changed)

    anti = agenda.get("anti_mainline_contract")
    if isinstance(anti, dict) and anti:
        metadata = _metadata("anti_mainline", [anti])
        _set_missing_metadata(anti, metadata, "anti_mainline_contract", changed)

    falsification = agenda.get("falsification_contract")
    if isinstance(falsification, dict) and falsification:
        target_sources = agenda_metadata_sources_for_target(
            agenda,
            falsification.get("target_hypothesis"),
        )
        metadata = _metadata("falsifier", target_sources + [falsification])
        _set_missing_metadata(falsification, metadata, "falsification_contract", changed)

    for idx, action in enumerate(agenda.get("consensus_actions") or []):
        if not isinstance(action, dict):
            continue
        role = str(action.get("assigned_role") or action.get("role") or "exploit")
        target_sources = agenda_metadata_sources_for_target(
            agenda,
            action.get("claim_or_hypothesis"),
        )
        metadata = _metadata(role, target_sources + [action])
        _set_missing_metadata(action, metadata, f"consensus_actions[{idx}]", changed)

    for idx, dissent in enumerate(agenda.get("DISSENT_TO_EXPERIMENT") or []):
        if not isinstance(dissent, dict):
            continue
        role = str(dissent.get("assigned_peer_role") or "falsifier")
        target_sources = agenda_metadata_sources_for_target(
            agenda,
            dissent.get("disputed_claim"),
        )
        metadata = _metadata(role, target_sources + [dissent])
        _set_missing_metadata(dissent, metadata, f"DISSENT_TO_EXPERIMENT[{idx}]", changed)

    contracts = agenda.get("peer_contracts")
    if isinstance(contracts, dict):
        for peer_id, contract in contracts.items():
            if not isinstance(contract, dict):
                continue
            role = str(contract.get("role") or "")
            target_sources = agenda_metadata_sources_for_target(
                agenda,
                contract.get("target_hypothesis"),
            )
            metadata = _metadata(role, target_sources + [contract])
            _set_missing_metadata(contract, metadata, f"peer_contracts[{peer_id}]", changed)
    return changed
