"""Cohort-level quality-diversity allocation for DIG-Lite contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import DIGCohortQDConfig, DIGLiteConfig
from .labels import CanonicalDIGLabels, canonical_labels_for_candidate, normalize_label
from .runner import DIGLiteResult
from .schema import (
    Candidate,
    CandidateReview,
    ContractAmendmentPolicy,
    ContractStep,
    DiversityCell,
    ExpectedMetricSignature,
    SelectedContract,
)
from .selection import (
    is_diagnostic_like_candidate,
    is_near_duplicate_candidate,
    violates_file_rules,
)
from .validator import (
    DIGValidationContext,
    compatible_with_peer_lane,
    validate_selected_contract,
    validate_selected_contract_matches_candidate,
)

_DEFAULT_TARGET_FIELDS = (
    "name",
    "mechanism_family",
    "intervention_surface",
    "intent",
    "hypothesis",
    "expected_gain_path",
    "changes",
)
_STANDARD_FORBIDDEN_CHANGES = (
    "do not modify evaluator",
    "do not change data split",
    "do not change metric calculation",
)
_AMENDMENT_REASONS = (
    "baseline assumption was wrong",
    "shape or API mismatch makes original implementation impossible",
    "contract would require touching a forbidden path",
)


@dataclass
class _CandidateItem:
    peer_index: int
    result: DIGLiteResult
    candidate: Candidate
    review: CandidateReview
    lane_fit: bool
    diagnostic_like: bool
    near_duplicate: bool
    local_selected: bool
    labels: CanonicalDIGLabels


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _risk_penalty(candidate: Candidate) -> float:
    values = (
        candidate.risk.implementation,
        candidate.risk.metric_gaming,
        candidate.risk.silent_bug,
        candidate.risk.compute,
    )
    weights = {"low": 0.0, "medium": 0.5, "high": 1.0}
    return sum(weights.get(_norm(value), 0.5) for value in values)


def _candidate_cell(item: _CandidateItem) -> tuple[str, str, str]:
    return item.labels.formal_cell()


def _contract_cell(candidate: Candidate) -> DiversityCell:
    sig = candidate.diversity_signature
    return DiversityCell(
        mechanism_family=sig.mechanism_family,
        intervention_surface=sig.intervention_surface,
        intent=sig.intent,
    )


def _candidate_text(candidate: Candidate, fields: list[str] | tuple[str, ...]) -> str:
    parts: list[str] = []
    fields = list(fields or _DEFAULT_TARGET_FIELDS)
    for field in fields:
        key = str(field or "").strip()
        if key == "name":
            parts.append(candidate.name)
        elif key == "mechanism_family":
            parts.append(candidate.mechanism_family)
        elif key == "intervention_surface":
            parts.append(candidate.intervention_surface)
        elif key == "intent":
            parts.append(candidate.intent)
        elif key == "hypothesis":
            parts.append(candidate.hypothesis)
        elif key == "expected_gain_path":
            parts.append(candidate.expected_gain_path)
        elif key == "changes":
            parts.extend(candidate.implementation_sketch.changes)
        elif key == "files_to_modify":
            parts.extend(candidate.implementation_sketch.files_to_modify)
        elif key == "diagnostic_prediction":
            parts.extend(
                [
                    candidate.diagnostic_prediction.primary_metric,
                    candidate.diagnostic_prediction.secondary_or_safety_metric,
                    candidate.diagnostic_prediction.internal_signal,
                ]
            )
    return " ".join(part for part in parts if part).lower()


def _matches_target_group(
    candidate: Candidate,
    group: Any,
    config: DIGLiteConfig | None = None,
) -> bool:
    synonyms = (
        dict(getattr(getattr(config, "cohort_qd", None), "label_synonyms", {}) or {})
        if config is not None
        else {}
    )
    keywords = []
    for keyword in getattr(group, "keywords", []):
        normalized = _norm(keyword)
        if not normalized:
            continue
        keywords.append(normalized)
        canonical = normalize_label(keyword, "", synonyms=synonyms)
        if canonical:
            keywords.append(canonical)
    if not keywords:
        return False
    text = _candidate_text(candidate, getattr(group, "fields", []) or _DEFAULT_TARGET_FIELDS)
    if config is not None:
        text = f"{text} {' '.join(canonical_labels_for_candidate(candidate, config).formal_cell())}"
    normalized_text = _norm(text)
    return any(keyword in normalized_text for keyword in keywords)


def _limit_from_count_and_fraction(
    *,
    explicit_count: int,
    fraction: float,
    cohort_size: int,
) -> int:
    if explicit_count > 0:
        return explicit_count
    if fraction > 0 and cohort_size > 0:
        return max(1, int(math.ceil(cohort_size * fraction)))
    return cohort_size


def _countable_parent_lineage(parent: str) -> bool:
    return _norm(parent) not in {
        "",
        "none",
        "unknown",
        "other",
        "independent",
        "independent_lineage",
    }


def _limits(config: DIGCohortQDConfig, cohort_size: int) -> dict[str, int]:
    return {
        "cell": config.max_same_diversity_cell_peers
        if config.max_same_diversity_cell_peers > 0
        else cohort_size,
        "family": _limit_from_count_and_fraction(
            explicit_count=config.max_same_mechanism_family_peers,
            fraction=config.max_same_mechanism_family_fraction,
            cohort_size=cohort_size,
        ),
        "surface": _limit_from_count_and_fraction(
            explicit_count=config.max_same_intervention_surface_peers,
            fraction=config.max_same_intervention_surface_fraction,
            cohort_size=cohort_size,
        ),
        "intent": _limit_from_count_and_fraction(
            explicit_count=config.max_same_intent_peers,
            fraction=config.max_same_intent_fraction,
            cohort_size=cohort_size,
        ),
        "semantic": _limit_from_count_and_fraction(
            explicit_count=config.max_same_semantic_family_peers,
            fraction=config.max_same_semantic_family_fraction,
            cohort_size=cohort_size,
        ),
        "parent": _limit_from_count_and_fraction(
            explicit_count=config.max_same_parent_lineage_peers,
            fraction=config.max_same_parent_lineage_fraction,
            cohort_size=cohort_size,
        ),
    }


def _would_violate_caps(
    item: _CandidateItem,
    counts: dict[str, dict[Any, int]],
    limits: dict[str, int],
) -> bool:
    cell = _candidate_cell(item)
    family = item.labels.canonical_mechanism_family
    surface = item.labels.canonical_intervention_surface
    intent = item.labels.canonical_intent
    semantic = item.labels.canonical_semantic_family
    parent = item.labels.canonical_parent_lineage
    parent_violates = (
        _countable_parent_lineage(parent) and counts["parent"].get(parent, 0) >= limits["parent"]
    )
    return (
        counts["cell"].get(cell, 0) >= limits["cell"]
        or counts["family"].get(family, 0) >= limits["family"]
        or counts["surface"].get(surface, 0) >= limits["surface"]
        or counts["intent"].get(intent, 0) >= limits["intent"]
        or counts["semantic"].get(semantic, 0) >= limits["semantic"]
        or parent_violates
    )


def _score_item(
    item: _CandidateItem,
    *,
    config: DIGLiteConfig,
    counts: dict[str, dict[Any, int]],
) -> float:
    qd = config.cohort_qd
    candidate = item.candidate
    score = float(item.review.quality_score) * qd.quality_weight
    if item.lane_fit:
        score += qd.lane_fit_bonus
    if item.local_selected:
        score += qd.local_selection_bonus
    if counts["family"].get(item.labels.canonical_mechanism_family, 0) == 0:
        score += qd.novelty_weight
    if counts["surface"].get(item.labels.canonical_intervention_surface, 0) == 0:
        score += qd.novelty_weight
    if counts["semantic"].get(item.labels.canonical_semantic_family, 0) == 0:
        score += qd.novelty_weight
    if counts["parent"].get(item.labels.canonical_parent_lineage, 0) == 0:
        score += qd.novelty_weight
    if counts["cell"].get(_candidate_cell(item), 0) == 0:
        score += qd.novelty_weight
    for group in qd.target_keyword_groups:
        if _matches_target_group(candidate, group, config):
            score += qd.target_keyword_bonus
    score -= _risk_penalty(candidate) * qd.risk_penalty_weight
    if (
        config.innovation.enabled
        and item.diagnostic_like
        and _norm(
            getattr(item.result.validation_context, "selection_policy", {}).get("intent_slot")
        )
        != "diagnostic"
    ):
        score -= float(config.innovation.diagnostic_score_penalty)
    if item.near_duplicate:
        score -= qd.novelty_weight * 2.0
    return score


def _eligible_items_for_result(
    index: int, result: DIGLiteResult, config: DIGLiteConfig
) -> list[_CandidateItem]:
    if (
        result.candidate_pool is None
        or result.candidate_reviews is None
        or result.validation_context is None
    ):
        return []
    review_by_id = {review.candidate_id: review for review in result.candidate_reviews.reviews}
    items: list[_CandidateItem] = []
    selected_id = result.selected_contract.selected_candidate_id
    for candidate in result.candidate_pool.candidates:
        review = review_by_id.get(candidate.candidate_id)
        if review is None or review.fatal_flaws:
            continue
        if violates_file_rules(candidate, result.validation_context):
            continue
        lane_fit = compatible_with_peer_lane(
            _contract_cell(candidate), result.validation_context.peer_lane
        )
        if not lane_fit and not config.diversity.allow_adjacent_lane_fallback:
            continue
        near_duplicate = config.diversity.reject_near_duplicate and is_near_duplicate_candidate(
            candidate, result.validation_context, config
        )
        if near_duplicate:
            continue
        items.append(
            _CandidateItem(
                peer_index=index,
                result=result,
                candidate=candidate,
                review=review,
                lane_fit=lane_fit,
                diagnostic_like=is_diagnostic_like_candidate(candidate, config),
                near_duplicate=near_duplicate,
                local_selected=candidate.candidate_id == selected_id,
                labels=canonical_labels_for_candidate(candidate, config),
            )
        )
    return items


def _standard_contract_from_candidate(
    *,
    item: _CandidateItem,
    rejected_alternatives: list[dict[str, str]],
    reason: str,
) -> SelectedContract:
    candidate = item.candidate
    variant_stem = _norm(candidate.name or candidate.candidate_id).strip("_") or _norm(
        candidate.candidate_id
    )
    peer_suffix = f"peer{item.peer_index}"
    if peer_suffix not in variant_stem:
        variant_name = f"{variant_stem}_{peer_suffix}"
    else:
        variant_name = variant_stem

    final_labels = item.labels.to_dict()
    local_candidate_id = str(item.result.selected_contract.selected_candidate_id or "")
    override_changed = local_candidate_id != candidate.candidate_id
    steps = [
        ContractStep(
            step=1, action="Create or update the variant-local files listed in this contract."
        ),
    ]
    for offset, change in enumerate(candidate.implementation_sketch.changes[:6], start=2):
        steps.append(ContractStep(step=offset, action=change))
    if len(steps) == 1:
        steps.append(
            ContractStep(
                step=2, action="Implement the selected mechanism with minimal local edits."
            )
        )

    ablation_hooks = list(candidate.ablation_hooks)
    if not ablation_hooks:
        ablation_hooks = [f"disable_{_norm(candidate.candidate_id) or 'selected_mechanism'}"]

    return SelectedContract(
        selected_candidate_id=candidate.candidate_id,
        variant_name=variant_name,
        diversity_cell=_contract_cell(candidate),
        semantic_family=candidate.semantic_family or item.labels.canonical_semantic_family,
        parent_lineage=candidate.parent_lineage or item.labels.canonical_parent_lineage,
        novelty_axis=candidate.novelty_axis or item.labels.canonical_novelty_axis,
        mechanism_hypothesis=candidate.hypothesis,
        why_selected=(
            f"{reason} Final candidate `{candidate.candidate_id}` was chosen after "
            f"canonical QD label normalization; local DIG choice was "
            f"`{local_candidate_id or 'unknown'}`. Mechanism: {candidate.hypothesis}"
        ),
        rejected_alternatives=rejected_alternatives,
        files_to_modify=list(candidate.implementation_sketch.files_to_modify),
        allowed_changes=list(candidate.implementation_sketch.changes),
        forbidden_changes=list(_STANDARD_FORBIDDEN_CHANGES),
        implementation_plan=steps,
        expected_metric_signature=ExpectedMetricSignature(
            primary=candidate.diagnostic_prediction.primary_metric
            or "Primary task metric should improve or remain stable.",
            secondary_or_safety=candidate.diagnostic_prediction.secondary_or_safety_metric
            or "Safety metrics should not regress materially.",
            diagnostic=candidate.diagnostic_prediction.internal_signal
            or "Mechanism-specific diagnostics should move in the predicted direction.",
        ),
        ablation_hooks=ablation_hooks,
        fail_fast_checks=[
            "selected files exist before editing",
            "modified code imports successfully",
            "output schema remains compatible with the task evaluator",
            "ablation hook can disable the new mechanism",
        ],
        contract_amendment_policy=ContractAmendmentPolicy(
            allowed_reasons=list(_AMENDMENT_REASONS),
            required_artifact="contract_amendment.yaml",
        ),
        canonical_labels=final_labels,
        dig_provenance={
            "cohort_qd_enabled": True,
            "cohort_qd_changed": override_changed,
            "local_selected_candidate_id": local_candidate_id,
            "final_selected_candidate_id": candidate.candidate_id,
            "override_reason": reason,
            "selected_contract_source": "cohort_qd_override" if override_changed else "local_dig",
            "canonical_labels": final_labels,
            "expected_metric_signature": {
                "primary": candidate.diagnostic_prediction.primary_metric
                or "Primary task metric should improve or remain stable.",
                "secondary_or_safety": candidate.diagnostic_prediction.secondary_or_safety_metric
                or "Safety metrics should not regress materially.",
                "diagnostic": candidate.diagnostic_prediction.internal_signal
                or "Mechanism-specific diagnostics should move in the predicted direction.",
            },
        },
    )


def _rejected_alternatives(
    item: _CandidateItem, items: list[_CandidateItem]
) -> list[dict[str, str]]:
    alternatives = [
        other
        for other in sorted(
            items, key=lambda candidate: candidate.review.quality_score, reverse=True
        )
        if other.candidate.candidate_id != item.candidate.candidate_id
    ]
    rejected: list[dict[str, str]] = []
    for other in alternatives[:6]:
        rejected.append(
            {
                "candidate_id": other.candidate.candidate_id,
                "reason": (
                    "not selected by cohort-level QD allocator; "
                    f"canonical_cell={'/'.join(other.labels.formal_cell())}; "
                    f"semantic_family={other.labels.canonical_semantic_family}; "
                    f"parent_lineage={other.labels.canonical_parent_lineage}"
                ),
            }
        )
    while len(rejected) < 3:
        rejected.append(
            {
                "candidate_id": f"cohort_alternative_{len(rejected) + 1}",
                "reason": "No additional validated alternative was available in this peer DIG pool.",
            }
        )
    return rejected


def _assign_item(
    item: _CandidateItem,
    *,
    assignments: dict[int, _CandidateItem],
    counts: dict[str, dict[Any, int]],
) -> None:
    assignments[item.peer_index] = item
    cell = _candidate_cell(item)
    family = item.labels.canonical_mechanism_family
    surface = item.labels.canonical_intervention_surface
    intent = item.labels.canonical_intent
    semantic = item.labels.canonical_semantic_family
    parent = item.labels.canonical_parent_lineage
    counts["cell"][cell] = counts["cell"].get(cell, 0) + 1
    counts["family"][family] = counts["family"].get(family, 0) + 1
    counts["surface"][surface] = counts["surface"].get(surface, 0) + 1
    counts["intent"][intent] = counts["intent"].get(intent, 0) + 1
    counts["semantic"][semantic] = counts["semantic"].get(semantic, 0) + 1
    if _countable_parent_lineage(parent):
        counts["parent"][parent] = counts["parent"].get(parent, 0) + 1


def _best_available(
    *,
    all_items: dict[int, list[_CandidateItem]],
    assignments: dict[int, _CandidateItem],
    counts: dict[str, dict[Any, int]],
    limits: dict[str, int],
    config: DIGLiteConfig,
    target_group: Any | None = None,
    predicate: Any | None = None,
) -> _CandidateItem | None:
    best: tuple[float, _CandidateItem] | None = None
    for index, items in all_items.items():
        if index in assignments:
            continue
        for item in items:
            if predicate is not None and not predicate(item):
                continue
            if target_group is not None and not _matches_target_group(
                item.candidate, target_group, config
            ):
                continue
            if _would_violate_caps(item, counts, limits):
                continue
            score = _score_item(item, config=config, counts=counts)
            if target_group is not None:
                score += config.cohort_qd.target_keyword_bonus
            if best is None or score > best[0]:
                best = (score, item)
    return best[1] if best is not None else None


def _write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def allocate_cohort_qd_contracts(
    *,
    dig_results: dict[int, DIGLiteResult],
    config: DIGLiteConfig,
    gen_dir: Path,
) -> dict[int, DIGLiteResult]:
    """Apply cohort-level quality-diversity allocation to successful DIG peers."""

    qd_config = config.cohort_qd
    if not qd_config.enabled or len(dig_results) <= 1:
        return dig_results

    all_items = {
        index: _eligible_items_for_result(index, result, config)
        for index, result in dig_results.items()
    }
    if not any(all_items.values()):
        return dig_results

    cohort_size = len(dig_results)
    limits = _limits(qd_config, cohort_size)
    counts: dict[str, dict[Any, int]] = {
        "cell": {},
        "family": {},
        "surface": {},
        "intent": {},
        "semantic": {},
        "parent": {},
    }
    assignments: dict[int, _CandidateItem] = {}
    assignment_reasons: dict[int, str] = {}
    missed_targets: list[dict[str, Any]] = []

    for group in qd_config.target_keyword_groups:
        target = max(0, int(getattr(group, "min_peers", 0) or 0))
        filled = 0
        while filled < target:
            item = _best_available(
                all_items=all_items,
                assignments=assignments,
                counts=counts,
                limits=limits,
                config=config,
                target_group=group,
            )
            if item is None:
                missed_targets.append(
                    {
                        "name": group.name,
                        "requested": target,
                        "filled": filled,
                        "reason": "no eligible candidate remained for this target group",
                    }
                )
                break
            _assign_item(item, assignments=assignments, counts=counts)
            assignment_reasons[item.peer_index] = f"target_keyword_group:{group.name}"
            filled += 1

    while qd_config.min_distinct_mechanism_families > 0 and len(counts["family"]) < min(
        qd_config.min_distinct_mechanism_families, cohort_size
    ):
        item = _best_available(
            all_items=all_items,
            assignments=assignments,
            counts=counts,
            limits=limits,
            config=config,
            predicate=lambda candidate_item: (
                candidate_item.labels.canonical_mechanism_family not in counts["family"]
            ),
        )
        if item is None:
            break
        _assign_item(item, assignments=assignments, counts=counts)
        assignment_reasons[item.peer_index] = "min_distinct_mechanism_family"

    while qd_config.min_distinct_intervention_surfaces > 0 and len(counts["surface"]) < min(
        qd_config.min_distinct_intervention_surfaces, cohort_size
    ):
        item = _best_available(
            all_items=all_items,
            assignments=assignments,
            counts=counts,
            limits=limits,
            config=config,
            predicate=lambda candidate_item: (
                candidate_item.labels.canonical_intervention_surface not in counts["surface"]
            ),
        )
        if item is None:
            break
        _assign_item(item, assignments=assignments, counts=counts)
        assignment_reasons[item.peer_index] = "min_distinct_intervention_surface"

    while len(assignments) < len(all_items):
        item = _best_available(
            all_items=all_items,
            assignments=assignments,
            counts=counts,
            limits=limits,
            config=config,
        )
        if item is None:
            break
        _assign_item(item, assignments=assignments, counts=counts)
        assignment_reasons[item.peer_index] = "cohort_qd_best_available"

    trace: dict[str, Any] = {
        "enabled": True,
        "peer_count": cohort_size,
        "limits": limits,
        "missed_targets": missed_targets,
        "assignments": [],
        "fallbacks": [],
        "counts": {
            "family": dict(counts["family"]),
            "surface": dict(counts["surface"]),
            "intent": dict(counts["intent"]),
            "semantic_family": dict(counts["semantic"]),
            "parent_lineage": dict(counts["parent"]),
            "cell": {"/".join(map(str, key)): value for key, value in counts["cell"].items()},
        },
    }

    for index, result in sorted(dig_results.items()):
        item = assignments.get(index)
        if item is None:
            trace["fallbacks"].append(
                {
                    "peer_index": index,
                    "reason": "kept local DIG contract because no cohort-eligible candidate was available",
                    "selected_candidate_id": result.selected_contract.selected_candidate_id,
                }
            )
            result.qd_selection = {
                **dict(result.qd_selection or {}),
                "cohort_qd": {"enabled": True, "changed": False, "fallback": True},
            }
            continue

        reason = assignment_reasons.get(index, "cohort_qd")
        changed = item.candidate.candidate_id != result.selected_contract.selected_candidate_id
        if changed:
            contract = _standard_contract_from_candidate(
                item=item,
                rejected_alternatives=_rejected_alternatives(item, all_items.get(index, [])),
                reason=(
                    "Selected by cohort-level QD allocator to preserve generation "
                    f"diversity ({reason})."
                ),
            )
        else:
            contract = result.selected_contract
            contract.why_selected = (
                f"{contract.why_selected}\n\nCohort-level QD allocator kept this local "
                f"DIG selection ({reason})."
            ).strip()
            contract.semantic_family = contract.semantic_family or item.candidate.semantic_family
            contract.parent_lineage = contract.parent_lineage or item.candidate.parent_lineage
            contract.novelty_axis = contract.novelty_axis or item.candidate.novelty_axis

        contract.semantic_family = contract.semantic_family or item.labels.canonical_semantic_family
        contract.parent_lineage = contract.parent_lineage or item.labels.canonical_parent_lineage
        contract.novelty_axis = contract.novelty_axis or item.labels.canonical_novelty_axis
        contract.canonical_labels = item.labels.to_dict()
        contract.dig_provenance = {
            **dict(contract.dig_provenance or {}),
            "cohort_qd_enabled": True,
            "cohort_qd_changed": changed,
            "local_selected_candidate_id": result.selected_contract.selected_candidate_id,
            "final_selected_candidate_id": contract.selected_candidate_id,
            "override_reason": reason,
            "selected_contract_source": "cohort_qd_override" if changed else "local_dig",
            "canonical_labels": item.labels.to_dict(),
            "expected_metric_signature": {
                "primary": contract.expected_metric_signature.primary,
                "secondary_or_safety": contract.expected_metric_signature.secondary_or_safety,
                "diagnostic": contract.expected_metric_signature.diagnostic,
            },
        }

        validation_ctx = result.validation_context or DIGValidationContext()
        contract_ctx = DIGValidationContext(
            peer_lane=validation_ctx.peer_lane,
            selection_policy=validation_ctx.selection_policy,
            allow_adjacent_lane_selected=not item.lane_fit,
            disallowed_file_rules=validation_ctx.disallowed_file_rules,
            known_diversity_signatures=validation_ctx.known_diversity_signatures,
            known_mechanism_texts=validation_ctx.known_mechanism_texts,
            duplicate_threshold=validation_ctx.duplicate_threshold,
        )
        validate_selected_contract(contract, contract_ctx, config)
        validate_selected_contract_matches_candidate(contract, item.candidate)

        result.selected_contract = contract
        _write_yaml(result.selected_contract_path, contract.to_dict())
        result.qd_selection = {
            **dict(result.qd_selection or {}),
            "selected_candidate_id": contract.selected_candidate_id,
            "cohort_qd": {
                "enabled": True,
                "changed": changed,
                "reason": reason,
                "peer_index": index,
                "mechanism_family": contract.diversity_cell.mechanism_family,
                "intervention_surface": contract.diversity_cell.intervention_surface,
                "intent": contract.diversity_cell.intent,
                "semantic_family": item.labels.canonical_semantic_family,
                "parent_lineage": item.labels.canonical_parent_lineage,
                "novelty_axis": item.labels.canonical_novelty_axis,
                "canonical_labels": item.labels.to_dict(),
            },
        }
        _write_yaml(
            result.dig_dir / "cohort_qd_override.yaml",
            result.qd_selection["cohort_qd"],
        )
        trace["assignments"].append(
            {
                "peer_index": index,
                "selected_candidate_id": contract.selected_candidate_id,
                "variant_name": contract.variant_name,
                "changed_from_local": changed,
                "reason": reason,
                "diversity_cell": {
                    "mechanism_family": contract.diversity_cell.mechanism_family,
                    "intervention_surface": contract.diversity_cell.intervention_surface,
                    "intent": contract.diversity_cell.intent,
                },
                "semantic_family": item.labels.canonical_semantic_family,
                "parent_lineage": item.labels.canonical_parent_lineage,
                "novelty_axis": item.labels.canonical_novelty_axis,
                "raw_and_canonical_labels": item.labels.to_dict(),
            }
        )

    _write_yaml(gen_dir / "dig_cohort_allocation.yaml", trace)
    return dig_results
