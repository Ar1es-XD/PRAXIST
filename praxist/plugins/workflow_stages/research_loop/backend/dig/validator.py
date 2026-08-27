"""Validation rules for DIG-Lite artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .config import DIGLiteConfig
from .schema import Candidate, CandidatePool, CandidateReviews, DiversityCell, SelectedContract

FORBIDDEN_PATH_KEYWORDS = (
    "evaluator",
    "eval_contract",
    "metric",
    "metrics",
    "data_split",
    "split",
)
DEFAULT_FORBIDDEN_PATH_FRAGMENTS = (
    "assets/harness/baseline/",
    "assets/harness/env/",
    "assets/harness/eval/",
    "assets/baselines/",
    "baseline_cache/",
    "data/",
    "evaluations/",
    "results/",
    "findings/",
    "frontier",
    "gems",
    "task.yaml",
)
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[a-zA-Z]:/")

REQUIRED_FORBIDDEN_CHANGE_PHRASES = (
    "do not modify evaluator",
    "do not change data split",
    "do not change metric calculation",
)


class DIGValidationError(ValueError):
    """Raised when a DIG artifact violates the gate contract."""


@dataclass
class DIGValidationContext:
    """Context used to validate DIG artifacts against task and peer constraints."""

    peer_lane: dict[str, Any] = field(default_factory=dict)
    selection_policy: dict[str, Any] = field(default_factory=dict)
    allow_adjacent_lane_selected: bool = False
    disallowed_file_rules: list[str] = field(default_factory=list)
    known_diversity_signatures: set[tuple[str, str, str]] = field(default_factory=set)
    known_mechanism_texts: list[str] = field(default_factory=list)
    duplicate_threshold: float = 0.82


def jaccard_token_similarity(a: str, b: str) -> float:
    """Return whitespace-token Jaccard similarity for duplicate checks."""

    ta = {token for token in a.lower().split() if token}
    tb = {token for token in b.lower().split() if token}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def contains_forbidden_path(path: str, task_disallowed_paths: list[str]) -> bool:
    """Return True when a path is disallowed by task rules or generic DIG guards."""

    normalized = str(path or "").replace("\\", "/").lower()
    if normalized.startswith("/") or _WINDOWS_ABSOLUTE_PATH_RE.match(normalized):
        return True
    if any(part == ".." for part in normalized.split("/")):
        return True
    for disallowed in task_disallowed_paths:
        needle = str(disallowed or "").replace("\\", "/").lower().strip()
        if needle and needle in normalized:
            return True
    if any(fragment in normalized for fragment in DEFAULT_FORBIDDEN_PATH_FRAGMENTS):
        return True
    return any(keyword in normalized for keyword in FORBIDDEN_PATH_KEYWORDS)


def validate_candidate_pool(candidate_pool: CandidatePool, config: DIGLiteConfig) -> None:
    """Validate candidate count, mechanism diversity, and falsifier coverage."""

    candidates = candidate_pool.candidates
    if len(candidates) < config.candidate_count:
        raise DIGValidationError(
            f"Expected at least {config.candidate_count} candidates, got {len(candidates)}."
        )

    families = {candidate.mechanism_family for candidate in candidates}
    surfaces = {candidate.intervention_surface for candidate in candidates}
    if len(families) < config.min_mechanism_families:
        raise DIGValidationError(
            f"Expected at least {config.min_mechanism_families} mechanism families."
        )
    if len(surfaces) < config.min_intervention_surfaces:
        raise DIGValidationError(
            f"Expected at least {config.min_intervention_surfaces} intervention surfaces."
        )

    has_falsifier = any(
        candidate.mechanism_family == "diagnostic_falsifier" or candidate.intent == "falsify"
        for candidate in candidates
    )
    if not has_falsifier:
        raise DIGValidationError(
            "Candidate pool must include at least one falsifier or diagnostic candidate."
        )


def validate_reviews(candidate_pool: CandidatePool, reviews: CandidateReviews) -> None:
    """Validate that every generated candidate has a review."""

    candidate_ids = {candidate.candidate_id for candidate in candidate_pool.candidates}
    reviewed_ids = {review.candidate_id for review in reviews.reviews}
    missing = candidate_ids - reviewed_ids
    if missing:
        raise DIGValidationError(f"Missing reviews for candidates: {sorted(missing)}")


def compatible_with_peer_lane(diversity_cell: DiversityCell, peer_lane: dict[str, Any]) -> bool:
    """Return True when a diversity cell satisfies the peer lane preferences."""

    if not peer_lane:
        return True
    family_prefs = {
        str(item)
        for item in peer_lane.get("mechanism_family_preferences", []) or []
        if str(item).strip()
    }
    surface_prefs = {
        str(item)
        for item in peer_lane.get("intervention_surface_preferences", []) or []
        if str(item).strip()
    }
    intent_pref = str(peer_lane.get("intent_preference") or "").strip()

    if family_prefs and diversity_cell.mechanism_family not in family_prefs:
        return False
    if surface_prefs and diversity_cell.intervention_surface not in surface_prefs:
        return False
    return not (intent_pref and diversity_cell.intent != intent_pref)


def is_near_duplicate_contract(
    contract: SelectedContract,
    ctx: DIGValidationContext,
) -> bool:
    """Return True when a selected contract duplicates known cells or text."""

    signature = contract.diversity_cell.as_tuple()
    if signature in ctx.known_diversity_signatures:
        return True

    text = " ".join(
        [
            contract.mechanism_hypothesis,
            contract.why_selected,
            " ".join(contract.allowed_changes),
        ]
    )
    for known_text in ctx.known_mechanism_texts:
        if jaccard_token_similarity(text, str(known_text)) >= ctx.duplicate_threshold:
            return True
    return False


def validate_selected_contract(
    contract: SelectedContract,
    ctx: DIGValidationContext,
    config: DIGLiteConfig,
    *,
    quality_diversity_enabled: bool = True,
) -> None:
    """Validate that the selected contract is safe and implementation-ready."""

    if not contract.variant_name.strip():
        raise DIGValidationError("variant_name is required.")
    if not contract.selected_candidate_id.strip():
        raise DIGValidationError("selected_candidate_id is required.")
    if not contract.mechanism_hypothesis.strip():
        raise DIGValidationError("mechanism_hypothesis is required.")
    if not contract.files_to_modify:
        raise DIGValidationError("files_to_modify must not be empty.")
    for path in contract.files_to_modify:
        if contains_forbidden_path(path, ctx.disallowed_file_rules):
            raise DIGValidationError(f"Contract tries to modify forbidden path: {path}")

    if (
        config.contract.require_ablation_hooks
        and len(contract.ablation_hooks) < config.contract.min_ablation_hooks
    ):
        raise DIGValidationError(
            f"At least {config.contract.min_ablation_hooks} ablation hook(s) required."
        )
    if len(contract.rejected_alternatives) < config.contract.min_rejected_alternatives:
        raise DIGValidationError(
            f"At least {config.contract.min_rejected_alternatives} rejected alternatives required."
        )

    if config.contract.require_forbidden_changes:
        lower_forbidden = [item.lower() for item in contract.forbidden_changes]
        for phrase in REQUIRED_FORBIDDEN_CHANGE_PHRASES:
            if not any(phrase in item for item in lower_forbidden):
                raise DIGValidationError(f"Missing forbidden change phrase: {phrase}")

    if config.contract.require_expected_metric_signature:
        if not contract.expected_metric_signature.primary.strip():
            raise DIGValidationError("expected_metric_signature.primary is required.")
        if not contract.expected_metric_signature.diagnostic.strip():
            raise DIGValidationError("expected_metric_signature.diagnostic is required.")

    if config.contract.require_fail_fast_checks and not contract.fail_fast_checks:
        raise DIGValidationError("fail_fast_checks are required.")

    if not ctx.allow_adjacent_lane_selected and not compatible_with_peer_lane(
        contract.diversity_cell, ctx.peer_lane
    ):
        raise DIGValidationError("Contract diversity_cell is incompatible with peer lane.")

    if (
        quality_diversity_enabled
        and config.diversity.reject_near_duplicate
        and is_near_duplicate_contract(contract, ctx)
    ):
        raise DIGValidationError("Contract is too similar to known frontier/Gems/sibling lanes.")


def validate_selected_contract_matches_candidate(
    contract: SelectedContract,
    candidate: Candidate,
) -> None:
    """Validate that the contract implements the deterministic QD-selected candidate."""

    if contract.selected_candidate_id != candidate.candidate_id:
        raise DIGValidationError(
            "selected_contract selected_candidate_id does not match qd_selection "
            f"({contract.selected_candidate_id!r} != {candidate.candidate_id!r})."
        )
    if contract.diversity_cell.as_tuple() != candidate.diversity_signature.as_tuple():
        raise DIGValidationError(
            "selected_contract diversity_cell does not match selected candidate signature."
        )
    candidate_files = {
        str(path or "").replace("\\", "/").strip()
        for path in candidate.implementation_sketch.files_to_modify
        if str(path or "").strip()
    }
    contract_files = {
        str(path or "").replace("\\", "/").strip()
        for path in contract.files_to_modify
        if str(path or "").strip()
    }
    unexpected_files = sorted(contract_files - candidate_files)
    if unexpected_files:
        raise DIGValidationError(
            "selected_contract files_to_modify must stay within the selected "
            f"candidate implementation sketch: {unexpected_files}"
        )
