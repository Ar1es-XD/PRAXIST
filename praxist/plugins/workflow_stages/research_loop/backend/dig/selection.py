"""Quality-diversity selection helpers for DIG-Lite."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .config import DIGLiteConfig
from .labels import canonical_labels_for_candidate
from .schema import Candidate, CandidatePool, CandidateReview, CandidateReviews
from .validator import (
    DIGValidationContext,
    compatible_with_peer_lane,
    contains_forbidden_path,
    jaccard_token_similarity,
)


@dataclass
class EligibleCandidate:
    """Candidate retained by DIG filtering with its diversity metadata."""

    candidate_id: str
    diversity_cell: dict[str, str]
    quality_score: int
    selection_score: float
    lane_fit: bool
    near_duplicate: bool
    diagnostic_like: bool


@dataclass
class _SelectionItem:
    candidate: Candidate
    review: CandidateReview
    quality_score: int
    selection_score: float
    lane_fit: bool
    near_duplicate: bool
    diagnostic_like: bool


@dataclass
class CellElite:
    """Top candidate retained for one quality-diversity cell."""

    diversity_cell: dict[str, str]
    selected_candidate_id: str
    quality_score: int


@dataclass
class QDSelection:
    """Serializable DIG quality-diversity selection trace."""

    quality_diversity_enabled: bool = True
    peer_lane: dict[str, Any] = field(default_factory=dict)
    selection_policy: dict[str, Any] = field(default_factory=dict)
    eligible_candidates: list[EligibleCandidate] = field(default_factory=list)
    cell_elites: list[CellElite] = field(default_factory=list)
    selected_candidate_id: str = ""
    selection_reason: str = ""
    rejected_close_alternatives: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_diversity_enabled": self.quality_diversity_enabled,
            "peer_lane": self.peer_lane,
            "selection_policy": self.selection_policy,
            "eligible_candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "diversity_cell": item.diversity_cell,
                    "quality_score": item.quality_score,
                    "selection_score": item.selection_score,
                    "lane_fit": item.lane_fit,
                    "near_duplicate": item.near_duplicate,
                    "diagnostic_like": item.diagnostic_like,
                }
                for item in self.eligible_candidates
            ],
            "cell_elites": [
                {
                    "diversity_cell": item.diversity_cell,
                    "selected_candidate_id": item.selected_candidate_id,
                    "quality_score": item.quality_score,
                }
                for item in self.cell_elites
            ],
            "selected_candidate_id": self.selected_candidate_id,
            "selection_reason": self.selection_reason,
            "rejected_close_alternatives": self.rejected_close_alternatives,
        }


def diversity_cell(
    candidate: Candidate, config: DIGLiteConfig | None = None
) -> tuple[str, str, str]:
    """Return the candidate's quality-diversity cell tuple."""

    if config is not None:
        return canonical_labels_for_candidate(candidate, config).formal_cell()
    return candidate.diversity_signature.as_tuple()


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _selection_policy(ctx: DIGValidationContext) -> dict[str, Any]:
    policy = getattr(ctx, "selection_policy", None)
    return dict(policy or {}) if isinstance(policy, dict) else {}


def is_diagnostic_like_candidate(candidate: Candidate, config: DIGLiteConfig) -> bool:
    """Return True when a candidate is primarily diagnostic/control work."""

    diagnostic_intents = {_norm(item) for item in config.innovation.diagnostic_intents}
    diagnostic_families = {_norm(item) for item in config.innovation.diagnostic_mechanism_families}
    family_values = {
        _norm(candidate.mechanism_family),
        _norm(candidate.diversity_signature.mechanism_family),
    }
    intent_values = {_norm(candidate.intent), _norm(candidate.diversity_signature.intent)}
    if intent_values & diagnostic_intents:
        return True
    return bool(family_values & diagnostic_families)


def _candidate_selection_score(
    candidate: Candidate,
    raw_quality_score: int,
    *,
    diagnostic_like: bool,
    slot: str,
    config: DIGLiteConfig,
) -> float:
    score = float(raw_quality_score)
    if not config.innovation.enabled:
        return score
    if slot != "diagnostic" and diagnostic_like:
        score -= float(config.innovation.diagnostic_score_penalty)
    return score


def _cell_dict(cell: tuple[str, str, str]) -> dict[str, str]:
    return {
        "mechanism_family": cell[0],
        "intervention_surface": cell[1],
        "intent": cell[2],
    }


def _candidate_cell_as_contract(candidate: Candidate):
    from .schema import DiversityCell

    sig = candidate.diversity_signature
    return DiversityCell(
        mechanism_family=sig.mechanism_family,
        intervention_surface=sig.intervention_surface,
        intent=sig.intent,
    )


def violates_file_rules(candidate: Candidate, ctx: DIGValidationContext) -> bool:
    """Return True when a candidate tries to modify disallowed paths."""

    return any(
        contains_forbidden_path(path, ctx.disallowed_file_rules)
        for path in candidate.implementation_sketch.files_to_modify
    )


def is_near_duplicate_candidate(
    candidate: Candidate,
    ctx: DIGValidationContext,
    config: DIGLiteConfig | None = None,
) -> bool:
    """Return True when a candidate duplicates known cells or mechanism text."""

    if diversity_cell(candidate, config) in ctx.known_diversity_signatures:
        return True
    text = " ".join(
        [
            candidate.hypothesis,
            candidate.expected_gain_path,
            " ".join(candidate.implementation_sketch.changes),
        ]
    )
    return any(
        jaccard_token_similarity(text, str(known_text)) >= ctx.duplicate_threshold
        for known_text in ctx.known_mechanism_texts
    )


def select_quality_diverse_candidate(
    candidate_pool: CandidatePool,
    reviews: CandidateReviews,
    ctx: DIGValidationContext,
    config: DIGLiteConfig,
    *,
    quality_diversity_enabled: bool = True,
) -> tuple[Candidate, CandidateReview, QDSelection]:
    """Select a strong candidate, optionally applying QD filtering."""

    review_by_id = {review.candidate_id: review for review in reviews.reviews}
    eligible: list[_SelectionItem] = []
    rejected: list[dict[str, str]] = []
    policy = _selection_policy(ctx)
    slot = _norm(policy.get("intent_slot") or policy.get("slot") or "")
    if slot not in {"diagnostic", "forward_innovation"}:
        slot = "forward_innovation" if config.innovation.enforce_forward_slots else ""

    for candidate in candidate_pool.candidates:
        review = review_by_id.get(candidate.candidate_id)
        if review is None:
            rejected.append({"candidate_id": candidate.candidate_id, "reason": "missing review"})
            continue
        if review.fatal_flaws:
            rejected.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "reason": "fatal flaws: " + "; ".join(review.fatal_flaws[:3]),
                }
            )
            continue
        if violates_file_rules(candidate, ctx):
            rejected.append(
                {"candidate_id": candidate.candidate_id, "reason": "violates file rules"}
            )
            continue
        lane_fit = compatible_with_peer_lane(_candidate_cell_as_contract(candidate), ctx.peer_lane)
        if not lane_fit and not config.diversity.allow_adjacent_lane_fallback:
            rejected.append({"candidate_id": candidate.candidate_id, "reason": "lane mismatch"})
            continue
        near_duplicate = (
            quality_diversity_enabled
            and config.diversity.reject_near_duplicate
            and is_near_duplicate_candidate(candidate, ctx, config)
        )
        if near_duplicate:
            rejected.append({"candidate_id": candidate.candidate_id, "reason": "near duplicate"})
            continue
        diagnostic_like = is_diagnostic_like_candidate(candidate, config)
        selection_score = _candidate_selection_score(
            candidate,
            review.quality_score,
            diagnostic_like=diagnostic_like,
            slot=slot,
            config=config,
        )
        eligible.append(
            _SelectionItem(
                candidate=candidate,
                review=review,
                quality_score=review.quality_score,
                selection_score=selection_score,
                lane_fit=lane_fit,
                near_duplicate=near_duplicate,
                diagnostic_like=diagnostic_like,
            )
        )

    if not eligible:
        raise ValueError("No eligible candidates after DIG eligibility filtering.")

    elites: list[_SelectionItem] = []
    if quality_diversity_enabled:
        buckets: dict[tuple[str, str, str], list[_SelectionItem]] = defaultdict(list)
        for item in eligible:
            buckets[diversity_cell(item.candidate, config)].append(item)
        for items in buckets.values():
            elites.append(max(items, key=lambda item: item.selection_score))

    selection_pool = elites if quality_diversity_enabled else eligible
    lane_elites = [item for item in selection_pool if item.lane_fit]
    pool = lane_elites or selection_pool
    if config.innovation.enabled and slot == "forward_innovation":
        forward_pool = [item for item in pool if not item.diagnostic_like]
        if forward_pool:
            pool = forward_pool
    elif config.innovation.enabled and slot == "diagnostic":
        diagnostic_pool = [item for item in pool if item.diagnostic_like]
        if diagnostic_pool:
            pool = diagnostic_pool
    selected = max(pool, key=lambda item: item.selection_score)

    if not quality_diversity_enabled:
        reason = (
            "Quality-diversity selection is disabled; selected the highest-scoring "
            "eligible candidate under the independent DIG innovation-slot policy."
        )
    elif config.innovation.enabled and slot == "forward_innovation":
        if selected.diagnostic_like:
            reason = (
                "Selected a diagnostic-like fallback only because no eligible "
                "forward-innovation candidate survived lane/file/duplicate filtering."
            )
        else:
            reason = (
                "Selected the highest-scoring forward-innovation candidate within "
                "the peer lane; diagnostic/control candidates were retained as "
                "pool coverage but not allowed to dominate this slot."
            )
    elif config.innovation.enabled and slot == "diagnostic":
        reason = (
            "Selected the highest-scoring diagnostic/control candidate for a "
            "cohort diagnostic slot."
            if selected.diagnostic_like
            else "Selected a forward candidate because no diagnostic/control candidate survived."
        )
    else:
        reason = (
            "Selected the highest-quality eligible candidate within the peer lane."
            if selected.lane_fit
            else "Selected the highest-quality adjacent-lane fallback candidate."
        )

    qd = QDSelection(
        quality_diversity_enabled=quality_diversity_enabled,
        peer_lane=dict(ctx.peer_lane or {}),
        selection_policy=policy,
        eligible_candidates=[
            EligibleCandidate(
                candidate_id=item.candidate.candidate_id,
                diversity_cell=_cell_dict(diversity_cell(item.candidate, config)),
                quality_score=item.quality_score,
                selection_score=item.selection_score,
                lane_fit=item.lane_fit,
                near_duplicate=item.near_duplicate,
                diagnostic_like=item.diagnostic_like,
            )
            for item in eligible
        ],
        cell_elites=[
            CellElite(
                diversity_cell=_cell_dict(diversity_cell(item.candidate, config)),
                selected_candidate_id=item.candidate.candidate_id,
                quality_score=item.quality_score,
            )
            for item in elites
        ],
        selected_candidate_id=selected.candidate.candidate_id,
        selection_reason=reason,
        rejected_close_alternatives=rejected[:8],
    )
    return selected.candidate, selected.review, qd
