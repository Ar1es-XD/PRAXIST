"""Generic evidence-maturity helpers for research-loop control.

The helpers in this module deliberately use task-agnostic effort/coverage
fields. Domain labels such as evidence-stage names remain audit context; they
are not hard-coded task logic.
"""

from __future__ import annotations

import math
from typing import Any

DEFAULT_MIN_EFFORT_RATIO = 0.75
DEFAULT_MIN_COVERAGE_RATIO = 0.80

_RESULT_ARTIFACT_PATH_KEYS = (
    "source_result_path",
    "result_artifact_path",
    "result_path",
    "summary_path",
)
_RESULT_PRODUCER_IDENTITY_KEYS = (
    "child_id",
    "sweep_child_id",
    "child_variant_id",
    "result_variant_id",
    "variant_id",
    "child_variant_name",
    "result_variant_name",
    "canonical_variant_name",
)
RESULT_RESEARCH_METADATA_KEYS = (
    "bottleneck_target",
    "evidence_stage",
    "tradeoff_class",
    "primary_tradeoff",
    "next_step_intent",
    "parent_candidate",
    "parent_usage",
    "evidence_valence",
    "is_negative",
    "failure_mode",
    "diagnostic_role",
    "source_lane",
    "target_lane",
    "coverage_check",
    "mechanism_hypothesis_deliverable",
    "protocol_integrity_violation_count",
)

_EFFORT_RATIO_KEYS = (
    "effort_ratio",
    "maturity_effort_ratio",
    "actual_effort_ratio",
    "compute_effort_ratio",
    "training_effort_ratio",
)
_COVERAGE_RATIO_KEYS = (
    "coverage_ratio",
    "maturity_coverage_ratio",
    "evaluation_coverage_ratio",
    "eval_coverage_ratio",
)
_EFFORT_VALUE_KEY_PAIRS = (
    (
        (
            "actual_effort",
            "actual_effort_units",
            "completed_effort",
            "completed_effort_units",
        ),
        (
            "reference_effort",
            "reference_effort_units",
            "required_effort",
            "required_effort_units",
            "planned_effort",
            "planned_effort_units",
        ),
    ),
    (
        ("actual_epochs", "completed_epochs"),
        ("reference_epochs", "required_epochs", "planned_epochs"),
    ),
    (
        ("actual_steps", "completed_steps"),
        ("reference_steps", "required_steps", "planned_steps"),
    ),
    (
        ("actual_iterations", "completed_iterations"),
        ("reference_iterations", "required_iterations", "planned_iterations"),
    ),
    (
        ("actual_rollouts", "completed_rollouts"),
        ("reference_rollouts", "required_rollouts", "planned_rollouts"),
    ),
)
_COVERAGE_VALUE_KEY_PAIRS = (
    (
        ("completed_required_eval_units", "completed_eval_units", "covered_eval_units"),
        ("total_required_eval_units", "total_eval_units", "required_eval_units"),
    ),
    (
        ("scored_cell_count", "n_scored_cells", "completed_cells"),
        ("n_eval_cells", "cell_count", "total_cells"),
    ),
)
_AUDIT_BOOL_KEYS = (
    "scout_only",
    "is_scout_eval",
    "is_smoke_eval",
    "smoke_only",
    "partial",
    "partial_eval",
    "is_partial_eval",
    "incomplete_eval",
    "is_incomplete_eval",
    "validation_only",
    "suspect",
    "suspect_protocol",
    "protocol_integrity_failed",
)
_DESCRIPTIVE_MODE_BOOL_KEYS = (
    "scout_only",
    "is_scout_eval",
    "is_smoke_eval",
    "smoke_only",
    "partial",
    "partial_cohort",
    "partial_eval",
    "is_partial_eval",
    "capped",
    "result_capped",
    "is_capped",
)
_COMPLETE_BOOL_KEYS = (
    "scored_complete",
    "is_scored_complete",
    "complete_eval",
    "is_complete_eval",
)
_DURABLE_SIGNAL_BOOL_KEYS = (
    "late_after_generation_boundary",
    "validation_only",
    "validation_only_result",
    "excluded_from_durable_frontier",
)
_DURABLE_SIGNAL_STATUS_KEYS = (
    "artifact_signal_status",
    "late_result_policy",
    "durability_scope",
)
_DURABLE_SIGNAL_STATUS_VALUES = frozenset(
    {
        "late_after_generation_boundary",
        "late_quarantined_protected_job",
        "quarantined_signal",
        "validation_signal_only",
        "validation_only",
    }
)


def result_artifact_key(payload: dict[str, Any]) -> tuple[str, str] | None:
    """Return non-conflicting source coordinates for one result artifact."""

    sources = _result_identity_sources(payload)
    coordinates: list[tuple[str, str]] = []
    for source in sources:
        paths = {
            str(source.get(key) or "").strip().replace("\\", "/").removeprefix("./")
            for key in _RESULT_ARTIFACT_PATH_KEYS
            if source.get(key)
        }
        digests = {
            str(source.get("source_result_sha256") or "").strip().lower()
            for _ in (0,)
            if source.get("source_result_sha256")
        }
        if len(paths) > 1 or len(digests) > 1:
            return None
        source_path = next(iter(paths), "")
        source_sha256 = next(iter(digests), "")
        if source_path or source_sha256:
            coordinates.append((source_path, source_sha256))
    if not coordinates:
        return None
    complete = {(path, digest) for path, digest in coordinates if path and digest}
    if len(complete) > 1:
        return None
    if complete:
        selected_path, selected_digest = next(iter(complete))
        if any(
            (path and path != selected_path) or (digest and digest != selected_digest)
            for path, digest in coordinates
        ):
            return None
        return selected_path, selected_digest
    paths = {path for path, _digest in coordinates if path}
    digests = {digest for _path, digest in coordinates if digest}
    if len(paths) > 1 or len(digests) > 1 or paths and digests:
        return None
    return next(iter(paths), ""), next(iter(digests), "")


def has_result_artifact_coordinates(payload: dict[str, Any]) -> bool:
    """Return whether any canonical container declares result coordinates."""

    return any(
        any(source.get(key) not in (None, "") for key in _RESULT_ARTIFACT_PATH_KEYS)
        or source.get("source_result_sha256") not in (None, "")
        for source in _result_identity_sources(payload)
    )


def compact_result_identity_container(source: Any) -> dict[str, Any]:
    """Return only result-identity fields from a canonical nested container."""

    if not isinstance(source, dict):
        return {}
    keys = {
        *_RESULT_PRODUCER_IDENTITY_KEYS,
        *_RESULT_ARTIFACT_PATH_KEYS,
        "canonical_variant_id",
        "source_result_sha256",
    }
    compact = {key: source[key] for key in keys if key in source}
    for container_name in ("metrics", "details", "extra", "current_aggregate"):
        nested = compact_result_identity_container(source.get(container_name))
        if nested:
            compact[container_name] = nested
    return compact


def same_result_artifact(
    left: tuple[str, str] | None,
    right: tuple[str, str] | None,
) -> bool:
    """Return true only for immutable, non-conflicting result snapshots."""

    if left is None or right is None:
        return False
    left_path, left_sha256 = left
    right_path, right_sha256 = right
    if not left_path or not right_path or not left_sha256 or not right_sha256:
        return False
    return left_path == right_path and left_sha256 == right_sha256


def result_snapshot_key(payload: dict[str, Any]) -> tuple[str, str, str] | None:
    """Return producer-scoped immutable identity for one result snapshot."""

    artifact = result_artifact_key(payload)
    if artifact is None or not all(artifact):
        return None
    sources = _result_identity_sources(payload)
    concrete_tokens = {
        token
        for key in _RESULT_PRODUCER_IDENTITY_KEYS
        if key != "variant_id"
        for source in sources
        if (token := _normalize_result_identity(source.get(key)))
    }
    if len(concrete_tokens) > 1:
        return None
    producer_identity = next(iter(concrete_tokens), "")
    if not producer_identity:
        # Keep immutable coordinates for deduplicating two equally
        # unattributed observations.  The empty producer is deliberately not
        # an exact snapshot and must never be resolved from a display name.
        return "", artifact[0], artifact[1]
    return producer_identity, artifact[0], artifact[1]


def same_result_snapshot(
    left: tuple[str, str, str] | None,
    right: tuple[str, str, str] | None,
) -> bool:
    """Match immutable observations without crossing producer attribution.

    Two unattributed observations may deduplicate each other, but an
    unattributed observation never matches an explicitly attributed one.
    """

    return (
        left is not None and right is not None and bool(left[1]) and bool(left[2]) and left == right
    )


def resolve_result_snapshot_producers(
    records: list[tuple[tuple[str, str, str] | None, Any]],
) -> list[tuple[str, str, str] | None]:
    """Return immutable snapshot identities without guessing attribution.

    The fallback value is retained in the call contract for compatibility,
    but display names are not evidence of result ownership. An empty producer
    remains empty and therefore cannot match an explicitly attributed result.
    """

    return [snapshot for snapshot, _fallback in records]


def _result_identity_sources(payload: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    pending = [payload]
    seen: set[int] = set()
    while pending:
        source = pending.pop(0)
        if id(source) in seen:
            continue
        seen.add(id(source))
        sources.append(source)
        pending.extend(
            value
            for key in ("metrics", "details", "extra", "current_aggregate")
            if isinstance((value := source.get(key)), dict)
        )
    return sources


def _normalize_result_identity(value: Any) -> str:
    # Producer-owned IDs are opaque. Case folding preserves compatibility
    # with filesystem-derived names without collapsing distinct separators.
    return str(value or "").strip().casefold()


_INFERRED_COMPLETE_BOOL_KEYS = ("scored_complete", "is_scored_complete")
_PROTOCOL_STAGE_KEYS = (
    "evidence_stage",
    "eval_stage",
    "stage",
)
_TIER_STAGE_KEYS = (
    "tier",
    "tier_reached",
    "completed_tier",
    "candidate_tier",
)
_STAGE_KEYS = (*_PROTOCOL_STAGE_KEYS, *_TIER_STAGE_KEYS)
_COMPLETION_STATUS_KEYS = (
    "result_status",
    "completion_status",
    "eval_status",
    "final_status",
)


def normalize_maturity_policy(raw: Any | None = None) -> dict[str, Any]:
    """Return a compact maturity policy with defaults.

    Only two objective thresholds are required: minimum effort ratio and
    minimum coverage ratio. Extra task fields are preserved only when they are
    generic and advisory.
    """

    raw = raw if isinstance(raw, dict) else {}
    return {
        "min_effort_ratio": _finite_float(
            raw.get("min_effort_ratio"),
            DEFAULT_MIN_EFFORT_RATIO,
        ),
        "min_coverage_ratio": _finite_float(
            raw.get("min_coverage_ratio"),
            DEFAULT_MIN_COVERAGE_RATIO,
        ),
        "require_ratio_gate": _truthy(raw.get("require_ratio_gate", False)),
        "complete_stage_labels": _normalized_labels(raw.get("complete_stage_labels")),
        "preliminary_stage_labels": _normalized_labels(raw.get("preliminary_stage_labels")),
    }


def evidence_maturity_snapshot(
    candidate: dict[str, Any], policy: Any | None = None
) -> dict[str, Any]:
    """Compute generic maturity telemetry for one finding/result candidate.

    ``mature_enough`` is True/False only when effort and coverage ratios can be
    computed. It is None when the candidate lacks ratio fields, allowing legacy
    task specs to fall back to their existing evidence-stage semantics without
    silently inventing maturity.
    """

    policy = normalize_maturity_policy(policy)
    effort_ratio = _ratio_from_candidate(
        candidate,
        ratio_keys=_EFFORT_RATIO_KEYS,
        value_key_pairs=_EFFORT_VALUE_KEY_PAIRS,
        ratio_map_keys=("effort_ratios", "effort_ratio_by_dimension"),
    )
    coverage_ratio = _ratio_from_candidate(
        candidate,
        ratio_keys=_COVERAGE_RATIO_KEYS,
        value_key_pairs=_COVERAGE_VALUE_KEY_PAIRS,
        ratio_map_keys=("coverage_ratios", "coverage_ratio_by_dimension"),
    )
    min_effort = float(policy["min_effort_ratio"])
    min_coverage = float(policy["min_coverage_ratio"])
    can_compute = effort_ratio is not None and coverage_ratio is not None
    observed_ratio_failure = bool(
        (effort_ratio is not None and effort_ratio < min_effort)
        or (coverage_ratio is not None and coverage_ratio < min_coverage)
    )
    configured_stage = _configured_stage_decision(candidate, policy)
    explicit_complete = _explicit_complete_decision(candidate)
    audit_tags = _audit_tags(candidate)
    if protocol_integrity_failed(candidate):
        mature_enough = False
        maturity_basis = "protocol_integrity"
    elif explicit_complete is False:
        # A producer-owned incomplete marker is stronger than derived ratio
        # telemetry. Keep the result as a validation signal, but never present
        # it as mature in any runtime or report view.
        mature_enough = False
        maturity_basis = "explicit_completion_flag"
    elif observed_ratio_failure:
        mature_enough = False
        maturity_basis = "effort_coverage_ratio"
    elif can_compute:
        mature_enough = bool(effort_ratio >= min_effort and coverage_ratio >= min_coverage)
        maturity_basis = "effort_coverage_ratio"
    elif policy["require_ratio_gate"]:
        mature_enough = None
        maturity_basis = "required_ratio_not_computable"
    elif configured_stage is not None:
        if (
            configured_stage
            and _has_descriptive_mode_marker(candidate)
            and not task_authorizes_descriptive_maturity(candidate, policy)
        ):
            mature_enough = False
            maturity_basis = "task_configured_stage_conflict"
        else:
            mature_enough = configured_stage
            maturity_basis = "task_configured_stage"
    elif explicit_complete is not None:
        mature_enough = explicit_complete
        maturity_basis = "explicit_completion_flag"
    else:
        mature_enough = None
        maturity_basis = "not_computable"
    return {
        "mature_enough": mature_enough,
        "maturity_basis": maturity_basis,
        "effort_ratio": effort_ratio,
        "coverage_ratio": coverage_ratio,
        "min_effort_ratio": min_effort,
        "min_coverage_ratio": min_coverage,
        "audit_tags": sorted(audit_tags),
    }


def protocol_integrity_failed(candidate: dict[str, Any]) -> bool:
    """Return the current producer-owned protocol-integrity decision."""

    for source in _fact_sources(candidate):
        observed = False
        if "protocol_integrity_failed" in source:
            observed = True
            if _truthy(source.get("protocol_integrity_failed")):
                return True
        if "protocol_integrity_passed" in source:
            observed = True
            if _completion_boolish(source.get("protocol_integrity_passed")) is False:
                return True
        count = _finite_or_none(source.get("protocol_integrity_violation_count"))
        if count is not None:
            observed = True
            if count > 0:
                return True
        for key in ("protocol_integrity_status", "protocol_status"):
            if key not in source:
                continue
            observed = True
            token = _normalize_label(source.get(key))
            if token in {"failed", "fail", "invalid", "protocol_invalid"}:
                return True
        if observed:
            return False
    return False


def task_authorizes_descriptive_maturity(
    candidate: dict[str, Any],
    policy: Any | None = None,
    *,
    maturity: dict[str, Any] | None = None,
) -> bool:
    """Return whether the task explicitly makes the reported mode mature.

    A task can express that authorization either through its configured stage
    vocabulary or through its configured effort/coverage thresholds.  Callers
    that already computed a maturity snapshot pass it here so every surface
    uses the same decision without recursively recomputing it.  Coarse
    ``partial``, ``scout``, or ``capped`` descriptors remain truthful audit
    context; they do not override a task-owned ratio decision.
    """

    if isinstance(maturity, dict):
        if maturity.get("maturity_basis") == "effort_coverage_ratio":
            return maturity.get("mature_enough") is True
        if maturity.get("mature_enough") is False:
            return False
    normalized = normalize_maturity_policy(policy)
    complete = set(normalized["complete_stage_labels"])
    preliminary = set(normalized["preliminary_stage_labels"])
    if not complete:
        return False
    for source in _fact_sources(candidate):
        observed = {
            token
            for key in _PROTOCOL_STAGE_KEYS
            if key in source
            for token in (_normalize_label(source.get(key)),)
            if token
        }
        if not observed:
            continue
        return bool(observed.intersection(complete)) and not bool(
            observed.intersection(preliminary)
        )
    return False


def _has_descriptive_mode_marker(candidate: dict[str, Any]) -> bool:
    return any(
        _truthy(value)
        for key in _DESCRIPTIVE_MODE_BOOL_KEYS
        for value in _candidate_values(candidate, key)
    )


def durable_promotion_exclusion(candidate: dict[str, Any]) -> str | None:
    """Return an explicit reason why a measured result remains signal-only.

    Measurement maturity and durable routing are separate contracts. A task
    may authorize a reduced protocol as mature, while still marking an
    individual result as validation-only or ineligible for promotion.
    """

    promotion, promotion_key = _resolved_fact_bool(
        candidate,
        "promotion_eligible",
        "clean_promotion_eligible",
    )
    if promotion is False:
        return f"{promotion_key}=false"
    signal_only, signal_key = _resolved_fact_bool(candidate, *_DURABLE_SIGNAL_BOOL_KEYS)
    if signal_only is True:
        return signal_key
    for source in _fact_sources(candidate):
        for key in _DURABLE_SIGNAL_STATUS_KEYS:
            if key not in source:
                continue
            if _normalize_label(source.get(key)) in _DURABLE_SIGNAL_STATUS_VALUES:
                return key
    return None


def resolved_fact_bool(candidate: dict[str, Any], *keys: str) -> bool | None:
    """Resolve equivalent boolean facts using current-to-legacy precedence."""

    decision, _ = _resolved_fact_bool(candidate, *keys)
    return decision


def _resolved_fact_bool(
    candidate: dict[str, Any],
    *keys: str,
) -> tuple[bool | None, str | None]:
    for source in _fact_sources(candidate):
        observed = [
            (key, parsed)
            for key in keys
            if key in source
            for parsed in (_completion_boolish(source.get(key)),)
            if parsed is not None
        ]
        if not observed:
            continue
        for key, parsed in observed:
            if parsed is False:
                return False, key
        return True, observed[0][0]
    return None, None


def missing_required_ratio_telemetry(
    candidate: dict[str, Any], policy: Any | None = None
) -> tuple[str, ...]:
    """Return required maturity ratios that cannot be computed from a result."""

    normalized_policy = normalize_maturity_policy(policy)
    if not normalized_policy["require_ratio_gate"]:
        return ()
    snapshot = evidence_maturity_snapshot(candidate, normalized_policy)
    return tuple(key for key in ("effort_ratio", "coverage_ratio") if snapshot.get(key) is None)


def has_explicit_false_completion(candidate: dict[str, Any]) -> bool:
    """Return whether a producer explicitly marked evaluation incomplete.

    Older Praxist releases sometimes persisted a derived ``scored_complete=false``
    together with ``_inferred_scored_complete=true``.  That derived value is
    not producer evidence and must not override task-owned ratio or stage
    maturity.  Explicit false values remain hard evidence.
    """

    return _explicit_complete_decision(candidate) is False


def _normalized_labels(raw: Any) -> tuple[str, ...]:
    values = raw if isinstance(raw, (list, tuple, set)) else [raw]
    return tuple(
        dict.fromkeys(token for token in (_normalize_label(value) for value in values) if token)
    )


def _normalize_label(value: Any) -> str:
    return "_".join(str(value or "").strip().lower().replace("-", " ").split())


def _configured_stage_decision(candidate: dict[str, Any], policy: dict[str, Any]) -> bool | None:
    for keys in (_PROTOCOL_STAGE_KEYS, _TIER_STAGE_KEYS):
        observed_in_group = False
        for source in _fact_sources(candidate):
            observed = {
                token
                for key in keys
                if key in source
                for token in (_normalize_label(source.get(key)),)
                if token
            }
            if not observed:
                continue
            observed_in_group = True
            configured_preliminary = bool(observed.intersection(policy["preliminary_stage_labels"]))
            configured_complete = bool(observed.intersection(policy["complete_stage_labels"]))
            if configured_preliminary != configured_complete:
                return configured_complete
            if configured_preliminary:
                return None
            generic_preliminary = bool(
                observed.intersection(
                    {
                        "smoke",
                        "scout",
                        "cheap_probe",
                        "preliminary",
                        "prelim",
                        "partial",
                        "partial_eval",
                        "summary_only",
                    }
                )
            )
            if generic_preliminary:
                return False
        if observed_in_group:
            # An explicit protocol-mode field, even an unknown one, prevents a
            # completion tier from silently redefining the user's protocol.
            return None
    return None


def _explicit_complete_decision(candidate: dict[str, Any]) -> bool | None:
    for source in _fact_sources(candidate):
        observed: list[bool] = []
        inferred_completion = _truthy(source.get("_inferred_scored_complete"))
        for key in _COMPLETE_BOOL_KEYS:
            if key not in source:
                continue
            if inferred_completion and key in _INFERRED_COMPLETE_BOOL_KEYS:
                continue
            parsed = _completion_boolish(source.get(key))
            if parsed is not None:
                observed.append(parsed)
        for key in (
            "incomplete_eval",
            "is_incomplete_eval",
            "summary_only",
            "is_summary_only",
            "unscored_artifact",
        ):
            if key in source and _truthy(source.get(key)):
                observed.append(False)
        inferred_status = _truthy(source.get("_inferred_result_status"))
        for key in _COMPLETION_STATUS_KEYS:
            if key not in source or (key == "result_status" and inferred_status):
                continue
            value = source.get(key)
            token = _normalize_label(value)
            if token in {"scored_complete", "complete_eval", "full_evaluation"}:
                observed.append(True)
            elif token in {
                "failed",
                "incomplete",
                "not_scored_complete",
                "unscored_artifact",
            }:
                observed.append(False)
        if observed:
            return all(observed)
    return None


def _completion_boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "yes", "1", "complete", "passed"}:
            return True
        if token in {"false", "no", "0", "incomplete", "failed"}:
            return False
    return None


def compact_maturity_metadata(
    candidate: dict[str, Any], policy: Any | None = None
) -> dict[str, Any]:
    """Return JSON-friendly metadata to copy into existing artifacts."""

    snap = evidence_maturity_snapshot(candidate, policy)
    out = {
        "mature_enough": snap["mature_enough"],
        "maturity_basis": snap["maturity_basis"],
        "min_effort_ratio": snap["min_effort_ratio"],
        "min_coverage_ratio": snap["min_coverage_ratio"],
    }
    if snap["effort_ratio"] is not None:
        out["effort_ratio"] = snap["effort_ratio"]
    if snap["coverage_ratio"] is not None:
        out["coverage_ratio"] = snap["coverage_ratio"]
    if snap["audit_tags"]:
        out["maturity_audit_tags"] = snap["audit_tags"]
    return out


def _ratio_from_candidate(
    candidate: dict[str, Any],
    *,
    ratio_keys: tuple[str, ...],
    value_key_pairs: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...],
    ratio_map_keys: tuple[str, ...],
) -> float | None:
    for source in _fact_sources(candidate):
        ratios = [
            parsed
            for key in ratio_keys
            if key in source
            for parsed in (_finite_or_none(source.get(key)),)
            if parsed is not None
        ]
        for key in ratio_map_keys:
            value = source.get(key)
            if not isinstance(value, dict):
                continue
            ratios.extend(
                parsed
                for parsed in (_finite_or_none(item) for item in value.values())
                if parsed is not None
            )
        for actual_keys, reference_keys in value_key_pairs:
            actual = next(
                (
                    parsed
                    for key in actual_keys
                    for parsed in (_finite_or_none(source.get(key)),)
                    if parsed is not None
                ),
                None,
            )
            reference = next(
                (
                    parsed
                    for key in reference_keys
                    for parsed in (_finite_or_none(source.get(key)),)
                    if parsed is not None
                ),
                None,
            )
            if actual is not None and reference is not None and reference > 0:
                ratios.append(_clamp_ratio(actual / reference))
        if ratios:
            return _clamp_ratio(min(ratios))
    return None


def _candidate_values(candidate: dict[str, Any], key: str) -> list[Any]:
    return [source.get(key) for source in _candidate_sources(candidate) if key in source]


def _candidate_sources(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    admission_metrics = (
        candidate.get("admission_metrics")
        if isinstance(candidate.get("admission_metrics"), dict)
        else {}
    )
    details = candidate.get("details") if isinstance(candidate.get("details"), dict) else {}
    extra = candidate.get("extra") if isinstance(candidate.get("extra"), dict) else {}
    nested_extra = extra.get("extra") if isinstance(extra.get("extra"), dict) else {}
    return _with_current_aggregate_sources(
        [metrics, admission_metrics, details, candidate, extra, nested_extra]
    )


def _fact_sources(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """Return duplicate fact containers in current-to-legacy precedence."""

    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    admission_metrics = (
        candidate.get("admission_metrics")
        if isinstance(candidate.get("admission_metrics"), dict)
        else {}
    )
    details = candidate.get("details") if isinstance(candidate.get("details"), dict) else {}
    extra = candidate.get("extra") if isinstance(candidate.get("extra"), dict) else {}
    nested_extra = extra.get("extra") if isinstance(extra.get("extra"), dict) else {}
    return _with_current_aggregate_sources(
        [candidate, metrics, admission_metrics, details, extra, nested_extra]
    )


def _with_current_aggregate_sources(
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand canonical aggregate facts without traversing arbitrary metrics."""

    expanded: list[dict[str, Any]] = []
    pending = list(sources)
    seen: set[int] = set()
    while pending:
        source = pending.pop(0)
        if id(source) in seen:
            continue
        seen.add(id(source))
        expanded.append(source)
        aggregate = source.get("current_aggregate")
        if isinstance(aggregate, dict):
            pending.append(aggregate)
    return expanded


def _audit_tags(candidate: dict[str, Any]) -> set[str]:
    tags: set[str] = set()
    for key in _AUDIT_BOOL_KEYS:
        for value in _candidate_values(candidate, key):
            if _truthy(value):
                tags.add(key)
    for key in ("evidence_stage", "stage", "tier", "result_status", "final_status"):
        for value in _candidate_values(candidate, key):
            token = str(value or "").strip().lower()
            if token:
                tags.add(f"{key}:{token}")
    return tags


def _finite_float(value: Any, default: float) -> float:
    parsed = _finite_or_none(value)
    return float(default) if parsed is None else parsed


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _clamp_ratio(value: float) -> float:
    if value < 0:
        return 0.0
    return float(value)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}
