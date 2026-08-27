"""
Frontier Store — manages promoted results across generations.

After each generation, the top-K findings (by primary metric) are "promoted"
to the Frontier Store. Their workspace snapshots are frozen and made available
to subsequent generations via the `get_frontier` MCP tool.
"""

import contextlib
import json
import logging
import math
import os
import re
import tarfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
    CANONICAL_STATE,
    artifact_semantics,
    explicit_entry_generation_id,
    is_committed_runtime_fact_file,
    is_committed_runtime_fact_source,
)
from praxist.plugins.workflow_stages.research_loop.backend.effective_config import (
    EFFECTIVE_CONFIG_METADATA_KEYS,
    result_effective_config_metadata,
)
from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
    _RESULT_ARTIFACT_PATH_KEYS,
    _RESULT_PRODUCER_IDENTITY_KEYS,
    RESULT_RESEARCH_METADATA_KEYS,
    _explicit_complete_decision,
    compact_maturity_metadata,
    evidence_maturity_snapshot,
    has_explicit_false_completion,
    normalize_maturity_policy,
    protocol_integrity_failed,
    resolve_result_snapshot_producers,
    resolved_fact_bool,
    result_artifact_key,
    result_snapshot_key,
    same_result_artifact,
    same_result_snapshot,
    task_authorizes_descriptive_maturity,
)
from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
    _RESULT_CHILD_IDENTITY_KEYS,
    _infer_result_generation,
    _late_generation_boundary_info,
    _result_summary_metrics,
    is_supported_result_summary_filename,
    iter_result_summary_paths,
    normalized_result_summary,
    result_summary_control_digest,
    result_summary_filename_variant,
    result_summary_variant_name,
)
from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
    _json_digest as _json_digest,
)
from praxist.plugins.workflow_stages.research_loop.backend.resume_state import (
    read_boundary_evidence_checkpoint,
)

logger = logging.getLogger(__name__)

_NONFINITE_NUMERIC_STRINGS = {
    "nan",
    "+nan",
    "-nan",
    "inf",
    "+inf",
    "-inf",
    "infinity",
    "+infinity",
    "-infinity",
}

_FRONTIER_DERIVED_METRIC_KEYS = (
    "frontier_lane",
    "parent_eligible",
    "peer_role",
    "risk_violating_frontier_candidate",
    "risk_repair_required",
    "risk_violation_reason",
    "lane_missing_tier_candidate",
    "lane_non_promotable_candidate",
)
_CANDIDATE_IDENTITY_ALIAS_KEYS = (
    "finding_id",
    "variant_name",
    "variant_id",
    "canonical_variant_id",
    "canonical_variant_name",
    "frontier_entity_key",
    "candidate_entity_key",
    "child_variant_name",
    "child_variant_id",
    "source_path",
    "result_path",
    "source_result_path",
    "selected_source_result_path",
    "canonical_source_result_path",
    "best_available_summary_path",
    "result_artifact_path",
    "summary_path",
)
_CANDIDATE_RETIREMENT_ALIAS_KEYS = (
    "finding_id",
    "source_finding_id",
    "gem_finding_id",
    "frontier_entity_key",
    "candidate_entity_key",
    "child_id",
    "sweep_child_id",
    "result_variant_id",
    "child_variant_name",
    "child_variant_id",
    "result_variant_name",
    "canonical_variant_name",
    "source_path",
    "result_path",
    "source_result_path",
    "result_artifact_path",
    "summary_path",
)
_CANDIDATE_RETIREMENT_PATH_KEYS = {
    "source_path",
    "result_path",
    "source_result_path",
    "result_artifact_path",
    "summary_path",
}
_RESULT_SOURCE_PATH_KEYS = (
    "source_result_path",
    "canonical_source_result_path",
    "best_available_summary_path",
    "summary_path",
    "result_artifact_path",
    "result_path",
)
_RESULT_SOURCE_AUDIT_PATH_KEYS = ("selected_source_result_path",)

_STATUS_TEXT_KEYS = (
    "tier_status",
    "final_status",
    "result_status",
    "status",
    "completion_status",
    "eval_status",
    "scoring_status",
)
_SMOKE_BOOL_KEYS = (
    "is_smoke_eval",
    "smoke_only",
    "unscored_artifact",
)
_SCOUT_BOOL_KEYS = (
    "scout_only",
    "is_scout_eval",
    "summary_only",
    "is_summary_only",
    "partial_cohort",
    "partial_eval",
    "is_partial_eval",
    "incomplete_eval",
    "is_incomplete_eval",
    "capped",
    "is_capped",
    "result_capped",
)


def _compact_result_identity_container(source: Any) -> dict[str, Any]:
    """Keep only immutable-result identity fields from a nested container."""

    if not isinstance(source, dict):
        return {}
    identity_keys = {
        *_CANDIDATE_IDENTITY_ALIAS_KEYS,
        *_RESULT_PRODUCER_IDENTITY_KEYS,
        *EFFECTIVE_CONFIG_METADATA_KEYS,
        "source_result_sha256",
    }
    compact = {key: source[key] for key in identity_keys if key in source}
    for container_name in ("metrics", "details", "extra", "current_aggregate"):
        nested = _compact_result_identity_container(source.get(container_name))
        if nested:
            compact[container_name] = nested
    return compact


def _clear_result_artifact_coordinates(source: Any) -> None:
    """Remove a stale path/digest pair before assigning a repaired source."""

    if not isinstance(source, dict):
        return
    for key in (*_RESULT_ARTIFACT_PATH_KEYS, "source_result_sha256"):
        source.pop(key, None)
    for container_name in ("metrics", "details", "extra", "current_aggregate"):
        _clear_result_artifact_coordinates(source.get(container_name))


_COMPLETE_BOOL_KEYS = (
    "scored_complete",
    "is_scored_complete",
    "complete_eval",
    "is_complete_eval",
)

_CANONICAL_SOURCE_FACT_KEYS = tuple(
    dict.fromkeys(
        (
            *_STATUS_TEXT_KEYS,
            *_SMOKE_BOOL_KEYS,
            *_SCOUT_BOOL_KEYS,
            *_COMPLETE_BOOL_KEYS,
            *EFFECTIVE_CONFIG_METADATA_KEYS,
            "tier",
            "tier_reached",
            "force_all_tiers",
            "wall_time_s",
            "partial",
            "partial_cohort",
            "validation_only",
            "validation_only_result",
            "late_after_generation_boundary",
            "artifact_signal_status",
            "late_result_policy",
            "durability_scope",
            "suspect",
            "suspect_protocol",
            "suspect_fixed_weight_eval",
            "protocol_integrity_failed",
            "protocol_integrity_passed",
            "protocol_integrity_status",
            "protocol_status",
            "protocol_integrity_violation_count",
            "n_hard_constraint_violations",
            "scored_complete_score",
            "evidence_stage",
            "evidence_maturity_rank",
            "mature_enough",
            "maturity_basis",
            "min_effort_ratio",
            "min_coverage_ratio",
            "effort_ratio",
            "coverage_ratio",
            "maturity_audit_tags",
            "scored_cell_count",
            "promotion_eligible",
            "clean_promotion_eligible",
            "parent_eligible",
            "excluded_from_durable_frontier",
            "exclusion_reason",
        )
    )
)


def _walk_for_metric(
    data: Any,
    metric_name: str,
    depth: int = 0,
    _seen: set | None = None,
    _strict_canonical: bool = False,
) -> float | None:
    """Safety-net metric lookup used when ingest-layer normalization
    didn't put the value at the canonical ``metrics[metric_name]`` path.

    Depth-limited DFS. Returns the FIRST canonical numeric value found
    under the given key name. Refuses to parse free-form strings (M3
    from review round 1): regex extraction from "epoch 50, acc=0.873"
    would return 50 not 0.873. The ingest layer is responsible for
    coercing strings; this function only accepts already-numeric
    values. Booleans (which are int subclasses in Python) are rejected
    to avoid True→1.0 / False→0.0 collisions. NaN and Inf are also
    rejected (M4 from review round 2).

    Round 3 M2 fix: tracks visited container ids to handle cyclic
    structures (rare but possible from YAML aliases / object_hook
    deserialization). Without this, a self-referencing dict could
    cause unbounded recursion until the depth-6 limit, but with
    exponential branch fanout per entry. Now: each container is
    visited at most once.
    """
    import math

    if depth > 6:
        return None
    if _seen is None:
        _seen = set()
    # Round 5 C2 fix: use post-pop semantics so siblings can revisit
    # the same shared sub-structure. The set was previously additive,
    # which falsely pruned legitimate sibling branches that referenced
    # a shared config sub-dict (e.g. summary AND metrics both wrapping
    # the same dict). With post-pop, the set only prevents re-entry
    # along the CURRENT recursion path, not across siblings — which is
    # the correct cycle-detection semantics.
    is_container = isinstance(data, (dict, list))
    data_id: int | None = None
    if is_container:
        data_id = id(data)
        if data_id in _seen:
            return None
        _seen.add(data_id)
    try:
        if isinstance(data, dict):
            if metric_name in data:
                v = data[metric_name]
                if isinstance(v, bool):
                    return None  # bool is int subclass; reject
                if isinstance(v, (int, float)):
                    f = float(v)
                    if not math.isfinite(f):
                        return None  # reject NaN / +Inf / -Inf
                    return f
                # String / complex: don't parse; fall through to recurse.
            # Round 5 M1 fix: prefer canonical parent keys before generic DFS.
            # If the finding has both summary[primary] and metrics[primary],
            # the canonical answer is metrics[primary], not first-DFS match.
            preferred_parents = ("metrics", "final_metrics", "aggregated")
            for pp in preferred_parents:
                if pp in data:
                    r = _walk_for_metric(
                        data[pp],
                        metric_name,
                        depth + 1,
                        _seen,
                        _strict_canonical=_strict_canonical,
                    )
                    if r is not None:
                        return r
            # R9-4 fix: when called from _get_metric (strict mode), ONLY
            # walk preferred parents — don't descend into details/notes/
            # exploration_log/seed_breakdown where peers may stash raw
            # per-seed values that would mis-rank as the primary metric.
            # The depth-6 generic DFS is too permissive for production
            # frontier ranking. Non-strict mode preserves legacy behavior.
            if _strict_canonical:
                return None
            for k, v in data.items():
                if k in preferred_parents:
                    continue  # already tried
                r = _walk_for_metric(
                    v, metric_name, depth + 1, _seen, _strict_canonical=_strict_canonical
                )
                if r is not None:
                    return r
        elif isinstance(data, list):
            if _strict_canonical:
                return None
            for item in data:
                r = _walk_for_metric(
                    item,
                    metric_name,
                    depth + 1,
                    _seen,
                    _strict_canonical=_strict_canonical,
                )
                if r is not None:
                    return r
        return None
    finally:
        if is_container and data_id is not None:
            _seen.discard(data_id)


def _norm_token(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value).strip().lower()
    except (TypeError, ValueError):
        return ""


def _norm_token_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {_norm_token(value)} if _norm_token(value) else set()
    if isinstance(value, dict):
        items = value.keys()
    elif isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [value]
    return {token for token in (_norm_token(item) for item in items) if token}


def _coerce_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _normalize_metric_direction(value: Any, *, default: str = "maximize") -> str:
    direction = str(value or default).strip().lower()
    if direction in {"maximize", "minimize"}:
        return direction
    fallback = default if default in {"maximize", "minimize"} else "maximize"
    logger.warning("invalid metric direction %r; using %s", value, fallback)
    return fallback


def _sanitize_nonfinite_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _NONFINITE_NUMERIC_STRINGS:
            return None
        return value
    if isinstance(value, dict):
        return {key: _sanitize_nonfinite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_nonfinite_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_nonfinite_json(item) for item in value]
    return value


def _merged_extra(finding: dict[str, Any]) -> dict[str, Any]:
    extra = finding.get("extra")
    if not isinstance(extra, dict):
        return {}
    nested_extra = extra.get("extra")
    if not isinstance(nested_extra, dict):
        return extra
    merged = dict(extra)
    merged.pop("extra", None)
    merged.update(nested_extra)
    return merged


def _research_metadata_from_finding(finding: dict[str, Any]) -> dict[str, Any]:
    details = finding.get("details") if isinstance(finding.get("details"), dict) else {}
    metrics = finding.get("metrics") if isinstance(finding.get("metrics"), dict) else {}
    extra = _merged_extra(finding)
    out: dict[str, Any] = {}
    for key in RESULT_RESEARCH_METADATA_KEYS:
        for source in (metrics, details, finding, extra):
            value = source.get(key) if isinstance(source, dict) else None
            if value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue
            out[key] = value
            break
    return out


def _metric_value(finding: dict[str, Any], name: str) -> float | None:
    """Return a lane metric from canonical, non-nested finding fields only.

    Lane gates and lane ranking are stricter than legacy primary-metric
    promotion. They should use aggregate metrics that the task explicitly
    reports at the top level of ``metrics`` / ``details`` / the finding
    itself. A recursive DFS can accidentally promote per-task, per-seed, or
    diagnostic nested values as if they were aggregate task evidence.
    """
    details = finding.get("details") if isinstance(finding.get("details"), dict) else {}
    metrics = finding.get("metrics") if isinstance(finding.get("metrics"), dict) else {}
    for source in (metrics, details, finding):
        if not isinstance(source, dict) or name not in source:
            continue
        value = _coerce_finite_float(source.get(name))
        if value is not None:
            return value
    return None


def _canonical_signal_metric_value(finding: dict[str, Any], name: str) -> float | None:
    """Return a validation-signal metric from canonical aggregate containers.

    Validation candidates are not durable frontier evidence, so they may accept
    numeric strings from aggregate containers. This helper still refuses generic
    DFS through arbitrary nested logs/seed breakdowns.
    """

    direct_value = _metric_value(finding, name)
    if direct_value is not None:
        return direct_value
    if str(finding.get("metric_name") or "") == name:
        value = _coerce_finite_float(finding.get("metric_value"))
        if value is not None:
            return value
    if not str(finding.get("metric_name") or "") and name:
        value = _coerce_finite_float(finding.get("metric_value"))
        if value is not None:
            return value
    if str(finding.get("lane_metric_name") or "") == name:
        value = _coerce_finite_float(finding.get("lane_metric_value"))
        if value is not None:
            return value

    def search_canonical_parents(data: Any, depth: int = 0) -> float | None:
        if depth > 3 or not isinstance(data, dict):
            return None
        for parent in ("final_metrics", "aggregated"):
            child = data.get(parent)
            if not isinstance(child, dict):
                continue
            if name in child:
                value = _coerce_finite_float(child.get(name))
                if value is not None:
                    return value
            value = search_canonical_parents(child, depth + 1)
            if value is not None:
                return value
        return None

    details = finding.get("details") if isinstance(finding.get("details"), dict) else {}
    metrics = finding.get("metrics") if isinstance(finding.get("metrics"), dict) else {}
    for source in (metrics, details, finding):
        value = search_canonical_parents(source)
        if value is not None:
            return value
    return None


def _candidate_lane(finding: dict[str, Any]) -> str:
    for key in ("frontier_lane", "promotion_lane", "lane"):
        for value in _candidate_field_values(finding, key):
            token = _norm_token(value)
            if token:
                return token
    return ""


def _candidate_family(finding: dict[str, Any]) -> str:
    for key in ("strategy_family", "family", "variant_family", "benchmark_type"):
        for value in _candidate_field_values(finding, key):
            token = _norm_token(value)
            if token:
                return token
    return ""


def _candidate_role(finding: dict[str, Any]) -> str:
    metrics = finding.get("metrics") if isinstance(finding.get("metrics"), dict) else {}
    details = finding.get("details") if isinstance(finding.get("details"), dict) else {}
    extra = finding.get("extra") if isinstance(finding.get("extra"), dict) else {}
    for key in ("peer_role", "role", "producer_role"):
        for source in (metrics, details, finding, extra):
            value = source.get(key) if isinstance(source, dict) else None
            token = _norm_token(value)
            if token:
                return token
    return ""


def _candidate_tags(finding: dict[str, Any]) -> set[str]:
    tags = set()
    for source in (
        finding,
        finding.get("metrics") if isinstance(finding.get("metrics"), dict) else {},
        finding.get("details") if isinstance(finding.get("details"), dict) else {},
        _merged_extra(finding),
    ):
        tags.update(_norm_token_set(source.get("tags")))
        tags.update(_norm_token_set(source.get("labels")))
        tags.update(_norm_token_set(source.get("roles")))
    for token in (_candidate_lane(finding), _candidate_family(finding), _candidate_role(finding)):
        if token:
            tags.add(token)
    for key in (
        "is_control",
        "is_benchmark",
        "is_process_audit",
        "risk_violating_frontier_candidate",
        "risk_repair_required",
    ):
        if _any_boolish_candidate_field_true(finding, key):
            tags.add(key)
    return tags


def _raw_candidate_field(candidate: dict[str, Any], *keys: str) -> Any:
    """Return the first non-empty field from canonical candidate containers."""

    for value in _candidate_field_values(candidate, *keys):
        return value
    return None


def _candidate_field_values(candidate: dict[str, Any], *keys: str) -> list[Any]:
    """Return all non-empty field values from canonical candidate containers."""

    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    admission_metrics = (
        candidate.get("admission_metrics")
        if isinstance(candidate.get("admission_metrics"), dict)
        else {}
    )
    details = candidate.get("details") if isinstance(candidate.get("details"), dict) else {}
    extra = _merged_extra(candidate)
    sources = [metrics, admission_metrics, details, candidate, extra]
    sources.extend(
        aggregate
        for source in tuple(sources)
        if isinstance((aggregate := source.get("current_aggregate")), dict)
    )
    values: list[Any] = []
    for key in keys:
        for source in sources:
            if not isinstance(source, dict) or key not in source:
                continue
            value = source.get(key)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            values.append(value)
    return values


def _candidate_identity_aliases(candidate: dict[str, Any]) -> set[str]:
    aliases = {_candidate_entity_key(candidate)}
    for key in _CANDIDATE_IDENTITY_ALIAS_KEYS:
        for value in _candidate_field_values(candidate, key):
            if isinstance(value, list):
                aliases.update(str(item).strip() for item in value if item not in (None, ""))
            elif value not in (None, ""):
                aliases.add(str(value).strip())
    for value in _candidate_field_values(candidate, "identity_aliases"):
        if isinstance(value, list):
            aliases.update(str(item).strip() for item in value if item not in (None, ""))
        elif value not in (None, ""):
            aliases.add(str(value).strip())
    return {alias for alias in aliases if alias}


def _candidate_retirement_aliases(candidate: dict[str, Any]) -> set[str]:
    """Specific identities safe for deleting a validation candidate.

    General identity aliases are intentionally broad so validators can detect
    when a scout candidate is being used as a durable parent. Retirement is
    stronger: it removes the scout from future follow-up. Keep that decision
    tied to canonical entity keys, concrete child ids, or artifact paths so a
    promoted sweep child cannot erase a distinct scout from the same family.
    """

    aliases: set[str] = set()
    immutable_snapshot = result_snapshot_key(candidate)
    if immutable_snapshot is not None:
        aliases.add(
            "result_snapshot::"
            + json.dumps(immutable_snapshot, ensure_ascii=True, separators=(",", ":"))
        )
    has_concrete_result_path = any(
        str(value or "").strip()
        for key in _CANDIDATE_RETIREMENT_PATH_KEYS
        for value in _candidate_field_values(candidate, key)
    )
    entity_key = _candidate_entity_key(candidate)
    entity_identity = _identity_variant_token(entity_key) or _identity_token(entity_key)
    if (
        not has_concrete_result_path
        and entity_key
        and not _looks_like_broad_result_family(entity_identity)
    ):
        aliases.add(entity_key)
        if entity_key.startswith("variant::") and entity_identity:
            aliases.add(entity_identity)
    for key in _CANDIDATE_RETIREMENT_ALIAS_KEYS:
        for value in _candidate_field_values(candidate, key):
            values = value if isinstance(value, list) else [value]
            for item in values:
                token = str(item).strip() if item not in (None, "") else ""
                if not token:
                    continue
                identity = _identity_variant_token(token) or _identity_token(token)
                if key in _CANDIDATE_RETIREMENT_PATH_KEYS:
                    if immutable_snapshot is not None:
                        continue
                    aliases.add(token)
                    artifact_variant = _result_artifact_variant_token(token)
                    if artifact_variant and not _looks_like_broad_result_family(artifact_variant):
                        canonical_artifact_variant = (
                            _canonical_variant_alias_token(
                                artifact_variant,
                                candidate=candidate,
                            )
                            or artifact_variant
                        )
                        aliases.add(artifact_variant)
                        aliases.add(canonical_artifact_variant)
                        aliases.add(_variant_entity_key(artifact_variant, candidate=candidate))
                    continue
                if key in {"finding_id", "source_finding_id", "gem_finding_id"}:
                    aliases.add(token)
                    continue
                if has_concrete_result_path and key in {
                    "frontier_entity_key",
                    "candidate_entity_key",
                }:
                    continue
                if not identity or _looks_like_broad_result_family(identity):
                    continue
                if key in {
                    "frontier_entity_key",
                    "candidate_entity_key",
                    "child_id",
                    "sweep_child_id",
                    "result_variant_id",
                    "child_variant_name",
                    "child_variant_id",
                    "result_variant_name",
                    "canonical_variant_name",
                }:
                    aliases.add(token)
                    aliases.add(identity)
                    aliases.add(
                        _canonical_variant_alias_token(identity, candidate=candidate) or identity
                    )
                    aliases.add(_variant_entity_key(identity, candidate=candidate))
    return {alias for alias in aliases if alias}


def _boolish_candidate_field(candidate: dict[str, Any], *keys: str) -> bool | None:
    for value in _candidate_field_values(candidate, *keys):
        parsed = _boolish_candidate_value(value)
        if parsed is not None:
            return parsed
    return None


def _boolish_candidate_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    token = _norm_token(value)
    if token in {"true", "yes", "1", "full", "complete", "completed", "confirmed"}:
        return True
    if token in {"false", "no", "0", "partial", "incomplete", "scout", "smoke"}:
        return False
    return None


def _any_boolish_candidate_field_true(candidate: dict[str, Any], *keys: str) -> bool:
    return any(
        _boolish_candidate_value(value) is True
        for value in _candidate_field_values(candidate, *keys)
    )


def _any_boolish_candidate_field_false(candidate: dict[str, Any], *keys: str) -> bool:
    return any(
        _boolish_candidate_value(value) is False
        for value in _candidate_field_values(candidate, *keys)
    )


def _resolved_boolish_candidate_field(candidate: dict[str, Any], *keys: str) -> bool | None:
    return resolved_fact_bool(candidate, *keys)


def _candidate_protocol_integrity_failed(candidate: dict[str, Any]) -> bool:
    if protocol_integrity_failed(candidate):
        return True
    for value in _candidate_field_values(candidate, "result_status"):
        if _status_has_any(str(value or ""), "protocol_invalid"):
            return True
    return _any_boolish_candidate_field_true(
        candidate,
        "suspect_protocol",
        # Legacy artifact alias. New task outputs should use suspect_protocol.
        "suspect_fixed_weight_eval",
    )


def _drop_legacy_protocol_alias(metrics: dict[str, Any]) -> dict[str, Any]:
    out = dict(metrics)
    legacy = out.pop("suspect_fixed_weight_eval", None)
    if _boolish_candidate_value(legacy) is True:
        out["suspect_protocol"] = True
    return out


def _result_artifact_variant_token(path_token: str) -> str:
    parts = [part for part in path_token.replace("\\", "/").split("/") if part]
    if _is_result_summary_path(path_token):
        result_indexes = [index for index, part in enumerate(parts[:-1]) if part == "results"]
        if result_indexes and result_indexes[-1] + 1 < len(parts) - 1:
            return _identity_token("__".join(parts[result_indexes[-1] + 1 : -1]))
        filename_variant = result_summary_filename_variant(parts[-1])
        if filename_variant:
            return _identity_token(filename_variant)
        if result_indexes:
            # A root-level standard summary has no path-owned variant identity.
            # Let the producer-authored variant_id/variant_name remain canonical.
            return ""
        return _identity_token(parts[-2])
    return ""


def _is_result_summary_path(path_token: str) -> bool:
    parts = [part for part in path_token.replace("\\", "/").split("/") if part]
    return len(parts) >= 2 and is_supported_result_summary_filename(parts[-1])


_FAKE_IDENTITY_TOKENS = {
    "n/a",
    "na",
    "nan",
    "nil",
    "none",
    "null",
    "undefined",
    "unknown",
}


def _identity_token(value: Any) -> str:
    token = _norm_token(value)
    if token in _FAKE_IDENTITY_TOKENS:
        return ""
    return token


def _identity_variant_token(value: Any) -> str:
    token = _identity_token(value)
    if not token:
        return ""
    if "::" not in token:
        if token.startswith("variant:"):
            return _identity_token(token.partition(":")[2])
        if token.startswith("artifact:"):
            return _identity_token(token.partition(":")[2])
        return token
    prefix, _, payload = token.partition("::")
    if prefix == "artifact":
        artifact_variant = _result_artifact_variant_token(payload)
        if artifact_variant:
            return artifact_variant
        if _is_result_summary_path(payload):
            return ""
        return _identity_token(payload)
    if prefix == "variant":
        return _identity_token(payload)
    return token


_ORCHESTRATOR_VARIANT_PREFIX_RE = re.compile(
    r"^(?:gen_?\d+_peer_?\d+_|peer_?\d+_)",
    re.IGNORECASE,
)


def _canonical_variant_alias_token(
    token: str,
    *,
    candidate: dict[str, Any] | None = None,
) -> str:
    """Collapse orchestration-only wrapper names without merging parameters.

    Task result artifacts can enter promotion both as orchestrator-wrapped
    names (for example ``gen0_peer5_strategy_t1``) and as peer-authored
    finding names (``strategy``). These are the same experimental entity and
    should not consume separate frontier/incubator/Gems slots. The
    normalization is deliberately narrow: it removes only generation/peer
    wrappers. Task-authored suffixes remain part of identity unless the producer
    supplies an explicit stable ``variant_id``.
    """

    normalized = _identity_token(token)
    if not normalized:
        return ""

    stripped = _ORCHESTRATOR_VARIANT_PREFIX_RE.sub("", normalized)
    if stripped == normalized:
        return normalized

    normalized = _ORCHESTRATOR_VARIANT_PREFIX_RE.sub("", stripped)
    return normalized or _identity_token(token)


def _variant_entity_key(
    token: str,
    *,
    candidate: dict[str, Any] | None = None,
) -> str:
    canonical = _canonical_variant_alias_token(token, candidate=candidate)
    return f"variant::{canonical or token}"


def _canonical_evidence_source_variant_token(token: str) -> str:
    normalized = _identity_token(token)
    if not normalized:
        return ""
    stripped = _ORCHESTRATOR_VARIANT_PREFIX_RE.sub("", normalized)
    if stripped == normalized:
        return normalized
    stripped = _ORCHESTRATOR_VARIANT_PREFIX_RE.sub("", stripped)
    return stripped or normalized


def _strip_evidence_stage_suffix(
    token: str,
    candidate: dict[str, Any] | None = None,
) -> str:
    # Praxist does not interpret task-authored suffixes. Producers that emit
    # multiple paths for one entity must provide the same stable variant id.
    return _identity_token(token)


def _result_summary_path_variant_token(path_token: str) -> str:
    return _result_artifact_variant_token(path_token)


def _source_group_key(kind: str, token: str) -> str:
    normalized = _identity_token(token)
    return f"{kind}::{normalized}" if normalized else ""


def _looks_like_broad_result_family(token: str) -> bool:
    # Names are opaque. Words such as "family", "grid", or "control" may be
    # ordinary domain terminology and must not alter identity semantics.
    return False


def _candidate_child_identity_token(candidate: dict[str, Any]) -> str:
    tokens: list[str] = []
    for key in _RESULT_CHILD_IDENTITY_KEYS:
        value = _raw_candidate_field(candidate, key)
        if value is None:
            continue
        token = _identity_token(value)
        if token:
            tokens.append(token)
    if not tokens:
        return ""
    return tokens[0]


def _producer_child_identity_token(candidate: dict[str, Any]) -> str:
    """Return a producer-owned top-level child id without nested fallbacks."""

    for key in _RESULT_CHILD_IDENTITY_KEYS:
        token = _identity_token(candidate.get(key))
        if token:
            return token
    return ""


def _candidate_result_path_identity(candidate: dict[str, Any]) -> tuple[str, str]:
    """Return first path token plus any concrete result-artifact child token."""

    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    admission_metrics = (
        candidate.get("admission_metrics")
        if isinstance(candidate.get("admission_metrics"), dict)
        else {}
    )
    details = candidate.get("details") if isinstance(candidate.get("details"), dict) else {}
    extra = _merged_extra(candidate)
    path_keys = (
        "result_path",
        "summary_path",
        "source_result_path",
        "source_path",
        "result_artifact_path",
    )
    path_key_rank = {key: idx for idx, key in enumerate(path_keys)}
    path_records: list[tuple[int, int, int, str, str]] = []
    seq = 0
    for source_rank, source in enumerate((candidate, details, metrics, admission_metrics, extra)):
        if not isinstance(source, dict):
            continue
        for key in path_keys:
            value = source.get(key)
            token = _identity_token(value)
            if not token:
                continue
            child_token = _result_artifact_variant_token(token)
            if _is_result_summary_path(token) and not child_token:
                continue
            path_records.append((source_rank, path_key_rank[key], seq, token, child_token))
            seq += 1
    if not path_records:
        return "", ""

    first_path_token = path_records[0][3]
    child_records = [record for record in path_records if record[4]]
    if not child_records:
        return first_path_token, ""

    broad_identity_tokens = {
        _identity_variant_token(_raw_candidate_field(candidate, key))
        for key in (
            "frontier_entity_key",
            "candidate_entity_key",
            "variant_id",
        )
    }
    broad_identity_tokens = {
        token for token in broad_identity_tokens if _looks_like_broad_result_family(token)
    }
    non_stale_records = [
        record for record in child_records if record[4] not in broad_identity_tokens
    ]
    if non_stale_records:
        child_records = non_stale_records

    non_family_records = [
        record for record in child_records if not _looks_like_broad_result_family(record[4])
    ]
    if non_family_records:
        child_records = non_family_records

    best = min(child_records, key=lambda record: (record[0], record[1], record[2]))
    return first_path_token, best[4]


def _candidate_entity_key(candidate: dict[str, Any]) -> str:
    """Stable variant-level identity used for lane/frontier dedup.

    A single result can enter through both task auto-summary artifacts and
    manual/SQLite findings. Those rows have different finding ids but the same
    variant identity. Dedup at this level prevents PI/Chair evidence packs from
    overweighting one result while preserving unrelated malformed rows by
    falling back to finding id.
    """

    variant_value = _raw_candidate_field(candidate, "variant_name")
    variant_token = _identity_token(variant_value) if variant_value is not None else ""
    producer_child_identity_token = _producer_child_identity_token(candidate)
    child_identity_token = _candidate_child_identity_token(candidate)
    variant_id_value = _raw_candidate_field(candidate, "variant_id")
    variant_id_token = _identity_token(variant_id_value) if variant_id_value is not None else ""
    producer_variant_id_token = _identity_token(candidate.get("variant_id"))
    source_path_token, source_child_token = _candidate_result_path_identity(candidate)

    # Producer-owned top-level child ids and variant ids are authoritative.
    # Artifact paths are only a fallback for new rows: moving one stable result
    # must not create a second entity. Older rows sometimes copied a parent id
    # into nested metrics for every child; those rows retain their historical
    # path fallback until their producer emits a distinct top-level id.
    if producer_child_identity_token:
        return _variant_entity_key(producer_child_identity_token, candidate=candidate)

    if producer_variant_id_token:
        return _variant_entity_key(producer_variant_id_token, candidate=candidate)

    if child_identity_token and not _looks_like_broad_result_family(child_identity_token):
        return _variant_entity_key(child_identity_token, candidate=candidate)

    result_artifact = result_artifact_key(candidate)
    if result_artifact is not None and all(result_artifact):
        artifact_path, artifact_digest = result_artifact
        return (
            f"artifact::{_identity_token(artifact_path)}#sha256={_identity_token(artifact_digest)}"
        )
    if source_child_token:
        return _variant_entity_key(source_child_token, candidate=candidate)

    deferred_broad_persisted_key = ""
    admission_metrics = (
        candidate.get("admission_metrics")
        if isinstance(candidate.get("admission_metrics"), dict)
        else {}
    )
    details = candidate.get("details") if isinstance(candidate.get("details"), dict) else {}
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    extra = _merged_extra(candidate)
    for key in ("frontier_entity_key", "candidate_entity_key"):
        value = next(
            (
                source.get(key)
                for source in (candidate, details, metrics, admission_metrics, extra)
                if source.get(key) not in (None, "")
            ),
            None,
        )
        token = _identity_token(value)
        if "::" in token or token.startswith(("variant:", "artifact:")):
            prefix, _, artifact_path = token.partition("::")
            if not artifact_path and ":" in token:
                prefix, _, artifact_path = token.partition(":")
            if prefix == "artifact":
                artifact_variant = _result_artifact_variant_token(artifact_path)
                if artifact_variant:
                    if _looks_like_broad_result_family(artifact_variant):
                        deferred_broad_persisted_key = _variant_entity_key(
                            artifact_variant,
                            candidate=candidate,
                        )
                        continue
                    return _variant_entity_key(artifact_variant, candidate=candidate)
            if prefix == "finding" and artifact_path:
                return f"finding::{_identity_token(artifact_path)}"
            entity_variant = _identity_variant_token(token)
            if not entity_variant:
                continue
            canonical_token = _variant_entity_key(entity_variant, candidate=candidate)
            if _looks_like_broad_result_family(entity_variant):
                deferred_broad_persisted_key = canonical_token
                continue
            return canonical_token
    if source_path_token:
        return f"artifact::{source_path_token}"
    if variant_token and not _looks_like_broad_result_family(variant_token):
        return _variant_entity_key(variant_token, candidate=candidate)
    if child_identity_token:
        return _variant_entity_key(child_identity_token, candidate=candidate)
    if variant_id_token:
        return _variant_entity_key(variant_id_token, candidate=candidate)
    if variant_token:
        return _variant_entity_key(variant_token, candidate=candidate)
    if deferred_broad_persisted_key:
        return deferred_broad_persisted_key
    for key in ("gem_variant_ref",):
        value = _raw_candidate_field(candidate, key)
        if value is None:
            continue
        token = _identity_token(value)
        if token:
            return _variant_entity_key(token, candidate=candidate)
    for key in ("id", "finding_id", "source_finding_id", "gem_finding_id"):
        value = _raw_candidate_field(candidate, key)
        token = _identity_token(value)
        if token:
            return f"finding::{token}"
    return f"object::{id(candidate)}"


def _lane_capacity_identity(candidate: dict[str, Any]) -> tuple[str, str, str]:
    """Return the identity that consumes one durable lane slot.

    Semantic entity identity remains authoritative everywhere else. For lane
    capacity, complete immutable artifact coordinates are stronger: aliases of
    one artifact share a slot, while independent artifacts from one lineage do
    not collapse together.
    """

    artifact = result_artifact_key(candidate)
    if artifact is not None and same_result_artifact(artifact, artifact):
        artifact_path, artifact_sha256 = artifact
        return "result_artifact", artifact_path, artifact_sha256
    return "semantic_entity", _candidate_entity_key(candidate), ""


def _source_group_token(key: str) -> str:
    return str(key or "").partition("::")[2]


def _safe_stage_source_group_keys(
    path_token: str,
    reported_token: str = "",
    candidate: dict[str, Any] | None = None,
) -> list[str]:
    path_token = _canonical_evidence_source_variant_token(path_token)
    reported_token = _canonical_evidence_source_variant_token(reported_token)
    stripped = _strip_evidence_stage_suffix(path_token, candidate)
    if not stripped or stripped == path_token:
        return []
    if reported_token and reported_token != stripped:
        return []
    keys = [_source_group_key("stage", stripped)]
    concrete_key = _source_group_key("result", stripped)
    if concrete_key:
        keys.append(concrete_key)
    return [key for key in keys if key]


def _candidate_source_group_keys(candidate: dict[str, Any]) -> list[str]:
    keys: list[str] = []

    def add(key: str) -> None:
        if key and key not in keys:
            keys.append(key)

    reported = _raw_candidate_field(candidate, "variant_name")
    reported_token = _identity_token(reported)
    top_level_path_values = [
        candidate.get(key)
        for key in _RESULT_SOURCE_PATH_KEYS
        if isinstance(candidate, dict) and candidate.get(key) not in (None, "")
    ]
    path_values = list(top_level_path_values)
    if not path_values:
        for key in _RESULT_SOURCE_PATH_KEYS:
            path_values.extend(_candidate_field_values(candidate, key))
    for value in path_values:
        child_token = _result_summary_path_variant_token(str(value or ""))
        if child_token:
            add(_source_group_key("result", _canonical_evidence_source_variant_token(child_token)))
            for stage_key in _safe_stage_source_group_keys(
                child_token,
                reported_token,
                candidate,
            ):
                add(stage_key)
    for key in ("result_variant_id", "result_variant_name"):
        value = _raw_candidate_field(candidate, key)
        token = _identity_variant_token(value) or _identity_token(value)
        if token:
            add(
                _source_group_key(
                    "result",
                    _canonical_evidence_source_variant_token(token),
                )
            )
    return keys


def _candidate_status_text(candidate: dict[str, Any]) -> str:
    return " ".join(_candidate_status_text_values(candidate))


def _candidate_status_text_values(candidate: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in _STATUS_TEXT_KEYS:
        values.extend(_candidate_field_values(candidate, key))
    return [str(value or "").lower() for value in values]


def _status_has_any(text: str, *needles: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    if not normalized:
        return False
    tokens = [token for token in normalized.split("_") if token]
    if not tokens:
        return False
    negators = {"not", "non", "no", "without"}
    false_tokens = {"false", "no", "0"}
    for needle in needles:
        needle_tokens = [
            token
            for token in re.sub(r"[^a-z0-9]+", "_", needle.strip().lower()).split("_")
            if token
        ]
        if not needle_tokens:
            continue
        width = len(needle_tokens)
        for idx in range(0, len(tokens) - width + 1):
            if tokens[idx : idx + width] != needle_tokens:
                continue
            if any(token in negators for token in tokens[max(0, idx - 3) : idx]):
                continue
            next_idx = idx + width
            if (
                next_idx < len(tokens)
                and tokens[next_idx] in false_tokens
                and not any(token in false_tokens for token in needle_tokens)
            ):
                continue
            return True
    return False


def _candidate_status_has_any(candidate: dict[str, Any], *needles: str) -> bool:
    return any(_status_has_any(text, *needles) for text in _candidate_status_text_values(candidate))


_NON_RUNTIME_FAILURE_STATUS_MARKERS = {
    "constraint",
    "constraints",
    "hard",
    "promotion",
    "promotable",
    "eligible",
    "eligibility",
    "repair",
    "risk",
    "incomplete",
    "partial",
    "preliminary",
    "prelim",
    "unscored",
}

_BAD_RUNTIME_STATUS_TOKENS = (
    "crash",
    "crashed",
    "error",
    "timeout",
    "timed_out",
    "cancel",
    "cancelled",
    "killed",
    "oom",
    "pending",
    "running",
    "stale",
)


def _status_is_bad_runtime_failure(text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower()).strip("_")
    if not normalized:
        return False
    tokens = {token for token in normalized.split("_") if token}
    if not tokens & {"failed", "failure"}:
        return False
    return not tokens & _NON_RUNTIME_FAILURE_STATUS_MARKERS


def _candidate_has_bad_runtime_status(candidate: dict[str, Any]) -> bool:
    return any(
        _status_has_any(text, *_BAD_RUNTIME_STATUS_TOKENS) or _status_is_bad_runtime_failure(text)
        for text in _candidate_status_text_values(candidate)
    )


def _candidate_has_validation_only_durability_marker(candidate: dict[str, Any]) -> bool:
    """Return True for existing markers that explicitly forbid durable parenting."""

    if _any_boolish_candidate_field_true(
        candidate,
        "late_after_generation_boundary",
        "validation_only",
        "validation_only_result",
    ):
        return True
    for key in ("artifact_signal_status", "late_result_policy", "durability_scope"):
        for value in _candidate_field_values(candidate, key):
            token = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
            if token in {
                "late_after_generation_boundary",
                "late_quarantined_protected_job",
                "quarantined_signal",
                "validation_signal_only",
                "validation_only",
            }:
                return True
    return False


_NON_SIGNAL_NUMERIC_METADATA_KEYS = {
    "generation_id",
    "source_generation_id",
    "n_eval_cells",
    "scored_cell_count",
    "wall_time_s",
    "elapsed_s",
    "duration_s",
    "seed",
    "pid",
}


def _candidate_has_numeric_validation_signal(candidate: dict[str, Any]) -> bool:
    sources: list[dict[str, Any]] = [candidate]
    metrics = candidate.get("metrics")
    if isinstance(metrics, dict):
        sources.append(metrics)
    for source in sources:
        for key, value in source.items():
            if str(key) in _NON_SIGNAL_NUMERIC_METADATA_KEYS:
                continue
            if _coerce_finite_float(value) is not None:
                return True
    return False


def _candidate_is_scored_nonpromotable_validation_signal(candidate: dict[str, Any]) -> bool:
    if not (
        _any_boolish_candidate_field_true(candidate, "excluded_from_durable_frontier")
        or _candidate_has_validation_only_durability_marker(candidate)
    ):
        return False
    if not _candidate_has_validation_only_durability_marker(candidate) and not any(
        str(reason or "").strip()
        for reason in _candidate_field_values(candidate, "exclusion_reason")
    ):
        return False
    return _candidate_has_numeric_validation_signal(candidate)


def _normalized_evidence_stage(candidate: dict[str, Any]) -> str:
    if _candidate_protocol_integrity_failed(candidate):
        return "smoke"
    if _candidate_has_bad_runtime_status(candidate):
        return "smoke"
    if _any_boolish_candidate_field_true(candidate, *_SMOKE_BOOL_KEYS):
        return "smoke"
    if _candidate_status_has_any(candidate, "smoke", "unscored", "not_scored"):
        return "smoke"
    if _candidate_status_has_any(candidate, "un_scored"):
        return "smoke"
    if _any_boolish_candidate_field_true(candidate, "scout_only", "is_scout_eval"):
        return "scout"
    if _candidate_status_has_any(candidate, "scout", "cheap_probe", "preliminary", "prelim"):
        return "scout"
    if _any_boolish_candidate_field_true(candidate, *_SCOUT_BOOL_KEYS):
        return "scout"
    if has_explicit_false_completion(candidate):
        return "scout"
    if _candidate_status_has_any(
        candidate,
        "partial",
        "preliminary",
        "prelim",
        "summary_only",
        "capped",
        "capped_at",
        "cap_at",
        "incomplete",
        "scored_complete_false",
        "complete_eval_false",
    ):
        return "scout"

    stage_tokens = [
        _norm_token(value).replace("-", "_").replace(" ", "_")
        for value in _candidate_field_values(
            candidate,
            "evidence_stage",
            "eval_stage",
            "stage",
            "tier",
            "tier_reached",
            "completed_tier",
            "candidate_tier",
        )
    ]
    if any(
        stage
        in {
            "smoke",
            "sanity",
            "unscored",
            "un_scored",
            "unscored_artifact",
            "failed_or_unscored",
        }
        for stage in stage_tokens
    ):
        return "smoke"
    if any(
        stage
        in {
            "scout",
            "cheap_probe",
            "probe",
            "incomplete",
            "preliminary",
            "prelim",
            "partial",
            "partial_cohort",
            "partial_eval",
            "summary_only",
            "capped",
            "capped_at",
            "cap_at",
        }
        for stage in stage_tokens
    ):
        return "scout"
    if _any_boolish_candidate_field_true(candidate, *_COMPLETE_BOOL_KEYS):
        return "scored_complete"
    return "unknown"


def _evidence_maturity_rank(candidate: dict[str, Any]) -> int:
    """Rank evidence maturity so preliminary runs do not crowd complete results."""

    if _any_boolish_candidate_field_true(candidate, "scout_only", "is_scout_eval"):
        return 1
    stage = _normalized_evidence_stage(candidate)
    maturity_flag = _boolish_candidate_field(candidate, "mature_enough")
    if maturity_flag is True:
        return 2
    ranks = {"unknown": 0, "smoke": 0, "scout": 1, "scored_complete": 2}
    return ranks.get(stage, 0)


def _has_mature_frontier_evidence(candidate: dict[str, Any]) -> bool:
    return _normalized_evidence_stage(candidate) == "scored_complete"


def _maturity_ratio_decision(candidate: dict[str, Any], policy: Any | None) -> bool | None:
    return evidence_maturity_snapshot(candidate, policy).get("mature_enough")


def _has_mature_durable_evidence(candidate: dict[str, Any], policy: Any | None = None) -> bool:
    if (
        _candidate_protocol_integrity_failed(candidate)
        or _candidate_has_bad_runtime_status(candidate)
        or _candidate_has_validation_only_durability_marker(candidate)
        or _any_boolish_candidate_field_true(candidate, "excluded_from_durable_frontier")
    ):
        return False
    maturity_flag = _boolish_candidate_field(candidate, "mature_enough")
    basis = str(_raw_candidate_field(candidate, "maturity_basis") or "")
    if policy is None and basis == "effort_coverage_ratio" and maturity_flag is not None:
        return bool(maturity_flag) and not _candidate_has_hard_incomplete_marker(candidate)
    maturity = evidence_maturity_snapshot(candidate, policy)
    task_stage_is_complete = bool(
        maturity.get("mature_enough") is True
        and task_authorizes_descriptive_maturity(candidate, policy, maturity=maturity)
        and not _candidate_has_non_authorizable_incomplete_marker(candidate)
    )
    if task_stage_is_complete:
        return True
    if _candidate_has_hard_incomplete_marker(candidate):
        return False
    ratio_decision = maturity.get("mature_enough")
    if ratio_decision is not None:
        return bool(ratio_decision)
    if normalize_maturity_policy(policy).get("require_ratio_gate"):
        return False
    if maturity_flag is True:
        return True
    if maturity_flag is False and basis == "effort_coverage_ratio":
        return False
    return _has_mature_frontier_evidence(candidate)


def _is_preliminary_or_incomplete_evidence(candidate: dict[str, Any]) -> bool:
    """Return True for evidence that must not enter the durable frontier.

    Explicit scout/smoke/partial/summary/unscored evidence is ordinary research
    memory only. Unknown-maturity results are handled separately by
    ``_is_durable_frontier_entry`` so they can be retained for validation without
    becoming frontier or Gem parents.
    """

    if _any_boolish_candidate_field_true(candidate, "excluded_from_durable_frontier"):
        return True
    if _candidate_has_bad_runtime_status(candidate):
        return True
    if any(
        str(reason or "").strip().lower() == "preliminary_or_incomplete_evidence"
        for reason in _candidate_field_values(candidate, "exclusion_reason")
    ):
        return True
    if _any_boolish_candidate_field_true(candidate, "scout_only", "is_scout_eval"):
        return True
    if _any_boolish_candidate_field_true(candidate, *_SMOKE_BOOL_KEYS):
        return True
    if _any_boolish_candidate_field_true(candidate, *_SCOUT_BOOL_KEYS):
        return True
    if has_explicit_false_completion(candidate):
        return True
    return _normalized_evidence_stage(candidate) in {"smoke", "scout"}


def _candidate_has_hard_incomplete_marker(candidate: dict[str, Any]) -> bool:
    if _candidate_has_non_authorizable_incomplete_marker(candidate):
        return True
    return _any_boolish_candidate_field_true(
        candidate,
        "partial",
        "partial_eval",
        "is_partial_eval",
        "capped",
        "is_capped",
        "result_capped",
    )


def _candidate_has_non_authorizable_incomplete_marker(candidate: dict[str, Any]) -> bool:
    if _candidate_has_bad_runtime_status(candidate):
        return True
    if _candidate_has_validation_only_durability_marker(candidate):
        return True
    if has_explicit_false_completion(candidate):
        return True
    return _any_boolish_candidate_field_true(
        candidate,
        "incomplete_eval",
        "is_incomplete_eval",
        "summary_only",
        "is_summary_only",
        "unscored_artifact",
        "validation_only",
    ) or _candidate_status_has_any(candidate, "summary_only", "unscored", "not_scored")


def _is_retainable_validation_candidate(candidate: dict[str, Any]) -> bool:
    if _candidate_has_bad_runtime_status(
        candidate
    ) and not _candidate_is_scored_nonpromotable_validation_signal(candidate):
        return False
    if _any_boolish_candidate_field_true(candidate, "summary_only", "is_summary_only"):
        return False
    if _candidate_status_has_any(candidate, "summary_only"):
        return False
    if _any_boolish_candidate_field_true(candidate, "excluded_from_durable_frontier"):
        return bool(
            any(
                str(reason or "").strip()
                for reason in _candidate_field_values(candidate, "exclusion_reason")
            )
        )
    return _normalized_evidence_stage(candidate) in {"smoke", "scout"}


def _is_retainable_hypothesis_validation_candidate(candidate: dict[str, Any]) -> bool:
    if _candidate_has_bad_runtime_status(candidate):
        return False
    if _any_boolish_candidate_field_true(candidate, "summary_only", "is_summary_only"):
        return False
    if _candidate_status_has_any(candidate, "summary_only"):
        return False
    return _normalized_evidence_stage(candidate) in {"unknown", "smoke", "scout"}


def _is_durable_frontier_entry(
    entry: dict[str, Any],
    maturity_policy: Any | None = None,
) -> bool:
    if not isinstance(entry, dict):
        return False
    if _candidate_protocol_integrity_failed(entry) or _candidate_has_bad_runtime_status(entry):
        return False
    if _any_boolish_candidate_field_true(entry, "excluded_from_durable_frontier"):
        return False
    if _has_mature_durable_evidence(entry, maturity_policy):
        return True
    if _candidate_has_hard_incomplete_marker(entry):
        return False
    maturity_flag = _boolish_candidate_field(entry, "mature_enough")
    basis = str(_raw_candidate_field(entry, "maturity_basis") or "")
    if maturity_policy is None and basis == "effort_coverage_ratio" and maturity_flag is not None:
        return bool(maturity_flag)
    ratio_decision = evidence_maturity_snapshot(entry, maturity_policy).get("mature_enough")
    if ratio_decision is False:
        return False
    if normalize_maturity_policy(maturity_policy).get("require_ratio_gate"):
        return False
    if maturity_flag is True:
        return True
    if maturity_flag is False and basis == "effort_coverage_ratio":
        return False
    return not _is_preliminary_or_incomplete_evidence(entry) and _has_mature_frontier_evidence(
        entry
    )


def _is_legacy_committed_tier_entry(
    entry: dict[str, Any],
    maturity_policy: Any | None = None,
) -> bool:
    """Preserve an already-committed complete fact during manifest migration.

    This compatibility path is used only while pruning an existing manifest.
    It does not admit new findings. Explicit failure, incompleteness, exclusion,
    or measured ratio-gate failure still wins over historical membership. A
    newly configured ratio gate must not erase a complete historical entry
    merely because older artifacts did not record ratios.
    """

    if _candidate_protocol_integrity_failed(entry) or _candidate_has_bad_runtime_status(entry):
        return False
    if _any_boolish_candidate_field_true(entry, "excluded_from_durable_frontier"):
        return False
    if _any_boolish_candidate_field_false(
        entry,
        "promotion_eligible",
        "clean_promotion_eligible",
    ):
        return False
    if _has_mature_durable_evidence(entry, maturity_policy):
        return True
    if _candidate_has_hard_incomplete_marker(entry):
        return False
    if evidence_maturity_snapshot(entry, maturity_policy).get("mature_enough") is False:
        return False
    if _has_mature_frontier_evidence(entry):
        return True
    if _normalized_evidence_stage(entry) != "unknown":
        return False
    legacy_stage = _norm_token(_raw_candidate_field(entry, "evidence_stage"))
    if legacy_stage in {
        "t1",
        "t2",
        "t3",
        "full_t1",
        "full_t2",
        "full_t3",
        "full_eval",
        "complete_eval",
        "scored_complete",
    }:
        return True
    return any(
        str(_raw_candidate_field(entry, key) or "").strip()
        for key in ("tier", "tier_reached", "completed_tier", "candidate_tier")
    )


def _is_committed_frontier_entry(
    entry: dict[str, Any],
    maturity_policy: Any | None = None,
) -> bool:
    """Interpret entries already present in the canonical frontier manifest."""

    return _is_durable_frontier_entry(entry, maturity_policy) or _is_legacy_committed_tier_entry(
        entry,
        maturity_policy,
    )


def _evidence_metadata_from_candidate(
    candidate: dict[str, Any],
    maturity_policy: Any | None = None,
) -> dict[str, Any]:
    evidence_stage: Any = _normalized_evidence_stage(candidate)
    reported_stage = _raw_candidate_field(candidate, "evidence_stage")
    reported_stage_token = _norm_token(reported_stage).replace("-", "_").replace(" ", "_")
    normalized_policy = normalize_maturity_policy(maturity_policy)
    task_stage_labels = {
        *normalized_policy.get("complete_stage_labels", ()),
        *normalized_policy.get("preliminary_stage_labels", ()),
    }
    if reported_stage_token and reported_stage_token in task_stage_labels:
        evidence_stage = reported_stage
    out: dict[str, Any] = {
        "evidence_stage": evidence_stage,
        "evidence_maturity_rank": _evidence_maturity_rank(candidate),
        "frontier_entity_key": _candidate_entity_key(candidate),
    }
    out.update(compact_maturity_metadata(candidate, maturity_policy))
    for key in ("scout_only",):
        value = _boolish_candidate_field(candidate, key)
        if value is not None:
            out[key] = bool(value)
    scored_complete = _explicit_complete_decision(candidate)
    if scored_complete is not None:
        out["scored_complete"] = bool(scored_complete)
    for key in ("scored_cell_count", "n_scored_cells", "n_eval_cells", "cell_count"):
        value = _raw_candidate_field(candidate, key)
        count = _coerce_finite_float(value)
        if count is not None:
            out["scored_cell_count"] = int(count)
            break
    return out


def _entity_key_matches_variant_name(entity_key: str, variant_name: str) -> bool:
    if not entity_key.startswith("variant::"):
        return False
    return _norm_token(variant_name) == entity_key.partition("::")[2]


def _matches_lane_filters(finding: dict[str, Any], lane: dict[str, Any]) -> bool:
    lane_name = _candidate_lane(finding)
    family = _candidate_family(finding)
    tags = _candidate_tags(finding)

    include_lanes = _norm_token_set(lane.get("include_lanes"))
    exclude_lanes = _norm_token_set(lane.get("exclude_lanes"))
    include_families = _norm_token_set(lane.get("include_families"))
    exclude_families = _norm_token_set(lane.get("exclude_families"))
    include_tags = _norm_token_set(lane.get("include_tags"))
    exclude_tags = _norm_token_set(lane.get("exclude_tags"))
    include_roles = _norm_token_set(lane.get("include_roles"))
    exclude_roles = _norm_token_set(lane.get("exclude_roles"))
    role = _candidate_role(finding)

    if include_lanes and lane_name not in include_lanes:
        return False
    if lane_name and lane_name in exclude_lanes:
        return False
    if include_families and family not in include_families:
        return False
    if family and family in exclude_families:
        return False
    if include_tags and tags.isdisjoint(include_tags):
        return False
    if exclude_tags and not tags.isdisjoint(exclude_tags):
        return False
    if include_roles and role not in include_roles:
        return False
    if role and role in exclude_roles:
        return False

    def _raw_field_value(name: str) -> Any:
        return _raw_candidate_field(finding, name)

    def _bool_field(name: str) -> bool | None:
        value = _raw_field_value(name)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value == 1:
                return True
            if value == 0:
                return False
        token = _norm_token(value)
        if token in {"true", "yes", "1", "promotable", "passed", "clean"}:
            return True
        if token in {"false", "no", "0", "non-promotable", "failed", "dirty"}:
            return False
        return None

    for metric in lane.get("require_metrics") or []:
        if _metric_value(finding, str(metric)) is None:
            return False
    for metric in lane.get("require_truthy_metrics") or []:
        if _bool_field(str(metric)) is not True:
            return False
    for metric in lane.get("require_falsey_metrics") or []:
        # Falsey requirements are exclusion guards, not mandatory schema
        # fields. Missing/unknown values should pass so permissive task lanes
        # can still accept hand-written or historical findings that predate a
        # new negative flag. Explicitly truthy values remain rejected.
        if _bool_field(str(metric)) is True:
            return False
    for metric, min_value in (lane.get("min_metrics") or {}).items():
        value = _metric_value(finding, str(metric))
        if value is None or value < float(min_value):
            return False
    for metric, max_value in (lane.get("max_metrics") or {}).items():
        value = _metric_value(finding, str(metric))
        if value is None or value > float(max_value):
            return False
    return True


def _lane_values(
    finding: dict[str, Any],
    axes: list[tuple[str, str]],
    optional_axes: list[tuple[str, str]] | None = None,
) -> dict[str, float] | None:
    values: dict[str, float] = {}
    for name, _direction in axes:
        value = _metric_value(finding, name)
        if value is None:
            return None
        values[name] = value
    for name, _direction in optional_axes or []:
        value = _metric_value(finding, name)
        if value is not None:
            values[name] = value
    return values


def _pareto_dominates(
    a: dict[str, float],
    b: dict[str, float],
    axes: list[tuple[str, str]],
) -> bool:
    better_or_equal = True
    strictly_better = False
    for name, direction in axes:
        av = a[name]
        bv = b[name]
        if direction == "minimize":
            if av > bv:
                better_or_equal = False
                break
            if av < bv:
                strictly_better = True
        else:
            if av < bv:
                better_or_equal = False
                break
            if av > bv:
                strictly_better = True
    return better_or_equal and strictly_better


def _normalized_lane_axes(
    lane: dict[str, Any],
    primary_metric: str,
    metric_direction: str,
) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    for entry in lane.get("axes") or []:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            name, direction = str(entry[0]), str(entry[1])
        elif isinstance(entry, dict) and (entry.get("name") or entry.get("metric")):
            name = str(entry.get("name") or entry.get("metric"))
            direction = str(entry.get("direction", "maximize"))
        else:
            continue
        if direction in {"maximize", "minimize"}:
            normalized.append((name, direction))
    return normalized or [(primary_metric, metric_direction)]


def _build_cumulative_lane_views(
    generations: dict[str, Any],
    frontier_lanes: list[dict[str, Any]],
    *,
    maturity_policy: Any | None,
    primary_metric: str,
    metric_direction: str,
    promote_top_k: int,
    entry_is_committed: Callable[[dict[str, Any]], bool] | None = None,
    trust_recorded_lane_membership: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Build canonical compact lane views from the append-only generation ledger."""

    def is_committed(entry: dict[str, Any]) -> bool:
        if entry_is_committed is not None:
            return entry_is_committed(entry)
        return _is_committed_frontier_entry(entry, maturity_policy)

    lanes_by_name = {str(lane.get("name")): lane for lane in frontier_lanes if lane.get("name")}
    entries_by_lane: dict[str, list[dict[str, Any]]] = {
        lane_name: [] for lane_name in lanes_by_name
    }
    uncategorized: list[dict[str, Any]] = []
    for gen_entries in generations.values():
        if not isinstance(gen_entries, list):
            continue
        for entry in gen_entries:
            if not isinstance(entry, dict) or not is_committed(entry):
                continue
            lane_name = str(entry.get("promoted_for_lane") or entry.get("frontier_lane") or "")
            lane = lanes_by_name.get(lane_name)
            if lane is None:
                uncategorized.append(entry)
                continue
            source_candidate = dict(entry)
            source_metrics = (
                dict(entry.get("metrics")) if isinstance(entry.get("metrics"), dict) else {}
            )
            source_lane = str(
                entry.get("source_frontier_lane")
                or source_metrics.get("source_frontier_lane")
                or entry.get("frontier_lane")
                or source_metrics.get("frontier_lane")
                or ""
            ).strip()
            if source_lane:
                source_candidate["frontier_lane"] = source_lane
                source_metrics["frontier_lane"] = source_lane
            source_candidate["metrics"] = source_metrics
            if trust_recorded_lane_membership or _matches_lane_filters(source_candidate, lane):
                entries_by_lane[lane_name].append(entry)
            else:
                uncategorized.append(entry)

    lane_frontiers: dict[str, list[dict[str, Any]]] = {}
    cumulative: list[dict[str, Any]] = []
    seen_capacity_identities: set[tuple[str, str, str]] = set()
    for lane in frontier_lanes:
        lane_name = str(lane.get("name") or "")
        if not lane_name:
            continue
        lane_entries = [
            entry
            for entry in entries_by_lane.get(lane_name, [])
            if _lane_capacity_identity(entry) not in seen_capacity_identities
        ]
        try:
            lane_k = max(1, int(lane.get("k", 1) or 1))
        except (TypeError, ValueError):
            lane_k = 1
        try:
            cap = max(1, int(lane.get("cumulative_cap") or lane_k * 2))
        except (TypeError, ValueError):
            cap = lane_k * 2
        axes = _normalized_lane_axes(lane, primary_metric, metric_direction)
        first_direction = axes[0][1]

        def cumulative_key(
            entry: dict[str, Any],
            direction: str = first_direction,
        ) -> tuple[int, float, int, str]:
            value = _coerce_finite_float(
                entry.get("lane_metric_value", entry.get("metric_value", 0.0))
            )
            directional_value = (
                float("-inf") if value is None else value if direction == "maximize" else -value
            )
            try:
                gen_value = int(entry.get("generation_id", -1))
            except (TypeError, ValueError):
                gen_value = -1
            stable_identity = "|".join(
                (
                    repr(_lane_capacity_identity(entry)),
                    str(
                        entry.get("finding_id")
                        or entry.get("id")
                        or entry.get("variant_name")
                        or ""
                    ),
                )
            )
            return (
                _evidence_maturity_rank(entry),
                directional_value,
                gen_value,
                stable_identity,
            )

        best_by_capacity: dict[tuple[str, str, str], dict[str, Any]] = {}
        for entry in lane_entries:
            capacity_identity = _lane_capacity_identity(entry)
            incumbent = best_by_capacity.get(capacity_identity)
            if incumbent is None or cumulative_key(entry) > cumulative_key(incumbent):
                best_by_capacity[capacity_identity] = entry
        if len(best_by_capacity) < len(lane_entries):
            logger.info(
                "frontier lane '%s': cumulative dedup removed %d duplicate evidence row(s)",
                lane_name,
                len(lane_entries) - len(best_by_capacity),
            )
        deduped_entries = list(best_by_capacity.values())
        ranked_entries = sorted(deduped_entries, key=cumulative_key, reverse=True)
        if lane.get("admit_new_high"):
            values_by_id = {
                id(entry): values
                for entry in deduped_entries
                if (values := _lane_values(entry, axes)) is not None
            }
            pareto_entries = [
                entry
                for entry in deduped_entries
                if id(entry) in values_by_id
                and not any(
                    other_id != id(entry)
                    and _pareto_dominates(other_values, values_by_id[id(entry)], axes)
                    for other_id, other_values in values_by_id.items()
                )
            ]
            kept = sorted(pareto_entries, key=cumulative_key, reverse=True)[:cap]
            kept_ids = {id(entry) for entry in kept}
            for entry in ranked_entries:
                if id(entry) in kept_ids:
                    continue
                if len(kept) >= cap:
                    break
                kept.append(entry)
                kept_ids.add(id(entry))
        else:
            kept = ranked_entries[:cap]
        lane_frontiers[lane_name] = kept
        for entry in kept:
            capacity_identity = _lane_capacity_identity(entry)
            if capacity_identity in seen_capacity_identities:
                continue
            seen_capacity_identities.add(capacity_identity)
            cumulative.append(entry)

    if uncategorized:
        reverse = metric_direction == "maximize"
        missing_default = float("-inf") if reverse else float("inf")

        def uncategorized_metric_value(entry: dict[str, Any]) -> float:
            value = _coerce_finite_float(entry.get("metric_value"))
            return missing_default if value is None else value

        uncategorized.sort(key=uncategorized_metric_value, reverse=reverse)
        for entry in uncategorized[:promote_top_k]:
            capacity_identity = _lane_capacity_identity(entry)
            if capacity_identity in seen_capacity_identities:
                continue
            seen_capacity_identities.add(capacity_identity)
            cumulative.append(entry)
    return lane_frontiers, cumulative


class FrontierStore:
    """
    Manages the frontier: top results promoted across generations.

    Directory layout:
        <base_dir>/
        ├── frontier_manifest.json
        ├── gen_0/
        │   ├── top_1_finding.json
        │   ├── top_1_snapshot.tar.gz
        │   ├── top_2_finding.json
        │   └── top_2_snapshot.tar.gz
        ├── gen_1/...
        └── cumulative_top/
    """

    def __init__(
        self,
        base_dir: Path,
        promote_top_k: int = 2,
        primary_metric: str = "metric_value",
        metric_direction: str = "maximize",
        anchor_metrics: list[tuple[str, str]] | None = None,
        frontier_lanes: list[dict[str, Any]] | None = None,
        validation_signal_metrics: list[Any] | None = None,
        require_tier: bool = False,
        maturity_policy: dict[str, Any] | None = None,
        risk_violating_frontier_enabled: bool = False,
        risk_violating_primary_threshold: float | None = None,
        result_cell_metric_derivations: list[dict[str, Any]] | None = None,
        result_metric_aliases: dict[str, str] | None = None,
    ):
        """
        Args:
            base_dir: Directory where frontier_manifest.json lives.
            promote_top_k: Number of findings to promote per generation
                under the primary_metric ranking.
            primary_metric: Metric used for the main top-K ranking.
            metric_direction: ``"maximize"`` or ``"minimize"`` for
                primary_metric.
            anchor_metrics: Optional list of secondary anchor specs as
                ``[(metric_name, direction), ...]``. Each anchor adds
                ONE additional finding to the per-generation promotion
                set: the variant with the best value on that secondary
                metric (subject to dedup against the primary picks).
                Default ``None`` = old single-metric behavior.
                Example:
                    anchor_metrics=[
                        ("best_case_task_score", "maximize"),
                        ("seed_robustness_std", "minimize"),
                        ("runtime_efficiency", "maximize"),
                    ]
                Anchors break the single-hub anchor effect; peers see
                multiple "best in different dimensions" findings instead
                of one super-hub.
            risk_violating_frontier_enabled: When true, a task may allow
                candidates that beat a task-defined primary-metric threshold
                but violate risk/promotion constraints into the frontier as
                repair targets for PI synthesis.
            risk_violating_primary_threshold: Task-defined baseline threshold
                that must be beaten before a risk-violating finding can be
                admitted as a repair candidate.
            frontier_lanes: Optional task-defined lane specs. If configured,
                each lane independently selects up to k findings using its own
                filters and Pareto axes. This separates deployable candidates
                from benchmark floors, controls, and process-audit artifacts.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.promote_top_k = promote_top_k
        self.primary_metric = primary_metric
        self.metric_direction = _normalize_metric_direction(metric_direction)
        self.anchor_metrics = self._normalize_anchor_metrics(anchor_metrics or [])
        self.validation_signal_metrics = self._normalize_anchor_metrics(
            validation_signal_metrics or []
        )
        self.result_cell_metric_derivations = list(result_cell_metric_derivations or [])
        self.result_metric_aliases = dict(result_metric_aliases or {})
        self.maturity_policy = normalize_maturity_policy(maturity_policy)
        self.frontier_lanes = list(frontier_lanes) if frontier_lanes else []
        self._lanes_allow_non_promotable = any(
            bool(lane.get("allow_non_promotable")) for lane in self.frontier_lanes
        )
        self._lanes_allow_missing_tier = any(
            bool(lane.get("allow_missing_tier")) for lane in self.frontier_lanes
        )
        self.risk_violating_frontier_enabled = bool(risk_violating_frontier_enabled)
        self.risk_violating_primary_threshold = risk_violating_primary_threshold
        # R6-N1 fix: opt-in strict mode for tasks that
        # MANDATE tier metadata in every promotable finding. Defends against
        # peers that forget the field — without strict mode, missing tier
        # silently passes the filter.
        self._require_tier = bool(require_tier)

        self.manifest_path = self.base_dir / "frontier_manifest.json"
        self._manifest = self._load_manifest()
        self._manifest.setdefault("lane_frontiers", {})
        self._manifest.setdefault(
            "validation_candidates",
            {
                "generations": {},
                "cumulative": [],
            },
        )
        lane_policy_changed = self._manifest.get("frontier_lanes") != self.frontier_lanes
        self._manifest["frontier_lanes"] = list(self.frontier_lanes)
        self._manifest["metric_direction"] = self.metric_direction
        changed = lane_policy_changed
        cumulative_rebuilt = False
        source_repaired = self._repair_manifest_canonical_result_sources()
        frontier_pruned = self._prune_durable_frontier_entries(migrate=True)
        if source_repaired or frontier_pruned:
            self._update_cumulative_top()
            cumulative_rebuilt = True
            self._retire_validation_candidates_for_durable_entities()
            changed = True
        if self.frontier_lanes and not cumulative_rebuilt:
            before = json.dumps(
                {
                    "lane_frontiers": self._manifest.get("lane_frontiers"),
                    "cumulative_top": self._manifest.get("cumulative_top"),
                },
                sort_keys=True,
                default=str,
            )
            self._update_cumulative_top()
            after = json.dumps(
                {
                    "lane_frontiers": self._manifest.get("lane_frontiers"),
                    "cumulative_top": self._manifest.get("cumulative_top"),
                },
                sort_keys=True,
                default=str,
            )
            changed = changed or before != after
        if changed:
            self._save_manifest()

    def _canonical_source_metric_specs(self) -> list[tuple[str, str]]:
        specs: list[tuple[str, str]] = []

        def add(name: Any, direction: Any) -> None:
            metric_name = str(name or "").strip()
            if not metric_name:
                return
            normalized_direction = _normalize_metric_direction(
                direction,
                default=self.metric_direction,
            )
            if metric_name not in {existing[0] for existing in specs}:
                specs.append((metric_name, normalized_direction))

        add(self.primary_metric, self.metric_direction)
        for name, direction in self.anchor_metrics:
            add(name, direction)
        for lane in self.frontier_lanes:
            for name, direction in [*self._lane_axes(lane), *self._lane_optional_axes(lane)]:
                add(name, direction)
        return specs

    @staticmethod
    def _summary_source_group_keys(
        summary: dict[str, Any],
        metrics: dict[str, Any],
        *,
        summary_path: Path,
        run_dir: Path,
    ) -> list[str]:
        aggregate = summary.get("current_aggregate")
        aggregate = aggregate if isinstance(aggregate, dict) else {}
        summary_metrics = summary.get("metrics")
        summary_metrics = summary_metrics if isinstance(summary_metrics, dict) else {}
        keys: list[str] = []

        def add(key: str) -> None:
            if key and key not in keys:
                keys.append(key)

        variant = result_summary_variant_name(summary_path, summary, run_dir)
        path_token = _identity_token(variant)
        if path_token:
            concrete = _canonical_evidence_source_variant_token(path_token)
            add(_source_group_key("result", concrete))
            reported = ""
            for source in (aggregate, summary_metrics, summary, metrics):
                if not isinstance(source, dict):
                    continue
                reported = _identity_token(source.get("variant_name"))
                if reported:
                    break
            stage_candidate = {
                **summary,
                "metrics": {**summary_metrics, **aggregate, **metrics},
            }
            for stage_key in _safe_stage_source_group_keys(
                concrete,
                reported,
                stage_candidate,
            ):
                add(stage_key)
        for key in ("result_variant_id", "result_variant_name"):
            for source in (aggregate, summary_metrics, summary, metrics):
                if not isinstance(source, dict):
                    continue
                token = _identity_variant_token(source.get(key)) or _identity_token(source.get(key))
                if token:
                    add(
                        _source_group_key(
                            "result",
                            _canonical_evidence_source_variant_token(token),
                        )
                    )
        return keys

    @staticmethod
    def _source_metric_value(
        candidate: dict[str, Any],
        metric_name: str,
    ) -> float | None:
        value = _metric_value(candidate, metric_name)
        if value is not None:
            return value
        if str(candidate.get("metric_name") or "") == metric_name:
            return _coerce_finite_float(candidate.get("metric_value"))
        return None

    def _best_source_metric(
        self,
        candidate: dict[str, Any],
        specs: list[tuple[str, str]],
    ) -> tuple[str, float | None, str]:
        for metric_name, direction in specs:
            value = self._source_metric_value(candidate, metric_name)
            if value is not None:
                return metric_name, value, direction
        return self.primary_metric, None, self.metric_direction

    def _result_source_rank(
        self,
        candidate: dict[str, Any],
        *,
        metric_name: str,
        metric_direction: str,
        mtime: float,
    ) -> tuple[int, int, int, int, int, int, int, float, float]:
        metric_value = self._source_metric_value(candidate, metric_name)
        if metric_value is None:
            directional_value = float("-inf")
            has_metric = 0
        else:
            directional_value = metric_value if metric_direction == "maximize" else -metric_value
            has_metric = 1
        maturity = evidence_maturity_snapshot(candidate, self.maturity_policy).get("mature_enough")
        maturity_rank = 2 if maturity is True else (0 if maturity is False else 1)
        promotion = resolved_fact_bool(candidate, "promotion_eligible", "clean_promotion_eligible")
        promotion_rank = 2 if promotion is True else (0 if promotion is False else 1)
        durable_rank = (
            0
            if (
                _candidate_has_validation_only_durability_marker(candidate)
                or _any_boolish_candidate_field_true(candidate, "excluded_from_durable_frontier")
            )
            else 1
        )
        return (
            0 if _candidate_protocol_integrity_failed(candidate) else 1,
            durable_rank,
            maturity_rank,
            promotion_rank,
            0 if _is_preliminary_or_incomplete_evidence(candidate) else 1,
            _evidence_maturity_rank(candidate),
            has_metric,
            directional_value,
            mtime,
        )

    def _entry_source_rank(
        self,
        record: dict[str, Any],
        *,
        metric_name: str,
        metric_direction: str,
    ) -> tuple[int, int, int, int, int, int, int, float, float]:
        candidate = record.get("candidate")
        candidate = candidate if isinstance(candidate, dict) else {}
        return self._result_source_rank(
            candidate,
            metric_name=metric_name,
            metric_direction=metric_direction,
            mtime=float(record.get("mtime") or 0.0),
        )

    def _canonical_result_source_index(
        self,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        run_dir = self.base_dir.parent
        if not (run_dir / "results").exists():
            return {}, {}
        specs = self._canonical_source_metric_specs()
        metric_names = [name for name, _direction in specs]
        by_source_path: dict[str, dict[str, Any]] = {}
        sources_by_variant: dict[str, list[dict[str, Any]]] = {}
        for summary_path in iter_result_summary_paths(run_dir):
            try:
                raw = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(raw, dict):
                continue
            try:
                rel = str(summary_path.relative_to(run_dir)).replace("\\", "/")
            except ValueError:
                continue
            summary = normalized_result_summary(
                raw,
                summary_path=summary_path,
                maturity_policy=self.maturity_policy,
            )
            if not isinstance(summary, dict):
                continue
            if not isinstance(summary.get("current_aggregate"), dict):
                flat_metrics = {
                    key: value
                    for key, value in summary.items()
                    if isinstance(value, (bool, int, float, str)) or value is None
                }
                if flat_metrics:
                    summary = dict(summary)
                    summary["current_aggregate"] = flat_metrics
            metrics = _result_summary_metrics(
                summary,
                cell_metric_derivations=self.result_cell_metric_derivations,
                metric_aliases=self.result_metric_aliases,
                scoring_metric_keys=metric_names,
                maturity_policy=self.maturity_policy,
            )
            metrics.update(result_effective_config_metadata(summary))
            metrics["source_result_path"] = rel
            metrics["source_result_sha256"] = result_summary_control_digest(summary)
            source_keys = self._summary_source_group_keys(
                summary,
                metrics,
                summary_path=summary_path,
                run_dir=run_dir,
            )
            if not source_keys:
                continue
            variant = result_summary_variant_name(summary_path, summary, run_dir)
            source_gen_id, gen_source = _infer_result_generation(
                run_dir=run_dir,
                summary_path=summary_path,
                summary=summary,
                variant=variant,
                boundary_gen_id=-1,
            )
            if gen_source == "boundary_fallback":
                summary_metrics = (
                    summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
                )
                fallback_gen = _coerce_finite_float(summary_metrics.get("generation_id"))
                source_gen_id = int(fallback_gen) if fallback_gen is not None else None
            candidate = {
                "variant_name": variant,
                "source_result_path": rel,
                "metrics": metrics,
            }
            try:
                mtime = summary_path.stat().st_mtime
            except OSError:
                mtime = 0.0
            metric_name, metric_value, metric_direction = self._best_source_metric(candidate, specs)
            record = {
                "source_group_keys": source_keys,
                "source_result_path": rel,
                "candidate": candidate,
                "legacy_source_result_sha256": _json_digest(summary),
                "metric_name": metric_name,
                "metric_value": metric_value,
                "metric_direction": metric_direction,
                "source_generation_id": source_gen_id,
                "mtime": mtime,
            }
            by_source_path[rel] = record
            for key in source_keys:
                sources_by_variant.setdefault(key, []).append(record)
        return by_source_path, sources_by_variant

    @staticmethod
    def _manifest_entry_source_path(entry: dict[str, Any]) -> str:
        for key in _RESULT_SOURCE_PATH_KEYS:
            value = entry.get(key)
            token = str(value or "").strip().replace("\\", "/")
            if token:
                return token
        for key in _RESULT_SOURCE_PATH_KEYS:
            value = _raw_candidate_field(entry, key)
            token = str(value or "").strip().replace("\\", "/")
            if token:
                return token
        return ""

    @staticmethod
    def _manifest_entry_generation_id(
        entry: dict[str, Any],
        generation_key: str | None = None,
    ) -> float | None:
        value = _coerce_finite_float(entry.get("generation_id"))
        if value is not None:
            return value
        value = _coerce_finite_float(generation_key)
        if value is not None:
            return value
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        value = _coerce_finite_float(metrics.get("generation_id"))
        if value is not None:
            return value
        details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
        return _coerce_finite_float(details.get("generation_id"))

    def _repair_manifest_entry_canonical_source(
        self,
        entry: dict[str, Any],
        *,
        by_source_path: dict[str, dict[str, Any]],
        sources_by_variant: dict[str, list[dict[str, Any]]],
        generation_key: str | None = None,
    ) -> bool:
        source_path = self._manifest_entry_source_path(entry)
        source_record = by_source_path.get(source_path)
        source_keys = _candidate_source_group_keys(entry)
        if source_record is not None:
            for key in source_record.get("source_group_keys") or []:
                if key and key not in source_keys:
                    source_keys.append(str(key))
        if not source_keys:
            return False
        source_records: list[dict[str, Any]] = []
        seen_record_ids: set[int] = set()
        for key in source_keys:
            for record in sources_by_variant.get(key) or []:
                record_id = id(record)
                if record_id in seen_record_ids:
                    continue
                source_records.append(record)
                seen_record_ids.add(record_id)
        entry_gen = self._manifest_entry_generation_id(entry, generation_key)
        source_family = self._result_source_family(source_record)
        if not source_family and source_path:
            source_family = self._result_source_family(
                {
                    "source_result_path": source_path,
                    "candidate": entry,
                }
            )
        source_stage_keys = {
            str(key)
            for key in (source_record or {}).get("source_group_keys", ())
            if str(key).startswith("stage::")
        }
        source_record_gen = (
            source_record.get("source_generation_id") if isinstance(source_record, dict) else None
        )
        eligible: list[dict[str, Any]] = []
        retained_late_signal = False
        for record in source_records:
            source_gen = record.get("source_generation_id")
            if record is not source_record:
                record_family = self._result_source_family(record)
                same_path_family = bool(
                    source_family
                    and record_family
                    and (
                        source_family == record_family
                        or source_family.startswith(f"{record_family.rstrip('/')}/")
                        or record_family.startswith(f"{source_family.rstrip('/')}/")
                    )
                )
                record_stage_keys = {
                    str(key)
                    for key in record.get("source_group_keys", ())
                    if str(key).startswith("stage::")
                }
                same_stage_family = bool(source_stage_keys.intersection(record_stage_keys))
                missing_source_has_one_unambiguous_match = bool(
                    source_record is None and len(source_records) == 1
                )
                source_path_token = _canonical_evidence_source_variant_token(
                    _result_summary_path_variant_token(source_path)
                )
                record_path_token = _canonical_evidence_source_variant_token(
                    _result_summary_path_variant_token(str(record.get("source_result_path") or ""))
                )
                source_result_keys = {
                    str(key)
                    for key in (source_record or {}).get("source_group_keys", ())
                    if str(key).startswith("result::")
                }
                record_result_keys = {
                    str(key)
                    for key in record.get("source_group_keys", ())
                    if str(key).startswith("result::")
                }
                shared_root_tokens = {
                    _source_group_token(key)
                    for key in source_result_keys.intersection(record_result_keys)
                }
                paths_share_reported_root = any(
                    token
                    and any(
                        source_path_token == token
                        or source_path_token.startswith(f"{token}{separator}")
                        for separator in ("_", "-", "/")
                    )
                    and any(
                        record_path_token == token
                        or record_path_token.startswith(f"{token}{separator}")
                        for separator in ("_", "-", "/")
                    )
                    for token in shared_root_tokens
                )
                source_candidate = (
                    source_record.get("candidate")
                    if isinstance(source_record, dict)
                    and isinstance(source_record.get("candidate"), dict)
                    else {}
                )
                record_candidate = (
                    record.get("candidate") if isinstance(record.get("candidate"), dict) else {}
                )
                upgrades_preliminary_source = bool(
                    source_record is not None
                    and paths_share_reported_root
                    and _is_preliminary_or_incomplete_evidence(source_candidate)
                    and not _is_preliminary_or_incomplete_evidence(record_candidate)
                    and source_gen == source_record_gen
                )
                repairs_future_source = bool(
                    entry_gen is not None
                    and source_record_gen is not None
                    and source_gen is not None
                    and int(source_record_gen) > int(entry_gen)
                    and int(source_gen) <= int(entry_gen)
                )
                if not (
                    same_path_family
                    or same_stage_family
                    or missing_source_has_one_unambiguous_match
                    or upgrades_preliminary_source
                    or repairs_future_source
                ):
                    continue
            boundary_gen = source_gen if source_gen is not None else entry_gen
            record_path = str(record.get("source_result_path") or "").strip()
            late_info = None
            if boundary_gen is not None and record_path:
                boundary_checkpoint = read_boundary_evidence_checkpoint(
                    self.base_dir.parent,
                    int(boundary_gen),
                )
                if boundary_checkpoint is None:
                    evidence_cutoff = None
                    evidence_source_snapshot = None
                else:
                    evidence_cutoff, evidence_source_snapshot = boundary_checkpoint
                late_info = _late_generation_boundary_info(
                    run_dir=self.base_dir.parent,
                    summary_path=self.base_dir.parent / record_path,
                    source_gen_id=int(boundary_gen),
                    evidence_cutoff=evidence_cutoff,
                    evidence_source_snapshot=evidence_source_snapshot,
                    current_result_control_digest=str(
                        _raw_candidate_field(record.get("candidate") or {}, "source_result_sha256")
                        or ""
                    ),
                    prior_result_control_digest=str(
                        _raw_candidate_field(entry, "source_result_sha256") or ""
                    ),
                )
            if late_info is not None:
                candidate = record.get("candidate")
                candidate = dict(candidate) if isinstance(candidate, dict) else {}
                candidate_metrics = candidate.get("metrics")
                candidate_metrics = (
                    dict(candidate_metrics) if isinstance(candidate_metrics, dict) else {}
                )
                candidate_metrics.update(late_info)
                candidate_metrics.update(
                    {
                        "late_after_generation_boundary": True,
                        "artifact_signal_status": "late_after_generation_boundary",
                        "validation_only_result": True,
                        "promotion_eligible": False,
                        "clean_promotion_eligible": False,
                        "parent_eligible": False,
                        "excluded_from_durable_frontier": True,
                        "exclusion_reason": "late_after_generation_boundary",
                    }
                )
                candidate["metrics"] = candidate_metrics
                validation_entry = self._validation_candidate_entry(
                    gen_id=int(entry_gen if entry_gen is not None else boundary_gen),
                    finding=candidate,
                )
                if validation_entry is not None:
                    self._record_validation_candidates(
                        gen_id=int(entry_gen if entry_gen is not None else boundary_gen),
                        entries=[validation_entry],
                    )
                    retained_late_signal = True
                continue
            if entry_gen is None:
                if source_gen is None or record is source_record:
                    eligible.append(record)
                continue
            if source_gen is None:
                source_record_keys = {
                    str(key)
                    for key in (source_record or {}).get("source_group_keys", ())
                    if str(key).startswith("result::")
                }
                record_keys = {
                    str(key)
                    for key in record.get("source_group_keys", ())
                    if str(key).startswith("result::")
                }
                if record is source_record or (
                    source_record_keys.intersection(record_keys)
                    and source_family
                    and self._result_source_family(record) == source_family
                ):
                    eligible.append(record)
                continue
            if int(source_gen) <= int(entry_gen):
                eligible.append(record)
        rank_metric_name = str(
            entry.get("lane_metric_name") or entry.get("metric_name") or self.primary_metric
        )
        rank_metric_direction = _normalize_metric_direction(
            entry.get("lane_metric_direction")
            or entry.get("metric_direction")
            or self.metric_direction,
            default=self.metric_direction,
        )
        best = (
            max(
                eligible,
                key=lambda record: self._entry_source_rank(
                    record,
                    metric_name=rank_metric_name,
                    metric_direction=rank_metric_direction,
                ),
            )
            if eligible
            else None
        )
        if best is None:
            return retained_late_signal

        best_path = str(best.get("source_result_path") or "")
        if not best_path:
            return False
        best_candidate = best.get("candidate")
        best_candidate = best_candidate if isinstance(best_candidate, dict) else {}
        best_artifact = result_artifact_key(best_candidate)
        entry_artifact_before = result_artifact_key(entry)
        same_artifact_path = bool(
            entry_artifact_before is not None
            and best_artifact is not None
            and entry_artifact_before[0]
            and entry_artifact_before[0] == best_artifact[0]
        )
        digest_changed = bool(
            same_artifact_path
            and entry_artifact_before
            and best_artifact
            and entry_artifact_before[1]
            and best_artifact[1]
            and entry_artifact_before[1]
            not in {
                best_artifact[1],
                str(best.get("legacy_source_result_sha256") or ""),
            }
        )
        entry_metric_before = self._source_metric_value(entry, rank_metric_name)
        best_metric_before = self._source_metric_value(best_candidate, rank_metric_name)
        legacy_facts_changed = any(
            (old_value := _raw_candidate_field(entry, key)) is not None
            and old_value != _raw_candidate_field(best_candidate, key)
            for key in (
                *RESULT_RESEARCH_METADATA_KEYS,
                "evidence_stage",
                "result_status",
                "mature_enough",
                "promotion_eligible",
                "parent_eligible",
                "validation_only",
                "validation_only_result",
                "excluded_from_durable_frontier",
            )
        )
        unverifiable_legacy_rewrite = bool(
            same_artifact_path
            and entry_artifact_before
            and best_artifact
            and not entry_artifact_before[1]
            and best_artifact[1]
            and (entry_metric_before != best_metric_before or legacy_facts_changed)
        )
        unscoped_same_path_rewrite = bool(
            entry_gen is not None
            and best.get("source_generation_id") is None
            and same_artifact_path
            and digest_changed
            and _has_mature_durable_evidence(entry, self.maturity_policy)
        )
        if entry_gen is not None and (digest_changed or unverifiable_legacy_rewrite):
            previous_snapshot = dict(entry)
            previous_metrics = entry.get("metrics")
            if isinstance(previous_metrics, dict):
                previous_snapshot["metrics"] = dict(previous_metrics)
            if unverifiable_legacy_rewrite:
                snapshot_metrics = previous_snapshot.get("metrics")
                if not isinstance(snapshot_metrics, dict):
                    snapshot_metrics = {}
                    previous_snapshot["metrics"] = snapshot_metrics
                for key in _CANONICAL_SOURCE_FACT_KEYS:
                    previous_snapshot.pop(key, None)
                    snapshot_metrics.pop(key, None)
                previous_snapshot.update(
                    {
                        "evidence_stage": "unknown",
                        "mature_enough": False,
                        "promotion_eligible": False,
                        "parent_eligible": False,
                        "excluded_from_durable_frontier": True,
                    }
                )
                snapshot_metrics.update(
                    {
                        "evidence_stage": "unknown",
                        "mature_enough": False,
                        "promotion_eligible": False,
                        "parent_eligible": False,
                        "excluded_from_durable_frontier": True,
                    }
                )
            validation_entry = self._validation_candidate_entry(
                gen_id=int(entry_gen),
                finding=previous_snapshot,
            )
            if validation_entry is not None:
                if unverifiable_legacy_rewrite:
                    validation_entry["signal_source"] = "manifest_snapshot"
                self._record_validation_candidates(
                    gen_id=int(entry_gen),
                    entries=[validation_entry],
                )
            if (
                unverifiable_legacy_rewrite
                and best.get("source_generation_id") is None
                and entry_metric_before != best_metric_before
            ):
                current_signal = self._validation_candidate_entry(
                    gen_id=int(entry_gen),
                    finding=best_candidate,
                )
                if current_signal is not None:
                    self._record_validation_candidates(
                        gen_id=int(entry_gen),
                        entries=[current_signal],
                    )
                entry["parent_eligible"] = False
                entry["excluded_from_durable_frontier"] = True
                entry["exclusion_reason"] = "source_changed_without_generation_or_digest"
                entry_metrics = entry.get("metrics")
                if isinstance(entry_metrics, dict):
                    entry_metrics["parent_eligible"] = False
                    entry_metrics["excluded_from_durable_frontier"] = True
                    entry_metrics["exclusion_reason"] = (
                        "source_changed_without_generation_or_digest"
                    )
                return True
        source_changed = best is not source_record
        if best is not source_record and source_record is not None and entry_gen is not None:
            previous_candidate = source_record.get("candidate")
            previous_candidate = previous_candidate if isinstance(previous_candidate, dict) else {}
            previous_maturity = evidence_maturity_snapshot(
                previous_candidate,
                self.maturity_policy,
            ).get("mature_enough")
            previous_promotion = resolved_fact_bool(
                previous_candidate,
                "promotion_eligible",
                "clean_promotion_eligible",
            )
            if (
                previous_maturity is False
                or previous_promotion is False
                or _is_retainable_validation_candidate(previous_candidate)
            ):
                validation_entry = self._validation_candidate_entry(
                    gen_id=int(entry_gen),
                    finding=previous_candidate,
                )
                if validation_entry is not None:
                    self._record_validation_candidates(
                        gen_id=int(entry_gen),
                        entries=[validation_entry],
                    )
        source_fact_before = json.dumps(entry, sort_keys=True, default=str)
        preserved_parent_eligible = (
            _resolved_boolish_candidate_field(entry, "parent_eligible")
            if not source_changed
            and _raw_candidate_field(best_candidate, "parent_eligible") is None
            else None
        )
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        best_metrics = best_candidate.get("metrics")
        best_metrics = dict(best_metrics) if isinstance(best_metrics, dict) else {}
        preserved_metrics = {
            key: metrics[key] for key in _FRONTIER_DERIVED_METRIC_KEYS if key in metrics
        }
        metrics = {**best_metrics, **preserved_metrics}
        entry["metrics"] = metrics
        entry.pop("design_dimensions", None)
        entry.pop("realized_dimensions", None)
        metrics.pop("design_dimensions", None)
        metrics.pop("realized_dimensions", None)
        if design_dimensions := _extract_design_dimensions(best_candidate):
            entry["design_dimensions"] = design_dimensions

        # Research metadata belongs to the selected immutable result, not to
        # the Frontier view.  Replace it atomically when source repair selects
        # a different result while retaining the previous result as a signal.
        for key in RESULT_RESEARCH_METADATA_KEYS:
            entry.pop(key, None)
            metrics.pop(key, None)
        for key in RESULT_RESEARCH_METADATA_KEYS:
            value = _raw_candidate_field(best_candidate, key)
            if value not in (None, ""):
                entry[key] = value
                metrics[key] = value

        # A frontier row is a derived view of exactly one immutable result.
        # When source repair selects a different summary, replace the complete
        # maturity tuple together rather than mixing the new path/score with
        # stale stage or ratio facts from the previously selected summary.
        evidence_metadata = _evidence_metadata_from_candidate(
            best_candidate,
            self.maturity_policy,
        )
        reported_stage = _raw_candidate_field(best_candidate, "evidence_stage")
        if reported_stage not in (None, ""):
            evidence_metadata["evidence_stage"] = reported_stage
        for key in _CANONICAL_SOURCE_FACT_KEYS:
            entry.pop(key, None)
            metrics.pop(key, None)
        for key in _CANONICAL_SOURCE_FACT_KEYS:
            value = _raw_candidate_field(best_candidate, key)
            if value is not None:
                entry[key] = value
                metrics[key] = value
        if preserved_parent_eligible is not None:
            entry["parent_eligible"] = preserved_parent_eligible
            metrics["parent_eligible"] = preserved_parent_eligible
        for key, value in evidence_metadata.items():
            if value not in (None, ""):
                entry[key] = value
                metrics[key] = value
        for key in (
            "promotion_eligible",
            "clean_promotion_eligible",
            "parent_eligible",
            "excluded_from_durable_frontier",
            "exclusion_reason",
        ):
            value = _raw_candidate_field(best_candidate, key)
            if value is not None:
                entry[key] = value
                metrics[key] = value
        promoted_lane_name = str(entry.get("promoted_for_lane") or entry.get("frontier_lane") or "")
        promoted_lane = next(
            (
                lane
                for lane in self.frontier_lanes
                if str(lane.get("name") or "") == promoted_lane_name
            ),
            None,
        )
        promotion_eligible = resolved_fact_bool(
            best_candidate,
            "promotion_eligible",
            "clean_promotion_eligible",
        )
        if promotion_eligible is False:
            entry["parent_eligible"] = False
            metrics["parent_eligible"] = False
            risk_exception = bool(
                promoted_lane is not None
                and promoted_lane.get("allow_risk_violating")
                and _any_boolish_candidate_field_true(
                    entry,
                    "risk_violating_frontier_candidate",
                )
            )
            if not risk_exception and not bool(
                promoted_lane is not None and promoted_lane.get("allow_non_promotable")
            ):
                entry["excluded_from_durable_frontier"] = True
                entry["exclusion_reason"] = "promotion_eligible_false"
                metrics["excluded_from_durable_frontier"] = True
                metrics["exclusion_reason"] = "promotion_eligible_false"
        elif (
            source_changed
            and promotion_eligible is True
            and _raw_candidate_field(best_candidate, "parent_eligible") is None
        ):
            lane_name = str(entry.get("frontier_lane") or entry.get("promoted_for_lane") or "")
            lane = next(
                (
                    configured
                    for configured in self.frontier_lanes
                    if str(configured.get("name") or "") == lane_name
                ),
                None,
            )
            parent_eligible = self._lane_allows_parents(lane) if lane is not None else True
            entry["parent_eligible"] = parent_eligible
            metrics["parent_eligible"] = parent_eligible

        if unscoped_same_path_rewrite:
            entry["parent_eligible"] = False
            entry["excluded_from_durable_frontier"] = True
            entry["exclusion_reason"] = "source_rewritten_without_generation"
            metrics["parent_eligible"] = False
            metrics["excluded_from_durable_frontier"] = True
            metrics["exclusion_reason"] = "source_rewritten_without_generation"

        changed = json.dumps(entry, sort_keys=True, default=str) != source_fact_before

        def assign(key: str, value: Any, *, mirror_metric: bool = True) -> None:
            nonlocal changed
            if value in (None, ""):
                return
            if entry.get(key) != value:
                entry[key] = value
                changed = True
            if mirror_metric and metrics.get(key) != value:
                metrics[key] = value
                changed = True

        best_digest = (
            best_artifact[1] if best_artifact is not None and best_artifact[0] == best_path else ""
        )

        original_path = (
            str(entry.get("selected_source_result_path") or "").strip().replace("\\", "/")
            or source_path
        )
        current_source = str(entry.get("source_result_path") or "").strip().replace("\\", "/")
        current_artifact = result_artifact_key(entry)
        replace_artifact = bool(best_digest and current_artifact != (best_path, best_digest))
        if original_path and original_path != best_path:
            assign("selected_source_result_path", original_path)
            assign("source_selection_warning", "better_canonical_result_source_available")
        elif entry.get("source_selection_warning"):
            entry.pop("source_selection_warning", None)
            metrics.pop("source_selection_warning", None)
            changed = True
        if replace_artifact or not current_source or current_source != best_path:
            _clear_result_artifact_coordinates(entry)
            assign("source_result_path", best_path)
        if best_digest:
            assign("source_result_sha256", best_digest)
        assign("canonical_source_result_path", best_path)
        assign("best_available_summary_path", best_path)
        best_keys = [str(key) for key in best.get("source_group_keys") or []]
        matching_key = next((key for key in source_keys if key in best_keys), "")
        canonical_token = _source_group_token(matching_key or (best_keys[0] if best_keys else ""))
        assign("canonical_variant_id", canonical_token)
        assign("source_selection_reason", "best_result_summary_by_protocol_maturity_metric")

        lane_metric_name = str(entry.get("lane_metric_name") or "").strip()
        requested_metric_name = str(
            entry.get("metric_name") or best.get("metric_name") or self.primary_metric
        )
        metric_name = requested_metric_name
        metric_value = self._source_metric_value(best_candidate, metric_name)
        if metric_value is None and lane_metric_name and lane_metric_name != metric_name:
            lane_fallback = self._source_metric_value(best_candidate, lane_metric_name)
            if lane_fallback is not None:
                metric_name = lane_metric_name
                metric_value = lane_fallback
                assign("metric_name", metric_name, mirror_metric=False)
                assign(
                    "metric_direction",
                    _normalize_metric_direction(
                        entry.get("lane_metric_direction"),
                        default=self.metric_direction,
                    ),
                    mirror_metric=False,
                )
        for key in (
            "metric_value",
            "canonical_metric_value",
            "selected_metric_value",
        ):
            entry.pop(key, None)
        metrics.pop(requested_metric_name, None)
        if metric_name != requested_metric_name:
            metrics.pop(metric_name, None)
        if metric_value is not None:
            assign("metric_value", metric_value, mirror_metric=False)
            if metrics.get(metric_name) != metric_value:
                metrics[metric_name] = metric_value
                changed = True
            assign("canonical_metric_value", metric_value)
        else:
            entry["excluded_from_durable_frontier"] = True
            entry["exclusion_reason"] = "canonical_source_missing_frontier_metric"
            metrics["excluded_from_durable_frontier"] = True
            metrics["exclusion_reason"] = "canonical_source_missing_frontier_metric"

        if lane_metric_name:
            for key in ("lane_metric_value", "selected_lane_metric_value"):
                entry.pop(key, None)
            metrics.pop(lane_metric_name, None)
            metrics.pop("lane_metric_value", None)
            lane_value = self._source_metric_value(best_candidate, lane_metric_name)
            if lane_value is not None:
                assign("lane_metric_value", lane_value, mirror_metric=False)
                metrics["lane_metric_value"] = lane_value
                metrics[lane_metric_name] = lane_value
                changed = True
            else:
                entry["excluded_from_durable_frontier"] = True
                entry["exclusion_reason"] = "canonical_source_missing_lane_metric"
                metrics["excluded_from_durable_frontier"] = True
                metrics["exclusion_reason"] = "canonical_source_missing_lane_metric"

        if promoted_lane is not None:
            source_candidate = dict(best_candidate)
            source_metrics = dict(best_metrics)
            source_lane = str(
                _raw_candidate_field(
                    best_candidate,
                    "source_frontier_lane",
                    "frontier_lane",
                    "promotion_lane",
                    "lane",
                )
                or entry.get("source_frontier_lane")
                or promoted_lane_name
            ).strip()
            if source_lane:
                source_candidate["frontier_lane"] = source_lane
                source_metrics["frontier_lane"] = source_lane
            source_candidate["metrics"] = source_metrics
            lane_values = _lane_values(
                source_candidate,
                self._lane_axes(promoted_lane),
                self._lane_optional_axes(promoted_lane),
            )
            if not _matches_lane_filters(source_candidate, promoted_lane) or lane_values is None:
                entry["excluded_from_durable_frontier"] = True
                entry["exclusion_reason"] = "canonical_source_fails_frontier_lane_contract"
                metrics["excluded_from_durable_frontier"] = True
                metrics["exclusion_reason"] = "canonical_source_fails_frontier_lane_contract"

        if entry_gen is not None and not _has_mature_durable_evidence(
            entry,
            self.maturity_policy,
        ):
            validation_entry = self._validation_candidate_entry(
                gen_id=int(entry_gen),
                finding=entry,
            )
            if validation_entry is not None:
                self._record_validation_candidates(
                    gen_id=int(entry_gen),
                    entries=[validation_entry],
                )
        return (
            retained_late_signal
            or changed
            or json.dumps(entry, sort_keys=True, default=str) != source_fact_before
        )

    @staticmethod
    def _result_source_family(record: dict[str, Any] | None) -> str:
        if not isinstance(record, dict):
            return ""
        source_path = str(record.get("source_result_path") or "").replace("\\", "/")
        if not source_path:
            return ""
        parent = Path(source_path).parent
        candidate = record.get("candidate")
        candidate = candidate if isinstance(candidate, dict) else {}
        stage = _norm_token(_raw_candidate_field(candidate, "evidence_stage")).replace("-", "_")
        parent_name = _norm_token(parent.name).replace("-", "_")
        if stage and parent_name == stage:
            parent = parent.parent
        return str(parent).replace("\\", "/")

    def _repair_manifest_canonical_result_sources(self) -> bool:
        by_source_path, sources_by_variant = self._canonical_result_source_index()
        if not sources_by_variant:
            return False
        changed = False
        generations = self._manifest.get("generations")
        if not isinstance(generations, dict):
            return False
        repaired_count = 0
        for gen_key, entries in generations.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if self._repair_manifest_entry_canonical_source(
                    entry,
                    by_source_path=by_source_path,
                    sources_by_variant=sources_by_variant,
                    generation_key=str(gen_key),
                ):
                    changed = True
                    repaired_count += 1
        if repaired_count:
            logger.info(
                "frontier: repaired canonical result source metadata on %d manifest entr%s",
                repaired_count,
                "y" if repaired_count == 1 else "ies",
            )
        return changed

    def _beats_risk_frontier_threshold(self, metric_value: float) -> bool:
        threshold = self.risk_violating_primary_threshold
        if threshold is None:
            return False
        if self.metric_direction == "minimize":
            return metric_value < threshold
        return metric_value > threshold

    @staticmethod
    def _hard_constraint_count(finding: dict[str, Any]) -> int:
        for container_name in ("metrics", "details"):
            container = finding.get(container_name)
            if not isinstance(container, dict):
                continue
            raw_count = container.get("n_hard_constraint_violations")
            if isinstance(raw_count, bool):
                continue
            if isinstance(raw_count, (int, float)):
                try:
                    return max(0, int(raw_count))
                except (TypeError, ValueError):
                    pass
            violations = container.get("hard_constraint_violations")
            if isinstance(violations, list):
                return len(violations)
        return 0

    def _risk_violating_reason(
        self,
        *,
        finding: dict[str, Any],
        tier: str | None,
        promotion_rejected: bool,
        metric_value: float,
    ) -> str | None:
        if not self.risk_violating_frontier_enabled:
            return None
        if not self._beats_risk_frontier_threshold(metric_value):
            return None

        hard_count = self._hard_constraint_count(finding)
        reasons: list[str] = []
        if promotion_rejected:
            reasons.append("promotion_eligible=false")
        if hard_count > 0:
            reasons.append(f"hard_constraint_violations={hard_count}")
        if not reasons:
            return None
        threshold = self.risk_violating_primary_threshold
        return (
            f"{self.primary_metric}={metric_value:.6g} beats repair threshold "
            f"{threshold:.6g}; risk issues: {', '.join(reasons)}"
        )

    def _load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                manifest = json.load(f)
            if isinstance(manifest, dict) and is_committed_runtime_fact_source(
                manifest,
                legacy_ok=True,
            ):
                return manifest
            logger.warning(
                "frontier: ignoring non-committed runtime manifest at %s",
                self.manifest_path,
            )
        return {
            "generations": {},
            "cumulative_top": [],
            "lane_frontiers": {},
            "validation_candidates": {
                "generations": {},
                "cumulative": [],
            },
            "frontier_lanes": self.frontier_lanes,
            "primary_metric": self.primary_metric,
            "metric_direction": self.metric_direction,
        }

    def _save_manifest(self):
        # Atomic write: tmp + rename. Without this, a peer calling the
        # ``mcp__frontier-tools__get_frontier`` handler mid-``promote()``
        # could ``open(manifest_path, "r")`` on a truncated or partial
        # JSON and raise JSONDecodeError. The handler has no retry.
        self._manifest = _sanitize_nonfinite_json(self._manifest)
        existing_semantics = self._manifest.get("artifact_semantics")
        created_at = (
            str(existing_semantics.get("created_at") or "")
            if isinstance(existing_semantics, dict)
            else ""
        )
        canonical_sources = [
            "results/*",
            "shared_findings/*",
            "shared_store.db",
        ]
        if is_committed_runtime_fact_file(self.base_dir.parent / "gems" / "gems_state.json"):
            canonical_sources.append("gems/gems_state.json")
        self._manifest["artifact_semantics"] = artifact_semantics(
            role=CANONICAL_STATE,
            stage="frontier_manifest",
            actor="research_loop:frontier_store",
            canonical_sources=canonical_sources,
            runtime_fact_source=True,
            notes=(
                "Canonical current frontier/incubator/validation-candidate state. "
                "Derived leaderboards, PI packs, and prompt snapshots must not "
                "override this manifest."
            ),
            created_at=created_at or None,
        )
        try:
            from praxist.plugins.workflow_stages.research_loop.backend.tools.atomic_io import (
                atomic_write_json,
            )

            atomic_write_json(self.manifest_path, self._manifest)
        except Exception as e:
            # Fallback to direct write — better than losing the manifest.
            # Loud warning: under fallback, concurrent readers may see
            # truncated JSON and raise JSONDecodeError (m3 from review
            # round 1).
            logger.error(
                "atomic_write_json unavailable (%s); using non-atomic "
                "fallback for %s. Concurrent readers may race on a "
                "truncated manifest.",
                e,
                self.manifest_path,
            )
            with open(self.manifest_path, "w") as f:
                json.dump(self._manifest, f, indent=2, default=str)

    def _lane_axes(self, lane: dict[str, Any]) -> list[tuple[str, str]]:
        return _normalized_lane_axes(lane, self.primary_metric, self.metric_direction)

    def _lane_optional_axes(self, lane: dict[str, Any]) -> list[tuple[str, str]]:
        axes = lane.get("optional_axes") or []
        normalized: list[tuple[str, str]] = []
        for entry in axes:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                name, direction = str(entry[0]), str(entry[1])
            elif isinstance(entry, dict) and (entry.get("name") or entry.get("metric")):
                name = str(entry.get("name") or entry.get("metric"))
                direction = str(entry.get("direction", "maximize"))
            else:
                continue
            if direction not in ("maximize", "minimize"):
                continue
            normalized.append((name, direction))
        return normalized

    def _validation_signal_metrics(self, finding: dict[str, Any]) -> list[dict[str, Any]]:
        """Return task-relevant numeric signals for a preliminary candidate.

        These signals are not promotion evidence. They are only used to retain
        promising scout/partial rows as follow-up targets for PI synthesis and
        next-generation validation. Durable frontier and Gems admission continue
        to use the stricter mature-evidence path.
        """

        signals: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        def add_signal(
            *,
            name: str,
            direction: str,
            value: float | None,
            source: str,
            source_priority: int,
            lane_name: str = "",
            lane_filter_match: bool = False,
        ) -> None:
            if value is None or direction not in {"maximize", "minimize"}:
                return
            key = (source, lane_name, name)
            if key in seen:
                return
            seen.add(key)
            signals.append(
                {
                    "metric_name": name,
                    "metric_value": value,
                    "metric_direction": direction,
                    "signal_source": source,
                    "signal_source_priority": source_priority,
                    "frontier_lane": lane_name,
                    "lane_filter_match": lane_filter_match,
                }
            )

        primary_value = _canonical_signal_metric_value(finding, self.primary_metric)
        add_signal(
            name=self.primary_metric,
            direction=self.metric_direction,
            value=primary_value,
            source="primary_metric",
            source_priority=3,
        )

        for lane in self.frontier_lanes:
            lane_name = str(lane.get("name") or "")
            lane_filter_match = _matches_lane_filters(finding, lane)
            for metric_name, direction in [
                *self._lane_axes(lane),
                *self._lane_optional_axes(lane),
            ]:
                add_signal(
                    name=metric_name,
                    direction=direction,
                    value=_canonical_signal_metric_value(finding, metric_name),
                    source="frontier_lane_axis",
                    source_priority=4 if lane_filter_match else 2,
                    lane_name=lane_name,
                    lane_filter_match=lane_filter_match,
                )

        for anchor_name, anchor_dir in self.anchor_metrics:
            add_signal(
                name=anchor_name,
                direction=anchor_dir,
                value=_canonical_signal_metric_value(finding, anchor_name),
                source="secondary_anchor",
                source_priority=1,
            )

        for metric_name, metric_dir in self.validation_signal_metrics:
            add_signal(
                name=metric_name,
                direction=metric_dir,
                value=_canonical_signal_metric_value(finding, metric_name),
                source="task_configured_validation_signal",
                source_priority=2,
            )

        return signals

    @staticmethod
    def _validation_candidate_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_anchor_metrics(raw: list[tuple[str, str]] | list[Any]) -> list[tuple[str, str]]:
        normalized: list[tuple[str, str]] = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                name, direction = item
            elif isinstance(item, dict) and item.get("name"):
                name = item.get("name")
                direction = item.get("direction", "maximize")
            elif isinstance(item, str):
                name = item
                direction = "maximize"
            else:
                continue
            metric_name = str(name).strip()
            if not metric_name:
                continue
            normalized.append((metric_name, _normalize_metric_direction(direction)))
        return normalized

    @staticmethod
    def _validation_candidate_sort_key(entry: dict[str, Any]) -> tuple[Any, ...]:
        try:
            metric_value = float(entry.get("metric_value"))
        except (TypeError, ValueError):
            directional_value = float("-inf")
        else:
            import math

            if not math.isfinite(metric_value):
                directional_value = float("-inf")
            else:
                direction = str(entry.get("metric_direction") or "maximize")
                if direction not in {"maximize", "minimize"}:
                    direction = "maximize"
                directional_value = metric_value if direction == "maximize" else -metric_value
        try:
            generation = int(entry.get("generation_id", -1))
        except (TypeError, ValueError):
            generation = -1
        return (
            FrontierStore._validation_candidate_int(entry.get("signal_source_priority"), 0),
            FrontierStore._validation_candidate_int(entry.get("evidence_maturity_rank"), 0),
            directional_value,
            generation,
            str(entry.get("variant_name") or entry.get("finding_id") or ""),
        )

    @staticmethod
    def _validation_candidate_display_name(finding: dict[str, Any]) -> str:
        variant = finding.get("variant_name")
        if variant not in (None, ""):
            return str(variant)
        notes = finding.get("notes")
        return str(notes or "")[:80]

    def _validation_candidate_entry(
        self,
        *,
        gen_id: int,
        finding: dict[str, Any],
    ) -> dict[str, Any] | None:
        signals = self._validation_signal_metrics(finding)
        if signals:
            signal = max(
                signals,
                key=lambda item: self._validation_candidate_sort_key(
                    {
                        **item,
                        "generation_id": gen_id,
                        "evidence_maturity_rank": _evidence_maturity_rank(finding),
                    }
                ),
            )
        else:
            artifact = result_artifact_key(finding)
            if artifact is None or not artifact[0]:
                return None
            signal = {
                "metric_name": "",
                "metric_value": None,
                "metric_direction": self.metric_direction,
                "signal_source": "artifact_state",
                "signal_source_priority": 0,
                "frontier_lane": "",
                "lane_filter_match": False,
            }
        metrics = finding.get("metrics") if isinstance(finding.get("metrics"), dict) else {}
        compact_metrics = _drop_legacy_protocol_alias(metrics)
        compact_metrics.pop("design_dimensions", None)
        compact_metrics.pop("realized_dimensions", None)

        compact_identity_containers = {
            container_name: compact
            for container_name in ("details", "extra", "current_aggregate")
            if (compact := _compact_result_identity_container(finding.get(container_name)))
        }
        compact_root_identity = {
            key: finding[key] for key in _RESULT_PRODUCER_IDENTITY_KEYS if key in finding
        }
        metadata = _evidence_metadata_from_candidate(finding, self.maturity_policy)
        for key, value in metadata.items():
            if key == "frontier_entity_key":
                compact_metrics[key] = value
            else:
                compact_metrics.setdefault(key, value)
        for key, value in _research_metadata_from_finding(finding).items():
            compact_metrics.setdefault(key, value)
        compact_metrics = _sanitize_nonfinite_json(compact_metrics)
        submitted_lane = _candidate_lane(finding)
        matched_lanes = sorted(
            {
                str(item.get("frontier_lane") or "")
                for item in signals
                if item.get("frontier_lane") and item.get("lane_filter_match") is True
            }
        )
        signal_axis_lanes = sorted(
            {str(item.get("frontier_lane") or "") for item in signals if item.get("frontier_lane")}
        )
        retained_validation_lanes = sorted(
            str(lane.get("name") or "")
            for lane in self.frontier_lanes
            if lane.get("name")
            and lane.get("allow_lower_tier")
            and _matches_lane_filters(finding, lane)
        )
        entry: dict[str, Any] = {
            "generation_id": gen_id,
            "finding_id": finding.get("id") or finding.get("finding_id") or "",
            "variant_name": self._validation_candidate_display_name(finding),
            "metric_name": signal["metric_name"],
            "metric_value": signal["metric_value"],
            "metric_direction": signal["metric_direction"],
            "signal_source": signal["signal_source"],
            "signal_source_priority": signal["signal_source_priority"],
            "submitted_frontier_lane": submitted_lane,
            "matched_frontier_lanes": matched_lanes,
            "retained_validation_lanes": retained_validation_lanes,
            "signal_axis_lanes": signal_axis_lanes,
            "excluded_from_durable_frontier": True,
            "exclusion_reason": "preliminary_or_incomplete_evidence",
            "recommended_next_step": "complete_scored_validation_before_frontier_or_gems",
            "metrics": compact_metrics,
            **compact_root_identity,
            **compact_identity_containers,
            "captured_at": datetime.now().isoformat(),
        }
        design_dimensions = _extract_design_dimensions(finding)
        if design_dimensions:
            entry["design_dimensions"] = design_dimensions
        for key, value in metadata.items():
            entry[key] = value
        for key in ("promotion_eligible", "clean_promotion_eligible"):
            value = _raw_candidate_field(finding, key)
            if value is not None:
                entry[key] = value
        for key, value in _research_metadata_from_finding(finding).items():
            entry.setdefault(key, value)
        entry["identity_aliases"] = sorted(
            _candidate_identity_aliases(finding) | _candidate_identity_aliases(entry)
        )
        return _sanitize_nonfinite_json(entry)

    def _record_validation_candidates(
        self,
        *,
        gen_id: int,
        entries: list[dict[str, Any]],
    ) -> None:
        validation = self._manifest.setdefault("validation_candidates", {})
        if not isinstance(validation, dict):
            validation = {}
            self._manifest["validation_candidates"] = validation
        generations = validation.setdefault("generations", {})
        if not isinstance(generations, dict):
            generations = {}
            validation["generations"] = generations
        alias_generations = validation.setdefault("validator_identity_aliases_by_generation", {})
        if not isinstance(alias_generations, dict):
            alias_generations = {}
            validation["validator_identity_aliases_by_generation"] = alias_generations

        def merge_aliases(target: dict[str, Any], source: dict[str, Any]) -> None:
            aliases = _candidate_identity_aliases(target) | _candidate_identity_aliases(source)
            if aliases:
                target["identity_aliases"] = sorted(aliases)
                metrics = target.get("metrics")
                if isinstance(metrics, dict):
                    metrics["identity_aliases"] = sorted(aliases)

        def validation_record_key(entry: dict[str, Any]) -> str:
            metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
            artifact = result_artifact_key(entry)
            if artifact is not None and all(artifact):
                late_outcome = bool(
                    _any_boolish_candidate_field_true(
                        entry,
                        "late_after_generation_boundary",
                    )
                    or "late_after_generation_boundary"
                    in _norm_token(_raw_candidate_field(entry, "artifact_signal_status"))
                )
                lane_outcome = {
                    "lane": _candidate_lane(entry),
                    "matched": sorted(_norm_token_set(entry.get("matched_frontier_lanes"))),
                    "retained": sorted(_norm_token_set(entry.get("retained_validation_lanes"))),
                }
                immutable_parts = [
                    str(entry.get("generation_id") or metrics.get("generation_id") or gen_id),
                    "result_snapshot",
                    artifact[0],
                    artifact[1],
                    str(entry.get("metric_name") or ""),
                    str(entry.get("metric_value")) if entry.get("metric_value") is not None else "",
                    str(entry.get("metric_direction") or ""),
                    _norm_token(_raw_candidate_field(entry, "exclusion_reason")),
                    "protocol_failed" if _candidate_protocol_integrity_failed(entry) else "",
                    "late" if late_outcome else "",
                    json.dumps(lane_outcome, sort_keys=True, separators=(",", ":")),
                ]
                return "\x1f".join(part.strip().lower() for part in immutable_parts)
            parts = [
                str(entry.get("generation_id") or metrics.get("generation_id") or gen_id),
                str(entry.get("finding_id") or ""),
                str(entry.get("variant_name") or ""),
                str(entry.get("metric_name") or ""),
                str(entry.get("signal_source") or ""),
                str(entry.get("source_path") or metrics.get("source_path") or ""),
                str(entry.get("result_path") or metrics.get("result_path") or ""),
                str(entry.get("source_result_path") or metrics.get("source_result_path") or ""),
                str(entry.get("result_artifact_path") or metrics.get("result_artifact_path") or ""),
            ]
            key = "\x1f".join(part.strip().lower() for part in parts if part not in ("", "None"))
            return key or json.dumps(entry, sort_keys=True, default=str)

        def validation_record_is_clean_enough(entry: dict[str, Any]) -> bool:
            if entry.get("signal_source") == "artifact_state" and result_artifact_key(entry):
                return True
            if _candidate_has_bad_runtime_status(
                entry
            ) and not _candidate_is_scored_nonpromotable_validation_signal(entry):
                return False
            if _any_boolish_candidate_field_true(entry, "summary_only", "is_summary_only"):
                return False
            return not _candidate_status_has_any(entry, "summary_only")

        def add_record(
            records_by_key: dict[str, dict[str, Any]],
            entry: dict[str, Any],
        ) -> None:
            if not validation_record_is_clean_enough(entry):
                return
            entity_key = _candidate_entity_key(entry)
            if entity_key:
                entry["frontier_entity_key"] = entity_key
                metrics = entry.get("metrics")
                if isinstance(metrics, dict):
                    metrics["frontier_entity_key"] = entity_key
            entry["identity_aliases"] = sorted(_candidate_identity_aliases(entry))
            record_key = validation_record_key(entry)
            incumbent = records_by_key.get(record_key)
            if incumbent is None or self._validation_candidate_sort_key(
                entry
            ) > self._validation_candidate_sort_key(incumbent):
                if incumbent is not None:
                    merge_aliases(entry, incumbent)
                records_by_key[record_key] = entry
            elif incumbent is not None:
                merge_aliases(incumbent, entry)

        records_for_generation: dict[str, dict[str, Any]] = {}
        raw_existing_generation = generations.get(str(gen_id))
        if isinstance(raw_existing_generation, list):
            for entry in raw_existing_generation:
                if isinstance(entry, dict):
                    add_record(records_for_generation, entry)
        for entry in entries:
            entity_key = _candidate_entity_key(entry)
            if entity_key:
                entry["frontier_entity_key"] = entity_key
                metrics = entry.get("metrics")
                if isinstance(metrics, dict):
                    metrics["frontier_entity_key"] = entity_key
            add_record(records_for_generation, entry)
        generation_aliases = {
            str(alias).strip()
            for raw in [
                *(raw_existing_generation if isinstance(raw_existing_generation, list) else []),
                *entries,
            ]
            if isinstance(raw, dict)
            for alias in _candidate_identity_aliases(raw)
            if alias not in (None, "")
        }
        existing_aliases = alias_generations.get(str(gen_id))
        if isinstance(existing_aliases, list):
            generation_aliases.update(str(alias).strip() for alias in existing_aliases if alias)
        if generation_aliases:
            alias_generations[str(gen_id)] = sorted(generation_aliases)
        if records_for_generation:
            generations[str(gen_id)] = sorted(
                records_for_generation.values(),
                key=self._validation_candidate_sort_key,
                reverse=True,
            )

        cumulative_by_record: dict[str, dict[str, Any]] = {}
        existing_cumulative = validation.get("cumulative")
        if isinstance(existing_cumulative, list):
            for entry in existing_cumulative:
                if isinstance(entry, dict):
                    add_record(cumulative_by_record, entry)
        for raw_entries in generations.values():
            if not isinstance(raw_entries, list):
                continue
            for entry in raw_entries:
                if not isinstance(entry, dict):
                    continue
                add_record(cumulative_by_record, entry)
        validation["cumulative"] = sorted(
            cumulative_by_record.values(),
            key=self._validation_candidate_sort_key,
            reverse=True,
        )

    def _entry_generation_id(self, entry: dict[str, Any], default: int = -1) -> int:
        value = explicit_entry_generation_id(entry)
        return value if value is not None else default

    def _durable_frontier_entity_keys(self) -> set[str]:
        keys: set[str] = set()
        for entry in self._durable_frontier_entries():
            keys.add(_candidate_entity_key(entry))
        return {key for key in keys if key}

    def _durable_frontier_entries(self) -> list[dict[str, Any]]:
        entries_out: list[dict[str, Any]] = []
        for entries in self._manifest.get("generations", {}).values():
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict) and _is_committed_frontier_entry(
                        entry, self.maturity_policy
                    ):
                        entries_out.append(entry)
        lane_frontiers = self._manifest.get("lane_frontiers")
        if isinstance(lane_frontiers, dict):
            for entries in lane_frontiers.values():
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict) and _is_committed_frontier_entry(
                            entry, self.maturity_policy
                        ):
                            entries_out.append(entry)
        for entry in self._manifest.get("cumulative_top", []) or []:
            if isinstance(entry, dict) and _is_committed_frontier_entry(
                entry, self.maturity_policy
            ):
                entries_out.append(entry)
        return entries_out

    def _durable_frontier_identity_aliases(self) -> set[str]:
        aliases: set[str] = set()
        for entry in self._durable_frontier_entries():
            aliases.update(_candidate_retirement_aliases(entry))
            key = _candidate_entity_key(entry)
            if key:
                aliases.add(key)
        return {alias for alias in aliases if alias}

    def _retire_validation_candidates_for_durable_entities(self) -> bool:
        durable_entries = self._durable_frontier_entries()
        durable_snapshot_records = [
            (result_snapshot_key(entry), entry.get("variant_name")) for entry in durable_entries
        ]
        if not any(
            snapshot is not None and all(snapshot)
            for snapshot, _fallback in durable_snapshot_records
        ):
            return False
        validation = self._manifest.get("validation_candidates")
        if not isinstance(validation, dict):
            return False
        changed = False
        stale_aliases: set[str] = set()
        retained_aliases: set[str] = set()

        def keep(entry: Any) -> bool:
            if not isinstance(entry, dict):
                return True
            snapshot = result_snapshot_key(entry)
            if snapshot is None or not all(snapshot):
                return True
            resolved = resolve_result_snapshot_producers(
                [*durable_snapshot_records, (snapshot, entry.get("variant_name"))]
            )
            validation_snapshot = resolved[-1]
            return not any(
                same_result_snapshot(validation_snapshot, durable_snapshot)
                for durable_snapshot in resolved[:-1]
            )

        generations = validation.get("generations")
        if isinstance(generations, dict):
            for gen_key, entries in list(generations.items()):
                if not isinstance(entries, list):
                    continue
                filtered = []
                removed: list[dict[str, Any]] = []
                for entry in entries:
                    if keep(entry):
                        filtered.append(entry)
                        if isinstance(entry, dict):
                            retained_aliases.update(_candidate_identity_aliases(entry))
                    elif isinstance(entry, dict):
                        removed.append(entry)
                if len(filtered) != len(entries):
                    generations[gen_key] = filtered
                    for entry in removed:
                        stale_aliases.update(_candidate_identity_aliases(entry))
                    changed = True
        cumulative = validation.get("cumulative")
        if isinstance(cumulative, list):
            filtered = []
            removed = []
            for entry in cumulative:
                if keep(entry):
                    filtered.append(entry)
                    if isinstance(entry, dict):
                        retained_aliases.update(_candidate_identity_aliases(entry))
                elif isinstance(entry, dict):
                    removed.append(entry)
            if len(filtered) != len(cumulative):
                validation["cumulative"] = filtered
                for entry in removed:
                    stale_aliases.update(_candidate_identity_aliases(entry))
                changed = True
        stale_aliases -= retained_aliases
        if stale_aliases:
            alias_generations = validation.get("validator_identity_aliases_by_generation")
            if isinstance(alias_generations, dict):
                for gen_key, values in list(alias_generations.items()):
                    if not isinstance(values, list):
                        continue
                    filtered_aliases = [
                        str(value).strip()
                        for value in values
                        if value not in (None, "") and str(value).strip() not in stale_aliases
                    ]
                    if len(filtered_aliases) != len(values):
                        if filtered_aliases:
                            alias_generations[gen_key] = sorted(set(filtered_aliases))
                        else:
                            alias_generations.pop(gen_key, None)
                        changed = True
            alias_values = validation.get("validator_identity_aliases")
            if isinstance(alias_values, list):
                filtered_aliases = [
                    str(value).strip()
                    for value in alias_values
                    if value not in (None, "") and str(value).strip() not in stale_aliases
                ]
                if len(filtered_aliases) != len(alias_values):
                    validation["validator_identity_aliases"] = sorted(set(filtered_aliases))
                    changed = True
        return changed

    def _prune_durable_frontier_entries(self, *, migrate: bool = False) -> bool:
        changed = False
        migrated_by_gen: dict[int, list[dict[str, Any]]] = {}

        def migrate_entry(entry: dict[str, Any], gen_hint: int = -1) -> None:
            if not migrate:
                return
            ratio_decision = evidence_maturity_snapshot(entry, self.maturity_policy).get(
                "mature_enough"
            )
            ratio_immature = ratio_decision is False
            required_ratio_missing = ratio_decision is None and bool(
                self.maturity_policy.get("require_ratio_gate")
            )
            if (
                not ratio_immature
                and not required_ratio_missing
                and not _is_retainable_validation_candidate(entry)
                and _normalized_evidence_stage(entry) != "unknown"
            ):
                return
            gen_id = self._entry_generation_id(entry, default=gen_hint)
            if gen_id < 0:
                return
            validation_entry = self._validation_candidate_entry(gen_id=gen_id, finding=entry)
            if validation_entry is not None:
                migrated_by_gen.setdefault(gen_id, []).append(validation_entry)

        generations = self._manifest.get("generations")
        if isinstance(generations, dict):
            for gen_key, entries in list(generations.items()):
                if not isinstance(entries, list):
                    continue
                try:
                    gen_hint = int(gen_key)
                except (TypeError, ValueError):
                    gen_hint = -1
                kept: list[dict[str, Any]] = []
                for entry in entries:
                    if not isinstance(entry, dict):
                        changed = True
                        continue
                    if _is_committed_frontier_entry(entry, self.maturity_policy):
                        kept.append(entry)
                    else:
                        migrate_entry(entry, gen_hint=gen_hint)
                        changed = True
                if len(kept) != len(entries):
                    generations[gen_key] = kept

        lane_frontiers = self._manifest.get("lane_frontiers")
        if isinstance(lane_frontiers, dict):
            for lane_name, entries in list(lane_frontiers.items()):
                if not isinstance(entries, list):
                    continue
                kept = []
                for entry in entries:
                    if not isinstance(entry, dict):
                        changed = True
                        continue
                    if _is_committed_frontier_entry(entry, self.maturity_policy):
                        kept.append(entry)
                    else:
                        migrate_entry(entry, gen_hint=0)
                        changed = True
                if len(kept) != len(entries):
                    lane_frontiers[lane_name] = kept

        cumulative = self._manifest.get("cumulative_top")
        if isinstance(cumulative, list):
            kept = []
            for entry in cumulative:
                if not isinstance(entry, dict):
                    changed = True
                    continue
                if _is_committed_frontier_entry(entry, self.maturity_policy):
                    kept.append(entry)
                else:
                    migrate_entry(entry, gen_hint=0)
                    changed = True
            if len(kept) != len(cumulative):
                self._manifest["cumulative_top"] = kept

        for gen_id, entries in migrated_by_gen.items():
            self._record_validation_candidates(gen_id=gen_id, entries=entries)
        if self._retire_validation_candidates_for_durable_entities():
            changed = True
        return changed

    @staticmethod
    def _lane_sort_key(
        finding: dict[str, Any],
        values: dict[str, float],
        axes: list[tuple[str, str]],
        optional_axes: list[tuple[str, str]] | None = None,
    ) -> tuple[Any, ...]:
        directional_values = [_evidence_maturity_rank(finding)]
        for name, direction in [*axes, *(optional_axes or [])]:
            value = values.get(name)
            if value is None:
                directional_values.append(float("-inf"))
            else:
                directional_values.append(value if direction == "maximize" else -value)
        return (*directional_values, str(finding.get("variant_name") or finding.get("id") or ""))

    def _select_lane_picks(
        self,
        *,
        lane: dict[str, Any],
        candidates: list[dict[str, Any]],
        selected_ids: set[str],
        selected_capacity_identities: set[tuple[str, str, str]],
        exclude_generation_id: int | None = None,
    ) -> list[dict[str, Any]]:
        import copy as _copy

        k = int(lane.get("k", 1) or 0)
        if k <= 0:
            return []
        axes = self._lane_axes(lane)
        optional_axes = self._lane_optional_axes(lane)
        lane_name = str(lane.get("name") or "")
        require_new_high = bool(lane.get("admit_new_high"))
        existing_lane_points = (
            self._existing_lane_value_points(
                lane_name,
                axes,
            )
            if require_new_high
            else []
        )
        scored: list[tuple[dict[str, Any], dict[str, float]]] = []
        for finding in candidates:
            f_id = str(finding.get("id", ""))
            capacity_identity = _lane_capacity_identity(finding)
            has_exact_artifact = capacity_identity[0] == "result_artifact"
            if f_id and f_id in selected_ids and not has_exact_artifact:
                continue
            if capacity_identity in selected_capacity_identities:
                continue
            if finding.get("_risk_violating_frontier_candidate") and not lane.get(
                "allow_risk_violating", False
            ):
                continue
            if finding.get("_lane_non_promotable_candidate") and not lane.get(
                "allow_non_promotable", False
            ):
                continue
            if finding.get("_lane_missing_tier_candidate") and not lane.get(
                "allow_missing_tier", False
            ):
                continue
            if (
                finding.get("_risk_violating_frontier_candidate")
                or finding.get("_lane_non_promotable_candidate")
                or finding.get("_lane_missing_tier_candidate")
            ) and not _has_mature_durable_evidence(finding, self.maturity_policy):
                continue
            if not _matches_lane_filters(finding, lane):
                continue
            values = _lane_values(finding, axes, optional_axes)
            if values is None:
                continue
            candidate_capacity_identity = _lane_capacity_identity(finding)
            if existing_lane_points and any(
                not (
                    exclude_generation_id is not None
                    and existing_generation_id == exclude_generation_id
                    and (
                        existing_capacity_identity == candidate_capacity_identity
                        or (f_id and existing_finding_id == f_id)
                    )
                )
                and (
                    _pareto_dominates(existing_values, values, axes)
                    or all(existing_values[name] == values[name] for name, _direction in axes)
                )
                for (
                    existing_generation_id,
                    existing_capacity_identity,
                    existing_finding_id,
                    existing_values,
                ) in existing_lane_points
            ):
                continue
            scored.append((finding, values))
        if not scored:
            return []
        # Deduplicate before Pareto analysis so aliases of one immutable result
        # artifact do not appear as independent points. Rows without complete
        # coordinates retain the prior semantic-entity fallback.
        best_by_capacity: dict[tuple[str, str, str], tuple[dict[str, Any], dict[str, float]]] = {}
        for finding, values in scored:
            capacity_identity = _lane_capacity_identity(finding)
            incumbent = best_by_capacity.get(capacity_identity)
            if incumbent is None or self._lane_sort_key(
                finding, values, axes, optional_axes
            ) > self._lane_sort_key(incumbent[0], incumbent[1], axes, optional_axes):
                best_by_capacity[capacity_identity] = (finding, values)
        if len(best_by_capacity) < len(scored):
            logger.info(
                "frontier lane '%s': deduped %d duplicate evidence row(s) before Pareto selection",
                lane.get("name"),
                len(scored) - len(best_by_capacity),
            )
        scored = list(best_by_capacity.values())
        if require_new_high:
            best_by_point: dict[tuple[float, ...], tuple[dict[str, Any], dict[str, float]]] = {}
            for finding, values in scored:
                point = tuple(values[name] for name, _direction in axes)
                incumbent = best_by_point.get(point)
                if incumbent is None or self._lane_sort_key(
                    finding, values, axes, optional_axes
                ) > self._lane_sort_key(incumbent[0], incumbent[1], axes, optional_axes):
                    best_by_point[point] = (finding, values)
            scored = list(best_by_point.values())

        pareto: list[tuple[dict[str, Any], dict[str, float]]] = []
        dominated: list[tuple[dict[str, Any], dict[str, float]]] = []
        for idx, item in enumerate(scored):
            _finding, values = item
            maturity = _evidence_maturity_rank(_finding)
            is_dominated = any(
                other_idx != idx
                and _evidence_maturity_rank(_other) >= maturity
                and _pareto_dominates(other_values, values, axes)
                for other_idx, (_other, other_values) in enumerate(scored)
            )
            (dominated if is_dominated else pareto).append(item)
        pareto.sort(
            key=lambda item: self._lane_sort_key(item[0], item[1], axes, optional_axes),
            reverse=True,
        )
        dominated.sort(
            key=lambda item: self._lane_sort_key(item[0], item[1], axes, optional_axes),
            reverse=True,
        )

        picks: list[dict[str, Any]] = []
        pareto_finding_ids = {id(finding) for finding, _values in pareto}
        for finding, values in [*pareto, *dominated]:
            if len(picks) >= k and (not require_new_high or id(finding) not in pareto_finding_ids):
                break
            capacity_identity = _lane_capacity_identity(finding)
            f_id = str(finding.get("id", ""))
            has_exact_artifact = capacity_identity[0] == "result_artifact"
            if f_id and f_id in selected_ids and not has_exact_artifact:
                continue
            if capacity_identity in selected_capacity_identities:
                continue
            picked = _copy.copy(finding)
            existing = picked.get("metrics")
            picked["metrics"] = dict(existing) if isinstance(existing, dict) else {}
            source_lane = _candidate_lane(finding)
            lane_metric_name, lane_metric_direction = axes[0]
            picked["_promoted_for_lane"] = lane_name
            picked["_lane_metric_name"] = lane_metric_name
            picked["_lane_metric_value"] = values[lane_metric_name]
            picked["_lane_metric_direction"] = lane_metric_direction
            if source_lane and source_lane != lane_name:
                picked["_source_frontier_lane"] = source_lane
                picked["metrics"]["source_frontier_lane"] = source_lane
            picked["metrics"]["frontier_lane"] = lane_name
            picked["metrics"]["lane_metric_name"] = lane_metric_name
            picked["metrics"]["lane_metric_value"] = values[lane_metric_name]
            picked["metrics"]["lane_metric_direction"] = lane_metric_direction
            research_metadata = _research_metadata_from_finding(picked)
            for key, value in _evidence_metadata_from_candidate(
                finding,
                self.maturity_policy,
            ).items():
                if key in RESULT_RESEARCH_METADATA_KEYS and key in research_metadata:
                    continue
                if key == "frontier_entity_key":
                    picked[key] = value
                    picked["metrics"][key] = value
                else:
                    picked[key] = value
                    picked["metrics"][key] = value
            picks.append(picked)
            if f_id and not has_exact_artifact:
                selected_ids.add(f_id)
            selected_capacity_identities.add(capacity_identity)
        if picks:
            logger.info(
                "frontier lane '%s': selected %d/%d candidate(s)",
                lane.get("name"),
                len(picks),
                k,
            )
        return picks

    def _existing_lane_value_points(
        self,
        lane_name: str,
        axes: list[tuple[str, str]],
    ) -> list[tuple[int, tuple[str, str, str], str, dict[str, float]]]:
        lane_frontiers = self._manifest.get("lane_frontiers")
        if not isinstance(lane_frontiers, dict) or not lane_name:
            return []
        points: list[tuple[int, tuple[str, str, str], str, dict[str, float]]] = []
        entries = lane_frontiers.get(lane_name)
        if not isinstance(entries, list):
            return []
        for entry in entries:
            if not isinstance(entry, dict) or not _is_committed_frontier_entry(
                entry, self.maturity_policy
            ):
                continue
            values = _lane_values(entry, axes)
            if values is not None:
                try:
                    entry_generation_id = int(entry.get("generation_id", -1))
                except (TypeError, ValueError):
                    entry_generation_id = -1
                points.append(
                    (
                        entry_generation_id,
                        _lane_capacity_identity(entry),
                        str(entry.get("finding_id") or entry.get("id") or ""),
                        values,
                    )
                )
        return points

    def _select_lane_frontier(
        self,
        candidates: list[dict[str, Any]],
        *,
        exclude_generation_id: int | None = None,
    ) -> list[dict[str, Any]]:
        selected_ids: set[str] = set()
        selected_capacity_identities: set[tuple[str, str, str]] = set()
        picks: list[dict[str, Any]] = []
        for lane in self.frontier_lanes:
            lane_picks = self._select_lane_picks(
                lane=lane,
                candidates=candidates,
                selected_ids=selected_ids,
                selected_capacity_identities=selected_capacity_identities,
                exclude_generation_id=exclude_generation_id,
            )
            picks.extend(lane_picks)
        return picks

    def promote(self, gen_id: int, findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Promote top-K findings from a generation to the frontier.

        Args:
            gen_id: Generation index
            findings: All findings from this generation

        Returns:
            List of promoted findings

        Note on tolerance: the original filter required
        ``finding_type == "result"`` AND ``metrics[primary_metric]`` present.
        That rejected nearly every agent-written file in the 2026-04-17
        run because peers used freeform titles like "insight" / "hypothesis"
        with metrics nested inside ``details`` rather than the flat
        ``metrics`` dict.

        The ingest layer (findings_ingest.py) now normalizes most of this
        at write time, but we keep the filter here also permissive:
          - Accept finding_type ∈ {"result", "insight"} (both can carry
            a measurement; "insight" is used by peers reporting "X is
            better than Y at 200ep" style findings).
          - Accept the primary metric nested inside ``details`` or any
            level down, as a last-mile safety net.
        """

        def _get_metric(f: dict[str, Any]) -> float | None:
            direct_value = _metric_value(f, self.primary_metric)
            if direct_value is not None:
                return direct_value
            # R9-4 fix: strict canonical mode — only walk into preferred
            # parents (metrics, final_metrics, aggregated). Don't fall
            # back to generic DFS, which previously could pick up a stray
            # nested per-seed task metric and
            # promote a per-seed (often best-seed) value as if it were
            # the aggregated mean.
            return _walk_for_metric(
                f,
                self.primary_metric,
                _strict_canonical=True,
            )

        acceptable_types = ("result", "insight")
        if self.frontier_lanes:
            # Lane-aware tasks can intentionally keep lower-tier/non-clean
            # scored evidence in durable-candidate/control lanes. Preserve that
            # behavior without changing legacy single-metric tasks, where
            # `intermediate_result` was historically not promotable.
            acceptable_types = (*acceptable_types, "intermediate_result")
        # M5 fix (review round 2): work on shallow copies so we never
        # mutate caller-owned finding dicts. Status writer + graph
        # maintainer may concurrently read the same finding objects.
        import copy

        candidates: list[dict[str, Any]] = []
        validation_candidates: list[dict[str, Any]] = []
        rejection_counts: dict[str, int] = {}
        rejection_samples: dict[str, list[str]] = {}

        def record_promotion_rejection(reason: str, finding: dict[str, Any]) -> None:
            reason = str(reason or "unknown").strip() or "unknown"
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            samples = rejection_samples.setdefault(reason, [])
            if len(samples) >= 3:
                return
            sample_id = str(
                finding.get("finding_id") or finding.get("id") or finding.get("variant_name") or ""
            ).strip()
            if sample_id:
                samples.append(sample_id)

        def flush_promotion_rejections() -> None:
            root = self._manifest.setdefault("promotion_rejections", {})
            if not isinstance(root, dict):
                root = {}
                self._manifest["promotion_rejections"] = root
            if rejection_counts:
                root[str(gen_id)] = {
                    "counts": dict(sorted(rejection_counts.items())),
                    "sample_ids": {
                        key: value for key, value in sorted(rejection_samples.items()) if value
                    },
                }
            else:
                root.pop(str(gen_id), None)

        def retain_validation_candidate(
            finding: dict[str, Any],
            *,
            exclusion_reason: str,
            recommended_next_step: str = "complete_scored_validation_before_frontier_or_gems",
        ) -> bool:
            record_promotion_rejection(exclusion_reason, finding)
            validation_candidate = self._validation_candidate_entry(
                gen_id=gen_id,
                finding=finding,
            )
            if validation_candidate is None:
                return False
            validation_candidate["exclusion_reason"] = exclusion_reason
            validation_candidate["recommended_next_step"] = recommended_next_step
            maturity_metadata = compact_maturity_metadata(finding, self.maturity_policy)
            validation_candidate.update(maturity_metadata)
            metrics = validation_candidate.get("metrics")
            if isinstance(metrics, dict):
                metrics.setdefault("exclusion_reason", exclusion_reason)
                metrics.setdefault("recommended_next_step", recommended_next_step)
                for key, value in maturity_metadata.items():
                    metrics.setdefault(key, value)
            validation_candidates.append(validation_candidate)
            return True

        validation_acceptable_types = (
            "result",
            "insight",
            "intermediate_result",
            "hypothesis",
        )

        def seeds_artifact_quarantine(finding: dict[str, Any]) -> bool:
            if _candidate_has_validation_only_durability_marker(
                finding
            ) or _candidate_protocol_integrity_failed(finding):
                return True
            if _has_mature_durable_evidence(finding, self.maturity_policy):
                return False
            if _maturity_ratio_decision(finding, self.maturity_policy) is False:
                return True
            return _is_preliminary_or_incomplete_evidence(finding)

        resolved_artifacts = resolve_result_snapshot_producers(
            [(result_snapshot_key(finding), finding.get("variant_name")) for finding in findings]
        )
        artifact_by_finding = {
            id(finding): artifact
            for finding, artifact in zip(findings, resolved_artifacts, strict=True)
        }
        quarantined_artifacts = {
            artifact_key
            for finding in findings
            if seeds_artifact_quarantine(finding)
            and (artifact_key := artifact_by_finding[id(finding)]) is not None
            and bool(artifact_key[0])
        }
        for f_orig in findings:
            finding_type = f_orig.get("finding_type")
            maturity_ratio_decision = _maturity_ratio_decision(f_orig, self.maturity_policy)
            ratio_immature = maturity_ratio_decision is False
            if finding_type in validation_acceptable_types and _candidate_protocol_integrity_failed(
                f_orig
            ):
                retained = retain_validation_candidate(
                    f_orig,
                    exclusion_reason="protocol_integrity_failed",
                    recommended_next_step="rerun_with_valid_evaluator_protocol",
                )
                logger.info(
                    "frontier: excluding protocol-invalid finding %s "
                    "from durable frontier promotion%s",
                    f_orig.get("id") or f_orig.get("finding_id") or "?",
                    "; retained as validation candidate" if retained else "",
                )
                continue
            artifact_key = artifact_by_finding[id(f_orig)]
            artifact_is_quarantined = artifact_key is not None and any(
                same_result_snapshot(artifact_key, quarantined_artifact)
                for quarantined_artifact in quarantined_artifacts
            )
            if finding_type in validation_acceptable_types and (
                _candidate_has_validation_only_durability_marker(f_orig) or artifact_is_quarantined
            ):
                retained = retain_validation_candidate(
                    f_orig,
                    exclusion_reason="validation_only_or_late_quarantined_signal",
                    recommended_next_step="rerun_or_revalidate_late_signal",
                )
                logger.info(
                    "frontier: excluding validation-only/late finding %s "
                    "from durable frontier promotion%s",
                    f_orig.get("id") or f_orig.get("finding_id") or "?",
                    "; retained as validation candidate" if retained else "",
                )
                continue
            preliminary_or_incomplete = _is_preliminary_or_incomplete_evidence(f_orig)
            ratio_overrides_audit_label = (
                finding_type != "hypothesis"
                and _has_mature_durable_evidence(f_orig, self.maturity_policy)
            )
            if finding_type in validation_acceptable_types and (
                (preliminary_or_incomplete and not ratio_overrides_audit_label)
                or finding_type == "hypothesis"
            ):
                retained = False
                if (
                    finding_type == "hypothesis"
                    and _is_retainable_hypothesis_validation_candidate(f_orig)
                ) or (finding_type != "hypothesis" and _is_retainable_validation_candidate(f_orig)):
                    exclusion_reason = next(
                        (
                            str(reason or "").strip()
                            for reason in _candidate_field_values(f_orig, "exclusion_reason")
                            if str(reason or "").strip()
                        ),
                        "preliminary_or_incomplete_evidence",
                    )
                    recommended_next_step = next(
                        (
                            str(step or "").strip()
                            for step in _candidate_field_values(
                                f_orig,
                                "recommended_next_step",
                            )
                            if str(step or "").strip()
                        ),
                        "complete_scored_validation_before_frontier_or_gems",
                    )
                    retained = retain_validation_candidate(
                        f_orig,
                        exclusion_reason=exclusion_reason,
                        recommended_next_step=recommended_next_step,
                    )
                logger.info(
                    "frontier: excluding preliminary/incomplete finding %s "
                    "from durable frontier promotion%s",
                    f_orig.get("id") or f_orig.get("finding_id") or "?",
                    "; retained as validation candidate" if retained else "",
                )
                continue
            if finding_type not in acceptable_types:
                continue
            # R3-N5 fix + R6-N1/N2 + R7-Issue1/2 hardening: defense-in-depth
            # tier filter. Each finding's `tier` and `promotion_eligible`
            # fields are validated for both PRESENCE and TYPE. The four
            # rejection paths:
            #   1. require_tier=True and tier is missing/invalid → reject (strict mode).
            #   2. promotion_eligible == False (strict bool) → reject.
            #   3. promotion_eligible is a string "false"/"no" → reject (loose-string).
            # Tier labels are opaque task metadata. Core must not classify or
            # rank tier values; tasks express promotion intent through the
            # generic promotion_eligible fields.
            # Type strictness defends against peers that send non-string tier
            # metadata or unparseable promotion_eligible values.
            raw_metrics = f_orig.get("metrics")
            raw_details = f_orig.get("details")
            metrics_dict = raw_metrics if isinstance(raw_metrics, dict) else {}
            details_dict = raw_details if isinstance(raw_details, dict) else {}
            # R8-Issue10 fix: also accept top-level `tier` field. Peers
            # writing findings via the `Write` tool may put tier at the
            # top level rather than under `metrics`. Lookup order:
            # metrics.tier → details.tier → top-level tier.
            f_tier_raw = metrics_dict.get("tier")
            if f_tier_raw is None:
                f_tier_raw = details_dict.get("tier")
            if f_tier_raw is None:
                f_tier_raw = f_orig.get("tier")
            f_promo = metrics_dict.get("promotion_eligible")
            if f_promo is None:
                f_promo = details_dict.get("promotion_eligible")
            if f_promo is None:
                f_promo = f_orig.get("promotion_eligible")
            if _candidate_protocol_integrity_failed(f_orig):
                retained = retain_validation_candidate(
                    f_orig,
                    exclusion_reason="protocol_integrity_failed",
                    recommended_next_step="rerun_with_valid_evaluator_protocol",
                )
                logger.info(
                    "frontier: excluding protocol-invalid finding %s "
                    "from durable frontier promotion%s",
                    f_orig.get("id") or f_orig.get("finding_id") or "?",
                    "; retained as validation candidate" if retained else "",
                )
                continue

            # Tier labels are opaque task metadata. Only validate presence/type
            # when the task opted into requires_tier; never interpret values.
            f_tier = None
            if isinstance(f_tier_raw, str):
                f_tier = f_tier_raw.strip() or None
                if f_tier is None:
                    logger.warning(
                        f"frontier: finding {f_orig.get('finding_id', '?')} "
                        "has empty tier metadata; treating as missing"
                    )
            elif f_tier_raw is not None:
                logger.warning(
                    f"frontier: finding {f_orig.get('finding_id', '?')} "
                    f"has non-string tier {f_tier_raw!r} (type={type(f_tier_raw).__name__}); "
                    f"treating as missing"
                )

            metric_value = _get_metric(f_orig)
            if metric_value is None and not self.frontier_lanes:
                retained = retain_validation_candidate(
                    f_orig,
                    exclusion_reason="missing_primary_metric_with_validation_signal",
                )
                logger.info(
                    "frontier: excluding unknown-maturity aggregate-signal finding %s "
                    "from durable frontier promotion%s",
                    f_orig.get("id") or f_orig.get("finding_id") or "?",
                    "; retained as validation candidate" if retained else "",
                )
                continue

            # promotion_eligible: accept strict bool or normalize-stringy.
            promo_rejects = False
            if f_promo is False:
                promo_rejects = True
            elif isinstance(f_promo, str):
                if f_promo.strip().lower() in ("false", "no", "0", "non-promotable"):
                    promo_rejects = True
                elif f_promo.strip().lower() not in ("true", "yes", "1", "promotable"):
                    logger.warning(
                        f"frontier: finding {f_orig.get('finding_id', '?')} "
                        f"has unparseable promotion_eligible={f_promo!r}; "
                        f"treating as ineligible"
                    )
                    promo_rejects = True
            elif f_promo == 0:  # numeric 0 is falsy — reject defensively
                promo_rejects = True

            risk_reason = None
            if metric_value is not None:
                risk_reason = self._risk_violating_reason(
                    finding=f_orig,
                    tier=f_tier,
                    promotion_rejected=promo_rejects,
                    metric_value=metric_value,
                )
            risk_candidate = risk_reason is not None
            missing_tier_candidate = False
            non_promotable_candidate = False

            if getattr(self, "_require_tier", False) and f_tier is None:
                if self._lanes_allow_missing_tier:
                    missing_tier_candidate = True
                    logger.warning(
                        f"frontier: admitting finding "
                        f"{f_orig.get('finding_id', '?')} with missing/invalid "
                        f"tier only for lanes that explicitly allow missing tier"
                    )
                else:
                    retained = retain_validation_candidate(
                        f_orig,
                        exclusion_reason="missing_required_tier_metadata",
                        recommended_next_step="repair_tier_metadata_or_complete_scored_validation",
                    )
                    logger.warning(
                        f"frontier: rejecting finding "
                        f"{f_orig.get('finding_id', '?')} — missing/invalid tier "
                        f"metadata and task requires it (requires_tier=True)%s",
                        "; retained as validation candidate" if retained else "",
                    )
                    continue

            if promo_rejects and not risk_candidate:
                if self._lanes_allow_non_promotable:
                    non_promotable_candidate = True
                    logger.info(
                        f"frontier: admitting non-promotable finding "
                        f"{f_orig.get('finding_id', '?')} only for lanes that "
                        f"explicitly allow non-promotable incubation "
                        f"(promotion_eligible={f_promo!r})"
                    )
                else:
                    retained = retain_validation_candidate(
                        f_orig,
                        exclusion_reason="promotion_eligible_false",
                        recommended_next_step="repair_or_ablate_non_promotable_signal",
                    )
                    logger.info(
                        "frontier: skipping non-promotable finding %s (promotion_eligible=%r)%s",
                        f_orig.get("finding_id", "?"),
                        f_promo,
                        "; retained as validation candidate" if retained else "",
                    )
                    continue
            if (
                risk_candidate or missing_tier_candidate or non_promotable_candidate
            ) and not _has_mature_durable_evidence(f_orig, self.maturity_policy):
                retained = retain_validation_candidate(
                    f_orig,
                    exclusion_reason="immature_lane_exception",
                )
                logger.info(
                    "frontier: excluding immature lane-exception finding %s "
                    "from durable frontier promotion%s",
                    f_orig.get("id") or f_orig.get("finding_id") or "?",
                    "; retained as validation candidate" if retained else "",
                )
                continue
            if ratio_immature:
                retained = retain_validation_candidate(
                    f_orig,
                    exclusion_reason="insufficient_mature_evidence_ratio",
                )
                logger.info(
                    "frontier: excluding ratio-immature finding %s "
                    "from durable frontier promotion%s",
                    f_orig.get("id") or f_orig.get("finding_id") or "?",
                    "; retained as validation candidate" if retained else "",
                )
                continue
            if not _has_mature_durable_evidence(f_orig, self.maturity_policy):
                retained = retain_validation_candidate(
                    f_orig,
                    exclusion_reason="unknown_maturity_or_incomplete_evidence",
                )
                logger.info(
                    "frontier: excluding unknown-maturity finding %s "
                    "from durable frontier promotion%s",
                    f_orig.get("id") or f_orig.get("finding_id") or "?",
                    "; retained as validation candidate" if retained else "",
                )
                continue
            f = copy.copy(f_orig)
            # Also copy the metrics sub-dict (which we modify), but
            # leave nested values referenced by both copies — we don't
            # mutate any deeper than `metrics`.
            existing_metrics = f.get("metrics")
            f["metrics"] = dict(existing_metrics) if isinstance(existing_metrics, dict) else {}
            for key, value in _research_metadata_from_finding(f_orig).items():
                if f["metrics"].get(key) in (None, ""):
                    f["metrics"][key] = value
            if metric_value is not None:
                f["metrics"][self.primary_metric] = metric_value
            for key, value in compact_maturity_metadata(f_orig, self.maturity_policy).items():
                f[key] = value
                f["metrics"][key] = value
            peer_role_str = _candidate_role(f_orig)
            if peer_role_str:
                f["_candidate_peer_role"] = peer_role_str
                f["metrics"].setdefault("peer_role", peer_role_str)
            if risk_candidate:
                f["_risk_violating_frontier_candidate"] = True
                f["metrics"]["risk_violating_frontier_candidate"] = True
                f["metrics"]["risk_repair_required"] = True
                f["metrics"]["risk_violation_reason"] = risk_reason
                f["metrics"]["clean_promotion_eligible"] = False
                logger.info(
                    "frontier: admitting risk-violating repair candidate %s (variant=%s): %s",
                    f_orig.get("finding_id", f_orig.get("id", "?")),
                    f_orig.get("variant_name", "?"),
                    risk_reason,
                )
            if missing_tier_candidate:
                f["_lane_missing_tier_candidate"] = True
                f["metrics"]["lane_missing_tier_candidate"] = True
            if non_promotable_candidate:
                f["_lane_non_promotable_candidate"] = True
                f["metrics"]["lane_non_promotable_candidate"] = True
                f["metrics"]["clean_promotion_eligible"] = False
            if promo_rejects:
                f["parent_eligible"] = False
                f["metrics"]["parent_eligible"] = False
            candidates.append(f)

        if self.frontier_lanes:
            routed_candidates: list[dict[str, Any]] = []
            for candidate in candidates:
                if any(_matches_lane_filters(candidate, lane) for lane in self.frontier_lanes):
                    routed_candidates.append(candidate)
                    continue
                retain_validation_candidate(
                    candidate,
                    exclusion_reason="no_matching_frontier_lane",
                    recommended_next_step=(
                        "preserve_signal_and_republish_with_an_explicit_task_owned_lane"
                    ),
                )
            candidates = routed_candidates

        self._record_validation_candidates(gen_id=gen_id, entries=validation_candidates)
        self._retire_validation_candidates_for_durable_entities()

        generations = self._manifest.get("generations")
        has_existing_generation = isinstance(generations, dict) and str(gen_id) in generations
        if not candidates:
            logger.warning(
                f"Generation {gen_id}: no findings carry "
                f"'{self.primary_metric}' — nothing to promote"
            )
            if not has_existing_generation:
                if validation_candidates:
                    flush_promotion_rejections()
                    self._save_manifest()
                return []
            top_k = []
        elif self.frontier_lanes:
            # Lane mode ranks inside each lane with lane-local required and
            # optional axes. Do not sort or deduplicate globally by the
            # primary metric first: durable-candidate/control lanes may intentionally
            # use task-local axes and may not carry the global primary metric.
            top_k = self._select_lane_frontier(
                candidates,
                exclude_generation_id=gen_id,
            )
        else:
            reverse = self.metric_direction == "maximize"

            # Dedup by concrete frontier entity BEFORE taking top-K.
            #
            # Why: the same variant can produce multiple result-findings via
            # several legitimate paths:
            #   (a) Peer calls share_finding via MCP (uuid4 random id), AND
            #       a task-side auto-emit also writes one (uuid5 deterministic
            #       id). Different ids, same variant, near-identical metrics
            #       → 2 candidates with similar scores.
            #   (b) Two peers accidentally pick the same variant_name and
            #       both run + emit (observed in a prior run where two peers
            #       independently produced "cosine_lr_min" variants).
            #   (c) Re-runs of a flaky variant where one of multiple peers
            #       got partial credit.
            #
            # Without this dedup, top-K could be occupied by N copies of the
            # same variant, crowding out other promising variants. With
            # promote_top_k=3, a single duplicated variant can eat 2/3 of
            # the slots — observed in a prior Gen 0 where the duplicated
            # variant took rank 2-3, hiding two distinct better variants
            # behind it.
            #
            # Result-artifact paths now take precedence over stale broad
            # persisted keys and broad sweep-family variant IDs. That keeps
            # distinct scored sweep children visible while still merging a
            # family summary row with its canonical child result row.
            def _primary_rank_key(finding: dict[str, Any]) -> tuple[Any, ...]:
                metric_value = finding["metrics"][self.primary_metric]
                directional_value = metric_value if reverse else -metric_value
                return (
                    _evidence_maturity_rank(finding),
                    directional_value,
                    str(finding.get("variant_name") or finding.get("id") or ""),
                )

            best_by_entity: dict[str, dict[str, Any]] = {}
            for f in candidates:
                entity_key = _candidate_entity_key(f)
                incumbent = best_by_entity.get(entity_key)
                if incumbent is None or _primary_rank_key(f) > _primary_rank_key(incumbent):
                    best_by_entity[entity_key] = f
            deduped = list(best_by_entity.values())
            deduped.sort(key=_primary_rank_key, reverse=True)
            n_dropped_dups = len(candidates) - len(deduped)
            if n_dropped_dups:
                logger.info(
                    f"Generation {gen_id}: dedup removed {n_dropped_dups} "
                    f"duplicate-entity finding(s) before top-K selection"
                )
            # Multi-anchor (Pareto) promotion: in addition to the top-K by
            # primary_metric, also promote the single best finding under each
            # of an optional set of "secondary anchor metrics" (configured
            # via ``self.anchor_metrics``). Each secondary anchor has its own
            # direction (maximize / minimize). The intent is to break the
            # single-hub anchor effect (documented in prior behavior
            # analyses): with multi-anchor, frontier consumers (next-gen peers)
            # see >= 2 different "winners" per generation, each optimal in a
            # different dimension, encouraging algorithmic diversity in
            # subsequent exploitation. The primary_metric anchors are
            # preserved at ranks 1..K; secondary anchors append after,
            # deduplicated against the primary picks. If no secondary anchors
            # are configured, behavior is identical to the pre-2026-04-30
            # single-metric mode.
            primary_picks = deduped[: self.promote_top_k]
            secondary_picks: list[dict[str, Any]] = []
            primary_pick_ids = {f.get("id", "") for f in primary_picks if f.get("id", "")}
            selected_entity_keys = {_candidate_entity_key(f) for f in primary_picks}
            # M8 fix (review round 2): anchor selection iterates the full
            # `candidates` (pre-dedup), not `deduped`. The dedup goal is
            # "don't crowd top-K with N copies of the same variant" —
            # but for anchor selection, dedup discards exactly the
            # diversification we want (a variant tied on primary metric
            # but breakthrough on secondary). We re-apply variant-level
            # dedup against primary picks separately below, AND track
            # anchor picks added so far so a single variant doesn't claim
            # multiple anchors.
            for anchor_name, anchor_dir in self.anchor_metrics or []:
                best_for_anchor = None
                best_val = None
                best_anchor_key = None
                for f in candidates:
                    f_id = str(f.get("id", "") or "")
                    entity_key = _candidate_entity_key(f)
                    # Skip if this finding is already in primary picks.
                    if f_id and f_id in primary_pick_ids:
                        continue
                    # Skip if this concrete entity already won primary or an
                    # earlier anchor. This preserves distinct result-artifact
                    # children under a shared sweep-family variant_name.
                    if entity_key in selected_entity_keys:
                        continue
                    v = _metric_value(f, anchor_name)
                    if v is None:
                        continue
                    directional_value = v if anchor_dir == "maximize" else -v
                    anchor_key = (
                        _evidence_maturity_rank(f),
                        directional_value,
                        str(f.get("variant_name") or f.get("id") or ""),
                    )
                    if best_anchor_key is None or anchor_key > best_anchor_key:
                        best_for_anchor = f
                        best_val = v
                        best_anchor_key = anchor_key
                if best_for_anchor is not None:
                    # Shallow-copy + mark, mirroring primary-pick handling.
                    import copy as _copy

                    best_for_anchor = _copy.copy(best_for_anchor)
                    existing = best_for_anchor.get("metrics")
                    best_for_anchor["metrics"] = (
                        dict(existing) if isinstance(existing, dict) else {}
                    )
                    best_for_anchor["metrics"][anchor_name] = best_val
                    best_for_anchor["_promoted_for_anchor"] = anchor_name
                    secondary_picks.append(best_for_anchor)
                    # Round 5 m3 fix: only add non-empty id to the
                    # exclusion set; empty-string sentinel would
                    # spuriously block legitimate later candidates with
                    # missing ids.
                    bf_id = best_for_anchor.get("id", "")
                    if bf_id:
                        primary_pick_ids.add(bf_id)
                    selected_entity_keys.add(_candidate_entity_key(best_for_anchor))
                    logger.info(
                        "Generation %d: secondary anchor '%s' (%s) selected variant %s with %s=%.4f",
                        gen_id,
                        anchor_name,
                        anchor_dir,
                        best_for_anchor.get("variant_name", "?"),
                        anchor_name,
                        best_val,
                    )

            top_k = primary_picks + secondary_picks

        # Persist to filesystem
        gen_dir = self.base_dir / f"gen_{gen_id}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        existing_generation_entries = self._manifest.get("generations", {}).get(str(gen_id), [])
        existing_promoted_at: dict[
            tuple[str, tuple[str, str, str], str],
            str,
        ] = {}
        if isinstance(existing_generation_entries, list):
            for entry in existing_generation_entries:
                if not isinstance(entry, dict):
                    continue
                identity = (
                    str(entry.get("frontier_lane") or entry.get("promoted_for_lane") or ""),
                    _lane_capacity_identity(entry),
                    str(entry.get("finding_id") or ""),
                )
                promoted_at = str(entry.get("promoted_at") or "")
                if promoted_at:
                    existing_promoted_at[identity] = promoted_at

        promoted_entries = []
        canonical_sources = self._canonical_result_source_index()
        expected_generation_artifacts: set[Path] = set()
        for rank, finding in enumerate(top_k, 1):
            import copy as _copy

            finding = _copy.copy(finding)
            existing_metrics = finding.get("metrics")
            finding["metrics"] = (
                _drop_legacy_protocol_alias(existing_metrics)
                if isinstance(existing_metrics, dict)
                else {}
            )
            if canonical_sources[1]:
                self._repair_manifest_entry_canonical_source(
                    finding,
                    by_source_path=canonical_sources[0],
                    sources_by_variant=canonical_sources[1],
                    generation_key=str(gen_id),
                )
            evidence_metadata = _evidence_metadata_from_candidate(finding, self.maturity_policy)
            research_metadata = _research_metadata_from_finding(finding)
            for key, value in evidence_metadata.items():
                if key in RESULT_RESEARCH_METADATA_KEYS and key in research_metadata:
                    continue
                if key == "frontier_entity_key":
                    finding[key] = value
                    finding["metrics"][key] = value
                else:
                    finding[key] = value
                    finding["metrics"][key] = value
            finding = _sanitize_nonfinite_json(finding)
            finding_metrics = (
                finding.get("metrics") if isinstance(finding.get("metrics"), dict) else {}
            )

            # Save finding
            finding_file = gen_dir / f"top_{rank}_finding.json"
            finding_tmp = finding_file.with_name(f".{finding_file.name}.{os.getpid()}.tmp")
            try:
                with open(finding_tmp, "w") as f:
                    json.dump(finding, f, indent=2, default=str)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(finding_tmp, finding_file)
            finally:
                with contextlib.suppress(OSError):
                    finding_tmp.unlink()
            expected_generation_artifacts.add(finding_file)

            # Try to freeze workspace snapshot if path is available
            snapshot_path = self._freeze_snapshot(finding, gen_dir / f"top_{rank}_snapshot.tar.gz")
            if snapshot_path is not None:
                expected_generation_artifacts.add(Path(snapshot_path))
            primary_metric_value = _coerce_finite_float(finding_metrics.get(self.primary_metric))
            lane_metric_name = str(finding.get("_lane_metric_name") or "").strip()
            lane_metric_value = _coerce_finite_float(finding.get("_lane_metric_value"))
            lane_metric_direction = str(finding.get("_lane_metric_direction") or "").strip()
            metric_value = (
                primary_metric_value if primary_metric_value is not None else lane_metric_value
            )
            metric_direction = (
                self.metric_direction
                if primary_metric_value is not None
                else _normalize_metric_direction(
                    lane_metric_direction, default=self.metric_direction
                )
            )
            metric_name = (
                self.primary_metric
                if primary_metric_value is not None
                else lane_metric_name or self.primary_metric
            )

            compact_entry_metrics = dict(finding_metrics)
            compact_entry_metrics.pop("design_dimensions", None)
            compact_entry_metrics.pop("realized_dimensions", None)
            entry = {
                "generation_id": gen_id,
                "rank": rank,
                "finding_id": finding.get("id", ""),
                "variant_name": finding.get("variant_name", finding.get("notes", "")[:80]),
                "metric_name": metric_name,
                "metric_value": metric_value,
                "metric_direction": metric_direction,
                "metrics": compact_entry_metrics,
                "snapshot_path": str(snapshot_path) if snapshot_path else None,
            }
            design_dimensions = _extract_design_dimensions(finding)
            if design_dimensions:
                entry["design_dimensions"] = design_dimensions
            for container_name in ("details", "extra", "current_aggregate"):
                compact_identity = _compact_result_identity_container(finding.get(container_name))
                if compact_identity:
                    entry[container_name] = compact_identity
            for identity_key in dict.fromkeys(
                (
                    *_RESULT_PRODUCER_IDENTITY_KEYS,
                    "variant_id",
                    "child_variant_name",
                    "child_variant_id",
                    "result_path",
                    "source_result_path",
                    "result_artifact_path",
                    "summary_path",
                    "source_result_sha256",
                    *_RESULT_SOURCE_AUDIT_PATH_KEYS,
                    "canonical_source_result_path",
                    "best_available_summary_path",
                    "source_selection_warning",
                    "source_selection_reason",
                    "canonical_variant_id",
                    "canonical_metric_value",
                )
            ):
                value = _raw_candidate_field(finding, identity_key)
                if value not in (None, ""):
                    entry.setdefault(identity_key, value)
            # Mark secondary anchor picks so frontier consumers can
            # display them as "best on dimension X" instead of "rank N
            # by FF". The transient ``_promoted_for_anchor`` field set
            # in the secondary-pick loop is propagated here.
            anchor_name = finding.get("_promoted_for_anchor")
            if anchor_name:
                entry["promoted_for_anchor"] = anchor_name
                entry["anchor_metric_value"] = _coerce_finite_float(
                    finding_metrics.get(anchor_name)
                )
            lane_name = finding.get("_promoted_for_lane")
            if lane_name:
                entry["promoted_for_lane"] = lane_name
                entry["frontier_lane"] = lane_name
                entry["lane_metric_name"] = finding.get("_lane_metric_name")
                entry["lane_metric_value"] = lane_metric_value
                entry["lane_metric_direction"] = _normalize_metric_direction(
                    lane_metric_direction,
                    default=self.metric_direction,
                )
                if finding.get("_source_frontier_lane"):
                    entry["source_frontier_lane"] = finding.get("_source_frontier_lane")
            if finding.get("_risk_violating_frontier_candidate"):
                entry["risk_violating_frontier_candidate"] = True
                entry["risk_repair_required"] = True
                entry["risk_violation_reason"] = finding_metrics.get("risk_violation_reason")
            for key, value in research_metadata.items():
                entry[key] = value
            for key, value in evidence_metadata.items():
                entry.setdefault(key, value)
            parent_eligible = _resolved_boolish_candidate_field(finding, "parent_eligible")
            if parent_eligible is not None:
                entry["parent_eligible"] = parent_eligible
            commit_identity = (
                str(entry.get("frontier_lane") or entry.get("promoted_for_lane") or ""),
                _lane_capacity_identity(entry),
                str(entry.get("finding_id") or ""),
            )
            entry["promoted_at"] = existing_promoted_at.get(
                commit_identity,
                datetime.now().isoformat(),
            )
            promoted_entries.append(entry)

        for artifact in gen_dir.iterdir():
            if not artifact.is_file() or artifact in expected_generation_artifacts:
                continue
            if re.fullmatch(
                r"top_\d+_(?:finding\.json|snapshot\.tar\.gz)(?:\.tmp)?",
                artifact.name,
            ):
                with contextlib.suppress(OSError):
                    artifact.unlink()

        # Update manifest
        self._manifest["generations"][str(gen_id)] = promoted_entries
        flush_promotion_rejections()
        self._repair_manifest_canonical_result_sources()

        # Update cumulative top
        self._update_cumulative_top()
        self._prune_durable_frontier_entries(migrate=False)
        self._save_manifest()
        promoted_entries = list(self._manifest["generations"].get(str(gen_id), []))

        # m4 fix (review round 2): guard against IndexError when
        # promoted_entries is empty (promote_top_k=0 + no anchors).
        if promoted_entries:
            best_metric = _coerce_finite_float(promoted_entries[0].get("metric_value"))
            if best_metric is not None:
                logger.info(
                    "Generation %d: promoted %d findings, best metric = %.4f",
                    gen_id,
                    len(promoted_entries),
                    best_metric,
                )
            else:
                logger.info(
                    "Generation %d: promoted %d findings without a numeric display metric",
                    gen_id,
                    len(promoted_entries),
                )
        else:
            logger.warning(
                "Generation %d: promoted 0 findings (promote_top_k=%d, "
                "anchors=%d). Did you set promote_top_k=0 by mistake?",
                gen_id,
                self.promote_top_k,
                len(self.anchor_metrics),
            )

        return promoted_entries

    def _freeze_snapshot(self, finding: dict[str, Any], target_path: Path) -> Path | None:
        """Freeze a workspace snapshot as tar.gz."""
        snapshot_src = finding.get("snapshot_local_path")
        if not snapshot_src or not Path(snapshot_src).exists():
            # Also check for snapshot_s3_key (would need download)
            return None

        # Round 5 m4 fix: write to .tmp then atomic-rename so
        # consumers (next-gen peers via download_snapshot) can never see
        # a half-written tarball. Without this, a download mid-write
        # raises tarfile.ReadError with no retry.
        target_path = Path(target_path)
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        try:
            with tarfile.open(tmp_path, "w:gz") as tar:
                tar.add(
                    snapshot_src,
                    arcname=Path(snapshot_src).name,
                    filter=self._tar_filter,
                )
            os.replace(tmp_path, target_path)
            return target_path
        except Exception as e:
            logger.warning(f"Failed to freeze snapshot: {e}")
            # Clean up the partial tmp file if it exists.
            with contextlib.suppress(OSError, AttributeError):
                tmp_path.unlink(missing_ok=True)
            return None

    @staticmethod
    def _tar_filter(tarinfo):
        """Exclude noise + reject security-risk entries from snapshots.

        M7 fix (review round 2): the prior version followed symlinks
        (tarfile default), so a malicious or buggy training run could
        symlink credentials / system files into the snapshot dir and
        ship them to the frozen tarball. We now reject:
          - symbolic and hard links (issym / islnk)
          - device files (chr / blk)
          - FIFOs
          - oversized regular files (> SNAPSHOT_FILE_SIZE_CAP)
        Only regular files and directories are kept.
        """
        skip = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            "node_modules",
            ".venv",
            "venv",
            "unsloth_compiled_cache",
        }
        name = tarinfo.name
        for s in skip:
            if s in name:
                return None
        # Reject symlinks / hardlinks / devices / FIFOs.
        if tarinfo.issym() or tarinfo.islnk():
            return None
        if tarinfo.ischr() or tarinfo.isblk() or tarinfo.isfifo():
            return None
        # Per-file size cap: 256 MB. A single ckpt rarely exceeds this;
        # anything larger is suspicious or accidental and bloats the
        # tarball + slows downstream snapshot transfers.
        SNAPSHOT_FILE_SIZE_CAP = 256 * 1024 * 1024
        if tarinfo.isfile() and tarinfo.size > SNAPSHOT_FILE_SIZE_CAP:
            return None
        return tarinfo

    def _update_cumulative_top(self):
        """Update cumulative top across all generations.

        Round 3 C1 fix: anchor picks are preserved across ALL
        generations, not just the latest. Without this, secondary
        anchor picks from gen 0..N-1 were silently truncated by the
        primary-metric sort because their selling point is precisely
        that they're NOT primary leaders. The diversity-penalty prompt
        block then saw only "primary picks + latest-gen anchors",
        which silently degraded after the first generation.

        New strategy:
        - All anchor picks across all generations are preserved
          unconditionally (small set: at most
          ``len(anchor_metrics) * num_generations`` entries).
        - Primary picks are sorted by primary metric and capped at
          ``(promote_top_k + len(anchor_metrics)) * 2``.
        - Cumulative_top is the union: primary-truncated + all anchors.
        """
        # m9 fix (review round 2): tolerate non-numeric / corrupt keys.
        gen_ids: list[int] = []
        for g in self._manifest.get("generations", {}):
            try:
                gen_ids.append(int(g))
            except (TypeError, ValueError):
                logger.warning("Skipping non-numeric generation key %r in manifest", g)

        if self.frontier_lanes:
            lane_frontiers, cumulative = _build_cumulative_lane_views(
                self._manifest["generations"],
                self.frontier_lanes,
                maturity_policy=self.maturity_policy,
                primary_metric=self.primary_metric,
                metric_direction=self.metric_direction,
                promote_top_k=self.promote_top_k,
            )
            self._manifest["lane_frontiers"] = lane_frontiers
            self._manifest["cumulative_top"] = cumulative
            return

        # Split entries into primary picks (no `promoted_for_anchor`)
        # and anchor picks (has `promoted_for_anchor`).
        all_primary: list[dict[str, Any]] = []
        all_anchors: list[dict[str, Any]] = []
        for gen_entries in self._manifest["generations"].values():
            for e in gen_entries:
                if not isinstance(e, dict) or not _is_committed_frontier_entry(
                    e, self.maturity_policy
                ):
                    continue
                if e.get("promoted_for_anchor"):
                    all_anchors.append(e)
                else:
                    all_primary.append(e)

        reverse = self.metric_direction == "maximize"

        def _legacy_cumulative_key(entry: dict[str, Any]) -> tuple[Any, ...]:
            value = _coerce_finite_float(entry.get("metric_value"))
            if value is None:
                directional_value = float("-inf")
            else:
                directional_value = value if reverse else -value
            try:
                gen_value = int(entry.get("generation_id", -1))
            except (TypeError, ValueError):
                gen_value = -1
            return (
                _evidence_maturity_rank(entry),
                directional_value,
                gen_value,
                str(entry.get("variant_name") or entry.get("finding_id") or ""),
            )

        best_primary_by_entity: dict[str, dict[str, Any]] = {}
        for entry in all_primary:
            entity_key = _candidate_entity_key(entry)
            incumbent = best_primary_by_entity.get(entity_key)
            if incumbent is None or _legacy_cumulative_key(entry) > _legacy_cumulative_key(
                incumbent
            ):
                best_primary_by_entity[entity_key] = entry
        if len(best_primary_by_entity) < len(all_primary):
            logger.info(
                "frontier legacy cumulative dedup removed %d duplicate evidence row(s)",
                len(all_primary) - len(best_primary_by_entity),
            )
        all_primary = sorted(
            best_primary_by_entity.values(),
            key=_legacy_cumulative_key,
            reverse=True,
        )

        # Cap primary picks at 2 × (top_k + anchors), unchanged in spirit.
        cap = (self.promote_top_k + len(self.anchor_metrics)) * 2
        truncated_primary = all_primary[:cap]
        truncated_ids = {e.get("finding_id") for e in truncated_primary if e.get("finding_id")}
        selected_entity_keys = {_candidate_entity_key(e) for e in truncated_primary}

        def _legacy_anchor_cumulative_key(entry: dict[str, Any]) -> tuple[Any, ...]:
            try:
                gen_value = int(entry.get("generation_id", -1))
            except (TypeError, ValueError):
                gen_value = -1
            return (
                _evidence_maturity_rank(entry),
                gen_value,
                str(entry.get("variant_name") or entry.get("finding_id") or ""),
            )

        best_anchor_by_entity: dict[str, dict[str, Any]] = {}
        for entry in all_anchors:
            entity_key = _candidate_entity_key(entry)
            if entity_key in selected_entity_keys:
                continue
            incumbent = best_anchor_by_entity.get(entity_key)
            if incumbent is None or _legacy_anchor_cumulative_key(
                entry
            ) > _legacy_anchor_cumulative_key(incumbent):
                best_anchor_by_entity[entity_key] = entry
        if len(best_anchor_by_entity) < len(all_anchors):
            logger.info(
                "frontier legacy anchor cumulative dedup removed %d duplicate evidence row(s)",
                len(all_anchors) - len(best_anchor_by_entity),
            )
        # Sort anchors by maturity and generation (newest first), but keep every
        # distinct anchor entity. Prompt builders compact their view separately;
        # cumulative frontier state should not silently drop diversity anchors.
        all_anchors_sorted = sorted(
            best_anchor_by_entity.values(),
            key=_legacy_anchor_cumulative_key,
            reverse=True,
        )
        preserved_anchors = []
        for ap in all_anchors_sorted:
            entity_key = _candidate_entity_key(ap)
            finding_id = ap.get("finding_id")
            if (
                not finding_id or finding_id not in truncated_ids
            ) and entity_key not in selected_entity_keys:
                preserved_anchors.append(ap)
                if finding_id:
                    truncated_ids.add(finding_id)
                selected_entity_keys.add(entity_key)

        self._manifest["cumulative_top"] = truncated_primary + preserved_anchors

    def get_summary(self) -> list[dict[str, Any]]:
        """Get frontier summary for prompt injection.

        Returns a NEW list (not the internal cumulative_top reference)
        so callers can safely iterate / filter without affecting other
        readers (M5 from review round 3).
        """
        trust_committed_membership = is_committed_runtime_fact_source(
            self._manifest,
            legacy_ok=False,
        )
        return [
            dict(entry)
            for entry in self._manifest.get("cumulative_top", [])
            if isinstance(entry, dict)
            and (
                trust_committed_membership
                or _is_committed_frontier_entry(entry, self.maturity_policy)
            )
        ]

    @staticmethod
    def _lane_allows_parents(lane: dict[str, Any]) -> bool:
        allow_lower_tier = bool(lane.get("allow_lower_tier"))
        return bool(lane.get("parent_eligible", not allow_lower_tier))

    def get_parent_summary_up_to_generation(self, gen_id: int) -> list[dict[str, Any]]:
        """Return only durable entries that task policy allows as parents."""

        if not self.frontier_lanes:
            return [
                entry
                for entry in self.get_summary_up_to_generation(gen_id)
                if _resolved_boolish_candidate_field(entry, "parent_eligible") is not False
            ]
        cutoff = int(gen_id)
        trust_committed_membership = is_committed_runtime_fact_source(
            self._manifest,
            legacy_ok=False,
        )
        lane_frontiers = self._manifest.get("lane_frontiers")
        lane_frontiers = lane_frontiers if isinstance(lane_frontiers, dict) else {}
        out: list[dict[str, Any]] = []
        seen_entities: set[str] = set()
        for lane in self.frontier_lanes:
            lane_name = str(lane.get("name") or "")
            if not lane_name or not self._lane_allows_parents(lane):
                continue
            entries = lane_frontiers.get(lane_name, [])
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if not trust_committed_membership and not _is_committed_frontier_entry(
                    entry,
                    self.maturity_policy,
                ):
                    continue
                explicit = _resolved_boolish_candidate_field(entry, "parent_eligible")
                if explicit is False:
                    continue
                generation = self._entry_generation_id(entry, default=cutoff + 1)
                if generation > cutoff:
                    continue
                entity_key = _candidate_entity_key(entry)
                if entity_key in seen_entities:
                    continue
                seen_entities.add(entity_key)
                out.append(dict(entry))
        return out

    def get_summary_up_to_generation(self, gen_id: int) -> list[dict[str, Any]]:
        """Get cumulative frontier entries whose source generation is not future."""

        cutoff = int(gen_id)
        trust_committed_membership = is_committed_runtime_fact_source(
            self._manifest,
            legacy_ok=False,
        )
        out: list[dict[str, Any]] = []
        for entry in self._manifest.get("cumulative_top", []):
            if not isinstance(entry, dict):
                continue
            generation = explicit_entry_generation_id(entry)
            if generation is None:
                continue
            if generation > cutoff:
                continue
            if trust_committed_membership or _is_committed_frontier_entry(
                entry,
                self.maturity_policy,
            ):
                out.append(dict(entry))
        return out

    def get_summary_for_generation(self, gen_id: int) -> list[dict[str, Any]]:
        """Get promoted frontier entries for one generation."""

        generations = self._manifest.get("generations")
        if not isinstance(generations, dict):
            return []
        entries = generations.get(str(gen_id), [])
        if not isinstance(entries, list):
            return []
        trust_committed_membership = is_committed_runtime_fact_source(
            self._manifest,
            legacy_ok=False,
        )
        return [
            dict(entry)
            for entry in entries
            if isinstance(entry, dict)
            and (
                trust_committed_membership
                or _is_committed_frontier_entry(entry, self.maturity_policy)
            )
            and (
                (
                    entry_generation := explicit_entry_generation_id(
                        entry,
                        generation_hint=gen_id,
                    )
                )
                is not None
                and entry_generation <= int(gen_id)
            )
        ]

    def get_generation_top_metrics(self) -> dict[int, float | None]:
        """Get top metric value per generation (for plateau detection)."""
        result = {}
        trust_committed_membership = is_committed_runtime_fact_source(
            self._manifest,
            legacy_ok=False,
        )
        for gen_str, entries in self._manifest.get("generations", {}).items():
            durable_entries = [
                entry
                for entry in entries
                if isinstance(entry, dict)
                and (
                    trust_committed_membership
                    or _is_committed_frontier_entry(entry, self.maturity_policy)
                )
            ]
            if durable_entries:
                result[int(gen_str)] = durable_entries[0].get("metric_value")
            else:
                result[int(gen_str)] = None
        return result

    def get_manifest(self) -> dict[str, Any]:
        """Get full manifest."""
        return self._manifest.copy()


# -----------------------------------------------------------------------------
# Diversity-overlap measurement (Option A: observe + log, no enforcement)
# -----------------------------------------------------------------------------


def _extract_design_dimensions(finding_or_entry: dict[str, Any]) -> dict[str, str] | None:
    """Pull the design_dimensions self-report from a finding or
    promoted-frontier entry. Peers are asked to populate this in their
    ``share_finding`` payload during explore phase (see
    ``_build_diversity_penalty_block``). Returns a dict
    {dim_name: value (str)} or None if missing/malformed.

    Tolerates top-level and nested ``metrics`` layouts. The canonical
    ``design_dimensions`` name is preferred; ``realized_dimensions`` is an
    accepted task-agnostic alias.
    """
    metrics = finding_or_entry.get("metrics")
    chosen: dict[str, Any] | None = None
    for key in ("design_dimensions", "realized_dimensions"):
        top = finding_or_entry.get(key)
        nested = metrics.get(key) if isinstance(metrics, dict) else None
        if isinstance(nested, dict):
            chosen = nested
            break
        if isinstance(top, dict):
            chosen = top
            break
    if chosen is None:
        return None
    # Normalize all values to str + lowercase + strip for comparison.
    out: dict[str, str] = {}
    for k, v in chosen.items():
        if v is None:
            continue
        try:
            out[str(k).strip()] = str(v).strip().lower()
        except (TypeError, ValueError):
            continue
    return out or None


def compute_dimension_overlap(
    finding: dict[str, Any],
    anchor: dict[str, Any],
) -> dict[str, Any] | None:
    """Compute design-dimension overlap between a finding and an anchor.

    Returns a dict with:
        overlap_count: int (matching dimensions)
        total_dims:    int (dimensions present in BOTH)
        overlap_fraction: float (overlap_count / total_dims)
        common_dims:   sorted list of dimension names compared

    Returns ``None`` if either side lacks ``design_dimensions``, or
    they share no dimensions in common (no signal).
    """
    f_dims = _extract_design_dimensions(finding)
    a_dims = _extract_design_dimensions(anchor)
    if not f_dims or not a_dims:
        return None
    common = set(f_dims.keys()) & set(a_dims.keys())
    if not common:
        return None
    matches = sum(1 for k in common if f_dims[k] == a_dims[k])
    total = len(common)
    return {
        "overlap_count": matches,
        "total_dims": total,
        "overlap_fraction": matches / total if total > 0 else 0.0,
        "common_dims": sorted(common),
    }


def annotate_findings_with_diversity_overlap(
    findings: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    expected_dim_count: int,
) -> list[dict[str, Any]]:
    """For each finding in ``findings``, compute its max-overlap against
    any anchor in ``anchors``, and annotate the finding's ``metrics``
    dict with diversity observation fields.

    **Option A semantics**: OBSERVE + LOG only. Promotion ordering is
    unaffected.

    **Three-tier signal** (replaces the prior boolean "differ ≥ ⌈M/2⌉"):

      - ``"clone"``  — finding matches anchor on ALL compared
                       dimensions (true clone; strong warning).
      - ``"narrow"`` — finding matches anchor on > ⌊M/2⌋ but < M
                       dimensions (narrow variation; soft warning).
                       NOT a violation — many legitimate research
                       contributions are narrow refinements of a baseline,
                       sharing most dimensions but introducing one important
                       mechanism.
      - ``"clean"``  — finding differs on ≥ ⌈M/2⌉ dimensions
                       (substantive novelty).
      - ``"no_data"`` — one side lacks comparable ``design_dimensions``.
      - ``"no_anchors"`` — frontier was empty (gen 0).

    Annotated fields on each finding's ``metrics`` dict:

      ``metrics.diversity_overlap_count``       — match count vs most-similar anchor
      ``metrics.diversity_overlap_total``       — dimensions compared (intersection)
      ``metrics.diversity_overlap_fraction``    — overlap / total
      ``metrics.diversity_most_similar_anchor`` — anchor variant_name
      ``metrics.diversity_overlap_status``      — clone / narrow / clean / no_data / no_anchors
      ``metrics.diversity_violated``            — bool, True iff status == "clone"
                                                   (kept for backward-compat with
                                                   dashboards that read this field;
                                                   semantics now stricter than
                                                   pre-2026-04-30: only clones.)
      ``metrics.diversity_narrow_variation``    — bool, True iff status == "narrow"
                                                   (new field, distinguishes "narrow
                                                   refinement" from "true clone")

    ``expected_dim_count`` should be ``len(diversity_dimensions)`` — the
    full M for threshold computation. The actual overlap_total may be
    smaller if the finding and anchor only have a subset in common.
    The clone/narrow thresholds are applied on the OVERLAP_TOTAL
    (dimensions present in BOTH), not on expected_dim_count, since
    we can't compare on dimensions one side didn't report.

    Returns NEW finding dicts (shallow-copied).
    """
    import copy

    annotated: list[dict[str, Any]] = []
    if not anchors:
        for f in findings:
            f2 = copy.copy(f)
            existing = f2.get("metrics")
            f2["metrics"] = dict(existing) if isinstance(existing, dict) else {}
            f2["metrics"]["diversity_overlap_status"] = "no_anchors"
            f2["metrics"]["diversity_violated"] = False
            f2["metrics"]["diversity_narrow_variation"] = False
            annotated.append(f2)
        return annotated

    for f in findings:
        f2 = copy.copy(f)
        existing = f2.get("metrics")
        f2["metrics"] = dict(existing) if isinstance(existing, dict) else {}

        max_count = -1
        max_record: dict[str, Any] | None = None
        most_similar_label: str | None = None
        for a in anchors:
            ov = compute_dimension_overlap(f2, a)
            if ov is None:
                continue
            if ov["overlap_count"] > max_count:
                max_count = ov["overlap_count"]
                max_record = ov
                most_similar_label = a.get("variant_name") or a.get("finding_id", "?")

        if max_record is None:
            f2["metrics"]["diversity_overlap_status"] = "no_data"
            finding_dimensions = _extract_design_dimensions(f2)
            anchor_dimensions = [
                dimensions
                for anchor in anchors
                if (dimensions := _extract_design_dimensions(anchor))
            ]
            if not finding_dimensions:
                reason = "finding_dimensions_missing"
            elif not anchor_dimensions:
                reason = "anchor_dimensions_missing"
            else:
                reason = "no_common_dimensions"
            f2["metrics"]["diversity_overlap_no_data_reason"] = reason
            f2["metrics"]["diversity_violated"] = False
            f2["metrics"]["diversity_narrow_variation"] = False
        else:
            matches = max_record["overlap_count"]
            total = max_record["total_dims"]
            # Three-tier classification on the OVERLAP_TOTAL (common dims):
            # - clone: matches == total (no differentiation on any axis
            #          where comparison is possible)
            # - narrow: matches > total/2 (matches majority of common dims)
            # - clean: matches <= total/2 (substantive differentiation)
            if matches >= total and total > 0:
                status = "clone"
            elif matches > total // 2:
                status = "narrow"
            else:
                status = "clean"
            f2["metrics"]["diversity_overlap_count"] = matches
            f2["metrics"]["diversity_overlap_total"] = total
            f2["metrics"]["diversity_overlap_fraction"] = max_record["overlap_fraction"]
            f2["metrics"]["diversity_most_similar_anchor"] = most_similar_label
            f2["metrics"]["diversity_overlap_status"] = status
            f2["metrics"]["diversity_violated"] = status == "clone"
            f2["metrics"]["diversity_narrow_variation"] = status == "narrow"
        annotated.append(f2)
    return annotated
