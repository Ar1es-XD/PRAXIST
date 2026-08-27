"""Gems cycle support for periodic diversity-restoring research-loop resets.

The Gems mechanism is opt-in and task configured.  At fixed generation
intervals it snapshots the compact Pareto/lane frontier into durable "Gems",
emits one detailed Gems finding per Gem, archives ordinary active findings, and
starts a new logical generation-0 cycle while preserving the absolute
generation counter on disk.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import math
import os
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
    CANONICAL_STATE,
    COMMITTED,
    PARTIAL,
    artifact_semantics,
    explicit_entry_generation_id,
    is_committed_runtime_fact_file,
    is_committed_runtime_fact_source,
)
from praxist.plugins.workflow_stages.research_loop.backend.effective_config import (
    EFFECTIVE_CONFIG_METADATA_KEYS,
    result_effective_config_metadata,
    strip_effective_config_fields,
)
from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
    compact_maturity_metadata,
    evidence_maturity_snapshot,
    has_explicit_false_completion,
    normalize_maturity_policy,
    protocol_integrity_failed,
    resolve_result_snapshot_producers,
    resolved_fact_bool,
    result_snapshot_key,
    same_result_snapshot,
    task_authorizes_descriptive_maturity,
)
from praxist.plugins.workflow_stages.research_loop.backend.exploration_bottleneck_detector import (
    ExplorationBottleneckDetector,
)
from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
    _existing_materialized_results,
    _infer_result_generation,
    _late_generation_boundary_info,
    _result_summary_metrics,
    canonicalize_evaluation_unit_metadata,
    iter_result_summary_paths,
    normalized_result_summary,
    result_artifact_options_from_task_spec,
    result_summary_control_digest,
    result_summary_variant_name,
)
from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
    _candidate_entity_key,
    _canonical_variant_alias_token,
    _compact_result_identity_container,
)
from praxist.plugins.workflow_stages.research_loop.backend.resume_state import (
    read_boundary_evidence_checkpoint,
)
from praxist.task_spec import (
    GEMS_MATURE_EVIDENCE_TOP_K,
    normalize_gems_selection_policy,
)
from praxist.task_spec_compat import migrate_legacy_gems_entry

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


_GENERIC_PREFERRED_LANES = (
    "confirmed",
    "performance",
    "candidate",
    "task_candidate",
    "reference",
    "diagnostic",
    "process",
)

_GENERIC_PERFORMANCE_LANES = {
    "confirmed",
    "performance",
    "candidate",
    "task_candidate",
    "learned_candidate",
}

_GENERIC_CONTROL_LANES = {
    "reference",
    "diagnostic",
    "negative_control",
}

_GENERIC_GEM_PRIMARY_METRIC_KEYS = (
    "metric_value",
    "lane_metric_value",
    "primary_metric_value",
    "mean_score",
    "score",
    "taskscore",
    "task_score",
    "test_score",
    "eval_score",
)

_GENERIC_GEM_SECONDARY_METRIC_KEYS = (
    "positive_cell_fraction",
    "secondary_score",
)

_GENERIC_GEM_LOWER_TAIL_METRIC_KEYS = (
    "q25_taskscore",
    "q25_score",
    "lower_tail_score",
)

_GENERIC_GEM_VALIDATION_METRIC_KEYS = ("validation_score",)

_GENERIC_GEM_RISK_METRIC_KEYS = (
    "risk_score",
    "max_loss",
)

_RESEARCH_METADATA_KEYS = (
    "bottleneck_target",
    "evidence_stage",
    "tradeoff_class",
    "primary_tradeoff",
    "next_step_intent",
    "parent_candidate",
    "parent_usage",
    "source_lane",
    "target_lane",
    "coverage_check",
    "mechanism_hypothesis_deliverable",
)

_DURABLE_ROUTING_MARKER_KEYS = (
    "validation_only_result",
    "late_after_generation_boundary",
    "artifact_signal_status",
    "late_result_policy",
    "durability_scope",
)

_DIVERSITY_METADATA_KEYS = (
    "diversity_overlap_status",
    "diversity_most_similar_anchor",
    "diversity_overlap_score",
    "diversity_overlap_fraction",
    "diversity_overlap_count",
    "diversity_overlap_total",
    "diversity_violated",
    "diversity_narrow_variation",
    "diversity_violation",
    "mechanism_family",
    "intervention_surface",
    "intent",
    "semantic_family",
    "parent_lineage",
    "novelty_axis",
)


@dataclass
class GemsTriggerResult:
    """Outcome of a Gems plateau check at a generation boundary."""

    triggered: bool
    reason: str = ""
    reset_count: int = 0
    cycle_index: int = 0
    admitted_count: int = 0
    archive_dir: str = ""


def utc_stamp() -> str:
    """Return a compact UTC timestamp suitable for archive directory names."""

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _json_dumps_compact(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def _with_canonical_artifact_semantics(
    data: dict[str, Any],
    *,
    stage: str,
    actor: str,
    canonical_sources: list[str],
    notes: str,
    status: str = COMMITTED,
    runtime_fact_source: bool = True,
) -> dict[str, Any]:
    out = dict(data)
    out["artifact_semantics"] = artifact_semantics(
        role=CANONICAL_STATE,
        status=status,
        stage=stage,
        actor=actor,
        canonical_sources=canonical_sources,
        runtime_fact_source=runtime_fact_source,
        notes=notes,
    )
    return out


def _with_frontier_manifest_semantics(
    data: dict[str, Any], *, gems_state_path: Path | None = None
) -> dict[str, Any]:
    canonical_sources = [
        "results/*",
        "shared_findings/*",
        "shared_store.db",
    ]
    if gems_state_path is not None and is_committed_runtime_fact_file(gems_state_path):
        canonical_sources.append("gems/gems_state.json")
    return _with_canonical_artifact_semantics(
        data,
        stage="frontier_manifest",
        actor="research_loop:gems_manager",
        canonical_sources=canonical_sources,
        notes=(
            "Canonical current frontier/incubator/validation-candidate state "
            "updated during Gems coordination. Derived leaderboards, PI packs, "
            "and prompt snapshots must not override this manifest."
        ),
    )


def _is_clean_runtime_state(data: Any) -> bool:
    return is_committed_runtime_fact_source(data, legacy_ok=True)


def _safe_metric(value: Any) -> float | str | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        if f == float("inf") or f == float("-inf") or f != f:
            return None
        return round(f, 6)
    if value is None:
        return None
    text = str(value).strip()
    return text[:120] if text else None


def _sanitize_nonfinite_json(value: Any) -> Any:
    if isinstance(value, float):
        return _safe_metric(value)
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


def _entry_key(entry: dict[str, Any]) -> str:
    return str(entry.get("finding_id") or entry.get("variant_name") or "")


def _entry_extra(entry: dict[str, Any]) -> dict[str, Any]:
    extra = entry.get("extra")
    if isinstance(extra, dict):
        return extra
    if isinstance(extra, str) and extra.strip():
        try:
            parsed = json.loads(extra)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def _entry_field(entry: dict[str, Any], *names: str) -> Any:
    metrics = _entry_metrics(entry)
    extra = _entry_extra(entry)
    details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
    sources = [entry, metrics, details, extra]
    sources.extend(
        aggregate
        for source in tuple(sources)
        if isinstance((aggregate := source.get("current_aggregate")), dict)
    )
    for name in names:
        for source in sources:
            if name in source and source.get(name) is not None:
                return source.get(name)
    return None


def _gem_identity_token(value: Any) -> str:
    token = str(value or "").strip().lower()
    token = re.sub(r"[^a-z0-9_./=-]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_.-/=")
    return token


def _gem_explicit_entity_key(
    value: Any,
    *,
    candidate: dict[str, Any] | None = None,
) -> str:
    raw = str(value or "").strip()
    if "::" not in raw:
        return ""
    prefix, rest = raw.split("::", 1)
    prefix_token = _gem_identity_token(prefix)
    if prefix_token not in {"variant", "artifact", "finding"}:
        return ""
    rest_token = _gem_identity_token(rest)
    if not rest_token:
        return ""
    if prefix_token == "variant":
        return f"variant:{_canonical_variant_alias_token(rest_token, candidate=candidate) or rest_token}"
    if prefix_token == "artifact":
        _path_token, child_token = _gem_source_result_child_token(rest)
        if child_token:
            return f"variant:{_canonical_variant_alias_token(child_token, candidate=candidate) or child_token}"
    return f"{prefix_token}:{rest_token}"


def _gem_source_result_child_token(value: Any) -> tuple[str, str]:
    raw = str(value or "").strip()
    if not raw:
        return "", ""
    normalized = raw.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part and part not in {".", ".."}]
    child = ""
    for idx, part in enumerate(parts[:-1]):
        if part == "results" and idx + 1 < len(parts):
            child = _gem_identity_token("__".join(parts[idx + 1 : -1]))
            break
    return _gem_identity_token(normalized), child


def _variant_key(entry: dict[str, Any]) -> str:
    """Stable identity for Gems admission.

    Finding ids are evidence ids, not strategy ids. A sweep can emit several
    findings for the same concrete variant, and a single strong variant can
    appear in both cumulative and lane frontiers. Gems capacity is intentionally
    tiny, so admission deduplicates by concrete variant/entity first and falls
    back to finding id only when no variant identity exists. Source result
    artifact paths are more specific than parent sweep names and must preserve
    child arms.
    """

    entity_key = _candidate_entity_key(entry)
    prefix, separator, payload = entity_key.partition("::")
    if separator and prefix in {"variant", "artifact", "finding"}:
        token = (
            str(payload or "").strip().lower()
            if prefix == "finding"
            else _gem_identity_token(payload)
        )
        if token:
            return f"{prefix}:{token}"
    return ""


def _entry_metrics(entry: dict[str, Any]) -> dict[str, Any]:
    metrics = entry.get("metrics")
    admission_metrics = entry.get("admission_metrics")
    details = entry.get("details")
    extra = entry.get("extra")
    merged: dict[str, Any] = {}
    for source in (admission_metrics, details, extra, metrics):
        if isinstance(source, dict):
            merged.update(source)
    return merged


def _metric_float(entry: dict[str, Any], *names: str, default: float = 0.0) -> float:
    value, _name = _metric_float_with_key(entry, *names, default=default)
    return value


def _metric_float_with_key(
    entry: dict[str, Any],
    *names: str,
    default: float = 0.0,
) -> tuple[float, str]:
    metrics = _entry_metrics(entry)
    for name in names:
        value = entry.get(name, metrics.get(name))
        matched_name = name
        if value is None and str(entry.get("metric_name") or "") == name:
            value = entry.get("metric_value")
            matched_name = "metric_value"
        lane_metric_name = entry.get("lane_metric_name", metrics.get("lane_metric_name"))
        if value is None and str(lane_metric_name or "") == name:
            value = entry.get("lane_metric_value", metrics.get("lane_metric_value"))
            matched_name = "lane_metric_value"
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            number = float(value)
        else:
            try:
                number = float(str(value).strip())
            except (TypeError, ValueError):
                continue
        if number == number and number not in {float("inf"), float("-inf")}:
            return number, matched_name
    return float(default), ""


def _metric_float_if_present(entry: dict[str, Any], *names: str) -> float | None:
    metrics = _entry_metrics(entry)
    for name in names:
        value = entry.get(name, metrics.get(name))
        if value is None and str(entry.get("metric_name") or "") == name:
            value = entry.get("metric_value")
        lane_metric_name = entry.get("lane_metric_name", metrics.get("lane_metric_name"))
        if value is None and str(lane_metric_name or "") == name:
            value = entry.get("lane_metric_value", metrics.get("lane_metric_value"))
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            number = float(value)
        else:
            try:
                number = float(str(value).strip())
            except (TypeError, ValueError):
                continue
        if number == number and number not in {float("inf"), float("-inf")}:
            return number
    return None


_SCORE_EVIDENCE_KEYS = (
    "metric_value",
    "lane_metric_value",
    "primary_metric_value",
    "mean_score",
    "score",
    "taskscore",
    "task_score",
    "test_score",
    "eval_score",
    "q25_taskscore",
    "q25_score",
    "positive_cell_fraction",
)


def _entry_has_score_evidence(entry: dict[str, Any]) -> bool:
    metrics = _entry_metrics(entry)
    configured_keys: list[str] = []
    for field_name in (
        "_gems_primary_metric_keys",
        "_gems_secondary_metric_keys",
        "_gems_lower_tail_metric_keys",
        "_gems_validation_metric_keys",
    ):
        for source in (entry, metrics):
            value = source.get(field_name)
            if not isinstance(value, (list, tuple, set)):
                continue
            for item in value:
                text = str(item or "").strip()
                if text:
                    configured_keys.append(text)
    return _metric_float_if_present(entry, *_SCORE_EVIDENCE_KEYS, *configured_keys) is not None


def _entry_failure_evidence_count(entry: dict[str, Any]) -> int:
    metrics = _entry_metrics(entry)
    total = 0
    for name in (
        "failed_units",
        "failed_eval_units",
        "error_units",
        "missing_units",
        "incomplete_units",
        "failed_unit_count",
        "n_failed_units",
        "error_unit_count",
        "n_error_units",
        "missing_unit_count",
        "n_missing_units",
        "incomplete_unit_count",
        "n_incomplete_units",
        "failed_cells",
        "failed_eval_cells",
        "error_cells",
        "missing_cells",
        "incomplete_cells",
        "failed_cell_count",
        "n_failed_cells",
        "error_cell_count",
        "n_error_cells",
        "missing_cell_count",
        "n_missing_cells",
        "incomplete_cell_count",
        "n_incomplete_cells",
    ):
        value = entry.get(name, metrics.get(name))
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (list, dict)):
            total += len(value)
        elif isinstance(value, (int, float)):
            try:
                total += max(0, int(value))
            except (TypeError, ValueError):
                continue
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            try:
                total += max(0, int(float(text)))
            except (TypeError, ValueError):
                total += 1
    return total


def _metric_int(entry: dict[str, Any], *names: str, default: int = 0) -> int:
    metrics = _entry_metrics(entry)
    for name in names:
        value = entry.get(name, metrics.get(name))
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return int(value)
        token = str(value).strip()
        if token.isdigit():
            return int(token)
        try:
            return int(float(token))
        except (TypeError, ValueError):
            continue
    return int(default)


def _entry_field_values(entry: dict[str, Any], *names: str) -> list[Any]:
    containers = [
        entry,
        entry.get("metrics"),
        entry.get("admission_metrics"),
        entry.get("details"),
        entry.get("extra"),
    ]
    containers.extend(
        aggregate
        for source in tuple(containers)
        if isinstance(source, dict)
        and isinstance((aggregate := source.get("current_aggregate")), dict)
    )
    values: list[Any] = []
    for name in names:
        for source in containers:
            if not isinstance(source, dict) or name not in source:
                continue
            value = source.get(name)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            values.append(value)
    return values


def _boolish_entry_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "promotable", "eligible"}:
        return True
    if token in {"0", "false", "no", "n", "non-promotable", "non_promotable", "ineligible"}:
        return False
    return None


def _boolish_entry_field(entry: dict[str, Any], *names: str) -> bool | None:
    for value in _entry_field_values(entry, *names):
        parsed = _boolish_entry_value(value)
        if parsed is not None:
            return parsed
    return None


def _any_boolish_entry_field_true(entry: dict[str, Any], *names: str) -> bool:
    return any(_boolish_entry_value(value) is True for value in _entry_field_values(entry, *names))


def _any_boolish_entry_field_false(entry: dict[str, Any], *names: str) -> bool:
    return any(_boolish_entry_value(value) is False for value in _entry_field_values(entry, *names))


def _infer_generation_id_from_text(text: str) -> int | None:
    matches = re.findall(r"gen[_-]?(\d+)", text or "", flags=re.IGNORECASE)
    if not matches:
        return None
    try:
        return max(int(match) for match in matches)
    except ValueError:
        return None


def _entry_eval_unit_count(entry: dict[str, Any]) -> int:
    return _metric_int(
        entry,
        "completed_required_eval_units",
        "actual_eval_units",
        "evaluation_units",
        "scored_cell_count",
        "n_scored_cells",
        "n_eval_cells",
        "cell_count",
        "n_cells",
        "n_primary_cells",
        default=0,
    )


def _entry_tier_text(entry: dict[str, Any]) -> str:
    metrics = _entry_metrics(entry)
    for name in (
        "evidence_stage",
        "tier",
        "tier_reached",
        "completed_tier",
        "candidate_tier",
    ):
        value = entry.get(name, metrics.get(name))
        text = str(value or "").strip().lower()
        if text:
            return text
    return ""


def _entry_has_explicit_complete_eval_evidence(entry: dict[str, Any]) -> bool:
    """Return True when an entry carries an explicit mature-evaluation marker."""

    if _any_boolish_entry_field_true(
        entry,
        "mature_enough",
        "scored_complete",
        "is_scored_complete",
        "complete_eval",
        "is_complete_eval",
    ):
        return True
    for name in (
        "tier_status",
        "final_status",
        "result_status",
        "completion_status",
        "eval_status",
    ):
        for value in _entry_field_values(entry, name):
            normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
            if not normalized:
                continue
            if normalized in {
                "scored_complete",
                "complete_eval",
                "complete_eval_true",
                "is_complete_eval_true",
                "scored_complete_true",
                "is_scored_complete_true",
            }:
                return True
    return False


def _entry_has_complete_eval_evidence(entry: dict[str, Any]) -> bool:
    if _entry_has_explicit_complete_eval_evidence(entry):
        return True
    return _entry_is_legacy_persisted_gem(entry)


def _entry_is_recoverable_legacy_or_control_gem_source(entry: dict[str, Any]) -> bool:
    """Allow old Gem/control anchors to finish an already-pending reset."""

    if _entry_is_legacy_persisted_gem(entry):
        return True
    if (
        isinstance(entry.get("admission_metrics"), dict)
        and _entry_has_score_evidence(entry)
        and any(
            "legacy" in str(value or "").strip().lower()
            for value in (entry.get("gem_finding_id"), entry.get("variant_name"))
        )
    ):
        return True
    lane = _entry_lane(entry)
    return bool(lane and lane in _GENERIC_CONTROL_LANES)


def _entry_is_legacy_persisted_gem(entry: dict[str, Any]) -> bool:
    """Return True for older Gems state rows with no source-gen provenance.

    Legacy persisted Gems may only have a ``gem_finding_id`` plus frozen
    ``admission_metrics``. They are safe to use while finishing an already
    pending reset transaction, but should not be admitted to prompt context
    through cutoff-sensitive loaders without real source-generation provenance.
    """

    return bool(
        entry.get("gem_finding_id")
        and _entry_source_generation_id(entry) is None
        and isinstance(entry.get("admission_metrics"), dict)
        and _entry_eval_unit_count(entry) == 0
        and _entry_has_score_evidence(entry)
    )


def _entry_status_texts(entry: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for name in (
        "evidence_stage",
        "tier_status",
        "final_status",
        "result_status",
        "completion_status",
        "status",
        "eval_status",
        "source_result_kind",
    ):
        for value in _entry_field_values(entry, name):
            text = str(value or "").strip().lower()
            if text:
                texts.append(text)
                normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
                if normalized != text:
                    texts.append(normalized)
    return texts


_PARTIAL_STATUS_EXACT_MARKERS = {
    "cancel",
    "cancelled",
    "canceled",
    "failed",
    "failure",
    "failed_or_unscored",
    "incomplete",
    "interrupt",
    "interrupted",
    "error",
    "exception",
    "crashed",
    "killed",
    "oom",
    "partial",
    "partial_cohort",
    "partial_eval",
    "pending",
    "running",
    "stale",
    "timeout",
    "smoke",
    "scout",
    "cheap_probe",
    "summary_only",
    "summary_only_true",
    "unscored",
    "un_scored",
    "unscored_artifact",
    "not_scored",
    "not_scored_complete",
    "preliminary",
    "prelim",
    "capped",
    "result_capped",
    "capped_at",
    "cap_at",
    "scored_complete_false",
    "scored_complete_0",
    "is_scored_complete_false",
    "is_scored_complete_0",
    "complete_eval_false",
    "complete_eval_0",
    "is_complete_eval_false",
    "is_complete_eval_0",
    "protocol_invalid",
}

_PARTIAL_STATUS_TOKEN_MARKERS = {
    "cancel",
    "cancelled",
    "canceled",
    "failed",
    "failure",
    "incomplete",
    "interrupt",
    "interrupted",
    "error",
    "exception",
    "crashed",
    "killed",
    "oom",
    "partial",
    "pending",
    "running",
    "stale",
    "timeout",
    "smoke",
    "scout",
    "unscored",
    "preliminary",
    "prelim",
}


def _entry_status_is_scout_or_partial(value: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    if not normalized:
        return False
    if normalized in _PARTIAL_STATUS_EXACT_MARKERS:
        return True
    if normalized.startswith("capped_at_") or normalized.startswith("cap_at_"):
        return True
    tokens = [token for token in normalized.split("_") if token]
    return _status_tokens_have_marker(tokens, _PARTIAL_STATUS_TOKEN_MARKERS)


def _status_tokens_have_marker(tokens: list[str], markers: set[str]) -> bool:
    ordered = [token for token in tokens if token]
    negators = {"not", "non", "no", "without"}
    for idx, token in enumerate(ordered):
        if token not in markers:
            continue
        if any(previous in negators for previous in ordered[max(0, idx - 3) : idx]):
            continue
        return True
    return False


def _entry_has_explicit_scout_or_partial_marker(entry: dict[str, Any]) -> bool:
    metrics = _entry_metrics(entry)
    if protocol_integrity_failed(entry):
        return True
    if _any_boolish_entry_field_true(
        entry,
        "suspect_protocol",
        # Legacy input alias only; Gems emits the generic suspect_protocol key.
        "suspect_fixed_weight_eval",
    ):
        return True
    if _any_boolish_entry_field_true(entry, "excluded_from_durable_frontier"):
        return True
    if (
        str(entry.get("exclusion_reason") or metrics.get("exclusion_reason") or "").strip().lower()
        == "preliminary_or_incomplete_evidence"
    ):
        return True
    if _entry_failure_evidence_count(entry) > 0:
        return True
    if _any_boolish_entry_field_true(
        entry,
        "scout_only",
        "is_scout_eval",
        "is_smoke_eval",
        "summary_only",
        "is_summary_only",
        "partial_cohort",
        "partial_eval",
        "is_partial_eval",
        "unscored_artifact",
        "incomplete_eval",
        "is_incomplete_eval",
        "capped",
        "result_capped",
        "is_capped",
    ):
        return True
    if has_explicit_false_completion(entry):
        return True
    for name in (
        "evidence_stage",
        "tier",
        "tier_reached",
        "completed_tier",
        "candidate_tier",
        "tier_status",
        "final_status",
        "result_status",
        "completion_status",
        "status",
        "eval_status",
    ):
        text = str(entry.get(name) or metrics.get(name) or "").strip().lower()
        if _entry_status_is_scout_or_partial(text):
            return True
    return any(_entry_status_is_scout_or_partial(text) for text in _entry_status_texts(entry))


def _entry_is_scout_or_partial(entry: dict[str, Any]) -> bool:
    if not _entry_has_score_evidence(entry):
        return True
    return _entry_has_explicit_scout_or_partial_marker(entry)


def _entry_hard_constraint_violation_count(entry: dict[str, Any]) -> int:
    total = 0
    for value in _entry_field_values(
        entry,
        "n_hard_constraint_violations",
        "hard_constraint_violations",
        "n_constraint_violations",
        "constraint_violations",
    ):
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (list, tuple, set, dict)):
            total += len(value)
            continue
        if isinstance(value, (int, float)):
            total += max(0, int(value))
            continue
        text = str(value).strip().lower()
        if not text or text in {"0", "none", "no", "false", "[]", "{}"}:
            continue
        try:
            total += max(0, int(float(text)))
        except (TypeError, ValueError):
            total += 1
    return total


def _entry_has_nonclean_gem_marker(entry: dict[str, Any]) -> bool:
    if resolved_fact_bool(entry, "promotion_eligible", "clean_promotion_eligible") is False:
        return True
    return _entry_hard_constraint_violation_count(entry) > 0


def _entry_has_explicit_gem_rejection_marker(entry: dict[str, Any]) -> bool:
    return resolved_fact_bool(
        entry, "promotion_eligible"
    ) is False or _entry_has_explicit_scout_or_partial_marker(entry)


def _entry_has_validation_only_durability_marker(entry: dict[str, Any]) -> bool:
    if (
        resolved_fact_bool(
            entry,
            "late_after_generation_boundary",
            "validation_only",
            "validation_only_result",
        )
        is True
    ):
        return True
    for name in ("artifact_signal_status", "late_result_policy", "durability_scope"):
        for value in _entry_field_values(entry, name):
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


def _durable_routing_marker_value(
    summary: dict[str, Any],
    aggregate: dict[str, Any],
    key: str,
) -> Any:
    """Return explicit durable-routing markers without false aggregate defaults hiding them."""

    for value in (summary.get(key), aggregate.get(key)):
        if value in (None, "", [], {}):
            continue
        if isinstance(value, bool):
            if value:
                return True
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value != 0:
                return value
            continue
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"0", "false", "no", "n", "off"}:
                continue
            return value.strip()
        return value
    return None


def _entry_has_gem_integrity_rejection_marker(entry: dict[str, Any]) -> bool:
    """Reject facts that cannot be made durable by task protocol choice.

    A task may deliberately make a reduced, partial, or scout stage mature.
    Those descriptive stage markers are handled separately. Runtime failure,
    protocol failure, explicit non-promotion, and validation-only routing remain
    authoritative regardless of how the task names its accepted protocol.
    """

    if protocol_integrity_failed(entry):
        return True
    if resolved_fact_bool(entry, "promotion_eligible") is False:
        return True
    if _entry_has_validation_only_durability_marker(entry):
        return True
    if _any_boolish_entry_field_true(
        entry,
        "suspect_protocol",
        # Legacy input alias only; Gems emits the generic suspect_protocol key.
        "suspect_fixed_weight_eval",
        "excluded_from_durable_frontier",
    ):
        return True
    if _entry_failure_evidence_count(entry) > 0:
        return True
    if has_explicit_false_completion(entry):
        return True
    if _any_boolish_entry_field_true(
        entry,
        "summary_only",
        "is_summary_only",
        "unscored_artifact",
        "incomplete_eval",
        "is_incomplete_eval",
    ):
        return True
    for name in (
        "tier_status",
        "final_status",
        "result_status",
        "completion_status",
        "status",
        "eval_status",
        "source_result_kind",
    ):
        for value in _entry_field_values(entry, name):
            normalized = re.sub(
                r"[^a-z0-9]+",
                "_",
                str(value or "").strip().lower(),
            ).strip("_")
            tokens = [token for token in normalized.split("_") if token]
            if _status_tokens_have_marker(
                tokens,
                {
                    "cancel",
                    "cancelled",
                    "canceled",
                    "failed",
                    "failure",
                    "incomplete",
                    "interrupt",
                    "interrupted",
                    "error",
                    "exception",
                    "crashed",
                    "killed",
                    "oom",
                    "pending",
                    "running",
                    "stale",
                    "timeout",
                    "unscored",
                    "invalid",
                },
            ):
                return True
    return False


def _entry_has_hard_gem_rejection_marker(entry: dict[str, Any]) -> bool:
    if _entry_has_gem_integrity_rejection_marker(entry):
        return True
    if _any_boolish_entry_field_true(
        entry,
        "summary_only",
        "is_summary_only",
        "partial_cohort",
        "partial_eval",
        "is_partial_eval",
        "unscored_artifact",
        "incomplete_eval",
        "is_incomplete_eval",
        "capped",
        "result_capped",
        "is_capped",
        "validation_only",
    ):
        return True
    status_text = " ".join(_entry_status_texts(entry))
    return any(
        token in status_text
        for token in (
            "failed",
            "crashed",
            "incomplete",
            "running",
            "invalid",
            "error",
            "summary_only",
            "unscored",
            "not_scored",
            "partial",
            "capped",
            "protocol_invalid",
        )
    )


def _entry_has_generic_mature_evidence(
    entry: dict[str, Any],
    maturity_policy: Any | None = None,
) -> bool:
    if not _entry_has_score_evidence(entry):
        return False
    if _entry_has_gem_integrity_rejection_marker(entry) or _entry_has_nonclean_gem_marker(entry):
        return False
    maturity_flag = _boolish_entry_field(entry, "mature_enough")
    basis_values = _entry_field_values(entry, "maturity_basis")
    basis = str(basis_values[0]).strip() if basis_values else ""
    maturity = evidence_maturity_snapshot(entry, maturity_policy)
    configured_stage_complete = bool(
        maturity.get("mature_enough") is True
        and task_authorizes_descriptive_maturity(
            entry,
            maturity_policy,
            maturity=maturity,
        )
    )
    if configured_stage_complete:
        return True
    if _entry_has_hard_gem_rejection_marker(entry):
        return False
    if maturity_policy is None and basis == "effort_coverage_ratio" and maturity_flag is not None:
        return bool(maturity_flag)
    decision = maturity.get("mature_enough")
    if decision is True:
        return True
    if decision is False:
        return False
    if _entry_has_hard_gem_rejection_marker(entry):
        return False
    if normalize_maturity_policy(maturity_policy).get("require_ratio_gate"):
        return False
    if _entry_has_explicit_gem_rejection_marker(entry):
        return False
    if maturity_flag is True:
        return True
    if maturity_flag is False and basis == "effort_coverage_ratio":
        return False
    return False


def _entry_is_clean_gem_admission_candidate(
    entry: dict[str, Any],
    maturity_policy: Any | None = None,
) -> bool:
    if _entry_has_generic_mature_evidence(entry, maturity_policy):
        return True
    if evidence_maturity_snapshot(entry, maturity_policy).get("mature_enough") is False:
        return False
    if normalize_maturity_policy(maturity_policy).get("require_ratio_gate"):
        return False
    return (
        not _entry_is_scout_or_partial(entry)
        and _entry_has_complete_eval_evidence(entry)
        and not _entry_has_hard_gem_rejection_marker(entry)
        and not _entry_has_nonclean_gem_marker(entry)
    )


def _entry_is_mature_gem_admission_candidate(
    entry: dict[str, Any],
    *,
    min_mature_eval_units: int,
    evidence_stage_min_units: dict[str, int] | None = None,
    maturity_policy: Any | None = None,
) -> bool:
    maturity = evidence_maturity_snapshot(entry, maturity_policy)
    if maturity.get("maturity_basis") == "effort_coverage_ratio":
        task_authorized_mode = task_authorizes_descriptive_maturity(
            entry,
            maturity_policy,
            maturity=maturity,
        )
        return bool(maturity.get("mature_enough")) and not (
            _entry_has_gem_integrity_rejection_marker(entry)
            or _entry_has_nonclean_gem_marker(entry)
            or (not task_authorized_mode and _entry_has_hard_gem_rejection_marker(entry))
        )
    configured_stage_complete = (
        maturity.get("maturity_basis") == "task_configured_stage"
        and maturity.get("mature_enough") is True
    )
    configured_mode_authorizes_descriptors = maturity.get(
        "mature_enough"
    ) is True and task_authorizes_descriptive_maturity(
        entry,
        maturity_policy,
        maturity=maturity,
    )
    if (
        maturity.get("maturity_basis") == "task_configured_stage"
        and maturity.get("mature_enough") is False
    ):
        return False
    if normalize_maturity_policy(maturity_policy).get("require_ratio_gate"):
        return False
    explicit_promotion_rejection = resolved_fact_bool(entry, "promotion_eligible") is False
    return (
        not _entry_has_gem_integrity_rejection_marker(entry)
        and (
            configured_mode_authorizes_descriptors
            or not _entry_has_hard_gem_rejection_marker(entry)
        )
        and not explicit_promotion_rejection
        and (
            configured_mode_authorizes_descriptors
            or not _entry_has_explicit_scout_or_partial_marker(entry)
        )
        and (configured_stage_complete or _entry_has_explicit_complete_eval_evidence(entry))
        and _is_mature_evaluation_or_better(
            entry,
            min_mature_eval_units=min_mature_eval_units,
            evidence_stage_min_units=evidence_stage_min_units,
            skip_performance_check=True,
            maturity_policy=maturity_policy,
        )
    )


def _entry_has_low_confidence_generation(entry: dict[str, Any]) -> bool:
    return _any_boolish_entry_field_true(entry, "source_generation_low_confidence")


def _entry_source_generation_id(entry: dict[str, Any]) -> int | None:
    metrics = _entry_metrics(entry)
    generations: list[int] = []
    for name in ("source_generation_id", "generation_id", "gen_id"):
        value = entry.get(name, metrics.get(name))
        if value is None:
            continue
        try:
            generations.append(int(value))
        except (TypeError, ValueError):
            continue
    for name in ("variant_name", "finding_id", "gem_finding_id", "source_finding_id"):
        value = str(entry.get(name) or metrics.get(name) or "").strip()
        gen = _infer_generation_id_from_text(value)
        if gen is not None:
            generations.append(gen)
    return max(generations) if generations else None


def _gem_sidecar_source_generation_id(
    sidecar: dict[str, Any],
    *,
    allow_identity_inference: bool = True,
) -> int | None:
    """Resolve the source generation from a persisted Gem sidecar.

    Gem findings intentionally store top-level ``generation_id=0`` so they
    survive logical generation resets. For cutoff-sensitive prompt loading, the
    real provenance is the nested frontier/source entry or an explicit
    source-generation field, not that logical visibility generation.
    """

    metrics = _entry_metrics(sidecar)
    generations: list[int] = []
    for source in (sidecar, metrics):
        if not isinstance(source, dict):
            continue
        for name in (
            "source_generation_id",
            "source_gen_id",
            "completed_generation",
            "completed_gen_id",
        ):
            value = source.get(name)
            if value is None:
                continue
            try:
                generations.append(int(value))
            except (TypeError, ValueError):
                continue
    for source in (sidecar, metrics):
        if not isinstance(source, dict):
            continue
        for name in (
            "source_frontier_entry",
            "source_entry",
            "frontier_entry",
            "source_result",
        ):
            nested = source.get(name)
            if isinstance(nested, dict):
                generation = (
                    _entry_source_generation_id(nested)
                    if allow_identity_inference
                    else explicit_entry_generation_id(nested)
                )
                if generation is not None:
                    generations.append(generation)
    if generations:
        return max(generations)
    return None


def _resolved_persisted_gem_source_generation_id(
    run_dir: Path | None,
    entry: dict[str, Any],
    *,
    allow_identity_inference: bool = True,
) -> int | None:
    """Resolve persisted Gem provenance without trusting reset visibility ids."""

    metrics = _entry_metrics(entry)
    explicit_generations: list[int] = []
    explicit_source_generations: list[int] = []
    for name in ("source_generation_id", "generation_id", "gen_id"):
        value = entry.get(name, metrics.get(name))
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        explicit_generations.append(parsed)
        if name == "source_generation_id":
            explicit_source_generations.append(parsed)

    finding_path = str(entry.get("finding_path") or "").strip()
    candidates: list[Path] = []
    if finding_path and run_dir is not None:
        path = Path(run_dir) / finding_path
        if path.is_file():
            candidates.append(path)

    gem_ids = {
        str(entry.get(name) or "").strip()
        for name in ("gem_finding_id", "source_finding_id", "id")
        if str(entry.get(name) or "").strip()
    }
    has_gem_sidecar_identity = bool(finding_path or gem_ids)
    if explicit_source_generations and has_gem_sidecar_identity:
        return max(explicit_source_generations)
    if explicit_generations and not has_gem_sidecar_identity:
        inferred = _entry_source_generation_id(entry) if allow_identity_inference else None
        return max(
            [*explicit_generations, inferred] if inferred is not None else explicit_generations
        )
    if gem_ids and run_dir is not None:
        shared = Path(run_dir) / "shared_findings"
        with contextlib.suppress(OSError):
            candidates.extend(path for path in shared.glob("*.json") if path.is_file())

    generations: list[int] = []
    seen_paths: set[Path] = set()
    for path in candidates:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if gem_ids and str(data.get("id") or "").strip() not in gem_ids:
            continue
        source = _gem_sidecar_source_generation_id(
            data,
            allow_identity_inference=allow_identity_inference,
        )
        if source is not None:
            generations.append(source)

    if generations:
        return max(generations)
    if has_gem_sidecar_identity:
        return None
    if allow_identity_inference:
        return _entry_source_generation_id(entry)
    return explicit_entry_generation_id(entry)


def _bottleneck_report_generation(report: Any) -> int | None:
    if not isinstance(report, dict):
        return None
    for key in ("completed_generation", "generation_id", "completed_gen_id"):
        value = report.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _persisted_gems_source_generation_limit(state: dict[str, Any]) -> int | None:
    """Recover the latest possible Gem source generation from reset state."""

    for key in ("gems_source_generation_limit", "source_generation_limit"):
        try:
            value = state.get(key)
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    try:
        cycle_start_generation = int(state.get("cycle_start_generation", 0))
    except (TypeError, ValueError):
        return None
    if cycle_start_generation <= 0:
        return None
    reset_events = state.get("reset_events")
    if isinstance(reset_events, list):
        for event in reversed(reset_events):
            if not isinstance(event, dict):
                continue
            next_absolute_raw = event.get("next_absolute_generation")
            completed_raw = event.get("completed_gen_id")
            if next_absolute_raw is None or completed_raw is None:
                continue
            try:
                next_absolute = int(next_absolute_raw)
                completed = int(completed_raw)
            except (TypeError, ValueError):
                continue
            if next_absolute == cycle_start_generation:
                return completed
    return cycle_start_generation - 1


def _filter_bottleneck_reports_for_generation(
    reports: Any,
    max_generation_id: int | None,
) -> list[dict[str, Any]]:
    raw_reports = list(reports or []) if isinstance(reports, list) else []
    if max_generation_id is None:
        return [report for report in raw_reports if isinstance(report, dict)]
    filtered: list[dict[str, Any]] = []
    for report in raw_reports:
        generation = _bottleneck_report_generation(report)
        if generation is not None and generation <= int(max_generation_id):
            filtered.append(report)
    return filtered


def _latest_soft_priors_for_generation(
    reports: Any,
    max_generation_id: int | None,
    fallback: Any,
) -> dict[str, Any]:
    if max_generation_id is None:
        return fallback if isinstance(fallback, dict) else {}
    raw_reports = list(reports or []) if isinstance(reports, list) else []
    if not raw_reports:
        return {}
    filtered = _filter_bottleneck_reports_for_generation(reports, max_generation_id)
    for report in reversed(filtered):
        priors = report.get("soft_agenda_priors")
        if isinstance(priors, dict):
            return priors
    return {}


def _state_bottleneck_reports(state: dict[str, Any]) -> list[dict[str, Any]]:
    active = state.get("active_bottleneck_reports", [])
    if isinstance(active, list) and active:
        return [report for report in active if isinstance(report, dict)]
    if not state.get("latest_soft_agenda_priors"):
        return []
    history = state.get("bottleneck_history", [])
    if isinstance(history, list):
        return [report for report in history if isinstance(report, dict)]
    return []


def _is_mature_evaluation_or_better(
    entry: dict[str, Any],
    *,
    min_mature_eval_units: int,
    evidence_stage_min_units: dict[str, int] | None = None,
    skip_performance_check: bool = False,
    maturity_policy: Any | None = None,
) -> bool:
    """Return True only for actually scored complete-evaluation candidates."""

    maturity = evidence_maturity_snapshot(entry, maturity_policy)
    configured_stage_complete = (
        maturity.get("maturity_basis") == "task_configured_stage"
        and maturity.get("mature_enough") is True
    )
    configured_mode_authorizes_descriptors = maturity.get(
        "mature_enough"
    ) is True and task_authorizes_descriptive_maturity(
        entry,
        maturity_policy,
        maturity=maturity,
    )
    if not configured_mode_authorizes_descriptors and _entry_is_scout_or_partial(entry):
        return False
    if not skip_performance_check and not _is_performance_entry(entry):
        return False
    units = _entry_eval_unit_count(entry)
    if units < int(min_mature_eval_units):
        return False
    stage = re.sub(r"[^a-z0-9]+", "_", _entry_tier_text(entry)).strip("_")
    stage_thresholds = {
        re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_"): max(1, int(value))
        for key, value in (evidence_stage_min_units or {}).items()
        if str(key).strip()
    }
    if stage and stage in stage_thresholds:
        return units >= stage_thresholds[stage]
    if configured_stage_complete:
        return True
    return _entry_has_complete_eval_evidence(entry)


def _evidence_rank(entry: dict[str, Any]) -> int:
    recorded_rank = _metric_float(entry, "evidence_rank", default=-1.0)
    if recorded_rank >= 0:
        return int(recorded_rank)
    if _any_boolish_entry_field_true(entry, "mature_enough"):
        return 2
    if _entry_has_explicit_complete_eval_evidence(entry):
        return 2
    if _entry_has_explicit_scout_or_partial_marker(entry):
        return 1
    return 0


def _entry_metric_key_list(
    entry: dict[str, Any],
    field_name: str,
    fallback: tuple[str, ...],
) -> tuple[str, ...]:
    metrics = _entry_metrics(entry)
    raw = entry.get(field_name, metrics.get(field_name))
    if raw is None:
        return fallback
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
    return tuple(out) or fallback


def _entry_has_metric_key(entry: dict[str, Any], key: str) -> bool:
    if key in entry:
        return True
    for source_name in ("metrics", "admission_metrics", "details", "extra", "final_metrics"):
        source = entry.get(source_name)
        if isinstance(source, dict) and key in source:
            return True
    final_metrics = entry.get("final_metrics")
    if isinstance(final_metrics, dict):
        aggregated = final_metrics.get("aggregated")
        if isinstance(aggregated, dict) and key in aggregated:
            return True
    return False


def _entry_metric_direction(entry: dict[str, Any], metric_key: str = "") -> str:
    metrics = _entry_metrics(entry)
    if metric_key:
        keyed_direction = (
            str(
                entry.get(f"{metric_key}_direction") or metrics.get(f"{metric_key}_direction") or ""
            )
            .strip()
            .lower()
        )
        if keyed_direction in {"maximize", "minimize"}:
            return keyed_direction
    if metric_key == "lane_metric_value":
        lane_direction = (
            str(entry.get("lane_metric_direction") or metrics.get("lane_metric_direction") or "")
            .strip()
            .lower()
        )
        if lane_direction in {"maximize", "minimize"}:
            return lane_direction
    if metric_key == "metric_value":
        metric_direction = (
            str(entry.get("metric_direction") or metrics.get("metric_direction") or "")
            .strip()
            .lower()
        )
        if metric_direction in {"maximize", "minimize"}:
            return metric_direction
    direction = (
        str(
            metrics.get("_gems_metric_direction")
            or entry.get("_gems_metric_direction")
            or entry.get("metric_direction")
            or metrics.get("metric_direction")
            or "maximize"
        )
        .strip()
        .lower()
    )
    return "minimize" if direction == "minimize" else "maximize"


def _gem_performance_key(entry: dict[str, Any]) -> tuple[float, float, float, float, int, float]:
    """Sort key for durable Gem admission.

    Gems are restart parents, so admission should not blindly prefer evidence
    maturity over clearly stronger task performance. The default key is
    task-generic: it first looks for the frontier metric value / task score,
    then optional secondary and lower-tail scores. Domain-specific metrics can
    still participate when a task emits them, but they are not core defaults.
    """

    primary_score, primary_key = _metric_float_with_key(
        entry,
        *_entry_metric_key_list(
            entry,
            "_gems_primary_metric_keys",
            _GENERIC_GEM_PRIMARY_METRIC_KEYS,
        ),
    )
    secondary_score = _metric_float(
        entry,
        *_entry_metric_key_list(
            entry,
            "_gems_secondary_metric_keys",
            _GENERIC_GEM_SECONDARY_METRIC_KEYS,
        ),
    )
    lower_tail_score = _metric_float(
        entry,
        *_entry_metric_key_list(
            entry,
            "_gems_lower_tail_metric_keys",
            _GENERIC_GEM_LOWER_TAIL_METRIC_KEYS,
        ),
    )
    validation_score = _metric_float(
        entry,
        *_entry_metric_key_list(
            entry,
            "_gems_validation_metric_keys",
            _GENERIC_GEM_VALIDATION_METRIC_KEYS,
        ),
    )
    risk = _metric_float(
        entry,
        *_entry_metric_key_list(
            entry,
            "_gems_cost_metric_keys",
            _GENERIC_GEM_RISK_METRIC_KEYS,
        ),
        default=10_000.0,
    )
    if not primary_key:
        primary_score = float("-inf")
    elif _entry_metric_direction(entry, primary_key) == "minimize":
        primary_score = -primary_score
    return (
        primary_score,
        secondary_score,
        lower_tail_score,
        validation_score,
        _evidence_rank(entry),
        -risk,
    )


def _performance_lane_priority(lane: str) -> int:
    if lane == "confirmed":
        return 2
    if lane in {"performance", "candidate"}:
        return 1
    return 0


def _is_performance_entry(entry: dict[str, Any]) -> bool:
    metrics = _entry_metrics(entry)
    strategy = (
        str(entry.get("strategy_family") or metrics.get("strategy_family") or "").strip().lower()
    )
    lane = _entry_lane(entry)
    if lane:
        return lane in _GENERIC_PERFORMANCE_LANES
    if strategy not in {"learned_candidate", "candidate", "task_candidate"}:
        return False
    return (
        _metric_float(
            entry,
            "mean_score",
            "score",
            "metric_value",
            "lane_metric_value",
        )
        > 0
    )


def _entry_lane(entry: dict[str, Any]) -> str:
    metrics = _entry_metrics(entry)
    return str(
        entry.get("promoted_for_lane")
        or entry.get("frontier_lane")
        or metrics.get("frontier_lane")
        or ""
    ).strip()


def _entry_family(entry: dict[str, Any]) -> str:
    metrics = _entry_metrics(entry)
    generic = {
        "learned_candidate",
        "task_candidate",
        "candidate",
        "unknown",
        "none",
        "null",
        "n/a",
    }
    for name in (
        "mechanism_family",
        "family",
        "innovation_surface",
        "strategy_family",
    ):
        value = entry.get(name, metrics.get(name))
        text = str(value or "").strip().lower()
        if text and text not in generic:
            return text
    strategy = (
        str(entry.get("strategy_family") or metrics.get("strategy_family") or "").strip().lower()
    )
    if strategy and strategy not in generic:
        return strategy
    return ""


class GemsManager:
    """Manage opt-in Gems snapshots and logical research-cycle resets."""

    def __init__(self, *, run_dir: Path, task_spec: Any, frontier: Any, local_mode: bool = True):
        self.run_dir = Path(run_dir)
        self.task_spec = task_spec
        self.frontier = frontier
        self.local_mode = bool(local_mode)
        self.config = getattr(task_spec, "gems", None)
        self.maturity_policy = getattr(
            getattr(task_spec, "evaluation", None),
            "maturity_policy",
            None,
        )
        self.gems_dir = self.run_dir / "gems"
        self.state_path = self.gems_dir / "gems_state.json"
        self.findings_dir = self.run_dir / "shared_findings"

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.config, "enabled", False))

    @property
    def cycle_start_generation(self) -> int:
        state = self.load_state()
        return int(state.get("cycle_start_generation", 0))

    @property
    def cycle_index(self) -> int:
        state = self.load_state()
        return int(state.get("cycle_index", 0))

    def logical_generation(self, absolute_gen_id: int) -> int:
        if not self.enabled:
            return absolute_gen_id
        return max(0, int(absolute_gen_id) - self.cycle_start_generation)

    def load_state(self) -> dict[str, Any]:
        state_path = self._safe_state_path()
        if state_path.exists():
            try:
                data = json.loads(state_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("gems: could not read state %s: %s", state_path, exc)
        return {
            "enabled": self.enabled,
            "cycle_index": 0,
            "cycle_start_generation": 0,
            "reset_count": 0,
            "max_resets": int(getattr(self.config, "max_resets", 0) or 0),
            "last_signature_hash": "",
            "signature_history": [],
            "bottleneck_history": [],
            "active_bottleneck_reports": [],
            "latest_soft_agenda_priors": {},
            "gems": [],
            "reset_events": [],
        }

    def save_state(self, state: dict[str, Any]) -> None:
        state["enabled"] = self.enabled
        state["max_resets"] = int(getattr(self.config, "max_resets", 0) or 0)
        pending_reset = isinstance(state.get("pending_reset"), dict)
        state = _with_canonical_artifact_semantics(
            state,
            stage="gems_state",
            actor="research_loop:gems_manager",
            canonical_sources=[
                "frontier/frontier_manifest.json",
                "results/*",
                "shared_findings/*",
                "shared_store.db",
            ],
            notes=(
                "Canonical current Gems reset seed state. Gems digests, prompts, "
                "and historical reset snapshots are derived/audit views."
                if not pending_reset
                else "Partial Gems reset state with a pending reset transaction. "
                "Resume/control must complete or repair the pending reset before "
                "treating Gems as a clean runtime fact source."
            ),
            status=PARTIAL if pending_reset else COMMITTED,
            runtime_fact_source=not pending_reset,
        )
        state_path = self._safe_state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(state_path, state)

    def prompt_context(
        self,
        absolute_gen_id: int,
        *,
        peer_index: int | None = None,
        cohort_size: int | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        state = self.load_state()
        if not _is_clean_runtime_state(state):
            logger.warning(
                "gems: ignoring non-committed runtime Gems state for prompt context: %s",
                self.state_path,
            )
            return {
                "cycle_index": state.get("cycle_index", 0),
                "reset_count": state.get("reset_count", 0),
                "cycle_start_generation": state.get("cycle_start_generation", 0),
                "entries": [],
                "bottleneck_reports": [],
                "latest_soft_agenda_priors": {},
                "warning": "gems_state_not_committed_runtime_fact_source",
            }
        evidence_cutoff = int(absolute_gen_id) - 1
        pruned_restart_cutoff = self._operator_pruned_restart_cutoff(
            state,
            int(absolute_gen_id),
        )
        if pruned_restart_cutoff is not None:
            evidence_cutoff = max(evidence_cutoff, pruned_restart_cutoff)
        gems = self.active_gems_from_state(state, max_generation_id=evidence_cutoff)
        logical_gen = self.logical_generation(absolute_gen_id)
        reset_count = int(state.get("reset_count", 0))
        gem_seeded_baseline_mode = bool(gems and reset_count > 0 and logical_gen == 0)
        primary_gem_anchor: dict[str, Any] = {}
        secondary_gem_anchor: dict[str, Any] = {}
        gem_anchor_roster: list[dict[str, Any]] = []
        gem_anchor_assignment_mode = ""
        independent_exploration_slots = 0
        gem_inheritance_slots = 0
        if gem_seeded_baseline_mode and gems:
            n_gems = len(gems)
            n_peers_for_assignment = 0
            if cohort_size is not None:
                try:
                    n_peers_for_assignment = max(0, int(cohort_size))
                except (TypeError, ValueError):
                    n_peers_for_assignment = 0
            configured_independent = max(
                0, int(getattr(self.config, "gem_seeded_independent_peers", 0) or 0)
            )
            minimum_independent = (
                1 if configured_independent > 0 and n_peers_for_assignment > 1 else 0
            )
            independent_exploration_slots = min(
                configured_independent,
                max(minimum_independent, n_peers_for_assignment - n_gems),
            )
            gem_inheritance_slots = max(0, n_peers_for_assignment - independent_exploration_slots)
            if peer_index is not None:
                try:
                    peer_idx = int(peer_index)
                except (TypeError, ValueError):
                    peer_idx = 0
                if n_peers_for_assignment and peer_idx >= gem_inheritance_slots:
                    gem_anchor_assignment_mode = "independent_exploration_or_recombination"
                    if n_gems:
                        # Independent peers still see Gems and may recombine
                        # insights, but they are not assigned one Gem as the
                        # implementation parent.
                        secondary_gem_anchor = dict(gems[peer_idx % n_gems])
                else:
                    gem_anchor_assignment_mode = "gem_inheritance"
                    idx = peer_idx % n_gems
                    primary_gem_anchor = dict(gems[idx])
                    if n_gems > 1:
                        secondary_gem_anchor = dict(gems[(idx + 1) % n_gems])
            if cohort_size is not None:
                try:
                    n_peers = max(0, int(cohort_size))
                except (TypeError, ValueError):
                    n_peers = 0
                for i in range(n_peers):
                    if i >= gem_inheritance_slots:
                        reference = gems[i % n_gems]
                        gem_anchor_roster.append(
                            {
                                "peer_index": i,
                                "peer_id": f"gen{absolute_gen_id}_peer{i}",
                                "assignment_type": "independent_exploration_or_recombination",
                                "primary_gem_index": 0,
                                "primary_variant_name": "",
                                "primary_gem_finding_id": "",
                                "secondary_variant_name": str(reference.get("variant_name") or ""),
                                "instruction": (
                                    "Do not inherit a single Gem as code parent. "
                                    "This is a protected independent slot, not "
                                    "leftover capacity: use Gems as context, "
                                    "recombine knowledge across Gems, falsify a "
                                    "Gem claim, or explore a distinct mechanism."
                                ),
                            }
                        )
                        continue
                    primary = gems[i % n_gems]
                    secondary = gems[(i + 1) % n_gems] if n_gems > 1 else {}
                    gem_anchor_roster.append(
                        {
                            "peer_index": i,
                            "peer_id": f"gen{absolute_gen_id}_peer{i}",
                            "assignment_type": "gem_inheritance",
                            "primary_gem_index": (i % n_gems) + 1,
                            "primary_variant_name": str(primary.get("variant_name") or ""),
                            "primary_gem_finding_id": str(primary.get("gem_finding_id") or ""),
                            "secondary_variant_name": str(secondary.get("variant_name") or ""),
                            "instruction": (
                                "Use this Gem as the primary implementation "
                                "parent unless your contract gives a clear "
                                "scientific reason to deviate."
                            ),
                        }
                    )
        return {
            "enabled": True,
            "cycle_index": int(state.get("cycle_index", 0)),
            "reset_count": reset_count,
            "cycle_start_generation": int(state.get("cycle_start_generation", 0)),
            "logical_generation": logical_gen,
            "gems_count": len(gems),
            "bottleneck_reports": _filter_bottleneck_reports_for_generation(
                _state_bottleneck_reports(state),
                evidence_cutoff,
            )[-5:],
            "latest_soft_agenda_priors": _latest_soft_priors_for_generation(
                _state_bottleneck_reports(state),
                evidence_cutoff,
                state.get("latest_soft_agenda_priors", {}) or {},
            ),
            "gem_seeded_baseline_mode": gem_seeded_baseline_mode,
            "baseline_code_policy": (
                "In Gem-seeded logical generation 0, peers should treat Gems "
                "as compact implementation or mechanism parents according to "
                "the task's Gems policy, while avoiding one-Gem relay collapse. "
                "If the task config reserves independent exploration or "
                "recombination slots, honor those slots. The original task "
                "baseline code is compatibility scaffolding only for direct "
                "Gem-inheritance peers; official baseline and benchmark records "
                "remain performance references."
            ),
            "official_baseline_performance_policy": (
                "Keep comparing against committed official baseline records "
                "and task-defined benchmark or diagnostic references. Do not "
                "overwrite or hide those performance references."
            ),
            "primary_gem_anchor": primary_gem_anchor,
            "secondary_gem_anchor": secondary_gem_anchor,
            "gem_anchor_roster": gem_anchor_roster,
            "gem_anchor_assignment_mode": gem_anchor_assignment_mode,
            "gem_inheritance_slots": gem_inheritance_slots,
            "independent_exploration_slots": independent_exploration_slots,
            # Gems are durable anchors, not an ordinary top-k summary. All Gem
            # ids/refs must stay visible after repeated diversity restarts; the
            # detailed payload lives in finding files, so this compact list is
            # bounded by the task's global max_gems_total.
            "gems": gems,
        }

    def active_gems_from_state(
        self,
        state: dict[str, Any],
        *,
        max_generation_id: int | None = None,
    ) -> list[dict[str, Any]]:
        raw_gems = state.get("gems") if isinstance(state.get("gems"), list) else []
        allow_unknown_source_generation = max_generation_id is None
        source_limit = self._state_source_generation_limit(state)
        if max_generation_id is not None:
            source_limit = (
                int(max_generation_id)
                if source_limit is None
                else min(source_limit, int(max_generation_id))
            )
        return self._compact_gems(
            raw_gems,
            sort_by_performance=self._mature_evidence_topk_policy_enabled(),
            max_generation_id=source_limit,
            allow_unknown_source_generation=allow_unknown_source_generation,
            preserve_committed_gems=True,
        )

    def active_gems(self) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        state = self.load_state()
        if not _is_clean_runtime_state(state):
            logger.warning(
                "gems: ignoring non-committed runtime Gems state for active gems: %s",
                self.state_path,
            )
            return []
        return self.active_gems_from_state(state)

    def frontier_manifest(self) -> dict[str, Any]:
        if hasattr(self.frontier, "get_manifest"):
            manifest = self.frontier.get_manifest()
            if isinstance(manifest, dict) and _is_clean_runtime_state(manifest):
                return manifest
        path = self.run_dir / "frontier" / "frontier_manifest.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and _is_clean_runtime_state(data):
                    return data
            except (OSError, json.JSONDecodeError):
                pass
        return {}

    def _frontier_signature_payload(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        lane_frontiers = manifest.get("lane_frontiers")
        entries: list[tuple[str, dict[str, Any]]] = []
        if isinstance(lane_frontiers, dict) and lane_frontiers:
            for lane in sorted(lane_frontiers):
                raw_entries = lane_frontiers.get(lane)
                if not isinstance(raw_entries, list):
                    continue
                for entry in raw_entries[
                    : int(getattr(self.config, "signature_entries_per_lane", 8) or 8)
                ]:
                    if isinstance(entry, dict):
                        entries.append((str(lane), entry))
        else:
            raw_entries = manifest.get("cumulative_top")
            if isinstance(raw_entries, list):
                for entry in raw_entries[: int(getattr(self.config, "signature_top_k", 16) or 16)]:
                    if isinstance(entry, dict):
                        entries.append(("", entry))

        payload: list[dict[str, Any]] = []
        for lane, entry in entries:
            metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
            payload.append(
                {
                    "lane": lane
                    or str(entry.get("promoted_for_lane") or entry.get("frontier_lane") or ""),
                    "finding_id": str(entry.get("finding_id") or ""),
                    "variant_name": str(entry.get("variant_name") or ""),
                    "metric_name": str(
                        entry.get("lane_metric_name") or entry.get("metric_name") or ""
                    ),
                    "metric_value": _safe_metric(
                        entry.get("lane_metric_value", entry.get("metric_value"))
                    ),
                    "tier": str(metrics.get("tier") or metrics.get("tier_reached") or ""),
                    "family": str(metrics.get("strategy_family") or metrics.get("family") or ""),
                }
            )
        payload.sort(key=lambda item: (item["lane"], item["variant_name"], item["finding_id"]))
        return payload

    def _signature_hash(self, payload: list[dict[str, Any]]) -> str:
        import hashlib

        blob = _json_dumps_compact(payload).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def _max_gems_total(self) -> int:
        fallback = int(getattr(self.config, "max_gems_per_reset", 4) or 4)
        return max(1, int(getattr(self.config, "max_gems_total", fallback) or fallback))

    def _max_gems_per_family(self) -> int:
        return max(1, int(getattr(self.config, "max_gems_per_family", 2) or 2))

    def _performance_lanes(self) -> set[str]:
        configured = {
            str(item).strip()
            for item in (getattr(self.config, "performance_lanes", None) or [])
            if str(item).strip()
        }
        if configured:
            return configured
        task_owned = {
            name for name, eligible in self._lane_parent_eligibility().items() if eligible
        }
        if task_owned:
            return task_owned
        return set(_GENERIC_PERFORMANCE_LANES)

    def _control_lanes(self) -> set[str]:
        configured = {
            str(item).strip()
            for item in (getattr(self.config, "control_lanes", None) or [])
            if str(item).strip()
        }
        if configured:
            return configured
        task_owned = {
            name for name, eligible in self._lane_parent_eligibility().items() if not eligible
        }
        if task_owned:
            return task_owned
        return set(_GENERIC_CONTROL_LANES)

    def _lane_parent_eligibility(self) -> dict[str, bool]:
        evaluation = getattr(self.task_spec, "evaluation", None)
        lanes = getattr(evaluation, "frontier_lanes", None)
        if not isinstance(lanes, list):
            return {}
        eligibility: dict[str, bool] = {}
        for lane in lanes:
            if not isinstance(lane, dict):
                continue
            name = str(lane.get("name") or "").strip()
            if not name:
                continue
            allow_lower_tier = bool(lane.get("allow_lower_tier", False))
            eligibility[name] = bool(lane.get("parent_eligible", not allow_lower_tier))
        return eligibility

    def _entry_parent_eligible(self, entry: dict[str, Any]) -> bool:
        explicit = _boolish_entry_field(entry, "parent_eligible")
        lane_policy = self._lane_parent_eligibility()
        if not lane_policy:
            return True if explicit is None else explicit
        configured = lane_policy.get(_entry_lane(entry))
        if configured is not True:
            return False
        return explicit is not False

    def _preferred_lanes(self) -> list[str]:
        include_lanes = [
            str(x) for x in (getattr(self.config, "include_lanes", None) or []) if str(x)
        ]
        ordered: list[str] = []
        task_lanes = list(self._lane_parent_eligibility())
        fallback_lanes = [] if task_lanes else list(_GENERIC_PREFERRED_LANES)
        for lane in [*include_lanes, *task_lanes, *fallback_lanes]:
            if lane not in ordered:
                ordered.append(lane)
        return ordered

    def _result_artifact_default_lane(self) -> str:
        return str(getattr(self.config, "result_artifact_default_lane", "") or "performance")

    def _result_artifact_default_family(self) -> str:
        return str(getattr(self.config, "result_artifact_default_family", "") or "task_candidate")

    def _entry_family_for_caps(self, entry: dict[str, Any]) -> str:
        family = _entry_family(entry)
        default_family = self._result_artifact_default_family().strip().lower()
        if family and default_family and family == default_family:
            return ""
        return family

    def _is_task_performance_entry(self, entry: dict[str, Any]) -> bool:
        lane = _entry_lane(entry)
        metrics = _entry_metrics(entry)
        strategy = str(entry.get("strategy_family") or metrics.get("strategy_family") or "").strip()
        if strategy and strategy in self._control_lanes():
            return False
        if lane and lane in self._control_lanes():
            return False
        if lane and lane in self._performance_lanes():
            return True
        if lane:
            return lane in self._performance_lanes()
        if strategy:
            default_family = self._result_artifact_default_family().strip()
            allowed = {"learned_candidate", "candidate", "task_candidate"}
            if default_family:
                allowed.add(default_family)
            return strategy in allowed
        return _is_performance_entry(entry)

    def _mature_evidence_topk_policy_enabled(self) -> bool:
        policy = normalize_gems_selection_policy(getattr(self.config, "selection_policy", ""))
        return policy == GEMS_MATURE_EVIDENCE_TOP_K

    def _min_mature_eval_units(self) -> int:
        return max(1, int(getattr(self.config, "min_mature_eval_units", 1) or 1))

    def _evidence_stage_min_units(self) -> dict[str, int]:
        raw = getattr(self.config, "evidence_stage_min_units", None) or {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, int] = {}
        for key, value in raw.items():
            try:
                out[str(key).strip().lower()] = max(1, int(value))
            except (TypeError, ValueError):
                continue
        return out

    def _configured_metric_keys(
        self, field_name: str, fallback: tuple[str, ...]
    ) -> tuple[str, ...]:
        raw = getattr(self.config, field_name, None) or []
        values = raw if isinstance(raw, (list, tuple)) else [raw]
        keys = [str(value).strip() for value in values if str(value).strip()]
        if field_name == "primary_metric_keys":
            primary_metric = str(
                getattr(getattr(self.task_spec, "evaluation", None), "primary_metric", "") or ""
            ).strip()
            if primary_metric and primary_metric not in keys:
                keys.append(primary_metric)
        for key in fallback:
            if key not in keys:
                keys.append(key)
        return tuple(keys)

    def _with_gems_sort_config(self, entry: dict[str, Any]) -> dict[str, Any]:
        item = dict(entry)
        item["_gems_metric_direction"] = str(
            getattr(getattr(self.task_spec, "evaluation", None), "direction", "maximize")
            or "maximize"
        )
        item["_gems_primary_metric_keys"] = self._configured_metric_keys(
            "primary_metric_keys",
            _GENERIC_GEM_PRIMARY_METRIC_KEYS,
        )
        item["_gems_secondary_metric_keys"] = self._configured_metric_keys(
            "secondary_metric_keys",
            _GENERIC_GEM_SECONDARY_METRIC_KEYS,
        )
        item["_gems_lower_tail_metric_keys"] = self._configured_metric_keys(
            "lower_tail_metric_keys",
            _GENERIC_GEM_LOWER_TAIL_METRIC_KEYS,
        )
        item["_gems_validation_metric_keys"] = self._configured_metric_keys(
            "validation_metric_keys",
            _GENERIC_GEM_VALIDATION_METRIC_KEYS,
        )
        item["_gems_cost_metric_keys"] = self._configured_metric_keys(
            "cost_metric_keys",
            _GENERIC_GEM_RISK_METRIC_KEYS,
        )
        return item

    def _all_configured_gem_metric_keys(self) -> tuple[str, ...]:
        keys: list[str] = []
        for field_name, fallback in (
            ("primary_metric_keys", _GENERIC_GEM_PRIMARY_METRIC_KEYS),
            ("secondary_metric_keys", _GENERIC_GEM_SECONDARY_METRIC_KEYS),
            ("lower_tail_metric_keys", _GENERIC_GEM_LOWER_TAIL_METRIC_KEYS),
            ("validation_metric_keys", _GENERIC_GEM_VALIDATION_METRIC_KEYS),
            ("cost_metric_keys", _GENERIC_GEM_RISK_METRIC_KEYS),
        ):
            for key in self._configured_metric_keys(field_name, fallback):
                if key not in keys:
                    keys.append(key)
        return tuple(keys)

    def _manifest_gems_config(self) -> dict[str, Any]:
        return {
            "selection_policy": str(
                getattr(self.config, "selection_policy", "frontier_lane_balanced")
                or "frontier_lane_balanced"
            ),
            "min_mature_eval_units": self._min_mature_eval_units(),
            "evidence_stage_min_units": self._evidence_stage_min_units(),
            "performance_lanes": sorted(self._performance_lanes()),
            "control_lanes": sorted(self._control_lanes()),
            "primary_metric_keys": list(
                self._configured_metric_keys(
                    "primary_metric_keys",
                    _GENERIC_GEM_PRIMARY_METRIC_KEYS,
                )
            ),
            "secondary_metric_keys": list(
                self._configured_metric_keys(
                    "secondary_metric_keys",
                    _GENERIC_GEM_SECONDARY_METRIC_KEYS,
                )
            ),
            "lower_tail_metric_keys": list(
                self._configured_metric_keys(
                    "lower_tail_metric_keys",
                    _GENERIC_GEM_LOWER_TAIL_METRIC_KEYS,
                )
            ),
            "validation_metric_keys": list(
                self._configured_metric_keys(
                    "validation_metric_keys",
                    _GENERIC_GEM_VALIDATION_METRIC_KEYS,
                )
            ),
            "cost_metric_keys": list(
                self._configured_metric_keys(
                    "cost_metric_keys",
                    _GENERIC_GEM_RISK_METRIC_KEYS,
                )
            ),
            "result_cell_metric_derivations": list(
                getattr(self.config, "result_cell_metric_derivations", []) or []
            ),
            "result_metric_aliases": dict(getattr(self.config, "result_metric_aliases", {}) or {}),
        }

    def _manifest_entries(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        metric_direction = str(manifest.get("metric_direction") or "").strip()
        lane_frontiers = manifest.get("lane_frontiers")
        if isinstance(lane_frontiers, dict):
            for lane, raw_entries in lane_frontiers.items():
                if not isinstance(raw_entries, list):
                    continue
                for raw in raw_entries:
                    if not isinstance(raw, dict):
                        continue
                    item = dict(raw)
                    if lane:
                        item.setdefault("frontier_lane", str(lane))
                        item.setdefault("promoted_for_lane", str(lane))
                    if metric_direction:
                        item.setdefault("metric_direction", metric_direction)
                    entries.append(item)
        raw_entries = manifest.get("cumulative_top")
        if isinstance(raw_entries, list):
            for entry in raw_entries:
                if not isinstance(entry, dict):
                    continue
                item = dict(entry)
                if metric_direction:
                    item.setdefault("metric_direction", metric_direction)
                entries.append(item)
        generations = manifest.get("generations")
        if isinstance(generations, dict):
            for gen_key, raw_entries in generations.items():
                if not isinstance(raw_entries, list):
                    continue
                try:
                    generation_id = int(gen_key)
                except (TypeError, ValueError):
                    generation_id = None
                for entry in raw_entries:
                    if not isinstance(entry, dict):
                        continue
                    item = dict(entry)
                    if generation_id is not None:
                        item.setdefault("generation_id", generation_id)
                    if metric_direction:
                        item.setdefault("metric_direction", metric_direction)
                    entries.append(item)
        return entries

    def _result_artifact_gem_candidates(
        self,
        *,
        max_generation_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Materialize scored result summaries as Gems candidates.

        Frontier promotion can miss variant-level result artifacts when a peer
        writes only a family/sweep insight. Gems are restart parents, so the
        top mature-evidence candidates must be recoverable from the authoritative
        `results/*/tiered_eval_summary.json` artifacts as well as ordinary
        findings.
        """

        if not bool(getattr(self.config, "result_artifact_materialization", True)):
            return []
        candidates: list[dict[str, Any]] = []
        result_options = result_artifact_options_from_task_spec(self.task_spec)
        existing_results = _existing_materialized_results(self.run_dir / "shared_findings")
        for summary_path in iter_result_summary_paths(self.run_dir):
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("gems: could not read result summary %s: %s", summary_path, exc)
                continue
            if not isinstance(summary, dict):
                continue
            summary = normalized_result_summary(
                summary,
                summary_path=summary_path,
                maturity_policy=result_options.get("result_maturity_policy"),
            )
            source_result_path = str(summary_path.relative_to(self.run_dir))
            current_result_control_digest = result_summary_control_digest(summary)
            existing_result = existing_results.get(source_result_path) or {}
            existing_result_metrics = (
                existing_result.get("metrics")
                if isinstance(existing_result.get("metrics"), dict)
                else {}
            )
            prior_result_control_digest = (
                ""
                if existing_result.get("trusted_materializer_record") is not True
                or existing_result_metrics.get("late_after_generation_boundary") is True
                else str(existing_result.get("source_result_sha256") or "")
            )
            normalized_metrics = _result_summary_metrics(
                summary,
                cell_metric_derivations=result_options.get("result_cell_metric_derivations"),
                metric_aliases=result_options.get("result_metric_aliases"),
                scoring_metric_keys=result_options.get("result_scoring_metric_keys"),
                maturity_policy=result_options.get("result_maturity_policy"),
            )
            # The directory name is the concrete variant artifact identity.
            # Some summaries carry a family-level or generic `variant_name`
            # that would merge distinct
            # candidate artifacts and recreate the leaderboard→Gems mismatch.
            variant = result_summary_variant_name(summary_path, summary, self.run_dir)
            reported_variant = str(summary.get("variant_name") or "").strip()
            metrics = dict(
                summary.get("current_aggregate")
                if isinstance(summary.get("current_aggregate"), dict)
                else {}
            )
            evaluation_units = next(
                (
                    value
                    for value in (
                        summary.get("completed_required_eval_units"),
                        summary.get("actual_eval_units"),
                        summary.get("evaluation_units"),
                        metrics.get("completed_required_eval_units"),
                        metrics.get("actual_eval_units"),
                        metrics.get("evaluation_units"),
                        summary.get("n_eval_cells"),
                        metrics.get("n_eval_cells"),
                    )
                    if value is not None
                ),
                None,
            )
            summary_fields = {
                "tier": summary.get("tier_reached") or summary.get("completed_tier") or "",
                "tier_reached": summary.get("tier_reached") or "",
                "completed_tier": summary.get("completed_tier") or "",
                "tier_status": summary.get("tier_status") or "",
                "final_status": summary.get("final_status") or "",
                "result_status": summary.get("result_status") or "",
                "completion_status": summary.get("completion_status") or "",
                "eval_status": summary.get("eval_status") or "",
                "scout_only": summary.get("scout_only"),
                "is_scout_eval": summary.get("is_scout_eval"),
                "is_smoke_eval": summary.get("is_smoke_eval"),
                "summary_only": summary.get("summary_only"),
                "is_summary_only": summary.get("is_summary_only"),
                "partial_cohort": summary.get("partial_cohort"),
                "partial_eval": summary.get("partial_eval"),
                "is_partial_eval": summary.get("is_partial_eval"),
                "unscored_artifact": summary.get("unscored_artifact"),
                "incomplete_eval": summary.get("incomplete_eval"),
                "scored_complete": summary.get("scored_complete"),
                "is_scored_complete": summary.get("is_scored_complete"),
                "complete_eval": summary.get("complete_eval"),
                "is_complete_eval": summary.get("is_complete_eval"),
                "capped": summary.get("capped"),
                "is_capped": summary.get("is_capped"),
                "result_capped": summary.get("result_capped"),
                "evaluation_units": evaluation_units,
                "source_result_path": str(summary_path.relative_to(self.run_dir)),
                "source_result_kind": summary_path.name,
                "reported_variant_name": reported_variant,
                "auto_materialized_from_result_artifact": True,
                "strategy_family": metrics.get("strategy_family")
                or self._result_artifact_default_family(),
                "frontier_lane": metrics.get("frontier_lane")
                or self._result_artifact_default_lane(),
                "generation_id": summary.get("generation_id"),
                "source_generation_id": summary.get("source_generation_id"),
                "gen_id": summary.get("gen_id"),
                "promotion_eligible": summary.get("promotion_eligible"),
                "clean_promotion_eligible": summary.get("clean_promotion_eligible"),
                "mature_enough": summary.get("mature_enough"),
                "maturity_basis": summary.get("maturity_basis"),
                "effort_ratio": summary.get("effort_ratio"),
                "coverage_ratio": summary.get("coverage_ratio"),
                "actual_effort_units": summary.get("actual_effort_units"),
                "reference_effort_units": summary.get("reference_effort_units"),
                "completed_required_eval_units": summary.get("completed_required_eval_units"),
                "total_required_eval_units": summary.get("total_required_eval_units"),
                "hard_constraint_violations": summary.get("hard_constraint_violations"),
                "protocol_integrity_status": summary.get("protocol_integrity_status"),
                "protocol_integrity_violation_count": summary.get(
                    "protocol_integrity_violation_count"
                ),
                "suspect_protocol": summary.get("suspect_protocol")
                or summary.get("suspect_fixed_weight_eval"),
            }
            aggregate = (
                summary.get("current_aggregate")
                if isinstance(summary.get("current_aggregate"), dict)
                else {}
            )
            for key in _DURABLE_ROUTING_MARKER_KEYS:
                value = _durable_routing_marker_value(summary, aggregate, key)
                if value is not None:
                    summary_fields[key] = value
            for key, value in summary_fields.items():
                if value not in ("", None, [], {}):
                    metrics[key] = value
            # Reuse the canonical result-artifact normalization from findings
            # collection so Gems sees the same completion/partial/failure
            # evidence as frontier and research memory. In particular,
            # `failed_cells` and unscored/incomplete statuses must block
            # mature-evidence Gems admission even if a summary also contains a
            # numeric score.
            for key, value in normalized_metrics.items():
                if value not in ("", None, [], {}):
                    metrics[key] = value
            for key in _DURABLE_ROUTING_MARKER_KEYS:
                value = _durable_routing_marker_value(summary, aggregate, key)
                if value is not None:
                    metrics[key] = value
            strip_effective_config_fields(metrics)
            metrics.update(result_effective_config_metadata(summary))
            for key in list(metrics):
                if metrics.get(key) in ("", None, [], {}):
                    metrics.pop(key, None)
            canonicalize_evaluation_unit_metadata(metrics, summary, aggregate)
            configured_primary_score = _metric_float_if_present(
                {"metrics": metrics},
                *self._configured_metric_keys("primary_metric_keys", ()),
            )
            if configured_primary_score is not None:
                metrics.setdefault("primary_metric_value", configured_primary_score)
                metrics.setdefault(
                    "primary_metric_value_direction",
                    str(
                        getattr(
                            getattr(self.task_spec, "evaluation", None), "direction", "maximize"
                        )
                        or "maximize"
                    ),
                )
            hard_violations = metrics.get("n_hard_constraint_violations")
            if hard_violations is None:
                raw_violations = summary.get("hard_constraint_violations")
                if isinstance(raw_violations, list):
                    hard_violations = len(raw_violations)
            if hard_violations is not None:
                metrics["n_hard_constraint_violations"] = hard_violations
            source_generation = _entry_source_generation_id(
                {"variant_name": variant, "metrics": metrics}
            )
            if source_generation is None:
                source_generation, gen_source = _infer_result_generation(
                    run_dir=self.run_dir,
                    summary_path=summary_path,
                    summary=summary,
                    variant=variant,
                    boundary_gen_id=(
                        int(max_generation_id) if max_generation_id is not None else 0
                    ),
                )
                metrics["source_generation_inference"] = gen_source
                if gen_source == "boundary_fallback":
                    metrics["source_generation_low_confidence"] = True
                    metrics["provenance_warning"] = "source_generation_boundary_fallback"
            if max_generation_id is not None and source_generation > int(max_generation_id):
                continue
            metrics["source_generation_id"] = source_generation
            boundary_checkpoint = read_boundary_evidence_checkpoint(
                self.run_dir,
                source_generation,
            )
            if boundary_checkpoint is None:
                evidence_cutoff = None
                evidence_source_snapshot = None
            else:
                evidence_cutoff, evidence_source_snapshot = boundary_checkpoint
            late_boundary_info = _late_generation_boundary_info(
                run_dir=self.run_dir,
                summary_path=summary_path,
                source_gen_id=source_generation,
                evidence_cutoff=evidence_cutoff,
                evidence_source_snapshot=evidence_source_snapshot,
                current_result_control_digest=current_result_control_digest,
                prior_result_control_digest=prior_result_control_digest,
            )
            if late_boundary_info:
                metrics.update(late_boundary_info)
                metrics["late_after_generation_boundary"] = True
                metrics["artifact_signal_status"] = "late_after_generation_boundary"
                metrics["promotion_eligible"] = False
                metrics["clean_promotion_eligible"] = False
                metrics["excluded_from_durable_frontier"] = True
                metrics.setdefault("exclusion_reason", "late_after_generation_boundary")
                metrics.setdefault(
                    "recommended_next_step",
                    "review_or_revalidate_late_result_before_promotion",
                )
            candidate = {
                "finding_id": f"result_artifact::{variant}",
                "variant_name": variant,
                "frontier_lane": str(
                    metrics.get("frontier_lane") or self._result_artifact_default_lane()
                ),
                "promoted_for_lane": str(
                    metrics.get("frontier_lane") or self._result_artifact_default_lane()
                ),
                "metrics": metrics,
                "tier": metrics.get("tier"),
                "evaluation_units": evaluation_units,
                "generation_id": source_generation,
                "source_result_path": str(summary_path.relative_to(self.run_dir)),
            }
            evidence_stage = (
                metrics.get("evidence_stage")
                or summary.get("evidence_stage")
                or metrics.get("tier")
                or summary.get("tier")
            )
            if evidence_stage:
                candidate["evidence_stage"] = evidence_stage
            candidate_for_admission = self._with_gems_sort_config(candidate)
            if self._mature_evidence_topk_policy_enabled():
                if not _entry_is_mature_gem_admission_candidate(
                    candidate_for_admission,
                    min_mature_eval_units=self._min_mature_eval_units(),
                    evidence_stage_min_units=self._evidence_stage_min_units(),
                    maturity_policy=self.maturity_policy,
                ):
                    continue
            elif not _entry_is_clean_gem_admission_candidate(
                candidate_for_admission,
                maturity_policy=self.maturity_policy,
            ):
                continue
            candidates.append(
                {key: value for key, value in candidate.items() if value not in ("", None, [], {})}
            )
        return candidates

    @staticmethod
    def _infer_generation_id_from_variant(variant: str) -> int | None:
        return _infer_generation_id_from_text(variant)

    def _state_source_generation_limit(self, state: dict[str, Any]) -> int | None:
        if not self._mature_evidence_topk_policy_enabled():
            return None
        return _persisted_gems_source_generation_limit(state)

    @staticmethod
    def _operator_pruned_restart_cutoff(
        state: dict[str, Any],
        absolute_gen_id: int,
    ) -> int | None:
        reset_events = state.get("reset_events")
        if not isinstance(reset_events, list):
            return None
        for event in reversed(reset_events):
            if not isinstance(event, dict):
                continue
            try:
                next_abs_raw = event.get("next_absolute_generation")
                completed_raw = event.get("completed_gen_id")
                restart_raw = event.get("operator_pruned_restart_generation")
                if next_abs_raw is None or completed_raw is None or restart_raw is None:
                    continue
                next_abs = int(next_abs_raw)
                completed = int(completed_raw)
                restart = int(restart_raw)
            except (TypeError, ValueError):
                continue
            if next_abs != int(absolute_gen_id) or restart != int(absolute_gen_id):
                continue
            if str(event.get("committed", True)).strip().lower() in {"0", "false", "no", "off"}:
                continue
            return completed
        return None

    def _resolved_entry_source_generation_id(self, entry: dict[str, Any]) -> int | None:
        """Return source generation, using Gem sidecars for legacy persisted Gems.

        New Gem records store ``source_generation_id`` directly. Older states
        may only store a Gem finding id, while the corresponding finding file
        still carries ``generation_id``. Treat such records as usable only when
        that provenance can be recovered from the sidecar; otherwise leave them
        unknown so cutoff-sensitive contexts cannot leak stale or future Gems.
        """

        return _resolved_persisted_gem_source_generation_id(self.run_dir, entry)

    def _select_mature_evidence_topk_entries(
        self,
        manifest: dict[str, Any],
        *,
        existing_gems: list[dict[str, Any]] | None = None,
        max_generation_id: int | None = None,
    ) -> list[dict[str, Any]]:
        total_cap = self._max_gems_total()
        cap = min(int(getattr(self.config, "max_gems_per_reset", 4) or 4), total_cap)
        if cap <= 0:
            return []
        min_cells = self._min_mature_eval_units()
        pool = [
            *self._manifest_entries(manifest),
            *self._result_artifact_gem_candidates(max_generation_id=max_generation_id),
            *[dict(gem) for gem in (existing_gems or []) if isinstance(gem, dict)],
        ]
        pool = [self._with_gems_sort_config(entry) for entry in pool]
        eligible = [
            entry
            for entry in pool
            if (
                (source_gen := self._resolved_entry_source_generation_id(entry)) is not None
                and (max_generation_id is None or source_gen <= int(max_generation_id))
                and not _entry_has_low_confidence_generation(entry)
                and self._entry_parent_eligible(entry)
                and self._is_task_performance_entry(entry)
                and _entry_is_mature_gem_admission_candidate(
                    entry,
                    min_mature_eval_units=min_cells,
                    evidence_stage_min_units=self._evidence_stage_min_units(),
                    maturity_policy=self.maturity_policy,
                )
            )
        ]
        by_variant: dict[str, dict[str, Any]] = {}
        for entry in sorted(eligible, key=_gem_performance_key, reverse=True):
            key = _variant_key(entry)
            if not key:
                continue
            current = by_variant.get(key)
            if current is None or _gem_performance_key(entry) > _gem_performance_key(current):
                by_variant[key] = dict(entry)
        ranked = sorted(by_variant.values(), key=_gem_performance_key, reverse=True)
        return ranked[:cap]

    def _compact_gems(
        self,
        gems: list[dict[str, Any]],
        *,
        cap: int | None = None,
        sort_by_performance: bool = False,
        preserve_lane_reserves: bool = False,
        max_generation_id: int | None = None,
        allow_legacy_unknown_source: bool = False,
        allow_unknown_source_generation: bool = False,
        preserve_committed_gems: bool = False,
    ) -> list[dict[str, Any]]:
        """Deduplicate and globally cap durable Gems.

        The cap is global across all diversity restarts. The same concrete
        variant may appear under multiple finding ids, so this compaction keeps
        the first occurrence by variant identity and applies a small per-family
        cap when a family label is available.
        """

        max_total = self._max_gems_total() if cap is None else max(1, int(cap))
        mature_evidence_topk = self._mature_evidence_topk_policy_enabled()

        def is_committed_context(gem: dict[str, Any]) -> bool:
            ratio_decision = evidence_maturity_snapshot(gem, self.maturity_policy).get(
                "mature_enough"
            )
            legacy_complete = bool(
                _boolish_entry_field(gem, "_legacy_committed_complete_evidence") is True
                and (
                    _entry_eval_unit_count(gem) == 0
                    or _entry_eval_unit_count(gem) >= self._min_mature_eval_units()
                )
            )
            return bool(
                preserve_committed_gems
                and gem.get("gem_finding_id")
                and _entry_has_score_evidence(gem)
                and (_entry_has_explicit_complete_eval_evidence(gem) or legacy_complete)
                and not _entry_is_scout_or_partial(gem)
                and not _entry_has_hard_gem_rejection_marker(gem)
                and not _entry_has_nonclean_gem_marker(gem)
                and ratio_decision is not False
            )

        def is_pending_reset_anchor(gem: dict[str, Any]) -> bool:
            return bool(
                allow_legacy_unknown_source
                and (
                    _entry_is_recoverable_legacy_or_control_gem_source(gem)
                    or _entry_lane(gem) in self._control_lanes()
                )
            )

        def has_compatible_complete_evidence(gem: dict[str, Any]) -> bool:
            return bool(
                not normalize_maturity_policy(self.maturity_policy).get("require_ratio_gate")
                and evidence_maturity_snapshot(gem, self.maturity_policy).get("mature_enough")
                is None
                and _entry_has_explicit_complete_eval_evidence(gem)
            )

        def is_mature_topk_candidate(gem: dict[str, Any]) -> bool:
            return bool(
                mature_evidence_topk
                and self._is_task_performance_entry(gem)
                and _entry_is_mature_gem_admission_candidate(
                    gem,
                    min_mature_eval_units=self._min_mature_eval_units(),
                    evidence_stage_min_units=self._evidence_stage_min_units(),
                    maturity_policy=self.maturity_policy,
                )
            )

        raw_gems = []
        for gem in gems:
            if not isinstance(gem, dict):
                continue
            configured = self._with_gems_sort_config(migrate_legacy_gems_entry(gem))
            if not self._entry_parent_eligible(configured) and not is_pending_reset_anchor(
                configured
            ):
                continue
            has_generic_mature_evidence = _entry_has_generic_mature_evidence(
                configured,
                self.maturity_policy,
            )
            if not has_generic_mature_evidence and _entry_is_scout_or_partial(configured):
                continue
            if _entry_has_hard_gem_rejection_marker(configured):
                continue
            if not mature_evidence_topk and _entry_has_nonclean_gem_marker(configured):
                continue
            raw_gems.append(configured)
        raw_gems = [
            gem
            for gem in raw_gems
            if (
                _entry_has_generic_mature_evidence(gem, self.maturity_policy)
                or is_mature_topk_candidate(gem)
                or has_compatible_complete_evidence(gem)
                or (
                    allow_legacy_unknown_source
                    and (
                        _entry_is_recoverable_legacy_or_control_gem_source(gem)
                        or _entry_lane(gem) in self._control_lanes()
                        or (
                            bool(gem.get("gem_finding_id"))
                            and isinstance(gem.get("admission_metrics"), dict)
                            and _entry_has_score_evidence(gem)
                        )
                    )
                )
                or (allow_legacy_unknown_source and _entry_is_legacy_persisted_gem(gem))
                or is_committed_context(gem)
            )
        ]

        def source_generation_allowed(gem: dict[str, Any]) -> bool:
            if max_generation_id is None:
                return True
            source_gen = self._resolved_entry_source_generation_id(gem)
            if source_gen is None:
                return allow_unknown_source_generation or allow_legacy_unknown_source
            return source_gen <= int(max_generation_id)

        raw_gems = [gem for gem in raw_gems if source_generation_allowed(gem)]
        if mature_evidence_topk:
            raw_gems = [
                gem
                for gem in raw_gems
                if (
                    (is_committed_context(gem) and self._is_task_performance_entry(gem))
                    or is_mature_topk_candidate(gem)
                )
            ]
        if mature_evidence_topk and sort_by_performance:
            compact: list[dict[str, Any]] = []
            seen: set[str] = set()
            for raw in sorted(
                raw_gems,
                key=_gem_performance_key,
                reverse=True,
            ):
                key = _variant_key(raw)
                if key and key in seen:
                    continue
                compact.append(dict(raw))
                if key:
                    seen.add(key)
                if len(compact) >= max_total:
                    break
            return compact
        max_family = self._max_gems_per_family()
        if sort_by_performance and preserve_lane_reserves:
            return self._compact_gems_with_lane_reserves(
                raw_gems,
                max_total=max_total,
                max_family=max_family,
            )
        compact: list[dict[str, Any]] = []
        seen: set[str] = set()
        family_counts: Counter[str] = Counter()
        if sort_by_performance:
            raw_gems = sorted(raw_gems, key=_gem_performance_key, reverse=True)
        for raw in raw_gems:
            if not isinstance(raw, dict):
                continue
            key = _variant_key(raw)
            if key and key in seen:
                continue
            family = self._entry_family_for_caps(raw)
            if family and family_counts[family] >= max_family:
                continue
            compact.append(dict(raw))
            if key:
                seen.add(key)
            if family:
                family_counts[family] += 1
            if len(compact) >= max_total:
                break
        return compact

    def _ordered_lanes_for_gems(self, gems: list[dict[str, Any]]) -> list[str]:
        present = {_entry_lane(gem) for gem in gems if _entry_lane(gem)}
        ordered: list[str] = []
        for lane in self._preferred_lanes():
            if lane in present and lane not in ordered:
                ordered.append(lane)
        for lane in sorted(present - set(ordered)):
            ordered.append(lane)
        if not ordered and gems:
            ordered.append("")
        return ordered

    def _compact_gems_with_lane_reserves(
        self,
        gems: list[dict[str, Any]],
        *,
        max_total: int,
        max_family: int,
    ) -> list[dict[str, Any]]:
        """Performance-sort Gems while keeping configured lane anchors visible."""

        ordered_lanes = self._ordered_lanes_for_gems(gems)
        lane_quotas = self._gem_lane_quotas(ordered_lanes=ordered_lanes, cap=max_total)
        compact: list[dict[str, Any]] = []
        seen: set[str] = set()
        family_counts: Counter[str] = Counter()

        def rebuild_performance_family_counts() -> None:
            family_counts.clear()
            for item in compact:
                if not self._is_task_performance_entry(item):
                    continue
                family = self._entry_family_for_caps(item)
                if family:
                    family_counts[family] += 1

        def add(raw: dict[str, Any], *, count_family: bool = True) -> bool:
            if len(compact) >= max_total:
                return False
            key = _variant_key(raw)
            if key and key in seen:
                return False
            family = self._entry_family_for_caps(raw)
            if count_family and family and family_counts[family] >= max_family:
                return False
            compact.append(dict(raw))
            if key:
                seen.add(key)
            if count_family and family:
                family_counts[family] += 1
            return True

        def remove_at(index: int) -> None:
            raw = compact.pop(index)
            key = _variant_key(raw)
            if key:
                seen.discard(key)
            rebuild_performance_family_counts()

        def weakest_performance_index() -> int | None:
            candidates = [
                (idx, raw)
                for idx, raw in enumerate(compact)
                if self._is_task_performance_entry(raw)
            ]
            if not candidates:
                return None
            idx, _ = min(candidates, key=lambda item: _gem_performance_key(item[1]))
            return idx

        by_lane: dict[str, list[dict[str, Any]]] = {
            lane: sorted(
                [gem for gem in gems if (_entry_lane(gem) or "") == lane],
                key=_gem_performance_key,
                reverse=True,
            )
            for lane in ordered_lanes
        }
        performance_candidates = [
            raw
            for raw in sorted(gems, key=_gem_performance_key, reverse=True)
            if self._is_task_performance_entry(raw)
        ]
        for raw in performance_candidates:
            if len(compact) >= max_total:
                break
            add(raw, count_family=True)

        for raw in sorted(gems, key=_gem_performance_key, reverse=True):
            if len(compact) >= max_total:
                break
            if self._is_task_performance_entry(raw):
                continue
            add(raw, count_family=False)

        for lane in ordered_lanes:
            if lane in self._performance_lanes():
                continue
            quota = lane_quotas.get(lane, 1)
            existing = sum(1 for raw in compact if (_entry_lane(raw) or "") == lane)
            taken = existing
            for raw in by_lane.get(lane, []):
                if taken >= quota:
                    break
                key = _variant_key(raw)
                if key and key in seen:
                    continue
                if len(compact) >= max_total:
                    victim = weakest_performance_index()
                    if victim is None:
                        break
                    remove_at(victim)
                # Non-performance anchors are useful context, but should not
                # consume the mechanism-family budget that protects performance Gems.
                if add(raw, count_family=False):
                    taken += 1

        for raw in sorted(gems, key=_gem_performance_key, reverse=True):
            if len(compact) >= max_total:
                break
            add(raw, count_family=self._is_task_performance_entry(raw))
        return compact

    def _select_gem_entries(
        self,
        manifest: dict[str, Any],
        *,
        existing_gems: list[dict[str, Any]] | None = None,
        completed_gen_id: int | None = None,
    ) -> list[dict[str, Any]]:
        if self._mature_evidence_topk_policy_enabled():
            return self._select_mature_evidence_topk_entries(
                manifest,
                existing_gems=existing_gems,
                max_generation_id=completed_gen_id,
            )
        manifest_entries = self._manifest_entries(manifest)
        lane_frontiers: dict[str, list[dict[str, Any]]] = {}
        unlaned_entries: list[dict[str, Any]] = []
        for entry in manifest_entries:
            lane = _entry_lane(entry)
            if lane:
                lane_frontiers.setdefault(lane, []).append(entry)
            else:
                unlaned_entries.append(entry)
        selected: list[dict[str, Any]] = []
        total_cap = self._max_gems_total()
        cap = min(int(getattr(self.config, "max_gems_per_reset", 12) or 12), total_cap)
        if cap <= 0:
            return []
        seen: set[str] = set()
        family_counts: Counter[str] = Counter()
        family_cap = self._max_gems_per_family()
        include_lanes = getattr(self.config, "include_lanes", None) or []
        include_lanes = [str(x) for x in include_lanes if str(x)]

        def admit(raw: dict[str, Any], lane: str = "") -> bool:
            configured = self._with_gems_sort_config(raw)
            if not self._entry_parent_eligible(configured):
                return False
            if not _entry_is_clean_gem_admission_candidate(
                configured,
                maturity_policy=self.maturity_policy,
            ):
                return False
            source_gen = _entry_source_generation_id(configured)
            if source_gen is None:
                return False
            if completed_gen_id is not None and source_gen > int(completed_gen_id):
                return False
            key = _variant_key(configured)
            if key and key in seen:
                return False
            family = self._entry_family_for_caps(configured)
            if family and family_counts[family] >= family_cap:
                return False
            item = dict(configured)
            if lane:
                item.setdefault("frontier_lane", lane)
                item.setdefault("promoted_for_lane", lane)
            selected.append(item)
            if key:
                seen.add(key)
            if family:
                family_counts[family] += 1
            return True

        if isinstance(lane_frontiers, dict) and lane_frontiers:
            ordered_lanes = []
            for lane in self._preferred_lanes():
                if lane in lane_frontiers and lane not in ordered_lanes:
                    ordered_lanes.append(lane)
            for lane in sorted(set(lane_frontiers) - set(ordered_lanes)):
                ordered_lanes.append(lane)
            lane_entries = {
                lane: [
                    self._with_gems_sort_config(raw)
                    for raw in lane_frontiers.get(lane, [])
                    if isinstance(raw, dict)
                ]
                for lane in ordered_lanes
                if isinstance(lane_frontiers.get(lane), list)
            }
            lane_quotas = self._gem_lane_quotas(ordered_lanes=ordered_lanes, cap=cap)

            # Durable Gems are restart parents. Preserve at least a small
            # performance-first slice from task performance lanes so strong
            # parents are not crowded out by lower-performing but more mature
            # artifacts or by coarse family labels.
            non_performance_lane_reserve = sum(
                lane_quotas.get(lane, 0)
                for lane in ordered_lanes
                if lane not in self._performance_lanes() and lane_entries.get(lane)
            )
            rescue_pool: list[tuple[str, dict[str, Any]]] = []
            for lane, raw_entries in lane_entries.items():
                if lane not in self._performance_lanes():
                    continue
                for raw in raw_entries:
                    rescue_pool.append((lane, raw))
            for raw in unlaned_entries:
                configured = self._with_gems_sort_config(raw)
                primary_keys = _entry_metric_key_list(
                    configured,
                    "_gems_primary_metric_keys",
                    _GENERIC_GEM_PRIMARY_METRIC_KEYS,
                )
                if self._is_task_performance_entry(configured) or any(
                    _metric_float_if_present(configured, key) is not None for key in primary_keys
                ):
                    rescue_pool.append(("", configured))
            rescue_slots = max(0, cap - non_performance_lane_reserve)
            for lane, raw in sorted(
                rescue_pool,
                key=lambda pair: (
                    *_gem_performance_key(pair[1]),
                    _performance_lane_priority(pair[0]),
                ),
                reverse=True,
            ):
                if len(selected) >= rescue_slots:
                    break
                admit(raw, lane)

            def take_next_non_performance(lane: str, limit: int) -> int:
                raw_entries = sorted(
                    lane_entries.get(lane, []),
                    key=_gem_performance_key,
                    reverse=True,
                )
                taken = 0
                for raw in raw_entries:
                    if taken >= limit or len(selected) >= cap:
                        break
                    if admit(raw, lane):
                        taken += 1
                return taken

            # Reserve only non-performance anchors. Performance lanes already
            # competed globally above, which prevents weak early lanes from
            # crowding out stronger continuation candidates.
            for lane in ordered_lanes:
                if len(selected) >= cap:
                    break
                if lane in self._performance_lanes():
                    continue
                take_next_non_performance(lane, lane_quotas.get(lane, 1))

            # Then fill any spare slot by global performance across all lanes.
            global_pool = [
                (lane, raw) for lane, raw_entries in lane_entries.items() for raw in raw_entries
            ]
            global_pool.extend(("", self._with_gems_sort_config(raw)) for raw in unlaned_entries)
            for lane, raw in sorted(
                global_pool,
                key=lambda pair: (
                    *_gem_performance_key(pair[1]),
                    _performance_lane_priority(pair[0]),
                ),
                reverse=True,
            ):
                if len(selected) >= cap:
                    break
                admit(raw, lane)
            if selected:
                return selected
        raw_entries = manifest.get("cumulative_top")
        if isinstance(raw_entries, list):
            fallback_entries = [
                self._with_gems_sort_config(entry)
                for entry in raw_entries
                if isinstance(entry, dict)
            ]
            if any(_entry_lane(entry) for entry in fallback_entries):
                fallback_entries = self._compact_gems(
                    fallback_entries,
                    cap=cap,
                    sort_by_performance=True,
                    preserve_lane_reserves=True,
                )
            else:
                fallback_entries = sorted(
                    fallback_entries,
                    key=_gem_performance_key,
                    reverse=True,
                )
            for raw in fallback_entries:
                if not isinstance(raw, dict):
                    continue
                admit(raw, _entry_lane(raw))
                if len(selected) >= cap:
                    break
        return selected

    def _gem_lane_quotas(self, *, ordered_lanes: list[str], cap: int) -> dict[str, int]:
        if cap <= 0:
            return {lane: 0 for lane in ordered_lanes}
        configured: dict[str, int] = {}
        evaluation = getattr(self.task_spec, "evaluation", None)
        frontier_lanes = getattr(evaluation, "frontier_lanes", None)
        if isinstance(frontier_lanes, list):
            for lane in frontier_lanes:
                if not isinstance(lane, dict):
                    continue
                name = str(lane.get("name") or "")
                if not name:
                    continue
                try:
                    configured[name] = max(1, int(lane.get("k") or 1))
                except (TypeError, ValueError):
                    configured[name] = 1
        quotas = {lane: configured.get(lane, 1) for lane in ordered_lanes}
        total = sum(quotas.values())
        if total <= cap:
            return quotas
        performance_lane_set = self._performance_lanes()
        if any(lane in performance_lane_set for lane in ordered_lanes):
            quotas = {lane: 0 for lane in ordered_lanes}
            non_perf_lanes = [lane for lane in ordered_lanes if lane not in performance_lane_set]
            max_non_perf = min(len(non_perf_lanes), max(0, cap // 2))
            for lane in non_perf_lanes[:max_non_perf]:
                quotas[lane] = 1
            remaining = cap - sum(quotas.values())
            performance_lanes = [lane for lane in ordered_lanes if lane in performance_lane_set]
            preferred_performance = performance_lanes[:1]
            if remaining > 0 and preferred_performance:
                quotas[preferred_performance[0]] = 1
                remaining -= 1
            for lane in performance_lanes:
                if remaining <= 0:
                    break
                if lane in preferred_performance and quotas[lane] > 0:
                    continue
                quotas[lane] = 1
                remaining -= 1
            while remaining > 0 and performance_lanes:
                for lane in performance_lanes:
                    if remaining <= 0:
                        break
                    quotas[lane] += 1
                    remaining -= 1
            return quotas
        # Fit oversized configurations without starving late lanes: everyone
        # keeps one slot first, remaining slots follow ordered lane priority.
        quotas = {lane: 1 for lane in ordered_lanes}
        remaining = max(0, cap - len(ordered_lanes))
        for lane in ordered_lanes:
            if remaining <= 0:
                break
            extra = max(0, configured.get(lane, 1) - 1)
            grant = min(extra, remaining)
            quotas[lane] += grant
            remaining -= grant
        return quotas

    def maybe_trigger_after_boundary(self, *, completed_gen_id: int) -> GemsTriggerResult:
        """Record advisory priors and perform a periodic Gems reset if due."""
        if not self.enabled:
            return GemsTriggerResult(False, reason="disabled")
        state = self.load_state()
        manifest = self.frontier_manifest()
        signature_payload = self._frontier_signature_payload(manifest)
        signature_hash = self._signature_hash(signature_payload)
        self._record_signature_audit(
            state=state,
            completed_gen_id=completed_gen_id,
            signature_hash=signature_hash,
            signature_payload=signature_payload,
        )
        self._record_bottleneck_report(
            state=state,
            completed_gen_id=completed_gen_id,
            manifest=manifest,
        )
        if not self.local_mode:
            # Server-mode findings are synchronized from an HTTP source of
            # truth. Until the server owns an archive/prune primitive and
            # cycle-aware visibility filters, a local Gems archive could be
            # undone by the next HTTP sync pass. Fail closed instead of
            # pretending ordinary findings were hidden. Soft bottleneck priors
            # are still persisted because they do not archive, reset, or prune.
            self.save_state(state)
            return GemsTriggerResult(False, reason="server_mode_not_supported")
        raw_max_resets = getattr(self.config, "max_resets", 3)
        max_resets = 3 if raw_max_resets is None else int(raw_max_resets)
        if int(state.get("reset_count", 0)) >= max_resets:
            self.save_state(state)
            return GemsTriggerResult(False, reason="max_resets_reached")
        min_entries = int(getattr(self.config, "min_frontier_entries", 1) or 1)
        if len(signature_payload) < min_entries and not self._mature_evidence_topk_policy_enabled():
            self.save_state(state)
            return GemsTriggerResult(False, reason="insufficient_frontier_entries")
        interval = int(getattr(self.config, "reset_interval_generations", 6) or 6)
        cycle_start = int(state.get("cycle_start_generation", 0) or 0)
        completed_in_cycle = int(completed_gen_id) - cycle_start + 1
        if completed_in_cycle < interval or completed_in_cycle % interval != 0:
            self.save_state(state)
            next_reset_at = (
                interval
                if completed_in_cycle <= 0
                else ((completed_in_cycle // interval) + 1) * interval
            )
            return GemsTriggerResult(
                False,
                reason=(
                    f"periodic_reset_waiting:"
                    f"completed_in_cycle={completed_in_cycle},"
                    f"next_reset_at={next_reset_at}"
                ),
                reset_count=int(state.get("reset_count", 0)),
                cycle_index=int(state.get("cycle_index", 0)),
            )
        existing_gems = state.get("gems") if isinstance(state.get("gems"), list) else []
        selected_entries = self._select_gem_entries(
            manifest,
            existing_gems=existing_gems,
            completed_gen_id=completed_gen_id,
        )
        if max(len(signature_payload), len(selected_entries)) < min_entries:
            self.save_state(state)
            return GemsTriggerResult(False, reason="insufficient_frontier_entries")
        return self._admit_gems_and_reset(
            state=state,
            manifest=manifest,
            signature_hash=signature_hash,
            completed_gen_id=completed_gen_id,
            entries=selected_entries,
            reason=(
                f"periodic_reset_every_{interval}_generations:"
                f"completed_in_cycle={completed_in_cycle}"
            ),
        )

    def _record_signature_audit(
        self,
        *,
        state: dict[str, Any],
        completed_gen_id: int,
        signature_hash: str,
        signature_payload: list[dict[str, Any]],
    ) -> None:
        """Persist compact frontier signatures for audit only.

        Signature stability is no longer a Gems trigger.  The history remains
        useful for post-run diagnosis and for reconstructing what frontier
        evidence was available at each periodic reset.
        """

        history = state.get("signature_history")
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "generation_id": int(completed_gen_id),
                "signature_hash": signature_hash,
                "entries": signature_payload,
                "recorded_at": datetime.now(UTC).isoformat(),
                "trigger_policy": "periodic",
            }
        )
        state["signature_history"] = history[-50:]
        state["last_signature_hash"] = signature_hash

    def _record_bottleneck_report(
        self,
        *,
        state: dict[str, Any],
        completed_gen_id: int,
        manifest: dict[str, Any],
    ) -> None:
        """Persist advisory Gems opportunity/bottleneck reports.

        These reports are soft priors for PI/Chair prompts. They are not reset
        triggers by themselves, do not create fixed peer quotas, and do not prune
        durable candidate entries.
        """

        try:
            mode = str(getattr(self.config, "bottleneck_detector_mode", "generic") or "generic")
            if mode.strip().lower() in {"disabled", "off", "none", "false", "0"}:
                state["latest_soft_agenda_priors"] = {}
                state["active_bottleneck_reports"] = []
                self._merge_bottlenecks_into_frontier_manifest(state)
                return
            report = ExplorationBottleneckDetector(
                run_dir=self.run_dir,
                mode=mode,
                performance_lanes=self._performance_lanes(),
            ).analyze(
                completed_gen_id=completed_gen_id,
                manifest=manifest,
            )
        except Exception as exc:  # noqa: BLE001 - advisory only.
            logger.warning("gems: bottleneck detector failed: %s", exc)
            state["latest_soft_agenda_priors"] = {}
            state["active_bottleneck_reports"] = []
            self._merge_bottlenecks_into_frontier_manifest(state)
            return
        records = report.get("records") if isinstance(report.get("records"), list) else []
        priors = (
            report.get("soft_agenda_priors")
            if isinstance(report.get("soft_agenda_priors"), dict)
            else {}
        )
        if not records and not priors:
            state["latest_soft_agenda_priors"] = {}
            state["active_bottleneck_reports"] = []
            self._merge_bottlenecks_into_frontier_manifest(state)
            return
        history = state.get("bottleneck_history")
        if not isinstance(history, list):
            history = []
        history.append(report)
        state["bottleneck_history"] = history[-50:]
        state["active_bottleneck_reports"] = [report]
        state["latest_soft_agenda_priors"] = priors or {}
        self._merge_bottlenecks_into_frontier_manifest(state)

    def _merge_bottlenecks_into_frontier_manifest(self, state: dict[str, Any]) -> None:
        manifest = self.frontier_manifest()
        if not isinstance(manifest, dict):
            manifest = {}
        gems_manifest = manifest.get("gems")
        if not isinstance(gems_manifest, dict):
            gems_manifest = {}
        gems_manifest.update(
            {
                "cycle_index": int(state.get("cycle_index", 0)),
                "reset_count": int(state.get("reset_count", 0)),
                "cycle_start_generation": int(state.get("cycle_start_generation", 0)),
                **self._manifest_gems_config(),
                "entries": self._compact_gems(
                    state.get("gems") if isinstance(state.get("gems"), list) else [],
                    sort_by_performance=self._mature_evidence_topk_policy_enabled(),
                    max_generation_id=self._state_source_generation_limit(state),
                    preserve_committed_gems=True,
                ),
                "bottleneck_reports": list(state.get("active_bottleneck_reports", []) or [])[-5:],
                "latest_soft_agenda_priors": state.get("latest_soft_agenda_priors", {}) or {},
                "state_path": str(self.state_path.relative_to(self.run_dir)),
            }
        )
        manifest["gems"] = gems_manifest
        if hasattr(self.frontier, "_manifest"):
            self.frontier._manifest.clear()
            self.frontier._manifest.update(manifest)
            if hasattr(self.frontier, "_save_manifest"):
                self.frontier._save_manifest()
        else:
            _atomic_write_json(
                self.run_dir / "frontier" / "frontier_manifest.json",
                _with_frontier_manifest_semantics(manifest, gems_state_path=self.state_path),
            )

    def recover_pending_reset(self, *, completed_gen_id: int) -> GemsTriggerResult:
        """Roll forward a previously-started Gems reset transaction.

        Gems reset archives ordinary active state before launching a new
        logical generation-0 cycle. That makes crash recovery more sensitive
        than ordinary generation-boundary work: replaying the normal boundary
        after archive/prune can synthesize from half-reset state. A pending
        reset is therefore completed first and exactly once.
        """

        if not self.enabled:
            return GemsTriggerResult(False, reason="disabled")
        state = self.load_state()
        pending = state.get("pending_reset")
        if not isinstance(pending, dict):
            return GemsTriggerResult(False, reason="no_pending_reset")
        try:
            pending_gen = int(pending.get("completed_gen_id", -1))
        except (TypeError, ValueError):
            return GemsTriggerResult(False, reason="malformed_pending_reset")
        if pending_gen != int(completed_gen_id):
            return GemsTriggerResult(
                False,
                reason=f"pending_reset_for_generation_{pending_gen}",
            )
        return self._complete_pending_reset(state=state, pending=pending, recovered=True)

    def _admit_gems_and_reset(
        self,
        *,
        state: dict[str, Any],
        manifest: dict[str, Any],
        signature_hash: str,
        completed_gen_id: int,
        reason: str,
        entries: list[dict[str, Any]] | None = None,
    ) -> GemsTriggerResult:
        existing_gems = state.get("gems") if isinstance(state.get("gems"), list) else []
        if entries is None:
            entries = self._select_gem_entries(
                manifest,
                existing_gems=existing_gems,
                completed_gen_id=completed_gen_id,
            )
        if not entries and not self._compact_gems(
            existing_gems,
            sort_by_performance=self._mature_evidence_topk_policy_enabled(),
            max_generation_id=completed_gen_id,
            preserve_committed_gems=True,
        ):
            return GemsTriggerResult(False, reason="no_gem_candidates")
        reset_count = int(state.get("reset_count", 0)) + 1
        next_cycle_index = int(state.get("cycle_index", 0)) + 1
        timestamp = utc_stamp()
        archive_dir = self.run_dir / "archive" / f"gems_cycle_{reset_count}_{timestamp}"
        archive_dir.mkdir(parents=True, exist_ok=True)
        pending = {
            "status": "pending",
            "reset_count": reset_count,
            "cycle_index": next_cycle_index,
            "completed_gen_id": int(completed_gen_id),
            "next_absolute_generation": int(completed_gen_id) + 1,
            "signature_hash": signature_hash,
            "reason": reason,
            "archive_dir": str(archive_dir),
            "selected_entries": entries,
            "started_at": datetime.now(UTC).isoformat(),
        }
        state["pending_reset"] = pending
        self.save_state(state)
        return self._complete_pending_reset(state=state, pending=pending, recovered=False)

    def _complete_pending_reset(
        self,
        *,
        state: dict[str, Any],
        pending: dict[str, Any],
        recovered: bool,
    ) -> GemsTriggerResult:
        reset_count = int(pending["reset_count"])
        next_cycle_index = int(pending["cycle_index"])
        completed_gen_id = int(pending["completed_gen_id"])
        reason = str(pending.get("reason") or "gems_reset")
        signature_hash = str(pending.get("signature_hash") or "")
        archive_dir = self._safe_archive_dir(Path(str(pending["archive_dir"])))
        archive_dir.mkdir(parents=True, exist_ok=True)
        raw_entries = pending.get("selected_entries")
        entries = (
            [self._with_gems_sort_config(entry) for entry in raw_entries if isinstance(entry, dict)]
            if isinstance(raw_entries, list)
            else []
        )

        def entry_match_keys(entry: dict[str, Any]) -> set[str]:
            keys: set[str] = set()
            for prefix, name in (
                ("finding", "finding_id"),
                ("finding", "source_finding_id"),
                ("variant", "variant_name"),
            ):
                value = str(entry.get(name) or "").strip()
                if value:
                    keys.add(f"{prefix}:{value.lower()}")
            return keys

        def entry_is_recoverable_gem_source(entry: dict[str, Any]) -> bool:
            entry = self._with_gems_sort_config(entry)
            mature_evidence_topk = self._mature_evidence_topk_policy_enabled()
            has_generic_mature_evidence = _entry_has_generic_mature_evidence(
                entry,
                self.maturity_policy,
            )
            if not has_generic_mature_evidence and _entry_is_scout_or_partial(entry):
                return False
            if _entry_has_hard_gem_rejection_marker(entry):
                return False
            if not mature_evidence_topk and _entry_has_nonclean_gem_marker(entry):
                return False
            is_legacy_or_control_anchor = _entry_is_recoverable_legacy_or_control_gem_source(entry)
            is_task_control_anchor = _entry_lane(entry) in self._control_lanes()
            min_cells = self._min_mature_eval_units()
            if (
                not _entry_has_explicit_complete_eval_evidence(entry)
                and not has_generic_mature_evidence
                and not is_legacy_or_control_anchor
                and not is_task_control_anchor
            ):
                return False
            source_gen = _entry_source_generation_id(entry)
            if source_gen is None:
                if not (is_legacy_or_control_anchor or is_task_control_anchor):
                    return False
            elif source_gen > int(completed_gen_id):
                return False
            if is_legacy_or_control_anchor or is_task_control_anchor:
                return not mature_evidence_topk
            if not mature_evidence_topk:
                return True
            return self._is_task_performance_entry(
                entry
            ) and _entry_is_mature_gem_admission_candidate(
                entry,
                min_mature_eval_units=min_cells,
                evidence_stage_min_units=self._evidence_stage_min_units(),
                maturity_policy=self.maturity_policy,
            )

        raw_selected_entries_present = isinstance(raw_entries, list) and bool(raw_entries)
        entries = [entry for entry in entries if entry_is_recoverable_gem_source(entry)]
        allowed_selected_keys = set().union(*(entry_match_keys(entry) for entry in entries))
        pending_records = pending.get("gem_records")
        gem_records: list[dict[str, Any]] = []
        if isinstance(pending_records, list) and pending_records:
            raw_gem_records = [
                self._with_gems_sort_config(record)
                for record in pending_records
                if isinstance(record, dict)
            ]
            gem_records = [
                record
                for record in raw_gem_records
                if entry_is_recoverable_gem_source(record)
                and (
                    (
                        not raw_selected_entries_present
                        and _boolish_entry_field(record, "source_is_existing_gem") is True
                    )
                    or bool(entry_match_keys(record) & allowed_selected_keys)
                )
            ]
        existing_gems = self._compact_gems(
            state.get("gems") if isinstance(state.get("gems"), list) else [],
            sort_by_performance=self._mature_evidence_topk_policy_enabled(),
            max_generation_id=completed_gen_id,
            allow_legacy_unknown_source=True,
            preserve_committed_gems=True,
        )
        if not entries and not existing_gems and not gem_records:
            state.pop("pending_reset", None)
            self.save_state(state)
            return GemsTriggerResult(False, reason="pending_reset_without_entries")
        if not gem_records:
            if entries:
                for rank, entry in enumerate(entries, start=1):
                    gem = self._write_gem_finding(
                        entry=entry,
                        rank=rank,
                        reset_count=reset_count,
                        next_cycle_index=next_cycle_index,
                        completed_gen_id=completed_gen_id,
                        reason=reason,
                    )
                    gem_records.append(gem)
            pending["gem_records"] = gem_records
            pending["gem_findings_written_at"] = datetime.now(UTC).isoformat()
            state["pending_reset"] = pending
            self.save_state(state)

        all_gems = self._compact_gems(
            [*existing_gems, *gem_records],
            sort_by_performance=True,
            preserve_lane_reserves=True,
            max_generation_id=completed_gen_id,
            allow_legacy_unknown_source=True,
            preserve_committed_gems=True,
        )
        final_gem_ids = {
            str(gem.get("gem_finding_id")) for gem in all_gems if gem.get("gem_finding_id")
        }
        admitted_gem_records = [
            gem for gem in gem_records if str(gem.get("gem_finding_id") or "") in final_gem_ids
        ]
        keep_ids = {str(g.get("gem_finding_id")) for g in all_gems if g.get("gem_finding_id")}
        keep_ids.update(
            str(g.get("source_finding_id"))
            for g in all_gems
            if g.get("source_is_existing_gem") and g.get("source_finding_id")
        )
        keep_path_map: dict[str, set[str]] = {}
        for gem in all_gems:
            if not isinstance(gem, dict):
                continue
            gem_id = str(gem.get("gem_finding_id") or "")
            finding_path = str(gem.get("finding_path") or "")
            if gem_id and finding_path:
                keep_path_map.setdefault(gem_id, set()).add(finding_path)
        missing_path_gems = {
            str(gem.get("gem_finding_id")): gem
            for gem in all_gems
            if isinstance(gem, dict)
            and gem.get("gem_finding_id")
            and str(gem.get("gem_finding_id")) not in keep_path_map
        }
        if missing_path_gems:
            keep_path_map.update(self._infer_legacy_gem_paths_by_id(missing_path_gems))
        known_gem_ids = {
            str(gem.get("gem_finding_id"))
            for gem in [*existing_gems, *gem_records, *all_gems]
            if isinstance(gem, dict) and gem.get("gem_finding_id")
        }
        raw_state_gems = state.get("gems") if isinstance(state.get("gems"), list) else []
        known_gem_ids.update(
            str(gem.get("gem_finding_id"))
            for gem in raw_state_gems
            if isinstance(gem, dict) and gem.get("gem_finding_id")
        )

        if bool(getattr(self.config, "archive_ordinary_findings", True)) and not pending.get(
            "archive_complete"
        ):
            self._archive_and_prune_active_context(
                archive_dir=archive_dir,
                keep_finding_ids=keep_ids,
                keep_finding_paths_by_id=keep_path_map,
            )
            pending["archive_complete"] = True
            pending["archive_completed_at"] = datetime.now(UTC).isoformat()
            state["pending_reset"] = pending
            self.save_state(state)
        self._prune_superseded_gem_findings(
            archive_dir=archive_dir,
            final_gem_ids=final_gem_ids,
            final_gem_paths_by_id=keep_path_map,
            known_gem_ids=known_gem_ids,
        )

        self._write_gem_edges(admitted_gem_records)
        committed_state = dict(state)
        committed_state["reset_count"] = reset_count
        committed_state["cycle_index"] = next_cycle_index
        committed_state["cycle_start_generation"] = int(completed_gen_id) + 1
        committed_state["last_signature_hash"] = ""
        committed_state["last_bottleneck_signature"] = ""
        committed_state["last_bottleneck_generation"] = None
        committed_state["gems"] = all_gems
        events = state.get("reset_events")
        if not isinstance(events, list):
            events = []
        events.append(
            {
                "reset_count": reset_count,
                "cycle_index": next_cycle_index,
                "completed_gen_id": int(completed_gen_id),
                "next_absolute_generation": int(completed_gen_id) + 1,
                "signature_hash": signature_hash,
                "reason": reason,
                "admitted_gems": len(admitted_gem_records),
                "archive_dir": str(archive_dir),
                "committed": True,
                "recovered": bool(recovered),
                "recorded_at": datetime.now(UTC).isoformat(),
            }
        )
        committed_state["reset_events"] = events
        committed_state.pop("pending_reset", None)
        # The frontier manifest is rewritten before the final committed state.
        # If the process dies here, resume sees `pending_reset` and rolls the
        # same transaction forward again instead of taking the ordinary PI path.
        self._merge_gems_into_frontier_manifest(all_gems, committed_state)
        self.save_state(committed_state)
        logger.info(
            "gems: admitted %d Gems and reset logical cycle to gen0 "
            "(reset_count=%d, next_abs_gen=%d)",
            len(admitted_gem_records),
            reset_count,
            int(completed_gen_id) + 1,
        )
        return GemsTriggerResult(
            True,
            reason=reason,
            reset_count=reset_count,
            cycle_index=next_cycle_index,
            admitted_count=len(admitted_gem_records),
            archive_dir=str(archive_dir),
        )

    def _write_gem_finding(
        self,
        *,
        entry: dict[str, Any],
        rank: int,
        reset_count: int,
        next_cycle_index: int,
        completed_gen_id: int,
        reason: str,
    ) -> dict[str, Any]:
        entry = _sanitize_nonfinite_json(entry)
        source_finding_id = str(entry.get("finding_id") or "")
        variant = str(entry.get("variant_name") or source_finding_id or f"gem_{rank}")
        gem_id = f"gem_r{reset_count:02d}_{rank:02d}_{self._slug(variant)[:40]}"
        metrics = dict(entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {})
        lane = str(
            entry.get("promoted_for_lane")
            or entry.get("frontier_lane")
            or metrics.get("frontier_lane")
            or ""
        )
        metric_name = str(entry.get("lane_metric_name") or entry.get("metric_name") or "")
        metric_value = entry.get("lane_metric_value", entry.get("metric_value"))
        if metric_value in (None, "", [], {}):
            metric_value, metric_name = _metric_float_with_key(
                entry,
                *self._configured_metric_keys(
                    "primary_metric_keys", _GENERIC_GEM_PRIMARY_METRIC_KEYS
                ),
            )
            if metric_name == "primary_metric_value":
                configured_keys = self._configured_metric_keys("primary_metric_keys", ())
                for key in configured_keys:
                    if _metric_float_if_present(entry, key) is not None:
                        metric_name = key
                        break
        metric_direction = _entry_metric_direction(
            entry,
            "lane_metric_value"
            if entry.get("lane_metric_value") not in (None, "", [], {})
            else metric_name,
        )
        snapshot_path = entry.get("snapshot_path")
        gem_variant_ref = self._copy_snapshot_if_available(
            snapshot_path=snapshot_path,
            gem_id=gem_id,
        )
        existing_is_gem = bool(metrics.get("is_gem_finding"))
        content = self._gem_content(
            entry=entry,
            variant=variant,
            lane=lane,
            metric_name=metric_name,
            metric_value=metric_value,
            reset_count=reset_count,
            next_cycle_index=next_cycle_index,
            completed_gen_id=completed_gen_id,
            reason=reason,
            gem_variant_ref=gem_variant_ref,
        )
        metrics.update(
            {
                "is_gem_finding": True,
                "gem_reset_count": reset_count,
                "gem_cycle_index": next_cycle_index,
                "gem_rank": rank,
                "source_finding_id": source_finding_id,
                "source_generation_id": entry.get("generation_id"),
                "frontier_lane": lane or metrics.get("frontier_lane", ""),
                "metric_name": metric_name,
                "metric_value": _safe_metric(metric_value),
                "metric_direction": metric_direction,
                "gem_reason": reason,
                "excluded_from_durable_frontier": entry.get(
                    "excluded_from_durable_frontier",
                    metrics.get("excluded_from_durable_frontier"),
                ),
                "exclusion_reason": entry.get(
                    "exclusion_reason",
                    metrics.get("exclusion_reason"),
                ),
            }
        )
        finding = {
            "id": gem_id,
            "finding_type": "result",
            "title": f"GEM {reset_count}.{rank}: {variant}",
            "content": content,
            "summary": content[:500],
            "metrics": metrics,
            "variant_name": variant,
            "notes": (
                "Durable Gems finding. This is intentionally more detailed "
                "than an ordinary finding and is preserved across logical "
                "generation-0 resets."
            ),
            "peer_id": "gems_agent",
            # Keep Gems visible to longitudinal summaries after ordinary
            # active findings are archived.
            "generation_id": 0,
            "timestamp": datetime.now(UTC).isoformat(),
            "gem_variant_ref": gem_variant_ref,
            "source_frontier_entry": entry,
        }
        self.findings_dir = self._safe_child_dir("shared_findings")
        self.findings_dir.mkdir(parents=True, exist_ok=True)
        path = self.findings_dir / f"{gem_id}_{self._slug(variant)[:60]}.json"
        _atomic_write_json(path, finding)
        try:
            from praxist.plugins.workflow_stages.research_loop.backend.tools.local_store import (
                insert_finding,
            )

            insert_finding(finding)
        except Exception as exc:  # noqa: BLE001 - filesystem copy still stands.
            logger.warning(
                "gems: failed to insert gem finding %s into local store: %s", gem_id, exc
            )
        sort_entry = self._with_gems_sort_config(entry)
        primary_score = _metric_float(
            sort_entry,
            *_entry_metric_key_list(
                sort_entry,
                "_gems_primary_metric_keys",
                _GENERIC_GEM_PRIMARY_METRIC_KEYS,
            ),
        )
        admission_metrics = {
            "primary_score": primary_score,
            "secondary_score": _metric_float(
                sort_entry,
                *_entry_metric_key_list(
                    sort_entry,
                    "_gems_secondary_metric_keys",
                    _GENERIC_GEM_SECONDARY_METRIC_KEYS,
                ),
            ),
            "lower_tail_score": _metric_float(
                sort_entry,
                *_entry_metric_key_list(
                    sort_entry,
                    "_gems_lower_tail_metric_keys",
                    _GENERIC_GEM_LOWER_TAIL_METRIC_KEYS,
                ),
            ),
            "validation_score": _metric_float(
                sort_entry,
                *_entry_metric_key_list(
                    sort_entry,
                    "_gems_validation_metric_keys",
                    _GENERIC_GEM_VALIDATION_METRIC_KEYS,
                ),
            ),
            "risk_score": _metric_float(
                sort_entry,
                *_entry_metric_key_list(
                    sort_entry,
                    "_gems_cost_metric_keys",
                    _GENERIC_GEM_RISK_METRIC_KEYS,
                ),
                default=10_000.0,
            ),
            "evidence_rank": _evidence_rank(entry),
            "evaluation_units": _entry_eval_unit_count(entry),
            "tier": entry.get("tier") or metrics.get("tier") or "",
            "tier_status": entry.get("tier_status") or metrics.get("tier_status") or "",
            "final_status": entry.get("final_status") or metrics.get("final_status") or "",
            "result_status": entry.get("result_status") or metrics.get("result_status") or "",
            "metric_name": metric_name,
            "metric_value": _safe_metric(metric_value),
            "metric_direction": metric_direction,
            "completion_status": entry.get("completion_status")
            or metrics.get("completion_status")
            or "",
            "eval_status": entry.get("eval_status") or metrics.get("eval_status") or "",
            "scout_only": entry.get("scout_only", metrics.get("scout_only")),
            "is_scout_eval": entry.get("is_scout_eval", metrics.get("is_scout_eval")),
            "is_smoke_eval": entry.get("is_smoke_eval", metrics.get("is_smoke_eval")),
            "summary_only": entry.get("summary_only", metrics.get("summary_only")),
            "is_summary_only": entry.get("is_summary_only", metrics.get("is_summary_only")),
            "partial_cohort": entry.get("partial_cohort", metrics.get("partial_cohort")),
            "partial_eval": entry.get("partial_eval", metrics.get("partial_eval")),
            "is_partial_eval": entry.get("is_partial_eval", metrics.get("is_partial_eval")),
            "unscored_artifact": entry.get("unscored_artifact", metrics.get("unscored_artifact")),
            "incomplete_eval": entry.get("incomplete_eval", metrics.get("incomplete_eval")),
            "scored_complete": entry.get("scored_complete", metrics.get("scored_complete")),
            "is_scored_complete": entry.get(
                "is_scored_complete",
                metrics.get("is_scored_complete"),
            ),
            "complete_eval": entry.get("complete_eval", metrics.get("complete_eval")),
            "is_complete_eval": entry.get("is_complete_eval", metrics.get("is_complete_eval")),
            "capped": entry.get("capped", metrics.get("capped")),
            "is_capped": entry.get("is_capped", metrics.get("is_capped")),
            "result_capped": entry.get("result_capped", metrics.get("result_capped")),
            "promotion_eligible": entry.get(
                "promotion_eligible",
                metrics.get("promotion_eligible"),
            ),
            "excluded_from_durable_frontier": entry.get(
                "excluded_from_durable_frontier",
                metrics.get("excluded_from_durable_frontier"),
            ),
            "exclusion_reason": entry.get(
                "exclusion_reason",
                metrics.get("exclusion_reason"),
            ),
            "clean_promotion_eligible": entry.get(
                "clean_promotion_eligible",
                metrics.get("clean_promotion_eligible"),
            ),
            "n_hard_constraint_violations": entry.get(
                "n_hard_constraint_violations",
                metrics.get("n_hard_constraint_violations"),
            ),
            "gems_selection_policy": str(
                getattr(self.config, "selection_policy", "frontier_lane_balanced")
                or "frontier_lane_balanced"
            ),
        }
        optional_admission_metrics = {
            key: _metric_float_if_present(entry, key)
            for key in self._all_configured_gem_metric_keys()
        }
        admission_metrics.update(
            {key: value for key, value in optional_admission_metrics.items() if value is not None}
        )
        admission_metrics.update(
            {
                key: value
                for key, value in compact_maturity_metadata(
                    sort_entry, self.maturity_policy
                ).items()
                if value is not None
            }
        )
        declared_metric = str(
            entry.get("metric_name")
            or metrics.get("metric_name")
            or entry.get("lane_metric_name")
            or metrics.get("lane_metric_name")
            or ""
        ).strip()
        if declared_metric and declared_metric not in admission_metrics:
            declared_value = _metric_float_if_present(entry, declared_metric)
            if declared_value is not None:
                admission_metrics[declared_metric] = declared_value
        for key in _DIVERSITY_METADATA_KEYS:
            value = entry.get(key, metrics.get(key))
            if value not in (None, "", [], {}):
                admission_metrics[key] = _sanitize_nonfinite_json(value)
        record = {
            "gem_finding_id": gem_id,
            "variant_name": variant,
            "source_finding_id": source_finding_id,
            "source_generation_id": _entry_source_generation_id(entry),
            "source_is_existing_gem": existing_is_gem,
            "frontier_lane": lane,
            "strategy_family": metrics.get("strategy_family") or metrics.get("family") or "",
            "mechanism_family": metrics.get("mechanism_family") or "",
            "innovation_surface": metrics.get("innovation_surface") or "",
            **{key: metrics.get(key) or entry.get(key) or "" for key in _RESEARCH_METADATA_KEYS},
            **{key: metrics.get(key) or entry.get(key) or "" for key in _DIVERSITY_METADATA_KEYS},
            "metric_name": metric_name,
            "metric_value": _safe_metric(metric_value),
            "metric_direction": metric_direction,
            "admission_metrics": admission_metrics,
            "gem_variant_ref": gem_variant_ref,
            "finding_path": str(path.relative_to(self.run_dir)),
            "admitted_at": datetime.now(UTC).isoformat(),
        }
        for key in (
            *EFFECTIVE_CONFIG_METADATA_KEYS,
            "source_result_path",
            "source_result_sha256",
            "result_path",
            "result_artifact_path",
            "child_id",
            "sweep_child_id",
            "result_variant_id",
            "child_variant_name",
            "child_variant_id",
            "result_variant_name",
            "canonical_variant_id",
            "canonical_variant_name",
            "frontier_entity_key",
            "candidate_entity_key",
            "variant_id",
        ):
            value = _entry_field(entry, key)
            if value not in (None, "", [], {}):
                record[key] = _sanitize_nonfinite_json(value)
        for container_name in ("details", "extra", "current_aggregate"):
            compact_identity = _compact_result_identity_container(entry.get(container_name))
            if compact_identity:
                record[container_name] = _sanitize_nonfinite_json(compact_identity)
        return record

    def _gem_content(
        self,
        *,
        entry: dict[str, Any],
        variant: str,
        lane: str,
        metric_name: str,
        metric_value: Any,
        reset_count: int,
        next_cycle_index: int,
        completed_gen_id: int,
        reason: str,
        gem_variant_ref: str,
    ) -> str:
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        metrics_lines = []
        for key in sorted(metrics):
            value = metrics[key]
            if isinstance(value, (dict, list)):
                text = json.dumps(value, default=str)[:500]
            else:
                text = str(value)
            metrics_lines.append(f"- `{key}`: {text}")
        if not metrics_lines:
            metrics_lines.append("- No structured metrics were present in the source entry.")
        source_gen = entry.get("generation_id")
        source_id = entry.get("finding_id")
        return "\n".join(
            [
                f"# Gems Finding: {variant}",
                "",
                "## Admission Context",
                (
                    f"This variant was admitted into the durable Gems set after generation "
                    f"{completed_gen_id} because the configured periodic Gems reset fired: "
                    f"{reason}. The system is starting Gems cycle {next_cycle_index} "
                    f"(reset #{reset_count}) from preserved high-confidence evidence rather "
                    "than continuing to narrow around transient ordinary findings. For tasks "
                    "using mature-evidence top-k Gems selection, admission requires actual "
                    "task-defined mature evaluation evidence."
                ),
                "",
                "## Pareto Role",
                (
                    f"- Source generation: `{source_gen}`\n"
                    f"- Source finding id: `{source_id}`\n"
                    f"- Frontier lane: `{lane or '(unspecified)'}`\n"
                    f"- Selection metric: `{metric_name or '(primary/lane metric)'}` = `{metric_value}`\n"
                    f"- Frozen variant artifact: `{gem_variant_ref or '(no snapshot available)'}`"
                ),
                "",
                "## Why This Gem Matters",
                (
                    "This finding should be treated as a durable research anchor. It is not "
                    "merely a scout or capped result; it represents a compact, high-confidence "
                    "restart parent selected from eligible mature evidence. Future peers should preserve the useful "
                    "mechanism, compare against the metric profile below, and design variants "
                    "that either extend its strength, repair its weakness, or combine it with "
                    "another Gem. Do not re-run generic copies of the same idea without a "
                    "declared mechanism change and falsification plan."
                ),
                "",
                "## Evidence and Measurements",
                *metrics_lines,
                "",
                "## Recommended Follow-Up Work",
                (
                    "1. Reproduce the key effect using the task's current evaluation protocol before "
                    "making a clean promotion claim.\n"
                    "2. Run a targeted ablation that removes the mechanism believed to create the "
                    "advantage.\n"
                    "3. If this is a durable or repair candidate, keep the beneficial metric "
                    "movement while directly addressing the stated risk, evidence, or constraint gap.\n"
                    "4. If this Gem is a reference/diagnostic/process anchor, use it to bound or "
                    "falsify task-performance claims rather than letting it crowd the primary "
                    "candidate lane."
                ),
                "",
                "## Provenance",
                (
                    "This Gems finding was generated by the orchestrator, not by a peer. It is "
                    "deliberately detailed so the next logical generation-0 cohort can restart "
                    "exploration from compact, durable, high-value evidence while ordinary prior "
                    "findings are archived out of active context."
                ),
            ]
        )

    def _copy_snapshot_if_available(self, *, snapshot_path: Any, gem_id: str) -> str:
        if not snapshot_path:
            return ""
        src = Path(str(snapshot_path))
        if not src.exists() or not src.is_file():
            return str(snapshot_path)
        dest_dir = self._safe_child_dir("gems") / "variants"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{gem_id}{src.suffix or '.snapshot'}"
        try:
            shutil.copy2(src, dest)
            return str(dest.relative_to(self.run_dir))
        except OSError as exc:
            logger.warning("gems: could not copy snapshot %s to %s: %s", src, dest, exc)
            return str(snapshot_path)

    def _archive_and_prune_active_context(
        self,
        *,
        archive_dir: Path,
        keep_finding_ids: set[str],
        keep_finding_paths_by_id: dict[str, set[str]],
    ) -> None:
        archive_dir = self._safe_archive_dir(archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        self._archive_sqlite_rows(archive_dir=archive_dir, keep_finding_ids=keep_finding_ids)
        self._archive_shared_findings(
            archive_dir=archive_dir,
            keep_finding_ids=keep_finding_ids,
            keep_finding_paths_by_id=keep_finding_paths_by_id,
        )
        # Remove old agendas so the next logical gen0 is not driven by a
        # stale PI contract synthesized from archived ordinary findings.
        agendas = self._safe_child_dir("agendas")
        if agendas.exists():
            target, target_fd = self._open_archive_child_dir_fd(archive_dir, "agendas")
            try:
                for path in agendas.glob("research_agenda_gen*.yaml"):
                    self._move_file_to_open_archive_dir(
                        path,
                        dst_dir=target,
                        dst_dir_fd=target_fd,
                    )
            finally:
                os.close(target_fd)
        for memory_dir_name in ("research_memory", "memory"):
            src = self._safe_child_dir(memory_dir_name)
            if not src.exists():
                continue
            dst = self._safe_archive_child_dir(archive_dir, memory_dir_name)
            if dst.exists():
                self._merge_directory_into_archive(src=src, dst=dst)
            else:
                shutil.move(str(src), str(dst))
            (self.run_dir / memory_dir_name).mkdir(parents=True, exist_ok=True)
        frontier_manifest = self.run_dir / "frontier" / "frontier_manifest.json"
        if frontier_manifest.exists():
            dest = self._safe_archive_file_path(
                archive_dir / "frontier_manifest_before_gems_reset.json"
            )
            self._copy_archive_file_once(frontier_manifest, dest)

    def _require_inside_run_dir(
        self, path: Path, *, label: str, allow_missing: bool = False
    ) -> Path:
        resolved_run = self.run_dir.resolve()
        resolved = path.resolve(strict=not allow_missing)
        try:
            resolved.relative_to(resolved_run)
        except ValueError as exc:
            raise RuntimeError(f"gems: {label} escapes run_dir: {path}") from exc
        return resolved

    def _safe_archive_dir(self, archive_dir: Path) -> Path:
        resolved_run = self.run_dir.resolve()
        archive_root = (self.run_dir / "archive").resolve()
        resolved = archive_dir.resolve(strict=False)
        try:
            resolved.relative_to(archive_root)
            resolved.relative_to(resolved_run)
        except ValueError as exc:
            raise RuntimeError(f"gems: archive_dir escapes run archive: {archive_dir}") from exc
        return archive_dir

    def _safe_archive_child_dir(self, archive_dir: Path, name: str) -> Path:
        archive_dir = self._safe_archive_dir(archive_dir)
        child = archive_dir / name
        resolved_archive = archive_dir.resolve(strict=False)
        resolved = child.resolve(strict=False)
        try:
            resolved.relative_to(resolved_archive)
            resolved.relative_to(self.run_dir.resolve())
        except ValueError as exc:
            raise RuntimeError(f"gems: archive child {name} escapes archive_dir: {child}") from exc
        if child.is_symlink():
            raise RuntimeError(f"gems: refusing symlinked archive child {name}: {child}")
        return child

    def _open_dir_fd_no_follow(self, path: Path, *, label: str) -> int:
        self._require_inside_run_dir(path, label=label)
        if path.is_symlink():
            raise RuntimeError(f"gems: refusing symlinked {label}: {path}")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        if nofollow:
            flags |= nofollow
        try:
            return os.open(path, flags)
        except OSError as exc:
            raise RuntimeError(f"gems: could not open {label} {path}: {exc}") from exc

    def _open_archive_child_dir_fd(self, archive_dir: Path, name: str) -> tuple[Path, int]:
        child = self._safe_archive_child_dir(archive_dir, name)
        archive_dir = self._safe_archive_dir(archive_dir)
        archive_fd = self._open_dir_fd_no_follow(archive_dir, label="archive_dir")
        try:
            with contextlib.suppress(FileExistsError):
                os.mkdir(name, dir_fd=archive_fd)
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            if nofollow:
                flags |= nofollow
            try:
                child_fd = os.open(name, flags, dir_fd=archive_fd)
            except OSError as exc:
                raise RuntimeError(f"gems: could not open archive child {child}: {exc}") from exc
        finally:
            os.close(archive_fd)
        return child, child_fd

    def _move_file_to_open_archive_dir(
        self,
        path: Path,
        *,
        dst_dir: Path,
        dst_dir_fd: int,
    ) -> None:
        self._require_inside_run_dir(path, label="archive source file")
        if path.is_symlink():
            raise RuntimeError(f"gems: refusing symlinked archive source file: {path}")
        self._require_inside_run_dir(dst_dir, label="archive destination dir")
        if dst_dir.is_symlink():
            raise RuntimeError(f"gems: refusing symlinked archive destination dir: {dst_dir}")
        src_fd = self._open_dir_fd_no_follow(path.parent, label="archive source parent")
        try:
            try:
                os.stat(path.name, dir_fd=src_fd, follow_symlinks=False)
            except FileNotFoundError:
                return
            try:
                os.stat(path.name, dir_fd=dst_dir_fd, follow_symlinks=False)
                os.unlink(path.name, dir_fd=src_fd)
                return
            except FileNotFoundError:
                pass
            try:
                os.rename(path.name, path.name, src_dir_fd=src_fd, dst_dir_fd=dst_dir_fd)
            except OSError as exc:
                raise RuntimeError(
                    f"gems: could not archive file {path} to {dst_dir}: {exc}"
                ) from exc
        finally:
            os.close(src_fd)

    def _safe_archive_file_path(self, path: Path) -> Path:
        archive_root = (self.run_dir / "archive").resolve()
        if path.is_symlink():
            raise RuntimeError(f"gems: refusing symlinked archive file: {path}")
        if path.parent.is_symlink():
            raise RuntimeError(f"gems: refusing symlinked archive file parent: {path.parent}")
        resolved = path.resolve(strict=False)
        resolved_parent = path.parent.resolve(strict=False)
        try:
            resolved.relative_to(archive_root)
            resolved.relative_to(resolved_parent)
            resolved.relative_to(self.run_dir.resolve())
        except ValueError as exc:
            raise RuntimeError(f"gems: archive file escapes run archive: {path}") from exc
        return path

    def _open_archive_output_once(self, path: Path) -> int | None:
        path = self._safe_archive_file_path(path)
        if path.exists():
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        if nofollow:
            parent_flags |= nofollow
        try:
            parent_fd = os.open(path.parent, parent_flags)
        except OSError as exc:
            raise RuntimeError(
                f"gems: could not open archive directory {path.parent}: {exc}"
            ) from exc
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if nofollow:
                flags |= nofollow
            try:
                return os.open(path.name, flags, 0o644, dir_fd=parent_fd)
            except FileExistsError as exc:
                if path.is_symlink():
                    raise RuntimeError(f"gems: refusing symlinked archive file: {path}") from exc
                return None
            except OSError as exc:
                raise RuntimeError(f"gems: could not open archive file {path}: {exc}") from exc
        finally:
            os.close(parent_fd)

    def _copy_archive_file_once(self, src: Path, dest: Path) -> None:
        self._require_inside_run_dir(src, label="archive source file")
        if src.is_symlink():
            raise RuntimeError(f"gems: refusing symlinked archive source file: {src}")
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        flags_in = os.O_RDONLY | nofollow
        try:
            fd_in = os.open(src, flags_in)
        except OSError as exc:
            raise RuntimeError(f"gems: could not open archive source file {src}: {exc}") from exc
        fd_out: int | None = None
        try:
            fd_out = self._open_archive_output_once(dest)
            if fd_out is None:
                os.close(fd_in)
                return
            with os.fdopen(fd_in, "rb") as input_handle, os.fdopen(fd_out, "wb") as output_handle:
                fd_in = -1
                fd_out = -1
                shutil.copyfileobj(input_handle, output_handle)
        except Exception:
            if fd_in >= 0:
                os.close(fd_in)
            if fd_out is not None and fd_out >= 0:
                os.close(fd_out)
            try:
                safe_dest = self._safe_archive_file_path(dest)
                if safe_dest.exists() and not safe_dest.is_symlink():
                    safe_dest.unlink()
            except Exception:
                pass
            raise

    def _safe_child_dir(self, name: str) -> Path:
        path = self.run_dir / name
        self._require_inside_run_dir(path, label=name, allow_missing=True)
        if path.is_symlink():
            raise RuntimeError(f"gems: refusing to archive symlinked {name}: {path}")
        return path

    def _safe_state_path(self) -> Path:
        return self._safe_child_dir("gems") / "gems_state.json"

    def _safe_shared_store_path(self) -> Path:
        db_path = self.run_dir / "shared_store.db"
        self._require_inside_run_dir(db_path, label="shared_store.db", allow_missing=True)
        if db_path.is_symlink():
            raise RuntimeError(f"gems: refusing symlinked shared_store.db: {db_path}")
        return db_path

    def _merge_directory_into_archive(self, *, src: Path, dst: Path) -> None:
        self._require_inside_run_dir(dst, label="archive destination", allow_missing=True)
        if dst.is_symlink():
            raise RuntimeError(f"gems: refusing symlinked archive destination: {dst}")
        dst.mkdir(parents=True, exist_ok=True)
        for child in list(src.iterdir()):
            self._require_inside_run_dir(child, label="archive source child")
            if child.is_symlink():
                raise RuntimeError(f"gems: refusing symlinked archive child: {child}")
            dest = dst / child.name
            self._require_inside_run_dir(
                dest, label="archive destination child", allow_missing=True
            )
            if dest.is_symlink():
                raise RuntimeError(f"gems: refusing symlinked archive destination child: {dest}")
            if child.is_dir():
                self._merge_directory_into_archive(src=child, dst=dest)
            elif dest.exists():
                child.unlink()
            else:
                shutil.move(str(child), str(dest))
        shutil.rmtree(src)

    def _archive_sqlite_rows(self, *, archive_dir: Path, keep_finding_ids: set[str]) -> None:
        db_path = self._safe_shared_store_path()
        if not db_path.exists():
            return
        try:
            with sqlite3.connect(db_path, timeout=30) as conn:
                conn.row_factory = sqlite3.Row
                findings = [dict(r) for r in conn.execute("SELECT * FROM findings").fetchall()]
                metrics = [dict(r) for r in conn.execute("SELECT * FROM metrics").fetchall()]
                edges = [dict(r) for r in conn.execute("SELECT * FROM finding_edges").fetchall()]
                self._write_jsonl_once(archive_dir / "findings_before_archive.jsonl", findings)
                self._write_jsonl_once(archive_dir / "metrics_before_archive.jsonl", metrics)
                self._write_jsonl_once(
                    archive_dir / "finding_edges_before_archive.jsonl",
                    edges,
                )
                if keep_finding_ids:
                    placeholders = ",".join("?" for _ in keep_finding_ids)
                    params = tuple(keep_finding_ids)
                    conn.execute(f"DELETE FROM findings WHERE id NOT IN ({placeholders})", params)
                    conn.execute("DELETE FROM metrics")
                    conn.execute(
                        f"DELETE FROM finding_edges WHERE "
                        f"src_finding_id NOT IN ({placeholders}) "
                        f"OR dst_finding_id NOT IN ({placeholders})",
                        params + params,
                    )
                else:
                    conn.execute("DELETE FROM findings")
                    conn.execute("DELETE FROM metrics")
                    conn.execute("DELETE FROM finding_edges")
        except sqlite3.Error as exc:
            raise RuntimeError(f"gems: SQLite archive/prune failed: {exc}") from exc

    def _archive_shared_findings(
        self,
        *,
        archive_dir: Path,
        keep_finding_ids: set[str],
        keep_finding_paths_by_id: dict[str, set[str]],
    ) -> None:
        archive_dir = self._safe_archive_dir(archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        self.findings_dir = self._safe_child_dir("shared_findings")
        if not self.findings_dir.exists():
            return
        target, target_fd = self._open_archive_child_dir_fd(archive_dir, "shared_findings")
        try:
            for path in sorted(self.findings_dir.glob("*.json")):
                self._require_inside_run_dir(path, label="shared finding")
                if path.is_symlink():
                    path.unlink(missing_ok=True)
                    continue
                rel_path = ""
                try:
                    rel_path = str(path.relative_to(self.run_dir))
                except ValueError:
                    rel_path = ""
                keep = False
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    fid = str(data.get("id") or "")
                    canonical_paths = keep_finding_paths_by_id.get(fid, set())
                    keep = fid in keep_finding_ids and (
                        not fid.startswith("gem_")
                        or not canonical_paths
                        or rel_path in canonical_paths
                    )
                except Exception:
                    keep = False
                if keep:
                    continue
                self._move_file_to_open_archive_dir(
                    path,
                    dst_dir=target,
                    dst_dir_fd=target_fd,
                )
        finally:
            os.close(target_fd)

    def _infer_legacy_gem_paths_by_id(
        self,
        gems_by_id: dict[str, dict[str, Any]],
    ) -> dict[str, set[str]]:
        if not gems_by_id:
            return {}
        findings_dir = self._safe_child_dir("shared_findings")
        if not findings_dir.exists():
            return {}
        candidates: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for path in sorted(findings_dir.glob("*.json")):
            self._require_inside_run_dir(path, label="shared finding")
            if path.is_symlink():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            fid = str(data.get("id") or "")
            if fid not in gems_by_id:
                continue
            metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
            expected_variant = str(gems_by_id[fid].get("variant_name") or "").strip()
            expected_slug = self._slug(expected_variant) if expected_variant else ""
            score = 0
            if str(data.get("peer_id") or "") == "gems_agent":
                score += 4
            if bool(metrics.get("is_gem_finding")):
                score += 2
            if str(data.get("title") or "").upper().startswith("GEM"):
                score += 1
            if expected_variant and str(data.get("variant_name") or "") == expected_variant:
                score += 4
            if expected_variant and expected_variant in str(data.get("title") or ""):
                score += 2
            try:
                rel_path = str(path.relative_to(self.run_dir))
            except ValueError:
                continue
            if expected_slug and expected_slug in path.name:
                score += 2
            candidates.setdefault(fid, []).append((score, rel_path))
        inferred: dict[str, set[str]] = {}
        for fid, scored_paths in candidates.items():
            scored_paths.sort(reverse=True)
            if not scored_paths or scored_paths[0][0] <= 0:
                inferred[fid] = {"__ambiguous_legacy_gem_path__"}
                continue
            top_score = scored_paths[0][0]
            top_paths = [rel for score, rel in scored_paths if score == top_score]
            if len(top_paths) == 1:
                inferred[fid] = {top_paths[0]}
            else:
                inferred[fid] = {"__ambiguous_legacy_gem_path__"}
        return inferred

    def _prune_superseded_gem_findings(
        self,
        *,
        archive_dir: Path,
        final_gem_ids: set[str],
        final_gem_paths_by_id: dict[str, set[str]],
        known_gem_ids: set[str],
    ) -> None:
        """Remove active Gem sidecars that are no longer in the durable set.

        A reset can write more candidate Gem findings than survive the tiny
        global cap after performance sorting. Those superseded files must leave
        active `shared_findings`; otherwise local reverse-sync can re-ingest
        them into SQLite during resume and make old Gems visible again.
        """

        self.findings_dir = self._safe_child_dir("shared_findings")
        if self.findings_dir.exists():
            target, target_fd = self._open_archive_child_dir_fd(
                archive_dir,
                "superseded_gem_findings",
            )
            try:
                for path in sorted(self.findings_dir.glob("*.json")):
                    self._require_inside_run_dir(path, label="shared finding")
                    if path.is_symlink():
                        path.unlink(missing_ok=True)
                        continue
                    rel_path = ""
                    try:
                        rel_path = str(path.relative_to(self.run_dir))
                    except ValueError:
                        rel_path = ""
                    try:
                        data = json.loads(path.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    fid = str(data.get("id") or "")
                    if fid not in known_gem_ids:
                        continue
                    canonical_paths = final_gem_paths_by_id.get(fid, set())
                    if fid in final_gem_ids and (
                        not canonical_paths or rel_path in canonical_paths
                    ):
                        continue
                    self._move_file_to_open_archive_dir(
                        path,
                        dst_dir=target,
                        dst_dir_fd=target_fd,
                    )
            finally:
                os.close(target_fd)

        db_path = self._safe_shared_store_path()
        if not db_path.exists():
            return
        stale_candidates = tuple(sorted(known_gem_ids - final_gem_ids))
        if not stale_candidates:
            return
        try:
            with sqlite3.connect(db_path, timeout=30) as conn:
                conn.row_factory = sqlite3.Row
                placeholders = ",".join("?" for _ in stale_candidates)
                rows = [
                    dict(row)
                    for row in conn.execute(
                        f"SELECT * FROM findings WHERE id IN ({placeholders})",
                        stale_candidates,
                    ).fetchall()
                ]
                stale = [row for row in rows if str(row.get("id") or "") in stale_candidates]
                if stale:
                    self._write_jsonl_once(
                        archive_dir / "superseded_gem_findings_before_prune.jsonl",
                        stale,
                    )
                    stale_ids = tuple(str(row["id"]) for row in stale)
                    placeholders = ",".join("?" for _ in stale_ids)
                    conn.execute(
                        f"DELETE FROM findings WHERE id IN ({placeholders})",
                        stale_ids,
                    )
                    conn.execute(
                        f"DELETE FROM finding_edges WHERE "
                        f"src_finding_id IN ({placeholders}) "
                        f"OR dst_finding_id IN ({placeholders})",
                        stale_ids + stale_ids,
                    )
        except sqlite3.Error as exc:
            raise RuntimeError(f"gems: superseded Gem prune failed: {exc}") from exc

    def _write_gem_edges(self, gem_records: list[dict[str, Any]]) -> None:
        if len(gem_records) < 2:
            return
        try:
            from praxist.plugins.workflow_stages.research_loop.backend.tools.local_store import (
                insert_edge,
            )

            for i, src in enumerate(gem_records):
                for dst in gem_records[i + 1 :]:
                    insert_edge(
                        {
                            "src_finding_id": src["gem_finding_id"],
                            "dst_finding_id": dst["gem_finding_id"],
                            "edge_type": "related_to",
                            "confidence": 0.7,
                            "created_by": "gems_agent",
                            "rationale": (
                                "Gems admitted together during a periodic diversity restart; "
                                "future logical generation-0 peers should compare and combine them."
                            ),
                            "provenance": {
                                "source": "gems_cycle",
                                "src_variant": src.get("variant_name"),
                                "dst_variant": dst.get("variant_name"),
                            },
                        }
                    )
        except Exception as exc:  # noqa: BLE001 - advisory graph only.
            logger.warning("gems: failed to insert gem graph edges: %s", exc)

    def _merge_gems_into_frontier_manifest(
        self, gems: list[dict[str, Any]], state: dict[str, Any]
    ) -> None:
        manifest = self.frontier_manifest()
        compact_gems = self._compact_gems(
            gems,
            sort_by_performance=self._mature_evidence_topk_policy_enabled(),
            max_generation_id=self._state_source_generation_limit(state),
            preserve_committed_gems=True,
        )
        validation_candidates = self._validation_candidates_without_gem_entities(
            manifest.get(
                "validation_candidates",
                {
                    "generations": {},
                    "cumulative": [],
                },
            ),
            compact_gems,
        )
        reset_manifest = {
            "generations": {},
            "cumulative_top": [],
            "lane_frontiers": {},
            "validation_candidates": validation_candidates,
            "frontier_lanes": manifest.get("frontier_lanes", []),
            "primary_metric": manifest.get("primary_metric", ""),
            "metric_direction": manifest.get("metric_direction", "maximize"),
        }
        reset_manifest["gems"] = {
            "cycle_index": int(state.get("cycle_index", 0)),
            "reset_count": int(state.get("reset_count", 0)),
            "cycle_start_generation": int(state.get("cycle_start_generation", 0)),
            **self._manifest_gems_config(),
            "entries": compact_gems,
            "bottleneck_reports": list(state.get("active_bottleneck_reports", []) or [])[-5:],
            "latest_soft_agenda_priors": state.get("latest_soft_agenda_priors", {}) or {},
            "state_path": str(self.state_path.relative_to(self.run_dir)),
        }
        if hasattr(self.frontier, "_manifest"):
            self.frontier._manifest.clear()
            self.frontier._manifest.update(reset_manifest)
            if hasattr(self.frontier, "_save_manifest"):
                self.frontier._save_manifest()
        else:
            _atomic_write_json(
                self.run_dir / "frontier" / "frontier_manifest.json",
                _with_frontier_manifest_semantics(
                    reset_manifest,
                    gems_state_path=self.state_path,
                ),
            )

    def _validation_candidates_without_gem_entities(
        self,
        validation: Any,
        gems: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(validation, dict):
            return {"generations": {}, "cumulative": []}
        out = copy.deepcopy(validation)

        def identity_payload(entry: dict[str, Any]) -> dict[str, Any]:
            payload = dict(entry)
            admission = entry.get("admission_metrics")
            metrics = dict(admission) if isinstance(admission, dict) else {}
            existing_metrics = entry.get("metrics")
            if isinstance(existing_metrics, dict):
                metrics.update(existing_metrics)
            if metrics:
                payload["metrics"] = metrics
            return payload

        def aliases_for(entry: dict[str, Any]) -> set[str]:
            payload = identity_payload(entry)
            aliases: set[str] = set()
            try:
                from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
                    _candidate_retirement_aliases,
                )

                aliases.update(_candidate_retirement_aliases(payload))
                return {alias for alias in aliases if alias}
            except (AttributeError, ImportError):
                logger.debug("falling back to local Gem/validation aliases", exc_info=True)
            for key in (
                "gem_finding_id",
                "finding_id",
                "source_finding_id",
                "gem_variant_ref",
                "frontier_entity_key",
                "candidate_entity_key",
                "source_result_path",
                "result_path",
                "result_artifact_path",
                "summary_path",
            ):
                value = payload.get(key)
                if value in (None, "", [], {}):
                    continue
                token = str(value).strip()
                aliases.add(token)
            variant_key = _variant_key(payload)
            if (
                variant_key
                and "sweep" not in variant_key
                and "family" not in variant_key
                and "grid" not in variant_key
            ):
                aliases.add(variant_key)
                if variant_key.startswith("variant:") and not variant_key.startswith("variant::"):
                    aliases.add(f"variant::{variant_key.partition(':')[2]}")
            return {alias for alias in aliases if alias}

        def all_identity_aliases_for(entry: dict[str, Any]) -> set[str]:
            payload = identity_payload(entry)
            aliases: set[str] = set()
            try:
                from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
                    _candidate_entity_key,
                    _candidate_identity_aliases,
                )

                aliases.update(_candidate_identity_aliases(payload))
                key = _candidate_entity_key(payload)
                if key:
                    aliases.add(key)
            except (AttributeError, ImportError):
                logger.debug("falling back to local full Gem/validation aliases", exc_info=True)
            for key in (
                "gem_finding_id",
                "finding_id",
                "source_finding_id",
                "variant_name",
                "variant_id",
                "gem_variant_ref",
                "frontier_entity_key",
                "candidate_entity_key",
                "source_result_path",
                "result_path",
                "result_artifact_path",
                "summary_path",
            ):
                value = payload.get(key)
                if value in (None, "", [], {}):
                    continue
                token = str(value).strip()
                aliases.add(token)
                if key in {"variant_name", "variant_id"}:
                    aliases.add(f"variant::{token.lower()}")
            variant_key = _variant_key(payload)
            if variant_key:
                aliases.add(variant_key)
                if variant_key.startswith("variant:") and not variant_key.startswith("variant::"):
                    aliases.add(f"variant::{variant_key.partition(':')[2]}")
            return {alias for alias in aliases if alias}

        valid_gems = [gem for gem in gems if isinstance(gem, dict)]
        gem_snapshot_records = [
            (result_snapshot_key(identity_payload(gem)), gem.get("variant_name"))
            for gem in valid_gems
        ]
        if not any(
            snapshot is not None and all(snapshot) for snapshot, _fallback in gem_snapshot_records
        ):
            return out

        stale_aliases: set[str] = set()

        def keep(entry: Any) -> bool:
            if not isinstance(entry, dict):
                return True
            payload = identity_payload(entry)
            snapshot = result_snapshot_key(payload)
            if snapshot is None or not all(snapshot):
                return True
            resolved = resolve_result_snapshot_producers(
                [*gem_snapshot_records, (snapshot, entry.get("variant_name"))]
            )
            if not any(
                same_result_snapshot(resolved[-1], gem_snapshot) for gem_snapshot in resolved[:-1]
            ):
                return True
            stale_aliases.update(all_identity_aliases_for(entry))
            return False

        generations = out.get("generations")
        if isinstance(generations, dict):
            for gen_key, entries in list(generations.items()):
                if isinstance(entries, list):
                    generations[gen_key] = [entry for entry in entries if keep(entry)]
        cumulative = out.get("cumulative")
        if isinstance(cumulative, list):
            out["cumulative"] = [entry for entry in cumulative if keep(entry)]
        if stale_aliases:
            alias_generations = out.get("validator_identity_aliases_by_generation")
            if isinstance(alias_generations, dict):
                for gen_key, values in list(alias_generations.items()):
                    if not isinstance(values, list):
                        continue
                    kept = [
                        str(value).strip()
                        for value in values
                        if value not in (None, "") and str(value).strip() not in stale_aliases
                    ]
                    if kept:
                        alias_generations[gen_key] = sorted(set(kept))
                    else:
                        alias_generations.pop(gen_key, None)
            aliases = out.get("validator_identity_aliases")
            if isinstance(aliases, list):
                out["validator_identity_aliases"] = sorted(
                    {
                        str(value).strip()
                        for value in aliases
                        if value not in (None, "") and str(value).strip() not in stale_aliases
                    }
                )
        return out

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(row, default=str) for row in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )

    def _write_jsonl_once(self, path: Path, rows: list[dict[str, Any]]) -> None:
        payload = "\n".join(json.dumps(row, default=str) for row in rows)
        if rows:
            payload += "\n"
        fd = self._open_archive_output_once(path)
        if fd is None:
            return
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)

    @staticmethod
    def _slug(text: str) -> str:
        out = []
        for ch in text.lower():
            if ch.isalnum() or ch in {"-", "_", "."}:
                out.append(ch)
            else:
                out.append("_")
        slug = "".join(out).strip("_")
        return slug or "variant"


class _PromptGemsFrontier:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)

    def get_manifest(self) -> dict[str, Any]:
        path = self.run_dir / "frontier" / "frontier_manifest.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and _is_clean_runtime_state(data):
                    return data
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def get_summary(self) -> list[dict[str, Any]]:
        manifest = self.get_manifest()
        rows = manifest.get("cumulative_top")
        return rows if isinstance(rows, list) else []


def load_active_gems_for_prompt(
    run_dir: Path,
    *,
    max_entries: int | None = None,
    max_generation_id: int | None = None,
) -> dict[str, Any]:
    """Load Gems for PI/Chair prompts through the same runtime filter as peers.

    Older prompt paths used ``state["gems"][:4]`` directly, which could expose
    stale, capped, or future-generation Gems after a repaired reset. This helper
    keeps all prompt surfaces aligned with the active Gems policy configured in
    the run's task spec. If the task spec is missing, fall back to the bounded
    legacy slice rather than failing PI synthesis.
    """

    run_path = Path(run_dir)
    state_path = run_path / "gems" / "gems_state.json"
    if not state_path.exists():
        return {}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - best-effort prompt context.
        logger.warning("gems: failed to load prompt Gems state %s: %s", state_path, exc)
        return {}
    if not isinstance(state, dict):
        return {}
    if not _is_clean_runtime_state(state):
        logger.warning(
            "gems: ignoring non-committed runtime Gems state for prompt load: %s",
            state_path,
        )
        return {}

    trust_committed_membership = is_committed_runtime_fact_source(
        state,
        legacy_ok=False,
    )

    limit = 4 if max_entries is None else max(0, int(max_entries))
    entries: list[dict[str, Any]]
    resolved_source_generation = _entry_source_generation_id
    used_task_spec_filter = False
    task_spec_path = run_path / "task_spec.yaml"
    if trust_committed_membership:
        raw_entries = state.get("gems")
        entries = (
            [dict(e) for e in raw_entries if isinstance(e, dict)]
            if isinstance(raw_entries, list)
            else []
        )

        state_source_limit = _persisted_gems_source_generation_limit(state)

        def resolved_source_generation(entry: dict[str, Any]) -> int | None:
            resolved = _resolved_persisted_gem_source_generation_id(
                run_path,
                entry,
                allow_identity_inference=False,
            )
            return resolved if resolved is not None else state_source_limit

    elif task_spec_path.exists():
        try:
            from praxist.task_spec import load_task_spec

            task_spec = load_task_spec(task_spec_path)
            mgr = GemsManager(
                run_dir=run_path,
                task_spec=task_spec,
                frontier=_PromptGemsFrontier(run_path),
            )
            entries = mgr.active_gems_from_state(
                state,
                max_generation_id=max_generation_id,
            )
            resolved_source_generation = mgr._resolved_entry_source_generation_id
            used_task_spec_filter = True
        except Exception as exc:  # noqa: BLE001 - prompt context should degrade.
            logger.warning("gems: failed to filter prompt Gems via task spec: %s", exc)
            raw_entries = state.get("gems")
            entries = (
                [dict(e) for e in raw_entries if isinstance(e, dict)]
                if isinstance(raw_entries, list)
                else []
            )
    else:
        raw_entries = state.get("gems")
        entries = (
            [dict(e) for e in raw_entries if isinstance(e, dict)]
            if isinstance(raw_entries, list)
            else []
        )

    def has_persisted_measurement(entry: dict[str, Any]) -> bool:
        if _entry_has_score_evidence(entry):
            return True
        if not entry.get("gem_finding_id") or not isinstance(entry.get("admission_metrics"), dict):
            return False
        return any(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in _entry_metrics(entry).values()
        )

    def prompt_gem_allowed(entry: dict[str, Any]) -> bool:
        if trust_committed_membership:
            return True
        if _entry_has_hard_gem_rejection_marker(entry):
            return False
        if used_task_spec_filter:
            return True
        if _boolish_entry_field(entry, "parent_eligible") is False:
            return False
        if evidence_maturity_snapshot(entry).get("mature_enough") is False:
            return False
        return bool(
            not _entry_has_nonclean_gem_marker(entry)
            and has_persisted_measurement(entry)
            and _entry_has_explicit_complete_eval_evidence(entry)
        )

    entries = [entry for entry in entries if prompt_gem_allowed(entry)]
    if trust_committed_membership:
        resolved_entries: list[dict[str, Any]] = []
        for entry in entries:
            source_gen = resolved_source_generation(entry)
            resolved = dict(entry)
            if source_gen is not None:
                resolved["source_generation_id"] = source_gen
            resolved_entries.append(resolved)
        entries = resolved_entries
    if max_generation_id is not None:

        def generation_allowed(entry: dict[str, Any]) -> bool:
            source_gen = resolved_source_generation(entry)
            if source_gen is not None:
                return source_gen <= int(max_generation_id)
            return False

        entries = [entry for entry in entries if generation_allowed(entry)]

    if limit:
        entries = entries[:limit]
    else:
        entries = []
    bottleneck_reports = _filter_bottleneck_reports_for_generation(
        _state_bottleneck_reports(state),
        max_generation_id,
    )
    latest_soft_agenda_priors = _latest_soft_priors_for_generation(
        _state_bottleneck_reports(state),
        max_generation_id,
        state.get("latest_soft_agenda_priors", {}) or {},
    )
    return {
        "cycle_index": state.get("cycle_index", 0),
        "reset_count": state.get("reset_count", 0),
        "cycle_start_generation": state.get("cycle_start_generation", 0),
        "entries": entries,
        "bottleneck_reports": bottleneck_reports[-5:],
        "latest_soft_agenda_priors": latest_soft_agenda_priors,
    }
