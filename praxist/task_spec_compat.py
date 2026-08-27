"""One-way migration of historical task-spec fields.

Only the task loading boundary and readers of historical run manifests may
import this module.  Runtime policy code consumes the generic fields produced
here and must not attach semantics to old task-local vocabulary.
"""

from __future__ import annotations

from typing import Any

_MATURE_EVIDENCE_POLICY = "mature_evidence_top_k"
_LEGACY_POLICY_ALIASES = frozenset(
    {
        "full_window_top_k",
        "full_window_top4",
        "full_window_performance_top_k",
    }
)
_LEGACY_CONFIG_ALIASES = {
    "min_full_t1_eval_cells": "min_mature_eval_units",
    "evidence_stage_min_cells": "evidence_stage_min_units",
}
_LEGACY_ENTRY_ALIASES = {
    "_gems_min_full_t1_eval_cells": "_gems_min_mature_eval_units",
    "_gems_evidence_stage_min_cells": "_gems_evidence_stage_min_units",
}
_LEGACY_PRIMARY_METRIC_KEYS = ("mean_test_taskscore",)
_LEGACY_COMMITTED_TIER_MARKERS = frozenset({"t1", "t2", "t3"})
_COMPLETION_KEYS = (
    "mature_enough",
    "scored_complete",
    "is_scored_complete",
    "complete_eval",
    "is_complete_eval",
)


def legacy_primary_metric_keys() -> tuple[str, ...]:
    """Return historical task metric aliases for read-only migration paths."""

    return _LEGACY_PRIMARY_METRIC_KEYS


def migrate_legacy_gems_config(raw: Any) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return generic Gems config plus the historical inputs that were used."""

    if not isinstance(raw, dict):
        return {}, ()
    migrated = dict(raw)
    used: list[str] = []
    policy = str(migrated.get("selection_policy") or "").strip().lower()
    if policy in _LEGACY_POLICY_ALIASES:
        migrated["selection_policy"] = _MATURE_EVIDENCE_POLICY
        used.append("selection_policy")
    for old_key, new_key in _LEGACY_CONFIG_ALIASES.items():
        if new_key not in migrated and old_key in migrated:
            migrated[new_key] = migrated[old_key]
            used.append(old_key)
    return migrated, tuple(used)


def migrate_legacy_gems_entry(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize old manifest metadata without mutating the stored artifact."""

    migrated = dict(raw)
    for old_key, new_key in _LEGACY_ENTRY_ALIASES.items():
        if new_key not in migrated and old_key in migrated:
            migrated[new_key] = migrated[old_key]
    for nested_key in ("metrics", "admission_metrics"):
        nested = migrated.get(nested_key)
        if not isinstance(nested, dict):
            continue
        normalized_nested = dict(nested)
        for old_key, new_key in _LEGACY_ENTRY_ALIASES.items():
            if new_key not in normalized_nested and old_key in normalized_nested:
                normalized_nested[new_key] = normalized_nested[old_key]
        policy = str(normalized_nested.get("selection_policy") or "").strip().lower()
        if policy in _LEGACY_POLICY_ALIASES:
            normalized_nested["selection_policy"] = _MATURE_EVIDENCE_POLICY
        migrated[nested_key] = normalized_nested
    policy = str(migrated.get("selection_policy") or "").strip().lower()
    if policy in _LEGACY_POLICY_ALIASES:
        migrated["selection_policy"] = _MATURE_EVIDENCE_POLICY
    sources: list[dict[str, Any]] = [migrated]
    for key in ("metrics", "admission_metrics"):
        nested = migrated.get(key)
        if isinstance(nested, dict):
            sources.append(nested)
    explicit_completion = next(
        (source[key] for source in sources for key in _COMPLETION_KEYS if key in source),
        None,
    )
    tier = next(
        (
            str(source[key]).strip().lower()
            for source in sources
            for key in ("tier", "tier_reached", "completed_tier")
            if source.get(key) not in (None, "")
        ),
        "",
    )
    if (
        migrated.get("gem_finding_id")
        and tier in _LEGACY_COMMITTED_TIER_MARKERS
        and explicit_completion is None
    ):
        migrated["_legacy_committed_complete_evidence"] = True
    return migrated
