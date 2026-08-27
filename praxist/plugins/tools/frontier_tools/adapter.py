"""
Frontier MCP tools — query and interact with the Frontier Store.
"""

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

import yaml

from praxist.plugins.tools.result_envelope import (
    active_run_dir,
    coerce_inline_limit,
    with_tool_output_envelope,
)
from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
    DERIVED_VIEW,
    attach_artifact_semantics,
    is_committed_runtime_fact_source,
    is_readable_signal_source,
)
from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
    durable_promotion_exclusion,
    evidence_maturity_snapshot,
    normalize_maturity_policy,
)
from praxist.plugins.workflow_stages.research_loop.backend.gems import (
    _resolved_persisted_gem_source_generation_id,
)
from praxist.task_spec import GEMS_MATURE_EVIDENCE_TOP_K, normalize_gems_selection_policy
from praxist.task_spec_compat import (
    migrate_legacy_gems_config,
    migrate_legacy_gems_entry,
)

try:
    from claude_agent_sdk import create_sdk_mcp_server, tool
except ImportError:
    tool = None
    create_sdk_mcp_server = None

logger = logging.getLogger(__name__)

_VALIDATION_CANDIDATES_OUTPUT_CAP = 48
_VALIDATION_RESEARCH_METADATA_KEYS = (
    "bottleneck_target",
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
_VALIDATION_DIVERSITY_METADATA_KEYS = (
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


def _annotate_frontier_tool_view(
    payload: dict[str, Any], *, up_to_generation: int
) -> dict[str, Any]:
    return attach_artifact_semantics(
        payload,
        role=DERIVED_VIEW,
        stage="frontier_tool_response",
        generation_id=up_to_generation if up_to_generation >= 0 else None,
        actor="tool_server:frontier_tools",
        derived_from=["frontier/frontier_manifest.json", "gems/gems_state.json"],
        canonical_sources=["frontier/frontier_manifest.json", "gems/gems_state.json"],
        runtime_fact_source=False,
        notes=(
            "Frontier tool responses are compact derived views. "
            "The canonical frontier/incubator state remains frontier_manifest.json."
        ),
    )


_VALIDATION_IDENTITY_KEYS = (
    "finding_id",
    "variant_name",
    "variant_id",
    "frontier_entity_key",
    "candidate_entity_key",
    "child_variant_name",
    "child_variant_id",
    "source_path",
    "result_path",
    "source_result_path",
    "result_artifact_path",
    "summary_path",
)


def _iter_validation_identity_values(entry: Any):
    if not isinstance(entry, dict):
        return
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    for source in (entry, metrics):
        for key in _VALIDATION_IDENTITY_KEYS:
            value = source.get(key)
            if isinstance(value, list):
                for item in value:
                    yield item
            else:
                yield value
        aliases = source.get("identity_aliases")
        if isinstance(aliases, list):
            for item in aliases:
                yield item


def _iter_validation_retirement_values(entry: Any):
    if not isinstance(entry, dict):
        return
    try:
        from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
            _candidate_retirement_aliases,
        )

        yield from _candidate_retirement_aliases(entry)
        return
    except (AttributeError, ImportError):
        logger.debug("falling back to validation identity aliases for retirement", exc_info=True)
    yield from _iter_validation_identity_values(entry)


def _text_result(data: Any) -> dict[str, Any]:
    text = json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data
    return {"content": [{"type": "text", "text": text}]}


def _task_maturity_policy(manifest_path: Path) -> tuple[dict[str, Any] | None, str]:
    """Load the task-owned maturity policy used to commit this frontier.

    The run snapshot is the sole safe policy source for a historical view. A
    live task file may have changed since the run committed its Frontier, so it
    must never be used to reinterpret that run's membership.
    """

    run_dir = manifest_path.parent.parent
    load_failed = False
    for name in ("effective_task_spec.yaml", "task_spec.yaml"):
        task_path = run_dir / name
        if not task_path.is_file():
            continue
        try:
            raw = yaml.safe_load(task_path.read_text(encoding="utf-8")) or {}
            if not isinstance(raw, dict):
                raise ValueError("task spec must be a YAML mapping")
            evaluation = raw.get("evaluation")
            policy = evaluation.get("maturity_policy") if isinstance(evaluation, dict) else None
            return (
                normalize_maturity_policy(policy) if isinstance(policy, dict) else None,
                name,
            )
        except Exception as exc:  # noqa: BLE001 - a read tool must report, not crash.
            load_failed = True
            logger.warning(
                "frontier tool: could not load maturity policy from %s: %s",
                task_path,
                exc,
            )
    return None, "load_failed" if load_failed else "unavailable"


def _has_explicit_committed_runtime_semantics(manifest: dict[str, Any]) -> bool:
    semantics = manifest.get("artifact_semantics")
    if not isinstance(semantics, dict):
        return False
    return (
        str(semantics.get("role") or "").strip().lower() == "canonical_state"
        and str(semantics.get("status") or "").strip().lower() == "committed"
        and semantics.get("runtime_fact_source") is True
        and semantics.get("derived") is not True
        and semantics.get("audit_only") is not True
    )


async def _handle_get_frontier(args: dict[str, Any]) -> dict[str, Any]:
    """Get durable frontier entries plus separate non-frontier validation signals."""
    completed_generation = os.environ.get("LAST_COMPLETED_GENERATION_ID") or os.environ.get(
        "COMPLETED_GEN_ID"
    )
    if completed_generation is not None:
        default_generation = _coerce_int(completed_generation, default=-1)
    else:
        current_generation = (
            os.environ.get("CURRENT_GEN_ID")
            or os.environ.get("GENERATION_ID")
            or os.environ.get("PRAXIST_GENERATION_ID")
        )
        current = _coerce_int(current_generation, default=-1)
        default_generation = current - 1 if current >= 0 else -1
    up_to_generation = _coerce_int(args.get("up_to_generation"), default=default_generation)
    allow_unbounded = str(os.environ.get("PRAXIST_FRONTIER_ALLOW_UNBOUNDED", "")).lower() in {
        "1",
        "true",
        "yes",
    }
    if (
        default_generation >= 0
        and not allow_unbounded
        and (up_to_generation < 0 or up_to_generation > default_generation)
    ):
        up_to_generation = default_generation
    top_k = _coerce_int(args.get("top_k"), default=10, minimum=1, maximum=100)
    inline_limit = coerce_inline_limit(args.get("inline_limit", top_k), default=top_k)

    frontier_dir = os.environ.get("FRONTIER_DIR", "")
    if not frontier_dir:
        return _text_result(
            _annotate_frontier_tool_view(
                {
                    "entries": [],
                    "note": "FRONTIER_DIR not set",
                },
                up_to_generation=up_to_generation,
            )
        )

    manifest_path = Path(frontier_dir) / "frontier_manifest.json"
    if not manifest_path.exists():
        return _text_result(
            _annotate_frontier_tool_view(
                {
                    "entries": [],
                    "note": "No frontier manifest found. This may be the first generation.",
                },
                up_to_generation=up_to_generation,
            )
        )

    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return _text_result(
            _annotate_frontier_tool_view(
                {
                    "entries": [],
                    "error": f"frontier manifest could not be read: {type(exc).__name__}: {exc}",
                },
                up_to_generation=up_to_generation,
            )
        )
    if not isinstance(manifest, dict):
        return _text_result(
            _annotate_frontier_tool_view(
                {
                    "entries": [],
                    "error": "frontier manifest must be a JSON object",
                },
                up_to_generation=up_to_generation,
            )
        )
    manifest_signal_only = False
    manifest_signal_status = ""
    if not is_committed_runtime_fact_source(manifest, legacy_ok=True):
        if not is_readable_signal_source(manifest, legacy_ok=True):
            return _text_result(
                _annotate_frontier_tool_view(
                    {
                        "entries": [],
                        "validation_candidates": [],
                        "error": (
                            "frontier manifest is superseded; ignoring it to avoid using "
                            "stale duplicate state"
                        ),
                    },
                    up_to_generation=up_to_generation,
                )
            )
        semantics = manifest.get("artifact_semantics")
        manifest_signal_status = (
            str(semantics.get("status") or "").strip().lower()
            if isinstance(semantics, dict)
            else "legacy"
        )
        manifest_signal_only = True
        logger.warning(
            "frontier tool: using non-runtime frontier manifest only as validation signal source: %s",
            manifest_path,
        )
        manifest = dict(manifest)
        manifest["_non_runtime_signal_source"] = True
        manifest["_non_runtime_signal_status"] = manifest_signal_status or "unknown"

    maturity_policy, maturity_policy_source = (
        (None, "signal_only") if manifest_signal_only else _task_maturity_policy(manifest_path)
    )
    # Legacy manifests predate artifact semantics but their frontier surfaces
    # were also committed state. Keep that compatibility while applying the
    # current task policy whenever it is available.
    trust_committed_membership = (
        not manifest_signal_only and _has_explicit_committed_runtime_semantics(manifest)
    )
    generation_skip_reasons: dict[str, int] = {}
    lane_skip_reasons: dict[str, int] = {}
    historical_frontier_replayed = False
    manifest_generation_ids = (
        [
            gen_id
            for key in manifest.get("generations", {})
            if (gen_id := _coerce_int(key, default=-1)) >= 0
        ]
        if isinstance(manifest.get("generations"), dict)
        else []
    )
    historical_cutoff = bool(
        manifest_generation_ids
        and up_to_generation >= 0
        and up_to_generation < max(manifest_generation_ids)
    )
    manifest_gems = manifest.get("gems")
    if isinstance(manifest_gems, dict) and up_to_generation >= 0:
        cycle_start = _coerce_int(manifest_gems.get("cycle_start_generation"), default=0)
        reset_count = _coerce_int(manifest_gems.get("reset_count"), default=0)
        historical_cutoff = historical_cutoff or (
            reset_count > 0 and cycle_start > 0 and up_to_generation < cycle_start - 1
        )

    if manifest_signal_only:
        entries = []
        generations = {}
        skipped_generations = 0
        skipped_entries = 0
        lane_frontiers = {}
        gems = {}
    else:
        entries = []
        generations = manifest.get("generations", {})
        if not isinstance(generations, dict):
            generations = {}
        skipped_generations = 0
        skipped_entries = 0
        for gen_str, gen_entries in generations.items():
            try:
                gen_id = int(gen_str)
            except (TypeError, ValueError):
                skipped_generations += 1
                continue
            if up_to_generation >= 0 and gen_id > up_to_generation:
                continue
            if not isinstance(gen_entries, list):
                skipped_entries += 1
                continue
            for entry in gen_entries:
                if not isinstance(entry, dict):
                    skipped_entries += 1
                    continue
                if not _entry_visible_at_generation(
                    entry,
                    up_to_generation,
                    commit_generation_hint=gen_id,
                ):
                    continue
                durable, reason = _frontier_entry_durability(
                    entry,
                    maturity_policy=maturity_policy,
                    trust_committed_membership=trust_committed_membership,
                    allow_non_promotable=_lane_allows_non_promotable(
                        manifest,
                        _entry_target_lane(entry),
                    ),
                )
                if not durable:
                    skipped_entries += 1
                    generation_skip_reasons[reason] = generation_skip_reasons.get(reason, 0) + 1
                    continue
                metric_value = _coerce_float(entry.get("metric_value"))
                entries.append(
                    {
                        "generation_id": _entry_commit_generation_id(entry)
                        if _entry_commit_generation_id(entry) is not None
                        else gen_id,
                        "rank": entry.get("rank"),
                        "variant_name": entry.get("variant_name", ""),
                        "metric_value": metric_value,
                        "metrics": entry.get("metrics", {}),
                        "finding_id": entry.get("finding_id", ""),
                        "promoted_at": entry.get("promoted_at", ""),
                    }
                )

        if not generations:
            for entry in manifest.get("cumulative_top") or []:
                if not isinstance(entry, dict):
                    skipped_entries += 1
                    continue
                if not _entry_visible_at_generation(entry, up_to_generation):
                    continue
                durable, reason = _frontier_entry_durability(
                    entry,
                    maturity_policy=maturity_policy,
                    trust_committed_membership=trust_committed_membership,
                )
                if not durable:
                    skipped_entries += 1
                    generation_skip_reasons[reason] = generation_skip_reasons.get(reason, 0) + 1
                    continue
                entries.append(
                    {
                        "generation_id": _entry_commit_generation_id(entry),
                        "rank": entry.get("rank"),
                        "variant_name": entry.get("variant_name", ""),
                        "metric_value": _coerce_float(entry.get("metric_value")),
                        "metrics": entry.get("metrics", {}),
                        "finding_id": entry.get("finding_id", ""),
                        "promoted_at": entry.get("promoted_at", ""),
                    }
                )

        lane_frontiers = _compact_historical_lane_frontiers(
            manifest,
            up_to_generation=up_to_generation,
            maturity_policy=maturity_policy,
            trust_committed_membership=trust_committed_membership,
            skipped_by_reason=lane_skip_reasons,
        )
        historical_frontier_replayed = lane_frontiers is not None
        if lane_frontiers is None:
            lane_frontiers = _compact_lane_frontiers(
                manifest,
                up_to_generation=up_to_generation,
                maturity_policy=maturity_policy,
                trust_committed_membership=trust_committed_membership,
                skipped_by_reason=lane_skip_reasons,
            )
        run_dir = manifest_path.parent.parent
        gems = _compact_gems(
            manifest,
            up_to_generation=up_to_generation,
            run_dir=run_dir,
            maturity_policy=maturity_policy,
            historical_view=historical_cutoff,
        )

    validation_candidates = _compact_validation_candidates(
        manifest,
        up_to_generation=up_to_generation,
        limit=None,
        run_dir=manifest_path.parent.parent,
        maturity_policy=maturity_policy,
        trust_committed_membership=trust_committed_membership,
    )
    can_store_full_result = active_run_dir() is not None
    returned_validation_candidates = (
        validation_candidates[:_VALIDATION_CANDIDATES_OUTPUT_CAP]
        if can_store_full_result
        else validation_candidates
    )
    # Sort by metric — check manifest for direction, default to maximize
    direction = manifest.get("metric_direction", "maximize")
    reverse = direction != "minimize"
    missing_default = float("-inf") if reverse else float("inf")
    entries.sort(
        key=lambda e: (
            e.get("metric_value") if e.get("metric_value") is not None else missing_default
        ),
        reverse=reverse,
    )
    output_entries = (
        _lane_priority_entries(lane_frontiers, manifest=manifest) if lane_frontiers else entries
    )
    canonical_frontier_entry_count = _canonical_frontier_entry_count(
        manifest,
        up_to_generation=up_to_generation,
        historical_lane_frontiers=lane_frontiers if historical_frontier_replayed else None,
    )
    returned_frontier_entry_count = (
        sum(len(lane) for lane in lane_frontiers.values()) if lane_frontiers else len(entries)
    )
    # Retirement compacts both cumulative and per-generation validation rows,
    # so their past membership cannot be reconstructed after a later promotion.
    historical_validation_view_complete = not historical_cutoff
    if manifest_signal_only:
        frontier_view_integrity_status = "signal_only"
    elif canonical_frontier_entry_count == 0:
        frontier_view_integrity_status = "empty"
    elif returned_frontier_entry_count == canonical_frontier_entry_count:
        frontier_view_integrity_status = "ok"
    elif returned_frontier_entry_count == 0:
        frontier_view_integrity_status = "canonical_entries_hidden"
    else:
        frontier_view_integrity_status = "canonical_entries_filtered"
    payload = {
        "entries": output_entries[:top_k],
        "lane_frontiers": lane_frontiers,
        "validation_candidates": returned_validation_candidates,
        "gems": gems,
        "lane_mode": bool(lane_frontiers),
        "total_matching_entries": len(output_entries),
        "total_validation_candidates": len(validation_candidates),
        "returned_validation_candidates": len(returned_validation_candidates),
        "validation_candidates_truncated": len(returned_validation_candidates)
        < len(validation_candidates),
        "validation_candidates_hard_cap": _VALIDATION_CANDIDATES_OUTPUT_CAP,
        "total_legacy_metric_entries": len(entries),
        "total_generations": len(generations) - skipped_generations,
        "skipped_generations": skipped_generations,
        "skipped_entries": skipped_entries,
        "skipped_by_reason": {
            "generation_entries": dict(sorted(generation_skip_reasons.items())),
            "lane_entries": dict(sorted(lane_skip_reasons.items())),
        },
        "canonical_frontier_entry_count": canonical_frontier_entry_count,
        "returned_frontier_entry_count": returned_frontier_entry_count,
        "maturity_policy_loaded": maturity_policy is not None,
        "maturity_policy_source": maturity_policy_source,
        "committed_membership_trusted": trust_committed_membership,
        "frontier_view_integrity_status": frontier_view_integrity_status,
        "historical_validation_view_complete": historical_validation_view_complete,
        "historical_frontier_replayed": historical_frontier_replayed,
        "manifest_scope": "signal_only" if manifest_signal_only else "runtime_fact_source",
        "manifest_runtime_fact_source": not manifest_signal_only,
        "manifest_signal_status": manifest_signal_status if manifest_signal_only else "",
    }
    if frontier_view_integrity_status == "canonical_entries_hidden":
        payload["frontier_view_integrity_error"] = (
            "Canonical frontier lanes contain committed entries, but this derived view "
            "returned none. Inspect skipped_by_reason and the task maturity policy; do "
            "not interpret the empty lane view as an empty canonical frontier."
        )
    full_payload = {
        **payload,
        "entries": output_entries,
        "validation_candidates": validation_candidates,
        "gems": gems,
        "returned_validation_candidates": len(validation_candidates),
        "validation_candidates_truncated": False,
    }

    view_payload = _annotate_frontier_tool_view(payload, up_to_generation=up_to_generation)
    full_view_payload = _annotate_frontier_tool_view(
        full_payload,
        up_to_generation=up_to_generation,
    )
    output = with_tool_output_envelope(
        view_payload,
        tool_name="get_frontier",
        list_fields=("entries",),
        inline_limit=inline_limit,
        full_payload=full_view_payload,
    )
    if payload["validation_candidates_truncated"]:
        tool_output = output.get("_tool_output")
        if isinstance(tool_output, dict):
            truncated_lists = tool_output.setdefault("truncated_lists", {})
            if isinstance(truncated_lists, dict):
                truncated_lists["validation_candidates"] = {
                    "returned": payload["returned_validation_candidates"],
                    "total": payload["total_validation_candidates"],
                }
            tool_output["truncated"] = True
    return _text_result(output)


def _compact_gems(
    manifest: dict[str, Any],
    *,
    up_to_generation: int = -1,
    run_dir: Path | None = None,
    maturity_policy: dict[str, Any] | None = None,
    historical_view: bool = False,
) -> dict[str, Any]:
    raw = manifest.get("gems")
    if not isinstance(raw, dict):
        return {}
    entries = raw.get("entries")
    if not isinstance(entries, list):
        entries = []
    compact_entries = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if historical_view:
            continue
        eligibility_entry = _gem_entry_with_manifest_selection_policy(
            entry,
            raw,
            frontier_manifest=manifest,
        )
        if not _gem_entry_is_prompt_eligible(
            eligibility_entry,
            maturity_policy=maturity_policy,
        ):
            continue
        source_generation = _resolved_persisted_gem_source_generation_id(run_dir, entry)
        if up_to_generation >= 0 and (
            source_generation is None or source_generation > up_to_generation
        ):
            continue
        compact_entries.append(
            {
                "gem_finding_id": entry.get("gem_finding_id", ""),
                "variant_name": entry.get("variant_name", ""),
                "frontier_lane": entry.get("frontier_lane", ""),
                "metric_name": entry.get("metric_name", ""),
                "metric_value": entry.get("metric_value"),
                "source_finding_id": entry.get("source_finding_id", ""),
                "source_generation_id": entry.get("source_generation_id"),
                "gem_variant_ref": entry.get("gem_variant_ref", ""),
                "finding_path": entry.get("finding_path", ""),
                "admitted_at": entry.get("admitted_at", ""),
                "admission_metrics": entry.get("admission_metrics", {}) or {},
            }
        )
    reports = []
    for report in raw.get("bottleneck_reports", []) or []:
        if not isinstance(report, dict):
            continue
        report_gen = _entry_generation_id(report)
        if up_to_generation >= 0 and (report_gen is None or report_gen > up_to_generation):
            continue
        reports.append(report)
    latest_soft_agenda_priors = raw.get("latest_soft_agenda_priors", {}) or {}
    if up_to_generation >= 0:
        last_report = reports[-1] if reports else {}
        latest_soft_agenda_priors = (
            last_report.get("soft_agenda_priors", {}) if isinstance(last_report, dict) else {}
        ) or {}
    compact = {
        "cycle_index": None if historical_view else raw.get("cycle_index", 0),
        "reset_count": None if historical_view else raw.get("reset_count", 0),
        "cycle_start_generation": None if historical_view else raw.get("cycle_start_generation", 0),
        "entries": compact_entries,
        "bottleneck_reports": reports[-5:],
        "latest_soft_agenda_priors": latest_soft_agenda_priors,
    }
    if historical_view:
        compact["historical_entries_complete"] = False
        compact["historical_entries_note"] = (
            "The persisted Gems block is current compact state, not an append-only history. "
            "Membership and cycle fields are omitted rather than attributed to the wrong "
            "generation; use the replayed Frontier lanes for historical committed evidence."
        )
    return compact


def _gem_entry_with_manifest_selection_policy(
    entry: dict[str, Any],
    gems_block: dict[str, Any] | None,
    *,
    frontier_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry = migrate_legacy_gems_entry(entry)
    updates: dict[str, Any] = {}
    selection_policy = ""
    if isinstance(gems_block, dict):
        gems_block, _migrated = migrate_legacy_gems_config(gems_block)
        raw_selection_policy = gems_block.get("selection_policy")
        if raw_selection_policy not in (None, ""):
            selection_policy = normalize_gems_selection_policy(raw_selection_policy)
        for key in (
            "min_mature_eval_units",
            "evidence_stage_min_units",
            "primary_metric_keys",
            "secondary_metric_keys",
            "lower_tail_metric_keys",
            "validation_metric_keys",
            "performance_lanes",
            "control_lanes",
        ):
            if key in gems_block:
                updates[f"_gems_{key}"] = gems_block[key]
    if selection_policy:
        updates["selection_policy"] = selection_policy
    lane_policy = _frontier_lane_parent_eligibility(frontier_manifest)
    if lane_policy:
        metrics = entry.get("admission_metrics")
        if not isinstance(metrics, dict):
            metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        lane = str(
            entry.get("promoted_for_lane")
            or entry.get("frontier_lane")
            or metrics.get("frontier_lane")
            or ""
        ).strip()
        explicit = _validation_boolish(entry.get("parent_eligible", metrics.get("parent_eligible")))
        updates["parent_eligible"] = lane_policy.get(lane) is True and explicit is not False
    if not updates:
        return entry
    return {**entry, **updates}


def _frontier_lane_parent_eligibility(
    manifest: dict[str, Any] | None,
) -> dict[str, bool]:
    if not isinstance(manifest, dict):
        return {}
    raw_lanes = manifest.get("frontier_lanes")
    if not isinstance(raw_lanes, list):
        return {}
    policy: dict[str, bool] = {}
    for lane in raw_lanes:
        if not isinstance(lane, dict):
            continue
        name = str(lane.get("name") or "").strip()
        if not name:
            continue
        allow_lower_tier = bool(lane.get("allow_lower_tier", False))
        policy[name] = bool(lane.get("parent_eligible", not allow_lower_tier))
    return policy


def _gem_entry_uses_mature_evidence_topk(entry: dict[str, Any]) -> bool:
    metrics = entry.get("admission_metrics")
    if not isinstance(metrics, dict):
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    policy = normalize_gems_selection_policy(
        entry.get("gems_selection_policy")
        or entry.get("selection_policy")
        or metrics.get("gems_selection_policy")
        or metrics.get("selection_policy")
        or ""
    )
    return policy == GEMS_MATURE_EVIDENCE_TOP_K


def _coerce_str_set(value: Any) -> set[str]:
    if value is None:
        return set()
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return {str(item).strip() for item in values if str(item).strip()}


def _gem_entry_is_manifest_performance_entry(entry: dict[str, Any]) -> bool:
    metrics = entry.get("admission_metrics")
    if not isinstance(metrics, dict):
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    lane = str(entry.get("frontier_lane") or metrics.get("frontier_lane") or "").strip()
    performance_lanes = _coerce_str_set(
        entry.get("_gems_performance_lanes") or metrics.get("_gems_performance_lanes")
    )
    control_lanes = _coerce_str_set(
        entry.get("_gems_control_lanes") or metrics.get("_gems_control_lanes")
    )
    if lane and lane in performance_lanes:
        return True
    if lane and (lane in control_lanes or performance_lanes):
        return False
    strategy = str(entry.get("strategy_family") or metrics.get("strategy_family") or "").strip()
    return strategy in {"learned_candidate", "candidate", "task_candidate"}


def _gem_entry_min_mature_eval_units(entry: dict[str, Any]) -> int:
    metrics = entry.get("admission_metrics")
    if not isinstance(metrics, dict):
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    raw = entry.get("_gems_min_mature_eval_units", metrics.get("_gems_min_mature_eval_units"))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def _gem_entry_has_mature_evidence_threshold(entry: dict[str, Any]) -> bool:
    metrics = entry.get("admission_metrics")
    if not isinstance(metrics, dict):
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    return (
        entry.get("_gems_min_mature_eval_units") not in (None, "")
        or metrics.get("_gems_min_mature_eval_units") not in (None, "")
        or isinstance(entry.get("_gems_evidence_stage_min_units"), dict)
        or isinstance(metrics.get("_gems_evidence_stage_min_units"), dict)
    )


def _gem_entry_evidence_stage_min_units(entry: dict[str, Any]) -> dict[str, int]:
    metrics = entry.get("admission_metrics")
    if not isinstance(metrics, dict):
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    raw = entry.get("_gems_evidence_stage_min_units", metrics.get("_gems_evidence_stage_min_units"))
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in raw.items():
        try:
            out[str(key).strip().lower()] = max(1, int(value))
        except (TypeError, ValueError):
            continue
    return out


def _gem_entry_has_known_maturity_failure(
    entry: dict[str, Any],
    maturity_policy: dict[str, Any] | None = None,
) -> bool:
    metrics = entry.get("admission_metrics")
    if not isinstance(metrics, dict):
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    if maturity_policy is None:
        recorded = _recorded_task_maturity_decision(entry)
        if recorded is not None:
            return not recorded
        if _validation_boolish(entry.get("mature_enough", metrics.get("mature_enough"))) is False:
            return True
    policy = maturity_policy or {
        key: entry.get(key, metrics.get(key))
        for key in ("min_effort_ratio", "min_coverage_ratio")
        if entry.get(key, metrics.get(key)) not in (None, "")
    }
    return evidence_maturity_snapshot(entry, policy).get("mature_enough") is False


def _gem_entry_has_legacy_committed_complete_evidence(entry: dict[str, Any]) -> bool:
    metrics = entry.get("admission_metrics")
    if not isinstance(metrics, dict):
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    marker = _validation_boolish(
        entry.get(
            "_legacy_committed_complete_evidence",
            metrics.get("_legacy_committed_complete_evidence"),
        )
    )
    if marker is not True or not entry.get("gem_finding_id"):
        return False
    count = 0
    for key in (
        "completed_required_eval_units",
        "actual_eval_units",
        "evaluation_units",
        "scored_cell_count",
        "n_scored_cells",
        "n_eval_cells",
        "cell_count",
        "n_cells",
    ):
        raw = entry.get(key, metrics.get(key))
        try:
            count = max(0, int(float(raw)))
        except (TypeError, ValueError):
            continue
        break
    return count == 0 or count >= _gem_entry_min_mature_eval_units(entry)


def _gem_entry_is_prompt_eligible(
    entry: dict[str, Any],
    *,
    maturity_policy: dict[str, Any] | None = None,
) -> bool:
    metrics = entry.get("admission_metrics")
    if not isinstance(metrics, dict):
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    if _validation_boolish(entry.get("parent_eligible", metrics.get("parent_eligible"))) is False:
        return False
    if _gem_entry_has_known_maturity_failure(entry, maturity_policy):
        return False
    try:
        from praxist.plugins.workflow_stages.research_loop.backend.gems import (
            _entry_eval_unit_count,
            _entry_has_explicit_complete_eval_evidence,
            _entry_has_explicit_gem_rejection_marker,
            _entry_has_explicit_scout_or_partial_marker,
            _entry_has_gem_integrity_rejection_marker,
            _entry_has_nonclean_gem_marker,
            _entry_is_clean_gem_admission_candidate,
            _entry_tier_text,
            _is_mature_evaluation_or_better,
            _is_performance_entry,
        )

        legacy_committed = _gem_entry_has_legacy_committed_complete_evidence(entry)
        if maturity_policy is None:
            recorded_mature = _recorded_task_maturity_decision(entry) is True
            if _entry_has_gem_integrity_rejection_marker(entry) or _entry_has_nonclean_gem_marker(
                entry
            ):
                return False
            if not recorded_mature:
                if _entry_has_explicit_scout_or_partial_marker(entry):
                    return False
                if _entry_has_explicit_gem_rejection_marker(entry):
                    return False
            if not _gem_entry_uses_mature_evidence_topk(entry):
                return (
                    recorded_mature
                    or _entry_has_explicit_complete_eval_evidence(entry)
                    or legacy_committed
                )
            if recorded_mature:
                units = _entry_eval_unit_count(entry)
                required_units = _gem_entry_min_mature_eval_units(entry)
                stage = re.sub(r"[^a-z0-9]+", "_", _entry_tier_text(entry)).strip("_")
                required_units = max(
                    required_units,
                    _gem_entry_evidence_stage_min_units(entry).get(stage, 1),
                )
                return (
                    _is_performance_entry(entry) or _gem_entry_is_manifest_performance_entry(entry)
                ) and units >= required_units
        clean_candidate = maturity_policy is None or _entry_is_clean_gem_admission_candidate(
            entry,
            maturity_policy,
        )
        if not (clean_candidate or legacy_committed):
            return False
        if _gem_entry_uses_mature_evidence_topk(entry):
            return (
                _is_performance_entry(entry) or _gem_entry_is_manifest_performance_entry(entry)
            ) and (
                legacy_committed
                or (
                    _gem_entry_has_mature_evidence_threshold(entry)
                    and _entry_has_explicit_complete_eval_evidence(entry)
                    and _is_mature_evaluation_or_better(
                        entry,
                        min_mature_eval_units=_gem_entry_min_mature_eval_units(entry),
                        evidence_stage_min_units=_gem_entry_evidence_stage_min_units(entry),
                        skip_performance_check=True,
                        maturity_policy=maturity_policy,
                    )
                )
            )
        return clean_candidate or legacy_committed
    except (AttributeError, ImportError):
        logger.debug("falling back to local Gem prompt eligibility check", exc_info=True)
    if _gem_entry_has_known_maturity_failure(entry, maturity_policy):
        return False
    if _validation_entry_is_preliminary({**entry, "metrics": metrics}):
        return False
    if _local_gem_entry_has_nonclean_marker(entry, metrics):
        return False
    return _validation_entry_has_complete_evidence(
        entry,
        metrics,
    ) or _gem_entry_has_legacy_committed_complete_evidence(entry)


def _validation_entry_has_complete_evidence(
    entry: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> bool:
    if metrics is None:
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    for key in (
        "mature_enough",
        "complete_eval",
        "is_complete_eval",
        "scored_complete",
        "is_scored_complete",
    ):
        if _validation_boolish(entry.get(key, metrics.get(key))) is True:
            return True
    for key in (
        "evidence_stage",
        "tier",
        "tier_reached",
        "completed_tier",
        "candidate_tier",
        "tier_status",
        "final_status",
        "result_status",
        "completion_status",
        "eval_status",
    ):
        value = entry.get(key, metrics.get(key))
        normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {
            "scored_complete",
            "complete_eval",
            "complete_eval_true",
            "is_complete_eval_true",
            "scored_complete_true",
            "is_scored_complete_true",
            "full_eval",
            "full_evaluation",
        }:
            return True
    return False


def _recorded_task_maturity_decision(entry: dict[str, Any]) -> bool | None:
    """Read a run-owned maturity decision whose task policy is unavailable."""

    sources = [entry]
    sources.extend(
        value
        for key in ("metrics", "admission_metrics")
        if isinstance((value := entry.get(key)), dict)
    )
    for source in sources:
        basis = str(source.get("maturity_basis") or "").strip().lower()
        if basis not in {"effort_coverage_ratio", "task_configured_stage"}:
            continue
        decision = _validation_boolish(source.get("mature_enough"))
        if decision is not None:
            return decision
    return None


def _local_gem_entry_has_nonclean_marker(entry: dict[str, Any], metrics: dict[str, Any]) -> bool:
    if (
        _validation_boolish(
            entry.get("protocol_integrity_passed", metrics.get("protocol_integrity_passed"))
        )
        is False
    ):
        return True
    for key in ("promotion_eligible", "clean_promotion_eligible"):
        if _validation_boolish(entry.get(key, metrics.get(key))) is False:
            return True
    for key in (
        "hard_constraint_violation_count",
        "hard_constraint_violations",
        "hard_constraint_failures",
        "constraint_violation_count",
        "constraint_violations",
    ):
        value = entry.get(key, metrics.get(key))
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (list, tuple, set, dict)):
            if len(value) > 0:
                return True
            continue
        if isinstance(value, (int, float)):
            if value > 0:
                return True
            continue
        text = str(value).strip().lower()
        if not text or text in {"0", "none", "no", "false", "[]", "{}"}:
            continue
        return True
    return False


def _entry_generation_value(entry: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    admission = (
        entry.get("admission_metrics") if isinstance(entry.get("admission_metrics"), dict) else {}
    )
    for key in keys:
        for source in (entry, metrics, admission):
            value = source.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _entry_commit_generation_id(entry: dict[str, Any]) -> int | None:
    return _entry_generation_value(
        entry,
        (
            "generation_id",
            "gen_id",
            "admission_generation_id",
            "promoted_generation_id",
        ),
    )


def _entry_source_generation_id(entry: dict[str, Any]) -> int | None:
    return _entry_generation_value(
        entry,
        (
            "source_generation_id",
            "source_gen_id",
            "completed_generation",
            "completed_gen_id",
        ),
    )


def _entry_generation_id(entry: dict[str, Any]) -> int | None:
    """Return the generation that committed an entry, not its source generation."""

    commit_generation = _entry_commit_generation_id(entry)
    return (
        commit_generation if commit_generation is not None else _entry_source_generation_id(entry)
    )


def _entry_visible_at_generation(
    entry: dict[str, Any],
    up_to_generation: int,
    *,
    commit_generation_hint: int | None = None,
    allow_source_as_commit: bool = False,
) -> bool:
    if up_to_generation < 0:
        return True
    if commit_generation_hint is not None and (
        commit_generation_hint < 0 or commit_generation_hint > up_to_generation
    ):
        return False
    commit_generation = _entry_commit_generation_id(entry)
    if commit_generation is None:
        commit_generation = commit_generation_hint
    if commit_generation is None and allow_source_as_commit:
        commit_generation = _entry_source_generation_id(entry)
    if commit_generation is None or commit_generation < 0 or commit_generation > up_to_generation:
        return False
    source_generation = _entry_source_generation_id(entry)
    return source_generation is None or source_generation <= up_to_generation


def _entry_target_lane(entry: dict[str, Any]) -> str:
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    return str(
        entry.get("promoted_for_lane")
        or metrics.get("promoted_for_lane")
        or entry.get("frontier_lane")
        or metrics.get("frontier_lane")
        or ""
    ).strip()


def _lane_allows_non_promotable(manifest: dict[str, Any], lane_name: str) -> bool:
    lanes = manifest.get("frontier_lanes")
    if not isinstance(lanes, list):
        return False
    return any(
        isinstance(lane, dict)
        and str(lane.get("name") or "").strip() == lane_name
        and bool(lane.get("allow_non_promotable"))
        for lane in lanes
    )


def _compact_frontier_lane_entry(entry: dict[str, Any], lane_name: str) -> dict[str, Any]:
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    promotion_eligible = _validation_boolish(
        entry.get("promotion_eligible", metrics.get("promotion_eligible"))
    )
    parent_eligible = _validation_boolish(
        entry.get("parent_eligible", metrics.get("parent_eligible"))
    )
    if promotion_eligible is False:
        parent_eligible = False
    compact = {
        "generation_id": entry.get("generation_id"),
        "rank": entry.get("rank"),
        "variant_name": entry.get("variant_name", ""),
        "metric_value": _coerce_float(entry.get("lane_metric_value", entry.get("metric_value"))),
        "metric_name": entry.get("lane_metric_name") or entry.get("metric_name", ""),
        "metric_direction": entry.get("lane_metric_direction") or entry.get("metric_direction", ""),
        "metrics": entry.get("metrics", {}),
        "finding_id": entry.get("finding_id", ""),
        "promoted_at": entry.get("promoted_at", ""),
        "frontier_lane": lane_name,
        "promoted_for_lane": entry.get("promoted_for_lane", lane_name),
        "source_frontier_lane": entry.get("source_frontier_lane", ""),
        "risk_violation_reason": entry.get("risk_violation_reason", ""),
    }
    if promotion_eligible is not None:
        compact["promotion_eligible"] = promotion_eligible
    if parent_eligible is not None:
        compact["parent_eligible"] = parent_eligible
    return compact


def _canonical_frontier_entry_count(
    manifest: dict[str, Any],
    *,
    up_to_generation: int = -1,
    historical_lane_frontiers: dict[str, list[dict[str, Any]]] | None = None,
) -> int:
    def count_entries(entries: Any, generation_hint: int | None = None) -> int:
        if not isinstance(entries, list):
            return 0
        count = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if not _entry_visible_at_generation(
                entry,
                up_to_generation,
                commit_generation_hint=generation_hint,
            ):
                continue
            count += 1
        return count

    if historical_lane_frontiers is not None:
        return sum(count_entries(entries) for entries in historical_lane_frontiers.values())

    generations = manifest.get("generations")
    generation_count = 0
    if isinstance(generations, dict):
        generation_count = sum(
            count_entries(entries, _coerce_int(gen_key, default=-1))
            for gen_key, entries in generations.items()
        )
    lane_frontiers = manifest.get("lane_frontiers")
    if isinstance(lane_frontiers, dict):
        lane_count = sum(count_entries(entries) for entries in lane_frontiers.values())
        if lane_count:
            return lane_count
    if isinstance(generations, dict) and generation_count:
        return generation_count
    return count_entries(manifest.get("cumulative_top"))


def _compact_lane_frontiers(
    manifest: dict[str, Any],
    *,
    up_to_generation: int = -1,
    maturity_policy: dict[str, Any] | None = None,
    trust_committed_membership: bool = False,
    skipped_by_reason: dict[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    raw = manifest.get("lane_frontiers")
    if not isinstance(raw, dict):
        return {}
    compact: dict[str, list[dict[str, Any]]] = {}
    for lane_name, entries in raw.items():
        if not isinstance(entries, list):
            continue
        lane_entries: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                if skipped_by_reason is not None:
                    skipped_by_reason["malformed_entry"] = (
                        skipped_by_reason.get("malformed_entry", 0) + 1
                    )
                continue
            if not _entry_visible_at_generation(
                entry,
                up_to_generation,
                allow_source_as_commit=not trust_committed_membership,
            ):
                continue
            durable, reason = _frontier_entry_durability(
                entry,
                maturity_policy=maturity_policy,
                trust_committed_membership=trust_committed_membership,
                allow_non_promotable=_lane_allows_non_promotable(
                    manifest,
                    str(lane_name),
                ),
            )
            if not durable:
                if skipped_by_reason is not None:
                    skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1
                continue
            lane_entries.append(_compact_frontier_lane_entry(entry, str(lane_name)))
        compact[str(lane_name)] = lane_entries
    return compact


def _compact_historical_lane_frontiers(
    manifest: dict[str, Any],
    *,
    up_to_generation: int,
    maturity_policy: dict[str, Any] | None,
    trust_committed_membership: bool,
    skipped_by_reason: dict[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]] | None:
    generations = manifest.get("generations")
    if not isinstance(generations, dict) or up_to_generation < 0:
        return None
    configured_lanes = [
        lane
        for lane in manifest.get("frontier_lanes", []) or []
        if isinstance(lane, dict) and str(lane.get("name") or "").strip()
    ]
    if not configured_lanes:
        return None
    generation_ids = [
        gen_id for key in generations if (gen_id := _coerce_int(key, default=-1)) >= 0
    ]
    if not generation_ids or up_to_generation >= max(generation_ids):
        return None

    bounded_generations: dict[str, list[dict[str, Any]]] = {}
    for generation_key, entries in generations.items():
        generation_hint = _coerce_int(generation_key, default=-1)
        if not isinstance(entries, list):
            continue
        bounded_entries: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if not _entry_visible_at_generation(
                entry,
                up_to_generation,
                commit_generation_hint=generation_hint,
            ):
                continue
            durable, reason = _frontier_entry_durability(
                entry,
                maturity_policy=maturity_policy,
                trust_committed_membership=trust_committed_membership,
                allow_non_promotable=_lane_allows_non_promotable(
                    manifest,
                    _entry_target_lane(entry),
                ),
            )
            if not durable:
                if skipped_by_reason is not None:
                    skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1
                continue
            bounded_entries.append(entry)
        if bounded_entries:
            bounded_generations[str(generation_key)] = bounded_entries

    from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
        _build_cumulative_lane_views,
    )

    raw_lanes, _ = _build_cumulative_lane_views(
        bounded_generations,
        configured_lanes,
        maturity_policy=maturity_policy,
        primary_metric=str(manifest.get("primary_metric") or "metric_value"),
        metric_direction=str(manifest.get("metric_direction") or "maximize"),
        promote_top_k=max(1, len(manifest.get("cumulative_top") or [])),
        entry_is_committed=lambda _entry: True,
        trust_recorded_lane_membership=trust_committed_membership,
    )
    return {
        lane_name: [_compact_frontier_lane_entry(entry, lane_name) for entry in entries]
        for lane_name, entries in raw_lanes.items()
    }


def _compact_validation_candidates(
    manifest: dict[str, Any],
    *,
    up_to_generation: int = -1,
    limit: int | None = _VALIDATION_CANDIDATES_OUTPUT_CAP,
    run_dir: Path | None = None,
    maturity_policy: dict[str, Any] | None = None,
    trust_committed_membership: bool = False,
) -> list[dict[str, Any]]:
    raw = manifest.get("validation_candidates")
    if not isinstance(raw, dict):
        raw = {}
    manifest_signal_only = bool(manifest.get("_non_runtime_signal_source"))
    manifest_signal_status = str(manifest.get("_non_runtime_signal_status") or "").strip()

    compact_by_record: dict[str, dict[str, Any]] = {}
    durable_entity_keys = (
        set()
        if manifest_signal_only
        else _durable_validation_entity_keys(
            manifest,
            up_to_generation=up_to_generation,
            run_dir=run_dir,
            maturity_policy=maturity_policy,
            trust_committed_membership=trust_committed_membership,
        )
    )
    durable_identity_aliases = (
        set()
        if manifest_signal_only
        else _durable_validation_identity_aliases(
            manifest,
            up_to_generation=up_to_generation,
            run_dir=run_dir,
            maturity_policy=maturity_policy,
            trust_committed_membership=trust_committed_membership,
        )
    )

    def merge_aliases(target: dict[str, Any], source: dict[str, Any]) -> None:
        aliases = {
            str(value).strip()
            for value in _iter_validation_identity_values(target)
            if value not in (None, "", [], {})
        }
        aliases.update(
            str(value).strip()
            for value in _iter_validation_identity_values(source)
            if value not in (None, "", [], {})
        )
        for key in _VALIDATION_IDENTITY_KEYS:
            if target.get(key) not in (None, "", [], {}):
                continue
            value = source.get(key)
            if value not in (None, "", [], {}):
                target[key] = value
        if aliases:
            target["identity_aliases"] = sorted(aliases)

    def add_entry(
        entry: dict[str, Any],
        generation_hint: int | None = None,
        *,
        replace_existing: bool = True,
    ) -> None:
        if not _entry_visible_at_generation(
            entry,
            up_to_generation,
            commit_generation_hint=generation_hint,
        ):
            return
        gen_id = _entry_commit_generation_id(entry)
        if gen_id is None:
            gen_id = generation_hint if generation_hint is not None else -1
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        if not _validation_candidate_is_retainable(entry):
            return

        def pick_metadata(key: str, default: Any = "") -> Any:
            value = entry.get(key)
            if value in (None, ""):
                value = metrics.get(key)
            return default if value is None else value

        compact_entry = {
            "generation_id": entry.get("generation_id", gen_id if gen_id >= 0 else None),
            "variant_name": entry.get("variant_name", ""),
            "metric_name": entry.get("metric_name", ""),
            "metric_value": _coerce_float(entry.get("metric_value")),
            "metric_direction": entry.get("metric_direction", ""),
            "signal_source": entry.get("signal_source", "")
            or ("non_runtime_frontier_manifest" if manifest_signal_only else ""),
            "signal_source_priority": entry.get("signal_source_priority"),
            "finding_id": entry.get("finding_id", ""),
            "submitted_frontier_lane": entry.get("submitted_frontier_lane", ""),
            "matched_frontier_lanes": entry.get("matched_frontier_lanes", []),
            "signal_axis_lanes": entry.get("signal_axis_lanes", []),
            "evidence_stage": pick_metadata("evidence_stage"),
            "evidence_maturity_rank": pick_metadata("evidence_maturity_rank", None),
            "result_status": pick_metadata("result_status"),
            "protocol_integrity_status": pick_metadata("protocol_integrity_status"),
            "scout_only": pick_metadata("scout_only", None),
            "scored_cell_count": pick_metadata("scored_cell_count", None),
            "excluded_from_durable_frontier": pick_metadata("excluded_from_durable_frontier", None),
            "frontier_entity_key": pick_metadata("frontier_entity_key"),
            "exclusion_reason": pick_metadata("exclusion_reason"),
            "recommended_next_step": pick_metadata("recommended_next_step"),
            **{key: pick_metadata(key) for key in _VALIDATION_RESEARCH_METADATA_KEYS},
            **{key: pick_metadata(key) for key in _VALIDATION_DIVERSITY_METADATA_KEYS},
        }
        if manifest_signal_only:
            compact_entry["artifact_signal_source"] = "non_runtime_frontier_manifest"
            compact_entry["artifact_signal_status"] = manifest_signal_status
            compact_entry["durability_scope"] = "validation_signal_only"
        raw_aliases = entry.get("identity_aliases")
        metric_aliases = metrics.get("identity_aliases")
        aliases = []
        for raw in (raw_aliases, metric_aliases):
            if isinstance(raw, list):
                aliases.extend(item for item in raw if item not in (None, ""))
        if aliases:
            compact_entry["identity_aliases"] = sorted(
                {str(item).strip() for item in aliases if item}
            )
        for key in _VALIDATION_IDENTITY_KEYS:
            if compact_entry.get(key) not in (None, "", [], {}):
                continue
            value = pick_metadata(key)
            if value not in (None, "", [], {}):
                compact_entry[key] = value
        for optional_key in (
            "metric_value",
            "evidence_stage",
            "evidence_maturity_rank",
            "result_status",
            "protocol_integrity_status",
            "scout_only",
            "scored_cell_count",
            "excluded_from_durable_frontier",
            "frontier_entity_key",
            "submitted_frontier_lane",
            "matched_frontier_lanes",
            "signal_axis_lanes",
            "exclusion_reason",
            "recommended_next_step",
            "artifact_signal_source",
            "artifact_signal_status",
            "durability_scope",
            "identity_aliases",
            *_VALIDATION_IDENTITY_KEYS,
            *_VALIDATION_RESEARCH_METADATA_KEYS,
            *_VALIDATION_DIVERSITY_METADATA_KEYS,
        ):
            if compact_entry.get(optional_key) in ("", None, [], {}):
                compact_entry.pop(optional_key, None)
        entity_key = _validation_candidate_entity_key(entry)
        if entity_key and not compact_entry.get("frontier_entity_key"):
            compact_entry["frontier_entity_key"] = entity_key
        compact_aliases = {
            str(value).strip()
            for value in _iter_validation_retirement_values(compact_entry)
            if value not in (None, "", [], {})
        }
        if entity_key in durable_entity_keys or compact_aliases & durable_identity_aliases:
            return
        record_key = _validation_candidate_record_key(compact_entry, entity_key)
        incumbent = compact_by_record.get(record_key)
        if not replace_existing and incumbent is not None:
            merge_aliases(incumbent, compact_entry)
            return
        if not replace_existing and entity_key:
            for existing in compact_by_record.values():
                if existing.get("frontier_entity_key") == entity_key:
                    merge_aliases(existing, compact_entry)
                    return
        if incumbent is None or _validation_candidate_sort_key(
            compact_entry
        ) > _validation_candidate_sort_key(incumbent):
            if incumbent is not None:
                merge_aliases(compact_entry, incumbent)
            compact_by_record[record_key] = compact_entry
        elif incumbent is not None:
            merge_aliases(incumbent, compact_entry)

    generations = raw.get("generations")
    if isinstance(generations, dict):
        for gen_key, gen_entries in sorted(
            generations.items(),
            key=lambda item: _coerce_int(item[0], default=-1),
        ):
            generation_hint = _coerce_int(gen_key, default=-1)
            if up_to_generation >= 0 and (
                generation_hint < 0 or generation_hint > up_to_generation
            ):
                continue
            if not isinstance(gen_entries, list):
                continue
            for entry in gen_entries:
                if isinstance(entry, dict):
                    add_entry(entry, generation_hint=generation_hint)

    entries = raw.get("cumulative")
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict):
                add_entry(entry, replace_existing=False)

    if manifest_signal_only:
        for entry in manifest.get("cumulative_top") or []:
            if isinstance(entry, dict):
                add_entry(entry, replace_existing=False)
        signal_generations = manifest.get("generations")
        if isinstance(signal_generations, dict):
            for gen_key, gen_entries in sorted(
                signal_generations.items(),
                key=lambda item: _coerce_int(item[0], default=-1),
            ):
                generation_hint = _coerce_int(gen_key, default=-1)
                if not isinstance(gen_entries, list):
                    continue
                for entry in gen_entries:
                    if isinstance(entry, dict):
                        add_entry(entry, generation_hint=generation_hint, replace_existing=False)
        signal_lanes = manifest.get("lane_frontiers")
        if isinstance(signal_lanes, dict):
            for lane_name, lane_entries in signal_lanes.items():
                if not isinstance(lane_entries, list):
                    continue
                for entry in lane_entries:
                    if not isinstance(entry, dict):
                        continue
                    candidate = dict(entry)
                    candidate.setdefault("submitted_frontier_lane", lane_name)
                    add_entry(candidate, replace_existing=False)

    sorted_candidates = sorted(
        compact_by_record.values(),
        key=_validation_candidate_sort_key,
        reverse=True,
    )
    return sorted_candidates if limit is None else sorted_candidates[: max(0, int(limit))]


def _validation_candidate_record_key(entry: dict[str, Any], entity_key: str = "") -> str:
    parts = [
        str(entry.get("generation_id") or ""),
        str(entry.get("finding_id") or ""),
        str(entry.get("variant_name") or ""),
        str(entry.get("metric_name") or ""),
        str(entry.get("signal_source") or ""),
        str(entry.get("source_path") or ""),
        str(entry.get("result_path") or ""),
        str(entry.get("source_result_path") or ""),
        str(entry.get("result_artifact_path") or ""),
    ]
    key = "\x1f".join(part.strip().lower() for part in parts if part and part != "None")
    return key or entity_key or json.dumps(entry, sort_keys=True, default=str)


def _validation_candidate_entity_key(entry: dict[str, Any]) -> str:
    try:
        from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
            _candidate_entity_key,
        )

        return _candidate_entity_key(entry)
    except (AttributeError, ImportError):
        logger.debug("falling back to local validation entity key", exc_info=True)
    frontier_key = _normalized_validation_token(entry.get("frontier_entity_key"))
    if frontier_key:
        if ":" in frontier_key and "::" not in frontier_key:
            prefix, _, payload = frontier_key.partition(":")
            if payload:
                return f"{prefix}::{payload}"
        if "::" in frontier_key:
            prefix, _, payload = frontier_key.partition("::")
            return f"{prefix}::{payload}" if payload else ""
        return frontier_key
    variant = _normalized_validation_token(entry.get("variant_name"))
    if variant:
        return f"variant::{variant}"
    finding_id = _normalized_validation_token(entry.get("finding_id"))
    if finding_id:
        return f"finding::{finding_id}"
    return _normalized_validation_token(json.dumps(entry, sort_keys=True, default=str)) or "entry"


def _durable_validation_entity_keys(
    manifest: dict[str, Any],
    *,
    up_to_generation: int = -1,
    run_dir: Path | None = None,
    maturity_policy: dict[str, Any] | None = None,
    trust_committed_membership: bool = False,
) -> set[str]:
    keys: set[str] = set()

    def add_entries(entries: Any, generation_hint: int | None = None) -> None:
        if not isinstance(entries, list):
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if not _entry_visible_at_generation(
                entry,
                up_to_generation,
                commit_generation_hint=generation_hint,
            ):
                continue
            if not _frontier_entry_is_durable(
                entry,
                maturity_policy=maturity_policy,
                trust_committed_membership=trust_committed_membership,
                allow_non_promotable=_lane_allows_non_promotable(
                    manifest,
                    _entry_target_lane(entry),
                ),
            ):
                continue
            keys.add(_validation_candidate_entity_key(entry))

    add_entries(manifest.get("cumulative_top"))
    generations = manifest.get("generations")
    if isinstance(generations, dict):
        for gen_key, entries in generations.items():
            add_entries(entries, generation_hint=_coerce_int(gen_key, default=-1))
    lane_frontiers = manifest.get("lane_frontiers")
    if isinstance(lane_frontiers, dict):
        for entries in lane_frontiers.values():
            add_entries(entries)
    gems = manifest.get("gems")
    gem_entries = gems.get("entries") if isinstance(gems, dict) else None
    if isinstance(gem_entries, list):
        for entry in gem_entries:
            if not isinstance(entry, dict):
                continue
            eligibility_entry = _gem_entry_with_manifest_selection_policy(
                entry,
                gems,
                frontier_manifest=manifest,
            )
            if not _gem_entry_is_prompt_eligible(
                eligibility_entry,
                maturity_policy=maturity_policy,
            ):
                continue
            source_generation = _resolved_persisted_gem_source_generation_id(run_dir, entry)
            if up_to_generation >= 0 and (
                source_generation is None or source_generation > up_to_generation
            ):
                continue
            keys.add(_validation_candidate_entity_key(_gem_validation_identity_entry(entry)))
    return {key for key in keys if key}


def _durable_validation_identity_aliases(
    manifest: dict[str, Any],
    *,
    up_to_generation: int = -1,
    run_dir: Path | None = None,
    maturity_policy: dict[str, Any] | None = None,
    trust_committed_membership: bool = False,
) -> set[str]:
    aliases: set[str] = set()

    def add_entry(entry: dict[str, Any]) -> None:
        aliases.update(
            str(value).strip()
            for value in _iter_validation_retirement_values(entry)
            if value not in (None, "", [], {})
        )
        key = _validation_candidate_entity_key(entry)
        if key:
            aliases.add(key)

    def add_entries(entries: Any, generation_hint: int | None = None) -> None:
        if not isinstance(entries, list):
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if not _entry_visible_at_generation(
                entry,
                up_to_generation,
                commit_generation_hint=generation_hint,
            ):
                continue
            if _frontier_entry_is_durable(
                entry,
                maturity_policy=maturity_policy,
                trust_committed_membership=trust_committed_membership,
                allow_non_promotable=_lane_allows_non_promotable(
                    manifest,
                    _entry_target_lane(entry),
                ),
            ):
                add_entry(entry)

    add_entries(manifest.get("cumulative_top"))
    generations = manifest.get("generations")
    if isinstance(generations, dict):
        for gen_key, entries in generations.items():
            add_entries(entries, generation_hint=_coerce_int(gen_key, default=-1))
    lane_frontiers = manifest.get("lane_frontiers")
    if isinstance(lane_frontiers, dict):
        for entries in lane_frontiers.values():
            add_entries(entries)
    gems = manifest.get("gems")
    gem_entries = gems.get("entries") if isinstance(gems, dict) else None
    if isinstance(gem_entries, list):
        for entry in gem_entries:
            if not isinstance(entry, dict):
                continue
            eligibility_entry = _gem_entry_with_manifest_selection_policy(
                entry,
                gems,
                frontier_manifest=manifest,
            )
            if not _gem_entry_is_prompt_eligible(
                eligibility_entry,
                maturity_policy=maturity_policy,
            ):
                continue
            source_generation = _resolved_persisted_gem_source_generation_id(run_dir, entry)
            if up_to_generation >= 0 and (
                source_generation is None or source_generation > up_to_generation
            ):
                continue
            add_entry(_gem_validation_identity_entry(entry))
    return {alias for alias in aliases if alias}


def _gem_validation_identity_entry(entry: dict[str, Any]) -> dict[str, Any]:
    payload = dict(entry)
    admission = entry.get("admission_metrics")
    metrics = dict(admission) if isinstance(admission, dict) else {}
    existing_metrics = entry.get("metrics")
    if isinstance(existing_metrics, dict):
        metrics.update(existing_metrics)
    if metrics:
        payload["metrics"] = metrics
    return payload


def _validation_boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    token = str(value or "").strip().lower()
    if token in {
        "true",
        "yes",
        "1",
        "full",
        "complete",
        "completed",
        "confirmed",
        "eligible",
        "promotable",
        "clean",
        "passed",
    }:
        return True
    if token in {
        "false",
        "no",
        "0",
        "partial",
        "incomplete",
        "scout",
        "smoke",
        "ineligible",
        "nonpromotable",
        "non_promotable",
        "blocked",
        "failed",
    }:
        return False
    return None


def _validation_entry_is_preliminary(entry: dict[str, Any]) -> bool:
    try:
        from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
            _is_preliminary_or_incomplete_evidence,
        )

        return _is_preliminary_or_incomplete_evidence(entry)
    except (AttributeError, ImportError):
        logger.debug("falling back to local validation maturity check", exc_info=True)
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    for key in (
        "excluded_from_durable_frontier",
        "scout_only",
        "is_scout_eval",
        "is_smoke_eval",
        "smoke_only",
        "unscored_artifact",
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
    ):
        if _validation_boolish(entry.get(key, metrics.get(key))) is True:
            return True
    protocol_status = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(
            entry.get("protocol_integrity_status") or metrics.get("protocol_integrity_status") or ""
        )
        .strip()
        .lower(),
    ).strip("_")
    if protocol_status in {"failed", "fail", "invalid", "protocol_invalid"}:
        return True
    for key in (
        "suspect_protocol",
        # Legacy input alias only; canonical outputs use suspect_protocol.
        "suspect_fixed_weight_eval",
        "protocol_integrity_failed",
    ):
        if _validation_boolish(entry.get(key, metrics.get(key))) is True:
            return True
    for key in (
        "scored_complete",
        "is_scored_complete",
        "complete_eval",
        "is_complete_eval",
    ):
        if _validation_boolish(entry.get(key, metrics.get(key))) is False:
            return True
    for key in (
        "tier_status",
        "final_status",
        "result_status",
        "status",
        "completion_status",
        "eval_status",
        "scoring_status",
    ):
        status = str(entry.get(key) or metrics.get(key) or "").strip().lower()
        if status in {
            "smoke",
            "unscored",
            "un_scored",
            "unscored_artifact",
            "not_scored",
            "scout",
            "cheap_probe",
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
            "protocol_invalid",
        }:
            return True
    if (
        str(entry.get("exclusion_reason") or metrics.get("exclusion_reason") or "").strip().lower()
        == "preliminary_or_incomplete_evidence"
    ):
        return True
    for key in ("evidence_stage", "eval_stage", "stage"):
        stage = str(entry.get(key) or metrics.get(key) or "").strip().lower().replace("-", "_")
        if stage in {
            "scout",
            "cheap_probe",
            "probe",
            "smoke",
            "sanity",
            "partial",
            "preliminary",
            "prelim",
            "incomplete",
            "partial_cohort",
            "partial_eval",
            "summary_only",
            "unscored",
            "un_scored",
            "unscored_artifact",
            "failed_or_unscored",
            "capped",
            "capped_at",
            "cap_at",
        }:
            return True
    return False


def _validation_status_has_any(text: str, *needles: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower()).strip("_")
    if not normalized:
        return False
    tokens = [token for token in normalized.split("_") if token]
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


def _validation_candidate_is_retainable(entry: dict[str, Any]) -> bool:
    try:
        from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
            _any_boolish_candidate_field_true,
            _candidate_has_bad_runtime_status,
            _candidate_status_has_any,
            _normalized_evidence_stage,
        )

        if _candidate_has_bad_runtime_status(entry):
            return False
        if _any_boolish_candidate_field_true(entry, "summary_only", "is_summary_only"):
            return False
        if _candidate_status_has_any(entry, "summary_only"):
            return False
        _normalized_evidence_stage(entry)
        return True
    except (AttributeError, ImportError):
        logger.debug(
            "falling back to local validation candidate retainability check", exc_info=True
        )
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    status_text = " ".join(
        str(entry.get(key) or metrics.get(key) or "")
        for key in (
            "tier_status",
            "final_status",
            "result_status",
            "status",
            "completion_status",
            "eval_status",
            "scoring_status",
            "exclusion_reason",
            "protocol_integrity_status",
        )
    )
    if _validation_status_has_any(
        status_text,
        "summary_only",
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
    ):
        return False
    return not any(
        _validation_boolish(entry.get(key, metrics.get(key))) is True
        for key in ("summary_only", "is_summary_only")
    )


def _frontier_entry_rejection_reason(
    entry: dict[str, Any],
    maturity_policy: dict[str, Any] | None,
) -> str:
    try:
        from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
            _candidate_has_bad_runtime_status,
            _candidate_has_validation_only_durability_marker,
            _candidate_protocol_integrity_failed,
        )

        if _candidate_protocol_integrity_failed(entry):
            return "protocol_integrity_failed"
        if _candidate_has_bad_runtime_status(entry):
            return "bad_runtime_status"
        if _candidate_has_validation_only_durability_marker(entry):
            return "validation_signal_only"
    except (AttributeError, ImportError):
        logger.debug("falling back to local frontier rejection reason", exc_info=True)
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    if (
        _validation_boolish(
            entry.get(
                "excluded_from_durable_frontier",
                metrics.get("excluded_from_durable_frontier"),
            )
        )
        is True
    ):
        return "explicit_durable_exclusion"
    maturity = evidence_maturity_snapshot(entry, maturity_policy)
    if maturity.get("mature_enough") is False:
        return "maturity_policy_rejected"
    if _validation_entry_is_preliminary(entry):
        return "preliminary_or_incomplete"
    return "durability_contract_rejected"


def _legacy_entry_has_identity_and_measurement(entry: dict[str, Any]) -> bool:
    has_identity = any(
        entry.get(key) not in (None, "", [], {}) for key in _VALIDATION_IDENTITY_KEYS
    )
    if not has_identity:
        return False
    for key in ("metric_value", "lane_metric_value"):
        if _coerce_float(entry.get(key)) is not None:
            return True
    for block_name in ("metrics", "admission_metrics"):
        block = entry.get(block_name)
        if not isinstance(block, dict):
            continue
        for key, value in block.items():
            if isinstance(value, bool) or "generation" in str(key).lower():
                continue
            if _coerce_float(value) is not None:
                return True
    return False


def _frontier_entry_durability(
    entry: dict[str, Any],
    *,
    maturity_policy: dict[str, Any] | None = None,
    trust_committed_membership: bool = False,
    allow_non_promotable: bool = False,
) -> tuple[bool, str]:
    """Classify signals; committed runtime membership is already authoritative."""

    if not isinstance(entry, dict):
        return False, "malformed_entry"
    if trust_committed_membership:
        return True, "committed_membership"

    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    if not _legacy_entry_has_identity_and_measurement(entry):
        return False, "legacy_entry_missing_identity_or_measurement"
    if (
        _validation_boolish(
            entry.get(
                "excluded_from_durable_frontier",
                metrics.get("excluded_from_durable_frontier"),
            )
        )
        is True
    ):
        return False, "explicit_durable_exclusion"
    if allow_non_promotable:
        try:
            from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
                _candidate_has_bad_runtime_status,
                _candidate_has_non_authorizable_incomplete_marker,
                _candidate_protocol_integrity_failed,
            )

            hard_failure = (
                _candidate_protocol_integrity_failed(entry)
                or _candidate_has_bad_runtime_status(entry)
                or _candidate_has_non_authorizable_incomplete_marker(entry)
            )
        except (AttributeError, ImportError):
            hard_failure = _validation_entry_is_preliminary(entry)
        promotion_exclusion = durable_promotion_exclusion(entry)
        if promotion_exclusion in {
            "promotion_eligible=false",
            "clean_promotion_eligible=false",
        }:
            recorded_maturity = _recorded_task_maturity_decision(entry)
            policy_rejected = (
                maturity_policy is not None
                and evidence_maturity_snapshot(entry, maturity_policy).get("mature_enough") is False
            )
            if hard_failure or recorded_maturity is False or policy_rejected:
                return False, _frontier_entry_rejection_reason(entry, maturity_policy)
            return True, "committed_non_promotable_lane"
    if maturity_policy is None and _validation_entry_is_preliminary(entry):
        return False, "preliminary_or_incomplete"
    try:
        from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
            _is_committed_frontier_entry,
        )

        durable = _is_committed_frontier_entry(entry, maturity_policy)
        return (
            (True, "committed_validated")
            if durable
            else (False, _frontier_entry_rejection_reason(entry, maturity_policy))
        )
    except (AttributeError, ImportError):
        logger.debug("falling back to local frontier durability check", exc_info=True)
    if _validation_entry_is_preliminary(entry):
        return False, "preliminary_or_incomplete"
    durable = _validation_entry_has_complete_evidence(entry, metrics)
    return (
        (True, "durable_fallback")
        if durable
        else (False, _frontier_entry_rejection_reason(entry, maturity_policy))
    )


def _frontier_entry_is_durable(
    entry: dict[str, Any],
    *,
    maturity_policy: dict[str, Any] | None = None,
    trust_committed_membership: bool = False,
    allow_non_promotable: bool = False,
) -> bool:
    return _frontier_entry_durability(
        entry,
        maturity_policy=maturity_policy,
        trust_committed_membership=trust_committed_membership,
        allow_non_promotable=allow_non_promotable,
    )[0]


def _normalized_validation_token(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        text = str(value).strip().lower()
    except (TypeError, ValueError):
        return ""
    return " ".join(text.split())


def _validation_candidate_sort_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    metric_value = _coerce_float(entry.get("metric_value"))
    if metric_value is None:
        directional_value = float("-inf")
    else:
        direction = str(entry.get("metric_direction") or "maximize")
        if direction not in {"maximize", "minimize"}:
            direction = "maximize"
        directional_value = metric_value if direction == "maximize" else -metric_value
    return (
        _coerce_int(entry.get("signal_source_priority"), default=0),
        _coerce_int(entry.get("evidence_maturity_rank"), default=0),
        directional_value,
        _coerce_int(entry.get("generation_id"), default=-1),
        str(entry.get("variant_name") or entry.get("finding_id") or ""),
    )


def _lane_priority_entries(
    lane_frontiers: dict[str, list[dict[str, Any]]],
    *,
    manifest: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    preferred = _configured_lane_order(manifest or {}, lane_frontiers)
    ordered: list[dict[str, Any]] = []
    max_len = max((len(entries) for entries in lane_frontiers.values()), default=0)
    for idx in range(max_len):
        for lane in preferred:
            entries = lane_frontiers.get(lane) or []
            if idx < len(entries):
                ordered.append(entries[idx])
    return ordered


def _configured_lane_order(
    manifest: dict[str, Any],
    lane_frontiers: dict[str, list[dict[str, Any]]],
) -> list[str]:
    configured: list[str] = []
    raw_lanes = manifest.get("frontier_lanes")
    if isinstance(raw_lanes, list):
        for lane in raw_lanes:
            name = ""
            if isinstance(lane, dict):
                name = str(lane.get("name") or "").strip()
            elif isinstance(lane, str):
                name = lane.strip()
            if name and name in lane_frontiers and name not in configured:
                configured.append(name)
    if configured:
        configured.extend(sorted(set(lane_frontiers) - set(configured)))
        return configured
    preferred = [
        "confirmed",
        "performance",
        "candidate",
        "reference",
        "diagnostic",
        "process",
    ]
    ordered: list[str] = []
    seen_lanes: set[str] = set()
    for lane in preferred:
        if lane in lane_frontiers:
            ordered.append(lane)
            seen_lanes.add(lane)
    for lane in sorted(set(lane_frontiers) - seen_lanes):
        ordered.append(lane)
    return ordered


def _coerce_int(
    value: Any,
    *,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None and parsed < minimum:
        parsed = default if default >= minimum else minimum
    if maximum is not None and parsed > maximum:
        parsed = maximum
    return parsed


def _coerce_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


# ---------------------------------------------------------------------------
# Tool definition (new SDK API)
# ---------------------------------------------------------------------------

get_frontier = None

if tool is not None:
    get_frontier = tool(
        "get_frontier",
        (
            "Get durable frontier entries from previous generations. "
            "The separate validation_candidates list contains non-frontier scout/partial "
            "signals that need mature validation before frontier or Gems use."
        ),
        {
            "up_to_generation": int,
            "top_k": int,
            "inline_limit": int,
        },
    )(_handle_get_frontier)


def create_frontier_tools_server():
    """Create MCP server for frontier tools."""
    if create_sdk_mcp_server is None or tool is None:
        raise ImportError("claude_agent_sdk is required for MCP tools")
    return create_sdk_mcp_server(
        "frontier-tools",
        tools=[get_frontier],
    )


def create_tool_plugin() -> dict[str, object]:
    """Manifest entrypoint that exposes frontier query tools."""
    return {
        "tool_server_ref": "tool_server:frontier_tools",
        "server_name": "frontier-tools",
        "factory": "praxist.plugins.tools.frontier_tools.adapter:create_frontier_tools_server",
        "tool_names": ["get_frontier"],
        "visibility": ["peer", "panel"],
        "required_capability": "tool_server.frontier_tools",
        "handlers": {
            "get_frontier": "praxist.plugins.tools.frontier_tools.adapter:_handle_get_frontier",
        },
    }
