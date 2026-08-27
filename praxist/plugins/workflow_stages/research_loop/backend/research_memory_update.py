"""Research-memory ledger updates for completed generations."""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_MINIMIZE_AXIS_HINTS = (
    "loss",
    "cost",
    "risk",
    "gap",
    "overhead",
    "latency",
    "time",
    "duration",
    "sharpness",
    "memory",
    "failure",
    "violation",
    "error",
)

_METRIC_NAME_FIELDS = (
    ("metric_name", "metric_value", "metric_direction"),
    ("primary_metric_name", "primary_metric_value", "primary_metric_direction"),
    ("anchor_metric_name", "anchor_metric_value", "anchor_metric_direction"),
    ("lane_metric_name", "lane_metric_value", "lane_metric_direction"),
)

_ALWAYS_IGNORED_NUMERIC_METRIC_KEYS = {
    "generation_id",
    "source_generation_id",
    "candidate_generation_id",
    "seed",
    "metric_value",
    "primary_metric_value",
    "anchor_metric_value",
    "lane_metric_value",
}

_VALID_DIRECTIONS = {"maximize", "minimize"}


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, dict):
        value = value.get("mean")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _entry_metrics(entry: Any) -> dict[str, Any]:
    metrics = getattr(entry, "metrics", None)
    if metrics is None and isinstance(entry, dict):
        metrics = entry.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def _entry_text(entry: Any, key: str) -> str:
    value = getattr(entry, key, None)
    if value is None and isinstance(entry, dict):
        value = entry.get(key)
    return str(value or "").strip()


def _direction_for_axis(
    axis: str,
    entry: Any,
    metrics: dict[str, Any],
    *,
    named_direction_field: str | None = None,
    configured_direction: str | None = None,
    allow_inference: bool = True,
) -> str | None:
    configured = str(configured_direction or "").strip().lower()
    if configured in _VALID_DIRECTIONS:
        return configured
    direction_fields = ((named_direction_field,) if named_direction_field else ()) + (
        f"{axis}_direction",
    )
    for key in direction_fields:
        raw = metrics.get(key)
        if raw is None and isinstance(entry, dict):
            raw = entry.get(key)
        direction = str(raw or "").strip().lower()
        if direction in _VALID_DIRECTIONS:
            return direction
    raw_metric = metrics.get(axis)
    if isinstance(raw_metric, dict):
        direction = str(raw_metric.get("direction") or "").strip().lower()
        if direction in _VALID_DIRECTIONS:
            return direction
    if allow_inference:
        raw = metrics.get("metric_direction")
        if raw is None and isinstance(entry, dict):
            raw = entry.get("metric_direction")
        direction = str(raw or "").strip().lower()
        if direction in _VALID_DIRECTIONS:
            return direction
    if not allow_inference:
        return None
    axis_lower = axis.lower()
    if any(hint in axis_lower for hint in _MINIMIZE_AXIS_HINTS):
        return "minimize"
    return "maximize"


def _is_auxiliary_numeric_metric(axis: str, entry: Any, metrics: dict[str, Any]) -> bool:
    """Keep structural counters out of Pareto axes unless the task opts in.

    Raw result summaries commonly contain sample counts and elapsed-time fields.
    Their names vary by domain, so the generic runtime recognizes their shape
    instead of carrying a task-specific denylist. A task can still promote any
    count-like value to a real objective by declaring its axis direction.
    """
    normalized = axis.strip().lower()
    if normalized in _ALWAYS_IGNORED_NUMERIC_METRIC_KEYS:
        return True
    if not (
        normalized.startswith("n_")
        or normalized.endswith("_count")
        or normalized.endswith("_seconds")
    ):
        return False

    direction_key = f"{axis}_direction"
    candidates = [metrics.get(direction_key)]
    if isinstance(entry, dict):
        candidates.append(entry.get(direction_key))
    raw = metrics.get(axis)
    if isinstance(raw, dict):
        candidates.append(raw.get("direction"))
    return not any(
        str(value or "").strip().lower() in {"maximize", "minimize"} for value in candidates
    )


def _axis_spec(axis: Any) -> tuple[str, str]:
    if isinstance(axis, str):
        return axis.strip(), "maximize"
    if isinstance(axis, dict):
        name = str(axis.get("name") or axis.get("metric") or "").strip()
        direction = str(axis.get("direction") or "").strip().lower()
        return name, direction
    if isinstance(axis, (list, tuple)) and len(axis) >= 2:
        return str(axis[0] or "").strip(), str(axis[1] or "").strip().lower()
    return "", ""


def _configured_metric_directions(evaluation: Any) -> dict[str, str]:
    """Return unambiguous task-declared directions for comparative metrics."""

    declared: dict[str, set[str]] = {}

    def add(name: Any, direction: Any) -> None:
        metric_name = str(name or "").strip()
        normalized = str(direction or "").strip().lower()
        if metric_name and normalized in _VALID_DIRECTIONS:
            declared.setdefault(metric_name, set()).add(normalized)

    if evaluation is None:
        return {}
    add(
        getattr(evaluation, "primary_metric", ""),
        getattr(evaluation, "direction", ""),
    )
    for axis in getattr(evaluation, "anchor_metrics", None) or []:
        add(*_axis_spec(axis))
    for lane in getattr(evaluation, "frontier_lanes", None) or []:
        if not isinstance(lane, dict):
            continue
        for field_name in ("axes", "optional_axes"):
            for axis in lane.get(field_name) or []:
                add(*_axis_spec(axis))

    directions: dict[str, str] = {}
    for name, values in declared.items():
        if len(values) == 1:
            directions[name] = next(iter(values))
        else:
            logger.warning(
                "research_memory: metric %s has conflicting task-declared directions; "
                "requiring an unambiguous task declaration before using it as a delta anchor",
                name,
            )
    return directions


def _numeric_axes_for_entry(
    entry: Any,
    *,
    configured_directions: dict[str, str] | None = None,
) -> dict[str, tuple[float, str]]:
    metrics = _entry_metrics(entry)
    out: dict[str, tuple[float, str]] = {}
    use_declared_contract = configured_directions is not None
    directions = configured_directions or {}
    for name_field, value_field, direction_field in _METRIC_NAME_FIELDS:
        name = _entry_text(entry, name_field) or str(metrics.get(name_field) or "").strip()
        value = _finite_float(
            getattr(entry, value_field, None)
            if not isinstance(entry, dict)
            else entry.get(value_field, metrics.get(value_field))
        )
        if name and value is not None:
            if use_declared_contract and name not in directions:
                continue
            direction = _direction_for_axis(
                name,
                entry,
                metrics,
                named_direction_field=direction_field,
                configured_direction=directions.get(name),
                allow_inference=not use_declared_contract,
            )
            if direction is not None:
                out[name] = (value, direction)
    for key, raw in metrics.items():
        key_str = str(key).strip()
        if use_declared_contract and key_str not in directions:
            continue
        if not key_str or (
            key_str not in directions and _is_auxiliary_numeric_metric(key_str, entry, metrics)
        ):
            continue
        value = _finite_float(raw)
        if value is None:
            continue
        direction = _direction_for_axis(
            key_str,
            entry,
            metrics,
            configured_direction=directions.get(key_str),
            allow_inference=not use_declared_contract,
        )
        if direction is not None:
            out.setdefault(key_str, (value, direction))
    return out


def update_research_memory_post_gen(
    *,
    run_dir: Path,
    gen_id: int,
    findings: list[dict[str, Any]],
    promoted: list[Any],
    evaluation: Any | None = None,
) -> None:
    """Update research-memory ledgers for one completed generation.

    Ledger updates are advisory and must not block result preservation. Callers
    should catch exceptions around this function if they need a hard isolation
    boundary around post-generation bookkeeping.
    """
    from praxist.plugins.workflow_stages.research_loop.backend.research_memory.card_builder import (
        _detect_negative,
    )
    from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers import (
        FrontierDeltaLedger,
        NegativeEvidenceLedger,
        RoleROILedger,
    )

    fd = FrontierDeltaLedger(run_dir)
    prior_anchors = fd.prior_anchors_for_generation(gen_id)
    delta_records: list[dict[str, Any]] = []
    if promoted:
        configured_directions = (
            _configured_metric_directions(evaluation) if evaluation is not None else None
        )
        promoted_axes = [
            (p, _numeric_axes_for_entry(p, configured_directions=configured_directions))
            for p in promoted
        ]
        present_axes = {axis for _promoted, item_axes in promoted_axes for axis in item_axes}
        axes: dict[str, str] = {}
        if configured_directions:
            axes.update(
                (axis, direction)
                for axis, direction in configured_directions.items()
                if axis in present_axes
            )
        for _p, item_axes in promoted_axes:
            for axis, (_value, direction) in item_axes.items():
                axes.setdefault(axis, direction)
        for axis, direction in list(axes.items())[:24]:
            best = None
            best_val = None
            for p, item_axes in promoted_axes:
                axis_value = item_axes.get(axis)
                if axis_value is None:
                    continue
                mv, item_direction = axis_value
                compare_direction = item_direction or direction
                is_better = (
                    (best_val is None)
                    or (compare_direction == "maximize" and mv > best_val)
                    or (compare_direction == "minimize" and mv < best_val)
                )
                if is_better:
                    best_val = mv
                    best = p
            if best is None:
                continue
            variant = _entry_text(best, "variant_name")
            finding_id = _entry_text(best, "finding_id") or _entry_text(best, "id")
            delta_records.append(
                {
                    "axis": axis,
                    "previous_anchor": prior_anchors.get(axis),
                    "current_anchor": {
                        "variant": variant,
                        "value": best_val,
                        "finding_id": finding_id,
                        "direction": direction,
                    },
                }
            )
    try:
        fd.replace_generation(
            gen_id,
            delta_records,
            created_by=f"orchestrator_gen{gen_id}",
        )
    except Exception as e:  # noqa: BLE001 - derived ledger updates are best-effort.
        logger.debug("frontier_delta generation replacement failed: %s", e)

    neg = NegativeEvidenceLedger(run_dir)
    for f in findings or []:
        if not isinstance(f, dict):
            continue
        if _detect_negative(f):
            neg_id = f"NEG::{f.get('id', '')[:8]}"
            with contextlib.suppress(Exception):
                neg.add(
                    neg_id=neg_id,
                    title=(f.get("title") or "")[:200],
                    category="surprising_failure",
                    finding_id=f.get("id", ""),
                    summary=(f.get("notes") or "")[:200],
                    created_by=f"orchestrator_gen{gen_id}",
                )

    role_roi = RoleROILedger(run_dir)
    per_role: dict[str, dict[str, Any]] = {}
    for f in findings or []:
        if not isinstance(f, dict):
            continue
        extra = f.get("extra") or {}
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except Exception:  # noqa: BLE001 - malformed role metadata is non-fatal.
                extra = {}
        role = ""
        if isinstance(extra, dict):
            role = str(extra.get("peer_role") or "")
        ftype = f.get("finding_type") or "unknown"
        bucket = per_role.setdefault(
            role or "unspecified",
            {
                "total": 0,
                "by_type": {},
                "promoted_count": 0,
            },
        )
        bucket["total"] += 1
        bucket["by_type"][ftype] = bucket["by_type"].get(ftype, 0) + 1
    try:
        role_roi.record_gen_summary(
            generation_id=gen_id,
            per_role=per_role,
            created_by=f"orchestrator_gen{gen_id}",
        )
    except Exception as e:  # noqa: BLE001 - advisory ledger update.
        logger.debug("role_roi record_gen_summary failed: %s", e)

    logger.info(
        "research_memory: ledgers updated for gen %d (neg=%d, roles=%d, frontier_axes_recorded=%d)",
        gen_id,
        sum(1 for f in (findings or []) if isinstance(f, dict) and _detect_negative(f)),
        len(per_role),
        sum(1 for _ in fd.latest_per_axis()),
    )
