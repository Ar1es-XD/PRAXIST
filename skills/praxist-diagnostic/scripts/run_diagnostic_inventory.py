#!/usr/bin/env python3
"""Read-only Praxist run diagnostic inventory.

The script intentionally avoids mutations. It summarizes small run artifacts,
generation boundaries, result summaries, diversity labels, recent findings, and
lightweight hardware/process state so a Codex agent can reason from a stable
JSON snapshot instead of ad hoc shell output.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Direct source-checkout execution puts only this script directory on sys.path.
# Prefer the adjacent checkout when present; installed-package execution falls
# through to the environment's normal import resolution.
_SOURCE_ROOT = Path(__file__).resolve().parents[3]
if (_SOURCE_ROOT / "praxist").is_dir() and str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
    is_supported_result_summary_filename,
    result_summary_variant_name,
)

DIVERSITY_KEYS = (
    "mechanism_family",
    "intervention_surface",
    "intent",
    "semantic_family",
    "parent_lineage",
    "novelty_axis",
)

RATE_LIMIT_PATTERNS = ("rate limit", "429", "too many requests", "throttl")
GUARD_PATTERNS = ("guard", "denied", "refused", "blocked command", "tokenization")
RESOURCE_PATTERNS = (
    "cuda error",
    "out of memory",
    "oom",
    "connection reset",
    "network",
    "timeout",
    "timed out",
    "no space left",
    "disk quota",
)

RESULT_METADATA_KEYS = (
    "frontier_lane",
    "promotion_lane",
    "lane",
    "evidence_stage",
    "evidence_valence",
    "is_negative",
    "failure_mode",
    "diagnostic_role",
    "next_step_intent",
    "parent_candidate",
    "parent_usage",
    "protocol_name",
    "mature_enough",
    "maturity_basis",
    "effort_ratio",
    "coverage_ratio",
    "n_eval_units",
    "completed_required_eval_units",
    "complete_protocol_evaluation_units",
    "scored_complete",
)


def _utc(ts: float | None) -> str | None:
    if not ts:
        return None
    return datetime.fromtimestamp(ts, UTC).isoformat()


def _json_load(path: Path, *, max_bytes: int | None = None) -> Any:
    try:
        if max_bytes is None:
            return json.loads(path.read_text(encoding="utf-8"))
        with path.open("rb") as handle:
            payload = handle.read(max_bytes + 1)
        if len(payload) > max_bytes:
            return None
        return json.loads(payload.decode("utf-8"))
    except Exception:
        return None


def _jsonl_tail(path: Path, *, max_bytes: int = 256 * 1024, max_rows: int = 200) -> list[Any]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            offset = max(0, size - max_bytes)
            handle.seek(offset)
            text = handle.read().decode("utf-8", errors="ignore")
    except OSError:
        return []
    lines = text.splitlines()
    if offset and lines:
        lines = lines[1:]
    rows: list[Any] = []
    for line in lines[-max_rows:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _artifact_semantics_summary(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    semantics = data.get("artifact_semantics")
    if not isinstance(semantics, dict):
        return None
    keys = (
        "role",
        "status",
        "stage",
        "generation_id",
        "actor",
        "runtime_fact_source",
        "derived",
        "audit_only",
    )
    return {key: semantics.get(key) for key in keys if key in semantics}


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _extract_generation_id(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    match = re.search(r"(?<![A-Za-z0-9])gen[_-]?(\d+)(?=$|[^A-Za-z0-9])", str(value))
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _result_summary_generation_id(summary_path: Path, data: dict[str, Any]) -> int | None:
    aggregate = (
        data.get("current_aggregate") if isinstance(data.get("current_aggregate"), dict) else {}
    )
    for value in (
        data.get("generation_id"),
        data.get("source_generation_id"),
        data.get("gen_id"),
        aggregate.get("generation_id"),
        aggregate.get("source_generation_id"),
        data.get("peer_id"),
        aggregate.get("peer_id"),
        str(summary_path.parent),
    ):
        generation_id = _extract_generation_id(value)
        if generation_id is not None:
            return generation_id
    return None


def _late_after_generation_boundary(
    run_dir: Path, summary_path: Path, generation_id: int | None
) -> dict[str, Any] | None:
    if generation_id is None:
        return None
    boundary_path = run_dir / f"gen_{generation_id}" / "generation_boundary.json"
    if not boundary_path.exists():
        return None
    try:
        summary_mtime = summary_path.stat().st_mtime
        boundary_mtime = boundary_path.stat().st_mtime
    except OSError:
        return None
    if summary_mtime <= boundary_mtime:
        return None
    return {
        "generation_boundary_path": _safe_rel(boundary_path, run_dir),
        "generation_boundary_mtime": _utc(boundary_mtime),
        "source_result_mtime": _utc(summary_mtime),
    }


def _latest_run(task_path: Path) -> Path | None:
    experiments = task_path / "experiments"
    if not experiments.exists():
        return None
    candidates = [p for p in experiments.glob("run_*") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _hhi(values: list[str]) -> float | None:
    cleaned = [v for v in values if v]
    if not cleaned:
        return None
    counts = Counter(cleaned)
    total = sum(counts.values())
    return sum((count / total) ** 2 for count in counts.values())


def _walk_json_scalars(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, sub in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(_walk_json_scalars(sub, child))
    elif isinstance(value, list):
        for index, sub in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            out.update(_walk_json_scalars(sub, child))
    else:
        out[prefix] = value
    return out


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _nonnegative_count(*values: Any) -> int | float | None:
    for value in values:
        numeric = _finite_number(value)
        if numeric is None or numeric < 0:
            continue
        return int(numeric) if numeric.is_integer() else numeric
    return None


def _numeric_scalars(data: Any, *, max_items: int = 20) -> dict[str, float]:
    if not isinstance(data, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in _walk_json_scalars(data).items():
        numeric = _finite_number(value)
        if numeric is not None:
            out[str(key)] = numeric
        if len(out) >= max_items:
            break
    return out


def _mean_numeric_evaluation_unit_metrics(
    evaluation_units: list[Any], *, max_items: int = 20
) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for unit in evaluation_units:
        if not isinstance(unit, dict) or unit.get("validation_only"):
            continue
        for key, value in _walk_json_scalars(unit).items():
            if key == "validation_only" or key.endswith(".validation_only"):
                continue
            numeric = _finite_number(value)
            if numeric is not None:
                values[str(key)].append(numeric)
    means = {key: sum(nums) / len(nums) for key, nums in values.items() if nums}
    return dict(sorted(means.items())[:max_items])


def _metric_direction(metric_name: str | None, explicit_direction: Any = None) -> str:
    direction = str(explicit_direction or "").strip().lower()
    if direction in {"minimize", "min", "lower_is_better", "ascending"}:
        return "minimize"
    if direction in {"maximize", "max", "higher_is_better", "descending"}:
        return "maximize"
    name = (metric_name or "").lower()
    if any(token in name for token in ("loss", "error", "cost", "latency", "time", "risk")):
        return "minimize"
    return "maximize"


def _select_ranking_metric(
    data: dict[str, Any],
    aggregate_metrics: dict[str, float],
    evaluation_unit_metric_means: dict[str, float],
) -> tuple[str | None, float | None, str]:
    explicit_direction = (
        data.get("metric_direction")
        or data.get("primary_metric_direction")
        or data.get("direction")
    )
    preferred_names: list[str] = []
    for raw in (
        data.get("primary_metric"),
        data.get("metric_name"),
        "future_fitness",
        "taskscore",
        "score",
        "objective",
        "reward",
        "accuracy",
        "success_rate",
        "metric_value",
    ):
        if isinstance(raw, str) and raw and raw not in preferred_names:
            preferred_names.append(raw)
    candidates = {**evaluation_unit_metric_means, **aggregate_metrics}
    candidates.update(_numeric_scalars(data, max_items=50))
    for name in preferred_names:
        if name in candidates:
            return name, candidates[name], _metric_direction(name, explicit_direction)
        suffix_matches = [key for key in candidates if key.endswith(f".{name}")]
        if suffix_matches:
            key = suffix_matches[0]
            return key, candidates[key], _metric_direction(key, explicit_direction)
    if not candidates:
        return None, None, _metric_direction(None, explicit_direction)
    name = sorted(candidates)[0]
    return name, candidates[name], _metric_direction(name, explicit_direction)


def _ranking_sort_key(item: dict[str, Any]) -> tuple[bool, float]:
    value = item.get("ranking_metric_value")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return (True, math.inf)
    numeric = float(value)
    if item.get("ranking_metric_direction") == "minimize":
        return (False, numeric)
    return (False, -numeric)


def _extract_diversity_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    data = _json_load(path) if path.suffix == ".json" else None
    if data is not None:
        flat = _walk_json_scalars(data)
        for key in DIVERSITY_KEYS:
            preferred = (
                f"canonical_labels.canonical_{key}",
                f"canonical_{key}",
                f"design_dimensions.{key}",
                f"diversity_cell.{key}",
                key,
            )
            for candidate in preferred:
                if candidate in flat and isinstance(flat[candidate], (str, int, float)):
                    labels[key] = str(flat[candidate])
                    break
            if key not in labels:
                suffix_matches = [
                    v
                    for k, v in flat.items()
                    if k.endswith(f".{key}") and isinstance(v, (str, int, float))
                ]
                if suffix_matches:
                    labels[key] = str(suffix_matches[0])
        return labels

    text = path.read_text(encoding="utf-8", errors="ignore")
    for key in DIVERSITY_KEYS:
        match = re.search(
            rf"(?m)^\s*(?:canonical_)?{re.escape(key)}\s*:\s*['\"]?([^'\"\n#]+)", text
        )
        if match:
            labels[key] = match.group(1).strip()
    return labels


def generation_inventory(run_dir: Path) -> list[dict[str, Any]]:
    generations: list[dict[str, Any]] = []
    for gen_dir in sorted(run_dir.glob("gen_*"), key=lambda p: p.name):
        if not gen_dir.is_dir() or not re.fullmatch(r"gen_\d+", gen_dir.name):
            continue
        generation = int(gen_dir.name.split("_", 1)[1])
        peer_dirs = [p for p in gen_dir.glob("gen*_peer*") if p.is_dir()]
        peer_memory = list((gen_dir / "peers").glob("*/memory/peer_state.yaml"))
        labels_by_key: dict[str, list[str]] = {key: [] for key in DIVERSITY_KEYS}
        label_sources = 0
        for pattern in ("*_prompt_layout.json", "**/*contract*.json", "**/*contract*.yaml"):
            for path in gen_dir.glob(pattern):
                if not path.is_file():
                    continue
                labels = _extract_diversity_labels(path)
                if labels:
                    label_sources += 1
                for key, value in labels.items():
                    if key in labels_by_key:
                        labels_by_key[key].append(value)
        boundary_data = _json_load(gen_dir / "generation_boundary.json")
        hhi = {
            key: {
                "hhi": _hhi(values),
                "n_labeled": len(values),
                "unique": len(set(values)),
                "top": Counter(values).most_common(3),
            }
            for key, values in labels_by_key.items()
        }
        generations.append(
            {
                "generation": generation,
                "path": str(gen_dir),
                "stop_signal": (gen_dir / "STOP_SIGNAL").exists(),
                "generation_results": (gen_dir / "generation_results.json").exists(),
                "generation_boundary": (gen_dir / "generation_boundary.json").exists(),
                "generation_boundary_semantics": _artifact_semantics_summary(boundary_data),
                "dig_allocation": (gen_dir / "dig_cohort_allocation.yaml").exists(),
                "research_topology": (gen_dir / "research_topology.json").exists(),
                "peer_dirs": len(peer_dirs),
                "peer_memory_files": len(peer_memory),
                "latest_mtime": _utc(
                    max((p.stat().st_mtime for p in gen_dir.rglob("*") if p.is_file()), default=0)
                ),
                "diversity_label_sources": label_sources,
                "diversity_hhi": hhi,
            }
        )
    return generations


def _result_aggregate(data: dict[str, Any]) -> dict[str, Any]:
    for key in ("current_aggregate", "metrics", "aggregate"):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    tiers = data.get("tiers")
    if isinstance(tiers, list):
        for record in reversed(tiers):
            if not isinstance(record, dict):
                continue
            value = record.get("metrics_summary")
            if isinstance(value, dict):
                return value
    return {}


def _result_metadata(data: dict[str, Any], aggregate: dict[str, Any]) -> dict[str, Any]:
    sources = [data, aggregate]
    extra = data.get("extra")
    if isinstance(extra, dict):
        sources.append(extra)
    metadata: dict[str, Any] = {}
    for key in RESULT_METADATA_KEYS:
        for source in sources:
            value = source.get(key)
            if value not in (None, "", [], {}):
                metadata[key] = value
                break
    return metadata


def result_inventory(run_dir: Path) -> dict[str, Any]:
    results_dir = run_dir / "results"
    summaries = []
    if not results_dir.exists():
        return {"results_dir_exists": False, "summaries": [], "status_counts": {}}
    summary_paths = sorted(
        path
        for path in results_dir.rglob("*.json")
        if path.is_file() and is_supported_result_summary_filename(path.name)
    )
    for summary in summary_paths:
        data = _json_load(summary)
        if not isinstance(data, dict):
            continue
        aggregate = _result_aggregate(data)
        metadata = _result_metadata(data, aggregate)
        evaluation_units: list[Any] = []
        for source in (data, aggregate):
            for key in ("all_eval_units", "evaluation_units", "all_eval_cells"):
                candidate = source.get(key)
                if isinstance(candidate, list):
                    evaluation_units = candidate
                    break
            if evaluation_units:
                break
        aggregate_metrics = _numeric_scalars(aggregate)
        evaluation_unit_metric_means = _mean_numeric_evaluation_unit_metrics(evaluation_units)
        evaluation_unit_count = _nonnegative_count(
            data.get("n_eval_units"),
            aggregate.get("n_eval_units"),
            data.get("evaluation_units"),
            aggregate.get("evaluation_units"),
            data.get("completed_required_eval_units"),
            aggregate.get("completed_required_eval_units"),
            metadata.get("completed_required_eval_units"),
            # Compatibility reads for task-local and historical summaries.
            data.get("n_eval_cells"),
            aggregate.get("n_eval_cells"),
            aggregate.get("scored_cell_count"),
        )
        if evaluation_unit_count is None:
            evaluation_unit_count = len(evaluation_units)
        ranking_metric_name, ranking_metric_value, ranking_metric_direction = (
            _select_ranking_metric(
                data,
                aggregate_metrics,
                evaluation_unit_metric_means,
            )
        )
        generation_id = _result_summary_generation_id(summary, data)
        late_boundary = _late_after_generation_boundary(run_dir, summary, generation_id)
        summaries.append(
            {
                "path": _safe_rel(summary, run_dir),
                "variant_name": result_summary_variant_name(summary, data, run_dir),
                "generation_id": generation_id,
                "late_after_generation_boundary": bool(late_boundary),
                "late_boundary": late_boundary or {},
                "status": data.get("final_status")
                or data.get("tier_status")
                or data.get("result_status")
                or aggregate.get("result_status"),
                "evidence_stage": metadata.get("evidence_stage")
                or data.get("tier_reached")
                or data.get("completed_tier"),
                "n_eval_units": evaluation_unit_count,
                "ranking_metric_name": ranking_metric_name,
                "ranking_metric_value": ranking_metric_value,
                "ranking_metric_direction": ranking_metric_direction,
                "aggregate_metrics": aggregate_metrics,
                "evaluation_unit_metric_means": evaluation_unit_metric_means,
                "future_fitness": aggregate["future_fitness"]
                if "future_fitness" in aggregate
                else data.get("future_fitness"),
                "protocol_integrity_status": data.get("protocol_integrity_status"),
                "suspect_protocol": data.get("suspect_protocol"),
                "structured_metadata": metadata,
                "mtime": _utc(summary.stat().st_mtime),
            }
        )
    return {
        "results_dir_exists": True,
        "summary_count": len(summaries),
        "late_after_boundary_count": sum(
            1 for summary in summaries if summary.get("late_after_generation_boundary")
        ),
        "late_after_boundary_summaries": [
            {
                "path": summary["path"],
                "variant_name": summary["variant_name"],
                "generation_id": summary.get("generation_id"),
                "ranking_metric_name": summary.get("ranking_metric_name"),
                "ranking_metric_value": summary.get("ranking_metric_value"),
                "late_boundary": summary.get("late_boundary") or {},
            }
            for summary in summaries
            if summary.get("late_after_generation_boundary")
        ][:20],
        "status_counts": dict(Counter(str(s["status"]) for s in summaries)),
        "top_by_ranking_metric": sorted(
            summaries,
            key=_ranking_sort_key,
        )[:20],
    }


def shared_findings_inventory(run_dir: Path, limit: int) -> dict[str, Any]:
    findings_dir = run_dir / "shared_findings"
    if not findings_dir.exists():
        return {"exists": False, "count": 0, "recent": []}
    files = sorted(findings_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    recent = []
    late_after_boundary_count = 0
    validation_signal_count = 0
    for path in files[:limit]:
        data = _json_load(path)
        title = None
        variant = None
        metrics = {}
        if isinstance(data, dict):
            title = data.get("title") or data.get("finding_type")
            variant = data.get("variant_name")
            metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
        if metrics.get("late_after_generation_boundary"):
            late_after_boundary_count += 1
        if metrics.get("excluded_from_durable_frontier") or metrics.get("artifact_signal_status"):
            validation_signal_count += 1
        recent.append(
            {
                "path": _safe_rel(path, run_dir),
                "mtime": _utc(path.stat().st_mtime),
                "variant_name": variant,
                "title": title,
                "artifact_signal_status": metrics.get("artifact_signal_status"),
                "late_after_generation_boundary": bool(
                    metrics.get("late_after_generation_boundary")
                ),
                "exclusion_reason": metrics.get("exclusion_reason"),
            }
        )
    return {
        "exists": True,
        "count": len(files),
        "late_after_boundary_count_in_recent": late_after_boundary_count,
        "validation_signal_count_in_recent": validation_signal_count,
        "recent": recent,
    }


def frontier_inventory(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "frontier" / "frontier_manifest.json"
    manifest = _json_load(manifest_path)
    if not isinstance(manifest, dict):
        return {"exists": manifest_path.exists(), "lanes": {}}
    lanes = (
        manifest.get("lane_frontiers") if isinstance(manifest.get("lane_frontiers"), dict) else {}
    )
    out: dict[str, Any] = {}
    for lane, entries in lanes.items():
        lane_entries = []
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
            lane_entries.append(
                {
                    "variant_name": entry.get("variant_name"),
                    "metric_name": entry.get("metric_name"),
                    "metric_value": entry.get("metric_value"),
                    "metrics_preview": _numeric_scalars(metrics, max_items=8),
                    "evaluation_units": _nonnegative_count(
                        metrics.get("evaluation_units"),
                        metrics.get("n_eval_units"),
                        metrics.get("completed_required_eval_units"),
                        # Compatibility reads for historical task summaries.
                        metrics.get("n_cells"),
                        metrics.get("n_eval_cells"),
                    ),
                    "tier": metrics.get("tier") or entry.get("tier"),
                    "tier_status": metrics.get("tier_status") or metrics.get("final_status"),
                }
            )
        out[str(lane)] = {"count": len(lane_entries), "entries": lane_entries[:20]}
    return {
        "exists": True,
        "mtime": _utc(manifest_path.stat().st_mtime),
        "artifact_semantics": _artifact_semantics_summary(manifest),
        "primary_metric": manifest.get("primary_metric"),
        "metric_direction": manifest.get("metric_direction"),
        "lanes": out,
    }


def gems_inventory(run_dir: Path) -> dict[str, Any]:
    state_path = run_dir / "gems" / "gems_state.json"
    state = _json_load(state_path)
    if not isinstance(state, dict):
        return {"exists": state_path.exists()}
    return {
        "exists": True,
        "mtime": _utc(state_path.stat().st_mtime),
        "artifact_semantics": _artifact_semantics_summary(state),
        "enabled": state.get("enabled"),
        "cycle_index": state.get("cycle_index"),
        "cycle_start_generation": state.get("cycle_start_generation"),
        "reset_count": state.get("reset_count"),
        "pending_reset": state.get("pending_reset"),
        "gems_count": len(state.get("gems") or []),
        "gems": [
            {
                "variant_name": gem.get("variant_name"),
                "metric_name": gem.get("metric_name"),
                "metric_value": gem.get("metric_value"),
                "tier": gem.get("tier"),
            }
            for gem in (state.get("gems") or [])[:20]
            if isinstance(gem, dict)
        ],
    }


def _run_text(command: list[str], timeout: int = 10) -> str:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT, timeout=timeout
        )
    except Exception as exc:
        return f"unavailable: {exc}"


def hardware_inventory(run_dir: Path | None) -> dict[str, Any]:
    system = platform.system().lower()
    hardware: dict[str, Any] = {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "uptime": _run_text(["uptime"]).strip() if shutil.which("uptime") else "unavailable",
        "df_h": _run_text(["df", "-h", str(run_dir or Path.cwd())]).strip(),
    }
    if shutil.which("free"):
        hardware["memory"] = {"source": "free", "value": _run_text(["free", "-h"]).strip()}
    elif system == "darwin" and shutil.which("vm_stat"):
        hardware["memory"] = {
            "source": "vm_stat",
            "value": _run_text(["vm_stat"]).strip(),
        }
    else:
        try:
            total = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        except (AttributeError, OSError, TypeError, ValueError):
            total = 0
        hardware["memory"] = {
            "source": "sysconf" if total else "unavailable",
            "total_bytes": total or None,
        }

    accelerators: list[dict[str, Any]] = []
    if shutil.which("nvidia-smi"):
        observed = _run_text(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        if not observed.startswith("unavailable"):
            for row in csv.reader(observed.splitlines()):
                if len(row) >= 6:
                    accelerators.append(
                        {
                            "backend": "nvidia",
                            "index": row[0].strip(),
                            "name": row[1].strip(),
                            "utilization_pct": row[2].strip(),
                            "memory_used_mib": row[3].strip(),
                            "memory_total_mib": row[4].strip(),
                            "temperature_c": row[5].strip(),
                        }
                    )
    elif shutil.which("rocm-smi"):
        accelerators.append(
            {
                "backend": "rocm",
                "telemetry": _run_text(
                    ["rocm-smi", "--showproductname", "--showuse", "--showmemuse", "--json"]
                ).strip(),
            }
        )
    elif system == "darwin" and shutil.which("system_profiler"):
        accelerators.append(
            {
                "backend": "darwin_display",
                "telemetry": _run_text(
                    ["system_profiler", "SPDisplaysDataType", "-json"], timeout=15
                ).strip(),
                "note": "Unified-memory accelerator telemetry may be unavailable.",
            }
        )
    hardware["accelerators"] = accelerators
    hardware["accelerator_probe_status"] = "observed" if accelerators else "not_observed"
    return hardware


def process_inventory(run_dir: Path) -> dict[str, Any]:
    out = _run_text(["ps", "-eo", "pid,ppid,etime,stat,pcpu,pmem,args"], timeout=10)
    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "processes": 0,
            "cpu_pct_sum": 0.0,
            "mem_pct_sum": 0.0,
            "pids": [],
            "max_etime": "",
        }
    )
    for line in out.splitlines()[1:]:
        if str(run_dir) not in line:
            continue
        match = re.search(r"--variant-name\s+(\S+)", line)
        name = match.group(1) if match else "main_or_other"
        parts = line.split(None, 6)
        if len(parts) < 7:
            continue
        pid, _ppid, etime, _stat, pcpu, pmem, _args = parts
        group = groups[name]
        group["processes"] += 1
        group["pids"].append(pid)
        group["max_etime"] = max(group["max_etime"], etime)
        try:
            group["cpu_pct_sum"] += float(pcpu)
            group["mem_pct_sum"] += float(pmem)
        except ValueError:
            pass
    return {"groups": dict(groups)}


def log_signal_inventory(run_dir: Path, max_bytes: int) -> dict[str, Any]:
    paths = []
    for pattern in (
        "*.log",
        "logs/*.log",
        "logs/*.jsonl",
        "trajectory.jsonl",
        "budget_ledger.jsonl",
    ):
        paths.extend(run_dir.glob(pattern))
    counts = {"guard": 0, "rate_limit": 0, "resource": 0}
    examples: dict[str, list[str]] = {"guard": [], "rate_limit": [], "resource": []}
    for path in paths:
        if not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - max_bytes))
                text = handle.read().decode("utf-8", errors="ignore").lower()
        except Exception:
            continue
        for line in text.splitlines():
            checks = (
                ("guard", GUARD_PATTERNS),
                ("rate_limit", RATE_LIMIT_PATTERNS),
                ("resource", RESOURCE_PATTERNS),
            )
            for bucket, patterns in checks:
                if any(pattern in line for pattern in patterns):
                    counts[bucket] += 1
                    if len(examples[bucket]) < 5:
                        examples[bucket].append(f"{_safe_rel(path, run_dir)}: {line[:240]}")
    return {"counts": counts, "examples": examples}


def _usage_view(usage: dict[str, Any], *, sessions: int) -> dict[str, Any]:
    def number(key: str) -> float | None:
        if key not in usage:
            return None
        try:
            value = float(usage[key])
        except (TypeError, ValueError):
            return None
        return max(0.0, value) if math.isfinite(value) else None

    input_tokens = number("input_tokens")
    total_input_tokens = number("total_input_tokens")
    if total_input_tokens is None:
        total_input_tokens = input_tokens
    cached_input_tokens = number("cached_input_tokens")
    cache_read_input_tokens = number("cache_read_input_tokens")
    if cached_input_tokens is None:
        cached_input_tokens = cache_read_input_tokens
    cache_creation_input_tokens = number("cache_creation_input_tokens")
    uncached_input_tokens = number("uncached_input_tokens")

    telemetry_issues: list[str] = []
    if (
        input_tokens is not None
        and total_input_tokens is not None
        and not math.isclose(input_tokens, total_input_tokens)
    ):
        telemetry_issues.append("input_total_mismatch")
    if (
        cached_input_tokens is not None
        and cache_read_input_tokens is not None
        and not math.isclose(cached_input_tokens, cache_read_input_tokens)
    ):
        telemetry_issues.append("cache_read_alias_mismatch")
    if (
        total_input_tokens is not None
        and cached_input_tokens is not None
        and cached_input_tokens > total_input_tokens
    ):
        telemetry_issues.append("cached_input_exceeds_total_input")
    if uncached_input_tokens is None:
        if total_input_tokens is not None and cached_input_tokens is not None:
            known_cache_input = cached_input_tokens + (cache_creation_input_tokens or 0.0)
            if known_cache_input <= total_input_tokens:
                uncached_input_tokens = total_input_tokens - known_cache_input
            elif "cached_input_exceeds_total_input" not in telemetry_issues:
                telemetry_issues.append("cache_components_exceed_total_input")
    elif total_input_tokens is not None:
        component_total = uncached_input_tokens + (cached_input_tokens or 0.0)
        component_total += cache_creation_input_tokens or 0.0
        if not math.isclose(component_total, total_input_tokens):
            telemetry_issues.append("input_components_do_not_match_total")

    telemetry_inconsistent = bool(telemetry_issues)
    return {
        "sessions": sessions,
        "input_tokens": input_tokens,
        "total_input_tokens": total_input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "output_tokens": number("output_tokens"),
        "reasoning_output_tokens": number("reasoning_output_tokens"),
        "total_tokens": number("total_tokens"),
        "telemetry_inconsistent": telemetry_inconsistent,
        "telemetry_issues": telemetry_issues,
        "cache_hit_ratio": (
            cached_input_tokens / total_input_tokens
            if not telemetry_inconsistent
            and total_input_tokens is not None
            and cached_input_tokens is not None
            and total_input_tokens > 0
            else None
        ),
        "average_input_tokens_per_session": (
            total_input_tokens / sessions
            if total_input_tokens is not None and sessions > 0
            else None
        ),
    }


def runtime_usage_inventory(run_dir: Path, run_summary: Any) -> dict[str, Any]:
    """Build a read-only session/token view from canonical generation results."""

    generations: list[dict[str, Any]] = []
    aggregate: dict[str, float] = defaultdict(float)
    aggregate_sessions = 0
    aggregate_peers = 0
    for result_path in sorted(run_dir.glob("gen_*/generation_results.json")):
        rows = _json_load(result_path, max_bytes=8 * 1024 * 1024)
        if not isinstance(rows, list):
            continue
        generation_usage: dict[str, float] = defaultdict(float)
        sessions = 0
        peers = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            peers += 1
            try:
                sessions += max(0, int(row.get("sessions", 0) or 0))
            except (TypeError, ValueError):
                pass
            usage = row.get("runtime_usage")
            if not isinstance(usage, dict):
                continue
            for key, raw in usage.items():
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and value >= 0:
                    generation_usage[str(key)] += value
        aggregate_sessions += sessions
        aggregate_peers += peers
        for key, value in generation_usage.items():
            aggregate[key] += value
        view = _usage_view(generation_usage, sessions=sessions)
        view.update(
            {
                "generation": _extract_generation_id(result_path.parent.name),
                "peers": peers,
                "sessions_per_peer": sessions / peers if peers > 0 else None,
            }
        )
        generations.append(view)

    summary_usage: dict[str, Any] | None = None
    runtime_ref = None
    provider_ref = None
    if isinstance(run_summary, dict):
        runtime_ref = run_summary.get("runtime_ref")
        provider_ref = run_summary.get("model_provider_ref")
        legacy = run_summary.get("legacy_generation_loop_summary")
        if isinstance(legacy, dict) and isinstance(legacy.get("runtime_usage"), dict):
            summary_usage = legacy["runtime_usage"]
    effective_usage = summary_usage or dict(aggregate)
    total = _usage_view(effective_usage, sessions=aggregate_sessions)
    total.update(
        {
            "peers": aggregate_peers,
            "peer_generations": aggregate_peers,
            "sessions_per_peer_generation": (
                aggregate_sessions / aggregate_peers if aggregate_peers > 0 else None
            ),
            "source": (
                "run_summary.runtime_usage"
                if summary_usage is not None
                else "generation_results.runtime_usage"
            ),
        }
    )
    return {
        "runtime_ref": runtime_ref,
        "model_provider_ref": provider_ref,
        "total": total,
        "by_generation": generations,
    }


def build_inventory(args: argparse.Namespace) -> dict[str, Any]:
    task_path = Path(args.task_path or Path.cwd()).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else _latest_run(task_path)
    if run_dir is None:
        raise SystemExit(f"no run directory found under {task_path / 'experiments'}")

    orchestrator = _json_load(run_dir / "orchestrator_status.json")
    run_summary = _json_load(run_dir / "run_summary.json")
    generations = generation_inventory(run_dir)
    return {
        "task_path": str(task_path),
        "run_dir": str(run_dir),
        "generated_at": datetime.now(UTC).isoformat(),
        "orchestrator_status": orchestrator if isinstance(orchestrator, dict) else None,
        "run_summary": run_summary if isinstance(run_summary, dict) else None,
        "runtime_usage": runtime_usage_inventory(run_dir, run_summary),
        "generations": generations,
        "results": result_inventory(run_dir),
        "shared_findings": shared_findings_inventory(run_dir, args.max_findings),
        "frontier": frontier_inventory(run_dir),
        "gems": gems_inventory(run_dir),
        "resource_scheduler": _json_load(
            run_dir / "resource_scheduler" / "status.json",
            max_bytes=2 * 1024 * 1024,
        ),
        "resource_scheduler_events": _jsonl_tail(run_dir / "resource_scheduler" / "events.jsonl"),
        "log_signals": log_signal_inventory(run_dir, args.log_tail_bytes),
        "processes": process_inventory(run_dir),
        "hardware": hardware_inventory(run_dir) if args.hardware else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-path", help="Praxist task directory; defaults to cwd")
    parser.add_argument(
        "--run-dir", help="Run directory; defaults to latest task experiments/run_*"
    )
    parser.add_argument("--max-findings", type=int, default=20)
    parser.add_argument("--log-tail-bytes", type=int, default=1_000_000)
    parser.add_argument("--hardware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json", action="store_true", help="Emit JSON; default is JSON as well")
    args = parser.parse_args()
    inventory = build_inventory(args)
    print(json.dumps(inventory, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
