"""Soft exploration-bottleneck detector for Gems-aware research loops.

The detector is intentionally advisory. It summarizes recent surface narrowing
and repeated failed families into soft agenda priors for PI/Chair prompts. It
does not assign peers, delete candidates, or block ordinary frontier promotion.
Core Praxist keeps this detector task-neutral; domain-specific detectors should
live in task projects or be expressed through task-owned metadata.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _lower_text(*parts: Any) -> str:
    return " ".join(str(p or "") for p in parts).lower()


def _metric(obj: dict[str, Any], name: str) -> Any:
    metrics = obj.get("metrics") if isinstance(obj.get("metrics"), dict) else {}
    if name in obj:
        return obj.get(name)
    return metrics.get(name)


def _generation_id(obj: dict[str, Any]) -> int | None:
    for key in ("generation_id", "gen", "source_generation_id"):
        value = obj.get(key)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    peer = str(obj.get("peer_id") or "")
    if peer.startswith("gen") and "_peer" in peer:
        try:
            return int(peer.split("_peer", 1)[0][3:])
        except (TypeError, ValueError):
            return None
    return None


def _read_json_files(paths: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            out.append(data)
    return out


def _frontier_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    lanes = manifest.get("lane_frontiers")
    if isinstance(lanes, dict):
        for lane, entries in lanes.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict):
                    item = dict(entry)
                    item.setdefault("frontier_lane", lane)
                    out.append(item)
    cumulative = manifest.get("cumulative_top")
    if isinstance(cumulative, list):
        for entry in cumulative:
            if isinstance(entry, dict):
                out.append(dict(entry))
    return out


def _entropy(labels: list[str]) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = float(sum(counts.values()))
    return -sum((n / total) * math.log(n / total) for n in counts.values())


class ExplorationBottleneckDetector:
    """Detect soft research-process bottlenecks over recent findings/frontier."""

    def __init__(
        self,
        *,
        run_dir: Path,
        lookback_generations: int = 3,
        mode: str = "generic",
        performance_lanes: set[str] | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.lookback_generations = max(1, int(lookback_generations))
        self.mode = (mode or "generic").strip().lower()
        self.performance_lanes = {
            str(lane).strip()
            for lane in (
                performance_lanes or {"confirmed", "performance", "candidate", "task_candidate"}
            )
            if str(lane).strip()
        }

    def _family_for_mode(self, obj: dict[str, Any]) -> str:
        explicit = _metric(obj, "mechanism_family") or _metric(obj, "strategy_family")
        if explicit:
            text = str(explicit).strip().lower()
            if text and text not in {"unknown", "none", "null", "n/a"}:
                return text
        text = _lower_text(
            obj.get("variant_name"),
            obj.get("title"),
            obj.get("content"),
            obj.get("summary"),
            _metric(obj, "innovation_surface"),
        )
        if "benchmark" in text or "floor" in text or "baseline" in text:
            return "reference_or_floor"
        if "control" in text or "ablation" in text or "falsif" in text:
            return "diagnostic_or_control"
        if "risk" in text or "robust" in text or "stability" in text:
            return "robustness"
        return "other"

    def analyze(self, *, completed_gen_id: int, manifest: dict[str, Any]) -> dict[str, Any]:
        lookback_start = max(0, int(completed_gen_id) - self.lookback_generations + 1)
        recent_gens = set(range(lookback_start, int(completed_gen_id) + 1))
        findings = [
            f
            for f in _read_json_files(sorted((self.run_dir / "shared_findings").glob("*.json")))
            if _generation_id(f) in recent_gens
        ]
        frontier = _frontier_entries(manifest)
        recent_frontier = [
            f for f in frontier if (_generation_id(f) is None or _generation_id(f) in recent_gens)
        ]
        evidence_pool = [*findings, *recent_frontier]

        families = [self._family_for_mode(item) for item in evidence_pool]
        family_counts = Counter(families)
        top_family_share = (
            max(family_counts.values()) / sum(family_counts.values()) if family_counts else 0.0
        )
        entropy = _entropy(families)

        lane_frontiers = manifest.get("lane_frontiers", {})
        lane_frontiers = lane_frontiers if isinstance(lane_frontiers, dict) else {}
        confirmed: list[dict[str, Any]] = []
        for lane_name in self.performance_lanes:
            entries = lane_frontiers.get(lane_name, [])
            if isinstance(entries, list):
                confirmed.extend(entry for entry in entries if isinstance(entry, dict))
        recent_confirmed = [
            item
            for item in confirmed
            if isinstance(item, dict) and _generation_id(item) in recent_gens
        ]

        def is_failed_item(item: dict[str, Any]) -> bool:
            if (_as_float(_metric(item, "n_hard_constraint_violations")) or 0.0) > 0.0:
                return True
            text_failed = "fail" in _lower_text(
                item.get("title"),
                item.get("content"),
                item.get("notes"),
                _metric(item, "status"),
                _metric(item, "result_status"),
            )
            if text_failed:
                return True
            for key in ("score_delta", "task_delta", "metric_delta", "primary_metric_delta"):
                value = _as_float(_metric(item, key))
                if value is not None and value < 0.0:
                    return True
            return False

        failed_families = [
            self._family_for_mode(item) for item in evidence_pool if is_failed_item(item)
        ]
        repeated_failed_family_count = sum(1 for n in Counter(failed_families).values() if n >= 2)

        metrics = {
            "lookback_generations": sorted(recent_gens),
            "finding_count": len(findings),
            "frontier_entry_count": len(recent_frontier),
            "evidence_item_count": len(evidence_pool),
            "detector_mode": self.mode,
            "confirmed_candidate_count": int(len(recent_confirmed)),
            "total_confirmed_candidate_count": int(len(confirmed)),
            "surface_entropy": round(entropy, 4),
            "surface_entropy_low": bool(entropy < 1.20 and len(family_counts) >= 2),
            "top_mechanism_family": family_counts.most_common(1)[0][0] if family_counts else "",
            "top_mechanism_family_share": round(top_family_share, 4),
            "repeated_failed_family_count": int(repeated_failed_family_count),
        }
        records = self._trigger_records(metrics)
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "completed_generation": int(completed_gen_id),
            "detector": "exploration_bottleneck_detector",
            "metrics": metrics,
            "records": records,
            "soft_agenda_priors": self._combine_soft_priors(records),
        }

    def _trigger_records(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        confirmed_count = int(metrics.get("confirmed_candidate_count", 0))
        narrowing_votes = sum(
            [
                float(metrics["top_mechanism_family_share"]) > 0.55,
                bool(metrics["surface_entropy_low"]),
                int(metrics["repeated_failed_family_count"]) > 2,
            ]
        )
        if confirmed_count == 0 and narrowing_votes >= 2:
            records.append(
                {
                    "gem_type": "surface_narrowing",
                    "severity": "medium",
                    "evidence": {
                        "top_mechanism_family": metrics["top_mechanism_family"],
                        "top_mechanism_family_share": metrics["top_mechanism_family_share"],
                        "surface_entropy_low_for_2_generations": metrics["surface_entropy_low"],
                        "repeated_failed_family_count": metrics["repeated_failed_family_count"],
                    },
                    "soft_agenda_priors": {
                        "increase_underused_surface_probability": 0.20,
                        "avoid_single_surface_overallocation": True,
                        "increase_anti_mainline_probability": 0.10,
                    },
                    "hard_constraints": [],
                }
            )
        return records

    @staticmethod
    def _combine_soft_priors(records: list[dict[str, Any]]) -> dict[str, Any]:
        combined: dict[str, Any] = {}
        for record in records:
            priors = record.get("soft_agenda_priors")
            if not isinstance(priors, dict):
                continue
            for key, value in priors.items():
                if isinstance(value, bool):
                    combined[key] = bool(combined.get(key, False) or value)
                elif isinstance(value, (int, float)):
                    combined[key] = round(float(combined.get(key, 0.0)) + float(value), 4)
                else:
                    combined[key] = value
        return combined
