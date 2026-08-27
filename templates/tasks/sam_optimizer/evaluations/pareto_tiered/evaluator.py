"""SAM tiered Pareto evaluation contract for the task project."""

from __future__ import annotations

import math
from typing import Any

PRIMARY_METRICS = (
    "mean_test_accuracy",
    "test_accuracy_cifar100",
    "test_accuracy_cifar10",
    "test_accuracy_tiny_imagenet",
)
EFFICIENCY_METRICS = ("compute_overhead_ratio", "wall_time_seconds_total", "memory_mb")
REQUIRED_TIERS = ("T1", "T2", "T3")


class SamParetoTieredEvaluation:
    """Task-local promotion/ranking contract for tiered Pareto metrics."""

    evaluation_ref = "evaluation:sam_pareto_tiered"

    def eligible_for_promotion(self, finding: dict[str, Any]) -> bool:
        metrics = finding.get("metrics") if isinstance(finding.get("metrics"), dict) else {}
        tier = str(
            finding.get("tier")
            or finding.get("eval_tier")
            or metrics.get("tier")
            or metrics.get("eval_tier")
            or ""
        )
        if tier not in REQUIRED_TIERS:
            return False
        if tier != "T3":
            return False
        if not _truthy(finding.get("promotion_eligible", metrics.get("promotion_eligible"))):
            return False
        return any(_finite(metrics.get(metric)) for metric in PRIMARY_METRICS)

    def rank(self, findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        eligible = [finding for finding in findings if self.eligible_for_promotion(finding)]
        return sorted(eligible, key=_pareto_sort_key, reverse=True)


def create_evaluation() -> SamParetoTieredEvaluation:
    """Factory entrypoint that constructs the task-local evaluation contract."""

    return SamParetoTieredEvaluation()


def _pareto_sort_key(finding: dict[str, Any]) -> tuple[float, float, str]:
    metrics = finding.get("metrics") or {}
    quality = max((_finite(metrics.get(metric)) for metric in PRIMARY_METRICS), default=0.0)
    efficiency_penalty = sum(_finite(metrics.get(metric)) for metric in EFFICIENCY_METRICS)
    title = str(finding.get("title") or finding.get("id") or "")
    return (quality, -efficiency_penalty, title)


def _finite(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if math.isfinite(numeric) else 0.0


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)):
        return value == 1
    return False
