"""Placeholder task-template evaluation contract."""

from __future__ import annotations

import math
from typing import Any

PRIMARY_METRICS = ("deterministic_score",)
EFFICIENCY_METRICS = ("cost",)
REQUIRED_TIERS: tuple[str, ...] = ()


class TemplateParetoTieredEvaluation:
    """Template task-local evaluation contract for task authors."""

    evaluation_ref = "evaluation:template_pareto_tiered"

    def eligible_for_promotion(self, finding: dict[str, Any]) -> bool:
        tier = str(finding.get("tier") or finding.get("eval_tier") or "")
        if REQUIRED_TIERS and tier and tier not in REQUIRED_TIERS:
            return False
        metrics = finding.get("metrics") or {}
        return any(_finite(metrics.get(metric)) for metric in PRIMARY_METRICS)

    def rank(self, findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        eligible = [finding for finding in findings if self.eligible_for_promotion(finding)]
        return sorted(eligible, key=_pareto_sort_key, reverse=True)


def create_evaluation() -> TemplateParetoTieredEvaluation:
    """Factory entrypoint that constructs a task-local evaluation contract."""
    return TemplateParetoTieredEvaluation()


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
