"""Deterministic fake tiered budget policy plugin."""

from __future__ import annotations

from praxist.core.budget import _invalid_budget_units
from praxist.core.protocol import BudgetDecision, BudgetGrant, BudgetRequest


class FakeTieredBudgetPolicy:
    """Deterministic policy that covers Gate B budget conformance cases."""

    strong_evidence_caps = {
        "tokens": 5_000_000.0,
        "wall_clock_seconds": 172_800.0,
        "gpu_hours": 2.0,
    }

    def decide(self, request: BudgetRequest) -> BudgetDecision:
        invalid_units = _invalid_budget_units(request.requested)
        if invalid_units:
            return BudgetDecision(
                decision="deny",
                reason_codes=["invalid_budget_value", *invalid_units],
                grant=None,
                review_target=None,
            )
        requested = request.requested
        tokens = requested.get("tokens", 0.0)
        wall_clock = requested.get("wall_clock_seconds", 0.0)
        gpu_hours = requested.get("gpu_hours", 0.0)
        confidence = str(request.expected_value.get("confidence", "weak"))

        if request.expected_value.get("impossible"):
            return BudgetDecision(
                decision="deny",
                reason_codes=["impossible_request"],
                grant=None,
                review_target=None,
            )

        if tokens <= 5_000 and wall_clock <= 120 and gpu_hours <= 0:
            return BudgetDecision(
                decision="grant",
                reason_codes=["cheap_probe_auto_grant"],
                grant=BudgetGrant(
                    grant_id=f"grant_{request.request_id}",
                    approved=dict(requested),
                    conditions=[],
                    expires_at_generation=None,
                ),
            )

        if confidence == "strong" and request.evidence_refs:
            approved = dict(requested)
            for unit, cap in self.strong_evidence_caps.items():
                if float(approved.get(unit, 0.0)) > cap:
                    approved[unit] = cap
            if approved != requested and request.expected_value.get("requires_full_stage_budget"):
                return BudgetDecision(
                    decision="require_review",
                    reason_codes=["required_stage_budget_exceeds_policy_caps"],
                    grant=None,
                    review_target="chair",
                )
            return BudgetDecision(
                decision="downscope" if approved != requested else "grant",
                reason_codes=["strong_evidence_conditional_grant"],
                grant=BudgetGrant(
                    grant_id=f"grant_{request.request_id}",
                    approved=approved,
                    conditions=["record_actual_usage", "abort_on_no_signal"],
                    expires_at_generation=None,
                ),
            )

        return BudgetDecision(
            decision="require_review",
            reason_codes=["weak_evidence_over_budget"],
            grant=None,
            review_target="chair",
        )


def create_policy() -> FakeTieredBudgetPolicy:
    return FakeTieredBudgetPolicy()
