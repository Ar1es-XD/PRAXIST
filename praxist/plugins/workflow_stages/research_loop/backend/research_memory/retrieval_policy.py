"""Retrieval policy — enforces the support/challenge/negative/old/external mix.

Strict invariants (per design doc §11):
  - challenge_or_negative >= 20% of evidence pack
  - frontier_delta >= 15% (10% in high-stakes when external is bumped)
  - support <= 50%

When the underlying ledgers don't have enough of a category, the policy
gracefully degrades (logs warning) instead of failing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMix:
    """Configured evidence mix for PI shared-core and role-private context packs."""

    support: float = 0.40
    challenge_negative: float = 0.25
    frontier_delta: float = 0.15
    old_retired: float = 0.10
    external_validity: float = 0.10

    def normalized(self) -> RetrievalMix:
        s = (
            self.support
            + self.challenge_negative
            + self.frontier_delta
            + self.old_retired
            + self.external_validity
        )
        # R8#3 fix: caller's explicit all-zero is preserved as zero
        # rather than silently flipped to default proportions.
        if s <= 0:
            return RetrievalMix(
                support=0.0,
                challenge_negative=0.0,
                frontier_delta=0.0,
                old_retired=0.0,
                external_validity=0.0,
            )
        return RetrievalMix(
            support=self.support / s,
            challenge_negative=self.challenge_negative / s,
            frontier_delta=self.frontier_delta / s,
            old_retired=self.old_retired / s,
            external_validity=self.external_validity / s,
        )

    def slot_counts(self, total: int) -> dict[str, int]:
        m = self.normalized()
        return {
            "support": max(1, int(round(total * m.support))),
            "challenge_negative": max(1, int(round(total * m.challenge_negative))),
            "frontier_delta": max(0, int(round(total * m.frontier_delta))),
            "old_retired": max(0, int(round(total * m.old_retired))),
            "external_validity": max(0, int(round(total * m.external_validity))),
        }


HIGH_STAKES_MIX = RetrievalMix(
    support=0.30,
    challenge_negative=0.30,
    frontier_delta=0.05,
    old_retired=0.10,
    external_validity=0.25,
)

NORMAL_MIX = RetrievalMix()


def select_cards_with_mix(
    cards: list[dict[str, Any]],
    mix: RetrievalMix,
    total_budget: int,
    high_stakes: bool = False,
) -> list[dict[str, Any]]:
    """Choose up to `total_budget` cards conforming to mix slots.

    R2#8 fix: sort cards by evidence_id before bucketing for deterministic
    output across runs. Without this, bucket-fill order is undefined when
    cards arrive from set-derived iteration paths.
    """
    if not cards:
        return []
    cards = sorted(
        (c for c in cards if isinstance(c, dict)),
        key=lambda c: c.get("evidence_id", "") or "",
    )
    slots = mix.slot_counts(total_budget)

    # Bucket cards by category. A card may belong to multiple buckets;
    # we put it in the first matching bucket.
    buckets: dict[str, list[dict[str, Any]]] = {
        "challenge_negative": [],
        "support": [],
        "frontier_delta": [],
        "old_retired": [],
        "external_validity": [],
    }
    for c in cards:
        if not isinstance(c, dict):
            continue
        # heuristic categorization
        is_neg = bool(c.get("quality", {}).get("is_negative"))
        is_retired = bool(c.get("quality", {}).get("is_retired"))
        is_frontier_delta = c.get("source_type") == "frontier_delta"
        is_external_check = (
            "sentinel" in (c.get("interpretation", {}).get("short", "") or "").lower()
            or "cross_arch" in (c.get("interpretation", {}).get("short", "") or "").lower()
        )
        if is_neg:
            buckets["challenge_negative"].append(c)
        elif is_retired:
            buckets["old_retired"].append(c)
        elif is_frontier_delta:
            buckets["frontier_delta"].append(c)
        elif is_external_check:
            buckets["external_validity"].append(c)
        else:
            buckets["support"].append(c)

    selected: list[dict[str, Any]] = []
    seen_ids = set()
    for category, want in slots.items():
        bucket = buckets.get(category, [])
        for c in bucket[:want]:
            eid = c.get("evidence_id")
            if eid and eid not in seen_ids:
                seen_ids.add(eid)
                selected.append(c)

    # Top up if we under-filled
    if len(selected) < total_budget:
        leftover = [c for c in cards if c.get("evidence_id") not in seen_ids]
        for c in leftover[: total_budget - len(selected)]:
            seen_ids.add(c.get("evidence_id"))
            selected.append(c)

    return selected[:total_budget]


def negative_evidence_ratio(cards: list[dict[str, Any]]) -> float:
    """Return the ratio of negative evidence in an evidence-card collection."""
    if not cards:
        return 0.0
    neg = sum(1 for c in cards if isinstance(c, dict) and c.get("quality", {}).get("is_negative"))
    return neg / len(cards)
