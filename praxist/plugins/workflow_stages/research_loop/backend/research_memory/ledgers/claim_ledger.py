"""Claim ledger — tracks scientific claims across generations.

Each claim has: status (active/weakened/killed/retired), confidence,
boundary, supports, challenges, missing_tests, and (for retired only)
revive_if conditions.

Critical anti-pattern this ledger prevents: a PI retiring a claim without
preserving the boundary that made the retirement valid. Replay through the
retired_claim_ledger pathway forces boundary + revive_if to be written, and
downstream summarizers cannot drop them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers._ledger_base import (
    LedgerEntry,
    LedgerStore,
)

VALID_STATUSES = ("active", "weakened", "killed", "retired", "conditionally_retired")


class ClaimLedger:
    """YAML ledger wrapper for active research claims."""

    def __init__(self, run_dir: Path):
        ledger_path = Path(run_dir) / "research_memory" / "ledgers" / "claim_ledger.yaml"
        self.store = LedgerStore(ledger_path, "claim_ledger")

    def upsert_claim(
        self,
        claim_id: str,
        title: str,
        status: str,
        confidence: float,
        boundary: str = "",
        supports: list[str] | None = None,
        challenges: list[str] | None = None,
        missing_tests: list[str] | None = None,
        revive_if: list[str] | None = None,
        scope: dict[str, Any] | None = None,
        created_by: str = "unknown",
        action: str = "upsert",
    ) -> LedgerEntry:
        if status not in VALID_STATUSES:
            raise ValueError(f"unknown claim status: {status} (must be {VALID_STATUSES})")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {confidence}")
        # Retired/killed claims SHOULD have boundary + revive_if; warn if missing.
        # (Validators downstream enforce; we just record what was passed.)
        data = {
            "title": title,
            "status": status,
            "confidence": confidence,
            "boundary": boundary,
            "supports": list(supports or []),
            "challenges": list(challenges or []),
            "missing_tests": list(missing_tests or []),
            "revive_if": list(revive_if or []),
            "scope": dict(scope or {}),
        }
        return self.store.upsert(claim_id, data, created_by, action)

    def list_active(self) -> list[LedgerEntry]:
        return self.store.filter(lambda e: e.data.get("status") == "active")

    def list_recently_killed(self, n: int = 5) -> list[LedgerEntry]:
        killed = self.store.filter(
            lambda e: e.data.get("status") in ("killed", "retired", "conditionally_retired")
        )
        killed.sort(key=lambda e: e.last_updated_at, reverse=True)
        return killed[:n]

    def get(self, claim_id: str) -> LedgerEntry | None:
        return self.store.get(claim_id)

    def all(self) -> list[LedgerEntry]:
        return self.store.list_entries()
