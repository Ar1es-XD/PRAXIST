"""Mechanism ledger — captures the effect / interaction / boundary of each
mechanism (for example scheduling, scaling, or decomposition).

Distinct from claim_ledger: mechanisms are durable building blocks; claims
are statements about (combinations of) mechanisms.
"""

from __future__ import annotations

from pathlib import Path

from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers._ledger_base import (
    LedgerEntry,
    LedgerStore,
)


class MechanismLedger:
    """YAML ledger wrapper for candidate mechanisms and explanatory claims."""

    def __init__(self, run_dir: Path):
        path = Path(run_dir) / "research_memory" / "ledgers" / "mechanism_ledger.yaml"
        self.store = LedgerStore(path, "mechanism_ledger")

    def upsert(
        self,
        mech_id: str,
        name: str,
        family: str = "",
        effect_summary: str = "",
        interactions: list[dict[str, str]] | None = None,
        boundary: str = "",
        observed_in_findings: list[str] | None = None,
        created_by: str = "unknown",
        action: str = "upsert",
    ) -> LedgerEntry:
        data = {
            "name": name,
            "family": family,
            "effect_summary": effect_summary,
            "interactions": list(interactions or []),
            "boundary": boundary,
            "observed_in_findings": list(observed_in_findings or []),
        }
        return self.store.upsert(mech_id, data, created_by, action)

    def get(self, mech_id: str) -> LedgerEntry | None:
        return self.store.get(mech_id)

    def all(self) -> list[LedgerEntry]:
        return self.store.list_entries()
