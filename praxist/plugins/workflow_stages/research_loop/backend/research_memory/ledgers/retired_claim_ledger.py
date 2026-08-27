"""Retired claim ledger — once-active claims that have been retired.

Critical invariant: every retired entry MUST have:
  - scope (architectures / datasets / protocol)
  - boundary (e.g. "current_protocol_only")
  - revive_if (conditions under which to re-investigate)

This is the structural fix for the "retired without a boundary" anti-pattern.
context_auditor refuses to publish an agenda whose retired_claim entry lacks
these fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers._ledger_base import (
    LedgerEntry,
    LedgerStore,
)


class RetiredClaimLedger:
    """YAML ledger wrapper for claims removed from the active agenda."""

    def __init__(self, run_dir: Path):
        path = Path(run_dir) / "research_memory" / "ledgers" / "retired_claim_ledger.yaml"
        self.store = LedgerStore(path, "retired_claim_ledger")

    def retire(
        self,
        claim_id: str,
        title: str,
        reason: str,
        boundary: str,
        revive_if: list[str],
        scope: dict[str, Any] | None = None,
        source_evidence: list[str] | None = None,
        created_by: str = "unknown",
    ) -> LedgerEntry:
        if not boundary:
            raise ValueError(
                f"retire(claim_id={claim_id}): boundary cannot be empty. "
                f"Retired claims must specify scope, e.g. 'current_protocol_only'."
            )
        if not revive_if or not isinstance(revive_if, list):
            raise ValueError(
                f"retire(claim_id={claim_id}): revive_if must be a non-empty list. "
                f"What conditions would justify re-investigating this claim?"
            )
        data = {
            "title": title,
            "status": "conditionally_retired",
            "reason": reason,
            "boundary": boundary,
            "scope": dict(scope or {}),
            "revive_if": list(revive_if),
            "source_evidence": list(source_evidence or []),
        }
        return self.store.upsert(claim_id, data, created_by, action="retire")

    def get(self, claim_id: str) -> LedgerEntry | None:
        return self.store.get(claim_id)

    def all(self) -> list[LedgerEntry]:
        return self.store.list_entries()
