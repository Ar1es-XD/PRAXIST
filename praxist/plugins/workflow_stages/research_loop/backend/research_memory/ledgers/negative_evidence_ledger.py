"""Negative evidence ledger — failed lineages, anti-synergy results,
killed hypotheses, surprising failures.

Why a separate ledger: retrieval_policy enforces ≥20% negative evidence
in evidence packs. A dedicated ledger makes that filter cheap.
"""

from __future__ import annotations

from pathlib import Path

from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers._ledger_base import (
    LedgerEntry,
    LedgerStore,
)


class NegativeEvidenceLedger:
    """YAML ledger wrapper for failed, falsifying, or boundary-setting evidence."""

    def __init__(self, run_dir: Path):
        path = Path(run_dir) / "research_memory" / "ledgers" / "negative_evidence_ledger.yaml"
        self.store = LedgerStore(path, "negative_evidence_ledger")

    def add(
        self,
        neg_id: str,
        title: str,
        category: str,  # "failed_lineage" | "anti_synergy" | "killed_hypothesis" | "surprising_failure"
        evidence_id: str = "",
        finding_id: str = "",
        summary: str = "",
        related_claim_id: str = "",
        created_by: str = "unknown",
    ) -> LedgerEntry:
        data = {
            "title": title,
            "category": category,
            "evidence_id": evidence_id,
            "finding_id": finding_id,
            "summary": summary,
            "related_claim_id": related_claim_id,
        }
        # idempotent on neg_id
        try:
            return self.store.append_only(neg_id, data, created_by)
        except ValueError:
            return self.store.upsert(neg_id, data, created_by, action="update")

    def list_recent(self, n: int = 12) -> list[LedgerEntry]:
        items = self.store.list_entries()
        items.sort(key=lambda e: e.created_at, reverse=True)
        return items[:n]

    def all(self) -> list[LedgerEntry]:
        return self.store.list_entries()
