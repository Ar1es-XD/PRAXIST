"""Dissent ledger — multi-PI disagreements + their resolution path.

Each entry records:
  disputed_claim
  pi_positions (per-role short statements)
  resolving_experiment
  decision_rule
  resolution_status: open | experiment_assigned | resolved
"""

from __future__ import annotations

from pathlib import Path

from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers._ledger_base import (
    LedgerEntry,
    LedgerStore,
)

VALID_STATUSES = ("open", "experiment_assigned", "resolved", "archived")


class DissentLedger:
    """YAML ledger wrapper for open objections and resolving experiments."""

    def __init__(self, run_dir: Path):
        path = Path(run_dir) / "research_memory" / "ledgers" / "dissent_ledger.yaml"
        self.store = LedgerStore(path, "dissent_ledger")

    def add(
        self,
        dissent_id: str,
        disputed_claim_id: str,
        pi_positions: dict[str, str],
        resolving_experiment: str = "",
        decision_rule: dict[str, str] | None = None,
        assigned_peer_role: str = "",
        status: str = "open",
        created_by: str = "unknown",
    ) -> LedgerEntry:
        if status not in VALID_STATUSES:
            raise ValueError(f"unknown dissent status: {status}")
        data = {
            "disputed_claim_id": disputed_claim_id,
            "pi_positions": dict(pi_positions),
            "resolving_experiment": resolving_experiment,
            "decision_rule": dict(decision_rule or {}),
            "assigned_peer_role": assigned_peer_role,
            "status": status,
        }
        return self.store.upsert(dissent_id, data, created_by, action="add")

    def update_status(
        self,
        dissent_id: str,
        new_status: str,
        resolution_summary: str = "",
        created_by: str = "unknown",
    ) -> LedgerEntry:
        if new_status not in VALID_STATUSES:
            raise ValueError(f"unknown dissent status: {new_status}")
        existing = self.store.get(dissent_id)
        if existing is None:
            raise KeyError(f"dissent {dissent_id} not found")
        data = dict(existing.data)
        data["status"] = new_status
        if resolution_summary:
            data["resolution_summary"] = resolution_summary
        return self.store.upsert(dissent_id, data, created_by, action="status_update")

    def list_open(self) -> list[LedgerEntry]:
        return self.store.filter(lambda e: e.data.get("status") in ("open", "experiment_assigned"))

    def all(self) -> list[LedgerEntry]:
        return self.store.list_entries()
