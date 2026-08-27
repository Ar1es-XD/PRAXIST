"""Hypothesis ledger — predictions, experiments, kill conditions, status.

Each hypothesis tracks:
  source: which agenda/PI proposed it
  prediction: what was predicted
  experiment: minimal test
  kill_condition / promote_condition
  status: pending | testing | confirmed | killed | pivoted
"""

from __future__ import annotations

from pathlib import Path

from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers._ledger_base import (
    LedgerEntry,
    LedgerStore,
)

VALID_STATUSES = ("pending", "testing", "confirmed", "killed", "pivoted")


class HypothesisLedger:
    """YAML ledger wrapper for active and proposed hypotheses."""

    def __init__(self, run_dir: Path):
        path = Path(run_dir) / "research_memory" / "ledgers" / "hypothesis_ledger.yaml"
        self.store = LedgerStore(path, "hypothesis_ledger")

    def upsert(
        self,
        hyp_id: str,
        title: str,
        prediction: str = "",
        minimal_test: str = "",
        kill_condition: str = "",
        promote_condition: str = "",
        status: str = "pending",
        source_agenda: str = "",
        source_findings: list[str] | None = None,
        created_by: str = "unknown",
        action: str = "upsert",
    ) -> LedgerEntry:
        if status not in VALID_STATUSES:
            raise ValueError(f"unknown hypothesis status: {status}")
        data = {
            "title": title,
            "prediction": prediction,
            "minimal_test": minimal_test,
            "kill_condition": kill_condition,
            "promote_condition": promote_condition,
            "status": status,
            "source_agenda": source_agenda,
            "source_findings": list(source_findings or []),
        }
        return self.store.upsert(hyp_id, data, created_by, action)

    def list_active(self) -> list[LedgerEntry]:
        return self.store.filter(lambda e: e.data.get("status") in ("pending", "testing"))

    def list_killed(self) -> list[LedgerEntry]:
        return self.store.filter(lambda e: e.data.get("status") == "killed")

    def get(self, hyp_id: str) -> LedgerEntry | None:
        return self.store.get(hyp_id)

    def all(self) -> list[LedgerEntry]:
        return self.store.list_entries()
