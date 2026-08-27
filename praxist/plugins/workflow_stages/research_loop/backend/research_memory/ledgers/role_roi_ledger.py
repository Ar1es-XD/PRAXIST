"""Role ROI ledger — per-peer-role contribution accounting.

Tracks per-generation:
  exploit_output_count
  falsifier_decisions (KEEP/KILL/PIVOT)
  bridge_output_count + bridge_zero_streak
  anti_mainline_novelty_score
  theorist_insight_count + theorist_adoption_rate
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers._ledger_base import (
    LedgerEntry,
    LedgerStore,
)


class RoleROILedger:
    """YAML ledger wrapper for role utility and return-on-investment observations."""

    def __init__(self, run_dir: Path):
        path = Path(run_dir) / "research_memory" / "ledgers" / "role_roi_ledger.yaml"
        self.store = LedgerStore(path, "role_roi_ledger")

    def record_gen_summary(
        self,
        generation_id: int,
        per_role: dict[str, dict[str, Any]],
        created_by: str = "unknown",
    ) -> LedgerEntry:
        roi_id = f"ROI::gen{generation_id}"
        data = {
            "generation_id": generation_id,
            "per_role": dict(per_role),
        }
        return self.store.upsert(roi_id, data, created_by, action="gen_summary")

    def get_gen(self, generation_id: int) -> LedgerEntry | None:
        return self.store.get(f"ROI::gen{generation_id}")

    def all(self) -> list[LedgerEntry]:
        return self.store.list_entries()
