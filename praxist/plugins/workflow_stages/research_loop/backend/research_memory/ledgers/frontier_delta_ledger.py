"""Frontier delta ledger — anchor changes per synthesis.

Avoids embedding full frontier table in PI prompt. Records only:
  axis (task-owned metric name)
  previous_anchor (variant + value + evidence_id)
  current_anchor (variant + value + evidence_id)
  raw arithmetic delta
  generation_id at which the change was promoted
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers._ledger_base import (
    LedgerEntry,
    LedgerStore,
    _utcnow_iso,
)


class FrontierDeltaLedger:
    """YAML ledger wrapper for generation-to-generation frontier changes."""

    def __init__(self, run_dir: Path):
        path = Path(run_dir) / "research_memory" / "ledgers" / "frontier_delta_ledger.yaml"
        self.store = LedgerStore(path, "frontier_delta_ledger")

    def record_promote(
        self,
        generation_id: int,
        axis: str,
        previous_anchor: dict[str, Any] | None,
        current_anchor: dict[str, Any],
        created_by: str = "unknown",
    ) -> LedgerEntry:
        axis = str(axis or "").strip()
        if not axis:
            raise ValueError("axis must be a non-empty task metric name")
        delta_id = f"FD::gen{generation_id}::{axis}"
        # Raw arithmetic delta; direction determines whether its sign is favorable.
        prev_v = previous_anchor.get("value") if previous_anchor else None
        cur_v = current_anchor.get("value") if current_anchor else None
        if (
            prev_v is not None
            and cur_v is not None
            and isinstance(prev_v, (int, float))
            and isinstance(cur_v, (int, float))
        ):
            raw_delta = float(cur_v) - float(prev_v)
        else:
            raw_delta = None
        data = {
            "generation_id": generation_id,
            "axis": axis,
            "previous_anchor": previous_anchor or {},
            "current_anchor": current_anchor,
            "raw_delta": raw_delta,
        }
        return self.store.upsert(delta_id, data, created_by, action="promote_record")

    def replace_generation(
        self,
        generation_id: int,
        records: list[dict[str, Any]],
        *,
        created_by: str = "unknown",
    ) -> list[LedgerEntry]:
        """Atomically replace one generation's derived axis rows."""

        target = int(generation_id)

        def _do() -> list[LedgerEntry]:
            payload = self.store._read_all()
            retained: list[dict[str, Any]] = []
            existing: dict[str, dict[str, Any]] = {}
            for raw in payload["entries"]:
                data = raw.get("data") if isinstance(raw, dict) else None
                generation_raw = data.get("generation_id") if isinstance(data, dict) else None
                try:
                    entry_gen = int(generation_raw) if generation_raw is not None else None
                except (TypeError, ValueError):
                    entry_gen = None
                if entry_gen == target:
                    existing[str(raw.get("id") or "")] = raw
                else:
                    retained.append(raw)

            now = _utcnow_iso()
            replacements: list[LedgerEntry] = []
            for record in records:
                axis = str(record.get("axis") or "").strip()
                if not axis:
                    continue
                previous_anchor = dict(record.get("previous_anchor") or {})
                current_anchor = dict(record.get("current_anchor") or {})
                prev_v = previous_anchor.get("value")
                cur_v = current_anchor.get("value")
                raw_delta = (
                    float(cur_v) - float(prev_v)
                    if isinstance(prev_v, (int, float))
                    and not isinstance(prev_v, bool)
                    and isinstance(cur_v, (int, float))
                    and not isinstance(cur_v, bool)
                    else None
                )
                data = {
                    "generation_id": target,
                    "axis": axis,
                    "previous_anchor": previous_anchor,
                    "current_anchor": current_anchor,
                    "raw_delta": raw_delta,
                }
                entry_id = f"FD::gen{target}::{axis}"
                old = existing.get(entry_id)
                if old is None:
                    entry = LedgerEntry(
                        id=entry_id,
                        created_at=now,
                        created_by=created_by,
                        last_updated_at=now,
                        update_trail=[{"at": now, "by": created_by, "action": "create"}],
                        data=data,
                    )
                else:
                    trail = list(old.get("update_trail", []))
                    trail.append(
                        {
                            "at": now,
                            "by": created_by,
                            "action": "replace_generation",
                            "diff_keys": sorted(
                                key
                                for key, value in data.items()
                                if old.get("data", {}).get(key) != value
                            ),
                        }
                    )
                    entry = LedgerEntry(
                        id=entry_id,
                        created_at=str(old.get("created_at") or now),
                        created_by=str(old.get("created_by") or created_by),
                        last_updated_at=now,
                        update_trail=trail[-50:],
                        data=data,
                    )
                retained.append(entry.to_dict())
                replacements.append(entry)
            payload["entries"] = retained
            self.store._atomic_write(payload)
            return replacements

        return self.store._with_lock(_do)

    def latest_per_axis(self) -> dict[str, LedgerEntry]:
        # R7#1 fix: read inside the same lock used by writers. atomic_write
        # makes torn reads unlikely, but a lock-free read could still
        # observe an in-flight YAML state if a different process is
        # mid-write before os.replace completes.
        def _do():
            out: dict[str, LedgerEntry] = {}
            for e in self.store.list_entries():
                ax = e.data.get("axis")
                if isinstance(ax, str) and ax:
                    cur = out.get(ax)
                    if cur is None or e.data.get("generation_id", -1) > cur.data.get(
                        "generation_id", -1
                    ):
                        out[ax] = e
            return out

        return self.store._with_lock(_do)

    def prior_anchors_for_generation(self, generation_id: int) -> dict[str, dict[str, Any]]:
        """Return stable predecessor anchors for an idempotent generation update."""

        target = int(generation_id)

        def _do() -> dict[str, dict[str, Any]]:
            latest_earlier: dict[str, LedgerEntry] = {}
            same_generation: dict[str, LedgerEntry] = {}
            for entry in self.store.list_entries():
                axis = entry.data.get("axis")
                generation_raw = entry.data.get("generation_id")
                if generation_raw is None:
                    continue
                try:
                    entry_gen = int(generation_raw)
                except (TypeError, ValueError):
                    continue
                if not isinstance(axis, str) or not axis:
                    continue
                if entry_gen == target:
                    same_generation[axis] = entry
                elif entry_gen < target:
                    current = latest_earlier.get(axis)
                    if current is None or entry_gen > int(current.data.get("generation_id", -1)):
                        latest_earlier[axis] = entry
            axes = set(latest_earlier) | set(same_generation)
            return {
                axis: dict(
                    same_generation[axis].data.get("previous_anchor", {})
                    if axis in same_generation
                    else latest_earlier[axis].data.get("current_anchor", {})
                )
                for axis in axes
            }

        return self.store._with_lock(_do)

    def history_for_axis(self, axis: str) -> list[LedgerEntry]:
        items = self.store.filter(lambda e: e.data.get("axis") == axis)
        items.sort(key=lambda e: e.data.get("generation_id", 0))
        return items

    def all(self) -> list[LedgerEntry]:
        return self.store.list_entries()
