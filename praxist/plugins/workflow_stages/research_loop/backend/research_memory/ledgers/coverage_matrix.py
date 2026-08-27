"""Coverage matrix — what (variant, hyperparameter) grids have been tested.

Critical use: bridge contracts MUST query this before assignment to avoid
the gen2_peer2 / gen3_peer2 0-output anti-pattern (PI re-assigned a target
already covered).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers._ledger_base import (
    LedgerEntry,
    LedgerStore,
)


def _grid_id(variant_family: str, parameter: str) -> str:
    return f"GRID::{variant_family}::{parameter}"


def _bridge_id(family_a: str, family_b: str, dimension: str) -> str:
    pair = sorted([family_a, family_b])
    return f"BRIDGE::{pair[0]}::{pair[1]}::{dimension}"


class CoverageMatrix:
    """YAML ledger wrapper for explored parameter grids and bridge coverage."""

    def __init__(self, run_dir: Path):
        path = Path(run_dir) / "research_memory" / "ledgers" / "coverage_matrix.yaml"
        self.store = LedgerStore(path, "coverage_matrix")

    # -------------------- single-family hyperparameter grids

    def record_grid_point(
        self,
        variant_family: str,
        parameter: str,
        value: float,
        seed_count: int = 0,
        source_evidence_id: str = "",
        created_by: str = "unknown",
    ) -> LedgerEntry:
        # R2#10 fix: read-modify-write inside ONE lock. The previous
        # implementation read existing outside the lock, then called
        # upsert() which acquired its own lock — leaving a TOCTOU window
        # where a concurrent writer's append could be lost.
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers._ledger_base import (
            LedgerEntry as _LE,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers._ledger_base import (
            _utcnow_iso,
        )

        gid = _grid_id(variant_family, parameter)

        def _do():
            d_all = self.store._read_all()
            existing_idx = None
            for i, e in enumerate(d_all["entries"]):
                if e.get("id") == gid:
                    existing_idx = i
                    break
            now = _utcnow_iso()
            if existing_idx is None:
                data = {
                    "variant_family": variant_family,
                    "parameter": parameter,
                    "values_tested": [value],
                    "seed_counts": {str(value): seed_count},
                    "sources": [source_evidence_id] if source_evidence_id else [],
                }
                entry = _LE(
                    id=gid,
                    created_at=now,
                    created_by=created_by,
                    last_updated_at=now,
                    update_trail=[{"at": now, "by": created_by, "action": "grid_init"}],
                    data=dict(data),
                )
                d_all["entries"].append(entry.to_dict())
                self.store._atomic_write(d_all)
                return entry
            old = d_all["entries"][existing_idx]
            d = dict(old.get("data", {}))
            values = list(d.get("values_tested", []))
            if value not in values:
                values.append(value)
                values.sort()
            # R5#1 fix: normalize seed_counts keys to str on read so a YAML
            # round-trip that preserved numeric keys doesn't produce mixed
            # str/int key types in the same dict.
            raw_seeds = d.get("seed_counts", {})
            seeds = {str(k): v for k, v in dict(raw_seeds).items()}
            seeds[str(value)] = max(seeds.get(str(value), 0), seed_count)
            sources = list(d.get("sources", []))
            if source_evidence_id and source_evidence_id not in sources:
                sources.append(source_evidence_id)
            d["values_tested"] = values
            d["seed_counts"] = seeds
            d["sources"] = sources
            trail = list(old.get("update_trail", []))
            trail.append({"at": now, "by": created_by, "action": "grid_extend"})
            old["data"] = d
            old["last_updated_at"] = now
            old["update_trail"] = trail[-50:]
            d_all["entries"][existing_idx] = old
            self.store._atomic_write(d_all)
            return _LE.from_dict(old)

        return self.store._with_lock(_do)

    def query_grid(self, variant_family: str, parameter: str) -> dict[str, Any] | None:
        e = self.store.get(_grid_id(variant_family, parameter))
        return e.data if e else None

    # -------------------- bridge coverage

    def record_bridge_point(
        self,
        family_a: str,
        family_b: str,
        dimension: str,
        point: Any,
        source_evidence_id: str = "",
        created_by: str = "unknown",
    ) -> LedgerEntry:
        # R2#10 fix: same TOCTOU pattern — RMW inside one lock.
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers._ledger_base import (
            LedgerEntry as _LE,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers._ledger_base import (
            _utcnow_iso,
        )

        bid = _bridge_id(family_a, family_b, dimension)

        def _do():
            d_all = self.store._read_all()
            existing_idx = None
            for i, e in enumerate(d_all["entries"]):
                if e.get("id") == bid:
                    existing_idx = i
                    break
            now = _utcnow_iso()
            if existing_idx is None:
                data = {
                    "variant_pair": sorted([family_a, family_b]),
                    "relation": "bridge",
                    "grid_dimension": dimension,
                    "bridge_points_tested": [point],
                    "sources": [source_evidence_id] if source_evidence_id else [],
                }
                entry = _LE(
                    id=bid,
                    created_at=now,
                    created_by=created_by,
                    last_updated_at=now,
                    update_trail=[{"at": now, "by": created_by, "action": "bridge_init"}],
                    data=dict(data),
                )
                d_all["entries"].append(entry.to_dict())
                self.store._atomic_write(d_all)
                return entry
            old = d_all["entries"][existing_idx]
            d = dict(old.get("data", {}))
            pts = list(d.get("bridge_points_tested", []))
            if point not in pts:
                pts.append(point)
            sources = list(d.get("sources", []))
            if source_evidence_id and source_evidence_id not in sources:
                sources.append(source_evidence_id)
            d["bridge_points_tested"] = pts
            d["sources"] = sources
            trail = list(old.get("update_trail", []))
            trail.append({"at": now, "by": created_by, "action": "bridge_extend"})
            old["data"] = d
            old["last_updated_at"] = now
            old["update_trail"] = trail[-50:]
            d_all["entries"][existing_idx] = old
            self.store._atomic_write(d_all)
            return _LE.from_dict(old)

        return self.store._with_lock(_do)

    def is_bridge_covered(
        self,
        family_a: str,
        family_b: str,
        dimension: str,
        min_points: int = 1,
    ) -> bool:
        e = self.store.get(_bridge_id(family_a, family_b, dimension))
        if e is None:
            return False
        return len(e.data.get("bridge_points_tested", [])) >= min_points

    def query_bridge(
        self,
        family_a: str,
        family_b: str,
        dimension: str,
    ) -> dict[str, Any] | None:
        e = self.store.get(_bridge_id(family_a, family_b, dimension))
        return e.data if e else None

    def all(self) -> list[LedgerEntry]:
        return self.store.list_entries()
