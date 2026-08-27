"""C5 materializers for legacy research_loop run directories.

The Step 14 boundary is deliberately conservative: legacy SQLite/YAML/graph
files remain operational inputs, while this module imports their contents into
canonical append-only ledgers and artifacts. The importer never deletes or
rewrites legacy files.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from praxist.core.storage import ArtifactWriter, rewrite_jsonl, utc_now

GRAPH_ARTIFACT_NAMES = (
    "graph_health.json",
    "unlinked_recent_findings.json",
    "graph.html",
    "graph_live.html",
)


@dataclass(frozen=True)
class LegacyResearchMemoryEntry:
    """Canonicalized research-memory row imported from legacy YAML ledgers."""

    ledger_name: str
    entry_id: str
    entry: dict[str, Any]
    source_path: Path


class LegacyRunDirAdapter:
    """Read-only adapter over the legacy run_dir layout."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)

    def collect_research_memory_entries(
        self,
    ) -> tuple[list[LegacyResearchMemoryEntry], dict[Path, dict[str, Any]]]:
        entries: list[LegacyResearchMemoryEntry] = []
        ledgers: dict[Path, dict[str, Any]] = {}
        ledger_dir = self.run_dir / "research_memory" / "ledgers"
        if not ledger_dir.exists():
            return entries, ledgers
        for path in sorted(ledger_dir.glob("*.yaml")):
            try:
                value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                continue
            if not isinstance(value, dict):
                continue
            ledger_name = str(value.get("ledger_name") or path.stem)
            ledgers[path] = {"ledger_name": ledger_name, "payload": value}
            raw_entries = value.get("entries")
            if not isinstance(raw_entries, list):
                continue
            for index, raw_entry in enumerate(raw_entries, start=1):
                if not isinstance(raw_entry, dict):
                    continue
                entry_id = str(raw_entry.get("id") or f"entry_{index:06d}")
                entries.append(
                    LegacyResearchMemoryEntry(
                        ledger_name=ledger_name,
                        entry_id=entry_id,
                        entry=raw_entry,
                        source_path=path,
                    )
                )
        return entries, ledgers

    def collect_graph_edges(self) -> list[dict[str, Any]]:
        db_path = self.run_dir / "shared_store.db"
        if not db_path.exists():
            return []
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
            conn.row_factory = sqlite3.Row
        except sqlite3.Error:
            return []
        try:
            rows = conn.execute(
                """SELECT edge_id, src_finding_id, dst_finding_id, edge_type,
                          confidence, created_by, created_at, rationale, provenance
                   FROM finding_edges
                   ORDER BY created_at ASC, edge_id ASC"""
            ).fetchall()
        except sqlite3.Error:
            return []
        finally:
            conn.close()
        edges = []
        for row in rows:
            edge = dict(row)
            provenance = edge.get("provenance")
            if isinstance(provenance, str):
                try:
                    edge["provenance"] = json.loads(provenance) if provenance else {}
                except json.JSONDecodeError:
                    edge["provenance"] = {"raw": provenance}
            edges.append(edge)
        return edges

    def collect_graph_artifacts(self) -> list[Path]:
        graph_dir = self.run_dir / "graph"
        if not graph_dir.exists():
            return []
        paths = []
        for name in GRAPH_ARTIFACT_NAMES:
            path = graph_dir / name
            if path.is_file():
                paths.append(path)
        return paths


def materialize_legacy_c5_views(
    prepared: Any,
    result: dict[str, Any],
    *,
    trajectory: Any,
    artifacts: ArtifactWriter,
) -> dict[str, int]:
    """Import legacy research_loop state into C5-compatible ledgers and artifact indexes."""
    adapter = LegacyRunDirAdapter(prepared.run_dir)
    memory_count = _materialize_research_memory(prepared, adapter, trajectory, artifacts)
    graph_edge_count = _materialize_graph_edges(prepared, adapter, trajectory, artifacts)
    graph_artifact_count = _materialize_graph_artifacts(prepared, adapter, trajectory, artifacts)
    return {
        "research_memory_record_count": memory_count,
        "graph_edge_count": graph_edge_count,
        "graph_artifact_count": graph_artifact_count,
    }


def _materialize_research_memory(
    prepared: Any,
    adapter: LegacyRunDirAdapter,
    trajectory: Any,
    artifacts: ArtifactWriter,
) -> int:
    entries, ledgers = adapter.collect_research_memory_entries()
    if not ledgers:
        rewrite_jsonl(Path(prepared.run_dir) / "memory" / "research_memory.jsonl", [])
        return 0
    import_event = trajectory.emit(
        "legacy.research_memory_imported",
        scope={"stage_id": "research_loop"},
        actor={"type": "core", "id": "c5_materializer"},
        payload={
            "ledger_count": len(ledgers),
            "entry_count": len(entries),
            "source": "legacy_research_memory_yaml",
        },
    )
    ledger_artifacts: dict[Path, dict[str, Any]] = {}
    for path, payload in ledgers.items():
        ledger_artifacts[path] = artifacts.persist_json(
            "research_memory_ledger",
            f"memory/legacy/{path.name}",
            {
                "source_path": _run_relative(prepared.run_dir, path),
                "legacy_ledger": payload["payload"],
            },
            schema_ref="core:legacy_research_memory_ledger.v1",
            producer={"stage_id": "research_loop", "role_ref": "workflow_stage:research_loop"},
            source_event_ids=[import_event["event_id"]],
        )
    records = [
        _research_memory_record(
            prepared,
            entry,
            ledger_artifacts.get(entry.source_path),
            source_event_ids=[import_event["event_id"]],
        )
        for entry in entries
    ]
    rewrite_jsonl(
        Path(prepared.run_dir) / "memory" / "research_memory.jsonl",
        sorted(records, key=lambda item: str(item.get("memory_record_id", ""))),
    )
    return len(records)


def _materialize_graph_edges(
    prepared: Any,
    adapter: LegacyRunDirAdapter,
    trajectory: Any,
    artifacts: ArtifactWriter,
) -> int:
    edges = adapter.collect_graph_edges()
    if not edges:
        rewrite_jsonl(Path(prepared.run_dir) / "memory" / "graph_edges.jsonl", [])
        return 0
    import_event = trajectory.emit(
        "legacy.graph_edges_imported",
        scope={"stage_id": "research_loop"},
        actor={"type": "core", "id": "c5_materializer"},
        payload={
            "edge_count": len(edges),
            "source": "legacy_sqlite_finding_edges",
        },
    )
    snapshot_artifact = artifacts.persist_json(
        "graph_edges_snapshot",
        "memory/legacy/graph_edges.json",
        {
            "source_path": "shared_store.db:finding_edges",
            "legacy_edges": edges,
        },
        schema_ref="core:legacy_graph_edges.v1",
        producer={"stage_id": "research_loop", "role_ref": "graph_maintainer:finding_graph_mvp"},
        source_event_ids=[import_event["event_id"]],
    )
    records = [
        _graph_edge_record(
            prepared,
            edge,
            index=index,
            artifact=snapshot_artifact,
            source_event_ids=[import_event["event_id"]],
        )
        for index, edge in enumerate(edges, start=1)
    ]
    rewrite_jsonl(Path(prepared.run_dir) / "memory" / "graph_edges.jsonl", records)
    return len(records)


def _materialize_graph_artifacts(
    prepared: Any,
    adapter: LegacyRunDirAdapter,
    trajectory: Any,
    artifacts: ArtifactWriter,
) -> int:
    paths = adapter.collect_graph_artifacts()
    if not paths:
        return 0
    import_event = trajectory.emit(
        "legacy.graph_artifacts_imported",
        scope={"stage_id": "research_loop"},
        actor={"type": "core", "id": "c5_materializer"},
        payload={"artifact_count": len(paths), "source": "legacy_graph_dir"},
    )
    count = 0
    for path in paths:
        rel = _run_relative(prepared.run_dir, path)
        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            artifacts.persist_json(
                "graph_materialized_artifact",
                f"graph/legacy/{path.name}",
                {"source_path": rel, "legacy_graph_artifact": payload},
                schema_ref="core:legacy_graph_artifact.v1",
                producer={
                    "stage_id": "research_loop",
                    "role_ref": "graph_maintainer:finding_graph_mvp",
                },
                source_event_ids=[import_event["event_id"]],
            )
            count += 1
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        artifacts.persist_text(
            "graph_materialized_artifact",
            f"graph/legacy/{path.name}",
            text,
            schema_ref="core:legacy_graph_artifact.v1",
            producer={
                "stage_id": "research_loop",
                "role_ref": "graph_maintainer:finding_graph_mvp",
            },
            content_type="text/html" if path.suffix.lower() == ".html" else "text/plain",
            source_event_ids=[import_event["event_id"]],
        )
        count += 1
    return count


def _research_memory_record(
    prepared: Any,
    entry: LegacyResearchMemoryEntry,
    artifact: dict[str, Any] | None,
    *,
    source_event_ids: list[str],
) -> dict[str, Any]:
    data = entry.entry.get("data") if isinstance(entry.entry.get("data"), dict) else {}
    record = {
        "schema_version": "praxist.research_memory.v1",
        "memory_record_id": f"{entry.ledger_name}:{entry.entry_id}",
        "run_id": prepared.run_id,
        "task_ref": prepared.task_ref,
        "stage_id": "research_loop",
        "entity_type": _entity_type_for_ledger(entry.ledger_name),
        "entity_id": entry.entry_id,
        "relation_type": "legacy_ledger_entry",
        "confidence": _optional_number(data.get("confidence")),
        "source_finding_ids": _source_finding_ids_from_refs(_source_refs_from_entry(entry.entry)),
        "source_refs": _source_refs_from_entry(entry.entry),
        "source_event_ids": source_event_ids,
        "artifact_refs": [artifact] if artifact else [],
        "created_at": str(entry.entry.get("created_at") or utc_now()),
        "legacy_payload": {
            "ledger_name": entry.ledger_name,
            "entry": entry.entry,
        },
    }
    return record


def _graph_edge_record(
    prepared: Any,
    edge: dict[str, Any],
    *,
    index: int,
    artifact: dict[str, Any],
    source_event_ids: list[str],
) -> dict[str, Any]:
    edge_id = str(edge.get("edge_id") or f"legacy_edge_{index:06d}")
    return {
        "schema_version": "praxist.graph_edge.v1",
        "graph_edge_id": edge_id,
        "run_id": prepared.run_id,
        "task_ref": prepared.task_ref,
        "stage_id": "research_loop",
        "source_entity_ref": {"kind": "finding", "id": str(edge.get("src_finding_id") or "")},
        "target_entity_ref": {"kind": "finding", "id": str(edge.get("dst_finding_id") or "")},
        "src_finding_id": str(edge.get("src_finding_id") or ""),
        "dst_finding_id": str(edge.get("dst_finding_id") or ""),
        "edge_type": str(edge.get("edge_type") or "related_to"),
        "confidence": _optional_number(edge.get("confidence")),
        "maintainer_ref": "graph_maintainer:finding_graph_mvp",
        "created_by": str(edge.get("created_by") or "legacy_finding_graph"),
        "advisory": True,
        "source_event_ids": source_event_ids,
        "artifact_refs": [artifact],
        "created_at": str(edge.get("created_at") or utc_now()),
        "legacy_payload": edge,
    }


def _entity_type_for_ledger(ledger_name: str) -> str:
    if ledger_name.endswith("_ledger"):
        return ledger_name.removesuffix("_ledger")
    return ledger_name


def _source_refs_from_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    data = entry.get("data") if isinstance(entry.get("data"), dict) else {}
    for key in ("source_ref", "source_refs", "supports", "challenges", "informs"):
        value = data.get(key)
        if isinstance(value, dict):
            refs.append(value)
        elif isinstance(value, list):
            refs.extend(item for item in value if isinstance(item, dict))
    return refs


def _source_finding_ids_from_refs(source_refs: list[dict[str, Any]]) -> list[str]:
    finding_ids: list[str] = []
    for ref in source_refs:
        value = ref.get("finding_id") or ref.get("id")
        if isinstance(value, str) and value:
            finding_ids.append(value)
        raw_many = ref.get("finding_ids")
        if isinstance(raw_many, list):
            finding_ids.extend(str(item) for item in raw_many if item)
    return sorted(set(finding_ids))


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _run_relative(run_dir: Path | str, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path(run_dir).resolve()))
    except ValueError:
        return str(path)
