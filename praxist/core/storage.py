"""Run directory, JSONL, and artifact helpers for Gate A."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from praxist.core.redaction import redact_json


def utc_now() -> str:
    """Return the current UTC timestamp used by Praxist run artifacts."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def utc_stamp() -> str:
    """Return a filesystem-safe UTC timestamp string."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")


def new_run_id(task_slug: str) -> str:
    """Generate a new opaque run identifier."""
    return f"{utc_stamp()}_{task_slug}_{secrets.token_hex(4)}"


def write_json(path: Path, value: Any) -> None:
    """Write a redacted JSON artifact with stable indentation and sorted keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    redacted, _ = redact_json(value)
    tmp.write_text(
        json.dumps(redacted, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def append_jsonl(path: Path, value: Any) -> None:
    """Append one redacted JSONL record to an append-only run ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    redacted, _ = redact_json(value)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(redacted, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read JSONL records from a run ledger, skipping blank lines."""
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.exists():
        return records, [f"missing:{path.name}"]
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
            else:
                errors.append(f"{path.name}:{line_no}:not_object")
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}:{line_no}:json_decode:{exc.msg}")
    return records, errors


def rewrite_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Atomically rewrite an entire JSONL ledger with redacted records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    for record in records:
        append_jsonl(path, record)


def sha256_bytes(data: bytes) -> str:
    """Return a SHA-256 digest for raw bytes."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


OUTPUT_LEDGER_RELS = (
    "findings/findings.jsonl",
    "findings/frontier.jsonl",
    "memory/research_memory.jsonl",
    "memory/graph_edges.jsonl",
)


def output_ledger_hashes(run_dir: Path) -> dict[str, str]:
    """Compute content hashes for canonical output ledgers in a run directory."""
    run_dir = Path(run_dir)
    hashes: dict[str, str] = {}
    for rel in OUTPUT_LEDGER_RELS:
        path = run_dir / rel
        hashes[rel] = sha256_bytes(path.read_bytes() if path.exists() else b"")
    return hashes


def ensure_run_dirs(run_dir: Path) -> None:
    """Create the minimum run directory layout used by startup and replay."""
    for rel in (
        "artifacts/by_id",
        "findings",
        "memory",
        "logs",
        "indexes",
        "replay",
    ):
        (run_dir / rel).mkdir(parents=True, exist_ok=True)


class ArtifactWriter:
    """Run-local artifact writer with stable ids, redaction, and artifact_index accounting."""

    def __init__(self, run_dir: Path, trajectory: Any | None = None) -> None:
        self.run_dir = run_dir
        self.trajectory = trajectory
        self.run_id = str(getattr(trajectory, "run_id", run_dir.name))
        self._seq = _existing_artifact_seq(run_dir)

    def persist_json(
        self,
        artifact_type: str,
        logical_path: str,
        payload: dict[str, Any],
        *,
        schema_ref: str | None,
        producer: dict[str, str],
        source_event_ids: list[str] | None = None,
        source_artifact_ids: list[str] | None = None,
        redaction_level: str = "redacted",
        artifact_role: str | None = None,
        artifact_status: str | None = None,
        runtime_fact_source: bool | None = None,
        derived_from: list[str] | None = None,
    ) -> dict[str, Any]:
        redacted_payload, hits = redact_json(payload)
        payload_bytes = (
            json.dumps(redacted_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        return self._persist_payload(
            artifact_type=artifact_type,
            logical_path=logical_path,
            payload_bytes=payload_bytes,
            payload_name="payload.json",
            content_type="application/json",
            schema_ref=schema_ref,
            producer=producer,
            source_event_ids=source_event_ids or [],
            source_artifact_ids=source_artifact_ids or [],
            redaction_level=redaction_level,
            redaction_hits=hits,
            artifact_role=artifact_role,
            artifact_status=artifact_status,
            runtime_fact_source=runtime_fact_source,
            derived_from=derived_from,
        )

    def persist_text(
        self,
        artifact_type: str,
        logical_path: str,
        payload: str,
        *,
        schema_ref: str | None,
        producer: dict[str, str],
        content_type: str = "text/markdown",
        source_event_ids: list[str] | None = None,
        artifact_role: str | None = None,
        artifact_status: str | None = None,
        runtime_fact_source: bool | None = None,
        derived_from: list[str] | None = None,
    ) -> dict[str, Any]:
        redacted_payload, hits = redact_json(payload)
        payload_bytes = str(redacted_payload).encode("utf-8")
        return self._persist_payload(
            artifact_type=artifact_type,
            logical_path=logical_path,
            payload_bytes=payload_bytes,
            payload_name="payload.md",
            content_type=content_type,
            schema_ref=schema_ref,
            producer=producer,
            source_event_ids=source_event_ids or [],
            source_artifact_ids=[],
            redaction_level="redacted",
            redaction_hits=hits,
            artifact_role=artifact_role,
            artifact_status=artifact_status,
            runtime_fact_source=runtime_fact_source,
            derived_from=derived_from,
        )

    def _persist_payload(
        self,
        *,
        artifact_type: str,
        logical_path: str,
        payload_bytes: bytes,
        payload_name: str,
        content_type: str,
        schema_ref: str | None,
        producer: dict[str, str],
        source_event_ids: list[str],
        source_artifact_ids: list[str],
        redaction_level: str,
        redaction_hits: list[str],
        artifact_role: str | None,
        artifact_status: str | None,
        runtime_fact_source: bool | None,
        derived_from: list[str] | None,
    ) -> dict[str, Any]:
        artifact_id, artifact_dir = self._next_artifact_dir()
        payload_path = artifact_dir / payload_name
        tmp_payload = payload_path.with_suffix(payload_path.suffix + ".tmp")
        tmp_payload.write_bytes(payload_bytes)
        os.replace(tmp_payload, payload_path)
        redacted_logical_path, logical_path_hits = redact_json(logical_path)
        metadata = {
            "schema_version": "praxist.artifact.v1",
            "artifact_id": artifact_id,
            "run_id": self.run_id,
            "artifact_type": artifact_type,
            "logical_path": str(redacted_logical_path),
            "payload_path": f"artifacts/by_id/{artifact_id}/{payload_name}",
            "content_hash": sha256_bytes(payload_bytes),
            "content_type": content_type,
            "schema_ref": schema_ref,
            "size_bytes": len(payload_bytes),
            "producer": producer,
            "source_event_ids": source_event_ids,
            "source_artifact_ids": source_artifact_ids,
            "redaction_level": redaction_level,
            "redaction_hits": sorted(set([*redaction_hits, *logical_path_hits])),
            "created_at": utc_now(),
        }
        if artifact_role:
            metadata["artifact_role"] = str(artifact_role)
        if artifact_status:
            metadata["artifact_status"] = str(artifact_status)
        if runtime_fact_source is not None:
            metadata["runtime_fact_source"] = bool(runtime_fact_source)
        if derived_from:
            metadata["derived_from"] = [str(item) for item in derived_from if str(item).strip()]
        write_json(artifact_dir / "metadata.json", metadata)
        append_jsonl(self.run_dir / "artifact_index.jsonl", metadata)
        if self.trajectory is not None:
            self.trajectory.emit(
                "artifact.persisted",
                actor={"type": "core", "id": "artifact_writer"},
                payload={
                    "artifact_id": artifact_id,
                    "artifact_type": artifact_type,
                    "logical_path": logical_path,
                    "content_hash": metadata["content_hash"],
                },
                artifact_refs=[metadata],
            )
        return metadata

    def _next_artifact_dir(self) -> tuple[str, Path]:
        while True:
            self._seq += 1
            artifact_id = f"art_{self._seq:06d}"
            artifact_dir = self.run_dir / "artifacts" / "by_id" / artifact_id
            try:
                artifact_dir.mkdir(parents=True, exist_ok=False)
                return artifact_id, artifact_dir
            except FileExistsError:
                continue


def _existing_artifact_seq(run_dir: Path) -> int:
    highest = 0
    artifact_root = run_dir / "artifacts" / "by_id"
    if artifact_root.exists():
        for path in artifact_root.glob("art_*"):
            suffix = path.name.removeprefix("art_")
            if suffix.isdigit():
                highest = max(highest, int(suffix))
    index_path = run_dir / "artifact_index.jsonl"
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            artifact_id = str(record.get("artifact_id", ""))
            suffix = artifact_id.removeprefix("art_")
            if suffix.isdigit():
                highest = max(highest, int(suffix))
    return highest
