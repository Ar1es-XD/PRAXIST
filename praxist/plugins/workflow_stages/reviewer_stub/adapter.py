"""Local reviewer implementation for the optional reviewer workflow stage.

The reviewer checks claims against the run record that already exists on disk.
It never re-runs task evaluators and never promotes or demotes candidates; the
output is an audit artifact for operators and agents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from praxist.core.storage import ArtifactWriter, read_jsonl, sha256_bytes
from praxist.core.trajectory import TrajectoryWriter
from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
    AUDIT_SNAPSHOT,
    COMMITTED,
    attach_artifact_semantics,
)

_SCHEMA_REF = "core:review_artifact.v1"


def run_local_artifact_review(
    *,
    run_dir: Path,
    run_id: str,
    stage_ref: str,
    source_event_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Review artifact/provenance consistency and persist an audit report."""

    run_dir = Path(run_dir)
    source_event_ids = source_event_ids or []
    findings: list[dict[str, Any]] = []
    artifact_records, artifact_errors = read_jsonl(run_dir / "artifact_index.jsonl")
    trajectory_records, trajectory_errors = read_jsonl(run_dir / "trajectory.jsonl")
    if _has_finalized_event(trajectory_records):
        return {
            "status": "failed",
            "success": False,
            "output_artifacts": [],
            "summary": {
                "promotion_effect": "none",
                "reason": "run_already_finalized",
                "write_effect": "none",
            },
            "error": (
                "local reviewer refuses to append artifacts or trajectory events after "
                "run.finalized; run it before finalization or copy the run for audit."
            ),
        }

    trajectory = TrajectoryWriter(run_dir, run_id)

    for error in artifact_errors:
        findings.append(
            _finding("warning", "artifact_index_read", error, "Inspect artifact_index.jsonl.")
        )
    for error in trajectory_errors:
        findings.append(_finding("warning", "trajectory_read", error, "Inspect trajectory.jsonl."))

    artifact_by_id = {
        str(record.get("artifact_id")): record
        for record in artifact_records
        if isinstance(record.get("artifact_id"), str)
    }
    _check_artifact_payload_hashes(run_dir, artifact_records, findings)
    _check_artifact_source_refs(artifact_by_id, artifact_records, findings)
    _check_trajectory_artifact_refs(artifact_by_id, trajectory_records, findings)
    _check_literature_artifact_roles(artifact_records, findings)
    _check_run_summary(run_dir, findings)

    severity_counts: dict[str, int] = {}
    for finding in findings:
        severity = str(finding.get("severity") or "info")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    if not findings:
        findings.append(
            _finding(
                "info",
                "no_record_mismatch_detected",
                "No artifact-index, trajectory-reference, or run-summary inconsistency was detected.",
                "Continue normal review if the scientific question needs deeper human or domain checks.",
            )
        )
        severity_counts["info"] = 1

    payload = attach_artifact_semantics(
        {
            "schema_version": _SCHEMA_REF,
            "reviewer": "local_artifact_reviewer",
            "review_scope": [
                "artifact_index content hashes",
                "artifact source references",
                "trajectory artifact references",
                "literature/database artifact roles",
                "run_summary structural consistency",
            ],
            "claim_policy": (
                "The reviewer checks whether claims in Praxist records are supported by "
                "existing run artifacts. It does not rerun experiments or judge whether "
                "the chosen scientific method was optimal."
            ),
            "findings": findings,
            "summary": {
                "artifact_count": len(artifact_records),
                "trajectory_event_count": len(trajectory_records),
                "finding_count": len(findings),
                "severity_counts": severity_counts,
                "promotion_effect": "none",
            },
        },
        role=AUDIT_SNAPSHOT,
        status=COMMITTED,
        stage="reviewer",
        actor="workflow_stage:reviewer_stub",
        canonical_sources=[
            "artifact_index.jsonl",
            "trajectory.jsonl",
            "run_summary.json",
        ],
        runtime_fact_source=False,
        notes="Audit-only reviewer report; not a runtime fact source.",
    )
    artifacts = ArtifactWriter(run_dir, trajectory)
    artifact = artifacts.persist_json(
        "review.report",
        "workflow/reviewer_report.json",
        payload,
        schema_ref=_SCHEMA_REF,
        producer={"stage_id": "reviewer", "workflow_stage_ref": stage_ref},
        source_event_ids=source_event_ids,
        artifact_role=AUDIT_SNAPSHOT,
        artifact_status=COMMITTED,
        runtime_fact_source=False,
    )
    trajectory.emit(
        "workflow.stage_succeeded",
        scope={"stage_id": "reviewer"},
        actor={"type": "workflow_stage", "id": stage_ref},
        payload={
            "artifact_id": artifact["artifact_id"],
            "finding_count": len(findings),
            "severity_counts": severity_counts,
            "promotion_effect": "none",
        },
        artifact_refs=[artifact],
        parent_event_ids=source_event_ids,
    )
    return {
        "status": "succeeded",
        "success": True,
        "output_artifacts": [artifact],
        "summary": payload["summary"],
    }


def _finding(severity: str, check_id: str, evidence: str, recommendation: str) -> dict[str, str]:
    return {
        "severity": severity,
        "check_id": check_id,
        "claim": check_id.replace("_", " "),
        "evidence": evidence,
        "recommendation": recommendation,
    }


def _has_finalized_event(events: list[dict[str, Any]]) -> bool:
    return any(event.get("kind") == "run.finalized" for event in events)


def _check_artifact_payload_hashes(
    run_dir: Path,
    records: list[dict[str, Any]],
    findings: list[dict[str, Any]],
) -> None:
    for record in records:
        artifact_id = str(record.get("artifact_id") or "")
        payload_rel = str(record.get("payload_path") or "")
        expected_hash = str(record.get("content_hash") or "")
        if not artifact_id or not payload_rel or not expected_hash:
            findings.append(
                _finding(
                    "warning",
                    "artifact_metadata_incomplete",
                    f"artifact={artifact_id or '<missing>'} lacks id, payload_path, or content_hash",
                    "Regenerate the artifact through ArtifactWriter if it must be retained.",
                )
            )
            continue
        payload_path = run_dir / payload_rel
        if not payload_path.is_file():
            findings.append(
                _finding(
                    "error",
                    "artifact_payload_missing",
                    f"artifact={artifact_id} payload_path={payload_rel} is missing",
                    "Treat claims backed only by this artifact as unsupported until the payload is restored.",
                )
            )
            continue
        try:
            actual_hash = sha256_bytes(payload_path.read_bytes())
        except OSError as exc:
            findings.append(
                _finding(
                    "error",
                    "artifact_payload_unreadable",
                    f"artifact={artifact_id} payload_path={payload_rel}: {exc}",
                    "Fix filesystem permissions or regenerate the artifact.",
                )
            )
            continue
        if actual_hash != expected_hash:
            findings.append(
                _finding(
                    "error",
                    "artifact_hash_mismatch",
                    f"artifact={artifact_id} expected={expected_hash} actual={actual_hash}",
                    "Do not trust claims tied to this artifact until the source record is repaired.",
                )
            )


def _check_artifact_source_refs(
    artifact_by_id: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
    findings: list[dict[str, Any]],
) -> None:
    for record in records:
        artifact_id = str(record.get("artifact_id") or "")
        source_ids = record.get("source_artifact_ids") or []
        if not isinstance(source_ids, list):
            continue
        missing = [
            str(source_id) for source_id in source_ids if str(source_id) not in artifact_by_id
        ]
        if missing:
            findings.append(
                _finding(
                    "warning",
                    "artifact_source_ref_missing",
                    f"artifact={artifact_id} references missing source_artifact_ids={missing}",
                    "Use the artifact as an audit clue only; provenance chain is incomplete.",
                )
            )


def _check_trajectory_artifact_refs(
    artifact_by_id: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
    findings: list[dict[str, Any]],
) -> None:
    for event in events:
        event_id = str(event.get("event_id") or "")
        refs = event.get("artifact_refs") or []
        if not isinstance(refs, list):
            continue
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            artifact_id = str(ref.get("artifact_id") or "")
            if artifact_id and artifact_id not in artifact_by_id:
                findings.append(
                    _finding(
                        "warning",
                        "trajectory_artifact_ref_missing",
                        f"event={event_id} references artifact_id={artifact_id} absent from artifact_index",
                        "Treat the event as incomplete provenance until the artifact index is repaired.",
                    )
                )


def _check_literature_artifact_roles(
    records: list[dict[str, Any]],
    findings: list[dict[str, Any]],
) -> None:
    for record in records:
        artifact_id = str(record.get("artifact_id") or "")
        artifact_type = str(record.get("artifact_type") or "").lower()
        logical_path = str(record.get("logical_path") or "").lower()
        runtime_fact_source = record.get("runtime_fact_source")
        if ("literature" in artifact_type or "literature" in logical_path) and runtime_fact_source:
            findings.append(
                _finding(
                    "warning",
                    "literature_marked_runtime_fact",
                    f"artifact={artifact_id} is literature/database context but runtime_fact_source=true",
                    "Mark literature/database records as contextual signals unless a task evaluator measured them.",
                )
            )


def _check_run_summary(run_dir: Path, findings: list[dict[str, Any]]) -> None:
    path = run_dir / "run_summary.json"
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        findings.append(
            _finding(
                "warning",
                "run_summary_unreadable",
                f"run_summary.json could not be parsed: {exc}",
                "Use trajectory and generation artifacts as the source of truth until summary is repaired.",
            )
        )
        return
    if not isinstance(payload, dict):
        findings.append(
            _finding(
                "warning",
                "run_summary_not_object",
                "run_summary.json is not a JSON object",
                "Regenerate run summary from canonical run artifacts.",
            )
        )
