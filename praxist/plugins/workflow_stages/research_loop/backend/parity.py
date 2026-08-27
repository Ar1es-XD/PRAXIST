"""Parity checks for legacy research_loop dogfood runs.

Step 17 keeps the expensive end-to-end run manual/nightly, but makes the
expected parity surface executable. The verifier reads a completed run_dir and
checks that legacy outputs, canonical ledgers, prompt-guidance surfaces, and
operator-facing artifacts still line up.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from praxist.core.replay import verify_run
from praxist.core.storage import read_jsonl, write_json

GRAPH_ARTIFACT_NAMES = (
    "graph_health.json",
    "unlinked_recent_findings.json",
    "graph.html",
    "graph_live.html",
)


@dataclass(frozen=True)
class ParityCheck:
    """One parity assertion produced by a task or research_loop verifier."""

    check_id: str
    status: str
    message: str
    severity: str = "error"
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.status == "fail" and self.severity == "error"

    @property
    def warning(self) -> bool:
        return self.status in {"fail", "warn"} and self.severity == "warning"


@dataclass(frozen=True)
class ParityReport:
    """Aggregate parity verification report with errors, warnings, and count summaries."""

    schema_version: str
    run_dir: str
    success: bool
    summary: dict[str, Any]
    checks: list[dict[str, Any]]
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_research_loop_parity(
    run_dir: Path | str,
    *,
    deliverables_dir: Path | str | None = None,
    strict: bool = False,
    write_report: bool = False,
) -> dict[str, Any]:
    """Verify old-vs-new parity for a completed research_loop run_dir."""

    run_dir = Path(run_dir)
    deliverables_path = Path(deliverables_dir) if deliverables_dir else None
    legacy = _collect_legacy_surfaces(run_dir)
    canonical = _collect_canonical_surfaces(run_dir)
    checks: list[ParityCheck] = []

    replay_report = verify_run(run_dir)
    checks.append(_check_replay(replay_report))
    checks.append(_check_task_ref(run_dir))
    checks.append(_check_legacy_findings_materialized(legacy, canonical))
    checks.append(_check_frontier_materialized(legacy, canonical))
    checks.append(_check_research_memory_materialized(legacy, canonical))
    checks.append(_check_graph_edges_materialized(legacy, canonical))
    checks.append(_check_graph_artifacts_materialized(legacy, canonical))
    checks.append(_check_prompt_guidance_surfaces(legacy, canonical, strict=strict))
    checks.append(_check_panel_agenda_surface(legacy, strict=strict))
    checks.append(_check_operator_status_surface(run_dir, legacy, canonical, strict=strict))
    checks.append(_check_resource_guard_usage(canonical, strict=strict))
    checks.append(_check_deliverables(deliverables_path, strict=strict))

    errors = [check.message for check in checks if check.failed]
    warnings = [check.message for check in checks if check.warning]
    success = not errors and (not strict or not warnings)
    report = ParityReport(
        schema_version="praxist.research_loop_parity.v1",
        run_dir=str(run_dir),
        success=success,
        summary={
            "legacy_findings": len(legacy["finding_ids"]),
            "canonical_findings": len(canonical["finding_ids"]),
            "legacy_frontier": len(legacy["frontier_ids"]),
            "canonical_frontier": len(canonical["frontier_ids"]),
            "legacy_memory_entries": legacy["research_memory_entry_count"],
            "canonical_memory_records": len(canonical["research_memory"]),
            "legacy_graph_edges": legacy["graph_edge_count"],
            "canonical_graph_edges": len(canonical["graph_edges"]),
            "postgen_prompt_count": len(legacy["postgen_prompt_paths"]),
        },
        checks=[asdict(check) for check in checks],
        errors=errors,
        warnings=warnings,
    )
    payload = report.to_dict()
    if write_report:
        write_json(run_dir / "research_loop_parity_report.json", payload)
    return payload


def _collect_legacy_surfaces(run_dir: Path) -> dict[str, Any]:
    legacy_findings = _legacy_findings(run_dir)
    frontier_ids = _legacy_frontier_ids(run_dir)
    prompts = _prompt_files(run_dir)
    postgen_prompts = [
        path
        for path in prompts
        if _prompt_generation(path) is not None and int(_prompt_generation(path) or 0) > 0
    ]
    return {
        "finding_ids": set(legacy_findings),
        "frontier_ids": frontier_ids,
        "research_memory_entry_count": _legacy_research_memory_entry_count(run_dir),
        "graph_edge_count": len(_legacy_graph_edges(run_dir)),
        "graph_artifact_names": _legacy_graph_artifact_names(run_dir),
        "prompt_paths": prompts,
        "postgen_prompt_paths": postgen_prompts,
        "prompt_texts": {path: _read_text(path) for path in prompts},
        "agenda_paths": sorted((run_dir / "agendas").glob("research_agenda_gen*.yaml")),
    }


def _collect_canonical_surfaces(run_dir: Path) -> dict[str, Any]:
    findings, _ = read_jsonl(run_dir / "findings" / "findings.jsonl")
    frontier, _ = read_jsonl(run_dir / "findings" / "frontier.jsonl")
    memory, _ = read_jsonl(run_dir / "memory" / "research_memory.jsonl")
    graph_edges, _ = read_jsonl(run_dir / "memory" / "graph_edges.jsonl")
    budget, _ = read_jsonl(run_dir / "budget_ledger.jsonl")
    artifacts, _ = read_jsonl(run_dir / "artifact_index.jsonl")
    return {
        "findings": findings,
        "finding_ids": {str(item.get("finding_id")) for item in findings if item.get("finding_id")},
        "frontier": frontier,
        "frontier_ids": {
            str(item.get("finding_id")) for item in frontier if item.get("finding_id")
        },
        "research_memory": memory,
        "graph_edges": graph_edges,
        "budget": budget,
        "artifacts": artifacts,
        "artifact_types": {str(item.get("artifact_type")) for item in artifacts},
        "artifact_logical_paths": {str(item.get("logical_path")) for item in artifacts},
    }


def _check_replay(report: dict[str, Any]) -> ParityCheck:
    if report.get("success"):
        return _pass("replay_verify", "replay verify passed")
    return _fail(
        "replay_verify",
        "replay verify failed before parity checks",
        details={"errors": report.get("errors", []), "warnings": report.get("warnings", [])},
    )


def _check_task_ref(run_dir: Path) -> ParityCheck:
    run_json = _read_json(run_dir / "run.json")
    task_ref = str(run_json.get("task_ref") or "")
    if task_ref.startswith("task:"):
        return _pass("task_ref", f"run has task ref {task_ref}")
    return _warn("task_ref", f"run task_ref is missing or malformed: {task_ref}")


def _check_legacy_findings_materialized(
    legacy: dict[str, Any], canonical: dict[str, Any]
) -> ParityCheck:
    if not legacy["finding_ids"]:
        return _warn("legacy_findings_materialized", "no legacy findings found to compare")
    missing = sorted(legacy["finding_ids"] - canonical["finding_ids"])
    if missing:
        return _fail(
            "legacy_findings_materialized",
            "legacy findings missing from canonical findings ledger",
            details={"missing_finding_ids": missing},
        )
    return _pass(
        "legacy_findings_materialized",
        "all legacy findings are present in canonical findings ledger",
        details={"count": len(legacy["finding_ids"])},
    )


def _check_frontier_materialized(legacy: dict[str, Any], canonical: dict[str, Any]) -> ParityCheck:
    if not legacy["frontier_ids"]:
        return _warn("frontier_materialized", "no legacy frontier entries found to compare")
    missing = sorted(legacy["frontier_ids"] - canonical["frontier_ids"])
    if missing:
        return _fail(
            "frontier_materialized",
            "legacy frontier entries missing from canonical frontier ledger",
            details={"missing_finding_ids": missing},
        )
    return _pass(
        "frontier_materialized",
        "legacy frontier entries are present in canonical frontier ledger",
        details={"count": len(legacy["frontier_ids"])},
    )


def _check_research_memory_materialized(
    legacy: dict[str, Any], canonical: dict[str, Any]
) -> ParityCheck:
    legacy_count = int(legacy["research_memory_entry_count"])
    if legacy_count == 0:
        return _warn("research_memory_materialized", "no legacy research-memory entries found")
    canonical_count = len(canonical["research_memory"])
    if canonical_count < legacy_count:
        return _fail(
            "research_memory_materialized",
            "canonical research-memory ledger has fewer records than legacy ledgers",
            details={"legacy": legacy_count, "canonical": canonical_count},
        )
    return _pass(
        "research_memory_materialized",
        "legacy research-memory entries are materialized",
        details={"legacy": legacy_count, "canonical": canonical_count},
    )


def _check_graph_edges_materialized(
    legacy: dict[str, Any], canonical: dict[str, Any]
) -> ParityCheck:
    legacy_count = int(legacy["graph_edge_count"])
    if legacy_count == 0:
        return _warn("graph_edges_materialized", "no legacy graph edges found")
    canonical_count = len(canonical["graph_edges"])
    if canonical_count < legacy_count:
        return _fail(
            "graph_edges_materialized",
            "canonical graph edge ledger has fewer records than legacy graph",
            details={"legacy": legacy_count, "canonical": canonical_count},
        )
    return _pass(
        "graph_edges_materialized",
        "legacy graph edges are materialized",
        details={"legacy": legacy_count, "canonical": canonical_count},
    )


def _check_graph_artifacts_materialized(
    legacy: dict[str, Any], canonical: dict[str, Any]
) -> ParityCheck:
    names = set(legacy["graph_artifact_names"])
    if not names:
        return _warn("graph_artifacts_materialized", "no legacy graph artifacts found")
    if "graph_materialized_artifact" not in canonical["artifact_types"]:
        return _fail("graph_artifacts_materialized", "legacy graph artifacts are not indexed")
    indexed = {
        Path(path).name
        for path in canonical["artifact_logical_paths"]
        if path.startswith("graph/legacy/")
    }
    missing = sorted(names - indexed)
    if missing:
        return _fail(
            "graph_artifacts_materialized",
            "some legacy graph artifacts are missing from artifact index",
            details={"missing": missing},
        )
    return _pass(
        "graph_artifacts_materialized",
        "legacy graph artifacts are indexed",
        details={"count": len(names)},
    )


def _check_prompt_guidance_surfaces(
    legacy: dict[str, Any],
    canonical: dict[str, Any],
    *,
    strict: bool,
) -> ParityCheck:
    postgen_prompts: list[Path] = legacy["postgen_prompt_paths"]
    if not postgen_prompts:
        return _warn_or_fail(
            "prompt_guidance_surfaces",
            "no post-generation peer prompts found; cannot verify graph/frontier prompt parity",
            strict,
        )
    needs_graph = int(legacy["graph_edge_count"]) > 0 or bool(canonical["graph_edges"])
    needs_frontier = bool(legacy["frontier_ids"] or canonical["frontier_ids"])
    graph_hits: list[str] = []
    frontier_hits: list[str] = []
    unlinked_tool_hits: list[str] = []
    for path in postgen_prompts:
        text = legacy["prompt_texts"].get(path, "")
        if "Graph-surfaced context" in text:
            graph_hits.append(str(path))
        if "frontier" in text.lower() and any(
            fid in text for fid in legacy["frontier_ids"] | canonical["frontier_ids"]
        ):
            frontier_hits.append(str(path))
        if "mcp__finding-graph-query__get_unlinked_recent_findings" in text:
            unlinked_tool_hits.append(str(path))
    missing = []
    if needs_graph and not graph_hits:
        missing.append("Graph-surfaced context")
    if needs_frontier and not frontier_hits:
        missing.append("frontier finding ids")
    if not unlinked_tool_hits:
        missing.append("unlinked-recent graph tool reminder")
    if missing:
        return _fail(
            "prompt_guidance_surfaces",
            "post-generation prompts are missing active guidance surfaces",
            details={"missing": missing, "prompt_count": len(postgen_prompts)},
        )
    return _pass(
        "prompt_guidance_surfaces",
        "post-generation prompts expose graph, frontier, and diversity guidance",
        details={
            "graph_prompt_count": len(graph_hits),
            "frontier_prompt_count": len(frontier_hits),
            "unlinked_tool_prompt_count": len(unlinked_tool_hits),
        },
    )


def _check_panel_agenda_surface(legacy: dict[str, Any], *, strict: bool) -> ParityCheck:
    agenda_paths: list[Path] = legacy["agenda_paths"]
    if not agenda_paths:
        return _warn_or_fail("panel_agenda_surface", "no PI agenda files found", strict)
    invalid: list[str] = []
    for path in agenda_paths:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            invalid.append(str(path))
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("peer_contracts"), dict):
            invalid.append(str(path))
    if invalid:
        return _fail(
            "panel_agenda_surface",
            "PI agenda files are not structurally valid",
            details={"invalid": invalid},
        )
    return _pass(
        "panel_agenda_surface",
        "PI agenda files are present and structurally valid",
        details={"count": len(agenda_paths)},
    )


def _check_operator_status_surface(
    run_dir: Path,
    legacy: dict[str, Any],
    canonical: dict[str, Any],
    *,
    strict: bool,
) -> ParityCheck:
    status = _read_json(run_dir / "orchestrator_status.final.json")
    if not status:
        return _warn_or_fail(
            "operator_status_surface", "missing orchestrator_status.final.json", strict
        )
    required = {
        "run_dir",
        "task_id",
        "task_name",
        "current_generation",
        "generations_completed",
        "findings_total",
        "frontier_candidates",
        "exit_condition",
    }
    missing = sorted(required - set(status))
    if missing:
        return _fail(
            "operator_status_surface",
            "operator final status is missing required fields",
            details={"missing": missing},
        )
    details = {
        "findings_total": status.get("findings_total"),
        "canonical_findings": len(canonical["finding_ids"]),
        "frontier_candidates": status.get("frontier_candidates"),
        "canonical_frontier": len(canonical["frontier_ids"]),
        "legacy_frontier": len(legacy["frontier_ids"]),
    }
    if int(status.get("frontier_candidates") or 0) < len(canonical["frontier_ids"]):
        return _warn_or_fail(
            "operator_status_surface",
            "operator status reports fewer frontier candidates than canonical ledger",
            strict,
            details=details,
        )
    return _pass(
        "operator_status_surface",
        "operator final status contains the key dogfood fields",
        details=details,
    )


def _check_resource_guard_usage(canonical: dict[str, Any], *, strict: bool) -> ParityCheck:
    actions = {
        str(record.get("action_type"))
        for record in canonical["budget"]
        if record.get("kind") in {"usage", "usage_unknown"}
    }
    expected_any = {"eval_runner", "gpu_slot", "tool.wait_for_file"}
    if actions & expected_any:
        return _pass(
            "resource_guard_usage",
            "resource guard usage records are present",
            details={"action_types": sorted(actions)},
        )
    return _warn_or_fail(
        "resource_guard_usage",
        "no eval/GPU/tool resource usage records found",
        strict,
        details={"action_types": sorted(actions)},
    )


def _check_deliverables(deliverables_dir: Path | None, *, strict: bool) -> ParityCheck:
    if deliverables_dir is None:
        return _warn_or_fail(
            "deliverables_package", "deliverables package not provided for parity check", strict
        )
    required = {
        "README.md",
        "executive_summary.md",
        "data/run_summary.json",
        "data/frontier_manifest.json",
    }
    missing = sorted(rel for rel in required if not (deliverables_dir / rel).exists())
    if missing:
        return _fail(
            "deliverables_package",
            "deliverables package is missing required files",
            details={"missing": missing},
        )
    return _pass(
        "deliverables_package", "deliverables package contains required human/export surfaces"
    )


def _legacy_findings(run_dir: Path) -> dict[str, dict[str, Any]]:
    findings: dict[str, dict[str, Any]] = {}
    for path in sorted((run_dir / "shared_findings").glob("*.json")):
        payload = _read_json(path)
        finding_id = str(payload.get("id") or payload.get("finding_id") or "")
        if finding_id:
            findings[finding_id] = payload
    for row in _sqlite_rows(run_dir, "SELECT * FROM findings"):
        finding_id = str(row.get("id") or "")
        if finding_id:
            findings[finding_id] = row
    return findings


def _legacy_frontier_ids(run_dir: Path) -> set[str]:
    manifest = _read_json(run_dir / "frontier" / "frontier_manifest.json")
    ids: set[str] = set()
    for entry in _frontier_entries(manifest):
        finding_id = str(entry.get("finding_id") or entry.get("id") or "")
        if finding_id:
            ids.add(finding_id)
    return ids


def _frontier_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    cumulative = manifest.get("cumulative_top")
    if isinstance(cumulative, list):
        entries.extend(item for item in cumulative if isinstance(item, dict))
    generations = manifest.get("generations")
    if isinstance(generations, dict):
        for value in generations.values():
            if isinstance(value, list):
                entries.extend(item for item in value if isinstance(item, dict))
    return entries


def _legacy_research_memory_entry_count(run_dir: Path) -> int:
    total = 0
    for path in sorted((run_dir / "research_memory" / "ledgers").glob("*.yaml")):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if isinstance(entries, list):
            total += len([item for item in entries if isinstance(item, dict)])
    return total


def _legacy_graph_edges(run_dir: Path) -> list[dict[str, Any]]:
    return _sqlite_rows(
        run_dir,
        """SELECT edge_id, src_finding_id, dst_finding_id, edge_type,
                  confidence, created_by, created_at, rationale, provenance
           FROM finding_edges
           ORDER BY created_at ASC, edge_id ASC""",
    )


def _legacy_graph_artifact_names(run_dir: Path) -> set[str]:
    graph_dir = run_dir / "graph"
    return {name for name in GRAPH_ARTIFACT_NAMES if (graph_dir / name).is_file()}


def _prompt_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("gen_*/*_prompt.md"))


def _prompt_generation(path: Path) -> int | None:
    try:
        dirname = path.parent.name
        if dirname.startswith("gen_"):
            return int(dirname.split("_", 1)[1])
    except (IndexError, ValueError):
        return None
    return None


def _sqlite_rows(run_dir: Path, query: str) -> list[dict[str, Any]]:
    db_path = run_dir / "shared_store.db"
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(row) for row in conn.execute(query).fetchall()]
        finally:
            conn.close()
    except sqlite3.Error:
        return []


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _pass(check_id: str, message: str, *, details: dict[str, Any] | None = None) -> ParityCheck:
    return ParityCheck(
        check_id=check_id, status="pass", severity="info", message=message, details=details or {}
    )


def _fail(check_id: str, message: str, *, details: dict[str, Any] | None = None) -> ParityCheck:
    return ParityCheck(
        check_id=check_id, status="fail", severity="error", message=message, details=details or {}
    )


def _warn(check_id: str, message: str, *, details: dict[str, Any] | None = None) -> ParityCheck:
    return ParityCheck(
        check_id=check_id, status="warn", severity="warning", message=message, details=details or {}
    )


def _warn_or_fail(
    check_id: str,
    message: str,
    strict: bool,
    *,
    details: dict[str, Any] | None = None,
) -> ParityCheck:
    return (
        _fail(check_id, message, details=details)
        if strict
        else _warn(check_id, message, details=details)
    )
