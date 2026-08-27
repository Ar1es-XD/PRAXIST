"""Human-readable Praxist run reports.

The report generator is intentionally deterministic and read-only over run
facts. It summarizes frontier/Pareto outcomes, lineage hints, and health
signals without changing promotion, frontier, PI, or guard behavior.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from praxist.plugins.workflow_stages.research_loop.backend.effective_config import (
    result_effective_config_metadata,
    strip_effective_config_fields,
)
from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
    durable_promotion_exclusion,
    evidence_maturity_snapshot,
    result_artifact_key,
    same_result_artifact,
)
from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
    _is_bad_result_status,
    _result_summary_metrics,
    iter_result_summary_paths,
    normalized_result_summary,
    result_summary_control_digest,
    result_summary_variant_name,
)
from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
    _json_digest as _json_digest,
)
from praxist.plugins.workflow_stages.research_loop.backend.orchestrator_status import (
    read_effective_orchestrator_status,
)
from praxist.task_spec import load_task_spec

REPORT_STATE_REL = Path("reports") / "human_report_state.json"
logger = logging.getLogger(__name__)

_VALID_METRIC_DIRECTIONS = frozenset({"maximize", "minimize"})
_REPORT_EXCLUSION_BOOL_KEYS = (
    "suspect",
    "suspect_protocol",
    "protocol_integrity_failed",
)


@dataclass(frozen=True)
class ReportWriteResult:
    """Result returned after writing a human-readable report."""

    path: Path
    trigger: str
    generation_id: int | None
    report_kind: str
    pdf_path: Path | None = None


@dataclass(frozen=True)
class _TaskMetricRegistry:
    """Task-owned metric directions and equivalent result metric names."""

    directions: dict[str, str]
    primary_metric: str | None
    aliases: dict[str, str]
    declared_metrics: frozenset[str]


def maybe_generate_boundary_report(
    *,
    run_dir: Path,
    task_dir: Path | None,
    generation_id: int,
    final: bool = False,
) -> ReportWriteResult | None:
    """Generate a report if a boundary trigger is due.

    Triggers:
    - first promoted credible frontier signal after baseline filtering;
    - every three completed generations;
    - final run completion.

    Failures are meant to be caught by callers and must not block the run.
    """

    run_dir = Path(run_dir)
    state = _read_json(run_dir / REPORT_STATE_REL, default={})
    trigger = _due_trigger(
        run_dir=run_dir,
        task_dir=Path(task_dir) if task_dir is not None else None,
        generation_id=generation_id,
        state=state,
        final=final,
    )
    if trigger is None:
        return None
    result = generate_run_report(
        run_dir=run_dir,
        task_dir=task_dir,
        report_kind="frontier_lineage_health",
        trigger=trigger,
        generation_id=generation_id,
    )
    state = dict(state)
    state.setdefault("generated_reports", []).append(
        {
            "path": str(result.path),
            "trigger": trigger,
            "generation_id": generation_id,
            "created_at": _utc_timestamp(),
        }
    )
    if trigger == "first_credible_baseline_beat":
        state["first_credible_baseline_beat_reported"] = True
    if trigger == "periodic_3_generation":
        state.setdefault("periodic_generations_reported", []).append(generation_id)
    if final:
        state["final_reported"] = True
    _atomic_write_json(run_dir / REPORT_STATE_REL, state)
    return result


def generate_run_report(
    *,
    run_dir: Path,
    task_dir: Path | None = None,
    report_kind: str = "frontier_lineage_health",
    trigger: str = "manual",
    generation_id: int | None = None,
) -> ReportWriteResult:
    """Generate and persist a Markdown report for a run.

    Reports are written to ``<task_dir>/docs/praxist_reports`` when a task directory
    is available, and otherwise to ``<run_dir>/docs/praxist_reports``. A copy is not
    written elsewhere to avoid multiplying fact surfaces.
    """

    run_dir = Path(run_dir)
    docs_root = _report_docs_root(run_dir=run_dir, task_dir=task_dir)
    docs_root.mkdir(parents=True, exist_ok=True)

    payload = collect_report_payload(run_dir=run_dir, task_dir=task_dir)
    run_name = _safe_slug(run_dir.name) or "run"
    gen_token = f"_gen{generation_id}" if generation_id is not None and generation_id >= 0 else ""
    filename = f"{_timestamp_for_filename()}_{run_name}_{_safe_slug(trigger)}{gen_token}.md"
    path = docs_root / filename
    pdf_path = path.with_suffix(".pdf")
    text = render_report_markdown(
        payload,
        run_dir=run_dir,
        trigger=trigger,
        generation_id=generation_id,
        report_kind=report_kind,
        companion_pdf_path=pdf_path,
    )
    path.write_text(text, encoding="utf-8")
    written_pdf_path: Path | None = None
    try:
        _write_pdf_report(
            path=pdf_path,
            payload=payload,
            run_dir=run_dir,
            trigger=trigger,
            generation_id=generation_id,
            report_kind=report_kind,
        )
        written_pdf_path = pdf_path
    except Exception as exc:  # noqa: BLE001 - PDF is a derived companion view.
        logger.warning("human run report PDF generation skipped: %s", exc)
    return ReportWriteResult(
        path=path,
        trigger=trigger,
        generation_id=generation_id,
        report_kind=report_kind,
        pdf_path=written_pdf_path,
    )


def generate_boundary_report_safely(
    *,
    run_dir: Path,
    task_dir: Path | None,
    generation_id: int,
    final: bool = False,
) -> ReportWriteResult | None:
    """Best-effort boundary report hook for the research loop."""

    try:
        result = maybe_generate_boundary_report(
            run_dir=run_dir,
            task_dir=task_dir,
            generation_id=generation_id,
            final=final,
        )
    except Exception as exc:  # noqa: BLE001 - report generation is advisory.
        logger.warning("human run report generation skipped: %s", exc)
        return None
    if result is not None:
        logger.info(
            "human run report generated (%s, gen=%s): %s",
            result.trigger,
            result.generation_id,
            result.path,
        )
    return result


def generate_loop_boundary_report(loop: Any, *, generation_id: int, final: bool = False) -> None:
    """Best-effort report hook for ``GenerationLoop`` without growing it."""

    generate_boundary_report_safely(
        run_dir=loop.run_dir,
        task_dir=getattr(loop.task_spec, "task_dir", None),
        generation_id=generation_id,
        final=final,
    )


def generate_terminal_report_safely(prepared: Any, summary: dict[str, Any]) -> None:
    """Cover terminal paths that do not return through ``GenerationLoop``."""

    completed = int(summary.get("generations_completed", 0) or 0)
    exit_condition = str(summary.get("exit_condition") or "").strip().lower()
    if completed <= 0 and exit_condition not in {
        "error",
        "failed",
        "interrupted",
        "stopped",
        "user_stop",
        "stop_requested",
    }:
        return
    task_dir = Path(prepared.task_project_path)
    generate_boundary_report_safely(
        run_dir=prepared.run_dir,
        task_dir=task_dir if task_dir.is_dir() else None,
        generation_id=max(-1, completed - 1),
        final=True,
    )


def collect_report_payload(*, run_dir: Path, task_dir: Path | None = None) -> dict[str, Any]:
    """Collect compact facts used by report rendering."""

    run_dir = Path(run_dir)
    frontier = _read_json(run_dir / "frontier" / "frontier_manifest.json", default={})
    run_summary = _read_json(run_dir / "run_summary.json", default={})
    orchestrator = read_effective_orchestrator_status(run_dir)
    gems = _read_json(run_dir / "gems" / "gems_state.json", default={})
    boundaries = _load_generation_boundaries(run_dir)
    findings = _load_shared_findings(run_dir, cap=300)
    metric_registry = _task_metric_registry(Path(task_dir) if task_dir is not None else None)
    frontier_entries = _collect_frontier_entries(frontier)
    validation_candidates = _collect_validation_candidates(frontier)
    shared_finding_entries = _collect_shared_finding_entries(findings)
    maturity_policy = _task_maturity_policy(Path(task_dir)) if task_dir is not None else None
    result_summary_entries = _collect_result_summary_entries(
        run_dir,
        cap=700,
        metric_aliases=metric_registry.aliases,
        maturity_policy=maturity_policy,
    )
    signal_entries = _dedupe_entries(
        result_summary_entries + frontier_entries + shared_finding_entries + validation_candidates
    )
    baseline_values = (
        _baseline_metric_values(Path(task_dir), metric_aliases=metric_registry.aliases)
        if task_dir is not None
        else {}
    )
    credible_entries = [
        entry
        for entry in signal_entries
        if _report_evidence_class(entry, maturity_policy=maturity_policy) == "credible"
    ]
    non_credible_entries = [entry for entry in signal_entries if entry not in credible_entries]
    dimension_winners = _collect_dimension_winners(
        credible_entries,
        baseline_values=baseline_values,
        metric_registry=metric_registry,
    )
    signal_dimension_winners = _collect_dimension_winners(
        non_credible_entries,
        baseline_values=baseline_values,
        metric_registry=metric_registry,
    )
    for winner in signal_dimension_winners:
        entry = winner.get("entry") if isinstance(winner.get("entry"), dict) else {}
        winner["block_reason"] = _report_block_reason(
            entry,
            maturity_policy=maturity_policy,
        )
    credible_frontier_entries = [
        entry
        for entry in frontier_entries
        if _report_evidence_class(entry, maturity_policy=maturity_policy) == "credible"
    ]
    ranked_frontier_entries = _rank_entries(
        credible_frontier_entries,
        metric_registry=metric_registry,
    )
    top_entries = ranked_frontier_entries[:12]
    unranked_frontier_entries = [
        entry
        for entry in credible_frontier_entries
        if not _entry_has_known_ranking_direction(entry, metric_registry=metric_registry)
    ][:12]
    strong_entries = top_entries or _entries_from_dimension_winners(dimension_winners)
    return {
        "frontier": frontier,
        "run_summary": run_summary,
        "orchestrator_status": orchestrator,
        "gems": gems,
        "generation_boundaries": boundaries,
        "shared_findings": findings,
        "frontier_entries": frontier_entries,
        "top_entries": top_entries,
        "unranked_frontier_entries": unranked_frontier_entries,
        "strong_entries": strong_entries,
        "validation_candidates": validation_candidates,
        "signal_entries": signal_entries,
        "dimension_winners": dimension_winners,
        "signal_dimension_winners": signal_dimension_winners,
        "unknown_direction_metrics": _unknown_direction_metric_names(
            signal_entries,
            metric_registry=metric_registry,
        ),
        "primary_metric": metric_registry.primary_metric,
        "metric_direction_registry": dict(metric_registry.directions),
        "charts": _build_report_charts(
            signal_entries=credible_entries or signal_entries,
            winners=dimension_winners or signal_dimension_winners,
            credible=bool(dimension_winners),
            metric_registry=metric_registry,
        ),
    }


def render_report_markdown(
    payload: dict[str, Any],
    *,
    run_dir: Path,
    trigger: str,
    generation_id: int | None,
    report_kind: str,
    companion_pdf_path: Path | None = None,
) -> str:
    """Render a compact human-readable report."""

    run_summary = payload.get("run_summary") if isinstance(payload.get("run_summary"), dict) else {}
    orchestrator = (
        payload.get("orchestrator_status")
        if isinstance(payload.get("orchestrator_status"), dict)
        else {}
    )
    top_entries = payload.get("top_entries") if isinstance(payload.get("top_entries"), list) else []
    unranked_frontier_entries = (
        payload.get("unranked_frontier_entries")
        if isinstance(payload.get("unranked_frontier_entries"), list)
        else []
    )
    strong_entries = (
        payload.get("strong_entries") if isinstance(payload.get("strong_entries"), list) else []
    )
    dimension_winners = (
        payload.get("dimension_winners")
        if isinstance(payload.get("dimension_winners"), list)
        else []
    )
    signal_dimension_winners = (
        payload.get("signal_dimension_winners")
        if isinstance(payload.get("signal_dimension_winners"), list)
        else []
    )
    unknown_direction_metrics = (
        payload.get("unknown_direction_metrics")
        if isinstance(payload.get("unknown_direction_metrics"), list)
        else []
    )
    charts = payload.get("charts") if isinstance(payload.get("charts"), list) else []
    validation_candidates = (
        payload.get("validation_candidates")
        if isinstance(payload.get("validation_candidates"), list)
        else []
    )
    boundaries = (
        payload.get("generation_boundaries")
        if isinstance(payload.get("generation_boundaries"), list)
        else []
    )
    primary_metric = str(payload.get("primary_metric") or "").strip() or None
    generated_at = _utc_timestamp()
    lines: list[str] = []
    lines.append(f"# Praxist Run Report: {run_dir.name}")
    lines.append("")
    lines.append(f"- Generated at: `{generated_at}`")
    lines.append(f"- Trigger: `{trigger}`")
    if generation_id is not None:
        lines.append(f"- Generation context: `{generation_id}`")
    lines.append(f"- Report kind: `{report_kind}`")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append("")

    lines.append("## A. Strongest Variants And Pareto Front")
    lines.append("")
    if top_entries:
        lines.append("### Clean Frontier / Pareto Entries")
        lines.append("")
        lines.append(
            "| Rank | Variant | Gen | Lane | Metric | Value | Stage | Principle / Evidence |"
        )
        lines.append("| --- | --- | ---: | --- | --- | ---: | --- | --- |")
        for idx, entry in enumerate(top_entries, start=1):
            metric_name, metric_value = _best_metric(
                entry,
                primary_metric=primary_metric,
            )
            lines.append(
                "| {rank} | `{variant}` | {gen} | {lane} | `{metric}` | {value} | {stage} | {summary} |".format(
                    rank=idx,
                    variant=_escape_table(_variant_name(entry)),
                    gen=_display(_generation_id(entry)),
                    lane=_escape_table(_lane(entry)),
                    metric=_escape_table(metric_name or ""),
                    value=_format_float(metric_value),
                    stage=_escape_table(_evidence_stage(entry)),
                    summary=_escape_table(_entry_summary(entry, limit=140)),
                )
            )
    elif unranked_frontier_entries:
        lines.append(
            "Clean frontier entries were found, but none were ranked because their "
            "metric direction is unknown."
        )
    else:
        lines.append(
            "No clean frontier or Pareto-front entries were found yet. The report is "
            "separating mature measurements from broader "
            "validation, shared-finding, Gems, and result-summary signals."
        )
    if unranked_frontier_entries:
        lines.append("")
        lines.append("### Unranked Frontier Entries (Direction Unknown)")
        lines.append("")
        lines.append("| Variant | Gen | Lane | Metric | Value | Stage |")
        lines.append("| --- | ---: | --- | --- | ---: | --- |")
        for entry in unranked_frontier_entries:
            metric_name, metric_value = _best_metric(
                entry,
                primary_metric=primary_metric,
            )
            lines.append(
                "| `{variant}` | {gen} | {lane} | `{metric}` | {value} | {stage} |".format(
                    variant=_escape_table(_variant_name(entry)),
                    gen=_display(_generation_id(entry)),
                    lane=_escape_table(_lane(entry)),
                    metric=_escape_table(metric_name or ""),
                    value=_format_float(metric_value),
                    stage=_escape_table(_evidence_stage(entry)),
                )
            )
    if dimension_winners:
        lines.append("")
        lines.append("### Mature Dimension Winners")
        lines.append("")
        lines.append(
            "| Metric | Direction | Best Variant | Gen | Value | Baseline Relation | Promotion | Evidence | Source |"
        )
        lines.append("| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |")
        for winner in dimension_winners:
            entry = winner.get("entry") if isinstance(winner.get("entry"), dict) else {}
            lines.append(
                "| `{metric}` | {direction} | `{variant}` | {gen} | {value} | {baseline} | {promotion} | {evidence} | {source} |".format(
                    metric=_escape_table(str(winner.get("metric") or "")),
                    direction=_escape_table(str(winner.get("direction") or "")),
                    variant=_escape_table(_variant_name(entry)),
                    gen=_display(_generation_id(entry)),
                    value=_format_float(_number(winner.get("value"))),
                    baseline=_escape_table(str(winner.get("baseline_relation") or "unknown")),
                    promotion=_escape_table(_report_promotion_status(entry)),
                    evidence=_escape_table(_entry_signal_label(entry)),
                    source=_escape_table(_source_ref(entry)),
                )
            )
    if signal_dimension_winners:
        lines.append("")
        lines.append("### Strong Signals Requiring Validation")
        lines.append("")
        lines.append("These measurements remain visible but are not presented as mature evidence.")
        lines.append("")
        lines.append(
            "| Metric | Direction | Best Signal | Gen | Value | Evidence | Blocker / Status | Source |"
        )
        lines.append("| --- | --- | --- | ---: | ---: | --- | --- | --- |")
        for winner in signal_dimension_winners:
            entry = winner.get("entry") if isinstance(winner.get("entry"), dict) else {}
            lines.append(
                "| `{metric}` | {direction} | `{variant}` | {gen} | {value} | {evidence} | {blocker} | {source} |".format(
                    metric=_escape_table(str(winner.get("metric") or "")),
                    direction=_escape_table(str(winner.get("direction") or "")),
                    variant=_escape_table(_variant_name(entry)),
                    gen=_display(_generation_id(entry)),
                    value=_format_float(_number(winner.get("value"))),
                    evidence=_escape_table(_entry_signal_label(entry)),
                    blocker=_escape_table(str(winner.get("block_reason") or "validation required")),
                    source=_escape_table(_source_ref(entry)),
                )
            )
    elif not dimension_winners and not top_entries and not unranked_frontier_entries:
        lines.append("No numeric validation or result-summary signals were found yet.")
    if unknown_direction_metrics:
        omitted = ", ".join(f"`{name}`" for name in unknown_direction_metrics[:16])
        suffix = " and others" if len(unknown_direction_metrics) > 16 else ""
        lines.append("")
        lines.append(
            "Comparative claims and charts were omitted for metrics with unknown or "
            f"conflicting direction: {omitted}{suffix}."
        )
    lines.append("")

    lines.append("## B. Strong-Variant Evolution And Lineage")
    lines.append("")
    lineage_rows = [_lineage_row(entry) for entry in strong_entries]
    if any(row["parent"] or row["parent_usage"] or row["source"] for row in lineage_rows):
        if any(row["effective_config"] for row in lineage_rows):
            lines.append(
                "| Variant | Gen | Parent | Result Source | Parent Usage | Effective Config | Development Note |"
            )
            lines.append("| --- | ---: | --- | --- | --- | --- | --- |")
            for row in lineage_rows:
                lines.append(
                    "| `{variant}` | {gen} | {parent} | {source} | {usage} | {config} | {note} |".format(
                        variant=_escape_table(row["variant"]),
                        gen=_display(row["gen"]),
                        parent=_escape_table(row["parent"] or "not declared"),
                        source=_escape_table(row["source"] or "not declared"),
                        usage=_escape_table(row["parent_usage"] or "not declared"),
                        config=_escape_table(row["effective_config"] or "not declared"),
                        note=_escape_table(row["note"]),
                    )
                )
        else:
            lines.append("| Variant | Gen | Parent / Source | Parent Usage | Development Note |")
            lines.append("| --- | ---: | --- | --- | --- |")
            for row in lineage_rows:
                lines.append(
                    "| `{variant}` | {gen} | {parent} | {usage} | {note} |".format(
                        variant=_escape_table(row["variant"]),
                        gen=_display(row["gen"]),
                        parent=_escape_table(row["parent"] or row["source"] or "not declared"),
                        usage=_escape_table(row["parent_usage"] or "not declared"),
                        note=_escape_table(row["note"]),
                    )
                )
    elif strong_entries:
        lines.append(
            "The strongest entries do not expose structured parent lineage fields. "
            "Future tasks should publish `parent_candidate`, `parent_usage`, and "
            "`source_result_path` in finding `extra` or metrics."
        )
    else:
        lines.append("No lineage can be inferred from the available result signals yet.")
    if charts:
        lines.append("")
        lines.append("## Visual Companion")
        lines.append("")
        if companion_pdf_path is not None:
            lines.append(f"- PDF report with charts: `{companion_pdf_path}`")
        for chart in charts:
            title = chart.get("title") if isinstance(chart, dict) else None
            if title:
                lines.append(f"- Chart: {_clip(str(title), 140)}")
    lines.append("")

    lines.append("## C. Run Health And Evidence State")
    lines.append("")
    health = _health_summary(run_summary=run_summary, orchestrator=orchestrator)
    for item in health:
        lines.append(f"- {item}")
    if validation_candidates:
        lines.append(
            f"- Validation candidates retained for follow-up: `{len(validation_candidates)}` "
            "(partial/scout/repair/late signals are preserved separately from clean frontier truth)."
        )
    if boundaries:
        latest = boundaries[-1]
        semantics = latest.get("artifact_semantics")
        boundary_status = latest.get("status")
        if not boundary_status and isinstance(semantics, dict):
            boundary_status = semantics.get("status")
        lines.append(
            "- Latest generation boundary: "
            f"`gen_{latest.get('generation_id', '?')}` status `{boundary_status or 'unknown'}`."
        )
    lines.append("")

    lines.append("## Report Semantics")
    lines.append("")
    lines.append(
        "This Markdown report is a derived human-readable view. Canonical facts "
        "remain in frontier, findings, result summaries, Gems state, and generation "
        "boundary artifacts. Do not hand-edit this report to change run truth."
    )
    if companion_pdf_path is not None:
        lines.append(
            f"A companion PDF is written next to this Markdown file at `{companion_pdf_path}`."
        )
    lines.append("")
    return "\n".join(lines)


def _due_trigger(
    *,
    run_dir: Path,
    task_dir: Path | None,
    generation_id: int,
    state: dict[str, Any],
    final: bool,
) -> str | None:
    if final and not state.get("final_reported"):
        return "final_run_completion"
    if not state.get("first_credible_baseline_beat_reported") and _has_baseline_beat(
        run_dir=run_dir,
        task_dir=task_dir,
    ):
        return "first_credible_baseline_beat"
    completed = generation_id + 1
    reported = set(state.get("periodic_generations_reported") or [])
    if completed > 0 and completed % 3 == 0 and generation_id not in reported:
        return "periodic_3_generation"
    return None


def _has_baseline_beat(*, run_dir: Path, task_dir: Path | None) -> bool:
    manifest = _read_json(run_dir / "frontier" / "frontier_manifest.json", default={})
    entries = _collect_frontier_entries(manifest)
    if task_dir is None:
        return False
    metric_registry = _task_metric_registry(Path(task_dir))
    baselines = _baseline_metric_values(
        Path(task_dir),
        metric_aliases=metric_registry.aliases,
    )
    if not baselines:
        return False
    maturity_policy = _task_maturity_policy(Path(task_dir))
    if not entries:
        findings = _load_shared_findings(run_dir, cap=300)
        entries = _dedupe_entries(
            _collect_validation_candidates(manifest)
            + _collect_shared_finding_entries(findings)
            + _collect_result_summary_entries(
                run_dir,
                cap=700,
                metric_aliases=metric_registry.aliases,
                maturity_policy=maturity_policy,
            )
        )
    for entry in entries:
        if not _credible_for_report_trigger(entry, maturity_policy=maturity_policy):
            continue
        checked_metrics: set[str] = set()
        metric_name, metric_value = _best_metric(
            entry,
            primary_metric=metric_registry.primary_metric,
        )
        if metric_name is not None and metric_value is not None:
            checked_metrics.add(metric_name)
            if _beats_baseline(
                metric_name,
                metric_value,
                entry,
                baselines,
                metric_registry=metric_registry,
            ):
                return True
        for candidate_name, candidate_value in _entry_numeric_metrics(entry).items():
            if candidate_name in checked_metrics:
                continue
            if _beats_baseline(
                candidate_name,
                candidate_value,
                entry,
                baselines,
                metric_registry=metric_registry,
            ):
                return True
    return False


def _beats_baseline(
    metric_name: str,
    metric_value: float,
    entry: dict[str, Any],
    baselines: dict[str, list[float]],
    *,
    metric_registry: _TaskMetricRegistry | dict[str, str] | None = None,
) -> bool:
    baseline_values = baselines.get(metric_name) or []
    if not baseline_values:
        return False
    direction = _metric_direction_for_name(
        metric_name,
        entry,
        metric_registry=metric_registry,
    )
    if direction == "minimize":
        return metric_value < min(baseline_values)
    if direction == "maximize":
        return metric_value > max(baseline_values)
    return False


def _credible_for_report_trigger(
    entry: dict[str, Any],
    *,
    maturity_policy: dict[str, Any] | None = None,
) -> bool:
    return (
        evidence_maturity_snapshot(entry, maturity_policy).get("mature_enough") is True
        and durable_promotion_exclusion(entry) is None
        and not _entry_has_report_exclusion(entry)
    )


def _load_report_task_spec(task_dir: Path) -> Any | None:
    for name in ("task.yaml", "task_spec.yaml"):
        path = Path(task_dir) / name
        if not path.is_file():
            continue
        try:
            return load_task_spec(path)
        except Exception as exc:  # noqa: BLE001 - reports tolerate malformed task metadata.
            logger.warning("run report could not load task spec from %s: %s", path, exc)
            return None
    return None


def _task_maturity_policy(task_dir: Path) -> dict[str, Any] | None:
    task_spec = _load_report_task_spec(task_dir)
    if task_spec is None:
        return None
    return dict(task_spec.evaluation.maturity_policy)


def _task_metric_registry(task_dir: Path | None) -> _TaskMetricRegistry:
    if task_dir is None:
        return _TaskMetricRegistry(
            directions={},
            primary_metric=None,
            aliases={},
            declared_metrics=frozenset(),
        )
    task_spec = _load_report_task_spec(Path(task_dir))
    if task_spec is None:
        return _TaskMetricRegistry(
            directions={},
            primary_metric=None,
            aliases={},
            declared_metrics=frozenset(),
        )

    declared: dict[str, set[str]] = {}

    def declare(metric_name: Any, direction: Any) -> None:
        name = str(metric_name or "").strip()
        normalized = str(direction or "").strip().lower()
        if name and normalized in _VALID_METRIC_DIRECTIONS:
            declared.setdefault(name, set()).add(normalized)

    evaluation = task_spec.evaluation
    primary_metric = str(getattr(evaluation, "primary_metric", "") or "").strip() or None
    if primary_metric is not None:
        declare(primary_metric, getattr(evaluation, "direction", ""))
    for metric_name, direction in getattr(evaluation, "anchor_metrics", []) or []:
        declare(metric_name, direction)
    for lane in getattr(evaluation, "frontier_lanes", []) or []:
        if not isinstance(lane, dict):
            continue
        for field_name in ("axes", "optional_axes"):
            for metric_name, direction in lane.get(field_name, []) or []:
                declare(metric_name, direction)
    for baseline in getattr(task_spec, "baselines", []) or []:
        declare(getattr(baseline, "metric_name", ""), getattr(baseline, "direction", ""))

    raw_aliases = getattr(getattr(task_spec, "gems", None), "result_metric_aliases", {}) or {}
    aliases = {
        str(out_name).strip(): str(source_name).strip()
        for out_name, source_name in raw_aliases.items()
        if str(out_name or "").strip() and str(source_name or "").strip()
    }
    alias_graph: dict[str, set[str]] = {}
    for out_name, source_name in aliases.items():
        alias_graph.setdefault(out_name, set()).add(source_name)
        alias_graph.setdefault(source_name, set()).add(out_name)

    directions: dict[str, str] = {}
    all_names = set(declared) | set(alias_graph)
    visited: set[str] = set()
    for start in sorted(all_names):
        if start in visited:
            continue
        component: set[str] = set()
        pending = [start]
        while pending:
            name = pending.pop()
            if name in component:
                continue
            component.add(name)
            pending.extend(alias_graph.get(name, ()))
        visited.update(component)
        component_directions = {
            direction for name in component for direction in declared.get(name, set())
        }
        if len(component_directions) == 1:
            direction = next(iter(component_directions))
            directions.update({name: direction for name in component})
        elif len(component_directions) > 1:
            logger.warning(
                "run report omitted conflicting task metric directions for %s",
                ", ".join(sorted(component)),
            )

    return _TaskMetricRegistry(
        directions=directions,
        primary_metric=primary_metric,
        aliases=aliases,
        declared_metrics=frozenset(all_names),
    )


def _baseline_metric_values(
    task_dir: Path,
    *,
    metric_aliases: dict[str, str] | None = None,
) -> dict[str, list[float]]:
    """Read compact baseline ledgers and return best known value per metric."""

    baseline_dir = Path(task_dir) / "assets" / "baselines"
    if not baseline_dir.exists():
        return {}
    values: dict[str, list[float]] = {}
    for path in sorted(baseline_dir.glob("*.jsonl")):
        if path.name.startswith("._"):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                _collect_numeric_leaf_values(payload, values)
    if not metric_aliases:
        return values
    graph: dict[str, set[str]] = {}
    for out_name, source_name in metric_aliases.items():
        graph.setdefault(out_name, set()).add(source_name)
        graph.setdefault(source_name, set()).add(out_name)
    expanded = {name: list(metric_values) for name, metric_values in values.items()}
    visited: set[str] = set()
    for start in sorted(graph):
        if start in visited:
            continue
        component: set[str] = set()
        pending = [start]
        while pending:
            name = pending.pop()
            if name in component:
                continue
            component.add(name)
            pending.extend(graph.get(name, ()))
        visited.update(component)
        component_values = [value for name in component for value in values.get(name, [])]
        if component_values:
            for name in component:
                expanded[name] = list(component_values)
    return expanded


def _collect_numeric_leaf_values(payload: dict[str, Any], out: dict[str, list[float]]) -> None:
    for key, value in payload.items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, dict):
            mean = _number(value.get("mean"))
            if mean is not None:
                out.setdefault(str(key), []).append(mean)
            else:
                _collect_numeric_leaf_values(value, out)
            continue
        numeric = _number(value)
        if numeric is None:
            continue
        out.setdefault(str(key), []).append(numeric)


def _report_docs_root(*, run_dir: Path, task_dir: Path | None) -> Path:
    run_path = Path(run_dir).resolve()
    if task_dir is not None:
        task_path = Path(task_dir).resolve()
        try:
            if run_path == task_path or run_path.is_relative_to(task_path):
                return task_path / "docs" / "praxist_reports"
        except OSError:
            pass
    return run_path / "docs" / "praxist_reports"


def _collect_frontier_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    defaults = _manifest_metric_defaults(manifest)
    for key in ("cumulative_top", "frontier", "frontier_summary"):
        raw = manifest.get(key)
        if isinstance(raw, list):
            entries.extend(
                [_apply_entry_defaults(entry, defaults) for entry in raw if isinstance(entry, dict)]
            )
    lane_frontiers = manifest.get("lane_frontiers")
    if isinstance(lane_frontiers, dict):
        for lane_name, lane_payload in lane_frontiers.items():
            lane_entries: list[Any] = []
            if isinstance(lane_payload, list):
                lane_entries = lane_payload
            elif isinstance(lane_payload, dict):
                for key in ("cumulative", "top", "entries", "generations"):
                    raw = lane_payload.get(key)
                    if isinstance(raw, list):
                        lane_entries.extend(raw)
                    elif isinstance(raw, dict):
                        for gen_entries in raw.values():
                            if isinstance(gen_entries, list):
                                lane_entries.extend(gen_entries)
            for entry in lane_entries:
                if isinstance(entry, dict):
                    copied = _apply_entry_defaults(entry, defaults)
                    copied.setdefault("frontier_lane", lane_name)
                    entries.append(copied)
    return _dedupe_entries(entries)


def _manifest_metric_defaults(manifest: dict[str, Any]) -> dict[str, str]:
    metric_name = (
        manifest.get("metric_name")
        or manifest.get("primary_metric")
        or manifest.get("primary_metric_name")
    )
    metric_direction = manifest.get("metric_direction") or manifest.get("direction")
    defaults: dict[str, str] = {}
    if isinstance(metric_name, str) and metric_name.strip():
        defaults["metric_name"] = metric_name.strip()
    if str(metric_direction).lower() in {"maximize", "minimize"}:
        defaults["metric_direction"] = str(metric_direction).lower()
    return defaults


def _apply_entry_defaults(entry: dict[str, Any], defaults: dict[str, str]) -> dict[str, Any]:
    copied = dict(entry)
    for key, value in defaults.items():
        copied.setdefault(key, value)
    return copied


def _collect_validation_candidates(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw = manifest.get("validation_candidates")
    candidates: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        cumulative = raw.get("cumulative")
        if isinstance(cumulative, list):
            candidates.extend([entry for entry in cumulative if isinstance(entry, dict)])
        generations = raw.get("generations")
        if isinstance(generations, dict):
            for gen_entries in generations.values():
                if isinstance(gen_entries, list):
                    candidates.extend([entry for entry in gen_entries if isinstance(entry, dict)])
    elif isinstance(raw, list):
        candidates.extend([entry for entry in raw if isinstance(entry, dict)])
    out: list[dict[str, Any]] = []
    for entry in candidates:
        copied = dict(entry)
        copied.setdefault("report_signal_source", "validation_candidate")
        out.append(copied)
    return _dedupe_entries(out)


def _collect_shared_finding_entries(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for finding in findings:
        raw_metrics = finding.get("metrics")
        metrics = dict(raw_metrics) if isinstance(raw_metrics, dict) else {}
        strip_effective_config_fields(metrics)
        if not metrics:
            continue
        copied = {
            "variant_name": _finding_variant_name(finding),
            "metrics": metrics,
            "summary": finding.get("summary") or finding.get("content") or finding.get("title"),
            "title": finding.get("title"),
            "report_signal_source": "shared_finding",
        }
        for key in (
            "generation_id",
            "source_generation_id",
            "frontier_lane",
            "source_result_path",
            "parent_candidate",
            "parent_usage",
            "evidence_stage",
        ):
            if key in metrics:
                copied[key] = metrics[key]
        entries.append(copied)
    return _dedupe_entries(entries)


def _finding_variant_name(finding: dict[str, Any]) -> str:
    metrics = finding.get("metrics") if isinstance(finding.get("metrics"), dict) else {}
    for source in (metrics, finding):
        for key in (
            "variant_name",
            "variant_id",
            "frontier_entity_key",
            "candidate_entity_key",
            "id",
        ):
            value = source.get(key)
            if value:
                text = str(value)
                if text.startswith("variant::"):
                    return text.split("variant::", 1)[1]
                return text
    title = str(finding.get("title") or "")
    match = re.search(r"(gen\d+_peer\d+[_A-Za-z0-9_.-]*)", title)
    if match:
        return match.group(1)
    return str(finding.get("id") or "unknown_variant")


def _collect_result_summary_entries(
    run_dir: Path,
    *,
    cap: int,
    metric_aliases: dict[str, str] | None = None,
    maturity_policy: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in iter_result_summary_paths(run_dir):
        if len(entries) >= cap:
            break
        payload = _read_json(path, default={})
        if not isinstance(payload, dict):
            continue
        payload = normalized_result_summary(
            payload,
            summary_path=path,
            maturity_policy=maturity_policy,
        )
        reported_metric_sources = [payload]
        for source_name in ("current_aggregate", "metrics"):
            source = payload.get(source_name)
            if isinstance(source, dict):
                reported_metric_sources.append(source)
        reported_metric_names = {
            key for source in reported_metric_sources for key in _numeric_metric_map(source)
        }
        if not reported_metric_names:
            continue
        metrics = _result_summary_metrics(
            payload,
            metric_aliases=metric_aliases,
            scoring_metric_keys=tuple(sorted(reported_metric_names)),
            maturity_policy=maturity_policy,
        )
        metrics.update(result_effective_config_metadata(payload))
        for source_name in (None, "metrics"):
            source = payload if source_name is None else payload.get(source_name)
            if not isinstance(source, dict):
                continue
            for key, value in _numeric_metric_map(source).items():
                metrics.setdefault(key, value)
        if not _numeric_metric_map(metrics):
            continue
        variant_name = result_summary_variant_name(path, payload, run_dir)
        copied: dict[str, Any] = {
            "variant_name": variant_name,
            "metrics": metrics,
            "summary": payload.get("summary") or payload.get("description"),
            "source_result_path": str(path.relative_to(run_dir)),
            "report_signal_source": "result_summary",
        }
        for key in (
            "generation_id",
            "source_generation_id",
            "tier",
            "tier_reached",
            "evidence_stage",
            "result_status",
            "frontier_lane",
            "parent_candidate",
            "parent_variant",
            "parent_usage",
        ):
            value = metrics.get(key) if key in metrics else payload.get(key)
            if value not in (None, ""):
                copied[key] = value
        copied["source_result_sha256"] = result_summary_control_digest(payload)
        entries.append(copied)
    return _dedupe_entries(entries)


def _numeric_metric_map(payload: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in payload.items():
        name = str(key)
        if not _is_report_metric_name(name):
            continue
        numeric = _number(value)
        if numeric is not None:
            out[name] = numeric
    return out


def _collect_dimension_winners(
    entries: list[dict[str, Any]],
    *,
    baseline_values: dict[str, list[float]],
    metric_registry: _TaskMetricRegistry | dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    observations: dict[str, list[tuple[float, dict[str, Any], str | None]]] = {}
    for entry in entries:
        for metric_name, metric_value in _entry_numeric_metrics(entry).items():
            if not _is_report_metric_name(metric_name):
                continue
            direction = _metric_direction_for_name(
                metric_name,
                entry,
                metric_registry=metric_registry,
            )
            observations.setdefault(metric_name, []).append((metric_value, entry, direction))

    winners: dict[str, dict[str, Any]] = {}
    for metric_name, candidates in observations.items():
        directions = {direction for _value, _entry, direction in candidates}
        if None in directions or len(directions) != 1:
            continue
        direction = next(iter(directions))
        best_value: float | None = None
        best_entry: dict[str, Any] | None = None
        for metric_value, entry, _direction in candidates:
            if _metric_value_better(metric_value, best_value, direction=direction):
                best_value = metric_value
                best_entry = entry
        if best_value is None or best_entry is None:
            continue
        winners[metric_name] = {
            "metric": metric_name,
            "direction": direction,
            "value": best_value,
            "entry": best_entry,
            "baseline_relation": _baseline_relation(
                metric_name,
                best_value,
                direction=direction,
                baseline_values=baseline_values,
            ),
        }
    return sorted(winners.values(), key=_dimension_winner_sort_key)


def _entries_from_dimension_winners(winners: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for winner in winners:
        entry = winner.get("entry")
        if not isinstance(entry, dict):
            continue
        key = _variant_name(entry)
        if key in seen:
            continue
        seen.add(key)
        entries.append(entry)
        if len(entries) >= 12:
            break
    return entries


def _entry_numeric_metrics(entry: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    raw_metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    for source in (entry, raw_metrics):
        for key, value in source.items():
            name = str(key)
            if name in {"metrics", "extra", "details"}:
                continue
            numeric = _number(value)
            if numeric is not None:
                metrics[name] = numeric
    explicit_name = entry.get("metric_name") or entry.get("primary_metric")
    explicit_value = _number(entry.get("metric_value"))
    if isinstance(explicit_name, str) and explicit_name.strip() and explicit_value is not None:
        metrics.setdefault(explicit_name.strip(), explicit_value)
    return metrics


def _metric_value_better(value: float, current: float | None, *, direction: str | None) -> bool:
    if direction not in _VALID_METRIC_DIRECTIONS:
        return False
    if current is None:
        return True
    return value < current if direction == "minimize" else value > current


def _baseline_relation(
    metric_name: str,
    value: float,
    *,
    direction: str | None,
    baseline_values: dict[str, list[float]],
) -> str:
    if direction not in _VALID_METRIC_DIRECTIONS:
        return "direction unknown"
    values = baseline_values.get(metric_name) or []
    if not values:
        return "baseline unknown"
    baseline = min(values) if direction == "minimize" else max(values)
    if _metric_value_better(value, baseline, direction=direction):
        return f"beats baseline {_format_float(baseline)}"
    if value == baseline:
        return f"ties baseline {_format_float(baseline)}"
    return f"worse than baseline {_format_float(baseline)}"


def _dimension_winner_sort_key(winner: dict[str, Any]) -> tuple[int, str]:
    metric = str(winner.get("metric") or "")
    return (_metric_priority(metric), metric)


def _is_report_metric_name(metric_name: str) -> bool:
    name = metric_name.lower()
    if name.startswith("_") or name in {
        "generation_id",
        "source_generation_id",
        "seed",
        "pid",
        "wall_time_s",
        "runtime_s",
        "created_at",
    }:
        return False
    blocked_tokens = (
        "sha",
        "path",
        "status",
        "eligible",
        "confirmed",
        "summary_only",
        "unscored",
        "is_smoke",
        "force_",
        "source_",
        "auto_",
    )
    return not any(token in name for token in blocked_tokens)


def _metric_priority(metric_name: str) -> int:
    name = metric_name.lower()
    groups = [
        ("fitness", "score", "return", "reward", "accuracy"),
        ("auc", "f1", "precision", "recall"),
        ("loss", "error", "risk", "regret", "violation"),
        ("capacity", "fill_rate", "effective", "effn", "diversity", "entropy"),
        ("latency", "cost"),
    ]
    for idx, group in enumerate(groups):
        if any(token in name for token in group):
            return idx
    return len(groups)


def _build_report_charts(
    *,
    signal_entries: list[dict[str, Any]],
    winners: list[dict[str, Any]],
    credible: bool = True,
    metric_registry: _TaskMetricRegistry | dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = []
    task_primary_metric = (
        metric_registry.primary_metric if isinstance(metric_registry, _TaskMetricRegistry) else None
    )
    primary_metric = _choose_primary_chart_metric(
        winners,
        primary_metric=task_primary_metric,
    )
    if primary_metric:
        primary_winner = next(
            (winner for winner in winners if winner.get("metric") == primary_metric),
            {},
        )
        trend = _generation_best_series(
            signal_entries,
            primary_metric,
            direction=str(primary_winner.get("direction") or "") or None,
            metric_registry=metric_registry,
        )
        if len(trend) >= 2:
            charts.append(
                {
                    "kind": "line",
                    "title": (
                        f"Best {primary_metric} by generation"
                        if credible
                        else f"Signal-only {primary_metric} observations by generation"
                    ),
                    "metric": primary_metric,
                    "points": trend,
                }
            )
    bar_items = [
        {
            "label": str(winner.get("metric") or ""),
            "value": _number(winner.get("value")),
            "variant": _variant_name(
                winner.get("entry") if isinstance(winner.get("entry"), dict) else {}
            ),
        }
        for winner in winners[:10]
    ]
    bar_items = [item for item in bar_items if item["label"] and item["value"] is not None]
    if bar_items:
        charts.append(
            {
                "kind": "bar",
                "title": (
                    "Single-metric winners by task dimension"
                    if credible
                    else "Signal-only metric leaders by task dimension"
                ),
                "items": bar_items,
            }
        )
    scatter = _risk_reward_scatter(signal_entries, metric_registry=metric_registry)
    if scatter:
        charts.append(scatter)
    return charts


def _choose_primary_chart_metric(
    winners: list[dict[str, Any]],
    *,
    primary_metric: str | None = None,
) -> str | None:
    if primary_metric and any(winner.get("metric") == primary_metric for winner in winners):
        return primary_metric
    for winner in winners:
        metric = str(winner.get("metric") or "")
        if metric and _metric_priority(metric) <= 1:
            return metric
    if winners:
        metric = str(winners[0].get("metric") or "")
        return metric or None
    return None


def _generation_best_series(
    entries: list[dict[str, Any]],
    metric_name: str,
    *,
    direction: str | None = None,
    metric_registry: _TaskMetricRegistry | dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if direction not in _VALID_METRIC_DIRECTIONS:
        direction = _metric_direction_for_name(metric_name, metric_registry=metric_registry)
    if direction not in _VALID_METRIC_DIRECTIONS:
        return []
    best_by_gen: dict[int, dict[str, Any]] = {}
    for entry in entries:
        gen = _generation_id(entry)
        if gen is None:
            continue
        value = _entry_numeric_metrics(entry).get(metric_name)
        if value is None:
            continue
        if (
            _metric_direction_for_name(
                metric_name,
                entry,
                metric_registry=metric_registry,
            )
            != direction
        ):
            continue
        current = best_by_gen.get(gen)
        if current is None or _metric_value_better(
            value,
            _number(current.get("value")),
            direction=direction,
        ):
            best_by_gen[gen] = {
                "generation": gen,
                "value": value,
                "variant": _variant_name(entry),
            }
    return [best_by_gen[gen] for gen in sorted(best_by_gen)]


def _risk_reward_scatter(
    entries: list[dict[str, Any]],
    *,
    metric_registry: _TaskMetricRegistry | dict[str, str] | None = None,
) -> dict[str, Any] | None:
    known_directions = _consistent_metric_directions(
        entries,
        metric_registry=metric_registry,
    )
    reward_metric = _choose_metric_by_tokens(
        {name for name, direction in known_directions.items() if direction == "maximize"},
        ("return", "reward", "score", "accuracy", "fitness"),
    )
    risk_metric = _choose_metric_by_tokens(
        {name for name, direction in known_directions.items() if direction == "minimize"},
        ("loss", "error", "risk", "regret", "cost", "violation"),
    )
    if not reward_metric or not risk_metric or reward_metric == risk_metric:
        return None
    points: list[dict[str, Any]] = []
    for entry in entries:
        metrics = _entry_numeric_metrics(entry)
        reward_value = metrics.get(reward_metric)
        risk_value = metrics.get(risk_metric)
        if reward_value is None or risk_value is None:
            continue
        if (
            _metric_direction_for_name(
                reward_metric,
                entry,
                metric_registry=metric_registry,
            )
            != "maximize"
            or _metric_direction_for_name(
                risk_metric,
                entry,
                metric_registry=metric_registry,
            )
            != "minimize"
        ):
            continue
        points.append(
            {
                "label": _variant_name(entry),
                "x": risk_value,
                "y": reward_value,
                "generation": _generation_id(entry),
            }
        )
    if len(points) < 2:
        return None
    points = sorted(points, key=lambda item: abs(float(item["y"])), reverse=True)[:24]
    return {
        "kind": "scatter",
        "title": f"{reward_metric} vs {risk_metric}",
        "x_metric": risk_metric,
        "y_metric": reward_metric,
        "points": points,
    }


def _choose_metric_by_tokens(metric_names: set[str], tokens: tuple[str, ...]) -> str | None:
    for token in tokens:
        matches = sorted(name for name in metric_names if token in name.lower())
        if matches:
            return matches[0]
    return None


def _rank_entries(
    entries: list[dict[str, Any]],
    *,
    metric_registry: _TaskMetricRegistry | dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    rankable = [
        entry
        for entry in entries
        if _entry_has_known_ranking_direction(entry, metric_registry=metric_registry)
    ]
    return sorted(
        rankable,
        key=lambda entry: _entry_sort_key(entry, metric_registry=metric_registry),
    )


def _entry_sort_key(
    entry: dict[str, Any],
    *,
    metric_registry: _TaskMetricRegistry | dict[str, str] | None = None,
) -> tuple[int, float, str]:
    lane = _lane(entry).lower()
    lane_rank = 0 if lane in {"confirmed", "performance", "frontier"} else 1
    primary_metric = (
        metric_registry.primary_metric if isinstance(metric_registry, _TaskMetricRegistry) else None
    )
    _metric, value = _best_metric(entry, primary_metric=primary_metric)
    if value is None:
        score = math.inf
    else:
        direction = _metric_direction(entry, metric_registry=metric_registry)
        score = -value if direction == "maximize" else value
    return (lane_rank, score, _variant_name(entry))


def _entry_has_known_ranking_direction(
    entry: dict[str, Any],
    *,
    metric_registry: _TaskMetricRegistry | dict[str, str] | None = None,
) -> bool:
    primary_metric = (
        metric_registry.primary_metric if isinstance(metric_registry, _TaskMetricRegistry) else None
    )
    metric_name, metric_value = _best_metric(entry, primary_metric=primary_metric)
    return (
        metric_value is not None
        and _metric_direction_for_name(
            metric_name or "",
            entry,
            metric_registry=metric_registry,
        )
        in _VALID_METRIC_DIRECTIONS
    )


def _dedupe_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positions: dict[str, list[int]] = {}
    artifact_positions: dict[tuple[str, str], int] = {}
    out: list[dict[str, Any]] = []
    for entry in entries:
        key = "|".join(
            [
                _variant_name(entry),
                str(_generation_id(entry)),
                str(_best_metric(entry)[0]),
                str(_best_metric(entry)[1]),
            ]
        )
        duplicate_position: int | None = None
        same_immutable_artifact = False
        artifact_key = result_artifact_key(entry)
        if artifact_key is not None and artifact_key[0] and artifact_key[1]:
            duplicate_position = artifact_positions.get(artifact_key)
            same_immutable_artifact = duplicate_position is not None
        if duplicate_position is None:
            for position in positions.get(key, []):
                current = out[position]
                same_artifact = same_result_artifact(
                    result_artifact_key(current),
                    artifact_key,
                )
                if current != entry and not same_artifact:
                    continue
                duplicate_position = position
                same_immutable_artifact = same_artifact
                break
        if duplicate_position is not None:
            current = out[duplicate_position]
            current_restricted = _entry_has_durable_report_exclusion(current)
            entry_restricted = _entry_has_durable_report_exclusion(entry)
            if same_immutable_artifact and entry_restricted and not current_restricted:
                out[duplicate_position] = entry
            continue
        positions.setdefault(key, []).append(len(out))
        if artifact_key is not None and artifact_key[0] and artifact_key[1]:
            artifact_positions[artifact_key] = len(out)
        out.append(entry)
    return out


def _entry_has_durable_report_exclusion(entry: dict[str, Any]) -> bool:
    return durable_promotion_exclusion(entry) is not None or _entry_has_report_exclusion(entry)


def _best_metric(
    entry: dict[str, Any],
    *,
    primary_metric: str | None = None,
) -> tuple[str | None, float | None]:
    explicit_name = entry.get("metric_name") or entry.get("primary_metric")
    explicit_value = _number(entry.get("metric_value"))
    if isinstance(explicit_name, str) and explicit_name.strip() and explicit_value is not None:
        return explicit_name.strip(), explicit_value
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    if primary_metric:
        for source in (entry, metrics):
            value = _number(source.get(primary_metric))
            if value is not None:
                return primary_metric, value
    preferred_names = [
        "primary_metric",
        "task_score",
        "score",
        "accuracy",
        "reward",
        "return",
        "metric_value",
    ]
    for name in preferred_names:
        value = _number(metrics.get(name))
        if value is not None:
            return name, value
    for name, value in metrics.items():
        if str(name).startswith("_"):
            continue
        numeric = _number(value)
        if numeric is not None:
            return str(name), numeric
    return None, None


def _metric_direction(
    entry: dict[str, Any],
    *,
    metric_registry: _TaskMetricRegistry | dict[str, str] | None = None,
) -> str | None:
    primary_metric = (
        metric_registry.primary_metric if isinstance(metric_registry, _TaskMetricRegistry) else None
    )
    metric_name, _metric_value = _best_metric(entry, primary_metric=primary_metric)
    return _metric_direction_for_name(
        metric_name or "",
        entry,
        metric_registry=metric_registry,
    )


def _metric_direction_for_name(
    metric_name: str,
    entry: dict[str, Any] | None = None,
    *,
    metric_registry: _TaskMetricRegistry | dict[str, str] | None = None,
) -> str | None:
    if isinstance(metric_registry, _TaskMetricRegistry):
        direction = str(metric_registry.directions.get(metric_name) or "").strip().lower()
        if direction in _VALID_METRIC_DIRECTIONS:
            return direction
        if metric_name in metric_registry.declared_metrics:
            # A task-declared metric with no resolved direction is conflicted,
            # not an invitation to trust stale per-result metadata.
            return None
    if entry is not None:
        sources = (
            entry,
            entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {},
            entry.get("extra") if isinstance(entry.get("extra"), dict) else {},
            entry.get("details") if isinstance(entry.get("details"), dict) else {},
        )
        for source in sources:
            direction = source.get(f"{metric_name}_direction")
            normalized = str(direction or "").strip().lower()
            if normalized in _VALID_METRIC_DIRECTIONS:
                return normalized
        for source in sources:
            for prefix in ("primary", "anchor", "lane", "metric"):
                named_metric = source.get(f"{prefix}_metric_name")
                if str(named_metric or "").strip() != metric_name:
                    continue
                normalized = str(source.get(f"{prefix}_metric_direction") or "").strip().lower()
                if normalized in _VALID_METRIC_DIRECTIONS:
                    return normalized
        for source in (
            entry,
            entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {},
        ):
            direction = source.get("metric_direction") or source.get("direction")
            normalized = str(direction or "").strip().lower()
            named_metric = entry.get("metric_name") or entry.get("primary_metric")
            if normalized in _VALID_METRIC_DIRECTIONS and (
                not named_metric or str(named_metric).strip() == metric_name
            ):
                return normalized
    directions = metric_registry if isinstance(metric_registry, dict) else {}
    direction = str(directions.get(metric_name) or "").strip().lower()
    return direction if direction in _VALID_METRIC_DIRECTIONS else None


def _consistent_metric_directions(
    entries: list[dict[str, Any]],
    *,
    metric_registry: _TaskMetricRegistry | dict[str, str] | None = None,
) -> dict[str, str]:
    observed: dict[str, set[str | None]] = {}
    for entry in entries:
        for metric_name in _entry_numeric_metrics(entry):
            if not _is_report_metric_name(metric_name):
                continue
            observed.setdefault(metric_name, set()).add(
                _metric_direction_for_name(
                    metric_name,
                    entry,
                    metric_registry=metric_registry,
                )
            )
    consistent: dict[str, str] = {}
    for metric_name, directions in observed.items():
        if None in directions or len(directions) != 1:
            continue
        direction = next(iter(directions))
        if direction is not None:
            consistent[metric_name] = direction
    return consistent


def _unknown_direction_metric_names(
    entries: list[dict[str, Any]],
    *,
    metric_registry: _TaskMetricRegistry | dict[str, str] | None = None,
) -> list[str]:
    known = _consistent_metric_directions(entries, metric_registry=metric_registry)
    all_metrics = {
        metric_name
        for entry in entries
        for metric_name in _entry_numeric_metrics(entry)
        if _is_report_metric_name(metric_name)
    }
    return sorted(all_metrics - set(known), key=lambda name: (_metric_priority(name), name))


def _generation_id(entry: dict[str, Any]) -> int | None:
    for source in (entry, entry.get("metrics"), entry.get("details"), entry.get("extra")):
        if isinstance(source, dict):
            for key in ("generation_id", "source_generation_id", "gen"):
                if source.get(key) is None:
                    continue
                value = _number(source.get(key))
                if value is not None:
                    return int(value)
    return None


def _variant_name(entry: dict[str, Any]) -> str:
    for key in ("variant_name", "variant_id", "frontier_entity_key", "candidate_entity_key", "id"):
        value = entry.get(key)
        if value:
            text = str(value)
            if text.startswith("variant::"):
                return text.split("variant::", 1)[1]
            return text
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    for key in ("variant_name", "variant_id"):
        value = metrics.get(key)
        if value:
            text = str(value)
            if text.startswith("variant::"):
                return text.split("variant::", 1)[1]
            return text
    return "unknown_variant"


def _lane(entry: dict[str, Any]) -> str:
    for source in (entry, entry.get("metrics"), entry.get("extra")):
        if isinstance(source, dict):
            value = source.get("frontier_lane") or source.get("lane") or source.get("source_lane")
            if value:
                return str(value)
    return "frontier"


def _evidence_stage(entry: dict[str, Any]) -> str:
    for source in (entry, entry.get("metrics"), entry.get("extra"), entry.get("details")):
        if isinstance(source, dict):
            value = source.get("evidence_stage") or source.get("eval_stage") or source.get("stage")
            if value:
                return str(value)
    return "unknown"


def _entry_summary(entry: dict[str, Any], *, limit: int) -> str:
    for key in ("headline", "title", "summary", "content", "rationale", "description"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return _clip(value, limit)
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    for key in ("headline", "title", "summary", "content", "rationale"):
        value = metrics.get(key)
        if isinstance(value, str) and value.strip():
            return _clip(value, limit)
    return "No principle summary declared."


def _lineage_row(entry: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in (entry, entry.get("metrics"), entry.get("extra"), entry.get("details")):
        if isinstance(source, dict):
            merged.update({k: v for k, v in source.items() if v not in (None, "")})
    parent = (
        merged.get("parent_candidate")
        or merged.get("parent_variant")
        or merged.get("parent_lineage")
        or merged.get("source_parent")
    )
    source = (
        merged.get("source_result_path")
        or merged.get("result_artifact_path")
        or merged.get("summary_path")
    )
    note = (
        merged.get("next_step_intent")
        or merged.get("mechanism_hypothesis_deliverable")
        or _entry_summary(entry, limit=100)
    )
    config_status = str(merged.get("source_result_effective_config_status") or "").strip()
    config_digest = str(merged.get("source_result_effective_config_sha256") or "").strip()
    replication_status = str(merged.get("replication_effective_config_status") or "").strip()
    config_label = replication_status or config_status
    if config_digest:
        config_label = f"{config_label or 'declared'} ({config_digest[:12]})"
    return {
        "variant": _variant_name(entry),
        "gen": _generation_id(entry),
        "parent": str(parent) if parent else "",
        "source": str(source) if source else "",
        "parent_usage": str(merged.get("parent_usage") or ""),
        "effective_config": config_label,
        "note": _clip(str(note), 120),
    }


def _entry_signal_label(entry: dict[str, Any]) -> str:
    source = str(entry.get("report_signal_source") or "")
    if source == "result_summary":
        return f"result summary / {_evidence_stage(entry)}"
    if source == "shared_finding":
        return f"shared finding / {_evidence_stage(entry)}"
    if source == "validation_candidate":
        return f"validation candidate / {_evidence_stage(entry)}"
    return f"clean frontier / {_evidence_stage(entry)}"


def _report_evidence_class(
    entry: dict[str, Any],
    *,
    maturity_policy: dict[str, Any] | None,
) -> str:
    """Separate credible measurements from useful but unvalidated signals."""

    maturity = evidence_maturity_snapshot(entry, maturity_policy)
    if maturity.get("mature_enough") is False:
        return "signal"
    if _entry_has_report_exclusion(entry):
        return "signal"
    if durable_promotion_exclusion(entry) is not None:
        return "signal"
    if _entry_has_runtime_failure(entry):
        return "signal"
    if maturity.get("mature_enough") is True:
        return "credible"
    source = str(entry.get("report_signal_source") or "")
    return (
        "signal"
        if source in {"result_summary", "shared_finding", "validation_candidate"}
        else "credible"
    )


def _report_promotion_status(entry: dict[str, Any]) -> str:
    values = [
        value
        for key in ("promotion_eligible", "clean_promotion_eligible")
        for value in (_entry_bool(entry, key),)
        if value is not None
    ]
    if False in values:
        return "not eligible"
    if True in values:
        return "eligible"
    return "not declared"


def _entry_bool(entry: dict[str, Any], key: str) -> bool | None:
    for source in (entry, entry.get("metrics"), entry.get("extra"), entry.get("details")):
        if isinstance(source, dict) and isinstance(source.get(key), bool):
            return bool(source[key])
    return None


def _entry_has_runtime_failure(entry: dict[str, Any]) -> bool:
    sources = (entry, entry.get("metrics"), entry.get("extra"), entry.get("details"))
    for key in ("result_status", "final_status", "tier_status", "status", "eval_status"):
        value = next(
            (
                source.get(key)
                for source in sources
                if isinstance(source, dict) and source.get(key) not in (None, "")
            ),
            None,
        )
        if value is not None and _is_bad_result_status(str(value), allow_partial=True):
            return True
    return False


def _entry_has_report_exclusion(entry: dict[str, Any]) -> bool:
    return any(_entry_bool(entry, key) is True for key in _REPORT_EXCLUSION_BOOL_KEYS)


def _report_block_reason(
    entry: dict[str, Any],
    *,
    maturity_policy: dict[str, Any] | None = None,
) -> str:
    routing_exclusion = durable_promotion_exclusion(entry)
    if routing_exclusion:
        return routing_exclusion
    for source in (entry, entry.get("metrics"), entry.get("extra"), entry.get("details")):
        if not isinstance(source, dict):
            continue
        for key in (
            "promotion_block_reason",
            "exclusion_reason",
            "rejection_reason",
            "protocol_status",
            "result_status",
        ):
            value = source.get(key)
            if value not in (None, ""):
                return _clip(str(value), 96)
    maturity = evidence_maturity_snapshot(entry, maturity_policy)
    audit_tags = maturity.get("audit_tags")
    blockers = [str(item) for item in audit_tags or [] if ":" not in str(item)]
    if blockers:
        return _clip(", ".join(blockers), 96)
    return str(maturity.get("maturity_basis") or "validation required")


def _source_ref(entry: dict[str, Any]) -> str:
    for source in (entry, entry.get("metrics"), entry.get("extra"), entry.get("details")):
        if isinstance(source, dict):
            value = (
                source.get("source_result_path")
                or source.get("result_artifact_path")
                or source.get("summary_path")
                or source.get("source_path")
            )
            if value:
                return _clip(str(value), 96)
    return str(entry.get("report_signal_source") or "frontier")


def _health_summary(*, run_summary: dict[str, Any], orchestrator: dict[str, Any]) -> list[str]:
    out: list[str] = []
    exit_condition = run_summary.get("exit_condition") or orchestrator.get("exit_condition")
    status = run_summary.get("status") or orchestrator.get("status")
    if not status and run_summary.get("exit_code") == 0:
        status = "succeeded"
    if not status and exit_condition in {"completed", "max_generations", "plateau"}:
        status = "succeeded"
    if not status and exit_condition == "error":
        status = "failed"
    status = status or "unknown"
    exit_condition = exit_condition or "unknown"
    out.append(f"Status: `{status}`, exit condition: `{exit_condition}`.")
    completed = run_summary.get("generations_completed") or orchestrator.get(
        "generations_completed"
    )
    max_generations = run_summary.get("max_generations") or orchestrator.get("max_generations")
    if completed is not None or max_generations is not None:
        out.append(f"Generation progress: `{_display(completed)}` / `{_display(max_generations)}`.")
    findings = run_summary.get("findings_total")
    if findings is None and isinstance(run_summary.get("finding_summary"), dict):
        findings = run_summary.get("finding_summary", {}).get("accepted")
    if findings is None:
        findings = orchestrator.get("findings_total")
    if findings is not None:
        out.append(f"Structured findings visible: `{findings}`.")
    warnings = []
    for source in (run_summary, orchestrator):
        raw = source.get("warnings") if isinstance(source, dict) else None
        if isinstance(raw, list):
            warnings.extend(str(item) for item in raw[:5])
    if warnings:
        out.append("Warnings: " + "; ".join(f"`{_clip(w, 120)}`" for w in warnings) + ".")
    else:
        out.append("No explicit run-summary warnings were found.")
    return out


def _write_pdf_report(
    *,
    path: Path,
    payload: dict[str, Any],
    run_dir: Path,
    trigger: str,
    generation_id: int | None,
    report_kind: str,
) -> None:
    pdf = _SimplePdf()
    top_entries = payload.get("top_entries") if isinstance(payload.get("top_entries"), list) else []
    strong_entries = (
        payload.get("strong_entries") if isinstance(payload.get("strong_entries"), list) else []
    )
    winners = (
        payload.get("dimension_winners")
        if isinstance(payload.get("dimension_winners"), list)
        else []
    )
    signal_winners = (
        payload.get("signal_dimension_winners")
        if isinstance(payload.get("signal_dimension_winners"), list)
        else []
    )
    charts = payload.get("charts") if isinstance(payload.get("charts"), list) else []
    run_summary = payload.get("run_summary") if isinstance(payload.get("run_summary"), dict) else {}
    orchestrator = (
        payload.get("orchestrator_status")
        if isinstance(payload.get("orchestrator_status"), dict)
        else {}
    )
    primary_metric = str(payload.get("primary_metric") or "").strip() or None

    pdf.heading(f"Praxist Run Report: {run_dir.name}", size=16)
    pdf.text(f"Generated at: {_utc_timestamp()}")
    pdf.text(f"Trigger: {trigger}")
    if generation_id is not None:
        pdf.text(f"Generation context: {generation_id}")
    pdf.text(f"Report kind: {report_kind}")
    pdf.text(f"Run directory: {run_dir}")
    pdf.space(10)

    pdf.heading("A. Strongest Variants And Pareto Front", size=13)
    if top_entries:
        pdf.text("Clean frontier / Pareto entries:")
        for idx, entry in enumerate(top_entries[:10], start=1):
            metric, value = _best_metric(
                entry,
                primary_metric=primary_metric,
            )
            pdf.bullet(
                f"{idx}. {_variant_name(entry)} | gen {_display(_generation_id(entry))} | "
                f"{metric or 'metric'}={_format_float(value)} | {_entry_signal_label(entry)}"
            )
    else:
        pdf.text(
            "No clean frontier/Pareto entries were found. Mature measurements and "
            "broader validation signals are shown separately."
        )
    if winners:
        pdf.space(4)
        pdf.text("Mature dimension winners:")
        for winner in winners[:24]:
            entry = winner.get("entry") if isinstance(winner.get("entry"), dict) else {}
            pdf.bullet(
                f"{winner.get('metric')} ({winner.get('direction')}): "
                f"{_variant_name(entry)} = {_format_float(_number(winner.get('value')))}; "
                f"{winner.get('baseline_relation') or 'baseline unknown'}; "
                f"promotion {_report_promotion_status(entry)}"
            )
    if signal_winners:
        pdf.space(4)
        pdf.text("Strong signals requiring validation:")
        for winner in signal_winners[:24]:
            entry = winner.get("entry") if isinstance(winner.get("entry"), dict) else {}
            pdf.bullet(
                f"{winner.get('metric')} ({winner.get('direction')}): "
                f"{_variant_name(entry)} = {_format_float(_number(winner.get('value')))}; "
                f"{winner.get('block_reason') or 'validation required'}"
            )
    pdf.space(8)

    pdf.heading("B. Strong-Variant Evolution And Lineage", size=13)
    lineage_rows = [_lineage_row(entry) for entry in strong_entries[:12]]
    if lineage_rows:
        for row in lineage_rows:
            parent = row["parent"] or row["source"] or "not declared"
            pdf.bullet(
                f"{row['variant']} | gen {_display(row['gen'])} | parent/source: {parent} | "
                f"{row['note']}"
            )
    else:
        pdf.text("No lineage can be inferred from the available result signals yet.")
    pdf.space(8)

    pdf.heading("C. Run Health And Evidence State", size=13)
    for item in _health_summary(run_summary=run_summary, orchestrator=orchestrator):
        pdf.bullet(item)

    if charts:
        pdf.new_page()
        pdf.heading("Visual Companion", size=15)
        for chart in charts:
            pdf.draw_chart(chart)
    pdf.save(path)


class _SimplePdf:
    """Small dependency-free PDF writer for derived report companions."""

    width = 612.0
    height = 792.0
    margin = 54.0

    def __init__(self) -> None:
        self._pages: list[list[str]] = []
        self._ops: list[str] = []
        self._y = self.height - self.margin

    def new_page(self) -> None:
        if self._ops:
            self._pages.append(self._ops)
        self._ops = []
        self._y = self.height - self.margin

    def save(self, path: Path) -> None:
        if self._ops or not self._pages:
            self._pages.append(self._ops)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_pdf_objects(path, self._pages, self.width, self.height)

    def heading(self, text: str, *, size: int) -> None:
        self._ensure_space(size + 14)
        self._text(self.margin, self._y, text, size=size)
        self._y -= size + 8

    def text(self, text: str, *, size: int = 9) -> None:
        for line in _wrap_text(text, width=95):
            self._ensure_space(size + 5)
            self._text(self.margin, self._y, line, size=size)
            self._y -= size + 4

    def bullet(self, text: str) -> None:
        for idx, line in enumerate(_wrap_text(text, width=88)):
            self._ensure_space(14)
            prefix = "- " if idx == 0 else "  "
            self._text(self.margin + 8, self._y, prefix + line, size=8)
            self._y -= 12

    def space(self, amount: float) -> None:
        self._y -= amount

    def draw_chart(self, chart: dict[str, Any]) -> None:
        kind = str(chart.get("kind") or "")
        if kind == "line":
            self._draw_line_chart(chart)
        elif kind == "bar":
            self._draw_bar_chart(chart)
        elif kind == "scatter":
            self._draw_scatter_chart(chart)

    def _draw_bar_chart(self, chart: dict[str, Any]) -> None:
        items = [item for item in chart.get("items", []) if isinstance(item, dict)]
        items = items[:10]
        if not items:
            return
        self._ensure_space(220)
        self.heading(str(chart.get("title") or "Bar chart"), size=12)
        x = self.margin + 150
        width = 300.0
        max_abs = max(abs(float(item["value"])) for item in items if item.get("value") is not None)
        max_abs = max(max_abs, 1e-9)
        for item in items:
            label = _clip(str(item.get("label") or ""), 28)
            value = float(item.get("value") or 0.0)
            self._text(self.margin, self._y, label, size=7)
            bar_width = abs(value) / max_abs * width
            self._rect(x, self._y - 2, bar_width, 7, fill=(0.24, 0.47, 0.72))
            self._text(x + bar_width + 5, self._y, _format_float(value), size=7)
            self._y -= 15
        self.space(8)

    def _draw_line_chart(self, chart: dict[str, Any]) -> None:
        points = [item for item in chart.get("points", []) if isinstance(item, dict)]
        if len(points) < 2:
            return
        self._ensure_space(220)
        self.heading(str(chart.get("title") or "Line chart"), size=12)
        x0, y0, w, h = self.margin + 20, self._y - 160, 460.0, 140.0
        values = [float(item.get("value") or 0.0) for item in points]
        min_v, max_v = min(values), max(values)
        if min_v == max_v:
            min_v -= 1.0
            max_v += 1.0
        self._line(x0, y0, x0 + w, y0)
        self._line(x0, y0, x0, y0 + h)
        coords: list[tuple[float, float]] = []
        for idx, item in enumerate(points):
            x = x0 + (idx / max(1, len(points) - 1)) * w
            y = y0 + ((float(item.get("value") or 0.0) - min_v) / (max_v - min_v)) * h
            coords.append((x, y))
        for left, right in zip(coords, coords[1:], strict=False):
            self._line(left[0], left[1], right[0], right[1], stroke=(0.1, 0.38, 0.68))
        for idx, (x, y) in enumerate(coords):
            self._rect(x - 1.5, y - 1.5, 3, 3, fill=(0.1, 0.38, 0.68))
            if idx in {0, len(coords) - 1}:
                self._text(x - 8, y + 8, _format_float(values[idx]), size=7)
        self._text(x0, y0 - 12, f"gen {points[0].get('generation')}", size=7)
        self._text(x0 + w - 36, y0 - 12, f"gen {points[-1].get('generation')}", size=7)
        self._y = y0 - 28

    def _draw_scatter_chart(self, chart: dict[str, Any]) -> None:
        points = [item for item in chart.get("points", []) if isinstance(item, dict)]
        if len(points) < 2:
            return
        self._ensure_space(230)
        self.heading(str(chart.get("title") or "Scatter chart"), size=12)
        x0, y0, w, h = self.margin + 40, self._y - 155, 420.0, 135.0
        xs = [float(item.get("x") or 0.0) for item in points]
        ys = [float(item.get("y") or 0.0) for item in points]
        min_x, max_x = _pad_range(min(xs), max(xs))
        min_y, max_y = _pad_range(min(ys), max(ys))
        self._line(x0, y0, x0 + w, y0)
        self._line(x0, y0, x0, y0 + h)
        for item in points:
            x = x0 + ((float(item.get("x") or 0.0) - min_x) / (max_x - min_x)) * w
            y = y0 + ((float(item.get("y") or 0.0) - min_y) / (max_y - min_y)) * h
            self._rect(x - 2, y - 2, 4, 4, fill=(0.76, 0.36, 0.18))
        self._text(x0, y0 - 13, str(chart.get("x_metric") or "x"), size=7)
        self._text(x0 + 6, y0 + h + 7, str(chart.get("y_metric") or "y"), size=7)
        self._y = y0 - 28

    def _ensure_space(self, amount: float) -> None:
        if self._y - amount < self.margin:
            self.new_page()

    def _text(self, x: float, y: float, text: str, *, size: int) -> None:
        self._ops.append(f"BT /F1 {size} Tf {x:.2f} {y:.2f} Td ({_pdf_text(text)}) Tj ET")

    def _line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        r, g, b = stroke
        self._ops.append(f"{r:.3f} {g:.3f} {b:.3f} RG {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def _rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: tuple[float, float, float],
    ) -> None:
        r, g, b = fill
        self._ops.append(
            f"{r:.3f} {g:.3f} {b:.3f} rg {x:.2f} {y:.2f} {width:.2f} {height:.2f} re f"
        )


def _write_pdf_objects(path: Path, pages: list[list[str]], width: float, height: float) -> None:
    objects: dict[int, bytes] = {}
    catalog_id = 1
    pages_id = 2
    font_id = 3
    page_ids: list[int] = []
    next_id = 4
    for ops in pages:
        page_id = next_id
        content_id = next_id + 1
        next_id += 2
        page_ids.append(page_id)
        stream = "\n".join(ops).encode("latin-1", "replace")
        objects[content_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream"
        )
        objects[page_id] = (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {width:.0f} {height:.0f}] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("ascii")
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[catalog_id] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("ascii")
    objects[pages_id] = (f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>").encode(
        "ascii"
    )
    objects[font_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for obj_id in range(1, next_id):
        offsets[obj_id] = len(out)
        out.extend(f"{obj_id} 0 obj\n".encode("ascii"))
        out.extend(objects[obj_id])
        out.extend(b"\nendobj\n")
    xref_start = len(out)
    out.extend(f"xref\n0 {next_id}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for obj_id in range(1, next_id):
        out.extend(f"{offsets[obj_id]:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer\n<< /Size {next_id} /Root {catalog_id} 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode(
            "ascii"
        )
    )
    path.write_bytes(bytes(out))


def _wrap_text(text: str, *, width: int) -> list[str]:
    words = re.sub(r"\s+", " ", str(text)).strip().split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word[:width]
    if current:
        lines.append(current)
    return lines or [""]


def _pdf_text(text: str) -> str:
    safe = str(text).encode("latin-1", "replace").decode("latin-1")
    return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pad_range(low: float, high: float) -> tuple[float, float]:
    if low == high:
        return low - 1.0, high + 1.0
    pad = (high - low) * 0.05
    return low - pad, high + pad


def _load_generation_boundaries(run_dir: Path) -> list[dict[str, Any]]:
    boundaries: list[dict[str, Any]] = []
    for path in sorted(Path(run_dir).glob("gen_*/generation_boundary.json")):
        payload = _read_json(path, default={})
        if isinstance(payload, dict):
            boundaries.append(payload)
    return boundaries


def _load_shared_findings(run_dir: Path, *, cap: int) -> list[dict[str, Any]]:
    findings_dir = Path(run_dir) / "shared_findings"
    out: list[dict[str, Any]] = []
    if not findings_dir.exists():
        return out
    for path in sorted(findings_dir.glob("*.json"))[:cap]:
        payload = _read_json(path, default={})
        if isinstance(payload, dict):
            out.append(payload)
    return out


def _read_json(path: Path, *, default: Any) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def _display(value: Any) -> str:
    return "?" if value is None else str(value)


def _escape_table(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _clip(value: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")[:120]


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _timestamp_for_filename() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
