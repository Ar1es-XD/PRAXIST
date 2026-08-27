"""Baseline-cache validation for research-loop runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from praxist.plugins.workflow_stages.research_loop.backend import (
    baseline_cache as baseline_cache_mod,
)

logger = logging.getLogger(__name__)


def validate_baseline_cache_for_run(
    *,
    task_spec: Any,
    workspace: Path,
    run_dir: Path,
) -> None:
    """Validate curated and cached baselines without blocking the run."""
    try:
        curated_baseline_path = None
        task_assets = getattr(task_spec, "_raw", {}).get("task_assets", {})
        if isinstance(task_assets, dict):
            baselines_assets = task_assets.get("baselines", {})
            if isinstance(baselines_assets, dict):
                curated_rel = baselines_assets.get("curated_results")
                if curated_rel:
                    curated_baseline_path = task_spec.task_dir / curated_rel
        curated_entries = baseline_cache_mod.load_curated_baseline_entries(curated_baseline_path)
        report = baseline_cache_mod.validate_cache(
            task_id=task_spec.task_id,
            workspace=workspace,
            expected_baseline_names=[b.name for b in task_spec.baselines],
            curated_entries=curated_entries,
        )
        baseline_cache_mod.write_report_for_peers(report, run_dir)
        if report.stale or report.missing_baselines:
            logger.warning(
                "baseline cache: %d stale, %d missing (%s). See %s/baseline_cache_status.json",
                report.stale,
                len(report.missing_baselines),
                ",".join(report.missing_baselines) or "none",
                run_dir,
            )
        else:
            logger.info(
                "baseline cache: %d fresh, %d curated, 0 stale, 0 missing",
                report.fresh,
                len(report.curated_baseline_names),
            )
    except Exception as e:  # noqa: BLE001 - baseline validation is advisory.
        logger.warning("baseline cache validation failed (non-fatal): %s", e)
