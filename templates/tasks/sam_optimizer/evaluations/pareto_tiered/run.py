#!/usr/bin/env python3
"""Run the SAM task's canonical tiered evaluation protocol.

The Praxist agent still owns the research idea and optimizer implementation. This
is the only public evaluation entrypoint that peers should call. The script
owns the repetitive mechanics: run T1/T2/T3 in order, apply task-local
promotion gates, preserve raw benchmark JSON/logs, and print a compact summary
that the agent can use for its notebook and finding. The benchmark runner under
``assets/harness/benchmark`` is an internal implementation detail.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

TIERS = ("T1", "T2", "T3")
TIER_REQUIRED_CELLS = {"T1": 3, "T2": 6, "T3": 15}
SUMMARY_SCHEMA_VERSION = "sam_optimizer.tiered_eval_summary.v1"
REFERENCE_MIN_TIER = "T1"
REFERENCE_MAX_TIER = "T3"
NON_PARENTABLE_MARKERS = (
    "is_smoke_eval",
    "partial",
    "scout_only",
    "validation_only",
    "validation_only_result",
    "late_after_generation_boundary",
    "suspect_protocol",
    "suspect_leakage",
    "protocol_failed",
    "protocol_integrity_failed",
    "excluded_from_durable_frontier",
)


def task_root_from_script() -> Path:
    """Return the task project root for this script."""

    return Path(__file__).resolve().parents[2]


def default_max_tier() -> str:
    """Return the default maximum tier, honoring smoke-run env overrides."""

    env_value = os.environ.get("PRAXIST_SAM_MAX_TIER", "T3").strip().upper()
    return env_value if env_value in TIERS else "T3"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    root = task_root_from_script()
    parser = argparse.ArgumentParser(
        description="Run SAM tiered evaluation and emit a compact gate summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--variant-path", required=True)
    parser.add_argument("--variant-name", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--data-dir", default=os.environ.get("PRAXIST_DATA_DIR", ""))
    parser.add_argument("--min-tier", choices=TIERS, default="T1")
    parser.add_argument("--max-tier", choices=TIERS, default=default_max_tier())
    parser.add_argument(
        "--benchmark-entrypoint",
        default=str(root / "assets" / "harness" / "benchmark" / "run_benchmark.py"),
        help="Internal benchmark runner used by this evaluation protocol.",
    )
    parser.add_argument(
        "--baseline-results",
        default=str(root / "assets" / "baselines" / "results.jsonl"),
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Use existing tier JSON files instead of re-running completed tiers.",
    )
    parser.add_argument(
        "--parallel-datasets",
        default="auto",
        help="Forwarded to the internal benchmark runner.",
    )
    return parser.parse_args(argv)


def variant_name_from_path(path: str) -> str:
    """Return the benchmark's custom variant stem for a variant path."""

    return Path(path).stem.replace(".", "_")


def tier_range(min_tier: str, max_tier: str) -> list[str]:
    """Return the inclusive tier sequence from min_tier to max_tier."""

    start = TIERS.index(min_tier)
    end = TIERS.index(max_tier)
    if start > end:
        raise ValueError(f"min-tier {min_tier} cannot be after max-tier {max_tier}")
    return list(TIERS[start : end + 1])


def load_baselines(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Load curated baseline rows keyed by optimizer and dataset."""

    baselines: dict[str, dict[str, dict[str, Any]]] = {}
    if not path.exists():
        return baselines
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        optimizer = row.get("optimizer")
        dataset = row.get("dataset")
        if not isinstance(optimizer, str) or not isinstance(dataset, str):
            continue
        baselines.setdefault(optimizer, {})[dataset] = row
    return baselines


def metric_mean(payload: dict[str, Any], metric_name: str) -> float | None:
    """Extract a numeric mean from a metric object."""

    metric = payload.get(metric_name)
    if isinstance(metric, dict):
        value = metric.get("mean")
    else:
        value = metric
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def marker_is_true(value: Any) -> bool:
    """Interpret explicit evaluator control markers without guessing."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 1
    return False


def dataset_metric(result: dict[str, Any], dataset: str, metric_name: str) -> float | None:
    """Extract a dataset-level metric mean from a benchmark result."""

    per_dataset = result.get("per_dataset")
    if not isinstance(per_dataset, dict):
        return None
    dataset_payload = per_dataset.get(dataset)
    if not isinstance(dataset_payload, dict):
        return None
    return metric_mean(dataset_payload, metric_name)


def dataset_metric_stat(
    result: dict[str, Any], dataset: str, metric_name: str, stat: str
) -> float | None:
    """Extract a dataset-level metric statistic such as mean or max."""

    per_dataset = result.get("per_dataset")
    if not isinstance(per_dataset, dict):
        return None
    dataset_payload = per_dataset.get(dataset)
    if not isinstance(dataset_payload, dict):
        return None
    metric = dataset_payload.get(metric_name)
    if not isinstance(metric, dict):
        return None
    value = metric.get(stat)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def tier_has_failure(result: dict[str, Any]) -> str | None:
    """Return a failure reason if any dataset visibly failed."""

    per_dataset = result.get("per_dataset")
    if not isinstance(per_dataset, dict) or not per_dataset:
        return "benchmark produced no per-dataset results"
    for dataset, payload in per_dataset.items():
        if not isinstance(payload, dict):
            return f"{dataset}: malformed result"
        status = payload.get("status", "ok")
        if status not in ("ok", "partial"):
            return f"{dataset}: status={status}"
        failed = payload.get("num_seeds_failed", 0)
        if isinstance(failed, int) and failed > 0:
            return f"{dataset}: {failed} seeds failed"
        accuracy = metric_mean(payload, "test_accuracy")
        if accuracy is None:
            return f"{dataset}: missing test_accuracy"
    return None


def gate_decision(
    tier: str,
    result: dict[str, Any],
    baselines: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Apply the task-local T1/T2/T3 gate for one benchmark result."""

    failure = tier_has_failure(result)
    if failure is not None:
        return {"passed": False, "reason": failure, "checks": []}

    vanilla = baselines.get("vanilla_sam", {})
    checks: list[dict[str, Any]] = []
    if tier == "T1":
        baseline = metric_mean(vanilla.get("cifar100", {}), "test_accuracy")
        observed = dataset_metric_stat(result, "cifar100", "test_accuracy", "max")
        if observed is None:
            observed = dataset_metric(result, "cifar100", "test_accuracy")
        if baseline is None or observed is None:
            return {
                "passed": False,
                "reason": "missing cifar100 baseline or observed accuracy",
                "checks": checks,
            }
        threshold = 0.85 * baseline
        passed = observed >= threshold
        checks.append(
            {
                "dataset": "cifar100",
                "metric": "test_accuracy",
                "observed": observed,
                "threshold": threshold,
                "baseline": baseline,
                "passed": passed,
            }
        )
        return {
            "passed": passed,
            "reason": "T1 passed" if passed else "cifar100 below 85% vanilla_sam",
            "checks": checks,
        }

    if tier == "T2":
        passed = True
        for dataset in ("cifar10", "cifar100"):
            baseline = metric_mean(vanilla.get(dataset, {}), "test_accuracy")
            observed = dataset_metric(result, dataset, "test_accuracy")
            if baseline is None or observed is None:
                return {
                    "passed": False,
                    "reason": f"missing {dataset} baseline or observed accuracy",
                    "checks": checks,
                }
            threshold = baseline - 0.01
            dataset_passed = observed >= threshold
            passed = passed and dataset_passed
            checks.append(
                {
                    "dataset": dataset,
                    "metric": "test_accuracy",
                    "observed": observed,
                    "threshold": threshold,
                    "baseline": baseline,
                    "passed": dataset_passed,
                }
            )
        return {
            "passed": passed,
            "reason": "T2 passed" if passed else "one or more T2 datasets below vanilla_sam - 1pp",
            "checks": checks,
        }

    promotable = bool(result.get("promotion_eligible"))
    return {
        "passed": promotable,
        "reason": "T3 promotable" if promotable else "T3 completed but is not promotable",
        "checks": [
            {
                "metric": "promotion_eligible",
                "observed": promotable,
                "passed": promotable,
            }
        ],
    }


def result_path_for(output_dir: Path, variant_name: str, tier: str) -> Path:
    """Return the run_benchmark multi-result path for a custom variant."""

    return output_dir / f"custom_{variant_name}_{tier}_multi_benchmark.json"


def summary_path_for(output_dir: Path, variant_name: str) -> Path:
    """Return the compact summary path for this tool invocation."""

    return output_dir / f"custom_{variant_name}_tiered_eval_summary.json"


def run_benchmark_tier(args: argparse.Namespace, tier: str, variant_name: str) -> dict[str, Any]:
    """Run one benchmark tier or reuse an existing tier JSON."""

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_path_for(output_dir, variant_name, tier)
    log_path = output_dir / f"custom_{variant_name}_{tier}_tiered_eval.log"
    if args.reuse_existing and result_path.exists():
        return {
            "tier": tier,
            "ran": False,
            "returncode": 0,
            "result_path": str(result_path),
            "log_path": str(log_path),
            "result": json.loads(result_path.read_text(encoding="utf-8")),
        }

    cmd = [
        sys.executable,
        str(Path(args.benchmark_entrypoint).resolve()),
        "--optimizer",
        "custom",
        "--variant-path",
        str(Path(args.variant_path).resolve()),
        "--tier",
        tier,
        "--output-dir",
        str(output_dir),
        "--parallel-datasets",
        str(args.parallel_datasets),
    ]
    if args.data_dir:
        cmd.extend(["--data-dir", str(args.data_dir)])

    env = os.environ.copy()
    env.setdefault("GPU_GOVERNOR_MAX_PER_GPU", "1")
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
    return {
        "tier": tier,
        "ran": True,
        "returncode": completed.returncode,
        "result_path": str(result_path),
        "log_path": str(log_path),
        "result": result,
    }


def observed_train_seconds(result: dict[str, Any]) -> float | None:
    """Return total observed train time across completed evaluation cells."""

    per_dataset = result.get("per_dataset")
    if not isinstance(per_dataset, dict):
        return None
    observed_seconds = 0.0
    completed_cells = 0
    for payload in per_dataset.values():
        if not isinstance(payload, dict):
            continue
        train_seconds = metric_mean(payload, "train_time_seconds")
        completed_seeds = payload.get("num_seeds_ok", payload.get("num_seeds"))
        if (
            train_seconds is None
            or isinstance(completed_seeds, bool)
            or not isinstance(completed_seeds, int)
            or completed_seeds <= 0
        ):
            continue
        observed_seconds += train_seconds * completed_seeds
        completed_cells += completed_seeds
    return observed_seconds if completed_cells > 0 else None


def summarize_metrics(result: dict[str, Any] | None) -> dict[str, Any]:
    """Return the small metric subset agents normally need."""

    if result is None:
        return {}
    scored_cell_count = 0
    summary: dict[str, Any] = {
        "tier": result.get("tier"),
        "tier_reached": result.get("tier"),
        "evidence_stage": result.get("tier"),
        "promotion_eligible": result.get("promotion_eligible"),
        "mean_test_accuracy": metric_mean(result, "mean_test_accuracy"),
        "mean_train_test_gap": metric_mean(result, "mean_train_test_gap"),
        "sharpness_top_eigen": metric_mean(result, "sharpness_top_eigen"),
        "per_dataset": {},
    }
    for marker in (*NON_PARENTABLE_MARKERS, "protocol_integrity_passed"):
        if marker in result:
            summary[marker] = result[marker]
    train_seconds = observed_train_seconds(result)
    if train_seconds is not None:
        summary["wall_time_seconds_total"] = train_seconds
    per_dataset = result.get("per_dataset")
    if isinstance(per_dataset, dict):
        for dataset, payload in per_dataset.items():
            if isinstance(payload, dict):
                summary["per_dataset"][dataset] = {
                    "status": payload.get("status"),
                    "test_accuracy": metric_mean(payload, "test_accuracy"),
                    "train_test_gap": metric_mean(payload, "train_test_gap"),
                    "num_seeds_ok": payload.get("num_seeds_ok"),
                    "num_seeds_total": payload.get("num_seeds_total"),
                }
                num_seeds_ok = payload.get("num_seeds_ok")
                if isinstance(num_seeds_ok, int):
                    scored_cell_count += max(0, num_seeds_ok)
    if scored_cell_count:
        summary["scored_cell_count"] = scored_cell_count
        summary["n_eval_cells"] = scored_cell_count
    return summary


def attach_maturity_ratios(
    metrics_summary: dict[str, Any],
    *,
    tier: str,
    min_tier: str = REFERENCE_MIN_TIER,
    max_tier: str = REFERENCE_MAX_TIER,
) -> None:
    """Attach generic Praxist maturity ratios using this task's tier/cell semantics."""

    try:
        completed_tier_steps = TIERS.index(tier) - TIERS.index(min_tier) + 1
        requested_tier_steps = TIERS.index(max_tier) - TIERS.index(min_tier) + 1
    except ValueError:
        completed_tier_steps = 0
        requested_tier_steps = 0
    effort_ratio = (
        max(0.0, min(1.0, completed_tier_steps / requested_tier_steps))
        if requested_tier_steps > 0
        else 0.0
    )
    scored_cells = metrics_summary.get("scored_cell_count")
    try:
        scored = max(0.0, float(scored_cells or 0.0))
    except (TypeError, ValueError):
        scored = 0.0
    target_cells = float(TIER_REQUIRED_CELLS.get(max_tier, TIER_REQUIRED_CELLS["T3"]))
    coverage_ratio = max(0.0, min(1.0, scored / target_cells)) if target_cells > 0 else 0.0
    metrics_summary["actual_effort_units"] = completed_tier_steps
    metrics_summary["reference_effort_units"] = requested_tier_steps
    metrics_summary["effort_ratio"] = round(effort_ratio, 4)
    metrics_summary["completed_required_eval_units"] = int(scored)
    metrics_summary["total_required_eval_units"] = int(target_cells)
    metrics_summary["coverage_ratio"] = round(coverage_ratio, 4)
    scored_complete = effort_ratio >= 1.0 and coverage_ratio >= 1.0
    explicitly_non_parentable = any(
        marker_is_true(metrics_summary.get(marker)) for marker in NON_PARENTABLE_MARKERS
    )
    if "protocol_integrity_passed" in metrics_summary:
        explicitly_non_parentable = explicitly_non_parentable or not marker_is_true(
            metrics_summary["protocol_integrity_passed"]
        )
    source_lane = (
        "performance" if scored_complete and not explicitly_non_parentable else "task_candidate"
    )
    metrics_summary["scored_complete"] = scored_complete
    metrics_summary["frontier_lane"] = source_lane
    metrics_summary["promotion_lane"] = source_lane


def sync_maturity_metadata_to_benchmark_result(
    result: dict[str, Any],
    metrics_summary: dict[str, Any],
    *,
    tier: str,
    result_path: str,
) -> None:
    """Mirror peer-facing maturity metadata into the raw benchmark JSON."""

    for key in (
        "tier",
        "tier_reached",
        "evidence_stage",
        "promotion_eligible",
        "actual_effort_units",
        "reference_effort_units",
        "effort_ratio",
        "completed_required_eval_units",
        "total_required_eval_units",
        "coverage_ratio",
        "scored_complete",
        "frontier_lane",
        "promotion_lane",
        "wall_time_seconds_total",
    ):
        if key in metrics_summary:
            result[key] = metrics_summary[key]
    result.setdefault("tier", tier)
    result.setdefault("tier_reached", tier)
    result.setdefault("evidence_stage", tier)
    if not str(result_path).strip():
        return
    path = Path(result_path)
    if path.exists():
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)


def run_tiered_evaluation(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    """Run the configured tier sequence and return process exit code plus summary."""

    variant_name = args.variant_name or variant_name_from_path(args.variant_path)
    output_dir = Path(args.output_dir or Path(args.variant_path).resolve().parent / "results")
    args.output_dir = str(output_dir)
    baselines = load_baselines(Path(args.baseline_results))
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "variant_name": variant_name,
        "variant_path": str(Path(args.variant_path).resolve()),
        "output_dir": str(output_dir.resolve()),
        "min_tier": args.min_tier,
        "max_tier": args.max_tier,
        "tiers": [],
        "final_status": "not_started",
    }

    for tier in tier_range(args.min_tier, args.max_tier):
        tier_record = run_benchmark_tier(args, tier, variant_name)
        result = tier_record.pop("result")
        tier_record["metrics_summary"] = summarize_metrics(result)
        attach_maturity_ratios(
            tier_record["metrics_summary"],
            tier=tier,
        )
        if isinstance(result, dict):
            sync_maturity_metadata_to_benchmark_result(
                result,
                tier_record["metrics_summary"],
                tier=tier,
                result_path=str(tier_record.get("result_path") or ""),
            )
        if tier_record["returncode"] != 0:
            tier_record["gate"] = {
                "passed": False,
                "reason": f"benchmark exited with {tier_record['returncode']}",
                "checks": [],
            }
            summary["tiers"].append(tier_record)
            summary["final_status"] = "benchmark_error"
            summary["stop_reason"] = tier_record["gate"]["reason"]
            _write_summary(output_dir, variant_name, summary)
            return 1, summary
        if not isinstance(result, dict):
            tier_record["gate"] = {
                "passed": False,
                "reason": "benchmark result JSON was not produced",
                "checks": [],
            }
            summary["tiers"].append(tier_record)
            summary["final_status"] = "benchmark_error"
            summary["stop_reason"] = tier_record["gate"]["reason"]
            _write_summary(output_dir, variant_name, summary)
            return 1, summary

        decision = gate_decision(tier, result, baselines)
        tier_record["gate"] = decision
        summary["tiers"].append(tier_record)
        if not decision["passed"]:
            summary["final_status"] = f"stopped_at_{tier}"
            summary["stop_reason"] = decision["reason"]
            _write_summary(output_dir, variant_name, summary)
            return 0, summary

    summary["final_status"] = "passed_max_tier"
    summary["stop_reason"] = "completed requested tier range"
    _write_summary(output_dir, variant_name, summary)
    return 0, summary


def _write_summary(output_dir: Path, variant_name: str, summary: dict[str, Any]) -> None:
    path = summary_path_for(output_dir, variant_name)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    summary["summary_path"] = str(path)


def compact_stdout_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the concise JSON printed to the agent context."""

    return {
        "schema_version": summary.get("schema_version"),
        "variant_name": summary.get("variant_name"),
        "final_status": summary.get("final_status"),
        "stop_reason": summary.get("stop_reason"),
        "summary_path": summary.get("summary_path"),
        "tiers": [
            {
                "tier": tier.get("tier"),
                "result_path": tier.get("result_path"),
                "log_path": tier.get("log_path"),
                "gate": tier.get("gate"),
                "metrics_summary": tier.get("metrics_summary"),
            }
            for tier in summary.get("tiers", [])
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    exit_code, summary = run_tiered_evaluation(args)
    print(json.dumps(compact_stdout_summary(summary), indent=2, default=str))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
