"""Run Praxist unittest profiles under coverage.py and write local reports.

The script intentionally uses ``unittest`` discovery because the long-lived Praxist
test tree is organized around unittest package entrypoints. Pytest remains
available for the narrow unit smoke guardrail, but coverage for unit and
integration profiles is computed here so CI and local runs use the same path.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_OUTPUT_DIR = REPO_ROOT / "cover"
PROFILE_MODULES: dict[str, tuple[str, ...]] = {
    "unit": (
        "tests.core",
        "tests.conformance",
        "tests.workflows",
        "tests.hardening",
        "tests.adversarial",
        "tests.legacy_migration",
        "tests.unit",
    ),
    "integration": ("tests.integration",),
}
PROFILE_PYTEST_PATHS: dict[str, tuple[str, ...]] = {
    "unit": ("tests/product_usage",),
    "integration": (),
}

# Unit coverage is the fast, deterministic contract surface used by CI.  It
# intentionally excludes modules whose meaningful coverage requires a full
# orchestrator run, subprocess/runtime adapters, or external-service boundary
# simulation.  Those modules are still exercised by the integration profile and
# targeted contract tests; keeping them out of the unit numerator prevents the
# unit ratchet from incentivizing brittle pseudo-integration tests.
_UNIT_PROFILE_OMIT_RELATIVE: tuple[str, ...] = (
    "praxist/cli/start.py",
    "praxist/core/budget.py",
    "praxist/core/execution_guards.py",
    "praxist/core/panel_topology.py",
    "praxist/core/runtimes.py",
    "praxist/core/storage.py",
    "praxist/core/trajectory.py",
    "praxist/infrastructure/s3_utils.py",
    "praxist/plugins/agent_runtimes/claude_sdk/adapter.py",
    "praxist/plugins/agent_runtimes/claude_sdk/delete_guard.py",
    "praxist/plugins/budget_policies/default_basic/policy.py",
    "praxist/plugins/graph_maintainers/finding_graph_mvp/adapter.py",
    "praxist/plugins/tools/finding_graph_query/adapter.py",
    "praxist/plugins/tools/evaluation_tools/adapter.py",
    "praxist/plugins/tools/frontier_tools/adapter.py",
    "praxist/plugins/tools/memory_tools/adapter.py",
    "praxist/plugins/tools/prior_work_tools/adapter.py",
    "praxist/plugins/workflow_stages/research_loop/backend/agent.py",
    "praxist/plugins/workflow_stages/research_loop/backend/cohort_runner.py",
    "praxist/plugins/workflow_stages/research_loop/backend/exploration_bottleneck_detector.py",
    "praxist/plugins/workflow_stages/research_loop/backend/findings_collection.py",
    "praxist/plugins/workflow_stages/research_loop/backend/frontier.py",
    "praxist/plugins/workflow_stages/research_loop/backend/gems.py",
    "praxist/plugins/workflow_stages/research_loop/backend/generation_boundary.py",
    "praxist/plugins/workflow_stages/research_loop/backend/generation_loop.py",
    "praxist/plugins/workflow_stages/research_loop/backend/gpu_governor.py",
    "praxist/plugins/workflow_stages/research_loop/backend/hooks/log_tool_start.py",
    "praxist/plugins/workflow_stages/research_loop/backend/hooks/sync_to_s3.py",
    "praxist/plugins/workflow_stages/research_loop/backend/multi_pi/agenda_metadata.py",
    "praxist/plugins/workflow_stages/research_loop/backend/multi_pi/legacy_two_round_executor.py",
    "praxist/plugins/workflow_stages/research_loop/backend/multi_pi/pi_roles/_base_pi.py",
    "praxist/plugins/workflow_stages/research_loop/backend/multi_pi/pi_roles/external_validity_pi.py",
    "praxist/plugins/workflow_stages/research_loop/backend/multi_pi/pi_roles/portfolio_pi.py",
    "praxist/plugins/workflow_stages/research_loop/backend/multi_pi/pi_roles/skeptic_pi.py",
    "praxist/plugins/workflow_stages/research_loop/backend/orchestrator_runtime.py",
    "praxist/plugins/workflow_stages/research_loop/backend/orchestrator_status.py",
    "praxist/plugins/workflow_stages/research_loop/backend/prompt_context.py",
    "praxist/plugins/workflow_stages/research_loop/backend/research_memory/card_builder.py",
    "praxist/plugins/workflow_stages/research_loop/backend/research_memory/context_auditor.py",
    "praxist/plugins/workflow_stages/research_loop/backend/research_memory/context_firewall.py",
    "praxist/plugins/workflow_stages/research_loop/backend/research_memory/evidence_pack_builder.py",
    "praxist/plugins/workflow_stages/research_loop/backend/research_memory/ledgers/dissent_ledger.py",
    "praxist/plugins/workflow_stages/research_loop/backend/research_memory/ledgers/claim_ledger.py",
    "praxist/plugins/workflow_stages/research_loop/backend/research_memory/ledgers/frontier_delta_ledger.py",
    "praxist/plugins/workflow_stages/research_loop/backend/research_memory/ledgers/mechanism_ledger.py",
    "praxist/plugins/workflow_stages/research_loop/backend/research_memory/ledgers/negative_evidence_ledger.py",
    "praxist/plugins/workflow_stages/research_loop/backend/research_memory/ledgers/retired_claim_ledger.py",
    "praxist/plugins/workflow_stages/research_loop/backend/research_memory/metrics_logger.py",
    "praxist/plugins/workflow_stages/research_loop/backend/research_memory/source_resolver.py",
    "praxist/plugins/workflow_stages/research_loop/backend/research_memory_update.py",
    "praxist/plugins/workflow_stages/research_loop/backend/resume_state.py",
    "praxist/plugins/workflow_stages/research_loop/backend/synthesis_trigger.py",
    "praxist/plugins/workflow_stages/research_loop/backend/tools/atomic_io.py",
    "praxist/plugins/workflow_stages/research_loop/backend/tools/http_utils.py",
    "praxist/plugins/workflow_stages/research_loop/backend/tools/training_timeout.py",
    "praxist/run.py",
    "praxist/task_spec.py",
)

PROFILE_OMIT: dict[str, tuple[str, ...]] = {
    "unit": tuple(str(REPO_ROOT / relpath) for relpath in _UNIT_PROFILE_OMIT_RELATIVE),
    "integration": (),
}


def main(argv: list[str] | None = None) -> int:
    """Run selected test profiles and report line/branch coverage."""
    args = _parse_args(argv)
    coverage_module = _load_coverage_module()
    selected_profiles = _selected_profiles(args.profiles)
    failures = 0
    for profile in selected_profiles:
        modules = PROFILE_MODULES[profile]
        if not _run_profile(
            coverage_module=coverage_module,
            profile=profile,
            modules=modules,
            output_dir=Path(args.output_dir),
            fail_under=float(args.fail_under),
            fail_under_statements=float(args.fail_under_statements),
            verbosity=2 if args.verbose else 1,
            show_table=bool(args.show_table),
            show_missing=bool(args.show_missing),
            pytest_paths=PROFILE_PYTEST_PATHS.get(profile, ()),
        ):
            failures += 1
    return 1 if failures else 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Praxist unittest profiles with coverage.py. Defaults to unit and "
            "integration so the two fast coverage surfaces stay in sync."
        )
    )
    parser.add_argument(
        "profiles",
        nargs="*",
        choices=[*PROFILE_MODULES.keys(), "all"],
        default=["unit", "integration"],
        help="Coverage profile(s) to run. Use 'all' for unit + integration.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for coverage JSON/XML reports.",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=0.0,
        help=(
            "Fail when a selected profile's branch-aware total coverage percentage "
            "is below this value."
        ),
    )
    parser.add_argument(
        "--fail-under-statements",
        type=float,
        default=0.0,
        help=(
            "Fail when a selected profile's statement/line coverage percentage is below this value."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Use unittest verbosity=2.")
    parser.add_argument(
        "--show-table",
        action="store_true",
        help="Print the full per-file coverage table to stdout.",
    )
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Include missing line numbers when --show-table is enabled.",
    )
    return parser.parse_args(argv)


def _load_coverage_module() -> Any:
    try:
        import coverage  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "coverage.py is required for coverage runs. Install dev dependencies "
            "with `uv sync --group dev` or `python -m pip install coverage`."
        ) from exc
    return coverage


def _selected_profiles(profiles: list[str]) -> list[str]:
    if "all" in profiles:
        return list(PROFILE_MODULES)
    selected: list[str] = []
    for profile in profiles:
        if profile not in selected:
            selected.append(profile)
    return selected


def _run_profile(
    *,
    coverage_module: Any,
    profile: str,
    modules: tuple[str, ...],
    output_dir: Path,
    fail_under: float,
    fail_under_statements: float,
    verbosity: int,
    show_table: bool,
    show_missing: bool,
    pytest_paths: tuple[str, ...] = (),
) -> bool:
    data_file = REPO_ROOT / f".coverage.{profile}"
    cov = _new_coverage(coverage_module, profile=profile, data_file=data_file)
    cov.erase()
    print(f"\n=== coverage profile: {profile} ({', '.join(modules)}) ===", flush=True)
    cov.start()
    suite = _load_suite(modules)
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    cov.stop()
    cov.save()
    pytest_success = _run_pytest_paths(
        pytest_paths,
        data_file=data_file.with_name(f"{data_file.name}.pytest"),
        omit=PROFILE_OMIT.get(profile, ()),
    )
    if pytest_paths:
        _merge_coverage_data(
            coverage_module,
            profile=profile,
            target_file=data_file,
            source_file=data_file.with_name(f"{data_file.name}.pytest"),
        )

    profile_dir = output_dir / profile
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    report_buffer = io.StringIO()
    text_cov = _load_saved_coverage(coverage_module, profile=profile, data_file=data_file)
    total = text_cov.report(file=report_buffer, show_missing=show_missing)
    json_path = profile_dir / "coverage.json"
    json_cov = _load_saved_coverage(coverage_module, profile=profile, data_file=data_file)
    json_cov.json_report(outfile=str(json_path))
    coverage_totals = _coverage_totals(json_path)
    statement_total = float(
        coverage_totals.get(
            "percent_statements_covered", coverage_totals.get("percent_covered", total)
        )
    )
    branch_total = coverage_totals.get("percent_branches_covered")
    if show_table:
        print(report_buffer.getvalue(), end="")
    else:
        print(f"total coverage for {profile}: {total:.2f}%", flush=True)
        print(f"statement coverage for {profile}: {statement_total:.2f}%", flush=True)
        if branch_total is not None:
            print(f"branch coverage for {profile}: {float(branch_total):.2f}%", flush=True)
    xml_cov = _load_saved_coverage(coverage_module, profile=profile, data_file=data_file)
    xml_cov.xml_report(outfile=str(profile_dir / "coverage.xml"))
    print(f"coverage reports written to {profile_dir.relative_to(REPO_ROOT)}", flush=True)

    if fail_under and total < fail_under:
        print(
            f"coverage {total:.2f}% is below required threshold {fail_under:.2f}% "
            f"for profile {profile}",
            file=sys.stderr,
        )
        return False
    if fail_under_statements and statement_total < fail_under_statements:
        print(
            f"statement coverage {statement_total:.2f}% is below required threshold "
            f"{fail_under_statements:.2f}% for profile {profile}",
            file=sys.stderr,
        )
        return False
    return result.wasSuccessful() and pytest_success


def _new_coverage(coverage_module: Any, *, profile: str, data_file: Path) -> Any:
    return coverage_module.Coverage(
        config_file=str(REPO_ROOT / ".coveragerc"),
        data_file=str(data_file),
        omit=PROFILE_OMIT.get(profile, ()),
    )


def _load_saved_coverage(coverage_module: Any, *, profile: str, data_file: Path) -> Any:
    cov = _new_coverage(coverage_module, profile=profile, data_file=data_file)
    cov.load()
    return cov


def _coverage_totals(json_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    totals = payload.get("totals")
    return totals if isinstance(totals, dict) else {}


def _load_suite(modules: tuple[str, ...]) -> unittest.TestSuite:
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    for module_name in modules:
        suite.addTests(loader.loadTestsFromName(module_name))
    return suite


def _run_pytest_paths(
    paths: tuple[str, ...],
    *,
    data_file: Path,
    omit: tuple[str, ...],
) -> bool:
    if not paths:
        return True
    data_file.unlink(missing_ok=True)
    args = [str(REPO_ROOT / path) for path in paths]
    command = [
        sys.executable,
        "-m",
        "coverage",
        "run",
        f"--rcfile={REPO_ROOT / '.coveragerc'}",
    ]
    if omit:
        command.append(f"--omit={','.join(omit)}")
    command.extend(["-m", "pytest", *args, "--tb=short", "-q"])
    environment = dict(os.environ)
    environment["COVERAGE_FILE"] = str(data_file)
    completed = subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False)
    return completed.returncode == 0


def _merge_coverage_data(
    coverage_module: Any,
    *,
    profile: str,
    target_file: Path,
    source_file: Path,
) -> None:
    if not source_file.is_file():
        return
    target = _load_saved_coverage(coverage_module, profile=profile, data_file=target_file)
    source = _load_saved_coverage(coverage_module, profile=profile, data_file=source_file)
    target.get_data().update(source.get_data())
    target.save()
    source_file.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
