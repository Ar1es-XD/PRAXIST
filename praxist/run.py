"""
Praxist — Unified CLI launcher.

Subcommands:
    run       Run multi-generation research loop
    peer       Run a single autonomous peer (for Docker/RunPod entrypoints)
    server    Start the orchestrator dashboard
"""

import argparse
import inspect
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path


def _default_run_dir_for_task_project(task_project, workspace: Path, task_ref: str) -> Path:
    """Choose the default run_dir for a resolved task project.

    Task projects can opt into keeping all run artifacts beside the task repo
    by setting ``runtime_outputs.root`` in ``task.yaml``. Relative paths are
    resolved under the task project root. Projects without that field use the
    task-local ``experiments/`` directory. The default never falls back to the
    Praxist source checkout.
    """
    from datetime import datetime

    task_slug = task_ref.split(":", 1)[1]
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_cfg = task_project.descriptor.get("runtime_outputs") or {}
    root_value = output_cfg.get("root") if isinstance(output_cfg, dict) else None
    root = Path(str(root_value or "experiments"))
    if not root.is_absolute():
        root = task_project.path / root
    return root / f"run_{ts}_{task_slug}"


def _default_run_dir_for_fake_fixture(task_ref: str) -> Path:
    """Choose a non-source-tree default run_dir for CLI fake fixture runs."""
    from datetime import datetime

    task_slug = task_ref.split(":", 1)[1] if ":" in task_ref else task_ref
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path(tempfile.gettempdir()) / "praxist" / "fake_runs" / f"run_{ts}_{task_slug}"


def _ensure_run_dir_not_in_source_checkout(run_dir: Path) -> None:
    """Reject run outputs that would be written into the Praxist source checkout."""
    repo_root = Path(__file__).resolve().parents[1]
    resolved = Path(run_dir).expanduser().resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        return
    raise ValueError(
        f"run_dir must live outside the Praxist source checkout: {resolved}. "
        "Pass an external --run-dir or use the task project's experiments directory."
    )


def _install_research_loop_signal_finalizer(prepared, finalize_research_loop_plugin_run):
    """Materialize canonical ledgers before a foreground research-loop run exits on TERM/INT."""
    import signal

    installed = {}
    finalized = {"done": False}

    def _handler(signum, _frame):
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = str(signum)
        exit_code = 128 + int(signum)
        if finalized["done"]:
            os._exit(exit_code)
        finalized["done"] = True
        try:
            finalize_research_loop_plugin_run(
                prepared,
                success=False,
                result={
                    "run_dir": str(prepared.run_dir),
                    "exit_condition": f"signal_{signal_name.lower()}",
                },
                error=f"terminated by {signal_name}",
                exit_code=exit_code,
            )
        except BaseException as exc:  # noqa: BLE001 - signal shutdown must still terminate.
            print(f"signal finalization failed: {exc}", file=sys.stderr)
        os._exit(exit_code)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            installed[sig] = signal.getsignal(sig)
            signal.signal(sig, _handler)
        except (OSError, ValueError):
            continue

    def _restore():
        for sig, previous in installed.items():
            try:
                signal.signal(sig, previous)
            except (OSError, ValueError):
                continue

    return _restore


def _prompt_for_product_usage_consent() -> None:
    """Offer consent without allowing the optional observer to block a Run."""

    try:
        from praxist.cli.product_usage import prompt_for_consent_if_unset

        prompt_for_consent_if_unset()
    except Exception:  # Product usage must never become a Run prerequisite.
        logging.getLogger(__name__).debug("product-usage consent prompt failed", exc_info=True)


def _start_product_usage_observer(run_dir: Path, *, planned_peer_count: int):
    """Create and notify the failure-isolated lifecycle observer."""

    try:
        from praxist import __version__
        from praxist.infrastructure.product_usage import ProductUsageObserver
        from praxist.plugins.workflow_stages.research_loop.lifecycle import (
            PeerLifecycleSummary,
            record_run_started_safely,
        )

        observer = ProductUsageObserver.create(
            run_dir=run_dir,
            praxist_version=__version__,
        )
        record_run_started_safely(
            observer,
            PeerLifecycleSummary.planned(
                generation_ordinal=0,
                planned_peer_count=planned_peer_count,
            ),
        )
        return observer
    except Exception:  # Product usage must never become a Run prerequisite.
        logging.getLogger(__name__).debug("product-usage observer setup failed", exc_info=True)
        return None


def _finish_product_usage_observer(
    observer,
    *,
    active_duration_seconds: float | None,
    failed: bool,
) -> None:
    """Finish and close the observer without changing Run control flow."""
    try:
        from praxist.plugins.workflow_stages.research_loop.lifecycle import (
            close_observer_safely,
            record_run_finished_safely,
        )

        record_run_finished_safely(
            observer,
            active_duration_seconds=active_duration_seconds,
            failed=failed,
        )
        close_observer_safely(observer)
    except Exception:  # Product usage must never replace Run termination.
        logging.getLogger(__name__).debug("product-usage observer finish failed", exc_info=True)


def _runner_accepts_lifecycle_observer(run_task) -> bool:
    """Require an explicit opt-in before extending a task-runner call."""

    try:
        parameters = inspect.signature(run_task).parameters
    except (TypeError, ValueError):
        return False
    return "run_lifecycle_observer" in parameters


def _task_project_planned_peer_count(task_project) -> int | None:
    """Read an explicit cohort size without guessing for custom task runners."""

    descriptor = getattr(task_project, "descriptor", None)
    if not isinstance(descriptor, dict):
        return None
    generation_policy = descriptor.get("generation_policy")
    if not isinstance(generation_policy, dict):
        return None
    value = generation_policy.get("cohort_size")
    if type(value) is not int or value < 1:
        return None
    return value


def cmd_run(args):
    """Run the full multi-generation research loop."""
    workspace = Path(args.workspace) if args.workspace else Path.cwd()
    resume_from = Path(args.resume_from) if getattr(args, "resume_from", "") else None
    run_dir = Path(args.run_dir) if args.run_dir else resume_from
    resume_enabled = bool(getattr(args, "resume", False) or resume_from is not None)
    resume_policy = getattr(args, "resume_policy", "completed_generation")
    if resume_from is not None and args.run_dir:
        explicit_run_dir = Path(args.run_dir).expanduser().resolve()
        explicit_resume_from = resume_from.expanduser().resolve()
        if explicit_run_dir != explicit_resume_from:
            print(
                "--resume-from and --run-dir refer to different directories; "
                "pass only --resume-from or make them identical",
                file=sys.stderr,
            )
            sys.exit(2)
    task_path_arg = getattr(args, "task_path", "")
    if args.fake and not task_path_arg:
        from praxist.testing.fake_workflow_fixture import run_fake_workflow_fixture

        task_ref = args.task or "task:fake_panel"
        if run_dir is None:
            run_dir = _default_run_dir_for_fake_fixture(task_ref)
        _ensure_run_dir_not_in_source_checkout(run_dir)
        result = run_fake_workflow_fixture(
            workspace=workspace,
            task_ref=task_ref,
            run_dir=run_dir,
            runtime_ref=args.runtime or "agent_runtime:fake_runtime",
            model_provider_ref=args.model_provider or "model_provider:fake_provider",
            budget_policy_ref=args.budget_policy or "budget_policy:fake_tiered",
            credential_profile=args.credential_profile,
            resolve_only=args.resolve_only,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if task_path_arg:
        from praxist.core.task_project import (
            load_task_project_runner,
            resolve_task_project,
            task_project_has_capability,
        )
        from praxist.plugins.workflow_stages.research_loop.startup import (
            default_budget_policy_for_task,
            default_model_provider_for_task,
            default_runtime_for_task,
            finalize_research_loop_plugin_run,
            is_research_loop_task_project,
            prepare_research_loop_plugin_run,
        )

        task_project = resolve_task_project(task_path_arg, workspace=workspace)
        if args.task and args.task != task_project.task_ref:
            print(
                f"task ref mismatch: --task {args.task} does not match {task_project.task_ref}",
                file=sys.stderr,
            )
            sys.exit(2)
        task_ref = task_project.task_ref
        fake_workflow_fixture_task = task_project_has_capability(
            task_project, "testing.fake_workflow_fixture"
        )
        if run_dir is None:
            run_dir = _default_run_dir_for_task_project(task_project, workspace, task_ref)
        if fake_workflow_fixture_task:
            runtime_ref = args.runtime or "agent_runtime:fake_runtime"
            model_provider_ref = args.model_provider or "model_provider:fake_provider"
            budget_policy_ref = args.budget_policy or "budget_policy:fake_tiered"
            _ensure_run_dir_not_in_source_checkout(run_dir)
            run_task = _task_runner_for_capability(
                task_project, "testing.fake_workflow_fixture", load_task_project_runner
            )

            if not args.resolve_only:
                _prompt_for_product_usage_consent()
            planned_peer_count = _task_project_planned_peer_count(task_project)
            run_lifecycle_observer = None
            if not args.resolve_only and planned_peer_count is not None:
                run_lifecycle_observer = _start_product_usage_observer(
                    run_dir,
                    planned_peer_count=planned_peer_count,
                )
            lifecycle_started_at = time.monotonic()
            result = None
            try:
                run_kwargs = {
                    "workspace": workspace,
                    "task_ref": task_ref,
                    "task_project": task_project,
                    "run_dir": run_dir,
                    "runtime_ref": runtime_ref,
                    "model_provider_ref": model_provider_ref,
                    "budget_policy_ref": budget_policy_ref,
                    "credential_profile": args.credential_profile,
                    "resolve_only": args.resolve_only,
                }
                if run_lifecycle_observer is not None and _runner_accepts_lifecycle_observer(
                    run_task
                ):
                    run_kwargs["run_lifecycle_observer"] = run_lifecycle_observer
                result = run_task(
                    **run_kwargs,
                )
            finally:
                _finish_product_usage_observer(
                    run_lifecycle_observer,
                    active_duration_seconds=time.monotonic() - lifecycle_started_at,
                    failed=not isinstance(result, dict)
                    or result.get("status") not in {"succeeded", "resolved"},
                )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        runtime_ref = default_runtime_for_task(task_ref, args.runtime)
        model_provider_ref = default_model_provider_for_task(task_ref, args.model_provider)
        budget_policy_ref = default_budget_policy_for_task(task_ref, args.budget_policy)
        if is_research_loop_task_project(task_project.path, workspace=workspace):
            from praxist.plugins.workflow_stages.research_loop.stage import (
                ResearchLoopStageContext,
                run_research_loop_stage,
            )

            if run_dir is None:
                run_dir = _default_run_dir_for_task_project(task_project, workspace, task_ref)
            _ensure_run_dir_not_in_source_checkout(run_dir)
            if not args.resolve_only:
                _prompt_for_product_usage_consent()
            try:
                prepared = prepare_research_loop_plugin_run(
                    task_ref=task_ref,
                    task_project=task_project,
                    workspace=workspace,
                    run_dir=run_dir,
                    runtime_ref=runtime_ref,
                    model_provider_ref=model_provider_ref,
                    budget_policy_ref=budget_policy_ref,
                    model=args.model or "",
                    local_mode=args.local,
                    frontier_strategy=args.frontier_strategy,
                    credential_profile=args.credential_profile,
                    command=" ".join(sys.argv),
                    resolve_only=args.resolve_only,
                    resume=resume_enabled,
                    resume_policy=resume_policy,
                )
            except Exception as exc:
                print(f"startup failed: {exc}", file=sys.stderr)
                sys.exit(3)
            if not prepared.stage_budget_grant_id:
                error = "budget gate did not grant research_loop stage"
                finalize_research_loop_plugin_run(prepared, success=False, error=error, exit_code=5)
                print(error, file=sys.stderr)
                sys.exit(5)
            run_lifecycle_observer = None
            if not args.resolve_only:
                run_lifecycle_observer = _start_product_usage_observer(
                    prepared.run_dir,
                    planned_peer_count=getattr(
                        getattr(prepared.task_spec, "generation_policy", None),
                        "cohort_size",
                        0,
                    ),
                )
            stage_context = ResearchLoopStageContext(
                task_spec=prepared.task_spec,
                workspace=prepared.task_execution_cwd,
                run_dir=prepared.run_dir,
                local_mode=args.local,
                model=prepared.startup_config["canonical_args"]["model"],
                runtime_ref=prepared.runtime_ref,
                model_provider_ref=prepared.model_provider_ref,
                frontier_strategy=args.frontier_strategy,
                budget_grant_id=prepared.stage_budget_grant_id,
                model_provider_credential_key_id=prepared.model_provider_credential_key_id,
                provider_env=prepared.provider_env,
                tool_server_refs=prepared.tool_server_refs,
                plugin_registry=prepared.registry,
                resolve_only=args.resolve_only,
                resume=resume_enabled,
                resume_policy=resume_policy,
                task_project_path=prepared.task_project_path,
                peer_role_ref=getattr(prepared, "peer_role_ref", None),
                peer_role_refs=tuple(getattr(prepared, "peer_role_refs", ()) or ()),
                run_lifecycle_observer=run_lifecycle_observer,
            )
            lifecycle_started_at = time.monotonic()
            stage_result = None
            result = None
            try:
                restore_signal_finalizer = _install_research_loop_signal_finalizer(
                    prepared,
                    finalize_research_loop_plugin_run,
                )
                try:
                    try:
                        stage_result = run_research_loop_stage(stage_context)
                    except Exception as exc:
                        finalize_research_loop_plugin_run(prepared, success=False, error=str(exc))
                        raise
                finally:
                    restore_signal_finalizer()
                result = stage_result.summary
                try:
                    finalize_research_loop_plugin_run(
                        prepared,
                        success=stage_result.success,
                        result=result,
                        error=stage_result.error,
                    )
                except Exception as exc:
                    finalize_research_loop_plugin_run(
                        prepared, success=False, error=f"finalize failed: {exc}"
                    )
                    print(f"finalize failed: {exc}", file=sys.stderr)
                    sys.exit(1)
            finally:
                observed_duration = (
                    result.get("total_duration_seconds") if isinstance(result, dict) else None
                )
                _finish_product_usage_observer(
                    run_lifecycle_observer,
                    active_duration_seconds=(
                        observed_duration
                        if isinstance(observed_duration, int | float)
                        else time.monotonic() - lifecycle_started_at
                    ),
                    failed=stage_result is None or not stage_result.success,
                )
            if not stage_result.success:
                print(stage_result.error or "research_loop stage failed", file=sys.stderr)
                sys.exit(1)
            if args.resolve_only:
                print(
                    json.dumps(
                        {
                            "run_id": prepared.run_id,
                            "run_dir": str(prepared.run_dir),
                            "status": "resolved",
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return
            print(f"\nRun complete: {result.get('generations_completed', 0)} generations")
            print(f"Run directory: {result.get('run_dir', '')}")
            return
        print(f"Unsupported task project workflow: {task_project.path}", file=sys.stderr)
        sys.exit(2)

    if args.task_spec:
        print(
            "ERROR: direct --task-spec execution is disabled; use --task-path <task-project> "
            "so startup, plugin resolution, budget, credentials, and replay artifacts are created.",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.task:
        print(
            f"ERROR: task refs are no longer discovered from bundled plugins ({args.task}); "
            "use --task-path <external-task-project>.",
            file=sys.stderr,
        )
        sys.exit(2)
    print("ERROR: --task-path is required", file=sys.stderr)
    sys.exit(2)


def _task_runner_for_capability(task_project, capability: str, loader):
    """Load a task project runner for lightweight task-owned execution modes."""
    try:
        runner = loader(task_project)
    except ValueError:
        from praxist.testing.fake_workflow_fixture import FakeWorkflowFixtureTaskRunner

        runner = FakeWorkflowFixtureTaskRunner(task_project)
    if hasattr(runner, "run"):
        return runner.run
    if callable(runner):
        return runner
    raise TypeError(f"{task_project.task_ref} entrypoint did not return a callable task runner")


def cmd_peer(args):
    """Run a single autonomous agent peer."""
    from praxist.infrastructure.execute_autonomous import main as peer_main

    # Set environment variables from args
    if args.peer_id:
        os.environ["PEER_ID"] = args.peer_id
    if args.generation_id is not None:
        os.environ["GENERATION_ID"] = str(args.generation_id)
    if args.max_runtime:
        os.environ["MAX_RUNTIME_SECONDS"] = str(args.max_runtime)
    if args.prompt_file:
        os.environ["TASK_PROMPT_FILE"] = args.prompt_file
    if args.model:
        os.environ["AGENT_MODEL"] = args.model
    if args.local:
        os.environ["LOCAL_MODE"] = "true"

    peer_main()


def cmd_server(args):
    """Start the orchestrator Flask server."""
    print("Server mode is not yet implemented in the Praxist package.")
    print("A web dashboard can be added as a future workflow-stage operator surface.")
    sys.exit(1)


def cmd_replay(args):
    """Inspect or verify an existing run directory."""
    from praxist.core.replay import dry_run, inspect_run, verify_run

    run_dir = Path(args.run_dir)
    if args.mode == "inspect":
        report = inspect_run(run_dir)
    elif args.mode == "verify":
        report = verify_run(
            run_dir,
            strict_tail=args.strict_tail,
            allow_plugin_drift=args.allow_plugin_drift,
            locked=args.locked,
        )
    else:
        report = dry_run(
            run_dir,
            strict_tail=args.strict_tail,
            allow_plugin_drift=args.allow_plugin_drift,
            locked=args.locked,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report.get("success", True):
        sys.exit(1)


def cmd_parity(args):
    """Verify research_loop dogfood parity for an existing run directory."""
    from praxist.plugins.workflow_stages.research_loop.backend.parity import (
        verify_research_loop_parity,
    )

    report = verify_research_loop_parity(
        Path(args.run_dir),
        deliverables_dir=Path(args.deliverables_dir) if args.deliverables_dir else None,
        strict=args.strict,
        write_report=args.write_report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report.get("success", True):
        sys.exit(1)


def main():
    """CLI entrypoint for running, replaying, and inspecting Praxist research runs."""
    parser = argparse.ArgumentParser(
        description="Praxist — Multi-generation autonomous research system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # --- run ---
    p_run = subparsers.add_parser("run", help="Run multi-generation research loop")
    p_run.add_argument(
        "--task", default="", help="Deprecated task ref; use --task-path for real tasks"
    )
    p_run.add_argument("--task-path", default="", help="Explicit external task project directory")
    p_run.add_argument(
        "--task-spec", default="", help="Deprecated legacy path; direct execution is disabled"
    )
    p_run.add_argument("--workspace", default="", help="Workspace directory (default: repo root)")
    p_run.add_argument(
        "--model", default="", help="Agent model (default: selected provider default)"
    )
    p_run.add_argument("--runtime", default="")
    p_run.add_argument("--model-provider", default="")
    p_run.add_argument("--budget-policy", default="")
    p_run.add_argument("--credential-profile", default="")
    p_run.add_argument(
        "--run-dir",
        default="",
        help=(
            "Run artifact directory. Defaults to the task project's runtime_outputs.root "
            "or experiments/ directory; paths inside the Praxist source checkout are rejected."
        ),
    )
    p_run.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a previous research-loop run in --run-dir from the safest "
            "completed generation boundary. Without this flag, existing Praxist "
            "run artifacts remain rejected."
        ),
    )
    p_run.add_argument(
        "--resume-from",
        default="",
        help=(
            "Path to a previous experiments/run_* directory to resume. "
            "Equivalent to --run-dir <path> --resume."
        ),
    )
    p_run.add_argument(
        "--resume-policy",
        default="completed_generation",
        choices=["completed_generation"],
        help="Resume policy. completed_generation skips only durably completed generations.",
    )
    p_run.add_argument("--fake", action="store_true", help="Run the fake workflow fixture")
    p_run.add_argument(
        "--resolve-only",
        action="store_true",
        help="Resolve task project and write startup artifacts without executing the workflow",
    )
    p_run.add_argument(
        "--local", action="store_true", help="Local mode (no S3, no multi-peer sync)"
    )
    p_run.add_argument(
        "--frontier-strategy",
        default="auto",
        choices=["explore", "exploit", "mixed", "auto"],
        help=(
            "Research-loop strategy override. Default 'auto' means gen 0 "
            "free-explore, then PI-directed per-peer role contracts. Explicit "
            "'mixed' / 'exploit' / 'explore' is a legacy operator override "
            "for the whole run and bypasses the default PI-directed schedule."
        ),
    )
    p_run.set_defaults(func=cmd_run)

    # --- peer ---
    p_peer = subparsers.add_parser("peer", help="Run a single autonomous peer")
    p_peer.add_argument("--peer-id", default="peer_0")
    p_peer.add_argument("--generation-id", type=int, default=0)
    p_peer.add_argument("--max-runtime", type=int, default=24 * 3600)
    p_peer.add_argument("--prompt-file", default="", help="Path to rendered prompt file")
    p_peer.add_argument("--model", default="")
    p_peer.add_argument("--local", action="store_true")
    p_peer.set_defaults(func=cmd_peer)

    # --- server ---
    p_server = subparsers.add_parser("server", help="Start orchestrator dashboard")
    p_server.add_argument("--port", type=int, default=8000)
    p_server.set_defaults(func=cmd_server)

    # --- replay ---
    p_replay = subparsers.add_parser("replay", help="Inspect or verify a run_dir")
    p_replay.add_argument("run_dir", help="Path to run directory")
    p_replay.add_argument(
        "--mode",
        default="inspect",
        choices=["inspect", "verify", "dry-run"],
        help="Replay mode",
    )
    p_replay.add_argument("--allow-plugin-drift", action="store_true")
    p_replay.add_argument("--strict-tail", action="store_true")
    p_replay.add_argument(
        "--locked",
        action="store_true",
        help="Fail on source/plugin drift and orphan artifacts for benchmark/release artifacts",
    )
    p_replay.set_defaults(func=cmd_replay)

    # --- parity ---
    p_parity = subparsers.add_parser("parity", help="Verify research_loop dogfood parity")
    p_parity.add_argument("run_dir", help="Path to run directory")
    p_parity.add_argument(
        "--deliverables-dir",
        default="",
        help="Optional deliverables package directory to verify",
    )
    p_parity.add_argument(
        "--strict",
        action="store_true",
        help="Treat missing optional dogfood surfaces as failures",
    )
    p_parity.add_argument(
        "--write-report",
        action="store_true",
        help="Write research_loop_parity_report.json into the run directory",
    )
    p_parity.set_defaults(func=cmd_parity)

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
