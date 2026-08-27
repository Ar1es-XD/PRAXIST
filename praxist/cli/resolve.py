"""``praxist resolve`` — task-project + plugin manifest resolution smoke test.

Convenience wrapper around ``python -m praxist.run run --task-path
<path> --resolve-only --local``: discovers and resolves the task
project's plugin manifest without making any LLM calls. Useful for:

* CI / sanity checks that a task contract is well-formed before
  committing to a paid run.
* Newcomer onboarding ("does my task project parse?").
* Pre-flight on a developer machine that has no provider credentials
  configured.

Standard task resolution builds an ``argparse.Namespace`` that
``praxist.run.cmd_run`` already understands, sets ``resolve_only=True`` and
``local=True``, and hands off. The optional ``--result-summary`` preflight
first checks one evaluator-produced summary against the same maturity parser
used at runtime, then follows that shared resolution path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from praxist.cli import start
from praxist.cli._setup_common import load_cli_environment, selected_config_file


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``praxist resolve`` subcommand on the parent parser."""
    parser = subparsers.add_parser(
        "resolve",
        help="Resolve a task project's plugin manifest (no LLM calls).",
        description=(
            "Discover and resolve a task project's plugin manifest without "
            "making any LLM calls. Equivalent to:\n\n"
            "    python -m praxist.run run --task-path <path> "
            "--resolve-only --local\n\n"
            "Exits non-zero on resolution failure (manifest schema error, "
            "missing plugin, etc.). On success, emits a JSON document on "
            "stdout summarizing the resolved run identity."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "task_path",
        nargs="?",
        default=".",
        help="Path to the task project directory (default: current directory).",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Config file to load (default: $PRAXIST_CONFIG_FILE or the user config).",
    )
    parser.add_argument(
        "--agent-system",
        choices=start.AGENT_SYSTEM_VALUES,
        default=None,
        help="Agent system used to resolve runtime/provider defaults.",
    )
    parser.add_argument(
        "--workspace",
        default="",
        help="Workspace directory (default: current working directory).",
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help=(
            "Override run artifact directory. Defaults to the task "
            "project's runtime_outputs.root / experiments directory; "
            "paths inside the Praxist source checkout are rejected."
        ),
    )
    parser.add_argument(
        "--runtime",
        default="",
        help="Override agent_runtime plugin ref.",
    )
    parser.add_argument(
        "--codex-native",
        action="store_true",
        help=(
            "Resolve with codex_sdk, native OpenAI, and saved ChatGPT login "
            "while ignoring API-key/custom-endpoint configuration."
        ),
    )
    parser.add_argument(
        "--model-provider",
        dest="model_provider",
        default="",
        help="Override model_provider plugin ref.",
    )
    parser.add_argument(
        "--budget-policy",
        dest="budget_policy",
        default="",
        help="Override budget_policy plugin ref.",
    )
    parser.add_argument(
        "--credential-profile",
        dest="credential_profile",
        default="",
        help="Override credential profile name (rarely needed for resolve-only).",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Override agent model. Not used by resolve-only itself, but propagated for parity.",
    )
    parser.add_argument(
        "--result-summary",
        default="",
        help=(
            "Validate one evaluator-produced JSON summary against the task's "
            "maturity telemetry contract before resolving."
        ),
    )
    parser.set_defaults(func=cmd_resolve)


def _validate_result_summary_contract(task_path: Path, summary_path: Path) -> None:
    """Validate actual evaluator output with the runtime maturity extractor."""

    from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
        missing_required_ratio_telemetry,
    )
    from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
        normalized_result_summary,
    )
    from praxist.task_spec import load_task_spec

    resolved_summary_path = summary_path.expanduser().resolve()
    try:
        payload = json.loads(resolved_summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"result summary is not valid JSON: {resolved_summary_path}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"result summary must contain a JSON object: {resolved_summary_path}")

    task_spec_path = task_path if task_path.name == "task.yaml" else task_path / "task.yaml"
    task_spec = load_task_spec(task_spec_path)
    summary = normalized_result_summary(payload, summary_path=resolved_summary_path)
    missing = missing_required_ratio_telemetry(
        summary,
        getattr(task_spec.evaluation, "maturity_policy", None),
    )
    if missing:
        fields = ", ".join(missing)
        raise ValueError(
            f"result summary {resolved_summary_path} cannot satisfy "
            "evaluation.maturity_policy.require_ratio_gate: missing finite "
            f"{fields} in a supported summary fact container"
        )


def cmd_resolve(args: argparse.Namespace) -> int:
    """Handler for ``praxist resolve`` — delegate to ``cmd_run`` with resolve flags."""
    try:
        task_path = start._resolve_task_path(args.task_path)
        load_cli_environment(
            task_path,
            config_file=selected_config_file(args.config_file),
        )
        start._validate_task_project(task_path)
        if args.result_summary:
            _validate_result_summary_contract(task_path, Path(args.result_summary))
        if getattr(args, "codex_native", False):
            start._select_codex_native_mode(
                agent_system=args.agent_system,
                runtime_ref=args.runtime,
                model_provider_ref=args.model_provider,
            )
            start._sanitize_codex_native_environment()
            args.agent_system = "codex_sdk"
            args.runtime = "agent_runtime:codex_sdk"
            args.model_provider = start.OPENAI_PROVIDER_REF
        agent_system, runtime_ref = start._resolve_runtime_selection(
            args.agent_system,
            args.runtime,
        )
        provider_ref = start._resolve_provider_ref(args.model_provider, agent_system)
        model = start._resolve_model(args.model, provider_ref, agent_system)
    except (OSError, start.StartError, ValueError) as exc:
        print(f"praxist resolve: {exc}", file=sys.stderr)
        return 1
    # cmd_run is the single source of truth for resolve-only semantics
    # (default run_dir, startup wiring, JSON output, exit codes). Build
    # a Namespace matching its expected attribute surface and hand off.
    from praxist.run import cmd_run

    cmd_run(
        argparse.Namespace(
            workspace=args.workspace,
            run_dir=args.run_dir,
            task_path=str(task_path),
            fake=False,
            task="",
            task_spec="",
            model=model,
            runtime=runtime_ref,
            model_provider=provider_ref,
            budget_policy=args.budget_policy,
            credential_profile=args.credential_profile,
            resolve_only=True,
            local=True,
            frontier_strategy="auto",
        )
    )
    return 0
