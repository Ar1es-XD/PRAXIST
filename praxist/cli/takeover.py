"""``praxist takeover`` - hand one selected project to an agent operator UI."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Any

from praxist.cli._env import agent_system_for_runtime_ref
from praxist.cli._setup_common import (
    provider_short_name,
    read_env_file,
    selected_config_file,
    task_env_file,
)
from praxist.cli._terminal_ui import (
    Choice,
    TerminalInteractionCancelled,
    TerminalInteractionError,
    confirm_action,
    interactive_terminal_available,
    read_visible_text,
    select_choice,
)
from praxist.cli.user_agreement import prompt_for_acceptance_if_needed
from praxist.user_agreement import current_acceptance


class TakeoverError(RuntimeError):
    """Raised when the operator handoff cannot be prepared safely."""


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``praxist takeover`` subcommand."""

    parser = subparsers.add_parser(
        "takeover",
        help="Open Codex or Claude Code and hand off a project to Praxist takeover.",
    )
    parser.add_argument(
        "--task-path",
        default=None,
        help="Research project to hand off (default: select locally or use the current directory).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--codex-native",
        action="store_true",
        help="Use the no-key Codex-native takeover skill.",
    )
    mode.add_argument(
        "--configured-provider",
        action="store_true",
        help="Use the configured-provider takeover skill.",
    )
    parser.add_argument(
        "--operator",
        choices=("codex", "claude"),
        default="codex",
        help="Agent CLI that hosts the takeover workflow. Default: codex.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Launch without the final Enter confirmation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the redacted handoff without starting the agent CLI.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit the redacted handoff as JSON.",
    )
    parser.set_defaults(func=cmd_takeover)


def cmd_takeover(args: argparse.Namespace) -> int:
    """Select a project and launch the matching complete agent takeover."""

    try:
        if args.json_output and not args.dry_run:
            raise TakeoverError(
                "--json requires --dry-run because the live handoff owns the terminal"
            )
        if not args.dry_run and current_acceptance() is None:
            if not interactive_terminal_available():
                raise TakeoverError(
                    "the Praxist License and User Agreement have not been accepted; run "
                    "`praxist user-agreement accept` in a local terminal"
                )
            if not prompt_for_acceptance_if_needed(output_stream=sys.stderr):
                raise TerminalInteractionCancelled("License and User Agreement were not accepted")
        while True:
            task_path, prepare_template = _select_task_path(args.task_path)
            if task_path is None:
                print(
                    "Praxist setup is complete; no research project was launched.",
                    file=sys.stderr,
                )
                return 0
            skill = None
            if prepare_template:
                command = _template_resolve_command(task_path)
            else:
                skill = _takeover_skill(
                    task_path=task_path,
                    force_codex_native=args.codex_native,
                    force_configured=args.configured_provider,
                )
                command = _operator_command(
                    task_path,
                    skill,
                    operator=args.operator,
                    require_installed=not args.dry_run,
                )
            payload: dict[str, Any] = {
                "task_path": str(task_path),
                "skill": skill,
                "operator": args.operator if skill is not None else None,
                "command": command,
                "launched": False,
            }
            if args.json_output:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print("Praxist first launch", file=sys.stderr)
                print(f"  project  {task_path}", file=sys.stderr)
                workflow = (
                    "offline task-resolution smoke test"
                    if skill is None
                    else (f"${skill}" if args.operator == "codex" else f"/{skill}")
                )
                print(f"  workflow {workflow}", file=sys.stderr)
            if args.dry_run:
                return 0
            if args.yes:
                break
            if not interactive_terminal_available():
                raise TakeoverError("a local terminal or --yes is required to launch the agent CLI")
            try:
                confirm_action(
                    (
                        "Run the offline task-resolution smoke test?"
                        if prepare_template
                        else f"Start the complete takeover in {args.operator}?"
                    ),
                    input_stream=sys.stdin,
                    output_stream=sys.stderr,
                )
            except TerminalInteractionCancelled:
                if args.task_path:
                    raise
                print("Returning to project selection.", file=sys.stderr)
                continue
            break
        if prepare_template:
            task_path = _prepare_template_project(task_path)
            command = _template_resolve_command(task_path)
        if prepare_template:
            return subprocess.run(command, check=False).returncode
        return subprocess.run(command, check=False, cwd=task_path).returncode
    except TerminalInteractionCancelled:
        print("Praxist takeover cancelled; configuration was preserved.", file=sys.stderr)
        return 130
    except (OSError, TakeoverError, TerminalInteractionError) as exc:
        print(f"praxist takeover: {exc}", file=sys.stderr)
        return 1


def _select_task_path(raw_path: str | None) -> tuple[Path | None, bool]:
    if raw_path:
        return _require_project_directory(Path(raw_path)), False
    if not interactive_terminal_available():
        return _require_project_directory(Path.cwd()), False

    while True:
        choice = select_choice(
            "Choose the first research project",
            (
                Choice("current", "Use the current project", str(Path.cwd())),
                Choice(
                    "template",
                    "Run the offline resolve smoke test",
                    "creates a toy-math fixture; it does not start real research",
                ),
                Choice("other", "Choose another project", "enter a project path"),
                Choice("finish", "Finish setup", "launch research later"),
            ),
            default=None,
            input_stream=sys.stdin,
            output_stream=sys.stderr,
        )
        if choice == "finish":
            return None, False
        if choice == "current":
            return _require_project_directory(Path.cwd()), False
        if choice == "template":
            destination = (Path.cwd() / "praxist-toy-math-demo").resolve()
            if destination.exists() and not (destination / "task.yaml").is_file():
                print(
                    f"Template destination already exists and is not a task: {destination}",
                    file=sys.stderr,
                )
                continue
            return destination, True
        try:
            entered = read_visible_text(
                "Project path: ",
                input_stream=sys.stdin,
                output_stream=sys.stderr,
            ).strip()
        except TerminalInteractionCancelled:
            print("Returning to project selection.", file=sys.stderr)
            continue
        if entered:
            try:
                return _require_project_directory(Path(entered)), False
            except TakeoverError as exc:
                print(f"Project not selected: {exc}", file=sys.stderr)


def _require_project_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise TakeoverError(f"research project directory does not exist: {resolved}")
    return resolved


def _prepare_template_project(destination: Path) -> Path:
    destination = destination.resolve()
    if destination.exists():
        if (destination / "task.yaml").is_file():
            return destination
        raise TakeoverError(f"template destination already exists and is not a task: {destination}")

    packaged = resources.files("praxist").joinpath("resources/templates/tasks/toy_math")
    source_tree = (
        packaged
        if packaged.is_dir()
        else Path(__file__).resolve().parents[2] / "templates/tasks/toy_math"
    )
    if not source_tree.is_dir():
        raise TakeoverError("the bundled guided template is unavailable")
    with resources.as_file(source_tree) as source:
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    return destination


def _takeover_skill(*, task_path: Path, force_codex_native: bool, force_configured: bool) -> str:
    if force_codex_native:
        return "praxist-takeover-codex"
    if force_configured:
        return "praxist-takeover"
    _, configured = read_env_file(selected_config_file())
    task_file = task_env_file(task_path)
    if task_file is not None:
        _, task_values = read_env_file(task_file)
        configured.update(task_values)

    def value(name: str) -> str:
        return (os.environ.get(name) or configured.get(name, "")).strip()

    runtime_ref = value("PRAXIST_AGENT_RUNTIME_REF") or value("RUNTIME_REF")
    agent_system = agent_system_for_runtime_ref(runtime_ref) or value("PRAXIST_AGENT_SYSTEM")
    agent_system = {"codex": "codex_sdk", "claude": "claude_sdk"}.get(agent_system, agent_system)
    provider_ref = value("PRAXIST_MODEL_PROVIDER_REF") or value("MODEL_PROVIDER_REF")
    provider = provider_short_name(provider_ref) if provider_ref else value("PRAXIST_LLM_PROVIDER")
    if agent_system == "codex_sdk" and provider == "openai":
        return "praxist-takeover-codex"
    return "praxist-takeover"


def _codex_command(task_path: Path, skill: str, *, require_installed: bool) -> list[str]:
    codex = (
        _bundled_codex_binary(require_installed=require_installed)
        if skill == "praxist-takeover-codex"
        else shutil.which("codex") or "codex"
    )
    if require_installed and codex == "codex":
        raise TakeoverError("Codex is not on PATH; install and authenticate Codex first")
    return [codex, "--yolo", "-C", str(task_path), _handoff_prompt(task_path, skill, "codex")]


def _operator_command(
    task_path: Path,
    skill: str,
    *,
    operator: str,
    require_installed: bool,
) -> list[str]:
    """Build a shell-free handoff for one supported interactive agent CLI."""

    if operator == "codex":
        return _codex_command(task_path, skill, require_installed=require_installed)
    if operator != "claude":
        raise TakeoverError(f"unsupported takeover operator: {operator}")
    claude = shutil.which("claude") or "claude"
    if require_installed and claude == "claude":
        raise TakeoverError("Claude Code is not on PATH; install and authenticate it first")
    return [
        claude,
        "--dangerously-skip-permissions",
        _handoff_prompt(task_path, skill, "claude"),
    ]


def _handoff_prompt(task_path: Path, skill: str, operator: str) -> str:
    invocation = f"${skill}" if operator == "codex" else f"/{skill}"
    return (
        f"Use the installed {invocation} skill to take over the research project at "
        f"{task_path}. Follow the complete installed workflow, preserve all operator "
        "choices, validate the task harness, and start the research run."
    )


def _template_resolve_command(task_path: Path) -> list[str]:
    return [sys.executable, "-m", "praxist", "resolve", str(task_path)]


def _bundled_codex_binary(*, require_installed: bool) -> str:
    """Resolve the package-pinned Codex used for a Codex-native handoff."""

    try:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import resolve_codex_binary

        binary = Path(resolve_codex_binary("codex")).resolve()
    except OSError:
        if require_installed:
            raise TakeoverError(
                "The package-pinned Codex CLI is unavailable; reinstall Praxist with "
                "the codex extra"
            ) from None
        return "codex"
    if require_installed and not binary.is_file():
        raise TakeoverError(
            "The package-pinned Codex CLI is unavailable; reinstall Praxist with the codex extra"
        )
    return str(binary)
