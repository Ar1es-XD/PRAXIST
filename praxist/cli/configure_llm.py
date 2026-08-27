"""``praxist configure-llm`` — persist user-level provider configuration."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from praxist.cli._env import AGENT_SYSTEM_TO_RUNTIME_REF, AGENT_SYSTEM_VALUES
from praxist.cli._setup_common import (
    SETUP_PROFILE_ENV_VAR,
    provider_key_var,
    provider_plugin_ref,
    provider_short_name,
    selected_config_file,
    write_env_file,
)
from praxist.cli._terminal_ui import (
    TerminalInteractionCancelled,
    TerminalInteractionError,
    read_masked_secret,
)

_MAX_API_KEY_LENGTH = 4096


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``praxist configure-llm`` subcommand."""
    parser = subparsers.add_parser(
        "configure-llm",
        help="Persist a built-in Praxist LLM provider profile.",
    )
    parser.add_argument(
        "--provider",
        required=True,
        help="Built-in provider name or compatible provider plugin reference.",
    )
    parser.add_argument("--model", default=None, help="Provider model name to persist.")
    parser.add_argument(
        "--agent-system",
        choices=AGENT_SYSTEM_VALUES,
        default=None,
        help="Agent runtime selection to persist.",
    )
    secret = parser.add_mutually_exclusive_group()
    secret.add_argument(
        "--api-key-stdin",
        action="store_true",
        help="Read the provider API key from stdin; a local TTY shows one * per character.",
    )
    secret.add_argument(
        "--api-key-env",
        default=None,
        help="Read the provider API key from this environment variable.",
    )
    secret.add_argument(
        "--no-api-key",
        action="store_true",
        help="Update non-secret provider settings without writing an API key.",
    )
    secret.add_argument(
        "--remove-api-key",
        action="store_true",
        help="Remove this provider's stored API key from the selected config file(s).",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Config file to update (default: $PRAXIST_CONFIG_FILE or the user config).",
    )
    parser.add_argument(
        "--project-env-file",
        default=None,
        help="Also write Praxist LLM config to this explicit task-local .env file.",
    )
    parser.add_argument(
        "--no-project-env",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--print-source-command",
        action="store_true",
        help="Print the shell command that loads the selected config file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit the result as JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report changes without writing files.",
    )
    parser.set_defaults(func=cmd_configure_llm)


def cmd_configure_llm(args: argparse.Namespace) -> int:
    """Write Praxist provider config to the user env file."""
    try:
        result = configure_llm(
            provider=args.provider,
            model=args.model,
            agent_system=args.agent_system,
            api_key_stdin=args.api_key_stdin,
            api_key_env=args.api_key_env,
            no_api_key=args.no_api_key,
            remove_api_key=args.remove_api_key,
            config_file=selected_config_file(args.config_file),
            project_config_file=(
                Path(args.project_env_file).expanduser()
                if args.project_env_file and not args.no_project_env
                else None
            ),
            dry_run=args.dry_run,
        )
    except ConfigureLLMCancelled as exc:
        print(f"praxist configure-llm: {exc}", file=sys.stderr)
        return 130
    except ConfigureLLMError as exc:
        print(f"praxist configure-llm: {exc}", file=sys.stderr)
        return 1

    if args.json_output:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        action = "would update" if args.dry_run else "updated"
        print(f"{action} Praxist LLM config: {result['config_file']}", file=sys.stderr)
        if result.get("project_config_file"):
            print(
                f"{action} Praxist project env: {result['project_config_file']}",
                file=sys.stderr,
            )
        print(f"  PRAXIST_LLM_PROVIDER={result['provider']}", file=sys.stderr)
        if result.get("agent_system"):
            print(f"  PRAXIST_AGENT_SYSTEM={result['agent_system']}", file=sys.stderr)
        if result.get("model"):
            print(f"  PRAXIST_MODEL={result['model']}", file=sys.stderr)
        if result.get("key_variable"):
            print(f"  {result['key_variable']}={result['key_status']}", file=sys.stderr)
        if args.print_source_command:
            print(
                f"source with: set -a; . {shlex.quote(result['config_file'])}; set +a",
                file=sys.stderr,
            )
    return 0


class ConfigureLLMError(RuntimeError):
    """Raised when LLM configuration cannot be written."""


class ConfigureLLMCancelled(ConfigureLLMError):
    """Raised when the operator cancels local credential input."""


def configure_llm(
    *,
    provider: str,
    model: str | None,
    agent_system: str | None,
    api_key_stdin: bool,
    api_key_env: str | None,
    no_api_key: bool,
    remove_api_key: bool = False,
    config_file: Path,
    project_config_file: Path | None,
    dry_run: bool,
    setup_profile: str | None = None,
) -> dict[str, str]:
    """Persist provider/model/key configuration without printing raw secrets."""
    try:
        provider_ref = provider_plugin_ref(provider)
    except ValueError as exc:
        raise ConfigureLLMError(str(exc)) from exc
    provider_name = provider_short_name(provider)
    try:
        key_var = provider_key_var(provider)
    except ValueError as exc:
        raise ConfigureLLMError(
            "configure-llm manages built-in provider profiles only. "
            "Configure a custom model_provider plugin through its documented "
            "task or host environment contract."
        ) from exc

    updates = {
        "PRAXIST_LLM_PROVIDER": provider_name,
        "PRAXIST_MODEL_PROVIDER_REF": provider_ref,
    }
    if agent_system:
        updates["PRAXIST_AGENT_SYSTEM"] = agent_system
        updates["PRAXIST_AGENT_RUNTIME_REF"] = AGENT_SYSTEM_TO_RUNTIME_REF[agent_system]
    if model:
        updates["PRAXIST_MODEL"] = model
    if setup_profile:
        updates[SETUP_PROFILE_ENV_VAR] = setup_profile

    key_status = "not written"
    if api_key_stdin:
        if dry_run:
            key_status = "would write"
        else:
            secret = _read_api_key_from_stdin(key_var)
            if not secret:
                raise ConfigureLLMError("--api-key-stdin received an empty key")
            updates[key_var] = secret
            key_status = "written"
    elif api_key_env:
        secret = os.environ.get(api_key_env, "")
        if not secret:
            raise ConfigureLLMError(f"--api-key-env {api_key_env} is not set or empty")
        updates[key_var] = secret
        key_status = "would write" if dry_run else "written"
    elif remove_api_key:
        key_status = "would remove" if dry_run else "removed"
    elif no_api_key:
        key_status = "unchanged"

    if not dry_run:
        remove_keys = {key_var} if remove_api_key else set()
        if not setup_profile:
            remove_keys.add(SETUP_PROFILE_ENV_VAR)
        try:
            write_env_file(config_file, updates, remove_keys=remove_keys)
            if project_config_file and project_config_file.resolve() != config_file.resolve():
                write_env_file(project_config_file, updates, remove_keys=remove_keys)
        except OSError as exc:
            raise ConfigureLLMError(f"could not write configuration: {exc}") from exc

    return {
        "config_file": str(config_file),
        "project_config_file": str(project_config_file) if project_config_file else "",
        "provider": provider_name,
        "agent_system": agent_system or "",
        "model": model or "",
        "setup_profile": setup_profile or "",
        "key_variable": key_var,
        "key_status": key_status,
        "dry_run": str(dry_run).lower(),
    }


def _read_api_key_from_stdin(key_var: str) -> str:
    """Read one bounded API key without exposing it in terminal history."""
    if sys.stdin.isatty():
        try:
            return read_masked_secret(
                f"Enter {key_var} (masked): ",
                input_stream=sys.stdin,
                output_stream=sys.stderr,
                max_length=_MAX_API_KEY_LENGTH,
            )
        except TerminalInteractionCancelled as exc:
            raise ConfigureLLMCancelled(f"{key_var} input was cancelled") from exc
        except TerminalInteractionError as exc:
            raise ConfigureLLMError(f"could not safely read {key_var}: {exc}") from exc
    print(f"Reading {key_var} from stdin...", file=sys.stderr)
    value = sys.stdin.readline(_MAX_API_KEY_LENGTH + 2)
    if value == "":
        raise ConfigureLLMError(f"--api-key-stdin reached EOF before reading {key_var}")
    if len(value) > _MAX_API_KEY_LENGTH + 1:
        raise ConfigureLLMError(
            f"--api-key-stdin exceeds the {_MAX_API_KEY_LENGTH}-character safety limit"
        )
    value = value.rstrip("\r\n")
    if len(value) > _MAX_API_KEY_LENGTH:
        raise ConfigureLLMError(
            f"--api-key-stdin exceeds the {_MAX_API_KEY_LENGTH}-character safety limit"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ConfigureLLMError("--api-key-stdin contains unsupported control characters")
    return value
