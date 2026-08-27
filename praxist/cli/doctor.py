"""``praxist doctor`` — read-only Praxist host readiness report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from praxist import __version__
from praxist.cli._env import AGENT_SYSTEM_VALUES, CODEX_NATIVE_DEFAULT_MODEL, PROVIDER_KEY_MAP
from praxist.cli._setup_common import (
    Check,
    bundled_skill_dirs,
    cli_checks,
    config_checks,
    default_codex_skills_dir,
    default_skills_dir,
    load_cli_environment,
    normalize_runtime_selection,
    platform_check,
    provider_key_var,
    provider_short_name,
    python_check,
    read_env_file,
    selected_config_file,
    version_checks,
    xdg_config_dir,
)
from praxist.cli.registry import state_dir


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``praxist doctor`` subcommand."""
    parser = subparsers.add_parser("doctor", help="Check Praxist host readiness.")
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit the readiness report as JSON.",
    )
    parser.add_argument(
        "--task-path",
        default=None,
        help="Also validate this task project and its runtime environment.",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Config file to inspect (default: $PRAXIST_CONFIG_FILE or the user config).",
    )
    parser.add_argument(
        "--agent-system",
        choices=AGENT_SYSTEM_VALUES,
        default=None,
        help="Check one research runtime (default: configured runtime or claude_sdk).",
    )
    parser.add_argument(
        "--model-provider",
        default=None,
        help="Check one model_provider ref using the same precedence as praxist start.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Check this selected model (Codex-native verifies it in the account catalog).",
    )
    parser.add_argument(
        "--codex-native",
        action="store_true",
        help="Check codex_sdk with native OpenAI and the saved ChatGPT login.",
    )
    parser.add_argument(
        "--target",
        choices=("auto", "codex", "claude"),
        default="auto",
        help="Check bundled skills for this agent host (default: detect managed installs).",
    )
    parser.add_argument(
        "--advisory",
        action="store_true",
        help="Always return exit 0 while retaining readiness failures in the report.",
    )
    parser.set_defaults(func=cmd_doctor)


def cmd_doctor(args: argparse.Namespace) -> int:
    """Run read-only diagnostics."""
    report = build_report(
        task_path=Path(args.task_path).expanduser() if args.task_path else None,
        target=args.target,
        agent_system=args.agent_system,
        model_provider_ref=args.model_provider,
        model=args.model,
        codex_native=args.codex_native,
        config_file=selected_config_file(args.config_file),
    )
    if args.json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)
    return 0 if report["ok"] or args.advisory else 1


def build_report(
    *,
    task_path: Path | None,
    target: str = "auto",
    agent_system: str | None = None,
    model_provider_ref: str | None = None,
    model: str | None = None,
    codex_native: bool = False,
    config_file: Path | None = None,
) -> dict[str, Any]:
    """Build a stable doctor report dictionary."""
    inspected_config = config_file or selected_config_file()
    _, persisted = read_env_file(inspected_config)
    persistent_configuration = {
        "agent_system": persisted.get("PRAXIST_AGENT_SYSTEM", ""),
        "provider": persisted.get("PRAXIST_LLM_PROVIDER", ""),
        "model": persisted.get("PRAXIST_MODEL", ""),
    }
    load_cli_environment(task_path, config_file=config_file)
    selection_error: Exception | None = None
    if codex_native:
        from praxist.cli import start

        try:
            start._select_codex_native_mode(
                agent_system=agent_system,
                runtime_ref=None,
                model_provider_ref=model_provider_ref,
            )
        except start.StartError as exc:
            selection_error = exc
        selected_agent = "codex_sdk"
        runtime_ref = "agent_runtime:codex_sdk"
        provider_ref = start.OPENAI_PROVIDER_REF
    else:
        try:
            selected_agent, runtime_ref = normalize_runtime_selection(
                agent_system=agent_system,
                runtime_ref=None,
                default_agent_system="claude_sdk",
            )
        except ValueError as exc:
            selection_error = exc
            selected_agent = str(
                agent_system or os.environ.get("PRAXIST_AGENT_SYSTEM") or "claude_sdk"
            )
            runtime_ref = ""
        from praxist.cli import start

        provider_ref = start._resolve_provider_ref(
            model_provider_ref,
            selected_agent,
        )
    if selection_error is not None:
        selection_check = Check("runtime_selection", "missing", str(selection_error))
    else:
        selection_check = Check(
            "runtime_selection",
            "ok",
            f"{selected_agent} -> {runtime_ref}",
        )
    provider = provider_short_name(provider_ref)
    selected_model = (
        str(model).strip()
        if model is not None
        else (
            CODEX_NATIVE_DEFAULT_MODEL
            if codex_native
            else os.environ.get("PRAXIST_MODEL", "").strip()
        )
    )
    auth_check = _provider_auth_check(
        selected_agent,
        provider=provider,
        require_saved_login=codex_native,
    )
    checks: list[Check] = [
        python_check(),
        platform_check(),
        *version_checks(),
        selection_check,
        *cli_checks(selected_agent),
        *config_checks(
            selected_agent,
            provider_override=provider,
            model_override=selected_model if codex_native or model is not None else None,
            saved_login_only=codex_native,
        ),
        auth_check,
        *_codex_route_checks(selected_agent, provider),
    ]
    if codex_native:
        checks.append(
            _codex_model_catalog_check(
                selected_model or CODEX_NATIVE_DEFAULT_MODEL,
                auth_ready=auth_check.status == "ok",
            )
        )
    checks.extend(_path_checks())
    checks.extend(_skill_checks(target))
    if task_path:
        checks.append(_task_path_check(task_path))

    required_failed = any(check.status == "missing" for check in checks)
    return {
        "ok": not required_failed,
        "agent_system": selected_agent,
        "runtime_ref": runtime_ref,
        "model_provider_ref": provider_ref,
        "auth_mode": "codex-native" if codex_native else "configured-provider",
        "diagnostic_scope": (
            "Codex-native readiness override; persistent configuration is unchanged"
            if codex_native
            else "configured runtime and provider"
        ),
        "persistent_configuration": persistent_configuration,
        "checks": [check.to_dict() for check in checks],
        "next_actions": _next_actions(checks),
    }


def _path_checks() -> list[Check]:
    return [
        Check("config_dir", "ok" if xdg_config_dir().exists() else "warn", str(xdg_config_dir())),
        Check(
            "registry_dir",
            "ok" if state_dir().exists() else "warn",
            str(state_dir()),
        ),
    ]


def _skill_checks(target: str) -> list[Check]:
    """Check bundled skill registrations for an explicit or detected host."""

    hosts = _detected_skill_hosts() if target == "auto" else (target,)
    return [check for host in hosts for check in _skill_host_checks(host)]


def _detected_skill_hosts() -> tuple[str, ...]:
    """Prefer managed installations without persisting a second host setting."""

    managed = tuple(
        host
        for host in ("codex", "claude")
        if (default_skills_dir(host) / ".praxist-skills.json").is_file()
    )
    if managed:
        return managed
    if os.environ.get("CLAUDE_SKILLS_DIR") and not os.environ.get("CODEX_SKILLS_DIR"):
        return ("claude",)
    return ("codex",)


def _skill_host_checks(host: str, *, target_dir: Path | None = None) -> list[Check]:
    skills = bundled_skill_dirs()
    target = target_dir or default_skills_dir(host)
    check_name = f"{host}_skills"
    if not skills:
        return [Check(check_name, "missing", "no bundled Praxist skills found")]
    installed = 0
    missing: list[str] = []
    stale: list[str] = []
    for source in skills:
        dest = target / source.name
        if not (dest / "SKILL.md").is_file():
            missing.append(source.name)
            continue
        installed += 1
        if dest.is_symlink():
            try:
                if dest.resolve(strict=True) != source.resolve(strict=True):
                    stale.append(source.name)
            except OSError:
                stale.append(source.name)
            continue
        marker = dest / ".praxist-skill.json"
        if not marker.exists():
            stale.append(source.name)
            continue
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stale.append(source.name)
            continue
        if (
            payload.get("managed_by") != "praxist"
            or payload.get("skill_name", source.name) != source.name
            or payload.get("version") != __version__
        ):
            stale.append(source.name)
    if missing:
        return [
            Check(
                check_name,
                "warn",
                f"{installed}/{len(skills)} installed in {target}; missing: {', '.join(missing)}",
            )
        ]
    if stale:
        return [
            Check(
                check_name,
                "warn",
                f"{installed}/{len(skills)} installed in {target}; unmarked: {', '.join(stale)}",
            )
        ]
    return [Check(check_name, "ok", f"{installed}/{len(skills)} installed in {target}")]


def _codex_skill_checks() -> list[Check]:
    """Compatibility wrapper for callers that check only Codex skills."""

    return _skill_host_checks("codex", target_dir=default_codex_skills_dir())


def _task_path_check(task_path: Path) -> Check:
    root = task_path.resolve()
    if not root.is_dir():
        return Check("task_path", "missing", str(root))
    if not (root / "task.yaml").is_file():
        return Check("task_path", "missing", f"{root} has no task.yaml")
    try:
        from praxist.task_spec import load_task_spec

        load_task_spec(root / "task.yaml")
    except Exception as exc:
        return Check("task_path", "missing", f"{root}/task.yaml is invalid: {exc}")
    return Check("task_path", "ok", str(root))


def _provider_auth_check(
    agent_system: str,
    *,
    provider: str,
    require_saved_login: bool = False,
) -> Check:
    if provider not in PROVIDER_KEY_MAP:
        return Check(
            "provider_auth",
            "warn",
            f"custom provider {provider!r}: authentication is plugin-managed",
        )
    key_var = provider_key_var(provider)
    if agent_system == "codex_sdk" and provider == "openai" and require_saved_login:
        codex_bin_override = (
            os.environ.pop("PRAXIST_CODEX_BIN", None) if require_saved_login else None
        )
        try:
            from praxist.plugins.agent_runtimes.codex_sdk._auth import verify_chatgpt_login

            verify_chatgpt_login()
        except Exception as exc:
            return Check("provider_auth", "missing", str(exc))
        finally:
            if require_saved_login and codex_bin_override is not None:
                os.environ["PRAXIST_CODEX_BIN"] = codex_bin_override
        return Check("provider_auth", "ok", "OpenAI via saved ChatGPT login")
    if os.environ.get(key_var):
        return Check("provider_auth", "ok", f"{provider}: {key_var} present", variable=key_var)
    if agent_system == "codex_sdk" and provider == "openai":
        return Check(
            "provider_auth",
            "warn",
            "not checked outside explicit Codex-native mode; use --codex-native to verify",
        )
    return Check(
        "provider_auth",
        "missing",
        f"{provider}: {key_var} is not set",
        variable=key_var,
    )


def _codex_route_checks(agent_system: str, provider: str) -> list[Check]:
    """Check dependencies used only by the selected Codex provider route."""

    if agent_system != "codex_sdk":
        return []
    from praxist.plugins.agent_runtimes.codex_sdk._relay import _relay_binary, needs_relay

    if not needs_relay(provider):
        return [Check("codex_relay", "ok", "not required for the native OpenAI route")]
    binary = _relay_binary()
    if binary:
        return [Check("codex_relay", "ok", f"{provider}: {binary}")]
    return [
        Check(
            "codex_relay",
            "missing",
            f"{provider}: codex-relay is required; install the Praxist codex extra",
        )
    ]


def _codex_model_catalog_check(model: str, *, auth_ready: bool) -> Check:
    """Verify the selected model through the same app-server path used at launch."""

    if not auth_ready:
        return Check(
            "codex_model_catalog",
            "warn",
            "not checked because saved ChatGPT authentication is unavailable",
        )
    codex_bin_override = os.environ.pop("PRAXIST_CODEX_BIN", None)
    try:
        from praxist.plugins.agent_runtimes.codex_sdk.adapter import (
            verify_chatgpt_model_available,
        )

        canonical = verify_chatgpt_model_available(model)
    except Exception as exc:
        return Check("codex_model_catalog", "missing", str(exc))
    finally:
        if codex_bin_override is not None:
            os.environ["PRAXIST_CODEX_BIN"] = codex_bin_override
    return Check("codex_model_catalog", "ok", canonical)


def _next_actions(checks: list[Check]) -> list[str]:
    actions: list[str] = []
    by_name = {check.name: check for check in checks}
    if by_name.get("PRAXIST_LLM_PROVIDER", Check("", "ok")).status == "warn":
        actions.append("praxist configure-llm --provider openrouter --api-key-stdin")
    if by_name.get("provider_auth", Check("", "ok")).status == "missing":
        var = by_name["provider_auth"].variable
        if not var:
            actions.append("repair the selected runtime/provider authentication, then rerun doctor")
        else:
            provider = provider_short_name(
                os.environ.get("PRAXIST_LLM_PROVIDER", "")
                or by_name["provider_auth"].detail.partition(":")[0]
                or "openrouter"
            )
            actions.append(f"praxist configure-llm --provider {provider} --api-key-stdin")
    if by_name.get("codex_model_catalog", Check("", "ok")).status == "missing":
        actions.append("reinstall the Praxist codex extra, then rerun Codex-native setup")
    if (
        by_name.get("codex_sdk", Check("", "ok")).status == "missing"
        or by_name.get("codex_relay", Check("", "ok")).status == "missing"
    ):
        actions.append("install the Praxist codex extra, then rerun: praxist doctor")
    for host in ("codex", "claude"):
        skill_check = by_name.get(f"{host}_skills", Check("", "ok"))
        if skill_check.status in {"warn", "missing"}:
            actions.append(f"praxist install-skills --target {host} --replace")
    return list(dict.fromkeys(actions))


def print_report(report: dict[str, Any]) -> None:
    """Render one readiness report for a human operator."""

    print("Praxist doctor", file=sys.stderr)
    print(
        f"  diagnostic scope   {report.get('diagnostic_scope', 'configured runtime')}",
        file=sys.stderr,
    )
    persistent = report.get("persistent_configuration") or {}
    if isinstance(persistent, dict) and any(persistent.values()):
        detail = " / ".join(
            str(persistent.get(key) or "unset") for key in ("agent_system", "provider", "model")
        )
        print(f"  persistent config  {detail}", file=sys.stderr)
    for raw in report["checks"]:
        check = raw if isinstance(raw, dict) else {}
        name = str(check.get("name", ""))
        status = str(check.get("status", ""))
        detail = str(check.get("detail", ""))
        variable = str(check.get("variable", ""))
        suffix = f" ({variable})" if variable else ""
        print(f"  {name:<18} {status:<7} {detail}{suffix}", file=sys.stderr)
    actions = report.get("next_actions") or []
    if actions:
        print("\nNext action:", file=sys.stderr)
        for action in actions:
            print(f"  {action}", file=sys.stderr)
