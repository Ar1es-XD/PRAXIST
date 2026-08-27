"""Translate Praxist sandbox intent into official Codex SDK settings."""

from __future__ import annotations

from dataclasses import dataclass, field

from praxist.core.protocol import AgentRunRequest, RuntimeSandboxIntent

_SHELL_TOOL_NAMES = frozenset({"Bash", "exec_command", "shell", "write_stdin"})
_READ_TOOL_NAMES = frozenset({"Read", "Glob", "Grep"})
_PATCH_TOOL_NAMES = frozenset({"Write", "Edit", "NotebookEdit", "apply_patch"})
_WEB_TOOL_NAMES = frozenset({"WebSearch", "WebFetch", "web_search"})
_IMAGE_TOOL_NAMES = frozenset({"view_image"})
_MULTI_AGENT_TOOL_NAMES = frozenset(
    {"Task", "spawn_agent", "send_input", "wait_agent", "close_agent"}
)


@dataclass(frozen=True)
class CodexSandboxSettings:
    """SDK enum names and app-server overrides for one request."""

    approval_mode: str
    sandbox: str
    config: dict[str, object] = field(default_factory=dict)


def sandbox_settings(request: AgentRunRequest) -> CodexSandboxSettings:
    """Resolve a validated runtime-neutral sandbox request."""

    intent = _sandbox_intent(request)
    if intent.approval != "auto":
        raise ValueError(
            "codex_sdk is headless and cannot honor interactive sandbox approval "
            f"mode {intent.approval!r}"
        )
    if intent.filesystem == "full" and intent.network == "off":
        raise ValueError(
            "codex_sdk cannot disable network access in a full-access filesystem sandbox"
        )
    approval_mode = "deny_all"
    sandbox = {
        "read_only": "read_only",
        "workspace_write": "workspace_write",
        "full": "full_access",
    }[intent.filesystem]
    config = _builtin_tool_config(request, intent)
    if intent.filesystem == "workspace_write":
        config["sandbox_workspace_write"] = {"network_access": intent.network == "on"}
    return CodexSandboxSettings(
        approval_mode=approval_mode,
        sandbox=sandbox,
        config=config,
    )


def _builtin_tool_config(
    request: AgentRunRequest,
    intent: RuntimeSandboxIntent,
) -> dict[str, object]:
    permissions = request.tool_permissions
    if permissions.mode == "allow_all":
        allowed: set[str] | None = None
    elif permissions.mode == "allow_list":
        allowed = set(permissions.allowed_tools)
    elif permissions.mode == "deny_list":
        allowed = None
    else:
        raise ValueError(f"unsupported ToolPermissionSet mode {permissions.mode!r}")
    denied = set(permissions.denied_tools)

    def enabled(names: frozenset[str]) -> bool:
        selected = allowed is None or bool(allowed.intersection(names))
        return selected and not bool(denied.intersection(names))

    shell = enabled(_SHELL_TOOL_NAMES)
    read_only_shell = enabled(_READ_TOOL_NAMES) and intent.filesystem == "read_only"
    if (
        allowed is not None
        and allowed.intersection(_READ_TOOL_NAMES)
        and not (shell or read_only_shell)
    ):
        raise ValueError(
            "codex_sdk cannot expose Read/Glob/Grep without either Bash permission "
            "or a read-only filesystem sandbox"
        )
    return {
        "features": {
            "shell_tool": shell or read_only_shell,
            "multi_agent": enabled(_MULTI_AGENT_TOOL_NAMES),
        },
        "include_apply_patch_tool": enabled(_PATCH_TOOL_NAMES),
        "tools": {"view_image": enabled(_IMAGE_TOOL_NAMES)},
        "web_search": (
            "live" if intent.network == "on" and enabled(_WEB_TOOL_NAMES) else "disabled"
        ),
    }


def _sandbox_intent(request: AgentRunRequest) -> RuntimeSandboxIntent:
    raw = request.runtime_options.get("sandbox_intent") if request.runtime_options else None
    if isinstance(raw, dict):
        return RuntimeSandboxIntent(
            filesystem=str(raw.get("filesystem", "full")),  # type: ignore[arg-type]
            network=str(raw.get("network", "on")),  # type: ignore[arg-type]
            approval=str(raw.get("approval", "auto")),  # type: ignore[arg-type]
        )
    return RuntimeSandboxIntent(filesystem="full", network="on", approval="auto")


__all__ = ["CodexSandboxSettings", "sandbox_settings"]
