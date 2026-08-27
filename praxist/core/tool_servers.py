"""Core ToolServer selection, construction, and result normalization.

Concrete MCP factories and handlers live under ``praxist/plugins/tools/*``.
This module owns registry-backed selection, mode gates, allowed-tool naming,
and normalized tool-call accounting.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from praxist.core.execution_guards import BudgetedActionGuard
from praxist.core.protocol import ToolCallResult, ToolServerRef
from praxist.core.redaction import redact_json, redact_text
from praxist.core.registry import (
    PluginLoader,
    PluginRef,
    PluginRegistry,
    PluginRoots,
    SelectedPlugin,
    require_execution_plugin,
)

BASE_PEER_TOOLS = (
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Glob",
    "Grep",
    "WebSearch",
    "WebFetch",
)

LITERATURE_LOOKUP_TOOL_SERVER_REF = "tool_server:literature_lookup"
RUN_REPORT_TOOL_SERVER_REF = "tool_server:run_report"
LITERATURE_LOOKUP_SERVER_NAME = "literature-lookup"
LITERATURE_LOOKUP_TOOL_NAMES = (
    "literature_search",
    "literature_resolve",
    "literature_source_guide",
    "literature_open_access_text",
    "scientific_database_search",
)
LITERATURE_LOOKUP_MCP_TOOL_NAMES = tuple(
    f"mcp__{LITERATURE_LOOKUP_SERVER_NAME}__{tool_name}"
    for tool_name in LITERATURE_LOOKUP_TOOL_NAMES
)

DEFAULT_RESEARCH_TOOL_SERVER_REFS = (
    "tool_server:evaluation_tools",
    "tool_server:frontier_tools",
    "tool_server:finding_graph_query",
    "tool_server:memory_tools",
    "tool_server:prior_work_tools",
    RUN_REPORT_TOOL_SERVER_REF,
    LITERATURE_LOOKUP_TOOL_SERVER_REF,
)

DEFAULT_PEER_TOOL_SERVER_REFS = (
    "tool_server:evaluation_tools",
    "tool_server:frontier_tools",
    "tool_server:finding_graph_query",
    "tool_server:prior_work_tools",
    LITERATURE_LOOKUP_TOOL_SERVER_REF,
)

PANEL_TOOL_SERVER_REFS = (
    "tool_server:evaluation_tools",
    "tool_server:frontier_tools",
    "tool_server:finding_graph_query",
    "tool_server:memory_tools",
    RUN_REPORT_TOOL_SERVER_REF,
    LITERATURE_LOOKUP_TOOL_SERVER_REF,
)


@dataclass(frozen=True)
class ToolServerSpec:
    """Resolved tool-server plugin contract used for MCP registration and tool-name permissions."""

    plugin_ref: str
    server_name: str
    factory: str | None
    tool_names: tuple[str, ...]
    visibility: tuple[str, ...]
    required_capability: str | None = None
    requires_run_dir: bool = False
    enabled_in_local_mode: bool = True
    enabled_in_server_mode: bool = True
    requires_multi_pi: bool = False
    enabled_by_default: bool = True
    handlers: dict[str, str] = field(default_factory=dict)

    def enabled_for(self, *, local_mode: bool, multi_pi_enabled: bool) -> bool:
        if not self.enabled_by_default:
            return False
        if local_mode and not self.enabled_in_local_mode:
            return False
        if not local_mode and not self.enabled_in_server_mode:
            return False
        return not (self.requires_multi_pi and not multi_pi_enabled)

    def to_ref(self) -> ToolServerRef:
        return ToolServerRef(
            ref=self.plugin_ref,
            server_name=self.server_name,
            transport="legacy_inprocess",
            tool_names=list(self.tool_names),
            metadata={
                "visibility": list(self.visibility),
                "requires_run_dir": self.requires_run_dir,
                "enabled_by_default": self.enabled_by_default,
            },
        )


@dataclass(frozen=True)
class ToolServerBuildResult:
    """Result of building in-process legacy MCP servers for a workflow stage."""

    servers: dict[str, Any]
    refs: list[ToolServerRef]
    registered: list[dict[str, Any]] = field(default_factory=list)
    unavailable: list[dict[str, str]] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)

    @property
    def connected_server_names(self) -> list[str]:
        return sorted(self.servers)


def tool_server_refs_from_task_descriptor(descriptor: dict[str, Any]) -> tuple[str, ...]:
    """Extract runtime-visible tool-server references from a task descriptor."""
    plugins = descriptor.get("praxist_plugins") or {}
    refs: list[str] = []
    for ref in plugins.get("tools") or []:
        _append_tool_server_ref(refs, ref)

    tool_servers = plugins.get("tool_servers") or {}
    if isinstance(tool_servers, dict):
        tool_specs = tool_servers.values()
    elif isinstance(tool_servers, list):
        tool_specs = tool_servers
    else:
        tool_specs = []
    for spec in tool_specs:
        if isinstance(spec, str):
            _append_tool_server_ref(refs, spec)
        elif isinstance(spec, dict) and spec.get("enabled", True):
            _append_tool_server_ref(refs, spec.get("ref"))
    return tuple(dict.fromkeys(refs))


def effective_research_tool_server_refs_from_task_descriptor(
    descriptor: dict[str, Any],
) -> tuple[str, ...]:
    """Return the research-loop tool-server refs Praxist will actually use.

    The research loop has long used ``DEFAULT_RESEARCH_TOOL_SERVER_REFS`` when a
    task descriptor does not declare tool servers. Startup plugin resolution must
    use the same effective selection so resolve-only cannot pass with a registry
    that later lacks the default tool descriptors.
    """

    refs = tool_server_refs_from_task_descriptor(descriptor)
    return refs or DEFAULT_RESEARCH_TOOL_SERVER_REFS


def _append_tool_server_ref(refs: list[str], raw_ref: Any) -> None:
    if not isinstance(raw_ref, str):
        return
    try:
        parsed = PluginRef.parse(raw_ref)
    except (KeyError, ValueError, TypeError):
        return
    if parsed.kind == "tool_server":
        refs.append(parsed.as_string())


def tool_server_for_ref(ref: str, registry: PluginRegistry | None = None) -> ToolServerSpec:
    """Resolve one tool_server plugin reference into a ToolServerSpec."""
    parsed = PluginRef.parse(ref)
    if parsed.kind != "tool_server":
        raise ValueError(f"Tool server ref must use kind tool_server: {ref}")
    canonical_ref = parsed.as_string()
    registry = registry or _load_single_plugin_registry(canonical_ref)
    selected = require_execution_plugin(registry, canonical_ref, kind="tool_server")
    if selected is None:
        raise ValueError(f"Unknown tool server plugin: {canonical_ref}")
    spec = _tool_server_spec_from_registry(canonical_ref, registry, selected)
    require_execution_plugin(
        registry,
        canonical_ref,
        kind="tool_server",
        capability=spec.required_capability,
    )
    return spec


def build_legacy_mcp_servers(
    tool_refs: Iterable[str] | None = None,
    *,
    run_dir: Path | str | None = None,
    local_mode: bool,
    multi_pi_enabled: bool = False,
    registry: PluginRegistry | None = None,
) -> ToolServerBuildResult:
    """Instantiate selected legacy in-process MCP servers for a run."""
    servers: dict[str, Any] = {}
    refs: list[ToolServerRef] = []
    registered: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for raw_ref in tool_refs or DEFAULT_RESEARCH_TOOL_SERVER_REFS:
        spec = tool_server_for_ref(str(raw_ref), registry)
        refs.append(spec.to_ref())
        if not spec.enabled_for(local_mode=local_mode, multi_pi_enabled=multi_pi_enabled):
            skipped.append(
                {
                    "ref": spec.plugin_ref,
                    "server_name": spec.server_name,
                    "reason": _skip_reason(
                        spec, local_mode=local_mode, multi_pi_enabled=multi_pi_enabled
                    ),
                }
            )
            continue
        if spec.factory is None:
            unavailable.append(
                {
                    "ref": spec.plugin_ref,
                    "server_name": spec.server_name,
                    "reason": "no_factory_configured",
                }
            )
            continue
        try:
            factory = _load_callable(spec.factory)
            server = factory(Path(run_dir)) if spec.requires_run_dir else factory()
            servers[spec.server_name] = server
            registered.append(
                {
                    "ref": spec.plugin_ref,
                    "server_name": spec.server_name,
                    "tool_names": list(spec.tool_names),
                    "transport": "legacy_inprocess",
                }
            )
        except Exception as exc:  # noqa: BLE001 - MCP tools are best-effort.
            unavailable.append(
                {
                    "ref": spec.plugin_ref,
                    "server_name": spec.server_name,
                    "reason": _short_error(exc),
                }
            )
    return ToolServerBuildResult(
        servers=servers,
        refs=refs,
        registered=registered,
        unavailable=unavailable,
        skipped=skipped,
    )


def allowed_mcp_tool_names(
    tool_refs: Iterable[str] | None = None,
    *,
    local_mode: bool,
    include_panel_tools: bool = False,
    include_peer_tools: bool = True,
    multi_pi_enabled: bool = False,
    registry: PluginRegistry | None = None,
) -> list[str]:
    """Return fully qualified MCP tool names allowed for a peer or panel role."""
    specs = [
        tool_server_for_ref(str(ref), registry)
        for ref in (tool_refs or DEFAULT_RESEARCH_TOOL_SERVER_REFS)
    ]
    return _allowed_names_for_specs(
        specs,
        local_mode=local_mode,
        include_panel_tools=include_panel_tools,
        include_peer_tools=include_peer_tools,
        multi_pi_enabled=multi_pi_enabled,
    )


def allowed_mcp_tool_names_for_servers(
    server_names: Iterable[str],
    *,
    include_panel_tools: bool = False,
    include_peer_tools: bool = True,
    tool_refs: Iterable[str] | None = None,
    registry: PluginRegistry | None = None,
) -> list[str]:
    """Return allowed MCP tool names from already connected server names."""
    selected = set(server_names)
    specs = [
        spec
        for ref in (tool_refs or DEFAULT_RESEARCH_TOOL_SERVER_REFS)
        for spec in [tool_server_for_ref(str(ref), registry)]
        if spec.server_name in selected
    ]
    return _allowed_names_for_specs(
        specs,
        local_mode=False,
        include_panel_tools=include_panel_tools,
        include_peer_tools=include_peer_tools,
        multi_pi_enabled=True,
        honor_enabled_flags=False,
    )


def visible_mcp_servers(
    servers: dict[str, Any],
    *,
    include_panel_tools: bool = False,
    include_peer_tools: bool = True,
    tool_refs: Iterable[str] | None = None,
    registry: PluginRegistry | None = None,
) -> dict[str, Any]:
    """Filter connected MCP servers by role visibility.

    Runtime adapters receive server descriptors separately from tool
    permissions. A panel-only server must therefore be removed from peer
    requests, not merely omitted from the allowed-tool list.
    """

    selected = set(servers)
    visible_names: set[str] = set()
    for ref in tool_refs or DEFAULT_RESEARCH_TOOL_SERVER_REFS:
        spec = tool_server_for_ref(str(ref), registry)
        if spec.server_name not in selected:
            continue
        if include_peer_tools and "peer" in spec.visibility:
            visible_names.add(spec.server_name)
        if include_panel_tools and "panel" in spec.visibility:
            visible_names.add(spec.server_name)
    return {name: server for name, server in servers.items() if name in visible_names}


def base_peer_allowed_tools(
    server_names: Iterable[str],
    *,
    include_panel_tools: bool = False,
    tool_refs: Iterable[str] | None = None,
    registry: PluginRegistry | None = None,
) -> list[str]:
    """Return built-in Claude Code tools plus allowed MCP tools for a peer."""
    return [
        *BASE_PEER_TOOLS,
        *allowed_mcp_tool_names_for_servers(
            server_names,
            include_panel_tools=include_panel_tools,
            include_peer_tools=True,
            tool_refs=tool_refs,
            registry=registry,
        ),
    ]


def peer_mcp_context(
    servers: dict[str, Any],
    *,
    tool_refs: Iterable[str] | None = None,
    registry: PluginRegistry | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Return peer-visible MCP servers plus matching allowed tool names."""

    peer_servers = visible_mcp_servers(servers, tool_refs=tool_refs, registry=registry)
    return (
        peer_servers,
        base_peer_allowed_tools(peer_servers.keys(), tool_refs=tool_refs, registry=registry),
    )


async def execute_legacy_tool_handler_async(
    server_ref: str,
    tool_name: str,
    args: dict[str, Any] | None = None,
    *,
    registry: PluginRegistry | None = None,
    run_dir: Path | str | None = None,
    run_id: str = "",
    budget_grant_id: str | None = None,
    budget_request_id: str | None = None,
    stage_id: str = "research_loop",
) -> ToolCallResult:
    """Execute a manifest-declared legacy tool handler with budget and redaction accounting."""
    spec = tool_server_for_ref(server_ref, registry)
    guard = BudgetedActionGuard(
        run_dir=Path(run_dir) if run_dir else None,
        run_id=run_id or (Path(run_dir).name if run_dir else "legacy_direct"),
        stage_id=stage_id,
        actor_ref=server_ref,
        action_type=f"tool.{tool_name}",
        budget_grant_id=budget_grant_id,
        request_id=budget_request_id,
        metadata={"tool_name": tool_name, "server_name": spec.server_name},
    )
    if tool_name not in spec.tool_names:
        return ToolCallResult(
            server_name=spec.server_name,
            tool_name=tool_name,
            success=False,
            output={"error": f"unknown tool for {server_ref}: {tool_name}"},
            error=f"unknown tool for {server_ref}: {tool_name}",
            failover_reason="invalid_request",
            raw_is_error=True,
        )
    handler_ref = spec.handlers.get(tool_name)
    if handler_ref is None:
        return ToolCallResult(
            server_name=spec.server_name,
            tool_name=tool_name,
            success=False,
            output={"error": f"no legacy handler adapter for {server_ref}/{tool_name}"},
            error=f"no legacy handler adapter for {server_ref}/{tool_name}",
            failover_reason="tool_unavailable",
            raw_is_error=True,
        )
    try:
        guard.start()
        handler = _load_callable(handler_ref)
        raw = handler(dict(args or {}))
        if inspect.isawaitable(raw):
            raw = await raw
        result = normalize_tool_result(spec.server_name, tool_name, raw)
        guard.finish(
            actual_usage={},
            expected_units=("wall_clock_seconds",),
            status="succeeded" if result.success else "failed",
            reason="tool_call_wall_clock_usage",
        )
        return result
    except Exception as exc:  # noqa: BLE001 - normalize tool failures.
        guard.finish(
            actual_usage={},
            expected_units=("wall_clock_seconds",),
            status="failed",
            reason="tool_call_wall_clock_usage",
        )
        message, hits = redact_text(str(exc))
        return ToolCallResult(
            server_name=spec.server_name,
            tool_name=tool_name,
            success=False,
            output={"error": message},
            error=message,
            failover_reason=_classify_error(message),
            raw_is_error=True,
            redaction_hits=hits,
        )


def execute_legacy_tool_handler(
    server_ref: str,
    tool_name: str,
    args: dict[str, Any] | None = None,
    *,
    registry: PluginRegistry | None = None,
    run_dir: Path | str | None = None,
    run_id: str = "",
    budget_grant_id: str | None = None,
    budget_request_id: str | None = None,
    stage_id: str = "research_loop",
) -> ToolCallResult:
    """Synchronous wrapper for executing a legacy tool handler."""
    return asyncio.run(
        execute_legacy_tool_handler_async(
            server_ref,
            tool_name,
            args,
            registry=registry,
            run_dir=run_dir,
            run_id=run_id,
            budget_grant_id=budget_grant_id,
            budget_request_id=budget_request_id,
            stage_id=stage_id,
        )
    )


def normalize_tool_result(server_name: str, tool_name: str, raw: Any) -> ToolCallResult:
    """Convert raw MCP-style handler output into a ToolCallResult."""
    payload = _extract_mcp_payload(raw)
    redacted_payload, hits = redact_json(payload)
    raw_is_error = bool(raw.get("is_error")) if isinstance(raw, dict) else False
    error = _error_from_payload(redacted_payload) if raw_is_error else None
    return ToolCallResult(
        server_name=server_name,
        tool_name=tool_name,
        success=not raw_is_error,
        output=redacted_payload,
        error=error,
        failover_reason="none" if not raw_is_error else _classify_error(error or ""),
        raw_is_error=raw_is_error,
        redaction_hits=hits,
    )


def _load_single_plugin_registry(ref: str) -> PluginRegistry:
    loader = PluginLoader(PluginRoots.defaults(Path.cwd()))
    manifest = loader.resolve(
        [ref],
        run_id="tool_server_spec",
        root_task_ref=ref,
        enforce_bundled_execution=True,
    )
    return loader.load(manifest)


def _tool_server_spec_from_registry(
    ref: str,
    registry: PluginRegistry,
    selected: SelectedPlugin,
) -> ToolServerSpec:
    parsed = PluginRef.parse(ref)
    raw_plugin = registry.require(parsed.kind, parsed.name)
    manifest = _read_plugin_manifest(Path(selected.path))
    manifest_contract = (
        manifest.get("tool_server") if isinstance(manifest.get("tool_server"), dict) else {}
    )
    plugin_contract = raw_plugin if isinstance(raw_plugin, dict) else {}
    contract = {**manifest_contract, **plugin_contract}
    if "tool_server" in contract and isinstance(contract["tool_server"], dict):
        contract = {**manifest_contract, **contract["tool_server"]}
    server_name = str(contract.get("server_name") or "").strip()
    if not server_name:
        raise ValueError(f"{ref} does not declare tool_server.server_name")
    tool_names = tuple(str(item) for item in contract.get("tool_names") or ())
    visibility = tuple(str(item) for item in contract.get("visibility") or ())
    handlers = {str(key): str(value) for key, value in dict(contract.get("handlers") or {}).items()}
    factory = contract.get("factory")
    return ToolServerSpec(
        plugin_ref=ref,
        server_name=server_name,
        factory=str(factory) if factory else None,
        tool_names=tool_names,
        visibility=visibility,
        required_capability=contract.get("required_capability"),
        requires_run_dir=bool(contract.get("requires_run_dir", False)),
        enabled_in_local_mode=bool(contract.get("enabled_in_local_mode", True)),
        enabled_in_server_mode=bool(contract.get("enabled_in_server_mode", True)),
        requires_multi_pi=bool(contract.get("requires_multi_pi", False)),
        enabled_by_default=bool(contract.get("enabled_by_default", True)),
        handlers=handlers,
    )


def _read_plugin_manifest(path: Path) -> dict[str, Any]:
    value = yaml.safe_load((path / "plugin.yaml").read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"plugin manifest must be an object: {path / 'plugin.yaml'}")
    return value


def _allowed_names_for_specs(
    specs: Iterable[ToolServerSpec],
    *,
    local_mode: bool,
    include_panel_tools: bool,
    include_peer_tools: bool,
    multi_pi_enabled: bool,
    honor_enabled_flags: bool = True,
) -> list[str]:
    allowed: list[str] = []
    for spec in specs:
        if honor_enabled_flags and not spec.enabled_for(
            local_mode=local_mode,
            multi_pi_enabled=multi_pi_enabled,
        ):
            continue
        visible = (include_peer_tools and "peer" in spec.visibility) or (
            include_panel_tools and "panel" in spec.visibility
        )
        if not visible:
            continue
        allowed.extend(f"mcp__{spec.server_name}__{tool_name}" for tool_name in spec.tool_names)
    return list(dict.fromkeys(allowed))


def _load_callable(ref: str) -> Callable[..., Any]:
    module_name, function_name = ref.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def _skip_reason(spec: ToolServerSpec, *, local_mode: bool, multi_pi_enabled: bool) -> str:
    if local_mode and not spec.enabled_in_local_mode:
        return "disabled_in_local_mode"
    if not local_mode and not spec.enabled_in_server_mode:
        return "disabled_in_server_mode"
    if spec.requires_multi_pi and not multi_pi_enabled:
        return "requires_multi_pi"
    if not spec.enabled_by_default:
        return "disabled_by_default"
    return "disabled"


def _short_error(exc: Exception) -> str:
    message, _ = redact_text(str(exc))
    return f"{type(exc).__name__}: {message}"


def _extract_mcp_payload(raw: Any) -> Any:
    if not isinstance(raw, dict):
        return raw
    content = raw.get("content")
    if not isinstance(content, list) or not content:
        return raw
    first = content[0]
    if not isinstance(first, dict):
        return raw
    text = first.get("text")
    if not isinstance(text, str):
        return raw
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _error_from_payload(payload: Any) -> str | None:
    if isinstance(payload, dict):
        error = payload.get("error")
        if error is not None:
            return str(error)
    if isinstance(payload, str):
        return payload
    return None


def _classify_error(message: str) -> str:
    lower = message.lower()
    if any(
        token in lower for token in ("auth", "api key", "unauthorized", "forbidden", "permission")
    ):
        return "auth_error"
    if any(token in lower for token in ("quota", "billing", "insufficient credits")):
        return "quota_exhausted"
    if any(token in lower for token in ("rate limit", "429")):
        return "rate_limited"
    if any(token in lower for token in ("timeout", "timed out", "deadline")):
        return "timeout"
    if any(
        token in lower for token in ("importerror", "not installed", "unavailable", "no_factory")
    ):
        return "tool_unavailable"
    if any(
        token in lower
        for token in ("required", "invalid", "must be", "outside allowed", "not found")
    ):
        return "invalid_request"
    return "runtime_error"
