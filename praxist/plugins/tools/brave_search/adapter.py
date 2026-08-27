"""Brave Search adapter providing a provider-agnostic ``web_search`` MCP tool.

The plugin owns one normalized search surface backed by the Brave Search REST
API. Every agent runtime receives the same tool schema and result contract.

Audience
--------
The tool is consumed by peers at agent-runtime time. Each handler
returns a JSON-serializable ``dict`` with a ``schema_version`` field
and never raises — errors come back as ``{"error": "..."}`` so the
LLM agent can recover.

Configuration
-------------
``BRAVE_API_KEY`` env var (an API credential; legitimate boundary
read). When unset, the handler returns ``{"error": "BRAVE_API_KEY is
not set"}`` instead of attempting the request — peers that depend on
search will surface the misconfiguration in their reply rather than
crash.

Rate limiting
-------------
Brave returns ``429`` when the configured tier's per-second / per-month
budget is exceeded. We retry once after a short sleep; the second 429
returns ``{"error": "brave search rate-limited after retry", ...}``
so the model knows to either re-issue later or fall back to a
non-search reply.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

try:  # claude_agent_sdk is only required when the MCP server is spun up.
    from claude_agent_sdk import (  # type: ignore[import-not-found]
        create_sdk_mcp_server,
        tool,
    )
except ImportError:  # pragma: no cover - SDK missing in some test envs
    tool = None
    create_sdk_mcp_server = None


_BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"
"""Brave Search Web Search REST endpoint."""

_DEFAULT_TIMEOUT_SECONDS = 10.0
"""Per-request HTTP timeout."""

_DEFAULT_COUNT = 10
_MAX_COUNT = 20
"""Brave's API caps ``count`` at 20 per request; clamp to keep latency predictable."""

_RATE_LIMIT_RETRY_DELAY_SECONDS = 1.5
"""Backoff before retrying a 429. One retry only — second 429 surfaces to caller."""

_FRESHNESS_VALUES: frozenset[str] = frozenset({"pd", "pw", "pm", "py"})
"""Accepted Brave freshness codes: past day / week / month / year."""

_SCHEMA_VERSION = 1


__all__ = [
    "create_brave_search_server",
    "create_tool_plugin",
    "web_search",
]


# --------------------------------------------------------------------------- #
# Pure handler (callable from tests without claude_agent_sdk)
# --------------------------------------------------------------------------- #


def web_search(
    query: str,
    count: int = _DEFAULT_COUNT,
    freshness: str | None = None,
    *,
    api_key: str | None = None,
    http_client_factory: Any = None,
) -> dict[str, Any]:
    """Run a Brave Search web query and return normalized results.

    Args:
        query: User search query. Stripped; empty/missing returns an
            error dict.
        count: Number of results to request. Clamped to ``[1, 20]``.
        freshness: Optional time filter — one of ``pd`` (past day),
            ``pw`` (week), ``pm`` (month), ``py`` (year). Other values
            are silently dropped.
        api_key: Override the ``BRAVE_API_KEY`` env read. Only used
            by tests; production calls leave this ``None`` and rely
            on the env boundary.
        http_client_factory: Test seam — a zero-arg callable returning
            a context manager wrapping an httpx.Client-shaped object.
            Production calls leave this ``None`` and build a fresh
            ``httpx.Client`` per call.

    Returns:
        On success: ``{"schema_version": 1, "query": "...", "results": [...]}``
        where each result has ``rank`` / ``title`` / ``url`` /
        ``snippet`` / ``age``.

        On error: ``{"schema_version": 1, "error": "..."}`` plus
        optional ``status_code`` for HTTP failures.
    """
    logger = logging.getLogger(__name__)

    if not isinstance(query, str) or not query.strip():
        return {"schema_version": _SCHEMA_VERSION, "error": "query is required"}

    resolved_key = api_key if api_key is not None else os.environ.get("BRAVE_API_KEY", "")
    if not resolved_key:
        return {"schema_version": _SCHEMA_VERSION, "error": "BRAVE_API_KEY is not set"}

    clamped_count = max(1, min(int(count), _MAX_COUNT))
    params: dict[str, Any] = {"q": query.strip(), "count": clamped_count}
    if isinstance(freshness, str) and freshness in _FRESHNESS_VALUES:
        params["freshness"] = freshness

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": resolved_key,
    }

    if http_client_factory is None:  # pragma: no cover - production-only
        # httpx is an ``[agents]`` extra (declared in pyproject.toml) and is
        # NOT installed in the unit-test env. The lazy import keeps the
        # plugin importable when only the core deps are present; the actual
        # body of this branch only executes in production where httpx is
        # available. Tests always inject ``http_client_factory`` to exercise
        # the rest of ``web_search`` without depending on the optional dep.
        import httpx  # type: ignore[import-not-found]

        def _factory() -> Any:
            return httpx.Client(timeout=_DEFAULT_TIMEOUT_SECONDS)

        factory: Any = _factory
        request_error: tuple[type[Exception], ...] = (httpx.RequestError,)
    else:
        factory = http_client_factory
        # When the test injects a factory, surface any exception it raises
        # as a generic transport failure.
        request_error = (Exception,)

    for attempt in range(2):
        try:
            with factory() as client:
                response = client.get(_BRAVE_API_URL, params=params, headers=headers)
        except request_error as exc:  # pragma: no cover - network error path
            return {
                "schema_version": _SCHEMA_VERSION,
                "error": f"brave search request failed: {exc}",
            }
        status = getattr(response, "status_code", None)
        if status == 429:
            if attempt == 0:
                time.sleep(_RATE_LIMIT_RETRY_DELAY_SECONDS)
                continue
            return {
                "schema_version": _SCHEMA_VERSION,
                "error": "brave search rate-limited after retry",
                "status_code": 429,
            }
        if status != 200:
            return {
                "schema_version": _SCHEMA_VERSION,
                "error": f"brave search returned status {status}",
                "status_code": status,
            }
        try:
            payload = response.json()
        except (ValueError, AttributeError) as exc:
            logger.warning("brave search response not JSON: %s", exc)
            return {
                "schema_version": _SCHEMA_VERSION,
                "error": f"brave search response not JSON: {exc}",
            }
        return _normalize_brave_results(query.strip(), payload)

    return {  # pragma: no cover - the for-loop always returns inside.
        "schema_version": _SCHEMA_VERSION,
        "error": "brave search unexpected fall-through",
    }


def _normalize_brave_results(query: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Reduce Brave's response to the framework's normalized schema."""
    web = payload.get("web") if isinstance(payload, dict) else None
    results = web.get("results") if isinstance(web, dict) else None
    if not isinstance(results, list):
        results = []
    normalized: list[dict[str, Any]] = []
    for rank, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "rank": rank,
                "title": str(item.get("title") or ""),
                "url": str(item.get("url") or ""),
                "snippet": str(item.get("description") or ""),
                "age": str(item.get("age") or ""),
            }
        )
    return {
        "schema_version": _SCHEMA_VERSION,
        "query": query,
        "results": normalized,
        "result_count": len(normalized),
    }


# --------------------------------------------------------------------------- #
# MCP server wiring
# --------------------------------------------------------------------------- #


def _text_result(data: Any) -> dict[str, Any]:
    """Wrap a JSON-serializable value as an MCP text content response."""
    text = json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data
    return {"content": [{"type": "text", "text": text}]}


def create_brave_search_server() -> Any:  # pragma: no cover - requires claude_agent_sdk
    """Create the MCP server exposing the single ``web_search`` tool."""
    if create_sdk_mcp_server is None or tool is None:
        raise ImportError("claude_agent_sdk is required for MCP tools")

    async def _handle_web_search(args: dict[str, Any]) -> dict[str, Any]:
        return _text_result(
            web_search(
                str(args.get("query", "")),
                int(args.get("count", _DEFAULT_COUNT)),
                args.get("freshness"),
            )
        )

    web_search_tool = tool(
        "web_search",
        (
            "Search the web via Brave Search. Returns ranked results "
            "with title, url, snippet, and freshness. Use freshness=pd "
            "for past-day, pw past-week, pm past-month, py past-year."
        ),
        {"query": str, "count": int, "freshness": str},
    )(_handle_web_search)

    return create_sdk_mcp_server("brave-search", tools=[web_search_tool])


def create_tool_plugin() -> dict[str, object]:
    """Manifest entrypoint exposing the brave-search tool server descriptor."""
    return {
        "tool_server_ref": "tool_server:brave_search",
        "server_name": "brave-search",
        "factory": ("praxist.plugins.tools.brave_search.adapter:create_brave_search_server"),
        "tool_names": ["web_search"],
        "visibility": ["peer", "panel"],
        "required_capability": "tool_server.brave_search",
        "handlers": {
            "web_search": "praxist.plugins.tools.brave_search.adapter:web_search",
        },
    }
