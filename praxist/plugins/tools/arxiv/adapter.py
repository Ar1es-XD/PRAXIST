"""arxiv adapter — search + metadata + recent listing (#128 PR-2).

The framework's peers need a stable surface for "find me papers about
X" / "what does this arxiv ID actually say" / "what's new this week
in cs.AI". Without it, the model hallucinates arxiv IDs from its
training cutoff and peers can't follow up.

Audience
--------
Consumed by peers at agent-runtime time. Three handlers, all return
``schema_version: 1``-versioned JSON dicts and never raise:

* ``arxiv_search(query, max_results, sort_by)`` — keyword search.
* ``arxiv_get(arxiv_id)`` — full metadata for one paper.
* ``arxiv_recent(category, days, max_results)`` — recent papers in
  a category, filtered by submission age.

API
---
arxiv's public API at ``https://export.arxiv.org/api/query`` returns
Atom XML. We parse with stdlib ``xml.etree.ElementTree`` (no new
dependency); fetch with httpx (already in ``[agents]``).

Rate limiting
-------------
arxiv's TOS asks for ~3 s between requests. We track the last call
time in a process-local module variable and sleep before the next
request if needed — gentle enough not to need an explicit token
bucket.
"""

from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from typing import Any

try:  # claude_agent_sdk is only required when the MCP server is spun up.
    from claude_agent_sdk import (  # type: ignore[import-not-found]
        create_sdk_mcp_server,
        tool,
    )
except ImportError:  # pragma: no cover - SDK missing in some test envs
    tool = None
    create_sdk_mcp_server = None


_ARXIV_API_URL = "https://export.arxiv.org/api/query"
"""arxiv public REST/Atom endpoint."""

_DEFAULT_TIMEOUT_SECONDS = 15.0

_DEFAULT_MAX_RESULTS = 10
_MAX_RESULTS_CAP = 50
_DEFAULT_RECENT_DAYS = 7
_MAX_RECENT_DAYS = 90

_RATE_LIMIT_INTERVAL_SECONDS = 3.0
"""Honor arxiv's polite-use guidance: ~3 s between requests."""

# Atom XML namespaces arxiv uses.
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

_SCHEMA_VERSION = 1

_SORT_BY_MAP: dict[str, str] = {
    "relevance": "relevance",
    "submitted_date": "submittedDate",
    "submitteddate": "submittedDate",
    "last_updated_date": "lastUpdatedDate",
    "lastupdateddate": "lastUpdatedDate",
}
"""Map operator-facing sort keys to arxiv's API enum values."""

# Process-local rate-limit cursor. We track the last query's start
# time and sleep just enough before the next one.
_LAST_REQUEST_TIME: float = 0.0


__all__ = [
    "arxiv_get",
    "arxiv_recent",
    "arxiv_search",
    "create_arxiv_server",
    "create_tool_plugin",
]


# --------------------------------------------------------------------------- #
# Pure handlers (callable from tests without httpx installed)
# --------------------------------------------------------------------------- #


def arxiv_search(
    query: str,
    max_results: int = _DEFAULT_MAX_RESULTS,
    sort_by: str = "relevance",
    *,
    http_client_factory: Any = None,
    clock: Any = None,
) -> dict[str, Any]:
    """Run a Brave-style keyword search against arxiv.

    Args:
        query: User search query (arxiv's ``search_query`` parameter).
            ``all:foo`` style operators pass through unchanged.
        max_results: Result cap. Clamped to ``[1, 50]``.
        sort_by: One of ``relevance`` (default), ``submitted_date``,
            ``last_updated_date``. Unknown values fall back to
            ``relevance``.
        http_client_factory: Test seam.
        clock: Test seam — pair of ``(time_fn, sleep_fn)`` for
            rate-limit testing. Production calls leave this ``None``.

    Returns:
        ``{schema_version, query, total_results, results: [...]}``
        on success; ``{schema_version, error, status_code?}`` on
        failure.
    """
    if not isinstance(query, str) or not query.strip():
        return {"schema_version": _SCHEMA_VERSION, "error": "query is required"}

    capped_max = max(1, min(int(max_results), _MAX_RESULTS_CAP))
    sort_value = _SORT_BY_MAP.get(str(sort_by).lower(), "relevance")

    params: dict[str, str | int] = {
        "search_query": query.strip(),
        "start": 0,
        "max_results": capped_max,
        "sortBy": sort_value,
        "sortOrder": "descending",
    }
    payload = _fetch_arxiv_xml(params, http_client_factory=http_client_factory, clock=clock)
    if "error" in payload:
        return payload
    return _build_search_response(query.strip(), payload["xml"])


def arxiv_get(
    arxiv_id: str,
    *,
    http_client_factory: Any = None,
    clock: Any = None,
) -> dict[str, Any]:
    """Fetch full metadata for one arxiv paper by id.

    Args:
        arxiv_id: arxiv identifier with or without version suffix
            (``2305.12345`` or ``2305.12345v3``). Required.

    Returns:
        ``{schema_version, paper: {...}}`` on success;
        ``{schema_version, error}`` on failure (including "no paper
        with that id").
    """
    if not isinstance(arxiv_id, str) or not arxiv_id.strip():
        return {"schema_version": _SCHEMA_VERSION, "error": "arxiv_id is required"}

    params: dict[str, str | int] = {"id_list": arxiv_id.strip()}
    payload = _fetch_arxiv_xml(params, http_client_factory=http_client_factory, clock=clock)
    if "error" in payload:
        return payload

    entries = _parse_entries(payload["xml"])
    if not entries:
        return {
            "schema_version": _SCHEMA_VERSION,
            "error": f"no arxiv paper with id {arxiv_id!r}",
        }
    return {"schema_version": _SCHEMA_VERSION, "paper": entries[0]}


def arxiv_recent(
    category: str,
    days: int = _DEFAULT_RECENT_DAYS,
    max_results: int = _DEFAULT_MAX_RESULTS,
    *,
    http_client_factory: Any = None,
    clock: Any = None,
) -> dict[str, Any]:
    """List recent submissions in an arxiv category, newest first.

    Args:
        category: arxiv category code (``cs.AI``, ``stat.ML``,
            ``cond-mat.mes-hall``, …). Required.
        days: Look-back window. Clamped to ``[1, 90]``.
        max_results: Result cap. Clamped to ``[1, 50]``.
    """
    if not isinstance(category, str) or not category.strip():
        return {"schema_version": _SCHEMA_VERSION, "error": "category is required"}

    capped_days = max(1, min(int(days), _MAX_RECENT_DAYS))
    capped_max = max(1, min(int(max_results), _MAX_RESULTS_CAP))

    params: dict[str, str | int] = {
        "search_query": f"cat:{category.strip()}",
        "start": 0,
        "max_results": capped_max,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    payload = _fetch_arxiv_xml(params, http_client_factory=http_client_factory, clock=clock)
    if "error" in payload:
        return payload

    entries = _parse_entries(payload["xml"])
    # Filter by submission age locally — arxiv's API doesn't natively
    # accept "since" so we over-fetch and drop older ones.
    cutoff = _utc_now(clock=clock) - capped_days * 86400.0
    filtered = [
        entry for entry in entries if _parse_iso8601(entry.get("submitted_date", "")) >= cutoff
    ]
    return {
        "schema_version": _SCHEMA_VERSION,
        "category": category.strip(),
        "days": capped_days,
        "results": filtered,
        "result_count": len(filtered),
    }


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _fetch_arxiv_xml(
    params: dict[str, str | int],
    *,
    http_client_factory: Any = None,
    clock: Any = None,
) -> dict[str, Any]:
    """Run the arxiv API GET with rate-limit guard. Returns ``{"xml": str}`` or error."""
    global _LAST_REQUEST_TIME

    time_fn, sleep_fn = _resolve_clock(clock)

    now = time_fn()
    # ``_LAST_REQUEST_TIME == 0.0`` means "no prior request this process"
    # → no sleep needed. Without this guard the first call would always
    # block 3 s waiting for an imagined previous request at time 0.
    if _LAST_REQUEST_TIME > 0.0:
        delta = now - _LAST_REQUEST_TIME
        if delta < _RATE_LIMIT_INTERVAL_SECONDS:
            sleep_fn(_RATE_LIMIT_INTERVAL_SECONDS - delta)
    _LAST_REQUEST_TIME = time_fn()

    if http_client_factory is None:  # pragma: no cover - production-only
        # httpx is an ``[agents]`` extra (declared in pyproject.toml) and is
        # NOT installed in the unit-test env. Tests always inject
        # ``http_client_factory`` to bypass this branch.
        import httpx  # type: ignore[import-not-found]

        def _factory() -> Any:
            return httpx.Client(timeout=_DEFAULT_TIMEOUT_SECONDS, follow_redirects=True)

        factory: Any = _factory
        request_error: tuple[type[Exception], ...] = (httpx.RequestError,)
    else:
        factory = http_client_factory
        request_error = (Exception,)

    try:
        with factory() as client:
            response = client.get(_ARXIV_API_URL, params=params)
    except request_error as exc:  # pragma: no cover - network error path
        return {
            "schema_version": _SCHEMA_VERSION,
            "error": f"arxiv request failed: {exc}",
        }

    status = getattr(response, "status_code", None)
    if status != 200:
        return {
            "schema_version": _SCHEMA_VERSION,
            "error": f"arxiv returned status {status}",
            "status_code": status,
        }
    text = getattr(response, "text", "")
    if not isinstance(text, str):
        text = str(text)
    return {"xml": text}


def _build_search_response(query: str, xml_text: str) -> dict[str, Any]:
    """Parse arxiv Atom XML into the framework's normalized search shape."""
    total = _parse_total_results(xml_text)
    entries = _parse_entries(xml_text)
    return {
        "schema_version": _SCHEMA_VERSION,
        "query": query,
        "total_results": total,
        "results": entries,
        "result_count": len(entries),
    }


def _parse_total_results(xml_text: str) -> int:
    """Extract ``<opensearch:totalResults>`` from arxiv's feed."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return 0
    node = root.find("opensearch:totalResults", _NS)
    if node is None or not node.text:
        return 0
    try:
        return int(node.text.strip())
    except ValueError:
        return 0


def _parse_entries(xml_text: str) -> list[dict[str, Any]]:
    """Parse ``<entry>`` elements into normalized paper dicts."""
    logger = logging.getLogger(__name__)
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("arxiv: failed to parse Atom XML: %s", exc)
        return []

    entries: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", _NS):
        entries.append(_parse_entry(entry))
    return entries


def _parse_entry(entry: ET.Element) -> dict[str, Any]:
    """Reduce one ``<entry>`` element to the framework's normalized shape."""
    raw_id = _entry_text(entry, "atom:id")
    arxiv_id = raw_id.rsplit("/", 1)[-1] if raw_id else ""

    authors: list[str] = []
    for author in entry.findall("atom:author", _NS):
        name = author.findtext("atom:name", default="", namespaces=_NS)
        if name:
            authors.append(name.strip())

    categories: list[str] = []
    for cat in entry.findall("atom:category", _NS):
        term = cat.get("term")
        if term:
            categories.append(term)

    pdf_url = ""
    html_url = ""
    for link in entry.findall("atom:link", _NS):
        link_type = link.get("type", "")
        href = link.get("href", "")
        if link_type == "application/pdf":
            pdf_url = href
        elif link.get("rel") == "alternate":
            html_url = href

    return {
        "arxiv_id": arxiv_id,
        "title": _entry_text(entry, "atom:title").strip(),
        "abstract": _entry_text(entry, "atom:summary").strip(),
        "authors": authors,
        "categories": categories,
        "submitted_date": _entry_text(entry, "atom:published").strip(),
        "updated_date": _entry_text(entry, "atom:updated").strip(),
        "pdf_url": pdf_url,
        "html_url": html_url,
    }


def _entry_text(entry: ET.Element, path: str) -> str:
    node = entry.find(path, _NS)
    return (node.text or "") if node is not None else ""


def _parse_iso8601(value: str) -> float:
    """Best-effort ISO-8601 → epoch seconds. Returns ``0.0`` on malformed input."""
    if not value:
        return 0.0
    try:
        import datetime

        # arxiv emits ``2024-05-18T12:34:56Z``; ``fromisoformat`` accepts that on 3.11+.
        return datetime.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _utc_now(*, clock: Any = None) -> float:
    """Return current epoch seconds; respects the test clock seam."""
    time_fn, _sleep = _resolve_clock(clock)
    return time_fn()


def _resolve_clock(clock: Any) -> tuple[Any, Any]:
    """Resolve the (time_fn, sleep_fn) pair from the test seam or fall back."""
    if clock is None:
        return time.time, time.sleep
    if isinstance(clock, tuple) and len(clock) == 2:
        return clock
    raise TypeError("clock must be None or a (time_fn, sleep_fn) tuple")


def _reset_rate_limit_for_tests() -> None:
    """Test helper: reset the process-local rate-limit cursor."""
    global _LAST_REQUEST_TIME
    _LAST_REQUEST_TIME = 0.0


# --------------------------------------------------------------------------- #
# MCP server wiring
# --------------------------------------------------------------------------- #


def _text_result(data: Any) -> dict[str, Any]:
    """Wrap a JSON-serialisable value as an MCP text content response."""
    text = json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data
    return {"content": [{"type": "text", "text": text}]}


def create_arxiv_server() -> Any:  # pragma: no cover - requires claude_agent_sdk
    """Create the MCP server exposing the three arxiv tools."""
    if create_sdk_mcp_server is None or tool is None:
        raise ImportError("claude_agent_sdk is required for MCP tools")

    async def _handle_search(args: dict[str, Any]) -> dict[str, Any]:
        return _text_result(
            arxiv_search(
                str(args.get("query", "")),
                int(args.get("max_results", _DEFAULT_MAX_RESULTS)),
                str(args.get("sort_by", "relevance")),
            )
        )

    async def _handle_get(args: dict[str, Any]) -> dict[str, Any]:
        return _text_result(arxiv_get(str(args.get("arxiv_id", ""))))

    async def _handle_recent(args: dict[str, Any]) -> dict[str, Any]:
        return _text_result(
            arxiv_recent(
                str(args.get("category", "")),
                int(args.get("days", _DEFAULT_RECENT_DAYS)),
                int(args.get("max_results", _DEFAULT_MAX_RESULTS)),
            )
        )

    search_tool = tool(
        "arxiv_search",
        (
            "Search arxiv. Returns ranked papers with arxiv_id / title / "
            "authors / abstract / categories / submission dates / PDF + "
            "HTML URLs. sort_by: relevance (default) / submitted_date / "
            "last_updated_date."
        ),
        {"query": str, "max_results": int, "sort_by": str},
    )(_handle_search)

    get_tool = tool(
        "arxiv_get",
        ("Fetch full metadata for one arxiv paper by id (e.g. '2305.12345' or '2305.12345v3')."),
        {"arxiv_id": str},
    )(_handle_get)

    recent_tool = tool(
        "arxiv_recent",
        (
            "List recent submissions in an arxiv category, newest first. "
            "category like 'cs.AI'; days clamped to [1,90]; max_results [1,50]."
        ),
        {"category": str, "days": int, "max_results": int},
    )(_handle_recent)

    return create_sdk_mcp_server("arxiv", tools=[search_tool, get_tool, recent_tool])


def create_tool_plugin() -> dict[str, object]:
    """Manifest entrypoint exposing the arxiv tool server descriptor."""
    return {
        "tool_server_ref": "tool_server:arxiv",
        "server_name": "arxiv",
        "factory": "praxist.plugins.tools.arxiv.adapter:create_arxiv_server",
        "tool_names": ["arxiv_search", "arxiv_get", "arxiv_recent"],
        "visibility": ["peer", "panel"],
        "required_capability": "tool_server.arxiv",
        "handlers": {
            "arxiv_search": "praxist.plugins.tools.arxiv.adapter:arxiv_search",
            "arxiv_get": "praxist.plugins.tools.arxiv.adapter:arxiv_get",
            "arxiv_recent": "praxist.plugins.tools.arxiv.adapter:arxiv_recent",
        },
    }
