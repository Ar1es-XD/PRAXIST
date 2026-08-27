"""Browser adapter providing a provider-agnostic ``web_read`` MCP tool.

The plugin gives every agent runtime the same URL-reading surface: fetch a
URL, strip navigation and advertisements through readability extraction,
and return the main article text.

Audience
--------
Consumed by peers at agent-runtime time. Handler returns a
JSON-serialisable ``dict`` with a ``schema_version`` field and never
raises — transport / extraction failures come back as
``{"error": "..."}`` so the LLM agent can recover.

Policy
------
* **robots.txt** is *not* respected. Research-tool convention; if a
  site bans us, the operator gets a non-200 and surfaces it.
* **User-Agent** self-identifies as ``Praxist-Research-Bot/0.1``
  so site operators can contact the project if they want to.
* **JS rendering** is out of scope. A future ``browser_render``
  (Playwright) tool can land separately if research targets ever
  require it. Most arxiv abstract pages / blogs / wikis / GitHub
  READMEs serve readable HTML on first byte.
* **Timeout** is 15 s per request; **body cap** is 5 MB to keep the
  agent's context budget from being eaten by an accidental gigabyte
  of HTML.
"""

from __future__ import annotations

import json
from typing import Any

try:  # claude_agent_sdk is only required when the MCP server is spun up.
    from claude_agent_sdk import (  # type: ignore[import-not-found]
        create_sdk_mcp_server,
        tool,
    )
except ImportError:  # pragma: no cover - SDK missing in some test envs
    tool = None
    create_sdk_mcp_server = None


_DEFAULT_TIMEOUT_SECONDS = 15.0
"""Per-request HTTP timeout."""

_MAX_RESPONSE_BYTES = 5 * 1024 * 1024
"""Reject responses larger than 5 MB — protects the agent's context budget."""

_USER_AGENT = "Praxist-Research-Bot/0.1 (+https://github.com/sapientinc/praxist)"
"""Self-identifying User-Agent. Sites can contact the project if they object."""

_VALID_MODES: frozenset[str] = frozenset({"readable", "raw"})

_SCHEMA_VERSION = 1


__all__ = [
    "create_browser_server",
    "create_tool_plugin",
    "web_read",
]


# --------------------------------------------------------------------------- #
# Pure handler (callable from tests without httpx / trafilatura installed)
# --------------------------------------------------------------------------- #


def web_read(
    url: str,
    mode: str = "readable",
    *,
    http_client_factory: Any = None,
    extract_factory: Any = None,
) -> dict[str, Any]:
    """Fetch a URL and return its main text content.

    Args:
        url: Target page. Must include a scheme (``http`` or ``https``).
        mode: ``"readable"`` (default) uses trafilatura to strip nav /
            ads / boilerplate and return main article text;
            ``"raw"`` returns the full HTML converted to plain text
            with minimal cleanup.
        http_client_factory: Test seam — zero-arg callable returning a
            context manager wrapping an ``httpx.Client``-shaped object.
            Production calls leave this ``None``.
        extract_factory: Test seam — callable that takes ``html: str``
            and returns ``(title: str, text: str)`` for readable mode.
            Production calls leave this ``None`` and use trafilatura.

    Returns:
        On success::

            {
              "schema_version": 1,
              "url": str, "title": str, "text": str,
              "mode": "readable" | "raw",
              "content_length": int, "fetched_at": str,
            }

        On error::

            {"schema_version": 1, "error": "...", "status_code"?: int}
    """
    if not isinstance(url, str) or not url.strip():
        return {"schema_version": _SCHEMA_VERSION, "error": "url is required"}
    if not (url.startswith("http://") or url.startswith("https://")):
        return {
            "schema_version": _SCHEMA_VERSION,
            "error": "url must start with http:// or https://",
        }

    if mode not in _VALID_MODES:
        return {
            "schema_version": _SCHEMA_VERSION,
            "error": f"mode must be one of {sorted(_VALID_MODES)}",
        }

    headers = {"User-Agent": _USER_AGENT}

    if http_client_factory is None:  # pragma: no cover - production-only
        # httpx is an ``[agents]`` extra (declared in pyproject.toml) and is
        # NOT installed in the unit-test env. The lazy import keeps the
        # plugin importable when only the core deps are present; tests
        # always inject ``http_client_factory``.
        import httpx  # type: ignore[import-not-found]

        def _factory() -> Any:
            return httpx.Client(timeout=_DEFAULT_TIMEOUT_SECONDS, headers=headers)

        factory: Any = _factory
        request_error: tuple[type[Exception], ...] = (httpx.RequestError,)
    else:
        factory = http_client_factory
        request_error = (Exception,)

    try:
        with factory() as client:
            response = client.get(url)
    except request_error as exc:  # pragma: no cover - network error path
        return {
            "schema_version": _SCHEMA_VERSION,
            "error": f"web_read request failed: {exc}",
        }

    status = getattr(response, "status_code", None)
    if status != 200:
        return {
            "schema_version": _SCHEMA_VERSION,
            "error": f"web_read returned status {status}",
            "status_code": status,
        }

    content = getattr(response, "text", "")
    if not isinstance(content, str):
        content = str(content)
    if len(content.encode("utf-8")) > _MAX_RESPONSE_BYTES:
        return {
            "schema_version": _SCHEMA_VERSION,
            "error": (
                f"web_read response exceeded {_MAX_RESPONSE_BYTES} bytes; "
                "fetch a specific path or a smaller mirror"
            ),
        }

    if mode == "readable":
        title, text = _extract_readable(content, extract_factory=extract_factory)
    else:
        title, text = _extract_raw(content)

    fetched_at = _now_iso_utc()
    return {
        "schema_version": _SCHEMA_VERSION,
        "url": url,
        "title": title,
        "text": text,
        "mode": mode,
        "content_length": len(text),
        "fetched_at": fetched_at,
    }


def _extract_readable(html: str, *, extract_factory: Any = None) -> tuple[str, str]:
    """Run trafilatura over ``html`` and return ``(title, text)``."""
    if extract_factory is not None:
        return extract_factory(html)
    return _extract_via_trafilatura(html)  # pragma: no cover - production-only


def _extract_via_trafilatura(html: str) -> tuple[str, str]:  # pragma: no cover - production-only
    """Default readable-mode extractor; only runs when trafilatura is installed."""
    import trafilatura  # type: ignore[import-not-found]

    text = trafilatura.extract(html, output_format="txt") or ""
    metadata = trafilatura.extract_metadata(html)
    title = ""
    if metadata is not None:
        raw_title = getattr(metadata, "title", "") or ""
        title = str(raw_title)
    return title, text


def _extract_raw(html: str) -> tuple[str, str]:
    """Cheap raw-mode extraction: strip tags via stdlib HTMLParser, no readability."""
    from html.parser import HTMLParser

    class _Stripper(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.title = ""
            self._in_title = False
            self._skip_depth = 0
            self._parts: list[str] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag in ("script", "style"):
                self._skip_depth += 1
            elif tag == "title":
                self._in_title = True

        def handle_endtag(self, tag: str) -> None:
            if tag in ("script", "style") and self._skip_depth > 0:
                self._skip_depth -= 1
            elif tag == "title":
                self._in_title = False

        def handle_data(self, data: str) -> None:
            if self._skip_depth > 0:
                return
            if self._in_title:
                self.title += data
            else:
                self._parts.append(data)

    stripper = _Stripper()
    stripper.feed(html)
    text = " ".join(part.strip() for part in stripper._parts if part.strip())
    return stripper.title.strip(), text


def _now_iso_utc() -> str:
    """Return current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    import datetime

    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# MCP server wiring
# --------------------------------------------------------------------------- #


def _text_result(data: Any) -> dict[str, Any]:
    """Wrap a JSON-serialisable value as an MCP text content response."""
    text = json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data
    return {"content": [{"type": "text", "text": text}]}


def create_browser_server() -> Any:  # pragma: no cover - requires claude_agent_sdk
    """Create the MCP server exposing the single ``web_read`` tool."""
    if create_sdk_mcp_server is None or tool is None:
        raise ImportError("claude_agent_sdk is required for MCP tools")

    async def _handle_web_read(args: dict[str, Any]) -> dict[str, Any]:
        return _text_result(
            web_read(
                str(args.get("url", "")),
                str(args.get("mode", "readable")),
            )
        )

    web_read_tool = tool(
        "web_read",
        (
            "Fetch a URL and return its main text. mode=readable (default) "
            "strips nav/ads via readability extraction; mode=raw returns "
            "the full HTML stripped to plain text. Use this for research "
            "blogs, arxiv abstract pages, wikis, GitHub READMEs."
        ),
        {"url": str, "mode": str},
    )(_handle_web_read)

    return create_sdk_mcp_server("browser", tools=[web_read_tool])


def create_tool_plugin() -> dict[str, object]:
    """Manifest entrypoint exposing the browser tool server descriptor."""
    return {
        "tool_server_ref": "tool_server:browser",
        "server_name": "browser",
        "factory": "praxist.plugins.tools.browser.adapter:create_browser_server",
        "tool_names": ["web_read"],
        "visibility": ["peer", "panel"],
        "required_capability": "tool_server.browser",
        "handlers": {
            "web_read": "praxist.plugins.tools.browser.adapter:web_read",
        },
    }
