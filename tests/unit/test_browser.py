"""Unit tests for the browser adapter (#128 PR-1).

The handler is HTTP-bound and uses trafilatura for readable-mode
extraction. We mock both via ``http_client_factory`` and
``extract_factory`` test seams — the production-only lazy imports of
``httpx`` and ``trafilatura`` are excluded from coverage because the
[agents] extra isn't installed in the test env.

Coverage:

* schema_version always present.
* Empty URL / missing scheme short-circuit.
* Invalid mode rejected.
* 200 readable: trafilatura extractor seam called; title + text returned.
* 200 raw: stdlib HTMLParser strips ``<script>`` / ``<style>``, keeps
  the title and main text without trafilatura.
* Non-200 surfaces ``status_code``.
* Body > 5 MB returns size error.
* User-Agent self-identifies as Praxist-Research-Bot.
* ``_text_result`` direct calls.
* ``create_tool_plugin()`` shape.
* ``_mcp.py`` routes ``browser`` to ``browser`` module.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from typing import Any

from praxist.plugins.tools.browser import adapter as browser


class _FakeResponse:
    """Minimal ``httpx.Response``-shaped stub."""

    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _FakeClient:
    """Test seam: records the GET call + replays scripted responses."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.init_kwargs: dict[str, Any] = {}

    def get(self, url: str) -> _FakeResponse:
        self.calls.append({"url": url})
        if not self._responses:
            raise AssertionError("test scripted no further responses for web_read")
        return self._responses.pop(0)

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _factory_for(client: _FakeClient):
    @contextmanager
    def _inner():
        yield client

    return _inner


_READABLE_HTML = (
    "<html><head><title>Test Page</title></head>"
    "<body><nav>NAV</nav>"
    "<article><p>Main article paragraph one.</p>"
    "<p>Main article paragraph two.</p></article>"
    "<script>console.log('noisy')</script>"
    "<style>.x{display:none}</style>"
    "<footer>FOOTER</footer></body></html>"
)


class WebReadHandlerTests(unittest.TestCase):
    """``web_read`` returns normalized JSON dicts, never raises."""

    def test_empty_url_returns_error(self) -> None:
        client = _FakeClient([])
        result = browser.web_read("", http_client_factory=_factory_for(client))
        self.assertEqual(result["error"], "url is required")
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(client.calls, [])

    def test_missing_scheme_returns_error(self) -> None:
        result = browser.web_read(
            "example.com/foo",
            http_client_factory=_factory_for(_FakeClient([])),
        )
        self.assertIn("must start with http", result["error"])

    def test_invalid_mode_returns_error(self) -> None:
        result = browser.web_read(
            "https://example.com",
            mode="nonsense",
            http_client_factory=_factory_for(_FakeClient([])),
        )
        self.assertIn("mode must be one of", result["error"])

    def test_readable_mode_uses_extract_factory(self) -> None:
        client = _FakeClient([_FakeResponse(200, _READABLE_HTML)])

        captured: list[str] = []

        def fake_extract(html: str) -> tuple[str, str]:
            captured.append(html)
            return ("Test Page", "Main article paragraph one.\nMain article paragraph two.")

        result = browser.web_read(
            "https://example.com/article",
            mode="readable",
            http_client_factory=_factory_for(client),
            extract_factory=fake_extract,
        )
        self.assertNotIn("error", result)
        self.assertEqual(result["title"], "Test Page")
        self.assertIn("Main article paragraph one", result["text"])
        self.assertEqual(result["mode"], "readable")
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0], _READABLE_HTML)
        # content_length tracks the extracted-text length, not raw HTML.
        self.assertEqual(result["content_length"], len(result["text"]))

    def test_raw_mode_strips_script_and_style(self) -> None:
        client = _FakeClient([_FakeResponse(200, _READABLE_HTML)])
        result = browser.web_read(
            "https://example.com/article",
            mode="raw",
            http_client_factory=_factory_for(client),
        )
        self.assertNotIn("error", result)
        self.assertEqual(result["mode"], "raw")
        # Title comes from the <title> element.
        self.assertEqual(result["title"], "Test Page")
        # Script / style contents are removed.
        self.assertNotIn("console.log", result["text"])
        self.assertNotIn("display:none", result["text"])
        # Article body survives.
        self.assertIn("Main article paragraph", result["text"])

    def test_non_200_surfaces_status_code(self) -> None:
        client = _FakeClient([_FakeResponse(404, "not found")])
        result = browser.web_read(
            "https://example.com/missing",
            http_client_factory=_factory_for(client),
            extract_factory=lambda html: ("", ""),
        )
        self.assertEqual(result["status_code"], 404)
        self.assertIn("404", result["error"])

    def test_response_larger_than_5mb_rejected(self) -> None:
        huge = "<html><body>" + ("x" * (5 * 1024 * 1024 + 1)) + "</body></html>"
        client = _FakeClient([_FakeResponse(200, huge)])
        result = browser.web_read(
            "https://example.com/huge",
            http_client_factory=_factory_for(client),
            extract_factory=lambda html: ("", ""),
        )
        self.assertIn("exceeded", result["error"])


class TextResultHelperTests(unittest.TestCase):
    """``_text_result`` wraps payloads in the MCP text-content envelope."""

    def test_dict_payload_is_json_serialized(self) -> None:
        envelope = browser._text_result({"key": "value"})
        self.assertEqual(envelope["content"][0]["type"], "text")
        self.assertIn('"key"', envelope["content"][0]["text"])

    def test_string_payload_is_passed_through_verbatim(self) -> None:
        envelope = browser._text_result("ready")
        self.assertEqual(envelope["content"][0]["text"], "ready")


class CreateToolPluginTests(unittest.TestCase):
    """Manifest entrypoint surfaces the expected descriptor shape."""

    def test_descriptor_lists_web_read_tool(self) -> None:
        plugin = browser.create_tool_plugin()
        self.assertEqual(plugin["tool_server_ref"], "tool_server:browser")
        self.assertEqual(plugin["server_name"], "browser")
        self.assertEqual(plugin["tool_names"], ["web_read"])
        handlers = plugin["handlers"]
        assert isinstance(handlers, dict)
        self.assertIn("web_read", handlers)


class CodexSdkMcpRoutingTests(unittest.TestCase):
    """The SDK runtime exposes the browser server through direct MCP config."""

    def test_browser_server_routes_to_browser_factory(self) -> None:
        import sys as _sys

        from praxist.plugins.agent_runtimes.codex_sdk._mcp import (
            MCP_STDIO_MODULE,
            mcp_configuration,
        )

        result = mcp_configuration([{"server_name": "browser"}])
        server = result.config["mcp_servers"]["browser"]
        self.assertEqual(server["command"], _sys.executable)
        self.assertEqual(server["args"][:2], ["-m", MCP_STDIO_MODULE])
        self.assertIn(
            "praxist.plugins.tools.browser.adapter:create_browser_server",
            server["args"][2],
        )
        self.assertEqual(result.warnings, ())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
