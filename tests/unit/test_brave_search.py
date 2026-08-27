"""Unit tests for the Brave Search adapter (#109).

The handler is HTTP-bound so we never hit the real Brave endpoint here
— ``web_search`` takes an ``http_client_factory`` test seam that wraps
a fake context manager around a stub response object. The fake
mirrors the parts of ``httpx.Client`` / ``httpx.Response`` the handler
actually uses (``.get(url, params, headers)`` returning an object
with ``.status_code`` and ``.json()``).

Coverage:

* schema_version always present on success and on error.
* missing API key / empty query short-circuit into error dicts.
* count clamping (negative / overflow → 1..20).
* freshness validation (invalid values dropped).
* 200 OK normalizes into the framework's flat result schema.
* 429 retries once then surfaces ``error`` + ``status_code`` on
  second failure.
* 200 after one 429 retry succeeds.
* Non-200 non-429 responses surface ``error`` + ``status_code``.
* ``create_tool_plugin()`` returns the expected manifest shape.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from praxist.plugins.tools.brave_search import adapter as brave


class _FakeResponse:
    """Minimal ``httpx.Response``-shaped stub for the brave adapter."""

    def __init__(
        self,
        status_code: int,
        payload: Any | None = None,
        json_exc: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self._json_exc = json_exc

    def json(self) -> Any:
        if self._json_exc is not None:
            raise self._json_exc
        return self._payload


class _FakeClient:
    """Test seam: records the GET call and replays scripted responses."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any], headers: dict[str, str]) -> _FakeResponse:
        self.calls.append({"url": url, "params": dict(params), "headers": dict(headers)})
        if not self._responses:
            raise AssertionError("test scripted no further responses for brave search")
        return self._responses.pop(0)

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _factory_for(client: _FakeClient):
    """Wrap a client instance so ``with factory() as c`` yields it once."""

    @contextmanager
    def _inner():
        yield client

    return _inner


_BRAVE_PAYLOAD = {
    "web": {
        "results": [
            {
                "title": "Result one",
                "url": "https://example.com/one",
                "description": "First snippet",
                "age": "1 day ago",
            },
            {
                "title": "Result two",
                "url": "https://example.com/two",
                "description": "Second snippet",
            },
            # Non-dict items must be skipped, not crash the normalizer.
            "ignored-string-entry",
        ]
    }
}


class WebSearchHandlerTests(unittest.TestCase):
    """``web_search`` returns normalized JSON dicts, never raises."""

    def test_missing_api_key_returns_error(self) -> None:
        client = _FakeClient([])
        result = brave.web_search(
            "anything",
            api_key="",
            http_client_factory=_factory_for(client),
        )
        self.assertEqual(result["error"], "BRAVE_API_KEY is not set")
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(client.calls, [])

    def test_empty_query_returns_error(self) -> None:
        client = _FakeClient([])
        result = brave.web_search(
            "   ",
            api_key="fake",
            http_client_factory=_factory_for(client),
        )
        self.assertEqual(result["error"], "query is required")
        self.assertEqual(client.calls, [])

    def test_success_normalizes_results_and_preserves_rank(self) -> None:
        client = _FakeClient([_FakeResponse(200, _BRAVE_PAYLOAD)])
        result = brave.web_search(
            "what is Praxist",
            count=5,
            freshness="pw",
            api_key="fake",
            http_client_factory=_factory_for(client),
        )
        self.assertNotIn("error", result)
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["query"], "what is Praxist")
        self.assertEqual(result["result_count"], 2)
        # Non-dict items in the payload were filtered out.
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["rank"], 1)
        self.assertEqual(result["results"][0]["title"], "Result one")
        self.assertEqual(result["results"][0]["snippet"], "First snippet")
        # Missing "age" defaults to empty string, not KeyError.
        self.assertEqual(result["results"][1]["age"], "")
        # Request included the api key + freshness + clamped count.
        call = client.calls[0]
        self.assertEqual(call["headers"]["X-Subscription-Token"], "fake")
        self.assertEqual(call["params"]["count"], 5)
        self.assertEqual(call["params"]["freshness"], "pw")
        self.assertEqual(call["params"]["q"], "what is Praxist")

    def test_count_is_clamped(self) -> None:
        client = _FakeClient([_FakeResponse(200, {"web": {"results": []}})])
        brave.web_search(
            "x",
            count=999,
            api_key="fake",
            http_client_factory=_factory_for(client),
        )
        self.assertEqual(client.calls[0]["params"]["count"], 20)

        client = _FakeClient([_FakeResponse(200, {"web": {"results": []}})])
        brave.web_search(
            "x",
            count=0,
            api_key="fake",
            http_client_factory=_factory_for(client),
        )
        self.assertEqual(client.calls[0]["params"]["count"], 1)

    def test_invalid_freshness_is_dropped(self) -> None:
        client = _FakeClient([_FakeResponse(200, {"web": {"results": []}})])
        brave.web_search(
            "x",
            freshness="bogus",
            api_key="fake",
            http_client_factory=_factory_for(client),
        )
        self.assertNotIn("freshness", client.calls[0]["params"])

    def test_429_retries_then_surfaces_rate_limit_error(self) -> None:
        client = _FakeClient([_FakeResponse(429), _FakeResponse(429)])
        with patch.object(brave.time, "sleep") as sleep_mock:
            result = brave.web_search(
                "x",
                api_key="fake",
                http_client_factory=_factory_for(client),
            )
        self.assertEqual(result["error"], "brave search rate-limited after retry")
        self.assertEqual(result["status_code"], 429)
        # Slept once between the two 429s.
        self.assertEqual(sleep_mock.call_count, 1)
        self.assertEqual(len(client.calls), 2)

    def test_200_after_one_429_succeeds(self) -> None:
        client = _FakeClient([_FakeResponse(429), _FakeResponse(200, {"web": {"results": []}})])
        with patch.object(brave.time, "sleep"):
            result = brave.web_search(
                "x",
                api_key="fake",
                http_client_factory=_factory_for(client),
            )
        self.assertNotIn("error", result)
        self.assertEqual(result["result_count"], 0)

    def test_non_200_surfaces_status_code(self) -> None:
        client = _FakeClient([_FakeResponse(500)])
        result = brave.web_search(
            "x",
            api_key="fake",
            http_client_factory=_factory_for(client),
        )
        self.assertEqual(result["status_code"], 500)
        self.assertIn("500", result["error"])

    def test_non_json_response_returns_error(self) -> None:
        client = _FakeClient([_FakeResponse(200, json_exc=ValueError("bad json"))])
        result = brave.web_search(
            "x",
            api_key="fake",
            http_client_factory=_factory_for(client),
        )
        self.assertIn("not JSON", result["error"])


class CreateToolPluginTests(unittest.TestCase):
    """Manifest entrypoint surfaces the expected descriptor shape."""

    def test_descriptor_lists_web_search_tool(self) -> None:
        plugin = brave.create_tool_plugin()
        self.assertEqual(plugin["tool_server_ref"], "tool_server:brave_search")
        self.assertEqual(plugin["server_name"], "brave-search")
        self.assertEqual(plugin["tool_names"], ["web_search"])
        handlers = plugin["handlers"]
        assert isinstance(handlers, dict)
        self.assertIn("web_search", handlers)


class NormalizeResultsTests(unittest.TestCase):
    """``_normalize_brave_results`` tolerates malformed payloads gracefully."""

    def test_missing_web_field_returns_empty_results(self) -> None:
        # Payload with no ``web`` key at all (e.g. an error response that
        # still parsed as JSON). Should give back ``results: []``, not crash.
        result = brave._normalize_brave_results("q", {})
        self.assertEqual(result["results"], [])
        self.assertEqual(result["result_count"], 0)

    def test_web_results_non_list_falls_back_to_empty(self) -> None:
        # ``web.results`` is a dict instead of a list — the normalizer
        # collapses to ``[]`` rather than iterating something nonsensical.
        result = brave._normalize_brave_results("q", {"web": {"results": {"nope": 1}}})
        self.assertEqual(result["results"], [])


class TextResultHelperTests(unittest.TestCase):
    """``_text_result`` wraps payloads in the MCP text-content envelope."""

    def test_dict_payload_is_json_serialized(self) -> None:
        envelope = brave._text_result({"hello": "world"})
        self.assertEqual(envelope["content"][0]["type"], "text")
        self.assertIn('"hello"', envelope["content"][0]["text"])
        self.assertIn('"world"', envelope["content"][0]["text"])

    def test_string_payload_is_passed_through_verbatim(self) -> None:
        # When the input is already a string we don't re-serialize it.
        envelope = brave._text_result("plain text")
        self.assertEqual(envelope["content"][0]["text"], "plain text")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
