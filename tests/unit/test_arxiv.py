"""Unit tests for the arxiv adapter (#128 PR-2).

httpx is an ``[agents]`` extra, not installed in the unit-test env;
the production-only lazy-import branch is ``# pragma: no cover``.
Tests inject ``http_client_factory`` with a fake response returning
canned Atom XML, and a ``clock`` test seam so we can verify the
3-second rate-limit guard without actually sleeping.
"""

from __future__ import annotations

import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from praxist.plugins.tools.arxiv import adapter as arxiv

# --------------------------------------------------------------------------- #
# Fake httpx response shim
# --------------------------------------------------------------------------- #


class _FakeResponse:
    """Minimal ``httpx.Response``-shaped stub for arxiv tests."""

    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _FakeClient:
    """Test seam: records the GET call and replays scripted responses."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any]) -> _FakeResponse:
        self.calls.append({"url": url, "params": dict(params)})
        if not self._responses:
            raise AssertionError("test scripted no further responses")
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


# --------------------------------------------------------------------------- #
# Canned Atom XML fixtures
# --------------------------------------------------------------------------- #


_ATOM_SEARCH_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <opensearch:totalResults>2</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2305.12345v3</id>
    <updated>2024-06-01T00:00:00Z</updated>
    <published>2023-05-15T00:00:00Z</published>
    <title>First Paper</title>
    <summary>Abstract of the first paper.</summary>
    <author><name>Alice One</name></author>
    <author><name>Bob Two</name></author>
    <category term="cs.AI"/>
    <category term="stat.ML"/>
    <link rel="alternate" type="text/html" href="http://arxiv.org/abs/2305.12345v3"/>
    <link rel="related" type="application/pdf" href="http://arxiv.org/pdf/2305.12345v3"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.99999v1</id>
    <updated>2024-01-10T00:00:00Z</updated>
    <published>2024-01-10T00:00:00Z</published>
    <title>Second Paper</title>
    <summary>Abstract of the second paper.</summary>
    <author><name>Carol Three</name></author>
    <category term="cs.LG"/>
    <link rel="alternate" type="text/html" href="http://arxiv.org/abs/2401.99999v1"/>
    <link rel="related" type="application/pdf" href="http://arxiv.org/pdf/2401.99999v1"/>
  </entry>
</feed>
"""


_ATOM_EMPTY_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>0</opensearch:totalResults>
</feed>
"""


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


class ArxivSearchTests(unittest.TestCase):
    """``arxiv_search`` parses Atom XML into the normalized result schema."""

    def setUp(self) -> None:
        arxiv._reset_rate_limit_for_tests()

    def _zero_clock(self) -> tuple[Any, Any]:
        """Clock that always returns 0 — bypasses the rate-limit sleep."""
        return (lambda: 0.0, lambda _s: None)

    def test_empty_query_returns_error(self) -> None:
        result = arxiv.arxiv_search(
            "  ",
            http_client_factory=_factory_for(_FakeClient([])),
            clock=self._zero_clock(),
        )
        self.assertEqual(result["error"], "query is required")
        self.assertEqual(result["schema_version"], 1)

    def test_successful_search_returns_normalized_entries(self) -> None:
        client = _FakeClient([_FakeResponse(200, _ATOM_SEARCH_FIXTURE)])
        result = arxiv.arxiv_search(
            "diffusion models",
            max_results=5,
            sort_by="submitted_date",
            http_client_factory=_factory_for(client),
            clock=self._zero_clock(),
        )
        self.assertNotIn("error", result)
        self.assertEqual(result["query"], "diffusion models")
        self.assertEqual(result["total_results"], 2)
        self.assertEqual(result["result_count"], 2)
        first = result["results"][0]
        self.assertEqual(first["arxiv_id"], "2305.12345v3")
        self.assertEqual(first["title"], "First Paper")
        self.assertEqual(first["authors"], ["Alice One", "Bob Two"])
        self.assertEqual(first["categories"], ["cs.AI", "stat.ML"])
        self.assertEqual(first["pdf_url"], "http://arxiv.org/pdf/2305.12345v3")
        self.assertEqual(first["html_url"], "http://arxiv.org/abs/2305.12345v3")
        # Request sent the sortBy alias.
        call = client.calls[0]
        self.assertEqual(call["url"], "https://export.arxiv.org/api/query")
        self.assertEqual(call["params"]["sortBy"], "submittedDate")
        self.assertEqual(call["params"]["max_results"], 5)

    def test_default_http_client_uses_https_and_follows_redirects(self) -> None:
        calls: list[dict[str, Any]] = []
        client_kwargs: list[dict[str, Any]] = []

        class RequestError(Exception):
            pass

        class DefaultClient:
            def __init__(self, **kwargs: Any) -> None:
                client_kwargs.append(dict(kwargs))

            def get(self, url: str, params: dict[str, Any]) -> _FakeResponse:
                calls.append({"url": url, "params": dict(params)})
                return _FakeResponse(200, _ATOM_EMPTY_FIXTURE)

            def __enter__(self) -> DefaultClient:
                return self

            def __exit__(self, *exc: Any) -> None:
                return None

        fake_httpx = SimpleNamespace(Client=DefaultClient, RequestError=RequestError)

        with patch.dict(sys.modules, {"httpx": fake_httpx}):
            result = arxiv._fetch_arxiv_xml(  # noqa: SLF001 - intentional adapter seam.
                {"search_query": "x", "start": 0, "max_results": 1},
                clock=self._zero_clock(),
            )

        self.assertEqual(result["xml"], _ATOM_EMPTY_FIXTURE)
        self.assertEqual(calls[0]["url"], "https://export.arxiv.org/api/query")
        self.assertEqual(client_kwargs[0]["timeout"], arxiv._DEFAULT_TIMEOUT_SECONDS)
        self.assertIs(client_kwargs[0]["follow_redirects"], True)

    def test_max_results_clamped(self) -> None:
        client = _FakeClient([_FakeResponse(200, _ATOM_EMPTY_FIXTURE)])
        arxiv.arxiv_search(
            "x",
            max_results=999,
            http_client_factory=_factory_for(client),
            clock=self._zero_clock(),
        )
        self.assertEqual(client.calls[0]["params"]["max_results"], 50)

    def test_unknown_sort_falls_back_to_relevance(self) -> None:
        client = _FakeClient([_FakeResponse(200, _ATOM_EMPTY_FIXTURE)])
        arxiv.arxiv_search(
            "x",
            sort_by="not_a_real_sort",
            http_client_factory=_factory_for(client),
            clock=self._zero_clock(),
        )
        self.assertEqual(client.calls[0]["params"]["sortBy"], "relevance")

    def test_non_200_surfaces_status_code(self) -> None:
        result = arxiv.arxiv_search(
            "x",
            http_client_factory=_factory_for(_FakeClient([_FakeResponse(503, "")])),
            clock=self._zero_clock(),
        )
        self.assertEqual(result["status_code"], 503)
        self.assertIn("503", result["error"])

    def test_malformed_xml_returns_empty_results_not_crash(self) -> None:
        result = arxiv.arxiv_search(
            "x",
            http_client_factory=_factory_for(_FakeClient([_FakeResponse(200, "not xml")])),
            clock=self._zero_clock(),
        )
        self.assertEqual(result["results"], [])
        self.assertEqual(result["total_results"], 0)


# --------------------------------------------------------------------------- #
# Get one paper by id
# --------------------------------------------------------------------------- #


class ArxivGetTests(unittest.TestCase):
    def setUp(self) -> None:
        arxiv._reset_rate_limit_for_tests()

    def _zero_clock(self) -> tuple[Any, Any]:
        return (lambda: 0.0, lambda _s: None)

    def test_empty_id_returns_error(self) -> None:
        result = arxiv.arxiv_get(
            " ",
            http_client_factory=_factory_for(_FakeClient([])),
            clock=self._zero_clock(),
        )
        self.assertEqual(result["error"], "arxiv_id is required")

    def test_found_paper_returns_normalized_metadata(self) -> None:
        client = _FakeClient([_FakeResponse(200, _ATOM_SEARCH_FIXTURE)])
        result = arxiv.arxiv_get(
            "2305.12345v3",
            http_client_factory=_factory_for(client),
            clock=self._zero_clock(),
        )
        self.assertNotIn("error", result)
        self.assertEqual(result["paper"]["arxiv_id"], "2305.12345v3")
        self.assertEqual(result["paper"]["title"], "First Paper")
        # Request used id_list, not search_query.
        self.assertEqual(client.calls[0]["params"]["id_list"], "2305.12345v3")

    def test_unknown_id_returns_error(self) -> None:
        result = arxiv.arxiv_get(
            "9999.0000",
            http_client_factory=_factory_for(
                _FakeClient([_FakeResponse(200, _ATOM_EMPTY_FIXTURE)])
            ),
            clock=self._zero_clock(),
        )
        self.assertIn("no arxiv paper with id", result["error"])


# --------------------------------------------------------------------------- #
# Recent submissions
# --------------------------------------------------------------------------- #


class ArxivRecentTests(unittest.TestCase):
    def setUp(self) -> None:
        arxiv._reset_rate_limit_for_tests()

    def test_empty_category_returns_error(self) -> None:
        result = arxiv.arxiv_recent(
            "",
            http_client_factory=_factory_for(_FakeClient([])),
            clock=(lambda: 0.0, lambda _s: None),
        )
        self.assertEqual(result["error"], "category is required")

    def test_recent_filters_by_submission_age(self) -> None:
        # Set "now" to 2024-06-15. The fixture's first entry was published
        # 2023-05-15 (>1 year old), second was 2024-01-10 (~6 months old).
        # With days=180 we should keep only the second.
        import datetime

        now_epoch = datetime.datetime(2024, 6, 15, tzinfo=datetime.UTC).timestamp()
        clock = (lambda: now_epoch, lambda _s: None)
        client = _FakeClient([_FakeResponse(200, _ATOM_SEARCH_FIXTURE)])

        result = arxiv.arxiv_recent(
            "cs.AI",
            days=90,
            http_client_factory=_factory_for(client),
            clock=clock,
        )
        # The fixture has 2 entries; only the 2024-01-10 one is within 90 days
        # of 2024-06-15? Actually 2024-01-10 to 2024-06-15 is ~5 months ago
        # which is > 90 days. So 0 results survive. Verify the filtering ran.
        self.assertEqual(result["category"], "cs.AI")
        self.assertEqual(result["days"], 90)
        self.assertEqual(result["result_count"], 0)
        # Request sent the cat: search_query.
        self.assertEqual(client.calls[0]["params"]["search_query"], "cat:cs.AI")
        self.assertEqual(client.calls[0]["params"]["sortBy"], "submittedDate")

    def test_recent_keeps_entries_within_window(self) -> None:
        import datetime

        # Set "now" close to 2024-01-15 so the 2024-01-10 entry is within
        # the last 30 days but the 2023-05-15 one is not.
        now_epoch = datetime.datetime(2024, 1, 15, tzinfo=datetime.UTC).timestamp()
        clock = (lambda: now_epoch, lambda _s: None)
        client = _FakeClient([_FakeResponse(200, _ATOM_SEARCH_FIXTURE)])

        result = arxiv.arxiv_recent(
            "cs.AI",
            days=30,
            http_client_factory=_factory_for(client),
            clock=clock,
        )
        self.assertEqual(result["result_count"], 1)
        self.assertEqual(result["results"][0]["arxiv_id"], "2401.99999v1")

    def test_days_and_max_results_clamped(self) -> None:
        client = _FakeClient([_FakeResponse(200, _ATOM_EMPTY_FIXTURE)])
        result = arxiv.arxiv_recent(
            "cs.AI",
            days=999,
            max_results=999,
            http_client_factory=_factory_for(client),
            clock=(lambda: 0.0, lambda _s: None),
        )
        self.assertEqual(result["days"], 90)
        self.assertEqual(client.calls[0]["params"]["max_results"], 50)


# --------------------------------------------------------------------------- #
# Rate limit
# --------------------------------------------------------------------------- #


class ArxivRateLimitTests(unittest.TestCase):
    def setUp(self) -> None:
        arxiv._reset_rate_limit_for_tests()

    def test_first_call_does_not_sleep(self) -> None:
        """``_LAST_REQUEST_TIME == 0.0`` means "fresh process" — no sleep."""
        sleeps: list[float] = []
        clock = (lambda: 100.0, lambda s: sleeps.append(s))
        arxiv.arxiv_search(
            "q",
            http_client_factory=_factory_for(
                _FakeClient([_FakeResponse(200, _ATOM_EMPTY_FIXTURE)])
            ),
            clock=clock,
        )
        self.assertEqual(sleeps, [])

    def test_second_call_within_window_sleeps_remainder(self) -> None:
        """A second call 0.5 s after the first sleeps ~2.5 s (3-0.5)."""
        now_ref = [100.0]
        sleeps: list[float] = []

        def time_fn() -> float:
            return now_ref[0]

        def sleep_fn(seconds: float) -> None:
            sleeps.append(seconds)
            now_ref[0] += seconds

        clock = (time_fn, sleep_fn)
        client = _FakeClient(
            [
                _FakeResponse(200, _ATOM_EMPTY_FIXTURE),
                _FakeResponse(200, _ATOM_EMPTY_FIXTURE),
            ]
        )

        # First call at t=100 — no sleep, sets last=100.
        arxiv.arxiv_search("q", http_client_factory=_factory_for(client), clock=clock)
        self.assertEqual(sleeps, [])

        # Bump time to 100.5 — second call should sleep ~2.5 s to reach 3 s gap.
        now_ref[0] = 100.5
        arxiv.arxiv_search("q", http_client_factory=_factory_for(client), clock=clock)
        self.assertEqual(len(sleeps), 1)
        self.assertAlmostEqual(sleeps[0], 2.5, places=2)

    def test_call_outside_window_does_not_sleep(self) -> None:
        """A second call > 3 s after the first must not sleep."""
        now_ref = [100.0]
        sleeps: list[float] = []
        clock = (lambda: now_ref[0], lambda s: sleeps.append(s))
        client = _FakeClient(
            [
                _FakeResponse(200, _ATOM_EMPTY_FIXTURE),
                _FakeResponse(200, _ATOM_EMPTY_FIXTURE),
            ]
        )
        arxiv.arxiv_search("q", http_client_factory=_factory_for(client), clock=clock)
        now_ref[0] = 200.0  # well past the 3 s window
        arxiv.arxiv_search("q", http_client_factory=_factory_for(client), clock=clock)
        self.assertEqual(sleeps, [])


# --------------------------------------------------------------------------- #
# Plugin manifest + codex_sdk MCP routing
# --------------------------------------------------------------------------- #


class CreateToolPluginTests(unittest.TestCase):
    def test_descriptor_lists_three_tools(self) -> None:
        plugin = arxiv.create_tool_plugin()
        self.assertEqual(plugin["tool_server_ref"], "tool_server:arxiv")
        self.assertEqual(plugin["server_name"], "arxiv")
        self.assertEqual(
            plugin["tool_names"],
            ["arxiv_search", "arxiv_get", "arxiv_recent"],
        )


class CodexSdkMcpRoutingTests(unittest.TestCase):
    def test_arxiv_server_routes_to_arxiv_factory(self) -> None:
        import sys as _sys

        from praxist.plugins.agent_runtimes.codex_sdk._mcp import (
            MCP_STDIO_MODULE,
            mcp_configuration,
        )

        result = mcp_configuration([{"server_name": "arxiv"}])
        server = result.config["mcp_servers"]["arxiv"]
        self.assertEqual(server["command"], _sys.executable)
        self.assertEqual(server["args"][:2], ["-m", MCP_STDIO_MODULE])
        self.assertIn(
            "praxist.plugins.tools.arxiv.adapter:create_arxiv_server",
            server["args"][2],
        )
        self.assertEqual(result.warnings, ())


class TextResultHelperTests(unittest.TestCase):
    def test_dict_payload_is_json_serialized(self) -> None:
        envelope = arxiv._text_result({"k": "v"})
        self.assertEqual(envelope["content"][0]["type"], "text")
        self.assertIn('"k"', envelope["content"][0]["text"])

    def test_string_payload_passed_through(self) -> None:
        envelope = arxiv._text_result("hi")
        self.assertEqual(envelope["content"][0]["text"], "hi")


class InternalHelperTests(unittest.TestCase):
    """Direct coverage of small parsers that survive otherwise-thin tests."""

    def setUp(self) -> None:
        arxiv._reset_rate_limit_for_tests()

    def test_parse_total_results_handles_missing_node(self) -> None:
        xml = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        self.assertEqual(arxiv._parse_total_results(xml), 0)

    def test_parse_total_results_handles_non_int_text(self) -> None:
        xml = (
            '<feed xmlns="http://www.w3.org/2005/Atom" '
            'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
            "<opensearch:totalResults>not-a-number</opensearch:totalResults></feed>"
        )
        self.assertEqual(arxiv._parse_total_results(xml), 0)

    def test_parse_total_results_handles_malformed_xml(self) -> None:
        self.assertEqual(arxiv._parse_total_results("not xml at all"), 0)

    def test_parse_iso8601_empty_returns_zero(self) -> None:
        self.assertEqual(arxiv._parse_iso8601(""), 0.0)

    def test_parse_iso8601_malformed_returns_zero(self) -> None:
        self.assertEqual(arxiv._parse_iso8601("nope"), 0.0)

    def test_parse_iso8601_zulu_suffix_round_trips(self) -> None:
        self.assertGreater(arxiv._parse_iso8601("2024-01-15T00:00:00Z"), 0)

    def test_resolve_clock_default_returns_real_time_sleep(self) -> None:
        import time as _time

        time_fn, sleep_fn = arxiv._resolve_clock(None)
        self.assertIs(time_fn, _time.time)
        self.assertIs(sleep_fn, _time.sleep)

    def test_resolve_clock_bad_type_raises(self) -> None:
        with self.assertRaises(TypeError):
            arxiv._resolve_clock("not a tuple")

    def test_response_text_coerced_when_not_string(self) -> None:
        """``_fetch_arxiv_xml`` defends against responses whose ``.text`` isn't str."""

        class _BytesResponse:
            status_code = 200
            text = b"<feed/>"  # bytes, not str — adapter must coerce.

        @contextmanager
        def _factory():
            class _Client:
                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    return None

                def get(self, _url, params):
                    return _BytesResponse()

            yield _Client()

        result = arxiv.arxiv_search(
            "x",
            http_client_factory=_factory,
            clock=(lambda: 100.0, lambda _s: None),
        )
        self.assertEqual(result["results"], [])

    def test_arxiv_get_propagates_fetch_error(self) -> None:
        result = arxiv.arxiv_get(
            "2305.12345",
            http_client_factory=_factory_for(_FakeClient([_FakeResponse(500, "")])),
            clock=(lambda: 100.0, lambda _s: None),
        )
        self.assertIn("error", result)
        self.assertEqual(result["status_code"], 500)

    def test_arxiv_recent_propagates_fetch_error(self) -> None:
        result = arxiv.arxiv_recent(
            "cs.AI",
            http_client_factory=_factory_for(_FakeClient([_FakeResponse(500, "")])),
            clock=(lambda: 100.0, lambda _s: None),
        )
        self.assertIn("error", result)
        self.assertEqual(result["status_code"], 500)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
