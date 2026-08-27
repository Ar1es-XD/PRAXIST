"""Unit tests for the pdf_reader adapter (#128 PR-3).

The production path requires ``poppler-utils`` + ``tesseract-ocr`` +
``pdf2image`` + ``pytesseract`` + ``pypdf`` + ``httpx`` — too much to
install just to test the adapter plumbing. Instead the handler
exposes four test seams:

* ``rasterize_engine(pdf_bytes, dpi, page_numbers) -> [image_bytes]``
* ``ocr_engine(image_bytes, lang) -> text``
* ``metadata_engine(pdf_bytes) -> {title, author, num_pages, ...}``
* ``http_client_factory`` + ``binary_probe`` for fetch / PATH checks.

Tests script all four to verify range parsing, cache hit/miss,
URL fetching, missing-binary error paths, size cap, dpi/lang clamp,
``pdf_metadata`` happy path, manifest shape, codex_sdk MCP routing.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from praxist.plugins.tools.pdf_reader import adapter as pdf_reader


def _stub_probe(missing: tuple[str, ...] = ()) -> Any:
    """Return a ``binary_probe`` callable that reports given binaries missing."""

    def _probe(name: str) -> str | None:
        if name in missing:
            return None
        return f"/usr/bin/{name}"

    return _probe


def _fake_pdf_bytes(page_count: int = 2) -> bytes:
    """Build a byte sequence that ``_quick_page_count`` will count correctly."""
    body = b"%PDF-fake\n" + b"/Type /Page  " * page_count
    # Pad to make the hash differ across test inputs.
    return body + b"\n%%EOF\n"


class PageRangeParserTests(unittest.TestCase):
    """``_parse_page_range`` handles the three documented input shapes."""

    def test_simple_range(self) -> None:
        result = pdf_reader._parse_page_range("1-5", num_pages=20)
        self.assertEqual(result["pages"], [1, 2, 3, 4, 5])

    def test_mixed_list_and_ranges(self) -> None:
        result = pdf_reader._parse_page_range("3,7-9,12", num_pages=20)
        self.assertEqual(result["pages"], [3, 7, 8, 9, 12])

    def test_all_keyword(self) -> None:
        result = pdf_reader._parse_page_range("all", num_pages=4)
        self.assertEqual(result["pages"], [1, 2, 3, 4])

    def test_clamp_to_doc_length(self) -> None:
        result = pdf_reader._parse_page_range("1-100", num_pages=5)
        self.assertEqual(result["pages"], [1, 2, 3, 4, 5])
        self.assertTrue(any("clamped" in w for w in result.get("warnings", [])))

    def test_dedupe_overlap(self) -> None:
        result = pdf_reader._parse_page_range("1-3,2-5", num_pages=10)
        self.assertEqual(result["pages"], [1, 2, 3, 4, 5])

    def test_invalid_token_returns_error(self) -> None:
        result = pdf_reader._parse_page_range("abc", num_pages=10)
        self.assertIn("invalid pages token", result["error"])

    def test_zero_start_rejected(self) -> None:
        result = pdf_reader._parse_page_range("0-3", num_pages=10)
        self.assertIn("invalid pages range", result["error"])

    def test_reversed_range_rejected(self) -> None:
        result = pdf_reader._parse_page_range("5-3", num_pages=10)
        self.assertIn("invalid pages range", result["error"])


class PdfReadHappyPathTests(unittest.TestCase):
    """``pdf_read`` rasterizes, OCRs, caches, and returns per-page text."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.pdf_path = self.root / "paper.pdf"
        self.pdf_path.write_bytes(_fake_pdf_bytes(page_count=10))
        self.cache_root = self.root / "cache"

    def _engines(self, ocr_text_by_page: dict[int, str]) -> tuple[Any, Any, list[Any]]:
        """Return (rasterize_engine, ocr_engine, calls) test pair."""
        calls: list[dict[str, Any]] = []

        def rasterize(pdf_bytes: bytes, dpi: int, pages: list[int]) -> list[bytes]:
            calls.append({"kind": "rasterize", "dpi": dpi, "pages": pages})
            return [f"<image-page-{page}>".encode() for page in pages]

        def ocr(image_bytes: bytes, lang: str) -> str:
            calls.append({"kind": "ocr", "lang": lang, "bytes": image_bytes})
            tag = image_bytes.decode()
            # Tag format: <image-page-N>
            page_num = int(tag.rsplit("-", 1)[1].rstrip(">"))
            return ocr_text_by_page.get(page_num, "")

        return rasterize, ocr, calls

    def test_basic_local_path_with_cache_miss(self) -> None:
        rasterize, ocr, calls = self._engines({1: "Page one text", 2: "Page two text"})
        result = pdf_reader.pdf_read(
            str(self.pdf_path),
            pages="1-2",
            dpi=200,
            lang="eng",
            rasterize_engine=rasterize,
            ocr_engine=ocr,
            cache_root=self.cache_root,
            binary_probe=_stub_probe(),
        )
        self.assertNotIn("error", result)
        self.assertEqual(result["num_pages"], 10)
        self.assertEqual(result["requested_pages"], [1, 2])
        self.assertEqual(len(result["pages"]), 2)
        self.assertEqual(result["pages"][0]["page_num"], 1)
        self.assertEqual(result["pages"][0]["text"], "Page one text")
        self.assertEqual(result["pages"][1]["text"], "Page two text")
        # All requested pages were cache misses.
        self.assertEqual(result["cached_pages"], [])
        self.assertEqual(result["dpi"], 200)
        # rasterize called once with all miss pages; ocr called once per page.
        rasterize_calls = [c for c in calls if c["kind"] == "rasterize"]
        ocr_calls = [c for c in calls if c["kind"] == "ocr"]
        self.assertEqual(len(rasterize_calls), 1)
        self.assertEqual(len(ocr_calls), 2)

    def test_second_run_hits_cache_zero_engine_calls(self) -> None:
        # First run populates the cache.
        rasterize, ocr, _ = self._engines({1: "First", 2: "Second"})
        pdf_reader.pdf_read(
            str(self.pdf_path),
            pages="1-2",
            rasterize_engine=rasterize,
            ocr_engine=ocr,
            cache_root=self.cache_root,
            binary_probe=_stub_probe(),
        )

        # Second run with engines that would explode if called.
        def fail(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("cached run must not call the rasterize/ocr engine")

        result = pdf_reader.pdf_read(
            str(self.pdf_path),
            pages="1-2",
            rasterize_engine=fail,
            ocr_engine=fail,
            cache_root=self.cache_root,
            binary_probe=_stub_probe(),
        )
        self.assertNotIn("error", result)
        self.assertEqual(result["cached_pages"], [1, 2])
        self.assertEqual(result["pages"][0]["text"], "First")
        self.assertEqual(result["pages"][1]["text"], "Second")

    def test_dpi_lang_change_invalidates_cache(self) -> None:
        rasterize, ocr, calls = self._engines({1: "default", 2: "default"})
        pdf_reader.pdf_read(
            str(self.pdf_path),
            pages="1-2",
            dpi=200,
            lang="eng",
            rasterize_engine=rasterize,
            ocr_engine=ocr,
            cache_root=self.cache_root,
            binary_probe=_stub_probe(),
        )
        # Different dpi → cache miss again.
        rasterize2, ocr2, calls2 = self._engines({1: "hi-dpi", 2: "hi-dpi"})
        result = pdf_reader.pdf_read(
            str(self.pdf_path),
            pages="1-2",
            dpi=300,
            lang="eng",
            rasterize_engine=rasterize2,
            ocr_engine=ocr2,
            cache_root=self.cache_root,
            binary_probe=_stub_probe(),
        )
        self.assertEqual(result["cached_pages"], [])
        self.assertEqual(result["pages"][0]["text"], "hi-dpi")
        rasterize_calls = [c for c in calls2 if c["kind"] == "rasterize"]
        self.assertEqual(len(rasterize_calls), 1)


class PdfReadValidationTests(unittest.TestCase):
    """Argument-validation and error-path tests for ``pdf_read``."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_empty_pdf_returns_error(self) -> None:
        result = pdf_reader.pdf_read("", binary_probe=_stub_probe())
        self.assertEqual(result["error"], "pdf is required")

    def test_missing_local_file_returns_error(self) -> None:
        result = pdf_reader.pdf_read(
            str(self.root / "missing.pdf"),
            binary_probe=_stub_probe(),
        )
        self.assertIn("local pdf not found", result["error"])

    def test_missing_pdftoppm_friendly_error(self) -> None:
        pdf_path = self.root / "x.pdf"
        pdf_path.write_bytes(_fake_pdf_bytes())
        result = pdf_reader.pdf_read(
            str(pdf_path),
            binary_probe=_stub_probe(missing=("pdftoppm",)),
        )
        self.assertIn("pdftoppm", result["error"])
        self.assertIn("brew install poppler", result["error"])

    def test_missing_tesseract_friendly_error(self) -> None:
        pdf_path = self.root / "x.pdf"
        pdf_path.write_bytes(_fake_pdf_bytes())
        result = pdf_reader.pdf_read(
            str(pdf_path),
            binary_probe=_stub_probe(missing=("tesseract",)),
        )
        self.assertIn("tesseract", result["error"])

    def test_pdf_too_large_rejected(self) -> None:
        pdf_path = self.root / "huge.pdf"
        # 51 MB > cap
        pdf_path.write_bytes(b"%PDF-fake\n" + b"x" * (51 * 1024 * 1024))
        result = pdf_reader.pdf_read(
            str(pdf_path),
            binary_probe=_stub_probe(),
            cache_root=self.root,
        )
        self.assertIn("exceeds", result["error"])

    def test_invalid_pages_token_returns_error(self) -> None:
        pdf_path = self.root / "x.pdf"
        pdf_path.write_bytes(_fake_pdf_bytes())

        def never_called(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("engine should never run for invalid pages spec")

        result = pdf_reader.pdf_read(
            str(pdf_path),
            pages="garbage",
            rasterize_engine=never_called,
            ocr_engine=never_called,
            cache_root=self.root,
            binary_probe=_stub_probe(),
        )
        self.assertIn("invalid pages token", result["error"])

    def test_dpi_clamped_to_max(self) -> None:
        pdf_path = self.root / "x.pdf"
        pdf_path.write_bytes(_fake_pdf_bytes())
        captured_dpi: list[int] = []

        def rasterize(_pdf: bytes, dpi: int, pages: list[int]) -> list[bytes]:
            captured_dpi.append(dpi)
            return [b"img" for _ in pages]

        def ocr(_image: bytes, _lang: str) -> str:
            return "ok"

        pdf_reader.pdf_read(
            str(pdf_path),
            pages="1",
            dpi=9999,
            rasterize_engine=rasterize,
            ocr_engine=ocr,
            cache_root=self.root / "cache",
            binary_probe=_stub_probe(),
        )
        self.assertEqual(captured_dpi[0], 450)

    def test_unsafe_lang_sanitized_to_eng(self) -> None:
        self.assertEqual(pdf_reader._sanitize_lang("eng+equ"), "eng+equ")
        # Punctuation / paths blocked — would otherwise let an operator
        # inject ``"; rm -rf /"`` style args into the tesseract subprocess.
        self.assertEqual(pdf_reader._sanitize_lang("eng;rm"), "eng")
        self.assertEqual(pdf_reader._sanitize_lang(""), "eng")


class PdfReadUrlFetchTests(unittest.TestCase):
    """URL-mode goes through ``http_client_factory``."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def _fake_url_factory(self, response: Any) -> Any:
        class _Client:
            def get(self, _url: str) -> Any:
                return response

            def __enter__(self) -> Any:
                return self

            def __exit__(self, *_exc: Any) -> None:
                return None

        @contextmanager
        def _inner():
            yield _Client()

        return _inner

    def test_successful_url_fetch(self) -> None:
        pdf_bytes = _fake_pdf_bytes()

        class _Response:
            status_code = 200
            content = pdf_bytes

        result = pdf_reader.pdf_read(
            "https://example.com/paper.pdf",
            pages="1",
            rasterize_engine=lambda _p, _d, pages: [b"img" for _ in pages],
            ocr_engine=lambda _i, _l: "url text",
            cache_root=self.root,
            http_client_factory=self._fake_url_factory(_Response()),
            binary_probe=_stub_probe(),
        )
        self.assertNotIn("error", result)
        self.assertEqual(result["pages"][0]["text"], "url text")

    def test_url_non_200_surfaces_status(self) -> None:
        class _Response:
            status_code = 503
            content = b""
            text = ""

        result = pdf_reader.pdf_read(
            "https://example.com/paper.pdf",
            pages="1",
            rasterize_engine=lambda *a, **k: [],
            ocr_engine=lambda *a, **k: "",
            cache_root=self.root,
            http_client_factory=self._fake_url_factory(_Response()),
            binary_probe=_stub_probe(),
        )
        self.assertEqual(result["status_code"], 503)
        self.assertIn("503", result["error"])


class PdfMetadataTests(unittest.TestCase):
    """``pdf_metadata`` uses the injected engine and never OCRs."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.pdf_path = self.root / "paper.pdf"
        self.pdf_path.write_bytes(_fake_pdf_bytes(page_count=12))

    def test_engine_runs_no_ocr(self) -> None:
        def metadata(pdf_bytes: bytes) -> dict[str, Any]:
            self.assertGreater(len(pdf_bytes), 0)
            return {"title": "Sample Paper", "author": "Author Name", "num_pages": 12}

        result = pdf_reader.pdf_metadata(str(self.pdf_path), metadata_engine=metadata)
        self.assertEqual(result["title"], "Sample Paper")
        self.assertEqual(result["author"], "Author Name")
        self.assertEqual(result["num_pages"], 12)
        self.assertEqual(result["pdf"], str(self.pdf_path))

    def test_empty_pdf_arg(self) -> None:
        result = pdf_reader.pdf_metadata("", metadata_engine=lambda _b: {})
        self.assertEqual(result["error"], "pdf is required")


class HelperTests(unittest.TestCase):
    def test_text_result_dict(self) -> None:
        envelope = pdf_reader._text_result({"a": 1})
        self.assertEqual(envelope["content"][0]["type"], "text")
        self.assertIn('"a"', envelope["content"][0]["text"])

    def test_text_result_string(self) -> None:
        envelope = pdf_reader._text_result("plain")
        self.assertEqual(envelope["content"][0]["text"], "plain")

    def test_resolve_cache_root_prefers_override(self) -> None:
        path = pdf_reader._resolve_cache_root(Path("/tmp/x"))
        self.assertEqual(path, Path("/tmp/x") / pdf_reader._CACHE_SUBDIR)

    def test_content_hash_used_as_cache_subdir(self) -> None:
        # The cache layout is intentionally content-addressed; changing
        # the bytes changes the directory.
        bytes_a = _fake_pdf_bytes(page_count=1)
        bytes_b = _fake_pdf_bytes(page_count=2)
        self.assertNotEqual(
            hashlib.sha256(bytes_a).hexdigest(),
            hashlib.sha256(bytes_b).hexdigest(),
        )

    def test_resolve_cache_root_honors_praxist_state_dir_env(self) -> None:
        import os as _os
        from unittest.mock import patch as _patch

        with _patch.dict(_os.environ, {"PRAXIST_STATE_DIR": "/tmp/praxist-state"}, clear=False):
            path = pdf_reader._resolve_cache_root(None)
        self.assertEqual(path, Path("/tmp/praxist-state") / pdf_reader._CACHE_SUBDIR)

    def test_resolve_cache_root_falls_back_to_xdg(self) -> None:
        import os as _os
        from unittest.mock import patch as _patch

        env = {"XDG_DATA_HOME": "/tmp/xdg"}
        with _patch.dict(_os.environ, env, clear=False):
            _os.environ.pop("PRAXIST_STATE_DIR", None)
            path = pdf_reader._resolve_cache_root(None)
        self.assertEqual(path, Path("/tmp/xdg") / "praxist" / pdf_reader._CACHE_SUBDIR)

    def test_parse_page_range_empty_string_returns_error(self) -> None:
        result = pdf_reader._parse_page_range("", num_pages=10)
        self.assertIn("pages spec is empty", result["error"])

    def test_parse_page_range_start_past_doc_dropped(self) -> None:
        # Start beyond doc length silently skips that token.
        result = pdf_reader._parse_page_range("100-200,1-2", num_pages=10)
        self.assertEqual(result["pages"], [1, 2])

    def test_page_range_truncates_when_expanded_past_max(self) -> None:
        """A range that expands past ``_MAX_PAGES_PER_REQUEST`` is truncated."""
        # The ``all`` keyword expands to num_pages; 50 pages > 30 cap.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_path = root / "big.pdf"
            pdf_path.write_bytes(_fake_pdf_bytes(page_count=50))
            captured_pages: list[int] = []

            def rasterize(_pdf: bytes, _dpi: int, pages: list[int]) -> list[bytes]:
                captured_pages.extend(pages)
                return [b"img" for _ in pages]

            def ocr(_image: bytes, _lang: str) -> str:
                return "ok"

            result = pdf_reader.pdf_read(
                str(pdf_path),
                pages="all",
                rasterize_engine=rasterize,
                ocr_engine=ocr,
                cache_root=root,
                binary_probe=_stub_probe(),
            )
            self.assertEqual(len(result["requested_pages"]), 30)
            self.assertTrue(any("truncating" in w for w in result["warnings"]))


class LocalPdfReadErrorPaths(unittest.TestCase):
    """``pdf_metadata`` shares ``_load_pdf_bytes``; cover its error branches."""

    def test_metadata_propagates_missing_file_error(self) -> None:
        result = pdf_reader.pdf_metadata(
            "/tmp/definitely-does-not-exist-9999.pdf",
            metadata_engine=lambda _b: {},
        )
        self.assertIn("local pdf not found", result["error"])

    def test_metadata_url_non_200_surfaces_status(self) -> None:
        class _Response:
            status_code = 503
            content = b""
            text = ""

        from contextlib import contextmanager

        @contextmanager
        def _factory():
            class _Client:
                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    return None

                def get(self, _url):
                    return _Response()

            yield _Client()

        result = pdf_reader.pdf_metadata(
            "https://example.com/x.pdf",
            metadata_engine=lambda _b: {},
            http_client_factory=_factory,
        )
        self.assertEqual(result["status_code"], 503)


class CreateToolPluginTests(unittest.TestCase):
    def test_descriptor_lists_both_tools(self) -> None:
        plugin = pdf_reader.create_tool_plugin()
        self.assertEqual(plugin["tool_server_ref"], "tool_server:pdf_reader")
        self.assertEqual(plugin["server_name"], "pdf-reader")
        self.assertEqual(plugin["tool_names"], ["pdf_read", "pdf_metadata"])


class CodexSdkMcpRoutingTests(unittest.TestCase):
    def test_pdf_reader_server_routes_to_pdf_reader_factory(self) -> None:
        import sys as _sys

        from praxist.plugins.agent_runtimes.codex_sdk._mcp import (
            MCP_STDIO_MODULE,
            mcp_configuration,
        )

        result = mcp_configuration([{"server_name": "pdf-reader"}])
        server = result.config["mcp_servers"]["pdf-reader"]
        self.assertEqual(server["command"], _sys.executable)
        self.assertEqual(server["args"][:2], ["-m", MCP_STDIO_MODULE])
        self.assertIn(
            "praxist.plugins.tools.pdf_reader.adapter:create_pdf_reader_server",
            server["args"][2],
        )
        self.assertEqual(result.warnings, ())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
