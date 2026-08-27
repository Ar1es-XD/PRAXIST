"""pdf_reader adapter — rasterize + OCR research PDFs (#128 PR-3).

PyMuPDF / pdfplumber struggle with two-column layouts, math formulas,
and figures — operator-confirmed observation that drove this design.
Rasterize each page to an image and run tesseract OCR instead;
results are dramatically better on actual research papers.

Audience
--------
Consumed by peers at agent-runtime time. Two handlers, both return
JSON dicts with ``schema_version: 1`` and never raise:

* ``pdf_read(pdf, pages, dpi, lang)`` — OCR a range of pages.
* ``pdf_metadata(pdf)`` — lightweight title / author / page count,
  no OCR.

Stack
-----
* ``httpx`` → fetch remote PDFs (URLs).
* ``pypdf`` → metadata only (no OCR).
* ``pdf2image`` → wraps ``pdftoppm`` (poppler-utils) for rasterization.
* ``pytesseract`` → wraps the ``tesseract`` binary for OCR.

All four are ``[agents]`` extras and lazy-imported; the production
path is excluded from coverage. The handler exposes three injection
seams (``rasterize_engine`` / ``ocr_engine`` / ``metadata_engine``)
plus the usual ``http_client_factory`` so unit tests exercise the
plumbing without the deps installed.

System deps
-----------
* ``poppler-utils`` (provides ``pdftoppm``).
* ``tesseract-ocr`` + ``tesseract-ocr-eng``.
* Optional ``tesseract-ocr-equ`` for the ``lang="eng+equ"`` math
  fallback.

Install on macOS: ``brew install poppler tesseract``. On Ubuntu:
``apt install poppler-utils tesseract-ocr tesseract-ocr-eng``.

Caching
-------
OCR is slow (~1-3 s per page at 300 DPI). The adapter caches per-page
output under ``<state_dir>/pdf_ocr_cache/<sha256(pdf_bytes)>/p<NNN>_dpi<D>_<lang>.txt``,
keyed on the content hash + page number + DPI + language. Re-reading
the same paper is zero-OCR-cost; updating the PDF (different bytes)
falls through to a fresh OCR pass.

Page range
----------
``"1-5"``, ``"3,7-9"``, ``"all"`` are accepted. Default is ``"1-5"``
— OCR'ing a 30-page paper is 30-90 s of peer time. Operators who want
the full paper must say so.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, cast

try:  # claude_agent_sdk is only required when the MCP server is spun up.
    from claude_agent_sdk import (  # type: ignore[import-not-found]
        create_sdk_mcp_server,
        tool,
    )
except ImportError:  # pragma: no cover - SDK missing in some test envs
    tool = None
    create_sdk_mcp_server = None


_DEFAULT_PAGE_RANGE = "1-5"
_DEFAULT_DPI = 300
_MIN_DPI = 72
_MAX_DPI = 450
_DEFAULT_LANG = "eng"

_DEFAULT_TIMEOUT_SECONDS = 30.0

_MAX_PDF_BYTES = 50 * 1024 * 1024
"""Reject PDFs larger than 50 MB — peers would otherwise OCR for many minutes."""

_MAX_PAGES_PER_REQUEST = 30
"""Hard cap on the number of pages OCR'd in one call (protects peer timing budget)."""

_SCHEMA_VERSION = 1

_CACHE_SUBDIR = "pdf_ocr_cache"


__all__ = [
    "create_pdf_reader_server",
    "create_tool_plugin",
    "pdf_metadata",
    "pdf_read",
]


# --------------------------------------------------------------------------- #
# Pure handlers (callable from tests without poppler / tesseract installed)
# --------------------------------------------------------------------------- #


def pdf_read(
    pdf: str,
    pages: str = _DEFAULT_PAGE_RANGE,
    dpi: int = _DEFAULT_DPI,
    lang: str = _DEFAULT_LANG,
    *,
    http_client_factory: Any = None,
    rasterize_engine: Any = None,
    ocr_engine: Any = None,
    cache_root: Path | None = None,
    binary_probe: Any = None,
) -> dict[str, Any]:
    """OCR a range of pages from a PDF and return per-page text.

    Args:
        pdf: URL (``https://...``) or local filesystem path.
        pages: ``"1-5"`` / ``"3,7-9"`` / ``"all"``. Default ``"1-5"``;
            never silently OCRs an entire long paper.
        dpi: Rasterization DPI. Clamped to ``[72, 450]``; default 300.
        lang: tesseract language code. Use ``"eng+equ"`` for the math
            language pack when available.
        http_client_factory: Test seam (URL fetch).
        rasterize_engine: Test seam — callable
            ``(pdf_bytes, dpi, page_numbers) -> list[bytes]`` returning
            one image per requested page. Production uses pdf2image.
        ocr_engine: Test seam — callable ``(image_bytes, lang) -> str``.
            Production uses pytesseract.
        cache_root: Override the cache directory. Production reads
            ``PRAXIST_STATE_DIR`` / ``$HOME/.local/share/praxist``.
        binary_probe: Test seam — callable ``(bin_name) -> str | None``
            returning a path or None. Production uses ``shutil.which``.

    Returns:
        On success::

            {
              "schema_version": 1, "pdf": str, "num_pages": int,
              "requested_pages": [int, ...],
              "pages": [{"page_num": int, "text": str, "image_hash": str}],
              "cached_pages": [int, ...], "warnings": [str],
            }

        On error::

            {"schema_version": 1, "error": "..."}
    """
    if not isinstance(pdf, str) or not pdf.strip():
        return {"schema_version": _SCHEMA_VERSION, "error": "pdf is required"}

    probe = binary_probe or shutil.which
    if rasterize_engine is None and probe("pdftoppm") is None:
        return {
            "schema_version": _SCHEMA_VERSION,
            "error": (
                "pdftoppm (poppler-utils) not on PATH. Install with "
                "`brew install poppler` (macOS) or "
                "`apt install poppler-utils` (Ubuntu)."
            ),
        }
    if ocr_engine is None and probe("tesseract") is None:
        return {
            "schema_version": _SCHEMA_VERSION,
            "error": (
                "tesseract not on PATH. Install with `brew install tesseract` "
                "(macOS) or `apt install tesseract-ocr tesseract-ocr-eng` (Ubuntu)."
            ),
        }

    pdf_bytes_result = _load_pdf_bytes(pdf, http_client_factory=http_client_factory)
    if "error" in pdf_bytes_result:
        return pdf_bytes_result
    pdf_bytes = pdf_bytes_result["bytes"]

    content_hash = hashlib.sha256(pdf_bytes).hexdigest()

    clamped_dpi = max(_MIN_DPI, min(int(dpi), _MAX_DPI))
    safe_lang = _sanitize_lang(lang)

    num_pages = _quick_page_count(pdf_bytes)
    requested = _parse_page_range(pages, num_pages)
    if "error" in requested:
        return requested
    page_numbers: list[int] = requested["pages"]
    warnings: list[str] = list(requested.get("warnings", []))
    if len(page_numbers) > _MAX_PAGES_PER_REQUEST:
        warnings.append(
            f"page range expanded to {len(page_numbers)} pages; truncating "
            f"to first {_MAX_PAGES_PER_REQUEST} to keep peer latency bounded"
        )
        page_numbers = page_numbers[:_MAX_PAGES_PER_REQUEST]

    cache_dir = _resolve_cache_root(cache_root) / content_hash
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Split into cached + needs-OCR.
    cached_pages: list[int] = []
    miss_pages: list[int] = []
    page_text: dict[int, str] = {}
    image_hashes: dict[int, str] = {}
    for page in page_numbers:
        cache_path = cache_dir / _cache_filename(page, clamped_dpi, safe_lang)
        if cache_path.is_file():
            try:
                page_text[page] = cache_path.read_text(encoding="utf-8")
                hash_path = cache_dir / _hash_filename(page, clamped_dpi, safe_lang)
                if hash_path.is_file():
                    image_hashes[page] = hash_path.read_text(encoding="utf-8").strip()
                cached_pages.append(page)
                continue
            except OSError as exc:
                warnings.append(f"cache read failed for page {page}: {exc}")
        miss_pages.append(page)

    if miss_pages:
        rasterize = rasterize_engine or _default_rasterize_engine
        ocr = ocr_engine or _default_ocr_engine
        try:
            images = rasterize(pdf_bytes, clamped_dpi, miss_pages)
        except Exception as exc:  # pragma: no cover - production path
            return {
                "schema_version": _SCHEMA_VERSION,
                "error": f"rasterization failed: {exc}",
            }
        if len(images) != len(miss_pages):
            warnings.append(
                f"rasterizer returned {len(images)} images for "
                f"{len(miss_pages)} requested pages; truncating"
            )
        for page, image in zip(miss_pages, images, strict=False):
            image_hash = hashlib.sha256(image).hexdigest()[:16]
            try:
                text = ocr(image, safe_lang)
            except Exception as exc:  # pragma: no cover - production path
                warnings.append(f"OCR failed for page {page}: {exc}")
                continue
            page_text[page] = text
            image_hashes[page] = image_hash
            try:
                (cache_dir / _cache_filename(page, clamped_dpi, safe_lang)).write_text(
                    text, encoding="utf-8"
                )
                (cache_dir / _hash_filename(page, clamped_dpi, safe_lang)).write_text(
                    image_hash, encoding="utf-8"
                )
            except OSError as exc:
                warnings.append(f"cache write failed for page {page}: {exc}")

    page_results = [
        {
            "page_num": page,
            "text": page_text.get(page, ""),
            "image_hash": image_hashes.get(page, ""),
        }
        for page in page_numbers
        if page in page_text
    ]

    return {
        "schema_version": _SCHEMA_VERSION,
        "pdf": pdf,
        "num_pages": num_pages,
        "requested_pages": page_numbers,
        "pages": page_results,
        "cached_pages": cached_pages,
        "dpi": clamped_dpi,
        "lang": safe_lang,
        "warnings": warnings,
    }


def pdf_metadata(
    pdf: str,
    *,
    http_client_factory: Any = None,
    metadata_engine: Any = None,
) -> dict[str, Any]:
    """Return lightweight metadata (title, author, page count). No OCR.

    Production uses ``pypdf``. Tests inject ``metadata_engine``.
    """
    if not isinstance(pdf, str) or not pdf.strip():
        return {"schema_version": _SCHEMA_VERSION, "error": "pdf is required"}

    pdf_bytes_result = _load_pdf_bytes(pdf, http_client_factory=http_client_factory)
    if "error" in pdf_bytes_result:
        return pdf_bytes_result
    pdf_bytes = pdf_bytes_result["bytes"]

    engine = metadata_engine or _default_metadata_engine
    try:
        meta = engine(pdf_bytes)
    except Exception as exc:  # pragma: no cover - production path
        return {
            "schema_version": _SCHEMA_VERSION,
            "error": f"metadata extraction failed: {exc}",
        }

    out: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "pdf": pdf,
    }
    if isinstance(meta, dict):
        out.update(meta)
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _load_pdf_bytes(pdf: str, *, http_client_factory: Any = None) -> dict[str, Any]:
    """Resolve ``pdf`` (URL or path) into raw bytes. Returns ``{"bytes": ...}`` or error."""
    pdf = pdf.strip()
    if pdf.startswith("http://") or pdf.startswith("https://"):
        return _fetch_pdf_url(pdf, http_client_factory=http_client_factory)

    path = Path(pdf).expanduser()
    if not path.is_file():
        return {
            "schema_version": _SCHEMA_VERSION,
            "error": f"local pdf not found: {path}",
        }
    try:
        data = path.read_bytes()
    except OSError as exc:
        return {
            "schema_version": _SCHEMA_VERSION,
            "error": f"failed to read pdf: {exc}",
        }
    if len(data) > _MAX_PDF_BYTES:
        return {
            "schema_version": _SCHEMA_VERSION,
            "error": (
                f"pdf exceeds {_MAX_PDF_BYTES} bytes ({len(data)} bytes); "
                "fetch a smaller version or specify a page range"
            ),
        }
    return {"bytes": data}


def _fetch_pdf_url(url: str, *, http_client_factory: Any = None) -> dict[str, Any]:
    """Download ``url`` via httpx. Test seam supplies ``http_client_factory``."""
    if http_client_factory is None:  # pragma: no cover - production-only
        import httpx  # type: ignore[import-not-found]

        def _factory() -> Any:
            return httpx.Client(timeout=_DEFAULT_TIMEOUT_SECONDS)

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
            "error": f"pdf fetch failed: {exc}",
        }

    status = getattr(response, "status_code", None)
    if status != 200:
        return {
            "schema_version": _SCHEMA_VERSION,
            "error": f"pdf fetch returned status {status}",
            "status_code": status,
        }
    data = getattr(response, "content", None)
    if not isinstance(data, bytes):
        data = bytes(getattr(response, "text", "") or "", encoding="utf-8")
    if len(data) > _MAX_PDF_BYTES:
        return {
            "schema_version": _SCHEMA_VERSION,
            "error": (
                f"pdf exceeds {_MAX_PDF_BYTES} bytes ({len(data)} bytes); "
                "fetch a smaller version or specify a page range"
            ),
        }
    return {"bytes": data}


_PAGE_RANGE_TOKEN = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+))?\s*$")


def _parse_page_range(spec: str, num_pages: int) -> dict[str, Any]:
    """Parse ``"1-5"`` / ``"3,7-9"`` / ``"all"`` into a deduped sorted page list."""
    if not isinstance(spec, str) or not spec.strip():
        return {"schema_version": _SCHEMA_VERSION, "error": "pages spec is empty"}

    text = spec.strip().lower()
    if text == "all":
        return {"pages": list(range(1, max(num_pages, 1) + 1))}

    selected: set[int] = set()
    warnings: list[str] = []
    for raw_token in text.split(","):
        match = _PAGE_RANGE_TOKEN.match(raw_token)
        if not match:
            return {
                "schema_version": _SCHEMA_VERSION,
                "error": f"invalid pages token {raw_token.strip()!r}; "
                "use formats like '1-5', '3,7-9', or 'all'",
            }
        start = int(match.group(1))
        end_str = match.group(2)
        end = int(end_str) if end_str is not None else start
        if start < 1 or end < start:
            return {
                "schema_version": _SCHEMA_VERSION,
                "error": f"invalid pages range {raw_token.strip()!r}; "
                "start must be >= 1 and <= end",
            }
        if num_pages > 0 and end > num_pages:
            warnings.append(
                f"clamped page range {raw_token.strip()!r} to document length {num_pages}"
            )
            end = num_pages
        if start > num_pages > 0:
            continue
        selected.update(range(start, end + 1))

    return {"pages": sorted(selected), "warnings": warnings}


def _quick_page_count(pdf_bytes: bytes) -> int:
    """Cheap page-count probe — counts ``/Type /Page`` tokens.

    pypdf would be more accurate but it's an ``[agents]`` extra and we
    want the page-count probe to work even when pypdf isn't installed
    (so the range validator can clamp ranges before OCR ever fires).
    The regex is the canonical PDF page-object pattern.
    """
    return len(re.findall(rb"/Type\s*/Page(?!s)\b", pdf_bytes))


def _resolve_cache_root(override: Path | None) -> Path:
    """Resolve the OCR cache root: explicit override > PRAXIST_STATE_DIR > XDG."""
    if override is not None:
        return Path(override) / _CACHE_SUBDIR
    state = os.environ.get("PRAXIST_STATE_DIR")
    if state:
        return Path(state).expanduser() / _CACHE_SUBDIR
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "praxist" / _CACHE_SUBDIR


def _cache_filename(page: int, dpi: int, lang: str) -> str:
    return f"p{page:04d}_dpi{dpi}_{lang}.txt"


def _hash_filename(page: int, dpi: int, lang: str) -> str:
    return f"p{page:04d}_dpi{dpi}_{lang}.hash"


_LANG_SAFE = re.compile(r"^[a-z]{2,4}(\+[a-z]{2,4})*$")


def _sanitize_lang(lang: str) -> str:
    """Reject anything that doesn't look like a tesseract lang code list."""
    text = lang.strip().lower() if isinstance(lang, str) else ""
    if not _LANG_SAFE.match(text):
        return "eng"
    return text


def _default_rasterize_engine(
    pdf_bytes: bytes, dpi: int, page_numbers: list[int]
) -> list[bytes]:  # pragma: no cover - production-only
    """pdf2image-backed rasterization. Only runs when poppler is installed."""
    from io import BytesIO

    from pdf2image import convert_from_bytes  # type: ignore[import-not-found]

    images: list[bytes] = []
    # convert_from_bytes accepts first_page / last_page; we call once per
    # contiguous run to minimize subprocess overhead.
    for page in page_numbers:
        pil_images = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=page, last_page=page)
        if not pil_images:
            continue
        buf = BytesIO()
        pil_images[0].save(buf, format="PNG")
        images.append(buf.getvalue())
    return images


def _default_ocr_engine(image_bytes: bytes, lang: str) -> str:  # pragma: no cover - production
    """pytesseract-backed OCR. Only runs when tesseract is installed."""
    from io import BytesIO

    import pytesseract  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]

    with Image.open(BytesIO(image_bytes)) as img:
        return cast(str, pytesseract.image_to_string(img, lang=lang))


def _default_metadata_engine(pdf_bytes: bytes) -> dict[str, Any]:  # pragma: no cover - production
    """pypdf-backed metadata extraction. Only runs when pypdf is installed."""
    from io import BytesIO

    from pypdf import PdfReader  # type: ignore[import-not-found]

    reader = PdfReader(BytesIO(pdf_bytes))
    meta = reader.metadata or {}
    return {
        "title": str(meta.get("/Title") or ""),
        "author": str(meta.get("/Author") or ""),
        "subject": str(meta.get("/Subject") or ""),
        "creator": str(meta.get("/Creator") or ""),
        "num_pages": len(reader.pages),
    }


# --------------------------------------------------------------------------- #
# MCP server wiring
# --------------------------------------------------------------------------- #


def _text_result(data: Any) -> dict[str, Any]:
    """Wrap a JSON-serialisable value as an MCP text content response."""
    text = json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data
    return {"content": [{"type": "text", "text": text}]}


def create_pdf_reader_server() -> Any:  # pragma: no cover - requires claude_agent_sdk
    """Create the MCP server exposing pdf_read + pdf_metadata."""
    if create_sdk_mcp_server is None or tool is None:
        raise ImportError("claude_agent_sdk is required for MCP tools")

    async def _handle_read(args: dict[str, Any]) -> dict[str, Any]:
        return _text_result(
            pdf_read(
                str(args.get("pdf", "")),
                str(args.get("pages", _DEFAULT_PAGE_RANGE)),
                int(args.get("dpi", _DEFAULT_DPI)),
                str(args.get("lang", _DEFAULT_LANG)),
            )
        )

    async def _handle_metadata(args: dict[str, Any]) -> dict[str, Any]:
        return _text_result(pdf_metadata(str(args.get("pdf", ""))))

    read_tool = tool(
        "pdf_read",
        (
            "OCR a range of pages from a PDF (URL or local path). "
            "pages='1-5' / '3,7-9' / 'all' (defaults to first 5). "
            "dpi 72..450 (300 default). lang='eng' (default) / 'eng+equ' "
            "for math formula support. Cached per page on disk."
        ),
        {"pdf": str, "pages": str, "dpi": int, "lang": str},
    )(_handle_read)

    metadata_tool = tool(
        "pdf_metadata",
        "Return lightweight PDF metadata (title, author, num_pages). No OCR.",
        {"pdf": str},
    )(_handle_metadata)

    return create_sdk_mcp_server("pdf-reader", tools=[read_tool, metadata_tool])


def create_tool_plugin() -> dict[str, object]:
    """Manifest entrypoint exposing the pdf-reader tool server descriptor."""
    return {
        "tool_server_ref": "tool_server:pdf_reader",
        "server_name": "pdf-reader",
        "factory": "praxist.plugins.tools.pdf_reader.adapter:create_pdf_reader_server",
        "tool_names": ["pdf_read", "pdf_metadata"],
        "visibility": ["peer", "panel"],
        "required_capability": "tool_server.pdf_reader",
        "handlers": {
            "pdf_read": "praxist.plugins.tools.pdf_reader.adapter:pdf_read",
            "pdf_metadata": "praxist.plugins.tools.pdf_reader.adapter:pdf_metadata",
        },
    }
