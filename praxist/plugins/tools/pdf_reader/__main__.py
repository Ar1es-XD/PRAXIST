"""Stdio MCP entrypoint for ``tool_server:pdf_reader``.

Runs the pdf-reader tool server as a standalone process so agent runtimes
can spawn it through their MCP configuration and call ``pdf_read`` /
``pdf_metadata`` inside an interactive session.

Invoked as::

    python -m praxist.plugins.tools.pdf_reader

Requires ``poppler-utils`` and ``tesseract-ocr`` at runtime — the
parent process must have them on PATH for the rasterization /
OCR fallbacks to work.
"""

from __future__ import annotations

import asyncio

from praxist.plugins.tools.pdf_reader.adapter import create_pdf_reader_server


async def _main() -> None:  # pragma: no cover - integration-tested via spawn
    """Bind the pdf-reader tool server to stdin/stdout and run until EOF."""
    from mcp.server.stdio import stdio_server  # type: ignore[import-not-found]

    config = create_pdf_reader_server()
    server = config["instance"]
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":  # pragma: no cover - integration-tested via spawn
    asyncio.run(_main())
