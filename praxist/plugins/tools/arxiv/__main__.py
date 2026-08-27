"""Stdio MCP entrypoint for ``tool_server:arxiv``.

Runs the arxiv tool server as a standalone process so agent runtimes can
spawn it through their MCP configuration and call the three ``arxiv_*``
tools inside an interactive session.

Invoked as::

    python -m praxist.plugins.tools.arxiv
"""

from __future__ import annotations

import asyncio

from praxist.plugins.tools.arxiv.adapter import create_arxiv_server


async def _main() -> None:  # pragma: no cover - integration-tested via spawn
    """Bind the arxiv tool server to stdin/stdout and run until EOF."""
    from mcp.server.stdio import stdio_server  # type: ignore[import-not-found]

    config = create_arxiv_server()
    server = config["instance"]
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":  # pragma: no cover - integration-tested via spawn
    asyncio.run(_main())
