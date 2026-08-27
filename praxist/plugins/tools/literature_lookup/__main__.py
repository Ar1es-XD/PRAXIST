"""Stdio MCP entrypoint for ``tool_server:literature_lookup``.

Runs the public literature lookup tool server as a standalone process so
external CLI agents can spawn it via MCP server config.
"""

from __future__ import annotations

import asyncio

from praxist.plugins.tools.literature_lookup.adapter import (
    create_literature_lookup_server,
)


async def _main() -> None:  # pragma: no cover - integration-tested via spawn
    """Bind the literature lookup tool server to stdin/stdout and run until EOF."""
    from mcp.server.stdio import stdio_server  # type: ignore[import-not-found]

    config = create_literature_lookup_server()
    server = config["instance"]
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":  # pragma: no cover - integration-tested via spawn
    asyncio.run(_main())
