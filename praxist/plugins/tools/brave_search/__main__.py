"""Stdio MCP entrypoint for ``tool_server:brave_search``.

Runs the Brave Search tool server as a standalone process so agent runtimes
can connect over stdio MCP and call ``web_search``.

Invoked as::

    python -m praxist.plugins.tools.brave_search

Runtime adapters may use this module directly or the generic MCP stdio
launcher when ``brave-search`` is selected in
``AgentRunRequest.tool_servers``.

Requires ``BRAVE_API_KEY`` at runtime — the parent process is
responsible for forwarding it to the subprocess env.
"""

from __future__ import annotations

import asyncio

from praxist.plugins.tools.brave_search.adapter import (
    create_brave_search_server,
)


async def _main() -> None:  # pragma: no cover - integration-tested via spawn
    """Bind the brave-search tool server to stdin/stdout and run until EOF."""
    from mcp.server.stdio import stdio_server  # type: ignore[import-not-found]

    config = create_brave_search_server()
    server = config["instance"]
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":  # pragma: no cover - integration-tested via spawn
    asyncio.run(_main())
