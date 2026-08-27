"""Stdio MCP entrypoint for ``tool_server:browser``.

Runs the browser tool server as a standalone process so agent runtimes can
connect over stdio MCP and call ``web_read``.

Invoked as::

    python -m praxist.plugins.tools.browser

Runtime adapters may use this module directly or the generic MCP stdio
launcher when ``browser`` is selected in ``AgentRunRequest.tool_servers``.
"""

from __future__ import annotations

import asyncio

from praxist.plugins.tools.browser.adapter import create_browser_server


async def _main() -> None:  # pragma: no cover - integration-tested via spawn
    """Bind the browser tool server to stdin/stdout and run until EOF."""
    from mcp.server.stdio import stdio_server  # type: ignore[import-not-found]

    config = create_browser_server()
    server = config["instance"]
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":  # pragma: no cover - integration-tested via spawn
    asyncio.run(_main())
