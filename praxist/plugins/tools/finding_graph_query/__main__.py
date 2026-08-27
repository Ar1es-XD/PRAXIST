"""Stdio MCP entrypoint for ``tool_server:finding_graph_query``.

Runs the finding-graph-query tool server as a standalone subprocess so agent
runtimes can connect to the resolved Praxist tool surface over stdio MCP.

Invoked as:

.. code-block:: bash

    python -m praxist.plugins.tools.finding_graph_query

Runtime adapters may use this module directly or the generic MCP stdio
launcher; both expose the same ``mcp__finding-graph-query__*`` surface.
"""

from __future__ import annotations

import asyncio

from praxist.plugins.tools.finding_graph_query.adapter import (
    create_finding_graph_query_server,
)


async def _main() -> None:  # pragma: no cover - integration-tested via spawn
    """Bind the finding-graph-query tool server to stdin/stdout and run until EOF."""
    from mcp.server.stdio import stdio_server  # type: ignore[import-not-found]

    config = create_finding_graph_query_server()
    server = config["instance"]
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":  # pragma: no cover - integration-tested via spawn
    asyncio.run(_main())
