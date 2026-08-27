"""Stdio MCP entrypoint for ``tool_server:system``.

Runs the system tool server as a standalone process so any configured MCP
client can call the five read-only ``system_*`` tools.

Invoked as:

.. code-block:: bash

    python -m praxist.plugins.tools.system

The function unpacks the dict returned by :func:`create_system_tools_server`
to get the underlying :class:`mcp.server.lowlevel.server.Server` instance
and runs it over stdio using the official ``mcp`` package's stdio
transport.

"""

from __future__ import annotations

import asyncio

from praxist.plugins.tools.system.adapter import create_system_tools_server


async def _main() -> None:  # pragma: no cover - integration-tested via spawn
    """Bind the system tool server to stdin/stdout and run until EOF.

    The ``mcp`` package is imported lazily so the module itself remains
    importable when only the base Praxist dependency set is installed
    (e.g. on CI runs that do not pull in the ``[agents]`` extra).
    Running this entrypoint requires ``mcp`` at runtime, which is a
    transitive dependency of ``claude_agent_sdk``.
    """
    from mcp.server.stdio import stdio_server  # type: ignore[import-not-found]

    config = create_system_tools_server()
    server = config["instance"]
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":  # pragma: no cover - integration-tested via spawn
    asyncio.run(_main())
