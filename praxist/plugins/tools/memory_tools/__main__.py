"""Stdio MCP entrypoint for ``tool_server:memory_tools``.

Runs the memory tool server as a standalone subprocess so agent runtimes can
connect to the resolved Praxist tool surface over stdio MCP.

Invoked as:

.. code-block:: bash

    python -m praxist.plugins.tools.memory_tools

Runtime adapters may use this module directly or the generic MCP stdio
launcher; both expose the same ``mcp__memory-tools__*`` surface.
"""

from __future__ import annotations

import asyncio
import os
import sys

from praxist.plugins.tools.memory_tools.adapter import create_memory_tools_server


async def _main() -> None:  # pragma: no cover - integration-tested via spawn
    """Bind the memory tool server to stdin/stdout and run until EOF.

    Unlike the other tool servers, ``create_memory_tools_server`` needs
    a ``run_dir`` argument (it reads evidence cards rooted there).
    When spawned as a subprocess we read it from the framework's
    propagated ``PRAXIST_RUN_DIR`` environment variable set by the runtime
    adapter when it launches the subprocess.
    """
    from mcp.server.stdio import stdio_server  # type: ignore[import-not-found]

    run_dir = os.environ.get("PRAXIST_RUN_DIR") or os.environ.get("RUN_DIR")
    if not run_dir:
        sys.stderr.write("memory_tools subprocess requires PRAXIST_RUN_DIR (or RUN_DIR) in env.\n")
        sys.exit(2)
    config = create_memory_tools_server(run_dir)
    server = config["instance"]
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":  # pragma: no cover - integration-tested via spawn
    asyncio.run(_main())
