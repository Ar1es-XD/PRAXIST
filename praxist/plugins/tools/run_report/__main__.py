"""Stdio MCP entrypoint for ``tool_server:run_report``."""

from __future__ import annotations

import asyncio

from praxist.plugins.tools.run_report.adapter import create_run_report_server


async def _main() -> None:  # pragma: no cover - integration-tested through MCP spawn
    from mcp.server.stdio import stdio_server  # type: ignore[import-not-found]

    config = create_run_report_server()
    server = config["instance"]
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_main())
