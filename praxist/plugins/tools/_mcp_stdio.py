"""Minimal stdio launcher for bundled Praxist MCP server factories."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import os
import sys
from typing import Any

_reconfigure = getattr(sys.stdout, "reconfigure", None)
if _reconfigure is not None:
    _reconfigure(line_buffering=True)


def _resolve_factory(factory_ref: str) -> Any:
    module_path, separator, attribute = factory_ref.partition(":")
    if not separator or not module_path or not attribute:
        raise ValueError(f"factory must be 'module:function', got {factory_ref!r}")
    return getattr(importlib.import_module(module_path), attribute)


def _invoke_factory(factory: Any) -> Any:
    parameters = inspect.signature(factory).parameters
    if not parameters:
        return factory()
    if "run_dir" in parameters:
        run_dir = os.environ.get("PRAXIST_RUN_DIR") or os.environ.get("RUN_DIR")
        if not run_dir:
            raise RuntimeError(f"{factory.__module__}:{factory.__name__} requires PRAXIST_RUN_DIR")
        return factory(run_dir)
    raise TypeError(
        f"{factory.__module__}:{factory.__name__} takes unsupported parameters: {list(parameters)}"
    )


async def _async_main(factory_ref: str) -> None:  # pragma: no cover - stdio integration
    from mcp.server.stdio import stdio_server  # type: ignore[import-not-found]

    config = _invoke_factory(_resolve_factory(factory_ref))
    server = config["instance"]
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main(argv: list[str] | None = None) -> int:
    """Run one factory-produced MCP server until stdio closes."""

    parser = argparse.ArgumentParser(
        prog="praxist.plugins.tools._mcp_stdio",
        description="Praxist stdio MCP launcher.",
    )
    parser.add_argument(
        "factory",
        help="'module.path:factory_function' returning a server config dict.",
    )
    args = parser.parse_args(argv)
    asyncio.run(_async_main(args.factory))
    return 0


if __name__ == "__main__":  # pragma: no cover - integration via spawn
    raise SystemExit(main())
