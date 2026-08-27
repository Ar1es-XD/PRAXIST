"""Generic stdio MCP factory launcher tests."""

from __future__ import annotations

import asyncio
import os
import sys
import types
import unittest
from typing import Any
from unittest.mock import patch

from praxist.plugins.tools import _mcp_stdio


class FactoryResolutionTest(unittest.TestCase):
    def test_resolve_factory_imports_declared_callable(self) -> None:
        module = types.ModuleType("test_mcp_factory_module")

        def create_server() -> dict[str, str]:
            return {"server": "ok"}

        module.create_server = create_server  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {module.__name__: module}):
            resolved = _mcp_stdio._resolve_factory("test_mcp_factory_module:create_server")
        self.assertIs(resolved, create_server)

    def test_invalid_factory_ref_is_rejected(self) -> None:
        for value in ("", "module", ":factory", "module:"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _mcp_stdio._resolve_factory(value)

    def test_invoke_zero_argument_factory(self) -> None:
        expected = {"instance": object()}

        def factory() -> dict[str, object]:
            return expected

        self.assertIs(_mcp_stdio._invoke_factory(factory), expected)

    def test_run_dir_factory_uses_scoped_environment(self) -> None:
        calls: list[str] = []

        def factory(run_dir: str) -> dict[str, str]:
            calls.append(run_dir)
            return {"run_dir": run_dir}

        with patch.dict(os.environ, {"PRAXIST_RUN_DIR": "/run/one"}, clear=True):
            result = _mcp_stdio._invoke_factory(factory)

        self.assertEqual(result, {"run_dir": "/run/one"})
        self.assertEqual(calls, ["/run/one"])

    def test_missing_run_dir_and_unknown_parameters_fail_explicitly(self) -> None:
        def needs_run_dir(run_dir: str) -> None:
            del run_dir

        def unsupported(config: str) -> None:
            del config

        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(RuntimeError, "PRAXIST_RUN_DIR"),
        ):
            _mcp_stdio._invoke_factory(needs_run_dir)
        with self.assertRaisesRegex(TypeError, "unsupported parameters"):
            _mcp_stdio._invoke_factory(unsupported)


class _Server:
    def __init__(self) -> None:
        self.run_args: tuple[Any, ...] | None = None

    def create_initialization_options(self) -> dict[str, bool]:
        return {"initialized": True}

    async def run(self, *args: Any) -> None:
        self.run_args = args


class _StdioContext:
    async def __aenter__(self) -> tuple[str, str]:
        return "read-stream", "write-stream"

    async def __aexit__(self, *_args: Any) -> None:
        return None


class AsyncLauncherTest(unittest.TestCase):
    def test_async_main_runs_factory_server_over_stdio(self) -> None:
        server = _Server()
        stdio_module = types.ModuleType("mcp.server.stdio")
        stdio_module.stdio_server = lambda: _StdioContext()  # type: ignore[attr-defined]
        server_module = types.ModuleType("mcp.server")
        mcp_module = types.ModuleType("mcp")

        with (
            patch.dict(
                sys.modules,
                {
                    "mcp": mcp_module,
                    "mcp.server": server_module,
                    "mcp.server.stdio": stdio_module,
                },
            ),
            patch.object(_mcp_stdio, "_resolve_factory", return_value=lambda: {"instance": server}),
        ):
            asyncio.run(_mcp_stdio._async_main("task.module:create_server"))

        self.assertEqual(
            server.run_args,
            ("read-stream", "write-stream", {"initialized": True}),
        )

    def test_main_forwards_factory_ref_to_async_entrypoint(self) -> None:
        with patch.object(_mcp_stdio.asyncio, "run") as run:
            self.assertEqual(_mcp_stdio.main(["task.module:create_server"]), 0)

        coroutine = run.call_args.args[0]
        self.assertEqual(coroutine.cr_frame.f_locals["factory_ref"], "task.module:create_server")
        coroutine.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
