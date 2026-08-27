from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class ToolServerCoreContractsTest(unittest.TestCase):
    def test_selection_build_allowed_names_and_handler_normalization(self) -> None:
        from praxist.core import tool_servers

        self.assertEqual(
            tool_servers.tool_server_refs_from_task_descriptor(
                {
                    "praxist_plugins": {
                        "tools": ["tool_server:a", "bad"],
                        "tool_servers": {
                            "b": {"ref": "tool_server:b"},
                            "c": {"ref": "tool_server:c", "enabled": False},
                        },
                    }
                }
            ),
            ("tool_server:a", "tool_server:b"),
        )
        self.assertEqual(
            tool_servers.tool_server_refs_from_task_descriptor(
                {"praxist_plugins": {"tool_servers": ["tool_server:list"]}}
            ),
            ("tool_server:list",),
        )
        self.assertEqual(
            tool_servers.tool_server_refs_from_task_descriptor(
                {"praxist_plugins": {"tool_servers": "bad-shape"}}
            ),
            (),
        )
        self.assertEqual(
            tool_servers.effective_research_tool_server_refs_from_task_descriptor(
                {"praxist_plugins": {"tools": []}}
            ),
            tool_servers.DEFAULT_RESEARCH_TOOL_SERVER_REFS,
        )
        self.assertIn(
            tool_servers.LITERATURE_LOOKUP_TOOL_SERVER_REF,
            tool_servers.DEFAULT_RESEARCH_TOOL_SERVER_REFS,
        )
        self.assertIn(
            tool_servers.LITERATURE_LOOKUP_TOOL_SERVER_REF,
            tool_servers.DEFAULT_PEER_TOOL_SERVER_REFS,
        )
        self.assertIn(
            tool_servers.LITERATURE_LOOKUP_TOOL_SERVER_REF,
            tool_servers.PANEL_TOOL_SERVER_REFS,
        )
        self.assertIn(
            tool_servers.RUN_REPORT_TOOL_SERVER_REF,
            tool_servers.DEFAULT_RESEARCH_TOOL_SERVER_REFS,
        )
        self.assertIn(
            tool_servers.RUN_REPORT_TOOL_SERVER_REF,
            tool_servers.PANEL_TOOL_SERVER_REFS,
        )
        self.assertNotIn(
            tool_servers.RUN_REPORT_TOOL_SERVER_REF,
            tool_servers.DEFAULT_PEER_TOOL_SERVER_REFS,
        )
        self.assertEqual(
            tool_servers.effective_research_tool_server_refs_from_task_descriptor(
                {"praxist_plugins": {"tools": ["tool_server:custom"]}}
            ),
            ("tool_server:custom",),
        )
        refs: list[str] = []
        tool_servers._append_tool_server_ref(refs, "model_provider:x")
        tool_servers._append_tool_server_ref(refs, 123)
        self.assertEqual(refs, [])
        with self.assertRaises(ValueError):
            tool_servers.tool_server_for_ref("model_provider:x")
        with (
            patch.object(tool_servers, "require_execution_plugin", return_value=None),
            self.assertRaises(ValueError),
        ):
            tool_servers.tool_server_for_ref("tool_server:missing", registry=object())

        spec = tool_servers.ToolServerSpec(
            plugin_ref="tool_server:test",
            server_name="test-tools",
            factory="module:create",
            tool_names=("ok", "panel_only"),
            visibility=("peer", "panel"),
            required_capability="tool_server.legacy_mcp",
            requires_run_dir=True,
        )
        self.assertTrue(spec.enabled_for(local_mode=True, multi_pi_enabled=True))
        self.assertFalse(
            tool_servers.ToolServerSpec(
                plugin_ref="tool_server:server-off",
                server_name="server-off",
                factory=None,
                tool_names=(),
                visibility=(),
                enabled_in_server_mode=False,
            ).enabled_for(local_mode=False, multi_pi_enabled=True)
        )
        self.assertEqual(spec.to_ref().server_name, "test-tools")
        self.assertEqual(
            tool_servers._allowed_names_for_specs(
                [spec],
                local_mode=True,
                include_panel_tools=False,
                include_peer_tools=True,
                multi_pi_enabled=True,
            ),
            ["mcp__test-tools__ok", "mcp__test-tools__panel_only"],
        )
        panel_only_spec = tool_servers.ToolServerSpec(
            plugin_ref="tool_server:panel",
            server_name="panel-tools",
            factory="module:create",
            tool_names=("panel_only",),
            visibility=("panel",),
        )
        with patch.object(
            tool_servers,
            "tool_server_for_ref",
            side_effect=lambda ref, registry=None: {
                "tool_server:test": spec,
                "tool_server:panel": panel_only_spec,
            }[ref],
        ):
            self.assertEqual(
                set(
                    tool_servers.visible_mcp_servers(
                        {"test-tools": object(), "panel-tools": object()},
                        include_peer_tools=True,
                        include_panel_tools=False,
                        tool_refs=["tool_server:test", "tool_server:panel"],
                    ).keys()
                ),
                {"test-tools"},
            )
        self.assertEqual(
            tool_servers._skip_reason(
                tool_servers.ToolServerSpec(
                    "tool_server:x",
                    "x",
                    None,
                    (),
                    (),
                    enabled_in_local_mode=False,
                ),
                local_mode=True,
                multi_pi_enabled=True,
            ),
            "disabled_in_local_mode",
        )
        self.assertEqual(
            tool_servers._skip_reason(
                tool_servers.ToolServerSpec(
                    "tool_server:x",
                    "x",
                    None,
                    (),
                    (),
                    requires_multi_pi=True,
                ),
                local_mode=False,
                multi_pi_enabled=False,
            ),
            "requires_multi_pi",
        )

        created: list[Path] = []

        def fake_factory(run_dir: Path):
            created.append(run_dir)
            return {"server": str(run_dir)}

        specs = {
            "tool_server:test": spec,
            "tool_server:manifest": tool_servers.ToolServerSpec(
                plugin_ref="tool_server:manifest",
                server_name="manifest",
                factory=None,
                tool_names=("x",),
                visibility=("peer",),
            ),
            "tool_server:skipped": tool_servers.ToolServerSpec(
                plugin_ref="tool_server:skipped",
                server_name="skipped",
                factory="module:create",
                tool_names=("x",),
                visibility=("peer",),
                enabled_by_default=False,
            ),
        }
        with (
            patch.object(
                tool_servers,
                "tool_server_for_ref",
                side_effect=lambda ref, registry=None: specs[ref],
            ),
            patch.object(tool_servers, "_load_callable", return_value=fake_factory),
        ):
            built = tool_servers.build_legacy_mcp_servers(
                specs,
                run_dir="/tmp/run",
                local_mode=True,
                multi_pi_enabled=True,
            )
        self.assertEqual(built.connected_server_names, ["test-tools"])
        self.assertEqual(created, [Path("/tmp/run")])
        self.assertEqual(built.unavailable[0]["reason"], "no_factory_configured")
        self.assertEqual(built.skipped[0]["reason"], "disabled_by_default")
        with (
            patch.object(
                tool_servers,
                "tool_server_for_ref",
                side_effect=lambda ref, registry=None: specs["tool_server:test"],
            ),
            patch.object(
                tool_servers, "_load_callable", side_effect=RuntimeError("api key unavailable")
            ),
        ):
            failed_build = tool_servers.build_legacy_mcp_servers(
                ["tool_server:test"],
                run_dir="/tmp/run",
                local_mode=True,
                multi_pi_enabled=True,
            )
        self.assertEqual(failed_build.connected_server_names, [])
        self.assertIn("RuntimeError", failed_build.unavailable[0]["reason"])

        with patch.object(
            tool_servers,
            "tool_server_for_ref",
            side_effect=lambda ref, registry=None: specs["tool_server:test"],
        ):
            self.assertIn(
                "mcp__test-tools__ok",
                tool_servers.allowed_mcp_tool_names(["tool_server:test"], local_mode=True),
            )
        with patch.object(
            tool_servers,
            "tool_server_for_ref",
            side_effect=lambda ref, registry=None: specs["tool_server:test"],
        ):
            self.assertIn(
                "mcp__test-tools__ok",
                tool_servers.base_peer_allowed_tools(["test-tools"]),
            )
        literature_spec = tool_servers.ToolServerSpec(
            plugin_ref="tool_server:literature_lookup",
            server_name="literature-lookup",
            factory=None,
            tool_names=("literature_search", "literature_source_guide"),
            visibility=("peer", "panel"),
        )
        default_spec = tool_servers.ToolServerSpec(
            plugin_ref="tool_server:evaluation_tools",
            server_name="evaluation-tools",
            factory=None,
            tool_names=("share_finding",),
            visibility=("peer",),
        )
        with patch.object(
            tool_servers,
            "tool_server_for_ref",
            side_effect=lambda ref, registry=None: {
                "tool_server:evaluation_tools": default_spec,
                "tool_server:literature_lookup": literature_spec,
            }[ref],
        ):
            allowed = tool_servers.base_peer_allowed_tools(
                ["literature-lookup"],
                tool_refs=["tool_server:literature_lookup"],
            )
        self.assertIn("mcp__literature-lookup__literature_search", allowed)
        self.assertIn("mcp__literature-lookup__literature_source_guide", allowed)

        success = tool_servers.normalize_tool_result(
            "server",
            "tool",
            {"content": [{"text": json.dumps({"ok": True, "secret": "sk-ant-secret"})}]},
        )
        self.assertTrue(success.success)
        self.assertNotIn("sk-ant-secret", json.dumps(success.output))
        failure = tool_servers.normalize_tool_result(
            "server",
            "tool",
            {"is_error": True, "content": [{"text": json.dumps({"error": "rate limit"})}]},
        )
        self.assertFalse(failure.success)
        self.assertEqual(failure.failover_reason, "rate_limited")
        self.assertEqual(
            tool_servers._extract_mcp_payload({"content": [{"text": "plain"}]}), "plain"
        )
        self.assertEqual(tool_servers._extract_mcp_payload("plain"), "plain")
        self.assertEqual(tool_servers._extract_mcp_payload({"content": []}), {"content": []})
        self.assertEqual(
            tool_servers._extract_mcp_payload({"content": ["bad"]}), {"content": ["bad"]}
        )
        self.assertEqual(
            tool_servers._extract_mcp_payload({"content": [{"text": 123}]}),
            {"content": [{"text": 123}]},
        )
        self.assertEqual(tool_servers._error_from_payload("bad"), "bad")
        self.assertIsNone(tool_servers._error_from_payload({"ok": True}))
        self.assertEqual(tool_servers._classify_error("permission denied"), "auth_error")
        self.assertEqual(tool_servers._classify_error("insufficient credits"), "quota_exhausted")
        self.assertEqual(tool_servers._classify_error("deadline exceeded"), "timeout")
        self.assertEqual(tool_servers._classify_error("not installed"), "tool_unavailable")
        self.assertEqual(tool_servers._classify_error("must be valid"), "invalid_request")
        self.assertEqual(tool_servers._classify_error("something else"), "runtime_error")

        handler_spec = tool_servers.ToolServerSpec(
            plugin_ref="tool_server:test",
            server_name="test-tools",
            factory=None,
            tool_names=("ok", "no_handler", "async", "boom"),
            visibility=("peer",),
            handlers={
                "ok": "module:ok",
                "async": "module:async_ok",
                "boom": "module:boom",
            },
        )

        async def async_handler(args):
            return {"content": [{"text": json.dumps({"async": args["x"]})}]}

        def load_callable(ref: str):
            if ref.endswith(":ok"):
                return lambda args: {"content": [{"text": json.dumps({"args": args})}]}
            if ref.endswith(":async_ok"):
                return async_handler
            if ref.endswith(":boom"):
                return lambda args: (_ for _ in ()).throw(RuntimeError("invalid api key"))
            raise AssertionError(ref)

        with (
            patch.object(tool_servers, "tool_server_for_ref", return_value=handler_spec),
            patch.object(tool_servers, "_load_callable", side_effect=load_callable),
        ):
            self.assertTrue(
                asyncio.run(
                    tool_servers.execute_legacy_tool_handler_async(
                        "tool_server:test",
                        "ok",
                        {"x": 1},
                    )
                ).success
            )
            self.assertTrue(
                tool_servers.execute_legacy_tool_handler(
                    "tool_server:test",
                    "async",
                    {"x": 2},
                ).success
            )
            self.assertFalse(
                tool_servers.execute_legacy_tool_handler(
                    "tool_server:test",
                    "missing",
                    {},
                ).success
            )
            self.assertFalse(
                tool_servers.execute_legacy_tool_handler(
                    "tool_server:test",
                    "no_handler",
                    {},
                ).success
            )
            boom = tool_servers.execute_legacy_tool_handler("tool_server:test", "boom", {})
            self.assertFalse(boom.success)
            self.assertEqual(boom.failover_reason, "auth_error")

    def test_tool_server_spec_from_registry_manifest_and_errors(self) -> None:
        import tempfile

        from praxist.core import tool_servers

        with tempfile.TemporaryDirectory() as tmp:
            plugin = Path(tmp)
            (plugin / "plugin.yaml").write_text(
                """
tool_server:
  server_name: manifest-server
  tool_names: [a]
  visibility: [peer]
  factory: manifest:create
""",
                encoding="utf-8",
            )
            selected = SimpleNamespace(path=str(plugin))
            registry = SimpleNamespace(
                require=lambda kind, name: {
                    "tool_server": {
                        "server_name": "registry-server",
                        "tool_names": ["b"],
                        "visibility": ["panel"],
                        "handlers": {"b": "module:b"},
                    }
                }
            )
            spec = tool_servers._tool_server_spec_from_registry(
                "tool_server:x",
                registry,
                selected,
            )
            self.assertEqual(spec.server_name, "registry-server")
            self.assertEqual(spec.tool_names, ("b",))
            registry_nested = SimpleNamespace(
                require=lambda kind, name: {
                    "tool_server": {
                        "server_name": "nested",
                        "tool_names": ["n"],
                        "visibility": ["peer"],
                    }
                }
            )
            nested = tool_servers._tool_server_spec_from_registry(
                "tool_server:x",
                registry_nested,
                selected,
            )
            self.assertEqual(nested.server_name, "nested")
            registry_missing = SimpleNamespace(require=lambda kind, name: {})
            (plugin / "plugin.yaml").write_text("tool_server: {}", encoding="utf-8")
            with self.assertRaises(ValueError):
                tool_servers._tool_server_spec_from_registry(
                    "tool_server:x",
                    registry_missing,
                    selected,
                )
            (plugin / "plugin.yaml").write_text("1", encoding="utf-8")
            with self.assertRaises(ValueError):
                tool_servers._read_plugin_manifest(plugin)


if __name__ == "__main__":
    unittest.main()
