"""Codex SDK app-server MCP configuration tests."""

from __future__ import annotations

import sys
import unittest

from praxist.plugins.agent_runtimes.codex_sdk._mcp import (
    MCP_STDIO_MODULE,
    McpConfiguration,
    mcp_configuration,
    mcp_server_key,
)


class McpConfigurationTest(unittest.TestCase):
    def test_empty_selection_has_no_servers_or_warnings(self) -> None:
        self.assertEqual(
            mcp_configuration([]),
            McpConfiguration(config={"mcp_servers": {}}, warnings=()),
        )

    def test_bundled_factory_uses_generic_stdio_launcher_and_scoped_env(self) -> None:
        result = mcp_configuration(
            [
                {
                    "server_name": "frontier-tools",
                    "tool_names": ["read_frontier", "publish_candidate"],
                }
            ],
            env={"PRAXIST_RUN_DIR": "/run", "TASK_MODE": "test"},
        )
        server = result.config["mcp_servers"]["frontier-tools"]  # type: ignore[index]

        self.assertEqual(server["command"], sys.executable)
        self.assertEqual(server["args"][0:2], ["-m", MCP_STDIO_MODULE])
        self.assertIn("frontier_tools.adapter:create_frontier_tools_server", server["args"][2])
        self.assertEqual(server["env"], {"PRAXIST_RUN_DIR": "/run", "TASK_MODE": "test"})
        self.assertEqual(server["default_tools_approval_mode"], "approve")
        self.assertTrue(server["required"])
        self.assertEqual(server["startup_timeout_sec"], 30)

    def test_ancillary_server_stays_optional(self) -> None:
        result = mcp_configuration(
            [{"server_name": "literature-lookup", "tool_names": ["literature_search"]}]
        )
        server = result.config["mcp_servers"]["literature-lookup"]  # type: ignore[index]
        self.assertFalse(server["required"])

    def test_declared_factory_and_python_override_support_task_plugins(self) -> None:
        result = mcp_configuration(
            [
                {
                    "server_name": "task-evaluator",
                    "factory": "task_tools.evaluator:create_server",
                    "tool_names": ["evaluate"],
                }
            ],
            python_executable="/task/.venv/bin/python",
        )
        server = result.config["mcp_servers"]["task-evaluator"]  # type: ignore[index]

        self.assertEqual(server["command"], "/task/.venv/bin/python")
        self.assertEqual(server["args"][-1], "task_tools.evaluator:create_server")
        self.assertEqual(result.warnings, ())

    def test_allow_list_filters_declared_tools_with_qualified_names(self) -> None:
        entries = [
            {
                "server_name": "evaluation-tools",
                "tool_names": ["evaluate", "inspect", "cancel"],
            }
        ]
        result = mcp_configuration(
            entries,
            allowed_tools=["mcp__evaluation_tools__evaluate", "inspect"],
        )
        server = result.config["mcp_servers"]["evaluation-tools"]  # type: ignore[index]

        self.assertEqual(server["enabled_tools"], ["evaluate", "inspect"])

    def test_allow_list_supplies_tool_names_for_legacy_bundled_descriptor(self) -> None:
        result = mcp_configuration(
            [{"server_name": "memory-tools"}],
            allowed_tools=[
                "mcp__memory_tools__get_evidence_card",
                "mcp__memory-tools__query_evidence_cards",
                "mcp__other__ignored",
            ],
        )
        server = result.config["mcp_servers"]["memory-tools"]  # type: ignore[index]

        self.assertEqual(
            server["enabled_tools"],
            ["get_evidence_card", "query_evidence_cards"],
        )

    def test_deny_list_wins_over_allow_list(self) -> None:
        result = mcp_configuration(
            [
                {
                    "server_name": "evaluation-tools",
                    "tool_names": ["evaluate", "inspect"],
                }
            ],
            allowed_tools=[
                "mcp__evaluation-tools__evaluate",
                "mcp__evaluation-tools__inspect",
            ],
            denied_tools=["mcp__evaluation_tools__inspect"],
        )
        server = result.config["mcp_servers"]["evaluation-tools"]  # type: ignore[index]

        self.assertEqual(server["enabled_tools"], ["evaluate"])

    def test_allow_list_omits_server_when_no_declared_tool_is_selected(self) -> None:
        result = mcp_configuration(
            [{"server_name": "memory-tools", "tool_names": ["remember"]}],
            allowed_tools=["mcp__frontier-tools__read_frontier"],
        )
        self.assertEqual(result.config, {"mcp_servers": {}})

    def test_explicit_empty_allow_list_omits_all_servers(self) -> None:
        result = mcp_configuration(
            [{"server_name": "memory-tools", "tool_names": ["remember"]}],
            allowed_tools=[],
        )
        self.assertEqual(result.config, {"mcp_servers": {}})

    def test_credentials_are_scoped_to_the_server_that_declares_them(self) -> None:
        result = mcp_configuration(
            [
                {"server_name": "brave-search", "tool_names": ["web_search"]},
                {"server_name": "arxiv", "tool_names": ["arxiv_search"]},
            ],
            env={"PRAXIST_RUN_DIR": "/run"},
            credential_env={"BRAVE_API_KEY": "brave-secret"},
        )
        servers = result.config["mcp_servers"]  # type: ignore[assignment]
        self.assertEqual(
            servers["brave-search"]["env"],  # type: ignore[index]
            {"PRAXIST_RUN_DIR": "/run", "BRAVE_API_KEY": "brave-secret"},
        )
        self.assertEqual(
            servers["arxiv"]["env"],  # type: ignore[index]
            {"PRAXIST_RUN_DIR": "/run"},
        )

    def test_deny_only_can_leave_server_with_remaining_tools(self) -> None:
        result = mcp_configuration(
            [
                {
                    "server_name": "memory-tools",
                    "tool_names": ["remember", "recall"],
                }
            ],
            denied_tools=["remember"],
        )
        server = result.config["mcp_servers"]["memory-tools"]  # type: ignore[index]
        self.assertEqual(server["enabled_tools"], ["recall"])

    def test_unknown_server_warns_and_does_not_create_stale_bridge(self) -> None:
        result = mcp_configuration([{"server_name": "unknown-tools"}])

        self.assertEqual(result.config, {"mcp_servers": {}})
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("unknown-tools", result.warnings[0])
        self.assertNotIn("bridge", result.warnings[0])

    def test_duplicate_and_malformed_entries_are_ignored(self) -> None:
        result = mcp_configuration(
            [
                None,  # type: ignore[list-item]
                {},
                {"server_name": ""},
                {"server_name": "arxiv"},
                {"server_name": "arxiv"},
            ]
        )

        self.assertEqual(list(result.config["mcp_servers"]), ["arxiv"])  # type: ignore[arg-type]

    def test_server_key_preserves_app_server_namespace(self) -> None:
        self.assertEqual(mcp_server_key("finding-graph-query"), "finding-graph-query")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
