from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from praxist.core.registry import PluginLoader, PluginRoots
from praxist.core.tool_servers import (
    DEFAULT_RESEARCH_TOOL_SERVER_REFS,
    allowed_mcp_tool_names,
    base_peer_allowed_tools,
    build_legacy_mcp_servers,
    execute_legacy_tool_handler,
    normalize_tool_result,
    tool_server_for_ref,
    tool_server_refs_from_task_descriptor,
)
from praxist.plugins.workflow_stages.research_loop.backend.generation_loop import (
    GenerationLoop,
)
from praxist.task_spec import load_task_spec


class Step13ToolServerMigrationTest(unittest.TestCase):
    def test_task_descriptor_exports_tool_server_refs(self) -> None:
        descriptor = {
            "praxist_plugins": {
                "tools": [
                    "tool_server:evaluation_tools",
                    "tool_server:frontier_tools",
                    "tool_server:finding_graph_query",
                    "tool_server:memory_tools",
                    "tool_server:prior_work_tools",
                ]
            }
        }

        self.assertEqual(
            tool_server_refs_from_task_descriptor(descriptor),
            (
                "tool_server:evaluation_tools",
                "tool_server:frontier_tools",
                "tool_server:finding_graph_query",
                "tool_server:memory_tools",
                "tool_server:prior_work_tools",
            ),
        )

    def test_tool_server_spec_is_registry_backed(self) -> None:
        loader = PluginLoader(PluginRoots.defaults(Path.cwd()))
        discovery = loader.discover()
        manifest = loader.resolve(
            ["tool_server:evaluation_tools"],
            discovery,
            enforce_bundled_execution=True,
        )
        registry = loader.load(manifest)

        spec = tool_server_for_ref("tool_server:evaluation_tools", registry)
        self.assertEqual(spec.server_name, "evaluation-tools")
        self.assertIn("share_finding", spec.tool_names)
        with self.assertRaises(KeyError):
            tool_server_for_ref("tool_server:frontier_tools", registry)

    def test_legacy_mcp_builder_uses_plugin_specs_and_mode_gates(self) -> None:
        seen: dict[str, Path] = {}

        def memory_factory(run_dir: Path):
            seen["run_dir"] = run_dir
            return "memory-server"

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with (
                patch(
                    "praxist.plugins.tools.evaluation_tools.adapter.create_evaluation_tools_server",
                    return_value="eval-server",
                ),
                patch(
                    "praxist.plugins.tools.frontier_tools.adapter.create_frontier_tools_server",
                    return_value="frontier-server",
                ),
                patch(
                    "praxist.plugins.tools.finding_graph_query.adapter.create_finding_graph_query_server",
                    return_value="graph-server",
                ),
                patch(
                    "praxist.plugins.tools.memory_tools.adapter.create_memory_tools_server",
                    side_effect=memory_factory,
                ),
                patch(
                    "praxist.plugins.tools.literature_lookup.adapter.create_literature_lookup_server",
                    return_value="literature-server",
                ),
                patch(
                    "praxist.plugins.tools.run_report.adapter.create_run_report_server",
                    return_value="run-report-server",
                ),
            ):
                result = build_legacy_mcp_servers(
                    DEFAULT_RESEARCH_TOOL_SERVER_REFS,
                    run_dir=run_dir,
                    local_mode=True,
                    multi_pi_enabled=True,
                )

        self.assertEqual(
            set(result.servers),
            {
                "evaluation-tools",
                "frontier-tools",
                "finding-graph-query",
                "memory-tools",
                "run-report",
                "literature-lookup",
            },
        )
        self.assertEqual(seen["run_dir"], run_dir)
        self.assertEqual(
            [(item["server_name"], item["reason"]) for item in result.skipped],
            [("prior-work-tools", "disabled_in_local_mode")],
        )
        self.assertEqual(result.unavailable, [])

    def test_allowed_tool_names_separate_peer_panel_and_mode_scope(self) -> None:
        peer_names = allowed_mcp_tool_names(
            DEFAULT_RESEARCH_TOOL_SERVER_REFS,
            local_mode=True,
            multi_pi_enabled=True,
        )
        self.assertIn("mcp__evaluation-tools__share_finding", peer_names)
        self.assertIn("mcp__finding-graph-query__get_finding_neighbors", peer_names)
        self.assertIn("mcp__literature-lookup__literature_search", peer_names)
        self.assertNotIn("mcp__prior-work-tools__download_snapshot", peer_names)
        self.assertNotIn("mcp__memory-tools__query_coverage_matrix", peer_names)

        panel_names = allowed_mcp_tool_names(
            DEFAULT_RESEARCH_TOOL_SERVER_REFS,
            local_mode=True,
            include_panel_tools=True,
            include_peer_tools=False,
            multi_pi_enabled=True,
        )
        self.assertIn("mcp__memory-tools__query_coverage_matrix", panel_names)
        self.assertIn("mcp__frontier-tools__get_frontier", panel_names)
        self.assertIn("mcp__literature-lookup__literature_source_guide", panel_names)
        self.assertIn("mcp__run-report__generate_run_report", panel_names)
        self.assertNotIn("mcp__run-report__generate_run_report", peer_names)

        connected_peer_tools = base_peer_allowed_tools(["evaluation-tools", "memory-tools"])
        self.assertIn("mcp__evaluation-tools__get_leaderboard", connected_peer_tools)
        self.assertNotIn("mcp__memory-tools__query_coverage_matrix", connected_peer_tools)

    def test_tool_result_normalization_redacts_and_classifies_failures(self) -> None:
        raw = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"error": "API key sk-or-v1-abcdefghijklmnop is invalid"}),
                }
            ],
            "is_error": True,
        }

        result = normalize_tool_result("evaluation-tools", "share_finding", raw)

        self.assertFalse(result.success)
        self.assertEqual(result.failover_reason, "auth_error")
        self.assertNotIn("sk-or-v1-abcdefghijklmnop", json.dumps(result.output))
        self.assertTrue(result.redaction_hits)

    def test_legacy_handler_adapter_executes_and_normalizes(self) -> None:
        with patch.dict("os.environ", {"FRONTIER_DIR": ""}, clear=False):
            frontier = execute_legacy_tool_handler(
                "tool_server:frontier_tools",
                "get_frontier",
                {"top_k": 3},
            )

        self.assertTrue(frontier.success)
        self.assertEqual(frontier.output["entries"], [])

        graph = execute_legacy_tool_handler(
            "tool_server:finding_graph_query",
            "get_finding_neighbors",
            {},
        )
        self.assertFalse(graph.success)
        self.assertEqual(graph.failover_reason, "invalid_request")
        self.assertIn("finding_id", graph.error or "")

    def test_generation_loop_uses_core_tool_server_builder(self) -> None:
        task_spec = load_task_spec("templates/tasks/toy_math/task.yaml")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_step13"
            with (
                patch(
                    "praxist.plugins.tools.evaluation_tools.adapter.create_evaluation_tools_server",
                    return_value="eval-server",
                ),
                patch(
                    "praxist.plugins.tools.frontier_tools.adapter.create_frontier_tools_server",
                    return_value="frontier-server",
                ),
                patch(
                    "praxist.plugins.tools.finding_graph_query.adapter.create_finding_graph_query_server",
                    return_value="graph-server",
                ),
                patch(
                    "praxist.plugins.tools.memory_tools.adapter.create_memory_tools_server",
                    return_value="memory-server",
                ),
                patch(
                    "praxist.plugins.tools.literature_lookup.adapter.create_literature_lookup_server",
                    return_value="literature-server",
                ),
                patch(
                    "praxist.plugins.tools.run_report.adapter.create_run_report_server",
                    return_value="run-report-server",
                ),
            ):
                loop = GenerationLoop(
                    task_spec=task_spec,
                    workspace=Path.cwd(),
                    run_dir=run_dir,
                    local_mode=True,
                    tool_server_refs=DEFAULT_RESEARCH_TOOL_SERVER_REFS,
                )

        self.assertEqual(
            set(loop.mcp_servers),
            {
                "evaluation-tools",
                "frontier-tools",
                "finding-graph-query",
                "run-report",
                "literature-lookup",
            },
        )
        self.assertIn("mcp__evaluation-tools__share_finding", loop._peer_allowed_tools)
        self.assertIn("mcp__literature-lookup__literature_search", loop._peer_allowed_tools)
        self.assertNotIn("mcp__memory-tools__query_coverage_matrix", loop._peer_allowed_tools)
        self.assertNotIn("mcp__prior-work-tools__download_snapshot", loop._peer_allowed_tools)
        self.assertNotIn("run-report", loop._peer_mcp_servers)
        self.assertIn("literature-lookup", loop._peer_mcp_servers)

    def test_panel_only_tool_server_is_not_visible_to_peers(self) -> None:
        task_spec = load_task_spec("templates/tasks/toy_math/task.yaml")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_panel_only"
            with patch(
                "praxist.plugins.tools.run_report.adapter.create_run_report_server",
                return_value="run-report-server",
            ):
                loop = GenerationLoop(
                    task_spec=task_spec,
                    workspace=Path.cwd(),
                    run_dir=run_dir,
                    local_mode=True,
                    tool_server_refs=("tool_server:run_report",),
                )

        self.assertEqual(set(loop.mcp_servers), {"run-report"})
        self.assertEqual(loop._peer_mcp_servers, {})
        self.assertNotIn("mcp__run-report__generate_run_report", loop._peer_allowed_tools)
