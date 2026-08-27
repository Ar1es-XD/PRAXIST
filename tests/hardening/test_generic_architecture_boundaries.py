from __future__ import annotations

import re
import unittest
from pathlib import Path

from praxist.core.panel_topology import panel_topology_for_ref

TASK_SPECIFIC_PATTERN = re.compile(
    r"(?i)(\bsam\b|sam_|task:sam|role:sam|legacy_multi_pi|cifar|tiny[-_]?imagenet|resnet|asam|gsam)"
)
AIST_SPECIFIC_PATTERN = re.compile(
    r"(?i)("
    r"\bAIST\b|AIST_ROOT|trading-HDT|aist_trading|"
    r"mean_active_alpha_vs_benchmark_pct|validation_2026|max_drawdown|"
    r"mean_mdd|mean_return_pct|q25_active_alpha|"
    r"l1_utilization|l1_evidence|uses_l1_features|"
    r"concentration_label|benchmark_relative"
    r")"
)

GENERIC_BOUNDARY_ALLOWLIST: set[Path] = set()


class GenericArchitectureBoundaryTest(unittest.TestCase):
    def test_generic_core_files_do_not_embed_task_specific_keywords(self) -> None:
        candidates = [*Path("praxist/core").glob("*.py"), Path("praxist/run.py")]
        offenders: list[str] = []
        for path in sorted(candidates):
            if path in GENERIC_BOUNDARY_ALLOWLIST:
                continue
            if TASK_SPECIFIC_PATTERN.search(str(path)):
                offenders.append(f"{path}: filename contains task-specific token")
            text = path.read_text(encoding="utf-8")
            for match in TASK_SPECIFIC_PATTERN.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{path}:{line_no}: {match.group(0)}")

        self.assertEqual(offenders, [])

    def test_praxist_core_prompts_and_code_do_not_embed_aist_trading_keywords(self) -> None:
        suffixes = {".py", ".jinja2", ".yaml", ".yml"}
        offenders: list[str] = []
        for path in sorted(Path("praxist").rglob("*")):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            text = path.read_text(encoding="utf-8")
            for match in AIST_SPECIFIC_PATTERN.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{path}:{line_no}: {match.group(0)}")

        self.assertEqual(offenders, [])

    def test_panel_topology_loader_is_manifest_driven_for_non_task_topology(self) -> None:
        topology = panel_topology_for_ref("panel_topology:fake_two_round", workspace=Path.cwd())

        self.assertEqual(topology.topology_ref, "panel_topology:fake_two_round")
        self.assertEqual(topology.roles_for_mode("mini"), ["fake_pi"])
        self.assertEqual(topology.role_refs_for_mode("full"), ["role:fake_pi"])
        self.assertEqual(topology.chair_role_ref, "role:fake_chair")
        self.assertEqual(
            [round_spec.round_id for round_spec in topology.rounds],
            ["draft_memo", "chair_synthesis"],
        )

    def test_literature_scout_role_is_task_local_not_bundled(self) -> None:
        self.assertFalse(Path("praxist/plugins/roles/literature_scout").exists())
        self.assertTrue(Path("praxist/plugins/tools/literature_lookup/plugin.yaml").exists())

        for task_root in [
            Path("templates/tasks/sam_optimizer"),
            Path("templates/tasks/template"),
            Path("templates/tasks/toy_math"),
            Path("templates/tasks/machine_learning_template"),
        ]:
            self.assertTrue((task_root / "roles/literature_scout/role.yaml").exists())
            self.assertTrue((task_root / "roles/literature_scout/skill.md").exists())
            task_yaml = (task_root / "task.yaml").read_text(encoding="utf-8")
            self.assertIn("role: task_role:literature_scout", task_yaml)
            self.assertIn("tool_server_ref: tool_server:literature_lookup", task_yaml)

    def test_finding_graph_query_is_the_canonical_tool_surface_name(self) -> None:
        """Keep graph maintenance and read-only graph querying physically distinct."""
        old_underscore_name = "finding_graph" + "_tools"
        old_hyphen_name = "finding-graph" + "-tools"

        self.assertFalse((Path("praxist/plugins/tools") / old_underscore_name).exists())
        self.assertTrue(Path("praxist/plugins/tools/finding_graph_query/adapter.py").exists())
        self.assertTrue(
            Path("praxist/plugins/graph_maintainers/finding_graph_mvp/engine.py").exists()
        )

        manifest_text = Path("praxist/plugins/tools/finding_graph_query/plugin.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("name: finding_graph_query", manifest_text)
        self.assertIn("server_name: finding-graph-query", manifest_text)

        checked_roots = [
            Path("praxist"),
            Path("praxist/plugins"),
            Path("templates/tasks"),
            Path("tests"),
        ]
        suffixes = {".py", ".yaml", ".yml", ".md", ".jinja2"}
        offenders: list[str] = []
        for root in checked_roots:
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix not in suffixes:
                    continue
                text = path.read_text(encoding="utf-8")
                if old_underscore_name in text or old_hyphen_name in text:
                    offenders.append(str(path))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
