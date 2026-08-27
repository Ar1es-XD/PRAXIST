"""Lock the developer-side guardrail scripts (backfill issue #40).

These scripts live under the tracked ``scripts/dev/`` directory and can be
called directly by contributors or automation.

The tests pin three things so dev and CI never disagree:

1. The script files exist, have a shebang, and are marked executable.
2. ``run_guardrails.py`` runs the same guardrail command set that CI runs.
3. ``leakage_audit.sh`` is a bash script that scans ``praxist/`` and
   ``tests/``, supports ``# noqa: leakage_audit``, and exits non-zero
   when a blacklisted token leaks.
4. AGENTS.md contains the test commands so a new contributor can discover
   them without grepping the codebase.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARDRAILS = REPO_ROOT / "scripts/dev/run_guardrails.py"
LEAKAGE = REPO_ROOT / "scripts/dev/leakage_audit.sh"
INSTALL_CODEX_SKILLS = REPO_ROOT / "scripts/install_codex_skills.sh"
UNINSTALL_CODEX_SKILLS = REPO_ROOT / "scripts/uninstall_codex_skills.sh"
DIAGNOSTIC_INVENTORY = REPO_ROOT / "skills/praxist-diagnostic/scripts/run_diagnostic_inventory.py"
TERMINAL_LINE_PLOT = REPO_ROOT / "skills/terminal-line-plot/scripts/plot_series.py"
AGENTS_MD = REPO_ROOT / "AGENTS.md"


def _is_executable(p: Path) -> bool:
    return bool(p.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))


class GuardrailsScriptShape(unittest.TestCase):
    def test_file_exists(self) -> None:
        self.assertTrue(GUARDRAILS.exists(), f"{GUARDRAILS.relative_to(REPO_ROOT)} missing")

    def test_starts_with_python_shebang(self) -> None:
        first = GUARDRAILS.read_text(encoding="utf-8").splitlines()[0]
        self.assertRegex(first, r"^#!/usr/bin/env python3?\b")

    def test_marked_executable(self) -> None:
        self.assertTrue(
            _is_executable(GUARDRAILS),
            f"{GUARDRAILS.relative_to(REPO_ROOT)} needs +x",
        )


class GuardrailsScriptCommandSet(unittest.TestCase):
    def setUp(self) -> None:
        self.body = GUARDRAILS.read_text(encoding="utf-8")

    def test_runs_ruff_check(self) -> None:
        self.assertRegex(self.body, r'"ruff",\s*"check"')

    def test_runs_ruff_format_check(self) -> None:
        self.assertRegex(self.body, r'"ruff",\s*"format",\s*"--check"')

    def test_runs_pyrefly(self) -> None:
        self.assertRegex(self.body, r'"pyrefly",\s*"check"')

    def test_runs_pytest_tests_unit(self) -> None:
        self.assertRegex(self.body, r'"pytest",\s*"tests/unit"')

    def test_runs_coverage_profiles(self) -> None:
        self.assertRegex(self.body, r'"scripts/run_test_coverage\.py",\s*"unit"')
        self.assertRegex(self.body, r'"--fail-under-statements",\s*"95"')
        self.assertRegex(self.body, r'"scripts/run_test_coverage\.py",\s*"integration"')

    def test_targets_praxist_not_auto_research(self) -> None:
        """Hook silently passed for months because it was checking the
        renamed `auto_research/` path. Pin that we are on the new name."""
        self.assertRegex(self.body, r'"praxist/"')
        self.assertNotRegex(self.body, r'"auto_research/"')


class LeakageAuditScriptShape(unittest.TestCase):
    def test_file_exists(self) -> None:
        self.assertTrue(LEAKAGE.exists(), f"{LEAKAGE.relative_to(REPO_ROOT)} missing")

    def test_starts_with_bash_shebang(self) -> None:
        first = LEAKAGE.read_text(encoding="utf-8").splitlines()[0]
        self.assertRegex(first, r"^#!/usr/bin/env bash\b")

    def test_marked_executable(self) -> None:
        self.assertTrue(
            _is_executable(LEAKAGE),
            f"{LEAKAGE.relative_to(REPO_ROOT)} needs +x",
        )

    def test_uses_strict_mode(self) -> None:
        self.assertIn("set -euo pipefail", LEAKAGE.read_text(encoding="utf-8"))


class LeakageAuditBehavior(unittest.TestCase):
    """Drive the script with a synthetic blacklist + temp source tree to
    verify it actually flags violations and respects the noqa exemption."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="leakage_audit_test_")
        self.addCleanup(__import__("shutil").rmtree, self.tmp, ignore_errors=True)
        (Path(self.tmp) / "praxist").mkdir()
        (Path(self.tmp) / "tests").mkdir()

    # Token names rotated so test fixtures cannot themselves trip the real
    # audit; word-bounded for the regex match inside the script.
    _BAD = "zzzleakaudit_sentinel"

    def _run(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(LEAKAGE)],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "LEAKAGE_AUDIT_ROOT": self.tmp,
                "LEAKAGE_TOKENS": rf"\b{self._BAD}\b",
            },
        )

    def test_clean_tree_exits_zero(self) -> None:
        (Path(self.tmp) / "praxist" / "ok.py").write_text("x = 1\n")
        r = self._run()
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)

    def test_violating_token_exits_nonzero(self) -> None:
        (Path(self.tmp) / "praxist" / "bad.py").write_text(f"{self._BAD} = 1\n")
        r = self._run()
        self.assertNotEqual(r.returncode, 0, msg=r.stdout + r.stderr)

    def test_noqa_marker_silences_violation(self) -> None:
        (Path(self.tmp) / "praxist" / "documented.py").write_text(
            f"{self._BAD} = 1  # noqa: leakage_audit\n"
        )
        r = self._run()
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)


class CodexSkillInstallAndUninstallScripts(unittest.TestCase):
    def _repo_skill_names(self) -> list[str]:
        return sorted(path.parent.name for path in (REPO_ROOT / "skills").glob("*/SKILL.md"))

    def test_bundled_skills_have_required_shape(self) -> None:
        skill_names = self._repo_skill_names()
        self.assertIn("praxist-control", skill_names)
        self.assertIn("praxist-takeover-codex", skill_names)
        self.assertIn("terminal-line-plot", skill_names)
        for skill_name in skill_names:
            skill_dir = REPO_ROOT / "skills" / skill_name
            skill_md = skill_dir / "SKILL.md"
            agent_yaml = skill_dir / "agents" / "openai.yaml"
            self.assertTrue(skill_md.exists(), f"{skill_md.relative_to(REPO_ROOT)} missing")
            self.assertTrue(agent_yaml.exists(), f"{agent_yaml.relative_to(REPO_ROOT)} missing")
            text = skill_md.read_text(encoding="utf-8")
            self.assertIn(f"name: {skill_name}", text)
            self.assertNotIn("[TODO", text)
            self.assertNotIn("Structuring This Skill", text)

    def test_control_skill_documents_irregular_resume_boundaries(self) -> None:
        skill_text = (REPO_ROOT / "skills" / "praxist-control" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        operator_text = (REPO_ROOT / "docs" / "guides" / "operators.md").read_text(encoding="utf-8")

        for required in (
            "Resume Breakpoint Classification",
            "pending_boundary_generation",
            "Completed generation with interrupted PI panel",
            "Committed Gems reset with an incomplete next generation",
            "Incomplete Gems reset transaction",
            "Prepare the run with Codex before calling `praxist resume`",
            "internally recoverable",
            "clean enough for handoff",
            "call `praxist resume`",
            "crop only with operator approval",
            "Codex crop",
        ):
            self.assertIn(required, skill_text)
        for required in (
            "finished cohort whose PI/Chair boundary did not finish",
            "committed Gems reset followed by a partial next generation",
            "operator agent should prepare the run directory before",
            "do not hand the partial state directly to `praxist resume`",
        ):
            self.assertIn(required, operator_text)
        self.assertNotIn("default repair path is still Praxist-owned", operator_text)

    def test_diagnostic_skill_documents_read_only_health_checks(self) -> None:
        skill_text = (REPO_ROOT / "skills" / "praxist-diagnostic" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        operator_text = (REPO_ROOT / "docs" / "guides" / "operators.md").read_text(encoding="utf-8")
        agent_yaml = (
            REPO_ROOT / "skills" / "praxist-diagnostic" / "agents" / "openai.yaml"
        ).read_text(encoding="utf-8")

        for required in (
            "analysis-only",
            "Do not stop, resume, crop, rerender, restart, repair, or mutate a run.",
            "Human-Readable Run Reports",
            "A. Strongest variants / Pareto front",
            "B. Strong-variant evolution / lineage",
            "C. Run health",
            "tool_server:run_report",
            "docs/praxist_reports",
            "DIG / QD / PI / Gems",
            "Artifact Information Consistency",
            "Diversity HHI",
            "Low Performance Protocol",
            "Agent Behavior Report",
            "Completeness Checklist",
            "must_fix_now",
            "task harness",
            "run_diagnostic_inventory.py",
            "Use the task's own metric definitions",
            "Produce a detailed agent behavior analysis report",
            "Performance Ceiling Detection",
            "Generated Reports",
            "ceiling_detected",
            "plateau_onset_generation",
            "Artifact source rule",
            "canonical state",
            "audit snapshots",
        ):
            self.assertIn(required, skill_text)
        for required in (
            "praxist-diagnostic",
            "diversity HHI",
            "analysis-only",
            "not edit task code",
            "chronological agent behavior analysis report",
            "performance ceiling",
            "A/B/C run reports",
            "strongest variants",
            "canonical state",
            "audit snapshots",
        ):
            self.assertIn(required, operator_text)
        self.assertIn("Praxist Diagnostic", agent_yaml)
        self.assertIn("diversity HHI", agent_yaml)
        self.assertIn("performance ceiling", agent_yaml)

    def test_diagnostic_inventory_script_summarizes_synthetic_run(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist_diag_inventory_") as tmp:
            task = Path(tmp) / "task"
            run = task / "experiments" / "run_2026-01-01_test"
            gen0 = run / "gen_0"
            gen0.mkdir(parents=True)
            (gen0 / "generation_results.json").write_text(
                json.dumps(
                    [
                        {
                            "peer_id": "gen0_peer0",
                            "sessions": 2,
                            "runtime_usage": {
                                "input_tokens": 1000,
                                "total_input_tokens": 1000,
                                "cached_input_tokens": 800,
                                "cache_read_input_tokens": 800,
                                "cache_creation_input_tokens": 100,
                                "output_tokens": 50,
                                "total_tokens": 1050,
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (gen0 / "generation_boundary.json").write_text(
                json.dumps(
                    {
                        "artifact_semantics": {
                            "role": "canonical_state",
                            "status": "committed",
                            "stage": "generation_boundary",
                            "runtime_fact_source": True,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (gen0 / "dig_cohort_allocation.yaml").write_text("peers: []\n", encoding="utf-8")
            (gen0 / "gen0_peer0_prompt_layout.json").write_text(
                json.dumps(
                    {
                        "canonical_labels": {
                            "canonical_mechanism_family": "optimizer",
                            "canonical_intervention_surface": "training_loop",
                            "canonical_intent": "explore",
                            "canonical_semantic_family": "sam_family",
                            "canonical_parent_lineage": "baseline",
                            "canonical_novelty_axis": "objective",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run / "orchestrator_status.json").write_text(
                json.dumps(
                    {
                        "current_generation": 1,
                        "generations_completed": 1,
                        "findings_total": 2,
                        "exit_condition": "in_progress",
                    }
                ),
                encoding="utf-8",
            )
            result_dir = run / "results" / "variant_a"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "variant_a",
                        "final_status": "stop_after_T1",
                        "tier_reached": "T1",
                        "current_aggregate": {
                            "mean_test_taskscore": 1.2,
                            "future_fitness": 3.4,
                        },
                        "all_eval_cells": [
                            {"task": "case_a", "score": 1.0, "loss": 0.2},
                            {"task": "case_b", "score": 2.0, "loss": 0.4},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            for variant_name, score in (("variant_zero", 0.0), ("variant_negative", -1.0)):
                variant_dir = run / "results" / variant_name
                variant_dir.mkdir(parents=True)
                (variant_dir / "tiered_eval_summary.json").write_text(
                    json.dumps(
                        {
                            "variant_name": variant_name,
                            "current_aggregate": {"future_fitness": score},
                            "all_eval_cells": [],
                        }
                    ),
                    encoding="utf-8",
                )
            variant_min = run / "results" / "variant_minimize"
            variant_min.mkdir(parents=True)
            (variant_min / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "variant_minimize",
                        "primary_metric": "score",
                        "metric_direction": "minimize",
                        "current_aggregate": {"score": -2.0},
                        "all_eval_cells": [],
                    }
                ),
                encoding="utf-8",
            )
            variant_loss_max = run / "results" / "variant_loss_maximize"
            variant_loss_max.mkdir(parents=True)
            (variant_loss_max / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "variant_loss_maximize",
                        "primary_metric": "loss",
                        "metric_direction": "maximize",
                        "current_aggregate": {"loss": 5.0},
                        "all_eval_cells": [],
                    }
                ),
                encoding="utf-8",
            )
            findings = run / "shared_findings"
            findings.mkdir()
            (findings / "f1.json").write_text(
                json.dumps({"title": "finding", "variant_name": "variant_a"}),
                encoding="utf-8",
            )
            frontier = run / "frontier"
            frontier.mkdir()
            (frontier / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "primary_metric": "future_fitness",
                        "artifact_semantics": {
                            "role": "canonical_state",
                            "status": "committed",
                            "stage": "frontier_manifest",
                            "runtime_fact_source": True,
                        },
                        "lane_frontiers": {
                            "incubator": [
                                {
                                    "variant_name": "variant_a",
                                    "metric_name": "future_fitness",
                                    "metric_value": 3.4,
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            gems = run / "gems"
            gems.mkdir()
            (gems / "gems_state.json").write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "artifact_semantics": {
                            "role": "canonical_state",
                            "status": "committed",
                            "stage": "gems_state",
                            "runtime_fact_source": True,
                        },
                        "gems": [{"variant_name": "variant_a"}],
                    }
                ),
                encoding="utf-8",
            )
            scheduler = run / "resource_scheduler"
            scheduler.mkdir()
            (scheduler / "status.json").write_text(
                json.dumps(
                    {
                        "resource_supply": {
                            "idle_waiters": 1,
                            "leases": [],
                            "stats": {"granted": 1, "consumed": 1},
                        }
                    }
                )
            )
            (scheduler / "events.jsonl").write_text(
                json.dumps({"event": "supply_granted", "lease_id": "lease-a"})
                + "\n"
                + json.dumps({"event": "supply_consumed", "lease_id": "lease-a"})
                + "\n"
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(DIAGNOSTIC_INVENTORY),
                    "--task-path",
                    str(task),
                    "--run-dir",
                    str(run),
                    "--no-hardware",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["orchestrator_status"]["current_generation"], 1)
            self.assertEqual(payload["results"]["summary_count"], 5)
            self.assertEqual(
                payload["results"]["top_by_ranking_metric"][0]["ranking_metric_value"], 5.0
            )
            self.assertEqual(
                payload["results"]["top_by_ranking_metric"][0]["ranking_metric_name"],
                "loss",
            )
            self.assertEqual(
                payload["results"]["top_by_ranking_metric"][0]["ranking_metric_direction"],
                "maximize",
            )
            self.assertEqual(
                [item["variant_name"] for item in payload["results"]["top_by_ranking_metric"]],
                [
                    "variant_loss_maximize",
                    "variant_a",
                    "variant_minimize",
                    "variant_zero",
                    "variant_negative",
                ],
            )
            self.assertEqual(
                payload["results"]["top_by_ranking_metric"][2]["ranking_metric_direction"],
                "minimize",
            )
            self.assertEqual(payload["results"]["top_by_ranking_metric"][3]["future_fitness"], 0.0)
            self.assertEqual(payload["frontier"]["lanes"]["incubator"]["count"], 1)
            self.assertEqual(payload["frontier"]["artifact_semantics"]["role"], "canonical_state")
            self.assertEqual(payload["gems"]["gems_count"], 1)
            self.assertEqual(payload["resource_scheduler"]["resource_supply"]["idle_waiters"], 1)
            self.assertEqual(
                [row["event"] for row in payload["resource_scheduler_events"]],
                ["supply_granted", "supply_consumed"],
            )
            self.assertEqual(payload["gems"]["artifact_semantics"]["stage"], "gems_state")
            self.assertEqual(payload["generations"][0]["diversity_hhi"]["intent"]["hhi"], 1.0)
            self.assertEqual(
                payload["generations"][0]["generation_boundary_semantics"]["stage"],
                "generation_boundary",
            )
            self.assertEqual(payload["runtime_usage"]["total"]["sessions"], 2)
            self.assertEqual(
                payload["runtime_usage"]["total"]["uncached_input_tokens"],
                100.0,
            )
            self.assertEqual(
                payload["runtime_usage"]["total"]["cache_creation_input_tokens"],
                100.0,
            )
            self.assertFalse(payload["runtime_usage"]["total"]["telemetry_inconsistent"])
            self.assertEqual(payload["runtime_usage"]["total"]["cache_hit_ratio"], 0.8)
            self.assertEqual(
                payload["runtime_usage"]["by_generation"][0]["sessions_per_peer"],
                2.0,
            )

            generation_results = run / "gen_0" / "generation_results.json"
            generation_results.write_text(
                json.dumps([{"peer_id": "gen0_peer0", "sessions": 1}]),
                encoding="utf-8",
            )
            second = subprocess.run(
                [
                    sys.executable,
                    str(DIAGNOSTIC_INVENTORY),
                    "--task-path",
                    str(task),
                    "--run-dir",
                    str(run),
                    "--no-hardware",
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            missing_usage = json.loads(second.stdout)["runtime_usage"]["total"]
            self.assertIsNone(missing_usage["input_tokens"])
            self.assertIsNone(missing_usage["cached_input_tokens"])
            self.assertIsNone(missing_usage["uncached_input_tokens"])
            self.assertIsNone(missing_usage["cache_hit_ratio"])

    def test_diagnostic_inventory_preserves_inconsistent_legacy_usage(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist_diag_legacy_usage_") as tmp:
            task = Path(tmp) / "task"
            run = task / "experiments" / "run_legacy_usage"
            gen0 = run / "gen_0"
            gen0.mkdir(parents=True)
            (gen0 / "generation_results.json").write_text(
                json.dumps(
                    [
                        {
                            "peer_id": "gen0_peer0",
                            "sessions": 1,
                            "runtime_usage": {
                                "input_tokens": 5,
                                "cached_input_tokens": 10,
                                "output_tokens": 1,
                                "total_tokens": 6,
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(DIAGNOSTIC_INVENTORY),
                    "--task-path",
                    str(task),
                    "--run-dir",
                    str(run),
                    "--no-hardware",
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            usage = json.loads(completed.stdout)["runtime_usage"]["total"]

            self.assertEqual(usage["input_tokens"], 5.0)
            self.assertEqual(usage["cached_input_tokens"], 10.0)
            self.assertIsNone(usage["uncached_input_tokens"])
            self.assertTrue(usage["telemetry_inconsistent"])
            self.assertIn("cached_input_exceeds_total_input", usage["telemetry_issues"])
            self.assertIsNone(usage["cache_hit_ratio"])

    def test_diagnostic_inventory_uses_canonical_recursive_result_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist_diag_identity_") as tmp:
            task = Path(tmp) / "task"
            run = task / "experiments" / "run_identity"
            results = run / "results"
            nested = results / "family" / "protocol"
            nested.mkdir(parents=True)
            (nested / "final_summary.json").write_text(
                json.dumps({"current_aggregate": {"score": 1.0}}),
                encoding="utf-8",
            )
            (results / "summary.json").write_text(
                json.dumps(
                    {
                        "variant_id": "root_owned_id",
                        "current_aggregate": {"score": 2.0},
                    }
                ),
                encoding="utf-8",
            )
            (results / "custom_named_arm_eval_summary.json").write_text(
                json.dumps({"current_aggregate": {"score": 3.0}}),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(DIAGNOSTIC_INVENTORY),
                    "--task-path",
                    str(task),
                    "--run-dir",
                    str(run),
                    "--no-hardware",
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["results"]["summary_count"], 3)
            self.assertEqual(
                {item["variant_name"] for item in payload["results"]["top_by_ranking_metric"]},
                {"family/protocol", "root_owned_id", "named_arm"},
            )

    def test_control_skill_documents_status_and_empty_request_guard(self) -> None:
        skill_text = (REPO_ROOT / "skills" / "praxist-control" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        operator_text = (REPO_ROOT / "docs" / "guides" / "operators.md").read_text(encoding="utf-8")
        agent_yaml = (
            REPO_ROOT / "skills" / "praxist-control" / "agents" / "openai.yaml"
        ).read_text(encoding="utf-8")
        architecture_text = (REPO_ROOT / "docs" / "concepts" / "architecture.md").read_text(
            encoding="utf-8"
        )
        task_projects = (REPO_ROOT / "docs" / "guides" / "task-projects.md").read_text(
            encoding="utf-8"
        )

        for required in (
            "Empty Request Guard",
            "If the user invokes this skill without naming an operation",
            "Status Workflow",
            "current generation number",
            "incubator/leaderboard performance",
            "Report hardware load",
            "Draw at most two standard score curves",
            "terminal-line-plot",
            "generated report paths",
            "docs/praxist_reports",
            "Status is read-only",
            "Do not stop, resume, crop, rerender, or edit files during a",
            "Artifact source rule",
            "artifact_semantics",
            "bounded local candidate scan",
            "Ask the user to confirm",
            "exclude `.git`, `.venv`, `node_modules`, `experiments`",
            "If multiple candidates are found",
            "Warn, but do not refuse solely for this reason",
            "loader can fall back to inline task text",
            "task-owned stage labels",
            "Praxist does not assign global",
            "meaning to common tier names",
            "praxist --monitor --run-id <run_id>",
            "Ctrl-C",
        ):
            self.assertIn(required, skill_text)
        for required in (
            "If invoked without an operation",
            "generation progress",
            "incubator or leaderboard",
            "CPU/memory/process/accelerator load",
            "score curves",
            "terminal-line-plot",
            "It must not stop, resume",
            "during a status request",
            "canonical state",
            "audit snapshots",
            "agent must know the exact task project before launching",
            "confirm the exact path",
            "It should not infer a task from a broad filesystem scan",
            "praxist --monitor --run-id <run-id>",
        ):
            self.assertIn(required, operator_text)
        self.assertIn("report run status", agent_yaml)
        self.assertIn("hardware load", agent_yaml)
        self.assertIn("score curves", agent_yaml)
        self.assertIn("confirm the exact task path before start", agent_yaml)
        self.assertIn("Task Projects", architecture_text)
        self.assertIn("timestamped", task_projects)
        self.assertIn("`$RUN_DIR`", task_projects)

    def test_user_facing_monitor_docs_have_no_tmux_dependency(self) -> None:
        files = [REPO_ROOT / "README.md"]
        for root_name in ("docs", "skills", "templates"):
            root = REPO_ROOT / root_name
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix.lower() in {".md", ".yaml", ".yml", ".sh", ".jinja2", ".txt"}
            )
        offenders = [
            str(path.relative_to(REPO_ROOT))
            for path in files
            if "tmux" in path.read_text(encoding="utf-8", errors="ignore").lower()
        ]
        self.assertEqual(offenders, [])

    def test_task_initialization_skill_documents_fixed_profile_and_gems_default(self) -> None:
        skill_text = (REPO_ROOT / "skills" / "praxist-task-initialization" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        agent_yaml = (
            REPO_ROOT / "skills" / "praxist-task-initialization" / "agents" / "openai.yaml"
        ).read_text(encoding="utf-8")

        for required in (
            "skip sizing and use default parameters",
            "cohort_size: 8",
            "max_generations: 20",
            "per_generation_hours: 2.0",
            "Gems reset**: default OFF",
            "gems.enabled: false",
            "plateau-onset generation id",
            "Artifact Ownership Instructions For Generated Tasks",
            "derived views or audit",
            "Human-readable run reports",
            "tool_server:run_report",
            "first credible above-baseline frontier signal",
            "incubator",
            "candidate_library",
            "admit_new_high",
            "lower-admission long-term variant library",
            "lower-standard",
            "parent-authorized, protocol-passed",
            "non-suspect candidates",
            "Pareto-front/new-high points",
            "incubator_axis",
            "incubator_candidate_reason",
            "source-routing contract",
            "shared task-owned source label",
            "every configured `parent_eligible: true` lane",
            "outside confirmed top-k",
            "must agree",
            "domain ranking convention",
            "lower confidence bounds",
            "protocol-integrity flag",
            "Do not impose a universal epoch, step, rollout, or wall-clock threshold",
            "baseline_bench_<timestamp>",
            "user_selected_zero_placeholder",
            "Measure baseline now",
            "Do not start baseline measurement without the user's selection",
            "bounded concurrency",
            "Staged evaluation alignment",
            "**Preliminary check**: the cheapest executable sanity check",
            "**Aligned evaluation**: the only early protocol that can rank",
            "same evaluator",
            "primary metric direction",
            "aggregation semantics",
            "Aligned evidence should be near-complete in",
            "Save compute primarily by reducing",
            "coverage ratio",
            "reduced training/optimization budget",
            "the complete protocol spans many evaluation units",
            "normally preliminary/partial, not aligned",
            "correlation",
            "Accelerator Handoff Contract",
            "PRAXIST_ASSIGNED_GPU_UUIDS",
            "authoritative ordered physical GPU assignment",
            "conflicting existing mask fails loudly",
            "evaluator -> trainer -> worker propagation",
            "non-zero-UUID parent/child CUDA",
            "This is a binding test, not CPU-vs-accelerator",
            "supply_signal_enabled: true",
            "natural independent units",
            "at most N directed one-experiment leases",
            "Do not count work",
            "highly utilized without crossing pressure",
            "every 100-200 ms",
            "mature_assessment_min_completion_probability: 0.25",
            "planned_dimensions",
            "design_dimensions",
            "planned and realized distributions separately",
            "source_result_path` plus its SHA-256",
            "unknown direction is display-only",
        ):
            self.assertIn(required, skill_text)
        self.assertIn("Gems reset off by default", agent_yaml)
        self.assertIn("explicit parent-eligible durable incubator/Pareto lanes", agent_yaml)
        self.assertIn("near-complete-coverage aligned evaluation", agent_yaml)
        self.assertIn("baseline performance found/measured/explicitly marked", agent_yaml)
        self.assertIn("applicable accelerator handoff validation", agent_yaml)
        self.assertIn("DEEPSEEK_API_KEY is available", agent_yaml)
        self.assertIn("available provider fallback", agent_yaml)

    def test_machine_learning_template_documents_aligned_scout_contract(self) -> None:
        template_dir = REPO_ROOT / "templates" / "tasks" / "machine_learning_template"
        combined = "\n".join(
            [
                (template_dir / "README.md").read_text(encoding="utf-8"),
                (template_dir / "task.yaml").read_text(encoding="utf-8"),
                (template_dir / "prompt_task.jinja2").read_text(encoding="utf-8"),
                (template_dir / "prompt_base.jinja2").read_text(encoding="utf-8"),
                (template_dir / "prompt_generation.jinja2").read_text(encoding="utf-8"),
                (template_dir / "assets" / "resource_plan.md").read_text(encoding="utf-8"),
                (template_dir / "evaluations" / "primary" / "evaluation.yaml").read_text(
                    encoding="utf-8"
                ),
                (template_dir / "audit_rules" / "scope_and_evidence" / "audit.yaml").read_text(
                    encoding="utf-8"
                ),
            ]
        )

        for required in (
            "preliminary",
            "aligned",
            "complete",
            "complete evaluator path",
            "metric direction",
            "aggregation",
            "invalid-result rules",
            "leakage checks",
            "same or near-complete data/evaluation coverage",
            "reduced training or optimization budget",
            "coverage ratio",
            "near_full_data_coverage_required",
            "user-owned protocol intent",
            "early score",
            "parent-promotion gate",
            "variance",
            "lower-bound",
            "protocol_integrity_passed",
            "incubator_axis",
            "incubator_candidate_reason",
            "lower-admission durable",
            "Do not apply a universal epoch, step",
            "PRAXIST_ASSIGNED_GPU_UUIDS",
            "authoritative `PRAXIST_ASSIGNED_GPU_UUIDS`",
            "non-zero UUID",
            "CPU-vs-accelerator timing",
        ):
            self.assertIn(required, combined)

    def test_launch_docs_require_generic_evaluator_canary_and_stable_reports(self) -> None:
        task_init = (REPO_ROOT / "skills/praxist-task-initialization/SKILL.md").read_text(
            encoding="utf-8"
        )
        takeover = (REPO_ROOT / "skills/praxist-takeover/SKILL.md").read_text(encoding="utf-8")
        codex_takeover = (REPO_ROOT / "skills/praxist-takeover-codex/SKILL.md").read_text(
            encoding="utf-8"
        )
        task_projects = (REPO_ROOT / "docs/guides/task-projects.md").read_text(encoding="utf-8")
        run_reports = (REPO_ROOT / "docs/guides/user-facing-reports-and-init.md").read_text(
            encoding="utf-8"
        )
        scheduler = (REPO_ROOT / "docs/guides/central-resource-scheduler.md").read_text(
            encoding="utf-8"
        )
        template_docs = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                REPO_ROOT / "templates/tasks/README.md",
                REPO_ROOT / "templates/tasks/template/README.md",
                REPO_ROOT / "templates/tasks/machine_learning_template/README.md",
            )
        )

        for text in (task_init, takeover, codex_takeover, task_projects, template_docs):
            self.assertIn("one-unit canary", text)
        for required in (
            "task-appropriate build, load, or startup check",
            "actual public invocation contract",
            "ordinary peer-authored",
            "modified or unattested",
            "does not imply a universal seed, epoch, iteration",
        ):
            self.assertIn(required, task_init)
        self.assertIn("current task-init evaluator fan-out preflight", takeover)
        self.assertIn("current task-init evaluator fan-out preflight", codex_takeover)
        normalized_task_projects = " ".join(task_projects.split())
        normalized_run_reports = " ".join(run_reports.split())
        self.assertIn("excluded from task identity", normalized_run_reports)
        self.assertIn("RoleSkills before the research loop", normalized_task_projects)
        self.assertIn("agenda-assigned", normalized_task_projects)
        self.assertIn("only zombie or exited members is terminal", scheduler)

        task_init_skill = (
            REPO_ROOT / "skills" / "praxist-task-initialization" / "SKILL.md"
        ).read_text(encoding="utf-8")
        self.assertIn("require_truthy_metrics: [scored_complete]", task_init_skill)
        self.assertIn("protocol_integrity_failed", task_init_skill)
        self.assertNotIn(
            "require_truthy_metrics: [scored_complete, protocol_integrity_passed]",
            task_init_skill,
        )

    def test_new_user_facing_skills_are_documented(self) -> None:
        takeover = (REPO_ROOT / "skills" / "praxist-takeover" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        interactive = (
            REPO_ROOT / "skills" / "praxist-interactive-task-init" / "SKILL.md"
        ).read_text(encoding="utf-8")
        skills_readme = (REPO_ROOT / "skills" / "README.md").read_text(encoding="utf-8")
        onboarding = (REPO_ROOT / "skills" / "praxist-onboarding" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        diagnostic = (REPO_ROOT / "skills" / "praxist-diagnostic" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        guide = (REPO_ROOT / "docs" / "guides" / "user-facing-reports-and-init.md").read_text(
            encoding="utf-8"
        )

        for required in (
            "praxist-onboarding",
            "praxist-task-initialization",
            "praxist-control",
            "Run the complete onboarding flow",
            "Do not replace onboarding with a shorter local checklist",
            "confirm the exact task path",
            "lower-admission durable",
            "admit_new_high: true",
            "complete current",
            "task-local lane-routing regression",
            "unreachable parent lane",
            "do not launch until it passes",
            "warn, but do not refuse solely for this reason",
            "runtime can still launch legacy tasks",
            "praxist start --task-path",
            "--daemonize --json",
            "praxist --monitor --run-id <run_id>",
        ):
            self.assertIn(required, takeover)

        for required in (
            "confirmation first",
            "at most **5 confirmation rounds**",
            "Metrics, ranking, and robustness",
            "lower-admission durable incubator",
            "admit_new_high: true",
            "universal epoch, step, rollout, or wall-clock threshold",
            "variance",
            "lower confidence bounds",
            "naturally independent",
            "directed idle resource-supply feedback",
        ):
            self.assertIn(required, interactive)

        self.assertIn("praxist-takeover", skills_readme)
        self.assertIn("praxist-interactive-task-init", skills_readme)
        self.assertIn("docs/praxist_reports", skills_readme)
        self.assertIn("docs/praxist_reports", guide)
        self.assertIn("Canonical truth remains", guide)
        self.assertNotIn("praxist-interactive-task-init", guide)
        self.assertIn("praxist --monitor --run-id <run_id>", skills_readme)
        self.assertIn("praxist-takeover", onboarding)
        self.assertIn("praxist --monitor --run-id", onboarding)
        self.assertIn("praxist-interactive-task-init", onboarding)
        self.assertIn("task_dir=<selected_task_dir>", diagnostic)

    def test_codex_native_takeover_is_isolated_and_documented(self) -> None:
        takeover = (REPO_ROOT / "skills" / "praxist-takeover" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        subscription = (REPO_ROOT / "skills" / "praxist-takeover-codex" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        metadata = (
            REPO_ROOT / "skills" / "praxist-takeover-codex" / "agents" / "openai.yaml"
        ).read_text(encoding="utf-8")
        subscription_text = " ".join(subscription.split())
        onboarding = (REPO_ROOT / "skills" / "praxist-onboarding" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        task_init = (REPO_ROOT / "skills" / "praxist-task-initialization" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        documented = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                REPO_ROOT / "README.md",
                REPO_ROOT / "skills" / "README.md",
                REPO_ROOT / "docs" / "getting-started" / "first-task.md",
            )
        )

        for required in (
            "Load the currently installed `praxist-takeover`",
            "agent_runtime:codex_sdk",
            "model_provider:openai_compatible",
            "gpt-5.6-luna",
            "praxist doctor --codex-native",
            "codex_model_catalog: ok",
            "env -u OPENAI_API_KEY",
            "praxist resolve /absolute/path/to/task",
            "min_rejected_alternatives: 2",
            "disabled for this launch",
            "do not ask for a redundant",
            "Never retry with an API key or another provider",
        ):
            self.assertIn(required, subscription_text)
        self.assertIn("$praxist-takeover-codex", metadata)
        self.assertIn("praxist-takeover-codex", onboarding)
        self.assertIn("praxist-takeover-codex", task_init)
        self.assertIn("`gpt-5.6-luna` unless the user explicitly selected", task_init)
        self.assertIn("`dig_lite.contract.min_rejected_alternatives: 2`", task_init)
        self.assertIn("$praxist-takeover-codex", documented)

        # The specialized skill must not change the standard takeover contract.
        self.assertIn("confirm the exact task path", takeover)
        self.assertNotIn("env -u OPENAI_API_KEY", takeover)
        self.assertNotIn("API-key use: disabled for this launch", takeover)
        self.assertNotIn("gpt-5.6-luna", takeover)
        template_task = (REPO_ROOT / "templates" / "tasks" / "template" / "task.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("min_rejected_alternatives: 3", template_task)

    def test_task_templates_document_continuous_evolution_default(self) -> None:
        for task_name in ("template", "toy_math", "machine_learning_template", "sam_optimizer"):
            task_yaml = (REPO_ROOT / "templates" / "tasks" / task_name / "task.yaml").read_text(
                encoding="utf-8"
            )
            self.assertIn("gems:\n  enabled: false", task_yaml)
            if task_name == "machine_learning_template":
                self.assertIn("mode: legacy", task_yaml)
                self.assertIn("must not invent an accelerator profile", task_yaml)
                self.assertNotIn("supply_signal_enabled:", task_yaml)
            else:
                self.assertIn("supply_signal_enabled: true", task_yaml)
                self.assertIn("supply_idle_samples: 3", task_yaml)
                self.assertIn(
                    "mature_assessment_min_completion_probability: 0.25",
                    task_yaml,
                )
            self.assertIn("name: incubator", task_yaml)
            self.assertIn("Lower-admission", task_yaml)

        docs = "\n".join(
            [
                (REPO_ROOT / "templates" / "README.md").read_text(encoding="utf-8"),
                (REPO_ROOT / "templates" / "tasks" / "README.md").read_text(encoding="utf-8"),
                (REPO_ROOT / "docs" / "guides" / "task-projects.md").read_text(encoding="utf-8"),
                (REPO_ROOT / "docs" / "reference" / "task-templates.md").read_text(
                    encoding="utf-8"
                ),
                (REPO_ROOT / "docs" / "guides" / "operators.md").read_text(encoding="utf-8"),
            ]
        )
        self.assertIn("continuous evolution", docs)
        self.assertIn("gems.enabled: false", docs)
        self.assertIn("performance ceiling", docs)
        self.assertIn("<run_dir>/logs/launcher.nohup.log", docs)
        self.assertNotIn("<run_dir>/launcher.nohup.log", docs)
        self.assertNotIn(
            "sam_optimizer` uses an\n8-generation research profile, explicitly enables", docs
        )

    def test_task_init_and_templates_gate_normal_close_on_mature_evidence(self) -> None:
        task_names = ("template", "toy_math", "machine_learning_template", "sam_optimizer")
        for task_name in task_names:
            task_path = REPO_ROOT / "templates" / "tasks" / task_name / "task.yaml"
            task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
            maturity = task["evaluation"]["maturity_policy"]
            quorum = task["synthesis_trigger"]["mature_quorum_fraction"]

            self.assertTrue(maturity["require_ratio_gate"], task_name)
            self.assertGreater(quorum, 0.0, task_name)
            self.assertLessEqual(quorum, 1.0, task_name)
            self.assertGreaterEqual(task["generation_policy"]["cohort_size"] * quorum, 0.25)

        skill_text = {
            name: " ".join(
                (REPO_ROOT / "skills" / name / "SKILL.md").read_text(encoding="utf-8").split()
            )
            for name in (
                "praxist-task-initialization",
                "praxist-interactive-task-init",
                "praxist-takeover",
                "praxist-takeover-codex",
                "praxist-control",
                "praxist-diagnostic",
                "praxist-onboarding",
            )
        }
        self.assertIn(
            "Never set the quorum to `0.0` merely to avoid a deadlock",
            skill_text["praxist-task-initialization"],
        )
        self.assertIn(
            "Run a lightweight closing-policy lifecycle regression",
            skill_text["praxist-task-initialization"],
        )
        self.assertIn(
            "`0.0` disables the mature normal-completion gate",
            skill_text["praxist-takeover"],
        )
        self.assertIn(
            "positive when the task distinguishes mature/complete evidence",
            skill_text["praxist-interactive-task-init"],
        )
        self.assertIn(
            "positive mature normal-close gate",
            skill_text["praxist-takeover-codex"],
        )
        self.assertIn(
            "Do not silently start a task",
            skill_text["praxist-control"],
        )
        self.assertIn(
            "raw information-density findings can close the generation",
            skill_text["praxist-diagnostic"],
        )
        self.assertIn(
            "scheduler mature-supply targets prioritize work and do not create this gate",
            skill_text["praxist-onboarding"],
        )

        docs = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                REPO_ROOT / "templates" / "README.md",
                REPO_ROOT / "templates" / "tasks" / "README.md",
                REPO_ROOT / "docs" / "guides" / "research-loop-flexibility-controls.md",
                REPO_ROOT / "docs" / "guides" / "task-projects.md",
                REPO_ROOT / "docs" / "guides" / "central-resource-scheduler.md",
            )
        )
        self.assertIn("mature supply target is advisory", docs)
        self.assertIn("cannot replace this gate", docs)
        self.assertNotIn("(`0.0` by default)", docs)

    def test_templates_use_generation_scoped_dig_and_independent_qd(self) -> None:
        for task_name in ("template", "toy_math", "machine_learning_template", "sam_optimizer"):
            task_path = REPO_ROOT / "templates" / "tasks" / task_name / "task.yaml"
            task = yaml.safe_load(task_path.read_text(encoding="utf-8"))

            self.assertEqual(task["dig_lite"]["generation_scope"], "initial_only")
            self.assertNotIn("cohort_qd", task["dig_lite"])
            self.assertTrue(task["quality_diversity"]["initial_generation_enabled"])
            if task_name in {"template", "toy_math"}:
                self.assertFalse(task["quality_diversity"]["later_generations_enabled"])
                self.assertFalse(task["evaluation"]["constructive_peer_mix_enabled"])
            else:
                self.assertTrue(task["quality_diversity"]["later_generations_enabled"])
                self.assertTrue(task["evaluation"]["constructive_peer_mix_enabled"])

        task_init = (REPO_ROOT / "skills" / "praxist-task-initialization" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("generation_scope: initial_only", task_init)
        self.assertIn("initial_generation_enabled: true", task_init)
        self.assertIn("later_generations_enabled: true", task_init)
        self.assertIn("enforce_forward_slots: true", task_init)
        self.assertIn("existing PI synthesis path", task_init)

    def test_task_init_documents_reachable_close_and_task_runtime_boundaries(self) -> None:
        task_init = (REPO_ROOT / "skills" / "praxist-task-initialization" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        takeover = (REPO_ROOT / "skills" / "praxist-takeover" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        task_projects = (REPO_ROOT / "docs" / "guides" / "task-projects.md").read_text(
            encoding="utf-8"
        )

        for text in (task_init, takeover, task_projects):
            self.assertIn("estimated_close_grade_eval_minutes * safety_factor", text)
            self.assertIn("estimated_heavy_eval_minutes", text)
            self.assertIn("p90", text)
            self.assertIn("task root", text)
            self.assertIn("run-like", text)
        self.assertIn("runner-owned `PYTHONPATH`/`PYTHONHOME`", takeover)
        self.assertIn("do not inherit the\nPraxist runner's `PYTHONPATH`", task_projects)

        for task_name in ("template", "toy_math", "machine_learning_template", "sam_optimizer"):
            task_text = (REPO_ROOT / "templates" / "tasks" / task_name / "task.yaml").read_text(
                encoding="utf-8"
            )
            self.assertIn("observed p90", task_text.lower())
            self.assertIn("estimate*safety", task_text)
            self.assertIn("estimated_close_grade_eval_minutes", task_text)

    def test_takeover_accelerator_coherence_preserves_nonaccelerated_tasks(self) -> None:
        takeover = " ".join(
            (REPO_ROOT / "skills" / "praxist-takeover" / "SKILL.md")
            .read_text(encoding="utf-8")
            .split()
        )

        for contract in (
            "task-runtime accelerator coherence check",
            "host accelerator inventory as available/unavailable/unknown",
            "selected task interpreter's backend capability",
            "public evaluator canary",
            "scheduler default resource profile",
            "does not prove that the host lacks acceleration",
            "current user or task explicitly requires an accelerator",
            "report any extra host capability as advisory and continue unchanged",
            "Never install or upgrade task dependencies, force an accelerator",
        ):
            self.assertIn(contract, takeover)

    def test_task_harness_guidance_normalizes_effective_config_and_explicit_retry(self) -> None:
        task_init = (REPO_ROOT / "skills/praxist-task-initialization/SKILL.md").read_text(
            encoding="utf-8"
        )
        takeover = (REPO_ROOT / "skills/praxist-takeover/SKILL.md").read_text(encoding="utf-8")
        task_projects = (REPO_ROOT / "docs/guides/task-projects.md").read_text(encoding="utf-8")
        scheduler = (REPO_ROOT / "docs/guides/central-resource-scheduler.md").read_text(
            encoding="utf-8"
        )
        template_docs = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (REPO_ROOT / "templates/tasks").rglob("*.md")
        )

        for text in (task_init, takeover, task_projects):
            self.assertIn("omitted", text)
            self.assertIn("resolved", text)
            self.assertIn("default", text)
        for text in (task_init, takeover, scheduler, template_docs):
            self.assertIn("--retry-terminal", text)
        self.assertNotIn("AIST_VARIANT_DD_INCREMENT_PENALTY", "\n".join((task_init, template_docs)))

    def test_generic_incubator_and_optional_axes_docs_do_not_misroute_signals(self) -> None:
        task_init_skill = (
            REPO_ROOT / "skills" / "praxist-task-initialization" / "SKILL.md"
        ).read_text(encoding="utf-8")
        onboarding = (REPO_ROOT / "skills" / "praxist-onboarding" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        flow_guide = (
            REPO_ROOT / "docs" / "guides" / "research-loop-variant-generation-flow.md"
        ).read_text(encoding="utf-8")
        task_projects = (REPO_ROOT / "docs" / "guides" / "task-projects.md").read_text(
            encoding="utf-8"
        )
        flexibility = (
            REPO_ROOT / "docs" / "guides" / "research-loop-flexibility-controls.md"
        ).read_text(encoding="utf-8")
        ml_task = (
            REPO_ROOT / "templates" / "tasks" / "machine_learning_template" / "task.yaml"
        ).read_text(encoding="utf-8")

        self.assertIn("Modes that the task marks non-parentable", onboarding)
        self.assertIn("protocol-suspect signals belong in validation", onboarding)
        self.assertIn("Immature scout or partial output\nremains a validation signal", flow_guide)
        self.assertIn("not incubator content", flow_guide)

        combined_optional_examples = "\n".join(
            [task_init_skill, task_projects, flexibility, ml_task]
        )
        for misplaced in (
            "<robust_or_lower_bound_metric>",
            "<risk_or_constraint_metric>",
            "<cost_or_efficiency_metric>",
            "robust_score_lcb",
            "risk_or_constraint_metric",
            "cost_or_efficiency_metric",
            "task_score_lower_confidence_bound",
        ):
            self.assertNotIn(misplaced, combined_optional_examples)
        for misplaced in (
            "task_score_lower_confidence_bound",
            "seed_robustness_std",
            "constraint_violation_rate",
            "inference_cost",
        ):
            self.assertNotIn(misplaced, ml_task)
        self.assertIn("secondary_tiebreak_metric", combined_optional_examples)
        self.assertIn("diagnostic_display_metric", combined_optional_examples)
        self.assertIn("validation_only_result", combined_optional_examples)
        self.assertIn("late_after_generation_boundary", combined_optional_examples)
        incubator_block = ml_task.split("name: incubator", 1)[1].split("name: task_candidate", 1)[0]
        self.assertNotIn("scout_aligned", incubator_block)
        self.assertNotIn("mini_T1", incubator_block)
        self.assertIn("validation_only_result", incubator_block)
        self.assertIn("late_after_generation_boundary", incubator_block)
        self.assertIn(
            "include_lanes: [task_candidate, candidate, preliminary, aligned, partial, repair]",
            ml_task,
        )

    def test_task_init_templates_and_docs_require_reachable_lane_producers(self) -> None:
        task_init = (REPO_ROOT / "skills" / "praxist-task-initialization" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        takeover = (REPO_ROOT / "skills" / "praxist-takeover" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        docs_and_templates = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                REPO_ROOT / "docs" / "guides" / "task-projects.md",
                REPO_ROOT / "docs" / "guides" / "user-facing-reports-and-init.md",
                REPO_ROOT / "templates" / "README.md",
                REPO_ROOT / "templates" / "tasks" / "README.md",
                REPO_ROOT
                / "templates"
                / "tasks"
                / "machine_learning_template"
                / "prompt_task.jinja2",
            )
        )

        for required in (
            "source-routing contract",
            "shared task-owned source label",
            "normally `performance`",
            "every configured `parent_eligible: true` lane",
            "more than `confirmed.k` parent-authorized fixtures",
            "single-metric task",
            "must agree",
        ):
            self.assertIn(required, task_init)
        for required in (
            "complete current",
            "`praxist-task-initialization` workflow",
            "current `SKILL.md` at execution time",
            "older remembered checklist",
            "task-local lane-routing regression",
            "do not launch until it passes",
            "unreachable configured",
            "parent-eligible lane",
            "regardless of whether the task is new or",
        ):
            self.assertIn(required, takeover)
        for required in (
            "source label",
            "performance",
            "confirmed",
            "incubator",
            "parent-eligible",
        ):
            self.assertIn(required, docs_and_templates)
        self.assertNotIn('"frontier_lane":"confirmed"', docs_and_templates)

    def test_launch_skills_keep_deepseek_and_advisory_controls_non_blocking(self) -> None:
        task_init = (REPO_ROOT / "skills" / "praxist-task-initialization" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        takeover = (REPO_ROOT / "skills" / "praxist-takeover" / "SKILL.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("when the host has `DEEPSEEK_API_KEY` available", task_init)
        self.assertIn("If `DEEPSEEK_API_KEY` is not configured, do not block", task_init)
        self.assertIn("when `DEEPSEEK_API_KEY` is available", task_init)
        self.assertIn("available-provider fallback", task_init)
        self.assertIn("warn and continue", takeover)
        self.assertIn("does not", takeover)
        self.assertIn("require ratio-gated maturity", takeover)
        self.assertIn("Repair initialization only when the declared", takeover)

    def test_task_init_protocol_policy_follows_explicit_user_intent(self) -> None:
        task_init = (REPO_ROOT / "skills" / "praxist-task-initialization" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        takeover = (REPO_ROOT / "skills" / "praxist-takeover" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        takeover_codex = (REPO_ROOT / "skills" / "praxist-takeover-codex" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        interactive = (
            REPO_ROOT / "skills" / "praxist-interactive-task-init" / "SKILL.md"
        ).read_text(encoding="utf-8")
        onboarding = (REPO_ROOT / "skills" / "praxist-onboarding" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        control = (REPO_ROOT / "skills" / "praxist-control" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        diagnostic = (REPO_ROOT / "skills" / "praxist-diagnostic" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        peer_prompt = (
            REPO_ROOT
            / "praxist"
            / "plugins"
            / "workflow_stages"
            / "research_loop"
            / "backend"
            / "prompt_base.jinja2"
        ).read_text(encoding="utf-8")
        generic_templates = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                REPO_ROOT / "templates" / "tasks" / "template" / "task.yaml",
                REPO_ROOT / "templates" / "tasks" / "machine_learning_template" / "task.yaml",
                REPO_ROOT
                / "templates"
                / "tasks"
                / "machine_learning_template"
                / "prompt_base.jinja2",
                REPO_ROOT
                / "templates"
                / "tasks"
                / "machine_learning_template"
                / "prompt_task.jinja2",
                REPO_ROOT
                / "templates"
                / "tasks"
                / "machine_learning_template"
                / "audit_rules"
                / "scope_and_evidence"
                / "audit.yaml",
            )
        )
        documentation = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                REPO_ROOT / "docs" / "guides" / "task-projects.md",
                REPO_ROOT / "docs" / "guides" / "user-facing-reports-and-init.md",
                REPO_ROOT / "templates" / "README.md",
                REPO_ROOT / "templates" / "tasks" / "README.md",
                REPO_ROOT / "templates" / "tasks" / "template" / "README.md",
                REPO_ROOT / "templates" / "tasks" / "machine_learning_template" / "README.md",
            )
        )

        self.assertIn("the user's explicit current instruction", task_init)
        self.assertIn("no Praxist-wide full-protocol-only rule", task_init)
        self.assertIn("Only **undeclared drift**", task_init)
        self.assertIn("not command keywords", task_init)
        self.assertIn("task-declared mature labels", task_init)
        self.assertIn("Set `require_ratio_gate: true` only", task_init)
        self.assertIn("obey it without adding another confirmation round", takeover)
        self.assertIn("Do not impose a full-protocol-only rule", takeover)
        self.assertIn("preserve it instead of adding a", takeover)
        self.assertIn("preserve `0.0`", takeover)
        self.assertIn("must not strengthen task semantics", takeover_codex)
        self.assertIn("The user's answer is authoritative", interactive)
        self.assertIn("do not re-litigate an explicit user choice", interactive)
        self.assertIn("explicitly authorized reduced mode", onboarding)
        self.assertIn("not from words such as", control)
        self.assertIn("user-approved task maturity contract", diagnostic)
        self.assertIn("deliberately reduced protocol", peer_prompt)
        self.assertIn("task-authorized, protocol-passed", generic_templates)
        self.assertIn("user-owned protocol intent explicitly authorizes", generic_templates)
        self.assertIn("complete relative to its", generic_templates)
        self.assertIn("not Praxist-wide restrictions", documentation)
        self.assertNotIn("clean complete source label", takeover)
        self.assertNotIn("more than `confirmed.k` complete fixtures", task_init)

    def test_sam_template_protocol_text_matches_tiered_evaluator(self) -> None:
        sam_dir = REPO_ROOT / "templates" / "tasks" / "sam_optimizer"
        combined_prompt_text = "\n".join(
            [
                (sam_dir / "task.yaml").read_text(encoding="utf-8"),
                (sam_dir / "description.md").read_text(encoding="utf-8"),
                (sam_dir / "prompt_task.jinja2").read_text(encoding="utf-8"),
                (sam_dir / "README.md").read_text(encoding="utf-8"),
                (
                    sam_dir
                    / "roles"
                    / "external_validity_pi"
                    / "private_kb"
                    / "cross_arch_protocols.md"
                ).read_text(encoding="utf-8"),
            ]
        )
        for stale in (
            "200 epochs",
            "batch size 128",
            "mid-fidelity ranking",
            "mid-fidelity rank",
            "ResNet-18 + CIFAR-100",
        ):
            self.assertNotIn(stale, combined_prompt_text)
        for required in (
            "batch size 256",
            "T3 is the full mature benchmark",
            "Tiny-ImageNet",
            "20 epochs",
            "not a near-full aligned-scout ranking protocol",
            "near-full ranking evidence",
        ):
            self.assertIn(required, combined_prompt_text)

        evaluator_contract = (
            sam_dir / "evaluations" / "pareto_tiered" / "evaluation.yaml"
        ).read_text(encoding="utf-8") + (
            sam_dir / "evaluations" / "pareto_tiered" / "evaluator.py"
        ).read_text(encoding="utf-8")
        for metric in (
            "mean_test_accuracy",
            "test_accuracy_cifar100",
            "test_accuracy_cifar10",
            "test_accuracy_tiny_imagenet",
            "compute_overhead_ratio",
            "wall_time_seconds_total",
        ):
            self.assertIn(metric, evaluator_contract)
        for stale_metric in ("miou", "dice", "latency_ms"):
            self.assertNotIn(stale_metric, evaluator_contract)

    def test_sam_task_evaluator_promotes_only_t3_complete_results(self) -> None:
        from templates.tasks.sam_optimizer.evaluations.pareto_tiered.evaluator import (
            create_evaluation,
        )

        evaluator = create_evaluation()
        t2_candidate = {
            "id": "t2",
            "metrics": {
                "tier": "T2",
                "promotion_eligible": False,
                "mean_test_accuracy": 0.9,
            },
        }
        t3_without_flag = {
            "id": "t3_missing_flag",
            "metrics": {"tier": "T3", "mean_test_accuracy": 0.8},
        }
        t3_promotable = {
            "id": "t3",
            "metrics": {
                "tier": "T3",
                "promotion_eligible": True,
                "mean_test_accuracy": 0.7,
            },
        }

        self.assertFalse(evaluator.eligible_for_promotion(t2_candidate))
        self.assertFalse(evaluator.eligible_for_promotion(t3_without_flag))
        self.assertTrue(evaluator.eligible_for_promotion(t3_promotable))
        self.assertEqual(
            evaluator.rank([t2_candidate, t3_without_flag, t3_promotable]), [t3_promotable]
        )

    def test_sam_tiered_runner_maturity_ratios_use_full_reference_protocol(self) -> None:
        from templates.tasks.sam_optimizer.evaluations.pareto_tiered import run

        metrics_summary = {
            "tier": "T1",
            "promotion_eligible": False,
            "scored_cell_count": 3,
        }
        run.attach_maturity_ratios(metrics_summary, tier="T1")

        self.assertLess(metrics_summary["effort_ratio"], 1.0)
        self.assertLess(metrics_summary["coverage_ratio"], 1.0)
        self.assertFalse(metrics_summary["scored_complete"])
        self.assertEqual(metrics_summary["frontier_lane"], "task_candidate")
        self.assertEqual(metrics_summary["promotion_lane"], "task_candidate")
        self.assertEqual(metrics_summary["reference_effort_units"], 3)
        self.assertEqual(metrics_summary["total_required_eval_units"], 15)

        complete_summary = {
            "tier": "T3",
            "promotion_eligible": False,
            "scored_cell_count": 15,
        }
        run.attach_maturity_ratios(complete_summary, tier="T3")
        self.assertTrue(complete_summary["scored_complete"])
        self.assertEqual(complete_summary["frontier_lane"], "performance")
        self.assertEqual(complete_summary["promotion_lane"], "performance")

        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "custom_variant_T1_multi_benchmark.json"
            result = {"tier": "T1", "promotion_eligible": False}
            result_path.write_text(json.dumps(result), encoding="utf-8")
            run.sync_maturity_metadata_to_benchmark_result(
                result,
                metrics_summary,
                tier="T1",
                result_path=str(result_path),
            )
            saved = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["effort_ratio"], metrics_summary["effort_ratio"])
        self.assertEqual(saved["coverage_ratio"], metrics_summary["coverage_ratio"])
        self.assertEqual(saved["total_required_eval_units"], 15)
        self.assertEqual(saved["frontier_lane"], "task_candidate")
        self.assertEqual(saved["promotion_lane"], "task_candidate")

    def test_sam_canonical_summary_reaches_each_configured_parent_lane(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
            FrontierStore,
        )
        from templates.tasks.sam_optimizer.evaluations.pareto_tiered import run

        task_dir = REPO_ROOT / "templates" / "tasks" / "sam_optimizer"
        task = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
        evaluation = task["evaluation"]
        result = {
            "tier": "T3",
            "promotion_eligible": True,
            "mean_test_accuracy": {"mean": 0.78},
            "mean_train_test_gap": {"mean": 0.04},
            "sharpness_top_eigen": {"mean": 8.5},
            "per_dataset": {
                dataset: {
                    "status": "ok",
                    "test_accuracy": {"mean": accuracy},
                    "train_test_gap": {"mean": gap},
                    "train_time_seconds": {"mean": train_seconds},
                    "num_seeds_ok": 5,
                    "num_seeds_total": 5,
                }
                for dataset, accuracy, gap, train_seconds in (
                    ("cifar10", 0.95, 0.02, 60.0),
                    ("cifar100", 0.78, 0.05, 60.0),
                    ("tiny-imagenet", 0.61, 0.08, 220.0),
                )
            },
        }
        metrics = run.summarize_metrics(result)
        run.attach_maturity_ratios(metrics, tier="T3")
        suspect_metrics = run.summarize_metrics({**result, "suspect_leakage": True})
        run.attach_maturity_ratios(suspect_metrics, tier="T3")

        self.assertEqual(metrics["wall_time_seconds_total"], 1700.0)
        self.assertEqual(metrics["frontier_lane"], "performance")
        self.assertEqual(metrics["promotion_lane"], "performance")
        self.assertTrue(suspect_metrics["suspect_leakage"])
        self.assertTrue(suspect_metrics["scored_complete"])
        self.assertEqual(suspect_metrics["frontier_lane"], "task_candidate")
        self.assertEqual(suspect_metrics["promotion_lane"], "task_candidate")
        for marker in run.NON_PARENTABLE_MARKERS:
            marked = run.summarize_metrics({**result, marker: True})
            run.attach_maturity_ratios(marked, tier="T3")
            self.assertTrue(marked[marker], marker)
            self.assertEqual(marked["frontier_lane"], "task_candidate", marker)
        protocol_failed = run.summarize_metrics({**result, "protocol_integrity_passed": False})
        run.attach_maturity_ratios(protocol_failed, tier="T3")
        self.assertEqual(protocol_failed["frontier_lane"], "task_candidate")
        finding = {
            "id": "canonical_sam_result",
            "finding_type": "result",
            "variant_name": "canonical_sam_result",
            "metrics": metrics,
        }
        parent_lanes = [
            lane for lane in evaluation["frontier_lanes"] if lane.get("parent_eligible")
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for lane in parent_lanes:
                store = FrontierStore(
                    Path(tmp) / lane["name"],
                    primary_metric=evaluation["primary_metric"],
                    metric_direction=evaluation["direction"],
                    require_tier=evaluation["requires_tier"],
                    maturity_policy=evaluation["maturity_policy"],
                    frontier_lanes=[lane],
                )
                promoted = store.promote(0, [finding])
                self.assertEqual(
                    [entry["finding_id"] for entry in promoted],
                    ["canonical_sam_result"],
                    lane["name"],
                )
                suspect = {
                    **finding,
                    "id": f"suspect_{lane['name']}",
                    "variant_name": f"suspect_{lane['name']}",
                    "metrics": suspect_metrics,
                }
                suspect_store = FrontierStore(
                    Path(tmp) / f"{lane['name']}-suspect",
                    primary_metric=evaluation["primary_metric"],
                    metric_direction=evaluation["direction"],
                    require_tier=evaluation["requires_tier"],
                    maturity_policy=evaluation["maturity_policy"],
                    frontier_lanes=[lane],
                )
                self.assertEqual(suspect_store.promote(0, [suspect]), [], lane["name"])

    def test_install_script_links_all_repo_skills_and_lists_invocations(self) -> None:
        skill_names = self._repo_skill_names()
        self.assertIn("praxist-control", skill_names)
        self.assertIn("terminal-line-plot", skill_names)

        with tempfile.TemporaryDirectory(prefix="codex_skill_install_") as tmp:
            target = Path(tmp) / "skills"
            run = subprocess.run(
                ["bash", str(INSTALL_CODEX_SKILLS), "--target-dir", str(target)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, msg=run.stdout + run.stderr)
            manifest = json.loads((target / ".praxist-skills.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["managed_by"], "praxist")
            for skill_name in skill_names:
                self.assertTrue((target / skill_name).is_symlink())
                self.assertIn(f"${skill_name}", run.stdout)
                self.assertIn(skill_name, manifest["skills"])

    def test_skill_scripts_exist_and_are_executable_bash(self) -> None:
        for script in (INSTALL_CODEX_SKILLS, UNINSTALL_CODEX_SKILLS):
            self.assertTrue(script.exists(), f"{script.relative_to(REPO_ROOT)} missing")
            body = script.read_text(encoding="utf-8")
            self.assertRegex(body.splitlines()[0], r"^#!/usr/bin/env bash\b")
            self.assertIn("set -euo pipefail", body)
            self.assertTrue(_is_executable(script), f"{script.relative_to(REPO_ROOT)} needs +x")
        self.assertNotIn("--all-known-targets", UNINSTALL_CODEX_SKILLS.read_text(encoding="utf-8"))

    def test_install_script_repairs_manifest_owned_stale_symlink(self) -> None:
        with tempfile.TemporaryDirectory(prefix="codex_skill_migration_") as tmp:
            target = Path(tmp) / "skills"
            installed = subprocess.run(
                ["bash", str(INSTALL_CODEX_SKILLS), "--target-dir", str(target)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                installed.returncode,
                0,
                msg=installed.stdout + installed.stderr,
            )
            skill_name = "praxist-control"
            link = target / skill_name
            manifest_path = target / ".praxist-skills.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stale_source = Path(tmp) / "moved" / "skills" / skill_name
            manifest["skills"][skill_name]["source"] = str(stale_source)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            link.unlink()
            link.symlink_to(stale_source)

            refreshed = subprocess.run(
                ["bash", str(INSTALL_CODEX_SKILLS), "--target-dir", str(target)],
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                refreshed.returncode,
                0,
                msg=refreshed.stdout + refreshed.stderr,
            )
            self.assertEqual(
                link.resolve(strict=True),
                (REPO_ROOT / "skills" / skill_name).resolve(strict=True),
            )

    def test_skill_scripts_refuse_a_user_retargeted_symlink(self) -> None:
        with tempfile.TemporaryDirectory(prefix="codex_skill_retarget_") as tmp:
            root = Path(tmp)
            target = root / "skills"
            installed = subprocess.run(
                ["bash", str(INSTALL_CODEX_SKILLS), "--target-dir", str(target)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                installed.returncode,
                0,
                msg=installed.stdout + installed.stderr,
            )
            skill_name = "praxist-control"
            user_skill = root / "user" / "skills" / skill_name
            user_skill.mkdir(parents=True)
            (user_skill / "SKILL.md").write_text("# User skill\n", encoding="utf-8")
            link = target / skill_name
            link.unlink()
            link.symlink_to(user_skill, target_is_directory=True)

            refresh = subprocess.run(
                ["bash", str(INSTALL_CODEX_SKILLS), "--target-dir", str(target)],
                capture_output=True,
                text=True,
            )
            removed = subprocess.run(
                ["bash", str(UNINSTALL_CODEX_SKILLS), "--target-dir", str(target)],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(refresh.returncode, 0)
            self.assertIn("not Praxist-managed", refresh.stderr)
            self.assertNotEqual(removed.returncode, 0)
            self.assertIn("refused unmanaged path", removed.stderr)
            self.assertEqual(link.resolve(strict=True), user_skill.resolve(strict=True))

    def test_skill_scripts_preserve_unowned_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="codex_skill_manifest_") as tmp:
            target = Path(tmp) / "skills"
            target.mkdir(parents=True)
            manifest = target / ".praxist-skills.json"
            sentinel = b'{"managed_by":"another-tool","skills":{}}\n'
            manifest.write_bytes(sentinel)

            install = subprocess.run(
                ["bash", str(INSTALL_CODEX_SKILLS), "--target-dir", str(target)],
                capture_output=True,
                text=True,
            )
            uninstall = subprocess.run(
                ["bash", str(UNINSTALL_CODEX_SKILLS), "--target-dir", str(target)],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(install.returncode, 0)
            self.assertNotEqual(uninstall.returncode, 0)
            self.assertIn("not owned by Praxist", install.stderr)
            self.assertIn("not owned by Praxist", uninstall.stderr)
            self.assertEqual(manifest.read_bytes(), sentinel)

    def test_skill_scripts_preserve_malformed_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="codex_skill_manifest_") as tmp:
            target = Path(tmp) / "skills"
            target.mkdir(parents=True)
            manifest = target / ".praxist-skills.json"
            sentinel = b"{not-json\n"
            manifest.write_bytes(sentinel)

            install = subprocess.run(
                ["bash", str(INSTALL_CODEX_SKILLS), "--target-dir", str(target)],
                capture_output=True,
                text=True,
            )
            uninstall = subprocess.run(
                ["bash", str(UNINSTALL_CODEX_SKILLS), "--target-dir", str(target)],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(install.returncode, 0)
            self.assertNotEqual(uninstall.returncode, 0)
            self.assertIn("unreadable or invalid", install.stderr)
            self.assertIn("unreadable or invalid", uninstall.stderr)
            self.assertEqual(manifest.read_bytes(), sentinel)

    def test_skill_scripts_remove_manifest_owned_retired_skill(self) -> None:
        with tempfile.TemporaryDirectory(prefix="codex_skill_retired_") as tmp:
            root = Path(tmp)
            target = root / "skills"
            installed = subprocess.run(
                ["bash", str(INSTALL_CODEX_SKILLS), "--target-dir", str(target)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                installed.returncode,
                0,
                msg=installed.stdout + installed.stderr,
            )
            retired_source = root / "old" / "skills" / "praxist-retired"
            retired_source.mkdir(parents=True)
            (retired_source / "SKILL.md").write_text("# Retired\n", encoding="utf-8")
            retired_dest = target / retired_source.name
            retired_dest.symlink_to(retired_source, target_is_directory=True)
            manifest_path = target / ".praxist-skills.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["skills"][retired_source.name] = {
                "managed_by": "praxist",
                "mode": "symlink",
                "source": str(retired_source),
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            refreshed = subprocess.run(
                ["bash", str(INSTALL_CODEX_SKILLS), "--target-dir", str(target)],
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                refreshed.returncode,
                0,
                msg=refreshed.stdout + refreshed.stderr,
            )
            self.assertFalse(retired_dest.exists() or retired_dest.is_symlink())
            refreshed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertNotIn(retired_source.name, refreshed_manifest["skills"])

            retired_dest.symlink_to(retired_source, target_is_directory=True)
            refreshed_manifest["skills"][retired_source.name] = {
                "managed_by": "praxist",
                "mode": "symlink",
                "source": str(retired_source),
            }
            manifest_path.write_text(json.dumps(refreshed_manifest), encoding="utf-8")
            removed = subprocess.run(
                ["bash", str(UNINSTALL_CODEX_SKILLS), "--target-dir", str(target)],
                capture_output=True,
                text=True,
            )

            self.assertEqual(removed.returncode, 0, msg=removed.stdout + removed.stderr)
            self.assertFalse(retired_dest.exists() or retired_dest.is_symlink())
            self.assertFalse(manifest_path.exists())

    def test_terminal_line_plot_script_renders_curve_from_points(self) -> None:
        self.assertTrue(TERMINAL_LINE_PLOT.exists(), "terminal line plot helper missing")
        proc = subprocess.run(
            [
                sys.executable,
                str(TERMINAL_LINE_PLOT),
                "--points",
                "0:1,1:3,2:2",
                "--title",
                "Test curve",
                "--x-label",
                "gen",
                "--y-label",
                "score",
                "--higher-better",
                "--provisional-x",
                "2",
                "--height",
                "6",
                "--x-step",
                "4",
                "--value-table",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertIn("Test curve", proc.stdout)
        self.assertIn("higher is better", proc.stdout)
        self.assertIn("○ = provisional/incomplete", proc.stdout)
        self.assertIn("●", proc.stdout)
        self.assertIn("○", proc.stdout)
        self.assertIn("values:", proc.stdout)

    def test_removes_all_repo_skill_symlinks_and_leaves_unknown_skills(self) -> None:
        skill_names = self._repo_skill_names()
        self.assertGreater(len(skill_names), 0)
        self.assertIn("praxist-onboarding", skill_names)
        self.assertIn("praxist-control", skill_names)

        with tempfile.TemporaryDirectory(prefix="codex_skill_uninstall_") as tmp:
            root = Path(tmp)
            target = root / "skills"
            installed = subprocess.run(
                ["bash", str(INSTALL_CODEX_SKILLS), "--target-dir", str(target)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                installed.returncode,
                0,
                msg=installed.stdout + installed.stderr,
            )
            unknown = target / "unrelated-skill"
            unknown.mkdir()

            dry_run = subprocess.run(
                [
                    "bash",
                    str(UNINSTALL_CODEX_SKILLS),
                    "--target-dir",
                    str(target),
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(dry_run.returncode, 0, msg=dry_run.stdout + dry_run.stderr)
            for skill_name in skill_names:
                self.assertTrue((target / skill_name).is_symlink())

            run = subprocess.run(
                ["bash", str(UNINSTALL_CODEX_SKILLS), "--target-dir", str(target)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, msg=run.stdout + run.stderr)
            self.assertIn(f"removed {len(skill_names)} Praxist skill(s)", run.stderr)
            for skill_name in skill_names:
                skill_link = target / skill_name
                self.assertFalse(skill_link.exists() or skill_link.is_symlink())
            self.assertFalse((target / ".praxist-skills.json").exists())
            self.assertTrue(unknown.exists())

    def test_refuses_copied_skill_and_has_no_force_override(self) -> None:
        with tempfile.TemporaryDirectory(prefix="codex_skill_uninstall_") as tmp:
            target = Path(tmp) / "skills"
            copied_skill = target / "praxist-runtime-install"
            copied_skill.mkdir(parents=True)
            (copied_skill / "SKILL.md").write_text("copied\n", encoding="utf-8")

            refused = subprocess.run(
                ["bash", str(UNINSTALL_CODEX_SKILLS), "--target-dir", str(target)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(refused.returncode, 0, msg=refused.stdout + refused.stderr)
            self.assertIn("refused unmanaged path", refused.stderr)
            self.assertTrue(copied_skill.exists())

            force_is_unsupported = subprocess.run(
                [
                    "bash",
                    str(UNINSTALL_CODEX_SKILLS),
                    "--target-dir",
                    str(target),
                    "--force",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(force_is_unsupported.returncode, 2)
            self.assertIn("unknown argument: --force", force_is_unsupported.stderr)
            self.assertTrue(copied_skill.exists())

    def test_rejects_all_known_targets_option(self) -> None:
        rejected = subprocess.run(
            ["bash", str(UNINSTALL_CODEX_SKILLS), "--all-known-targets"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(rejected.returncode, 2, msg=rejected.stdout + rejected.stderr)
        self.assertIn("unknown argument: --all-known-targets", rejected.stderr)


class AgentsMdWiringRecipe(unittest.TestCase):
    """A new contributor who only reads AGENTS.md should discover how to
    wire the tracked scripts under the gitignored .agent/ tree."""

    def setUp(self) -> None:
        self.text = AGENTS_MD.read_text(encoding="utf-8")

    def test_pre_commit_section_mentions_scripts_dev_path(self) -> None:
        self.assertIn("scripts/dev/run_guardrails.py", self.text)

    def test_documents_symlink_recipe(self) -> None:
        # Either a literal `ln -s` example or an explicit "symlink"
        # mention is acceptable; the recipe must be present.
        self.assertRegex(self.text, r"(ln\s+-s|symlink).*\.agent")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
