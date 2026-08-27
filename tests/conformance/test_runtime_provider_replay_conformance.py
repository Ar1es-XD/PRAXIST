from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from praxist.core.replay import verify_run
from praxist.core.storage import write_json
from praxist.plugins.workflow_stages.research_loop.backend.agent import BaseAgent
from praxist.plugins.workflow_stages.research_loop.startup import (
    finalize_research_loop_plugin_run,
    is_research_loop_task_project,
    prepare_research_loop_plugin_run,
)
from praxist.testing.fake_workflow_fixture import run_fake_workflow_fixture


class Step21RuntimeProviderConformanceTests(unittest.TestCase):
    def test_research_loop_task_eligibility_is_manifest_backed_for_non_sam_task(self) -> None:
        startup_text = Path("praxist/plugins/workflow_stages/research_loop/startup.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("RESEARCH_LOOP_PLUGIN_TASKS", startup_text)
        self.assertTrue(is_research_loop_task_project(_toy_task_path(), workspace=Path.cwd()))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_toy_math_step21"
            prepared = _prepare_toy(run_dir, root)
            finalize_research_loop_plugin_run(
                prepared,
                success=True,
                result={
                    "generations_completed": 0,
                    "run_dir": str(run_dir),
                    "exit_condition": "resolve_only",
                },
            )

            report = verify_run(run_dir)
            self.assertTrue(report["success"], report)
            resolution = json.loads(
                (run_dir / "plugin_resolution.json").read_text(encoding="utf-8")
            )
            selected_refs = {
                item["metadata"]["kind"] + ":" + item["metadata"]["name"]
                for item in resolution["selected"]
            }
            self.assertNotIn("task:toy_math", selected_refs)
            self.assertIn("workflow_stage:research_loop", selected_refs)
            self.assertIn("agent_runtime:fake_runtime", selected_refs)
            self.assertTrue((run_dir / "task_project_manifest.json").exists())

            cache_policy = json.loads((run_dir / "cache_policy.json").read_text(encoding="utf-8"))
            self.assertEqual(cache_policy["mode"], "disabled")
            self.assertIsNone(cache_policy["frozen_prefix_hash"])

            profiles = json.loads((run_dir / "model_profiles.json").read_text(encoding="utf-8"))
            conformance = profiles["runtime_provider_conformance"]
            self.assertEqual(conformance["runtime_ref"], "agent_runtime:fake_runtime")
            self.assertEqual(conformance["model_provider_ref"], "model_provider:fake_provider")
            self.assertEqual(conformance["cache_mode"], "disabled")
            self.assertEqual(conformance["runtime_cache_strategy"], "deterministic_no_cache")
            self.assertEqual(conformance["provider_cache_strategy"], "deterministic_no_cache")

    def test_runtime_provider_compatibility_rejects_manifest_mismatch_before_execution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "not compatible"):
                prepare_research_loop_plugin_run(
                    task_project_path=_toy_task_path(),
                    workspace=root,
                    run_dir=root / "run_incompatible_runtime_provider",
                    runtime_ref="agent_runtime:claude_sdk",
                    model_provider_ref="model_provider:fake_provider",
                    budget_policy_ref="budget_policy:fake_tiered",
                    model="fake-deterministic",
                    local_mode=True,
                    frontier_strategy="auto",
                    credential_profile="fake_multi_key",
                    command="step21 incompatible test",
                )

    def test_base_agent_uses_non_claude_runtime_adapter_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "PRAXIST_RUN_ID": "run_step21_base_agent",
                "PRAXIST_STAGE_ID": "research_loop",
                "PRAXIST_AGENT_RUNTIME_REF": "agent_runtime:fake_runtime",
                "PRAXIST_MODEL_PROVIDER_REF": "model_provider:fake_provider",
                "PRAXIST_MODEL_PROFILE_REF": "cheap_peer",
            }
            with patch.dict(os.environ, env, clear=False):
                result = asyncio.run(
                    BaseAgent(
                        name="peer0",
                        allowed_tools=[],
                        workspace=root,
                        mcp_servers={},
                        model="fake-deterministic",
                    ).execute("return deterministic output")
                )

        self.assertTrue(result.success, result)
        self.assertEqual(result.duration, 0.0)
        self.assertEqual(result.iteration_count, 0)
        self.assertIn(
            "deterministic fake response", " ".join(result.output.get("text_outputs", []))
        )

    def test_replay_rejects_runtime_provider_cache_contract_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_cache_drift"
            prepared = _prepare_toy(run_dir, root)
            finalize_research_loop_plugin_run(
                prepared,
                success=True,
                result={
                    "generations_completed": 0,
                    "run_dir": str(run_dir),
                    "exit_condition": "resolve_only",
                },
            )

            cache_policy = json.loads((run_dir / "cache_policy.json").read_text(encoding="utf-8"))
            cache_policy["mode"] = "provider_default"
            write_json(run_dir / "cache_policy.json", cache_policy)

            report = verify_run(run_dir)
            self.assertFalse(report["success"], report)
            self.assertTrue(
                any(
                    "cache_policy mode mismatch for runtime/provider" in error
                    for error in report["errors"]
                ),
                report,
            )


class Step22StateRecoveryConformanceTests(unittest.TestCase):
    def test_replay_reports_state_surfaces_not_recovered_to_canonical_ledgers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_state_recovery_drift"
            prepared = _prepare_toy(run_dir, root)
            finalize_research_loop_plugin_run(
                prepared,
                success=True,
                result={
                    "generations_completed": 0,
                    "run_dir": str(run_dir),
                    "exit_condition": "resolve_only",
                },
            )

            _write_unrecovered_legacy_state(run_dir)
            report = verify_run(run_dir)
            self.assertFalse(report["success"], report)
            joined = "\n".join(report["errors"])
            self.assertIn("canonical findings missing shared_findings ids", joined)
            self.assertIn("canonical findings missing SQLite finding ids", joined)
            self.assertIn("canonical frontier missing legacy frontier ids", joined)
            self.assertIn("canonical graph_edges missing SQLite edge ids", joined)
            self.assertTrue((run_dir / "replay" / "state_recovery_report.json").exists())

    def test_fake_panel_replay_writes_empty_state_recovery_report_for_clean_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            report = verify_run(run_dir)
            self.assertTrue(report["success"], report)
            state_report = json.loads(
                (run_dir / "replay" / "state_recovery_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state_report["schema_version"], "praxist.state_recovery.v1")
            self.assertEqual(state_report["missing"]["shared_findings_not_in_canonical"], [])


def _prepare_toy(run_dir: Path, workspace: Path):
    return prepare_research_loop_plugin_run(
        task_project_path=_toy_task_path(),
        workspace=workspace,
        run_dir=run_dir,
        runtime_ref="agent_runtime:fake_runtime",
        model_provider_ref="model_provider:fake_provider",
        budget_policy_ref="budget_policy:fake_tiered",
        model="fake-deterministic",
        local_mode=True,
        frontier_strategy="auto",
        credential_profile="fake_multi_key",
        command="step21/22 toy test",
    )


def _toy_task_path() -> Path:
    return Path.cwd() / "templates" / "tasks" / "toy_math"


def _write_unrecovered_legacy_state(run_dir: Path) -> None:
    shared_dir = run_dir / "shared_findings"
    shared_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        shared_dir / "legacy_shared_only.json",
        {
            "id": "legacy_shared_only",
            "title": "unrecovered shared finding",
            "metrics": {"score": 1.0},
        },
    )

    frontier_dir = run_dir / "frontier"
    frontier_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        frontier_dir / "frontier_manifest.json",
        {"cumulative_top": [{"finding_id": "legacy_frontier_only", "metric_value": 1.0}]},
    )

    db_path = run_dir / "shared_store.db"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE findings (id TEXT)")
        conn.execute("INSERT INTO findings VALUES ('legacy_sqlite_only')")
        conn.execute("CREATE TABLE finding_edges (edge_id TEXT)")
        conn.execute("INSERT INTO finding_edges VALUES ('legacy_edge_only')")
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    unittest.main()
