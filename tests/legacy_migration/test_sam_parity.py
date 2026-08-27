from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from praxist.core.ledgers import BudgetLedger
from praxist.core.storage import write_json
from praxist.deliver import package_deliverables
from praxist.plugins.workflow_stages.research_loop.backend.parity import (
    verify_research_loop_parity,
)
from praxist.plugins.workflow_stages.research_loop.startup import (
    finalize_research_loop_plugin_run,
    prepare_research_loop_plugin_run,
)
from tests.helpers.paths import REPO_ROOT


class Step17SamParityDogfoodTest(unittest.TestCase):
    def test_full_parity_harness_accepts_complete_synthetic_research_loop_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_step17"
            prepared = _prepare(run_dir, root)
            _write_legacy_dogfood_surfaces(run_dir)
            _record_resource_usage(prepared)
            finalize_research_loop_plugin_run(
                prepared,
                success=True,
                result={
                    "generations_completed": 2,
                    "max_generations": 2,
                    "run_dir": str(run_dir),
                    "exit_condition": "completed",
                    "frontier_summary": [
                        {
                            "finding_id": "legacy_finding_1",
                            "variant_name": "step17 variant",
                            "metric_name": "mean_test_accuracy",
                            "metric_value": 0.91,
                            "peer_id": "gen0_peer0",
                        }
                    ],
                },
            )
            deliverables = package_deliverables(
                str(run_dir),
                str(root / "deliverables"),
                name="step17_pack",
                overwrite=True,
            )

            report = verify_research_loop_parity(
                run_dir, deliverables_dir=deliverables, strict=True
            )

            self.assertTrue(report["success"], report)
            checks = {item["check_id"]: item for item in report["checks"]}
            for check_id in (
                "legacy_findings_materialized",
                "frontier_materialized",
                "research_memory_materialized",
                "graph_edges_materialized",
                "graph_artifacts_materialized",
                "prompt_guidance_surfaces",
                "panel_agenda_surface",
                "operator_status_surface",
                "resource_guard_usage",
                "deliverables_package",
            ):
                self.assertEqual(checks[check_id]["status"], "pass", check_id)

    def test_parity_harness_rejects_missing_graph_guidance_in_postgen_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_step17_bad_prompt"
            prepared = _prepare(run_dir, root)
            _write_legacy_dogfood_surfaces(run_dir, include_graph_prompt=False)
            _record_resource_usage(prepared)
            finalize_research_loop_plugin_run(
                prepared,
                success=True,
                result={
                    "generations_completed": 2,
                    "run_dir": str(run_dir),
                    "exit_condition": "completed",
                    "frontier_summary": [
                        {"finding_id": "legacy_finding_1", "peer_id": "gen0_peer0"}
                    ],
                },
            )

            report = verify_research_loop_parity(run_dir, strict=True)

            self.assertFalse(report["success"])
            checks = {item["check_id"]: item for item in report["checks"]}
            self.assertEqual(checks["prompt_guidance_surfaces"]["status"], "fail")
            self.assertIn(
                "Graph-surfaced context", checks["prompt_guidance_surfaces"]["details"]["missing"]
            )

    def test_parity_cli_returns_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_step17_cli"
            prepared = _prepare(run_dir, root)
            _write_legacy_dogfood_surfaces(run_dir)
            _record_resource_usage(prepared)
            finalize_research_loop_plugin_run(
                prepared,
                success=True,
                result={
                    "generations_completed": 2,
                    "run_dir": str(run_dir),
                    "exit_condition": "completed",
                    "frontier_summary": [
                        {"finding_id": "legacy_finding_1", "peer_id": "gen0_peer0"}
                    ],
                },
            )
            deliverables = package_deliverables(
                str(run_dir),
                str(root / "deliverables"),
                name="step17_cli_pack",
                overwrite=True,
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "praxist.run",
                    "parity",
                    str(run_dir),
                    "--deliverables-dir",
                    str(deliverables),
                    "--strict",
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            payload = json.loads(proc.stdout)
            self.assertTrue(payload["success"], payload)
            self.assertEqual(payload["schema_version"], "praxist.research_loop_parity.v1")


def _prepare(run_dir: Path, workspace: Path):
    with patch.dict(os.environ, {}, clear=False):
        return prepare_research_loop_plugin_run(
            task_project_path=Path.cwd() / "templates" / "tasks" / "toy_math",
            workspace=workspace,
            run_dir=run_dir,
            runtime_ref="agent_runtime:fake_runtime",
            model_provider_ref="model_provider:fake_provider",
            budget_policy_ref="budget_policy:fake_tiered",
            model="fake-deterministic",
            local_mode=True,
            frontier_strategy="auto",
            credential_profile="fake_multi_key",
            command="step17 test",
        )


def _write_legacy_dogfood_surfaces(run_dir: Path, *, include_graph_prompt: bool = True) -> None:
    _write_findings_and_edges(run_dir)
    _write_frontier_manifest(run_dir)
    _write_research_memory(run_dir)
    _write_graph_artifacts(run_dir)
    _write_prompts_and_agenda(run_dir, include_graph_prompt=include_graph_prompt)
    _write_operator_status(run_dir)


def _write_findings_and_edges(run_dir: Path) -> None:
    shared = run_dir / "shared_findings"
    shared.mkdir(parents=True, exist_ok=True)
    for finding in (
        {
            "id": "legacy_finding_1",
            "finding_type": "result",
            "title": "Step17 frontier result",
            "content": "Validated frontier direction with graph context.",
            "metrics": {"mean_test_accuracy": 0.91},
            "variant_name": "step17_variant",
            "peer_id": "gen0_peer0",
            "generation_id": 0,
        },
        {
            "id": "legacy_finding_2",
            "finding_type": "hypothesis",
            "title": "Step17 supporting hypothesis",
            "content": "Supports the frontier result with related mechanism.",
            "metrics": {},
            "variant_name": "step17_variant",
            "peer_id": "gen0_peer1",
            "generation_id": 0,
        },
    ):
        (shared / f"{finding['id']}.json").write_text(
            json.dumps(finding, sort_keys=True), encoding="utf-8"
        )

    with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}, clear=False):
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        local_store.init_db()
        local_store.insert_finding(
            {
                "id": "legacy_finding_1",
                "finding_type": "result",
                "title": "Step17 frontier result",
                "content": "Validated frontier direction with graph context.",
                "metrics": {"mean_test_accuracy": 0.91},
                "variant_name": "step17_variant",
                "peer_id": "gen0_peer0",
                "generation_id": 0,
            }
        )
        local_store.insert_finding(
            {
                "id": "legacy_finding_2",
                "finding_type": "hypothesis",
                "title": "Step17 supporting hypothesis",
                "content": "Supports the frontier result with related mechanism.",
                "metrics": {},
                "variant_name": "step17_variant",
                "peer_id": "gen0_peer1",
                "generation_id": 0,
            }
        )
        local_store.insert_edge(
            {
                "edge_id": "edge_step17_1",
                "src_finding_id": "legacy_finding_1",
                "dst_finding_id": "legacy_finding_2",
                "edge_type": "supports",
                "confidence": 0.82,
                "created_by": "legacy_graph",
                "created_at": "2026-05-10T00:00:01Z",
                "rationale": "shared mechanism",
                "provenance": {"source": "unit_test"},
            }
        )


def _write_frontier_manifest(run_dir: Path) -> None:
    frontier = run_dir / "frontier"
    frontier.mkdir(parents=True, exist_ok=True)
    write_json(
        frontier / "frontier_manifest.json",
        {
            "primary_metric": "mean_test_accuracy",
            "cumulative_top": [
                {
                    "finding_id": "legacy_finding_1",
                    "variant_name": "step17_variant",
                    "metric_name": "mean_test_accuracy",
                    "metric_value": 0.91,
                    "peer_id": "gen0_peer0",
                }
            ],
            "generations": {
                "0": [
                    {
                        "finding_id": "legacy_finding_1",
                        "variant_name": "step17_variant",
                        "metric_name": "mean_test_accuracy",
                        "metric_value": 0.91,
                        "peer_id": "gen0_peer0",
                    }
                ]
            },
        },
    )


def _write_research_memory(run_dir: Path) -> None:
    ledger_dir = run_dir / "research_memory" / "ledgers"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    (ledger_dir / "claim_ledger.yaml").write_text(
        """
ledger_name: claim_ledger
entries:
  - id: claim_step17_1
    created_at: "2026-05-10T00:00:00Z"
    data:
      confidence: 0.82
      source_ref:
        finding_id: legacy_finding_1
        kind: finding
""".lstrip(),
        encoding="utf-8",
    )


def _write_graph_artifacts(run_dir: Path) -> None:
    graph = run_dir / "graph"
    graph.mkdir(parents=True, exist_ok=True)
    (graph / "graph_health.json").write_text(
        '{"node_count": 2, "edge_count": 1}\n', encoding="utf-8"
    )
    (graph / "unlinked_recent_findings.json").write_text(
        '[{"id": "legacy_finding_2"}]\n', encoding="utf-8"
    )
    (graph / "graph.html").write_text("<html><body>legacy graph</body></html>\n", encoding="utf-8")


def _write_prompts_and_agenda(run_dir: Path, *, include_graph_prompt: bool) -> None:
    gen1 = run_dir / "gen_1"
    gen1.mkdir(parents=True, exist_ok=True)
    graph_block = (
        "## Graph-surfaced context\n"
        "- supports (0.82) `legacy_finding_2`\n"
        "Call `mcp__finding-graph-query__get_unlinked_recent_findings`.\n"
        if include_graph_prompt
        else "No graph context rendered here.\n"
    )
    (gen1 / "gen1_peer0_prompt.md").write_text(
        (
            "# Peer prompt\n"
            f"{graph_block}\n"
            "Current frontier includes `legacy_finding_1`; inspect frontier gap before proposing work.\n"
            "Research agenda contract: build on step17_variant.\n"
        ),
        encoding="utf-8",
    )
    agendas = run_dir / "agendas"
    agendas.mkdir(parents=True, exist_ok=True)
    (agendas / "research_agenda_gen1.yaml").write_text(
        """
generation: 1
mainline_observation:
  summary: continue the frontier result
cross_peer_hypotheses:
  - id: h_step17
    description: graph-supported followup
peer_contracts:
  gen1_peer0:
    role: builder
    objective: replicate and extend legacy_finding_1
""".lstrip(),
        encoding="utf-8",
    )


def _write_operator_status(run_dir: Path) -> None:
    write_json(
        run_dir / "orchestrator_status.final.json",
        {
            "run_started_at": "2026-05-10T00:00:00Z",
            "updated_at": "2026-05-10T00:10:00Z",
            "run_dir": str(run_dir),
            "task_id": "toy_math",
            "task_name": "Toy Math",
            "current_generation": 2,
            "max_generations": 2,
            "cohort_size": 1,
            "strategy": "auto",
            "generations_completed": 2,
            "findings_total": 2,
            "frontier_candidates": 1,
            "exit_condition": "completed",
        },
    )


def _record_resource_usage(prepared) -> None:
    ledger = BudgetLedger(prepared.run_dir, prepared.run_id)
    grant = ledger.require_active_grant(prepared.stage_budget_grant_id)
    request_id = str(grant.get("request_id") or "budget_request_research_loop_start")
    ledger.append_usage(
        request_id=request_id,
        grant_id=prepared.stage_budget_grant_id,
        actor_ref="evaluation_runner:toy_math",
        stage_id="research_loop",
        action_type="eval_runner",
        actual_usage={"wall_clock_seconds": 1.0},
        reason="step17_test_eval_usage",
    )
    ledger.append_usage(
        request_id=request_id,
        grant_id=prepared.stage_budget_grant_id,
        actor_ref="resource_guard:gpu_governor",
        stage_id="research_loop",
        action_type="gpu_slot",
        actual_usage={"gpu_hours": 0.001},
        reason="step17_test_gpu_usage",
    )


if __name__ == "__main__":
    unittest.main()
