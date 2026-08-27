from __future__ import annotations

import unittest
from pathlib import Path


class ResearchLoopDecompositionTest(unittest.TestCase):
    """Guard the workflow-stage-internal decomposition boundaries."""

    repo_root = Path(__file__).resolve().parents[2]

    def _read(self, relpath: str) -> str:
        return (self.repo_root / relpath).read_text(encoding="utf-8")

    def test_generation_loop_delegates_peer_execution_and_boundary_work(self) -> None:
        source = self._read(
            "praxist/plugins/workflow_stages/research_loop/backend/generation_loop.py"
        )
        self.assertLess(
            len(source.splitlines()),
            925,
            "GenerationLoop should remain an orchestration facade, not absorb backend logic again.",
        )
        self.assertNotIn("AutonomousAgentLoop", source)
        self.assertNotIn("resolve_prompt_with_layout", source)
        self.assertIn("run_generation_cohort", source)
        self.assertIn("complete_generation_boundary", source)
        self.assertIn("build_prompt_context", source)
        self.assertIn("build_orchestrator_status_snapshot", source)
        self.assertNotIn("record_generation_finished_safely", source)
        boundary = self._read(
            "praxist/plugins/workflow_stages/research_loop/backend/generation_boundary.py"
        )
        self.assertIn("record_completed_generation_observation", boundary)

    def test_research_loop_backend_has_named_responsibility_modules(self) -> None:
        backend = self.repo_root / "praxist/plugins/workflow_stages/research_loop/backend"
        expected_modules = {
            "baseline_runtime.py",
            "cohort_runner.py",
            "findings_collection.py",
            "generation_boundary.py",
            "prompt_artifacts.py",
            "prompt_context.py",
            "prompt_strategy.py",
            "research_memory_update.py",
            "runtime_environment.py",
            "sidecars.py",
            "status_snapshot.py",
        }
        self.assertTrue(expected_modules.issubset({path.name for path in backend.glob("*.py")}))

    def test_startup_does_not_own_legacy_materialization_internals(self) -> None:
        startup = self._read("praxist/plugins/workflow_stages/research_loop/startup.py")
        materializer = self._read(
            "praxist/plugins/workflow_stages/research_loop/legacy_output_materializer.py"
        )
        # Line-count is a soft architectural budget guarding against
        # legacy materialization code leaking back into startup. The
        # structural assertions below (``_collect_legacy_findings`` /
        # ``_canonical_frontier_record`` live in the materializer, not
        # here) carry the load; the line cap exists to catch silent
        # growth. #75 batch 8b raised the cap from 1300 to 1310 to
        # accommodate the ``PRAXIST_*`` env-name alias loops in
        # ``_apply_internal_env_overrides`` — those are env-discipline
        # additions, not materialization internals. The feature_merge branches
        # added provider/runtime env plumbing plus task-writable-root guard
        # compatibility for declared venv bootstrap; keep the structural
        # materializer assertions as the hard guard.
        self.assertLess(
            len(startup.splitlines()),
            1380,
            "Startup should not re-absorb legacy output materialization internals.",
        )
        self.assertIn("_materialize_legacy_outputs", startup)
        self.assertNotIn("def _collect_legacy_findings", startup)
        self.assertNotIn("def _canonical_frontier_record", startup)
        self.assertIn("def _collect_legacy_findings", materializer)
        self.assertIn("def _canonical_frontier_record", materializer)


if __name__ == "__main__":
    unittest.main()
