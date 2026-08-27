from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from praxist.core.storage import ArtifactWriter
from praxist.core.trajectory import TrajectoryWriter
from praxist.core.workflow import (
    OPTIONAL_STAGE_CONTRACTS,
    OptionalWorkflowStageContext,
    optional_workflow_stage,
)
from praxist.plugins.workflow_stages.reviewer_stub.adapter import run_local_artifact_review


class ReviewerStageTests(unittest.TestCase):
    def test_trajectory_writer_serializes_concurrent_processes(self) -> None:
        script = """
import sys
from pathlib import Path
from praxist.core.trajectory import TrajectoryWriter

run_dir = Path(sys.argv[1])
for index in range(int(sys.argv[2])):
    TrajectoryWriter(run_dir, run_dir.name).emit(
        "test.concurrent",
        actor={"type": "test", "id": sys.argv[3]},
        payload={"index": index},
    )
"""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_concurrent_trajectory"
            run_dir.mkdir()
            processes = [
                subprocess.Popen(
                    [sys.executable, "-c", script, str(run_dir), "20", str(worker)],
                    cwd=Path(__file__).resolve().parents[2],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for worker in range(4)
            ]
            for process in processes:
                stdout, stderr = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 0, stdout + stderr)

            records = [
                json.loads(line)
                for line in (run_dir / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([record["seq"] for record in records], list(range(1, 81)))
        self.assertEqual(len({record["event_id"] for record in records}), 80)
        from praxist.core.replay import _verify_trajectory

        errors: list[str] = []
        _verify_trajectory(records, errors)
        self.assertEqual(errors, [])

    def test_local_reviewer_persists_audit_report_without_runtime_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_review"
            run_dir.mkdir()
            trajectory = TrajectoryWriter(run_dir, run_dir.name)
            writer = ArtifactWriter(run_dir, trajectory)
            artifact = writer.persist_json(
                "result.summary",
                "results/variant/summary.json",
                {"score": 1.0},
                schema_ref="task:summary.v1",
                producer={"stage_id": "research_loop"},
                artifact_role="canonical_state",
                artifact_status="committed",
                runtime_fact_source=True,
            )
            trajectory.emit(
                "test.event",
                artifact_refs=[artifact],
                payload={"note": "source event"},
            )

            result = asyncio.run(
                optional_workflow_stage("reviewer").execute(
                    OptionalWorkflowStageContext(
                        run_dir=run_dir,
                        run_id=run_dir.name,
                        enabled=True,
                        mode="local",
                    )
                )
            )

            self.assertTrue(result.success)
            self.assertEqual(result.status, "succeeded")
            self.assertEqual(result.summary["promotion_effect"], "none")
            review_artifact = result.output_artifacts[0]
            self.assertFalse(review_artifact["runtime_fact_source"])
            payload_path = run_dir / review_artifact["payload_path"]
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["artifact_semantics"]["role"], "audit_snapshot")
            self.assertEqual(payload["summary"]["artifact_count"], 1)

    def test_local_reviewer_reports_corrupt_artifact_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_review_corrupt"
            run_dir.mkdir()
            trajectory = TrajectoryWriter(run_dir, run_dir.name)
            writer = ArtifactWriter(run_dir, trajectory)
            artifact = writer.persist_json(
                "result.summary",
                "results/variant/summary.json",
                {"score": 1.0},
                schema_ref="task:summary.v1",
                producer={"stage_id": "research_loop"},
            )
            (run_dir / artifact["payload_path"]).write_text('{"score": 2.0}\n', encoding="utf-8")

            result = asyncio.run(
                optional_workflow_stage("reviewer").execute(
                    OptionalWorkflowStageContext(
                        run_dir=run_dir,
                        run_id=run_dir.name,
                        enabled=True,
                        mode="claim_check",
                    )
                )
            )

            payload = json.loads(
                (run_dir / result.output_artifacts[0]["payload_path"]).read_text(encoding="utf-8")
            )
            check_ids = {finding["check_id"] for finding in payload["findings"]}
            self.assertIn("artifact_hash_mismatch", check_ids)

    def test_local_reviewer_reports_provenance_and_summary_issues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_review_provenance"
            run_dir.mkdir()
            (run_dir / "artifact_index.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "artifact_id": "artifact:incomplete",
                                "artifact_type": "result.summary",
                            }
                        ),
                        json.dumps(
                            {
                                "artifact_id": "artifact:missing_payload",
                                "artifact_type": "result.summary",
                                "payload_path": "missing.json",
                                "content_hash": "sha256:missing",
                                "source_artifact_ids": ["artifact:nope"],
                            }
                        ),
                        json.dumps(
                            {
                                "artifact_id": "artifact:literature",
                                "artifact_type": "literature.record",
                                "logical_path": "literature/context.json",
                                "payload_path": "literature/context.json",
                                "content_hash": "sha256:bad",
                                "runtime_fact_source": True,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "trajectory.jsonl").write_text(
                json.dumps(
                    {
                        "event_id": "event:1",
                        "artifact_refs": [{"artifact_id": "artifact:not_in_index"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "run_summary.json").write_text("[1]\n", encoding="utf-8")

            result = run_local_artifact_review(
                run_dir=run_dir,
                run_id=run_dir.name,
                stage_ref="workflow_stage:reviewer_stub",
            )

            self.assertTrue(result["success"])
            payload = json.loads(
                (run_dir / result["output_artifacts"][0]["payload_path"]).read_text(
                    encoding="utf-8"
                )
            )
            check_ids = {finding["check_id"] for finding in payload["findings"]}
            self.assertIn("artifact_metadata_incomplete", check_ids)
            self.assertIn("artifact_payload_missing", check_ids)
            self.assertIn("artifact_source_ref_missing", check_ids)
            self.assertIn("trajectory_artifact_ref_missing", check_ids)
            self.assertIn("literature_marked_runtime_fact", check_ids)
            self.assertIn("run_summary_not_object", check_ids)

    def test_local_reviewer_reports_read_errors_and_bad_run_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_review_bad_records"
            run_dir.mkdir()
            (run_dir / "artifact_index.jsonl").write_text("{bad\n", encoding="utf-8")
            (run_dir / "trajectory.jsonl").write_text("{bad\n", encoding="utf-8")
            (run_dir / "run_summary.json").write_text("{bad\n", encoding="utf-8")

            result = run_local_artifact_review(
                run_dir=run_dir,
                run_id=run_dir.name,
                stage_ref="workflow_stage:reviewer_stub",
                source_event_ids=["event:source"],
            )

            payload = json.loads(
                (run_dir / result["output_artifacts"][0]["payload_path"]).read_text(
                    encoding="utf-8"
                )
            )
            check_ids = {finding["check_id"] for finding in payload["findings"]}
            self.assertIn("artifact_index_read", check_ids)
            self.assertIn("trajectory_read", check_ids)
            self.assertIn("run_summary_unreadable", check_ids)

    def test_local_reviewer_refuses_to_append_after_run_finalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_review_finalized"
            run_dir.mkdir()
            trajectory = TrajectoryWriter(run_dir, run_dir.name)
            trajectory.emit("run.finalized", payload={"status": "succeeded"})
            before = (run_dir / "trajectory.jsonl").read_text(encoding="utf-8")
            self.assertFalse((run_dir / "artifact_index.jsonl").exists())

            result = asyncio.run(
                optional_workflow_stage("reviewer").execute(
                    OptionalWorkflowStageContext(
                        run_dir=run_dir,
                        run_id=run_dir.name,
                        enabled=True,
                        mode="local",
                    )
                )
            )

            after = (run_dir / "trajectory.jsonl").read_text(encoding="utf-8")
            self.assertFalse(result.success)
            self.assertEqual(result.summary["write_effect"], "none")
            self.assertEqual(before, after)
            self.assertFalse((run_dir / "artifact_index.jsonl").exists())
            self.assertFalse((run_dir / "artifacts" / "by_id").exists())
            self.assertFalse((run_dir / "workflow" / "reviewer_report.json").exists())

    def test_direct_local_reviewer_refuses_to_append_after_run_finalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_review_direct_finalized"
            run_dir.mkdir()
            trajectory = TrajectoryWriter(run_dir, run_dir.name)
            trajectory.emit("run.finalized", payload={"status": "succeeded"})
            before = (run_dir / "trajectory.jsonl").read_text(encoding="utf-8")

            result = run_local_artifact_review(
                run_dir=run_dir,
                run_id=run_dir.name,
                stage_ref="workflow_stage:reviewer_stub",
            )

            self.assertFalse(result["success"])
            self.assertEqual(result["summary"]["write_effect"], "none")
            self.assertEqual(before, (run_dir / "trajectory.jsonl").read_text(encoding="utf-8"))
            self.assertFalse((run_dir / "artifact_index.jsonl").exists())
            self.assertFalse((run_dir / "artifacts" / "by_id").exists())

    def test_reviewer_contract_matches_local_implementation_modes_and_outputs(self) -> None:
        contract = OPTIONAL_STAGE_CONTRACTS["reviewer"]
        outputs = set(contract["outputs"])
        self.assertEqual(outputs, {"review_artifact", "reviewer_findings"})
        self.assertNotIn("review_score", outputs)

        import yaml

        plugin = yaml.safe_load(
            Path("praxist/plugins/workflow_stages/reviewer_stub/plugin.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(plugin["contracts"]["outputs"]), outputs)
        self.assertEqual(
            set(plugin["contracts"]["execution_modes"]),
            {"local", "artifact", "artifacts", "run_artifact", "claim_check", "review"},
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
