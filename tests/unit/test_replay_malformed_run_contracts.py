from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class ReplayMalformedRunContractsTest(unittest.TestCase):
    def test_verify_run_reports_many_independent_replay_faults_without_abort(self) -> None:
        from praxist.core import replay
        from praxist.core.storage import sha256_bytes

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            payload = run_dir / "artifacts" / "by_id" / "a1" / "payload.txt"
            payload.parent.mkdir(parents=True)
            payload.write_text("payload", encoding="utf-8")
            metadata = {
                "artifact_id": "a1",
                "artifact_type": "generic",
                "payload_path": "artifacts/wrong/payload.txt",
                "content_hash": sha256_bytes(payload.read_bytes()),
            }
            _write_json(payload.parent / "metadata.json", {"artifact_id": "different"})
            (payload.parent / "extra.txt").write_text("extra", encoding="utf-8")

            _write_json(
                run_dir / "run.json",
                {
                    "run_id": 123,
                    "status": "running",
                    "task_ref": "task:external",
                    "workflow_ref": "workflow_stage:research_loop",
                    "workspace_hash": "",
                    "source_hash_algorithm": "md5",
                    "source_file_count": 0,
                },
            )
            _write_json(
                run_dir / "run_summary.json",
                {
                    "run_id": "other",
                    "status": "succeeded",
                    "generations_completed": 1,
                    "frontier_records": 2,
                    "finding_summary": {"drafts": "bad", "accepted": 9},
                    "output_hashes": {"unexpected.jsonl": "sha256:" + "0" * 64},
                },
            )
            _write_json(
                run_dir / "startup_config.json",
                {
                    "canonical_args": {
                        "runtime": "agent_runtime:claude_sdk",
                        "model_provider": "model_provider:openrouter",
                        "budget_policy": "budget_policy:missing",
                    }
                },
            )
            _write_json(
                run_dir / "plugin_resolution.json",
                {
                    "algorithm_version": 1,
                    "run_id": "wrong",
                    "selected": [
                        {
                            "metadata": {
                                "kind": "agent_runtime",
                                "name": "claude_sdk",
                                "version": "0.1.0",
                                "protocol_version": 1,
                                "stability": "v1_stable",
                                "dependencies": [{"kind": "model_provider", "name": "openrouter"}],
                                "capabilities": [],
                                "code": [],
                                "assets": [],
                            },
                            "source": "project",
                            "path": "builtin://static/agent_runtime/claude_sdk",
                            "content_hash": "bad",
                            "selected_by": ["test"],
                        },
                        "bad",
                    ],
                    "dependency_edges": [
                        {"from": "agent_runtime:claude_sdk", "to": "model_provider:missing"}
                    ],
                },
            )
            _write_json(
                run_dir / "model_profiles.json",
                {
                    "runtime_ref": "agent_runtime:other",
                    "provider_adapters": {"model_provider:other": {}},
                    "profiles": {
                        "cheap": {
                            "provider_ref": "model_provider:other",
                            "model": "bad/model",
                        },
                        "bad": "not-object",
                    },
                    "runtime_provider_conformance": {
                        "runtime_ref": "agent_runtime:wrong",
                        "model_provider_ref": "model_provider:wrong",
                        "cache_mode": "wrong",
                    },
                },
            )
            _write_json(run_dir / "credentials_redacted.json", {"credential_profiles": "bad"})
            _write_json(
                run_dir / "cache_policy.json",
                {
                    "mode": "wrong",
                    "runtime_cache_strategy": "wrong",
                    "provider_cache_strategy": "wrong",
                },
            )
            _write_jsonl(
                run_dir / "trajectory.jsonl",
                [
                    {
                        "seq": 2,
                        "event_id": "evt_bad",
                        "kind": "workflow.stage_succeeded",
                        "scope": {"stage_id": "research_loop"},
                        "payload": {
                            "findings": 9,
                            "frontier_records": 9,
                            "research_memory_records": 9,
                            "graph_edges": 9,
                            "result": {"frontier_summary": [1, 2, 3]},
                        },
                        "parent_event_ids": "bad",
                    },
                    {
                        "seq": 2,
                        "event_id": "evt_bad",
                        "kind": "agent.run_finished",
                        "actor": {"type": "agent_runtime", "id": "agent_runtime:wrong"},
                        "payload": {"budget_grant_id": ""},
                    },
                    {
                        "seq": 3,
                        "event_id": "evt_000003",
                        "kind": "run.finalized",
                        "payload": {"run_id": "final", "status": "failed", "output_hashes": {}},
                    },
                    {"seq": 4, "event_id": "evt_000004", "kind": "after.finalized"},
                ],
            )
            _write_jsonl(
                run_dir / "artifact_index.jsonl",
                [
                    metadata,
                    metadata,
                    {"artifact_id": "", "payload_path": "../escape", "content_hash": "bad"},
                ],
            )
            request_record = {
                "request_id": "req",
                "requester_id": "peer",
                "experiment_id": "exp",
                "requested": {"tokens": "bad", "bad_unit": 1},
            }
            _write_jsonl(
                run_dir / "budget_ledger.jsonl",
                [
                    {
                        "kind": "request",
                        "record_id": "r1",
                        "grant_id": "g1",
                        "requested_budget": [],
                        "request_record": request_record,
                        "source_event_ids": ["missing"],
                    },
                    {
                        "kind": "decision",
                        "record_id": "d1",
                        "grant_id": "g1",
                        "requested_budget": {"tokens": 1},
                        "granted_budget": {"tokens": 1},
                        "request_record": {**request_record, "requested": {"tokens": 1}},
                        "decision_record": {"decision": "grant"},
                    },
                    {
                        "kind": "usage",
                        "record_id": "u1",
                        "grant_id": "missing",
                        "actual_usage": {"tokens": -1},
                    },
                    {
                        "kind": "usage_unknown",
                        "record_id": "u2",
                        "grant_id": "g1",
                        "unknown_units": "bad",
                    },
                ],
            )
            _write_jsonl(
                run_dir / "findings" / "findings.jsonl",
                [
                    {
                        "run_id": "wrong",
                        "finding_id": "f1",
                        "status": "draft",
                        "source_event_ids": ["missing"],
                        "artifact_refs": [{"artifact_id": "missing"}],
                    }
                ],
            )
            _write_jsonl(
                run_dir / "findings" / "frontier.jsonl",
                [{"finding_id": "missing", "source_event_ids": "bad"}],
            )
            _write_jsonl(run_dir / "memory" / "research_memory.jsonl", [{"run_id": "wrong"}])
            _write_jsonl(run_dir / "memory" / "graph_edges.jsonl", [{"graph_edge_id": "edge"}])
            (run_dir / "logs").mkdir()
            (run_dir / "logs" / "plain.log").write_text("ok", encoding="utf-8")
            (run_dir / "shared_findings").mkdir()
            (run_dir / "shared_findings" / "shared.json").write_text(
                json.dumps({"id": "shared-only"}),
                encoding="utf-8",
            )
            (run_dir / "frontier").mkdir()
            (run_dir / "frontier" / "frontier_manifest.json").write_text(
                json.dumps({"generations": {"0": [{"finding_id": "frontier-only"}]}}),
                encoding="utf-8",
            )
            db = run_dir / "shared_store.db"
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE findings (id TEXT)")
            conn.execute("INSERT INTO findings VALUES ('sqlite-only')")
            conn.execute("CREATE TABLE finding_edges (edge_id TEXT)")
            conn.execute("INSERT INTO finding_edges VALUES ('sqlite-edge')")
            conn.commit()
            conn.close()

            with patch(
                "praxist.core.replay.build_core_source_snapshot",
                return_value={"workspace_hash": "current"},
            ):
                report = replay.verify_run(run_dir, locked=False)

            self.assertFalse(report["success"])
            errors = "\n".join(report["errors"])
            warnings = "\n".join(report["warnings"])
            self.assertIn("run.json missing required run_id", errors)
            self.assertIn("trajectory duplicate event_id", errors)
            self.assertIn("trajectory has events after run.finalized", errors)
            self.assertIn("plugin_resolution algorithm_version", errors)
            self.assertIn("credentials_redacted credential_profiles is not a list", errors)
            self.assertIn("artifact duplicate artifact_id", errors)
            self.assertIn("state recovery: canonical findings missing", errors)
            self.assertIn("shared_findings has ids not present in SQLite", warnings)
            self.assertTrue((run_dir / "replay" / "replay_report.json").exists())


if __name__ == "__main__":
    unittest.main()
