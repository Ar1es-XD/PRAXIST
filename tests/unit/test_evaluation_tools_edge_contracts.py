from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _payload(response: dict) -> dict:
    return json.loads(response["content"][0]["text"])


class EvaluationToolsEdgeContractsTest(unittest.TestCase):
    def test_log_metrics_uses_run_env_when_run_id_arg_is_missing(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_2026-06-28_metrics"
            run_dir.mkdir()
            env = {
                "PRAXIST_RUN_DIR": str(run_dir),
                "PRAXIST_RUN_ID": run_dir.name,
                "LOCAL_MODE": "true",
                "GENERATION_ID": "bad",
            }
            with patch.dict(os.environ, env, clear=True):
                result = asyncio.run(
                    adapter._handle_log_experiment_metrics(
                        {
                            "variant_name": "candidate_a",
                            "metrics": {"score": 1.25},
                            "peer_id": "gen4_peer3",
                        }
                    )
                )

            payload = _payload(result)
            self.assertEqual(payload["status"], "recorded")
            self.assertEqual(payload["run_id"], run_dir.name)
            metrics_log = run_dir / "logs" / "metrics_log.jsonl"
            self.assertTrue(metrics_log.exists())
            record = json.loads(metrics_log.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["run_id"], run_dir.name)
            self.assertEqual(record["generation_id"], 4)
            self.assertTrue((run_dir / "shared_store.db").exists())

    def test_log_metrics_returns_structured_error_without_run_context(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter

        with patch.dict(os.environ, {}, clear=True):
            result = asyncio.run(
                adapter._handle_log_experiment_metrics(
                    {"variant_name": "candidate_a", "metrics": {"score": 1.25}}
                )
            )

        self.assertTrue(result["is_error"])
        self.assertIn("run_id", _payload(result)["error"])

    def test_wait_for_file_helper_edges_and_large_file_scan(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter

        self.assertEqual(adapter._gen_id_from_peer_id(" Gen12_Peer3 "), 12)
        self.assertIsNone(adapter._gen_id_from_peer_id(""))
        self.assertIsNone(adapter._gen_id_from_peer_id("peer3"))
        with patch.object(
            adapter,
            "_PEER_ID_GEN_RE",
            SimpleNamespace(match=lambda _value: SimpleNamespace(group=lambda _idx: "bad")),
        ):
            self.assertIsNone(adapter._gen_id_from_peer_id("genx_peer0"))
        with patch.dict(os.environ, {"PEER_ID": "", "GENERATION_ID": "bad"}, clear=False):
            self.assertIsNone(adapter._gen_id_from_wait_context({}))
        self.assertEqual(adapter._gen_id_from_wait_context({"generation_id": "5"}), 5)
        with patch.dict(
            os.environ,
            {"PEER_ID": "gen9_peer1", "GENERATION_ID": "9"},
            clear=False,
        ):
            self.assertEqual(
                adapter._gen_id_from_wait_context(
                    {},
                    ["/tmp/run_demo/peer_workspaces/gen3_peer8/results/summary.json"],
                ),
                3,
            )
            self.assertIsNone(
                adapter._gen_id_from_wait_paths(
                    ["/tmp/run_demo/gen_2/a.json", "/tmp/run_demo/gen_3/b.json"]
                )
            )

        with patch.dict(os.environ, {"PRAXIST_WAIT_FOR_FILE_MAX_SECONDS": "bad"}, clear=False):
            self.assertEqual(adapter._wait_for_file_hard_cap_seconds(), 18 * 3600)
        with patch.dict(os.environ, {"PRAXIST_WAIT_FOR_FILE_MAX_SECONDS": "1"}, clear=False):
            self.assertEqual(adapter._wait_for_file_hard_cap_seconds(), 60)
        with patch.dict(os.environ, {"PRAXIST_WAIT_FOR_FILE_MAX_SECONDS": "999999"}, clear=False):
            self.assertEqual(adapter._wait_for_file_hard_cap_seconds(), 24 * 3600)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_2026-06-18_eval"
            run_dir.mkdir()
            (run_dir / "run.json").write_text("{}", encoding="utf-8")
            target = run_dir / "artifact.bin"
            target.write_bytes(b"A" * (4 * 1024 * 1024 + 10) + b"TAIL_MARKER")
            middle_target = run_dir / "artifact_middle.bin"
            middle_target.write_bytes(
                b"A" * (2 * 1024 * 1024) + b"MIDDLE_MARKER" + b"B" * (3 * 1024 * 1024)
            )

            env = {
                "LOCAL_STORE_DIR": str(root),
                "PRAXIST_WAIT_FOR_FILE_MAX_SECONDS": "120",
                "PEER_ID": "gen2_peer1",
            }
            with patch.dict(os.environ, env, clear=False):
                ready = asyncio.run(
                    adapter._handle_wait_for_file_impl(
                        {
                            "path": f"{target},{target}",
                            "timeout_seconds": "999999",
                            "poll_interval_seconds": "bad",
                            "min_bytes": "bad",
                            "contains_text": "TAIL_MARKER",
                            "mode": "invalid",
                        }
                    )
                )
                payload = _payload(ready)
                self.assertEqual(payload["status"], "ready")
                self.assertTrue(payload["timeout_clamped_to_hard_cap"])
                self.assertEqual(payload["deduped_count"], 1)

                middle_ready = asyncio.run(
                    adapter._handle_wait_for_file_impl(
                        {
                            "path": str(middle_target),
                            "timeout_seconds": 5,
                            "contains_text": "MIDDLE_MARKER",
                        }
                    )
                )
                self.assertEqual(_payload(middle_ready)["status"], "ready")

                too_long = asyncio.run(
                    adapter._handle_wait_for_file_impl(
                        {"path": str(target), "contains_text": "x" * 1025}
                    )
                )
                self.assertTrue(too_long["is_error"])

                bad_timeout = asyncio.run(
                    adapter._handle_wait_for_file_impl({"path": str(target), "timeout_seconds": 0})
                )
                self.assertTrue(bad_timeout["is_error"])

                bad_path = asyncio.run(
                    adapter._handle_wait_for_file_impl({"path": f"{target}\nboom"})
                )
                self.assertTrue(bad_path["is_error"])

    def test_wait_for_file_accepts_future_symlink_to_allowed_regular_file(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter

        async def _run() -> dict:
            with tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp) / "run_2026-07-03_symlink"
                result_dir = run_dir / "results"
                result_dir.mkdir(parents=True)
                (run_dir / "run.json").write_text("{}", encoding="utf-8")
                target = result_dir / "real_summary.json"
                link = result_dir / "linked_summary.json"

                async def publish() -> None:
                    await asyncio.sleep(0.05)
                    target.write_text('{"test_equity": [1, 2, 3]}', encoding="utf-8")
                    link.symlink_to(target)

                with patch.dict(
                    os.environ,
                    {"PRAXIST_RUN_DIR": str(run_dir), "PEER_ID": "gen0_peer0"},
                    clear=False,
                ):
                    task = asyncio.create_task(publish())
                    try:
                        return await adapter._handle_wait_for_file_impl(
                            {
                                "path": str(link),
                                "timeout_seconds": 5,
                                "poll_interval_seconds": 2,
                                "contains_text": "test_equity",
                            }
                        )
                    finally:
                        await task

        payload = _payload(asyncio.run(_run()))
        self.assertEqual(payload["status"], "ready")

    def test_wait_for_file_returns_ready_before_stop_signal(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_2026-07-03_ready_stop"
            gen_dir = run_dir / "gen_3"
            result_dir = run_dir / "results"
            result_dir.mkdir(parents=True)
            gen_dir.mkdir(parents=True)
            (run_dir / "run.json").write_text("{}", encoding="utf-8")
            target = result_dir / "summary.json"
            target.write_text('{"test_equity": true}', encoding="utf-8")
            (gen_dir / "STOP_SIGNAL").write_text("stop", encoding="utf-8")

            with patch.dict(
                os.environ,
                {"PRAXIST_RUN_DIR": str(run_dir), "PEER_ID": "gen3_peer0"},
                clear=False,
            ):
                result = asyncio.run(
                    adapter._handle_wait_for_file_impl(
                        {
                            "path": str(target),
                            "timeout_seconds": 5,
                            "contains_text": "test_equity",
                        }
                    )
                )

        self.assertEqual(_payload(result)["status"], "ready")

    def test_wait_for_file_defers_empty_runtime_task_output_to_task_notification(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_2026-07-13_runtime_task"
            task_dir = (
                run_dir
                / "peer_workspaces"
                / "gen0_peer8"
                / "tmp"
                / "claude-0"
                / "project"
                / "d43e9661-47d3-4776-b2df-f0e8cfb1d171"
                / "tasks"
            )
            task_dir.mkdir(parents=True)
            (run_dir / "run.json").write_text("{}", encoding="utf-8")
            output_path = task_dir / "btot585gw.output"
            output_path.write_bytes(b"")

            started = time.monotonic()
            result = asyncio.run(
                adapter._handle_wait_for_file_impl(
                    {
                        "path": str(output_path),
                        "timeout_seconds": 20_000,
                        "poll_interval_seconds": 1_800,
                        "min_bytes": 1,
                    }
                )
            )

        payload = _payload(result)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(payload["status"], "runtime_task_notification_required")
        self.assertFalse(payload["completion_inferred"])
        self.assertEqual(payload["task_ids"], ["btot585gw"])
        self.assertIn("Do not call wait_for_file again", payload["hint"])

    def test_wait_for_file_releases_passive_wait_when_generation_closes(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_2026-07-13_closing_wait"
            result_dir = run_dir / "peer_workspaces" / "gen0_peer8" / "results"
            gen_dir = run_dir / "gen_0"
            result_dir.mkdir(parents=True)
            gen_dir.mkdir(parents=True)
            (run_dir / "run.json").write_text("{}", encoding="utf-8")
            target = result_dir / "summary.json"
            target.write_bytes(b"")
            (gen_dir / "CLOSING_SIGNAL").write_text("gen_id=0\n", encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "PRAXIST_RUN_DIR": str(run_dir),
                    # In-process MCP handlers cannot trust a child runtime's
                    # mutable/global peer environment. Stable path identity wins.
                    "PEER_ID": "gen9_peer1",
                    "GENERATION_ID": "9",
                },
                clear=False,
            ):
                result = asyncio.run(
                    adapter._handle_wait_for_file_impl(
                        {
                            "path": str(target),
                            "timeout_seconds": 20_000,
                            "poll_interval_seconds": 1_800,
                            "min_bytes": 1,
                        }
                    )
                )

        payload = _payload(result)
        self.assertEqual(payload["status"], "released_for_generation_closing")
        self.assertFalse(payload["completion_inferred"])
        self.assertIn("No evaluator or background process was stopped", payload["hint"])

    def test_wait_for_file_clamps_to_canonical_generation_deadline(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_2026-07-13_deadline"
            target = run_dir / "results" / "candidate" / "ready.json"
            target.parent.mkdir(parents=True)
            target.write_text('{"phase":"finished"}', encoding="utf-8")
            (run_dir / "run.json").write_text("{}", encoding="utf-8")
            scheduler_dir = run_dir / "resource_scheduler"
            scheduler_dir.mkdir()
            (scheduler_dir / "status.json").write_text(
                json.dumps({"generation_deadlines": {"0": time.time() + 30}}),
                encoding="utf-8",
            )
            env = {"PRAXIST_RUN_DIR": str(run_dir), "PEER_ID": "gen0_peer0"}
            with patch.dict(os.environ, env, clear=False):
                result = asyncio.run(
                    adapter._handle_wait_for_file_impl(
                        {"path": str(target), "timeout_seconds": 20_000}
                    )
                )

        payload = _payload(result)
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["timeout_clamped_to_generation_deadline"])
        self.assertLessEqual(payload["generation_budget_seconds"], 30)

    def test_wait_for_file_checks_ready_evidence_before_elapsed_generation_deadline(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_2026-07-13_elapsed_deadline"
            result_dir = run_dir / "results" / "candidate"
            result_dir.mkdir(parents=True)
            (run_dir / "run.json").write_text("{}", encoding="utf-8")
            scheduler_dir = run_dir / "resource_scheduler"
            scheduler_dir.mkdir()
            (scheduler_dir / "status.json").write_text(
                json.dumps({"generation_deadlines": {"0": time.time() - 1}}),
                encoding="utf-8",
            )
            ready = result_dir / "ready.json"
            ready.write_text('{"phase":"finished"}', encoding="utf-8")
            missing = result_dir / "missing.json"
            env = {"PRAXIST_RUN_DIR": str(run_dir), "PEER_ID": "gen0_peer0"}
            with patch.dict(os.environ, env, clear=False):
                ready_result = asyncio.run(
                    adapter._handle_wait_for_file_impl(
                        {"path": str(ready), "timeout_seconds": 20_000}
                    )
                )
                missing_result = asyncio.run(
                    adapter._handle_wait_for_file_impl(
                        {"path": str(missing), "timeout_seconds": 20_000}
                    )
                )

        self.assertEqual(_payload(ready_result)["status"], "ready")
        self.assertEqual(
            _payload(missing_result)["status"],
            "generation_deadline_elapsed",
        )

    def test_wait_for_file_infers_external_run_dirs_for_stop_signal(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "task" / "experiments" / "run_2026-05-13_demo"
            target = run_dir / "results" / "variant" / "done.json"
            target.parent.mkdir(parents=True)
            (run_dir / "run.json").write_text("{}", encoding="utf-8")

            candidates = adapter._candidate_run_dirs_for_wait_paths([str(target)])

        self.assertEqual(candidates, [run_dir.resolve()])

    def test_wait_for_file_ignores_stale_generation_stop_signal(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "task" / "experiments" / "run_2026-05-31_demo"
            target = run_dir / "results" / "variant" / "done.json"
            target.parent.mkdir(parents=True)
            (run_dir / "run.json").write_text("{}", encoding="utf-8")
            stale = run_dir / "gen_0" / "STOP_SIGNAL"
            stale.parent.mkdir()
            stale.write_text("gen_id=0\n", encoding="utf-8")

            env = {
                "PEER_ID": "gen1_peer3",
                "GENERATION_ID": "1",
                "LOCAL_STORE_DIR": str(root),
            }
            with patch.dict(os.environ, env, clear=False):
                result = asyncio.run(
                    adapter._handle_wait_for_file_impl(
                        {
                            "path": str(target),
                            "timeout_seconds": 1,
                            "poll_interval_seconds": 2,
                        }
                    )
                )

        self.assertEqual(_payload(result)["status"], "timeout")

    def test_wait_for_file_honors_current_generation_stop_signal(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "task" / "experiments" / "run_2026-05-31_demo"
            target = run_dir / "results" / "variant" / "done.json"
            target.parent.mkdir(parents=True)
            (run_dir / "run.json").write_text("{}", encoding="utf-8")
            current = run_dir / "gen_1" / "STOP_SIGNAL"
            current.parent.mkdir()
            current.write_text("gen_id=1\n", encoding="utf-8")

            env = {
                "PEER_ID": "gen1_peer3",
                "GENERATION_ID": "1",
                "LOCAL_STORE_DIR": str(root),
            }
            with patch.dict(os.environ, env, clear=False):
                result = asyncio.run(
                    adapter._handle_wait_for_file_impl(
                        {
                            "path": str(target),
                            "timeout_seconds": 30,
                            "poll_interval_seconds": 2,
                        }
                    )
                )

        payload = _payload(result)
        self.assertEqual(payload["status"], "aborted_by_stop_signal")
        self.assertEqual(payload["generation_id"], 1)

    def test_wait_for_file_honors_run_level_shutdown_across_generations(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "task" / "experiments" / "run_2026-05-31_demo"
            target = run_dir / "results" / "variant" / "done.json"
            target.parent.mkdir(parents=True)
            (run_dir / "run.json").write_text("{}", encoding="utf-8")
            (run_dir / "ORCHESTRATOR_SHUTDOWN").write_text("shutdown\n", encoding="utf-8")

            env = {
                "PEER_ID": "gen7_peer2",
                "GENERATION_ID": "7",
                "LOCAL_STORE_DIR": str(root),
            }
            with patch.dict(os.environ, env, clear=False):
                result = asyncio.run(
                    adapter._handle_wait_for_file_impl(
                        {
                            "path": str(target),
                            "timeout_seconds": 30,
                            "poll_interval_seconds": 2,
                        }
                    )
                )

        payload = _payload(result)
        self.assertEqual(payload["status"], "aborted_by_stop_signal")
        self.assertEqual(payload["generation_id"], 7)

    def test_wait_wrapper_and_share_finding_degrade_without_losing_result(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter

        statuses: list[str] = []

        class FakeGuard:
            def start(self):
                return None

            def finish(self, *, status: str, **_kwargs):
                statuses.append(status)

        with (
            patch.object(adapter.BudgetedActionGuard, "from_env", return_value=FakeGuard()),
            patch.object(adapter, "_handle_wait_for_file_impl", return_value={"is_error": True}),
        ):
            self.assertTrue(asyncio.run(adapter._handle_wait_for_file({"path": "x"}))["is_error"])
        self.assertEqual(statuses[-1], "failed")

        with (
            patch.object(adapter.BudgetedActionGuard, "from_env", return_value=FakeGuard()),
            patch.object(adapter, "_handle_wait_for_file_impl", side_effect=RuntimeError("boom")),
            self.assertRaises(RuntimeError),
        ):
            asyncio.run(adapter._handle_wait_for_file({"path": "x"}))
        self.assertEqual(statuses[-1], "failed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agenda_dir = root / "agendas"
            agenda_dir.mkdir()
            (agenda_dir / "research_agenda_gen0.yaml").write_text("agenda: []", encoding="utf-8")
            env = {
                "LOCAL_MODE": "false",
                "LOCAL_STORE_DIR": str(root),
                "LOCAL_FINDINGS_DIR": str(root / "findings"),
                "GENERATION_ID": "0",
            }
            with (
                patch.dict(os.environ, env, clear=False),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.tools.findings_sync.save_finding_to_dir",
                    side_effect=RuntimeError("disk"),
                ),
                patch.object(adapter, "get_server_url", return_value="http://server"),
                patch.object(adapter, "async_http_post", side_effect=RuntimeError("offline")),
            ):
                result = asyncio.run(
                    adapter._handle_share_finding(
                        {
                            "finding_type": "result",
                            "title": "kept",
                            "content": "content",
                            "metrics": {"score": 1.0},
                            "peer_id": "gen0_peer0",
                            "links": [{"target_finding_id": "a"}],
                            "extra": {},
                        }
                    )
                )
            self.assertEqual(_payload(result)["status"], "shared")

    def test_share_finding_accepts_challenge_negative_evidence(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.card_builder import (
            build_card_from_finding,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "LOCAL_MODE": "true",
                "LOCAL_STORE_DIR": str(root),
                "LOCAL_FINDINGS_DIR": str(root / "shared_findings"),
                "GENERATION_ID": "0",
            }
            with patch.dict(os.environ, env, clear=False):
                result = asyncio.run(
                    adapter._handle_share_finding(
                        {
                            "finding_type": "challenge",
                            "title": "falsifier failed parent",
                            "content": "negative evidence: parent did not reproduce",
                            "metrics": {"score": -1.0, "promotion_eligible": False},
                            "variant_name": "parent_falsifier",
                            "peer_id": "gen0_peer7",
                            "extra": {"is_negative": True, "peer_role": "falsifier"},
                        }
                    )
                )
                self.assertEqual(_payload(result)["status"], "shared")
                findings = local_store.get_findings(finding_type="challenge")

            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0]["variant_name"], "parent_falsifier")
            card = build_card_from_finding(findings[0], root)
            self.assertTrue(card["quality"]["is_negative"])

    def test_leaderboard_env_server_and_filesystem_fallback_shapes_are_stable(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter

        with patch.dict(os.environ, {"ANCHOR_METRICS": "{bad"}, clear=False):
            self.assertEqual(adapter._parse_anchor_metrics_env(), [])
        with patch.dict(os.environ, {"ANCHOR_METRICS": json.dumps({"name": "gap"})}, clear=False):
            self.assertEqual(adapter._parse_anchor_metrics_env(), [])
        with patch.dict(
            os.environ,
            {
                "ANCHOR_METRICS": json.dumps(
                    [
                        "bad",
                        {"name": ""},
                        {"name": "bad_direction", "direction": "sideways"},
                        {"name": "gap", "direction": "minimize"},
                    ]
                )
            },
            clear=False,
        ):
            self.assertEqual(
                adapter._parse_anchor_metrics_env(), [{"name": "gap", "direction": "minimize"}]
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            findings_dir = root / "findings"
            with patch.dict(os.environ, {"LOCAL_FINDINGS_DIR": str(findings_dir)}, clear=False):
                missing = json.loads(adapter._filesystem_leaderboard(0, 3))
            self.assertEqual(missing["note"], "No findings directory found")

            findings_dir.mkdir()
            rows = {
                "bad.json": "{bad",
                "note.json": {"finding_type": "note", "metrics": {"score": 9}, "generation_id": 0},
                "wrong_gen.json": {
                    "finding_type": "result",
                    "variant_name": "wrong",
                    "metrics": {"score": 9, "tier": "T3"},
                    "generation_id": 1,
                },
                "tier2.json": {
                    "finding_type": "result",
                    "variant_name": "tier2",
                    "metrics": {"score": 8, "tier": "T2"},
                    "generation_id": 0,
                },
                "missing_metric.json": {
                    "finding_type": "insight",
                    "variant_name": "missing",
                    "metrics": {"tier": "T3"},
                    "details": {"promotion_eligible": True},
                    "generation_id": 0,
                },
                "best.json": {
                    "finding_type": "result",
                    "variant_name": "best",
                    "title": "Best",
                    "metrics": {"score": 0.9, "tier": "T3"},
                    "promotion_eligible": "yes",
                    "generation_id": 0,
                    "peer_id": "gen0_peer0",
                },
            }
            for name, payload in rows.items():
                path = findings_dir / name
                path.write_text(
                    payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
                )
            with patch.dict(
                os.environ,
                {
                    "LOCAL_FINDINGS_DIR": str(findings_dir),
                    "PRIMARY_METRIC": "score",
                    "METRIC_DIRECTION": "maximize",
                },
                clear=False,
            ):
                fallback = json.loads(adapter._filesystem_leaderboard(0, 5))
            self.assertEqual(
                [row["variant_name"] for row in fallback["entries"]],
                ["tier2", "best", "missing"],
            )
            self.assertTrue(fallback["degraded_filtering"])

        async def server_payload(result):
            return _payload(await adapter._handle_get_leaderboard({"top_k": "2"}))

        with (
            patch.dict(os.environ, {"LOCAL_MODE": "false", "ANCHOR_METRICS": ""}, clear=True),
            patch.object(adapter, "get_server_url", return_value="http://server"),
            patch.object(adapter, "async_http_get", return_value=[{"variant_name": "v"}]),
        ):
            self.assertEqual(asyncio.run(server_payload(None))["mode"], "server_legacy")

        with (
            patch.dict(os.environ, {"LOCAL_MODE": "false", "ANCHOR_METRICS": ""}, clear=True),
            patch.object(adapter, "get_server_url", return_value="http://server"),
            patch.object(adapter, "async_http_get", return_value={"entries": []}),
        ):
            self.assertEqual(asyncio.run(server_payload(None))["mode"], "server_legacy")

        with (
            patch.dict(
                os.environ,
                {
                    "LOCAL_MODE": "false",
                    "ANCHOR_METRICS": json.dumps([{"name": "gap", "direction": "minimize"}]),
                },
                clear=True,
            ),
            patch.object(
                adapter,
                "_sqlite_leaderboard",
                side_effect=[RuntimeError("sqlite"), json.dumps({"mode": "fallback"})],
            ),
            patch.object(adapter, "get_server_url", return_value="http://server"),
            patch.object(adapter, "async_http_get", return_value={"error": "bad"}),
        ):
            self.assertEqual(asyncio.run(server_payload(None))["mode"], "fallback")

        self.assertIn("tool_server_ref", adapter.create_tool_plugin())
        with (
            patch.object(adapter, "create_sdk_mcp_server", None),
            patch.object(adapter, "tool", None),
            self.assertRaises(ImportError),
        ):
            adapter.create_evaluation_tools_server()

    def test_evaluation_tool_remaining_error_edges_are_stable(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "ready.txt"
            target.write_text("READY", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "LOCAL_STORE_DIR": "/",
                    "LOCAL_FINDINGS_DIR": str(root),
                    "PEER_ID": "gen0_peer0",
                },
                clear=False,
            ):
                rejected = asyncio.run(
                    adapter._handle_wait_for_file_impl(
                        {
                            "path": f"{target}\tbad",
                            "timeout_seconds": 1,
                        }
                    )
                )
            self.assertTrue(rejected["is_error"])

            run_dir = root / "run_2026_eval"
            watched = run_dir / "outputs" / "missing.txt"
            watched.parent.mkdir(parents=True)
            (run_dir / "shared_store.db").write_text("", encoding="utf-8")
            self.assertEqual(
                adapter._candidate_run_dirs_for_wait_paths([str(watched)]),
                [run_dir.resolve()],
            )

            with (
                patch.object(adapter, "read_tool_result_ref", side_effect=ValueError("bad ref")),
                patch.dict(os.environ, {}, clear=False),
            ):
                bad_ref = asyncio.run(
                    adapter._handle_read_tool_result(
                        {"ref": "tool_result:bad", "offset": "bad", "max_chars": "bad"}
                    )
                )
            self.assertTrue(bad_ref["is_error"])
            missing_ref = asyncio.run(adapter._handle_read_tool_result({"ref": ""}))
            self.assertTrue(missing_ref["is_error"])

            agenda_dir = root / "agendas"
            agenda_dir.mkdir(exist_ok=True)
            (agenda_dir / "research_agenda_gen0.yaml").write_text("agenda: []", encoding="utf-8")
            with (
                patch.dict(
                    os.environ,
                    {
                        "LOCAL_MODE": "true",
                        "LOCAL_STORE_DIR": str(root),
                        "LOCAL_FINDINGS_DIR": str(root / "findings"),
                    },
                    clear=False,
                ),
                self.assertLogs(adapter.logger.name, level="WARNING") as logs,
            ):
                result = asyncio.run(
                    adapter._handle_share_finding(
                        {
                            "finding_type": "result",
                            "title": "title",
                            "content": "content",
                            "metrics": "{}",
                            "peer_id": "gen0_peer0",
                            "extra": "[1, 2]",
                        }
                    )
                )
            self.assertEqual(_payload(result)["status"], "shared")
            self.assertTrue(any("JSON-decode to dict" in item for item in logs.output))


if __name__ == "__main__":
    unittest.main()
