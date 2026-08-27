from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class S3UtilitiesContractsTest(unittest.TestCase):
    def test_s3_client_upload_download_and_snapshot_paths(self) -> None:
        from praxist import config
        from praxist.infrastructure import s3_utils

        class FakeClient:
            def __init__(self) -> None:
                self.uploads: list[tuple[str, str, str, dict[str, str]]] = []
                self.downloads: list[tuple[str, str, str]] = []
                self.fail_upload = False
                self.fail_download = False

            def upload_file(self, path, bucket, key, ExtraArgs=None):
                if self.fail_upload:
                    raise RuntimeError("upload boom")
                self.uploads.append((path, bucket, key, ExtraArgs or {}))

            def download_file(self, bucket, key, path):
                if self.fail_download:
                    raise RuntimeError("download boom")
                self.downloads.append((bucket, key, path))
                Path(path).write_text("downloaded", encoding="utf-8")

        fake_client = FakeClient()
        fake_boto3 = SimpleNamespace(client=lambda *args, **kwargs: fake_client)

        def fake_config(**kwargs):
            return {"config": kwargs}

        with patch.object(s3_utils, "HAS_BOTO3", False), self.assertRaises(ImportError):
            s3_utils.get_s3_client()

        with (
            patch.object(s3_utils, "HAS_BOTO3", True),
            patch.object(s3_utils, "boto3", fake_boto3, create=True),
            patch.object(s3_utils, "Config", fake_config, create=True),
            patch.object(config, "AWS_ACCESS_KEY_ID", "ak"),
            patch.object(config, "AWS_SECRET_ACCESS_KEY", "sk"),
            patch.object(config, "S3_ENDPOINT_URL", "http://s3"),
            patch.object(config, "S3_REGION", "us-test-1"),
            tempfile.TemporaryDirectory() as tmp,
        ):
            root = Path(tmp)
            self.assertIs(s3_utils.get_s3_client(), fake_client)
            file_path = root / "artifact.txt"
            file_path.write_text("payload", encoding="utf-8")
            self.assertTrue(
                s3_utils.upload_file_to_s3(file_path, "key.txt", "bucket", "text/plain")
            )
            self.assertEqual(fake_client.uploads[-1][3]["ContentType"], "text/plain")
            fake_client.fail_upload = True
            self.assertFalse(s3_utils.upload_file_to_s3(file_path, "bad", "bucket"))
            fake_client.fail_upload = False

            src_dir = root / "src"
            src_dir.mkdir()
            (src_dir / "keep.py").write_text("x", encoding="utf-8")
            (src_dir / "__pycache__").mkdir()
            (src_dir / "__pycache__" / "skip.pyc").write_text("cache", encoding="utf-8")
            self.assertTrue(s3_utils.upload_directory_to_s3(src_dir, "src.tar.gz", "bucket"))
            self.assertTrue(fake_client.uploads[-1][2].endswith("src.tar.gz"))

            target = root / "downloads" / "file.txt"
            self.assertTrue(s3_utils.download_s3_file("remote.txt", target, "bucket"))
            self.assertEqual(target.read_text(encoding="utf-8"), "downloaded")
            fake_client.fail_download = True
            self.assertFalse(s3_utils.download_s3_file("remote.txt", target, "bucket"))
            fake_client.fail_download = False

            safe_tar = root / "safe.tar.gz"
            with tarfile.open(safe_tar, "w:gz") as tar:
                member_file = root / "member.txt"
                member_file.write_text("inside", encoding="utf-8")
                tar.add(member_file, arcname="snapshot/member.txt")

            def fake_download(_key: str, local_path: Path, _bucket: str) -> bool:
                shutil.copyfile(safe_tar, local_path)
                return True

            with patch.object(s3_utils, "download_s3_file", fake_download):
                names = s3_utils.download_snapshot_from_s3("safe.tar.gz", root / "out", "bucket")
            self.assertEqual(names, ["snapshot/member.txt"])
            self.assertEqual((root / "out" / "snapshot" / "member.txt").read_text(), "inside")

            escape_tar = root / "escape.tar.gz"
            with tarfile.open(escape_tar, "w:gz") as tar:
                member_file = root / "escape.txt"
                member_file.write_text("escape", encoding="utf-8")
                tar.add(member_file, arcname="../escape.txt")

            def fake_escape(_key: str, local_path: Path, _bucket: str) -> bool:
                shutil.copyfile(escape_tar, local_path)
                return True

            with patch.object(s3_utils, "download_s3_file", fake_escape):
                self.assertEqual(
                    s3_utils.download_snapshot_from_s3("bad.tar.gz", root / "bad", "bucket"), []
                )
            self.assertIsInstance(s3_utils.generate_idea_uid(), str)

        with (
            patch.object(s3_utils, "HAS_BOTO3", True),
            patch.object(config, "AWS_ACCESS_KEY_ID", ""),
        ):
            self.assertIsNone(s3_utils.get_s3_client())


class AutonomousEntrypointContractsTest(unittest.TestCase):
    def test_launch_upload_and_main_paths_are_patchable(self) -> None:
        from praxist import config
        from praxist.infrastructure import execute_autonomous

        stream = io.StringIO()
        log = io.StringIO()
        tee = execute_autonomous.TeeOutput(stream, log)
        tee.write("hello")
        tee.flush()
        self.assertEqual(stream.getvalue(), "hello")
        self.assertEqual(log.getvalue(), "hello")

        class FakeLoop:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

            async def run(self):
                return {
                    "run_id": "run123",
                    "sessions": 1,
                    "tools": list(self.kwargs["mcp_servers"]),
                }

        build = SimpleNamespace(
            servers={"evaluation-tools": object()},
            unavailable=[{"server_name": "missing", "reason": "not installed"}],
        )
        with (
            patch(
                "praxist.core.tool_servers.build_legacy_mcp_servers",
                return_value=build,
            ) as build_servers,
            patch(
                "praxist.core.tool_servers.base_peer_allowed_tools",
                return_value=["Read", "Bash"],
            ),
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.AutonomousAgentLoop",
                FakeLoop,
            ),
            patch.dict(os.environ, {"PRAXIST_RUN_DIR": "/tmp/run"}, clear=False),
        ):
            launched = asyncio.run(
                execute_autonomous.launch_autonomous_loop(
                    "gen0_peer1",
                    0,
                    "prompt",
                    5,
                    local_mode=True,
                    model="fake-model",
                )
            )
        self.assertEqual(launched["sessions"], 1)
        self.assertIn("evaluation-tools", launched["tools"])
        build_servers.assert_called_once()

        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "logs"
            logs.mkdir()
            with (
                patch.dict(
                    os.environ,
                    {"LOGS_DIR": "", "S3_BUCKET": "", "S3_RESULTS_PREFIX": ""},
                    clear=False,
                ),
                patch.object(config, "LOGS_DIR", str(logs)),
                patch.object(config, "S3_BUCKET", ""),
            ):
                self.assertIsNone(
                    execute_autonomous.upload_final_artifacts("peer", "run", {"ok": True})
                )

            uploads: list[str] = []
            with (
                patch.dict(
                    os.environ,
                    {"LOGS_DIR": "", "S3_BUCKET": "", "S3_RESULTS_PREFIX": ""},
                    clear=False,
                ),
                patch.object(config, "LOGS_DIR", str(logs)),
                patch.object(config, "S3_BUCKET", "bucket"),
                patch.object(config, "S3_RESULTS_PREFIX", "prefix/"),
                patch(
                    "praxist.infrastructure.s3_utils.upload_file_to_s3",
                    side_effect=lambda _path, key, _bucket, _ctype: uploads.append(key) or True,
                ),
            ):
                execute_autonomous.upload_final_artifacts("peer", "run", {"ok": True})
            self.assertEqual(json.loads((logs / "run_result.json").read_text())["ok"], True)
            self.assertEqual(uploads, ["prefix/peer/run/run_result.json"])

        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "logs"
            prompt_file = Path(tmp) / "prompt.txt"
            prompt_file.write_text("from file", encoding="utf-8")
            captured: list[dict[str, object]] = []

            async def fake_launch(**kwargs):
                captured.append(kwargs)
                return {"run_id": "run456"}

            with (
                patch.dict(
                    os.environ,
                    {
                        "PEER_ID": "gen9_peer3",
                        "GENERATION_ID": "9",
                        "MAX_RUNTIME_SECONDS": "7",
                        "LOCAL_MODE": "yes",
                        "TASK_PROMPT_FILE": str(prompt_file),
                        "LOGS_DIR": str(logs),
                    },
                    clear=False,
                ),
                patch.object(execute_autonomous, "launch_autonomous_loop", fake_launch),
            ):
                execute_autonomous.main()
            self.assertEqual(captured[0]["peer_id"], "gen9_peer3")
            self.assertEqual(captured[0]["task_prompt"], "from file")

            with (
                patch.dict(os.environ, {"TASK_PROMPT": "", "TASK_PROMPT_FILE": ""}, clear=False),
                self.assertRaises(SystemExit) as cm,
            ):
                execute_autonomous.main()
            self.assertEqual(cm.exception.code, 1)


class OperationalCliContractsTest(unittest.TestCase):
    def test_gpu_governor_cli_and_finding_graph_cli_paths(self) -> None:
        from praxist.plugins.graph_maintainers.finding_graph_mvp import cli as graph_cli
        from praxist.plugins.workflow_stages.research_loop.backend import gpu_governor
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        def run_gpu_cli(argv: list[str]) -> tuple[int, str, str]:
            out = io.StringIO()
            err = io.StringIO()
            with (
                patch("sys.argv", ["gpu_governor", *argv]),
                contextlib.redirect_stdout(out),
                contextlib.redirect_stderr(err),
            ):
                try:
                    gpu_governor._cli()
                    code = 0
                except SystemExit as exc:
                    code = int(exc.code or 0)
            return code, out.getvalue(), err.getvalue()

        def load_gpu_payload(out: str) -> dict:
            payload = json.loads(out)
            serialized = json.dumps(payload, sort_keys=True)
            self.assertNotIn("next_step", serialized)
            self.assertNotIn("recommendation", serialized)
            self.assertNotIn("guidance", serialized)
            self.assertEqual(payload["schema_version"], "praxist.gpu_governor.cli.v1")
            return payload

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            own_pid = os.getpid()
            code, out, err = run_gpu_cli(
                [
                    "acquire",
                    "--gpu",
                    "0",
                    "--pid",
                    str(own_pid),
                    "--peer",
                    "test-peer",
                    "--tag",
                    "train-a",
                    "--max-per-gpu",
                    "1",
                    "--run-dir",
                    str(run_dir),
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual(err, "")
            payload = load_gpu_payload(out)
            self.assertEqual(payload["command"], "acquire")
            self.assertEqual(payload["status"], "acquired")
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["gpu_id"], 0)
            self.assertEqual(payload["pid"], own_pid)
            self.assertEqual(payload["peer"], "test-peer")
            self.assertEqual(payload["tag"], "train-a")
            self.assertEqual(payload["max_per_gpu"], 1)
            self.assertEqual(payload["occupied"], 1)
            self.assertEqual(payload["available"], 0)
            self.assertEqual(payload["current_slots"]["slots"][0]["pid"], own_pid)

            code, out, err = run_gpu_cli(
                [
                    "acquire",
                    "--gpu",
                    "0",
                    "--pid",
                    str(own_pid + 100000),
                    "--max-per-gpu",
                    "1",
                    "--run-dir",
                    str(run_dir),
                    "--non-blocking",
                ]
            )
            self.assertEqual(code, 1)
            self.assertEqual(err, "")
            payload = load_gpu_payload(out)
            self.assertEqual(payload["status"], "busy")
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["reason"], "capacity_reached")
            self.assertEqual(payload["occupied"], 1)
            self.assertEqual(payload["available"], 0)

            code, out, _err = run_gpu_cli(
                ["list", "--gpus", "1", "--max-per-gpu", "1", "--run-dir", str(run_dir)]
            )
            self.assertEqual(code, 0)
            payload = load_gpu_payload(out)
            self.assertEqual(payload["command"], "list")
            self.assertEqual(payload["status"], "listed")
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["occupied"], 1)
            self.assertEqual(payload["available"], 0)
            self.assertEqual(payload["gpus"]["0"]["slots"][0]["pid"], own_pid)

            code, out, err = run_gpu_cli(
                [
                    "release",
                    "--gpu",
                    "0",
                    "--pid",
                    str(own_pid),
                    "--max-per-gpu",
                    "1",
                    "--run-dir",
                    str(run_dir),
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual(err, "")
            payload = load_gpu_payload(out)
            self.assertEqual(payload["command"], "release")
            self.assertEqual(payload["status"], "released")
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["occupied"], 0)
            self.assertEqual(payload["available"], 1)

            code, out, err = run_gpu_cli(
                [
                    "release",
                    "--gpu",
                    "0",
                    "--pid",
                    str(own_pid),
                    "--max-per-gpu",
                    "1",
                    "--run-dir",
                    str(run_dir),
                ]
            )
            self.assertEqual(code, 1)
            self.assertEqual(err, "")
            payload = load_gpu_payload(out)
            self.assertEqual(payload["status"], "not_found")
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["reason"], "slot_not_found")

            with patch.dict(os.environ, {"BYPASS_GPU_GOVERNOR": "1"}, clear=False):
                code, out, err = run_gpu_cli(["acquire", "--gpu", "0", "--pid", "123"])
            self.assertEqual(code, 0)
            self.assertEqual(err, "")
            payload = load_gpu_payload(out)
            self.assertEqual(payload["status"], "bypassed")
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["reason"], "bypassed")

            with patch.object(
                gpu_governor, "acquire_slot", side_effect=gpu_governor.GovernorBusy("full")
            ):
                code, out, err = run_gpu_cli(
                    ["acquire", "--gpu", "0", "--pid", "123", "--run-dir", str(run_dir)]
                )
            self.assertEqual(code, 2)
            self.assertEqual(err, "")
            payload = load_gpu_payload(out)
            self.assertEqual(payload["status"], "timeout")
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["reason"], "timeout")
            self.assertEqual(payload["error"], "full")

        def run_graph_cli(argv: list[str]) -> tuple[int, str]:
            out = io.StringIO()
            with patch("sys.argv", ["finding_graph", *argv]), contextlib.redirect_stdout(out):
                try:
                    graph_cli.main()
                    code = 0
                except SystemExit as exc:
                    code = int(exc.code or 0)
            return code, out.getvalue()

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}, clear=False):
                local_store.init_db()
                old = {
                    "id": "old",
                    "finding_type": "result",
                    "title": "VARIANT-X baseline",
                    "content": "result",
                    "metrics": {"score": 0.5},
                    "variant_name": "VARIANT-X",
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "timestamp": "2026-01-01T00:00:00",
                }
                new = {
                    **old,
                    "id": "new",
                    "content": "confirmed",
                    "timestamp": "2026-01-02T00:00:00",
                }
                local_store.insert_finding(old)
                local_store.insert_finding(new)

                code, out = run_graph_cli(["--run-dir", str(run_dir), "--mode", "backfill"])
                self.assertEqual(code, 0)
                self.assertIn("inserted", out)
                code, out = run_graph_cli(["--run-dir", str(run_dir), "--mode", "health"])
                self.assertEqual(code, 0)
                self.assertIn("num_edges", out)
                viz_out = run_dir / "graph" / "side.html"
                code, out = run_graph_cli(
                    ["--run-dir", str(run_dir), "--mode", "viz", "--output", str(viz_out)]
                )
                self.assertEqual(code, 0)
                self.assertTrue(viz_out.exists())
                code, out = run_graph_cli(["--run-dir", str(run_dir), "--mode", "wipe", "--yes"])
                self.assertEqual(code, 0)
                self.assertIn("deleted", out)

            with self.assertRaises(SystemExit):
                graph_cli._setup_env(run_dir / "missing")


class FindingGraphRuntimeContractsTest(unittest.TestCase):
    def test_graph_builder_context_health_and_maintainer_paths(self) -> None:
        from praxist.plugins.graph_maintainers.finding_graph_mvp import engine
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        def finding(fid: str, *, peer: str, ts: str, title: str, content: str = "result"):
            return {
                "id": fid,
                "finding_type": "result",
                "title": title,
                "content": content,
                "metrics": {"score": 0.5},
                "variant_name": "VARIANT-X alpha=0.3",
                "peer_id": peer,
                "generation_id": 0,
                "timestamp": ts,
                "extra": {"notes": ["复现", {"ref": "fs_" + "a" * 32}]},
            }

        old = finding(
            "old",
            peer="gen0_peer1",
            ts="2026-01-01T00:00:00",
            title="VARIANT-X original",
        )
        mid = finding(
            "mid",
            peer="gen0_peer1",
            ts="2026-01-02T00:00:00",
            title="VARIANT-X second",
            content="updated result",
        )
        new = finding(
            "new",
            peer="gen0_peer2",
            ts="2026-01-03T00:00:00",
            title="VARIANT-X confirmed",
            content="confirmed and supports old",
        )
        linked = {
            **finding(
                "linked",
                peer="gen0_peer3",
                ts="2026-01-04T00:00:00",
                title="VARIANT-X link",
                content="not consistent with old and failed",
            ),
            "links": json.dumps(
                [
                    {"target_finding_id": "old", "edge_type": "unknown"},
                    {"target_finding_id": "linked", "edge_type": "supports"},
                    {"target_finding_id": "missing", "edge_type": "supports"},
                    "bad",
                ]
            ),
        }
        builder = engine.FindingGraphBuilder([old, mid, new, linked, {"id": "", "title": ""}])
        self.assertEqual(
            engine._normalize_title_tokens("VARIANT-X and METHOD-A1"), {"VARIANT-X", "METHOD-A1"}
        )
        self.assertFalse(
            engine._has_any_non_negated("not consistent with variant-x", engine._SUPPORT_EN)
        )
        self.assertIn(
            "fs_" + "a" * 32,
            engine._extract_referenced_ids(engine.FindingGraphBuilder._text_blob(old)),
        )
        proposed = builder.propose_edges_for(linked)
        self.assertTrue(any(edge["edge_type"] == "challenges" for edge in proposed))
        self.assertTrue(any(edge["edge_type"] == "related_to" for edge in proposed))
        self.assertTrue(any(edge["src_finding_id"] == "new" for edge in builder.build_all_edges()))
        resolved = builder._resolve(
            [
                builder._mk_edge("x", "y", "supports", 0.60, "rule", "rule_engine", {}),
                builder._mk_edge("x", "y", "challenges", 0.55, "agent", "agent_declared", {}),
                builder._mk_edge("x", "y", "related_to", 0.99, "weak", "rule_engine", {}),
                builder._mk_edge("x", "x", "supports", 0.99, "self", "rule_engine", {}),
            ],
            "x",
        )
        self.assertEqual([edge["edge_type"] for edge in resolved], ["challenges", "related_to"])
        self.assertGreater(
            engine._score_edge_pair({"edge_type": "supports", "confidence": 0.9}, "a", "b"), 1.0
        )
        self.assertIn("`​``", engine._snippet("a\n```code```", 20))
        self.assertEqual(engine._previous_generation_peer_id("gen2_peer3"), "gen1_peer3")
        self.assertIsNone(engine._previous_generation_peer_id("gen0_peer3"))

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}, clear=False):
                local_store.init_db()
                for row in [old, mid, new, linked]:
                    local_store.insert_finding(row)
                inserted = local_store.insert_edges_batch(builder.build_all_edges())
                self.assertGreater(inserted, 0)
                health = engine.write_graph_health(run_dir / "graph")
                self.assertGreaterEqual(health["num_findings"], 4)
                self.assertGreater(health["linked_finding_ratio"], 0)
                self.assertTrue((run_dir / "graph" / "graph_health.json").exists())

                context = engine.build_session_start_graph_context("gen0_peer1", max_neighbors=4)
                self.assertIn("Graph-surfaced context", context)
                lineage = engine.build_session_start_graph_context("gen1_peer1", max_neighbors=4)
                self.assertIn("lineage predecessor", lineage)
                orientation = engine._render_orientation_context(2)
                self.assertIn("most-connected", orientation)

                maintainer = engine.FindingGraphMaintainer(run_dir, poll_interval=1)
                empty_lock = maintainer._cycle_lock
                empty_lock.acquire()
                try:
                    self.assertEqual(maintainer.sync_once()["status"], "busy")
                finally:
                    empty_lock.release()
                with (
                    patch.object(
                        engine,
                        "write_graph_health",
                        side_effect=RuntimeError("health boom"),
                    ),
                    patch(
                        "praxist.plugins.graph_maintainers.finding_graph_mvp.viz.render_graph_html",
                        side_effect=RuntimeError("viz boom"),
                    ),
                ):
                    self.assertEqual(maintainer.sync_once()["status"], "ok")
                with patch.object(local_store, "init_db", side_effect=RuntimeError("db boom")):
                    self.assertEqual(maintainer._sync_once_inner()["status"], "error")
                with patch.object(
                    local_store, "get_all_findings", side_effect=RuntimeError("find boom")
                ):
                    self.assertEqual(maintainer._sync_once_inner()["status"], "error")

                with patch.object(
                    engine,
                    "wait_for_filesystem_event",
                    side_effect=RuntimeError("watch boom"),
                ):
                    maintainer.start()
                    maintainer.start()
                    maintainer.stop()
                    maintainer.stop()

                engine.reset_graph_observability_state()
                self.assertEqual(
                    engine._report_maintainer_status(None)["last_cycle_status"], "never"
                )


class DeliverablesContractsTest(unittest.TestCase):
    def test_deliverable_reports_package_data_and_cli(self) -> None:
        from praxist import deliver
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "run_summary.json").write_text(
                json.dumps(
                    {
                        "task_name": "Demo Task",
                        "task_id": "demo",
                        "generations_completed": 2,
                        "total_duration_seconds": 7200,
                        "run_dir": str(run_dir),
                    }
                ),
                encoding="utf-8",
            )
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir()
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "primary_metric": "score",
                        "metric_direction": "maximize",
                        "cumulative_top": [
                            {
                                "generation_id": 1,
                                "variant_name": "Variant Long Name",
                                "metric_value": 0.91,
                                "metrics": {"score": 0.91, "gap": 0.02},
                            }
                        ],
                        "generations": {
                            "0": [],
                            "1": [{"metric_value": 0.91, "variant_name": "Variant"}],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (frontier_dir / "best_finding.json").write_text("{}", encoding="utf-8")
            snapshot_src = root / "snapshot_src"
            snapshot_src.mkdir()
            (snapshot_src / "optimizer.py").write_text("class Optimizer: pass\n", encoding="utf-8")
            with tarfile.open(frontier_dir / "best_snapshot.tar.gz", "w:gz") as tar:
                tar.add(snapshot_src / "optimizer.py", arcname="optimizer.py")
                link = tarfile.TarInfo("link")
                link.type = tarfile.SYMTYPE
                link.linkname = "/etc/passwd"
                tar.addfile(link)

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}, clear=False):
                local_store.init_db()
                local_store.insert_finding(
                    {
                        "id": "f1",
                        "finding_type": "result",
                        "title": "Strong result",
                        "content": "x" * 2100,
                        "metrics": {"score": 0.91},
                        "variant_name": "Variant Long Name",
                        "peer_id": "gen1_peer0",
                        "generation_id": 1,
                        "timestamp": "2026-01-01T00:00:00",
                        "extra": {"tier": "T3"},
                    }
                )
                local_store.insert_metric(
                    {
                        "run_id": "r",
                        "variant_name": "Variant Long Name",
                        "metrics": {"score": 0.91, "gap": 0.02},
                        "step": 1,
                        "peer_id": "gen1_peer0",
                        "generation_id": 1,
                        "timestamp": "2026-01-01T00:00:00",
                    }
                )
            findings = deliver.load_all_findings(run_dir)
            metrics = deliver.load_all_metrics(run_dir)
            self.assertEqual(findings[0]["id"], "f1")
            self.assertEqual(metrics[0]["metrics"]["score"], 0.91)

            self.assertIsNone(deliver.load_run_summary(root / "missing"))
            self.assertIsNone(deliver.load_frontier_manifest(root / "missing"))
            summary = deliver.generate_executive_summary(
                deliver.load_run_summary(run_dir),
                deliver.load_frontier_manifest(run_dir),
                findings,
                metrics,
            )
            self.assertIn("Best Results", summary)
            self.assertIn("Generation Progression", summary)
            self.assertIn("Experiment Volume", summary)
            findings_report = deliver.generate_findings_report(findings)
            self.assertIn("*[truncated]*", findings_report)
            self.assertIn(
                "| Variant | Gen | Peer | score | gap |",
                deliver.generate_metrics_table(metrics, "score"),
            )
            self.assertIn("No metrics recorded", deliver.generate_metrics_table([], "score"))

            out_dir = root / "deliverables"
            packaged = deliver.package_deliverables(run_dir, out_dir, name="bundle")
            self.assertTrue((packaged / "executive_summary.md").exists())
            self.assertTrue((packaged / "findings_report.md").exists())
            self.assertTrue((packaged / "metrics_table.md").exists())
            self.assertTrue((packaged / "code" / "best_snapshot" / "optimizer.py").exists())
            self.assertTrue((packaged / "frontier" / "best_finding.json").exists())
            with self.assertRaises(FileExistsError):
                deliver.package_deliverables(run_dir, out_dir, name="bundle")
            overwritten = deliver.package_deliverables(
                run_dir, out_dir, name="bundle", overwrite=True
            )
            self.assertEqual(overwritten, packaged)
            with self.assertRaises(FileNotFoundError):
                deliver.package_deliverables(root / "missing", out_dir)

            fallback_run = root / "fallback"
            (fallback_run / "shared_findings").mkdir(parents=True)
            (fallback_run / "shared_findings" / "a.json").write_text(
                json.dumps({"id": "a", "finding_type": "insight", "generation_id": 0}),
                encoding="utf-8",
            )
            (fallback_run / "shared_findings" / "bad.json").write_text("{bad", encoding="utf-8")
            (fallback_run / "logs").mkdir()
            (fallback_run / "logs" / "metrics_log.jsonl").write_text(
                json.dumps({"variant_name": "v", "metrics": {"score": "0.5"}}) + "\n{bad\n\n",
                encoding="utf-8",
            )
            self.assertEqual(len(deliver.load_all_findings(fallback_run)), 1)
            self.assertEqual(len(deliver.load_all_metrics(fallback_run)), 1)
            self.assertEqual(
                deliver.extract_frontier_snapshots(fallback_run, root / "no_frontier"), 0
            )

            out = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "deliver",
                        "--run-dir",
                        str(run_dir),
                        "--out-dir",
                        str(out_dir),
                        "--name",
                        "cli_bundle",
                    ],
                ),
                contextlib.redirect_stdout(out),
            ):
                deliver.main()
            self.assertIn("Deliverables ready", out.getvalue())


if __name__ == "__main__":
    unittest.main()
