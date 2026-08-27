"""Tests for ``execute_autonomous`` — peer subprocess entry point (#75 batch 5).

``main()`` reads seven env vars at the subprocess boundary; this test
file pins the contract of ``PeerInvocationConfig.from_environ`` and
verifies the inner ``run_peer`` helper has no env reads of its own (the
migration target so future callers can bypass the env round-trip).
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class PeerInvocationConfigTest(unittest.TestCase):
    def test_from_environ_applies_documented_defaults(self) -> None:
        from praxist.infrastructure.execute_autonomous import PeerInvocationConfig

        cfg = PeerInvocationConfig.from_environ({})
        self.assertEqual(cfg.peer_id, "peer_0")
        self.assertEqual(cfg.generation_id, 0)
        self.assertEqual(cfg.max_runtime_seconds, 24 * 3600)
        self.assertFalse(cfg.local_mode)
        self.assertEqual(cfg.model, "")
        self.assertEqual(cfg.task_prompt, "")
        self.assertEqual(cfg.task_prompt_file, "")
        self.assertEqual(cfg.logs_dir, Path("logs"))

    def test_from_environ_reads_every_documented_env_var(self) -> None:
        from praxist.infrastructure.execute_autonomous import PeerInvocationConfig

        env = {
            "PEER_ID": "gen3_peer7",
            "GENERATION_ID": "3",
            "MAX_RUNTIME_SECONDS": "60",
            "LOCAL_MODE": "true",
            "AGENT_MODEL": "claude-opus-4-7",
            "TASK_PROMPT": "inline prompt body",
            "TASK_PROMPT_FILE": "/tmp/task.md",
            "LOGS_DIR": "/var/log/praxist",
        }
        cfg = PeerInvocationConfig.from_environ(env)
        self.assertEqual(cfg.peer_id, "gen3_peer7")
        self.assertEqual(cfg.generation_id, 3)
        self.assertEqual(cfg.max_runtime_seconds, 60)
        self.assertTrue(cfg.local_mode)
        self.assertEqual(cfg.model, "claude-opus-4-7")
        self.assertEqual(cfg.task_prompt, "inline prompt body")
        self.assertEqual(cfg.task_prompt_file, "/tmp/task.md")
        self.assertEqual(cfg.logs_dir, Path("/var/log/praxist"))

    def test_from_environ_local_mode_accepts_documented_truthy_variants(self) -> None:
        from praxist.infrastructure.execute_autonomous import PeerInvocationConfig

        for truthy in ("1", "true", "TRUE", "Yes", "yes"):
            with self.subTest(local_mode=truthy):
                cfg = PeerInvocationConfig.from_environ({"LOCAL_MODE": truthy})
                self.assertTrue(cfg.local_mode)
        for falsy in ("0", "false", "no", "", "off"):
            with self.subTest(local_mode=falsy):
                cfg = PeerInvocationConfig.from_environ({"LOCAL_MODE": falsy})
                self.assertFalse(cfg.local_mode)

    def test_from_environ_malformed_ints_fall_back_to_defaults(self) -> None:
        from praxist.infrastructure.execute_autonomous import PeerInvocationConfig

        cfg = PeerInvocationConfig.from_environ(
            {"GENERATION_ID": "not-an-int", "MAX_RUNTIME_SECONDS": "x"}
        )
        self.assertEqual(cfg.generation_id, 0)
        self.assertEqual(cfg.max_runtime_seconds, 24 * 3600)

    def test_resolve_task_prompt_prefers_inline_over_file(self) -> None:
        from praxist.infrastructure.execute_autonomous import PeerInvocationConfig

        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "task.md"
            file_path.write_text("from file", encoding="utf-8")
            cfg = PeerInvocationConfig(task_prompt="inline", task_prompt_file=str(file_path))
            self.assertEqual(cfg.resolve_task_prompt(), "inline")

    def test_resolve_task_prompt_reads_file_when_inline_empty(self) -> None:
        from praxist.infrastructure.execute_autonomous import PeerInvocationConfig

        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "task.md"
            file_path.write_text("from file body", encoding="utf-8")
            cfg = PeerInvocationConfig(task_prompt_file=str(file_path))
            self.assertEqual(cfg.resolve_task_prompt(), "from file body")

    def test_resolve_task_prompt_returns_none_when_both_missing(self) -> None:
        from praxist.infrastructure.execute_autonomous import PeerInvocationConfig

        self.assertIsNone(PeerInvocationConfig().resolve_task_prompt())
        # File path set but file missing → still None.
        self.assertIsNone(
            PeerInvocationConfig(task_prompt_file="/nonexistent/path").resolve_task_prompt()
        )


class RunPeerTest(unittest.TestCase):
    """``run_peer`` is the env-free inner driver (#75 batch 5)."""

    def test_run_peer_exits_one_when_task_prompt_missing(self) -> None:
        from praxist.infrastructure.execute_autonomous import (
            PeerInvocationConfig,
            run_peer,
        )

        with tempfile.TemporaryDirectory() as tmp:
            cfg = PeerInvocationConfig(logs_dir=Path(tmp))
            with self.assertRaises(SystemExit) as cm:
                run_peer(cfg)
        self.assertEqual(cm.exception.code, 1)

    def test_run_peer_drives_launch_autonomous_loop_with_config_fields(self) -> None:
        """End-to-end: ``run_peer`` reads no env, just calls

        ``launch_autonomous_loop`` with config fields and uploads
        artifacts when not in local mode.
        """
        from praxist.infrastructure import execute_autonomous

        captured: dict = {}

        async def fake_launch(**kwargs):
            captured.update(kwargs)
            return {"run_id": "run-from-loop", "ok": True}

        uploaded: list = []

        def fake_upload(peer_id, run_id, result):
            uploaded.append((peer_id, run_id, result))

        with tempfile.TemporaryDirectory() as tmp:
            cfg = execute_autonomous.PeerInvocationConfig(
                peer_id="peer_42",
                generation_id=7,
                max_runtime_seconds=120,
                local_mode=False,
                model="claude-opus-4-7",
                task_prompt="run this experiment",
                logs_dir=Path(tmp) / "logs",
            )
            with (
                patch.object(execute_autonomous, "launch_autonomous_loop", fake_launch),
                patch.object(execute_autonomous, "upload_final_artifacts", fake_upload),
            ):
                # run_peer swaps stdout/stderr for TeeOutput; restore in finally
                # so pytest's capture isn't left dangling on assertion failure.
                try:
                    execute_autonomous.run_peer(cfg)
                finally:
                    sys.stdout = sys.__stdout__
                    sys.stderr = sys.__stderr__
        self.assertEqual(captured["peer_id"], "peer_42")
        self.assertEqual(captured["generation_id"], 7)
        self.assertEqual(captured["task_prompt"], "run this experiment")
        self.assertEqual(captured["max_runtime_seconds"], 120)
        self.assertEqual(captured["model"], "claude-opus-4-7")
        self.assertFalse(captured["local_mode"])
        self.assertEqual(len(uploaded), 1)
        self.assertEqual(uploaded[0][0], "peer_42")
        self.assertEqual(uploaded[0][1], "run-from-loop")

    def test_run_peer_local_mode_skips_artifact_upload(self) -> None:
        from praxist.infrastructure import execute_autonomous

        async def fake_launch(**_kwargs):
            return {"run_id": "local-run"}

        upload_called = False

        def fake_upload(*_args, **_kwargs):
            nonlocal upload_called
            upload_called = True

        with tempfile.TemporaryDirectory() as tmp:
            cfg = execute_autonomous.PeerInvocationConfig(
                peer_id="peer_local",
                task_prompt="inline",
                local_mode=True,
                logs_dir=Path(tmp) / "logs",
            )
            with (
                patch.object(execute_autonomous, "launch_autonomous_loop", fake_launch),
                patch.object(execute_autonomous, "upload_final_artifacts", fake_upload),
            ):
                try:
                    execute_autonomous.run_peer(cfg)
                finally:
                    sys.stdout = sys.__stdout__
                    sys.stderr = sys.__stderr__
        self.assertFalse(upload_called)


class MainEntryPointTest(unittest.TestCase):
    """``main()`` is the documented env-reading subprocess boundary."""

    def test_main_builds_config_from_env_and_hands_off_to_run_peer(self) -> None:
        from praxist.infrastructure import execute_autonomous

        captured: dict = {}

        def fake_run_peer(cfg):
            captured["peer_id"] = cfg.peer_id
            captured["generation_id"] = cfg.generation_id
            captured["model"] = cfg.model
            captured["task_prompt"] = cfg.task_prompt
            captured["local_mode"] = cfg.local_mode

        env = {
            "PEER_ID": "from-env-peer",
            "GENERATION_ID": "2",
            "MAX_RUNTIME_SECONDS": "30",
            "LOCAL_MODE": "true",
            "AGENT_MODEL": "claude-opus-4-7",
            "TASK_PROMPT": "do the experiment",
        }
        with (
            patch.dict("os.environ", env, clear=False),
            patch.object(execute_autonomous, "run_peer", fake_run_peer),
        ):
            execute_autonomous.main()
        self.assertEqual(captured["peer_id"], "from-env-peer")
        self.assertEqual(captured["generation_id"], 2)
        self.assertEqual(captured["model"], "claude-opus-4-7")
        self.assertEqual(captured["task_prompt"], "do the experiment")
        self.assertTrue(captured["local_mode"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


# Suppress unused-import lints; importing asyncio reserves it for the
# fake_launch coroutines above. (ruff treats it as unused otherwise.)
_ = asyncio
