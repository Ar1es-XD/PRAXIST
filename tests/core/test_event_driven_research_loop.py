import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from praxist.plugins.tools.evaluation_tools.adapter import _handle_wait_for_file_impl
from praxist.plugins.workflow_stages.research_loop.backend.agent import (
    AgentResult,
    AutonomousAgentLoop,
    BaseAgent,
)
from praxist.plugins.workflow_stages.research_loop.backend.event_wait import (
    FileEventWaitResult,
)
from praxist.plugins.workflow_stages.research_loop.backend.synthesis_trigger import (
    SynthesisTrigger,
)


class EventDrivenResearchLoopTests(unittest.TestCase):
    def test_peer_waits_for_event_between_runtime_sessions(self) -> None:
        async def _run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp) / "run"
                gen_dir = run_dir / "gen_0"
                peer_dir = gen_dir / "gen0_peer0"
                findings_dir = run_dir / "shared_findings"
                peer_dir.mkdir(parents=True)
                findings_dir.mkdir(parents=True)
                stop_signal = gen_dir / "STOP_SIGNAL"

                loop = AutonomousAgentLoop(
                    peer_id="gen0_peer0",
                    generation_id=0,
                    task_prompt="test",
                    workspace=run_dir,
                    max_runtime_seconds=60,
                    logs_dir=peer_dir,
                    findings_dir=findings_dir,
                    local_mode=True,
                    stop_signal_path=stop_signal,
                )

                events: list[str] = []

                async def fake_run_session() -> None:
                    events.append("session")
                    if events.count("session") == 2:
                        stop_signal.write_text("stop", encoding="utf-8")

                async def fake_wait(*_args, **_kwargs) -> FileEventWaitResult:
                    events.append("wait")
                    event_filter = _kwargs["event_filter"]
                    self.assertFalse(event_filter(run_dir / "shared_store.db-wal"))
                    self.assertFalse(event_filter(peer_dir / "session.log"))
                    self.assertFalse(event_filter(run_dir / "graph" / "graph_health.json"))
                    self.assertFalse(event_filter(run_dir / "results" / "cell" / "results.json"))
                    self.assertFalse(event_filter(run_dir / "protected_pids" / "gen0_peer0.json"))
                    self.assertFalse(event_filter(run_dir / "variants" / "v" / "optimizer.py"))
                    self.assertTrue(event_filter(findings_dir / "finding.json"))
                    self.assertTrue(event_filter(stop_signal))
                    return FileEventWaitResult(
                        reason="filesystem_event",
                        elapsed_seconds=0.01,
                        paths=(str(findings_dir / "f.json"),),
                        used_inotify=True,
                    )

                loop._run_session = fake_run_session  # type: ignore[method-assign]
                with patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.agent.wait_for_filesystem_event",
                    side_effect=fake_wait,
                ):
                    result = await loop.run()

                self.assertEqual(events, ["session", "wait", "session", "wait"])
                self.assertEqual(result["sessions"], 2)
                self.assertEqual(result["stop_reason"], "synthesis_trigger")

        asyncio.run(_run())

    def test_synthesis_trigger_uses_event_wait(self) -> None:
        async def _run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp) / "run"
                gen_dir = run_dir / "gen_0"
                gen_dir.mkdir(parents=True)
                trigger = SynthesisTrigger(
                    run_dir=run_dir,
                    gen_dir=gen_dir,
                    gen_id=0,
                    gen_start_time=time.time(),
                    min_findings=999,
                    min_interval_minutes=999,
                    max_interval_minutes=999,
                    min_contributing_peers=5,
                    poll_interval_seconds=30,
                )

                calls: list[float] = []

                async def fake_wait(*_args, **kwargs) -> FileEventWaitResult:
                    calls.append(float(kwargs["timeout_seconds"]))
                    event_filter = kwargs["event_filter"]
                    self.assertTrue(event_filter(run_dir / "shared_findings" / "f.json"))
                    self.assertTrue(event_filter(run_dir / "shared_store.db-wal"))
                    self.assertTrue(event_filter(run_dir / "shared_store.db-shm"))
                    self.assertFalse(event_filter(gen_dir / "session.log"))
                    return FileEventWaitResult(reason="stop", elapsed_seconds=0.01)

                with patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.synthesis_trigger.wait_for_filesystem_event",
                    side_effect=fake_wait,
                ):
                    snap = await trigger.wait_until_fire()

                self.assertFalse(snap.fired)
                self.assertTrue(calls)
                self.assertGreaterEqual(trigger.poll_interval_seconds, 60)

        asyncio.run(_run())

    def test_unproductive_session_wait_ignores_sibling_events(self) -> None:
        async def _run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp) / "run"
                gen_dir = run_dir / "gen_0"
                peer_dir = gen_dir / "gen0_peer3"
                findings_dir = run_dir / "shared_findings"
                peer_dir.mkdir(parents=True)
                findings_dir.mkdir(parents=True)
                stop_signal = gen_dir / "STOP_SIGNAL"

                loop = AutonomousAgentLoop(
                    peer_id="gen0_peer3",
                    generation_id=0,
                    task_prompt="test",
                    workspace=run_dir,
                    max_runtime_seconds=60,
                    logs_dir=peer_dir,
                    findings_dir=findings_dir,
                    local_mode=True,
                    stop_signal_path=stop_signal,
                )

                async def fake_wait(*_args, **kwargs) -> FileEventWaitResult:
                    event_filter = kwargs["event_filter"]
                    self.assertFalse(event_filter(findings_dir / "sibling.json"))
                    self.assertFalse(event_filter(run_dir / "variants" / "v" / "optimizer.py"))
                    self.assertTrue(event_filter(stop_signal))
                    return FileEventWaitResult(reason="timeout", elapsed_seconds=0.01)

                with patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.agent.wait_for_filesystem_event",
                    side_effect=fake_wait,
                ):
                    await loop._wait_for_next_session_event(productive=False)

        asyncio.run(_run())

    def test_bootstrap_wait_response_retries_immediately(self) -> None:
        async def _run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp) / "run"
                gen_dir = run_dir / "gen_0"
                peer_dir = gen_dir / "gen0_peer4"
                findings_dir = run_dir / "shared_findings"
                peer_dir.mkdir(parents=True)
                findings_dir.mkdir(parents=True)

                loop = AutonomousAgentLoop(
                    peer_id="gen0_peer4",
                    generation_id=0,
                    task_prompt="Praxist task body",
                    workspace=run_dir,
                    max_runtime_seconds=60,
                    logs_dir=peer_dir,
                    findings_dir=findings_dir,
                    local_mode=True,
                    stop_signal_path=gen_dir / "STOP_SIGNAL",
                )

                prompts: list[str] = []

                async def fake_execute(_self: BaseAgent, task: str) -> AgentResult:
                    prompts.append(task)
                    if len(prompts) == 1:
                        return AgentResult(
                            success=True,
                            output={
                                "text_outputs": [
                                    "You haven't asked me to do anything yet. "
                                    "What would you like me to do?"
                                ],
                                "tool_uses": [],
                            },
                            duration=0.1,
                            iteration_count=0,
                        )
                    return AgentResult(
                        success=True,
                        output={
                            "text_outputs": ["Starting work now."],
                            "tool_uses": [{"tool": "Read", "input": {}}],
                        },
                        duration=0.2,
                        iteration_count=1,
                    )

                with patch.object(BaseAgent, "execute", fake_execute):
                    result = await loop._run_session()

                self.assertEqual(len(prompts), 2)
                self.assertTrue(prompts[0].startswith("Praxist task body"))
                self.assertIn("Praxist Peer-Local Structured Memory", prompts[0])
                self.assertIn("Praxist Bootstrap Recovery", prompts[1])
                self.assertEqual(result.iteration_count, 1)
                log_text = next(peer_dir.glob("session_*.log")).read_text(encoding="utf-8")
                self.assertIn("Bootstrap failure", log_text)

        asyncio.run(_run())

    def test_wait_for_file_wakes_on_file_creation(self) -> None:
        async def _run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp) / "run"
                run_dir.mkdir()
                target = run_dir / "result.json"

                async def create_target() -> None:
                    await asyncio.sleep(0.05)
                    target.write_text('{"status": "ok"}', encoding="utf-8")

                with patch.dict("os.environ", {"LOCAL_STORE_DIR": str(run_dir)}, clear=False):
                    create_task = asyncio.create_task(create_target())
                    result = await _handle_wait_for_file_impl(
                        {
                            "path": str(target),
                            "timeout_seconds": 3,
                            "poll_interval_seconds": 2,
                            "min_bytes": 1,
                        }
                    )
                    await create_task

                payload = json.loads(result["content"][0]["text"])
                self.assertEqual(payload["status"], "ready")
                self.assertEqual(payload["matched_paths"], [str(target.resolve())])

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
