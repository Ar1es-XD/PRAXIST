from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class ClaudeSdkRuntimeAdapterContractsTest(unittest.TestCase):
    def test_deepseek_thinking_uses_the_anthropic_compatibility_contract(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        common = dict(
            name="agent",
            allowed_tools=[],
            workspace=Path("."),
            mcp_servers={},
            model="deepseek-v4-pro",
            permission_mode="default",
            model_provider_ref="model_provider:deepseek_alias",
        )
        enabled = adapter.LegacyClaudeRuntimeOptions(
            **common,
            premium_mode=True,
            reasoning_effort="high",
        )
        disabled = adapter.LegacyClaudeRuntimeOptions(
            **common,
            premium_mode=True,
            reasoning_effort="off",
        )
        automatic = adapter.LegacyClaudeRuntimeOptions(
            **common,
            reasoning_effort="auto",
        )
        defaulted = adapter.LegacyClaudeRuntimeOptions(**common)

        self.assertEqual(
            adapter._claude_reasoning_options(enabled),
            {
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "effort": "high",
            },
        )
        self.assertEqual(
            adapter._claude_reasoning_options(disabled),
            {"thinking": {"type": "disabled"}},
        )
        self.assertEqual(adapter._claude_reasoning_options(automatic), {})
        self.assertEqual(
            adapter._claude_reasoning_options(defaulted),
            {
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "effort": "max",
            },
        )

    def test_non_deepseek_premium_mode_remains_adaptive(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        options = adapter.LegacyClaudeRuntimeOptions(
            name="agent",
            allowed_tools=[],
            workspace=Path("."),
            mcp_servers={},
            model="model",
            permission_mode="default",
            premium_mode=True,
        )
        self.assertEqual(
            adapter._claude_reasoning_options(options),
            {"thinking": {"type": "adaptive"}, "effort": "max"},
        )

    def test_usage_normalization_includes_claude_cache_read_and_creation(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        usage = adapter._normalized_usage(
            {
                "input_tokens": 10,
                "cache_read_input_tokens": 70,
                "cache_creation_input_tokens": 20,
                "output_tokens": 5,
            }
        )

        self.assertEqual(
            usage,
            {
                "input_tokens": 100.0,
                "total_input_tokens": 100.0,
                "uncached_input_tokens": 10.0,
                "cached_input_tokens": 70.0,
                "cache_read_input_tokens": 70.0,
                "cache_creation_input_tokens": 20.0,
                "output_tokens": 5.0,
                "total_tokens": 105.0,
            },
        )

    def test_sdk_error_results_and_failed_background_tasks_are_runtime_failures(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        class ClaudeAgentOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class HookMatcher:
            def __init__(self, matcher=None, hooks=None, timeout=None):
                self.matcher = matcher
                self.hooks = list(hooks or [])
                self.timeout = timeout

        class ResultMessage:
            is_error = True
            errors = ["provider unavailable"]
            result = "provider unavailable"

        class TaskNotificationMessage:
            task_id = "task-1"
            status = "failed"
            output_file = ""
            summary = "worker exited with status 7"

        class ToolUseBlock:
            name = "Task"
            input = {"prompt": "run evaluator"}

        class AssistantMessage:
            content = [ToolUseBlock()]

        async def error_query(prompt: str, options):
            yield ResultMessage()

        async def failed_task_query(prompt: str, options):
            yield AssistantMessage()
            yield TaskNotificationMessage()

        fake_sdk = {
            "ClaudeAgentOptions": ClaudeAgentOptions,
            "HookMatcher": HookMatcher,
            "AssistantMessage": AssistantMessage,
            "ResultMessage": ResultMessage,
            "ToolUseBlock": ToolUseBlock,
        }

        def execute(query):
            with (
                tempfile.TemporaryDirectory() as tmp,
                patch.object(
                    adapter,
                    "_load_claude_sdk",
                    return_value={**fake_sdk, "query": query},
                ),
            ):
                return asyncio.run(
                    adapter.ClaudeSdkAgentRuntime().execute_legacy(
                        "task",
                        adapter.LegacyClaudeRuntimeOptions(
                            name="agent",
                            allowed_tools=[],
                            workspace=Path(tmp),
                            mcp_servers={},
                            model="fake",
                            permission_mode="default",
                        ),
                    )
                )

        sdk_error = execute(error_query)
        self.assertFalse(sdk_error.success)
        self.assertIn("provider unavailable", sdk_error.error or "")
        self.assertTrue(sdk_error.output["result_is_error"])
        self.assertEqual(sdk_error.output["result_errors"], ["provider unavailable"])

        failed_task = execute(failed_task_query)
        self.assertFalse(failed_task.success)
        self.assertIn("background tasks failed", failed_task.error or "")
        self.assertEqual(failed_task.iteration_count, 1)
        self.assertTrue(failed_task.output["terminal_background_only"])
        self.assertEqual(failed_task.output["background_tasks"][0]["status"], "failed")

    def test_sdk_iterator_failure_preserves_collected_background_metadata(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        class ClaudeAgentOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class HookMatcher:
            def __init__(self, matcher=None, hooks=None, timeout=None):
                self.matcher = matcher
                self.hooks = list(hooks or [])
                self.timeout = timeout

        class TaskNotificationMessage:
            task_id = "task-42"
            status = "failed"
            output_file = "/tmp/task-42.output"
            summary = "worker exited 7"

        async def failing_stream(prompt: str, options):
            yield TaskNotificationMessage()
            raise RuntimeError("provider transport reset after terminal task")

        fake_sdk = {
            "ClaudeAgentOptions": ClaudeAgentOptions,
            "HookMatcher": HookMatcher,
            "query": failing_stream,
            "AssistantMessage": type("AssistantMessage", (), {}),
            "ResultMessage": type("ResultMessage", (), {}),
            "ToolUseBlock": type("ToolUseBlock", (), {}),
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_claude_sdk", return_value=fake_sdk),
        ):
            result = asyncio.run(
                adapter.ClaudeSdkAgentRuntime().execute_legacy(
                    "task",
                    adapter.LegacyClaudeRuntimeOptions(
                        name="agent",
                        allowed_tools=[],
                        workspace=Path(tmp),
                        mcp_servers={},
                        model="fake",
                        permission_mode="default",
                    ),
                )
            )

        self.assertFalse(result.success)
        self.assertIn("transport reset", result.error or "")
        self.assertTrue(result.output["terminal_background_only"])
        self.assertEqual(
            result.output["background_tasks"],
            [
                {
                    "task_id": "task-42",
                    "status": "failed",
                    "output_file": "/tmp/task-42.output",
                    "summary": "worker exited 7",
                }
            ],
        )

    def test_close_freeze_hook_blocks_new_bash_work_but_allows_result_reads(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            gen_dir = run_dir / "gen_0"
            gen_dir.mkdir(parents=True)
            env = {
                "PRAXIST_LAUNCH_GUARD_ENABLED": "1",
                "PRAXIST_RUN_DIR": str(run_dir),
                "GENERATION_ID": "0",
                "PRAXIST_EVALUATION_ENTRYPOINT": "evaluations/run.py",
            }
            self.assertIsNone(
                adapter._closing_signal_bash_reason("python evaluations/run.py --full", env)
            )
            (gen_dir / "CLOSING_SIGNAL").write_text("trigger_reason=mature_quorum\n")
            for command in (
                "python evaluations/run.py --full",
                "nohup evaluations/run.py --full &",
                "./evaluations/run.py --full",
                "cat result.json; python evaluations/run.py --full",
                'echo "$(python evaluations/run.py --full)"',
                "nohup task-evaluator --full &",
                "python tools/summarize.py 2>&1 &",
            ):
                self.assertIn(
                    "generation is closing",
                    adapter._closing_signal_bash_reason(command, env) or "",
                )
            self.assertIsNone(
                adapter._closing_signal_bash_reason("cat result.json && rg score results", env)
            )
            self.assertIsNone(
                adapter._closing_signal_bash_reason(
                    "python evaluations/summarize.py --input results/raw.json",
                    env,
                )
            )
            self.assertIsNone(
                adapter._closing_signal_bash_reason(
                    'echo "$(python tools/summarize.py results/raw.json)"',
                    env,
                )
            )
            self.assertIsNone(
                adapter._closing_signal_bash_reason("python tools/summarize.py 2>&1", env)
            )
            self.assertIsNone(adapter._closing_signal_bash_reason("cmd &>combined.log", env))
            self.assertIsNone(adapter._closing_signal_bash_reason("cat evaluations/run.py", env))
            self.assertIsNone(
                adapter._closing_signal_bash_reason(
                    "${PRAXIST_TASK_PYTHON:-python} assets/evaluate.py",
                    env,
                )
            )
            self.assertIsNone(
                adapter._closing_signal_bash_reason("python tools/run.py --summarize", env)
            )
            self.assertIsNone(adapter._closing_signal_bash_reason('python -c "', env))

    def test_run_shutdown_blocks_new_work_without_enabling_generation_guard(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "ORCHESTRATOR_SHUTDOWN").write_text("source=praxist_stop\n")
            env = {
                "PRAXIST_RUN_DIR": str(run_dir),
                "PRAXIST_EVALUATION_ENTRYPOINT": "evaluations/run.py",
            }

            reason = adapter._closing_signal_bash_reason(
                "python evaluations/run.py --full",
                env,
            )
            self.assertIn("run is shutting down", reason or "")
            self.assertIsNone(adapter._closing_signal_bash_reason("cat results/final.json", env))

    def test_runtime_pretool_hook_rejects_direct_evaluator_after_close(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        class ClaudeAgentOptions:
            calls: list[dict] = []

            def __init__(self, **kwargs):
                ClaudeAgentOptions.calls.append(kwargs)

        class HookMatcher:
            def __init__(self, matcher=None, hooks=None, timeout=None):
                self.matcher = matcher
                self.hooks = list(hooks or [])
                self.timeout = timeout

        class ResultMessage:
            result = {"done": True}

        async def query(prompt: str, options):
            yield ResultMessage()

        fake_sdk = {
            "ClaudeAgentOptions": ClaudeAgentOptions,
            "HookMatcher": HookMatcher,
            "query": query,
            "AssistantMessage": type("AssistantMessage", (), {}),
            "ResultMessage": ResultMessage,
            "ToolUseBlock": type("ToolUseBlock", (), {}),
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_claude_sdk", return_value=fake_sdk),
        ):
            run_dir = Path(tmp) / "run"
            (run_dir / "gen_0").mkdir(parents=True)
            (run_dir / "gen_0" / "CLOSING_SIGNAL").write_text("close\n")
            result = asyncio.run(
                adapter.ClaudeSdkAgentRuntime().execute_legacy(
                    "finish current work",
                    adapter.LegacyClaudeRuntimeOptions(
                        name="gen0_peer8-session",
                        allowed_tools=["Bash"],
                        workspace=Path(tmp),
                        mcp_servers={},
                        model="test",
                        permission_mode="acceptEdits",
                        env={
                            "PRAXIST_LAUNCH_GUARD_ENABLED": "1",
                            "PRAXIST_RUN_DIR": str(run_dir),
                            "GENERATION_ID": "0",
                            "PRAXIST_EVALUATION_ENTRYPOINT": "eval/run.py",
                        },
                    ),
                )
            )
            bash_hook = next(
                matcher.hooks[0]
                for matcher in ClaudeAgentOptions.calls[-1]["hooks"]["PreToolUse"]
                if matcher.matcher == "Bash"
            )
            denied = asyncio.run(
                bash_hook(
                    {"tool_name": "Bash", "tool_input": {"command": "python eval/run.py --T1"}},
                    None,
                    None,
                )
            )
            allowed = asyncio.run(
                bash_hook(
                    {"tool_name": "Bash", "tool_input": {"command": "cat results/summary.json"}},
                    None,
                    None,
                )
            )
        self.assertTrue(result.success)
        self.assertEqual(denied["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertEqual(allowed, {})

    def test_direct_evaluator_is_rewritten_and_visible_as_protected_work(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import adapter
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            evaluator = root / "evaluate.py"
            ready = root / "evaluator.ready"
            release = root / "evaluator.release"
            evaluator.write_text(
                "from pathlib import Path\n"
                "import time\n"
                f"ready = Path({str(ready)!r})\n"
                f"release = Path({str(release)!r})\n"
                "ready.write_text('ready\\n', encoding='utf-8')\n"
                "deadline = time.monotonic() + 10.0\n"
                "while not release.exists() and time.monotonic() < deadline:\n"
                "    time.sleep(0.02)\n",
                encoding="utf-8",
            )
            env = {
                "PRAXIST_RUN_DIR": str(run_dir),
                "PRAXIST_PEER_ID": "gen0_peer8",
                "PRAXIST_EVALUATION_ENTRYPOINT": str(evaluator),
            }
            evaluator_log = root / "evaluator.log"
            command = f"TASK_LABEL=smoke {sys.executable} {evaluator} > {evaluator_log} 2>&1 &"
            rewritten = adapter._protected_evaluator_command(command, env)
            self.assertIsNotNone(rewritten)
            self.assertIsNone(adapter._protected_evaluator_command(f"cat {evaluator}", env))
            self.assertIsNone(
                adapter._protected_evaluator_command(
                    f"{sys.executable} {root / 'benchmark.py'}",
                    env,
                )
            )
            proc = subprocess.Popen(
                rewritten,
                shell=True,
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, **env},
            )
            try:
                deadline = time.monotonic() + 2.0
                active = []
                while time.monotonic() < deadline:
                    active = protected_pids.list_active_jobs(run_dir=run_dir)
                    if active:
                        break
                    time.sleep(0.02)
                self.assertEqual(len(active), 1)
                self.assertEqual(active[0].peer_id, "gen0_peer8")
                self.assertTrue(active[0].tag.startswith("auto-command-"))
            finally:
                release.write_text("release\n", encoding="utf-8")
                self.assertEqual(proc.wait(timeout=5), 0)
            self.assertEqual(protected_pids.list_active_jobs(run_dir=run_dir), [])

    def _request(self, runtime_ref: str = "agent_runtime:claude_sdk"):
        from praxist.core.protocol import (
            AgentRunRequest,
            CachePolicy,
            EnvPolicy,
            ModelCallSpec,
            ToolPermissionSet,
        )

        return AgentRunRequest(
            request_id="req/adapter",
            run_id="run",
            stage_id="research_loop",
            role_ref=None,
            agent_runtime_ref=runtime_ref,
            prompt_ref={"kind": "test"},
            system_prompt_ref=None,
            cwd=str(Path.cwd()),
            model_profile_ref="profile",
            model_call=ModelCallSpec(
                profile_id="profile",
                provider_ref="model_provider:fake",
                api_format="fake",
                model="fake-model",
                parameters={},
                credential_ref=None,
            ),
            tool_permissions=ToolPermissionSet(mode="allow_list", allowed_tools=["Bash"]),
            tool_servers=[],
            env_policy=EnvPolicy(),
            credential_ref=None,
            credential_mode="single",
            budget_grant_id=None,
            artifact_scope="run",
            timeout_seconds=1,
            cache_policy=CachePolicy(mode="runtime_auto_cache", frozen_prefix_hash="hash"),
        )

    def _liveness_fake_sdk(self):
        class ClaudeAgentOptions:
            calls: list[dict] = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                ClaudeAgentOptions.calls.append(kwargs)

        class HookMatcher:
            def __init__(self, matcher=None, hooks=None, timeout=None):
                self.matcher = matcher
                self.hooks = list(hooks or [])
                self.timeout = timeout

        class ToolUseBlock:
            def __init__(self, name: str, tool_id: str):
                self.name = name
                self.id = tool_id
                self.input = {}

        class ToolResultBlock:
            def __init__(self, tool_id: str):
                self.tool_use_id = tool_id

        class AssistantMessage:
            def __init__(self, content):
                self.content = list(content)

        class StreamEvent:
            pass

        class ResultMessage:
            result = {"status": "done"}
            is_error = False
            errors = []
            usage = {}

        sdk = {
            "ClaudeAgentOptions": ClaudeAgentOptions,
            "HookMatcher": HookMatcher,
            "query": None,
            "AssistantMessage": AssistantMessage,
            "ResultMessage": ResultMessage,
            "StreamEvent": StreamEvent,
            "ToolResultBlock": ToolResultBlock,
            "ToolUseBlock": ToolUseBlock,
        }
        messages = SimpleNamespace(
            AssistantMessage=AssistantMessage,
            ResultMessage=ResultMessage,
            StreamEvent=StreamEvent,
            ToolResultBlock=ToolResultBlock,
            ToolUseBlock=ToolUseBlock,
            option_calls=ClaudeAgentOptions.calls,
        )
        return sdk, messages

    def test_execute_legacy_normalizes_streams_billing_and_runtime_mismatch(self) -> None:
        from praxist.core.runtimes import AgentRuntimeExecutionContext
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        class TextBlock:
            def __init__(self, text: str):
                self.text = text

        class ToolUseBlock:
            name = "Bash"
            input = {"command": "echo ok"}

        class AssistantMessage:
            def __init__(self, text: str):
                self.content = [TextBlock(text), ToolUseBlock()]

        class ResultMessage:
            def __init__(
                self,
                *,
                result=None,
                is_error: bool = False,
                errors=None,
            ):
                self.result = {"done": True} if result is None else result
                self.is_error = is_error
                self.errors = [] if errors is None else errors

        class ClaudeAgentOptions:
            calls: list[dict] = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                ClaudeAgentOptions.calls.append(kwargs)

        class HookMatcher:
            def __init__(self, matcher=None, hooks=None, timeout=None):
                self.matcher = matcher
                self.hooks = list(hooks or [])
                self.timeout = timeout

        async def query(prompt: str, options):
            yield AssistantMessage("hello")
            yield ResultMessage()

        fake_sdk = {
            "ClaudeAgentOptions": ClaudeAgentOptions,
            "HookMatcher": HookMatcher,
            "query": query,
            "AssistantMessage": AssistantMessage,
            "ResultMessage": ResultMessage,
            "ToolUseBlock": ToolUseBlock,
        }

        callbacks = []
        stop_calls = 0

        def stop_check() -> bool:
            nonlocal stop_calls
            stop_calls += 1
            return stop_calls >= 1

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_claude_sdk", return_value=fake_sdk),
        ):
            result = asyncio.run(
                adapter.ClaudeSdkAgentRuntime().execute_legacy(
                    "task",
                    adapter.LegacyClaudeRuntimeOptions(
                        name="agent",
                        allowed_tools=["Bash"],
                        workspace=Path(tmp),
                        mcp_servers={"server": object()},
                        model="fake-model",
                        permission_mode="acceptEdits",
                        cli_path="/bin/claude",
                        message_callback=lambda msg: callbacks.append(type(msg).__name__),
                        system_prompt="system",
                        stop_check_fn=stop_check,
                        premium_mode=True,
                        env={
                            "A": "B",
                            "PRAXIST_RUN_DIR": str(Path(tmp) / "run"),
                            "PRAXIST_PEER_ID": "gen0_peer0",
                            "PRAXIST_EVALUATION_ENTRYPOINT": "evaluations/generic/run.py",
                        },
                    ),
                )
            )
        self.assertTrue(result.success)
        self.assertEqual(result.iteration_count, 0)
        self.assertEqual(callbacks, ["AssistantMessage"])
        self.assertEqual(ClaudeAgentOptions.calls[-1]["thinking"], {"type": "adaptive"})
        self.assertEqual(ClaudeAgentOptions.calls[-1]["env"]["A"], "B")
        self.assertNotIn("BASH_ENV", ClaudeAgentOptions.calls[-1]["env"])
        self.assertIn("PRAXIST_PEER_WORKSPACE", ClaudeAgentOptions.calls[-1]["env"])
        self.assertIn("PRAXIST_GUARD_WARNINGS_PATH", ClaudeAgentOptions.calls[-1]["env"])
        self.assertIn(".runtime_guards", ClaudeAgentOptions.calls[-1]["env"].get("PYTHONPATH", ""))
        self.assertIn("hooks", ClaudeAgentOptions.calls[-1])
        hook = ClaudeAgentOptions.calls[-1]["hooks"]["PreToolUse"][0].hooks[0]
        allowed = asyncio.run(
            hook(
                {
                    "tool_name": "Bash",
                    "tool_input": {"command": "echo ok"},
                },
                None,
                None,
            )
        )
        self.assertEqual(allowed, {})
        rewritten = asyncio.run(
            hook(
                {
                    "tool_name": "Bash",
                    "tool_input": {"command": "python evaluations/generic/run.py --full"},
                },
                None,
                None,
            )
        )
        self.assertEqual(
            rewritten["hookSpecificOutput"]["permissionDecision"],
            "allow",
        )
        self.assertIn(
            "protected_pids launch",
            rewritten["hookSpecificOutput"]["updatedInput"]["command"],
        )
        warned = asyncio.run(
            hook(
                {
                    "tool_name": "Bash",
                    "tool_input": {"command": "PYTHONPATH= python -c 'print(1)'"},
                },
                None,
                None,
            )
        )
        self.assertEqual(
            warned["hookSpecificOutput"]["permissionDecision"],
            "allow",
        )
        denied = asyncio.run(
            hook(
                {
                    "tool_name": "Bash",
                    "tool_input": {
                        "command": "rm -rf "
                        + ClaudeAgentOptions.calls[-1]["env"]["PRAXIST_DELETE_GUARD_RUN_DIR"]
                    },
                },
                None,
                None,
            )
        )
        self.assertEqual(
            denied["hookSpecificOutput"]["permissionDecision"],
            "deny",
        )
        self.assertEqual(ClaudeAgentOptions.calls[-1]["cli_path"], "/bin/claude")

        async def billing_query(prompt: str, options):
            yield ResultMessage(
                result="credit balance is too low",
                is_error=True,
                errors=["credit balance is too low"],
            )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                adapter,
                "_load_claude_sdk",
                return_value={**fake_sdk, "query": billing_query},
            ),
        ):
            billing = asyncio.run(
                adapter.ClaudeSdkAgentRuntime().execute_legacy(
                    "task",
                    adapter.LegacyClaudeRuntimeOptions(
                        name="agent",
                        allowed_tools=[],
                        workspace=Path(tmp),
                        mcp_servers={},
                        model="fake",
                        permission_mode="default",
                    ),
                )
            )
        self.assertFalse(billing.success)
        self.assertIn("API error", billing.error or "")
        self.assertIn("hooks", ClaudeAgentOptions.calls[-1])

        async def research_text_query(prompt: str, options):
            yield AssistantMessage(
                "agenda_version: '2.0'\n"
                "mainline_observation:\n"
                "  summary: authoritative quota-aware allocation is the hypothesis"
            )
            yield ResultMessage()

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                adapter,
                "_load_claude_sdk",
                return_value={**fake_sdk, "query": research_text_query},
            ),
        ):
            research_text = asyncio.run(
                adapter.ClaudeSdkAgentRuntime().execute_legacy(
                    "task",
                    adapter.LegacyClaudeRuntimeOptions(
                        name="chair_arbiter",
                        allowed_tools=[],
                        workspace=Path(tmp),
                        mcp_servers={},
                        model="fake",
                        permission_mode="default",
                    ),
                )
            )
        self.assertTrue(research_text.success, research_text.error)
        self.assertIn("agenda_version", research_text.output["text_outputs"][0])

        async def failing_query(prompt: str, options):
            raise RuntimeError("invalid api key sk-ant-secret")
            yield

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                adapter,
                "_load_claude_sdk",
                return_value={**fake_sdk, "query": failing_query},
            ),
        ):
            failed = asyncio.run(
                adapter.ClaudeSdkAgentRuntime().execute_legacy(
                    "task",
                    adapter.LegacyClaudeRuntimeOptions(
                        name="agent",
                        allowed_tools=[],
                        workspace=Path(tmp),
                        mcp_servers={},
                        model="fake",
                        permission_mode="default",
                    ),
                )
            )
        self.assertFalse(failed.success)
        self.assertNotIn("sk-ant-secret", failed.error or "")

        request = self._request()
        context = AgentRuntimeExecutionContext(
            tool_servers={},
            env={"ANTHROPIC_AUTH_TOKEN": "test-token"},
        )
        with patch.object(
            adapter.ClaudeSdkAgentRuntime,
            "execute_legacy",
            return_value=adapter.LegacyAgentResult(
                success=True,
                output={"text_outputs": ["ok"], "tool_uses": [{"tool": "Bash"}]},
                duration=0.1,
                iteration_count=1,
            ),
        ):
            normalized = asyncio.run(adapter.ClaudeSdkAgentRuntime().execute(request, context))
        self.assertTrue(normalized.success)
        self.assertEqual(normalized.tool_uses[0].tool_name, "Bash")

        with patch.object(
            adapter.ClaudeSdkAgentRuntime,
            "execute_legacy",
            return_value=adapter.LegacyAgentResult(
                success=False,
                output={},
                duration=0.1,
                iteration_count=0,
                error="401 User not found",
            ),
        ):
            failed = asyncio.run(adapter.ClaudeSdkAgentRuntime().execute(request, context))
        final = [event for event in failed.events if event.type == "final_result"][-1]
        self.assertEqual(
            final.payload["failure_context"]["runtime_ref"], "agent_runtime:claude_sdk"
        )
        self.assertEqual(final.payload["failure_context"]["provider_ref"], "model_provider:fake")
        self.assertEqual(final.payload["failure_context"]["relay_used"], False)

        mismatch = asyncio.run(
            adapter.ClaudeSdkAgentRuntime().execute(self._request("agent_runtime:other"), context)
        )
        self.assertFalse(mismatch.success)
        self.assertEqual(mismatch.failover_reason, "runtime_error")

        self.assertTrue(adapter._is_instance(SimpleNamespace(), "SimpleNamespace", {}))
        self.assertEqual(adapter._classify_legacy_failure("authentication_error"), "auth_error")
        self.assertEqual(adapter._classify_legacy_failure("quota exceeded"), "quota_exhausted")
        self.assertEqual(adapter._classify_legacy_failure("other"), "runtime_error")

    def test_execute_legacy_runtime_timeout_returns_failure_result(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        class ClaudeAgentOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class HookMatcher:
            def __init__(self, matcher=None, hooks=None, timeout=None):
                self.matcher = matcher
                self.hooks = list(hooks or [])
                self.timeout = timeout

        async def hanging_query(prompt: str, options):
            await asyncio.sleep(5)
            yield SimpleNamespace(content=[])

        fake_sdk = {
            "ClaudeAgentOptions": ClaudeAgentOptions,
            "HookMatcher": HookMatcher,
            "query": hanging_query,
            "AssistantMessage": object,
            "ResultMessage": object,
            "ToolUseBlock": object,
        }

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_claude_sdk", return_value=fake_sdk),
        ):
            result = asyncio.run(
                adapter.ClaudeSdkAgentRuntime().execute_legacy(
                    "task",
                    adapter.LegacyClaudeRuntimeOptions(
                        name="agent",
                        allowed_tools=[],
                        workspace=Path(tmp),
                        mcp_servers={},
                        model="fake",
                        permission_mode="default",
                        runtime_timeout_seconds=1,
                    ),
                )
            )

        self.assertFalse(result.success)
        self.assertEqual(result.error, "agent runtime timeout")

    def test_twelve_claude_streams_do_not_starve_parent_loop_and_aggregate_system_bursts(
        self,
    ) -> None:
        from praxist.core.runtimes import AgentRuntimeExecutionContext
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        class ClaudeAgentOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class HookMatcher:
            def __init__(self, matcher=None, hooks=None, timeout=None):
                self.matcher = matcher
                self.hooks = list(hooks or [])
                self.timeout = timeout

        class SystemMessage:
            def __init__(self, subtype: str):
                self.subtype = subtype
                self.data = {"subtype": subtype}

        class ResultMessage:
            result = {"status": "done"}
            is_error = False
            errors = []
            usage = {"input_tokens": 1, "output_tokens": 1}

        async def burst_query(prompt: str, options):
            # A synchronous SDK/parser stall in one session must not stop the
            # research-loop timer that supervises every session.
            time.sleep(0.12)
            yield SystemMessage("init")
            for _ in range(1000):
                yield SystemMessage("thinking_tokens")
            yield ResultMessage()

        fake_sdk = {
            "ClaudeAgentOptions": ClaudeAgentOptions,
            "HookMatcher": HookMatcher,
            "query": burst_query,
            "AssistantMessage": type("AssistantMessage", (), {}),
            "ResultMessage": ResultMessage,
            "ToolResultBlock": type("ToolResultBlock", (), {}),
            "ToolUseBlock": type("ToolUseBlock", (), {}),
        }
        observed = []

        async def exercise() -> tuple[list, float]:
            runtime = adapter.ClaudeSdkAgentRuntime()
            context = AgentRuntimeExecutionContext(
                tool_servers={},
                message_callback=observed.append,
                env={},
            )
            request = replace(self._request(), timeout_seconds=5)
            sessions = [asyncio.create_task(runtime.execute(request, context)) for _ in range(12)]
            await asyncio.sleep(0)
            started = time.monotonic()
            await asyncio.sleep(0.02)
            parent_delay = time.monotonic() - started
            return await asyncio.gather(*sessions), parent_delay

        with patch.object(adapter, "_load_claude_sdk", return_value=fake_sdk):
            results, parent_delay = asyncio.run(exercise())

        # Without per-session isolation the 12 synchronous stalls serialize
        # on the parent loop for about 1.44s. Coverage tracing adds sizeable
        # thread/GIL overhead, so keep a wide margin below that old behavior.
        self.assertLess(parent_delay, 0.8)
        self.assertTrue(all(result.success for result in results))
        self.assertEqual(len(observed), 24)
        self.assertEqual(
            [message.subtype for message in observed if type(message).__name__ == "SystemMessage"],
            ["init"] * 12,
        )
        for result in results:
            final = next(event for event in result.events if event.type == "final_result")
            self.assertEqual(
                final.payload["legacy_output"]["sdk_system_event_counts"],
                {"thinking_tokens": 1000},
            )

    def test_active_long_tools_are_healthy_instead_of_model_stalls(self) -> None:
        from praxist.core.runtimes import AgentRuntimeExecutionContext
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        tool_names = ("mcp__evaluation-tools__wait_for_file", "TaskOutput", "Bash")
        for index, tool_name in enumerate(tool_names):
            with self.subTest(tool_name=tool_name):
                sdk, messages = self._liveness_fake_sdk()
                tool_started = threading.Event()
                tool_observed = threading.Event()
                tool_finished = threading.Event()
                warnings_while_tool_active: list[str] = []

                async def long_tool_query(
                    prompt: str,
                    options,
                    *,
                    name=tool_name,
                    number=index,
                    message_types=messages,
                    started=tool_started,
                    observed=tool_observed,
                    finished=tool_finished,
                ):
                    tool_id = f"tool-{number}"
                    yield message_types.AssistantMessage(
                        [message_types.ToolUseBlock(name, tool_id)]
                    )
                    started.set()
                    while not observed.is_set():
                        await asyncio.sleep(0.005)
                    finished.set()
                    yield message_types.AssistantMessage([message_types.ToolResultBlock(tool_id)])
                    yield message_types.ResultMessage()

                def observe_warning(
                    message,
                    *args,
                    started=tool_started,
                    finished=tool_finished,
                    recorded=warnings_while_tool_active,
                    **kwargs,
                ) -> None:
                    del args, kwargs
                    if started.is_set() and not finished.is_set():
                        recorded.append(str(message))

                def observe_info(message, *args, observed=tool_observed, **kwargs) -> None:
                    del message, kwargs
                    if len(args) > 1 and args[1] == "foreground_tool_running":
                        observed.set()

                sdk["query"] = long_tool_query
                context = AgentRuntimeExecutionContext(tool_servers={}, env={})
                request = replace(self._request(), timeout_seconds=2)
                with (
                    patch.object(adapter, "_load_claude_sdk", return_value=sdk),
                    patch.object(adapter, "_SDK_LIVENESS_POLL_SECONDS", 0.005),
                    patch.object(adapter, "_SDK_STREAM_POLL_SECONDS", 0.005),
                    patch.object(adapter, "_SDK_STALL_WARNING_SECONDS", 0.02),
                    patch.object(adapter.logger, "warning", side_effect=observe_warning),
                    patch.object(adapter.logger, "info", side_effect=observe_info) as info,
                ):
                    result = asyncio.run(adapter.ClaudeSdkAgentRuntime().execute(request, context))

                self.assertTrue(result.success)
                self.assertFalse(
                    any(
                        "observed_state=model_waiting" in message
                        for message in warnings_while_tool_active
                    )
                )
                self.assertTrue(tool_observed.is_set())
                self.assertTrue(
                    any(
                        len(call.args) > 2 and call.args[2] == "foreground_tool_running"
                        for call in info.call_args_list
                    )
                )
                self.assertTrue(messages.option_calls[-1]["include_partial_messages"])

    def test_model_waiting_stall_is_stateful_and_reported_once(self) -> None:
        from praxist.core.runtimes import AgentRuntimeExecutionContext
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        sdk, messages = self._liveness_fake_sdk()

        async def idle_model_query(prompt: str, options):
            await asyncio.sleep(0.08)
            yield messages.ResultMessage()

        sdk["query"] = idle_model_query
        context = AgentRuntimeExecutionContext(tool_servers={}, env={})
        request = replace(self._request(), timeout_seconds=2)
        with (
            patch.object(adapter, "_load_claude_sdk", return_value=sdk),
            patch.object(adapter, "_SDK_LIVENESS_POLL_SECONDS", 0.005),
            patch.object(adapter, "_SDK_STALL_WARNING_SECONDS", 0.02),
            patch.object(adapter.logger, "warning") as warning,
        ):
            result = asyncio.run(adapter.ClaudeSdkAgentRuntime().execute(request, context))

        model_waiting_warnings = [
            call
            for call in warning.call_args_list
            if "observed_state=model_waiting" in str(call.args[0])
        ]
        self.assertTrue(result.success)
        self.assertEqual(len(model_waiting_warnings), 1)

    def test_active_background_work_is_healthy_instead_of_model_stall(self) -> None:
        from praxist.core.runtimes import AgentRuntimeExecutionContext
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        sdk, messages = self._liveness_fake_sdk()

        class TaskStartedMessage:
            task_id = "task-1"
            status = "running"
            output_file = ""
            summary = ""

        class TaskUpdatedMessage:
            task_id = "task-1"
            status = "completed"
            output_file = ""
            summary = "done"

        async def background_query(prompt: str, options):
            yield TaskStartedMessage()
            await asyncio.sleep(0.08)
            yield TaskUpdatedMessage()
            yield messages.ResultMessage()

        sdk["query"] = background_query
        context = AgentRuntimeExecutionContext(tool_servers={}, env={})
        request = replace(self._request(), timeout_seconds=2)
        with (
            patch.object(adapter, "_load_claude_sdk", return_value=sdk),
            patch.object(adapter, "_SDK_LIVENESS_POLL_SECONDS", 0.005),
            patch.object(adapter, "_SDK_STREAM_POLL_SECONDS", 0.005),
            patch.object(adapter, "_SDK_STALL_WARNING_SECONDS", 0.02),
            patch.object(adapter.logger, "warning") as warning,
            patch.object(adapter.logger, "info") as info,
        ):
            result = asyncio.run(adapter.ClaudeSdkAgentRuntime().execute(request, context))

        self.assertTrue(result.success)
        self.assertFalse(
            any(
                "observed_state=model_waiting" in str(call.args[0])
                for call in warning.call_args_list
            )
        )
        self.assertTrue(
            any(
                len(call.args) > 2 and call.args[2] == "background_work_running"
                for call in info.call_args_list
            )
        )

    def test_partial_stream_progress_prevents_false_model_stall(self) -> None:
        from praxist.core.runtimes import AgentRuntimeExecutionContext
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        sdk, messages = self._liveness_fake_sdk()

        async def partial_query(prompt: str, options):
            # Emit once before sleeping so isolated-worker startup time is not
            # mistaken for model-stream inactivity on slower CI interpreters.
            yield messages.StreamEvent()
            for _ in range(40):
                await asyncio.sleep(0.025)
                yield messages.StreamEvent()
            yield messages.ResultMessage()

        sdk["query"] = partial_query
        observed = []
        context = AgentRuntimeExecutionContext(
            tool_servers={}, message_callback=observed.append, env={}
        )
        request = replace(self._request(), timeout_seconds=2)
        with (
            patch.object(adapter, "_load_claude_sdk", return_value=sdk),
            patch.object(adapter, "_SDK_LIVENESS_POLL_SECONDS", 0.01),
            patch.object(adapter, "_SDK_STALL_WARNING_SECONDS", 0.5),
            patch.object(adapter.logger, "warning") as warning,
        ):
            result = asyncio.run(adapter.ClaudeSdkAgentRuntime().execute(request, context))

        self.assertTrue(result.success)
        self.assertFalse(
            any(
                "observed_state=model_waiting" in str(call.args[0])
                for call in warning.call_args_list
            )
        )
        self.assertEqual([type(message).__name__ for message in observed], ["ResultMessage"])

    def test_tool_completion_restarts_model_waiting_clock(self) -> None:
        from praxist.core.runtimes import AgentRuntimeExecutionContext
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        sdk, messages = self._liveness_fake_sdk()
        tool_finished = threading.Event()
        post_tool_waiting_warnings: list[str] = []

        async def tool_then_model_wait_query(prompt: str, options):
            yield messages.AssistantMessage([messages.ToolUseBlock("Bash", "tool-1")])
            await asyncio.sleep(0.04)
            yield messages.AssistantMessage([messages.ToolResultBlock("tool-1")])
            # The generator resumes only after the adapter processes the tool
            # result, so warnings recorded from here belong to the restarted
            # model-waiting interval rather than isolated-worker startup.
            tool_finished.set()
            await asyncio.sleep(0.05)
            yield messages.ResultMessage()

        def record_post_tool_warning(message, *args, **kwargs) -> None:
            del args, kwargs
            if tool_finished.is_set() and "observed_state=model_waiting" in str(message):
                post_tool_waiting_warnings.append(str(message))

        sdk["query"] = tool_then_model_wait_query
        context = AgentRuntimeExecutionContext(tool_servers={}, env={})
        request = replace(self._request(), timeout_seconds=2)
        with (
            patch.object(adapter, "_load_claude_sdk", return_value=sdk),
            patch.object(adapter, "_SDK_LIVENESS_POLL_SECONDS", 0.005),
            patch.object(adapter, "_SDK_STREAM_POLL_SECONDS", 0.005),
            patch.object(adapter, "_SDK_STALL_WARNING_SECONDS", 0.02),
            patch.object(adapter.logger, "warning", side_effect=record_post_tool_warning),
        ):
            result = asyncio.run(adapter.ClaudeSdkAgentRuntime().execute(request, context))

        self.assertTrue(result.success)
        self.assertEqual(len(post_tool_waiting_warnings), 1)

    def test_isolated_supervisor_enforces_existing_timeout_when_sdk_loop_is_blocked(
        self,
    ) -> None:
        from praxist.core.runtimes import AgentRuntimeExecutionContext
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        class ClaudeAgentOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class HookMatcher:
            def __init__(self, matcher=None, hooks=None, timeout=None):
                self.matcher = matcher
                self.hooks = list(hooks or [])
                self.timeout = timeout

        class ResultMessage:
            result = {"status": "late"}
            is_error = False
            errors = []
            usage = {}

        worker_finished = threading.Event()

        async def blocked_query(prompt: str, options):
            # Simulate a synchronous SDK/hook path that prevents the worker's
            # own asyncio timeout from running.
            try:
                time.sleep(1.2)
                yield ResultMessage()
            finally:
                worker_finished.set()

        fake_sdk = {
            "ClaudeAgentOptions": ClaudeAgentOptions,
            "HookMatcher": HookMatcher,
            "query": blocked_query,
            "AssistantMessage": type("AssistantMessage", (), {}),
            "ResultMessage": ResultMessage,
            "ToolResultBlock": type("ToolResultBlock", (), {}),
            "ToolUseBlock": type("ToolUseBlock", (), {}),
        }
        observed = []
        context = AgentRuntimeExecutionContext(
            tool_servers={},
            message_callback=observed.append,
            env={},
        )

        started = time.monotonic()
        with (
            patch.object(adapter, "_load_claude_sdk", return_value=fake_sdk),
            patch.object(adapter, "_SDK_ISOLATED_SHUTDOWN_SECONDS", 0.01),
        ):
            result = asyncio.run(adapter.ClaudeSdkAgentRuntime().execute(self._request(), context))
        elapsed = time.monotonic() - started

        self.assertFalse(result.success)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.failover_reason, "timeout")
        self.assertLess(elapsed, 1.15)
        self.assertEqual(observed, [])
        self.assertTrue(worker_finished.wait(1.0))
        self.assertFalse(
            any(thread.name == "praxist-claude-sdk-peer" for thread in threading.enumerate())
        )

    def test_isolated_supervisor_delivers_callbacks_on_caller_event_loop(self) -> None:
        from praxist.core.runtimes import AgentRuntimeExecutionContext
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        class ClaudeAgentOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class HookMatcher:
            def __init__(self, matcher=None, hooks=None, timeout=None):
                self.matcher = matcher
                self.hooks = list(hooks or [])
                self.timeout = timeout

        class ResultMessage:
            result = {"status": "done"}
            is_error = False
            errors = []
            usage = {}

        async def query(prompt: str, options):
            yield ResultMessage()

        fake_sdk = {
            "ClaudeAgentOptions": ClaudeAgentOptions,
            "HookMatcher": HookMatcher,
            "query": query,
            "AssistantMessage": type("AssistantMessage", (), {}),
            "ResultMessage": ResultMessage,
            "ToolResultBlock": type("ToolResultBlock", (), {}),
            "ToolUseBlock": type("ToolUseBlock", (), {}),
        }
        observed: list[tuple[asyncio.AbstractEventLoop, int]] = []

        async def exercise():
            caller_loop = asyncio.get_running_loop()
            caller_thread = threading.get_ident()

            def callback(message) -> None:
                observed.append((asyncio.get_running_loop(), threading.get_ident()))

            context = AgentRuntimeExecutionContext(
                tool_servers={}, message_callback=callback, env={}
            )
            result = await adapter.ClaudeSdkAgentRuntime().execute(self._request(), context)
            return result, caller_loop, caller_thread

        with patch.object(adapter, "_load_claude_sdk", return_value=fake_sdk):
            result, caller_loop, caller_thread = asyncio.run(exercise())

        self.assertTrue(result.success)
        self.assertEqual(observed, [(caller_loop, caller_thread)])

    def test_isolated_supervisor_cancels_async_sdk_stream_on_timeout(self) -> None:
        from praxist.core.runtimes import AgentRuntimeExecutionContext
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        class ClaudeAgentOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class HookMatcher:
            def __init__(self, matcher=None, hooks=None, timeout=None):
                self.matcher = matcher
                self.hooks = list(hooks or [])
                self.timeout = timeout

        worker_finished = threading.Event()

        async def idle_query(prompt: str, options):
            try:
                await asyncio.Event().wait()
                yield SimpleNamespace(content=[])
            finally:
                worker_finished.set()

        fake_sdk = {
            "ClaudeAgentOptions": ClaudeAgentOptions,
            "HookMatcher": HookMatcher,
            "query": idle_query,
            "AssistantMessage": type("AssistantMessage", (), {}),
            "ResultMessage": type("ResultMessage", (), {}),
            "ToolResultBlock": type("ToolResultBlock", (), {}),
            "ToolUseBlock": type("ToolUseBlock", (), {}),
        }
        request = replace(self._request(), timeout_seconds=1)
        context = AgentRuntimeExecutionContext(tool_servers={}, env={})

        with (
            patch.object(adapter, "_load_claude_sdk", return_value=fake_sdk),
            patch.object(adapter, "_SDK_ISOLATED_SHUTDOWN_SECONDS", 0.5),
        ):
            result = asyncio.run(adapter.ClaudeSdkAgentRuntime().execute(request, context))

        self.assertTrue(result.timed_out)
        self.assertTrue(worker_finished.wait(0.2))
        self.assertFalse(
            any(thread.name == "praxist-claude-sdk-peer" for thread in threading.enumerate())
        )

    def test_terminal_background_update_closes_idle_sdk_stream(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        class ClaudeAgentOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class HookMatcher:
            def __init__(self, matcher=None, hooks=None, timeout=None):
                self.matcher = matcher
                self.hooks = list(hooks or [])
                self.timeout = timeout

        class TaskUpdatedMessage:
            task_id = "task-1"
            status = "completed"
            output_file = ""
            summary = "evaluation finished"

        async def terminal_then_open_query(prompt: str, options):
            yield TaskUpdatedMessage()
            await asyncio.Event().wait()

        fake_sdk = {
            "ClaudeAgentOptions": ClaudeAgentOptions,
            "HookMatcher": HookMatcher,
            "query": terminal_then_open_query,
            "AssistantMessage": type("AssistantMessage", (), {}),
            "ResultMessage": type("ResultMessage", (), {}),
            "ToolUseBlock": type("ToolUseBlock", (), {}),
        }

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_claude_sdk", return_value=fake_sdk),
            patch.object(adapter, "_SDK_STREAM_POLL_SECONDS", 0.01),
            patch.object(adapter, "_SDK_TERMINAL_TASK_IDLE_SECONDS", 0.02),
            patch.object(adapter, "_SDK_COMPLETED_TASK_GRACE_SECONDS", 0.02),
        ):
            started = time.monotonic()
            result = asyncio.run(
                adapter.ClaudeSdkAgentRuntime().execute_legacy(
                    "task",
                    adapter.LegacyClaudeRuntimeOptions(
                        name="agent",
                        allowed_tools=[],
                        workspace=Path(tmp),
                        mcp_servers={},
                        model="fake",
                        permission_mode="default",
                        runtime_timeout_seconds=5,
                    ),
                )
            )

        self.assertLess(time.monotonic() - started, 1.0)
        self.assertTrue(result.success)
        self.assertTrue(result.output["terminal_background_only"])
        self.assertEqual(
            result.output["background_tasks"],
            [
                {
                    "task_id": "task-1",
                    "status": "completed",
                    "output_file": "",
                    "summary": "evaluation finished",
                }
            ],
        )

    def test_successful_sdk_stream_closes_query_iterator(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        class ClaudeAgentOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class HookMatcher:
            def __init__(self, matcher=None, hooks=None, timeout=None):
                self.matcher = matcher
                self.hooks = list(hooks or [])
                self.timeout = timeout

        class ResultMessage:
            result = {"status": "done"}

        class CloseTrackingIterator:
            def __init__(self):
                self.sent = False
                self.closed = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.sent:
                    await asyncio.Event().wait()
                self.sent = True
                return ResultMessage()

            async def aclose(self):
                await asyncio.sleep(5.05)
                self.closed = True

        iterator = CloseTrackingIterator()
        fake_sdk = {
            "ClaudeAgentOptions": ClaudeAgentOptions,
            "HookMatcher": HookMatcher,
            "query": lambda **_kwargs: iterator,
            "AssistantMessage": type("AssistantMessage", (), {}),
            "ResultMessage": ResultMessage,
            "ToolResultBlock": type("ToolResultBlock", (), {}),
            "ToolUseBlock": type("ToolUseBlock", (), {}),
        }

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_claude_sdk", return_value=fake_sdk),
        ):
            result = asyncio.run(
                adapter.ClaudeSdkAgentRuntime().execute_legacy(
                    "task",
                    adapter.LegacyClaudeRuntimeOptions(
                        name="agent",
                        allowed_tools=[],
                        workspace=Path(tmp),
                        mcp_servers={},
                        model="fake",
                        permission_mode="default",
                        runtime_timeout_seconds=2,
                    ),
                )
            )

        self.assertTrue(result.success)
        self.assertTrue(iterator.closed)

    def test_sdk_iterator_close_timeout_is_observable(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        class ClaudeAgentOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class HookMatcher:
            def __init__(self, matcher=None, hooks=None, timeout=None):
                self.matcher = matcher
                self.hooks = list(hooks or [])
                self.timeout = timeout

        class ResultMessage:
            result = {"status": "done"}

        class BlockingCloseIterator:
            def __init__(self):
                self.sent = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.sent:
                    raise StopAsyncIteration
                self.sent = True
                return ResultMessage()

            async def aclose(self):
                await asyncio.Event().wait()

        fake_sdk = {
            "ClaudeAgentOptions": ClaudeAgentOptions,
            "HookMatcher": HookMatcher,
            "query": lambda **_kwargs: BlockingCloseIterator(),
            "AssistantMessage": type("AssistantMessage", (), {}),
            "ResultMessage": ResultMessage,
            "ToolResultBlock": type("ToolResultBlock", (), {}),
            "ToolUseBlock": type("ToolUseBlock", (), {}),
        }

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_claude_sdk", return_value=fake_sdk),
            patch.object(adapter, "_SDK_QUERY_CLOSE_SECONDS", 0.01),
            patch.object(adapter.logger, "warning") as warning,
        ):
            result = asyncio.run(
                adapter.ClaudeSdkAgentRuntime().execute_legacy(
                    "task",
                    adapter.LegacyClaudeRuntimeOptions(
                        name="agent",
                        allowed_tools=[],
                        workspace=Path(tmp),
                        mcp_servers={},
                        model="fake",
                        permission_mode="default",
                        runtime_timeout_seconds=2,
                    ),
                )
            )

        self.assertTrue(result.success)
        self.assertTrue(
            any("did not close within" in str(call.args[0]) for call in warning.call_args_list)
        )

    def test_cancellation_during_sdk_iterator_close_is_not_suppressed(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        class ClaudeAgentOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class HookMatcher:
            def __init__(self, matcher=None, hooks=None, timeout=None):
                self.matcher = matcher
                self.hooks = list(hooks or [])
                self.timeout = timeout

        class ResultMessage:
            result = {"status": "done"}

        class BlockingCloseIterator:
            def __init__(self):
                self.sent = False
                self.close_started = asyncio.Event()

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.sent:
                    raise StopAsyncIteration
                self.sent = True
                return ResultMessage()

            async def aclose(self):
                self.close_started.set()
                await asyncio.Event().wait()

        iterator = BlockingCloseIterator()
        fake_sdk = {
            "ClaudeAgentOptions": ClaudeAgentOptions,
            "HookMatcher": HookMatcher,
            "query": lambda **_kwargs: iterator,
            "AssistantMessage": type("AssistantMessage", (), {}),
            "ResultMessage": ResultMessage,
            "ToolResultBlock": type("ToolResultBlock", (), {}),
            "ToolUseBlock": type("ToolUseBlock", (), {}),
        }

        async def run_and_cancel() -> None:
            with (
                tempfile.TemporaryDirectory() as tmp,
                patch.object(adapter, "_load_claude_sdk", return_value=fake_sdk),
            ):
                task = asyncio.create_task(
                    adapter.ClaudeSdkAgentRuntime().execute_legacy(
                        "task",
                        adapter.LegacyClaudeRuntimeOptions(
                            name="agent",
                            allowed_tools=[],
                            workspace=Path(tmp),
                            mcp_servers={},
                            model="fake",
                            permission_mode="default",
                            runtime_timeout_seconds=2,
                        ),
                    )
                )
                await asyncio.wait_for(iterator.close_started.wait(), timeout=1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

        asyncio.run(run_and_cancel())

    def test_completed_background_task_waits_for_delayed_final_result(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        class ClaudeAgentOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class HookMatcher:
            def __init__(self, matcher=None, hooks=None, timeout=None):
                self.matcher = matcher
                self.hooks = list(hooks or [])
                self.timeout = timeout

        class TaskUpdatedMessage:
            task_id = "task-1"
            status = "completed"
            output_file = ""
            summary = "evaluation finished"

        class ResultMessage:
            result = {"status": "done"}

        async def delayed_result_query(prompt: str, options):
            yield TaskUpdatedMessage()
            await asyncio.sleep(0.05)
            yield ResultMessage()

        fake_sdk = {
            "ClaudeAgentOptions": ClaudeAgentOptions,
            "HookMatcher": HookMatcher,
            "query": delayed_result_query,
            "AssistantMessage": type("AssistantMessage", (), {}),
            "ResultMessage": ResultMessage,
            "ToolUseBlock": type("ToolUseBlock", (), {}),
        }

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_claude_sdk", return_value=fake_sdk),
            patch.object(adapter, "_SDK_STREAM_POLL_SECONDS", 0.005),
            patch.object(adapter, "_SDK_TERMINAL_TASK_IDLE_SECONDS", 0.01),
            patch.object(adapter, "_SDK_COMPLETED_TASK_GRACE_SECONDS", 0.2),
        ):
            result = asyncio.run(
                adapter.ClaudeSdkAgentRuntime().execute_legacy(
                    "task",
                    adapter.LegacyClaudeRuntimeOptions(
                        name="agent",
                        allowed_tools=[],
                        workspace=Path(tmp),
                        mcp_servers={},
                        model="fake",
                        permission_mode="default",
                        runtime_timeout_seconds=2,
                    ),
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(result.output["result_message"], {"status": "done"})
        self.assertNotIn("terminal_background_only", result.output)

    def test_foreground_tool_after_terminal_background_task_suspends_idle_close(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        class ClaudeAgentOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class HookMatcher:
            def __init__(self, matcher=None, hooks=None, timeout=None):
                self.matcher = matcher
                self.hooks = list(hooks or [])
                self.timeout = timeout

        class TaskUpdatedMessage:
            task_id = "task-1"
            status = "completed"
            output_file = ""
            summary = "background evaluation finished"

        class ToolUseBlock:
            name = "Bash"
            input = {"command": "run-foreground-evaluation"}

        class AssistantMessage:
            content = [ToolUseBlock()]

        class ResultMessage:
            result = {"status": "foreground-done"}

        async def foreground_after_background_query(prompt: str, options):
            yield TaskUpdatedMessage()
            yield AssistantMessage()
            await asyncio.sleep(0.05)
            yield ResultMessage()

        fake_sdk = {
            "ClaudeAgentOptions": ClaudeAgentOptions,
            "HookMatcher": HookMatcher,
            "query": foreground_after_background_query,
            "AssistantMessage": AssistantMessage,
            "ResultMessage": ResultMessage,
            "ToolUseBlock": ToolUseBlock,
        }

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_claude_sdk", return_value=fake_sdk),
            patch.object(adapter, "_SDK_STREAM_POLL_SECONDS", 0.005),
            patch.object(adapter, "_SDK_COMPLETED_TASK_GRACE_SECONDS", 0.02),
        ):
            result = asyncio.run(
                adapter.ClaudeSdkAgentRuntime().execute_legacy(
                    "task",
                    adapter.LegacyClaudeRuntimeOptions(
                        name="agent",
                        allowed_tools=[],
                        workspace=Path(tmp),
                        mcp_servers={},
                        model="fake",
                        permission_mode="default",
                        runtime_timeout_seconds=2,
                    ),
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(result.output["result_message"], {"status": "foreground-done"})
        self.assertNotIn("terminal_background_only", result.output)

    def test_terminal_background_update_does_not_rearm_during_foreground_tool(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        class ClaudeAgentOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class HookMatcher:
            def __init__(self, matcher=None, hooks=None, timeout=None):
                self.matcher = matcher
                self.hooks = list(hooks or [])
                self.timeout = timeout

        class ToolUseBlock:
            id = "foreground-1"
            name = "Bash"
            input = {"command": "run-foreground-evaluation"}

        class AssistantMessage:
            content = [ToolUseBlock()]

        class TaskUpdatedMessage:
            task_id = "task-1"
            status = "completed"
            output_file = ""
            summary = "earlier background work finished"

        class ResultMessage:
            result = {"status": "foreground-done"}

        async def terminal_update_during_foreground_query(prompt: str, options):
            yield AssistantMessage()
            yield TaskUpdatedMessage()
            await asyncio.sleep(0.05)
            yield ResultMessage()

        fake_sdk = {
            "ClaudeAgentOptions": ClaudeAgentOptions,
            "HookMatcher": HookMatcher,
            "query": terminal_update_during_foreground_query,
            "AssistantMessage": AssistantMessage,
            "ResultMessage": ResultMessage,
            "ToolResultBlock": type("ToolResultBlock", (), {}),
            "ToolUseBlock": ToolUseBlock,
        }

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_claude_sdk", return_value=fake_sdk),
            patch.object(adapter, "_SDK_STREAM_POLL_SECONDS", 0.005),
            patch.object(adapter, "_SDK_COMPLETED_TASK_GRACE_SECONDS", 0.02),
        ):
            result = asyncio.run(
                adapter.ClaudeSdkAgentRuntime().execute_legacy(
                    "task",
                    adapter.LegacyClaudeRuntimeOptions(
                        name="agent",
                        allowed_tools=[],
                        workspace=Path(tmp),
                        mcp_servers={},
                        model="fake",
                        permission_mode="default",
                        runtime_timeout_seconds=2,
                    ),
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(result.output["result_message"], {"status": "foreground-done"})

    def test_terminal_background_timer_rearms_after_foreground_tool_result(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        class ClaudeAgentOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class HookMatcher:
            def __init__(self, matcher=None, hooks=None, timeout=None):
                self.matcher = matcher
                self.hooks = list(hooks or [])
                self.timeout = timeout

        class TaskUpdatedMessage:
            task_id = "task-1"
            status = "completed"
            output_file = ""
            summary = "background work finished"

        class ToolUseBlock:
            id = "foreground-1"
            name = "Bash"
            input = {"command": "finish-foreground-work"}

        class ToolResultBlock:
            tool_use_id = "foreground-1"
            content = "done"

        class AssistantMessage:
            content = [ToolUseBlock()]

        class ToolResultMessage:
            content = [ToolResultBlock()]

        async def terminal_then_foreground_query(prompt: str, options):
            yield TaskUpdatedMessage()
            yield AssistantMessage()
            await asyncio.sleep(0.04)
            yield ToolResultMessage()
            await asyncio.Event().wait()

        fake_sdk = {
            "ClaudeAgentOptions": ClaudeAgentOptions,
            "HookMatcher": HookMatcher,
            "query": terminal_then_foreground_query,
            "AssistantMessage": AssistantMessage,
            "ResultMessage": type("ResultMessage", (), {}),
            "ToolResultBlock": ToolResultBlock,
            "ToolUseBlock": ToolUseBlock,
        }

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_claude_sdk", return_value=fake_sdk),
            patch.object(adapter, "_SDK_STREAM_POLL_SECONDS", 0.005),
            patch.object(adapter, "_SDK_COMPLETED_TASK_GRACE_SECONDS", 0.02),
        ):
            started = time.monotonic()
            result = asyncio.run(
                adapter.ClaudeSdkAgentRuntime().execute_legacy(
                    "task",
                    adapter.LegacyClaudeRuntimeOptions(
                        name="agent",
                        allowed_tools=[],
                        workspace=Path(tmp),
                        mcp_servers={},
                        model="fake",
                        permission_mode="default",
                        runtime_timeout_seconds=2,
                    ),
                )
            )

        self.assertLess(time.monotonic() - started, 1.0)
        self.assertTrue(result.success)

    def test_stop_check_remains_reachable_while_sdk_stream_is_idle(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        class ClaudeAgentOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class HookMatcher:
            def __init__(self, matcher=None, hooks=None, timeout=None):
                self.matcher = matcher
                self.hooks = list(hooks or [])
                self.timeout = timeout

        async def idle_query(prompt: str, options):
            await asyncio.Event().wait()
            yield SimpleNamespace(content=[])

        fake_sdk = {
            "ClaudeAgentOptions": ClaudeAgentOptions,
            "HookMatcher": HookMatcher,
            "query": idle_query,
            "AssistantMessage": type("AssistantMessage", (), {}),
            "ResultMessage": type("ResultMessage", (), {}),
            "ToolUseBlock": type("ToolUseBlock", (), {}),
        }
        polls = 0

        def stop_check() -> bool:
            nonlocal polls
            polls += 1
            return polls >= 2

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_claude_sdk", return_value=fake_sdk),
            patch.object(adapter, "_SDK_STREAM_POLL_SECONDS", 0.01),
        ):
            started = time.monotonic()
            result = asyncio.run(
                adapter.ClaudeSdkAgentRuntime().execute_legacy(
                    "task",
                    adapter.LegacyClaudeRuntimeOptions(
                        name="agent",
                        allowed_tools=[],
                        workspace=Path(tmp),
                        mcp_servers={},
                        model="fake",
                        permission_mode="default",
                        stop_check_fn=stop_check,
                        runtime_timeout_seconds=5,
                    ),
                )
            )

        self.assertLess(time.monotonic() - started, 1.0)
        self.assertGreaterEqual(polls, 2)
        self.assertTrue(result.success)
        self.assertEqual(result.output, {"text_outputs": [], "tool_uses": []})

    def test_private_or_blank_sdk_results_remain_empty_for_retry_accounting(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import adapter
        from praxist.plugins.workflow_stages.research_loop.backend import agent

        class ResultMessage:
            def __init__(self, result):
                self.result = result

        class ThinkingBlock:
            thinking = "private reasoning"

        class AssistantMessage:
            content = [ThinkingBlock()]

        for result_message in (None, "", "   ", {}, []):
            output = adapter.extract_legacy_output([ResultMessage(result_message)])
            self.assertTrue(
                agent.AutonomousAgentLoop._session_was_empty(
                    agent.AgentResult(
                        success=True,
                        output=output,
                        duration=0,
                        iteration_count=0,
                    )
                )
            )

        thinking_output = adapter.extract_legacy_output([AssistantMessage()])
        self.assertEqual(thinking_output["text_outputs"], [])
        self.assertEqual(thinking_output["thinking_outputs"], ["private reasoning"])
        self.assertTrue(
            agent.AutonomousAgentLoop._session_was_empty(
                agent.AgentResult(
                    success=True,
                    output=thinking_output,
                    duration=0,
                    iteration_count=0,
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
