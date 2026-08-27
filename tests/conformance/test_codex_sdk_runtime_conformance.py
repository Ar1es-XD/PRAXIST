"""Offline conformance for the official Codex SDK AgentRuntime plugin."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from praxist.core.protocol import (
    AgentRunRequest,
    CachePolicy,
    EnvPolicy,
    ModelCallSpec,
    ToolPermissionSet,
)
from praxist.core.runtimes import AgentRuntimeExecutionContext, runtime_for_ref
from praxist.plugins.agent_runtimes.codex_sdk import adapter


@dataclass(frozen=True)
class _Payload:
    data: dict[str, Any]

    def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
        return dict(self.data)


@dataclass(frozen=True)
class _Notification:
    method: str
    payload: _Payload


def _notification(method: str, payload: dict[str, Any]) -> _Notification:
    return _Notification(method=method, payload=_Payload(payload))


class _Config:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _Turn:
    def __init__(self, prompt: str, turn_options: dict[str, Any]) -> None:
        self.prompt = prompt
        self.turn_options = turn_options

    def stream(self) -> Any:
        return iter(
            [
                _notification(
                    "item/completed",
                    {
                        "item": {
                            "id": "message-1",
                            "type": "agentMessage",
                            "text": "SDK conformance complete",
                        }
                    },
                ),
                _notification(
                    "item/started",
                    {
                        "startedAtMs": 10,
                        "item": {
                            "id": "tool-1",
                            "type": "mcpToolCall",
                            "server": "evaluation-tools",
                            "tool": "inspect",
                            "arguments": {"candidate": "a"},
                            "status": "inProgress",
                        },
                    },
                ),
                _notification(
                    "item/completed",
                    {
                        "completedAtMs": 20,
                        "item": {
                            "id": "tool-1",
                            "type": "mcpToolCall",
                            "server": "evaluation-tools",
                            "tool": "inspect",
                            "arguments": {"candidate": "a"},
                            "status": "completed",
                        },
                    },
                ),
                _notification(
                    "thread/tokenUsage/updated",
                    {
                        "tokenUsage": {
                            "total": {
                                "inputTokens": 8,
                                "cachedInputTokens": 2,
                                "outputTokens": 4,
                                "reasoningOutputTokens": 1,
                                "totalTokens": 12,
                            }
                        }
                    },
                ),
                _notification(
                    "turn/completed",
                    {"turn": {"id": "turn-1", "status": "completed", "items": []}},
                ),
            ]
        )

    def interrupt(self) -> None:
        return None


class _Thread:
    def __init__(self, harness: _SdkHarness, options: dict[str, Any]) -> None:
        self.harness = harness
        self.options = options

    def turn(self, prompt: str, **kwargs: Any) -> _Turn:
        self.harness.prompt = prompt
        self.harness.turn_options = kwargs
        return _Turn(prompt, kwargs)


class _Client:
    def __init__(self, harness: _SdkHarness, config: _Config) -> None:
        self.harness = harness
        self.config = config
        self.closed = False

    def thread_start(self, **kwargs: Any) -> _Thread:
        self.harness.thread_options = kwargs
        return _Thread(self.harness, kwargs)

    def close(self) -> None:
        self.closed = True


class _ApprovalMode:
    deny_all = "deny_all"
    auto_review = "auto_review"


class _Sandbox:
    read_only = "read_only"
    workspace_write = "workspace_write"
    full_access = "full_access"


class _ReasoningEffort:
    xhigh = "xhigh"


class _SdkHarness:
    def __init__(self) -> None:
        self.prompt: str | None = None
        self.turn_options: dict[str, Any] | None = None
        self.thread_options: dict[str, Any] | None = None
        self.client: _Client | None = None

    def sdk(self) -> dict[str, Any]:
        harness = self

        def codex(config: _Config) -> _Client:
            harness.client = _Client(harness, config)
            return harness.client

        return {
            "ApprovalMode": _ApprovalMode,
            "Codex": codex,
            "CodexConfig": _Config,
            "ReasoningEffort": _ReasoningEffort,
            "Sandbox": _Sandbox,
        }


def _request(cwd: str) -> AgentRunRequest:
    return AgentRunRequest(
        request_id="codex-sdk-conformance",
        run_id="run-conformance",
        stage_id="stage-conformance",
        role_ref="role:peer",
        agent_runtime_ref="agent_runtime:codex_sdk",
        prompt_ref={"text": "Use the canonical prompt field."},
        system_prompt_ref=None,
        cwd=cwd,
        model_profile_ref="model_profile:default",
        model_call=ModelCallSpec(
            profile_id="default",
            provider_ref="model_provider:openai",
            api_format="responses",
            model="gpt-5",
            parameters={},
            credential_ref=None,
        ),
        tool_permissions=ToolPermissionSet(),
        tool_servers=[],
        env_policy=EnvPolicy(),
        credential_ref=None,
        credential_mode="env",
        budget_grant_id=None,
        artifact_scope="run",
        timeout_seconds=30,
        cache_policy=CachePolicy(mode="deterministic_no_cache", frozen_prefix_hash=None),
        runtime_options={"run_dir": cwd},
    )


class CodexSdkRuntimeConformanceTest(unittest.IsolatedAsyncioTestCase):
    async def test_plugin_executes_mocked_official_app_server_contract(self) -> None:
        harness = _SdkHarness()
        runtime = runtime_for_ref("agent_runtime:codex_sdk")
        self.addAsyncCleanup(runtime.aclose)
        observed: list[Any] = []

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
        ):
            result = await runtime.execute(
                _request(tmp),
                AgentRuntimeExecutionContext(
                    env={"OPENAI_API_KEY": "offline-test-key"},
                    message_callback=observed.append,
                ),
            )

        self.assertTrue(result.success, result.error)
        self.assertEqual(result.terminal_status, "completed")
        self.assertEqual(result.usage["total_tokens"], 12.0)
        self.assertEqual(harness.prompt, "Use the canonical prompt field.")
        self.assertIsNotNone(harness.thread_options)
        self.assertEqual(harness.thread_options["model_provider"], "openai")
        self.assertEqual(
            [event.type for event in result.events],
            [
                "agent_run_started",
                "assistant_text",
                "tool_use",
                "tool_result",
                "usage",
                "final_result",
            ],
        )
        self.assertEqual(
            [event.type for event in observed], [event.type for event in result.events]
        )
        self.assertEqual(result.tool_uses[0].server_name, "evaluation-tools")
        self.assertEqual(result.tool_uses[0].tool_name, "inspect")

    async def test_plugin_failure_is_normalized_without_provider_objects(self) -> None:
        runtime = runtime_for_ref("agent_runtime:codex_sdk")
        self.addAsyncCleanup(runtime.aclose)
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=RuntimeError("SDK unavailable")),
        ):
            result = await runtime.execute(
                _request(tmp),
                AgentRuntimeExecutionContext(env={"OPENAI_API_KEY": "offline-test-key"}),
            )

        self.assertFalse(result.success)
        self.assertEqual(result.failover_reason, "runtime_error")
        self.assertIn("SDK unavailable", result.error or "")
        self.assertTrue(all(isinstance(event.payload, dict) for event in result.events))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
