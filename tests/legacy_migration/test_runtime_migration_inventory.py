from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from praxist.core.cache import build_cache_policy
from praxist.core.credentials import CredentialRef
from praxist.core.modeling import default_model_profile, provider_for_ref
from praxist.core.protocol import AgentRunRequest, EnvPolicy, ToolPermissionSet
from praxist.core.runtimes import AgentRuntimeExecutionContext
from praxist.plugins.agent_runtimes.claude_sdk.adapter import (
    ClaudeSdkAgentRuntime,
    LegacyAgentResult,
    claude_setting_sources_from_env,
)
from praxist.plugins.workflow_stages.research_loop.backend.agent import BaseAgent


class Step12AgentRuntimeMigrationTest(unittest.TestCase):
    def test_claude_runtime_native_execute_normalizes_legacy_result(self) -> None:
        observed: dict[str, object] = {}

        async def fake_execute_legacy(self, task, options):
            observed["task"] = task
            observed["name"] = options.name
            observed["model"] = options.model
            observed["env"] = options.env
            return LegacyAgentResult(
                success=True,
                output={
                    "text_outputs": ["hello"],
                    "tool_uses": [{"tool": "Bash", "input": {"command": "echo ok"}}],
                },
                duration=1.25,
                iteration_count=1,
            )

        request = self._agent_request()
        with patch.object(ClaudeSdkAgentRuntime, "execute_legacy", fake_execute_legacy):
            result = asyncio.run(
                ClaudeSdkAgentRuntime().execute(
                    request,
                    AgentRuntimeExecutionContext(
                        env={"ANTHROPIC_AUTH_TOKEN": "token"},
                    ),
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(observed["task"], "do work")
        self.assertEqual(observed["name"], "peer0")
        self.assertEqual(observed["model"], "anthropic/claude-opus-4.7")
        self.assertEqual(observed["env"], {"ANTHROPIC_AUTH_TOKEN": "token"})
        self.assertEqual(
            [event.type for event in result.events],
            ["agent_run_started", "assistant_text", "tool_use", "final_result"],
        )
        self.assertEqual(result.tool_uses[0].tool_name, "Bash")
        final_payload = result.events[-1].payload
        self.assertEqual(final_payload["legacy_output"]["text_outputs"], ["hello"])
        self.assertEqual(final_payload["iteration_count"], 1)

    def test_base_agent_emits_replayable_agent_run_request(self) -> None:
        async def fake_execute_legacy(self, task, options):
            return LegacyAgentResult(
                success=True,
                output={
                    "text_outputs": ["done"],
                    "tool_uses": [{"tool": "Read", "input": {"file_path": "x"}}],
                },
                duration=0.5,
                iteration_count=1,
            )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_agent_request"
            run_dir.mkdir()
            env = {
                "PRAXIST_RUN_DIR": str(run_dir),
                "PRAXIST_RUN_ID": run_dir.name,
                "PRAXIST_MODEL_PROVIDER_REF": "model_provider:openrouter",
                "PRAXIST_MODEL_CREDENTIAL_KEY_ID": "openrouter:env:abc123",
                "PRAXIST_AGENT_RUNTIME_REF": "agent_runtime:claude_sdk",
                "PRAXIST_BUDGET_GRANT_ID": "grant_test",
                "PRAXIST_CREDENTIAL_MODE": "single",
                "PRAXIST_TASK_PROJECT_PATH": "/tmp/external-task",
                "ANTHROPIC_BASE_URL": "https://openrouter.ai/api/v1",
                "ANTHROPIC_AUTH_TOKEN": "openrouter-token",
                "ANTHROPIC_API_KEY": "should-not-be-exposed",
                "OPENROUTER_API_KEY": "openrouter-native-token",
            }
            with patch.object(ClaudeSdkAgentRuntime, "execute_legacy", fake_execute_legacy):
                with patch.dict(os.environ, env, clear=False):
                    result = asyncio.run(
                        BaseAgent(
                            name="peer0",
                            allowed_tools=["Read"],
                            workspace=Path.cwd(),
                            mcp_servers={"memory-tools": object()},
                            model="anthropic/claude-opus-4.7",
                        ).execute("inspect")
                    )

            self.assertTrue(result.success)
            self.assertEqual(result.output["text_outputs"], ["done"])
            events = [
                json.loads(line) for line in (run_dir / "trajectory.jsonl").read_text().splitlines()
            ]
            started = next(event for event in events if event["kind"] == "agent.run_started")
            request = started["payload"]["request"]
            self.assertEqual(request["agent_runtime_ref"], "agent_runtime:claude_sdk")
            self.assertEqual(request["system_prompt_ref"]["kind"], "legacy_inline_system_prompt")
            self.assertIn(
                "Praxist autonomous research agent",
                request["runtime_options"]["system_prompt"],
            )
            self.assertEqual(request["model_call"]["provider_ref"], "model_provider:openrouter")
            self.assertEqual(
                request["model_call"]["credential_ref"]["key_id"], "openrouter:env:abc123"
            )
            self.assertEqual(request["tool_permissions"]["mode"], "allow_list")
            self.assertEqual(request["tool_permissions"]["allowed_tools"], ["Read"])
            exposed_env_keys = set(request["env_policy"]["exposed_env_keys"])
            self.assertIn("ANTHROPIC_AUTH_TOKEN", exposed_env_keys)
            self.assertIn("ANTHROPIC_BASE_URL", exposed_env_keys)
            self.assertIn("OPENROUTER_API_KEY", exposed_env_keys)
            self.assertIn("PRAXIST_TASK_PROJECT_PATH", exposed_env_keys)
            self.assertIn("PRAXIST_MODEL_PROVIDER_REF", exposed_env_keys)
            self.assertIn("PRAXIST_AGENT_RUNTIME_REF", exposed_env_keys)
            self.assertNotIn("ANTHROPIC_API_KEY", exposed_env_keys)
            self.assertEqual(
                request["tool_servers"],
                [{"server_name": "memory-tools", "transport": "legacy_inprocess"}],
            )
            finished = next(event for event in events if event["kind"] == "agent.run_finished")
            self.assertIn("final_result", finished["payload"]["runtime_event_types"])

    def test_claude_runtime_settings_scope_defaults_to_local_only(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(claude_setting_sources_from_env(), ["local"])
        with patch.dict(
            os.environ, {"PRAXIST_CLAUDE_SETTING_SOURCES": "project,user,local,project"}, clear=True
        ):
            self.assertEqual(
                claude_setting_sources_from_env(),
                ["project", "user", "local"],
            )
        with patch.dict(os.environ, {"PRAXIST_CLAUDE_SETTING_SOURCES": "none"}, clear=True):
            self.assertEqual(claude_setting_sources_from_env(), [])

    def _agent_request(self) -> AgentRunRequest:
        credential = CredentialRef(
            scope="model_provider",
            provider="openrouter",
            target_ref="model_provider:openrouter",
            key_id="openrouter:env:abc123",
            source="test",
        )
        profile = default_model_profile(
            "model_provider:openrouter", model="anthropic/claude-opus-4.7"
        )
        model_call = provider_for_ref("model_provider:openrouter").build_call(
            profile, credential_ref=credential
        )
        return AgentRunRequest(
            request_id="peer0",
            run_id="run_test",
            stage_id="research_loop",
            role_ref="task_role:peer",
            agent_runtime_ref="agent_runtime:claude_sdk",
            prompt_ref={"kind": "test", "sha256": "sha256:prompt", "text": "do work"},
            system_prompt_ref=None,
            cwd=str(Path.cwd()),
            model_profile_ref=profile.profile_id,
            model_call=model_call,
            tool_permissions=ToolPermissionSet(mode="allow_list", allowed_tools=["Bash"]),
            tool_servers=[],
            env_policy=EnvPolicy(),
            credential_ref=credential,
            credential_mode="single",
            budget_grant_id="grant_test",
            artifact_scope="run",
            timeout_seconds=30,
            cache_policy=build_cache_policy(frozen_prefix_parts={"task": "native"}),
            runtime_options={"permission_mode": "acceptEdits"},
        )


if __name__ == "__main__":
    unittest.main()
