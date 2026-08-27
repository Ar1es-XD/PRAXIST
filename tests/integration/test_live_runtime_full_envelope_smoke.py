"""Opt-in live full-envelope runtime smoke tests for production providers.

These tests are skipped by default so the normal suite remains offline. Run
with ``PRAXIST_LIVE_RUNTIME_SMOKE=1`` plus the relevant provider keys after runtime
or provider changes that need end-to-end validation.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from dataclasses import replace

from praxist.core.credentials import CredentialRef
from praxist.core.protocol import (
    AgentRunRequest,
    AgentRunResult,
    CachePolicy,
    EnvPolicy,
    ModelCallSpec,
    ToolPermissionSet,
)
from praxist.core.runtimes import AgentRuntimeExecutionContext


def _live_enabled() -> bool:
    return os.environ.get("PRAXIST_LIVE_RUNTIME_SMOKE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _chatgpt_live_enabled() -> bool:
    return os.environ.get("PRAXIST_LIVE_CHATGPT_SMOKE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _request(
    *,
    runtime_ref: str,
    provider_ref: str,
    model: str,
    cwd: str,
    key_id: str,
    credential_source: str = "env",
) -> AgentRunRequest:
    return AgentRunRequest(
        request_id=f"live-{runtime_ref.rsplit(':', 1)[-1]}-{provider_ref.rsplit(':', 1)[-1]}",
        run_id="live-runtime-smoke",
        stage_id="research_loop",
        role_ref="role:smoke",
        agent_runtime_ref=runtime_ref,
        prompt_ref={
            "text": (
                "Reply with one short sentence, then use one available shell/tool surface "
                "to inspect the current directory. Do not modify files."
            )
        },
        system_prompt_ref=None,
        cwd=cwd,
        model_profile_ref="model_profile:live_smoke",
        model_call=ModelCallSpec(
            profile_id="live_smoke",
            provider_ref=provider_ref,
            api_format="responses" if runtime_ref == "agent_runtime:codex_sdk" else "messages",
            model=model,
            parameters={},
            credential_ref=None,
        ),
        tool_permissions=ToolPermissionSet(
            mode="allow_list",
            allowed_tools=["Read", "Bash", "mcp__system-tools__run_summary"],
        ),
        tool_servers=[
            {"server_name": "system-tools", "transport": "stdio", "tool_names": ["run_summary"]}
        ],
        env_policy=EnvPolicy(),
        credential_ref=CredentialRef(
            scope="model_provider",
            provider=provider_ref.rsplit(":", 1)[-1],
            target_ref=provider_ref,
            key_id=key_id,
            source=credential_source,
        ),
        credential_mode="env",
        budget_grant_id="live-smoke",
        artifact_scope="run",
        timeout_seconds=180,
        cache_policy=CachePolicy(mode="runtime_auto_cache", frozen_prefix_hash=None),
        runtime_options={
            "provider_base_url": os.environ.get("OPENAI_BASE_URL", ""),
            "permission_mode": "default",
            "run_dir": cwd,
            "system_prompt": "You are running a Praxist runtime smoke test.",
        },
    )


async def _execute_codex_live(
    request: AgentRunRequest,
    *,
    env: dict[str, str],
) -> AgentRunResult:
    """Execute the production SDK path and always release app-server state."""

    from praxist.plugins.agent_runtimes.codex_sdk.adapter import CodexSdkRuntime

    runtime = CodexSdkRuntime()
    try:
        return await runtime.execute(
            request,
            AgentRuntimeExecutionContext(env=env),
        )
    finally:
        await runtime.aclose()


@unittest.skipUnless(_live_enabled(), "set PRAXIST_LIVE_RUNTIME_SMOKE=1 to run live provider smoke")
class LiveRuntimeFullEnvelopeSmokeTest(unittest.TestCase):
    def test_codex_sdk_chatgpt_subscription_full_envelope(self) -> None:
        if not _chatgpt_live_enabled():
            self.skipTest("set PRAXIST_LIVE_CHATGPT_SMOKE=1 to use saved ChatGPT auth")
        from praxist.plugins.agent_runtimes.codex_sdk._auth import (
            chatgpt_credential_key_id,
            operator_codex_home,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = asyncio.run(
                _execute_codex_live(
                    _request(
                        runtime_ref="agent_runtime:codex_sdk",
                        provider_ref="model_provider:openai_compatible",
                        model=os.environ.get("PRAXIST_LIVE_OPENAI_MODEL", "gpt-5.4-mini"),
                        cwd=tmp,
                        key_id=chatgpt_credential_key_id(operator_codex_home()),
                        credential_source="runtime_session",
                    ),
                    env={"PRAXIST_RUN_DIR": tmp},
                )
            )
        self.assertTrue(result.events)
        self.assertTrue(result.success, f"Codex SDK ChatGPT live smoke failed: {result.error}")

    def test_codex_sdk_chatgpt_non_strict_schema_preflight(self) -> None:
        if not _chatgpt_live_enabled():
            self.skipTest("set PRAXIST_LIVE_CHATGPT_SMOKE=1 to use saved ChatGPT auth")
        from praxist.plugins.agent_runtimes.codex_sdk._auth import (
            chatgpt_credential_key_id,
            operator_codex_home,
        )

        with tempfile.TemporaryDirectory() as tmp:
            request = _request(
                runtime_ref="agent_runtime:codex_sdk",
                provider_ref="model_provider:openai_compatible",
                model=os.environ.get("PRAXIST_LIVE_OPENAI_MODEL", "gpt-5.6-luna"),
                cwd=tmp,
                key_id=chatgpt_credential_key_id(operator_codex_home()),
                credential_source="runtime_session",
            )
            request = replace(
                request,
                prompt_ref={"text": 'Return exactly this JSON object: {"ok": true}'},
                tool_permissions=ToolPermissionSet(),
                tool_servers=[],
                runtime_options={
                    **request.runtime_options,
                    "output_schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                    },
                },
            )
            result = asyncio.run(_execute_codex_live(request, env={"PRAXIST_RUN_DIR": tmp}))

        self.assertTrue(result.events)
        self.assertTrue(result.success, f"Codex SDK schema preflight failed: {result.error}")

    def test_codex_sdk_deepseek_alias_full_envelope(self) -> None:
        if not os.environ.get("DEEPSEEK_API_KEY"):
            self.skipTest("DEEPSEEK_API_KEY is required")

        with tempfile.TemporaryDirectory() as tmp:
            request = _request(
                runtime_ref="agent_runtime:codex_sdk",
                provider_ref="model_provider:deepseek_alias",
                model=os.environ.get("PRAXIST_LIVE_DEEPSEEK_MODEL", "deepseek-v4-pro"),
                cwd=tmp,
                key_id="deepseek:env:DEEPSEEK_API_KEY",
            )
            request = replace(
                request,
                runtime_options={**request.runtime_options, "reasoning_effort": "max"},
            )
            result = asyncio.run(
                _execute_codex_live(
                    request,
                    env={
                        "DEEPSEEK_API_KEY": os.environ["DEEPSEEK_API_KEY"],
                        "PRAXIST_RUN_DIR": tmp,
                    },
                )
            )
        self.assertTrue(result.events)
        self.assertTrue(result.success, "Codex SDK DeepSeek live smoke failed")
        event_types = {event.type for event in result.events}
        self.assertIn("reasoning", event_types)
        self.assertIn("tool_use", event_types)

    def test_codex_sdk_openrouter_full_envelope(self) -> None:
        if not os.environ.get("OPENROUTER_API_KEY"):
            self.skipTest("OPENROUTER_API_KEY is required")

        with tempfile.TemporaryDirectory() as tmp:
            result = asyncio.run(
                _execute_codex_live(
                    _request(
                        runtime_ref="agent_runtime:codex_sdk",
                        provider_ref="model_provider:openrouter",
                        model=os.environ.get(
                            "PRAXIST_LIVE_OPENROUTER_MODEL", "deepseek/deepseek-v4-pro"
                        ),
                        cwd=tmp,
                        key_id="openrouter:env:OPENROUTER_API_KEY",
                    ),
                    env={
                        "OPENROUTER_API_KEY": os.environ["OPENROUTER_API_KEY"],
                        "PRAXIST_RUN_DIR": tmp,
                    },
                )
            )
        self.assertTrue(result.events)
        self.assertTrue(result.success, "Codex SDK OpenRouter live smoke failed")

    def test_claude_sdk_openrouter_full_envelope(self) -> None:
        if not os.environ.get("OPENROUTER_API_KEY"):
            self.skipTest("OPENROUTER_API_KEY is required")
        from praxist.plugins.agent_runtimes.claude_sdk.adapter import ClaudeSdkAgentRuntime

        with tempfile.TemporaryDirectory() as tmp:
            request = _request(
                runtime_ref="agent_runtime:claude_sdk",
                provider_ref="model_provider:openrouter",
                model=os.environ.get(
                    "PRAXIST_LIVE_CLAUDE_OPENROUTER_MODEL", "anthropic/claude-opus-4.7"
                ),
                cwd=tmp,
                key_id="openrouter:env:OPENROUTER_API_KEY",
            )
            context = AgentRuntimeExecutionContext(
                env={
                    "ANTHROPIC_AUTH_TOKEN": os.environ["OPENROUTER_API_KEY"],
                    "ANTHROPIC_BASE_URL": os.environ.get(
                        "ANTHROPIC_BASE_URL", "https://openrouter.ai/api"
                    ),
                    "PRAXIST_RUN_DIR": tmp,
                },
            )
            result = asyncio.run(ClaudeSdkAgentRuntime().execute(request, context))
        self.assertTrue(result.events)
        self.assertTrue(result.success, "Claude SDK OpenRouter live smoke failed")

    def test_claude_sdk_deepseek_alias_full_envelope(self) -> None:
        if not os.environ.get("DEEPSEEK_API_KEY"):
            self.skipTest("DEEPSEEK_API_KEY is required")
        from praxist.plugins.agent_runtimes.claude_sdk.adapter import ClaudeSdkAgentRuntime

        with tempfile.TemporaryDirectory() as tmp:
            request = _request(
                runtime_ref="agent_runtime:claude_sdk",
                provider_ref="model_provider:deepseek_alias",
                model=os.environ.get(
                    "PRAXIST_LIVE_DEEPSEEK_MODEL",
                    "deepseek-v4-pro[1m]",
                ),
                cwd=tmp,
                key_id="deepseek:env:DEEPSEEK_API_KEY",
            )
            request = replace(
                request,
                runtime_options={**request.runtime_options, "reasoning_effort": "max"},
            )
            context = AgentRuntimeExecutionContext(
                env={
                    "ANTHROPIC_AUTH_TOKEN": os.environ["DEEPSEEK_API_KEY"],
                    "ANTHROPIC_BASE_URL": os.environ.get(
                        "DEEPSEEK_ANTHROPIC_BASE_URL",
                        "https://api.deepseek.com/anthropic",
                    ),
                    "PRAXIST_RUN_DIR": tmp,
                },
            )
            result = asyncio.run(ClaudeSdkAgentRuntime().execute(request, context))
        self.assertTrue(result.events)
        self.assertTrue(result.success, "Claude SDK DeepSeek live smoke failed")
        final = next(event for event in result.events if event.type == "final_result")
        legacy_output = final.payload["legacy_output"]
        self.assertTrue(legacy_output.get("thinking_outputs"))
        self.assertTrue(result.tool_uses)
