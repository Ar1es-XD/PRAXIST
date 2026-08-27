"""Runtime-neutral sandbox intent to official Codex SDK setting tests."""

from __future__ import annotations

import unittest
from typing import Any

from praxist.core.protocol import (
    AgentRunRequest,
    CachePolicy,
    EnvPolicy,
    ModelCallSpec,
    ToolPermissionSet,
)
from praxist.plugins.agent_runtimes.codex_sdk._sandbox import (
    CodexSandboxSettings,
    sandbox_settings,
)


def _request(
    sandbox_intent: dict[str, Any] | None = None,
    *,
    permissions: ToolPermissionSet | None = None,
) -> AgentRunRequest:
    return AgentRunRequest(
        request_id="request-1",
        run_id="run-1",
        stage_id="stage-1",
        role_ref=None,
        agent_runtime_ref="agent_runtime:codex_sdk",
        prompt_ref={"text": "Inspect the task."},
        system_prompt_ref=None,
        cwd="/tmp",
        model_profile_ref="model_profile:default",
        model_call=ModelCallSpec(
            profile_id="default",
            provider_ref="model_provider:openai",
            api_format="responses",
            model="gpt-5",
            parameters={},
            credential_ref=None,
        ),
        tool_permissions=permissions or ToolPermissionSet(),
        tool_servers=[],
        env_policy=EnvPolicy(),
        credential_ref=None,
        credential_mode="env",
        budget_grant_id=None,
        artifact_scope="run",
        timeout_seconds=60,
        cache_policy=CachePolicy(mode="deterministic_no_cache", frozen_prefix_hash=None),
        runtime_options={"sandbox_intent": sandbox_intent} if sandbox_intent else {},
    )


class SandboxSettingsTest(unittest.TestCase):
    def test_default_maps_to_noninteractive_full_access(self) -> None:
        self.assertEqual(
            sandbox_settings(_request()),
            CodexSandboxSettings(
                approval_mode="deny_all",
                sandbox="full_access",
                config={
                    "features": {"shell_tool": True, "multi_agent": True},
                    "include_apply_patch_tool": True,
                    "tools": {"view_image": True},
                    "web_search": "live",
                },
            ),
        )

    def test_read_only_maps_without_workspace_network_override(self) -> None:
        settings = sandbox_settings(
            _request({"filesystem": "read_only", "network": "off", "approval": "auto"})
        )
        self.assertEqual(settings.approval_mode, "deny_all")
        self.assertEqual(settings.sandbox, "read_only")
        self.assertEqual(settings.config["web_search"], "disabled")
        self.assertTrue(settings.config["features"]["shell_tool"])  # type: ignore[index]

    def test_workspace_write_maps_network_access_both_ways(self) -> None:
        for network, expected in (("on", True), ("off", False)):
            with self.subTest(network=network):
                settings = sandbox_settings(
                    _request(
                        {
                            "filesystem": "workspace_write",
                            "network": network,
                            "approval": "auto",
                        }
                    )
                )
                self.assertEqual(settings.sandbox, "workspace_write")
                self.assertEqual(
                    settings.config["sandbox_workspace_write"],
                    {"network_access": expected},
                )
                self.assertEqual(
                    settings.config["web_search"],
                    "live" if expected else "disabled",
                )

    def test_interactive_approval_is_rejected_in_headless_runtime(self) -> None:
        for approval in ("on_risk", "always_ask"):
            with (
                self.subTest(approval=approval),
                self.assertRaisesRegex(ValueError, "cannot honor interactive"),
            ):
                sandbox_settings(
                    _request({"filesystem": "full", "network": "on", "approval": approval})
                )

    def test_full_access_cannot_claim_network_isolation(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot disable network"):
            sandbox_settings(_request({"filesystem": "full", "network": "off", "approval": "auto"}))

    def test_mcp_only_allow_list_disables_every_builtin_tool_category(self) -> None:
        settings = sandbox_settings(
            _request(
                permissions=ToolPermissionSet(
                    mode="allow_list",
                    allowed_tools=["mcp__evaluation_tools__evaluate"],
                )
            )
        )
        self.assertEqual(
            settings.config,
            {
                "features": {"shell_tool": False, "multi_agent": False},
                "include_apply_patch_tool": False,
                "tools": {"view_image": False},
                "web_search": "disabled",
            },
        )

    def test_builtin_allow_list_selects_only_requested_categories(self) -> None:
        settings = sandbox_settings(
            _request(
                {"filesystem": "workspace_write", "network": "on", "approval": "auto"},
                permissions=ToolPermissionSet(
                    mode="allow_list",
                    allowed_tools=["Bash", "WebSearch"],
                ),
            )
        )
        self.assertEqual(
            settings.config["features"],
            {"shell_tool": True, "multi_agent": False},
        )
        self.assertFalse(settings.config["include_apply_patch_tool"])
        self.assertEqual(settings.config["web_search"], "live")

    def test_invalid_intent_axes_fail_before_sdk_execution(self) -> None:
        invalid = [
            {"filesystem": "outside", "network": "on", "approval": "auto"},
            {"filesystem": "full", "network": "sometimes", "approval": "auto"},
            {"filesystem": "full", "network": "on", "approval": "never"},
        ]
        for intent in invalid:
            with self.subTest(intent=intent), self.assertRaises(ValueError):
                sandbox_settings(_request(intent))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
