"""Regression: ``_build_agent_run_request`` must put the task body into ``prompt_ref['text']``.

The codex_sdk runtime materializes the prompt by reading
``request.prompt_ref`` for one of ``system_prompt`` / ``user_prompt`` /
``prompt`` / ``text`` and writing the concatenation to the codex
subprocess stdin.  When the field is absent the subprocess sees an
empty stdin and codex 0.131 aborts with ``No prompt provided via
stdin.``.

Both construction branches inside ``_build_agent_run_request``
(legacy_inline_prompt and prompt_layout_v1) must therefore carry the
rendered task text.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from praxist.core.prompt_layout import sha256_text
from praxist.plugins.workflow_stages.research_loop.backend import agent


class PromptRefCarriesTaskTextTest(unittest.TestCase):
    """``prompt_ref['text']`` is populated on both construction branches."""

    def test_legacy_inline_prompt_branch_carries_task_text(self) -> None:
        """No ``prompt_layout_manifest`` → legacy_inline_prompt kind, text present."""
        with tempfile.TemporaryDirectory() as tmp:
            base = agent.BaseAgent(
                name="peer_agent",
                allowed_tools=["Read"],
                workspace=Path(tmp),
                mcp_servers={},
                model="fake-model",
            )
            with patch.dict(
                "os.environ",
                {"PRAXIST_AGENT_RUNTIME_REF": "agent_runtime:codex_sdk"},
                clear=False,
            ):
                request = base._build_agent_run_request("rendered task body", {})
        self.assertEqual(request.prompt_ref["kind"], "legacy_inline_prompt")
        self.assertEqual(request.prompt_ref["text"], "rendered task body")

    def test_prompt_layout_v1_branch_carries_task_text(self) -> None:
        """``prompt_layout_manifest`` present → prompt_layout_v1 kind, text present."""
        with tempfile.TemporaryDirectory() as tmp:
            base = agent.BaseAgent(
                name="peer_agent",
                allowed_tools=["Read"],
                workspace=Path(tmp),
                mcp_servers={},
                model="fake-model",
                prompt_layout_manifest={
                    "layout_hash": "lh",
                    "frozen_prefix_hash": "fh",
                    "dynamic_payload_hash": "dh",
                },
            )
            with patch.dict(
                "os.environ",
                {"PRAXIST_AGENT_RUNTIME_REF": "agent_runtime:codex_sdk"},
                clear=False,
            ):
                request = base._build_agent_run_request("layout-rendered body", {})
        self.assertEqual(request.prompt_ref["kind"], "prompt_layout_v1")
        self.assertEqual(request.prompt_ref["text"], "layout-rendered body")
        # Hash fields stay attached too — the runtime still wants them for
        # cache-policy provenance.
        self.assertEqual(request.prompt_ref["layout_hash"], "lh")
        self.assertEqual(request.prompt_ref["frozen_prefix_hash"], "fh")
        self.assertEqual(request.prompt_ref["dynamic_payload_hash"], "dh")

    def test_prompt_layout_v1_runtime_overlay_marks_memory_augmented_prompt(self) -> None:
        """Runtime-added memory keeps base layout provenance and records overlay hash."""
        with tempfile.TemporaryDirectory() as tmp:
            base = agent.BaseAgent(
                name="peer_agent",
                allowed_tools=["Read"],
                workspace=Path(tmp),
                mcp_servers={},
                model="fake-model",
                prompt_layout_manifest={
                    "layout_hash": "lh",
                    "frozen_prefix_hash": "fh",
                    "dynamic_payload_hash": "dh",
                    "rendered_prompt_hash": sha256_text("layout-rendered body"),
                },
            )
            with patch.dict(
                "os.environ",
                {"PRAXIST_AGENT_RUNTIME_REF": "agent_runtime:codex_sdk"},
                clear=False,
            ):
                request = base._build_agent_run_request(
                    "layout-rendered body\n\n## Praxist Peer-Local Structured Memory\nstate",
                    {},
                )

        overlay = request.prompt_ref["runtime_overlay"]
        self.assertEqual(
            overlay["overlay_kind"],
            "peer_local_memory_or_bootstrap_runtime_block",
        )
        self.assertEqual(overlay["base_rendered_prompt_hash"], sha256_text("layout-rendered body"))
        self.assertEqual(
            overlay["runtime_composed_prompt_hash"],
            sha256_text("layout-rendered body\n\n## Praxist Peer-Local Structured Memory\nstate"),
        )
        self.assertEqual(
            request.runtime_options["prompt_layout"]["runtime_overlay"],
            overlay,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
