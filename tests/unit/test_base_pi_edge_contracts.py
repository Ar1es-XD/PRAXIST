from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class BasePIEdgeContractsTest(unittest.TestCase):
    def test_multi_pi_forwards_plugin_registry_to_runtime_agent(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"

        registry = object()
        observed: list[object | None] = []

        class CapturingAgent:
            def __init__(self, **kwargs):
                observed.append(kwargs.get("plugin_registry"))

            async def execute(self, task: str):
                return SimpleNamespace(
                    success=True,
                    error=None,
                    output={
                        "text_outputs": [
                            "role: tester\n"
                            "top_claims:\n"
                            "  - id: claim-1\n"
                            "    statement: bounded evidence claim\n"
                        ]
                    },
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = TestPI(
                root / "run",
                root,
                "fake",
                max_runtime_minutes=1,
                mcp_servers={"memory-tools": object()},
                plugin_registry=registry,
            )
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                CapturingAgent,
            ):
                memo = asyncio.run(pi.run({}, [], []))

        self.assertTrue(memo.success)
        self.assertEqual(observed, [registry])

    def test_skill_kb_parse_and_prompt_fallback_edges(self) -> None:
        from praxist.core.role_skills import RoleSkill
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"
            prompt_template_name = "missing-template.jinja2"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb_dir = root / "kb"
            kb_dir.mkdir()
            kb1 = kb_dir / "kb1.md"
            kb1.write_text("alpha beta claim", encoding="utf-8")
            kb2 = kb_dir / "kb2.yaml"
            kb2.write_text("beta gamma", encoding="utf-8")
            (kb_dir / "ignore.txt").write_text("ignore", encoding="utf-8")
            (kb_dir / "subdir").mkdir()

            self.assertEqual(_base_pi._strip_yaml_fence(123), "")
            self.assertEqual(_base_pi._strip_yaml_fence("\ufeff```yaml\nrole: x\n```"), "role: x")
            self.assertEqual(_base_pi._strip_trailing_prose(""), "")
            self.assertEqual(_base_pi._parse_memo_text("[]")["_parse_error"], True)

            pi_without_role = TestPI(root / "run", root, "fake")
            self.assertIsNone(pi_without_role.skill())
            self.assertIsNone(pi_without_role.skill())

            skill = RoleSkill(
                role_ref="task_role:tester",
                role_id="tester",
                display_name="Tester",
                role_kind="pi",
                skill_markdown="skill",
                output_schema_ref="schema",
                default_model_profile_ref=None,
                tool_scope=(),
                fixed_questions=("Q1",),
                private_kb_paths=(kb1, kb_dir / "missing.md"),
                plugin_path=kb_dir,
                content_hash="hash",
            )
            pi_with_skill = TestPI(root / "run", root, "fake", role_ref="task_role:tester")
            with patch.object(_base_pi, "load_role_skill", return_value=skill):
                self.assertIs(pi_with_skill.skill(), skill)
            self.assertEqual(pi_with_skill._private_kb_files(), [kb1])
            self.assertEqual(pi_with_skill.fixed_questions(), ["Q1"])
            self.assertEqual(pi_with_skill._fixed_questions_or(["fallback"]), ["Q1"])

            pi_bad_skill = TestPI(root / "run", root, "fake", role_ref="task_role:bad")
            with patch.object(_base_pi, "load_role_skill", side_effect=RuntimeError("bad")):
                self.assertIsNone(pi_bad_skill.skill())
                self.assertIsNone(pi_bad_skill.skill())

            pi_legacy = TestPI(root / "run", root, "fake")
            pi_legacy._private_kb_path = lambda: kb_dir  # type: ignore[method-assign]
            pi_legacy._MAX_KB_FILES = 1
            ranked = pi_legacy.load_private_kb(top_k=5, query_blob="alpha")
            self.assertEqual(len(ranked), 1)
            self.assertEqual(ranked[0]["id"], "kb1")
            self.assertEqual(pi_legacy.load_private_kb(top_k=1, query_blob="!!!")[0]["id"], "kb1")

            prompt = pi_legacy.render_prompt(
                shared_core={"x": 1},
                private_pack=[],
                private_kb_entries=[],
                target_decisions=[],
            )
            self.assertIn("tester", prompt.lower())

    def test_base_pi_exposes_literature_lookup_prompt_and_tools(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = TestPI(
                root / "run",
                root,
                "fake",
                max_runtime_minutes=1,
                mcp_servers={"literature-lookup": object()},
            )
            prompts: dict[str, str] = {}
            for runtime_ref in ("agent_runtime:claude_sdk", "agent_runtime:codex_sdk"):
                with patch.dict(os.environ, {"PRAXIST_AGENT_RUNTIME_REF": runtime_ref}):
                    prompts[runtime_ref] = pi.render_prompt(
                        shared_core={"x": 1},
                        private_pack=[],
                        private_kb_entries=[],
                        target_decisions=[],
                    )

            self.assertEqual(
                prompts["agent_runtime:claude_sdk"],
                prompts["agent_runtime:codex_sdk"],
            )
            for prompt in prompts.values():
                self.assertIn("External Literature / Database Context", prompt)
                self.assertIn("mcp__literature-lookup__*", prompt)
                self.assertIn("bounded public", prompt)
                self.assertIn("do not recommend acquiring", prompt)
            captured: list[list[str]] = []

            class FakeAgent:
                def __init__(self, **kwargs):
                    captured.append(list(kwargs.get("allowed_tools") or []))

                async def execute(self, task: str):
                    return SimpleNamespace(
                        success=True, error=None, output={"text_outputs": ["[]"]}
                    )

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                FakeAgent,
            ):
                memo = asyncio.run(pi.run({}, [], []))

            self.assertFalse(memo.success)
            self.assertIn("mcp__literature-lookup__literature_search", captured[-1])
            self.assertIn("mcp__literature-lookup__scientific_database_search", captured[-1])

    def test_round1_and_round2_failure_modes_return_structured_memos(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"

        class FakeAgent:
            def __init__(self, **_kwargs):
                pass

            async def execute(self, task: str):
                return SimpleNamespace(success=True, error=None, output={"text_outputs": ["[]"]})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = TestPI(root / "run", root, "fake", max_runtime_minutes=1)
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                FakeAgent,
            ):
                memo = asyncio.run(pi.run({}, [], []))
                round2 = asyncio.run(pi.run_cross_review({}, {}, round2_max_runtime_minutes=1))
            self.assertFalse(memo.success)
            self.assertEqual(memo.error, "parse_error")
            self.assertFalse(round2.success)
            self.assertEqual(round2.error, "parse_error")

            class TimeoutAgent:
                def __init__(self, **_kwargs):
                    pass

                def execute(self, task: str):
                    return object()

            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                    TimeoutAgent,
                ),
                patch.object(_base_pi.asyncio, "wait_for", side_effect=TimeoutError()),
            ):
                self.assertEqual(asyncio.run(pi.run({}, [], [])).error, "timeout")
                self.assertEqual(
                    asyncio.run(pi.run_cross_review({}, {}, round2_max_runtime_minutes=1)).error,
                    "timeout",
                )

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                side_effect=RuntimeError("ctor"),
            ):
                self.assertIn("ctor", asyncio.run(pi.run({}, [], [])).error or "")
                self.assertIn(
                    "ctor",
                    asyncio.run(pi.run_cross_review({}, {}, round2_max_runtime_minutes=1)).error
                    or "",
                )

            with patch("jinja2.Environment.get_template", side_effect=RuntimeError("missing")):
                skipped = asyncio.run(pi.run_cross_review({}, {}, round2_max_runtime_minutes=1))
            self.assertEqual(skipped.error, "round2 template missing")

    def test_base_pi_threads_task_project_path_to_load_role_skill(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"
            role_ref = "task_role:tester"

        captured: dict[str, object] = {}

        def fake_load(role_ref, *, workspace=None, registry=None, task_project_path=None):
            captured["role_ref"] = role_ref
            captured["workspace"] = workspace
            captured["task_project_path"] = task_project_path
            raise RuntimeError("stop here; only the call shape matters")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = TestPI(
                run_dir=root / "run",
                workspace=root / "ws",
                model="dummy",
                task_project_path=root / "task",
            )
            self.assertEqual(pi.task_project_path, root / "task")
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles._base_pi.load_role_skill",
                side_effect=fake_load,
            ):
                self.assertIsNone(pi.skill())
        self.assertEqual(captured["role_ref"], "task_role:tester")
        self.assertEqual(captured["workspace"], root / "ws")
        self.assertEqual(captured["task_project_path"], root / "task")

    def test_base_pi_without_task_project_path_passes_none_to_load_role_skill(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"
            role_ref = "task_role:tester"

        captured: dict[str, object] = {}

        def fake_load(role_ref, *, workspace=None, registry=None, task_project_path=None):
            captured["task_project_path"] = task_project_path
            raise RuntimeError("stop")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = TestPI(run_dir=root, workspace=root, model="dummy")
            self.assertIsNone(pi.task_project_path)
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles._base_pi.load_role_skill",
                side_effect=fake_load,
            ):
                pi.skill()
        self.assertIsNone(captured["task_project_path"])

    def test_round1_and_round2_recover_yaml_memos_from_tool_outputs(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"

        round1_yaml = """
role: tester
private_knowledge_used:
  - kb1
top_claims:
  - id: c1
    statement: recovered from Bash output
proposed_experiments:
  - id: e1
    title: follow up
objections_or_warnings: []
"""
        round2_yaml = """
role: tester
round: 2
strongest_agreement:
  peer_label: "PI #A"
  claim_id: c_peer
  why: keep the promising lane visible
strongest_objection:
  peer_label: "PI #A"
  claim_id: c_peer
  objection: needs complete-protocol evidence
  proposed_kill_test: run the complete protocol
missing_experiment:
  description: cross-condition ablation
  why_critical: isolate mechanism
private_kb_revealed_blind_spot:
  triggered: false
  peer_label: null
  blind_spot: none
claim_that_should_be_downgraded:
  claim_id: c1
  current_language: broad claim
  recommended_language: bounded claim
  reason: needs validation
singleton_high_upside_idea_to_preserve:
  source: self
  peer_label: null
  idea_summary: minority alpha lane
  protected_budget_recommendation: 1 peer
confidence_revisions: []
"""

        class ToolYamlAgent:
            call_count = 0

            def __init__(self, **_kwargs):
                pass

            async def execute(self, task: str):
                ToolYamlAgent.call_count += 1
                yaml_text = round1_yaml if ToolYamlAgent.call_count == 1 else round2_yaml
                return SimpleNamespace(
                    success=True,
                    error=None,
                    output={
                        "text_outputs": ["I will write the memo now."],
                        "tool_uses": [{"tool": "Bash", "output": yaml_text}],
                    },
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = TestPI(root / "run", root, "fake", max_runtime_minutes=1)
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                ToolYamlAgent,
            ):
                memo = asyncio.run(pi.run({}, [], []))
                round2 = asyncio.run(pi.run_cross_review({}, {}, round2_max_runtime_minutes=1))

        self.assertTrue(memo.success)
        self.assertEqual(memo.parsed["role"], "tester")
        self.assertEqual(memo.parsed["top_claims"][0]["id"], "c1")
        self.assertEqual(memo.private_kb_used, ["kb1"])
        self.assertIn("top_claims:", memo.raw_text)
        self.assertTrue(round2.success)
        self.assertEqual(round2.parsed["round"], 2)
        self.assertEqual(
            round2.parsed["strongest_agreement"]["why"],
            "keep the promising lane visible",
        )
        self.assertIn("strongest_objection:", round2.raw_text)

    def test_round2_accepts_negative_private_kb_with_omitted_optional_fields(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"

        class OmittedOptionalFieldsAgent:
            def __init__(self, **_kwargs):
                pass

            async def execute(self, task: str):
                return SimpleNamespace(
                    success=True,
                    error=None,
                    output={
                        "text_outputs": [
                            "role: tester\n"
                            "round: 2\n"
                            "strongest_agreement:\n"
                            '  peer_label: "PI #A"\n'
                            "  claim_id: peer\n"
                            "  why: useful\n"
                            "strongest_objection:\n"
                            '  peer_label: "PI #A"\n'
                            "  claim_id: peer\n"
                            "  objection: weak evidence\n"
                            "  proposed_kill_test: rerun\n"
                            "missing_experiment:\n"
                            "  description: ablation\n"
                            "  why_critical: isolate mechanism\n"
                            "private_kb_revealed_blind_spot:\n"
                            "  triggered: false\n"
                            "claim_that_should_be_downgraded:\n"
                            "  claim_id: c1\n"
                            "  current_language: useful\n"
                            "  recommended_language: bounded useful\n"
                            "  reason: needs validation\n"
                            "singleton_high_upside_idea_to_preserve:\n"
                            "  source: self\n"
                            "  idea_summary: preserve weak signal\n"
                            "  protected_budget_recommendation: 1 peer\n"
                        ]
                    },
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = TestPI(root / "run", root, "fake", max_runtime_minutes=1)
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                OmittedOptionalFieldsAgent,
            ):
                round2 = asyncio.run(
                    pi.run_cross_review(
                        own_memo={"role": "tester", "top_claims": [{"id": "c1"}]},
                        anon_peers={"PI #A": {"top_claims": [{"id": "peer"}]}},
                        round2_max_runtime_minutes=1,
                    )
                )

        self.assertTrue(round2.success)
        self.assertEqual(round2.parsed["role"], "tester")
        self.assertFalse(round2.parsed["private_kb_revealed_blind_spot"]["triggered"])
        self.assertEqual(round2.parsed["singleton_high_upside_idea_to_preserve"]["source"], "self")

    def test_round2_recovers_builder_style_heredoc_with_negative_private_kb(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"

        heredoc_command = """cat <<'YAML'
role: tester
round: 2
strongest_agreement:
  peer_label: "PI #A"
  claim_id: peer
  why: useful
strongest_objection:
  peer_label: "PI #A"
  claim_id: peer
  objection: weak evidence
  proposed_kill_test: rerun
missing_experiment:
  description: ablation
  why_critical: isolate mechanism
private_kb_revealed_blind_spot:
  triggered: false
claim_that_should_be_downgraded:
  claim_id: c1
  current_language: useful
  recommended_language: bounded useful
  reason: needs validation
singleton_high_upside_idea_to_preserve:
  source: peer
  peer_label: "PI #A"
  idea_summary: preserve weak signal
  protected_budget_recommendation: 1 peer
YAML"""

        class HeredocRound2Agent:
            def __init__(self, **_kwargs):
                pass

            async def execute(self, task: str):
                return SimpleNamespace(
                    success=True,
                    error=None,
                    output={
                        "text_outputs": ["The cross-review YAML above is complete."],
                        "tool_uses": [
                            {
                                "name": "Bash",
                                "input": {"command": heredoc_command},
                            }
                        ],
                    },
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = TestPI(root / "run", root, "fake", max_runtime_minutes=1)
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                HeredocRound2Agent,
            ):
                round2 = asyncio.run(
                    pi.run_cross_review(
                        own_memo={"role": "tester", "top_claims": [{"id": "c1"}]},
                        anon_peers={"PI #A": {"top_claims": [{"id": "peer"}]}},
                        round2_max_runtime_minutes=1,
                    )
                )

        self.assertTrue(round2.success)
        self.assertEqual(round2.parsed["round"], 2)
        self.assertIn("strongest_agreement:", round2.raw_text)

    def test_round2_private_kb_triggered_requires_peer_and_blind_spot(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"

        class MissingPrivateKbDetailsAgent:
            def __init__(self, **_kwargs):
                pass

            async def execute(self, task: str):
                return SimpleNamespace(
                    success=True,
                    error=None,
                    output={
                        "text_outputs": [
                            "role: tester\n"
                            "round: 2\n"
                            "strongest_agreement:\n"
                            '  peer_label: "PI #A"\n'
                            "  claim_id: peer\n"
                            "  why: useful\n"
                            "strongest_objection:\n"
                            '  peer_label: "PI #A"\n'
                            "  claim_id: peer\n"
                            "  objection: weak evidence\n"
                            "  proposed_kill_test: rerun\n"
                            "missing_experiment:\n"
                            "  description: ablation\n"
                            "  why_critical: isolate mechanism\n"
                            "private_kb_revealed_blind_spot:\n"
                            "  triggered: true\n"
                            "  peer_label: null\n"
                            "claim_that_should_be_downgraded:\n"
                            "  claim_id: c1\n"
                            "  current_language: useful\n"
                            "  recommended_language: bounded useful\n"
                            "  reason: needs validation\n"
                            "singleton_high_upside_idea_to_preserve:\n"
                            "  source: self\n"
                            "  idea_summary: preserve weak signal\n"
                            "  protected_budget_recommendation: 1 peer\n"
                        ]
                    },
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = TestPI(root / "run", root, "fake", max_runtime_minutes=1)
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                MissingPrivateKbDetailsAgent,
            ):
                round2 = asyncio.run(
                    pi.run_cross_review(
                        own_memo={"role": "tester", "top_claims": [{"id": "c1"}]},
                        anon_peers={"PI #A": {"top_claims": [{"id": "peer"}]}},
                        round2_max_runtime_minutes=1,
                    )
                )

        self.assertFalse(round2.success)
        self.assertTrue(round2.parsed["_schema_error"])

    def test_round1_recovers_yaml_from_bash_heredoc_input_command(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"

        long_padding = "\n".join(f"# padding {i}" for i in range(125))
        heredoc_command = f"""cat <<'YAML' > memo_builder.yaml
role: tester
private_knowledge_used: []
top_claims:
  - id: c_heredoc
    statement: recovered from heredoc input command
proposed_experiments:
  - id: e_heredoc
    title: follow up
objections_or_warnings: []
{long_padding}
YAML"""

        class HeredocInputAgent:
            def __init__(self, **_kwargs):
                pass

            async def execute(self, task: str):
                return SimpleNamespace(
                    success=True,
                    error=None,
                    output={
                        "text_outputs": [
                            "Let me analyze the evidence systematically to produce my memo. I need to:"
                        ],
                        "tool_uses": [{"tool": "Bash", "input": {"command": heredoc_command}}],
                    },
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = TestPI(root / "run", root, "fake", max_runtime_minutes=1)
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                HeredocInputAgent,
            ):
                memo = asyncio.run(pi.run({}, [], []))

        self.assertTrue(memo.success)
        self.assertEqual(memo.parsed["top_claims"][0]["id"], "c_heredoc")
        self.assertIn("proposed_experiments:", memo.raw_text)

    def test_round2_fragment_is_not_successful_cross_review(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"

        class FragmentAgent:
            def __init__(self, **_kwargs):
                pass

            async def execute(self, task: str):
                return SimpleNamespace(
                    success=True,
                    error=None,
                    output={
                        "text_outputs": [
                            "```yaml\nown_revisions:\n  - claim_id: c1\n    revision: keep\n```"
                        ]
                    },
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = TestPI(root / "run", root, "fake", max_runtime_minutes=1)
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                FragmentAgent,
            ):
                round2 = asyncio.run(
                    pi.run_cross_review(
                        own_memo={"role": "tester", "top_claims": [{"id": "c1"}]},
                        anon_peers={"PI #A": {"top_claims": [{"id": "peer"}]}},
                        round2_max_runtime_minutes=1,
                    )
                )

        self.assertFalse(round2.success)
        self.assertEqual(round2.error, "parse_error")
        self.assertTrue(round2.parsed["_schema_error"])

    def test_round2_scalar_fields_are_not_successful_cross_review(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"

        class ScalarAgent:
            def __init__(self, **_kwargs):
                pass

            async def execute(self, task: str):
                return SimpleNamespace(
                    success=True,
                    error=None,
                    output={
                        "text_outputs": [
                            "role: tester\n"
                            "round: 2\n"
                            "strongest_agreement: keep\n"
                            "strongest_objection: no\n"
                            "missing_experiment: ablation\n"
                            "private_kb_revealed_blind_spot: none\n"
                            "claim_that_should_be_downgraded: none\n"
                            "singleton_high_upside_idea_to_preserve: idea\n"
                        ]
                    },
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = TestPI(root / "run", root, "fake", max_runtime_minutes=1)
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                ScalarAgent,
            ):
                round2 = asyncio.run(
                    pi.run_cross_review(
                        own_memo={"role": "tester", "top_claims": [{"id": "c1"}]},
                        anon_peers={"PI #A": {"top_claims": [{"id": "peer"}]}},
                        round2_max_runtime_minutes=1,
                    )
                )

        self.assertFalse(round2.success)
        self.assertTrue(round2.parsed["_schema_error"])

    def test_round2_peer_label_placeholder_is_not_successful(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"

        class PeerPlaceholderAgent:
            def __init__(self, **_kwargs):
                pass

            async def execute(self, task: str):
                return SimpleNamespace(
                    success=True,
                    error=None,
                    output={
                        "text_outputs": [
                            "role: tester\n"
                            "round: 2\n"
                            "strongest_agreement:\n"
                            "  peer_label: PI #?\n"
                            "  claim_id: peer\n"
                            "  why: useful\n"
                            "strongest_objection:\n"
                            "  peer_label: PI #?\n"
                            "  claim_id: peer\n"
                            "  objection: weak evidence\n"
                            "  proposed_kill_test: rerun\n"
                            "missing_experiment:\n"
                            "  description: ablation\n"
                            "  why_critical: isolate mechanism\n"
                            "private_kb_revealed_blind_spot:\n"
                            "  triggered: false\n"
                            "  peer_label: null\n"
                            "  blind_spot: none\n"
                            "claim_that_should_be_downgraded:\n"
                            "  claim_id: c1\n"
                            "  current_language: useful\n"
                            "  recommended_language: bounded useful\n"
                            "  reason: needs validation\n"
                            "singleton_high_upside_idea_to_preserve:\n"
                            "  source: self\n"
                            "  peer_label: null\n"
                            "  idea_summary: preserve weak signal\n"
                            "  protected_budget_recommendation: 1 peer\n"
                        ]
                    },
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = TestPI(root / "run", root, "fake", max_runtime_minutes=1)
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                PeerPlaceholderAgent,
            ):
                round2 = asyncio.run(
                    pi.run_cross_review(
                        own_memo={"role": "tester", "top_claims": [{"id": "c1"}]},
                        anon_peers={"PI #A": {"top_claims": [{"id": "peer"}]}},
                        round2_max_runtime_minutes=1,
                    )
                )

        self.assertFalse(round2.success)
        self.assertTrue(round2.parsed["_schema_error"])

    def test_round2_missing_required_peer_label_is_not_successful(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"

        class MissingPeerAgent:
            def __init__(self, **_kwargs):
                pass

            async def execute(self, task: str):
                return SimpleNamespace(
                    success=True,
                    error=None,
                    output={
                        "text_outputs": [
                            "role: tester\n"
                            "round: 2\n"
                            "strongest_agreement:\n"
                            "  peer_label: null\n"
                            "  claim_id: peer\n"
                            "  why: useful\n"
                            "strongest_objection:\n"
                            '  peer_label: "PI #A"\n'
                            "  claim_id: peer\n"
                            "  objection: weak evidence\n"
                            "  proposed_kill_test: rerun\n"
                            "missing_experiment:\n"
                            "  description: ablation\n"
                            "  why_critical: isolate mechanism\n"
                            "private_kb_revealed_blind_spot:\n"
                            "  triggered: false\n"
                            "  peer_label: null\n"
                            "  blind_spot: none\n"
                            "claim_that_should_be_downgraded:\n"
                            "  claim_id: c1\n"
                            "  current_language: useful\n"
                            "  recommended_language: bounded useful\n"
                            "  reason: needs validation\n"
                            "singleton_high_upside_idea_to_preserve:\n"
                            "  source: self\n"
                            "  peer_label: null\n"
                            "  idea_summary: preserve weak signal\n"
                            "  protected_budget_recommendation: 1 peer\n"
                        ]
                    },
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = TestPI(root / "run", root, "fake", max_runtime_minutes=1)
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                MissingPeerAgent,
            ):
                round2 = asyncio.run(
                    pi.run_cross_review(
                        own_memo={"role": "tester", "top_claims": [{"id": "c1"}]},
                        anon_peers={"PI #A": {"top_claims": [{"id": "peer"}]}},
                        round2_max_runtime_minutes=1,
                    )
                )

        self.assertFalse(round2.success)
        self.assertTrue(round2.parsed["_schema_error"])

    def test_round1_empty_top_claims_is_not_successful(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"

        class EmptyClaimsAgent:
            def __init__(self, **_kwargs):
                pass

            async def execute(self, task: str):
                return SimpleNamespace(
                    success=True,
                    error=None,
                    output={"text_outputs": ["role: tester\ntop_claims: []\n"]},
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = TestPI(root / "run", root, "fake", max_runtime_minutes=1)
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                EmptyClaimsAgent,
            ):
                memo = asyncio.run(pi.run({}, [], []))

        self.assertFalse(memo.success)
        self.assertTrue(memo.parsed["_schema_error"])

    def test_round1_claim_id_only_shell_is_not_successful(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"

        class IdOnlyAgent:
            def __init__(self, **_kwargs):
                pass

            async def execute(self, task: str):
                return SimpleNamespace(
                    success=True,
                    error=None,
                    output={"text_outputs": ["role: tester\ntop_claims:\n  - id: c1\n"]},
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = TestPI(root / "run", root, "fake", max_runtime_minutes=1)
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                IdOnlyAgent,
            ):
                memo = asyncio.run(pi.run({}, [], []))

        self.assertFalse(memo.success)
        self.assertTrue(memo.parsed["_schema_error"])

    def test_memo_placeholder_schema_is_not_successful(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"

        class PlaceholderAgent:
            def __init__(self, **_kwargs):
                pass

            async def execute(self, task: str):
                return SimpleNamespace(
                    success=True,
                    error=None,
                    output={
                        "text_outputs": [
                            "role: tester\ntop_claims:\n  - id: c1\n    statement: <one sentence>\n"
                        ]
                    },
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = TestPI(root / "run", root, "fake", max_runtime_minutes=1)
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                PlaceholderAgent,
            ):
                memo = asyncio.run(pi.run({}, [], []))

        self.assertFalse(memo.success)
        self.assertTrue(memo.parsed["_schema_error"])

    def test_prompt_template_placeholder_variants_are_not_successful(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"

        class PlaceholderAgent:
            def __init__(self, **_kwargs):
                pass

            async def execute(self, task: str):
                return SimpleNamespace(
                    success=True,
                    error=None,
                    output={
                        "text_outputs": [
                            "role: tester\n"
                            "top_claims:\n"
                            "  - id: C_tester_g<gen>_<NN>\n"
                            "    statement: real statement\n"
                            "    confidence: <0..1>\n"
                            '    boundary: "<scope: current protocol only>"\n'
                            "    supports: [<evidence_id>]\n"
                        ]
                    },
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = TestPI(root / "run", root, "fake", max_runtime_minutes=1)
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                PlaceholderAgent,
            ):
                memo = asyncio.run(pi.run({}, [], []))

        self.assertFalse(memo.success)
        self.assertTrue(memo.parsed["_schema_error"])

    def test_legitimate_angle_bracket_notation_is_allowed(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"

        class AngleNotationAgent:
            def __init__(self, **_kwargs):
                pass

            async def execute(self, task: str):
                return SimpleNamespace(
                    success=True,
                    error=None,
                    output={
                        "text_outputs": [
                            "role: tester\n"
                            "top_claims:\n"
                            "  - id: c1\n"
                            '    statement: "Use <X, Y> pair features, <lookahead-k>, and <identity map>."\n'
                        ]
                    },
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = TestPI(root / "run", root, "fake", max_runtime_minutes=1)
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                AngleNotationAgent,
            ):
                memo = asyncio.run(pi.run({}, [], []))

        self.assertTrue(memo.success)

    def test_round2_no_peers_recovers_from_tool_use(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"

        heredoc_command = """cat <<'YAML' > round2.yaml
role: tester
round: 2
_no_peers: true
note: "Only PI with successful Round 1 memo; cross-review skipped."
YAML"""

        class NoPeersToolAgent:
            def __init__(self, **_kwargs):
                pass

            async def execute(self, task: str):
                return SimpleNamespace(
                    success=True,
                    error=None,
                    output={
                        "text_outputs": ["I will write the no-peers response."],
                        "tool_uses": [{"tool": "Bash", "input": {"command": heredoc_command}}],
                    },
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = TestPI(root / "run", root, "fake", max_runtime_minutes=1)
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                NoPeersToolAgent,
            ):
                round2 = asyncio.run(
                    pi.run_cross_review(
                        own_memo={"role": "tester", "top_claims": [{"id": "c1"}]},
                        anon_peers={},
                        round2_max_runtime_minutes=1,
                    )
                )

        self.assertTrue(round2.success)
        self.assertTrue(round2.parsed["_no_peers"])

    def test_tool_recovery_requires_matching_pi_role(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"

        wrong_role_yaml = """
role: other
private_knowledge_used: []
top_claims:
  - id: c1
    statement: wrong role memo
"""

        class WrongRoleToolAgent:
            def __init__(self, **_kwargs):
                pass

            async def execute(self, task: str):
                return SimpleNamespace(
                    success=True,
                    error=None,
                    output={
                        "text_outputs": ["I need to:"],
                        "tool_uses": [{"tool": "Bash", "output": wrong_role_yaml}],
                    },
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = TestPI(root / "run", root, "fake", max_runtime_minutes=1)
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                WrongRoleToolAgent,
            ):
                memo = asyncio.run(pi.run({}, [], []))

        self.assertFalse(memo.success)
        self.assertTrue(memo.parsed["_schema_error"])

    def test_schema_incomplete_yaml_is_unavailable_not_successful(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"

        class IncompleteAgent:
            def __init__(self, **_kwargs):
                pass

            async def execute(self, task: str):
                return SimpleNamespace(
                    success=True,
                    error=None,
                    output={"text_outputs": ["Let me analyze the evidence. I need to:"]},
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = TestPI(root / "run", root, "fake", max_runtime_minutes=1)
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                IncompleteAgent,
            ):
                memo = asyncio.run(pi.run({}, [], []))

        self.assertFalse(memo.success)
        self.assertTrue(memo.parsed["_schema_error"])
        self.assertIn("top_claims", memo.parsed["missing_required_keys"])


if __name__ == "__main__":
    unittest.main()
