from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
    agenda_validator_v2,
)
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.chair_arbiter import (
    ChairArbiter,
    _parse_chair_agenda_text,
    _strip_trailing_prose,
)
from praxist.plugins.workflow_stages.research_loop.backend.tools.findings_ingest import (
    parse_finding_file,
)


class ResearchStateAdversarialContracts(unittest.TestCase):
    def test_filesystem_finding_ingest_preserves_gating_and_graph_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            finding_path = Path(tmp) / "gen0_peer0_result.json"
            finding_path.write_text(
                json.dumps(
                    {
                        "title": "candidate",
                        "summary": "candidate result",
                        "finding_type": "result",
                        "variant_name": "variant_a",
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "metrics": {"score": 0.7},
                        "tier": "T1",
                        "promotion_eligible": False,
                        "extra": {"peer_role": "falsifier"},
                        "links": [{"target_id": "finding_b", "relation": "challenges"}],
                        "design_dimensions": {"family": "probe"},
                    }
                ),
                encoding="utf-8",
            )

            row = parse_finding_file(finding_path)
            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row.get("tier"), "T1")
            self.assertIs(row.get("promotion_eligible"), False)
            self.assertEqual(row.get("extra", {}).get("peer_role"), "falsifier")
            self.assertEqual(
                row.get("links"), [{"target_id": "finding_b", "relation": "challenges"}]
            )
            self.assertEqual(row.get("design_dimensions"), {"family": "probe"})

    def test_chair_arbiter_never_receives_mcp_servers(self) -> None:
        captured: dict[str, object] = {}

        class FakeBaseAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            async def execute(self, task: str):
                return type(
                    "Result",
                    (),
                    {
                        "success": True,
                        "output": {"text_outputs": ["next_generation_id: 1\npeer_contracts: []\n"]},
                        "error": None,
                    },
                )()

        async def _run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                arbiter = ChairArbiter(
                    run_dir=Path(tmp) / "run",
                    workspace=Path(tmp),
                    model="fake-model",
                    mcp_servers={"memory-tools": object()},
                )
                with patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                    FakeBaseAgent,
                ):
                    await arbiter.run(
                        shared_core_digest={},
                        pi_memos={},
                        cross_reviews={},
                        confidence_revisions={},
                        next_gen_id=1,
                        completed_gen_id=0,
                        panel_mode="single_pi",
                        shared_core_id="shared_core",
                    )

        asyncio.run(_run())
        self.assertEqual(captured.get("mcp_servers"), {})
        self.assertEqual(captured.get("allowed_tools"), [])

    def test_chair_agenda_extractor_ignores_yaml_parseable_preamble(self) -> None:
        raw = """I am reconciling PI memos and protected candidates
: anti-EDC, subspace revival, and tuned baseline.

agenda_version: "2.0"
generation: 1
mainline_observation:
  main_risk: "lateboost may be an artifact"
cross_peer_hypotheses:
  - id: H_g1_01
    claim: "test claim"
    minimal_test: "run the gate"
    kill_condition: "fails gate"
    promote_condition: "passes gate"
bridge_hypothesis:
  id: B_g1_01
anti_mainline_contract:
  target_axes: ["baseline"]
falsification_contract:
  target_hypothesis: H_g1_01
peer_contracts:
  gen1_peer0:
    role: exploit
    target_hypothesis: H_g1_01
    success_signal: "passes"
"""
        cleaned = _strip_trailing_prose(raw)
        self.assertTrue(cleaned.startswith('agenda_version: "2.0"'))
        self.assertIn("peer_contracts:", cleaned)
        self.assertNotIn("I am reconciling", cleaned)

    def test_chair_agenda_parser_finds_agenda_after_long_prose(self) -> None:
        raw = "\n".join(
            [
                *(f"reasoning line {i}: not agenda yet" for i in range(140)),
                'agenda_version: "2.0"',
                "generation: 4",
                "mainline_observation:",
                '  main_risk: "control missing"',
                "cross_peer_hypotheses:",
                "  - id: H_g4_01",
                '    claim: "test claim"',
                '    minimal_test: "run control"',
                '    kill_condition: "control wins"',
                '    promote_condition: "claim survives"',
                "peer_contracts:",
                "  gen4_peer0:",
                "    role: exploit",
                "    target_hypothesis: H_g4_01",
                '    success_signal: "result lands"',
                "closing prose that must be ignored",
            ]
        )

        parsed = _parse_chair_agenda_text(raw)

        self.assertIsNotNone(parsed.agenda)
        assert parsed.agenda is not None
        self.assertEqual(parsed.agenda["generation"], 4)
        self.assertTrue(parsed.cleaned_text.startswith('agenda_version: "2.0"'))
        self.assertNotIn("reasoning line 139", parsed.cleaned_text)

    def test_chair_falls_back_to_valid_agenda_on_unparseable_output(self) -> None:
        class FakeBaseAgent:
            def __init__(self, **kwargs):
                pass

            async def execute(self, task: str):
                return type(
                    "Result",
                    (),
                    {
                        "success": True,
                        "output": {
                            "text_outputs": [
                                "I am thinking around a few critical points: all PIs agree "
                                "that the control is unresolved, but I will not emit YAML."
                            ]
                        },
                        "error": None,
                    },
                )()

        pi_memos = {
            "builder": {
                "role": "builder",
                "top_claims": [
                    {
                        "id": "C_builder_01",
                        "statement": "Candidate mechanism may improve the primary metric.",
                        "supports": ["E_builder_01"],
                    }
                ],
                "proposed_peer_contracts": [
                    {
                        "role": "bridge",
                        "target_hypothesis": "C_builder_01",
                        "rationale": "Combine the two panel-supported axes.",
                    }
                ],
                "private_knowledge_used": [],
            },
            "skeptic": {
                "role": "skeptic",
                "top_claims": [
                    {
                        "id": "C_skeptic_01",
                        "statement": "The leading claim may be a matched-control artifact.",
                        "supports": ["NEG_skeptic_01"],
                    }
                ],
                "objections_or_warnings": [
                    {
                        "target_claim": "C_builder_01",
                        "objection": "No matched control has been run.",
                        "resolving_experiment": (
                            "Run a matched-control falsifier with the same resource envelope."
                        ),
                    }
                ],
                "private_knowledge_used": [],
            },
            "portfolio": {
                "role": "portfolio",
                "top_claims": [
                    {
                        "id": "C_portfolio_01",
                        "statement": "The search needs one non-mainline corridor.",
                        "supports": ["E_portfolio_01"],
                    }
                ],
                "private_knowledge_used": [],
            },
        }

        async def _run():
            with tempfile.TemporaryDirectory() as tmp:
                arbiter = ChairArbiter(
                    run_dir=Path(tmp) / "run",
                    workspace=Path(tmp),
                    model="fake-model",
                    peer_budget=5,
                )
                with patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                    FakeBaseAgent,
                ):
                    return await arbiter.run(
                        shared_core_digest={},
                        pi_memos=pi_memos,
                        cross_reviews={},
                        confidence_revisions={},
                        next_gen_id=4,
                        completed_gen_id=3,
                        panel_mode="full",
                        shared_core_id="shared_core",
                    )

        result = asyncio.run(_run())

        self.assertTrue(result.success)
        self.assertIn("fallback_agenda", result.error or "")
        self.assertTrue(result.agenda.get("fallback_metadata", {}).get("reason"))
        validation = agenda_validator_v2.validate_agenda_v2(
            result.agenda,
            next_gen_id=4,
            cohort_size=5,
            pi_memos=pi_memos,
        )
        self.assertTrue(validation.valid, validation.blocking_issues)
        self.assertEqual(
            sorted(result.agenda["peer_contracts"].keys()),
            [f"gen4_peer{i}" for i in range(5)],
        )
