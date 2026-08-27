from __future__ import annotations

import asyncio
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml


def _valid_agenda(next_gen_id: int, cohort_size: int = 5) -> dict:
    roles = ["exploit", "falsifier", "bridge", "anti-mainline", "theorist"]
    metadata = {
        "bottleneck_target": "process_or_evidence_gap",
        "evidence_stage": "full_T1",
        "tradeoff_class": "incomplete_evidence",
        "primary_tradeoff": "custom",
        "next_step_intent": "preserve_and_validate",
        "parent_candidate": "",
        "parent_usage": "none",
    }
    return {
        "generation": f"gen{next_gen_id}",
        "mainline_observation": {
            "current_dominant_mechanisms": ["A"],
            "main_risk": "risk",
            "key_tradeoff": "tradeoff",
        },
        "cross_peer_hypotheses": [
            {
                "id": "H1",
                "claim": "claim",
                **metadata,
                "minimal_test": "test",
                "kill_condition": "kill",
                "promote_condition": "promote",
            },
            None,
        ],
        "peer_contracts": {
            f"gen{next_gen_id}_peer{i}": {
                "role": roles[i % len(roles)],
                "target_hypothesis": "H1",
                **metadata,
                "success_signal": "publish a result",
            }
            for i in range(cohort_size)
        },
        "bridge_hypothesis": {},
        "anti_mainline_contract": {},
        "falsification_contract": {},
        "success_metrics": {},
    }


class PIAgentContractsTest(unittest.TestCase):
    def test_multi_pi_qd_policy_is_prompt_only_and_does_not_mutate_evidence(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.legacy_two_round_executor import (
            _shared_core_for_panel,
        )

        shared_core = {"shared_core_id": "evidence-id", "cards": [{"id": "F1"}]}
        policy = {"enabled": True, "candidate_source": "existing_pi_synthesis"}

        prompt_core = _shared_core_for_panel(shared_core, policy)

        self.assertIsNot(prompt_core, shared_core)
        self.assertNotIn("quality_diversity_policy", shared_core)
        self.assertEqual(prompt_core["quality_diversity_policy"], policy)
        self.assertEqual(prompt_core["shared_core_id"], "evidence-id")

    def test_single_pi_prompt_includes_later_generation_qd_policy_only_when_enabled(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import pi_agent
        from praxist.plugins.workflow_stages.research_loop.backend.dig.config import (
            DIGLiteConfig,
            QualityDiversityConfig,
        )

        dig = DIGLiteConfig.from_raw({"enabled": True})
        qd = QualityDiversityConfig.from_task_spec(
            {
                "quality_diversity": {
                    "enabled": True,
                    "initial_generation_enabled": True,
                    "later_generations_enabled": True,
                }
            },
            dig_config=dig,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent = pi_agent.PIAgent(
                run_dir=root,
                workspace=root,
                cohort_size=2,
                model="noop",
                quality_diversity_config=qd,
            )
            policy = agent._quality_diversity_policy(1)
            prompt = agent._build_synthesis_prompt(
                completed_gen_id=0,
                findings=[],
                edges=[],
                frontier=[],
                prior_agenda=None,
                prior_agendas_summary=[],
                prior_findings_summary=[],
                agenda_output_path=root / "agenda.yaml",
                quality_diversity_policy=policy,
            )
            qd.later_generations_enabled = False
            disabled_policy = agent._quality_diversity_policy(1)
            disabled_prompt = agent._build_synthesis_prompt(
                completed_gen_id=0,
                findings=[],
                edges=[],
                frontier=[],
                prior_agenda=None,
                prior_agendas_summary=[],
                prior_findings_summary=[],
                agenda_output_path=root / "agenda-disabled.yaml",
                quality_diversity_policy=disabled_policy,
            )

        self.assertEqual(policy["candidate_source"], "existing_pi_synthesis")
        self.assertIn("risk_penalty_weight", policy["scoring_guidance"])
        self.assertIn("target_keyword_bonus", policy["scoring_guidance"])
        self.assertIn("Quality-Diversity allocation", prompt)
        self.assertIn("existing_pi_synthesis", prompt)
        self.assertEqual(disabled_policy, {})
        self.assertNotIn("Quality-Diversity allocation", disabled_prompt)

    def test_chair_prompt_receives_later_qd_policy_as_soft_allocation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            chair_arbiter,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arbiter = chair_arbiter.ChairArbiter(root, root, "fake")
            prompt = arbiter.render_prompt(
                shared_core_digest={
                    "quality_diversity_policy": {
                        "enabled": True,
                        "candidate_source": "existing_pi_synthesis",
                        "selection_mode": "prompt_guided_quality_diversity",
                    }
                },
                pi_memos={},
                cross_reviews={},
                confidence_revisions={},
                next_gen_id=1,
                completed_gen_id=0,
                panel_mode="full",
                shared_core_id="core",
            )

        self.assertIn("Quality-Diversity allocation", prompt)
        self.assertIn("soft allocation controls", prompt)
        self.assertIn("existing_pi_synthesis", prompt)

    def test_single_pi_literature_lookup_prompt_is_runtime_neutral(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import pi_agent

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent = pi_agent.PIAgent(
                run_dir=root,
                workspace=root,
                cohort_size=1,
                model="noop",
                mcp_servers={"literature-lookup": object()},
                prompt_template_path=Path(pi_agent.__file__).with_name("synthesis_prompt.jinja2"),
            )

            prompts: dict[str, str] = {}
            for runtime_ref in ("agent_runtime:claude_sdk", "agent_runtime:codex_sdk"):
                with patch.dict(os.environ, {"PRAXIST_AGENT_RUNTIME_REF": runtime_ref}):
                    prompts[runtime_ref] = agent._build_synthesis_prompt(
                        completed_gen_id=0,
                        findings=[],
                        edges=[],
                        frontier=[],
                        prior_agenda=None,
                        prior_agendas_summary=[],
                        prior_findings_summary=[],
                        agenda_output_path=root / "agenda.yaml",
                    )

        self.assertEqual(
            prompts["agent_runtime:claude_sdk"],
            prompts["agent_runtime:codex_sdk"],
        )
        for prompt in prompts.values():
            self.assertIn("mcp__literature-lookup__*", prompt)
            self.assertIn("bounded public", prompt)
            self.assertIn("unavailable data", prompt)
            self.assertIn("acquisition or installation work", prompt)

    def test_single_pi_frontier_summary_filters_future_generations(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import PIAgent

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontier = root / "frontier"
            frontier.mkdir()
            (frontier / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "cumulative_top": [
                            {
                                "variant_name": "current_candidate",
                                "generation_id": 0,
                                "metrics": {"score": 1.0, "scored_complete": True},
                            },
                            {
                                "variant_name": "legacy_preliminary",
                                "generation_id": 0,
                                "metric_value": 99.0,
                                "evidence_stage": "preliminary",
                                "metrics": {"score": 99.0},
                            },
                            {
                                "variant_name": "unknown_generation_candidate",
                                "metrics": {"score": 50.0},
                            },
                            {
                                "variant_name": "future_candidate",
                                "generation_id": 2,
                                "metrics": {"score": 99.0, "scored_complete": True},
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            pi = PIAgent(
                run_dir=root,
                workspace=root,
                cohort_size=1,
                model="noop",
            )

            rows = pi._load_frontier_summary(completed_gen_id=0)

        names = [row["variant_name"] for row in rows]
        self.assertEqual(names, ["current_candidate"])

    def test_full_panel_missing_theorist_is_repaired_before_validation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            legacy_two_round_executor,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.agenda_validator_v2 import (
            validate_agenda_v2,
        )

        agenda = _valid_agenda(2, cohort_size=8)
        agenda["agenda_version"] = "2.0"
        agenda["panel_mode"] = "full"
        agenda["cross_peer_hypotheses"] = [
            {
                "id": f"H{i}",
                "claim": "claim",
                "minimal_test": "test",
                "kill_condition": "kill",
                "promote_condition": "promote",
            }
            for i in range(1, 4)
        ]
        for i, role in enumerate(
            [
                "exploit",
                "falsifier",
                "bridge",
                "anti_mainline",
                "exploit",
                "falsifier",
                "exploit",
                "falsifier",
            ]
        ):
            agenda["peer_contracts"][f"gen2_peer{i}"]["role"] = role

        before = validate_agenda_v2(agenda, next_gen_id=2, cohort_size=8)
        self.assertFalse(before.valid)
        self.assertIn("theorist", "\n".join(before.blocking_issues))

        repairs = legacy_two_round_executor._ensure_full_panel_required_roles(
            agenda,
            panel_mode="full",
            cohort_size=8,
        )

        self.assertTrue(any(r.get("new_role") == "theorist" for r in repairs))
        roles = {c["role"] for c in agenda["peer_contracts"].values()}
        self.assertIn("theorist", roles)
        self.assertIn("auto_repairs", agenda)
        self.assertEqual(agenda["_runtime_panel_mode"], "full")
        repaired_contracts = [
            c
            for c in agenda["peer_contracts"].values()
            if isinstance(c, dict) and c.get("role") == "theorist"
        ]
        self.assertTrue(repaired_contracts)
        self.assertIn("Theorist deliverable", repaired_contracts[0]["success_signal"])
        for key in (
            "bottleneck_target",
            "evidence_stage",
            "tradeoff_class",
            "primary_tradeoff",
            "next_step_intent",
            "parent_candidate",
            "parent_usage",
        ):
            self.assertIn(key, repaired_contracts[0])
        after = validate_agenda_v2(agenda, next_gen_id=2, cohort_size=8)
        self.assertTrue(after.valid, after.blocking_issues)

    def test_runtime_full_panel_missing_role_blocks_even_if_agenda_claims_mini(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.agenda_validator_v2 import (
            validate_agenda_v2,
        )

        agenda = _valid_agenda(2, cohort_size=8)
        agenda["agenda_version"] = "2.0"
        agenda["panel_mode"] = "mini"
        agenda["_runtime_panel_mode"] = "full"
        for i, role in enumerate(
            [
                "exploit",
                "falsifier",
                "bridge",
                "anti_mainline",
                "exploit",
                "falsifier",
                "exploit",
                "falsifier",
            ]
        ):
            agenda["peer_contracts"][f"gen2_peer{i}"]["role"] = role

        result = validate_agenda_v2(agenda, next_gen_id=2, cohort_size=8)

        self.assertFalse(result.valid)
        self.assertIn("theorist", "\n".join(result.blocking_issues))

    def test_custom_required_roles_replace_bundled_full_panel_roles(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.agenda_validator_v2 import (
            validate_agenda_v2,
        )

        agenda = _valid_agenda(2, cohort_size=4)
        agenda["agenda_version"] = "2.0"
        agenda["_runtime_panel_mode"] = "full"
        agenda["cross_peer_hypotheses"] = [
            {**agenda["cross_peer_hypotheses"][0], "id": f"H{i}"} for i in range(1, 4)
        ]
        for i, role in enumerate(["specialist_a", "specialist_b", "specialist_a", "specialist_b"]):
            agenda["peer_contracts"][f"gen2_peer{i}"]["role"] = role

        result = validate_agenda_v2(
            agenda,
            next_gen_id=2,
            cohort_size=4,
            required_peer_roles=("specialist_a", "specialist_b"),
        )

        self.assertTrue(result.valid, result.blocking_issues)
        self.assertNotIn("theorist", "\n".join(result.blocking_issues + result.warnings))

    def test_custom_required_role_rotation_preserves_duplicate_role_quota(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.agenda_validator_v2 import (
            validate_agenda_v2,
        )

        agenda = _valid_agenda(2, cohort_size=3)
        agenda["agenda_version"] = "2.0"
        agenda["_runtime_panel_mode"] = "full"
        agenda["cross_peer_hypotheses"] = [
            {**agenda["cross_peer_hypotheses"][0], "id": f"H{i}"} for i in range(1, 4)
        ]
        for i, role in enumerate(["builder", "skeptic", "skeptic"]):
            agenda["peer_contracts"][f"gen2_peer{i}"]["role"] = role

        blocked = validate_agenda_v2(
            agenda,
            next_gen_id=2,
            cohort_size=3,
            required_peer_roles=("builder", "builder", "skeptic"),
        )

        self.assertFalse(blocked.valid)
        self.assertIn("builder", "\n".join(blocked.blocking_issues))

        agenda["peer_contracts"]["gen2_peer1"]["role"] = "builder"
        allowed = validate_agenda_v2(
            agenda,
            next_gen_id=2,
            cohort_size=3,
            required_peer_roles=("builder", "builder", "skeptic"),
        )
        self.assertTrue(allowed.valid, allowed.blocking_issues)

    def test_single_pi_custom_role_rotation_normalizes_role_names(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import PIAgent

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agenda = _valid_agenda(2, cohort_size=2)
            agenda["peer_contracts"]["gen2_peer0"]["role"] = "specialist_a"
            agenda["peer_contracts"]["gen2_peer1"]["role"] = "specialist_b"
            pi = PIAgent(
                run_dir=root,
                workspace=root,
                cohort_size=2,
                model="noop",
                peer_role_rotation=("specialist-a", "specialist-b"),
            )

            error = pi.validate_agenda(agenda, 2)

        self.assertIsNone(error)

    def test_full_panel_role_repair_uses_validator_role_normalization(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            legacy_two_round_executor,
        )

        agenda = _valid_agenda(2, cohort_size=5)
        for role, contract in zip(
            ["exploit", "falsifier", "bridge", "anti/mainline", "theorist"],
            agenda["peer_contracts"].values(),
            strict=True,
        ):
            contract["role"] = role

        repairs = legacy_two_round_executor._ensure_full_panel_required_roles(
            agenda,
            panel_mode="full",
            cohort_size=5,
        )

        self.assertEqual(repairs, [])
        self.assertEqual(agenda["peer_contracts"]["gen2_peer3"]["role"], "anti/mainline")

    def test_full_panel_role_repair_does_not_mutate_custom_topology_roles(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            legacy_two_round_executor,
        )

        agenda = _valid_agenda(2, cohort_size=4)
        for i, role in enumerate(["specialist_a", "specialist_a", "specialist_a", "specialist_a"]):
            agenda["peer_contracts"][f"gen2_peer{i}"]["role"] = role

        repairs = legacy_two_round_executor._ensure_full_panel_required_roles(
            agenda,
            panel_mode="full",
            cohort_size=4,
            required_peer_roles=("specialist_a", "specialist_b"),
        )

        self.assertEqual(repairs, [])
        self.assertEqual(
            {contract["role"] for contract in agenda["peer_contracts"].values()},
            {"specialist_a"},
        )

    def test_claim_boundary_update_cannot_self_authorize_unknown_claim(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.agenda_validator_v2 import (
            validate_agenda_v2,
        )

        agenda = _valid_agenda(2, cohort_size=5)
        agenda["cross_peer_hypotheses"] = [
            {**agenda["cross_peer_hypotheses"][0], "id": f"H{i}"} for i in range(1, 4)
        ]
        agenda["claim_boundary_updates"] = [
            {
                "claim_id": "C_fabricated",
                "old_language": "claim",
                "new_language": "bounded claim",
                "required_validation_before_upgrade": ["run the named check"],
            }
        ]

        blocked = validate_agenda_v2(
            agenda,
            next_gen_id=2,
            cohort_size=5,
            pi_memos={"builder": {"top_claims": [{"id": "H1"}]}},
        )

        self.assertFalse(blocked.valid)
        self.assertIn("unknown claim_id", "\n".join(blocked.blocking_issues))

        agenda["claim_boundary_updates"][0]["claim_id"] = "H1"
        allowed = validate_agenda_v2(
            agenda,
            next_gen_id=2,
            cohort_size=5,
            pi_memos={"builder": {"top_claims": [{"id": "H1"}]}},
        )
        self.assertTrue(allowed.valid, allowed.blocking_issues)

        agenda["claim_boundary_updates"] = [
            {
                "id": "C_fabricated",
                "old_language": "claim",
                "new_language": "bounded claim",
                "required_validation_before_upgrade": ["run the named check"],
            }
        ]
        blocked_id_alias = validate_agenda_v2(
            agenda,
            next_gen_id=2,
            cohort_size=5,
            pi_memos={"builder": {"top_claims": [{"claim_id": "C1"}]}},
        )
        self.assertFalse(blocked_id_alias.valid)
        self.assertIn("unknown claim_id", "\n".join(blocked_id_alias.blocking_issues))

        agenda["claim_boundary_updates"] = [
            {
                "old_language": "claim",
                "new_language": "bounded claim",
                "required_validation_before_upgrade": ["run the named check"],
            }
        ]
        blocked_missing_id = validate_agenda_v2(
            agenda,
            next_gen_id=2,
            cohort_size=5,
            pi_memos={"builder": {"top_claims": [{"claim_id": "C1"}]}},
        )
        self.assertFalse(blocked_missing_id.valid)
        self.assertIn(
            "must reference a known claim_id", "\n".join(blocked_missing_id.blocking_issues)
        )

        agenda["claim_boundary_updates"] = [
            {
                "id": "C1",
                "claim_id": "C2",
                "old_language": "claim",
                "new_language": "bounded claim",
                "required_validation_before_upgrade": ["run the named check"],
            }
        ]
        blocked_conflict = validate_agenda_v2(
            agenda,
            next_gen_id=2,
            cohort_size=5,
            pi_memos={"builder": {"top_claims": [{"claim_id": "C1"}]}},
        )
        self.assertFalse(blocked_conflict.valid)
        self.assertIn("conflicting claim_id/id", "\n".join(blocked_conflict.blocking_issues))

        agenda["claim_boundary_updates"][0]["id"] = "C1"
        agenda["claim_boundary_updates"][0].pop("claim_id", None)
        allowed_claim_id_alias = validate_agenda_v2(
            agenda,
            next_gen_id=2,
            cohort_size=5,
            pi_memos={"builder": {"top_claims": [{"claim_id": "C1"}]}},
        )
        self.assertTrue(allowed_claim_id_alias.valid, allowed_claim_id_alias.blocking_issues)

        agenda["claim_boundary_updates"] = [
            {
                "old_language": "claim",
                "new_language": "bounded claim",
                "required_validation_before_upgrade": ["run the named check"],
            }
        ]
        direct_missing_id = validate_agenda_v2(
            agenda,
            next_gen_id=2,
            cohort_size=5,
        )
        self.assertFalse(direct_missing_id.valid)
        self.assertIn(
            "must reference a known claim_id", "\n".join(direct_missing_id.blocking_issues)
        )

    def test_validation_candidate_parent_requires_validation_usage(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.agenda_validator_v2 import (
            validate_agenda_v2,
        )

        agenda = _valid_agenda(2, cohort_size=5)
        agenda["cross_peer_hypotheses"] = [
            {**agenda["cross_peer_hypotheses"][0], "id": f"H{i}"} for i in range(1, 4)
        ]
        agenda["peer_contracts"]["gen2_peer0"]["parent_candidate"] = "scout-high"
        agenda["peer_contracts"]["gen2_peer0"]["parent_usage"] = "exploit"

        blocked = validate_agenda_v2(
            agenda,
            next_gen_id=2,
            cohort_size=5,
            validation_candidate_ids={"scout-high"},
        )
        self.assertFalse(blocked.valid)
        self.assertIn("validation candidate", "\n".join(blocked.blocking_issues))

        agenda["peer_contracts"]["gen2_peer0"]["parent_usage"] = "repair"
        allowed = validate_agenda_v2(
            agenda,
            next_gen_id=2,
            cohort_size=5,
            validation_candidate_ids={"scout-high"},
        )
        self.assertTrue(allowed.valid, allowed.blocking_issues)

        agenda["peer_contracts"]["gen2_peer0"]["parent_usage"] = "ablate"
        allowed_ablate = validate_agenda_v2(
            agenda,
            next_gen_id=2,
            cohort_size=5,
            validation_candidate_ids={"scout-high"},
        )
        self.assertTrue(allowed_ablate.valid, allowed_ablate.blocking_issues)

        agenda["peer_contracts"]["gen2_peer0"]["parent_usage"] = "ablate_or_falsify"
        allowed_ablate_or_falsify = validate_agenda_v2(
            agenda,
            next_gen_id=2,
            cohort_size=5,
            validation_candidate_ids={"scout-high"},
        )
        self.assertTrue(
            allowed_ablate_or_falsify.valid,
            allowed_ablate_or_falsify.blocking_issues,
        )

        agenda["cross_peer_hypotheses"][0]["parent_candidate"] = "variant::scout-high"
        agenda["cross_peer_hypotheses"][0]["parent_usage"] = "preserve"
        blocked_hyp = validate_agenda_v2(
            agenda,
            next_gen_id=2,
            cohort_size=5,
            validation_candidate_ids={"scout-high"},
        )
        self.assertFalse(blocked_hyp.valid)
        self.assertIn("hypothesis H1", "\n".join(blocked_hyp.blocking_issues))

        agenda["cross_peer_hypotheses"][0]["parent_candidate"] = "Variant:Scout-High"
        agenda["cross_peer_hypotheses"][0]["parent_usage"] = "preserve"
        blocked_case_alias = validate_agenda_v2(
            agenda,
            next_gen_id=2,
            cohort_size=5,
            validation_candidate_ids={"variant::scout-high"},
        )
        self.assertFalse(blocked_case_alias.valid)
        self.assertIn("hypothesis H1", "\n".join(blocked_case_alias.blocking_issues))

        agenda["cross_peer_hypotheses"][0]["parent_usage"] = "validate"
        agenda["consensus_actions"] = [
            {
                "action_id": "A1",
                "claim_or_hypothesis": "H1",
                "parent_candidate": "scout-high",
                "parent_usage": "exploit",
            }
        ]
        blocked_action = validate_agenda_v2(
            agenda,
            next_gen_id=2,
            cohort_size=5,
            validation_candidate_ids={"variant::scout-high"},
        )
        self.assertFalse(blocked_action.valid)
        self.assertIn("consensus_actions[A1]", "\n".join(blocked_action.blocking_issues))

        agenda["consensus_actions"][0]["parent_usage"] = "validate"
        agenda["bridge_hypothesis"] = {
            "source_anchor_A": {"variant": "Variant:Scout-High"},
            "parent_usage": "preserve",
        }
        blocked_bridge = validate_agenda_v2(
            agenda,
            next_gen_id=2,
            cohort_size=5,
            validation_candidate_ids={"variant::scout-high"},
        )
        self.assertFalse(blocked_bridge.valid)
        self.assertIn(
            "bridge_hypothesis.source_anchor_A", "\n".join(blocked_bridge.blocking_issues)
        )

        agenda["bridge_hypothesis"] = {
            "source_anchor_A": "Variant:Scout-High",
            "parent_usage": "preserve",
        }
        blocked_scalar_bridge = validate_agenda_v2(
            agenda,
            next_gen_id=2,
            cohort_size=5,
            validation_candidate_ids={"variant::scout-high"},
        )
        self.assertFalse(blocked_scalar_bridge.valid)
        self.assertIn(
            "bridge_hypothesis.source_anchor_A",
            "\n".join(blocked_scalar_bridge.blocking_issues),
        )

        bridge_support_agenda = _valid_agenda(2, cohort_size=5)
        bridge_support_agenda["cross_peer_hypotheses"] = [
            {**bridge_support_agenda["cross_peer_hypotheses"][0], "id": f"H{i}"}
            for i in range(1, 4)
        ]
        bridge_support_agenda["bridge_hypothesis"] = {
            "source_anchor_A": {
                "variant": "mature-anchor",
                "supports": ["scout-high"],
                "source_findings": [{"finding_id": "variant::scout-high"}],
            },
            "parent_usage": "preserve",
        }
        allowed_bridge_support = validate_agenda_v2(
            bridge_support_agenda,
            next_gen_id=2,
            cohort_size=5,
            validation_candidate_ids={"variant::scout-high"},
        )
        self.assertTrue(
            allowed_bridge_support.valid,
            allowed_bridge_support.blocking_issues,
        )

        agenda["bridge_hypothesis"]["parent_usage"] = "compare"
        agenda["DISSENT_TO_EXPERIMENT"] = [
            {
                "dissent_id": "D1",
                "resolving_experiment": "validate scout",
                "parent_candidate": "scout-high",
                "parent_usage": "promote",
            }
        ]
        blocked_dissent = validate_agenda_v2(
            agenda,
            next_gen_id=2,
            cohort_size=5,
            validation_candidate_ids={"scout-high"},
        )
        self.assertFalse(blocked_dissent.valid)
        self.assertIn("DISSENT_TO_EXPERIMENT", "\n".join(blocked_dissent.blocking_issues))

        source_agenda = _valid_agenda(2, cohort_size=5)
        source_agenda["cross_peer_hypotheses"] = [
            {**source_agenda["cross_peer_hypotheses"][0], "id": f"H{i}"} for i in range(1, 4)
        ]
        source_agenda["cross_peer_hypotheses"][0]["source_findings"] = [
            {"finding_id": "scout-high"}
        ]
        allowed_source = validate_agenda_v2(
            source_agenda,
            next_gen_id=2,
            cohort_size=5,
            validation_candidate_ids={"scout-high"},
        )
        self.assertTrue(allowed_source.valid, allowed_source.blocking_issues)

        supports_agenda = _valid_agenda(2, cohort_size=5)
        supports_agenda["cross_peer_hypotheses"] = [
            {**supports_agenda["cross_peer_hypotheses"][0], "id": f"H{i}"} for i in range(1, 4)
        ]
        supports_agenda["consensus_actions"] = [
            {
                "action_id": "A2",
                "claim_or_hypothesis": "H1",
                "supports": ["results/scout/tiered_eval_summary.json"],
            }
        ]
        allowed_support = validate_agenda_v2(
            supports_agenda,
            next_gen_id=2,
            cohort_size=5,
            validation_candidate_ids={"results/scout/tiered_eval_summary.json"},
        )
        self.assertTrue(allowed_support.valid, allowed_support.blocking_issues)

        seed_agenda = _valid_agenda(2, cohort_size=5)
        seed_agenda["cross_peer_hypotheses"] = [
            {**seed_agenda["cross_peer_hypotheses"][0], "id": f"H{i}"} for i in range(1, 4)
        ]
        seed_agenda["anti_mainline_contract"]["seed_findings"] = [{"finding_id": "scout-high"}]
        allowed_seed = validate_agenda_v2(
            seed_agenda,
            next_gen_id=2,
            cohort_size=5,
            validation_candidate_ids={"scout-high"},
        )
        self.assertTrue(allowed_seed.valid, allowed_seed.blocking_issues)

        source_id_agenda = _valid_agenda(2, cohort_size=5)
        source_id_agenda["cross_peer_hypotheses"] = [
            {**source_id_agenda["cross_peer_hypotheses"][0], "id": f"H{i}"} for i in range(1, 4)
        ]
        source_id_agenda["consensus_actions"] = [
            {
                "action_id": "A3",
                "claim_or_hypothesis": "H1",
                "source_id": "scout-high",
            }
        ]
        allowed_source_id = validate_agenda_v2(
            source_id_agenda,
            next_gen_id=2,
            cohort_size=5,
            validation_candidate_ids={"scout-high"},
        )
        self.assertTrue(allowed_source_id.valid, allowed_source_id.blocking_issues)

        source_evidence_agenda = _valid_agenda(2, cohort_size=5)
        source_evidence_agenda["cross_peer_hypotheses"] = [
            {**source_evidence_agenda["cross_peer_hypotheses"][0], "id": f"H{i}"}
            for i in range(1, 4)
        ]
        source_evidence_agenda["cross_peer_hypotheses"][0]["source_evidence"] = {
            "finding_id": "scout-high"
        }
        allowed_source_evidence = validate_agenda_v2(
            source_evidence_agenda,
            next_gen_id=2,
            cohort_size=5,
            validation_candidate_ids={"scout-high"},
        )
        self.assertTrue(allowed_source_evidence.valid, allowed_source_evidence.blocking_issues)

    def test_validation_candidate_parent_usage_normalizer_repairs_prompt_vocab(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.agenda_validator_v2 import (
            normalize_validation_candidate_parent_usages,
            validate_agenda_v2,
        )

        agenda = _valid_agenda(2, cohort_size=5)
        agenda["cross_peer_hypotheses"] = [
            {**agenda["cross_peer_hypotheses"][0], "id": f"H{i}"} for i in range(1, 4)
        ]
        agenda["cross_peer_hypotheses"][0]["parent_candidate"] = "variant::scout-high"
        agenda["cross_peer_hypotheses"][0]["parent_usage"] = "preserve"
        agenda["cross_peer_hypotheses"][1]["parent_candidate"] = "variant::scout-bridge"
        agenda["cross_peer_hypotheses"][1]["parent_usage"] = "combine"
        agenda["cross_peer_hypotheses"][1]["next_step_intent"] = "combine_with_other_mechanism"
        agenda["peer_contracts"]["gen2_peer0"]["parent_candidate"] = "variant::scout-none"
        agenda["peer_contracts"]["gen2_peer0"]["parent_usage"] = "none"
        agenda["peer_contracts"]["gen2_peer0"]["next_step_intent"] = "pivot_to_distinct_surface"

        before = validate_agenda_v2(
            agenda,
            next_gen_id=2,
            cohort_size=5,
            validation_candidate_ids={
                "variant::scout-high",
                "variant::scout-bridge",
                "variant::scout-none",
            },
        )
        self.assertFalse(before.valid)

        repairs = normalize_validation_candidate_parent_usages(
            agenda,
            validation_candidate_ids={
                "variant::scout-high",
                "variant::scout-bridge",
                "variant::scout-none",
            },
        )

        self.assertEqual(len(repairs), 3)
        self.assertEqual(
            agenda["cross_peer_hypotheses"][0]["parent_usage"],
            "complete_validation",
        )
        self.assertEqual(agenda["cross_peer_hypotheses"][1]["parent_usage"], "repair")
        self.assertEqual(agenda["peer_contracts"]["gen2_peer0"]["parent_usage"], "compare")
        after = validate_agenda_v2(
            agenda,
            next_gen_id=2,
            cohort_size=5,
            validation_candidate_ids={
                "variant::scout-high",
                "variant::scout-bridge",
                "variant::scout-none",
            },
        )
        self.assertTrue(after.valid, after.blocking_issues)

    def test_validation_candidate_parent_usage_normalizer_covers_repair_vocabulary(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.agenda_validator_v2 import (
            _validation_parent_repaired_usage,
            normalize_validation_candidate_parent_usages,
        )

        agenda = _valid_agenda(2, cohort_size=2)
        agenda["consensus_actions"] = "malformed"
        agenda["mainline_observation"] = {
            "parent_candidate": "fals-id",
            "parent_usage": "preserve",
            "next_step_intent": "falsify this scout",
        }
        agenda["bridge_hypothesis"] = {
            "parent_candidate": "ablate-id",
            "parent_usage": "preserve",
            "next_step_intent": "ablate mechanism",
        }
        agenda["anti_mainline_contract"] = {
            "parent_candidate": "audit-id",
            "parent_usage": "preserve",
            "next_step_intent": "audit protocol",
        }
        agenda["falsification_contract"] = {
            "parent_candidate": "stress-id",
            "parent_usage": "preserve",
            "next_step_intent": "stress test",
        }
        agenda["success_metrics"] = {
            "parent_candidate": "full-id",
            "parent_usage": "promote",
            "evidence_stage": "full_T1_replication",
        }
        agenda["panel_summary"] = {
            "source_anchor_A": [
                {"finding_id": "scout-empty"},
                {"variant_name": "nonmatching"},
            ],
            "parent_usage": "",
        }
        agenda["peer_contracts"]["gen2_peer0"]["parent_candidate"] = "already-valid"
        agenda["peer_contracts"]["gen2_peer0"]["parent_usage"] = "validate"

        self.assertEqual(normalize_validation_candidate_parent_usages([]), [])
        self.assertEqual(
            _validation_parent_repaired_usage({"parent_usage": "compare"}),
            "compare",
        )

        repairs = normalize_validation_candidate_parent_usages(
            agenda,
            validation_candidate_ids={
                "fals-id",
                "ablate-id",
                "audit-id",
                "stress-id",
                "full-id",
                "scout-empty",
                "already-valid",
            },
        )

        by_label = {repair["label"]: repair["new_parent_usage"] for repair in repairs}
        self.assertEqual(by_label["mainline_observation"], "falsify")
        self.assertEqual(by_label["bridge_hypothesis"], "ablate")
        self.assertEqual(by_label["anti_mainline_contract"], "audit")
        self.assertEqual(by_label["falsification_contract"], "stress_validate")
        self.assertEqual(by_label["success_metrics"], "complete_validation")
        self.assertEqual(by_label["panel_summary"], "compare")
        self.assertNotIn("peer_contract gen2_peer0", by_label)
        self.assertEqual(agenda["peer_contracts"]["gen2_peer0"]["parent_usage"], "validate")

    def test_validation_candidate_denylist_uses_full_manifest_not_render_cap(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import pi_agent
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            legacy_two_round_executor,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontier = root / "frontier"
            frontier.mkdir()
            (frontier / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "validation_candidates": {
                            "validator_identity_aliases_by_generation": {
                                "1": ["uncapped_runtime_alias"],
                            },
                            "cumulative": [
                                {
                                    "generation_id": 1,
                                    "finding_id": f"scout_{idx}",
                                    "variant_name": f"scout_{idx}",
                                    "metric_name": "score",
                                    "metric_value": 100 - idx,
                                    "metric_direction": "maximize",
                                    "frontier_entity_key": f"variant::scout_{idx}",
                                    "source_result_path": (
                                        f"results/scout_{idx}/tiered_eval_summary.json"
                                    ),
                                }
                                for idx in range(17)
                            ]
                            + [
                                {
                                    "generation_id": 1,
                                    "finding_id": "scout_0_alias",
                                    "variant_name": "scout_0",
                                    "metric_name": "score",
                                    "metric_value": 1,
                                    "metric_direction": "maximize",
                                    "frontier_entity_key": "variant::scout_0",
                                    "result_artifact_path": (
                                        "results/scout_0_alias/tiered_eval_summary.json"
                                    ),
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            prompt_template = root / "prompt.jinja2"
            prompt_template.write_text("prompt", encoding="utf-8")
            pi = pi_agent.PIAgent(
                run_dir=root,
                workspace=root,
                cohort_size=5,
                model="fake",
                prompt_template_path=prompt_template,
            )

            rendered = pi._load_validation_candidates(completed_gen_id=1)
            self.assertEqual(len(rendered), 16)
            self.assertNotIn("scout_16", {entry["finding_id"] for entry in rendered})
            full_ids = pi._load_validation_candidate_ids(completed_gen_id=1)
            self.assertIn("scout_16", full_ids)
            self.assertIn("results/scout_16/tiered_eval_summary.json", full_ids)
            self.assertIn("results/scout_0_alias/tiered_eval_summary.json", full_ids)
            self.assertIn("uncapped_runtime_alias", full_ids)
            self.assertIn(
                "scout_16",
                legacy_two_round_executor._validation_candidate_ids_from_manifest(root, 1),
            )
            self.assertIn(
                "results/scout_0_alias/tiered_eval_summary.json",
                legacy_two_round_executor._validation_candidate_ids_from_manifest(root, 1),
            )
            self.assertIn(
                "uncapped_runtime_alias",
                legacy_two_round_executor._validation_candidate_ids_from_manifest(root, 1),
            )
            self.assertIn(
                "results/scout_16/tiered_eval_summary.json",
                legacy_two_round_executor._validation_candidate_ids_from_pack(
                    {
                        "audit": {
                            "validation_candidate_ids": [
                                "results/scout_16/tiered_eval_summary.json"
                            ]
                        },
                        "shared_core": {
                            "validation_candidates": [],
                        },
                    }
                ),
            )

            agenda = _valid_agenda(2)
            agenda["peer_contracts"]["gen2_peer0"]["parent_candidate"] = "scout_16"
            agenda["peer_contracts"]["gen2_peer0"]["parent_usage"] = "preserve"
            self.assertIn(
                "validation candidate",
                pi.validate_agenda(agenda, 2, validation_candidate_ids=full_ids),
            )

    def test_non_dict_peer_contract_blocks_validation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.agenda_validator_v2 import (
            validate_agenda_v2,
        )

        agenda = _valid_agenda(2, cohort_size=8)
        agenda["agenda_version"] = "2.0"
        agenda["_runtime_panel_mode"] = "full"
        agenda["cross_peer_hypotheses"] = [
            {
                "id": f"H{i}",
                "claim": "claim",
                "minimal_test": "test",
                "kill_condition": "kill",
                "promote_condition": "promote",
            }
            for i in range(1, 4)
        ]
        agenda["peer_contracts"]["gen2_peer7"] = "not a contract"

        result = validate_agenda_v2(agenda, next_gen_id=2, cohort_size=8)

        self.assertFalse(result.valid)
        self.assertIn("must be a dict", "\n".join(result.blocking_issues))

    def test_round2_invalid_peer_label_is_sanitized_against_label_map(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            legacy_two_round_executor,
        )

        parsed = {
            "role": "tester",
            "strongest_agreement": {"peer_label": "PI #Z", "claim_id": "c", "why": "x"},
            "strongest_objection": {"peer_label": "PI #A", "claim_id": "c", "objection": "x"},
            "private_kb_revealed_blind_spot": {"peer_label": "PI #Q", "triggered": True},
            "singleton_high_upside_idea_to_preserve": {"peer_label": None, "source": "self"},
        }

        out = legacy_two_round_executor._sanitize_round2_peer_labels(
            parsed,
            {"PI #A": "builder"},
        )

        self.assertIsNone(out["strongest_agreement"]["peer_label"])
        self.assertEqual(out["strongest_agreement"]["_invalid_peer_label_dropped"], "PI #Z")
        self.assertEqual(out["strongest_objection"]["peer_label"], "PI #A")
        self.assertIsNone(out["private_kb_revealed_blind_spot"]["peer_label"])

    def test_full_panel_missing_bridge_repair_adds_coverage_contract(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            legacy_two_round_executor,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.agenda_validator_v2 import (
            validate_agenda_v2,
        )

        agenda = _valid_agenda(2, cohort_size=8)
        agenda["agenda_version"] = "2.0"
        agenda["panel_mode"] = "full"
        agenda["bridge_hypothesis"] = {"id": "B_g2_01"}
        agenda["cross_peer_hypotheses"] = [
            {
                "id": f"H{i}",
                "claim": "claim",
                "minimal_test": "test",
                "kill_condition": "kill",
                "promote_condition": "promote",
            }
            for i in range(1, 4)
        ]
        for i, role in enumerate(
            [
                "exploit",
                "falsifier",
                "anti_mainline",
                "theorist",
                "exploit",
                "falsifier",
                "exploit",
                "falsifier",
            ]
        ):
            agenda["peer_contracts"][f"gen2_peer{i}"]["role"] = role
            agenda["peer_contracts"][f"gen2_peer{i}"]["parent_candidate"] = "stale_parent"
            agenda["peer_contracts"][f"gen2_peer{i}"]["parent_usage"] = "preserve"

        repairs = legacy_two_round_executor._ensure_full_panel_required_roles(
            agenda,
            panel_mode="full",
            cohort_size=8,
        )

        self.assertTrue(any(r.get("new_role") == "bridge" for r in repairs))
        self.assertIn("source_anchor_A", agenda["bridge_hypothesis"])
        self.assertIn("source_anchor_B", agenda["bridge_hypothesis"])
        bridge_contracts = [
            c
            for c in agenda["peer_contracts"].values()
            if isinstance(c, dict) and c.get("role") == "bridge"
        ]
        self.assertTrue(bridge_contracts)
        self.assertIn("coverage_check", bridge_contracts[0])
        self.assertEqual(bridge_contracts[0]["target_hypothesis"], "B_g2_01")
        result = validate_agenda_v2(agenda, next_gen_id=2, cohort_size=8)
        self.assertTrue(result.valid, result.blocking_issues)
        self.assertNotIn("bridge contract", "\n".join(result.warnings))

    def test_missing_bridge_repair_creates_bridge_hypothesis_when_absent(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            legacy_two_round_executor,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.agenda_validator_v2 import (
            validate_agenda_v2,
        )

        agenda = _valid_agenda(2, cohort_size=8)
        agenda["agenda_version"] = "2.0"
        agenda["panel_mode"] = "full"
        agenda["bridge_hypothesis"] = {}
        agenda["cross_peer_hypotheses"] = [
            {
                "id": f"H{i}",
                "claim": "claim",
                "minimal_test": "test",
                "kill_condition": "kill",
                "promote_condition": "promote",
            }
            for i in range(1, 4)
        ]
        for i, role in enumerate(
            [
                "exploit",
                "falsifier",
                "anti_mainline",
                "theorist",
                "exploit",
                "falsifier",
                "exploit",
                "falsifier",
            ]
        ):
            agenda["peer_contracts"][f"gen2_peer{i}"]["role"] = role

        repairs = legacy_two_round_executor._ensure_full_panel_required_roles(
            agenda,
            panel_mode="full",
            cohort_size=8,
        )

        self.assertTrue(any(r.get("new_role") == "bridge" for r in repairs))
        self.assertEqual(agenda["bridge_hypothesis"]["id"], "B_auto_role_repair")
        self.assertIsInstance(agenda["bridge_hypothesis"]["source_anchor_A"], dict)
        self.assertIn("variant", agenda["bridge_hypothesis"]["source_anchor_A"])
        self.assertIn("extracted_mechanism", agenda["bridge_hypothesis"]["source_anchor_A"])
        bridge_contract = [
            c
            for c in agenda["peer_contracts"].values()
            if isinstance(c, dict) and c.get("role") == "bridge"
        ][0]
        self.assertEqual(bridge_contract["target_hypothesis"], "B_auto_role_repair")
        self.assertEqual(bridge_contract["parent_usage"], "compare")
        self.assertNotEqual(bridge_contract["parent_candidate"], "stale_parent")
        for key in (
            "bottleneck_target",
            "evidence_stage",
            "tradeoff_class",
            "primary_tradeoff",
            "next_step_intent",
            "parent_candidate",
            "parent_usage",
        ):
            self.assertIn(key, bridge_contract)
            self.assertIn(key, agenda["bridge_hypothesis"])
        result = validate_agenda_v2(agenda, next_gen_id=2, cohort_size=8)
        self.assertTrue(result.valid, result.blocking_issues)

    def test_agenda_metadata_normalizer_backfills_chair_surfaces(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.agenda_metadata import (
            normalize_agenda_research_metadata,
        )

        expected = {
            "bottleneck_target": "drawdown_regression",
            "evidence_stage": "forced_T3",
            "tradeoff_class": "high_return_drawdown_repair_target",
            "primary_tradeoff": "return_vs_mdd",
            "next_step_intent": "repair_failure_mode",
            "parent_candidate": "parent_alpha",
            "parent_usage": "repair",
        }
        agenda = {
            "cross_peer_hypotheses": [
                {
                    "id": "H1",
                    "claim": "Repair the high-return parent.",
                    **expected,
                },
                {"id": "H2", "claim": "Independent bridge source."},
            ],
            "bridge_hypothesis": {
                "id": "B1",
                "source_anchor_A": {"variant": "H1", "extracted_mechanism": "H1"},
                "source_anchor_B": {"variant": "H2", "extracted_mechanism": "H2"},
            },
            "anti_mainline_contract": {"target_axes": ["distinct surface"]},
            "falsification_contract": {"target_hypothesis": "H1"},
            "consensus_actions": [{"action_id": "A1", "claim_or_hypothesis": "H1"}],
            "DISSENT_TO_EXPERIMENT": [{"dissent_id": "D1", "disputed_claim": "H1"}],
            "peer_contracts": {
                "gen2_peer0": {
                    "role": "exploit",
                    "target_hypothesis": "H1",
                    "success_signal": "test",
                },
                "gen2_peer1": {
                    "role": "falsifier",
                    "target_hypothesis": "falsification_contract",
                    "success_signal": "test",
                },
                "gen2_peer2": {
                    "role": "bridge",
                    "target_hypothesis": "B1",
                    "success_signal": "test",
                },
            },
        }

        changed = normalize_agenda_research_metadata(agenda)

        self.assertTrue(changed)
        self.assertEqual(agenda["bridge_hypothesis"]["parent_usage"], "compare")
        for surface in (
            agenda["bridge_hypothesis"],
            agenda["falsification_contract"],
            agenda["consensus_actions"][0],
            agenda["DISSENT_TO_EXPERIMENT"][0],
            agenda["peer_contracts"]["gen2_peer0"],
            agenda["peer_contracts"]["gen2_peer1"],
            agenda["peer_contracts"]["gen2_peer2"],
        ):
            for key, value in expected.items():
                if surface is agenda["bridge_hypothesis"] and key == "parent_usage":
                    continue
                self.assertEqual(surface[key], value)
        self.assertEqual(
            agenda["anti_mainline_contract"]["next_step_intent"],
            "pivot_to_distinct_surface",
        )

    def test_agenda_metadata_normalizer_preserves_explicit_labels(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.agenda_metadata import (
            normalize_agenda_research_metadata,
        )

        agenda = {
            "cross_peer_hypotheses": [
                {
                    "id": "H1",
                    "claim": "Explicit Chair label should win.",
                    "bottleneck_target": "cash_drag_or_underinvestment",
                    "evidence_stage": "T2",
                    "tradeoff_class": "diversified_low_alpha_candidate",
                    "primary_tradeoff": "return_vs_cash",
                    "next_step_intent": "repair_failure_mode",
                    "parent_candidate": "chair_parent",
                    "parent_usage": "repair",
                },
                {"id": "H2", "claim": "Missing labels should stay conservative."},
            ],
            "peer_contracts": {
                "gen2_peer0": {
                    "role": "exploit",
                    "target_hypothesis": "H1",
                    "evidence_stage": "forced_T3",
                    "next_step_intent": "ablate_or_falsify",
                },
                "gen2_peer1": {
                    "role": "exploit",
                    "target_hypothesis": "H2",
                },
            },
        }

        normalize_agenda_research_metadata(agenda)

        self.assertEqual(agenda["cross_peer_hypotheses"][0]["evidence_stage"], "T2")
        self.assertEqual(
            agenda["cross_peer_hypotheses"][0]["bottleneck_target"],
            "cash_drag_or_underinvestment",
        )
        self.assertEqual(agenda["peer_contracts"]["gen2_peer0"]["evidence_stage"], "forced_T3")
        self.assertEqual(
            agenda["peer_contracts"]["gen2_peer0"]["next_step_intent"],
            "ablate_or_falsify",
        )
        self.assertEqual(agenda["cross_peer_hypotheses"][1]["evidence_stage"], "scout")
        self.assertEqual(agenda["peer_contracts"]["gen2_peer1"]["evidence_stage"], "scout")

    def test_agenda_metadata_helper_edges_are_stable(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            agenda_metadata,
        )

        self.assertEqual(agenda_metadata.research_metadata_overrides("bad"), {})
        self.assertEqual(agenda_metadata.agenda_metadata_sources_for_target({}, None), [])
        self.assertEqual(agenda_metadata.agenda_metadata_sources_for_target({}, " "), [])
        self.assertEqual(agenda_metadata.normalize_agenda_research_metadata("bad"), [])
        self.assertFalse(agenda_metadata._nonempty("  "))
        self.assertTrue(agenda_metadata._nonempty(0))
        self.assertEqual(
            agenda_metadata.research_metadata_from_sources(
                "theorist",
                [
                    {
                        "metrics": {
                            "extra": {
                                "bottleneck_target": "nested",
                                "evidence_stage": "T2",
                            }
                        }
                    }
                ],
            )["bottleneck_target"],
            "nested",
        )

        agenda = {
            "cross_peer_hypotheses": [
                {"id": "H1", "claim": "one", "metrics": {"evidence_stage": "T3"}},
                {"id": "H2", "claim": "two", "tradeoff_class": "explicit"},
                {"id": "", "claim": "ignored"},
            ],
            "bridge_hypothesis": {
                "id": "B1",
                "source_anchor_A": "bad",
                "source_anchor_B": {"variant": "H1"},
            },
            "falsification_contract": {"target_hypothesis": "falsification_contract"},
            "consensus_actions": [
                "bad",
                {"action_id": "A1", "claim_or_hypothesis": "H1", "assigned_role": "bridge"},
            ],
            "DISSENT_TO_EXPERIMENT": [
                "bad",
                {"dissent_id": "D1", "disputed_claim": "A1", "assigned_peer_role": "falsifier"},
            ],
            "minority_high_upside": ["bad", {"idea_id": "M1", "parent_usage": "compare"}],
            "peer_contracts": {"gen1_peer0": "bad"},
        }

        changed = agenda_metadata.normalize_agenda_research_metadata(agenda)

        self.assertTrue(changed)
        self.assertEqual(
            agenda_metadata.agenda_metadata_sources_for_target(agenda, "bridge_hypothesis")[0][
                "id"
            ],
            "H1",
        )
        self.assertEqual(
            agenda_metadata.agenda_metadata_sources_for_target(
                agenda,
                "falsification_contract",
            )[-1],
            agenda["falsification_contract"],
        )
        self.assertEqual(
            agenda_metadata.agenda_metadata_sources_for_target(agenda, "A1")[0]["action_id"],
            "A1",
        )
        self.assertEqual(
            agenda_metadata.agenda_metadata_sources_for_target(agenda, "D1")[0]["dissent_id"],
            "D1",
        )
        self.assertEqual(
            agenda_metadata.agenda_metadata_sources_for_target(agenda, "M1")[0]["idea_id"],
            "M1",
        )
        self.assertEqual(agenda["bridge_hypothesis"]["evidence_stage"], "T3")

    def test_runtime_mini_panel_does_not_block_if_agenda_claims_full(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            legacy_two_round_executor,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.agenda_validator_v2 import (
            validate_agenda_v2,
        )

        agenda = _valid_agenda(2, cohort_size=4)
        agenda["agenda_version"] = "2.0"
        agenda["panel_mode"] = "full"
        agenda["cross_peer_hypotheses"] = [
            {
                "id": f"H{i}",
                "claim": "claim",
                "minimal_test": "test",
                "kill_condition": "kill",
                "promote_condition": "promote",
            }
            for i in range(1, 4)
        ]
        for i, role in enumerate(["exploit", "falsifier", "bridge", "anti_mainline"]):
            agenda["peer_contracts"][f"gen2_peer{i}"]["role"] = role

        repairs = legacy_two_round_executor._ensure_full_panel_required_roles(
            agenda,
            panel_mode="mini",
            cohort_size=4,
        )
        result = validate_agenda_v2(agenda, next_gen_id=2, cohort_size=4)

        self.assertEqual(repairs, [])
        self.assertEqual(agenda["_runtime_panel_mode"], "mini")
        self.assertTrue(result.valid, result.blocking_issues)
        self.assertIn("theorist", "\n".join(result.warnings))

    def test_agenda_loaders_validation_and_state_assembly(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import pi_agent
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_template = root / "prompt.jinja2"
            prompt_template.write_text(
                "{{ completed_gen_id }} {{ findings|length }} {{ prior_agenda|tojson }}"
                "{% if validation_candidates %} Validation candidates (non-frontier screening signals)"
                " They are **not** clean winners."
                "{% for entry in validation_candidates %} {{ entry.variant_name }}{% endfor %}"
                "{% endif %}",
                encoding="utf-8",
            )
            pi = pi_agent.PIAgent(
                run_dir=root,
                workspace=root,
                cohort_size=5,
                model="fake",
                prompt_template_path=prompt_template,
            )
            self.assertEqual(pi._load_gen_findings(0), [])
            self.assertEqual(pi._load_gen_edges(0), [])
            self.assertEqual(pi._load_prior_agenda(0), None)

            pi.agendas_dir.mkdir()
            agenda1 = _valid_agenda(1)
            (pi.agendas_dir / "research_agenda_gen1.yaml").write_text(
                "\ufeff```yaml\n" + yaml.safe_dump(agenda1, allow_unicode=True) + "\n```\n",
                encoding="utf-8",
            )
            (pi.agendas_dir / "research_agenda_gen2.yaml").write_text("[]", encoding="utf-8")
            (pi.agendas_dir / "research_agenda_gen6.yaml").write_text("", encoding="utf-8")
            (pi.agendas_dir / "research_agenda_gen7.yaml").write_text("x: [", encoding="utf-8")
            self.assertIsNotNone(
                pi_agent._parse_agenda_file(pi.agendas_dir / "research_agenda_gen1.yaml")
            )
            self.assertIsNone(pi_agent._parse_agenda_file(pi.agendas_dir / "missing.yaml"))
            self.assertIsNone(
                pi_agent._parse_agenda_file(pi.agendas_dir / "research_agenda_gen2.yaml")
            )
            self.assertIsNone(
                pi_agent._parse_agenda_file(pi.agendas_dir / "research_agenda_gen6.yaml")
            )
            self.assertIsNone(
                pi_agent._parse_agenda_file(pi.agendas_dir / "research_agenda_gen7.yaml")
            )
            summary = pi._load_prior_agendas_summary(3)
            self.assertEqual(summary[0]["generation"], 1)
            self.assertIn("H1", summary[0]["hypothesis_ids"])
            (pi.agendas_dir / "research_agenda_gen8.yaml").write_text(
                yaml.safe_dump(
                    {
                        **_valid_agenda(8),
                        "mainline_observation": {
                            "current_dominant_mechanisms": "scalar-mechanism",
                            "main_risk": None,
                            "key_tradeoff": 17,
                        },
                        "anti_mainline_contract": {"forbidden_mechanisms": "scalar-ban"},
                    }
                ),
                encoding="utf-8",
            )
            scalar_summary = pi._load_prior_agendas_summary(9)[-1]
            self.assertEqual(scalar_summary["mainline_dominant"], ["scalar-mechanism"])
            self.assertEqual(scalar_summary["main_risk"], "")
            self.assertEqual(scalar_summary["key_tradeoff"], "17")

            frontier = root / "frontier"
            frontier.mkdir()
            (frontier / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {
                            "2": {"members": [{"variant": "B", "metrics": {"score": math.inf}}]},
                            "1": [
                                {
                                    "variant": "A",
                                    "metrics": {"score": math.nan, "scored_complete": True},
                                }
                            ],
                        },
                        "validation_candidates": {
                            "cumulative": [
                                {
                                    "finding_id": "scout-high",
                                    "variant_name": "scout_high",
                                    "generation_id": 1,
                                    "metric_name": "score",
                                    "metric_value": "10.0",
                                    "metric_direction": "maximize",
                                    "recommended_next_step": "complete_scored_validation_before_frontier_or_gems",
                                    "evidence_stage": "scout",
                                    "exclusion_reason": "preliminary_or_incomplete_evidence",
                                    "scored_cell_count": 6,
                                    "metrics": {
                                        "bottleneck_target": "drawdown_regression",
                                        "tradeoff_class": "return_vs_mdd",
                                        "next_step_intent": "complete_validation",
                                    },
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            loaded_frontier = pi._load_frontier_summary()
            self.assertEqual(loaded_frontier[0]["metrics"]["score"], None)
            loaded_validation = pi._load_validation_candidates(completed_gen_id=1)
            self.assertEqual(loaded_validation[0]["finding_id"], "scout-high")
            self.assertEqual(loaded_validation[0]["evidence_stage"], "scout")
            self.assertEqual(loaded_validation[0]["evaluation_units"], 6)
            self.assertEqual(loaded_validation[0]["bottleneck_target"], "drawdown_regression")
            self.assertEqual(loaded_validation[0]["next_step_intent"], "complete_validation")
            (frontier / "frontier_manifest.json").write_text("{bad", encoding="utf-8")
            self.assertEqual(pi._load_frontier_summary(), [])

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root)}, clear=False):
                local_store.init_db()
                local_store.insert_finding(
                    {
                        "id": "f0",
                        "finding_type": "result",
                        "title": "T",
                        "content": "C",
                        "metrics": {"score": 0.8, "huge": "x" * 1000},
                        "variant_name": "V",
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "timestamp": "2026-05-12T00:00:00",
                        "extra": {
                            "x": 1,
                            "bottleneck_target": "drawdown_regression",
                            "evidence_stage": "full_T1",
                            "tradeoff_class": "high_return_drawdown_repair_target",
                            "primary_tradeoff": "return_vs_mdd",
                            "next_step_intent": "repair_failure_mode",
                            "parent_candidate": "parent_v",
                            "parent_usage": "repair",
                        },
                    }
                )
                local_store.insert_edges_batch(
                    [
                        {
                            "edge_id": "e1",
                            "src_finding_id": "f0",
                            "dst_finding_id": "f0",
                            "edge_type": "supports",
                            "confidence": 0.9,
                            "created_by": "test",
                            "rationale": "r",
                        }
                    ]
                )
                pi.db_path = root / "shared_store.db"
                loaded_gen_metrics = pi._load_gen_findings(0)[0]["metrics"]
                self.assertEqual(loaded_gen_metrics["score"], 0.8)
                self.assertEqual(loaded_gen_metrics["bottleneck_target"], "drawdown_regression")
                self.assertEqual(loaded_gen_metrics["evidence_stage"], "full_T1")
                self.assertEqual(
                    loaded_gen_metrics["tradeoff_class"],
                    "high_return_drawdown_repair_target",
                )
                self.assertEqual(loaded_gen_metrics["primary_tradeoff"], "return_vs_mdd")
                self.assertEqual(loaded_gen_metrics["next_step_intent"], "repair_failure_mode")
                self.assertEqual(loaded_gen_metrics["parent_candidate"], "parent_v")
                self.assertEqual(loaded_gen_metrics["parent_usage"], "repair")
                self.assertEqual(pi._load_gen_edges(0)[0]["edge_type"], "supports")
                self.assertEqual(
                    pi._load_prior_findings_summary(1)[0]["metrics"],
                    {
                        "score": 0.8,
                        "bottleneck_target": "drawdown_regression",
                        "evidence_stage": "full_T1",
                        "tradeoff_class": "high_return_drawdown_repair_target",
                        "primary_tradeoff": "return_vs_mdd",
                        "next_step_intent": "repair_failure_mode",
                        "parent_candidate": "parent_v",
                        "parent_usage": "repair",
                    },
                )
                self.assertEqual(pi._build_findings_summary_for_panel(0)["by_type"], {"result": 1})

            sanitized_values = pi._sanitize_json_value({1: {math.inf, object()}})["1"]
            self.assertIn(None, sanitized_values)
            self.assertEqual(pi._trim_prior_metrics("bad"), {})
            too_large = {key: "x" * 100 for key in pi._PRIOR_METRICS_ALLOWLIST}
            too_large.update(
                {
                    "bottleneck_target": "drawdown_regression",
                    "evidence_stage": "full_T1",
                    "tradeoff_class": "high_return_drawdown_repair_target",
                    "primary_tradeoff": "return_vs_mdd",
                    "next_step_intent": "repair_failure_mode",
                    "parent_candidate": "parent_v",
                    "parent_usage": "repair",
                }
            )
            self.assertTrue(pi._trim_prior_metrics(too_large)["_truncated"])
            trimmed_large = pi._trim_prior_metrics(too_large)
            self.assertEqual(trimmed_large["bottleneck_target"], "drawdown_regression")
            self.assertEqual(trimmed_large["evidence_stage"], "full_T1")
            self.assertEqual(trimmed_large["tradeoff_class"], "high_return_drawdown_repair_target")
            self.assertEqual(trimmed_large["primary_tradeoff"], "return_vs_mdd")
            self.assertEqual(trimmed_large["next_step_intent"], "repair_failure_mode")
            self.assertEqual(trimmed_large["parent_candidate"], "parent_v")
            self.assertEqual(trimmed_large["parent_usage"], "repair")
            task_metrics = {
                "future_fitness": 1.23,
                "task_primary_delta": 2.5,
                "task_lower_tail_score": -0.2,
                "task_secondary_score": 5.4,
                "task_cost_metric": 18.0,
                "tier_status": "completed_T3_forced",
                "task_surface_label": "configured_surface",
                "bottleneck_target": "drawdown_regression",
                "evidence_stage": "full_T1",
                "tradeoff_class": "high_return_drawdown_repair_target",
                "primary_tradeoff": "return_vs_mdd",
                "next_step_intent": "repair_failure_mode",
                "parent_candidate": "parent_v",
                "parent_usage": "repair",
                "ignored": "not copied",
            }
            trimmed_without_task_spec = pi._trim_prior_metrics(task_metrics)
            self.assertNotIn("task_primary_delta", trimmed_without_task_spec)
            (root / "task_spec.yaml").write_text(
                yaml.safe_dump(
                    {
                        "evaluation": {
                            "primary_metric": "future_fitness",
                            "aux_metrics": ["task_secondary_score"],
                            "anchor_metrics": [
                                {"name": "task_cost_metric", "direction": "minimize"}
                            ],
                        },
                        "gems": {
                            "primary_metric_keys": ["task_primary_delta"],
                            "secondary_metric_keys": [
                                "task_secondary_score",
                                "task_surface_label",
                            ],
                            "lower_tail_metric_keys": ["task_lower_tail_score"],
                            "cost_metric_keys": ["task_cost_metric"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            pi._task_prior_metric_names_cache = None
            trimmed = pi._trim_prior_metrics(task_metrics)
            self.assertEqual(trimmed["future_fitness"], 1.23)
            self.assertEqual(trimmed["task_primary_delta"], 2.5)
            self.assertEqual(trimmed["task_secondary_score"], 5.4)
            self.assertEqual(trimmed["task_lower_tail_score"], -0.2)
            self.assertEqual(trimmed["task_cost_metric"], 18.0)
            self.assertEqual(trimmed["task_surface_label"], "configured_surface")
            self.assertEqual(trimmed["bottleneck_target"], "drawdown_regression")
            self.assertEqual(trimmed["primary_tradeoff"], "return_vs_mdd")
            self.assertEqual(trimmed["tier_status"], "completed_T3_forced")
            self.assertNotIn("ignored", trimmed)

            prompt = pi._build_synthesis_prompt(
                completed_gen_id=1,
                findings=[{"x": object()}],
                edges=[{"y": object()}],
                frontier=loaded_frontier,
                prior_agenda=agenda1,
                prior_agendas_summary=summary,
                prior_findings_summary=[],
                agenda_output_path=root / "agenda.yaml",
                validation_candidates=loaded_validation,
            )
            self.assertIn("1 1", prompt)
            self.assertIn("Validation candidates (non-frontier screening signals)", prompt)
            self.assertIn("scout_high", prompt)
            self.assertIn("not** clean winners", prompt)
            real_prompt_pi = pi_agent.PIAgent(
                run_dir=root,
                workspace=root,
                cohort_size=5,
                model="fake",
                prompt_template_path=Path(pi_agent.__file__).with_name("synthesis_prompt.jinja2"),
            )
            real_prompt = real_prompt_pi._build_synthesis_prompt(
                completed_gen_id=1,
                findings=[],
                edges=[],
                frontier=[],
                prior_agenda=None,
                prior_agendas_summary=[],
                prior_findings_summary=[],
                agenda_output_path=root / "agenda.yaml",
                validation_candidates=loaded_validation,
            )
            self.assertIn("stage=`scout`", real_prompt)
            self.assertIn("eval_units=`6`", real_prompt)
            self.assertIn("bottleneck=`drawdown_regression`", real_prompt)
            self.assertIn("intent=`complete_validation`", real_prompt)

            self.assertEqual(pi._normalize_role("Anti-Mainline/Role"), "anti_mainline_role")
            self.assertEqual(pi.expected_peer_ids(2), [f"gen2_peer{i}" for i in range(5)])
            good = _valid_agenda(2)
            self.assertIsNone(pi.validate_agenda(good, 2))
            self.assertEqual(len(good["cross_peer_hypotheses"]), 1)
            bad_validation_parent = _valid_agenda(2)
            bad_validation_parent["peer_contracts"]["gen2_peer0"]["parent_candidate"] = (
                "Variant:Scout-High"
            )
            bad_validation_parent["peer_contracts"]["gen2_peer0"]["parent_usage"] = "preserve"
            self.assertIn(
                "validation candidate",
                pi.validate_agenda(
                    bad_validation_parent,
                    2,
                    validation_candidate_ids={"variant::scout-high"},
                ),
            )
            bad_validation_parent["peer_contracts"]["gen2_peer0"]["parent_usage"] = "repair"
            self.assertIsNone(
                pi.validate_agenda(
                    bad_validation_parent,
                    2,
                    validation_candidate_ids={"variant::scout-high"},
                )
            )
            invalid_cases = [
                [],
                {"generation": 2},
                {**_valid_agenda(2), "generation": "no digits"},
                {**_valid_agenda(2), "generation": 9},
                {**_valid_agenda(2), "cross_peer_hypotheses": []},
                {**_valid_agenda(2), "peer_contracts": []},
                {**_valid_agenda(2), "peer_contracts": {}},
                {
                    **_valid_agenda(2),
                    "peer_contracts": {**_valid_agenda(2)["peer_contracts"], "extra": {}},
                },
                {
                    **_valid_agenda(2),
                    "peer_contracts": {
                        **_valid_agenda(2)["peer_contracts"],
                        "gen2_peer0": "bad",
                    },
                },
                {
                    **_valid_agenda(2),
                    "cross_peer_hypotheses": [
                        {**_valid_agenda(2)["cross_peer_hypotheses"][0], "claim": "<one paragraph>"}
                    ],
                },
                {
                    **_valid_agenda(2),
                    "peer_contracts": {
                        **_valid_agenda(2)["peer_contracts"],
                        "gen2_peer0": {
                            **_valid_agenda(2)["peer_contracts"]["gen2_peer0"],
                            "success_signal": "<exact success signal>",
                        },
                    },
                },
                {**_valid_agenda(2), "mainline_observation": []},
                {**_valid_agenda(2), "bridge_hypothesis": []},
            ]
            for case in invalid_cases:
                self.assertIsInstance(pi.validate_agenda(case, 2), str)

            (pi.agendas_dir / "research_agenda_gen4.yaml").write_text(
                yaml.safe_dump({**_valid_agenda(4), "peer_contracts": []}),
                encoding="utf-8",
            )
            self.assertIsNone(pi_agent.load_agenda_for_gen(root, 4))
            (pi.agendas_dir / "research_agenda_gen5.yaml").write_text(
                yaml.safe_dump(_valid_agenda(5)),
                encoding="utf-8",
            )
            self.assertIsNotNone(pi_agent.load_agenda_for_gen(root, 5))
            boundary_path = root / "gen_4" / "generation_boundary.json"
            boundary_path.parent.mkdir(parents=True, exist_ok=True)
            boundary_path.write_text(
                json.dumps({"pi_status": "failed_non_strict"}),
                encoding="utf-8",
            )
            self.assertIsNone(pi_agent.load_agenda_for_gen(root, 5))
            boundary_path.write_text(
                json.dumps({"pi_status": "succeeded"}),
                encoding="utf-8",
            )
            self.assertIsNotNone(pi_agent.load_agenda_for_gen(root, 5))
            for bad_gen, payload in (
                (9, {**_valid_agenda(9), "mainline_observation": []}),
                (10, {**_valid_agenda(10), "cross_peer_hypotheses": {}}),
            ):
                (pi.agendas_dir / f"research_agenda_gen{bad_gen}.yaml").write_text(
                    yaml.safe_dump(payload),
                    encoding="utf-8",
                )
                self.assertIsNone(pi_agent.load_agenda_for_gen(root, bad_gen))

    def test_pi_invocation_and_multi_pi_fallback_paths(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import pi_agent

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = object()
            prompt_template = root / "prompt.jinja2"
            prompt_template.write_text("prompt", encoding="utf-8")
            pi = pi_agent.PIAgent(
                run_dir=root,
                workspace=root,
                cohort_size=5,
                model="fake",
                max_runtime_minutes=1,
                prompt_template_path=prompt_template,
                plugin_registry=registry,
            )
            out_path = root / "agenda.yaml"
            created_agent_kwargs: list[dict] = []

            class FakeBaseAgent:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                    created_agent_kwargs.append(kwargs)

                async def execute(self, task: str):
                    return SimpleNamespace(
                        success=True,
                        error=None,
                        request_id=self.kwargs["request_id"],
                    )

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                FakeBaseAgent,
            ):
                self.assertIsNone(
                    asyncio.run(
                        pi._invoke_synthesizer(
                            "prompt",
                            out_path,
                            request_id="legacy_pi_synthesizer_no_output",
                        )
                    )
                )

            class CanonicalWritingBaseAgent(FakeBaseAgent):
                async def execute(self, task: str):
                    out_path.write_text(yaml.safe_dump(_valid_agenda(1)), encoding="utf-8")
                    return SimpleNamespace(
                        success=True,
                        error=None,
                        request_id=self.kwargs["request_id"],
                    )

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                CanonicalWritingBaseAgent,
            ):
                self.assertIsNone(
                    asyncio.run(
                        pi._invoke_synthesizer(
                            "prompt",
                            out_path,
                            request_id="legacy_pi_synthesizer_direct",
                        )
                    )
                )
            self.assertFalse(out_path.exists())
            self.assertTrue(created_agent_kwargs)
            self.assertTrue(
                all(kwargs.get("plugin_registry") is registry for kwargs in created_agent_kwargs)
            )
            self.assertTrue(
                all(
                    kwargs.get("runtime_env_overrides")
                    == {
                        "PRAXIST_PEER_ID": kwargs["request_id"],
                        "PEER_ID": kwargs["request_id"],
                        "PRAXIST_RUN_DIR": str(root),
                        "AUTO_RESEARCH_RUN_DIR": str(root),
                    }
                    for kwargs in created_agent_kwargs
                )
            )

            class FailingBaseAgent(FakeBaseAgent):
                async def execute(self, task: str):
                    candidate = root / "peer_workspaces" / self.kwargs["request_id"] / out_path.name
                    candidate.parent.mkdir(parents=True, exist_ok=True)
                    candidate.write_text(yaml.safe_dump(_valid_agenda(1)), encoding="utf-8")
                    return SimpleNamespace(
                        success=False,
                        error="failed",
                        request_id=self.kwargs["request_id"],
                    )

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                FailingBaseAgent,
            ):
                self.assertIsNotNone(
                    asyncio.run(
                        pi._invoke_synthesizer(
                            "prompt",
                            out_path,
                            request_id="legacy_pi_synthesizer_failed",
                        )
                    )
                )

            class WorkspaceWritingBaseAgent(FakeBaseAgent):
                async def execute(self, task: str):
                    request_id = self.kwargs["request_id"]
                    candidate = root / "peer_workspaces" / request_id / out_path.name
                    candidate.parent.mkdir(parents=True, exist_ok=True)
                    candidate.write_text(
                        yaml.safe_dump(_valid_agenda(1)),
                        encoding="utf-8",
                    )
                    return SimpleNamespace(success=True, error=None, request_id=request_id)

            out_path.unlink(missing_ok=True)
            stale_other = (
                root / "peer_workspaces" / "legacy_pi_synthesizer_89abcdef" / out_path.name
            )
            stale_other.parent.mkdir(parents=True, exist_ok=True)
            stale_other.write_text(yaml.safe_dump(_valid_agenda(9)), encoding="utf-8")
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                WorkspaceWritingBaseAgent,
            ):
                recovered = asyncio.run(
                    pi._invoke_synthesizer(
                        "prompt",
                        out_path,
                        request_id="legacy_pi_synthesizer_01234567",
                    )
                )
            self.assertIsNotNone(recovered)
            self.assertEqual(recovered["generation"], "gen1")
            self.assertTrue(out_path.exists())
            self.assertFalse(
                (
                    root / "peer_workspaces" / "legacy_pi_synthesizer_01234567" / out_path.name
                ).exists()
            )

            class ConflictingOutputBaseAgent(FakeBaseAgent):
                async def execute(self, task: str):
                    request_id = self.kwargs["request_id"]
                    stale_target = root / "stale-shared-agenda.yaml"
                    stale_target.write_text(yaml.safe_dump(_valid_agenda(9)), encoding="utf-8")
                    out_path.symlink_to(stale_target)
                    candidate = root / "peer_workspaces" / request_id / out_path.name
                    candidate.parent.mkdir(parents=True, exist_ok=True)
                    candidate.write_text(yaml.safe_dump(_valid_agenda(1)), encoding="utf-8")
                    return SimpleNamespace(success=True, error=None, request_id=request_id)

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                ConflictingOutputBaseAgent,
            ):
                self.assertIsNone(
                    asyncio.run(
                        pi._invoke_synthesizer(
                            "prompt",
                            out_path,
                            request_id="legacy_pi_synthesizer_fedcba98",
                        )
                    )
                )
            self.assertFalse(out_path.exists())
            self.assertEqual(
                yaml.safe_load((root / "stale-shared-agenda.yaml").read_text(encoding="utf-8"))[
                    "generation"
                ],
                "gen9",
            )

            class RaisingBaseAgent(FakeBaseAgent):
                async def execute(self, task: str):
                    out_path.write_text(yaml.safe_dump(_valid_agenda(1)), encoding="utf-8")
                    raise RuntimeError("sdk down")

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                RaisingBaseAgent,
            ):
                self.assertIsNone(
                    asyncio.run(
                        pi._invoke_synthesizer(
                            "prompt",
                            out_path,
                            request_id="legacy_pi_synthesizer_raises",
                        )
                    )
                )
            self.assertFalse(out_path.exists())

            class FailingPanelConfig:
                fallback_to_single_pi_on_panel_failure = False
                panel_mode_default = "full"

            pi_no_fallback = pi_agent.PIAgent(
                run_dir=root,
                workspace=root,
                cohort_size=5,
                model="fake",
                prompt_template_path=prompt_template,
                use_multi_pi_panel=True,
                multi_pi_config=FailingPanelConfig(),
            )
            with patch.object(
                pi_no_fallback, "_run_multi_pi_panel", side_effect=RuntimeError("boom")
            ):
                result = asyncio.run(pi_no_fallback.run(0))
            self.assertFalse(result.success)
            self.assertIn("no fallback", result.error or "")

            pi_fallback = pi_agent.PIAgent(
                run_dir=root,
                workspace=root,
                cohort_size=5,
                model="fake",
                prompt_template_path=prompt_template,
                use_multi_pi_panel=True,
                multi_pi_config=SimpleNamespace(fallback_to_single_pi_on_panel_failure=True),
            )

            async def fake_invoke(_prompt: str, output_path: Path, *, request_id: str):
                output_path.write_text(yaml.safe_dump(_valid_agenda(1)), encoding="utf-8")
                return _valid_agenda(1)

            with (
                patch.object(pi_fallback, "_run_multi_pi_panel", side_effect=RuntimeError("boom")),
                patch.object(pi_fallback, "_invoke_synthesizer", fake_invoke),
            ):
                result = asyncio.run(pi_fallback.run(0))
            self.assertTrue(result.success)

            async def fake_panel_result_failure(_completed_gen_id: int, _out_path: Path):
                return pi_agent.PIAgentResult(
                    success=False,
                    error="panel returned failure",
                    next_gen_id=1,
                )

            with (
                patch.object(
                    pi_fallback,
                    "_run_multi_pi_panel",
                    fake_panel_result_failure,
                ),
                patch.object(pi_fallback, "_invoke_synthesizer", fake_invoke),
            ):
                result = asyncio.run(pi_fallback.run(0))
            self.assertTrue(result.success)

            async def fake_direct_final(
                _prompt: str,
                _output_path: Path,
                *,
                request_id: str,
            ):
                del request_id
                final_path = root / "agendas" / "research_agenda_gen1.yaml"
                final_path.write_text(yaml.safe_dump(_valid_agenda(1)), encoding="utf-8")
                return _valid_agenda(1)

            with (
                patch.object(
                    pi_fallback,
                    "_run_multi_pi_panel",
                    fake_panel_result_failure,
                ),
                patch.object(pi_fallback, "_invoke_synthesizer", fake_direct_final),
            ):
                result = asyncio.run(pi_fallback.run(0))
            self.assertFalse(result.success)
            self.assertFalse((root / "agendas" / "research_agenda_gen1.yaml").exists())

            async def fake_cancelled(
                _prompt: str,
                _output_path: Path,
                *,
                request_id: str,
            ):
                del request_id
                final_path = root / "agendas" / "research_agenda_gen1.yaml"
                final_path.write_text(yaml.safe_dump(_valid_agenda(1)), encoding="utf-8")
                raise asyncio.CancelledError

            with (
                patch.object(
                    pi_fallback,
                    "_run_multi_pi_panel",
                    fake_panel_result_failure,
                ),
                patch.object(pi_fallback, "_invoke_synthesizer", fake_cancelled),
                self.assertRaises(asyncio.CancelledError),
            ):
                asyncio.run(pi_fallback.run(0))
            self.assertFalse((root / "agendas" / "research_agenda_gen1.yaml").exists())

            candidate = root / "agendas" / "research_agenda_gen1.yaml.candidate"
            original_unlink = Path.unlink

            def fail_candidate_retirement(path: Path, *args, **kwargs):
                if path == candidate and path.exists():
                    raise OSError("candidate busy")
                return original_unlink(path, *args, **kwargs)

            with (
                patch.object(
                    pi_fallback,
                    "_run_multi_pi_panel",
                    fake_panel_result_failure,
                ),
                patch.object(pi_fallback, "_invoke_synthesizer", fake_invoke),
                patch.object(Path, "unlink", fail_candidate_retirement),
            ):
                result = asyncio.run(pi_fallback.run(0))
            self.assertFalse(result.success)
            self.assertIn("could not be retired", result.error or "")
            self.assertFalse((root / "agendas" / "research_agenda_gen1.yaml").exists())
            original_unlink(candidate, missing_ok=True)

            async def fake_panel_success(**_kwargs):
                return SimpleNamespace(
                    success=True,
                    agenda=_valid_agenda(1),
                    error=None,
                    panel_mode="mini",
                )

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.multi_pi.run_panel",
                fake_panel_success,
            ):
                result = asyncio.run(pi_no_fallback._run_multi_pi_panel(0, root / "panel.yaml"))
            self.assertTrue(result.success)

            async def fake_panel_failure(**_kwargs):
                return SimpleNamespace(
                    success=False,
                    agenda=None,
                    error="panel failed",
                    panel_mode="mini",
                )

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.multi_pi.run_panel",
                fake_panel_failure,
            ):
                result = asyncio.run(
                    pi_no_fallback._run_multi_pi_panel(0, root / "panel_failed.yaml")
                )
            self.assertFalse(result.success)
            self.assertEqual(result.error, "panel failed")

    def test_single_pi_allows_literature_lookup_when_server_registered(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import pi_agent

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_template = root / "prompt.jinja2"
            prompt_template.write_text("prompt", encoding="utf-8")
            pi = pi_agent.PIAgent(
                run_dir=root,
                workspace=root,
                cohort_size=5,
                model="fake",
                max_runtime_minutes=1,
                prompt_template_path=prompt_template,
                mcp_servers={"literature-lookup": object()},
            )
            out_path = root / "agenda.yaml"
            out_path.write_text(yaml.safe_dump(_valid_agenda(1)), encoding="utf-8")
            captured: list[list[str]] = []

            class FakeBaseAgent:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                    captured.append(list(kwargs.get("allowed_tools") or []))

                async def execute(self, task: str):
                    candidate = root / "peer_workspaces" / self.kwargs["request_id"] / out_path.name
                    candidate.parent.mkdir(parents=True, exist_ok=True)
                    candidate.write_text(yaml.safe_dump(_valid_agenda(1)), encoding="utf-8")
                    return SimpleNamespace(
                        success=True,
                        error=None,
                        request_id=self.kwargs["request_id"],
                    )

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                FakeBaseAgent,
            ):
                self.assertIsNotNone(
                    asyncio.run(
                        pi._invoke_synthesizer(
                            "prompt",
                            out_path,
                            request_id="legacy_pi_synthesizer_literature",
                        )
                    )
                )

            self.assertTrue(captured)
            self.assertIn("mcp__literature-lookup__literature_search", captured[-1])
            self.assertIn("mcp__literature-lookup__scientific_database_search", captured[-1])

            async def never_used_invoke(
                _prompt: str,
                _output_path: Path,
                *,
                request_id: str,
            ):
                return _valid_agenda(1)

            def timeout_wait_for(coro, timeout):
                coro.close()
                target = pi.agendas_dir / "research_agenda_gen1.yaml.candidate"
                target.write_text(yaml.safe_dump(_valid_agenda(1)), encoding="utf-8")
                raise TimeoutError

            with (
                patch.object(pi, "_invoke_synthesizer", never_used_invoke),
                patch.object(pi_agent.asyncio, "wait_for", side_effect=timeout_wait_for),
            ):
                result = asyncio.run(pi.run(0))
            self.assertFalse(result.success)
            self.assertFalse((pi.agendas_dir / "research_agenda_gen1.yaml.candidate").exists())


class ChairArbiterContractsTest(unittest.TestCase):
    def test_chair_passes_existing_plugin_registry_to_base_agent(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            chair_arbiter,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = object()
            arbiter = chair_arbiter.ChairArbiter(
                root,
                root,
                "fake",
                plugin_registry=registry,
            )
            created_agent_kwargs: list[dict[str, object]] = []

            class FakeBaseAgent:
                def __init__(self, **kwargs):
                    created_agent_kwargs.append(kwargs)

                async def execute(self, task: str):
                    return SimpleNamespace(success=False, error="done", output={})

            with (
                patch.object(arbiter, "render_prompt", return_value="prompt"),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                    FakeBaseAgent,
                ),
            ):
                asyncio.run(arbiter.run({}, {}, {}, {}, 1, 0, "full", "core"))

            self.assertEqual(len(created_agent_kwargs), 1)
            self.assertIs(created_agent_kwargs[0]["plugin_registry"], registry)

    def test_chair_prompt_injects_task_local_role_skill(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            chair_arbiter,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            role_dir = root / "roles" / "chair"
            role_dir.mkdir(parents=True)
            (role_dir / "role.yaml").write_text(
                """
role:
  role_id: chair
  display_name: Task Chair
  role_kind: chair
""",
                encoding="utf-8",
            )
            (role_dir / "skill.md").write_text(
                "ADVANTAGE FIRST CHAIR TEST SENTINEL\n"
                "Preserve localized advantages before rejecting a candidate.",
                encoding="utf-8",
            )
            arbiter = chair_arbiter.ChairArbiter(
                root,
                root,
                "fake",
                role_ref="task_role:chair",
                task_project_path=root,
            )
            prompt = arbiter.render_prompt(
                shared_core_digest={},
                pi_memos={},
                cross_reviews={},
                confidence_revisions={},
                next_gen_id=1,
                completed_gen_id=0,
                panel_mode="full",
                shared_core_id="core",
            )
            self.assertIn("Task-local Chair RoleSkill contract", prompt)
            self.assertIn("ADVANTAGE FIRST CHAIR TEST SENTINEL", prompt)

    def test_chair_prompt_gates_gem_seeded_baseline_mode(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            chair_arbiter,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arbiter = chair_arbiter.ChairArbiter(root, root, "fake")
            gems = {
                "cycle_index": 1,
                "reset_count": 1,
                "cycle_start_generation": 6,
                "entries": [{"variant_name": "gem_alpha"}],
            }
            prompt_gen0 = arbiter.render_prompt(
                shared_core_digest={"gems": gems},
                pi_memos={},
                cross_reviews={},
                confidence_revisions={},
                next_gen_id=6,
                completed_gen_id=5,
                panel_mode="full",
                shared_core_id="core",
            )
            prompt_later = arbiter.render_prompt(
                shared_core_digest={"gems": gems},
                pi_memos={},
                cross_reviews={},
                confidence_revisions={},
                next_gen_id=7,
                completed_gen_id=6,
                panel_mode="full",
                shared_core_id="core",
            )

            self.assertIn("## Gem-Seeded Baseline Mode", prompt_gen0)
            self.assertNotIn("## Gem-Seeded Baseline Mode", prompt_later)

    def test_chair_parser_and_fallback_helper_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            chair_arbiter,
        )

        self.assertEqual(chair_arbiter._strip_yaml_fence(123), "")
        self.assertEqual(chair_arbiter._strip_yaml_fence("plain"), "plain")
        self.assertEqual(chair_arbiter._unique_strings(["a", "b", "c"], limit=2), ["a", "b"])
        self.assertEqual(chair_arbiter._clip("", fallback="fallback"), "fallback")
        self.assertEqual(chair_arbiter._iter_chair_yaml_candidates(123), [])

        anchored = "intro\nagenda_version: '2.0'\npeer_contracts: {}\nclosing"
        self.assertIn("agenda_version", chair_arbiter._strip_trailing_prose(anchored))
        leading = "intro\nmainline_observation: {}\npeer_contracts: {}"
        self.assertTrue(
            chair_arbiter._strip_trailing_prose(leading).startswith("mainline_observation")
        )
        non_yaml = "not: [valid"
        parsed = chair_arbiter._parse_chair_agenda_text(non_yaml)
        self.assertIsNone(parsed.agenda)
        self.assertTrue(parsed.error)
        parsed_list = chair_arbiter._parse_chair_agenda_text("- a\n- b\n")
        self.assertIn("list", parsed_list.error or "")

        self.assertEqual(
            chair_arbiter._memo_summary(
                "builder", {"top_claims": [], "objections_or_warnings": []}
            ),
            "builder provided no active top claim.",
        )
        self.assertEqual(
            chair_arbiter._claim_supports("builder", {"id": "C1"}),
            [{"finding_id": "PI_MEMO::builder::C1", "peer": "pi_panel", "role": "builder"}],
        )
        self.assertIn(
            "claim-x",
            chair_arbiter._claim_minimal_test("builder", {"id": "claim-x"}, {}),
        )
        self.assertIsNone(chair_arbiter._first_objection({"skeptic": {"_pi_unavailable": True}}))
        self.assertIsNone(
            chair_arbiter._first_bridge_contract({"builder": {"proposed_peer_contracts": ["bad"]}})
        )
        self.assertEqual(
            chair_arbiter._minority_ideas(
                {
                    "portfolio": {
                        "proposed_experiments": [{"id": "P1", "description": "try odd axis"}]
                    }
                },
                {},
                next_gen_id=4,
            )[0]["originating_pi"],
            "portfolio",
        )
        self.assertEqual(
            chair_arbiter._claim_boundary_updates_from_round2(
                {"builder": {"own_revisions": ["bad", {"claim_id": "", "boundary_new": "drop"}]}}
            ),
            [],
        )

        fallback = chair_arbiter._build_deterministic_fallback_agenda(
            pi_memos={},
            cross_reviews={},
            next_gen_id=2,
            completed_gen_id=1,
            panel_mode="mini",
            shared_core_id="core",
            peer_budget=7,
            parse_error="x" * 800,
        )
        self.assertEqual(len(fallback["peer_contracts"]), 7)
        required_metadata = {
            "bottleneck_target",
            "evidence_stage",
            "tradeoff_class",
            "primary_tradeoff",
            "next_step_intent",
            "parent_candidate",
            "parent_usage",
        }
        for contract in fallback["peer_contracts"].values():
            self.assertTrue(required_metadata.issubset(contract))
        self.assertTrue(required_metadata.issubset(fallback["cross_peer_hypotheses"][0]))
        self.assertTrue(required_metadata.issubset(fallback["consensus_actions"][0]))
        self.assertIn("chair_yaml_parse_failed", fallback["fallback_metadata"]["reason"])
        self.assertLessEqual(len(fallback["fallback_metadata"]["parse_error"]), 500)

    def test_chair_fallback_preserves_pi_claim_research_metadata(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            chair_arbiter,
        )

        expected = {
            "bottleneck_target": "drawdown_regression",
            "evidence_stage": "forced_T3",
            "tradeoff_class": "high_return_drawdown_repair_target",
            "primary_tradeoff": "return_vs_mdd",
            "next_step_intent": "repair_failure_mode",
            "parent_candidate": "parent_alpha",
            "parent_usage": "repair",
        }
        fallback = chair_arbiter._build_deterministic_fallback_agenda(
            pi_memos={
                "builder": {
                    "top_claims": [
                        {
                            "id": "C_meta",
                            "statement": "Repair a high-return parent without losing return.",
                            "supports": ["finding_meta"],
                            "extra": dict(expected),
                        }
                    ]
                }
            },
            cross_reviews={},
            next_gen_id=2,
            completed_gen_id=1,
            panel_mode="full",
            shared_core_id="core",
            peer_budget=5,
            parse_error="bad yaml",
        )

        metadata_surfaces = [
            fallback["cross_peer_hypotheses"][0],
            fallback["falsification_contract"],
            fallback["consensus_actions"][0],
            fallback["DISSENT_TO_EXPERIMENT"][0],
            fallback["peer_contracts"]["gen2_peer0"],
            fallback["peer_contracts"]["gen2_peer1"],
            fallback["peer_contracts"]["gen2_peer2"],
            fallback["peer_contracts"]["gen2_peer4"],
        ]
        for surface in metadata_surfaces:
            for key, value in expected.items():
                self.assertEqual(surface[key], value)
        for key, value in expected.items():
            if key == "parent_usage":
                continue
            self.assertEqual(fallback["bridge_hypothesis"][key], value)
        self.assertEqual(fallback["bridge_hypothesis"]["parent_usage"], "compare")

    def test_chair_parsing_fallback_and_runtime_paths(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            chair_arbiter,
        )

        agenda_text = "Preamble\n```yaml\nagenda_version: '2.0'\npeer_contracts: {}\n```\nDone"
        parsed = chair_arbiter._parse_chair_agenda_text(agenda_text)
        self.assertIsNotNone(parsed.agenda)
        self.assertEqual(
            chair_arbiter._strip_yaml_fence("\ufeff```yaml # note\na: 1\n```").strip(), "a: 1"
        )
        self.assertEqual(
            chair_arbiter._unique_strings(["a", "a", {"finding_id": "b"}, None]), ["a", "b"]
        )
        self.assertTrue(chair_arbiter._clip("x" * 500, limit=10).endswith("…"))

        pi_memos = {
            "builder": {
                "top_claims": [
                    {
                        "id": "C1",
                        "statement": "Builder claim",
                        "supports": ["f1", {"finding_id": "f2"}],
                    }
                ],
                "proposed_experiments": [{"description": "run builder test"}],
                "proposed_peer_contracts": [
                    {"role": "bridge", "target_hypothesis": "H2", "rationale": "bridge rationale"}
                ],
            },
            "skeptic": {
                "top_claims": [{"id": "C2", "statement": "Skeptic claim"}],
                "objections_or_warnings": [
                    {
                        "target_claim": "C1",
                        "objection": "risk",
                        "resolving_experiment": "run control",
                    }
                ],
            },
            "portfolio": {"top_claims": [{"id": "C3", "statement": "Portfolio claim"}]},
            "external_validity": {"_pi_unavailable": True},
        }
        cross_reviews = {
            "skeptic": {
                "singleton_high_upside_idea_to_preserve": {
                    "idea_summary": "minority idea",
                    "source": "private",
                    "protected_budget_recommendation": "1 peer",
                },
                "own_revisions": [
                    {
                        "claim_id": "C2",
                        "boundary_old": "old",
                        "boundary_new": "new",
                        "triggered_by": "review",
                    }
                ],
            }
        }
        fallback = chair_arbiter._build_deterministic_fallback_agenda(
            pi_memos=pi_memos,
            cross_reviews=cross_reviews,
            next_gen_id=3,
            completed_gen_id=2,
            panel_mode="full",
            shared_core_id="core",
            peer_budget=5,
            parse_error="bad yaml",
        )
        self.assertEqual(len(fallback["peer_contracts"]), 5)
        self.assertEqual(fallback["minority_high_upside"][0]["rationale"], "minority idea")
        self.assertEqual(fallback["claim_boundary_updates"][0]["claim_id"], "C2")
        self.assertIn("run control", fallback["falsification_contract"]["required_controls"][0])

        with tempfile.TemporaryDirectory() as tmp:
            arbiter = chair_arbiter.ChairArbiter(
                Path(tmp), Path(tmp), "fake", max_runtime_minutes=1
            )

            class FakeBaseAgent:
                calls: list[dict[str, object]] = []
                tasks: list[str] = []

                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                    FakeBaseAgent.calls.append(kwargs)

                async def execute(self, task: str):
                    FakeBaseAgent.tasks.append(task)
                    return SimpleNamespace(
                        success=True,
                        error=None,
                        output={"text_outputs": [agenda_text]},
                    )

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                FakeBaseAgent,
            ):
                result = asyncio.run(
                    arbiter.run({}, pi_memos, cross_reviews, {}, 3, 2, "full", "core")
                )
            self.assertTrue(result.success)
            self.assertEqual(result.agenda["agenda_version"], "2.0")
            self.assertEqual(FakeBaseAgent.calls[0]["allowed_tools"], [])

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                FakeBaseAgent,
            ):
                corrected = asyncio.run(
                    arbiter.run(
                        {},
                        pi_memos,
                        cross_reviews,
                        {},
                        3,
                        2,
                        "full",
                        "core",
                        validation_feedback=(
                            "peer_contracts missing roles: ['peer_generalist'] in full panel",
                        ),
                        validation_candidate={
                            "agenda_version": "2.0",
                            "peer_contracts": {"peer0": {"role": "exploit"}},
                        },
                    )
                )
            self.assertTrue(corrected.success)
            self.assertIn("MACHINE VALIDATION CORRECTION", FakeBaseAgent.tasks[-1])
            self.assertIn("peer_generalist", FakeBaseAgent.tasks[-1])
            self.assertIn("REJECTED AGENDA CANDIDATE", FakeBaseAgent.tasks[-1])
            self.assertIn('"peer0"', FakeBaseAgent.tasks[-1])

            class BadYamlAgent(FakeBaseAgent):
                async def execute(self, task: str):
                    return SimpleNamespace(
                        success=True, error=None, output={"text_outputs": ["not agenda: x"]}
                    )

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                BadYamlAgent,
            ):
                result = asyncio.run(
                    arbiter.run({}, pi_memos, cross_reviews, {}, 3, 2, "full", "core")
                )
            self.assertTrue(result.success)
            self.assertIn("fallback_agenda", result.error or "")

            class FailingAgent(FakeBaseAgent):
                async def execute(self, task: str):
                    return SimpleNamespace(success=False, error="agent failed", output={})

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                FailingAgent,
            ):
                result = asyncio.run(
                    arbiter.run({}, pi_memos, cross_reviews, {}, 3, 2, "full", "core")
                )
            self.assertFalse(result.success)
            self.assertEqual(result.error, "agent failed")

            class EmptyOutputAgent(FakeBaseAgent):
                async def execute(self, task: str):
                    return SimpleNamespace(success=True, error=None, output="not a dict")

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                EmptyOutputAgent,
            ):
                result = asyncio.run(
                    arbiter.run({}, pi_memos, cross_reviews, {}, 3, 2, "full", "core")
                )
            self.assertTrue(result.success)
            self.assertIn("fallback_agenda", result.error or "")

            with (
                patch.object(
                    arbiter,
                    "render_prompt",
                    side_effect=RuntimeError("render failed"),
                ),
            ):
                result = asyncio.run(
                    arbiter.run({}, pi_memos, cross_reviews, {}, 3, 2, "full", "core")
                )
            self.assertFalse(result.success)
            self.assertIn("render failed", result.error or "")

            class SlowAgent(FakeBaseAgent):
                async def execute(self, task: str):
                    await asyncio.sleep(0)
                    return SimpleNamespace(
                        success=True, error=None, output={"text_outputs": [agenda_text]}
                    )

            async def fake_timeout(coro, timeout):
                coro.close()
                raise TimeoutError

            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                    SlowAgent,
                ),
                patch.object(chair_arbiter.asyncio, "wait_for", side_effect=fake_timeout),
            ):
                result = asyncio.run(
                    arbiter.run({}, pi_memos, cross_reviews, {}, 3, 2, "full", "core")
                )
            self.assertFalse(result.success)
            self.assertEqual(result.error, "chair timeout")

            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                    FakeBaseAgent,
                ),
                patch.object(
                    chair_arbiter,
                    "_parse_chair_agenda_text",
                    return_value=chair_arbiter._AgendaParseResult(
                        agenda=[],
                        cleaned_text="[]",
                    ),
                ),
            ):
                result = asyncio.run(
                    arbiter.run({}, pi_memos, cross_reviews, {}, 3, 2, "full", "core")
                )
            self.assertTrue(result.success)
            self.assertEqual(result.error, "fallback_agenda_from_non_mapping_chair_output")


class LegacyTwoRoundExecutorContractsTest(unittest.TestCase):
    def test_chair_role_coverage_retry_filter_is_narrow(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            legacy_two_round_executor as executor,
        )

        role_issue = "peer_contracts missing roles: ['peer_generalist'] in full panel"
        self.assertEqual(executor._role_coverage_retry_issues([role_issue]), (role_issue,))
        self.assertEqual(executor._role_coverage_retry_issues([]), ())
        self.assertEqual(
            executor._role_coverage_retry_issues(["unknown claim_id: H9"]),
            (),
        )
        self.assertEqual(
            executor._role_coverage_retry_issues([role_issue, "unknown claim_id: H9"]),
            (),
        )

    def test_round2_peer_anonymization_truncation_and_failure_paths(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            legacy_two_round_executor as executor,
        )

        self.assertEqual(executor._select_pi_roles("mini"), ["builder", "skeptic"])
        self.assertEqual(executor._select_chair_role_ref(), "task_role:chair")
        self.assertTrue(
            executor._has_high_stakes_signal(
                {
                    "claim_ledger_digest": {
                        "active": [{"title": "This is a generally dominant mechanism."}]
                    }
                }
            )
        )
        anon, label_map = executor._anonymize_peers(
            {"builder": {"x": 1}, "skeptic": {"x": 2}},
            me="builder",
            rng_seed=7,
        )
        self.assertEqual(set(label_map.values()), {"skeptic"})
        self.assertEqual(set(anon), {"PI #A"})
        self.assertEqual(executor._truncate_memo_for_round2(None), {})
        long_memo = {
            "top_claims": [{"id": "C1", "statement": "s" * 2000}],
            "proposed_experiments": [{"description": "x" * 2000} for _ in range(20)],
            "private_knowledge_used": "k" * 2000,
            "ignored": "drop",
        }
        truncated = executor._truncate_memo_for_round2(long_memo, max_tokens=10)
        self.assertEqual(truncated["top_claims"][0]["id"], "C1")
        self.assertLessEqual(len(truncated["proposed_experiments"]), 8)
        self.assertNotIn("ignored", truncated)

        class ParsedResult:
            def __init__(self, parsed: dict) -> None:
                self.parsed = parsed

        class FakePI:
            def __init__(self, role_name: str, parsed: dict | Exception) -> None:
                self.role_name = role_name
                self.parsed = parsed

            async def run(self, **_kwargs):
                if isinstance(self.parsed, Exception):
                    raise self.parsed
                return self.parsed

            async def run_cross_review(self, **_kwargs):
                if isinstance(self.parsed, Exception):
                    raise self.parsed
                return ParsedResult(self.parsed)

        memos = asyncio.run(
            executor._run_pi_parallel(
                [
                    FakePI("builder", {"top_claims": [{"id": "C1"}]}),
                    FakePI("skeptic", RuntimeError("panic")),
                ],
                shared_core={"facts": [1]},
                private_packs={"builder": [{"private": True}]},
                target_decisions=["decide"],
            )
        )
        self.assertEqual(memos["builder"]["top_claims"][0]["id"], "C1")
        self.assertTrue(memos["skeptic"]["_panic"])

        cross_reviews, maps = asyncio.run(
            executor._run_round2_parallel(
                [
                    FakePI(
                        "builder",
                        {
                            "own_revisions": [
                                {"claim_id": "C1", "boundary_new": "ok"},
                                {"claim_id": "hallucinated", "boundary_new": "drop"},
                            ]
                        },
                    ),
                    FakePI("skeptic", RuntimeError("round2")),
                    FakePI("portfolio", {"_parse_error": True, "error": "bad yaml"}),
                    FakePI("external_validity", RuntimeError("round2 panic")),
                ],
                pi_memos={
                    "builder": {"top_claims": [{"id": "C1"}]},
                    "skeptic": {"top_claims": [{"id": "C2"}], "_pi_unavailable": True},
                    "portfolio": {"top_claims": [{"id": "C3"}]},
                    "external_validity": {},
                },
                round2_max_runtime_minutes=1,
                rng_seed=3,
            )
        )
        self.assertEqual(
            cross_reviews["builder"]["own_revisions"], [{"claim_id": "C1", "boundary_new": "ok"}]
        )
        self.assertTrue(cross_reviews["skeptic"]["_round2_skipped"])
        self.assertTrue(cross_reviews["portfolio"]["_round2_unavailable"])
        self.assertTrue(cross_reviews["external_validity"]["_round2_failed"])
        self.assertIn("builder", maps)

    def test_legacy_executor_helper_edges_and_atomic_write_contracts(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            legacy_two_round_executor as executor,
        )

        self.assertEqual(executor.PanelMode.MINI.value, "mini")
        self.assertFalse(executor.PanelResult(success=False, panel_mode="full").success)
        self.assertEqual(executor._normalize_peer_role("Anti-Mainline"), "anti_mainline")
        self.assertEqual(executor._normalize_peer_role(None), "")
        for role in ("theorist", "bridge", "falsifier", "anti_mainline", "exploit"):
            self.assertIn("context", executor._role_repair_success_signal(role, "context"))

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "out.txt"
            executor._atomic_write_text(target, lambda f: f.write("ok"))
            self.assertEqual(target.read_text(encoding="utf-8"), "ok")

            broken = Path(tmp) / "broken" / "out.txt"

            def fail_after_write(handle) -> None:
                handle.write("partial")
                raise RuntimeError("boom")

            with self.assertRaises(RuntimeError):
                executor._atomic_write_text(broken, fail_after_write)
            self.assertFalse(broken.exists())
            self.assertEqual(list((Path(tmp) / "broken").glob("*.tmp")), [])

        agenda = {"cross_peer_hypotheses": [{"id": "H1"}]}
        self.assertEqual(
            executor._ensure_bridge_hypothesis_for_repair(agenda, "old_target"),
            "B_auto_role_repair",
        )
        self.assertIn("source_anchor_A", agenda["bridge_hypothesis"])
        existing = {
            "bridge_hypothesis": {"id": "B_existing"},
            "cross_peer_hypotheses": [{"id": "H1"}, {"id": "H2"}],
        }
        self.assertEqual(
            executor._ensure_bridge_hypothesis_for_repair(existing, None),
            "B_existing",
        )
        self.assertEqual(existing["bridge_hypothesis"]["source_anchor_B"]["variant"], "H2")

        contract: dict[str, object] = {}
        executor._apply_role_repair_contract_fields(
            existing,
            contract,
            "falsifier",
            "old_hypothesis",
        )
        self.assertEqual(contract["target_hypothesis"], "falsification_contract")
        contract = {}
        executor._apply_role_repair_contract_fields(existing, contract, "anti_mainline", "old")
        self.assertEqual(contract["source"], "auto_repaired_anti_mainline_contract")
        contract = {}
        executor._apply_role_repair_contract_fields(existing, contract, "theorist", "old")
        self.assertEqual(contract["source"], "auto_repaired_theorist_contract")
        contract = {}
        executor._apply_role_repair_contract_fields(existing, contract, "exploit", "old")
        self.assertEqual(contract["source"], "auto_repaired_exploit_contract")

        self.assertEqual(executor._ensure_full_panel_required_roles([], "full", 5), [])
        mini_agenda: dict[str, object] = {"peer_contracts": {}}
        self.assertEqual(executor._ensure_full_panel_required_roles(mini_agenda, "mini", 5), [])
        self.assertEqual(mini_agenda["_runtime_panel_mode"], "mini")
        self.assertEqual(
            executor._ensure_full_panel_required_roles({"peer_contracts": []}, "full", 5),
            [],
        )
        unrepaired = executor._ensure_full_panel_required_roles(
            {"peer_contracts": {"peer0": {"role": "exploit"}}},
            "full",
            5,
        )
        self.assertTrue(all(item["status"] == "unrepaired" for item in unrepaired))

        cleaned = executor._sanitize_round2_peer_labels(
            {
                "role": "builder",
                "strongest_agreement": {"peer_label": "PI #A"},
                "strongest_objection": {"peer_label": "PI #Z"},
                "private_kb_revealed_blind_spot": "not a dict",
                "singleton_high_upside_idea_to_preserve": {"peer_label": None},
            },
            {"PI #A": "skeptic"},
        )
        self.assertEqual(cleaned["strongest_agreement"]["peer_label"], "PI #A")
        self.assertIsNone(cleaned["strongest_objection"]["peer_label"])
        self.assertEqual(
            cleaned["strongest_objection"]["_invalid_peer_label_dropped"],
            "PI #Z",
        )
        self.assertEqual(executor._sanitize_round2_peer_labels([], {"PI #A": "x"}), [])
        self.assertEqual(
            executor._sanitize_round2_peer_labels({"x": 1}, {}),
            {"x": 1},
        )

        anon, label_map = executor._anonymize_peers({"builder": "not dict"}, "builder", 1)
        self.assertEqual(anon, {})
        self.assertEqual(label_map, {})
        memo = {
            "top_claims": [{"id": f"C{i}", "statement": "s" * 1000} for i in range(12)],
            "proposed_experiments": [{"id": f"E{i}", "description": "d" * 1000} for i in range(12)],
            "claim_boundaries": [{"claim_id": f"C{i}", "boundary": "b" * 1000} for i in range(12)],
            "private_knowledge_used": ["k" * 1000 for _ in range(12)],
        }
        truncated = executor._truncate_memo_for_round2(memo, max_tokens=10)
        self.assertEqual(len(truncated["top_claims"]), 12)
        self.assertEqual(len(truncated["claim_boundaries"]), 12)
        self.assertEqual(len(truncated["proposed_experiments"]), 8)
        self.assertIn("[truncated for round2]", truncated["top_claims"][0]["statement"])

    def test_legacy_executor_round2_no_peer_and_label_sanitization_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            legacy_two_round_executor as executor,
        )

        class ParsedResult:
            def __init__(self, parsed: dict) -> None:
                self.parsed = parsed

        class FakePI:
            role_name = "solo"

            async def run_cross_review(self, **_kwargs):
                return ParsedResult(
                    {
                        "_no_peers": True,
                        "error": "alone",
                        "strongest_agreement": {"peer_label": "PI #Z"},
                    }
                )

        cross_reviews, label_maps = asyncio.run(
            executor._run_round2_parallel(
                [FakePI()],
                {"solo": {"top_claims": [{"id": "C1"}]}},
                round2_max_runtime_minutes=1,
                rng_seed=1,
            )
        )

        self.assertTrue(cross_reviews["solo"]["_round2_unavailable"])
        self.assertTrue(cross_reviews["solo"]["_no_peers"])
        self.assertEqual(label_maps["solo"], {})

    def test_validation_alias_helper_excludes_retired_durable_candidates(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
            _validation_candidate_aliases_from_manifest,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontier = root / "frontier"
            frontier.mkdir()
            (frontier / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {
                            "1": [
                                {
                                    "generation_id": 1,
                                    "finding_id": "full_x",
                                    "variant_name": "candidate_x",
                                    "metrics": {"score": 1.0, "scored_complete": True},
                                }
                            ]
                        },
                        "validation_candidates": {
                            "generations": {
                                "0": [
                                    {
                                        "generation_id": 0,
                                        "finding_id": "scout_x",
                                        "variant_name": "candidate_x",
                                        "frontier_entity_key": "variant::candidate_x",
                                        "identity_aliases": [
                                            "scout_x",
                                            "candidate_x",
                                            "variant::candidate_x",
                                        ],
                                    }
                                ]
                            },
                            "validator_identity_aliases_by_generation": {
                                "0": ["scout_x", "candidate_x", "variant::candidate_x"]
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            aliases = _validation_candidate_aliases_from_manifest(root, current_gen_id=1)

        self.assertNotIn("scout_x", aliases)
        self.assertNotIn("variant::candidate_x", aliases)

    def test_validation_alias_helper_excludes_candidates_retired_by_gems(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
            _digest_validation_candidates,
            _validation_candidate_aliases_from_manifest,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontier = root / "frontier"
            frontier.mkdir()
            (frontier / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "gems": {
                            "entries": [
                                {
                                    "gem_finding_id": "gem_x",
                                    "variant_name": "candidate_x",
                                    "frontier_lane": "alpha_incubator",
                                    "admission_metrics": {
                                        "source_generation_id": 0,
                                        "mean_test_taskscore": 1.0,
                                        "complete_eval": True,
                                        "evidence_stage": "full_T1",
                                    },
                                }
                            ]
                        },
                        "validation_candidates": {
                            "generations": {
                                "0": [
                                    {
                                        "generation_id": 0,
                                        "finding_id": "scout_x",
                                        "variant_name": "candidate_x",
                                        "frontier_entity_key": "variant::candidate_x",
                                        "identity_aliases": [
                                            "scout_x",
                                            "candidate_x",
                                            "variant::candidate_x",
                                        ],
                                    }
                                ]
                            },
                            "validator_identity_aliases_by_generation": {
                                "0": ["scout_x", "candidate_x", "variant::candidate_x"]
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(_digest_validation_candidates(root, current_gen_id=0), [])
            self.assertEqual(
                _validation_candidate_aliases_from_manifest(root, current_gen_id=0),
                set(),
            )

    def test_validation_alias_helper_does_not_retire_candidates_with_nonclean_gems(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
            _digest_validation_candidates,
            _validation_candidate_aliases_from_manifest,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontier = root / "frontier"
            frontier.mkdir()
            (frontier / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "gems": {
                            "entries": [
                                {
                                    "gem_finding_id": "gem_x",
                                    "variant_name": "candidate_x",
                                    "frontier_lane": "alpha_incubator",
                                    "admission_metrics": {
                                        "source_generation_id": 0,
                                        "mean_test_taskscore": 1.0,
                                        "complete_eval": True,
                                        "evidence_stage": "full_T1",
                                        "clean_promotion_eligible": False,
                                    },
                                }
                            ]
                        },
                        "validation_candidates": {
                            "generations": {
                                "0": [
                                    {
                                        "generation_id": 0,
                                        "finding_id": "scout_x",
                                        "variant_name": "candidate_x",
                                        "metric_value": 7.0,
                                        "frontier_entity_key": "variant::candidate_x",
                                        "identity_aliases": [
                                            "scout_x",
                                            "candidate_x",
                                            "variant::candidate_x",
                                        ],
                                    }
                                ]
                            },
                            "validator_identity_aliases_by_generation": {
                                "0": ["scout_x", "candidate_x", "variant::candidate_x"]
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                [
                    entry["finding_id"]
                    for entry in _digest_validation_candidates(root, current_gen_id=0)
                ],
                ["scout_x"],
            )
            self.assertIn(
                "variant::candidate_x",
                _validation_candidate_aliases_from_manifest(root, current_gen_id=0),
            )

    def test_validation_digest_fallback_does_not_retire_unknown_or_nonclean_durable_sources(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
            _digest_validation_candidates,
        )

        real_import = __import__

        def fail_backend_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in {
                "praxist.plugins.workflow_stages.research_loop.backend.frontier",
                "praxist.plugins.workflow_stages.research_loop.backend.gems",
            }:
                raise ImportError("forced fallback")
            return real_import(name, globals, locals, fromlist, level)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontier = root / "frontier"
            frontier.mkdir()
            (frontier / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "cumulative_top": [
                            {
                                "generation_id": 0,
                                "variant_name": "candidate_x",
                                "frontier_entity_key": "variant::candidate_x",
                                "metric_value": 1.0,
                            },
                            {
                                "generation_id": 0,
                                "variant_name": "candidate_y",
                                "frontier_entity_key": "variant::candidate_y",
                                "metric_value": 1.0,
                                "scored_complete": True,
                            },
                        ],
                        "gems": {
                            "entries": [
                                {
                                    "gem_finding_id": "gem_z",
                                    "variant_name": "candidate_z",
                                    "admission_metrics": {
                                        "mean_test_taskscore": 1.0,
                                        "complete_eval": True,
                                        "clean_promotion_eligible": False,
                                    },
                                },
                                {
                                    "gem_finding_id": "legacy",
                                    "variant_name": "candidate_legacy",
                                    "admission_metrics": {"mean_test_taskscore": 1.0},
                                },
                            ]
                        },
                        "validation_candidates": {
                            "generations": {
                                "0": [
                                    {
                                        "generation_id": 0,
                                        "finding_id": "scout_x",
                                        "variant_name": "candidate_x",
                                        "metric_value": 7.0,
                                        "frontier_entity_key": "variant::candidate_x",
                                    },
                                    {
                                        "generation_id": 0,
                                        "finding_id": "scout_y",
                                        "variant_name": "candidate_y",
                                        "metric_value": 7.0,
                                        "frontier_entity_key": "variant::candidate_y",
                                    },
                                    {
                                        "generation_id": 0,
                                        "finding_id": "scout_z",
                                        "variant_name": "candidate_z",
                                        "metric_value": 7.0,
                                        "frontier_entity_key": "variant::candidate_z",
                                    },
                                    {
                                        "generation_id": 0,
                                        "finding_id": "scout_legacy",
                                        "variant_name": "candidate_legacy",
                                        "metric_value": 7.0,
                                        "frontier_entity_key": "variant::candidate_legacy",
                                    },
                                ]
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch("builtins.__import__", side_effect=fail_backend_import):
                visible = _digest_validation_candidates(root, current_gen_id=0, max_entries=10)

        self.assertEqual(
            {entry["finding_id"] for entry in visible},
            {"scout_x", "scout_z"},
        )

    def test_evidence_pack_validation_candidates_meta_points_to_full_artifact(
        self,
    ) -> None:
        from praxist.plugins.tools.memory_tools.adapter import resolve_source_ref
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
            build_evidence_pack,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontier = root / "frontier"
            frontier.mkdir()
            candidates = [
                {
                    "generation_id": 0,
                    "finding_id": f"scout_{idx}",
                    "variant_name": f"scout_{idx}",
                    "metric_value": float(idx),
                    "metric_name": "score",
                    "frontier_entity_key": f"variant::scout_{idx}",
                    "diversity_overlap_status": "narrow" if idx == 17 else "distinct",
                }
                for idx in range(20)
            ]
            (frontier / "frontier_manifest.json").write_text(
                json.dumps({"validation_candidates": {"generations": {"0": candidates}}}),
                encoding="utf-8",
            )

            pack = build_evidence_pack(
                root,
                panel_mode="full",
                current_gen_id=0,
                target_decisions=["plan next generation"],
                pi_roles=["builder"],
            )

            meta = pack.shared_core["validation_candidates_meta"]
            self.assertEqual(meta["total"], 20)
            self.assertEqual(meta["returned"], 16)
            self.assertTrue(meta["truncated"])
            self.assertIsInstance(meta["full_source_ref"], dict)
            resolved = resolve_source_ref(root, meta["full_source_ref"], max_generation_id=0)
            self.assertEqual(resolved["content"]["total"], 20)
            self.assertEqual(len(resolved["content"]["validation_candidates"]), 20)
            self.assertEqual(
                resolved["content"]["validation_candidates"][2]["diversity_overlap_status"],
                "narrow",
            )


if __name__ == "__main__":
    unittest.main()
