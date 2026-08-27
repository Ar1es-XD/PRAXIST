from __future__ import annotations

import unittest
from pathlib import Path


class PromptTemplateContractsTest(unittest.TestCase):
    def test_research_loop_prompts_are_task_neutral(self) -> None:
        root = Path(__file__).resolve().parents[2]
        backend = root / "praxist/plugins/workflow_stages/research_loop/backend"
        prompt_text = "\n".join(
            [
                (backend / "prompt_base.jinja2").read_text(encoding="utf-8"),
                (backend / "prompt_generation.jinja2").read_text(encoding="utf-8"),
                (backend / "synthesis_prompt.jinja2").read_text(encoding="utf-8"),
                (backend / "multi_pi" / "prompts" / "base.jinja2").read_text(encoding="utf-8"),
                (backend / "multi_pi" / "prompts" / "chair.jinja2").read_text(encoding="utf-8"),
            ]
        )

        stale_fragments = [
            "optimizer implementations",
            "optimizer hot paths",
            "raw accuracy",
            "baseline accuracy",
            "cached SGD baseline",
            "accuracy/efficiency arms",
            "acc_anchor",
            "gap_anchor",
            "primary metric alone",
        ]
        for fragment in stale_fragments:
            self.assertNotIn(fragment, prompt_text)

    def test_research_behavior_and_gems_preservation_prompt_terms(self) -> None:
        root = Path(__file__).resolve().parents[2]
        backend = root / "praxist/plugins/workflow_stages/research_loop/backend"
        prompt_text = "\n".join(
            [
                (backend / "prompt_base.jinja2").read_text(encoding="utf-8"),
                (backend / "prompt_generation.jinja2").read_text(encoding="utf-8"),
                (backend / "synthesis_prompt.jinja2").read_text(encoding="utf-8"),
                (backend / "multi_pi" / "prompts" / "base.jinja2").read_text(encoding="utf-8"),
                (backend / "multi_pi" / "prompts" / "chair.jinja2").read_text(encoding="utf-8"),
            ]
        )

        required_terms = [
            "Research Behavior Principle",
            "Stage-Gated Evidence Discipline",
            "Tradeoff-Aware Result Interpretation",
            "Preserve, Repair, Or Pivot",
            "Agenda Synthesis Rule: Preserve, Repair, Or Pivot",
            "Strong Non-Clean Candidate Handling",
            "Gems Preservation Guard",
            "bottleneck_target",
            "evidence_stage",
            "tradeoff_class",
            "next_step_intent",
            "primary_tradeoff",
            "parent_usage",
        ]
        for term in required_terms:
            self.assertIn(term, prompt_text)

        self.assertIn("not fixed peer quotas", prompt_text)
        self.assertIn("must not weaken Gems persistence", prompt_text)

        core_prompt = (backend / "prompt_base.jinja2").read_text(encoding="utf-8")
        self.assertIn("user-approved protocol permissions", core_prompt)
        self.assertIn("if the user authorized it", core_prompt)
        self.assertNotIn("Scout evidence is screening only", core_prompt)

    def test_protected_pids_prompt_commands_pin_run_dir(self) -> None:
        root = Path(__file__).resolve().parents[2]
        prompt = (
            root / "praxist/plugins/workflow_stages/research_loop/backend/prompt_base.jinja2"
        ).read_text(encoding="utf-8")

        for command in ("launch", "list", "wait"):
            snippet_start = prompt.index(f"protected_pids {command}")
            snippet = prompt[snippet_start : snippet_start + 220]
            self.assertIn("--run-dir={{ run_dir }}", snippet)
        runner_prefix = 'PYTHONPATH="$PRAXIST_WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"'
        self.assertEqual(prompt.count(runner_prefix), 3)
        self.assertEqual(prompt.count('"$PRAXIST_RUNNER_PYTHON" -m praxist.'), 3)
        self.assertNotIn("protected_pids register", prompt)
        self.assertNotIn("protected_pids unregister", prompt)
        self.assertIn("--profile=<task_declared_profile>", prompt)
        self.assertIn("--work-class=<scout|ordinary|mature>", prompt)
        self.assertIn("--retry-terminal", prompt)
        self.assertIn("never use that flag for queued, running, completed", prompt)

    def test_task_templates_mention_closing_and_stop_signals_for_long_work(self) -> None:
        root = Path(__file__).resolve().parents[2]
        prompt_paths = [
            root / "praxist/plugins/workflow_stages/research_loop/backend/prompt_base.jinja2",
            root / "templates/tasks/machine_learning_template/prompt_base.jinja2",
        ]

        for path in prompt_paths:
            prompt = path.read_text(encoding="utf-8")
            self.assertIn("CLOSING_SIGNAL", prompt)
            self.assertIn("STOP_SIGNAL", prompt)
            self.assertIn("{{ run_dir }}/gen_{{ gen_id }}/CLOSING_SIGNAL", prompt)
            self.assertIn("{{ run_dir }}/gen_{{ gen_id }}/STOP_SIGNAL", prompt)
            self.assertNotIn("{{ logs_dir }}/CLOSING_SIGNAL", prompt)
            self.assertNotIn("{{ logs_dir }}/STOP_SIGNAL", prompt)

    def test_role_contract_metadata_propagates_to_peer_findings(self) -> None:
        root = Path(__file__).resolve().parents[2]
        backend = root / "praxist/plugins/workflow_stages/research_loop/backend"
        generation_prompt = (backend / "prompt_generation.jinja2").read_text(encoding="utf-8")
        synthesis_prompt = (backend / "synthesis_prompt.jinja2").read_text(encoding="utf-8")

        self.assertIn("tradeoff_class:", synthesis_prompt)
        self.assertIn("primary_tradeoff:", synthesis_prompt)
        chair_prompt = (backend / "multi_pi" / "prompts" / "chair.jinja2").read_text(
            encoding="utf-8"
        )
        multi_pi_base = (backend / "multi_pi" / "prompts" / "base.jinja2").read_text(
            encoding="utf-8"
        )
        self.assertIn("bottleneck_target:", chair_prompt)
        self.assertIn("evidence_stage:", chair_prompt)
        self.assertIn("tradeoff_class:", chair_prompt)
        self.assertIn("primary_tradeoff:", chair_prompt)
        self.assertIn("next_step_intent:", chair_prompt)
        self.assertIn("parent_candidate:", chair_prompt)
        self.assertIn("parent_usage:", chair_prompt)
        for field in [
            "bottleneck_target:",
            "evidence_stage:",
            "tradeoff_class:",
            "primary_tradeoff:",
            "next_step_intent:",
            "parent_candidate:",
            "parent_usage:",
        ]:
            self.assertIn(field, synthesis_prompt)
        for field in [
            "bottleneck_target:",
            "evidence_stage:",
            "tradeoff_class:",
            "primary_tradeoff:",
            "next_step_intent:",
            "parent_candidate:",
            "parent_usage:",
        ]:
            self.assertIn(field, multi_pi_base)
        for field in [
            "bottleneck_target",
            "evidence_stage",
            "tradeoff_class",
            "primary_tradeoff",
            "next_step_intent",
            "parent_candidate",
            "parent_usage",
            "source_lane",
            "target_lane",
            "coverage_check",
            "mechanism_hypothesis_deliverable",
            "is_negative",
            "evidence_valence",
            "failure_mode",
            "disconfirming_claim_ids",
        ]:
            self.assertIn(f'"{field}"', generation_prompt)

        stale_extra = 'extra={"peer_role": "<your role>", "target_hypothesis": "<id>"}'
        self.assertNotIn(stale_extra, generation_prompt)
        self.assertIn("If `parent_candidate` names a validation candidate", generation_prompt)
        self.assertIn("do not mark that parent as `none` or `preserve`", generation_prompt)
        self.assertIn("If parent_candidate names a validation candidate", synthesis_prompt)
        self.assertIn("validation parent as none or preserve", synthesis_prompt)

    def test_parent_usage_prompt_examples_match_validation_followups(self) -> None:
        root = Path(__file__).resolve().parents[2]
        backend = root / "praxist/plugins/workflow_stages/research_loop/backend"
        prompt_paths = [
            backend / "prompt_generation.jinja2",
            backend / "synthesis_prompt.jinja2",
            backend / "multi_pi" / "prompts" / "base.jinja2",
            backend / "multi_pi" / "prompts" / "chair.jinja2",
        ]
        required_usages = [
            "ablate",
            "ablate_or_falsify",
            "ablation_followup",
            "complete_validation",
            "complete_scored_validation",
        ]

        for path in prompt_paths:
            prompt_text = path.read_text(encoding="utf-8")
            for usage in required_usages:
                self.assertIn(usage, prompt_text, path)

    def test_generation_prompt_renders_extended_peer_contract_fields(self) -> None:
        import jinja2

        root = Path(__file__).resolve().parents[2]
        backend = root / "praxist/plugins/workflow_stages/research_loop/backend"
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(backend)),
            undefined=jinja2.StrictUndefined,
        )
        template = env.get_template("prompt_generation.jinja2")

        rendered = template.render(
            peer_id="gen2_peer0",
            gen_id=2,
            gems_context={"enabled": False},
            research_agenda={
                "synthesized_from_gen": 1,
                "mainline_observation": {
                    "current_dominant_mechanisms": [],
                    "main_risk": "",
                    "key_tradeoff": "",
                },
                "peer_contracts": {
                    "gen2_peer0": {
                        "role": "bridge",
                        "target_hypothesis": "H_bridge",
                        "bottleneck_target": "",
                        "evidence_stage": "",
                        "tradeoff_class": "",
                        "primary_tradeoff": "",
                        "next_step_intent": "",
                        "parent_candidate": "",
                        "parent_usage": "",
                        "source_lane": "alpha_incubator",
                        "target_lane": "confirmed_alpha",
                        "coverage_check": "span alpha and risk axes",
                        "required_controls": ["control A", "control B"],
                        "mechanism_hypothesis_deliverable": "write mechanism note",
                        "forbidden_actions": ["do not change split"],
                        "success_signal": "bridge succeeds",
                    }
                },
                "cross_peer_hypotheses": [],
                "current_peer_source_context": {
                    "consensus_actions": [
                        {
                            "action_id": "A1",
                            "minimal_experiment": "run source experiment",
                        }
                    ]
                },
                "bridge_hypothesis": None,
                "anti_mainline_contract": None,
                "falsification_contract": None,
                "success_metrics": {"required": []},
            },
            peer_role_descriptions={},
            frontier_summary=[],
            local_mode=True,
            variant_hint="## DIVERSITY GUIDANCE (mandatory self-report; soft enforcement)\n"
            "Populate `design_dimensions` in the finding payload.",
        )

        self.assertIn("source=`alpha_incubator`, target=`confirmed_alpha`", rendered)
        self.assertIn("Coverage check", rendered)
        self.assertIn("span alpha and risk axes", rendered)
        self.assertIn("Required controls", rendered)
        self.assertIn("control A", rendered)
        self.assertIn("Mechanism hypothesis deliverable", rendered)
        self.assertIn("write mechanism note", rendered)
        self.assertIn("Source context", rendered)
        self.assertIn("run source experiment", rendered)
        self.assertIn('"source_lane": "alpha_incubator"', rendered)
        self.assertIn('"target_lane": "confirmed_alpha"', rendered)
        self.assertIn('"coverage_check": "span alpha and risk axes"', rendered)
        self.assertIn('"mechanism_hypothesis_deliverable": "write mechanism note"', rendered)
        self.assertIn("Role-Aligned Diversity Self-Report", rendered)
        self.assertIn("DIVERSITY GUIDANCE", rendered)
        self.assertIn("design_dimensions", rendered)
        self.assertNotIn('"required_controls": [', rendered)
        self.assertNotIn("An exact-replication label", rendered)

    def test_generation_prompt_renders_validation_signals_without_frontier(self) -> None:
        import jinja2

        root = Path(__file__).resolve().parents[2]
        backend = root / "praxist/plugins/workflow_stages/research_loop/backend"
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(backend)),
            undefined=jinja2.StrictUndefined,
        )
        template = env.get_template("prompt_generation.jinja2")

        rendered = template.render(
            peer_id="gen2_peer0",
            gen_id=2,
            gems_context={"enabled": False},
            research_agenda={},
            peer_role_descriptions={},
            frontier_summary=[],
            incubator_top_k=[],
            validation_candidate_top_k=[
                {
                    "generation_id": 1,
                    "variant_name": "needs_full_validation",
                    "metric_name": "score",
                    "metric_value": 1.2,
                    "recommended_next_step": "complete_scored_validation",
                    "effort_ratio": 0.5,
                    "coverage_ratio": 0.9,
                }
            ],
            diagnostic_control_top_k=[],
            validation_candidates=[
                {
                    "generation_id": 1,
                    "variant_name": "needs_full_validation",
                    "metric_name": "score",
                    "metric_value": 1.2,
                    "artifact_signal_status": "validation_signal",
                    "evidence_stage": "partial",
                    "recommended_next_step": "complete_scored_validation",
                    "exclusion_reason": "insufficient_mature_evidence_ratio",
                    "effort_ratio": 0.5,
                    "coverage_ratio": 0.9,
                }
            ],
            validation_candidates_meta={"truncated": False},
            research_loop_control={
                "peer_mix": {"mature_constructive_ratio": 0.25},
                "stop_audit": {"trigger_reason": "safety_cap"},
            },
            local_mode=True,
            variant_hint="continue diverse work",
        )

        self.assertIn("Strong Signal Visibility", rendered)
        self.assertIn("Revalidate-only signals", rendered)
        self.assertIn("Validation Signals", rendered)
        self.assertIn("needs_full_validation", rendered)
        self.assertIn("maturity=effort:0.5,coverage:0.9", rendered)
        self.assertIn("Research Loop Control Feedback", rendered)
        self.assertNotIn("First Generation", rendered)

    def test_machine_learning_generation_prompt_keeps_signal_views_with_agenda(self) -> None:
        import jinja2

        root = Path(__file__).resolve().parents[2]
        template_dir = root / "templates/tasks/machine_learning_template"
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            undefined=jinja2.StrictUndefined,
        )
        template = env.get_template("prompt_generation.jinja2")

        rendered = template.render(
            peer_id="gen2_peer0",
            gen_id=2,
            research_agenda={
                "mainline_observation": {
                    "current_dominant_mechanisms": ["family_a"],
                    "main_risk": "overfit",
                    "key_tradeoff": "quality_vs_cost",
                },
                "peer_contracts": {
                    "gen2_peer0": {
                        "role": "exploit",
                        "target_hypothesis": "H1",
                        "success_signal": "clean improvement",
                    }
                },
            },
            peer_role_descriptions={},
            frontier_summary=[],
            incubator_top_k=[
                {
                    "generation_id": 1,
                    "variant_name": "parentable",
                    "metric_name": "score",
                    "metric_value": 1.5,
                    "frontier_lane": "incubator",
                    "effort_ratio": 0.9,
                    "coverage_ratio": 0.9,
                }
            ],
            validation_candidate_top_k=[
                {
                    "generation_id": 1,
                    "variant_name": "needs_validation",
                    "metric_name": "score",
                    "metric_value": 1.2,
                    "recommended_next_step": "complete_scored_validation",
                }
            ],
            diagnostic_control_top_k=[],
            validation_candidates=[],
            research_loop_control={
                "peer_mix": {"constructive_target_ratio": 0.75},
                "stop_audit": {"trigger_reason": "mature_quorum"},
            },
        )

        self.assertIn("Panel Agenda", rendered)
        self.assertIn("Strong Signal Visibility", rendered)
        self.assertIn("Parentable incubator signals", rendered)
        self.assertIn("parentable", rendered)
        self.assertIn("Revalidate-only signals", rendered)
        self.assertIn("needs_validation", rendered)
        self.assertIn("Research Loop Control Feedback", rendered)
        self.assertIn("constructive_target_ratio", rendered)
        self.assertIn("If `parent_candidate` names a validation candidate", rendered)
        self.assertIn("do not mark that parent as `none` or", rendered)

    def test_generation_prompt_renders_sliced_agenda_without_unrelated_contracts(self) -> None:
        import jinja2

        root = Path(__file__).resolve().parents[2]
        backend = root / "praxist/plugins/workflow_stages/research_loop/backend"
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(backend)),
            undefined=jinja2.StrictUndefined,
        )
        template = env.get_template("prompt_generation.jinja2")

        rendered = template.render(
            peer_id="gen3_peer0",
            gen_id=3,
            gems_context={"enabled": False},
            research_agenda={
                "synthesized_from_gen": 2,
                "full_agenda_path": "/runs/demo/agendas/research_agenda_gen3.yaml",
                "mainline_observation": {"main_risk": "crowded repair lane"},
                "peer_contracts": {
                    "gen3_peer0": {
                        "role": "exploit",
                        "target_hypothesis": "H1",
                        "success_signal": "improve clean full eval",
                    }
                },
                "sibling_roster": [
                    {
                        "peer_id": "gen3_peer1",
                        "role": "falsifier",
                        "target_hypothesis": "H2",
                    }
                ],
                "cross_peer_hypotheses": [
                    {
                        "id": "H1",
                        "claim": "candidate mechanism",
                    }
                ],
                "success_metrics": {"required": ["publish clean result"]},
            },
            peer_role_descriptions={},
            frontier_summary=[],
            local_mode=True,
            variant_hint="",
        )

        self.assertIn("peer-local agenda slice", rendered)
        self.assertIn("Full cohort agenda artifact", rendered)
        self.assertIn("Sibling roster", rendered)
        self.assertIn("gen3_peer1", rendered)
        self.assertIn("candidate mechanism", rendered)
        self.assertNotIn("Bridge hypothesis", rendered)
        self.assertNotIn("Anti-mainline contract", rendered)
        self.assertNotIn("Falsification contract", rendered)


if __name__ == "__main__":
    unittest.main()
