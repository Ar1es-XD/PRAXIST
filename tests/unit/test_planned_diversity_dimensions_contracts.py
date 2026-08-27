from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

DIMENSIONS = [
    {
        "name": "mechanism",
        "description": "Core computational mechanism.",
        "examples": "spectral, local",
    },
    {
        "name": "information_source",
        "description": "Signals consumed by the implementation.",
        "examples": "history, current state",
    },
]


def _valid_panel_agenda(next_gen_id: int = 2) -> dict:
    roles = ["exploit", "falsifier", "bridge", "anti_mainline", "theorist"]
    return {
        "agenda_version": "2.0",
        "panel_mode": "full",
        "mainline_observation": {},
        "cross_peer_hypotheses": [
            {
                "id": f"H{i}",
                "claim": f"claim {i}",
                "minimal_test": f"test {i}",
                "kill_condition": f"kill {i}",
                "promote_condition": f"promote {i}",
            }
            for i in range(1, 4)
        ],
        "bridge_hypothesis": {"id": "B1"},
        "anti_mainline_contract": {},
        "falsification_contract": {"target_hypothesis": "H1"},
        "peer_contracts": {
            f"gen{next_gen_id}_peer{i}": {
                "role": role,
                "target_hypothesis": "B1" if role == "bridge" else "H1",
                "success_signal": (
                    "coverage_check recorded" if role == "bridge" else "publish result"
                ),
            }
            for i, role in enumerate(roles)
        },
    }


class PlannedDiversityDimensionsContractsTest(unittest.TestCase):
    def test_pi_policy_and_planner_prompts_receive_configured_dimensions_only_when_qd_enabled(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import pi_agent
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.chair_arbiter import (
            ChairArbiter,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles._base_pi import (
            BasePI,
        )

        class QDConfig:
            enabled = True

            def pi_planning_policy(self, _generation_id: int) -> dict:
                if not self.enabled:
                    return {}
                return {
                    "enabled": True,
                    "candidate_source": "existing_pi_synthesis",
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qd = QDConfig()
            agent = pi_agent.PIAgent(
                run_dir=root,
                workspace=root,
                cohort_size=2,
                model="noop",
                quality_diversity_config=qd,
                diversity_dimensions=DIMENSIONS,
            )
            policy = agent._quality_diversity_policy(1)
            single_prompt = agent._build_synthesis_prompt(
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
            base_prompt = BasePI(root, root, "noop").render_prompt(
                shared_core={"quality_diversity_policy": policy},
                private_pack=[],
                private_kb_entries=[],
                target_decisions=[],
            )
            chair_prompt = ChairArbiter(root, root, "noop").render_prompt(
                shared_core_digest={"quality_diversity_policy": policy},
                pi_memos={},
                cross_reviews={},
                confidence_revisions={},
                next_gen_id=1,
                completed_gen_id=0,
                panel_mode="full",
                shared_core_id="core",
            )

            qd.enabled = False
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

        self.assertEqual(policy["diversity_dimensions"], DIMENSIONS)
        for prompt_text in (single_prompt, base_prompt, chair_prompt):
            normalized_prompt = " ".join(prompt_text.split())
            self.assertIn("planned_dimensions", prompt_text)
            self.assertIn("mechanism", prompt_text)
            self.assertIn("information_source", prompt_text)
            self.assertIn("never evidence", normalized_prompt)
        self.assertEqual(disabled_policy, {})
        self.assertNotIn("planned_dimensions", disabled_prompt)

    def test_panel_validator_treats_planned_dimensions_as_advisory(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.agenda_validator_v2 import (
            validate_agenda_v2,
        )

        agenda = _valid_panel_agenda()
        contracts = agenda["peer_contracts"]
        contracts["gen2_peer0"]["planned_dimensions"] = {
            "mechanism": "spectral",
            "information_source": "history",
        }
        contracts["gen2_peer2"]["planned_dimensions"] = ["not", "a", "mapping"]
        contracts["gen2_peer3"]["planned_dimensions"] = {
            "mechanism": "",
            "unconfigured_axis": "novel",
        }
        contracts["gen2_peer4"]["design_dimensions"] = {"mechanism": "implemented"}

        result = validate_agenda_v2(
            agenda,
            next_gen_id=2,
            cohort_size=5,
            diversity_dimensions=DIMENSIONS,
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.blocking_issues, [])
        warning_text = "\n".join(result.warnings)
        self.assertIn("optional planned_dimensions omitted", warning_text)
        self.assertIn("planned_dimensions should be a flat mapping", warning_text)
        self.assertIn("omits configured axes", warning_text)
        self.assertIn("includes unconfigured axes", warning_text)
        self.assertIn("blank values", warning_text)
        self.assertIn("design_dimensions is realized implementation/evidence", warning_text)

        qd_disabled = validate_agenda_v2(
            _valid_panel_agenda(),
            next_gen_id=2,
            cohort_size=5,
        )
        self.assertTrue(qd_disabled.valid)
        self.assertFalse(any("planned_dimensions" in warning for warning in qd_disabled.warnings))

    def test_peer_prompt_preserves_plan_and_requires_honest_realized_dimensions(self) -> None:
        import jinja2

        from praxist.plugins.workflow_stages.research_loop.backend import prompt_context

        class FakeFrontier:
            def get_summary(self):
                return []

        agenda = {
            "synthesized_from_gen": 0,
            "mainline_observation": {},
            "peer_contracts": {
                "gen1_peer0": {
                    "role": "exploit",
                    "target_hypothesis": "H1",
                    "success_signal": "publish result",
                    "planned_dimensions": {
                        "mechanism": "spectral",
                        "information_source": "history",
                    },
                }
            },
            "cross_peer_hypotheses": [],
            "success_metrics": {},
        }
        task_spec = SimpleNamespace(
            evaluation=SimpleNamespace(
                diversity_dimensions=DIMENSIONS,
                must_explore_axes=[],
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch(
                    "praxist.plugins.graph_maintainers.finding_graph_mvp.engine.build_session_start_graph_context",
                    return_value="",
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.pi_agent.load_agenda_for_gen",
                    return_value=agenda,
                ),
            ):
                context = prompt_context.build_prompt_context(
                    task_spec=task_spec,
                    workspace=root,
                    run_dir=root / "run",
                    results_dir=root / "results",
                    variants_dir=root / "variants",
                    findings_dir=root / "findings",
                    frontier=FakeFrontier(),
                    local_mode=True,
                    gen_id=1,
                    peer_index=0,
                    cohort_size=1,
                    strategy="pi_directed",
                )
            context["gems_context"] = {"enabled": False}

            backend = (
                Path(__file__).resolve().parents[2]
                / "praxist/plugins/workflow_stages/research_loop/backend"
            )
            env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(str(backend)),
                undefined=jinja2.StrictUndefined,
            )
            rendered = env.get_template("prompt_generation.jinja2").render(**context)

        contract = context["research_agenda"]["peer_contracts"]["gen1_peer0"]
        self.assertEqual(contract["planned_dimensions"]["mechanism"], "spectral")
        self.assertEqual(context["diversity_dimensions"], DIMENSIONS)
        self.assertIn("planned_dimensions", rendered)
        self.assertIn("plan, never evidence", " ".join(rendered.split()))
        self.assertIn("Realized / `design_dimensions`", rendered)
        self.assertIn("actual implementation", rendered)
        self.assertIn("Core computational mechanism", rendered)


if __name__ == "__main__":
    unittest.main()
