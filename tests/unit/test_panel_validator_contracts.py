from __future__ import annotations

import unittest


class AgendaValidatorV2ContractsTest(unittest.TestCase):
    def test_agenda_validator_accepts_valid_shape_and_reports_contract_risks(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            agenda_validator_v2 as validator,
        )

        self.assertFalse(validator._is_placeholder("rho < 0.20"))
        self.assertTrue(validator._is_placeholder("<one paragraph>"))
        self.assertEqual(validator._normalize_role("Anti Mainline"), "anti_mainline")
        self.assertEqual(validator.expected_peer_ids(4, 2), ["gen4_peer0", "gen4_peer1"])

        agenda = {
            "agenda_version": "2.0",
            "panel_mode": "high_stakes",
            "shared_core_id": "core",
            "panel_summary": {"builder_summary": "b"},
            "mainline_observation": {},
            "cross_peer_hypotheses": [
                {
                    "id": "H1",
                    "claim": "claim 1",
                    "minimal_test": "test",
                    "kill_condition": "kill",
                    "promote_condition": "promote",
                },
                {
                    "id": "H2",
                    "claim": "claim 2",
                    "minimal_test": "test",
                    "kill_condition": "kill",
                    "promote_condition": "promote",
                },
                {
                    "id": "H3",
                    "claim": "claim 3",
                    "minimal_test": "test",
                    "kill_condition": "kill",
                    "promote_condition": "promote",
                },
                "dropped",
                {"claim": "missing id"},
            ],
            "bridge_hypothesis": {"id": "B1"},
            "anti_mainline_contract": {},
            "falsification_contract": {"target_hypothesis": "H1"},
            "consensus_actions": [{"action_id": "A1", "claim_or_hypothesis": "H1"}],
            "minority_high_upside": [{"idea_id": "M1"}],
            "claim_boundary_updates": [
                {
                    "claim_id": "C1",
                    "old_language": "old",
                    "new_language": "retired unless revived",
                    "revive_if": ["new evidence"],
                }
            ],
            "DISSENT_TO_EXPERIMENT": [{"dissent_id": "D1", "resolving_experiment": "run"}],
            "peer_contracts": {
                "gen4_peer0": {
                    "role": "exploit",
                    "target_hypothesis": "H1",
                    "success_signal": "success",
                },
                "gen4_peer1": {
                    "role": "falsifier",
                    "target_hypothesis": "H2",
                    "success_signal": "success",
                },
                "gen4_peer2": {
                    "role": "bridge",
                    "target_hypothesis": "B1",
                    "success_signal": "coverage_matrix reported",
                },
                "gen4_peer3": {
                    "role": "anti_mainline",
                    "target_hypothesis": "anti_mainline_contract",
                    "success_signal": "success",
                },
                "gen4_peer4": {
                    "role": "theorist",
                    "target_hypothesis": "H3",
                    "success_signal": "success",
                },
            },
        }
        pi_memos = {
            "builder": {
                "top_claims": [{"id": "C1"}],
                "proposed_peer_contracts": [{"target_hypothesis": "H2"}],
            },
            "external_validity": {"_pi_unavailable": True, "top_claims": [{"id": "ignored"}]},
        }
        known_ids = validator._collect_known_claim_ids(agenda, pi_memos)
        self.assertIn("C1", known_ids)
        self.assertNotIn("ignored", known_ids)
        result = validator.validate_agenda_v2(
            agenda,
            next_gen_id=4,
            cohort_size=5,
            pi_memos=pi_memos,
        )
        self.assertTrue(result.valid)
        self.assertTrue(any("external_validity" in warning for warning in result.warnings))

        malformed_inputs = [
            [],
            {"cross_peer_hypotheses": []},
            {**agenda, "mainline_observation": []},
            {**agenda, "cross_peer_hypotheses": "bad"},
            {**agenda, "peer_contracts": []},
            {
                **agenda,
                "peer_contracts": {**agenda["peer_contracts"], "bad_key": {}},
            },
            {
                **agenda,
                "cross_peer_hypotheses": [
                    {**agenda["cross_peer_hypotheses"][0], "claim": "<exact id>"},
                    *agenda["cross_peer_hypotheses"][1:3],
                ],
            },
            {
                **agenda,
                "peer_contracts": {
                    **agenda["peer_contracts"],
                    "gen4_peer0": {
                        **agenda["peer_contracts"]["gen4_peer0"],
                        "target_hypothesis": "<peer_id>",
                    },
                },
            },
            {
                **agenda,
                "claim_boundary_updates": [
                    {"claim_id": "C2", "old_language": "obsolete", "new_language": "retired"}
                ],
            },
            {
                **agenda,
                "DISSENT_TO_EXPERIMENT": [{"dissent_id": "D2"}],
            },
        ]
        for bad in malformed_inputs:
            self.assertFalse(
                validator.validate_agenda_v2(
                    bad,
                    next_gen_id=4,
                    cohort_size=5,
                    pi_memos=pi_memos,
                ).valid
            )

        warning_result = validator.validate_agenda_v2(
            {
                **agenda,
                "panel_mode": "unknown",
                "peer_contracts": {
                    **agenda["peer_contracts"],
                    "gen4_peer0": {"role": "unknown", "target_hypothesis": "fabricated"},
                    "gen4_peer2": {
                        "role": "bridge",
                        "target_hypothesis": "B1",
                        "success_signal": "missing coverage note",
                    },
                },
            },
            next_gen_id=4,
            cohort_size=5,
            pi_memos=pi_memos,
        )
        self.assertTrue(warning_result.valid)
        self.assertGreaterEqual(len(warning_result.warnings), 3)


if __name__ == "__main__":
    unittest.main()
