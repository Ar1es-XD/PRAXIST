from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from praxist.core.credentials import CredentialResolver
from praxist.core.redaction import redact_text, scan_file
from praxist.core.replay import dry_run, verify_run
from praxist.testing.fake_workflow_fixture import run_fake_workflow_fixture


class CoreFoundationTest(unittest.TestCase):
    def test_redaction_masks_secret_patterns(self) -> None:
        text = "Authorization: Bearer abcdefghijklmnop and key sk-test-redaction-000000"
        redacted, hits = redact_text(text)
        self.assertNotIn("Bearer abcdefghijklmnop", redacted)
        self.assertNotIn("sk-test-redaction-000000", redacted)
        self.assertTrue({"bearer_token", "openai_style_key"}.issubset(set(hits)))

    def test_credential_resolver_never_exposes_raw_secret(self) -> None:
        resolver = CredentialResolver({"ANTHROPIC_API_KEY": "sk-test-redaction-000000"})
        credential_set = resolver.discover()
        snapshot = resolver.snapshot(credential_set)
        encoded = json.dumps(snapshot)
        self.assertNotIn("sk-test-redaction-000000", encoded)
        self.assertIs(snapshot["raw_secret_fields_present"], False)
        self.assertEqual(credential_set.mode, "single")
        self.assertEqual(credential_set.credentials[0].scope, "model_provider")

    def test_fake_panel_writes_replayable_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            required = [
                "run.json",
                "startup_config.json",
                "effective_task_spec.yaml",
                "plugin_resolution.json",
                "model_profiles.json",
                "credentials_redacted.json",
                "trajectory.jsonl",
                "budget_ledger.jsonl",
                "artifact_index.jsonl",
                "run_summary.json",
                "findings/findings.jsonl",
                "findings/frontier.jsonl",
                "memory/research_memory.jsonl",
                "memory/graph_edges.jsonl",
            ]
            for rel in required:
                self.assertTrue((run_dir / rel).exists(), rel)

            self.assertTrue(verify_run(run_dir)["success"])
            self.assertTrue(dry_run(run_dir)["success"])
            self.assertEqual(scan_file(run_dir / "trajectory.jsonl"), [])

            findings = (run_dir / "findings" / "findings.jsonl").read_text().splitlines()
            frontier = (run_dir / "findings" / "frontier.jsonl").read_text().splitlines()
            self.assertGreaterEqual(len(findings), 3)
            self.assertGreaterEqual(len(frontier), 1)


if __name__ == "__main__":
    unittest.main()
