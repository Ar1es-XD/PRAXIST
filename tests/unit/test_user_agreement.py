"""Contracts for the canonical User Agreement and first-use acceptance."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from praxist import user_agreement as agreement_model
from praxist.cli import user_agreement as agreement_cli
from praxist.product_usage.notice import consent_notice_v2
from praxist.user_agreement import (
    FAIR_SOURCE_LICENSE_VERSION,
    USER_AGREEMENT_VERSION,
    current_acceptance,
    fair_source_license_text,
    product_usage_notice_text,
    record_acceptance,
    user_agreement_text,
)


class UserAgreementDocumentTest(unittest.TestCase):
    def test_documents_match_current_product_and_have_no_release_placeholders(self) -> None:
        agreement = user_agreement_text()
        normalized = " ".join(agreement.split())
        self.assertNotIn("【", agreement)
        self.assertNotIn("AI" + " Scientist", agreement)
        self.assertNotIn("web-based", agreement)
        self.assertNotIn("replace with official", agreement.lower())
        self.assertNotIn("development placeholder", agreement.lower())
        self.assertIn("does not require a separate Praxist account", agreement)
        self.assertIn("# Fair Source License Agreement (Version 1.0)", agreement)
        self.assertIn("Sapi" + "ent Intelligence Pte Ltd", agreement)
        self.assertIn("**Contact:** praxist@sapi" + "ent.inc", agreement)
        self.assertEqual(FAIR_SOURCE_LICENSE_VERSION, "1.0")
        self.assertIn("does not enable product-usage collection", normalized)
        self.assertIn("stays on the local machine", normalized)
        self.assertIn("local outbox for later delivery", normalized)
        self.assertEqual(agreement.count("# Appendix A:"), 1)

    def test_license_loader_uses_the_canonical_root_document(self) -> None:
        expected = (Path(__file__).resolve().parents[2] / "LICENSE.md").read_text(encoding="utf-8")
        self.assertEqual(fair_source_license_text(), expected.strip())

    def test_license_keeps_product_usage_collection_optional(self) -> None:
        license_text = fair_source_license_text()
        normalized = " ".join(license_text.split())

        self.assertIn("does not constitute consent to product-usage collection", normalized)
        self.assertIn("Declining or withdrawing consent shall not reduce", normalized)
        self.assertNotIn("data collection obligation", normalized)
        self.assertNotIn("generation-count data collection mechanism", normalized)
        self.assertNotIn('count as one (1) "Generation" event', normalized)
        self.assertNotIn("[Singapore]", license_text)
        self.assertNotIn("[Singapore International Arbitration Centre]", license_text)

    def test_installed_license_loader_uses_standard_distribution_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            entry = Path("praxist-0.3.0.dist-info/licenses/LICENSE.md")
            installed = root / entry
            installed.parent.mkdir(parents=True)
            installed.write_text("installed license", encoding="utf-8")
            distribution = Mock(files=[entry])
            distribution.locate_file.side_effect = lambda path: root / path
            missing_source = root / "site-packages" / "praxist" / "user_agreement.py"
            with (
                patch.object(agreement_model.resources, "files", return_value=root),
                patch.object(agreement_model, "__file__", str(missing_source)),
                patch.object(agreement_model.metadata, "distribution", return_value=distribution),
            ):
                self.assertEqual(fair_source_license_text(), "installed license")

    def test_product_usage_cli_loads_the_same_authored_appendix(self) -> None:
        self.assertEqual(consent_notice_v2(), product_usage_notice_text())
        self.assertIn("**Notice version:** 3", consent_notice_v2())

    def test_packaged_document_is_preferred_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            packaged = root / "resources" / "docs" / "legal" / "sample.md"
            packaged.parent.mkdir(parents=True)
            packaged.write_text("packaged", encoding="utf-8")
            with patch.object(agreement_model.resources, "files", return_value=root):
                self.assertEqual(agreement_model._document_text("legal/sample.md"), "packaged")


class UserAgreementRecordTest(unittest.TestCase):
    def test_acceptance_is_bound_to_current_version_and_exact_text(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "agreement.json"
            written = record_acceptance(source="direct", path=path)
            self.assertEqual(written.agreement_version, USER_AGREEMENT_VERSION)
            self.assertEqual(current_acceptance(path), written)
            if os.name == "posix":
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["agreement_sha256"] = "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(current_acceptance(path))

    def test_malformed_or_unknown_source_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "agreement.json"
            path.write_text("not-json", encoding="utf-8")
            self.assertIsNone(current_acceptance(path))
            record_acceptance(source="agent", path=path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["source"] = "inferred"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(current_acceptance(path))

    def test_stale_version_and_invalid_writer_source_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "agreement.json"
            record_acceptance(source="direct", path=path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["agreement_version"] = "stale"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(current_acceptance(path))
            with self.assertRaisesRegex(ValueError, "source"):
                record_acceptance(source="inferred", path=path)  # type: ignore[arg-type]


class UserAgreementCliTest(unittest.TestCase):
    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = agreement_cli.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_agent_acceptance_requires_exact_operator_reply(self) -> None:
        with tempfile.TemporaryDirectory() as raw, patch.dict(os.environ, {"XDG_CONFIG_HOME": raw}):
            code, _out, err = self._run(["accept", "--agent-reply", "Yes"])
            self.assertEqual(code, 2)
            self.assertIn("exact `Agree`", err)

            code, out, err = self._run(["accept", "--agent-reply", "Agree"])
            self.assertEqual(code, 0, msg=out + err)
            code, out, err = self._run(["status", "--json"])
            self.assertEqual(code, 0, msg=out + err)
            status = json.loads(out)
            self.assertTrue(status["accepted"])
            self.assertEqual(status["source"], "agent")

    def test_noninteractive_review_links_instead_of_dumping_legal_text(self) -> None:
        with patch.object(agreement_cli, "interactive_terminal_available", return_value=False):
            code, out, err = self._run(["review"])
        self.assertEqual(code, 0)
        self.assertEqual(
            out.strip().splitlines(),
            [agreement_cli.FAIR_SOURCE_LICENSE_URL, agreement_cli.USER_AGREEMENT_URL],
        )
        self.assertNotIn("Chapter 1", out)
        self.assertIn("--print", err)

    def test_help_print_review_and_text_status_paths(self) -> None:
        code, out, err = self._run([])
        self.assertEqual(code, 0, msg=out + err)
        self.assertIn("usage:", out)

        code, out, err = self._run(["review", "--print"])
        self.assertEqual(code, 0, msg=out + err)
        self.assertIn("# Praxist User Agreement", out)

        with tempfile.TemporaryDirectory() as raw, patch.dict(os.environ, {"XDG_CONFIG_HOME": raw}):
            code, out, err = self._run(["status"])
            self.assertEqual(code, 0, msg=out + err)
            self.assertIn("not accepted", out)
            record_acceptance(source="direct")
            code, out, err = self._run(["status"])
            self.assertEqual(code, 0, msg=out + err)
            self.assertIn("accepted", out)
            self.assertIn("direct", out)
            code, out, err = self._run(["accept"])
            self.assertEqual(code, 0, msg=out + err)
            self.assertIn("already accepted", out)

    def test_terminal_failures_are_reported_without_accepting(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch.dict(os.environ, {"XDG_CONFIG_HOME": raw}),
            patch.object(agreement_cli, "interactive_terminal_available", return_value=True),
            patch.object(
                agreement_cli,
                "_review_in_terminal",
                side_effect=agreement_cli.TerminalInteractionError("terminal unavailable"),
            ),
        ):
            code, _out, err = self._run(["review"])
            self.assertEqual(code, 130)
            self.assertIn("terminal unavailable", err)

        with (
            tempfile.TemporaryDirectory() as raw,
            patch.dict(os.environ, {"XDG_CONFIG_HOME": raw}),
            patch.object(agreement_cli, "interactive_terminal_available", return_value=False),
        ):
            code, _out, err = self._run(["accept"])
            self.assertEqual(code, 2)
            self.assertIn("interactive terminal", err)

    def test_interactive_review_success_and_cancelled_acceptance(self) -> None:
        with (
            patch.object(agreement_cli, "interactive_terminal_available", return_value=True),
            patch.object(agreement_cli, "_review_in_terminal") as review,
        ):
            code, out, err = self._run(["review"])
        self.assertEqual(code, 0, msg=out + err)
        review.assert_called_once()

        with (
            tempfile.TemporaryDirectory() as raw,
            patch.dict(os.environ, {"XDG_CONFIG_HOME": raw}),
            patch.object(agreement_cli, "interactive_terminal_available", return_value=True),
            patch.object(agreement_cli, "prompt_for_acceptance_if_needed", return_value=False),
        ):
            code, _out, _err = self._run(["accept"])
        self.assertEqual(code, 130)

    def test_interactive_accept_failure_and_review_fallback(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch.dict(os.environ, {"XDG_CONFIG_HOME": raw}),
            patch.object(agreement_cli, "interactive_terminal_available", return_value=True),
            patch.object(
                agreement_cli,
                "prompt_for_acceptance_if_needed",
                side_effect=agreement_cli.TerminalInteractionCancelled("cancelled"),
            ),
        ):
            code, _out, err = self._run(["accept"])
            self.assertEqual(code, 130)
            self.assertIn("cancelled", err)

        output = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as raw,
            patch.dict(os.environ, {"XDG_CONFIG_HOME": raw}),
            patch.object(agreement_cli, "select_choice", side_effect=["review", "cancel"]),
            patch.object(
                agreement_cli,
                "_review_in_terminal",
                side_effect=agreement_cli.TerminalInteractionError("no terminal"),
            ),
        ):
            self.assertFalse(agreement_cli.prompt_for_acceptance_if_needed(output_stream=output))
        self.assertIn(agreement_cli.USER_AGREEMENT_URL, output.getvalue())
        self.assertIn(agreement_cli.FAIR_SOURCE_LICENSE_URL, output.getvalue())

    def test_current_acceptance_short_circuits_gate_and_view_delegates(self) -> None:
        with tempfile.TemporaryDirectory() as raw, patch.dict(os.environ, {"XDG_CONFIG_HOME": raw}):
            record_acceptance(source="direct")
            with patch.object(agreement_cli, "select_choice") as select:
                self.assertTrue(agreement_cli.prompt_for_acceptance_if_needed())
            select.assert_not_called()

        with patch.object(agreement_cli, "view_scrollable_text") as view:
            agreement_cli._review_in_terminal(output_stream=io.StringIO())
        view.assert_called_once()

    def test_interactive_gate_reviews_then_records_direct_acceptance(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch.dict(os.environ, {"XDG_CONFIG_HOME": raw}),
            patch.object(
                agreement_cli,
                "select_choice",
                side_effect=["review", "agree"],
            ),
            patch.object(agreement_cli, "_review_in_terminal") as review,
        ):
            self.assertTrue(agreement_cli.prompt_for_acceptance_if_needed())
            self.assertIsNotNone(current_acceptance())
        review.assert_called_once()

    def test_cancel_does_not_create_acceptance(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch.dict(os.environ, {"XDG_CONFIG_HOME": raw}),
            patch.object(agreement_cli, "select_choice", return_value="cancel"),
        ):
            self.assertFalse(agreement_cli.prompt_for_acceptance_if_needed())
            self.assertIsNone(current_acceptance())


if __name__ == "__main__":
    unittest.main()
