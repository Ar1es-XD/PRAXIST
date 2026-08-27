"""Failure-isolation contracts for the built-in product-usage component."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from praxist.cli import product_usage as product_usage_cli
from praxist.plugins.workflow_stages.research_loop.lifecycle import (
    PeerLifecycleSummary,
    close_observer_safely,
    record_run_finished_safely,
    record_run_started_safely,
)
from praxist.run import _prompt_for_product_usage_consent


class ProductUsageFailureIsolationContract(unittest.TestCase):
    def test_safe_boundaries_preserve_operator_control_flow_errors(self) -> None:
        class BrokenObserver:
            def record_run_started(self, _summary: PeerLifecycleSummary) -> None:
                raise SystemExit(91)

            def record_run_finished(self, **_kwargs: object) -> None:
                raise KeyboardInterrupt

            def close(self) -> None:
                raise RuntimeError("close failed")

        observer = BrokenObserver()
        with self.assertRaises(SystemExit):
            record_run_started_safely(
                observer,  # type: ignore[arg-type]
                PeerLifecycleSummary.planned(generation_ordinal=0, planned_peer_count=1),
            )
        with self.assertRaises(KeyboardInterrupt):
            record_run_finished_safely(
                observer,  # type: ignore[arg-type]
                active_duration_seconds=None,
                failed=True,
            )
        close_observer_safely(observer)  # type: ignore[arg-type]

    def test_unavailable_built_in_client_returns_controlled_cli_error(self) -> None:
        with (
            patch("sys.stderr") as stderr,
            patch(
                "praxist.cli.product_usage.importlib.import_module",
                side_effect=ImportError("component unavailable"),
            ),
        ):
            exit_code = product_usage_cli.main(["status"])

        self.assertEqual(exit_code, 1)
        self.assertIn("built-in product-usage component", str(stderr.write.call_args_list))

    def test_first_run_prompt_preserves_keyboard_interrupt(self) -> None:
        with (
            patch.object(
                product_usage_cli,
                "prompt_for_consent_if_unset",
                side_effect=KeyboardInterrupt,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            _prompt_for_product_usage_consent()


if __name__ == "__main__":
    unittest.main()
