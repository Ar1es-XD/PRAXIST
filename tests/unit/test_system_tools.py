"""Unit tests for :mod:`praxist.plugins.tools.system.adapter`.

Tests construct temporary run-dir fixtures and call the handler
functions directly (bypassing the MCP server wiring) so the suite is
independent of ``claude_agent_sdk``.

Coverage:

* manifest entrypoint shape (``create_tool_plugin``);
* active-run discovery with the freshness window;
* run summary aggregation including partial budget ledger;
* recent-findings tail from canonical ``findings/`` and legacy
  ``shared_findings/`` layouts;
* frontier snapshot read;
* recent-errors heuristic against a trajectory;
* graceful warnings when expected files are missing.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from praxist.plugins.tools.system.adapter import (
    create_tool_plugin,
    system_active_runs,
    system_frontier_snapshot,
    system_recent_errors,
    system_recent_findings,
    system_run_summary,
)


class CreateToolPluginTest(unittest.TestCase):
    """Manifest entrypoint exposes the documented surface."""

    def test_create_tool_plugin_shape(self) -> None:
        """``create_tool_plugin`` returns the expected descriptor dict."""
        descriptor = create_tool_plugin()
        self.assertEqual(descriptor["tool_server_ref"], "tool_server:system")
        self.assertEqual(descriptor["server_name"], "system-tools")
        self.assertEqual(descriptor["required_capability"], "tool_server.system")
        self.assertEqual(set(descriptor["visibility"]), {"peer", "panel"})

        expected = {
            "system_active_runs",
            "system_run_summary",
            "system_recent_findings",
            "system_frontier_snapshot",
            "system_recent_errors",
        }
        self.assertEqual(set(descriptor["tool_names"]), expected)
        self.assertEqual(set(descriptor["handlers"].keys()), expected)
        for handler_ref in descriptor["handlers"].values():
            self.assertIn(":", handler_ref)
            self.assertTrue(
                handler_ref.startswith("praxist.plugins.tools.system.adapter:"),
                f"handler ref escapes plugin boundary: {handler_ref}",
            )


class SystemActiveRunsTest(unittest.TestCase):
    """Active-run discovery and the freshness window."""

    def test_missing_workspace_returns_error_with_empty_runs(self) -> None:
        """A workspace path that does not exist yields a friendly error."""
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope"
            out = system_active_runs(missing)
        self.assertEqual(out["runs"], [])
        self.assertIn("error", out)

    def test_discovers_recent_run_and_skips_stale(self) -> None:
        """Fresh status files appear; stale (>24h) ones are filtered out."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            fresh = workspace / "run_fresh"
            stale = workspace / "run_stale"
            for run in (fresh, stale):
                run.mkdir()
                (run / "run_summary.json").write_text("{}", encoding="utf-8")

            stale_age = time.time() - (30 * 60 * 60)  # 30h ago
            os.utime(stale / "run_summary.json", (stale_age, stale_age))

            out = system_active_runs(workspace)
        run_dirs = {Path(entry["run_dir"]).name for entry in out["runs"]}
        self.assertIn("run_fresh", run_dirs)
        self.assertNotIn("run_stale", run_dirs)
        self.assertGreaterEqual(out["runs"][0]["last_active_ms"], 0)

    def test_skips_directories_without_status_files(self) -> None:
        """Random sibling directories without status files are ignored."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            (workspace / "logs").mkdir()
            run = workspace / "run_a"
            run.mkdir()
            (run / "run_summary.json").write_text("{}", encoding="utf-8")

            out = system_active_runs(workspace)
        self.assertEqual([Path(e["run_dir"]).name for e in out["runs"]], ["run_a"])


class SystemRunSummaryTest(unittest.TestCase):
    """Run summary aggregation."""

    def test_summary_and_budget_totals_assembled(self) -> None:
        """Summary JSON and budget ledger sums show up; warnings stay empty."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run_x"
            run.mkdir()
            (run / "run_summary.json").write_text(
                json.dumps({"task": "demo", "generations": 3}), encoding="utf-8"
            )
            ledger = run / "budget_ledger.jsonl"
            ledger.write_text(
                "\n".join(
                    [
                        json.dumps({"unit": "tokens", "granted": 1000, "used": 800}),
                        json.dumps({"unit": "tokens", "used": 200}),
                        json.dumps({"unit": "gpu_hours", "granted": 2.0, "used": 1.5}),
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            out = system_run_summary(run)
        self.assertEqual(out["summary"], {"task": "demo", "generations": 3})
        self.assertEqual(out["warnings"], [])
        self.assertEqual(
            out["budget_totals"]["tokens"],
            {"granted": 1000.0, "used": 1000.0, "denied": 0.0},
        )
        self.assertEqual(
            out["budget_totals"]["gpu_hours"],
            {"granted": 2.0, "used": 1.5, "denied": 0.0},
        )

    def test_missing_summary_records_warning(self) -> None:
        """Missing ``run_summary.json`` produces a warning, not a raise."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run_empty"
            run.mkdir()
            out = system_run_summary(run)
        self.assertIsNone(out["summary"])
        self.assertTrue(any("run_summary.json" in w for w in out["warnings"]))


class SystemRecentFindingsTest(unittest.TestCase):
    """Recent findings tail across both storage layouts."""

    def test_findings_jsonl_directory_returns_newest_first(self) -> None:
        """Canonical ``findings/*.jsonl`` layout yields newest record first."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            findings = run / "findings"
            findings.mkdir(parents=True)
            (findings / "findings.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "f1", "metric": 0.5}),
                        json.dumps({"id": "f2", "metric": 0.6}),
                        json.dumps({"id": "f3", "metric": 0.7}),
                    ]
                ),
                encoding="utf-8",
            )

            out = system_recent_findings(run, n=2)
        self.assertEqual(out["findings"][0]["id"], "f3")
        self.assertEqual(out["findings"][1]["id"], "f2")
        self.assertEqual(len(out["findings"]), 2)
        self.assertTrue(out["source"].endswith("findings"))

    def test_shared_findings_fallback_used_when_no_canonical(self) -> None:
        """Legacy ``shared_findings/*.json`` is used as a fallback."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            shared = run / "shared_findings"
            shared.mkdir(parents=True)
            for name in ("a.json", "b.json"):
                (shared / name).write_text(json.dumps({"name": name}), encoding="utf-8")
            # Stagger mtimes so 'b.json' is newest.
            os.utime(shared / "a.json", (time.time() - 60, time.time() - 60))

            out = system_recent_findings(run, n=5)
        self.assertEqual(len(out["findings"]), 2)
        self.assertEqual(out["findings"][0]["name"], "b.json")
        self.assertTrue(out["source"].endswith("shared_findings"))

    def test_no_findings_records_warning(self) -> None:
        """Run with no findings directory produces a warning."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run_empty"
            run.mkdir()
            out = system_recent_findings(run)
        self.assertEqual(out["findings"], [])
        self.assertTrue(out["warnings"])

    def test_limit_is_clamped(self) -> None:
        """Requested ``n`` is clamped to the upper bound."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            findings = run / "findings"
            findings.mkdir(parents=True)
            lines = [json.dumps({"id": f"f{i}"}) for i in range(120)]
            (findings / "findings.jsonl").write_text("\n".join(lines), encoding="utf-8")

            out = system_recent_findings(run, n=10_000)
        self.assertLessEqual(len(out["findings"]), 100)


class SystemFrontierSnapshotTest(unittest.TestCase):
    """Frontier manifest read."""

    def test_returns_manifest_contents(self) -> None:
        """The manifest JSON is returned verbatim."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            (run / "frontier").mkdir(parents=True)
            payload = {"top_k": 2, "entries": [{"id": "alpha", "metric": 0.9}]}
            (run / "frontier" / "frontier_manifest.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            out = system_frontier_snapshot(run)
        self.assertEqual(out["frontier"], payload)
        self.assertEqual(out["warnings"], [])

    def test_missing_manifest_records_warning(self) -> None:
        """Missing manifest yields a warning and ``frontier=None``."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run_empty"
            run.mkdir()
            out = system_frontier_snapshot(run)
        self.assertIsNone(out["frontier"])
        self.assertTrue(out["warnings"])


class SystemRecentErrorsTest(unittest.TestCase):
    """Recent-error heuristic from trajectory."""

    def test_matches_typed_and_payload_errors_newest_first(self) -> None:
        """Both ``type=*error*`` events and payload-flagged events are picked up."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            lines = [
                json.dumps({"type": "session.started", "payload": {}}),
                json.dumps({"type": "tool.error", "payload": {"message": "boom"}}),
                json.dumps({"type": "session.tick", "payload": {"level": "warning"}}),
                json.dumps({"type": "session.tick", "payload": {"level": "info"}}),
                json.dumps({"type": "peer.exception", "payload": {"traceback": "..."}}),
            ]
            (run / "trajectory.jsonl").write_text("\n".join(lines), encoding="utf-8")

            out = system_recent_errors(run, n=10)
        types = [event["type"] for event in out["errors"]]
        self.assertIn("tool.error", types)
        self.assertIn("session.tick", types)
        self.assertIn("peer.exception", types)
        # Newest first → peer.exception (last line) leads.
        self.assertEqual(types[0], "peer.exception")

    def test_no_trajectory_records_warning(self) -> None:
        """Run with no trajectory.jsonl yields a warning."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run_empty"
            run.mkdir()
            out = system_recent_errors(run)
        self.assertEqual(out["errors"], [])
        self.assertTrue(out["warnings"])

    def test_malformed_jsonl_lines_are_skipped(self) -> None:
        """Bad JSON lines are silently skipped, valid ones still parsed."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "trajectory.jsonl").write_text(
                "\n".join(
                    [
                        "{not json",
                        json.dumps({"type": "tool.error", "payload": {}}),
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            out = system_recent_errors(run)
        self.assertEqual(len(out["errors"]), 1)
        self.assertEqual(out["errors"][0]["type"], "tool.error")


class ErrorPathCoverageTest(unittest.TestCase):
    """Cover error/decode branches that the happy-path tests skip."""

    def test_run_summary_malformed_json_warns(self) -> None:
        """A malformed ``run_summary.json`` warns and yields ``summary=None``."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "run_summary.json").write_text("{ broken", encoding="utf-8")
            out = system_run_summary(run)
        self.assertIsNone(out["summary"])
        self.assertTrue(any("run_summary.json" in w for w in out["warnings"]))

    def test_frontier_snapshot_malformed_json_warns(self) -> None:
        """A malformed frontier manifest warns and yields ``frontier=None``."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            (run / "frontier").mkdir(parents=True)
            (run / "frontier" / "frontier_manifest.json").write_text("{nope", encoding="utf-8")
            out = system_frontier_snapshot(run)
        self.assertIsNone(out["frontier"])
        self.assertTrue(any("frontier_manifest.json" in w for w in out["warnings"]))

    def test_budget_totals_malformed_ledger_lines_skipped(self) -> None:
        """Malformed lines in the budget ledger are skipped, valid lines summed."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "run_summary.json").write_text("{}", encoding="utf-8")
            (run / "budget_ledger.jsonl").write_text(
                "\n".join(
                    [
                        "{not json",
                        json.dumps({"unit": "tokens", "granted": 500}),
                        json.dumps(["not a dict"]),
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            out = system_run_summary(run)
        self.assertEqual(out["budget_totals"]["tokens"]["granted"], 500.0)

    def test_recent_findings_non_dict_records_filtered(self) -> None:
        """Non-dict JSONL records are skipped silently."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            findings = run / "findings"
            findings.mkdir(parents=True)
            (findings / "x.jsonl").write_text(
                "\n".join(
                    [
                        "[1, 2, 3]",
                        json.dumps({"id": "good"}),
                        '"just a string"',
                    ]
                ),
                encoding="utf-8",
            )
            out = system_recent_findings(run)
        self.assertEqual(len(out["findings"]), 1)
        self.assertEqual(out["findings"][0]["id"], "good")

    def test_shared_findings_malformed_file_warns(self) -> None:
        """A malformed shared_findings JSON file produces a warning."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            shared = run / "shared_findings"
            shared.mkdir(parents=True)
            (shared / "bad.json").write_text("{not json", encoding="utf-8")
            (shared / "good.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            out = system_recent_findings(run)
        self.assertTrue(any("bad.json" in w for w in out["warnings"]))
        self.assertTrue(any(rec.get("ok") for rec in out["findings"]))

    def test_active_runs_honors_praxist_workspace_env(self) -> None:
        """The ``PRAXIST_WORKSPACE`` env var overrides the default fallback."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp).resolve()
            (ws / "run_a").mkdir()
            (ws / "run_a" / "run_summary.json").write_text("{}", encoding="utf-8")
            prior = os.environ.get("PRAXIST_WORKSPACE")
            try:
                os.environ["PRAXIST_WORKSPACE"] = str(ws)
                out = system_active_runs()
            finally:
                if prior is None:
                    os.environ.pop("PRAXIST_WORKSPACE", None)
                else:
                    os.environ["PRAXIST_WORKSPACE"] = prior
        self.assertEqual([Path(e["run_dir"]).name for e in out["runs"]], ["run_a"])

    def test_recent_findings_empty_findings_dir_falls_back_to_shared(self) -> None:
        """An empty ``findings/`` dir does not block ``shared_findings/`` fallback."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            (run / "findings").mkdir(parents=True)  # exists but no .jsonl
            shared = run / "shared_findings"
            shared.mkdir(parents=True)
            (shared / "a.json").write_text(json.dumps({"x": 1}), encoding="utf-8")
            out = system_recent_findings(run)
        self.assertEqual(out["findings"], [{"x": 1}])
        self.assertTrue(out["source"].endswith("shared_findings"))

    def test_recent_errors_skips_records_without_payload(self) -> None:
        """Records without a payload field do not crash and do not match."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "trajectory.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"type": "session.idle"}),  # no payload
                        json.dumps({"type": "session.idle", "payload": "not a dict"}),
                        json.dumps({"type": "session.idle", "payload": {"error": "boom"}}),
                    ]
                ),
                encoding="utf-8",
            )
            out = system_recent_errors(run)
        self.assertEqual(len(out["errors"]), 1)
        self.assertEqual(out["errors"][0]["payload"]["error"], "boom")

    def test_active_runs_skips_regular_files_in_workspace(self) -> None:
        """Plain files sitting next to run dirs are skipped, not crashed on."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp).resolve()
            (ws / "stray.log").write_text("noise", encoding="utf-8")
            (ws / "run_a").mkdir()
            (ws / "run_a" / "run_summary.json").write_text("{}", encoding="utf-8")
            out = system_active_runs(ws)
        self.assertEqual([Path(e["run_dir"]).name for e in out["runs"]], ["run_a"])

    def test_recent_errors_skips_non_dict_trajectory_records(self) -> None:
        """Top-level array entries in trajectory.jsonl are silently skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "trajectory.jsonl").write_text(
                "\n".join(
                    [
                        "[1, 2, 3]",
                        json.dumps({"type": "tool.error", "payload": {}}),
                    ]
                ),
                encoding="utf-8",
            )
            out = system_recent_errors(run)
        self.assertEqual(len(out["errors"]), 1)
        self.assertEqual(out["errors"][0]["type"], "tool.error")

    def test_recent_errors_stops_at_limit(self) -> None:
        """The ``n`` parameter caps the returned list size."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            lines = [json.dumps({"type": "tool.error", "i": i}) for i in range(20)]
            (run / "trajectory.jsonl").write_text("\n".join(lines), encoding="utf-8")
            out = system_recent_errors(run, n=3)
        self.assertEqual(len(out["errors"]), 3)

    def test_shared_findings_skips_non_dict_payloads(self) -> None:
        """A shared_findings JSON file containing an array is silently skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            shared = run / "shared_findings"
            shared.mkdir(parents=True)
            (shared / "array.json").write_text("[1, 2, 3]", encoding="utf-8")
            (shared / "dict.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            out = system_recent_findings(run)
        ids = [rec for rec in out["findings"] if isinstance(rec, dict)]
        self.assertEqual(len(ids), 1)
        self.assertTrue(ids[0]["ok"])

    def test_active_runs_default_workspace_uses_cwd_runs(self) -> None:
        """Default workspace falls back to ``<cwd>/runs`` when no arg given."""
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp).resolve()
            runs = cwd / "runs"
            runs.mkdir()
            run = runs / "run_a"
            run.mkdir()
            (run / "run_summary.json").write_text("{}", encoding="utf-8")
            prior = Path.cwd()
            try:
                os.chdir(cwd)
                out = system_active_runs()
            finally:
                os.chdir(prior)
        self.assertEqual([Path(e["run_dir"]).name for e in out["runs"]], ["run_a"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
