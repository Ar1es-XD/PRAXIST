from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch


class BaselineCacheContractsTest(unittest.TestCase):
    def test_git_jsonl_and_curated_cache_edges_are_tolerant(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import baseline_cache

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patcher = patch.dict(
                baseline_cache.os.environ,
                {"PRAXIST_BASELINE_CACHE_DIR": str(root / "baseline_cache")},
                clear=False,
            )
            patcher.start()
            self.addCleanup(patcher.stop)
            entry = baseline_cache.CachedBaseline.from_dict(
                {
                    "name": "base",
                    "accuracy": 0.5,
                    "seeds": None,
                    "unknown": "ignored",
                }
            )
            self.assertEqual(entry.seeds, [])
            self.assertEqual(entry.metric_name, "metric_value")
            self.assertEqual(entry.metric_value, 0.5)
            generic_entry = baseline_cache.CachedBaseline.from_dict(
                {
                    "name": "base",
                    "metric_name": "absolute_return_pct",
                    "metric_value": 12.3,
                    "seeds": None,
                    "unknown": "ignored",
                }
            )
            self.assertEqual(generic_entry.accuracy, 12.3)
            self.assertEqual(generic_entry.metric_name, "absolute_return_pct")

            with patch.object(
                baseline_cache.subprocess,
                "check_output",
                return_value=b"abc123\n",
            ):
                self.assertEqual(baseline_cache.get_current_code_hash(root), "abc123")
                self.assertEqual(
                    baseline_cache.get_changed_files_since("old", root),
                    ["abc123"],
                )
            with patch.object(
                baseline_cache.subprocess,
                "check_output",
                side_effect=subprocess.TimeoutExpired("git", 5),
            ):
                self.assertEqual(baseline_cache.get_current_code_hash(root), "")
                self.assertEqual(baseline_cache.get_changed_files_since("old", root), [])
            self.assertEqual(baseline_cache.get_changed_files_since("", root), [])

            cache_path = baseline_cache._default_cache_path(root, "task")
            with patch.dict(
                baseline_cache.os.environ,
                {"PRAXIST_BASELINE_CACHE_DIR": str(root / "run" / "baseline_cache")},
                clear=False,
            ):
                self.assertEqual(
                    baseline_cache._default_cache_path(root, "task"),
                    (root / "run" / "baseline_cache" / "task" / "baselines.jsonl").resolve(),
                )
            cache_path.parent.mkdir(parents=True)
            cache_path.write_text(
                "\n{bad}\n"
                + json.dumps({"name": "ok", "accuracy": 0.6, "measured_at": "t"})
                + "\n[]\n",
                encoding="utf-8",
            )
            self.assertEqual([row.name for row in baseline_cache.load_cache("task", root)], ["ok"])
            with patch("builtins.open", side_effect=OSError("read failed")):
                self.assertEqual(baseline_cache.load_cache("task", root), [])

            newer = baseline_cache.CachedBaseline(
                "newer", 0.7, measured_at="2026-05-12T00:00:00+00:00"
            )
            older = baseline_cache.CachedBaseline(
                "older", 0.6, measured_at="2026-01-01T00:00:00+00:00"
            )
            with patch.object(baseline_cache.os, "fsync", side_effect=OSError("no fsync")):
                written = baseline_cache.save_cache("task", root, [older, newer])
            lines = written.read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(lines[0])["name"], "newer")

            with (
                patch.object(baseline_cache.os, "replace", side_effect=RuntimeError("rename")),
                self.assertRaises(RuntimeError),
            ):
                baseline_cache.save_cache("task", root, [newer])
            self.assertFalse(
                any(path.name.endswith(".tmp") for path in cache_path.parent.iterdir())
            )

            curated = root / "curated.jsonl"
            curated.write_text(
                "\n{bad}\n[]\n"
                + json.dumps({"note": "metadata"})
                + "\n"
                + json.dumps({"optimizer": "adamw", "accuracy": 0.8})
                + "\n"
                + json.dumps({"name": "sgd", "accuracy": 0.7})
                + "\n"
                + json.dumps({"baseline": "generic", "deterministic_score": 0.3})
                + "\n",
                encoding="utf-8",
            )
            curated_entries = baseline_cache.load_curated_baseline_entries(curated)
            self.assertEqual([row["name"] for row in curated_entries], ["adamw", "sgd", "generic"])
            self.assertEqual(curated_entries[0]["metric_value"], 0.8)
            self.assertEqual(curated_entries[2]["metric_name"], "deterministic_score")
            self.assertEqual(curated_entries[2]["metric_value"], 0.3)
            self.assertEqual(baseline_cache.load_curated_baseline_entries(None), [])
            with patch("builtins.open", side_effect=OSError("read failed")):
                self.assertEqual(baseline_cache.load_curated_baseline_entries(curated), [])

            recorded = baseline_cache.record_measurement(
                "task",
                root,
                "generic",
                1.0,
                metric_name="absolute_return_pct",
                metric_value=12.5,
                direction="maximize",
                epochs=3,
            )
            self.assertEqual(recorded.metric_name, "absolute_return_pct")
            self.assertEqual(recorded.metric_value, 12.5)
            self.assertEqual(recorded.accuracy, 1.0)

    def test_validation_marks_age_parse_and_code_relevance_without_dropping_rows(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import baseline_cache

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patcher = patch.dict(
                baseline_cache.os.environ,
                {"PRAXIST_BASELINE_CACHE_DIR": str(root / "baseline_cache")},
                clear=False,
            )
            patcher.start()
            self.addCleanup(patcher.stop)
            now = datetime.now(UTC)
            rows = [
                baseline_cache.CachedBaseline(
                    "fresh",
                    0.9,
                    measured_at=now.isoformat(),
                    code_hash="new",
                ),
                baseline_cache.CachedBaseline(
                    "old_age",
                    0.7,
                    measured_at=(now - timedelta(days=90)).replace(tzinfo=None).isoformat(),
                    code_hash="new",
                ),
                baseline_cache.CachedBaseline(
                    "bad_time",
                    0.6,
                    measured_at="not-a-date",
                    code_hash="new",
                ),
                baseline_cache.CachedBaseline(
                    "old_code",
                    0.5,
                    measured_at=now.isoformat(),
                    code_hash="old",
                ),
            ]
            baseline_cache.save_cache("task", root, rows)
            with patch.object(
                baseline_cache,
                "get_changed_files_since",
                return_value=["baseline/train.py"],
            ):
                report = baseline_cache.validate_cache(
                    "task",
                    root,
                    ["fresh", "old_age", "bad_time", "old_code", "curated", "missing"],
                    current_code_hash="new",
                    stale_after_days=30,
                    curated_entries=[{"name": "curated"}],
                )
            self.assertEqual(report.total, 4)
            self.assertEqual(report.fresh, 1)
            self.assertEqual(report.stale, 3)
            self.assertEqual(report.curated_baseline_names, ["curated"])
            self.assertEqual(
                report.missing_baselines, ["old_age", "bad_time", "old_code", "missing"]
            )
            self.assertEqual(
                set(report.missing_runtime_cache_baselines),
                {"old_age", "bad_time", "old_code", "curated", "missing"},
            )
            stale_by_name = {row["name"]: row["stale_reason"] for row in report.stale_entries}
            self.assertIn("older than", stale_by_name["old_age"])
            self.assertIn("older than", stale_by_name["bad_time"])
            self.assertIn("touches baseline-relevant files", stale_by_name["old_code"])


if __name__ == "__main__":
    unittest.main()
