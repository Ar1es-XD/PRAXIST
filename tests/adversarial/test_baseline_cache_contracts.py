import json
import tempfile
import unittest
from pathlib import Path

from praxist.plugins.workflow_stages.research_loop.backend import baseline_cache


class BaselineCacheContracts(unittest.TestCase):
    def test_curated_baselines_satisfy_availability_when_runtime_cache_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            curated_path = root / "assets" / "baselines" / "results.jsonl"
            curated_path.parent.mkdir(parents=True)
            curated_path.write_text(
                "\n".join(
                    [
                        json.dumps({"_protocol": "metadata-only row"}),
                        json.dumps(
                            {
                                "optimizer": "vanilla_sam",
                                "dataset": "cifar100",
                                "test_accuracy": {"mean": 0.7155, "std": 0.0009},
                                "n_seeds": 5,
                            }
                        ),
                    ]
                )
                + "\n"
            )

            curated_entries = baseline_cache.load_curated_baseline_entries(curated_path)
            report = baseline_cache.validate_cache(
                task_id="sam_optimizer",
                workspace=root,
                expected_baseline_names=["vanilla_sam"],
                current_code_hash="abc123",
                curated_entries=curated_entries,
            )

            self.assertEqual(report.total, 0)
            self.assertEqual(report.fresh, 0)
            self.assertEqual(report.missing_baselines, [])
            self.assertEqual(report.missing_runtime_cache_baselines, ["vanilla_sam"])
            self.assertEqual(report.curated_baseline_names, ["vanilla_sam"])
            self.assertEqual(len(report.curated_entries), 1)

    def test_missing_baselines_means_missing_from_runtime_cache_and_curated_assets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report = baseline_cache.validate_cache(
                task_id="toy_task",
                workspace=Path(td),
                expected_baseline_names=["baseline_a"],
                current_code_hash="abc123",
                curated_entries=[],
            )

            self.assertEqual(report.missing_baselines, ["baseline_a"])
            self.assertEqual(report.missing_runtime_cache_baselines, ["baseline_a"])
            self.assertEqual(report.curated_baseline_names, [])


if __name__ == "__main__":
    unittest.main()
