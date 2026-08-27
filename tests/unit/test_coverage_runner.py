"""Lock the coverage command contract for unit and integration profiles."""

from __future__ import annotations

import configparser
import importlib.util
import json
import subprocess
import sys
import tempfile
import tomllib
import types
import unittest
from pathlib import Path
from typing import TextIO
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
COVERAGE_RC = REPO_ROOT / ".coveragerc"
COVERAGE_SCRIPT = REPO_ROOT / "scripts" / "run_test_coverage.py"
PYPROJECT = REPO_ROOT / "pyproject.toml"


class CoverageConfiguration(unittest.TestCase):
    def test_coveragerc_tracks_praxist_package(self) -> None:
        parser = configparser.ConfigParser()
        parser.read(COVERAGE_RC)
        self.assertTrue(parser.getboolean("run", "branch"))
        self.assertIn("praxist", parser.get("run", "source"))

    def test_dev_dependencies_include_coverage_py(self) -> None:
        pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        dev_deps = pyproject.get("dependency-groups", {}).get("dev", [])
        self.assertIn("coverage==7.15.1", dev_deps)


class CoverageRunnerScript(unittest.TestCase):
    def test_script_exists(self) -> None:
        self.assertTrue(COVERAGE_SCRIPT.exists())

    def test_help_does_not_require_coverage_import(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(COVERAGE_SCRIPT), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("unit", completed.stdout)
        self.assertIn("integration", completed.stdout)
        self.assertIn("--fail-under-statements", completed.stdout)

    def test_unit_profile_uses_explicit_omit_list_but_integration_remains_unfiltered(self) -> None:
        spec = importlib.util.spec_from_file_location("run_test_coverage", COVERAGE_SCRIPT)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        unit_omit = tuple(module.PROFILE_OMIT["unit"])
        integration_omit = tuple(module.PROFILE_OMIT["integration"])

        self.assertGreater(len(unit_omit), 10)
        self.assertEqual(integration_omit, ())
        self.assertEqual(len(unit_omit), len(set(unit_omit)))
        self.assertTrue(all("*" not in item and "?" not in item for item in unit_omit))
        self.assertTrue(
            any(item.endswith("backend/gems.py") for item in unit_omit),
            "Gems is integration/orchestrator-heavy and should not dilute unit coverage.",
        )
        self.assertTrue(
            any(item.endswith("claude_sdk/delete_guard.py") for item in unit_omit),
            "Delete guard is subprocess/runtime-heavy and should stay in integration coverage.",
        )

    def test_unit_profile_includes_product_usage_pytest_suite(self) -> None:
        spec = importlib.util.spec_from_file_location("run_test_coverage", COVERAGE_SCRIPT)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertEqual(module.PROFILE_PYTEST_PATHS["unit"], ("tests/product_usage",))
        self.assertEqual(module.PROFILE_PYTEST_PATHS["integration"], ())

    def test_pytest_backed_coverage_suite_uses_isolated_coverage_process(self) -> None:
        spec = importlib.util.spec_from_file_location("run_test_coverage", COVERAGE_SCRIPT)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        completed = types.SimpleNamespace(returncode=0)
        data_file = REPO_ROOT / ".coverage.unit.pytest-probe"

        with patch.object(module.subprocess, "run", return_value=completed) as run:
            self.assertTrue(
                module._run_pytest_paths(
                    ("tests/product_usage",),
                    data_file=data_file,
                    omit=(str(REPO_ROOT / "praxist/run.py"),),
                )
            )

        command = run.call_args.args[0]
        self.assertEqual(command[:4], [sys.executable, "-m", "coverage", "run"])
        self.assertIn(str(REPO_ROOT / "tests/product_usage"), command)
        self.assertIn(f"--omit={REPO_ROOT / 'praxist/run.py'}", command)
        self.assertEqual(run.call_args.kwargs["cwd"], REPO_ROOT)
        self.assertEqual(run.call_args.kwargs["env"]["COVERAGE_FILE"], str(data_file))

    def test_report_formats_use_independent_coverage_instances(self) -> None:
        spec = importlib.util.spec_from_file_location("run_test_coverage", COVERAGE_SCRIPT)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class SingleReportCoverage:
            instances: list[SingleReportCoverage] = []

            def __init__(self, **_kwargs: object) -> None:
                self.report_kind: str | None = None
                self.instances.append(self)

            def erase(self) -> None:
                pass

            def start(self) -> None:
                pass

            def stop(self) -> None:
                pass

            def save(self) -> None:
                pass

            def load(self) -> None:
                pass

            def _claim_report(self, kind: str) -> None:
                if self.report_kind is not None:
                    raise AssertionError("a Coverage instance was reused for multiple reports")
                self.report_kind = kind

            def report(self, *, file: TextIO, show_missing: bool) -> float:
                del show_missing
                self._claim_report("text")
                file.write("coverage table\n")
                return 96.0

            def json_report(self, *, outfile: str) -> None:
                self._claim_report("json")
                Path(outfile).write_text(
                    json.dumps(
                        {
                            "totals": {
                                "percent_covered": 96.0,
                                "percent_statements_covered": 97.0,
                                "percent_branches_covered": 95.0,
                            }
                        }
                    ),
                    encoding="utf-8",
                )

            def xml_report(self, *, outfile: str) -> None:
                self._claim_report("xml")
                Path(outfile).write_text("<coverage />", encoding="utf-8")

        coverage_module = types.SimpleNamespace(Coverage=SingleReportCoverage)
        with (
            tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp,
            patch.object(module, "_load_suite", return_value=unittest.TestSuite()),
        ):
            passed = module._run_profile(
                coverage_module=coverage_module,
                profile="unit",
                modules=("tests.unit",),
                output_dir=Path(tmp),
                fail_under=90.0,
                fail_under_statements=95.0,
                verbosity=0,
                show_table=False,
                show_missing=False,
            )

        self.assertTrue(passed)
        self.assertEqual(
            [instance.report_kind for instance in SingleReportCoverage.instances],
            [None, "text", "json", "xml"],
        )


if __name__ == "__main__":
    unittest.main()
