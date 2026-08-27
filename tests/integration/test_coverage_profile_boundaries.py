from __future__ import annotations

import importlib.util
import unittest

from tests.helpers.paths import REPO_ROOT


class CoverageProfileBoundaryIntegrationTest(unittest.TestCase):
    def test_integration_coverage_profile_keeps_full_source_surface(self) -> None:
        """Integration coverage must stay unfiltered while unit coverage is scoped."""

        script = REPO_ROOT / "scripts" / "run_test_coverage.py"
        spec = importlib.util.spec_from_file_location("run_test_coverage", script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertEqual(tuple(module.PROFILE_OMIT["integration"]), ())
        self.assertGreater(len(tuple(module.PROFILE_OMIT["unit"])), 10)


if __name__ == "__main__":
    unittest.main()
