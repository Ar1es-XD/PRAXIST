from __future__ import annotations

import unittest
from pathlib import Path

from tests.helpers.paths import REPO_ROOT


def load_package_tests(
    package_file: str,
    loader: unittest.TestLoader,
    pattern: str | None,
) -> unittest.TestSuite:
    package_dir = Path(package_file).resolve().parent
    return loader.discover(
        start_dir=str(package_dir),
        pattern=pattern or "test*.py",
        top_level_dir=str(REPO_ROOT),
    )
