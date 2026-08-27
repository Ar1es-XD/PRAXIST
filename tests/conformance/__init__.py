from __future__ import annotations

from tests.helpers.discovery import load_package_tests


def load_tests(loader, tests, pattern):
    return load_package_tests(__file__, loader, pattern)
