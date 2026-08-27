from __future__ import annotations

import os
from pathlib import Path

from tests.helpers.discovery import load_package_tests

_FIXTURE_PLUGIN_ROOT = Path(__file__).resolve().parent / "fixtures" / "plugins"
if _FIXTURE_PLUGIN_ROOT.exists():
    existing = os.environ.get("PRAXIST_BUNDLED_PLUGIN_ROOTS", "")
    roots = [item for item in existing.split(os.pathsep) if item]
    fixture_root = str(_FIXTURE_PLUGIN_ROOT)
    if fixture_root not in roots:
        os.environ["PRAXIST_BUNDLED_PLUGIN_ROOTS"] = os.pathsep.join([fixture_root, *roots])


def load_tests(loader, tests, pattern):
    return load_package_tests(__file__, loader, pattern)
