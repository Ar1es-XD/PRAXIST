#!/usr/bin/env python3
"""Guardrail runner: ruff, pyrefly, pytest/unit, and coverage profiles.

The commands here MUST stay aligned with ``.github/workflows/ci.yml`` so
a contributor cannot land a PR that passes locally but fails in CI
(or vice versa). The single source of truth is the test suite in
``tests/unit/test_ci_workflow.py`` and ``tests/unit/test_dev_scripts.py``.

Exit code: 0 on all-pass; non-zero with ``--block`` when any check
fails. With ``--block`` the script emits structured JSON for callers that
support blocking checks. Without it, the same failures are advisory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# scripts/dev/run_guardrails.py → scripts/dev → scripts → repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(REPO_ROOT)

block = "--block" in sys.argv

checks = [
    ("ruff", ["uv", "run", "ruff", "check", "praxist/", "tests/"]),
    ("fmt", ["uv", "run", "ruff", "format", "--check", "praxist/", "tests/"]),
    ("pyrefly", ["uv", "run", "pyrefly", "check"]),
    ("pytest", ["uv", "run", "pytest", "tests/unit", "--tb=short", "-q"]),
    (
        "coverage-unit",
        [
            "uv",
            "run",
            "python",
            "scripts/run_test_coverage.py",
            "unit",
            "--fail-under",
            "90",
            "--fail-under-statements",
            "95",
        ],
    ),
    (
        "coverage-integration",
        ["uv", "run", "python", "scripts/run_test_coverage.py", "integration"],
    ),
]

fails: list[str] = []
details: list[str] = []
for name, cmd in checks:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        fails.append(name)
        out = "\n".join((r.stdout + r.stderr).strip().split("\n")[-10:])
        details.append(f"[{name}]\n{out}")

if not fails:
    print(
        json.dumps(
            {
                "systemMessage": (
                    "✓ guardrails passed: ruff · fmt · pyrefly · pytest/unit · "
                    "coverage/unit · coverage/integration"
                )
            }
        )
    )
elif block:
    msg = (
        "Commit blocked — fix guardrails first: " + ", ".join(fails) + "\n\n" + "\n\n".join(details)
    )
    print(json.dumps({"continue": False, "stopReason": msg, "systemMessage": msg}))
else:
    print(json.dumps({"systemMessage": "⚠ guardrails: " + ", ".join(f + " FAIL" for f in fails)}))
