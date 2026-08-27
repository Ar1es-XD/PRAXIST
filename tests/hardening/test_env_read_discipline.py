"""Discipline check: no new env reads under ``praxist/{core,plugins,infrastructure}``.

The configuration-discipline rule (``docs/concepts/config_discipline.md``,
issue #75) forbids env-var reads in core domain code. Reads happen at
boundaries: CLI entrypoints on the way in, subprocess env construction
on the way out.

This test enforces the rule via a frozen allowlist. ``tests/hardening/
env_read_allowlist.txt`` lists every ``(path, var_name)`` pair the
codebase currently performs; any new read that doesn't appear there
fails the test until the author either:

* migrates the read away (preferred — usually by threading the value
  through :class:`praxist.core.run_config.RunConfig`), or
* adds the new pair to the allowlist with a one-line reason.

Stale allowlist entries (listed but no longer observed) also fail the
test — they signal a successful migration whose allowlist entry should
be removed for the diff to match. This keeps the allowlist honest:
shrinking it is how #75 reports progress.

Detection lives in :mod:`tests.hardening._env_read_audit`; this module
only orchestrates the comparison and renders an actionable error
message.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from tests.hardening._env_read_audit import collapse_to_pairs, collect_env_reads

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALLOWLIST_PATH = _REPO_ROOT / "tests" / "hardening" / "env_read_allowlist.txt"

# Roots the discipline rule covers. CLI entrypoints (``praxist/cli/``)
# are explicitly out of scope — they ARE the boundary that constructs
# RunConfig from env in the first place.
_AUDITED_ROOTS = (
    _REPO_ROOT / "praxist" / "core",
    _REPO_ROOT / "praxist" / "plugins",
    _REPO_ROOT / "praxist" / "infrastructure",
)


def _load_allowlist(path: Path) -> set[tuple[str, str]]:
    """Parse the allowlist file into a set of ``(path, var)`` pairs."""
    pairs: set[tuple[str, str]] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        body = raw.split("#", 1)[0].strip()
        if not body:
            continue
        if "::" not in body:
            raise ValueError(
                f"env_read_allowlist.txt entry must use ``path::var`` syntax; got {raw!r}"
            )
        file_path, _, var_name = body.partition("::")
        pairs.add((file_path.strip(), var_name.strip()))
    return pairs


class EnvReadDisciplineTest(unittest.TestCase):
    """Audit observed env reads against the frozen allowlist."""

    def test_observed_env_reads_match_allowlist(self) -> None:
        observed = collapse_to_pairs(collect_env_reads(_AUDITED_ROOTS, repo_root=_REPO_ROOT))
        allowed = _load_allowlist(_ALLOWLIST_PATH)

        new_violations = observed - allowed
        stale_entries = allowed - observed

        if not new_violations and not stale_entries:
            return

        lines: list[str] = ["env_read_allowlist.txt is out of sync with the codebase."]
        if new_violations:
            lines.append("")
            lines.append("NEW env reads detected (migrate to RunConfig, or allowlist with reason):")
            for path, var in sorted(new_violations):
                lines.append(f"  + {path}::{var}")
        if stale_entries:
            lines.append("")
            lines.append("STALE allowlist entries (the read was removed — drop the line):")
            for path, var in sorted(stale_entries):
                lines.append(f"  - {path}::{var}")
        lines.append("")
        lines.append(f"Edit {_ALLOWLIST_PATH.relative_to(_REPO_ROOT)} and re-run the test.")
        self.fail("\n".join(lines))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
