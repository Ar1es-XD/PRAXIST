"""Unit tests for the AST-based env-read audit helper.

These tests verify the detection logic itself — that the visitor
finds every shape of env read we care about, and only those shapes.
Without this, a refactor of the helper could silently weaken the
discipline rule.
"""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from tests.hardening._env_read_audit import (
    EnvRead,
    collapse_to_pairs,
    collect_env_reads,
)


class _ModuleHarness(unittest.TestCase):
    """Write a Python module to a tmpdir and run the audit against it."""

    def _audit(self, source: str) -> list[EnvRead]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        module = root / "mod.py"
        module.write_text(textwrap.dedent(source), encoding="utf-8")
        return collect_env_reads([root], repo_root=root)


class DetectsCanonicalEnvReadShapesTests(_ModuleHarness):
    """The three documented shapes are all caught."""

    def test_os_environ_get_with_literal_var(self) -> None:
        reads = self._audit(
            """
            import os
            x = os.environ.get("MY_VAR")
            """
        )
        self.assertEqual([(r.path, r.var_name) for r in reads], [("mod.py", "MY_VAR")])

    def test_os_environ_get_with_default(self) -> None:
        reads = self._audit(
            """
            import os
            x = os.environ.get("MY_VAR", "default")
            """
        )
        self.assertEqual([(r.path, r.var_name) for r in reads], [("mod.py", "MY_VAR")])

    def test_os_getenv_call(self) -> None:
        reads = self._audit(
            """
            import os
            x = os.getenv("OTHER_VAR")
            """
        )
        self.assertEqual([(r.path, r.var_name) for r in reads], [("mod.py", "OTHER_VAR")])

    def test_os_environ_subscript_read(self) -> None:
        reads = self._audit(
            """
            import os
            x = os.environ["SUBSCRIPT_VAR"]
            """
        )
        self.assertEqual([(r.path, r.var_name) for r in reads], [("mod.py", "SUBSCRIPT_VAR")])


class DistinguishesReadFromWriteTests(_ModuleHarness):
    """Writes through ``os.environ[X] = ...`` are subprocess-builder code, not reads."""

    def test_subscript_write_is_not_counted_as_read(self) -> None:
        reads = self._audit(
            """
            import os
            os.environ["WRITE_VAR"] = "value"
            """
        )
        self.assertEqual(reads, [])

    def test_mixed_read_and_write_only_counts_the_read(self) -> None:
        reads = self._audit(
            """
            import os
            os.environ["WRITTEN"] = "value"
            x = os.environ["READ"]
            """
        )
        self.assertEqual([r.var_name for r in reads], ["READ"])


class DynamicLookupSentinelTests(_ModuleHarness):
    """Non-literal var names collapse to ``<dynamic>`` so the allowlist key is stable."""

    def test_get_with_variable_name(self) -> None:
        reads = self._audit(
            """
            import os
            name = "WHATEVER"
            x = os.environ.get(name)
            """
        )
        self.assertEqual([(r.path, r.var_name) for r in reads], [("mod.py", "<dynamic>")])

    def test_subscript_with_variable_name(self) -> None:
        reads = self._audit(
            """
            import os
            name = "DYN"
            x = os.environ[name]
            """
        )
        self.assertEqual([(r.path, r.var_name) for r in reads], [("mod.py", "<dynamic>")])


class IgnoresNonEnvAccessTests(_ModuleHarness):
    """The visitor must not flag access that looks similar but isn't ``os.environ``."""

    def test_other_get_call_is_ignored(self) -> None:
        reads = self._audit(
            """
            cache = {}
            x = cache.get("MY_VAR")
            """
        )
        self.assertEqual(reads, [])

    def test_other_environ_attribute_is_ignored(self) -> None:
        # A custom ``cfg.environ`` attribute should not be confused for ``os.environ``.
        reads = self._audit(
            """
            class C:
                environ = {"X": "1"}
            cfg = C()
            x = cfg.environ.get("MY_VAR")
            """
        )
        self.assertEqual(reads, [])

    def test_other_module_getenv_is_ignored(self) -> None:
        # A function named ``getenv`` imported from a non-``os`` module is fine.
        reads = self._audit(
            """
            class fake:
                @staticmethod
                def getenv(name):
                    return name
            x = fake.getenv("MY_VAR")
            """
        )
        self.assertEqual(reads, [])


class CollapseToPairsTests(unittest.TestCase):
    """Multiple reads of the same var in one file collapse to one allowlist entry."""

    def test_duplicate_reads_collapse(self) -> None:
        reads = [
            EnvRead(path="m.py", var_name="X", line=1),
            EnvRead(path="m.py", var_name="X", line=42),
            EnvRead(path="m.py", var_name="Y", line=7),
        ]
        self.assertEqual(collapse_to_pairs(reads), {("m.py", "X"), ("m.py", "Y")})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
