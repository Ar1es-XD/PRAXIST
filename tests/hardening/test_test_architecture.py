from __future__ import annotations

import unittest

from tests.helpers.paths import REPO_ROOT


class TestArchitectureRefactorTest(unittest.TestCase):
    def test_tests_are_owned_by_long_term_architecture_directories(self) -> None:
        tests_root = REPO_ROOT / "tests"
        required_dirs = {
            "core",
            "conformance",
            "integration",
            "workflows",
            "hardening",
            "legacy_migration",
            "helpers",
            "unit",
        }
        for dirname in required_dirs:
            self.assertTrue((tests_root / dirname).is_dir(), dirname)

        root_test_files = sorted(path.name for path in tests_root.glob("test_*.py"))
        self.assertEqual(root_test_files, [])

    def test_layer_packages_support_direct_unittest_entrypoints(self) -> None:
        tests_root = REPO_ROOT / "tests"
        for rel in (
            "core",
            "conformance",
            "integration",
            "workflows",
            "hardening",
            "legacy_migration",
            "helpers",
            "unit",
        ):
            init_path = tests_root / rel / "__init__.py"
            self.assertTrue(init_path.exists(), rel)
            text = init_path.read_text(encoding="utf-8")
            self.assertIn("load_tests", text, rel)

    def test_old_file_mapping_and_migration_test_names_are_removed(self) -> None:
        tests_root = REPO_ROOT / "tests"
        forbidden_doc_phrases = (
            "Old" + " file",
            "Migration " + "ownership",
            "pre-" + "Step-23",
        )
        old_layout_prefixes = ("test_" + "gate_", "test_" + "step")
        for path in tests_root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".md"}:
                self.assertFalse(path.name.startswith(old_layout_prefixes), str(path))
                text = path.read_text(encoding="utf-8")
                for phrase in forbidden_doc_phrases:
                    self.assertNotIn(phrase, text, str(path))


if __name__ == "__main__":
    unittest.main()
