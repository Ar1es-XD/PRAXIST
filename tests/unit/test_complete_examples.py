"""Integrity contracts for bundled complete examples."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples"
EXAMPLE_ROOT = EXAMPLES_ROOT / "rocket_booster_recovery"
RUST_EXAMPLE_ROOT = EXAMPLES_ROOT / "rocket_booster_recovery_rust"
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LEGACY_IDENTITY_RE = re.compile(r"s" r"f[-_ ]?cac|s" r"fcac", re.IGNORECASE)
TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".jinja2",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CompleteExampleIntegrityTest(unittest.TestCase):
    def test_examples_and_templates_have_distinct_documented_roles(self) -> None:
        examples_readme = (EXAMPLES_ROOT / "README.md").read_text(encoding="utf-8")
        template_readme = (REPO_ROOT / "templates" / "README.md").read_text(encoding="utf-8")

        self.assertIn("complete, runnable reference research projects", examples_readme)
        self.assertIn("not authoring scaffolds", examples_readme)
        self.assertIn("replaceable scaffolds", template_readme)

    def test_example_contains_no_nested_git_information_old_name_or_chinese(self) -> None:
        git_artifacts = [
            path.relative_to(EXAMPLE_ROOT).as_posix()
            for path in EXAMPLE_ROOT.rglob("*")
            if path.name.lower().startswith(".git")
        ]
        old_name_hits: list[str] = []
        cjk_hits: list[str] = []
        for path in EXAMPLE_ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8")
            relative = path.relative_to(EXAMPLE_ROOT).as_posix()
            if LEGACY_IDENTITY_RE.search(text) or LEGACY_IDENTITY_RE.search(relative):
                old_name_hits.append(relative)
            if CJK_RE.search(text):
                cjk_hits.append(relative)

        self.assertEqual(git_artifacts, [])
        self.assertEqual(old_name_hits, [])
        self.assertEqual(cjk_hits, [])

    def test_example_checksum_manifest_covers_every_bundled_file(self) -> None:
        checksum_path = EXAMPLE_ROOT / "SHA256SUMS"
        expected: dict[str, str] = {}
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            expected[relative] = digest

        actual_paths = {
            path.relative_to(EXAMPLE_ROOT).as_posix()
            for path in EXAMPLE_ROOT.rglob("*")
            if path.is_file() and path != checksum_path
        }
        self.assertEqual(set(expected), actual_paths)
        for relative, digest in expected.items():
            self.assertEqual(_sha256(EXAMPLE_ROOT / relative), digest, relative)

    def test_task_asset_manifests_match_every_task_and_external_frozen_asset(self) -> None:
        for task_root in (EXAMPLE_ROOT / "task_GPU_server", EXAMPLE_ROOT / "task_PC"):
            manifest_path = task_root / "assets" / "task_assets_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            task_files = {
                path.relative_to(task_root).as_posix()
                for path in task_root.rglob("*")
                if path.is_file() and path != manifest_path
            }
            self.assertEqual(set(manifest["task_asset_sha256"]), task_files)
            for relative, digest in manifest["task_asset_sha256"].items():
                self.assertEqual(_sha256(task_root / relative), digest, relative)

            project_root = (task_root / manifest["research_project"]).resolve()
            self.assertEqual(project_root, EXAMPLE_ROOT.resolve())
            for relative, digest in manifest["external_frozen_sha256"].items():
                self.assertEqual(_sha256(project_root / relative), digest, relative)

    def test_baseline_controller_and_configuration_copies_are_identical(self) -> None:
        controller = EXAMPLE_ROOT / "src" / "rocket_booster_recovery_controller.py"
        config = EXAMPLE_ROOT / "configs" / "rocket_booster_recovery_v0.json"
        for task_name in ("task_GPU_server", "task_PC"):
            baseline = EXAMPLE_ROOT / task_name / "assets" / "baseline"
            self.assertEqual((baseline / "controller.py").read_bytes(), controller.read_bytes())
            self.assertEqual(
                (baseline / "controller_config.json").read_bytes(),
                config.read_bytes(),
            )

    def test_task_harnesses_keep_current_evidence_and_generation_contracts(self) -> None:
        expected_windows = {
            "task_GPU_server": (2.0, 90),
            "task_PC": (2.5, 120),
        }
        for task_name, (generation_hours, research_minutes) in expected_windows.items():
            task_root = EXAMPLE_ROOT / task_name
            task = yaml.safe_load((task_root / "task.yaml").read_text(encoding="utf-8"))

            self.assertEqual(task["task_version"], "2.0.5")
            self.assertEqual(task["generation_policy"]["per_generation_hours"], generation_hours)
            trigger = task["synthesis_trigger"]
            self.assertEqual(trigger["min_interval_minutes"], research_minutes)
            self.assertEqual(trigger["max_interval_minutes"], research_minutes)
            self.assertFalse(trigger["adaptive"]["enabled"])

            confirmed = next(
                lane for lane in task["evaluation"]["frontier_lanes"] if lane["name"] == "confirmed"
            )
            self.assertIn(
                "confirmed_performance_gate_passed",
                confirmed["require_truthy_metrics"],
            )
            self.assertNotIn(
                "tool_server:prior_work_tools",
                task["praxist_plugins"]["tools"],
            )
            protocol_intent = (task_root / "assets" / "protocol_intent.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("confirmed_performance_gate_passed", protocol_intent)

            for role_path in (task_root / "roles").glob("*/role.yaml"):
                role = yaml.safe_load(role_path.read_text(encoding="utf-8"))["role"]
                if role["role_kind"] == "peer":
                    self.assertNotIn("prior_work_tools.non_local", role["tool_scope"])

    def test_no_runtime_outputs_or_symlinks_are_bundled(self) -> None:
        forbidden_directories = {"experiments", "runs", "workspaces", "__pycache__"}
        offenders = [
            path.relative_to(EXAMPLE_ROOT).as_posix()
            for path in EXAMPLE_ROOT.rglob("*")
            if (path.is_dir() and path.name in forbidden_directories) or path.is_symlink()
        ]
        self.assertEqual(offenders, [])


class RustCompleteExampleIntegrityTest(unittest.TestCase):
    def test_rust_example_contains_expected_profiles_and_identity(self) -> None:
        manifest = tomllib.loads((RUST_EXAMPLE_ROOT / "Cargo.toml").read_text(encoding="utf-8"))

        self.assertEqual(manifest["package"]["name"], "rocket-booster-recovery-rust")
        self.assertTrue((RUST_EXAMPLE_ROOT / ".cargo" / "config.toml").is_file())
        self.assertTrue((RUST_EXAMPLE_ROOT / "vendor").is_dir())
        for task_name in ("task_GPU_server", "task_linux", "task_macos"):
            task_root = RUST_EXAMPLE_ROOT / task_name
            task = yaml.safe_load((task_root / "task.yaml").read_text(encoding="utf-8"))
            self.assertTrue(task["task_id"].startswith("rocket_booster_recovery_rust_"))
            self.assertIn("Rocket Booster Recovery", task["task_name"])
            self.assertTrue((task_root / "Cargo.toml").is_file())

    def test_rust_example_has_no_git_metadata_old_identity_or_authored_chinese(self) -> None:
        git_artifacts = [
            path.relative_to(RUST_EXAMPLE_ROOT).as_posix()
            for path in RUST_EXAMPLE_ROOT.rglob("*")
            if path.name.lower().startswith(".git")
        ]
        old_name_hits: list[str] = []
        cjk_hits: list[str] = []
        for path in RUST_EXAMPLE_ROOT.rglob("*"):
            if not path.is_file():
                continue
            relative_path = path.relative_to(RUST_EXAMPLE_ROOT)
            relative = relative_path.as_posix()
            if LEGACY_IDENTITY_RE.search(relative):
                old_name_hits.append(relative)
            if "vendor" in relative_path.parts or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8")
            if LEGACY_IDENTITY_RE.search(text):
                old_name_hits.append(relative)
            if CJK_RE.search(text):
                cjk_hits.append(relative)

        self.assertEqual(git_artifacts, [])
        self.assertEqual(old_name_hits, [])
        self.assertEqual(cjk_hits, [])

    def test_rust_example_bundles_no_runtime_state_or_symlinks(self) -> None:
        forbidden_directories = {
            "experiments",
            "results",
            "runs",
            "scratch",
            "target",
            "workspaces",
        }
        offenders = [
            path.relative_to(RUST_EXAMPLE_ROOT).as_posix()
            for path in RUST_EXAMPLE_ROOT.rglob("*")
            if (path.is_dir() and path.name in forbidden_directories) or path.is_symlink()
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
