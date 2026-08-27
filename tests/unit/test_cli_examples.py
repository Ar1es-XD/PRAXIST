"""Contracts for installing complete examples outside Praxist source."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from praxist.cli import main
from praxist.cli.examples import (
    EXAMPLES_HOME_ENV,
    ExampleInstallError,
    _bundled_examples_root,
    default_examples_home,
    materialize_example,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "examples"


class ExampleInstallationTest(unittest.TestCase):
    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                main(argv)
            code = 0
        except SystemExit as exc:
            code = int(exc.code or 0)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_cli_lists_complete_examples_and_default_destinations(self) -> None:
        code, out, err = self._run(["examples", "list", "--json"])

        self.assertEqual(code, 0, msg=out + err)
        payload = json.loads(out)
        self.assertEqual(
            [item["name"] for item in payload],
            ["rocket_booster_recovery", "rocket_booster_recovery_rust"],
        )
        self.assertTrue(payload[0]["default_destination"].endswith("rocket_booster_recovery"))
        self.assertTrue(payload[1]["default_destination"].endswith("rocket_booster_recovery_rust"))

    def test_cli_text_list_and_configured_home_are_human_readable(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp_raw,
            patch.dict("os.environ", {EXAMPLES_HOME_ENV: tmp_raw}),
        ):
            code, out, err = self._run(["examples", "list"])

            self.assertEqual(code, 0, msg=out + err)
            self.assertEqual(default_examples_home(), Path(tmp_raw))
            self.assertIn("rocket_booster_recovery\t", out)
            self.assertIn(str(Path(tmp_raw) / "rocket_booster_recovery"), out)
            self.assertIn("rocket_booster_recovery_rust\t", out)
            self.assertIn(str(Path(tmp_raw) / "rocket_booster_recovery_rust"), out)

    def test_bundled_root_prefers_an_installed_package_resource(self) -> None:
        packaged = MagicMock()
        packaged.is_dir.return_value = True
        package_root = MagicMock()
        package_root.joinpath.return_value = packaged

        with patch("praxist.cli.examples.resources.files", return_value=package_root):
            self.assertIs(_bundled_examples_root(), packaged)

    def test_materialization_copies_a_writable_project_without_git_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            destination = Path(tmp_raw) / "rocket"
            result = materialize_example(
                "rocket_booster_recovery",
                destination=destination,
                source_root=SOURCE_ROOT,
            )

            self.assertEqual(result.status, "installed")
            self.assertTrue((destination / "README.md").is_file())
            self.assertTrue((destination / "task_GPU_server" / "task.yaml").is_file())
            self.assertEqual(
                [path for path in destination.rglob("*") if path.name.lower().startswith(".git")],
                [],
            )

    def test_materialization_resolves_the_source_checkout_resource(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            destination = Path(tmp_raw) / "rocket"

            result = materialize_example(
                "rocket_booster_recovery",
                destination=destination,
            )

            self.assertEqual(result.status, "installed")
            self.assertTrue((destination / "task_PC" / "task.yaml").is_file())

    def test_materialization_copies_the_rust_example_and_all_task_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            destination = Path(tmp_raw) / "rocket-rust"

            result = materialize_example(
                "rocket_booster_recovery_rust",
                destination=destination,
                source_root=SOURCE_ROOT,
            )

            self.assertEqual(result.status, "installed")
            self.assertTrue((destination / "Cargo.toml").is_file())
            for task_name in ("task_GPU_server", "task_linux", "task_macos"):
                self.assertTrue((destination / task_name / "task.yaml").is_file())
            self.assertEqual(
                [path for path in destination.rglob("*") if path.name.lower().startswith(".git")],
                [],
            )

    def test_existing_destination_is_preserved_without_copying_over_user_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            destination = Path(tmp_raw) / "rocket"
            destination.mkdir()
            sentinel = destination / "operator-work.txt"
            sentinel.write_text("keep", encoding="utf-8")

            result = materialize_example(
                "rocket_booster_recovery",
                destination=destination,
                source_root=SOURCE_ROOT,
            )

            self.assertEqual(result.status, "preserved_existing")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertFalse((destination / "README.md").exists())

    def test_existing_non_directory_destination_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            destination = Path(tmp_raw) / "rocket"
            destination.write_text("occupied", encoding="utf-8")

            with self.assertRaisesRegex(ExampleInstallError, "not a directory"):
                materialize_example(
                    "rocket_booster_recovery",
                    destination=destination,
                    source_root=SOURCE_ROOT,
                )

    def test_dry_run_reports_destination_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            destination = Path(tmp_raw) / "rocket"
            result = materialize_example(
                "rocket_booster_recovery",
                destination=destination,
                dry_run=True,
                source_root=SOURCE_ROOT,
            )

            self.assertEqual(result.status, "would_install")
            self.assertFalse(destination.exists())

    def test_destination_inside_praxist_source_is_rejected(self) -> None:
        destination = REPO_ROOT / "example-working-copy"

        with self.assertRaisesRegex(ExampleInstallError, "outside the Praxist"):
            materialize_example(
                "rocket_booster_recovery",
                destination=destination,
                dry_run=True,
                source_root=SOURCE_ROOT,
            )
        self.assertFalse(destination.exists())

    def test_unknown_example_and_git_metadata_are_rejected(self) -> None:
        with self.assertRaisesRegex(ExampleInstallError, "unknown bundled example"):
            materialize_example("unknown", source_root=SOURCE_ROOT)

        with tempfile.TemporaryDirectory() as tmp_raw:
            source_root = Path(tmp_raw) / "source"
            example = source_root / "rocket_booster_recovery"
            (example / ".git").mkdir(parents=True)
            with self.assertRaisesRegex(ExampleInstallError, "Git metadata"):
                materialize_example(
                    "rocket_booster_recovery",
                    destination=Path(tmp_raw) / "destination",
                    source_root=source_root,
                )

    def test_missing_source_and_copy_failure_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            root = Path(tmp_raw)
            with self.assertRaisesRegex(ExampleInstallError, "bundled example is missing"):
                materialize_example(
                    "rocket_booster_recovery",
                    destination=root / "missing-destination",
                    source_root=root / "missing-source",
                )

            with (
                patch("praxist.cli.examples.shutil.copytree", side_effect=OSError("disk")),
                self.assertRaisesRegex(ExampleInstallError, "could not install example"),
            ):
                materialize_example(
                    "rocket_booster_recovery",
                    destination=root / "failed-destination",
                    source_root=SOURCE_ROOT,
                )

    def test_concurrent_destination_creation_preserves_the_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            destination = Path(tmp_raw) / "rocket"

            def create_stage_and_winner(_source: Path, stage: Path) -> None:
                stage.mkdir()
                destination.mkdir()
                (destination / "winner.txt").write_text("keep", encoding="utf-8")

            with patch("praxist.cli.examples.shutil.copytree", side_effect=create_stage_and_winner):
                result = materialize_example(
                    "rocket_booster_recovery",
                    destination=destination,
                    source_root=SOURCE_ROOT,
                )

            self.assertEqual(result.status, "preserved_existing")
            self.assertEqual((destination / "winner.txt").read_text(encoding="utf-8"), "keep")

    def test_cli_install_human_outputs_and_error_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            destination = Path(tmp_raw) / "rocket"
            argv = [
                "examples",
                "install",
                "rocket_booster_recovery",
                "--destination",
                str(destination),
            ]

            code, out, err = self._run([*argv, "--dry-run", "--json"])
            self.assertEqual(code, 0, msg=out + err)
            self.assertEqual(json.loads(out)["status"], "would_install")

            code, out, err = self._run([*argv, "--dry-run"])
            self.assertEqual(code, 0, msg=out + err)
            self.assertIn(str(destination), out)
            self.assertIn("Would install", err)

            code, out, err = self._run(argv)
            self.assertEqual(code, 0, msg=out + err)
            self.assertIn("Installed writable", err)

            code, out, err = self._run(argv)
            self.assertEqual(code, 0, msg=out + err)
            self.assertIn("preserved unchanged", err)

        code, out, err = self._run(
            [
                "examples",
                "install",
                "rocket_booster_recovery",
                "--destination",
                str(REPO_ROOT / "forbidden-example-copy"),
            ]
        )
        self.assertEqual(code, 1, msg=out + err)
        self.assertIn("outside the Praxist", err)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
