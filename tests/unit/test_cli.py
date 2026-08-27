"""Lock the public surface of ``praxist.cli`` (backfill issue #32).

This is the foundation for every later ``praxist <subcommand>`` issue
(#33, #44, #45, #47, #48, #50, #51, #52, #53). The tests here only assert
the dispatcher contract; concrete subcommands are added in their own
issues and own their own tests.
"""

from __future__ import annotations

import io
import os
import tomllib
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]


class CliModuleSurface(unittest.TestCase):
    def test_module_is_importable(self) -> None:
        import praxist.cli as cli  # noqa: PLC0415

        self.assertTrue(callable(cli.main), "praxist.cli.main must be callable")

    def test_build_parser_returns_argument_parser(self) -> None:
        import argparse  # noqa: PLC0415

        from praxist.cli import _build_parser  # noqa: PLC0415

        parser = _build_parser()
        self.assertIsInstance(parser, argparse.ArgumentParser)
        self.assertEqual(parser.prog, "praxist")


class CliDispatcher(unittest.TestCase):
    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        from praxist.cli import main  # noqa: PLC0415

        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                main(argv)
            code = 0
        except SystemExit as exc:
            code = int(exc.code or 0)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_no_args_prints_help_exit_zero(self) -> None:
        code, out, err = self._run([])
        combined = out + err
        self.assertEqual(code, 0, msg=combined)
        self.assertIn("praxist", combined.lower())

    def test_help_flag_exits_zero_and_mentions_program(self) -> None:
        code, out, err = self._run(["--help"])
        self.assertEqual(code, 0)
        self.assertIn("praxist", (out + err).lower())

    def test_unknown_subcommand_exits_nonzero(self) -> None:
        code, _out, _err = self._run(["definitely-not-a-subcommand"])
        self.assertNotEqual(code, 0)

    def test_docs_prints_url_without_opening_when_requested(self) -> None:
        from praxist.cli.docs import DOCUMENTATION_URL  # noqa: PLC0415

        with patch("praxist.cli.docs.webbrowser.open_new_tab") as open_browser:
            code, out, err = self._run(["docs", "--no-open"])

        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), DOCUMENTATION_URL)
        self.assertEqual(err, "")
        open_browser.assert_not_called()

    def test_docs_opens_browser_on_local_linux_desktop(self) -> None:
        clean_environment = {
            key: value
            for key, value in os.environ.items()
            if key not in {"CI", "SSH_CONNECTION", "SSH_TTY", "PRAXIST_DOCS_NO_OPEN"}
        }
        clean_environment["DISPLAY"] = ":0"
        with (
            patch.dict(os.environ, clean_environment, clear=True),
            patch("praxist.cli.docs.sys.platform", "linux"),
            patch("praxist.cli.docs.webbrowser.open_new_tab", return_value=True) as open_browser,
        ):
            code, _out, err = self._run(["docs"])

        self.assertEqual(code, 0)
        self.assertIn("Opened Praxist documentation", err)
        open_browser.assert_called_once()

    def test_docs_does_not_open_remote_browser_over_ssh(self) -> None:
        with (
            patch.dict(os.environ, {"DISPLAY": ":0", "SSH_CONNECTION": "remote"}, clear=True),
            patch("praxist.cli.docs.sys.platform", "linux"),
            patch("praxist.cli.docs.webbrowser.open_new_tab") as open_browser,
        ):
            code, _out, err = self._run(["docs"])

        self.assertEqual(code, 0)
        self.assertIn("No local browser detected", err)
        open_browser.assert_not_called()


class PyprojectConsoleScripts(unittest.TestCase):
    """The whole point of #32: ``pip install -e .`` must produce an ``praxist`` binary."""

    def setUp(self) -> None:
        path = REPO_ROOT / "pyproject.toml"
        self.assertTrue(path.exists(), "pyproject.toml is missing from repo root")
        self.data = tomllib.loads(path.read_text(encoding="utf-8"))

    def test_project_metadata_declares_python_requirement(self) -> None:
        requires = self.data.get("project", {}).get("requires-python", "")
        self.assertRegex(requires, r">=\s*3\.1[1-9]")

    def test_praxist_console_script_points_to_cli_main(self) -> None:
        scripts = self.data.get("project", {}).get("scripts", {})
        self.assertEqual(scripts.get("praxist"), "praxist.cli:main")

    def test_project_metadata_exposes_documentation_url(self) -> None:
        from praxist.cli.docs import DOCUMENTATION_URL  # noqa: PLC0415

        urls = self.data.get("project", {}).get("urls", {})
        self.assertEqual(urls.get("Documentation"), DOCUMENTATION_URL)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
