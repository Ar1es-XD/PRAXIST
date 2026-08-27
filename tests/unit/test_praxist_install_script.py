"""Regression tests for the top-level ``praxist-install.sh`` bootstrapper."""

from __future__ import annotations

import os
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALLER = REPO_ROOT / "praxist-install.sh"
UNINSTALLER = REPO_ROOT / "praxist-uninstall.sh"
RUNTIME_INSTALLER = (
    REPO_ROOT / "skills" / "praxist-runtime-install" / "scripts" / "install_runtime_deps.sh"
)
PYPROJECT = REPO_ROOT / "pyproject.toml"
CLAUDE_AGENT_SDK_REQUIREMENT = "claude-agent-sdk==0.2.136"
CODEX_SDK_REQUIREMENT = "openai-codex==0.147.0"


class PraxistDependencyContract(unittest.TestCase):
    def test_codex_extra_includes_all_sdk_runtime_dependencies(self) -> None:
        with PYPROJECT.open("rb") as file_handle:
            codex_extra = tomllib.load(file_handle)["project"]["optional-dependencies"]["codex"]

        for dependency in (
            CODEX_SDK_REQUIREMENT,
            "codex-relay==0.5.5",
            CLAUDE_AGENT_SDK_REQUIREMENT,
            "mcp>=1.0",
        ):
            with self.subTest(dependency=dependency):
                self.assertIn(dependency, codex_extra)
        self.assertEqual(
            [dependency for dependency in codex_extra if dependency.startswith("openai-codex")],
            [CODEX_SDK_REQUIREMENT],
            "the Codex SDK dependency must match the tested runtime version",
        )
        with PYPROJECT.open("rb") as file_handle:
            agents_extra = tomllib.load(file_handle)["project"]["optional-dependencies"]["agents"]
        self.assertIn(CLAUDE_AGENT_SDK_REQUIREMENT, agents_extra)

    def test_product_usage_client_is_built_in_and_server_dependencies_are_isolated(self) -> None:
        with PYPROJECT.open("rb") as file_handle:
            project = tomllib.load(file_handle)["project"]

        self.assertIn("pydantic>=2.7,<3", project["dependencies"])
        self.assertNotIn("telemetry", project["optional-dependencies"])
        server = project["optional-dependencies"]["product-usage-server"]
        for dependency in ("alembic", "fastapi", "psycopg", "sqlalchemy", "uvicorn"):
            self.assertTrue(any(item.startswith(dependency) for item in server), dependency)
        self.assertEqual(
            project["scripts"]["praxist-collector"],
            "praxist.product_usage.app:main",
        )


class PraxistInstallScriptBehavior(unittest.TestCase):
    """Exercise installer decision logic in dry-run mode only."""

    def _temp_path_with_python(
        self, tmp: Path, *, include_codex: bool = False, include_npm: bool = False
    ) -> str:
        bin_dir = tmp / "bin"
        bin_dir.mkdir()
        python_link = bin_dir / "python3.11"
        try:
            python_link.symlink_to(sys.executable)
        except OSError:
            shutil.copy2(sys.executable, python_link)

        if include_npm:
            npm = bin_dir / "npm"
            npm.write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
            npm.chmod(npm.stat().st_mode | stat.S_IXUSR)

        if include_codex:
            codex = bin_dir / "codex"
            codex.write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
            codex.chmod(codex.stat().st_mode | stat.S_IXUSR)

        return os.pathsep.join([str(bin_dir), "/usr/bin", "/bin"])

    def _run_installer(
        self,
        argv: list[str],
        *,
        include_codex: bool = False,
        include_npm: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory(prefix="praxist_install_test_") as tmp_raw:
            tmp = Path(tmp_raw)
            env = {
                **os.environ,
                "NONINTERACTIVE": "1",
                "PATH": self._temp_path_with_python(
                    tmp, include_codex=include_codex, include_npm=include_npm
                ),
                "XDG_DATA_HOME": str(tmp / "data"),
                "XDG_BIN_HOME": str(tmp / "local-bin"),
                "XDG_CONFIG_HOME": str(tmp / "config"),
            }
            return subprocess.run(
                ["bash", str(INSTALLER), *argv],
                cwd=tmp,
                capture_output=True,
                text=True,
                env=env,
            )

    def _run_fake_interactive_install(
        self,
        *extra_args: str,
        redirect_stderr: bool = False,
        existing_config: str | None = None,
    ) -> tuple[int, str, str]:
        with tempfile.TemporaryDirectory(prefix="praxist_install_oobe_test_") as tmp_raw:
            tmp = Path(tmp_raw)
            path = self._temp_path_with_python(tmp, include_codex=True)
            bin_dir = tmp / "bin"
            call_log = tmp / "praxist-calls.log"
            fake_praxist = tmp / "fake-praxist"
            fake_praxist.write_text(
                '#!/usr/bin/env sh\nprintf "%s\\n" "$*" >> "$PRAXIST_TEST_CALL_LOG"\n',
                encoding="utf-8",
            )
            fake_praxist.chmod(fake_praxist.stat().st_mode | stat.S_IXUSR)
            fake_uv = bin_dir / "uv"
            fake_uv.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'if [[ "${1:-}" == "venv" ]]; then\n'
                '  target="${!#}"\n'
                '  mkdir -p "$target/bin"\n'
                '  cp "$PRAXIST_TEST_FAKE_BIN" "$target/bin/praxist"\n'
                '  chmod +x "$target/bin/praxist"\n'
                "fi\n",
                encoding="utf-8",
            )
            fake_uv.chmod(fake_uv.stat().st_mode | stat.S_IXUSR)
            config_home = tmp / "config"
            if existing_config is not None:
                config_file = config_home / "praxist" / "env"
                config_file.parent.mkdir(parents=True)
                config_file.write_text(existing_config, encoding="utf-8")
            env = {
                **os.environ,
                "INTERACTIVE": "1",
                "PATH": path,
                "PRAXIST_TEST_CALL_LOG": str(call_log),
                "PRAXIST_TEST_FAKE_BIN": str(fake_praxist),
                "XDG_DATA_HOME": str(tmp / "data"),
                "XDG_BIN_HOME": str(tmp / "local-bin"),
                "XDG_CONFIG_HOME": str(config_home),
            }
            env.pop("NONINTERACTIVE", None)
            env.pop("CI", None)
            command = [
                "bash",
                str(INSTALLER),
                "--package",
                "./praxist-0.3.0.tar.gz",
                "--method",
                "uv",
                "--skip-doctor",
                "--no-open-docs",
                "--no-modify-path",
                *extra_args,
            ]
            shell_command = shlex.join(command)
            redirected_error = tmp / "installer-stderr.log"
            if redirect_stderr:
                shell_command += f" 2>{shlex.quote(str(redirected_error))}"
            if sys.platform == "darwin":
                script_command = [
                    "/usr/bin/script",
                    "-q",
                    "/dev/null",
                    "bash",
                    "-c",
                    shell_command,
                ]
            else:
                script_command = [
                    "/usr/bin/script",
                    "-q",
                    "-c",
                    shell_command,
                    "/dev/null",
                ]
            result = subprocess.run(
                script_command,
                cwd=tmp,
                capture_output=True,
                text=True,
                env=env,
            )
            calls = call_log.read_text(encoding="utf-8") if call_log.is_file() else ""
            redirected = (
                redirected_error.read_text(encoding="utf-8") if redirected_error.is_file() else ""
            )
            return result.returncode, result.stdout + result.stderr + redirected, calls

    def test_interactive_codex_native_uses_saved_login_without_api_key_prompt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist_install_interactive_test_") as tmp_raw:
            tmp = Path(tmp_raw)
            path = self._temp_path_with_python(tmp, include_codex=True)
            bin_dir = tmp / "bin"
            call_log = tmp / "praxist-calls.log"
            fake_praxist = tmp / "fake-praxist"
            fake_praxist.write_text(
                '#!/usr/bin/env sh\nprintf "%s\\n" "$*" >> "$PRAXIST_TEST_CALL_LOG"\n',
                encoding="utf-8",
            )
            fake_praxist.chmod(fake_praxist.stat().st_mode | stat.S_IXUSR)
            fake_uv = bin_dir / "uv"
            fake_uv.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'if [[ "${1:-}" == "venv" ]]; then\n'
                '  target="${!#}"\n'
                '  mkdir -p "$target/bin"\n'
                '  cp "$PRAXIST_TEST_FAKE_BIN" "$target/bin/praxist"\n'
                '  chmod +x "$target/bin/praxist"\n'
                "fi\n",
                encoding="utf-8",
            )
            fake_uv.chmod(fake_uv.stat().st_mode | stat.S_IXUSR)
            env = {
                **os.environ,
                "INTERACTIVE": "1",
                "PATH": path,
                "PRAXIST_TEST_CALL_LOG": str(call_log),
                "PRAXIST_TEST_FAKE_BIN": str(fake_praxist),
                "XDG_DATA_HOME": str(tmp / "data"),
                "XDG_BIN_HOME": str(tmp / "local-bin"),
            }
            env.pop("NONINTERACTIVE", None)
            env.pop("CI", None)
            installer_command = [
                "bash",
                str(INSTALLER),
                "--package",
                "./praxist-0.3.0.tar.gz",
                "--method",
                "uv",
                "--provider",
                "openai",
                "--agent-system",
                "codex_sdk",
                "--model",
                "gpt-5.6-luna",
                "--install-skills",
                "none",
                "--install-agent",
                "none",
                "--skip-doctor",
                "--no-start-codex",
                "--no-modify-path",
            ]
            if sys.platform == "darwin":
                script_command = ["/usr/bin/script", "-q", "/dev/null", *installer_command]
            else:
                script_command = [
                    "/usr/bin/script",
                    "-q",
                    "-c",
                    shlex.join(installer_command),
                    "/dev/null",
                ]
            result = subprocess.run(
                script_command,
                cwd=tmp,
                capture_output=True,
                text=True,
                env=env,
            )
            combined = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, msg=combined)
            calls = call_log.read_text(encoding="utf-8")
            configure_call = next(
                line for line in calls.splitlines() if line.startswith("configure-llm ")
            )
            self.assertIn("--no-api-key", configure_call)
            self.assertNotIn("--api-key-stdin", configure_call)
            self.assertIn("product-usage consent", calls)

    def test_noninteractive_package_install_defers_runtime_choice(self) -> None:
        """A non-TTY install cannot convert defaults into an operator decision."""

        result = self._run_installer(
            [
                "--dry-run",
                "--package",
                "./praxist-0.3.0.tar.gz",
                "--skip-doctor",
                "--no-start-codex",
                "--no-modify-path",
            ]
        )
        combined = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, msg=combined)
        self.assertIn("agent install  none via none", combined)
        self.assertIn(
            r"package        ./praxist-0.3.0.tar.gz[agents,codex]",
            combined,
        )
        self.assertIn(
            "product usage  privacy choice offered when collection is available", combined
        )
        self.assertIn("Codex-native support adds roughly 100-150 MB", combined)
        self.assertNotIn("existing consent state preserved (not queried in dry run)", combined)
        self.assertNotIn("consent remains unset; no data is collected", combined)
        self.assertIn("Installing codex-relay from public PyPI", combined)
        self.assertIn("--index-url https://pypi.org/simple codex-relay==0.5.5", combined)
        self.assertIn("Installing Codex SDK from public PyPI", combined)
        self.assertIn(
            f"--index-url https://pypi.org/simple {CODEX_SDK_REQUIREMENT}",
            combined,
        )
        self.assertNotIn("Installing missing codex CLI", combined)
        self.assertIn("Installing Claude Agent SDK for claude_sdk", combined)
        self.assertIn(CLAUDE_AGENT_SDK_REQUIREMENT, combined)
        self.assertIn("mcp", combined)
        self.assertNotIn("--provider deepseek", combined)
        self.assertNotIn("--agent-system claude_sdk", combined)
        self.assertNotIn("--model deepseek-v4-pro", combined)
        self.assertNotIn("install-skills --target codex --replace", combined)
        self.assertNotIn("--force-unmanaged", combined)
        self.assertIn("write Praxist virtualenv ownership marker", combined)
        self.assertIn("praxist-uninstall", combined)
        self.assertIn("Praxist documentation", combined)
        self.assertIn("praxist docs", combined)
        self.assertNotIn("codex -C ", combined)
        self.assertNotIn("Cannot install missing agent CLIs", combined)
        self.assertIn("OOBE checkpoint package installed", combined)
        self.assertIn("setup --agent-managed", combined)
        self.assertIn("setup_decisions_complete", combined)
        self.assertIn("examples install rocket_booster_recovery", combined)
        self.assertIn("examples install rocket_booster_recovery_rust", combined)

    def test_skip_setup_reports_resumable_oobe_checkpoint(self) -> None:
        result = self._run_installer(
            [
                "--dry-run",
                "--package",
                "./praxist-0.3.0.tar.gz",
                "--skip-setup",
                "--no-modify-path",
            ]
        )
        combined = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, msg=combined)
        self.assertIn("OOBE checkpoint package installed", combined)
        self.assertIn("setup --agent-managed", combined)
        self.assertIn("setup --interactive", combined)

    def test_installer_contains_actionable_tls_failure_without_insecure_bypass(self) -> None:
        text = INSTALLER.read_text(encoding="utf-8")

        self.assertIn("SSLCertVerificationError", text)
        self.assertIn("Install Certificates.command", text)
        self.assertIn("Do not disable TLS verification", text)
        self.assertNotIn("PIP_TRUSTED_HOST", text)

    def test_one_click_uninstaller_is_a_portable_bash_wrapper(self) -> None:
        text = UNINSTALLER.read_text(encoding="utf-8")

        self.assertTrue(os.access(UNINSTALLER, os.X_OK))
        self.assertIn("praxist", text)
        self.assertIn('uninstall "$@"', text)
        self.assertNotIn("rm -", text)

    def test_no_open_docs_prints_without_browser_launch(self) -> None:
        result = self._run_installer(
            [
                "--dry-run",
                "--package",
                "./praxist-0.3.0.tar.gz",
                "--skip-setup",
                "--skip-doctor",
                "--no-open-docs",
                "--no-modify-path",
            ]
        )
        combined = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, msg=combined)
        self.assertIn("praxist docs --no-open", combined)

    def test_skip_setup_also_skips_provider_doctor(self) -> None:
        result = self._run_installer(
            [
                "--dry-run",
                "--package",
                "./praxist-0.3.0.tar.gz",
                "--skip-setup",
                "--no-open-docs",
                "--no-modify-path",
            ]
        )
        combined = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, msg=combined)
        self.assertNotIn("doctor --agent-system", combined)

    def test_local_package_archive_includes_requested_storage_extra(self) -> None:
        result = self._run_installer(
            [
                "--dry-run",
                "--package",
                "./praxist-0.3.0.tar.gz",
                "--with-storage",
                "--skip-setup",
                "--skip-doctor",
                "--no-modify-path",
            ]
        )
        combined = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, msg=combined)
        self.assertIn(
            r"package        ./praxist-0.3.0.tar.gz[agents,codex,storage]",
            combined,
        )

    def test_source_checkout_package_receives_default_runtime_extras(self) -> None:
        result = self._run_installer(
            [
                "--dry-run",
                "--package",
                str(REPO_ROOT),
                "--skip-setup",
                "--skip-doctor",
                "--no-modify-path",
            ]
        )
        combined = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, msg=combined)
        self.assertIn(
            f"package        {REPO_ROOT}[agents,codex]",
            combined,
        )

    def test_explicit_package_extras_are_not_rewritten(self) -> None:
        result = self._run_installer(
            [
                "--dry-run",
                "--package",
                "./praxist-0.3.0.tar.gz[agents]",
                "--skip-setup",
                "--skip-doctor",
                "--no-modify-path",
            ]
        )
        combined = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, msg=combined)
        self.assertIn(r"package        ./praxist-0.3.0.tar.gz[agents]", combined)
        self.assertNotIn("tar.gz[agents,codex]", combined)
        self.assertNotIn("Installing codex-relay from public PyPI", combined)
        self.assertNotIn("Installing Codex SDK from public PyPI", combined)

    def test_start_codex_preserves_runtime_install_provisioning(self) -> None:
        result = self._run_installer(
            [
                "--dry-run",
                "--package",
                "./praxist-0.3.0.tar.gz",
                "--provider",
                "deepseek",
                "--api-key-env",
                "DEEPSEEK_API_KEY",
                "--skip-doctor",
                "--start-codex",
                "--no-modify-path",
            ],
            include_codex=True,
        )
        combined = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, msg=combined)
        self.assertIn("Launching Codex with Praxist Runtime Install", combined)
        self.assertIn("codex -C", combined)
        self.assertIn("$praxist-runtime-install", combined)
        self.assertNotIn("Starting Praxist first-project takeover", combined)

    def test_start_takeover_uses_the_guided_cli_handoff(self) -> None:
        result = self._run_installer(
            [
                "--dry-run",
                "--package",
                "./praxist-0.3.0.tar.gz",
                "--provider",
                "deepseek",
                "--skip-doctor",
                "--start-takeover",
                "--no-modify-path",
            ]
        )
        combined = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, msg=combined)
        self.assertIn("Starting Praxist first-project takeover", combined)
        self.assertIn("praxist --takeover", combined)

    def test_live_takeover_requires_codex_before_installing(self) -> None:
        result = self._run_installer(
            [
                "--package",
                "./praxist-0.3.0.tar.gz",
                "--start-takeover",
                "--no-modify-path",
            ]
        )
        combined = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Codex CLI is required for guided takeover", combined)
        self.assertNotIn("Installing Praxist", combined)

    def test_live_takeover_requires_an_interactive_terminal_before_installing(self) -> None:
        result = self._run_installer(
            ["--package", "./praxist-0.3.0.tar.gz", "--start-takeover"],
            include_codex=True,
        )
        combined = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires a local interactive terminal", combined)
        self.assertNotIn("Installing Praxist", combined)

    def test_disabled_agent_installer_cannot_satisfy_takeover_preflight(self) -> None:
        result = self._run_installer(
            [
                "--package",
                "./praxist-0.3.0.tar.gz",
                "--start-takeover",
                "--install-agent",
                "codex",
                "--agent-installer",
                "none",
            ]
        )
        combined = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Codex CLI is required for guided takeover", combined)
        self.assertNotIn("Installing Praxist", combined)

    def test_unavailable_requested_agent_installer_fails_before_install(self) -> None:
        result = self._run_installer(
            [
                "--package",
                "./praxist-0.3.0.tar.gz",
                "--start-takeover",
                "--install-agent",
                "codex",
                "--agent-installer",
                "brew",
            ]
        )
        combined = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("brew requested, but brew is not on PATH", combined)
        self.assertNotIn("Installing Praxist", combined)

    def test_start_takeover_rejects_setup_or_skill_bypass(self) -> None:
        for option in ("--skip-setup", "--install-skills"):
            argv = ["--dry-run", "--start-takeover", option]
            if option == "--install-skills":
                argv.append("none")
            with self.subTest(option=option):
                result = self._run_installer(argv, include_codex=True)
                combined = result.stdout + result.stderr
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--start-takeover", combined)
                self.assertNotIn("Installing Praxist", combined)

    def test_wizard_without_skill_install_does_not_auto_start_takeover(self) -> None:
        code, output, calls = self._run_fake_interactive_install("--install-skills", "none")
        self.assertEqual(code, 0, msg=output)
        self.assertIn("setup --interactive --install-skills none", calls)
        self.assertNotIn("--takeover", calls)

    def test_redirected_stderr_defers_setup_without_partial_wizard(self) -> None:
        code, output, calls = self._run_fake_interactive_install(redirect_stderr=True)
        self.assertEqual(code, 0, msg=output)
        self.assertNotIn("setup --interactive", calls)
        self.assertNotIn("configure-llm", calls)
        self.assertNotIn("--takeover", calls)
        self.assertIn("OOBE checkpoint package installed", output)
        self.assertIn("setup --agent-managed", output)

    def test_explicit_no_env_file_preserves_noninteractive_install_flow(self) -> None:
        code, output, calls = self._run_fake_interactive_install("--no-env-file")
        self.assertEqual(code, 0, msg=output)
        self.assertNotIn("setup --interactive", calls)
        self.assertIn("configure-llm --provider deepseek", calls)
        self.assertNotIn("--takeover", calls)

    def test_existing_setup_does_not_auto_start_first_takeover(self) -> None:
        code, output, calls = self._run_fake_interactive_install(
            existing_config=(
                "PRAXIST_AGENT_SYSTEM=codex_sdk\n"
                "PRAXIST_LLM_PROVIDER=openai\n"
                "PRAXIST_MODEL=gpt-5.6-luna\n"
            )
        )
        self.assertEqual(code, 0, msg=output)
        self.assertIn("setup --interactive", calls)
        self.assertNotIn("--takeover", calls)
        self.assertIn("choose a research project with:", output)
        self.assertIn("--takeover", output)

    def test_last_start_mode_flag_wins(self) -> None:
        result = self._run_installer(
            [
                "--dry-run",
                "--package",
                "./praxist-0.3.0.tar.gz",
                "--provider",
                "deepseek",
                "--skip-doctor",
                "--start-codex",
                "--start-takeover",
                "--no-modify-path",
            ]
        )
        combined = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, msg=combined)
        self.assertIn("Starting Praxist first-project takeover", combined)
        self.assertNotIn("Launching Codex with Praxist Runtime Install", combined)

    def test_api_key_env_without_explicit_provider_fails_before_install(self) -> None:
        result = self._run_installer(
            [
                "--dry-run",
                "--package",
                "./praxist-0.3.0.tar.gz",
                "--api-key-env",
                "OPENAI_API_KEY",
                "--no-modify-path",
            ]
        )
        combined = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--api-key-env requires --provider", combined)
        self.assertNotIn("Installing Praxist", combined)

    def test_unconfigured_interactive_install_uses_setup_and_takeover_wizards(self) -> None:
        code, output, calls = self._run_fake_interactive_install()
        self.assertEqual(code, 0, msg=output)
        self.assertIn(
            "setup --interactive --install-skills codex --skip-doctor",
            calls,
        )
        self.assertIn("--takeover", calls)
        self.assertNotIn("configure-llm", calls)

    def test_claude_skill_host_flows_through_setup_doctor_and_takeover(self) -> None:
        result = self._run_installer(
            [
                "--dry-run",
                "--package",
                "./praxist-0.3.0.tar.gz",
                "--provider",
                "deepseek",
                "--agent-system",
                "claude_sdk",
                "--install-skills",
                "claude",
                "--start-takeover",
                "--no-open-docs",
                "--no-modify-path",
            ]
        )
        combined = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, msg=combined)
        self.assertIn("install-skills --target claude --replace", combined)
        self.assertIn("doctor --agent-system claude_sdk --target claude", combined)
        self.assertIn("--takeover --operator claude", combined)

    def test_codex_sdk_uses_bundled_runtime_and_relay(self) -> None:
        result = self._run_installer(
            [
                "--dry-run",
                "--package",
                "./praxist-0.3.0.tar.gz",
                "--agent-system",
                "codex_sdk",
                "--install-agent",
                "none",
                "--skip-setup",
                "--skip-doctor",
                "--no-start-codex",
                "--no-modify-path",
            ],
        )
        combined = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, msg=combined)
        self.assertNotIn("Installing missing codex CLI", combined)
        self.assertIn("Installing Codex SDK runtime dependencies for codex_sdk", combined)
        self.assertIn(CODEX_SDK_REQUIREMENT, combined)
        self.assertIn("Installing codex-relay from public PyPI", combined)
        self.assertIn("--index-url https://pypi.org/simple codex-relay==0.5.5", combined)
        self.assertIn("Installing Codex SDK from public PyPI", combined)
        self.assertIn(
            f"--index-url https://pypi.org/simple {CODEX_SDK_REQUIREMENT}",
            combined,
        )
        self.assertIn("codex-relay==0.5.5", combined)
        self.assertIn(CLAUDE_AGENT_SDK_REQUIREMENT, combined)
        self.assertIn(r"mcp\>=1.0", combined)
        self.assertNotIn("Installing missing claude CLI", combined)

    def test_claude_sdk_runtime_installs_without_external_agent_cli(self) -> None:
        result = self._run_installer(
            [
                "--dry-run",
                "--package",
                "./praxist-0.3.0.tar.gz",
                "--agent-system",
                "claude_sdk",
                "--install-agent",
                "none",
                "--skip-setup",
                "--skip-doctor",
                "--no-start-codex",
                "--no-modify-path",
            ],
        )
        combined = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, msg=combined)
        self.assertNotIn("Installing missing claude CLI", combined)
        self.assertNotIn("Installing missing codex CLI", combined)
        self.assertIn("Installing Claude Agent SDK for claude_sdk", combined)
        self.assertIn(CLAUDE_AGENT_SDK_REQUIREMENT, combined)

    def test_legacy_claude_runtime_alias_remains_supported(self) -> None:
        result = self._run_installer(
            [
                "--dry-run",
                "--package",
                "./praxist-0.3.0.tar.gz",
                "--agent-system",
                "claude",
                "--install-agent",
                "none",
                "--skip-doctor",
                "--no-start-codex",
                "--no-modify-path",
            ],
        )
        combined = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, msg=combined)
        self.assertIn("Installing Claude Agent SDK for claude_sdk", combined)
        self.assertIn(CLAUDE_AGENT_SDK_REQUIREMENT, combined)
        self.assertIn("--agent-system claude_sdk", combined)
        self.assertNotIn("Installing missing claude CLI", combined)


class PraxistRuntimeInstallScriptBehavior(unittest.TestCase):
    def _run_source_dry_run(self, method: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory(prefix="praxist_runtime_install_test_") as tmp:
            return subprocess.run(
                [
                    "bash",
                    str(RUNTIME_INSTALLER),
                    "--repo",
                    str(REPO_ROOT),
                    "--method",
                    method,
                    "--target",
                    str(Path(tmp) / "venv"),
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                env={**os.environ, "NONINTERACTIVE": "1"},
            )

    def test_pip_source_install_fetches_codex_packages_from_public_pypi(self) -> None:
        result = self._run_source_dry_run("pip")
        combined = result.stdout + result.stderr

        self.assertEqual(result.returncode, 0, msg=combined)
        self.assertIn(
            "--index-url https://pypi.org/simple codex-relay==0.5.5",
            combined,
        )
        self.assertIn(
            f"--index-url https://pypi.org/simple {CODEX_SDK_REQUIREMENT}",
            combined,
        )

    def test_uv_source_sync_defers_codex_extra_to_public_pypi_install(self) -> None:
        result = self._run_source_dry_run("uv")
        combined = result.stdout + result.stderr

        self.assertEqual(result.returncode, 0, msg=combined)
        self.assertIn("uv sync --group dev --extra agents", combined)
        self.assertNotIn("uv sync --group dev --extra agents --extra codex", combined)
        self.assertIn(
            "--index-url https://pypi.org/simple codex-relay==0.5.5",
            combined,
        )
        self.assertIn(
            f"--index-url https://pypi.org/simple {CODEX_SDK_REQUIREMENT}",
            combined,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
