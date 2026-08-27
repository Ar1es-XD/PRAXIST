#!/usr/bin/env bash
# Install Praxist from PyPI into a user-managed virtual environment.
#
# This script is intentionally a thin bootstrapper. It installs the Praxist
# Python package, default runtime SDK packages, and bundled Codex skills. Codex
# may already be usable or may be installed explicitly with --install-agent.
# The $praxist-runtime-install skill remains the interactive repair/provisioning
# path when an existing install needs attention.

set -euo pipefail

APP_NAME="Praxist"
# Filesystem slug for user-level install paths. Keep this as a top-level
# replacement point alongside PRAXIST_PYPI_PACKAGE_NAME.
PRAXIST_INSTALL_APP_SLUG="${PRAXIST_INSTALL_APP_SLUG:-praxist}"
# PyPI distribution name for Praxist. Keep this as a top-level replacement point:
# the public package name may change before the installer is published.
PRAXIST_PYPI_PACKAGE_NAME="${PRAXIST_PYPI_PACKAGE_NAME:-praxist}"
# Codex CLI package names. These are top-level replacement points because
# upstream Homebrew cask and npm package names may change.
PRAXIST_INSTALL_CODEX_BREW="${PRAXIST_INSTALL_CODEX_BREW:-codex}"
PRAXIST_INSTALL_CODEX_NPM="${PRAXIST_INSTALL_CODEX_NPM:-@openai/codex}"
PRAXIST_INSTALL_CODEX_SDK_PACKAGE="${PRAXIST_INSTALL_CODEX_SDK_PACKAGE:-openai-codex==0.147.0}"
PRAXIST_INSTALL_CODEX_RELAY_PACKAGE="${PRAXIST_INSTALL_CODEX_RELAY_PACKAGE:-codex-relay==0.5.5}"
PRAXIST_INSTALL_CODEX_INDEX_URL="${PRAXIST_INSTALL_CODEX_INDEX_URL:-https://pypi.org/simple}"
PRAXIST_INSTALL_CLAUDE_SDK_PACKAGE="${PRAXIST_INSTALL_CLAUDE_SDK_PACKAGE:-claude-agent-sdk==0.2.136}"
PRAXIST_INSTALL_MCP_PACKAGE="${PRAXIST_INSTALL_MCP_PACKAGE:-mcp>=1.0}"
install_defaults_explicit=0
if [[ -n "${PRAXIST_INSTALL_DEFAULT_PROVIDER+x}" \
  || -n "${PRAXIST_INSTALL_DEFAULT_AGENT_SYSTEM+x}" \
  || -n "${PRAXIST_INSTALL_DEFAULT_MODEL+x}" ]]; then
  install_defaults_explicit=1
fi
setup_inputs_explicit="${install_defaults_explicit}"
PRAXIST_INSTALL_DEFAULT_PROVIDER="${PRAXIST_INSTALL_DEFAULT_PROVIDER:-deepseek}"
PRAXIST_INSTALL_DEFAULT_AGENT_SYSTEM="${PRAXIST_INSTALL_DEFAULT_AGENT_SYSTEM:-claude_sdk}"
PRAXIST_INSTALL_DEFAULT_MODEL="${PRAXIST_INSTALL_DEFAULT_MODEL:-deepseek-v4-pro}"
PRAXIST_INSTALL_CODEX_PROMPT="${PRAXIST_INSTALL_CODEX_PROMPT:-}"
DEFAULT_EXTRAS="agents,codex"

abort() {
  printf "%s\n" "$@" >&2
  exit 1
}

if [[ -z "${BASH_VERSION:-}" ]]; then
  abort "Bash is required to interpret this script."
fi

if [[ -n "${POSIXLY_CORRECT+1}" ]]; then
  abort "Bash must not run in POSIX mode. Please unset POSIXLY_CORRECT and try again."
fi

if [[ -n "${INTERACTIVE:-}" && -n "${NONINTERACTIVE:-}" ]]; then
  abort 'Both `$INTERACTIVE` and `$NONINTERACTIVE` are set. Please unset one.'
fi

if [[ -z "${NONINTERACTIVE:-}" ]]; then
  if [[ -n "${CI:-}" ]]; then
    NONINTERACTIVE=1
  elif [[ ! -t 0 && -z "${INTERACTIVE:-}" ]]; then
    NONINTERACTIVE=1
  fi
fi

if [[ -t 1 ]]; then
  tty_escape() { printf "\033[%sm" "$1"; }
else
  tty_escape() { :; }
fi
tty_mkbold() { tty_escape "1;$1"; }
tty_blue="$(tty_mkbold 34)"
tty_red="$(tty_mkbold 31)"
tty_bold="$(tty_mkbold 39)"
tty_reset="$(tty_escape 0)"

ohai() {
  printf "${tty_blue}==>${tty_bold} %s${tty_reset}\n" "$*"
}

warn() {
  printf "${tty_red}Warning${tty_reset}: %s\n" "$*" >&2
}

usage() {
  cat <<'EOF'
Praxist installer

Usage:
  praxist-install.sh [options]

Install options:
  --version VERSION          Install a specific Praxist package version.
  --package SPEC             Override the pip package spec. Local .whl/.tar.gz/.zip
                             archives receive the default runtime extras unless
                             SPEC already declares extras.
  --with-storage             Include the storage extra.
  --venv-dir DIR             Virtualenv path. Default: ${XDG_DATA_HOME:-$HOME/.local/share}/$PRAXIST_INSTALL_APP_SLUG/venv
  --bin-dir DIR              Symlink directory for `praxist`. Default: ${XDG_BIN_HOME:-$HOME/.local/bin}
  --python BIN               Python executable. Default: python3.11, python3, then python
  --method auto|uv|pip       Install method. Default: auto
  --no-modify-path           Do not print shell rc PATH instructions.

Praxist setup options:
  --agent-system VALUE       Persist PRAXIST_AGENT_SYSTEM via `praxist configure-llm`.
  --provider VALUE           Persist PRAXIST_LLM_PROVIDER via `praxist configure-llm`.
  --model VALUE              Persist PRAXIST_MODEL via `praxist configure-llm`.
  --api-key-env VAR          Pass a provider API key by env var name only.
  --env-file PATH            Explicit task env file to write. Default: none
  --no-env-file              Do not write a task env file (the default).
  --install-skills TARGET    codex, claude, or none. Default: codex
  --install-agent TARGET     codex or none. Default: none
  --agent-installer TOOL     auto, brew, npm, or none. Default: none
  --no-install-agent         Alias for --install-agent none.
  --skip-setup               Install package only; skip LLM config and skills.
  --skip-doctor              Do not run the final `praxist doctor`.
  --start-codex              Launch Codex with the runtime-install skill.
  --no-start-codex           Do not launch the runtime-install skill (default).
  --start-takeover           Enter guided first-project takeover after setup.
  --no-start-takeover        Finish after setup without entering takeover.
  --no-open-docs             Print the documentation URL without opening a browser.

Other:
  --dry-run                  Print commands without executing them.
  -h, --help                 Show this help.

Package name:
  PRAXIST_INSTALL_APP_SLUG       Filesystem slug for install paths. Default: praxist
  PRAXIST_PYPI_PACKAGE_NAME      PyPI distribution name. Default: praxist
  PRAXIST_INSTALL_CODEX_BREW     Homebrew cask for Codex CLI. Default: codex
  PRAXIST_INSTALL_CODEX_NPM      npm package for Codex CLI. Default: @openai/codex
  PRAXIST_INSTALL_CODEX_SDK_PACKAGE      Python package for codex_sdk. Default: openai-codex==0.147.0
  PRAXIST_INSTALL_CODEX_RELAY_PACKAGE    Python package for codex-relay. Default: codex-relay==0.5.5
  PRAXIST_INSTALL_CODEX_INDEX_URL        Package index used only for Codex SDK/relay. Default: https://pypi.org/simple
  PRAXIST_INSTALL_CLAUDE_SDK_PACKAGE     Python package for SDK MCP factories. Default: claude-agent-sdk==0.2.136
  PRAXIST_INSTALL_MCP_PACKAGE            Python MCP package used by SDK tools. Default: mcp>=1.0
  PRAXIST_INSTALL_DEFAULT_PROVIDER       Default Praxist provider. Default: deepseek
  PRAXIST_INSTALL_DEFAULT_AGENT_SYSTEM   Default Praxist agent system. Default: claude_sdk
  PRAXIST_INSTALL_DEFAULT_MODEL          Default Praxist model. Default: deepseek-v4-pro
  PRAXIST_INSTALL_CODEX_PROMPT           Override the Codex provisioning prompt.

Codex must already be installed and usable unless --install-agent codex is
selected. Authentication is required only when Codex is used interactively or
by Codex-native mode. This installer does not install CUDA or task dependencies.
Interactive API keys are entered in a local length-preserving masked prompt.
EOF
}

version=""
package_spec=""
with_storage=0
default_venv_dir="${XDG_DATA_HOME:-${HOME}/.local/share}/${PRAXIST_INSTALL_APP_SLUG}/venv"
venv_dir="${default_venv_dir}"
bin_dir="${XDG_BIN_HOME:-${HOME}/.local/bin}"
python_bin=""
method="auto"
modify_path=1
agent_system=""
provider=""
model=""
api_key_env=""
project_env_file=""
project_env_explicit=0
install_skills="codex"
install_agent="none"
agent_installer="none"
skip_setup=0
skip_doctor=0
start_codex=0
start_takeover=0
start_takeover_explicit=0
open_docs=1
dry_run=0
interactive_setup=0
setup_wizard=0
provider_explicit=0
existing_setup=0
venv_created=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      [[ $# -ge 2 ]] || abort "error: --version requires a value"
      version="$2"
      shift 2
      ;;
    --package)
      [[ $# -ge 2 ]] || abort "error: --package requires a value"
      package_spec="$2"
      shift 2
      ;;
    --with-storage)
      with_storage=1
      shift
      ;;
    --venv-dir)
      [[ $# -ge 2 ]] || abort "error: --venv-dir requires a path"
      venv_dir="$2"
      shift 2
      ;;
    --bin-dir)
      [[ $# -ge 2 ]] || abort "error: --bin-dir requires a path"
      bin_dir="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || abort "error: --python requires an executable"
      python_bin="$2"
      shift 2
      ;;
    --method)
      [[ $# -ge 2 ]] || abort "error: --method requires auto, uv, or pip"
      method="$2"
      shift 2
      ;;
    --no-modify-path)
      modify_path=0
      shift
      ;;
    --agent-system)
      [[ $# -ge 2 ]] || abort "error: --agent-system requires a value"
      agent_system="$2"
      setup_inputs_explicit=1
      shift 2
      ;;
    --provider)
      [[ $# -ge 2 ]] || abort "error: --provider requires a value"
      provider="$2"
      provider_explicit=1
      setup_inputs_explicit=1
      shift 2
      ;;
    --model)
      [[ $# -ge 2 ]] || abort "error: --model requires a value"
      model="$2"
      setup_inputs_explicit=1
      shift 2
      ;;
    --api-key-env)
      [[ $# -ge 2 ]] || abort "error: --api-key-env requires an environment variable name"
      api_key_env="$2"
      setup_inputs_explicit=1
      shift 2
      ;;
    --env-file)
      [[ $# -ge 2 ]] || abort "error: --env-file requires a path"
      project_env_file="$2"
      project_env_explicit=1
      shift 2
      ;;
    --no-env-file)
      project_env_file=""
      project_env_explicit=1
      shift
      ;;
    --install-skills)
      [[ $# -ge 2 ]] || abort "error: --install-skills requires codex, claude, or none"
      install_skills="$2"
      shift 2
      ;;
    --install-agent)
      [[ $# -ge 2 ]] || abort "error: --install-agent requires codex or none"
      install_agent="$2"
      shift 2
      ;;
    --agent-installer)
      [[ $# -ge 2 ]] || abort "error: --agent-installer requires auto, brew, npm, or none"
      agent_installer="$2"
      shift 2
      ;;
    --no-install-agent)
      install_agent="none"
      shift
      ;;
    --skip-setup)
      skip_setup=1
      shift
      ;;
    --skip-doctor)
      skip_doctor=1
      shift
      ;;
    --start-codex)
      start_codex=1
      start_takeover=0
      start_takeover_explicit=1
      shift
      ;;
    --no-start-codex)
      start_codex=0
      start_takeover=0
      start_takeover_explicit=1
      shift
      ;;
    --start-takeover)
      start_codex=0
      start_takeover=1
      start_takeover_explicit=1
      shift
      ;;
    --no-start-takeover)
      start_takeover=0
      start_takeover_explicit=1
      shift
      ;;
    --no-open-docs)
      open_docs=0
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      abort "error: unknown argument: $1"
      ;;
  esac
done

if [[ "${start_takeover}" -eq 1 && "${skip_setup}" -eq 1 ]]; then
  abort "error: --start-takeover cannot be combined with --skip-setup; run praxist --takeover after setup instead"
fi
if [[ "${start_takeover}" -eq 1 && "${install_skills}" == "none" ]]; then
  abort "error: --start-takeover requires --install-skills codex or claude"
fi

case "${method}" in
  auto|uv|pip) ;;
  *) abort "error: --method must be auto, uv, or pip" ;;
esac

case "${install_skills}" in
  codex|claude|none) ;;
  *) abort "error: --install-skills must be codex, claude, or none" ;;
esac

case "${install_agent}" in
  codex|none) ;;
  *) abort "error: --install-agent must be codex or none" ;;
esac

case "${agent_installer}" in
  auto|brew|npm|none) ;;
  *) abort "error: --agent-installer must be auto, brew, npm, or none" ;;
esac

if [[ -z "${NONINTERACTIVE:-}" && -t 0 && -t 1 && -t 2 ]]; then
  interactive_setup=1
fi

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${dry_run}" -eq 0 ]]; then
    "$@"
  fi
}

run_install_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${dry_run}" -eq 1 ]]; then
    return
  fi
  local error_log status
  error_log="$(mktemp "${TMPDIR:-/tmp}/praxist-install.XXXXXX")"
  if "$@" 2>&1 | tee "${error_log}" >&2; then
    status=0
  else
    status="${PIPESTATUS[0]}"
  fi
  if [[ "${status}" -eq 0 ]]; then
    rm -f "${error_log}"
    return
  fi
  if grep -Eiq 'SSLCertVerificationError|CERTIFICATE_VERIFY_FAILED|certificate verify failed' "${error_log}"; then
    warn "Python could not verify the package index TLS certificate."
    if [[ "${platform:-}" == "Darwin" ]]; then
      printf '%s\n' \
        "Repair the selected Python certificate bundle (python.org installers provide an 'Install Certificates.command'), then rerun this installer." \
        "Do not disable TLS verification or place API keys in command arguments." >&2
    else
      printf '%s\n' \
        "Repair the host CA bundle or the selected Python trust configuration, then rerun this installer." \
        "Do not disable TLS verification or place API keys in command arguments." >&2
    fi
  fi
  rm -f "${error_log}"
  return "${status}"
}

normalize_agent_system_alias() {
  case "${agent_system}" in
    codex) agent_system="codex_sdk" ;;
    claude) agent_system="claude_sdk" ;;
  esac
}

choose_python() {
  if [[ -n "${python_bin}" ]]; then
    command -v "${python_bin}" >/dev/null 2>&1 || abort "Python executable not found: ${python_bin}"
    printf '%s\n' "${python_bin}"
    return
  fi
  for candidate in python3.11 python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  abort "Python 3.11+ is required, but no python executable was found."
}

validate_python() {
  local candidate="$1"
  if ! "${candidate}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
    abort "Python 3.11+ is required; ${candidate} is too old."
  fi
}

package_requirement() {
  local extras="${DEFAULT_EXTRAS}"
  if [[ "${with_storage}" -eq 1 ]]; then
    extras="${extras},storage"
  fi
  if [[ -n "${package_spec}" ]]; then
    case "${package_spec}" in
      *"["*"]"*|*"://"*|*" @ "*)
        ;;
      *.whl|*.tar.gz|*.zip)
        package_spec="${package_spec}[${extras}]"
        ;;
      *)
        if [[ -d "${package_spec}" ]]; then
          package_spec="${package_spec}[${extras}]"
        fi
        ;;
    esac
    printf '%s\n' "${package_spec}"
    return
  fi
  if [[ -n "${version}" ]]; then
    printf '%s[%s]==%s\n' "${PRAXIST_PYPI_PACKAGE_NAME}" "${extras}" "${version}"
  else
    printf '%s[%s]\n' "${PRAXIST_PYPI_PACKAGE_NAME}" "${extras}"
  fi
}

apply_setup_defaults() {
  if [[ -z "${provider}" ]]; then
    provider="${PRAXIST_INSTALL_DEFAULT_PROVIDER}"
  fi
  if [[ -z "${agent_system}" ]]; then
    agent_system="${PRAXIST_INSTALL_DEFAULT_AGENT_SYSTEM}"
  fi
  if [[ -z "${model}" ]]; then
    model="${PRAXIST_INSTALL_DEFAULT_MODEL}"
  fi
  normalize_agent_system_alias
}

choose_agent_installer() {
  case "${agent_installer}" in
    brew)
      command -v brew >/dev/null 2>&1 || abort "--agent-installer brew requested, but brew is not on PATH"
      printf 'brew\n'
      return
      ;;
    npm)
      command -v npm >/dev/null 2>&1 || abort "--agent-installer npm requested, but npm is not on PATH"
      printf 'npm\n'
      return
      ;;
    none)
      printf 'none\n'
      return
      ;;
  esac

  if [[ "${platform}" == "Darwin" && "$(command -v brew || true)" ]]; then
    printf 'brew\n'
  elif command -v npm >/dev/null 2>&1; then
    printf 'npm\n'
  elif command -v brew >/dev/null 2>&1; then
    printf 'brew\n'
  else
    abort "Cannot install missing agent CLIs: neither brew nor npm is on PATH. Use --no-install-agent to skip."
  fi
}

install_agent_cli() {
  local installer

  if command -v codex >/dev/null 2>&1; then
    printf "agent cli      codex (%s)\n" "$(command -v codex)" >&2
    return
  fi

  installer="$(choose_agent_installer)"
  if [[ "${installer}" == "none" ]]; then
    warn "Skipping missing codex CLI because --agent-installer none was selected."
    return
  fi

  ohai "Installing missing Codex CLI"
  case "${installer}" in
    brew)
      run_cmd brew install --cask "${PRAXIST_INSTALL_CODEX_BREW}"
      ;;
    npm)
      run_cmd npm install -g "${PRAXIST_INSTALL_CODEX_NPM}"
      ;;
  esac

  if [[ "${dry_run}" -eq 0 && "$(command -v codex || true)" == "" ]]; then
    abort "Installed Codex, but codex is still not on PATH. Check your brew/npm global bin path."
  fi
}

install_missing_agent_clis() {
  case "${install_agent}" in
    none) return ;;
    codex) install_agent_cli ;;
  esac
}

install_python_packages() {
  if [[ "${selected_method}" == "uv" && "$(command -v uv || true)" ]]; then
    run_install_cmd uv pip install --python "${venv_dir}/bin/python" --upgrade "$@"
  else
    run_install_cmd "${venv_dir}/bin/python" -m pip install --upgrade "$@"
  fi
}

mark_managed_venv() {
  if [[ "${venv_created}" -ne 1 && "${venv_dir}" != "${default_venv_dir}" ]]; then
    return
  fi
  local marker="${venv_dir}/.praxist-managed-venv"
  if [[ "${dry_run}" -eq 1 ]]; then
    printf '+ write Praxist virtualenv ownership marker %q\n' "${marker}"
    return
  fi
  printf 'managed_by=praxist\n' >"${marker}"
}

codex_packages_required() {
  if [[ "${setup_wizard}" -eq 1 ]]; then
    return 0
  fi
  if [[ "${agent_system}" == "codex_sdk" ]]; then
    return 0
  fi
  case "${req}" in
    *"["*"]"*)
      local extras="${req#*[}"
      extras="${extras%%]*}"
      case ",${extras}," in
        *,codex,*) return 0 ;;
      esac
      ;;
  esac
  return 1
}

install_codex_packages_from_public_index() {
  if ! codex_packages_required; then
    return
  fi
  ohai "Installing codex-relay from public PyPI"
  if [[ "${selected_method}" == "uv" && "$(command -v uv || true)" ]]; then
    run_install_cmd uv pip install \
      --python "${venv_dir}/bin/python" \
      --index-url "${PRAXIST_INSTALL_CODEX_INDEX_URL}" \
      "${PRAXIST_INSTALL_CODEX_RELAY_PACKAGE}"
    ohai "Installing Codex SDK from public PyPI"
    run_install_cmd uv pip install \
      --python "${venv_dir}/bin/python" \
      --index-url "${PRAXIST_INSTALL_CODEX_INDEX_URL}" \
      "${PRAXIST_INSTALL_CODEX_SDK_PACKAGE}"
  else
    run_install_cmd "${venv_dir}/bin/python" -m pip install \
      --index-url "${PRAXIST_INSTALL_CODEX_INDEX_URL}" \
      "${PRAXIST_INSTALL_CODEX_RELAY_PACKAGE}"
    ohai "Installing Codex SDK from public PyPI"
    run_install_cmd "${venv_dir}/bin/python" -m pip install \
      --index-url "${PRAXIST_INSTALL_CODEX_INDEX_URL}" \
      "${PRAXIST_INSTALL_CODEX_SDK_PACKAGE}"
  fi
}

install_codex_sdk_if_needed() {
  if [[ "${agent_system}" != "codex_sdk" ]]; then
    return
  fi
  ohai "Installing Codex SDK runtime dependencies for codex_sdk"
  install_python_packages \
    "${PRAXIST_INSTALL_CLAUDE_SDK_PACKAGE}" \
    "${PRAXIST_INSTALL_MCP_PACKAGE}"
}

install_claude_sdk_if_needed() {
  if [[ "${agent_system}" != "claude_sdk" ]]; then
    return
  fi
  ohai "Installing Claude Agent SDK for claude_sdk"
  install_python_packages "${PRAXIST_INSTALL_CLAUDE_SDK_PACKAGE}" "${PRAXIST_INSTALL_MCP_PACKAGE}"
}

install_skills_for_target() {
  local target="$1"
  run_cmd "${praxist_bin}" install-skills \
    --target "${target}" \
    --replace
}

install_requested_skills() {
  case "${install_skills}" in
    none) return ;;
    codex) install_skills_for_target codex ;;
    claude) install_skills_for_target claude ;;
  esac
}

ensure_codex_sdk() {
  if command -v codex >/dev/null 2>&1; then
    printf "codex cli      %s\n" "$(command -v codex)" >&2
    return
  fi
  abort "Codex CLI is required for Praxist provisioning. Install/login to codex first, then rerun this installer."
}

codex_provision_prompt() {
  if [[ -n "${PRAXIST_INSTALL_CODEX_PROMPT}" ]]; then
    printf '%s\n' "${PRAXIST_INSTALL_CODEX_PROMPT}"
    return
  fi
  local api_key_note="No provider API key env var was supplied to the installer."
  if [[ -n "${api_key_env}" ]]; then
    api_key_note="Provider API key env var supplied by name: ${api_key_env}. Do not print its value."
  fi
  cat <<EOF
Use \$praxist-runtime-install to finish provisioning this Praxist install.

Context:
- Praxist CLI: ${praxist_bin}
- Praxist venv: ${venv_dir}
- install directory for user binaries: ${bin_dir}
- provider: ${provider}
- model: ${model}
- agent system: ${agent_system}
- project env file: ${project_env_file:-none}
- product-usage client: built into Praxist
- ${api_key_note}

Verify the Praxist CLI, Python runtime extras, Codex skill registration, and user-level Praxist provider config. Before runtime setup, inspect `${praxist_bin} user-agreement status --json`; if the current version is not accepted, follow the packaged Agent OOBE runbook and record acceptance only after the operator explicitly reviews and agrees to the Fair Source License and User Agreement. Legal acceptance is not optional product-usage consent. Then inspect `${praxist_bin} product-usage status --json` and preserve or request that separate choice as the runbook specifies. Keep task dependencies separate and do not start a Praxist run unless the user explicitly asks.

Immediately after package installation, run `${praxist_bin} setup --agent-managed` and follow its `next_required_action`. Do not infer a runtime profile from saved credentials, environment variables, provider defaults, or a successful doctor report. Do not report OOBE completion until the operator has explicitly selected a profile and `setup_decisions_complete` is true.
EOF
}

offer_product_usage_consent() {
  if [[ "${skip_setup}" -ne 0 ]]; then
    return
  fi
  if [[ "${setup_wizard}" -eq 1 ]]; then
    printf "product usage  handled by the interactive setup wizard\n" >&2
    return
  fi
  if [[ "${start_codex}" -eq 1 ]]; then
    printf "product usage  delegated to Codex explicit-consent flow\n" >&2
    return
  fi
  if [[ "${interactive_setup}" -eq 1 && "${dry_run}" -eq 0 ]]; then
    run_cmd "${praxist_bin}" product-usage consent
    return
  fi
  if [[ "${dry_run}" -eq 1 ]]; then
    printf "product usage  existing consent state preserved (not queried in dry run)\n" >&2
    return
  fi
  local status_json consent_status
  if status_json="$("${praxist_bin}" product-usage status --json 2>/dev/null)" \
    && consent_status="$(
      printf '%s' "$status_json" \
        | "${python_cmd}" -c 'import json,sys; print(json.load(sys.stdin)["status"])'
    )"; then
    printf "product usage  existing consent state preserved: %s\n" "$consent_status" >&2
    return
  fi
  printf "product usage  existing consent state preserved; status unavailable\n" >&2
}

run_first_takeover() {
  ohai "Starting Praxist first-project takeover"
  run_cmd "${praxist_bin}" --takeover --operator "${install_skills}"
}

run_codex_provisioner() {
  ensure_codex_sdk
  ohai "Launching Codex with Praxist Runtime Install"
  run_cmd codex -C "${PWD}" "$(codex_provision_prompt)"
}

show_documentation() {
  local docs_args=(docs)
  if [[ "${open_docs}" -eq 0 || ( "${setup_wizard}" -eq 1 && "${start_takeover}" -eq 1 ) ]]; then
    docs_args+=(--no-open)
  fi
  ohai "Praxist documentation"
  if ! run_cmd "${praxist_bin}" "${docs_args[@]}"; then
    warn "Could not open the documentation automatically. Run: ${praxist_bin} docs"
  fi
}

print_path_hint() {
  if [[ "${modify_path}" -eq 0 ]]; then
    return
  fi
  case ":${PATH}:" in
    *":${bin_dir}:"*) return ;;
  esac
  cat >&2 <<EOF

${APP_NAME} installed \`praxist\` into:
  ${bin_dir}

That directory is not currently on PATH. Add it to your shell profile, e.g.:
  export PATH="${bin_dir}:\$PATH"
EOF
}

platform="$(uname -s 2>/dev/null || printf unknown)"
case "${platform}" in
  Darwin|Linux) ;;
  *) abort "Unsupported platform: ${platform}. Praxist installer supports Linux and macOS." ;;
esac

python_cmd="$(choose_python)"
validate_python "${python_cmd}"
req="$(package_requirement)"
normalize_agent_system_alias

setup_config_file="${PRAXIST_CONFIG_FILE:-${XDG_CONFIG_HOME:-${HOME}/.config}/praxist/env}"
if [[ -s "${setup_config_file}" ]]; then
  existing_setup=1
fi

if [[ -n "${api_key_env}" && "${provider_explicit}" -ne 1 ]]; then
  abort "error: --api-key-env requires --provider so Praxist can map the provider key"
fi
if [[ "${skip_setup}" -eq 0 && "${interactive_setup}" -eq 1 && "${dry_run}" -eq 0 \
  && "${start_codex}" -eq 0 \
  && "${install_defaults_explicit}" -eq 0 && "${project_env_explicit}" -eq 0 \
  && -z "${provider}" && -z "${agent_system}" && -z "${model}" && -z "${api_key_env}" ]]; then
  setup_wizard=1
  if [[ "${start_takeover_explicit}" -eq 0 && "${install_skills}" != "none" \
    && "${existing_setup}" -eq 0 ]]; then
    start_takeover=1
  fi
fi
if [[ "${skip_setup}" -eq 0 && "${interactive_setup}" -eq 0 \
  && "${setup_inputs_explicit}" -eq 0 ]]; then
  skip_setup=1
fi
apply_setup_defaults

if [[ "${start_takeover}" -eq 1 && "${dry_run}" -eq 0 ]]; then
  if [[ "${install_skills}" == "claude" ]] && ! command -v claude >/dev/null 2>&1; then
    abort "Claude Code is required for guided takeover with --install-skills claude. Install and authenticate Claude Code, or finish setup with --no-start-takeover."
  fi
  if [[ "${install_skills}" == "codex" ]] && ! command -v codex >/dev/null 2>&1; then
    if [[ "${install_agent}" != "codex" ]]; then
      abort "Codex CLI is required for guided takeover. Install Codex, use --install-agent codex with an available installer, or finish setup with --no-start-takeover."
    fi
    takeover_agent_installer="$(choose_agent_installer)"
    if [[ "${takeover_agent_installer}" == "none" ]]; then
      abort "Codex CLI is required for guided takeover. Select brew or npm, or finish setup with --no-start-takeover."
    fi
  fi
fi
if [[ "${start_takeover}" -eq 1 && "${dry_run}" -eq 0 \
  && "${interactive_setup}" -ne 1 ]]; then
  abort "Guided takeover requires a local interactive terminal; finish setup with --no-start-takeover and run praxist --takeover later."
fi

ohai "Installing ${APP_NAME}"
printf "platform       %s\n" "${platform}" >&2
printf "python         %s\n" "$("${python_cmd}" -c 'import sys; print(sys.executable)')" >&2
printf "venv           %s\n" "${venv_dir}" >&2
printf "bin dir        %s\n" "${bin_dir}" >&2
printf "package        %s\n" "${req}" >&2
printf "agreement      explicit acceptance requested during first-use setup\n" >&2
printf "product usage  privacy choice offered when collection is available\n" >&2
printf "agent install  %s via %s\n" "${install_agent}" "${agent_installer}" >&2
if codex_packages_required; then
  printf "download       Codex-native support adds roughly 100-150 MB; first install may take several minutes\n" >&2
else
  printf "download       first install fetches runtime packages and may take several minutes\n" >&2
fi

selected_method="${method}"
if [[ "${selected_method}" == "auto" ]]; then
  if command -v uv >/dev/null 2>&1; then
    selected_method="uv"
  else
    selected_method="pip"
  fi
fi

if [[ "${selected_method}" == "uv" ]]; then
  command -v uv >/dev/null 2>&1 || abort "--method uv requested, but uv is not on PATH"
  if [[ ! -x "${venv_dir}/bin/python" ]]; then
    run_cmd uv venv --python "${python_cmd}" "${venv_dir}"
    venv_created=1
  fi
  install_codex_packages_from_public_index
  run_install_cmd uv pip install --python "${venv_dir}/bin/python" --upgrade "${req}"
else
  if [[ ! -x "${venv_dir}/bin/python" ]]; then
    run_cmd "${python_cmd}" -m venv "${venv_dir}"
    venv_created=1
  fi
  run_install_cmd "${venv_dir}/bin/python" -m pip install --upgrade pip
  install_codex_packages_from_public_index
  run_install_cmd "${venv_dir}/bin/python" -m pip install --upgrade "${req}"
fi

mark_managed_venv

run_cmd mkdir -p "${bin_dir}"
if [[ "${dry_run}" -eq 0 ]]; then
  ln -sfn "${venv_dir}/bin/praxist" "${bin_dir}/praxist"
  if [[ -x "${venv_dir}/bin/praxist-uninstall" ]]; then
    ln -sfn "${venv_dir}/bin/praxist-uninstall" "${bin_dir}/praxist-uninstall"
  else
    warn "Installed package does not provide praxist-uninstall; keep praxist-uninstall.sh from the release bundle."
  fi
else
  printf '+ ln -sfn %q %q\n' "${venv_dir}/bin/praxist" "${bin_dir}/praxist"
  printf '+ ln -sfn %q %q\n' \
    "${venv_dir}/bin/praxist-uninstall" "${bin_dir}/praxist-uninstall"
fi

praxist_bin="${bin_dir}/praxist"
if [[ "${dry_run}" -eq 0 && ! -x "${praxist_bin}" ]]; then
  abort "installed praxist is not executable: ${praxist_bin}"
fi

install_missing_agent_clis
install_codex_sdk_if_needed
install_claude_sdk_if_needed

ohai "Preparing writable Praxist examples"
run_cmd "${praxist_bin}" examples install rocket_booster_recovery
run_cmd "${praxist_bin}" examples install rocket_booster_recovery_rust

if [[ "${skip_setup}" -eq 0 ]]; then
  if [[ "${setup_wizard}" -eq 1 ]]; then
    setup_args=(setup --interactive --install-skills "${install_skills}")
    if [[ "${skip_doctor}" -eq 1 ]]; then
      setup_args+=(--skip-doctor)
    fi
    run_cmd "${praxist_bin}" "${setup_args[@]}"
  elif [[ -n "${provider}" ]]; then
    configure_args=(configure-llm --provider "${provider}")
    if [[ -n "${project_env_file}" ]]; then
      configure_args+=(--project-env-file "${project_env_file}")
    else
      configure_args+=(--no-project-env)
    fi
    if [[ -n "${agent_system}" ]]; then
      configure_args+=(--agent-system "${agent_system}")
    fi
    if [[ -n "${model}" ]]; then
      configure_args+=(--model "${model}")
    fi
    if [[ -n "${api_key_env}" ]]; then
      configure_args+=(--api-key-env "${api_key_env}")
    elif [[ "${provider}" == "openai" && "${agent_system}" == "codex_sdk" ]]; then
      configure_args+=(--no-api-key)
    elif [[ "${interactive_setup}" -eq 1 && "${dry_run}" -eq 0 ]]; then
      configure_args+=(--api-key-stdin)
    else
      configure_args+=(--no-api-key)
    fi
    run_cmd "${praxist_bin}" "${configure_args[@]}"
  fi
fi

print_path_hint

if [[ "${dry_run}" -eq 1 || -x "${bin_dir}/praxist-uninstall" ]]; then
  printf "uninstall      %s\n" "${bin_dir}/praxist-uninstall" >&2
fi

if [[ "${skip_setup}" -eq 0 && "${setup_wizard}" -ne 1 ]]; then
  install_requested_skills
fi

if [[ "${skip_setup}" -eq 0 && "${skip_doctor}" -eq 0 && "${setup_wizard}" -ne 1 ]]; then
  doctor_args=(doctor --agent-system "${agent_system}")
  if [[ "${install_skills}" != "none" ]]; then
    doctor_args+=(--target "${install_skills}")
  fi
  run_cmd "${praxist_bin}" "${doctor_args[@]}"
fi

offer_product_usage_consent

if [[ "${skip_setup}" -eq 1 ]]; then
  printf "OOBE checkpoint package installed; License and User Agreement, runtime profile, skills, and research project are not completed\n" >&2
  printf "agent next      %s setup --agent-managed\n" "${praxist_bin}" >&2
  printf "local next      %s setup --interactive\n" "${praxist_bin}" >&2
  printf "completion      do not report OOBE complete until setup_decisions_complete is true\n" >&2
fi

show_documentation

if [[ "${skip_setup}" -eq 0 && "${setup_wizard}" -eq 1 && "${start_takeover}" -eq 0 \
  && "${install_skills}" != "none" ]]; then
  printf "next step       choose a research project with: %s --takeover --operator %s\n" "${praxist_bin}" "${install_skills}" >&2
fi

if [[ "${start_codex}" -eq 1 ]]; then
  run_codex_provisioner
fi

if [[ "${start_takeover}" -eq 1 ]]; then
  run_first_takeover
fi
