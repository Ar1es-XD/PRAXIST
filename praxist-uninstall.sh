#!/usr/bin/env bash
# Locate the installed Praxist CLI and delegate all uninstall semantics to it.

set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  printf 'Praxist uninstall requires Bash.\n' >&2
  exit 1
fi

cli="${PRAXIST_UNINSTALL_CLI:-}"
if [[ -z "${cli}" ]]; then
  cli="$(command -v praxist || true)"
fi
if [[ -z "${cli}" ]]; then
  for candidate in \
    "${XDG_BIN_HOME:-${HOME}/.local/bin}/praxist" \
    "${XDG_DATA_HOME:-${HOME}/.local/share}/praxist/venv/bin/praxist"; do
    if [[ -x "${candidate}" ]]; then
      cli="${candidate}"
      break
    fi
  done
fi

if [[ -n "${cli}" && -x "${cli}" ]]; then
  exec "${cli}" uninstall "$@"
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -n "${python_bin}" && -f "${script_dir}/praxist/__main__.py" ]]; then
  export PYTHONPATH="${script_dir}${PYTHONPATH:+:${PYTHONPATH}}"
  exec "${python_bin}" -m praxist uninstall "$@"
fi

printf '%s\n' \
  'Praxist CLI was not found.' \
  'Run this script from a Praxist source checkout/release, or set PRAXIST_UNINSTALL_CLI.' >&2
exit 1
