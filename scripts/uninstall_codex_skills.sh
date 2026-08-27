#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Remove Praxist-managed Codex or Claude Code skills from one target directory.

Usage:
  scripts/uninstall_codex_skills.sh [--target codex|claude] [--target-dir DIR] [--dry-run]

Defaults:
  target: codex
  Codex dir: ${CODEX_SKILLS_DIR:-$HOME/.agents/skills}
  Claude Code dir: ${CLAUDE_SKILLS_DIR:-$HOME/.claude/skills}

This compatibility wrapper delegates ownership checks, locking, removal, and
manifest updates to the Praxist CLI. It never removes unmanaged skills.
EOF
}

target="codex"
target_dir=""
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      [[ $# -ge 2 ]] || {
        echo "error: --target requires codex or claude" >&2
        exit 2
      }
      target="$2"
      shift 2
      ;;
    --target-dir)
      [[ $# -ge 2 ]] || {
        echo "error: --target-dir requires a path" >&2
        exit 2
      }
      target_dir="$2"
      shift 2
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
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${target}" in
  codex) target_dir="${target_dir:-${CODEX_SKILLS_DIR:-${HOME}/.agents/skills}}" ;;
  claude) target_dir="${target_dir:-${CLAUDE_SKILLS_DIR:-${HOME}/.claude/skills}}" ;;
  *) echo "error: --target must be codex or claude" >&2; exit 2 ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
python_bin="${PYTHON:-}"
if [[ -z "${python_bin}" && -x "${repo_root}/.venv/bin/python" ]]; then
  python_bin="${repo_root}/.venv/bin/python"
fi
if [[ -z "${python_bin}" ]]; then
  python_bin="$(command -v python3 || command -v python || true)"
fi
if [[ -z "${python_bin}" ]]; then
  echo "error: python3 or python is required to uninstall Praxist skills" >&2
  exit 1
fi

args=(
  -m praxist uninstall-skills
  --target "${target}"
  --target-dir "${target_dir}"
)
if [[ "${dry_run}" -eq 1 ]]; then
  args+=(--dry-run)
fi

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${python_bin}" "${args[@]}"
