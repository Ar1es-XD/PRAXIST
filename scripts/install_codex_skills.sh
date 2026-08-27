#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Install repository Praxist skills for Codex or Claude Code as managed symlinks.

Usage:
  scripts/install_codex_skills.sh [--target codex|claude] [--target-dir DIR]
                                  [--migrate-legacy-symlinks]

Defaults:
  target: codex
  Codex dir: ${CODEX_SKILLS_DIR:-$HOME/.agents/skills}
  Claude Code dir: ${CLAUDE_SKILLS_DIR:-$HOME/.claude/skills}

This compatibility wrapper delegates ownership, locking, replacement, and
manifest updates to the Praxist CLI so every installation path follows the
same safety protocol.
EOF
}

target="codex"
target_dir=""
migrate_legacy=0

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
    --migrate-legacy-symlinks)
      migrate_legacy=1
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
  echo "error: python3 or python is required to install Praxist skills" >&2
  exit 1
fi

args=(
  -m praxist install-skills
  --target "${target}"
  --target-dir "${target_dir}"
  --mode symlink
  --replace
)
if [[ "${migrate_legacy}" -eq 1 ]]; then
  args+=(--migrate-legacy-symlinks)
fi

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
"${python_bin}" "${args[@]}"
while IFS= read -r -d '' skill_file; do
  skill_name="$(basename -- "$(dirname -- "${skill_file}")")"
  if [[ "${target}" == "claude" ]]; then
    printf '/%s\n' "${skill_name}"
  else
    printf '$%s\n' "${skill_name}"
  fi
done < <(find "${repo_root}/skills" -mindepth 2 -maxdepth 2 -name SKILL.md -print0 | sort -z)
