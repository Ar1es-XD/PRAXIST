#!/usr/bin/env bash
# Task-specific token leakage audit (framework boundary gate).
#
# Praxist is a task-agnostic research framework. Task-specific
# tokens (concrete task names, custom optimizer terms, dataset
# enumerations, etc.) MUST NOT appear in any `praxist/` or
# `tests/` source file — they belong inside the external task project
# (see AGENTS.md §5).
#
# This script greps for known-bad tokens. A match fails the build.
#
# How to extend:
#   - Edit the `TOKENS` array below to add patterns relevant to your
#     dogfood task. Patterns are POSIX extended regex; word-boundary
#     anchors (`\b`) are recommended for short tokens.
#   - For one-off, intentional, documented occurrences (e.g. a
#     docstring saying "Praxist does not parse <task-token> handles"),
#     append `# noqa: leakage_audit` to that line.
#
# How to override for tests:
#   - Set the `LEAKAGE_TOKENS` env var to a single pipe-delimited
#     regex; this replaces the built-in TOKENS array entirely.
#   - Set the `LEAKAGE_AUDIT_ROOT` env var to point the scanner at a
#     different repo root (mainly used by the test suite to isolate
#     against a fixture tree).

set -euo pipefail

ROOT="${LEAKAGE_AUDIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"

if [[ -n "${LEAKAGE_TOKENS:-}" ]]; then
  PATTERN="$LEAKAGE_TOKENS"
else
  # Default blacklist. Replace these with tokens relevant to your active
  # dogfood task. The names below are examples from a prior internal
  # dogfood run and should be customized; leaving them in place is
  # harmless when your task does not use those words.
  TOKENS=(
    '\bvc\.[a-z]'
    '\btorsion\b'
    '\bMobius\b'
    '\bpole_section\b'
    '\bmujoco\b'
    '\bfalcon_ppo\b'
    '\brocket_env\b'
    '\bgimbal\b'
    '\bsepnet\b'
    '\bvf_coef\b'
    '\bclip_eps\b'
    '\bent_coef\b'
    '\bProtocol [A-G]\b'
  )
  PATTERN="$(IFS='|'; echo "${TOKENS[*]}")"
fi

# Scan only system code. Task-local code under `templates/tasks/**` and
# `tasks/**` is allowed to contain task-specific vocabulary.
MATCHES=$(grep -REn --include='*.py' --include='*.sh' "$PATTERN" praxist/ tests/ 2>/dev/null || true)

# Lines that explicitly opt out via "# noqa: leakage_audit" are exempt.
FILTERED=$(printf '%s\n' "$MATCHES" | grep -v "# noqa: leakage_audit" || true)
# `grep -v` with an empty input still prints a blank line; normalize.
FILTERED="$(printf '%s' "$FILTERED" | sed '/^$/d')"

if [[ -n "$FILTERED" ]]; then
  echo "ERROR: task-specific token(s) leaked into praxist/ or tests/:" >&2
  echo "$FILTERED" >&2
  echo >&2
  echo "If a match is documented (e.g. a docstring stating 'Praxist does not" >&2
  echo "parse <task-token> handles'), append '# noqa: leakage_audit' to" >&2
  echo "that line." >&2
  exit 1
fi

echo "✓ leakage audit clean (no task-specific tokens in praxist/ or tests/)"
