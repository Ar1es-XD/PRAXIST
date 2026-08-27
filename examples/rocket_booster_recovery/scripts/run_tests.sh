#!/usr/bin/env bash
set -euo pipefail

# Keep even lightweight verification from creating cache files in a project copy.
export PYTHONDONTWRITEBYTECODE=1

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"
task_name="${ROCKET_BOOSTER_RECOVERY_TASK_DIR:-task_GPU_server}"
task_dir="${repo_root}/${task_name}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python environment not found: ${python_bin}" >&2
  exit 2
fi

if [[ ! -f "${task_dir}/task.yaml" ]]; then
  echo "Task directory not found: ${task_dir}" >&2
  echo "Set ROCKET_BOOSTER_RECOVERY_TASK_DIR to task_GPU_server or task_PC." >&2
  exit 2
fi

cd "${repo_root}"
"${python_bin}" tests/test_repository_baseline_alignment.py
"${python_bin}" "${task_name}/assets/regression_fixtures/test_first_contact_metrics.py"
"${python_bin}" "${task_name}/assets/regression_fixtures/test_accelerator_binding.py"
"${python_bin}" "${task_name}/assets/regression_fixtures/test_effective_config.py"

if [[ "${RUN_GPU_INTEGRITY_TEST:-0}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "${python_bin}" "${task_name}/assets/regression_fixtures/test_evaluator_integrity.py"
fi
