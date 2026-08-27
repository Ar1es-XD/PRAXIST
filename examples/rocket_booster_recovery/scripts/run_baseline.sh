#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"
task_name="${ROCKET_BOOSTER_RECOVERY_TASK_DIR:-task_GPU_server}"
task_dir="${repo_root}/${task_name}"
mode="${1:-canary}"
output_dir="${2:-${task_dir}/experiments/manual_${mode}}"
gpu_device="${CUDA_VISIBLE_DEVICES:-0}"

case "${mode}" in
  canary|development|roll_diagnostic|complete) ;;
  *)
    echo "mode must be canary, development, roll_diagnostic, or complete" >&2
    exit 2
    ;;
esac

if [[ ! -x "${python_bin}" ]]; then
  echo "Python environment not found: ${python_bin}" >&2
  echo "Create .venv and install requirements.txt first." >&2
  exit 2
fi

if [[ ! -f "${task_dir}/task.yaml" ]]; then
  echo "Task directory not found: ${task_dir}" >&2
  echo "Set ROCKET_BOOSTER_RECOVERY_TASK_DIR to task_GPU_server or task_PC." >&2
  exit 2
fi

cd "${task_dir}"
CUDA_VISIBLE_DEVICES="${gpu_device}" \
NVIDIA_VISIBLE_DEVICES="${gpu_device}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "${python_bin}" evaluations/controller_ood/run.py \
  --variant-dir assets/baseline \
  --mode "${mode}" \
  --output-dir "${output_dir}" \
  --batch-size 1024
