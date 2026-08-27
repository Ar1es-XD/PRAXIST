#!/usr/bin/env python3
"""Task-local accelerator handoff contract tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK_ROOT))
from assets.harness.accelerator_binding import apply_scheduler_assignment


def main() -> None:
    one = {"PRAXIST_ASSIGNED_GPU_UUIDS": "GPU-a"}
    assert apply_scheduler_assignment(one) == "GPU-a"
    assert one["CUDA_VISIBLE_DEVICES"] == one["NVIDIA_VISIBLE_DEVICES"] == "GPU-a"

    multi = {"PRAXIST_ASSIGNED_GPU_UUIDS": "GPU-a,GPU-b"}
    assert apply_scheduler_assignment(multi) == "GPU-a,GPU-b"
    assert multi["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b"

    missing = {"PRAXIST_ASSIGNED_GPU_UUIDS": "MIG-GPU-a/1/0"}
    apply_scheduler_assignment(missing)
    assert missing["NVIDIA_VISIBLE_DEVICES"] == "MIG-GPU-a/1/0"

    try:
        apply_scheduler_assignment(
            {"PRAXIST_ASSIGNED_GPU_UUIDS": "GPU-a", "CUDA_VISIBLE_DEVICES": "0"}
        )
    except RuntimeError as exc:
        assert "accelerator binding mismatch" in str(exc)
    else:
        raise AssertionError("conflicting ordinal mask was accepted")

    standalone = {"CUDA_VISIBLE_DEVICES": "2"}
    assert apply_scheduler_assignment(standalone) == ""
    assert standalone["CUDA_VISIBLE_DEVICES"] == "2"

    cpu = {
        "PRAXIST_ASSIGNED_GPU_UUIDS": "",
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "",
    }
    assert apply_scheduler_assignment(cpu) == ""
    assert cpu["CUDA_VISIBLE_DEVICES"] == ""

    # Evaluator-to-worker descendants inherit the exact UUID strings.
    child_env = os.environ.copy()
    child_env.update(multi)
    payload = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import json,os;print(json.dumps({k:os.environ.get(k) for k in "
            "['PRAXIST_ASSIGNED_GPU_UUIDS','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES']}))",
        ],
        env=child_env,
        text=True,
    )
    observed = json.loads(payload)
    assert len(set(observed.values())) == 1
    assert observed["PRAXIST_ASSIGNED_GPU_UUIDS"] == "GPU-a,GPU-b"
    print("accelerator binding regression: PASS (7 contracts)")


if __name__ == "__main__":
    main()
