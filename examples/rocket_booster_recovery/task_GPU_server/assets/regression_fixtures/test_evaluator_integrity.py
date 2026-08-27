#!/usr/bin/env python3
"""Independent evaluator static, attestation, and channel-lock regressions."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK_ROOT))

from evaluations.controller_ood import evaluator


def manifest() -> dict:
    return {
        "variant_id": "intentional_bad_channel_fixture",
        "method_class": "deterministic_classical_control",
        "changed_modules": ["roll_rcs_controller"],
        "design_dimensions": {
            "mechanism_family": "contract_falsifier",
            "intervention_surface": "roll_rcs_controller",
            "intent": "falsify",
            "semantic_family": "forbidden_channel_fixture",
            "parent_lineage": "frozen_baseline",
            "novelty_axis": "intentional_rcs_y_violation",
        },
    }


def main() -> None:
    # Frozen hash mismatch must fail closed.
    key = next(iter(evaluator.FROZEN_HASHES))
    original = evaluator.FROZEN_HASHES[key]
    evaluator.FROZEN_HASHES[key] = "0" * 64
    try:
        try:
            evaluator._attest_frozen_assets()
        except evaluator.EvaluationError as exc:
            assert "hash mismatch" in str(exc)
        else:
            raise AssertionError("frozen hash mismatch was accepted")
    finally:
        evaluator.FROZEN_HASHES[key] = original

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        static = root / "static"
        static.mkdir()
        (static / "bad.py").write_text("import torch\n", encoding="utf-8")
        try:
            evaluator._scan_candidate_tree(static)
        except evaluator.EvaluationError as exc:
            assert "forbidden imports" in str(exc)
        else:
            raise AssertionError("neural import was accepted")

        variant = root / "variant"
        variant.mkdir()
        shutil.copyfile(TASK_ROOT / "assets/baseline/controller.py", variant / "controller_base.py")
        shutil.copyfile(
            TASK_ROOT / "assets/baseline/controller_config.json",
            variant / "controller_config.json",
        )
        (variant / "variant.json").write_text(
            json.dumps(manifest(), indent=2) + "\n", encoding="utf-8"
        )
        (variant / "controller.py").write_text(
            "from controller_base import *\n"
            "from controller_base import control_step as _base_control_step\n"
            "def control_step(state, memory, cfg):\n"
            "    action, next_memory, diagnostic = _base_control_step(state, memory, cfg)\n"
            "    return action.at[4].set(0.25), next_memory, diagnostic\n",
            encoding="utf-8",
        )
        out = root / "out"
        args = argparse.Namespace(
            variant_dir=variant,
            mode="canary",
            out_dir=out,
            batch_size=1024,
        )
        summary, _ = evaluator.evaluate(args)
        assert summary["metrics"]["forbidden_action_max_abs"] == 0.25
        assert not summary["metrics"]["contract_lock_passed"]
        assert not summary["metrics"]["protocol_integrity_passed"]
        assert summary["frontier_lane"] == summary["promotion_lane"] == "diagnostic"
    print("evaluator integrity regression: PASS (hash, neural import, forbidden channel)")


if __name__ == "__main__":
    main()
