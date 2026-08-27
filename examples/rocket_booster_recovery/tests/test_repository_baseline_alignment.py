#!/usr/bin/env python3
"""Regression checks for repository-wide baseline/protocol alignment."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import metrics as root_metrics
from task_GPU_server.evaluations.controller_ood import operational_metrics as gpu_metrics
from task_PC.evaluations.controller_ood import operational_metrics as pc_metrics


def nominal_state(rows: int) -> np.ndarray:
    state = np.zeros((rows, 16), dtype=np.float64)
    state[:, 0] = 3.0
    state[:, 3] = -0.5
    state[:, 6] = 1.0
    state[:, 13] = (
        root_metrics.MASS_EMPTY_KG + 0.0201 * root_metrics.INITIAL_FUEL_KG
    )
    return state


def main() -> None:
    state = nominal_state(10)
    detected = np.ones(10, dtype=bool)
    leg_sink = np.full(10, 0.5)

    state[0, 1] = 5.0
    state[1, 13] = (
        root_metrics.MASS_EMPTY_KG + 0.02 * root_metrics.INITIAL_FUEL_KG
    )
    state[2, 3] = -1.0
    state[3, 3] = -1.01
    leg_sink[4] = 1.0
    leg_sink[5] = 1.01
    detected[6] = False
    state[7, 1] = 5.01
    state[8, 3] = 0.001
    state[9, 4] = 0.301

    implementations = (root_metrics, gpu_metrics, pc_metrics)
    outputs = [
        module.first_contact_arrays(state, detected, leg_sink)
        for module in implementations
    ]
    expected = [True, False, True, False, True, False, False, False, False, False]
    assert outputs[0]["landing_success_pass"].tolist() == expected
    for output in outputs:
        assert "standard_pass" not in output
        assert "strict_pass" not in output
        assert "engineering_pass" not in output
    assert set(outputs[0]) == set(outputs[1]) == set(outputs[2])
    for name in outputs[0]:
        np.testing.assert_allclose(outputs[0][name], outputs[1][name])
        np.testing.assert_allclose(outputs[0][name], outputs[2][name])

    for config_path in (
        ROOT / "configs/rocket_booster_recovery_v0.json",
        ROOT / "configs/rocket_booster_recovery_v0_frozen_formal.json",
    ):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["initial_fuel_kg"] == 7000.0
        assert config["fuel_reserve_fraction"] == 0.02
        assert not any(key.startswith("gear_sink_") for key in config)
        assert config_path.read_bytes() == (
            ROOT / "task_GPU_server/assets/baseline/controller_config.json"
        ).read_bytes()

    assert (ROOT / "src/rocket_booster_recovery_controller.py").read_bytes() == (
        ROOT / "task_GPU_server/assets/baseline/controller.py"
    ).read_bytes()

    for task_name in ("task_GPU_server", "task_PC"):
        variant_path = ROOT / task_name / "assets/baseline/variant.json"
        variant = json.loads(variant_path.read_text(encoding="utf-8"))
        provenance = variant["provenance"]
        for path_key, hash_key in (
            ("source_controller", "source_controller_sha256"),
            ("source_config", "source_config_sha256"),
        ):
            source = (variant_path.parent / provenance[path_key]).resolve()
            source.relative_to(ROOT)
            assert source.is_file()
            assert hashlib.sha256(source.read_bytes()).hexdigest() == provenance[hash_key]
        assert provenance["copied_without_semantic_change"] is True
    print("repository baseline/single-success alignment: PASS")


if __name__ == "__main__":
    main()
