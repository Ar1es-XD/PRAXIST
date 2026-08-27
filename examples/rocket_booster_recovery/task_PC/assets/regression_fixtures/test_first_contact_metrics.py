#!/usr/bin/env python3
"""Boundary regression for the sole first-contact landing-success predicate."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


TASK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK_ROOT))

from evaluations.controller_ood.operational_metrics import (  # noqa: E402
    INITIAL_FUEL_KG,
    MASS_EMPTY_KG,
    first_contact_arrays,
    summarize,
)


def nominal_state(rows: int) -> np.ndarray:
    state = np.zeros((rows, 16), dtype=np.float64)
    state[:, 0] = 3.0
    state[:, 3] = -0.5
    state[:, 6] = 1.0
    state[:, 13] = MASS_EMPTY_KG + 0.0201 * INITIAL_FUEL_KG
    return state


def main() -> None:
    state = nominal_state(8)
    detected = np.ones(8, dtype=bool)
    leg_sink = np.full(8, 0.5)

    state[0, 1] = 5.0  # inclusive lateral boundary passes
    state[1, 13] = MASS_EMPTY_KG + 0.02 * INITIAL_FUEL_KG  # strict fuel >2% fails
    state[2, 3] = -1.0  # inclusive COM sink boundary passes
    state[3, 3] = -1.01
    leg_sink[4] = 1.0  # inclusive contacting-leg boundary passes
    leg_sink[5] = 1.01
    detected[6] = False
    state[7, 1] = 5.01

    arrays = first_contact_arrays(state, detected, leg_sink)
    passed = arrays["landing_success_pass"].tolist()
    assert passed == [True, False, True, False, True, False, False, False], passed
    assert "engineering_pass" not in arrays
    assert "strict_pass" not in arrays
    assert "standard_pass" not in arrays
    summary = summarize(arrays)
    assert summary["landing_success_pass"]["count"] == 3
    assert summary["single_success_gate_counts"]["fuel_reserve_gt_2pct"] == 6
    print("first-contact single-success regression: PASS")


if __name__ == "__main__":
    main()
