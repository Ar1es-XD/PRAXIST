"""Single operational landing-success predicate evaluated at first leg contact.

No post-contact spring or damper response is used.  Every stratified success
rate in the task is a view of the same ``landing_success_pass`` array.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np


MASS_EMPTY_KG = 22_200.0
INITIAL_FUEL_KG = 7_000.0
INITIAL_MASS_KG = MASS_EMPTY_KG + INITIAL_FUEL_KG

SUCCESS_THRESHOLDS = {
    "lateral_error_max_m": 5.0,
    "com_sink_speed_max_mps": 1.0,
    "contact_leg_sink_speed_max_mps": 1.0,
    "lateral_speed_max_mps": 0.3,
    "tilt_max_deg": 1.5,
    "roll_rate_max_radps": 0.02,
    "pitch_yaw_rate_max_radps": 0.03,
    "fuel_reserve_min_fraction": 0.02,
}

# A non-contact trajectory must not appear artificially attractive on the
# first-contact speed risk axis.  The sentinel is only an aggregation penalty;
# it is not a fabricated physical measurement.
NO_CONTACT_SINK_PENALTY_MPS = 100.0


def _tilt_from_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    q0, q1, q2, q3 = q.T
    r00 = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
    return np.arccos(np.clip(r00, -1.0, 1.0))


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / den
    radius = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) / den
    return max(0.0, center - radius), min(1.0, center + radius)


def first_contact_arrays(
    endpoint_state: np.ndarray,
    first_contact_detected: np.ndarray,
    first_contact_leg_sink_speed_mps: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return diagnostics and the task's only landing-success predicate.

    ``endpoint_state`` is the interpolated first-contact state when contact was
    detected, otherwise the terminal failure state.  The latter remains useful
    for diagnostics but can never pass because ``first_contact_detected`` is a
    required conjunct.
    """

    state = np.asarray(endpoint_state, dtype=np.float64)
    detected = np.asarray(first_contact_detected, dtype=bool)
    leg_sink = np.asarray(first_contact_leg_sink_speed_mps, dtype=np.float64)

    pos = state[:, 0:3]
    vel = state[:, 3:6]
    omega = state[:, 10:13]
    lateral = np.linalg.norm(pos[:, 1:3], axis=1)
    lateral_speed = np.linalg.norm(vel[:, 1:3], axis=1)
    total_speed = np.linalg.norm(vel, axis=1)
    tilt = _tilt_from_quaternion(state[:, 6:10])
    pitch_yaw_rate = np.linalg.norm(omega[:, 1:3], axis=1)
    fuel_fraction = (state[:, 13] - MASS_EMPTY_KG) / INITIAL_FUEL_KG
    com_sink_speed = np.maximum(-vel[:, 0], 0.0)
    finite = (
        np.all(np.isfinite(state), axis=1)
        & np.isfinite(leg_sink)
        & detected
    )

    t = SUCCESS_THRESHOLDS
    success = (
        detected
        & (lateral <= t["lateral_error_max_m"])
        & (vel[:, 0] >= -t["com_sink_speed_max_mps"])
        & (vel[:, 0] <= 0.0)
        & (leg_sink <= t["contact_leg_sink_speed_max_mps"])
        & (lateral_speed <= t["lateral_speed_max_mps"])
        & (tilt <= np.deg2rad(t["tilt_max_deg"]))
        & (np.abs(omega[:, 0]) <= t["roll_rate_max_radps"])
        & (pitch_yaw_rate <= t["pitch_yaw_rate_max_radps"])
        & (fuel_fraction > t["fuel_reserve_min_fraction"])
        & finite
    )

    return {
        "first_contact_detected": detected,
        "landing_success_pass": success,
        "finite_first_contact": finite,
        "lateral_error_m": lateral,
        "vertical_velocity_mps": vel[:, 0],
        "com_sink_speed_mps": com_sink_speed,
        "contact_leg_sink_speed_mps": leg_sink,
        "lateral_speed_mps": lateral_speed,
        "total_speed_mps": total_speed,
        "tilt_rad": tilt,
        "roll_rate_radps": omega[:, 0],
        "pitch_yaw_rate_radps": pitch_yaw_rate,
        "fuel_fraction": fuel_fraction,
        "fuel_gate_pass": detected
        & finite
        & (fuel_fraction > t["fuel_reserve_min_fraction"]),
        "vertical_gate_pass": detected
        & finite
        & (vel[:, 0] >= -t["com_sink_speed_max_mps"])
        & (vel[:, 0] <= 0.0)
        & (leg_sink <= t["contact_leg_sink_speed_max_mps"]),
    }


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def summarize(
    arrays: Mapping[str, np.ndarray], mask: np.ndarray | None = None
) -> dict[str, object]:
    n_all = len(arrays["landing_success_pass"])
    if mask is None:
        mask = np.ones(n_all, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return {"trajectories": 0}

    out: dict[str, object] = {"trajectories": n}
    for name in (
        "first_contact_detected",
        "landing_success_pass",
        "finite_first_contact",
        "fuel_gate_pass",
        "vertical_gate_pass",
    ):
        count = int(np.count_nonzero(np.asarray(arrays[name])[mask]))
        low, high = wilson_interval(count, n)
        out[name] = {
            "count": count,
            "rate": count / n,
            "wilson_95_low": low,
            "wilson_95_high": high,
        }

    for name in (
        "lateral_error_m",
        "vertical_velocity_mps",
        "com_sink_speed_mps",
        "contact_leg_sink_speed_mps",
        "lateral_speed_mps",
        "total_speed_mps",
        "tilt_rad",
        "fuel_fraction",
    ):
        out[name] = _stats(np.asarray(arrays[name])[mask])

    t = SUCCESS_THRESHOLDS
    out["single_success_gate_counts"] = {
        "first_contact": int(np.count_nonzero(arrays["first_contact_detected"][mask])),
        "lateral_le_5m": int(
            np.count_nonzero(arrays["lateral_error_m"][mask] <= t["lateral_error_max_m"])
        ),
        "vertical_first_contact_gate": int(
            np.count_nonzero(arrays["vertical_gate_pass"][mask])
        ),
        "fuel_reserve_gt_2pct": int(np.count_nonzero(arrays["fuel_gate_pass"][mask])),
        "landing_success_joint": int(
            np.count_nonzero(arrays["landing_success_pass"][mask])
        ),
    }
    return out
