#!/usr/bin/env python3
"""Independently recompute the sole first-contact landing predicate.

This file intentionally does not import ``evaluate.py`` or ``metrics.py``.
It is an audit implementation for v2 result archives, not a compatibility
reader for pre-v2 terminal-state archives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


MASS_EMPTY_KG = 22_200.0
INITIAL_FUEL_KG = 7_000.0
THRESHOLDS = {
    "lateral_error_max_m": 5.0,
    "com_sink_speed_max_mps": 1.0,
    "contact_leg_sink_speed_max_mps": 1.0,
    "lateral_speed_max_mps": 0.3,
    "tilt_max_deg": 1.5,
    "roll_rate_max_radps": 0.02,
    "pitch_yaw_rate_max_radps": 0.03,
    "fuel_reserve_min_fraction": 0.02,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(
        p * (1.0 - p) / n + z * z / (4.0 * n * n)
    ) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def counts(mask: np.ndarray, source: np.ndarray, source_names: list[str]) -> dict:
    out = {}
    groups = [(-1, "overall"), *enumerate(source_names)]
    for source_id, name in groups:
        take = np.ones(len(mask), bool) if source_id < 0 else source == source_id
        n = int(take.sum())
        k = int(np.count_nonzero(mask[take]))
        out[name] = {
            "count": k,
            "total": n,
            "rate": k / n if n else None,
            "wilson_95": wilson(k, n) if n else None,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_npz", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scalar = (
        json.loads(args.summary.read_text(encoding="utf-8"))
        if args.summary
        else None
    )

    required = {
        "terminal_state",
        "first_contact_detected",
        "first_contact_leg_sink_speed_mps",
        "landing_success_pass",
        "source_id",
        "max_abs_action",
        "audit",
        "audit_columns",
    }
    with np.load(args.result_npz, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise SystemExit(
                "result archive predates the first-contact v2 protocol or is incomplete; "
                f"missing fields: {', '.join(missing)}"
            )
        state = np.asarray(archive["terminal_state"], np.float64)
        detected = np.asarray(archive["first_contact_detected"], bool)
        leg_sink = np.asarray(
            archive["first_contact_leg_sink_speed_mps"], np.float64
        )
        stored_success = np.asarray(archive["landing_success_pass"], bool)
        source = np.asarray(archive["source_id"], np.int8)
        max_action = np.asarray(archive["max_abs_action"], np.float64)
        audit = np.asarray(archive["audit"], np.float64)
        audit_columns = np.asarray(archive["audit_columns"]).astype(str).tolist()

    if scalar is not None:
        source_names = [
            str(name) for name in scalar["dataset"].get("source_names", [])
        ]
    else:
        source_names = []
    required_names = int(np.max(source)) + 1 if len(source) else 0
    source_names.extend(
        f"source_{i}" for i in range(len(source_names), required_names)
    )

    pos = state[:, 0:3]
    vel = state[:, 3:6]
    quat = state[:, 6:10]
    quat = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12)
    r00 = (
        quat[:, 0] ** 2
        + quat[:, 1] ** 2
        - quat[:, 2] ** 2
        - quat[:, 3] ** 2
    )
    tilt = np.arccos(np.clip(r00, -1.0, 1.0))
    omega = state[:, 10:13]
    lateral = np.linalg.norm(pos[:, 1:3], axis=1)
    lateral_speed = np.linalg.norm(vel[:, 1:3], axis=1)
    pitch_yaw_rate = np.linalg.norm(omega[:, 1:3], axis=1)
    fuel_fraction = (state[:, 13] - MASS_EMPTY_KG) / INITIAL_FUEL_KG
    finite = (
        np.all(np.isfinite(state), axis=1)
        & np.isfinite(leg_sink)
        & detected
    )
    t = THRESHOLDS
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
    audit_idx = {name: i for i, name in enumerate(audit_columns)}
    success_count = int(success.sum())

    report = {
        "method": (
            "standalone NumPy first-contact predicate; no imports from evaluator "
            "or metrics module"
        ),
        "protocol": "rocket_booster_recovery_first_contact_7000kg_ood_evaluation_v2",
        "result_npz": {
            "absolute_path": str(args.result_npz.resolve()),
            "sha256": sha256(args.result_npz),
        },
        "rows": len(state),
        "source_rows": {
            name: int(np.sum(source == source_id))
            for source_id, name in enumerate(source_names)
        },
        "success_definition": {
            "name": "landing_success",
            "only_success_standard": True,
            "endpoint": "interpolated_first_landing_leg_contact",
            "post_contact_damping_credit": False,
            "thresholds": THRESHOLDS,
        },
        "landing_success": counts(success, source, source_names),
        "archive_consistency": {
            "landing_success_array_exact_match": bool(
                np.array_equal(success, stored_success)
            ),
            "landing_success_disagreements": int(
                np.count_nonzero(success != stored_success)
            ),
        },
        "hard_constraints": {
            "max_abs_rcs_pitch_action": float(np.max(max_action[:, 4])),
            "max_abs_rcs_yaw_action": float(np.max(max_action[:, 5])),
            "max_abs_grid_roll_action": float(np.max(max_action[:, 8])),
            "max_plant_realized_lateral_rcs_nm": float(
                np.max(audit[:, audit_idx["max_plant_lateral_rcs_nm"]])
            ),
            "nonfinite_trajectories": int(np.count_nonzero(~finite & detected)),
        },
        "single_success_gate_counts": {
            "first_contact": int(np.count_nonzero(detected)),
            "lateral_le_5m": int(np.count_nonzero(lateral <= 5.0)),
            "com_sink_le_1mps_and_not_rising": int(
                np.count_nonzero((vel[:, 0] >= -1.0) & (vel[:, 0] <= 0.0))
            ),
            "contact_leg_sink_le_1mps": int(np.count_nonzero(leg_sink <= 1.0)),
            "fuel_reserve_gt_2pct": int(np.count_nonzero(fuel_fraction > 0.02)),
            "landing_success_joint": success_count,
        },
    }
    if scalar is not None:
        report["summary_consistency"] = {
            "summary_path": str(args.summary.resolve()),
            "summary_sha256": sha256(args.summary),
            "landing_success_count_match": (
                scalar["overall"]["landing_success_pass"]["count"]
                == success_count
            ),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
