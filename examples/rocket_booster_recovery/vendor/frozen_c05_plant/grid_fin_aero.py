"""Grid-fin station + authority shaping parameters for the C03 variant (gen1_pod7).

This module owns the grid-fin force/torque authority coefficients and the fin
station placement for the "analytic restoring-moment fin-station and authority
shaping" mechanism. The variant-local harness imports it and applies the
resolved config to the forward grid-fin aero model in `step_one`.

The aerodynamic force/torque sign conventions and the 9-dim action interface
are unchanged from the canonical harness; only the deflection-to-force/torque
coefficients (split per lateral axis) and the fin-station moment arm are
rescaled. `fin_torque_damping` and `grid_fin_roll_control_cl` are left at
baseline so roll damping is kept intact.

Mechanism hypothesis (C03):
    The restoring moment about the COM scales with fin_station_x, but the same
    off-center fin side force also produces a translation-rotation coupling
    torque that must stay inside the RCS budget during outer-disk lateral
    capture. The static moment balance (analysis/fin_moment_balance.md)
    predicts the optimum station is ~6 m (bounded neighborhood {4,6,8,10}),
    refining the empirical fin_x=8. Authority shaping keeps x*CL_ctrl ~ const
    so authority can be raised only when the station is lowered.

Axis mapping (canonical body frame, x-up lateral plane):
    grid_y -> body-y force -> yaw (body-z) moment about the fin station
    grid_z -> body-z force -> pitch (body-y) moment about the fin station
    Pitch (omega_y) is the artifact-free tip-over axis.

Environment overrides (empirical authority-landscape sweep):
    SWORDFISH_GRID_CL         -> default lateral force slope (both axes)
    SWORDFISH_GRID_CL_Y       -> grid_y (yaw)  lateral force slope
    SWORDFISH_GRID_CL_Z       -> grid_z (pitch) lateral force slope
    SWORDFISH_GRID_ROLL_CL    -> grid_fin_roll_control_cl (roll moment slope)
    SWORDFISH_GRID_MAX        -> grid_fin_control_max (deflection limit)
    SWORDFISH_FIN_STATION_X   -> fin_station_x (m from COM, forward/nose +)
    SWORDFISH_FIN_DAMPING     -> fin_torque_damping (dimensionless)
"""
from __future__ import annotations

import os

# Canonical baseline values (assets/harness/ppo_rocket_6dof_finned_jax.py).
BASELINE = {
    "grid_fin_control_cl": 1.6,
    "grid_fin_roll_control_cl": 0.55,
    "grid_fin_control_max": 0.35,
    "fin_station_x": 16.0,
    "fin_torque_damping": 0.22,
}

# Primary config for this pod (D_g17_01): fuel1.3 vs fuel1.0 on the
# structurally-deployable near-COM fin_x=0.5 station (fin_x!=0 => non-zero
# grid-fin pitch/yaw moment arm). All other authority/damping coefficients
# stay at the champion baseline. Override with SWORDFISH_FIN_STATION_X for
# the fin_station_x ablation ({0.0, 0.5}).
PRIMARY = {
    "grid_fin_control_cl": 1.6,
    "grid_fin_roll_control_cl": 0.55,
    "grid_fin_control_max": 0.35,
    "fin_station_x": 0.5,
    "fin_torque_damping": 0.22,
}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def authority() -> dict:
    """Return the resolved grid-fin authority config.

    Defaults to PRIMARY; each field can be overridden via environment variables
    so a single variant-local harness can sweep the station/authority landscape
    without touching the evaluator, metric, or tier code.
    """
    cfg = dict(PRIMARY)
    cfg["grid_fin_control_cl"] = _env_float("SWORDFISH_GRID_CL", cfg["grid_fin_control_cl"])
    cfg["grid_fin_control_cl_y"] = _env_float("SWORDFISH_GRID_CL_Y", cfg["grid_fin_control_cl"])
    cfg["grid_fin_control_cl_z"] = _env_float("SWORDFISH_GRID_CL_Z", cfg["grid_fin_control_cl"])
    cfg["grid_fin_roll_control_cl"] = _env_float("SWORDFISH_GRID_ROLL_CL", cfg["grid_fin_roll_control_cl"])
    cfg["grid_fin_control_max"] = _env_float("SWORDFISH_GRID_MAX", cfg["grid_fin_control_max"])
    cfg["fin_station_x"] = _env_float("SWORDFISH_FIN_STATION_X", cfg["fin_station_x"])
    cfg["fin_torque_damping"] = _env_float("SWORDFISH_FIN_DAMPING", cfg["fin_torque_damping"])
    return cfg


def describe(cfg: dict) -> str:
    return (
        "grid_fin_control_cl=%.3f (y=%.3f z=%.3f) grid_fin_roll_control_cl=%.3f "
        "grid_fin_control_max=%.3f fin_station_x=%.2f fin_torque_damping=%.3f"
        % (
            cfg.get("grid_fin_control_cl", float("nan")),
            cfg.get("grid_fin_control_cl_y", cfg.get("grid_fin_control_cl", float("nan"))),
            cfg.get("grid_fin_control_cl_z", cfg.get("grid_fin_control_cl", float("nan"))),
            cfg["grid_fin_roll_control_cl"],
            cfg["grid_fin_control_max"],
            cfg["fin_station_x"],
            cfg["fin_torque_damping"],
        )
    )
