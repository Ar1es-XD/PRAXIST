"""Outward-rim radial-gated lateral-nulling divert (DIG C01 / H_g47_02).

Mechanism (contract C01, amended per contract_amendment.yaml v1)
--------------------------------------------------------------------
The fuel1.5 far-outer capture is lost to outward-rim (vr > 0) slide-outs whose
dominant mode is an RCS over-authority attitude tumble when the guidance prior
aggressively nulls outward radial velocity. The naive nulling direction
(divert_sign=+1) was implemented first and FALSIFIED (it doubled the >50 m
slide-out tail). The beneficial direction is the sign-reversed form
(divert_sign=-1): a deterministic guidance-level term that SOFTENS the inward
tilt demand for fast outward-rim trajectories, preventing the over-authority
tumble while the canonical ZEM/ZEV position term still captures the vehicle.

The term added to the inertial lateral thrust_acc is

    a_div = -divert_sign * divert_gain * vr_eff * r_hat * gate_r * gate_h

where
    r_hat   = unit radial vector in the inertial y-z plane,
    vr      = (vel_yz . r_hat)  (outward radial-velocity component),
    vr_eff  = max(vr, 0) when divert_vr_only else vr,
    gate_r  = radial gate on the trajectory INITIAL lateral radius (0 inside
              divert_gate_radius, 1 outside divert_gate_radius+width),
    gate_h  = optional altitude gate (0 below divert_alt_low_m, 1 above
              divert_alt_high_m).

With divert_sign=-1 (treat default), a_div points outward for vr>0, i.e. it
reduces the commanded inward tilt (softens the nulling demand). With
divert_sign=+1 (naive null, falsified ablation), a_div points inward and adds
tilt demand. The term is hard-capped to a tilt fraction of the vertical
acceleration so it can never demand an over-tilt. This is a guidance prior
only: it never touches the evaluator, metric computation, data split, or
disk_radius.

Ablation hooks (via SWORDFISH_DIVERT_* env vars / CLI):
    divert_on            1.0 treat / 0.0 control (divert fully off)
    divert_gate_on       1.0 radial-gated (contract) / 0.0 radial gate off
    divert_sign         +1.0 treat / -1.0 sign-reversed ablation
    divert_gain          outward-velocity nulling gain (1/s)
    divert_gate_radius   radial activation boundary (m), contract 1124.0
    divert_gate_width    radial ramp width (m)
    divert_vr_only       1.0 null only vr>0 / 0.0 damp all radial velocity
    divert_alt_gate_on   1.0 altitude gate on / 0.0 off
    divert_alt_low_m     below this height the divert is fully OFF
    divert_alt_high_m    above this height the divert is fully ON
    divert_tilt_cap_rad  hard lateral-acc tilt cap (rad)
"""
from __future__ import annotations

import jax.numpy as jnp


def _radial_gate(initial_r, cfg):
    """0 below divert_gate_radius, 1 above divert_gate_radius+width."""
    if cfg.divert_gate_on <= 0.5:
        return 1.0
    lo = cfg.divert_gate_radius
    w = jnp.maximum(cfg.divert_gate_width, 1e-6)
    return jnp.clip((initial_r - lo) / w, 0.0, 1.0)


def _altitude_gate(height, cfg):
    """0 below divert_alt_low_m, 1 above divert_alt_high_m."""
    if cfg.divert_alt_gate_on <= 0.5:
        return 1.0
    lo = cfg.divert_alt_low_m
    hi = cfg.divert_alt_high_m
    return jnp.clip((height - lo) / jnp.maximum(hi - lo, 1e-6), 0.0, 1.0)


def compute_divert(pos, vel, initial_r, cfg, gain_scale=1.0):
    """Return a 2-vector of inertial lateral acceleration (y, z) to add.

    Args:
        pos: full inertial position vector (3,).
        vel: full inertial velocity vector (3,).
        initial_r: initial lateral radius (m) carried in state[15].
        cfg: EnvCfg namedtuple with the divert_* fields.
        gain_scale: fuel-margin-keyed multiplier on divert_gain (C01 schedule).
            Default 1.0 leaves the parent byte-identical.
    """
    if cfg.divert_on <= 0.5:
        return jnp.zeros(2, dtype=pos.dtype)

    pos_lat = pos[1:3]
    vel_lat = vel[1:3]
    r = jnp.linalg.norm(pos_lat) + 1e-8
    r_hat = pos_lat / r
    vr = vel_lat[0] * r_hat[0] + vel_lat[1] * r_hat[1]

    vr_eff = jnp.where(cfg.divert_vr_only > 0.5, jnp.maximum(vr, 0.0), vr)

    a = -cfg.divert_sign * cfg.divert_gain * gain_scale * vr_eff * r_hat
    a = a * (_radial_gate(initial_r, cfg) * _altitude_gate(pos[0], cfg))

    # Hard lateral-acc cap relative to the vertical command so the divert can
    # never demand an over-tilt. Vertical acc is not passed in; the caller
    # applies the cap using its own |acc_x| (see guidance_action).
    return a


def tilt_cap_scale(acc_x_abs, cfg):
    """Max |a_div| = tan(divert_tilt_cap) * max(|acc_x|, 1.0)."""
    return jnp.tan(cfg.divert_tilt_cap_rad) * jnp.maximum(acc_x_abs, 1.0)


def describe(cfg) -> str:
    return (
        "divert_on=%.2f gate_on=%.2f sign=%.2f gain=%.3f gate_r=%.0f gate_w=%.0f "
        "vr_only=%.2f alt_gate=%.2f alt_lo=%.0f alt_hi=%.0f tilt_cap=%.3f"
        % (
            cfg.divert_on,
            cfg.divert_gate_on,
            cfg.divert_sign,
            cfg.divert_gain,
            cfg.divert_gate_radius,
            cfg.divert_gate_width,
            cfg.divert_vr_only,
            cfg.divert_alt_gate_on,
            cfg.divert_alt_low_m,
            cfg.divert_alt_high_m,
            cfg.divert_tilt_cap_rad,
        )
    )
