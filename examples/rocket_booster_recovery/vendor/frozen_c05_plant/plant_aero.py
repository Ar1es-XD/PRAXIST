"""Variant-local static fin incidence / cant passive lateral-force channel (C02).

Owns the static (non-deflection) grid-fin incidence/cant parameters plus the
radial handoff gate and the optional tapered ALTITUDE gate that hands the
incidence channel off to the landing-gear suspension near touchdown.

A fixed geometric incidence produces a BODY-FIXED passive lateral force
proportional to dynamic pressure (independent of body crossflow), plus a small
induced-drag delta. It is a NEW labeled passive lateral force channel distinct
from (a) the v_lat-dependent passive crossflow lift already in the plant and
(b) the active grid-fin deflection commands a[6:9].

Conventions (body frame, x-up; pitch axis = body-y = omega_y):
  * fin_incidence_y_deg : cant about body-y axis -> steady body-Z lateral force
  * fin_incidence_z_deg : cant about body-z axis -> steady body-Y lateral force
  * fin_incidence_cl    : passive lateral-force slope per rad of incidence
  * fin_incidence_cd    : induced-drag coefficient per rad^2 of incidence

Handoff surface (contract C02):
  * radial band gate [fin_incidence_radial_radius, fin_incidence_radial_upper]
    with sigmoid width fin_incidence_radial_width (hard gate = small width,
    tapered gate = large width). upper == 0 means a lower-only gate.
  * tapered altitude gate [fin_incidence_alt_low_m, fin_incidence_alt_high_m]
    (only when fin_incidence_alt_gate_on > 0.5) so incidence fades below the
    altitude handoff and the gear suspension owns terminal deceleration.

Null ablation (incidence == 0) returns EXACT zeros for both force and drag.
"""
from __future__ import annotations

import jax.numpy as jnp


def describe(cfg) -> str:
    return (
        "fin_incidence_on=%.3f y=%.3fdeg z=%.3fdeg cl=%.3f cd=%.3f "
        "rgate=%.1f r=%.0f up=%.0f w=%.0f agate=%.1f alo=%.0f ahi=%.0f aw=%.0f"
        % (
            cfg.fin_incidence_on,
            cfg.fin_incidence_y_deg,
            cfg.fin_incidence_z_deg,
            cfg.fin_incidence_cl,
            cfg.fin_incidence_cd,
            cfg.fin_incidence_radial_gate_on,
            cfg.fin_incidence_radial_radius,
            cfg.fin_incidence_radial_upper,
            cfg.fin_incidence_radial_width,
            cfg.fin_incidence_alt_gate_on,
            cfg.fin_incidence_alt_low_m,
            cfg.fin_incidence_alt_high_m,
            cfg.fin_incidence_alt_width_m,
        )
    )


def _sigmoid_step(x: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (1.0 + jnp.tanh(x))


def radial_gate(initial_r, cfg):
    """Radial gate on the static incidence channel.

    Returns a scalar in [0,1] multiplying the incidence force + drag delta.
    gate_on <= 0.5 -> static-always 1.0. Otherwise a sigmoid band on the
    trajectory's INITIAL lateral radius:
      * lower threshold: 0 inside fin_incidence_radial_radius -> 1 outside
      * optional upper threshold (handoff): 1 inside -> 0 outside.
    """
    if cfg.fin_incidence_radial_gate_on <= 0.5:
        return 1.0
    w = jnp.maximum(cfg.fin_incidence_radial_width, 1e-6)
    x_lo = (initial_r - cfg.fin_incidence_radial_radius) / w
    u_lo = _sigmoid_step(x_lo)
    if cfg.fin_incidence_radial_upper > cfg.fin_incidence_radial_radius + 1e-6:
        x_up = (cfg.fin_incidence_radial_upper - initial_r) / w
        u_up = _sigmoid_step(x_up)
        return u_lo * u_up
    return u_lo


def altitude_gate(altitude, cfg):
    """Tapered altitude gate: 1 inside [alt_low, alt_high], 0 outside.

    Below alt_low_m the incidence channel is fully OFF (handed off to the
    landing-gear suspension). Above alt_high_m it is fully OFF too, so the
    incidence channel only operates through the mid/terminal descent band.
    """
    if cfg.fin_incidence_alt_gate_on <= 0.5:
        return 1.0
    w = jnp.maximum(cfg.fin_incidence_alt_width_m, 1e-6)
    u_lo = _sigmoid_step((altitude - cfg.fin_incidence_alt_low_m) / w)
    u_hi = _sigmoid_step((cfg.fin_incidence_alt_high_m - altitude) / w)
    return u_lo * u_hi


def incidence_gate(initial_r, altitude, cfg):
    """Combined radial band gate x tapered altitude gate."""
    return radial_gate(initial_r, cfg) * altitude_gate(altitude, cfg)


def incidence_terms(q_dyn, fin_area_total, initial_r, altitude, cfg):
    """Return (F_inc_B, cd_inc) for the static fin incidence channel.

    Args:
        q_dyn: dynamic pressure 0.5*rho*speed^2 (scalar tracer).
        fin_area_total: fin_area_each * fin_count (scalar tracer).
        initial_r: trajectory initial lateral radius (scalar tracer) for the
            radial handoff gate.
        altitude: current COM height above ground (scalar tracer) for the
            optional tapered altitude gate.
        cfg: EnvCfg carrying the fin_incidence_* fields.

    Returns:
        F_inc_B: body-frame passive lateral force vector (length 3).
        cd_inc: induced-drag coefficient delta (scalar).
    """
    iy = jnp.deg2rad(cfg.fin_incidence_y_deg)
    iz = jnp.deg2rad(cfg.fin_incidence_z_deg)
    sy = jnp.sin(iy)
    sz = jnp.sin(iz)
    on = cfg.fin_incidence_on
    cl = cfg.fin_incidence_cl
    cd_i = cfg.fin_incidence_cd

    gate = incidence_gate(initial_r, altitude, cfg)
    # Body-fixed lateral force: y-incidence -> body-z force, z-incidence ->
    # body-y force. Exactly zero when incidence is zero (null ablation).
    F_inc_B = on * gate * q_dyn * fin_area_total * cl * jnp.array([0.0, sz, sy])
    # Induced-drag delta (quadratic in sine of incidence), zero at null.
    cd_inc = on * gate * cd_i * (sy * sy + sz * sz)
    return F_inc_B, cd_inc
