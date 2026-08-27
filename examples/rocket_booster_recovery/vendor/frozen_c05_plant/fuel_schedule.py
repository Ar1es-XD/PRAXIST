#!/usr/bin/env python3
"""Fuel-margin-keyed deterministic schedule helpers (gen52_pod9 C01/C02 arms).

These are the *forward metric-driver* schedule arms that pair with the C04
fuel-keyed residual mask guard. They key on the fuel CELL (`fuel_scale`, the
initial-propellant-capacity multiplier) exactly as the C04 mask does, so the
fuel1.5 neighborhood is a hard identity plateau and the fuel1.5 zero-tail is
preserved as an invariant.

Two independent schedules (both additive to the guidance prior, never touching
the evaluator/metric/split/disk_radius):

  * `divert_gain_schedule` (C01): a fuel-margin-keyed multiplier on the
    cross-range tangential tilt-demand prior gain. Eases lateral authority near
    exhaustion (fuel0.7) and raises terminal capture authority at high fuel
    (fuel2.0). Identity inside the fuel1.5 neighborhood.

  * `descent_commit_schedule` (C02): a fuel-margin-keyed vertical descent
    commitment bias. Zero inside the fuel1.5 neighborhood (identity), positive
    at fuel2.0 where the guidance prior otherwise settles into a stable
    hover-never-touch equilibrium ~1.5 m above the pad.
"""
from __future__ import annotations

FUEL_LO = 0.7
FUEL_HI = 2.0
FUEL_CENTER = 1.5
FUEL_HALFWIDTH = 0.25


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def fuel_schedule_multiplier(
    fuel_scale: float,
    center: float = FUEL_CENTER,
    halfwidth: float = FUEL_HALFWIDTH,
    lo_val: float = 1.0,
    anchor_val: float = 1.0,
    hi_val: float = 1.0,
) -> float:
    """Piecewise-linear, fuel-margin-keyed multiplier.

    * ``anchor_val`` inside ``|fuel_scale - center| <= halfwidth`` (hard
      identity plateau -> fuel1.5 zero-tail preserved).
    * ``lo_val`` at ``fuel_scale == FUEL_LO`` (0.7).
    * ``hi_val`` at ``fuel_scale == FUEL_HI`` (2.0).
    * Linear ramps in between, clamped so the plateaus stay hard.
    """
    lo_edge = center - halfwidth
    hi_edge = center + halfwidth
    if abs(fuel_scale - center) <= halfwidth:
        return anchor_val
    if fuel_scale < lo_edge:
        if fuel_scale <= FUEL_LO:
            return lo_val
        u = _clamp01((lo_edge - fuel_scale) / (lo_edge - FUEL_LO))
        return anchor_val + (lo_val - anchor_val) * u
    # fuel_scale > hi_edge
    if fuel_scale >= FUEL_HI:
        return hi_val
    u = _clamp01((fuel_scale - hi_edge) / (FUEL_HI - hi_edge))
    return anchor_val + (hi_val - anchor_val) * u


def divert_gain_multiplier(cfg) -> float:
    """C01 fuel-keyed cross-range prior gain multiplier.

    Identity at fuel1.5; eases authority at fuel0.7 (lo < 1) and raises
    terminal capture authority at fuel2.0 (hi > 1).
    """
    return fuel_schedule_multiplier(
        float(cfg.fuel_scale),
        lo_val=float(cfg.divert_gain_sched_lo),
        anchor_val=1.0,
        hi_val=float(cfg.divert_gain_sched_hi),
    )


def descent_commit_multiplier(cfg) -> float:
    """C02 fuel-keyed vertical descent-commitment multiplier.

    Zero at fuel1.5 (identity) and at fuel0.7 (leave the exhaustion-bound cell
    alone); ramps to ``descent_commit_hi`` at fuel2.0 where the guidance prior
    hovers and never touches down.
    """
    return fuel_schedule_multiplier(
        float(cfg.fuel_scale),
        lo_val=0.0,
        anchor_val=0.0,
        hi_val=float(cfg.descent_commit_hi),
    )


def gear_height_multiplier(cfg) -> float:
    """C03-adjacent fuel-margin-keyed landing-gear contact-height schedule.

    The fuel1.5 zero-tail anchor uses gear_contact_height=4.5 m, which sits on
    the documented gear contact-height knife-edge for the HEAVIER fuel2.0
    vehicle (the bottom-out-guard spring cap stretches the final ~0.1 m/s sink
    phase past the 90 s horizon -> hover-never-touch). Lowering the contact
    height at high fuel margin shortens the required sink travel so the vehicle
    completes touchdown within the horizon, while identity at fuel1.5 keeps the
    zero-tail untouched.
    """
    return fuel_schedule_multiplier(
        float(cfg.fuel_scale),
        lo_val=float(cfg.gear_h_sched_lo_mult),
        anchor_val=1.0,
        hi_val=float(cfg.gear_h_sched_hi_mult),
    )


def describe(cfg) -> str:
    return (
        "fuel_schedule(fuel=%.2f divert_gain_mult=%.3f descent_commit_mult=%.3f)"
        % (
            float(cfg.fuel_scale),
            divert_gain_multiplier(cfg),
            descent_commit_multiplier(cfg),
        )
    )
