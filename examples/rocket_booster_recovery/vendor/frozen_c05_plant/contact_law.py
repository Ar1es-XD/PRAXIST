#!/usr/bin/env python3
"""Continuous fuel-margin-tracked landing-gear contact-height law (gen53_pod4 C01).

This is the DIG C01 forward-innovation repair over the gen52_pod9 fuel-keyed
gear contact-height schedule. The archived keyed schedule pins the touchdown
knife-edge to fixed fuel keys with a hard identity plateau inside the fuel1.5
neighborhood and a piecewise-linear ramp to fuel2.0:

    archived keyed anchors (multiplier on base gear_contact_height_m=4.5):
        fuel0.7 -> 1.0   (4.5 m)
        fuel1.5 -> 1.0   (4.5 m)
        fuel2.0 -> 0.667 (3.0015 m = 4.5 * 0.667)

C01 replaces that piecewise plateau+ramp with a single continuous monotone
map h(fuel_margin). It interpolates the exact same anchors and has a zero
derivative at fuel1.5 (smoothstep), so the fuel1.5 zero-tail anchor is
preserved while every intermediate fuel cell (1.5 < fuel < 2.0) receives a
strictly lower contact height than the keyed plateau. This is the mechanism
by which earlier-schedule (plateau-held) cells recover from never-touch.

Monotonicity convention
-----------------------
`fuel_margin = FUEL_HI - fuel_scale` is the contact-height-relevant margin:
it is 0.0 at the heaviest cell (fuel2.0) and 1.3 at the lightest (fuel0.7).
As the vehicle carries more fuel it needs a *shorter* deployed gear to finish
its bottom-out-guard sink within the 900-step horizon, so h decreases with
fuel_scale. Equivalently, h(fuel_margin) is monotone non-decreasing in
fuel_margin (and monotone non-increasing in fuel_scale). Both are asserted by
the fail-fast checks below.
"""
from __future__ import annotations

FUEL_LO = 0.7
FUEL_CENTER = 1.5
FUEL_HI = 2.0

# Archived keyed anchor multipliers (from gen52_pod9 fuel_schedule.py).
H_LO_MULT = 1.0          # fuel0.7 anchor
H_ANCHOR_MULT = 1.0      # fuel1.5 anchor
H_HI_MULT = 0.667        # fuel2.0 anchor (4.5 m * 0.667 = 3.0015 m)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def smoothstep(t: float) -> float:
    """C1-smooth Hermite step with zero slope at t=0 and t=1."""
    t = _clamp01(t)
    return t * t * (3.0 - 2.0 * t)


def fuel_margin(fuel_scale: float) -> float:
    """Contact-height-relevant fuel margin (0.0 at fuel2.0, 1.3 at fuel0.7)."""
    return float(FUEL_HI - fuel_scale)


def contact_height_multiplier(
    fuel_scale: float,
    mode: str = "smoothstep",
    lo_mult: float = H_LO_MULT,
    anchor_mult: float = H_ANCHOR_MULT,
    hi_mult: float = H_HI_MULT,
) -> float:
    """Continuous monotone contact-height multiplier in [hi_mult, anchor_mult].

    The returned multiplier is monotone non-decreasing in fuel_margin and
    monotone non-increasing in fuel_scale. It matches the archived keyed
    anchors exactly at fuel0.7/1.5/2.0 and is continuous everywhere.

    mode:
      "smoothstep"  - C1 smoothstep between fuel1.5 and fuel2.0 (default)
      "linear"      - piecewise-linear interpolation of the anchors
    """
    m = fuel_margin(fuel_scale)
    m_hi = fuel_margin(FUEL_HI)      # 0.0
    m_c = fuel_margin(FUEL_CENTER)   # 0.5
    m_lo = fuel_margin(FUEL_LO)      # 1.3

    if m <= m_c:
        # fuel >= 1.5: ramp from anchor_mult (fuel1.5) down to hi_mult (fuel2.0)
        t = _clamp01((m_c - m) / (m_c - m_hi))
        s = t if mode == "linear" else smoothstep(t)
        return anchor_mult + (hi_mult - anchor_mult) * s

    # fuel < 1.5: the archived anchors are equal (lo == anchor == 1.0), so the
    # monotone continuous interpolation is the constant anchor plateau.
    if abs(lo_mult - anchor_mult) < 1e-12:
        return anchor_mult
    t = _clamp01((m - m_c) / (m_lo - m_c))
    s = t if mode == "linear" else smoothstep(t)
    return anchor_mult + (lo_mult - anchor_mult) * s


def contact_height_m(
    fuel_scale: float,
    base_height_m: float,
    mode: str = "smoothstep",
) -> float:
    """Continuous contact height in meters: base_height_m * multiplier(fuel)."""
    return base_height_m * contact_height_multiplier(fuel_scale, mode=mode)


def describe(fuel_scale: float, base_height_m: float, mode: str) -> str:
    return (
        "contact_height_law(fuel=%.3f margin=%.3f mode=%s base=%.4f m -> h=%.4f m)"
        % (
            float(fuel_scale),
            fuel_margin(fuel_scale),
            mode,
            float(base_height_m),
            contact_height_m(fuel_scale, base_height_m, mode=mode),
        )
    )


def assert_monotone(fuel_grid=("0.7", "1.5", "2.0"), mode: str = "smoothstep") -> str:
    """Fail-fast monotonicity assertion over a fuel grid.

    Asserts h(fuel_margin) is monotone non-decreasing in fuel_margin and
    h(fuel_scale) is monotone non-increasing in fuel_scale. Returns a compact
    description of the evaluated grid for telemetry.
    """
    fuels = sorted(float(x) for x in fuel_grid)
    margins = [fuel_margin(f) for f in fuels]       # decreasing as fuel rises
    mults = [contact_height_multiplier(f, mode=mode) for f in fuels]

    # h(margin) non-decreasing: iterate in ascending margin order.
    order = sorted(range(len(fuels)), key=lambda i: margins[i])
    for a, b in zip(order, order[1:]):
        assert mults[a] <= mults[b] + 1e-12, (
            "h(fuel_margin) is not monotone non-decreasing: "
            "fuel %.2f -> %.4f, fuel %.2f -> %.4f"
            % (fuels[a], mults[a], fuels[b], mults[b])
        )

    # h(fuel_scale) non-increasing: iterate in ascending fuel order.
    for a, b in zip(range(len(fuels)), range(1, len(fuels))):
        assert mults[a] >= mults[b] - 1e-12, (
            "h(fuel_scale) is not monotone non-increasing: "
            "fuel %.2f -> %.4f, fuel %.2f -> %.4f"
            % (fuels[a], mults[a], fuels[b], mults[b])
        )
    return "fuel=%s margin=%s mult=%s" % (fuels, margins, mults)
