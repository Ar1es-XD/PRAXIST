#!/usr/bin/env python3
"""Fuel-margin-keyed residual action mask (gen52_pod9 DIG C04).

Contract: selected_candidate_id C04, semantic_family
`fuel_margin_keyed_residual_mask`, mechanism_family `deployability_filter`,
intervention_surface `residual_action_mask`, intent `repair`.

Mechanism
---------
A deployability filter multiplies the PPO residual action by a scalar mask
keyed on the *fuel cell* (initial propellant-capacity multiplier `fuel_scale`).

  * `guard` (the C04 mechanism): identity outside the fuel1.5 neighborhood,
    zero inside it. This nulls learned residual authority in the proven-safe
    fuel1.5 cell so a schedule arm exploring fuel0.7/2.0 cannot re-open the
    fuel1.5 zero-tail.
  * `disabled`  (ablation / control): identity everywhere (residual active
    everywhere).
  * `extremes_only` (inverse ablation): zero outside the fuel1.5 neighborhood
    (i.e. nulling at fuel0.7/2.0 extremes), identity inside.
  * `all` (ablation): zero everywhere (residual authority nulled at all cells).

The mask is a scalar per environment cell (fuel_scale is fixed per run), so it
is a hard invariant on the protected cell and never forces `updates=0` or
`residual_scale=0.0` globally.
"""
from __future__ import annotations

import jax.numpy as jnp

FUEL_CENTER = 1.5
FUEL_HALFWIDTH = 0.25


def fuel_mask_value(fuel_scale, mode: str = "guard", center: float = FUEL_CENTER,
                    halfwidth: float = FUEL_HALFWIDTH):
    """Return the scalar residual-authority mask for a given fuel cell.

    The returned value multiplies the full 9-dim residual action before
    `residual_scale`. A value of 1.0 leaves the residual unchanged; a value of
    0.0 nulls residual authority.
    """
    in_neighborhood = jnp.abs(fuel_scale - center) <= halfwidth
    if mode == "guard":
        return jnp.where(in_neighborhood, 0.0, 1.0)
    if mode == "disabled":
        return jnp.where(in_neighborhood, 1.0, 1.0)  # identity everywhere
    if mode == "extremes_only":
        return jnp.where(in_neighborhood, 1.0, 0.0)  # inverse: nulls at extremes
    if mode == "all":
        return jnp.where(in_neighborhood, 0.0, 0.0)  # zero everywhere
    raise ValueError(f"unknown fuel mask mode: {mode!r}")


def fuel_mask_vector(cfg, mode: str | None = None):
    """Broadcastable 9-element mask for the standard action interface."""
    m = getattr(cfg, "fuel_mask_mode", "guard") if mode is None else mode
    return jnp.full((9,), fuel_mask_value(
        float(getattr(cfg, "fuel_scale", 1.5)), mode=m,
        center=float(getattr(cfg, "fuel_mask_center", FUEL_CENTER)),
        halfwidth=float(getattr(cfg, "fuel_mask_halfwidth", FUEL_HALFWIDTH)),
    ), dtype=jnp.float32)


def mask_description(cfg) -> str:
    return (
        f"fuel_mask(mode={getattr(cfg, 'fuel_mask_mode', 'guard')}, "
        f"fuel_scale={float(getattr(cfg, 'fuel_scale', 1.5))}, "
        f"center={float(getattr(cfg, 'fuel_mask_center', FUEL_CENTER))}, "
        f"halfwidth={float(getattr(cfg, 'fuel_mask_halfwidth', FUEL_HALFWIDTH))})"
    )
