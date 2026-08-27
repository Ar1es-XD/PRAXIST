"""gen47_pod1 DIG C03 — pinned-knee dose schedule with 2.5-3.25 operating envelope.

Mechanism (selected_contract.yaml C03):

  The RCS far-outer authority dose (SWORDFISH_RCS_FAR_OUTER_SCALE) has a
  PPO knife-edge on the fuel1.5 x hc4.5 gear-leader plant: the strict-Jiang
  plateau ends at dose 3.375 (full 0.9957 / far-outer 0.9901), first
  degradation appears at 3.4375, and catastrophic collapse lands at 3.5
  (full 0.7275 / far-outer 0.383) and 4.0 (0.335 / 0.0).

  The parent C07 guard used a HARD all-axes saturation step at knee=3.25:
      effective = min(dose, band_hi=3.25)  for dose > sat_threshold=3.25.
  That works, but it abandons the full 3.25->3.375 safe headroom and switches
  abruptly at the knee.

  This variant instead PINs the guard knee at 3.375 with an operating
  envelope [2.5, 3.25] and a SMOOTH (C1 smoothstep) transition across the
  knee, so the effective far-outer RCS dose:
      - equals the configured dose for dose <= band_hi (3.25), preserving the
        dose-3.0 win BITWISE (off-state identical to the parent plant);
      - rises smoothly from 3.25 to 3.375 over the [3.25, 3.375] window; and
      - is pinned flat at 3.375 (the last-known-safe PPO dose) for all
        configured doses above the knee.
  The transition window [3.25, 3.375] has width 0.125, exactly half the
  natural 3.25->3.5 cliff width, so any configured dose >= 3.375 maps to the
  safe 3.375 operating point (reduced sensitivity to dose drift).

  The guard is a plant-level deployability filter: it applies identically in
  controller-only evaluation and PPO residual evaluation.

Canonical 9-dim action interface (order unchanged):
    [gimbal_y, gimbal_z, throttle_raw, rcs_x, rcs_y, rcs_z,
     grid_y, grid_z, grid_roll]

Ablation hooks (env-var overrides of config.yaml `guard:` section):
  SWORDFISH_DOSE_GUARD_ON            guard master switch (1/0)
  SWORDFISH_DOSE_GUARD_BAND_LO       operating envelope lower bound (default 2.5)
  SWORDFISH_DOSE_GUARD_BAND_HI       operating envelope upper bound (default 3.25)
  SWORDFISH_DOSE_GUARD_SAT_THRESHOLD guard knee (default 3.375)
  SWORDFISH_DOSE_GUARD_BACKOFF_MODE  "smooth" | "saturate" | "none"
  SWORDFISH_DOSE_GUARD_SCOPE         "all" | "lateral"
  SWORDFISH_DOSE_GUARD_ENVELOPE_ON   1 = flat identity below band_hi (default);
                                     0 = remove the envelope (ramp spans
                                     [band_lo, knee]).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import jax.numpy as jnp

# Semantic config keys -> (env var name, default value).
_GUARD_SPEC = {
    "on": ("SWORDFISH_DOSE_GUARD_ON", 1.0),
    "band_lo": ("SWORDFISH_DOSE_GUARD_BAND_LO", 2.5),
    "band_hi": ("SWORDFISH_DOSE_GUARD_BAND_HI", 3.25),
    "sat_threshold": ("SWORDFISH_DOSE_GUARD_SAT_THRESHOLD", 3.375),
    "backoff_mode": ("SWORDFISH_DOSE_GUARD_BACKOFF_MODE", "smooth"),
    "scope": ("SWORDFISH_DOSE_GUARD_SCOPE", "all"),
    "envelope_on": ("SWORDFISH_DOSE_GUARD_ENVELOPE_ON", 1.0),
}

_DEFAULTS = {
    "on": 1.0,
    "band_lo": 2.5,
    "band_hi": 3.25,
    "sat_threshold": 3.375,
    "backoff_mode": "smooth",
    "scope": "all",
    "envelope_on": 1.0,
}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def load_guard_config(base_dir: str | Path) -> Dict:
    """Load the dose-margin guard config.

    Reads the `guard:` section of variant-local config.yaml, then applies
    SWORDFISH_DOSE_GUARD_* environment overrides. Returns a dict of the
    resolved scalar parameters (not JAX arrays).
    """
    params = dict(_DEFAULTS)
    cfg_path = Path(base_dir) / "config.yaml"
    if cfg_path.exists():
        try:
            import yaml
            loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            guard_section = loaded.get("guard") or {}
            for key in _GUARD_SPEC:
                if key in guard_section:
                    params[key] = guard_section[key]
        except Exception:
            # Fall back to defaults; the harness printout exposes the values.
            pass

    # Env-var overrides always win (sweep/ablation hooks).
    for key, (env_name, _default) in _GUARD_SPEC.items():
        if key in ("backoff_mode", "scope"):
            raw = os.environ.get(env_name)
            if raw and raw.strip():
                params[key] = raw.strip()
        else:
            if os.environ.get(env_name):
                params[key] = _env_float(env_name, params[key])
    return params


def env_defaults(params: Dict) -> Dict[str, str]:
    """Map resolved guard params to SWORDFISH_DOSE_GUARD_* env defaults."""
    out = {}
    for key, (env_name, _default) in _GUARD_SPEC.items():
        out[env_name] = str(params.get(key, _default))
    return out


def _guard_on(params: Dict) -> bool:
    try:
        return float(params.get("on", 0.0)) > 0.5
    except (TypeError, ValueError):
        return bool(params.get("on", False))


def _smoothstep(x: float) -> float:
    """C1 smoothstep on [0,1] (0->0, 1->1, zero slope at both ends)."""
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _jnp_smoothstep(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _effective_dose_scalar(dose: float, params: Dict) -> float:
    """Scalar implementation of the pinned-knee smooth schedule."""
    d = float(dose)
    if not _guard_on(params):
        return d
    mode = str(params.get("backoff_mode", "smooth")).strip().lower()
    if mode in ("none", "off", "bypass"):
        return d
    band_hi = float(params["band_hi"])
    knee = float(params["sat_threshold"])

    if mode == "saturate":
        # Legacy hard-step (parent C07): cap at band_hi above the knee.
        if d > knee:
            return min(d, band_hi)
        return d

    # mode == "smooth" (default): pinned-knee smoothstep schedule.
    envelope_on = float(params.get("envelope_on", 1.0)) > 0.5
    if envelope_on:
        # Operating envelope: identity below band_hi; ramp [band_hi, knee].
        if d <= band_hi:
            return d
        span = max(knee - band_hi, 1e-6)
        return band_hi + (knee - band_hi) * _smoothstep((d - band_hi) / span)

    # Without the operating envelope: identity below band_lo; ramp [band_lo, knee].
    band_lo = float(params["band_lo"])
    if d <= band_lo:
        return d
    span = max(knee - band_lo, 1e-6)
    return band_lo + (knee - band_lo) * _smoothstep((d - band_lo) / span)


def effective_lateral_dose_py(dose: float, params: Dict) -> float:
    """Python-level (trace-time) effective far-outer lateral RCS dose.

    Safe to call inside JAX-traced functions because it only touches Python
    scalars. Mirrors effective_lateral_dose semantics.
    """
    return _effective_dose_scalar(dose, params)


def effective_lateral_dose(dose: jnp.ndarray | float, params: Dict) -> jnp.ndarray:
    """Effective far-outer lateral RCS dose after the pinned-knee schedule.

    Guard off / mode none  -> dose unchanged.
    mode smooth (default):
        dose <= band_hi       -> dose (bit-identical to the unguarded plant).
        band_hi < dose < knee -> smoothstep ramp band_hi -> knee.
        dose >= knee          -> pinned flat at knee (last-known-safe dose).
    mode saturate (legacy)   -> min(dose, band_hi) for dose > knee.
    """
    d = jnp.asarray(dose, dtype=jnp.float32)
    if not _guard_on(params):
        return d
    mode = str(params.get("backoff_mode", "smooth")).strip().lower()
    if mode in ("none", "off", "bypass"):
        return d
    band_hi = jnp.asarray(float(params["band_hi"]), dtype=jnp.float32)
    knee = jnp.asarray(float(params["sat_threshold"]), dtype=jnp.float32)

    if mode == "saturate":
        engaged = d > knee
        return jnp.where(engaged, jnp.minimum(d, band_hi), d)

    envelope_on = float(params.get("envelope_on", 1.0)) > 0.5
    if envelope_on:
        span = jnp.maximum(knee - band_hi, 1e-6)
        s = _jnp_smoothstep((d - band_hi) / span)
        ramp = band_hi + (knee - band_hi) * s
        return jnp.where(d <= band_hi, d, ramp)
    band_lo = jnp.asarray(float(params["band_lo"]), dtype=jnp.float32)
    span = jnp.maximum(knee - band_lo, 1e-6)
    s = _jnp_smoothstep((d - band_lo) / span)
    ramp = band_lo + (knee - band_lo) * s
    return jnp.where(d <= band_lo, d, ramp)


def describe(params: Dict) -> str:
    on = _guard_on(params)
    envelope_on = float(params.get("envelope_on", 1.0)) > 0.5
    return (
        "dose_guard(on=%s band=[%.3f, %.3f] knee=%.3f mode=%s scope=%s envelope=%s)"
        % (
            "1" if on else "0",
            float(params["band_lo"]),
            float(params["band_hi"]),
            float(params["sat_threshold"]),
            str(params["backoff_mode"]),
            str(params.get("scope", "all")),
            "1" if envelope_on else "0",
        )
    )


# ── Standard residual-mask interface (kept for compatibility) ────────────────
def build_mask(cfg) -> jnp.ndarray:
    """Canonical 9-dimensional residual mask (static channel authority).

    The pinned-knee dose schedule operates on the realized RCS torque
    (plant level), not on this static residual mask; this function is kept
    only so the module remains a drop-in residual-mask provider for the
    canonical interface.
    """
    return jnp.array([
        cfg.residual_gimbal, cfg.residual_gimbal,
        cfg.residual_throttle,
        cfg.residual_rcs, cfg.residual_rcs, cfg.residual_rcs,
        cfg.residual_gridfin, cfg.residual_gridfin, cfg.residual_gridfin,
    ], dtype=jnp.float32)


# ── gen54_pod9 C01 compose: far-outer zero-tail residual mask ────────────────
# A radial-bin-gated residual-action mask. For trajectories whose INITIAL
# lateral radius lies in the far-outer bin (initial_r >= radius, Definition B
# far-outer = 1124 m), the PPO residual authority is zeroed so the deterministic
# guidance prior drives the descent exactly. This removes the fuel2.0
# outward-rim slide-out tail that the learned residual introduces in the far
# outer bin while preserving nominal (inner/mid-disk) fused-arm residual
# behavior. It is a deployability filter on the residual: it applies during both
# PPO training and evaluation (consistent plant), so the policy only ever
# receives learning signal from the inner/mid disk.
#
# Ablation hooks (env vars, config.yaml `far_outer_mask:` section):
#   SWORDFISH_FAR_OUTER_MASK_ON      master switch (1/0)
#   SWORDFISH_FAR_OUTER_MASK_RADIUS  radial boundary initial_r (default 1124 m)
#   SWORDFISH_FAR_OUTER_MASK_WIDTH   smoothstep ramp width (default 60 m)
#   SWORDFISH_FAR_OUTER_MASK_MODE    "soft" (smoothstep) | "hard" (step)

_FAR_OUTER_MASK_DEFAULTS = {
    "on": 0.0,
    "radius": 1124.0,
    "width": 60.0,
    "mode": "soft",
}


def far_outer_zero_tail_gate(
    initial_r: jnp.ndarray | float,
    radius: float = 1124.0,
    width: float = 60.0,
    mode: str = "soft",
) -> jnp.ndarray:
    """Scalar residual-authority gate keyed on INITIAL lateral radius.

    Returns 1.0 (residual active) for inner/mid-disk bins and 0.0 (residual
    nulled) for the far-outer bin, with a C1 smoothstep ramp across
    [radius, radius+width]. This is the far-outer zero-tail residual mask.
    """
    r = jnp.asarray(initial_r, dtype=jnp.float32)
    span = jnp.maximum(jnp.asarray(float(width), dtype=jnp.float32), 1e-6)
    x = (r - jnp.asarray(float(radius), dtype=jnp.float32)) / span
    if mode == "hard":
        u = jnp.where(x > 0.0, 1.0, 0.0)
    else:
        s = jnp.clip(x, 0.0, 1.0)
        u = s * s * (3.0 - 2.0 * s)
    return 1.0 - u


def far_outer_mask_description(params: Dict) -> str:
    return (
        "far_outer_mask(on=%s radius=%.3f m width=%.3f m mode=%s)"
        % (
            "1" if float(params.get("on", 0.0)) > 0.5 else "0",
            float(params.get("radius", 1124.0)),
            float(params.get("width", 60.0)),
            str(params.get("mode", "soft")),
        )
    )
