#!/usr/bin/env python3
"""Provenance labels for the C01 rk4/uni3 uniform-g3 far-outer re-anchor reference.

This module is intentionally dependency-free (no jax / no harness import) so it
can be imported by the variant-local harness without any circularity.  It turns
the resolved EnvCfg/PPOCfg/disk radius into a compact, machine-readable
provenance record that every run emits alongside eval_summary.json.

Contract provenance claims (must match the selected DIG-Lite contract C01):
  * integrator = rk4
  * coast-drag gate = uniform-g3 for r>1124  (terminal_drag_far_gain == terminal_drag_gain == 3.0,
    radial gate ON so r>1124 sees exactly g3 with no far boost)
  * plant = uni3@rk4 champion: fuel_scale=1.3, fin_station_x=0.5, RCS-1x radial schedule ON,
    capture-preserving bridge h80, canonical guidance, gimbal OFF
  * controller-only: actual_updates==0 and actual_residual_scale==0.0
  * disk_radius == 1500.0 (fail-fast asserted in the harness)
"""
from __future__ import annotations

from typing import Any, Mapping

CONTRACT = {
    "variant_name": "gen46_pod9/C07_dose_margin_guard",
    "contract_id": "C07",
    "target_hypothesis": "H_g46_09",
    "integrator_contract": "rk4",
    "coast_drag_gate_contract": "uniform_g3_r1124",
    "terminal_drag_gain_contract": 3.0,
    "terminal_drag_far_gain_contract": 3.0,
    "terminal_drag_radial_gate_on_contract": 1.0,
    "terminal_drag_far_radius_contract": 1124.0,
    "fuel_scale_contract": 1.5,
    "fin_station_x_contract": 0.5,
    "rcs_schedule_on_contract": 1.0,
    "bridge_on_contract": 1.0,
    "guidance_mode_contract": "canonical",
    "disk_radius_m_contract": 1500.0,
}


def _f(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _gate_label(cfg: Mapping[str, Any]) -> str:
    gain = _f(getattr(cfg, "terminal_drag_gain", None), 3.0)
    far_gain = _f(getattr(cfg, "terminal_drag_far_gain", None), 3.0)
    radial_on = _f(getattr(cfg, "terminal_drag_radial_gate_on", None), 1.0)
    far_radius = _f(getattr(cfg, "terminal_drag_far_radius", None), 1124.0)
    if radial_on <= 0.5:
        return f"uniform_g{gain:g}_r{far_radius:g}_gate_off"
    if abs(far_gain - gain) < 1e-9:
        return f"uniform_g{gain:g}_r{far_radius:g}"
    return f"radial_g{gain:g}_to_g{far_gain:g}_r{far_radius:g}"


def _plant_label(cfg: Mapping[str, Any], fin_station_x: float) -> str:
    fuel = _f(getattr(cfg, "fuel_scale", None), 1.3)
    rcs = _f(getattr(cfg, "rcs_schedule_on", None), 1.0)
    bridge = _f(getattr(cfg, "bridge_on", None), 1.0)
    integrator = str(getattr(cfg, "integrator", "rk4"))
    return f"{integrator}_fuel{fuel:g}_fin{fin_station_x:g}_rcs{rcs:g}_bridge{bridge:g}"


def build_provenance(env_cfg: Any, ppo_cfg: Any, disk_radius: float) -> dict:
    """Build the provenance record from resolved config objects."""
    fin_station_x = _f(getattr(env_cfg, "fin_station_x", None), 0.5)
    integrator = str(getattr(env_cfg, "integrator", "rk4"))
    gate = _gate_label(env_cfg)
    plant = _plant_label(env_cfg, fin_station_x)

    gain = _f(getattr(env_cfg, "terminal_drag_gain", None), 3.0)
    far_gain = _f(getattr(env_cfg, "terminal_drag_far_gain", None), 3.0)
    radial_on = _f(getattr(env_cfg, "terminal_drag_radial_gate_on", None), 1.0)

    is_contract_integrator = integrator == CONTRACT["integrator_contract"]
    is_contract_gate = (radial_on > 0.5) and abs(far_gain - gain) < 1e-9 and abs(gain - 3.0) < 1e-9
    is_contract_disk = abs(disk_radius - 1500.0) < 1e-6
    is_controller_only = bool(
        int(getattr(ppo_cfg, "updates", 0)) == 0
        and abs(_f(getattr(env_cfg, "residual_scale", None), 0.0)) < 1e-12
    )

    return {
        "variant_name": CONTRACT["variant_name"],
        "contract_id": CONTRACT["contract_id"],
        "target_hypothesis": CONTRACT["target_hypothesis"],
        "integrator": integrator,
        "integrator_contract": CONTRACT["integrator_contract"],
        "coast_drag_gate": gate,
        "coast_drag_gate_contract": CONTRACT["coast_drag_gate_contract"],
        "terminal_drag_gain": gain,
        "terminal_drag_far_gain": far_gain,
        "terminal_drag_radial_gate_on": radial_on,
        "terminal_drag_far_radius_m": _f(getattr(env_cfg, "terminal_drag_far_radius", None), 1124.0),
        "terminal_drag_on": _f(getattr(env_cfg, "terminal_drag_on", None), 1.0),
        "plant_label": plant,
        "fin_station_x_m": fin_station_x,
        "fuel_scale": _f(getattr(env_cfg, "fuel_scale", None), 1.3),
        "rcs_schedule_on": _f(getattr(env_cfg, "rcs_schedule_on", None), 1.0),
        "bridge_on": _f(getattr(env_cfg, "bridge_on", None), 1.0),
        "bridge_terminal_h_m": _f(getattr(env_cfg, "bridge_terminal_h", None), 80.0),
        "guidance_mode": str(getattr(env_cfg, "guidance_mode", "canonical")),
        "guidance_gimbal_on": _f(getattr(env_cfg, "guidance_gimbal_on", None), 0.0),
        "actual_updates": int(getattr(ppo_cfg, "updates", 0)),
        "actual_residual_scale": _f(getattr(env_cfg, "residual_scale", None), 0.0),
        "controller_only": is_controller_only,
        "disk_radius_m": float(disk_radius),
        "contract_integrator_ok": is_contract_integrator,
        "contract_gate_ok": is_contract_gate,
        "contract_disk_ok": is_contract_disk,
        "contract_controller_only_ok": is_controller_only,
        "provenance_label": f"{plant}|{gate}|controller_only_updates{int(getattr(ppo_cfg, 'updates', 0))}_rs{_f(getattr(env_cfg, 'residual_scale', None), 0.0):g}|disk{float(disk_radius):g}",
    }
