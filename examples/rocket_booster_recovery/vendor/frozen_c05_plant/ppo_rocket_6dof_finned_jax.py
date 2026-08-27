#!/usr/bin/env python3
"""
Guided pure-JAX PPO for an active-forward-grid-fin 6DOF rocket landing toy environment.

Purpose
-------
- Runs on macOS with only: jax, numpy, matplotlib.
- Uses a vectorized 6DOF rocket model: position, velocity, quaternion attitude,
  body angular velocity, mass.
- Trains a tanh-Gaussian PPO residual policy around a deterministic
  powered-descent guidance prior, from a fixed 2000 m initial height and a
  configurable fixed initial attitude.
- Saves a 3D trajectory view after training.

This is intentionally self-contained: no gymnax, flax, optax, distrax, or your
internal src.agents.ppo stack.

Example
-------
python ppo_rocket_6dof_finned_jax.py --updates 300 --num-envs 128 --num-steps 256
python ppo_rocket_6dof_finned_jax.py --updates 0 --controller-only
python ppo_rocket_6dof_finned_jax.py --fin-station-x 16 --fin-cd0 0.18 --controller-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import NamedTuple

# Keep memory use moderate on Apple Silicon / unified memory.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# ---------------------------------------------------------------------------
# Deterministic-JAX recipe (gen21_pod15 fd7420a7 lineage), applied verbatim
# for the C01 coast-drag g3 full-T5 validation. Applied BEFORE `import jax`;
# pass `--no-deterministic` for the canonical non-deterministic control arm.
# This pins the FP execution path so same-seed replica runs are bitwise
# identical (deterministic same-seed spread == 0.0).
# ---------------------------------------------------------------------------
_DETERMINISTIC = "--no-deterministic" not in sys.argv


def _build_determinism_env(deterministic: bool) -> str:
    if not deterministic:
        return os.environ.get("XLA_FLAGS", "")
    add = ["--xla_gpu_deterministic_ops=true", "--xla_gpu_autotune_level=0"]
    parts = os.environ.get("XLA_FLAGS", "").split()
    keys = {x.split("=")[0] for x in parts if x}
    for a in add:
        k = a.split("=")[0]
        if k not in keys:
            parts.append(a)
            keys.add(k)
    merged = " ".join(parts)
    os.environ["XLA_FLAGS"] = merged
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    return merged


_XLA_FLAGS = _build_determinism_env(_DETERMINISTIC)

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# ── gen46_pod9 DIG C07: variant-local config.yaml plant defaults + guard ──
# Load config.yaml (if present) and apply SWORDFISH_* env defaults via
# os.environ.setdefault so (a) the grid-fin authority module resolves the
# fin-station constant at import and (b) parse_args picks up the plant config.
# Explicit caller env vars always win (dose/band/guard sweep hooks).
import yaml as _yaml

_VARIANT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_VARIANT_DIR / "harness"))
import integrator_flags as _iflags  # gen71_pod3 C05: rk2_full/rk4 dt-ladder flags
try:
    _c07_loaded = _yaml.safe_load((_VARIANT_DIR / "config.yaml").read_text(encoding="utf-8")) or {}
    _c07_plant = _c07_loaded.get("plant") or {}
    for _c07_k, _c07_v in _c07_plant.items():
        os.environ.setdefault(str(_c07_k), str(_c07_v))
except Exception as _c07_e:
    print("C07 config.yaml load warning:", _c07_e, file=sys.stderr)

import residual_mask as _dosemask
_GUARD_CFG = _dosemask.load_guard_config(_VARIANT_DIR)
for _c07_env, _c07_val in _dosemask.env_defaults(_GUARD_CFG).items():
    os.environ.setdefault(_c07_env, _c07_val)

import grid_fin_aero as _grid_fin_aero
import reward_shaping as _reward
import plant_aero as _plant_aero
import provenance_labels as _provenance
import guidance_divert as _divert
import residual_fuel_mask as _fuelmask
import fuel_schedule as _fuelsched
import contact_law as _contactlaw

if _DETERMINISTIC:
    try:
        jax.config.update("jax_threefry_partitionable", True)
    except Exception:
        pass
    try:
        jax.config.update("jax_default_matmul_precision", "highest")
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------


class EnvCfg(NamedTuple):
    dt: float = 0.10
    max_steps: int = 900
    g: float = 9.81

    mass_full: float = 25600.0
    mass_empty: float = 22200.0
    # Propellant-capacity multiplier applied ONLY to initial fuel mass
    # (mass_full - mass_empty) * fuel_scale. Thrust and Isp are unchanged, so
    # burn rate is identical and only endurance increases. mass_frac observation
    # is normalized by the scaled capacity (same observation space).
    fuel_scale: float = 1.3
    fuel_mask_mode: str = "guard"
    fuel_mask_center: float = 1.5
    fuel_mask_halfwidth: float = 0.25
    thrust_max: float = 845000.0
    isp: float = 282.0
    length: float = 40.0
    radius: float = 1.83
    gimbal_max: float = 0.0873
    rcs_torque_max: float = 180000.0
    # ── Radial RCS-authority schedule (gen20_pod0 sched700_1to2, ported) ──
    # rcs_torque_max is multiplied by rcs_authority_scale * radial_gain(lateral_r).
    # schedule_on=0 -> flat unity gain; schedule_on=1 -> near/far sigmoid ramp.
    rcs_authority_scale: float = 1.0
    rcs_schedule_on: float = 1.0
    rcs_near_com_scale: float = 1.0
    rcs_far_outer_scale: float = 2.0
    rcs_boost_radius: float = 700.0
    rcs_schedule_width: float = 140.0

    lever_x: float = -15.0
    cd: float = 0.60
    rho: float = 1.05

    # Forward grid-fin aero extension. The inherited rocket model only had
    # isotropic inertial drag plus thrust-vector/RCS torques. For tail-first
    # powered descent the relative wind travels from engine/aft toward the
    # nose, so stabilizing aerodynamic surfaces should sit forward of the COM
    # (Falcon-9-style grid fins), not on the aft tail. These parameters add a
    # low-order four-grid-fin model: passive lateral aerodynamic force opposes
    # body-axis crossflow, and three active grid-fin commands add body-y/body-z
    # side force plus roll moment. Because the station is forward of COM, these
    # forces produce useful control/restoring moments in tail-first descent. This is intentionally differentiable and
    # task-scale, not CFD or a Mach-resolved grid-fin model.
    fins_on: float = 1.0
    # C04 sub-channel residual-mask hooks: independently gate the three
    # forward-grid-fin degrees of freedom (passive force = lift+drag,
    # passive damping torque, active grid lateral force + roll torque).
    # Defaults reproduce the control plant exactly.
    fin_force_on: float = 1.0
    # B_g13_01 composition hooks: decouple passive lift/drag channels
    # (ported from gen12_pod10/C01). Defaults trace fin_force_on exactly.
    fin_lift_on: float = 1.0
    fin_drag_on: float = 1.0
    fin_damping_on: float = 1.0
    fin_active_on: float = 1.0
    fin_area_each: float = 0.75      # m^2 projected/reference area per grid fin
    fin_count: float = 4.0
    fin_station_x: float = 8.0      # m from COM in body frame; forward/nose is positive
    fin_cl_alpha: float = 2.0        # effective lift slope after stall-softening
    fin_cd0: float = 0.18            # grid fins are intentionally high-drag
    fin_cd_alpha: float = 1.20       # induced/crossflow drag coefficient
    fin_alpha_stall: float = 0.45    # rad, smooth saturation scale for grid-fin force
    fin_torque_damping: float = 0.22 # extra aero damping moment, dimensionless
    grid_fin_control_max: float = 0.35      # rad-equivalent active grid-fin deflection limit
    grid_fin_control_cl: float = 1.6        # active lateral-force slope per rad deflection
    grid_fin_roll_control_cl: float = 0.55  # active roll-moment slope per rad deflection

    # ── Static fin incidence/cant passive lateral-force channel (C02, composed) ──
    # Fixed geometric incidence injects a body-fixed passive lateral force +
    # induced-drag delta into the plant. incidence=0 is the exact null ablation.
    # Radial band gate (lower/upper on INITIAL lateral radius) plus an optional
    # tapered ALTITUDE gate provide the incidence<->gear handoff surface.
    fin_incidence_on: float = 0.0     # master switch (1 = channel live)
    fin_incidence_y_deg: float = 0.0  # cant about body-y -> body-z force (deg)
    fin_incidence_z_deg: float = 0.0  # cant about body-z -> body-y force (deg)
    fin_incidence_cl: float = 2.0     # lateral-force slope per rad incidence
    fin_incidence_cd: float = 1.2     # induced-drag coefficient per rad^2 incidence
    fin_incidence_radial_gate_on: float = 0.0   # 0=static-always, 1=radial gate
    fin_incidence_radial_radius: float = 920.0  # lower gate radius (m)
    fin_incidence_radial_upper: float = 0.0     # optional upper/handoff gate radius (m); 0=none
    fin_incidence_radial_width: float = 60.0    # radial sigmoid blend width (m)
    fin_incidence_alt_gate_on: float = 0.0      # tapered altitude gate master switch
    fin_incidence_alt_low_m: float = 80.0       # below this height incidence fully OFF (handoff to gear)
    fin_incidence_alt_high_m: float = 400.0     # above this height incidence fully ON
    fin_incidence_alt_width_m: float = 100.0    # altitude sigmoid blend width (m)

    init_x: float = 2000.0
    init_y: float = -250.0
    init_z: float = 80.0
    init_vx: float = -75.0
    init_vy: float = 20.0
    init_vz: float = -6.0
    init_yaw_deg: float = -20.0     # rotation about inertial/body z in this x-up convention
    init_pitch_deg: float = 0.0
    init_roll_deg: float = 0.0
    # ── gen50_pod8 C07 landscape sweep: cross-regime stress IC hooks (diagnostic only; disk stays 1500) ──
    # stress_radial_offset_m > 0 fixes the initial radial offset (r1000 ring sampler).
    # stress_tilt_deg is added to init_pitch_deg (tilt stress).
    stress_radial_offset_m: float = 0.0
    stress_tilt_deg: float = 0.0

    randomize: float = 1.0
    pos_noise: float = 30.0
    vel_noise: float = 8.0
    attitude_noise_deg: float = 3.0
    omega_noise: float = 0.03

    landing_radius: float = 8.0
    landing_speed: float = 4.0
    landing_tilt: float = 0.10
    landing_omega: float = 0.15
    # ── H_g22_05 reward-side gate cliffs (metric-untouched) ──
    # Tighten ONLY the terminal +250 'safe' bonus gate. The metric landing_*
    # fields are unchanged and still feed the evaluator phase_corrected_safe /
    # strict-Jiang computations. Defaults reproduce the standard reward.
    reward_tilt_cliff: float = 0.10
    reward_speed_cliff: float = 4.0
    reward_radius_cliff: float = 8.0
    # ── C06 Jiang speed-knee terminal reward (reward-shaping only, no lateral bonus) ──
    reward_tilt_knee_rad: float = 0.05      # terminal-reward tilt knee (treat 0.05 / revert 0.10)
    reward_tilt_bonus_on: float = 1.0       # tilt-pass-shaped bonus master switch
    reward_tilt_bonus_scale: float = 50.0   # tilt bonus magnitude at tilt=0 (0 disables)
    reward_speed_knee_mps: float = 2.0      # terminal-reward speed knee (treat 2.0 = Jiang gate / revert 4.0)
    reward_speed_bonus_on: float = 1.0      # speed-pass-shaped bonus master switch
    reward_speed_bonus_scale: float = 50.0  # speed bonus magnitude at speed=0 (0 disables)
    reward_lateral_bonus_on: float = 0.0    # lateral-capture bonus master switch (C06 default OFF)
    reward_lateral_bonus_scale: float = 50.0  # lateral bonus magnitude at lateral=0 (only if on)
    reward_lateral_floor_m: float = 3.0     # lateral regularization floor for inverse-proportional bonus (m)
    max_tilt: float = 1.45
    max_lateral: float = 1800.0

    # Residual-guided PPO. The policy controls a small residual around this
    # deterministic powered-descent prior. This makes the 2000 m task learnable
    # on CPU while preserving a PPO training loop.
    guidance_on: float = 1.0
    guidance_tgo: float = 30.0
    # Lateral velocity-damping coefficient in the ZEM/ZEV lateral law
    # (canonical 4.0 -> acc_lat = -6*pos/tgo^2 - 4*vel/tgo). Raising it damps
    # residual lateral velocity harder across the whole descent.
    guidance_lat_vel_k: float = 4.0
    # ── Decoupled lateral-horizon attitude-authority arm (gen22_pod11) ──
    # Lateral (y,z) ZEM/ZEV uses its own time-to-go so the lateral capture can
    # be completed earlier/smoother than the vertical braking law. Defaults
    # reproduce the parent exactly (tgo_lat == tgo, decreasing schedule).
    guidance_tgo_lat: float = 30.0
    # 0 = decreasing schedule clip(tgo_lat - t, floor, tgo_lat) (parent-like);
    # 1 = constant lateral horizon (fixed tgo_lat for the whole descent).
    guidance_tgo_lat_mode: float = 0.0
    guidance_kp: float = 8.0
    guidance_kd: float = 20.0
    # Labeled roll-axis RCS derivative gain (gen0 roll-kd repair): high
    # derivative gain + explicit-Euler on the weakly-restored roll axis
    # produces a period-2 omega_x limit cycle. 2.0 removes the artifact;
    # pitch/yaw keep guidance_kd=20.0.
    guidance_kd_roll: float = 2.0
    residual_scale: float = 0.05
    residual_gimbal: float = 1.0
    residual_throttle: float = 1.0
    residual_rcs: float = 1.0
    residual_gridfin: float = 1.0
    # ── Per-channel residual action mask (DIG C01 / H_g47_02) ──
    residual_rcs_x: float = 1.0
    residual_rcs_y: float = 1.0
    residual_rcs_z: float = 1.0
    # ── Far-outer zero-tail residual mask (gen54_pod9 C01 compose) ──
    # A radial-bin-gated residual-action mask: residual authority is zeroed for
    # trajectories whose INITIAL lateral radius lies in the far-outer bin
    # (initial_r >= radius). Removes the fuel2.0 outward-rim slide-out tail
    # introduced by the learned residual while preserving inner-disk behavior.
    far_outer_mask_on: float = 0.0        # master switch (1 = zero residual in far-outer)
    far_outer_mask_radius: float = 1124.0  # radial boundary on initial_r (Definition B)
    far_outer_mask_width: float = 60.0    # smoothstep ramp width (m)
    far_outer_mask_mode: str = "soft"     # "soft" = smoothstep, "hard" = step
    # ── Radius-gated axis-switch lateral damper (DIG C04, gen37_pod4) ──
    # A small lateral-acceleration damper whose AXIS switches by radius band:
    #   boundary band r in [axis_switch_boundary_lo, axis_switch_switch_r)
    #       -> boundary_axis damping (downrange default)
    #   far-outer band r >= axis_switch_switch_r -> far_axis damping
    #       (crossrange default)
    #   inner disk r < boundary_lo -> damper OFF.
    # It is added to the ZEM/ZEV lateral command (post-bridge), so the RCS
    # attitude loop tracks it through body tilt — NO direct gimbal torque.
    axis_switch_damper_on: float = 0.0           # master switch
    axis_switch_damper_k: float = 0.5            # max lateral accel (m/s^2)
    axis_switch_damper_sat_m: float = 3.0        # position-error saturation (m)
    axis_switch_damper_tilt_cap: float = 0.06    # hard tilt cap (rad)
    axis_switch_boundary_lo: float = 920.0
    axis_switch_switch_r: float = 1124.0
    axis_switch_boundary_axis: float = 0.0       # 0=downrange 1=isotropic 2=crossrange
    axis_switch_far_axis: float = 2.0            # 0=downrange 1=isotropic 2=crossrange
    # ── Outward-rim radial-gated tilt-demand-softening divert (DIG C01 / H_g47_02) ──
    divert_on: float = 1.0               # master switch (1=treat, 0=control)
    divert_gate_on: float = 1.0          # 1=radial-gated (contract), 0=gate off
    divert_sign: float = -1.0            # -1 softens outward-nulling tilt demand (treat); +1 naive null (falsified)
    divert_gain: float = 0.5             # outward radial-velocity gain (1/s)
    divert_gate_radius: float = 1124.0   # radial activation boundary (m)
    divert_gate_width: float = 60.0      # radial ramp width (m)
    divert_vr_only: float = 1.0          # 1=null only vr>0, 0=damp all radial velocity
    divert_alt_gate_on: float = 1.0      # 1=altitude gate on, 0=off
    divert_alt_low_m: float = 80.0       # below this height divert fully OFF
    divert_alt_high_m: float = 400.0     # above this height divert fully ON
    divert_tilt_cap_rad: float = 0.25    # hard lateral-acc tilt cap (rad)
    divert_placebo_axis: float = 0.0     # 0=radial treat, 1=tangential (cross-range) placebo, 2=vertical (throttle) placebo
    # ── Fuel-margin-keyed schedule arms (gen52_pod9 C01/C02) ────────────
    # C01: fuel-keyed multiplier on the cross-range tangential prior gain.
    #   Identity at fuel1.5 (hard plateau), eases at fuel0.7, raises at fuel2.0.
    divert_gain_sched_on: float = 0.0     # 1 = fuel-keyed gain schedule active
    divert_gain_sched_lo: float = 0.4     # gain multiplier at fuel0.7 (ease)
    divert_gain_sched_hi: float = 1.6     # gain multiplier at fuel2.0 (raise)
    # C02: fuel-keyed vertical descent commitment (converts fuel2.0
    #   hover-never-touch into a bounded touchdown). Zero at fuel1.5/fuel0.7.
    descent_commit_on: float = 0.0        # 1 = descent commitment active
    descent_commit_hi: float = 1.0        # commitment multiplier at fuel2.0
    descent_commit_gain: float = 1.0      # downward acc bias (m/s^2) at full mult
    descent_commit_alt: float = 80.0      # activation altitude (m)
    descent_commit_gate_radius: float = 0.0  # 0=all radii; >0 = only r >= gate
    # C03-adjacent: fuel-margin-keyed landing-gear contact-height schedule.
    # Identity at fuel1.5 (4.5 m anchor), shorter at fuel2.0 to clear the
    # gear contact-height knife-edge for the heavier vehicle.
    gear_h_sched_on: float = 0.0        # 1 = fuel-keyed gear height schedule
    gear_h_sched_lo_mult: float = 1.0   # multiplier at fuel0.7 (leave alone)
    gear_h_sched_hi_mult: float = 0.667 # multiplier at fuel2.0 (4.5 -> 3.0 m)

    # ── H_g69_04 guidance-prior gear-sink throttle tail ─────────────────
    # The canonical composed_stack_rk4 prior settles into a ~2.4 m AGL gear-sink
    # hover and burns the full propellant tank holding it. This lever scales the
    # guidance-prior throttle command toward `fuel_tail_scale` in the gear-sink
    # zone (altitude < fuel_tail_alt_m AND vertical speed < fuel_tail_vx_max_mps),
    # letting gravity + gear damper complete the sink instead of hovering.
    # fuel_tail_on=0 reproduces the parent exactly (matched-control arm).
    fuel_tail_on: float = 0.0          # 1 = gear-sink throttle tail active
    fuel_tail_alt_m: float = 6.0       # AGL below which the tail may engage (m)
    fuel_tail_vx_max_mps: float = 1.0  # engage only while vx < this (descending/slow)
    fuel_tail_scale: float = 0.0       # throttle multiplier in the tail zone (0=full cut)
    fuel_tail_tilt_gate_rad: float = 0.03   # engage only when |tilt| < this (already verticalized)
    fuel_tail_lateral_gate_m: float = 2.5   # engage only when lateral error < this (captured)
    fuel_tail_vlat_gate_mps: float = 0.5    # engage only when |v_lat| < this (captured)
    fuel_tail_max_initial_r: float = 1124.0 # engage only when initial_r < this (inner disk)
    # ── Second-harness-family integrator toggle (H_g28_02) ──
    # 'euler'    = canonical explicit-Euler velocity/omega + exponential-map quat.
    # 'rk2'      = Heun predictor-corrector velocity/omega + exponential-map quat.
    # 'rk2_full' = full midpoint RK2 (Lie-group midpoint quaternion update).
    # 'euler_half' = two canonical-euler half-steps (dt/2, convergence ladder).
    # 'euler_quarter' = four canonical-euler quarter-steps (dt/4).
    # 'rk4' = classical RK4 reference.
    integrator: str = "euler"
    # Fixed-cadence finite-dt substep ladder (gen71_pod3 C05). Subdivides
    # the 0.1 s physics macro step into `integrator_substeps` sub-integrations
    # while control resolution stays at 10 Hz (resolved action held fixed).
    # 1 = canonical single-step (bit-identical); 2/4 = dt 0.05 / 0.025 rungs.
    # Only the rk2 / rk2_full / rk4 branches honor it.
    integrator_substeps: int = 1

    # ── Engine-gimbal attitude augmentation (H_g23_06 / gen24_pod7 port) ──
    # Deterministic gimbal PD attitude actuator layered on the guidance prior
    # (the prior normally keeps gimbal at zero). Torque-matched arm: kp=0.20,
    # kd=0.0, max=0.014 rad. Sign map (torque_engine = cross([-lever,0,0],F_B)):
    #   pitch torque_y = +lever*T*gz  -> gz = +kp*err_B[1] - kd*omega[1]
    #   yaw   torque_z = -lever*T*gy  -> gy = -kp*err_B[2] + kd*omega[2]
    # The same deflection injects a body-y/z lateral force (T*[gy,gz]).
    guidance_gimbal_on: float = 0.0             # master switch (0 = baseline, 1 = active)
    guidance_gimbal_kp: float = 0.10            # gimbal deflection (rad) per rad attitude error
    guidance_gimbal_kd: float = 0.0             # gimbal deflection (rad) per rad/s body rate
    guidance_gimbal_max: float = 0.0873         # gimbal deflection limit (rad)
    guidance_gimbal_feedforward: float = 0.0    # 1 = subtract gimbal lateral accel from commanded lateral accel

    # ── Terminal-only coast-drag ramp (H_g19_05 repair / DIG C01 assembly) ──
    # Gates ONLY the fin DRAG channel (F_fin_drag_B); passive lift and active
    # grid-fin control force are left unreshaped. The ramp is OFF above
    # terminal_drag_alt_high (mid-disk coast: unity drag) and ramps to
    # terminal_drag_gain below terminal_drag_alt_low (h<300 m terminal phase),
    # clipping terminal-speed p95 without touching mid-disk capture.
    terminal_drag_on: float = 1.0         # master switch (1 = terminal ramp active, 0 = baseline unity drag)
    terminal_drag_gain: float = 3.0       # fin-drag multiplier in the terminal phase (h < terminal_drag_alt_low) — coast-drag g3 arm
    terminal_drag_alt_low: float = 300.0  # below this height (m) the ramp is fully engaged
    terminal_drag_alt_high: float = 350.0 # above this height (m) the ramp is fully off (unity drag)
    # Radial-gated coast-drag (gen23_pod11 f2ed5bba/9d055fe3 follow-up). When
    # terminal_drag_radial_gate_on>0.5, the terminal drag gain blends from
    # terminal_drag_gain inside terminal_drag_far_radius to terminal_drag_far_gain
    # outside it (sigmoid over terminal_drag_radial_width), gated on the INITIAL
    # lateral radius (state[15]), not the current lateral radius.
    terminal_drag_radial_gate_on: float = 0.0   # 1 = radial-gated, 0 = uniform terminal_drag_gain
    terminal_drag_far_gain: float = 5.0         # far-outer (r>far_radius) drag multiplier
    terminal_drag_far_radius: float = 1124.0    # radial boundary of the gate (m)
    terminal_drag_radial_width: float = 60.0    # sigmoid blend width across the radius (m)

    # ── Regime-gated lift-down (C08 deployability filter, ported to fin_x=0) ──
    # The passive fin LIFT channel is scaled by gate_lift_scale ONLY inside the
    # altitude / absolute-tilt regime and falls back to full lift (1.0) outside.
    # This is the deployable form of the lift0.5 lever: it keeps full lift
    # authority in the terminal attitude-nulling phase and far-outer descent.
    regime_gate_on: float = 0.0           # master switch (1 = gate active, 0 = global base lift)
    gate_lift_scale: float = 1.0          # engaged lift scale inside the regime (<1)
    gate_h_min_m: float = 80.0            # regime altitude lower bound (m)
    gate_h_max_m: float = 2000.0          # regime altitude upper bound (m)
    gate_tilt_max_rad: float = 0.30       # regime |tilt| upper bound (rad)

    # ── Capture-preserving terminal bridge (gen2_pod13) ─────────────────
    # Below bridge_terminal_h m the champion ZEM/ZEV capture law is PRESERVED
    # (full 3-D lateral+vertical nulling for the whole descent) and augmented
    # with (a) closed-loop vertical-speed nulling toward -bridge_v_td and (b) a
    # gradual attitude verticalization (blend the desired thrust direction
    # toward body-up). Unlike the fuel-aware hover-slam replacement, this does
    # NOT abandon outer-disk lateral authority: it only verticalizes once the
    # ZEM/ZEV capture has already pulled lateral error down near the pad.
    bridge_on: float = 1.0
    bridge_terminal_h: float = 80.0     # verticalization/lateral-taper activation altitude (m) — leader geometry (fin_x=1/h80)
    bridge_v_td: float = 1.0            # soft-touchdown vertical speed (m/s)
    bridge_speed_k: float = 0.3         # vertical-speed nulling gain (1/s), gentle
    bridge_verticalize_gain: float = 1.0  # lateral-acc taper at h=0 (1.0 -> pure vertical)
    bridge_tgo_floor: float = 3.0       # ZEM/ZEV tgo lower clip (champion = 3.0)
    # ── Terminal lateral-velocity nulling split (gen22_pod11 session_002) ──
    # The bridge tapers the whole lateral ZEM/ZEV command to zero below
    # bridge_terminal_h. Split the taper so the lateral POSITION (zem) term is
    # tapered independently of the lateral VELOCITY (zev) nulling term. Keeping
    # velocity nulling active to the ground kills residual lateral velocity,
    # which is the dominant boundary r920-1124 speed/lateral binding constraint.
    bridge_pos_taper: float = 1.0    # taper on lateral POSITION (zem) term (1 = current full taper)
    bridge_vel_taper: float = 1.0    # taper on lateral VELOCITY (zev) term (1 = current full taper)
    bridge_lat_vel_gain: float = 1.0 # extra multiplier on the lateral velocity-nulling term

    # ── Variant-local TV-LQR-family guidance selector (gen30_pod2 C06) ──
    # Screen arms select guidance_mode; 'canonical' reproduces the matched
    # control exactly. Never touches evaluator/metric/split/disk_radius.
    guidance_mode: str = 'canonical'


    # ── Passive landing-gear suspension contact model (gen42_pod15 C01, H_g42_02) ──
    # N legs placed symmetrically about the body -x (tail/engine) axis. Each leg
    # tip sits gear_contact_height_m below the COM (body -x) and
    # gear_footprint_radius_m outward in the body y-z plane. When a tip penetrates
    # the ground (inertial tip height < 0), a spring-damper normal reaction plus
    # Coulomb-like friction is applied at that tip, generating a restoring torque.
    # gear_on=0 is the default and makes the contact path bitwise-identical to the
    # parent uniform-g3@rk4 floor.
    gear_on: float = 0.0
    gear_contact_height_m: float = 3.0      # tip drop below COM (body -x)
    gear_footprint_radius_m: float = 6.0    # tip radial offset in body y-z plane
    gear_n_legs: int = 4
    gear_spring_k_npm: float = 7.0e4        # normal spring stiffness (N/m) per leg
    gear_damper_c_nspm: float = 4.0e4       # normal damper (N/(m/s)) per leg
    gear_friction_mu: float = 0.6           # Coulomb-like tangential friction
    gear_contact_restore_scale: float = 1.0 # restoring-torque scale (0 = zero torque ablation)
    gear_bottom_out_guard: float = 0.0     # 1 = cap spring support below weight so the gear always settles to ground (bottom-out guard)
    gear_bottom_out_force_frac: float = 0.95 # fraction of per-leg weight the spring may support when guard on

    # ── gen46_pod9 DIG C07: dose-saturation margin guard (deployability filter) ──
    dose_guard_on: float = 1.0
    dose_guard_band_lo: float = 2.75
    dose_guard_band_hi: float = 3.25
    dose_guard_sat_threshold: float = 3.25
    dose_guard_backoff_mode: str = "saturate"
    dose_guard_scope: str = "all"


# Resolved grid-fin station + authority config (module-level so step_one traces
# bake the same constants that main() reports). Per-axis lateral force slopes
# let the sweep scale pitch authority independently of yaw. Roll damping
# (fin_torque_damping, grid_fin_roll_control_cl) is kept at baseline.
_AUTH = _grid_fin_aero.authority()
_CL_Y = _AUTH.get("grid_fin_control_cl_y", _AUTH["grid_fin_control_cl"])
_CL_Z = _AUTH.get("grid_fin_control_cl_z", _AUTH["grid_fin_control_cl"])
_CL_ROLL = _AUTH["grid_fin_roll_control_cl"]
_CL_MAX = _AUTH["grid_fin_control_max"]
_FIN_STATION_X = _AUTH["fin_station_x"]
_FIN_DAMPING = _AUTH["fin_torque_damping"]

# State vector layout:
# [x,y,z, vx,vy,vz, q0,q1,q2,q3, wx,wy,wz, mass, time]
STATE_DIM = 15
OBS_DIM = 16
ACT_DIM = 9
DISK_RADIUS = 1500.0


def quat_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return jnp.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def axis_angle_quat(axis: jax.Array, angle: float | jax.Array) -> jax.Array:
    axis = axis / jnp.maximum(jnp.linalg.norm(axis), 1e-8)
    h = 0.5 * angle
    return jnp.concatenate([jnp.array([jnp.cos(h)]), jnp.sin(h) * axis])


def euler_xyz_quat(roll: float, pitch: float, yaw: float) -> jax.Array:
    # Apply x-roll, then y-pitch, then z-yaw. The environment uses x as height.
    qx = axis_angle_quat(jnp.array([1.0, 0.0, 0.0]), roll)
    qy = axis_angle_quat(jnp.array([0.0, 1.0, 0.0]), pitch)
    qz = axis_angle_quat(jnp.array([0.0, 0.0, 1.0]), yaw)
    q = quat_mul(qz, quat_mul(qy, qx))
    return q / jnp.maximum(jnp.linalg.norm(q), 1e-8)


def quat_to_rot(q: jax.Array) -> jax.Array:
    q = q / jnp.maximum(jnp.linalg.norm(q), 1e-8)
    q0, q1, q2, q3 = q
    return jnp.array([
        [q0*q0 + q1*q1 - q2*q2 - q3*q3, 2.0*(q1*q2 - q0*q3), 2.0*(q1*q3 + q0*q2)],
        [2.0*(q1*q2 + q0*q3), q0*q0 - q1*q1 + q2*q2 - q3*q3, 2.0*(q2*q3 - q0*q1)],
        [2.0*(q1*q3 - q0*q2), 2.0*(q2*q3 + q0*q1), q0*q0 - q1*q1 - q2*q2 + q3*q3],
    ])


def omega_matrix(w: jax.Array) -> jax.Array:
    wx, wy, wz = w
    return jnp.array([
        [0.0, -wx, -wy, -wz],
        [wx, 0.0, wz, -wy],
        [wy, -wz, 0.0, wx],
        [wz, wy, -wx, 0.0],
    ])


def integrate_quat_exact(q: jax.Array, omega: jax.Array, dt: float) -> jax.Array:
    """Advance attitude with an exponential-map delta quaternion.

    The older explicit-Euler quaternion update can create terminal phase
    artifacts in omega-based safety checks. This keeps the update on SO(3).

    gen31_pod8 C01 (variant-local) differentiability repair: the canonical
    ``jnp.linalg.norm(omega)`` has a 0/0 gradient at omega=0, which makes
    ``jax.jacfwd`` of the rk4 path NaN for zero-rate nominals and NaN-poisoned
    the iLQR feedback gains (a distinct source from the arccos terminal-cost
    Hessian). Adding a 1e-12 floor inside the norm makes the map smooth at
    omega=0 while leaving the forward update unchanged to ~1e-10 for realistic
    |omega| >= 1e-3 rad/s.
    """
    n = jnp.sqrt(jnp.sum(omega * omega) + 1e-12)
    angle = n * dt
    half = 0.5 * angle
    c = jnp.cos(half)
    s = jnp.sin(half)
    axis = omega / n
    dq = jnp.concatenate([jnp.array([c]), s * axis])
    q_next = quat_mul(dq, q)
    return q_next / jnp.maximum(jnp.linalg.norm(q_next), 1e-8)


def gimbal_rot(dy: jax.Array, dz: jax.Array) -> jax.Array:
    cy, sy = jnp.cos(dy), jnp.sin(dy)
    cz, sz = jnp.cos(dz), jnp.sin(dz)
    return jnp.array([
        [cy * cz, -sy, -cy * sz],
        [sy * cz,  cy, -sy * sz],
        [sz,       0.0, cz],
    ])


def inertia_diag(m: jax.Array, cfg: EnvCfg) -> jax.Array:
    ixx = 0.5 * m * cfg.radius ** 2
    iyy = (1.0 / 12.0) * m * (cfg.length ** 2 + 3.0 * cfg.radius ** 2)
    return jnp.array([ixx, iyy, iyy])


def tilt_angle_from_q(q: jax.Array) -> jax.Array:
    # Body +x axis relative to inertial +x height axis.
    R = quat_to_rot(q)
    c = jnp.clip(R[0, 0], -1.0, 1.0)
    return jnp.arccos(c)


def observe(state: jax.Array, cfg: EnvCfg) -> jax.Array:
    pos = state[0:3]
    vel = state[3:6]
    q = state[6:10]
    omega = state[10:13]
    mass = state[13]
    time = state[14]
    lateral = jnp.sqrt(pos[1] ** 2 + pos[2] ** 2 + 1e-8)
    mass_frac = (mass - cfg.mass_empty) / ((cfg.mass_full - cfg.mass_empty) * cfg.fuel_scale + 1e-8)
    return jnp.concatenate([
        pos / jnp.array([cfg.init_x, 600.0, 600.0]),
        vel / 140.0,
        q,
        omega / 0.8,
        jnp.array([mass_frac, time / cfg.max_steps, lateral / 600.0]),
    ])




# ── gen30_pod2 C06 variant-local TV-LQR-family guidance overrides ───────
# These are screen arms only (controller-only, T5_jiang_screen). They replace
# the canonical guidance prior while leaving the plant, metric, split, action
# interface, and disk_radius untouched.
_SS_K_POS = 0.03      # far-outer steady-state LQR lateral position gain (1/s^2)
_SS_K_VEL = 0.2449    # far-outer steady-state LQR lateral velocity gain (1/s)

_TVLQR_X = None
_TVLQR_U = None
_TVLQR_K = None


def _load_tvlqr_frozen():
    """Lazily load the archived iLQR reference + rk4-computed TV-LQR gains."""
    global _TVLQR_X, _TVLQR_U, _TVLQR_K
    if _TVLQR_X is not None:
        return
    import numpy as _np
    here = Path(__file__).resolve().parent / "controller"
    _TVLQR_X = jnp.array(_np.load(here / "x_ref.npy").astype(jnp.float32))
    _TVLQR_U = jnp.array(_np.load(here / "u_ref.npy").astype(jnp.float32))
    _TVLQR_K = jnp.array(_np.load(here / "Ks_rk4_g5.npy").astype(jnp.float32))


def _guidance_ss_lqr(state: jax.Array, cfg: EnvCfg) -> jax.Array:
    """Steady-state LQR terminal nuller (C01 ablation: replace TV-LQR gains
    with steady-state LQR). Lateral position/velocity nulled with constant
    closed-form double-integrator LQR gains; vertical + attitude channels are
    identical to the canonical guidance prior."""
    pos = state[0:3]
    vel = state[3:6]
    q = state[6:10]
    omega = state[10:13]
    mass = state[13]
    t = state[14] * cfg.dt

    R_BI = quat_to_rot(q)
    gravity = jnp.array([-cfg.g, 0.0, 0.0])
    tgo_x = jnp.clip(cfg.guidance_tgo - t, cfg.bridge_tgo_floor, cfg.guidance_tgo)
    zem_x = -(pos[0] + vel[0] * tgo_x + 0.5 * gravity[0] * tgo_x * tgo_x)
    zev_x = -(vel[0] + gravity[0] * tgo_x)
    acc_x = 6.0 * zem_x / (tgo_x * tgo_x) - 2.0 * zev_x / tgo_x

    acc_lat_pos = -_SS_K_POS * pos[1:3]
    acc_lat_vel = -_SS_K_VEL * vel[1:3]
    thrust_acc = jnp.concatenate([jnp.array([acc_x]), acc_lat_pos + acc_lat_vel])

    h = pos[0]
    vx = vel[0]
    a_vert_fb = cfg.bridge_speed_k * ((-cfg.bridge_v_td) - vx)
    ramp = jnp.clip(1.0 - h / cfg.bridge_terminal_h, 0.0, 1.0) ** 2
    s_lat_pos = 1.0 - cfg.bridge_verticalize_gain * cfg.bridge_pos_taper * ramp
    s_lat_vel = 1.0 - cfg.bridge_verticalize_gain * cfg.bridge_vel_taper * ramp
    use_bridge = (h < cfg.bridge_terminal_h) & (cfg.bridge_on > 0.5)
    thrust_acc = thrust_acc.at[0].add(jnp.where(use_bridge, a_vert_fb, 0.0))
    s_lat_pos = jnp.where(use_bridge, s_lat_pos, 1.0)
    s_lat_vel = jnp.where(use_bridge, s_lat_vel, 1.0)
    thrust_acc = thrust_acc.at[1].set(acc_lat_pos[0] * s_lat_pos + cfg.bridge_lat_vel_gain * acc_lat_vel[0] * s_lat_vel)
    thrust_acc = thrust_acc.at[2].set(acc_lat_pos[1] * s_lat_pos + cfg.bridge_lat_vel_gain * acc_lat_vel[1] * s_lat_vel)

    # ── C02 fuel-margin-keyed vertical descent commitment ────────────────
    # At fuel2.0 the parent prior settles into a stable hover-never-touch
    # equilibrium ~2*bridge_v_td above the pad. A small downward bias keyed to
    # the fuel cell (zero inside the fuel1.5 neighborhood) pushes the vehicle
    # through the equilibrium to touchdown without touching the fuel1.5 cell.
    if cfg.descent_commit_on > 0.5:
        commit_mult = _fuelsched.descent_commit_multiplier(cfg)
        commit_r = jnp.linalg.norm(pos[1:3]) + 1e-8
        commit_gate = (h < cfg.descent_commit_alt) & (h > 1e-4)
        if cfg.descent_commit_gate_radius > 0.0:
            commit_gate = commit_gate & (commit_r >= cfg.descent_commit_gate_radius)
        thrust_acc = thrust_acc.at[0].add(
            -cfg.descent_commit_gain * commit_mult
            * jnp.where(commit_gate, 1.0, 0.0))

    # ── Radius-gated axis-switch damper (post-bridge lateral-acc residual) ──
    initial_r = state[15]
    lateral_r = jnp.linalg.norm(pos[1:3]) + 1e-8
    dr_axis = jnp.array([cfg.init_vy, cfg.init_vz])
    dr_axis = dr_axis / (jnp.linalg.norm(dr_axis) + 1e-8)
    cr_axis = jnp.array([-dr_axis[1], dr_axis[0]])
    radial_axis = pos[1:3] / lateral_r
    axis_mode = jnp.where(lateral_r < cfg.axis_switch_switch_r,
                          cfg.axis_switch_boundary_axis, cfg.axis_switch_far_axis)
    axis_unit = jnp.where(
        axis_mode < 0.5, dr_axis,
        jnp.where(axis_mode < 1.5, radial_axis, cr_axis))
    pos_err_axis = pos[1] * axis_unit[0] + pos[2] * axis_unit[1]
    u_dr = -cfg.axis_switch_damper_k * jnp.clip(
        pos_err_axis / cfg.axis_switch_damper_sat_m, -1.0, 1.0)
    u_dr_vec = u_dr * axis_unit
    max_dr_acc = jnp.tan(cfg.axis_switch_damper_tilt_cap) * jnp.maximum(
        jnp.abs(acc_x), 1.0)
    u_dr_norm = jnp.linalg.norm(u_dr_vec) + 1e-8
    u_dr_vec = u_dr_vec * jnp.minimum(1.0, max_dr_acc / u_dr_norm)
    dr_gate = (cfg.axis_switch_damper_on > 0.5) & (lateral_r >= cfg.axis_switch_boundary_lo)
    u_dr_vec = jnp.where(dr_gate, u_dr_vec, jnp.zeros_like(u_dr_vec))
    thrust_acc = thrust_acc.at[1].add(u_dr_vec[0])
    thrust_acc = thrust_acc.at[2].add(u_dr_vec[1])

    max_thrust_acc = cfg.thrust_max / mass
    acc_norm = jnp.linalg.norm(thrust_acc) + 1e-8
    thrust_acc = thrust_acc * jnp.minimum(1.0, max_thrust_acc / acc_norm)
    acc_norm = jnp.linalg.norm(thrust_acc) + 1e-8

    throttle = jnp.clip(acc_norm / max_thrust_acc, 0.0, 1.0)
    desired_body_x_I = thrust_acc / acc_norm
    current_body_x_I = R_BI[:, 0]
    err_I = jnp.cross(current_body_x_I, desired_body_x_I)
    err_B = R_BI.T @ err_I
    kd_vec = jnp.array([cfg.guidance_kd_roll, cfg.guidance_kd, cfg.guidance_kd], dtype=omega.dtype)
    rcs = jnp.clip(cfg.guidance_kp * err_B - kd_vec * omega, -1.0, 1.0)
    return jnp.concatenate([jnp.zeros(2), jnp.array([2.0 * throttle - 1.0]), rcs, jnp.zeros(3)])


def _guidance_tvlqr_frozen(state: jax.Array, cfg: EnvCfg) -> jax.Array:
    """Frozen-reference feedforward + time-varying LQR tracking.

    u = u_ref[t] - Ks[t] @ (state - x_ref[t]), clipped to the action box.
    The reference is the archived worst-azimuth r=1200 m iLQR trajectory and
    Ks[t] are the rk4 backward-Riccati gains (scripts/build_tvlqr_gains.py).
    Known to have only a ~5 m closed-loop recovery basin (0862947e)."""
    _load_tvlqr_frozen()
    T = _TVLQR_U.shape[0]
    idx = jnp.clip(state[14].astype(jnp.int32), 0, T - 1)
    err = state - _TVLQR_X[idx]
    a = _TVLQR_U[idx] - _TVLQR_K[idx] @ err
    return jnp.clip(a, -1.0, 1.0)


def guidance_action(state: jax.Array, cfg: EnvCfg) -> jax.Array:
    """Deterministic powered-descent guidance prior.

    The prior uses a ZEM/ZEV-like acceleration command in the inertial frame,
    then uses RCS to align body +x with the requested thrust vector. Gimbal is
    intentionally kept at zero in the prior: in this toy model a gimbaled engine
    has a long lever arm and can create very large torques. PPO may still add a
    small residual gimbal correction through the action residual.
    """
    mode = cfg.guidance_mode
    if mode == "ss_lqr":
        return _guidance_ss_lqr(state, cfg)
    if mode == "tvlqr_frozen":
        return _guidance_tvlqr_frozen(state, cfg)

    pos = state[0:3]
    vel = state[3:6]
    q = state[6:10]
    omega = state[10:13]
    mass = state[13]
    t = state[14] * cfg.dt

    R_BI = quat_to_rot(q)
    gravity = jnp.array([-cfg.g, 0.0, 0.0])
    target_pos = jnp.zeros(3)
    target_vel = jnp.zeros(3)
    # Vertical (x) keeps the parent coupled-horizon law exactly.
    tgo_x = jnp.clip(cfg.guidance_tgo - t, cfg.bridge_tgo_floor, cfg.guidance_tgo)
    # Lateral (y,z) horizon is decoupled: either a decreasing schedule with its
    # own top value, or a constant horizon. Defaults trace the parent.
    tgo_lat = jnp.where(
        cfg.guidance_tgo_lat_mode > 0.5,
        cfg.guidance_tgo_lat,
        jnp.clip(cfg.guidance_tgo_lat - t, cfg.bridge_tgo_floor, cfg.guidance_tgo_lat),
    )

    # Gravity acts only on the vertical axis, so the lateral ZEM/ZEV reduces to
    # the pure position+velocity nulling law; the vertical axis keeps the exact
    # canonical gravity-augmented ZEM/ZEV.
    zem_x = target_pos[0] - (pos[0] + vel[0] * tgo_x + 0.5 * gravity[0] * tgo_x * tgo_x)
    zev_x = target_vel[0] - (vel[0] + gravity[0] * tgo_x)
    acc_x = 6.0 * zem_x / (tgo_x * tgo_x) - 2.0 * zev_x / tgo_x

    # Net lateral ZEM/ZEV command, decomposed into a position-restoring term and
    # a velocity-damping term (the two algebraic pieces that together give the
    # canonical law -6*pos/tgo^2 - 4*vel/tgo). Keeping only the position term
    # would anti-damp velocity; keeping only the velocity term is a bounded,
    # sign-correct damping law suitable for the terminal phase.
    acc_lat_pos = -6.0 * pos[1:3] / (tgo_lat * tgo_lat)
    acc_lat_vel = -cfg.guidance_lat_vel_k * vel[1:3] / tgo_lat
    acc_lat = acc_lat_pos + acc_lat_vel
    thrust_acc = jnp.concatenate([jnp.array([acc_x]), acc_lat])

    # ── Capture-preserving terminal bridge ────────────────────────────────
    # (a) Vertical-speed nulling: gentle proportional feedback on the body-up
    #     axis drives descent speed toward -bridge_v_td before touchdown.
    # (b) Lateral-acc taper: as h -> 0 the lateral ZEM/ZEV components are
    #     tapered to zero (not a separate law), so the commanded thrust
    #     verticalizes while the ZEM/ZEV lateral capture remains active for
    #     the whole descent. The squared ramp keeps the onset gentle.
    h = pos[0]
    vx = vel[0]
    a_vert_fb = cfg.bridge_speed_k * ((-cfg.bridge_v_td) - vx)
    ramp = jnp.clip(1.0 - h / cfg.bridge_terminal_h, 0.0, 1.0) ** 2
    s_lat_pos = 1.0 - cfg.bridge_verticalize_gain * cfg.bridge_pos_taper * ramp
    s_lat_vel = 1.0 - cfg.bridge_verticalize_gain * cfg.bridge_vel_taper * ramp
    use_bridge = (h < cfg.bridge_terminal_h) & (cfg.bridge_on > 0.5)
    thrust_acc = thrust_acc.at[0].add(jnp.where(use_bridge, a_vert_fb, 0.0))
    s_lat_pos = jnp.where(use_bridge, s_lat_pos, 1.0)
    s_lat_vel = jnp.where(use_bridge, s_lat_vel, 1.0)
    # Independent position/velocity taper: default (pos_taper=1, vel_taper=1,
    # vel_gain=1) reproduces the parent exactly (both terms share the same
    # taper), while vel_taper=0 keeps lateral-velocity nulling active to the
    # ground so residual lateral velocity is cancelled near touchdown.
    thrust_acc = thrust_acc.at[1].set(acc_lat_pos[0] * s_lat_pos + cfg.bridge_lat_vel_gain * acc_lat_vel[0] * s_lat_vel)
    thrust_acc = thrust_acc.at[2].set(acc_lat_pos[1] * s_lat_pos + cfg.bridge_lat_vel_gain * acc_lat_vel[1] * s_lat_vel)

    # ── C02 fuel-margin-keyed vertical descent commitment ────────────────
    # At fuel2.0 the parent prior settles into a stable hover-never-touch
    # equilibrium ~2*bridge_v_td above the pad. A small downward bias keyed to
    # the fuel cell (zero inside the fuel1.5 neighborhood) pushes the vehicle
    # through the equilibrium to touchdown without touching the fuel1.5 cell.
    if cfg.descent_commit_on > 0.5:
        commit_mult = _fuelsched.descent_commit_multiplier(cfg)
        commit_r = jnp.linalg.norm(pos[1:3]) + 1e-8
        commit_gate = (h < cfg.descent_commit_alt) & (h > 1e-4)
        if cfg.descent_commit_gate_radius > 0.0:
            commit_gate = commit_gate & (commit_r >= cfg.descent_commit_gate_radius)
        thrust_acc = thrust_acc.at[0].add(
            -cfg.descent_commit_gain * commit_mult
            * jnp.where(commit_gate, 1.0, 0.0))

    # ── Radius-gated axis-switch damper (post-bridge lateral-acc residual) ──
    initial_r = state[15]
    lateral_r = jnp.linalg.norm(pos[1:3]) + 1e-8
    dr_axis = jnp.array([cfg.init_vy, cfg.init_vz])
    dr_axis = dr_axis / (jnp.linalg.norm(dr_axis) + 1e-8)
    cr_axis = jnp.array([-dr_axis[1], dr_axis[0]])
    radial_axis = pos[1:3] / lateral_r
    axis_mode = jnp.where(lateral_r < cfg.axis_switch_switch_r,
                          cfg.axis_switch_boundary_axis, cfg.axis_switch_far_axis)
    axis_unit = jnp.where(
        axis_mode < 0.5, dr_axis,
        jnp.where(axis_mode < 1.5, radial_axis, cr_axis))
    pos_err_axis = pos[1] * axis_unit[0] + pos[2] * axis_unit[1]
    u_dr = -cfg.axis_switch_damper_k * jnp.clip(
        pos_err_axis / cfg.axis_switch_damper_sat_m, -1.0, 1.0)
    u_dr_vec = u_dr * axis_unit
    max_dr_acc = jnp.tan(cfg.axis_switch_damper_tilt_cap) * jnp.maximum(
        jnp.abs(acc_x), 1.0)
    u_dr_norm = jnp.linalg.norm(u_dr_vec) + 1e-8
    u_dr_vec = u_dr_vec * jnp.minimum(1.0, max_dr_acc / u_dr_norm)
    dr_gate = (cfg.axis_switch_damper_on > 0.5) & (lateral_r >= cfg.axis_switch_boundary_lo)
    u_dr_vec = jnp.where(dr_gate, u_dr_vec, jnp.zeros_like(u_dr_vec))
    thrust_acc = thrust_acc.at[1].add(u_dr_vec[0])
    thrust_acc = thrust_acc.at[2].add(u_dr_vec[1])

    # ── Outward-rim radial-gated tilt-demand-softening divert (C01/H_g47_02) ──
    # C01 fuel-margin-keyed gain schedule on the cross-range tangential prior.
    divert_gain_scale = _fuelsched.divert_gain_multiplier(cfg) if (
        cfg.divert_gain_sched_on > 0.5) else 1.0
    a_div = _divert.compute_divert(pos, vel, state[15], cfg,
                                   gain_scale=divert_gain_scale)
    max_div_acc = _divert.tilt_cap_scale(jnp.abs(acc_x), cfg)
    a_div_norm = jnp.linalg.norm(a_div) + 1e-8
    a_div = a_div * jnp.minimum(1.0, max_div_acc / a_div_norm)
    # ── Placebo routing (attribution falsifier, gen49_pod5) ──
    # divert_placebo_axis 0 = radial treat (contract), 1 = tangential/cross-range
    # placebo (identical magnitude + gates, rotated 90 deg so it cannot shed the
    # radial lateral error), 2 = vertical throttle placebo (|a_div| added to
    # vertical acc instead of the lateral plane). OFF is divert_on=0.
    a_tang = jnp.stack([-a_div[1], a_div[0]])
    a_vert = jnp.linalg.norm(a_div) + 1e-8
    a_lat = jnp.where(
        cfg.divert_placebo_axis < 0.5, a_div,
        jnp.where(cfg.divert_placebo_axis < 1.5, a_tang, jnp.zeros_like(a_div)),
    )
    thrust_acc = thrust_acc.at[1].add(a_lat[0])
    thrust_acc = thrust_acc.at[2].add(a_lat[1])
    thrust_acc = thrust_acc.at[0].add(
        jnp.where(cfg.divert_placebo_axis >= 1.5, a_vert, 0.0)
    )

    max_thrust_acc = cfg.thrust_max / mass
    acc_norm = jnp.linalg.norm(thrust_acc) + 1e-8
    thrust_acc = thrust_acc * jnp.minimum(1.0, max_thrust_acc / acc_norm)
    acc_norm = jnp.linalg.norm(thrust_acc) + 1e-8

    throttle = jnp.clip(acc_norm / max_thrust_acc, 0.0, 1.0)
    desired_body_x_I = thrust_acc / acc_norm
    current_body_x_I = R_BI[:, 0]

    # Small-angle attitude error: rotate current body +x toward desired thrust.
    err_I = jnp.cross(current_body_x_I, desired_body_x_I)
    err_B = R_BI.T @ err_I
    kd_vec = jnp.array([cfg.guidance_kd_roll, cfg.guidance_kd, cfg.guidance_kd], dtype=omega.dtype)
    rcs = jnp.clip(cfg.guidance_kp * err_B - kd_vec * omega, -1.0, 1.0)

    # ── Engine-gimbal attitude augmentation (H_g23_06 / gen24_pod7 port) ──
    # gimbal PD attitude actuator, sign map in EnvCfg docstring. Deflection is
    # in radians and clipped to guidance_gimbal_max. When feedforward is on,
    # the induced body-y/z lateral acceleration is subtracted from the
    # commanded thrust_acc so body-tilt + gimbal do not double-count lateral
    # capture (prevents the documented off-pad overshoot).
    gy_cmd = -cfg.guidance_gimbal_kp * err_B[2] + cfg.guidance_gimbal_kd * omega[2]
    gz_cmd =  cfg.guidance_gimbal_kp * err_B[1] - cfg.guidance_gimbal_kd * omega[1]
    gimbal_cmd = jnp.clip(
        jnp.stack([gy_cmd, gz_cmd]), -cfg.guidance_gimbal_max, cfg.guidance_gimbal_max
    ) * cfg.guidance_gimbal_on

    gy_app = gimbal_cmd[0]
    gz_app = gimbal_cmd[1]
    gimbal_lat_acc_B = acc_norm * jnp.stack([jnp.zeros_like(gy_app), gy_app, gz_app])
    gimbal_lat_acc_I = R_BI @ gimbal_lat_acc_B
    thrust_acc_ff = thrust_acc - cfg.guidance_gimbal_feedforward * gimbal_lat_acc_I
    acc_norm_ff = jnp.linalg.norm(thrust_acc_ff) + 1e-8
    desired_body_x_ff = thrust_acc_ff / acc_norm_ff
    err_ff_I = jnp.cross(current_body_x_I, desired_body_x_ff)
    err_ff_B = R_BI.T @ err_ff_I
    err_use = jnp.where(cfg.guidance_gimbal_feedforward > 0.5, err_ff_B, err_B)
    rcs = jnp.where(cfg.guidance_gimbal_feedforward > 0.5,
                    jnp.clip(cfg.guidance_kp * err_use - kd_vec * omega, -1.0, 1.0),
                    rcs)

    # Action layout:
    # [gimbal_y, gimbal_z, throttle, rcs_x, rcs_y, rcs_z, grid_y, grid_z, grid_roll]
    # The guidance prior does not actively schedule grid fins; PPO/controller
    # residuals may use them as additional aerodynamic actuators.
    return jnp.concatenate([gimbal_cmd, jnp.array([2.0 * throttle - 1.0]), rcs, jnp.zeros(3)])

def residual_mask(cfg: EnvCfg) -> jax.Array:
    # Per-channel residual action mask (DIG C01 / H_g47_02). rcs_y is the
    # outward-vr-nulling authority channel; rcs_x is the optional roll counter.
    return jnp.array([
        cfg.residual_gimbal, cfg.residual_gimbal,
        cfg.residual_throttle,
        cfg.residual_rcs_x, cfg.residual_rcs_y, cfg.residual_rcs_z,
        cfg.residual_gridfin, cfg.residual_gridfin, cfg.residual_gridfin,
    ], dtype=jnp.float32)


def apply_guidance_residual(state: jax.Array, residual_action: jax.Array, cfg: EnvCfg) -> jax.Array:
    residual_action = jnp.clip(residual_action, -1.0, 1.0) * residual_mask(cfg)
    # DIG C04 fuel-margin-keyed residual action mask (deployability filter).
    # `guard` => zero residual authority inside the fuel1.5 neighborhood,
    # identity outside it (fuel0.7/2.0 extremes). Scalar per fuel cell.
    fuel_gate = jnp.asarray(
        _fuelmask.fuel_mask_value(
            float(cfg.fuel_scale),
            mode=cfg.fuel_mask_mode,
            center=float(cfg.fuel_mask_center),
            halfwidth=float(cfg.fuel_mask_halfwidth),
        ),
        dtype=jnp.float32,
    )
    residual_action = residual_action * fuel_gate
    # gen54_pod9 C01 compose: far-outer zero-tail residual mask gated by the
    # INITIAL radial bin (state[15] == initial_r). Zeroes learned residual
    # authority in the far-outer bin so those trajectories follow the
    # deterministic prior exactly (removes the outward-rim slide-out tail).
    if cfg.far_outer_mask_on > 0.5:
        far_gate = _dosemask.far_outer_zero_tail_gate(
            state[15],
            radius=float(cfg.far_outer_mask_radius),
            width=float(cfg.far_outer_mask_width),
            mode=cfg.far_outer_mask_mode,
        )
        residual_action = residual_action * far_gate
    prior = guidance_action(state, cfg)
    # ── H_g69_04 guidance-prior gear-sink throttle tail ──────────────────
    # Scale the deterministic-prior throttle channel in the gear-sink zone so
    # the vehicle sinks through the gear contact instead of hovering until fuel
    # exhaustion. Applied to the PRIOR only; the learned residual still adds on
    # top afterwards. fuel_tail_on=0 reproduces the parent byte-for-byte.
    if cfg.fuel_tail_on > 0.5:
        _tail_throttle = 0.5 * (prior[2] + 1.0)
        _tail_tilt = tilt_angle_from_q(state[6:10])
        _tail_lat = jnp.linalg.norm(state[1:3]) + 1e-8
        _tail_vlat = jnp.linalg.norm(state[4:6]) + 1e-8
        _tail_gate = (
            (state[15] < cfg.fuel_tail_max_initial_r)
            & (state[0] < cfg.fuel_tail_alt_m)
            & (state[3] < cfg.fuel_tail_vx_max_mps)
            & (_tail_tilt < cfg.fuel_tail_tilt_gate_rad)
            & (_tail_lat < cfg.fuel_tail_lateral_gate_m)
            & (_tail_vlat < cfg.fuel_tail_vlat_gate_mps)
        )
        _tail_throttle = _tail_throttle * jnp.where(_tail_gate, cfg.fuel_tail_scale, 1.0)
        prior = prior.at[2].set(jnp.clip(2.0 * _tail_throttle - 1.0, -1.0, 1.0))
    guided = jnp.clip(prior + cfg.residual_scale * residual_action, -1.0, 1.0)
    return jnp.where(cfg.guidance_on > 0.5, guided, residual_action)

def fin_terminal_drag_fraction(altitude: jax.Array, cfg: EnvCfg) -> jax.Array:
    """0 above terminal_drag_alt_high (mid-disk, unity drag), 1 below
    terminal_drag_alt_low (terminal phase), linear ramp in between."""
    span = cfg.terminal_drag_alt_high - cfg.terminal_drag_alt_low
    u = jnp.clip((cfg.terminal_drag_alt_high - altitude) / (span + 1e-8), 0.0, 1.0)
    return u

def fin_terminal_drag_gain(lateral_r: jax.Array, cfg: EnvCfg) -> jax.Array:
    """Radial-gated terminal drag gain. Off => uniform terminal_drag_gain."""
    uniform = cfg.terminal_drag_gain
    if cfg.terminal_drag_radial_gate_on <= 0.5:
        return uniform
    x = (lateral_r - cfg.terminal_drag_far_radius) / jnp.maximum(cfg.terminal_drag_radial_width, 1e-6)
    u = 0.5 * (1.0 + jnp.tanh(x))  # 0 inside far_radius -> 1 outside
    return uniform + (cfg.terminal_drag_far_gain - uniform) * u


def fin_terminal_drag_gate(altitude: jax.Array, lateral_r: jax.Array, cfg: EnvCfg) -> jax.Array:
    frac = fin_terminal_drag_fraction(altitude, cfg)
    gain = fin_terminal_drag_gain(lateral_r, cfg)
    target = 1.0 + (gain - 1.0) * frac
    return cfg.terminal_drag_on * target + (1.0 - cfg.terminal_drag_on)


def fin_lift_effective(altitude: jax.Array, tilt: jax.Array, cfg: EnvCfg) -> jax.Array:
    """Regime-gated passive fin LIFT scale (C08 deployability filter).

    Gate ON: inside the altitude / absolute-tilt regime use gate_lift_scale,
             otherwise fall back to full lift (1.0).
    Gate OFF: use the global base lift scale (cfg.fin_lift_on) everywhere.
    """
    in_regime = (
        (altitude >= cfg.gate_h_min_m)
        & (altitude <= cfg.gate_h_max_m)
        & (jnp.abs(tilt) <= cfg.gate_tilt_max_rad)
    )
    gated = jnp.where(in_regime, cfg.gate_lift_scale, 1.0)
    base = jnp.where(jnp.isfinite(altitude), cfg.fin_lift_on, 1.0)
    return jnp.where(cfg.regime_gate_on > 0.5, gated, base)

def reset_one(key: jax.Array, cfg: EnvCfg) -> jax.Array:
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    pos0 = jnp.array([cfg.init_x, cfg.init_y, cfg.init_z])
    vel0 = jnp.array([cfg.init_vx, cfg.init_vy, cfg.init_vz])
    pos_noise = jax.random.uniform(k1, (3,), minval=-cfg.pos_noise, maxval=cfg.pos_noise)
    vel_noise = jax.random.uniform(k2, (3,), minval=-cfg.vel_noise, maxval=cfg.vel_noise)
    pos = pos0 + cfg.randomize * pos_noise
    vel = vel0 + cfg.randomize * vel_noise
    pos = pos.at[0].set(jnp.maximum(pos[0], 1500.0))

    base_q = euler_xyz_quat(
        math.radians(cfg.init_roll_deg),
        math.radians(cfg.init_pitch_deg),
        math.radians(cfg.init_yaw_deg),
    )
    noise_axis = jax.random.normal(k3, (3,))
    noise_ang = cfg.randomize * math.radians(cfg.attitude_noise_deg) * jax.random.uniform(k4, minval=-1.0, maxval=1.0)
    q_noise = axis_angle_quat(noise_axis, noise_ang)
    q = quat_mul(q_noise, base_q)
    q = q / jnp.maximum(jnp.linalg.norm(q), 1e-8)

    omega = cfg.randomize * jax.random.uniform(k5, (3,), minval=-cfg.omega_noise, maxval=cfg.omega_noise)
    initial_r = jnp.sqrt(pos[1] ** 2 + pos[2] ** 2 + 1e-8)
    return jnp.concatenate([pos, vel, q, omega, jnp.array([cfg.mass_empty + (cfg.mass_full - cfg.mass_empty) * cfg.fuel_scale, 0.0, initial_r])])


def reset_one_uniform_disk(key: jax.Array, cfg: EnvCfg) -> jax.Array:
    """Sample initial position uniformly by area inside a 1500m disk in y-z plane.

    The approach corridor (fixed single point at y=-250, z=80 with small noise)
    does not test the full capture-basin capability. T5 requires uniform-area
    sampling within the 1500 m disk centered on (0, 0) in the landing-pad y-z
    plane at x=2000 m. This function replaces the point IC with a true uniform
    disk sample. Altitude and velocity components follow the base defaults.
    """
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
    # Uniform disk sampling: sample radius² ~ Uniform(0, R²), angle ~ Uniform(0, 2π)
    disk_radius = DISK_RADIUS
    r2 = jax.random.uniform(k1) * disk_radius * disk_radius
    r_sample = jnp.where(
        cfg.stress_radial_offset_m > 0.0,
        jnp.array(cfg.stress_radial_offset_m, dtype=jnp.float32),
        jnp.sqrt(r2),
    )
    theta = jax.random.uniform(k2) * 2.0 * jnp.pi
    y0 = r_sample * jnp.cos(theta)
    z0 = r_sample * jnp.sin(theta)
    pos0 = jnp.array([cfg.init_x, y0, z0])
    vel0 = jnp.array([cfg.init_vx, cfg.init_vy, cfg.init_vz])
    pos_noise = jax.random.uniform(k3, (3,), minval=-cfg.pos_noise, maxval=cfg.pos_noise)
    vel_noise = jax.random.uniform(k4, (3,), minval=-cfg.vel_noise, maxval=cfg.vel_noise)
    pos = pos0 + cfg.randomize * pos_noise
    vel = vel0 + cfg.randomize * vel_noise
    pos = pos.at[0].set(jnp.maximum(pos[0], 1500.0))

    base_q = euler_xyz_quat(
        math.radians(cfg.init_roll_deg),
        math.radians(cfg.init_pitch_deg),
        math.radians(cfg.init_yaw_deg),
    )
    noise_axis = jax.random.normal(k5, (3,))
    noise_ang = cfg.randomize * math.radians(cfg.attitude_noise_deg) * jax.random.uniform(k6, minval=-1.0, maxval=1.0)
    q_noise = axis_angle_quat(noise_axis, noise_ang)
    q = quat_mul(q_noise, base_q)
    q = q / jnp.maximum(jnp.linalg.norm(q), 1e-8)

    omega = cfg.randomize * jax.random.uniform(k7, (3,), minval=-cfg.omega_noise, maxval=cfg.omega_noise)
    initial_r = jnp.sqrt(pos[1] ** 2 + pos[2] ** 2 + 1e-8)
    return jnp.concatenate([pos, vel, q, omega, jnp.array([cfg.mass_empty + (cfg.mass_full - cfg.mass_empty) * cfg.fuel_scale, 0.0, initial_r])])


def reset_fn_for_ic_mode(ic_mode: str):
    if ic_mode == "uniform_disk":
        return reset_one_uniform_disk
    return reset_one



def rcs_radial_gain(lateral_r, cfg: EnvCfg, far_outer_scale=None):
    """JAX-compatible radial schedule for rcs_torque_max.
    gain(r) = near_com + (far_outer - near_com) * sigmoid((r - boost_radius)/width),
    sigmoid(x) = 0.5*(1+tanh(x)). schedule_on<=0.5 -> flat unity gain.
    """
    far = cfg.rcs_far_outer_scale if far_outer_scale is None else far_outer_scale
    x = (lateral_r - cfg.rcs_boost_radius) / jnp.maximum(cfg.rcs_schedule_width, 1e-6)
    sched = 0.5 * (1.0 + jnp.tanh(x))
    gain = cfg.rcs_near_com_scale + (far - cfg.rcs_near_com_scale) * sched
    flat = jnp.ones_like(gain)
    return jnp.where(cfg.rcs_schedule_on > 0.5, gain, flat)


def compute_rcs_torque(a, lateral_r, cfg: EnvCfg, guard_params):
    """RCS torque with the C07 dose-margin guard on the lateral (pitch/yaw) axes.

    roll (rcs_x) uses the configured far-outer dose; lateral (rcs_y/rcs_z) uses
    the guard-saturated effective dose. Returns (torque, realized_lat_norm,
    nominal_lat_norm), where realized/nominal are the applied (post-guard) and
    unmodified lateral RCS torque magnitudes used for the saturation-vs-mask
    diagnostic.
    """
    dose = float(cfg.rcs_far_outer_scale)
    clamped = _dosemask.effective_lateral_dose_py(dose, guard_params)
    scope = str(guard_params.get("scope", "all")).strip().lower()
    clamp_lat = scope in ("all", "all_axes", "lateral")
    clamp_roll = scope in ("all", "all_axes", "roll")
    lat_dose = clamped if clamp_lat else dose
    roll_dose = clamped if clamp_roll else dose

    # When the guard is inactive (off, or dose within the safe band), keep the
    # exact parent expression so the off-state is bit-identical to the parent
    # plant. This is the fail-fast "off-state reproduces dose 3.0" property.
    inactive = (abs(lat_dose - dose) < 1e-9) and (abs(roll_dose - dose) < 1e-9)
    if inactive:
        rcs = a[3:6] * cfg.rcs_torque_max * (cfg.rcs_authority_scale * rcs_radial_gain(lateral_r, cfg))
        lat_norm = jnp.linalg.norm(a[4:6]) * cfg.rcs_torque_max * (cfg.rcs_authority_scale * rcs_radial_gain(lateral_r, cfg))
        return rcs, lat_norm, lat_norm

    gain_roll = rcs_radial_gain(lateral_r, cfg, roll_dose)
    gain_lat = rcs_radial_gain(lateral_r, cfg, lat_dose)
    authority = cfg.rcs_authority_scale * cfg.rcs_torque_max
    lat_cmd = a[4:6]
    realized_lat_norm = jnp.linalg.norm(lat_cmd) * authority * gain_lat
    nominal_lat_norm = jnp.linalg.norm(lat_cmd) * authority * rcs_radial_gain(lateral_r, cfg, dose)
    torque = jnp.stack([
        a[3] * authority * gain_roll,
        a[4] * authority * gain_lat,
        a[5] * authority * gain_lat,
    ])
    return torque, realized_lat_norm, nominal_lat_norm


def landing_gear_contact(pos, vel, q, omega, mass, cfg):
    """Passive landing-gear suspension contact model (gen42_pod15 C01, H_g42_02).

    Returns (F_contact_I, torque_contact_B). Exactly zero when gear_on == 0 so
    the gear-off path stays bitwise-identical to the parent uniform-g3@rk4 floor.
    """
    R_BI = quat_to_rot(q)
    omega_I = R_BI @ omega
    n = int(cfg.gear_n_legs)
    h_c = cfg.gear_contact_height_m
    r_f = cfg.gear_footprint_radius_m
    k = cfg.gear_spring_k_npm
    c = cfg.gear_damper_c_nspm
    mu = cfg.gear_friction_mu
    restore = cfg.gear_contact_restore_scale
    # Bottom-out guard: cap the static spring support to a fraction of the
    # per-leg weight so the suspension can decelerate the descent but can
    # never statically hover above the ground. This removes the contact-height
    # knife-edge (hover/touch=0 at hc >= ~2.8 m) while preserving soft descent.
    support_cap = cfg.gear_bottom_out_force_frac * mass * cfg.g / float(n)

    F_total = jnp.zeros(3)
    T_total = jnp.zeros(3)

    for i in range(n):
        phi = 2.0 * math.pi * i / n
        tip_B = jnp.array([-h_c, r_f * math.cos(phi), r_f * math.sin(phi)])
        r_I = R_BI @ tip_B
        tip_height = pos[0] + r_I[0]
        pen = jnp.maximum(-tip_height, 0.0)
        contact = pen > 0.0
        v_tip_I = vel + jnp.cross(omega_I, r_I)
        vn_down = -v_tip_I[0]
        spring = k * pen
        spring = jnp.where(cfg.gear_bottom_out_guard > 0.5, jnp.minimum(spring, support_cap), spring)
        fn = jnp.maximum(spring + contact * c * jnp.maximum(vn_down, 0.0), 0.0)
        F_n = fn * jnp.array([1.0, 0.0, 0.0])
        v_tan = jnp.array([0.0, v_tip_I[1], v_tip_I[2]])
        v_tan_norm = jnp.linalg.norm(v_tan) + 1e-8
        F_fric = -mu * fn * v_tan / (v_tan_norm + 0.5)
        F_i = F_n + F_fric
        F_total = F_total + F_i
        T_total = T_total + jnp.cross(r_I, F_i)

    F_contact_I = cfg.gear_on * F_total
    torque_contact_B = cfg.gear_on * restore * (R_BI.T @ T_total)
    return F_contact_I, torque_contact_B


def _forces_accel(state: jax.Array, action: jax.Array, cfg: EnvCfg):
    """Recompute (acc, domega) from a given state/action for the RK2 midpoint.

    Expression-identical copy of the force/torque block in step_one. Used ONLY
    by the rk2 integrator; the euler integrator keeps the inline computation in
    step_one untouched so the canonical (euler) path stays byte-identical to the
    parent gate-ablation harness.
    """
    pos = state[0:3]
    vel = state[3:6]
    q = state[6:10]
    omega = state[10:13]
    mass = state[13]
    initial_r = state[15]

    a = apply_guidance_residual(state, action, cfg)
    gy = a[0] * cfg.gimbal_max
    gz = a[1] * cfg.gimbal_max
    throttle = 0.5 * (a[2] + 1.0)
    lateral_r = jnp.sqrt(pos[1] ** 2 + pos[2] ** 2 + 1e-8)
    rcs, _rcs_lat_realized, _rcs_lat_nominal = compute_rcs_torque(a, lateral_r, cfg, _GUARD_CFG)
    grid_deflect = a[6:9] * _CL_MAX

    has_fuel = mass > cfg.mass_empty
    thrust = throttle * cfg.thrust_max * has_fuel

    R_BI = quat_to_rot(q)
    R_TB = gimbal_rot(gy, gz)
    F_thrust_B = R_TB @ jnp.array([thrust, 0.0, 0.0])
    F_thrust_I = R_BI @ F_thrust_B

    speed = jnp.linalg.norm(vel) + 1e-8
    coast_drag_gate = fin_terminal_drag_gate(pos[0], initial_r, cfg)
    area = math.pi * cfg.radius ** 2
    F_drag_I = -0.5 * cfg.rho * cfg.cd * area * speed * vel

    v_B = R_BI.T @ vel
    vBx = v_B[0]
    v_lat_B = jnp.array([0.0, v_B[1], v_B[2]])
    v_lat = jnp.linalg.norm(v_lat_B) + 1e-8
    alpha_eff = jnp.arctan2(v_lat, jnp.abs(vBx) + 1e-8)
    alpha_soft = cfg.fin_alpha_stall * jnp.tanh(alpha_eff / (cfg.fin_alpha_stall + 1e-8))
    q_dyn = 0.5 * cfg.rho * speed * speed
    fin_area_total = cfg.fin_area_each * cfg.fin_count
    lift_mag = cfg.fins_on * q_dyn * fin_area_total * cfg.fin_cl_alpha * alpha_soft
    tilt_cur = tilt_angle_from_q(q)
    lift_eff = fin_lift_effective(pos[0], tilt_cur, cfg)
    F_fin_lift_B = -lift_eff * lift_mag * (v_lat_B / v_lat)
    grid_y, grid_z, grid_roll = grid_deflect
    F_grid_ctrl_B = (
        cfg.fins_on
        * cfg.fin_active_on
        * q_dyn
        * fin_area_total
        * jnp.array([0.0, _CL_Y * grid_y, _CL_Z * grid_z])
    )
    F_inc_B, cd_inc = _plant_aero.incidence_terms(q_dyn, fin_area_total, initial_r, pos[0], cfg)
    cd_fin = (
        cfg.fin_cd0
        + cfg.fin_cd_alpha * alpha_soft * alpha_soft
        + 0.25 * (grid_y * grid_y + grid_z * grid_z + grid_roll * grid_roll)
        + cd_inc
    )
    F_fin_drag_B = -cfg.fins_on * cfg.fin_drag_on * q_dyn * fin_area_total * cd_fin * (v_B / (speed + 1e-8)) * coast_drag_gate
    F_fin_B = F_fin_lift_B + F_inc_B + F_grid_ctrl_B + F_fin_drag_B
    F_fin_I = R_BI @ F_fin_B

    F_grav_I = jnp.array([-mass * cfg.g, 0.0, 0.0])
    F_gear_I, torque_gear_B = landing_gear_contact(pos, vel, q, omega, mass, cfg)
    F_I = F_thrust_I + F_drag_I + F_fin_I + F_grav_I + F_gear_I
    acc = F_I / mass

    r_engine_B = jnp.array([cfg.lever_x, 0.0, 0.0])
    torque_engine_B = jnp.cross(r_engine_B, F_thrust_B)
    r_fin_B = jnp.array([_FIN_STATION_X, 0.0, 0.0])
    torque_fin_B = jnp.cross(r_fin_B, F_fin_B)
    torque_grid_roll_B = jnp.array([
        cfg.fins_on * cfg.fin_active_on * q_dyn * fin_area_total * cfg.radius * _CL_ROLL * grid_roll,
        0.0,
        0.0,
    ])
    torque_fin_damp_B = -cfg.fins_on * cfg.fin_damping_on * _FIN_DAMPING * q_dyn * fin_area_total * cfg.radius * omega
    torque_B = torque_engine_B + rcs + torque_fin_B + torque_grid_roll_B + torque_fin_damp_B + torque_gear_B
    J = inertia_diag(mass, cfg)
    domega = (torque_B - jnp.cross(omega, J * omega)) / J

    return acc, domega, thrust


def step_one(key: jax.Array, state: jax.Array, action: jax.Array, cfg: EnvCfg):
    del key
    pos = state[0:3]
    vel = state[3:6]
    q = state[6:10]
    omega = state[10:13]
    mass = state[13]
    time = state[14]
    initial_r = state[15]

    # The policy action is a residual. The physical action applied to the
    # environment is guidance_prior + residual_scale * policy_action.
    a = apply_guidance_residual(state, action, cfg)
    gy = a[0] * cfg.gimbal_max
    gz = a[1] * cfg.gimbal_max
    throttle = 0.5 * (a[2] + 1.0)
    lateral_r = jnp.sqrt(pos[1] ** 2 + pos[2] ** 2 + 1e-8)
    rcs, _rcs_lat_realized, _rcs_lat_nominal = compute_rcs_torque(a, lateral_r, cfg, _GUARD_CFG)
    grid_deflect = a[6:9] * _CL_MAX

    has_fuel = mass > cfg.mass_empty
    thrust = throttle * cfg.thrust_max * has_fuel

    R_BI = quat_to_rot(q)  # body -> inertial
    R_TB = gimbal_rot(gy, gz)
    F_thrust_B = R_TB @ jnp.array([thrust, 0.0, 0.0])
    F_thrust_I = R_BI @ F_thrust_B

    # Simple inertial drag from relative airspeed. No wind in the clean baseline.
    speed = jnp.linalg.norm(vel) + 1e-8
    coast_drag_gate = fin_terminal_drag_gate(pos[0], initial_r, cfg)
    area = math.pi * cfg.radius ** 2
    F_drag_I = -0.5 * cfg.rho * cfg.cd * area * speed * vel

    # Active forward grid-fin aerodynamics in body coordinates. Body +x is the
    # rocket longitudinal/thrust axis. In the landing task the rocket descends
    # tail-first, so v_B[0] is typically negative and the relative wind travels
    # aft -> nose. Lateral body velocity components encode angle of attack /
    # sideslip. Grid fins produce passive side force opposing lateral crossflow, active
    # commanded side force/roll moment, plus deliberately large profile/induced drag. With fin_station_x > 0, this
    # yields a restoring moment for tail-first descent; moving the station aft
    # would be a separate destabilizing tail-fin ablation.
    v_B = R_BI.T @ vel
    vBx = v_B[0]
    v_lat_B = jnp.array([0.0, v_B[1], v_B[2]])
    v_lat = jnp.linalg.norm(v_lat_B) + 1e-8
    alpha_eff = jnp.arctan2(v_lat, jnp.abs(vBx) + 1e-8)
    alpha_soft = cfg.fin_alpha_stall * jnp.tanh(alpha_eff / (cfg.fin_alpha_stall + 1e-8))
    q_dyn = 0.5 * cfg.rho * speed * speed
    fin_area_total = cfg.fin_area_each * cfg.fin_count
    lift_mag = cfg.fins_on * q_dyn * fin_area_total * cfg.fin_cl_alpha * alpha_soft
    tilt_cur = tilt_angle_from_q(q)
    lift_eff = fin_lift_effective(pos[0], tilt_cur, cfg)
    F_fin_lift_B = -lift_eff * lift_mag * (v_lat_B / v_lat)
    grid_y, grid_z, grid_roll = grid_deflect
    F_grid_ctrl_B = (
        cfg.fins_on
        * cfg.fin_active_on
        * q_dyn
        * fin_area_total
        * jnp.array([0.0, _CL_Y * grid_y, _CL_Z * grid_z])
    )
    F_inc_B, cd_inc = _plant_aero.incidence_terms(q_dyn, fin_area_total, initial_r, pos[0], cfg)
    cd_fin = (
        cfg.fin_cd0
        + cfg.fin_cd_alpha * alpha_soft * alpha_soft
        + 0.25 * (grid_y * grid_y + grid_z * grid_z + grid_roll * grid_roll)
        + cd_inc
    )
    F_fin_drag_B = -cfg.fins_on * cfg.fin_drag_on * q_dyn * fin_area_total * cd_fin * (v_B / (speed + 1e-8)) * coast_drag_gate
    F_fin_B = F_fin_lift_B + F_inc_B + F_grid_ctrl_B + F_fin_drag_B
    F_fin_I = R_BI @ F_fin_B

    F_grav_I = jnp.array([-mass * cfg.g, 0.0, 0.0])
    F_gear_I, torque_gear_B = landing_gear_contact(pos, vel, q, omega, mass, cfg)
    F_I = F_thrust_I + F_drag_I + F_fin_I + F_grav_I + F_gear_I
    acc = F_I / mass

    r_engine_B = jnp.array([cfg.lever_x, 0.0, 0.0])
    torque_engine_B = jnp.cross(r_engine_B, F_thrust_B)
    r_fin_B = jnp.array([_FIN_STATION_X, 0.0, 0.0])
    torque_fin_B = jnp.cross(r_fin_B, F_fin_B)
    torque_grid_roll_B = jnp.array([
        cfg.fins_on * cfg.fin_active_on * q_dyn * fin_area_total * cfg.radius * _CL_ROLL * grid_roll,
        0.0,
        0.0,
    ])
    J_for_damp = inertia_diag(mass, cfg)
    torque_fin_damp_B = -cfg.fins_on * cfg.fin_damping_on * _FIN_DAMPING * q_dyn * fin_area_total * cfg.radius * omega
    torque_B = torque_engine_B + rcs + torque_fin_B + torque_grid_roll_B + torque_fin_damp_B + torque_gear_B
    J = inertia_diag(mass, cfg)
    domega = (torque_B - jnp.cross(omega, J * omega)) / J

    if cfg.integrator == "rk2":
        # Heun (predictor-corrector) position/velocity/omega, then a single
        # exponential-map attitude update with the corrected angular velocity.
        # This is the second-harness-family integration path (H_g28_02 C01).
        if cfg.integrator_substeps == 1:
            vel_pred = vel + acc * cfg.dt
            pos_pred = pos + vel * cfg.dt
            omega_pred = omega + domega * cfg.dt
            pred_state = state.at[0:3].set(pos_pred).at[3:6].set(vel_pred).at[10:13].set(omega_pred)
            acc2, domega2, _ = _forces_accel(pred_state, action, cfg)
            vel_new = vel + 0.5 * (acc + acc2) * cfg.dt
            pos_new = pos + 0.5 * (vel + vel_new) * cfg.dt
            omega_new = omega + 0.5 * (domega + domega2) * cfg.dt
            q_new = integrate_quat_exact(q, omega_new, cfg.dt)
        else:
            n_sub = cfg.integrator_substeps
            dt_sub = cfg.dt / n_sub

            def _rk2_substep(i, carry):
                pos_i, vel_i, q_i, omega_i, thrust_int = carry
                s_i = (state.at[0:3].set(pos_i).at[3:6].set(vel_i).at[6:10].set(q_i)
                       .at[10:13].set(omega_i).at[14].set(time + i / n_sub))
                a1, w1, t1 = _forces_accel(s_i, action, cfg)
                vel_pred = vel_i + a1 * dt_sub
                pos_pred = pos_i + vel_i * dt_sub
                omega_pred = omega_i + w1 * dt_sub
                s_pred = (s_i.at[0:3].set(pos_pred).at[3:6].set(vel_pred)
                          .at[10:13].set(omega_pred).at[14].set(time + (i + 1.0) / n_sub))
                a2, w2, t2 = _forces_accel(s_pred, action, cfg)
                vel_new = vel_i + 0.5 * (a1 + a2) * dt_sub
                pos_new = pos_i + 0.5 * (vel_i + vel_new) * dt_sub
                omega_new = omega_i + 0.5 * (w1 + w2) * dt_sub
                q_new = integrate_quat_exact(q_i, omega_new, dt_sub)
                thrust_int = thrust_int + 0.5 * (t1 + t2) * dt_sub
                return pos_new, vel_new, q_new, omega_new, thrust_int

            pos_new, vel_new, q_new, omega_new, thrust_int = jax.lax.fori_loop(
                0, n_sub, _rk2_substep, (pos, vel, q, omega, jnp.array(0.0, dtype=jnp.float32)))
            thrust = thrust_int / cfg.dt
    elif cfg.integrator == "rk2_full":
        # Full midpoint RK2 (Lie-group midpoint). Half-step predictor ->
        # recompute forces/torques at the midpoint state -> midpoint rule for
        # position/velocity/omega and a midpoint exponential-map quaternion
        # update q_{n+1} = exp(dt * omega_mid) * q_n. This is the "full RK2
        # quaternion integration" fix recommended in prior_run_context.md.
        if cfg.integrator_substeps == 1:
            vel_h = vel + 0.5 * acc * cfg.dt
            pos_h = pos + 0.5 * vel * cfg.dt
            omega_h = omega + 0.5 * domega * cfg.dt
            q_h = integrate_quat_exact(q, omega, 0.5 * cfg.dt)
            mid_state = state.at[0:3].set(pos_h).at[3:6].set(vel_h).at[6:10].set(q_h).at[10:13].set(omega_h).at[14].set(time + 0.5)
            acc_m, domega_m, thrust_m = _forces_accel(mid_state, action, cfg)
            vel_new = vel + acc_m * cfg.dt
            pos_new = pos + vel_h * cfg.dt
            omega_new = omega + domega_m * cfg.dt
            q_new = integrate_quat_exact(q, omega_h, cfg.dt)
            # mass uses the midpoint throttle/action
            thrust = thrust_m
        else:
            n_sub = cfg.integrator_substeps
            dt_sub = cfg.dt / n_sub

            def _rk2f_substep(i, carry):
                pos_i, vel_i, q_i, omega_i, thrust_int = carry
                s_i = (state.at[0:3].set(pos_i).at[3:6].set(vel_i).at[6:10].set(q_i)
                       .at[10:13].set(omega_i).at[14].set(time + i / n_sub))
                a1, w1, t1 = _forces_accel(s_i, action, cfg)
                vel_h = vel_i + 0.5 * a1 * dt_sub
                pos_h = pos_i + 0.5 * vel_i * dt_sub
                omega_h = omega_i + 0.5 * w1 * dt_sub
                q_h = integrate_quat_exact(q_i, omega_i, 0.5 * dt_sub)
                s_mid = (s_i.at[0:3].set(pos_h).at[3:6].set(vel_h).at[6:10].set(q_h)
                         .at[10:13].set(omega_h).at[14].set(time + (i + 0.5) / n_sub))
                a_m, w_m, t_m = _forces_accel(s_mid, action, cfg)
                vel_new = vel_i + a_m * dt_sub
                pos_new = pos_i + vel_h * dt_sub
                omega_new = omega_i + w_m * dt_sub
                q_new = integrate_quat_exact(q_i, omega_h, dt_sub)
                thrust_int = thrust_int + t_m * dt_sub
                return pos_new, vel_new, q_new, omega_new, thrust_int

            pos_new, vel_new, q_new, omega_new, thrust_int = jax.lax.fori_loop(
                0, n_sub, _rk2f_substep, (pos, vel, q, omega, jnp.array(0.0, dtype=jnp.float32)))
            thrust = thrust_int / cfg.dt
    elif cfg.integrator == "euler_half":
        # Two explicit-Euler half-steps (dt/2) using the SAME canonical update
        # formula, to build a dt-convergence ladder of the canonical integrator
        # family. Mass update stays identical to the single-step euler path so
        # the only difference is position/velocity/attitude substepping.
        dt_h = 0.5 * cfg.dt
        acc1, domega1, _ = _forces_accel(state, action, cfg)
        vel_h = vel + acc1 * dt_h
        pos_h = pos + vel * dt_h + 0.5 * acc1 * dt_h * dt_h
        omega_h = omega + domega1 * dt_h
        q_h = integrate_quat_exact(q, omega_h, dt_h)
        half_state = state.at[0:3].set(pos_h).at[3:6].set(vel_h).at[6:10].set(q_h).at[10:13].set(omega_h).at[14].set(time + 0.5)
        acc2, domega2, _ = _forces_accel(half_state, action, cfg)
        vel_new = vel_h + acc2 * dt_h
        pos_new = pos_h + vel_h * dt_h + 0.5 * acc2 * dt_h * dt_h
        omega_new = omega_h + domega2 * dt_h
        q_new = integrate_quat_exact(q_h, omega_new, dt_h)
    elif cfg.integrator == "euler_quarter":
        # Four canonical explicit-Euler quarter-steps (dt/4) to complete the
        # dt-convergence ladder of the canonical integrator family. Mass update
        # stays identical to the single-step euler path (thrust from stage 1).
        dt_q = 0.25 * cfg.dt
        s_cur = state
        for _ in range(4):
            acc_s, domega_s, _ = _forces_accel(s_cur, action, cfg)
            p = s_cur[0:3]
            v = s_cur[3:6]
            qq = s_cur[6:10]
            w = s_cur[10:13]
            tt = s_cur[14]
            v2 = v + acc_s * dt_q
            p2 = p + v * dt_q + 0.5 * acc_s * dt_q * dt_q
            w2 = w + domega_s * dt_q
            q2 = integrate_quat_exact(qq, w2, dt_q)
            s_cur = s_cur.at[0:3].set(p2).at[3:6].set(v2).at[6:10].set(q2).at[10:13].set(w2).at[14].set(tt + 0.25)
        pos_new = s_cur[0:3]
        vel_new = s_cur[3:6]
        q_new = s_cur[6:10]
        omega_new = s_cur[10:13]
    elif cfg.integrator == "rk4":
        # Classical RK4 for position/velocity/omega with an exponential-map
        # attitude update using the RK4 stage-averaged angular velocity (a
        # consistent high-order reference for the integrator-spread bracket).
        if cfg.integrator_substeps == 1:
            dt_h = 0.5 * cfg.dt
            k1_acc, k1_domega = acc, domega
            vel2 = vel + dt_h * k1_acc
            pos2 = pos + dt_h * vel
            omega2 = omega + dt_h * k1_domega
            q2 = integrate_quat_exact(q, omega, dt_h)
            s2 = state.at[0:3].set(pos2).at[3:6].set(vel2).at[6:10].set(q2).at[10:13].set(omega2).at[14].set(time + 0.5)
            k2_acc, k2_domega, t2 = _forces_accel(s2, action, cfg)
            vel3 = vel + dt_h * k2_acc
            pos3 = pos + dt_h * vel2
            omega3 = omega + dt_h * k2_domega
            q3 = integrate_quat_exact(q, omega2, dt_h)
            s3 = state.at[0:3].set(pos3).at[3:6].set(vel3).at[6:10].set(q3).at[10:13].set(omega3).at[14].set(time + 0.5)
            k3_acc, k3_domega, t3 = _forces_accel(s3, action, cfg)
            vel4 = vel + cfg.dt * k3_acc
            pos4 = pos + cfg.dt * vel3
            omega4 = omega + cfg.dt * k3_domega
            q4 = integrate_quat_exact(q, omega3, cfg.dt)
            s4 = state.at[0:3].set(pos4).at[3:6].set(vel4).at[6:10].set(q4).at[10:13].set(omega4).at[14].set(time + 1.0)
            k4_acc, k4_domega, t4 = _forces_accel(s4, action, cfg)
            vel_new = vel + cfg.dt / 6.0 * (k1_acc + 2.0 * k2_acc + 2.0 * k3_acc + k4_acc)
            pos_new = pos + cfg.dt / 6.0 * (vel + 2.0 * vel2 + 2.0 * vel3 + vel4)
            omega_new = omega + cfg.dt / 6.0 * (k1_domega + 2.0 * k2_domega + 2.0 * k3_domega + k4_domega)
            omega_avg = (omega + 2.0 * omega2 + 2.0 * omega3 + omega_new) / 6.0
            q_new = integrate_quat_exact(q, omega_avg, cfg.dt)
            thrust = (thrust + 2.0 * t2 + 2.0 * t3 + t4) / 6.0
        else:
            n_sub = cfg.integrator_substeps
            dt_sub = cfg.dt / n_sub

            def _rk4_substep(i, carry):
                pos_i, vel_i, q_i, omega_i, thrust_int = carry
                s_i = (state.at[0:3].set(pos_i).at[3:6].set(vel_i).at[6:10].set(q_i)
                       .at[10:13].set(omega_i).at[14].set(time + i / n_sub))
                a1, w1, t1 = _forces_accel(s_i, action, cfg)
                dt_h = 0.5 * dt_sub
                vel2 = vel_i + dt_h * a1
                pos2 = pos_i + dt_h * vel_i
                omega2 = omega_i + dt_h * w1
                q2 = integrate_quat_exact(q_i, omega_i, dt_h)
                s2 = (s_i.at[0:3].set(pos2).at[3:6].set(vel2).at[6:10].set(q2)
                      .at[10:13].set(omega2).at[14].set(time + (i + 0.5) / n_sub))
                a2, w2, t2 = _forces_accel(s2, action, cfg)
                vel3 = vel_i + dt_h * a2
                pos3 = pos_i + dt_h * vel2
                omega3 = omega_i + dt_h * w2
                q3 = integrate_quat_exact(q_i, omega2, dt_h)
                s3 = (s_i.at[0:3].set(pos3).at[3:6].set(vel3).at[6:10].set(q3)
                      .at[10:13].set(omega3).at[14].set(time + (i + 0.5) / n_sub))
                a3, w3, t3 = _forces_accel(s3, action, cfg)
                vel4 = vel_i + dt_sub * a3
                pos4 = pos_i + dt_sub * vel3
                omega4 = omega_i + dt_sub * w3
                q4 = integrate_quat_exact(q_i, omega3, dt_sub)
                s4 = (s_i.at[0:3].set(pos4).at[3:6].set(vel4).at[6:10].set(q4)
                      .at[10:13].set(omega4).at[14].set(time + (i + 1.0) / n_sub))
                a4, w4, t4 = _forces_accel(s4, action, cfg)
                vel_new = vel_i + dt_sub / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
                pos_new = pos_i + dt_sub / 6.0 * (vel_i + 2.0 * vel2 + 2.0 * vel3 + vel4)
                omega_new = omega_i + dt_sub / 6.0 * (w1 + 2.0 * w2 + 2.0 * w3 + w4)
                omega_avg = (omega_i + 2.0 * omega2 + 2.0 * omega3 + omega_new) / 6.0
                q_new = integrate_quat_exact(q_i, omega_avg, dt_sub)
                thrust_int = thrust_int + (t1 + 2.0 * t2 + 2.0 * t3 + t4) / 6.0 * dt_sub
                return pos_new, vel_new, q_new, omega_new, thrust_int

            pos_new, vel_new, q_new, omega_new, thrust_int = jax.lax.fori_loop(
                0, n_sub, _rk4_substep, (pos, vel, q, omega, jnp.array(0.0, dtype=jnp.float32)))
            thrust = thrust_int / cfg.dt
    else:
        # canonical explicit-Euler velocity/omega + exponential-map quat
        vel_new = vel + acc * cfg.dt
        pos_new = pos + vel * cfg.dt + 0.5 * acc * cfg.dt * cfg.dt
        omega_new = omega + domega * cfg.dt
        q_new = integrate_quat_exact(q, omega_new, cfg.dt)

    dm = thrust / (cfg.g * cfg.isp + 1e-8) * cfg.dt
    mass_new = jnp.maximum(mass - dm, cfg.mass_empty)
    time_new = time + 1.0

    # If below ground, project to ground for visualization and terminal metrics.
    pos_new = pos_new.at[0].set(jnp.maximum(pos_new[0], 0.0))

    new_state = jnp.concatenate([pos_new, vel_new, q_new, omega_new, jnp.array([mass_new, time_new, initial_r])])

    lateral = jnp.sqrt(pos_new[1] ** 2 + pos_new[2] ** 2 + 1e-8)
    speed_new = jnp.linalg.norm(vel_new)
    omega_mag = jnp.linalg.norm(omega_new)
    tilt = tilt_angle_from_q(q_new)

    touched = pos_new[0] <= 1e-4
    safe = touched & (lateral < cfg.landing_radius) & (speed_new < cfg.landing_speed) & (tilt < cfg.landing_tilt) & (omega_mag < cfg.landing_omega)
    # ── C06 Jiang speed-knee terminal reward (training signal only) ──
    # `safe` above keeps the standard 0.10 rad tilt knee and is still reported
    # in info[0] as the legacy raw-safe audit signal. The PPO training reward
    # uses reward_safe with separate tilt and speed knees at the strict-Jiang
    # gates (tilt 0.05, speed 2.0). With knee=0.10/speed=4.0 and bonuses off,
    # reward_safe == safe and the original reward is reproduced exactly.
    reward_tilt_knee = cfg.reward_tilt_knee_rad
    reward_speed_knee = cfg.reward_speed_knee_mps
    reward_safe = (
        touched
        & (lateral < cfg.landing_radius)
        & (speed_new < reward_speed_knee)
        & (tilt < reward_tilt_knee)
        & (omega_mag < cfg.landing_omega)
    )
    tilt_pass = touched & (tilt < reward_tilt_knee)
    tilt_margin = jnp.clip(1.0 - tilt / jnp.maximum(reward_tilt_knee, 1e-6), 0.0, 1.0)
    tilt_bonus = (
        (cfg.reward_tilt_bonus_on > 0.5)
        * cfg.reward_tilt_bonus_scale
        * tilt_pass
        * tilt_margin
    )
    speed_pass = touched & (speed_new < reward_speed_knee)
    speed_margin = jnp.clip(1.0 - speed_new / jnp.maximum(reward_speed_knee, 1e-6), 0.0, 1.0)
    speed_bonus = (
        (cfg.reward_speed_bonus_on > 0.5)
        * cfg.reward_speed_bonus_scale
        * speed_pass
        * speed_margin
    )
    lateral_gate = touched & (lateral < cfg.landing_radius)
    lateral_bonus = (
        (cfg.reward_lateral_bonus_on > 0.5)
        * cfg.reward_lateral_bonus_scale
        * lateral_gate
        * (cfg.reward_lateral_floor_m / (lateral + cfg.reward_lateral_floor_m))
    )
    flipped = tilt > cfg.max_tilt
    oob = (lateral > cfg.max_lateral) | (pos_new[0] > cfg.init_x + 500.0)
    timeout = time_new >= cfg.max_steps
    done = touched | flipped | oob | timeout

    # Dense reward. It is deliberately simple and sign-transparent.
    dist_old = jnp.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2 + 1e-8)
    dist_new = jnp.sqrt(pos_new[0] ** 2 + pos_new[1] ** 2 + pos_new[2] ** 2 + 1e-8)
    progress = (dist_old - dist_new) / 20.0

    # Altitude-dependent speed target: fast high up, slow near the pad.
    v_ref = jnp.where(pos_new[0] > 600.0, 85.0, jnp.where(pos_new[0] > 120.0, 35.0, 5.0))
    descent_speed = jnp.maximum(-vel_new[0], 0.0)
    speed_track = -0.015 * jnp.abs(descent_speed - v_ref)

    reward = (
        progress
        + speed_track
        - 0.004 * lateral
        - 1.50 * tilt
        - 0.15 * omega_mag
        - 0.01 * jnp.sum(a[0:2] ** 2)
        - 0.002 * throttle
        - 0.0005 * jnp.sum(a[3:6] ** 2)
        - 0.002 * jnp.sum(a[6:9] ** 2)
    )
    crash_penalty = -80.0 - 0.4 * speed_new - 0.1 * lateral - 20.0 * tilt
    reward = (
        reward
        + reward_safe * 250.0
        + (touched & (~reward_safe)) * crash_penalty
        + tilt_bonus
        + speed_bonus
        + lateral_bonus
        + flipped * (-120.0)
        + oob * (-120.0)
    )

    info = jnp.array([
        safe.astype(jnp.float32),
        touched.astype(jnp.float32),
        lateral,
        speed_new,
        tilt,
        omega_mag,
        throttle,
        alpha_eff,
        lift_mag,
        jnp.linalg.norm(torque_fin_B + torque_grid_roll_B + torque_fin_damp_B),
        jnp.linalg.norm(grid_deflect),
        jnp.linalg.norm(F_grid_ctrl_B),
        jnp.linalg.norm(F_fin_drag_B),  # applied (post-gate) fin drag magnitude
        coast_drag_gate,                        # terminal ramp drag gate value
        fin_terminal_drag_fraction(pos[0], cfg),  # terminal phase fraction (1 below 300 m / 0 above 350 m)
        _rcs_lat_realized,   # C07: realized (post-guard) lateral RCS torque norm (N*m)
        _rcs_lat_nominal,    # C07: nominal (pre-guard) lateral RCS torque norm (N*m)
    ])
    return observe(new_state, cfg), new_state, reward, done.astype(jnp.float32), info


# -----------------------------------------------------------------------------
# Pure JAX actor-critic and Adam
# -----------------------------------------------------------------------------


def init_layer(key, fan_in: int, fan_out: int, scale: float = math.sqrt(2.0)):
    w = jax.random.normal(key, (fan_in, fan_out)) * (scale / math.sqrt(fan_in))
    b = jnp.zeros((fan_out,))
    return {"w": w, "b": b}


def init_params(key: jax.Array, hidden: int):
    keys = jax.random.split(key, 7)
    return {
        "pi": [
            init_layer(keys[0], OBS_DIM, hidden),
            init_layer(keys[1], hidden, hidden),
            init_layer(keys[2], hidden, ACT_DIM, scale=0.01),
        ],
        "vf": [
            init_layer(keys[3], OBS_DIM, hidden),
            init_layer(keys[4], hidden, hidden),
            init_layer(keys[5], hidden, 1, scale=1.0),
        ],
        "log_std": jnp.ones((ACT_DIM,)) * -2.0,
    }


def mlp(layers, x: jax.Array) -> jax.Array:
    h = x
    for layer in layers[:-1]:
        h = jnp.tanh(h @ layer["w"] + layer["b"])
    return h @ layers[-1]["w"] + layers[-1]["b"]


def policy_value(params, obs: jax.Array):
    mean = mlp(params["pi"], obs)
    value = jnp.squeeze(mlp(params["vf"], obs), axis=-1)
    log_std = jnp.clip(params["log_std"], -4.0, 0.5)
    log_std = jnp.broadcast_to(log_std, mean.shape)
    return mean, log_std, value


def atanh_clip(a: jax.Array) -> jax.Array:
    a = jnp.clip(a, -0.999999, 0.999999)
    return 0.5 * jnp.log((1.0 + a) / (1.0 - a))


def gaussian_logp(z: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    var_term = ((z - mean) / (jnp.exp(log_std) + 1e-8)) ** 2
    return -0.5 * jnp.sum(var_term + 2.0 * log_std + math.log(2.0 * math.pi), axis=-1)


def tanh_logp(action: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    z = atanh_clip(action)
    log_det = jnp.sum(jnp.log(1.0 - action ** 2 + 1e-6), axis=-1)
    return gaussian_logp(z, mean, log_std) - log_det


def sample_action(key: jax.Array, params, obs: jax.Array):
    mean, log_std, value = policy_value(params, obs)
    z = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
    action = jnp.tanh(z)
    logp = gaussian_logp(z, mean, log_std) - jnp.sum(jnp.log(1.0 - action ** 2 + 1e-6), axis=-1)
    return action, logp, value


def deterministic_action(params, obs: jax.Array) -> jax.Array:
    mean, _, _ = policy_value(params, obs)
    return jnp.tanh(mean)


def tree_zeros_like(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def adam_init(params):
    return {"m": tree_zeros_like(params), "v": tree_zeros_like(params), "t": jnp.array(0, dtype=jnp.int32)}


def adam_update(params, grads, opt_state, lr: float, beta1=0.9, beta2=0.999, eps=1e-8):
    t = opt_state["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: beta1 * m + (1.0 - beta1) * g, opt_state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: beta2 * v + (1.0 - beta2) * (g * g), opt_state["v"], grads)
    b1 = 1.0 - beta1 ** t
    b2 = 1.0 - beta2 ** t
    params = jax.tree_util.tree_map(lambda p, m, v: p - lr * (m / b1) / (jnp.sqrt(v / b2) + eps), params, m, v)
    return params, {"m": m, "v": v, "t": t}


# -----------------------------------------------------------------------------
# PPO
# -----------------------------------------------------------------------------


class PPOCfg(NamedTuple):
    num_envs: int = 128
    num_steps: int = 128
    updates: int = 800
    epochs: int = 4
    minibatches: int = 8
    gamma: float = 0.995
    lam: float = 0.95
    lr: float = 3e-4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.7
    hidden: int = 128


class Rollout(NamedTuple):
    obs: jax.Array       # [T,N,obs]
    actions: jax.Array   # [T,N,act]
    logp: jax.Array      # [T,N]
    rewards: jax.Array   # [T,N]
    dones: jax.Array     # [T,N]
    values: jax.Array    # [T,N]
    infos: jax.Array     # [T,N,info]


def make_collect_rollout(env_cfg: EnvCfg, ppo_cfg: PPOCfg, ic_mode: str = "approach_corridor"):
    reset_fn = reset_fn_for_ic_mode(ic_mode)
    reset_v = jax.vmap(lambda k: reset_fn(k, env_cfg))
    step_v = jax.vmap(lambda k, s, a: step_one(k, s, a, env_cfg))

    def collect(params, states, key):
        def body(carry, _):
            states, key = carry
            key, k_act, k_step, k_reset = jax.random.split(key, 4)
            obs = jax.vmap(lambda s: observe(s, env_cfg))(states)
            action, logp, value = sample_action(k_act, params, obs)
            step_keys = jax.random.split(k_step, ppo_cfg.num_envs)
            next_obs, next_states, reward, done, info = step_v(step_keys, states, action)
            reset_keys = jax.random.split(k_reset, ppo_cfg.num_envs)
            reset_states = reset_v(reset_keys)
            next_states = jnp.where(done[:, None] > 0.5, reset_states, next_states)
            transition = (obs, action, logp, reward, done, value, info)
            return (next_states, key), transition

        (states, key), trans = jax.lax.scan(body, (states, key), None, length=ppo_cfg.num_steps)
        rollout = Rollout(*trans)
        last_obs = jax.vmap(lambda s: observe(s, env_cfg))(states)
        _, _, last_value = policy_value(params, last_obs)
        return states, key, rollout, last_value

    return jax.jit(collect)


def compute_gae(rollout: Rollout, last_value: jax.Array, cfg: PPOCfg):
    def scan_fn(carry, x):
        next_gae, next_value = carry
        reward, done, value = x
        nonterminal = 1.0 - done
        delta = reward + cfg.gamma * next_value * nonterminal - value
        gae = delta + cfg.gamma * cfg.lam * nonterminal * next_gae
        return (gae, value), gae

    (_, _), adv = jax.lax.scan(
        scan_fn,
        (jnp.zeros_like(last_value), last_value),
        (rollout.rewards, rollout.dones, rollout.values),
        reverse=True,
    )
    returns = adv + rollout.values
    return adv, returns


def global_norm(tree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum([jnp.sum(x * x) for x in leaves]) + 1e-8)


def clip_grads(grads, max_norm: float):
    norm = global_norm(grads)
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-8))
    return jax.tree_util.tree_map(lambda g: g * scale, grads)


def make_ppo_update(cfg: PPOCfg):
    batch_size = cfg.num_envs * cfg.num_steps
    mb_size = batch_size // cfg.minibatches

    def loss_fn(params, batch):
        obs, actions, old_logp, adv, returns, old_values = batch
        mean, log_std, values = policy_value(params, obs)
        new_logp = tanh_logp(actions, mean, log_std)
        ratio = jnp.exp(new_logp - old_logp)
        adv = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)
        pg1 = ratio * adv
        pg2 = jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
        policy_loss = -jnp.mean(jnp.minimum(pg1, pg2))

        v_clipped = old_values + jnp.clip(values - old_values, -cfg.clip_eps, cfg.clip_eps)
        vf1 = (values - returns) ** 2
        vf2 = (v_clipped - returns) ** 2
        value_loss = 0.5 * jnp.mean(jnp.maximum(vf1, vf2))

        entropy = jnp.mean(jnp.sum(log_std + 0.5 * math.log(2.0 * math.pi * math.e), axis=-1))
        loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy
        metrics = jnp.array([loss, policy_loss, value_loss, entropy])
        return loss, metrics

    def update(params, opt_state, rollout: Rollout, last_value: jax.Array, key: jax.Array, lr: float):
        adv, returns = compute_gae(rollout, last_value, cfg)
        flat = lambda x: x.reshape((batch_size,) + x.shape[2:])
        obs = flat(rollout.obs)
        actions = flat(rollout.actions)
        logp = rollout.logp.reshape((batch_size,))
        values = rollout.values.reshape((batch_size,))
        adv_f = adv.reshape((batch_size,))
        returns_f = returns.reshape((batch_size,))

        def epoch_body(carry, _):
            params, opt_state, key = carry
            key, k_perm = jax.random.split(key)
            perm = jax.random.permutation(k_perm, batch_size)

            def mb_body(carry2, mb_idx):
                params, opt_state = carry2
                idx = jax.lax.dynamic_slice(perm, (mb_idx * mb_size,), (mb_size,))
                batch = (
                    obs[idx], actions[idx], logp[idx], adv_f[idx], returns_f[idx], values[idx]
                )
                (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
                grads = clip_grads(grads, cfg.max_grad_norm)
                params, opt_state = adam_update(params, grads, opt_state, lr)
                return (params, opt_state), metrics

            (params, opt_state), mb_metrics = jax.lax.scan(mb_body, (params, opt_state), jnp.arange(cfg.minibatches))
            return (params, opt_state, key), jnp.mean(mb_metrics, axis=0)

        (params, opt_state, key), metrics = jax.lax.scan(epoch_body, (params, opt_state, key), jnp.arange(cfg.epochs))
        return params, opt_state, key, jnp.mean(metrics, axis=0)

    return jax.jit(update)


# -----------------------------------------------------------------------------
# Evaluation and plotting
# -----------------------------------------------------------------------------


def rollout_deterministic(params, cfg: EnvCfg, seed: int = 0, max_steps: int | None = None, ic_mode: str = "approach_corridor"):
    """Fast deterministic evaluation using a JIT-compiled scan."""
    max_steps = int(max_steps or cfg.max_steps)

    def scan_rollout():
        key = jax.random.PRNGKey(seed)
        reset_fn = reset_fn_for_ic_mode(ic_mode)
        state0 = reset_fn(key, cfg._replace(randomize=0.0))

        def body(carry, t):
            state, done, total = carry
            obs = observe(state, cfg)
            residual = deterministic_action(params, obs[None, :])[0]
            _, state2, reward, done2, info = step_one(jax.random.fold_in(key, t), state, residual, cfg)
            state_next = jnp.where(done, state, state2)
            reward_eff = jnp.where(done, 0.0, reward)
            info_eff = jnp.where(done, jnp.zeros_like(info), info)
            done_next = done | (done2 > 0.5)
            return (state_next, done_next, total + reward_eff), (state_next, residual, info_eff, done_next)

        init = (state0, jnp.array(False), jnp.array(0.0))
        (state_f, done_f, total), (states, actions, infos, dones) = jax.lax.scan(
            body, init, jnp.arange(max_steps)
        )
        states = jnp.concatenate([state0[None, :], states], axis=0)
        return states, actions, infos, dones, total

    states, actions, infos, dones, total = jax.jit(scan_rollout)()
    states_np = np.array(states)
    actions_np = np.array(actions)
    infos_np = np.array(infos)
    dones_np = np.array(dones)
    if np.any(dones_np):
        n = int(np.argmax(dones_np)) + 1
        states_np = states_np[: n + 1]
        actions_np = actions_np[:n]
        infos_np = infos_np[:n]
    return states_np, actions_np, infos_np, float(total)


def rollout_batch_summary(params, cfg: EnvCfg, seed: int = 0, n: int = 1, max_steps: int | None = None, ic_mode: str = "approach_corridor"):
    """Vectorized deterministic evaluation over many independent IC samples."""
    max_steps = int(max_steps or cfg.max_steps)
    n = int(n)
    reset_fn = reset_fn_for_ic_mode(ic_mode)
    fuel_gate = jnp.asarray(
        _fuelmask.fuel_mask_value(
            float(cfg.fuel_scale),
            mode=cfg.fuel_mask_mode,
            center=float(cfg.fuel_mask_center),
            halfwidth=float(cfg.fuel_mask_halfwidth),
        ),
        dtype=jnp.float32,
    )

    def single_rollout(key):
        state0 = reset_fn(key, cfg._replace(randomize=0.0))

        def body(carry, t):
            state, done, total, final_info, prev_info, phase_acc, peak_lat_real, peak_lat_nom, res_raw, res_masked, n_active = carry
            obs = observe(state, cfg)
            residual = deterministic_action(params, obs[None, :])[0]
            masked_residual = residual * fuel_gate
            _, state2, reward, done2, info = step_one(jax.random.fold_in(key, t), state, residual, cfg)
            active = ~done
            state_next = jnp.where(active, state2, state)
            reward_eff = jnp.where(active, reward, 0.0)
            done_next = done | (done2 > 0.5)
            prev_info_next = jnp.where(active, final_info, prev_info)
            final_info_next = jnp.where(active, info, final_info)

            active_float = (~done).astype(jnp.float32)
            drag = info[12]
            gate = info[13]
            deflect = info[10]
            phase = info[14]
            is_terminal = jnp.where(phase > 0.5, 1.0, 0.0)
            coast_sel = 1.0 - is_terminal
            phase_acc_next = phase_acc + active_float * jnp.array([
                coast_sel * drag,
                is_terminal * drag,
                coast_sel * deflect,
                is_terminal * deflect,
                coast_sel,
                is_terminal,
            ])
            peak_lat_real_next = jnp.maximum(peak_lat_real, active_float * info[15])
            peak_lat_nom_next = jnp.maximum(peak_lat_nom, active_float * info[16])
            res_raw_next = res_raw + active_float * jnp.mean(jnp.abs(residual))
            res_masked_next = res_masked + active_float * jnp.mean(jnp.abs(masked_residual))
            n_active_next = n_active + active_float
            return (state_next, done_next, total + reward_eff, final_info_next, prev_info_next, phase_acc_next, peak_lat_real_next, peak_lat_nom_next, res_raw_next, res_masked_next, n_active_next), None

        init_info = jnp.zeros((17,), dtype=jnp.float32)
        init_phase_acc = jnp.zeros((6,), dtype=jnp.float32)
        init = (state0, jnp.array(False), jnp.array(0.0), init_info, init_info, init_phase_acc, jnp.array(0.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32))
        state_f, done_f, total, info_f, prev_info_f, phase_acc_f, peak_lat_real, peak_lat_nom, res_raw_f, res_masked_f, n_active_f = jax.lax.scan(body, init, jnp.arange(max_steps))[0]
        res_raw_mean = res_raw_f / jnp.maximum(n_active_f, 1.0)
        res_masked_mean = res_masked_f / jnp.maximum(n_active_f, 1.0)
        phase_corrected_omega = 0.5 * (info_f[5] + prev_info_f[5])
        init_yz = state0[1:3]
        return jnp.concatenate([
            jnp.array([total, done_f.astype(jnp.float32)]),
            state_f[0:6],
            state_f[10:13],
            info_f[0:6],
            jnp.array([phase_corrected_omega]),
            init_yz,
            jnp.array([prev_info_f[5]]),  # D_g11_02 PVR fix: raw omega_mag at T-1
            jnp.array([info_f[9], info_f[10], info_f[11]]),  # C03: fin torque, grid deflect, grid ctrl force
            phase_acc_f,
            jnp.array([peak_lat_real, peak_lat_nom]),  # C07: peak realized/nominal lateral RCS torque (N*m)
            jnp.array([res_raw_mean, res_masked_mean, n_active_f]),  # C04: residual action magnitude
            jnp.array([state_f[13]]),  # H_g69_04: final mass (fuel-remaining diagnostic)
        ])

    @jax.jit
    def run(keys):
        return jax.vmap(single_rollout)(keys)

    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    arr = np.array(run(keys))
    total = arr[:, 0]
    done = arr[:, 1]
    height = arr[:, 2]
    pos_yz = arr[:, 3:5]
    vel = arr[:, 5:8]
    omega_vec = arr[:, 8:11]
    info = arr[:, 11:17]
    raw_safe = info[:, 0]
    touched = info[:, 1]
    lateral = info[:, 2]
    speed = info[:, 3]
    tilt = info[:, 4]
    omega = info[:, 5]
    phase_corrected_omega = arr[:, 17]
    # D_g11_02 PVR fix: compute phase_variance_ratio from raw (unaveraged) omega at last 2 steps
    prev_omega_raw = arr[:, 20]  # raw omega_mag at step T-1 (from prev_info_f[5])
    fin_torque_terminal = arr[:, 21]      # terminal fin aero torque norm (info[9])
    grid_deflect_terminal = arr[:, 22]    # terminal grid deflection norm (info[10])
    grid_ctrl_force_terminal = arr[:, 23]  # terminal grid control force norm (info[11])
    phase_acc = arr[:, 24:30]
    rcs_lat_peak_realized = arr[:, 30]
    rcs_lat_peak_nominal = arr[:, 31]
    residual_action_mag_mean = arr[:, 32]
    residual_action_mag_masked_mean = arr[:, 33]
    active_steps_mean = arr[:, 34]
    # ── H_g69_04 fuel-remaining diagnostic (final mass at last active step) ──
    final_mass = arr[:, 35]
    _mass_empty = float(cfg.mass_empty)
    _mass_full_init = float(cfg.mass_empty + (cfg.mass_full - cfg.mass_empty) * cfg.fuel_scale)
    fuel_remaining_fraction = np.clip((final_mass - _mass_empty) / max(_mass_full_init - _mass_empty, 1e-6), 0.0, None)
    fuel_exhausted = fuel_remaining_fraction <= 1e-6
    fuel_remaining_fraction_touched = fuel_remaining_fraction[touched.astype(bool)] if np.any(touched.astype(bool)) else np.array([0.0])
    grid_deflect_sat_frac = float(np.mean(grid_deflect_terminal >= 0.9 * _CL_MAX))
    omega_raw = omega  # raw omega_mag at step T (from info_f[5])
    # Per-trajectory PVR: |odd - even| / max(|odd|, |even|, 1e-3)
    cross_diff = np.abs(omega_raw - prev_omega_raw)
    max_mag_pvr = np.maximum(np.maximum(np.abs(omega_raw), np.abs(prev_omega_raw)), 1e-3)
    pvr_per_traj = cross_diff / max_mag_pvr
    phase_variance_ratio = float(np.mean(pvr_per_traj))
    phase_variance_ratio_max = float(np.max(pvr_per_traj))
    phase_corrected_safe = (
        (touched > 0.5)
        & (lateral < cfg.landing_radius)
        & (speed < cfg.landing_speed)
        & (tilt < cfg.landing_tilt)
        & (phase_corrected_omega < cfg.landing_omega)
    )
    jiang_safe = (
        (height <= 1e-4)
        & (lateral < 3.0)
        & (speed < 2.0)
        & (tilt < 0.05)
        & (phase_corrected_omega < 0.10)
    )
    init_yz = arr[:, 18:20]
    init_radius = np.linalg.norm(init_yz, axis=1)
    per_traj_initial_y = init_yz[:, 0].tolist()
    per_traj_initial_z = init_yz[:, 1].tolist()
    per_traj_initial_r = init_radius.tolist()
    per_traj_height = height.tolist()
    per_traj_lateral = lateral.tolist()
    per_traj_speed = speed.tolist()
    per_traj_tilt = tilt.tolist()
    per_traj_omega = omega.tolist()
    per_traj_phase_corrected_omega = phase_corrected_omega.tolist()
    per_traj_omega_y = omega_vec[:, 1].tolist()  # FIX: actual omega_y (pitch axis)
    per_traj_touched = touched.astype(bool).tolist()
    per_traj_safe = phase_corrected_safe.astype(bool).tolist()
    # ── gen54_pod9 C01: far-outer zero-tail residual-mask diagnostic ──
    # Count outward-rim slide-out / never-touch trajectories in the far-outer
    # radial bin (initial_r >= radius). The contract target is gt50/gt100/never
    # == 0 for the far-outer bin while inner-disk behavior is preserved.
    _far_mask_r = float(cfg.far_outer_mask_radius)
    _far_sel = init_radius >= _far_mask_r
    far_outer_n = int(np.sum(_far_sel))
    far_outer_slideout_gt50 = int(np.sum(_far_sel & (lateral > 50.0)))
    far_outer_slideout_gt100 = int(np.sum(_far_sel & (lateral > 100.0)))
    far_outer_never_touch = int(np.sum(_far_sel & (~(touched > 0.5))))
    far_outer_jiang = float(np.mean(jiang_safe[_far_sel])) if far_outer_n else 0.0
    far_outer_lateral_max = float(np.max(lateral[_far_sel])) if far_outer_n else 0.0
    residual_action_mag_mean_pooled = float(np.mean(residual_action_mag_mean))
    residual_action_mag_masked_mean_pooled = float(np.mean(residual_action_mag_masked_mean))
    residual_action_mag_masked_max_pooled = float(np.max(residual_action_mag_masked_mean))
    # ── C06: per-axis strict-Jiang pass decomposition (touch-conditioned) ──
    _tch = touched > 0.5
    tilt_pass = _tch & (tilt < _reward.JIANG_TILT_GATE_RAD)
    speed_pass = _tch & (speed < _reward.JIANG_SPEED_GATE_MPS)
    lateral_pass = _tch & (lateral < _reward.JIANG_LATERAL_GATE_M)
    omega_pass = _tch & (phase_corrected_omega < _reward.JIANG_OMEGA_GATE_RADPS)
    tilt_pass_rate = float(np.mean(tilt_pass))
    speed_pass_rate = float(np.mean(speed_pass))
    lateral_pass_rate = float(np.mean(lateral_pass))
    omega_pass_rate = float(np.mean(omega_pass))
    per_traj_tilt_pass = tilt_pass.astype(bool).tolist()
    per_traj_speed_pass = speed_pass.astype(bool).tolist()
    # Terminal velocity decomposition (vertical vs lateral) for boundary speed
    # attribution: the Jiang speed gate is mostly residual lateral velocity.
    per_traj_vx = vel[:, 0].tolist()
    per_traj_vlat = np.linalg.norm(vel[:, 1:3], axis=1).tolist()

    coast_drag_sum = phase_acc[:, 0]
    term_drag_sum = phase_acc[:, 1]
    coast_deflect_sum = phase_acc[:, 2]
    term_deflect_sum = phase_acc[:, 3]
    coast_steps = phase_acc[:, 4]
    term_steps = phase_acc[:, 5]
    total_coast_steps = float(np.maximum(coast_steps.sum(), 1))
    total_term_steps = float(np.maximum(term_steps.sum(), 1))
    coast_drag_mean = float(coast_drag_sum.sum() / total_coast_steps)
    term_drag_mean = float(term_drag_sum.sum() / total_term_steps)
    coast_deflect_mean = float(coast_deflect_sum.sum() / total_coast_steps)
    term_deflect_mean = float(term_deflect_sum.sum() / total_term_steps)
    drag_reduction_ratio = float(coast_drag_mean / (term_drag_mean + 1e-8))

    return {
        "n": n,
        "safe_rate": float(np.mean(phase_corrected_safe)),
        "phase_corrected_safe_rate": float(np.mean(phase_corrected_safe)),
        "raw_safe_rate": float(np.mean(raw_safe)),
        "safe_rate_jiang_2024": float(np.mean(jiang_safe)),
        "touch_rate": float(np.mean(touched)),
        "done_rate": float(np.mean(done)),
        "return_mean": float(np.mean(total)),
        "return_std": float(np.std(total)),
        "final_height_mean_m": float(np.mean(height)),
        "final_lateral_mean_m": float(np.mean(lateral)),
        "final_lateral_p95_m": float(np.percentile(lateral, 95)),
        "final_lateral_max_m": float(np.max(lateral)),
        "final_speed_mean_mps": float(np.mean(speed)),
        "final_speed_p95_mps": float(np.percentile(speed, 95)),
        "final_tilt_mean_rad": float(np.mean(tilt)),
        "final_tilt_p95_rad": float(np.percentile(tilt, 95)),
        "final_omega_mean_radps": float(np.mean(omega)),
        "final_omega_p95_radps": float(np.percentile(omega, 95)),
        "phase_corrected_omega_mean_radps": float(np.mean(phase_corrected_omega)),
        "phase_corrected_omega_p95_radps": float(np.percentile(phase_corrected_omega, 95)),
        "phase_variance_ratio": phase_variance_ratio,  # D_g11_02 fix: from raw omega T/T-1
        "phase_variance_ratio_max": phase_variance_ratio_max,
        "phase_variance_ratio_per_traj": pvr_per_traj.tolist(),
        "omega_vector_norm_mean_radps": float(np.mean(np.linalg.norm(omega_vec, axis=1))),
        "final_yz_radius_mean_m": float(np.mean(np.linalg.norm(pos_yz, axis=1))),
        "fin_torque_mean_Nm": float(np.mean(fin_torque_terminal)),
        "grid_fin_deflection_mean_rad": float(np.mean(grid_deflect_terminal)),
        "grid_fin_deflection_saturation_fraction": grid_deflect_sat_frac,
        "grid_fin_control_force_mean_N": float(np.mean(grid_ctrl_force_terminal)),
        "grid_fin_authority_cl_y": float(_CL_Y),
        "grid_fin_authority_cl_z": float(_CL_Z),
        "grid_fin_authority_roll_cl": float(_CL_ROLL),
        "rcs_lateral_peak_realized_nm_mean": float(np.mean(rcs_lat_peak_realized)),
        "rcs_lateral_peak_realized_nm_p95": float(np.percentile(rcs_lat_peak_realized, 95)),
        "rcs_lateral_peak_nominal_nm_mean": float(np.mean(rcs_lat_peak_nominal)),
        "rcs_lateral_peak_nominal_nm_p95": float(np.percentile(rcs_lat_peak_nominal, 95)),
        "fin_station_x_used": float(_FIN_STATION_X),
        "fin_torque_damping_used": float(_FIN_DAMPING),
        "per_trajectory_omega": per_traj_phase_corrected_omega,
        "per_trajectory_omega_y": per_traj_omega_y,  # FIX: actual omega_y, not magnitude
        "terminal_drag_on": bool(cfg.terminal_drag_on > 0.5),
        "terminal_drag_gain": float(cfg.terminal_drag_gain),
        "terminal_drag_alt_low_m": float(cfg.terminal_drag_alt_low),
        "terminal_drag_alt_high_m": float(cfg.terminal_drag_alt_high),
        "terminal_drag_radial_gate_on": bool(cfg.terminal_drag_radial_gate_on > 0.5),
        "terminal_drag_far_gain": float(cfg.terminal_drag_far_gain),
        "terminal_drag_far_radius_m": float(cfg.terminal_drag_far_radius),
        "terminal_drag_radial_width_m": float(cfg.terminal_drag_radial_width),
        "coast_drag_integral_sum": float(coast_drag_sum.sum()),
        "terminal_drag_integral_sum": float(term_drag_sum.sum()),
        "coast_drag_mean_per_step": coast_drag_mean,
        "terminal_drag_mean_per_step": term_drag_mean,
        "coast_grid_deflect_mean": coast_deflect_mean,
        "terminal_grid_deflect_mean": term_deflect_mean,
        "coast_steps_total": float(coast_steps.sum()),
        "terminal_steps_total": float(term_steps.sum()),
        "coast_terminal_drag_ratio": drag_reduction_ratio,
        "reward_tilt_knee_rad_used": float(cfg.reward_tilt_knee_rad),
        "reward_tilt_bonus_on_used": float(cfg.reward_tilt_bonus_on),
        "reward_tilt_bonus_scale_used": float(cfg.reward_tilt_bonus_scale),
        "reward_speed_knee_mps_used": float(cfg.reward_speed_knee_mps),
        "reward_speed_bonus_on_used": float(cfg.reward_speed_bonus_on),
        "reward_speed_bonus_scale_used": float(cfg.reward_speed_bonus_scale),
        "reward_lateral_bonus_on_used": float(cfg.reward_lateral_bonus_on),
        "reward_lateral_bonus_scale_used": float(cfg.reward_lateral_bonus_scale),
        "reward_lateral_floor_m_used": float(cfg.reward_lateral_floor_m),
        "tilt_pass_rate": tilt_pass_rate,
        "speed_pass_rate": speed_pass_rate,
        "lateral_pass_rate": lateral_pass_rate,
        "omega_pass_rate": omega_pass_rate,
        "residual_action_mag_mean": residual_action_mag_mean_pooled,
        "residual_action_mag_masked_mean": residual_action_mag_masked_mean_pooled,
        "residual_action_mag_masked_max": residual_action_mag_masked_max_pooled,
        "far_outer_mask_on": bool(cfg.far_outer_mask_on > 0.5),
        "far_outer_mask_radius_m": float(cfg.far_outer_mask_radius),
        "far_outer_mask_width_m": float(cfg.far_outer_mask_width),
        "far_outer_mask_mode": cfg.far_outer_mask_mode,
        "far_outer_n": far_outer_n,
        "far_outer_slideout_gt50": far_outer_slideout_gt50,
        "far_outer_slideout_gt100": far_outer_slideout_gt100,
        "far_outer_never_touch": far_outer_never_touch,
        "far_outer_jiang": far_outer_jiang,
        "far_outer_lateral_max_m": far_outer_lateral_max,
        "fuel_remaining_fraction_mean": float(np.mean(fuel_remaining_fraction)),
        "fuel_remaining_fraction_p05": float(np.percentile(fuel_remaining_fraction, 5)),
        "fuel_remaining_fraction_p95": float(np.percentile(fuel_remaining_fraction, 95)),
        "fuel_remaining_fraction_touched_mean": float(np.mean(fuel_remaining_fraction_touched)),
        "fuel_exhausted_rate": float(np.mean(fuel_exhausted)),
        "fuel_remaining_kg_mean": float(np.mean(final_mass - _mass_empty)),
        "per_traj": {
            "initial_y": per_traj_initial_y,
            "initial_z": per_traj_initial_z,
            "initial_r": per_traj_initial_r,
            "height": per_traj_height,
            "lateral": per_traj_lateral,
            "speed": per_traj_speed,
            "tilt": per_traj_tilt,
            "omega": per_traj_omega,
            "omega_y": per_traj_omega_y,  # FIX: actual omega_y data
            "phase_corrected_omega": per_traj_phase_corrected_omega,
            "touched": per_traj_touched,
            "safe": per_traj_safe,
            "tilt_pass": per_traj_tilt_pass,
            "speed_pass": per_traj_speed_pass,
            "vx": per_traj_vx,
            "vlat": per_traj_vlat,
            "fuel_remaining_fraction": fuel_remaining_fraction.tolist(),
        },
    }

def params_signature(params) -> str:
    """SHA-256 over the ordered JAX parameter leaves (bitwise identity check)."""
    h = hashlib.sha256()
    for i, leaf in enumerate(jax.tree_util.tree_leaves(params)):
        arr = np.ascontiguousarray(np.asarray(leaf))
        h.update(str(i).encode("ascii"))
        h.update(arr.dtype.str.encode("ascii"))
        h.update(repr(arr.shape).encode("ascii"))
        h.update(arr.tobytes())
    return h.hexdigest()


def write_determinism_manifest(out_dir, args, env_cfg, ppo_cfg, params, eval_summary=None):
    """Record the execution fingerprint (deterministic recipe + seed + params
    signature) for the same-seed spread / determinism audit."""
    manifest = {
        "variant_name": "gen46_pod9/C07_dose_margin_guard",
        "pod_id": "gen46_pod9",
        "contract": "C07",
        "seed": int(args.seed),
        "replica_index": int(args.replica_index),
        "deterministic": bool(_DETERMINISTIC),
        "xla_flags": _XLA_FLAGS,
        "jax_config": {
            "jax_threefry_partitionable": bool(getattr(jax.config, "jax_threefry_partitionable", None)),
            "jax_default_matmul_precision": str(getattr(jax.config, "jax_default_matmul_precision", None)),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
        },
        "jax_version": str(jax.__version__),
        "devices": [str(d) for d in jax.devices()],
        "num_envs": int(ppo_cfg.num_envs),
        "num_steps": int(ppo_cfg.num_steps),
        "updates": int(ppo_cfg.updates),
        "residual_scale": float(env_cfg.residual_scale),
        "terminal_drag_on": float(env_cfg.terminal_drag_on),
        "terminal_drag_gain": float(env_cfg.terminal_drag_gain),
        "rcs_schedule_on": float(env_cfg.rcs_schedule_on),
        "fuel_scale": float(env_cfg.fuel_scale),
        "fin_station_x": float(env_cfg.fin_station_x),
        "integrator": str(env_cfg.integrator),
        "disk_radius_m": 1500.0,
        "params_signature_sha256": params_signature(params),
    }
    if eval_summary is not None:
        manifest["metrics"] = {
            "safe_rate_jiang_2024": eval_summary.get("safe_rate_jiang_2024"),
            "phase_corrected_safe_rate": eval_summary.get("phase_corrected_safe_rate"),
            "omega_y_safe_rate": eval_summary.get("omega_y_safe_rate"),
            "safe_rate": eval_summary.get("safe_rate"),
            "touch_rate": eval_summary.get("touch_rate"),
            "phase_dependence_delta": eval_summary.get("phase_dependence_delta"),
            "phase_variance_ratio": eval_summary.get("phase_variance_ratio"),
            "return_mean": eval_summary.get("return_mean"),
        }
        mh = hashlib.sha256()
        mh.update(json.dumps(eval_summary, sort_keys=True).encode("utf-8"))
        manifest["eval_summary_sha256"] = mh.hexdigest()
    with open(out_dir / "determinism_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return manifest


def plot_trajectory(states: np.ndarray, out_png: str):
    pos = states[:, 0:3]
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos[:, 1], pos[:, 2], pos[:, 0], linewidth=2)
    ax.scatter([0], [0], [0], s=80, marker="x")
    ax.scatter([pos[0, 1]], [pos[0, 2]], [pos[0, 0]], s=50, marker="o")
    ax.scatter([pos[-1, 1]], [pos[-1, 2]], [pos[-1, 0]], s=50, marker="^")
    ax.set_xlabel("y lateral [m]")
    ax.set_ylabel("z lateral [m]")
    ax.set_zlabel("x height [m]")
    ax.set_title("PPO rocket landing trajectory")
    ax.set_box_aspect((1, 1, 1.4))
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args():
    _default_env = EnvCfg()
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=800)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--replica-index", type=int, default=0, help="Replica index for the determinism manifest (same-seed spread probe).")
    p.add_argument("--init-height", type=float, default=_default_env.init_x)
    p.add_argument("--init-y", type=float, default=_default_env.init_y)
    p.add_argument("--init-z", type=float, default=_default_env.init_z)
    p.add_argument("--init-vx", type=float, default=_default_env.init_vx)
    p.add_argument("--init-vy", type=float, default=_default_env.init_vy)
    p.add_argument("--init-vz", type=float, default=_default_env.init_vz)
    p.add_argument("--init-yaw-deg", type=float, default=_default_env.init_yaw_deg)
    p.add_argument("--init-pitch-deg", type=float, default=float(os.environ.get("SWORDFISH_INIT_PITCH_DEG", _default_env.init_pitch_deg)), help="Base initial pitch attitude (deg).")
    p.add_argument("--stress-radial-offset-m", type=float, default=float(os.environ.get("SWORDFISH_STRESS_RADIAL_OFFSET", _default_env.stress_radial_offset_m)), help="Cross-regime stress: fix initial radial offset (m); 0 = uniform disk.")
    p.add_argument("--stress-tilt-deg", type=float, default=float(os.environ.get("SWORDFISH_STRESS_TILT_DEG", _default_env.stress_tilt_deg)), help="Cross-regime stress: pitch attitude offset added to init_pitch_deg (deg).")
    p.add_argument("--out-dir", type=str, default="runs_swordfish_finned_jax")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--guidance-residual-scale", type=float, default=float(os.environ.get("SWORDFISH_RESIDUAL_SCALE", _default_env.residual_scale)))
    p.add_argument("--rcs-torque-max", type=float, default=float(os.environ.get("SWORDFISH_RCS_TORQUE_MAX", _default_env.rcs_torque_max)), help="RCS terminal-attitude torque magnitude (N*m); canonical 180000, 2x=360000.")
    p.add_argument("--rcs-authority-scale", type=float, default=float(os.environ.get("SWORDFISH_RCS_AUTHORITY_SCALE", _default_env.rcs_authority_scale)), help="Flat multiplier on rcs_torque_max only (thrust untouched).")
    p.add_argument("--rcs-schedule-on", type=float, default=float(os.environ.get("SWORDFISH_RCS_SCHEDULE_ON", _default_env.rcs_schedule_on)), help="Enable radial RCS-authority schedule (1=on, 0=flat).")
    p.add_argument("--rcs-near-com-scale", type=float, default=float(os.environ.get("SWORDFISH_RCS_NEAR_COM_SCALE", _default_env.rcs_near_com_scale)), help="RCS authority gain near the pad (r->0).")
    p.add_argument("--rcs-far-outer-scale", type=float, default=float(os.environ.get("SWORDFISH_RCS_FAR_OUTER_SCALE", _default_env.rcs_far_outer_scale)), help="RCS authority gain in the far-outer bin (r->1500).")
    p.add_argument("--rcs-boost-radius", type=float, default=float(os.environ.get("SWORDFISH_RCS_BOOST_RADIUS", _default_env.rcs_boost_radius)), help="Radial boost boundary (m).")
    p.add_argument("--rcs-schedule-width", type=float, default=float(os.environ.get("SWORDFISH_RCS_SCHEDULE_WIDTH", _default_env.rcs_schedule_width)), help="Radial schedule transition width (m).")
    p.add_argument("--guidance-tgo", type=float, default=_default_env.guidance_tgo)
    p.add_argument("--guidance-lat-vel-k", type=float, default=float(os.environ.get("SWORDFISH_GUIDANCE_LAT_VEL_K", _default_env.guidance_lat_vel_k)),
                   help="Lateral velocity-damping coefficient in the ZEM/ZEV lateral law (canonical 4.0).")
    p.add_argument("--guidance-tgo-lat", type=float, default=float(os.environ.get("SWORDFISH_TGO_LAT", _default_env.guidance_tgo_lat)),
                   help="Decoupled lateral (y,z) ZEM/ZEV horizon (default = vertical guidance_tgo).")
    p.add_argument("--tgo-lat-mode", type=float, default=float(os.environ.get("SWORDFISH_TGO_LAT_MODE", _default_env.guidance_tgo_lat_mode)),
                   help="0 = decreasing lateral horizon, 1 = constant lateral horizon.")
    p.add_argument("--guidance-kp", type=float, default=_default_env.guidance_kp)
    p.add_argument("--guidance-kd", type=float, default=_default_env.guidance_kd)
    p.add_argument("--guidance-kd-roll", type=float,
                   default=float(os.environ.get("SWORDFISH_GUIDANCE_KD_ROLL", _default_env.guidance_kd_roll)),
                   help="Roll-axis RCS derivative gain (separate from pitch/yaw guidance_kd). Env hook SWORDFISH_GUIDANCE_KD_ROLL pins the composed-stack kd_roll=1.5 lever.")
    p.add_argument("--bridge-on", type=float, default=float(os.environ.get("SWORDFISH_BRIDGE_ON", _default_env.bridge_on)),
                   help="Enable the capture-preserving terminal bridge (vertical-speed nulling + verticalization).")
    p.add_argument("--bridge-h", type=float, default=float(os.environ.get("SWORDFISH_BRIDGE_H", _default_env.bridge_terminal_h)),
                   help="Terminal-bridge activation altitude (m).")
    p.add_argument("--bridge-v-td", type=float, default=float(os.environ.get("SWORDFISH_BRIDGE_V_TD", _default_env.bridge_v_td)),
                   help="Terminal-bridge soft-touchdown vertical speed target (m/s).")
    p.add_argument("--bridge-speed-k", type=float, default=float(os.environ.get("SWORDFISH_BRIDGE_SPEED_K", _default_env.bridge_speed_k)),
                   help="Terminal-bridge vertical-speed nulling gain (1/s).")
    p.add_argument("--bridge-vert-gain", type=float, default=float(os.environ.get("SWORDFISH_BRIDGE_VERT_GAIN", _default_env.bridge_verticalize_gain)),
                   help="Terminal-bridge lateral-acc taper gain at h=0 (1.0 -> pure vertical).")
    p.add_argument("--bridge-tgo-floor", type=float, default=float(os.environ.get("SWORDFISH_BRIDGE_TGO_FLOOR", _default_env.bridge_tgo_floor)),
                   help="ZEM/ZEV tgo lower clip (champion = 3.0).")
    p.add_argument("--bridge-pos-taper", type=float, default=float(os.environ.get("SWORDFISH_BRIDGE_POS_TAPER", _default_env.bridge_pos_taper)),
                   help="Terminal lateral POSITION (zem) taper at ground (1 = full taper, 0 = keep position nulling).")
    p.add_argument("--bridge-vel-taper", type=float, default=float(os.environ.get("SWORDFISH_BRIDGE_VEL_TAPER", _default_env.bridge_vel_taper)),
                   help="Terminal lateral VELOCITY (zev) taper at ground (1 = full taper, 0 = keep velocity nulling).")
    p.add_argument("--bridge-lat-vel-gain", type=float, default=float(os.environ.get("SWORDFISH_BRIDGE_LAT_VEL_GAIN", _default_env.bridge_lat_vel_gain)),
                   help="Extra multiplier on the lateral velocity-nulling term in the terminal phase.")
    p.add_argument("--randomize", type=float, default=_default_env.randomize)
    p.add_argument("--pos-noise", type=float, default=_default_env.pos_noise)
    p.add_argument("--vel-noise", type=float, default=_default_env.vel_noise)
    p.add_argument("--attitude-noise-deg", type=float, default=_default_env.attitude_noise_deg)
    p.add_argument("--omega-noise", type=float, default=_default_env.omega_noise)
    p.add_argument("--fins-on", type=float, default=float(os.environ.get("SWORDFISH_FINS_ON", _default_env.fins_on)), help="Enable/scale forward grid-fin aerodynamics (0 disables).")
    p.add_argument("--fin-force-on", type=float, default=float(os.environ.get("SWORDFISH_FIN_FORCE_ON", _default_env.fin_force_on)), help="Enable/scale passive grid-fin lift+drag force channel (0 disables).")
    p.add_argument("--fin-lift-on", type=float, default=float(os.environ.get("SWORDFISH_FIN_LIFT_ON", float(os.environ.get("SWORDFISH_FIN_FORCE_ON", _default_env.fin_force_on)))), help="Enable/scale passive grid-fin LIFT force component only (defaults to uniform fin-force multiplier).")
    p.add_argument("--fin-drag-on", type=float, default=float(os.environ.get("SWORDFISH_FIN_DRAG_ON", float(os.environ.get("SWORDFISH_FIN_FORCE_ON", _default_env.fin_force_on)))), help="Enable/scale passive grid-fin DRAG force component only (defaults to uniform fin-force multiplier).")
    p.add_argument("--fin-damping-on", type=float, default=float(os.environ.get("SWORDFISH_FIN_DAMPING_ON", _default_env.fin_damping_on)), help="Enable/scale passive grid-fin aero damping torque channel (0 disables).")
    p.add_argument("--fin-active-on", type=float, default=float(os.environ.get("SWORDFISH_FIN_ACTIVE_ON", _default_env.fin_active_on)), help="Enable/scale active grid-fin lateral force + roll torque channel (0 disables).")
    p.add_argument("--fin-station-x", type=float, default=_default_env.fin_station_x, help="Grid-fin station in body x meters from COM; forward/nose is positive.")
    p.add_argument("--fin-area-each", type=float, default=_default_env.fin_area_each, help="Projected/reference area per grid fin, m^2.")
    p.add_argument("--fin-cd0", type=float, default=_default_env.fin_cd0, help="Grid-fin profile drag coefficient.")
    p.add_argument("--fin-cd-alpha", type=float, default=_default_env.fin_cd_alpha, help="Grid-fin crossflow/induced drag coefficient.")
    p.add_argument("--fin-cl-alpha", type=float, default=_default_env.fin_cl_alpha, help="Grid-fin effective lift slope.")
    p.add_argument("--fin-alpha-stall", type=float, default=_default_env.fin_alpha_stall, help="Smooth stall/saturation angle scale in radians.")
    p.add_argument("--fin-torque-damping", type=float, default=_default_env.fin_torque_damping, help="Grid-fin aerodynamic damping coefficient.")
    p.add_argument("--grid-fin-control-max", type=float, default=_default_env.grid_fin_control_max, help="Active grid-fin deflection limit, rad-equivalent.")
    p.add_argument("--grid-fin-control-cl", type=float, default=_default_env.grid_fin_control_cl, help="Active grid-fin lateral-force slope per rad deflection.")
    p.add_argument("--grid-fin-roll-control-cl", type=float, default=_default_env.grid_fin_roll_control_cl, help="Active grid-fin roll-moment slope per rad deflection.")
    p.add_argument("--terminal-drag-on", type=float, default=float(os.environ.get("SWORDFISH_TERMINAL_DRAG_ON", _default_env.terminal_drag_on)), help="Master switch for terminal-only coast-drag ramp (1=active, 0=baseline unity drag).")
    p.add_argument("--terminal-drag-gain", type=float, default=float(os.environ.get("SWORDFISH_TERMINAL_DRAG_GAIN", _default_env.terminal_drag_gain)), help="Fin-drag multiplier in the terminal phase (h < terminal_drag_alt_low).")
    p.add_argument("--terminal-drag-alt-low", type=float, default=float(os.environ.get("SWORDFISH_TERMINAL_DRAG_ALT_LOW", _default_env.terminal_drag_alt_low)), help="Below this height (m) the terminal ramp is fully engaged.")
    p.add_argument("--terminal-drag-alt-high", type=float, default=float(os.environ.get("SWORDFISH_TERMINAL_DRAG_ALT_HIGH", _default_env.terminal_drag_alt_high)), help="Above this height (m) the terminal ramp is fully off.")
    p.add_argument("--terminal-drag-radial-gate-on", type=float, default=float(os.environ.get("SWORDFISH_TERMINAL_DRAG_RADIAL_GATE_ON", "1.0")), help="Radial-gate master switch for terminal drag gain (1=gate on initial_r; contract default 1.0 uniform g3).")
    p.add_argument("--terminal-drag-far-gain", type=float, default=float(os.environ.get("SWORDFISH_TERMINAL_DRAG_FAR_GAIN", "3.0")), help="Far-outer (r>far_radius) terminal drag multiplier (contract default = uniform g3).")
    p.add_argument("--terminal-drag-far-radius", type=float, default=float(os.environ.get("SWORDFISH_TERMINAL_DRAG_FAR_RADIUS", _default_env.terminal_drag_far_radius)), help="Radial boundary of the drag-gain gate (m).")
    p.add_argument("--terminal-drag-radial-width", type=float, default=float(os.environ.get("SWORDFISH_TERMINAL_DRAG_RADIAL_WIDTH", _default_env.terminal_drag_radial_width)), help="Sigmoid blend width across the gate radius (m).")
    p.add_argument("--reward-tilt-cliff", type=float, default=float(os.environ.get("SWORDFISH_REWARD_TILT_CLIFF", _default_env.reward_tilt_cliff)), help="Reward-side terminal +250 bonus tilt gate (metric untouched).")
    p.add_argument("--reward-speed-cliff", type=float, default=float(os.environ.get("SWORDFISH_REWARD_SPEED_CLIFF", _default_env.reward_speed_cliff)), help="Reward-side terminal +250 bonus speed gate (metric untouched).")
    p.add_argument("--reward-radius-cliff", type=float, default=float(os.environ.get("SWORDFISH_REWARD_RADIUS_CLIFF", _default_env.reward_radius_cliff)), help="Reward-side terminal +250 bonus lateral gate (metric untouched).")
    p.add_argument("--reward-tilt-knee", type=float, default=float(os.environ.get("SWORDFISH_REWARD_TILT_KNEE_RAD", _default_env.reward_tilt_knee_rad)), help="C06 terminal-reward tilt knee (treat 0.05 / revert 0.10).")
    p.add_argument("--reward-tilt-bonus-on", type=float, default=float(os.environ.get("SWORDFISH_REWARD_TILT_BONUS_ON", _default_env.reward_tilt_bonus_on)), help="C06 tilt-pass-shaped bonus master switch (1=on).")
    p.add_argument("--reward-tilt-bonus-scale", type=float, default=float(os.environ.get("SWORDFISH_REWARD_TILT_BONUS_SCALE", _default_env.reward_tilt_bonus_scale)), help="C06 tilt-pass-shaped bonus magnitude at tilt=0 (0 disables).")
    p.add_argument("--reward-speed-knee", type=float, default=float(os.environ.get("SWORDFISH_REWARD_SPEED_KNEE_MPS", _default_env.reward_speed_knee_mps)), help="C06 terminal-reward speed knee (treat 2.0 / revert 4.0 / ablation 3.0).")
    p.add_argument("--reward-speed-bonus-on", type=float, default=float(os.environ.get("SWORDFISH_REWARD_SPEED_BONUS_ON", _default_env.reward_speed_bonus_on)), help="C06 speed-pass-shaped bonus master switch (1=on).")
    p.add_argument("--reward-speed-bonus-scale", type=float, default=float(os.environ.get("SWORDFISH_REWARD_SPEED_BONUS_SCALE", _default_env.reward_speed_bonus_scale)), help="C06 speed-pass-shaped bonus magnitude at speed=0 (0 disables).")
    p.add_argument("--reward-lateral-bonus-on", type=float, default=float(os.environ.get("SWORDFISH_REWARD_LATERAL_BONUS_ON", _default_env.reward_lateral_bonus_on)), help="C06 lateral-capture bonus master switch (default OFF).")
    p.add_argument("--reward-lateral-bonus-scale", type=float, default=float(os.environ.get("SWORDFISH_REWARD_LATERAL_BONUS_SCALE", _default_env.reward_lateral_bonus_scale)), help="C06 lateral bonus magnitude at lateral=0.")
    p.add_argument("--reward-lateral-floor", type=float, default=float(os.environ.get("SWORDFISH_REWARD_LATERAL_FLOOR_M", _default_env.reward_lateral_floor_m)), help="C06 lateral regularization floor (m).")
    p.add_argument("--guidance-gimbal-on", type=float, default=float(os.environ.get("SWORDFISH_GIMBAL_ON", _default_env.guidance_gimbal_on)),
                   help="Master switch for deterministic engine-gimbal attitude augmentation (0/1).")
    p.add_argument("--guidance-gimbal-kp", type=float, default=float(os.environ.get("SWORDFISH_GIMBAL_KP", _default_env.guidance_gimbal_kp)),
                   help="Gimbal deflection (rad) per rad attitude error.")
    p.add_argument("--guidance-gimbal-kd", type=float, default=float(os.environ.get("SWORDFISH_GIMBAL_KD", _default_env.guidance_gimbal_kd)),
                   help="Gimbal deflection (rad) per rad/s body rate (damping).")
    p.add_argument("--guidance-gimbal-max", type=float, default=float(os.environ.get("SWORDFISH_GIMBAL_MAX", _default_env.guidance_gimbal_max)),
                   help="Gimbal deflection limit (rad).")
    p.add_argument("--guidance-gimbal-feedforward", type=float, default=float(os.environ.get("SWORDFISH_GIMBAL_FEEDFORWARD", _default_env.guidance_gimbal_feedforward)),
                   help="Feed-forward subtraction of gimbal lateral accel from lateral command (0/1).")
    p.add_argument("--fuel-scale", type=float, default=float(os.environ.get("SWORDFISH_FUEL_SCALE", _default_env.fuel_scale)), help="Propellant-capacity multiplier applied ONLY to initial fuel mass (thrust/Isp unchanged).")
    p.add_argument("--fuel-mask-mode", choices=["guard", "disabled", "extremes_only", "all"], default=os.environ.get("SWORDFISH_FUEL_MASK_MODE", _default_env.fuel_mask_mode), help="Fuel-keyed residual mask mode (guard=null residual inside fuel1.5 neighborhood).")
    p.add_argument("--regime-gate-on", type=float, default=float(os.environ.get("SWORDFISH_REGIME_GATE_ON", _default_env.regime_gate_on)), help="Master switch for regime-gated lift-down (1=gate active, 0=global base lift).")
    p.add_argument("--gate-lift-scale", type=float, default=float(os.environ.get("SWORDFISH_GATE_LIFT_SCALE", _default_env.gate_lift_scale)), help="Lift scale engaged inside the altitude/tilt regime (<1).")
    p.add_argument("--gate-h-min", type=float, default=float(os.environ.get("SWORDFISH_GATE_H_MIN_M", _default_env.gate_h_min_m)), help="Regime altitude lower bound (m).")
    p.add_argument("--gate-h-max", type=float, default=float(os.environ.get("SWORDFISH_GATE_H_MAX_M", _default_env.gate_h_max_m)), help="Regime altitude upper bound (m).")
    p.add_argument("--gate-tilt-max", type=float, default=float(os.environ.get("SWORDFISH_GATE_TILT_MAX_RAD", _default_env.gate_tilt_max_rad)), help="Regime |tilt| upper bound (rad).")
    p.add_argument("--residual-mode", choices=["full", "rcs", "gimbal", "throttle", "gridfin", "aero", "none"], default="full")
    p.add_argument("--no-guidance", action="store_true", help="Disable the deterministic guidance prior and train raw actions.")
    p.add_argument("--controller-only", action="store_true", help="Set residual scale to zero and evaluate the guidance prior itself.")
    p.add_argument("--uniform-disk-eval", action="store_true",
                   help="Use uniform 1500m-disk IC sampling for T5 evaluation instead of fixed approach-corridor point.")
    p.add_argument("--ic-mode", choices=["approach_corridor", "uniform_disk"], default="approach_corridor",
                   help="Initial condition mode for evaluation. 'approach_corridor' uses the fixed single-point IC; 'uniform_disk' samples uniformly within 1500m in y-z.")
    p.add_argument("--eval-trajectories", type=int, default=1,
                   help="Number of deterministic trajectories to score after training.")
    p.add_argument("--plot-only-smoke", action="store_true", help="Run tiny training and still save a trajectory plot.")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--lr-schedule", choices=["linear_decay", "constant"], default="linear_decay",
                   help="LR schedule: linear_decay (SOTA) or constant (ablation)")
    p.add_argument("--ent-coef", type=float, default=0.0,
                   help="Entropy coefficient (default 0.0)")
    p.add_argument("--integrator", choices=["euler", "rk2", "rk2_full", "euler_half", "euler_quarter", "rk4"], default=str(os.environ.get("SWORDFISH_INTEGRATOR", "rk4")),
                   help="Integration scheme: euler (canonical family), rk2 (Heun second family), or rk2_full (midpoint RK2 with Lie-group quaternion).")
    p.add_argument("--integrator-substeps", type=int, default=int(os.environ.get("SWORDFISH_INTEGRATOR_SUBSTEPS", "1")),
                   help="Subdivide the 0.1s physics macro step into N sub-integrations (dt ladder: 1=0.10, 2=0.05, 4=0.025). Only rk2/rk2_full/rk4 honor it.")
    p.add_argument("--fin-incidence-on", type=float, default=float(os.environ.get("SWORDFISH_FIN_INCIDENCE_ON", _default_env.fin_incidence_on)), help="Static fin incidence master switch (1=on).")
    p.add_argument("--fin-incidence-y-deg", type=float, default=float(os.environ.get("SWORDFISH_FIN_INCIDENCE_Y_DEG", _default_env.fin_incidence_y_deg)), help="Static fin cant about body-y -> body-z force (deg).")
    p.add_argument("--fin-incidence-z-deg", type=float, default=float(os.environ.get("SWORDFISH_FIN_INCIDENCE_Z_DEG", _default_env.fin_incidence_z_deg)), help="Static fin cant about body-z -> body-y force (deg).")
    p.add_argument("--fin-incidence-cl", type=float, default=float(os.environ.get("SWORDFISH_FIN_INCIDENCE_CL", _default_env.fin_incidence_cl)), help="Static incidence lateral-force slope per rad.")
    p.add_argument("--fin-incidence-cd", type=float, default=float(os.environ.get("SWORDFISH_FIN_INCIDENCE_CD", _default_env.fin_incidence_cd)), help="Static incidence induced-drag coefficient per rad^2.")
    p.add_argument("--fin-incidence-radial-gate-on", type=float, default=float(os.environ.get("SWORDFISH_FIN_INCIDENCE_RADIAL_GATE_ON", _default_env.fin_incidence_radial_gate_on)), help="Radial gate master switch on static incidence.")
    p.add_argument("--fin-incidence-radial-radius", type=float, default=float(os.environ.get("SWORDFISH_FIN_INCIDENCE_RADIAL_RADIUS", _default_env.fin_incidence_radial_radius)), help="Lower radial gate radius (m).")
    p.add_argument("--fin-incidence-radial-upper", type=float, default=float(os.environ.get("SWORDFISH_FIN_INCIDENCE_RADIAL_UPPER", _default_env.fin_incidence_radial_upper)), help="Upper/handoff radial gate radius (m); 0=none.")
    p.add_argument("--fin-incidence-radial-width", type=float, default=float(os.environ.get("SWORDFISH_FIN_INCIDENCE_RADIAL_WIDTH", _default_env.fin_incidence_radial_width)), help="Radial sigmoid blend width (m).")
    p.add_argument("--fin-incidence-alt-gate-on", type=float, default=float(os.environ.get("SWORDFISH_FIN_INCIDENCE_ALT_GATE_ON", _default_env.fin_incidence_alt_gate_on)), help="Tapered altitude gate master switch on static incidence.")
    p.add_argument("--fin-incidence-alt-low", type=float, default=float(os.environ.get("SWORDFISH_FIN_INCIDENCE_ALT_LOW_M", _default_env.fin_incidence_alt_low_m)), help="Below this height incidence fully OFF.")
    p.add_argument("--fin-incidence-alt-high", type=float, default=float(os.environ.get("SWORDFISH_FIN_INCIDENCE_ALT_HIGH_M", _default_env.fin_incidence_alt_high_m)), help="Above this height incidence fully ON.")
    p.add_argument("--fin-incidence-alt-width", type=float, default=float(os.environ.get("SWORDFISH_FIN_INCIDENCE_ALT_WIDTH_M", _default_env.fin_incidence_alt_width_m)), help="Altitude sigmoid blend width (m).")
    p.add_argument("--gear-on", type=float, default=float(os.environ.get("SWORDFISH_GEAR_ON", _default_env.gear_on)), help="Passive landing-gear suspension master switch (1=on).")
    p.add_argument("--gear-contact-height-m", type=float, default=float(os.environ.get("SWORDFISH_GEAR_CONTACT_HEIGHT_M", _default_env.gear_contact_height_m)), help="Leg tip drop below COM (m).")
    p.add_argument("--gear-footprint-radius-m", type=float, default=float(os.environ.get("SWORDFISH_GEAR_FOOTPRINT_RADIUS_M", _default_env.gear_footprint_radius_m)), help="Leg tip radial offset in body y-z (m).")
    p.add_argument("--gear-n-legs", type=int, default=int(os.environ.get("SWORDFISH_GEAR_N_LEGS", _default_env.gear_n_legs)), help="Number of landing legs.")
    p.add_argument("--gear-spring-k-npm", type=float, default=float(os.environ.get("SWORDFISH_GEAR_SPRING_K_NPM", _default_env.gear_spring_k_npm)), help="Normal spring stiffness per leg (N/m).")
    p.add_argument("--gear-damper-c-nspm", type=float, default=float(os.environ.get("SWORDFISH_GEAR_DAMPER_C_NSPM", _default_env.gear_damper_c_nspm)), help="Normal damper per leg (N/(m/s)).")
    p.add_argument("--gear-friction-mu", type=float, default=float(os.environ.get("SWORDFISH_GEAR_FRICTION_MU", _default_env.gear_friction_mu)), help="Coulomb-like tangential friction coefficient.")
    p.add_argument("--gear-contact-restore-scale", type=float, default=float(os.environ.get("SWORDFISH_GEAR_CONTACT_RESTORE_SCALE", _default_env.gear_contact_restore_scale)), help="Restoring-torque scale (0 = zero-torque ablation).")
    p.add_argument("--gear-bottom-out-guard", type=float, default=float(os.environ.get("SWORDFISH_GEAR_BOTTOM_OUT_GUARD", _default_env.gear_bottom_out_guard)), help="Bottom-out guard: cap spring support below weight so gear always settles to ground (1=on).")
    p.add_argument("--gear-bottom-out-force-frac", type=float, default=float(os.environ.get("SWORDFISH_GEAR_BOTTOM_OUT_FORCE_FRAC", _default_env.gear_bottom_out_force_frac)), help="Fraction of per-leg weight the spring may support when guard on.")
    p.add_argument("--far-outer-mask-on", type=float, default=float(os.environ.get("SWORDFISH_FAR_OUTER_MASK_ON", _default_env.far_outer_mask_on)), help="Far-outer zero-tail residual mask master switch (1=zero residual in far-outer radial bin).")
    p.add_argument("--far-outer-mask-radius", type=float, default=float(os.environ.get("SWORDFISH_FAR_OUTER_MASK_RADIUS", _default_env.far_outer_mask_radius)), help="Far-outer residual-mask radial boundary on initial_r (m).")
    p.add_argument("--far-outer-mask-width", type=float, default=float(os.environ.get("SWORDFISH_FAR_OUTER_MASK_WIDTH", _default_env.far_outer_mask_width)), help="Far-outer residual-mask smoothstep ramp width (m).")
    p.add_argument("--far-outer-mask-mode", choices=["soft", "hard"], default=os.environ.get("SWORDFISH_FAR_OUTER_MASK_MODE", _default_env.far_outer_mask_mode), help="Far-outer residual-mask ramp mode (soft=smoothstep, hard=step).")
    p.add_argument("--dose-guard-on", type=float, default=float(os.environ.get("SWORDFISH_DOSE_GUARD_ON", "1")), help="C07 dose-margin guard master switch (1=on).")
    p.add_argument("--dose-guard-band-lo", type=float, default=float(os.environ.get("SWORDFISH_DOSE_GUARD_BAND_LO", "2.75")), help="C07 safe-dose band lower bound.")
    p.add_argument("--dose-guard-band-hi", type=float, default=float(os.environ.get("SWORDFISH_DOSE_GUARD_BAND_HI", "3.25")), help="C07 safe-dose band upper bound (saturation target).")
    p.add_argument("--dose-guard-sat-threshold", type=float, default=float(os.environ.get("SWORDFISH_DOSE_GUARD_SAT_THRESHOLD", "3.25")), help="C07 dose above which lateral back-off engages.")
    p.add_argument("--dose-guard-backoff-mode", type=str, default=os.environ.get("SWORDFISH_DOSE_GUARD_BACKOFF_MODE", "saturate"), help="C07 back-off mode (saturate|none).")
    p.add_argument("--dose-guard-scope", type=str, default=os.environ.get("SWORDFISH_DOSE_GUARD_SCOPE", "lateral"), help="C07 guard scope (lateral|all).")
    p.add_argument("--gate-mask", type=str, default=os.environ.get("SWORDFISH_GATE_MASK", ""),
                   help="Comma-separated gate ablation masks (coast_ramp,radial,rcs,speed_lever). "
                        "Each listed mechanism is independently ZEROED without touching PPO updates or residual_scale.")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        _iflags.resolve_integrator({"SWORDFISH_INTEGRATOR": str(args.integrator)})
        _iflags.resolve_substeps({"SWORDFISH_INTEGRATOR_SUBSTEPS": str(args.integrator_substeps)})
    except ValueError as _e:
        raise SystemExit(f"gen71_pod3 C05 integrator-flag validation failed: {_e}")
    print("C05 integrator flags:", _iflags.describe_cell(str(args.integrator), int(args.integrator_substeps)))
    print("Grid-fin station/authority config:", _grid_fin_aero.describe(_AUTH))

    # ── H_g27_01 gate-mask ablation switches (variant-local, metric-untouched) ──
    # The radial-gated coast-drag champion exposes three independently-removable
    # plant gates plus a reward-side speed lever:
    #   coast_ramp  -> terminal_drag_on = 0.0                  (terminal coast-drag altitude ramp)
    #   radial      -> terminal_drag_radial_gate_on = 0.0      (far-outer g3->g5 radial boost)
    #   rcs         -> rcs_schedule_on = 0.0                   (radial RCS-authority schedule)
    #   speed_lever -> reward_speed_knee_mps = 4.0, reward_speed_bonus_on/scale = 0.0
    # These zero ONLY the named mechanism; PPO updates and residual_scale are
    # untouched. Evaluator / metric / data-split / disk_radius are untouched.
    _gate_mask = {t.strip() for t in (args.gate_mask or "").split(",") if t.strip()}
    if "coast_ramp" in _gate_mask:
        args.terminal_drag_on = 0.0
    if "radial" in _gate_mask:
        args.terminal_drag_radial_gate_on = 0.0
    if "rcs" in _gate_mask:
        args.rcs_schedule_on = 0.0
    if "speed_lever" in _gate_mask:
        args.reward_speed_knee = 4.0
        args.reward_speed_bonus_on = 0.0
        args.reward_speed_bonus_scale = 0.0
    print("H_g27_01 gate-mask:", sorted(_gate_mask) if _gate_mask else "none")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_masks = {
        "full": (1.0, 1.0, 1.0, 1.0),
        "rcs": (0.0, 0.0, 1.0, 0.0),
        "gimbal": (1.0, 0.0, 0.0, 0.0),
        "throttle": (0.0, 1.0, 0.0, 0.0),
        "gridfin": (0.0, 0.0, 0.0, 1.0),
        "aero": (0.0, 0.0, 1.0, 1.0),
        "none": (0.0, 0.0, 0.0, 0.0),
    }
    rg, rt, rr, rf = mode_masks[args.residual_mode]
    _rr_x = float(os.environ.get("SWORDFISH_RESIDUAL_RCS_X", rr))
    _rr_y = float(os.environ.get("SWORDFISH_RESIDUAL_RCS_Y", rr))
    _rr_z = float(os.environ.get("SWORDFISH_RESIDUAL_RCS_Z", rr))
    # gen53_pod4 C01: continuous fuel-margin-tracked gear contact-height law.
    # Compute the contact-height multiplier BEFORE EnvCfg so the plant carries
    # the scheduled gear geometry. The continuous law is the DEFAULT plant;
    # "keyed" preserves the archived discrete schedule as the control arm and
    # "off" is the plain fixed-gear plant. Both laws match the archived keyed
    # anchors exactly at fuel0.7/1.5/2.0 and differ only in between.
    _gear_law_mode = str(os.environ.get("SWORDFISH_GEAR_CONTACT_LAW_MODE", "continuous"))
    if _gear_law_mode == "keyed":
        _gear_law_interp = "keyed_schedule"
        _gear_h_mult = _fuelsched.fuel_schedule_multiplier(
            float(args.fuel_scale),
            lo_val=float(os.environ.get("SWORDFISH_GEAR_H_SCHED_LO_MULT", 1.0)),
            anchor_val=1.0,
            hi_val=float(os.environ.get("SWORDFISH_GEAR_H_SCHED_HI_MULT", 0.667)),
        )
    elif _gear_law_mode == "continuous":
        _gear_law_interp = str(os.environ.get("SWORDFISH_GEAR_CONTACT_LAW_INTERP", "smoothstep"))
        # Fail-fast: continuous law must be monotone over the fuel grid.
        _contactlaw.assert_monotone(fuel_grid=("0.7", "1.5", "2.0"), mode=_gear_law_interp)
        _gear_h_mult = _contactlaw.contact_height_multiplier(
            float(args.fuel_scale), mode=_gear_law_interp,
        )
    else:
        _gear_law_interp = "off"
        _gear_h_mult = 1.0
    _gear_h_effective = float(args.gear_contact_height_m) * _gear_h_mult
    env_cfg = EnvCfg(
        init_x=args.init_height,
        init_y=args.init_y,
        init_z=args.init_z,
        init_vx=args.init_vx,
        init_vy=args.init_vy,
        init_vz=args.init_vz,
        init_yaw_deg=args.init_yaw_deg,
        init_pitch_deg=args.init_pitch_deg + args.stress_tilt_deg,
        stress_radial_offset_m=args.stress_radial_offset_m,
        stress_tilt_deg=args.stress_tilt_deg,
        randomize=args.randomize,
        pos_noise=args.pos_noise,
        vel_noise=args.vel_noise,
        attitude_noise_deg=args.attitude_noise_deg,
        omega_noise=args.omega_noise,
        fins_on=args.fins_on,
        fin_force_on=args.fin_force_on,
        fin_lift_on=args.fin_lift_on,
        fin_drag_on=args.fin_drag_on,
        fin_damping_on=args.fin_damping_on,
        fin_active_on=args.fin_active_on,
        fin_station_x=_AUTH.get("fin_station_x", args.fin_station_x),
        fin_area_each=args.fin_area_each,
        fin_cd0=args.fin_cd0,
        fin_cd_alpha=args.fin_cd_alpha,
        fin_cl_alpha=args.fin_cl_alpha,
        fin_alpha_stall=args.fin_alpha_stall,
        fin_torque_damping=_AUTH.get("fin_torque_damping", args.fin_torque_damping),
        fin_incidence_on=args.fin_incidence_on,
        fin_incidence_y_deg=args.fin_incidence_y_deg,
        fin_incidence_z_deg=args.fin_incidence_z_deg,
        fin_incidence_cl=args.fin_incidence_cl,
        fin_incidence_cd=args.fin_incidence_cd,
        fin_incidence_radial_gate_on=args.fin_incidence_radial_gate_on,
        fin_incidence_radial_radius=args.fin_incidence_radial_radius,
        fin_incidence_radial_upper=args.fin_incidence_radial_upper,
        fin_incidence_radial_width=args.fin_incidence_radial_width,
        fin_incidence_alt_gate_on=args.fin_incidence_alt_gate_on,
        fin_incidence_alt_low_m=args.fin_incidence_alt_low,
        fin_incidence_alt_high_m=args.fin_incidence_alt_high,
        fin_incidence_alt_width_m=args.fin_incidence_alt_width,
        grid_fin_control_max=_AUTH.get("grid_fin_control_max", args.grid_fin_control_max),
        grid_fin_control_cl=_AUTH.get("grid_fin_control_cl", args.grid_fin_control_cl),
        grid_fin_roll_control_cl=_AUTH.get("grid_fin_roll_control_cl", args.grid_fin_roll_control_cl),
        fuel_scale=args.fuel_scale,
        fuel_mask_mode=args.fuel_mask_mode,
        fuel_mask_center=float(os.environ.get("SWORDFISH_FUEL_MASK_CENTER", 1.5)),
        fuel_mask_halfwidth=float(os.environ.get("SWORDFISH_FUEL_MASK_HALFWIDTH", 0.25)),
        rcs_torque_max=args.rcs_torque_max,
        rcs_authority_scale=args.rcs_authority_scale,
        rcs_schedule_on=args.rcs_schedule_on,
        rcs_near_com_scale=args.rcs_near_com_scale,
        rcs_far_outer_scale=args.rcs_far_outer_scale,
        rcs_boost_radius=args.rcs_boost_radius,
        rcs_schedule_width=args.rcs_schedule_width,
        terminal_drag_on=args.terminal_drag_on,
        terminal_drag_gain=args.terminal_drag_gain,
        terminal_drag_alt_low=args.terminal_drag_alt_low,
        terminal_drag_alt_high=args.terminal_drag_alt_high,
        terminal_drag_radial_gate_on=args.terminal_drag_radial_gate_on,
        terminal_drag_far_gain=args.terminal_drag_far_gain,
        terminal_drag_far_radius=args.terminal_drag_far_radius,
        terminal_drag_radial_width=args.terminal_drag_radial_width,
        reward_tilt_cliff=args.reward_tilt_cliff,
        reward_speed_cliff=args.reward_speed_cliff,
        reward_radius_cliff=args.reward_radius_cliff,
        reward_tilt_knee_rad=args.reward_tilt_knee,
        reward_tilt_bonus_on=args.reward_tilt_bonus_on,
        reward_tilt_bonus_scale=args.reward_tilt_bonus_scale,
        reward_speed_knee_mps=args.reward_speed_knee,
        reward_speed_bonus_on=args.reward_speed_bonus_on,
        reward_speed_bonus_scale=args.reward_speed_bonus_scale,
        reward_lateral_bonus_on=args.reward_lateral_bonus_on,
        reward_lateral_bonus_scale=args.reward_lateral_bonus_scale,
        reward_lateral_floor_m=args.reward_lateral_floor,
        regime_gate_on=args.regime_gate_on,
        gate_lift_scale=args.gate_lift_scale,
        gate_h_min_m=args.gate_h_min,
        gate_h_max_m=args.gate_h_max,
        gate_tilt_max_rad=args.gate_tilt_max,
        guidance_on=0.0 if args.no_guidance else 1.0,
        guidance_tgo=args.guidance_tgo,
        guidance_lat_vel_k=args.guidance_lat_vel_k,
        guidance_tgo_lat=args.guidance_tgo_lat,
        guidance_tgo_lat_mode=args.tgo_lat_mode,
        guidance_kp=args.guidance_kp,
        guidance_kd=args.guidance_kd,
        guidance_kd_roll=args.guidance_kd_roll,
        guidance_gimbal_on=args.guidance_gimbal_on,
        guidance_gimbal_kp=args.guidance_gimbal_kp,
        guidance_gimbal_kd=args.guidance_gimbal_kd,
        guidance_gimbal_max=args.guidance_gimbal_max,
        guidance_gimbal_feedforward=args.guidance_gimbal_feedforward,
        bridge_on=args.bridge_on,
        bridge_terminal_h=args.bridge_h,
        bridge_v_td=args.bridge_v_td,
        bridge_speed_k=args.bridge_speed_k,
        bridge_verticalize_gain=args.bridge_vert_gain,
        bridge_tgo_floor=args.bridge_tgo_floor,
        bridge_pos_taper=args.bridge_pos_taper,
        bridge_vel_taper=args.bridge_vel_taper,
        bridge_lat_vel_gain=args.bridge_lat_vel_gain,
        residual_scale=0.0 if args.controller_only else args.guidance_residual_scale,
        residual_gimbal=rg,
        residual_throttle=rt,
        residual_rcs=rr,
        residual_gridfin=rf,
        residual_rcs_x=_rr_x,
        residual_rcs_y=_rr_y,
        residual_rcs_z=_rr_z,
        far_outer_mask_on=args.far_outer_mask_on,
        far_outer_mask_radius=args.far_outer_mask_radius,
        far_outer_mask_width=args.far_outer_mask_width,
        far_outer_mask_mode=args.far_outer_mask_mode,
        divert_on=float(os.environ.get("SWORDFISH_DIVERT_ON", 1.0)),
        divert_gate_on=float(os.environ.get("SWORDFISH_DIVERT_GATE_ON", 1.0)),
        divert_sign=float(os.environ.get("SWORDFISH_DIVERT_SIGN", -1.0)),
        divert_gain=float(os.environ.get("SWORDFISH_DIVERT_GAIN", 0.5)),
        divert_gate_radius=float(os.environ.get("SWORDFISH_DIVERT_GATE_RADIUS", 1124.0)),
        divert_gate_width=float(os.environ.get("SWORDFISH_DIVERT_GATE_WIDTH", 60.0)),
        divert_vr_only=float(os.environ.get("SWORDFISH_DIVERT_VR_ONLY", 1.0)),
        divert_alt_gate_on=float(os.environ.get("SWORDFISH_DIVERT_ALT_GATE_ON", 1.0)),
        divert_alt_low_m=float(os.environ.get("SWORDFISH_DIVERT_ALT_LOW_M", 80.0)),
        divert_alt_high_m=float(os.environ.get("SWORDFISH_DIVERT_ALT_HIGH_M", 400.0)),
        divert_tilt_cap_rad=float(os.environ.get("SWORDFISH_DIVERT_TILT_CAP_RAD", 0.25)),
        divert_placebo_axis=float(os.environ.get("SWORDFISH_DIVERT_PLACEBO_AXIS", 0.0)),
        divert_gain_sched_on=float(os.environ.get("SWORDFISH_DIVERT_GAIN_SCHED_ON", 0.0)),
        divert_gain_sched_lo=float(os.environ.get("SWORDFISH_DIVERT_GAIN_SCHED_LO", 0.4)),
        divert_gain_sched_hi=float(os.environ.get("SWORDFISH_DIVERT_GAIN_SCHED_HI", 1.6)),
        descent_commit_on=float(os.environ.get("SWORDFISH_DESCENT_COMMIT_ON", 0.0)),
        descent_commit_hi=float(os.environ.get("SWORDFISH_DESCENT_COMMIT_HI", 1.0)),
        descent_commit_gain=float(os.environ.get("SWORDFISH_DESCENT_COMMIT_GAIN", 1.0)),
        descent_commit_alt=float(os.environ.get("SWORDFISH_DESCENT_COMMIT_ALT", 80.0)),
        descent_commit_gate_radius=float(os.environ.get("SWORDFISH_DESCENT_COMMIT_GATE_RADIUS", 0.0)),
        fuel_tail_on=float(os.environ.get("SWORDFISH_FUEL_TAIL_ON", 0.0)),
        fuel_tail_alt_m=float(os.environ.get("SWORDFISH_FUEL_TAIL_ALT_M", 6.0)),
        fuel_tail_vx_max_mps=float(os.environ.get("SWORDFISH_FUEL_TAIL_VX_MAX_MPS", 1.0)),
        fuel_tail_scale=float(os.environ.get("SWORDFISH_FUEL_TAIL_SCALE", 0.0)),
        fuel_tail_tilt_gate_rad=float(os.environ.get("SWORDFISH_FUEL_TAIL_TILT_GATE_RAD", 0.03)),
        fuel_tail_lateral_gate_m=float(os.environ.get("SWORDFISH_FUEL_TAIL_LATERAL_GATE_M", 2.5)),
        fuel_tail_vlat_gate_mps=float(os.environ.get("SWORDFISH_FUEL_TAIL_VLAT_GATE_MPS", 0.5)),
        fuel_tail_max_initial_r=float(os.environ.get("SWORDFISH_FUEL_TAIL_MAX_INITIAL_R", 1124.0)),
        integrator=args.integrator,
        integrator_substeps=args.integrator_substeps,
        guidance_mode=os.environ.get('SWORDFISH_GUIDANCE_MODE', 'canonical'),
        gear_on=args.gear_on,
        gear_contact_height_m=_gear_h_effective,
        gear_footprint_radius_m=args.gear_footprint_radius_m,
        gear_n_legs=args.gear_n_legs,
        gear_spring_k_npm=args.gear_spring_k_npm,
        gear_damper_c_nspm=args.gear_damper_c_nspm,
        gear_friction_mu=args.gear_friction_mu,
        gear_contact_restore_scale=args.gear_contact_restore_scale,
        gear_bottom_out_guard=args.gear_bottom_out_guard,
        gear_bottom_out_force_frac=args.gear_bottom_out_force_frac,
        dose_guard_on=args.dose_guard_on,
        dose_guard_band_lo=args.dose_guard_band_lo,
        dose_guard_band_hi=args.dose_guard_band_hi,
        dose_guard_sat_threshold=args.dose_guard_sat_threshold,
        dose_guard_backoff_mode=args.dose_guard_backoff_mode,
        dose_guard_scope=args.dose_guard_scope,
    )
    ppo_cfg = PPOCfg(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        updates=args.updates,
        epochs=args.epochs,
        minibatches=args.minibatches,
        lr=args.lr,
        ent_coef=args.ent_coef,
        hidden=args.hidden,
    )
    if args.plot_only_smoke:
        ppo_cfg = ppo_cfg._replace(updates=min(args.updates, 5), num_envs=min(args.num_envs, 16), num_steps=min(args.num_steps, 64), epochs=1, minibatches=2)

    if ppo_cfg.num_envs * ppo_cfg.num_steps % ppo_cfg.minibatches != 0:
        raise ValueError("num_envs * num_steps must be divisible by minibatches")

    # Fail-fast: canonical disk radius is fixed at 1500.0 m.
    assert abs(DISK_RADIUS - 1500.0) < 1e-6, "disk_radius must be 1500.0"
    print("JAX devices:", jax.devices())
    print("Config:", env_cfg)
    print("PPO:", ppo_cfg)

    key = jax.random.PRNGKey(args.seed)
    key, k_params, k_reset = jax.random.split(key, 3)
    params = init_params(k_params, ppo_cfg.hidden)
    opt_state = adam_init(params)

    if args.checkpoint:
        with open(args.checkpoint, "rb") as f:
            params = pickle.load(f)
        print(f"Loaded checkpoint: {args.checkpoint}")

    ic_mode = "uniform_disk" if args.uniform_disk_eval else args.ic_mode
    train_reset_fn = reset_fn_for_ic_mode(ic_mode)
    reset_v = jax.vmap(lambda k: train_reset_fn(k, env_cfg))
    states = reset_v(jax.random.split(k_reset, ppo_cfg.num_envs))
    collect = make_collect_rollout(env_cfg, ppo_cfg, ic_mode=ic_mode)
    update_fn = make_ppo_update(ppo_cfg)

    for upd in range(ppo_cfg.updates):
        if args.lr_schedule == "constant":
            lr = ppo_cfg.lr
        else:
            lr = ppo_cfg.lr * (1.0 - upd / max(ppo_cfg.updates - 1, 1))
        states, key, rollout, last_value = collect(params, states, key)
        params, opt_state, key, metrics = update_fn(params, opt_state, rollout, last_value, key, lr)
        if upd % max(1, ppo_cfg.updates // 20) == 0 or upd == ppo_cfg.updates - 1:
            safe_rate = float(jnp.mean(rollout.infos[..., 0]))
            touched_rate = float(jnp.mean(rollout.infos[..., 1]))
            mean_ret = float(jnp.mean(jnp.sum(rollout.rewards, axis=0)))
            loss, pg, vf, ent = [float(x) for x in metrics]
            print(
                f"update {upd:5d}/{ppo_cfg.updates} | return {mean_ret:8.2f} | "
                f"safe-step {safe_rate:6.3f} | touch-step {touched_rate:6.3f} | "
                f"loss {loss:8.3f} pg {pg:8.3f} vf {vf:8.3f} ent {ent:6.3f}"
            )

    ckpt = out_dir / "ppo_rocket_params.pkl"
    with open(ckpt, "wb") as f:
        pickle.dump(params, f)
    print(f"Saved checkpoint: {ckpt}")

    if args.eval_trajectories <= 1:
        write_determinism_manifest(out_dir, args, env_cfg, ppo_cfg, params)

    if args.eval_trajectories > 1:
        summary = rollout_batch_summary(
            params,
            env_cfg,
            seed=args.seed + 1000,
            n=args.eval_trajectories,
            ic_mode=ic_mode,
        )
        summary["ic_mode"] = ic_mode
        summary["disk_radius"] = float(DISK_RADIUS)
        summary["seed"] = args.seed
        summary["bridge_on"] = float(env_cfg.bridge_on)
        summary["bridge_terminal_h"] = float(env_cfg.bridge_terminal_h)
        summary["fins_on_used"] = float(env_cfg.fins_on)
        summary["fin_force_on_used"] = float(env_cfg.fin_force_on)
        summary["fin_lift_on_used"] = float(env_cfg.fin_lift_on)
        summary["fin_drag_on_used"] = float(env_cfg.fin_drag_on)
        summary["fin_damping_on_used"] = float(env_cfg.fin_damping_on)
        summary["fin_active_on_used"] = float(env_cfg.fin_active_on)
        summary["fuel_scale_used"] = float(env_cfg.fuel_scale)
        summary["fuel_mask_mode_used"] = env_cfg.fuel_mask_mode
        summary["fuel_mask_description"] = _fuelmask.mask_description(env_cfg)
        summary["fuel_mask_gate_value"] = float(np.asarray(_fuelmask.fuel_mask_value(
            float(env_cfg.fuel_scale), mode=env_cfg.fuel_mask_mode,
            center=float(env_cfg.fuel_mask_center), halfwidth=float(env_cfg.fuel_mask_halfwidth))))
        summary["initial_fuel_mass_kg"] = float((env_cfg.mass_full - env_cfg.mass_empty) * env_cfg.fuel_scale)
        summary["initial_total_mass_kg"] = float(env_cfg.mass_empty + (env_cfg.mass_full - env_cfg.mass_empty) * env_cfg.fuel_scale)
        summary["regime_gate_on_used"] = float(env_cfg.regime_gate_on)
        summary["gate_lift_scale_used"] = float(env_cfg.gate_lift_scale)
        summary["gate_h_min_m_used"] = float(env_cfg.gate_h_min_m)
        summary["gate_h_max_m_used"] = float(env_cfg.gate_h_max_m)
        summary["gate_tilt_max_rad_used"] = float(env_cfg.gate_tilt_max_rad)
        summary["bridge_v_td"] = float(env_cfg.bridge_v_td)
        summary["bridge_speed_k"] = float(env_cfg.bridge_speed_k)
        summary["bridge_verticalize_gain"] = float(env_cfg.bridge_verticalize_gain)
        summary["bridge_tgo_floor"] = float(env_cfg.bridge_tgo_floor)
        # ── v1.2 controller-only enforcement: report what was ACTUALLY run ──
        summary["actual_updates"] = int(ppo_cfg.updates)
        summary["actual_residual_scale"] = float(env_cfg.residual_scale)
        summary["rcs_torque_max_used"] = float(env_cfg.rcs_torque_max)
        summary["rcs_authority_scale_used"] = float(env_cfg.rcs_authority_scale)
        summary["rcs_schedule_on_used"] = float(env_cfg.rcs_schedule_on)
        summary["rcs_near_com_scale_used"] = float(env_cfg.rcs_near_com_scale)
        summary["rcs_far_outer_scale_used"] = float(env_cfg.rcs_far_outer_scale)
        summary["rcs_boost_radius_used"] = float(env_cfg.rcs_boost_radius)
        summary["rcs_schedule_width_used"] = float(env_cfg.rcs_schedule_width)
        summary["dose_guard_on_used"] = float(env_cfg.dose_guard_on)
        summary["dose_guard_band_lo_used"] = float(env_cfg.dose_guard_band_lo)
        summary["dose_guard_band_hi_used"] = float(env_cfg.dose_guard_band_hi)
        summary["dose_guard_sat_threshold_used"] = float(env_cfg.dose_guard_sat_threshold)
        summary["dose_guard_backoff_mode_used"] = str(env_cfg.dose_guard_backoff_mode)
        summary["dose_guard_scope_used"] = str(env_cfg.dose_guard_scope)
        summary["dose_guard_envelope_on_used"] = float(_GUARD_CFG.get("envelope_on", 1.0))
        summary["dose_guard_effective_lateral_dose"] = _dosemask.effective_lateral_dose_py(env_cfg.rcs_far_outer_scale, _GUARD_CFG)
        summary["divert_on_used"] = float(env_cfg.divert_on)
        summary["divert_gate_on_used"] = float(env_cfg.divert_gate_on)
        summary["divert_sign_used"] = float(env_cfg.divert_sign)
        summary["divert_gain_used"] = float(env_cfg.divert_gain)
        summary["divert_gate_radius_used"] = float(env_cfg.divert_gate_radius)
        summary["divert_gate_width_used"] = float(env_cfg.divert_gate_width)
        summary["divert_vr_only_used"] = float(env_cfg.divert_vr_only)
        summary["divert_alt_gate_on_used"] = float(env_cfg.divert_alt_gate_on)
        summary["divert_alt_low_m_used"] = float(env_cfg.divert_alt_low_m)
        summary["divert_alt_high_m_used"] = float(env_cfg.divert_alt_high_m)
        summary["divert_tilt_cap_rad_used"] = float(env_cfg.divert_tilt_cap_rad)
        summary["divert_placebo_axis_used"] = float(env_cfg.divert_placebo_axis)
        summary["divert_gain_sched_on_used"] = float(env_cfg.divert_gain_sched_on)
        summary["divert_gain_sched_lo_used"] = float(env_cfg.divert_gain_sched_lo)
        summary["divert_gain_sched_hi_used"] = float(env_cfg.divert_gain_sched_hi)
        summary["divert_gain_sched_mult"] = _fuelsched.divert_gain_multiplier(env_cfg) if env_cfg.divert_gain_sched_on > 0.5 else 1.0
        summary["descent_commit_on_used"] = float(env_cfg.descent_commit_on)
        summary["descent_commit_hi_used"] = float(env_cfg.descent_commit_hi)
        summary["descent_commit_gain_used"] = float(env_cfg.descent_commit_gain)
        summary["descent_commit_alt_used"] = float(env_cfg.descent_commit_alt)
        summary["descent_commit_mult"] = _fuelsched.descent_commit_multiplier(env_cfg)
        summary["fuel_tail_on_used"] = float(env_cfg.fuel_tail_on)
        summary["fuel_tail_alt_m_used"] = float(env_cfg.fuel_tail_alt_m)
        summary["fuel_tail_vx_max_mps_used"] = float(env_cfg.fuel_tail_vx_max_mps)
        summary["fuel_tail_scale_used"] = float(env_cfg.fuel_tail_scale)
        summary["fuel_tail_tilt_gate_rad_used"] = float(env_cfg.fuel_tail_tilt_gate_rad)
        summary["fuel_tail_lateral_gate_m_used"] = float(env_cfg.fuel_tail_lateral_gate_m)
        summary["fuel_tail_vlat_gate_mps_used"] = float(env_cfg.fuel_tail_vlat_gate_mps)
        summary["fuel_tail_max_initial_r_used"] = float(env_cfg.fuel_tail_max_initial_r)
        summary["fuel_schedule_description"] = _fuelsched.describe(env_cfg)
        summary["residual_rcs_x_used"] = float(env_cfg.residual_rcs_x)
        summary["residual_rcs_y_used"] = float(env_cfg.residual_rcs_y)
        summary["residual_rcs_z_used"] = float(env_cfg.residual_rcs_z)
        summary["far_outer_mask_on_used"] = float(env_cfg.far_outer_mask_on)
        summary["far_outer_mask_radius_used"] = float(env_cfg.far_outer_mask_radius)
        summary["far_outer_mask_width_used"] = float(env_cfg.far_outer_mask_width)
        summary["far_outer_mask_mode_used"] = str(env_cfg.far_outer_mask_mode)
        summary["controller_only"] = bool(ppo_cfg.updates == 0 or float(env_cfg.residual_scale) <= 0.0)
        summary["integrator_used"] = str(env_cfg.integrator)
        summary["integrator_substeps_used"] = int(env_cfg.integrator_substeps)
        summary["integrator_dt_used"] = float(env_cfg.dt) / float(max(env_cfg.integrator_substeps, 1))
        summary["guidance_kd_roll_used"] = float(env_cfg.guidance_kd_roll)
        summary["gear_on_used"] = float(env_cfg.gear_on)
        summary["gear_contact_height_m_used"] = float(env_cfg.gear_contact_height_m)
        summary["gear_contact_law_mode_used"] = _gear_law_mode
        summary["gear_contact_law_interp_used"] = _gear_law_interp
        summary["gear_contact_law_mult"] = float(_gear_h_mult)
        summary["fuel_margin_used"] = float(_contactlaw.fuel_margin(env_cfg.fuel_scale))
        summary["gear_contact_law_description"] = _contactlaw.describe(
            env_cfg.fuel_scale, float(args.gear_contact_height_m), _gear_law_interp,
        )
        summary["gear_h_sched_effective_m"] = float(env_cfg.gear_contact_height_m)
        summary["gear_footprint_radius_m_used"] = float(env_cfg.gear_footprint_radius_m)
        summary["gear_n_legs_used"] = int(env_cfg.gear_n_legs)
        summary["gear_spring_k_npm_used"] = float(env_cfg.gear_spring_k_npm)
        summary["gear_damper_c_nspm_used"] = float(env_cfg.gear_damper_c_nspm)
        summary["gear_friction_mu_used"] = float(env_cfg.gear_friction_mu)
        summary["gear_contact_restore_scale_used"] = float(env_cfg.gear_contact_restore_scale)
        summary["gear_bottom_out_guard_used"] = float(env_cfg.gear_bottom_out_guard)
        summary["gear_bottom_out_force_frac_used"] = float(env_cfg.gear_bottom_out_force_frac)
        # ── C01 provenance labels: integrator / coast-drag gate / plant ──
        _prov = _provenance.build_provenance(env_cfg, ppo_cfg, float(DISK_RADIUS))
        for _k, _v in _prov.items():
            summary.setdefault(_k, _v)
        summary["provenance"] = _prov
        (out_dir / "provenance_labels.json").write_text(
            json.dumps(_prov, indent=2, sort_keys=True), encoding="utf-8"
        )
        with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        write_determinism_manifest(out_dir, args, env_cfg, ppo_cfg, params, eval_summary=summary)
        print(
            f"EvalBatch: n={summary['n']}, safe_rate={summary['safe_rate']:.6f}, "
            f"touch_rate={summary['touch_rate']:.6f}, return={summary['return_mean']:.1f}, "
            f"lateral_mean={summary['final_lateral_mean_m']:.2f} m, "
            f"speed_mean={summary['final_speed_mean_mps']:.2f} m/s, "
            f"tilt_mean={summary['final_tilt_mean_rad']:.3f} rad, "
            f"omega_mean={summary['final_omega_mean_radps']:.3f} rad/s"
        )
        return

    states_np, actions_np, infos_np, total = rollout_deterministic(params, env_cfg, seed=args.seed + 1000, ic_mode=ic_mode)
    np.savez(out_dir / "trajectory_eval.npz", states=states_np, actions=actions_np, infos=infos_np, total_return=total)
    last = states_np[-1]
    final_lateral = float(np.linalg.norm(last[1:3]))
    final_speed = float(np.linalg.norm(last[3:6]))
    final_tilt = float(infos_np[-1, 4]) if len(infos_np) else float("nan")
    final_omega = float(np.linalg.norm(last[10:13]))
    final_safe = (last[0] <= 1e-4 and final_lateral < env_cfg.landing_radius and final_speed < env_cfg.landing_speed and final_tilt < env_cfg.landing_tilt and final_omega < env_cfg.landing_omega)
    print(
        f"Eval: steps={len(states_np)-1}, return={total:.1f}, "
        f"height={last[0]:.2f} m, lateral={final_lateral:.2f} m, "
        f"speed={final_speed:.2f} m/s, tilt={final_tilt:.3f} rad, "
        f"omega={final_omega:.3f} rad/s, safe={final_safe}"
    )

    if not args.no_plot:
        out_png = out_dir / "trajectory_3d.png"
        plot_trajectory(states_np, str(out_png))
        print(f"Saved 3D view: {out_png}")


if __name__ == "__main__":
    main()
