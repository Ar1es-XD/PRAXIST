"""Rocket Booster Recovery v0: deterministic classical composite rocket controller.

Architecture
------------
* rolling ZEM/ZEV candidate guidance in the high-altitude phase;
* phase-scheduled terminal velocity-corridor / lateral PD guidance through
  first landing-leg contact;
* geometric body-axis PD attitude control;
* bounded allocation between engine gimbal and active grid fins;
* roll-rate damping through tightly capped roll-only RCS.

There are no learned parameters, neural-network calls, checkpoints, optimizer
states, or training routines in this module.  The three forbidden physical
channels (RCS pitch, RCS yaw, grid-fin roll) are constructed as literal zeros.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp


class ControllerMemory(NamedTuple):
    phase: jax.Array
    dwell: jax.Array
    p1_released: jax.Array
    p1_hold_steps: jax.Array
    prev_gimbal_rad: jax.Array
    prev_grid_rad: jax.Array
    prev_throttle: jax.Array
    prev_rcs_roll_nm: jax.Array
    prev_thrust_accel: jax.Array
    prev_body_x_ref: jax.Array
    prev_tgo_s: jax.Array


DIAGNOSTIC_COLUMNS = (
    "phase",
    "emergency",
    "tgo_s",
    "brake_margin_m",
    "dynamic_pressure_pa",
    "thrust_accel_x",
    "thrust_accel_y",
    "thrust_accel_z",
    "requested_torque_x_nm",
    "requested_torque_y_nm",
    "requested_torque_z_nm",
    "gimbal_y_rad",
    "gimbal_z_rad",
    "grid_y_rad",
    "grid_z_rad",
    "throttle",
    "rcs_roll_nm",
    "body_axis_error_rad",
)


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_norm(x: jax.Array, eps: float = 1e-8) -> jax.Array:
    return jnp.sqrt(jnp.sum(x * x) + eps)


def _unit(x: jax.Array) -> jax.Array:
    return x / _safe_norm(x)


def quat_to_rot(q: jax.Array) -> jax.Array:
    q = _unit(q)
    q0, q1, q2, q3 = q
    return jnp.array(
        [
            [q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0*q0 - q1*q1 - q2*q2 + q3*q3],
        ]
    )


def inertia_diag(mass: jax.Array, cfg: Mapping[str, float]) -> jax.Array:
    radius = float(cfg["vehicle_radius_m"])
    length = float(cfg["vehicle_length_m"])
    ixx = 0.5 * mass * radius * radius
    iyy = mass * (length * length + 3.0 * radius * radius) / 12.0
    return jnp.array([ixx, iyy, iyy])


def tilt_from_q(q: jax.Array) -> jax.Array:
    return jnp.arccos(jnp.clip(quat_to_rot(q)[0, 0], -1.0, 1.0))


def _rate_limit(value: jax.Array, previous: jax.Array, max_delta: float | jax.Array) -> jax.Array:
    return previous + jnp.clip(value - previous, -max_delta, max_delta)


def _unit_vector_rate_limit(desired: jax.Array, previous: jax.Array, max_angle: jax.Array) -> jax.Array:
    desired = _unit(desired)
    previous = _unit(previous)
    angle = jnp.arccos(jnp.clip(jnp.dot(previous, desired), -1.0, 1.0))
    fraction = jnp.minimum(1.0, max_angle / (angle + 1e-7))
    return _unit((1.0 - fraction) * previous + fraction * desired)


def init_memory(state: jax.Array, cfg: Mapping[str, float]) -> ControllerMemory:
    body_x = quat_to_rot(state[6:10])[:, 0]
    hover = float(cfg["gravity"]) * state[13] / float(cfg["thrust_max_n"])
    return ControllerMemory(
        phase=jnp.asarray(0, dtype=jnp.int32),
        dwell=jnp.asarray(0, dtype=jnp.int32),
        p1_released=jnp.asarray(False),
        p1_hold_steps=jnp.asarray(0, dtype=jnp.int32),
        prev_gimbal_rad=jnp.zeros(2, dtype=state.dtype),
        prev_grid_rad=jnp.zeros(2, dtype=state.dtype),
        prev_throttle=jnp.clip(hover, 0.02, 0.98),
        prev_rcs_roll_nm=jnp.asarray(0.0, dtype=state.dtype),
        prev_thrust_accel=jnp.array([float(cfg["gravity"]), 0.0, 0.0], dtype=state.dtype),
        prev_body_x_ref=body_x,
        # The first rolling solve is free to choose any candidate up to 40 s;
        # later solves may shorten, but never silently extend, that commitment.
        prev_tgo_s=jnp.asarray(float(cfg["p0_tgo_max_s"]), dtype=state.dtype),
    )


def _update_phase(state: jax.Array, memory: ControllerMemory, cfg: Mapping[str, float]):
    h = state[0]
    r = _safe_norm(state[1:3])
    vlat = _safe_norm(state[4:6])
    tilt = tilt_from_q(state[6:10])
    omega = state[10:13]
    gate_p0 = (
        (h < float(cfg["terminal_entry_height_m"]))
        & (r < float(cfg["terminal_entry_radius_m"]))
        & (vlat < float(cfg["terminal_entry_vlat_mps"]))
        # A stricter capture-ready subset of the published upper gate keeps
        # P0 in its 275 m layer until the slow TVC loop has shed lateral energy.
        & (r < float(cfg["capture_release_radius_m"]))
        & (vlat < float(cfg["capture_release_vlat_mps"]))
        & (tilt < float(cfg["terminal_entry_tilt_rad"]))
        & (jnp.abs(omega[0]) < float(cfg["terminal_entry_roll_rate_radps"]))
    )
    gate_p1 = h < float(cfg["flare_entry_height_m"])
    # V2 has no pre-contact "gear sink" phase.  P2 remains a powered
    # velocity-corridor controller until first leg contact; the evaluator ends
    # scoring at that instant, before suspension damping can change velocity.
    gate = jnp.where(memory.phase == 0, gate_p0, jnp.where(memory.phase == 1, gate_p1, False))
    gate = gate & (memory.phase < 2)
    dwell = jnp.where(gate, memory.dwell + 1, 0)
    required_dwell = jnp.where(
        memory.phase == 0,
        int(cfg["p0_capture_dwell_steps"]),
        int(cfg["phase_dwell_steps"]),
    )
    transition = dwell >= required_dwell
    phase = jnp.where(transition, jnp.minimum(memory.phase + 1, 2), memory.phase)
    dwell = jnp.where(transition, 0, dwell)
    return phase.astype(jnp.int32), dwell.astype(jnp.int32)


def _rolling_zem_zev(
    state: jax.Array,
    previous_accel: jax.Array,
    previous_tgo: jax.Array,
    cfg: Mapping[str, float],
):
    pos = state[0:3]
    vel = state[3:6]
    mass = state[13]
    h = pos[0]
    radius = _safe_norm(pos[1:3])
    # Once below the capture-layer trigger, retain the zero-velocity 275 m
    # target until the state machine explicitly releases P0. This hysteresis
    # avoids alternating between the 300 m waypoint and the capture layer.
    capture = h < float(cfg["capture_trigger_height_m"])
    target_h = jnp.where(capture, float(cfg["capture_layer_height_m"]), float(cfg["normal_waypoint_height_m"]))
    target_vx = jnp.where(capture, 0.0, float(cfg["normal_waypoint_vx_mps"]))
    target_pos = jnp.array([target_h, 0.0, 0.0], dtype=state.dtype)
    target_vel = jnp.array([target_vx, 0.0, 0.0], dtype=state.dtype)
    t_min = jnp.where(capture, float(cfg["capture_tgo_min_s"]), float(cfg["p0_tgo_min_s"]))
    t_max = jnp.where(capture, float(cfg["capture_tgo_max_s"]), float(cfg["p0_tgo_max_s"]))
    u = jnp.linspace(0.0, 1.0, int(cfg["tgo_candidates"]), dtype=state.dtype)
    tgo = t_min + (t_max - t_min) * u
    gravity = jnp.array([-float(cfg["gravity"]), 0.0, 0.0], dtype=state.dtype)
    zem = target_pos[None, :] - (
        pos[None, :] + vel[None, :] * tgo[:, None] + 0.5 * gravity[None, :] * tgo[:, None] ** 2
    )
    zev = target_vel[None, :] - (vel[None, :] + gravity[None, :] * tgo[:, None])
    thrust_accel = 6.0 * zem / tgo[:, None] ** 2 - 2.0 * zev / tgo[:, None]

    accel_norm = jnp.sqrt(jnp.sum(thrust_accel * thrust_accel, axis=1) + 1e-8)
    lateral_norm = jnp.sqrt(jnp.sum(thrust_accel[:, 1:3] ** 2, axis=1) + 1e-8)
    tilt = jnp.arctan2(lateral_norm, jnp.maximum(thrust_accel[:, 0], 1e-4))
    max_accel = float(cfg["thrust_max_n"]) / mass
    margin_max = float(cfg["actuator_margin_fraction"]) * max_accel
    violation = (
        jnp.maximum(accel_norm / margin_max - 1.0, 0.0) ** 2
        + jnp.maximum(tilt / float(cfg["p0_tilt_limit_rad"]) - 1.0, 0.0) ** 2
        + jnp.maximum(0.35 - thrust_accel[:, 0] / float(cfg["gravity"]), 0.0) ** 2
    )
    jerk = jnp.sum(((thrust_accel - previous_accel[None, :]) / max_accel) ** 2, axis=1)
    continuity_target = jnp.clip(
        previous_tgo - float(cfg["dt"]) * int(cfg["guidance_period_steps"]),
        t_min,
        t_max,
    )
    tgo_continuity = (tgo - continuity_target) ** 2
    fuel_proxy = accel_norm * tgo
    score = (
        float(cfg["candidate_violation_weight"]) * violation
        + float(cfg["candidate_fuel_weight"]) * fuel_proxy
        + float(cfg["candidate_time_weight"]) * tgo
        + float(cfg["candidate_jerk_weight"]) * jerk
        + float(cfg["tgo_continuity_weight"]) * tgo_continuity
    )
    index = jnp.argmin(score)
    replanned_tgo = tgo[index]
    countdown_tgo = jnp.clip(previous_tgo - float(cfg["dt"]), t_min, t_max)
    selected_tgo = jnp.minimum(replanned_tgo, countdown_tgo)
    # Recompute at the monotonically decreasing committed horizon.  This keeps
    # rolling feasibility checks while preventing the receding-horizon hover
    # equilibrium that appears when tgo is repeatedly extended.
    zem_selected = target_pos - (
        pos + vel * selected_tgo + 0.5 * gravity * selected_tgo * selected_tgo
    )
    zev_selected = target_vel - (vel + gravity * selected_tgo)
    selected = 6.0 * zem_selected / (selected_tgo * selected_tgo) - 2.0 * zev_selected / selected_tgo
    # TVC attitude reference is deliberately rate-limited.  A short lateral
    # horizon would reverse the demanded tilt faster than that physical loop
    # can follow, producing a capture-layer limit cycle.  Keep the rolling
    # candidate for vertical braking, but impose the documented slow outer-loop
    # character on lateral ZEM/ZEV through a safeguarded horizon floor.
    lateral_floor = jnp.where(
        capture,
        float(cfg["capture_lateral_horizon_floor_s"]),
        float(cfg["p0_lateral_horizon_floor_s"]),
    )
    lateral_tgo = jnp.maximum(selected_tgo, lateral_floor)
    lateral = -6.0 * pos[1:3] / (lateral_tgo * lateral_tgo) - 4.0 * vel[1:3] / lateral_tgo
    selected = selected.at[1:3].set(lateral)
    return selected, selected_tgo


def _terminal_guidance(
    state: jax.Array,
    phase: jax.Array,
    p1_released: jax.Array,
    cfg: Mapping[str, float],
):
    h = state[0]
    vel = state[3:6]
    pos_lat = state[1:3]
    mass = state[13]
    g = float(cfg["gravity"])
    h_leg = float(cfg["terminal_h_leg_m"])
    a_flare = float(cfg["terminal_flare_accel_mps2"])
    v_touch = float(cfg["terminal_v_touch_mps"])
    raw_v_ref = -jnp.sqrt(v_touch * v_touch + 2.0 * a_flare * jnp.maximum(h - h_leg, 0.0))
    # The design contract caps P1 descent at -15 m/s; P2 follows the analytic
    # curve without this cap. The curve feed-forward is zero while the cap is
    # active because the capped reference is constant in height.
    p1_cap_active = (phase == 1) & (raw_v_ref < -15.0)
    v_ref = jnp.where(phase == 1, jnp.maximum(raw_v_ref, -15.0), raw_v_ref)
    v_ref = jnp.where(
        (phase == 1) & (~p1_released),
        float(cfg["p1_capture_hold_vx_mps"]),
        v_ref,
    )
    vertical_feedforward = jnp.where(p1_cap_active, 0.0, a_flare)
    net_vertical = vertical_feedforward + float(cfg["terminal_vertical_gain"]) * (v_ref - vel[0])
    thrust_x = g + net_vertical

    schedule = jnp.clip(
        (float(cfg["terminal_lateral_schedule_high_m"]) - h)
        / (float(cfg["terminal_lateral_schedule_high_m"]) - float(cfg["terminal_lateral_schedule_low_m"])),
        0.0,
        1.0,
    )
    wn = float(cfg["terminal_lateral_wn_high"]) + schedule * (
        float(cfg["terminal_lateral_wn_low"]) - float(cfg["terminal_lateral_wn_high"])
    )
    zeta = float(cfg["terminal_lateral_zeta"])
    capture_hold = (phase == 1) & (~p1_released)
    wn = jnp.where(capture_hold, float(cfg["p1_capture_hold_wn"]), wn)
    zeta = jnp.where(capture_hold, float(cfg["p1_capture_hold_zeta"]), zeta)
    lateral = -(wn * wn) * pos_lat - 2.0 * zeta * wn * vel[1:3]
    normal = jnp.concatenate([jnp.array([thrust_x], dtype=state.dtype), lateral])

    return normal, jnp.maximum(1.0, h / jnp.maximum(-v_ref, 0.5))


def _direct_ground_guidance(state: jax.Array, cfg: Mapping[str, float]):
    """Fuel-bounded touchdown branch of the analytic ZEM/ZEV family.

    It keeps the same equations as the rolling candidate solver but commits to
    a 30 s touchdown clock.  This prevents an otherwise feasible controller
    from repeatedly paying hover fuel at the intermediate capture waypoint.
    """
    pos = state[0:3]
    vel = state[3:6]
    elapsed = state[14] * float(cfg["dt"])
    tgo = jnp.clip(
        float(cfg["p0_direct_ground_tgo_s"]) - elapsed,
        3.0,
        float(cfg["p0_direct_ground_tgo_s"]),
    )
    gravity = jnp.array([-float(cfg["gravity"]), 0.0, 0.0], dtype=state.dtype)
    zem_x = -(pos[0] + vel[0] * tgo + 0.5 * gravity[0] * tgo * tgo)
    zev_x = -(vel[0] + gravity[0] * tgo)
    thrust_x = 6.0 * zem_x / (tgo * tgo) - 2.0 * zev_x / tgo
    lateral_tgo = jnp.maximum(tgo, float(cfg["p0_direct_lateral_tgo_floor_s"]))
    lateral = -6.0 * pos[1:3] / (lateral_tgo * lateral_tgo) - 4.0 * vel[1:3] / lateral_tgo

    h = pos[0]
    bridge_ramp = jnp.clip(1.0 - h / 80.0, 0.0, 1.0) ** 2
    thrust_x = thrust_x + jnp.where(h < 80.0, 0.3 * (-1.0 - vel[0]), 0.0)
    # Taper only the position term; retain velocity damping through touchdown.
    lat_pos = -6.0 * pos[1:3] / (lateral_tgo * lateral_tgo)
    lat_vel = -4.0 * vel[1:3] / lateral_tgo
    lateral = jnp.where(
        h < 80.0,
        (1.0 - bridge_ramp) * lat_pos + lat_vel,
        lateral,
    )
    return jnp.concatenate([jnp.array([thrust_x]), lateral]), tgo


def _cone_and_thrust_limit(thrust_accel: jax.Array, tilt_cap: jax.Array, mass: jax.Array, cfg):
    g = float(cfg["gravity"])
    max_accel = 0.98 * float(cfg["thrust_max_n"]) / mass
    vertical = jnp.clip(thrust_accel[0], 0.02 * g, max_accel)
    lateral = thrust_accel[1:3]
    lateral_norm = _safe_norm(lateral)
    lateral_cap = jnp.maximum(vertical, 0.1 * g) * jnp.tan(tilt_cap)
    lateral = lateral * jnp.minimum(1.0, lateral_cap / lateral_norm)
    out = jnp.concatenate([jnp.array([vertical], dtype=thrust_accel.dtype), lateral])
    out_norm = _safe_norm(out)
    return out * jnp.minimum(1.0, max_accel / out_norm)


def _rcs_radial_authority(radius: jax.Array, cfg: Mapping[str, float]) -> jax.Array:
    x = (radius - float(cfg["rcs_boost_radius_m"])) / float(cfg["rcs_schedule_width_m"])
    s = 0.5 * (1.0 + jnp.tanh(x))
    gain = float(cfg["rcs_near_scale"]) + (float(cfg["rcs_far_scale"]) - float(cfg["rcs_near_scale"])) * s
    return float(cfg["rcs_base_torque_nm"]) * gain


def control_step(
    state: jax.Array,
    memory: ControllerMemory,
    cfg: Mapping[str, float],
):
    """Compute one physical action, updated controller memory, and diagnostics."""
    phase, dwell = _update_phase(state, memory, cfg)
    h = state[0]
    vel = state[3:6]
    q = state[6:10]
    omega = state[10:13]
    mass = state[13]
    radius = _safe_norm(state[1:3])
    speed = _safe_norm(vel)
    g = float(cfg["gravity"])
    max_net_brake = float(cfg["emergency_brake_efficiency"]) * jnp.maximum(
        float(cfg["thrust_max_n"]) / mass - g, 0.1
    )
    down_speed = jnp.maximum(-vel[0], 0.0)
    stop_distance = down_speed * down_speed / (2.0 * max_net_brake)
    emergency_height = (
        float(cfg["emergency_distance_factor"]) * stop_distance
        + float(cfg["emergency_height_buffer_m"])
    )
    brake_margin = h - emergency_height
    emergency = (brake_margin < 0.0) & (h > float(cfg["terminal_h_leg_m"]))

    p1_ready_now = (
        (_safe_norm(state[1:3]) < float(cfg["p1_release_radius_m"]))
        & (_safe_norm(state[4:6]) < float(cfg["p1_release_vlat_mps"]))
    )
    p1_hold_steps = jnp.where(
        (phase == 1) & (~memory.p1_released),
        memory.p1_hold_steps + 1,
        memory.p1_hold_steps,
    )
    p1_timeout = p1_hold_steps >= int(cfg["p1_max_capture_hold_steps"])
    p1_released = memory.p1_released | (
        (phase >= 1) & (p1_ready_now | p1_timeout | (phase >= 2))
    )
    p0_accel, p0_tgo = _rolling_zem_zev(
        state, memory.prev_thrust_accel, memory.prev_tgo_s, cfg
    )
    direct_accel, direct_tgo = _direct_ground_guidance(state, cfg)
    use_direct = float(cfg["p0_direct_ground_on"]) > 0.5
    p0_accel = jnp.where(use_direct, direct_accel, p0_accel)
    p0_tgo = jnp.where(use_direct, direct_tgo, p0_tgo)
    terminal_accel, terminal_tgo = _terminal_guidance(state, phase, p1_released, cfg)
    requested = jnp.where(phase == 0, p0_accel, terminal_accel)
    tgo = jnp.where(phase == 0, p0_tgo, terminal_tgo)
    # Emergency mode reserves the thrust cone for vertical braking, but remains
    # a velocity-corridor controller instead of holding full thrust until the
    # stopping-distance flag clears.  That prevents a late trigger from
    # launching the vehicle back upward and wasting the remaining propellant.
    emergency_v_ref = -jnp.sqrt(
        float(cfg["terminal_v_touch_mps"]) ** 2
        + 2.0 * float(cfg["terminal_flare_accel_mps2"])
        * jnp.maximum(h - float(cfg["terminal_h_leg_m"]), 0.0)
    )
    emergency_vertical = (
        g
        + float(cfg["terminal_flare_accel_mps2"])
        + 2.5 * (emergency_v_ref - vel[0])
    )
    emergency_vertical = jnp.clip(
        emergency_vertical,
        0.02 * g,
        0.98 * float(cfg["thrust_max_n"]) / mass,
    )
    emergency_accel = jnp.concatenate(
        [
            jnp.array([emergency_vertical], dtype=state.dtype),
            requested[1:3],
        ]
    )
    requested = jnp.where(emergency, emergency_accel, requested)

    p2_schedule = jnp.clip(
        (h - float(cfg["terminal_h_leg_m"]))
        / (float(cfg["flare_entry_height_m"]) - float(cfg["terminal_h_leg_m"])),
        0.0,
        1.0,
    )
    p2_cap = float(cfg["p2_tilt_low_rad"]) + p2_schedule * (
        float(cfg["p2_tilt_high_rad"]) - float(cfg["p2_tilt_low_rad"])
    )
    tilt_cap = jnp.where(
        phase == 0,
        float(cfg["p0_tilt_limit_rad"]),
        jnp.where(phase == 1, float(cfg["p1_tilt_limit_rad"]), p2_cap),
    )
    tilt_cap = jnp.where(emergency, jnp.minimum(tilt_cap, jnp.deg2rad(5.0)), tilt_cap)
    requested = _cone_and_thrust_limit(requested, tilt_cap, mass, cfg)

    update_guidance = (state[14].astype(jnp.int32) % int(cfg["guidance_period_steps"])) == 0
    accel_delta = float(cfg["guidance_accel_slew_mps3"]) * float(cfg["dt"]) * int(cfg["guidance_period_steps"])
    limited_accel = _rate_limit(requested, memory.prev_thrust_accel, accel_delta)
    thrust_accel = jnp.where(update_guidance | emergency, limited_accel, memory.prev_thrust_accel)
    thrust_accel = _cone_and_thrust_limit(thrust_accel, tilt_cap, mass, cfg)

    throttle_request = _safe_norm(thrust_accel) * mass / float(cfg["thrust_max_n"])
    throttle_request = jnp.clip(
        throttle_request,
        float(cfg["throttle_normal_min"]),
        float(cfg["throttle_normal_max"]),
    )
    throttle = _rate_limit(
        throttle_request,
        memory.prev_throttle,
        float(cfg["throttle_rate_per_step"]),
    )

    desired_body_x = _unit(thrust_accel)
    ref_rate = jnp.where(
        phase == 0,
        float(cfg["attitude_rate_p0_radps"]),
        jnp.where(phase == 1, float(cfg["attitude_rate_p1_radps"]), float(cfg["attitude_rate_p2_radps"])),
    )
    body_x_ref = _unit_vector_rate_limit(
        desired_body_x,
        memory.prev_body_x_ref,
        ref_rate * float(cfg["dt"]),
    )
    rot = quat_to_rot(q)
    current_body_x = rot[:, 0]
    error_i = jnp.cross(current_body_x, body_x_ref)
    error_b = rot.T @ error_i
    body_axis_error = jnp.arcsin(jnp.clip(_safe_norm(error_i), 0.0, 1.0))

    wn_py = jnp.where(
        phase == 0,
        float(cfg["attitude_wn_py_p0"]),
        jnp.where(phase == 1, float(cfg["attitude_wn_py_p1"]), float(cfg["attitude_wn_py_p2"])),
    )
    zeta_py = jnp.where(
        phase == 0,
        float(cfg["attitude_zeta_py_p0"]),
        jnp.where(phase == 1, float(cfg["attitude_zeta_py_p1"]), float(cfg["attitude_zeta_py_p2"])),
    )
    inertia = inertia_diag(mass, cfg)
    torque_yz = inertia[1:3] * (wn_py * wn_py * error_b[1:3] - 2.0 * zeta_py * wn_py * omega[1:3])

    # Grid fins receive a bounded q-dependent share; gimbal closes the torque
    # residual. At the selected 0.5 m station this naturally leaves TVC primary.
    q_dyn = 0.5 * float(cfg["body_rho_kgpm3"]) * speed * speed
    grid_blend = jnp.clip(
        (q_dyn - float(cfg["grid_blend_q_off_pa"]))
        / (float(cfg["grid_blend_q_full_pa"]) - float(cfg["grid_blend_q_off_pa"])),
        0.0,
        1.0,
    )
    grid_torque_target = float(cfg["grid_torque_fraction"]) * grid_blend * torque_yz
    grid_gain = (
        float(cfg["grid_station_m"])
        * q_dyn
        * float(cfg["grid_area_total_m2"])
        * float(cfg["grid_control_cl"])
    )
    # tau_y=-gain*delta_z, tau_z=+gain*delta_y
    grid_request = jnp.array(
        [grid_torque_target[1] / (grid_gain + 1e-6), -grid_torque_target[0] / (grid_gain + 1e-6)],
        dtype=state.dtype,
    )
    grid_limit = jnp.where(emergency, float(cfg["grid_absolute_rad"]), float(cfg["grid_normal_rad"]))
    grid_request = jnp.clip(grid_request, -grid_limit, grid_limit)
    grid_rad = _rate_limit(grid_request, memory.prev_grid_rad, float(cfg["grid_rate_rad_per_step"]))
    grid_torque_realized = jnp.array(
        [-grid_gain * grid_rad[1], grid_gain * grid_rad[0]], dtype=state.dtype
    )

    torque_residual = torque_yz - grid_torque_realized
    engine_thrust = jnp.maximum(throttle * float(cfg["thrust_max_n"]), 1.0)
    arm = float(cfg["engine_arm_m"])
    # tau_y=+arm*T*gimbal_z, tau_z=-arm*T*gimbal_y
    gimbal_request = jnp.array(
        [-torque_residual[1] / (arm * engine_thrust), torque_residual[0] / (arm * engine_thrust)],
        dtype=state.dtype,
    )
    gimbal_limit = jnp.where(emergency, float(cfg["gimbal_absolute_rad"]), float(cfg["gimbal_normal_rad"]))
    gimbal_request = jnp.clip(gimbal_request, -gimbal_limit, gimbal_limit)
    gimbal_rad = _rate_limit(
        gimbal_request,
        memory.prev_gimbal_rad,
        float(cfg["gimbal_rate_rad_per_step"]),
    )

    wn_roll = jnp.where(phase <= 1, float(cfg["attitude_wn_roll_high"]), float(cfg["attitude_wn_roll_low"]))
    roll_torque_request = -2.0 * float(cfg["attitude_zeta_roll"]) * wn_roll * inertia[0] * omega[0]
    roll_cap = jnp.where(
        emergency,
        float(cfg["rcs_roll_emergency_cap_nm"]),
        float(cfg["rcs_roll_normal_cap_nm"]),
    )
    roll_torque_request = jnp.clip(roll_torque_request, -roll_cap, roll_cap)
    rcs_roll_nm = _rate_limit(
        roll_torque_request,
        memory.prev_rcs_roll_nm,
        float(cfg["rcs_roll_rate_nm_per_step"]),
    )
    rcs_roll_norm = rcs_roll_nm / _rcs_radial_authority(radius, cfg)

    zero = jnp.asarray(0.0, dtype=state.dtype)
    action = jnp.stack(
        [
            gimbal_rad[0] / float(cfg["gimbal_absolute_rad"]),
            gimbal_rad[1] / float(cfg["gimbal_absolute_rad"]),
            2.0 * throttle - 1.0,
            rcs_roll_norm,
            zero,  # RCS pitch: hard lock
            zero,  # RCS yaw: hard lock
            grid_rad[0] / float(cfg["grid_absolute_rad"]),
            grid_rad[1] / float(cfg["grid_absolute_rad"]),
            zero,  # grid-fin roll: hard lock
        ]
    )
    action = jnp.clip(action, -1.0, 1.0)

    new_memory = ControllerMemory(
        phase=phase,
        dwell=dwell,
        p1_released=p1_released,
        p1_hold_steps=p1_hold_steps,
        prev_gimbal_rad=gimbal_rad,
        prev_grid_rad=grid_rad,
        prev_throttle=throttle,
        prev_rcs_roll_nm=rcs_roll_nm,
        prev_thrust_accel=thrust_accel,
        prev_body_x_ref=body_x_ref,
        prev_tgo_s=tgo,
    )
    torque_request = jnp.concatenate([jnp.array([roll_torque_request]), torque_yz])
    diagnostics = jnp.concatenate(
        [
            jnp.array([phase, emergency, tgo, brake_margin, q_dyn], dtype=state.dtype),
            thrust_accel,
            torque_request,
            gimbal_rad,
            grid_rad,
            jnp.array([throttle, rcs_roll_nm, body_axis_error], dtype=state.dtype),
        ]
    )
    return action, new_memory, diagnostics
