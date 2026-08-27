use crate::{
    config::ControllerConfig,
    math::{
        Vec2, Vec3, clip, controller_quat_to_rot, cross, dot, mat_t_vec, safe_norm, safe_norm2,
        scale, unit,
    },
    plant::State,
};

pub const DIAGNOSTIC_COLUMNS: [&str; 18] = [
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
];

#[derive(Clone, Copy, Debug)]
pub struct ControllerMemory {
    pub phase: i32,
    pub dwell: i32,
    pub p1_released: bool,
    pub p1_hold_steps: i32,
    pub prev_gimbal_rad: Vec2,
    pub prev_grid_rad: Vec2,
    pub prev_throttle: f32,
    pub prev_rcs_roll_nm: f32,
    pub prev_thrust_accel: Vec3,
    pub prev_body_x_ref: Vec3,
    pub prev_tgo_s: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct ControlOutput {
    pub action: [f32; 9],
    pub memory: ControllerMemory,
    pub diagnostic: [f32; 18],
}

#[inline(always)]
fn state_vec3(state: &State, start: usize) -> Vec3 {
    [state[start], state[start + 1], state[start + 2]]
}

#[inline(always)]
fn state_quat(state: &State) -> [f32; 4] {
    [state[6], state[7], state[8], state[9]]
}

#[inline(always)]
fn inertia_diag(mass: f32, cfg: &ControllerConfig) -> Vec3 {
    let ixx = 0.5 * mass * cfg.vehicle_radius_m * cfg.vehicle_radius_m;
    let iyy = mass
        * (cfg.vehicle_length_m * cfg.vehicle_length_m
            + 3.0 * cfg.vehicle_radius_m * cfg.vehicle_radius_m)
        / 12.0;
    [ixx, iyy, iyy]
}

#[inline(always)]
fn tilt_from_q(q: [f32; 4]) -> f32 {
    clip(controller_quat_to_rot(q)[0][0], -1.0, 1.0).acos()
}

#[inline(always)]
fn rate_limit(value: f32, previous: f32, max_delta: f32) -> f32 {
    previous + clip(value - previous, -max_delta, max_delta)
}

#[inline(always)]
fn rate_limit2(value: Vec2, previous: Vec2, max_delta: f32) -> Vec2 {
    [
        rate_limit(value[0], previous[0], max_delta),
        rate_limit(value[1], previous[1], max_delta),
    ]
}

#[inline(always)]
fn rate_limit3(value: Vec3, previous: Vec3, max_delta: f32) -> Vec3 {
    [
        rate_limit(value[0], previous[0], max_delta),
        rate_limit(value[1], previous[1], max_delta),
        rate_limit(value[2], previous[2], max_delta),
    ]
}

#[inline(always)]
fn unit_vector_rate_limit(desired: Vec3, previous: Vec3, max_angle: f32) -> Vec3 {
    let desired = unit(desired);
    let previous = unit(previous);
    let angle = clip(dot(previous, desired), -1.0, 1.0).acos();
    let fraction = (max_angle / (angle + 1.0e-7)).min(1.0);
    unit([
        (1.0 - fraction) * previous[0] + fraction * desired[0],
        (1.0 - fraction) * previous[1] + fraction * desired[1],
        (1.0 - fraction) * previous[2] + fraction * desired[2],
    ])
}

pub fn init_memory(state: &State, cfg: &ControllerConfig) -> ControllerMemory {
    let rot = controller_quat_to_rot(state_quat(state));
    let body_x = [rot[0][0], rot[1][0], rot[2][0]];
    let hover = cfg.gravity * state[13] / cfg.thrust_max_n;
    ControllerMemory {
        phase: 0,
        dwell: 0,
        p1_released: false,
        p1_hold_steps: 0,
        prev_gimbal_rad: [0.0; 2],
        prev_grid_rad: [0.0; 2],
        prev_throttle: clip(hover, 0.02, 0.98),
        prev_rcs_roll_nm: 0.0,
        prev_thrust_accel: [cfg.gravity, 0.0, 0.0],
        prev_body_x_ref: body_x,
        prev_tgo_s: cfg.p0_tgo_max_s,
    }
}

#[inline]
fn update_phase(state: &State, memory: &ControllerMemory, cfg: &ControllerConfig) -> (i32, i32) {
    let h = state[0];
    let r = safe_norm2([state[1], state[2]]);
    let vlat = safe_norm2([state[4], state[5]]);
    let tilt = tilt_from_q(state_quat(state));
    let gate_p0 = h < cfg.terminal_entry_height_m
        && r < cfg.terminal_entry_radius_m
        && vlat < cfg.terminal_entry_vlat_mps
        && r < cfg.capture_release_radius_m
        && vlat < cfg.capture_release_vlat_mps
        && tilt < cfg.terminal_entry_tilt_rad
        && state[10].abs() < cfg.terminal_entry_roll_rate_radps;
    let gate_p1 = h < cfg.flare_entry_height_m;
    let mut gate = match memory.phase {
        0 => gate_p0,
        1 => gate_p1,
        _ => false,
    };
    gate &= memory.phase < 2;
    let mut dwell = if gate { memory.dwell + 1 } else { 0 };
    let required = if memory.phase == 0 {
        cfg.p0_capture_dwell_steps
    } else {
        cfg.phase_dwell_steps
    };
    let transition = dwell >= required;
    let phase = if transition {
        (memory.phase + 1).min(2)
    } else {
        memory.phase
    };
    if transition {
        dwell = 0;
    }
    (phase, dwell)
}

fn rolling_zem_zev(
    state: &State,
    previous_accel: Vec3,
    previous_tgo: f32,
    cfg: &ControllerConfig,
) -> (Vec3, f32) {
    let pos = state_vec3(state, 0);
    let vel = state_vec3(state, 3);
    let mass = state[13];
    let capture = pos[0] < cfg.capture_trigger_height_m;
    let target_h = if capture {
        cfg.capture_layer_height_m
    } else {
        cfg.normal_waypoint_height_m
    };
    let target_vx = if capture {
        0.0
    } else {
        cfg.normal_waypoint_vx_mps
    };
    let target_pos = [target_h, 0.0, 0.0];
    let target_vel = [target_vx, 0.0, 0.0];
    let t_min = if capture {
        cfg.capture_tgo_min_s
    } else {
        cfg.p0_tgo_min_s
    };
    let t_max = if capture {
        cfg.capture_tgo_max_s
    } else {
        cfg.p0_tgo_max_s
    };
    let gravity = [-cfg.gravity, 0.0, 0.0];
    let max_accel = cfg.thrust_max_n / mass;
    let margin_max = cfg.actuator_margin_fraction * max_accel;
    let continuity_target = clip(
        previous_tgo - cfg.dt * cfg.guidance_period_steps as f32,
        t_min,
        t_max,
    );
    let mut best_score = f32::INFINITY;
    let mut replanned_tgo = t_min;

    for index in 0..cfg.tgo_candidates {
        let u = index as f32 / (cfg.tgo_candidates - 1) as f32;
        let tgo = t_min + (t_max - t_min) * u;
        let tgo2 = tgo * tgo;
        let mut thrust_accel = [0.0; 3];
        for k in 0..3 {
            let zem = target_pos[k] - (pos[k] + vel[k] * tgo + 0.5 * gravity[k] * tgo2);
            let zev = target_vel[k] - (vel[k] + gravity[k] * tgo);
            thrust_accel[k] = 6.0 * zem / tgo2 - 2.0 * zev / tgo;
        }
        let accel_norm = safe_norm(thrust_accel);
        let lateral_norm = safe_norm2([thrust_accel[1], thrust_accel[2]]);
        let tilt = lateral_norm.atan2(thrust_accel[0].max(1.0e-4));
        let violation = (accel_norm / margin_max - 1.0).max(0.0).powi(2)
            + (tilt / cfg.p0_tilt_limit_rad - 1.0).max(0.0).powi(2)
            + (0.35 - thrust_accel[0] / cfg.gravity).max(0.0).powi(2);
        let jerk = ((thrust_accel[0] - previous_accel[0]) / max_accel).powi(2)
            + ((thrust_accel[1] - previous_accel[1]) / max_accel).powi(2)
            + ((thrust_accel[2] - previous_accel[2]) / max_accel).powi(2);
        let tgo_continuity = (tgo - continuity_target).powi(2);
        let fuel_proxy = accel_norm * tgo;
        let score = cfg.candidate_violation_weight * violation
            + cfg.candidate_fuel_weight * fuel_proxy
            + cfg.candidate_time_weight * tgo
            + cfg.candidate_jerk_weight * jerk
            + cfg.tgo_continuity_weight * tgo_continuity;
        if score < best_score {
            best_score = score;
            replanned_tgo = tgo;
        }
    }

    let countdown_tgo = clip(previous_tgo - cfg.dt, t_min, t_max);
    let selected_tgo = replanned_tgo.min(countdown_tgo);
    let tgo2 = selected_tgo * selected_tgo;
    let mut selected = [0.0; 3];
    for k in 0..3 {
        let zem = target_pos[k] - (pos[k] + vel[k] * selected_tgo + 0.5 * gravity[k] * tgo2);
        let zev = target_vel[k] - (vel[k] + gravity[k] * selected_tgo);
        selected[k] = 6.0 * zem / tgo2 - 2.0 * zev / selected_tgo;
    }
    let lateral_floor = if capture {
        cfg.capture_lateral_horizon_floor_s
    } else {
        cfg.p0_lateral_horizon_floor_s
    };
    let lateral_tgo = selected_tgo.max(lateral_floor);
    selected[1] = -6.0 * pos[1] / (lateral_tgo * lateral_tgo) - 4.0 * vel[1] / lateral_tgo;
    selected[2] = -6.0 * pos[2] / (lateral_tgo * lateral_tgo) - 4.0 * vel[2] / lateral_tgo;
    (selected, selected_tgo)
}

fn terminal_guidance(
    state: &State,
    phase: i32,
    p1_released: bool,
    cfg: &ControllerConfig,
) -> (Vec3, f32) {
    let h = state[0];
    let raw_v_ref = -(cfg.terminal_v_touch_mps * cfg.terminal_v_touch_mps
        + 2.0 * cfg.terminal_flare_accel_mps2 * (h - cfg.terminal_h_leg_m).max(0.0))
    .sqrt();
    let p1_cap_active = phase == 1 && raw_v_ref < -15.0;
    let mut v_ref = if phase == 1 {
        raw_v_ref.max(-15.0)
    } else {
        raw_v_ref
    };
    if phase == 1 && !p1_released {
        v_ref = cfg.p1_capture_hold_vx_mps;
    }
    let vertical_feedforward = if p1_cap_active {
        0.0
    } else {
        cfg.terminal_flare_accel_mps2
    };
    let net_vertical = vertical_feedforward + cfg.terminal_vertical_gain * (v_ref - state[3]);
    let thrust_x = cfg.gravity + net_vertical;
    let schedule = clip(
        (cfg.terminal_lateral_schedule_high_m - h)
            / (cfg.terminal_lateral_schedule_high_m - cfg.terminal_lateral_schedule_low_m),
        0.0,
        1.0,
    );
    let mut wn = cfg.terminal_lateral_wn_high
        + schedule * (cfg.terminal_lateral_wn_low - cfg.terminal_lateral_wn_high);
    let mut zeta = cfg.terminal_lateral_zeta;
    if phase == 1 && !p1_released {
        wn = cfg.p1_capture_hold_wn;
        zeta = cfg.p1_capture_hold_zeta;
    }
    let lateral = [
        -(wn * wn) * state[1] - 2.0 * zeta * wn * state[4],
        -(wn * wn) * state[2] - 2.0 * zeta * wn * state[5],
    ];
    (
        [thrust_x, lateral[0], lateral[1]],
        (h / (-v_ref).max(0.5)).max(1.0),
    )
}

fn direct_ground_guidance(state: &State, cfg: &ControllerConfig) -> (Vec3, f32) {
    let elapsed = state[14] * cfg.dt;
    let tgo = clip(
        cfg.p0_direct_ground_tgo_s - elapsed,
        3.0,
        cfg.p0_direct_ground_tgo_s,
    );
    let gravity_x = -cfg.gravity;
    let zem_x = -(state[0] + state[3] * tgo + 0.5 * gravity_x * tgo * tgo);
    let zev_x = -(state[3] + gravity_x * tgo);
    let mut thrust_x = 6.0 * zem_x / (tgo * tgo) - 2.0 * zev_x / tgo;
    let lateral_tgo = tgo.max(cfg.p0_direct_lateral_tgo_floor_s);
    let bridge_ramp = clip(1.0 - state[0] / 80.0, 0.0, 1.0).powi(2);
    if state[0] < 80.0 {
        thrust_x += 0.3 * (-1.0 - state[3]);
    }
    let mut lateral = [0.0; 2];
    for k in 0..2 {
        let pos = state[1 + k];
        let vel = state[4 + k];
        let lat_pos = -6.0 * pos / (lateral_tgo * lateral_tgo);
        let lat_vel = -4.0 * vel / lateral_tgo;
        lateral[k] = if state[0] < 80.0 {
            (1.0 - bridge_ramp) * lat_pos + lat_vel
        } else {
            lat_pos + lat_vel
        };
    }
    ([thrust_x, lateral[0], lateral[1]], tgo)
}

#[inline]
fn cone_and_thrust_limit(
    thrust_accel: Vec3,
    tilt_cap: f32,
    mass: f32,
    cfg: &ControllerConfig,
) -> Vec3 {
    let max_accel = 0.98 * cfg.thrust_max_n / mass;
    let vertical = clip(thrust_accel[0], 0.02 * cfg.gravity, max_accel);
    let lateral_norm = safe_norm2([thrust_accel[1], thrust_accel[2]]);
    let lateral_cap = vertical.max(0.1 * cfg.gravity) * tilt_cap.tan();
    let lateral_scale = (lateral_cap / lateral_norm).min(1.0);
    let mut out = [
        vertical,
        thrust_accel[1] * lateral_scale,
        thrust_accel[2] * lateral_scale,
    ];
    let out_scale = (max_accel / safe_norm(out)).min(1.0);
    out = scale(out, out_scale);
    out
}

#[inline(always)]
fn rcs_radial_authority(radius: f32, cfg: &ControllerConfig) -> f32 {
    let x = (radius - cfg.rcs_boost_radius_m) / cfg.rcs_schedule_width_m;
    let s = 0.5 * (1.0 + x.tanh());
    cfg.rcs_base_torque_nm * (cfg.rcs_near_scale + (cfg.rcs_far_scale - cfg.rcs_near_scale) * s)
}

pub fn control_step(
    state: &State,
    memory: &ControllerMemory,
    cfg: &ControllerConfig,
) -> ControlOutput {
    let (phase, dwell) = update_phase(state, memory, cfg);
    let h = state[0];
    let vel = state_vec3(state, 3);
    let omega = state_vec3(state, 10);
    let mass = state[13];
    let radius = safe_norm2([state[1], state[2]]);
    let speed = safe_norm(vel);
    let max_net_brake =
        cfg.emergency_brake_efficiency * (cfg.thrust_max_n / mass - cfg.gravity).max(0.1);
    let down_speed = (-vel[0]).max(0.0);
    let stop_distance = down_speed * down_speed / (2.0 * max_net_brake);
    let emergency_height =
        cfg.emergency_distance_factor * stop_distance + cfg.emergency_height_buffer_m;
    let brake_margin = h - emergency_height;
    let emergency = brake_margin < 0.0 && h > cfg.terminal_h_leg_m;

    let p1_ready_now = safe_norm2([state[1], state[2]]) < cfg.p1_release_radius_m
        && safe_norm2([state[4], state[5]]) < cfg.p1_release_vlat_mps;
    let p1_hold_steps = if phase == 1 && !memory.p1_released {
        memory.p1_hold_steps + 1
    } else {
        memory.p1_hold_steps
    };
    let p1_timeout = p1_hold_steps >= cfg.p1_max_capture_hold_steps;
    let p1_released =
        memory.p1_released || (phase >= 1 && (p1_ready_now || p1_timeout || phase >= 2));

    let (mut p0_accel, mut p0_tgo) =
        rolling_zem_zev(state, memory.prev_thrust_accel, memory.prev_tgo_s, cfg);
    let (direct_accel, direct_tgo) = direct_ground_guidance(state, cfg);
    if cfg.p0_direct_ground_on > 0.5 {
        p0_accel = direct_accel;
        p0_tgo = direct_tgo;
    }
    let (terminal_accel, terminal_tgo) = terminal_guidance(state, phase, p1_released, cfg);
    let (mut requested, tgo) = if phase == 0 {
        (p0_accel, p0_tgo)
    } else {
        (terminal_accel, terminal_tgo)
    };

    let emergency_v_ref = -(cfg.terminal_v_touch_mps * cfg.terminal_v_touch_mps
        + 2.0 * cfg.terminal_flare_accel_mps2 * (h - cfg.terminal_h_leg_m).max(0.0))
    .sqrt();
    let emergency_vertical = clip(
        cfg.gravity + cfg.terminal_flare_accel_mps2 + 2.5 * (emergency_v_ref - vel[0]),
        0.02 * cfg.gravity,
        0.98 * cfg.thrust_max_n / mass,
    );
    if emergency {
        requested[0] = emergency_vertical;
    }

    let p2_schedule = clip(
        (h - cfg.terminal_h_leg_m) / (cfg.flare_entry_height_m - cfg.terminal_h_leg_m),
        0.0,
        1.0,
    );
    let p2_cap = cfg.p2_tilt_low_rad + p2_schedule * (cfg.p2_tilt_high_rad - cfg.p2_tilt_low_rad);
    let mut tilt_cap = match phase {
        0 => cfg.p0_tilt_limit_rad,
        1 => cfg.p1_tilt_limit_rad,
        _ => p2_cap,
    };
    if emergency {
        tilt_cap = tilt_cap.min(5.0_f32.to_radians());
    }
    requested = cone_and_thrust_limit(requested, tilt_cap, mass, cfg);

    let update_guidance = (state[14] as i32 % cfg.guidance_period_steps) == 0;
    let accel_delta = cfg.guidance_accel_slew_mps3 * cfg.dt * cfg.guidance_period_steps as f32;
    let limited_accel = rate_limit3(requested, memory.prev_thrust_accel, accel_delta);
    let mut thrust_accel = if update_guidance || emergency {
        limited_accel
    } else {
        memory.prev_thrust_accel
    };
    thrust_accel = cone_and_thrust_limit(thrust_accel, tilt_cap, mass, cfg);

    let throttle_request = clip(
        safe_norm(thrust_accel) * mass / cfg.thrust_max_n,
        cfg.throttle_normal_min,
        cfg.throttle_normal_max,
    );
    let throttle = rate_limit(
        throttle_request,
        memory.prev_throttle,
        cfg.throttle_rate_per_step,
    );

    let desired_body_x = unit(thrust_accel);
    let ref_rate = match phase {
        0 => cfg.attitude_rate_p0_radps,
        1 => cfg.attitude_rate_p1_radps,
        _ => cfg.attitude_rate_p2_radps,
    };
    let body_x_ref =
        unit_vector_rate_limit(desired_body_x, memory.prev_body_x_ref, ref_rate * cfg.dt);
    let rot = controller_quat_to_rot(state_quat(state));
    let current_body_x = [rot[0][0], rot[1][0], rot[2][0]];
    let error_i = cross(current_body_x, body_x_ref);
    let error_b = mat_t_vec(rot, error_i);
    let body_axis_error = clip(safe_norm(error_i), 0.0, 1.0).asin();

    let wn_py = match phase {
        0 => cfg.attitude_wn_py_p0,
        1 => cfg.attitude_wn_py_p1,
        _ => cfg.attitude_wn_py_p2,
    };
    let zeta_py = match phase {
        0 => cfg.attitude_zeta_py_p0,
        1 => cfg.attitude_zeta_py_p1,
        _ => cfg.attitude_zeta_py_p2,
    };
    let inertia = inertia_diag(mass, cfg);
    let torque_yz = [
        inertia[1] * (wn_py * wn_py * error_b[1] - 2.0 * zeta_py * wn_py * omega[1]),
        inertia[2] * (wn_py * wn_py * error_b[2] - 2.0 * zeta_py * wn_py * omega[2]),
    ];

    let q_dyn = 0.5 * cfg.body_rho_kgpm3 * speed * speed;
    let grid_blend = clip(
        (q_dyn - cfg.grid_blend_q_off_pa) / (cfg.grid_blend_q_full_pa - cfg.grid_blend_q_off_pa),
        0.0,
        1.0,
    );
    let grid_torque_target = [
        cfg.grid_torque_fraction * grid_blend * torque_yz[0],
        cfg.grid_torque_fraction * grid_blend * torque_yz[1],
    ];
    let grid_gain = cfg.grid_station_m * q_dyn * cfg.grid_area_total_m2 * cfg.grid_control_cl;
    let mut grid_request = [
        grid_torque_target[1] / (grid_gain + 1.0e-6),
        -grid_torque_target[0] / (grid_gain + 1.0e-6),
    ];
    let grid_limit = if emergency {
        cfg.grid_absolute_rad
    } else {
        cfg.grid_normal_rad
    };
    grid_request[0] = clip(grid_request[0], -grid_limit, grid_limit);
    grid_request[1] = clip(grid_request[1], -grid_limit, grid_limit);
    let grid_rad = rate_limit2(
        grid_request,
        memory.prev_grid_rad,
        cfg.grid_rate_rad_per_step,
    );
    let grid_torque_realized = [-grid_gain * grid_rad[1], grid_gain * grid_rad[0]];

    let torque_residual = [
        torque_yz[0] - grid_torque_realized[0],
        torque_yz[1] - grid_torque_realized[1],
    ];
    let engine_thrust = (throttle * cfg.thrust_max_n).max(1.0);
    let mut gimbal_request = [
        -torque_residual[1] / (cfg.engine_arm_m * engine_thrust),
        torque_residual[0] / (cfg.engine_arm_m * engine_thrust),
    ];
    let gimbal_limit = if emergency {
        cfg.gimbal_absolute_rad
    } else {
        cfg.gimbal_normal_rad
    };
    gimbal_request[0] = clip(gimbal_request[0], -gimbal_limit, gimbal_limit);
    gimbal_request[1] = clip(gimbal_request[1], -gimbal_limit, gimbal_limit);
    let gimbal_rad = rate_limit2(
        gimbal_request,
        memory.prev_gimbal_rad,
        cfg.gimbal_rate_rad_per_step,
    );

    let wn_roll = if phase <= 1 {
        cfg.attitude_wn_roll_high
    } else {
        cfg.attitude_wn_roll_low
    };
    let mut roll_torque_request = -2.0 * cfg.attitude_zeta_roll * wn_roll * inertia[0] * omega[0];
    let roll_cap = if emergency {
        cfg.rcs_roll_emergency_cap_nm
    } else {
        cfg.rcs_roll_normal_cap_nm
    };
    roll_torque_request = clip(roll_torque_request, -roll_cap, roll_cap);
    let rcs_roll_nm = rate_limit(
        roll_torque_request,
        memory.prev_rcs_roll_nm,
        cfg.rcs_roll_rate_nm_per_step,
    );
    let rcs_roll_norm = rcs_roll_nm / rcs_radial_authority(radius, cfg);

    let mut action = [
        gimbal_rad[0] / cfg.gimbal_absolute_rad,
        gimbal_rad[1] / cfg.gimbal_absolute_rad,
        2.0 * throttle - 1.0,
        rcs_roll_norm,
        0.0,
        0.0,
        grid_rad[0] / cfg.grid_absolute_rad,
        grid_rad[1] / cfg.grid_absolute_rad,
        0.0,
    ];
    for value in &mut action {
        *value = clip(*value, -1.0, 1.0);
    }
    // Defense in depth: preserve +0.0 even if surrounding code changes.
    action[4] = 0.0;
    action[5] = 0.0;
    action[8] = 0.0;

    let memory = ControllerMemory {
        phase,
        dwell,
        p1_released,
        p1_hold_steps,
        prev_gimbal_rad: gimbal_rad,
        prev_grid_rad: grid_rad,
        prev_throttle: throttle,
        prev_rcs_roll_nm: rcs_roll_nm,
        prev_thrust_accel: thrust_accel,
        prev_body_x_ref: body_x_ref,
        prev_tgo_s: tgo,
    };
    let diagnostic = [
        phase as f32,
        if emergency { 1.0 } else { 0.0 },
        tgo,
        brake_margin,
        q_dyn,
        thrust_accel[0],
        thrust_accel[1],
        thrust_accel[2],
        roll_torque_request,
        torque_yz[0],
        torque_yz[1],
        gimbal_rad[0],
        gimbal_rad[1],
        grid_rad[0],
        grid_rad[1],
        throttle,
        rcs_roll_nm,
        body_axis_error,
    ];
    ControlOutput {
        action,
        memory,
        diagnostic,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn locked_channels_are_positive_zero() {
        let cfg = ControllerConfig::load(Path::new("config/controller.json")).unwrap();
        let mut state = [0.0; 16];
        state[0] = 2_000.0;
        state[3] = -75.0;
        state[6] = 1.0;
        state[13] = 29_200.0;
        state[15] = 500.0;
        let memory = init_memory(&state, &cfg);
        let output = control_step(&state, &memory, &cfg);
        for index in [4, 5, 8] {
            assert_eq!(output.action[index].to_bits(), 0.0_f32.to_bits());
        }
    }
}
