use std::f32::consts::PI;

use crate::{
    config::ControllerConfig,
    math::{
        Mat3, Vec3, add, all_finite, clip, cross, gimbal_rot, integrate_quat_exact, mat_t_vec,
        mat_vec, mul, norm, quat_to_rot, scale,
    },
};

pub type State = [f32; 16];
pub type Action = [f32; 9];

#[derive(Clone, Copy, Debug)]
pub struct PlantConfig {
    pub dt: f32,
    pub max_steps: i32,
    pub g: f32,
    pub mass_full: f32,
    pub mass_empty: f32,
    pub thrust_max: f32,
    pub isp: f32,
    pub length: f32,
    pub radius: f32,
    pub lever_x: f32,
    pub gimbal_max: f32,
    pub rcs_torque_max: f32,
    pub rcs_near_scale: f32,
    pub rcs_far_scale: f32,
    pub rcs_boost_radius: f32,
    pub rcs_schedule_width: f32,
    pub cd: f32,
    pub rho: f32,
    pub fin_area_each: f32,
    pub fin_count: f32,
    pub fin_station_x: f32,
    pub fin_cl_alpha: f32,
    pub fin_cd0: f32,
    pub fin_cd_alpha: f32,
    pub fin_alpha_stall: f32,
    pub fin_torque_damping: f32,
    pub grid_control_max: f32,
    pub grid_control_cl_y: f32,
    pub grid_control_cl_z: f32,
    pub grid_roll_control_cl: f32,
    pub terminal_drag_gain: f32,
    pub terminal_drag_alt_low: f32,
    pub terminal_drag_alt_high: f32,
    pub gear_contact_height_m: f32,
    pub gear_footprint_radius_m: f32,
    pub gear_n_legs: usize,
    pub gear_spring_k_npm: f32,
    pub gear_damper_c_nspm: f32,
    pub gear_friction_mu: f32,
    pub gear_contact_restore_scale: f32,
    pub gear_bottom_out_force_frac: f32,
    pub max_tilt: f32,
    pub max_lateral: f32,
    pub init_x: f32,
}

impl PlantConfig {
    pub fn frozen(controller: &ControllerConfig) -> Self {
        debug_assert_eq!(controller.initial_mass_kg(), 29_200.0);
        Self {
            dt: 0.1,
            max_steps: 900,
            g: 9.81,
            mass_full: 29_200.0,
            mass_empty: 22_200.0,
            thrust_max: 845_000.0,
            isp: 282.0,
            length: 40.0,
            radius: 1.83,
            lever_x: -15.0,
            gimbal_max: 0.0873,
            rcs_torque_max: 180_000.0,
            rcs_near_scale: 1.0,
            rcs_far_scale: 3.0,
            rcs_boost_radius: 700.0,
            rcs_schedule_width: 140.0,
            cd: 0.60,
            rho: 1.05,
            fin_area_each: 0.75,
            fin_count: 4.0,
            fin_station_x: 0.5,
            fin_cl_alpha: 2.0,
            fin_cd0: 0.18,
            fin_cd_alpha: 1.20,
            fin_alpha_stall: 0.45,
            fin_torque_damping: 0.22,
            grid_control_max: 0.35,
            grid_control_cl_y: 1.6,
            grid_control_cl_z: 1.6,
            grid_roll_control_cl: 0.55,
            terminal_drag_gain: 3.0,
            terminal_drag_alt_low: 300.0,
            terminal_drag_alt_high: 350.0,
            gear_contact_height_m: 3.0015,
            gear_footprint_radius_m: 6.0,
            gear_n_legs: 4,
            gear_spring_k_npm: 2.0e4,
            gear_damper_c_nspm: 3.0e4,
            gear_friction_mu: 0.3,
            gear_contact_restore_scale: 0.0,
            gear_bottom_out_force_frac: 0.90,
            max_tilt: 1.45,
            max_lateral: 1_800.0,
            init_x: 2_000.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ForceResult {
    pub accel: Vec3,
    pub domega: Vec3,
    pub thrust: f32,
    pub lateral_rcs_realized_nm: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct PlantStep {
    pub state: State,
    pub done: bool,
    pub finite: bool,
    pub lateral_rcs_realized_nm: f32,
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
pub fn inertia_diag(mass: f32, cfg: &PlantConfig) -> Vec3 {
    let ixx = 0.5 * mass * cfg.radius * cfg.radius;
    let iyy = (1.0 / 12.0) * mass * (cfg.length * cfg.length + 3.0 * cfg.radius * cfg.radius);
    [ixx, iyy, iyy]
}

#[inline(always)]
pub fn tilt_angle_from_q(q: [f32; 4]) -> f32 {
    clip(quat_to_rot(q)[0][0], -1.0, 1.0).acos()
}

#[inline(always)]
fn apply_action_boundary(mut action: Action) -> Action {
    for value in &mut action {
        *value = clip(*value, -1.0, 1.0);
    }
    action[4] = 0.0;
    action[5] = 0.0;
    action[8] = 0.0;
    action
}

#[inline(always)]
fn rcs_radial_gain(radius: f32, cfg: &PlantConfig) -> f32 {
    let x = (radius - cfg.rcs_boost_radius) / cfg.rcs_schedule_width.max(1.0e-6);
    let schedule = 0.5 * (1.0 + x.tanh());
    cfg.rcs_near_scale + (cfg.rcs_far_scale - cfg.rcs_near_scale) * schedule
}

#[inline(always)]
fn terminal_drag_gate(altitude: f32, cfg: &PlantConfig) -> f32 {
    let span = cfg.terminal_drag_alt_high - cfg.terminal_drag_alt_low;
    let fraction = clip(
        (cfg.terminal_drag_alt_high - altitude) / (span + 1.0e-8),
        0.0,
        1.0,
    );
    1.0 + (cfg.terminal_drag_gain - 1.0) * fraction
}

fn landing_gear_contact(
    pos: Vec3,
    vel: Vec3,
    rot: Mat3,
    omega: Vec3,
    mass: f32,
    cfg: &PlantConfig,
) -> (Vec3, Vec3) {
    let omega_i = mat_vec(rot, omega);
    let support_cap = cfg.gear_bottom_out_force_frac * mass * cfg.g / cfg.gear_n_legs as f32;
    let mut force_total = [0.0; 3];
    let mut torque_total = [0.0; 3];
    for i in 0..cfg.gear_n_legs {
        let phi = 2.0 * PI * i as f32 / cfg.gear_n_legs as f32;
        let tip_b = [
            -cfg.gear_contact_height_m,
            cfg.gear_footprint_radius_m * phi.cos(),
            cfg.gear_footprint_radius_m * phi.sin(),
        ];
        let r_i = mat_vec(rot, tip_b);
        let tip_height = pos[0] + r_i[0];
        let penetration = (-tip_height).max(0.0);
        let contact = penetration > 0.0;
        let v_tip_i = add(vel, cross(omega_i, r_i));
        let vn_down = -v_tip_i[0];
        let spring = (cfg.gear_spring_k_npm * penetration).min(support_cap);
        let fnormal = (spring
            + if contact {
                cfg.gear_damper_c_nspm * vn_down.max(0.0)
            } else {
                0.0
            })
        .max(0.0);
        let v_tan = [0.0, v_tip_i[1], v_tip_i[2]];
        let v_tan_norm = norm(v_tan) + 1.0e-8;
        let friction_scale = -cfg.gear_friction_mu * fnormal / (v_tan_norm + 0.5);
        let force_i = [
            fnormal,
            friction_scale * v_tan[1],
            friction_scale * v_tan[2],
        ];
        force_total = add(force_total, force_i);
        torque_total = add(torque_total, cross(r_i, force_i));
    }
    let torque_body = scale(mat_t_vec(rot, torque_total), cfg.gear_contact_restore_scale);
    (force_total, torque_body)
}

pub fn forces(state: &State, raw_action: &Action, cfg: &PlantConfig) -> ForceResult {
    let pos = state_vec3(state, 0);
    let vel = state_vec3(state, 3);
    let q = state_quat(state);
    let omega = state_vec3(state, 10);
    let mass = state[13];
    let initial_r = state[15];
    let action = apply_action_boundary(*raw_action);
    let gy = action[0] * cfg.gimbal_max;
    let gz = action[1] * cfg.gimbal_max;
    let throttle = 0.5 * (action[2] + 1.0);
    let lateral_r = (pos[1] * pos[1] + pos[2] * pos[2] + 1.0e-8).sqrt();
    let authority = cfg.rcs_torque_max * rcs_radial_gain(lateral_r, cfg);
    let rcs = [
        action[3] * authority,
        action[4] * authority,
        action[5] * authority,
    ];
    let lateral_rcs_realized_nm =
        (action[4] * action[4] + action[5] * action[5]).sqrt() * authority;
    let grid = [
        action[6] * cfg.grid_control_max,
        action[7] * cfg.grid_control_max,
        action[8] * cfg.grid_control_max,
    ];
    let thrust = if mass > cfg.mass_empty {
        throttle * cfg.thrust_max
    } else {
        0.0
    };

    let rot = quat_to_rot(q);
    let thrust_b = mat_vec(gimbal_rot(gy, gz), [thrust, 0.0, 0.0]);
    let thrust_i = mat_vec(rot, thrust_b);
    let speed = norm(vel) + 1.0e-8;
    let coast_drag_gate = terminal_drag_gate(pos[0], cfg);
    let area = PI * cfg.radius * cfg.radius;
    let drag_i = scale(vel, -0.5 * cfg.rho * cfg.cd * area * speed);

    let v_b = mat_t_vec(rot, vel);
    let v_lat_b = [0.0, v_b[1], v_b[2]];
    let v_lat = norm(v_lat_b) + 1.0e-8;
    let alpha_eff = v_lat.atan2(v_b[0].abs() + 1.0e-8);
    let alpha_soft = cfg.fin_alpha_stall * (alpha_eff / (cfg.fin_alpha_stall + 1.0e-8)).tanh();
    let q_dyn = 0.5 * cfg.rho * speed * speed;
    let fin_area_total = cfg.fin_area_each * cfg.fin_count;
    let lift_mag = q_dyn * fin_area_total * cfg.fin_cl_alpha * alpha_soft;
    let fin_lift_b = scale(v_lat_b, -lift_mag / v_lat);
    let grid_ctrl_b = [
        0.0,
        q_dyn * fin_area_total * cfg.grid_control_cl_y * grid[0],
        q_dyn * fin_area_total * cfg.grid_control_cl_z * grid[1],
    ];
    let cd_fin = cfg.fin_cd0
        + cfg.fin_cd_alpha * alpha_soft * alpha_soft
        + 0.25 * (grid[0] * grid[0] + grid[1] * grid[1] + grid[2] * grid[2]);
    let fin_drag_b = scale(
        v_b,
        -q_dyn * fin_area_total * cd_fin * coast_drag_gate / (speed + 1.0e-8),
    );
    let fin_b = add(add(fin_lift_b, grid_ctrl_b), fin_drag_b);
    let fin_i = mat_vec(rot, fin_b);
    let gravity_i = [-mass * cfg.g, 0.0, 0.0];
    let (gear_i, gear_torque_b) = landing_gear_contact(pos, vel, rot, omega, mass, cfg);
    let force_i = add(add(add(add(thrust_i, drag_i), fin_i), gravity_i), gear_i);
    let accel = scale(force_i, 1.0 / mass);

    let torque_engine_b = cross([cfg.lever_x, 0.0, 0.0], thrust_b);
    let torque_fin_b = cross([cfg.fin_station_x, 0.0, 0.0], fin_b);
    let torque_grid_roll_b = [
        q_dyn * fin_area_total * cfg.radius * cfg.grid_roll_control_cl * grid[2],
        0.0,
        0.0,
    ];
    let torque_fin_damp_b = scale(
        omega,
        -cfg.fin_torque_damping * q_dyn * fin_area_total * cfg.radius,
    );
    let torque_b = add(
        add(
            add(add(torque_engine_b, rcs), torque_fin_b),
            torque_grid_roll_b,
        ),
        add(torque_fin_damp_b, gear_torque_b),
    );
    let inertia = inertia_diag(mass, cfg);
    let angular_momentum = mul(inertia, omega);
    let gyro = cross(omega, angular_momentum);
    let domega = [
        (torque_b[0] - gyro[0]) / inertia[0],
        (torque_b[1] - gyro[1]) / inertia[1],
        (torque_b[2] - gyro[2]) / inertia[2],
    ];
    let _ = initial_r; // Radial drag gain is frozen uniform at 3.0 in this protocol.
    ForceResult {
        accel,
        domega,
        thrust,
        lateral_rcs_realized_nm,
    }
}

#[inline]
fn stage_state(base: &State, pos: Vec3, vel: Vec3, q: [f32; 4], omega: Vec3, time: f32) -> State {
    let mut state = *base;
    state[0..3].copy_from_slice(&pos);
    state[3..6].copy_from_slice(&vel);
    state[6..10].copy_from_slice(&q);
    state[10..13].copy_from_slice(&omega);
    state[14] = time;
    state
}

pub fn step(state: &State, action: &Action, cfg: &PlantConfig) -> PlantStep {
    let pos = state_vec3(state, 0);
    let vel = state_vec3(state, 3);
    let q = state_quat(state);
    let omega = state_vec3(state, 10);
    let mass = state[13];
    let time = state[14];
    let initial_r = state[15];

    let k1 = forces(state, action, cfg);
    let half_dt = 0.5 * cfg.dt;
    let vel2 = add(vel, scale(k1.accel, half_dt));
    let pos2 = add(pos, scale(vel, half_dt));
    let omega2 = add(omega, scale(k1.domega, half_dt));
    let q2 = integrate_quat_exact(q, omega, half_dt);
    let s2 = stage_state(state, pos2, vel2, q2, omega2, time + 0.5);
    let k2 = forces(&s2, action, cfg);

    let vel3 = add(vel, scale(k2.accel, half_dt));
    let pos3 = add(pos, scale(vel2, half_dt));
    let omega3 = add(omega, scale(k2.domega, half_dt));
    let q3 = integrate_quat_exact(q, omega2, half_dt);
    let s3 = stage_state(state, pos3, vel3, q3, omega3, time + 0.5);
    let k3 = forces(&s3, action, cfg);

    let vel4 = add(vel, scale(k3.accel, cfg.dt));
    let pos4 = add(pos, scale(vel3, cfg.dt));
    let omega4 = add(omega, scale(k3.domega, cfg.dt));
    let q4 = integrate_quat_exact(q, omega3, cfg.dt);
    let s4 = stage_state(state, pos4, vel4, q4, omega4, time + 1.0);
    let k4 = forces(&s4, action, cfg);

    let sixth_dt = cfg.dt / 6.0;
    let vel_new = [
        vel[0] + sixth_dt * (k1.accel[0] + 2.0 * k2.accel[0] + 2.0 * k3.accel[0] + k4.accel[0]),
        vel[1] + sixth_dt * (k1.accel[1] + 2.0 * k2.accel[1] + 2.0 * k3.accel[1] + k4.accel[1]),
        vel[2] + sixth_dt * (k1.accel[2] + 2.0 * k2.accel[2] + 2.0 * k3.accel[2] + k4.accel[2]),
    ];
    let mut pos_new = [
        pos[0] + sixth_dt * (vel[0] + 2.0 * vel2[0] + 2.0 * vel3[0] + vel4[0]),
        pos[1] + sixth_dt * (vel[1] + 2.0 * vel2[1] + 2.0 * vel3[1] + vel4[1]),
        pos[2] + sixth_dt * (vel[2] + 2.0 * vel2[2] + 2.0 * vel3[2] + vel4[2]),
    ];
    let omega_new = [
        omega[0]
            + sixth_dt * (k1.domega[0] + 2.0 * k2.domega[0] + 2.0 * k3.domega[0] + k4.domega[0]),
        omega[1]
            + sixth_dt * (k1.domega[1] + 2.0 * k2.domega[1] + 2.0 * k3.domega[1] + k4.domega[1]),
        omega[2]
            + sixth_dt * (k1.domega[2] + 2.0 * k2.domega[2] + 2.0 * k3.domega[2] + k4.domega[2]),
    ];
    let omega_avg = [
        (omega[0] + 2.0 * omega2[0] + 2.0 * omega3[0] + omega_new[0]) / 6.0,
        (omega[1] + 2.0 * omega2[1] + 2.0 * omega3[1] + omega_new[1]) / 6.0,
        (omega[2] + 2.0 * omega2[2] + 2.0 * omega3[2] + omega_new[2]) / 6.0,
    ];
    let q_new = integrate_quat_exact(q, omega_avg, cfg.dt);
    let thrust = (k1.thrust + 2.0 * k2.thrust + 2.0 * k3.thrust + k4.thrust) / 6.0;
    let dm = thrust / (cfg.g * cfg.isp + 1.0e-8) * cfg.dt;
    let mass_new = (mass - dm).max(cfg.mass_empty);
    let time_new = time + 1.0;
    pos_new[0] = pos_new[0].max(0.0);

    let mut next = *state;
    next[0..3].copy_from_slice(&pos_new);
    next[3..6].copy_from_slice(&vel_new);
    next[6..10].copy_from_slice(&q_new);
    next[10..13].copy_from_slice(&omega_new);
    next[13] = mass_new;
    next[14] = time_new;
    next[15] = initial_r;
    let lateral = (pos_new[1] * pos_new[1] + pos_new[2] * pos_new[2] + 1.0e-8).sqrt();
    let tilt = tilt_angle_from_q(q_new);
    let touched = pos_new[0] <= 1.0e-4;
    let flipped = tilt > cfg.max_tilt;
    let oob = lateral > cfg.max_lateral || pos_new[0] > cfg.init_x + 500.0;
    let timeout = time_new >= cfg.max_steps as f32;
    PlantStep {
        finite: all_finite(&next),
        state: next,
        done: touched || flipped || oob || timeout,
        lateral_rcs_realized_nm: k1.lateral_rcs_realized_nm,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn frozen_mass_and_channel_contract() {
        let controller = ControllerConfig::load(Path::new("config/controller.json")).unwrap();
        let cfg = PlantConfig::frozen(&controller);
        assert_eq!(cfg.mass_full, 29_200.0);
        assert_eq!(cfg.mass_empty, 22_200.0);
        let mut state = [0.0; 16];
        state[0] = 2_000.0;
        state[6] = 1.0;
        state[13] = cfg.mass_full;
        state[15] = 500.0;
        let mut action = [0.0; 9];
        action[4] = 1.0;
        action[5] = -1.0;
        action[8] = 1.0;
        assert_eq!(forces(&state, &action, &cfg).lateral_rcs_realized_nm, 0.0);
    }
}
