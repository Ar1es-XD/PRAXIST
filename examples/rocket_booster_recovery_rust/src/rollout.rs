use std::f32::consts::PI;

use rayon::prelude::*;

use crate::{
    config::ControllerConfig,
    controller::{ControllerMemory, control_step, init_memory},
    math::{add, cross, mat_vec, normalize_quat, quat_to_rot},
    plant::{PlantConfig, State, step},
};

pub const AUDIT_COLUMNS: [&str; 21] = [
    "active_steps",
    "max_forbidden_action_abs",
    "max_plant_lateral_rcs_nm",
    "rcs_roll_abs_impulse_nms",
    "rcs_roll_signed_impulse_nms",
    "gimbal_total_variation_rad",
    "grid_total_variation_rad",
    "throttle_total_variation",
    "emergency_steps",
    "phase0_steps",
    "phase1_steps",
    "phase2_steps",
    "phase3_steps",
    "gimbal_saturation_steps",
    "grid_saturation_steps",
    "throttle_saturation_steps",
    "rcs_roll_cap_steps",
    "max_body_axis_error_rad",
    "min_brake_margin_m",
    "max_dynamic_pressure_pa",
    "nonfinite_seen",
];

#[derive(Clone, Copy, Debug)]
pub struct Audit {
    pub active_steps: u32,
    pub max_forbidden_action_abs: f32,
    pub max_plant_lateral_rcs_nm: f32,
    pub rcs_roll_abs_impulse_nms: f32,
    pub rcs_roll_signed_impulse_nms: f32,
    pub gimbal_total_variation_rad: f32,
    pub grid_total_variation_rad: f32,
    pub throttle_total_variation: f32,
    pub emergency_steps: u32,
    pub phase_steps: [u32; 4],
    pub gimbal_saturation_steps: u32,
    pub grid_saturation_steps: u32,
    pub throttle_saturation_steps: u32,
    pub rcs_roll_cap_steps: u32,
    pub max_body_axis_error_rad: f32,
    pub min_brake_margin_m: f32,
    pub max_dynamic_pressure_pa: f32,
    pub nonfinite_seen: bool,
}

impl Default for Audit {
    fn default() -> Self {
        Self {
            active_steps: 0,
            max_forbidden_action_abs: 0.0,
            max_plant_lateral_rcs_nm: 0.0,
            rcs_roll_abs_impulse_nms: 0.0,
            rcs_roll_signed_impulse_nms: 0.0,
            gimbal_total_variation_rad: 0.0,
            grid_total_variation_rad: 0.0,
            throttle_total_variation: 0.0,
            emergency_steps: 0,
            phase_steps: [0; 4],
            gimbal_saturation_steps: 0,
            grid_saturation_steps: 0,
            throttle_saturation_steps: 0,
            rcs_roll_cap_steps: 0,
            max_body_axis_error_rad: 0.0,
            min_brake_margin_m: f32::INFINITY,
            max_dynamic_pressure_pa: 0.0,
            nonfinite_seen: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RolloutResult {
    pub terminal_state: State,
    pub terminal_previous_omega: [f32; 3],
    pub done: bool,
    pub max_abs_action: [f32; 9],
    pub audit: Audit,
    pub first_contact_state: State,
    pub first_contact_detected: bool,
    pub first_contact_step: u32,
    pub first_contact_leg_sink_speed_mps: f32,
}

#[inline]
fn tip_metrics(state: &State, cfg: &PlantConfig) -> (f32, f32) {
    let rot = quat_to_rot([state[6], state[7], state[8], state[9]]);
    let omega_i = mat_vec(rot, [state[10], state[11], state[12]]);
    let mut heights = [0.0_f32; 4];
    let mut sinks = [0.0_f32; 4];
    for i in 0..cfg.gear_n_legs {
        let phase = 2.0 * PI * i as f32 / cfg.gear_n_legs as f32;
        let tip_b = [
            -cfg.gear_contact_height_m,
            cfg.gear_footprint_radius_m * phase.cos(),
            cfg.gear_footprint_radius_m * phase.sin(),
        ];
        let offset_i = mat_vec(rot, tip_b);
        heights[i] = state[0] + offset_i[0];
        let tip_velocity = add([state[3], state[4], state[5]], cross(omega_i, offset_i));
        sinks[i] = (-tip_velocity[0]).max(0.0);
    }
    let min_height = heights[..cfg.gear_n_legs]
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let mut leg_sink = 0.0_f32;
    for i in 0..cfg.gear_n_legs {
        if heights[i] <= min_height + 1.0e-3 {
            leg_sink = leg_sink.max(sinks[i]);
        }
    }
    (min_height, leg_sink)
}

#[inline]
fn interpolate_state(state: &State, proposed: &State, alpha: f32) -> State {
    let mut value = [0.0; 16];
    for i in 0..16 {
        value[i] = state[i] + alpha * (proposed[i] - state[i]);
    }
    let q = normalize_quat([value[6], value[7], value[8], value[9]]);
    value[6..10].copy_from_slice(&q);
    value
}

#[inline]
fn contact_state(
    state: &State,
    proposed: &State,
    height_now: f32,
    height_next: f32,
    cfg: &PlantConfig,
) -> (State, f32) {
    let alpha = if height_now > 0.0 {
        let mut lo = 0.0_f32;
        let mut hi = 1.0_f32;
        for _ in 0..10 {
            let mid = 0.5 * (lo + hi);
            let mid_state = interpolate_state(state, proposed, mid);
            if tip_metrics(&mid_state, cfg).0 <= 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        hi
    } else {
        (height_now / (height_now - height_next).max(1.0e-8)).clamp(0.0, 1.0)
    };
    let contact = interpolate_state(state, proposed, alpha);
    let sink = tip_metrics(&contact, cfg).1;
    (contact, sink)
}

#[inline]
fn update_audit(
    audit: &mut Audit,
    action: &[f32; 9],
    diagnostic: &[f32; 18],
    previous_memory: &ControllerMemory,
    plant_lateral_rcs_nm: f32,
    cfg: &ControllerConfig,
) {
    audit.active_steps += 1;
    for index in [4, 5, 8] {
        audit.max_forbidden_action_abs = audit.max_forbidden_action_abs.max(action[index].abs());
    }
    audit.max_plant_lateral_rcs_nm = audit
        .max_plant_lateral_rcs_nm
        .max(plant_lateral_rcs_nm.abs());
    let rcs_nm = diagnostic[16];
    audit.rcs_roll_abs_impulse_nms += rcs_nm.abs() * cfg.dt;
    audit.rcs_roll_signed_impulse_nms += rcs_nm * cfg.dt;
    audit.gimbal_total_variation_rad += (diagnostic[11] - previous_memory.prev_gimbal_rad[0]).abs()
        + (diagnostic[12] - previous_memory.prev_gimbal_rad[1]).abs();
    audit.grid_total_variation_rad += (diagnostic[13] - previous_memory.prev_grid_rad[0]).abs()
        + (diagnostic[14] - previous_memory.prev_grid_rad[1]).abs();
    audit.throttle_total_variation += (diagnostic[15] - previous_memory.prev_throttle).abs();
    let emergency = diagnostic[1] > 0.5;
    audit.emergency_steps += u32::from(emergency);
    let phase = (diagnostic[0] as i32).clamp(0, 3) as usize;
    audit.phase_steps[phase] += 1;
    if diagnostic[11].abs().max(diagnostic[12].abs()) >= cfg.gimbal_normal_rad - 1.0e-6 {
        audit.gimbal_saturation_steps += 1;
    }
    if diagnostic[13].abs().max(diagnostic[14].abs()) >= cfg.grid_normal_rad - 1.0e-6 {
        audit.grid_saturation_steps += 1;
    }
    if diagnostic[15] <= cfg.throttle_normal_min + 1.0e-6
        || diagnostic[15] >= cfg.throttle_normal_max - 1.0e-6
    {
        audit.throttle_saturation_steps += 1;
    }
    let cap = if emergency {
        cfg.rcs_roll_emergency_cap_nm
    } else {
        cfg.rcs_roll_normal_cap_nm
    };
    if rcs_nm.abs() >= cap - 1.0 {
        audit.rcs_roll_cap_steps += 1;
    }
    audit.max_body_axis_error_rad = audit.max_body_axis_error_rad.max(diagnostic[17]);
    audit.min_brake_margin_m = audit.min_brake_margin_m.min(diagnostic[3]);
    audit.max_dynamic_pressure_pa = audit.max_dynamic_pressure_pa.max(diagnostic[4]);
}

pub fn rollout_one(
    initial_state: &State,
    controller_cfg: &ControllerConfig,
    plant_cfg: &PlantConfig,
) -> RolloutResult {
    let mut state = *initial_state;
    let mut memory = init_memory(&state, controller_cfg);
    let mut audit = Audit::default();
    let mut max_abs_action = [0.0_f32; 9];
    let mut terminal_previous_omega = [state[10], state[11], state[12]];
    let mut first_contact_state = [0.0_f32; 16];
    let mut first_contact_detected = false;
    let mut first_contact_step = 0_u32;
    let mut first_contact_leg_sink_speed_mps = 0.0_f32;
    let mut done = false;

    for _ in 0..plant_cfg.max_steps {
        let previous_memory = memory;
        let output = control_step(&state, &memory, controller_cfg);
        let proposed = step(&state, &output.action, plant_cfg);
        let finite = proposed.finite;
        let height_now = tip_metrics(&state, plant_cfg).0;
        let height_next = tip_metrics(&proposed.state, plant_cfg).0;
        let crossing = finite && !first_contact_detected && height_next <= 0.0;
        let (scored_state, contact_sink) = if crossing {
            contact_state(&state, &proposed.state, height_now, height_next, plant_cfg)
        } else {
            (proposed.state, 0.0)
        };

        for (maximum, value) in max_abs_action.iter_mut().zip(output.action) {
            *maximum = maximum.max(value.abs());
        }
        update_audit(
            &mut audit,
            &output.action,
            &output.diagnostic,
            &previous_memory,
            proposed.lateral_rcs_realized_nm,
            controller_cfg,
        );
        if !finite {
            audit.nonfinite_seen = true;
        }
        memory = output.memory;

        if crossing {
            first_contact_state = scored_state;
            first_contact_detected = true;
            first_contact_step = audit.active_steps;
            first_contact_leg_sink_speed_mps = contact_sink;
        }
        let plant_terminal = proposed.done || !finite;
        if crossing || plant_terminal {
            terminal_previous_omega = [state[10], state[11], state[12]];
            state = scored_state;
            done = true;
            break;
        }
        state = proposed.state;
    }
    let terminal_state = if first_contact_detected {
        first_contact_state
    } else {
        state
    };
    RolloutResult {
        terminal_state,
        terminal_previous_omega,
        done,
        max_abs_action,
        audit,
        first_contact_state,
        first_contact_detected,
        first_contact_step,
        first_contact_leg_sink_speed_mps,
    }
}

pub fn rollout_all(
    states: &[State],
    controller_cfg: &ControllerConfig,
    plant_cfg: &PlantConfig,
    threads: usize,
) -> Vec<RolloutResult> {
    let run = || {
        states
            .par_iter()
            .map(|state| rollout_one(state, controller_cfg, plant_cfg))
            .collect()
    };
    if threads == 0 {
        run()
    } else {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("build deterministic trajectory thread pool")
            .install(run)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contact_interpolation_lands_on_tip_plane() {
        let mut before = [0.0; 16];
        before[0] = 3.1015;
        before[6] = 1.0;
        before[13] = 29_200.0;
        let mut after = before;
        after[0] = 2.9015;
        let cfg = PlantConfig::frozen(
            &ControllerConfig::load(std::path::Path::new("config/controller.json")).unwrap(),
        );
        let (contact, _) = contact_state(&before, &after, 0.1, -0.1, &cfg);
        assert!(tip_metrics(&contact, &cfg).0.abs() < 3.0e-4);
    }
}
