use std::f32::consts::PI;

use rayon::prelude::*;
use rocket_booster_recovery_rust::{
    math::{add, cross, mat_vec, normalize_quat, quat_to_rot},
    metrics::quantile,
    plant::{PlantConfig, State, step},
    rollout::{Audit, RolloutResult},
};
use serde_json::{Value, json};

use crate::candidate_api::CandidateController;

const DT: f32 = 0.1;
const GIMBAL_NORMAL_RAD: f32 = 0.075;
const GRID_NORMAL_RAD: f32 = 0.25;
const THROTTLE_NORMAL_MIN: f32 = 0.02;
const THROTTLE_NORMAL_MAX: f32 = 0.98;
const RCS_ROLL_NORMAL_CAP_NM: f32 = 30_000.0;
const RCS_ROLL_EMERGENCY_CAP_NM: f32 = 60_000.0;

#[inline]
fn tip_metrics(state: &State, cfg: &PlantConfig) -> (f32, f32) {
    let rot = quat_to_rot([state[6], state[7], state[8], state[9]]);
    let omega_i = mat_vec(rot, [state[10], state[11], state[12]]);
    let mut heights = [0.0_f32; 4];
    let mut sinks = [0.0_f32; 4];
    for index in 0..cfg.gear_n_legs {
        let phase = 2.0 * PI * index as f32 / cfg.gear_n_legs as f32;
        let tip_body = [
            -cfg.gear_contact_height_m,
            cfg.gear_footprint_radius_m * phase.cos(),
            cfg.gear_footprint_radius_m * phase.sin(),
        ];
        let offset_i = mat_vec(rot, tip_body);
        heights[index] = state[0] + offset_i[0];
        let tip_velocity = add([state[3], state[4], state[5]], cross(omega_i, offset_i));
        sinks[index] = (-tip_velocity[0]).max(0.0);
    }
    let minimum = heights[..cfg.gear_n_legs]
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let mut leg_sink = 0.0_f32;
    for index in 0..cfg.gear_n_legs {
        if heights[index] <= minimum + 1.0e-3 {
            leg_sink = leg_sink.max(sinks[index]);
        }
    }
    (minimum, leg_sink)
}

#[inline]
fn interpolate_state(state: &State, proposed: &State, alpha: f32) -> State {
    let mut value = [0.0; 16];
    for index in 0..16 {
        value[index] = state[index] + alpha * (proposed[index] - state[index]);
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
        let mut low = 0.0_f32;
        let mut high = 1.0_f32;
        for _ in 0..10 {
            let middle = 0.5 * (low + high);
            let middle_state = interpolate_state(state, proposed, middle);
            if tip_metrics(&middle_state, cfg).0 <= 0.0 {
                high = middle;
            } else {
                low = middle;
            }
        }
        high
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
    previous_gimbal: [f32; 2],
    previous_grid: [f32; 2],
    previous_throttle: f32,
    plant_lateral_rcs_nm: f32,
) {
    audit.active_steps += 1;
    for index in [4, 5, 8] {
        audit.max_forbidden_action_abs = audit.max_forbidden_action_abs.max(action[index].abs());
    }
    audit.max_plant_lateral_rcs_nm = audit
        .max_plant_lateral_rcs_nm
        .max(plant_lateral_rcs_nm.abs());
    let rcs_nm = diagnostic[16];
    audit.rcs_roll_abs_impulse_nms += rcs_nm.abs() * DT;
    audit.rcs_roll_signed_impulse_nms += rcs_nm * DT;
    audit.gimbal_total_variation_rad +=
        (diagnostic[11] - previous_gimbal[0]).abs() + (diagnostic[12] - previous_gimbal[1]).abs();
    audit.grid_total_variation_rad +=
        (diagnostic[13] - previous_grid[0]).abs() + (diagnostic[14] - previous_grid[1]).abs();
    audit.throttle_total_variation += (diagnostic[15] - previous_throttle).abs();
    let emergency = diagnostic[1] > 0.5;
    audit.emergency_steps += u32::from(emergency);
    let phase = (diagnostic[0] as i32).clamp(0, 3) as usize;
    audit.phase_steps[phase] += 1;
    if diagnostic[11].abs().max(diagnostic[12].abs()) >= GIMBAL_NORMAL_RAD - 1.0e-6 {
        audit.gimbal_saturation_steps += 1;
    }
    if diagnostic[13].abs().max(diagnostic[14].abs()) >= GRID_NORMAL_RAD - 1.0e-6 {
        audit.grid_saturation_steps += 1;
    }
    if diagnostic[15] <= THROTTLE_NORMAL_MIN + 1.0e-6
        || diagnostic[15] >= THROTTLE_NORMAL_MAX - 1.0e-6
    {
        audit.throttle_saturation_steps += 1;
    }
    let cap = if emergency {
        RCS_ROLL_EMERGENCY_CAP_NM
    } else {
        RCS_ROLL_NORMAL_CAP_NM
    };
    if rcs_nm.abs() >= cap - 1.0 {
        audit.rcs_roll_cap_steps += 1;
    }
    audit.max_body_axis_error_rad = audit.max_body_axis_error_rad.max(diagnostic[17]);
    audit.min_brake_margin_m = audit.min_brake_margin_m.min(diagnostic[3]);
    audit.max_dynamic_pressure_pa = audit.max_dynamic_pressure_pa.max(diagnostic[4]);
}

fn landing_one<C: CandidateController>(
    initial: &State,
    config: &C::Config,
    plant_cfg: &PlantConfig,
) -> RolloutResult {
    let mut state = *initial;
    let mut memory = C::init_memory(&state, config);
    let mut audit = Audit::default();
    let mut max_abs_action = [0.0_f32; 9];
    let mut terminal_previous_omega = [state[10], state[11], state[12]];
    let mut first_contact_state = [0.0_f32; 16];
    let mut first_contact_detected = false;
    let mut first_contact_step = 0_u32;
    let mut first_contact_leg_sink_speed_mps = 0.0_f32;
    let mut done = false;

    for _ in 0..plant_cfg.max_steps {
        let (previous_gimbal, previous_grid, previous_throttle) = C::previous_actuators(&memory);
        let output = C::control_step(&state, &memory, config);
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
            previous_gimbal,
            previous_grid,
            previous_throttle,
            proposed.lateral_rcs_realized_nm,
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
        if crossing || proposed.done || !finite {
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

pub fn landing_all<C: CandidateController>(
    states: &[State],
    config: &C::Config,
    plant_cfg: &PlantConfig,
    threads: usize,
) -> Vec<RolloutResult> {
    let run = || {
        states
            .par_iter()
            .map(|state| landing_one::<C>(state, config, plant_cfg))
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

#[derive(Clone, Debug)]
struct RollRow {
    terminal_state: State,
    terminal_previous_omega: [f32; 3],
    last_unsettled_step: u32,
    peak_abs_roll_rate: f32,
    peak_pitch_yaw_rate: f32,
    rcs_switches: u32,
    rcs_total_variation_nm: f32,
    max_forbidden_action_abs: f32,
    nonfinite_seen: bool,
}

fn roll_one<C: CandidateController>(
    initial: &State,
    config: &C::Config,
    plant_cfg: &PlantConfig,
) -> RollRow {
    let mut state = *initial;
    let mut memory = C::init_memory(&state, config);
    let mut terminal_previous_omega = [state[10], state[11], state[12]];
    let mut active_steps = 0_u32;
    let mut last_unsettled_step = 0_u32;
    let mut peak_abs_roll_rate = state[10].abs();
    let mut peak_pitch_yaw_rate = (state[11] * state[11] + state[12] * state[12]).sqrt();
    let mut previous_rcs = 0.0_f32;
    let mut rcs_switches = 0_u32;
    let mut rcs_total_variation_nm = 0.0_f32;
    let mut max_forbidden_action_abs = 0.0_f32;
    let mut nonfinite_seen = false;

    for _ in 0..plant_cfg.max_steps {
        let output = C::control_step(&state, &memory, config);
        let proposed = step(&state, &output.action, plant_cfg);
        let finite = proposed.finite;
        active_steps += 1;
        if proposed.state[10].abs() >= 0.02 {
            last_unsettled_step = active_steps;
        }
        peak_abs_roll_rate = peak_abs_roll_rate.max(proposed.state[10].abs());
        peak_pitch_yaw_rate = peak_pitch_yaw_rate.max(
            (proposed.state[11] * proposed.state[11] + proposed.state[12] * proposed.state[12])
                .sqrt(),
        );
        let rcs = output.diagnostic[16];
        if previous_rcs.abs() > 1.0 && rcs.abs() > 1.0 && previous_rcs.signum() != rcs.signum() {
            rcs_switches += 1;
        }
        rcs_total_variation_nm += (rcs - previous_rcs).abs();
        previous_rcs = rcs;
        for index in [4, 5, 8] {
            max_forbidden_action_abs = max_forbidden_action_abs.max(output.action[index].abs());
        }
        if !finite {
            nonfinite_seen = true;
        }
        memory = output.memory;
        if proposed.done || !finite {
            terminal_previous_omega = [state[10], state[11], state[12]];
            state = proposed.state;
            break;
        }
        state = proposed.state;
    }

    RollRow {
        terminal_state: state,
        terminal_previous_omega,
        last_unsettled_step,
        peak_abs_roll_rate,
        peak_pitch_yaw_rate,
        rcs_switches,
        rcs_total_variation_nm,
        max_forbidden_action_abs,
        nonfinite_seen,
    }
}

pub fn roll_metrics<C: CandidateController>(
    states: &[State],
    config: &C::Config,
    plant_cfg: &PlantConfig,
    threads: usize,
) -> (Value, Value) {
    let run = || {
        states
            .par_iter()
            .map(|state| roll_one::<C>(state, config, plant_cfg))
            .collect::<Vec<_>>()
    };
    let rows = if threads == 0 {
        run()
    } else {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("build deterministic roll thread pool")
            .install(run)
    };
    let stable_count = rows
        .iter()
        .filter(|row| {
            row.terminal_state[10].abs() < 0.02
                && row.terminal_previous_omega[0].abs() < 0.02
                && !row.nonfinite_seen
        })
        .count();
    let settling: Vec<f64> = rows
        .iter()
        .map(|row| row.last_unsettled_step as f64 * 0.1_f64)
        .collect();
    let peak_roll: Vec<f64> = rows
        .iter()
        .map(|row| row.peak_abs_roll_rate as f64)
        .collect();
    let peak_pitch_yaw: Vec<f64> = rows
        .iter()
        .map(|row| row.peak_pitch_yaw_rate as f64)
        .collect();
    let switches_mean =
        rows.iter().map(|row| row.rcs_switches as f64).sum::<f64>() / rows.len() as f64;
    let variation_mean = rows
        .iter()
        .map(|row| row.rcs_total_variation_nm as f64)
        .sum::<f64>()
        / rows.len() as f64;
    let forbidden = rows
        .iter()
        .map(|row| row.max_forbidden_action_abs)
        .fold(0.0_f32, f32::max);
    let nonfinite_count = rows.iter().filter(|row| row.nonfinite_seen).count();
    let metrics = json!({
        "roll_stable_rate": stable_count as f64 / rows.len() as f64,
        "roll_settling_time_p95_s": quantile(&settling, 0.95),
        "roll_peak_rate_p95_radps": quantile(&peak_roll, 0.95),
        "roll_pitch_yaw_coupling_p95_radps": quantile(&peak_pitch_yaw, 0.95),
        "roll_rcs_switches_mean": switches_mean,
        "roll_rcs_total_variation_mean_nm": variation_mean,
        "roll_forbidden_action_max_abs": forbidden,
        "roll_nonfinite_rate": nonfinite_count as f64 / rows.len() as f64,
    });
    let detail = json!({
        "trajectories": rows.len(),
        "stable_count": stable_count,
        "initial_roll_rates_radps": [-0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8],
        "settling_definition": "time of last sample outside |omega_x|<0.02 rad/s",
    });
    (metrics, detail)
}
