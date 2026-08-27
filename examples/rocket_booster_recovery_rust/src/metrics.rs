use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::{
    dataset::{Dataset, INITIAL_FUEL_KG, MASS_EMPTY_KG},
    rollout::RolloutResult,
};

pub const REFERENCE_SUCCESS_COUNT: usize = 495;
pub const REFERENCE_SUCCESS_RATE: f64 = 0.040_283_203_125;
pub const NO_CONTACT_SINK_PENALTY_MPS: f64 = 100.0;

#[derive(Clone, Copy, Debug)]
pub struct SuccessThresholds {
    pub lateral_error_max_m: f64,
    pub com_sink_speed_max_mps: f64,
    pub contact_leg_sink_speed_max_mps: f64,
    pub lateral_speed_max_mps: f64,
    pub tilt_max_deg: f64,
    pub roll_rate_max_radps: f64,
    pub pitch_yaw_rate_max_radps: f64,
    pub fuel_reserve_min_fraction: f64,
}

pub const SUCCESS_THRESHOLDS: SuccessThresholds = SuccessThresholds {
    lateral_error_max_m: 5.0,
    com_sink_speed_max_mps: 1.0,
    contact_leg_sink_speed_max_mps: 1.0,
    lateral_speed_max_mps: 0.3,
    tilt_max_deg: 1.5,
    roll_rate_max_radps: 0.02,
    pitch_yaw_rate_max_radps: 0.03,
    fuel_reserve_min_fraction: 0.02,
};

impl SuccessThresholds {
    pub fn as_json(self) -> Value {
        json!({
            "lateral_error_max_m": self.lateral_error_max_m,
            "com_sink_speed_max_mps": self.com_sink_speed_max_mps,
            "contact_leg_sink_speed_max_mps": self.contact_leg_sink_speed_max_mps,
            "lateral_speed_max_mps": self.lateral_speed_max_mps,
            "tilt_max_deg": self.tilt_max_deg,
            "roll_rate_max_radps": self.roll_rate_max_radps,
            "pitch_yaw_rate_max_radps": self.pitch_yaw_rate_max_radps,
            "fuel_reserve_min_fraction": self.fuel_reserve_min_fraction,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ContactArrays {
    pub first_contact_detected: Vec<bool>,
    pub landing_success_pass: Vec<bool>,
    pub finite_first_contact: Vec<bool>,
    pub lateral_error_m: Vec<f64>,
    pub vertical_velocity_mps: Vec<f64>,
    pub com_sink_speed_mps: Vec<f64>,
    pub contact_leg_sink_speed_mps: Vec<f64>,
    pub lateral_speed_mps: Vec<f64>,
    pub total_speed_mps: Vec<f64>,
    pub tilt_rad: Vec<f64>,
    pub roll_rate_radps: Vec<f64>,
    pub pitch_yaw_rate_radps: Vec<f64>,
    pub fuel_fraction: Vec<f64>,
    pub fuel_gate_pass: Vec<bool>,
    pub vertical_gate_pass: Vec<bool>,
}

#[derive(Clone, Debug)]
pub struct LandingReport {
    pub metrics: Value,
    pub details: Value,
    pub arrays: ContactArrays,
}

#[inline]
fn norm2(a: f64, b: f64) -> f64 {
    (a * a + b * b).sqrt()
}

#[inline]
fn norm3(a: f64, b: f64, c: f64) -> f64 {
    (a * a + b * b + c * c).sqrt()
}

fn contact_arrays(results: &[RolloutResult]) -> ContactArrays {
    let n = results.len();
    let mut arrays = ContactArrays {
        first_contact_detected: Vec::with_capacity(n),
        landing_success_pass: Vec::with_capacity(n),
        finite_first_contact: Vec::with_capacity(n),
        lateral_error_m: Vec::with_capacity(n),
        vertical_velocity_mps: Vec::with_capacity(n),
        com_sink_speed_mps: Vec::with_capacity(n),
        contact_leg_sink_speed_mps: Vec::with_capacity(n),
        lateral_speed_mps: Vec::with_capacity(n),
        total_speed_mps: Vec::with_capacity(n),
        tilt_rad: Vec::with_capacity(n),
        roll_rate_radps: Vec::with_capacity(n),
        pitch_yaw_rate_radps: Vec::with_capacity(n),
        fuel_fraction: Vec::with_capacity(n),
        fuel_gate_pass: Vec::with_capacity(n),
        vertical_gate_pass: Vec::with_capacity(n),
    };
    for result in results {
        let state = &result.terminal_state;
        let detected = result.first_contact_detected;
        let leg_sink = result.first_contact_leg_sink_speed_mps as f64;
        let lateral = norm2(state[1] as f64, state[2] as f64);
        let lateral_speed = norm2(state[4] as f64, state[5] as f64);
        let total_speed = norm3(state[3] as f64, state[4] as f64, state[5] as f64);
        let mut q = [
            state[6] as f64,
            state[7] as f64,
            state[8] as f64,
            state[9] as f64,
        ];
        let qnorm = norm3(q[0], q[1], q[2]).hypot(q[3]).max(1.0e-12);
        for value in &mut q {
            *value /= qnorm;
        }
        let r00 = q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3];
        let tilt = r00.clamp(-1.0, 1.0).acos();
        let pitch_yaw_rate = norm2(state[11] as f64, state[12] as f64);
        let fuel_fraction = (state[13] as f64 - MASS_EMPTY_KG as f64) / INITIAL_FUEL_KG as f64;
        let com_sink = (-(state[3] as f64)).max(0.0);
        let finite =
            state.iter().all(|value| value.is_finite()) && leg_sink.is_finite() && detected;
        let fuel_gate =
            detected && finite && fuel_fraction > SUCCESS_THRESHOLDS.fuel_reserve_min_fraction;
        let vertical_gate = detected
            && finite
            && state[3] as f64 >= -SUCCESS_THRESHOLDS.com_sink_speed_max_mps
            && state[3] <= 0.0
            && leg_sink <= SUCCESS_THRESHOLDS.contact_leg_sink_speed_max_mps;
        let success = detected
            && lateral <= SUCCESS_THRESHOLDS.lateral_error_max_m
            && state[3] as f64 >= -SUCCESS_THRESHOLDS.com_sink_speed_max_mps
            && state[3] <= 0.0
            && leg_sink <= SUCCESS_THRESHOLDS.contact_leg_sink_speed_max_mps
            && lateral_speed <= SUCCESS_THRESHOLDS.lateral_speed_max_mps
            && tilt <= SUCCESS_THRESHOLDS.tilt_max_deg.to_radians()
            && (state[10] as f64).abs() <= SUCCESS_THRESHOLDS.roll_rate_max_radps
            && pitch_yaw_rate <= SUCCESS_THRESHOLDS.pitch_yaw_rate_max_radps
            && fuel_fraction > SUCCESS_THRESHOLDS.fuel_reserve_min_fraction
            && finite;
        arrays.first_contact_detected.push(detected);
        arrays.landing_success_pass.push(success);
        arrays.finite_first_contact.push(finite);
        arrays.lateral_error_m.push(lateral);
        arrays.vertical_velocity_mps.push(state[3] as f64);
        arrays.com_sink_speed_mps.push(com_sink);
        arrays.contact_leg_sink_speed_mps.push(leg_sink);
        arrays.lateral_speed_mps.push(lateral_speed);
        arrays.total_speed_mps.push(total_speed);
        arrays.tilt_rad.push(tilt);
        arrays.roll_rate_radps.push(state[10] as f64);
        arrays.pitch_yaw_rate_radps.push(pitch_yaw_rate);
        arrays.fuel_fraction.push(fuel_fraction);
        arrays.fuel_gate_pass.push(fuel_gate);
        arrays.vertical_gate_pass.push(vertical_gate);
    }
    arrays
}

pub fn wilson_interval(successes: usize, total: usize) -> (f64, f64) {
    if total == 0 {
        return (f64::NAN, f64::NAN);
    }
    let z = 1.959_963_984_540_054_f64;
    let p = successes as f64 / total as f64;
    let den = 1.0 + z * z / total as f64;
    let center = (p + z * z / (2.0 * total as f64)) / den;
    let radius = z
        * (p * (1.0 - p) / total as f64 + z * z / (4.0 * total as f64 * total as f64)).sqrt()
        / den;
    ((center - radius).max(0.0), (center + radius).min(1.0))
}

pub fn quantile(values: &[f64], q: f64) -> f64 {
    assert!(!values.is_empty());
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = (sorted.len() - 1) as f64 * q;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] + (index - lower as f64) * (sorted[upper] - sorted[lower])
    }
}

fn stats(values: &[f64], indices: &[usize]) -> Value {
    let selected: Vec<f64> = indices.iter().map(|&index| values[index]).collect();
    let mean = selected.iter().sum::<f64>() / selected.len() as f64;
    json!({
        "mean": mean,
        "p50": quantile(&selected, 0.50),
        "p95": quantile(&selected, 0.95),
        "p99": quantile(&selected, 0.99),
        "min": selected.iter().copied().fold(f64::INFINITY, f64::min),
        "max": selected.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    })
}

fn bool_summary(values: &[bool], indices: &[usize]) -> Value {
    let count = indices.iter().filter(|&&index| values[index]).count();
    let (low, high) = wilson_interval(count, indices.len());
    json!({
        "count": count,
        "rate": count as f64 / indices.len() as f64,
        "wilson_95_low": low,
        "wilson_95_high": high,
    })
}

fn summarize(arrays: &ContactArrays, indices: &[usize]) -> Value {
    if indices.is_empty() {
        return json!({"trajectories": 0});
    }
    let mut object = serde_json::Map::new();
    object.insert("trajectories".to_owned(), json!(indices.len()));
    for (name, values) in [
        ("first_contact_detected", &arrays.first_contact_detected),
        ("landing_success_pass", &arrays.landing_success_pass),
        ("finite_first_contact", &arrays.finite_first_contact),
        ("fuel_gate_pass", &arrays.fuel_gate_pass),
        ("vertical_gate_pass", &arrays.vertical_gate_pass),
    ] {
        object.insert(name.to_owned(), bool_summary(values, indices));
    }
    for (name, values) in [
        ("lateral_error_m", &arrays.lateral_error_m),
        ("vertical_velocity_mps", &arrays.vertical_velocity_mps),
        ("com_sink_speed_mps", &arrays.com_sink_speed_mps),
        (
            "contact_leg_sink_speed_mps",
            &arrays.contact_leg_sink_speed_mps,
        ),
        ("lateral_speed_mps", &arrays.lateral_speed_mps),
        ("total_speed_mps", &arrays.total_speed_mps),
        ("tilt_rad", &arrays.tilt_rad),
        ("fuel_fraction", &arrays.fuel_fraction),
    ] {
        object.insert(name.to_owned(), stats(values, indices));
    }
    let count = |values: &[bool]| indices.iter().filter(|&&index| values[index]).count();
    object.insert(
        "single_success_gate_counts".to_owned(),
        json!({
            "first_contact": count(&arrays.first_contact_detected),
            "lateral_le_5m": indices.iter().filter(|&&i| arrays.lateral_error_m[i] <= 5.0).count(),
            "vertical_first_contact_gate": count(&arrays.vertical_gate_pass),
            "fuel_reserve_gt_2pct": count(&arrays.fuel_gate_pass),
            "landing_success_joint": count(&arrays.landing_success_pass),
        }),
    );
    Value::Object(object)
}

fn rate(values: &[bool], indices: Option<&[usize]>) -> f64 {
    match indices {
        Some(indices) => {
            indices.iter().filter(|&&index| values[index]).count() as f64 / indices.len() as f64
        }
        None => values.iter().filter(|&&value| value).count() as f64 / values.len() as f64,
    }
}

pub fn landing_report(dataset: &Dataset, results: &[RolloutResult]) -> LandingReport {
    assert_eq!(dataset.states.len(), results.len());
    let arrays = contact_arrays(results);
    let n = results.len();
    let all_indices: Vec<usize> = (0..n).collect();
    let success_count = arrays
        .landing_success_pass
        .iter()
        .filter(|&&value| value)
        .count();
    let contact_count = arrays
        .first_contact_detected
        .iter()
        .filter(|&&value| value)
        .count();
    let (wilson_low, wilson_high) = wilson_interval(success_count, n);
    let penalized_sink: Vec<f64> = (0..n)
        .map(|i| {
            if arrays.first_contact_detected[i] {
                arrays.com_sink_speed_mps[i]
            } else {
                NO_CONTACT_SINK_PENALTY_MPS
            }
        })
        .collect();
    let penalized_leg_sink: Vec<f64> = (0..n)
        .map(|i| {
            if arrays.first_contact_detected[i] {
                arrays.contact_leg_sink_speed_mps[i]
            } else {
                NO_CONTACT_SINK_PENALTY_MPS
            }
        })
        .collect();
    let penalized_total_speed: Vec<f64> = (0..n)
        .map(|i| {
            if arrays.first_contact_detected[i] {
                arrays.total_speed_mps[i]
            } else {
                NO_CONTACT_SINK_PENALTY_MPS
            }
        })
        .collect();
    let active_total: u64 = results.iter().map(|r| r.audit.active_steps as u64).sum();
    let sum_steps =
        |get: fn(&RolloutResult) -> u32| -> u64 { results.iter().map(|r| get(r) as u64).sum() };
    let mean_audit = |get: fn(&RolloutResult) -> f32| -> f64 {
        results.iter().map(|r| get(r) as f64).sum::<f64>() / n as f64
    };
    let success_fuel: Vec<f64> = (0..n)
        .filter(|&i| arrays.landing_success_pass[i])
        .map(|i| arrays.fuel_fraction[i])
        .collect();
    let success_min_fuel = success_fuel.iter().copied().reduce(f64::min).unwrap_or(0.0);
    let tilts_deg: Vec<f64> = arrays
        .tilt_rad
        .iter()
        .map(|value| value.to_degrees())
        .collect();
    let roll_abs: Vec<f64> = arrays
        .roll_rate_radps
        .iter()
        .map(|value| value.abs())
        .collect();
    let contact_times: Vec<f64> = results
        .iter()
        .map(|result| {
            if result.first_contact_detected {
                result.first_contact_step as f64 * 0.1
            } else {
                90.0
            }
        })
        .collect();
    let metrics = json!({
        "landing_success_count": success_count,
        "landing_success_rate": success_count as f64 / n as f64,
        "landing_success_wilson_95_low": wilson_low,
        "landing_success_wilson_95_high": wilson_high,
        "first_contact_count": contact_count,
        "first_contact_rate": rate(&arrays.first_contact_detected, None),
        "fuel_gate_pass_rate": rate(&arrays.fuel_gate_pass, None),
        "vertical_first_contact_gate_pass_rate": rate(&arrays.vertical_gate_pass, None),
        "fuel_reserve_mean_fraction": arrays.fuel_fraction.iter().sum::<f64>() / n as f64,
        "fuel_reserve_p05_fraction": quantile(&arrays.fuel_fraction, 0.05),
        "fuel_depletion_rate": arrays.fuel_fraction.iter().filter(|&&value| value <= 0.0).count() as f64 / n as f64,
        "landing_success_min_fuel_reserve_fraction": success_min_fuel,
        "first_contact_sink_speed_mean_mps": penalized_sink.iter().sum::<f64>() / n as f64,
        "first_contact_sink_speed_p50_mps": quantile(&penalized_sink, 0.50),
        "first_contact_sink_speed_p95_mps": quantile(&penalized_sink, 0.95),
        "first_contact_sink_speed_p99_mps": quantile(&penalized_sink, 0.99),
        "first_contact_sink_speed_max_mps": penalized_sink.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        "first_contact_leg_sink_speed_p95_mps": quantile(&penalized_leg_sink, 0.95),
        "first_contact_lateral_error_p95_m": quantile(&arrays.lateral_error_m, 0.95),
        "first_contact_lateral_speed_p95_mps": quantile(&arrays.lateral_speed_mps, 0.95),
        "first_contact_total_speed_p95_mps": quantile(&penalized_total_speed, 0.95),
        "first_contact_tilt_p95_deg": quantile(&tilts_deg, 0.95),
        "first_contact_abs_roll_rate_p95_radps": quantile(&roll_abs, 0.95),
        "first_contact_pitch_yaw_rate_p95_radps": quantile(&arrays.pitch_yaw_rate_radps, 0.95),
        "first_contact_time_p95_s": quantile(&contact_times, 0.95),
        "grid_saturation_rate": sum_steps(|r| r.audit.grid_saturation_steps) as f64 / active_total as f64,
        "gimbal_saturation_rate": sum_steps(|r| r.audit.gimbal_saturation_steps) as f64 / active_total as f64,
        "throttle_saturation_rate": sum_steps(|r| r.audit.throttle_saturation_steps) as f64 / active_total as f64,
        "rcs_roll_cap_rate": sum_steps(|r| r.audit.rcs_roll_cap_steps) as f64 / active_total as f64,
        "forbidden_action_max_abs": results.iter().map(|r| r.audit.max_forbidden_action_abs).fold(0.0_f32, f32::max),
        "plant_lateral_rcs_max_nm": results.iter().map(|r| r.audit.max_plant_lateral_rcs_nm).fold(0.0_f32, f32::max),
        "nonfinite_trajectory_rate": results.iter().filter(|r| r.audit.nonfinite_seen).count() as f64 / n as f64,
        "gimbal_total_variation_mean_rad": mean_audit(|r| r.audit.gimbal_total_variation_rad),
        "grid_total_variation_mean_rad": mean_audit(|r| r.audit.grid_total_variation_rad),
        "throttle_total_variation_mean": mean_audit(|r| r.audit.throttle_total_variation),
        "gear_damping_credit_rate": 0.0,
        "post_contact_scored_steps": 0,
    });

    let mut by_source = serde_json::Map::new();
    let mut source_rates = BTreeMap::new();
    for (source_id, source_name) in dataset.source_names.iter().enumerate() {
        let indices: Vec<usize> = dataset
            .source_ids
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| (value as usize == source_id).then_some(index))
            .collect();
        by_source.insert(source_name.clone(), summarize(&arrays, &indices));
        source_rates.insert(
            format!("{source_name}_landing_success_rate"),
            json!(rate(&arrays.landing_success_pass, Some(&indices))),
        );
    }
    let mut metrics_object = metrics.as_object().expect("metrics object").clone();
    metrics_object.extend(source_rates);

    let edges = [0.0, 450.0, 900.0, 1_200.0, 1_450.0, 1_651.0];
    let initial_radius: Vec<f64> = dataset
        .states
        .iter()
        .map(|s| norm2(s[1] as f64, s[2] as f64))
        .collect();
    let mut by_radius = serde_json::Map::new();
    for pair in edges.windows(2) {
        let lower = pair[0];
        let upper = pair[1];
        let indices: Vec<usize> = initial_radius
            .iter()
            .enumerate()
            .filter_map(|(index, &radius)| (radius >= lower && radius < upper).then_some(index))
            .collect();
        if !indices.is_empty() {
            let count = indices
                .iter()
                .filter(|&&i| arrays.landing_success_pass[i])
                .count();
            by_radius.insert(
                format!("r{}_{}m", lower as i32, upper as i32),
                json!({
                    "trajectories": indices.len(),
                    "landing_success_count": count,
                    "landing_success_rate": count as f64 / indices.len() as f64,
                    "first_contact_rate": rate(&arrays.first_contact_detected, Some(&indices)),
                }),
            );
        }
    }
    let mut max_action = [0.0_f32; 9];
    for result in results {
        for (maximum, &value) in max_action.iter_mut().zip(result.max_abs_action.iter()) {
            *maximum = maximum.max(value);
        }
    }
    let action_names = [
        "gimbal_y",
        "gimbal_z",
        "throttle",
        "rcs_roll",
        "rcs_pitch",
        "rcs_yaw",
        "grid_y",
        "grid_z",
        "grid_roll",
    ];
    let action_map: serde_json::Map<String, Value> = action_names
        .into_iter()
        .zip(max_action)
        .map(|(name, value)| (name.to_owned(), json!(value)))
        .collect();
    let details = json!({
        "overall": summarize(&arrays, &all_indices),
        "by_source": Value::Object(by_source),
        "by_initial_radius": Value::Object(by_radius),
        "max_abs_normalized_action_by_channel": Value::Object(action_map),
    });
    LandingReport {
        metrics: Value::Object(metrics_object),
        details,
        arrays,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rollout::{Audit, RolloutResult};

    fn scored_state(vertical: f32, fuel_fraction: f32) -> RolloutResult {
        let mut state = [0.0; 16];
        state[3] = vertical;
        state[4] = 0.1;
        state[5] = 0.1;
        state[6] = 1.0;
        state[13] = MASS_EMPTY_KG + fuel_fraction * INITIAL_FUEL_KG;
        RolloutResult {
            terminal_state: state,
            terminal_previous_omega: [0.0; 3],
            done: true,
            max_abs_action: [0.0; 9],
            audit: Audit::default(),
            first_contact_state: state,
            first_contact_detected: true,
            first_contact_step: 1,
            first_contact_leg_sink_speed_mps: 0.7,
        }
    }

    #[test]
    fn joint_success_gate_and_strict_fuel_boundary() {
        assert!(contact_arrays(&[scored_state(-0.6, 0.03)]).landing_success_pass[0]);
        assert!(!contact_arrays(&[scored_state(-0.6, 0.02)]).landing_success_pass[0]);
        assert!(!contact_arrays(&[scored_state(0.01, 0.03)]).landing_success_pass[0]);
    }

    #[test]
    fn numpy_linear_quantile_definition() {
        assert_eq!(quantile(&[0.0, 10.0], 0.95), 9.5);
    }
}
