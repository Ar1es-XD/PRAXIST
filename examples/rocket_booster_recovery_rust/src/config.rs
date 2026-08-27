use std::{fs, path::Path};

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct ControllerConfig {
    pub version: String,
    pub dt: f32,
    pub gravity: f32,
    pub thrust_max_n: f32,
    pub isp_s: f32,
    pub mass_empty_kg: f32,
    pub initial_fuel_kg: f32,
    pub fuel_reserve_fraction: f32,
    pub vehicle_length_m: f32,
    pub vehicle_radius_m: f32,
    pub engine_arm_m: f32,
    pub gimbal_absolute_rad: f32,
    pub gimbal_normal_rad: f32,
    pub gimbal_rate_rad_per_step: f32,
    pub grid_absolute_rad: f32,
    pub grid_normal_rad: f32,
    pub grid_rate_rad_per_step: f32,
    pub grid_area_total_m2: f32,
    pub grid_control_cl: f32,
    pub grid_station_m: f32,
    pub grid_blend_q_off_pa: f32,
    pub grid_blend_q_full_pa: f32,
    pub grid_torque_fraction: f32,
    pub throttle_normal_min: f32,
    pub throttle_normal_max: f32,
    pub throttle_rate_per_step: f32,
    pub rcs_roll_normal_cap_nm: f32,
    pub rcs_roll_emergency_cap_nm: f32,
    pub rcs_roll_rate_nm_per_step: f32,
    pub rcs_base_torque_nm: f32,
    pub rcs_near_scale: f32,
    pub rcs_far_scale: f32,
    pub rcs_boost_radius_m: f32,
    pub rcs_schedule_width_m: f32,
    pub guidance_period_steps: i32,
    pub guidance_accel_slew_mps3: f32,
    pub tgo_candidates: usize,
    pub preview_nodes: usize,
    pub p0_tgo_min_s: f32,
    pub p0_tgo_max_s: f32,
    pub capture_tgo_min_s: f32,
    pub capture_tgo_max_s: f32,
    pub p0_lateral_horizon_floor_s: f32,
    pub p0_direct_ground_on: f32,
    pub p0_direct_ground_tgo_s: f32,
    pub p0_direct_lateral_tgo_floor_s: f32,
    pub tgo_continuity_weight: f32,
    pub normal_waypoint_height_m: f32,
    pub normal_waypoint_vx_mps: f32,
    pub capture_layer_height_m: f32,
    pub capture_trigger_height_m: f32,
    pub capture_trigger_radius_m: f32,
    pub capture_release_radius_m: f32,
    pub capture_release_vlat_mps: f32,
    pub capture_lateral_horizon_floor_s: f32,
    pub p0_capture_dwell_steps: i32,
    pub p0_tilt_limit_rad: f32,
    pub p1_tilt_limit_rad: f32,
    pub p2_tilt_high_rad: f32,
    pub p2_tilt_low_rad: f32,
    pub attitude_rate_p0_radps: f32,
    pub attitude_rate_p1_radps: f32,
    pub attitude_rate_p2_radps: f32,
    pub phase_dwell_steps: i32,
    pub terminal_entry_height_m: f32,
    pub terminal_entry_radius_m: f32,
    pub terminal_entry_vlat_mps: f32,
    pub terminal_entry_tilt_rad: f32,
    pub terminal_entry_roll_rate_radps: f32,
    pub flare_entry_height_m: f32,
    pub p1_capture_hold_vx_mps: f32,
    pub p1_release_radius_m: f32,
    pub p1_release_vlat_mps: f32,
    pub p1_max_capture_hold_steps: i32,
    pub p1_capture_hold_wn: f32,
    pub p1_capture_hold_zeta: f32,
    pub terminal_v_touch_mps: f32,
    pub terminal_h_leg_m: f32,
    pub terminal_flare_accel_mps2: f32,
    pub terminal_vertical_gain: f32,
    pub terminal_lateral_zeta: f32,
    pub terminal_lateral_wn_high: f32,
    pub terminal_lateral_wn_low: f32,
    pub terminal_lateral_schedule_high_m: f32,
    pub terminal_lateral_schedule_low_m: f32,
    pub attitude_wn_py_p0: f32,
    pub attitude_wn_py_p1: f32,
    pub attitude_wn_py_p2: f32,
    pub attitude_zeta_py_p0: f32,
    pub attitude_zeta_py_p1: f32,
    pub attitude_zeta_py_p2: f32,
    pub attitude_wn_roll_high: f32,
    pub attitude_wn_roll_low: f32,
    pub attitude_zeta_roll: f32,
    pub actuator_margin_fraction: f32,
    pub emergency_brake_efficiency: f32,
    pub emergency_distance_factor: f32,
    pub emergency_height_buffer_m: f32,
    pub body_rho_kgpm3: f32,
    pub body_cd: f32,
    pub body_reference_area_m2: f32,
    pub candidate_fuel_weight: f32,
    pub candidate_time_weight: f32,
    pub candidate_jerk_weight: f32,
    pub candidate_violation_weight: f32,
}

impl ControllerConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let cfg: Self =
            serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            (self.dt - 0.1).abs() <= f32::EPSILON,
            "controller dt must be 0.1 s"
        );
        ensure!(self.mass_empty_kg == 22_200.0, "dry mass must be 22,200 kg");
        ensure!(
            self.initial_fuel_kg == 7_000.0,
            "initial fuel must be 7,000 kg"
        );
        ensure!(
            self.guidance_period_steps > 0,
            "guidance period must be positive"
        );
        ensure!(
            self.tgo_candidates >= 2,
            "at least two tgo candidates are required"
        );
        Ok(())
    }

    #[inline]
    pub fn initial_mass_kg(&self) -> f32 {
        self.mass_empty_kg + self.initial_fuel_kg
    }
}
