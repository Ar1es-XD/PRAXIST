//! The deliberately small API exposed to peer-authored controller candidates.

use std::{collections::BTreeMap, fmt::Debug, ops::Deref};

use serde::de::DeserializeOwned;
use serde_json::Value;

pub use rocket_booster_recovery_rust::math::{
    Vec2, Vec3, clip, controller_quat_to_rot, cross, dot, mat_t_vec, safe_norm, safe_norm2, scale,
    unit,
};
pub use rocket_booster_recovery_rust::plant::State;
pub use serde::Deserialize;

/// Read-only controller parameters exposed without the root crate's filesystem
/// loading API. The evaluator alone owns configuration I/O and deserialization.
#[derive(Clone, Debug, Deserialize)]
#[serde(transparent)]
pub struct ControllerConfig(rocket_booster_recovery_rust::config::ControllerConfig);

impl ControllerConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.0.validate().map_err(|error| error.to_string())
    }

    #[inline]
    pub fn initial_mass_kg(&self) -> f32 {
        self.0.initial_mass_kg()
    }
}

impl Deref for ControllerConfig {
    type Target = rocket_booster_recovery_rust::config::ControllerConfig;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Frozen controller fields plus a small candidate-owned parameter block.
///
/// Candidates can introduce a typed mechanism without copying the roughly one
/// hundred frozen controller fields into `controller.rs`:
///
/// ```ignore
/// #[derive(Clone, Debug, Default, Deserialize)]
/// pub struct Params {
///     #[serde(default)]
///     pub mechanism_gain: f32,
/// }
/// pub type ControllerConfig = VariantConfig<Params>;
/// ```
///
/// The matching JSON is added under the top-level `variant_params` key.  All
/// original controller fields remain top-level and continue to be validated by
/// the frozen configuration implementation.
#[derive(Clone, Debug, Deserialize)]
pub struct VariantConfig<P> {
    #[serde(flatten)]
    base: rocket_booster_recovery_rust::config::ControllerConfig,
    #[serde(default)]
    pub variant_params: P,
}

impl<P> VariantConfig<P> {
    pub fn validate(&self) -> Result<(), String> {
        self.base.validate().map_err(|error| error.to_string())
    }

    #[inline]
    pub fn initial_mass_kg(&self) -> f32 {
        self.base.initial_mass_kg()
    }

    #[inline]
    pub fn base(&self) -> &rocket_booster_recovery_rust::config::ControllerConfig {
        &self.base
    }
}

impl<P> Deref for VariantConfig<P> {
    type Target = rocket_booster_recovery_rust::config::ControllerConfig;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

/// Zero-boilerplate configuration for candidates that need a modest number of
/// scalar or schedule parameters.  The frozen fields remain available through
/// `Deref`; candidate-owned values live in `variant_params` and are accessed by
/// the typed helpers below.
#[derive(Clone, Debug, Deserialize)]
pub struct DynamicControllerConfig {
    #[serde(flatten)]
    base: rocket_booster_recovery_rust::config::ControllerConfig,
    #[serde(default)]
    variant_params: BTreeMap<String, Value>,
}

impl DynamicControllerConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.base.validate().map_err(|error| error.to_string())
    }

    #[inline]
    pub fn initial_mass_kg(&self) -> f32 {
        self.base.initial_mass_kg()
    }

    #[inline]
    pub fn base(&self) -> &rocket_booster_recovery_rust::config::ControllerConfig {
        &self.base
    }

    #[inline]
    pub fn parameter_f32(&self, name: &str, default: f32) -> f32 {
        self.variant_params
            .get(name)
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite())
            .map(|value| value as f32)
            .filter(|value| value.is_finite())
            .unwrap_or(default)
    }

    #[inline]
    pub fn parameter_i32(&self, name: &str, default: i32) -> i32 {
        self.variant_params
            .get(name)
            .and_then(Value::as_i64)
            .and_then(|value| i32::try_from(value).ok())
            .unwrap_or(default)
    }

    #[inline]
    pub fn parameter_bool(&self, name: &str, default: bool) -> bool {
        self.variant_params
            .get(name)
            .and_then(Value::as_bool)
            .unwrap_or(default)
    }

    #[inline]
    pub fn parameter_f32_array(&self, name: &str) -> Option<Vec<f32>> {
        self.variant_params
            .get(name)?
            .as_array()?
            .iter()
            .map(|value| {
                let value = value.as_f64()? as f32;
                value.is_finite().then_some(value)
            })
            .collect()
    }

    #[inline]
    pub fn has_parameter(&self, name: &str) -> bool {
        self.variant_params.contains_key(name)
    }
}

impl Deref for DynamicControllerConfig {
    type Target = rocket_booster_recovery_rust::config::ControllerConfig;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

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

#[derive(Clone, Debug)]
pub struct ControlOutput<M> {
    pub action: [f32; 9],
    pub memory: M,
    pub diagnostic: [f32; 18],
}

/// Adapter implemented by the frozen generated runner around `controller.rs`.
pub trait CandidateController: Send + Sync + 'static {
    type Config: DeserializeOwned + Sync;
    type Memory: Debug;

    fn validate_config(config: &Self::Config) -> Result<(), String>;
    fn init_memory(state: &State, config: &Self::Config) -> Self::Memory;
    fn control_step(
        state: &State,
        memory: &Self::Memory,
        config: &Self::Config,
    ) -> ControlOutput<Self::Memory>;
    fn previous_actuators(memory: &Self::Memory) -> ([f32; 2], [f32; 2], f32);
    fn diagnostic_columns() -> &'static [&'static str; 18];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, Default, Deserialize)]
    struct TypedParameters {
        #[serde(default)]
        mechanism_gain: f32,
    }

    fn baseline_json() -> Value {
        serde_json::from_str(include_str!("../assets/baseline/controller_config.json"))
            .expect("baseline controller JSON")
    }

    #[test]
    fn typed_variant_config_adds_parameters_without_redeclaring_base() {
        let mut raw = baseline_json();
        raw.as_object_mut().unwrap().insert(
            "variant_params".to_owned(),
            serde_json::json!({"mechanism_gain": 1.25}),
        );
        let config: VariantConfig<TypedParameters> = serde_json::from_value(raw).unwrap();
        config.validate().unwrap();
        assert_eq!(config.initial_fuel_kg, 7_000.0);
        assert_eq!(config.variant_params.mechanism_gain, 1.25);
    }

    #[test]
    fn dynamic_variant_config_exposes_typed_parameter_helpers() {
        let mut raw = baseline_json();
        raw.as_object_mut().unwrap().insert(
            "variant_params".to_owned(),
            serde_json::json!({
                "mechanism_gain": 0.75,
                "window_length_steps": 4,
                "enabled": true,
                "schedule_breakpoints": [0.2, 0.5, 0.8]
            }),
        );
        let config: DynamicControllerConfig = serde_json::from_value(raw).unwrap();
        config.validate().unwrap();
        assert_eq!(config.parameter_f32("mechanism_gain", 0.0), 0.75);
        assert_eq!(config.parameter_i32("window_length_steps", 0), 4);
        assert!(config.parameter_bool("enabled", false));
        assert_eq!(
            config.parameter_f32_array("schedule_breakpoints").unwrap(),
            [0.2, 0.5, 0.8]
        );
    }
}
