use std::path::Path;

use rocket_booster_recovery_rust::{
    config::ControllerConfig,
    dataset::{Mode, load},
    plant::PlantConfig,
    rollout::rollout_one,
};

#[test]
fn python_jax_canary_endpoint_is_reproduced() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let cfg = ControllerConfig::load(&root.join("config/controller.json")).unwrap();
    let plant = PlantConfig::frozen(&cfg);
    let data = load(root, Mode::Canary).unwrap();
    assert_eq!(data.source_ids, [0]);
    assert_eq!(data.source_rows, [530]);

    let result = rollout_one(&data.states[0], &cfg, &plant);
    let python_jax = [
        3.175_139,
        -2.539_862,
        -0.814_499_14,
        -41.868,
        -0.006_022_597_7,
        -0.013_420_388,
        0.999_625_9,
        -0.021_883_601,
        -0.006_614_622,
        0.015_016_765,
        -1.144_075e-12,
        0.003_790_099_6,
        -0.002_051_169,
        22_200.0,
        743.619_14,
        1_448.061_9,
    ];
    let max_delta = result
        .terminal_state
        .iter()
        .zip(python_jax)
        .map(|(&rust, python)| (rust - python).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_delta < 3.0e-5, "maximum state delta was {max_delta}");
    assert!(result.first_contact_detected);
    assert_eq!(result.first_contact_step, 744);
    assert!((result.first_contact_leg_sink_speed_mps - 41.856_04).abs() < 1.0e-5);
}
