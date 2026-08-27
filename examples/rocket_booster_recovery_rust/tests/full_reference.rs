use std::path::Path;

use rocket_booster_recovery_rust::{
    config::ControllerConfig,
    dataset::{COMPLETE_ROWS, Mode, load},
    metrics::{REFERENCE_SUCCESS_COUNT, landing_report},
    plant::PlantConfig,
    rollout::rollout_all,
};

#[test]
#[ignore = "formal 12,288-trajectory reference; run with cargo test --release -- --ignored"]
fn complete_reference_is_495_of_12288() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let cfg = ControllerConfig::load(&root.join("config/controller.json")).unwrap();
    let plant = PlantConfig::frozen(&cfg);
    let data = load(root, Mode::Complete).unwrap();
    let results = rollout_all(&data.states, &cfg, &plant, 0);
    let report = landing_report(&data, &results);
    assert_eq!(data.states.len(), COMPLETE_ROWS);
    assert_eq!(
        report.metrics["landing_success_count"].as_u64(),
        Some(REFERENCE_SUCCESS_COUNT as u64)
    );
    assert_eq!(
        report.metrics["nominal_unseen_landing_success_rate"].as_f64(),
        Some(0.056_640_625)
    );
    assert_eq!(
        report.metrics["near_ood_landing_success_rate"].as_f64(),
        Some(0.064_208_984_375)
    );
    assert_eq!(
        report.metrics["hard_ood_landing_success_rate"].as_f64(),
        Some(0.0)
    );
}
