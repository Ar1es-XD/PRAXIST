#[path = "../assets/baseline/controller.rs"]
mod candidate;

#[test]
fn baseline_candidate_api_and_disabled_channels_are_valid() {
    assert_eq!(
        candidate::DIAGNOSTIC_COLUMNS,
        rocket_booster_recovery_task::candidate_api::DIAGNOSTIC_COLUMNS
    );
    let config_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../config/controller.json");
    let config: candidate::ControllerConfig =
        serde_json::from_slice(&std::fs::read(config_path).unwrap()).unwrap();
    candidate::validate_config(&config).unwrap();
    let mut state = [0.0_f32; 16];
    state[0] = 2_000.0;
    state[3] = -75.0;
    state[6] = 1.0;
    state[13] = 29_200.0;
    state[15] = 500.0;
    let memory = candidate::init_memory(&state, &config);
    let output = candidate::control_step(&state, &memory, &config);
    let previous = candidate::previous_actuators(&output.memory);
    assert!(previous.2.is_finite());
    for index in [4, 5, 8] {
        assert_eq!(output.action[index].to_bits(), 0.0_f32.to_bits());
    }
}

#[test]
fn baseline_complete_evidence_is_durable_but_not_confirmed() {
    let summary_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("assets/baselines/baseline_evaluation_summary.json");
    let summary: serde_json::Value =
        serde_json::from_slice(&std::fs::read(summary_path).unwrap()).unwrap();
    assert_eq!(summary["scored_complete"], serde_json::json!(true));
    assert_eq!(summary["promotion_eligible"], serde_json::json!(true));
    assert_eq!(
        summary["metrics"]["research_independence_attested"],
        serde_json::json!(true)
    );
    assert_eq!(
        summary["metrics"]["suspect_leakage"],
        serde_json::json!(false)
    );
    assert_eq!(
        summary["confirmed_performance_gate_passed"],
        serde_json::json!(false)
    );
    assert_eq!(
        summary["metrics"]["hard_ood_success_nonzero"],
        serde_json::json!(false)
    );
    assert_eq!(
        summary["metrics"]["radius_strata_gate_nonzero"],
        serde_json::json!(false)
    );
}
