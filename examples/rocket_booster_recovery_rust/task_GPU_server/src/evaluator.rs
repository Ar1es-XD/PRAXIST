use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, ValueEnum};
use rocket_booster_recovery_rust::{
    config::ControllerConfig,
    dataset::{self, Dataset, Mode as DatasetMode},
    metrics::{NO_CONTACT_SINK_PENALTY_MPS, SUCCESS_THRESHOLDS, landing_report, quantile},
    plant::{PlantConfig, State},
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::{
    candidate_api::{CandidateController, DIAGNOSTIC_COLUMNS},
    manifest::{
        CandidateAudit, inspect_variant, sha256_file, sha256_named_rust_sources, variant_id_hint,
    },
    rollout::{landing_all, roll_metrics},
};

const PROTOCOL_NAME: &str = "rocket_booster_recovery_first_contact_7000kg_validation_v2";
const PROTOCOL_VERSION: u64 = 2;
const COMPLETE_LANDING_UNITS: usize = 12_288;
const ROLL_UNITS: usize = 1_024;
const COMPLETE_UNITS: usize = COMPLETE_LANDING_UNITS + ROLL_UNITS;
const MASS_EMPTY_KG: f64 = 22_200.0;
const INITIAL_FUEL_KG: f64 = 7_000.0;
const INITIAL_MASS_KG: f64 = MASS_EMPTY_KG + INITIAL_FUEL_KG;
const ROOT_CONFIG_HASH: &str = "9aad17d00c3c29081600081ea0a9bd4a0dbb0a0217ea68fd0c42c3762294f5e8";
const FROZEN_SOURCE_HASHES: [(&str, &str); 4] = [
    (
        "src/plant.rs",
        "5e1dadfe048339b02f0cf4a5a587d57b2ee2e281decc92c923b9f0e0495b2dce",
    ),
    (
        "src/math.rs",
        "5f783c02a7d6a03d6c701e0b849b11f60b64f9a988cc3f83a17b4553f2a67e1c",
    ),
    (
        "src/dataset.rs",
        "cbd194c43da63343897cfa210b9acc5c9503eb4f1cd73e52ce0456730583d000",
    ),
    (
        "src/metrics.rs",
        "c4e668d1280461d3fefb12ecc6e4d71d84f6cecaf72f547ec968afbe2366228d",
    ),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum EvaluationMode {
    Canary,
    Development,
    RollDiagnostic,
    Complete,
}

impl EvaluationMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Canary => "canary",
            Self::Development => "development",
            Self::RollDiagnostic => "roll_diagnostic",
            Self::Complete => "complete",
        }
    }

    fn cli_value(self) -> &'static str {
        match self {
            Self::RollDiagnostic => "roll-diagnostic",
            _ => self.as_str(),
        }
    }

    fn landing(self) -> bool {
        !matches!(self, Self::RollDiagnostic)
    }

    fn roll(self) -> bool {
        matches!(self, Self::RollDiagnostic | Self::Complete)
    }

    fn landing_units(self) -> usize {
        match self {
            Self::Canary => 1,
            Self::Development => 2_048,
            Self::RollDiagnostic => 0,
            Self::Complete => COMPLETE_LANDING_UNITS,
        }
    }

    fn roll_units(self) -> usize {
        if self.roll() { ROLL_UNITS } else { 0 }
    }

    fn effort_ratio(self) -> f64 {
        (self.landing_units() + self.roll_units()) as f64 / COMPLETE_UNITS as f64
    }

    fn coverage_ratio(self) -> f64 {
        match self {
            Self::Complete => 1.0,
            Self::RollDiagnostic => ROLL_UNITS as f64 / COMPLETE_UNITS as f64,
            Self::Canary | Self::Development => 0.0,
        }
    }
}

#[derive(Clone, Debug, Parser)]
#[command(
    about = "Evaluate one deterministic Rocket Booster Recovery controller variant implemented in Rust"
)]
struct PublicArgs {
    #[arg(long)]
    variant_dir: PathBuf,
    #[arg(long, value_enum, default_value_t = EvaluationMode::Development)]
    mode: EvaluationMode,
    #[arg(long, alias = "out-dir")]
    output_dir: PathBuf,
    /// CPU trajectory workers. Zero uses all logical CPUs visible to the process.
    #[arg(long, default_value_t = 16)]
    threads: usize,
    /// Shared Cargo cache used for generated, statically audited candidate runners.
    #[arg(long)]
    compile_cache_dir: Option<PathBuf>,
}

#[derive(Clone, Debug, Parser)]
struct RunnerArgs {
    #[arg(long)]
    variant_dir: PathBuf,
    #[arg(long, value_enum)]
    mode: EvaluationMode,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long)]
    project_root: PathBuf,
    #[arg(long)]
    expected_source_tree_sha256: String,
    #[arg(long)]
    expected_config_sha256: String,
    #[arg(long)]
    expected_manifest_sha256: String,
    #[arg(long, default_value_t = 16)]
    threads: usize,
}

fn absolute(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(env::current_dir()?.join(path))
    }
}

fn task_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn canonical_json(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let sorted: BTreeMap<_, _> = map
                .iter()
                .map(|(key, value)| (key.clone(), canonical_json(value)))
                .collect();
            Value::Object(sorted.into_iter().collect())
        }
        Value::Array(values) => Value::Array(values.iter().map(canonical_json).collect()),
        other => other.clone(),
    }
}

fn canonical_digest(value: &Value) -> Result<String> {
    let bytes = serde_json::to_vec(&canonical_json(value))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    fs::write(path, bytes).with_context(|| format!("write {}", path.display()))
}

fn generated_main() -> &'static str {
    r#"mod candidate;

use rocket_booster_recovery_task::candidate_api::{CandidateController, ControlOutput, State};

struct Adapter;

impl CandidateController for Adapter {
    type Config = candidate::ControllerConfig;
    type Memory = candidate::ControllerMemory;

    fn validate_config(config: &Self::Config) -> Result<(), String> {
        candidate::validate_config(config)
    }

    fn init_memory(state: &State, config: &Self::Config) -> Self::Memory {
        candidate::init_memory(state, config)
    }

    fn control_step(
        state: &State,
        memory: &Self::Memory,
        config: &Self::Config,
    ) -> ControlOutput<Self::Memory> {
        candidate::control_step(state, memory, config)
    }

    fn previous_actuators(memory: &Self::Memory) -> ([f32; 2], [f32; 2], f32) {
        candidate::previous_actuators(memory)
    }

    fn diagnostic_columns() -> &'static [&'static str; 18] {
        &candidate::DIAGNOSTIC_COLUMNS
    }
}

fn main() {
    rocket_booster_recovery_task::run_candidate_from_env::<Adapter>();
}
"#
}

fn stage_candidate_runner(
    variant_dir: &Path,
    audit: &CandidateAudit,
    cache_root: &Path,
) -> Result<PathBuf> {
    let digest = &audit.source_tree_sha256[..20];
    let runner_root = cache_root.join(format!("candidate-v2-{digest}"));
    let source_dir = runner_root.join("src");
    let candidate_dir = source_dir.join("candidate");
    fs::create_dir_all(&source_dir)?;
    fs::create_dir_all(&candidate_dir)?;
    let task_path = task_root();
    let task_path_toml = serde_json::to_string(&task_path.to_string_lossy().as_ref())?;
    let manifest = format!(
        "[workspace]\n\n[package]\nname = \"rocket-booster-recovery-rust-candidate-{digest}\"\nversion = \"0.0.0\"\nedition = \"2024\"\npublish = false\n\n[dependencies]\nserde = {{ version = \"1.0\", features = [\"derive\"] }}\nrocket-booster-recovery-task = {{ path = {task_path_toml} }}\n\n[profile.release]\nopt-level = 3\ncodegen-units = 1\n"
    );
    fs::write(runner_root.join("Cargo.toml"), manifest)?;
    fs::write(source_dir.join("main.rs"), generated_main())?;

    let (_, current_audit) = inspect_variant(variant_dir)?;
    ensure!(
        current_audit.source_tree_sha256 == audit.source_tree_sha256
            && current_audit.config_sha256 == audit.config_sha256
            && current_audit.manifest_sha256 == audit.manifest_sha256,
        "candidate changed after static inspection"
    );
    let mut sources = Vec::with_capacity(audit.rust_files.len());
    for relative in &audit.rust_files {
        sources.push((relative.clone(), fs::read(variant_dir.join(relative))?));
    }
    ensure!(
        sha256_named_rust_sources(&sources) == audit.source_tree_sha256,
        "candidate Rust source tree changed while staging"
    );
    for (relative, bytes) in sources {
        let destination = if relative == "controller.rs" {
            candidate_dir.join("mod.rs")
        } else {
            candidate_dir.join(&relative)
        };
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(destination, bytes)?;
    }
    Ok(runner_root.join("Cargo.toml"))
}

pub fn public_main() -> Result<()> {
    let mut args = PublicArgs::parse();
    args.variant_dir = absolute(&args.variant_dir)?
        .canonicalize()
        .with_context(|| format!("resolve {}", args.variant_dir.display()))?;
    args.output_dir = absolute(&args.output_dir)?;
    fs::create_dir_all(&args.output_dir)?;
    let (_, audit) = match inspect_variant(&args.variant_dir) {
        Ok(value) => value,
        Err(error) => {
            let reason = error.to_string();
            let suspect_leakage = reason.contains("research independence")
                || reason.contains("research_independence");
            write_failure_summary(
                &args.output_dir,
                &variant_id_hint(&args.variant_dir),
                args.mode,
                &reason,
                suspect_leakage,
            )?;
            return Err(error);
        }
    };
    let cache_root = match args.compile_cache_dir {
        Some(path) => absolute(&path)?,
        None => task_root().join("scratch/candidate_builds"),
    };
    let manifest_path = stage_candidate_runner(&args.variant_dir, &audit, &cache_root)?;
    let target_dir = task_root().join("scratch/cargo-target");
    fs::create_dir_all(&target_dir)?;
    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let status = Command::new(cargo)
        .args([
            "run",
            "--quiet",
            "--release",
            "--offline",
            "--manifest-path",
        ])
        .arg(&manifest_path)
        .arg("--")
        .arg("--variant-dir")
        .arg(&args.variant_dir)
        .arg("--mode")
        .arg(args.mode.cli_value())
        .arg("--output-dir")
        .arg(&args.output_dir)
        .arg("--project-root")
        .arg(
            task_root()
                .parent()
                .context("task root has no project parent")?,
        )
        .arg("--expected-source-tree-sha256")
        .arg(&audit.source_tree_sha256)
        .arg("--expected-config-sha256")
        .arg(&audit.config_sha256)
        .arg("--expected-manifest-sha256")
        .arg(&audit.manifest_sha256)
        .arg("--threads")
        .arg(args.threads.to_string())
        .env("CARGO_TARGET_DIR", &target_dir)
        .env("CARGO_NET_OFFLINE", "true")
        .current_dir(task_root())
        .status()
        .context("launch generated Rust candidate runner")?;
    if !status.success() {
        let summary = args.output_dir.join("evaluation_summary.json");
        if !summary.is_file() {
            write_failure_summary(
                &args.output_dir,
                &variant_id_hint(&args.variant_dir),
                args.mode,
                &format!("candidate compile/evaluation process exited with {status}"),
                false,
            )?;
        }
        bail!("candidate evaluation failed with {status}");
    }
    Ok(())
}

fn attest_frozen_assets(project_root: &Path) -> Result<BTreeMap<String, String>> {
    let mut observed: BTreeMap<String, String> =
        dataset::attest_assets(project_root)?.into_iter().collect();
    let config = project_root.join("config/controller.json");
    let config_hash = sha256_file(&config)?;
    ensure!(
        config_hash == ROOT_CONFIG_HASH,
        "frozen root controller config hash mismatch"
    );
    observed.insert("config/controller.json".to_owned(), config_hash);
    for (relative, expected) in FROZEN_SOURCE_HASHES {
        let actual = sha256_file(&project_root.join(relative))?;
        ensure!(
            actual == expected,
            "frozen source hash mismatch for {relative}: expected {expected}, got {actual}"
        );
        observed.insert(relative.to_owned(), actual);
    }
    Ok(observed)
}

fn load_candidate_config<C: CandidateController>(path: &Path) -> Result<(C::Config, Value)> {
    let raw: Value = serde_json::from_slice(&fs::read(path)?)
        .with_context(|| format!("parse {}", path.display()))?;
    let object = raw
        .as_object()
        .context("controller config must be a JSON object")?;
    let number = |name: &str| -> Result<f64> {
        object
            .get(name)
            .and_then(Value::as_f64)
            .with_context(|| format!("controller config requires finite number `{name}`"))
    };
    ensure!(
        (number("dt")? - 0.1).abs() <= 1.0e-12,
        "controller dt must remain 0.1 s"
    );
    ensure!(
        (number("mass_empty_kg")? - MASS_EMPTY_KG).abs() <= 1.0e-9,
        "controller mass_empty_kg must remain 22200 kg"
    );
    ensure!(
        (number("initial_fuel_kg")? - INITIAL_FUEL_KG).abs() <= 1.0e-9,
        "controller initial_fuel_kg must remain 7000 kg"
    );
    let resolved: C::Config = serde_json::from_value(raw.clone())
        .context("deserialize candidate controller configuration")?;
    C::validate_config(&resolved).map_err(anyhow::Error::msg)?;
    Ok((resolved, raw))
}

fn default_metrics() -> Map<String, Value> {
    let mut value = json!({
        "landing_success_count": 0,
        "landing_success_rate": 0.0,
        "landing_success_wilson_95_low": 0.0,
        "landing_success_wilson_95_high": 0.0,
        "first_contact_count": 0,
        "first_contact_rate": 0.0,
        "fuel_gate_pass_rate": 0.0,
        "vertical_first_contact_gate_pass_rate": 0.0,
        "hard_ood_landing_success_rate": 0.0,
        "hard_ood_first_contact_sink_speed_p95_mps": NO_CONTACT_SINK_PENALTY_MPS,
        "worst_radius_bin_landing_success_rate": 0.0,
        "fuel_reserve_mean_fraction": 0.0,
        "fuel_reserve_p05_fraction": 0.0,
        "fuel_depletion_rate": 0.0,
        "fuel_gate_shortfall_rate": 0.0,
        "fuel_reserve_margin_above_2pct_p05_fraction": 0.0,
        "landing_success_min_fuel_reserve_fraction": 0.0,
        "first_contact_sink_speed_mean_mps": NO_CONTACT_SINK_PENALTY_MPS,
        "first_contact_sink_speed_p50_mps": NO_CONTACT_SINK_PENALTY_MPS,
        "first_contact_sink_speed_p95_mps": NO_CONTACT_SINK_PENALTY_MPS,
        "first_contact_sink_speed_p99_mps": NO_CONTACT_SINK_PENALTY_MPS,
        "first_contact_sink_speed_max_mps": NO_CONTACT_SINK_PENALTY_MPS,
        "first_contact_leg_sink_speed_mean_mps": NO_CONTACT_SINK_PENALTY_MPS,
        "first_contact_leg_sink_speed_p95_mps": NO_CONTACT_SINK_PENALTY_MPS,
        "first_contact_total_speed_p95_mps": NO_CONTACT_SINK_PENALTY_MPS,
        "first_contact_lateral_speed_p95_mps": 0.0,
        "first_contact_lateral_error_p95_m": 0.0,
        "first_contact_tilt_p95_deg": 0.0,
        "first_contact_abs_roll_rate_p95_radps": 0.0,
        "first_contact_pitch_yaw_rate_p95_radps": 0.0,
        "first_contact_time_p95_s": 90.0,
        "unsafe_first_contact_rate": 0.0,
        "gear_damping_credit_rate": 0.0,
        "post_contact_scored_steps": 0,
        "grid_saturation_rate": 0.0,
        "gimbal_saturation_rate": 0.0,
        "throttle_saturation_rate": 0.0,
        "rcs_roll_cap_rate": 0.0,
        "forbidden_action_max_abs": 0.0,
        "plant_lateral_rcs_max_nm": 0.0,
        "nonfinite_trajectory_rate": 0.0,
        "gimbal_total_variation_mean_rad": 0.0,
        "grid_total_variation_mean_rad": 0.0,
        "throttle_total_variation_mean": 0.0,
        "roll_stable_rate": 0.0,
        "roll_settling_time_p95_s": 0.0,
        "roll_peak_rate_p95_radps": 0.0,
        "roll_pitch_yaw_coupling_p95_radps": 0.0,
        "roll_rcs_switches_mean": 0.0,
        "roll_rcs_total_variation_mean_nm": 0.0,
        "roll_forbidden_action_max_abs": 0.0,
        "roll_nonfinite_rate": 0.0
    });
    value.as_object_mut().expect("literal object").clone()
}

fn merge_metrics(target: &mut Map<String, Value>, source: &Value) -> Result<()> {
    let object = source
        .as_object()
        .context("metric payload must be an object")?;
    target.extend(
        object
            .iter()
            .map(|(key, value)| (key.clone(), value.clone())),
    );
    Ok(())
}

fn landing_metric_extensions(
    dataset: &Dataset,
    results: &[rocket_booster_recovery_rust::rollout::RolloutResult],
    report: &rocket_booster_recovery_rust::metrics::LandingReport,
    metrics: &mut Map<String, Value>,
) {
    let arrays = &report.arrays;
    let n = results.len();
    let unsafe_contacts = (0..n)
        .filter(|&index| arrays.first_contact_detected[index] && !arrays.vertical_gate_pass[index])
        .count();
    let margins: Vec<f64> = arrays
        .fuel_fraction
        .iter()
        .map(|value| (value - 0.02).max(0.0))
        .collect();
    let penalized_leg: Vec<f64> = (0..n)
        .map(|index| {
            if arrays.first_contact_detected[index] {
                arrays.contact_leg_sink_speed_mps[index]
            } else {
                NO_CONTACT_SINK_PENALTY_MPS
            }
        })
        .collect();
    metrics.insert(
        "unsafe_first_contact_rate".to_owned(),
        json!(unsafe_contacts as f64 / n as f64),
    );
    metrics.insert(
        "fuel_gate_shortfall_rate".to_owned(),
        json!(1.0 - arrays.fuel_gate_pass.iter().filter(|&&value| value).count() as f64 / n as f64),
    );
    metrics.insert(
        "fuel_reserve_margin_above_2pct_p05_fraction".to_owned(),
        json!(quantile(&margins, 0.05)),
    );
    metrics.insert(
        "first_contact_leg_sink_speed_mean_mps".to_owned(),
        json!(penalized_leg.iter().sum::<f64>() / n as f64),
    );

    for (source_id, name) in dataset.source_names.iter().enumerate() {
        let indices: Vec<usize> = dataset
            .source_ids
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| (value as usize == source_id).then_some(index))
            .collect();
        if indices.is_empty() {
            continue;
        }
        let contact_count = indices
            .iter()
            .filter(|&&index| arrays.first_contact_detected[index])
            .count();
        let source_sink: Vec<f64> = indices
            .iter()
            .map(|&index| {
                if arrays.first_contact_detected[index] {
                    arrays.com_sink_speed_mps[index]
                } else {
                    NO_CONTACT_SINK_PENALTY_MPS
                }
            })
            .collect();
        metrics.insert(
            format!("{name}_first_contact_rate"),
            json!(contact_count as f64 / indices.len() as f64),
        );
        metrics.insert(
            format!("{name}_first_contact_sink_speed_p95_mps"),
            json!(quantile(&source_sink, 0.95)),
        );
    }

    let radius_rates: Vec<f64> = report.details["by_initial_radius"]
        .as_object()
        .into_iter()
        .flat_map(|object| object.values())
        .filter_map(|value| value["landing_success_rate"].as_f64())
        .collect();
    metrics.insert(
        "worst_radius_bin_landing_success_rate".to_owned(),
        json!(radius_rates.into_iter().reduce(f64::min).unwrap_or(0.0)),
    );
}

fn roll_disturbance_states(dataset: &Dataset) -> Result<Vec<State>> {
    ensure!(
        dataset.states.len() >= 128,
        "complete bank needs at least 128 roll base states"
    );
    let patterns = [
        (-std::f64::consts::PI, 0.80_f32),
        (-3.0 * std::f64::consts::PI / 4.0, 0.60),
        (-std::f64::consts::PI / 2.0, 0.40),
        (-std::f64::consts::PI / 4.0, 0.20),
        (std::f64::consts::PI / 4.0, -0.20),
        (std::f64::consts::PI / 2.0, -0.40),
        (3.0 * std::f64::consts::PI / 4.0, -0.60),
        (std::f64::consts::PI, -0.80),
    ];
    let mut states = Vec::with_capacity(ROLL_UNITS);
    for (angle, roll_rate) in patterns {
        let roll = [(angle / 2.0).cos(), (angle / 2.0).sin(), 0.0, 0.0];
        for base in &dataset.states[..128] {
            let base_q = [
                base[6] as f64,
                base[7] as f64,
                base[8] as f64,
                base[9] as f64,
            ];
            let mut q = [
                base_q[0] * roll[0]
                    - base_q[1] * roll[1]
                    - base_q[2] * roll[2]
                    - base_q[3] * roll[3],
                base_q[0] * roll[1] + base_q[1] * roll[0] + base_q[2] * roll[3]
                    - base_q[3] * roll[2],
                base_q[0] * roll[2] - base_q[1] * roll[3]
                    + base_q[2] * roll[0]
                    + base_q[3] * roll[1],
                base_q[0] * roll[3] + base_q[1] * roll[2] - base_q[2] * roll[1]
                    + base_q[3] * roll[0],
            ];
            let norm = (q.iter().map(|value| value * value).sum::<f64>())
                .sqrt()
                .max(1.0e-12);
            q.iter_mut().for_each(|value| *value /= norm);
            let mut state = *base;
            for index in 0..4 {
                state[6 + index] = q[index] as f32;
            }
            state[10] = roll_rate;
            state[11] = 0.0;
            state[12] = 0.0;
            states.push(state);
        }
    }
    ensure!(states.len() == ROLL_UNITS, "roll bank size mismatch");
    Ok(states)
}

fn producer_identity() -> Map<String, Value> {
    let peer_values: std::collections::BTreeSet<String> = ["PRAXIST_PEER_ID", "PEER_ID"]
        .into_iter()
        .filter_map(|key| env::var(key).ok())
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .collect();
    if peer_values.len() != 1 {
        return Map::new();
    }
    let peer_id = peer_values.iter().next().expect("one peer");
    let Some(rest) = peer_id.strip_prefix("gen") else {
        return Map::new();
    };
    let Some((generation, peer)) = rest.split_once("_peer") else {
        return Map::new();
    };
    if peer.is_empty() || !peer.bytes().all(|byte| byte.is_ascii_digit()) {
        return Map::new();
    }
    let Ok(generation_id) = generation.parse::<u64>() else {
        return Map::new();
    };
    let declared: std::collections::BTreeSet<u64> =
        ["PRAXIST_LOGICAL_GENERATION_ID", "GENERATION_ID"]
            .into_iter()
            .filter_map(|key| env::var(key).ok())
            .filter_map(|value| value.trim().parse::<u64>().ok())
            .collect();
    if !declared.is_empty() && declared != [generation_id].into_iter().collect() {
        return Map::new();
    }
    Map::from_iter([
        ("peer_id".to_owned(), json!(peer_id)),
        ("generation_id".to_owned(), json!(generation_id)),
        ("source_generation_id".to_owned(), json!(generation_id)),
    ])
}

fn cpu_model() -> Option<String> {
    let text = fs::read_to_string("/proc/cpuinfo").ok()?;
    text.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        (key.trim() == "model name").then(|| value.trim().to_owned())
    })
}

fn proc_status_kb(name: &str) -> Option<u64> {
    let text = fs::read_to_string("/proc/self/status").ok()?;
    text.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        (key == name)
            .then(|| value.split_whitespace().next()?.parse().ok())
            .flatten()
    })
}

fn metric_f64(metrics: &Map<String, Value>, name: &str) -> f64 {
    metrics
        .get(name)
        .and_then(Value::as_f64)
        .unwrap_or(f64::NAN)
}

fn source_rows_digest(dataset: &Dataset) -> Result<String> {
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(
        &dataset.source_rows,
    )?)))
}

fn evaluate<C: CandidateController>(args: &RunnerArgs) -> Result<Value> {
    let started = Instant::now();
    let project_root = absolute(&args.project_root)?
        .canonicalize()
        .context("resolve Rust project root")?;
    let variant_dir = absolute(&args.variant_dir)?
        .canonicalize()
        .context("resolve candidate variant directory")?;
    let output_dir = absolute(&args.output_dir)?;
    fs::create_dir_all(&output_dir)?;
    let (manifest, candidate_audit) = inspect_variant(&variant_dir)?;
    ensure!(
        candidate_audit.source_tree_sha256 == args.expected_source_tree_sha256,
        "compiled Rust source tree differs from the inspected candidate"
    );
    ensure!(
        candidate_audit.config_sha256 == args.expected_config_sha256,
        "candidate configuration differs from the inspected candidate"
    );
    ensure!(
        candidate_audit.manifest_sha256 == args.expected_manifest_sha256,
        "candidate manifest differs from the inspected candidate"
    );
    ensure!(
        C::diagnostic_columns() == &DIAGNOSTIC_COLUMNS,
        "DIAGNOSTIC_COLUMNS layout differs from the frozen interface"
    );
    let frozen_hashes = attest_frozen_assets(&project_root)?;
    let (controller_config, effective_config) =
        load_candidate_config::<C>(&variant_dir.join("controller_config.json"))?;
    let fixed_controller = ControllerConfig::load(&project_root.join("config/controller.json"))?;
    let plant_cfg = PlantConfig::frozen(&fixed_controller);

    let mut metrics = default_metrics();
    let mut details = Map::new();
    let mut stage_seconds = Vec::new();
    let mut completed_units = 0_usize;
    let mut complete_dataset: Option<Dataset> = None;
    let mut selected_rows_digest: Option<String> = None;

    if args.mode.landing() {
        let data = match args.mode {
            EvaluationMode::Canary => dataset::load(&project_root, DatasetMode::Canary)?,
            EvaluationMode::Development => dataset::load(&project_root, DatasetMode::Development)?,
            EvaluationMode::Complete => dataset::load(&project_root, DatasetMode::Complete)?,
            EvaluationMode::RollDiagnostic => unreachable!(),
        };
        let stage_started = Instant::now();
        let results = landing_all::<C>(&data.states, &controller_config, &plant_cfg, args.threads);
        let report = landing_report(&data, &results);
        stage_seconds.push(stage_started.elapsed().as_secs_f64());
        merge_metrics(&mut metrics, &report.metrics)?;
        landing_metric_extensions(&data, &results, &report, &mut metrics);
        let digest = source_rows_digest(&data)?;
        selected_rows_digest = Some(digest.clone());
        let mut landing_detail = report.details;
        if let Some(object) = landing_detail.as_object_mut() {
            object.insert("source_rows_sha256".to_owned(), json!(digest));
            object.insert(
                "endpoint_previous_omega_available".to_owned(),
                json!(results.iter().all(|result| {
                    result
                        .terminal_previous_omega
                        .iter()
                        .all(|value| value.is_finite())
                })),
            );
            object.insert(
                "success_definition".to_owned(),
                json!({
                    "name": "landing_success",
                    "only_success_standard": true,
                    "endpoint": "interpolated first landing-leg contact before spring/damper response",
                    "thresholds": SUCCESS_THRESHOLDS.as_json(),
                }),
            );
            object.insert(
                "anti_damping_protocol".to_owned(),
                json!("Landing rollout terminates for scoring at first leg contact; post-contact spring/damper states contribute zero scored steps and zero success credit."),
            );
            object.insert(
                "contact_observability_limit".to_owned(),
                json!("First leg contact is observable from frozen gear geometry. Post-contact dwell, bounce height, leg loads, slip, and overturn remain unscored."),
            );
        }
        details.insert("landing".to_owned(), landing_detail);
        completed_units += data.states.len();
        if args.mode == EvaluationMode::Complete {
            complete_dataset = Some(data);
        }
    }

    if args.mode.roll() {
        let data = match complete_dataset.as_ref() {
            Some(data) => data,
            None => {
                complete_dataset = Some(dataset::load(&project_root, DatasetMode::Complete)?);
                complete_dataset.as_ref().expect("just inserted")
            }
        };
        if selected_rows_digest.is_none() {
            selected_rows_digest = Some(source_rows_digest(data)?);
        }
        let states = roll_disturbance_states(data)?;
        let stage_started = Instant::now();
        let (roll_values, roll_detail) =
            roll_metrics::<C>(&states, &controller_config, &plant_cfg, args.threads);
        stage_seconds.push(stage_started.elapsed().as_secs_f64());
        merge_metrics(&mut metrics, &roll_values)?;
        details.insert("roll".to_owned(), roll_detail);
        completed_units += states.len();
    }

    let expected_units = args.mode.landing_units() + args.mode.roll_units();
    let contract_lock_passed = metric_f64(&metrics, "forbidden_action_max_abs") == 0.0
        && metric_f64(&metrics, "plant_lateral_rcs_max_nm") == 0.0
        && metric_f64(&metrics, "roll_forbidden_action_max_abs") == 0.0
        && metric_f64(&metrics, "nonfinite_trajectory_rate") == 0.0
        && metric_f64(&metrics, "roll_nonfinite_rate") == 0.0
        && metric_f64(&metrics, "gear_damping_credit_rate") == 0.0
        && metric_f64(&metrics, "post_contact_scored_steps") == 0.0;
    let complete_budget_reached = completed_units == expected_units;
    let research_independence_attested = candidate_audit.research_independence_attested;
    let suspect_leakage = !research_independence_attested;
    let protocol_integrity_passed =
        contract_lock_passed && complete_budget_reached && research_independence_attested;
    let scored_complete = args.mode == EvaluationMode::Complete && protocol_integrity_passed;
    let landing_success_nonzero = metric_f64(&metrics, "landing_success_rate") > 0.0;
    let hard_ood_nonzero = metric_f64(&metrics, "hard_ood_landing_success_rate") > 0.0;
    let radius_nonzero = metric_f64(&metrics, "worst_radius_bin_landing_success_rate") > 0.0;
    let evidence_eligibility = evidence_eligibility(
        scored_complete,
        protocol_integrity_passed,
        contract_lock_passed,
        landing_success_nonzero,
        hard_ood_nonzero,
        radius_nonzero,
    );
    let promotion_eligible = evidence_eligibility.promotion_eligible;
    let confirmed_performance_gate_passed = evidence_eligibility.confirmed_performance_gate_passed;
    let source_lane = if scored_complete {
        "performance"
    } else if args.mode == EvaluationMode::Development && protocol_integrity_passed {
        "task_candidate"
    } else {
        "diagnostic"
    };
    let elapsed = started.elapsed().as_secs_f64();
    let effort_ratio = args.mode.effort_ratio();
    let coverage_ratio = args.mode.coverage_ratio();
    metrics.extend(Map::from_iter([
        ("effort_ratio".to_owned(), json!(effort_ratio)),
        ("coverage_ratio".to_owned(), json!(coverage_ratio)),
        (
            "evaluation_units_completed".to_owned(),
            json!(completed_units),
        ),
        (
            "evaluation_units_required".to_owned(),
            json!(COMPLETE_UNITS),
        ),
        ("mode_units_expected".to_owned(), json!(expected_units)),
        ("scored_complete".to_owned(), json!(scored_complete)),
        ("complete_eval".to_owned(), json!(scored_complete)),
        (
            "protocol_integrity_passed".to_owned(),
            json!(protocol_integrity_passed),
        ),
        (
            "protocol_integrity_failed".to_owned(),
            json!(!protocol_integrity_passed),
        ),
        (
            "contract_lock_passed".to_owned(),
            json!(contract_lock_passed),
        ),
        (
            "research_independence_attested".to_owned(),
            json!(research_independence_attested),
        ),
        (
            "confirmed_performance_gate_passed".to_owned(),
            json!(confirmed_performance_gate_passed),
        ),
        (
            "landing_success_nonzero".to_owned(),
            json!(landing_success_nonzero),
        ),
        (
            "hard_ood_success_nonzero".to_owned(),
            json!(hard_ood_nonzero),
        ),
        (
            "radius_strata_gate_nonzero".to_owned(),
            json!(radius_nonzero),
        ),
        ("promotion_eligible".to_owned(), json!(promotion_eligible)),
        ("parent_authorized".to_owned(), json!(scored_complete)),
        ("close_eligible".to_owned(), json!(scored_complete)),
        (
            "partial".to_owned(),
            json!(args.mode != EvaluationMode::Complete),
        ),
        (
            "is_smoke_eval".to_owned(),
            json!(args.mode == EvaluationMode::Canary),
        ),
        (
            "validation_only".to_owned(),
            json!(args.mode == EvaluationMode::RollDiagnostic),
        ),
        (
            "scout_only".to_owned(),
            json!(args.mode == EvaluationMode::Canary),
        ),
        ("suspect_protocol".to_owned(), json!(false)),
        ("suspect_leakage".to_owned(), json!(suspect_leakage)),
        ("late_after_generation_boundary".to_owned(), json!(false)),
        ("evaluator_wall_seconds".to_owned(), json!(elapsed)),
    ]));
    let effective_digest = canonical_digest(&effective_config)?;
    let output_summary = output_dir.join("evaluation_summary.json");
    let mut source_bank_hashes = Map::new();
    for (name, _, hash) in dataset::SOURCE_BANKS {
        source_bank_hashes.insert(name.to_owned(), json!(hash));
    }
    let unix_time_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let candidate_audit_value = serde_json::to_value(&candidate_audit)?;
    let research_independence_value = serde_json::to_value(&manifest.research_independence)?;
    let manifest_dimensions = serde_json::to_value(&manifest.design_dimensions)?;
    let mut summary = json!({
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "variant_id": manifest.variant_id,
        "variant_name": manifest.display_name.as_deref().unwrap_or(&manifest.variant_id),
        "success_definition": {
            "name": "landing_success",
            "only_success_standard": true,
            "endpoint": "interpolated_first_landing_leg_contact",
            "post_contact_damping_credit": false,
            "thresholds": SUCCESS_THRESHOLDS.as_json(),
        },
        "metrics": Value::Object(metrics),
        "frontier_lane": source_lane,
        "promotion_lane": source_lane,
        "evidence_stage": args.mode.as_str(),
        "eval_stage": args.mode.as_str(),
        "tier": args.mode.as_str(),
        "result_status": if scored_complete { "scored_complete" } else { args.mode.as_str() },
        "scored_complete": scored_complete,
        "complete_eval": scored_complete,
        "promotion_eligible": promotion_eligible,
        "confirmed_performance_gate_passed": confirmed_performance_gate_passed,
        "effort_ratio": effort_ratio,
        "coverage_ratio": coverage_ratio,
        "actual_evaluation_units": completed_units,
        "reference_evaluation_units": COMPLETE_UNITS,
        "evaluation_units_completed": completed_units,
        "evaluation_units_required": COMPLETE_UNITS,
        "effective_config": effective_config,
        "effective_config_complete": true,
        "effective_config_digest": effective_digest,
        "effective_config_schema": "serde_json::from_value(controller_config.json):rust:v2_extensible",
        "design_dimensions": manifest_dimensions,
        "changed_modules": manifest.changed_modules,
        "method_class": manifest.method_class,
        "research_independence": research_independence_value,
        "research_independence_attested": research_independence_attested,
        "extra": {
            "frontier_lane": source_lane,
            "promotion_lane": source_lane,
            "evidence_stage": args.mode.as_str(),
            "evaluation_summary": output_summary,
            "design_dimensions": manifest.design_dimensions,
            "effective_config_digest": effective_digest,
        },
        "protocol_integrity": {
            "passed": protocol_integrity_passed,
            "contract_lock_passed": contract_lock_passed,
            "frozen_assets_attested": true,
            "outcome_dependent_selection": false,
            "forbidden_learning_methods_detected": false,
            "research_independence_attested": research_independence_attested,
            "suspect_leakage": suspect_leakage,
            "candidate_static_scan": candidate_audit_value,
            "frozen_audit_limits": {
                "dt": 0.1,
                "gimbal_normal_rad": 0.075,
                "grid_normal_rad": 0.25,
                "throttle_normal_min": 0.02,
                "throttle_normal_max": 0.98,
                "rcs_roll_normal_cap_nm": 30000.0,
                "rcs_roll_emergency_cap_nm": 60000.0,
            },
            "single_success_standard": true,
            "post_contact_damping_credit_forbidden": true,
        },
        "dataset": {
            "selection_seed_near_hard": 20260821,
            "selection_seed_nominal": 20260822,
            "source_rows_sha256": selected_rows_digest,
            "complete_landing_units": COMPLETE_LANDING_UNITS,
            "roll_units": ROLL_UNITS,
            "complete_units": COMPLETE_UNITS,
            "mode_units": expected_units,
            "source_bank_hashes": source_bank_hashes,
            "historical_formal_overlap": false,
            "initial_mass_kg": INITIAL_MASS_KG,
            "initial_fuel_kg": INITIAL_FUEL_KG,
            "mass_empty_kg": MASS_EMPTY_KG,
        },
        "plant": {
            "integrator": "rk4",
            "substeps": 1,
            "dt": plant_cfg.dt,
            "max_steps": plant_cfg.max_steps,
            "protocol_initial_mass_override_kg": INITIAL_MASS_KG,
            "protocol_fuel_scale": 1.0,
            "landing_scoring_endpoint": "interpolated_first_leg_contact",
            "post_contact_scored_steps": 0,
            "implementation": "Rust f32 implementation of the frozen C05 6DoF equations",
            "attested_hashes": frozen_hashes,
        },
        "execution": {
            "unix_time_seconds": unix_time_seconds,
            "elapsed_seconds": elapsed,
            "stage_seconds": stage_seconds,
            "runtime_language": "Rust",
            "harness_crate_version": env!("CARGO_PKG_VERSION"),
            "target_arch": std::env::consts::ARCH,
            "target_os": std::env::consts::OS,
            "backend": "cpu",
            "gpu_required": false,
            "trajectory_parallelism": "Rayon indexed parallel iterator",
            "threads": if args.threads == 0 { rayon::current_num_threads() } else { args.threads },
            "available_parallelism": std::thread::available_parallelism().map(usize::from).unwrap_or(1),
            "cpu_model": cpu_model(),
            "resident_memory_kb": proc_status_kb("VmRSS"),
            "peak_resident_memory_kb": proc_status_kb("VmHWM"),
            "scheduler_managed": env::var_os("PRAXIST_PEER_ID").is_some(),
            "resource_profile": env::var("PRAXIST_RESOURCE_PROFILE").unwrap_or_else(|_| "cpu_evaluation".to_owned()),
        },
        "details": Value::Object(details),
    });
    let identity = producer_identity();
    if let Some(object) = summary.as_object_mut() {
        object.extend(identity.clone());
        if let Some(extra) = object.get_mut("extra").and_then(Value::as_object_mut) {
            extra.extend(identity);
        }
    }
    write_json(&output_summary, &summary)?;
    write_json(
        &output_dir.join("resolved_effective_config.json"),
        &json!({
            "variant_id": summary["variant_id"],
            "effective_config": summary["effective_config"],
            "effective_config_complete": true,
            "effective_config_digest": summary["effective_config_digest"],
        }),
    )?;
    Ok(summary)
}

fn add_failure_metric_contract(
    metrics: &mut Map<String, Value>,
    mode: EvaluationMode,
    suspect_leakage: bool,
) {
    metrics.extend(Map::from_iter([
        ("effort_ratio".to_owned(), json!(0.0)),
        ("coverage_ratio".to_owned(), json!(0.0)),
        ("evaluation_units_completed".to_owned(), json!(0)),
        (
            "evaluation_units_required".to_owned(),
            json!(COMPLETE_UNITS),
        ),
        ("scored_complete".to_owned(), json!(false)),
        ("complete_eval".to_owned(), json!(false)),
        ("protocol_integrity_passed".to_owned(), json!(false)),
        ("protocol_integrity_failed".to_owned(), json!(true)),
        ("contract_lock_passed".to_owned(), json!(false)),
        ("research_independence_attested".to_owned(), json!(false)),
        ("confirmed_performance_gate_passed".to_owned(), json!(false)),
        ("promotion_eligible".to_owned(), json!(false)),
        ("parent_authorized".to_owned(), json!(false)),
        ("close_eligible".to_owned(), json!(false)),
        ("partial".to_owned(), json!(true)),
        (
            "is_smoke_eval".to_owned(),
            json!(mode == EvaluationMode::Canary),
        ),
        (
            "validation_only".to_owned(),
            json!(mode == EvaluationMode::RollDiagnostic),
        ),
        (
            "scout_only".to_owned(),
            json!(mode == EvaluationMode::Canary),
        ),
        ("suspect_protocol".to_owned(), json!(true)),
        ("suspect_leakage".to_owned(), json!(suspect_leakage)),
        ("late_after_generation_boundary".to_owned(), json!(false)),
        ("evaluator_wall_seconds".to_owned(), json!(0.0)),
    ]));
}

fn write_failure_summary(
    output_dir: &Path,
    variant_id: &str,
    mode: EvaluationMode,
    reason: &str,
    suspect_leakage: bool,
) -> Result<PathBuf> {
    fs::create_dir_all(output_dir)?;
    let mut metrics = default_metrics();
    add_failure_metric_contract(&mut metrics, mode, suspect_leakage);
    let path = output_dir.join("evaluation_summary.json");
    let identity = producer_identity();
    let mut summary = json!({
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "variant_id": variant_id,
        "metrics": Value::Object(metrics),
        "frontier_lane": "diagnostic",
        "promotion_lane": "diagnostic",
        "evidence_stage": mode.as_str(),
        "eval_stage": mode.as_str(),
        "tier": mode.as_str(),
        "result_status": "protocol_failed",
        "scored_complete": false,
        "complete_eval": false,
        "promotion_eligible": false,
        "confirmed_performance_gate_passed": false,
        "research_independence_attested": false,
        "effort_ratio": 0.0,
        "coverage_ratio": 0.0,
        "actual_evaluation_units": 0,
        "reference_evaluation_units": COMPLETE_UNITS,
        "evaluation_units_completed": 0,
        "evaluation_units_required": COMPLETE_UNITS,
        "effective_config": {},
        "effective_config_complete": false,
        "design_dimensions": {},
        "extra": {
            "frontier_lane": "diagnostic",
            "promotion_lane": "diagnostic",
            "evidence_stage": mode.as_str(),
            "evaluation_summary": path,
        },
        "protocol_integrity": {
            "passed": false,
            "research_independence_attested": false,
            "suspect_leakage": suspect_leakage,
            "failure_type": "RustEvaluationError",
            "failure_reason": reason.chars().take(2000).collect::<String>(),
        },
    });
    if let Some(object) = summary.as_object_mut() {
        object.extend(identity.clone());
        if let Some(extra) = object.get_mut("extra").and_then(Value::as_object_mut) {
            extra.extend(identity);
        }
    }
    write_json(&path, &summary)?;
    fs::write(output_dir.join("failure.log"), format!("{reason}\n"))?;
    Ok(path)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EvidenceEligibility {
    promotion_eligible: bool,
    confirmed_performance_gate_passed: bool,
}

fn evidence_eligibility(
    scored_complete: bool,
    protocol_integrity_passed: bool,
    contract_lock_passed: bool,
    landing_success_nonzero: bool,
    hard_ood_nonzero: bool,
    radius_nonzero: bool,
) -> EvidenceEligibility {
    // Praxist interprets `promotion_eligible=false` as signal-only evidence and
    // excludes it from durable routing and generation maturity.  Keep that
    // generic contract independent of task performance.  The stricter Rocket Booster Recovery
    // confirmed gate remains an explicit, task-owned scientific fact.
    let promotion_eligible = scored_complete && protocol_integrity_passed && contract_lock_passed;
    let confirmed_performance_gate_passed =
        promotion_eligible && landing_success_nonzero && hard_ood_nonzero && radius_nonzero;
    EvidenceEligibility {
        promotion_eligible,
        confirmed_performance_gate_passed,
    }
}

pub fn run_candidate_from_env<C: CandidateController>() {
    let args = RunnerArgs::parse();
    if let Err(error) = fs::create_dir_all(&args.output_dir) {
        eprintln!("cannot create evaluator output directory: {error:#}");
        std::process::exit(2);
    }
    match evaluate::<C>(&args) {
        Ok(summary) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "summary": absolute(&args.output_dir).unwrap_or_else(|_| args.output_dir.clone()).join("evaluation_summary.json"),
                    "variant_id": summary["variant_id"],
                    "stage": summary["evidence_stage"],
                    "scored_complete": summary["scored_complete"],
                    "landing_success_rate": summary["metrics"]["landing_success_rate"],
                    "first_contact_sink_speed_p95_mps": summary["metrics"]["first_contact_sink_speed_p95_mps"],
                    "elapsed_seconds": summary["execution"]["elapsed_seconds"],
                }))
                .expect("serialize compact evaluator result")
            );
        }
        Err(error) => {
            let variant_id = variant_id_hint(&args.variant_dir);
            let reason = format!("{error:#}");
            let suspect_leakage = reason.contains("research independence")
                || reason.contains("research_independence");
            let path = write_failure_summary(
                &args.output_dir,
                &variant_id,
                args.mode,
                &reason,
                suspect_leakage,
            )
            .unwrap_or_else(|write_error| {
                eprintln!("failed to write failure summary: {write_error:#}");
                args.output_dir.join("evaluation_summary.json")
            });
            eprintln!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "summary": path,
                    "variant_id": variant_id,
                    "status": "protocol_failed",
                    "error": reason,
                }))
                .unwrap_or_else(|_| "candidate evaluation failed".to_owned())
            );
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn maturity_ratios_match_the_13312_unit_protocol() {
        assert_eq!(EvaluationMode::Complete.effort_ratio(), 1.0);
        assert_eq!(EvaluationMode::Complete.coverage_ratio(), 1.0);
        assert_eq!(
            EvaluationMode::Development.effort_ratio(),
            2_048.0 / 13_312.0
        );
        assert_eq!(EvaluationMode::Development.coverage_ratio(), 0.0);
        assert_eq!(
            EvaluationMode::RollDiagnostic.coverage_ratio(),
            1_024.0 / 13_312.0
        );
        assert_eq!(
            EvaluationMode::RollDiagnostic.cli_value(),
            "roll-diagnostic"
        );
        assert_eq!(EvaluationMode::RollDiagnostic.as_str(), "roll_diagnostic");
    }

    #[test]
    fn failure_metrics_can_never_mature_or_parent() {
        let mut metrics = default_metrics();
        add_failure_metric_contract(&mut metrics, EvaluationMode::Complete, false);
        assert_eq!(metrics["scored_complete"], json!(false));
        assert_eq!(metrics["parent_authorized"], json!(false));
        assert_eq!(metrics["promotion_eligible"], json!(false));
        assert_eq!(metrics["confirmed_performance_gate_passed"], json!(false));
        assert_eq!(metrics["close_eligible"], json!(false));
        assert_eq!(metrics["protocol_integrity_failed"], json!(true));
        assert_eq!(metrics["evaluation_units_completed"], json!(0));
    }

    #[test]
    fn durable_evidence_is_independent_of_confirmed_performance() {
        let eligibility = evidence_eligibility(true, true, true, true, false, false);
        assert!(eligibility.promotion_eligible);
        assert!(!eligibility.confirmed_performance_gate_passed);
    }

    #[test]
    fn confirmed_performance_requires_every_scientific_gate() {
        let eligibility = evidence_eligibility(true, true, true, true, true, true);
        assert!(eligibility.promotion_eligible);
        assert!(eligibility.confirmed_performance_gate_passed);

        let protocol_failed = evidence_eligibility(false, false, false, true, true, true);
        assert!(!protocol_failed.promotion_eligible);
        assert!(!protocol_failed.confirmed_performance_gate_passed);
    }

    #[test]
    #[ignore = "nested Cargo preflight; run explicitly before starting a research run"]
    fn generated_runner_compiles_typed_params_and_candidate_modules() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "rocket-booster-recovery-rust-generated-runner-test-{}-{unique}",
            std::process::id()
        ));
        let variant = root.join("variant");
        fs::create_dir_all(&variant).unwrap();
        let baseline = task_root().join("assets/baseline");
        let controller = fs::read_to_string(baseline.join("controller.rs")).unwrap();
        let controller = controller.replacen(
            "pub use rocket_booster_recovery_task::candidate_api::ControllerConfig;",
            r#"use rocket_booster_recovery_task::candidate_api::{Deserialize, VariantConfig};

#[derive(Clone, Debug, Default, Deserialize)]
pub struct CandidateParameters {
    #[serde(default)]
    pub mechanism_gain: f32,
}

pub type ControllerConfig = VariantConfig<CandidateParameters>;
mod mechanism;"#,
            1,
        );
        fs::write(variant.join("controller.rs"), controller).unwrap();
        fs::write(
            variant.join("mechanism.rs"),
            "pub fn bounded(value: f32) -> f32 { value.clamp(0.0, 1.0) }\n",
        )
        .unwrap();
        let mut config: Value =
            serde_json::from_slice(&fs::read(baseline.join("controller_config.json")).unwrap())
                .unwrap();
        config
            .as_object_mut()
            .unwrap()
            .insert("variant_params".to_owned(), json!({"mechanism_gain": 1.0}));
        write_json(&variant.join("controller_config.json"), &config).unwrap();
        fs::copy(baseline.join("variant.json"), variant.join("variant.json")).unwrap();

        let (_, audit) = inspect_variant(&variant).unwrap();
        let manifest_path = stage_candidate_runner(&variant, &audit, &root.join("cache")).unwrap();
        let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
        let status = Command::new(cargo)
            .args([
                "run",
                "--quiet",
                "--release",
                "--offline",
                "--manifest-path",
            ])
            .arg(manifest_path)
            .arg("--")
            .arg("--variant-dir")
            .arg(&variant)
            .arg("--mode")
            .arg("canary")
            .arg("--output-dir")
            .arg(root.join("output"))
            .arg("--project-root")
            .arg(task_root().parent().unwrap())
            .arg("--expected-source-tree-sha256")
            .arg(&audit.source_tree_sha256)
            .arg("--expected-config-sha256")
            .arg(&audit.config_sha256)
            .arg("--expected-manifest-sha256")
            .arg(&audit.manifest_sha256)
            .arg("--threads")
            .arg("1")
            .env("CARGO_TARGET_DIR", root.join("target"))
            .env("CARGO_NET_OFFLINE", "true")
            .status()
            .unwrap();
        assert!(status.success());
        assert!(root.join("output/evaluation_summary.json").is_file());
        fs::remove_dir_all(root).unwrap();
    }
}
