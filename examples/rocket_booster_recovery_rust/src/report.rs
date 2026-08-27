use std::{
    collections::BTreeMap,
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    config::ControllerConfig,
    dataset::{
        self, COMPLETE_ROWS, INITIAL_FUEL_KG, INITIAL_MASS_KG, MASS_EMPTY_KG, Mode,
        ROWS_PER_COMPLETE_SOURCE, SOURCE_BANKS, SOURCE_ROWS_DIGEST,
    },
    metrics::{
        REFERENCE_SUCCESS_COUNT, REFERENCE_SUCCESS_RATE, SUCCESS_THRESHOLDS, landing_report,
    },
    plant::PlantConfig,
    rollout::rollout_all,
};

pub const PROTOCOL_NAME: &str = "rocket_booster_recovery_first_contact_7000kg_validation_v2";
pub const PROTOCOL_VERSION: u32 = 2;
pub const CONFIG_HASH: &str = "9aad17d00c3c29081600081ea0a9bd4a0dbb0a0217ea68fd0c42c3762294f5e8";

#[derive(Clone, Debug)]
pub struct EvaluationOptions {
    pub root: PathBuf,
    pub mode: Mode,
    pub output_dir: PathBuf,
    pub threads: usize,
    pub write_trajectories: bool,
}

#[derive(Clone, Debug)]
pub struct EvaluationOutcome {
    pub summary: Value,
    pub path: PathBuf,
    pub integrity_passed: bool,
}

fn write_trajectories(
    path: &Path,
    data: &dataset::Dataset,
    results: &[crate::rollout::RolloutResult],
) -> Result<()> {
    let file = fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    write!(writer, "index,source_id,source_row")?;
    for name in [
        "x_m",
        "y_m",
        "z_m",
        "vx_mps",
        "vy_mps",
        "vz_mps",
        "q_w",
        "q_x",
        "q_y",
        "q_z",
        "omega_x",
        "omega_y",
        "omega_z",
        "mass_kg",
        "time_step",
        "initial_r_m",
    ] {
        write!(writer, ",{name}")?;
    }
    writeln!(
        writer,
        ",first_contact_detected,first_contact_step,contact_leg_sink_speed_mps"
    )?;
    for (index, result) in results.iter().enumerate() {
        write!(
            writer,
            "{index},{},{}",
            data.source_ids[index], data.source_rows[index]
        )?;
        for value in result.terminal_state {
            write!(writer, ",{value:.9}")?;
        }
        writeln!(
            writer,
            ",{},{},{:.9}",
            u8::from(result.first_contact_detected),
            result.first_contact_step,
            result.first_contact_leg_sink_speed_mps,
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn sha256_bytes(bytes: impl AsRef<[u8]>) -> String {
    hex::encode(Sha256::digest(bytes.as_ref()))
}

fn source_hashes(root: &Path) -> Result<BTreeMap<String, String>> {
    let paths = [
        "Cargo.toml",
        "Cargo.lock",
        "src/config.rs",
        "src/controller.rs",
        "src/dataset.rs",
        "src/math.rs",
        "src/metrics.rs",
        "src/plant.rs",
        "src/report.rs",
        "src/rollout.rs",
        "src/main.rs",
        "config/controller.json",
    ];
    paths
        .iter()
        .map(|relative| {
            let path = root.join(relative);
            let hash = dataset::sha256_file(&path)?;
            Ok(((*relative).to_owned(), hash))
        })
        .collect()
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
        if key == name {
            value.split_whitespace().next()?.parse().ok()
        } else {
            None
        }
    })
}

fn metric_f64(metrics: &Value, key: &str) -> f64 {
    metrics[key].as_f64().unwrap_or(f64::NAN)
}

fn metric_u64(metrics: &Value, key: &str) -> u64 {
    metrics[key].as_u64().unwrap_or(u64::MAX)
}

pub fn evaluate(options: &EvaluationOptions) -> Result<EvaluationOutcome> {
    let total_started = Instant::now();
    let asset_hashes = dataset::attest_assets(&options.root)?;
    let config_path = options.root.join("config/controller.json");
    let config_hash = dataset::sha256_file(&config_path)?;
    ensure!(
        config_hash == CONFIG_HASH,
        "controller config hash mismatch: expected {CONFIG_HASH}, got {config_hash}"
    );
    let controller_cfg = ControllerConfig::load(&config_path)?;
    let source_hashes = source_hashes(&options.root)?;
    let data = dataset::load(&options.root, options.mode)?;
    let rows_json = serde_json::to_vec(&data.source_rows)?;
    let observed_rows_digest = sha256_bytes(rows_json);
    let selection_passed =
        options.mode != Mode::Complete || observed_rows_digest == SOURCE_ROWS_DIGEST;
    ensure!(
        selection_passed,
        "complete source-row selection digest mismatch"
    );

    let plant_cfg = PlantConfig::frozen(&controller_cfg);
    let rollout_started = Instant::now();
    let results = rollout_all(&data.states, &controller_cfg, &plant_cfg, options.threads);
    let rollout_seconds = rollout_started.elapsed().as_secs_f64();
    let scoring_started = Instant::now();
    let report = landing_report(&data, &results);
    let scoring_seconds = scoring_started.elapsed().as_secs_f64();

    let channel_lock_passed = metric_f64(&report.metrics, "forbidden_action_max_abs") == 0.0
        && metric_f64(&report.metrics, "plant_lateral_rcs_max_nm") == 0.0;
    let finite_passed = metric_f64(&report.metrics, "nonfinite_trajectory_rate") == 0.0;
    let endpoint_passed = metric_f64(&report.metrics, "gear_damping_credit_rate") == 0.0
        && metric_u64(&report.metrics, "post_contact_scored_steps") == 0;
    let row_count_passed = data.states.len() == options.mode.expected_rows();
    let reference_match = options.mode != Mode::Complete
        || (metric_u64(&report.metrics, "landing_success_count") == REFERENCE_SUCCESS_COUNT as u64
            && metric_f64(&report.metrics, "landing_success_rate") == REFERENCE_SUCCESS_RATE);
    let terminal_previous_omega_finite = results.iter().all(|result| {
        result
            .terminal_previous_omega
            .iter()
            .all(|value| value.is_finite())
    });
    let integrity_passed = channel_lock_passed
        && finite_passed
        && endpoint_passed
        && row_count_passed
        && selection_passed
        && reference_match
        && terminal_previous_omega_finite;

    let source_bank_hashes: BTreeMap<String, String> = SOURCE_BANKS
        .iter()
        .map(|(name, _, expected)| ((*name).to_owned(), (*expected).to_owned()))
        .collect();
    let asset_hash_map: BTreeMap<String, String> = asset_hashes.into_iter().collect();
    let effective_threads = if options.threads == 0 {
        rayon::current_num_threads()
    } else {
        options.threads
    };
    let unix_time_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let total_seconds = total_started.elapsed().as_secs_f64();
    let summary = json!({
        "schema_version": 1,
        "project": "rocket_booster_recovery_rust",
        "controller": {
            "name": "Rocket Booster Recovery deterministic classical baseline",
            "version": controller_cfg.version,
            "implementation_language": "Rust",
            "source_sha256": source_hashes["src/controller.rs"],
            "config_sha256": config_hash,
            "config_digest": sha256_bytes(serde_json::to_vec(&serde_json::from_slice::<Value>(&fs::read(&config_path)?)?)?),
            "learned_parameters": false,
        },
        "protocol": {
            "name": PROTOCOL_NAME,
            "version": PROTOCOL_VERSION,
            "mode": options.mode.as_str(),
            "single_success_standard": true,
            "endpoint": "interpolated_first_landing_leg_contact",
            "post_contact_spring_or_damper_credit": false,
            "thresholds": SUCCESS_THRESHOLDS.as_json(),
        },
        "dataset": {
            "landing_rows": data.states.len(),
            "complete_landing_rows": COMPLETE_ROWS,
            "rows_per_source_in_complete_mode": ROWS_PER_COMPLETE_SOURCE,
            "selection_seed_nominal": 20260822,
            "selection_seed_near_hard": 20260821,
            "source_rows_sha256": observed_rows_digest,
            "source_bank_hashes": source_bank_hashes,
            "initial_mass_kg": INITIAL_MASS_KG,
            "initial_fuel_kg": INITIAL_FUEL_KG,
            "mass_empty_kg": MASS_EMPTY_KG,
        },
        "metrics": report.metrics,
        "reference": {
            "expected_success_count": REFERENCE_SUCCESS_COUNT,
            "expected_denominator": COMPLETE_ROWS,
            "expected_success_rate": REFERENCE_SUCCESS_RATE,
            "reference_runtime": "Python 3.11.15 / JAX 0.9.2",
            "match": reference_match,
        },
        "integrity": {
            "passed": integrity_passed,
            "asset_hashes_passed": true,
            "source_selection_passed": selection_passed,
            "channel_lock_passed": channel_lock_passed,
            "finite_trajectories_passed": finite_passed,
            "first_contact_endpoint_passed": endpoint_passed,
            "row_count_passed": row_count_passed,
            "outcome_dependent_selection": false,
            "attested_asset_hashes": asset_hash_map,
            "implementation_source_hashes": source_hashes,
        },
        "plant": {
            "integrator": "rk4",
            "substeps": 1,
            "dt_s": plant_cfg.dt,
            "max_steps": plant_cfg.max_steps,
            "implementation": "allocation-free fixed-size Rust f32 dynamics",
            "source_model": "frozen C05 6DoF equations from the Python/JAX reference snapshot",
            "action_adapter": "full physical action; clip; hard-zero indices 4,5,8",
            "checkpoint_loaded": false,
            "parameter_updates": 0,
        },
        "execution": {
            "unix_time_seconds": unix_time_seconds,
            "elapsed_seconds": total_seconds,
            "rollout_seconds": rollout_seconds,
            "scoring_seconds": scoring_seconds,
            "runtime_language": "Rust",
            "crate_version": env!("CARGO_PKG_VERSION"),
            "target_arch": std::env::consts::ARCH,
            "target_os": std::env::consts::OS,
            "backend": "cpu",
            "gpu_required": false,
            "trajectory_parallelism": "Rayon indexed parallel iterator",
            "threads": effective_threads,
            "available_parallelism": std::thread::available_parallelism().map(usize::from).unwrap_or(1),
            "cpu_model": cpu_model(),
            "resident_memory_kb": proc_status_kb("VmRSS"),
            "peak_resident_memory_kb": proc_status_kb("VmHWM"),
            "endpoint_previous_omega_finite": terminal_previous_omega_finite,
        },
        "details": report.details,
    });
    fs::create_dir_all(&options.output_dir)
        .with_context(|| format!("create {}", options.output_dir.display()))?;
    if options.write_trajectories {
        write_trajectories(
            &options.output_dir.join("trajectory_endpoints.csv"),
            &data,
            &results,
        )?;
    }
    let path = options.output_dir.join("evaluation_summary.json");
    fs::write(&path, serde_json::to_vec_pretty(&summary)?)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(EvaluationOutcome {
        summary,
        path,
        integrity_passed,
    })
}
