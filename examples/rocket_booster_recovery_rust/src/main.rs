use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::Parser;
use rocket_booster_recovery_rust::{
    PROJECT_ROOT,
    dataset::Mode,
    report::{EvaluationOptions, evaluate},
};

#[derive(Debug, Parser)]
#[command(
    name = "rocket-booster-recovery-rust",
    about = "Rust CPU evaluator for the deterministic Rocket Booster Recovery baseline"
)]
struct Args {
    #[arg(long, value_enum, default_value_t = Mode::Complete)]
    mode: Mode,

    #[arg(long, default_value = "results/latest")]
    output_dir: PathBuf,

    /// Number of trajectory worker threads; 0 uses all logical CPUs visible to the process.
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Write deterministic per-trajectory first-contact endpoints as CSV.
    #[arg(long)]
    write_trajectories: bool,

    /// Repository root containing config/ and data/.
    #[arg(long, default_value = PROJECT_ROOT)]
    project_root: PathBuf,
}

fn absolute(path: PathBuf) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let root = absolute(args.project_root)?;
    let output_dir = absolute(args.output_dir)?;
    let outcome = evaluate(&EvaluationOptions {
        root,
        mode: args.mode,
        output_dir,
        threads: args.threads,
        write_trajectories: args.write_trajectories,
    })?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "summary": outcome.path,
            "mode": args.mode.as_str(),
            "landing_success_count": outcome.summary["metrics"]["landing_success_count"],
            "landing_rows": outcome.summary["dataset"]["landing_rows"],
            "landing_success_rate": outcome.summary["metrics"]["landing_success_rate"],
            "integrity_passed": outcome.integrity_passed,
            "elapsed_seconds": outcome.summary["execution"]["elapsed_seconds"],
            "rollout_seconds": outcome.summary["execution"]["rollout_seconds"],
            "threads": outcome.summary["execution"]["threads"],
        }))?
    );
    if !outcome.integrity_passed {
        bail!(
            "evaluation integrity check failed; inspect {}",
            outcome.path.display()
        );
    }
    Ok(())
}
