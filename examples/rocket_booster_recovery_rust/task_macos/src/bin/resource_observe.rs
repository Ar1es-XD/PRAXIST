use std::{
    fs,
    path::PathBuf,
    process::{Command, ExitStatus},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde_json::json;

#[derive(Debug, Parser)]
#[command(about = "Observe wall time and peak RSS for one CPU evaluator command")]
struct Args {
    #[arg(long)]
    output: PathBuf,
    #[arg(last = true, required = true)]
    command: Vec<String>,
}

#[cfg(target_os = "linux")]
fn rss_kb(pid: u32) -> Option<u64> {
    let text = fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    text.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        (key == "VmRSS")
            .then(|| value.split_whitespace().next()?.parse().ok())
            .flatten()
    })
}

#[cfg(target_os = "macos")]
fn rss_kb(pid: u32) -> Option<u64> {
    let pid = pid.to_string();
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", pid.as_str()])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().parse().ok())
        .flatten()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn rss_kb(_pid: u32) -> Option<u64> {
    None
}

fn wait_and_observe(child: &mut std::process::Child) -> Result<(ExitStatus, u64)> {
    let mut peak_rss_kb = 0_u64;
    loop {
        peak_rss_kb = peak_rss_kb.max(rss_kb(child.id()).unwrap_or(0));
        if let Some(status) = child.try_wait()? {
            return Ok((status, peak_rss_kb));
        }
        thread::sleep(Duration::from_millis(10));
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (program, command_args) = args.command.split_first().context("missing command")?;
    let started = Instant::now();
    let mut child = Command::new(program)
        .args(command_args)
        .spawn()
        .with_context(|| format!("launch {program}"))?;
    let (status, peak_rss_kb) = wait_and_observe(&mut child)?;
    let payload = json!({
        "schema_version": 1,
        "command": args.command,
        "elapsed_seconds": started.elapsed().as_secs_f64(),
        "peak_direct_child_rss_kb": peak_rss_kb,
        "available_parallelism": std::thread::available_parallelism().map(usize::from).unwrap_or(1),
        "backend": "cpu",
        "gpu_required": false,
        "exit_code": status.code(),
        "success": status.success(),
    });
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut bytes = serde_json::to_vec_pretty(&payload)?;
    bytes.push(b'\n');
    fs::write(&args.output, bytes)?;
    println!("{}", serde_json::to_string_pretty(&payload)?);
    if !status.success() {
        bail!("observed command failed with {status}");
    }
    Ok(())
}
