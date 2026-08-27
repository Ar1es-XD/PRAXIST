# Rust macOS Apple Silicon CPU resource plan

## Measurement status

The frozen controller, simulator, evaluator, source banks, and success gate are
identical to the Linux task. No complete baseline or full Praxist run has yet
been measured on macOS/Apple Silicon. Therefore Mac baseline score, complete
wall time, throughput, process-tree peak RSS, and safe concurrency remain
unmeasured. The historical `495 / 12,288` result belongs only to the declared
Linux/x86_64 reference platform.

The default envelope targets a modern Apple Silicon notebook with roughly 18
CPU cores and 36 GiB unified memory. It is a conservative launch profile, not a
claim of measured Mac throughput. The task has no Metal or GPU backend and does
not reserve unified memory for accelerator computation.

## Native runtime contract

- macOS 11 or newer and native `aarch64-apple-darwin` Rust 1.85 or newer.
- Xcode Command Line Tools must provide the native linker.
- `cargo` resolves through `PATH`; Rosetta/x86_64 Rust is not the supported
  qualification path.
- Frozen crates are stored in repository-root `vendor/`; Cargo remains
  `--locked --offline` during research.
- No task Python, JAX, CUDA, Metal, GPU, or venv is required. Praxist itself is
  installed separately and its agent provider still requires normal network
  access even though online literature search is disabled.

## Conservative scheduler envelope

- Profile: `cpu_evaluation`.
- Initial/minimum/maximum concurrent evaluators: 2 / 1 / 2.
- Each evaluator uses 12 Rayon workers; Cargo uses at most 2 build jobs.
- Each peer may have at most one active evaluator; the cohort remains 16.
- The pressure domain is CPU only. Current Praxist host memory/I/O observation
  relies on Linux procfs and is not treated as authoritative on macOS. The low
  concurrency cap supplies the safety boundary until native host telemetry is
  qualified.
- The task-local resource observer polls `ps` for direct-child RSS on macOS;
  evaluator summaries use `sysctl` for CPU identity and `ps` for current RSS.
- Supply lease: 600 seconds; mature fraction: 0.25; mature redundancy: 3.0;
  one exploration slot remains reserved.
- Synthesis stays fixed at 90 minutes, with a two-hour peer budget. Gen0 DIG is
  outside the peer timer. This preserves the successful research cadence and
  does not shorten complex-mechanism work for a faster evaluator.

## First-host qualification

Before starting research, run the baseline in canary, development, and complete
modes with 12 threads. Record:

- native target triple and Rust version;
- complete landing count and every protocol/hash field;
- complete wall time and cold/warm build time;
- current and peak child RSS, host memory pressure, and thermal behavior;
- one-way versus two-way complete-evaluator throughput;
- stop/resume behavior for one bounded smoke run.

Only retain two-way concurrency if both complete evaluations are bitwise
consistent in protocol/row identity, produce the same scientific metrics, and
increase total throughput without memory pressure. Otherwise set the scheduler
to 1 / 1 / 1. Keep the first Mac measurement separate from the Linux reference;
small architecture-dependent floating-point differences must remain visible.

For a 30-generation campaign, reserve at least 50 GiB free disk, connect power,
and prevent system sleep. The fixed research windows alone total 45 hours, so a
full run is expected to span roughly 50--60 hours including DIG, PI panels,
draining, and retries.
