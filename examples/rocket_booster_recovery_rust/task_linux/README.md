# Rocket Booster Recovery (Rust) Praxist task — Linux

This is the portable Linux counterpart of `task_GPU_server`. It contains no
server-private absolute paths and runs with a native Rust toolchain on x86_64
or aarch64 Linux. The evaluator, candidate-controller API, 6DoF simulation,
metric computation, and resource observer run on CPU in Rust; they do not use
Python, NumPy, JAX, XLA, CUDA, a GPU, or a task venv. The operator installs
Praxist separately.

The research policy remains 16 peers over 30 generations, DIG in Gen0 only,
QD in every generation, a 75% constructive-mix soft constraint, no online
literature search, and multi-metric Pareto retention in the frontier and
incubator.

## Frozen Baseline And Success Standard

The baseline under `assets/baseline/` contains `controller.rs`,
`controller_config.json`, and `variant.json`. Every research candidate must
retain those three root files, but may add local `.rs` modules. Candidates may
not add a Cargo manifest or third-party dependency.

The complete protocol contains 12,288 landing trajectories across three
strata and 1,024 roll-disturbance trajectories. The joint success predicate is
evaluated at interpolated first landing-leg contact. It requires no more than
5 m lateral error, 1 m/s center-of-mass and lowest-contacting-leg sink speed,
0.3 m/s lateral speed, 1.5 degrees tilt, 0.02 rad/s absolute roll rate, and
0.03 rad/s pitch/yaw rate norm, with strictly more than 2% of the initial
7,000 kg main fuel remaining. Post-contact spring and damper response receives
no score credit.

The Linux reference baseline reproduces
`495 / 12,288 = 4.0283203125%` on a Xeon 8457C. Complete evidence and the
compact ledger are under `assets/baselines/`; they are baseline assets, not
historical run records. Remeasure the baseline after moving to a new CPU or
architecture and record that platform separately. Never represent the server
record as a local measurement. Runtime-created `experiments/` and `scratch/`
directories are excluded from version control.

## Rust Toolchain

Rust 1.85 or newer and a native `cargo` on `PATH` are required. The root
`vendor/` directory contains locked Cargo dependencies, and each portable task
uses it through its own `.cargo/config.toml`. Formal builds therefore use
`--locked --offline` without accessing crates.io during research. There is no
simulator venv or task venv. Do not copy another host's Rust toolchain or
Python environment.

Build and verify from the example root:

```bash
cargo --version
cargo build --release --locked --offline --manifest-path task_linux/Cargo.toml
cargo test --release --locked --offline --manifest-path task_linux/Cargo.toml

cargo run --release --locked --offline \
  --manifest-path task_linux/Cargo.toml \
  --bin rocket-booster-recovery-task-eval -- \
  --variant-dir task_linux/assets/baseline \
  --mode complete \
  --output-dir task_linux/scratch/manual-baseline-complete \
  --threads 16
```

The four evaluator modes are `canary`, `development`, `roll-diagnostic`, and
`complete`; Praxist normalizes the roll evidence label to `roll_diagnostic`.
Only a complete 13,312-unit result may mature, serve as a parent, or support a
normal generation close.

A complete, protocol-clean result sets `promotion_eligible=true`, making it
eligible for durable evidence, incubator retention, and generation-close
counts. This does not make it a confirmed champion. The confirmed lane also
requires `confirmed_performance_gate_passed=true`: overall, hard-OOD, and
worst initial-radius-stratum success rates must all be nonzero. A complete but
weak result can therefore remain auditable evidence and a Pareto parent
without masquerading as a strict champion or blocking the maturity quorum.

## Validate And Launch With Praxist

```bash
resolve_dir="$(mktemp -d)"
praxist resolve "$PWD/task_linux" --run-dir "$resolve_dir"

praxist start --task-path "$PWD/task_linux" \
  --cohort 16 --generations 30 --daemonize --json
```

Provider configuration, agent runtime, and credentials belong to the operator
environment and are not stored in the harness. Every formal peer evaluation
must launch through the central scheduler's `cpu_evaluation` profile using a
supported command template from the prompt. Portable Linux defaults to at
most two concurrent evaluators, each with 16 Rayon workers for a complete
evaluation. Raise that limit on a high-core server only after measurement.
The candidate static audit is not an operating-system sandbox; do not run
untrusted candidate code on a host with sensitive credentials or mounts.
Report security issues through the enclosing Praxist project's process.

## Candidate Isolation

A peer may modify only the three root files in its run-owned variant and
optional local `.rs` modules. The frozen launcher parses and content-addresses
the full Rust source tree, performs TOCTOU verification, rejects `unsafe`,
custom `#[path]`, file/network/subprocess access, dynamic includes, learning
frameworks, and evaluator/data references, then compiles the candidate in an
isolated Cargo cache under `scratch/`. The generated crate provides Serde
derives directly. Candidates may use `DynamicControllerConfig` or
`VariantConfig<LocalParams>` to add a small set of mechanism parameters under
`variant_params` without redeclaring roughly one hundred frozen fields. The
generated runner can call only the task's limited state, math, and control
output API. The frozen harness always owns the plant, data selection, contact
interpolation, success gate, and evidence classification.

Generation synthesis starts after the full 90-minute formal peer research
window, aligned with the validated Python harness clock. Early findings,
contributing peers, or mature-result quorum do not remove time reserved for
implementing, falsifying, ablating, and completely reevaluating complex
mechanisms. Gen0 DIG occurs before this timer and is excluded from the
90-minute interval. These clock settings affect orchestration only; they do
not change trajectories, success gates, or baseline metrics.

## Fair-Evolution Boundary

This task contains no prior run, peer workspace, champion package, historical
score trajectory, or cross-language winning-mechanism prior. Researchers may
use only the frozen Rust baseline, generic directions bundled with this task,
and evidence created after the current run starts. They must not search for,
read, translate, or reuse old controllers, configurations, or reports, nor use
Git history, adjacent checkouts, or external paths to recover answers. The
complete policy is in `assets/research_independence_policy.md`.

Every `variant.json` must contain four `research_independence` declarations,
all set to `false`; the frozen evaluator rejects missing or non-clean
declarations. The declaration must agree with session-access evidence. If a
PI or Chair detects access to old results, the candidate and descendants must
be marked as leakage and excluded from parenting and promotion. This fairness
boundary is independent of multi-file Rust candidates, `variant_params`, the
pinned Cargo toolchain, and the two infrastructure retries retained for first-
run interface or compilation-environment failures.
