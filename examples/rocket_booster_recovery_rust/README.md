# Rocket Booster Recovery (Rust)

Rocket Booster Recovery (Rust) implements the deterministic Rocket Booster Recovery reusable-rocket landing
baseline in Rust for CPU execution. It provides a native Rust binary in place
of the Python, NumPy, JAX, XLA, CUDA, and GPU execution stack while retaining
the frozen controller, 6DoF dynamics, RK4 integration, mass flow, landing-gear
geometry, first-contact interpolation, evaluation banks, and the single joint
landing-success predicate.

The formal 12,288-trajectory evaluation matches the reference result:

| Metric | Rust result | Python/JAX reference |
|---|---:|---:|
| Landing successes | 495 / 12,288 | 495 / 12,288 |
| Landing success rate | **4.0283203125%** | **4.0283203125%** |
| Nominal-unseen | 5.6640625% | 5.6640625% |
| Near-OOD | 6.4208984375% | 6.4208984375% |
| Hard-OOD | 0.0% | 0.0% |

The committed Rust result is in `baseline/evaluation_summary.json`; the
reference result is retained as
`baseline/python_jax_reference_evaluation_summary.json`.

The copy bundled with Praxist is a read-only distribution asset. Install a
writable project before building, testing, evaluating, or starting research:

```bash
praxist examples install rocket_booster_recovery_rust
cd ~/PraxistExamples/rocket_booster_recovery_rust
```

Praxist preserves an existing destination during upgrades.

## What is implemented

- Rolling ZEM/ZEV high-altitude guidance and terminal velocity-corridor logic.
- Geometric pitch/yaw attitude control and constrained TVC/grid-fin allocation.
- Rate-limited, roll-only RCS control.
- Frozen 16-state 6DoF plant, aerodynamic forces and moments, fuel burn, gear
  contact forces, and one-substep RK4 integration.
- Interpolated first-landing-leg contact with ten bisection iterations.
- Fixed canary, development, and complete dataset protocols.
- The complete metric suite, source/radius stratification, Wilson intervals,
  linear quantiles, channel-lock audits, and integrity checks.
- Optional deterministic per-trajectory endpoint export.

There are no learned parameters, policy checkpoints, Python extensions, GPU
kernels, CUDA libraries, or runtime calls to Python/JAX.

## Praxist task harnesses

The repository contains three equivalent Rust ports of the established
Rocket Booster Recovery research harness:

- `task_GPU_server/`: the high-concurrency server profile; despite its
  compatibility name, task execution is CPU-only.
- `task_linux/`: a portable native x86_64/aarch64 Linux profile.
- `task_macos/`: a conservative native Apple Silicon macOS profile.

The three profiles use the 16-peer/30-generation research policy, Gen0 DIG,
all-generation QD, constructive-mix soft constraint, offline literature
policy, four evidence stages, single landing-success predicate, and
multi-metric confirmed/incubator Pareto lanes. The portable profiles retain a
fixed 90-minute peer research window, matching the server profile's minimum
research duration, while reducing simultaneous evaluators to two for
notebook-class thermal and memory headroom.

Peer variants contain only `controller.rs`, `controller_config.json`, and
`variant.json`. A frozen Rust launcher statically audits and compiles each
candidate against a narrow controller API, while the plant, data selection,
first-contact metric code, channel locks, and maturity policy remain outside
the candidate. See the README inside the selected task directory for its build,
evaluation, host-qualification, resolve, and launch commands. The macOS
baseline metrics intentionally remain null until a native Apple Silicon
complete-protocol measurement is committed; the Linux server result is stored
only as a separate reference.

## Success standard

A trajectory succeeds only if every condition is true at interpolated first
landing-leg contact:

| Condition | Threshold |
|---|---:|
| First leg contact | detected and finite |
| Lateral position error | no more than 5 m |
| COM vertical velocity | -1 m/s through 0 m/s |
| Lowest contacting-leg sink speed | no more than 1 m/s |
| Lateral speed | no more than 0.3 m/s |
| Tilt | no more than 1.5 degrees |
| Absolute roll rate | no more than 0.02 rad/s |
| Pitch/yaw angular-rate norm | no more than 0.03 rad/s |
| Remaining fuel | strictly greater than 2% of 7,000 kg |

This is one joint Boolean predicate. Other values in the report are diagnostic
metrics, not alternative success definitions. Scoring stops at first contact;
post-contact spring or damper response cannot rescue an unsafe impact.

## CPU parallelism design

JAX evaluated fixed-size batches to expose thousands of trajectories to XLA.
That execution shape is unnecessary on a CPU. Each trajectory is independent,
but the 900 time steps and four RK stages inside one trajectory are causally
ordered. Rocket Booster Recovery (Rust) therefore:

1. uses fixed-size `f32` arrays and allocation-free math inside each trajectory;
2. executes independent trajectories with Rayon's indexed parallel iterator;
3. keeps each trajectory serial, preserving operation order and determinism;
4. restores results in source-row order, independent of worker scheduling.

This uses trajectory-level parallelism without JIT compilation, padded
batches, device transfers, or a GPU memory model. `--threads 1` selects one
worker; `--threads 0` uses all logical CPUs visible to the process.

## Requirements

- Rust 1.85 or newer; the repository pins the stable toolchain profile.
- A Linux, macOS, or Windows CPU supported by Rust.
- Approximately 10 MB of runtime memory for this dataset on the measured Linux
  server; build-time Cargo storage is separate.

The locked dependency sources are committed under `vendor/`. Repository and
task builds therefore support `--locked --offline` without relying on a
machine-specific Cargo cache.

No GPU, CUDA driver, Python interpreter, virtual environment, or BLAS runtime
is required.

## Safety and trust boundary

This repository is a deterministic research simulator and Praxist example. It
is **not flight software**, has not been qualified for real vehicles, and must
not be used as the sole basis for hardware or safety-critical decisions.

The task harness performs a restrictive static audit before compiling candidate
controllers, but that audit is a research-integrity control rather than an
operating-system sandbox. Run Praxist and candidate code only from agents and
contributors you trust, preferably inside a disposable container or virtual
machine with least-privilege credentials and no sensitive mounts. Security
issues should follow the reporting process of the enclosing Praxist project.

## Build and run

```bash
cargo build --release
./target/release/rocket-booster-recovery-rust --mode complete --output-dir results/latest
```

Equivalent one-command execution:

```bash
cargo run --release -- --mode complete --output-dir results/latest
```

Pin the number of CPU workers when sharing a machine:

```bash
./target/release/rocket-booster-recovery-rust --mode complete --threads 16 \
  --output-dir results/complete-16t
```

Available modes are:

| Mode | Trajectories | Intended use |
|---|---:|---|
| `canary` | 1 | startup and numerical smoke test |
| `development` | 2,048 | fast controller iteration |
| `complete` | 12,288 | formal reference evaluation |

Add `--write-trajectories` to emit
`trajectory_endpoints.csv` alongside the JSON report. The formal run fails with
a nonzero exit status if assets, row selection, disabled channels, finite-state
requirements, first-contact protocol, or the 495/12,288 reference outcome do
not match.

## Test and audit

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --release
cargo test --release --test full_reference -- --ignored
```

The final command reruns all 12,288 trajectories and asserts the complete and
per-source reference rates. `tests/parity.rs` checks a Python/JAX golden
first-contact endpoint. The evaluator also hashes all fixed data assets and the
precomputed source-row selection before execution.

## Measured performance

On 2026-08-24, on an Intel Xeon Platinum 8457C host with 168 logical CPUs
visible to the process, the complete CPU evaluation measured:

| Implementation | Workers/backend | End-to-end evaluation | Peak RSS |
|---|---:|---:|---:|
| Python 3.11.15 + JAX 0.9.2 | JAX CPU | 23.179 s | 822,320 KiB |
| Rust, fixed worker pool | 16 CPU threads | about 1.9 s | under 15 MiB |
| Rust, automatic pool | 168 CPU threads | about 0.35 s | under 15 MiB |

Thus the native evaluator was about 12x faster with 16 Rust workers and about
67x faster when allowed to use all visible CPUs in this server measurement.
These are machine-specific numbers, not a promise for every CPU. See
`docs/BENCHMARKS.md` for commands, warm-JIT context, and limitations.

## Repository layout

```text
config/controller.json     frozen controller configuration
src/controller.rs          deterministic composite controller
src/plant.rs               frozen 6DoF dynamics and RK4 integration
src/rollout.rs             CPU-parallel first-contact rollout
src/metrics.rs             joint success predicate and diagnostics
src/dataset.rs             NPZ loading, fixed row selection, asset hashes
src/report.rs              evaluator, integrity checks, JSON/CSV output
src/main.rs                native command-line entry point
data/                      unchanged fixed evaluation banks and selection asset
baseline/                  Rust result and original Python/JAX reference
tests/                     contracts, numerical parity, and formal reference
docs/                      architecture, parity, benchmarks, and port audit
task_GPU_server/            Rust CPU Praxist research harness
task_linux/                 portable native Linux research harness
task_macos/                 Apple Silicon macOS research harness
vendor/                     locked offline Cargo dependency sources
```

The port was validated against a frozen Python/JAX reference implementation.
The sanitized reference result and numerical comparison are committed under
`baseline/` and `docs/NUMERICAL_PARITY.md`; no external source checkout is
required to build, test, or run this repository.

Each package under `vendor/` retains its upstream license files. Contribution,
security, and project-level licensing policies are supplied by the enclosing
Praxist repository.
