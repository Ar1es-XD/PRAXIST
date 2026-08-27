# Porting audit

## Source boundary

The conversion source was a frozen Python/JAX reference snapshot used
read-only. This repository is an independent implementation and contains no
Git submodule or runtime dependency on that source snapshot. The sanitized
reference result and source-component hashes needed for the parity audit are
retained under `baseline/`.

## Component map

| Source component | Rust component | Status |
|---|---|---|
| `rocket_booster_recovery/controller.py` | `src/controller.rs` | complete formal control path |
| `config/controller.json` | unchanged JSON + `src/config.rs` | hash-identical, strongly typed |
| `rocket_booster_recovery/plant_adapter.py` | `src/plant.rs` action boundary/config | complete formal adapter |
| `vendor/.../rocket_6dof_jax.py` | `src/plant.rs`, `src/math.rs` | frozen forces, torque, contact, mass flow, RK4 |
| `vendor/.../grid_fin_aero.py` | frozen constants in `src/plant.rs` | resolved formal authority |
| `vendor/.../plant_aero.py` | `src/plant.rs` | formal incidence-off behavior retained |
| `vendor/.../residual_mask.py` | resolved formal RCS dose in `src/plant.rs` | dose 3.0 remains below guard knee |
| `rocket_booster_recovery/rollout.py` | `src/rollout.rs` | complete first-contact and audit path |
| `rocket_booster_recovery/metrics.py` | `src/metrics.rs` | complete success/diagnostic path |
| `rocket_booster_recovery/evaluate.py` | `src/dataset.rs`, `src/report.rs`, `src/main.rs` | all three modes and reports |
| `tests/test_baseline.py` | unit/integration tests under `src/` and `tests/` | expanded contracts and parity |
| `requirements.txt` | `Cargo.toml`, `Cargo.lock` | Python/JAX/CUDA removed |
| `scripts/run_*.sh` | native Cargo/binary commands | shell runtime wrappers unnecessary |

## Behavioral specialization

The vendored Python simulator was a historical general environment containing
random-reset utilities, RL reward shaping, observation normalization, and
integrator branches that the baseline adapter did not call or explicitly
rejected. The Rust executable implements the formal baseline path used by the
fixed-bank evaluation: physical actions, one-substep RK4, and first-contact
scoring. The random-reset, reward-shaping, and observation-normalization
utilities are outside the supported interface. The force, moment,
gear-contact, quaternion, termination, and fuel equations used by the formal
trajectories are included.

## Fixed assets

The four NPZ files and `controller.json` are byte-for-byte copies with their
original SHA-256 values. Formal NumPy permutation output is materialized as a
small hashed row-index asset so the native executable does not require NumPy.

## Safety and integrity contracts

- RCS pitch, RCS yaw, and grid-fin roll are written as positive zero in the
  controller and again at the plant boundary.
- Initial mass is overridden to 29,200 kg for every selected row.
- Dry mass is 22,200 kg and initial propellant is 7,000 kg.
- No post-contact step contributes to success.
- A nonfinite state is a failure and is recorded by the audit.
- Complete mode requires the fixed row digest and exactly 495 successes.
- Data and configuration hashes are verified before simulation.
- Source hashes for the running Rust implementation are recorded in every
  report.
