# Rocket Booster Recovery

Rocket Booster Recovery is a deterministic classical-control baseline for rocket
landing. This complete Praxist example includes two isolated task harnesses, a
7,000 kg first-contact baseline, the frozen Swordfish C05 6DoF plant, evaluators,
data banks, roles, and audit rules. It contains no historical runs or experiment
directories.

Project identifiers use the distribution name consistently; controller parameters,
datasets, evaluation semantics, and measured values were not changed.

- `task_GPU_server/` is the complete GPU-server task with evidence measured on an
  8x H100 host and a fixed 90-minute research window per generation.
- `task_PC/` is the scientifically equivalent PC qualification harness. Its baseline
  score, wall time, throughput, and resource behavior have not been measured on a
  personal computer, so the corresponding values remain `null` or undeclared. It
  provisionally supports two evaluators sharing one GPU, with a fixed 120-minute
  research window and a 150-minute hard generation limit.

Both tasks permit candidates to originate only from the frozen baseline or a
canonical parent produced by the current run. They expose no prior-work tool and
bundle no historical run, champion, or experiment report that could seed a new run.

## Fixed Scientific Boundary

- Neural networks, reinforcement learning, learned residuals, policy networks, and
  online learning are prohibited.
- `RCS_x` may produce only body-axis roll torque.
- `RCS_y = RCS_z = grid_roll = +0.0` is locked at both the controller and plant
  boundaries.
- Pitch and yaw may be controlled only through `gimbal_y/z + grid_y/z`.
- The plant, integrator, contact model, data banks, and evaluator are frozen assets.
- Initial fuel is 7,000 kg, dry mass is 22,200 kg, and initial total mass is
  29,200 kg.

The task has one trajectory-level success predicate. At the interpolated instant of
first leg-tip contact, lateral error must be no greater than 5 m; center-of-mass and
lowest-leg sink speed must be no greater than 1 m/s; lateral speed must be no greater
than 0.3 m/s; tilt must be no greater than 1.5 degrees; absolute roll rate must be no
greater than 0.02 rad/s; combined pitch/yaw rate must be no greater than 0.03 rad/s;
and main-fuel reserve must be strictly greater than 2%. Scoring stops before the
damped response, so a strategy cannot receive success credit by hitting the landing
gear hard and relying on its damping.

## Baseline Measurement

The authoritative controller baseline is
`rocket_booster_recovery_v2_first_contact_7000kg_baseline`, with identical files in
both tasks' `assets/baseline/` directories. The frozen complete protocol contains
12,288 landing trajectories and 1,024 roll disturbances, for 13,312 evaluation
units:

- Landing success rate: 495 / 12,288 = 0.040283203125.
- Hard-OOD and worst-radius-bin success rates: 0.
- First-contact center-of-mass sink-speed P95: 66.3928455 m/s.
- Fuel-gate pass rate: 0.0758464; main-fuel depletion rate: 0.8863932.
- Frozen roll-disturbance stability rate: 1.0.
- Forbidden action, plant lateral RCS, non-finite trajectory, and post-contact
  damping-credit values: all 0.

These values were first measured on 2026-08-22 and completely remeasured on
2026-08-23 on a server with **8x NVIDIA H100 80GB HBM3** accelerators. One baseline
evaluator used one H100. Both runs produced 495 / 12,288 and identical scientific
metrics. The complete machine-readable evidence is
`task_GPU_server/assets/baselines/baseline_evaluation_summary.json`; the remeasurement
record is in the same directory. Neither file is a Praxist run record.

These scores and execution characteristics have never been measured on a personal
computer. `task_PC/assets/baselines/` contains explicit `null` placeholders. The H100
values must not be represented as measurements from any PC platform.

## Environment Setup

The verified path requires Python 3.11, a CUDA-capable GPU, and a driver compatible
with JAX CUDA 12:

```bash
cd ~/PraxistExamples/rocket_booster_recovery
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The operator environment supplies the Praxist CLI, agent runtime, and model-provider
credentials; this example does not bundle them. See
[`docs/VENV_SETUP.md`](docs/VENV_SETUP.md) for environment and accelerator checks and
[`docs/PLATFORM_BASELINES.md`](docs/PLATFORM_BASELINES.md) for platform-evidence
isolation.

## Verification And Evaluation

Run the lightweight contract tests:

```bash
./scripts/run_tests.sh
```

Run a canary:

```bash
./scripts/run_baseline.sh canary
```

Run the public development or complete promotion-eligible protocol:

```bash
./scripts/run_baseline.sh development
./scripts/run_baseline.sh complete
```

The auxiliary root entry point `python -m src.evaluate` implements the same v2 joint
success predicate. It resolves the frozen-bank initial mass to 29,200 kg, interpolates
first leg-tip contact, stops scoring immediately, and emits only
`landing_success_pass`. It does not calculate the retired `standard`, `strict`, or
`engineering` success rates. `src.independent_recompute` audits the result without
importing the primary metrics module.

The default task is `task_GPU_server`, with output under the ignored
`task_GPU_server/experiments/manual_<mode>/` directory. Select the PC qualification
harness explicitly:

```bash
ROCKET_BOOSTER_RECOVERY_TASK_DIR=task_PC ./scripts/run_baseline.sh complete
```

## Start Praxist

The task harness is complete and does not require initialization. First-use
`praxist setup` places a writable copy at `~/PraxistExamples/rocket_booster_recovery` by default;
`PRAXIST_EXAMPLES_HOME` changes its parent directory. Do not start a run from the
read-only copy inside the Praxist source tree or installed package resources.

```bash
cd ~/PraxistExamples/rocket_booster_recovery
resolve_dir="$(mktemp -d)"
praxist resolve "$PWD/task_GPU_server" --run-dir "$resolve_dir"
praxist start --task-path "$PWD/task_GPU_server" \
  --cohort 16 --generations 30 --daemonize --json
```

The resolve artifacts are written outside the task. A fresh example has no
run records, frontier, findings, peer workspaces, logs, or result directories;
Praxist creates runtime output only when `start` runs.

## Directory Map

- `task_GPU_server/`: complete server task harness and measured baseline evidence.
- `task_PC/`: scientifically equivalent, unmeasured PC qualification harness.
- `src/`: frozen plant adapter, v2 reference controller, and joint-success audit code.
- `vendor/frozen_c05_plant/`: frozen Swordfish C05 physical model.
- `data/`: three 40,960-row source banks and frozen development/formal subsets.
- `configs/`: reference controller configurations.
- `scripts/`: local tests and baseline-evaluation entry points.
- `docs/VENV_SETUP.md`: reproducible environment and accelerator verification.
- `EXPORT_SCOPE.md`: inclusion, exclusion, provenance, and large-file boundaries.
