# Rocket Booster Recovery Praxist PC Qualification Task

This directory is scientifically equivalent to `task_GPU_server/`: it retains the
same controller, plant, data, evaluation protocol, roles, multi-metric Pareto lanes,
generation-0-only DIG, all-generation QD, and constructive-peer soft target. It
exists to establish independent evidence on a target personal computer.

Every generation retains a fixed 120-minute research window and must drain and
synthesize within the 150-minute hard generation limit. A small number of fast
complete evaluations cannot terminate exploration of more complex mechanisms.
Candidates may originate only from the frozen baseline or canonical parents in the
current run; earlier runs, packaged champions, sibling checkouts, and Git history
are outside the task boundary.

The authoritative baseline is under `assets/baseline/`, with evidence records under
`assets/baselines/`. The PC baseline has not run, so its score, wall time, throughput,
RAM, accelerator or unified-memory pressure, and resource bottleneck are unknown.
Machine-readable performance values are `null`, and `task.yaml:baselines` is empty.
The 8x H100 server values remain only in `task_GPU_server/` and were not copied as PC
results.

This is a complete PC qualification harness, not a qualified platform result. It uses
a conservative two-way shared-GPU admission envelope: each evaluator reserves 8 GiB
and 50% utilization, with no more than two concurrent GPU evaluators globally. Before
formal research, run the same 13,312-unit complete baseline on the target PC and
measure single-versus-dual concurrency. The configuration permits sharing but does
not claim qualified two-way throughput. Linux/NVIDIA CUDA and macOS CPU or Metal
require their own matching environments and cannot stand in for each other's results.

The repository does not distribute `experiments/`. Praxist creates it at first
launch, so a fresh example contains no runs, findings, frontier, PI panel, Gems, peer
workspaces, or runtime logs.

Resolve from the example root:

```bash
resolve_dir="$(mktemp -d)"
praxist resolve "$PWD/task_PC" --run-dir "$resolve_dir"
```

After platform dependencies, baseline measurement, and resource calibration, launch
with:

```bash
praxist start --task-path "$PWD/task_PC" \
  --cohort 16 --generations 30 --daemonize --json
```

Measure the PC baseline with:

```bash
ROCKET_BOOSTER_RECOVERY_TASK_DIR=task_PC ./scripts/run_baseline.sh complete
```

`task.yaml` selects the evaluator environment at `../.venv`. The model provider,
agent runtime, and credentials belong to the operator environment. The central
scheduler remains the sole owner of GPU UUID assignment; peers must not launch JAX
around it. Two-way sharing is a ceiling, not a minimum. The scheduler serializes work
when accelerator memory, utilization, CPU, RAM, or I/O pressure requires it.
