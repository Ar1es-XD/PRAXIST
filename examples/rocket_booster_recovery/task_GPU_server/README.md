# Rocket Booster Recovery Praxist Task

This directory is a complete, launch-ready GPU-server Praxist task harness. It uses
16 peers, 30 generations, generation-0-only DIG, QD in every generation, a 75%
constructive-peer soft target, and no online literature search.

Every generation retains a fixed 90-minute research window; a small number of fast
replications or local tuning results cannot close it early. Candidates may originate
only from the frozen baseline or canonical parents in the current run. Earlier runs,
packaged champions, sibling checkouts, and Git history are outside the task boundary.

The authoritative baseline is under `assets/baseline/`. Measurement evidence is in
`assets/baselines/baseline_evaluation_summary.json`. It was produced on 2026-08-22 on
a server with 8x NVIDIA H100 80GB HBM3 accelerators; one evaluator used one H100.
Scores, wall time, throughput, and resource behavior have not been measured on a
personal computer. These assets are not Praxist run records.

The repository does not distribute `experiments/`. Praxist creates it at first
launch, so a fresh example contains no runs, findings, frontier, PI panel, Gems, peer
workspaces, or runtime logs.

Resolve from the example root:

```bash
resolve_dir="$(mktemp -d)"
praxist resolve "$PWD/task_GPU_server" --run-dir "$resolve_dir"
```

Start the run:

```bash
praxist start --task-path "$PWD/task_GPU_server" \
  --cohort 16 --generations 30 --daemonize --json
```

`task.yaml` selects the evaluator Python environment at `../.venv`. The model
provider, agent runtime, and credentials belong to the operator environment and are
not stored in the task harness.
