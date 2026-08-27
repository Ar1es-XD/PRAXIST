# Platform-Specific Baseline Evidence

Both tasks share one controller, frozen plant, data banks, evaluator, success
predicate, and 13,312-unit complete protocol. Their platform evidence remains
isolated.

| Task | Evidence status | Measurement host | Result values |
|---|---|---|---|
| `task_GPU_server/` | measured | 8x NVIDIA H100 80GB HBM3 server; one evaluator used one H100 | measured values retained |
| `task_PC/` | not measured | `null` | scores and execution characteristics are `null` |

The server measurement took place on 2026-08-22. It establishes the baseline result
and capacity shape on that server, not on any personal computer. It must not be
represented as a measurement from any PC platform.

## PC Qualification Rule

The target PC must complete the same protocol under its final backend, JAX version,
and virtual environment. Record at least the platform and OS, CPU, memory,
accelerator, backend, JAX version, all 13,312 completed units, every task metric, wall
time, peak memory, and evaluator concurrency shape.

Before qualification:

- `task_PC/assets/baselines/results.jsonl` keeps `metric_value` as `null`.
- Performance fields in
  `task_PC/assets/baselines/baseline_evaluation_summary.json` remain `null`.
- `task_PC/task.yaml:baselines` remains an empty list so Praxist cannot silently
  convert YAML `null` to 0.
- Inherited server scheduler parameters are treated only as an unqualified starting
  envelope.

One Praxist run must use one fixed backend. Results from different platforms must not
share a frontier or serve as baseline deltas unless case-level equivalence has been
audited.
