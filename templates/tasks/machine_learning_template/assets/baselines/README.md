# Baseline Records

Replace this placeholder with compact baseline evidence.

Recommended files:

- `results.jsonl`: machine-readable baseline metrics.
- `curated_baseline_summary.md`: human-readable baseline interpretation.
- `baseline_performance_status.md`: whether values are measured, copied from
  prior project artifacts, measured by a task-local baseline bench run, selected
  as explicit zero placeholders, or still missing.

Do not invent baseline numbers. If baselines are missing but the task can
measure them locally, ask the operator whether to write clearly marked zero
placeholders or run `experiments/baseline_bench_<timestamp>/` with bounded
parallelism before filling these files.
