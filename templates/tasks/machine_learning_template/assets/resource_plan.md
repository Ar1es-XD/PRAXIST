# Resource Plan Template

Replace this file before a real Praxist run.

## Hardware Summary

- CPU:
- Memory:
- Accelerator backend/devices (if any):
- Storage and I/O risk:

## Bottleneck Classification

Choose one or more:

- accelerator-bound
- CPU-bound
- memory-bound
- I/O-bound
- unknown or mixed

## Unchanged Baseline Observation

- Exact public baseline command:
- Runtime/environment used without modification:
- Observation duration and sampling interval:
- CPU utilization/load:
- Memory and I/O pressure:
- Accelerator identity, memory, and utilization when observable:
- Progress/completion observed:
- Whether the run was stopped after its resource shape became clear:
- Uncertainty:
- Raw timestamped sample path, sample count, and process/PID attribution:
- Per-run full-lifetime accelerator utilization mean/quantiles and cross-run robust estimate:
- Cross-run peak accelerator memory plus selected headroom:

Do not construct CPU-only and accelerator-only rewrites to compare speed. Run the
original baseline as-is and observe it externally. If the smallest smoke is not
representative, observe one larger unchanged protocol unit and stop it once the
resource shape is clear; never publish that partial probe as performance.
Start observation before the child process and sample the active accelerator
backend every 100-200 ms when trustworthy telemetry is supported. A task
shorter than roughly five seconds needs at least
three safe identical repetitions or a longer unchanged unit. Do not infer zero
demand from a teardown sample or an undersampled trace; keep uncertain accelerator
demand unknown and exclusive.

## Experiment Cost Estimate

- Domain ranking convention:
- Whether variance/confidence/lower-bound robustness affects ranking:
- Robust frontier metric or Pareto axis, if needed:
- Preliminary evaluator expected runtime:
- Aligned evaluator expected runtime:
- Aligned coverage versus complete protocol:
- Aligned reduced training/optimization effort:
- Complete evaluator expected runtime:
- Complete evaluator expected budget: epochs/steps/seeds/folds/evaluation units:
- Complete evaluator protocol-integrity fields to emit:
- Expected calibration check between aligned and complete evidence:
- Expected accelerator-hours per complete experiment, when applicable:
- Expected memory demand:
- Evidence source for this estimate:

Under the recommended default, preliminary evidence is for wiring and failure
triage only. Aligned evidence is the cheapest early-ranking protocol that still
uses the same evaluator path, primary metric direction, aggregation,
invalid-result rules, and leakage checks as the complete protocol. Preserve the
same or near-complete data/evaluation coverage whenever possible. Save compute
mainly by reducing fixed training or optimization budget:
epochs, gradient steps, rollout horizon, simulator steps, inner-loop
iterations, or repeated restarts. If coverage is reduced, record the coverage
ratio, stratification, hard-case coverage, and omitted-unit rationale.

Do not silently change the protocol authority when a budget is materially below
project evidence, the task-defined reference effort, or measured convergence
behavior. Ask only when user intent is absent or ambiguous. If the user
explicitly authorizes that reduced protocol, record and obey the choice while
keeping its actual budget visible. Do not apply a universal epoch, step,
rollout, or time threshold.

## Central Experiment Scheduler

- `compute_budget.resource_scheduler.mode`: `central`
- Named profiles and accelerator mode:
- Initial total experiment concurrency:
- Minimum/maximum total experiment concurrency:
- Accelerator memory/utilization demand per managed profile (omit when unknown):
- Pressure domains per profile:
- Explicitly declared equivalence between execution backends, if any:
- Default profile and why it matches the public evaluator:
- Idle resource-supply feedback and consecutive sample count:
- Mature supply fraction, completed evidence source, debt, and redundancy:
- Exploration reserve:
- Infrastructure retries (exit code 75 only):

Profiles without Praxist-managed discrete accelerators do not reserve a fixed
number of CPU cores. Praxist changes only total experiment concurrency from live
host pressure. Managed accelerator profiles may share a device
until measured/reserved memory or utilization fills it; unknown managed-device
demand is exclusive. Profiles reserve a whole submitted experiment's envelope across
setup, CPU, accelerator, and evaluation phases. `running_activity` may describe
the current observation, but it does not release that envelope. Do not add
implicit accelerator-to-CPU fallback.

## Natural Parallel Units

For every evaluator class, record its independent units (seeds, folds,
scenarios, simulator instances, datasets, benchmark cases, restart
trials, or equivalent), unit count, aggregation order, safe concurrency, and
resource shape. Ordinary non-evaluator commands stay outside the scheduler;
evaluator calls name an explicit measured profile whenever their invocation
path supports it.

Add a wide accelerator profile only when the selected backend can identify and
bind every assigned device and the evaluator deterministically distributes
independent units across them. Validate two units on distinct devices,
order-independent aggregation, process-group cleanup, and exact binding
handoff. For memory, I/O,
simulator, license, or external-service bottlenecks, record the measured safe
task count and enforce it inside the evaluator or conservative global scheduler
limit instead of inventing accelerator settings. Do not represent the same unit
both as a top-level scheduler job and an internal child.

For a coarse serial evaluator, record monotonic progress and define the smallest
independently valid retry unit. Add fail-fast only for consecutive identical
infrastructure or implementation errors that make remaining units non-runnable;
valid low scores and heterogeneous scientific failures still run to the protocol
boundary.

Directed resource-supply leases may wake an idle peer for at most one already
justified planned experiment. Follow the current research and exploration
priorities; do not create filler, relax evaluation, or launch after Closing
merely to consume capacity. A peer that has no justified work declines the lease
and enters a bounded same-priority exponential cooldown; a new experiment,
changed priority, or new generation resets the backoff. It may become eligible
again if later capacity and the research plan justify new work. The default
lease response window is 600 seconds. It limits when the existing plan may be
submitted, not how long an experiment admitted before expiry may run. Record
per-priority conversion and declined/expired/revoked/stale reasons rather than
treating offers as launches.

## Optional Managed NVIDIA/CUDA Handoff

Use this section only when the unchanged baseline uses and task initialization
selects the compatible Praxist-managed NVIDIA/CUDA backend. CPU-only,
unified-memory, task-managed, and other accelerator backends document their own
observed process handoff instead. For the compatible backend, test the full
process chain:

```text
Praxist scheduler -> public evaluator -> trainer -> worker/container
```

`PRAXIST_ASSIGNED_GPU_UUIDS` is authoritative. Every child must inherit the exact
ordered physical GPU UUID value through `CUDA_VISIBLE_DEVICES` and
`NVIDIA_VISIBLE_DEVICES`; framework-local `cuda:0` must never replace the
physical mask in a child environment. Record the UUID, multi-UUID,
missing-mask restoration, conflicting-mask rejection, standalone, and
forced-CPU test results. When at least two usable GPUs exist, also record the
bounded non-zero UUID parent/child CUDA check, driver-observed UUID, and cleanup
result. Do not substitute CPU-vs-accelerator timing or a first-device-only
check.

## Praxist Run Settings

- `compute_budget.per_experiment_gpu_hours` (zero/omitted unless applicable):
- `compute_budget.max_parallel_runs_per_peer`:
- `generation_policy.cohort_size`:
- `generation_policy.per_generation_hours`:
- `generation_policy.max_generations`:

## Research-Loop Controls

- DIG/QD: DIG is limited to absolute gen0 by default; QD remains independently
  enabled for gen0 DIG pools and as soft guidance in later PI synthesis.
- Gems reset: disabled by default for continuous evolution; enable only after
  an operator request or plateau diagnosis.
- Frontier lanes: confirmed, lower-admission durable incubator,
  task_candidate, diagnostic.
- Incubator policy: keep task-authorized, protocol-passed, non-suspect
  Pareto/new-high variants as long-term parents even when they are not clean
  confirmed winners.
- Mature evidence telemetry: when the user-approved contract uses ratios,
  canonical evaluator summaries emit `effort_ratio` and `coverage_ratio`;
  derive them from the task's mature reference effort and required coverage.
  Praxist projects them into auto-materialized findings; standalone findings without
  a canonical summary reference must carry them directly.
- Mature close gate: when the user-approved task contract distinguishes
  close-grade evidence from validation or diagnostic evidence, keep
  `synthesis_trigger.mature_quorum_fraction` positive (normally `0.25`).
  `mature_supply_fraction` only prioritizes evidence production and cannot
  substitute for this normal-close gate. Preserve an explicit user choice of
  `0.0` for information-density closing or no separate close-grade contract.
- Launch guard: set `evaluation.launch_guard.estimated_heavy_eval_minutes` from
  the most expensive ordinary evaluation and
  `estimated_close_grade_eval_minutes` from the measured p90 wall time of the
  task-authorized close protocol. With the launch guard enabled, generation
  close-out freezes new training/evaluation launches while already-started work
  drains naturally.
- Work classes: start enough complete `mature` evaluations early; use
  `ordinary` and `scout` for other work. Keep one exploration slot when such
  work is queued. Do not launch a no-checkpoint full run that cannot plausibly
  fit the remaining generation time.

## Blockers

Record missing data, evaluator, runtime, baseline, or hardware requirements
that block a real Praxist run.
