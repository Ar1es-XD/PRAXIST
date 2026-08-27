# General Machine Learning Task Template

This is the recommended reference template for creating a general machine
learning Praxist task project. It is a pure template: do not run it as a real
research task until the placeholders have been replaced with task-owned data
metadata, evaluator code, baseline evidence, and resource planning.

## Use This For

- supervised or self-supervised ML projects with a measurable prediction or
  model-output artifact;
- projects that already have runnable code and a credible evaluator or
  validation protocol;
- task initialization through the `praxist-task-initialization` skill.

Context efficiency stays outside this template. Praxist automatically applies
lossless finding-event coalescing to Codex-native mode and
OpenRouter runs, while direct DeepSeek runs keep their existing behavior. Do
not add task-local prompt compression, cache routing, or a second memory store.

Use `templates/tasks/template` when you only need the smallest scaffold/smoke
shape. Use `templates/tasks/sam_optimizer` when you need a rich domain-specific
reference.

## Replace Before Real Runs

- `task.yaml`: replace task id, research direction, metric, budget, runtime
  paths, diversity axes, frontier lanes, and plugin refs if needed. If the
  domain ranks by confidence/lower-bound, variance, risk, or Pareto tradeoffs,
  encode those robust metrics or axes in the frontier lanes.
- `description.md`: write the real ML objective, data boundary, prediction
  artifact format, and success criteria.
- `prompt_task.jinja2`: replace the generic ML task block with project facts.
- `assets/task_context/context_template.json`: fill data schemas, artifact
  contract, evaluator contract, staged evaluation protocols, and leakage
  boundaries.
- `assets/resource_plan.md`: record hardware, bottleneck, expected experiment
  cost, preliminary versus aligned cost, aligned coverage ratio
  and reduced training budget, complete expected budget/integrity fields, cohort
  size, generation duration, gen0 QD-DIG, later-generation independent QD, and
  Gems policy.
  Only for tasks that detect and select the Praxist-managed NVIDIA/CUDA backend,
  also record the authoritative `PRAXIST_ASSIGNED_GPU_UUIDS`
  handoff through evaluator/trainer/worker boundaries, contract-test results,
  and a bounded non-zero physical UUID verification when hardware permits.
- `evaluations/primary/run.py`: replace the placeholder with a real public
  evaluator entrypoint.
- `assets/baselines/`: add measured or clearly labeled baseline records.
- `roles/`: adjust role language only where the project needs domain expertise.
- `.praxist/plugins/panel_topologies/machine_learning_template/plugin.yaml`:
  copy or regenerate this file when the new `task.yaml` still references
  `panel_topology:machine_learning_template`; otherwise change the topology ref
  to a known core topology and document the choice.
- `synthesis_trigger`: keep this block compatible with the selected run profile.
  Short smoke profiles need short intervals; real runs should size intervals and
  contributing-peer thresholds from `assets/resource_plan.md`. Keep a positive
  mature close quorum whenever the user-approved task contract distinguishes
  close-grade evidence from validation or diagnostic evidence; an explicit
  information-density policy may instead use zero. The mature supply target
  alone does not gate normal close. Derive the minimum interval from the earliest credible
  assessment point and the maximum interval from the full implementation,
  queue, evaluation, publication, and drain lifecycle. Use resolve-only checks
  to validate profile-specific bounds against the current Praxist task-spec
  rules.

## Prompt Layout

This template demonstrates task-local prompt layout:

- `prompt_base.jinja2`: stable ML peer workflow and evidence discipline.
- `prompt_generation.jinja2`: generation-specific PI agenda and frontier
  context handling.
- `prompt_task.jinja2`: task facts, evaluator contract, and first actions.

Keeping these separate helps Praxist preserve stable prompt cache boundaries while
still letting the task own its ML-specific instructions.

Before expensive evaluator fan-out, exercise the actual task-appropriate
build/load/startup boundary, validate the evaluator's real public invocation
interface, and run a task-defined **one-unit canary** through the public
evaluator, scheduler path when applicable, and canonical summary writer.
Validate both the summary and its finding projection. The
smallest valid unit is owned by the task; this template does not impose a seed,
epoch, split, iteration, or hardware convention. A valid low score remains
scientific evidence rather than a wiring failure.

The template also explains Praxist-maintained research memory, gen0 QD-DIG,
later-generation independent QD, Gems, and frontier lanes as context channels
rather than task-owned files for peers to mutate.

QD plans use `peer_contracts[].planned_dimensions`; evaluated findings use
`design_dimensions` for the implementation that actually ran. Diagnostics
compare their HHI separately and treat missing realized labels as missing data.

Praxist-maintained context channels are not all fact owners. Measured result
summaries, structured findings, frontier/incubator state, committed Gems state,
and generation boundary markers own current facts. Leaderboards, PI evidence
packs, rendered prompts, prompt-layout manifests, diagnostics, and behavior
reports are derived views or audit snapshots. Use them to understand what an
agent saw, not to override evaluator evidence or promotion state. The generated
`docs/praxist_reports/` subtree is excluded from task identity, while ordinary
task documentation remains identity-bearing.
Preliminary, aligned, partial, repair, diagnostic, ablation, failed,
lower-stage, or late-after-boundary records may
still be valuable validation signals. The evaluator should publish ordinary
task-authorized, protocol-passed, non-suspect results through the shared
`performance` source accepted by confirmed and the lower-admission durable
incubator; Praxist then chooses the final targets independently. Publish modes
the task marks lower-confidence through `task_candidate` or diagnostic lanes so
Praxist can show them compactly to later peers, DIG, PI, and diagnostics without
promoting them to durable confirmed facts.
Derive `min_mature_eval_units` and `evidence_stage_min_units` from this task's
own parent-authorized and staged evaluator protocols; never copy another task's
counts.

The default tool set includes literature/database lookup for scientific context.
If a source requires unavailable datasets, checkpoints, packages, licenses,
hardware, or runtime environments, do not acquire them during the Praxist run.
Translate the idea into a variant that works with the task's existing local
dataset metadata, evaluator, dependencies, and hardware, and record the missing
resource only as a task-local note.

The task-local Multi-PI topology is part of the template contract. A generated
task that keeps `panel_topology:machine_learning_template` must carry the
matching `.praxist/plugins/panel_topologies/machine_learning_template`
plugin directory.

## Evidence Maturity

First replace the template's protocol defaults with the user-approved intent:
for each mode, state whether it may launch, rank, count as mature, become a
parent, or satisfy close. Full/partial is not a global Praxist policy. A reduced
or partial mode may be authoritative when the user explicitly chooses it, but
its actual effort, coverage, and stage must stay visible throughout the task.
Only undeclared drift is treated as insufficient integrity.

When the user-approved maturity policy uses ratios, task adaptations emit
`effort_ratio` and `coverage_ratio`, use `require_ratio_gate: true`, and list their task-owned labels in
`complete_stage_labels` / `preliminary_stage_labels`. Literal names do not grant
maturity. Confirmed and incubator lanes set `parent_eligible: true`; candidate
and diagnostic lanes set it false.

Before launch, execute the shortest valid scored evaluator path using the real
summary writer, then run `praxist resolve <task_path> --result-summary
<summary_path>`. The task is not launch-ready with `require_ratio_gate: true`
unless Praxist extracts finite `effort_ratio` and `coverage_ratio` from that
actual output.

The default frontier lanes are:

- `confirmed`: complete-evaluation prediction artifacts with complete evaluator
  evidence.
- `incubator`: lower-admission durable long-term library for complete
  protocol-passed non-suspect Pareto/new-high prediction artifacts that need
  follow-up before clean confirmation.
- `task_candidate`: preliminary, aligned, partial,
  repair, or promising immature evidence.
- `diagnostic`: controls, invalid-output diagnostics, negative evidence, and
  process observations.

Before a real run, add a task-local lane-routing regression that invokes the
evaluator summary builder and proves every parent-eligible target lane is reachable.
Include more than the confirmed top-k of parent-authorized fixtures. When the adapted
task has a justified distinct incubator axis, include a Pareto point outside
that top-k and prove it remains incubator-eligible; do not invent an axis for a
single-metric task. Fixtures from modes marked non-parentable by the
user-owned intent, plus suspect fixtures, must not become parents.

Peers should publish raw evaluator metrics unchanged. A result that satisfies
the user-declared mature protocol can set `scored_complete=true`; if that
protocol is reduced, its actual stage, effort, and coverage must remain visible.
Non-parentable evidence should remain in candidate or diagnostic lanes.

The adapted evaluator must also capture result-affecting launch arguments,
environment overrides, and task-local config values in one secret-free
top-level `effective_config` object whenever code alone does not identify the
treatment. Store evaluator-resolved values after defaults and parsing, not a
raw environment snapshot; omitted and explicit forms of the same default must
hash identically. Set `effective_config_complete` explicitly. Exact replications add
the selected parent's `replication_of_effective_config_sha256` and may use that
label only after Praxist reports a match; missing metadata does not block an
ordinary experiment.

Write compact summaries recursively under `results/**/` as `summary.json`,
`evaluation_summary.json`, `eval_summary.json`, `tiered_eval_summary.json`, or
`custom_*_tiered_eval_summary.json`; `result_summary.json` is accepted for
compatibility. Put lane, maturity, effort/coverage ratio, protocol, and
diagnostic metadata in structured fields so Praxist can materialize it.

For expensive ML tasks, the recommended default does not use a tiny or
arbitrary low-effort preliminary check as a proxy for complete protocol
performance. Unless the user's protocol intent explicitly says otherwise, keep
two early stages distinct:

- preliminary: the smallest safe evaluator check for wiring, impossible-idea
  filtering, and failure diagnosis. By default it is context rather than
  ranking, parent-selection, Gems, or confirmed-frontier evidence.
- aligned: a fixed intermediate protocol
  that uses the same evaluator path, primary metric direction, aggregation,
  invalid-result rules, and leakage checks as complete evaluation. It should keep
  the same or near-complete data/evaluation coverage as complete evaluation: the same
  folds, datasets, seeds, episodes, scenarios, or other evaluation units
  whenever possible. Save
  compute mainly by reducing fixed training or optimization budget, such as
  epochs, gradient steps, rollout horizon, or repeated restarts. If evaluation-unit
  coverage must be reduced, store the coverage ratio, stratification, hard-case
  inclusion, omitted-unit rationale, and budget in task assets; otherwise label
  the result partial rather than aligned. Use this stage as the only
  early score for prioritizing variants before complete evaluation.

Reports should compare preliminary-to-complete and aligned-to-complete
calibration separately. Weak preliminary correlation is expected; weak aligned
correlation means the task protocol needs redesign before Praxist
trusts early rankings.

Do not silently define a budget as complete evidence when it is materially
below project evidence, the task's reference effort, or measured convergence
behavior. Record explicit task-owner approval or downgrade it to aligned
evidence and define an adequate complete protocol. Do not apply a universal
epoch, step, rollout, or wall-clock threshold.

## Resolve-Only Check

The checked-in template should resolve as a task project. After replacing
placeholders enough for your project, run the same resolve-only check:

```bash
RUN_ROOT=$(mktemp -d /tmp/praxist-ml-template-XXXXXX)
praxist resolve templates/tasks/machine_learning_template --run-dir "$RUN_ROOT/run"
```

The checked-in evaluator exits with an explanatory error by design.
