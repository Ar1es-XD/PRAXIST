# Minimal Task Project Template

This directory is the smallest full-layout scaffold for a new external Praxist
task project. It is intentionally domain-neutral and resolve-only: the fake
workflow runner supports startup, plugin-resolution, budget-ledger, and replay
tests, while the placeholder evaluator refuses real scoring until it is
replaced.

## What To Replace

Do not add provider cache, prompt-compression, or session-batching fields while
adapting this template. Praxist owns those runtime concerns: Codex-native mode
and OpenRouter routes use lossless finding-event coalescing, while
direct DeepSeek behavior is unchanged and all canonical artifacts remain
available.

- `task.yaml`: replace task id, research direction, metrics, budget defaults,
  workflow refs, roles, audits, evaluations, tools, and model/runtime defaults.
- `description.md`: replace the agent-facing research brief and hard scope
  constraints.
- `prompt_task.jinja2`: replace the task-owned prompt block with instructions
  for this task, or remove it when structured prompt fragments are sufficient.
- `roles/`: replace role contracts and private KB files with task-owned roles.
  `roles/literature_scout/` is optional and disabled by default; keep its
  domain query policy here, not in Praxist core.
- `audit_rules/`: replace declarative scope, conclusion, and agenda criteria.
- `evaluations/`: replace metric interpretation, scoring code, and the
  placeholder public evaluator `evaluations/pareto_tiered/run.py`.
- `assets/harness/`: replace benchmark, simulation, or execution scripts.
  Keep these internal; route agents through the public evaluator.
- `assets/baselines/`: replace curated baseline summaries and small result
  ledgers.
- `assets/literature/`: replace prior-art packs.
- `tool_server:literature_lookup` is active by default, but literature/database
  signals follow a current-environment-only policy: do not download new data or
  install dependencies from search results; adapt ideas to this task's existing
  local assets.
- `tool_server:run_report` is active by default for human-readable derived
  reports under `docs/praxist_reports/`. It summarizes run evidence for operators
  and must not be treated as canonical promotion truth.
- `assets/reference_implementations/`: replace exemplar solutions.
- `assets/regression_fixtures/`: replace small deterministic fixtures.

## Smoke Test

```bash
RUN_ROOT=$(mktemp -d /tmp/praxist-template-XXXXXX)
praxist resolve templates/tasks/template --run-dir "$RUN_ROOT/run"
```

When this scaffold is adapted to a real evaluator with a required ratio gate,
also pass one file produced by its canonical summary writer through
`--result-summary`; do not validate a hand-written substitute.

Before an adapted evaluator fans out expensive work, exercise its actual
task-appropriate build/load/startup boundary, validate the public CLI, function,
RPC, simulator, container, notebook, or service interface it really uses, and
run a task-defined **one-unit canary** through the same public evaluator,
scheduler path when applicable, and canonical summary writer. Validate the resulting summary and finding
projection. The canary tests wiring, not score quality, and does not define a
universal seed, epoch, split, iteration, or hardware rule. If this task chooses
an independently trusted evaluator, its task-owned verifier must also reject
altered or unattested results; do not add that requirement to ordinary
peer-authored evaluators.

The template is not a real research task until its task-owned files are
replaced. Its checked-in evaluator exits with an explanatory error by design.
Do not put task-specific logic in Praxist core or `praxist/plugins`.

## Research-Loop Defaults To Keep

Before keeping any protocol default, write down the task owner's protocol
intent: which evaluator modes may launch, rank, count as mature, supply parents,
and satisfy close. The checked-in full/partial split is a demonstration, not a
Praxist-wide restriction. An explicit user-selected reduced or partial protocol
is valid when evaluator metadata and every downstream policy describe it
consistently. Never infer a launch violation from words in a command or path.

`task.yaml` includes current generation-scoped DIG and independent QD blocks, an explicit
continuous-evolution Gems policy, and a demonstrative multi-lane frontier:

- `dig_lite.enabled: true` with `generation_scope: initial_only`
- `quality_diversity.initial_generation_enabled: true`
- `quality_diversity.later_generations_enabled: false` because this smoke
  fixture has no later generation
- `evaluation.constructive_peer_mix_enabled: false` because no next generation
  can consume the advisory feedback
- optional `evaluation.frontier_lanes` showing confirmed, lower-admission
  durable incubator, task-candidate, and diagnostic evidence streams; these are
  not a universal initialization default
- `gems.enabled: false` for the task-initialization default
- task-owned `min_mature_eval_units` and `evidence_stage_min_units` placeholders
- `synthesis_trigger.mature_quorum_fraction: 0.25` so raw progress findings
  cannot normal-close a generation before task-defined mature evidence exists

When QD is enabled, PI/Chair contracts use `planned_dimensions`; findings use
`design_dimensions` for what was actually implemented. Keep these separate so
planned and realized HHI remain meaningful.

The mature supply target is advisory scheduler policy, not a close gate.
Generated tasks should retain a positive close quorum when the user-approved
contract distinguishes close-grade evidence from validation or diagnostic
evidence. Preserve `0.0` when the user explicitly chooses
information-density closing or no separate close-grade class.

The scaffold keeps `generation_policy.max_generations: 1` only because it is a
smoke fixture. For real research, replace the placeholder diversity dimensions
and keep only frontier lanes justified by task evidence. A cheap
single-protocol task must not retain demonstrative lanes merely to match this
file. Enable periodic Gems reset
later only when the operator requests it or a diagnostic pass identifies a
performance ceiling and recommends a reset cadence.

## Artifact Ownership

Generated tasks should keep Praxist artifact ownership intact: peers write task
result summaries and structured findings, while Praxist owns frontier/incubator
state, Gems state, prompt-layout artifacts, PI evidence packs, and generation
boundaries. Treat leaderboards, PI packs, prompts, diagnostics, and
`docs/praxist_reports/` reports as derived views or audit snapshots, not as
independent sources of current truth. Praxist excludes that exact generated
report subtree from task identity; other task documentation remains
identity-bearing.
In a real adaptation, the evaluator should publish ordinary complete
protocol-passed non-suspect results through the shared `performance` source so
confirmed and the lower-admission durable incubator can select independently.
Their target lanes explicitly set `parent_eligible: true`. Promising
preliminary, aligned, partial, repair,
diagnostic, lower-tier, late-after-boundary, or failed-but-informative evidence
should be published as structured validation signals in lanes with
`parent_eligible: false`. Praxist may surface those signals to later peers and DIG
for follow-up, but they are not durable parents until complete canonical
evidence revalidates them. A real task owns its stage labels through
`complete_stage_labels` and `preliminary_stage_labels`; maturity should normally
be gated by explicit `effort_ratio` and `coverage_ratio`, not a universal tier
name or fixed training count.

If launch-time settings can change the treatment independently of variant
code, keep their complete secret-free values in the existing result summary as
top-level `effective_config` plus `effective_config_complete`. Record resolved
values after defaults and parsing, so an omitted default and the same explicit
default have one canonical representation. A claimed exact
replication must also publish the parent digest under
`replication_of_effective_config_sha256`; otherwise call it a control or
code-only rerun. These fields do not alter ordinary promotion or maturity.

Central submissions are idempotent by stable semantic tag. If a failed or
rejected launch is corrected, use the same tag with `--retry-terminal`; do not
rename the scientific experiment to bypass terminal state.

Result summaries may be nested under `results/**/` and named `summary.json`,
`evaluation_summary.json`, `eval_summary.json`, `tiered_eval_summary.json`, or
`custom_*_tiered_eval_summary.json` (`result_summary.json` remains a
compatibility name). Praxist materializes lane, maturity, protocol, ratio, and
diagnostic metadata into structured findings.
