# Praxist Task Templates

Templates are tracked task projects. They are deliberately outside
`praxist/plugins`: system plugins live in `praxist/plugins`, while research tasks live
in external task repositories, ignored local `tasks/` workspaces, or this
tracked templates tree.

Evaluation strictness is task-owned too. Before adapting a template, record
which modes the user permits for launch, ranking, mature evidence, durable
parents, and generation close. The checked-in complete/partial policies are
demonstrations, not Praxist-wide restrictions; actual stage and effort/coverage
must remain visible.

Templates are replaceable scaffolds, not complete worked examples. Study the
complete projects under `examples/` when you need preserved evaluator, evidence,
and execution references. The canonical distinction is documented in
[`docs/guides/examples-and-templates.md`](../docs/guides/examples-and-templates.md).

## Task Projects

- `tasks/sam_optimizer/`: runnable SAM task reference template. It keeps the
  runnable task contract, roles, audits, evaluation, harness, literature pack,
  baselines, dataset metadata, and small regression fixtures, but excludes live
  run artifacts and reference-implementation corpora. Copy it to an external
  task project and provide its declared datasets before a real run.
- `tasks/machine_learning_template/`: recommended pure template for initializing
  general machine learning tasks. It demonstrates task-local prompt layout,
  evaluator evidence maturity, ML peer/PI roles, frontier lanes, generation-scoped DIG/QD,
  research memory, and continuous-evolution Gems defaults without embedding a
  benchmark-specific grader.
- `tasks/template/`: authoring scaffold with the same directory shape as the
  SAM task. Its `task.yaml` and `runner.py` stay offline-fixture compatible so
  contributors can smoke-test a new task before replacing the placeholder
  domain material. It remains the minimal scaffold/smoke reference.
- `tasks/toy_math/`: small non-ML smoke task with the same directory shape. It
  demonstrates that a task can be outside ML while still using the same Praxist
  startup, plugin resolution, budget, replay, and test path.

## Default Research-Loop Controls

Current templates include explicit task-local settings for generation-scoped
DIG, independent QD, and Gems:

These are illustrative defaults. Task initialization must first preserve the
user's explicit permissions for launch, ranking, maturity, parent use, Gems,
and close; it must not turn this template profile into Praxist-wide restrictions.

- `dig_lite.enabled: true` with `generation_scope: initial_only` turns on the
  pre-code Deep Innovation Gate only for absolute generation 0.
- `quality_diversity.initial_generation_enabled: true` applies QD to validated
  gen0 DIG pools; `later_generations_enabled: true` applies soft QD guidance to
  normal PI synthesis without rerunning DIG.
- `gems.enabled: false` is the default for template-style initialized tasks:
  run in continuous-evolution mode first, then enable periodic reset only after
  an operator request or diagnostic evidence of a performance ceiling.
- `evaluation.frontier_lanes` includes a strict confirmed lane, a
  lower-admission durable incubator lane, a lower-confidence candidate lane,
  and a diagnostic/control lane. The incubator is the long-term library for
  task-authorized, protocol-passed, non-suspect Pareto/new-high variants that
  should not disappear before clean confirmation.
- `synthesis_trigger.mature_quorum_fraction: 0.25` is the recommended template
  default when the task distinguishes close-grade evidence. The
  scheduler's mature supply target is advisory and cannot replace this gate.
  A user-approved task may use `0.0` when it intentionally chooses
  information-density closing or has no distinct close-grade evidence class.
- The template target lanes both accept the shared evaluator source label
  `performance`. Real task evaluators should emit that label for ordinary clean
  results authorized for durable selection, then let Praxist independently
  select confirmed and incubator; do not make every such result source-only
  `confirmed`.
- When a task explicitly enables Gems reset, `gems.max_gems_per_reset`,
  `gems.max_gems_total`, and `gems.prompt_max_gems` should remain compact,
  typically capped at 4.

`template` and `toy_math` keep `generation_policy.max_generations: 1` because
they are resolve-only smoke fixtures. They explicitly disable later-generation
QD and constructive next-generation feedback because those controls cannot take
effect in a one-generation run; their Gems block documents continuous evolution
with reset disabled. The `sam_optimizer` reference template uses
`max_generations: 8` and also defaults to continuous evolution; its task-owned
Gems thresholds and 6-generation cadence are retained only as an opt-in profile
for operators who explicitly enable periodic reset after plateau evidence.

## Required Task Layout

Every serious task project should keep this shape:

- `task.yaml`: canonical task contract. It declares the task id, research
  direction, metrics, budget defaults, workflow refs, model/runtime defaults,
  task-local roles, audits, evaluations, tools, optional stages, task-owned
  runtime boundaries, and task-owned output roots.
- `README.md`: operator-facing overview. Explain what the task is, how to run
  it, which assets are required, and which files a task author may edit.
- `description.md`: agent-facing research brief. Keep hard scope constraints,
  success criteria, and evaluation rules here or in structured `task.yaml`
  fields, not in Praxist core code.
- `prompt_task.jinja2`: task-owned Jinja prompt block consumed by the research
  loop. Keep task-specific instructions here rather than in Praxist core.
- `roles/<role>/role.yaml` and `roles/<role>/skill.md`: task-local role
  contracts. PI, Chair, peer, and reviewer behavior belongs here when it is
  task-specific.
- `audit_rules/<name>/audit.yaml`: declarative task-specific claim, scope, and
  agenda criteria. Do not require task authors to write Praxist audit framework code
  in the default path.
- `evaluations/<name>/evaluation.yaml` plus code: task-specific result scoring
  and metric interpretation. When agents need to run evaluation, expose one
  public command such as `evaluations/<name>/run.py`.
- `assets/harness/`: executable task harness, benchmarks, probes, and local
  adapters. These are internal task implementation details; prompt and role
  instructions should route agents through the public evaluation entrypoint.
- `assets/baselines/`: curated baseline summaries and small result ledgers.
  When baseline metrics are missing but locally measurable, task initialization
  should ask whether to create explicit zero placeholders or run a bounded
  `experiments/baseline_bench_*` measurement before writing these records.
- `assets/literature/`: prior-art packs used by peers, PI roles, and optional
  literature-scout workflows. These are task-owned context sources, not measured
  performance records.
- `assets/reference_implementations/`: optional compact reference variants or
  exemplar solutions. Do not put large run output here. The tracked SAM template
  intentionally omits this directory.
- `assets/regression_fixtures/`: small deterministic fixtures for tests.
- `.gitignore`: exclude raw datasets, runs, logs, caches, local secrets, and
  bulky generated artifacts.

Task descriptors should keep these runtime/output fields task-local:

- `task_entrypoints.evaluation`: the public evaluator peers should call.
- `runtime_outputs.root`: where task run artifacts live, usually
  `experiments`.
- `runtime_environment.venv` / `python` / `writable_roots`: optional task
  interpreter and task-owned writable paths.
- `runtime_environment.protected_child_paths`: task-owned directories that are
  readable/executable but protected from peer writes.
- `dig_lite`: task-local initial design-gate policy, including candidate-pool
  size, diversity cell fields, allowed/disallowed file rules, and write-gate
  behavior.
- `quality_diversity`: independent gen0 and later-generation QD switches plus
  cohort/PI allocation caps and task-owned label groups.
- `gems`: task-local periodic Gems policy, including reset interval, caps,
  lanes, metric keys, and task-specific evaluation-unit thresholds when the task
  has staged/full-coverage evaluation semantics.

Keep task-owned evaluator, trainer, config, and harness paths relative to the
task root; reserve verified absolute paths for external data, simulators, or
  environments. A runnable template or generated task must exercise its public
evaluator through the declared task interpreter from both the task root and a
run-like directory. When normal close requires complete evidence, calibrate the
complete evaluator's p90 runtime and require
`estimated_close_grade_eval_minutes * safety_factor` to fit before the earliest
close horizon with at least the documented drain margin. The close-grade value
describes the evaluator authorized by the task maturity contract; a longer,
optional heavy protocol may remain available without making normal close
unreachable.

## Literature, Scientific Database, And Open-Access Lookup

Templates include `tool_server:literature_lookup` in the active tool list by
default so Peers and PI memo agents can use no-key public scientific context
when it is useful; Chair agents inherit source-backed signals from PI memos and
shared evidence. The task-owned `assets/literature/` directory remains the
durable prior-art pack. A task can also declare a disabled `literature_scout`
role to show where search policy belongs. Keep `panel.optional_roles` entries
disabled unless a task-specific panel topology explicitly implements that
optional role execution path.

`tool_server:literature_lookup` uses no-key public sources such as arXiv,
OpenAlex, PubMed, Crossref, Semantic Scholar metadata, Europe PMC, UniProt, and
ClinicalTrials.gov where available. It can also fetch open-access text or PDF
provenance without bypassing paywalls. Its records are contextual literature or
database signals for hypothesis generation and prior-art screening. They must
not be copied into leaderboards, frontier promotion facts, or measured task
results. The tool being available does not perform network access by itself;
network calls happen only when an agent explicitly invokes a lookup tool.
If a source points to unavailable datasets, checkpoints, simulators, packages,
licenses, APIs, or runtime environments, agents must not download, install, or
provision them during the Praxist run. They should use the source to improve the
best solution possible under the task's current local assets, evaluator,
dependencies, and hardware, then record missing resources only as task-local
notes.

## Human-Readable Run Reports

Templates include `tool_server:run_report` in the active tool list. Praxist may
write derived Markdown reports under `<task>/docs/praxist_reports/` when a credible
above-baseline frontier signal appears, every 3 completed generations, and at
run completion. These reports summarize strongest variants/Pareto front,
strong-variant lineage, and run health for humans. They are derived views, not
canonical promotion facts.

## Optional Reviewer

Templates list `workflow_stage:reviewer_stub` as a disabled optional stage. When
an operator explicitly executes it in local/artifact review mode, it can produce
an audit-only reviewer report checking artifact hashes, trajectory references,
run summaries, and literature/database context roles. This reviewer does not
change generation scheduling or promotion state. It refuses to append to a run
after `run.finalized`, so replay verification for completed runs is not broken.

## Run Artifact Semantics

All templates follow the current artifact ownership model:

- Canonical current state lives in measured result/finding summaries,
  `frontier/frontier_manifest.json`, committed `gems/gems_state.json`, and
  generation boundary markers.
- The durable incubator lane is lower-admission than confirmed promotion. It
  keeps task-authorized, protocol-passed, non-suspect Pareto/new-high variants
  as long-term parents for repair, escalation, ablation, and validation. Mature
  parent lanes set `parent_eligible: true`; modes the task marks non-parentable
  and diagnostic lanes set it false. New Gems configuration uses
  `selection_policy: mature_evidence_top_k` with a task-derived
  `min_mature_eval_units`; staged tasks use `evidence_stage_min_units` for
  task-owned cumulative evaluation-unit thresholds.
- Validation candidates are compact non-frontier signals retained for follow-up.
  They can include task-defined preliminary, aligned, partial, repair,
  failed-but-informative, ablation, diagnostic, lower-stage, or
  late-after-boundary evidence. They should be published through
  structured findings/result summaries and task-owned lanes, not hand-written
  side files. Under the recommended expensive-task default, preliminary
  evidence is triage only and near-complete-coverage aligned evidence can
  prioritize follow-up. An explicit user protocol may assign different
  authority while preserving the actual stage, effort, and coverage.
  Late-after-boundary result summaries are preserved for future review, but
  templates should treat them as revalidation leads rather than clean promotion
  parents.
- Leaderboards, PI evidence packs, prompt layout manifests, diagnostics, and
  `docs/praxist_reports/` Markdown reports are derived views or audit snapshots.
  They are useful for operator inspection and behavior analysis, but they must
  not become independent sources of truth.
- Peers and task harnesses should publish structured findings and result
  summaries. They must not hand-write Praxist frontier, Gems, prompt-layout,
  research-memory, or diagnostic state.
- Resume/control logic should ignore or crop partial `.tmp`, `.candidate`,
  `.rejected`, or incomplete final-generation outputs before continuing a run.

## Authoring Standard

- Start from an external task root with `praxist start`, or select it with
  `--task-path`. Tracked templates are authoring and smoke-test fixtures; copy or
  adapt one into an external task project before a real run so run artifacts do
  not enter the Praxist source checkout. Templates never rely on an implicit
  bundled-task fallback.
- A task may select `agent_runtime:codex_sdk` with native OpenAI. Authentication
  remains operator-owned: use `OPENAI_API_KEY`, or a saved ChatGPT login when
  no API key is exported. Never put either credential in a task template.
- A task must not require files under `praxist/plugins` except generic system
  plugins that it references in `task.yaml`.
- Task-local roles, audits, evaluations, budget profiles, harnesses, datasets,
  literature, and optional reference implementations stay inside the task repo.
- Paths inside task files should be relative to the task project or use
  `$PRAXIST_TASK_PROJECT_PATH`; they should not assume the task is located at
  `tasks/<name>` inside the Praxist source tree.
- Raw datasets and large experiment outputs are external. Commit metadata,
  resolvers, small fixtures, and curated summaries instead.

## Test Standard

Minimum checks for a task template:

```bash
RUN_ROOT=$(mktemp -d /tmp/praxist-toy-math-XXXXXX)
praxist resolve templates/tasks/toy_math --run-dir "$RUN_ROOT/run"
```

For the full SAM reference template, provide a real provider credential and use a
resolve-only smoke first:

```bash
RUN_ROOT=$(mktemp -d /tmp/praxist-sam-resolve-XXXXXX)
DEEPSEEK_API_KEY=... praxist resolve templates/tasks/sam_optimizer \
  --run-dir "$RUN_ROOT/run" \
  --model-provider model_provider:deepseek_alias \
  --runtime agent_runtime:claude_sdk \
  --model deepseek-v4-pro
```

Task repositories should add their own tests under `tests/` when they need
task-owned harness checks, metric parsing checks, or regression fixtures.
