# Task Project Templates

This directory contains task-project templates. Each subdirectory is a
self-contained authoring or smoke-test fixture that can be passed to Praxist
with `--task-path`. For a real run, first copy or adapt the template into an
external task project: Praxist intentionally refuses to place run artifacts
inside its own source checkout. In an external task root, `praxist start`
defaults the task path to the current directory.

These are scaffolds, not complete examples. Finished runnable reference projects
live under `examples/`; see
[`Examples And Templates`](../../docs/guides/examples-and-templates.md) for the
single canonical distinction.

Adapt each template to the user's explicit protocol intent. Partial, scout,
reduced, or complete modes may have different authorized uses; preserve their
actual stage and effort/coverage, and prevent only undeclared drift from
claiming a stronger evidence class. No template defines a global full-only rule.

Provider-specific context efficiency is a Praxist runtime concern, not part of
a task-template contract. Codex-native mode and OpenRouter runs
automatically coalesce finding-only continuation wakeups while retaining every
canonical artifact; control and resource events remain immediate. Direct
DeepSeek runs preserve their established event timing. Do not add prompt
compression, duplicate memory stores, cache-routing keys, or session intervals
to copied `task.yaml` files or task prompts.

## Directories

- `sam_optimizer/`: runnable SAM optimizer reference template with a tiered
  PyTorch evaluator, curated task assets, and explicit accelerator handoff.
  Copy it outside the Praxist checkout before a real run and provide the
  declared datasets.
- `machine_learning_template/`: recommended pure template for general machine
  learning task authoring. It demonstrates task-local base/generation/task
  prompts, ML role vocabulary, evaluator evidence maturity, frontier lanes,
  generation-scoped DIG/QD, research memory, and Multi-PI without copying any benchmark-specific
  grader or campaign semantics. Replace its placeholders before real runs.
- `template/`: minimal full-layout scaffold for creating a new task. Replace the
  placeholder task contract, role skills, audit rules, evaluations, harness
  code, literature pack, baselines, and regression fixtures before using it for
  real research. It remains the minimal scaffold/smoke reference.
- `toy_math/`: offline fixture template with the same full layout. It keeps a tiny
  deterministic runner for tests but includes the same task-owned directories a
  real task should provide.

## File Responsibilities

- `task.yaml`: source of truth for task id, metrics, budget defaults, workflow
  refs, plugin refs, and task-local components. Current task descriptors should
  also declare `schema_version`, `task_version`, `task_entrypoints.evaluation`,
  `runtime_outputs`, and any task-owned `runtime_environment` paths.
- `README.md`: operator and contributor guide.
- `description.md`: agent-facing research brief.
- `prompt_task.jinja2`: task-owned prompt block for research-loop tasks.
- `prompt_base.jinja2` / `prompt_generation.jinja2`: optional task-local prompt
  layout overrides. The general ML template demonstrates these for tasks that
  need stronger task-owned prompt framing.
- `.praxist/plugins/panel_topologies/<name>/plugin.yaml`: optional
  task-local panel topology. Include it whenever `task.yaml` references
  `panel_topology:<name>` and `<name>` is not a core bundled topology.
- `roles/`: task-local peer, PI, Chair, reviewer, or specialist roles. Praxist
  injects each peer's agenda-assigned RoleSkill into its prompt and records the
  effective reference and content hash on its runtime request.
- `audit_rules/`: task-local scope, claim, and agenda criteria. These are
  declarative text files by default, not Python audit hooks.
- `evaluations/`: task-local metric and Pareto interpretation. Executable
  tasks should expose a single public evaluator such as
  `evaluations/<name>/run.py`.
- `assets/harness/`: executable benchmark or simulation harness used by the
  evaluator. Agents should not rely on internal harness paths as the stable
  task interface.
- `assets/baselines/`: curated small baseline evidence.
- `assets/literature/`: prior-art packs and reading material.
- `assets/reference_implementations/`: optional compact known variants or
  exemplars. The tracked SAM reference template intentionally omits this directory.
- `assets/regression_fixtures/`: small fixtures used by tests.

## Runtime Boundary Fields

Task-owned runtime fields keep Praxist generic while giving peers the correct
execution boundary:

- `runtime_environment.venv` / `runtime_environment.python`: optional
  task-local interpreter paths. Praxist injects `PRAXIST_TASK_VENV`, `VIRTUAL_ENV`,
  and `PRAXIST_TASK_PYTHON` when present.
- `runtime_environment.writable_roots`: optional task-owned paths such as
  `.venv` or `scratch` that peers may create or update.
- `runtime_environment.protected_child_paths`: task-owned source, assets,
  evaluator, or audit directories that peers may read and execute but must not
  overwrite.
- `runtime_outputs.root`: task-owned run artifact root, normally
  `experiments`.

Internal evaluator, trainer, config, and harness references should be
task-root-relative. Verify intentional external absolute paths before launch,
and smoke the evaluator through the declared task interpreter from both the
task root and a run-like directory. If normal close requires mature/complete
evidence, the task-authorized close evaluator's observed p90 runtime times its
safety factor must fit inside the earliest effective close horizon after the
drain margin; a longer optional heavy protocol may remain separate.

Every adapted template must also pass the task-initialization evaluator fan-out
preflight: its task-appropriate build/load/startup check, validation of its real
public invocation interface, a task-defined one-unit canary through the public
evaluator and scheduler path when applicable, and canonical summary plus
finding-projection validation. This is a wiring check,
not a universal seed, epoch, split, iteration, or hardware convention. Only a
task that explicitly claims independently trusted evidence adds its own
attestation/tamper check; normal peer-authored evaluators remain valid.

## DIG, QD, And Gems Defaults

Every tracked task descriptor now shows the standard research-loop controls:

- `dig_lite.enabled: true` with `generation_scope: initial_only` for pre-code
  design gating only at absolute gen0.
- `quality_diversity.initial_generation_enabled: true` for gen0 DIG-pool QD and
  `later_generations_enabled: true` for prompt-guided QD in later PI synthesis
  without DIG. One-generation smoke fixtures set the later switch to `false`.
- QD-enabled PI/Chair agendas record intended task axes under
  `peer_contracts[].planned_dimensions`. Findings record the actual implemented
  values under `design_dimensions`. Templates and generated tasks must not copy
  planned values into missing result evidence; diagnostics compare planned and
  realized HHI separately.
- `evaluation.frontier_lanes` for task-owned confirmed, lower-admission
  durable incubator, candidate, and diagnostic/control evidence streams. This
  is what materializes incubator-style long-term variant libraries in
  `frontier/frontier_manifest.json`.
- Evaluator lane fields are source labels, while configured frontier lanes are
  Praxist-selected targets. The templates use `performance` as the shared source for
  ordinary parent-authorized results so confirmed and incubator are both
  reachable; generated tasks must regression-test this contract rather than
  relying on prompt wording.
- One exact immutable result artifact consumes one durable lane slot even when
  several aliases reference it; different paths or hashes remain independent
  replications. Every metric used for ordering or human-report winner claims
  must also have an explicit task-owned direction.
- `evaluation.maturity_policy` for generic mature-evidence thresholds using
  evaluator-emitted `effort_ratio` and `coverage_ratio`.
- `evaluation.constructive_peer_mix_enabled` and
  `evaluation.constructive_target_ratio` for switchable advisory feedback when
  too few peers produce constructive solution candidates. One-generation smoke
  fixtures disable it because there is no next generation to receive feedback.
- `evaluation.launch_guard` and
  `synthesis_trigger.mature_quorum_fraction: 0.25` so task-defined mature
  evidence gates normal close while close-out freezes new
  training/evaluation launches and lets existing work drain. The separate
  `mature_supply_fraction` only prioritizes evidence production. Use a `0.0`
  close quorum only for a task that intentionally has no distinct close-grade
  evidence contract and after explicit operator confirmation.
- `compute_budget.resource_scheduler` starts as either a deliberately
  uncalibrated `legacy` placeholder or a measured `central` policy. The general
  ML authoring template keeps `mode: legacy` until task initialization observes
  the unchanged baseline. Templates with measured task-owned profiles show the
  central form. CPU work is controlled by host-wide experiment count, not
  per-job core reservations; a selected managed discrete-accelerator backend
  may additionally describe measured memory/utilization, and unknown demand
  remains exclusive.
- Resource calibration uses a timestamped process-lifetime trace. For short
  work on a selected observable accelerator backend, sample frequently enough
  to cover the process lifetime. Very short tasks require safe identical
  repetitions or remain unknown/exclusive; one teardown `0%` snapshot is not a
  valid demand estimate.
- Scheduler `running` is a process-group lifecycle count, while
  `running_activity` is point-in-time resource telemetry. Any declared
  accelerator envelope remains reserved across phase changes; a low sample
  does not authorize transient oversubscription.
- `mature_supply_fraction: 0.25` and `mature_supply_redundancy: 3.0` provide a
  bounded mature-evidence debt target. When the selected backend exposes
  separate compute and memory pressure, keep them as independent admission
  dimensions; spare capacity returns to Pareto follow-up and planned
  preliminary work after mature commitments are supplied.
- `supply_lease_seconds: 600` gives a woken peer a bounded response window.
  Expiry limits submission, not the runtime of an already admitted experiment;
  conversion and terminal reasons remain visible in scheduler status.
- `mature_assessment_min_completion_probability: 0.25` lets assessment-stage
  mature top-ups use the scheduler's compact calibrated wall-time model.
- Evaluation prompts submit through `protected_pids launch` with a stable
  semantic tag, declared profile, and `scout|ordinary|mature` work class. The
  central scheduler owns process creation, accelerator visibility, retry, and
  release; raw shell backgrounding and manual device selection are not
  supported. A corrected failed/rejected request retains its tag and uses
  `--retry-terminal`; ordinary duplicates remain idempotent. Only tasks that
  explicitly select the Praxist-managed NVIDIA/CUDA
  backend use its UUID handoff contract. CPU-only execution, macOS
  unified-memory execution, other unified-memory systems, task-managed
  accelerators, and other backends are normal paths and validate their own
  observed process handoff.
- The default profile matches the public evaluator's normal resource shape;
  ordinary analysis stays outside the experiment queue and evaluator launches
  name explicit profiles where supported. Event-driven resource
  supply feedback wakes idle peers only for already planned work, and full
  evaluators declare wide profiles only after their natural seeds/folds/
  scenarios or equivalent units actually use every assigned device.
- `gems.enabled: false` for continuous-evolution task-initialization defaults.
- compact Gems caps of 4 per reset, 4 total, and 4 visible in prompts are
  declared as an opt-in reset profile; they stay inactive while
  `gems.enabled: false`.

For smoke fixtures (`template`, `toy_math`), `max_generations: 1` deliberately
keeps the default offline smoke short. For general ML task initialization, use
`machine_learning_template` as the stronger authoring reference: keep gen0 DIG and independent QD
on, start with Gems reset disabled, and enable periodic Gems only after an
operator request or a diagnostic pass identifies a performance ceiling and
recommends a reset cadence.

For expensive tasks whose user-owned intent leaves the protocol open, task
initialization should not let the cheapest preliminary check represent the
complete protocol. The recommended default uses three semantic evidence levels
when needed:

- preliminary: minimal wiring/failure triage, visible as a validation signal
  but not ranking or promotion evidence under the default policy.
- aligned: fixed intermediate protocol
  with the same evaluator path, metric direction, aggregation, invalid-result
  rules, leakage checks, and same or near-complete data/evaluation coverage as complete
  evaluation. Save compute mainly by reducing training or optimization budget,
  not by using a small convenience subset of evaluation units. Under the
  default policy this is the only early score that prioritizes variants before
  complete evaluation.
- complete mature evaluation: the default clean parent-promotion gate.

An explicit user protocol may assign those permissions differently. Keep the
actual stage, effort, and coverage visible and apply one decision consistently
to ranking, lanes, Gems, parent use, and close.

Tasks choose their own literal stage labels and list them in
`preliminary_stage_labels` / `complete_stage_labels`. When their user-approved
maturity policy uses ratios, evaluators emit `effort_ratio` and
`coverage_ratio` in the canonical result summary and use
`require_ratio_gate: true`; Praxist preserves the normalized ratios in the
auto-materialized finding, so the task need not duplicate them. Labels alone
do not create global maturity.

Keep public completion metadata coherent with that task-owned policy. Do not
reject a `capped` label by itself: reaching a declared terminal budget may be a
successful mature result, while an early cap must remain incomplete. Test both
through the real summary writer and follow the canonical-summary contract in
[`docs/guides/task-projects.md`](../../docs/guides/task-projects.md).

After adapting an evaluator, run its shortest valid scored path through the
real summary writer and check the resulting file with `praxist resolve
<task_path> --result-summary <summary_path>`. This verifies output telemetry;
it does not require task stage labels or certify the measured performance.

Incubator lanes should set `admit_new_high: true`, `parent_eligible: true`, and use task-owned Pareto axes
from distinct metric families. This keeps the long-term variant library compact
while preserving mature candidates that improve any meaningful axis.
Before launch, exercise more than the confirmed per-generation cap of
parent-authorized fixtures. For tasks with a justified distinct incubator axis, verify that a
non-dominated candidate outside confirmed top-k can still reach incubator;
single-metric tasks should not invent an axis for this check. Fixtures from
modes marked non-parentable by the task intent, as well as protocol-failed and
suspect fixtures, must remain non-parentable. A reduced, partial, or scout mode
explicitly authorized for parent use is instead a positive fixture and must
retain its true stage, effort, and coverage metadata.

Short smoke profiles should also carry compatible `synthesis_trigger` intervals
instead of depending on research-run defaults. Real research profiles should
derive the minimum interval from the earliest credible assessment point and the
maximum interval from the full implementation, queue, complete-evaluation,
publication, and drain lifecycle. They should derive contributing-peer
thresholds from the task resource plan. Use resolve-only checks and lightweight
closing-policy fixtures to validate smoke profiles against the current Praxist
task-spec rules.

For staged or full-coverage tasks, set task-owned Gems maturity thresholds
instead of copying another task's values. New templates use
`selection_policy: mature_evidence_top_k` and set `min_mature_eval_units` to
the number of units required by the protocol authorized for Gems and parent
use, normally the complete protocol. Use
`evidence_stage_min_units` when task-owned stage labels need cumulative unit
thresholds; Praxist configuration always calls the count evaluation units. The
`sam_optimizer` labels remain local to that template; its mature threshold is
15 seed/dataset units. Other tasks derive their own parent-authorized protocol
unit count.

## Literature, Database, And Open-Access Lookup

The standard template tool set includes evaluation, frontier, finding graph,
memory, prior-work, `tool_server:run_report`, and
`tool_server:literature_lookup` surfaces. This gives Peers and PI memo agents the
ability to consult no-key public scientific context when they explicitly need
it, and gives operators/diagnostics a deterministic manual report generator.
Chair agents inherit source-backed signals from PI memos and shared evidence. A
task-local `literature_scout` role can document source policy, but
`panel.optional_roles` should remain disabled unless the task's panel topology
explicitly implements optional-role execution.

The lookup tool uses public no-key sources first, currently including arXiv,
OpenAlex, PubMed-style metadata, Crossref, Semantic Scholar metadata, Europe
PMC, UniProt, and ClinicalTrials.gov where available. It can fetch open-access
HTML/XML text or record PDF provenance without bypassing paywalls. Treat its
output as contextual literature/database signal only. It can shape research
directions, prior-art risk, and source notes under `assets/literature/`; it is
not evaluator truth and does not replace task-owned metrics. The tool is passive
until an agent calls it; merely listing it in `praxist_plugins.tools` does
not perform network access.

External sources often describe stronger setups that require extra datasets,
checkpoints, simulators, package installs, licenses, APIs, or runtime
environments. Task templates use a current-environment-only resource policy:
agents should not acquire those resources during a run. They should translate
the useful idea into a variant that works with the task's existing local assets,
evaluator, dependencies, and hardware, and record missing resources only as
task-local notes.

## Optional Reviewer

Tracked task descriptors include `workflow_stage:reviewer_stub` as a disabled
optional stage. It can be explicitly run in local/artifact review mode to check
artifact hash consistency, trajectory artifact references, run-summary parsing,
and whether literature/database context was accidentally marked as runtime
truth. Reviewer reports are audit snapshots; they are not promotion truth and
do not change frontier, incubator, Gems, or leaderboard state. The local reviewer
does not append artifacts or trajectory events after `run.finalized`; copy a
completed run first if you need a post-hoc audit artifact.

## Artifact Ownership In Templates

Template prompts and roles should preserve this boundary:

- Peers publish findings and task result summaries.
- Compact task summaries may be nested under `results/**/` and use
  `summary.json`, `evaluation_summary.json`, `eval_summary.json`,
  `tiered_eval_summary.json`, or `custom_*_tiered_eval_summary.json`;
  `result_summary.json` remains a compatibility name. Put lane, maturity,
  effort/coverage ratio, parent-use, protocol, and diagnostic metadata in
  structured fields so Praxist can transfer it into canonical findings.
  If task-owned launch arguments, environment overrides, or config values can
  change the scientific treatment without changing variant code, also publish
  a secret-free top-level `effective_config` and
  `effective_config_complete`. Record evaluator-resolved values after defaults
  and parsing; omitted and explicitly supplied defaults must be equivalent, and
  unrelated runtime environment must be excluded. Exact-replication summaries should name the
  parent's `replication_of_effective_config_sha256`; ordinary summaries and
  legacy template fixtures do not require these optional fields.
- Praxist owns frontier/incubator state, Gems state, prompt-layout artifacts,
  research memory, generation boundaries, and derived leaderboards.
- `frontier/frontier_manifest.json`, committed `gems/gems_state.json`, measured
  result/finding summaries, and `gen_N/generation_boundary.json` are canonical
  current state.
- A task evaluator publishes stable result summaries only. Praxist performs
  idempotent finding materialization and boundary serialization; template
  harnesses must not add their own frontier writes, boundary markers, or sync
  locks. A generation is complete only after its boundary marker exists.
- Modes the user-owned protocol marks non-parentable, commonly preliminary,
  aligned, partial, repair, failed-but-informative, ablation,
  diagnostic, lower-stage, and
  late-after-boundary evidence may be retained as validation candidates. They
  remain visible to later peers, DIG, PI, and diagnostics as compact
  non-frontier signals, but they are not promotion facts unless the task
  explicitly grants that mode ranking or parent authority and canonical
  evidence satisfies it. Under the recommended default, preliminary evidence
  does not rank and aligned evidence only prioritizes mature follow-up.
  Late-after-boundary result summaries should be rerun or explicitly
  revalidated before a task treats them as parents.
- Findings should explicitly report negative-evidence metadata in `extra`:
  `is_negative`, `evidence_valence`, `failure_mode`, and
  `disconfirming_claim_ids`. This lets research memory retrieve failed,
  falsifying, null-ablation, constraint-violating, or non-generalizing results
  without relying on title wording.
- PI evidence packs, PI agendas, rendered prompts, leaderboards, diagnostics,
  and behavior reports are derived views or audit snapshots. They remain
  inspectable but should not be hand-edited or treated as promotion truth.

When adapting a template to a real task, do not add task-specific code that
writes Praxist system state directly. Improve the public evaluator, result summary,
finding metadata, and task prompts instead.

## Testing

Use `toy_math` for no-network offline fixture smoke and `sam_optimizer` for a realistic
resolve-only smoke with provider credentials. Pass an external `--run-dir`
such as a `mktemp -d /tmp/praxist-...` directory; Praxist rejects run directories
inside its own source checkout. New task templates should include at least one
`--resolve-only` smoke path and task-owned tests for any executable harness code.
