---
name: praxist-diagnostic
description: Diagnose Praxist run health, artifact integrity, research-loop completeness, generation-scoped DIG/QD, PI/Gems/frontier/incubator consistency, peer memory freshness, diversity HHI, hardware utilization, LLM/runtime friction, task harness health, sustained low-performance causes, strongest variants/Pareto front, strong-variant lineage, and human-readable run reports. Use when the user asks an agent to investigate whether a current or historical Praxist run is healthy, why progress or performance is weak, whether artifacts or promotions are missing, whether guard or resource issues are blocking peers, to produce a detailed agent behavior analysis report, to generate an A/B/C run report, or to improve or optimize a task after diagnosis using task-directory-only parameter and prompt changes. Default diagnostics are analysis-only; explicit improvement mode may stop the selected run and edit task-level configuration/prompts, but must not modify Praxist core logic.
---

# Praxist Diagnostic

Use this skill as a read-only expert diagnostic pass over an existing
Praxist task/run. The job is to determine whether the run is healthy,
whether the research loop is complete and internally consistent, whether useful
evidence is being transferred correctly, whether resources or runtime systems
are blocking progress, and whether weak performance has a process-level or
task-harness cause.

The skill is task-agnostic. Do not encode domain-specific rules, metrics,
thresholds, paths, or project names into the diagnosis. Use the task's own
`task.yaml`, evaluator outputs, manifests, logs, and documented metrics to
interpret results. If a fact is not available in artifacts, label it unknown
rather than inventing a task-specific assumption.

Artifact source rule: distinguish current fact owners from views. Measured
result/finding summaries, `frontier/frontier_manifest.json`, committed
`gems/gems_state.json`, and `gen_N/generation_boundary.json` are canonical
state. Leaderboards, PI evidence packs, PI agendas, prompt layouts, rendered
prompts, diagnostics, and behavior reports are derived views or audit
snapshots. Use audit snapshots to reconstruct what an agent saw at the time;
use canonical state to judge current promotion, Gems, evidence-transfer, and
resume truth. If `artifact_semantics` exists, report role/status/source fields;
if it is absent, treat the artifact as a legacy file and infer cautiously.
Discover compact evaluator summaries recursively under `results/**/`. Recognize
`summary.json`, `evaluation_summary.json`, `eval_summary.json`,
`tiered_eval_summary.json`, `custom_*_tiered_eval_summary.json`, and the
compatibility name `result_summary.json`. Reconcile their structured
lane/maturity/diagnostic metadata with auto-materialized findings; do not assume
one fixed directory depth or one task-specific filename.
When available, separately inspect task-defined preliminary, aligned, partial,
repair, failed-but-informative, ablation, diagnostic, lower-stage, or
late-after-boundary evidence. They can indicate missed opportunities or broken
promotion flow, but they are not clean frontier/Gems facts. For expensive
tasks, diagnostics must not mix preliminary triage results with aligned ranking
evidence, and must check whether aligned evaluation kept the same or
near-complete data/evaluation coverage while reducing training or optimization
effort first. Resolve labels from the task maturity policy; never infer global
maturity from a familiar-looking tier name.
Late-after-boundary signals are result summaries written after
`gen_N/generation_boundary.json`; diagnostics should verify whether they were
materialized into `shared_findings` and whether the run needs targeted
revalidation.
For generation counts, enumerate contiguous committed boundary markers and use
that value as truth. Report any larger status/summary count and any frontier
generation lacking a marker as pending boundary state, not as completed work.
Repeated unchanged auto-materialized findings or a sync loop waking on its own
outputs are runtime defects; they are not new evidence.

For `agent_runtime:codex_sdk`, assess runtime health from normalized trajectory
events, terminal status, usage records, and
`<run_dir>/runtime_state/codex_sdk/`. Healthy runs use long-lived local
app-server clients with independent peer threads and direct MCP servers. A
local relay is expected only for supported Chat Completions providers such as
DeepSeek or OpenRouter; OpenAI connects directly. Diagnose from the normalized
runtime contract rather than human Codex CLI interaction artifacts. Distinguish
provider/relay failure, MCP startup failure, app-server failure,
timeout/cancellation, and task-harness failure.

For every run with reported token usage, include a **Runtime Context Cost**
section. Use `runtime_usage` from `run_diagnostic_inventory.py` and report:

- peer session count by generation and in total;
- sessions per peer-generation;
- inclusive input, cache-read input, cache-creation input when reported,
  uncached input, output, and total tokens;
- cache-hit ratio (`cached_input_tokens / total_input_tokens`) only when the
  reported components are internally consistent; preserve and flag legacy
  inconsistent telemetry instead of clamping it;
- average input tokens per session;
- whether repeated broad task/findings/memory scans appear in continuation
  session tool commands.

Interpret session churn separately from cache behavior. For Codex-native mode
and OpenRouter routes, check whether lossless context efficiency
was expected, whether finding-only sessions still occur more frequently than
the configured interval, and whether control/resource events explain the extra
sessions. Direct DeepSeek is intentionally outside that policy. Do not label a
high session count as a defect when sessions were required by stop, closing,
resource supply, runtime recovery, or explicit operator settings.

## Human-Readable Run Reports

When the user asks for a run report, final report, Pareto/frontier summary,
strongest-variant explanation, lineage report, or wants the result displayed in
the CLI, generate or reuse human-readable reports without changing run facts.

Use the built-in run-report surface when available:

- Praxist automatically writes derived Markdown reports and companion PDFs with
  compact charts under `<task>/docs/praxist_reports/` when a first credible
  above-baseline frontier signal appears, every 3 completed generations, and at
  final run completion.
- If the active task declares `tool_server:run_report`, a panel/tool-capable
  agent can call `generate_run_report` for the selected run. Resolve the
  selected task directory first and pass `task_dir`; otherwise the report may
  be written under the run directory instead of the task's `docs/praxist_reports/`.
- From Codex outside the running agent, import
  `praxist.plugins.workflow_stages.research_loop.backend.run_report` and
  call `generate_run_report(run_dir=..., task_dir=<selected_task_dir>,
  trigger="manual_cli")` when the source checkout is available.

Each report should prioritize human interest in this order:

1. **A. Strongest variants / Pareto front**: strongest mature or Pareto-front
   variants, absolute/task-owned metrics, credibility, and mechanism summary.
   If no clean frontier/Pareto entry exists yet, use single-metric dimension
   winners from validation candidates, shared findings/Gems, and result-summary
   signals so that one-sided baseline beats are visible. Label those rows as
   signals rather than clean promotion truth.
2. **B. Strong-variant evolution / lineage**: only the strong variants'
   development chain, parent generation, parent variant, source result paths,
   and any fusion of prior ideas.
3. **C. Run health**: diagnostic completeness, DIG/QD/PI/Gems/frontier health,
   artifact consistency, resource friction, and caveats.

The companion PDF should include the same A/B/C summary plus task-metric charts
when enough numeric evidence exists: a generation trend for the most salient
metric, a single-metric winner bar chart, and a risk/reward or loss/score
scatter when the task exposes suitable metric names.

Before doing heavy A/B/C synthesis over a large run, decide whether to use
multiple subagents to slice logs/findings by generation, peer group, artifact
type, or time interval. Use deterministic report generation for the compact
base view, then add deeper agent behavior analysis only when the user asks for
it or low-performance analysis requires it.

When showing report content in the CLI, print a concise summary first and then
list every generated or reused report's absolute path. Reports are derived
views; canonical facts remain frontier, findings, result summaries, Gems state,
and generation boundaries.

## Safety Boundary

Default diagnostic mode is read-only:

- Do not stop, resume, crop, rerender, restart, repair, or mutate a run.
- Do not modify `task.yaml`, roles, harness code, evaluator code, variants,
  dependency files, `.venv`, containers, model/provider configuration, frontier,
  Gems, shared findings, peer memory, or generation artifacts.
- Do not run training/evaluation jobs. Use existing artifacts only.
- You may create a new diagnostic Markdown/JSON report in `<task>/docs/` when
  requested or when low-performance analysis requires a durable report.
- If a fix is needed, describe it. Only implement fixes when the user separately
  asks for repair work outside this diagnostic skill.
- Diagnostic recommendations must not add stricter guard or promotion rules by
  default. If a guard or validator issue is found, distinguish "too strict and
  blocking normal work" from "not strict enough"; report both, but do not turn
  the diagnostic into a policy-hardening task.
- Never print secrets, provider keys, tokens, private endpoints, or raw
  credential files. Redact sensitive values in reports.

Exception: if the user explicitly asks to improve or optimize the task, or uses
equivalent wording that asks this diagnostic skill to apply improvements, use
the Task-Only Improvement Mode below. Do not infer improvement mode from a
normal diagnostic/status request.

## Task-Only Improvement Mode

Use this workflow only when the user explicitly asks for task improvement after
or as part of a diagnostic.

1. Ensure a complete diagnostic exists. Reuse a full current-conversation
   diagnostic only if it still applies to the selected run; otherwise run the
   complete Diagnostic Workflow. A short status summary is not enough.
2. Decide whether the biggest issue is fixable by task-directory prompt/config
   changes only: `task.yaml` parameters, research direction text, prompt
   templates, role/audit prompt files, task docs, exploration lists, exposed
   tier/evaluation settings, generation/peer schedule, Gems and generation-scoped DIG/QD settings,
   or task-level runtime settings. If not, say so and stop without editing.
3. Never modify Praxist source/plugin/guard/runtime code, evaluator or harness code,
   baseline/model/training/variant code, data, dependency files, `.venv`,
   containers, credentials, current run artifacts, frontier, Gems, shared
   findings, peer memory, or finding graph state.
4. Produce a concise improvement plan: diagnostic evidence, exact task files,
   expected run-process effect, and why Praxist core changes are out of scope.
5. Before stopping the run, display this conspicuous warning:

   ```text
   **The current Praxist run will now stop. Only task-directory parameters and
   prompt configuration will change, and no new run will start.**
   ```

6. Stop only the selected current run. Prefer
   `praxist stop <run_id> --grace 300 --json`; refuse if multiple live runs match or
   the target is ambiguous; never use `praxist stop --all` unless explicitly asked.
   If no live run exists, report that no stop was needed and continue only when
   the task directory is unambiguous.
7. Apply minimal task-only edits. Preserve metric semantics, baseline records,
   data protocol, evaluator contract, and task-agnostic Praxist boundaries. Avoid
   adding stricter guard/promotion rules by default. Prefer improving
   exploration direction, prompt clarity, role balance, evaluation maturity
   guidance, runtime scale, and task-level schedule/DIG/QD/Gems settings when
   diagnosis supports them.
8. Validate without restarting: use lightweight YAML/resolve/config checks when
   safe. Do not launch training, evaluator sweeps, Praxist start/resume, Gems
   rerender, PI rerender, or any new run. Finish by reporting the diagnostic
   basis, stop result, files changed, validation, risks, and that Codex is
   waiting for the next instruction.

## Quick Inventory

Prefer deterministic collection before interpretation. If the helper script is
available, run it first:

```bash
python skills/praxist-diagnostic/scripts/run_diagnostic_inventory.py \
  --task-path /path/to/task \
  --run-dir /path/to/run \
  --json
```

If the script is unavailable or does not fit the environment, collect the same
information manually with lightweight commands: `praxist status --json`, scoped
platform-appropriate process, accelerator, uptime, memory, and storage probes,
plus small run artifacts such
as `orchestrator_status.json`, `run_summary.json`, `frontier_manifest.json`,
`gems_state.json`, `generation_results.json`, `generation_boundary.json`,
`dig_cohort_allocation.yaml`, prompt layouts, shared findings, and peer memory.
When present, include `orchestrator_status.json` fields such as
`last_stop_audit`, `last_peer_mix`, and `mature_quorum_required`; these are
compact research-loop control telemetry, not separate sources of truth.

Never use broad filesystem scans. Restrict searches to the task directory, the
selected run directory, and the Praxist source checkout only when needed.

When a live run exists, prefer non-invasive observation: read status files,
logs, manifests, and process metadata; do not attach debuggers, alter processes,
touch sentinel files, or execute commands inside peer workspaces.

## Target Resolution

1. Resolve the task path:
   - Use an explicit path from the user when provided.
   - Otherwise use the current directory if it looks like a Praxist task.
   - Otherwise inspect only nearby likely task roots, not the whole machine.
2. Resolve the run:
   - Use an explicit run id/path when provided.
   - If the user says "current run", match live `praxist status --json` entries for
     the task path.
   - If no run is specified and one live run matches the task, use that run.
   - If no live run exists, use the latest `experiments/run_*` directory.
   - If multiple live runs match, list candidates and refuse to guess.
3. Record the exact selected run id, path, PID/state, source checkout, branch or
   package version when discoverable, model/provider/runtime, and timestamp of
   the diagnostic.

## Diagnostic Workflow

### 1. Current Run State

Establish the run's current state before deeper interpretation:

   - state, PID, model/provider/runtime, current generation, completed
     generations, findings count, variants count, Gems count, frontier count,
     wall-clock elapsed time, and latest update time;
   - mature evidence quorum, last stop audit, and last constructive peer-mix
     feedback when present in `orchestrator_status.json` or
     `gen_N/generation_boundary.json`;
   - whether the active generation has `STOP_SIGNAL`,
     `CLOSING_SIGNAL`, `generation_results.json`, and
     `generation_boundary.json`;
   - when `CLOSING_SIGNAL` exists, whether protected PID records remain visible
     while draining and whether logs show a post-close training/evaluation or
     shell-launch attempt; classify such an attempt as a lifecycle-boundary
     defect rather than treating normal result publication as a failure;
   - compare `active_evals` with current-generation protected process groups;
     a task evaluator launched outside the protected manifest is a telemetry
     and drain-integrity defect even when active peer sessions keep the run from
     closing prematurely;
   - whether `run_summary.json`, `orchestrator_status.json`, registry state,
     and live process state agree, or whether some files are stale from a prior
     stop/resume;
   - whether any other Praxist runs share the same task directory, run directory,
     process tree, GPU/CPU resources, or writable artifacts.

### 2. Research-Loop Completeness

Check every stage that should have completed before the current moment. Do not
penalize the active in-progress stage merely because its final boundary files
are not written yet.

   - completed generations should have cohort outputs and boundary artifacts;
   - unfinished final generations should be clearly marked as in progress,
     closing, interrupted, or failed;
   - read `dig_lite.generation_scope` before judging DIG completeness. Under
     the current default, only absolute generation 0 should have DIG
     candidate/review/selection/contract artifacts; their absence in later
     generations is expected, including after a Gems logical-generation reset;
   - read both quality-diversity generation switches. For initial QD, inspect
     the gen0 DIG selection/allocation trace. For later QD, inspect the final
     PI agenda and peer-contract breadth. In Multi-PI mode also inspect PI memo
     proposals and Chair allocation; in single-PI mode do not require PI memos,
     Chair output, DIG, or a separate QD artifact;
   - if PI/Chair has run, expected agenda/synthesis/boundary artifacts should
     be present, validated, and internally coherent;
   - if Gems reset has occurred, Gems state, reset events, frontier references,
     archived candidates, and previous-cycle candidate visibility should be
     complete and credible.
   - result summaries written after their source generation boundary should be
     reported as late-after-boundary validation signals, not silently ignored
     and not counted as clean promotion facts.

### 3. Artifact Integrity

Check whether produced artifacts exist, were updated at plausible times, and
can be parsed:

- peer outputs: variants, logs, notebooks, summaries, handoffs, local memory,
  result directories, and expected peer-local files;
- initial-generation DIG/QD outputs: allocation, contracts, amendments,
  selected candidates, prompt layouts, diversity cells, no-candidate records,
  and failure reasons;
- later-generation QD evidence: final PI agenda/peer-contract breadth, plus PI
  proposal-pool breadth and Chair allocation only in Multi-PI mode, using
  existing PI/agenda artifacts only;
- PI/Chair outputs: evidence packs, role memos, round outputs, final agenda,
  validation/audit files, and per-peer plan slicing when applicable;
- Gems outputs: state file, reset events, selected Gems, archived candidates,
  source lanes, evidence references, and whether each Gem remains credible;
- shared systems: incubator/frontier/leaderboard/shared findings/finding graph
  and peer memory update cadence;
- run summaries and manifests: stale or contradictory status, stale mtime,
  missing generation entries, malformed JSON/YAML, missing checkpoint metadata,
  and partial files left by interruptions.

Explicitly check for large-scale read/write failure patterns: many missing
summaries, repeated zero-byte files, identical truncated logs, unparseable
manifests, failed tool-server writes, permission errors, file locks, disk-full
events, or artifact mtime gaps inconsistent with active work.

When the local reviewer stage or a reviewer report is available, use it as an
additional audit signal for artifact hash mismatches, missing artifact refs,
broken provenance chains, and literature/database context accidentally marked as
runtime truth. Do not treat the reviewer report as promotion truth; it is an
audit snapshot. Do not append a new local reviewer artifact to a run that already
contains `run.finalized`; generate an external diagnostic report instead, or ask
the user before auditing a copied run directory.

### 4. Artifact Information Consistency

Check whether the same scientific fact is represented consistently across
artifacts:

- result summaries versus leaderboard/incubator/frontier entries;
- high-performing results versus Gems selection and prior-cycle Gems archives;
- shared findings versus finding graph nodes and edges;
- generation boundary summaries versus per-peer logs/results;
- PI/Chair agenda claims versus actual evaluated variants;
- peer memory claims versus written result summaries;
- task metric names and directions versus promotion/filter logic;
- live status versus stale `run_summary.json` after stop/resume.

When inconsistencies exist, mark whether they are expected because the active
generation has not closed, or problematic because a completed boundary failed to
ingest or promote evidence.

Also classify each inconsistency by source role:

- canonical-vs-canonical disagreement: highest severity; it may corrupt
  promotion, Gems, resume, or PI planning.
- canonical-vs-derived disagreement: usually a stale view/cache/snapshot; note
  whether regenerating the view would fix it.
- derived-vs-derived disagreement: useful for behavior analysis but not a
  current-state failure unless a runtime reader consumed the stale view.
- partial/failed artifact visibility: a resume or status risk only when normal
  runtime readers treat it as committed.

### 5. Diversity HHI

For each generation with DIG contracts, PI/Chair peer contracts, or peer labels,
compute HHI twice when the data exists: planned HHI from the generation's
canonical PI/Chair `peer_contracts[].planned_dimensions`, and realized HHI from
the findings' actual `design_dimensions`. Never substitute planned values for a
missing realized report. Compare the two distributions and report whether
execution preserved, broadened, or collapsed the planned portfolio. Use the
task's configured axes first; for legacy tasks without them, inspect these
generic labels when present:

- `mechanism_family`
- `intervention_surface`
- `intent`
- `semantic_family`
- `parent_lineage`
- `novelty_axis`

Also report:

- sample size `N` per generation;
- missing label rate per axis;
- dominant label and count per axis;
- whether diversity is broadening, stable, or narrowing over time;
- whether narrowing appears evidence-driven, PI-driven, Gems-driven,
  evaluator-driven, or caused by accidental prompt/task constraints.
- per-axis planned-to-realized coverage and the peers/findings that could not be
  paired without guessing ownership;
- whether aliases of the same exact result artifact are incorrectly consuming
  multiple durable frontier/incubator slots;
- whether result findings have canonical peer ownership from scheduler facts,
  while internal materializer identities remain non-contributing system rows.

Treat higher HHI as narrower diversity. If labels are missing or not in the
expected location, explain that separately rather than claiming diversity is
healthy.

### 6. Experiment Flow And Friction

Check whether the run is flowing normally:

- guard denies/warnings, shell tokenization warnings, permission failures,
  deletion/copy restrictions, and whether guard behavior blocks routine Praxist
  work or only prevents high-risk behavior;
- evaluator/harness crashes, train failures, import errors, dependency errors,
  missing environment activation, container/runtime failures, and checkpoint
  load/save failures;
- tool server failures, MCP failures, registry/plugin resolution failures,
  memory/frontier/finding-graph write failures, and API schema mismatches;
- network or external-resource issues when the task legitimately needs network,
  datasets, simulators, model downloads, remote APIs, or licenses;
- LLM rate limits, provider errors, context-window failures, repeated timeout
  retries, malformed responses, unusually high cost, low cache hit rate, or
  API-key/provider failover;
- hardware problems: accelerator/device loss, backend runtime errors, OOM,
  process death, disk full,
  IO stalls, clock skew, node reboot, or resource preemption;
- operator interruption, external process termination, concurrent runs, or other
  objective causes.

Classify normal exploratory failure separately from systemic friction. A few
failed variants may be expected; repeated failures with the same signature are
diagnostic.

### 7. Hardware Utilization And Scale Fit

Probe the platform and the task-selected execution backend first, then report
only applicable current and inferred utilization. CPU-only execution, macOS
unified-memory execution, other unified-memory systems, and other accelerator
backends are normal paths; the
absence of CUDA/NVIDIA/UUID telemetry is not a defect unless the task selected
that compatible backend.

- for a selected discrete accelerator backend, its available utilization,
  memory, temperature, visibility, and whether the task is actually using it;
- CPU load, process groups, per-variant process counts, memory, swap, disk, and
  IO pressure;
- when `resource_scheduler/status.json` exists, reconcile queued, running,
  completed, failed, and rejected jobs; current host-wide concurrency; frozen
  generations; assigned accelerator devices when exposed by the active backend;
  work classes; attempts; immutable attempt
  logs; protected process groups; and evaluator outputs;
- compare lifecycle `running` with `running_activity.by_resource_phase` and each
  job's `resource_activity`. Treat `no_gpu_process_observed` as a point-in-time
  fact, `gpu_process_attribution_unavailable` as a PID-namespace attribution
  gap, `attribution: partial` as incomplete job-level accounting, and `unknown`
  as a missing observation, not automatic failure. Flag a
  sustained mismatch only when resource samples, process trees, progress, logs,
  and result mtimes agree that useful work is not advancing; report queued jobs
  blocked by per-peer capacity separately from resource-admission pressure;
- report idle supply waiters and leases (including `release_pending` cleanup),
  inspect `resource_scheduler/events.jsonl`
  and `resource_supply.stats` for overall and per-priority conversion. Reconcile
  grant/consume/decline/expire/revoke/stale-submission/genuine-reuse outcomes to
  verify one-experiment lease consumption, and determine whether the run is
  capacity-limited, response-window-limited, or instead has admissible capacity
  with too little planned experiment supply. Do not call an expired unconsumed
  lease a consumed reuse;
- reconcile scheduler `Q/M/D/A_target`, mature queued/running work,
  mature-priority leases, first-wave allocation, and assessment-only admission
  against canonical effort/coverage results; flag stale debt, duplicate
  commitments, or premature Closing without treating GPU utilization as
  evidence maturity;
- compare the task maturity policy with
  `synthesis_trigger.mature_quorum_fraction`. If the task distinguishes
  close-grade evidence but the quorum is `0.0`, report that raw
  information-density findings can close the generation without mature
  results. Treat an `info_density` Closing snapshot with
  `required_mature_result_peers=0` and no mature results as a task-harness
  closing-policy mismatch, not as evidence that the scientific work matured;
- whether the bottleneck is CPU, GPU, IO, memory, API/LLM, scheduler, evaluator
  concurrency, or idle waiting;
- whether peer count, generation duration, per-peer concurrency, and evaluation
  parallelism look under-sized, over-sized, or reasonably matched to hardware;
- whether resource allocation is likely to converge to a balanced state or
  leave material compute idle.
- whether the public evaluator default profile is accurate and complete evaluators
  exploit safe natural parallel units without double-counting top-level jobs
  and internal children;
- whether CPU pressure changes only total experiment concurrency; for a
  selected managed discrete-accelerator backend, whether jobs are packed by
  measured memory/utilization and unknown demand is exclusive; and whether any
  failed job silently changed execution backend without task-declared
  scientific equivalence.

Do not recommend increasing concurrency if artifacts show instability, shared
resource interference, or task-level evaluator contention.

### 8. Performance And Evidence Transfer

Use the task's own metric definitions. If a metric is relative, label it as
relative; if a user asks for absolute values, use absolute metric fields from
the artifacts and say when they are unavailable.

Check:

- baseline availability and whether comparisons use the same protocol, tier,
  seed count, completed task-owned evaluation-unit count, and evaluator
  version;
- mature variants versus baseline and current frontier/incubator/Gems;
- preliminary, aligned, and partial variants separately from complete/mature
  variants;
- when both aligned and complete evidence exist, estimate aligned-to-complete
  calibration or correlation separately for preliminary and aligned evidence;
- whether aligned evidence preserved near-complete data/evaluation coverage
  and mainly reduced training/optimization budget, rather than using a small
  evaluation-unit subset that should be labeled partial;
- historical trend by generation: best variant, average mature performance,
  weighted performance if reliability weights are available, and variance;
- whether generation count greater than 5 shows plateau, regression, or
  continued frontier movement;
- whether high-performance variants are suspicious, immature, protocol-failed,
  evaluator-integrity flagged, leakage-prone, or merely not yet ingested;
- whether high-potential but non-clean variants are retained as validation
  candidates or repair targets rather than silently dropped;
- whether a durable incubator lane has stayed empty for multiple completed
  generations despite task-authorized, protocol-passed, non-suspect
  Pareto/new-high candidates. Treat that as a task-harness evidence-retention
  problem that can cause stagnation: later agents may stop developing strong
  parents even though the evaluations already produced them.
- whether frontier/incubator/leaderboard/Gems omitted any confirmed or
  high-potential variants after a completed boundary.

When identifying missing candidates, mark each as one of:

- `confirmed`: authorized by the task as mature and reliable enough that
  omission is likely a bug;
- `immature`: promising but produced by a mode the task does not authorize for
  durable promotion;
- `risky`: useful as repair/falsification evidence but not clean;
- `suspicious`: protocol, leakage, or evaluator-integrity issue;
- `not_yet_ingested`: active generation has not closed or promotion has not run.

### 9. Performance Ceiling Detection

Every diagnostic must end with a performance-ceiling check, even when the
answer is "unknown". Use task-owned metrics and maturity definitions.

1. Select the mature evidence set:
   - use variants that satisfy the user-approved task maturity contract on the
     task primary metric;
   - keep modes marked non-mature by that contract, plus suspicious and
     protocol-failed evidence, out of the ceiling calculation unless no mature
     evidence exists;
   - mark the active in-progress generation as provisional.
2. Build per-generation series:
   - best mature score by generation;
   - mean mature score by generation when sample sizes are comparable;
   - cumulative best frontier score.
3. Determine metric direction from task artifacts. If direction is unavailable,
   report that ceiling detection is unknown.
4. Detect a significant ceiling conservatively:
   - use task-defined material-improvement thresholds when documented;
   - otherwise estimate a practical threshold from metric scale and recent
     variation, and label the result heuristic;
   - call a ceiling only when the cumulative best has not materially improved
     for several completed generations after the first strong peak and mature
     sample counts are not trivially small.
5. Report:
   - `ceiling_detected`: yes / no / unknown;
   - `plateau_onset_generation`: the first generation after which no material
     cumulative improvement appears, or unknown;
   - supporting metric series and sample sizes;
   - whether weak later generations are explained by deliberate falsifiers,
     exploratory diversity, evaluator failures, lost evidence transfer, or true
     search stagnation;
   - if periodic Gems reset is disabled, whether a reset should be considered
     and what reset interval would match the observed plateau onset. This is a
     recommendation only; do not edit task config during diagnostics.

When `$terminal-line-plot` is available and the user wants visual output, use it
to draw at most two ceiling-related curves. Mark incomplete points as
provisional.

### 10. Task Harness Health

Inspect the harness only through task files and existing artifacts:

- task directory completeness and whether the declared project can run in the
  current environment when Praxist runs it;
- task-owned stage protocol meaning, effort/coverage references,
  evaluation-unit/seed/sample definitions, and whether the difficulty ladder is
  reasonable for this task;
- baseline definition, baseline performance records, and evaluator
  determinism/reproducibility;
- metric names, metric directions, promotion criteria, canary/validation
  evaluation units, and whether any metric can be gamed;
- when the user-approved maturity contract uses ratios, whether mature
  evaluator outputs include generic `effort_ratio` and `coverage_ratio`, and
  whether `evaluation.maturity_policy`,
  `evaluation.launch_guard`, `evaluation.constructive_peer_mix_enabled`,
  `evaluation.constructive_target_ratio`, `dig_lite.generation_scope`, both
  `quality_diversity` generation switches, and
  `synthesis_trigger.mature_quorum_fraction` match the observed task cost and
  evidence cadence;
- whether `complete_stage_labels` and `preliminary_stage_labels` match the
  task-owned protocol; with `require_ratio_gate: true`, missing ratios must
  remain unknown, while a legacy false setting may fall back only to configured
  labels or explicit completion flags;
- whether every durable mature parent lane sets `parent_eligible: true` and
  every lower-tier/diagnostic lane sets it false, even when
  `allow_lower_tier: true` retains signals for revalidation;
- whether Gems uses `selection_policy: mature_evidence_top_k` with a
  task-derived `min_mature_eval_units`, and whether that count matches the
  protocol authorized for Gems/parent use rather than a label imported from
  another task; when
  `evidence_stage_min_units` is present, verify every threshold is a task-owned
  evaluation-unit count and agrees with the evaluator protocol;
- dataset/simulator/model/resource metadata when the task depends on them;
- role prompts, PI/Chair instructions, exploration directions, and whether they
  are too few, too narrow, too domain-specific, or accidentally constrain
  diversity;
- whether task harness changes alone could fix any diagnosed problem without
  changing Praxist core.

Do not import task-specific domain assumptions from another task. If the task
lacks enough information to interpret its metrics, report that as a task-harness
gap.

### 11. Issue Classification

For every negative finding, provide:

- severity:
  - `must_fix_now`: threatens current run validity, artifact integrity,
    promotion correctness, or normal execution;
  - `fix_next_run`: does not invalidate the current run but should be fixed
    before future campaigns;
  - `minor_or_expected`: normal stochastic failure, isolated bug, harmless
    stale artifact, or expected failed experiment;
- likely cause:
  - Praxist core;
  - task harness;
  - objective environment/hardware/API;
  - operator interruption;
  - unknown / needs more evidence;
- evidence: exact files, timestamps, log snippets, counts, or metrics;
- current-run impact: invalidates, degrades, delays, biases selection, or no
  material impact;
- task-harness-only mitigation when available;
- Praxist-core mitigation only when task-harness changes are insufficient.

## Low Performance Protocol

If the run shows sustained low performance, long periods below baseline, or an
apparent ceiling after generation 5, the diagnostic must go beyond a short
status summary.

1. Produce a detailed agent behavior analysis report unless the user explicitly
   asked for a brief status-only response.
2. Use multiple subagents when available and safe: split logs/findings/peer
   memories by generation, peer group, artifact type, or time segment; then
   synthesize their outputs in the main agent.
3. The report must include a timeline of key peer, PI, Chair, DIG/QD, Gems,
   frontier/incubator/leaderboard, shared-findings, memory, and evaluation
   events.
4. After the timeline, explain root causes of weak performance:
   - poor or narrow research directions;
   - bad evidence transfer or lost high-potential parents;
   - PI/Chair agenda mismatch;
   - overly narrow diversity;
   - evaluator or tier protocol mismatch;
   - baseline/harness flaws;
   - resource/API/guard friction;
   - objective task difficulty or stochastic variance.
5. Explain which causes are fixable by task harness changes alone and which
   require Praxist core/runtime changes.
6. Save the report under `<task>/docs/` as Markdown when the user requested a
   durable report or when low-performance analysis is mandatory.

## Agent Behavior Report

When the user explicitly asks for an agent behavior analysis report:

- read logs, trajectory, notebooks, shared findings, generation artifacts,
  peer memory, frontier/incubator/Gems state, and result summaries;
- preserve the distinction between "what the agent saw then" from prompt,
  agenda, leaderboard, and evidence-pack snapshots, and "what the run knows
  now" from canonical result/frontier/Gems/boundary state;
- use subagents for large runs when available, with isolated slices and no
  leaked conclusions;
- produce a chronological Markdown report containing key events, cross-peer
  influence, PI/Chair decisions, memory updates, experiment launches/results,
  failures, and evidence transfer;
- save it to `<task>/docs/` with a descriptive filename;
- keep raw secrets and provider keys out of the report.

For very large runs, explicitly state the slicing method: by generation, peer
range, event type, or time interval. Each subagent or analysis slice should work
from raw artifacts rather than leaked conclusions. The final synthesis should
connect events across slices and identify causal chains, not merely concatenate
summaries.

## Output Format

For ordinary diagnostics, return a concise report with these sections:

- `Overall Health`: healthy / degraded / invalid / unknown.
- `Current Stage`: generation, completed boundaries, active work.
- `Artifact Integrity`: missing, stale, contradictory, or healthy artifacts.
- `DIG / QD / PI / Gems`: generation-scoped completeness and credibility.
- `Diversity HHI`: per-generation table and interpretation.
- `Performance`: mature variants vs baseline, plateau risk, missing candidates.
- `Performance Ceiling`: ceiling_detected, plateau_onset_generation, evidence
  series, and whether Gems reset should be considered.
- `Flow Blockers`: guard, LLM rate limits, tool/resource/network/hardware issues.
- `Hardware Utilization`: current and inferred bottlenecks.
- `Task Harness`: tier protocol and exploration-direction health.
- `Issues`: severity, cause, evidence, and task-harness-only mitigation.
- `Next Actions`: diagnostic recommendations only, not code edits.
- `Generated Reports`: absolute paths of any A/B/C human-readable run reports
  generated or reused during this diagnostic.

Use exact metric names from artifacts. If a metric is relative, label it as
relative; do not rename it as absolute performance.

When a table is useful, include one. For performance tables, include protocol
maturity such as task-owned stage/evaluation-unit/seed counts so partial
results are not mixed with
full results. For diversity tables, include `N` and missing-label caveats.

## Completeness Checklist

Before finalizing, verify that the report has answered these questions or marked
them unknown:

- Is the selected run unambiguous?
- Are completed stages complete, and is the active stage correctly identified?
- Are peer, generation-scoped DIG/QD, PI/Chair, Gems, frontier/incubator/leaderboard,
  shared-findings, finding-graph, and memory artifacts present and parseable?
- Are high-performing or high-potential variants missing from durable records?
- Are HHI diversity metrics computed for the requested axes?
- Are guard, LLM/API, tool-server, dependency, hardware, dataset/simulator, and
  external-resource problems separated from normal failed experiments?
- Are session count, input/cached/uncached tokens, cache-hit ratio, and repeated
  continuation bootstrap behavior reported when runtime metering exists?
- Is hardware underused, overloaded, or balanced for the current task?
- Are task parameters such as peer count, generation duration, and evaluation
  concurrency reasonable for observed cost and throughput?
- Does the task harness define clear tiers, baseline, metrics, promotion
  criteria, and exploration directions?
- If performance is weak or plateaued, is there an agent behavior timeline and
  a root-cause analysis?
- Did the diagnostic explicitly report whether a performance ceiling was
  detected, the plateau onset generation when detectable, and the evidence
  series or reason for unknown?
- Are all negative findings classified by severity and likely cause?
- Did the diagnostic avoid modifying code, artifacts, active dependencies, Praxist
  core, provider configuration, frontier, Gems, shared findings, and peer memory?
