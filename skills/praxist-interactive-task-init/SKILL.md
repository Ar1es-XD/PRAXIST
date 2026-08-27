---
name: praxist-interactive-task-init
description: Build a Praxist task project through a confirmation-first interactive agent workflow. Use when a user wants Praxist task initialization with human confirmation of research goals, constraints, metrics, ranking rules, evaluation protocol, compute budget, baseline handling, or launch readiness; when the user asks for an interactive task init skill; or when the agent should propose a task harness first and ask the user to approve or revise it before writing files.
---

# Praxist Interactive Task Init

Use this skill when task initialization needs explicit user confirmation. It is
a thin interactive layer over `praxist-task-initialization`, not a separate
task format.

This layer inherits the complete current runtime contract from
`praxist-task-initialization`. Keep `agent_runtime:claude_sdk` as the default
unless the user approves another runtime. If the user selects
`agent_runtime:codex_sdk`, apply the SDK/MCP/relay readiness and task-boundary
rules from that skill; do not treat the interactive agent CLI hosting this
conversation as the peer runtime. Its saved ChatGPT authentication may be
approved for native OpenAI, but must remain operator-owned and outside the
generated task. Preserve the tested runtime pins inherited from task
initialization: `claude-agent-sdk==0.2.136`, `openai-codex==0.147.0`, and
`codex-relay==0.5.5`.

Provider-specific context efficiency is runtime-owned. When the user selects
Codex-native mode or OpenRouter, explain that Praxist automatically
coalesces finding-only continuation wakeups without deleting or compressing
canonical evidence. Do not offer task-local cache, memory-store, or
session-batching fields. Direct DeepSeek runs preserve their existing behavior.

Reasoning effort is inherited from the complete task-init contract. Propose
`agent.reasoning_effort: max`; ask for confirmation only when the user raised
reasoning depth, latency, or cost as a constraint, and preserve any explicit
`off`, `low`, `high`, or `max` answer.

## Interaction Principle

Use confirmation first, not input first:

1. Inspect the project and propose a concrete harness decision.
2. Ask the user whether to accept or correct it.
3. Apply accepted decisions to the normal task-initialization workflow.

Keep interaction short. Use at most **5 confirmation rounds** by default. This
limit follows working-memory and UX evidence: practical working memory is often
closer to 3-4 chunks, choice overload hurts decisions, and progressive
disclosure keeps advanced questions deferred until needed. Group related
decisions into one round and use progressive disclosure for rare edge cases.

## Opening Banner

```text
**IMPORTANT PRECONDITION**
Praxist task initialization assumes the research project already runs on
this machine with all required code, data/simulator assets, runtime/container,
and credentials available. If the project cannot run locally, this skill must
stop and ask for the missing path or environment.
```

## Five-Round Confirmation Flow

Round 1: Project scope and local assets

- Propose the research project root and output task path.
- Summarize available code, data, simulator, runtime, and prior results.
- Ask the user to confirm or correct missing/ambiguous assets.

Round 2: Research goal, constraints, and allowed interventions

- Propose the task objective, hard constraints, allowed code surfaces, and
  invalid-result conditions.
- Confirm what Praxist peers may change and what must remain fixed.
- Propose which evaluator modes may launch and which may rank, count as mature,
  become durable parents, or satisfy close. The user's answer is authoritative;
  do not assume every task must use a full-only protocol.

Round 3: Metrics, ranking, and robustness

- Propose primary metric, direction, auxiliary metrics, baseline definition,
  and the minimum frontier-lane structure justified by the task. For staged,
  diagnostic, or multi-axis evaluation, propose strict confirmed,
  lower-admission durable incubator, lower-confidence task-candidate, and
  diagnostic/control lanes. For a cheap single-protocol task, do not invent
  unsupported lanes or maturity stages.
- Use project evidence first; when ambiguous, run bounded no-key web/literature
  lookup to identify domain ranking conventions.
- Explicitly ask whether variance, seed sensitivity, confidence intervals,
  lower confidence bounds, safety/regret constraints, or Pareto fronts should
  affect ranking. If yes, encode robust metrics or Pareto axes in the task.
- Confirm the incubator policy when the task needs a separate durable
  promising-candidate lane. It should be a
  low-admission long-term variant library, not a high-standard confirmed
  frontier. Under the recommended default it retains parent-authorized,
  protocol-passed, non-suspect Pareto/new-high candidates for later repair, validation,
  escalation, ablation, or falsification. If the user authorizes a reduced
  parent protocol, use that protocol consistently instead. Confirm the incubator lane sets
  `admit_new_high: true`.
- Confirm how canonical evaluator **source labels** reach those target lanes.
  When confirmed and incubator both need ordinary clean parent-authorized evidence,
  propose one shared task-owned source label (normally `performance`) accepted
  by both instead of forcing the user to classify every parent-authorized result as a
  final target lane. The full task-init workflow must generate and pass a
  lane-routing regression before launch.

Round 4: Evaluation protocol and compute budget

- First restate the confirmed protocol-intent table. If the user intentionally
  selected partial, scout, reduced-coverage, or other incomplete evidence for
  ranking or mature use, preserve it and require transparent stage and
  effort/coverage metadata. Only undeclared drift is invalid.
- Then decide whether staged evaluation is justified only for details the user
  has not already decided. If the target protocol is expensive, normally
  propose task-owned preliminary, aligned, and complete mature modes. If it is
  cheap, normally propose one complete mode. Do not add or re-propose a full
  mode when the user has explicitly selected an intentionally reduced protocol
  for the run. Literal labels carry no global semantics.
- When aligned evaluation exists, preserve near-complete data/evaluation
  coverage and save compute primarily through fewer training/optimization steps.
- If the confirmed maturity definition uses effort/coverage ratios, confirm
  canonical evaluator summaries emit exact `effort_ratio` and `coverage_ratio`
  fields and confirm the `evaluation.maturity_policy` thresholds that use them.
  Use `require_ratio_gate: true` only for that choice; otherwise preserve the
  user's explicit label/flag or information-density semantics. List task-owned
  labels only when staged protocols exist, for audit and explicit fallback. Praxist
  projects these facts into auto-materialized findings; require them directly
  only for standalone findings without a canonical summary reference.
- Before launch, run the shortest valid scored path through the real summary
  writer and validate its output with `praxist resolve <task_path>
  --result-summary <summary_path>`. Missing required ratios must lead to
  evaluator repair or an explicit user-approved gate disable, not invented
  stage labels.
- For every configured lane, confirm mature parent lanes have
  `parent_eligible: true`, while
  lower-stage/diagnostic lanes use `parent_eligible: false`. A lane with
  `allow_lower_tier: true` must not become a durable implementation-parent lane.
- If the task enables Gems, confirm its configuration uses
  `selection_policy: mature_evidence_top_k` and derives
  `min_mature_eval_units` from the protocol authorized for Gems and parent use,
  normally the complete protocol. If stages have
  cumulative requirements, define them with `evidence_stage_min_units` using
  task-owned labels and evaluation-unit counts. Do not generate a
  compatibility-only historical maturity field.
- Confirm compact summaries are written recursively under `results/**/` using
  `summary.json`, `evaluation_summary.json`, `eval_summary.json`,
  `tiered_eval_summary.json`, or `custom_*_tiered_eval_summary.json`
  (`result_summary.json` is a compatibility name), reuse a stable top-level
  candidate identity across stages (or an explicit child-result ID), and carry
  structured lane, maturity, protocol, and diagnostic metadata for materialization.
- Confirm evaluator execution prefers the synchronous public entrypoint. If a
  background evaluation is explicitly supported, use the Praxist submission facade
  and a documented task-owned progress/result contract. Never use the byte size
  of a runtime-private `tasks/<task-id>.output` transcript as completion;
  successful commands may emit no text, and the runtime notification/exit
  status owns completion.
- Confirm `evaluation.constructive_peer_mix_enabled` and
  `evaluation.constructive_target_ratio`,
  `evaluation.launch_guard.estimated_heavy_eval_minutes`, the separate
  `estimated_close_grade_eval_minutes`, and whether optional
  `synthesis_trigger.mature_quorum_fraction` is positive when the task
  distinguishes mature/complete evidence from preliminary, partial,
  diagnostic, or progress findings. Explain that `0.0` allows raw information
  density to become the normal close condition; propose it only when the user
  explicitly confirms that the task has no separate close-grade evidence
  contract.
- When close requires mature/complete evidence, show the measured complete
  evaluator p90, safety factor, earliest effective close horizon, and drain
  margin. Require
  `estimated_close_grade_eval_minutes * safety_factor < effective close
  horizon - drain margin` before approval. Keep the heavier ordinary estimate
  separate when a user-authorized reduced protocol owns close. If the
  inequality fails, propose longer bounds or a user-authorized protocol change
  rather than silently weakening evidence.
- Confirm internal evaluator/trainer/config paths are task-root relative,
  intentional external absolute paths exist, and the public evaluator resolves
  identically from the task root and a run-like subdirectory using the declared
  task interpreter without runner-owned Python import paths.
- Confirm absolute-gen0-only DIG, the independent initial/later
  `quality_diversity` switches, and the separate
  `dig_lite.innovation.enforce_forward_slots` choice. Explain that later QD is
  soft guidance inside the existing single-PI or Multi-PI synthesis path and
  does not run DIG again. If the user changes cohort size, re-check these soft
  mix choices and ask for confirmation only when they materially conflict.
- Confirm task prompts state the close boundary explicitly: after
  `CLOSING_SIGNAL`, existing training/evaluation work drains naturally, but a
  peer may only inspect results, publish findings, and update notebook/memory;
  it must not launch another evaluator, script, shell launcher, or background
  process.
- If a proposed complete protocol is materially below the project's established
  reference effort or measured convergence evidence and the user has not
  already decided, pause and ask whether to keep it, increase it, or downgrade
  it to aligned evidence. Do not impose universal epoch, step, rollout, or wall-clock threshold,
  and do not re-litigate an explicit user choice.
- Estimate hardware bottleneck and propose peers, generation count, generation
  duration, evaluator concurrency, and baseline-measurement handling.
- Show the unchanged baseline command used for resource observation. When Praxist
  can own and observe local process launch, propose central scheduler named
  profiles, initial/max host-wide concurrency, and measured pressure for the
  backend actually used by the unchanged baseline. CUDA/NVIDIA/UUID checks are
  applicable only when that compatible managed backend was detected and
  selected. CPU-only, unified-memory, task-managed, and other accelerator
  backends are normal paths. Otherwise propose a
  bounded documented external/legacy owner rather than forcing central mode.
  Never propose synthetic CPU-vs-accelerator rewrites or per-experiment CPU-core
  reservations.
- Confirm the public evaluator's normal default profile, naturally independent
  seeds/folds/scenarios or other task-owned evaluation units, which layer owns their
  concurrency, and whether a declared multi-device profile really uses every
  assigned device. Confirm directed idle resource-supply feedback; it may wake
  only productive idle peers for one already planned experiment and must not
  weaken protocol maturity. Use a 600-second bounded response window by default;
  it limits submission timing, not the runtime of work admitted before expiry.
  Propose the default mature supply target
  `Q=ceil(peers*0.25)` with bounded `3D` in-flight redundancy, while preserving
  an exploration first wave. Keep the hard close quorum independently positive
  for tasks with maturity distinctions. If measured generation feasibility is
  insufficient, propose a longer generation, different concurrency, or a
  user-approved evidence protocol; do not turn raw findings into normal-success
  close merely to avoid a deadlock.

Round 5: Final harness and launch readiness

- Show the final task path, baseline status, evaluator command, provider/runtime
  recommendation, absolute-gen0 DIG status, initial/later QD status,
  continuous-evolution/Gems policy, and report
  generation settings.
- Ask for approval to write/update the task directory. Do not start a run unless
  the user separately asks or this skill is being used inside
  `praxist-takeover`.

## Execution Rules

- Use `praxist-task-initialization` for the actual task-project creation
  steps after decisions are confirmed.
- Abort on missing required code, data/simulator, runtime dependencies, or
  executable baseline/evaluation path.
- Do not store raw API keys in the task directory.
- Do not edit Praxist core or guard code.
- Do not create task-specific hacks in Praxist plugins.
- Do not launch long baseline measurements without user approval.

## Final Output

Report:

- confirmed decisions and any user corrections;
- task directory path;
- baseline status: existing, measured, zero placeholder, or blocked;
- metric/ranking rule including robustness treatment;
- incubator policy and distinct metric families used for Pareto/new-high
  retention;
- evaluator source-lane contract and lane-routing regression result;
- task-owned evaluation stages, maturity ratio fields, and expected
  budget/integrity fields;
- user-approved protocol intent for launch, ranking, maturity, parent use, and close;
- selected maturity policy, constructive target, launch guard, and mature quorum
  settings;
- selected Praxist run parameters and launch recommendation;
- selected provider/runtime and whether runtime-owned lossless context
  efficiency is expected (Codex-native mode/OpenRouter), unchanged
  (direct DeepSeek), or explicitly disabled by the operator;
- validation commands run and results;
- remaining questions, if any.
