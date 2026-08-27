---
name: praxist-takeover
description: Orchestrate first-use Praxist onboarding, current task initialization, validation, and detached run launch from a supported agent interface. Use when a user wants a one-command or one-conversation path from an existing runnable research project to a started Praxist run, asks to onboard plus initialize plus start, asks for repo-to-task-to-run setup, or wants the agent to prepare and launch Praxist with minimal interaction. This skill composes onboarding, full task initialization or task-harness repair, runtime checks, control start, and confirmation gates without changing Praxist core.
---

# Praxist Takeover

Use this skill to move a user from an existing runnable research project to a
detached Praxist run in one guided flow. It composes existing skills rather
than duplicating their detailed procedures:

- `praxist-onboarding` for Praxist context and local readiness scan.
- `praxist-task-initialization` or
  `praxist-interactive-task-init` for task-project creation.
- `praxist-control` for task validation and detached launch.

The operator-facing `praxist --takeover` command is only a first-project
handoff into this skill. It does not replace, abbreviate, or independently
implement any step below. Treat its selected absolute project path and final
Enter confirmation as explicit operator context.

The first operational action must be the complete `praxist-onboarding`
workflow. Do not replace onboarding with a shorter local checklist, and do not
proceed to task initialization or launch until onboarding has produced its
readiness summary. This applies even when Codex already appears familiar with
Praxist from prior context.

When no launch-ready task exists, the next action must be the **complete current
`praxist-task-initialization` workflow**. `praxist-takeover` must invoke and
follow that skill rather than recreating a shortened initializer from this file. Load
and read the installed skill's current `SKILL.md` at execution time; do not rely
on a remembered or summarized task-init procedure. Wait for task-init's
evaluator, lane-routing regression, resource plan, and required validations to
finish before continuing. A user who invokes only `praxist-takeover` must
receive the same harness contract as a direct task-init user.

## Hard Preconditions

Print this before doing work:

```text
**IMPORTANT PRECONDITION**
This takeover assumes the research project already runs on this machine,
with required code, data, simulator, runtime/container, and credentials present.
If those are missing, initialization must stop instead of creating a weak Praxist task.
```

## Workflow

1. Run the complete onboarding flow:
   - invoke `praxist-onboarding` and follow its `SKILL.md` procedure;
   - let onboarding inspect nearby Praxist source or pip installs, `praxist` CLI
     availability, installed skills, provider-key presence by name only, and
     bounded local environment context;
   - keep onboarding read-only and bounded as defined by that skill;
   - stop if onboarding reports that Praxist itself is unavailable or unusable.
2. Resolve the research project path:
   - use the user-provided path if present;
   - otherwise use the current working directory.
   - if that path is a bundled example inside a Praxist source checkout or
     package resource, do not initialize or launch it there. Identify its name
     with `praxist examples list`, run `praxist examples install <name>`, report
     the writable destination, and continue from that destination. Preserve an
     existing destination unchanged.
3. Decide whether a runnable task directory already exists:
   - if an explicit task directory is in context, validate that path;
   - otherwise scan nearby roots up to depth 3 for `task.yaml`, excluding
     `.git`, `.venv`, `node_modules`, `experiments`, run directories, caches,
     and raw data roots.
4. If no valid task exists, initialize one:
   - load the currently installed `praxist-task-initialization` skill and
     complete it for the default low-friction path; do not substitute a reduced
     `praxist-takeover` implementation or an older remembered checklist;
   - use `praxist-interactive-task-init` when the user asks for
     confirmation, an interactive flow, strict goal confirmation, or when the
     task objective/constraints are ambiguous. The interactive path must still
     delegate task construction and final validation to the current task-init
     workflow.
5. Resolve and show the absolute task path before launch; confirm the exact task path
   when selection is ambiguous or the user has not authorized launch. If the user's
   current instruction already names or unambiguously identifies the task and
   explicitly authorizes direct/no-confirm launch, obey it without adding another confirmation round.
6. Check and warn about durable evidence-retention lanes before launch:
   - read the protocol-intent decisions produced by current task-init. User
     instructions own which evaluator stages may launch, rank, count as mature,
     become parents, or satisfy close. Do not impose a full-protocol-only rule;
     an explicitly authorized reduced/partial/scout protocol is valid when its
     actual stage and effort/coverage remain visible and every downstream
     policy agrees. Only undeclared drift must be prevented from impersonating
     mature evidence;
   - inspect `task.yaml:evaluation.frontier_lanes`;
   - prefer the minimum retention structure supported by the task protocol;
     staged, diagnostic, or multi-axis tasks should normally have a strict
     confirmed/mature lane, a lower-admission durable incubator lane, a
     lower-confidence candidate lane, and a diagnostic/control lane, while a
     cheap single-protocol task must not invent unsupported lanes;
   - accept generic incubator-family lane names such as `incubator`,
     `<task>_incubator`, or `candidate_library`;
   - when a lower-admission incubator lane is present, confirm it sets
     `admit_new_high: true`, so it retains only new-high or Pareto-improving
     candidates authorized for durable retention instead of replaying stale points;
   - when parent-capable confirmed/incubator lanes are present, confirm they explicitly
     set `parent_eligible: true`, while lower-stage and diagnostic lanes set
     `parent_eligible: false`; `allow_lower_tier: true` is retention for
     revalidation, not permission to use that lane as a durable parent;
   - distinguish evaluator **source labels** from Praxist-selected durable target
     lanes. Inspect the canonical evaluator summary path and execute the
     task-local lane-routing regression produced by task-init. Every configured
     parent-eligible lane must intersect at least one protocol-passed source
     label from an evaluator mode the user-owned intent authorizes as a parent;
   - when confirmed and incubator should independently consider ordinary
     parent-authorized performance evidence, prefer one shared task-owned source label
     (normally `performance`) accepted by both lanes. Reject a newly generated
     harness that labels every parent-authorized result `confirmed` while the incubator
     accepts only `incubator`/`performance`, because that makes the incubator a
     dead lane. If both `frontier_lane` and `promotion_lane` are emitted, require
     them to agree;
   - require the regression to exercise more than `confirmed.k` parent-authorized
     fixtures. When the task has a justified distinct incubator axis, show that
     a candidate outside confirmed top-k but non-dominated on that axis remains
     routable to incubator; do not invent an axis for a single-metric task. The
     regression must also prove every mode marked non-parentable by the intent
     table cannot become a durable parent. Under the recommended default these
     fixtures include preliminary, partial, smoke, validation-only, late,
     protocol-failed, and suspect evidence;
     do not fail merely because a reachable incubator is empty in one real
     generation where no new Pareto/new-high candidate exists. Adapt these
     fixtures to the user-owned protocol intent: an explicitly parent-eligible
     reduced stage is not an invalid partial fixture;
   - if a newly initialized task fails this check, return immediately to the
     full task-init workflow, repair only the task evaluator/config/prompts,
     rerun the regression, and do not launch until it passes. Apply the same
     launch gate to a pre-existing task with an unreachable parent lane that is
     configured as parent-eligible: use task-init in task-harness repair mode
     when the intended semantics are clear, or stop and ask the user when
     changing scientific selection semantics is ambiguous. This gate does not
     require an optional absent lane or a reachable lane to receive an entry in
     every generation;
   - warn, but do not refuse solely for this reason, if a task that needs staged
     or multi-axis retention lacks these lanes. The runtime can still launch legacy tasks;
     recommend diagnostic/task-initialization repair after launch
     unless the user asks to stop and fix the task first. A long-empty configured
     incubator can cause later generations to lose strong parents and make the
     run stagnate even when good variants were already evaluated.
   - prefer `evaluation.maturity_policy`, `evaluation.launch_guard`,
     `evaluation.constructive_peer_mix_enabled`, and
     `evaluation.constructive_target_ratio` exist. When the task distinguishes
     close-grade evidence from modes the task treats as preliminary, diagnostic, or progress
     findings, require a positive
     `synthesis_trigger.mature_quorum_fraction`; `0.0` disables the mature
     normal-completion gate and allows raw information density to close a
     generation. Repair this mismatch before unattended launch for newly
     initialized tasks. For an existing task, require explicit current-user
     confirmation of evidence-blind close semantics or route it through
     task-initialization repair; do not describe `0.0` as merely disabling an
     optional early-assessment path. If the task does not require ratio-gated maturity
     and has no other separate close-grade evidence contract, warn and continue
     rather than inventing a mature gate.
   - confirm new task descriptors use `dig_lite.generation_scope: initial_only`
     by default and declare `quality_diversity.initial_generation_enabled` and
     `quality_diversity.later_generations_enabled` independently. Do not infer
     that missing DIG artifacts after absolute gen0 are a launch failure.
     Confirm later-generation QD is routed through the configured single-PI or
     Multi-PI synthesis path as soft allocation guidance. Re-check independent
     DIG diagnostic-slot and constructive-mix settings after any CLI cohort-size
     override; warn on a contradictory allocation, but do not silently rewrite
     it or block a resolved legacy task solely for that reason.
   - for QD-enabled tasks, confirm `evaluation.diversity_dimensions` is
     non-empty, PI/Chair contracts plan those axes under `planned_dimensions`,
     and peer prompts require actual implemented values under
     `design_dimensions`. Planned values must never be copied into evidence when
     actual values are missing.
   - confirm every metric used for frontier ordering, report dimension winners,
     baseline comparisons, or charts has an explicit task-owned direction.
     Unknown directions are not launch blockers for display-only diagnostics,
     but they must not silently default to maximize.
   - confirm every verified measurement retained under `assets/baselines/` is
     explicitly wired into `task.yaml:baselines` with its metric and direction.
     A `praxist resolve` warning about measured-looking assets with an empty
     baseline contract is a harness wiring defect for a newly initialized task;
     verify the provenance and repair the declaration rather than auto-trusting
     arbitrary files or hiding the warning.
   - when the user-approved maturity definition uses effort/coverage ratios,
     confirm the evaluator emits task-defined `effort_ratio` and
     `coverage_ratio` with `require_ratio_gate: true`. If the approved contract
     uses another explicit maturity definition, preserve it instead of adding a
     ratio gate. Configure task-owned stage labels only when the evaluator
     actually has staged protocols; labels do not supply global maturity by
     themselves.
   - execute the evaluator's real summary writer for a completed
     task-authorized terminal mode and an incomplete mode. Check the complete
     field combination against the task-owned policy rather than rejecting
     words such as `capped`: reaching a declared fixed budget may be mature,
     while stopping before it must remain incomplete. Repair genuinely
     contradictory decisions without weakening the maturity policy.
   - for tasks that require mature/complete evidence before normal close,
     record the most expensive ordinary evaluator as
     `estimated_heavy_eval_minutes`, record the task-authorized close
     protocol's observed p90 as `estimated_close_grade_eval_minutes`, and
     require `estimated_close_grade_eval_minutes * safety_factor < effective
     close horizon - drain margin`, using at least a 30-minute drain margin
     unless task measurements justify more. A user-approved reduced protocol
     may be the close-grade evaluator while a longer optional protocol remains
     available. Recalibrate the generation and synthesis bounds when the
     inequality fails; do not launch a task whose required evidence is
     unreachable by construction.
   - confirm task-owned evaluator, trainer, config, and harness paths are
     task-root relative. Verify every intentional external absolute path, then
     run the public evaluator from the task root and a run-like subdirectory
     through the declared task interpreter. The task child must not import
     runner-owned `PYTHONPATH`/`PYTHONHOME` state unless the task explicitly
     declares and validates those variables.
   - confirm recursive result discovery supports `summary.json`,
     `evaluation_summary.json`, `eval_summary.json`,
     `tiered_eval_summary.json`, and `custom_*_tiered_eval_summary.json`
     (`result_summary.json` is a compatibility name), with lane, maturity,
     ratio, protocol, and diagnostic metadata available to the materializer.
   - when central scheduling owns evaluator launch, inherit the current
     task-initialization result-attribution check: each submission declares a
     result-specific output directory, and the canary's materialized finding
     retains its canonical generation and peer owner rather than relying on a
     run-wide results `cwd`.
   - when the task declares close-grade evidence, use
     `mature_supply_fraction: 0.25` and
     `mature_supply_redundancy: 3.0`, plus
     `supply_lease_seconds: 600` and
     `mature_assessment_min_completion_probability: 0.25`, for newly
     initialized tasks unless the
     resource plan records a measured override. Keep the hard mature close
     quorum positive whenever the user-approved task contract defines
     close-grade evidence. When the user explicitly chooses information-density
     close with no close-grade mode, preserve `0.0` and do not manufacture
     mature-supply debt.
     Mature-priority supply is advisory and cannot replace the close gate. If
     `Q` mature results are not physically feasible, recalibrate the generation
     bound, concurrency, or user-approved task protocol, or let the bounded
     safety/cohort-drained path record insufficient maturity; do not use `0.0`
     as a deadlock workaround. Treat the supply lease as a bounded submission
     response window, not a runtime cap on admitted work.
   - before launch, exercise the generated closing policy with a lightweight
     task-local regression: raw finding density with zero mature results must
     not be normal-success close when a maturity distinction exists; below
     quorum must preserve mature top-up admission; quorum completion may close
     after active work drains; and safety-cap/cohort-drained exits must remain
     available with explicit insufficient-evidence telemetry.
   - confirm new Gems configuration uses
     `selection_policy: mature_evidence_top_k` and task-derived
     `min_mature_eval_units`, plus `evidence_stage_min_units` when staged
     evidence has cumulative task-owned unit requirements; compatibility-only
     historical maturity keys are not valid task-initialization output.
   - confirm newly initialized tasks use the task-init central resource
     scheduler contract when Praxist owns observable local process launch, derived
     from an unchanged baseline observation:
     task-owned profiles, host-wide total concurrency, and measured pressure for
     the backend used by the unchanged baseline, with no per-job CPU-core
     reservation and no implicit accelerator-to-CPU fallback. A documented external cluster, remote
     service, license queue, or task-native scheduler may retain bounded
     external/legacy ownership; do not force central mode where Praxist cannot own
     or observe launch. Legacy tasks may warn and continue.
   - before launch, perform a task-runtime accelerator coherence check without
     changing the baseline or comparing CPU and accelerator speed. Record
     separately: the current user's and task's explicit backend intent, host
     accelerator inventory as available/unavailable/unknown, the selected task
     interpreter's backend capability, the backend actually exercised by the
     public evaluator canary and recorded in its effective summary, and the
     scheduler default resource profile. A blocked or missing vendor utility
     does not prove that the host lacks acceleration, and a CPU-only framework
     build proves only that the task interpreter cannot use that backend. When
     the current user or task explicitly requires an accelerator, an unresolved
     mismatch across these layers is launch-blocking and must return to
     task-initialization repair. When no accelerator has been requested or
     declared and the public baseline intentionally uses CPU, unified memory,
     an external service, or another task-owned backend, report any extra host
     capability as advisory and continue unchanged. Never install or upgrade
     task dependencies, force an accelerator, or rewrite evaluator/profile
     defaults solely because the host exposes one.
   - confirm the default profile matches the public evaluator's normal resource
     shape, ordinary analysis is not submitted as an experiment, explicit
     measured profiles are used where supported, and event-driven directed idle
     resource-supply leases are enabled. If a complete evaluator
     declares multiple accelerators, require proof that its natural independent
     units actually consume distinct assigned devices. Treat simulator, license,
     memory, I/O, and external-service limits as task-owned concurrency bounds.
   - only when task initialization detected and selected the Praxist-managed
     NVIDIA/CUDA backend, require its UUID handoff result:
     `PRAXIST_ASSIGNED_GPU_UUIDS` remains authoritative through the
     evaluator-to-compute-child chain, applicable contract tests pass, and the
     bounded non-zero physical UUID check passed when multiple devices were
     available. Do not launch a newly generated task that silently rewrites the
     assignment to local ordinal `0`. CPU-only, unified-memory, task-managed,
     and other accelerator backends use their own verified handoff contract and
     must not be forced through CUDA/UUID checks. Legacy tasks may warn and
     request a task-level compatibility repair rather than modifying Praxist core.
   - run the shortest valid scored evaluator path through its real summary
     writer, then validate that actual output with `praxist resolve <task_path>
     --result-summary <summary_path>`. Confirm it emits finite `effort_ratio`
     and `coverage_ratio` fields when the task declares a required ratio policy,
     and verify a summary-to-finding round trip preserves both values.
     Require ratios directly from result-finding instructions only for
     standalone findings without a canonical summary reference. Repair initialization only when the declared
     policy requires those fields and the task cannot produce them, or when the
     user explicitly chooses repair. Do not launch a newly generated task with
     a required ratio gate when this preflight fails; repair the evaluator or
     obtain explicit user approval to disable the gate.
   - complete the current task-init evaluator fan-out preflight before launch:
     task-appropriate build/load/startup and public-interface validation, a
     one-unit canary through the real public evaluator and scheduler path when
     applicable, canonical-summary validation, and summary-to-finding
     projection. If the task declares an
     independently trusted evaluator, also run its task-owned tamper/attestation
     checks. Never impose external-attestation semantics on peer-authored tasks.
   - inherit the current task-initialization effective-configuration provenance
     check whenever non-code settings can change a treatment; do not replace it
     with code-only replication or make it a gate for ordinary experiments.
     Confirm the evaluator records resolved values after defaults and parsing,
     and that omitted-default and explicit-default fixtures produce the same
     digest while a changed effective value does not. Derived-child preflight
     must use that same task-owned schema instead of requiring peers to copy a
     parent process environment by hand.
   - confirm compact evaluator summaries use a supported recursive
     `results/**/` summary filename, reuse a stable top-level candidate identity
     across stages (or an explicit child-result ID), and expose structured lane,
     maturity, and diagnostic metadata so Praxist can materialize canonical findings.
   - when the task explicitly permits background complete evaluation, confirm its
     prompt uses the central `protected_pids launch` submission facade with
     `$PRAXIST_PEER_ID`, a stable semantic tag, declared profile, and work class
     rather than raw `&` or `nohup`, so close-out can drain the evaluator before
     final Frontier resolution. A corrected failed/rejected submission keeps
     that tag and uses the explicit `--retry-terminal` flag; ordinary duplicate
     submissions remain idempotent. With `evaluation.launch_guard.enabled: true`,
     close-out freezes new evaluator/script launches but preserves already
     started work for natural drain. Also confirm no prompt treats a
     runtime-private `tasks/<task-id>.output` file becoming non-empty as
     completion; empty output is valid, so the runtime task notification and
     exit status own that fact. Warn and continue for legacy tasks unless the
     user requested pre-launch repair.
7. Validate through Praxist:

   ```bash
   praxist resolve /path/to/task
   praxist resolve /path/to/task --result-summary /path/to/actual/evaluation_summary.json
   ```

   Use `uv run praxist resolve ...` only when `praxist` is not on `PATH` and the current
   directory is a Praxist source checkout.
8. Start detached:

   ```bash
   praxist start --task-path /path/to/task --daemonize --json
   ```

   Preserve any user-specified `--runtime`, `--model-provider`, `--model`,
   `--cohort`, `--generations`, or strategy overrides. If no override is given
   and `DEEPSEEK_API_KEY` exists, prefer DeepSeek V4 Pro plus the Claude SDK runtime
   using the launch guidance from task initialization.

   Preserve any explicit task/user `agent.reasoning_effort` selection. New or
   repaired harnesses use `max` unless the user requested `auto`, `off`, `low`,
   or `high`; takeover must not infer effort from the task domain or implement
   provider-specific thinking parameters in task files.

   For the default `agent_runtime:claude_sdk`, verify
   `claude-agent-sdk==0.2.136`. If the selected runtime is
   `agent_runtime:codex_sdk`, verify `openai_codex` and `mcp` import in the
   Praxist environment and confirm `openai-codex==0.147.0`,
   `claude-agent-sdk==0.2.136`, and, for DeepSeek/OpenRouter,
   `codex-relay==0.5.5`. For native OpenAI, an exported `OPENAI_API_KEY` wins;
   otherwise require the SDK-bundled Codex binary to report
   `Logged in using ChatGPT`. Do not copy auth files or treat an API-key login
   as saved ChatGPT authentication. The peer runtime uses the official Python
   SDK and a long-lived local app-server. Praxist starts any required run-scoped relay.
   Codex-native mode and OpenRouter routes automatically use the
   runtime-owned lossless context-efficiency policy: finding-only wakeups are
   coalesced, control/resource events remain immediate, and canonical artifacts
   remain available by exact reference. Do not add task-local compression,
   duplicate memory stores, or cache/session settings. Direct DeepSeek behavior
   remains unchanged.
9. Confirm with `praxist status --json`. Report run id, PID, run directory, task
   path, model/provider/runtime, and how to query status.
10. Complete the launch handoff by clearly reporting the independent foreground
    TUI command. Prefer `extra.monitor_command` from the start JSON when present;
    otherwise derive it from the returned `run_id`:

    ```bash
    praxist --monitor --run-id <run_id>
    ```

    Do not open the monitor automatically. The monitor is a read-only operator
    view and must not edit task artifacts or affect the Praxist run. `Ctrl-C` exits
    only the monitor interface; the run remains active. Mention
    `praxist --monitor --plain` only when the user needs non-interactive or
    append-friendly text output. Other direct selectors are `praxist --monitor` and
    `praxist --monitor --latest`.

## Refusal Conditions

Stop instead of launching when:

- no runnable research project can be identified;
- required local code, dataset, simulator, or runtime is missing;
- no valid task directory exists and task initialization aborts;
- the selected task path is ambiguous or not confirmed by the user;
- `praxist resolve` fails;
- the selected task has an unreachable configured parent-eligible lane or its
  lane-routing regression fails, regardless of whether the task is new or
  pre-existing;
- the current user or task explicitly requires an accelerator but the task
  interpreter, public evaluator canary, and scheduler profile cannot establish
  one coherent execution backend;
- required provider/runtime configuration is absent and no fallback is declared.

## Output

End with a compact launch summary:

| Item | Value |
|---|---|
| Research project | absolute path |
| Task path | absolute path |
| Praxist command | redacted command |
| Run id | id or unavailable |
| Run dir | absolute path |
| PID/state | value from start/status |
| Provider/runtime | selected values |
| Lane routing | task-local regression result and reachable parent lanes |
| Monitor | `praxist --monitor --run-id <run_id>`; foreground TUI is not opened automatically |
| Next command | `praxist-control` with a research-status request; `praxist --monitor --run-id <run_id>` |

Do not edit Praxist core, guard code, provider credentials, or existing run
artifacts in this skill. Do not run long baseline measurements unless task
initialization explicitly asks the user and receives approval.
