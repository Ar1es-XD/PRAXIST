# Research-Loop Flexibility Controls

These controls preserve useful signals while making maturity, retention, and
generation close follow the task owner's declared protocol. Exact task fields
and the recommended combined profile are defined in
[Task Projects](task-projects.md#deep-innovation-gate-quality-diversity-and-gems-defaults).

## Mature Evidence

When a task enables ratio-based maturity, its canonical evaluator summary emits:

- `effort_ratio`: actual evaluation effort divided by the task's mature
  reference effort;
- `coverage_ratio`: completed required evaluation units divided by all
  required units.

Praxist copies these normalized facts into auto-materialized findings. A
standalone finding without a canonical summary reference must carry them
itself. With `require_ratio_gate: true`, missing or non-finite ratios remain
unknown; stage names cannot fill the gap. A task that deliberately uses labels
or completion flags instead may leave ratio gating disabled and define those
semantics explicitly.

Initialization validates one real file from the evaluator's production summary
writer with `praxist resolve --result-summary`. During a run, a durable-looking
result missing required ratios remains visible as a validation signal and
produces one bounded warning; it is not promoted or counted for mature close.

Gems, frontier lanes, reports, and close all consume the same maturity decision.
Praxist counts generic evaluation units and never gives a domain-specific stage
name global meaning.

## Generation Close

`synthesis_trigger.mature_quorum_fraction` controls normal close when a task
distinguishes close-grade evidence:

| State | Behavior |
|---|---|
| Positive quorum reached | Freeze new work, drain active work, synchronize evidence, and close normally. |
| Positive quorum missing at assessment | Fence ordinary admission while deadline-safe mature top-ups remain eligible. |
| Quorum `0.0` | Information density may close normally; valid only when this is the task owner's intended protocol. |
| Safety bound or fully drained cohort | Preserve liveness and record insufficient maturity where applicable. |

The scheduler's mature supply target is advisory and cannot replace this gate.
When close begins, `CLOSING_SIGNAL` blocks every new experiment while already
running protected work finishes. A bounded drain grace lets agent sessions
publish final evidence before `STOP_SIGNAL`; it never kills an active protected
evaluator.

Runtime-owned stdout files are not completion facts because a successful command
may emit no text. Structured runtime completion and exit status own that state.
Task-owned progress files remain valid only when their readiness contract is
explicit.

The committed boundary records close reason, maturity outcome, and optional
peer-mix telemetry. `orchestrator_status.json` exposes that compact state to
status, monitor, and diagnostics.

## Durable Incubator

An incubator lane is a lower-admission, long-term library, not a stricter winner
lane. It can retain protocol-authorized, protocol-passed, non-suspect candidates
that establish a task-defined Pareto point or new high even when the confirmed
lane is full.

Confirmed and mature incubator lanes may be parent-eligible. Preliminary,
diagnostic, suspect, protocol-failed, and other task-declared non-parent modes
remain in validation lanes for follow-up. Optional display/tiebreak axes do not
silently become Pareto dimensions. The complete lane schema, source-routing
rules, and reachability test are owned by
[Task Projects](task-projects.md#frontier-lanes-and-incubator-evidence).

Promotion rejection summaries remain inside
`frontier/frontier_manifest.json`; Praxist does not create a second runtime
fact file.

## Constructive Peer Mix

When enabled, Praxist estimates constructive solution work versus
diagnostic/control work at each committed boundary. The next generation sees
the result as advisory feedback, not a quota or execution gate. Disabling the
feature stops both calculation and prompt injection, including historical
telemetry on resume.

This control is independent of the Deep Innovation Gate (DIG) initial
innovation-slot policy and Quality-Diversity (QD) allocation. Those switches
are described in
[Deep Innovation Gate](deep-innovation-gate.md) and
[Quality-Diversity Allocation](qdig-cohort-allocator.md).

## Launch Freeze

The launch guard cooperates with the central scheduler to freeze queued and new
submissions before `CLOSING_SIGNAL`. It covers training, evaluation, scripts,
shell launchers, and background processes while still allowing result reading,
finding publication, and memory updates. Existing protected jobs drain
naturally.

Tasks may disable the guard only when they explicitly own an equivalent
close-safe boundary. Disabling it does not waive timing feasibility or mature
close requirements. Heavy-work and close-grade runtime estimates remain
separate because a task may permit long optional evaluation while authorizing a
shorter protocol for normal close.

Central submission, semantic retry, queue ownership, and process-group behavior
are defined only in [Central Experiment Scheduler](central-resource-scheduler.md).
