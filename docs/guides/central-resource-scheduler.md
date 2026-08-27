# Central Experiment Scheduler

Praxist can make one run-level scheduler the supported launch path for task
experiments. Peers submit task-defined work; they do not choose devices or create
scheduled task processes themselves.

## Why It Exists

Resource arithmetic is simple only when one component owns four facts:

1. which scientific experiment was submitted;
2. which attempt may start;
3. which process group and physical accelerator were assigned;
4. when that allocation is released.

The scheduler owns queueing, final `Popen`, process environment, infrastructure
retry, and release. Frontier promotion, evidence maturity,
[Deep Innovation Gate (DIG)](deep-innovation-gate.md),
[Quality-Diversity (QD)](qdig-cohort-allocator.md), Principal Investigator (PI)
and Chair planning, Gems, and generation policy remain separate.

## Task Contract

```yaml
compute_budget:
  resource_scheduler:
    mode: central
    initial_concurrent_experiments: 2
    min_concurrent_experiments: 1
    max_concurrent_experiments: 8
    supply_signal_enabled: true
    supply_idle_samples: 3
    supply_lease_seconds: 600
    mature_supply_fraction: 0.25
    mature_supply_redundancy: 3.0
    mature_assessment_min_completion_probability: 0.25
    exploration_reserve: 1
    infrastructure_retries: 1
    default_profile: gpu_work
    profiles:
      cpu_ordinary:
        accelerator: cpu
        pressure_domains: [cpu, memory, io]
      gpu_work:
        accelerator: gpu
        gpu_count: 1
        gpu_memory_gb: 20
        gpu_utilization_pct: 45
        pressure_domains: [cpu, memory, io]
```

Task initialization obtains these values by running the unchanged public
baseline and observing it externally. It must not create artificial CPU-only
and GPU-only rewrites. Observation is a timestamped process-lifetime series,
normally sampled every 100-200 ms. Short tasks are safely repeated or replaced
by a longer unchanged representative unit; a teardown `0%` sample or fewer
than ten useful samples does not establish zero GPU demand. Utilization uses a
robust upper estimate of full-lifetime means, while VRAM uses the observed peak
plus modest headroom. Ambiguous or undersampled demand remains unknown and
therefore exclusive. The default profile matches the public evaluator's
normal resource shape because runtime-assisted submission may omit an explicit
profile; ordinary analysis commands should not enter the experiment queue. CPU
profiles do not reserve cores per experiment; the operating system shares CPU
time and Praxist changes total experiment concurrency from live pressure. A profile
that declares CPU, memory, or I/O pressure is not newly admitted while that
declared domain exceeds its high-pressure threshold; gradual concurrency
adjustment remains the recovery path once pressure falls. GPU
profiles use two independent per-device hard limits: declared utilization plus
observed external load remains at or below 100%, and declared peak memory plus
observed external memory remains below 95% of physical VRAM. A settled job's
declared envelope is retained across setup, CPU, accelerator, evaluation, and
waiting phases because a later phase may return to its peak. Driver activity is
reported separately for diagnosis; it does not authorize transient
oversubscription. Feasible devices are ordered by the tighter of their compute
and memory headroom. Omit either GPU demand
field when demand is unknown; the profile then receives exclusive placement.

Omitted scheduler fields use documented defaults. In `mode: central`, explicit
unknown keys, misspelled booleans, malformed numbers, profiles, or pressure
domains fail during task resolution instead of silently changing scheduling
policy. Valid numeric values outside a bounded policy range retain the documented
normalization behavior.

CPU and GPU profiles are not interchangeable unless the task explicitly says
they are scientifically equivalent. Praxist never changes a failed GPU experiment
into CPU work implicitly. Omitting `--profile` selects `default_profile`; an
explicit unknown profile is rejected instead of silently using another device
class.

## Mature Evidence And Idle Supply Feedback

The scheduler adjusts capacity, but it cannot invent scientific work. When an
open generation has unused concurrency slots, the queue cannot fill them, and
consecutive host samples show headroom in task-declared pressure domains, it
writes short-lived directed leases under `gen_N/resource_supply/`. Completed
peer sessions register as idle and watch only their own lease file through the
existing event-driven loop.

The evidence controller uses the task's existing maturity policy and canonical
result store to maintain:

```text
Q = max(1, ceil(cohort_size * mature_supply_fraction))
M = unique mature results already published for this generation
D = max(0, Q - M)
A_target = min(cohort_size, ceil(mature_supply_redundancy * D))
```

When a task deliberately configures a larger hard mature close quorum, that
larger target replaces `Q` for first-wave and debt supply; otherwise the close
contract could never receive enough mature work. In that mode `M` is distinct
mature peers, matching the existing peer-quorum semantics; without a hard peer
quorum, `M` is unique mature results. Setting mature supply fraction
or redundancy to zero disables maturity-priority supply even when a hard close
quorum exists.

The inverse is also important: a positive mature supply fraction does not
create a hard close gate. Tasks that distinguish close-grade evidence must set
a positive `synthesis_trigger.mature_quorum_fraction`; otherwise raw
information density can normal-close the generation while maturity debt
remains.

Queued/running mature semantic experiments and outstanding mature-priority
leases count toward `A_target`; retries retain one semantic identity. The
default `0.25` and `3.0` values are calibrated general defaults rather than
domain truth. Setting either value to zero disables mature-priority supply
without disabling ordinary idle backfill.

During assessment, ordinary admission stops while mature top-ups remain
eligible when their calibrated probability of finishing before the generation
deadline is at least `mature_assessment_min_completion_probability`. The
compact log-normal calibration uses successful wall time divided by declared
ETA, with a neutral prior that avoids overconfidence from the first few jobs.
Before assessment, existing deadline admission remains unchanged. An unknown
ETA remains unknown rather than being converted into false precision.

Each lease names currently admissible profiles, carries `mature` or
`frontier_followup` priority, and expires if unused. `supply_lease_seconds`
sets the bounded response window (default 600 seconds, normalized to
180-3600). Expiry limits when the peer may submit an existing plan; it does not
limit the runtime of an experiment admitted before expiry. A peer may
respond with at most one already justified experiment selected by the current
research plan, evidence priorities, and exploration commitments. The signal
selects an evidence class but does not choose a hypothesis, create variants, or weaken
evaluation standards, expose a device assignment, or bypass the central queue.
The scheduler records a short-lived host-wide capacity claim so concurrent runs
cannot promise the same slot; the final launch still performs normal admission
and assigns the actual device UUID. A real queued job atomically preempts its
own or physically conflicting speculative claims, so idle feedback cannot
delay submitted research while unrelated CPU-only runs remain independent.
Once published, the claim remains stable across later pressure samples until it
is consumed, expires, is atomically preempted by real work, the generation
closes, or the run stops. Final launch always rechecks live pressure, so this
bounded response stability does not bypass admission. `supply_signal_enabled`
disables this feedback, while
`supply_idle_samples` controls its consecutive-sample requirement. Outstanding
leases count against supply capacity, so N free slots wake at most N peers. A
peer that declines an unused lease enters a bounded same-priority exponential
cooldown rather than losing eligibility for the rest of the generation. A new
experiment submission, a changed priority, or a new generation resets that
backoff; retained idle registrations allow later maturity debt to wake the peer
again without a long polling delay.

`resource_supply.stats` reports `conversion_rate=consumed/granted` and attributes
known-priority counts under `by_priority.mature` and
`by_priority.frontier_followup`. Unused
offers terminate as `declined`, `expired`, or `revoked`; a submission carrying
an already expired locator is `stale_submission`, while `reuse_ignored` is
reserved for a lease that was genuinely consumed once. These are operational
facts only and do not replace mature-result quality or quantity. Grant and
terminal transitions are durable before they affect the live lease; restart
replays the same event ledger and revokes any grant interrupted before a
response, so run-wide conversion does not reset with the scheduler process.
If host claim release is temporarily unavailable, the lease remains visible with
`release_pending: true`, is excluded from actionable maturity commitments, and is
retried by reconciliation instead of becoming hidden capacity.

At the first wave, up to `Q` peers receive direct-mature advice while at least
one peer retains exploration when the cohort has multiple peers. Once mature
commitments satisfy `A_target`, spare leases return to Pareto-relevant
follow-ups and then already planned scouts. Scientific selection remains with
the research loop.

This mechanism is resource-type neutral. CPU, memory, and I/O pressure come
from live host observations; accelerator placement remains governed by
measured memory/utilization profiles. Simulator instances, licenses, remote
services, and other bounded resources remain task-owned limits expressed in
the evaluator or through a conservative global experiment cap.
The goal is to keep the measured bottleneck supplied without forcing every
resource to a fixed utilization percentage.

## Natural-Unit Parallelism

Task harnesses should identify independent complete-evaluation units such as seeds,
folds, scenarios, simulator instances, datasets, benchmark cases, or
restart trials. A multi-accelerator profile is valid only when the evaluator
actually distributes those units across every assigned physical GPU UUID,
preserves binding through all descendants, aggregates independently of
completion order, and drains the complete process group. Declaring
`gpu_count > 1` without that implementation does not create parallelism.
Do not count the same work both as multiple top-level scheduler experiments and
as internal child units of one experiment.

Prefer the smallest unit that is independently valid, retryable, and
aggregatable without changing the scientific protocol. A long wrapper that
serially mixes setup, accelerator work, CPU evaluation, and waiting remains one
lifecycle job. Where it cannot be split safely, expose monotonic task-owned
progress. A task may fail fast after repeated identical infrastructure or
implementation errors prove the remaining units non-runnable, but must retain a
structured failure summary. Low scores, negative findings, and heterogeneous
scientific failures are not fail-fast conditions.

## Submission

```bash
PYTHONPATH="$PRAXIST_WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PRAXIST_RUNNER_PYTHON" -m praxist.plugins.workflow_stages.research_loop.backend.protected_pids launch \
  --run-dir="$PRAXIST_RUN_DIR" \
  --peer="$PRAXIST_PEER_ID" \
  --tag=<stable_semantic_experiment_id> \
  --profile=<declared_profile> \
  --work-class=<scout|ordinary|mature> \
  --eta=<seconds> -- <task evaluator command>
```

The tag identifies science, not execution syntax. Variant/protocol/data
coverage/seeds/tier belong in it; timestamps, output paths, retry numbers,
logging flags, and harmless command spelling do not. Repeated submissions with
the same identity share one queued, running, or completed job. Only exit code
75 is an automatically retryable infrastructure failure. A corrected request
whose existing job is `failed` or `rejected` must use the same scientific tag
with `--retry-terminal`; this creates a new attempt under the existing semantic
identity. An identical retransmission of an already accepted retry remains
idempotent while queued, running, completed, or `drained_unknown`; a changed
request with the flag is rejected in those states. Without the flag, a terminal
duplicate is reported explicitly instead of silently rerunning. Do not append
arbitrary retry text to a scientific tag.
Pre-launch admission timeouts and transient accelerator-probe rejections are
the exception: no experiment attempt ran, so they release the reservation and
the same scientific request may be submitted normally after capacity or host
inventory recovers.
When a peer's compatibility cap is already occupied, another central submission
from that peer remains queued until the active process group drains. Other peers
may continue to launch, and the blocked job does not consume an attempt or
create a capacity-failure record.

The final child receives an immutable attempt directory plus exact accelerator
variables. Task descendants must preserve them:

- `PRAXIST_EXPERIMENT_ID`
- `PRAXIST_EXPERIMENT_ATTEMPT_ID`
- `PRAXIST_EXPERIMENT_ATTEMPT_DIR`
- `PRAXIST_RESOURCE_PROFILE`
- `PRAXIST_ASSIGNED_GPU_UUIDS`
- `CUDA_VISIBLE_DEVICES`
- `NVIDIA_VISIBLE_DEVICES`

The built-in Linux observer inventories physical NVIDIA GPU UUIDs. It does not
advertise automatic MIG-slice discovery or placement. A task wrapper may remain
compatible with an opaque externally supplied identifier, but task
initialization must not claim that standard central scheduling validated MIG
placement.

The submitted evaluator may create ordinary worker or trainer descendants;
they inherit the same process group and resource envelope. It must not submit
each internal worker as a second top-level experiment. If a compatibility
launcher is encountered inside an active attempt, Praxist keeps that child inside
the existing allocation instead of recursively queueing it. This reuse is
verified against the run-owned attempt directory, committed READY/GO handshake,
the caller's live process group, and the scheduler's current in-memory attempt
state. Mutable attempt environment variables or copied handshake files alone do
not establish ownership.

For an active central run, `<run>/resource_scheduler/endpoint.json` is the
run-owned launch authority. Peer shell commands may not downgrade that run to
legacy launching by changing scheduler environment variables. The environment
endpoint remains the compatibility source for legacy callers and runs that do
not have run-owned endpoint metadata.

### Optional Managed NVIDIA/CUDA Descendant Binding

This subsection applies only after the task's unchanged baseline was observed
to use, and task initialization explicitly selected, the compatible
Praxist-managed NVIDIA/CUDA backend. `PRAXIST_ASSIGNED_GPU_UUIDS` is then the authoritative ordered physical GPU
assignment. A task harness must preserve that exact value in
`CUDA_VISIBLE_DEVICES` and `NVIDIA_VISIBLE_DEVICES` across evaluator, trainer,
worker, shell, and container boundaries. A framework may use local `cuda:0`
inside the mask, but a launcher must not write that local ordinal back into a
new child's visibility environment. Missing masks may be restored from the Praxist
assignment; conflicting masks must fail clearly instead of silently rebinding.

Generated tasks using this backend should carry fast UUID, multi-UUID, missing
mask, conflicting mask, standalone, and forced-CPU contract tests. On a host
with multiple usable GPUs, launch readiness also includes a bounded non-zero
UUID parent/child CUDA check and driver-observed PID-to-UUID comparison. This is
a placement-integrity check, not a CPU/accelerator benchmark or training run.
CPU-only tasks, unified-memory systems, task-managed devices, and other
accelerator backends are valid scheduler paths and do not inherit this UUID
contract merely because an accelerator is present.

Before the evaluator executes, a small local launch barrier waits until the
semantic intent, process group, resource allocation, and protected-process
record are durable. Resume rebinds that same allocation before releasing the
barrier, so a crash cannot turn one semantic experiment into duplicate work.

## Timing And Close

Complete mature evaluations should begin early, not only after assessment
reports mature debt. `work-class=mature` has queue priority, while
`exploration_reserve` prevents mature work from consuming every slot when
exploration is queued.

When a configured mature close quorum is still missing at assessment, Praxist
stops ordinary queued/new admission but keeps deadline-safe mature top-ups
eligible. Assessment is not `CLOSING_SIGNAL`. Once the quota is met, or the
generation reaches its safety bound, the normal strict close path takes over.

At generation close, Praxist freezes the scheduler queue before writing
`CLOSING_SIGNAL`. Queued/new work is rejected; already-running process groups
continue to drain and publish evidence. Once protected work is gone, the
existing adaptive drain grace bounds agent-only cleanup and passive tool waits;
it does not kill evaluator processes. Runtime-private background-task output
files are never used as lifecycle facts because empty stdout/stderr is a valid
successful result. `praxist stop` similarly freezes all new admission before
discovering and terminating scheduler-owned process groups. The existing run
shutdown sentinel is the primary fence; a confirmed central-scheduler freeze is
an equivalent fallback. If neither succeeds, that run is left untouched and
reported in `failed_run_ids` with a nonzero CLI result. Fenced runs receive a
bounded rescan until scheduler-owned and exact run-environment descendants are
stably absent. A union bulk stop also skips its independent process-name scan
when any registry run could not be fenced, because portable hosts may not expose
enough evidence to distinguish that run's orphan from an unrelated controller.
Explicit process-scan-only operation remains unchanged. A framework-owned
`peer_workspaces/` cwd is a narrow fallback for
a descendant that cleared its environment; the run root by itself is not process
ownership. Process identities are revalidated before every signal, using the
portable `ps` start identity when procfs is unavailable, and unrelated operator
or monitor processes are not selected by broad command matching.
The live central scheduler supplies its complete active process-group set even
after a launcher exits. A legacy manifest with a live group but no verifiable
launcher is not guessed at or marked stopped; its run is returned in
`failed_run_ids` for explicit operator follow-up.

## State And Compatibility

Current state is a compact derived view at:

```text
<run>/resource_scheduler/status.json
```

`running` is a lifecycle count: the wrapper or a descendant process group is
still alive. It is not a GPU-activity count. `running_activity` summarizes the
separate observation, and active jobs may include `resource_activity` with
`gpu_compute_active`, `gpu_context_idle`, `gpu_context_present`,
`no_gpu_process_observed`, `gpu_process_attribution_unavailable`,
`non_gpu_allocation`, or `unknown`. These fields are derived telemetry only and
never alter maturity, ranking, retries, or result validity. A single
`no_gpu_process_observed` sample can be setup, CPU work, evaluation, or a phase
transition. `gpu_process_attribution_unavailable` means the accelerator
reported a process that could not be mapped into the scheduler's PID namespace;
the accompanying `attribution` value distinguishes `complete`, `partial`, and
`unavailable` ownership mapping. `unknown` means the underlying observation was
unavailable. Admission remains conservative when ownership cannot be mapped so
shared-device external load is not mistaken for Praxist work. Use progress, logs,
process trees, and result mtimes before diagnosing a stall.

On Linux, a process group containing only zombie or exited members is terminal
work, even though `killpg(..., 0)` can still report that group as present.
Praxist reconciles that kernel state before retaining a running slot, then reaps
the wrapper and releases the allocation. A sleeping or otherwise live process
is not treated as a zombie, and platforms without procfs retain the portable
process-group check.

Attempt logs live under `<run>/logs/experiments/`; immutable attempt metadata
lives under `<run>/resource_scheduler/attempts/`. Existing protected-PID
manifests remain the process-lifecycle compatibility surface used by close,
stop, diagnostics, and late/quarantined result handling.

The small launch-barrier interpreter runs without Python `site` initialization
so a peer's generated `sitecustomize` cannot mistake the trusted READY handshake
for a task write. The barrier preserves the submitted environment unchanged
when it executes the real task command, so Python task descendants still load
the normal runtime guard. Peers never receive write access to scheduler state.

Tasks without `mode: central` retain the legacy launch behavior. Central mode
does not silently fall back to peer-local launching if its service cannot start.
One run-local owner lock prevents two scheduler services from controlling the
same queue. Acknowledged submissions and terminal queue rejections are fsynced
before their in-memory transitions, so restart replay neither loses accepted
work nor revives work rejected by close, freeze, or stop. Runtime environment
values that look credential-bearing by name or value shape are stored only as
hashes.
