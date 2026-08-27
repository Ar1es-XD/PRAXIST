---
name: praxist-task-initialization
description: Convert an existing runnable computer-based research project into a formal Praxist task project, or repair a task harness that fails task-init validation. Use when a user wants an agent to transform AI algorithm, robotics, control, simulation, SLAM, LLM, optimization, or other executable research code into a Praxist task directory with task.yaml, baseline harness, evaluator, baseline performance records, robust metric/ranking policy, protocol-integrity checks, reachable task-justified durable/Pareto retention lanes, role prompts, audit rules, dataset/simulator metadata, high-value research directions, initial-generation DIG plus independently controlled QD, continuous-evolution/Gems research-loop settings, run-report tooling, and hardware-aware or user-selected fixed Praxist run parameters. The skill requires a project that already runs on the current machine or in an available environment/container. Abort when required code, data/simulator assets, or declared runtime dependencies are missing.
---

# Praxist Task Initialization

Use this skill to turn a user's existing runnable research project into a Praxist task project. The output is a new task directory selected later with `praxist start --task-path ...`; do not edit Praxist core or generic plugins.

## Mandatory Opening Banner

Before scanning or editing, display this prominently:

```text
**IMPORTANT PRECONDITION**
Praxist task initialization assumes:
1. You already have all source code for the research project.
2. The project already runs smoothly on this machine in some environment/container.
3. Required datasets, simulators, benchmark fixtures, or control environments are already present or reachable from this machine.
```

If any item is false, stop and ask the user to provide the missing path, environment, dataset, or simulator before building the task.

When `praxist-takeover`, `praxist-takeover-codex`, or the user invokes this
skill to repair an existing task that
fails a required task-init check, preserve its established scientific
objective, evaluator protocol, metric directions, baseline, and runtime
ownership. Limit edits to the task harness components responsible for the
failed contract, rerun the same validations, and avoid rebuilding unrelated
task files. Ask the user before changing scientific selection semantics that
cannot be derived unambiguously from the existing task and project evidence.

## Inputs

Resolve paths in this order:

1. User-provided research project path.
2. Current agent working directory.

Default output task path is `<research_project>/praxist_task` unless the user provides another output path. Do not mutate the original project except to read it and, if explicitly requested, run its documented smoke/evaluation commands.

## Locate Task Templates

Resolve the template root before reading or copying a scaffold. Do not assume a
source checkout exists after pip installation:

1. In a source checkout, use `<praxist-repo>/templates/tasks/`.
2. Otherwise locate the installed package with `importlib.resources` and use
   `<praxist-package>/resources/templates/tasks/`:

   ```bash
   python - <<'PY'
   from importlib.resources import files

   print(files("praxist").joinpath("resources", "templates", "tasks"))
   PY
   ```

Call the resulting directory `TASK_TEMPLATE_ROOT`. References below such as
`templates/tasks/machine_learning_template` name the source-checkout form; an
installed-only session must read the equivalent directory under
`TASK_TEMPLATE_ROOT`.

Do not use `examples/` as the generic scaffold. Complete examples preserve
task-specific code, metrics, evidence, and resource assumptions. If the user
selects a bundled example, operate only on the writable copy produced by
`praxist examples install`; never create or modify a harness inside Praxist
source or package resources.

## User-Owned Protocol Intent

Protocol strictness is not universal. Before designing the evaluator, maturity
policy, lanes, or close gate, derive one task-owned protocol-intent table from:

1. the user's explicit current instruction;
2. an existing project protocol that does not conflict with that instruction;
3. a scientifically defensible agent proposal only where neither source decides.

The table must list each allowed evaluator mode/stage and whether it may launch,
produce a comparable score, rank candidates, count as mature evidence, supply a
durable parent, or satisfy generation close. Keep the table in an existing
task-facing description or protocol document and encode the same decisions in
the evaluator summary, `evaluation.maturity_policy`, frontier lanes, Gems, and
close settings. Do not create a second runtime fact registry for it.

There is **no Praxist-wide full-protocol-only rule**. If the user explicitly
requests a partial, reduced-coverage, scout, diagnostic, or otherwise
incomplete protocol for ranking, promotion, or the whole run, preserve that
choice, expose its actual effort/coverage and stage in every result, and align
all downstream policies with it. Do not silently upgrade it to a complete
protocol, refuse it merely for being incomplete, or repeatedly ask the user to
reverse an already clear decision. If the user has not decided, recommend a
complete mature protocol plus clearly labeled cheaper signals when justified,
but present that as a proposal rather than a system requirement.

Keep existing schema names for compatibility: `complete_stage_labels` means
"task-declared mature labels," not "globally full-protocol labels." Likewise,
`scored_complete` means the result satisfies the task's declared comparable
evidence contract; it does not erase a reduced mode's actual metadata.

Make each canonical evaluator summary internally coherent under the task-owned
policy. Do not infer incompleteness from a token such as `capped` alone: a
fixed-budget task may explicitly define reaching its configured cap as mature
completion. Instead, verify that completion booleans, status fields, achieved
effort/coverage, and the declared policy produce one unambiguous decision. A
premature stop must remain incomplete; a successfully reached terminal budget
may retain a truthful cap descriptor when the task contract clearly authorizes
it. Exercise both cases through the evaluator's real summary writer in a
task-local regression; never repair a genuine conflict by weakening the
maturity policy.

Only **undeclared drift** is a protocol-integrity failure. A run whose actual
mode, coverage, or effort differs from the task-owned table must remain visible
as a validation/diagnostic signal and must not impersonate a more mature class.
Validate structured evaluator mode and output metadata, not command keywords:
never reject a launch merely because an argv, path, or source file contains
words such as `smoke`, `scout`, or `partial`.

## Abort Conditions

Abort instead of producing a weak task when clear evidence shows:

- necessary source code is missing, e.g. evaluator exists but model/training/simulator interface is absent;
- required dataset, checkpoint, simulator, robot environment, or benchmark fixture is missing;
- the user specified a runtime environment/container and it lacks required imports, binaries, or simulator bindings;
- no executable baseline or no credible evaluation path can be identified.

Report: what is missing, why it blocks Praxist, and whether the user should supply a path, install dependencies, mount data, or name the correct environment.

## Scan Workflow

1. Announce the precondition banner.
2. Identify the project root and output task path.
3. Run the bundled inventory helper:

   ```bash
   python skills/praxist-task-initialization/scripts/project_inventory.py \
     --root /path/to/research-project \
     --out /path/to/output-task/assets/project_scan
   ```

4. Read the generated `inventory.json` and `inventory_summary.md`.
5. Deep-read the project material needed to understand the task:
   - README, docs, design notes, papers, markdown, notebooks, configs.
   - PDFs and reports; extract text when available and inspect figures/tables when relevant.
   - Presentation files or report documents when they explain methods/results.
   - Code for training, inference, model definition, simulator/control loop, evaluation, metrics, and data loading.
   - Dataset manifests, schemas, resolvers, loader code, and non-secret path variables.
   - Simulator adapters and documented simulator launch commands.
   - Existing result/log directories; summarize text logs and structured summaries. For large history, split by directory/time range and use subagents, then merge into one evidence summary.
6. Detect runtime environment:
   - conda envs: `environment.yml`, `conda.yaml`, shell docs.
   - venv/uv/poetry/pip: `.venv`, `pyproject.toml`, `requirements.txt`, `uv.lock`, `poetry.lock`.
   - containers: `Dockerfile`, `compose.yaml`, launch scripts.
   - external runtimes: ROS, MuJoCo, Isaac, Gymnasium, MATLAB, database services, or task-local services.
7. Verify the declared or inferred environment with the lightest safe command: import check, `--help`, unit smoke, dry-run, tiny fixture, or simulator startup check. Do not run a long training or evaluation sweep unless the user asks.

## Hardware And Run-Parameter Planning

Do this before writing `task.yaml`. Praxist task initialization must produce an execution plan, not just static task files.

Default path: estimate parameters from hardware and task cost. If the user
explicitly requests a fixed default profile, a no-load-measurement profile, or
wording equivalent to "skip sizing and use default parameters", do not derive
cohort size and duration from current utilization. Use this fixed research
profile instead and document that it is user-selected rather than measured:

```yaml
generation_policy:
  cohort_size: 8
  max_generations: 20
  per_generation_hours: 2.0   # 120 minutes
```

The fixed profile is an operator convenience, not the normal recommendation.
Still verify that the project can run and that required data/simulator/runtime
assets exist; only skip load-derived sizing.

1. Inspect current hardware with bounded, read-only commands:
   - CPU: use a platform-appropriate summary such as `lscpu`, `/proc/cpuinfo`,
     or `sysctl`.
   - Memory: use `free -h`, `vm_stat`, `sysctl`, or an available portable
     library. Treat macOS unified-memory execution and other unified-memory
     platforms as normal paths and account for their shared capacity.
   - Accelerators: first infer the backend actually used by the unchanged
     baseline, then use a matching read-only telemetry source when available.
     Record backend, device count, capacity, and current load without assuming
     a vendor, discrete GPU, CUDA, or any concrete device model. Absence of a
     vendor utility is not an error and does not imply absence of acceleration.
   - Disk and I/O risk: `df -h` for relevant project/data/output mounts; avoid full-disk scans.
2. Observe the **unchanged public baseline program** under its normal runtime.
   Prefer an existing smoke, one normal evaluation unit, or the shortest
   representative complete-protocol unit. Do not write synthetic CPU-only and
   GPU-only competitors, do not move baseline computation between devices,
   and do not change data loading, worker counts, model code, precision, or
   accelerator visibility merely to classify the bottleneck. Sample external
   CPU utilization/load, RAM, I/O pressure, observable accelerator
   memory/utilization, elapsed
   time, and completion progress while the original command runs. Stop after a
   representative observation if continuing would be expensive. If a tiny
   preliminary check does not create representative pressure, use one larger
   unchanged protocol unit and terminate it after the resource shape is clear;
   never publish a terminated probe as performance evidence.
   Resource observation must cover the process lifetime, not use one snapshot.
   Start the sampler before launching the unchanged baseline child, record
   timestamped accelerator utilization and memory every 100-200 ms when the
   active backend exposes trustworthy telemetry, and stop only after the child
   exits. Sample CPU/RAM/I/O at a
   similarly bounded cadence. For workloads shorter than roughly five seconds,
   use at least three safe repetitions of the exact same workload or select a
   longer unchanged representative unit so startup/teardown cannot dominate the
   samples. Do not concatenate repetitions into a different scientific
   protocol. If neither is safe, record accelerator demand as unknown.
   Record the baseline process tree and backend-visible compute PIDs when
   available so unrelated
   host jobs are not attributed to the task. When attribution is ambiguous,
   postpone calibration or keep the profile unknown/exclusive.
3. Classify the observed pressure, allowing more than one domain and an
   explicit `unknown` result:
   - accelerator-bound: the task's existing accelerator backend is the limiting resource.
   - CPU-bound: host compute or CPU-side orchestration is the limiting resource.
   - unified-memory, memory, I/O, simulator, license, service, or mixed bound.
   - Memory-bound: large datasets, huge replay buffers, large model checkpoints.
   - I/O-bound: heavy log/data loading, large dataset scans, video/image decoding from slow storage.
4. Estimate one experiment's wall time and, for GPU work only, its peak GPU
   memory and sustained utilization from the unchanged run. Compute each
   repetition's time-weighted utilization over the full child lifetime,
   including real CPU/data-loading gaps; use a robust upper estimate across
   repetitions rather than a single instantaneous maximum or the last sample.
   Use the maximum observed VRAM across repetitions plus modest headroom.
   Preserve raw timestamped samples in the initialization experiment directory
   and summarize sample count, cadence, run duration, mean/quantiles, peak, and
   uncertainty in `assets/resource_plan.md`. A zero measured only at teardown,
   fewer than ten useful samples, or inconsistent short-run traces is not
   evidence of zero demand. CPU work does not
   declare or reserve a fixed number of cores: the operating system schedules
   CPU threads and Praxist controls total experiment concurrency from live host
   pressure. If GPU memory/utilization is not measurable, omit those two fields
   so central scheduling gives that profile exclusive GPU placement. Never
   invent precision.
5. When experiments are local child processes, their resources are observable,
   and the evaluator can use Praxist-owned launch, configure
   `compute_budget.resource_scheduler.mode: central` with task-owned named
   profiles. Peers select a profile; only the central scheduler starts the
   process, sets final accelerator visibility, retries infrastructure exit code
   75, and releases resources. If work is owned by an external cluster, remote
   service, license queue, or task-native scheduler that Praxist cannot observe and
   launch safely, preserve that owner and document a bounded legacy/external
   path instead of forcing central mode. A typical central contract is:

   ```yaml
   compute_budget:
     resource_scheduler:
       mode: central
       initial_concurrent_experiments: <conservative measured start>
       min_concurrent_experiments: 1
       max_concurrent_experiments: <hardware/task upper bound>
       supply_signal_enabled: true
       supply_idle_samples: 3
       supply_lease_seconds: 600
       mature_supply_fraction: 0.25
       mature_supply_redundancy: 3.0
       mature_assessment_min_completion_probability: 0.25
       exploration_reserve: 1
       infrastructure_retries: 1
       default_profile: <public evaluator normal profile>
       profiles:
         cpu_ordinary:
           accelerator: cpu
           pressure_domains: [cpu, memory, io]
         gpu_train:
           accelerator: gpu
           gpu_count: 1
           gpu_memory_gb: <observed peak plus modest headroom>
           gpu_utilization_pct: <observed sustained demand>
           pressure_domains: [cpu, memory, io]
   ```

   Use bounded concurrency; do not add a per-experiment CPU-core reservation. CPU is a live
   whole-host pressure and wall-time signal, not a hard per-job packing dimension. GPU average
   utilization and peak VRAM are separate additive per-device limits; never collapse them into a
   single score or a count of occupied GPUs. Profiles describe the complete
   submitted experiment envelope, not transient execution phases; a later GPU
   phase may reuse the declared peak after an observed CPU or setup phase. Do not create implicit
   GPU-to-CPU fallback: declare a separate CPU profile only when the task's
   scientific protocol says CPU and GPU execution are equivalent. GPU sharing
   is allowed when observed memory and utilization leave headroom; unknown GPU
   demand is exclusive. The scheduler may slowly change total concurrency from
   live CPU/memory pressure even when the bottleneck classification remains
   unknown.

   Set `default_profile` to the normal resource shape of the public evaluator,
   because runtime-assisted evaluator submission may omit an explicit profile.
   Analysis, result aggregation, and ordinary shell commands should not be
   submitted as scheduler experiments. Use explicit profiles for alternate
   evaluator classes, but never move the default away from the public
   evaluator's normal shape merely because documented calls usually pass
   `--profile`. Enable the resource-supply
   signal by default. It only
   asks idle peers to submit already justified work when consecutive host samples
   show headroom and the central queue cannot fill available slots; it does not
   invent experiments, relax evidence standards, or bypass generation Closing.

6. Map each evaluator class to its natural independent units before choosing
   profiles and concurrency. Examples include seeds, folds, scenarios,
   simulator instances, datasets, benchmark cases, or restart trials. Record
   for each class: unit count, safe parallelism, aggregation
   order, CPU/memory/I/O demand, accelerator demand, external-license/service
   limits, and whether units may execute independently without changing the
   scientific protocol.

   When a full/mature evaluation contains multiple safe independent accelerator
   units, teach its existing evaluator to consume every scheduler-assigned UUID
   and distribute units deterministically across those UUIDs. Add a wider GPU
   profile only after this complete child-process path is implemented and tested;
   otherwise retain a one-GPU profile. Scouts may remain narrow when appropriate.
   Never claim multi-device support from `gpu_count > 1` alone. For CPU-bound,
   memory-bound, I/O-bound, simulator-bound, licensed, or service-limited tasks,
   keep the same model: central experiment count controls concurrency while the
   task plan records the real bounded resource and enforces its safe maximum in
   the evaluator or a conservative global experiment cap. Do not count work
   both as top-level scheduler experiments and as internal evaluator children.

   Prefer the smallest unit that is independently valid, retryable, and
   aggregatable without changing the protocol. If a coarse evaluator must run
   many serial units, expose bounded monotonic progress and document its
   setup/CPU/accelerator/evaluation phases. Add a task-owned fail-fast rule only
   when consecutive identical infrastructure or implementation failures prove
   the remaining units non-runnable. Preserve a structured failure summary and
   completed-unit counts. Never fail fast merely because valid scores are weak
   or because distinct units have heterogeneous scientific outcomes.

### Optional Managed-Accelerator Handoff Contract

Apply this section only when the unchanged baseline uses a discrete accelerator
backend that Praxist can bind through the following public environment contract.
Do not generate these variables for CPU-only, task-managed, unified-memory, or
other backends merely because a device exists. For a compatible evaluator,
wrapper, trainer launcher, worker launcher, shell bridge, or container bridge:

- `PRAXIST_ASSIGNED_GPU_UUIDS`: authoritative ordered physical GPU assignment;
- `CUDA_VISIBLE_DEVICES`: CUDA visibility mask delivered by Praxist;
- `NVIDIA_VISIBLE_DEVICES`: container/runtime visibility mask delivered by Praxist.

When `PRAXIST_ASSIGNED_GPU_UUIDS` is non-empty, the harness must preserve that exact
GPU UUID list across every child-process boundary. If either visibility variable
is missing, restore it from `PRAXIST_ASSIGNED_GPU_UUIDS`. If a present visibility
variable disagrees, fail the evaluator with a clear accelerator-binding
integrity error instead of silently choosing a device. Framework-local devices
such as `cuda:0` are correct *inside* the inherited mask, but must never be
written back to a descendant's visibility environment as host ordinal `0`.
Legacy lease markers may remain compatibility signals but must never be the
only way the harness recognizes a Praxist assignment.

Only standalone mode, where no Praxist assignment is present, may construct a
visibility mask from an operator-selected integer device. Explicit CPU mode
must clear Praxist/CUDA/NVIDIA accelerator assignment for that child. Do not add
automatic GPU-to-CPU fallback unless the task protocol explicitly declares
both executions scientifically equivalent.

Implement this handoff once in the smallest existing evaluator/launcher helper
and reuse it throughout the process chain. Do not copy resource-binding logic
from an older task. Inspect Python launchers with AST-aware analysis where
practical and inspect shell/container wrappers too. Treat assignments such as
`CUDA_VISIBLE_DEVICES = "0"` or `str(gpu_id)` after a Praxist assignment has been
observed as launch-blocking harness defects.

Add focused task-owned contract tests, preferably to an existing harness test
module, covering:

1. one UUID is preserved exactly;
2. an ordered multi-UUID mask is preserved exactly;
3. missing CUDA/NVIDIA masks are restored from `PRAXIST_ASSIGNED_GPU_UUIDS`;
4. a conflicting existing mask fails loudly;
5. standalone integer selection remains compatible;
6. explicit CPU mode clears accelerator visibility;
7. evaluator -> trainer -> worker propagation keeps the same physical mask.

For a task that explicitly selects the Praxist-managed NVIDIA/CUDA backend on a
host with at least two usable devices, run one bounded non-zero-device
launch-readiness preflight. Select a non-zero device by UUID, pass it through
the real evaluator-to-compute-child path, create only a tiny CUDA allocation,
and compare the compute PID's physical UUID from the available driver telemetry
with the scheduler assignment. This is a binding test, not CPU-vs-accelerator
benchmarking, bottleneck detection, or training. Stop it promptly and verify
the process and allocation are gone. If the selected backend cannot perform
this physical check, run all applicable environment-contract tests and record
physical verification as unavailable; never claim it passed. A task requiring
this managed multi-device placement is not launch-ready when its binding
preflight fails. CPU-only tasks, unified-memory platforms, task-managed
accelerators, and other backends follow their own observed launch contract and
must not be forced through CUDA/UUID checks.

For the first real run of a newly generated compatible Praxist-managed
NVIDIA/CUDA harness,
perform one bounded,
read-only consistency audit after several jobs start: compare scheduler UUIDs,
evaluator/trainer/worker visibility masks, and driver-observed process UUIDs.
Record the result in the initialization report and stop auditing after a clean
sample. A mismatch is a task-harness compatibility failure, not evidence about
the research variant.
7. Set Praxist parameters to use the scheduler without obvious oversubscription:
   - `compute_budget.per_experiment_gpu_hours`: estimated full-eval GPU-hours
     only when that accounting unit actually applies; otherwise omit it or use 0.
   - `compute_budget.max_parallel_runs_per_peer`: keep as a legacy compatibility
     cap; central mode's host-wide limit is authoritative.
   - `generation_policy.cohort_size`: number of peers that should keep the measured bottleneck highly utilized without crossing pressure, safety, license, service, simulator, or memory limits.
   - `generation_policy.per_generation_hours`: safety upper bound long enough for one serious peer loop: implement, smoke, evaluate, publish, and drain.
   - `generation_policy.max_generations`: long enough for several research
     cycles unless the user requests a smoke-only task.
   - task-local evaluator timeouts and tier expected durations: 2-3x the measured or estimated runtime of each tier, with a hard timeout for stuck training/simulation.
8. Reserve real work capacity for both evidence maturity and exploration.
   Set `mature_supply_fraction: 0.25`, `mature_supply_redundancy: 3.0`, and
   `mature_assessment_min_completion_probability: 0.25`
   unless measured task physics justify an explicit override. For `P` peers the
   default evidence target is `Q=max(1, ceil(P*0.25))`; the controller maintains
   at most `min(P, ceil(3*(Q-M)))` queued/running mature plans and
   mature-priority leases, where `M` comes from canonical effort/coverage
   results. This is bounded tail-risk redundancy, not three GPU slots per
   result and not permission to create filler work.
   Require peer prompts to submit complete-protocol work with
   `--work-class=mature` early in the generation rather than waiting until the
   assessment boundary. Allocate the first wave so up to `Q` peers begin a
   justified direct mature evaluation while at least one peer explores when the
   cohort has multiple peers; use `ordinary` and `scout` for the other classes. The
   scheduler retains at least one exploration slot when such work is queued.
   Do not start a no-checkpoint complete evaluation when its measured/estimated duration
   cannot fit the remaining generation time.
   Independently set `synthesis_trigger.mature_quorum_fraction: 0.25` whenever
   the task distinguishes close-grade evidence through a ratio gate, complete
   stage, protocol-integrity contract, or mature parent lane. This positive
   quorum makes task-defined mature evidence the normal-completion condition.
   `mature_supply_fraction` only prioritizes work and cannot replace this close
   gate. Never set the quorum to `0.0` merely to avoid a deadlock: `0.0` allows
   fixed/adaptive information density, including progress and diagnostic
   findings, to become the normal close condition. Safety-cap and
   cohort-drained outcomes already provide bounded liveness when mature
   evidence remains insufficient.
9. Prefer two operator launch profiles when useful:
   - smoke profile: small cohort and short tiers for validating wiring.
   - research profile: hardware-efficient cohort and complete stages for real Praxist runs.

Write the unchanged observation command, sampling interval, observed pressure,
uncertainty, scheduler profiles, initial/max concurrency, and duration evidence
to `assets/resource_plan.md`; mirror only supported compact values in
`task.yaml`. If current Praxist docs/templates do not expose an exact field name for
a setting, do not invent a silent no-op key.

## Praxist Model And Agent Runtime Defaults

When generating operator instructions, README launch examples, or `assets/resource_plan.md`, recommend this Praxist launch profile when the host has `DEEPSEEK_API_KEY` available, unless the user explicitly supplied another provider, model, runtime, or agent-system preference:

```text
model provider: model_provider:deepseek_alias
model: deepseek-v4-pro
agent runtime: agent_runtime:claude_sdk
```

This corresponds to DeepSeek V4 Pro through the DeepSeek alias provider plus the Claude SDK runtime. It is the recommended initial profile for large Praxist research runs because the long-context DeepSeek route is cost-effective when prompt layout stability is preserved. If `DEEPSEEK_API_KEY` is not configured, do not block initialization solely for that reason; document the missing recommended key and preserve any user-configured provider fallback or explicit launch override.

User-provided configuration always wins. If the user specifies a provider, model, runtime, `PRAXIST_MODEL_PROVIDER_REF`, `PRAXIST_MODEL`, `PRAXIST_AGENT_RUNTIME_REF`, `PRAXIST_AGENT_SYSTEM`, or equivalent CLI flags, preserve that choice and document it in `assets/resource_plan.md` and the generated README instead of overwriting it.

Generate a run-wide `agent.reasoning_effort` policy for peers, DIG, PIs, and
Chair. Use `max` unless the user explicitly requests `auto`, `off`, `low`, or
`high`; preserve an explicit choice without inferring one from the research
domain. `auto` explicitly leaves provider/runtime behavior unchanged. Keep legacy
`premium_mode` only when repairing a task that already uses it. Praxist owns
the provider-specific mapping, so do not add DeepSeek request fields, relay
arguments, or runtime wrappers to the task harness.

Do not store raw API keys in the task directory. Provider keys belong to user-level Praxist config or the shell environment, usually prepared by the runtime-install skill. In task initialization, only document required key names such as `DEEPSEEK_API_KEY`.

When writing launch examples, verify the current CLI help when possible. The intended profile is:

```bash
praxist start \
  --task-path /path/to/task \
  --model-provider model_provider:deepseek_alias \
  --model deepseek-v4-pro \
  --runtime agent_runtime:claude_sdk
```

Do not change the default runtime solely because a particular agent operator interface
is present. When the user explicitly chooses `agent_runtime:codex_sdk`, carry
that ref into generated launch instructions and record the choice in
`assets/resource_plan.md`. Verify the Praxist environment provides
`openai-codex==0.147.0`, `claude-agent-sdk==0.2.136`,
`codex-relay==0.5.5`, and MCP support. For the default
`agent_runtime:claude_sdk`, verify `claude-agent-sdk==0.2.136`. Do not generate
task-owned relay launchers, per-peer relay ports, or runtime transport helpers: Praxist owns
long-lived local app-server clients, direct MCP attachment, and the private
run-scoped relay needed by DeepSeek/OpenRouter. OpenAI uses the direct SDK path.
For native OpenAI, generated instructions may use either `OPENAI_API_KEY` or a
saved ChatGPT login, but authentication remains operator-owned and must never
be copied into the task. State that environment API credentials take normal
priority outside explicit `--codex-native` mode; that mode suppresses API and
inherited provider/model overrides and is valid only for
`model_provider:openai_compatible`.
Invocation through `praxist-takeover-codex` is already an explicit user choice
of `agent_runtime:codex_sdk`, `model_provider:openai_compatible`, and saved
ChatGPT authentication. Preserve that choice without asking for a provider or
API key, do not replace it with the DeepSeek/Claude recommendation, and use
`gpt-5.6-luna` unless the user explicitly selected another account-supported
model. Require `praxist doctor --codex-native --task-path <task>` to report the
selected model catalog entry as ready before resolve or launch. For a newly
created task on this invocation path, use
`dig_lite.contract.min_rejected_alternatives: 2` unless the user chose a
stricter value; keep every other task-justified DIG diversity and integrity
check. Do not create a task-local runtime shadow or model-specific schema
adapter to support this selection.

Praxist owns provider-specific context efficiency. Do not copy context batching,
cache routing, memory-store, prompt compression, or session-interval settings
into `task.yaml`, task prompts, roles, or evaluator code. Codex-native mode and
OpenRouter runs automatically use Praxist's lossless
event-coalescing path; direct DeepSeek runs deliberately retain their established
event timing and cache behavior. This policy preserves canonical artifacts and
only changes when a continuation session opens. Record an operator override such
as `PRAXIST_CONTEXT_EFFICIENCY_MODE=off` only in launch documentation when the
user explicitly requests it. Never summarize away or delete task evidence to
reduce token use.

## Required Research-Loop Defaults

Enable these by default for real research tasks unless the user explicitly asks for a minimal smoke fixture:

- **Initial DIG**: default ON only for absolute generation 0 through
  `dig_lite.enabled: true` and `dig_lite.generation_scope: initial_only`.
  Do not enable DIG for later generations unless the user explicitly requests
  the legacy all-generation behavior.
- **Quality diversity**: default ON independently through
  `quality_diversity.enabled: true`, with both
  `initial_generation_enabled: true` and `later_generations_enabled: true`.
  Generation 0 applies QD to DIG candidate pools. Later generations keep DIG
  off and apply soft QD guidance through the existing PI synthesis path. In a
  Multi-PI topology, the PI memo proposal union is the candidate pool and Chair
  assigns contracts. In a single-PI topology, the PI forms and selects
  proposals in its existing synthesis call. Preserve task-owned diversity
  labels so both paths have useful axes. Each generation switch may be disabled
  without disabling the other. Initial QD still depends on gen0 DIG for its
  candidate pool; later QD is independent of DIG execution.
  Define `evaluation.diversity_dimensions` as task-owned axes before enabling
  QD. PI/Chair contracts use those axes under `planned_dimensions` to state the
  intended design before work starts. Peers must publish the implemented,
  evaluated values under `design_dimensions`; this is realized evidence, not a
  copy of the plan. Never backfill missing realized values from the PI plan.
  Diagnostics must compare planned and realized distributions separately and
  report per-axis HHI, sample size, missingness, and plan-to-execution drift.
  Missing dimension reports are advisory evidence-quality gaps, not a reason to
  block an otherwise valid experiment.
- **Constructive peer mix**: default advisory feedback is explicitly controlled
  by `evaluation.constructive_peer_mix_enabled: true`; the target remains
  `constructive_target_ratio: 0.75`. Set the boolean false when the task should
  not bias the next generation toward a constructive-work floor. This switch is
  independent of `dig_lite.innovation.enforce_forward_slots`.
- **Gems reset**: default OFF for newly initialized user tasks. Use continuous
  evolution first through `gems.enabled: false` or an omitted Gems reset block
  when current templates/docs support omission. Do not guess a reset cadence
  during task initialization. If the user explicitly asks for periodic Gems, or
  a later diagnostic run detects a significant performance ceiling, enable Gems
  reset then and set `gems.reset_interval_generations` from the observed
  plateau-onset generation id or the task owner's explicit choice.
- **Frontier lanes / incubator**: define only the task-owned
  `evaluation.frontier_lanes` justified by the measured protocol and metric
  space. Expensive, staged, diagnostic, or multi-axis tasks should normally
  distinguish strict confirmed results, a lower-admission durable incubator,
  and lower-confidence validation candidates. A cheap single-protocol,
  single-metric task may use a minimal confirmed plus durable-promising
  structure and rely on Praxist's compact validation-candidate view; do not invent
  maturity stages, Pareto axes, or diagnostic lanes merely to fill a template.
  The incubator, when configured, is a core Praxist cross-generation memory
  structure, not a cosmetic leaderboard. Use a clear task-owned name such as
  `incubator`, `<task>_incubator`, or `candidate_library`. It is lower-standard
  relative to confirmed promotion, not low-quality or unfiltered. Its job is to
  retain promising complete evidence so future agents can keep developing it.
  If the incubator stays empty for many generations while complete
  protocol-passed variants exist, later Peers and PI/Chair synthesis may lose
  the best parents and keep exploring unrelated ideas, causing stagnant
  performance. This includes fully clean candidates that are not selected by
  the stricter confirmed lane because of ranking, capacity, or earlier-lane
  deduplication but remain Pareto/new-high on another important axis.

  Treat evaluator lane metadata as a **source-routing contract**, not as the
  evaluator deciding the final durable target. When ordinary complete,
  protocol-passed, non-suspect performance evidence should be considered by
  both strict confirmed selection and lower-admission incubator selection,
  emit one shared task-owned source label (normally `performance`) and include
  that label in both target lanes' `include_lanes`. Do not stamp every
  parent-authorized result as `confirmed` when the incubator cannot accept that source
  label: this makes the configured incubator unreachable and discards strong
  non-primary Pareto parents after the confirmed top-k cap. A different shared
  label is valid when the task defines it consistently; the invariant is
  reachability, not the literal word `performance`.

  Make the generated evaluator own this source label in its canonical result
  summary. Peers and `share_finding` should reference that summary and must not
  relabel the same result as confirmed or incubator based on agent judgment. If
  both `frontier_lane` and `promotion_lane` are emitted, they must agree. Praxist
  owns the final target-lane choice from the configured axes, filters, and
  capacities.

  Add a small task-local lane-routing regression beside the evaluator (reuse an
  existing test module when possible). It must execute representative summary
  construction rather than merely grep prompt text, and prove all of the
  following:

  - every configured `parent_eligible: true` lane has at least one reachable,
    protocol-passed, non-suspect evaluator source label from a mode that the
    user-owned intent permits to supply durable parents;
  - when confirmed and incubator independently consume parent-authorized performance
    evidence, more than `confirmed.k` parent-authorized fixtures remain routable. When
    the task has a scientifically justified distinct incubator axis, include a
    fixture outside confirmed top-k that is non-dominated on that axis and
    prove it can reach the incubator selector. A genuinely single-metric task
    must not invent another metric merely to make this fixture possible;
  - fixtures from modes the intent table marks non-parentable cannot reach a
    parent-eligible lane. Under the recommended default this includes
    preliminary, partial, smoke, protocol-failed, validation-only, late, and
    suspect fixtures; an explicitly parent-authorized reduced mode must instead
    receive a positive fixture with its true effort and coverage;
  - explicit task-owned incubator source labels, when used instead of a shared
    source, carry non-empty `incubator_axis` and
    `incubator_candidate_reason`.

  A newly generated task is not launch-ready when a configured durable parent
  lane is unreachable. Repair evaluator output and lane `include_lanes`, then
  rerun the regression; do not remove the lane merely to make validation pass
  when the task's measured metric space justifies that retention surface. Do
  not paper over a producer bug only by adding `confirmed` to incubator inputs,
  and do not require incubator to be non-empty in every generation: a reachable
  lane may correctly receive nothing when no new Pareto/new-high point exists.

  The mature confirmed lane should remain strict and fact-only relative to the
  user's declared protocol. The durable
  incubator is deliberately lower-standard than confirmed promotion, but it is
  not an unfiltered dump. Under the recommended default it retains
  parent-authorized, protocol-passed, non-suspect candidates that are excellent
  on at least one important distinct
  evaluation axis or are Pareto-front/new-high points relative to the current
  incubator, even if they still need repair, seed escalation, ablation,
  falsification, or extra validation before confirmed promotion. Do not use the
  incubator for any mode the protocol-intent table marks non-parentable.
  Scout-only, partial, validation-only, protocol-failed, unscored, late, or
  suspect evidence normally belongs in validation-candidate or diagnostic
  lanes. If the user explicitly authorizes a reduced mode as parent evidence,
  adapt the lane filters and positive regression fixtures to that decision
  while preserving its real stage and effort/coverage. A stale label alone
  never grants that authority.

  Build incubator axes from genuinely different metric families, not the same
  metric repeated on several data subsets. Examples of distinct families:
  primary task quality, robust/lower-bound quality, safety/risk,
  constraint satisfaction, robustness, compute/cost/latency, resource
  efficiency, calibration/generalization, simplicity, or explanation quality.
  Derive the exact axis set from the task's metric discovery and avoid copying
  another task's metric vocabulary into this project.

  When an incubator is configured, the canonical evaluator summary must publish
  the shared or explicit source lane and all metrics needed by its Pareto axes,
  plus protocol-integrity, complete-scoring, suspect, and promotion-eligibility
  fields. Task prompts should tell Peers to reference this summary instead of
  overriding its lane. For an explicit incubator source, also publish
  `incubator_axis` and `incubator_candidate_reason`; a shared `performance`
  source leaves final confirmed/incubator selection to Praxist.
  Capacity is charged to immutable result evidence, not aliases: one exact
  canonical result artifact (`source_result_path` plus its SHA-256) may occupy
  only one durable slot even when several findings or variant names reference
  it. Different artifacts, including independent replications at different
  paths or with different hashes, remain independent candidates. Add this case
  to the task-local lane regression when the evaluator can emit aliases.
  Praxist exposes validation candidates as a compact companion context for peers,
  DIG, PI, diagnostics, and follow-up planning; tasks should not duplicate them
  into hand-written leaderboards or derived files.
- **Staged evaluation alignment**: when the user/project intent permits staged
  evidence and the target protocol is expensive,
  do not let a cheap preliminary check stand in for complete-protocol
  performance. Generated tasks must distinguish preliminary, aligned, and
  complete semantic stages, while allowing the task to choose literal labels.
- **Research-loop MCP tools**: explicitly declare the standard research-loop
  tool servers under `praxist_plugins.tools`. Praxist has a runtime fallback
  for legacy descriptors, but generated tasks should still be explicit so
  `praxist resolve` and `praxist start` use the same plugin closure.
- **Human-readable run reports**: include `tool_server:run_report` in the
  active `praxist_plugins.tools` list. Praxist also generates derived
  Markdown run reports automatically at major boundaries when enough canonical
  evidence exists: first credible above-baseline frontier signal, every 3
  completed generations, and final run completion. These reports are views
  under `docs/praxist_reports/`; they must never be treated as canonical promotion
  truth. Every metric that can affect ranking, a dimension winner, a baseline
  comparison, or a chart must have an explicit task-owned `maximize`/`minimize`
  direction in the primary metric, anchor metrics, frontier lane axes, optional
  axes, or baseline record. Keep result aliases mapped to that same canonical
  metric. An unknown direction is display-only and must not be guessed as
  maximize.
- **Evaluator execution contract**: generated peer role prompts and task prompts
  should prefer the task evaluator's synchronous public entrypoint. Do not teach
  peers to launch ad-hoc background evaluations and wait on arbitrary files. Use
  `wait_for_file` only when the task harness explicitly documents a supported
  task-owned progress/result file contract that still publishes the standard
  result summaries and finding evidence paths. Runtime-private
  `tasks/<task-id>.output` files are transcripts, not completion sentinels:
  successful commands may leave them empty, so generated prompts must require
  the runtime's structured task notification and exit status instead. For a
  supported long-running background
  evaluation, require the Praxist submission facade rather than raw `&`/`nohup`:
  `PYTHONPATH="$PRAXIST_WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PRAXIST_RUNNER_PYTHON" -m praxist.plugins.workflow_stages.research_loop.backend.protected_pids launch --peer "$PRAXIST_PEER_ID" --tag <stable_semantic_id> --profile <task_profile> --work-class <scout|ordinary|mature> -- <task evaluator command>`.
  In central mode Praxist owns the final process, accelerator environment,
  process-group tracking, retry identity, and release. A repeated ordinary
  submission is idempotent. After correcting a `failed` or `rejected` terminal
  request, retain the scientific tag and add `--retry-terminal`; never add that
  flag to active or completed work and never invent a new scientific identity
  merely to bypass an execution failure.
  Keep `synthesis_trigger.adaptive.drain_grace_minutes` explicit (normally 5):
  it starts only after protected work has drained and bounds agent-only result
  publication/passive waits before `STOP_SIGNAL`; it must never be used as an
  evaluator runtime limit.
- **Literature/database/open-access lookup**: include
  `tool_server:literature_lookup` in the active `praxist_plugins.tools`
  list by default so Peers and PI memo agents can use no-key public scientific
  context when it is useful; Chair agents should inherit source-backed signals
  from PI memos rather than call tools directly. Also add a disabled task-local
  `literature_scout` role to document source policy, but keep
  `panel.optional_roles` disabled unless the task topology implements
  optional-role execution. Treat lookup records, open-access text, public
  database entries, and PDF provenance as contextual research signals, not
  evaluator facts. The tool is passive until an agent calls it; listing it in
  the tool set does not itself make network calls. Task prompts and role skills
  should explicitly ask Peers/PI to perform a bounded lookup before finalizing a
  non-trivial mechanism, prior-art claim, domain metric, or research direction
  that is not already grounded in local task documents.
  Use a current-environment-only resource policy: if a source mentions
  unavailable datasets, checkpoints, simulators, packages, licenses, APIs, or
  runtime environments, do not tell agents to download, install, or provision
  them during Praxist runs. Tell agents to adapt the idea to the task's existing
  local assets, evaluator, dependencies, hardware, and runtime, and record the
  missing resource only as a task-local note.
  The current public tool surface includes `literature_search`,
  `literature_resolve`, `literature_open_access_text`,
  `scientific_database_search`, and `literature_source_guide`.
- **Optional reviewer**: document that `workflow_stage:reviewer_stub` can be
  explicitly run in local/artifact review mode to check artifact hashes,
  trajectory references, run summaries, and whether literature/database context
  was incorrectly marked as runtime truth. Do not wire it into the main
  generation loop during task initialization; it is an audit-only helper. The
  local reviewer refuses to append after `run.finalized`, so post-hoc analysis of
  completed runs should use diagnostic reports or copied run directories.
- **Research memory / PI / Chair**: default ON when supported by the current Praxist templates and docs. Keep peer contracts per-peer and diversity-aware; do not broadcast every peer's detailed plan to every peer.

When writing exact config for a machine learning project, inspect
`TASK_TEMPLATE_ROOT/machine_learning_template/task.yaml` first (the source form
is `templates/tasks/machine_learning_template/task.yaml`). Use
`TASK_TEMPLATE_ROOT/template/task.yaml` only as the minimal scaffold/smoke
shape, and use richer domain reference templates such as
`TASK_TEMPLATE_ROOT/sam_optimizer/task.yaml`
only for task-specific inspiration. Also inspect current Praxist docs/CLI help. Use
supported field names. The current standard task-local fields are:

```yaml
praxist_plugins:
  tools:
    - tool_server:evaluation_tools
    - tool_server:frontier_tools
    - tool_server:finding_graph_query
    - tool_server:memory_tools
    - tool_server:prior_work_tools
    - tool_server:run_report
    - tool_server:literature_lookup

evaluation:
  maturity_policy:
    min_effort_ratio: 0.75
    min_coverage_ratio: 0.80
    require_ratio_gate: true
    # Include labels only when this task actually uses staged evaluation.
    complete_stage_labels: [complete]
    # Omit this field when the task has no early stages.
    preliminary_stage_labels: [preliminary, aligned]
  constructive_peer_mix_enabled: true
  constructive_target_ratio: 0.75
  launch_guard:
    enabled: true
    # Calibrate from observed p90 runtimes. Close-grade is the task-authorized
    # evaluator that can satisfy normal close.
    estimated_heavy_eval_minutes: <estimate_from_resource_plan>
    estimated_close_grade_eval_minutes: <close_grade_p90_from_resource_plan>
    safety_factor: 1.25
  frontier_lanes:
    - name: confirmed
      description: "Fully scored, promotable candidates on the task primary metric."
      k: 3
      cumulative_cap: 10
      axes:
        - {name: <primary_metric>, direction: maximize}
      # Default only: the evaluator emits `performance` for ordinary clean
      # parent-authorized results. Adapt this to the user-owned protocol-intent table,
      # allowing both parent lanes to select independently.
      include_lanes: [confirmed, performance]
      parent_eligible: true
      require_metrics: [<primary_metric>]
      require_truthy_metrics: [scored_complete]
      require_falsey_metrics: [protocol_integrity_failed, suspect_protocol, suspect_leakage]
    - name: incubator
      description: "Lower-admission durable Pareto/new-high retention library for task-authorized, protocol-passed, non-suspect candidates that need follow-up before clean confirmation."
      k: 8
      cumulative_cap: 48
      admit_new_high: true
      axes:
        - {name: <primary_metric>, direction: maximize}
        # Add 2-6 truly distinct task axes here when the evaluator always
        # emits them and they should define Pareto/new-high retention, for
        # example robust lower bound, safety/risk, robustness,
        # compute/cost, constraint success, calibration/generalization, or
        # explanation/simplicity.
      optional_axes:
        # Optional axes are secondary sort/display signals only. Do not place
        # a metric here if missing it would change whether a candidate should
        # be retained on the long-term Pareto/new-high surface.
        - {name: <secondary_tiebreak_metric>, direction: maximize}
        - {name: <diagnostic_display_metric>, direction: minimize}
      include_lanes: [incubator, performance]
      require_metrics: [<primary_metric>]
      require_truthy_metrics: [scored_complete]
      # Remove a stage marker from this default exclusion list only when the
      # user-owned protocol explicitly authorizes that mode as durable evidence.
      require_falsey_metrics: [protocol_integrity_failed, is_smoke_eval, partial, scout_only, validation_only, validation_only_result, late_after_generation_boundary, suspect_protocol, suspect_leakage]
      parent_eligible: true
      allow_non_promotable: true
      allow_missing_tier: true
      allow_risk_violating: true
    - name: task_candidate
      description: "Promising preliminary, aligned, partial, validation, or repair evidence retained for follow-up."
      k: 5
      cumulative_cap: 20
      axes:
        - {name: <primary_metric>, direction: maximize}
      include_lanes: [task_candidate, candidate]
      parent_eligible: false
      allow_lower_tier: true
      allow_non_promotable: true
      allow_missing_tier: true
    - name: diagnostic
      description: "Controls, falsifiers, negative evidence, and process diagnostics."
      k: 2
      cumulative_cap: 10
      axes:
        - {name: <primary_metric>, direction: maximize}
      include_lanes: [diagnostic, control, process, reference, negative_control]
      parent_eligible: false
      allow_lower_tier: true
      allow_non_promotable: true
      allow_missing_tier: true

dig_lite:
  enabled: true
  generation_scope: initial_only
  candidate_count: 8
  min_mechanism_families: 4
  min_intervention_surfaces: 3
  max_attempts: 10
  max_total_runtime_minutes: 40
  fallback_to_direct_on_failure: true
  diversity:
    cell_fields: [mechanism_family, intervention_surface, intent]
    selection: best_within_lane
    reject_near_duplicate: true
    duplicate_threshold: 0.82
  innovation:
    enabled: true
    enforce_forward_slots: true
    max_diagnostic_fraction: 0.20
    max_diagnostic_peers: 2

quality_diversity:
  enabled: true
  initial_generation_enabled: true
  later_generations_enabled: true
  max_same_diversity_cell_peers: 1

synthesis_trigger:
  enabled: true
  min_findings: <derive_from_cohort_size_and_expected_finding_rate>
  min_interval_minutes: <derive_from_implementation_queue_and_first_close_grade_result_time>
  max_interval_minutes: <include_implementation_queue_close_grade_evaluation_publication_and_margin>
  min_contributing_peers: <min(3, cohort_size)>
  mature_quorum_fraction: 0.25  # normal close requires task-defined mature evidence

gems:
  enabled: false  # continuous-evolution default; enable reset after diagnosis or user request
  selection_policy: mature_evidence_top_k
  min_mature_eval_units: <parent_authorized_protocol_required_evaluation_units>
  evidence_stage_min_units:
    <parent_authorized_stage_label>: <parent_authorized_protocol_required_evaluation_units>
```

If the unchanged baseline observation and generation bound do not make `Q`
mature results physically plausible, do not silently weaken effort/coverage or
set `mature_quorum_fraction: 0.0` as a liveness workaround. Recalibrate the
generation bound, peer/concurrency plan, or the task-owned accepted evidence
protocol; changing scientific acceptance semantics requires user confirmation.
If mature evidence still cannot be produced, retain the positive gate so
safety-cap or cohort-drained close records the insufficiency explicitly. Use
`0.0` only when the task intentionally has no separate maturity distinction
and the user explicitly confirms that ordinary information-density findings
are sufficient to close a generation; record that decision in the resource
plan and initialization report.

If the user explicitly enables periodic Gems reset, add the task-owned reset
policy then:

```yaml
gems:
  enabled: true
  selection_policy: mature_evidence_top_k
  min_mature_eval_units: <parent_authorized_protocol_required_evaluation_units>
  evidence_stage_min_units:
    <preliminary_stage_label>: <preliminary_protocol_required_evaluation_units>
    <parent_authorized_stage_label>: <parent_authorized_protocol_required_evaluation_units>
  reset_interval_generations: <plateau_generation_id_or_user_choice>
  max_resets: 3
  max_gems_per_reset: 4
  max_gems_total: 4
  max_gems_per_family: 2
  prompt_max_gems: 4
  archive_ordinary_findings: true
```

Set `gems.min_mature_eval_units` from the number of task-owned evaluation units
required by the protocol the user authorizes for Gems and durable parent use.
That is normally the complete protocol, but it may be an explicitly selected
reduced protocol. When the task has staged evidence, map its own labels to
cumulative unit requirements with `evidence_stage_min_units`.
Always use units in Praxist configuration, even when a task-local evaluator uses a
different term. Do not emit compatibility-only historical maturity keys in a
newly generated task, and never copy another task's labels or unit counts.

If a task omits `evaluation.frontier_lanes`, Praxist still has a legacy primary
metric frontier, but promising validation candidates may not appear as a
separate incubator-style lane in `frontier/frontier_manifest.json`. For any
expensive task with preliminary, aligned, partial, or diagnostic validation, or
multiple meaningful metric axes, configure both a durable incubator lane and a
lower-confidence candidate lane so future PI/Chair synthesis can keep strong
parents and weaker signals in view. Size the incubator to the metric space:
use at least `k = max(8, 2 * number_of_distinct_metric_families)` for real
multi-axis tasks and a `cumulative_cap` about 4-6x `k`; use larger values such
as `k: 16` and `cumulative_cap: 96` for broad multi-axis research.

The incubator is intentionally a lower-admission long-term variant library, not
a stricter confirmed-winner lane. It should preserve protocol-authorized,
protocol-passed, non-suspect Pareto/new-high candidates across every genuinely
distinct metric family. Configure `admit_new_high: true` so a candidate must improve the
incubator's current Pareto surface rather than merely repeat an older point.
Do not make incubator stricter than confirmed promotion; if it stays empty over
many generations, later peers and PI/Chair synthesis can lose promising parents
and the run can stagnate.

When the user-approved maturity contract uses effort and coverage ratios,
newly generated task evaluators must emit `effort_ratio` and `coverage_ratio`
in each canonical machine-readable result summary.
`effort_ratio` compares actual effort to the task-defined reference effort;
`coverage_ratio` compares completed required evaluation units to total required
units. Praxist uses the same extractor to project these facts into the
auto-materialized finding; do not make task code duplicate source-owned facts.
A standalone manually authored result finding without a canonical summary
reference must carry the ratios itself. These fields let Praxist start generation
assessment on mature evidence without making stage labels into hard-coded task
logic. Set `require_ratio_gate: true` only for a contract that uses those ratios.
If the user explicitly chooses label/flag-based or information-density semantics,
preserve that choice and document the limitation instead of manufacturing ratio
requirements. With a required ratio gate, missing ratios remain unknown even
when a complete-looking label is present.

When ratio gating is enabled, before declaring the task launch-ready execute the shortest valid scored path
that uses the evaluator's real canonical summary writer. Then validate that
actual file with `praxist resolve <task_path> --result-summary <summary_path>`.
For an expensive evaluator, a deterministic contract probe may stop before a
performance claim, but it must call the same serialization path as real scored
evaluation. A hand-written lookalike JSON file is not proof. The preflight must
pass with finite ratios when `require_ratio_gate: true`; stage labels are
optional task audit vocabulary and cannot substitute for missing ratios.
The same preflight must reject completion metadata only when the task-owned
policy actually resolves it to contradictory decisions. A configured terminal
cap may be mature; an early cap is incomplete. Validate the achieved protocol
and ratios instead of rejecting status vocabulary by substring.

### Evaluator Fan-Out Preflight

Before any aligned, complete, or otherwise expensive multi-unit fan-out, prove
the actual evaluator path in increasing-cost order:

1. Run the task-appropriate build, load, or startup check in the exact runtime
   that fan-out will use. For interpreted code this may be compile/import; for
   a binary, library call, simulator, container, notebook, or service, exercise
   its corresponding real loading or startup boundary. Source-text inspection
   alone is not execution proof.
2. Validate the evaluator's actual public invocation contract before reserving
   expensive resources. Exercise its real parser/help path when it has a CLI;
   otherwise validate the function, RPC, simulator, container, notebook, or
   service interface that the task actually calls.
3. Run a **one-unit canary** through the same public evaluator, scheduler path
   when Praxist owns launch, runtime environment, and canonical summary writer
   that fan-out will use. One unit means the smallest scientifically valid
   task-defined case; it does not imply a universal seed, epoch, iteration,
   split, or device model. For a protocol without discrete units, use its
   shortest valid invocation.
4. Validate the produced summary and its summary-to-finding projection. Only
   then fan out. A repaired command or implementation must pass a new canary;
   an earlier canary does not authorize changed code.

The canary verifies wiring and publication, not performance or maturity. Keep
valid low scores and negative scientific outcomes; stop fan-out only for an
execution, schema, publication, or declared protocol-integrity failure.

If the task explicitly claims that evidence is produced by an external or
otherwise independently trusted evaluator, also prove the task's declared
trust boundary: a peer must not be able to replace the authoritative result,
and the verifier must reject a modified or unattested result. The attestation
mechanism remains task-owned. Do not impose this mode on ordinary peer-authored
evaluators, and do not hard-code a particular signature scheme, filesystem
owner, dataset split, or service into Praxist task initialization.

Task result summaries may be nested under `results/**/` and should use one of
the compact names `summary.json`, `evaluation_summary.json`,
`eval_summary.json`, `tiered_eval_summary.json`, or
`custom_*_tiered_eval_summary.json`. `result_summary.json` remains a supported
compatibility name. Every summary must carry a stable top-level `variant_id`,
or an explicit child-result ID when one evaluator emits multiple candidates;
repeated stages for the same candidate must reuse that identity. Put lane,
maturity, effort/coverage ratio, protocol, and diagnostic metadata in structured
top-level, `metrics`, `current_aggregate`,
or `extra` fields so the Praxist
materializer can transfer fields such as `frontier_lane`, `promotion_lane`,
`evidence_stage`, `evidence_valence`, `diagnostic_role`, `failure_mode`,
`parent_candidate`, and `next_step_intent` into canonical findings. Do not hide
these facts only in prose, filenames, or side files.

When central scheduling owns evaluator launch, every submission must pass a
result-specific directory through a recognized explicit output option such as
`--out-dir`, `--output-dir`, or `--result-dir`; do not rely on the process `cwd`
or claim the run-wide `results/` root as one peer's output. The canary must
confirm that summary-to-finding materialization preserves the scheduler's
canonical generation and peer attribution. Where available, also write the
scheduler-provided `generation_id` and `peer_id` into the summary as structured
cross-check fields rather than deriving them from free-form names.

Use `evaluation.launch_guard.estimated_heavy_eval_minutes` for the most
expensive ordinary evaluator and `estimated_close_grade_eval_minutes` for the
task-authorized protocol that can satisfy normal close. Record both observed
p90 values in `assets/resource_plan.md` and task prompts. With `enabled: true`,
`CLOSING_SIGNAL` freezes every new evaluation or training launch in the
standard peer runtime, while preserving already-started work for natural drain.
Generated prompts must say that after close a peer may inspect outputs, publish
results, and update notebook/memory, but must not run another evaluator,
script, shell launcher, or background process. Background evaluations still
need the standard `protected_pids launch` facade so Praxist can queue and track
their process group during final synchronization. Its semantic tag must not
change for retries, output paths, timestamps, or harmless flags.

For a task that requires mature/complete evidence before normal close, measure
the unchanged complete evaluator enough times to estimate its p90 wall time and
enforce this launch-readiness inequality:

```text
estimated_close_grade_eval_minutes * safety_factor
  < effective_generation_close_horizon_minutes - drain_margin_minutes
```

Use a drain margin of at least 30 minutes unless measured publication and
shutdown latency justify more. The close-grade estimate may describe an
explicitly user-authorized reduced protocol while the heavy estimate preserves
a longer optional protocol. The effective horizon is the earliest enabled
generation/peer hard bound, including an enabled adaptive synthesis ceiling. If
the inequality fails, raise the relevant generation and synthesis bounds or ask
the user to authorize a different protocol; never relabel an incomplete run as
complete. An explicitly user-authorized late-signal or reduced protocol remains
valid when the task's launch, maturity, lane, and close policies consistently
encode that choice.

Declare the single public evaluator in
`task_entrypoints.evaluation.command`. Praxist normalizes that structured field into
the legacy-compatible `toolchain.eval_entrypoint` when needed, so do not store
the same path twice unless a legacy task consumer explicitly requires it. The
default Claude SDK runtime uses the resulting entrypoint to register a direct
evaluator invocation with the protected-PID manifest, keeping
`active_evals` and generation drain truthful. Prompts must still require the
explicit wrapper for compound or task-owned background shell commands.
Keep task-owned evaluator, trainer, config, and harness paths relative to the
task root. Absolute paths are appropriate only for external assets such as an
existing dataset, simulator, or environment that cannot live inside the task;
verify every such path immediately before launch and record why it is external.

Before marking the task launch-ready, run the public evaluator from both the
task root and a temporary run-like subdirectory through the declared task
interpreter. Verify that the child imports task dependencies without runner
site-packages, resolves the same evaluator/config/data paths in both cases, and
writes its canonical summary under the requested output directory. Do not
inject the Praxist checkout through task `PYTHONPATH`; if a task genuinely owns
a custom `PYTHONPATH`, declare it explicitly in `runtime_environment.env` and
test that exact environment.

## Staged Evaluation Alignment

When the target task evaluation is too expensive for every early attempt, normally
build a three-level task-owned evaluation contract. The user may instead
explicitly choose a simpler or intentionally incomplete protocol. For expensive
repeated-case, seeded, episodic, simulation, benchmark-suite, or multi-dataset tasks,
use the following default unless the protocol-intent table says otherwise:

- **Preliminary check**: the cheapest executable sanity check. By default it may use a
  tiny fixture, one split, one episode, very low effort, or failure-only checks.
  Its purpose is wiring validation, impossible-idea filtering, failure
  diagnosis, and qualitative validation candidates. It must not be used for
  ranking variants, selecting clean parents, Gems reset, mature frontier
  promotion, performance-ceiling detection, or "best so far" claims unless the
  user-owned protocol intent explicitly authorizes those uses and reports the
  reduced protocol transparently.
- **Aligned evaluation**: the only early protocol that can rank or prioritize
  variants before complete evaluation. It must use the same evaluator code
  path, primary metric direction, aggregation semantics, invalid-result rules,
  leakage checks, and protocol-integrity checks as the complete protocol.
  Preserve the complete protocol's evaluation-unit coverage by default: use
  the same folds, datasets, task cases, scenes, seeds, episodes, or
  operating regimes whenever possible. Save compute primarily by reducing
  optimization budget, such as epochs, gradient steps, rollout horizon, simulator steps,
  inner-loop iterations, checkpoint sweeps, or repeated restarts. This keeps
  the measured loss/metric landscape close to complete evaluation while accepting a
  less-converged parameter location.
  Reducing evaluation units is a last resort, not the default. If full coverage
  is truly impossible, the aligned evaluation must still use a near-complete, fixed,
  stratified coverage set that preserves all major regimes and hard cases, and
  must record the coverage ratio and omitted-unit rationale in task-owned
  assets. If the coverage is materially small or convenience-sampled, label the
  result preliminary or partial instead of aligned. For example, if
  the complete protocol spans many evaluation units, a small convenience
  subset is
  normally preliminary/partial, not aligned; a better aligned evaluation would
  keep near-complete unit coverage and reduce epochs, iterations, rollouts, or
  training steps. Do not mix failed-unit smoke checks, ad-hoc partial runs, or
  varying epoch/step budgets into this ranking stage.
- **Complete evaluation**: the default highest-evidence task protocol. This
  stage, or any reduced protocol explicitly declared equivalent or sufficient
  by the user-owned intent, can create clean implementation parents, confirmed
  frontier entries, or Gems parents.

Tasks with genuinely staged protocols choose their own literal labels.
`preliminary`, `aligned`, and `complete` are recommended neutral defaults; an
existing staged task may retain its established names by listing them under
`preliminary_stage_labels` and `complete_stage_labels`. Unstaged protocols may
omit stage labels. Labels never carry global effort, coverage, ordering, or
parent-eligibility semantics by themselves.

Generated evaluator summaries and findings must report enough fields for later
agents to tell which protocol produced a result:

- for staged protocols, `extra.evidence_stage`: the task-local preliminary,
  aligned, or complete label;
- `extra.protocol_name` or `extra.eval_mode`;
- `metrics.evaluation_units` or `extra.completed_required_eval_units`, plus
  task-local counts such as `n_seeds` or `n_episodes` when relevant;
- fixed budget fields such as epochs, environment steps, wall-clock cap, or
  simulation horizon;
- coverage fields such as `extra.coverage_ratio`,
  `extra.complete_protocol_evaluation_units`, and
  `extra.aligned_evaluation_units` when the task has discrete evaluation
  units;
- exact maturity fields: `effort_ratio` and `coverage_ratio` in a supported
  scalar fact container for every scored canonical evaluator summary; Praxist
  projects them into auto-materialized findings, while standalone findings
  without a canonical summary reference must carry them directly;
- `metrics.scored_complete` or a task-owned authority marker only for evidence
  that satisfies the declared mature protocol. If that protocol is explicitly
  reduced, the result must still expose its reduced stage, effort, and coverage;
- protocol-integrity flags or invalid-result reasons when a run is not clean.
- structured negative-evidence metadata on every finding:
  `extra.is_negative` must be explicit, `extra.evidence_valence` should be one
  of `positive`, `neutral`, `negative`, or `mixed`, and negative or mixed
  results should include compact `extra.failure_mode` plus
  `extra.disconfirming_claim_ids` when a hypothesis/claim is weakened.
- when non-code launch settings can change the treatment, one secret-free
  top-level `effective_config` object containing all result-affecting CLI
  arguments, environment overrides, protocol choices, and task-local config
  values, plus explicit `effective_config_complete`. This object must contain
  values after the evaluator has applied defaults, parsing, aliases, and type
  conversion. Do not hash a raw environment snapshot or distinguish an omitted
  setting from an explicit setting equal to the same resolved default. Exclude
  unrelated runtime environment. Keep the full object only in the canonical
  evaluator summary; downstream Praxist artifacts use its digest and
  `source_result_path`. For derived work, make the task evaluator or its existing
  launch helper read the selected parent summary, resolve the child through the
  same schema, and report a secret-free key-level mismatch before expensive
  execution. Do not make each peer manually replay a parent process environment
  and do not blindly export every parent environment entry.
- for a task-declared exact replication,
  `replication_of_effective_config_sha256` copied from the selected parent's
  canonical result metadata. The evaluator result may use the exact-replication
  label only when Praxist reports `replication_effective_config_status:
  matched`; otherwise label it as a control, code-only rerun, or incomplete
  replication. Never turn this provenance check into an ordinary promotion,
  maturity, resource, or closing gate.

Unless the user-owned protocol-intent table assigns different permissions,
task prompts, role prompts, and resource plans should use these defaults:

- Preliminary results are useful validation signals, not ranking evidence.
- Only aligned results may be used to prioritize which variants deserve
  complete evaluation.
- The aligned protocol should be close to complete in data/evaluation coverage and
  cheaper mainly because training or optimization budget is reduced.
- Negative evidence is still useful research memory. Failed, dominated,
  constraint-violating, non-generalizing, null-ablation, or falsifying results
  should be published with the structured fields above instead of being
  omitted or relabeled as neutral.
- The protocol explicitly authorized for durable parents remains the final
  parent-promotion gate; by default this is complete mature evidence.
- Reports that compare early to complete performance must compute calibration
  separately for preliminary and aligned evidence. Weak preliminary
  correlation is not a failure; weak aligned-to-complete correlation means the
  task protocol should be redesigned before trusting early rankings.

## Artifact Ownership Instructions For Generated Tasks

When writing the generated task README, task prompt, role skills, and resource
plan, include this rule in task-specific language:

- Peers and task harnesses publish task evidence through evaluator result
  summaries and `share_finding`.
- Praxist owns frontier/incubator state, Gems state, prompt-layout artifacts, PI
  evidence packs, research memory, diagnostic tables, and generation boundary
  markers.
- Current facts come from result/finding evidence,
  `frontier/frontier_manifest.json`, committed `gems/gems_state.json`, and
  `gen_N/generation_boundary.json`.
- When configured, the durable incubator is part of frontier state. It should
  contain task-authorized, protocol-passed, non-suspect Pareto/new-high
  candidates that are not yet clean enough for confirmed promotion. If it is
  empty despite such evidence, treat the task lane policy or peer metadata as a
  high-priority task-harness problem.
- Validation candidates are non-frontier signals. They may include partial,
  lower-stage, failed-but-informative, preliminary, aligned, repair, ablation,
  or diagnostic evidence,
  plus late-after-boundary result summaries that Praxist retained after a
  generation closed. Generated tasks should route them through structured
  findings/result summaries with task-owned lanes, not through new side files.
  Later agents may use them for validation, repair, falsification, comparison,
  ablation, or full scoring, but not as clean implementation parents unless
  canonical evidence later promotes or revalidates them.
- Leaderboards, PI evidence packs, PI agendas, rendered prompts, prompt-layout
  manifests, diagnostics, and behavior reports are derived views or audit
  snapshots. They are useful context and replay evidence, but generated peer or
  PI instructions must not tell agents to hand-write them or use them to
  override evaluator evidence.
- Partial `.tmp`, `.candidate`, `.rejected`, or incomplete final-generation
  files should be handled by control/resume tooling, not by task peers.

This is not a new task entity. It is guidance that keeps generated tasks aligned
with Praxist's current artifact semantics and reduces cross-generation information
drift.

## Metric Discovery

Derive metrics from the project first: evaluator code, papers/docs, result tables, benchmark scripts, and logs. If metrics are absent or ambiguous, use no-key public literature/database/open-access lookup when available, otherwise agent-host web search, for the domain's standard benchmark metrics. Cite sources in `assets/literature/research_directions.md` and label them as literature/context signals rather than measured task facts. If metric standards depend on external datasets, services, packages, or licensed assets that are not present locally, do not make task execution depend on acquiring them; document them as external requirements and design the Praxist evaluator around the currently available local resources.

Also research the domain's **ranking convention**, not just metric names. For
each task domain, determine whether the field normally ranks by a single mean
score, a mean-plus-risk tradeoff, confidence intervals, lower confidence
bounds, statistical significance, Pareto efficiency, safety/regret constraints,
or multi-objective dominance. Use project docs first; if absent, search
relevant papers, benchmark leaderboards, challenge rules, and methodology
guides. If the domain commonly cares about variance, seed sensitivity,
tail risk, simulator stochasticity, confidence intervals, or lower-bound
robustness, encode that in `evaluation.frontier_lanes` through task-owned axes
or explicit robust metrics such as `primary_metric_lower_confidence_bound`,
`primary_metric_std`, `seed_robustness_std`, `risk_metric`, or
`constraint_violation_rate`. Do not leave the mature lane as raw-primary-only
when the domain standard says robustness matters.

Record:

- primary metric and direction;
- an explicit direction for every metric used as a comparative anchor, declared
  through `anchor_metrics` or lane `axes`/`optional_axes`; never assume the
  primary metric's direction applies to auxiliary metrics;
- primary ranking rule: raw mean, confidence/lower-bound, Pareto lane, or
  task-specific robust composite;
- incubator axis families: the distinct metric dimensions that should define
  durable Pareto/new-high retention before clean confirmation;
- auxiliary metrics;
- constraints and invalid-result conditions;
- aggregation over seeds, folds, maps, scenes, episodes, datasets, or other
  task-owned evaluation units;
- whether variance, confidence intervals, or lower confidence bounds should
  affect frontier ordering;
- preliminary sanity protocol, if any, and its user-approved permissions;
- aligned protocol, if complete evaluation is expensive, and its permissions;
- for aligned evaluation, the expected evaluation-unit coverage ratio and
  reduced training/optimization budget that makes it cheaper than complete;
- the protocol or protocols the user authorizes as mature;
- how future diagnostics should compare each early stage to the mature protocol
  without mixing cheap preliminary checks into ranking statistics.
- how future diagnostics should detect an empty or stale incubator despite
  parent-authorized, protocol-passed candidates.

## Build The Task Directory

For machine learning projects, create a task directory shaped like
`TASK_TEMPLATE_ROOT/machine_learning_template`, adapted to the project. For
non-ML projects, use only the file/directory layout of
`TASK_TEMPLATE_ROOT/template` as the minimal shape. Its checked-in policy
demonstrates optional Praxist features; do not copy its stage labels, lane set,
metrics, or resource values unless the target task independently justifies
them:

```text
task.yaml
README.md
description.md
prompt_base.jinja2
prompt_generation.jinja2
prompt_task.jinja2
.gitignore
.praxist/plugins/panel_topologies/<task_topology>/plugin.yaml
roles/
audit_rules/
evaluations/<primary_eval>/run.py
assets/harness/
assets/baselines/
assets/dataset_metadata/
assets/literature/
assets/project_scan/
assets/task_context/
assets/resource_plan.md
assets/regression_fixtures/
```

If `task.yaml` references a task-local panel topology such as
`panel_topology:<task_topology>`, the generated task directory must include the
matching plugin file under
`.praxist/plugins/panel_topologies/<task_topology>/plugin.yaml`. Do not
leave a task-local topology ref pointing at a plugin that was not copied or
generated. If you intentionally do not want a task-local topology, switch
`praxist_plugins.panel.topology` to a known core topology and document that
choice in the README/resource plan.

When creating a short smoke profile with small `per_generation_hours`, include a
compatible `synthesis_trigger` block rather than relying on core defaults. The
smoke profile should still allow resolve-only and short validation runs to pass
Praxist task-spec checks. Let `praxist ... --resolve-only` or the current task-spec
validator catch profile-specific bounds instead of encoding those bounds in the
skill. For real runs, size `synthesis_trigger.min_interval_minutes`,
`max_interval_minutes`, and `min_contributing_peers` from the resource plan.
Treat `min_interval_minutes` as the earliest assessment point, not as proof that
close-grade evidence exists. Include implementation, queue delay, complete
evaluation, result publication, and drain margin in `max_interval_minutes`;
keep the generation safety bound longer than that end-to-end path. For
complete-evidence close, verify the calibrated p90 inequality above after all
task parameters are finalized; a successful short smoke run does not prove the
complete evaluator can finish before close.

Keep the baseline code system compact. Prefer one public evaluator and a small number of harness files over copying a large repository. Copy or wrap only the minimum code needed for:

- training or fitting;
- inference or rollout;
- model/control/strategy definitions;
- simulator or data interface;
- metric computation;
- data/simulator path resolution;
- small smoke fixtures.

Do not copy raw datasets, large checkpoints, large logs, or simulator installs. Write metadata/resolver files that point to existing local paths or required environment variables.

## Baseline Requirements

Baseline code must run locally in the detected environment. The public Praxist-facing command should be:

```text
evaluations/<primary_eval>/run.py
```

This command should accept at least:

- variant path or variant directory;
- output directory;
- optional data/simulator root;
- optional preliminary/complete mode;
- explicit staged protocol modes when evaluation is expensive:
  preliminary, aligned, and complete mature evaluation, using task-owned
  literal labels. The aligned mode should call the same evaluator path and metric
  aggregation semantics as complete evaluation, preserve near-complete data/evaluation
  coverage, and reduce training/optimization budget first. If it cannot preserve
  near-complete coverage, mark the result preliminary or partial rather than aligned.
- explicit expected-budget fields for clean complete evidence, such as epochs,
  gradient steps, environment steps, rollout horizon, simulator episodes,
  restart count, seed count, fold count, evaluation-unit count, or wall-clock
  cap.
- a machine-checkable protocol-integrity result. If the intended complete protocol
  is 3 epochs and a run only finishes 2 epochs, the evaluator must set a failed
  integrity flag, explain the incomplete budget, and avoid marking the result as
  clean promotion evidence.

If a proposed highest-evidence protocol uses materially less effort than the
project's established baseline, published task convention, or measured
convergence evidence and the user has not already made an explicit choice,
pause before finalizing the task. Show the task-defined reference effort and
proposed ratio, explain the tradeoff, and ask the user to confirm one of:

1. keep this budget as the complete protocol because the task is small or already
   justified by project evidence;
2. increase the complete budget to the user's supplied value;
3. downgrade this protocol to aligned evidence and define a higher-effort
   complete protocol.

Do not impose a universal epoch, step, rollout, or wall-clock threshold. A small
task may legitimately converge quickly, while another domain may need a much
larger budget. The gate is task evidence and declared effort/coverage ratios,
not a fixed global count. An explicit user-selected reduced protocol remains a
valid task contract; label it accurately and obey it.

Baseline performance records go under `assets/baselines/`:

- `results.jsonl` for machine-readable baseline metrics when known;
- `curated_baseline_summary.md` for human-readable baseline evidence;
- `baseline_performance_status.md` stating whether values came from existing project artifacts or still require measurement.

The asset files are provenance records, not an implicit runtime baseline. After
verifying measured values, declare the corresponding metric name, value, and
direction under `task.yaml:baselines`. Run `praxist resolve <task_path>` and
repair any warning that measured-looking baseline assets exist while the
`baselines` list is empty. Never silence that warning by copying unverified
numbers into the task contract.

If reliable baseline performance already exists in project artifacts, copy only
the compact measured facts and record the source path, timestamp, command, and
metric definitions.

If baseline performance is missing, first decide whether the current machine has
all conditions needed to measure it now:

- baseline code or a faithful baseline command exists;
- the public evaluator or benchmark command exists;
- required data, simulator, checkpoints, licenses, and environment variables are
  present;
- the detected runtime environment can execute the evaluator;
- the expected run time is acceptable or can be reduced by a documented smoke or
  tiered protocol.

When these conditions are not met, do not invent numbers. Write the
missing-baseline status clearly and report the missing path, dependency,
dataset, simulator, or command.

When all conditions are met but no reliable baseline performance record exists,
pause and ask the user to choose one of two explicit paths:

1. **Zero placeholder**: write every declared baseline metric as `0.0` in
   `assets/baselines/results.jsonl`, and mark
   `baseline_performance_status.md` as `user_selected_zero_placeholder`. State
   clearly that these zeros are placeholders, not measured evidence, and should
   be replaced before interpreting performance deltas.
2. **Measure baseline now**: create
   `experiments/baseline_bench_<timestamp>/` inside the output task directory
   and write a small baseline bench runner there. The runner should call the
   task's public evaluator or documented baseline command, use the same metrics
   and aggregation as `task.yaml`, and save raw logs plus a compact
   `summary.json`. Before launching, inspect hardware and decide whether
   parallel execution is safe. Apply the task's observed scheduling rules:
   host compute is controlled by total experiment concurrency, while any
   managed accelerator uses backend-supported memory/utilization profiles.
   Keep the bottleneck resource near full utilization without oversubscribing
   memory, accelerator capacity, licenses, or
   simulator instances. When measurement finishes, write the
   measured rows to `assets/baselines/results.jsonl` and summarize provenance in
   `curated_baseline_summary.md` and `baseline_performance_status.md`.

Do not start baseline measurement without the user's selection. If the user
selects measurement and the run is long, launch it only after reporting the
estimated duration, concurrency, output directory, and exact redacted command.

## Task YAML Configuration

`task.yaml` must include supported fields for:

- primary metric, auxiliary metrics, direction, seeds, and aggregation;
- runtime outputs and runtime environment;
- provider-neutral `agent.reasoning_effort` (`max` by default unless the user
  explicitly selects another supported value);
- task-owned evaluator entrypoint;
- compute budget, central scheduler profiles, and host-wide experiment concurrency;
- generation policy: `max_generations`, `cohort_size`, and `per_generation_hours`;
- staged evaluation or equivalent preliminary/complete protocol when the task
  has expensive evaluations;
- staged evaluation alignment for expensive tasks: cheap preliminary checks
  separated from aligned ranking checks and complete mature evaluation.
  Aligned evidence should be near-complete in
  data/evaluation coverage and cheaper mainly through reduced training or
  optimization budget;
- task-local assets and data/simulator metadata.

For real runs, choose defaults from the resource plan rather than copying template values. Do not leave `cohort_size: 1`, `max_generations: 1`, or placeholder compute budgets unless the task is intentionally a smoke-only fixture.

Do not put Praxist provider secrets in `task.yaml`. If `DEEPSEEK_API_KEY` is
available, generated operator instructions should recommend DeepSeek V4 Pro
plus the Claude SDK runtime unless the user supplied a different profile. If
that key is missing, preserve the runnable user/provider fallback and document
the missing recommended key without creating an alternate startup path.

## Role Prompt Design

Create task-local roles that inject domain expertise without over-constraining exploration:

- `roles/peer_generalist/skill.md`: implement/evaluate variants, preserve evidence, follow evaluation protocol.
- `roles/starter/skill.md`: build a strong editable starting system from task evidence and broadly applicable scientific or engineering knowledge.
- `roles/solver/skill.md`: improve the task objective and solution quality through focused implementation and evaluation.
- `roles/analyst/skill.md`: analyze data, frontier methods, attempts, logs, and failures before testing a proposal.
- `roles/pi_builder/skill.md`: synthesize promising mechanisms and propose next contracts.
- `roles/pi_skeptic/skill.md`: find leakage, brittle assumptions, invalid metrics, simulator/data artifacts.
- `roles/pi_portfolio/skill.md`: protect diversity and allocate directions across mechanism families.
- `roles/chair/skill.md`: merge PI views into concise per-peer agenda.
- Optional specialist roles only when the domain requires them, e.g. `simulator_specialist`, `data_validity_reviewer`, `literature_scout`.

For `literature_scout`, write task-local search policy: preferred sources,
forbidden leakage-prone sources, screening criteria, and the exact output fields
needed for prior-art risk. Keep the `panel.optional_roles` entry disabled unless
the task topology explicitly implements optional-role execution. Keep
`tool_server:literature_lookup` in the active `praxist_plugins.tools` list
by default, and mention the search policy in peer/PI instructions that actually
execute. Those executable instructions should ask for 1-3 focused lookup calls
when a proposal depends on external scientific context not already present in
local task documents, then require source title/identifier/URL and a short
"how this changed the hypothesis" note in findings or notebooks. The role
policy must also say that external lookup results may not trigger downloads,
installs, dataset acquisition, simulator setup, license requests, or runtime
provisioning during a run; they should be converted into current-resource
solution ideas or missing-resource notes.

Avoid overly narrow role definitions. Roles should guide expertise and risk checks, not force one algorithm family.

Role prompts must preserve Praxist machine-contract vocabulary. In particular,
if a role references preliminary, aligned, partial, incomplete,
repair-only, or other validation-candidate evidence as `parent_candidate`, its
`parent_usage` must be
a maturity action such as `validate`, `complete_validation`, `repair`,
`complete_scored_validation`, `falsify`, `compare`, `ablate`,
`ablate_or_falsify`, `ablation_followup`, or `audit`. Preserve/combine/pivot
language belongs in `next_step_intent` or rationale. Do not generate role
prompts that teach `parent_usage: preserve`, `parent_usage: combine`, or
`parent_usage: none` for a non-empty validation-candidate parent.

For expensive tasks, role prompts must preserve the user-owned staged-evaluation
contract. Under the recommended default, preliminary evidence motivates repair,
falsification, debugging, or aligned follow-up without ranking variants, while
aligned evidence can prioritize because it keeps near-complete coverage. If the
user explicitly assigns a reduced mode different ranking, maturity, parent, or
close permissions, prompts must obey those permissions and retain the actual
stage, effort, and coverage.

When the task configures an incubator, role prompts must preserve its metadata.
The evaluator should publish task-authorized, protocol-passed, non-suspect
performance results through the shared task-owned source lane accepted by confirmed and
incubator, with the primary metric, every incubator-axis metric,
`scored_complete=true`, protocol-integrity status, and suspect flags. Peers must
not predict which final target lane Praxist will select or overwrite the canonical
source label in a duplicate finding. If the task intentionally uses an explicit
incubator source instead, require `incubator_axis` and
`incubator_candidate_reason`. PI and Chair roles should preferentially use
materialized incubator entries as parents for repair, seed escalation,
ablation, falsification, comparison, and mechanism refinement.

## High-Value Research Directions

Write `assets/literature/research_directions.md` and mirror the compact version
in `task.yaml: research_direction`. Use local project evidence first, then the
Praxist scientific/literature lookup skill or `tool_server:literature_lookup`
when available, and bounded web search for gaps that local/public lookup cannot
answer. Include:

- 8-15 high-value exploration directions;
- why each direction is plausible;
- expected metric signature;
- implementation surface;
- risk/leakage/failure mode;
- whether the direction is feasible with the current local data, simulator,
  dependencies, hardware, and evaluator; if not, describe how to adapt the idea
  to current resources instead of requiring new downloads or installs;
- diversity labels such as mechanism family, intervention surface, intent, semantic family, parent lineage, novelty axis.

These directions strongly shape Praxist diversity and performance ceiling. Favor breadth across mechanisms and intervention surfaces.

The diversity labels are shared by initial-generation QD-DIG and later
PI-synthesis QD. Make them broad enough for exploration and concrete enough to
distinguish candidate mechanisms without encoding one domain into Praxist itself.

## Required Validations

Before finishing:

1. Run the task evaluator in smoke/resolve mode if available.
2. Run `praxist resolve` or equivalent resolve-only command if Praxist is installed.
3. Confirm `task.yaml` paths are relative to the task project.
4. Confirm datasets/simulators are metadata references, not copied raw assets.
5. Confirm no raw API keys or private credentials were written.
6. Confirm `.gitignore` excludes `experiments/`, raw data, logs, checkpoints, caches, and local secrets.
7. Confirm `assets/resource_plan.md` records the unchanged baseline observation,
   hardware, observed or unknown bottleneck, GPU shape where applicable,
   central scheduler profiles, initial/max total concurrency, experiment cost,
   chosen cohort size, generation duration, initial-DIG/later-QD defaults, and
   continuous-evolution/Gems reset policy.
8. Confirm baseline performance is either measured from compact existing
   evidence, measured by a user-approved `experiments/baseline_bench_*` run,
   explicitly marked as a user-selected zero placeholder, or reported as blocked
   with exact missing requirements.
9. Confirm real-research task configs default to `dig_lite.enabled: true`,
   `dig_lite.generation_scope: initial_only`,
   `quality_diversity.enabled: true`, both generation-specific QD switches
   enabled, and `gems.enabled: false` unless the user explicitly requested an
   override. Confirm later-generation QD is described as soft guidance in the
   existing single-PI or Multi-PI synthesis path, not as a second DIG allocator.
   If `dig_lite.innovation.enforce_forward_slots` and constructive peer-mix
   feedback are both enabled, report their independent settings and check that
   the chosen cohort still leaves useful forward capacity; treat a mismatch as
   an advisory task-design issue rather than a launch guard.
10. Confirm `praxist_plugins.tools` explicitly lists
    `tool_server:evaluation_tools`, `tool_server:frontier_tools`,
    `tool_server:finding_graph_query`, `tool_server:memory_tools`,
    `tool_server:prior_work_tools`, `tool_server:run_report`, and
    `tool_server:literature_lookup`.
11. Confirm `evaluation.frontier_lanes` includes task-owned mature-result,
    durable incubator, validation-candidate, and diagnostic/control lanes when
    the task has expensive, staged, or multi-axis evaluation. Run the
    task-local lane-routing regression and confirm every parent-eligible lane
    is reachable from an evaluator-produced clean summary whose mode is
    parent-authorized by the protocol-intent table. For shared
    confirmed/incubator evidence, verify the evaluator emits a common source
    label accepted by both lanes and does not hard-code every parent-authorized result
    to confirmed. When a distinct incubator axis is justified and configured,
    also verify a parent-authorized non-dominated fixture outside the confirmed top-k
    remains eligible for incubator selection. Never invent an axis for a
    single-metric task. Fixtures from modes marked non-parentable by the
    user-owned protocol intent, plus protocol-failed and suspect fixtures, must
    remain non-parentable; an explicitly authorized reduced mode is a positive
    parent fixture, not a negative test.
12. Confirm `evaluation.maturity_policy`,
    `evaluation.constructive_peer_mix_enabled`,
    `evaluation.constructive_target_ratio`, and `evaluation.launch_guard` are
    present. When the maturity policy distinguishes close-grade results from
    preliminary, partial, diagnostic, or progress evidence, require a positive
    `synthesis_trigger.mature_quorum_fraction` (normally `0.25`) and verify
    `ceil(cohort_size * fraction) >= 1` is physically reachable. Treat `0.0` as
    an explicit evidence-blind normal-close choice, not as the safe default or
    a deadlock workaround; require current-user confirmation and record the
    rationale before generating it.
13. Run the shortest valid scored evaluator path through its real canonical
    summary writer. Validate the produced file with
    `praxist resolve <task_path> --result-summary <summary_path>`, then run the
    summary-to-finding round trip and verify Praxist preserves exact finite
    `effort_ratio` and `coverage_ratio` values. Standalone result findings
    without a canonical summary reference must carry the ratios directly. If
    the evaluator cannot emit them, do not launch with `require_ratio_gate:
    true`: repair the evaluator, or obtain explicit user approval to disable
    the gate and document the task-owned legacy fallback. Do not solve this by
    inventing or requiring stage labels.
    When the task has result-affecting non-code settings, run three deterministic
    contract fixtures. Verify that an omitted setting and the same explicitly
    supplied resolved default produce equal
    `source_result_effective_config_sha256` values, while a genuinely different
    effective value produces a different digest. Then
    verify an exact-replication fixture reports `matched` only when its complete
    configuration equals the selected parent digest. A task with no such
    settings may omit this optional contract; do not block legacy or ordinary
    results for missing effective-config metadata.
14. When a separate durable incubator lane is justified and configured,
    confirm it is not stricter than the confirmed lane:
    it should accept protocol-authorized, protocol-passed, non-suspect
    Pareto/new-high candidates that are not yet clean promotable, while
    excluding every mode the intent table marks non-parentable. Under the
    recommended default those exclusions include smoke, partial, scout-only,
    validation-only, protocol-failed, unscored, and suspect evidence.
15. When an incubator has multiple axes, confirm they represent distinct metric
    families, not repeated measurements of the same metric across subsets.
16. When an incubator is configured, confirm task prompts explain that an empty
    incubator over many generations can make Praxist lose strong parents and
    stagnate, and that Peers should publish incubator-critical metadata in
    top-level fields or `metrics`.
17. Confirm task prompts explain that validation candidates remain visible as
    compact non-frontier signals for follow-up, while durable frontier facts
    remain restricted to clean measured evidence. For QD-enabled tasks, also
    confirm PI/Chair peer contracts plan task-owned axes under
    `planned_dimensions`, peer findings report actual implemented values under
    `design_dimensions`, and diagnostics can compute planned and realized HHI
    separately without copying plans into result evidence.
18. Confirm generated launch instructions recommend `model_provider:deepseek_alias`, `deepseek-v4-pro`, and `agent_runtime:claude_sdk` when `DEEPSEEK_API_KEY` is available, unless the user explicitly supplied a different Praxist provider/model/runtime. Confirm `agent.reasoning_effort` is `max` unless the user explicitly selected `auto`, `off`, `low`, or `high`, and ensure no task-local provider transport code was generated. Verify the default runtime uses `claude-agent-sdk==0.2.136`. Treat `praxist-takeover-codex` as an explicit Codex-native mode choice (`codex_sdk + openai_compatible + gpt-5.6-luna`) even when provider keys are present, unless the user selected another catalog-verified model. If the selected runtime is `agent_runtime:codex_sdk`, verify `openai-codex==0.147.0`, `claude-agent-sdk==0.2.136`, SDK/MCP dependencies, and `codex-relay==0.5.5` when applicable without creating task-local runtime infrastructure. For native OpenAI, accept a matching environment API key or saved ChatGPT login and keep both outside the task.
19. Confirm generated PI/Chair role prompts distinguish `next_step_intent`
    research intent from `parent_usage` machine-contract maturity actions for
    validation-candidate parents.
20. Confirm generated peer role prompts and task prompts prefer synchronous
    evaluator calls. Any explicitly supported background evaluation must use
    the central `protected_pids launch` submission facade with `$PRAXIST_PEER_ID`, a
    stable semantic tag, declared profile, and work class; publish
    the standard result summary, and avoid arbitrary `wait_for_file` polling.
    Confirm they explicitly forbid using runtime-private
    `tasks/<task-id>.output` byte count as task completion; use the runtime's
    structured task notification/exit status or a documented task-owned
    progress/result contract.
21. For expensive tasks whose user-owned intent permits staged evidence,
    confirm the task defines separate preliminary, aligned, and complete mature
    protocols, or records the explicitly selected simpler/reduced protocol.
22. When preliminary evidence exists, confirm generated prompts apply exactly
    the intent table's ranking and promotion permissions. The recommended
    default keeps preliminary evidence as triage and lets aligned evidence
    prioritize variants, but an explicit user decision may assign different
    semantics when the real stage and coverage remain visible.
23. When aligned evaluation exists, confirm it preserves near-complete
    evaluation-unit coverage and saves compute mainly through reduced epochs, steps, rollout
    horizon, simulator steps, inner-loop iterations, or other fixed
    training/optimization budget. If evaluation-unit coverage is reduced,
    confirm the coverage ratio, stratification, hard-case inclusion, and
    omitted-unit rationale are documented; otherwise label it preliminary or
    partial.
24. When early stages exist, confirm resource plans and reporting instructions
    require early-vs-complete calibration checks to be computed separately for
    preliminary and aligned evidence.
25. Confirm metric discovery recorded the domain ranking convention and whether
    variance/confidence/lower-bound robustness should affect frontier ordering.
    If robustness matters, confirm `evaluation.frontier_lanes` includes the
    required robust metric or Pareto axis rather than raw-primary-only sorting.
26. Confirm evaluator outputs authorized as mature include expected and
    completed budget fields plus a protocol-integrity flag. If the proposed
    highest-authority effort is materially below task evidence or the declared
    reference effort, confirm the user approved it or the protocol was
    downgraded to non-parent evidence. Never apply a universal epoch/step
    threshold or override an explicit reduced-protocol choice.
27. Before expensive multi-unit fan-out, confirm the task-appropriate
    build/load/startup check, public-interface validation, one-unit canary,
    canonical-summary validation, and summary-to-finding projection all pass
    through the real task runtime. If the task declares an
    independently trusted evaluator, also confirm its task-owned verifier
    rejects modified or unattested evidence and peers cannot replace the
    authoritative result. Do not require that trust mode for peer-authored
    evaluators.
28. When central scheduling is applicable, confirm its profiles and concurrency
    came from an unchanged baseline observation. When an external owner is
    necessary, confirm the limitation and bounded concurrency are documented.
    Run a tiny scheduler/evaluator
    integration preflight through the same agent-runtime guard environment used
    by a real peer, not from an unguarded Codex/controller process. The READY/GO
    launch barrier, final evaluator, and one descendant must complete; the final
    evaluator and descendant must preserve the scheduler-assigned UUID mask
    exactly. Also verify that task code remains unable to overwrite scheduler,
    frontier, or another peer's state. Do not accept a descendant that rewrites a
    UUID to ordinal `0`. Confirm CPU profiles have no per-job core reservation,
    GPU utilization and peak VRAM are checked independently, settled
    reservations are not double-counted against live driver load, GPU demand
    unknowns are exclusive, lifecycle `running` is reported separately from
    point-in-time resource activity, and no accelerator fallback exists
    unless the task declares scientific equivalence.
29. Only for tasks that selected the Praxist-managed NVIDIA/CUDA backend, confirm
    the generated harness uses
    `PRAXIST_ASSIGNED_GPU_UUIDS` as the authoritative physical assignment across
    evaluator, trainer, worker, shell, and container boundaries. Confirm the
    UUID, multi-device, missing-mask recovery, conflicting-mask rejection,
    standalone, and CPU contract tests pass. Scan for descendant-environment
    assignments that replace a Praxist UUID mask with `0` or `str(gpu_id)` and
    reject launch readiness when found.
30. When that selected backend exposes at least two usable devices and the task
    is multi-device capable, run the bounded non-zero-UUID parent/child CUDA
    preflight and verify the compute
    PID against driver-observed UUID data. Confirm cleanup leaves no process or
    allocation. Otherwise record why physical verification was unavailable;
    do not substitute a first-device-only check or CPU-vs-accelerator speed
    comparison. For CPU-only, unified-memory, task-managed, and other
    accelerator backends, validate the task's actual process handoff instead of
    inventing CUDA/UUID requirements.
31. Confirm the scheduler default profile matches the public evaluator's normal
    resource shape, including runtime-assisted calls that omit `--profile`.
    Confirm ordinary analysis is not submitted as an experiment, explicit
    profile selection is used where supported, `supply_signal_enabled: true`,
    `supply_lease_seconds: 600`,
    `mature_supply_fraction: 0.25`, `mature_supply_redundancy: 3.0`,
    `mature_assessment_min_completion_probability: 0.25`, and a bounded
    consecutive-sample threshold unless the user deliberately disables
    idle supply or mature-priority feedback.
32. For every expensive full/mature evaluator, record its natural independent
    unit count and prove that any declared multi-accelerator profile actually
    spreads at least two units across distinct assigned UUIDs. For other
    bottleneck types, confirm task-owned concurrency respects the measured
    memory, I/O, simulator, license, service, or external-capacity bound.
    Confirm repeated identical non-scientific failures stop remaining
    non-runnable units promptly while valid low scores and heterogeneous
    scientific outcomes still complete normally.
33. Exercise the supply-feedback contract: with an open generation, unused
    profile-admissible slots, insufficient queued work, and consecutive
    low-pressure samples, at most N directed one-experiment leases appear for N
    slots and wake only their selected productive idle peers. Confirm Gen0 works,
    leases request only work already justified by the current research plan,
    duplicate submission does not consume a lease, lease expiry limits only the
    submission response window rather than admitted experiment runtime, and no
    usable lease remains after Closing or Stop; cleanup failures may remain
    visible as `release_pending`. Verify mature and follow-up
    conversion rates plus declined/expired/revoked/stale reasons remain
    internally consistent.
34. Exercise maturity-debt feedback with canonical result evidence. Confirm
    `Q/M/D/A_target` is visible in scheduler status; queued/running mature
    semantic jobs and mature leases count once; three bounded commitments per
    missing result are attempted only while physical capacity exists; and
    reaching `Q` withdraws excess mature leases without deleting ordinary
    Pareto follow-ups. Confirm the first wave retains exploration and that
    assessment rejects new scout/ordinary work while allowing mature top-ups;
    `CLOSING_SIGNAL` must remain the strict zero-new-work boundary.
35. Run a lightweight closing-policy lifecycle regression against the generated
    task configuration. Show that raw finding/peer/time thresholds with zero
    mature results do not produce a normal-success `CLOSING_SIGNAL` when the
    task defines close-grade evidence; below-quorum assessment keeps eligible
    mature top-ups available; reaching the configured mature-peer quorum
    permits normal close after active work drains; and safety-cap or
    cohort-drained close remains available while explicitly recording
    insufficient maturity. Also verify protected work is allowed to drain.

## Final Report

End with a concise report:

| Item | Status |
|---|---|
| Research root | path |
| Output task path | path |
| Runtime env | detected and verified / missing |
| Data/simulator | detected / missing |
| Baseline code | runnable / blocked |
| Baseline performance | found / missing measurement |
| Metrics | primary + auxiliary |
| Roles | files created |
| Research directions | count + source basis |
| Resource plan | bottleneck + cohort size + per-generation hours |
| Closing policy | mature quorum + required peer count + lifecycle regression result |
| Accelerator handoff | selected backend + applicable contract tests / not applicable / unavailable reason |
| Praxist model/runtime | DeepSeek V4 Pro + Claude SDK when `DEEPSEEK_API_KEY` is available / user-specified or available-provider fallback |
| Frontier lanes | minimum task-justified retention structure; add incubator/candidate/diagnostic lanes only when protocol or metric evidence supports them |
| DIG / QD | absolute gen0 DIG status; gen0 QD status; later PI-synthesis QD status |
| Gems reset | disabled continuous evolution / enabled from user or diagnostic plateau evidence / blocked with reason |
| Validation | commands run and results |
| Blockers | exact next action |

Do not start a long Praxist run unless the user explicitly asks.
