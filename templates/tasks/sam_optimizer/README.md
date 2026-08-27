# SAM Optimizer Research Template

This is a runnable, domain-specific reference template for researching
drop-in PyTorch SAM optimizer variants. Unlike the placeholder templates, it
ships a tiered evaluator and benchmark harness. Real runs require a task-owned
Python 3.11+ environment with NumPy, PyTorch, torchvision, a compatible CUDA
runtime, and local CIFAR/Tiny-ImageNet data. This repository intentionally does
not pin or create that environment. Copy the task to an external project,
declare its verified `runtime_environment`, and keep run artifacts out of the
Praxist source checkout.

The scientific intervention surface is deliberately narrow: peers may change
the optimizer implementation, but not the model, training loop, datasets, or
evaluation protocol. The primary metric is `mean_test_accuracy`; mature parent
evidence is the task-owned T3 protocol, while T1/T2 remain preliminary signals.

## Validate The Template

Resolve the task contract without starting research:

```bash
RUN_ROOT=$(mktemp -d /tmp/praxist-sam-template-XXXXXX)
DEEPSEEK_API_KEY=... praxist resolve templates/tasks/sam_optimizer \
  --run-dir "$RUN_ROOT/run" \
  --model-provider model_provider:deepseek_alias \
  --runtime agent_runtime:claude_sdk \
  --model deepseek-v4-pro
```

Check the public evaluator interface independently:

```bash
python templates/tasks/sam_optimizer/evaluations/pareto_tiered/run.py --help
```

## Praxist Contract

Provider-specific context efficiency is runtime-owned. This template requires no
task setting for it: Codex-native mode and OpenRouter runs use
lossless finding-event coalescing automatically, and direct DeepSeek runs remain
unchanged. Canonical findings and full tool-result references remain available.

This task implements the current **Praxist research-loop contract**:
- `task.yaml` defines `evaluation.anchor_metrics` (multi-anchor
  Pareto frontier: accuracy / efficiency / generalization-gap / flatness
  each get their own anchor slot per generation)
- `task.yaml` defines `evaluation.diversity_dimensions` (6 axes
  of within-scope innovation — see `prompt_task.jinja2` for what peers
  are asked to articulate)
- PI/Chair contracts use those axes under `planned_dimensions`; result findings
  use `design_dimensions` for the optimizer that actually ran.
- `task.yaml` defines the task-local `tiered_eval` labels (T1/T2/T3) — peers escalate
  compute gradually; the evaluator is the single public entrypoint for the
  per-tier seed/dataset/epoch values. `task.yaml:tiered_eval` mirrors
  `TIER_DEFAULTS` in the internal benchmark runner at
  `assets/harness/benchmark/run_benchmark.py`. These labels and epoch counts
  describe this SAM benchmark only; they are not Praxist-wide maturity semantics.
- This template explicitly selects its compatible NVIDIA/CUDA execution path.
  `compute_budget.resource_scheduler` declares central `gpu_benchmark` and
  `cpu_probe` profiles. Praxist performs host-wide admission and supplies the exact
  assigned GPU UUID to the final evaluator process. Directed idle-supply offers
  use the standard 600-second submission response window; accepted experiment
  runtime remains governed by the generation and evaluator contract. This
  backend-specific handoff is not a requirement for CPU-only, unified-memory,
  or other accelerator tasks.
- `generation_policy.max_generations = 8` with **event-driven
  termination** (no plateau early-stop, no 3-cycle annealing). A gen
  ends when `findings >= 30 AND elapsed >= 2h AND >= 3 distinct peers
  contributed`, OR at the 4h safety cap.

- **DIG runs only at absolute gen0 by default** through
  `dig_lite.enabled=true` and `generation_scope=initial_only`. Gen0 QD can
  reselect among each peer's validated DIG candidates. Later generations use
  the normal PI proposal pool and Chair allocation under the independent
  `quality_diversity.later_generations_enabled` switch, without rerunning DIG.

- **Gems reset is configured but disabled by default**. This template keeps a
  task-specific 6-generation cadence and compact caps in `task.yaml` as an
  opt-in profile, but starts in continuous-evolution mode like newly
  initialized tasks. Enable periodic reset only after an operator request or
  diagnostic plateau evidence. For this tiered SAM task, T1/T2 are
  preliminary task-local labels; Gems uses
  `selection_policy: mature_evidence_top_k` with
  `min_mature_eval_units: 15`, the complete T3 seed/dataset-unit count, and
  `evidence_stage_min_units: {T1: 3, T2: 6, T3: 15}` for its task-local staged
  thresholds. The Praxist keys still count evaluation units, not generic cells.
  Its positive mature close quorum prevents preliminary or diagnostic findings
  from normal-closing a generation without complete task-defined evidence.
  The scheduler's mature supply target remains advisory and is not a substitute
  for that close gate.

- **PI synthesis** runs between every pair of generations and assigns
  per-peer role contracts (`exploit` / `falsifier` / `bridge` /
  `anti_mainline` / `theorist`) via `<run_dir>/agendas/research_agenda_genN.yaml`.
  With `multi_pi.enabled=true`, Builder, Skeptic, Portfolio, and optional
  External Validity roles write independent memos; the Chair merges them into
  the final agenda using evidence cards from
  `<run_dir>/research_memory/ledgers/*.yaml`. The single-PI fallback remains
  available through the task setting.

The fixed scientific constraint is that only the optimizer implementation
differs. Timing estimates must be measured again on the target host instead of
being copied from this reference template.

## Local experiment outputs

This task keeps complete run artifacts under a local gitignored
`experiments/` directory. Praxist run directories should be created as
`experiments/run_<timestamp>_sam_optimizer/`, containing `results/`,
`variants/`, `shared_findings/`, logs, ledgers, frontier state, and graph
artifacts.

Within those run directories, current facts are owned by measured result
summaries, structured findings, `frontier/frontier_manifest.json`, committed
`gems/gems_state.json`, and generation boundary markers. PI evidence packs,
PI agendas, leaderboards, rendered prompts, prompt layouts, diagnostics,
`docs/praxist_reports/` reports, and behavior reports are derived views or audit
snapshots. They remain important for replay and behavior analysis, but they
should not be hand-edited or treated as independent promotion truth.
Lower-tier, preliminary, repair, partial, failed-but-informative,
diagnostic, and late-after-boundary SAM records should remain visible as
validation signals through structured findings and task-owned lanes. They can
guide triage, repair, and escalation decisions, but they are not clean
frontier/Gems facts until the evaluator produces canonical T3 or revalidated
evidence. Complete non-suspect T3 results that pass this task's ratio gate use
the shared `performance` source, allowing confirmed and the parentable
`incubator` target to select independently. T1/T2 signals remain in the
non-parentable `task_candidate` lane until T3 revalidates them. SAM's T1/T2 tiers share metric semantics and gate rules
with T3, but they intentionally cover fewer seed/dataset cells; do not treat
them as near-full ranking evidence. If compute must be saved in a new aligned
profile, prefer reduced training budget over dropping most evaluation cells.
The task expresses this contract with `complete_stage_labels: [T3]`,
`preliminary_stage_labels: [T1, T2]`, `require_ratio_gate: true`, and explicit
`parent_eligible` flags on every lane.

The materializer discovers nested result summaries named `summary.json`,
`evaluation_summary.json`, `eval_summary.json`, `tiered_eval_summary.json`, or
`custom_*_tiered_eval_summary.json` (`result_summary.json` is a compatibility
name) and transfers lane, maturity, ratio, protocol, and diagnostic metadata
into structured findings.

Before launching an adapted task, run the shortest valid scored evaluator path
and validate its actual summary with `praxist resolve <task_path>
--result-summary <summary_path>`. The check proves that the ratio gate can read
finite telemetry; it does not certify the result's score.

The packaged `assets/` tree is a read-only input bundle for harnesses,
baselines, metadata, literature, and small regression fixtures. Do not write
live run outputs, generated variants, JSON metrics, figures, logs, or findings
there. If a run produces a variant worth preserving, promote it in a separate
human-reviewed curation step outside the Praxist system repository.

Literature lookup is available for optimizer context. If a paper depends on
unavailable datasets, checkpoints, packages, licenses, hardware, or training
environments, do not acquire them during a run. Adapt the method to the existing
SAM benchmark harness and record missing resources only as notes.

Praxist injects `PRAXIST_TASK_PROJECT_PATH`, `PRAXIST_WORKSPACE_ROOT`, `PYTHONPATH`,
optional runtime-environment vars (`PRAXIST_TASK_PYTHON`, `PRAXIST_TASK_VENV`,
`PRAXIST_TASK_SHELL_PREFIX`), the central scheduler endpoint, and the dataset env
(`PRAXIST_DATA_DIR` / `PRAXIST_SAM_DATA_DIR`) into each peer session. Evaluation
launches should use the absolute
`$PRAXIST_TASK_PROJECT_PATH/evaluations/pareto_tiered/run.py` entrypoint and, when
specifying data explicitly, `--data-dir "$PRAXIST_DATA_DIR"` so the task works from
any current working directory.

## Task venv integration

External tasks that require a dedicated virtual environment should keep that
configuration inside the task project. The preferred path is to declare it in
`task.yaml`:

```yaml
runtime_environment:
  cwd: task_project
  venv: .venv
  # python: .venv/bin/python  # optional; inferred from venv when omitted
  path_prepend:
    - bin
  env:
    TASK_MODE: dogfood       # non-secret task vars only
```

Praxist validates the paths and injects `PRAXIST_TASK_PYTHON`, `PRAXIST_TASK_VENV`,
`VIRTUAL_ENV`, `PRAXIST_TASK_SHELL_PREFIX`, and `PATH` into peer sessions. Task
commands, harnesses, and evaluations should therefore call
`"${PRAXIST_TASK_PYTHON:-python}"` instead of hard-coding `python`.

Keep launch behavior in the Praxist CLI rather than adding a task-local research
launcher. Declare the working directory, virtual environment, interpreter, and
non-secret environment in `runtime_environment`; keep dependency installation
as a separate task setup step. Start or resume the task through `praxist start`
or `praxist resume` so lifecycle state remains registry-backed.

For peer-driven optimizer experiments, use the compact task-local evaluator:

```bash
"${PRAXIST_TASK_PYTHON:-python}" "$PRAXIST_TASK_PROJECT_PATH/evaluations/pareto_tiered/run.py" \
  --variant-path "<path-to-variant.py>" \
  --output-dir "<run-results-dir>/<experiment-name>" \
  --data-dir "$PRAXIST_DATA_DIR" \
  --max-tier "${PRAXIST_SAM_MAX_TIER:-T3}"
```

The wrapper runs T1/T2/T3 in order, applies the documented gates, writes raw
benchmark JSON/logs under the run results directory, and prints a small summary
for the agent context.

## Multi-peer GPU coordination

Peers submit evaluator commands through the Praxist launcher. This checked-in
template has no timestamped process-lifetime resource trace, so its GPU demand
is deliberately unknown and receives exclusive placement. A generated task may
enable shared packing only after measuring utilization and peak VRAM from the
unchanged evaluator. The central scheduler writes the assigned UUID
to `CUDA_VISIBLE_DEVICES`, `NVIDIA_VISIBLE_DEVICES`, and
`PRAXIST_ASSIGNED_GPU_UUIDS`, and tracks the whole process group. The benchmark
runner and its descendants preserve that mask exactly. Standalone invocation
without a scheduler keeps the legacy local picker only for single-user use.
If a delivered CUDA/NVIDIA mask conflicts with `PRAXIST_ASSIGNED_GPU_UUIDS`, the
runner fails with a binding-integrity error rather than replacing the physical
GPU UUID assignment with a framework-local integer device.

The public benchmark evaluator therefore uses the `gpu_benchmark` default
profile. Explicitly CPU-only probes use `cpu_probe`. This template keeps each
top-level benchmark at one assigned GPU because its current evaluator does not
distribute independent dataset/seed cells across multiple scheduler-assigned
UUIDs. A generated task may add a wider profile only after implementing and
testing that child-unit distribution; configuration alone is not multi-GPU
support.

## Research scope is intentionally constrained

Unlike open architecture-exploration tasks, this task pins
the training harness, model, dataset, and protocol. The only locus
of innovation is the `torch.optim.Optimizer` subclass itself.

**Why this constraint matters**: the deliverable of this research is
a portable optimizer that works in any PyTorch training framework
(HuggingFace, Lightning, raw PyTorch, etc.). If a variant requires
custom training-loop modifications to deliver its gains, it has lost
the property that makes it useful in the first place. The constraint
is not a research limitation — it IS the research goal.

This is enforced by the `prompt_task.jinja2` "Research Scope" block,
not by a generic Praxist diversity check. The 6 `diversity_dimensions`
in `task.yaml` describe **within-scope** innovation axes; peers
should articulate variation along these axes WITHOUT escaping the
PyTorch Optimizer subclass / fixed train loop / fixed protocol
constraints.

Initial QD-DIG and later PI-synthesis QD use those axes plus candidate-level labels
to keep a generation from collapsing into one SAM family. Gems preserves the strongest mature candidates
after reset, but it should remain a compact evidence and parent-candidate
source, not a fixed quota that forces every peer to copy the same optimizer.

## Implementation notes

- `baseline/train.py` provides a faithful `ASAM` class (Kwon et al.,
  2021) that scales the perturbation elementwise by `|w|`. This is
  distinct from vanilla `SAM`. Reminder: ASAM typically uses larger
  `rho` (0.5–2.0) than SAM's default 0.05.
- The cached ASAM row in `assets/baselines/results.jsonl` does not use the
  task's ASAM implementation. It is a regression fixture, not eligible ASAM
  evidence. Remeasure ASAM through the task evaluator before using it as a
  scientific baseline.

## Supplementary material

- `assets/literature/recent_sam_top_venue_review_2024_2025.md` — curated top-venue SAM
  survey (English). **Required reading** for gen 0 free-explore peers
  and for any gen ≥ 1 peer whose PI role contract spawns a new variant.

Prescriptive "direction hints" are intentionally omitted from
`description.md` so that an agent is not biased toward any particular
family of SAM variants during exploration. The 6 diversity dimensions
in `task.yaml` provide STRUCTURAL guidance (which axes count as
"different directions") without saying which direction to take.
