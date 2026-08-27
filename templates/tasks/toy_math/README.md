# Toy Math Task Template

This is a deterministic, resolve-only, non-ML fixture. It demonstrates how a
mathematical conjecture task can own its research brief, roles, audit rules,
evidence lanes, and task assets while reusing Praxist startup, plugin
resolution, budget, replay, and test contracts. It does not prove conjectures
or provide a production evaluator.

## Layout

This fixture deliberately has no provider-specific context settings. Praxist
automatically coalesces finding-only continuation wakeups for Codex-native mode
and OpenRouter runs, preserves direct DeepSeek behavior, and
keeps canonical evidence available for exact re-reading.

The directory mirrors the full external task layout:

- `task.yaml`: canonical task descriptor for the toy math smoke task.
- `runner.py`: fake workflow fixture runner hook.
- `description.md`: agent-facing task brief placeholder.
- `prompt_task.jinja2`: task-owned prompt block placeholder.
- `roles/`: task-local role templates, including an optional disabled
  `literature_scout` role that demonstrates where domain search policy belongs.
- `audit_rules/`: task-local declarative audit templates.
- `evaluations/`: task-local evaluation templates, including a placeholder
  public evaluator at `evaluations/pareto_tiered/run.py`.
- `assets/`: harness, baselines, literature, reference implementation
  placeholders, and regression fixtures.

The active literature lookup tool is for context only. If a source mentions
unavailable data, software, or a new runtime environment, keep the run local and
adapt the idea to the current toy harness.

The internal smoke runner uses deterministic fake agents. It does not prove a
real theorem and its placeholder evaluator is not meant for production
experiments; it proves the core-plugin path is task-agnostic.

## Smoke Test

```bash
RUN_ROOT=$(mktemp -d /tmp/praxist-toy-math-XXXXXX)
praxist resolve templates/tasks/toy_math --run-dir "$RUN_ROOT/run"
```

An adapted scored evaluator can additionally validate its actual output with
`praxist resolve <task_path> --result-summary <summary_path>`. This fixture's
task-owned stage labels do not replace finite maturity ratios.

## DIG, QD, And Gems

The fixture declares gen0 DIG, independent QD, and a continuous-evolution Gems policy explicitly to
demonstrate the standard task shape even for non-ML research:

- Gen0 QD-DIG uses math-specific diversity labels such as conjecture family,
  intervention surface, intent, semantic family, lineage, and novelty axis.
- PI/Chair contracts record intended axes as `planned_dimensions`; findings
  record the realized mathematical construction under `design_dimensions`.
- Later PI-synthesis QD and constructive next-generation feedback are explicitly
  disabled because this fixture has only one generation.
- Periodic Gems reset is disabled by default (`gems.enabled: false`).
- The smoke fixture uses one task-owned evaluation unit for both
  `min_mature_eval_units` and `evidence_stage_min_units.complete`.
- Its positive mature close quorum prevents raw progress findings from
  normal-closing the generation without task-defined complete evidence.
- Operators can enable Gems reset later after a diagnostic pass finds a
  performance ceiling and recommends a reset cadence.

The default fixture still runs only one generation.

The scheduler's mature supply target is advisory and cannot replace this close
gate. A generated task should use a zero mature quorum only when it intentionally
has no distinct close-grade evidence contract and the operator confirms that
information-density closing is acceptable.

## Artifact Ownership

Toy-math peers should publish conjecture, counterexample, and diagnostic
evidence as structured findings. They should not hand-write Praxist frontier, Gems,
prompt-layout, research-memory, or diagnostic state. Leaderboards and PI
evidence packs, plus `docs/praxist_reports/` reports, are derived views for
context; canonical current state is owned by findings/results, frontier, Gems,
and committed generation boundaries.
Partial conjectures, failed proof attempts, counterexamples, and diagnostics
may still be useful validation signals. Keep them in structured findings so
later peers and PI can revisit them without treating them as durable facts or
parents. The complete and incubator lanes in `task.yaml` explicitly set
`parent_eligible: true`; candidate and diagnostic lanes set it false. This
fixture also declares task-owned complete/preliminary labels and a required
effort/coverage ratio gate instead of relying on global tier names.
An expanded evaluator should emit the shared `performance` source for ordinary
clean complete conjectures and regression-test that both parent targets remain
reachable; Peers should not guess the final target lane.

A real evaluator derived from this fixture may write nested summaries under
`results/**/` as `summary.json`, `evaluation_summary.json`,
`eval_summary.json`, `tiered_eval_summary.json`, or
`custom_*_tiered_eval_summary.json` (`result_summary.json` is accepted for
compatibility). Lane, maturity, ratio, protocol, and diagnostic metadata should
be structured so the materializer can transfer it into findings.
