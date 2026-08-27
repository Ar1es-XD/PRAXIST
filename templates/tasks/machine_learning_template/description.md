# General Machine Learning Task Template

This is a pure template for turning a runnable machine learning project
into a Praxist task project. It is not a real task until the task owner replaces
the placeholder data metadata, evaluator, baselines, resource plan, and research
brief.

## Intended Task Shape

Use this template when a project has:

- public or task-owned data metadata;
- a clear prediction artifact or model output format;
- a credible evaluator or validation protocol;
- baseline evidence or a plan to measure baselines;
- a machine learning objective that can improve through iterative experiments.

## Evidence Model

The user-approved protocol determines how each evaluator mode may be used.
Under the template's recommended default, Praxist receives four kinds of
evidence:

- task-defined mature results with the required metric and protocol evidence;
- lower-admission durable incubator results: task-authorized, protocol-passed,
  non-suspect Pareto/new-high variants that are worth developing further but
  are not yet clean confirmed winners;
- task-defined preliminary or aligned evidence, repairs, partial runs, and
  smoke checks that are promising but immature;
- diagnostics, controls, negative evidence, and invalid-output analysis.

Evidence authorized as mature can enter the `confirmed` frontier lane when the
finding includes the task primary metric and `scored_complete=true`.
Non-suspect Pareto/new-high mature evidence that is not yet clean should enter the
`incubator` lane instead of disappearing or being treated as confirmed. Partial
and diagnostic evidence should remain visible without pretending to be mature.
The task owns the stage vocabulary through `complete_stage_labels` and
`preliminary_stage_labels`; labels such as `preliminary`, `aligned`, and
`complete` are illustrative names rather than Praxist-wide tiers. When early ranking is
needed, an aligned protocol should keep the complete protocol's metric
semantics and same or near-complete coverage, while saving compute through a lower
task-defined effort ratio. Only lanes with `parent_eligible: true` may supply
durable implementation parents. Lower-tier and diagnostic lanes must remain
`parent_eligible: false` until evidence from a parent-authorized mode revalidates
them. An explicitly authorized reduced protocol may be mature; preserve its real
stage, effort, and coverage instead of relabeling it as full.

## Replace Before Real Use

Replace the placeholder evaluator under `evaluations/primary/`, fill
`assets/task_context/context_template.json`, write a real `assets/resource_plan.md`,
and update `task.yaml` before starting a non-smoke Praxist run.
