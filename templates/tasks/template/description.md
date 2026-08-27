# Praxist Task Template

This is a scaffold for creating an external Praxist task project. It is not a
scientific benchmark and should not be used for real research until the task
owner replaces the placeholder domain material.

## Required Replacements

Before running a real experiment, replace:

1. `task.yaml` task identity, metrics, budget defaults, workflow refs, roles,
   audits, evaluations, tools, and output policy.
2. `description.md` and `prompt_task.jinja2` with the task's actual research
   brief, constraints, and agent instructions.
3. `roles/` with task-owned role skills.
4. `audit_rules/` with task-owned scope, claim, and agenda criteria.
5. `evaluations/pareto_tiered/run.py` with the task's single public evaluation
   entrypoint.
6. `assets/` with the task's harness, baselines, fixtures, dataset metadata,
   literature packs, and optional reference implementations.

## Boundary

The task project is an explicit Praxist input selected with `--task-path`. Task
facts, benchmark rules, and domain-specific prompts belong here, not in
`praxist/core` or generic system plugins.

## Default Research Loop Policy

This scaffold declares gen0 DIG, independently switchable initial/later QD, and a continuous-evolution Gems policy in
`task.yaml` so new tasks start from the standard research-loop shape:

- pre-code DIG planning is enabled;
- gen0 cohort-level QD allocation is enabled; later QD is explicitly disabled
  for this one-generation smoke fixture;
- periodic Gems reset is disabled by default.

Replace the placeholder diversity dimensions with task-specific axes before a
real run. Enable periodic Gems reset later only if the operator requests it or a
diagnostic pass identifies a performance ceiling and recommends a reset cadence.
