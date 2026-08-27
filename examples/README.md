# Complete Examples

`examples/` contains complete, runnable reference research projects. Each
example preserves its task-specific code, evaluator, task harness, small
redistributable assets, and evidence needed to understand the demonstrated
workflow.

Examples are not authoring scaffolds. Use `templates/tasks/` to create a new
task, and use an example to study a finished integration. Do not copy an
example's domain assumptions, metrics, resource profile, or promotion policy
into an unrelated task.

## Available Examples

- [`rocket_booster_recovery/`](rocket_booster_recovery/README.md): a complete
  classical-control research project for rocket-booster first-contact recovery,
  with frozen simulation assets, two task harnesses, measured server evidence,
  portable tests, and explicit platform-evidence boundaries.
- [`rocket_booster_recovery_rust/`](rocket_booster_recovery_rust/README.md): an
  independent native Rust implementation of the same research problem, with
  vendored offline dependencies and server, Linux, and Apple Silicon task
  profiles.

Treat the bundled trees as read-only. Run `praxist examples list`, install the
selected example, and execute tests, evaluators, or research only from the
printed external working directory. Praxist preserves existing working copies
during upgrades.
