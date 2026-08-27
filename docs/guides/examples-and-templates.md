---
description: Choose a Praxist task scaffold or study a complete runnable reference project.
---

# Examples And Templates

Praxist ships two kinds of task-oriented assets. Their purposes are deliberately
different.

| Asset | Use it when | What it contains |
|---|---|---|
| Template | You are creating or testing a new task project | Replaceable scaffolding, placeholders, and deterministic smoke fixtures |
| Example | You want to inspect a finished integration | A complete task harness, evaluator, task-specific code, redistributable assets, evidence, and tests |

Templates under `templates/tasks/` are meant to be copied and adapted. Their
defaults illustrate contract shape; they are not universal scientific policy
and may intentionally omit datasets or production evaluators.

Examples under `examples/` are runnable reference projects. They retain their
own domain assumptions, metrics, resource profiles, and evidence boundaries.
Those choices demonstrate one project and must not become Praxist-wide defaults
or be copied uncritically into another field.

Both remain outside the `praxist` system package. Praxist core and generic
plugins contain no facts from either asset. Run lightweight checks in place,
but copy a template or example outside the Praxist checkout before starting a
research run so generated artifacts remain external to product source.

## Materialize A Complete Example

First-use setup creates writable copies under
`${PRAXIST_EXAMPLES_HOME:-~/PraxistExamples}`. Inspect or recreate them with:

```bash
praxist examples list
praxist examples install rocket_booster_recovery
praxist examples install rocket_booster_recovery_rust
```

Pass `--destination /absolute/path` to install one example elsewhere. Existing
destinations are preserved unless the operator explicitly chooses another
path.

## Choose A Starting Point

- Start with a [task template](../reference/task-templates.md) when authoring a
  new task contract.
- Study [Rocket Booster Recovery](../examples/rocket-booster-recovery.md) for a
  Python/JAX integration or [Rocket Booster Recovery
  (Rust)](../examples/rocket-booster-recovery-rust.md) for an offline native
  Rust integration of the same research problem.
- Read [Task Projects](task-projects.md) for the canonical task contract shared
  by both.
