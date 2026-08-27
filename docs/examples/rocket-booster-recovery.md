---
description: Explore the complete Rocket Booster Recovery Praxist reference project.
---

# Rocket Booster Recovery

Rocket Booster Recovery is a complete classical-control example with a frozen
six-degree-of-freedom plant, deterministic data banks, a first-contact landing
evaluator, task-local research roles, and preserved baseline evidence. It shows
how a real research project and its Praxist task harness fit together without
moving domain logic into the framework.

The source checkout contains the example at:

```text
examples/rocket_booster_recovery/
```

A Praxist wheel exposes the same tree as the package resource
`praxist/resources/examples/rocket_booster_recovery/`.

## What It Demonstrates

- a complete task harness rather than a scaffold;
- one public evaluator backed by a frozen simulation boundary;
- task-owned metrics, maturity rules, frontier lanes, roles, and audit rules;
- measured evidence kept separate from unmeasured platform claims;
- small frozen data banks and manifests protected by checksums;
- lightweight contract tests that do not launch a research run.

The controller design, physical constraints, metrics, and hardware profiles are
specific to this example. They are not Praxist defaults. See
[Examples And Templates](../guides/examples-and-templates.md) before adapting
any part of it.

## Inspect And Test

The source and package-resource copies are inspection-only. Materialize a
writable project before creating an environment or running any code:

```bash
praxist examples install rocket_booster_recovery
cd ~/PraxistExamples/rocket_booster_recovery
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./scripts/run_tests.sh
```

The project README owns the scientific protocol, environment details, baseline
measurements, and evaluator commands.

## Start A Research Run

The first `praxist setup` after pip installation creates a writable copy at
`~/PraxistExamples/rocket_booster_recovery` and prints the absolute path.
Recreate it explicitly or choose another destination with:

```bash
praxist examples install rocket_booster_recovery
praxist examples install rocket_booster_recovery \
  --destination /path/to/rocket_booster_recovery
```

An existing destination is preserved without replacement. Prepare that working
copy's environment, then choose the task harness that matches the host:

| Harness | Intended host | Evidence status |
|---|---|---|
| `task_GPU_server` | A compatible accelerator server | Includes the project's measured server baseline |
| `task_PC` | A workstation or laptop | Requires a baseline measured on that machine before research |

```bash
cd ~/PraxistExamples/rocket_booster_recovery

praxist resolve "$PWD/task_GPU_server" --run-dir "$(mktemp -d)"
praxist start --task-path "$PWD/task_GPU_server" --daemonize --json
```

The bundled server evidence must not be relabeled as evidence from another
machine. The project README owns hardware details and scientific protocol.
