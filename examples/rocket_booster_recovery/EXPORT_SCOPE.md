# Example Distribution Scope

This example contains a runnable 7,000 kg initial-fuel-load controller project
and two Praxist task profiles. The task evaluates first-leg-tip contact and one
joint success gate. Runtime research outputs are deliberately outside the
distributed source tree.

## Included

- Task contracts, prompts, roles, audit rules, evaluators, and task-owned
  assets for `task_GPU_server/` and `task_PC/`.
- The frozen baseline controller, configuration, and variant inputs used by
  both task profiles.
- Measured server baseline evidence under
  `task_GPU_server/assets/baselines/`.
- Explicitly unmeasured personal-computer records with `null` performance
  values under `task_PC/assets/baselines/`.
- The plant, adapter, source data banks, development data, configurations,
  tests, and reproducible Python dependency list.

## Excluded

- Research runs, peer workspaces, findings, frontier state, PI artifacts,
  Gems, logs, scheduler state, trajectories, plots, and generated reports.
- Virtual environments, package caches, bytecode, temporary files, and debug
  output.
- Standalone candidate packages or generated dependency distributions.
- API keys, provider credentials, saved agent login state, and access tokens.

Praxist creates runtime output directories in the installed writable example,
not in the package source. Server measurements must not be represented as
personal-computer measurements. The PC profile requires complete-protocol
remeasurement on its target machine before its performance can be compared.

## Ownership

All source code, task harnesses, configurations, and data artifacts included in
this example were developed in-house for Praxist and are distributed under the
repository root `LICENSE.md`. External runtime dependencies retain their own
licenses and are not vendored by this example.
