# Platform Support

## Release Validation Matrix

| Operator host | Release status | Required verification |
|---|---|---|
| Linux on CPython 3.11 or 3.12 | Continuously tested by release CI | Run `praxist doctor` for runtime and provider readiness |
| macOS on CPython 3.11+ | Package and CLI compatibility target; not continuously tested by release CI | Run `praxist doctor` and validate all task-owned dependencies |
| Linux on other CPython 3.11+ versions | Package compatibility target; not continuously tested by release CI | Run `praxist doctor` and a task-specific smoke test |
| Windows-native | Outside the current research-runtime contract | Use a supported Linux environment instead |

The package metadata accepts CPython 3.11 or newer, but that compatibility
range must not be read as a claim that every interpreter, operating-system and
hardware combination has passed release CI. Codex or Claude Code must already
be installed and usable for skill-driven operation. Headless Linux and remote
shells are normal environments.

## Research Hardware

Praxist does not require a particular accelerator. CPU-only systems, macOS
unified memory, NVIDIA/CUDA, task-managed accelerators, and other task-owned
backends are valid when the research project itself supports them.

Task initialization observes the unchanged baseline on the current host before
declaring resource behavior. It must not infer a platform from product names,
compare artificial CPU-only and accelerator-only rewrites, or invent an
accelerator profile.

The central scheduler manages only resource classes explicitly represented by
the task harness. Its full contract is documented in
[Central Experiment Scheduler](../guides/central-resource-scheduler.md).

## Out of Scope

The Praxist package and setup wizard do not provide:

- GPU drivers or CUDA;
- model-training frameworks;
- datasets or simulators;
- task-specific containers;
- cluster schedulers.

Those belong to the research project or host administrator.

`praxist resolve` can still parse task and plugin configuration without loading
POSIX locking at module import time. `praxist doctor` reports unsupported native
platforms before a research launch, and an attempted central-scheduler run fails
with a direct platform message rather than silently weakening host locking.

## Filesystem and Process Expectations

Praxist supports ordinary local paths and symlinked task or experiment storage.
The operator must have permission to create task run directories, user
configuration and registry state. The selected Python environment must be
writable by its owner.

Daemonized runs are independent of the agent conversation that launched them.
The live monitor is a separate foreground process and `Ctrl-C` exits only that
monitor.
