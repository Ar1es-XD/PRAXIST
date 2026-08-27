# Harness Template

Replace this directory with task-owned benchmark, simulation, or execution code
before running real research.

Agents should call the public evaluator declared in `task.yaml`, not internal
harness files directly.

Probe the platform and execution backend before generating resource handoff
logic. If the task explicitly selects the compatible Praxist-managed NVIDIA/CUDA
backend, preserve `PRAXIST_ASSIGNED_GPU_UUIDS` exactly in `CUDA_VISIBLE_DEVICES` and
`NVIDIA_VISIBLE_DEVICES` through evaluator, trainer, worker, shell, and
container launches. A local device ordinal is valid inside that mask but must
not replace it for descendants. Add focused contract tests for the selected
backend and complete child chain. CPU-only execution, unified-memory systems,
and other accelerators are equally valid and must not be forced through CUDA or
UUID checks.
