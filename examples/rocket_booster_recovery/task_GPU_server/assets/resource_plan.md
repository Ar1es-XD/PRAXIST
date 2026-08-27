# Baseline Resource Plan

Every measurement in this file comes from a server with
**8x NVIDIA H100 80GB HBM3** accelerators. One baseline evaluator used one H100.
Wall time, throughput, RAM, accelerator memory, and bottlenecks have not been measured
on a personal computer and must not be extrapolated as measured results.

## Runtime Envelope

- Python 3.11; JAX/JAXLIB 0.9.2; CUDA 12 backend.
- One GPU per evaluator, with declared demand of 2 GiB accelerator memory and 100%
  accelerator utilization.
- Initial/minimum/maximum central-scheduler concurrency: 8 / 1 / 8; cohort: 16.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`; normal peers do not inherit accelerator
  visibility.
- Complete protocol: 12,288 landing units plus 1,024 roll units.

One complete baseline evaluator took approximately 25.39 seconds on the original H100
host, with approximately 1.20 GiB peak process-tree RSS and 643 MiB peak board memory
on the selected GPU. This is a historical capacity observation, not a performance
commitment for another machine.

## Scheduler Policy

- Profile: `gpu_evaluation`; exactly one GPU per job; no automatic CPU fallback.
- Supply lease: 600 s; mature fraction: 0.25; mature redundancy: 3.0.
- One exploration slot remains available in every generation.
- Synthesis opens and closes at 90 minutes; adaptive early close is disabled so a
  few fast complete evaluations cannot truncate the generation.
- Complete budget record: 0.0075 GPU-hours; launch safety factor: 1.5.

Scientific baseline evidence is in
`assets/baselines/baseline_evaluation_summary.json`. Runtime results belong under the
ignored `experiments/` directory.
