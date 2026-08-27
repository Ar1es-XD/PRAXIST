# PC Baseline Resource Plan (Two-Way Sharing Unqualified)

## Measurement Status

- Target PC platform: `null`.
- Baseline wall time: `null`.
- Throughput: `null`.
- Process-tree peak RSS: `null`.
- Accelerator-memory or unified-memory pressure: `null`.
- Accelerator utilization: `null`.
- CPU pressure and bottleneck classification: `null`.
- Provisional concurrent evaluator ceiling: `2`.
- Measured safe concurrent evaluator count: `null`.

The scientific protocol remains aligned with the GPU-server task. The PC scheduler
uses a conservative single-GPU, two-way shared admission envelope. This is a launch
ceiling, not a measured throughput conclusion. Recalibrate it with single-versus-dual
A/B measurements on the unchanged baseline.

## Runtime Envelope

- Python 3.11; JAX/JAXLIB 0.9.2; CUDA 12 backend.
- Each evaluator uses one scheduler-assigned physical GPU UUID and declares 8 GiB
  memory plus 50% utilization demand. Two evaluations may be admitted on one UUID.
- Initial/minimum/maximum central-scheduler concurrency: 2 / 1 / 2; cohort: 16.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`; normal peers do not inherit accelerator
  visibility.
- Complete protocol: 12,288 landing units plus 1,024 roll units.

PC complete-evaluator wall time, peak process-tree RSS, and accelerator memory are
`null`. Historical capacity observations from the 8x H100 server were not copied.

## Scheduler Policy

- Profile: `gpu_evaluation`; exactly one scheduler-owned GPU UUID per job; no more
  than two allocations may share one UUID; no automatic CPU fallback.
- The 8 GiB / 50% values are admission reservations, not JAX hard limits. The
  scheduler also checks live memory and utilization and must queue and serialize work
  when the envelope does not fit.
- Supply lease: 600 s; mature fraction: 0.25; mature redundancy: 3.0.
- One exploration slot remains available in every generation.
- Synthesis opens and closes at 120 minutes; adaptive early close is disabled so a
  few fast complete evaluations cannot truncate the generation.
- The 150-minute per-generation limit reserves 30 minutes after the research window
  for evaluator drain and synthesis.
- PC complete GPU-hours: `null`. The concurrency values in `task.yaml` are an
  unqualified conservative startup envelope, not measured platform values.
- Launch safety factor 1.5 is a policy constant, not a PC performance measurement.

## Required Concurrency Qualification

The first target-platform qualification must run the same frozen candidate through
single- and dual-evaluator complete protocols and record:

- complete evaluators finished per hour;
- mean and P95 wall time per job, including cold-versus-warm compilation;
- peak GPU memory and utilization plus CPU, RAM, and I/O pressure;
- OOM, CUDA/XLA failure, and scheduler rejection counts;
- whether all 12,288 landing outcomes, auxiliary metrics, and protocol digests match
  the single-evaluator run.

Only if dual execution increases complete-evaluation throughput, preserves metrics,
and remains resource-stable may `2` become the verified safe concurrency. Otherwise,
restore all three concurrency values to `1 / 1 / 1`; never increase concurrency to
hide a bottleneck.

The placeholder PC record is
`assets/baselines/baseline_evaluation_summary.json`. Runtime results belong under the
ignored `experiments/` directory. Update baseline and resource records only after a
confirmed complete result and concurrency qualification.
