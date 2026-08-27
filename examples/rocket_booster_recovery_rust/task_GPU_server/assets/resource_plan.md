# Rust CPU-server resource plan

## Measured platform

Measurements were taken on 2026-08-24 on an Intel Xeon Platinum 8457C
server with 168 logical CPUs visible to the process. With 16 Rayon workers,
the Rust evaluator completed 12,288 landing units and 1,024 roll units in
about 1.97 seconds of scoring-process wall time, with about 12.9 MiB recorded
peak RSS. With candidate and dependencies warm, outer Cargo startup took about
7 seconds. The first release build in a fresh checkout is slower and depends
on disk and Rust-cache performance.

These measurements describe only this CPU server and are not a performance
promise for another server or personal computer. GPU utilization,
accelerator-memory use, and CUDA dependencies are zero; this task has no GPU
backend.

## Scheduler envelope

- Profile: `cpu_evaluation`; pressure domains: CPU, memory, and I/O.
- Each evaluator defaults to 16 rollout workers; each peer may run at most one
  evaluator concurrently.
- Central scheduler initial/minimum/maximum concurrency is 8 / 1 / 8, for a
  planned peak of 128 rollout workers.
- Supply lease: 600 s; mature fraction: 0.25; mature redundancy: 3.0; one
  exploration slot is reserved per generation.
- Complete scoring itself takes far less than a minute. The launch guard keeps
  a conservative 0.45-minute estimate for candidate compilation, Cargo locks,
  and shared-host jitter.
- Earliest and latest synthesis boundaries are both 90 minutes, aligned with
  the validated Python harness. The full research window remains even when
  mature evidence arrives early, so quick replication and local tuning do not
  displace complex mechanism work.
- Gen0 DIG runs before the formal peer timer and is excluded from the
  90-minute synthesis interval.
- Each generation gives a peer 2.0 hours, leaving 30 minutes after the
  90-minute synthesis boundary for drain, PI panel, and cleanup rather than
  colliding peer timeout with orchestrator closing.
- The scheduler resolves `cargo` from the operator's `PATH`. `vendor/` and
  `Cargo.lock` pin dependencies, two infrastructure retries are allowed, and
  the task stores no machine-specific absolute path.

## Storage and build cache

`scratch/candidate_builds` stores generated runners isolated by the complete
Rust source-tree SHA-256. `scratch/cargo-target` reuses frozen-dependency build
artifacts. Both are reproducible runtime caches ignored by Git. Formal results
must be written to the peer-exclusive
`results/gen_<N>/<peer_id>/<variant_id>/<mode>` directory.
