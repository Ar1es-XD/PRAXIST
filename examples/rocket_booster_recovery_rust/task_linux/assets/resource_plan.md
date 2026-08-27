# Rust Linux CPU resource plan

## Reference measurement and portability boundary

The unchanged complete baseline was measured on 2026-08-24 on an x86_64 Linux
CPU server with an Intel Xeon Platinum 8457C. With 16 Rayon workers, 12,288
landing units plus 1,024 roll units took about 1.996 seconds inside the scoring
process and reported 12,892 KiB peak RSS. A warm outer Cargo launch in that
workspace took about seven seconds. These are reference measurements, not a
performance promise for every Linux host.

A later 15-minute observation of the full 16-peer Rust research process on the
same server found approximately 6.2 CPU cores used on average, 7.4 cores at
P95, short peaks near 22.6 cores, and about 7.3--7.7 GiB process-tree RSS. No
task GPU process or GPU utilization was observed. The run was normally limited
by agent/research cadence rather than evaluator throughput.

On a new Linux CPU or architecture, run the frozen complete baseline before
research and record the local `target_arch`, `target_os`, Rust version, worker
count, score, wall time, and memory. Do not relabel the Xeon record as a local
measurement.

## Portable runtime

- Native Rust 1.85 or newer; `cargo` must resolve through `PATH`.
- Supported intent: native x86_64 or aarch64 Linux, CPU only.
- Frozen crates are stored in repository-root `vendor/`; Cargo stays
  `--locked --offline` during research.
- No task Python, JAX, CUDA, GPU, or venv is required. Praxist itself remains an
  operator-installed orchestration dependency.
- Each evaluator uses 16 Rayon workers. Each peer may have at most one active
  evaluator.

## Conservative scheduler envelope

- Profile: `cpu_evaluation`; pressure domains are CPU, memory, and I/O.
- Initial/minimum/maximum concurrent evaluators: 2 / 1 / 2.
- This portable default avoids assuming a high-core server. After a local
  one-way/two-way complete-evaluation qualification, a high-core Linux server
  may raise the cap in its own deployment profile without changing science.
- Supply lease: 600 seconds; mature fraction: 0.25; mature redundancy: 3.0;
  one exploration slot remains reserved.
- Complete scoring is expected to be far below one minute after a warm build;
  the task launch guard retains a conservative 0.45-minute estimate for build
  locks and host jitter.
- Synthesis remains fixed at 90 minutes. Gen0 DIG occurs before the peer
  research timer. The two-hour peer budget leaves 30 minutes for draining,
  panel synthesis, and cleanup.

## Storage and qualification

`scratch/candidate_builds` stores content-addressed generated runners and
`scratch/cargo-target` reuses build products. A multi-generation run can retain
large peer workspaces; reserve at least 50 GiB free storage for a 30-generation
campaign.

Before raising concurrency, compare one and two simultaneous complete baseline
evaluations. Require identical protocol digests and landing counts, no Cargo
lock failures, no memory/I/O pressure, and higher total throughput. If the
two-way test is unstable or slower, use 1 / 1 / 1.
