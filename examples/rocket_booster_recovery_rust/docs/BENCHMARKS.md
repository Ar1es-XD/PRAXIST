# CPU benchmark report

## Host and protocol

- Date: 2026-08-24 UTC
- CPU: Intel Xeon Platinum 8457C
- Logical CPUs visible to the process: 168
- Dataset: complete 12,288-trajectory protocol
- Integration: RK4, one substep, `dt = 0.1 s`, up to 900 steps
- Python stack: Python 3.11.15, JAX/JAXLIB 0.9.2, NumPy 2.4.6
- Rust: stable 1.98.0, release profile with thin LTO and one codegen unit
- All compared executions produced 495 successes.

## CPU-to-CPU result

| Runtime | Parallel setting | Total evaluator time | Rollout/block context | Peak RSS |
|---|---:|---:|---:|---:|
| Python/JAX CPU | `TFRT_CPU_0` | 23.1795 s | first block 4.5873 s; later blocks 1.6241–1.6472 s | 822,320 KiB |
| Rust CPU | 16 Rayon workers | about 1.9 s | allocation-free native rollout | under 15 MiB |
| Rust CPU | 168 Rayon workers | about 0.35 s | native rollout about 0.29 s | under 15 MiB |

The measured speedups are about 12x at 16 Rust workers and 67x with all 168
visible workers. Peak memory was roughly two orders of magnitude lower. The
Rust result includes asset hashing, NPZ loading, rollout, metric reduction, and
JSON generation, but not `cargo build`; the JAX result includes its first-call
JIT compilation. For a long-lived warmed JAX process, use the later-block times
rather than the first block. Rust has no runtime compilation or warm-up phase.

## Development-set thread scaling

The 2,048-row development evaluation produced 57 successes at every setting:

| Rust workers | Total time | Rollout time |
|---:|---:|---:|
| 1 | 4.869 s | 4.859 s |
| 2 | 2.482 s | 2.473 s |
| 4 | 1.233 s | 1.223 s |
| 8 | 0.641 s | 0.631 s |
| 16 | 0.327 s | 0.317 s |
| 32 | 0.168 s | 0.158 s |
| 64 | 0.096 s | 0.085 s |

The measurements show near-linear scaling over this worker range on the
measured host. Smaller consumer CPUs can use the default visible-core count or
an explicit number that leaves capacity for other work.

## Commands

Rust:

```bash
cargo build --release
./target/release/rocket-booster-recovery-rust --mode complete --threads 16 \
  --output-dir results/benchmark-16t
./target/release/rocket-booster-recovery-rust --mode complete --threads 0 \
  --output-dir results/benchmark-auto
```

The Python/JAX timing was captured in the frozen conversion environment with
the equivalent command below. That reference environment is not required or
included in this Rust repository; this command is provenance, not a setup
instruction for this project:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
  python -m rocket_booster_recovery.evaluate --mode complete \
  --output-dir /tmp/rocket-booster-recovery-rust-python-cpu --batch-size 1024
```

## Interpretation limits

- The host is a large server, not a laptop; absolute times will not transfer.
- Worker counts above the physical-core count can behave differently on other
  CPUs and under contention.
- CPU frequency, NUMA placement, page cache, compiler version, and thermal
  state were not pinned.
- The Rust runs were faster than the Python/JAX CPU run on the measured host.
  The result is not a cross-machine performance estimate.
