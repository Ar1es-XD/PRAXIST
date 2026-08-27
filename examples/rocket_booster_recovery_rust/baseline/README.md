# Committed evidence

- `evaluation_summary.json`: formal Rust CPU evaluation produced by the current
  source tree with automatic CPU parallelism.
- `python_jax_reference_evaluation_summary.json`: unchanged source-repository
  reference used for outcome and metric comparison.
- `benchmark_server_2026-08-24.json`: raw timing and memory observations from
  the conversion host.

Regenerate the Rust record with:

```bash
cargo build --release
./target/release/rocket-booster-recovery-rust --mode complete --threads 0 \
  --output-dir results/reference
```

The command must report `integrity_passed = true` and exactly 495 successes.
