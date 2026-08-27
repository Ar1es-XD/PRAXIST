# Rust harness regression fixtures

All regression checks run through Cargo; this directory contains no Python
tests.

```bash
cargo test --release --locked --offline --manifest-path task_linux/Cargo.toml

cargo run --release --locked --offline \
  --manifest-path task_linux/Cargo.toml \
  --bin rocket-booster-recovery-task-eval -- \
  --variant-dir task_linux/assets/baseline \
  --mode complete \
  --output-dir task_linux/scratch/regression-complete \
  --threads 16
```

The tests cover candidate static isolation, mass/time-step locks, the single
first-contact success gate, disabled action channels, complete 13,312-unit
maturity fields, separation of durable-promotion and confirmed-performance
semantics, effective-configuration identity, and the Rust baseline reference
result of 495/12,288. They also cover typed and dynamic `variant_params`
deserialization, multi-file source-tree audits, rejection of custom `#[path]`,
and direct Serde-derive compilation in generated candidate crates.
