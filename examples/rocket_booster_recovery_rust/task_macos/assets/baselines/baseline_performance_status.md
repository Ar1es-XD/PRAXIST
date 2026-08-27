# Baseline performance status — macOS

Status: configuration complete; native Apple Silicon measurement pending.

- Variant: `rocket_booster_recovery_rust_v2_first_contact_7000kg_baseline`
- Runtime: Rust, CPU-only, 12 Rayon workers for the Mac profile
- Target: native `aarch64-apple-darwin`
- GPU / Metal / CUDA / task venv requirement: none
- Protocol: `rocket_booster_recovery_first_contact_7000kg_validation_v2`
- Required units: 12,288 landing + 1,024 roll = 13,312
- Mac landing success, wall time, RSS, and safe concurrency: `null`
- Linux/x86_64 reference: 495 / 12,288, stored separately and not reused as
  Mac evidence
- Post-contact scoring: disabled; gear damping earns no credit
- Initial fuel: 7000 kg; initial/dry mass: 29200 / 22200 kg

Run the complete baseline on the target Mac before interpreting performance
deltas or populating `task.yaml:baselines`. Preserve the generated
`target_arch`, `target_os`, protocol hashes, source-row digest, and all metrics.
If the native score differs from the Linux reference, keep both measurements
and investigate floating-point/target effects instead of overwriting history.
