# Python/JAX to Rust numerical parity

## Reference

- Reference artifact: `baseline/python_jax_reference_evaluation_summary.json`
- Reference source: frozen Python/JAX conversion snapshot
- Python: 3.11.15
- JAX/JAXLIB: 0.9.2
- Formal rows: 12,288
- Rust arithmetic: fixed-size `f32` controller/plant and `f64` metric reduction

## Outcome parity

| Check | Result |
|---|---:|
| First-contact detection disagreements | 0 / 12,288 |
| Landing-success disagreements | 0 / 12,288 |
| Python/JAX successes | 495 |
| Rust successes | 495 |
| Nominal-unseen rate | exact match: 5.6640625% |
| Near-OOD rate | exact match: 6.4208984375% |
| Hard-OOD rate | exact match: 0.0% |
| Fuel-gate pass rate | exact match: 7.58463541667% |
| Vertical first-contact gate rate | exact match: 11.4176432292% |
| Forbidden-channel maximum | exact match: 0.0 |
| Nonfinite trajectory rate | exact match: 0.0 |

The Wilson interval and all success-related integer counts are also exact.

## Continuous metrics

Selected full-evaluation differences (`Rust - Python/JAX`) were:

| Metric | Absolute difference |
|---|---:|
| Fuel-reserve mean fraction | 1.16e-8 |
| Contact sink-speed P95 | 0.0 at reported precision |
| Contact lateral-error P95 | 1.02e-4 m |
| Contact lateral-speed P95 | 3.48e-6 m/s |
| Contact tilt P95 | 9.20e-6 deg |
| Pitch/yaw-rate P95 | 7.45e-8 rad/s |
| Grid saturation rate | 3.12e-7 |
| Mean grid total variation | 5.42e-5 rad |

For per-trajectory endpoint comparison:

- 95th percentile of the largest state-component absolute difference per row:
  `0.001953125`;
- 99th percentile: `0.00390625`;
- 95th percentile leg-sink absolute difference: `1.145e-5 m/s`;
- contact-step disagreements: 2 trajectories, each differing by one 0.1 s step.

Those two trajectories are high-speed failures close to the discrete
contact-crossing boundary. Their one-step difference creates the isolated
maximum velocity/sink delta, but neither trajectory approaches the success
envelope and no gate outcome changes. The cause is the expected last-bit
difference between Rust/libc and XLA/CUDA transcendental arithmetic accumulated
over hundreds of `f32` steps.

## Golden and formal tests

`tests/parity.rs` evaluates source row 530 and compares its complete
first-contact state against a Python/JAX golden endpoint. Its maximum component
difference must remain below `3e-5`; contact step and sink speed are checked
separately.

The formal test can be run with:

```bash
cargo test --release --test full_reference -- --ignored
```

It recomputes all 12,288 trajectories and asserts the overall and per-source
reference outcomes.
