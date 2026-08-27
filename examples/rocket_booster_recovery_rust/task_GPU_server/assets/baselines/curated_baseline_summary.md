# 7,000 kg Rust First-Contact Baseline

`rocket_booster_recovery_rust_v2_first_contact_7000kg_baseline` was measured
on 2026-08-24 on an Intel Xeon Platinum 8457C CPU server using the Rust
`complete` protocol: 12,288 landing trajectories plus 1,024 frozen roll
disturbances, for 13,312 evaluation units. The evaluator used 16 Rayon
workers. Results are in `baseline_evaluation_summary.json`; measurement
details are in `baseline_remeasurement_20260825.json`.

The task has one success definition. At interpolated first landing-gear
contact, lateral error must be at most 5 m, center-of-mass and lowest-leg sink
speed at most 1 m/s, lateral speed at most 0.3 m/s, tilt and angular rates
within their gates, and main-fuel reserve strictly greater than 2% of the
initial 7,000 kg. Post-contact spring and damper response is not evaluated.

## Success And Strata

- Landing success rate: 0.040283203125 (495/12,288)
- 95% Wilson interval: 0.036947959247708835–0.043905789326106996
- Nominal-unseen / near-OOD / hard-OOD: 0.056640625 / 0.064208984375 / 0.0
- Worst-radius-stratum success rate: 0.0; all three strata beyond 900 m are 0
- First-contact rate: 1.0

## Key Continuous Metrics

- First-contact joint vertical-gate pass rate: 0.11417643229166667
- First-contact COM sink-speed p50 / p95 / p99: 43.668594 / 66.392846 /
  67.399624 m/s
- Lowest-leg sink-speed p95: 66.398733 m/s
- Lateral-speed p95: 1.405924 m/s; lateral-error p95: 7.092804 m
- Tilt p95: 4.560631 degrees; pitch/yaw rate p95: 0.0127614 rad/s
- 2% fuel-gate pass rate: 0.07584635416666667
- Mean / p05 fuel-reserve fraction: 0.00473303897040231 / 0.0
- Main-fuel depletion rate: 0.8863932291666666
- Minimum fuel-reserve fraction among successful trajectories:
  0.020097935267857144

## Actuation, Roll, Resources, And Contract

- Grid / gimbal / throttle saturated-step rate: 0.5115837315 /
  0.00009502136 / 0.00055160889
- Frozen roll-disturbance stability rate: 1.0; settling-time p95: 2.5 s
- Landing-gear damping-credit rate and post-contact scored steps: 0 / 0
- Forbidden actions, plant lateral RCS, and non-finite trajectories: all 0
- 16-worker scoring-process wall time: 1.996 s; recorded peak RSS: 12,892 KiB
- GPU, accelerator memory, CUDA, and Python/JAX runtime: not required
- Research-independence declaration: passed; `suspect_leakage=false`

These are independent performance metrics and Pareto dimensions, not
additional success definitions. Component-gate pass rates diagnose failures
of the joint success predicate. Hardware other than the stated CPU server has
not been measured.
