# 7,000 kg First-Contact Baseline

`rocket_booster_recovery_v2_first_contact_7000kg_baseline` was first measured on
2026-08-22 and completely remeasured on 2026-08-23 on a server with
**8x NVIDIA H100 80GB HBM3** accelerators. The evaluator process used one H100. The
real complete protocol contains 12,288 landing trajectories and 1,024 frozen roll
disturbances, for 13,312 evaluation units. The authoritative raw result is
`assets/baselines/baseline_evaluation_summary.json`; the second complete measurement
is `assets/baselines/baseline_remeasurement_20260823.json`. Every scientific metric
was identical, including 495 successes from 12,288 landing trajectories.

No score, wall time, throughput, or resource utilization has been measured on a
personal computer. The values below belong only to the 8x H100 server measurement
and must not be represented as results from an RTX laptop, Apple Silicon device, or
another PC.

The task has one success definition. At the interpolated instant of first landing-gear
contact, lateral error must be no greater than 5 m; center-of-mass and lowest-leg sink
speed must be no greater than 1 m/s; lateral speed must be no greater than 0.3 m/s;
the attitude and angular-rate gates must pass; and main-fuel reserve must be strictly
greater than 2% of the initial 7,000 kg. Spring and damping response after contact is
not evaluated.

## Success And Strata

- Joint landing success rate: 0.040283203125 (495/12,288).
- 95% Wilson interval: 0.036947959247708835 to 0.043905789326106996.
- Nominal-unseen / near-OOD / hard-OOD: 0.056640625 / 0.064208984375 / 0.0.
- Worst-radius-stratum success rate: 0.0; all three strata beyond 900 m are 0.
- First-contact rate: 1.0.

## Key Continuous Metrics

- First-contact joint vertical-gate pass rate: 0.11417643229166667.
- First-contact COM sink-speed p50 / p95 / p99: 43.668596 / 66.392846 /
  67.399636 m/s.
- Lowest-leg sink-speed p95: 66.398738 m/s.
- Lateral-speed p95: 1.405921 m/s; lateral-error p95: 7.092907 m.
- Tilt p95: 4.560622 degrees; combined pitch/yaw rate p95: 0.012761 rad/s.
- Two-percent fuel-component gate pass rate: 0.07584635416666667.
- Mean / p05 fuel-reserve fraction: 0.004733027367364793 / 0.0.
- Main-fuel depletion rate: 0.8863932291666666.
- Minimum fuel-reserve fraction among successful trajectories:
  0.020097935267857144.

## Actuators, Roll, And Contracts

- Grid / gimbal / throttle saturation step rates: 0.5115840435 / 0.0000950214 /
  0.0005514988.
- Frozen roll-disturbance stability rate: 1.0; settling-time p95: 2.5 s.
- Landing-gear damping-credit rate and post-contact scored steps: 0 / 0.
- Forbidden action, plant lateral RCS, and non-finite trajectory values: all 0.

These are independent performance metrics and Pareto dimensions, not additional
success definitions. Component gate rates diagnose why the one joint success
predicate fails.
