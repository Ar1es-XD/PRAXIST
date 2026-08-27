# 7,000 kg First-Contact PC Baseline (Unmeasured)

The controller and complete protocol for
`rocket_booster_recovery_v2_first_contact_7000kg_baseline` are present, but they have
not been executed on a personal computer. The planned measurement contains 12,288
landing trajectories and 1,024 frozen roll disturbances, for 13,312 evaluation
units. The placeholder record is
`assets/baselines/baseline_evaluation_summary.json`.

The task has one success definition. At the interpolated instant of first landing-gear
contact, lateral error must be no greater than 5 m; center-of-mass and lowest-leg sink
speed must be no greater than 1 m/s; lateral speed must be no greater than 0.3 m/s;
the attitude and angular-rate gates must pass; and main-fuel reserve must be strictly
greater than 2% of the initial 7,000 kg. Spring and damping response after contact is
not evaluated.

## Success And Strata

- Joint landing success rate: `null`.
- 95% Wilson interval: `null`.
- Nominal-unseen / near-OOD / hard-OOD: `null`.
- Worst-radius-stratum success rate: `null`.
- First-contact rate: `null`.

## Key Continuous Metrics

- First-contact joint vertical-gate pass rate: `null`.
- First-contact COM sink-speed p50 / p95 / p99: `null`.
- Lowest-leg sink-speed p95: `null`.
- Lateral-speed and lateral-error p95: `null`.
- Tilt and combined pitch/yaw-rate p95: `null`.
- Two-percent fuel-component gate pass rate: `null`.
- Mean / p05 fuel-reserve fraction: `null`.
- Main-fuel depletion rate: `null`.
- Minimum fuel-reserve fraction among successful trajectories: `null`.

## Actuators, Roll, And Contracts

- Grid / gimbal / throttle saturation step rates: `null`.
- Frozen roll-disturbance stability rate and settling-time p95: `null`.
- Landing-gear damping-credit rate and post-contact scored steps: `null`.
- Forbidden action, plant lateral RCS, and non-finite trajectory values: `null`.

These are independent performance metrics and Pareto dimensions, not additional
success definitions. Component gate rates diagnose why the one joint success
predicate fails. The 8x H100 server result was not reused as a PC placeholder.
