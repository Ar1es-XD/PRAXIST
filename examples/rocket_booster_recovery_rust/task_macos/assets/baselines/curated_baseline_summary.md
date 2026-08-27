# 7,000 kg Rust First-Contact macOS Baseline (Unmeasured)

The controller, Rust simulator, and complete protocol for
`rocket_booster_recovery_rust_v2_first_contact_7000kg_baseline` are configured
but have not been executed on Apple Silicon macOS. The planned measurement is
12,288 landing trajectories and 1,024 frozen roll disturbances, for 13,312
evaluation units. The Mac placeholder is in
`baseline_evaluation_summary.json`; every performance value remains `null`.

The task has one success definition. At interpolated first landing-gear
contact, lateral error must be at most 5 m, center-of-mass and lowest-leg sink
speed at most 1 m/s, lateral speed at most 0.3 m/s, tilt at most 1.5 degrees,
roll rate at most 0.02 rad/s, pitch/yaw rate norm at most 0.03 rad/s, and
main-fuel reserve strictly greater than 2% of the initial 7,000 kg.
Post-contact spring and damper response is not evaluated.

## Metrics Awaiting Measurement

- Landing success rate and Wilson 95% interval: `null`
- Nominal-unseen / near-OOD / hard-OOD and radius strata: `null`
- First-contact vertical, leg-tip, and lateral speeds, error, and tilt
  quantiles: `null`
- Fuel gate, reserve, depletion rate, and minimum successful reserve: `null`
- Grid / gimbal / throttle saturation and action smoothness: `null`
- Roll stability, settling time, RCS switching, and coupling metrics: `null`
- Complete wall time, RSS, and one/two-evaluator throughput: `null`

The Linux/x86_64 Xeon result of `495 / 12,288` is retained only as an
independent reference in `baseline_reference_x86_64_linux.json` and
`reference_x86_64_linux_remeasurement_20260825.json`. It cannot support Mac
baseline promotion. The first Mac complete result must preserve the same
frozen protocol and identify its platform separately.
