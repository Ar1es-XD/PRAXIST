# Baseline performance status

Status: measured on the declared GPU-server platform, complete, protocol-clean,
launch-ready for that platform.

- Measurement platform: server with 8 × NVIDIA H100 80GB HBM3 GPUs
- GPUs used by this evaluator process: 1 × H100
- Personal-computer measurement status: **not measured**
- PC scores, wall time, throughput, memory pressure, and accelerator performance: **unknown**

- Variant: `rocket_booster_recovery_v2_first_contact_7000kg_baseline`
- Protocol: `rocket_booster_recovery_first_contact_7000kg_private_validation_v2`
- Evaluation units: 13,312 / 13,312
- Single landing success: 495 / 12,288 = 0.040283203125
- Contract lock: passed
- Protocol integrity: passed
- Post-contact scoring: disabled (0 steps, 0 damping credit)
- Initial fuel: 7000 kg; initial/dry mass: 29200 / 22200 kg
- Canonical summary: `assets/baselines/baseline_evaluation_summary.json`
- Complete remeasurement: `assets/baselines/baseline_remeasurement_20260823.json`
  (2026-08-23, 1 × H100, 25.0897 s); all recorded scientific metrics reproduced
  exactly, including 495 / 12,288 landing successes.

The baseline is deliberately not promotion-eligible because hard-OOD and the
worst radius bin remain zero. Its nonzero overall success and broad continuous
metric profile make it a useful, non-degenerate optimization reference.

Do not treat these values as measurements for an RTX 5090 laptop, an Apple
M5 Max system, or any other personal computer. Those platforms require their
own complete-protocol baseline record.
