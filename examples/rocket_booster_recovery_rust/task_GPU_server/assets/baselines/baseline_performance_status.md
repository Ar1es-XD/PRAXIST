# Baseline performance status

Status: measured on the declared Rust CPU-server platform with the complete
protocol.

- Variant: `rocket_booster_recovery_rust_v2_first_contact_7000kg_baseline`
- Runtime: Rust, CPU-only, 16 Rayon workers
- Measurement host: Intel Xeon Platinum 8457C; 168 logical CPUs visible
- GPU / CUDA / venv requirement: none
- Other server and personal-computer platforms: not measured
- Protocol: `rocket_booster_recovery_first_contact_7000kg_validation_v2`
- Evaluation units: 13,312 / 13,312
- Single landing success: 495 / 12,288 = 0.040283203125
- Contract lock and protocol integrity: passed
- Research-independence manifest attestation: passed; `suspect_leakage=false`
- Post-contact scoring: disabled (0 steps, 0 damping credit)
- Initial fuel: 7000 kg; initial/dry mass: 29200 / 22200 kg
- Canonical summary: `assets/baselines/baseline_evaluation_summary.json`
- Complete remeasurement: `assets/baselines/baseline_remeasurement_20260825.json`
- Scoring-process wall time / peak RSS: 1.996 s / 12,892 KiB
- Warm outer Cargo launch observed in this workspace: about 7 s

The baseline is durable-evidence and normal-close eligible because its complete
protocol is contract-clean (`promotion_eligible=true`). It does not pass the
stricter confirmed-performance gate because hard-OOD and the worst radius bin
remain zero (`confirmed_performance_gate_passed=false`). Its nonzero overall
success and continuous metrics provide an optimization and incubator reference;
it is not recorded as a confirmed champion.
