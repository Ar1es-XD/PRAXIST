# Rocket Booster Recovery 7,000 kg First-Contact Classical Landing Research

Improve an auditable deterministic classical controller on the frozen Swordfish C05
6DoF plant. Initial main propellant is fixed at 7,000 kg. The sole primary metric is
`landing_success_rate` at first leg-tip contact. Success requires touchdown within
5 m, controlled first-contact center-of-mass and leg-tip impact speed, the established
attitude and angular-rate limits, and strictly more than 2% main propellant reserve.

Every stratified success rate reuses the same predicate. Selection remains
multi-objective: overall and hard-OOD coverage, first-contact vertical risk, the 2%
fuel-gate pass rate, grid-fin saturation, and roll-disturbance rejection remain
separate dimensions. The incubator retains Pareto or new-high candidates backed by
complete, protocol-clean evidence.

Scoring stops at first leg-tip contact. Spring, damping, rebound, and later
center-of-mass landing state do not count toward success. Deliberately striking the
landing gear at high speed and relying on damping is prohibited and cannot receive
evaluator credit.

Researchers may modify only `controller.py`, `controller_config.json`, and
`variant.json` in a candidate directory. The plant, RK4 integrator, contact model,
data, evaluator, task harness, and Praxist core are immutable.

## GPU Execution Contract

- A normal peer shell must not see or initialize a GPU by default. It must not clear,
  replace, or bypass task-injected `CUDA_VISIBLE_DEVICES` or
  `NVIDIA_VISIBLE_DEVICES` isolation.
- Every formal GPU evaluation must launch through the Praxist central scheduler and
  use only the assigned physical GPU UUID. Unscheduled GPU JAX scripts are
  prohibited.
- Candidate design, trajectory attribution, small-sample checks, and other ad hoc
  analyses must explicitly use CPU. Their results are diagnostic signals, not
  promotion-eligible evaluator evidence.
- `gpu_evaluation` reserves one GPU, 2 GiB accelerator memory, and 100% utilization
  per job. This envelope is based on the measured 643 MiB / 100% peak with memory
  margin, preserving at most one formal evaluator per GPU.
