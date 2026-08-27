# Rocket Booster Recovery (Rust) 7000 kg first-contact classical landing research

Improve an auditable deterministic classical controller against the frozen,
pure-Rust CPU implementation of the Swordfish C05 6DoF plant. Initial main
propellant is fixed at 7,000 kg. The primary metric is
`landing_success_rate` at first landing-leg contact. Success requires touchdown
within 5 m, controlled first-contact vertical and leg-tip impact speeds, the
established lateral-speed, attitude, and angular-rate safety bounds, and
strictly more than 2% main propellant remaining.

Every stratified success rate reuses the same joint predicate. Selection
remains multi-objective: overall and hard-OOD coverage, first-contact vertical
risk, 2% fuel-gate pass rate, grid-fin saturation, and roll-disturbance
rejection remain separate axes. The incubator admits Pareto and new-high
candidates from complete, protocol-clean evidence.

Scoring stops at first leg contact. Post-contact springs, damping, rebound, and
center-of-mass landing state receive no success credit. A strategy cannot
deliberately strike the landing gear at high speed and rely on damping to earn
a successful evaluation.

Researchers may modify only `controller.rs`, `controller_config.json`,
`variant.json`, and candidate-local `.rs` modules. The plant, RK4 integrator,
contact model, data, evaluator, task harness, Cargo manifests, and Praxist core
are frozen. Candidates must pass source-tree static audit, content hashing,
and isolated compilation.

## CPU Execution Contract

- Every formal evaluation launches through the Praxist central scheduler's
  `cpu_evaluation` profile.
- Each evaluator defaults to 16 Rayon trajectory workers; one peer may run at
  most one evaluator concurrently.
- Server concurrency starts at 8, producing at most 128 planned rollout
  workers; the central scheduler reduces concurrency under CPU, memory, or I/O
  pressure.
- The evaluator and simulator have no GPU backend. Candidates may not add
  external dependencies, network access, or subprocesses.
- A candidate's first isolated build may incur one-time Cargo overhead;
  evidence records scoring wall time separately from build state.
