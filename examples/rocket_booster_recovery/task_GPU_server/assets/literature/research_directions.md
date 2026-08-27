# Operator-provided research directions: 7000 kg first-contact protocol

This pack is task context supplied by the human expert. Network search and the
Praxist literature-lookup tool are deliberately disabled for this run. The
links below were not fetched or independently verified during initialization;
they are hypotheses/prior-art pointers, not measured task performance.

## Frozen scientific boundary

- No neural network, reinforcement learning, learned residual, policy network,
  or online learning.
- Allowed: analytic guidance; PID/PD; LQR/LQG; H-infinity; MPC; QP/SOCP/SCP;
  EKF/UKF; disturbance observer; reference/constraint governor; deterministic
  design search.
- `RCS_x` is roll-only. `RCS_y`, `RCS_z`, and `grid_roll` are exact `+0.0` at
  controller and plant boundaries. Pitch/yaw use only gimbal/grid y-z.
- RCS fuel is uncharged, but torque, slew, switching, coupling, and numerical
  limits are evaluated.
- Plant, RK4 integrator, contact model, evaluator, and data are immutable.
- Initial main propellant is fixed at 7000 kg (29200 kg total initial mass).
- There is one landing-success predicate only. It is evaluated at interpolated
  first leg contact with lateral error <=5 m, first-contact COM/leg sink speed
  <=1 m/s, the existing lateral-speed/upright/rate limits, and fuel >2%.

## Prohibited terminal mechanism

Do not cut the engine or command a deliberately under-supported descent so the
vehicle hits the legs at unsafe speed and lets the gear damper create a benign
later COM-ground velocity. Do not optimize contact spring/damper absorption,
post-contact bounce, or COM-terminal state as a surrogate for landing. The
evaluator gives exactly zero scored steps after first contact, so such a
mechanism is both scientifically forbidden and incapable of earning success.

Permitted terminal designs keep a powered, closed-loop height-velocity corridor
through first contact. Ballistic/coast segments are allowed only when the
controller demonstrably brakes to the first-contact speed gate before a leg
touches; they may not rely on suspension damping.

## Ten attributable research lines

1. `energy_manager`: braking distance, bang-bang structure, ignition root solve,
   powered first-contact vertical corridor, and a >2% touchdown-mass safety cage.
2. `trajectory_guidance`: 1–2 Hz warm-started SCP/SOCP/QP-MPC, with analytic
   energy fallback on solver failure; output references, not actuators.
3. `attitude_controller_yz`: SO(3) PD/LQR/LQG/loop shaping scheduled by mass,
   inertia, thrust, dynamic pressure, and phase; never output RCS.
4. `allocator_yz`: constrained weighted least-squares/QP TVC-grid allocation,
   respecting amplitude/rate/effectiveness and preserving gimbal translation
   margin; no RCS or grid roll in the y-z allocation matrix.
5. `fin_effectiveness_model`: deterministic physical/Jacobian lookup or online
   finite-difference identification of the frozen plant's grid-fin authority.
6. `state_disturbance_estimator`: observable EKF/UKF/ESO/DOB estimates of wind,
   drag, thrust scale, and mass bias; avoid jointly unidentifiable states.
7. `roll_rcs_controller`: roll-rate PD with deadband/hysteresis, PWPF, or
   constrained bang-bang; prioritize stability and low switching.
8. `constraint_governor`: feasibility filter after nominal guidance and before
   attitude/allocation; modify infeasible references only.
9. `terminal_landing_manager`: near-ground powered velocity corridor, upright
   freeze, smooth throttle, and first-contact state logic. A low-thrust
   gear-sink/drop phase that depends on damper arrest is prohibited.
10. `robust_design_validation`: Sobol/Latin-hypercube sensitivity,
    deterministic direct/SQP search, counterexample discovery, and numerical
    sensitivity; produces finite auditable parameters, not a policy.

## Integration order and gates

Integrate only after single-module ablation:
`(1+9) -> (2+8) -> (5+4) -> 3 -> 6 -> 7 -> 10`.

Promotion is hierarchical: channel/numeric/anti-damping contract; nonzero
single-gate landing success; hard-OOD improvement; no severe radius-bin
regression; then compare first-contact vertical risk, 2% fuel-gate coverage,
actuator load, and roll robustness. Do not invent a second success standard.

## User-provided prior-art pointers

- NASA fuel-optimal descent analysis:
  https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20100025515.pdf
- Ignition-point reachability: https://arxiv.org/pdf/2503.11862v1
- 6DoF successive convexification: https://arxiv.org/pdf/1811.10803v1.pdf
- NASA SLS flight control and robustness:
  https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20140007339.pdf and
  https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20150005708.pdf
- NASA generic control allocation:
  https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20190000993.pdf
- DLR grid-fin study: https://elib.dlr.de/210892/
- NASA navigation-filter practices:
  https://ntrs.nasa.gov/api/citations/20205000801/downloads/NESC%20Technical%20Bulletin%2020-03%2C%20Navigation%20Filter%20Design%20Best%20Practices.pdf?attachment=true
- NASA switching RCS control:
  https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20160001840.pdf
- Explicit reference governor and thrust pointing:
  https://arxiv.org/pdf/1905.00387v1 and
  https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20130009793.pdf
- DLR touchdown experiment: https://elib.dlr.de/215847/
- NASA Monte Carlo and robust descent:
  https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20100038453.pdf and
  https://ntrs.nasa.gov/citations/20220010431
