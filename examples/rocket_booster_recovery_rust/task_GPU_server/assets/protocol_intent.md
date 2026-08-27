# Evaluator protocol intent

| Mode | Launch | Rank | Mature | Parent | Normal close | Units |
|---|---:|---:|---:|---:|---:|---:|
| `canary` | yes | no | no | no | no | 1 development trajectory |
| `development` | yes | yes, within-stage only | no | no | no | 2,048 public development trajectories |
| `roll_diagnostic` | yes | no | no | no | no | 1,024 frozen roll cases |
| `complete` | yes | yes | yes | yes | yes | 12,288 landing + 1,024 roll |

Every landing mode uses the same `landing_success_pass` predicate. There is no
engineering/strict/standard family of alternative success definitions.
Overall, source-stratified, hard-OOD, and initial-radius rates are aggregations
of that single array.

The public evaluator, candidate interface, plant execution, data loading, and
metric aggregation are implemented in Rust for CPU execution. A candidate has
root `controller.rs`, `controller_config.json`, and `variant.json` files and may
contain candidate-local `.rs` modules. The frozen launcher hashes, statically
inspects, stages, and compiles the entire source tree without permitting custom
module paths, extra Cargo dependencies, Python/JAX helpers, GPU backends, or
runtime file/network/process access. New mechanism parameters belong under
`variant_params` through the frozen extensible configuration adapters; they do
not require candidates to redeclare all baseline fields.

The endpoint is interpolated first landing-leg contact before spring/damper
response. Success requires lateral error <=5 m, COM and lowest-leg downward
first-contact speeds <=1 m/s, lateral speed <=0.3 m/s, tilt <=1.5 deg,
roll/pitch-yaw rate limits, finite state, and remaining main fuel strictly >2%
of the fixed 7000 kg capacity. There is no lower impact-speed bound: a softer
powered contact is not penalized.

Post-contact scored steps are structurally zero. A high-speed impact cannot be
laundered into success by gear damping. Deliberate pre-contact engine-cut or
low-thrust-drop strategies whose safety depends on suspension energy absorption
are forbidden even if they improve an unscored COM-ground terminal state.

Complete clean results emit the shared source label `performance`; Praxist's
confirmed and incubator selectors decide durable membership independently. The
incubator is parent-capable only for complete, contract-clean evidence and
retains Pareto/new-high points across distinct outcome, vertical-impact risk,
2% fuel feasibility, actuator-load, and roll-robustness families.

For a complete, contract-clean result, `promotion_eligible=true` means the
evidence is eligible for Praxist durable routing and normal generation close;
it is deliberately independent of measured landing performance. The stricter
task-owned `confirmed_performance_gate_passed` requires nonzero overall,
hard-OOD, and worst-radius-bin success. Confirmed selection requires that gate,
while the incubator may retain a complete Pareto/new-high result that has not
yet crossed it. Performance failure must never make a valid complete result
invisible to maturity accounting.

The complete near/hard rows are disjoint from both the 2,048 development set
and the repository's other baseline-validation fixtures. No external historical
run, result, controller, or score is admissible research evidence. Every
candidate must pass the frozen research-independence manifest check, and PI/
Chair review must invalidate candidates when session evidence contradicts the
self-attestation.

First contact is observable from frozen gear geometry. Post-contact 3-5 s
dwell, bounce, loads, slip, and overturn are intentionally unscored and must
not be claimed.

## Result ownership contract

Every centrally scheduled evaluation uses the core-recognized
`--output-dir` option, an explicit immutable-task-root cwd, and a unique
`results/gen_<N>/<peer_id>/<variant_id>/<mode>` output root. Canonical summaries
also record matching `peer_id`, `generation_id`, and `source_generation_id`
when executed inside a peer runtime. These operational identity fields do not
change the scientific protocol or any metric; they prevent one peer's broad
working directory from claiming another peer's evidence.
