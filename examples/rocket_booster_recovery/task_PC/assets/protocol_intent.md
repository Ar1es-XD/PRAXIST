# Evaluation Protocol Intent

The frozen evaluator exposes several execution modes, but only one mode can
create durable scientific evidence:

| Mode | Purpose | Comparable for durable ranking | Mature / parent / generation close |
|---|---|---:|---:|
| `canary` | Wiring and contract smoke test | No | No |
| `development` | Cheap hypothesis prioritization | No | No |
| `roll_diagnostic` | Isolated roll-controller diagnostics | No | No |
| `complete` | Frozen 13,312-unit formal evaluation | Yes | Yes |

A protocol-clean `complete` result is mature evidence. Admission to the
confirmed frontier additionally requires `confirmed_performance_gate_passed`;
the incubator separately retains task-justified Pareto trade-offs.

Candidate provenance is also frozen. A candidate may begin only from the
shipped baseline or a canonical parent produced by the current run. Earlier
runs, packaged champions, sibling checkouts, historical scores, and Git
history are outside the evidence boundary and must not be inspected or reused.

All modes use the same immutable plant, integrator, contact model, source banks,
actuator locks, 7,000 kg initial fuel, and first-contact success predicate.
Canary, development, roll-only, partial, suspect, failed, and late results
cannot parent or close a generation.

