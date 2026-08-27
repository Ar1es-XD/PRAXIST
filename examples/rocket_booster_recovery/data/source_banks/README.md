# Frozen Source Banks

Each bank contains a `float32` `state` array with shape `(40960, 16)`:

| File | Distribution | SHA-256 |
|---|---|---|
| `nominal_unseen_40960.npz` | nominal unseen | `674d119f8f0cd36c2553f0cc9134ec23886fd00d1a34396d515d957132311a3d` |
| `near_ood_easy_velocity_40960.npz` | near-OOD easy velocity | `d35803608dd3dedbbd4db76d9b39aebe37c2eedbdba4ad9ec0dcd94a690d7730` |
| `hard_ood_fast_outer_annulus_40960.npz` | hard-OOD fast outer annulus | `55d3e0054ef5d225b78f6558a4e15a3a908bedb84daabb7cc66b8e6579fd297c` |

The first-contact v2 complete protocol selects 4,096 rows from each bank with fixed
seeds, for 12,288 landing trajectories, plus 1,024 frozen roll disturbances. The
evaluator verifies these hashes and prohibits candidates from modifying data or
selecting initial states from outcomes.

`development_ood_2048.npz` is the public development set. Every path resolves relative
to the example root and does not depend on the original workspace.
