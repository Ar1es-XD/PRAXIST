# Architecture and CPU execution model

## Design objective

The port preserves the behavior of the frozen Rocket Booster Recovery baseline while replacing
an array-program/JIT execution model with a native CPU program. The main
constraint is not merely producing a similar score: the same 12,288 source
rows, physical equations, contact endpoint, success gate, and audit contracts
must remain attributable and reproducible.

## Data flow

```text
fixed NPZ banks + hashed row selection
                  |
                  v
        Vec<[f32; 16]> initial states
                  |
                  v
      Rayon indexed trajectory workers
                  |
       +----------+-----------+
       | one serial trajectory|
       | controller -> action |
       | frozen plant RK4     |
       | contact interpolation|
       +----------+-----------+
                  |
                  v
 ordered first-contact endpoints + audits
                  |
                  v
 f64 metrics, stratification, integrity checks
                  |
                  v
 evaluation_summary.json (+ optional CSV)
```

## Why trajectory-level parallelism

There are three potential levels of parallel work:

- trajectories;
- time steps within a trajectory;
- RK stages within a time step.

Only trajectories are independent. A controller memory update at step `k`
feeds step `k+1`, and each RK stage evaluates a state constructed from the
preceding stage. Parallelizing either of those levels would change the
algorithm. Rayon distributes complete trajectories instead. Its indexed
parallel iterator returns outputs in input order, so worker scheduling cannot
alter source alignment or aggregate results.

The inner loop uses `[f32; N]` values and hand-written 2D/3D operations. It has
no dynamic matrix allocation, virtual dispatch, locks, or shared mutable state.
Dynamic arrays are used at the dataset and report boundaries.

## Numerical choices

- Controller and plant state arithmetic remains IEEE-754 `f32`, matching JAX's
  traced state dtype.
- Aggregate metrics use `f64`, matching the NumPy scorer's conversions.
- Quantiles use NumPy's default linear interpolation definition.
- Quaternion normalization deliberately preserves the controller/plant
  distinction in the source: controller normalization adds `1e-8` inside the
  square root, whereas plant normalization clamps the ordinary norm.
- RK4 stages, quaternion integration, weighted thrust, fuel burn, and final
  state assembly retain source operation ordering where Rust permits.
- The plant boundary clips every channel and writes RCS pitch, RCS yaw, and
  grid-fin roll to exact positive zero a second time.

Rust and CUDA do not guarantee bit-identical implementations of transcendental
functions such as `sin`, `cos`, and `tanh`. The formal Boolean result is exact;
continuous parity is quantified in `NUMERICAL_PARITY.md`.

## Dataset selection

The source project used NumPy PCG64 permutations to select formal rows. Runtime
reimplementation of NumPy's RNG would add unrelated code and a version-sensitive
dependency. The exact 12,288 selected row indices are therefore committed as
little-endian `i32` values in `data/complete_source_rows_le_i32.bin`.

Both hashes are checked:

- binary SHA-256:
  `4f7ab2fcdefdf7adfcb6f9a4528e9eb53019a4323e0a606f68e5ba98505b80bc`;
- canonical row-list SHA-256:
  `2694073419a116a12c611f8e952cf399b0a2172ea0c93aa8c6bac36ceee9b78c`.

This keeps RNG work outside the runtime while retaining the committed row
selection.

## Dependency boundary

The runtime dependencies are Rust crates:

- `clap` for command-line parsing;
- `rayon` for CPU trajectory scheduling;
- `ndarray` and `ndarray-npy` only for reading fixed NPZ arrays;
- `serde`/`serde_json` for configuration and reports;
- `sha2`/`hex` for integrity attestation;
- `anyhow` for error context.

No crate binds to Python, JAX, CUDA, a GPU driver, or a native BLAS library.
