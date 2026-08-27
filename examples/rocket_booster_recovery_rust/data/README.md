# Evaluation data

The NPZ banks and `controller.json` are unchanged assets from the frozen
Python/JAX reference snapshot used for the port. Their SHA-256 values are
defined in `manifest.json` and enforced by the Rust evaluator.

`complete_source_rows_le_i32.bin` stores the exact 12,288 NumPy-selected row
indices in source order: 4,096 nominal-unseen, 4,096 near-OOD, then 4,096
hard-OOD. It is little-endian signed `i32`; its generation protocol and hashes
are recorded in `complete_source_rows.json`.

The row list is data, not executable Python state. Committing it makes formal
selection version-independent and keeps the runtime entirely Rust.
