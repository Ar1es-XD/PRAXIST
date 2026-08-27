# Legacy Migration Guide

Legacy migration is allowed only when it preserves behavior while moving code
toward the current core-plugin-task boundaries.

## Migration Pattern

1. Characterize the old behavior with tests.
2. Add the new protocol, plugin, or task boundary.
3. Route execution through the new boundary.
4. Add old-vs-new parity checks.
5. Preserve partial outputs and weak provenance when needed.
6. Delete the old path after the new path is proven.
7. Record migration context in the commit or pull-request description when the
   migration changes an architecture contract.

## What Not To Preserve

Do not preserve obsolete package names, duplicate task catalogs, shell-owned
semantics, fake production plugins, or hidden SAM-specific global defaults.

Do not keep compatibility shims after they stop serving an active migration.

## Migration Tests

Use characterization tests for old behavior, unit tests for the new interface,
workflow smoke tests for run shape, and replay tests for artifact consistency.

Long real GPU/API dogfood runs are manual/on-demand gates, not default unit
tests.
