# Tool Servers

Tool servers are generic plugins that expose bounded capabilities to peers and
panels. Task projects decide when a capability is scientifically relevant;
task-specific instructions do not belong in the server.

## Catalog

| Tool server | Purpose |
|---|---|
| `evaluation_tools` | Compact evaluator and leaderboard access |
| `frontier_tools` | Committed frontier views |
| `memory_tools` | Peer-memory lookup |
| `finding_graph_query` | Advisory finding-graph queries |
| `prior_work_tools` | Existing-work lookup |
| `run_report` | Human-readable derived reports |
| `literature_lookup` | Public scientific literature/database context |
| `existing_mcp_tools_shim` | Compatibility bridge for declared external tools |

The finding graph is built by a graph-maintainer plugin; its tool server is only
the query surface.

## Specialized Contracts

[Scientific Literature and Database Lookup](scientific-literature-lookup.md)
owns source coverage, provenance, current-environment limits, and runtime
behavior for `literature_lookup`.

[Run Reports](user-facing-reports-and-init.md) owns automatic/manual report
triggers, structure, metric interpretation, and canonical-state boundaries for
`run_report`.

## Frontier Views

`frontier_tools.get_frontier` reads committed membership from
`frontier/frontier_manifest.json`; it does not rerun promotion. The effective
task-spec maturity policy remains available for interpretation, but a live task
file cannot replace the run's frozen policy.

The latest-generation view uses current compact lanes. Historical cutoffs are
reconstructed from the canonical per-generation ledger so later capacity
eviction does not rewrite earlier membership. Compact state without immutable
history, such as unverifiable historical Gems membership, is omitted and marked
incomplete rather than guessed.

The response reports canonical and returned counts, categorized skips, policy
source, and `frontier_view_integrity_status`. If view construction hides every
entry from a non-empty canonical lane, it returns
`canonical_entries_hidden` as an integrity error. Validation candidates remain
separate and are never promoted by the reader.

## Tool Conformance

Tool-server tests cover manifest resolution, handler construction, allowed tool
names, bounded normalized output, failure normalization, redaction, and missing
optional credentials. Full payload recovery and inline output limits are
defined in [Cost Optimization](cost-optimization.md#tool-output-limits-and-full-lookup).
