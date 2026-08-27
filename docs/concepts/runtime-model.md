# Runtime Model

This page explains the research loop at the agent-session boundary. It does not
define API provider routing, resource scheduling, evidence policy, or budgets;
those contracts have dedicated guides.

## Agent Sessions

A peer opens an agent session with a rendered prompt layout and a normalized
runtime request. The request identifies the model profile, credential reference,
tools, sandbox intent, cache policy, and timeout. The selected runtime turns SDK
events into the common Praxist event/result protocol.

After a session returns, the peer waits for a meaningful event instead of
immediately resending the same context. Examples include a revised shared
finding, lifecycle signal, resource-supply event, or heartbeat expiry. Event
cadence and continuation behavior are runtime policies; canonical findings and
task state remain on disk.

[Agent Runtimes](../guides/agent-runtimes.md) owns adapter behavior and
[Cost Optimization](../guides/cost-optimization.md) owns lossless finding-event
batching.

## Prompt Layout

PromptLayout V1 separates:

- **frozen blocks** shared across compatible calls;
- **semi-static blocks** changed by task or role contracts; and
- **dynamic blocks** containing generation, frontier, memory, and run state.

The rendered prompt and its layout manifest are audit snapshots. They preserve
what the agent saw for replay, but later stages regenerate views from current
canonical result, finding, frontier, Gems, memory, and boundary state.

Peer memory is a navigation index into those artifacts, not a replacement for
them. [Peer Memory](../guides/peer-local-structured-memory-long-context.md)
defines its lifecycle.

## Related Contracts

- [Central Experiment Scheduler](../guides/central-resource-scheduler.md):
  experiment admission, process ownership, and resource release.
- [Budget Policies](../guides/budget-policies.md): budget decisions and usage
  records.
- [Cost Estimation](../guides/costs.md): interpretation of measured usage.
- [Research Loop](../guides/research-loop-variant-generation-flow.md): full
  generation sequence and artifact inheritance.
