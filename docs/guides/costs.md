# Cost Estimation

Praxist can spend tokens, task-selected compute time, wall-clock time, tool quota,
and external API quota. Cost estimates are advisory, not a replacement for
BudgetPolicy.

## Inputs To Estimate

Estimate cost from:

- cohort size;
- number of generations;
- model profile per stage;
- expected peer session count;
- expected Principal Investigator (PI) and Chair planning calls;
- prompt size and cache stability;
- tool calls;
- evaluation runtime;
- platform/backend capacity and queue behavior, including accelerator capacity
  only when the task actually uses one.

## Prompt Cache Readiness

PromptLayout V1 keeps frozen and dynamic prompt blocks separate. Stable frozen
prefixes improve the chance that caches managed by the agent runtime or API
provider are useful.

Agent runtime cache behavior is currently treated as runtime-managed. Praxist records
layout hashes and cache provenance rather than injecting raw cache directives
where the runtime does not expose them.

## Interpreting Usage

Exact token or cache usage depends on what the agent runtime/API provider returns. Missing
metering should be recorded as unknown rather than zero.

Use run artifacts, budget ledgers, and API provider invoices together when analyzing
cost after a dogfood run.

For runtimes that report cache usage, `input_tokens` is the inclusive logical
input total and `cached_input_tokens` is the cache-read subset. Calculate:

```text
uncached_input_tokens = input_tokens - cached_input_tokens - cache_creation_input_tokens
cache_hit_ratio = cached_input_tokens / input_tokens
sessions_per_peer_generation = peer session count / peer-generation count
```

Treat an unreported cache-creation value as zero. If the reported components do
not fit inside inclusive input, keep the raw values and mark them inconsistent
instead of forcing the equation to balance.

Claude SDK telemetry additionally preserves cache-creation input separately.
Its API provider's native `input_tokens` value is uncached input, so the adapter
normalizes inclusive input as uncached + cache read + cache creation. Historical
or third-party records whose cached input exceeds their declared total are kept
unchanged and marked `telemetry_inconsistent`; Praxist does not clamp them or
publish a misleading cache-hit ratio.

Treat these as separate signals. Prompt caching can reduce billed compute while
logical input remains high; reducing unnecessary fresh sessions reduces both
repeated tool work and logical input. The read-only diagnostic inventory derives
these values from canonical `generation_results.json` rows and the run summary;
it does not create another usage ledger.

Session-reuse and tool-output mechanisms are defined in
[Cost Optimization](cost-optimization.md); this page only defines how to
estimate and interpret cost.
