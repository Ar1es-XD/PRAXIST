# Cost Optimization

This guide describes low-risk token optimizations that preserve research facts
while reducing repeated context inflation.

## Goals

Praxist cost optimization follows the result-preservation principle:

- keep peer outputs and raw evidence available;
- return compact summaries by default;
- make full data available through explicit lookup;
- avoid task-specific logic in core;
- avoid asking agents to re-read large logs or broad JSON files when a small
  summary is enough.

## Lossless Session Efficiency By API Provider

Praxist automatically enables lossless session efficiency for two expensive
routes:

- Codex-native mode (`agent_runtime:codex_sdk` with native OpenAI and a saved
  ChatGPT login);
- `model_provider:openrouter` on any supported runtime.

Direct `model_provider:deepseek_alias` runs are always excluded. Their event
cadence, memory limits, and prompts remain unchanged, even if an operator sets
the lossless override.

The policy does not compress history or remove findings. It changes how peers
consume the same canonical artifacts:

1. stop, closing, and resource-supply events remain immediate;
2. individual shared-finding events within a short interval are collected and
   followed by one continuation session;
3. the next prompt carries a larger bounded batch of unseen finding IDs plus
   the existing peer state and handoff;
4. the complete task prompt remains present;
5. the continuation is told to use exact references first and reopen original
   artifacts whenever details are needed or uncertain.

An already-consumed finding suppresses a duplicate wake only when both its
explicit identity and content version are unchanged. A corrected or expanded
payload with the same identity is surfaced again; missing or unparseable
identity fails open and wakes the peer.

The canonical findings, full tool outputs, session logs, and task documents
remain on disk. This is event coalescing and reference-first navigation, not
context compression.

Configuration:

```bash
# Default: auto-detect Codex-native mode and OpenRouter routes.
export PRAXIST_CONTEXT_EFFICIENCY_MODE=auto

# Change the finding-only batching interval (default 300 seconds).
export PRAXIST_CONTEXT_EFFICIENCY_MIN_SESSION_INTERVAL_SECONDS=300

# Disable finding batching for a comparison run.
export PRAXIST_CONTEXT_EFFICIENCY_MODE=off
```

`lossless` explicitly enables the policy for a non-DeepSeek route. Unknown
mode values fall back to `auto` rather than blocking a run.

When OpenRouter is selected with `agent_runtime:codex_sdk`, its existing private
relay additionally receives a non-secret, run-scoped `session_id`. This provides
sticky routing for prompt-cache locality. Praxist does not enable OpenRouter
response caching: research replies and tool calls must not be replayed verbatim.
See the
[OpenRouter prompt-caching guide](https://openrouter.ai/docs/guides/best-practices/prompt-caching)
and [response-caching distinction](https://openrouter.ai/docs/guides/features/response-caching).

Native OpenAI prompt caching remains automatic. Cache hits require exact prefix
matches, so stable instructions should precede dynamic generation/session
content. See the
[OpenAI prompt-caching guide](https://developers.openai.com/api/docs/guides/prompt-caching).

## Tool Output Limits And Full Lookup

Problem: MCP tools such as leaderboard, frontier, and finding-graph queries can
return large JSON payloads. Even when an agent only needs the top few records,
the whole response is fed back into the LLM context.

Design:

- tool handlers keep their existing business fields, such as `entries`,
  `pareto_front`, `neighbor_findings`, `nodes`, and `edges`;
- handlers attach `_tool_output` metadata with:
  - `schema_version`;
  - `tool_name`;
  - `view = summary`;
  - `truncated` and `truncated_lists`;
  - `full_result_ref`;
  - the follow-up tool name;
- the complete JSON payload is written under the active run directory:
  `tool_results/*.json`;
- agents use `mcp__evaluation-tools__read_tool_result` to read bounded chunks
  of a stored result by `offset` and `max_chars`.

This gives Praxist character-window pagination without requiring a separate
cursor protocol for every tool:

```text
summary tool response -> _tool_output.full_result_ref -> read_tool_result(ref, offset, max_chars)
```

The mechanism is lossless for system-generated tool data: inline summaries may
be truncated, but the full JSON remains in the run artifacts.

Current scope:

- `evaluation-tools.get_leaderboard`;
- `frontier-tools.get_frontier`;
- `finding-graph-query.get_finding_neighbors`;
- `finding-graph-query.get_finding_subgraph`;
- `finding-graph-query.get_unlinked_recent_findings`;
- `evaluation-tools.read_tool_result`.

Out of scope:

- built-in agent CLI `Bash`, `Read`, and shell command output. Praxist cannot
  hard-cap tools built into the agent runtime from a generic MCP tool server.
  Prompts should still instruct agents to avoid `cat` on large JSON/log files
  and to prefer compact Praxist tool responses.

## Task-Local Evaluation Toolification

Problem: expensive tasks can give peers long prompt instructions for manually
launching staged benchmarks, waiting for files, parsing benchmark JSON, applying
gates, and deciding whether to escalate. Agents often copy commands, inspect
large raw JSON files, or repeat shell checks.

Design:

- keep task-specific evaluation mechanics inside the external task project;
- provide one compact public evaluator entrypoint under the task's
  `evaluations/` directory;
- keep Praxist core and generic plugins unchanged;
- have the peer implement a variant, then run one task-local command;
- preserve raw benchmark JSON and logs under the run results directory;
- print only a compact gate summary to stdout.

Example command shape:

```bash
python "$PRAXIST_TASK_PROJECT_PATH/evaluations/<task_evaluator>/run.py" \
  --variant-path "<variant.py>" \
  --output-dir "<run-results-dir>/<experiment-name>" \
  --data-dir "<task-data-dir>" \
  --max-stage "<task-defined-stage>"
```

The task-local tool owns benchmark invocation, promotion gates, raw evidence
preservation, concise stdout summaries, and exact maturity telemetry such as
`effort_ratio` and `coverage_ratio`. For smoke runs, expose a task-owned
argument or environment variable that limits the evaluator stage without
changing Praxist core behavior.

## Measure The Effect

Savings depend on runtime caching, provider metering, evaluator output size,
event cadence, and agent behavior. Praxist does not promise a fixed token or
billing reduction.

Compare equivalent runs using canonical usage artifacts. Record input, cached
input, uncached input, output, session count, sessions per peer-generation, and
cache-hit ratio. Also verify task correctness and artifact recoverability; a
lower token count is not useful if peers repeat work or miss evidence.

Use `praxist-diagnostic` to report input, cached input, uncached input, output,
session count, sessions per peer-generation, and cache-hit ratio. A high cache
hit rate can coexist with excessive logical token use when every new thread
repeats broad bootstrap reads.

## Testing Expectations

Changes in this area should include:

- unit tests for result-ref storage, path confinement, chunk reads, and list
  truncation metadata;
- tool adapter tests proving original business fields still exist;
- task-local tests for evaluator gates and reuse-existing summaries;
- offline integration tests proving startup exposes the declared tool names;
- API provider gating tests proving direct DeepSeek behavior is unchanged;
- event tests proving finding bursts coalesce while lifecycle/resource events
  remain immediate;
- relay tests proving only OpenRouter receives sticky-session metadata;
- docs build after guide or docstring changes.

Do not test exact prompt prose or temporary output ordering. Test stable
contracts: bounded inline output, full-result recoverability, and task-local
gate decisions.
