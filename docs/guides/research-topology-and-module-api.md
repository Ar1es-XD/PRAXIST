# Research Topology Audit API

Praxist records the executable research topology for each generation and
exposes a structured, read-mostly API for external modules. These surfaces are
audit and integration boundaries; they do not replace findings, frontier,
Gems, memory, or result artifacts.

## Executed Topology

The bundled executor runs one parallel cohort per generation:

```text
generation -> peer cohort -> findings -> generation boundary
           -> frontier / Gems / memory / PI and Chair planning
```

Before the cohort starts, Praxist writes:

```text
gen_<N>/research_topology.json
```

The sidecar contains a `ResearchTopologySpec` with worker nodes, edges, policy,
and metadata. It records what the executor is about to run. Generic worker
types in the schema are descriptive vocabulary; the bundled executor does not
claim to execute a worker type unless that node appears in the materialized
topology.

## Module API

`ResearchLoopModuleAPI` provides these operations:

```text
submit_recommendation
request_topology_change
list_commands
list_findings
get_frontier_summary
get_validation_signals
get_gems_summary
get_memory_summary
get_run_status
```

`get_frontier_summary` returns durable frontier evidence.
`get_validation_signals` returns compact task-defined signals for triage and
follow-up planning. Validation signals retain their actual stage and coverage;
they do not become clean parents unless the task contract explicitly grants
that authority.

## Command Queue

Recommendations and topology-change requests are appended to:

```text
external_requests/research_commands.jsonl
```

The bundled executor records these commands but does not inject them into peer
prompts or mutate a live topology. A queued command is operator intent, not an
experiment contract. PI and Chair planning continue to own executable peer
assignments.

```text
external module -> command queue -> audit and operator review
```

External modules should use this API instead of editing prompts, agendas, or
generation artifacts directly.

## Boundary Rules

- `GenerationLoop` owns lifecycle, resume, and generation boundaries.
- The topology executor owns cohort execution behind that boundary.
- Topology policy and worker adapters belong in plugins, not task-specific core
  branches.
- Requests use the current `queue_for_generation_boundary` policy.
- Queued requests have no scientific or promotion authority by themselves.
