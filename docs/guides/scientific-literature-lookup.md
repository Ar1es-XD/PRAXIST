# Scientific Literature and Database Lookup

Praxist provides optional public literature, scientific-database, open-access,
and provenance lookup through `tool_server:literature_lookup`. It requires no
additional API key and does not change the automated experiment loop.

## Capabilities

The tool server exposes:

- `literature_search(query, sources, max_results)` for normalized cross-source
  search;
- `literature_resolve(identifier)` for DOI, PMID, arXiv, or OpenAlex work
  identifiers;
- `literature_open_access_text(identifier_or_url, max_chars)` for lawful open
  HTML/XML text or metadata, hash, and provenance for an open PDF;
- `scientific_database_search(query, sources, max_results)` for public
  scientific databases;
- `literature_source_guide(domain, objective)` for source-selection and
  verification guidance.

Public sources include arXiv, OpenAlex, PubMed metadata, Crossref, Semantic
Scholar metadata, Europe PMC, UniProt, and ClinicalTrials.gov. Individual
services may rate-limit or fail. Praxist reports per-source warnings instead of
failing an unrelated research run.

## Enable in a Task

The standard tool set includes the passive lookup server. A task descriptor
should list its complete active tool set so resolve and runtime selection agree:

```yaml
praxist_plugins:
  tools:
    - tool_server:evaluation_tools
    - tool_server:frontier_tools
    - tool_server:finding_graph_query
    - tool_server:memory_tools
    - tool_server:prior_work_tools
    - tool_server:run_report
    - tool_server:literature_lookup
```

The tool is passive: network access occurs only when an agent explicitly calls
it. A task may define a task-local literature role, but Praxist core must not
contain domain-specific search strategy. Principal Investigator (PI) memo
agents and peers with the tool perform lookup; the Chair planning agent
synthesizes the evidence provided to it.

## Current-Environment-Only Rule

Search results may mention datasets, checkpoints, simulators, dependencies,
licenses, APIs, or environments that are not available locally. During a run,
agents must not acquire or install those missing resources. They should extract
the useful scientific idea and adapt it to the task's existing data, evaluator,
dependencies, hardware, and runtime.

Missing resources may be recorded as task-local limitations or future
requirements. They are not permission to mutate the host.

## Evidence and Provenance

Literature is context, not measured task performance. Records should preserve:

- title, authors, year, venue, and stable identifier;
- source and retrieval time;
- open-access status;
- exact claim supported by the source;
- uncertainty, contradiction, or negative evidence;
- a pointer that lets a later agent retrieve the original record.

A negative lookup result is also evidence. Report the query, sources attempted,
warnings, and coverage limits. Do not rewrite "not found in searched sources"
as "does not exist."

## Runtime Behavior

The lookup server:

- uses bounded timeouts and result limits;
- normalizes records without claiming cross-source identity when uncertain;
- degrades one failing source without disabling other tools;
- does not bypass paywalls or authentication controls;
- does not download task data or install dependencies;
- keeps retrieved material separate from evaluator evidence.

Use `praxist-scientific-research` when an operator agent should gather task
context before a run. Use the tool server when a running peer or PI needs a
focused source check.
