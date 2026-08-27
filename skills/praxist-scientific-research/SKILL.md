---
name: praxist-scientific-research
description: Gather task-agnostic scientific research context for Praxist task projects using no-key public literature/database/open-access lookup, agent-host web search when available, local project documents, and source/provenance notes. Use when an agent needs to identify domain metrics, benchmarks, prior art, scientific databases, open-access provenance, high-value research directions, or literature-backed hypotheses for a Praxist task without starting a run, changing Praxist core logic, or treating literature as measured task performance.
---

# Praxist Scientific Research

Use this skill to build a source-backed research context for a Praxist task
project. This is a planning and context-gathering skill, not an experiment
runner.

## Boundaries

- Do not start, stop, resume, or mutate a Praxist run.
- Do not edit Praxist core or generic plugins.
- Do not require additional API keys. Prefer public no-key sources and Codex web
  search when available.
- Do not treat papers, database entries, blog posts, benchmark pages, or
  open-access text as measured task performance.
- Do not download datasets, checkpoints, simulators, packages, licensed assets,
  or new runtime environments just because a source mentions them. Use those
  sources to improve the best solution possible under the task's current local
  data, simulator, dependencies, evaluator, hardware, and runtime. Record
  missing resources only as task-local notes or user-facing requirements.
- Keep domain-specific search policy and conclusions in the task project,
  usually under `assets/literature/`, `description.md`, role skills, or task
  prompt files.

## Source Priority

1. Local project docs, papers, README files, benchmark docs, existing logs, and
   task-owned `assets/literature/`.
2. Public no-key literature/database tools when `tool_server:literature_lookup`
   is available in the standard or task-declared Praxist tool set.
3. Agent-host web search/open web browsing for official benchmark pages, dataset
   cards, standards, methods, and recent papers.
4. Optional credentialed or licensed sources only when the user explicitly
   provides them and the task requires them.

## Workflow

1. Identify the task or research project root. Prefer the current directory when
   the user does not provide a path.
2. Read local task/project material first:
   - `task.yaml`, `description.md`, `README.md`, `prompt_task.jinja2`;
   - `assets/literature/`, dataset metadata, baseline summaries, result logs;
   - domain papers or PDFs supplied by the user.
3. Determine the domain and research objective. Keep categories broad enough to
   preserve diversity: machine learning, biology, chemistry, medicine,
   robotics/control, physics/materials, quantitative systems, software systems,
   or generic optimization.
4. If `tool_server:literature_lookup` is available, use:
   - `literature_source_guide(domain, objective)` to pick source families;
   - `literature_search(query, sources, max_results)` for compact public
     records;
   - `literature_resolve(identifier)` for DOI/PMID/arXiv/OpenAlex records.
   - `literature_open_access_text(identifier_or_url, max_chars)` when an
     open-access full text or PDF provenance record is needed;
   - `scientific_database_search(query, sources, max_results)` for public
     scientific databases such as Europe PMC, UniProt, or ClinicalTrials.gov.
5. If the tool is not available, use agent-host web search for official and primary
   sources where current information matters.
6. Produce compact notes with:
   - source title, URL or identifier, authors/organization, year/date;
   - what claim or design hint the source supports;
   - how reliable or directly applicable it is;
   - whether it affects metrics, benchmarks, constraints, high-value directions,
     or prior-art risk.
   - whether the source assumes resources not present locally; if yes, write a
     current-resource adaptation and a missing-resource note, not an install or
     download plan.
7. Save notes only when the user asked for task updates or task initialization
   is in progress. Recommended paths:
   - `assets/literature/research_directions.md`;
   - `assets/literature/source_notes.md`;
   - `assets/literature/prior_art_risks.md`.

## Output Shape

For direct analysis, report:

- recommended metrics and why;
- benchmark/evaluation standards and official sources;
- high-value research directions, grouped by mechanism family and intervention
  surface;
- prior-art risks and leakage risks;
- open-access/provenance notes for sources used in task prompts;
- open questions that should become Praxist validation candidates.

For task files, write concise task-owned text. Do not create duplicate
leaderboards, fake baselines, or Praxist runtime artifacts.

## Evidence Language

Use this distinction consistently:

- **Literature signal**: a paper, database record, benchmark page, or source
  note suggesting a hypothesis or constraint.
- **Open-access provenance**: retrieval URL, timestamp, content hash, and
  credential scope showing where a public source came from.
- **Measured task fact**: evaluator output or structured finding produced by
  the task harness during a Praxist run.

Only measured task facts should drive performance claims. Literature signals can
drive search directions and skepticism.
