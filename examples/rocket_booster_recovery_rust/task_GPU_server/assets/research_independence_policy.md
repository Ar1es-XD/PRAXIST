# Research independence and fair-evolution policy

This task measures mechanisms discovered and tested inside one newly created
Praxist run. Generic classical-control prior art already frozen in
`assets/literature/research_directions.md` is admissible. Measured baseline
records are admissible only as the common starting condition.

## Admissible evidence

- Immutable files in this Rust checkout that define the baseline, candidate
  API, plant, evaluator, metric gates, and task protocol.
- The measured baseline ledger under `assets/baselines/`.
- Canonical notebook, findings, Frontier/incubator, PI packs, variants, and
  evaluator summaries created by the current run after its start boundary.
- The operator-provided, task-agnostic research directions frozen in this task.

## Forbidden prior-solution access

Researchers, PIs, the Chair, DIG, and QD must not search for, open, quote,
copy, translate, replay, or use any of the following:

- any earlier Praxist run, peer workspace, session transcript, finding,
  Frontier/incubator state, score table, report, or packaged champion;
- any controller implementation or configuration from another task,
  language, repository checkout, branch, tag, commit, archive, or export;
- any document that identifies a previously successful mechanism sequence,
  winning parameter values, candidate names, lineages, or measured score
  trajectory;
- filesystem paths outside this Rust checkout and the current run for the
  purpose of locating controller ideas or performance evidence;
- Git history, reflogs, remote branches, network services, or online search as
  a way to recover prior solutions or historical results.

The prohibition includes indirect use: asking another agent to inspect a
forbidden source, translating an older implementation into Rust, or using a
historical score to prioritize a mechanism is still prior-solution access.
Ordinary access to the configured Rust compiler, Cargo cache, and Praxist
runtime is allowed only for building and operating this task.

## Required candidate attestation

Every `variant.json` must preserve a `research_independence` object with all
four fields set to `false`:

```json
{
  "research_independence": {
    "prior_run_artifacts_accessed": false,
    "external_controller_implementation_accessed": false,
    "historical_performance_results_used": false,
    "copied_or_translated_prior_solution": false
  }
}
```

These values are a truthful provenance declaration, not a checkbox to copy
blindly. If any field would be true, the candidate is ineligible: stop work,
record the incident, and do not evaluate, publish, parent, synthesize, or
promote the candidate. The frozen evaluator rejects a missing or non-clean
attestation. PI and Chair reviews must also inspect session command/file-access
evidence; a contradictory trajectory overrides the self-attestation and makes
all downstream descendants suspect leakage.

This policy does not suppress negative results or collaboration inside the
current run. Peers should share current-run evidence normally so that later
generations can combine independently discovered findings.
