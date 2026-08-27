# Deep Innovation Gate

The Deep Innovation Gate (DIG) is a deep-reasoning innovation process that
compares mechanism-level alternatives before implementation.

DIG is not an experiment loop. It does not train, evaluate, write variants, or
encode task metrics. It uses the selected Praxist runtime/API provider and the
task's existing prompt, baseline, and file boundaries.

## Generation Scope and Flow

The recommended profile runs DIG only before absolute generation zero. Later
generations use committed agendas from Principal Investigator (PI) agents and,
in multi-PI mode, a Chair; Gems resets do not reactivate DIG.

For an enabled generation:

1. build the normal peer context;
2. map the baseline and generate/critique candidate mechanisms with read-only
   planner tools;
3. validate one selected contract;
4. add that contract as a dynamic prompt block; and
5. launch the ordinary implementation peer.

The selected contract identifies the variant, mechanism, intervention surface,
rejected alternatives, planned files/changes, expected metric signature,
ablation hooks, and fail-fast checks. A material implementation deviation
requires an auditable `contract_amendment.yaml`.

## Artifacts

```text
gen_<N>/peers/<peer_id>/dig/
  baseline_mechanism_map.yaml
  candidate_pool.yaml
  candidate_reviews.yaml
  qd_selection.yaml
  selected_contract.yaml
  dig_summary.md
```

These are design/audit artifacts, never empirical findings. Result findings may
reference their metadata after evaluation. With `generation_scope:
initial_only`, their absence after generation zero is expected.

## Retry and Fallback

Malformed planner output or an invalid candidate/contract retries within the
configured attempt and total-time bounds. Valid phase checkpoints can be reused
only when prompt and artifact fingerprints still match.

After all attempts fail, Praxist writes `dig_failure_summary.json`. The default
then starts the ordinary implementation path, preserving liveness and an
audit trail. Strict tasks may disable fallback, accepting that one planning
failure can suppress a peer.

## Control Surface

Task initialization can enable or disable DIG independently, limit it to the
initial generation, bound planning time and candidate breadth, and choose
whether planner failure falls back to direct implementation. These controls
affect pre-code reasoning only; they do not change task evaluation or evidence.

## Validation

The gate requires enough mechanism and intervention diversity, critiques for
every candidate, at least one falsifying or diagnostic alternative, and a
complete selected contract. Evaluator, data split, and metric-calculation
changes are forbidden by default. Lane fit and duplicate checks apply only when
their corresponding task policies are active.

This validation predicts whether a plan is coherent. It never claims measured
performance or makes the contract a parent.

## Relationship to Quality-Diversity and Gems

DIG and Quality-Diversity (QD) have independent switches. At generation zero,
QD can allocate one validated candidate from each peer's own DIG pool.
Disabling that QD path leaves DIG's quality-first local selection intact.

Later QD uses existing PI/Chair proposals and does not call DIG or create DIG
artifacts. [Quality-Diversity Allocation](qdig-cohort-allocator.md) owns both
allocation paths and their failure behavior.

Gems runs after measured evidence reaches a generation boundary. DIG may read
Gems as lineage/duplicate context, but cannot create or promote a Gem. New tasks
normally start with continuous evolution and enable periodic reset only after
an operator decision or diagnosis.
