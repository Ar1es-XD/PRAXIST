# Quality-Diversity Allocation

Quality-Diversity (QD) seeks a varied set of strong solutions rather than one
winner. The term follows Pugh, Soros, and Stanley's
[Quality Diversity: A New Frontier for Evolutionary Computation](https://doi.org/10.3389/frobt.2016.00040).
Praxist applies that principle to candidate-plan allocation, not evolutionary
genotype search.

QD is generation-aware and independently configurable. In absolute generation
0 it extends the Deep Innovation Gate (DIG) read-only planning phase. In later
generations, where DIG is off by default, it guides the existing Principal
Investigator (PI) synthesis path instead of creating another planner or
allocator artifact. Multi-PI planning adds a Chair that consolidates proposals.

The goal is to keep DIG's rigor while restoring the exploration breadth that a
direct no-DIG run can have.

## Problem

DIG improves individual peer plans:

- each peer maps the baseline before editing code;
- each peer generates several mechanism-level candidates;
- each candidate receives a structured critique;
- each implementation is locked by `selected_contract.yaml`.

The weak point is that peer-local selection can converge. If every peer sees the
same frontier, Gems, and research agenda, then many peers may select the same
obvious family: reward shaping, calibration, risk repair, or the latest Gem
lineage. This produces careful contracts, but a narrower generation.

The desired behavior is:

```text
deep individual reasoning
+ cohort-level diversity control
+ no fixed task-specific algorithm quota
+ no extra experiments during DIG
```

## Non-Goals

Initial-generation QD does not:

- change generation semantics;
- add an experiment loop;
- run training or any task-defined preliminary, aligned, or complete evaluation
  during DIG;
- create a new agent runtime;
- encode task-specific metrics in Praxist core;
- force every peer into a hand-written algorithm family;
- replace Chair or PI judgment.

## Architecture

Initial-generation flow:

```text
peer context
  -> DIG candidate generation
  -> DIG critique
  -> peer-local QD selection
  -> selected_contract.yaml
  -> implementation peer
```

Initial-generation QD flow:

```text
all peer contexts
  -> run each peer's DIG candidate generation and critique concurrently
  -> collect candidate pools and reviews
  -> cohort-level QD allocator chooses one candidate per peer
  -> selected_contract.yaml is updated per peer
  -> implementation peers launch
```

Each peer still owns its own candidate pool. The allocator never assigns peer A's
candidate to peer B. It only decides which candidate from each peer's own DIG
pool should become that peer's locked contract.

Later-generation Multi-PI flow:

```text
completed-generation evidence
  -> independent PI memos propose experiments and peer contracts
  -> union of PI proposals is the candidate pool
  -> Chair applies prompt-guided, soft quality-diversity allocation
  -> normal research_agenda peer_contracts
  -> direct implementation peers (no DIG call)
```

Later-generation single-PI flow uses that PI's normal synthesis over findings,
frontier, prior agendas, validation signals, and Gems. The PI forms proposals
and chooses the final `peer_contracts` in one existing synthesis call under the
same soft QD policy. There is no PI-memo union or Chair in this topology.

Both paths are the established non-DIG planning path with a compact policy in
prompt context. Later-generation QD is intentionally prompt-guided rather than
a second deterministic allocator: it creates no planner, contract format,
candidate file, or fact artifact. Diagnostics must therefore inspect the final
agenda and, for Multi-PI, the existing PI memos; they must not expect a
post-gen0 `dig_cohort_allocation.yaml`.

When a task declares `evaluation.diversity_dimensions`, QD-enabled PI/Chair
planning records the intended value of each applicable axis in
`peer_contracts[].planned_dimensions`. This is a plan, not experimental
evidence. The peer reports the implemented/evaluated values under the existing
finding field `design_dimensions`. Diagnostics derive the planned
Herfindahl-Hirschman Index (HHI) from the agenda and realized HHI from findings,
then report missingness and drift. Praxist does not copy plans into missing
result evidence and does not turn a missing dimension report into a hard
execution gate.

## Selection Policy

The allocator works over generic descriptors:

```text
mechanism_family
intervention_surface
intent
candidate text
risk labels
peer lane fit
local DIG selection
known frontier/Gems/sibling signatures
```

It scores candidates with:

```text
selection_score =
  quality_score
  + lane_fit_bonus
  + local_selection_bonus
  + novelty_bonus
  + target_keyword_bonus
  - risk_penalty
  - diagnostic_penalty_when_not_in_diagnostic_slot
```

The gen0 deterministic allocator then applies cohort constraints. Later PI
synthesis receives the applicable quality, novelty, lane-fit, risk, target,
label-group, and diversity-cap controls as soft allocation guidance:

- max peers per exact diversity cell;
- max peers per mechanism family;
- max peers per intervention surface;
- max peers per intent;
- max peers sharing an intent, without assuming which task-owned intents are
  diagnostic;
- optional task-defined keyword targets.

Keyword targets are generic and task-owned. Praxist core only sees named text
groups such as `architecture_or_representation` or `input_feature_use`; a task
project chooses the keywords and minimum counts.

## Target Groups

Task projects may define soft minimums under the independent policy block:

```yaml
quality_diversity:
  enabled: true
  initial_generation_enabled: true
  later_generations_enabled: true
  target_keyword_groups:
    - name: architecture_or_representation
      min_peers: 2
      fields: [mechanism_family, intervention_surface, hypothesis, changes]
      keywords: [architecture, representation, encoder, attention, model_def]
```

`initial_generation_enabled` is an independent disable switch, but gen0 QD
still needs an active gen0 DIG scope because its candidate pool comes from DIG.
Later-generation QD does not depend on DIG.

Targets are not fixed quotas. If no valid candidate matches a target, the
allocator records the miss and continues. The generation must stay live.

## Contract Construction

If the cohort allocator keeps a peer's local selected candidate, it preserves the
existing LLM-authored contract.

If the allocator chooses a different candidate from the same peer's candidate
pool, it creates a deterministic contract from the validated candidate sketch:

- `variant_name` from candidate name and peer id;
- `diversity_cell` from candidate signature;
- `mechanism_hypothesis` from candidate hypothesis;
- `files_to_modify` from candidate implementation sketch;
- `allowed_changes` from candidate changes;
- standard forbidden changes for evaluator, split, and metric calculation;
- implementation steps derived from the sketch;
- expected metric signature from the candidate diagnostic prediction;
- ablation hooks from the candidate, with a fallback hook if needed.

The normal DIG validator still checks the contract. Invalid allocations are not
allowed to reach implementation.

## Allocation Failure Behavior

QD is conservative:

- if initial-generation QD is disabled, DIG uses quality-first eligible
  selection without duplicate/cell allocation;
- if later-generation QD is disabled, single-PI or PI/Chair behavior is
  identical to the prior non-DIG agenda path;
- if a peer has no valid candidate alternatives, it keeps its local DIG
  contract;
- if allocator validation fails for a peer, it keeps that peer's local DIG
  contract and records the reason;
- if all DIG attempts fail for a peer, the existing DIG fallback-to-direct
  behavior remains unchanged.

## Expected Effects

Compared with peer-local DIG, initial-generation QD should:

- preserve the stronger mechanism hypotheses and ablation discipline;
- reduce repeated same-family contracts in a generation;
- allocate at least some peers to architecture, representation, input-feature,
  off-mainline, or independent exploration when candidate pools support it;
- keep diagnostic/control work bounded through the independent gen0 DIG
  innovation-slot policy;
- keep Gems useful without letting every peer inherit the same Gem lineage.

Compared with no DIG, the initial-generation QD stage should:

- produce more explicit implementation contracts;
- reduce first-intuition coding;
- make failures easier to interpret;
- avoid changing task metrics, evaluator, data split, or baseline contract by
  accident.

## Test Requirements

Tests should cover:

- independent DIG scope and initial/later QD config parsing;
- default DIG execution only at absolute gen0, including across Gems resets;
- cohort allocation preserves one selected candidate per peer;
- max same mechanism family is enforced when alternatives exist;
- max same intervention surface is enforced when alternatives exist;
- target keyword groups are filled when candidates exist;
- local DIG contract is preserved when allocation is disabled;
- quality-first DIG selection when initial QD is disabled;
- later single-PI synthesis and Multi-PI/Chair prompts receive QD policy only
  when enabled;
- later QD does not call DIG or create a separate candidate artifact;
- deterministic override contracts pass the existing DIG validator;
- generation prompt injection uses the final cohort-selected contract;
- fallback-to-direct behavior still works after repeated DIG failure.
