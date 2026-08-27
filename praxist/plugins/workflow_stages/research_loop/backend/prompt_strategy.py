"""Prompt guidance strategy for research-loop peers."""

from __future__ import annotations

from typing import Any

# Universal mechanism-only diversity dimensions. Designed to apply to
# ANY research project, INCLUDING constrained-scope tasks where the
# research range is fixed by task definition. Each dimension is a
# WITHIN-SCOPE innovation axis, not a scope-defining axis — so a
# variant in a constrained task (e.g. "must be a PyTorch Optimizer
# subclass") can still legitimately differ from frontier anchors on
# every dimension here.
#
# The 6 default axes:
#   1. computational_mechanism — WHAT it does (the operator / formula)
#   2. adaptation_strategy — WHAT changes during execution (vs static)
#   3. information_source — WHAT info it consumes
#   4. compute_efficiency_profile — WHAT it costs (cost / quality position)
#   5. inductive_prior — WHAT structural assumption is hard-coded
#   6. decomposition_topology — HOW the problem is subdivided
#
# These were chosen so the prior 4-dim default's "scope" axes
# (problem_formulation, methodological_approach) — which were pinned
# in constrained tasks and caused false-positive diversity violations
# — are no longer in the default. The 4-dim "operating_regime" was
# generalized into the more precise compute_efficiency_profile +
# inductive_prior. Tasks can still override with their own custom
# dimensions via `task_spec.yaml: evaluation.diversity_dimensions`.
_GENERIC_DIVERSITY_DIMENSIONS_DEFAULT: list[dict[str, str]] = [
    {
        "name": "computational_mechanism",
        "description": "The specific algorithmic operator / formula / "
        "computational mechanism this variant "
        "introduces. What does it DO internally that "
        "is distinctive from prior work?",
        "examples": "'two-step correction' / 'lookahead k-step' / "
        "'single-pass approximation' / 'spectral method' / "
        "'domain-decomposition solver' / the core computational mechanism "
        "in 1 sentence.",
    },
    {
        "name": "adaptation_strategy",
        "description": "What is dynamic / adaptive / state-dependent "
        "in this variant vs fixed / static? Captures "
        "the scheduling and adaptation behavior.",
        "examples": "fixed-throughout / cosine schedule / adaptive "
        "(state-dependent) / per-element / per-layer / "
        "curriculum-like / regime-switching.",
    },
    {
        "name": "information_source",
        "description": "What information / signal does this variant USE "
        "that prior variants don't (or vice versa)? "
        "Captures the input-dependence axis.",
        "examples": "'primary measurement only' / 'measurement + "
        "context summary' / 'measurement + historical outcome' / "
        "'governing equations only' / '+ boundary heuristics' / "
        "'+ adjoint information'.",
    },
    {
        "name": "compute_efficiency_profile",
        "description": "Where the variant sits on the cost / quality "
        "tradeoff axis (compute, memory, latency).",
        "examples": "'2x baseline compute' / '1.2x periodic' / "
        "'1.0x single-pass approximation' / "
        "'memory-bounded streaming' / "
        "'low-rank approximation'.",
    },
    {
        "name": "inductive_prior",
        "description": "What structural assumption / prior is "
        "hard-coded into the variant? What does it "
        "assume to be true about the problem?",
        "examples": "'local smoothness' / 'conservation laws' / "
        "'periodicity' / 'recent context shifts future outcomes' / "
        "'component sensitivity varies across the system'.",
    },
    {
        "name": "decomposition_topology",
        "description": "How is the problem subdivided / structured? "
        "Flat vs hierarchical, end-to-end vs pipeline, "
        "sequential vs parallel, global vs local.",
        "examples": "'single global rule' / 'component-wise rule' / "
        "'hierarchical decomposition' / 'domain decomposition' / "
        "'multigrid' / 'global solver'.",
    },
]


def _format_dimensions_block(dimensions: list[dict[str, Any]]) -> str:
    """Render a list of dimensions as the prompt's design-axis block."""
    lines = []
    for i, d in enumerate(dimensions, 1):
        name = d.get("name", f"dim_{i}")
        desc = d.get("description", "")
        ex = d.get("examples", "")
        lines.append(f"  {i}. **{name}**: {desc}")
        if ex:
            lines.append(f"     Examples: {ex}")
    return "\n".join(lines)


def _build_diversity_penalty_block(
    target_anchors: list[dict[str, Any]],
    diversity_dimensions: list[dict[str, Any]] | None = None,
    *,
    phase_label: str = "explore phase",
) -> str:
    """Render the diversity-penalty prompt block.

    Round 4 C2 fix: takes the explicit list of anchor entries so the
    prompt can NAME them (no per-peer ordering ambiguity).

            2026-04-30 user-feedback fix: dimensions are now CONFIGURABLE per
            task. The prior version hard-coded task-family-specific dimensions,
            which was wrong for unrelated domains. Tasks specify their dimensions
            via ``evaluation.diversity_dimensions`` in task_spec.yaml; if absent,
            the generic mechanism-only defaults above are used.

    The "differ in ≥ N of M" rule scales with len(dimensions); tasks can
    tune the bar by providing their own dimension set.

    ``target_anchors`` is the list of frontier entries to differentiate
    against (caller slices to top-N before calling).
    ``diversity_dimensions`` is the list of dimension dicts; defaults
    to the generic mechanism-only set.
    """
    if not target_anchors:
        return ""

    if not diversity_dimensions:
        diversity_dimensions = _GENERIC_DIVERSITY_DIMENSIONS_DEFAULT

    # "≥ ⌈M/2⌉ of M must differ" — scales with dimension count.
    # 4 dims → 2; 6 dims → 3; 8 dims → 4. Rounds up so 5 dims → 3.
    n_dims = len(diversity_dimensions)
    differ_threshold = max(1, (n_dims + 1) // 2)

    # Render anchor list (round 5 M6: full finding_id, not truncated).
    anchor_lines = []
    for i, a in enumerate(target_anchors, 1):
        v = a.get("variant_name") or a.get("finding_id", "?")
        mv = a.get("metric_value")
        mv_s = f"{mv:+.3f}" if isinstance(mv, (int, float)) else "?"
        anchor_lines.append(
            f"  {i}. variant={v!r}  metric={mv_s}  finding_id={a.get('finding_id', '')}"
        )
    anchor_block = "\n".join(anchor_lines)
    n = len(target_anchors)

    dims_block = _format_dimensions_block(diversity_dimensions)

    return f"""

## DIVERSITY GUIDANCE (realized self-report; soft enforcement)

This generation is in **{phase_label}**. Articulate your variant
along the {n_dims} design dimensions below. The system will compare
your articulation against the {n} frontier anchor(s) listed below
and classify the result into one of three tiers:

  - **clone** (matches all common dims): strong warning logged. Avoid.
  - **narrow** (matches majority of common dims): soft signal logged.
    Many legitimate variants are narrow refinements of a baseline.
    Acceptable but conservative.
  - **clean** (differs on ≥ half of common dims): substantive novelty.

Aim for **clean**. The system DOES NOT block your submission based on
this signal. Promotion and continuation are determined by the task's
frontier policy: the primary metric may be one axis, but task-defined
lane filters, required evidence fields, optional anchors, and hard
constraints remain authoritative. The diversity signal is observational
and visible to future generations.

If your task scope is constrained (e.g. all variants must be a
specific algorithmic family, or must match a fixed interface), it is
fine to match anchors on whatever dimension is task-pinned. The point
is to articulate where your genuine novelty sits within the allowed
scope. Aim to be **clean** on the dimensions that ARE free to vary.

### Frontier anchors to compare against:

{anchor_block}

### Design dimensions to articulate:

{dims_block}

### Required action:

  (a) In your `share_finding` payload, populate a `design_dimensions`
      field as a flat dict mapping each dimension name to the short
      value that was actually implemented and evaluated. This is the
      machine-readable realized self-report. A PI/Chair
      `planned_dimensions` value is planning context only: do not copy
      it here when the implementation differs or the actual value is
      unknown. Lying defeats the purpose; the system trusts your honest
      self-assessment.
  (b) Read each anchor's `design_dimensions` (load their finding
      files) to identify their values. If an anchor lacks
      `design_dimensions`, report what you can infer from its
      `variant_name` and any documented metadata.
  (c) Aim for **clean** classification (differ from each anchor on
      ≥ {differ_threshold} of the dimensions you both report on).
      If your variant lands in **narrow** category, that is fine
      — just be aware of it.

Rationale: this dampens anchor-driven narrowing — the documented
phenomenon where a single high-metric anchor causes subsequent
generations to converge on its neighborhood. The diversity signal
helps future generations + analysts distinguish "follow-up
refinement" (narrow) from "new direction" (clean).
"""


def _build_axis_assignment_block(
    gen_id: int,
    peer_index: int,
    cohort_size: int,
    must_explore_axes: list[dict[str, str]],
) -> str:
    """If `must_explore_axes` is non-empty AND the round-robin slot
    falls within range, return a strong hint to attack the assigned
    axis. Otherwise return empty string.

    Plan C semantics:
    - Per-cycle rotation: `axis_idx = (peer_index + (gen_id // 3)) %
      len(axes)`. Under current `frontier_strategy="auto"`, axis
      assignment is used only for gen 0 because gen >= 1 is PI-directed.
      The `gen_id // 3` rotation remains only for explicit legacy
      `frontier_strategy="explore"` operator overrides, so a repeated
      explore run does not permanently lock a peer to the same axis.
    - Peers are still assigned only when `peer_index < len(axes)` after
      no-rotation modulo — i.e., we round-robin AT MOST `min(cohort,
      axes)` peers per cycle. Extras get free explore.
    - Assignment is **one-shot per explore-phase cycle** — caller
      must gate on `strategy == "explore"`.
    """
    if not must_explore_axes or peer_index >= len(must_explore_axes):
        return ""
    # Rotation is only observable for explicit legacy
    # `frontier_strategy == "explore"` overrides. In default auto mode,
    # gen 0 explores once and gen >= 1 is PI-directed.
    cycle = max(0, gen_id) // 3
    axis_idx = (peer_index + cycle) % len(must_explore_axes)
    axis = must_explore_axes[axis_idx]
    # Normalizer guarantees `name` is a non-empty string; no need to
    # re-validate here.
    name = axis["name"]
    desc = axis.get("description", "")
    detail = f" ({desc})" if desc else ""
    # R3 Issue 4 fix: count of free-explore peers is cohort_size minus
    # the number of peers that got an assignment (= min(cohort, axes)).
    # R4 Issue 1/3 fix: special-case n_assigned == 1 (the calling peer
    # is the only assignee) and cohort_size <= 1.
    n_assigned = min(cohort_size, len(must_explore_axes))
    n_free = max(0, cohort_size - n_assigned)
    if cohort_size <= 1:
        cohort_text = "You are the only peer in this cohort."
    elif n_assigned == 1:
        cohort_text = (
            f"You are the only peer assigned an axis this cycle; peers "
            f"1..{cohort_size - 1} are free-explore."
        )
    elif n_free > 0:
        # R6 Major #1 fix: drop "Other" — peer is itself in the
        # 0..n_assigned-1 range; saying "Other peers (peers 0..2)"
        # to peer 0 is self-contradicting wording.
        cohort_text = (
            f"Peers 0..{n_assigned - 1} have axis assignments "
            f"(you are one of them); peers "
            f"{n_assigned}..{cohort_size - 1} are free-explore."
        )
    else:
        cohort_text = (
            f"All {cohort_size} peers in this cohort have axis "
            f"assignments (no free-explore peers this cycle)."
        )
    return (
        # R4 Issue 5 fix: demote axis-block heading to h3 to nest
        # cleanly under the template's `### Your Exploration Mandate`
        # h3 host (avoids markdown outline inversion).
        "\n\n### Cohort axis assignment (cycle-start coordination)\n\n"
        f"You are peer_index {peer_index} in this cohort (cycle "
        f"{cycle}). To prevent the cold-start convergence pattern "
        f"(where all peers attack the same hot axis), this generation "
        f"has pre-assigned **must-explore axes** to the first N peers "
        f"(round-robin, rotated each cycle so the same peer doesn't "
        f"get the same axis twice).\n\n"
        f"**Your assigned axis: `{name}`**{detail}\n\n"
        f"Attack this axis as your primary direction. The standard "
        f"research scope still applies (and any diversity-penalty "
        f"rules above, if present); the assignment is a strong hint, "
        f"not a hard constraint. If your assigned axis is genuinely "
        f"infeasible "
        f"(e.g. requires out-of-scope changes), document why in your "
        f"`share_finding(notes=...)` and pivot — but the default is "
        f"that you make a serious attempt at the assigned axis.\n\n"
        f"{cohort_text}"
        + (
            " Use `mcp__finding-graph-query__get_unlinked_recent_findings` "
            "to see what hypotheses your cohort-mates have posted.\n"
            if cohort_size > 1
            else "\n"
        )
    )


def _generate_variant_hint(
    gen_id: int,
    peer_index: int,
    cohort_size: int,
    strategy: str,
    frontier: Any,
    diversity_dimensions: list[dict[str, Any]] | None = None,
    must_explore_axes: list[dict[str, str]] | None = None,
    frontier_summary: list[dict[str, Any]] | None = None,
) -> str:
    """Generate a variant hint for a specific peer in a generation.

    Strategies:
    - explore: encourage diverse new directions + diversity penalty
    - exploit: build on specific frontier entries
    - mixed: first half explores, second half exploits

    Args:
        diversity_dimensions: Optional task-specific dimensions used by
            the explore-phase diversity-penalty prompt block. If None,
            falls back to the generic-science default. See
            ``_build_diversity_penalty_block`` for details.
        must_explore_axes: Optional list of {name, description} dicts.
            When non-empty AND `strategy == "explore"`, the orchestrator
            round-robin-assigns axes to the first N peers (Plan C). In
            mixed/exploit mode, axis assignment is suppressed —
            homogeneous peer roles restored. Empty / None → no
            assignment, all peers get the generic explore hint.
    """
    if frontier_summary is None:
        frontier_summary = frontier.get_summary()

    # Axis assignment is gated on explore strategy ONLY (Plan C).
    # In mixed mode it's suppressed even for the explore-half peers,
    # because mixed mode is a transitional cycle phase — peers at this
    # point should be reading frontier signals, not following a fresh
    # axis quota. In exploit mode, axis assignment would conflict
    # with the "iterate on frontier entry X" instruction.
    axis_block = ""
    if strategy == "explore" and must_explore_axes:
        axis_block = _build_axis_assignment_block(
            gen_id,
            peer_index,
            cohort_size,
            must_explore_axes,
        )

    if not frontier_summary:
        # First generation: no frontier yet. axis_block was already
        # computed above gated on `strategy == "explore"`, so it's
        # correctly empty for explicit exploit/mixed overrides.
        # R4 Issue 2 + R6 Major #2 fix: when axis_block is non-empty,
        # return ONLY the axis block — the surrounding template's
        # cold-start body already provides generic "first generation"
        # framing. When axis_block is empty (axis-less task), defer
        # entirely to the template (return "") so we don't duplicate
        # the template's generic first-gen prose with our own.
        if axis_block:
            return axis_block.lstrip()
        return ""

    # R2#3 fix (v2026-05-04): when PI is driving the gen, the role
    # contract in the agenda is the authoritative directive. Keep the
    # leaderboard-driven explore/exploit hint from contradicting the
    # PI's role contract (e.g. telling an `anti_mainline` peer to "build
    # on the top result"). Still render the diversity self-report block
    # so generation-boundary telemetry can classify PI-directed findings
    # as clone/narrow/clean instead of no_data. The block is observational
    # only and explicitly defers to task/frontier policy.
    if strategy == "pi_directed":
        target_anchors = list(frontier_summary[:3])
        return (
            "Your role contract from the PI agenda above is your "
            "authoritative directive. Do NOT default to extending the "
            "leaderboard winners unless your contract explicitly tells "
            "you to (e.g. role=exploit). For non-exploit roles "
            "(falsifier, bridge, anti_mainline), follow the "
            "contract semantics — leaderboard-chasing is forbidden."
            + _build_diversity_penalty_block(
                target_anchors,
                diversity_dimensions=diversity_dimensions,
                phase_label="PI-directed phase",
            )
        )

    if strategy == "explore":
        # round 3 M6 fix: cap diversity-penalty target count at 3.
        # round 4 C2 fix: pass the actual anchor entries (not just a
        # count), so the prompt names them explicitly and all peers see
        # the same canonical list.
        target_anchors = list(frontier_summary[:3])
        return (
            "Explore a DIFFERENT direction from previous generations. "
            "Do NOT replicate what has already been tried. "
            "Check the frontier for what's been done and propose something novel."
            + _build_diversity_penalty_block(
                target_anchors,
                diversity_dimensions=diversity_dimensions,
            )
            + axis_block
        )

    if strategy == "exploit":
        # Pick a specific entry to build on
        entries = frontier_summary
        target = entries[peer_index % len(entries)]
        return (
            f"Build on the top result from Generation {target['generation_id']}: "
            f"'{target.get('variant_name', 'unknown')}' "
            f"(metric={target.get('metric_value', 'N/A')}). "
            f"Download its snapshot and iterate to improve it. "
            f'Snapshot: download_snapshot("{target.get("finding_id", "")}")'
        )

    # mixed: first half explores, second half exploits.
    # Plan C: do NOT propagate `must_explore_axes` into the inner
    # recursive call. Mixed mode is a transitional cycle phase where
    # peers should be reading frontier signals, not following a fresh
    # axis quota — homogeneous peer roles restored.
    if peer_index < cohort_size // 2:
        return _generate_variant_hint(
            gen_id,
            peer_index,
            cohort_size,
            "explore",
            frontier,
            diversity_dimensions=diversity_dimensions,
            must_explore_axes=None,
            frontier_summary=frontier_summary,
        )
    else:
        return _generate_variant_hint(
            gen_id,
            peer_index,
            cohort_size,
            "exploit",
            frontier,
            diversity_dimensions=diversity_dimensions,
            must_explore_axes=None,
            frontier_summary=frontier_summary,
        )
