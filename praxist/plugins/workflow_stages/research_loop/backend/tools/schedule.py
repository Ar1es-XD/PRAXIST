"""
Schedule helpers parameterized by epoch-fraction, not step-count.

Problem this solves:
    L:22566 peer0 "total_steps=3910 (10ep) vs 78200 (200ep) makes lambda
    fully active in short runs but inactive in long runs."

This silent bug invalidated cross-run comparisons because the same YAML
hyperparameter (`lambda_schedule`) behaved differently at 10 vs 200
epochs. The fix is to express every schedule in terms of epoch-fraction
(`progress ∈ [0, 1]`) so its shape is identical across run lengths.

Intended usage inside a variant implementation:

    from praxist.plugins.workflow_stages.research_loop.backend.tools.schedule import (
        epoch_fraction, cosine_schedule,
    )

    progress = epoch_fraction(current_epoch, total_epochs)
    rho = cosine_schedule(progress, start=0.05, end=0.10)

All schedules are pure functions. They do NOT touch global state, torch,
or numpy; they return plain Python floats. Optimizers can wrap them in
whatever tensor framework they use.

This module is topic-agnostic — it also helps runtime schedules, penalty
coefficients, curriculum temperature, and other progress-shaped parameters.
"""

from __future__ import annotations

import math


def epoch_fraction(
    current_epoch: float,
    total_epochs: float,
    *,
    clamp: bool = True,
) -> float:
    """Convert (current_epoch, total_epochs) to a fraction in [0, 1].

    Parameters
    ----------
    current_epoch : float
        Zero-indexed or one-indexed; treat consistently within your variant.
        Fractional values (e.g. 1.5 for mid-epoch-1) are supported.
    total_epochs : float
        The planned total (e.g. 10 or 200).
    clamp : bool
        If True (default), output is clamped to [0, 1]. If False, values
        outside the range pass through — useful for schedules defined
        outside [0, 1] (rare).

    Returns
    -------
    float
    """
    if total_epochs <= 0:
        return 0.0
    p = current_epoch / total_epochs
    if clamp:
        if p < 0.0:
            return 0.0
        if p > 1.0:
            return 1.0
    return p


def linear_schedule(
    progress: float,
    *,
    start: float,
    end: float,
    clamp_progress: bool = True,
) -> float:
    """Linear interpolation from `start` at progress=0 to `end` at progress=1."""
    if clamp_progress:
        progress = max(0.0, min(1.0, progress))
    return start + (end - start) * progress


def cosine_schedule(
    progress: float,
    *,
    start: float,
    end: float,
    clamp_progress: bool = True,
) -> float:
    """Cosine interpolation from `start` (progress=0) to `end` (progress=1).

    Shape: smooth start and end; steepest change at progress=0.5.
    """
    if clamp_progress:
        progress = max(0.0, min(1.0, progress))
    # cos goes 1 → -1 over [0, pi]; map to [0, 1] via (1 - cos) / 2.
    t = 0.5 * (1.0 - math.cos(math.pi * progress))
    return start + (end - start) * t


def peaked_schedule(
    progress: float,
    *,
    start: float,
    peak: float,
    end: float | None = None,
    peak_at: float = 0.5,
    clamp_progress: bool = True,
) -> float:
    """Rise-and-fall schedule, peaks at `peak_at`.

    Useful for curriculum-style regularization (e.g. SAGO's peaked gamma from
    the 2026-04-16 run, where γ(t) = γ_max · sin(π t)).

    If `end` is None, defaults to `start` (symmetric rise/fall).
    The left and right halves use cosine interpolation for smoothness.
    """
    if end is None:
        end = start
    if clamp_progress:
        progress = max(0.0, min(1.0, progress))
    if peak_at <= 0.0:
        return cosine_schedule(progress, start=peak, end=end)
    if peak_at >= 1.0:
        return cosine_schedule(progress, start=start, end=peak)
    if progress <= peak_at:
        sub = progress / peak_at
        return cosine_schedule(sub, start=start, end=peak)
    else:
        sub = (progress - peak_at) / (1.0 - peak_at)
        return cosine_schedule(sub, start=peak, end=end)


def warmup_then_schedule(
    progress: float,
    *,
    warmup_fraction: float,
    warmup_start: float,
    base_start: float,
    base_end: float,
    base_kind: str = "cosine",
    clamp_progress: bool = True,
) -> float:
    """Linear warmup in [0, warmup_fraction], then cosine/linear to end.

    Handles the common "first X%: ramp in, rest: decay" pattern in an
    epoch-fraction-agnostic way. Kind must be "cosine" or "linear".
    """
    if clamp_progress:
        progress = max(0.0, min(1.0, progress))
    if warmup_fraction <= 0.0:
        if base_kind == "linear":
            return linear_schedule(progress, start=base_start, end=base_end)
        return cosine_schedule(progress, start=base_start, end=base_end)
    if progress < warmup_fraction:
        sub = progress / warmup_fraction
        return linear_schedule(sub, start=warmup_start, end=base_start)
    sub = (progress - warmup_fraction) / max(1e-9, 1.0 - warmup_fraction)
    if base_kind == "linear":
        return linear_schedule(sub, start=base_start, end=base_end)
    return cosine_schedule(sub, start=base_start, end=base_end)


# ---------------------------------------------------------------------------
# Static check helper
# ---------------------------------------------------------------------------

# Variant-scaffolding CI / peer self-check can invoke this to warn about the
# step-count anti-pattern at variant-registration time (I-5 advisory check).
SUSPICIOUS_STEP_PATTERNS = (
    "total_steps",
    "num_training_steps",
    "max_steps",
)


def scan_for_step_anti_pattern(source: str) -> list:
    """Return a list of (line_no, line_text, pattern) triples.

    For each occurrence of `total_steps` / `max_steps` / `num_training_steps`
    that isn't inside a comment OR inside a triple-quoted string (heuristic).
    Not a real Python parser — good enough to avoid alerting on docstrings
    while catching the step-count anti-pattern in actual code.
    """
    hits = []
    in_triple = False
    triple_delim = None  # '"""' or "'''"
    for i, line in enumerate(source.splitlines(), start=1):
        # Update triple-quote state line-by-line. We only track the state
        # crossing lines; the pattern inside a line-spanning triple block
        # is still skipped. Single-line `"""foo"""` toggles twice and
        # ends up outside — correct.
        idx = 0
        line_has_active_triple = in_triple
        while True:
            if not in_triple:
                # look for an opening triple
                next_tri = None
                for cand in ('"""', "'''"):
                    pos = line.find(cand, idx)
                    if pos != -1 and (next_tri is None or pos < next_tri[0]):
                        next_tri = (pos, cand)
                if next_tri is None:
                    break
                pos, cand = next_tri
                # Enter triple mode; content after pos on this line is inside.
                in_triple = True
                triple_delim = cand
                idx = pos + len(cand)
                line_has_active_triple = True
            else:
                close = line.find(triple_delim or "", idx)
                if close == -1:
                    break
                in_triple = False
                triple_delim = None
                idx = close + 3
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line_has_active_triple:
            # Some or all of this line is inside a triple-quoted string —
            # skip to avoid docstring false positives.
            continue
        for pat in SUSPICIOUS_STEP_PATTERNS:
            if pat in line:
                hits.append((i, line.rstrip(), pat))
                break
    return hits
