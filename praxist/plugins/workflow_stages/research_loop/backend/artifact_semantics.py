"""Artifact role metadata for research-loop run products.

The research loop keeps a small set of canonical state owners and many useful
views/snapshots. This module makes that distinction explicit without adding a
new persistent state file.
"""

from __future__ import annotations

import json
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CANONICAL_STATE = "canonical_state"
DERIVED_VIEW = "derived_view"
AUDIT_SNAPSHOT = "audit_snapshot"
DERIVED_AUDIT_SNAPSHOT = "derived_audit_snapshot"
PARTIAL_OUTPUT = "partial_output"

COMMITTED = "committed"
PARTIAL = "partial"
FAILED = "failed"
SUPERSEDED = "superseded"


def utc_now_iso() -> str:
    """Return the current UTC timestamp in compact ISO-8601 form."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def artifact_semantics(
    *,
    role: str,
    status: str = COMMITTED,
    stage: str,
    generation_id: int | None = None,
    actor: str = "",
    derived_from: list[str] | None = None,
    canonical_sources: list[str] | None = None,
    runtime_fact_source: bool | None = None,
    created_at: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Return compact, JSON/YAML-safe artifact role metadata.

    ``runtime_fact_source`` defaults to true only for committed canonical
    state. Derived views and audit snapshots are preserved for observability,
    but runtime readers must not treat them as owners of current facts.
    """

    normalized_role = str(role or "").strip() or DERIVED_VIEW
    normalized_status = str(status or "").strip() or COMMITTED
    if runtime_fact_source is None:
        runtime_fact_source = normalized_role == CANONICAL_STATE and normalized_status == COMMITTED
    payload: dict[str, Any] = {
        "schema_version": "praxist.artifact_semantics.v1",
        "role": normalized_role,
        "status": normalized_status,
        "stage": str(stage or "").strip(),
        "runtime_fact_source": bool(runtime_fact_source),
        "derived": normalized_role in {DERIVED_VIEW, DERIVED_AUDIT_SNAPSHOT},
        "audit_only": normalized_role in {AUDIT_SNAPSHOT, DERIVED_AUDIT_SNAPSHOT},
        "created_at": created_at or utc_now_iso(),
    }
    if generation_id is not None:
        payload["generation_id"] = int(generation_id)
    if actor:
        payload["actor"] = str(actor)
    if derived_from:
        payload["derived_from"] = [str(item) for item in derived_from if str(item or "").strip()]
    if canonical_sources:
        payload["canonical_sources"] = [
            str(item) for item in canonical_sources if str(item or "").strip()
        ]
    if notes:
        payload["notes"] = str(notes)
    return payload


def attach_artifact_semantics(
    payload: dict[str, Any],
    *,
    role: str,
    status: str = COMMITTED,
    stage: str,
    generation_id: int | None = None,
    actor: str = "",
    derived_from: list[str] | None = None,
    canonical_sources: list[str] | None = None,
    runtime_fact_source: bool | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Return a shallow copy of ``payload`` with explicit artifact semantics."""

    out = dict(payload)
    out["artifact_semantics"] = artifact_semantics(
        role=role,
        status=status,
        stage=stage,
        generation_id=generation_id,
        actor=actor,
        derived_from=derived_from,
        canonical_sources=canonical_sources,
        runtime_fact_source=runtime_fact_source,
        notes=notes,
    )
    return out


def is_runtime_fact_source(payload: Any) -> bool:
    """Return true when a payload is explicitly marked as a runtime fact owner."""

    if not isinstance(payload, dict):
        return False
    semantics = payload.get("artifact_semantics")
    if not isinstance(semantics, dict):
        return False
    return bool(semantics.get("runtime_fact_source") is True)


def has_explicit_artifact_semantics(payload: Any) -> bool:
    """Return true when a payload carries Praxist artifact role metadata."""

    return isinstance(payload, dict) and isinstance(payload.get("artifact_semantics"), dict)


def is_committed_runtime_fact_source(
    payload: Any,
    *,
    legacy_ok: bool = True,
    require_canonical: bool = True,
) -> bool:
    """Return whether a payload may be used as committed runtime state.

    Legacy run products did not include ``artifact_semantics``. When
    ``legacy_ok`` is true, those products remain readable for resume and
    historical runs. Once a file explicitly declares artifact semantics, runtime
    readers must honor it and reject derived, audit, partial, failed, or
    superseded artifacts as fact owners.
    """

    if not isinstance(payload, dict):
        return False
    semantics = payload.get("artifact_semantics")
    if not isinstance(semantics, dict):
        return bool(legacy_ok)
    role = str(semantics.get("role") or "").strip()
    status = str(semantics.get("status") or "").strip().lower()
    if require_canonical and role != CANONICAL_STATE:
        return False
    if status != COMMITTED:
        return False
    if semantics.get("runtime_fact_source") is not True:
        return False
    return not (semantics.get("audit_only") or semantics.get("derived"))


def is_committed_runtime_fact_file(path: str | Path, *, legacy_ok: bool = True) -> bool:
    """Return whether ``path`` contains usable committed runtime state."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return is_committed_runtime_fact_source(payload, legacy_ok=legacy_ok)


def is_readable_signal_source(payload: Any, *, legacy_ok: bool = True) -> bool:
    """Return whether a payload may be mined for non-durable research signals.

    This is intentionally looser than ``is_committed_runtime_fact_source``:
    derived views, audit snapshots, and partial outputs may still contain
    useful validation-candidate signals. Failed artifacts may still contain
    useful negative evidence or repair targets, so they are readable as signals
    only. They must not become durable frontier, Gems, or resume facts, but
    agents should not lose them merely because the container artifact is not
    authoritative. Superseded artifacts remain excluded because their useful
    signal should have moved to the replacement artifact.
    """

    if not isinstance(payload, dict):
        return False
    semantics = payload.get("artifact_semantics")
    if not isinstance(semantics, dict):
        return bool(legacy_ok)
    status = str(semantics.get("status") or "").strip().lower()
    return status != SUPERSEDED


def is_audit_or_derived(payload: Any) -> bool:
    """Return true when a payload is marked as a derived view or audit snapshot."""

    if not isinstance(payload, dict):
        return False
    semantics = payload.get("artifact_semantics")
    if not isinstance(semantics, dict):
        return False
    return bool(semantics.get("audit_only") or semantics.get("derived"))


def explicit_entry_generation_id(
    payload: Any,
    *,
    generation_hint: int | None = None,
) -> int | None:
    """Return the latest explicitly recorded generation for an entry.

    Canonical state readers use this helper for temporal cutoffs.  Identity
    strings are deliberately ignored: a variant name may legitimately contain
    text that resembles a generation number.  A containing generation bucket
    is also provenance, so callers may supply it as ``generation_hint``.
    """

    generations: list[int] = []
    if generation_hint is not None:
        with suppress(TypeError, ValueError):
            generations.append(int(generation_hint))
    if not isinstance(payload, dict):
        return max(generations) if generations else None
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    for source in (payload, metrics):
        for key in ("source_generation_id", "generation_id", "gen_id"):
            value = source.get(key)
            if value is None:
                continue
            try:
                generations.append(int(value))
            except (TypeError, ValueError):
                continue
    return max(generations) if generations else None
