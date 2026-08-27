"""Canonical label helpers for DIG quality-diversity allocation.

The normalizer is deliberately task-agnostic.  Task projects may provide
keyword groups in DIG config, but Praxist core only owns light lexical cleanup and
generic fallback rules.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_CORE_SYNONYMS = {
    "calibrate": "calibration",
    "score_calibration": "calibration",
    "regularize": "regularization",
    "regularizer": "regularization",
    "safety": "constraint_safety",
    "constraint": "constraint_safety",
    "diagnostic": "diagnostic_falsifier",
    "falsifier": "diagnostic_falsifier",
    "falsification": "diagnostic_falsifier",
    "control": "diagnostic_falsifier",
    "ablation": "diagnostic_falsifier",
    "exploit_parent": "exploit",
    "validate": "exploit",
    "stress_validate": "exploit",
}


@dataclass(frozen=True)
class CanonicalDIGLabels:
    """Raw and canonical labels used by DIG QD selection and audit traces."""

    raw_mechanism_family: str = ""
    raw_intervention_surface: str = ""
    raw_intent: str = ""
    raw_semantic_family: str = ""
    raw_parent_lineage: str = ""
    raw_novelty_axis: str = ""
    canonical_mechanism_family: str = "other"
    canonical_intervention_surface: str = "other"
    canonical_intent: str = "explore"
    canonical_semantic_family: str = "other"
    canonical_parent_lineage: str = "none"
    canonical_novelty_axis: str = "unspecified"
    reasons: tuple[str, ...] = ()

    def formal_cell(self) -> tuple[str, str, str]:
        return (
            self.canonical_mechanism_family,
            self.canonical_intervention_surface,
            self.canonical_intent,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        return data


def normalize_label(
    value: Any,
    default: str = "other",
    *,
    synonyms: dict[str, str] | None = None,
) -> str:
    """Return a stable lower-snake label for arbitrary LLM text."""

    text = str(value or "").strip().lower()
    text = _NON_ALNUM_RE.sub("_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = default
    configured = {
        normalize_label(key, "", synonyms={}): normalize_label(value, "", synonyms={})
        for key, value in (synonyms or {}).items()
    }
    return configured.get(text, _CORE_SYNONYMS.get(text, text))


def _candidate_text(candidate: Any, fields: list[str] | tuple[str, ...] | None = None) -> str:
    fields = list(fields or [])
    if not fields:
        fields = [
            "name",
            "mechanism_family",
            "intervention_surface",
            "intent",
            "hypothesis",
            "expected_gain_path",
            "changes",
            "semantic_family",
            "parent_lineage",
            "novelty_axis",
        ]
    parts: list[str] = []
    for field in fields:
        key = str(field or "").strip()
        if key == "name":
            parts.append(str(getattr(candidate, "name", "") or ""))
        elif key == "mechanism_family":
            parts.append(str(getattr(candidate, "mechanism_family", "") or ""))
        elif key == "intervention_surface":
            parts.append(str(getattr(candidate, "intervention_surface", "") or ""))
        elif key == "intent":
            parts.append(str(getattr(candidate, "intent", "") or ""))
        elif key == "hypothesis":
            parts.append(str(getattr(candidate, "hypothesis", "") or ""))
        elif key == "expected_gain_path":
            parts.append(str(getattr(candidate, "expected_gain_path", "") or ""))
        elif key == "semantic_family":
            parts.append(str(getattr(candidate, "semantic_family", "") or ""))
        elif key == "parent_lineage":
            parts.append(str(getattr(candidate, "parent_lineage", "") or ""))
        elif key == "novelty_axis":
            parts.append(str(getattr(candidate, "novelty_axis", "") or ""))
        elif key == "changes":
            sketch = getattr(candidate, "implementation_sketch", None)
            parts.extend(str(item or "") for item in getattr(sketch, "changes", []) or [])
        elif key == "files_to_modify":
            sketch = getattr(candidate, "implementation_sketch", None)
            parts.extend(str(item or "") for item in getattr(sketch, "files_to_modify", []) or [])
    return " ".join(part for part in parts if part).lower()


def _match_config_group(
    candidate: Any,
    groups: list[Any],
    *,
    synonyms: dict[str, str] | None = None,
) -> tuple[str, str] | None:
    for group in groups or []:
        name = normalize_label(getattr(group, "name", ""), "", synonyms=synonyms)
        keywords = [
            normalize_label(item, "", synonyms=synonyms)
            for item in getattr(group, "keywords", []) or []
        ]
        keywords = [item for item in keywords if item]
        if not name or not keywords:
            continue
        text = normalize_label(
            _candidate_text(candidate, getattr(group, "fields", []) or []),
            "",
            synonyms=synonyms,
        )
        if any(keyword and keyword in text for keyword in keywords):
            return name, f"matched task-local label group `{name}`"
    return None


def canonical_labels_for_candidate(candidate: Any, config: Any | None = None) -> CanonicalDIGLabels:
    """Infer canonical QD labels while retaining raw labels and reasons."""

    qd = getattr(config, "cohort_qd", None)
    semantic_groups = list(getattr(qd, "semantic_label_groups", []) or [])
    parent_groups = list(getattr(qd, "parent_lineage_label_groups", []) or [])
    novelty_groups = list(getattr(qd, "novelty_axis_label_groups", []) or [])
    label_synonyms = dict(getattr(qd, "label_synonyms", {}) or {})

    raw_mechanism = str(getattr(candidate, "mechanism_family", "") or "")
    raw_surface = str(getattr(candidate, "intervention_surface", "") or "")
    raw_intent = str(getattr(candidate, "intent", "") or "")
    sig = getattr(candidate, "diversity_signature", None)
    if sig is not None:
        raw_mechanism = str(getattr(sig, "mechanism_family", raw_mechanism) or raw_mechanism)
        raw_surface = str(getattr(sig, "intervention_surface", raw_surface) or raw_surface)
        raw_intent = str(getattr(sig, "intent", raw_intent) or raw_intent)

    raw_semantic = str(getattr(candidate, "semantic_family", "") or "")
    raw_parent = str(getattr(candidate, "parent_lineage", "") or "")
    raw_novelty = str(getattr(candidate, "novelty_axis", "") or "")

    reasons: list[str] = []
    semantic_match = _match_config_group(candidate, semantic_groups, synonyms=label_synonyms)
    parent_match = _match_config_group(candidate, parent_groups, synonyms=label_synonyms)
    novelty_match = _match_config_group(candidate, novelty_groups, synonyms=label_synonyms)

    canonical_semantic = normalize_label(raw_semantic, "", synonyms=label_synonyms)
    if semantic_match:
        canonical_semantic, reason = semantic_match
        reasons.append(reason)
    if not canonical_semantic:
        canonical_semantic = normalize_label(raw_mechanism, "other", synonyms=label_synonyms)
        reasons.append("semantic_family defaulted from mechanism_family")

    canonical_parent = normalize_label(raw_parent, "", synonyms=label_synonyms)
    if parent_match:
        canonical_parent, reason = parent_match
        reasons.append(reason)
    if not canonical_parent:
        text = normalize_label(_candidate_text(candidate), "", synonyms=label_synonyms)
        if "gem" in text:
            canonical_parent = "gem_lineage"
            reasons.append("parent_lineage inferred from Gem reference")
        elif "frontier" in text:
            canonical_parent = "frontier_lineage"
            reasons.append("parent_lineage inferred from frontier reference")
        elif "baseline" in text:
            canonical_parent = "baseline_lineage"
            reasons.append("parent_lineage inferred from baseline reference")
        else:
            canonical_parent = "none"
            reasons.append("parent_lineage defaulted to none")

    canonical_novelty = normalize_label(raw_novelty, "", synonyms=label_synonyms)
    if novelty_match:
        canonical_novelty, reason = novelty_match
        reasons.append(reason)
    if not canonical_novelty:
        canonical_novelty = normalize_label(
            raw_surface or raw_mechanism,
            "unspecified",
            synonyms=label_synonyms,
        )
        reasons.append("novelty_axis defaulted from intervention surface")

    return CanonicalDIGLabels(
        raw_mechanism_family=raw_mechanism,
        raw_intervention_surface=raw_surface,
        raw_intent=raw_intent,
        raw_semantic_family=raw_semantic,
        raw_parent_lineage=raw_parent,
        raw_novelty_axis=raw_novelty,
        canonical_mechanism_family=normalize_label(
            raw_mechanism,
            "other",
            synonyms=label_synonyms,
        ),
        canonical_intervention_surface=normalize_label(
            raw_surface,
            "other",
            synonyms=label_synonyms,
        ),
        canonical_intent=normalize_label(raw_intent, "explore", synonyms=label_synonyms),
        canonical_semantic_family=canonical_semantic,
        canonical_parent_lineage=canonical_parent,
        canonical_novelty_axis=canonical_novelty,
        reasons=tuple(reasons),
    )
