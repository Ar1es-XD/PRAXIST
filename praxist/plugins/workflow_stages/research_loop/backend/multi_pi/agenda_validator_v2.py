"""Schema v2.0 validator for final_research_agenda.yaml.

Backward-compatible: v1 fields (mainline_observation, cross_peer_hypotheses,
bridge_hypothesis, anti_mainline_contract, falsification_contract,
peer_contracts, success_metrics) are still required.

New v2 fields:
  agenda_version: "2.0"
  panel_mode: mini | full | high_stakes
  shared_core_id
  panel_summary
  consensus_actions
  DISSENT_TO_EXPERIMENT
  minority_high_upside (optional)
  claim_boundary_updates (optional)
  validation_status

Returns ValidationResult { valid: bool, blocking_issues: [...], warnings: [...] }.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

PI_ROLES = ("exploit", "falsifier", "bridge", "anti_mainline", "theorist")
PANEL_MODES = ("mini", "full", "high_stakes")
VALIDATION_PARENT_USAGES = {
    "ablate",
    "ablate_or_falsify",
    "ablation",
    "ablation_followup",
    "stress_validate",
    "falsify",
    "compare",
    "validate",
    "validation",
    "repair",
    "complete_validation",
    "complete_scored_validation",
    "audit",
}
_VALIDATION_PARENT_USAGE_ALIASES = {
    "ablate_or_falsify": "ablate_or_falsify",
    "ablation_followup": "ablation_followup",
    "complete_scored_validation": "complete_scored_validation",
    "complete_validation": "complete_validation",
    "stress_validate": "stress_validate",
}
VALIDATION_PARENT_IDENTITY_KEYS = (
    "finding_id",
    "variant_name",
    "variant_id",
    "frontier_entity_key",
    "candidate_entity_key",
    "child_variant_name",
    "child_variant_id",
    "source_path",
    "result_path",
    "source_result_path",
    "result_artifact_path",
    "summary_path",
)
_VALIDATION_PARENT_REF_COLLECTION_KEYS = {
    "source_findings",
    "source_finding_ids",
    "source_id",
    "source_evidence",
    "seed_findings",
    "support_findings",
    "supporting_findings",
    "supports",
    "supporting_evidence",
    "evidence_refs",
}
_VALIDATION_PARENT_REF_FIELD_KEYS = {
    *VALIDATION_PARENT_IDENTITY_KEYS,
    "id",
    "variant",
}
# R3#2 fix: narrow patterns. The catch-all `<.+?>` over-matched legitimate
# scientific notation like "<lookahead-k>", "rho<0.20", "<X, Y> pair".
# Match ONLY known placeholder keywords from the prompt schemas.
PLACEHOLDER_PATTERNS = [
    re.compile(r"<one paragraph>", re.IGNORECASE),
    re.compile(r"<exact id>", re.IGNORECASE),
    re.compile(r"<exact success signal>", re.IGNORECASE),
    re.compile(r"<combination \d+>", re.IGNORECASE),
    re.compile(r"<one sentence>", re.IGNORECASE),
    re.compile(r"<criteria>", re.IGNORECASE),
    re.compile(r"<short>", re.IGNORECASE),
    re.compile(r"<\s*id\s*>", re.IGNORECASE),
    re.compile(r"<exp_id>", re.IGNORECASE),
    re.compile(r"<claim_id>", re.IGNORECASE),
    re.compile(r"<original>", re.IGNORECASE),
    re.compile(r"<bounded>", re.IGNORECASE),
    re.compile(r"<term>", re.IGNORECASE),
    re.compile(r"<rule>", re.IGNORECASE),
    re.compile(r"<bounded version>", re.IGNORECASE),
    re.compile(r"<scope:", re.IGNORECASE),
    re.compile(r"<value>", re.IGNORECASE),
    re.compile(r"<copy forbidden_mechanisms[^>]*>", re.IGNORECASE),
    re.compile(r"<one sentence describing[^>]*>", re.IGNORECASE),
    re.compile(r"<must reference[^>]*>", re.IGNORECASE),
    re.compile(r"<query_coverage_matrix[^>]*>", re.IGNORECASE),
    re.compile(r"<at least \d+>", re.IGNORECASE),
    re.compile(r"<which Pareto[^>]*>", re.IGNORECASE),
    re.compile(r"<finding_id>", re.IGNORECASE),
    re.compile(r"<peer_id>", re.IGNORECASE),
    re.compile(r"<variant from one[^>]*>", re.IGNORECASE),
    re.compile(r"<variant from another[^>]*>", re.IGNORECASE),
    re.compile(r"<e\.g\.[^>]*>", re.IGNORECASE),
    # R5#2 fix: previous catch-all "<.{0,10}>" matched legitimate notation
    # like "<X, Y>", "<a vs b>", "rho<0.20". Replaced with a tight pattern
    # for single uppercase placeholder letters AND lowercase keyword id-style
    # that the schema actually uses ("<X>", "<id>", "<a>", "<exp>", "<short>").
    re.compile(r"<[A-Z]>"),  # <X>, <Y>
    re.compile(r"<(id|exp|short|claim|kb|H\d*|B\d*|D\d*|A\d*)>", re.IGNORECASE),
]


@dataclass
class ValidationResult:
    """Structured result of validating a PI-generated research agenda."""

    valid: bool = True
    blocking_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _is_placeholder(s: str) -> bool:
    if not isinstance(s, str):
        return False
    return any(pat.search(s) for pat in PLACEHOLDER_PATTERNS)


def _normalize_role(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    s = raw.strip().lower()
    s = re.sub(r"[\s./-]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def _normalize_required_peer_roles(raw: Any) -> tuple[str, ...]:
    if not raw:
        return PI_ROLES
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, (list, tuple, set)):
        values = list(raw)
    else:
        return PI_ROLES
    roles: list[str] = []
    for value in values:
        role = _normalize_role(value)
        if not role:
            continue
        roles.append(role)
    return tuple(roles) or PI_ROLES


def _diversity_dimension_names(raw: Any) -> tuple[str, ...]:
    """Return configured dimension names for advisory contract checks."""

    if not isinstance(raw, (list, tuple)):
        return ()
    names: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
        else:
            name = str(item or "").strip()
        if name and name not in names:
            names.append(name)
    return tuple(names)


def expected_peer_ids(next_gen_id: int, cohort_size: int) -> list[str]:
    """Return the peer ids expected for a generation and cohort size."""
    return [f"gen{next_gen_id}_peer{i}" for i in range(cohort_size)]


def _collect_known_claim_ids(
    agenda: dict[str, Any],
    pi_memos: dict[str, Any] | None = None,
) -> set:
    """Collect every claim/hypothesis/contract id mentioned in the agenda
    or in any PI memo. Used to verify that peer_contracts reference known
    targets and Chair didn't invent new claims. (R1#11 defense.)"""
    known: set = set()
    # Agenda-side ids
    for h in agenda.get("cross_peer_hypotheses") or []:
        if isinstance(h, dict) and h.get("id"):
            known.add(str(h["id"]))
    bh = agenda.get("bridge_hypothesis") or {}
    if isinstance(bh, dict) and bh.get("id"):
        known.add(str(bh["id"]))
    # Special tokens accepted as targets
    special_tokens = ["anti_mainline_contract", "falsification_contract"]
    if isinstance(bh, dict) and bh:
        special_tokens.append("bridge_hypothesis")
    for tok in special_tokens:
        known.add(tok)
    fc = agenda.get("falsification_contract") or {}
    if isinstance(fc, dict) and fc.get("target_hypothesis"):
        known.add(str(fc["target_hypothesis"]))
    for a in agenda.get("consensus_actions") or []:
        if isinstance(a, dict):
            for k in ("action_id", "claim_or_hypothesis"):
                v = a.get(k)
                if v:
                    known.add(str(v))
    for m in agenda.get("minority_high_upside") or []:
        if isinstance(m, dict) and m.get("idea_id"):
            known.add(str(m["idea_id"]))
    # Do not add claim_boundary_updates here: a boundary update must point at
    # an already-known claim/hypothesis/action, not make its own claim_id known.
    # PI memo-side ids
    for memo in (pi_memos or {}).values():
        if not isinstance(memo, dict):
            continue
        # R7#4 fix: don't silently treat _pi_unavailable stubs as
        # contributors to the known-id universe; they have empty
        # claims and would mask Chair fabrication detection.
        if memo.get("_pi_unavailable"):
            continue
        for c in memo.get("top_claims") or []:
            if isinstance(c, dict):
                for key in ("id", "claim_id"):
                    if c.get(key):
                        known.add(str(c[key]))
        for c in memo.get("proposed_peer_contracts") or []:
            if isinstance(c, dict) and c.get("target_hypothesis"):
                known.add(str(c["target_hypothesis"]))
    return known


def _validation_parent_tokens(value: Any) -> set[str]:
    text = re.sub(r"\s+", "", str(value or "").strip().lower())
    if not text:
        return set()
    tokens = {text}
    match = re.match(r"^([^:]+):{1,2}(.+)$", text)
    if match:
        prefix, payload = match.groups()
        if payload:
            tokens.add(payload)
            tokens.add(f"{prefix}:{payload}")
            tokens.add(f"{prefix}::{payload}")
    return tokens


def _validation_parent_identity_refs(container: dict[str, Any]) -> list[tuple[str, Any]]:
    refs: list[tuple[str, Any]] = []

    def collect_anchor(value: Any, label: str) -> None:
        if isinstance(value, dict):
            for key in _VALIDATION_PARENT_REF_FIELD_KEYS:
                if key in value:
                    refs.append((f"{label}.{key}", value.get(key)))
            return
        if isinstance(value, list):
            for idx, item in enumerate(value):
                collect_anchor(item, f"{label}[{idx}]")
            return
        if value not in (None, ""):
            refs.append((label, value))

    for anchor_key in ("source_anchor_A", "source_anchor_B"):
        if anchor_key in container:
            collect_anchor(container.get(anchor_key), anchor_key)
    return refs


def _normalize_parent_usage(raw: Any) -> str:
    text = re.sub(r"[\s./-]+", "_", str(raw or "").strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return _VALIDATION_PARENT_USAGE_ALIASES.get(text, text)


def _validation_parent_match(container: dict[str, Any], validation_ids: set[str]) -> str:
    parent_text = str(container.get("parent_candidate") or "").strip()
    if parent_text and (_validation_parent_tokens(parent_text) & validation_ids):
        return parent_text
    for _label, ref_value in _validation_parent_identity_refs(container):
        ref_text = str(ref_value or "").strip()
        if ref_text and (_validation_parent_tokens(ref_text) & validation_ids):
            return ref_text
    return ""


def _validation_parent_repaired_usage(container: dict[str, Any]) -> str:
    usage = _normalize_parent_usage(container.get("parent_usage"))
    if usage in VALIDATION_PARENT_USAGES:
        return usage
    if usage == "none":
        return "compare"

    intent = _normalize_parent_usage(container.get("next_step_intent"))
    stage = _normalize_parent_usage(container.get("evidence_stage"))
    role = _normalize_role(container.get("role"))
    combined = " ".join(value for value in (usage, intent, stage, role) if value)

    if "fals" in combined:
        return "falsify"
    if "ablat" in combined:
        return "ablate"
    if "audit" in combined:
        return "audit"
    if "stress" in combined:
        return "stress_validate"
    if "repair" in combined or "combine" in combined or role == "bridge":
        return "repair"
    if any(token in stage for token in ("full", "replication", "promotion")):
        return "complete_validation"
    if "valid" in combined or "preserve" in combined:
        return "validate"
    if not usage:
        return "compare"
    return "validate"


def _iter_parent_usage_containers(agenda: dict[str, Any]):
    for key in (
        "mainline_observation",
        "bridge_hypothesis",
        "anti_mainline_contract",
        "falsification_contract",
        "success_metrics",
        "panel_summary",
    ):
        value = agenda.get(key)
        if isinstance(value, dict):
            yield key, value
    for key in (
        "cross_peer_hypotheses",
        "consensus_actions",
        "DISSENT_TO_EXPERIMENT",
        "minority_high_upside",
        "claim_boundary_updates",
    ):
        value = agenda.get(key) or []
        if not isinstance(value, list):
            continue
        for idx, item in enumerate(value):
            if isinstance(item, dict):
                label_id = (
                    item.get("id")
                    or item.get("action_id")
                    or item.get("dissent_id")
                    or item.get("idea_id")
                    or item.get("claim_id")
                    or idx
                )
                yield f"{key}[{label_id}]", item
    pcs = agenda.get("peer_contracts") or {}
    if isinstance(pcs, dict):
        for pid, pc in pcs.items():
            if isinstance(pc, dict):
                yield f"peer_contract {pid}", pc


def normalize_validation_candidate_parent_usages(
    agenda: dict[str, Any],
    validation_candidate_ids: set[str] | None = None,
) -> list[dict[str, str]]:
    """Normalize validation-candidate follow-up wording before hard validation.

    PI/Chair prompts use `next_step_intent` for research intent
    (preserve/repair/combine/pivot) and `parent_usage` as a machine-readable
    maturity action. Older task prompts often conflated those vocabularies. This
    repair keeps the parent reference visible while converting validation
    candidates to validation/repair/falsification/comparison actions that the
    validator can safely reason about.
    """

    if not isinstance(agenda, dict):
        return []
    validation_ids: set[str] = set()
    for item in validation_candidate_ids or set():
        validation_ids.update(_validation_parent_tokens(item))
    if not validation_ids:
        return []

    repairs: list[dict[str, str]] = []
    for label, container in _iter_parent_usage_containers(agenda):
        parent_ref = _validation_parent_match(container, validation_ids)
        if not parent_ref:
            continue
        old_usage = str(container.get("parent_usage") or "").strip()
        normalized = _normalize_parent_usage(old_usage)
        if normalized in VALIDATION_PARENT_USAGES:
            continue
        new_usage = _validation_parent_repaired_usage(container)
        container["parent_usage"] = new_usage
        repairs.append(
            {
                "kind": "validation_candidate_parent_usage_normalized",
                "label": label,
                "parent_candidate": parent_ref,
                "old_parent_usage": old_usage,
                "new_parent_usage": new_usage,
            }
        )
    return repairs


def validate_agenda_v2(
    agenda: dict[str, Any],
    next_gen_id: int,
    cohort_size: int = 5,
    require_high_stakes_external: bool = False,
    pi_memos: dict[str, Any] | None = None,
    validation_candidate_ids: set[str] | None = None,
    required_peer_roles: Any = None,
    diversity_dimensions: Any = None,
) -> ValidationResult:
    """Validate agenda contracts, peer coverage, and role assignments before launch."""
    r = ValidationResult()

    if not isinstance(agenda, dict):
        r.valid = False
        r.blocking_issues.append("agenda is not a dict")
        return r

    # ---- v1 required fields ----
    for k in (
        "mainline_observation",
        "cross_peer_hypotheses",
        "bridge_hypothesis",
        "anti_mainline_contract",
        "falsification_contract",
        "peer_contracts",
    ):
        if k not in agenda:
            r.valid = False
            r.blocking_issues.append(f"missing required field: {k}")

    # ---- v2 new fields ----
    panel_mode = agenda.get("panel_mode")
    if panel_mode and panel_mode not in PANEL_MODES:
        r.warnings.append(f"unknown panel_mode: {panel_mode}")

    if not agenda.get("agenda_version"):
        r.warnings.append("agenda_version not set; assuming v1")

    # ---- type checks (R4-M1 pattern from prior PI) ----
    for key in (
        "mainline_observation",
        "bridge_hypothesis",
        "anti_mainline_contract",
        "falsification_contract",
    ):
        v = agenda.get(key)
        if v is not None and not isinstance(v, dict):
            r.valid = False
            r.blocking_issues.append(f"{key} must be a dict, got {type(v).__name__}")

    # ---- cross_peer_hypotheses ----
    hyps = agenda.get("cross_peer_hypotheses") or []
    if not isinstance(hyps, list):
        r.valid = False
        r.blocking_issues.append("cross_peer_hypotheses must be a list")
        hyps = []
    valid_hyps = []
    for h in hyps:
        if not isinstance(h, dict):
            r.warnings.append("cross_peer_hypotheses entry is not a dict; dropped")
            continue
        if not h.get("id"):
            r.warnings.append("hypothesis missing id; dropped")
            continue
        # Check placeholders
        for fld in ("claim", "minimal_test", "kill_condition", "promote_condition"):
            v = h.get(fld, "")
            if _is_placeholder(v):
                r.blocking_issues.append(
                    f"hypothesis {h.get('id')}: field {fld} contains placeholder text"
                )
                r.valid = False
        valid_hyps.append(h)
    if len(valid_hyps) < 3:
        r.valid = False
        r.blocking_issues.append(f"need ≥3 cross_peer_hypotheses, got {len(valid_hyps)}")

    effective_required_roles = _normalize_required_peer_roles(required_peer_roles)
    effective_required_role_set = set(effective_required_roles)
    effective_required_role_counts = Counter(effective_required_roles)
    configured_dimension_names = _diversity_dimension_names(diversity_dimensions)

    # ---- peer_contracts ----
    pcs = agenda.get("peer_contracts") or {}
    if not isinstance(pcs, dict):
        r.valid = False
        r.blocking_issues.append("peer_contracts must be a dict")
        pcs = {}

    expected = expected_peer_ids(next_gen_id, cohort_size)
    bad_keys = [k for k in pcs if k not in expected]
    missing = [k for k in expected if k not in pcs]
    if bad_keys:
        r.valid = False
        r.blocking_issues.append(
            f"peer_contracts has non-canonical keys: {bad_keys}; expected canonical {expected}"
        )
    if missing:
        r.valid = False
        r.blocking_issues.append(f"peer_contracts missing canonical peer IDs: {missing}")

    # All PI roles should appear at least once in full-panel agendas. This is
    # advisory so smaller cohorts and task-specific allocations can still run.
    role_counts: Counter[str] = Counter()
    # R1#11 fix: precompute the set of known claim/hypothesis ids so we can
    # reject peer_contracts that target inventions not in any PI memo.
    known_ids = _collect_known_claim_ids(agenda, pi_memos)
    validation_ids: set[str] = set()
    for item in validation_candidate_ids or set():
        validation_ids.update(_validation_parent_tokens(item))

    def _check_validation_parent_value(parent: Any, usage: Any, label: str) -> None:
        parent_text = str(parent or "").strip()
        if not parent_text or not (_validation_parent_tokens(parent_text) & validation_ids):
            return
        usage_text = str(usage or "").strip().lower()
        if usage_text in VALIDATION_PARENT_USAGES:
            return
        r.valid = False
        r.blocking_issues.append(
            f"{label}: parent_candidate {parent_text!r} is a validation candidate, "
            "not a durable frontier/Gems parent; use validation/repair/falsify/"
            "compare parent_usage or choose a mature parent"
        )

    def _check_validation_parent(container: dict[str, Any], label: str) -> None:
        _check_validation_parent_value(
            container.get("parent_candidate"),
            container.get("parent_usage"),
            label,
        )
        for ref_label, ref_value in _validation_parent_identity_refs(container):
            _check_validation_parent_value(
                ref_value,
                container.get("parent_usage"),
                f"{label}.{ref_label}",
            )

    def _check_validation_parent_surfaces() -> None:
        for key in (
            "mainline_observation",
            "bridge_hypothesis",
            "anti_mainline_contract",
            "falsification_contract",
            "success_metrics",
            "panel_summary",
        ):
            value = agenda.get(key)
            if isinstance(value, dict):
                _check_validation_parent(value, key)
        for key in (
            "DISSENT_TO_EXPERIMENT",
            "minority_high_upside",
            "claim_boundary_updates",
        ):
            value = agenda.get(key) or []
            if not isinstance(value, list):
                continue
            for idx, item in enumerate(value):
                if isinstance(item, dict):
                    _check_validation_parent(item, f"{key}[{item.get('id') or idx}]")

    _check_validation_parent_surfaces()

    for h in valid_hyps:
        _check_validation_parent(h, f"hypothesis {h.get('id')}")

    consensus_actions = agenda.get("consensus_actions") or []
    if isinstance(consensus_actions, list):
        for idx, action in enumerate(consensus_actions):
            if isinstance(action, dict):
                _check_validation_parent(
                    action,
                    f"consensus_actions[{action.get('action_id') or idx}]",
                )

    for pid, pc in pcs.items():
        if not isinstance(pc, dict):
            r.valid = False
            r.blocking_issues.append(f"peer_contract for {pid} must be a dict")
            continue
        norm = _normalize_role(pc.get("role"))
        if norm and norm not in effective_required_role_set:
            r.warnings.append(f"peer_contract {pid}: unknown role {pc.get('role')!r}")
        if norm:
            role_counts[norm] += 1
        # bridge contracts must reference coverage_check
        if norm == "bridge":
            blob = str(pc).lower()
            if "coverage_check" not in blob and "coverage_matrix" not in blob:
                r.warnings.append(f"bridge contract for {pid} did not document coverage_check")
        # placeholder check
        for fld in ("target_hypothesis", "success_signal"):
            v = pc.get(fld, "")
            if _is_placeholder(v):
                r.blocking_issues.append(f"peer_contract {pid}: field {fld} contains placeholder")
                r.valid = False
        if "planned_dimensions" not in pc:
            if configured_dimension_names:
                r.warnings.append(
                    f"peer_contract {pid}: optional planned_dimensions omitted for "
                    f"configured axes {list(configured_dimension_names)}"
                )
        else:
            planned = pc.get("planned_dimensions")
            if not isinstance(planned, dict):
                r.warnings.append(
                    f"peer_contract {pid}: planned_dimensions should be a flat mapping; "
                    "treating it as unavailable planning guidance"
                )
            else:
                planned_keys = {str(key).strip() for key in planned if str(key).strip()}
                if configured_dimension_names:
                    configured_keys = set(configured_dimension_names)
                    missing_dimensions = sorted(configured_keys - planned_keys)
                    unknown_dimensions = sorted(planned_keys - configured_keys)
                    if missing_dimensions:
                        r.warnings.append(
                            f"peer_contract {pid}: planned_dimensions omits configured axes "
                            f"{missing_dimensions}; planning coverage is advisory"
                        )
                    if unknown_dimensions:
                        r.warnings.append(
                            f"peer_contract {pid}: planned_dimensions includes unconfigured axes "
                            f"{unknown_dimensions}; planning coverage is advisory"
                        )
                blank_dimensions = sorted(
                    str(key)
                    for key, value in planned.items()
                    if value is None or (isinstance(value, str) and not value.strip())
                )
                nested_dimensions = sorted(
                    str(key)
                    for key, value in planned.items()
                    if isinstance(value, (dict, list, tuple, set))
                )
                if blank_dimensions:
                    r.warnings.append(
                        f"peer_contract {pid}: planned_dimensions has blank values for "
                        f"{blank_dimensions}; planning coverage is advisory"
                    )
                if nested_dimensions:
                    r.warnings.append(
                        f"peer_contract {pid}: planned_dimensions should use short scalar "
                        f"values; nested values found for {nested_dimensions}"
                    )
        for realized_field in ("design_dimensions", "realized_dimensions"):
            if realized_field in pc:
                r.warnings.append(
                    f"peer_contract {pid}: {realized_field} is realized implementation/evidence, "
                    "not PI planning; use planned_dimensions for allocation intent"
                )
        # R1#11: target_hypothesis must reference a known id (only checked
        # when pi_memos is provided; otherwise we don't know the universe).
        if pi_memos is not None:
            tgt = pc.get("target_hypothesis")
            if isinstance(tgt, str) and tgt and tgt not in known_ids:
                r.warnings.append(
                    f"peer_contract {pid}: target_hypothesis {tgt!r} not "
                    f"found in agenda hypotheses or any PI memo. "
                    f"Possible Chair fabrication."
                )
        _check_validation_parent(pc, f"peer_contract {pid}")
    panel_mode = _normalize_role(
        agenda.get("_runtime_panel_mode")
        or agenda.get("runtime_panel_mode")
        or agenda.get("panel_mode")
    )
    missing_roles: list[str] = []
    for role, required_count in effective_required_role_counts.items():
        missing_count = max(0, required_count - role_counts.get(role, 0))
        missing_roles.extend([role] * missing_count)
    if missing_roles:
        msg = f"peer_contracts missing roles: {missing_roles}"
        if panel_mode in {"full", "high_stakes"} and cohort_size >= len(effective_required_roles):
            r.valid = False
            r.blocking_issues.append(f"{msg} in {panel_mode} panel")
        else:
            r.warnings.append(f"{msg} (may be intentional for non-full panel modes)")

    # ---- claim_boundary_updates: every retired claim must have revive_if ----
    # (R1#8 fix) Distinguish "key absent" from "non-empty list". An empty
    # list `[]` represents "permanently retired with no revive conditions",
    # which is dangerous — force the author to either supply at least one
    # condition or to restate the claim status explicitly.
    boundary_updates = agenda.get("claim_boundary_updates") or []
    if isinstance(boundary_updates, list):
        for u in boundary_updates:
            if not isinstance(u, dict):
                continue
            raw_claim_id = str(u.get("claim_id") or "").strip()
            raw_id = str(u.get("id") or "").strip()
            claim_id = raw_claim_id or raw_id
            if raw_claim_id and raw_id and raw_claim_id != raw_id:
                r.valid = False
                r.blocking_issues.append(
                    "claim_boundary_update has conflicting claim_id/id "
                    f"{raw_claim_id!r} != {raw_id!r}"
                )
            elif not claim_id:
                r.valid = False
                r.blocking_issues.append("claim_boundary_update must reference a known claim_id")
            elif pi_memos is not None and claim_id not in known_ids:
                r.valid = False
                r.blocking_issues.append(
                    f"claim_boundary_update references unknown claim_id {claim_id!r}"
                )
            if (
                "obsolete" in str(u.get("old_language", "")).lower()
                or "retired" in str(u.get("new_language", "")).lower()
            ):
                rv = u.get("required_validation_before_upgrade") or []
                ri = u.get("revive_if") or []
                # Both must be non-empty lists (or strings), not just truthy.
                rv_ok = bool(rv) and (not isinstance(rv, list) or len(rv) > 0)
                ri_ok = bool(ri) and (not isinstance(ri, list) or len(ri) > 0)
                if not (rv_ok or ri_ok):
                    r.valid = False
                    r.blocking_issues.append(
                        f"claim_boundary_update for {u.get('claim_id')}: "
                        f"retired/obsolete claim must include a non-empty "
                        f"revive_if or required_validation_before_upgrade list"
                    )

    # ---- DISSENT_TO_EXPERIMENT: every entry must have resolving_experiment ----
    dissent = agenda.get("DISSENT_TO_EXPERIMENT") or []
    if isinstance(dissent, list):
        for d in dissent:
            if not isinstance(d, dict):
                continue
            if not d.get("resolving_experiment"):
                r.valid = False
                r.blocking_issues.append(
                    f"dissent {d.get('dissent_id')} missing resolving_experiment"
                )

    # ---- high-stakes panel must include external_validity in panel_summary ----
    if require_high_stakes_external or panel_mode == "high_stakes":
        ps = agenda.get("panel_summary") or {}
        if not isinstance(ps, dict) or not ps.get("external_validity_summary"):
            r.warnings.append("high_stakes panel_summary missing external_validity_summary")

    return r
