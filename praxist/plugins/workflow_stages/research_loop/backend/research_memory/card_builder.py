"""Card builder — convert raw findings into evidence cards.

Evidence cards are the unit of currency between PI panel and the raw store.
They are small (≤500 chars short interpretation), reference the raw via
relative source_ref paths, and tag claim_relevance + is_negative.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from praxist.plugins.workflow_stages.research_loop.backend.effective_config import (
    EFFECTIVE_CONFIG_METADATA_KEYS,
)

logger = logging.getLogger(__name__)


def _safe_get_metric(m: Any, key: str) -> float | None:
    """Return one finite numeric metric by key.

    Kept as a compatibility helper for callers/tests; card construction uses
    ``_safe_metric_map`` so Praxist core does not privilege any fixed task axis.
    """
    if not isinstance(m, dict):
        return None
    v = m.get(key)
    if v is None:
        return None
    if isinstance(v, dict):
        v = v.get("mean")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _safe_metric_map(metrics: Any) -> dict[str, float]:
    """Return every finite numeric metric from a finding metrics dict.

    Evidence cards are task-generic, so the card builder must not preserve
    only Praxist-core demo axes. External tasks define their own promotion lanes
    and active metrics; dropping them makes PI panels see summary-only
    evidence. Booleans and nested structures are intentionally excluded from
    this numeric map.
    """
    if not isinstance(metrics, dict):
        return {}
    out: dict[str, float] = {}
    for key, raw in metrics.items():
        if isinstance(raw, bool):
            continue
        if isinstance(raw, dict):
            raw = raw.get("mean")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value != value or value in (float("inf"), float("-inf")):
            continue
        out[str(key)] = value
    return out


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "yes", "1", "promotable", "passed"}:
            return True
        if token in {"false", "no", "0", "non-promotable", "failed"}:
            return False
    return None


def _safe_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [stripped]
    if isinstance(value, list | tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _structured_value(
    key: str,
    finding: dict[str, Any],
    metrics: dict[str, Any],
    extra: dict[str, Any],
) -> Any:
    value = extra.get(key)
    if value is None:
        value = metrics.get(key)
    if value is None:
        value = finding.get(key)
    return value


def _evidence_valence(
    finding: dict[str, Any],
    metrics: dict[str, Any],
    extra: dict[str, Any],
) -> str:
    value = _structured_value("evidence_valence", finding, metrics, extra)
    token = str(value or "").strip().lower()
    aliases = {
        "neg": "negative",
        "negative_evidence": "negative",
        "disconfirming": "negative",
        "falsifying": "negative",
        "challenge": "negative",
        "positive_evidence": "positive",
        "supportive": "positive",
        "support": "positive",
    }
    token = aliases.get(token, token)
    if token in {"positive", "neutral", "negative", "mixed"}:
        return token
    return ""


def _failure_mode(
    finding: dict[str, Any],
    metrics: dict[str, Any],
    extra: dict[str, Any],
) -> str:
    value = _structured_value("failure_mode", finding, metrics, extra)
    return str(value or "").strip()[:160]


def _disconfirming_claim_ids(
    finding: dict[str, Any],
    metrics: dict[str, Any],
    extra: dict[str, Any],
) -> list[str]:
    ids: list[str] = []
    for key in (
        "disconfirming_claim_ids",
        "disconfirmed_claim_ids",
        "challenged_claim_ids",
        "challenge_claim_ids",
    ):
        ids.extend(_safe_list(_structured_value(key, finding, metrics, extra)))
    out: list[str] = []
    seen: set[str] = set()
    for item in ids:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _negative_status_signal(
    finding: dict[str, Any],
    metrics: dict[str, Any],
    extra: dict[str, Any],
) -> bool:
    parts: list[str] = []
    for key in (
        "protocol_integrity_status",
        "result_status",
        "final_status",
        "eval_status",
        "completion_status",
        "risk_violation_reason",
        "risk_violation_type",
        "constraint_status",
    ):
        value = _structured_value(key, finding, metrics, extra)
        if value is not None:
            parts.append(str(value).lower())
    blob = " ".join(parts)
    return any(
        marker in blob
        for marker in (
            "protocol_invalid",
            "protocol_integrity_failed",
            "constraint_violation",
            "risk_violation",
            "safety_bound_failed",
            "generalization_failure",
            "mechanism_falsified",
        )
    )


def _safe_categorical_map(finding: dict[str, Any], metrics: Any, extra: Any) -> dict[str, Any]:
    categories: dict[str, Any] = {}
    if not isinstance(metrics, dict):
        metrics = {}
    if not isinstance(extra, dict):
        extra = {}
    nested_extra = extra.get("extra")
    if isinstance(nested_extra, dict):
        merged_extra = dict(extra)
        merged_extra.pop("extra", None)
        merged_extra.update(nested_extra)
        extra = merged_extra
    for key in (
        *EFFECTIVE_CONFIG_METADATA_KEYS,
        "frontier_lane",
        "source_frontier_lane",
        "promotion_lane",
        "strategy_family",
        "family",
        "variant_family",
        "benchmark_type",
        "peer_role",
        "target_hypothesis",
        "tier_status",
        "historical_promotion_status",
        "promoted_for_lane",
        "submitted_frontier_lane",
        "lane_metric_name",
        "risk_violation_reason",
        "risk_violation_type",
        "candidate_tier",
        "incubator_reason",
        "repair_reason",
        "bottleneck_target",
        "evidence_stage",
        "tradeoff_class",
        "primary_tradeoff",
        "next_step_intent",
        "parent_candidate",
        "parent_usage",
        "result_status",
        "final_status",
        "completion_status",
        "eval_status",
        "protocol_integrity_status",
        "provenance_warning",
        "source_generation_inference",
        "exclusion_reason",
        "recommended_next_step",
        "dig_expected_vs_actual_alignment",
        "dig_selected_contract_path",
        "evidence_valence",
        "failure_mode",
        "disconfirming_claim_ids",
    ):
        value = metrics.get(key)
        if value is None:
            value = finding.get(key)
        if value is None:
            value = extra.get(key)
        if isinstance(value, list):
            categories[key] = [str(item) for item in value]
        elif isinstance(value, (str, int, float, bool)) and not isinstance(value, bool):
            categories[key] = str(value)
        elif isinstance(value, bool):
            categories[key] = bool(value)
    for key in (
        "promotion_eligible",
        "clean_promotion_eligible",
        "risk_violating_frontier_candidate",
        "risk_repair_required",
        "scored_complete",
        "source_generation_low_confidence",
        "suspect_protocol",
        "excluded_from_durable_frontier",
        "partial_cohort",
        "unscored_artifact",
        "summary_only",
        "is_summary_only",
        "scout_only",
        "is_scout_eval",
        "is_smoke_eval",
        "is_negative",
    ):
        value = metrics.get(key)
        if value is None:
            value = finding.get(key)
        if value is None:
            value = extra.get(key)
        parsed = _safe_bool(value)
        if parsed is not None:
            categories[key] = parsed
    if "suspect_protocol" not in categories:
        for source in (metrics, finding, extra):
            if not isinstance(source, dict):
                continue
            parsed = _safe_bool(source.get("suspect_fixed_weight_eval"))
            if parsed is not None:
                categories["suspect_protocol"] = parsed
                break
    return categories


def _evidence_id(
    finding_id: str,
    gen: int,
    peer: str,
    seq: int = 0,
    content_seed: str = "",
) -> str:
    """Build a stable, human-readable, collision-resistant evidence_id.

    R1#14 fix: 8-char prefix of finding_id alone has birthday collisions
    around 65k findings. Combine peer + gen + finding_id and hash to
    12 chars for both human readability and ~10^14 collision threshold.

    R7#2 fix: when finding_id is empty, prefer a content-hash seed over
    a sequence counter so re-running card-build on the same finding
    produces the same evidence_id. content_seed should be a stable
    function of the finding (e.g. title + variant_name + timestamp).
    """
    import hashlib

    peer_short = (peer or "").replace("gen", "g").replace("_peer", "_p")
    if not finding_id:
        # Fall back to content-derived seed for stability across re-runs.
        seed = (
            f"{peer_short}::g{gen}::{content_seed}"
            if content_seed
            else f"{peer_short}::g{gen}::seq{seq}"
        )
        h = hashlib.blake2b(seed.encode("utf-8"), digest_size=6).hexdigest()
        return f"E_{peer_short}_x{h}"
    h = hashlib.blake2b(
        f"{peer_short}::g{gen}::{finding_id}".encode(),
        digest_size=6,
    ).hexdigest()
    # Keep readable prefix from raw id to aid debugging.
    fid_short = finding_id[:8]
    return f"E_{peer_short}_{fid_short}_{h}"


def _detect_negative(finding: dict[str, Any]) -> bool:
    """Heuristic: is this a negative/failure finding?

    R1#13 fix: don't rely solely on English string matching. Prefer
    structural signals (finding_type=challenge, peer_role=falsifier,
    explicit extra.is_negative flag), then fall back to keyword match.
    """
    # 1. Structural signal: finding_type
    ftype = (finding.get("finding_type") or "").lower()
    if ftype == "challenge":
        return True

    # 2. Structural signal: peer_role from extra
    extra = finding.get("extra") or {}
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except Exception:
            extra = {}
    metrics = finding.get("metrics") or {}
    if isinstance(metrics, str):
        try:
            metrics = json.loads(metrics)
        except Exception:
            metrics = {}
    if not isinstance(metrics, dict):
        metrics = {}

    if isinstance(extra, dict):
        if extra.get("is_negative") is True:
            return True
        if _evidence_valence(finding, metrics, extra) == "negative":
            return True
        if _disconfirming_claim_ids(finding, metrics, extra):
            return True
        failure_mode = _failure_mode(finding, metrics, extra).lower()
        if failure_mode and any(
            marker in failure_mode
            for marker in (
                "regression",
                "constraint",
                "underperformance",
                "falsified",
                "no_effect",
                "generalization",
                "protocol_invalid",
                "complexity_without_gain",
            )
        ):
            return True
        if _negative_status_signal(finding, metrics, extra):
            return True
        peer_role = str(extra.get("peer_role", "")).lower()
        if peer_role == "falsifier":
            # Falsifier outputs that conclude KILL/CHALLENGE are negative.
            # Be conservative: still gate on language because falsifier can
            # also conclude KEEP.
            pass

    # 3. Fall back to English keyword match (acknowledged locale-fragile;
    # caller can supplement extra.is_negative=true for non-English findings).
    title = (finding.get("title") or "").lower()
    notes = (finding.get("notes") or "").lower()
    content = (finding.get("content") or "").lower()
    neg_markers = (
        "kill",
        "fail",
        "underperform",
        "dominated",
        "no improvement",
        "no synergy",
        "additive",
        "rejected",
        "not synergistic",
        "downgrade",
        "obsolete",
        "did not",
        "does not",
        "regress",
    )
    blob = title + " " + notes + " " + content[:1200]
    return any(mk in blob for mk in neg_markers)


def _derive_claim_relevance(
    claim_relevance: dict[str, list[str]] | None,
    finding: dict[str, Any],
    metrics: dict[str, Any],
    extra: dict[str, Any],
    *,
    is_negative: bool,
) -> dict[str, list[str]]:
    rel = {
        "supports": list((claim_relevance or {}).get("supports") or []),
        "challenges": list((claim_relevance or {}).get("challenges") or []),
        "informs": list((claim_relevance or {}).get("informs") or []),
    }
    for claim_id in _disconfirming_claim_ids(finding, metrics, extra):
        if claim_id not in rel["challenges"]:
            rel["challenges"].append(claim_id)
    target = str(_structured_value("target_hypothesis", finding, metrics, extra) or "").strip()
    if target:
        bucket = "challenges" if is_negative else "informs"
        if target not in rel[bucket]:
            rel[bucket].append(target)
    return rel


def build_card_from_finding(
    finding: dict[str, Any],
    run_dir: Path,
    claim_relevance: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Build a single evidence card from a finding dict.

    finding dict shape (from shared_store.db or shared_findings JSON):
      id, finding_type, title, content, metrics (dict), variant_name,
      notes, peer_id, generation_id, timestamp, extra
    """
    finding_id = finding.get("id") or finding.get("finding_id") or ""
    peer_id = finding.get("peer_id") or ""
    gen_id = int(finding.get("generation_id") or 0)
    variant = finding.get("variant_name") or ""
    metrics = finding.get("metrics") or {}
    if isinstance(metrics, str):
        try:
            metrics = json.loads(metrics)
        except Exception:
            metrics = {}

    # Find the relative path to a candidate raw artifact.
    rel_finding_path = None
    if finding_id and variant:
        candidate = Path(run_dir) / "shared_findings" / f"{finding_id}_{variant}.json"
        if candidate.exists():
            rel_finding_path = str(candidate.relative_to(run_dir))
    if rel_finding_path is None and finding_id:
        # fallback: glob for prefix match
        try:
            for p in (Path(run_dir) / "shared_findings").glob(f"{finding_id}_*.json"):
                rel_finding_path = str(p.relative_to(run_dir))
                break
        except Exception:
            pass

    short_title = (finding.get("title") or "")[:200]
    short_notes = (finding.get("notes") or "")[:300]
    short_content = (finding.get("content") or "")[:300]
    interpretation_short = short_title
    if short_notes and short_notes not in short_title:
        interpretation_short = (short_title + " — " + short_notes)[:500]
    elif not interpretation_short and short_content:
        interpretation_short = short_content[:500]

    extra = finding.get("extra") or {}
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except Exception:
            extra = {}

    peer_role = extra.get("peer_role", "") if isinstance(extra, dict) else ""

    metric_payload = _safe_metric_map(metrics)
    metric_payload.update(
        _safe_categorical_map(finding, metrics if isinstance(metrics, dict) else {}, extra)
    )
    promotion_value = metrics.get("promotion_eligible") if isinstance(metrics, dict) else None
    if promotion_value is None:
        promotion_value = finding.get("promotion_eligible")
    tier_value = metrics.get("tier") if isinstance(metrics, dict) else None
    if tier_value is None:
        tier_value = finding.get("tier")
    promotion_bool = _safe_bool(promotion_value)
    if promotion_bool is not None:
        metric_payload["promotion_eligible"] = promotion_bool
    # R8#1 fix: stringify tier only when explicitly present. If a malformed
    # finding has tier as a nested dict, str() makes it serializable rather
    # than letting a dict slip into downstream JSON paths.
    if tier_value is not None:
        metric_payload["tier"] = str(tier_value)
    if isinstance(metrics, dict) and isinstance(metrics.get("seed_count"), (int, float)):
        metric_payload["seed_count"] = int(metrics["seed_count"])
    finding_for_detection = {**finding, "metrics": metrics, "extra": extra}
    is_negative = _detect_negative(finding_for_detection)
    valence = _evidence_valence(finding, metrics if isinstance(metrics, dict) else {}, extra)
    failure_mode = _failure_mode(finding, metrics if isinstance(metrics, dict) else {}, extra)
    disconfirming_claims = _disconfirming_claim_ids(
        finding, metrics if isinstance(metrics, dict) else {}, extra
    )
    claim_relevance_payload = _derive_claim_relevance(
        claim_relevance,
        finding,
        metrics if isinstance(metrics, dict) else {},
        extra,
        is_negative=is_negative,
    )
    quality_payload: dict[str, Any] = {
        "duplicate_risk": "low",
        "is_negative": is_negative,
        "is_retired": False,
    }
    if valence:
        quality_payload["evidence_valence"] = valence
    if failure_mode:
        quality_payload["failure_mode"] = failure_mode
    if disconfirming_claims:
        quality_payload["disconfirming_claim_ids"] = disconfirming_claims

    source_ref = {
        "finding_id": finding_id,
        "finding_path": rel_finding_path or "",
        "variant_name": variant,
        "generation_id": gen_id,
        "peer_id": peer_id,
        "tier": (metrics.get("tier") if isinstance(metrics, dict) else "") or "",
        "peer_role": peer_role,
    }
    source_result_path = str(
        (metrics.get("source_result_path") if isinstance(metrics, dict) else "")
        or finding.get("source_result_path")
        or ""
    )
    if source_result_path and any(key in metric_payload for key in EFFECTIVE_CONFIG_METADATA_KEYS):
        source_ref["source_result_path"] = source_result_path

    card = {
        "evidence_id": _evidence_id(finding_id, gen_id, peer_id),
        "source_type": finding.get("finding_type") or "unknown",
        "source_ref": source_ref,
        "claim_relevance": claim_relevance_payload,
        "metrics": metric_payload,
        "interpretation": {
            "short": interpretation_short,
            "uncertainty": "",
        },
        "quality": quality_payload,
        "created_at": finding.get("timestamp") or datetime.now(UTC).isoformat(timespec="seconds"),
        "created_by_gen": gen_id,
    }
    return card


def build_cards_from_db(
    run_dir: Path,
    db_path: Path | None = None,
    only_gen: int | None = None,
    max_gen: int | None = None,
) -> list[dict[str, Any]]:
    """Bulk-build cards from an existing shared_store.db (used in shadow mode)."""
    if db_path is None:
        db_path = Path(run_dir) / "shared_store.db"
    if not Path(db_path).exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            q = (
                "SELECT id, finding_type, title, content, metrics, "
                "variant_name, notes, peer_id, generation_id, timestamp, extra "
                "FROM findings"
            )
            params: tuple = ()
            if only_gen is not None:
                q += " WHERE generation_id = ?"
                params = (int(only_gen),)
            elif max_gen is not None:
                q += " WHERE generation_id <= ?"
                params = (int(max_gen),)
            cur = conn.execute(q, params)
            for row in cur:
                d = dict(row)
                # parse JSON-text fields
                for jk in ("metrics", "extra"):
                    if isinstance(d.get(jk), str):
                        try:
                            d[jk] = json.loads(d[jk])
                        except Exception:
                            d[jk] = {}
                out.append(build_card_from_finding(d, run_dir))
    except Exception as e:
        logger.warning("card_builder.build_cards_from_db: %s", e)
    return out
