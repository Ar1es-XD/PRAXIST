"""
Finding Graph rule engine + health metrics.

Implements the sidecar graph index over shared findings specified in the
finding-graph section of ``docs/concepts/architecture.md``.

Philosophy (quoting the doc):
  - edges are navigation, not conclusions
  - conservative first: prefer related_to over strong edges when unsure
  - do not merge nodes — originals remain source of truth
  - rules only; no LLM calls in v1

Runtime: deterministic Python. Safe to run in shadow mode alongside the
orchestrator without affecting findings writes or frontier behavior.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from praxist.plugins.workflow_stages.research_loop.backend.event_wait import (
    wait_for_filesystem_event,
)

logger = logging.getLogger(__name__)


# --- ID detection -----------------------------------------------------------
# finding ids in this codebase come in two shapes:
#   - UUID v4 from share_finding MCP path: 84d85198-d1b1-4290-a8c0-0b667a4ef593
#   - deterministic hash from Write-tool path: fs_<sha256[:32]> (32 hex chars)
_UUID_RX = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_FS_ID_RX = re.compile(r"\bfs_[0-9a-f]{32}\b", re.IGNORECASE)


def _extract_referenced_ids(text: str) -> list[str]:
    """Return all finding-id-shaped substrings in ``text``, deduped."""
    if not text:
        return []
    seen = set()
    out = []
    for m in _UUID_RX.findall(text):
        k = m.lower()
        if k not in seen:
            seen.add(k)
            out.append(m)
    for m in _FS_ID_RX.findall(text):
        k = m.lower()
        if k not in seen:
            seen.add(k)
            out.append(m)
    return out


# --- Language cues ----------------------------------------------------------
# EN + ZH keyword sets per design doc §6.2. Matched case-insensitively
# against the text blob; ZH handled by direct substring match since Chinese
# has no word boundaries in the same sense.

_CHALLENGE_EN = (
    "fail",
    "failed",
    "failure",
    "contradict",
    "contradiction",
    "wrong",
    "invalid",
    "not reproduce",
    "cannot reproduce",
    "bug",
    "error",
    "regression",
    "refuted",
    "disproven",
    "falsified",
    "negative result",
    "does not work",
    "doesn't work",
)
_CHALLENGE_ZH = (
    "失败",
    "冲突",
    "错误",
    "无效",
    "不能复现",
    "反例",
    "回退",
    "未复现",
    "证伪",
    "推翻",
    "负结果",
)

_SUPPORT_EN = (
    "confirm",
    "confirmed",
    "validate",
    "validated",
    "reproduce",
    "reproduced",
    "replicated",
    "consistent with",
    "supports",
    "agrees with",
    "independently verifies",
)
_SUPPORT_ZH = (
    "确认",
    "验证",
    "复现",
    "一致",
    "支持",
    "再现",
)


def _has_any(blob_lower: str, en_set, zh_set=()) -> bool:
    """Return True if any English keyword appears with word boundaries,
    or any Chinese keyword appears as a substring. Word-boundary
    matching on English avoids false positives like `"unconfirmed"`
    triggering on the keyword `"confirm"`."""
    for kw in en_set:
        if re.search(rf"\b{re.escape(kw)}\b", blob_lower):
            return True
    # Chinese has no word boundaries in the same sense; substring is
    # the conventional match.
    return any(kw in blob_lower for kw in zh_set)


# Polarity-negation checks. Substring keyword matching is brittle for
# opposing-polarity phrases:
#   - "consistent with X" → supports
#   - "not consistent with X" → should NOT count as supports
#   - "一致" (consistent) → supports
#   - "不一致" (inconsistent) → should NOT count as supports
#
# We match English negations with a flexible regex that allows 0-3
# filler words between "not" and the target (catches "not yet
# confirmed", "does not clearly confirm" etc.) and applies word
# boundaries so that "unconfirmed" is ALSO caught (via the
# morphological-negation list below) without requiring a matching
# "not" particle.
#
# Chinese negations stay as substring because "不" / "未" / "无" are
# single-char negators directly adjacent to the keyword in typical
# phrasing; regex word boundaries don't apply to CJK.
_NEGATION_RX_EN = (
    # "not X" / "never X" / "nothing X" with up to 3 words between
    re.compile(
        r"\b(?:not|never|nothing)\s+(\w+\s+){0,3}"
        r"(consistent|reproduced|supported|confirmed|validated|"
        r"replicated|aligned|agreeing)\b"
    ),
    # "can't" / "cannot" / "can not" reproduce|confirm|...
    re.compile(
        r"\b(?:can(?:no|\s+no)?t|can't)\s+(\w+\s+){0,3}"
        r"(reproduce|confirm|validate|replicate|support)\b"
    ),
    # "does not" / "doesn't" / "do not" / "don't" ...
    re.compile(
        r"\b(?:does\s+not|doesn't|do\s+not|don't)\s+(\w+\s+){0,3}"
        r"(reproduce|confirm|validate|replicate|support)\b"
    ),
    # "failed to" / "fails to"
    re.compile(
        r"\bfail(?:ed|s)?\s+to\s+(\w+\s+){0,3}"
        r"(reproduce|confirm|validate|replicate|support)\b"
    ),
    # "reject" / "rejects" / "rejected" the/this/our hypothesis/claim
    re.compile(r"\breject(?:ed|s|ing)?\b"),
    # Morphological negations — one word that encodes the negation.
    # These DO NOT match as substrings of the positive keyword; they
    # are whole words themselves, so we word-boundary them.
    # `non-?` catches "non-confirmed" and "nonconfirmed"; a separate
    # two-word regex catches "non supported" (space-separated).
    re.compile(
        r"\b(?:un|dis|non-?)(?:confirmed|validated|supported|"
        r"reproduced|replicated)\b"
    ),
    re.compile(r"\bnon\s+(?:confirmed|validated|supported|reproduced|replicated)\b"),
    re.compile(r"\binconsistent\b|\bincompatible\b|\bnegative\s+result\b"),
    re.compile(r"\brefut(?:ed|es|ing)\b|\bfalsif(?:ied|ies)\b|\bdisprov(?:ed|en)\b"),
)
# Chinese negation patterns — kept as substring matching (CJK has no
# word boundaries in regex terms). De-duplicated against semantic
# equivalents: 无法复现 / 不能复现 both mean "cannot reproduce", we
# keep both so findings using either phrasing are caught.
_NEGATION_PATTERNS_ZH = (
    "不一致",
    "未一致",
    "未复现",
    "未能复现",
    "无法复现",
    "不能复现",
    "不再现",
    "未再现",
    "不支持",
    "未支持",
    "不确认",
    "未确认",
    "不验证",
    "未验证",
    "不符合",
)


def _has_any_non_negated(blob_lower: str, en_set, zh_set=()) -> bool:
    """Like _has_any but rejects matches where the keyword is part of
    an explicit negation. Used by Rule 4 (supports) to avoid firing on
    `"not consistent"` / `"不一致"` / `"unconfirmed"`. Rule 5
    (challenges) uses plain _has_any because its own keywords (fail /
    失败) don't have the same polarity-flipping substring problem.

    Note: this is coarse-grained. A finding with BOTH a valid supports
    keyword AND an unrelated negation (`"we confirmed X, but Y is not
    supported"`) has its supports edge suppressed. The conservative
    direction is correct — a mixed-polarity finding should not emit a
    confident supports edge. Agents can still declare explicit links
    via the `links` argument when they want a specific direction.
    """
    for neg_rx in _NEGATION_RX_EN:
        if neg_rx.search(blob_lower):
            return False
    for neg in _NEGATION_PATTERNS_ZH:
        if neg in blob_lower:
            return False
    return _has_any(blob_lower, en_set, zh_set)


def _normalize_title_tokens(title: str) -> set:
    """Extract capitalized variant-like tokens from a title for rough matching."""
    if not title:
        return set()
    return {
        m.group(1).upper()
        for m in re.finditer(r"\b([A-Z][A-Za-z0-9]*(?:-[A-Z][A-Za-z0-9]*)+)\b", title)
    }


# --- Main rule engine -------------------------------------------------------


class FindingGraphBuilder:
    """Stateless rule engine.

    Usage:
        builder = FindingGraphBuilder(all_findings)
        for f in builder.chronological():
            proposed = builder.propose_edges_for(f)
            # insert_edges_batch(proposed)

    One-shot backfill:
        edges = builder.build_all_edges()
    """

    MIN_CONFIDENCE = 0.55

    # Per-rule cap on number of prior findings a new finding can connect to.
    # Without this, a single "CONFIRMED: VARIANT-X" insight fans out
    # to every prior matching finding in the corpus — on a large run
    # this produced 48k edges from noise. The cap keeps the graph navigable
    # while preserving the signal: the N most-recent prior findings on the
    # same object are the ones most likely to be what the new finding is
    # actually commenting on.
    MAX_PRIORS_PER_RULE = 5

    # Rule 2 (agent-declared links) has its own cap — an agent that
    # declares 10,000 links in a single share_finding call would
    # otherwise fan out uncontrollably. Slightly higher than the
    # per-rule cap because agent-declared links are intentional
    # signals, not inferred from text.
    MAX_LINKS_PER_FINDING = 20

    def __init__(self, findings: list[dict[str, Any]]):
        self.findings = list(findings)
        # index by id
        self.by_id: dict[str, dict[str, Any]] = {f["id"]: f for f in self.findings if f.get("id")}
        # index by variant_name (normalized)
        self._by_variant: dict[str, list[dict[str, Any]]] = {}
        for f in self.findings:
            v = (f.get("variant_name") or "").strip()
            if not v:
                continue
            self._by_variant.setdefault(self._norm_variant(v), []).append(f)
        # index by title-token set
        self._by_title_token: dict[str, list[dict[str, Any]]] = {}
        for f in self.findings:
            toks = _normalize_title_tokens(f.get("title", ""))
            for t in toks:
                self._by_title_token.setdefault(t, []).append(f)

    @staticmethod
    def _norm_variant(v: str) -> str:
        """Normalize variant_name for comparison — lowercase, collapse
        whitespace, drop common hyperparam tails like 'alpha=0.3' so that
        'VARIANT-X' and 'VARIANT-X alpha=0.3' match as the same variant."""
        v = v.lower().strip()
        # drop trailing hyperparam annotations
        v = re.split(r"\s+(alpha|α|rho|ρ|lmax|lmin|gamma|γ)\s*=", v)[0]
        v = re.sub(r"\s+", " ", v)
        return v.strip()

    def chronological(self) -> list[dict[str, Any]]:
        """Yield findings in timestamp-ascending order (stable on ties by id)."""
        return sorted(
            self.findings,
            key=lambda f: (f.get("timestamp", ""), f.get("id", "")),
        )

    @staticmethod
    def _is_older(other: dict[str, Any], new_finding: dict[str, Any]) -> bool:
        """Is `other` strictly older than `new_finding` for edge purposes?
        Use (timestamp, id) tuple comparison so same-timestamp findings
        still resolve a direction (lex order on id) — plain `<` on
        timestamp alone dropped every edge between peers that shared at
        the same second."""
        a = (other.get("timestamp", ""), other.get("id", ""))
        b = (new_finding.get("timestamp", ""), new_finding.get("id", ""))
        return a < b

    # -- rules ----------------------------------------------------------------

    def propose_edges_for(self, new_finding: dict[str, Any]) -> list[dict[str, Any]]:
        """Run all rules against ``new_finding``, return resolved edge list.

        The returned edges always have ``src_finding_id = new_finding["id"]``
        (i.e. new → old; see design §3.1).
        """
        nid = new_finding.get("id")
        if not nid:
            return []

        proposed: list[dict[str, Any]] = []

        # Rule 1: explicit id reference in content/notes/extra
        proposed.extend(self._rule1_explicit_id_ref(new_finding))

        # Rule 2: explicit links field
        proposed.extend(self._rule2_explicit_links(new_finding))

        # Rule 3: same variant_name / time series
        proposed.extend(self._rule3_same_variant_time_series(new_finding))

        # Rule 4: supports language + shared variant/title token
        proposed.extend(self._rule4_supports(new_finding))

        # Rule 5: challenges language + shared variant/title token
        proposed.extend(self._rule5_challenges(new_finding))

        # Rule 6: weak related_to on shared variant or title tokens
        proposed.extend(self._rule6_related_to(new_finding))

        # Resolve conflicts per design §6.3
        return self._resolve(proposed, nid)

    # -- rule 1: explicit id reference ---------------------------------------

    def _rule1_explicit_id_ref(self, new_finding: dict[str, Any]) -> list[dict[str, Any]]:
        nid = new_finding["id"]
        text = self._text_blob(new_finding)
        refs = _extract_referenced_ids(text)
        out = []
        blob_lower = text.lower()
        is_challenge = _has_any(blob_lower, _CHALLENGE_EN, _CHALLENGE_ZH)
        for rid in refs:
            if rid == nid:
                continue
            if rid not in self.by_id:
                continue
            target = self.by_id[rid]
            # skip self-refs on same timestamp (e.g. UUID in own content block)
            if not self._is_older(target, new_finding):
                # refs only go new → strictly-older; tuple compare with
                # id tiebreaks same-timestamp cases.
                continue
            if is_challenge:
                out.append(
                    self._mk_edge(
                        nid,
                        rid,
                        "challenges",
                        0.90,
                        rationale=f"rule1: new finding content references {rid} and uses challenge language",
                        created_by="rule_engine",
                        provenance={"rule": 1, "matched_id": rid, "lang_cue": "challenge"},
                    )
                )
            else:
                out.append(
                    self._mk_edge(
                        nid,
                        rid,
                        "derived_from",
                        0.95,
                        rationale=f"rule1: new finding content explicitly references {rid}",
                        created_by="rule_engine",
                        provenance={"rule": 1, "matched_id": rid},
                    )
                )
        return out

    # -- rule 2: explicit links field -----------------------------------------

    # Whitelist for agent-declared edge types. An agent can still supply
    # adversarial `target_finding_id` or wrong `rationale` text, but the
    # edge_type at least stays in the known-good set — otherwise an
    # invalid value either gets silently dropped by `_resolve` (if it
    # classifies as neither strong nor weak) or, worse, inserted into
    # SQLite as a junk edge_type that consumers don't expect.
    _VALID_AGENT_EDGE_TYPES = frozenset(
        {
            "related_to",
            "derived_from",
            "updates",
            "supports",
            "challenges",
        }
    )

    def _rule2_explicit_links(self, new_finding: dict[str, Any]) -> list[dict[str, Any]]:
        nid = new_finding["id"]
        links = new_finding.get("links") or []
        if isinstance(links, str):
            try:
                links = json.loads(links)
            except (json.JSONDecodeError, TypeError):
                return []
        if not isinstance(links, list):
            return []
        # Cap the number of declared links per share_finding call.
        # Rules 3-6 all use MAX_PRIORS_PER_RULE; without the same cap
        # here, an agent (intentionally or by bug) could declare
        # thousands of links in one call and blow up the edge table.
        if len(links) > self.MAX_LINKS_PER_FINDING:
            logger.warning(
                "rule2: finding %s declared %d links; capping to %d",
                new_finding.get("id", "?"),
                len(links),
                self.MAX_LINKS_PER_FINDING,
            )
            links = links[: self.MAX_LINKS_PER_FINDING]
        out = []
        for link in links:
            if not isinstance(link, dict):
                continue
            target = link.get("target_finding_id")
            etype = link.get("edge_type", "related_to")
            # Validate the type — fall through to related_to for unknown
            # values so the link survives as weak navigation rather than
            # being silently dropped.
            if etype not in self._VALID_AGENT_EDGE_TYPES:
                logger.debug(
                    "rule2: unknown edge_type %r for finding %s → coerced to related_to",
                    etype,
                    nid,
                )
                etype = "related_to"
            if not target or target not in self.by_id:
                continue
            if target == nid:
                continue
            rationale = link.get("rationale", f"rule2: agent-declared {etype}")
            out.append(
                self._mk_edge(
                    nid,
                    target,
                    etype,
                    0.90,
                    rationale=rationale,
                    created_by="agent_declared",
                    provenance={"rule": 2, "source": "share_finding.links"},
                )
            )
        return out

    # -- rule 3: same variant_name, time series --------------------------------

    def _rule3_same_variant_time_series(self, new_finding: dict[str, Any]) -> list[dict[str, Any]]:
        nid = new_finding["id"]
        v = (new_finding.get("variant_name") or "").strip()
        if not v:
            return []
        key = self._norm_variant(v)
        matches = self._by_variant.get(key, [])
        if not matches:
            return []

        new_finding.get("timestamp", "")
        ftype = new_finding.get("finding_type", "")
        is_weak = ftype in ("hypothesis", "insight")

        # Take the N most recent prior findings on this variant. Use
        # heapq.nlargest to cap BEFORE a full sort — a variant with 2000
        # matches was sorting all 2000 just to slice the top 5.
        import heapq

        priors_iter = (m for m in matches if m["id"] != nid and self._is_older(m, new_finding))
        priors = heapq.nlargest(
            self.MAX_PRIORS_PER_RULE,
            priors_iter,
            key=lambda f: (f.get("timestamp", ""), f.get("id", "")),
        )

        out = []
        new_peer = (new_finding.get("peer_id") or "").strip()
        for old in priors:
            old_peer = (old.get("peer_id") or "").strip()
            # Cross-peer findings with the same variant are INDEPENDENT
            # measurements, not "new updates older". Rule 3's `updates`
            # edge type implies continuity of an experimental thread,
            # which only applies when one peer is revising its own prior
            # result. When a sibling peer publishes the same variant,
            # demote to related_to — the relationship is real (shared
            # object) but doesn't carry the "supersedes" semantics that
            # `updates` implies.
            is_cross_peer = bool(new_peer) and bool(old_peer) and (new_peer != old_peer)
            if is_weak or is_cross_peer:
                label = "cross-peer" if is_cross_peer else ftype
                out.append(
                    self._mk_edge(
                        nid,
                        old["id"],
                        "related_to",
                        0.65,
                        rationale=f"rule3: same variant '{v}'; {label}",
                        created_by="rule_engine",
                        provenance={
                            "rule": 3,
                            "variant": v,
                            "weak": True,
                            "cross_peer": is_cross_peer,
                        },
                    )
                )
            else:
                out.append(
                    self._mk_edge(
                        nid,
                        old["id"],
                        "updates",
                        0.70,
                        rationale=f"rule3: same variant '{v}'; new ({ftype}) updates older record (same peer)",
                        created_by="rule_engine",
                        provenance={"rule": 3, "variant": v, "cross_peer": False},
                    )
                )
        return out

    # -- rule 4: supports language ---------------------------------------------

    def _rule4_supports(self, new_finding: dict[str, Any]) -> list[dict[str, Any]]:
        blob_lower = self._text_blob(new_finding).lower()
        # Polarity-aware: "不一致" / "not consistent" should NOT fire
        # rule 4 even though they contain the substrings "一致" /
        # "consistent". Rule 5 (challenges) picks up the negation
        # separately via its own keyword set.
        if not _has_any_non_negated(blob_lower, _SUPPORT_EN, _SUPPORT_ZH):
            return []
        return self._language_cued_edges(
            new_finding,
            "supports",
            0.75,
            rule_num=4,
            lang_cue="support",
        )

    # -- rule 5: challenges language -------------------------------------------

    def _rule5_challenges(self, new_finding: dict[str, Any]) -> list[dict[str, Any]]:
        blob_lower = self._text_blob(new_finding).lower()
        if not _has_any(blob_lower, _CHALLENGE_EN, _CHALLENGE_ZH):
            return []
        return self._language_cued_edges(
            new_finding,
            "challenges",
            0.75,
            rule_num=5,
            lang_cue="challenge",
        )

    def _language_cued_edges(
        self,
        new_finding: dict[str, Any],
        etype: str,
        conf: float,
        rule_num: int,
        lang_cue: str,
    ) -> list[dict[str, Any]]:
        """Rules 4 & 5 share a scaffold: language cue present, plus shared
        variant_name OR shared title token with an earlier finding."""
        import heapq

        nid = new_finding["id"]
        candidates = self._shared_object_candidates(new_finding, exclude_id=nid)
        # Cap to N most recent priors. heapq.nlargest avoids the full
        # sort — `_shared_object_candidates` can return hundreds of
        # findings for a hot variant / popular title token; we only
        # need the top 5.
        priors = heapq.nlargest(
            self.MAX_PRIORS_PER_RULE,
            (c for c in candidates if self._is_older(c, new_finding)),
            key=lambda f: (f.get("timestamp", ""), f.get("id", "")),
        )
        out = []
        for old in priors:
            out.append(
                self._mk_edge(
                    nid,
                    old["id"],
                    etype,
                    conf,
                    rationale=(
                        f"rule{rule_num}: {lang_cue} language in new finding + "
                        f"shared object '{old.get('variant_name') or self._shared_token(new_finding, old)}'"
                    ),
                    created_by="rule_engine",
                    provenance={"rule": rule_num, "lang_cue": lang_cue},
                )
            )
        return out

    # -- rule 6: weak related_to -----------------------------------------------

    def _rule6_related_to(self, new_finding: dict[str, Any]) -> list[dict[str, Any]]:
        nid = new_finding["id"]
        import heapq

        candidates = self._shared_object_candidates(new_finding, exclude_id=nid)
        priors = heapq.nlargest(
            self.MAX_PRIORS_PER_RULE,
            (c for c in candidates if self._is_older(c, new_finding)),
            key=lambda f: (f.get("timestamp", ""), f.get("id", "")),
        )
        out = []
        for old in priors:
            # shared title tokens → slightly higher; metrics-keys-only → lower
            tok = self._shared_token(new_finding, old)
            conf = 0.62 if tok else 0.57
            out.append(
                self._mk_edge(
                    nid,
                    old["id"],
                    "related_to",
                    conf,
                    rationale=f"rule6: shared object/variant '{tok or old.get('variant_name', '')}'",
                    created_by="rule_engine",
                    provenance={"rule": 6, "token": tok or None},
                )
            )
        return out

    # -- helpers ---------------------------------------------------------------

    def _shared_object_candidates(
        self,
        new_finding: dict[str, Any],
        exclude_id: str,
    ) -> list[dict[str, Any]]:
        """Prior findings sharing either variant_name (normalized) or a
        title-level variant token with ``new_finding``."""
        out = []
        seen = set()
        v = (new_finding.get("variant_name") or "").strip()
        if v:
            for m in self._by_variant.get(self._norm_variant(v), []):
                if m["id"] != exclude_id and m["id"] not in seen:
                    seen.add(m["id"])
                    out.append(m)
        toks = _normalize_title_tokens(new_finding.get("title", ""))
        for t in toks:
            for m in self._by_title_token.get(t, []):
                if m["id"] != exclude_id and m["id"] not in seen:
                    seen.add(m["id"])
                    out.append(m)
        return out

    def _shared_token(self, a: dict[str, Any], b: dict[str, Any]) -> str | None:
        ta = _normalize_title_tokens(a.get("title", ""))
        tb = _normalize_title_tokens(b.get("title", ""))
        inter = ta & tb
        # Deterministic choice — `next(iter(set))` is hash-order and
        # varies across Python versions, which makes edge rationales
        # non-reproducible.
        return min(inter) if inter else None

    @staticmethod
    def _text_blob(f: dict[str, Any]) -> str:
        parts = [
            str(f.get("title", "")),
            str(f.get("content", "")),
            str(f.get("notes", "")),
        ]
        # include extra values as text (ids may live in nested extra)
        extra = f.get("extra") or {}
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except (json.JSONDecodeError, TypeError):
                extra = {}
        if isinstance(extra, dict):
            for v in extra.values():
                if isinstance(v, (str, int, float)):
                    parts.append(str(v))
                elif isinstance(v, (list, dict)):
                    # ensure_ascii=False keeps Chinese characters as-is
                    # in the blob. Without this, `json.dumps` escapes
                    # "失败" to "\u5931\u8d25", and rule 4/5 ZH keyword
                    # matching silently fails for every finding whose
                    # language cue lives inside `extra`.
                    parts.append(json.dumps(v, default=str, ensure_ascii=False))
        return " ".join(parts)

    @staticmethod
    def _mk_edge(
        src: str,
        dst: str,
        etype: str,
        conf: float,
        rationale: str,
        created_by: str,
        provenance: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "edge_id": str(uuid.uuid4()),
            "src_finding_id": src,
            "dst_finding_id": dst,
            "edge_type": etype,
            "confidence": float(conf),
            "created_by": created_by,
            "created_at": datetime.now(UTC).isoformat(),
            "rationale": rationale,
            "provenance": provenance,
        }

    # -- conflict resolution ---------------------------------------------------

    _STRONG_TYPES = ("derived_from", "updates", "supports", "challenges")

    def _resolve(
        self,
        proposed: list[dict[str, Any]],
        nid: str,
    ) -> list[dict[str, Any]]:
        """Apply design §6.3 conflict policy.

        For each (src, dst) pair:
          - Keep the single highest-confidence STRONG edge.
          - Keep a single ``related_to`` edge (highest-conf) alongside the strong
            one if present; it's harmless navigation.
          - If no strong edge exists, keep the top ``related_to``.
        """
        by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for e in proposed:
            key = (e["src_finding_id"], e["dst_finding_id"])
            by_pair.setdefault(key, []).append(e)

        out = []
        for (src, dst), edges in by_pair.items():
            if src == dst:
                continue  # defensive; never self-loop
            strong = [e for e in edges if e["edge_type"] in self._STRONG_TYPES]
            weak = [e for e in edges if e["edge_type"] == "related_to"]
            if strong:
                # Agent-declared edges (rule 2, created_by="agent_declared")
                # represent the agent's EXPLICIT semantic intent. A mechanical
                # rule 1 `derived_from` at conf 0.95 should not silently
                # override an agent-declared `supports` at conf 0.90 — the
                # agent knows what they meant and `get_finding_neighbors`
                # returning a different edge type than what they declared is
                # a confusing MCP round-trip. Sort key: prefer agent_declared
                # first, then by confidence.
                strong.sort(
                    key=lambda e: (
                        1 if e.get("created_by") == "agent_declared" else 0,
                        e["confidence"],
                    ),
                    reverse=True,
                )
                out.append(strong[0])
                # additionally keep a related_to if present (navigation breadcrumb)
                if weak:
                    weak.sort(key=lambda e: e["confidence"], reverse=True)
                    out.append(weak[0])
            elif weak:
                weak.sort(key=lambda e: e["confidence"], reverse=True)
                out.append(weak[0])
        return out

    # -- backfill entry --------------------------------------------------------

    def build_all_edges(self) -> list[dict[str, Any]]:
        """Iterate all findings in chronological order and propose edges.

        The edges are NOT deduped against SQLite here — caller should use
        ``insert_edges_batch`` which silently skips duplicates via UNIQUE.
        """
        out = []
        for f in self.chronological():
            out.extend(self.propose_edges_for(f))
        return out


# --- Graph health report ----------------------------------------------------

# Module-level observability state, guarded by _STATE_LOCK. Both maps
# are read by compute_graph_health() and written from multiple threads
# (peer prompt renders, maintainer cycle, CLI health checks). Without
# the lock, the read-modify-write in _record_session_failure can lose
# increments under concurrent peer spawns, and the read copy in
# compute_graph_health could observe mid-update state.
_STATE_LOCK = threading.Lock()

_MAINTAINER_STATUS: dict[str, Any] = {
    "last_cycle_at": None,
    "last_cycle_status": "never",
    "last_cycle_error": None,
}

# Failure counters for the session-start context helper. Every failure
# path in `build_session_start_graph_context` returns "" to keep peer
# spawn safe, but the operator needs to know whether the empty string
# means "graph deliberately empty" or "helper is silently broken". We
# tag each failure type so graph_health.json shows a breakdown.
_SESSION_CONTEXT_FAILURES: dict[str, int] = {
    "init_db": 0,
    "fetch_prior": 0,
    "fetch_lineage": 0,
    "neighbor_load": 0,
    "orientation_query": 0,
}


def _record_session_failure(kind: str) -> None:
    with _STATE_LOCK:
        _SESSION_CONTEXT_FAILURES[kind] = _SESSION_CONTEXT_FAILURES.get(kind, 0) + 1


def _report_maintainer_status(maintainer) -> dict[str, Any]:
    """Capture the maintainer's last-cycle observables into the module
    status dict so compute_graph_health() can surface them even when
    called from a context without the maintainer instance (e.g. the
    CLI --mode health path). Returns a snapshot copy."""
    if maintainer is None:
        with _STATE_LOCK:
            return dict(_MAINTAINER_STATUS)
    with _STATE_LOCK:
        _MAINTAINER_STATUS["last_cycle_at"] = getattr(maintainer, "_last_cycle_at", None)
        _MAINTAINER_STATUS["last_cycle_status"] = getattr(maintainer, "_last_cycle_status", "never")
        _MAINTAINER_STATUS["last_cycle_error"] = getattr(maintainer, "_last_cycle_error", None)
        return dict(_MAINTAINER_STATUS)


def reset_graph_observability_state() -> None:
    """Zero the module-level counters and status. Intended for
    FindingGraphMaintainer.__init__ (fresh-run contract) and for test
    harnesses that spin up multiple orchestrators in one Python
    process. Without this, counters monotonically accumulate across
    logically-independent runs."""
    with _STATE_LOCK:
        _MAINTAINER_STATUS["last_cycle_at"] = None
        _MAINTAINER_STATUS["last_cycle_status"] = "never"
        _MAINTAINER_STATUS["last_cycle_error"] = None
        for k in _SESSION_CONTEXT_FAILURES:
            _SESSION_CONTEXT_FAILURES[k] = 0


def compute_graph_health() -> dict[str, Any]:
    """Snapshot of graph size + coverage + edge-type distribution.

    Intended to be written each maintainer cycle to ``run_dir/graph/graph_health.json``.
    """
    from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store as ls

    try:
        num_findings = ls.count_findings()
    except Exception as e:
        logger.debug("compute_graph_health count_findings failed: %s", e)
        num_findings = 0
    try:
        num_edges = ls.count_edges()
    except Exception as e:
        logger.debug("compute_graph_health count_edges failed: %s", e)
        num_edges = 0
    try:
        etypes = ls.edge_count_by_type()
    except Exception as e:
        logger.debug("compute_graph_health edge_count_by_type failed: %s", e)
        etypes = {edge_type: 0 for edge_type in getattr(ls, "_VALID_EDGE_TYPES", ())}
    # linked_finding_ratio: fraction of findings with ≥1 edge

    try:
        with ls._get_conn(readonly=True) as conn:
            row = conn.execute(
                """SELECT COUNT(DISTINCT fid) AS c FROM (
                     SELECT src_finding_id AS fid FROM finding_edges
                     UNION
                     SELECT dst_finding_id AS fid FROM finding_edges
                   )"""
            ).fetchone()
            linked = int(row["c"]) if row else 0
            low_conf_row = conn.execute(
                "SELECT COUNT(*) AS c FROM finding_edges WHERE confidence < 0.60"
            ).fetchone()
            low_conf = int(low_conf_row["c"]) if low_conf_row else 0
    except Exception as e:  # readonly path may race init
        logger.debug("compute_graph_health query failed: %s", e)
        linked = 0
        low_conf = 0

    linked_ratio = (linked / num_findings) if num_findings else 0.0
    low_conf_ratio = (low_conf / num_edges) if num_edges else 0.0
    try:
        recent_unlinked = ls.get_unlinked_recent_findings(hours=6.0, limit=100)
    except Exception as e:
        logger.debug("compute_graph_health recent_unlinked query failed: %s", e)
        recent_unlinked = []

    with _STATE_LOCK:
        session_failures = dict(_SESSION_CONTEXT_FAILURES)
        maintainer_snapshot = dict(_MAINTAINER_STATUS)
    return {
        "num_findings": num_findings,
        "num_edges": num_edges,
        "linked_finding_ratio": round(linked_ratio, 4),
        "unlinked_recent_count": len(recent_unlinked),
        "edge_type_distribution": etypes,
        "low_confidence_edge_ratio": round(low_conf_ratio, 4),
        "largest_component_ratio": None,  # v1: not computed; add in v2 if needed
        "session_context_failures": session_failures,
        "maintainer": maintainer_snapshot,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def write_graph_health(out_dir: Path) -> dict[str, Any]:
    """Compute health + atomic-write to ``<out_dir>/graph_health.json``."""
    from .atomic_io import atomic_write_json

    out_dir.mkdir(parents=True, exist_ok=True)
    health = compute_graph_health()
    atomic_write_json(out_dir / "graph_health.json", health)
    return health


# --- Session-start graph context --------------------------------------------
# Injected into every peer's prompt at render time. The goal is to take the
# graph from a passive lookup surface (agents have to remember to call the
# MCP tools) to an active orientation primitive (each session begins with a
# curated view of the most-relevant findings already in the graph).
#
# We do NOT try to be smart — this is a deterministic ranking over rule
# engine edges, identical to what an agent would see via
# ``get_finding_neighbors`` calls. The purpose is purely to save the
# "forgot to check" cost.

_EDGE_WEIGHT = {
    "supports": 1.0,
    "challenges": 1.0,
    "updates": 0.8,
    "derived_from": 0.7,
    "related_to": 0.3,
}


def _score_edge_pair(edge: dict[str, Any], anchor_peer_id: str, neighbor_peer_id: str) -> float:
    """Rank key: prefer strong edges, higher confidence, cross-peer
    neighbors (a sibling peer commenting on your work is usually more
    informative than you citing yourself)."""
    w = _EDGE_WEIGHT.get(edge.get("edge_type", "related_to"), 0.3)
    cross = (
        1.2 if (anchor_peer_id and neighbor_peer_id and anchor_peer_id != neighbor_peer_id) else 1.0
    )
    return w * float(edge.get("confidence", 0.55)) * cross


def _snippet(text: str | None, n: int = 180) -> str:
    """Return a one-line, prompt-safe snippet of ``text``.

    Agent-authored titles/content can contain triple-backtick fences,
    raw newlines, and other Markdown structural characters. When those
    get interpolated into a prompt's Markdown bullet list, they can
    break out of a surrounding code block or list item — and an LLM
    reading the prompt then misinterprets the structure. Sanitize:
      - collapse ALL whitespace (including newlines, tabs) to single
        spaces so one bullet stays one bullet
      - replace triple-backticks with a safe marker so an agent can
        tell something was there but can't close a code fence
      - strip leading/trailing whitespace
    We keep single backticks (useful for quoting variant names) and
    other inline Markdown — only the structure-breakers get touched.
    """
    if not text:
        return ""
    t = str(text)
    # Collapse all whitespace runs → single space.
    t = re.sub(r"\s+", " ", t)
    # Defang code fences. ~~~ is the CommonMark alternate fence.
    t = t.replace("```", "`\u200b``").replace("~~~", "~\u200b~~")
    t = t.strip()
    return t if len(t) <= n else t[:n] + "…"


def build_session_start_graph_context(
    peer_id: str,
    max_prior_findings: int = 5,
    max_neighbors: int = 6,
    max_anchors: int = 4,
) -> str:
    """Markdown snippet summarizing the graph neighborhood most relevant to
    this peer's starting session.

    Returns a string that either (a) lists the top N cross-peer neighbors of
    this peer's recent findings (for a peer that's been active), or (b) lists
    the current graph's orientation anchors (for a fresh peer). Returns
    empty string on any failure — the caller treats this as optional.
    """
    try:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            local_store as ls,
        )

        ls.init_db()
    except Exception as e:
        logger.debug("graph context: init_db failed: %s", e)
        _record_session_failure("init_db")
        return ""

    try:
        # SQL-side filter — calling get_all_findings() + Python-side
        # filter transferred every row for every peer render, which
        # serialized cohort launch on full-table scans.
        recent = ls.get_findings(peer_id=peer_id, limit=max_prior_findings)
    except Exception as e:
        logger.debug("graph context: fetch prior findings failed: %s", e)
        _record_session_failure("fetch_prior")
        return ""

    if recent:
        return _render_prior_work_context(peer_id, recent, max_neighbors)

    # Lineage fallback — a fresh gen{N}_peer{k} inherits from
    # gen{N-1}_peer{k}. This is the handoff convention: same slot-index
    # across generations represents a continuing research thread. If a
    # sibling from the previous generation has findings, use those as
    # the anchor set instead of jumping straight to orientation anchors.
    lineage_peer_id = _previous_generation_peer_id(peer_id)
    if lineage_peer_id:
        try:
            lineage = ls.get_findings(
                peer_id=lineage_peer_id,
                limit=max_prior_findings,
            )
        except Exception as e:
            logger.debug("graph context: lineage fetch failed: %s", e)
            _record_session_failure("fetch_lineage")
            return ""
        if lineage:
            return _render_prior_work_context(
                peer_id,
                lineage,
                max_neighbors,
                lineage_peer_id=lineage_peer_id,
            )

    return _render_orientation_context(max_anchors)


def _previous_generation_peer_id(peer_id: str) -> str | None:
    """Parse `gen{N}_peer{k}` and return `gen{N-1}_peer{k}` if N > 0.

    Returns None for peer_ids that don't match the expected shape or
    are already in generation 0 (no earlier generation to inherit from).
    """
    import re

    m = re.fullmatch(r"gen(\d+)_peer(\d+)", peer_id or "")
    if not m:
        return None
    gen = int(m.group(1))
    if gen <= 0:
        return None
    return f"gen{gen - 1}_peer{m.group(2)}"


def _render_prior_work_context(
    peer_id: str,
    anchors: list[dict[str, Any]],
    max_neighbors: int,
    lineage_peer_id: str | None = None,
) -> str:
    """Given a few findings authored by this peer (or by its lineage
    predecessor), rank their graph neighbors and emit a Markdown list.

    If ``lineage_peer_id`` is set, the header explains that the anchors
    come from the previous generation's same-slot peer, not from this
    peer directly — otherwise the agent would be confused reading
    "your prior work" as a fresh gen{N}_peer{k}.
    """
    from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store as ls

    # Prompt-surface confidence floor — stricter than the MCP tool's
    # 0.55. This surface is shown to agents BEFORE they've read any
    # findings, so a false-positive `supports` edge at 0.55 is more
    # dangerous here than when the agent explicitly queried
    # get_finding_neighbors. Agents can still see weak edges via the
    # MCP tool when they decide to dig.
    PROMPT_CONFIDENCE_FLOOR = 0.70

    seen: dict[str, dict[str, Any]] = {}  # neighbor_id → {score, edge, finding, anchor}
    for anchor in anchors:
        aid = anchor.get("id")
        if not aid:
            continue
        try:
            out_edges = ls.get_edges_for_finding(
                aid,
                direction="out",
                min_confidence=PROMPT_CONFIDENCE_FLOOR,
                limit=20,
            )
            in_edges = ls.get_edges_for_finding(
                aid,
                direction="in",
                min_confidence=PROMPT_CONFIDENCE_FLOOR,
                limit=20,
            )
        except Exception as e:
            logger.debug("graph context: edges for %s failed: %s", aid, e)
            continue

        edge_targets = [(e, e["dst_finding_id"], "out") for e in out_edges]
        edge_targets += [(e, e["src_finding_id"], "in") for e in in_edges]
        if not edge_targets:
            continue

        # Batch-load neighbor findings in one query.
        nids = {t[1] for t in edge_targets if t[1] != aid}
        if not nids:
            # Skip empty IN-clause: `SELECT ... WHERE id IN ()` is a
            # SQL syntax error. This is a legitimate input case — all
            # edges from this anchor pointed back at itself (possible
            # when the anchor has a self-loop rule-2 link that got
            # filtered at insert time).
            continue
        try:
            with ls._get_conn(readonly=True) as conn:
                placeholders = ",".join("?" * len(nids))
                rows = conn.execute(
                    f"SELECT * FROM findings WHERE id IN ({placeholders})",
                    list(nids),
                ).fetchall()
            nmap = {r["id"]: ls._row_to_finding(r) for r in rows}
        except Exception as e:
            logger.debug("graph context: neighbor load failed: %s", e)
            _record_session_failure("neighbor_load")
            continue

        for edge, nid, direction in edge_targets:
            nf = nmap.get(nid)
            if not nf:
                continue
            score = _score_edge_pair(edge, peer_id, nf.get("peer_id", ""))
            prev = seen.get(nid)
            if prev is None or score > prev["score"]:
                seen[nid] = {
                    "score": score,
                    "edge": edge,
                    "finding": nf,
                    "anchor": anchor,
                    "direction": direction,
                }

    ranked = sorted(seen.values(), key=lambda x: -x["score"])[:max_neighbors]
    if not ranked:
        return ""

    # Put the advisory warning ABOVE the anchor summary so the agent
    # reads "this is navigation, not evidence" before any specific
    # confidence numbers. The bullets that follow are high-signal but
    # low-rigor — a keyword-matched `supports` edge at conf=0.75
    # should never be mistaken for "replicated 5-seed evidence".
    advisory = (
        "> **ADVISORY CONTEXT — not evidence, and a snapshot.** The "
        "bullets below were produced by lightweight rules (keyword "
        "match + shared variant name), not by reading finding "
        "content. Confidence scores are about rule strength, NOT "
        "about scientific rigor. This list was frozen at the start "
        "of your peer session — sibling peers publish continuously, "
        "so for anything time-sensitive call "
        "`mcp__finding-graph-query__get_finding_neighbors` to get "
        "live edges. Always open the raw finding via its "
        "`finding_id` before citing, replicating, or contradicting."
    )
    if lineage_peer_id:
        anchor_summary = (
            f"Top {len(ranked)} graph-surfaced findings related to the "
            f"work of your lineage predecessor `{lineage_peer_id}` "
            f"(same peer slot, previous generation). These are the threads "
            f"you're inheriting."
        )
    else:
        anchor_summary = (
            f"Top {len(ranked)} graph-surfaced findings related to your "
            f"prior work (ranked by edge type × confidence × cross-peer "
            f"crossover)."
        )
    lines = [
        "## Graph-surfaced context",
        "",
        advisory,
        "",
        anchor_summary,
        "",
    ]
    for item in ranked:
        e = item["edge"]
        nf = item["finding"]
        anchor = item["anchor"]
        direction = item["direction"]
        arrow = "→" if direction == "out" else "←"
        title = _snippet(nf.get("title"), 80) or "(untitled)"
        neighbor_peer = nf.get("peer_id") or "?"
        ftype = nf.get("finding_type") or "?"
        conf = float(e.get("confidence", 0.55))
        etype = e.get("edge_type", "related_to")
        rationale = _snippet(e.get("rationale"), 110)
        snippet = _snippet(nf.get("content"), 140)
        anchor_title = _snippet(anchor.get("title"), 60) or anchor.get("id", "")

        lines.append(
            f"- **{etype}** ({conf:.2f}) {arrow} **[{ftype}]** `{neighbor_peer}` — {title}"
        )
        lines.append(f"  - `finding_id`: `{nf.get('id', '')}`")
        lines.append(f"  - via your finding: _{anchor_title}_")
        if rationale:
            lines.append(f"  - rule rationale: {rationale}")
        if snippet:
            lines.append(f"  - snippet: {snippet}")

    lines.append("")
    lines.append(
        "To chase any of these further, call "
        "`mcp__finding-graph-query__get_finding_neighbors` with the id."
    )
    return "\n".join(lines)


def _render_orientation_context(max_anchors: int) -> str:
    """Fallback context for a peer with no prior findings: show the most
    connected findings currently in the graph so the agent has some
    anchors to sanity-check its direction against."""
    try:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            local_store as ls,
        )

        with ls._get_conn(readonly=True) as conn:
            rows = conn.execute(
                """SELECT f.*, COUNT(e.edge_id) AS deg
                   FROM findings f
                   LEFT JOIN finding_edges e
                     ON f.id = e.src_finding_id OR f.id = e.dst_finding_id
                   GROUP BY f.id
                   ORDER BY deg DESC, f.timestamp DESC
                   LIMIT ?""",
                (max_anchors,),
            ).fetchall()
        anchors = [(dict(r), int(r["deg"])) for r in rows]
    except Exception as e:
        logger.debug("graph context: orientation query failed: %s", e)
        _record_session_failure("orientation_query")
        return ""

    if not anchors or all(deg == 0 for _, deg in anchors):
        # Empty graph → nothing useful to inject.
        return ""

    lines = [
        "## Graph-surfaced context",
        "",
        "> **ADVISORY CONTEXT — not evidence, and a snapshot.** The "
        "bullets below are orientation anchors chosen by local-degree. "
        "They mark where sibling peers are converging, not where the "
        "evidence is strongest. Frozen at session start — for live "
        "edges call `mcp__finding-graph-query__get_finding_neighbors`. "
        "Always open the raw finding via its `finding_id` before "
        "building on it.",
        "",
        "You have no findings yet, so the graph is showing its "
        "most-connected current findings. Look here first before "
        "proposing a direction that duplicates a crowded thread.",
        "",
    ]
    for f, deg in anchors:
        title = _snippet(f.get("title"), 80) or "(untitled)"
        peer = f.get("peer_id") or "?"
        ftype = f.get("finding_type") or "?"
        snippet = _snippet(f.get("content"), 140)
        lines.append(f"- **[{ftype}]** `{peer}` degree={deg} — {title}")
        lines.append(f"  - `finding_id`: `{f.get('id', '')}`")
        if snippet:
            lines.append(f"  - snippet: {snippet}")
    lines.append("")
    lines.append(
        "Call `mcp__finding-graph-query__get_finding_neighbors` on any id "
        "to see the local cluster, or "
        "`mcp__finding-graph-query__get_unlinked_recent_findings` to find "
        "threads the graph hasn't absorbed yet."
    )
    return "\n".join(lines)


# --- Maintainer daemon ------------------------------------------------------


class FindingGraphMaintainer:
    """Background daemon that periodically runs the rule engine over new
    findings and upserts their edges into SQLite.

    Mirrors the FindingsSync shape (start/stop/sync_once) for a consistent
    orchestrator lifecycle.
    """

    def __init__(self, run_dir: Path, poll_interval: int = 120):
        # Zero module-level observability state at the start of each
        # run. Without this, counters accumulated from prior runs in
        # the same Python process (CLI health checks, test harnesses,
        # back-to-back orchestrator instances) leak into this run's
        # graph_health.json, making the numbers meaningless.
        reset_graph_observability_state()

        self.run_dir = Path(run_dir)
        self.poll_interval = poll_interval
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._graph_dir = self.run_dir / "graph"
        self._graph_dir.mkdir(parents=True, exist_ok=True)
        self._last_cycle_at: str | None = None
        self._last_cycle_status: str = "never"
        self._last_cycle_error: str | None = None
        # Non-blocking lock — sync_once is called from three sites
        # (the 120s timer thread, the inter-generation hook, and the
        # run() finally block). Without this guard, all three can
        # queue up at a generation boundary and run back-to-back for
        # 30+ seconds of duplicated CPU. tryacquire + skip keeps
        # correctness (next caller will see a fresh state anyway)
        # while preventing pileups.
        self._cycle_lock = threading.Lock()

    def sync_once(self) -> dict[str, Any]:
        """Run one maintainer cycle. Returns a small summary dict.

        Safe to call from multiple threads — concurrent calls no-op
        (return status=busy) rather than duplicating work. Use
        ``sync_once_blocking()`` when the caller NEEDS the cycle to
        run (e.g. inter-generation barrier)."""
        if not self._cycle_lock.acquire(blocking=False):
            logger.debug("graph maintainer: cycle already in progress, skipping")
            return {"status": "busy"}
        try:
            return self._sync_once_inner()
        finally:
            self._cycle_lock.release()

    def sync_once_blocking(self, timeout: float = 300.0) -> dict[str, Any]:
        """Run a cycle, waiting for any in-progress cycle to finish
        first. Used at generation boundaries where the orchestrator
        needs the graph to absorb the just-finished generation's
        findings before the next generation's prompts render —
        silently skipping would leave gen N+1 peers reading stale edges.

        Returns ``{"status": "timeout"}`` if the lock cannot be
        acquired within ``timeout`` seconds; this caps the worst-case
        delay at a generation boundary."""
        if not self._cycle_lock.acquire(blocking=True, timeout=timeout):
            logger.warning(
                "graph maintainer: blocking sync timed out after %.0fs",
                timeout,
            )
            return {"status": "timeout"}
        try:
            return self._sync_once_inner()
        finally:
            self._cycle_lock.release()

    def _sync_once_inner(self) -> dict[str, Any]:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            local_store as ls,
        )

        try:
            ls.init_db()
        except Exception as e:
            logger.debug("graph maintainer: init_db failed: %s", e)
            self._last_cycle_status = "error"
            self._last_cycle_error = f"init_db: {e}"
            # Propagate to the module-level status dict so health JSON
            # reflects the error. Without this, a string of failed
            # cycles leaves _MAINTAINER_STATUS stuck at the last "ok",
            # and the graph_health.json shown to operators silently
            # misrepresents a broken maintainer as healthy.
            _report_maintainer_status(self)
            return {"status": "error", "error": str(e)}

        try:
            all_findings = ls.get_all_findings()
        except Exception as e:
            logger.debug("graph maintainer: get_all_findings failed: %s", e)
            self._last_cycle_status = "error"
            self._last_cycle_error = f"get_all_findings: {e}"
            _report_maintainer_status(self)
            return {"status": "error", "error": str(e)}

        if not all_findings:
            self._last_cycle_at = datetime.now(UTC).isoformat()
            self._last_cycle_status = "empty"
            self._last_cycle_error = None
            _report_maintainer_status(self)
            return {"status": "empty", "proposed": 0, "inserted": 0}

        builder = FindingGraphBuilder(all_findings)
        proposed_edges = builder.build_all_edges()
        # Filter below the confidence floor before insert.
        proposed_edges = [e for e in proposed_edges if e["confidence"] >= builder.MIN_CONFIDENCE]
        inserted = ls.insert_edges_batch(proposed_edges)

        # Tag the cycle as successful BEFORE writing health so the
        # single write_graph_health call emits the current cycle's
        # status — not the previous cycle's stale one. Earlier versions
        # wrote health twice (once with stale status, once with fresh)
        # which produced a transient "never"/"error" state visible to
        # filesystem watchers on every successful cycle.
        self._last_cycle_at = datetime.now(UTC).isoformat()
        self._last_cycle_status = "ok"
        self._last_cycle_error = None
        _report_maintainer_status(self)

        try:
            health = write_graph_health(self._graph_dir)
        except Exception as e:
            logger.warning("graph maintainer: write_graph_health failed: %s", e)
            health = {}

        # Render the live HTML visualization alongside the health JSON.
        # Failures here must never break the maintainer loop — the graph
        # itself is the source of truth; viz is convenience.
        try:
            from praxist.plugins.graph_maintainers.finding_graph_mvp.viz import (
                render_graph_html,
            )

            render_graph_html(self._graph_dir / "graph.html")
        except Exception as e:
            logger.warning("graph maintainer: render_graph_html failed: %s", e)

        # Also dump the current unlinked-recent list for human inspection.
        try:
            from .atomic_io import atomic_write_json

            atomic_write_json(
                self._graph_dir / "unlinked_recent_findings.json",
                {
                    "generated_at": datetime.now(UTC).isoformat(),
                    "findings": [
                        {
                            "id": f["id"],
                            "peer_id": f.get("peer_id"),
                            "finding_type": f.get("finding_type"),
                            "title": f.get("title"),
                            "timestamp": f.get("timestamp"),
                        }
                        for f in ls.get_unlinked_recent_findings(hours=6.0, limit=30)
                    ],
                },
            )
        except Exception as e:
            logger.debug("graph maintainer: unlinked dump failed: %s", e)
        logger.info(
            "graph maintainer: proposed=%d inserted=%d total_edges=%d",
            len(proposed_edges),
            inserted,
            health.get("num_edges", 0),
        )
        return {
            "status": "ok",
            "proposed": len(proposed_edges),
            "inserted": inserted,
            "total_edges": health.get("num_edges"),
            "linked_ratio": health.get("linked_finding_ratio"),
        }

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self.sync_once()
            except Exception as e:
                logger.debug("graph maintainer cycle failed: %s", e)
            if self._stop_event.is_set():
                break
            try:
                asyncio.run(
                    wait_for_filesystem_event(
                        [
                            self.run_dir / "shared_findings",
                        ],
                        timeout_seconds=max(300, int(self.poll_interval)),
                        stop_check=self._stop_event.is_set,
                        recursive=False,
                        max_dirs=128,
                        fallback_interval_seconds=max(300, int(self.poll_interval)),
                        stop_check_interval_seconds=30.0,
                        event_filter=lambda p: Path(p).suffix.lower() == ".json",
                    )
                )
            except Exception as e:
                logger.debug("graph maintainer event wait failed: %s", e)
                self._stop_event.wait(timeout=max(300, int(self.poll_interval)))

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
