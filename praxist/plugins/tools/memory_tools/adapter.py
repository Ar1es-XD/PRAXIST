"""MCP tools for PI panel — read-only queries over research memory.

These tools are exposed ONLY to the PI synthesizer agents (Builder /
Skeptic / Portfolio / External Validity / Chair). Peers (research workers)
do NOT receive these tools.

The MCP server is created via `create_memory_tools_server(run_dir)`
which closes over the run_dir — the SDK tool handlers don't need to
take it as a parameter. This mirrors the registration pattern used
by frontier_tools / finding_graph_query.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

# Optional SDK imports (mirror pattern from frontier_tools.py).
try:
    from claude_agent_sdk import create_sdk_mcp_server, tool
except ImportError:
    tool = None  # type: ignore
    create_sdk_mcp_server = None  # type: ignore

logger = logging.getLogger(__name__)

_ALLOWED_LEDGER_NAMES = {
    "claim_ledger",
    "hypothesis_ledger",
    "mechanism_ledger",
    "coverage_matrix",
    "negative_evidence_ledger",
    "retired_claim_ledger",
    "dissent_ledger",
    "frontier_delta_ledger",
    "role_roi_ledger",
}


def _safe_run_dir(run_dir) -> Path:
    p = Path(run_dir).resolve()
    if not p.exists():
        raise ValueError(f"run_dir does not exist: {run_dir}")
    return p


def _coerce_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _runtime_generation_limit(requested: int | None = None) -> int | None:
    """Return the maximum generation a PI memory tool may read.

    PI prompts receive evidence packs capped at the completed generation. These
    MCP tools must honor the same boundary; otherwise a PI can bypass the pack
    by directly querying research memory. Direct offline calls without runtime
    generation environment remain unbounded for backward-compatible tooling.
    """

    completed_generation = os.environ.get("LAST_COMPLETED_GENERATION_ID") or os.environ.get(
        "COMPLETED_GEN_ID"
    )
    if completed_generation is not None:
        default_generation = _coerce_int(completed_generation, default=-1)
    else:
        current_generation = (
            os.environ.get("CURRENT_GEN_ID")
            or os.environ.get("GENERATION_ID")
            or os.environ.get("PRAXIST_GENERATION_ID")
        )
        current = _coerce_int(current_generation, default=None)
        default_generation = current - 1 if current is not None and current >= 0 else None

    allow_unbounded = str(os.environ.get("PRAXIST_MEMORY_TOOLS_ALLOW_UNBOUNDED", "")).lower() in {
        "1",
        "true",
        "yes",
    }
    if allow_unbounded:
        return requested
    if default_generation is None or default_generation < 0:
        return requested
    if requested is None or requested < 0 or requested > default_generation:
        return default_generation
    return requested


def _generation_from_obj(obj: Any) -> int | None:
    if not isinstance(obj, dict):
        return None
    for key in (
        "generation_id",
        "gen_id",
        "source_generation_id",
        "completed_generation",
        "completed_gen_id",
    ):
        value = obj.get(key)
        parsed = _coerce_int(value)
        if parsed is not None:
            return parsed
    for key in ("source_ref", "metrics", "extra", "data"):
        nested = obj.get(key)
        parsed = _generation_from_obj(nested)
        if parsed is not None:
            return parsed
    return None


def _card_generation_map(run_dir: Path) -> dict[str, int]:
    from praxist.plugins.workflow_stages.research_loop.backend.research_memory.card_builder import (
        build_cards_from_db,
    )

    mapping: dict[str, int] = {}
    try:
        cards = build_cards_from_db(run_dir)
    except Exception:
        return mapping
    for card in cards:
        if not isinstance(card, dict):
            continue
        evidence_id = str(card.get("evidence_id") or "").strip()
        generation = _generation_from_obj(card)
        if evidence_id and generation is not None:
            mapping[evidence_id] = generation
    return mapping


def _source_ids_from_obj(obj: Any) -> list[str]:
    ids: list[str] = []
    if isinstance(obj, dict):
        for key in (
            "supports",
            "challenges",
            "informs",
            "sources",
            "source_evidence_ids",
            "evidence_ids",
        ):
            value = obj.get(key)
            if isinstance(value, list):
                ids.extend(str(item) for item in value if str(item).strip())
            elif isinstance(value, str) and value.strip():
                ids.append(value.strip())
        for value in obj.values():
            if isinstance(value, (dict, list)):
                ids.extend(_source_ids_from_obj(value))
    elif isinstance(obj, list):
        for item in obj:
            ids.extend(_source_ids_from_obj(item))
    return ids


def _visible_at_generation(
    obj: Any,
    max_generation_id: int | None,
    *,
    evidence_generations: dict[str, int] | None = None,
) -> bool:
    if max_generation_id is None:
        return True
    generation = _generation_from_obj(obj)
    if generation is not None:
        return generation <= int(max_generation_id)
    evidence_generations = evidence_generations or {}
    source_ids = _source_ids_from_obj(obj)
    if source_ids:
        generations = [
            evidence_generations[source_id]
            for source_id in source_ids
            if source_id in evidence_generations
        ]
        return bool(generations) and max(generations) <= int(max_generation_id)
    return False


def _resolved_source_visible(result: dict[str, Any], max_generation_id: int | None) -> bool:
    if max_generation_id is None:
        return True
    if not isinstance(result, dict):
        return False
    if "error" in result:
        return True
    content = result.get("content")
    generation = _generation_from_obj(content)
    if generation is not None:
        return generation <= int(max_generation_id)
    generation = _generation_from_obj(result)
    if generation is not None:
        return generation <= int(max_generation_id)
    return False


def _cutoff_error(kind: str, max_generation_id: int | None) -> dict[str, Any]:
    return {
        "error": f"{kind} is outside the current PI evidence cutoff",
        "max_generation_id": max_generation_id,
    }


def get_evidence_card(
    run_dir,
    evidence_id: str,
    max_generation_id: int | None = None,
) -> dict[str, Any]:
    """Return full evidence card by id, or {error: ...} if not found.

    Use this when the role-specific pack you received omitted details for
    a card you need to inspect more closely.
    """
    from praxist.plugins.workflow_stages.research_loop.backend.research_memory.card_builder import (
        build_cards_from_db,
    )

    rd = _safe_run_dir(run_dir)
    cutoff = _runtime_generation_limit(max_generation_id)
    cards = build_cards_from_db(rd, max_gen=cutoff)
    for c in cards:
        if c.get("evidence_id") == evidence_id:
            return c
    return {"error": f"evidence_id not found: {evidence_id}", "evidence_id": evidence_id}


def query_evidence_cards(
    run_dir,
    claim_id: str | None = None,
    mechanism: str | None = None,
    peer_id: str | None = None,
    generation_id: int | None = None,
    is_negative: bool | None = None,
    limit: int = 20,
    max_generation_id: int | None = None,
) -> list[dict[str, Any]]:
    """Filter evidence cards. Returns id + short interpretation only.

    To get a full card, follow up with get_evidence_card(evidence_id).
    """
    from praxist.plugins.workflow_stages.research_loop.backend.research_memory.card_builder import (
        build_cards_from_db,
    )

    rd = _safe_run_dir(run_dir)
    cutoff = _runtime_generation_limit(max_generation_id)
    cards = build_cards_from_db(rd, max_gen=cutoff)
    out: list[dict[str, Any]] = []
    for c in cards:
        if not isinstance(c, dict):
            continue
        ref = c.get("source_ref", {}) or {}
        rel = c.get("claim_relevance", {}) or {}
        if peer_id and ref.get("peer_id") != peer_id:
            continue
        if generation_id is not None and ref.get("generation_id") != generation_id:
            continue
        if is_negative is not None and bool(c.get("quality", {}).get("is_negative")) != bool(
            is_negative
        ):
            continue
        if claim_id:
            relevant_claims = (
                (rel.get("supports") or [])
                + (rel.get("challenges") or [])
                + (rel.get("informs") or [])
            )
            if claim_id not in relevant_claims:
                continue
        if mechanism:
            interp = (c.get("interpretation", {}) or {}).get("short", "")
            if mechanism.lower() not in interp.lower():
                continue
        out.append(
            {
                "evidence_id": c.get("evidence_id"),
                "source_type": c.get("source_type"),
                "interpretation_short": (c.get("interpretation", {}) or {}).get("short", "")[:200],
                "is_negative": c.get("quality", {}).get("is_negative", False),
                "evidence_valence": c.get("quality", {}).get("evidence_valence"),
                "failure_mode": c.get("quality", {}).get("failure_mode"),
                "disconfirming_claim_ids": c.get("quality", {}).get("disconfirming_claim_ids", []),
                "metrics": c.get("metrics"),
                "generation_id": ref.get("generation_id"),
                "peer_id": ref.get("peer_id"),
                "variant_name": ref.get("variant_name"),
            }
        )
        if len(out) >= limit:
            break
    return out


def query_coverage_matrix(
    run_dir,
    variant_family: str | None = None,
    parameter: str | None = None,
    bridge_pair: list[str] | None = None,
    bridge_dimension: str | None = None,
    max_generation_id: int | None = None,
) -> dict[str, Any]:
    """Check if (variant_family, parameter) grid or (pair, dimension) bridge is covered.

    MUST be called by Bridge contracts before assignment.
    Returns {covered: bool, points: [...], sources: [...], or empty}.
    """
    from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.coverage_matrix import (
        CoverageMatrix,
    )

    rd = _safe_run_dir(run_dir)
    cutoff = _runtime_generation_limit(max_generation_id)
    evidence_generations = _card_generation_map(rd) if cutoff is not None else {}
    cov = CoverageMatrix(rd)
    if bridge_pair and bridge_dimension and len(bridge_pair) == 2:
        d = cov.query_bridge(bridge_pair[0], bridge_pair[1], bridge_dimension)
        if d is None:
            return {
                "covered": False,
                "kind": "bridge",
                "pair": bridge_pair,
                "dimension": bridge_dimension,
            }
        if not _visible_at_generation(d, cutoff, evidence_generations=evidence_generations):
            return {
                "covered": False,
                "kind": "bridge",
                "pair": bridge_pair,
                "dimension": bridge_dimension,
                "max_generation_id": cutoff,
            }
        return {
            "covered": len(d.get("bridge_points_tested", [])) > 0,
            "kind": "bridge",
            "pair": d.get("variant_pair"),
            "dimension": d.get("grid_dimension"),
            "points": d.get("bridge_points_tested", []),
            "sources": d.get("sources", []),
        }
    if variant_family:
        d = cov.query_grid(variant_family, parameter or "rho_max")
        if d is None:
            return {
                "covered": False,
                "kind": "grid",
                "variant_family": variant_family,
                "parameter": parameter,
            }
        if not _visible_at_generation(d, cutoff, evidence_generations=evidence_generations):
            return {
                "covered": False,
                "kind": "grid",
                "variant_family": variant_family,
                "parameter": parameter,
                "max_generation_id": cutoff,
            }
        return {
            "covered": len(d.get("values_tested", [])) > 0,
            "kind": "grid",
            "variant_family": d.get("variant_family"),
            "parameter": d.get("parameter"),
            "values_tested": d.get("values_tested", []),
            "seed_counts": d.get("seed_counts", {}),
            "sources": d.get("sources", []),
        }
    return {"error": "must specify either variant_family or bridge_pair+bridge_dimension"}


def list_active_claims(run_dir, max_generation_id: int | None = None) -> list[dict[str, Any]]:
    """List active claims with status, confidence, boundary, and supports/challenges counts."""
    from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.claim_ledger import (
        ClaimLedger,
    )

    rd = _safe_run_dir(run_dir)
    cutoff = _runtime_generation_limit(max_generation_id)
    evidence_generations = _card_generation_map(rd) if cutoff is not None else {}
    cl = ClaimLedger(rd)
    out = []
    for e in cl.list_active():
        if not _visible_at_generation(
            e.to_dict(),
            cutoff,
            evidence_generations=evidence_generations,
        ):
            continue
        out.append(
            {
                "id": e.id,
                "title": e.data.get("title", ""),
                "status": e.data.get("status"),
                "confidence": e.data.get("confidence"),
                "boundary": e.data.get("boundary", ""),
                "supports_count": len(e.data.get("supports", [])),
                "challenges_count": len(e.data.get("challenges", [])),
                "missing_tests": e.data.get("missing_tests", []),
            }
        )
    return out


def list_open_objections(run_dir, max_generation_id: int | None = None) -> list[dict[str, Any]]:
    """List open / experiment-assigned dissent entries."""
    from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.dissent_ledger import (
        DissentLedger,
    )

    rd = _safe_run_dir(run_dir)
    cutoff = _runtime_generation_limit(max_generation_id)
    evidence_generations = _card_generation_map(rd) if cutoff is not None else {}
    dl = DissentLedger(rd)
    out = []
    for e in dl.list_open():
        if not _visible_at_generation(
            e.to_dict(),
            cutoff,
            evidence_generations=evidence_generations,
        ):
            continue
        out.append(
            {
                "id": e.id,
                "disputed_claim_id": e.data.get("disputed_claim_id"),
                "status": e.data.get("status"),
                "resolving_experiment": e.data.get("resolving_experiment", ""),
                "decision_rule": e.data.get("decision_rule", {}),
            }
        )
    return out


def get_ledger_entry(
    run_dir,
    ledger_name: str,
    entry_id: str,
    max_generation_id: int | None = None,
) -> dict[str, Any]:
    """Read a single entry from any ledger.

    Supported ledger_names: claim_ledger, hypothesis_ledger, mechanism_ledger,
    coverage_matrix, negative_evidence_ledger, retired_claim_ledger,
    dissent_ledger, frontier_delta_ledger, role_roi_ledger.
    """
    from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers._ledger_base import (
        LedgerStore,
    )

    rd = _safe_run_dir(run_dir)
    cutoff = _runtime_generation_limit(max_generation_id)
    if ledger_name not in _ALLOWED_LEDGER_NAMES:
        return {"error": f"unsupported ledger_name: {ledger_name}"}
    ledgers_dir = (rd / "research_memory" / "ledgers").resolve()
    path = (ledgers_dir / f"{ledger_name}.yaml").resolve()
    if not path.is_relative_to(ledgers_dir):
        return {"error": f"ledger path escapes ledgers directory: {ledger_name}"}
    if not path.exists():
        return {"error": f"ledger {ledger_name} not initialized at {path}"}
    store = LedgerStore(path, ledger_name)
    e = store.get(entry_id)
    if e is None:
        return {"error": f"entry not found: {ledger_name}/{entry_id}"}
    evidence_generations = _card_generation_map(rd) if cutoff is not None else {}
    if not _visible_at_generation(
        e.to_dict(),
        cutoff,
        evidence_generations=evidence_generations,
    ):
        return _cutoff_error(f"{ledger_name}/{entry_id}", cutoff)
    return e.to_dict()


def resolve_source_ref(
    run_dir,
    source_ref: dict[str, Any],
    max_generation_id: int | None = None,
) -> dict[str, Any]:
    """Lazy-load the raw file behind an evidence_card.source_ref.

    Use SPARINGLY: this loads the original JSON / YAML, which is verbose.
    Prefer get_evidence_card for normal operation.
    """
    from praxist.plugins.workflow_stages.research_loop.backend.research_memory.source_resolver import (
        SourceResolver,
    )

    rd = _safe_run_dir(run_dir)
    cutoff = _runtime_generation_limit(max_generation_id)
    if not _visible_at_generation(source_ref, cutoff):
        return _cutoff_error("source_ref", cutoff)
    resolved = SourceResolver(rd).resolve(source_ref)
    if not _resolved_source_visible(resolved, cutoff):
        return _cutoff_error("resolved_source_ref", cutoff)
    return resolved


# ----------------------------------------------------------------------------
# MCP server factory — closes over run_dir so SDK handlers don't need it.
# ----------------------------------------------------------------------------


def create_memory_tools_server(run_dir):
    """Create an MCP server exposing the memory query tools to PI agents.

    The PI panel attaches this server when multi_pi.enabled=True. Peers do
    NOT see this server (their allowed_tools list does not include
    mcp__memory-tools__* entries).
    """
    if create_sdk_mcp_server is None or tool is None:
        raise ImportError("claude_agent_sdk is required for MCP tools")

    rd = Path(run_dir).resolve()

    @tool(
        "get_evidence_card",
        "Fetch the full evidence card for a given evidence_id. Use when "
        "the role-specific pack omitted detail you need.",
        {"evidence_id": str, "max_generation_id": int},
    )
    async def _get_evidence_card(args):
        out = get_evidence_card(
            rd,
            args.get("evidence_id", ""),
            max_generation_id=args.get("max_generation_id"),
        )
        return {"content": [{"type": "text", "text": json.dumps(out, default=str)}]}

    @tool(
        "query_evidence_cards",
        "Filter evidence cards by claim_id / mechanism / peer_id / generation_id "
        "/ is_negative. Returns id + short interpretation; follow up with "
        "get_evidence_card for full details.",
        {
            "claim_id": str,
            "mechanism": str,
            "peer_id": str,
            "generation_id": int,
            "is_negative": bool,
            "limit": int,
            "max_generation_id": int,
        },
    )
    async def _query_evidence_cards(args):
        out = query_evidence_cards(
            rd,
            claim_id=args.get("claim_id") or None,
            mechanism=args.get("mechanism") or None,
            peer_id=args.get("peer_id") or None,
            generation_id=args.get("generation_id"),
            is_negative=args.get("is_negative"),
            limit=int(args.get("limit") or 20),
            max_generation_id=args.get("max_generation_id"),
        )
        return {"content": [{"type": "text", "text": json.dumps(out, default=str)}]}

    @tool(
        "query_coverage_matrix",
        "Check if a (variant_family, parameter) grid or a (bridge_pair, "
        "bridge_dimension) has been covered. MUST be called by Bridge "
        "contracts before assignment.",
        {
            "variant_family": str,
            "parameter": str,
            "bridge_pair": list,
            "bridge_dimension": str,
            "max_generation_id": int,
        },
    )
    async def _query_coverage_matrix(args):
        out = query_coverage_matrix(
            rd,
            variant_family=args.get("variant_family") or None,
            parameter=args.get("parameter") or None,
            bridge_pair=args.get("bridge_pair") or None,
            bridge_dimension=args.get("bridge_dimension") or None,
            max_generation_id=args.get("max_generation_id"),
        )
        return {"content": [{"type": "text", "text": json.dumps(out, default=str)}]}

    @tool(
        "list_active_claims",
        "List active claims with status, confidence, boundary, and supports/challenges counts.",
        {},
    )
    async def _list_active_claims(args):
        out = list_active_claims(rd, max_generation_id=args.get("max_generation_id"))
        return {"content": [{"type": "text", "text": json.dumps(out, default=str)}]}

    @tool(
        "list_open_objections",
        "List open / experiment-assigned dissent entries.",
        {},
    )
    async def _list_open_objections(args):
        out = list_open_objections(rd, max_generation_id=args.get("max_generation_id"))
        return {"content": [{"type": "text", "text": json.dumps(out, default=str)}]}

    @tool(
        "get_ledger_entry",
        "Read a single entry from any ledger by name + entry id.",
        {"ledger_name": str, "entry_id": str, "max_generation_id": int},
    )
    async def _get_ledger_entry(args):
        out = get_ledger_entry(
            rd,
            args.get("ledger_name", ""),
            args.get("entry_id", ""),
            max_generation_id=args.get("max_generation_id"),
        )
        return {"content": [{"type": "text", "text": json.dumps(out, default=str)}]}

    @tool(
        "resolve_source_ref",
        "Lazy-load the raw file behind an evidence_card.source_ref. Use SPARINGLY (verbose).",
        {"source_ref": dict, "max_generation_id": int},
    )
    async def _resolve_source_ref(args):
        out = resolve_source_ref(
            rd,
            args.get("source_ref", {}),
            max_generation_id=args.get("max_generation_id"),
        )
        return {"content": [{"type": "text", "text": json.dumps(out, default=str)}]}

    return create_sdk_mcp_server(
        "memory-tools",
        tools=[
            _get_evidence_card,
            _query_evidence_cards,
            _query_coverage_matrix,
            _list_active_claims,
            _list_open_objections,
            _get_ledger_entry,
            _resolve_source_ref,
        ],
    )


def create_tool_plugin() -> dict[str, object]:
    """Manifest entrypoint that exposes PI research-memory query tools."""
    return {
        "tool_server_ref": "tool_server:memory_tools",
        "server_name": "memory-tools",
        "factory": "praxist.plugins.tools.memory_tools.adapter:create_memory_tools_server",
        "tool_names": [
            "get_evidence_card",
            "query_evidence_cards",
            "query_coverage_matrix",
            "list_active_claims",
            "list_open_objections",
            "get_ledger_entry",
            "resolve_source_ref",
        ],
        "visibility": ["panel"],
        "required_capability": "tool_server.memory_tools",
        "requires_run_dir": True,
        "requires_multi_pi": True,
    }
