"""
Finding Graph Query MCP tools — read-only navigation over the sidecar graph.

Per the finding-graph section of docs/concepts/architecture.md, v1 exposes
three read-only tools:

    get_finding_neighbors       one-hop around a finding
    get_finding_subgraph        depth-limited subgraph, default depth=1
    get_unlinked_recent_findings  recent findings not yet in the graph
                                  (protects exploration diversity)

These are ADVISORY. Edges are navigation, not conclusions. Raw findings
in shared_findings/ remain the source of truth. If the graph is absent
(maintainer disabled, DB locked, etc.) these tools return empty results
gracefully — they never block the main MCP flow.
"""

import json
import logging
from typing import Any

from praxist.plugins.tools.result_envelope import (
    coerce_inline_limit,
    with_tool_output_envelope,
)

try:
    from claude_agent_sdk import create_sdk_mcp_server, tool
except ImportError:
    tool = None
    create_sdk_mcp_server = None

logger = logging.getLogger(__name__)


def _text_result(data: Any) -> dict[str, Any]:
    text = json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data
    return {"content": [{"type": "text", "text": text}]}


def _error_result(msg: str) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps({"error": msg})}],
        "is_error": True,
    }


def _trim_finding(finding: dict[str, Any], max_content: int = 800) -> dict[str, Any]:
    """Keep the MCP response small. Truncate content/notes to prevent blowing
    up an agent's context when a subgraph has 20+ nodes."""
    out = {
        "id": finding.get("id"),
        "finding_type": finding.get("finding_type"),
        "title": finding.get("title"),
        "variant_name": finding.get("variant_name"),
        "metrics": finding.get("metrics", {}),
        "peer_id": finding.get("peer_id"),
        "generation_id": finding.get("generation_id"),
        "timestamp": finding.get("timestamp"),
    }
    if "graph_depth" in finding:
        out["graph_depth"] = finding["graph_depth"]
    content = finding.get("content") or ""
    if len(content) > max_content:
        out["content"] = content[:max_content] + " ...[truncated]"
    else:
        out["content"] = content
    return out


# --- handlers ---------------------------------------------------------------


async def _handle_get_finding_neighbors(args: dict[str, Any]) -> dict[str, Any]:
    """Return the one-hop neighborhood of a finding.

    The graph is ADVISORY: edges describe stated relationships, not proven
    conclusions. Raw findings remain the source of truth.
    """
    finding_id = args.get("finding_id", "").strip()
    if not finding_id:
        return _error_result("finding_id required")
    edge_types = args.get("edge_types") or None
    if isinstance(edge_types, str):
        try:
            edge_types = json.loads(edge_types) if edge_types.startswith("[") else [edge_types]
        except json.JSONDecodeError:
            edge_types = [edge_types]
    min_confidence = float(args.get("min_confidence", 0.55))
    limit = int(args.get("limit", 30))
    inline_limit = coerce_inline_limit(args.get("inline_limit", limit), default=limit)

    try:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            local_store as ls,
        )

        ls.init_db()
        with ls._get_conn(readonly=True) as conn:
            frow = conn.execute(
                "SELECT * FROM findings WHERE id = ?",
                (finding_id,),
            ).fetchone()
        if frow is None:
            return _error_result(f"finding_id {finding_id} not found")
        finding = ls._row_to_finding(frow)

        out_edges = ls.get_edges_for_finding(
            finding_id,
            direction="out",
            edge_types=edge_types,
            min_confidence=min_confidence,
            limit=limit,
        )
        in_edges = ls.get_edges_for_finding(
            finding_id,
            direction="in",
            edge_types=edge_types,
            min_confidence=min_confidence,
            limit=limit,
        )
        neighbor_ids = {e["dst_finding_id"] for e in out_edges} | {
            e["src_finding_id"] for e in in_edges
        }
        neighbor_findings_raw = []
        neighbor_findings = []
        if neighbor_ids:
            with ls._get_conn(readonly=True) as conn:
                placeholders = ",".join("?" * len(neighbor_ids))
                rows = conn.execute(
                    f"SELECT * FROM findings WHERE id IN ({placeholders})",
                    list(neighbor_ids),
                ).fetchall()
            neighbor_findings_raw = [ls._row_to_finding(r) for r in rows]
            neighbor_findings = [_trim_finding(f) for f in neighbor_findings_raw]

        payload = {
            "finding": _trim_finding(finding),
            "incoming_edges": in_edges,
            "outgoing_edges": out_edges,
            "neighbor_findings": neighbor_findings,
            "note": (
                "Edges are advisory navigation, not conclusions. Raw finding "
                "content remains the source of truth."
            ),
        }
        raw_payload = {
            **payload,
            "finding": finding,
            "neighbor_findings": neighbor_findings_raw,
        }
        return _text_result(
            with_tool_output_envelope(
                payload,
                tool_name="get_finding_neighbors",
                list_fields=("incoming_edges", "outgoing_edges", "neighbor_findings"),
                inline_limit=inline_limit,
                full_payload=raw_payload,
            )
        )
    except Exception as e:
        logger.debug("get_finding_neighbors failed: %s", e)
        return _error_result(f"graph query failed: {e}")


async def _handle_get_finding_subgraph(args: dict[str, Any]) -> dict[str, Any]:
    """Depth-limited subgraph starting from a finding.

    Default max_depth=1 to keep responses compact. Set to 2 only when you
    need context beyond immediate neighbors.
    """
    finding_id = args.get("finding_id", "").strip()
    if not finding_id:
        return _error_result("finding_id required")
    max_depth = int(args.get("max_depth", 1))
    min_confidence = float(args.get("min_confidence", 0.55))
    max_nodes = int(args.get("max_nodes", 50))
    inline_limit = coerce_inline_limit(args.get("inline_limit", max_nodes), default=max_nodes)
    edge_types = args.get("edge_types") or None
    if isinstance(edge_types, str):
        try:
            edge_types = json.loads(edge_types) if edge_types.startswith("[") else [edge_types]
        except json.JSONDecodeError:
            edge_types = [edge_types]

    try:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            local_store as ls,
        )

        ls.init_db()
        sg = ls.get_subgraph(
            finding_id,
            max_depth=max_depth,
            min_confidence=min_confidence,
            edge_types=edge_types,
            max_nodes=max_nodes,
        )
        payload = {
            "start_finding_id": sg["start_finding_id"],
            "max_depth": sg["max_depth"],
            "min_confidence": sg["min_confidence"],
            "nodes": [_trim_finding(n) for n in sg["nodes"]],
            "edges": sg["edges"],
            "truncated": sg["truncated"],
            "note": (
                "Edges are advisory. When truncated=true, increase max_nodes "
                "or narrow edge_types to see the full local structure."
            ),
        }
        raw_payload = {**payload, "nodes": sg["nodes"]}
        return _text_result(
            with_tool_output_envelope(
                payload,
                tool_name="get_finding_subgraph",
                list_fields=("nodes", "edges"),
                inline_limit=inline_limit,
                full_payload=raw_payload,
            )
        )
    except Exception as e:
        logger.debug("get_finding_subgraph failed: %s", e)
        return _error_result(f"graph query failed: {e}")


async def _handle_get_unlinked_recent_findings(args: dict[str, Any]) -> dict[str, Any]:
    """Recent findings that have NO graph edges yet.

    Use this to surface exploration directions the graph hasn't absorbed —
    helps you avoid being pulled into the most-connected thread when
    something new and unrelated deserves attention.
    """
    hours = float(args.get("hours", 6.0))
    limit = int(args.get("limit", 30))
    inline_limit = coerce_inline_limit(args.get("inline_limit", limit), default=limit)
    try:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            local_store as ls,
        )

        ls.init_db()
        findings = ls.get_unlinked_recent_findings(hours=hours, limit=limit)
        payload = {
            "unlinked_findings": [_trim_finding(f) for f in findings],
            "window_hours": hours,
            "note": (
                "These findings have no edges yet. Good candidates for new "
                "directions or unabsorbed signal."
            ),
        }
        raw_payload = {**payload, "unlinked_findings": findings}
        return _text_result(
            with_tool_output_envelope(
                payload,
                tool_name="get_unlinked_recent_findings",
                list_fields=("unlinked_findings",),
                inline_limit=inline_limit,
                full_payload=raw_payload,
            )
        )
    except Exception as e:
        logger.debug("get_unlinked_recent_findings failed: %s", e)
        return _error_result(f"graph query failed: {e}")


# --- tool registrations -----------------------------------------------------

get_finding_neighbors = None
get_finding_subgraph = None
get_unlinked_recent_findings = None

if tool is not None:
    get_finding_neighbors = tool(
        "get_finding_neighbors",
        "Return the one-hop graph neighborhood of a finding. edge_types is "
        "optional (JSON array of: related_to, derived_from, updates, "
        "supports, challenges). Graph is ADVISORY — edges are navigation, "
        "not conclusions; raw finding content is the source of truth.",
        {
            "finding_id": str,
            "edge_types": str,  # JSON string of list, or single type, or empty
            "min_confidence": float,
            "limit": int,
            "inline_limit": int,
        },
    )(_handle_get_finding_neighbors)

    get_finding_subgraph = tool(
        "get_finding_subgraph",
        "Depth-limited subgraph from a start finding. Default max_depth=1 "
        "keeps the response compact; set max_depth=2 only when you need "
        "context beyond immediate neighbors. max_nodes caps output size.",
        {
            "finding_id": str,
            "max_depth": int,
            "min_confidence": float,
            "max_nodes": int,
            "edge_types": str,
            "inline_limit": int,
        },
    )(_handle_get_finding_subgraph)

    get_unlinked_recent_findings = tool(
        "get_unlinked_recent_findings",
        "Recent findings (default last 6 hours) that have NO graph edges "
        "yet. Use this to surface exploration directions not yet absorbed "
        "into any cluster — helps preserve research diversity.",
        {
            "hours": float,
            "limit": int,
            "inline_limit": int,
        },
    )(_handle_get_unlinked_recent_findings)


def create_finding_graph_query_server():
    """Create the finding-graph-query MCP server."""
    if create_sdk_mcp_server is None or tool is None:
        raise ImportError("claude_agent_sdk is required for MCP tools")
    return create_sdk_mcp_server(
        "finding-graph-query",
        tools=[get_finding_neighbors, get_finding_subgraph, get_unlinked_recent_findings],
    )


def create_tool_plugin() -> dict[str, object]:
    """Manifest entrypoint that exposes finding graph query tools."""
    return {
        "tool_server_ref": "tool_server:finding_graph_query",
        "server_name": "finding-graph-query",
        "factory": "praxist.plugins.tools.finding_graph_query.adapter:create_finding_graph_query_server",
        "tool_names": [
            "get_finding_neighbors",
            "get_finding_subgraph",
            "get_unlinked_recent_findings",
        ],
        "visibility": ["peer", "panel"],
        "required_capability": "tool_server.finding_graph_query",
        "handlers": {
            "get_finding_neighbors": "praxist.plugins.tools.finding_graph_query.adapter:_handle_get_finding_neighbors",
            "get_finding_subgraph": "praxist.plugins.tools.finding_graph_query.adapter:_handle_get_finding_subgraph",
            "get_unlinked_recent_findings": "praxist.plugins.tools.finding_graph_query.adapter:_handle_get_unlinked_recent_findings",
        },
    }
