"""
Finding Graph visualization — produces a single self-contained HTML file
that renders the current state of the graph with vis-network.

Why HTML instead of matplotlib PNG:
  - 665 nodes × 3412 edges is already busy on a static image; interactivity
    (zoom, pan, click, hover tooltips) is how you actually navigate this
    scale of graph
  - the "graph is navigation, not conclusion" philosophy (design doc §1.3)
    maps naturally to hoverable UI with filters — users should be able to
    see the raw finding content on demand, not squint at a thumbnail
  - one self-contained .html drops anywhere and works — vis-network is
    inlined into the page so the file works completely offline once rendered

What the page shows:
  - one node per finding
    - color encodes `finding_type` (result/insight/hypothesis/error)
    - size encodes local degree (how much cross-peer traffic it attracted)
    - hover tooltip: title, peer, generation, metrics, content snippet
  - one visual edge per SQLite edge row
    - color encodes `edge_type` (supports/challenges/updates/derived_from/related_to)
    - opacity and width encode `confidence`
    - hover tooltip: edge_type, confidence, rationale
  - sidebar filters: edge_type checkboxes, confidence slider, search box
  - legend + graph health summary in the header

Written each FindingGraphMaintainer cycle to ``run_dir/graph/graph.html``
via atomic write so the file in place is always complete.
"""

from __future__ import annotations

import contextlib
import hashlib
import html
import json
import logging
import math
import os
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# --- vis-network bundling ---------------------------------------------------
# We inline the vis-network JS + CSS into each rendered HTML so the file is
# truly self-contained and works offline after download. Must use the
# `standalone/umd` build — the `peer/umd` build is ~260 KB smaller but
# requires hammerjs / vis-data / component-emitter to be loaded separately,
# which silently fails with no visible error (vis object defined but
# vis.Network() renders nothing). `standalone` bundles everything.
_VIS_NETWORK_JS_URL = (
    "https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js"
)
_VIS_NETWORK_CSS_URL = "https://cdn.jsdelivr.net/npm/vis-network@9.1.9/styles/vis-network.min.css"

# Cache location: keep it outside the repo, one copy shared across runs.
_ASSET_CACHE_DIR = Path(os.environ.get("PRAXIST_VIZ_CACHE", Path.home() / ".cache" / "praxist_viz"))


# Minimum sizes for cached assets. The standalone vis-network bundle is
# ~670 KB and the stylesheet is ~215 KB. A file shorter than this is a
# truncated download (urllib can silently return partial bodies on a
# dropped connection for gzipped responses). Without this check, a
# corrupt cache persists across renders and the HTML silently fails to
# load — code on the page never reaches the `vis-missing-banner` check
# because the bundle itself throws a syntax error mid-parse.
_MIN_SIZES = {
    "vis-network-9.1.9.min.js": 400_000,  # standalone bundle ≥ 600 KB
    "vis-network-9.1.9.min.css": 150_000,  # stylesheet ≥ 200 KB
}


def _download(url: str, dest: Path) -> None:
    """Atomic download: fetch to a tmp file, verify minimum size, then
    rename into place. A truncated/corrupt response never replaces a
    good cached copy, and a partially-written file never survives a
    crash mid-write."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = resp.read()
    min_size = _MIN_SIZES.get(dest.name, 1)
    if len(data) < min_size:
        raise OSError(
            f"downloaded {dest.name} is {len(data)} bytes, "
            f"expected at least {min_size} — treating as corrupt"
        )
    # Atomic write: tmp file in same dir, then rename.
    import tempfile

    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dest.name}.",
        suffix=".tmp",
        dir=str(dest.parent),
    )
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(data)
        os.replace(tmp_name, dest)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _asset_is_healthy(path: Path) -> bool:
    """Reject corrupt, truncated, or layout-incompatible cached assets."""
    try:
        if not path.exists():
            return False
        min_size = _MIN_SIZES.get(path.name, 1)
        return path.stat().st_size >= min_size
    except OSError:
        return False


def _load_vis_assets() -> dict[str, str | None]:
    """Return {"js": ..., "css": ...} with inline asset content. On fetch
    failure returns None for the missing asset; caller falls back to CDN
    <script>/<link> tags."""
    js_path = _ASSET_CACHE_DIR / "vis-network-9.1.9.min.js"
    css_path = _ASSET_CACHE_DIR / "vis-network-9.1.9.min.css"

    out: dict[str, str | None] = {"js": None, "css": None}
    for url, path, key in (
        (_VIS_NETWORK_JS_URL, js_path, "js"),
        (_VIS_NETWORK_CSS_URL, css_path, "css"),
    ):
        if not _asset_is_healthy(path):
            # Remove any stale/corrupt file before re-fetching so we
            # never end up in a "looks present but broken" state.
            if path.exists():
                try:
                    path.unlink()
                    logger.warning("removed stale cached asset: %s", path.name)
                except OSError as e:
                    logger.warning("could not unlink %s: %s", path, e)
            try:
                _download(url, path)
                logger.info(
                    "cached vis-network asset: %s (%d KB)", path.name, path.stat().st_size // 1024
                )
            except Exception as e:
                logger.warning(
                    "could not cache %s (%s); HTML will fall back to remote CDN tag",
                    path.name,
                    e,
                )
                continue
        try:
            out[key] = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("could not read cached %s: %s", path, e)
    return out


# --- visual encoding tables -------------------------------------------------

# finding_type → hex color. Chosen to be visually distinct and
# colorblind-friendly (Okabe-Ito palette).
_TYPE_COLOR = {
    "result": "#009E73",  # green   — confirmed measurements
    "insight": "#0072B2",  # blue    — observed patterns
    "hypothesis": "#E69F00",  # orange  — proposals, not yet tested
    "error": "#D55E00",  # red     — negatives / dead ends
}
_TYPE_COLOR_DEFAULT = "#888888"

# Rest length of a real edge's spring when physics is on. Shorter
# length → physics pulls the two ends closer together at equilibrium,
# which for cross-ring edges translates into smaller angular
# separation (the radial distance is locked by the type leash). The
# tiers encode "how direct is this progression":
#   - derived_from: explicit id reference → most direct lineage signal
#   - supports/challenges: language + shared object → direct but inferred
#   - updates: same variant iteration → lateral progression
#   - related_to: fallback weak coupling → stay far by default
#   - challenges: disagreement → REPEL (large preferred spring length
#     so the spring wants to keep them far apart; against the
#     competing attractive forces from other edges + peer-ring anchors,
#     this lands challenged pairs visibly farther apart than
#     supporting pairs, which matches their semantic meaning).
# Values picked so a strong pair ends up roughly 1/3 as far apart
# angularly as a weak pair, while still leaving edges slack enough
# that repulsion between unrelated nodes can spread them out.
_EDGE_SPRING_LENGTH = {
    "derived_from": 45,
    "supports": 75,
    "challenges": 400,  # ← repulsive: much longer preferred length
    "updates": 85,
    "related_to": 150,
}

# edge_type → hex color
_EDGE_COLOR = {
    "supports": "#009E73",  # green   — reinforcement
    "challenges": "#D55E00",  # red     — opposition
    "updates": "#CC79A7",  # magenta — revision
    "derived_from": "#56B4E9",  # sky     — lineage
    "related_to": "#BBBBBB",  # gray    — weak coupling (default)
}


def _truncate(s: str | None, n: int = 400) -> str:
    if not s:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n] + "…"


def _short_title(s: str | None, n: int = 60) -> str:
    if not s:
        return "(untitled)"
    s = str(s).replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def _node_size(degree: int) -> float:
    """Map degree → node size in pixels. Sub-linear so very-high-degree
    nodes don't dominate the canvas. Returns float in [10, 40]."""
    import math

    return 10.0 + 30.0 * (1.0 - math.exp(-degree / 12.0))


def _edge_width(conf: float) -> float:
    """Confidence → line width (1.0 px at 0.55, 3.5 px at 1.00)."""
    return 1.0 + 2.5 * max(0.0, (float(conf) - 0.55) / 0.45)


def _edge_opacity(conf: float) -> float:
    """Confidence → line opacity (0.3 at 0.55, 1.0 at 1.00)."""
    return 0.3 + 0.7 * max(0.0, (float(conf) - 0.55) / 0.45)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


# --- top-K + Pareto extraction ----------------------------------------------
# Ranking semantics come exclusively from the task snapshot. Findings remain
# fully visible without one, but Praxist does not guess which free-form number is
# a primary metric, which direction is better, or which pair forms a Pareto
# plane.


def _load_run_task_spec() -> dict[str, Any] | None:
    """Load the task_spec snapshot that the orchestrator drops into
    ``<run_dir>/task_spec.yaml`` at init. This is the canonical source
    for `primary_metric`, `direction`, `aux_metrics`, `baselines` —
    everything the viz needs to render a task-appropriate leaderboard.

    Returns None if the snapshot is absent. The graph can still render every
    finding and metric, but ranking decorations are omitted because their
    meaning cannot be recovered safely.
    """
    run_dir_str = os.environ.get("LOCAL_STORE_DIR")
    if not run_dir_str:
        return None
    p = Path(run_dir_str) / "task_spec.yaml"
    if not p.exists():
        return None
    try:
        import yaml as _yaml

        with open(p) as f:
            return _yaml.safe_load(f) or None
    except Exception:
        return None


def _task_primary_metric(spec: dict[str, Any] | None) -> str | None:
    if not spec:
        return None
    return ((spec.get("evaluation") or {}).get("primary_metric")) or None


def _task_direction(spec: dict[str, Any] | None) -> str:
    """'maximize' (default) or 'minimize'."""
    if not spec:
        return "maximize"
    d = (spec.get("evaluation") or {}).get("direction") or "maximize"
    return "minimize" if str(d).lower().startswith("min") else "maximize"


def _task_secondary_metric(spec: dict[str, Any] | None) -> str | None:
    """First task-declared auxiliary display axis, if one exists."""
    if not spec:
        return None
    aux = (spec.get("evaluation") or {}).get("aux_metrics") or []
    if isinstance(aux, list) and aux:
        return str(aux[0])
    return None


def _task_metric_direction(spec: dict[str, Any] | None, metric: str | None) -> str | None:
    """Return an explicitly declared direction for ``metric``."""
    if not spec or not metric:
        return None
    evaluation = spec.get("evaluation") or {}
    if metric == evaluation.get("primary_metric"):
        return _task_direction(spec)

    groups: list[Any] = [evaluation.get("anchor_metrics") or []]
    for lane in evaluation.get("frontier_lanes") or []:
        if not isinstance(lane, dict):
            continue
        groups.extend((lane.get("axes") or [], lane.get("optional_axes") or []))
    for group in groups:
        for entry in group if isinstance(group, list) else []:
            if isinstance(entry, dict):
                name, direction = entry.get("name"), entry.get("direction")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                name, direction = entry[0], entry[1]
            else:
                continue
            if str(name) == metric and str(direction).lower() in {"maximize", "minimize"}:
                return str(direction).lower()
    return None


def _extract_primary_secondary(
    finding: dict[str, Any],
    task_spec: dict[str, Any] | None = None,
) -> tuple[float | None, float | None, str | None, str | None]:
    """Return (primary, secondary, primary_key, secondary_key).

    Task-spec-driven (preferred): if task_spec defines primary_metric and
    aux_metrics, use those exact keys from `finding.metrics` — no value-
    range filtering, no substring heuristics. This is how the function
    should be called for any task; it's fully generic.

    Without a task-declared primary metric, return no ranking axes. The raw
    metrics still render in the finding payload; only unsupported semantic
    inference is withheld.
    """
    metrics = finding.get("metrics") or {}
    if not isinstance(metrics, dict):
        return None, None, None, None
    # ---- task-spec-driven, no heuristics ---------------------------------
    prim_key = _task_primary_metric(task_spec)
    sec_key = _task_secondary_metric(task_spec)
    if prim_key:
        primary_val: float | None = None
        secondary_val: float | None = None
        pv = metrics.get(prim_key)
        if isinstance(pv, (int, float)) and not isinstance(pv, bool):
            try:
                primary_val = float(pv)
            except (TypeError, ValueError):
                primary_val = None
        if sec_key:
            sv = metrics.get(sec_key)
            if isinstance(sv, (int, float)) and not isinstance(sv, bool):
                try:
                    secondary_val = float(sv)
                except (TypeError, ValueError):
                    secondary_val = None
        return (
            primary_val,
            secondary_val,
            prim_key if primary_val is not None else None,
            sec_key if secondary_val is not None else None,
        )

    return None, None, None, None


def _compute_top_and_pareto(
    all_findings: list[dict[str, Any]],
    top_k: int = 3,
    task_spec: dict[str, Any] | None = None,
) -> tuple[dict[str, int], set]:
    """Return (top_k_ranks, pareto_ids).

    top_k_ranks: {finding_id → rank 1..top_k} for the K BEST primary
    values (best = largest if direction='maximize', smallest if
    'minimize'). Rank 1 is the best.

    pareto_ids: set of finding ids that sit on the Pareto frontier of
    (primary ↑, secondary ↓) — needs both metrics present. A point is
    on the front iff no other point strictly dominates it on BOTH axes.
    Findings without a secondary metric are not eligible for the front
    but may still appear in top_k.
    """
    direction = _task_direction(task_spec)
    # Multiplier used to flip compares for minimize-direction tasks
    # without duplicating the sort/dominate code below.
    dir_mul = -1.0 if direction == "minimize" else 1.0
    secondary_metric = _task_secondary_metric(task_spec)
    secondary_direction = _task_metric_direction(task_spec, secondary_metric)
    secondary_mul = -1.0 if secondary_direction == "minimize" else 1.0

    extracted = []
    for f in all_findings:
        fid = f.get("id")
        if not fid:
            continue
        # Only `result`-type findings are eligible for leaderboard
        # ranking. `insight`-type findings often quote OTHER variants'
        # numbers in their metrics (e.g., a synthesis note comparing
        # peer0 and peer2 results) — including them in leaderboard
        # ranking causes cross-contamination where a finding from
        # variant X shows up with variant Y's measurement. Results
        # are the only findings that authoritatively own their own
        # measurement.
        if f.get("finding_type") != "result":
            continue
        primary, secondary, pk, sk = _extract_primary_secondary(f, task_spec=task_spec)
        if primary is None:
            continue
        extracted.append(
            {
                "id": fid,
                "primary": primary,
                "secondary": secondary,
                "primary_key": pk,
                "secondary_key": sk,
            }
        )

    # Top-K by primary. For direction='minimize', we want smallest first.
    top_k_ranks: dict[str, int] = {}
    ranked = sorted(extracted, key=lambda x: -dir_mul * x["primary"])[:top_k]
    for rank, item in enumerate(ranked, start=1):
        top_k_ranks[item["id"]] = rank

    # Pareto frontier. For minimize-direction tasks, "better on primary"
    # means smaller; we flip the sign so the dominance compares stay
    # "larger is better" and the logic below is unchanged.
    with_both = (
        [x for x in extracted if x["secondary"] is not None]
        if secondary_direction is not None
        else []
    )
    pareto_ids: set = set()
    for i, a in enumerate(with_both):
        dominated = False
        for j, b in enumerate(with_both):
            if i == j:
                continue
            # Both directions are task-declared; no metric-name heuristics.
            ap = dir_mul * a["primary"]
            bp = dir_mul * b["primary"]
            asecondary = secondary_mul * a["secondary"]
            bsecondary = secondary_mul * b["secondary"]
            ge = bp >= ap
            secondary_ge = bsecondary >= asecondary
            strict = (bp > ap) or (bsecondary > asecondary)
            if ge and secondary_ge and strict:
                dominated = True
                break
        if not dominated:
            pareto_ids.add(a["id"])

    return top_k_ranks, pareto_ids


# --- payload construction ---------------------------------------------------


def build_viz_payload() -> dict[str, Any]:
    """Query local_store and return a {nodes, edges, meta} dict ready for
    embedding in the HTML template. Read-only; does not modify any tables."""
    from praxist.plugins.graph_maintainers.finding_graph_mvp.engine import compute_graph_health
    from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store as ls

    ls.init_db()

    # Task-spec snapshot (dropped into run_dir by the orchestrator). This
    # is what tells us the primary metric name, direction and aux metric.
    task_spec_yaml = _load_run_task_spec()

    all_findings = ls.get_all_findings()
    findings_by_id = {f["id"]: f for f in all_findings if f.get("id")}

    # --- Top-K + Pareto extraction -----------------------------------------
    # Use primary_metric / direction / aux_metrics verbatim. Without a task
    # snapshot, findings still render but receive no inferred leaderboard or
    # Pareto decoration.
    top_k_ranks, pareto_ids = _compute_top_and_pareto(
        all_findings,
        top_k=3,
        task_spec=task_spec_yaml,
    )

    # Pull all edges in one query; degree is accumulated from this.
    degrees: dict[str, int] = {}
    edge_rows: list[dict[str, Any]] = []
    with ls._get_conn(readonly=True) as conn:
        for r in conn.execute(
            "SELECT * FROM finding_edges ORDER BY confidence DESC, created_at ASC"
        ).fetchall():
            e = dict(r)
            edge_rows.append(e)
            degrees[e["src_finding_id"]] = degrees.get(e["src_finding_id"], 0) + 1
            degrees[e["dst_finding_id"]] = degrees.get(e["dst_finding_id"], 0) + 1

    # Deterministic cluster layout: place each peer at an angle derived
    # from a hash of its peer_id (NOT sort-position). Hash-based angles
    # keep layout stable across re-renders: when a new peer joins, the
    # existing peers stay where they were — only the new one appears.
    # Sort-based indexing (the old approach) rotated every peer whenever
    # a new peer_id was added, making "refresh and compare" impossible.
    # Ship coordinates in the payload and DISABLE physics by default;
    # force-atlas2 on 665 nodes × 3k+ edges freezes the main thread.
    # Users can opt in via the "Run layout" button.
    peer_ids = {(f.get("peer_id") or "_none") for f in findings_by_id.values()}

    def _peer_angle(pid: str) -> float:
        # Use the top 32 bits of md5 as a deterministic angle in [0, 2π).
        h = int(hashlib.md5(pid.encode("utf-8")).hexdigest()[:8], 16)
        return (h / 0xFFFFFFFF) * 2 * math.pi

    peer_angle = {p: _peer_angle(p) for p in peer_ids}
    num_peers = max(1, len(peer_ids))

    # Radial banding by finding_type: place each type on a concentric
    # ring so they're visually separated even when shown together.
    # Radii are spaced so there's a VISIBLE empty band between rings
    # even after jitter is added (jitter_max < half the ring spacing).
    # Under physics, triple-stacked leash edges (see below) keep
    # nodes pinned to their ring so the separation survives.
    #  - hypothesis:  compact inner ring
    #  - insight:     middle ring with 350 unit empty gap from inner
    #  - result:      outer ring with 400 unit empty gap from middle
    # The edges between types then read as radial lines (cross-ring)
    # vs same-type edges that stay tangential.
    _TYPE_RADIUS = {
        "hypothesis": 250.0,  # inner
        "insight": 700.0,  # middle
        "result": 1250.0,  # outer
        "error": 1000.0,  # between insight and result
        "unknown": 700.0,
    }
    # Tight jitter so nodes stay clearly inside their own ring.
    # With jitter_radius=110 and ring spacing 450-550, there's a ~200
    # unit empty band between the outermost jittered node of the inner
    # ring and the innermost jittered node of the next ring. The rings
    # read as three distinct orbits even on a dense graph.
    jitter_radius = 90.0 + 4.0 * num_peers

    nodes = []
    for fid, f in findings_by_id.items():
        ftype = (f.get("finding_type") or "").lower()
        color = _TYPE_COLOR.get(ftype, _TYPE_COLOR_DEFAULT)
        deg = degrees.get(fid, 0)
        metrics = f.get("metrics") or {}
        metrics_str = ""
        if isinstance(metrics, dict) and metrics:
            pairs = list(metrics.items())[:4]
            metrics_str = ", ".join(
                f"<b>{html.escape(str(k))}</b>={html.escape(str(v))}" for k, v in pairs
            )
        # Per-node performance tags. Rank 1/2/3 go on the TOP-K podium;
        # Pareto frontier membership is an orthogonal signal (a node
        # can be top-K AND on the Pareto front — both get rendered).
        rank = top_k_ranks.get(fid)
        is_pareto = fid in pareto_ids
        primary_val, secondary_val, pk_key, sk_key = _extract_primary_secondary(
            f,
            task_spec=task_spec_yaml,
        )

        # Initial coordinates.
        pid = f.get("peer_id") or "_none"
        sector_theta = peer_angle.get(pid, 0.0)
        cluster_radius = _TYPE_RADIUS.get(ftype, _TYPE_RADIUS["unknown"])
        cx = cluster_radius * math.cos(sector_theta)
        cy = cluster_radius * math.sin(sector_theta)
        h = int(hashlib.md5(fid.encode("utf-8")).hexdigest()[:12], 16)
        # Polar jitter within the sector; sqrt keeps distribution uniform
        # across disk area instead of clumping near the center.
        jr = jitter_radius * math.sqrt(((h >> 32) & 0xFFFF) / 0xFFFF)
        jt = (h & 0xFFFFFFFF) / 0xFFFFFFFF * 2 * math.pi
        x = cx + jr * math.cos(jt)
        y = cy + jr * math.sin(jt)

        nodes.append(
            {
                "id": fid,
                "label": _short_title(f.get("title"), n=28),
                "title_full": f.get("title") or "",
                "content_snippet": _truncate(f.get("content"), 300),
                "finding_type": ftype or "unknown",
                "color": color,
                "size": _node_size(deg),
                "peer_id": f.get("peer_id") or "",
                "generation_id": f.get("generation_id"),
                "timestamp": f.get("timestamp") or "",
                "degree": deg,
                "metrics_html": metrics_str,
                "variant_name": f.get("variant_name") or "",
                "x": round(x, 1),
                "y": round(y, 1),
                # Performance tags — None when the finding has no primary
                # metric, populated for results / quantitative insights.
                "rank": rank,
                "is_pareto": is_pareto,
                "primary_value": round(primary_val, 4) if primary_val is not None else None,
                "secondary_value": round(secondary_val, 4) if secondary_val is not None else None,
                "primary_key": pk_key,
                "secondary_key": sk_key,
            }
        )

    edges = []
    for e in edge_rows:
        src = e["src_finding_id"]
        dst = e["dst_finding_id"]
        if src not in findings_by_id or dst not in findings_by_id:
            continue  # orphan edge — shouldn't happen but guard
        etype = e["edge_type"]
        conf = float(e["confidence"])
        base = _EDGE_COLOR.get(etype, "#999999")
        # Per-edge spring length — stronger / more direct progression
        # types get shorter rest length, so physics pulls the two ends
        # closer (which translates to smaller angular separation when
        # the two ends are also held at fixed ring radii by the leash
        # system below). Weak `related_to` edges use a long length so
        # they never dominate the layout.
        edge_length = _EDGE_SPRING_LENGTH.get(etype, 120)
        edges.append(
            {
                "id": e["edge_id"],
                "from": src,
                "to": dst,
                "edge_type": etype,
                "confidence": conf,
                "color": _hex_to_rgba(base, _edge_opacity(conf)),
                "width": _edge_width(conf),
                "rationale": e.get("rationale") or "",
                "created_by": e.get("created_by") or "",
                "created_at": e.get("created_at") or "",
                "length": edge_length,
            }
        )

    # --- Ring-preserving physics scaffold ---------------------------------
    # Three invisible anchor nodes, one per finding_type, pinned at the
    # origin. Each finding node gets a hidden "leash" edge to its
    # type's anchor with rest length = that type's ring radius. Under
    # physics, the leash pulls the node back toward the correct ring
    # radius regardless of how real edges try to drag it across rings
    # — but leaves the angular position free to be determined by the
    # real edge forces. Net: banding survives physics, and nodes with
    # strong cross-ring edges end up at similar ANGLES (drawing short
    # radial lines), which is how the user sees "this hypothesis led
    # directly to that result".
    anchor_nodes = []
    leash_edges = []
    for ftype_key in ("hypothesis", "insight", "result"):
        anchor_id = f"__anchor__{ftype_key}"
        anchor_nodes.append(
            {
                "id": anchor_id,
                "label": "",
                "x": 0,
                "y": 0,
                "fixed": {"x": True, "y": True},
                "hidden": True,
                "_is_anchor": True,
                "finding_type": ftype_key,
                # Give it zero size so even if something unhides it it's not distracting.
                "size": 0,
            }
        )
    # Stack 3 parallel leash edges per node. vis-network's physics
    # sums the spring force across all edges incident on a node, so N
    # identical parallel leashes = N× radial stiffness. One leash
    # alone is not enough to keep a node on its ring when that node
    # has 5+ real edges pulling it toward other rings — the real
    # edges win at equilibrium and banding dissolves. 3 leashes make
    # the radial constraint roughly as strong as 3 real edges,
    # enough to survive typical edge loads.
    _LEASH_STACK = 3
    for n in nodes:
        ftype_key = n.get("finding_type")
        if ftype_key not in ("hypothesis", "insight", "result"):
            continue
        for i in range(_LEASH_STACK):
            leash_edges.append(
                {
                    "id": f"__leash__{n['id']}__{i}",
                    "from": n["id"],
                    "to": f"__anchor__{ftype_key}",
                    "length": _TYPE_RADIUS[ftype_key],
                    "hidden": True,
                    "_is_leash": True,
                    "color": {"opacity": 0},
                }
            )

    # Import health without re-querying the full table.
    health = compute_graph_health()
    # Assemble the leaderboard (top-K ordered) for the sidebar.
    leaderboard = []
    for rank in sorted({r for r in top_k_ranks.values()}):
        match = [n for n in nodes if n.get("rank") == rank]
        if match:
            n = match[0]
            leaderboard.append(
                {
                    "rank": rank,
                    "id": n["id"],
                    "title": n["title_full"],
                    "peer_id": n["peer_id"],
                    "variant_name": n["variant_name"],
                    "primary_value": n["primary_value"],
                    "primary_key": n["primary_key"],
                    "secondary_value": n["secondary_value"],
                    "secondary_key": n["secondary_key"],
                    "is_pareto": n["is_pareto"],
                }
            )
    meta = {
        "generated_at": datetime.now(UTC).isoformat(),
        "num_findings": health["num_findings"],
        "num_edges": health["num_edges"],
        "linked_finding_ratio": health["linked_finding_ratio"],
        "edge_type_distribution": health["edge_type_distribution"],
        "low_confidence_edge_ratio": health["low_confidence_edge_ratio"],
        "top_k_count": len(top_k_ranks),
        "pareto_count": len(pareto_ids),
        "leaderboard": leaderboard,
        # Baseline reference values from task_spec.yaml — the client
        # renders these above the leaderboard so every measurement can
        # be compared at a glance. `primary_key` + `direction` let the
        # client label the axis correctly without inferring anything.
        "baselines": [
            {
                "name": b.get("name", ""),
                "metric_name": b.get("metric_name") or _task_primary_metric(task_spec_yaml),
                "value": b.get("metric_value", b.get("expected_acc")),
            }
            for b in ((task_spec_yaml or {}).get("baselines") or [])
            if isinstance(b, dict)
        ],
        "primary_key": _task_primary_metric(task_spec_yaml) or "primary_metric",
        "primary_direction": _task_direction(task_spec_yaml),
        "secondary_key": _task_secondary_metric(task_spec_yaml) or "",
        # Radial ring definitions so the client can draw persistent
        # background guides in canvas coordinates. Rings stay put even
        # when physics is on and rearranges nodes — users always have
        # a reference for "this node belongs on the result ring".
        "rings": [
            {
                "type": "hypothesis",
                "radius": _TYPE_RADIUS["hypothesis"],
                "color": _TYPE_COLOR["hypothesis"],
            },
            {"type": "insight", "radius": _TYPE_RADIUS["insight"], "color": _TYPE_COLOR["insight"]},
            {"type": "result", "radius": _TYPE_RADIUS["result"], "color": _TYPE_COLOR["result"]},
        ],
    }
    # Anchors + leash edges go in separate top-level arrays so the JS
    # can merge them into the vis DataSets without inflating any
    # user-visible counts or filter lists (which iterate `nodes` /
    # `edges`).
    return {
        "nodes": nodes,
        "edges": edges,
        "meta": meta,
        "anchors": anchor_nodes,
        "leash_edges": leash_edges,
    }


# --- HTML template ---------------------------------------------------------
# Self-contained: everything embedded except vis-network loaded from cdnjs.
# Inline all styles + the data payload as JSON.

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Finding Graph — __META_TITLE__</title>
__VIS_CSS_BLOCK__
__VIS_JS_BLOCK__
<style>
:root {
  --bg:#111; --fg:#eee; --dim:#888;
  --panel:#1a1a1a; --border:#333;
}
html, body { margin:0; padding:0; background:var(--bg); color:var(--fg);
  font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
#layout { display:flex; width:100vw; height:100vh; }
#graph  { flex:1; background:#0c0c0c; position:relative; }
#side   { width:320px; background:var(--panel); border-left:1px solid var(--border);
  padding:16px; overflow-y:auto; font-size:13px; }
h2 { font-size:15px; margin:0 0 8px 0; color:#fff; }
h3 { font-size:13px; margin:14px 0 6px 0; color:var(--dim);
  text-transform:uppercase; letter-spacing:0.05em; }
.row { display:flex; justify-content:space-between; font-size:12px;
  padding:2px 0; color:var(--fg); }
.row span.k { color:var(--dim); }
.legend-box { display:inline-block; width:12px; height:12px;
  margin-right:6px; vertical-align:middle; border-radius:2px; }
.edge-bar { display:inline-block; width:20px; height:3px; margin-right:6px;
  vertical-align:middle; }
.f-row { padding:3px 0; cursor:pointer; border-radius:3px;
  padding-left:4px; }
.f-row:hover { background:#222; }
.f-row input { margin-right:6px; }
input[type=range] { width: 100%; }
input[type=text] { width:100%; padding:6px 8px; background:#0c0c0c;
  color:var(--fg); border:1px solid var(--border); border-radius:3px;
  box-sizing:border-box; }
#detail { font-size:12px; line-height:1.5; max-height:40vh; overflow:auto; }
#detail b { color:#fff; }
#detail .muted { color:var(--dim); }
#detail .metrics { margin-top:6px; font-family: ui-monospace, monospace; font-size:11px; }
.tip { background:#222; color:#fff; padding:8px 10px; border-radius:4px;
  font-size:12px; max-width:360px; line-height:1.45;
  border:1px solid #444; }
.tip b { color:#fff; }
.tip .muted { color:#aaa; }
/* vis-network wraps string titles in .vis-tooltip; style it to match */
div.vis-tooltip {
  background:#222 !important; color:#fff !important;
  padding:8px 10px !important; border-radius:4px !important;
  font-size:12px !important; max-width:360px !important; line-height:1.45 !important;
  border:1px solid #444 !important; font-family: inherit !important;
}
div.vis-tooltip b { color:#fff; }
div.vis-tooltip .muted { color:#aaa; }
button { background:#222; color:var(--fg); border:1px solid var(--border);
  padding:4px 10px; margin-right:4px; border-radius:3px; cursor:pointer;
  font-size:12px; }
button:hover { background:#333; }
button.active { background:#007acc; border-color:#007acc; color:#fff; }

/* Leaderboard + pareto legend */
.leader-row { display:flex; align-items:center; gap:6px; padding:4px 0;
  border-bottom: 1px dashed #2a2a2a; font-size:12px; }
.leader-row:last-child { border-bottom:none; }
.medal { font-size:14px; width:20px; text-align:center; flex:0 0 auto; }
.leader-body { flex:1; min-width:0; }
.leader-title { white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  color:#e0e0e0; }
.leader-meta { color:var(--dim); font-size:11px; }
.pareto-tag { display:inline-block; padding:0 6px; border-radius:8px;
  background:rgba(255,255,255,0.1); color:#fff; font-size:10px;
  border:1px solid rgba(255,255,255,0.3); margin-left:4px; }
.perf-banner { background:linear-gradient(90deg,rgba(255,215,0,0.08),transparent);
  border-left:2px solid #ffd700; padding:4px 8px; margin:4px 0;
  font-size:11px; color:#f0c040; }
</style>
</head>
<body>
<div id="vis-missing-banner" style="display:none; position:fixed; top:0; left:0; right:0;
  background:#D55E00; color:#fff; padding:10px 16px; font-size:13px;
  z-index:9999; text-align:center;">
  <b>vis-network failed to load.</b>
  The graph canvas can't render without it. Open your browser devtools
  for the exact error, or re-render this file on a machine with network
  access so it can be re-cached inline.
</div>
<div id="layout">
  <div id="graph"></div>
  <div id="side">
    <h2>Finding Graph</h2>
    <div style="font-size:11px; color:#888; margin:4px 0 10px 0; line-height:1.45">
      Edges are <b>advisory navigation</b>, not conclusions. Open the
      raw finding before treating any edge as evidence.
    </div>
    <div class="row"><span class="k">generated</span><span id="gen-ts"></span></div>
    <div class="row"><span class="k">findings</span><span id="m-n"></span></div>
    <div class="row"><span class="k">edges</span><span id="m-e"></span></div>
    <div class="row"><span class="k">linked ratio</span><span id="m-lr"></span></div>
    <div class="row"><span class="k">low-conf ratio</span><span id="m-lc"></span></div>

    <h3>🏆 Leaderboard</h3>
    <div id="leaderboard" style="margin-bottom: 4px;"></div>
    <div class="muted" style="font-size:10px; margin-bottom:12px; line-height:1.4">
      🥇/🥈/🥉 = top by primary metric · ⚪ = Pareto frontier
    </div>

    <h3>Node types</h3>
    <div id="type-legend"></div>
    <div class="muted" style="font-size:10px; margin-top:4px;">
      uncheck to hide all nodes of that type
    </div>

    <h3>Edge types</h3>
    <div id="edge-legend"></div>

    <h3>Min confidence: <span id="conf-val"></span></h3>
    <input type="range" id="conf-slider" min="0.55" max="1.00" step="0.01" value="0.55"/>

    <h3>Search</h3>
    <input type="text" id="search" placeholder="title, variant, peer_id, or id…"/>

    <h3>Actions</h3>
    <button id="btn-fit">Fit</button>
    <button id="btn-toggle-physics">Run layout</button>
    <div class="muted" style="font-size:11px; margin-top:8px; line-height:1.5">
      pan: right-click or middle-click drag<br/>
      zoom: mouse wheel<br/>
      move node: left-click drag
    </div>

    <h3>Selection</h3>
    <div id="detail" class="muted">Click a node or edge.</div>
  </div>
</div>

<!-- Loading overlay shown until the graph initialization finishes.
     Even with physics off and diff-only filters, parsing the vis-network
     bundle + building the 3412-edge DataSet + first canvas paint takes a
     few seconds on a large graph — so we show something instead of the
     browser's generic tab spinner. -->
<div id="loading-overlay" style="position:fixed; inset:0; background:rgba(0,0,0,0.65);
  color:#fff; display:flex; align-items:center; justify-content:center;
  font-size:16px; z-index:8888; pointer-events:none;">
  <div style="text-align:center">
    <div style="margin-bottom:8px">Building graph…</div>
    <div id="loading-status" style="color:#aaa; font-size:13px">preparing</div>
  </div>
</div>

<script>
const PAYLOAD = __PAYLOAD_JSON__;

if (typeof vis === "undefined" || !vis.Network) {
  document.getElementById("vis-missing-banner").style.display = "block";
  console.error("vis-network did not load. Check network/CSP and reload.");
}

function setLoading(msg) {
  const el = document.getElementById("loading-status");
  if (el) el.textContent = msg;
}
function hideLoading() {
  const el = document.getElementById("loading-overlay");
  if (el) el.style.display = "none";
}

// Everything below runs inside a deferred block so the UI shell (loading
// overlay, empty sidebar) paints first. Without this the browser bundles
// the overlay paint with the heavy init work and the user sees nothing
// until the whole pipeline completes.
function startGraphInit() {

setLoading("populating sidebar");

// --- setup ----------------------------------------------------------------
document.getElementById("gen-ts").textContent =
  PAYLOAD.meta.generated_at.replace("T", " ").slice(0, 19) + " UTC";
document.getElementById("m-n").textContent = PAYLOAD.meta.num_findings;
document.getElementById("m-e").textContent = PAYLOAD.meta.num_edges;
document.getElementById("m-lr").textContent =
  (PAYLOAD.meta.linked_finding_ratio * 100).toFixed(1) + "%";
document.getElementById("m-lc").textContent =
  (PAYLOAD.meta.low_confidence_edge_ratio * 100).toFixed(1) + "%";

const TYPE_COLORS = {
  result:     "#009E73", insight:    "#0072B2",
  hypothesis: "#E69F00", error:      "#D55E00", unknown: "#888888",
};
const EDGE_COLORS = {
  supports:     "#009E73", challenges:  "#D55E00",
  updates:      "#CC79A7", derived_from:"#56B4E9",
  related_to:   "#BBBBBB",
};

// Leaderboard — top-K findings ranked by primary metric. Pareto
// frontier findings get a ⚪ marker alongside (can be both top-K and
// Pareto). Clicking a leaderboard row selects that node in the
// canvas.
(function() {
  // Format a metric value using its key name as a unit hint. We do NOT
  // assume any particular task's metric is a fraction in [0,1] — accuracy
  // and cost-like scores live in totally different ranges. If the
  // key looks like a percent (*_pct, *_ratio) we just append "%" to the
  // raw number; else we show a plain decimal. No *100 multiplication.
  function fmtMetric(key, val, decimals) {
    if (val === null || val === undefined) return "—";
    const d = (decimals === undefined) ? 3 : decimals;
    const kl = (key || "").toLowerCase();
    const isPct = kl.endsWith("_pct") || kl.indexOf("_pct_") !== -1
               || kl.endsWith("_percent") || kl.endsWith("_percentage");
    return Number(val).toFixed(d) + (isPct ? "%" : "");
  }

  const box = document.getElementById("leaderboard");
  const lb = PAYLOAD.meta.leaderboard || [];
  const baselines = PAYLOAD.meta.baselines || [];
  const primaryKey = PAYLOAD.meta.primary_key || "primary";
  const primaryDir = PAYLOAD.meta.primary_direction || "maximize";

  // Render baselines first (always shown, even when leaderboard is empty)
  // so the user has a reference bar to compare measurements against.
  if (baselines.length > 0) {
    const baseDiv = document.createElement("div");
    baseDiv.style.cssText = 'margin-bottom:8px;padding:6px 8px;border:1px dashed var(--border);border-radius:4px;font-size:11px;background:rgba(150,150,150,0.06);';
    const header = '<div style="color:var(--dim);margin-bottom:3px;"><strong>baselines</strong> · ' + escapeHTML(primaryKey) + ' (' + primaryDir + ')</div>';
    const rows = baselines.map(b => {
      const valStr = (b.value === null || b.value === undefined)
        ? "—" : fmtMetric(primaryKey, b.value, 3);
      return '<div>' + escapeHTML(b.name) + ' = ' + valStr + '</div>';
    }).join("");
    baseDiv.innerHTML = header + rows;
    box.appendChild(baseDiv);
  }
  if (lb.length === 0) {
    const stub = document.createElement("div");
    stub.className = "muted";
    stub.style.fontSize = "11px";
    stub.textContent = 'no scored findings yet';
    box.appendChild(stub);
    return;
  }
  const medals = {1: "🥇", 2: "🥈", 3: "🥉"};
  for (const entry of lb) {
    const row = document.createElement("div");
    row.className = "leader-row";
    row.style.cursor = "pointer";
    row.title = `click to focus on this node\n${entry.id}`;
    const medal = medals[entry.rank] || `#${entry.rank}`;
    const variant = entry.variant_name ? escapeHTML(entry.variant_name) : "(unnamed)";
    const epk = escapeHTML(entry.primary_key || "primary");
    const esk = escapeHTML(entry.secondary_key || "");
    const primary_fmt = epk + "=" + fmtMetric(entry.primary_key, entry.primary_value, 3);
    const secondary_fmt = entry.secondary_value !== null && entry.secondary_key ?
      esk + "=" + fmtMetric(entry.secondary_key, entry.secondary_value, 2) : "";
    const pareto_tag = entry.is_pareto ? '<span class="pareto-tag">⚪ pareto</span>' : '';
    row.innerHTML = `
      <span class="medal">${medal}</span>
      <div class="leader-body">
        <div class="leader-title">${primary_fmt} · ${variant}${pareto_tag}</div>
        <div class="leader-meta">${escapeHTML(entry.peer_id || '?')}${secondary_fmt ? ' · ' + secondary_fmt : ''}</div>
      </div>`;
    row.addEventListener("click", () => {
      network.selectNodes([entry.id]);
      network.focus(entry.id, { scale: 1.6, animation: { duration: 600 } });
    });
    box.appendChild(row);
  }
})();

// Build node-type filter checkboxes (legend + toggle combined). Each
// type is shown in its ring color; unchecking hides all nodes of
// that type AND every edge touching a hidden node. Together with the
// radial banding by type in the initial layout, this lets you reduce
// visual noise — e.g. hide 'hypothesis' to see only observations and
// confirmed results, or hide 'result' to focus on the proposal space.
const activeNodeTypes = new Set();
(function() {
  const counts = {};
  for (const n of PAYLOAD.nodes)
    counts[n.finding_type] = (counts[n.finding_type] || 0) + 1;
  const box = document.getElementById("type-legend");
  Object.entries(counts).sort((a,b) => b[1]-a[1]).forEach(([t,c]) => {
    activeNodeTypes.add(t);
    const row = document.createElement("div");
    row.className = "f-row";
    row.innerHTML = `<label>
      <input type="checkbox" checked data-type="${escapeHTML(t)}"/>
      <span class="legend-box" style="background:${TYPE_COLORS[t] || "#888"}"></span>
      ${escapeHTML(t)} (${c})
    </label>`;
    box.appendChild(row);
  });
  box.querySelectorAll("input[type=checkbox]").forEach(cb => {
    cb.addEventListener("change", e => {
      const t = e.target.dataset.type;
      if (e.target.checked) activeNodeTypes.add(t);
      else activeNodeTypes.delete(t);
      applyFilters();
    });
  });
})();

// Edge-type legend + filter checkboxes.
const activeEdgeTypes = new Set(Object.keys(PAYLOAD.meta.edge_type_distribution));
(function() {
  const box = document.getElementById("edge-legend");
  for (const [t, c] of Object.entries(PAYLOAD.meta.edge_type_distribution)) {
    if (c === 0) continue;
    const row = document.createElement("div");
    row.className = "f-row";
    row.innerHTML = `<label>
      <input type="checkbox" checked data-type="${escapeHTML(t)}"/>
      <span class="edge-bar" style="background:${EDGE_COLORS[t] || "#888"}"></span>
      ${escapeHTML(t)} (${c})
    </label>`;
    box.appendChild(row);
  }
  box.querySelectorAll("input[type=checkbox]").forEach(cb => {
    cb.addEventListener("change", e => {
      const t = e.target.dataset.type;
      if (e.target.checked) activeEdgeTypes.add(t);
      else activeEdgeTypes.delete(t);
      applyFilters();
      // NEW: when the user changes the edge-type selection, don't just
      // hide/show — also tell the physics engine which edges should
      // exert force right now. Unchecked types stop pulling nodes;
      // checked types resume pulling. The sim restarts so the graph
      // re-equilibrates under the new constraint set.
      applyEdgePhysicsGate();
      kickPhysicsRestart();
    });
  });
})();

// --- physics gating by edge_type ---------------------------------------
// For each real edge, set `physics: true` iff its edge_type is currently
// checked in the sidebar. Leash edges + unrelated node springs are
// untouched so the radial banding scaffolding keeps working. Uses a
// diff-only batched update so a type toggle is a single change event
// to the DataSet (and therefore one redraw schedule).
const _edgePhysicsState = new Map(PAYLOAD.edges.map(e => [e.id, true]));
function applyEdgePhysicsGate() {
  const updates = [];
  for (const e of PAYLOAD.edges) {
    const active = activeEdgeTypes.has(e.edge_type);
    if (_edgePhysicsState.get(e.id) !== active) {
      updates.push({ id: e.id, physics: active });
      _edgePhysicsState.set(e.id, active);
    }
  }
  if (updates.length) edgesDS.update(updates);
}

// Enable physics with the standard ForceAtlas2-based config and make the
// Run-layout button reflect the new active state. Called whenever the
// edge-type selection changes so the user visibly sees the simulation
// restart.
function kickPhysicsRestart() {
  network.setOptions({ physics: {
    enabled: true,
    solver: "forceAtlas2Based",
    forceAtlas2Based: {
      gravitationalConstant: -35,
      centralGravity: 0,
      springLength: 100,
      springConstant: 0.04,
    },
    adaptiveTimestep: true,
    stabilization: false,
    maxVelocity: 50,
    timestep: 0.35,
  }});
  physicsOn = true;
  const btn = document.getElementById("btn-toggle-physics");
  if (btn) {
    btn.textContent = "Pause layout";
    btn.classList.add("active");
  }
}

let minConf = 0.55;
document.getElementById("conf-val").textContent = minConf.toFixed(2);
document.getElementById("conf-slider").addEventListener("input", e => {
  minConf = parseFloat(e.target.value);
  document.getElementById("conf-val").textContent = minConf.toFixed(2);
  applyFilters();
});

// --- build DataSets --------------------------------------------------------
// String tooltips (not DOM elements) — vis-network renders them via
// innerHTML at hover time, so we avoid creating 4000+ DOM nodes upfront.
// EVERY interpolation into innerHTML MUST go through escapeHTML —
// finding_type is enum-validated Python-side but peer_id, variant_name,
// generation_id, created_by, content_snippet, title_full, rationale are
// all agent-controlled. A finding with peer_id='<img src=x onerror=...>'
// renders JavaScript at hover-time without this gate. metrics_html is
// pre-escaped in Python (html.escape on both keys and values) so we
// insert it raw — if that contract ever changes, escape here too.
function perfBanner(n) {
  // One-liner banner showing rank/pareto membership + the resolved
  // primary (+ optional secondary) metric this node was scored on.
  // Returns "" when the node has no perf tag so normal findings stay
  // uncluttered.
  const parts = [];
  if (n.rank) {
    const medals = {1: "🥇 Rank 1", 2: "🥈 Rank 2", 3: "🥉 Rank 3"};
    parts.push(`<b style="color:#ffd700">${medals[n.rank] || ("#" + n.rank)}</b>`);
  }
  if (n.is_pareto) parts.push('<b style="color:#fff">⚪ Pareto frontier</b>');
  if (parts.length === 0) return "";
  let metrics = "";
  if (n.primary_value !== null) {
    metrics = `<span style="color:#aaa"> · ${escapeHTML(n.primary_key || 'primary')}=` +
              `${(n.primary_value*100).toFixed(2)}%`;
    if (n.secondary_value !== null) {
      metrics += ` · ${escapeHTML(n.secondary_key || 'gap')}=${(n.secondary_value*100).toFixed(2)}%`;
    }
    metrics += `</span>`;
  }
  return `<div class="perf-banner">${parts.join(' · ')}${metrics}</div>`;
}

function buildNodeTip(n) {
  const m = n.metrics_html ? `<div class="muted">${n.metrics_html}</div>` : "";
  return `${perfBanner(n)}<div><b>${escapeHTML(n.title_full)}</b></div>
    <div class="muted">${escapeHTML(n.finding_type)} · ${escapeHTML(n.peer_id)} · gen ${escapeHTML(n.generation_id)} · degree ${escapeHTML(n.degree)}</div>
    ${m}
    <div style="margin-top:6px">${escapeHTML(n.content_snippet)}</div>`;
}
function buildEdgeTip(e) {
  return `<div><b>${escapeHTML(e.edge_type)}</b> · conf ${e.confidence.toFixed(2)} · ${escapeHTML(e.created_by)}</div>
    <div class="muted" style="margin-top:4px">${escapeHTML(e.rationale)}</div>`;
}

setLoading("building node dataset (" + PAYLOAD.nodes.length + ")");
// Medal borders for top-3: gold / silver / bronze. Pareto frontier
// (for findings with both metrics) gets a bright white ring. The
// visual bleeds up to 8 GPU-bright pixels around top nodes so they
// pop out of the cluster without changing their position.
const MEDAL_BORDER = {1: "#ffd700", 2: "#c0c0c0", 3: "#cd7f32"};
const nodesDS = new vis.DataSet(PAYLOAD.nodes.map(n => {
  let borderColor = "#222";
  let borderWidth = 1;
  let shadow = undefined;
  let size = n.size;
  if (n.rank && MEDAL_BORDER[n.rank]) {
    borderColor = MEDAL_BORDER[n.rank];
    borderWidth = 5;
    shadow = { enabled: true, color: borderColor, size: 22, x: 0, y: 0 };
    // Scale top-K nodes up a bit so the badge is readable even if the
    // finding has low degree.
    size = Math.max(n.size, 24);
  } else if (n.is_pareto) {
    borderColor = "#ffffff";
    borderWidth = 3;
    shadow = { enabled: true, color: "rgba(255,255,255,0.55)", size: 10, x: 0, y: 0 };
    size = Math.max(n.size, 18);
  }
  return {
    id: n.id, label: n.label,
    color: { background: n.color, border: borderColor,
             highlight: { background: n.color, border: "#fff" } },
    size: size, borderWidth: borderWidth, shadow: shadow,
    font: { color: "#ddd", size: 10 },
    title: buildNodeTip(n),
    x: n.x, y: n.y, _raw: n,
  };
}));
setLoading("building edge dataset (" + PAYLOAD.edges.length + ")");
const edgesDS = new vis.DataSet(PAYLOAD.edges.map(e => ({
  id: e.id,
  from: e.from, to: e.to,
  color: { color: e.color, highlight: "#fff" },
  width: e.width,
  arrows: { to: { enabled: true, scaleFactor: 0.5 } },
  title: buildEdgeTip(e),
  smooth: false,
  length: e.length,  // per-edge spring length for the physics engine
  _raw: e,
})));

// Merge in the ring-preserving physics scaffold — 3 fixed anchors at
// origin + one hidden leash edge per real finding. Hidden means
// vis-network doesn't paint them but they STILL participate in physics.
// Users never see anchors / leashes; the filter/search code already
// skips them because they're not in PAYLOAD.nodes / PAYLOAD.edges.
if (PAYLOAD.anchors && PAYLOAD.anchors.length) {
  nodesDS.add(PAYLOAD.anchors.map(a => ({
    id: a.id, label: "", hidden: true, size: 0, fixed: a.fixed,
    x: a.x, y: a.y, physics: true,
  })));
}
if (PAYLOAD.leash_edges && PAYLOAD.leash_edges.length) {
  edgesDS.add(PAYLOAD.leash_edges.map(le => ({
    id: le.id, from: le.from, to: le.to,
    length: le.length, hidden: true, physics: true,
    // Even if a future vis-network bug un-hides these, fully
    // transparent + no arrow stops them from rendering visibly.
    color: { color: "rgba(0,0,0,0)", opacity: 0 },
    arrows: { to: { enabled: false } },
  })));
}
function escapeHTML(s) {
  // `s || ""` would coerce the number 0 to "" — guard explicitly so
  // degree=0 and generation=0 show as "0" instead of disappearing.
  const v = (s === null || s === undefined) ? "" : String(s);
  return v
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

// --- network ---------------------------------------------------------------
setLoading("initializing network");
// Physics OFF by default — 665 nodes × 3k+ edges running forceAtlas2
// stabilization freezes the browser main thread for tens of seconds.
// Nodes come with deterministic x/y baked into the payload; users can
// opt into live physics via the "Run layout" button.
const network = new vis.Network(
  document.getElementById("graph"),
  { nodes: nodesDS, edges: edgesDS },
  {
    physics: { enabled: false },
    layout: { improvedLayout: false, randomSeed: 1 },
    interaction: {
      hover: true, tooltipDelay: 120,
      dragNodes: true,
      dragView: true,   // left-click-drag on empty space pans
      zoomView: true,   // mouse-wheel zoom
    },
    nodes: { shape: "dot", borderWidth: 1 },
    edges: { smooth: false },
  }
);

// Background ring guides — one per finding_type. Drawn in graph
// coordinates so they zoom + pan with the content but stay fixed
// relative to the world origin. Physics can shuffle nodes freely;
// the rings never move, giving the viewer a persistent reference
// for "which ring does this node logically belong to". Without this
// overlay the radial banding is only visible until the user clicks
// "Run layout" and forceAtlas2 pulls nodes along edges, flattening
// the initial structure.
const RINGS = PAYLOAD.meta.rings || [];
network.on("beforeDrawing", function(ctx) {
  ctx.save();
  for (const ring of RINGS) {
    // Soft filled ring: draw an annulus with even-odd winding so the
    // hole in the middle shows through.
    ctx.beginPath();
    ctx.arc(0, 0, ring.radius, 0, 2 * Math.PI, false);
    ctx.strokeStyle = hexToRGBA(ring.color, 0.35);
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 6]);
    ctx.stroke();
    ctx.setLineDash([]);
    // Label at the top of the ring (12 o'clock).
    ctx.font = "500 13px -apple-system, 'Segoe UI', sans-serif";
    ctx.fillStyle = hexToRGBA(ring.color, 0.85);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(ring.type, 0, -ring.radius - 14);
  }
  ctx.restore();
});
function hexToRGBA(hex, a) {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0,2),16), g=parseInt(h.slice(2,4),16), b=parseInt(h.slice(4,6),16);
  return `rgba(${r},${g},${b},${a})`;
}

// Right-click-drag and middle-click-drag pan, always available regardless
// of what's under the cursor. vis-network's built-in dragView only fires
// on left-click-on-empty-space, which is hard to hit on a dense graph
// with 665 nodes spread across the canvas.
(function attachPan() {
  const container = document.getElementById("graph");
  let panning = false;
  let lastX = 0, lastY = 0;
  // Suppress the browser context menu so right-click-drag feels natural.
  container.addEventListener("contextmenu", e => e.preventDefault());
  container.addEventListener("mousedown", e => {
    if (e.button === 2 || e.button === 1) {  // right or middle
      panning = true;
      lastX = e.clientX;
      lastY = e.clientY;
      container.style.cursor = "grabbing";
      e.preventDefault();
    }
  });
  window.addEventListener("mousemove", e => {
    if (!panning) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    const pos = network.getViewPosition();
    const scale = network.getScale() || 1;
    network.moveTo({
      position: { x: pos.x - dx / scale, y: pos.y - dy / scale },
      animation: false,
    });
  });
  const endPan = e => {
    if (!panning) return;
    if (e && e.button !== undefined && e.button !== 2 && e.button !== 1) return;
    panning = false;
    container.style.cursor = "";
  };
  window.addEventListener("mouseup", endPan);
  window.addEventListener("mouseleave", endPan);
})();
// Fit the view to all nodes on first paint — coords in the payload put
// clusters at radius ~600-800 so the default viewport would miss them.
// Hide the loading overlay on the same event so the user sees the real
// graph the moment it's painted.
network.once("afterDrawing", () => {
  network.fit({ animation: false });
  hideLoading();
});

// --- filters ---------------------------------------------------------------
// Track current hidden state locally — cheaper than reading it from the
// DataSet and lets us only send diff updates. vis.DataSet.update([...])
// fires one change event for the whole batch instead of one per row,
// which is the difference between "snappy" and "tab spinner for 30s"
// on 3412 edges.
const nodeHidden = new Map(PAYLOAD.nodes.map(n => [n.id, false]));
const edgeHidden = new Map(PAYLOAD.edges.map(e => [e.id, false]));

function applyFilters() {
  const q = (document.getElementById("search").value || "").toLowerCase().trim();
  const nodeVisible = new Set();
  for (const n of PAYLOAD.nodes) {
    // Type filter — nodes whose finding_type was unchecked don't
    // qualify regardless of search match.
    if (!activeNodeTypes.has(n.finding_type)) continue;
    if (!q) { nodeVisible.add(n.id); continue; }
    const hay = [n.title_full, n.peer_id, n.variant_name, n.id].join(" ").toLowerCase();
    if (hay.includes(q)) nodeVisible.add(n.id);
  }
  const edgeVisible = new Set();
  const kept = [];
  for (const e of PAYLOAD.edges) {
    const pass = activeEdgeTypes.has(e.edge_type)
      && e.confidence >= minConf
      && nodeVisible.has(e.from) && nodeVisible.has(e.to);
    if (pass) { edgeVisible.add(e.id); kept.push(e); }
  }
  if (q) {
    const touched = new Set();
    for (const e of kept) { touched.add(e.from); touched.add(e.to); }
    nodeVisible.clear();
    touched.forEach(id => nodeVisible.add(id));
  }

  // Diff-only batched update — emit two change events total.
  const nodeUpdates = [];
  for (const n of PAYLOAD.nodes) {
    const shouldHide = !nodeVisible.has(n.id);
    if (nodeHidden.get(n.id) !== shouldHide) {
      nodeUpdates.push({ id: n.id, hidden: shouldHide });
      nodeHidden.set(n.id, shouldHide);
    }
  }
  if (nodeUpdates.length) nodesDS.update(nodeUpdates);

  const edgeUpdates = [];
  for (const e of PAYLOAD.edges) {
    const shouldHide = !edgeVisible.has(e.id);
    if (edgeHidden.get(e.id) !== shouldHide) {
      edgeUpdates.push({ id: e.id, hidden: shouldHide });
      edgeHidden.set(e.id, shouldHide);
    }
  }
  if (edgeUpdates.length) edgesDS.update(edgeUpdates);
}
document.getElementById("search").addEventListener("input", applyFilters);
document.getElementById("btn-fit").addEventListener("click", () => network.fit());
// Physics starts OFF; clicking the button turns it on (with a light
// forceAtlas2 config) and clicking again pauses it.
let physicsOn = false;
document.getElementById("btn-toggle-physics").addEventListener("click", e => {
  physicsOn = !physicsOn;
  if (physicsOn) {
    network.setOptions({ physics: {
      enabled: true,
      solver: "forceAtlas2Based",
      // Global springLength is the FALLBACK for edges without a
      // `length` property. Our real edges all carry per-type lengths
      // (45-150) and leash edges carry the ring radius (290/551/866),
      // so this global value only matters for legacy / unlabeled edges.
      // centralGravity=0 so the only center-ward force is via leash
      // edges — otherwise result-ring nodes would get pulled inward
      // against the leash.
      forceAtlas2Based: {
        gravitationalConstant: -35,
        centralGravity: 0,
        springLength: 100,
        springConstant: 0.04,
      },
      // Short wall between leash pulling and edges settling: the
      // first few ticks should be gentle so rings stabilize before
      // real edges crank up cross-ring pull.
      adaptiveTimestep: true,
      stabilization: false,
      maxVelocity: 50,
      timestep: 0.35,
    }});
    e.target.textContent = "Pause layout";
    e.target.classList.add("active");
  } else {
    network.setOptions({ physics: { enabled: false } });
    e.target.textContent = "Run layout";
    e.target.classList.remove("active");
  }
});

// --- selection detail ------------------------------------------------------
const detail = document.getElementById("detail");
network.on("selectNode", ev => {
  const id = ev.nodes[0];
  const n = PAYLOAD.nodes.find(x => x.id === id);
  if (!n) return;
  detail.classList.remove("muted");
  detail.innerHTML = `
    ${perfBanner(n)}
    <div><b>${escapeHTML(n.title_full)}</b></div>
    <div class="muted">type=${escapeHTML(n.finding_type)} · peer=${escapeHTML(n.peer_id)} · gen=${escapeHTML(n.generation_id)} · degree=${escapeHTML(n.degree)}</div>
    <div class="muted">${escapeHTML(n.id)}</div>
    ${n.variant_name ? `<div>variant: <code>${escapeHTML(n.variant_name)}</code></div>` : ""}
    ${n.metrics_html ? `<div class="metrics">${n.metrics_html}</div>` : ""}
    <div style="margin-top:8px">${escapeHTML(n.content_snippet)}</div>`;
});
network.on("selectEdge", ev => {
  const id = ev.edges[0];
  const e = PAYLOAD.edges.find(x => x.id === id);
  if (!e) return;
  const src = PAYLOAD.nodes.find(x => x.id === e.from);
  const dst = PAYLOAD.nodes.find(x => x.id === e.to);
  detail.classList.remove("muted");
  detail.innerHTML = `
    <div><b>${escapeHTML(e.edge_type)}</b> · confidence ${e.confidence.toFixed(2)} · ${escapeHTML(e.created_by)}</div>
    <div class="muted" style="margin-top:4px">${escapeHTML(e.rationale)}</div>
    <div style="margin-top:10px"><b>from</b>: ${escapeHTML(src ? src.title_full : e.from)}</div>
    <div><b>to</b>: ${escapeHTML(dst ? dst.title_full : e.to)}</div>`;
});
network.on("deselectNode", () => detail.classList.add("muted"));
network.on("deselectEdge", () => detail.classList.add("muted"));

// No stabilization (physics starts off) — apply filters on the next tick
// so the DataSet subscribers see a populated network before we start
// hiding nodes.
setTimeout(applyFilters, 0);

}  // end startGraphInit

// Two rAF() nests guarantee the sidebar + loading overlay paint once
// before we start the heavy init. Without this the browser bundles the
// overlay paint with the init work and the user sees nothing until the
// whole thing is done.
requestAnimationFrame(() => requestAnimationFrame(startGraphInit));
</script>
</body>
</html>
"""


def render_graph_html(
    out_path: Path,
    payload: dict[str, Any] | None = None,
) -> Path:
    """Write a self-contained HTML graph visualization to ``out_path``.

    The file is written atomically (tmp + rename) so a concurrent open in a
    browser always sees a complete document.
    """

    if payload is None:
        payload = build_viz_payload()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the meta title string for the <title> tag.
    meta = payload["meta"]
    meta_title = (
        f"{meta['num_findings']} findings · {meta['num_edges']} edges · "
        f"{meta['linked_finding_ratio'] * 100:.0f}% linked"
    )

    # Inline vis-network so the HTML works fully offline after download.
    # If the fetch ever fails (no network at render time), fall back to
    # the remote CDN <script> / <link> tags so the file at least works
    # online. `__VIS_JS_BLOCK__` / `__VIS_CSS_BLOCK__` are placeholders
    # in the template rather than literal tags so either path keeps the
    # document well-formed.
    assets = _load_vis_assets()
    if assets["js"]:
        js_block = "<script>\n" + assets["js"] + "\n</script>"
    else:
        js_block = f'<script src="{_VIS_NETWORK_JS_URL}"></script>'
    if assets["css"]:
        css_block = "<style>\n" + assets["css"] + "\n</style>"
    else:
        css_block = f'<link rel="stylesheet" href="{_VIS_NETWORK_CSS_URL}"/>'

    # Inject payload + meta title into the template. Keep JSON compact to
    # minimize file size (this file is written every maintainer cycle).
    # vis-network's JS contains `</script>` in its string literals; guard
    # payload JSON the same way by escaping `</` before injection.
    payload_json = json.dumps(payload, separators=(",", ":"), default=str)
    payload_json = payload_json.replace("</", "<\\/")
    html_doc = (
        _HTML_TEMPLATE.replace("__VIS_JS_BLOCK__", js_block)
        .replace("__VIS_CSS_BLOCK__", css_block)
        .replace("__PAYLOAD_JSON__", payload_json)
        .replace("__META_TITLE__", html.escape(meta_title))
    )

    # Atomic write: tmp + rename (can't use atomic_write_json since content
    # isn't JSON). Implement the same pattern locally.
    import os
    import tempfile

    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{out_path.name}.",
        suffix=".tmp",
        dir=str(out_path.parent),
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(html_doc)
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, out_path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise

    logger.info(
        "wrote graph visualization: %s (%d nodes, %d edges, %.0f KB)",
        out_path,
        len(payload["nodes"]),
        len(payload["edges"]),
        out_path.stat().st_size / 1024,
    )
    return out_path
