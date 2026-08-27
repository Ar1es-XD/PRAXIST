"""
Local shared store — SQLite-based IPC for single-server deployment.

Replaces the HTTP server dependency with a single WAL-mode SQLite database.
All peers (asyncio tasks) on the same machine read/write this DB directly.
Supports: metrics logging, findings sharing, leaderboard queries.
"""

import copy
import json
import logging
import math
import os
import re
import sqlite3
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
    _RESULT_PRODUCER_IDENTITY_KEYS,
)

logger = logging.getLogger(__name__)

_DB_LOCK = threading.Lock()

# Default DB path — overridable via env var
_DEFAULT_DB_DIR = os.path.join(tempfile.gettempdir(), "praxist", "local_store")


def _get_db_path() -> str:
    db_dir = (
        os.environ.get("LOCAL_STORE_DIR")
        or os.environ.get("PRAXIST_RUN_DIR")
        or os.environ.get("AUTO_RESEARCH_RUN_DIR")
        or _DEFAULT_DB_DIR
    )
    Path(db_dir).mkdir(parents=True, exist_ok=True)
    return os.path.join(db_dir, "shared_store.db")


@contextmanager
def _get_conn(readonly: bool = False):
    """Get a SQLite connection with WAL mode and appropriate settings."""
    db_path = _get_db_path()
    uri = f"file:{db_path}?mode=ro" if readonly else f"file:{db_path}"
    conn = sqlite3.connect(
        uri if readonly else db_path,
        uri=readonly,
        timeout=30,
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        if not readonly:
            conn.commit()
    except Exception:
        if not readonly:
            conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize the shared store schema. Idempotent."""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS findings (
                id          TEXT PRIMARY KEY,
                finding_type TEXT NOT NULL,
                title       TEXT NOT NULL,
                content     TEXT NOT NULL DEFAULT '',
                metrics     TEXT NOT NULL DEFAULT '{}',
                variant_name TEXT NOT NULL DEFAULT '',
                notes       TEXT NOT NULL DEFAULT '',
                peer_id      TEXT NOT NULL DEFAULT '',
                generation_id INTEGER NOT NULL DEFAULT 0,
                timestamp   TEXT NOT NULL,
                extra       TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_findings_gen
                ON findings(generation_id);
            CREATE INDEX IF NOT EXISTS idx_findings_type
                ON findings(finding_type);
            -- Session-start graph context queries `WHERE peer_id = ?`
            -- once per peer per render. Without this index the filter
            -- is a full-table scan; at 10k findings × N peers that's
            -- serializing every cohort launch on O(N) scans.
            CREATE INDEX IF NOT EXISTS idx_findings_peer
                ON findings(peer_id);

            CREATE TABLE IF NOT EXISTS metrics (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      TEXT NOT NULL,
                variant_name TEXT NOT NULL DEFAULT '',
                metrics     TEXT NOT NULL DEFAULT '{}',
                notes       TEXT NOT NULL DEFAULT '',
                step        INTEGER NOT NULL DEFAULT 0,
                peer_id      TEXT NOT NULL DEFAULT '',
                generation_id INTEGER NOT NULL DEFAULT 0,
                timestamp   TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_metrics_gen
                ON metrics(generation_id);
            CREATE INDEX IF NOT EXISTS idx_metrics_run
                ON metrics(run_id);

            -- finding_edges: sidecar graph index over findings.
            -- See docs/concepts/architecture.md.
            --
            -- Edges are advisory navigation, not conclusions. The findings
            -- table remains the source of truth; closing this feature
            -- (stopping the maintainer, dropping edges) leaves core behavior
            -- unchanged.
            CREATE TABLE IF NOT EXISTS finding_edges (
                edge_id        TEXT PRIMARY KEY,
                src_finding_id TEXT NOT NULL,
                dst_finding_id TEXT NOT NULL,
                edge_type      TEXT NOT NULL CHECK (
                    edge_type IN (
                        'related_to', 'derived_from',
                        'updates', 'supports', 'challenges'
                    )
                ),
                confidence     REAL NOT NULL CHECK (
                    confidence >= 0.0 AND confidence <= 1.0
                ),
                created_by     TEXT NOT NULL,
                created_at     TEXT NOT NULL,
                rationale      TEXT NOT NULL DEFAULT '',
                provenance     TEXT NOT NULL DEFAULT '{}',
                UNIQUE (src_finding_id, dst_finding_id, edge_type)
            );

            CREATE INDEX IF NOT EXISTS idx_finding_edges_src
                ON finding_edges(src_finding_id);
            CREATE INDEX IF NOT EXISTS idx_finding_edges_dst
                ON finding_edges(dst_finding_id);
            CREATE INDEX IF NOT EXISTS idx_finding_edges_type
                ON finding_edges(edge_type);
            CREATE INDEX IF NOT EXISTS idx_finding_edges_created_at
                ON finding_edges(created_at);
            CREATE INDEX IF NOT EXISTS idx_finding_edges_src_type
                ON finding_edges(src_finding_id, edge_type);
            CREATE INDEX IF NOT EXISTS idx_finding_edges_dst_type
                ON finding_edges(dst_finding_id, edge_type);
        """)


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


def insert_finding(finding: dict[str, Any]) -> str:
    """Insert a finding. Returns finding ID."""
    finding_id = finding.get("id") or str(uuid.uuid4())

    with _DB_LOCK, _get_conn() as conn:
        existing_row = conn.execute("SELECT * FROM findings WHERE id = ?", (finding_id,)).fetchone()
        if existing_row is not None:
            existing = _row_to_finding(existing_row)
            if (
                existing.get("source_filepath")
                and finding.get("source_filepath")
                and not _same_ingested_source(existing, finding)
            ):
                finding_id = _source_scoped_finding_id(finding_id, finding["source_filepath"])
                source_row = conn.execute(
                    "SELECT * FROM findings WHERE id = ?", (finding_id,)
                ).fetchone()
                if source_row is not None:
                    finding = _merge_existing_finding(_row_to_finding(source_row), finding)
            else:
                finding = _merge_existing_finding(existing, finding)
        conn.execute(
            """INSERT OR REPLACE INTO findings
                   (id, finding_type, title, content, metrics, variant_name,
                    notes, peer_id, generation_id, timestamp, extra)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                finding_id,
                finding.get("finding_type", "result"),
                finding.get("title", ""),
                finding.get("content", ""),
                json.dumps(finding.get("metrics", {})),
                finding.get("variant_name", ""),
                finding.get("notes", ""),
                finding.get("peer_id", ""),
                int(finding.get("generation_id", 0)),
                finding.get("timestamp", datetime.now().isoformat()),
                json.dumps(
                    {
                        k: v
                        for k, v in finding.items()
                        if k
                        not in (
                            "id",
                            "finding_type",
                            "title",
                            "content",
                            "metrics",
                            "variant_name",
                            "notes",
                            "peer_id",
                            "generation_id",
                            "timestamp",
                        )
                    }
                ),
            ),
        )
    _touch_operator_status({"last_finding_id": finding_id})
    return finding_id


_BOUNDARY_VALIDATION_METRIC_KEYS = frozenset(
    {
        "generation_boundary_path",
        "generation_boundary_pending_commit",
        "generation_boundary_evidence_cutoff_at",
        "late_after_generation_boundary",
        "late_observed_generation_id",
        "artifact_signal_status",
        "source_result_mtime",
    }
)


def mark_finding_boundary_validation(finding_id: str, metrics: dict[str, Any]) -> bool:
    """Atomically add boundary status without replacing measured evidence."""

    boundary_metrics = {
        key: copy.deepcopy(value)
        for key, value in metrics.items()
        if key in _BOUNDARY_VALIDATION_METRIC_KEYS
    }
    if not finding_id or not boundary_metrics:
        return False
    with _DB_LOCK, _get_conn() as conn:
        row = conn.execute("SELECT metrics FROM findings WHERE id = ?", (finding_id,)).fetchone()
        if row is None:
            return False
        try:
            current_metrics = json.loads(row[0] or "{}")
        except (TypeError, json.JSONDecodeError):
            current_metrics = {}
        if not isinstance(current_metrics, dict):
            current_metrics = {}
        current_metrics.update(boundary_metrics)
        conn.execute(
            "UPDATE findings SET metrics = ? WHERE id = ?",
            (json.dumps(current_metrics), finding_id),
        )
    return True


def clear_pending_boundary_validation(generation_id: int) -> int:
    """Remove provisional boundary state when that boundary is abandoned."""

    updated = 0
    with _DB_LOCK, _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, metrics FROM findings WHERE generation_id = ?",
            (int(generation_id),),
        ).fetchall()
        for row in rows:
            try:
                metrics = json.loads(row["metrics"] or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(metrics, dict) or not bool(
                metrics.get("generation_boundary_pending_commit")
            ):
                continue
            observed_generation = metrics.get("late_observed_generation_id")
            try:
                if observed_generation is not None and int(observed_generation) != int(
                    generation_id
                ):
                    continue
            except (TypeError, ValueError):
                continue
            for key in _BOUNDARY_VALIDATION_METRIC_KEYS:
                metrics.pop(key, None)
            # Compatibility cleanup for rows written before boundary routing was
            # made non-destructive. These values were introduced as one bundle.
            if metrics.get("exclusion_reason") == "late_after_generation_boundary":
                for key in (
                    "validation_only_result",
                    "promotion_eligible",
                    "clean_promotion_eligible",
                    "excluded_from_durable_frontier",
                    "exclusion_reason",
                    "recommended_next_step",
                ):
                    metrics.pop(key, None)
            conn.execute(
                "UPDATE findings SET metrics = ? WHERE id = ?",
                (json.dumps(metrics), row["id"]),
            )
            updated += 1
    return updated


def snapshot_findings_at_cutoff(generation_id: int) -> tuple[datetime, list[dict[str, Any]]]:
    """Atomically order canonical findings before or after one cutoff instant."""

    with _DB_LOCK, _get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            "SELECT * FROM findings WHERE generation_id = ? ORDER BY timestamp DESC",
            (int(generation_id),),
        ).fetchall()
        cutoff = datetime.now(UTC)
    return cutoff, [_row_to_finding(row) for row in rows]


def _has_finding_value(value: Any) -> bool:
    return value not in (None, "", [], {})


def _merge_finding_dicts(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(existing)
    for key, value in incoming.items():
        if _has_finding_value(value) or key not in merged:
            merged[key] = copy.deepcopy(value)
    return merged


_RESULT_SNAPSHOT_REPLACE_KEYS = (
    *_RESULT_PRODUCER_IDENTITY_KEYS,
    "source_result_path",
    "result_artifact_path",
    "result_path",
    "summary_path",
    "source_result_sha256",
    "validation_only",
    "validation_only_result",
    "late_after_generation_boundary",
    "artifact_signal_status",
    "late_result_policy",
    "durability_scope",
    "excluded_from_durable_frontier",
    "exclusion_reason",
    "promotion_eligible",
    "clean_promotion_eligible",
    "lane",
    "promotion_lane",
    "frontier_lane",
    "scored_complete",
    "is_scored_complete",
    "complete_eval",
    "is_complete_eval",
    "evidence_stage",
    "eval_stage",
    "stage",
    "tier",
    "tier_reached",
    "completed_tier",
    "candidate_tier",
    "status",
    "final_status",
    "tier_status",
    "result_status",
    "completion_status",
    "eval_status",
    "scout_only",
    "is_scout_eval",
    "is_smoke_eval",
    "smoke_only",
    "partial",
    "partial_cohort",
    "partial_eval",
    "is_partial_eval",
    "incomplete_eval",
    "is_incomplete_eval",
    "capped",
    "is_capped",
    "result_capped",
    "summary_only",
    "is_summary_only",
    "unscored_artifact",
    "suspect",
    "suspect_protocol",
    "suspect_fixed_weight_eval",
    "protocol_integrity_failed",
    "protocol_integrity_passed",
    "protocol_integrity_status",
    "protocol_integrity_violation_count",
    "source_generation_low_confidence",
    "mature_enough",
    "maturity_basis",
    "evidence_rank",
    "_inferred_scored_complete",
    "_inferred_result_status",
    "effort_ratio",
    "maturity_effort_ratio",
    "actual_effort_ratio",
    "compute_effort_ratio",
    "training_effort_ratio",
    "effort_ratios",
    "effort_ratio_by_dimension",
    "actual_effort",
    "actual_effort_units",
    "completed_effort",
    "completed_effort_units",
    "reference_effort",
    "reference_effort_units",
    "required_effort",
    "required_effort_units",
    "planned_effort",
    "planned_effort_units",
    "actual_epochs",
    "completed_epochs",
    "reference_epochs",
    "required_epochs",
    "planned_epochs",
    "actual_steps",
    "completed_steps",
    "reference_steps",
    "required_steps",
    "planned_steps",
    "actual_iterations",
    "completed_iterations",
    "reference_iterations",
    "required_iterations",
    "planned_iterations",
    "actual_rollouts",
    "completed_rollouts",
    "reference_rollouts",
    "required_rollouts",
    "planned_rollouts",
    "coverage_ratio",
    "maturity_coverage_ratio",
    "evaluation_coverage_ratio",
    "eval_coverage_ratio",
    "coverage_ratios",
    "coverage_ratio_by_dimension",
    "completed_required_eval_units",
    "completed_eval_units",
    "covered_eval_units",
    "total_required_eval_units",
    "total_eval_units",
    "required_eval_units",
    "actual_eval_units",
    "evaluation_units",
    "scored_cell_count",
    "n_scored_cells",
    "completed_cells",
    "n_eval_cells",
    "cell_count",
    "total_cells",
    "n_cells",
    "n_primary_cells",
    "n_hard_constraint_violations",
    "hard_constraint_violations",
    "n_constraint_violations",
    "constraint_violations",
    "failed_units",
    "failed_eval_units",
    "error_units",
    "missing_units",
    "incomplete_units",
    "failed_unit_count",
    "n_failed_units",
    "error_unit_count",
    "n_error_units",
    "missing_unit_count",
    "n_missing_units",
    "incomplete_unit_count",
    "n_incomplete_units",
    "failed_cells",
    "failed_eval_cells",
    "error_cells",
    "missing_cells",
    "incomplete_cells",
    "failed_cell_count",
    "n_failed_cells",
    "error_cell_count",
    "n_error_cells",
    "missing_cell_count",
    "n_missing_cells",
    "incomplete_cell_count",
    "n_incomplete_cells",
    "source_result_kind",
)


def _refresh_result_snapshot_fields(merged: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key in _RESULT_SNAPSHOT_REPLACE_KEYS:
        merged.pop(key, None)
        if key in incoming:
            merged[key] = incoming[key]
    for container_name in ("metrics", "details", "extra", "current_aggregate"):
        merged_container = merged.get(container_name)
        incoming_container = incoming.get(container_name)
        if isinstance(merged_container, dict):
            _refresh_result_snapshot_fields(
                merged_container,
                incoming_container if isinstance(incoming_container, dict) else {},
            )


def _same_ingested_source(existing: dict[str, Any], incoming: dict[str, Any]) -> bool:
    """Match one filesystem source across run-directory relocation."""

    existing_path = existing.get("source_filepath")
    incoming_path = incoming.get("source_filepath")
    if not existing_path:
        return True
    if not incoming_path:
        return False
    if existing_path == incoming_path:
        return True

    from .findings_ingest import _source_cache_key

    return _source_cache_key(Path(str(existing_path))) == _source_cache_key(
        Path(str(incoming_path))
    )


def _source_scoped_finding_id(finding_id: str, source_filepath: Any) -> str:
    """Disambiguate independent source files that accidentally reuse one ID."""

    from .findings_ingest import _source_cache_key

    source_key = _source_cache_key(Path(str(source_filepath)))
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"praxist-finding-source:{finding_id}:{source_key}"))


def _merge_existing_finding(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge a same-id filesystem/SQLite finding without losing metadata.

    Filesystem findings can be richer than an already-ingested SQLite row.
    Keep incoming non-empty values authoritative, but retain prior non-empty
    metrics/extra/details fields when the incoming shape omits them.
    """

    existing_metrics = existing.get("metrics") if isinstance(existing.get("metrics"), dict) else {}
    incoming_metrics = incoming.get("metrics") if isinstance(incoming.get("metrics"), dict) else {}
    if bool(existing_metrics.get("auto_materialized_from_result_artifact")) and bool(
        incoming_metrics.get("auto_materialized_from_result_artifact")
    ):
        return dict(incoming)

    merged = _merge_finding_dicts(existing, incoming)
    for key in ("metrics", "extra", "details", "current_aggregate"):
        old = existing.get(key) if isinstance(existing.get(key), dict) else {}
        new = incoming.get(key) if isinstance(incoming.get(key), dict) else {}
        if old or new:
            merged[key] = _merge_finding_dicts(old, new)
    try:
        ingest_schema_version = int(incoming.get("ingest_schema_version") or 0)
    except (TypeError, ValueError):
        ingest_schema_version = 0
    if (
        ingest_schema_version >= 3
        and incoming.get("source_filepath")
        and _same_ingested_source(existing, incoming)
    ):
        _refresh_result_snapshot_fields(merged, incoming)
    return merged


def delete_findings_by_ids(finding_ids: list[str] | tuple[str, ...] | set[str]) -> int:
    """Delete findings and advisory graph edges by finding ID.

    The research loop normally treats findings as append-only evidence.  This
    narrow deletion API exists for derived/cache rows such as auto-materialized
    result artifacts: when the source artifact disappears or the task opts out
    of automatic materialization, stale derived rows must not continue to feed
    frontier, Gems, or research-memory evidence packs.
    """

    ids = [str(item) for item in finding_ids if str(item or "").strip()]
    if not ids:
        return 0
    placeholders = ",".join("?" for _ in ids)
    with _DB_LOCK, _get_conn() as conn:
        has_edges = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'finding_edges'"
        ).fetchone()
        if has_edges is not None:
            conn.execute(
                f"""DELETE FROM finding_edges
                    WHERE src_finding_id IN ({placeholders})
                       OR dst_finding_id IN ({placeholders})""",
                [*ids, *ids],
            )
        cursor = conn.execute(
            f"DELETE FROM findings WHERE id IN ({placeholders})",
            ids,
        )
        deleted = int(cursor.rowcount or 0)
    if deleted:
        _touch_operator_status({"deleted_findings": deleted})
    return deleted


def get_findings(
    generation_id: int | None = None,
    finding_type: str | None = None,
    peer_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Query findings, optionally filtered. All filters are additive.

    The `peer_id` filter was added for the session-start graph context
    helper: previously that helper called `get_all_findings()` and
    filtered in Python, which transferred every row for every peer
    render. At 10k findings × 10 peers per cohort that becomes 100k
    row copies per generation for a single string match — the SQL
    filter keeps it O(matches)."""
    clauses = []
    params: list = []
    if generation_id is not None:
        clauses.append("generation_id = ?")
        params.append(generation_id)
    if finding_type:
        clauses.append("finding_type = ?")
        params.append(finding_type)
    if peer_id:
        clauses.append("peer_id = ?")
        params.append(peer_id)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    limit_clause = f" LIMIT {int(limit)}" if limit is not None else ""

    with _get_conn(readonly=True) as conn:
        rows = conn.execute(
            f"SELECT * FROM findings {where} ORDER BY timestamp DESC{limit_clause}",
            params,
        ).fetchall()

    return [_row_to_finding(r) for r in rows]


def get_all_findings() -> list[dict[str, Any]]:
    """Get all findings (for sync compatibility)."""
    return get_findings()


def _row_to_finding(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["metrics"] = json.loads(d.get("metrics", "{}"))
    extra = json.loads(d.pop("extra", "{}"))
    d.update(extra)
    return d


def _finding_field(finding: dict[str, Any], *keys: str) -> Any:
    metrics = finding.get("metrics") if isinstance(finding.get("metrics"), dict) else {}
    details = finding.get("details") if isinstance(finding.get("details"), dict) else {}
    extra_raw = finding.get("extra")
    extra: dict[str, Any] = {}
    if isinstance(extra_raw, dict):
        extra = extra_raw
    elif isinstance(extra_raw, str) and extra_raw.strip():
        try:
            parsed = json.loads(extra_raw)
            if isinstance(parsed, dict):
                extra = parsed
        except json.JSONDecodeError:
            extra = {}
    for key in keys:
        for source in (finding, metrics, details, extra):
            if key in source and source.get(key) is not None:
                return source.get(key)
    return None


def _boolish_finding_field(finding: dict[str, Any], *keys: str) -> bool | None:
    value = _finding_field(finding, *keys)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "yes", "1", "on", "y"}:
            return True
        if token in {"false", "no", "0", "off", "n"}:
            return False
    return None


def _identity_token(value: Any) -> str:
    token = str(value or "").strip().lower()
    token = re.sub(r"[^a-z0-9_./=-]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_.-/=")
    return token


def _source_result_child_token(path_value: Any) -> tuple[str, str]:
    raw = str(path_value or "").strip()
    if not raw:
        return "", ""
    normalized = raw.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part and part not in {".", ".."}]
    child = ""
    for idx, part in enumerate(parts[:-1]):
        if part == "results" and idx + 1 < len(parts):
            child = _identity_token(parts[idx + 1])
            break
    return _identity_token(normalized), child


def _explicit_entity_key(value: Any) -> str:
    raw = str(value or "").strip()
    if "::" not in raw:
        return ""
    prefix, rest = raw.split("::", 1)
    prefix_token = _identity_token(prefix)
    if prefix_token not in {"variant", "artifact", "finding"}:
        return ""
    rest_token = _identity_token(rest)
    if not rest_token:
        return ""
    return f"{prefix_token}::{rest_token}"


def _pareto_entity_key(finding: dict[str, Any]) -> str:
    for key in ("source_result_path", "result_path", "result_artifact_path"):
        path_token, child_token = _source_result_child_token(_finding_field(finding, key))
        if child_token:
            return f"variant::{child_token}"
        if path_token:
            return f"artifact::{path_token}"
    for key in ("frontier_entity_key", "candidate_entity_key"):
        explicit = _explicit_entity_key(_finding_field(finding, key))
        if explicit:
            return explicit
    for key in ("variant_id", "child_id", "variant_name"):
        token = _identity_token(_finding_field(finding, key))
        if token:
            return f"variant::{token}"
    token = _identity_token(finding.get("id"))
    return f"finding::{token}" if token else f"object::{id(finding)}"


def _is_scout_or_partial_finding(finding: dict[str, Any]) -> bool:
    if (
        _boolish_finding_field(
            finding,
            "scout_only",
            "is_scout_eval",
            "is_smoke_eval",
            "summary_only",
            "is_summary_only",
            "partial_cohort",
            "partial_eval",
            "is_partial_eval",
            "unscored_artifact",
            "incomplete_eval",
            "is_incomplete_eval",
            "capped",
            "result_capped",
            "is_capped",
        )
        is True
    ):
        return True
    if (
        _boolish_finding_field(finding, "scored_complete", "is_scored_complete", "complete_eval")
        is False
    ):
        return True
    stage_raw = _finding_field(finding, "evidence_stage", "eval_stage", "stage")
    if stage_raw is not None:
        stage = re.sub(r"[^a-z0-9]+", "_", str(stage_raw).strip().lower()).strip("_")
        if stage in {
            "scout",
            "cheap_probe",
            "probe",
            "smoke",
            "sanity",
            "partial",
            "partial_cohort",
            "summary_only",
            "incomplete",
            "unscored",
            "unscored_artifact",
            "failed_or_unscored",
            "unknown",
        }:
            return True
    for key in ("tier_status", "final_status", "result_status", "completion_status", "status"):
        value = _finding_field(finding, key)
        if value is None:
            continue
        token = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
        if any(
            marker in token
            for marker in (
                "scout",
                "smoke",
                "partial",
                "summary_only",
                "unscored",
                "not_scored",
                "incomplete",
                "capped",
                "scored_complete_false",
            )
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def insert_metric(record: dict[str, Any]) -> int:
    """Insert a metric record. Returns row ID."""
    with _DB_LOCK, _get_conn() as conn:
        cursor = conn.execute(
            """INSERT INTO metrics
                   (run_id, variant_name, metrics, notes, step,
                    peer_id, generation_id, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.get("run_id", ""),
                record.get("variant_name", ""),
                json.dumps(record.get("metrics", {})),
                record.get("notes", ""),
                int(record.get("step", 0)),
                record.get("peer_id", ""),
                int(record.get("generation_id", 0)),
                record.get("timestamp", datetime.now().isoformat()),
            ),
        )
        row_id = cursor.lastrowid
    _touch_operator_status({"last_metric_row_id": row_id})
    return row_id


def _touch_operator_status(extra: dict[str, Any] | None = None) -> None:
    """Refresh orchestrator_status.json after local finding/metric writes."""
    raw_run_dir = os.environ.get("LOCAL_STORE_DIR", "")
    if not raw_run_dir:
        return
    run_dir = Path(raw_run_dir)
    status_path = run_dir / "orchestrator_status.json"
    if not status_path.exists():
        return
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return
        payload["updated_at"] = datetime.now().isoformat()
        payload["findings_total"] = count_findings()
        payload["metrics_total"] = _count_metrics()
        if extra:
            payload.update(extra)
        tmp_path = status_path.with_suffix(status_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp_path.replace(status_path)
    except Exception as exc:  # noqa: BLE001 - operator status is best-effort.
        logger.debug("local_store: orchestrator status touch failed: %s", exc)


def _count_metrics() -> int:
    """Return the number of metric records in the local store."""
    with _get_conn(readonly=True) as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM metrics").fetchone()
    return int(row["c"] if row else 0)


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Finding edges (graph index)
#
# See docs/concepts/architecture.md for design.
# All helpers here are idempotent: duplicate edges (same src+dst+type) are
# silently ignored via the UNIQUE constraint.
# ---------------------------------------------------------------------------

_VALID_EDGE_TYPES = (
    "related_to",
    "derived_from",
    "updates",
    "supports",
    "challenges",
)


def insert_edge(edge: dict[str, Any]) -> str | None:
    """Insert a single edge. Returns edge_id if inserted, None if duplicate.

    Required fields: src_finding_id, dst_finding_id, edge_type, confidence,
    created_by. Auto-filled: edge_id (uuid), created_at (now), rationale, provenance.

    Duplicate edges (same src+dst+type) are silently ignored (returns None) —
    this makes the maintainer idempotent on re-runs.
    """
    if edge["edge_type"] not in _VALID_EDGE_TYPES:
        raise ValueError(f"edge_type must be one of {_VALID_EDGE_TYPES}, got {edge['edge_type']!r}")
    conf = float(edge["confidence"])
    if not (0.0 <= conf <= 1.0):
        raise ValueError(f"confidence must be in [0, 1], got {conf}")

    edge_id = edge.get("edge_id") or str(uuid.uuid4())
    created_at = edge.get("created_at") or datetime.now().isoformat()
    rationale = edge.get("rationale", "")
    provenance = edge.get("provenance", {})
    if isinstance(provenance, dict):
        provenance = json.dumps(provenance)

    with _DB_LOCK, _get_conn() as conn:
        try:
            conn.execute(
                """INSERT INTO finding_edges
                       (edge_id, src_finding_id, dst_finding_id, edge_type,
                        confidence, created_by, created_at, rationale, provenance)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    edge_id,
                    edge["src_finding_id"],
                    edge["dst_finding_id"],
                    edge["edge_type"],
                    conf,
                    edge["created_by"],
                    created_at,
                    rationale,
                    provenance,
                ),
            )
            return edge_id
        except sqlite3.IntegrityError:
            # UNIQUE(src, dst, type) violation — edge already exists.
            return None


# Ordering of strong edge types, strongest first. Used when reconciling
# cross-pass inserts on the same (src, dst) pair so that a later pass
# can't accidentally accumulate BOTH a supports (from an immediate
# share_finding materialization) AND a derived_from (from a later
# maintainer sweep) between the same pair — design §6.3 wants at most
# one strong edge per pair. We keep the stronger type and delete the
# weaker one; confidence is a tiebreaker within the same strength.
_STRONG_EDGE_TYPE_RANK = {
    "derived_from": 4,
    "updates": 3,
    "challenges": 2,
    "supports": 2,
    "related_to": 0,
}


def _existing_strong_edge(conn, src: str, dst: str):
    """Return (edge_id, edge_type, confidence, created_by) of any
    existing strong edge between src and dst, or None."""
    row = conn.execute(
        """SELECT edge_id, edge_type, confidence, created_by
           FROM finding_edges
           WHERE src_finding_id = ? AND dst_finding_id = ?
             AND edge_type IN ('derived_from', 'updates', 'challenges', 'supports')
           ORDER BY confidence DESC LIMIT 1""",
        (src, dst),
    ).fetchone()
    return row


def insert_edges_batch(edges: list[dict[str, Any]]) -> int:
    """Insert many edges in one transaction. Returns number actually inserted
    (duplicates and superseded-weaker-than-existing skipped). Per-edge
    failures do not abort the batch.

    Cross-pass conflict policy: if the batch contains a STRONG edge
    and a strictly stronger edge_type already exists between the same
    (src, dst) pair from a prior pass, the new weaker edge is
    dropped. If the new edge is stronger than the existing one, the
    existing weaker strong edge is removed first.  Agent-declared
    edges (created_by="agent_declared") always win against
    rule-engine edges regardless of rank, mirroring the in-memory
    _resolve() policy — the agent's explicit intent is authoritative.
    """
    inserted = 0
    with _DB_LOCK, _get_conn() as conn:
        for edge in edges:
            try:
                if edge["edge_type"] not in _VALID_EDGE_TYPES:
                    continue
                conf = float(edge["confidence"])
                if not (0.0 <= conf <= 1.0):
                    continue
                src = edge["src_finding_id"]
                dst = edge["dst_finding_id"]
                new_type = edge["edge_type"]
                new_rank = _STRONG_EDGE_TYPE_RANK.get(new_type, 0)
                new_created_by = edge.get("created_by", "")

                # Cross-pass conflict for STRONG edges only. Keep
                # related_to as-is (harmless breadcrumb — design
                # §6.3 allows one strong + one weak per pair).
                if new_rank > 0:
                    existing = _existing_strong_edge(conn, src, dst)
                    if existing is not None:
                        ex_type = existing["edge_type"]
                        if ex_type == new_type:
                            # Same edge_type on same pair — UNIQUE
                            # will catch. Fall through.
                            pass
                        else:
                            ex_rank = _STRONG_EDGE_TYPE_RANK.get(ex_type, 0)
                            ex_created_by = existing["created_by"]
                            ex_agent = ex_created_by == "agent_declared"
                            new_agent = new_created_by == "agent_declared"
                            # Decide whether new supersedes existing.
                            # 1) agent_declared beats rule_engine.
                            # 2) otherwise, higher rank wins.
                            # 3) same rank: higher conf wins.
                            if new_agent and not ex_agent:
                                supersede = True
                            elif ex_agent and not new_agent:
                                supersede = False
                            elif new_rank > ex_rank:
                                supersede = True
                            elif new_rank < ex_rank:
                                supersede = False
                            else:
                                supersede = conf > float(existing["confidence"])
                            if supersede:
                                conn.execute(
                                    "DELETE FROM finding_edges WHERE edge_id = ?",
                                    (existing["edge_id"],),
                                )
                            else:
                                # Existing edge is the winner — drop
                                # the new one silently.
                                continue

                edge_id = edge.get("edge_id") or str(uuid.uuid4())
                created_at = edge.get("created_at") or datetime.now().isoformat()
                rationale = edge.get("rationale", "")
                provenance = edge.get("provenance", {})
                if isinstance(provenance, dict):
                    provenance = json.dumps(provenance)
                conn.execute(
                    """INSERT INTO finding_edges
                           (edge_id, src_finding_id, dst_finding_id, edge_type,
                            confidence, created_by, created_at, rationale, provenance)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        edge_id,
                        src,
                        dst,
                        new_type,
                        conf,
                        new_created_by,
                        created_at,
                        rationale,
                        provenance,
                    ),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                continue  # duplicate — skip silently
            except Exception as e:
                logger.warning(f"insert_edges_batch: skipping edge {edge}: {e}")
    return inserted


def _row_to_edge(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    try:
        d["provenance"] = json.loads(d.get("provenance", "{}"))
    except (json.JSONDecodeError, TypeError):
        d["provenance"] = {}
    return d


def get_edges_for_finding(
    finding_id: str,
    direction: str = "both",  # "out", "in", "both"
    edge_types: list[str] | None = None,
    min_confidence: float = 0.0,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return edges adjacent to ``finding_id``.

    direction="out" → edges where finding_id is src
    direction="in"  → edges where finding_id is dst
    direction="both"→ either end
    """
    clauses = []
    params: list = []
    if direction == "out":
        clauses.append("src_finding_id = ?")
        params.append(finding_id)
    elif direction == "in":
        clauses.append("dst_finding_id = ?")
        params.append(finding_id)
    else:
        clauses.append("(src_finding_id = ? OR dst_finding_id = ?)")
        params.extend([finding_id, finding_id])

    clauses.append("confidence >= ?")
    params.append(float(min_confidence))

    if edge_types:
        placeholders = ",".join("?" * len(edge_types))
        clauses.append(f"edge_type IN ({placeholders})")
        params.extend(edge_types)

    where = " WHERE " + " AND ".join(clauses)
    with _get_conn(readonly=True) as conn:
        rows = conn.execute(
            f"SELECT * FROM finding_edges{where} ORDER BY confidence DESC, created_at ASC LIMIT ?",
            params + [int(limit)],
        ).fetchall()
    return [_row_to_edge(r) for r in rows]


def get_subgraph(
    start_finding_id: str,
    max_depth: int = 1,
    min_confidence: float = 0.55,
    edge_types: list[str] | None = None,
    max_nodes: int = 50,
) -> dict[str, Any]:
    """BFS the undirected graph from ``start_finding_id``.

    Returns ``{"start_finding_id", "nodes", "edges", "truncated"}``.
    Each node carries ``graph_depth``. Edges are bounded at most_nodes × max_nodes
    pairs but capped early via max_nodes.

    This implementation uses Python-side BFS (not the recursive CTE from the
    design doc) because it gives us cleaner control over max_nodes truncation
    and keeps the SQLite query simple (one SELECT per depth level).
    """
    depth_by_node: dict[str, int] = {start_finding_id: 0}
    frontier = {start_finding_id}
    truncated = False

    with _get_conn(readonly=True) as conn:
        for d in range(max_depth):
            if not frontier:
                break
            placeholders = ",".join("?" * len(frontier))
            type_clause = ""
            type_params: list = []
            if edge_types:
                ph = ",".join("?" * len(edge_types))
                type_clause = f" AND edge_type IN ({ph})"
                type_params = list(edge_types)

            rows = conn.execute(
                f"""SELECT src_finding_id, dst_finding_id
                    FROM finding_edges
                    WHERE (src_finding_id IN ({placeholders})
                           OR dst_finding_id IN ({placeholders}))
                      AND confidence >= ?
                      {type_clause}""",
                list(frontier) + list(frontier) + [float(min_confidence)] + type_params,
            ).fetchall()
            next_frontier = set()
            for r in rows:
                for nid in (r["src_finding_id"], r["dst_finding_id"]):
                    if nid not in depth_by_node:
                        depth_by_node[nid] = d + 1
                        next_frontier.add(nid)
                        if len(depth_by_node) >= max_nodes:
                            truncated = True
                            break
                if truncated:
                    break
            if truncated:
                break
            frontier = next_frontier

        node_ids = list(depth_by_node.keys())
        node_placeholders = ",".join("?" * len(node_ids))
        nodes_rows = conn.execute(
            f"SELECT * FROM findings WHERE id IN ({node_placeholders})",
            node_ids,
        ).fetchall()
        nodes = []
        for r in nodes_rows:
            n = _row_to_finding(r)
            n["graph_depth"] = depth_by_node[n["id"]]
            nodes.append(n)
        nodes.sort(key=lambda n: (n["graph_depth"], n.get("timestamp", "")))

        type_clause2 = ""
        type_params2: list = []
        if edge_types:
            ph = ",".join("?" * len(edge_types))
            type_clause2 = f" AND edge_type IN ({ph})"
            type_params2 = list(edge_types)
        edge_rows = conn.execute(
            f"""SELECT * FROM finding_edges
                WHERE src_finding_id IN ({node_placeholders})
                  AND dst_finding_id IN ({node_placeholders})
                  AND confidence >= ?
                  {type_clause2}
                ORDER BY confidence DESC, created_at ASC""",
            node_ids + node_ids + [float(min_confidence)] + type_params2,
        ).fetchall()
        edges = [_row_to_edge(r) for r in edge_rows]

    return {
        "start_finding_id": start_finding_id,
        "max_depth": max_depth,
        "min_confidence": min_confidence,
        "nodes": nodes,
        "edges": edges,
        "truncated": truncated,
    }


def get_unlinked_recent_findings(
    hours: float = 6.0,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Return findings from the last ``hours`` that have NO edges attached.

    Guards exploration diversity: surfaces findings the graph hasn't
    absorbed into any thread yet, so agents aren't pulled toward
    existing clusters when a new direction deserves attention.
    """
    cutoff_ts = datetime.now().timestamp() - float(hours) * 3600.0
    cutoff_iso = datetime.fromtimestamp(cutoff_ts).isoformat()

    with _get_conn(readonly=True) as conn:
        rows = conn.execute(
            """SELECT f.*
               FROM findings f
               WHERE f.timestamp >= ?
                 AND NOT EXISTS (
                   SELECT 1 FROM finding_edges e
                   WHERE e.src_finding_id = f.id OR e.dst_finding_id = f.id
                 )
               ORDER BY f.timestamp DESC
               LIMIT ?""",
            (cutoff_iso, int(limit)),
        ).fetchall()
    return [_row_to_finding(r) for r in rows]


def edge_count_by_type() -> dict[str, int]:
    """Return {edge_type: count} across the whole graph."""
    with _get_conn(readonly=True) as conn:
        rows = conn.execute(
            "SELECT edge_type, COUNT(*) AS c FROM finding_edges GROUP BY edge_type"
        ).fetchall()
    counts = {t: 0 for t in _VALID_EDGE_TYPES}
    for r in rows:
        counts[r["edge_type"]] = r["c"]
    return counts


def count_findings() -> int:
    """Return the number of findings currently stored in the local SQLite index."""
    with _get_conn(readonly=True) as conn:
        r = conn.execute("SELECT COUNT(*) AS c FROM findings").fetchone()
    return int(r["c"]) if r else 0


def count_edges() -> int:
    """Return the number of finding graph edges currently stored locally."""
    with _get_conn(readonly=True) as conn:
        r = conn.execute("SELECT COUNT(*) AS c FROM finding_edges").fetchone()
    return int(r["c"]) if r else 0


def get_leaderboard(
    primary_metric: str = "metric_value",
    direction: str = "maximize",
    generation_id: int | None = None,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Build leaderboard from findings with type='result'."""
    findings = get_findings(
        generation_id=generation_id,
        finding_type="result",
    )

    # Filter to entries that have the primary metric
    scored = []
    for f in findings:
        val = f.get("metrics", {}).get(primary_metric)
        if val is not None:
            scored.append(f)

    reverse = direction == "maximize"
    scored.sort(
        key=lambda f: f["metrics"][primary_metric],
        reverse=reverse,
    )

    return scored[:top_k]


# ---------------------------------------------------------------------------
# Pareto-frontier leaderboard (Praxist v2026-05-01+)
# ---------------------------------------------------------------------------


def _pareto_dominates(
    a_vals: dict[str, float], b_vals: dict[str, float], axes: list[dict[str, str]]
) -> bool:
    """Return True iff `a` Pareto-dominates `b` across `axes`.

    A dominates B when A is no-worse than B on every axis AND strictly
    better on at least one axis. `axes` is a list of
    ``{"name": str, "direction": "maximize"|"minimize"}``.
    """
    no_worse_all = True
    strictly_better_one = False
    for ax in axes:
        n = ax["name"]
        d = ax.get("direction", "maximize")
        av = a_vals[n]
        bv = b_vals[n]
        if d == "maximize":
            if av < bv:
                no_worse_all = False
                break
            if av > bv:
                strictly_better_one = True
        else:  # minimize
            if av > bv:
                no_worse_all = False
                break
            if av < bv:
                strictly_better_one = True
    return no_worse_all and strictly_better_one


def get_pareto_leaderboard(
    primary_metric: str,
    direction: str,
    anchor_metrics: list[dict[str, str]],
    generation_id: int | None = None,
    top_k_dominated: int = 20,
    requires_tier: bool = True,
) -> dict[str, Any]:
    # R8 M1 fix: clamp non-negative — negative slice silently drops the
    # last entry instead of returning an empty list.
    """Build the Pareto leaderboard view used by peers and PI guidance."""
    top_k_dominated = max(0, int(top_k_dominated))
    """Compute the Pareto-non-dominated set of variants across primary +
    anchor metrics.

    Peers see this instead of a single-metric ranking — a variant on the
    Pareto front is one that NO other variant beats on every axis. This
    surfaces accuracy/efficiency/generalization tradeoffs explicitly,
    instead of letting one anchor metric silently override
    everything else.

    Args:
        primary_metric: task-owned metric name, e.g. "metric_value".
        direction: "maximize" or "minimize" for primary.
        anchor_metrics: list of ``{"name", "direction"}`` for secondary
            axes (e.g. secondary accuracy, gap, overhead).
        generation_id: filter to one generation; None for all.
        top_k_dominated: include this many dominated entries (sorted by
            primary) for context — the next-best non-frontier variants.

    Returns:
        ``{"axes": [...], "pareto_front": [...], "dominated_top": [...],
           "n_total": int, "n_pareto": int, "best_in": {axis: variant}}``
    """
    # Build axis list (primary first, then unique anchors)
    axes: list[dict[str, str]] = [{"name": primary_metric, "direction": direction}]
    seen_names = {primary_metric}
    for a in anchor_metrics or []:
        name = a.get("name") if isinstance(a, dict) else None
        if not name or name in seen_names:
            continue
        axes.append(
            {
                "name": name,
                "direction": a.get("direction", "maximize"),
            }
        )
        seen_names.add(name)

    # R8 m1: per-call axis-miss counter; emitted as a single summary
    # warning before return. Helps diagnose `n_total=0` when a task spec
    # lists an anchor metric the runner never emits.
    _axis_miss_counts: dict[str, int] = {}

    # Pull `result` AND `insight` findings, mirroring frontier.py:233.
    # R6 Issue 3 fix: previously the leaderboard accepted only "result",
    # but the frontier accepts both. A peer posting a measured result as
    # `insight` (legitimate per the existing prompt's intent) would be
    # promoted by the frontier but invisible on the leaderboard. Two
    # ranking surfaces should agree on which findings count.
    # `hypothesis` deliberately bypasses leaderboard — pre-test claims
    # have no measurements to rank.
    findings = []
    for ft in ("result", "insight"):
        findings.extend(get_findings(generation_id=generation_id, finding_type=ft))

    # Keep only findings where:
    #   1. tier metadata is present/non-empty when requires_tier=True. Tier
    #      labels are opaque task metadata; core does not classify or rank
    #      values. R2 MAJ-1 fix: full parity with frontier.py:283-345 —
    #      3-place lookup (metrics.tier → details.tier → top-level tier),
    #      reject non-string tier with warning in strict mode, reject all the
    #      promotion_eligible falsy variants the frontier rejects (including
    #      "non-promotable" and numeric 0).
    #   2. every axis metric is present, finite (not NaN/Inf), and numeric
    #      (rejects bools, since bool-as-int slips past `isinstance(_, int)`).
    #      R1 C2 fix: NaN comparisons are always False, which lets a NaN
    #      entry sit on the front forever (can't be dominated, can't
    #      dominate).
    candidates: list[dict[str, Any]] = []
    for f in findings:
        m = f.get("metrics") or {}
        details_dict = f.get("details") or {}
        # ---- Tier filter (mirrors frontier.py:283-321) ----
        tier_raw = m.get("tier")
        if tier_raw is None:
            tier_raw = details_dict.get("tier")
        if tier_raw is None:
            tier_raw = f.get("tier")
        norm_tier = None
        if isinstance(tier_raw, str):
            norm_tier = tier_raw.strip() or None
            if norm_tier is None:
                logger.warning(
                    "pareto leaderboard: finding %s has empty tier metadata; treating as missing",
                    f.get("id", "?"),
                )
        elif tier_raw is not None:
            logger.warning(
                "pareto leaderboard: finding %s has non-string tier %r "
                "(type=%s); treating as missing",
                f.get("id", "?"),
                tier_raw,
                type(tier_raw).__name__,
            )
        # Tier policy depends on `requires_tier` (R6 Issue 2 fix):
        # - When True: strict — require a non-empty string tier metadata field,
        #   but do not interpret the tier value.
        # - When False: permissive — tier metadata is ignored.
        if requires_tier:
            if norm_tier is None:
                continue
            smoke = m.get("is_smoke_eval")
            if smoke is None:
                smoke = details_dict.get("is_smoke_eval")
            if smoke is None:
                smoke = f.get("is_smoke_eval")
            if (
                smoke is True
                or (isinstance(smoke, int | float) and not isinstance(smoke, bool) and smoke != 0)
                or (isinstance(smoke, str) and smoke.strip().lower() in ("true", "yes", "1", "on"))
            ):
                continue
            if _is_scout_or_partial_finding(f):
                continue
        else:
            pass
        # ---- promotion_eligible filter ----
        # When requires_tier=True: strict — explicit-true required (R4 Issue 1).
        # When requires_tier=False: permissive — reject only explicit-false
        # (frontier-style). Missing in either mode follows the same policy.
        elig = m.get("promotion_eligible")
        if elig is None:
            elig = details_dict.get("promotion_eligible")
        if elig is None:
            elig = f.get("promotion_eligible")
        if requires_tier:
            promo_accepts = False
            if elig is True:
                promo_accepts = True
            elif isinstance(elig, str):
                low = elig.strip().lower()
                if low in ("true", "yes", "1", "promotable"):
                    promo_accepts = True
                elif low not in ("false", "no", "0", "non-promotable"):
                    logger.warning(
                        "pareto leaderboard: finding %s has unparseable "
                        "promotion_eligible=%r; treating as ineligible",
                        f.get("id", "?"),
                        elig,
                    )
            elif elig == 1 and not isinstance(elig, bool):
                promo_accepts = True
            if not promo_accepts:
                continue
            clean = m.get("clean_promotion_eligible")
            if clean is None:
                clean = details_dict.get("clean_promotion_eligible")
            if clean is None:
                clean = f.get("clean_promotion_eligible")
            clean_accepts = (
                clean is True
                or (isinstance(clean, int | float) and not isinstance(clean, bool) and clean == 1)
                or (
                    isinstance(clean, str)
                    and clean.strip().lower() in ("true", "yes", "1", "promotable")
                )
            )
            if not clean_accepts:
                continue
        else:
            # Permissive: reject only explicitly-false variants.
            if elig is False:
                continue
            if isinstance(elig, str):
                low = elig.strip().lower()
                if low in ("false", "no", "0", "non-promotable"):
                    continue
            elif elig == 0 and not isinstance(elig, bool):
                continue
        # ---- Numeric / finite axis-value check (C2) ----
        vals: dict[str, float] = {}
        ok = True
        missing_axis: str | None = None
        for ax in axes:
            v = m.get(ax["name"])
            if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
                ok = False
                missing_axis = ax["name"]
                break
            fv = float(v)
            if not math.isfinite(fv):
                ok = False
                missing_axis = ax["name"]
                break
            vals[ax["name"]] = fv
        if ok:
            candidates.append({"finding": f, "vals": vals})
        else:
            # R8 m1: track axis-miss reasons. Emit a single warning at the
            # end (not per-finding) summarizing how many tier-present eligible
            # findings were excluded for missing/non-numeric axes — the
            # most likely cause of an unexpected `n_total=0`.
            if missing_axis is not None:
                _axis_miss_counts.setdefault(missing_axis, 0)
                _axis_miss_counts[missing_axis] += 1

    # Deduplicate by concrete entity — keep the entry with the best primary
    # value per variant/artifact child. The key deliberately includes
    # source_result_path child identity when present so sweep children such as
    # bridge_l1_c005 and bridge_l1_c025 do not collapse under their parent
    # family variant_name.
    rev = direction == "maximize"
    by_variant: dict[str, dict[str, Any]] = {}
    for c in candidates:
        vname = (c["finding"].get("variant_name") or "").strip()
        if not vname:
            # Anonymous result — skip; we can't dedup or attribute it.
            continue
        entity_key = _pareto_entity_key(c["finding"])
        prev = by_variant.get(entity_key)
        if prev is None:
            by_variant[entity_key] = c
            continue
        prev_v = prev["vals"][primary_metric]
        cur_v = c["vals"][primary_metric]
        if (rev and cur_v > prev_v) or ((not rev) and cur_v < prev_v):
            by_variant[entity_key] = c
        elif cur_v == prev_v:
            # Tie on primary — pick the lexicographically larger id for
            # determinism (matches "latest UUID" intuition without
            # depending on timestamp resolution).
            cur_id = str(c["finding"].get("id") or "")
            prev_id = str(prev["finding"].get("id") or "")
            if cur_id > prev_id:
                by_variant[entity_key] = c
    deduped = list(by_variant.values())

    # Pareto partition.
    # R5 m4 note: O(N²). At cohort scale (~50-200 unique variants) this
    # is a few thousand comparisons — sub-millisecond. Past ~1000
    # candidates the cost grows fast (~1M comparisons); log a warning
    # so operators see the scale ceiling approach before it bites.
    if len(deduped) > 1000:
        logger.warning(
            "pareto leaderboard: %d unique candidates exceeds 1000 — "
            "O(N²) partition will be slow; consider per-generation "
            "filtering or capping the candidate set",
            len(deduped),
        )
    pareto: list[dict[str, Any]] = []
    dominated: list[dict[str, Any]] = []
    for i, c in enumerate(deduped):
        is_dom = False
        for j, other in enumerate(deduped):
            if i == j:
                continue
            if _pareto_dominates(other["vals"], c["vals"], axes):
                is_dom = True
                break
        (dominated if is_dom else pareto).append(c)

    # Sort each list by primary metric.
    # R2 MIN-1 fix: tiebreak on finding id so display order is
    # deterministic across reads even when multiple front variants tie
    # on the primary value. Without the tiebreak, dict-insertion-order
    # (which mirrors timestamp-DESC SQL order) could flicker for
    # same-second concurrent inserts.
    def _primary_sort_key(c):
        v = c["vals"][primary_metric]
        # Negate when reversing so the secondary (id) tiebreak stays
        # ascending — gives stable lex-id ordering on ties for both
        # maximize and minimize directions.
        primary_key = -v if rev else v
        return (primary_key, str(c["finding"].get("id") or ""))

    pareto.sort(key=_primary_sort_key)
    dominated.sort(key=_primary_sort_key)

    # Tag best-in-class per axis (within the Pareto front).
    # M4 fix: when multiple front variants tie on an axis, return the
    # full set rather than an arbitrary single winner. The previous
    # impl exposed the first-iterated variant only, which made the
    # `best_in` mapping flicker between calls based on Pareto-front
    # sort order. Now: `best_in[ax_name]` is a list of variant names
    # tied for #1 on that axis (single-element list when no tie).
    # R2 MIN-3 fix: pre-init every axis to `[]` so peers reading
    # `best_in[ax_name]` never KeyError, even when the front is empty
    # (early generation, all findings still use low-count maturity labels).
    best_in: dict[str, list[str]] = {ax["name"]: [] for ax in axes}
    if pareto:
        for ax in axes:
            ax_dir = ax.get("direction", "maximize")
            ax_name = ax["name"]
            best_v = None
            leaders: list[str] = []
            for c in pareto:
                v = c["vals"][ax_name]
                if (
                    best_v is None
                    or (ax_dir == "maximize" and v > best_v)
                    or (ax_dir == "minimize" and v < best_v)
                ):
                    best_v = v
                    leaders = []
                # Equal to best_v → also a leader (tie)
                if v == best_v:
                    nm = (c["finding"].get("variant_name") or "").strip()
                    if nm and nm not in leaders:
                        leaders.append(nm)
            # Deterministic order
            leaders.sort()
            best_in[ax_name] = leaders

    def _serialize(c: dict[str, Any]) -> dict[str, Any]:
        f = c["finding"]
        vname = (f.get("variant_name") or "").strip()
        # `best_in[ax_name]` is now a list of leader names; this variant
        # wins on `ax_name` iff its name appears in that list.
        wins = [ax["name"] for ax in axes if vname and vname in (best_in.get(ax["name"]) or [])]
        return {
            "variant_name": vname,
            "metrics": f.get("metrics", {}),
            "axis_values": c["vals"],
            "best_in": wins,  # axes where this variant is #1 on the Pareto front
            "generation_id": f.get("generation_id"),
            "peer_id": f.get("peer_id", ""),
            "title": f.get("title", ""),
        }

    # R8 m1: surface axis-miss counts so a misconfigured anchor (or a
    # runner that drops an expected metric) becomes visible in peer logs
    # and in the response payload.
    if _axis_miss_counts:
        logger.warning(
            "pareto leaderboard: dropped %d findings due to missing/non-numeric axis values: %s",
            sum(_axis_miss_counts.values()),
            dict(_axis_miss_counts),
        )

    return {
        "axes": axes,
        "pareto_front": [_serialize(c) for c in pareto],
        "dominated_top": [_serialize(c) for c in dominated[:top_k_dominated]],
        "n_total": len(deduped),
        "n_pareto": len(pareto),
        # R2 MIN-2 fix: surface the full dominated count so peers know
        # whether `dominated_top` is the complete set or truncated.
        "n_dominated_total": len(dominated),
        "best_in": best_in,
        # R8 m1: per-axis count of findings excluded for missing values.
        "n_excluded_missing_axis": dict(_axis_miss_counts),
    }
