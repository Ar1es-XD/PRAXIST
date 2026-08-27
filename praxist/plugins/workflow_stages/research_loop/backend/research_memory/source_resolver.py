"""Source resolver — turns evidence_card.source_ref (relative paths) into
loaded raw content on demand.

Why relative paths:
  - run_dir is the canonical anchor; portable across mounts/backups
  - PI MCP tools can resolve on demand (lazy load)
  - Ledger files stay tiny (no embedded raw JSON)

This module is read-only.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SourceResolver:
    """Resolve evidence_card.source_ref into raw file content.

    Supported source_ref keys (any subset; resolver uses what's present):
      finding_path: relative to run_dir, e.g. "shared_findings/<id>.json"
      finding_id:   used for SQLite lookup if path missing
      agenda_path:  relative to run_dir, e.g. "agendas/research_agenda_genN.yaml"
      result_path:  relative to run_dir, e.g. "results/<variant>/<file>.json"
      raw_log_path: relative to run_dir
    """

    def __init__(self, run_dir: Path, sqlite_db: Path | None = None):
        # R2#5 fix: use os.path.realpath at init to anchor a stable, fully
        # symlink-resolved run_dir. .resolve() can produce paths that
        # relative_to() accepts even when the underlying location escapes
        # the intended directory via symlinks.
        import os

        self.run_dir = Path(os.path.realpath(str(Path(run_dir))))
        # Optional SQLite fallback for finding_id-only refs
        if sqlite_db is None:
            candidate = self.run_dir / "shared_store.db"
            self.sqlite_db = candidate if candidate.exists() else None
        else:
            self.sqlite_db = Path(sqlite_db) if Path(sqlite_db).exists() else None

    def _safe_join(self, rel: str) -> Path | None:
        """Join run_dir with relative path; reject path traversal."""
        if not rel:
            return None
        # R4#4 fix: explicitly reject absolute paths. Path("/a") / "/b"
        # returns Path("/b") (the absolute path discards the LHS), which
        # would let a caller bypass the run_dir containment guarantee
        # by passing an absolute path that happens to live inside run_dir.
        if Path(rel).is_absolute():
            logger.warning(
                "source_resolver: rejecting absolute path: %r "
                "(only relative paths under run_dir are allowed)",
                rel,
            )
            return None
        # R2#5 fix: realpath instead of resolve() so symlinks anywhere on
        # the path are followed before the contains-check. Reject paths
        # that don't exist before realpath (avoids resolving phantom paths).
        import os

        candidate = self.run_dir / rel
        try:
            real = Path(os.path.realpath(str(candidate)))
        except Exception:
            return None
        # Containment check via realpath strings, not Path.relative_to
        # (which can be fooled by symlink chains in some edge cases).
        try:
            run_real = str(self.run_dir)
            real_str = str(real)
            if not (real_str == run_real or real_str.startswith(run_real + os.sep)):
                logger.warning("source_resolver: rejecting path outside run_dir: %r", rel)
                return None
        except Exception:
            return None
        return real

    def resolve(self, source_ref: dict[str, Any]) -> dict[str, Any]:
        """Return a dict {kind, path, content} or {kind, path, error}."""
        # Priority order: explicit path > finding_id (DB lookup)
        for key, kind in (
            ("finding_path", "finding"),
            ("agenda_path", "agenda"),
            ("result_path", "result"),
            ("raw_log_path", "log"),
        ):
            rel = source_ref.get(key)
            if not rel:
                continue
            p = self._safe_join(rel)
            if p is None or not p.exists():
                return {
                    "kind": kind,
                    "path": rel,
                    "error": f"path not found or rejected: {rel}",
                }
            try:
                if p.suffix.lower() == ".json":
                    with open(p, encoding="utf-8") as f:
                        content = json.load(f)
                elif p.suffix.lower() in (".yaml", ".yml"):
                    import yaml

                    with open(p, encoding="utf-8") as f:
                        content = yaml.safe_load(f)
                else:
                    with open(p, encoding="utf-8", errors="replace") as f:
                        content = f.read()
                return {"kind": kind, "path": str(p.relative_to(self.run_dir)), "content": content}
            except Exception as e:
                return {
                    "kind": kind,
                    "path": rel,
                    "error": f"read failed: {type(e).__name__}: {e}",
                }

        # Fallback: SQLite lookup by finding_id
        finding_id = source_ref.get("finding_id")
        if finding_id and self.sqlite_db is not None:
            try:
                with sqlite3.connect(str(self.sqlite_db)) as conn:
                    conn.row_factory = sqlite3.Row
                    cur = conn.execute(
                        "SELECT id, finding_type, title, content, metrics, "
                        "variant_name, notes, peer_id, generation_id, timestamp, extra "
                        "FROM findings WHERE id = ?",
                        (finding_id,),
                    )
                    row = cur.fetchone()
                    if row is None:
                        return {
                            "kind": "finding",
                            "error": f"finding_id not in shared_store.db: {finding_id}",
                        }
                    out = dict(row)
                    # parse JSON fields
                    for jk in ("metrics", "extra"):
                        if isinstance(out.get(jk), str):
                            with contextlib.suppress(Exception):
                                out[jk] = json.loads(out[jk])
                    return {"kind": "finding_db", "content": out}
            except Exception as e:
                return {
                    "kind": "finding_db",
                    "error": f"sqlite read failed: {type(e).__name__}: {e}",
                }

        return {"error": "source_ref has no resolvable key"}
