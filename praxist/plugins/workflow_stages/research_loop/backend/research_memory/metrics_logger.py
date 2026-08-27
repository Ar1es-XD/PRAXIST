"""Metrics logger — record per-synthesis context metrics.

Writes <run_dir>/research_memory/metrics_genN.json with:
  prompt_kb_slope, raw_history_ratio, citation_coverage,
  negative_evidence_ratio, multi_pi_context_multiplier, etc.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def log_synthesis_metrics(
    run_dir: Path,
    generation_id: int,
    prompt_size_bytes: int,
    pack_size_bytes: int,
    n_evidence_cards: int,
    citation_coverage: float,
    negative_evidence_ratio: float,
    panel_mode: str,
    pi_count: int,
    audit_warnings: int,
    audit_blocking: int,
    extra: dict[str, Any] = None,
) -> Path:
    """Write metrics describing PI synthesis quality and evidence coverage."""
    out_dir = Path(run_dir) / "research_memory"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"metrics_gen{generation_id}.json"
    # R7#7 fix: log when overwriting (synthesis retry); preserve prior
    # attempt under .prev so audit trail isn't lost.
    if out_path.exists():
        try:
            prev = out_dir / f"metrics_gen{generation_id}.prev.json"
            out_path.replace(prev)
            logger.warning(
                "metrics_logger: gen %d metrics already existed; "
                "preserved prior attempt at %s before overwrite",
                generation_id,
                prev,
            )
        except OSError:
            pass
    payload = {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "generation_id": generation_id,
        "panel_mode": panel_mode,
        "prompt_size_bytes": prompt_size_bytes,
        "pack_size_bytes": pack_size_bytes,
        "n_evidence_cards": n_evidence_cards,
        "citation_coverage": citation_coverage,
        "negative_evidence_ratio": negative_evidence_ratio,
        "pi_count": pi_count,
        "audit_warnings": audit_warnings,
        "audit_blocking": audit_blocking,
    }
    if extra:
        payload.update(extra)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("metrics_logger: write failed for gen %d: %s", generation_id, e)
    return out_path


def compute_prompt_kb_slope(run_dir: Path, last_n: int = 3) -> float:
    """Return relative growth slope (mean) over last N synthesis events.

    >0.25 sustained → alert per design doc.
    """
    out_dir = Path(run_dir) / "research_memory"
    if not out_dir.exists():
        return 0.0
    sizes = []
    for p in sorted(out_dir.glob("metrics_gen*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            sz = d.get("prompt_size_bytes", 0)
            if sz > 0:
                sizes.append(sz)
        except Exception:
            continue
    if len(sizes) < 2:
        return 0.0
    sizes = sizes[-(last_n + 1) :]
    deltas = []
    for prev, cur in zip(sizes[:-1], sizes[1:], strict=False):
        if prev > 0:
            deltas.append((cur - prev) / prev)
    if not deltas:
        return 0.0
    return sum(deltas) / len(deltas)
