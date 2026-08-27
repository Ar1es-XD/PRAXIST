"""
Praxist — Deliverables Packager.

Packages results from a completed run into a clean deliverables directory:
  - Executive summary (Markdown)
  - Frontier progression report
  - Best findings with metrics
  - Code snapshots (from frontier entries)
  - Aggregated metrics tables

Usage (CLI):
    python -m praxist.deliver --run-dir <path> --out-dir <path>

Usage (library):
    from praxist.deliver import package_deliverables
    package_deliverables(run_dir, out_dir)
"""

import argparse
import json
import logging
import shutil
import sqlite3
import tarfile
from datetime import datetime
from pathlib import Path

from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
    is_committed_runtime_fact_source,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────


def load_run_summary(run_dir: Path) -> dict | None:
    """Load the finalized run summary used by deliverable packaging."""
    path = run_dir / "run_summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_frontier_manifest(run_dir: Path) -> dict | None:
    """Load the frontier manifest used to select deliverable highlights."""
    path = run_dir / "frontier" / "frontier_manifest.json"
    if path.exists():
        with open(path) as f:
            manifest = json.load(f)
        if isinstance(manifest, dict) and is_committed_runtime_fact_source(
            manifest,
            legacy_ok=True,
        ):
            return manifest
        logger.warning("deliver: ignoring non-committed runtime frontier manifest: %s", path)
    return None


def load_all_findings(run_dir: Path) -> list[dict]:
    """Load all findings from SQLite or filesystem."""
    # Try SQLite
    db_path = run_dir / "shared_store.db"
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path), timeout=5)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM findings ORDER BY generation_id, timestamp"
            ).fetchall()
            conn.close()
            findings = []
            for row in rows:
                d = dict(row)
                d["metrics"] = json.loads(d.get("metrics", "{}"))
                extra = json.loads(d.pop("extra", "{}"))
                d.update(extra)
                findings.append(d)
            return findings
        except Exception:
            pass

    # Filesystem fallback. The findings directory lives AT `<run_dir>/shared_findings`
    # (see generation_loop.py where self.findings_dir is constructed). The
    # previous `run_dir.parent / "shared_findings"` was a latent bug:
    # in normal SQLite-healthy runs the DB path above returned first and
    # masked the error, but if SQLite ever failed (e.g. disk full during DB
    # init, or db file permissions), this fallback silently returned [].
    findings_dir = run_dir / "shared_findings"
    findings = []
    if findings_dir.exists():
        for fp in sorted(findings_dir.glob("*.json")):
            try:
                with open(fp) as f:
                    findings.append(json.load(f))
            except Exception:
                continue

    return findings


def load_all_metrics(run_dir: Path) -> list[dict]:
    """Load all metrics from SQLite or JSONL logs."""
    # Try SQLite
    db_path = run_dir / "shared_store.db"
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path), timeout=5)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM metrics ORDER BY generation_id, timestamp"
            ).fetchall()
            conn.close()
            metrics = []
            for row in rows:
                d = dict(row)
                d["metrics"] = json.loads(d.get("metrics", "{}"))
                metrics.append(d)
            return metrics
        except Exception:
            pass

    # JSONL fallback
    metrics = []
    for log_path in run_dir.rglob("metrics_log.jsonl"):
        try:
            with open(log_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        metrics.append(json.loads(line))
        except Exception:
            continue
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_executive_summary(
    run_summary: dict | None,
    manifest: dict | None,
    findings: list[dict],
    metrics: list[dict],
) -> str:
    """Generate executive summary in Markdown."""
    lines = ["# Praxist — Executive Summary\n"]
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Run overview
    if run_summary:
        lines.append("## Run Overview\n")
        lines.append(f"- **Task**: {run_summary.get('task_name', '?')}")
        lines.append(f"- **Task ID**: {run_summary.get('task_id', '?')}")
        gens = run_summary.get("generations_completed", "?")
        lines.append(f"- **Generations completed**: {gens}")
        dur_s = run_summary.get("total_duration_seconds", 0)
        lines.append(f"- **Total duration**: {dur_s / 3600:.1f} hours")
        lines.append(f"- **Run directory**: `{run_summary.get('run_dir', '?')}`")
        lines.append("")

    # Best results
    if manifest:
        cumulative = manifest.get("cumulative_top", [])
        primary = manifest.get("primary_metric", "?")
        direction = manifest.get("metric_direction", "maximize")
        if cumulative:
            lines.append(f"## Best Results (metric: `{primary}`, {direction})\n")
            lines.append("| Rank | Gen | Variant | Metric | Full Metrics |")
            lines.append("|------|-----|---------|--------|-------------|")
            for i, entry in enumerate(cumulative[:10], 1):
                gen = entry.get("generation_id", "?")
                variant = entry.get("variant_name", "?")[:50]
                val = entry.get("metric_value", "?")
                all_m = entry.get("metrics", {})
                m_str = ", ".join(f"{k}={v}" for k, v in all_m.items())
                lines.append(f"| {i} | {gen} | {variant} | {val} | {m_str} |")
            lines.append("")

    # Findings summary
    if findings:
        type_counts: dict[str, int] = {}
        for f in findings:
            ftype = f.get("finding_type", "unknown")
            type_counts[ftype] = type_counts.get(ftype, 0) + 1

        lines.append("## Findings Summary\n")
        lines.append(f"Total findings: {len(findings)}\n")
        for ftype, count in sorted(type_counts.items()):
            lines.append(f"- **{ftype}**: {count}")
        lines.append("")

    # Generation progression
    if manifest:
        gens = manifest.get("generations", {})
        if gens:
            lines.append("## Generation Progression\n")
            for gen_str in sorted(gens.keys(), key=int):
                entries = gens[gen_str]
                if entries:
                    best = entries[0]
                    lines.append(
                        f"- **Gen {gen_str}**: best = {best.get('metric_value', '?')} "
                        f"({best.get('variant_name', '?')[:40]})"
                    )
            lines.append("")

    # Metrics volume
    if metrics:
        gen_counts: dict[int, int] = {}
        for m in metrics:
            g = m.get("generation_id", 0)
            gen_counts[g] = gen_counts.get(g, 0) + 1
        lines.append("## Experiment Volume\n")
        lines.append(f"Total metric records: {len(metrics)}\n")
        for g in sorted(gen_counts.keys()):
            lines.append(f"- Gen {g}: {gen_counts[g]} metric records")
        lines.append("")

    return "\n".join(lines)


def generate_findings_report(findings: list[dict]) -> str:
    """Generate detailed findings report."""
    lines = ["# Research Findings\n"]

    # Group by generation
    by_gen: dict[int, list[dict]] = {}
    for f in findings:
        g = f.get("generation_id", 0)
        by_gen.setdefault(g, []).append(f)

    for gen_id in sorted(by_gen.keys()):
        gen_findings = by_gen[gen_id]
        lines.append(f"## Generation {gen_id}\n")

        for f in gen_findings:
            ftype = f.get("finding_type", "?")
            title = f.get("title", "Untitled")
            peer = f.get("peer_id", "?")
            variant = f.get("variant_name", "")
            metrics = f.get("metrics", {})
            content = f.get("content", "")

            lines.append(f"### [{ftype.upper()}] {title}\n")
            lines.append(f"- **Peer**: {peer}")
            if variant:
                lines.append(f"- **Variant**: {variant}")
            if metrics:
                m_str = ", ".join(f"`{k}={v}`" for k, v in metrics.items())
                lines.append(f"- **Metrics**: {m_str}")
            if content:
                lines.append(f"\n{content[:2000]}")
                if len(content) > 2000:
                    lines.append("\n*[truncated]*")
            lines.append("")

    return "\n".join(lines)


def generate_metrics_table(metrics: list[dict], primary_metric: str = "") -> str:
    """Generate aggregated metrics table."""
    lines = ["# Aggregated Metrics\n"]

    # Group by variant
    by_variant: dict[str, list[dict]] = {}
    for m in metrics:
        v = m.get("variant_name", "unknown")
        by_variant.setdefault(v, []).append(m)

    if not by_variant:
        lines.append("No metrics recorded.\n")
        return "\n".join(lines)

    # Collect all metric keys
    all_keys = set()
    for entries in by_variant.values():
        for e in entries:
            m_dict = e.get("metrics", {})
            if isinstance(m_dict, str):
                try:
                    m_dict = json.loads(m_dict)
                except Exception:
                    continue
            all_keys.update(m_dict.keys())

    # Sort: primary first
    sorted_keys = sorted(all_keys)
    if primary_metric and primary_metric in sorted_keys:
        sorted_keys.remove(primary_metric)
        sorted_keys.insert(0, primary_metric)

    lines.append("| Variant | Gen | Peer | " + " | ".join(sorted_keys) + " |")
    lines.append("|" + "---|" * (3 + len(sorted_keys)))

    for variant in sorted(by_variant.keys()):
        for entry in by_variant[variant][-10:]:  # last 10 per variant
            m_dict = entry.get("metrics", {})
            if isinstance(m_dict, str):
                try:
                    m_dict = json.loads(m_dict)
                except Exception:
                    m_dict = {}
            gen = entry.get("generation_id", "?")
            peer = entry.get("peer_id", "?")
            vals = [str(m_dict.get(k, "")) for k in sorted_keys]
            lines.append(f"| {variant[:30]} | {gen} | {peer} | " + " | ".join(vals) + " |")

    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot extraction
# ─────────────────────────────────────────────────────────────────────────────


def extract_frontier_snapshots(run_dir: Path, out_dir: Path) -> int:
    """Extract frontier workspace snapshots to deliverables/code/."""
    code_dir = out_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    frontier_dir = run_dir / "frontier"
    if not frontier_dir.exists():
        return 0

    for snapshot in sorted(frontier_dir.rglob("*_snapshot.tar.gz")):
        try:
            target = code_dir / snapshot.stem.replace(".tar", "")
            target.mkdir(parents=True, exist_ok=True)
            with tarfile.open(snapshot, "r:gz") as tar:
                for member in tar.getmembers():
                    if not _safe_tar_member(member, target):
                        logger.warning(f"Skipping unsafe tar member: {member.name}")
                        continue
                    _extract_safe_tar_member(tar, member, target)
            count += 1
        except Exception as e:
            logger.warning(f"Could not extract {snapshot}: {e}")

    return count


def _safe_tar_member(member: tarfile.TarInfo, target: Path) -> bool:
    """Return True only for path-confined regular files/directories.

    Snapshot producers should already filter archives, but deliverable
    unpacking is a separate trust boundary. Do not extract links or special
    files, and do not call extractall() after filtering only some members.
    """
    if not (member.isfile() or member.isdir()):
        return False
    try:
        (target / member.name).resolve().relative_to(target.resolve())
    except ValueError:
        return False
    return True


def _extract_safe_tar_member(tar: tarfile.TarFile, member: tarfile.TarInfo, target: Path) -> None:
    try:
        tar.extract(member, path=str(target), filter="data")
    except TypeError:
        tar.extract(member, path=str(target))


# ─────────────────────────────────────────────────────────────────────────────
# Main packager
# ─────────────────────────────────────────────────────────────────────────────


def package_deliverables(
    run_dir: str,
    out_dir: str,
    name: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Package a completed run into deliverables.

    Returns the output directory path.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    run_summary = load_run_summary(run_dir)
    manifest = load_frontier_manifest(run_dir)
    findings = load_all_findings(run_dir)
    metrics = load_all_metrics(run_dir)

    # Determine output directory
    if not name:
        task_id = run_summary.get("task_id", "unknown") if run_summary else "unknown"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{task_id}_deliverables_{ts}"

    out = Path(out_dir) / name
    if out.exists():
        if overwrite:
            shutil.rmtree(out)
        else:
            raise FileExistsError(f"Output directory exists: {out}. Use --overwrite.")

    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"Packaging deliverables to {out}")

    # Determine primary metric
    primary_metric = ""
    if manifest:
        primary_metric = manifest.get("primary_metric", "")

    # 1. Executive summary
    summary_md = generate_executive_summary(run_summary, manifest, findings, metrics)
    (out / "executive_summary.md").write_text(summary_md)
    logger.info("  Written: executive_summary.md")

    # 2. Detailed findings report
    if findings:
        findings_md = generate_findings_report(findings)
        (out / "findings_report.md").write_text(findings_md)
        logger.info("  Written: findings_report.md")

    # 3. Metrics table
    if metrics:
        metrics_md = generate_metrics_table(metrics, primary_metric)
        (out / "metrics_table.md").write_text(metrics_md)
        logger.info("  Written: metrics_table.md")

    # 4. Raw data
    data_dir = out / "data"
    data_dir.mkdir(exist_ok=True)

    if findings:
        with open(data_dir / "all_findings.json", "w") as f:
            json.dump(findings, f, indent=2, default=str)
    if metrics:
        with open(data_dir / "all_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)
    if manifest:
        with open(data_dir / "frontier_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, default=str)
    if run_summary:
        with open(data_dir / "run_summary.json", "w") as f:
            json.dump(run_summary, f, indent=2, default=str)

    logger.info(f"  Written: data/ ({len(list(data_dir.iterdir()))} files)")

    # 5. Code snapshots from frontier
    n_snapshots = extract_frontier_snapshots(run_dir, out)
    if n_snapshots:
        logger.info(f"  Extracted: code/ ({n_snapshots} snapshots)")

    # 6. Copy frontier finding JSONs
    frontier_dir = run_dir / "frontier"
    if frontier_dir.exists():
        frontier_out = out / "frontier"
        frontier_out.mkdir(exist_ok=True)
        for fj in frontier_dir.rglob("*_finding.json"):
            shutil.copy2(fj, frontier_out / fj.name)

    # 7. README
    readme = _generate_readme(out, run_summary, manifest, findings, metrics)
    (out / "README.md").write_text(readme)

    logger.info(f"Deliverables packaged: {out}")
    return out


def _generate_readme(
    out_dir: Path,
    run_summary: dict | None,
    manifest: dict | None,
    findings: list[dict],
    metrics: list[dict],
) -> str:
    task = run_summary.get("task_name", "?") if run_summary else "?"
    gens = run_summary.get("generations_completed", "?") if run_summary else "?"

    best = "N/A"
    if manifest and manifest.get("cumulative_top"):
        top = manifest["cumulative_top"][0]
        best = f"{top.get('metric_value', '?')} ({top.get('variant_name', '?')[:40]})"

    contents = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    dirs = sorted(p.name for p in out_dir.iterdir() if p.is_dir())

    lines = [
        f"# {task} — Deliverables\n",
        f"- **Generations**: {gens}",
        f"- **Best result**: {best}",
        f"- **Findings**: {len(findings)}",
        f"- **Metric records**: {len(metrics)}",
        f"- **Packaged**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        "## Contents\n",
    ]

    for f in contents:
        lines.append(f"- `{f}`")
    for d in dirs:
        lines.append(f"- `{d}/`")

    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    """CLI entrypoint for packaging a run deliverable bundle."""
    parser = argparse.ArgumentParser(
        description="Package Praxist run results into deliverables",
    )
    parser.add_argument("--run-dir", required=True, help="Path to run directory")
    parser.add_argument("--out-dir", default="deliverables", help="Output base directory")
    parser.add_argument("--name", default="", help="Custom deliverables folder name")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    result = package_deliverables(
        run_dir=args.run_dir,
        out_dir=args.out_dir,
        name=args.name or None,
        overwrite=args.overwrite,
    )
    print(f"\nDeliverables ready: {result}")


if __name__ == "__main__":
    main()
