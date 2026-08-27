"""CLI: build the Finding Graph index for a run.

Modes:
  --mode backfill   Run rule engine over all findings in the run's SQLite,
                    upsert edges. Idempotent via UNIQUE(src, dst, type).
  --mode health     Compute + print graph health stats; write graph_health.json.
  --mode daemon     Start FindingGraphMaintainer and block (shadow mode).
                    Orchestrator already does this automatically in local mode;
                    this is for ad-hoc / post-hoc usage.
  --mode viz        Render <run_dir>/graph/graph.html (a self-contained
                    interactive vis-network page) from the current graph.
  --mode wipe       Delete all edges. Leaves findings untouched. Irreversible.

Examples:

  # Backfill edges for a completed run
  python -m praxist.plugins.graph_maintainers.finding_graph_mvp.cli \\
      --run-dir <task-project>/experiments/run_... \\
      --mode backfill

  # Just check health of an already-built graph
  python -m praxist.plugins.graph_maintainers.finding_graph_mvp.cli \\
      --run-dir <path> --mode health

Design reference: the finding-graph section of docs/concepts/architecture.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


def _setup_env(run_dir: Path):
    """Point local_store at the target run's SQLite + findings_dir."""
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        sys.exit(f"run_dir does not exist: {run_dir}")
    os.environ["LOCAL_STORE_DIR"] = str(run_dir)
    os.environ["LOCAL_FINDINGS_DIR"] = str(run_dir / "shared_findings")


def cmd_backfill(args):
    """Backfill finding graph edges from the current SQLite findings store."""
    _setup_env(args.run_dir)
    from praxist.plugins.graph_maintainers.finding_graph_mvp.engine import (
        FindingGraphBuilder,
        write_graph_health,
    )
    from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store as ls

    ls.init_db()
    all_findings = ls.get_all_findings()
    if not all_findings:
        print("no findings in this run — nothing to build.")
        return
    print(f"loaded {len(all_findings)} findings")

    builder = FindingGraphBuilder(all_findings)
    edges = builder.build_all_edges()
    min_conf = builder.MIN_CONFIDENCE
    edges = [e for e in edges if e["confidence"] >= min_conf]
    print(f"proposed {len(edges)} edges (confidence >= {min_conf})")

    inserted = ls.insert_edges_batch(edges)
    print(f"inserted {inserted} new edges (duplicates silently skipped)")

    graph_dir = args.run_dir / "graph"
    health = write_graph_health(graph_dir)
    print(f"\nhealth (written to {graph_dir}/graph_health.json):")
    print(json.dumps(health, indent=2))


def cmd_health(args):
    """Write finding graph health diagnostics for operator inspection."""
    _setup_env(args.run_dir)
    from praxist.plugins.graph_maintainers.finding_graph_mvp.engine import write_graph_health
    from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store as ls

    ls.init_db()
    graph_dir = args.run_dir / "graph"
    health = write_graph_health(graph_dir)
    print(json.dumps(health, indent=2))


def cmd_daemon(args):
    """Run the finding graph maintainer daemon until stopped."""
    _setup_env(args.run_dir)
    from praxist.plugins.graph_maintainers.finding_graph_mvp.engine import (
        FindingGraphMaintainer,
    )
    from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store as ls

    ls.init_db()
    maintainer = FindingGraphMaintainer(
        run_dir=args.run_dir,
        poll_interval=args.poll_interval,
    )
    print(
        f"starting FindingGraphMaintainer on {args.run_dir} "
        f"(event-driven; fallback {args.poll_interval}s). Ctrl+C to stop."
    )
    maintainer.start()
    try:
        import time

        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nstopping...")
        maintainer.stop()


def cmd_viz(args):
    """Render a static finding graph visualization artifact."""
    _setup_env(args.run_dir)
    from praxist.plugins.graph_maintainers.finding_graph_mvp.viz import (
        build_viz_payload,
        render_graph_html,
    )
    from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store as ls

    ls.init_db()
    payload = build_viz_payload()
    graph_dir = args.run_dir / "graph"
    # Default target is the canonical graph.html; pass --output to
    # write somewhere the orchestrator's maintainer won't overwrite
    # (e.g. when testing a newer viz codepath against a running
    # orchestrator that still has the old render cached in memory).
    out_path = args.output if args.output else graph_dir / "graph.html"
    out = render_graph_html(out_path, payload=payload)
    print(
        f"rendered {out} — {len(payload['nodes'])} nodes, "
        f"{len(payload['edges'])} edges, "
        f"{out.stat().st_size / 1024:.1f} KB"
    )


def cmd_wipe(args):
    """Delete finding graph sidecar edges from the local store."""
    _setup_env(args.run_dir)
    from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store as ls

    ls.init_db()
    before = ls.count_edges()
    if not args.yes:
        reply = input(f"Delete {before} edges? [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("aborted.")
            return
    with ls._get_conn() as conn:
        conn.execute("DELETE FROM finding_edges")
    print(f"deleted {before} edges. Findings table unchanged.")


def main():
    """Command-line entrypoint for the finding_graph_mvp maintenance tool."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build/inspect the Finding Graph sidecar index for a run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to a Praxist run directory",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["backfill", "health", "daemon", "viz", "wipe"],
        help="What to do. See module docstring for descriptions.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=120,
        help="(daemon mode only) seconds between rule passes",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="(wipe mode only) skip confirmation prompt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="(viz mode only) override output HTML path. Default: "
        "<run_dir>/graph/graph.html. Use this to write a side-by-side "
        "file that won't be overwritten by the orchestrator's "
        "maintainer cycle.",
    )

    args = parser.parse_args()
    {
        "backfill": cmd_backfill,
        "health": cmd_health,
        "daemon": cmd_daemon,
        "viz": cmd_viz,
        "wipe": cmd_wipe,
    }[args.mode](args)


if __name__ == "__main__":
    main()
