#!/usr/bin/env python3
"""
CLI wrapper for Praxist deliverables packaging.

Usage:
    python scripts/deliver_auto_research.py --run-dir ../sam_optimizer/experiments/run_... --out-dir deliverables
    python scripts/deliver_auto_research.py --latest --out-dir deliverables

Current external-task runs should pass ``--run-dir`` explicitly. ``--latest`` is
kept for legacy local ``experiments_tracking/`` directories only.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def find_latest_run() -> str:
    """Find the newest legacy ``experiments_tracking`` run directory."""
    tracking = Path("experiments_tracking")
    if not tracking.exists():
        print("No experiments_tracking/ directory found.")
        sys.exit(1)
    runs = sorted(tracking.glob("run_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        print("No run directories found.")
        sys.exit(1)
    return str(runs[-1])


def main():
    """CLI entrypoint for packaging a Praxist run directory."""
    parser = argparse.ArgumentParser(
        description="Package Praxist run deliverables",
    )
    parser.add_argument("--run-dir", help="Path to run directory")
    parser.add_argument("--latest", action="store_true", help="Use latest run")
    parser.add_argument("--out-dir", default="deliverables", help="Output directory")
    parser.add_argument("--name", default="", help="Custom folder name")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.latest:
        run_dir = find_latest_run()
        print(f"Using latest run: {run_dir}")
    elif args.run_dir:
        run_dir = args.run_dir
    else:
        parser.print_help()
        sys.exit(1)

    import logging

    from praxist.deliver import package_deliverables

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    result = package_deliverables(
        run_dir=run_dir,
        out_dir=args.out_dir,
        name=args.name or None,
        overwrite=args.overwrite,
    )
    print(f"\nDeliverables ready: {result}")


if __name__ == "__main__":
    main()
