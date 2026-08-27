#!/usr/bin/env python3
"""Placeholder evaluator for the general machine learning task template."""

from __future__ import annotations

import argparse


def main() -> None:
    """Explain that real tasks must replace the placeholder evaluator."""

    parser = argparse.ArgumentParser(
        description=(
            "Placeholder evaluator for the general machine learning task template. "
            "Replace this file with a task-local evaluator before real Praxist runs."
        )
    )
    parser.add_argument(
        "--prediction-artifact",
        metavar="PATH",
        help="Path to the task-owned prediction artifact accepted by the real evaluator.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Optional path where the real evaluator writes a compact JSON summary.",
    )
    parser.parse_args()

    raise SystemExit(
        "Replace templates/tasks/machine_learning_template/evaluations/primary/run.py "
        "with a task-local evaluator before running real ML experiments."
    )


if __name__ == "__main__":
    main()
