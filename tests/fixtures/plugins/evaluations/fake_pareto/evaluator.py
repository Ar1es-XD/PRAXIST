"""Executable deterministic fake evaluation plugin."""

from __future__ import annotations

from typing import Any


class FakeParetoEvaluation:
    evaluation_ref = "evaluation:fake_pareto"

    def rank(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(records, key=lambda item: str(item.get("id") or item.get("title") or ""))


def create_evaluation() -> FakeParetoEvaluation:
    return FakeParetoEvaluation()
