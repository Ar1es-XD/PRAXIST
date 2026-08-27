#!/usr/bin/env python3
"""Variant-local integrator/dt flag resolution for gen71_pod3 C05.

DIG contract C05 mechanism: diagnose whether the whole-disk throttle-tail
joint-gate opening is finite-dt / integrator conditional. The variant keeps the
throttle-tail arms identical across integrators and only varies:

    integrator in {rk2_full, rk4}
    integrator_substeps in {1, 2, 4}  ->  effective physics dt 0.10 / 0.05 / 0.025 s

This module is the single source of truth for the flag names, allowed values,
and the env vars the harness/evaluator use. It does not import or modify the
evaluator, the data split, the metric calculation, or the canonical harness.
"""
from __future__ import annotations

import os
from typing import Mapping

INTEGRATOR_ENV = "SWORDFISH_INTEGRATOR"
SUBSTEPS_ENV = "SWORDFISH_INTEGRATOR_SUBSTEPS"

SUPPORTED_INTEGRATORS = ("rk2_full", "rk4")
SUPPORTED_SUBSTEPS = (1, 2, 4)
MACRO_DT_S = 0.1


def resolve_integrator(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    value = str(env.get(INTEGRATOR_ENV, "rk4")).strip()
    if value not in SUPPORTED_INTEGRATORS:
        raise ValueError(
            f"{INTEGRATOR_ENV}={value!r} not in {SUPPORTED_INTEGRATORS} "
            "(contract C05 restricts the ladder to rk2_full vs rk4)"
        )
    return value


def resolve_substeps(env: Mapping[str, str] | None = None) -> int:
    env = os.environ if env is None else env
    raw = env.get(SUBSTEPS_ENV, "1")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{SUBSTEPS_ENV}={raw!r} is not an integer") from exc
    if value not in SUPPORTED_SUBSTEPS:
        raise ValueError(
            f"{SUBSTEPS_ENV}={value} not in {SUPPORTED_SUBSTEPS} "
            "(dt ladder 0.10/0.05/0.025 requires 1/2/4 substeps)"
        )
    return value


def effective_dt_s(substeps: int) -> float:
    return MACRO_DT_S / float(substeps)


def env_dict(integrator: str, substeps: int) -> dict[str, str]:
    resolve_integrator({"SWORDFISH_INTEGRATOR": integrator})
    resolve_substeps({"SWORDFISH_INTEGRATOR_SUBSTEPS": str(substeps)})
    return {
        INTEGRATOR_ENV: integrator,
        SUBSTEPS_ENV: str(substeps),
    }


def ladder_cells() -> list[tuple[str, int]]:
    """Ordered rk2_full/rk4 x substeps{1,2,4} factorial cells."""
    cells: list[tuple[str, int]] = []
    for integrator in SUPPORTED_INTEGRATORS:
        for substeps in SUPPORTED_SUBSTEPS:
            cells.append((integrator, substeps))
    return cells


def describe_cell(integrator: str, substeps: int) -> str:
    return f"{integrator}@dt{effective_dt_s(substeps):.3f}"


if __name__ == "__main__":
    for cell in ladder_cells():
        print(describe_cell(*cell))
