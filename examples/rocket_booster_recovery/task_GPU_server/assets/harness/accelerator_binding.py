"""Preserve Praxist's physical CUDA UUID assignment before importing JAX.

The central scheduler owns device selection.  This helper never translates a
physical UUID to a process-local ordinal, and standalone integer masks remain
untouched for ordinary operator invocations outside Praxist.
"""

from __future__ import annotations

from collections.abc import MutableMapping


def apply_scheduler_assignment(env: MutableMapping[str, str]) -> str:
    """Apply the authoritative scheduler mask and reject conflicting masks."""

    assigned = env.get("PRAXIST_ASSIGNED_GPU_UUIDS", "").strip()
    if not assigned:
        return ""
    tokens = [item.strip() for item in assigned.split(",") if item.strip()]
    if not tokens or any(not item.startswith(("GPU-", "MIG-")) for item in tokens):
        raise RuntimeError(
            "invalid PRAXIST_ASSIGNED_GPU_UUIDS: expected physical GPU/MIG UUIDs"
        )
    canonical = ",".join(tokens)
    for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        current = env.get(key, "").strip()
        if current and current != canonical:
            raise RuntimeError(
                f"accelerator binding mismatch: {key}={current!r}, "
                f"Praxist assignment={canonical!r}"
            )
        env[key] = canonical
    env["PRAXIST_ASSIGNED_GPU_UUIDS"] = canonical
    return canonical


def binding_snapshot(env: MutableMapping[str, str]) -> dict[str, object]:
    """Return non-secret device-binding provenance for the result summary."""

    assigned = env.get("PRAXIST_ASSIGNED_GPU_UUIDS", "").strip()
    return {
        "scheduler_managed": bool(assigned),
        "assigned_gpu_uuids": [x for x in assigned.split(",") if x],
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_visible_devices": env.get("NVIDIA_VISIBLE_DEVICES", ""),
    }
