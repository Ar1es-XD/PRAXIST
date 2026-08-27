#!/usr/bin/env python3
"""Effective-configuration and exact-replication provenance regression."""

from __future__ import annotations

import hashlib
import json


DEFAULTS = {"gain": 1.0, "deadband": 0.02, "enabled": True}


def resolve(raw: dict[str, object]) -> dict[str, object]:
    return {**DEFAULTS, **raw}


def digest(cfg: dict[str, object]) -> str:
    packed = json.dumps(
        cfg, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()


def exact_replication(a: dict[str, object], b: dict[str, object]) -> bool:
    return digest(resolve(a)) == digest(resolve(b))


def main() -> None:
    omitted = resolve({})
    explicit = resolve(DEFAULTS)
    changed = resolve({"gain": 1.1})
    assert digest(omitted) == digest(explicit)
    assert digest(changed) != digest(explicit)
    assert exact_replication({}, DEFAULTS)
    assert not exact_replication({}, {"gain": 1.1})
    assert set(omitted) == set(DEFAULTS)
    print("effective configuration regression: PASS")


if __name__ == "__main__":
    main()
