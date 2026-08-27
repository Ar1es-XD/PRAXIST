#!/usr/bin/env python3
"""Precommit disjoint development and formal OOD subsets.

The formal bank is fixed before controller tuning: 8,192 rows from the
near-OOD velocity bank and 8,192 rows from the hard outer-annulus bank.
No outcome-dependent filtering is performed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_BANK_DIR = ROOT / "data/source_banks"
SOURCES = {
    "near_ood": SOURCE_BANK_DIR / "near_ood_easy_velocity_40960.npz",
    "hard_ood": SOURCE_BANK_DIR / "hard_ood_fast_outer_annulus_40960.npz",
}
SELECTION_SEED = 20260821
DEV_PER_SOURCE = 1024
FORMAL_PER_SOURCE = 8192


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    out = ROOT / "data"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SELECTION_SEED)
    formal_states: list[np.ndarray] = []
    formal_labels: list[np.ndarray] = []
    formal_rows: list[np.ndarray] = []
    dev_states: list[np.ndarray] = []
    dev_labels: list[np.ndarray] = []
    dev_rows: list[np.ndarray] = []
    source_meta = {}

    for source_id, (label, path) in enumerate(SOURCES.items()):
        if not path.is_file():
            raise FileNotFoundError(f"repository-local source bank is missing: {path}")
        with np.load(path, allow_pickle=False) as bank:
            states = np.asarray(bank["state"], dtype=np.float32)
        order = rng.permutation(states.shape[0])
        dev_idx = order[:DEV_PER_SOURCE]
        formal_idx = order[DEV_PER_SOURCE:DEV_PER_SOURCE + FORMAL_PER_SOURCE]
        if np.intersect1d(dev_idx, formal_idx).size:
            raise RuntimeError(f"dev/formal overlap for {label}")
        dev_states.append(states[dev_idx])
        dev_labels.append(np.full(DEV_PER_SOURCE, source_id, np.int8))
        dev_rows.append(dev_idx.astype(np.int32))
        formal_states.append(states[formal_idx])
        formal_labels.append(np.full(FORMAL_PER_SOURCE, source_id, np.int8))
        formal_rows.append(formal_idx.astype(np.int32))
        source_meta[label] = {
            "source_id": source_id,
            "repository_path": str(path.relative_to(ROOT)),
            "absolute_path": str(path.resolve()),
            "sha256": sha256(path),
            "source_rows": int(states.shape[0]),
            "dev_rows": int(DEV_PER_SOURCE),
            "formal_rows": int(FORMAL_PER_SOURCE),
        }

    formal_path = out / "formal_ood_16384.npz"
    dev_path = out / "development_ood_2048.npz"
    np.savez_compressed(
        formal_path,
        state=np.concatenate(formal_states),
        source_id=np.concatenate(formal_labels),
        source_row=np.concatenate(formal_rows),
        source_names=np.asarray(list(SOURCES), dtype="U16"),
    )
    np.savez_compressed(
        dev_path,
        state=np.concatenate(dev_states),
        source_id=np.concatenate(dev_labels),
        source_row=np.concatenate(dev_rows),
        source_names=np.asarray(list(SOURCES), dtype="U16"),
    )
    manifest = {
        "protocol": "precommitted_stratified_ood_subset_v1",
        "selection_seed": SELECTION_SEED,
        "selection_algorithm": "numpy.default_rng(seed).permutation; first 1024 dev, next 8192 formal per source",
        "outcome_dependent_filtering": False,
        "sources": source_meta,
        "development": {
            "repository_path": str(dev_path.relative_to(ROOT)),
            "absolute_path": str(dev_path.resolve()),
            "rows": int(2 * DEV_PER_SOURCE),
            "sha256": sha256(dev_path),
        },
        "formal": {
            "repository_path": str(formal_path.relative_to(ROOT)),
            "absolute_path": str(formal_path.resolve()),
            "rows": int(2 * FORMAL_PER_SOURCE),
            "sha256": sha256(formal_path),
            "success_threshold": 0.05,
        },
    }
    manifest_path = out / "ood_subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
