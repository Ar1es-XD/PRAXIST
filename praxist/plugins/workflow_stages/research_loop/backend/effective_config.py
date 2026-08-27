"""Optional evaluator-authored effective-configuration provenance."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

EFFECTIVE_CONFIG_METADATA_KEYS = (
    "source_result_effective_config_sha256",
    "source_result_effective_config_complete",
    "source_result_effective_config_status",
    "replication_of_effective_config_sha256",
    "replication_effective_config_match",
    "replication_effective_config_status",
)
EFFECTIVE_CONFIG_SUMMARY_KEYS = (
    "effective_config",
    "effective_config_complete",
    "effective_config_incomplete_reasons",
    "replication_of_effective_config_sha256",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def strip_effective_config_fields(value: Any) -> None:
    """Remove source-only declarations and system-computed projections in place."""

    if isinstance(value, dict):
        for key in (*EFFECTIVE_CONFIG_METADATA_KEYS, *EFFECTIVE_CONFIG_SUMMARY_KEYS):
            value.pop(key, None)
        for child in value.values():
            strip_effective_config_fields(child)
    elif isinstance(value, list):
        for child in value:
            strip_effective_config_fields(child)


def has_effective_config_metadata(value: Any) -> bool:
    """Return whether a compact trusted context carries config provenance."""

    if isinstance(value, dict):
        if any(value.get(key) not in (None, "", [], {}) for key in EFFECTIVE_CONFIG_METADATA_KEYS):
            return True
        return any(has_effective_config_metadata(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(has_effective_config_metadata(child) for child in value)
    return False


def _canonical_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def result_effective_config_metadata(summary: dict[str, Any]) -> dict[str, Any]:
    """Return compact provenance without changing legacy result semantics.

    The evaluator summary owns the full configuration. Downstream Praxist
    artifacts carry only its digest and verification state, while the existing
    ``source_result_path`` remains the pointer to the complete declaration.
    """

    if not any(key in summary for key in EFFECTIVE_CONFIG_SUMMARY_KEYS):
        return {}

    metadata: dict[str, Any] = {}
    config_present = "effective_config" in summary
    config = summary.get("effective_config")
    config_digest = ""
    config_valid = isinstance(config, dict) and bool(config)
    if config_valid:
        try:
            config_digest = _canonical_digest(config)
        except (TypeError, ValueError):
            config_valid = False

    reasons = summary.get("effective_config_incomplete_reasons")
    reasons_valid = reasons is None or (
        isinstance(reasons, list)
        and all(isinstance(reason, str) and reason.strip() for reason in reasons)
    )
    has_reasons = bool(reasons) if isinstance(reasons, list) else False
    declared_complete = summary.get("effective_config_complete")
    complete = bool(
        config_valid and declared_complete is True and reasons_valid and not has_reasons
    )

    if config_digest:
        metadata["source_result_effective_config_sha256"] = config_digest
    metadata["source_result_effective_config_complete"] = complete
    if not config_present:
        config_status = "missing"
    elif not config_valid:
        config_status = "invalid"
    elif not reasons_valid:
        config_status = "invalid_incomplete_reasons"
    elif declared_complete is True and not has_reasons:
        config_status = "complete"
    elif declared_complete is False or has_reasons:
        config_status = "declared_incomplete"
    else:
        config_status = "completeness_unverified"
    metadata["source_result_effective_config_status"] = config_status

    raw_parent_digest = summary.get("replication_of_effective_config_sha256")
    if raw_parent_digest is None:
        return metadata
    parent_digest = str(raw_parent_digest).strip().lower()
    if not _SHA256_RE.fullmatch(parent_digest):
        metadata["replication_effective_config_status"] = "invalid_parent_digest"
        return metadata

    metadata["replication_of_effective_config_sha256"] = parent_digest
    if not complete or not config_digest:
        metadata["replication_effective_config_status"] = "current_config_unverified"
        return metadata

    matched = config_digest == parent_digest
    metadata["replication_effective_config_match"] = matched
    metadata["replication_effective_config_status"] = "matched" if matched else "mismatch"
    return metadata
