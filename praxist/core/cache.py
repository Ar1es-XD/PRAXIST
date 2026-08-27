"""Prompt cache policy helpers for Gate B."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from praxist.core.protocol import CachePolicy


def frozen_prefix_hash(parts: list[str] | dict[str, Any]) -> str:
    """Return a stable hash for prompt/cache prefixes.

    The hash input is deliberately JSON-normalized so equivalent structured
    prefixes do not drift because of dict ordering or platform line endings.
    """

    normalized = json.dumps(
        _normalize(parts), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_cache_policy(
    *,
    mode: str = "prompt_cache",
    frozen_prefix_parts: list[str] | dict[str, Any],
    cache_breakpoints: list[str] | None = None,
    runtime_cache_strategy: str | None = None,
    provider_cache_strategy: str | None = None,
) -> CachePolicy:
    """Build the runtime cache policy implied by a model profile, provider, and runtime capability set."""
    return CachePolicy(
        mode=mode,
        frozen_prefix_hash=frozen_prefix_hash(frozen_prefix_parts) if mode != "disabled" else None,
        cache_breakpoints=cache_breakpoints or [],
        runtime_cache_strategy=runtime_cache_strategy,
        provider_cache_strategy=provider_cache_strategy,
    )


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n")
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    return value
