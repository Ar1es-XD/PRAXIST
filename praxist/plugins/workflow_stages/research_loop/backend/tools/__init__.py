"""MCP tools for autonomous research agents."""

from .http_utils import (
    DEFAULT_HEADERS,
    HAS_HTTPX,
    HAS_REQUESTS,
    async_http_get,
    async_http_post,
    get_server_url,
    validate_safe_identifier,
    validate_safe_path,
)

# Training-subprocess timeout utilities used by per-task tiered_eval
# pipelines. Tasks should import these from here, not from the
# implementation module directly, to keep adoption stable across
# refactors of the timeout subsystem.
from .training_timeout import (
    PartialSummaryPolicy,
    TimeoutPolicy,
    apply_frontier_discount,
    monitor_subprocess_with_grace,
    parse_current_epoch,
    should_emit_partial_summary,
)

__all__ = [
    "get_server_url",
    "async_http_post",
    "async_http_get",
    "validate_safe_identifier",
    "validate_safe_path",
    "HAS_HTTPX",
    "HAS_REQUESTS",
    "DEFAULT_HEADERS",
    "TimeoutPolicy",
    "PartialSummaryPolicy",
    "parse_current_epoch",
    "monitor_subprocess_with_grace",
    "should_emit_partial_summary",
    "apply_frontier_discount",
]
