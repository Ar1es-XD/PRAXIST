"""Read-only lifecycle observation boundaries for Research Runs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_CANCELLED_STOP_REASONS = {"cancelled", "user_interrupt"}
_FAILED_STOP_REASONS = {"runtime_empty", "runtime_failure"}


@dataclass(frozen=True, slots=True)
class PeerLifecycleSummary:
    """Privacy-bounded aggregate status for one generation boundary."""

    generation_ordinal: int
    planned_peer_count: int
    peer_planned_count: int = 0
    peer_running_count: int = 0
    peer_completed_count: int = 0
    peer_cancelled_count: int = 0
    peer_failed_count: int = 0
    peer_unknown_count: int = 0

    @classmethod
    def planned(cls, *, generation_ordinal: int, planned_peer_count: int) -> PeerLifecycleSummary:
        """Describe a cohort before execution without probing or yielding control."""

        return cls(
            generation_ordinal=generation_ordinal,
            planned_peer_count=planned_peer_count,
            peer_planned_count=planned_peer_count,
        )


@runtime_checkable
class ResearchRunLifecycleObserver(Protocol):
    """Observe stable Research Run boundaries without controlling execution."""

    def record_run_started(self, summary: PeerLifecycleSummary) -> object: ...

    def record_generation_finished(self, summary: PeerLifecycleSummary) -> object: ...

    def record_run_finished(
        self,
        *,
        active_duration_seconds: float | None,
        failed: bool = False,
    ) -> object: ...

    def close(self) -> None: ...


def summarize_generation_results(
    *,
    generation_ordinal: int,
    planned_peer_count: int,
    results: list[dict[str, Any]],
) -> PeerLifecycleSummary:
    """Aggregate canonical peer results without exposing result payloads to observers."""

    expected = {
        f"gen{generation_ordinal}_peer{peer_index}" for peer_index in range(planned_peer_count)
    }
    by_peer: dict[str, dict[str, Any]] = {}
    for result in results:
        if not isinstance(result, dict) or result.get("late_result_policy"):
            continue
        peer_id = str(result.get("peer_id") or "")
        if peer_id in expected and peer_id not in by_peer:
            by_peer[peer_id] = result

    completed = cancelled = failed = unknown = 0
    for peer_id in expected:
        result = by_peer.get(peer_id)
        if result is None:
            unknown += 1
            continue
        status = str(result.get("status") or "").lower()
        stop_reason = str(result.get("stop_reason") or "").lower()
        if "cancel" in status or stop_reason in _CANCELLED_STOP_REASONS:
            cancelled += 1
        elif result.get("success") is False or stop_reason in _FAILED_STOP_REASONS:
            failed += 1
        else:
            # A canonical Peer result means the Peer lifecycle returned cleanly.
            # Domain success is intentionally outside this observer boundary.
            completed += 1

    return PeerLifecycleSummary(
        generation_ordinal=generation_ordinal,
        planned_peer_count=planned_peer_count,
        peer_completed_count=completed,
        peer_cancelled_count=cancelled,
        peer_failed_count=failed,
        peer_unknown_count=unknown,
    )


def record_run_started_safely(
    observer: ResearchRunLifecycleObserver | None,
    summary: PeerLifecycleSummary,
) -> None:
    """Notify an optional observer that a Research Run has started."""

    _notify_safely(observer, "record_run_started", summary)


def record_generation_finished_safely(
    observer: ResearchRunLifecycleObserver | None,
    *,
    generation_ordinal: int,
    planned_peer_count: int,
    results: list[dict[str, Any]],
) -> None:
    """Summarize one completed generation and notify an optional observer."""

    if observer is None:
        return
    try:
        summary = summarize_generation_results(
            generation_ordinal=generation_ordinal,
            planned_peer_count=planned_peer_count,
            results=results,
        )
        observer.record_generation_finished(summary)
    except Exception:
        logger.debug("run lifecycle observer failed in record_generation_finished", exc_info=True)


def record_run_finished_safely(
    observer: ResearchRunLifecycleObserver | None,
    *,
    active_duration_seconds: float | None,
    failed: bool,
) -> None:
    """Notify an optional observer that a Research Run has finished."""

    _notify_safely(
        observer,
        "record_run_finished",
        active_duration_seconds=active_duration_seconds,
        failed=failed,
    )


def close_observer_safely(observer: ResearchRunLifecycleObserver | None) -> None:
    """Close an optional observer without changing Research Run behavior."""

    _notify_safely(observer, "close")


def _notify_safely(
    observer: ResearchRunLifecycleObserver | None,
    method_name: str,
    *args: object,
    **kwargs: object,
) -> None:
    """Prevent an observation failure from changing Research Run behavior."""

    if observer is None:
        return
    try:
        getattr(observer, method_name)(*args, **kwargs)
    except Exception:  # Process-control exceptions must retain their normal semantics.
        logger.debug("run lifecycle observer failed in %s", method_name, exc_info=True)
