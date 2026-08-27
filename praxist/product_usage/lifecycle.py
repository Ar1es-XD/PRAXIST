"""Run-scoped Event ID, Telemetry Run ID, sequence, and lifecycle generation."""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID, uuid4

from pydantic import Field, model_validator

from .protocol import (
    CONSENT_NOTICE_VERSION,
    SCHEMA_VERSION,
    ErrorSummary,
    GenerationFinishedEvent,
    PraxistVersion,
    RunFinishedEvent,
    RunReconciledEvent,
    RunStartedEvent,
    StrictModel,
    UsageEvent,
    canonical_product_version,
    utc_now_seconds,
)


class LifecycleInvariantError(ValueError):
    """Raised when integration code attempts to duplicate a lifecycle boundary."""


class PeerStatusSummary(StrictModel):
    """Aggregate Peer lifecycle counts for one generation boundary."""

    generation_ordinal: int = Field(ge=0)
    planned_peer_count: int = Field(ge=0)
    peer_planned_count: int = Field(ge=0)
    peer_running_count: int = Field(ge=0)
    peer_completed_count: int = Field(ge=0)
    peer_cancelled_count: int = Field(ge=0)
    peer_failed_count: int = Field(ge=0)
    peer_unknown_count: int = Field(ge=0)

    @model_validator(mode="after")
    def peer_counts_match_plan(self) -> PeerStatusSummary:
        actual = (
            self.peer_planned_count
            + self.peer_running_count
            + self.peer_completed_count
            + self.peer_cancelled_count
            + self.peer_failed_count
            + self.peer_unknown_count
        )
        if actual != self.planned_peer_count:
            raise ValueError("peer status counts must sum to planned_peer_count")
        return self

    def wire_fields(self) -> dict[str, int]:
        return self.model_dump()


class RunTelemetryContext:
    """Generate pseudonymous events for exactly one Research Run."""

    def __init__(
        self,
        praxist_version: PraxistVersion,
        *,
        environment_id: UUID,
        clock: Callable[[], str] = utc_now_seconds,
    ) -> None:
        if environment_id.version != 4:
            raise ValueError("environment_id must be UUIDv4")
        self._praxist_version = canonical_product_version(str(praxist_version))
        self._clock = clock
        self._environment_id = environment_id
        self._telemetry_run_id = uuid4()
        self._next_sequence = 1
        self._run_started_emitted = False
        self._finished_generations: set[int] = set()

    @classmethod
    def resume(
        cls,
        praxist_version: PraxistVersion,
        *,
        environment_id: UUID,
        telemetry_run_id: UUID,
        next_sequence: int,
        run_started_emitted: bool,
        finished_generations: set[int] | None = None,
        clock: Callable[[], str] = utc_now_seconds,
    ) -> RunTelemetryContext:
        """Restore state that Praxist persists as part of its own Run state."""

        if environment_id.version != 4:
            raise ValueError("environment_id must be UUIDv4")
        if telemetry_run_id.version != 4:
            raise ValueError("telemetry_run_id must be UUIDv4")
        if next_sequence < 1:
            raise ValueError("next_sequence must be positive")
        context = cls.__new__(cls)
        context._praxist_version = canonical_product_version(str(praxist_version))
        context._clock = clock
        context._environment_id = environment_id
        context._telemetry_run_id = telemetry_run_id
        context._next_sequence = next_sequence
        context._run_started_emitted = run_started_emitted
        context._finished_generations = set(finished_generations or ())
        return context

    @property
    def telemetry_run_id(self) -> UUID:
        return self._telemetry_run_id

    @property
    def environment_id(self) -> UUID:
        return self._environment_id

    @property
    def next_sequence(self) -> int:
        return self._next_sequence

    def run_started(
        self,
        peer_status: PeerStatusSummary,
        *,
        error_summaries: tuple[ErrorSummary, ...] = (),
        error_summaries_truncated: bool = False,
    ) -> RunStartedEvent:
        if self._run_started_emitted:
            raise LifecycleInvariantError("run_started is emitted at most once, including Resume")
        event = RunStartedEvent.model_validate(
            {
                **self._common_fields(
                    "run_started",
                    error_summaries,
                    error_summaries_truncated,
                ),
                **peer_status.wire_fields(),
            }
        )
        self._run_started_emitted = True
        self._advance()
        return event

    def generation_finished(
        self,
        peer_status: PeerStatusSummary,
        *,
        error_summaries: tuple[ErrorSummary, ...] = (),
        error_summaries_truncated: bool = False,
    ) -> GenerationFinishedEvent:
        ordinal = peer_status.generation_ordinal
        if ordinal in self._finished_generations:
            raise LifecycleInvariantError(
                f"generation_finished already emitted for Generation {ordinal}"
            )
        event = GenerationFinishedEvent.model_validate(
            {
                **self._common_fields(
                    "generation_finished",
                    error_summaries,
                    error_summaries_truncated,
                ),
                **peer_status.wire_fields(),
            }
        )
        self._finished_generations.add(ordinal)
        self._advance()
        return event

    def run_finished(
        self,
        active_duration_minutes: int | None,
        *,
        duration_capped: bool,
        error_summaries: tuple[ErrorSummary, ...] = (),
        error_summaries_truncated: bool = False,
    ) -> RunFinishedEvent:
        event = RunFinishedEvent.model_validate(
            {
                **self._common_fields(
                    "run_finished",
                    error_summaries,
                    error_summaries_truncated,
                ),
                "active_duration_minutes": active_duration_minutes,
                "duration_capped": duration_capped,
            }
        )
        self._advance()
        return event

    def run_reconciled(
        self,
        active_duration_minutes: int | None,
        *,
        duration_capped: bool,
        error_summaries: tuple[ErrorSummary, ...] = (),
        error_summaries_truncated: bool = False,
    ) -> RunReconciledEvent:
        event = RunReconciledEvent.model_validate(
            {
                **self._common_fields(
                    "run_reconciled",
                    error_summaries,
                    error_summaries_truncated,
                ),
                "active_duration_minutes": active_duration_minutes,
                "duration_capped": duration_capped,
            }
        )
        self._advance()
        return event

    def _common_fields(
        self,
        event_type: str,
        error_summaries: tuple[ErrorSummary, ...],
        error_summaries_truncated: bool,
    ) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "praxist_version": self._praxist_version,
            "consent_notice_version": CONSENT_NOTICE_VERSION,
            "event_id": uuid4(),
            "environment_id": self._environment_id,
            "telemetry_run_id": self._telemetry_run_id,
            "event_sequence": self._next_sequence,
            "event_type": event_type,
            "occurred_at": self._clock(),
            "error_summaries": error_summaries,
            "error_summaries_truncated": error_summaries_truncated,
        }

    def _advance(self) -> None:
        self._next_sequence += 1


def snapshot_context(context: RunTelemetryContext) -> tuple[UUID, int]:
    """Return the pseudonymous Run ID and next sequence for Run-state storage."""

    return context.telemetry_run_id, context.next_sequence


def event_belongs_to_context(event: UsageEvent, context: RunTelemetryContext) -> bool:
    """Return whether an event belongs to the supplied pseudonymous Run context."""

    return (
        event.environment_id == context.environment_id
        and event.telemetry_run_id == context.telemetry_run_id
    )
