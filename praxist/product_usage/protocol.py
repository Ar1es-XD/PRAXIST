"""Closed Usage Schema V2 shared by the SDK and collector core."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = 2
CONSENT_NOTICE_VERSION = 3
MAX_BATCH_EVENTS = 50
MAX_REQUEST_BYTES = 32 * 1024
MAX_ERROR_SUMMARIES = 16
MAX_ERROR_COUNT = 65_535
MAX_DURATION_MINUTES = 43_200

_UTC_SECOND_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_CANONICAL_PRODUCT_VERSION_RE = re.compile(
    r"^(?P<public>(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*))*"
    r"(?:(?:a|b|rc)(?:0|[1-9][0-9]*))?"
    r"(?:\.post(?:0|[1-9][0-9]*))?"
    r"(?:\.dev(?:0|[1-9][0-9]*))?)"
    r"(?:\+(?:[0-9a-z]+(?:[.-][0-9a-z]+)*))?$"
)

UtcSecondTimestamp = Annotated[
    str,
    StringConstraints(
        strip_whitespace=False,
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
    ),
]

PraxistVersion = Annotated[
    str,
    StringConstraints(
        strip_whitespace=False,
        min_length=1,
        max_length=32,
        pattern=r"^[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*$",
    ),
]


def canonical_product_version(value: str) -> str:
    """Return a canonical public version safe for the closed wire schema."""

    match = _CANONICAL_PRODUCT_VERSION_RE.fullmatch(value)
    if match is None:
        raise ValueError("praxist_version must be a canonical public version")
    return match.group("public")


PraxistErrorCode: TypeAlias = Literal[
    "PRX-CAPACITY",
    "PRX-PEER-LAUNCH",
    "PRX-PEER-RUNTIME",
    "PRX-RUNTIME",
    "PRX-RUN-FAILED",
    "PRX-UNKNOWN",
]


class StrictModel(BaseModel):
    """Base model that makes schema expansion explicit."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ErrorScope(StrEnum):
    """Closed scope vocabulary for aggregate operational errors."""

    RUN = "run"
    GENERATION = "generation"
    PEER = "peer"


class ErrorStage(StrEnum):
    """Closed lifecycle-stage vocabulary for aggregate errors."""

    SETUP = "setup"
    LAUNCH = "launch"
    EXECUTION = "execution"
    FINALIZATION = "finalization"
    RECONCILIATION = "reconciliation"


class ErrorType(StrEnum):
    """Closed, task-agnostic category vocabulary for aggregate errors."""

    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    ORCHESTRATION = "orchestration"
    RUNTIME = "runtime"
    EXTERNAL_DEPENDENCY = "external_dependency"
    STORAGE = "storage"
    UNKNOWN = "unknown"


class ReasonCode(StrEnum):
    """Closed reason vocabulary for privacy-bounded failure summaries."""

    AUTH_ERROR = "auth_error"
    QUOTA_EXHAUSTED = "quota_exhausted"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    RUNTIME_ERROR = "runtime_error"
    TOOL_UNAVAILABLE = "tool_unavailable"
    INVALID_REQUEST = "invalid_request"
    BUDGET_DENIED = "budget_denied"
    BUDGET_EXPIRED = "budget_expired"
    CAPACITY_UNAVAILABLE = "capacity_unavailable"
    PROCESS_START_FAILED = "process_start_failed"
    STATE_UNREADABLE = "state_unreadable"
    UNEXPECTED_TERMINATION = "unexpected_termination"
    UNKNOWN = "unknown"


class ErrorSummary(StrictModel):
    """Bounded aggregate error count without message or task content."""

    scope: ErrorScope
    stage: ErrorStage
    error_type: ErrorType
    error_code: PraxistErrorCode
    reason_code: ReasonCode
    count: int = Field(ge=1, le=MAX_ERROR_COUNT)
    count_capped: bool = False

    @model_validator(mode="after")
    def capped_count_is_the_cap(self) -> ErrorSummary:
        if self.count_capped and self.count != MAX_ERROR_COUNT:
            raise ValueError("count must equal 65535 when count_capped is true")
        return self


class CommonEvent(StrictModel):
    """Fields shared by every Usage Schema V2 lifecycle event."""

    schema_version: Literal[2]
    praxist_version: PraxistVersion
    # The collector remains compatible with the already released V2 notice;
    # current clients always emit CONSENT_NOTICE_VERSION.
    consent_notice_version: Literal[2, 3]
    event_id: UUID4
    environment_id: UUID4
    telemetry_run_id: UUID4
    event_sequence: int = Field(ge=1)
    occurred_at: UtcSecondTimestamp
    error_summaries: tuple[ErrorSummary, ...] = Field(
        default=(),
        max_length=MAX_ERROR_SUMMARIES,
    )
    error_summaries_truncated: bool = False

    @field_validator("occurred_at")
    @classmethod
    def occurred_at_is_utc_seconds(cls, value: str) -> str:
        validate_utc_second(value)
        return value


class PeerStatusEvent(CommonEvent):
    """Common aggregate Peer counts for generation-scoped events."""

    generation_ordinal: int = Field(ge=0)
    planned_peer_count: int = Field(ge=0)
    peer_planned_count: int = Field(ge=0)
    peer_running_count: int = Field(ge=0)
    peer_completed_count: int = Field(ge=0)
    peer_cancelled_count: int = Field(ge=0)
    peer_failed_count: int = Field(ge=0)
    peer_unknown_count: int = Field(ge=0)

    @model_validator(mode="after")
    def peer_counts_match_plan(self) -> PeerStatusEvent:
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


class RunStartedEvent(PeerStatusEvent):
    """Aggregate snapshot emitted at a Research Run start boundary."""

    event_type: Literal["run_started"]

    @model_validator(mode="after")
    def error_window_is_setup_or_first_launch(self) -> RunStartedEvent:
        allowed = {ErrorStage.SETUP, ErrorStage.LAUNCH}
        if any(summary.stage not in allowed for summary in self.error_summaries):
            raise ValueError("run_started errors must be from setup or launch")
        return self


class GenerationFinishedEvent(PeerStatusEvent):
    """Aggregate snapshot emitted after a durable generation boundary."""

    event_type: Literal["generation_finished"]

    @model_validator(mode="after")
    def error_window_is_current_generation(self) -> GenerationFinishedEvent:
        allowed = {
            ErrorStage.LAUNCH,
            ErrorStage.EXECUTION,
            ErrorStage.FINALIZATION,
        }
        if any(summary.stage not in allowed for summary in self.error_summaries):
            raise ValueError(
                "generation_finished errors must be from launch, execution, or finalization"
            )
        return self


class DurationEvent(CommonEvent):
    """Common bounded active-duration fields for terminal events."""

    active_duration_minutes: int | None = Field(ge=0, le=MAX_DURATION_MINUTES)
    duration_capped: bool

    @model_validator(mode="after")
    def capped_duration_is_the_cap(self) -> DurationEvent:
        if self.duration_capped and self.active_duration_minutes != MAX_DURATION_MINUTES:
            raise ValueError(
                "active_duration_minutes must equal 43200 when duration_capped is true"
            )
        return self


class RunFinishedEvent(DurationEvent):
    """Terminal aggregate snapshot for a normally observed Run process."""

    event_type: Literal["run_finished"]

    @model_validator(mode="after")
    def error_window_is_finalization(self) -> RunFinishedEvent:
        if any(summary.stage is not ErrorStage.FINALIZATION for summary in self.error_summaries):
            raise ValueError("run_finished errors must be from finalization")
        return self


class RunReconciledEvent(DurationEvent):
    """Terminal aggregate snapshot reconstructed after an interrupted process."""

    event_type: Literal["run_reconciled"]

    @model_validator(mode="after")
    def error_window_is_reconciliation(self) -> RunReconciledEvent:
        if any(summary.stage is not ErrorStage.RECONCILIATION for summary in self.error_summaries):
            raise ValueError("run_reconciled errors must be from reconciliation")
        return self


UsageEvent: TypeAlias = Annotated[
    RunStartedEvent | GenerationFinishedEvent | RunFinishedEvent | RunReconciledEvent,
    Field(discriminator="event_type"),
]


class UsageBatch(StrictModel):
    """A bounded request containing only closed-schema lifecycle events."""

    events: tuple[UsageEvent, ...] = Field(min_length=1, max_length=MAX_BATCH_EVENTS)


_EVENT_ADAPTER: TypeAdapter[UsageEvent] = TypeAdapter(UsageEvent)


def validate_utc_second(value: str) -> datetime:
    """Validate strict RFC 3339 UTC second precision and return the instant."""

    if not _UTC_SECOND_RE.fullmatch(value):
        raise ValueError("timestamp must be RFC 3339 UTC seconds with a Z suffix")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError("timestamp is not a valid calendar instant") from exc
    return parsed.replace(tzinfo=UTC)


def utc_now_seconds(now: datetime | None = None) -> str:
    """Format a real instant as RFC 3339 UTC with no fractional seconds."""

    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        raise ValueError("a timezone-aware datetime is required")
    return current.astimezone(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_event_json(payload: str | bytes) -> UsageEvent:
    """Parse one event through the same discriminated schema used by the server."""

    return _EVENT_ADAPTER.validate_json(payload)


def canonical_event_json(event: UsageEvent) -> str:
    """Return deterministic compact JSON suitable for the local outbox."""

    value = event.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def event_schema() -> dict[str, object]:
    """Return the draft JSON Schema for one lifecycle event."""

    return _EVENT_ADAPTER.json_schema()


def batch_schema() -> dict[str, object]:
    """Return the draft JSON Schema for the batch envelope."""

    return UsageBatch.model_json_schema()


def order_run_events(events: list[UsageEvent]) -> list[UsageEvent]:
    """Order a single Run by sequence, never by client timestamp."""

    run_ids = {event.telemetry_run_id for event in events}
    if len(run_ids) > 1:
        raise ValueError("events from different Research Runs cannot be ordered together")
    return sorted(events, key=lambda event: event.event_sequence)
