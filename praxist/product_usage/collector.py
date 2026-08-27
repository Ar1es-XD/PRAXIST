"""Transport-neutral Collector validation and idempotency core."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .batching import parse_batch_bytes
from .ports import EventStore
from .protocol import UsageEvent, utc_now_seconds, validate_utc_second


class IngestionDisabledError(RuntimeError):
    """Raised before parsing when the global ingestion kill switch is active."""


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Counts accepted and duplicate events for one collector request."""

    accepted: int
    duplicates: int


@dataclass(frozen=True, slots=True)
class StoredEvent:
    """One validated event paired with its collector receipt time."""

    event: UsageEvent
    received_at: str


class CollectorCore:
    """Validate bounded batches and persist each Event ID at most once."""

    def __init__(
        self,
        store: EventStore,
        *,
        clock: Callable[[], str] = utc_now_seconds,
        enabled: Callable[[], bool] = lambda: True,
    ) -> None:
        self._store = store
        self._clock = clock
        self._enabled = enabled

    def ingest(self, body: bytes) -> IngestResult:
        if not self._enabled():
            raise IngestionDisabledError("ingestion is disabled")
        batch = parse_batch_bytes(body)
        accepted = 0
        duplicates = 0
        for event in batch.events:
            received_at = self._clock()
            validate_utc_second(received_at)
            if self._store.insert_if_absent(event, received_at):
                accepted += 1
            else:
                duplicates += 1
        return IngestResult(accepted=accepted, duplicates=duplicates)


class MemoryEventStore:
    """Deterministic test/reference store; not a production persistence adapter."""

    def __init__(self) -> None:
        self.events: dict[str, StoredEvent] = {}

    def insert_if_absent(self, event: UsageEvent, received_at: str) -> bool:
        event_id = str(event.event_id)
        if event_id in self.events:
            return False
        self.events[event_id] = StoredEvent(event=event, received_at=received_at)
        return True

    def ping(self) -> None:
        return None
