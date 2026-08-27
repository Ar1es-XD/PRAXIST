"""Narrow ports shared by the product-usage client and collector adapters."""

from __future__ import annotations

from collections.abc import Collection
from typing import Protocol

from .protocol import UsageEvent


class BatchSender(Protocol):
    """Send one already-bounded JSON request and return acknowledged Event IDs."""

    def send(self, body: bytes) -> Collection[str]: ...


class EventStore(Protocol):
    """Persist a server-stamped event once by Event ID."""

    def insert_if_absent(self, event: UsageEvent, received_at: str) -> bool: ...


class CollectorStore(EventStore, Protocol):
    """Persistence used by the HTTP Collector, including its health probe."""

    def ping(self) -> None: ...


class RetentionBackend(Protocol):
    """Apply already-computed retention cutoffs to a concrete persistence layer."""

    def delete_raw_events_before(self, received_at: str) -> int: ...
