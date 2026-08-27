from __future__ import annotations

import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from praxist.product_usage.batching import encode_next_batch
from praxist.product_usage.collector import (
    CollectorCore,
    IngestionDisabledError,
    MemoryEventStore,
)
from praxist.product_usage.outbox import QueuedEvent
from praxist.product_usage.protocol import canonical_event_json
from tests.helpers.product_usage import make_event


def row_for_event() -> QueuedEvent:
    event = make_event(event_id=uuid4())
    payload = canonical_event_json(event)
    return QueuedEvent(
        event_id=str(event.event_id),
        grant_id="00000000-0000-4000-8000-000000000001",
        payload=payload,
        payload_size=len(payload.encode()),
        created_at_epoch=1,
        expires_at_epoch=2,
    )


def test_collector_server_stamps_and_deduplicates() -> None:
    store = MemoryEventStore()
    collector = CollectorCore(store, clock=lambda: "2026-07-30T03:04:05Z")
    batch = encode_next_batch([row_for_event()])
    assert batch is not None

    first = collector.ingest(batch.body)
    second = collector.ingest(batch.body)

    assert first.accepted == 1
    assert first.duplicates == 0
    assert second.accepted == 0
    assert second.duplicates == 1
    stored = next(iter(store.events.values()))
    assert stored.received_at == "2026-07-30T03:04:05Z"


def test_client_cannot_supply_received_at() -> None:
    row = row_for_event()
    payload = json.loads(row.payload)
    payload["received_at"] = "2026-07-30T03:04:05Z"
    body = json.dumps({"events": [payload]}).encode()

    with pytest.raises(ValidationError):
        CollectorCore(MemoryEventStore()).ingest(body)


def test_unknown_event_field_is_rejected() -> None:
    row = row_for_event()
    payload = json.loads(row.payload)
    payload["path"] = "/Users/alice/private"
    body = json.dumps({"events": [payload]}).encode()

    with pytest.raises(ValidationError):
        CollectorCore(MemoryEventStore()).ingest(body)


def test_kill_switch_rejects_before_parsing() -> None:
    collector = CollectorCore(MemoryEventStore(), enabled=lambda: False)

    with pytest.raises(IngestionDisabledError):
        collector.ingest(b"not-json")
