"""Bounded JSON batch encoding shared with the collector."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .outbox import QueuedEvent
from .protocol import (
    MAX_BATCH_EVENTS,
    MAX_REQUEST_BYTES,
    UsageBatch,
    UsageEvent,
    parse_event_json,
)


class SingleEventTooLargeError(ValueError):
    """Raised when an outbox row cannot fit in a legal one-event request."""


class UnsupportedSchemaVersionError(ValueError):
    """Raised when a syntactically visible event declares a non-V2 schema."""


@dataclass(frozen=True, slots=True)
class EncodedBatch:
    """Canonical request bytes paired with the included outbox Event IDs."""

    body: bytes
    event_ids: frozenset[str]


def encode_next_batch(rows: list[QueuedEvent]) -> EncodedBatch | None:
    """Greedily encode the oldest rows without crossing either hard limit."""

    if not rows:
        return None

    accepted: list[UsageEvent] = []
    accepted_ids: list[str] = []
    accepted_body: bytes | None = None

    for row in rows[:MAX_BATCH_EVENTS]:
        event = parse_event_json(row.payload)
        candidate = UsageBatch(events=tuple([*accepted, event]))
        body = _canonical_batch_bytes(candidate)
        if len(body) > MAX_REQUEST_BYTES:
            if not accepted:
                raise SingleEventTooLargeError(
                    f"event {row.event_id} cannot fit in a legal request"
                )
            break
        accepted.append(event)
        accepted_ids.append(row.event_id)
        accepted_body = body

    assert accepted_body is not None
    return EncodedBatch(body=accepted_body, event_ids=frozenset(accepted_ids))


def parse_batch_bytes(body: bytes) -> UsageBatch:
    """Validate one bounded V2 request body and return its typed batch."""

    if len(body) > MAX_REQUEST_BYTES:
        raise ValueError("request body exceeds 32 KiB")
    try:
        raw = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raw = None
    if isinstance(raw, dict) and isinstance(raw.get("events"), list):
        versions = {
            event.get("schema_version")
            for event in raw["events"]
            if isinstance(event, dict) and "schema_version" in event
        }
        if any(version != 2 for version in versions):
            raise UnsupportedSchemaVersionError("only Usage Schema V2 is accepted")
    return UsageBatch.model_validate_json(body)


def _canonical_batch_bytes(batch: UsageBatch) -> bytes:
    payload = batch.model_dump(mode="json")
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
