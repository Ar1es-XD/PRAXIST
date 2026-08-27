from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from praxist.product_usage.batching import encode_next_batch, parse_batch_bytes
from praxist.product_usage.outbox import Outbox
from praxist.product_usage.protocol import MAX_BATCH_EVENTS, MAX_REQUEST_BYTES
from tests.helpers.product_usage import make_event

GRANT_ID = "00000000-0000-4000-8000-000000000001"


def test_batch_contains_at_most_fifty_events(tmp_path: Path) -> None:
    outbox = Outbox._at_path_for_tests(tmp_path / "outbox.sqlite3")
    for _ in range(MAX_BATCH_EVENTS + 1):
        outbox.enqueue(make_event(event_id=uuid4()), grant_id=GRANT_ID)

    batch = encode_next_batch(outbox.fetch_oldest(grant_id=GRANT_ID))

    assert batch is not None
    assert len(parse_batch_bytes(batch.body).events) == MAX_BATCH_EVENTS
    assert len(batch.event_ids) == MAX_BATCH_EVENTS


def test_batch_never_exceeds_request_limit(tmp_path: Path) -> None:
    outbox = Outbox._at_path_for_tests(tmp_path / "outbox.sqlite3")
    summary = {
        "scope": "peer",
        "stage": "launch",
        "error_type": "runtime",
        "error_code": "PRX-RUNTIME",
        "reason_code": "runtime_error",
        "count": 65_535,
        "count_capped": True,
    }
    for _ in range(40):
        outbox.enqueue(
            make_event(
                event_id=uuid4(),
                error_summaries=[summary] * 16,
                error_summaries_truncated=True,
            ),
            grant_id=GRANT_ID,
        )

    batch = encode_next_batch(outbox.fetch_oldest(grant_id=GRANT_ID))

    assert batch is not None
    assert len(batch.body) <= MAX_REQUEST_BYTES
    assert 1 <= len(parse_batch_bytes(batch.body).events) <= MAX_BATCH_EVENTS


def test_parser_rejects_oversized_body_before_json() -> None:
    with pytest.raises(ValueError, match="32 KiB"):
        parse_batch_bytes(b" " * (MAX_REQUEST_BYTES + 1))
