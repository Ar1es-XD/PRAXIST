from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path
from uuid import uuid4

import pytest

import praxist.product_usage.outbox as outbox_module
from praxist.product_usage.outbox import OUTBOX_RETENTION_SECONDS, Outbox
from tests.helpers.product_usage import make_event

GRANT_ID = "00000000-0000-4000-8000-000000000001"


def test_event_id_is_idempotent(tmp_path: Path) -> None:
    outbox = Outbox._at_path_for_tests(tmp_path / "outbox.sqlite3")
    event = make_event()

    assert outbox.enqueue(event, grant_id=GRANT_ID)
    assert not outbox.enqueue(event, grant_id=GRANT_ID)
    assert outbox.count() == 1


def test_empty_fetch_and_logical_size_are_bounded(tmp_path: Path) -> None:
    outbox = Outbox._at_path_for_tests(tmp_path / "outbox.sqlite3")

    assert outbox.fetch_oldest(grant_id=GRANT_ID, limit=0) == []
    assert outbox.logical_payload_bytes() == 0


@pytest.mark.parametrize("grant_id", ["", "x" * 81, "non-ascii-é"])
def test_grant_identity_is_nonempty_bounded_ascii(tmp_path: Path, grant_id: str) -> None:
    outbox = Outbox._at_path_for_tests(tmp_path / "outbox.sqlite3")

    with pytest.raises(ValueError, match="bounded ASCII"):
        outbox.fetch_oldest(grant_id=grant_id)


def test_single_event_must_fit_request_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outbox = Outbox._at_path_for_tests(tmp_path / "outbox.sqlite3")
    monkeypatch.setattr(outbox_module, "MAX_REQUEST_BYTES", 1)

    with pytest.raises(ValueError, match="maximum request body"):
        outbox.enqueue(make_event(), grant_id=GRANT_ID)


def test_expired_events_are_deleted(tmp_path: Path) -> None:
    now = 100
    outbox = Outbox._at_path_for_tests(
        tmp_path / "outbox.sqlite3",
        clock=lambda: now,
    )
    assert outbox.enqueue(make_event(), grant_id=GRANT_ID)

    now += OUTBOX_RETENTION_SECONDS
    assert outbox.fetch_oldest(grant_id=GRANT_ID) == []


def test_count_limit_evicts_oldest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(outbox_module, "MAX_OUTBOX_EVENTS", 2)
    now = 100
    outbox = Outbox._at_path_for_tests(
        tmp_path / "outbox.sqlite3",
        clock=lambda: now,
    )
    first = make_event(event_id=uuid4())
    assert outbox.enqueue(first, grant_id=GRANT_ID)
    now += 1
    second = make_event(event_id=uuid4())
    assert outbox.enqueue(second, grant_id=GRANT_ID)
    now += 1
    third = make_event(event_id=uuid4())
    assert outbox.enqueue(third, grant_id=GRANT_ID)

    ids = [row.event_id for row in outbox.fetch_oldest(grant_id=GRANT_ID)]
    assert ids == [str(second.event_id), str(third.event_id)]


def test_size_limit_evicts_oldest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = make_event(event_id=uuid4())
    second = make_event(event_id=uuid4())
    approximate_one_event_size = len(
        __import__("praxist.product_usage.protocol", fromlist=["canonical_event_json"])
        .canonical_event_json(first)
        .encode()
    )
    monkeypatch.setattr(outbox_module, "MAX_OUTBOX_BYTES", approximate_one_event_size + 10)
    now = 100
    outbox = Outbox._at_path_for_tests(
        tmp_path / "outbox.sqlite3",
        clock=lambda: now,
    )
    assert outbox.enqueue(first, grant_id=GRANT_ID)
    now += 1
    assert outbox.enqueue(second, grant_id=GRANT_ID)

    assert [row.event_id for row in outbox.fetch_oldest(grant_id=GRANT_ID)] == [
        str(second.event_id)
    ]


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits do not represent Windows ACLs")
def test_outbox_permissions_are_user_only(tmp_path: Path) -> None:
    outbox = Outbox._at_path_for_tests(tmp_path / "private" / "outbox.sqlite3")
    outbox.enqueue(make_event(), grant_id=GRANT_ID)

    assert stat.S_IMODE(outbox.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(outbox.path.stat().st_mode) == 0o600


def test_acknowledge_only_deletes_named_events(tmp_path: Path) -> None:
    outbox = Outbox._at_path_for_tests(tmp_path / "outbox.sqlite3")
    first = make_event(event_id=uuid4())
    second = make_event(event_id=uuid4())
    outbox.enqueue(first, grant_id=GRANT_ID)
    outbox.enqueue(second, grant_id=GRANT_ID)

    assert (
        outbox.acknowledge(
            {str(first.event_id), str(uuid4())},
            grant_id=GRANT_ID,
        )
        == 1
    )
    assert [row.event_id for row in outbox.fetch_oldest(grant_id=GRANT_ID)] == [
        str(second.event_id)
    ]


def test_grant_identity_scopes_fetch_acknowledgement_and_cleanup(tmp_path: Path) -> None:
    outbox = Outbox._at_path_for_tests(tmp_path / "outbox.sqlite3")
    prior_grant = "00000000-0000-4000-8000-000000000001"
    current_grant = "00000000-0000-4000-8000-000000000002"
    old_event = make_event(event_id=uuid4())
    current_event = make_event(event_id=uuid4())
    outbox.enqueue(old_event, grant_id=prior_grant)
    outbox.enqueue(current_event, grant_id=current_grant)

    assert [row.event_id for row in outbox.fetch_oldest(grant_id=current_grant)] == [
        str(current_event.event_id)
    ]
    assert outbox.acknowledge({str(old_event.event_id)}, grant_id=current_grant) == 0
    assert outbox.discard_other_grants(current_grant) == 1
    assert outbox.count() == 1


def test_v3_outbox_discards_rows_without_a_grant_identity(tmp_path: Path) -> None:
    path = tmp_path / "outbox.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE usage_outbox (
            event_id TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            payload_size INTEGER NOT NULL,
            created_at_epoch INTEGER NOT NULL,
            expires_at_epoch INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        "INSERT INTO usage_outbox VALUES (?, ?, ?, ?, ?)",
        ("legacy-event", "{}", 2, 1, 9999999999),
    )
    connection.commit()
    connection.close()

    outbox = Outbox._at_path_for_tests(path)

    assert outbox.count() == 0
