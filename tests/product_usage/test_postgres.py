from __future__ import annotations

import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select, text

from praxist.product_usage import postgres
from praxist.product_usage.collector import IngestionDisabledError
from praxist.product_usage.postgres import PostgresEventStore, raw_events
from tests.helpers.product_usage import make_event


def test_store_configuration_health_and_disposal_use_the_engine_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        PostgresEventStore(object(), max_table_bytes=0)  # type: ignore[arg-type]

    class Result:
        pass

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, _statement: object) -> Result:
            return Result()

    class Engine:
        disposed = False

        def connect(self) -> Connection:
            return Connection()

        def dispose(self) -> None:
            self.disposed = True

    engine = Engine()
    monkeypatch.setattr(postgres, "create_engine", lambda *_args, **_kwargs: engine)

    store = PostgresEventStore.from_url("postgresql+psycopg://local/test")
    store.ping()
    store.dispose()

    assert engine.disposed


def test_delete_removes_current_and_rollback_archive_rows() -> None:
    class Result:
        def __init__(self, *, rowcount: int = 0, scalar: object = None) -> None:
            self.rowcount = rowcount
            self._scalar = scalar

        def scalar_one_or_none(self) -> object:
            return self._scalar

    class Connection:
        def execute(self, statement: object, *_args: object) -> Result:
            rendered = str(statement)
            if "SELECT to_regclass" in rendered:
                return Result(scalar="raw_events_v1_archive")
            if "DELETE FROM raw_events_v1_archive" in rendered:
                return Result(rowcount=3)
            return Result(rowcount=2)

    class Transaction:
        def __enter__(self) -> Connection:
            return Connection()

        def __exit__(self, *_args: object) -> None:
            return None

    class Engine:
        def begin(self) -> Transaction:
            return Transaction()

    store = PostgresEventStore(Engine())  # type: ignore[arg-type]

    assert store.delete_raw_events_before("2026-08-04T01:02:04Z") == 5


def test_postgres_store_inserts_once() -> None:
    database_url = os.environ.get("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not configured")

    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(raw_events.delete())
        connection.execute(text("DELETE FROM raw_events_v1_archive"))
        connection.execute(
            text(
                """
                INSERT INTO raw_events_v1_archive (
                    event_id, telemetry_run_id, event_sequence, event_type,
                    schema_version, occurred_at, received_at, payload
                ) VALUES (
                    :event_id, :run_id, 1, 'run_started', 1,
                    :occurred_at, :received_at, CAST(:payload AS JSONB)
                )
                """
            ),
            {
                "event_id": str(uuid4()),
                "run_id": str(uuid4()),
                "occurred_at": "2025-01-01T00:00:00Z",
                "received_at": "2025-01-01T00:00:00Z",
                "payload": "{}",
            },
        )

    event = make_event()
    with pytest.raises(IngestionDisabledError, match="storage quota"):
        PostgresEventStore(engine, max_table_bytes=1).insert_if_absent(
            event,
            "2026-08-04T01:02:02Z",
        )
    store = PostgresEventStore(engine)
    assert store.insert_if_absent(event, "2026-08-04T01:02:03Z") is True
    assert store.insert_if_absent(event, "2026-08-04T01:02:04Z") is False
    store.ping()

    with engine.connect() as connection:
        row = connection.execute(select(raw_events)).one()

    assert row.event_id == event.event_id
    assert row.environment_id == event.environment_id
    assert row.telemetry_run_id == event.telemetry_run_id
    assert row.received_at.isoformat() == "2026-08-04T01:02:03+00:00"
    assert row.payload["event_type"] == "run_started"

    assert store.delete_raw_events_before("2026-08-04T01:02:04Z") == 2
    with engine.connect() as connection:
        assert connection.execute(select(raw_events)).all() == []
        assert connection.execute(text("SELECT * FROM raw_events_v1_archive")).all() == []
    engine.dispose()
