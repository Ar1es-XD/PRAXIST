"""PostgreSQL persistence for the Collector."""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    Index,
    MetaData,
    SmallInteger,
    String,
    Table,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID, insert
from sqlalchemy.engine import Engine

from .collector import IngestionDisabledError
from .protocol import UsageEvent, validate_utc_second

DEFAULT_MAX_TABLE_BYTES = 2 * 1024 * 1024 * 1024

metadata = MetaData()

raw_events = Table(
    "raw_events",
    metadata,
    Column("event_id", UUID(as_uuid=True), primary_key=True),
    Column("environment_id", UUID(as_uuid=True), nullable=False),
    Column("telemetry_run_id", UUID(as_uuid=True), nullable=False),
    Column("event_sequence", BigInteger, nullable=False),
    Column("event_type", String(32), nullable=False),
    Column("schema_version", SmallInteger, nullable=False),
    Column("occurred_at", DateTime(timezone=True), nullable=False),
    Column("received_at", DateTime(timezone=True), nullable=False),
    Column("payload", JSONB, nullable=False),
    CheckConstraint("event_sequence >= 1", name="ck_raw_events_sequence_positive"),
    CheckConstraint("schema_version = 2", name="ck_raw_events_schema_version"),
    CheckConstraint(
        "event_type IN ('run_started', 'generation_finished', 'run_finished', 'run_reconciled')",
        name="ck_raw_events_event_type",
    ),
)

Index("ix_raw_events_received_at", raw_events.c.received_at)
Index("ix_raw_events_environment_received", raw_events.c.environment_id, raw_events.c.received_at)
Index(
    "ix_raw_events_run_sequence",
    raw_events.c.telemetry_run_id,
    raw_events.c.event_sequence,
)


class PostgresEventStore:
    """Persist validated events in PostgreSQL under a fixed storage quota."""

    def __init__(self, engine: Engine, *, max_table_bytes: int = DEFAULT_MAX_TABLE_BYTES) -> None:
        if max_table_bytes < 1:
            raise ValueError("max_table_bytes must be positive")
        self._engine = engine
        self._max_table_bytes = max_table_bytes

    @classmethod
    def from_url(
        cls,
        database_url: str,
        *,
        max_table_bytes: int = DEFAULT_MAX_TABLE_BYTES,
    ) -> PostgresEventStore:
        return cls(
            create_engine(database_url, pool_pre_ping=True),
            max_table_bytes=max_table_bytes,
        )

    def insert_if_absent(self, event: UsageEvent, received_at: str) -> bool:
        statement = (
            insert(raw_events)
            .values(
                event_id=event.event_id,
                environment_id=event.environment_id,
                telemetry_run_id=event.telemetry_run_id,
                event_sequence=event.event_sequence,
                event_type=event.event_type,
                schema_version=event.schema_version,
                occurred_at=validate_utc_second(event.occurred_at),
                received_at=validate_utc_second(received_at),
                payload=event.model_dump(mode="json"),
            )
            .on_conflict_do_nothing(index_elements=[raw_events.c.event_id])
            .returning(raw_events.c.event_id)
        )
        with self._engine.begin() as connection:
            table_bytes = int(
                connection.execute(
                    text(
                        "SELECT pg_total_relation_size('raw_events') + "
                        "COALESCE(pg_total_relation_size("
                        "to_regclass('raw_events_v1_archive')), 0)"
                    )
                ).scalar_one()
            )
            if table_bytes >= self._max_table_bytes:
                raise IngestionDisabledError("collector storage quota is exhausted")
            return connection.execute(statement).scalar_one_or_none() is not None

    def ping(self) -> None:
        with self._engine.connect() as connection:
            connection.execute(text("SELECT 1"))

    def delete_raw_events_before(self, received_at: str) -> int:
        cutoff = validate_utc_second(received_at)
        statement = raw_events.delete().where(raw_events.c.received_at < cutoff)
        with self._engine.begin() as connection:
            result = connection.execute(statement)
            archived = 0
            archive_exists = connection.execute(
                text("SELECT to_regclass('raw_events_v1_archive')")
            ).scalar_one_or_none()
            if archive_exists is not None:
                archive_result = connection.execute(
                    text("DELETE FROM raw_events_v1_archive WHERE received_at < :cutoff"),
                    {"cutoff": cutoff},
                )
                archived = max(0, int(archive_result.rowcount or 0))
        return max(0, int(result.rowcount or 0)) + archived

    def dispose(self) -> None:
        self._engine.dispose()
