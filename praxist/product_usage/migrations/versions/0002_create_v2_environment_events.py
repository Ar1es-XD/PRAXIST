"""Introduce V2 while retaining a rollback-compatible V1 archive."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Archive V1 storage and create the closed V2 event table."""

    op.drop_index("ix_raw_events_run_sequence", table_name="raw_events")
    op.drop_index("ix_raw_events_received_at", table_name="raw_events")
    op.rename_table("raw_events", "raw_events_v1_archive")
    op.execute(
        "ALTER TABLE raw_events_v1_archive "
        "RENAME CONSTRAINT raw_events_pkey TO raw_events_v1_archive_pkey"
    )
    op.create_table(
        "raw_events",
        sa.Column("event_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("environment_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("telemetry_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_sequence", sa.BigInteger(), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("schema_version", sa.SmallInteger(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.CheckConstraint(
            "event_type IN ('run_started', 'generation_finished', "
            "'run_finished', 'run_reconciled')",
            name="ck_raw_events_event_type",
        ),
        sa.CheckConstraint("schema_version = 2", name="ck_raw_events_schema_version"),
        sa.CheckConstraint("event_sequence >= 1", name="ck_raw_events_sequence_positive"),
        sa.PrimaryKeyConstraint("event_id"),
    )
    op.create_index("ix_raw_events_received_at", "raw_events", ["received_at"])
    op.create_index(
        "ix_raw_events_environment_received",
        "raw_events",
        ["environment_id", "received_at"],
    )
    op.create_index(
        "ix_raw_events_run_sequence",
        "raw_events",
        ["telemetry_run_id", "event_sequence"],
    )


def downgrade() -> None:
    """Remove V2 storage and restore the rollback-compatible V1 table."""

    op.drop_index("ix_raw_events_run_sequence", table_name="raw_events")
    op.drop_index("ix_raw_events_environment_received", table_name="raw_events")
    op.drop_index("ix_raw_events_received_at", table_name="raw_events")
    op.drop_table("raw_events")
    op.rename_table("raw_events_v1_archive", "raw_events")
    op.execute(
        "ALTER TABLE raw_events RENAME CONSTRAINT raw_events_v1_archive_pkey TO raw_events_pkey"
    )
    op.create_index("ix_raw_events_received_at", "raw_events", ["received_at"])
    op.create_index(
        "ix_raw_events_run_sequence",
        "raw_events",
        ["telemetry_run_id", "event_sequence"],
    )
