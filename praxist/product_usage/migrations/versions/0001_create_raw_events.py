"""Create the raw usage-event table."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the initial raw usage-event table and indexes."""

    op.create_table(
        "raw_events",
        sa.Column("event_id", postgresql.UUID(as_uuid=True), nullable=False),
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
        sa.CheckConstraint(
            "schema_version = 1",
            name="ck_raw_events_schema_version",
        ),
        sa.CheckConstraint(
            "event_sequence >= 1",
            name="ck_raw_events_sequence_positive",
        ),
        sa.PrimaryKeyConstraint("event_id"),
    )
    op.create_index("ix_raw_events_received_at", "raw_events", ["received_at"])
    op.create_index(
        "ix_raw_events_run_sequence",
        "raw_events",
        ["telemetry_run_id", "event_sequence"],
    )


def downgrade() -> None:
    """Remove the initial raw usage-event table and indexes."""

    op.drop_index("ix_raw_events_run_sequence", table_name="raw_events")
    op.drop_index("ix_raw_events_received_at", table_name="raw_events")
    op.drop_table("raw_events")
