from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid4

from praxist.product_usage.protocol import (
    GenerationFinishedEvent,
    RunFinishedEvent,
    RunReconciledEvent,
    RunStartedEvent,
    UsageEvent,
)


def event_dict(
    event_type: str = "run_started",
    *,
    event_id: UUID | None = None,
    environment_id: UUID | None = None,
    telemetry_run_id: UUID | None = None,
    sequence: int = 1,
    **overrides: Any,
) -> dict[str, Any]:
    common: dict[str, Any] = {
        "schema_version": 2,
        "praxist_version": "0.2.0",
        "consent_notice_version": 3,
        "event_id": str(event_id or uuid4()),
        "environment_id": str(environment_id or uuid4()),
        "telemetry_run_id": str(telemetry_run_id or uuid4()),
        "event_sequence": sequence,
        "event_type": event_type,
        "occurred_at": "2026-07-30T02:03:04Z",
        "error_summaries": [],
        "error_summaries_truncated": False,
    }
    if event_type in {"run_started", "generation_finished"}:
        common.update(
            {
                "generation_ordinal": 0,
                "planned_peer_count": 3,
                "peer_planned_count": 0,
                "peer_running_count": 2,
                "peer_completed_count": 0,
                "peer_cancelled_count": 0,
                "peer_failed_count": 1,
                "peer_unknown_count": 0,
            }
        )
    else:
        common.update(
            {
                "active_duration_minutes": 12,
                "duration_capped": False,
            }
        )
    common.update(overrides)
    return common


def make_event(
    event_type: str = "run_started",
    *,
    event_id: UUID | None = None,
    environment_id: UUID | None = None,
    telemetry_run_id: UUID | None = None,
    sequence: int = 1,
    **overrides: Any,
) -> UsageEvent:
    payload = event_dict(
        event_type,
        event_id=event_id,
        environment_id=environment_id,
        telemetry_run_id=telemetry_run_id,
        sequence=sequence,
        **overrides,
    )
    model_by_type = {
        "run_started": RunStartedEvent,
        "generation_finished": GenerationFinishedEvent,
        "run_finished": RunFinishedEvent,
        "run_reconciled": RunReconciledEvent,
    }
    return model_by_type[event_type].model_validate_json(json.dumps(payload))
