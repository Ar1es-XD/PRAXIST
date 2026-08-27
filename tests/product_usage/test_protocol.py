from __future__ import annotations

import json
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from praxist.product_usage.protocol import (
    ErrorSummary,
    GenerationFinishedEvent,
    RunFinishedEvent,
    RunReconciledEvent,
    RunStartedEvent,
    UsageBatch,
    canonical_event_json,
    order_run_events,
    parse_event_json,
    utc_now_seconds,
    validate_utc_second,
)
from tests.helpers.product_usage import event_dict, make_event


def test_valid_peer_snapshot_round_trips() -> None:
    event = make_event("run_started")
    parsed = parse_event_json(canonical_event_json(event))

    assert isinstance(parsed, RunStartedEvent)
    assert parsed.planned_peer_count == 3


def test_v2_event_requires_a_random_environment_identifier() -> None:
    environment_id = uuid4()
    payload = event_dict(
        schema_version=2,
        consent_notice_version=3,
        environment_id=str(environment_id),
    )

    event = RunStartedEvent.model_validate_json(json.dumps(payload))

    assert event.environment_id == environment_id


def test_v1_event_is_rejected_without_compatibility_parsing() -> None:
    payload = event_dict(schema_version=1, consent_notice_version=1)

    with pytest.raises(ValidationError):
        RunStartedEvent.model_validate_json(json.dumps(payload))


def test_v2_event_rejects_v1_consent_notice() -> None:
    payload = event_dict(schema_version=2, consent_notice_version=1)

    with pytest.raises(ValidationError):
        RunStartedEvent.model_validate_json(json.dumps(payload))


def test_v2_event_accepts_previous_release_consent_notice() -> None:
    payload = event_dict(schema_version=2, consent_notice_version=2)

    event = RunStartedEvent.model_validate_json(json.dumps(payload))

    assert event.consent_notice_version == 2


def test_peer_counts_must_equal_planned_count() -> None:
    with pytest.raises(ValidationError, match="must sum"):
        RunStartedEvent.model_validate_json(json.dumps(event_dict(peer_running_count=3)))


def test_generation_ordinal_zero_is_valid() -> None:
    assert make_event(generation_ordinal=0).generation_ordinal == 0


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-07-30T02:03:04.123Z",
        "2026-07-30T02:03:04+00:00",
        "2026-07-30 02:03:04Z",
        "2026-13-30T02:03:04Z",
    ],
)
def test_timestamp_rejects_non_utc_second_values(timestamp: str) -> None:
    with pytest.raises(ValueError):
        validate_utc_second(timestamp)


def test_timestamp_formatter_removes_fractional_seconds() -> None:
    from datetime import UTC, datetime

    value = utc_now_seconds(datetime(2026, 7, 30, 2, 3, 4, 999999, tzinfo=UTC))
    assert value == "2026-07-30T02:03:04Z"


def test_identifiers_must_be_uuid4() -> None:
    with pytest.raises(ValidationError, match="UUID version 4"):
        RunStartedEvent.model_validate_json(
            json.dumps(event_dict(event_id=UUID("00000000-0000-1000-8000-000000000001")))
        )


def test_duration_cap_invariant() -> None:
    capped = RunFinishedEvent.model_validate_json(
        json.dumps(
            event_dict(
                "run_finished",
                active_duration_minutes=43_200,
                duration_capped=True,
            )
        )
    )
    assert capped.duration_capped

    with pytest.raises(ValidationError, match="43200"):
        RunFinishedEvent.model_validate_json(
            json.dumps(
                event_dict(
                    "run_finished",
                    active_duration_minutes=100,
                    duration_capped=True,
                )
            )
        )


def test_nullable_unreliable_duration_is_valid() -> None:
    event = RunFinishedEvent.model_validate_json(
        json.dumps(event_dict("run_finished", active_duration_minutes=None))
    )
    assert event.active_duration_minutes is None


def test_capped_error_count_requires_maximum() -> None:
    payload = {
        "scope": "peer",
        "stage": "launch",
        "error_type": "resource",
        "error_code": "PRX-CAPACITY",
        "reason_code": "capacity_unavailable",
        "count": 3,
        "count_capped": True,
    }
    with pytest.raises(ValidationError, match="65535"):
        ErrorSummary.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("event_type", "stage"),
    [
        ("run_started", "execution"),
        ("generation_finished", "setup"),
        ("run_finished", "execution"),
        ("run_reconciled", "finalization"),
    ],
)
def test_error_reporting_windows_are_closed(event_type: str, stage: str) -> None:
    summary = {
        "scope": "run",
        "stage": stage,
        "error_type": "unknown",
        "error_code": "PRX-UNKNOWN",
        "reason_code": "unknown",
        "count": 1,
        "count_capped": False,
    }
    payload = event_dict(event_type, error_summaries=[summary])
    model_by_type = {
        "run_started": RunStartedEvent,
        "generation_finished": GenerationFinishedEvent,
        "run_finished": RunFinishedEvent,
        "run_reconciled": RunReconciledEvent,
    }
    with pytest.raises(ValidationError):
        model_by_type[event_type].model_validate_json(json.dumps(payload))


def test_at_most_sixteen_error_groups() -> None:
    summary = {
        "scope": "peer",
        "stage": "launch",
        "error_type": "resource",
        "error_code": "PRX-CAPACITY",
        "reason_code": "capacity_unavailable",
        "count": 1,
        "count_capped": False,
    }
    with pytest.raises(ValidationError):
        RunStartedEvent.model_validate_json(json.dumps(event_dict(error_summaries=[summary] * 17)))


def test_event_shape_is_discriminated_and_closed() -> None:
    payload = event_dict("run_finished")
    payload["planned_peer_count"] = 2
    with pytest.raises(ValidationError):
        RunFinishedEvent.model_validate_json(json.dumps(payload))


def test_batch_is_nonempty_and_bounded() -> None:
    event = make_event()
    assert UsageBatch(events=(event,)).events == (event,)
    with pytest.raises(ValidationError):
        UsageBatch(events=())


def test_sequence_is_authoritative_over_timestamp() -> None:
    run_id = uuid4()
    later_sequence_earlier_clock = make_event(
        "run_finished",
        telemetry_run_id=run_id,
        sequence=2,
        occurred_at="2026-07-29T01:00:00Z",
    )
    earlier_sequence_later_clock = make_event(
        telemetry_run_id=run_id,
        sequence=1,
        occurred_at="2026-07-30T01:00:00Z",
    )

    ordered = order_run_events([later_sequence_earlier_clock, earlier_sequence_later_clock])
    assert [event.event_sequence for event in ordered] == [1, 2]


def test_canonical_json_has_no_fractional_or_unknown_fields() -> None:
    encoded = json.loads(canonical_event_json(make_event()))
    assert encoded["occurred_at"].endswith("Z")
    assert "received_at" not in encoded
