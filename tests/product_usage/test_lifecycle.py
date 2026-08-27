from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from praxist.product_usage.lifecycle import (
    LifecycleInvariantError,
    PeerStatusSummary,
    RunTelemetryContext,
    event_belongs_to_context,
    snapshot_context,
)


def peer_status(generation: int = 0) -> PeerStatusSummary:
    return PeerStatusSummary(
        generation_ordinal=generation,
        planned_peer_count=3,
        peer_planned_count=0,
        peer_running_count=2,
        peer_completed_count=0,
        peer_cancelled_count=0,
        peer_failed_count=1,
        peer_unknown_count=0,
    )


def test_context_generates_run_scoped_ids_and_sequences() -> None:
    context = RunTelemetryContext(
        "0.2.0",
        environment_id=uuid4(),
        clock=lambda: "2026-07-30T02:03:04Z",
    )

    started = context.run_started(peer_status())
    finished = context.generation_finished(peer_status())
    run_finished = context.run_finished(12, duration_capped=False)

    assert started.telemetry_run_id == finished.telemetry_run_id
    assert finished.telemetry_run_id == run_finished.telemetry_run_id
    assert len({started.event_id, finished.event_id, run_finished.event_id}) == 3
    assert [started.event_sequence, finished.event_sequence, run_finished.event_sequence] == [
        1,
        2,
        3,
    ]


def test_new_runs_never_reuse_telemetry_id() -> None:
    environment_id = uuid4()
    first = RunTelemetryContext("0.2.0", environment_id=environment_id)
    second = RunTelemetryContext("0.2.0", environment_id=environment_id)
    assert first.telemetry_run_id != second.telemetry_run_id


def test_context_strips_local_build_metadata_from_wire_events() -> None:
    context = RunTelemetryContext(
        "0.2.0.dev1+linux.x86-64",
        environment_id=uuid4(),
    )

    assert context.run_started(peer_status()).praxist_version == "0.2.0.dev1"


def test_context_rejects_noncanonical_versions() -> None:
    with pytest.raises(ValueError, match="canonical public version"):
        RunTelemetryContext("01.2.0.dev1", environment_id=uuid4())


def test_environment_identifier_is_stable_across_distinct_runs() -> None:
    environment_id = uuid4()
    first = RunTelemetryContext("0.2.0", environment_id=environment_id)
    second = RunTelemetryContext("0.2.0", environment_id=environment_id)

    first_event = first.run_started(peer_status())
    second_event = second.run_started(peer_status())

    assert first_event.environment_id == second_event.environment_id == environment_id
    assert first_event.telemetry_run_id != second_event.telemetry_run_id


def test_event_context_membership_requires_environment_and_run_ids() -> None:
    environment_id = uuid4()
    context = RunTelemetryContext("0.2.0", environment_id=environment_id)
    event = context.run_started(peer_status())
    other_environment = RunTelemetryContext.resume(
        "0.2.0",
        environment_id=uuid4(),
        telemetry_run_id=context.telemetry_run_id,
        next_sequence=context.next_sequence,
        run_started_emitted=True,
    )

    assert event_belongs_to_context(event, context)
    assert not event_belongs_to_context(event, other_environment)


def test_run_started_is_not_repeated_on_resume() -> None:
    resumed = RunTelemetryContext.resume(
        "0.2.0",
        environment_id=uuid4(),
        telemetry_run_id=uuid4(),
        next_sequence=4,
        run_started_emitted=True,
    )

    with pytest.raises(LifecycleInvariantError, match="at most once"):
        resumed.run_started(peer_status())


def test_generation_finished_is_emitted_once_per_generation() -> None:
    context = RunTelemetryContext("0.2.0", environment_id=uuid4())
    context.generation_finished(peer_status(1))

    with pytest.raises(LifecycleInvariantError, match="already emitted"):
        context.generation_finished(peer_status(1))


def test_lifecycle_state_rejects_incoherent_counts_and_identifiers() -> None:
    with pytest.raises(ValueError, match="sum to planned_peer_count"):
        PeerStatusSummary(
            generation_ordinal=0,
            planned_peer_count=2,
            peer_planned_count=0,
            peer_running_count=0,
            peer_completed_count=1,
            peer_cancelled_count=0,
            peer_failed_count=0,
            peer_unknown_count=0,
        )

    non_v4 = UUID(int=0)
    with pytest.raises(ValueError, match="environment_id"):
        RunTelemetryContext("0.2.0", environment_id=non_v4)
    with pytest.raises(ValueError, match="environment_id"):
        RunTelemetryContext.resume(
            "0.2.0",
            environment_id=non_v4,
            telemetry_run_id=uuid4(),
            next_sequence=1,
            run_started_emitted=False,
        )
    with pytest.raises(ValueError, match="telemetry_run_id"):
        RunTelemetryContext.resume(
            "0.2.0",
            environment_id=uuid4(),
            telemetry_run_id=non_v4,
            next_sequence=1,
            run_started_emitted=False,
        )
    with pytest.raises(ValueError, match="next_sequence"):
        RunTelemetryContext.resume(
            "0.2.0",
            environment_id=uuid4(),
            telemetry_run_id=uuid4(),
            next_sequence=0,
            run_started_emitted=False,
        )


def test_context_snapshot_exposes_only_resume_identity_and_sequence() -> None:
    context = RunTelemetryContext("0.2.0", environment_id=uuid4())

    assert snapshot_context(context) == (context.telemetry_run_id, 1)
