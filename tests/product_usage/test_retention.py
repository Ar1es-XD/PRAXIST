from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from praxist.product_usage import postgres, retention
from praxist.product_usage.retention import (
    MAX_RETENTION_INTERVAL_SECONDS,
    RetentionWorker,
    retention_cutoffs,
    run_retention_schedule,
    write_retention_health,
)


def test_confirmed_retention_cutoffs() -> None:
    now = datetime(2026, 7, 30, 12, 13, 14, tzinfo=UTC)
    cutoffs = retention_cutoffs(now)

    assert cutoffs.raw_received_before == "2026-02-01T12:13:14Z"


def test_worker_applies_cutoffs_without_choosing_database_layout() -> None:
    calls: list[tuple[str, str]] = []

    class Backend:
        def delete_raw_events_before(self, received_at: str) -> int:
            calls.append(("raw", received_at))
            return 1

    result = RetentionWorker(Backend()).run_once(datetime(2026, 7, 30, 12, 13, 14, tzinfo=UTC))

    assert result.raw_events_deleted == 1
    assert [name for name, _ in calls] == ["raw"]


def test_periodic_retention_runs_immediately_before_waiting() -> None:
    runs = 0

    class Worker:
        def run_once(self):
            nonlocal runs
            runs += 1

    run_retention_schedule(Worker(), interval_seconds=60, wait=lambda _seconds: True)

    assert runs == 1


def test_initial_retention_failure_exits_for_container_health() -> None:
    class Worker:
        def run_once(self) -> None:
            raise RuntimeError("database unavailable")

    with pytest.raises(RuntimeError, match="database unavailable"):
        run_retention_schedule(
            Worker(),  # type: ignore[arg-type]
            interval_seconds=60,
            wait=lambda _seconds: True,
        )


def test_later_retention_failure_exits_for_container_restart() -> None:
    attempts = 0
    successes = 0

    class Worker:
        def run_once(self) -> None:
            nonlocal attempts
            attempts += 1
            if attempts > 1:
                raise RuntimeError("temporary failure")

    def record_success() -> None:
        nonlocal successes
        successes += 1

    with pytest.raises(RuntimeError, match="temporary failure"):
        run_retention_schedule(
            Worker(),  # type: ignore[arg-type]
            interval_seconds=60,
            wait=lambda _seconds: False,
            on_success=record_success,
        )

    assert attempts == 2
    assert successes == 1


def test_retention_health_marker_is_atomically_written(tmp_path: Path) -> None:
    health_file = tmp_path / "retention-health"

    write_retention_health(health_file)

    assert health_file.read_text(encoding="utf-8").endswith("Z\n")
    assert not (tmp_path / ".retention-health.tmp").exists()


def test_retention_schedule_rejects_an_interval_that_breaks_the_180_day_cap() -> None:
    class Worker:
        def run_once(self) -> None:
            raise AssertionError("invalid schedules must fail before running")

    with pytest.raises(ValueError, match="at most"):
        run_retention_schedule(
            Worker(),  # type: ignore[arg-type]
            interval_seconds=MAX_RETENTION_INTERVAL_SECONDS + 1,
            wait=lambda _seconds: True,
        )


def test_retention_rejects_naive_time_and_nonpositive_schedule() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        retention_cutoffs(datetime(2026, 7, 30, 12, 13, 14))

    class Worker:
        def run_once(self) -> None:
            raise AssertionError("invalid schedules must fail before running")

    with pytest.raises(ValueError, match="positive"):
        run_retention_schedule(
            Worker(),  # type: ignore[arg-type]
            interval_seconds=0,
            wait=lambda _seconds: True,
        )


def test_retention_main_requires_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(SystemExit):
        retention.main(["--interval-seconds", "60"])


def test_retention_main_runs_once_marks_health_and_disposes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    health_file = tmp_path / "health"

    class Store:
        disposed = False

        def delete_raw_events_before(self, _received_at: str) -> int:
            return 0

        def dispose(self) -> None:
            self.disposed = True

    store = Store()
    handlers: list[object] = []
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://local/test")
    monkeypatch.setattr(postgres.PostgresEventStore, "from_url", lambda _url: store)
    monkeypatch.setattr(
        retention.signal, "signal", lambda _signal, handler: handlers.append(handler)
    )
    monkeypatch.setattr(
        retention,
        "run_retention_schedule",
        lambda _worker, *, interval_seconds, wait, on_success: on_success(),
    )

    assert retention.main(["--interval-seconds", "60", "--health-file", str(health_file)]) == 0
    assert health_file.exists()
    assert store.disposed
    assert len(handlers) == 2
