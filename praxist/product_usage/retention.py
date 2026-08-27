"""Bounded retention policy shared by scheduled storage adapters."""

from __future__ import annotations

import argparse
import os
import signal
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .ports import RetentionBackend
from .protocol import utc_now_seconds

MAX_RETENTION_DAYS = 180
RETENTION_SAFETY_MARGIN_DAYS = 1
MAX_RETENTION_INTERVAL_SECONDS = 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class RetentionCutoffs:
    """UTC retention boundaries applied by one worker pass."""

    raw_received_before: str


@dataclass(frozen=True, slots=True)
class RetentionResult:
    """Deletion counts produced by one retention pass."""

    raw_events_deleted: int


def retention_cutoffs(now: datetime | None = None) -> RetentionCutoffs:
    """Compute bounded UTC cutoffs without exceeding the retention maximum."""

    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        raise ValueError("a timezone-aware datetime is required")
    current = current.astimezone(UTC).replace(microsecond=0)
    # The worker may run just after an event crosses a cutoff. Entering the
    # deletion window one day early keeps a daily schedule within the declared
    # 180-day maximum instead of allowing nearly 181 days of retention.
    raw = current - timedelta(days=MAX_RETENTION_DAYS - RETENTION_SAFETY_MARGIN_DAYS)
    return RetentionCutoffs(
        raw_received_before=utc_now_seconds(raw),
    )


class RetentionWorker:
    """Apply confirmed cutoffs through a database-independent backend port."""

    def __init__(self, backend: RetentionBackend) -> None:
        self._backend = backend

    def run_once(self, now: datetime | None = None) -> RetentionResult:
        cutoffs = retention_cutoffs(now)
        return RetentionResult(
            raw_events_deleted=self._backend.delete_raw_events_before(cutoffs.raw_received_before),
        )


def run_retention_schedule(
    worker: RetentionWorker,
    *,
    interval_seconds: int,
    wait: Callable[[float], bool],
    on_success: Callable[[], None] | None = None,
) -> None:
    """Run retention immediately and then at a fixed interval until stopped."""

    if interval_seconds < 1:
        raise ValueError("retention interval must be positive")
    if interval_seconds > MAX_RETENTION_INTERVAL_SECONDS:
        raise ValueError("retention interval must be at most 86400 seconds")
    while True:
        worker.run_once()
        if on_success is not None:
            on_success()
        if wait(float(interval_seconds)):
            return


def write_retention_health(path: Path) -> None:
    """Atomically mark the latest successful retention pass."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(utc_now_seconds() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded product-usage retention worker."""

    parser = argparse.ArgumentParser(description="Run V2 product-usage retention")
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=int(
            os.environ.get(
                "RETENTION_INTERVAL_SECONDS",
                str(MAX_RETENTION_INTERVAL_SECONDS),
            )
        ),
    )
    parser.add_argument(
        "--health-file",
        type=Path,
        default=Path(value) if (value := os.environ.get("RETENTION_HEALTH_FILE")) else None,
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        parser.error("DATABASE_URL is required")

    from .postgres import PostgresEventStore

    store = PostgresEventStore.from_url(database_url)
    stopped = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stopped.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        run_retention_schedule(
            RetentionWorker(store),
            interval_seconds=args.interval_seconds,
            wait=stopped.wait,
            on_success=(
                (lambda: write_retention_health(args.health_file))
                if args.health_file is not None
                else None
            ),
        )
    finally:
        store.dispose()
    return 0
