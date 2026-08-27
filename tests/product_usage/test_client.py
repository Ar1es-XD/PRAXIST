from __future__ import annotations

import os
import threading
from pathlib import Path
from uuid import uuid4

import pytest

from praxist.product_usage.client import UploadCoordinator, UsageSdk
from praxist.product_usage.consent import ConsentDecision, ConsentStatus, ConsentStore
from praxist.product_usage.outbox import Outbox
from tests.helpers.product_usage import make_event


class Sender:
    def __init__(self, *, fail: bool = False, acknowledge: bool = True) -> None:
        self.fail = fail
        self.acknowledge = acknowledge
        self.calls = 0

    def send(self, body: bytes) -> set[str]:
        self.calls += 1
        if self.fail:
            raise OSError("offline")
        from praxist.product_usage.batching import parse_batch_bytes

        events = parse_batch_bytes(body).events
        if not self.acknowledge:
            return set()
        return {str(event.event_id) for event in events}


def stores(tmp_path: Path) -> tuple[ConsentStore, Outbox]:
    consent = ConsentStore._at_path_for_tests(tmp_path / "consent.json")
    outbox = Outbox._at_path_for_tests(tmp_path / "outbox.sqlite3")
    return consent, outbox


def enqueue_for_current_grant(
    consent: ConsentStore,
    outbox: Outbox,
) -> None:
    grant_id = consent.grant_id()
    assert grant_id is not None
    assert outbox.enqueue(make_event(), grant_id=grant_id)


def test_granted_capture_and_upload(tmp_path: Path) -> None:
    consent, outbox = stores(tmp_path)
    consent.write(ConsentDecision.GRANTED)
    sdk = UsageSdk(consent, _outbox_factory=lambda: outbox)
    event = make_event(event_id=uuid4())
    assert sdk.capture(event)

    sender = Sender()
    uploaded = UploadCoordinator(consent, outbox, sender).flush_once()

    assert uploaded == 1
    assert sender.calls == 1
    assert outbox.count() == 0


def test_offline_sender_never_changes_capture_result(tmp_path: Path) -> None:
    consent, outbox = stores(tmp_path)
    consent.write(ConsentDecision.GRANTED)
    enqueue_for_current_grant(consent, outbox)
    sender = Sender(fail=True)

    assert UploadCoordinator(consent, outbox, sender).flush_once() == 0
    assert outbox.count() == 1


def test_no_upload_without_current_consent(tmp_path: Path) -> None:
    consent, outbox = stores(tmp_path)
    outbox.enqueue(
        make_event(),
        grant_id="00000000-0000-4000-8000-000000000001",
    )
    sender = Sender()

    assert UploadCoordinator(consent, outbox, sender).flush_once() == 0
    assert sender.calls == 0
    assert outbox.count() == 1


def test_unrecognized_acknowledgements_are_ignored(tmp_path: Path) -> None:
    consent, outbox = stores(tmp_path)
    consent.write(ConsentDecision.GRANTED)
    enqueue_for_current_grant(consent, outbox)

    class BadSender:
        def send(self, body: bytes) -> set[str]:
            del body
            return {str(uuid4())}

    assert UploadCoordinator(consent, outbox, BadSender()).flush_once() == 0
    assert outbox.count() == 1


def test_sdk_swallows_outbox_failure(tmp_path: Path) -> None:
    consent = ConsentStore._at_path_for_tests(tmp_path / "consent.json")
    consent.write(ConsentDecision.GRANTED)

    def broken_factory() -> Outbox:
        raise RuntimeError("collection is broken")

    sdk = UsageSdk(consent, _outbox_factory=broken_factory)
    assert not sdk.capture(make_event())


def test_sdk_accessors_fail_closed_when_private_stores_are_unreadable(tmp_path: Path) -> None:
    class BrokenConsent:
        @staticmethod
        def status() -> ConsentStatus:
            return ConsentStatus.GRANTED

        @staticmethod
        def grant_id() -> str:
            raise OSError("unreadable")

    class BrokenIdentity:
        @staticmethod
        def get_or_create() -> object:
            raise OSError("unwritable")

    sdk = UsageSdk(  # type: ignore[arg-type]
        BrokenConsent(),
        identity_store=BrokenIdentity(),  # type: ignore[arg-type]
    )

    assert sdk.consent_status is ConsentStatus.GRANTED
    assert sdk.consent_grant_id is None
    assert sdk.environment_id is None


def test_failed_consent_persistence_returns_unset(tmp_path: Path) -> None:
    consent = ConsentStore._at_path_for_tests(tmp_path / "consent.json")
    sdk = UsageSdk(consent)

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("read-only")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(consent, "write", fail_write)
        assert sdk.record_direct_choice("Yes") is ConsentStatus.UNSET


def test_capture_bound_to_an_old_grant_is_rejected_after_regrant(tmp_path: Path) -> None:
    consent, outbox = stores(tmp_path)
    consent.write(ConsentDecision.GRANTED)
    sdk = UsageSdk(consent, _outbox_factory=lambda: outbox)
    old_grant_id = sdk.consent_grant_id
    assert old_grant_id is not None

    assert sdk.withdraw()
    consent.write(ConsentDecision.GRANTED)

    assert not sdk.capture(make_event(), expected_grant_id=old_grant_id)
    assert sdk.capture(make_event())
    assert outbox.count() == 1


def test_withdraw_waits_for_the_bounded_in_flight_upload_boundary(
    tmp_path: Path,
) -> None:
    upload_consent, outbox = stores(tmp_path)
    withdraw_consent = ConsentStore._at_path_for_tests(tmp_path / "consent.json")
    upload_consent.write(ConsentDecision.GRANTED)
    enqueue_for_current_grant(upload_consent, outbox)
    outbox.close()
    send_started = threading.Event()
    allow_send_to_finish = threading.Event()
    withdrawal_finished = threading.Event()

    class BlockingSender(Sender):
        def send(self, body: bytes) -> set[str]:
            send_started.set()
            assert allow_send_to_finish.wait(timeout=2)
            return super().send(body)

    sender = BlockingSender()
    upload_results: list[int] = []
    upload = threading.Thread(
        target=lambda: upload_results.append(
            UploadCoordinator(upload_consent, outbox, sender).flush_once()
        ),
    )
    sdk = UsageSdk(
        withdraw_consent,
        _outbox_factory=lambda: Outbox._at_path_for_tests(outbox.path),
    )
    withdrawal = threading.Thread(
        target=lambda: (sdk.withdraw(), withdrawal_finished.set()),
    )

    upload.start()
    assert send_started.wait(timeout=2)
    withdrawal.start()
    assert not withdrawal_finished.wait(timeout=0.1)
    allow_send_to_finish.set()
    upload.join(timeout=2)
    withdrawal.join(timeout=2)

    assert withdrawal_finished.is_set()
    assert upload_results == [1]
    assert withdraw_consent.status().value == "denied"
    assert not outbox.path.exists()
    assert UploadCoordinator(upload_consent, outbox, sender).flush_once() == 0
    assert sender.calls == 1


def test_capture_remains_local_while_upload_waits_on_network(tmp_path: Path) -> None:
    upload_consent, outbox = stores(tmp_path)
    capture_consent = ConsentStore._at_path_for_tests(tmp_path / "consent.json")
    upload_consent.write(ConsentDecision.GRANTED)
    enqueue_for_current_grant(upload_consent, outbox)
    outbox.close()
    upload_outbox = Outbox._at_path_for_tests(outbox.path)
    capture_outbox = Outbox._at_path_for_tests(outbox.path)
    send_started = threading.Event()
    allow_send_to_finish = threading.Event()

    class BlockingSender(Sender):
        def send(self, body: bytes) -> set[str]:
            send_started.set()
            assert allow_send_to_finish.wait(timeout=10)
            return super().send(body)

    upload = threading.Thread(
        target=UploadCoordinator(
            upload_consent,
            upload_outbox,
            BlockingSender(),
        ).flush_once,
    )
    sdk = UsageSdk(capture_consent, _outbox_factory=lambda: capture_outbox)
    capture_result: list[bool] = []
    capture_finished = threading.Event()

    upload.start()
    assert send_started.wait(timeout=2)
    capture = threading.Thread(
        target=lambda: (
            capture_result.append(sdk.capture(make_event())),
            capture_finished.set(),
        ),
    )
    capture.start()
    try:
        assert capture_finished.wait(timeout=2)
        assert capture_result == [True]
        canonical = Outbox._at_path_for_tests(outbox.path)
        assert canonical.count() == 2
        canonical.close()
    finally:
        allow_send_to_finish.set()
        upload.join(timeout=2)
        capture.join(timeout=2)


def test_regrant_discards_rows_left_by_an_interrupted_withdrawal(tmp_path: Path) -> None:
    consent, outbox = stores(tmp_path)
    consent.write(ConsentDecision.GRANTED)
    enqueue_for_current_grant(consent, outbox)
    consent.write(ConsentDecision.DENIED)
    consent.write(ConsentDecision.GRANTED)
    sender = Sender()

    assert UploadCoordinator(consent, outbox, sender).flush_once() == 0
    assert sender.calls == 0
    assert outbox.count() == 0


def test_withdrawal_purge_and_regrant_share_one_process_boundary(tmp_path: Path) -> None:
    consent = ConsentStore._at_path_for_tests(tmp_path / "consent.json")
    consent.write(ConsentDecision.GRANTED)
    purge_started = threading.Event()
    release_purge = threading.Event()
    withdrawal_done = threading.Event()
    regrant_done = threading.Event()

    class BlockingOutbox:
        def close_and_delete(self) -> None:
            purge_started.set()
            assert release_purge.wait(timeout=2)

    sdk = UsageSdk(consent, _outbox_factory=BlockingOutbox)  # type: ignore[arg-type]
    withdrawal_result: list[bool] = []
    withdrawal = threading.Thread(
        target=lambda: (withdrawal_result.append(sdk.withdraw()), withdrawal_done.set()),
    )

    def regrant() -> None:
        ConsentStore._at_path_for_tests(consent.path).write(ConsentDecision.GRANTED)
        regrant_done.set()

    withdrawal.start()
    assert purge_started.wait(timeout=2)
    grant = threading.Thread(target=regrant)
    grant.start()
    assert not regrant_done.wait(timeout=0.1)
    release_purge.set()
    withdrawal.join(timeout=2)
    grant.join(timeout=2)

    assert withdrawal_done.is_set()
    assert withdrawal_result == [True]
    assert regrant_done.is_set()
    assert consent.status().value == "granted"


@pytest.mark.skipif(os.name == "nt", reason="Windows does not unlink an open SQLite database")
def test_open_outbox_reopens_after_withdrawal_replaces_database(tmp_path: Path) -> None:
    consent, stale_outbox = stores(tmp_path)
    consent.write(ConsentDecision.GRANTED)
    stale_sdk = UsageSdk(consent, _outbox_factory=lambda: stale_outbox)
    assert stale_sdk.capture(make_event())

    withdrawing_sdk = UsageSdk(
        ConsentStore._at_path_for_tests(consent.path),
        _outbox_factory=lambda: Outbox._at_path_for_tests(stale_outbox.path),
    )
    assert withdrawing_sdk.withdraw()
    consent.write(ConsentDecision.GRANTED)
    assert stale_sdk.capture(make_event())

    canonical = Outbox._at_path_for_tests(stale_outbox.path)
    assert canonical.count() == 1
    canonical.close()
