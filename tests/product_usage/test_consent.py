from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from praxist.product_usage.client import UsageSdk
from praxist.product_usage.consent import (
    ConsentDecision,
    ConsentStatus,
    ConsentStore,
    parse_agent_reply,
)
from praxist.product_usage.outbox import Outbox
from tests.helpers.product_usage import make_event


@pytest.fixture
def consent_store(tmp_path: Path) -> ConsentStore:
    return ConsentStore._at_path_for_tests(tmp_path / "state" / "consent.json")


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("Yes", ConsentDecision.GRANTED),
        ("No", ConsentDecision.DENIED),
        ("Agree", ConsentDecision.GRANTED),
        ("Disagree", ConsentDecision.DENIED),
        (" Agree ", ConsentDecision.GRANTED),
    ],
)
def test_agent_parses_only_approved_keywords(
    reply: str,
    expected: ConsentDecision,
) -> None:
    assert parse_agent_reply(reply) is expected


@pytest.mark.parametrize(
    "reply",
    ["yes", "YES", "agree", "I agree", "同意", "不同意", "", "Agree to collect"],
)
def test_agent_does_not_infer_natural_language(reply: str) -> None:
    assert parse_agent_reply(reply) is None


def test_missing_corrupt_and_wrong_version_fail_closed(
    consent_store: ConsentStore,
) -> None:
    assert consent_store.status() is ConsentStatus.UNSET

    consent_store.path.parent.mkdir(parents=True)
    consent_store.path.write_text("{", encoding="utf-8")
    assert consent_store.status() is ConsentStatus.UNSET

    consent_store.path.write_text(
        json.dumps({"decision": "granted", "consent_notice_version": 1}),
        encoding="utf-8",
    )
    assert consent_store.status() is ConsentStatus.UNSET


def test_current_record_preserves_consent_audit_metadata_without_raw_reply(
    consent_store: ConsentStore,
) -> None:
    consent_store.write(ConsentDecision.GRANTED, source="agent", language="en")
    payload = json.loads(consent_store.path.read_text(encoding="utf-8"))

    assert payload["consent_notice_version"] == 3
    assert payload["decision"] == "granted"
    assert payload["decision_id"]
    assert payload["source"] == "agent"
    assert payload["language"] == "en"
    assert payload["decided_at"].endswith("Z")


def test_each_explicit_grant_has_a_distinct_durable_identity(
    consent_store: ConsentStore,
) -> None:
    consent_store.write(ConsentDecision.GRANTED)
    first_grant_id = consent_store.grant_id()
    consent_store.write(ConsentDecision.DENIED)
    consent_store.write(ConsentDecision.GRANTED)

    assert first_grant_id is not None
    assert consent_store.grant_id() is not None
    assert consent_store.grant_id() != first_grant_id


def test_legacy_grant_without_decision_id_has_stable_identity(
    consent_store: ConsentStore,
) -> None:
    consent_store.write(ConsentDecision.GRANTED)
    payload = json.loads(consent_store.path.read_text(encoding="utf-8"))
    payload.pop("decision_id")
    consent_store.path.write_text(json.dumps(payload), encoding="utf-8")

    first = consent_store.grant_id()
    second = ConsentStore._at_path_for_tests(consent_store.path).grant_id()

    assert first is not None
    assert first.startswith("legacy-")
    assert second == first


def test_reset_removes_current_and_recovery_records(
    consent_store: ConsentStore,
) -> None:
    consent_store.write(ConsentDecision.GRANTED)
    revoked = consent_store._revoked_path()
    revoked.write_text("recovery", encoding="utf-8")

    consent_store.reset()

    assert consent_store.status() is ConsentStatus.UNSET
    assert not consent_store.path.exists()
    assert not revoked.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits do not represent Windows ACLs")
def test_state_permissions_are_user_only(consent_store: ConsentStore) -> None:
    consent_store.write(ConsentDecision.GRANTED)

    assert stat.S_IMODE(consent_store.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(consent_store.path.stat().st_mode) == 0o600


def test_unset_and_denied_create_no_outbox(
    tmp_path: Path,
    consent_store: ConsentStore,
) -> None:
    created = 0

    def factory() -> Outbox:
        nonlocal created
        created += 1
        return Outbox._at_path_for_tests(tmp_path / "outbox.sqlite3")

    sdk = UsageSdk(consent_store, _outbox_factory=factory)
    assert not sdk.capture(make_event())
    assert created == 0

    assert sdk.record_direct_choice("No") is ConsentStatus.DENIED
    # Denial invokes best-effort cleanup, but capture itself must not reopen it.
    created_after_denial = created
    assert not sdk.capture(make_event())
    assert created == created_after_denial


def test_only_exact_direct_yes_grants(consent_store: ConsentStore) -> None:
    sdk = UsageSdk(consent_store)

    assert sdk.record_direct_choice("yes") is ConsentStatus.UNSET
    assert sdk.record_direct_choice("Yes") is ConsentStatus.GRANTED


def test_agent_raw_reply_is_not_persisted(consent_store: ConsentStore) -> None:
    sdk = UsageSdk(consent_store)
    assert sdk.record_agent_reply(" Agree ") is ConsentStatus.GRANTED

    stored = consent_store.path.read_text(encoding="utf-8")
    assert "Agree" not in stored
    assert "granted" in stored


def test_withdraw_sets_denied_and_deletes_outbox(
    tmp_path: Path,
    consent_store: ConsentStore,
) -> None:
    outbox = Outbox._at_path_for_tests(tmp_path / "outbox.sqlite3")
    sdk = UsageSdk(consent_store, _outbox_factory=lambda: outbox)
    assert sdk.record_direct_choice("Yes") is ConsentStatus.GRANTED
    assert sdk.capture(make_event())
    assert outbox.path.exists()

    assert sdk.withdraw()
    assert consent_store.status() is ConsentStatus.DENIED
    assert not outbox.path.exists()
    assert not Path(f"{outbox.path}-wal").exists()
    assert not Path(f"{outbox.path}-shm").exists()


def test_withdraw_waits_for_in_flight_capture_then_deletes_it(
    tmp_path: Path,
    consent_store: ConsentStore,
) -> None:
    capture_started = threading.Event()
    allow_capture_to_finish = threading.Event()
    withdrawal_finished = threading.Event()

    class BlockingOutbox:
        deleted = False

        def discard_other_grants(self, _grant_id: str) -> int:
            return 0

        def enqueue(self, _event: object, *, grant_id: str) -> bool:
            assert grant_id
            capture_started.set()
            assert allow_capture_to_finish.wait(timeout=2)
            return True

        def close_and_delete(self) -> None:
            self.deleted = True

        def close(self) -> None:
            pass

    outbox = BlockingOutbox()
    consent_store.write(ConsentDecision.GRANTED)
    capture_sdk = UsageSdk(consent_store, _outbox_factory=lambda: outbox)  # type: ignore[arg-type,return-value]
    withdraw_sdk = UsageSdk(
        ConsentStore._at_path_for_tests(consent_store.path),
        _outbox_factory=lambda: outbox,  # type: ignore[arg-type,return-value]
    )
    capture = threading.Thread(target=lambda: capture_sdk.capture(make_event()))
    withdrawal = threading.Thread(
        target=lambda: (withdraw_sdk.withdraw(), withdrawal_finished.set()),
    )

    capture.start()
    assert capture_started.wait(timeout=2)
    withdrawal.start()
    assert not withdrawal_finished.wait(timeout=0.1)
    allow_capture_to_finish.set()
    capture.join(timeout=2)
    withdrawal.join(timeout=2)

    assert withdrawal_finished.is_set()
    assert outbox.deleted
    assert consent_store.status() is ConsentStatus.DENIED


def test_failed_withdrawal_write_latches_denial_for_every_store_in_process(
    tmp_path: Path,
    consent_store: ConsentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consent_store.write(ConsentDecision.GRANTED)
    outbox = Outbox._at_path_for_tests(tmp_path / "outbox.sqlite3")
    sdk = UsageSdk(consent_store, _outbox_factory=lambda: outbox)
    assert sdk.capture(make_event())

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("read-only consent store")

    monkeypatch.setattr(consent_store, "_write_unlocked", fail_write)
    assert not sdk.withdraw()
    assert consent_store.status() is ConsentStatus.DENIED
    assert ConsentStore._at_path_for_tests(consent_store.path).status() is ConsentStatus.DENIED
    assert not sdk.capture(make_event())
    assert not consent_store.path.exists()
    assert not consent_store._revoked_path().exists()
    restarted_status = subprocess.check_output(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from praxist.product_usage.consent import ConsentStore; "
                "import sys; "
                "print(ConsentStore._at_path_for_tests(Path(sys.argv[1])).status().value)"
            ),
            str(consent_store.path),
        ],
        text=True,
    ).strip()
    assert restarted_status == "unset"
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from praxist.product_usage.consent import ConsentDecision, ConsentStore; "
                "import sys; "
                "ConsentStore._at_path_for_tests(Path(sys.argv[1])).write(ConsentDecision.GRANTED)"
            ),
            str(consent_store.path),
        ]
    )
    assert consent_store.status() is ConsentStatus.GRANTED


def test_withdrawal_reports_failed_outbox_cleanup_but_remains_denied(
    consent_store: ConsentStore,
) -> None:
    class UndeletableOutbox:
        def close_and_delete(self) -> None:
            raise OSError("busy")

    consent_store.write(ConsentDecision.GRANTED)
    sdk = UsageSdk(
        consent_store,
        _outbox_factory=lambda: UndeletableOutbox(),  # type: ignore[arg-type,return-value]
    )

    assert not sdk.withdraw()
    assert consent_store.status() is ConsentStatus.DENIED
    assert not sdk.capture(make_event())
