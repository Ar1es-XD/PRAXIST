from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from praxist.product_usage.client import UsageSdk
from praxist.product_usage.consent import ConsentDecision, ConsentStore
from praxist.product_usage.identity import EnvironmentIdentityStore


def test_environment_identity_is_stable_across_store_instances(tmp_path) -> None:
    path = tmp_path / "environment.json"

    first = EnvironmentIdentityStore._at_path_for_tests(path).get_or_create()
    second = EnvironmentIdentityStore._at_path_for_tests(path).get_or_create()

    assert first == second
    assert first.version == 4


def test_sdk_creates_environment_identity_only_after_v2_consent(tmp_path) -> None:
    consent = ConsentStore._at_path_for_tests(tmp_path / "consent.json")
    identity = EnvironmentIdentityStore._at_path_for_tests(tmp_path / "environment.json")
    sdk = UsageSdk(consent, identity_store=identity)

    assert sdk.environment_id is None
    assert not identity.path.exists()

    consent.write(ConsentDecision.GRANTED)

    assert sdk.environment_id is not None
    assert identity.path.exists()


def test_concurrent_first_use_converges_on_one_environment_identity(tmp_path) -> None:
    path = tmp_path / "environment.json"
    workers = 8
    barrier = threading.Barrier(workers)

    def create_identity() -> object:
        store = EnvironmentIdentityStore._at_path_for_tests(path)
        barrier.wait()
        return store.get_or_create()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        identities = list(pool.map(lambda _index: create_identity(), range(workers)))

    assert len(set(identities)) == 1


def test_identity_creation_fails_when_atomic_write_cannot_be_read_back(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = EnvironmentIdentityStore._at_path_for_tests(tmp_path / "environment.json")
    monkeypatch.setattr(store, "_read", lambda: None)
    monkeypatch.setattr(store, "_write", lambda _record: None)

    with pytest.raises(OSError, match="could not be persisted"):
        store.get_or_create()
