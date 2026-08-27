from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from praxist.product_usage import app as collector_app
from praxist.product_usage.app import _max_table_bytes, create_app
from praxist.product_usage.collector import MemoryEventStore
from praxist.product_usage.protocol import MAX_REQUEST_BYTES
from tests.helpers.product_usage import event_dict


def enabled_app(store: MemoryEventStore):
    return create_app(store, enabled=lambda: True)


def test_default_app_requires_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        create_app()


def test_owned_store_is_disposed_after_application_lifespan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Store(MemoryEventStore):
        disposed = False

        def dispose(self) -> None:
            self.disposed = True

    store = Store()
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://local/test")
    monkeypatch.setattr(
        collector_app.PostgresEventStore,
        "from_url",
        lambda _url, *, max_table_bytes: store,
    )

    with TestClient(create_app()) as client:
        assert client.get("/healthz").json() == {"status": "ok"}

    assert store.disposed


def test_collector_configuration_rejects_invalid_storage_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLLECTOR_MAX_TABLE_BYTES", "invalid")
    with pytest.raises(RuntimeError, match="integer"):
        _max_table_bytes()
    monkeypatch.setenv("COLLECTOR_MAX_TABLE_BYTES", "0")
    with pytest.raises(RuntimeError, match="positive"):
        _max_table_bytes()


def test_collector_main_uses_the_hardened_server_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        collector_app.uvicorn,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    collector_app.main()

    assert calls == [
        (
            ("praxist.product_usage.app:create_app",),
            {
                "factory": True,
                "host": "0.0.0.0",
                "port": 8000,
                "access_log": False,
                "server_header": False,
            },
        )
    ]


def test_ingest_accepts_and_deduplicates_valid_batch() -> None:
    client = TestClient(enabled_app(MemoryEventStore()))
    body = {"events": [event_dict()]}

    first = client.post("/v1/events", json=body)
    second = client.post("/v1/events", json=body)

    assert first.status_code == 202
    assert first.json() == {"accepted": 1, "duplicates": 0}
    assert second.status_code == 202
    assert second.json() == {"accepted": 0, "duplicates": 1}
    assert "set-cookie" not in first.headers


def test_ingest_rejects_non_json_without_parsing() -> None:
    client = TestClient(enabled_app(MemoryEventStore()))

    response = client.post(
        "/v1/events",
        content=b"not-json",
        headers={"content-type": "text/plain"},
    )

    assert response.status_code == 415
    assert response.json() == {"error": "json_required"}


def test_ingest_rejects_oversized_stream() -> None:
    client = TestClient(enabled_app(MemoryEventStore()))

    response = client.post(
        "/v1/events",
        content=b"x" * (MAX_REQUEST_BYTES + 1),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json() == {"error": "request_too_large"}


def test_ingest_returns_fixed_error_for_invalid_schema() -> None:
    client = TestClient(enabled_app(MemoryEventStore()))
    payload = event_dict()
    payload["path"] = "/private/path"

    response = client.post("/v1/events", json={"events": [payload]})

    assert response.status_code == 400
    assert response.json() == {"error": "invalid_request"}
    assert "/private/path" not in response.text


def test_v1_batch_receives_a_fixed_unsupported_version_error() -> None:
    client = TestClient(enabled_app(MemoryEventStore()))
    payload = event_dict()
    payload["schema_version"] = 1
    payload["consent_notice_version"] = 1

    response = client.post("/v1/events", json={"events": [payload]})

    assert response.status_code == 400
    assert response.json() == {"error": "unsupported_schema_version"}


def test_kill_switch_rejects_before_parsing() -> None:
    client = TestClient(create_app(MemoryEventStore(), enabled=lambda: False))

    response = client.post(
        "/v1/events",
        content=b"not-json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 503
    assert response.json() == {"error": "ingestion_disabled"}


def test_health_checks_store_without_exposing_failure() -> None:
    class UnavailableStore(MemoryEventStore):
        def ping(self) -> None:
            raise RuntimeError("database host is private")

    response = TestClient(enabled_app(UnavailableStore())).get("/healthz")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}


def test_ingest_hides_unexpected_store_failures() -> None:
    class BrokenStore(MemoryEventStore):
        def insert_if_absent(self, _event: object, _received_at: str) -> bool:
            raise RuntimeError("private database detail")

    response = TestClient(enabled_app(BrokenStore())).post(
        "/v1/events",
        json={"events": [event_dict()]},
    )

    assert response.status_code == 503
    assert response.json() == {"error": "temporarily_unavailable"}
    assert "private database detail" not in response.text
    assert "database host" not in response.text


def test_json_with_charset_is_accepted() -> None:
    client = TestClient(enabled_app(MemoryEventStore()))
    body = json.dumps({"events": [event_dict()]}).encode()

    response = client.post(
        "/v1/events",
        content=body,
        headers={"content-type": "application/json; charset=utf-8"},
    )

    assert response.status_code == 202


def test_collector_exposes_no_delete_method_or_environment_deletion_route() -> None:
    app = enabled_app(MemoryEventStore())

    assert all("DELETE" not in (route.methods or set()) for route in app.routes)
    response = TestClient(app).delete(
        "/v1/environments/00000000-0000-4000-8000-000000000000/events"
    )
    assert response.status_code in {404, 405}


def test_storage_quota_configuration_is_positive(
    monkeypatch,
) -> None:
    monkeypatch.setenv("COLLECTOR_MAX_TABLE_BYTES", "4096")
    assert _max_table_bytes() == 4096

    monkeypatch.setenv("COLLECTOR_MAX_TABLE_BYTES", "0")
    with pytest.raises(RuntimeError, match="positive"):
        _max_table_bytes()


def test_ingestion_defaults_to_disabled(monkeypatch) -> None:
    monkeypatch.delenv("COLLECTOR_INGESTION_ENABLED", raising=False)
    client = TestClient(create_app(MemoryEventStore()))

    response = client.post("/v1/events", json={"events": [event_dict()]})

    assert response.status_code == 503
    assert response.json() == {"error": "ingestion_disabled"}
