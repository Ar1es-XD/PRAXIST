from __future__ import annotations

import json

import pytest

from praxist import __version__ as praxist_version
from praxist.product_usage import __version__ as product_usage_version
from praxist.product_usage.transport import (
    DEV_COLLECTOR_ENDPOINT,
    MAX_ACKNOWLEDGEMENT_BYTES,
    PRODUCTION_COLLECTOR_ENDPOINT,
    DevHttpBatchSender,
    ProductionHttpsBatchSender,
    _HttpBatchSender,
    _RejectRedirects,
    default_batch_sender,
)
from tests.helpers.product_usage import make_event


def test_development_sender_uses_the_fixed_development_collector() -> None:
    sender = DevHttpBatchSender()

    assert sender.endpoint == DEV_COLLECTOR_ENDPOINT
    assert sender.endpoint == "http://45.78.201.249/v1/events"


def test_development_build_defaults_only_to_fixed_development_sender() -> None:
    sender = default_batch_sender("0.2.1.dev0")

    assert isinstance(sender, DevHttpBatchSender)
    assert sender.endpoint == DEV_COLLECTOR_ENDPOINT


def test_released_build_uses_the_fixed_production_sender() -> None:
    sender = default_batch_sender("0.3.0")

    assert isinstance(sender, ProductionHttpsBatchSender)
    assert sender.endpoint == PRODUCTION_COLLECTOR_ENDPOINT
    assert sender.endpoint == "https://telemetry.theaiscientist.com/v1/events"


@pytest.mark.parametrize(
    "version",
    [
        "0.2.0-dev1",
        "dev0",
        "0.2.0.dev",
        "01.2.0.dev1",
        "1.02.dev1",
        "1.dev01",
    ],
)
def test_dev_substrings_do_not_enable_plain_http_sender(version: str) -> None:
    with pytest.raises(RuntimeError, match="canonical Praxist version"):
        default_batch_sender(version)


def test_release_with_dev_text_only_in_local_metadata_uses_production_sender() -> None:
    sender = default_batch_sender("0.3.0+build.dev1")

    assert isinstance(sender, ProductionHttpsBatchSender)
    assert sender.endpoint == PRODUCTION_COLLECTOR_ENDPOINT


def test_canonical_dev_release_with_local_metadata_uses_development_sender() -> None:
    assert isinstance(default_batch_sender("0.2.0.dev1+linux.x86-64"), DevHttpBatchSender)


def test_single_component_canonical_dev_release_uses_development_sender() -> None:
    assert isinstance(default_batch_sender("1.dev1"), DevHttpBatchSender)


def test_product_usage_release_gate_uses_the_praxist_artifact_version() -> None:
    assert product_usage_version == praxist_version


def test_sender_acknowledges_the_batch_when_server_counts_cover_every_event() -> None:
    event = make_event()
    body = json.dumps({"events": [event.model_dump(mode="json")]}).encode()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit: int) -> bytes:
            return b'{"accepted":0,"duplicates":1}'

    sender = DevHttpBatchSender(opener=lambda *_args, **_kwargs: Response())

    assert sender.send(body) == {str(event.event_id)}


def test_production_sender_rejects_plain_http_endpoints() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        ProductionHttpsBatchSender("http://collector.example/v1/events")


def test_default_sender_rejects_redirects_and_bounds_acknowledgement_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = make_event()
    body = json.dumps({"events": [event.model_dump(mode="json")]}).encode()
    observed: dict[str, object] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, limit: int) -> bytes:
            observed["read_limit"] = limit
            return b'{"accepted":1,"duplicates":0}'

    class Opener:
        def open(self, request, *, timeout: float):
            observed["request"] = request
            observed["timeout"] = timeout
            return Response()

    def build_opener(handler):
        observed["handler"] = handler
        return Opener()

    monkeypatch.setattr("urllib.request.build_opener", build_opener)

    sender = _HttpBatchSender("https://collector.example/v1/events")
    assert sender.send(body) == {str(event.event_id)}
    assert isinstance(observed["handler"], _RejectRedirects)
    assert observed["read_limit"] == MAX_ACKNOWLEDGEMENT_BYTES + 1
    assert observed["timeout"] == 2.0
    request = observed["request"]
    assert request.get_header("User-agent") == "Praxist-Product-Usage/2"


def test_sender_does_not_acknowledge_an_oversized_response() -> None:
    event = make_event()
    body = json.dumps({"events": [event.model_dump(mode="json")]}).encode()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, limit: int) -> bytes:
            return b"x" * limit

    sender = DevHttpBatchSender(opener=lambda *_args, **_kwargs: Response())

    assert sender.send(body) == set()


@pytest.mark.parametrize(
    "response_body",
    [
        b"[]",
        b'{"accepted":"1","duplicates":0}',
        b'{"accepted":-1,"duplicates":2}',
    ],
)
def test_sender_rejects_malformed_or_incoherent_acknowledgements(
    response_body: bytes,
) -> None:
    event = make_event()
    body = json.dumps({"events": [event.model_dump(mode="json")]}).encode()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit: int) -> bytes:
            return response_body

    sender = DevHttpBatchSender(opener=lambda *_args, **_kwargs: Response())

    assert sender.send(body) == set()


def test_redirects_are_rejected_and_valid_production_endpoint_is_retained() -> None:
    assert _RejectRedirects().redirect_request() is None
    sender = ProductionHttpsBatchSender("https://collector.example/v1/events")
    assert sender.endpoint == "https://collector.example/v1/events"
