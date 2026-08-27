"""Environment-specific HTTP transports for V2 product-usage batches."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from .batching import parse_batch_bytes
from .protocol import canonical_product_version

DEV_COLLECTOR_ENDPOINT = "http://45.78.201.249/v1/events"
PRODUCTION_COLLECTOR_ENDPOINT = "https://telemetry.theaiscientist.com/v1/events"
MAX_ACKNOWLEDGEMENT_BYTES = 4 * 1024


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _open_without_redirects(request: urllib.request.Request, *, timeout: float) -> Any:
    opener = urllib.request.build_opener(_RejectRedirects())
    return opener.open(request, timeout=timeout)


class _HttpBatchSender:
    def __init__(
        self,
        endpoint: str,
        *,
        opener: Callable[..., Any] | None = None,
        timeout_seconds: float = 2.0,
    ) -> None:
        self._endpoint = endpoint
        self._opener = opener or _open_without_redirects
        self._timeout_seconds = timeout_seconds

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def send(self, body: bytes) -> set[str]:
        batch = parse_batch_bytes(body)
        request = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Praxist-Product-Usage/2",
            },
            method="POST",
        )
        with self._opener(request, timeout=self._timeout_seconds) as response:
            raw_response = response.read(MAX_ACKNOWLEDGEMENT_BYTES + 1)
        if len(raw_response) > MAX_ACKNOWLEDGEMENT_BYTES:
            return set()
        payload = json.loads(raw_response)
        if not isinstance(payload, dict):
            return set()
        accepted = payload.get("accepted")
        duplicates = payload.get("duplicates")
        if type(accepted) is not int or type(duplicates) is not int:
            return set()
        if accepted < 0 or duplicates < 0 or accepted + duplicates != len(batch.events):
            return set()
        return {str(event.event_id) for event in batch.events}


class DevHttpBatchSender(_HttpBatchSender):
    """Send internal development data to the fixed development collector."""

    def __init__(
        self,
        *,
        opener: Callable[..., Any] | None = None,
        timeout_seconds: float = 2.0,
    ) -> None:
        super().__init__(
            DEV_COLLECTOR_ENDPOINT,
            opener=opener,
            timeout_seconds=timeout_seconds,
        )


class ProductionHttpsBatchSender(_HttpBatchSender):
    """Send released-client batches to an explicitly baked HTTPS endpoint."""

    def __init__(
        self,
        endpoint: str,
        *,
        opener: Callable[..., Any] | None = None,
        timeout_seconds: float = 2.0,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username is not None:
            raise ValueError("production product-usage endpoint must be an HTTPS URL")
        super().__init__(endpoint, opener=opener, timeout_seconds=timeout_seconds)


def default_batch_sender(
    praxist_version: str,
) -> DevHttpBatchSender | ProductionHttpsBatchSender:
    """Return the baked sender selected by the canonical Praxist build version."""

    return _default_batch_sender_for_version(praxist_version)


def _default_batch_sender_for_version(
    version: str,
) -> DevHttpBatchSender | ProductionHttpsBatchSender:
    try:
        public_version = canonical_product_version(version)
    except ValueError as exc:
        raise RuntimeError("product-usage transport requires a canonical Praxist version") from exc
    if ".dev" in public_version:
        return DevHttpBatchSender()
    return ProductionHttpsBatchSender(PRODUCTION_COLLECTOR_ENDPOINT)
