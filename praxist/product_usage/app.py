"""Pseudonymous HTTP boundary for the Praxist Usage Collector."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .batching import UnsupportedSchemaVersionError
from .collector import CollectorCore, IngestionDisabledError
from .ports import CollectorStore
from .postgres import DEFAULT_MAX_TABLE_BYTES, PostgresEventStore
from .protocol import MAX_REQUEST_BYTES


class RequestTooLargeError(ValueError):
    """Raised when a collector request crosses the fixed body limit."""


def create_app(
    store: CollectorStore | None = None,
    *,
    enabled: Callable[[], bool] | None = None,
) -> FastAPI:
    """Build the privacy-bounded collector application."""

    owned_store: PostgresEventStore | None = None
    if store is None:
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise RuntimeError("DATABASE_URL is required")
        owned_store = PostgresEventStore.from_url(
            database_url,
            max_table_bytes=_max_table_bytes(),
        )
        store = owned_store

    ingestion_enabled = enabled or _ingestion_enabled
    collector = CollectorCore(store, enabled=ingestion_enabled)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if owned_store is not None:
                owned_store.dispose()

    app = FastAPI(
        title="Praxist Usage Collector",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @app.get("/healthz", include_in_schema=False)
    def health() -> JSONResponse:
        try:
            store.ping()
        except Exception:  # noqa: BLE001 - health must not expose database details.
            return JSONResponse({"status": "unavailable"}, status_code=503)
        return JSONResponse({"status": "ok"})

    @app.post("/v1/events", include_in_schema=False)
    async def ingest(request: Request) -> JSONResponse:
        media_type = request.headers.get("content-type", "").partition(";")[0]
        if media_type.strip().lower() != "application/json":
            return JSONResponse({"error": "json_required"}, status_code=415)
        try:
            body = await _bounded_body(request)
            result = collector.ingest(body)
        except RequestTooLargeError:
            return JSONResponse({"error": "request_too_large"}, status_code=413)
        except IngestionDisabledError:
            return JSONResponse({"error": "ingestion_disabled"}, status_code=503)
        except UnsupportedSchemaVersionError:
            return JSONResponse({"error": "unsupported_schema_version"}, status_code=400)
        except (ValidationError, ValueError):
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        except Exception:  # noqa: BLE001 - never expose persistence or runtime details.
            return JSONResponse({"error": "temporarily_unavailable"}, status_code=503)
        return JSONResponse(
            {"accepted": result.accepted, "duplicates": result.duplicates},
            status_code=202,
        )

    return app


async def _bounded_body(request: Request) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_REQUEST_BYTES:
            raise RequestTooLargeError
        body.extend(chunk)
    return bytes(body)


def _ingestion_enabled() -> bool:
    value = os.environ.get("COLLECTOR_INGESTION_ENABLED", "false")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _max_table_bytes() -> int:
    raw = os.environ.get("COLLECTOR_MAX_TABLE_BYTES", str(DEFAULT_MAX_TABLE_BYTES))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError("COLLECTOR_MAX_TABLE_BYTES must be an integer") from exc
    if value < 1:
        raise RuntimeError("COLLECTOR_MAX_TABLE_BYTES must be positive")
    return value


def main() -> None:
    """Run the product-usage collector HTTP service."""

    uvicorn.run(
        "praxist.product_usage.app:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        access_log=False,
        server_header=False,
    )
