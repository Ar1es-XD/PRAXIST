#!/usr/bin/env python3
"""Run a loopback-only product-usage collector for local acceptance."""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from praxist.product_usage.collector import CollectorCore
from praxist.product_usage.protocol import MAX_REQUEST_BYTES, UsageEvent


class JsonlEventStore:
    """Append validated, server-stamped events once by Event ID."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._event_ids = self._load_event_ids()

    def insert_if_absent(self, event: UsageEvent, received_at: str) -> bool:
        event_id = str(event.event_id)
        with self._lock:
            if event_id in self._event_ids:
                return False
            record = {
                "received_at": received_at,
                "event": event.model_dump(mode="json"),
            }
            with self._path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
                stream.write("\n")
            self._event_ids.add(event_id)
            return True

    def _load_event_ids(self) -> set[str]:
        if not self._path.exists():
            return set()
        event_ids: set[str] = set()
        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                payload = json.loads(line)
                event_id = payload.get("event", {}).get("event_id")
                if event_id:
                    event_ids.add(str(event_id))
        except (OSError, AttributeError, json.JSONDecodeError):
            return set()
        return event_ids


class LocalCollectorServer(ThreadingHTTPServer):
    """HTTP server carrying only the SDK collector and output path."""

    def __init__(self, address: tuple[str, int], output: Path) -> None:
        super().__init__(address, LocalCollectorHandler)
        self.collector = CollectorCore(JsonlEventStore(output))


class LocalCollectorHandler(BaseHTTPRequestHandler):
    """Minimal local route without request, header, address, or body logging."""

    server: LocalCollectorServer

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        if self.path != "/healthz":
            self._json_response(404, {"error": "not_found"})
            return
        self._json_response(200, {"status": "ok"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        if self.path != "/v1/events":
            self._json_response(404, {"error": "not_found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json_response(400, {"error": "invalid_content_length"})
            return
        if content_length < 1 or content_length > MAX_REQUEST_BYTES:
            self._json_response(413, {"error": "request_too_large_or_empty"})
            return
        body = self.rfile.read(content_length)
        try:
            result = self.server.collector.ingest(body)
        except Exception:
            self._json_response(400, {"error": "invalid_usage_batch"})
            return
        self._json_response(
            202,
            {
                "accepted": result.accepted,
                "duplicates": result.duplicates,
            },
        )

    def log_message(self, _format: str, *args: Any) -> None:
        del args

    def _json_response(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    """Run the loopback V2 acceptance collector until interrupted."""

    parser = argparse.ArgumentParser(
        description="Run the loopback-only Praxist product-usage collector.",
    )
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "::1"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    server = LocalCollectorServer((args.host, args.port), args.output)
    host, port = server.server_address[:2]
    print(f"collector_url=http://{host}:{port}/v1/events", flush=True)
    print(f"collector_output={args.output}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
