"""
Shared HTTP utilities for MCP tools.

Provides common HTTP functions used across tool modules:
- Async HTTP POST/GET with httpx fallback to requests
- Server URL resolution from environment variables
- Input validation helpers for security
"""

import os
import re
from pathlib import Path
from typing import Any

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# Default headers to bypass Cloudflare bot detection on RunPod proxies
DEFAULT_HEADERS = {
    "User-Agent": "curl/8.0.0",
    "Accept": "*/*",
}


def get_server_url() -> str:
    """Get the server URL from environment variable."""
    url = os.environ.get("ORCHESTRATOR_API_URL") or os.environ.get("SERVER_URL")
    if not url:
        raise ValueError("ORCHESTRATOR_API_URL or SERVER_URL environment variable is not set.")
    return url.rstrip("/")


async def async_http_post(
    url: str,
    json_data: dict[str, Any],
    timeout: int = 60,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Make an async HTTP POST request with httpx fallback to requests."""
    h = {**DEFAULT_HEADERS, "Content-Type": "application/json"}
    if headers:
        h.update(headers)

    if HAS_HTTPX:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=json_data, headers=h)
            if response.status_code >= 400:
                try:
                    body = response.json()
                    if "error" in body:
                        return body
                except Exception:
                    pass
                response.raise_for_status()
            return response.json()
    elif HAS_REQUESTS:
        response = requests.post(url, json=json_data, headers=h, timeout=timeout)
        if response.status_code >= 400:
            try:
                body = response.json()
                if "error" in body:
                    return body
            except Exception:
                pass
            response.raise_for_status()
        return response.json()
    else:
        raise ImportError("Either httpx or requests is required for HTTP operations")


async def async_http_get(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = 30,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Make an async HTTP GET request with httpx fallback to requests."""
    h = {**DEFAULT_HEADERS}
    if headers:
        h.update(headers)

    if HAS_HTTPX:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, params=params, headers=h)
            response.raise_for_status()
            return response.json()
    elif HAS_REQUESTS:
        response = requests.get(url, params=params, headers=h, timeout=timeout)
        response.raise_for_status()
        return response.json()
    else:
        raise ImportError("Either httpx or requests is required for HTTP operations")


def validate_safe_identifier(value: str, name: str) -> str:
    """Validate that an identifier is safe (no path traversal)."""
    if not value:
        raise ValueError(f"{name} is required")
    value = value.strip()
    if ".." in value or "/" in value or "\\" in value:
        raise ValueError(f"{name} contains invalid characters (path traversal)")
    if not re.match(r"^[a-zA-Z0-9_\-]+$", value):
        raise ValueError(
            f"{name} contains invalid characters (only alphanumeric, underscore, hyphen allowed)"
        )
    return value


def validate_safe_path(path: str, name: str, allowed_base: str = "/workspace") -> str:
    """Validate that a path is safe and within allowed base directory."""
    if not path:
        raise ValueError(f"{name} is required")
    path = path.strip()
    resolved = Path(path).resolve()
    allowed_resolved = Path(allowed_base).resolve()
    try:
        resolved.relative_to(allowed_resolved)
    except ValueError as err:
        raise ValueError(f"{name} must be within {allowed_base}. Got: {resolved}") from err
    return str(resolved)
