"""Fixed per-OS-user paths; no environment or project override is supported."""

from __future__ import annotations

import os
import sys
from hashlib import sha256
from pathlib import Path


def consent_path() -> Path:
    """Return the fixed current-user consent-record path for this platform."""

    home = _user_home()
    if sys.platform == "darwin":
        return (
            home / "Library" / "Application Support" / "Praxist" / "product-usage" / "consent.json"
        )
    if os.name == "nt":
        return home / "AppData" / "Local" / "Praxist" / "product-usage" / "consent.json"
    return home / ".config" / "praxist" / "product-usage" / "consent.json"


def outbox_path() -> Path:
    """Return the fixed current-user SQLite outbox path for this platform."""

    home = _user_home()
    if sys.platform == "darwin":
        return (
            home
            / "Library"
            / "Application Support"
            / "Praxist"
            / "product-usage"
            / "outbox.sqlite3"
        )
    if os.name == "nt":
        return home / "AppData" / "Local" / "Praxist" / "product-usage" / "outbox.sqlite3"
    return home / ".local" / "share" / "praxist" / "product-usage" / "outbox.sqlite3"


def environment_identity_path() -> Path:
    """Return the fixed current-user environment-identity path."""

    home = _user_home()
    if sys.platform == "darwin":
        return (
            home
            / "Library"
            / "Application Support"
            / "Praxist"
            / "product-usage"
            / "environment.json"
        )
    if os.name == "nt":
        return home / "AppData" / "Local" / "Praxist" / "product-usage" / "environment.json"
    return home / ".local" / "share" / "praxist" / "product-usage" / "environment.json"


def run_state_path(run_dir: Path) -> Path:
    """Return private resume state keyed by the local canonical run path."""

    canonical = str(Path(run_dir).expanduser().resolve(strict=False)).encode(
        "utf-8",
        errors="surrogateescape",
    )
    return environment_identity_path().parent / "runs" / f"{sha256(canonical).hexdigest()}.json"


def _user_home() -> Path:
    if os.name == "nt":
        # Windows does not expose pwd; Path.home resolves the current account.
        return Path.home()
    import pwd

    return Path(pwd.getpwuid(os.getuid()).pw_dir)
