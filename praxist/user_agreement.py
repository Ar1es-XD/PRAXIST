"""Canonical Praxist legal terms and local acceptance record."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata, resources
from pathlib import Path
from typing import Literal, cast

USER_AGREEMENT_VERSION = "2026-08-28"
FAIR_SOURCE_LICENSE_VERSION = "1.0"
FAIR_SOURCE_LICENSE_DOCUMENT = "LICENSE.md"
USER_AGREEMENT_DOCUMENT = "legal/user-agreement.md"
PRODUCT_USAGE_NOTICE_DOCUMENT = "legal/product-usage-data-notice.md"


@dataclass(frozen=True)
class AgreementAcceptance:
    """A minimal local record of acceptance for one exact agreement."""

    agreement_version: str
    agreement_sha256: str
    accepted_at: str
    source: Literal["direct", "agent"]


def user_agreement_text() -> str:
    """Return every legal document covered by the first-use acceptance."""

    license_text = fair_source_license_text()
    body = _document_text(USER_AGREEMENT_DOCUMENT)
    appendix = product_usage_notice_text()
    return f"{license_text.rstrip()}\n\n---\n\n{body.rstrip()}\n\n---\n\n{appendix.rstrip()}"


def fair_source_license_text() -> str:
    """Return the canonical Fair Source License from the repository root."""

    packaged = resources.files("praxist").joinpath("resources/LICENSE.md")
    if packaged.is_file():
        return packaged.read_text(encoding="utf-8").strip()
    source = Path(__file__).resolve().parents[1] / FAIR_SOURCE_LICENSE_DOCUMENT
    if source.is_file():
        return source.read_text(encoding="utf-8").strip()
    distribution = metadata.distribution("praxist")
    for entry in distribution.files or ():
        if entry.as_posix().endswith(".dist-info/licenses/LICENSE.md"):
            installed = Path(str(distribution.locate_file(entry)))
            return installed.read_text(encoding="utf-8").strip()
    raise FileNotFoundError("installed Praxist distribution does not contain LICENSE.md")


def product_usage_notice_text() -> str:
    """Return the canonical optional product-usage data notice."""

    return _document_text(PRODUCT_USAGE_NOTICE_DOCUMENT).strip()


def user_agreement_sha256() -> str:
    """Return the digest that binds acceptance to all displayed legal terms."""

    return hashlib.sha256(user_agreement_text().encode("utf-8")).hexdigest()


def acceptance_path() -> Path:
    """Return the current user's Praxist Agreement acceptance path."""

    config_home = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return config_home / "praxist" / "user-agreement.json"


def current_acceptance(path: Path | None = None) -> AgreementAcceptance | None:
    """Return a valid acceptance for the current exact Agreement, if present."""

    target = path or acceptance_path()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        raw_source = payload["source"]
        if raw_source not in {"direct", "agent"}:
            return None
        record = AgreementAcceptance(
            agreement_version=str(payload["agreement_version"]),
            agreement_sha256=str(payload["agreement_sha256"]),
            accepted_at=str(payload["accepted_at"]),
            source=cast(Literal["direct", "agent"], raw_source),
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if record.agreement_version != USER_AGREEMENT_VERSION:
        return None
    if record.agreement_sha256 != user_agreement_sha256():
        return None
    return record


def record_acceptance(
    *,
    source: Literal["direct", "agent"],
    path: Path | None = None,
) -> AgreementAcceptance:
    """Atomically record an explicit acceptance for the current Agreement."""

    if source not in {"direct", "agent"}:
        raise ValueError("acceptance source must be direct or agent")
    target = path or acceptance_path()
    record = AgreementAcceptance(
        agreement_version=USER_AGREEMENT_VERSION,
        agreement_sha256=user_agreement_sha256(),
        accepted_at=datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        source=source,
    )
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _chmod(target.parent, 0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".user-agreement-",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(asdict(record), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        _chmod(target, 0o600)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return record


def _document_text(relative_path: str) -> str:
    packaged = resources.files("praxist").joinpath(f"resources/docs/{relative_path}")
    if packaged.is_file():
        return packaged.read_text(encoding="utf-8")
    source = Path(__file__).resolve().parents[1] / "docs" / relative_path
    return source.read_text(encoding="utf-8")


def _chmod(path: Path, mode: int) -> None:
    with suppress(OSError):
        path.chmod(mode)
