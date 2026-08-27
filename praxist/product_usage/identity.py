"""Persistent random identity for one local Praxist execution environment."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import UUID4

from .paths import environment_identity_path
from .protocol import StrictModel


class EnvironmentIdentityRecord(StrictModel):
    """Versioned persistent random identity for one local environment."""

    identity_schema_version: int = 1
    environment_id: UUID4


class EnvironmentIdentityStore:
    """Read or create a UUIDv4 that persists across Research Runs."""

    def __init__(self) -> None:
        self._path = environment_identity_path()

    @classmethod
    def _at_path_for_tests(cls, path: Path) -> EnvironmentIdentityStore:
        store = cls.__new__(cls)
        store._path = path
        return store

    @property
    def path(self) -> Path:
        return self._path

    def get_or_create(self) -> UUID:
        existing = self._read()
        if existing is not None:
            return UUID(str(existing.environment_id))
        environment_id = uuid4()
        self._write(EnvironmentIdentityRecord(environment_id=environment_id))
        persisted = self._read()
        if persisted is None:
            raise OSError("environment identity could not be persisted")
        return UUID(str(persisted.environment_id))

    def _read(self) -> EnvironmentIdentityRecord | None:
        try:
            return EnvironmentIdentityRecord.model_validate_json(self._path.read_bytes())
        except (OSError, ValueError):
            return None

    def _write(self, record: EnvironmentIdentityRecord) -> None:
        directory = self._path.parent
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        _chmod_user_only(directory, 0o700)
        encoded = (
            json.dumps(
                record.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        temporary: Path | None = None
        try:
            descriptor, name = tempfile.mkstemp(prefix=".environment-", dir=directory)
            temporary = Path(name)
            os.chmod(temporary, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            with suppress(FileExistsError):
                os.link(temporary, self._path)
            temporary.unlink()
            temporary = None
            _chmod_user_only(self._path, 0o600)
            _fsync_directory(directory)
        finally:
            if temporary is not None:
                with suppress(FileNotFoundError):
                    temporary.unlink()


def _chmod_user_only(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        if os.name != "nt":
            raise


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
