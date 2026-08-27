from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from praxist.product_usage import paths
from praxist.product_usage.paths import consent_path, environment_identity_path, outbox_path


@pytest.mark.skipif(os.name == "nt", reason="Unix account database behavior")
def test_environment_cannot_override_fixed_user_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    before = (consent_path(), outbox_path())
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    assert (consent_path(), outbox_path()) == before


@pytest.mark.parametrize(
    ("platform", "os_name", "expected_root"),
    [
        ("darwin", "posix", Path("/Users/test/Library/Application Support/Praxist")),
        ("win32", "nt", Path("/Users/test/AppData/Local/Praxist")),
        ("linux", "posix", Path("/Users/test/.local/share/praxist")),
    ],
)
def test_platform_paths_share_the_expected_private_product_usage_root(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    os_name: str,
    expected_root: Path,
) -> None:
    monkeypatch.setattr(paths, "_user_home", lambda: Path("/Users/test"))
    monkeypatch.setattr(paths.sys, "platform", platform)
    monkeypatch.setattr(paths, "os", SimpleNamespace(name=os_name))

    assert environment_identity_path() == expected_root / "product-usage/environment.json"
    if platform == "linux":
        assert consent_path() == Path("/Users/test/.config/praxist/product-usage/consent.json")
    else:
        assert consent_path() == expected_root / "product-usage/consent.json"
    assert outbox_path() == expected_root / "product-usage/outbox.sqlite3"


def test_windows_user_home_does_not_require_the_posix_account_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(paths, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        paths,
        "Path",
        SimpleNamespace(home=lambda: Path("C:/Users/test")),
    )

    assert paths._user_home() == Path("C:/Users/test")
