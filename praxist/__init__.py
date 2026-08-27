"""Praxist platform package for autonomous research control-plane code."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("praxist")
except PackageNotFoundError:  # pragma: no cover - source tree without package metadata
    __version__ = "0+unknown"
