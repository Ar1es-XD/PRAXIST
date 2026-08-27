"""Executable fake model provider plugin."""

from __future__ import annotations

from praxist.core.modeling import ModelProviderAdapter


def create_provider() -> ModelProviderAdapter:
    return ModelProviderAdapter("model_provider:fake_provider", api_format="fake")
