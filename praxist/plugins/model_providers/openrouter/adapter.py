"""Executable OpenRouter model provider plugin."""

from __future__ import annotations

from praxist.core.modeling import ModelProviderAdapter


def create_provider() -> ModelProviderAdapter:
    """Manifest entrypoint for the OpenRouter model-provider adapter."""
    return ModelProviderAdapter("model_provider:openrouter", api_format="openrouter")
