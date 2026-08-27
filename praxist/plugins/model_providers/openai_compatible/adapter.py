"""Executable OpenAI-compatible model provider plugin."""

from __future__ import annotations

from praxist.core.modeling import ModelProviderAdapter


def create_provider() -> ModelProviderAdapter:
    """Manifest entrypoint for generic OpenAI-compatible model providers."""
    return ModelProviderAdapter("model_provider:openai_compatible", api_format="openai_compatible")
