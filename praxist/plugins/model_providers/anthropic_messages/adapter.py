"""Executable Anthropic Messages model provider plugin."""

from __future__ import annotations

from praxist.core.modeling import ModelProviderAdapter


def create_provider() -> ModelProviderAdapter:
    """Manifest entrypoint for the Anthropic Messages model-provider adapter."""
    return ModelProviderAdapter(
        "model_provider:anthropic_messages", api_format="anthropic_messages"
    )
