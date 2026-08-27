"""Executable DeepSeek OpenAI-compatible model provider plugin."""

from __future__ import annotations

from praxist.core.modeling import ModelProviderAdapter


def create_provider() -> ModelProviderAdapter:
    """Manifest entrypoint for the DeepSeek OpenAI-compatible provider alias."""
    return ModelProviderAdapter("model_provider:deepseek_alias", api_format="openai_compatible")
