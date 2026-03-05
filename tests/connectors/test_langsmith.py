"""Unit tests for the LangSmith helper and provider integration."""
from __future__ import annotations

import pytest

from kgrag.connectors.langchain_ollama_provider import LangChainOllamaProvider
from kgrag.core.config import OllamaConfig
from kgrag.telemetry.langsmith import get_langsmith_callbacks


@pytest.mark.parametrize("enabled", [True, False])
def test_get_langsmith_callbacks_respects_env(monkeypatch, enabled: bool) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true" if enabled else "false")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    callbacks = get_langsmith_callbacks()

    if enabled:
        assert callbacks is None or isinstance(callbacks, list)
    else:
        assert callbacks is None


def test_provider_construction_with_optional_callbacks(monkeypatch) -> None:
    # Ensure provider initialization doesn't crash when tracing is on/off
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    cfg = OllamaConfig()
    provider = LangChainOllamaProvider(cfg)
    assert provider.chat_model is not None
    assert provider.embeddings is not None
