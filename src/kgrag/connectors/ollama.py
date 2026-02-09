"""Ollama LLM connector — generation and embedding."""

from __future__ import annotations

from typing import Any

import structlog
from ollama import AsyncClient

from kgrag.core.config import OllamaConfig
from kgrag.core.exceptions import OllamaConnectionError

logger = structlog.get_logger(__name__)


class OllamaConnector:
    """Async client wrapping the Ollama REST API for generation and embedding.

    Used by answer generation (:mod:`kgrag.agents.answer_generator`) and
    embedding-based search (:class:`~kgrag.retrieval.vector.VectorRetriever`).
    """

    def __init__(self, config: OllamaConfig) -> None:
        self._config = config
        self._client: AsyncClient | None = None

    # -- lifecycle ----------------------------------------------------------

    async def connect(self) -> None:
        """Create the Ollama async client and verify connectivity."""
        try:
            self._client = AsyncClient(host=self._config.base_url)
            # Verify by listing models
            await self._client.list()
            logger.info("ollama.connected", base_url=self._config.base_url)
        except Exception as exc:
            raise OllamaConnectionError(f"Cannot connect to Ollama: {exc}") from exc

    async def close(self) -> None:
        """No persistent connection to close, but kept for API consistency."""
        self._client = None
        logger.info("ollama.closed")

    @property
    def client(self) -> AsyncClient:
        if self._client is None:
            raise OllamaConnectionError("Ollama client not initialised — call connect() first.")
        return self._client

    # -- generation ---------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        format: str | None = None,
    ) -> str:
        """Generate a text completion from the configured generation model."""
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        elif self._config.temperature is not None:
            options["temperature"] = self._config.temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        else:
            options["num_predict"] = self._config.max_tokens

        kwargs: dict[str, Any] = {
            "model": self._config.generation_model,
            "prompt": prompt,
            "options": options,
            "stream": False,
        }
        if system:
            kwargs["system"] = system
        if format:
            kwargs["format"] = format

        response = await self.client.generate(**kwargs)
        return response["response"]

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        format: str | None = None,
    ) -> str:
        """Chat-style generation with message history."""
        options: dict[str, Any] = {
            "temperature": temperature or self._config.temperature,
            "num_predict": self._config.max_tokens,
        }
        kwargs: dict[str, Any] = {
            "model": self._config.generation_model,
            "messages": messages,
            "options": options,
            "stream": False,
        }
        if format:
            kwargs["format"] = format

        response = await self.client.chat(**kwargs)
        return response["message"]["content"]

    # -- embedding ----------------------------------------------------------

    async def embed(self, text: str) -> list[float]:
        """Embed a single text using the configured embedding model."""
        response = await self.client.embed(
            model=self._config.embedding_model,
            input=text,
        )
        return response["embeddings"][0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in one call."""
        response = await self.client.embed(
            model=self._config.embedding_model,
            input=texts,
        )
        return response["embeddings"]
