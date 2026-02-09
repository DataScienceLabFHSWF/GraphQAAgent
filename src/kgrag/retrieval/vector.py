"""VectorRetriever (C3.3.1) — Classic RAG baseline.

Embeds the question via Ollama and retrieves top-k nearest document chunks
from Qdrant.  Serves as the baseline for strategy comparison in C3.5.
"""

from __future__ import annotations

import time

import structlog

from kgrag.connectors.langchain_ollama_provider import LangChainOllamaProvider
from kgrag.connectors.qdrant import QdrantConnector
from kgrag.core.config import RetrievalConfig
from kgrag.core.models import (
    Provenance,
    QAQuery,
    RetrievalSource,
    RetrievedContext,
)

logger = structlog.get_logger(__name__)


class VectorRetriever:
    """Classic RAG: embed question -> search Qdrant -> return top-k chunks."""

    def __init__(
        self,
        qdrant: QdrantConnector,
        ollama: LangChainOllamaProvider,
        config: RetrievalConfig,
    ) -> None:
        self._qdrant = qdrant
        self._ollama = ollama
        self._config = config

    async def retrieve(self, query: QAQuery) -> list[RetrievedContext]:
        """Embed the raw question and search Qdrant for nearest chunks.

        Steps:
            1. Embed ``query.raw_question`` via Ollama embedding model.
            2. Search Qdrant collection for top-k nearest chunks.
            3. Wrap results as ``RetrievedContext(source=VECTOR)``.
        """
        t0 = time.perf_counter()

        # 1. Embed the question
        query_vector = await self._ollama.embed(query.raw_question)

        # 2. Search Qdrant
        results = await self._qdrant.search(
            query_vector=query_vector,
            top_k=self._config.vector_top_k,
        )

        # 3. Build RetrievedContext list
        contexts: list[RetrievedContext] = []
        for chunk, score in results:
            contexts.append(
                RetrievedContext(
                    source=RetrievalSource.VECTOR,
                    text=chunk.content,
                    score=score,
                    chunk=chunk,
                    provenance=Provenance(
                        doc_id=chunk.doc_id,
                        source_id=chunk.id,
                        retrieval_strategy="vector_only",
                        retrieval_score=score,
                    ),
                )
            )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "vector.retrieve",
            question=query.raw_question[:80],
            num_results=len(contexts),
            latency_ms=round(elapsed, 1),
        )
        return contexts
