"""Read-only Qdrant connector — consumes KGB's vector store."""

from __future__ import annotations

from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import ScoredPoint

from kgrag.core.config import QdrantConfig
from kgrag.core.exceptions import QdrantConnectionError
from kgrag.core.models import DocumentChunk

logger = structlog.get_logger(__name__)


class QdrantConnector:
    """Async read-only client for Qdrant vector similarity search.

    Used by :class:`~kgrag.retrieval.vector.VectorRetriever` to fetch
    the top-k document chunks nearest to an embedded query.
    """

    def __init__(self, config: QdrantConfig) -> None:
        self._config = config
        self._client: AsyncQdrantClient | None = None

    # -- lifecycle ----------------------------------------------------------

    async def connect(self) -> None:
        """Open the Qdrant connection and verify the collection exists."""
        try:
            self._client = AsyncQdrantClient(url=self._config.url)
            info = await self._client.get_collection(self._config.collection_name)
            
            # Extract vector count safely (handle different Qdrant versions)
            vectors_count = getattr(info, 'vectors_count', getattr(info, 'points_count', 'unknown'))
            
            logger.info(
                "qdrant.connected",
                url=self._config.url,
                collection=self._config.collection_name,
                vectors_count=vectors_count,
            )
        except Exception as exc:
            raise QdrantConnectionError(f"Cannot connect to Qdrant: {exc}") from exc

    async def close(self) -> None:
        """Close the client."""
        if self._client:
            await self._client.close()
            logger.info("qdrant.closed")

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise QdrantConnectionError("Qdrant client not initialised — call connect() first.")
        return self._client

    # -- search -------------------------------------------------------------

    async def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 10,
        score_threshold: float | None = None,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Search the collection for the top-k nearest neighbours.

        Returns a list of ``(DocumentChunk, score)`` tuples sorted by similarity
        descending.
        """
        results = await self.client.query_points(
            collection_name=self._config.collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=filter_conditions,
        )

        return [
            (self._point_to_chunk(point), point.score)
            for point in results.points
        ]

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _point_to_chunk(point: ScoredPoint) -> DocumentChunk:
        payload = point.payload or {}
        return DocumentChunk(
            id=payload.get("id", str(point.id)),
            doc_id=payload.get("doc_id", ""),
            content=payload.get("content", ""),
            strategy=payload.get("strategy", ""),
            embedding=None,  # omit for memory efficiency
        )
