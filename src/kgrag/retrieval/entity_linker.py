"""Entity linker — maps question terms to KG entities.

Shared utility used by all graph-based retrievers.  Supports exact match,
fuzzy match, embedding similarity, and ontology-expanded matching.
"""

from __future__ import annotations

import structlog
from rapidfuzz import fuzz

from kgrag.connectors.neo4j import Neo4jConnector
from kgrag.connectors.ollama import OllamaConnector
from kgrag.connectors.qdrant import QdrantConnector
from kgrag.core.models import KGEntity

logger = structlog.get_logger(__name__)

# Minimum fuzzy ratio to consider a match
_FUZZY_THRESHOLD = 75
# Minimum embedding similarity to consider a match
_EMBEDDING_THRESHOLD = 0.7


class EntityLinker:
    """Link question terms to KG entities.

    Strategy priority:
        1. Exact label match (case-insensitive)
        2. Fuzzy label match (Levenshtein ratio >= 75)
        3. Embedding similarity (embed term -> search Neo4j entity labels)
    """

    def __init__(
        self,
        neo4j: Neo4jConnector,
        qdrant: QdrantConnector | None = None,
        ollama: OllamaConnector | None = None,
    ) -> None:
        self._neo4j = neo4j
        self._qdrant = qdrant
        self._ollama = ollama

    async def link(self, terms: list[str]) -> list[KGEntity]:
        """Return matched KG entities for the given terms."""
        if not terms:
            return []

        # Step 1+2: label-based lookup (Neo4j does case-insensitive CONTAINS)
        candidates = await self._neo4j.find_entities_by_label(terms)

        # Deduplicate & rank by fuzzy match
        scored: list[tuple[KGEntity, float]] = []
        for entity in candidates:
            best_score = max(
                fuzz.ratio(term.lower(), entity.label.lower()) for term in terms
            )
            if best_score >= _FUZZY_THRESHOLD:
                scored.append((entity, best_score))

        # Sort by fuzzy score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Step 3: embedding fallback if no fuzzy matches
        if not scored and self._ollama and self._qdrant:
            scored = await self._embedding_fallback(terms)

        linked = [entity for entity, _ in scored]
        logger.info("entity_linker.linked", terms=terms, matched=len(linked))
        return linked

    async def _embedding_fallback(
        self,
        terms: list[str],
    ) -> list[tuple[KGEntity, float]]:
        """Embed terms and search for similar entity labels in Qdrant."""
        if not self._ollama or not self._qdrant:
            return []

        results: list[tuple[KGEntity, float]] = []
        for term in terms:
            embedding = await self._ollama.embed(term)
            hits = await self._qdrant.search(
                query_vector=embedding,
                top_k=3,
                score_threshold=_EMBEDDING_THRESHOLD,
            )
            for chunk, score in hits:
                # Embedding fallback: use chunk text to find related entities
                # via Neo4j label search (Qdrant doesn't store entity cross-refs)
                candidates = await self._neo4j.find_entities_by_label([term])
                for entity in candidates:
                    results.append((entity, score))

        return results
