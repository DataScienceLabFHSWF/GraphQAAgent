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
_FUZZY_THRESHOLD = 50
# Minimum embedding similarity to consider a match
_EMBEDDING_THRESHOLD = 0.6


class EntityLinker:
    """Link question terms to KG entities.

    Strategy priority:
        1. Exact ID match (case-insensitive)
        2. Label / alias search via Neo4j (searches id, label, properties JSON)
        3. Fuzzy ranking (Levenshtein ratio) with lenient threshold
        4. Embedding similarity fallback (embed term -> search Qdrant)
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

        # Step 1: Try exact ID lookup first (e.g. "AtG", "StrlSchG")
        id_entities = await self._neo4j.find_entities_by_ids(terms)

        # Step 2: Label / alias / property search (Neo4j does CONTAINS)
        label_candidates = await self._neo4j.find_entities_by_label(terms, limit=30)

        # Merge, dedup by id
        seen: set[str] = set()
        all_candidates: list[KGEntity] = []
        # ID matches get priority
        for e in id_entities:
            if e.id not in seen:
                seen.add(e.id)
                all_candidates.append(e)
        for e in label_candidates:
            if e.id not in seen:
                seen.add(e.id)
                all_candidates.append(e)

        # Step 3: Rank by best fuzzy match against any term
        scored: list[tuple[KGEntity, float]] = []
        for entity in all_candidates:
            best_score = 0.0
            for term in terms:
                t = term.lower()
                # Score against multiple fields
                scores = [
                    fuzz.ratio(t, entity.label.lower()),
                    fuzz.partial_ratio(t, entity.label.lower()),
                    fuzz.ratio(t, entity.id.lower()) * 1.2,  # Boost ID matches
                ]
                if entity.description:
                    scores.append(fuzz.partial_ratio(t, entity.description.lower()) * 0.8)
                best_score = max(best_score, max(scores))
            # Accept if above threshold OR if it was an exact ID match
            if best_score >= _FUZZY_THRESHOLD or entity.id.lower() in [t.lower() for t in terms]:
                scored.append((entity, best_score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Step 4: Embedding fallback if no matches at all
        if not scored and self._ollama and self._qdrant:
            scored = await self._embedding_fallback(terms)

        linked = [entity for entity, _ in scored[:15]]  # cap at 15
        logger.info(
            "entity_linker.linked",
            terms=terms,
            candidates=len(all_candidates),
            matched=len(linked),
            top_ids=[e.id for e in linked[:5]],
        )
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
                # Use chunk text to find related entities via Neo4j
                candidates = await self._neo4j.find_entities_by_label([term])
                for entity in candidates:
                    results.append((entity, score))

        return results
