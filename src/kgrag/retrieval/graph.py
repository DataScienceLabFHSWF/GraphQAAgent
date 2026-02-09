"""GraphRetriever (C3.3.2) — KG-only retrieval with entity-centric, subgraph, and path modes.

Queries Neo4j for structured evidence.  The retrieved subgraphs are serialised
to natural language so the LLM can reason over them — a key transparency
contribution of this thesis.
"""

from __future__ import annotations

import time
from enum import Enum

import structlog

from kgrag.connectors.neo4j import Neo4jConnector
from kgrag.core.config import RetrievalConfig
from kgrag.core.models import (
    KGEntity,
    KGRelation,
    Provenance,
    QAQuery,
    QuestionType,
    RetrievalSource,
    RetrievedContext,
)
from kgrag.retrieval.entity_linker import EntityLinker

logger = structlog.get_logger(__name__)


class GraphMode(Enum):
    """Sub-strategy for graph retrieval."""

    ENTITY_CENTRIC = "entity_centric"
    SUBGRAPH = "subgraph"
    PATH = "path"


class GraphRetriever:
    """KG-only retrieval: entity-centric, subgraph expansion, or path finding.

    Transparently exposes the subgraph used for reasoning so that downstream
    components (Explainer, UI) can visualise exactly which KG evidence was used.
    """

    def __init__(
        self,
        neo4j: Neo4jConnector,
        entity_linker: EntityLinker,
        config: RetrievalConfig,
    ) -> None:
        self._neo4j = neo4j
        self._entity_linker = entity_linker
        self._config = config

    # -- public API ---------------------------------------------------------

    async def retrieve(
        self,
        query: QAQuery,
        *,
        mode: GraphMode | None = None,
    ) -> list[RetrievedContext]:
        """Retrieve graph evidence matching the query.

        If *mode* is ``None``, the mode is automatically selected based on
        ``query.question_type``.
        """
        if mode is None:
            mode = self._auto_select_mode(query)

        t0 = time.perf_counter()

        # 1. Entity-link question terms to KG nodes
        linked = await self._entity_linker.link(query.detected_entities)
        if not linked:
            logger.warning("graph.no_entities_linked", question=query.raw_question[:80])
            return []

        entity_ids = [e.id for e in linked]

        # 2. Dispatch to sub-strategy
        if mode == GraphMode.PATH and len(entity_ids) >= 2:
            contexts = await self._path_retrieve(entity_ids, query)
        elif mode == GraphMode.SUBGRAPH:
            contexts = await self._subgraph_retrieve(entity_ids, query)
        else:
            contexts = await self._entity_centric_retrieve(entity_ids, query)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "graph.retrieve",
            mode=mode.value,
            linked_entities=len(linked),
            contexts=len(contexts),
            latency_ms=round(elapsed, 1),
        )
        return contexts

    # -- sub-strategies -----------------------------------------------------

    async def _entity_centric_retrieve(
        self,
        entity_ids: list[str],
        query: QAQuery,
    ) -> list[RetrievedContext]:
        """Fetch 1-hop neighbourhood and serialise as text."""
        entities, relations = await self._neo4j.get_neighbourhood(
            entity_ids,
            max_hops=1,
            max_nodes=self._config.graph_max_nodes,
        )
        text = self._serialise_subgraph(entities, relations)
        return [
            RetrievedContext(
                source=RetrievalSource.GRAPH,
                text=text,
                score=self._subgraph_score(relations),
                subgraph=[*entities, *relations],
                provenance=Provenance(
                    entity_ids=[e.id for e in entities],
                    retrieval_strategy="graph_entity_centric",
                ),
            )
        ]

    async def _subgraph_retrieve(
        self,
        entity_ids: list[str],
        query: QAQuery,
    ) -> list[RetrievedContext]:
        """Expand to k-hop subgraph between matched entities."""
        entities, relations = await self._neo4j.get_neighbourhood(
            entity_ids,
            max_hops=self._config.graph_max_hops,
            max_nodes=self._config.graph_max_nodes,
        )
        text = self._serialise_subgraph(entities, relations)
        return [
            RetrievedContext(
                source=RetrievalSource.GRAPH,
                text=text,
                score=self._subgraph_score(relations),
                subgraph=[*entities, *relations],
                provenance=Provenance(
                    entity_ids=[e.id for e in entities],
                    retrieval_strategy="graph_subgraph",
                ),
            )
        ]

    async def _path_retrieve(
        self,
        entity_ids: list[str],
        query: QAQuery,
    ) -> list[RetrievedContext]:
        """Find shortest paths between entity pairs."""
        contexts: list[RetrievedContext] = []
        # Pair the first entity with each other
        source_id = entity_ids[0]
        for target_id in entity_ids[1:]:
            paths = await self._neo4j.find_shortest_paths(
                source_id,
                target_id,
                max_hops=self._config.graph_max_hops,
            )
            for path_entities, path_relations in paths:
                text = self._serialise_subgraph(path_entities, path_relations)
                contexts.append(
                    RetrievedContext(
                        source=RetrievalSource.GRAPH,
                        text=text,
                        score=self._subgraph_score(path_relations),
                        subgraph=[*path_entities, *path_relations],
                        provenance=Provenance(
                            entity_ids=[e.id for e in path_entities],
                            retrieval_strategy="graph_path",
                        ),
                    )
                )
        return contexts

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _auto_select_mode(query: QAQuery) -> GraphMode:
        """Pick sub-strategy based on question type."""
        if query.question_type in (QuestionType.CAUSAL, QuestionType.COMPARATIVE):
            return GraphMode.PATH
        if query.question_type == QuestionType.BOOLEAN and len(query.detected_entities) >= 2:
            return GraphMode.PATH
        if query.question_type == QuestionType.LIST:
            return GraphMode.ENTITY_CENTRIC
        return GraphMode.SUBGRAPH

    @staticmethod
    def _serialise_subgraph(
        entities: list[KGEntity],
        relations: list[KGRelation],
    ) -> str:
        """Render subgraph as structured natural-language text for the LLM.

        This serialisation is central to the **transparency contribution** —
        the LLM (and the user) can inspect exactly which entities and relations
        were retrieved.
        """
        lines: list[str] = ["Entities:"]
        for e in entities:
            lines.append(f'- "{e.label}" ({e.entity_type}, confidence: {e.confidence:.2f})')
        lines.append("")
        lines.append("Relations:")
        for r in relations:
            lines.append(
                f'- "{r.source_id}" --[{r.relation_type}]--> "{r.target_id}" '
                f"(confidence: {r.confidence:.2f})"
            )
        return "\n".join(lines)

    @staticmethod
    def _subgraph_score(relations: list[KGRelation]) -> float:
        """Average confidence of relations as a proxy score."""
        if not relations:
            return 0.0
        return sum(r.confidence for r in relations) / len(relations)
