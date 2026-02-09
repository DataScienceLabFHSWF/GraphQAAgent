"""HybridRetriever / FusionRAG (C3.3.3) — Primary retrieval strategy.

**Key contribution**: ontology-informed fusion where graph results are weighted
higher when the question targets known ontology relations, combined with
Reciprocal Rank Fusion and cross-encoder reranking.

SOTA enhancements (v2):
- **Three-way fusion**: vector + graph + Think-on-Graph iterative exploration
- **PPR-guided graph retrieval**: Personalized PageRank replaces uniform k-hop
  (inspired by HippoRAG, Yu 2025)
- **Relation-aware path ranking**: ontology-informed path scoring
  (inspired by PathCon, PRA)
- **Adaptive operator selection**: question-type-aware operator mix
  (validated by Yu 2025, Finding #2 & #5)

The hybrid approach provides transparent provenance by tracking exactly which
source (vector, graph, ontology, exploration) contributed each context piece.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict

import structlog

from kgrag.connectors.fuseki import FusekiConnector
from kgrag.core.config import RetrievalConfig
from kgrag.core.models import (
    GraphExplorationState,
    Provenance,
    QAQuery,
    QuestionType,
    RetrievalSource,
    RetrievedContext,
)
from kgrag.retrieval.graph import GraphRetriever
from kgrag.retrieval.graph_reasoning import GraphReasoner
from kgrag.retrieval.path_ranker import PathRanker
from kgrag.retrieval.reranker import CrossEncoderReranker
from kgrag.retrieval.vector import VectorRetriever

logger = structlog.get_logger(__name__)


class HybridRetriever:
    """FusionRAG v2: three-way retrieval with PPR, path ranking, and adaptive fusion.

    Pipeline:
    1. Parallel retrieval: vector + graph + (optional) ToG exploration
    2. Ontology-informed adaptive weight adjustment
    3. Relation-aware path ranking on graph contexts
    4. Reciprocal Rank Fusion (RRF) merge
    5. Cross-encoder reranking
    6. Deduplication + provenance attachment
    """

    def __init__(
        self,
        vector: VectorRetriever,
        graph: GraphRetriever,
        fuseki: FusekiConnector,
        reranker: CrossEncoderReranker,
        config: RetrievalConfig,
        *,
        graph_reasoner: GraphReasoner | None = None,
        path_ranker: PathRanker | None = None,
    ) -> None:
        self._vector = vector
        self._graph = graph
        self._fuseki = fuseki
        self._reranker = reranker
        self._config = config
        self._graph_reasoner = graph_reasoner
        self._path_ranker = path_ranker or PathRanker()

    async def retrieve(self, query: QAQuery) -> list[RetrievedContext]:
        """Full hybrid retrieval pipeline (v2).

        1. Parallel vector + graph + (ToG exploration).
        2. Relation-aware path ranking on graph results.
        3. Adaptive weight computation.
        4. RRF merge.
        5. Cross-encoder reranking.
        6. Provenance finalization.
        """
        t0 = time.perf_counter()

        # Phase 1 — Parallel retrieval (three-way if ToG is available)
        retrieval_tasks: dict[str, asyncio.Task] = {}
        async with asyncio.TaskGroup() as tg:
            retrieval_tasks["vector"] = tg.create_task(self._vector.retrieve(query))
            retrieval_tasks["graph"] = tg.create_task(self._graph.retrieve(query))
            if self._graph_reasoner and self._should_explore(query):
                retrieval_tasks["exploration"] = tg.create_task(
                    self._explore_wrapper(query)
                )

        vector_ctx = retrieval_tasks["vector"].result()
        graph_ctx = retrieval_tasks["graph"].result()

        exploration_ctx: list[RetrievedContext] = []
        exploration_state: GraphExplorationState | None = None
        if "exploration" in retrieval_tasks:
            exploration_ctx, exploration_state = retrieval_tasks["exploration"].result()

        # Phase 2 — Relation-aware path ranking on graph + exploration contexts
        all_graph_ctx = graph_ctx + exploration_ctx
        if all_graph_ctx and query.expected_relations:
            all_graph_ctx = self._path_ranker.rank_paths(all_graph_ctx, query)

        # Phase 3 — Adaptive weights
        weights = await self._compute_adaptive_weights(query, exploration_state)

        # Phase 4 — Reciprocal Rank Fusion
        ranked_lists: dict[str, list[RetrievedContext]] = {
            "vector": vector_ctx,
            "graph": all_graph_ctx,
        }
        fused = reciprocal_rank_fusion(ranked_lists, weights)

        # Phase 5 — Cross-encoder reranking
        reranked = self._reranker.rerank(
            query=query.raw_question,
            contexts=fused,
            top_k=self._config.vector_top_k,
        )

        # Phase 6 — Mark provenance as hybrid
        for ctx in reranked:
            if ctx.provenance:
                ctx.provenance.retrieval_strategy = (
                    f"hybrid({ctx.provenance.retrieval_strategy})"
                )
            ctx.source = RetrievalSource.HYBRID

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "hybrid.retrieve",
            vector_count=len(vector_ctx),
            graph_count=len(graph_ctx),
            exploration_count=len(exploration_ctx),
            fused_count=len(fused),
            final_count=len(reranked),
            weights=weights,
            explored=exploration_state is not None,
            latency_ms=round(elapsed, 1),
        )
        return reranked

    # -- Think-on-Graph integration -----------------------------------------

    def _should_explore(self, query: QAQuery) -> bool:
        """Decide whether to trigger iterative graph exploration.

        Use ToG for complex questions that benefit from multi-hop reasoning:
        - CAUSAL / COMPARATIVE questions (path-heavy)
        - Queries with sub-questions (multi-hop)
        - Queries with 2+ entities (need connecting subgraph)
        """
        if query.question_type in (QuestionType.CAUSAL, QuestionType.COMPARATIVE):
            return True
        if query.sub_questions and len(query.sub_questions) > 1:
            return True
        if len(query.detected_entities) >= 2:
            return True
        return False

    async def _explore_wrapper(
        self,
        query: QAQuery,
    ) -> tuple[list[RetrievedContext], GraphExplorationState]:
        """Wrapper for graph exploration to match asyncio.TaskGroup interface."""
        assert self._graph_reasoner is not None
        return await self._graph_reasoner.explore(query)

    # -- adaptive weighting -------------------------------------------------

    async def _compute_adaptive_weights(
        self,
        query: QAQuery,
        exploration_state: GraphExplorationState | None = None,
    ) -> dict[str, float]:
        """Adjust fusion weights based on query + exploration characteristics.

        Enhanced v2 heuristics:
        - Entities in KG → boost graph
        - CAUSAL/COMPARATIVE → boost graph (needs path reasoning)
        - No entities / FACTOID → boost vector (fuzzy matching)
        - Successful ToG exploration → boost graph further
        - Ontology types with many relations → boost graph
        """
        base = dict(self._config.fusion_weights)

        # Boost graph for structured question types
        if query.question_type in (QuestionType.CAUSAL, QuestionType.COMPARATIVE):
            base["graph"] = base.get("graph", 0.4) + 0.15
            base["vector"] = base.get("vector", 0.4) - 0.10

        # Boost graph when entities are detected
        if query.detected_entities:
            base["graph"] = base.get("graph", 0.4) + 0.05
        else:
            # No entities → fall back more on vector
            base["vector"] = base.get("vector", 0.4) + 0.15
            base["graph"] = base.get("graph", 0.4) - 0.10

        # Boost graph when ToG exploration found good evidence
        if exploration_state and exploration_state.sufficient_evidence:
            base["graph"] = base.get("graph", 0.4) + 0.10
        elif exploration_state and exploration_state.collected_relations:
            base["graph"] = base.get("graph", 0.4) + 0.05

        # Boost graph when ontology types have rich relations
        if query.detected_types:
            for type_uri in query.detected_types[:3]:  # limit lookups
                props = await self._fuseki.get_class_properties(type_uri)
                if len(props) > 5:
                    base["graph"] = base.get("graph", 0.4) + 0.05
                    break

        # Normalise to sum to 1
        total = sum(base.values())
        if total > 0:
            base = {k: v / total for k, v in base.items()}

        return base


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[RetrievedContext]],
    weights: dict[str, float],
    k: int = 60,
) -> list[RetrievedContext]:
    """Standard RRF with per-source weights.

    For each document *d* appearing in any ranked list::

        rrf_score(d) = sum_{source} weight[source] / (k + rank[source](d))

    Returns merged list sorted by RRF score descending.
    """
    # Use text as dedup key
    scores: dict[str, float] = defaultdict(float)
    context_map: dict[str, RetrievedContext] = {}

    for source, contexts in ranked_lists.items():
        weight = weights.get(source, 1.0)
        for rank, ctx in enumerate(contexts, start=1):
            key = ctx.text[:200]  # fingerprint
            scores[key] += weight / (k + rank)
            if key not in context_map:
                context_map[key] = ctx

    # Sort by fused score
    sorted_keys = sorted(scores, key=lambda k_: scores[k_], reverse=True)
    result: list[RetrievedContext] = []
    for key in sorted_keys:
        ctx = context_map[key]
        ctx.score = scores[key]
        result.append(ctx)

    return result
