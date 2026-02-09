"""Think-on-Graph iterative graph exploration (C3.3.5).

**SOTA technique** inspired by Think-on-Graph (Sun et al. 2023) and
HippoRAG (Liu et al. 2024).  Instead of one-shot k-hop retrieval, the
agent iteratively explores the KG:

1. Start from entity-linked seed nodes.
2. Compute Personalized PageRank (PPR) to focus on relevant subgraph.
3. LLM evaluates current evidence and decides which edges to follow.
4. Expand along chosen edges, accumulate evidence.
5. Repeat until sufficient evidence or max iterations.

This addresses the **multi-hop reasoning gap** identified by Yu (2025): at
scale, static k-hop expansion introduces noise whereas iterative exploration
keeps the subgraph focused and discriminative.
"""

from __future__ import annotations

import asyncio
import json
import time

import structlog

from kgrag.connectors.neo4j import Neo4jConnector
from kgrag.connectors.langchain_ollama_provider import LangChainOllamaProvider
from kgrag.core.config import RetrievalConfig
from kgrag.core.models import (
    GraphExplorationState,
    KGEntity,
    KGRelation,
    Provenance,
    QAQuery,
    RetrievalSource,
    RetrievedContext,
)
from kgrag.retrieval.entity_linker import EntityLinker

logger = structlog.get_logger(__name__)

_EXPLORATION_PROMPT = """\
You are a graph exploration agent. Given a question and the current subgraph \
evidence, decide which entities to explore next.

Question: {question}

Current evidence (entities and relations found so far):
{current_evidence}

Frontier entities (neighbours not yet explored):
{frontier}

Instructions:
- If the current evidence is SUFFICIENT to answer the question, respond: \
{{"action": "stop", "reason": "..."}}
- If more evidence is needed, select up to {breadth} entities to explore next: \
{{"action": "explore", "entity_ids": ["id1", "id2"], "reason": "..."}}

Respond with ONLY valid JSON."""


class GraphReasoner:
    """Iterative Think-on-Graph exploration with PPR-guided focus.

    Combines two SOTA techniques:
    - **PPR scoring** (HippoRAG): prioritise nodes most relevant to seed entities
    - **LLM-guided exploration** (Think-on-Graph): let the LLM decide which
      edges to follow, avoiding exhaustive traversal
    """

    def __init__(
        self,
        neo4j: Neo4jConnector,
        ollama: LangChainOllamaProvider,
        entity_linker: EntityLinker,
        config: RetrievalConfig,
    ) -> None:
        self._neo4j = neo4j
        self._ollama = ollama
        self._entity_linker = entity_linker
        self._config = config
        self._reasoning = config.reasoning

    async def explore(
        self,
        query: QAQuery,
        seed_entity_ids: list[str] | None = None,
    ) -> tuple[list[RetrievedContext], GraphExplorationState]:
        """Iterative graph exploration from seed entities.

        Returns collected contexts and the full exploration state for
        transparency/visualisation.
        """
        t0 = time.perf_counter()

        # 1. Entity-link if no seeds given
        if not seed_entity_ids:
            linked = await self._entity_linker.link(query.detected_entities)
            # link() returns entity ID strings
            seed_entity_ids = list(linked) if linked else []

        if not seed_entity_ids:
            logger.warning("graph_reasoner.no_seeds", question=query.raw_question[:80])
            return [], GraphExplorationState()

        # 2. PPR to get initial focus subgraph
        ppr_results = await self._neo4j.compute_ppr(
            seed_entity_ids,
            damping=self._reasoning.ppr_damping,
            top_k=self._reasoning.ppr_top_k,
        )

        # 3. Initialise exploration state
        state = GraphExplorationState(
            visited_entity_ids=set(seed_entity_ids),
            frontier_entity_ids={e.id for e, _ in ppr_results if e.id not in seed_entity_ids}
            if ppr_results
            else set(),
        )

        # Collect initial PPR-scored entities
        for entity, score in ppr_results:
            entity.confidence = max(entity.confidence, score)
            state.collected_entities.append(entity)

        state.exploration_path.append(
            f"PPR from seeds {seed_entity_ids[:3]}… → found {len(ppr_results)} relevant nodes"
        )

        # 4. Fetch initial subgraph between PPR-top entities
        top_ids = [e.id for e, _ in ppr_results[:10]] if ppr_results else seed_entity_ids
        if len(top_ids) >= 2:
            ents, rels = await self._neo4j.get_subgraph_between(
                top_ids, max_hops=self._reasoning.max_exploration_iterations
            )
            state.collected_entities.extend(
                e for e in ents if e.id not in {x.id for x in state.collected_entities}
            )
            state.collected_relations.extend(rels)

        # 5. Iterative exploration (if enabled)
        if self._reasoning.enable_iterative_exploration:
            state = await self._iterative_explore(query, state)

        # 6. Build contexts from collected evidence
        contexts = self._build_contexts(state)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "graph_reasoner.explore",
            seeds=len(seed_entity_ids),
            iterations=state.iterations,
            entities=len(state.collected_entities),
            relations=len(state.collected_relations),
            sufficient=state.sufficient_evidence,
            latency_ms=round(elapsed, 1),
        )
        return contexts, state

    async def _iterative_explore(
        self,
        query: QAQuery,
        state: GraphExplorationState,
    ) -> GraphExplorationState:
        """LLM-guided iterative exploration (Think-on-Graph)."""
        for iteration in range(self._reasoning.max_exploration_iterations):
            state.iterations = iteration + 1

            # Get frontier: neighbours of visited nodes not yet explored
            frontier_neighbours = await self._get_frontier(state)
            if not frontier_neighbours:
                state.exploration_path.append(f"Iteration {iteration + 1}: no frontier — stopping")
                state.sufficient_evidence = True
                break

            # Ask LLM whether to continue
            current_evidence = self._format_evidence(state)
            frontier_text = self._format_frontier(frontier_neighbours)

            prompt = _EXPLORATION_PROMPT.format(
                question=query.raw_question,
                current_evidence=current_evidence,
                frontier=frontier_text,
                breadth=self._reasoning.exploration_breadth,
            )

            try:
                response = await self._ollama.generate(
                    prompt=prompt,
                    temperature=0.1,
                    format="json",
                )
                decision = json.loads(response)
            except (json.JSONDecodeError, Exception) as exc:
                logger.warning("graph_reasoner.llm_decision_failed", error=str(exc))
                break

            action = decision.get("action", "stop")
            reason = decision.get("reason", "")

            if action == "stop":
                state.sufficient_evidence = True
                state.exploration_path.append(
                    f"Iteration {iteration + 1}: LLM decided to stop — {reason}"
                )
                break

            # Explore chosen entities
            explore_ids = decision.get("entity_ids", [])[:self._reasoning.exploration_breadth]
            state.exploration_path.append(
                f"Iteration {iteration + 1}: exploring {explore_ids} — {reason}"
            )

            for eid in explore_ids:
                if eid in state.visited_entity_ids:
                    continue
                state.visited_entity_ids.add(eid)

                neighbours = await self._neo4j.get_entity_neighbours(
                    eid, limit=self._reasoning.exploration_breadth,
                )
                for entity, relation in neighbours:
                    if entity.id not in {e.id for e in state.collected_entities}:
                        state.collected_entities.append(entity)
                    rel_key = (relation.source_id, relation.relation_type, relation.target_id)
                    existing_keys = {
                        (r.source_id, r.relation_type, r.target_id)
                        for r in state.collected_relations
                    }
                    if rel_key not in existing_keys:
                        state.collected_relations.append(relation)

        return state

    async def _get_frontier(
        self,
        state: GraphExplorationState,
    ) -> list[tuple[KGEntity, KGRelation]]:
        """Get unexplored neighbours of all visited entities."""
        frontier: list[tuple[KGEntity, KGRelation]] = []
        # Pick a subset of visited entities to expand from (avoid explosion)
        recent_ids = list(state.visited_entity_ids)[-5:]

        tasks = [
            self._neo4j.get_entity_neighbours(eid, limit=self._reasoning.exploration_breadth)
            for eid in recent_ids
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                continue
            # Accept either: iterable[tuple(entity, relation)] OR (list[entity], list[relation])
            if (
                isinstance(result, tuple)
                and len(result) == 2
                and isinstance(result[0], list)
                and isinstance(result[1], list)
            ):
                for entity, relation in zip(result[0], result[1]):
                    if entity.id not in state.visited_entity_ids:
                        frontier.append((entity, relation))
            else:
                for pair in result:
                    try:
                        entity, relation = pair
                    except Exception:
                        # Skip unexpected shapes
                        continue
                    if entity.id not in state.visited_entity_ids:
                        frontier.append((entity, relation))

        return frontier[: self._reasoning.exploration_breadth * 3]

    def _build_contexts(
        self,
        state: GraphExplorationState,
    ) -> list[RetrievedContext]:
        """Convert exploration state into RetrievedContext list."""
        if not state.collected_entities:
            return []

        # Serialise the full collected subgraph
        lines: list[str] = ["Entities:"]
        for e in state.collected_entities:
            lines.append(f'- "{e.label}" ({e.entity_type}, confidence: {e.confidence:.2f})')
        lines.append("")
        lines.append("Relations:")
        for r in state.collected_relations:
            lines.append(
                f'- "{r.source_id}" --[{r.relation_type}]--> "{r.target_id}" '
                f"(confidence: {r.confidence:.2f})"
            )

        text = "\n".join(lines)

        # Compute score as average confidence
        scores = [e.confidence for e in state.collected_entities if e.confidence > 0]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return [
            RetrievedContext(
                source=RetrievalSource.GRAPH,
                text=text,
                score=avg_score,
                subgraph=[*state.collected_entities, *state.collected_relations],
                provenance=Provenance(
                    entity_ids=[e.id for e in state.collected_entities],
                    retrieval_strategy="graph_reasoning_tog",
                ),
            )
        ]

    @staticmethod
    def _format_evidence(state: GraphExplorationState) -> str:
        """Format current evidence for LLM prompt."""
        lines: list[str] = []
        for e in state.collected_entities[:20]:
            lines.append(f"Entity: {e.label} ({e.entity_type})")
        for r in state.collected_relations[:20]:
            lines.append(f"Relation: {r.source_id} --[{r.relation_type}]--> {r.target_id}")
        return "\n".join(lines) if lines else "(no evidence yet)"

    @staticmethod
    def _format_frontier(frontier: list[tuple[KGEntity, KGRelation]]) -> str:
        """Format frontier entities for LLM prompt."""
        lines: list[str] = []
        seen: set[str] = set()
        for entity, relation in frontier:
            if entity.id in seen:
                continue
            seen.add(entity.id)
            lines.append(
                f"- {entity.id}: \"{entity.label}\" ({entity.entity_type}) "
                f"via [{relation.relation_type}]"
            )
        return "\n".join(lines) if lines else "(no unexplored neighbours)"
