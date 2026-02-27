"""AgenticGraphRAG — LangGraph ReAct agent that orchestrates all retrieval tools.

This is the "everything working together in an agentic/tooly fashion" retriever.
It uses an LLM agent with tool-calling to dynamically decide:
- When to search vectors (semantic similarity over document chunks)
- When to generate and run Cypher (structured graph queries)
- When to explore multi-hop neighborhoods (entity-centric traversal)
- When to find connections between entities (shortest-path fact chains)
- When to consult the ontology (what types/relations exist)
- When it has enough context to stop

The agent is ontology-informed: the full TBox schema is injected into its
system prompt so it understands what classes, properties, and hierarchies
exist in the knowledge graph.

**Graph reasoning** is implemented via ``find_connections``: shortest-path
traversal produces verifiable *fact chains* where every claim is grounded
in specific KG entity IDs and relation types.

Implements the ``Retriever`` protocol.
"""

from __future__ import annotations

import json
import asyncio
import structlog
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from kgrag.connectors.neo4j import Neo4jConnector
from kgrag.connectors.langchain_ollama_provider import LangChainOllamaProvider
from kgrag.connectors.qdrant import QdrantConnector
from kgrag.core.config import Neo4jConfig, RetrievalConfig
from kgrag.core.models import (
    KGEntity,
    KGRelation,
    Provenance,
    QAQuery,
    RetrievalSource,
    RetrievedContext,
)
from kgrag.core.domain import DomainConfig
from kgrag.retrieval.ontology_context import OntologyContext
from kgrag.retrieval.active_ontology import OntologyGapDetector

logger = structlog.get_logger(__name__)

# Max iterations for the agent loop to prevent infinite loops
_MAX_ITERATIONS = 8
# Max total contexts to collect before stopping
_MAX_CONTEXTS = 25

# Adaptive iteration limits by question type
_ITERATIONS_BY_TYPE: dict[str | None, int] = {
    "factoid": 4,
    "boolean": 3,
    "list": 5,
    "comparative": 7,
    "causal": 7,
    "aggregation": 5,
    None: _MAX_ITERATIONS,              # unknown type → default
}

# Reflection prompt injected after N iterations to check sufficiency
_REFLECTION_THRESHOLD = 3
_REFLECTION_PROMPT = (
    "You have completed {n} tool calls and collected {ctx} evidence pieces. "
    "Evaluate: do you have SUFFICIENT evidence to answer the question? "
    "If yes, call collect_evidence now. If not, identify what is still missing "
    "and make targeted tool calls to fill those gaps."
)

# Retrieval plan prompt (ReWOO-inspired plan-then-execute)
_PLAN_PROMPT = """\
You are about to answer a question using tools that search a knowledge graph.
Before making any tool calls, briefly plan your retrieval strategy.

Question: {question}
Question type: {qtype}
Detected entities: {entities}

Available tools: search_vectors, query_graph, explore_entity, find_connections, \
lookup_ontology, aggregate_subgraph, semantic_search_entities, compare_entities, \
count_and_aggregate, collect_evidence

Write a short retrieval plan (2-4 steps) as a numbered list, then proceed with \
your first tool call. Example:
1. Look up entity X to understand its neighbourhood
2. Search vectors for background context about Y
3. Find connections between X and Y
4. Collect evidence

Plan:"""

# Minimum embedding similarity to keep an evidence piece (quality filter)
_EVIDENCE_QUALITY_THRESHOLD = 0.15


class AgenticGraphRAG:
    """LangGraph-style ReAct agent combining all retrieval strategies.

    Tools:
    - ``search_vectors``: Semantic search over document chunks
    - ``query_graph``: Run Cypher queries against Neo4j
    - ``explore_entity``: Multi-hop neighborhood expansion
    - ``find_connections``: Shortest-path fact chains between entities
    - ``lookup_ontology``: Check available types, relations, hierarchy
    - ``collect_evidence``: Finalize collected evidence (signals done)
    """

    def __init__(
        self,
        neo4j: Neo4jConnector,
        neo4j_config: Neo4jConfig,
        qdrant: QdrantConnector,
        ollama: LangChainOllamaProvider,
        ontology_context: OntologyContext,
        config: RetrievalConfig | None = None,
        domain_config: DomainConfig | None = None,
    ) -> None:
        self._neo4j = neo4j
        self._neo4j_config = neo4j_config
        self._qdrant = qdrant
        self._ollama = ollama
        self._ontology = ontology_context
        self._config = config
        self._domain = domain_config or DomainConfig.load()
        self._neo4j_graph = None  # Lazy-loaded Neo4jGraph for schema
        self._gap_detector = OntologyGapDetector(ontology_context)
        self._detected_gaps: list[Any] = []  # OntologyGap objects from this session

    def _get_neo4j_schema(self) -> str:
        """Get the Neo4j schema string (lazy-loaded)."""
        if self._neo4j_graph is None:
            try:
                from langchain_neo4j import Neo4jGraph
                self._neo4j_graph = Neo4jGraph(
                    url=self._neo4j_config.uri,
                    username=self._neo4j_config.user,
                    password=self._neo4j_config.password,
                    database=self._neo4j_config.database,
                    enhanced_schema=False,
                )
            except Exception as exc:
                logger.warning("agentic.neo4j_schema_failed", error=str(exc))
                return "(schema unavailable)"
        return self._neo4j_graph.schema[:4000]  # Truncate for prompt size

    # -- Evidence quality scoring -------------------------------------------

    async def _score_evidence(
        self,
        query_text: str,
        contexts: list[RetrievedContext],
    ) -> list[RetrievedContext]:
        """Score and filter evidence by embedding similarity to the query.

        Returns only contexts whose cosine similarity with the query exceeds
        ``_EVIDENCE_QUALITY_THRESHOLD``.  Contexts that already carry a
        relevance score ≥ the threshold are kept as-is.
        """
        if not contexts:
            return contexts

        try:
            texts = [query_text] + [c.text for c in contexts]
            embeddings = await self._ollama.embed_batch(texts)
            query_emb = embeddings[0]

            import math

            def _cosine(a: list[float], b: list[float]) -> float:
                dot = sum(x * y for x, y in zip(a, b))
                na = math.sqrt(sum(x * x for x in a))
                nb = math.sqrt(sum(x * x for x in b))
                return dot / (na * nb) if na and nb else 0.0

            scored: list[RetrievedContext] = []
            for ctx, emb in zip(contexts, embeddings[1:]):
                sim = _cosine(query_emb, emb)
                if sim >= _EVIDENCE_QUALITY_THRESHOLD:
                    # Carry the relevance score in the context
                    ctx.relevance_score = max(ctx.relevance_score or 0.0, sim)
                    scored.append(ctx)
                else:
                    logger.debug(
                        "agentic.evidence_filtered",
                        text=ctx.text[:60],
                        similarity=round(sim, 3),
                    )
            return scored
        except Exception as exc:
            logger.warning("agentic.evidence_scoring_failed", error=str(exc))
            return contexts  # fail-open: keep everything

    # -- Retriever protocol -------------------------------------------------

    async def retrieve(self, query: QAQuery) -> list[RetrievedContext]:
        """Run the agentic retrieval loop.

        Enhancements over basic ReAct:
        - **Retrieval plan**: before tool execution a ReWOO-style plan is
          generated so the agent has a strategy to follow
        - **Tool result caching**: identical calls are memo-ised within the loop
        - **Adaptive iteration control**: limit adjusted by QuestionType
        - **Self-reflection**: after ``_REFLECTION_THRESHOLD`` iterations, a
          sufficiency prompt nudges the agent to stop or be targeted
        - **Evidence quality scoring**: low-relevance evidence is filtered via
          embedding cosine similarity before accumulation
        - **Tool trace**: every tool call is recorded for transparency
        """
        llm = self._ollama.get_chat_model()

        # Build tools (closures that capture self)
        tools = self._build_tools()
        tool_map = {t.name: t for t in tools}

        # Bind tools to LLM
        try:
            llm_with_tools = llm.bind_tools(tools)
        except Exception as exc:
            logger.warning("agentic.bind_tools_failed", error=str(exc))
            return await self._fallback_retrieve(query)

        # System prompt with ontology + schema
        neo4j_schema = self._get_neo4j_schema()
        system = SystemMessage(
            content=self._domain.render_prompt(
                "agentic_system",
                ontology_summary=self._ontology.schema_summary,
                neo4j_schema=neo4j_schema,
            )
        )
        human = HumanMessage(content=query.raw_question)

        messages = [system, human]

        # ReWOO-style retrieval plan: ask the LLM to plan before executing
        entity_labels = [e.label for e in (query.entities or [])[:10]]
        plan_prompt = _PLAN_PROMPT.format(
            question=query.raw_question,
            qtype=qtype or "unknown",
            entities=", ".join(entity_labels) if entity_labels else "(none detected)",
        )
        messages.append(HumanMessage(content=plan_prompt))

        collected_contexts: list[RetrievedContext] = []
        collected_entities: list[KGEntity] = []
        collected_relations: list[KGRelation] = []
        fact_chains: list[dict[str, Any]] = []
        tool_trace: list[dict[str, Any]] = []

        # Tool result cache: (tool_name, canonical_args) -> result_str
        _cache: dict[str, str] = {}

        # Adaptive iteration limit based on question type
        qtype = query.question_type.value if query.question_type else None
        max_iters = _ITERATIONS_BY_TYPE.get(qtype, _MAX_ITERATIONS)
        logger.info("agentic.start", question_type=qtype, max_iterations=max_iters)

        reflection_injected = False

        for iteration in range(1, max_iters + 1):
            logger.info("agentic.iteration", iteration=iteration)

            # Self-reflection: after threshold, nudge the agent
            if (
                iteration > _REFLECTION_THRESHOLD
                and not reflection_injected
                and collected_contexts
            ):
                reflection_msg = HumanMessage(content=_REFLECTION_PROMPT.format(
                    n=iteration - 1, ctx=len(collected_contexts),
                ))
                messages.append(reflection_msg)
                reflection_injected = True

            try:
                response = await llm_with_tools.ainvoke(messages)
            except Exception as exc:
                logger.warning("agentic.llm_error", error=str(exc), iteration=iteration)
                break

            messages.append(response)

            # Check if there are tool calls
            if not response.tool_calls:
                logger.info("agentic.no_tool_calls", iteration=iteration)
                break

            # Execute each tool call
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc.get("id", tool_name)

                logger.info(
                    "agentic.tool_call",
                    tool=tool_name,
                    args_keys=list(tool_args.keys()),
                )

                if tool_name == "collect_evidence":
                    summary = tool_args.get("summary", "")
                    logger.info(
                        "agentic.done",
                        summary=summary[:100],
                        contexts=len(collected_contexts),
                    )
                    tool_trace.append({
                        "tool": tool_name, "args": tool_args,
                        "result_summary": summary[:100], "iteration": iteration,
                    })
                    messages.append(
                        ToolMessage(content="Evidence collected.", tool_call_id=tool_id)
                    )
                    return self._finalize(
                        collected_contexts, collected_entities,
                        collected_relations, fact_chains, tool_trace,
                    )

                if tool_name in tool_map:
                    # Cache key: tool name + sorted args
                    cache_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True, default=str)}"

                    if cache_key in _cache:
                        result_str = _cache[cache_key]
                        logger.info("agentic.cache_hit", tool=tool_name)
                        messages.append(ToolMessage(
                            content=f"(cached) {result_str[:3000]}",
                            tool_call_id=tool_id,
                        ))
                        tool_trace.append({
                            "tool": tool_name, "args": tool_args,
                            "result_summary": "(cached)", "iteration": iteration,
                        })
                        continue

                    try:
                        result = await tool_map[tool_name].ainvoke(tool_args)
                        result_str = (
                            result if isinstance(result, str)
                            else json.dumps(result, default=str)
                        )

                        # Store in cache
                        _cache[cache_key] = result_str

                        # Parse results and accumulate contexts
                        new_ctxs, new_ents, new_rels, new_chains = self._parse_tool_result(
                            tool_name, tool_args, result_str,
                        )

                        # Evidence quality filter: score new contexts
                        if new_ctxs:
                            new_ctxs = await self._score_evidence(
                                query.raw_question, new_ctxs,
                            )
                        collected_contexts.extend(new_ctxs)
                        collected_entities.extend(new_ents)
                        collected_relations.extend(new_rels)
                        fact_chains.extend(new_chains)

                        # Record tool trace
                        tool_trace.append({
                            "tool": tool_name, "args": tool_args,
                            "result_summary": result_str[:150],
                            "iteration": iteration,
                        })

                        messages.append(ToolMessage(
                            content=result_str[:3000],
                            tool_call_id=tool_id,
                        ))
                    except Exception as exc:
                        logger.warning(
                            "agentic.tool_error", tool=tool_name, error=str(exc),
                        )
                        tool_trace.append({
                            "tool": tool_name, "args": tool_args,
                            "result_summary": f"Error: {str(exc)[:100]}",
                            "iteration": iteration,
                        })
                        messages.append(ToolMessage(
                            content=f"Error: {str(exc)[:200]}",
                            tool_call_id=tool_id,
                        ))
                else:
                    messages.append(ToolMessage(
                        content=f"Unknown tool: {tool_name}",
                        tool_call_id=tool_id,
                    ))

            if len(collected_contexts) >= _MAX_CONTEXTS:
                logger.info(
                    "agentic.max_contexts_reached",
                    contexts=len(collected_contexts),
                )
                break

        return self._finalize(
            collected_contexts, collected_entities,
            collected_relations, fact_chains, tool_trace,
        )

    # -- Tool definitions ---------------------------------------------------

    def _build_tools(self) -> list:
        """Build LangChain tools that close over this instance's connectors."""

        neo4j = self._neo4j
        qdrant = self._qdrant
        ollama = self._ollama
        ontology = self._ontology
        neo4j_config = self._neo4j_config

        @tool
        async def search_vectors(query: str) -> str:
            """Search document chunks by semantic similarity. Returns text snippets.
            Use this for general questions, finding relevant documents, or when
            you need background context. Input: the search query string."""
            try:
                embedding = await ollama.embed(query)
                results = await qdrant.search(query_vector=embedding, top_k=5)
                if not results:
                    return "No vector results found."
                lines = []
                for chunk, score in results:
                    lines.append(f"[score={score:.3f}] {chunk.content[:300]}")
                return "\n---\n".join(lines)
            except Exception as e:
                return f"Vector search error: {e}"

        @tool
        async def query_graph(cypher: str) -> str:
            """Execute a Cypher query against the Neo4j knowledge graph.
            Returns the query results as JSON. Write valid Cypher using the
            schema and ontology provided in the system prompt.
            Always LIMIT to at most 25 rows.
            Input: a valid Cypher query string."""
            try:
                from neo4j import AsyncGraphDatabase
                driver = AsyncGraphDatabase.driver(
                    neo4j_config.uri,
                    auth=(neo4j_config.user, neo4j_config.password),
                )
                async with driver.session(database=neo4j_config.database) as session:
                    result = await session.run(cypher)
                    records = [dict(r) async for r in result]
                await driver.close()

                if not records:
                    return "Query returned 0 rows."

                lines = []
                for rec in records[:25]:
                    row = {}
                    for k, v in rec.items():
                        if hasattr(v, "items"):
                            row[k] = dict(v)
                        else:
                            row[k] = v
                    lines.append(json.dumps(row, default=str, ensure_ascii=False))
                return f"{len(records)} rows:\n" + "\n".join(lines)
            except Exception as e:
                return f"Cypher error: {e}"

        @tool
        async def explore_entity(entity_id: str, max_hops: int = 1) -> str:
            """Explore the neighborhood of a specific entity in the knowledge graph.
            Returns connected entities and relationships as structured JSON.
            Use this to discover what is connected to a specific entity.
            Input: entity_id (e.g. 'AtG', 'ent_rule_0000'), max_hops (1-3)."""
            try:
                hops = min(max(max_hops, 1), 3)
                entities, relations = await neo4j.get_neighbourhood(
                    [entity_id], max_hops=hops, max_nodes=30,
                )
                # Fallback: search by label if ID not found
                if not entities and not relations:
                    found = await neo4j.find_entities_by_label([entity_id], limit=3)
                    if found:
                        entities, relations = await neo4j.get_neighbourhood(
                            [f.id for f in found], max_hops=hops, max_nodes=30,
                        )
                        if not entities:
                            entities = found

                if not entities and not relations:
                    return json.dumps({"error": f"No data found for '{entity_id}'."})

                result = {
                    "entity_id": entity_id,
                    "hops": hops,
                    "entities": [
                        {
                            "id": e.id,
                            "label": e.label[:80],
                            "type": e.entity_type,
                        }
                        for e in entities[:20]
                    ],
                    "relations": [
                        {
                            "src": r.source_id,
                            "type": r.relation_type,
                            "tgt": r.target_id,
                        }
                        for r in relations[:30]
                    ],
                }
                return json.dumps(result, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": f"Explore error: {e}"})

        @tool
        async def find_connections(source_id: str, target_id: str, max_hops: int = 4) -> str:
            """Find shortest paths between two entities in the knowledge graph.
            Returns fact chains: verifiable sequences of entity-relation-entity
            triples that connect the two entities.
            Use this when the question asks about relationships or connections
            between two concepts (e.g. how law X relates to domain Y).
            Input: source_id and target_id (entity IDs), max_hops (1-6)."""
            try:
                hops = min(max(max_hops, 1), 6)
                paths = await neo4j.find_shortest_paths(
                    source_id, target_id, max_hops=hops,
                )
                if not paths:
                    # Try label search fallback
                    src_found = await neo4j.find_entities_by_label([source_id], limit=1)
                    tgt_found = await neo4j.find_entities_by_label([target_id], limit=1)
                    if src_found and tgt_found:
                        paths = await neo4j.find_shortest_paths(
                            src_found[0].id, tgt_found[0].id, max_hops=hops,
                        )

                if not paths:
                    return json.dumps({
                        "source": source_id,
                        "target": target_id,
                        "paths": [],
                        "message": "No paths found between these entities.",
                    })

                result_paths = []
                for path_ents, path_rels in paths:
                    chain_steps = []
                    for rel in path_rels:
                        chain_steps.append(
                            f"{rel.source_id} -[{rel.relation_type}]-> {rel.target_id}"
                        )
                    result_paths.append({
                        "nodes": [e.id for e in path_ents],
                        "node_labels": {e.id: e.label[:60] for e in path_ents},
                        "edges": [
                            {
                                "src": r.source_id,
                                "type": r.relation_type,
                                "tgt": r.target_id,
                            }
                            for r in path_rels
                        ],
                        "chain": " → ".join(chain_steps) if chain_steps else "direct",
                    })

                return json.dumps({
                    "source": source_id,
                    "target": target_id,
                    "paths": result_paths,
                }, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": f"Path finding error: {e}"})

        @tool
        async def lookup_ontology(class_name: str = "", relation_name: str = "") -> str:
            """Look up ontology information about classes and relations.
            Use this to understand what types exist, what relations connect them,
            and what subclasses a type has.
            Input: class_name (e.g. 'Facility', 'PlanningDomain') and/or
            relation_name (e.g. 'hasAction', 'governedBy')."""
            lines = []
            if class_name and class_name in ontology.classes:
                cls = ontology.classes[class_name]
                lines.append(f"Class: {cls.name}")
                if cls.parent:
                    lines.append(f"  Parent: {cls.parent}")
                if cls.children:
                    lines.append(f"  Subclasses: {', '.join(cls.children)}")
                if cls.properties_as_domain:
                    lines.append(f"  Outgoing relations: {', '.join(cls.properties_as_domain)}")
                if cls.properties_as_range:
                    lines.append(f"  Incoming relations: {', '.join(cls.properties_as_range)}")
                related = ontology.get_related_types(class_name)
                if related:
                    lines.append(f"  Connected types: {', '.join(related)}")
            elif class_name:
                lines.append(f"Class '{class_name}' not in ontology.")
                lines.append(f"Available: {', '.join(sorted(ontology.classes.keys()))}")

            if relation_name and relation_name in ontology.properties:
                prop = ontology.properties[relation_name]
                lines.append(f"Relation: {prop.name} ({prop.prop_type})")
                lines.append(f"  Domain: {prop.domain} -> Range: {prop.range}")
            elif relation_name:
                lines.append(f"Relation '{relation_name}' not in ontology.")
                lines.append(f"Available: {', '.join(sorted(ontology.properties.keys()))}")

            if not class_name and not relation_name:
                lines.append(
                    f"Classes ({len(ontology.classes)}): "
                    f"{', '.join(sorted(ontology.classes.keys()))}"
                )
                lines.append(
                    f"Relations ({len(ontology.properties)}): "
                    f"{', '.join(sorted(ontology.properties.keys()))}"
                )

            return "\n".join(lines) if lines else "No ontology info found."

        @tool
        async def aggregate_subgraph(entity_ids: list[str], max_hops: int = 1) -> str:
            """Get the merged induced subgraph for multiple entities at once.
            More efficient than calling explore_entity multiple times.
            Use this when you have identified several relevant entity IDs and
            want to see how they are all connected.
            Input: entity_ids (list of entity IDs), max_hops (1-2)."""
            try:
                hops = min(max(max_hops, 1), 2)
                all_ents, all_rels = await neo4j.get_neighbourhood(
                    entity_ids, max_hops=hops, max_nodes=60,
                )
                # Fallback: resolve labels to IDs
                if not all_ents and not all_rels:
                    found = await neo4j.find_entities_by_label(entity_ids, limit=10)
                    if found:
                        resolved_ids = [f.id for f in found]
                        all_ents, all_rels = await neo4j.get_neighbourhood(
                            resolved_ids, max_hops=hops, max_nodes=60,
                        )
                        if not all_ents:
                            all_ents = found

                if not all_ents and not all_rels:
                    return json.dumps({"error": f"No data found for {entity_ids}."})

                result = {
                    "seed_ids": entity_ids,
                    "hops": hops,
                    "total_entities": len(all_ents),
                    "total_relations": len(all_rels),
                    "entities": [
                        {"id": e.id, "label": e.label[:80], "type": e.entity_type}
                        for e in all_ents[:40]
                    ],
                    "relations": [
                        {"src": r.source_id, "type": r.relation_type, "tgt": r.target_id}
                        for r in all_rels[:60]
                    ],
                }
                return json.dumps(result, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": f"Aggregate subgraph error: {e}"})

        @tool
        async def semantic_search_entities(query: str, top_k: int = 5) -> str:
            """Search for KG entities by natural language description using embeddings.
            Unlike explore_entity (which needs an exact entity ID), this tool finds
            entities whose text matches your description semantically.
            Use when you don't know the exact entity ID or label.
            Input: query (natural language), top_k (1-10)."""
            try:
                k = min(max(top_k, 1), 10)
                embedding = await ollama.embed(query)
                results = await qdrant.search(query_vector=embedding, top_k=k)
                if not results:
                    return "No matching entities found."

                # Extract entity IDs from chunk metadata and look them up
                entity_labels: set[str] = set()
                for chunk, score in results:
                    content = chunk.content[:200]
                    entity_labels.add(content.split("\n")[0][:80])

                # Try to find actual KG entities matching these terms
                found = await neo4j.find_entities_by_label(
                    list(entity_labels)[:5], limit=k,
                )

                lines = []
                if found:
                    for ent in found:
                        lines.append(
                            f"  [{ent.entity_type}] {ent.id}: {ent.label[:80]}"
                        )
                else:
                    for chunk, score in results[:k]:
                        lines.append(f"  [score={score:.3f}] {chunk.content[:150]}")

                return f"Found {len(found) if found else len(results)} results:\n" + "\n".join(lines)
            except Exception as e:
                return f"Semantic entity search error: {e}"

        @tool
        async def compare_entities(entity_id_a: str, entity_id_b: str) -> str:
            """Compare two entities: show their types, shared relations, unique
            relations, and common neighbours. Use for comparative questions like
            'What is the difference between X and Y?' or 'How do X and Y relate?'.
            Input: entity_id_a and entity_id_b (entity IDs or labels)."""
            try:
                # Resolve both entities
                ids_a = [entity_id_a]
                ids_b = [entity_id_b]
                ents_a, rels_a = await neo4j.get_neighbourhood(ids_a, max_hops=1, max_nodes=30)
                ents_b, rels_b = await neo4j.get_neighbourhood(ids_b, max_hops=1, max_nodes=30)

                # Fallback via label search
                if not ents_a:
                    found = await neo4j.find_entities_by_label([entity_id_a], limit=1)
                    if found:
                        ents_a, rels_a = await neo4j.get_neighbourhood(
                            [found[0].id], max_hops=1, max_nodes=30,
                        )
                if not ents_b:
                    found = await neo4j.find_entities_by_label([entity_id_b], limit=1)
                    if found:
                        ents_b, rels_b = await neo4j.get_neighbourhood(
                            [found[0].id], max_hops=1, max_nodes=30,
                        )

                neighbour_ids_a = {e.id for e in ents_a}
                neighbour_ids_b = {e.id for e in ents_b}
                shared_neighbours = neighbour_ids_a & neighbour_ids_b
                unique_a = neighbour_ids_a - neighbour_ids_b
                unique_b = neighbour_ids_b - neighbour_ids_a

                rel_types_a = {r.relation_type for r in rels_a}
                rel_types_b = {r.relation_type for r in rels_b}
                shared_rel_types = rel_types_a & rel_types_b
                unique_rel_a = rel_types_a - rel_types_b
                unique_rel_b = rel_types_b - rel_types_a

                ent_map = {e.id: e for e in ents_a + ents_b}

                result = {
                    "entity_a": entity_id_a,
                    "entity_b": entity_id_b,
                    "a_neighbours": len(neighbour_ids_a),
                    "b_neighbours": len(neighbour_ids_b),
                    "shared_neighbours": [
                        {"id": nid, "label": ent_map.get(nid, KGEntity(id=nid, label=nid, entity_type="")).label[:60]}
                        for nid in list(shared_neighbours)[:10]
                    ],
                    "unique_to_a": [
                        {"id": nid, "label": ent_map.get(nid, KGEntity(id=nid, label=nid, entity_type="")).label[:60]}
                        for nid in list(unique_a)[:10]
                    ],
                    "unique_to_b": [
                        {"id": nid, "label": ent_map.get(nid, KGEntity(id=nid, label=nid, entity_type="")).label[:60]}
                        for nid in list(unique_b)[:10]
                    ],
                    "shared_relation_types": sorted(shared_rel_types),
                    "unique_relation_types_a": sorted(unique_rel_a),
                    "unique_relation_types_b": sorted(unique_rel_b),
                }
                return json.dumps(result, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": f"Compare error: {e}"})

        @tool
        async def count_and_aggregate(description: str) -> str:
            """Run a counting or aggregation query against the knowledge graph.
            Use for questions like 'How many facilities...?', 'List all laws that...',
            or 'What types of activities exist?'.
            Input: description of what to count or aggregate (natural language).
            The tool generates and runs an optimized Cypher query internally."""
            try:
                # Ask LLM to generate an aggregation Cypher query
                prompt = (
                    f"Generate a Cypher aggregation query for this request:\n"
                    f'"{description}"\n\n'
                    f"Rules:\n"
                    f"- Use MATCH, WITH, RETURN with count(), collect(), or UNWIND\n"
                    f"- Always include LIMIT 50\n"
                    f"- Return descriptive column names\n"
                    f"- Use labels and relationship types from the schema\n\n"
                    f"Return ONLY the Cypher query, nothing else."
                )
                cypher = await ollama.generate(
                    prompt=prompt,
                    system="You are a Cypher query expert. Return only valid Cypher.",
                    temperature=0.1,
                )
                # Clean: strip markdown fences
                cypher = cypher.strip().strip("`").strip()
                if cypher.lower().startswith("cypher"):
                    cypher = cypher[6:].strip()

                # Safety: block mutations
                lower = cypher.lower()
                if any(kw in lower for kw in ("create", "delete", "merge", "set ", "remove", "drop")):
                    return "Error: mutation queries are not allowed."

                from neo4j import AsyncGraphDatabase
                driver = AsyncGraphDatabase.driver(
                    neo4j_config.uri,
                    auth=(neo4j_config.user, neo4j_config.password),
                )
                async with driver.session(database=neo4j_config.database) as session:
                    result = await session.run(cypher)
                    records = [dict(r) async for r in result]
                await driver.close()

                if not records:
                    return f"Query returned 0 rows.\nCypher used: {cypher}"

                lines = [f"Cypher: {cypher}", f"{len(records)} results:"]
                for rec in records[:30]:
                    row = {k: (dict(v) if hasattr(v, "items") else v) for k, v in rec.items()}
                    lines.append(json.dumps(row, default=str, ensure_ascii=False))
                return "\n".join(lines)
            except Exception as e:
                return f"Aggregation error: {e}"

        @tool
        async def collect_evidence(summary: str) -> str:
            """Call this when you have gathered enough evidence to answer the question.
            Input: a brief summary of what evidence was collected and from which sources."""
            return "Evidence collection complete."

        return [
            search_vectors, query_graph, explore_entity,
            find_connections, lookup_ontology,
            aggregate_subgraph, semantic_search_entities,
            compare_entities, count_and_aggregate,
            collect_evidence,
        ]

    # -- Result parsing and finalization ------------------------------------

    def _parse_tool_result(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result_str: str,
    ) -> tuple[list[RetrievedContext], list[KGEntity], list[KGRelation], list[dict[str, Any]]]:
        """Parse a tool result into contexts, entities, relations, and fact chains."""
        if tool_name == "search_vectors":
            return self._parse_vector_result(result_str)
        elif tool_name == "query_graph":
            return self._parse_cypher_result(tool_args, result_str)
        elif tool_name == "explore_entity":
            return self._parse_explore_result(tool_args, result_str)
        elif tool_name == "find_connections":
            return self._parse_path_result(result_str)
        elif tool_name == "lookup_ontology":
            lookup_q = tool_args.get("class_name", "") or tool_args.get("relation_name", "")
            return self._parse_ontology_result(result_str, lookup_query=lookup_q)
        elif tool_name == "aggregate_subgraph":
            return self._parse_aggregate_result(tool_args, result_str)
        elif tool_name == "semantic_search_entities":
            return self._parse_semantic_entity_result(result_str)
        elif tool_name == "compare_entities":
            return self._parse_compare_result(tool_args, result_str)
        elif tool_name == "count_and_aggregate":
            return self._parse_aggregation_result(result_str)
        return [], [], [], []

    def _parse_vector_result(
        self, result_str: str,
    ) -> tuple[list[RetrievedContext], list[KGEntity], list[KGRelation], list[dict[str, Any]]]:
        """Parse vector search results."""
        contexts: list[RetrievedContext] = []
        chunks = result_str.split("\n---\n")
        for i, chunk in enumerate(chunks):
            if chunk.startswith("No vector") or chunk.startswith("Vector search error"):
                continue
            contexts.append(RetrievedContext(
                source=RetrievalSource.VECTOR,
                text=chunk,
                score=0.9 - (i * 0.05),
                provenance=Provenance(
                    retrieval_strategy="agentic_vector",
                    retrieval_score=0.9 - (i * 0.05),
                ),
            ))
        return contexts, [], [], []

    def _parse_cypher_result(
        self,
        tool_args: dict[str, Any],
        result_str: str,
    ) -> tuple[list[RetrievedContext], list[KGEntity], list[KGRelation], list[dict[str, Any]]]:
        """Parse Cypher query results — extract entities from JSON rows."""
        contexts: list[RetrievedContext] = []
        entities: list[KGEntity] = []

        if "rows:" not in result_str:
            return contexts, entities, [], []

        lines = result_str.split("\n")[1:]  # Skip header
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
                # Try multiple key patterns for entity extraction
                eid = row.get("id", row.get("n.id", row.get("e.id", "")))
                elabel = row.get("label", row.get("n.label", row.get("e.label", str(row)[:100])))
                etype = row.get("type", row.get("n.node_type", row.get("node_type", "")))

                if eid:
                    entities.append(KGEntity(
                        id=str(eid), label=str(elabel), entity_type=str(etype),
                        confidence=1.0,
                    ))

                contexts.append(RetrievedContext(
                    source=RetrievalSource.GRAPH,
                    text=f"{elabel} [{etype}]" if elabel else line[:200],
                    score=1.0 - (i * 0.02),
                    provenance=Provenance(
                        entity_ids=[str(eid)] if eid else [],
                        retrieval_strategy="agentic_cypher",
                        retrieval_score=1.0 - (i * 0.02),
                    ),
                ))
            except json.JSONDecodeError:
                contexts.append(RetrievedContext(
                    source=RetrievalSource.GRAPH,
                    text=line[:300],
                    score=0.5,
                    provenance=Provenance(
                        retrieval_strategy="agentic_cypher",
                        retrieval_score=0.5,
                    ),
                ))
        return contexts, entities, [], []

    def _parse_explore_result(
        self,
        tool_args: dict[str, Any],
        result_str: str,
    ) -> tuple[list[RetrievedContext], list[KGEntity], list[KGRelation], list[dict[str, Any]]]:
        """Parse explore_entity JSON results into proper entities and relations."""
        contexts: list[RetrievedContext] = []
        entities: list[KGEntity] = []
        relations: list[KGRelation] = []
        entity_id = tool_args.get("entity_id", "")

        try:
            data = json.loads(result_str)
        except json.JSONDecodeError:
            # Fallback for non-JSON results
            if "error" not in result_str.lower():
                contexts.append(RetrievedContext(
                    source=RetrievalSource.GRAPH,
                    text=result_str[:500],
                    score=0.6,
                    provenance=Provenance(
                        entity_ids=[entity_id] if entity_id else [],
                        retrieval_strategy="agentic_explore",
                        retrieval_score=0.6,
                    ),
                ))
            return contexts, entities, relations, []

        if "error" in data:
            return contexts, entities, relations, []

        # Parse entities from structured JSON
        for ent_dict in data.get("entities", []):
            eid = ent_dict.get("id", "")
            if eid:
                entities.append(KGEntity(
                    id=eid,
                    label=ent_dict.get("label", ""),
                    entity_type=ent_dict.get("type", ""),
                    confidence=0.9,
                ))

        # Parse relations from structured JSON
        for rel_dict in data.get("relations", []):
            src = rel_dict.get("src", "")
            tgt = rel_dict.get("tgt", "")
            rtype = rel_dict.get("type", "")
            if src and tgt and rtype:
                relations.append(KGRelation(
                    source_id=src,
                    target_id=tgt,
                    relation_type=rtype,
                    confidence=0.9,
                ))

        # Build a readable text for the context
        text_lines = [f"Neighborhood of '{entity_id}' ({data.get('hops', 1)}-hop):"]
        text_lines.append(f"Entities ({len(entities)}):")
        for e in entities[:15]:
            text_lines.append(f"  [{e.entity_type}] {e.id}: {e.label}")
        text_lines.append(f"Relations ({len(relations)}):")
        for r in relations[:20]:
            text_lines.append(f"  {r.source_id} -[{r.relation_type}]-> {r.target_id}")

        contexts.append(RetrievedContext(
            source=RetrievalSource.GRAPH,
            text="\n".join(text_lines),
            score=0.85,
            subgraph=[*entities[:30], *relations[:50]],
            provenance=Provenance(
                entity_ids=[entity_id] + [e.id for e in entities[:10]],
                retrieval_strategy="agentic_explore",
                retrieval_score=0.85,
            ),
        ))

        return contexts, entities, relations, []

    def _parse_path_result(
        self, result_str: str,
    ) -> tuple[list[RetrievedContext], list[KGEntity], list[KGRelation], list[dict[str, Any]]]:
        """Parse find_connections JSON results into fact chains with provenance.

        Each path becomes a verified fact chain: a sequence of
        entity-[relation]->entity triples that are directly grounded
        in the knowledge graph.
        """
        contexts: list[RetrievedContext] = []
        entities: list[KGEntity] = []
        relations: list[KGRelation] = []
        chains: list[dict[str, Any]] = []

        try:
            data = json.loads(result_str)
        except json.JSONDecodeError:
            return contexts, entities, relations, chains

        if "error" in data:
            return contexts, entities, relations, chains

        source = data.get("source", "")
        target = data.get("target", "")
        paths = data.get("paths", [])

        if not paths:
            return contexts, entities, relations, chains

        for path_idx, path in enumerate(paths):
            node_ids = path.get("nodes", [])
            node_labels = path.get("node_labels", {})
            edges = path.get("edges", [])
            chain_str = path.get("chain", "")

            # Collect entities from this path
            for nid in node_ids:
                lbl = node_labels.get(nid, nid)
                entities.append(KGEntity(
                    id=nid, label=lbl, entity_type="",
                    confidence=0.95,
                ))

            # Collect relations from this path
            for edge in edges:
                src = edge.get("src", "")
                tgt = edge.get("tgt", "")
                rtype = edge.get("type", "")
                if src and tgt and rtype:
                    relations.append(KGRelation(
                        source_id=src,
                        target_id=tgt,
                        relation_type=rtype,
                        confidence=0.95,
                    ))

            # Build the fact chain
            chain = {
                "source": source,
                "target": target,
                "path_index": path_idx,
                "node_ids": node_ids,
                "node_labels": node_labels,
                "edges": edges,
                "chain_text": chain_str,
            }
            chains.append(chain)

            # Create a high-priority context for this fact chain
            chain_text = f"FACT CHAIN ({source} -> {target}, path {path_idx + 1}):\n"
            chain_text += f"  {chain_str}\n"
            chain_text += f"  Nodes: {', '.join(f'{nid} ({node_labels.get(nid, nid)})' for nid in node_ids)}\n"
            for edge in edges:
                chain_text += f"  TRIPLE: [{edge.get('src')}] -[{edge.get('type')}]-> [{edge.get('tgt')}]\n"

            contexts.append(RetrievedContext(
                source=RetrievalSource.GRAPH,
                text=chain_text,
                score=0.95 - (path_idx * 0.02),
                subgraph=[*entities, *relations],
                provenance=Provenance(
                    entity_ids=node_ids,
                    retrieval_strategy="agentic_fact_chain",
                    retrieval_score=0.95,
                ),
            ))

        return contexts, entities, relations, chains

    def _parse_ontology_result(
        self, result_str: str, lookup_query: str = "",
    ) -> tuple[list[RetrievedContext], list[KGEntity], list[KGRelation], list[dict[str, Any]]]:
        """Parse ontology lookup results and detect gaps."""
        contexts: list[RetrievedContext] = []
        if not result_str.startswith("No ontology"):
            contexts.append(RetrievedContext(
                source=RetrievalSource.ONTOLOGY,
                text=result_str,
                score=0.3,
                provenance=Provenance(
                    retrieval_strategy="agentic_ontology",
                    retrieval_score=0.3,
                ),
            ))
        else:
            # Active ontology learning: detect gap
            gap = self._gap_detector.detect_from_failed_lookup(
                lookup_query or "unknown", result_str,
            )
            if gap:
                self._detected_gaps.append(gap)
                logger.info(
                    "agentic.ontology_gap_detected",
                    gap_type=gap.gap_type,
                    query=gap.query_context,
                )
        return contexts, [], [], []

    def _parse_aggregate_result(
        self,
        tool_args: dict[str, Any],
        result_str: str,
    ) -> tuple[list[RetrievedContext], list[KGEntity], list[KGRelation], list[dict[str, Any]]]:
        """Parse aggregate_subgraph JSON results."""
        contexts: list[RetrievedContext] = []
        entities: list[KGEntity] = []
        relations: list[KGRelation] = []

        try:
            data = json.loads(result_str)
        except json.JSONDecodeError:
            if "error" not in result_str.lower():
                contexts.append(RetrievedContext(
                    source=RetrievalSource.GRAPH, text=result_str[:500], score=0.7,
                    provenance=Provenance(retrieval_strategy="agentic_aggregate", retrieval_score=0.7),
                ))
            return contexts, entities, relations, []

        if "error" in data:
            return contexts, entities, relations, []

        for ent_dict in data.get("entities", []):
            eid = ent_dict.get("id", "")
            if eid:
                entities.append(KGEntity(
                    id=eid, label=ent_dict.get("label", ""),
                    entity_type=ent_dict.get("type", ""), confidence=0.9,
                ))
        for rel_dict in data.get("relations", []):
            src, tgt, rtype = rel_dict.get("src", ""), rel_dict.get("tgt", ""), rel_dict.get("type", "")
            if src and tgt and rtype:
                relations.append(KGRelation(source_id=src, target_id=tgt, relation_type=rtype, confidence=0.9))

        seed_ids = data.get("seed_ids", [])
        text_lines = [
            f"Merged subgraph for {seed_ids} ({data.get('hops', 1)}-hop):",
            f"  {data.get('total_entities', 0)} entities, {data.get('total_relations', 0)} relations",
        ]
        for e in entities[:15]:
            text_lines.append(f"  [{e.entity_type}] {e.id}: {e.label}")

        contexts.append(RetrievedContext(
            source=RetrievalSource.GRAPH, text="\n".join(text_lines), score=0.88,
            subgraph=[*entities[:30], *relations[:50]],
            provenance=Provenance(
                entity_ids=[e.id for e in entities[:15]],
                retrieval_strategy="agentic_aggregate", retrieval_score=0.88,
            ),
        ))
        return contexts, entities, relations, []

    def _parse_semantic_entity_result(
        self, result_str: str,
    ) -> tuple[list[RetrievedContext], list[KGEntity], list[KGRelation], list[dict[str, Any]]]:
        """Parse semantic_search_entities results."""
        contexts: list[RetrievedContext] = []
        if result_str.startswith("No matching") or result_str.startswith("Semantic entity search error"):
            return contexts, [], [], []
        contexts.append(RetrievedContext(
            source=RetrievalSource.VECTOR, text=result_str, score=0.7,
            provenance=Provenance(retrieval_strategy="agentic_semantic_entity", retrieval_score=0.7),
        ))
        return contexts, [], [], []

    def _parse_compare_result(
        self,
        tool_args: dict[str, Any],
        result_str: str,
    ) -> tuple[list[RetrievedContext], list[KGEntity], list[KGRelation], list[dict[str, Any]]]:
        """Parse compare_entities JSON results."""
        contexts: list[RetrievedContext] = []
        try:
            data = json.loads(result_str)
        except json.JSONDecodeError:
            if "error" not in result_str.lower():
                contexts.append(RetrievedContext(
                    source=RetrievalSource.GRAPH, text=result_str[:500], score=0.7,
                    provenance=Provenance(retrieval_strategy="agentic_compare", retrieval_score=0.7),
                ))
            return contexts, [], [], []

        if "error" in data:
            return contexts, [], [], []

        a = data.get("entity_a", "")
        b = data.get("entity_b", "")
        lines = [
            f"Comparison of '{a}' vs '{b}':",
            f"  {a} has {data.get('a_neighbours', 0)} neighbours, {b} has {data.get('b_neighbours', 0)} neighbours",
            f"  Shared neighbours ({len(data.get('shared_neighbours', []))}):",
        ]
        for sn in data.get("shared_neighbours", [])[:5]:
            lines.append(f"    - {sn.get('id', '')}: {sn.get('label', '')}")
        lines.append(f"  Shared relation types: {', '.join(data.get('shared_relation_types', []))}")
        lines.append(f"  Unique to {a}: {', '.join(data.get('unique_relation_types_a', []))}")
        lines.append(f"  Unique to {b}: {', '.join(data.get('unique_relation_types_b', []))}")

        contexts.append(RetrievedContext(
            source=RetrievalSource.GRAPH, text="\n".join(lines), score=0.9,
            provenance=Provenance(
                entity_ids=[a, b], retrieval_strategy="agentic_compare", retrieval_score=0.9,
            ),
        ))
        return contexts, [], [], []

    def _parse_aggregation_result(
        self, result_str: str,
    ) -> tuple[list[RetrievedContext], list[KGEntity], list[KGRelation], list[dict[str, Any]]]:
        """Parse count_and_aggregate results."""
        contexts: list[RetrievedContext] = []
        if result_str.startswith("Aggregation error") or result_str.startswith("Error:"):
            return contexts, [], [], []
        contexts.append(RetrievedContext(
            source=RetrievalSource.GRAPH, text=result_str, score=0.85,
            provenance=Provenance(retrieval_strategy="agentic_aggregation", retrieval_score=0.85),
        ))
        return contexts, [], [], []

    def _finalize(
        self,
        contexts: list[RetrievedContext],
        entities: list[KGEntity],
        relations: list[KGRelation],
        fact_chains: list[dict[str, Any]] | None = None,
        tool_trace: list[dict[str, Any]] | None = None,
    ) -> list[RetrievedContext]:
        """Deduplicate, sort, and attach subgraph + fact chains to contexts.

        Also stores ``fact_chains`` and ``tool_trace`` on the first context's
        provenance so the orchestrator can forward them to ``QAAnswer``.
        """
        fact_chains = fact_chains or []
        tool_trace = tool_trace or []

        # Inject a graph reasoning summary context if we have fact chains
        if fact_chains:
            summary_lines = ["GRAPH REASONING — Verified fact chains from knowledge graph:"]
            for i, chain in enumerate(fact_chains, 1):
                summary_lines.append(f"\n  Chain {i}: {chain.get('source', '?')} -> {chain.get('target', '?')}")
                summary_lines.append(f"    Path: {chain.get('chain_text', 'n/a')}")
                for edge in chain.get("edges", []):
                    summary_lines.append(
                        f"    GROUNDED TRIPLE: [{edge.get('src')}] "
                        f"-[{edge.get('type')}]-> [{edge.get('tgt')}]"
                    )
                node_labels = chain.get("node_labels", {})
                if node_labels:
                    label_parts = [f"{k}={v}" for k, v in node_labels.items()]
                    summary_lines.append(f"    Labels: {', '.join(label_parts)}")

            all_chain_ids = []
            for c in fact_chains:
                all_chain_ids.extend(c.get("node_ids", []))

            contexts.append(RetrievedContext(
                source=RetrievalSource.GRAPH,
                text="\n".join(summary_lines),
                score=0.98,
                provenance=Provenance(
                    entity_ids=list(set(all_chain_ids)),
                    retrieval_strategy="agentic_graph_reasoning",
                    retrieval_score=0.98,
                ),
            ))

        # Deduplicate by text prefix
        seen: set[str] = set()
        unique: list[RetrievedContext] = []
        for ctx in contexts:
            key = ctx.text[:150]
            if key not in seen:
                seen.add(key)
                unique.append(ctx)

        # Sort by score descending
        unique.sort(key=lambda c: c.score, reverse=True)

        # Deduplicate entities
        seen_ids: set[str] = set()
        unique_ents: list[KGEntity] = []
        for e in entities:
            if e.id and e.id not in seen_ids:
                seen_ids.add(e.id)
                unique_ents.append(e)

        # Attach full subgraph to top context
        if unique and (unique_ents or relations):
            unique[0].subgraph = [*unique_ents[:50], *relations[:80]]

        # Mark all as hybrid source
        for ctx in unique:
            ctx.source = RetrievalSource.HYBRID

        # Store fact_chains and tool_trace on the instance so the orchestrator
        # can forward them to QAAnswer after retrieval completes.
        self._last_fact_chains = fact_chains
        self._last_tool_trace = tool_trace

        logger.info(
            "agentic.finalized",
            total_contexts=len(unique),
            entities=len(unique_ents),
            relations=len(relations),
            fact_chains=len(fact_chains),
            tool_calls=len(tool_trace),
        )
        return unique[:_MAX_CONTEXTS]

    # -- Fallback (when tool-binding fails) ---------------------------------

    async def _fallback_retrieve(self, query: QAQuery) -> list[RetrievedContext]:
        """Simple parallel vector+cypher fallback when agent can't bind tools."""
        from kgrag.retrieval.vector import VectorRetriever
        from kgrag.retrieval.cypher import CypherRetriever

        logger.info("agentic.fallback", reason="tool binding failed")

        vector = VectorRetriever(self._qdrant, self._ollama, self._config)
        cypher_r = CypherRetriever(self._neo4j_config, self._ollama, self._config)

        vector_task = asyncio.create_task(vector.retrieve(query))
        cypher_task = asyncio.create_task(cypher_r.retrieve(query))

        vector_ctx = await vector_task
        cypher_ctx = await cypher_task

        combined = cypher_ctx + vector_ctx
        for ctx in combined:
            ctx.source = RetrievalSource.HYBRID
        return combined[:_MAX_CONTEXTS]
