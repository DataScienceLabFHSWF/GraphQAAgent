"""AgenticGraphRAG — LangGraph ReAct agent that orchestrates all retrieval tools.

This is the "everything working together in an agentic/tooly fashion" retriever.
It uses an LLM agent with tool-calling to dynamically decide:
- When to search vectors (semantic similarity over document chunks)
- When to generate and run Cypher (structured graph queries)
- When to explore multi-hop neighborhoods (entity-centric traversal)
- When to consult the ontology (what types/relations exist)
- When it has enough context to stop

The agent is ontology-informed: the full TBox schema is injected into its
system prompt so it understands what classes, properties, and hierarchies
exist in the knowledge graph.

Implements the ``Retriever`` protocol.
"""

from __future__ import annotations

import json
import asyncio
import structlog
from typing import Any, Annotated

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel

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
from kgrag.retrieval.ontology_context import OntologyContext

logger = structlog.get_logger(__name__)

# Max iterations for the agent loop to prevent infinite loops
_MAX_ITERATIONS = 6
# Max total contexts to collect before stopping
_MAX_CONTEXTS = 20


def _build_system_prompt(ontology_summary: str, neo4j_schema: str) -> str:
    """Build the system prompt with full ontology + graph schema awareness."""
    return f"""\
You are a knowledge graph retrieval agent for a nuclear decommissioning domain.
Your job is to gather comprehensive evidence from multiple sources to answer the user's question.
You have access to tools for searching a vector store, querying a Neo4j knowledge graph via Cypher,
exploring entity neighborhoods, and looking up ontology information.

{ontology_summary}

NEO4J GRAPH SCHEMA (actual database):
{neo4j_schema}

IMPORTANT DATA MODEL NOTES:
- Every node has: id (short unique id/abbreviation), label (full name), node_type, properties (JSON string)
- Law graph: Gesetzbuch (law books like AtG, StrlSchG) <-[:teilVon]- Paragraf -[:referenziert]-> Paragraf
- Domain entities connect to laws via: entity -[:LINKED_GOVERNED_BY]-> Paragraf
- The data is primarily in GERMAN. Key translations:
  Stilllegung=decommissioning, Abbau=dismantling, Kernkraftwerk=nuclear power plant,
  Genehmigung=permit, Strahlenschutz=radiation protection, Anlage=facility

STRATEGY:
1. Start by searching vectors for semantic context about the question
2. If the question mentions specific entities (laws, facilities, domains), query the graph via Cypher
3. For relationship questions, explore multi-hop neighborhoods
4. Use the ontology tool to understand what types/relations exist before writing complex Cypher
5. Collect evidence from multiple sources for comprehensive answers
6. When you have enough evidence (3-10 pieces), call collect_evidence to finalize

Think step by step about which tools to use. DO NOT answer the question — just gather evidence.
Respond ONLY with tool calls. When done gathering, call collect_evidence with all findings."""


class AgenticGraphRAG:
    """LangGraph-style ReAct agent combining all retrieval strategies.

    Tools:
    - ``search_vectors``: Semantic search over document chunks
    - ``query_graph``: Run Cypher queries against Neo4j
    - ``explore_entity``: Multi-hop neighborhood expansion
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
    ) -> None:
        self._neo4j = neo4j
        self._neo4j_config = neo4j_config
        self._qdrant = qdrant
        self._ollama = ollama
        self._ontology = ontology_context
        self._config = config
        self._neo4j_graph = None  # Lazy-loaded Neo4jGraph for schema

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

    # -- Retriever protocol -------------------------------------------------

    async def retrieve(self, query: QAQuery) -> list[RetrievedContext]:
        """Run the agentic retrieval loop."""
        llm = self._ollama.get_chat_model()

        # Build tools (closures that capture self)
        tools = self._build_tools()
        tool_map = {t.name: t for t in tools}

        # Bind tools to LLM
        try:
            llm_with_tools = llm.bind_tools(tools)
        except Exception as exc:
            logger.warning("agentic.bind_tools_failed", error=str(exc))
            # Fallback: just do vector + cypher in parallel
            return await self._fallback_retrieve(query)

        # System prompt with ontology + schema
        neo4j_schema = self._get_neo4j_schema()
        system = SystemMessage(
            content=_build_system_prompt(self._ontology.schema_summary, neo4j_schema)
        )
        human = HumanMessage(content=query.raw_question)

        messages = [system, human]
        collected_contexts: list[RetrievedContext] = []
        collected_entities: list[KGEntity] = []
        collected_relations: list[KGRelation] = []

        for iteration in range(1, _MAX_ITERATIONS + 1):
            logger.info("agentic.iteration", iteration=iteration)

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

                logger.info("agentic.tool_call", tool=tool_name, args_keys=list(tool_args.keys()))

                if tool_name == "collect_evidence":
                    # Terminal tool — agent is done
                    summary = tool_args.get("summary", "")
                    logger.info("agentic.done", summary=summary[:100], contexts=len(collected_contexts))
                    messages.append(ToolMessage(content="Evidence collected.", tool_call_id=tool_id))
                    # Return what we have
                    return self._finalize(collected_contexts, collected_entities, collected_relations)

                if tool_name in tool_map:
                    try:
                        result = await tool_map[tool_name].ainvoke(tool_args)
                        result_str = result if isinstance(result, str) else json.dumps(result, default=str)

                        # Parse results and accumulate contexts
                        new_ctxs, new_ents, new_rels = self._parse_tool_result(
                            tool_name, tool_args, result_str
                        )
                        collected_contexts.extend(new_ctxs)
                        collected_entities.extend(new_ents)
                        collected_relations.extend(new_rels)

                        # Truncate result for message history
                        messages.append(ToolMessage(
                            content=result_str[:3000],
                            tool_call_id=tool_id,
                        ))
                    except Exception as exc:
                        logger.warning("agentic.tool_error", tool=tool_name, error=str(exc))
                        messages.append(ToolMessage(
                            content=f"Error: {str(exc)[:200]}",
                            tool_call_id=tool_id,
                        ))
                else:
                    messages.append(ToolMessage(
                        content=f"Unknown tool: {tool_name}",
                        tool_call_id=tool_id,
                    ))

            # Check if we have enough
            if len(collected_contexts) >= _MAX_CONTEXTS:
                logger.info("agentic.max_contexts_reached", contexts=len(collected_contexts))
                break

        return self._finalize(collected_contexts, collected_entities, collected_relations)

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

                # Serialize, handling Neo4j types
                lines = []
                for rec in records[:25]:
                    row = {}
                    for k, v in rec.items():
                        if hasattr(v, "items"):  # Node
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
            Returns connected entities and relationships (multi-hop).
            Use this to find what's connected to a specific entity.
            Input: entity_id (e.g. 'AtG', 'ent_rule_0000'), max_hops (1-3)."""
            try:
                hops = min(max(max_hops, 1), 3)
                entities, relations = await neo4j.get_neighbourhood(
                    [entity_id], max_hops=hops, max_nodes=30,
                )
                if not entities and not relations:
                    return f"No neighbors found for '{entity_id}'."

                lines = [f"Neighborhood of '{entity_id}' ({hops}-hop):"]
                lines.append(f"Entities ({len(entities)}):")
                for e in entities[:15]:
                    lines.append(f"  [{e.entity_type}] {e.id}: {e.label[:60]}")
                lines.append(f"Relations ({len(relations)}):")
                for r in relations[:20]:
                    lines.append(f"  {r.source_id} -[{r.relation_type}]-> {r.target_id}")
                return "\n".join(lines)
            except Exception as e:
                return f"Explore error: {e}"

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
                lines.append(f"  Domain: {prop.domain} → Range: {prop.range}")
            elif relation_name:
                lines.append(f"Relation '{relation_name}' not in ontology.")
                lines.append(f"Available: {', '.join(sorted(ontology.properties.keys()))}")

            if not class_name and not relation_name:
                lines.append(f"Classes ({len(ontology.classes)}): {', '.join(sorted(ontology.classes.keys()))}")
                lines.append(f"Relations ({len(ontology.properties)}): {', '.join(sorted(ontology.properties.keys()))}")

            return "\n".join(lines) if lines else "No ontology info found."

        @tool
        async def collect_evidence(summary: str) -> str:
            """Call this when you have gathered enough evidence to answer the question.
            Input: a brief summary of what evidence was collected and from which sources."""
            return "Evidence collection complete."

        return [search_vectors, query_graph, explore_entity, lookup_ontology, collect_evidence]

    # -- Result parsing and finalization ------------------------------------

    def _parse_tool_result(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result_str: str,
    ) -> tuple[list[RetrievedContext], list[KGEntity], list[KGRelation]]:
        """Parse a tool result into RetrievedContexts."""
        contexts: list[RetrievedContext] = []
        entities: list[KGEntity] = []
        relations: list[KGRelation] = []

        if tool_name == "search_vectors":
            # Split by separator and create one context per chunk
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

        elif tool_name == "query_graph":
            # Parse Cypher results
            cypher = tool_args.get("cypher", "")
            if "rows:" in result_str:
                lines = result_str.split("\n")[1:]  # Skip header
                for i, line in enumerate(lines):
                    try:
                        row = json.loads(line)
                        eid = row.get("id", row.get("n.id", ""))
                        elabel = row.get("label", row.get("n.label", str(row)[:100]))
                        etype = row.get("type", row.get("n.node_type", ""))

                        if eid:
                            entities.append(KGEntity(
                                id=eid, label=str(elabel), entity_type=str(etype),
                                confidence=1.0,
                            ))

                        contexts.append(RetrievedContext(
                            source=RetrievalSource.GRAPH,
                            text=f"{elabel} [{etype}]" if elabel else line[:200],
                            score=1.0 - (i * 0.02),
                            provenance=Provenance(
                                entity_ids=[eid] if eid else [],
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

        elif tool_name == "explore_entity":
            entity_id = tool_args.get("entity_id", "")
            # The whole exploration result as one context
            if not result_str.startswith("No neighbors") and not result_str.startswith("Explore error"):
                # Parse entities and relations from the text
                for line in result_str.split("\n"):
                    line = line.strip()
                    if line.startswith("[") and "]" in line:
                        # Entity line: [Type] id: label
                        try:
                            etype = line[1:line.index("]")]
                            rest = line[line.index("]")+2:]
                            eid, elabel = rest.split(": ", 1)
                            entities.append(KGEntity(
                                id=eid.strip(), label=elabel.strip(),
                                entity_type=etype, confidence=0.9,
                            ))
                        except (ValueError, IndexError):
                            pass
                    elif "-[" in line and "]->" in line:
                        # Relation line: src -[type]-> tgt
                        try:
                            src = line.split(" -[")[0].strip()
                            rtype = line.split("-[")[1].split("]->")[0]
                            tgt = line.split("]-> ")[1].strip()
                            relations.append(KGRelation(
                                source_id=src, target_id=tgt,
                                relation_type=rtype, confidence=0.9,
                            ))
                        except (ValueError, IndexError):
                            pass

                contexts.append(RetrievedContext(
                    source=RetrievalSource.GRAPH,
                    text=result_str,
                    score=0.85,
                    subgraph=[*entities, *relations],
                    provenance=Provenance(
                        entity_ids=[entity_id] + [e.id for e in entities[:10]],
                        retrieval_strategy="agentic_explore",
                        retrieval_score=0.85,
                    ),
                ))

        elif tool_name == "lookup_ontology":
            # Ontology info as a low-priority context
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

        return contexts, entities, relations

    def _finalize(
        self,
        contexts: list[RetrievedContext],
        entities: list[KGEntity],
        relations: list[KGRelation],
    ) -> list[RetrievedContext]:
        """Deduplicate, sort, and attach subgraph to all contexts."""
        # Deduplicate by text prefix
        seen: set[str] = set()
        unique: list[RetrievedContext] = []
        for ctx in contexts:
            key = ctx.text[:150]
            if key not in seen:
                seen.add(key)
                unique.append(ctx)

        # Sort by score
        unique.sort(key=lambda c: c.score, reverse=True)

        # Attach full subgraph to top context
        if unique and (entities or relations):
            # Deduplicate entities
            seen_ids: set[str] = set()
            unique_ents: list[KGEntity] = []
            for e in entities:
                if e.id not in seen_ids:
                    seen_ids.add(e.id)
                    unique_ents.append(e)

            unique[0].subgraph = [*unique_ents[:30], *relations[:50]]

        # Mark all as hybrid source
        for ctx in unique:
            ctx.source = RetrievalSource.HYBRID

        logger.info(
            "agentic.finalized",
            total_contexts=len(unique),
            entities=len(entities),
            relations=len(relations),
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
