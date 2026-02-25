"""Cypher-based retriever using LangChain's GraphCypherQAChain.

Translates natural-language questions into Cypher via an LLM, executes them
against Neo4j, and converts the rows back into ``RetrievedContext`` objects
that the rest of the pipeline understands.

This bypasses our hand-rolled entity-linker entirely — the LLM generates
targeted Cypher that can do fuzzy lookups, multi-hop traversals, aggregation,
etc.
"""

from __future__ import annotations

import json
import structlog
from typing import Any

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

from kgrag.connectors.langchain_ollama_provider import LangChainOllamaProvider
from kgrag.core.config import Neo4jConfig, RetrievalConfig
from kgrag.core.models import (
    KGEntity,
    KGRelation,
    Provenance,
    QAQuery,
    RetrievalSource,
    RetrievedContext,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Domain-aware Cypher generation prompt
# ---------------------------------------------------------------------------

_CYPHER_GENERATION_TEMPLATE = """\
You are a Neo4j Cypher expert for a nuclear decommissioning knowledge graph.
Generate ONLY valid Cypher. No explanation, no markdown fences.

Schema:
{schema}

CRITICAL DATA MODEL RULES:
- Every node has: id (unique short name / abbreviation), label (full name),
  node_type, properties (JSON string with description, aliases, confidence).
- To find a node by name or abbreviation: WHERE n.id = 'name'
  or WHERE toLower(n.label) CONTAINS toLower('search term')
  or WHERE toLower(n.properties) CONTAINS toLower('term')
- Node labels (types) include: Gesetzbuch (law book), Paragraf (paragraph/section),
  Facility, PlanningDomain, DomainConstant, Action, State, Organization, etc.
- Key relationships:
  * teilVon (part of): Paragraf -[:teilVon]-> Gesetzbuch
  * referenziert (references): Paragraf -[:referenziert]-> Paragraf
  * LINKED_GOVERNED_BY: domain entities -[:LINKED_GOVERNED_BY]-> Paragraf
  * hasAction, hasPredicate, hasConstant, hasRequirement, hasGoalState, etc.
- The data is primarily in GERMAN. Common German terms:
  * Stilllegung = decommissioning, Abbau = dismantling
  * Kernkraftwerk (KKW) = nuclear power plant, Anlage = facility/plant
  * Genehmigung = permit/license, Strahlenschutz = radiation protection
  * AtG = Atomgesetz (Atomic Energy Act), StrlSchG = Strahlenschutzgesetz
  * BBergG = Bundesberggesetz, BImSchG = Bundes-Immissionsschutzgesetz
  * KrWG = Kreislaufwirtschaftsgesetz
- Always LIMIT results to at most 25 rows.
- Prefer returning id, label, and properties over returning full nodes.

Question: {question}

Cypher:"""

_CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=_CYPHER_GENERATION_TEMPLATE,
)

_QA_GENERATION_TEMPLATE = """\
You are a helpful assistant for nuclear decommissioning and regulatory questions.
Use ONLY the following context from a Neo4j knowledge graph to answer.
If the context is empty or insufficient, say you don't have enough information.
Answer in the same language as the question.

Context from knowledge graph:
{context}

Question: {question}

Answer:"""

_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=_QA_GENERATION_TEMPLATE,
)


class CypherRetriever:
    """LLM-to-Cypher retriever — generates and executes Cypher queries.

    Implements the ``Retriever`` protocol from ``kgrag.core.protocols``.
    """

    def __init__(
        self,
        neo4j_config: Neo4jConfig,
        ollama: LangChainOllamaProvider,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._neo4j_config = neo4j_config
        self._ollama = ollama
        self._config = config
        self._graph: Neo4jGraph | None = None
        self._chain: GraphCypherQAChain | None = None

    # -- lazy init (called once, after connectors are ready) ----------------

    def _ensure_chain(self) -> GraphCypherQAChain:
        """Build the LangChain chain lazily on first use."""
        if self._chain is not None:
            return self._chain

        self._graph = Neo4jGraph(
            url=self._neo4j_config.uri,
            username=self._neo4j_config.user,
            password=self._neo4j_config.password,
            database=self._neo4j_config.database,
            enhanced_schema=False,
        )

        llm = self._ollama.get_chat_model()

        self._chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=self._graph,
            verbose=False,
            allow_dangerous_requests=True,
            return_intermediate_steps=True,
            cypher_prompt=_CYPHER_PROMPT,
            qa_prompt=_QA_PROMPT,
            validate_cypher=True,
        )
        logger.info("cypher_retriever.chain_ready")
        return self._chain

    # -- Retriever protocol -------------------------------------------------

    async def retrieve(self, query: QAQuery) -> list[RetrievedContext]:
        """Generate Cypher, execute, and wrap results as RetrievedContext."""
        chain = self._ensure_chain()

        try:
            result = await chain.ainvoke({"query": query.raw_question})
        except Exception as exc:
            logger.warning("cypher_retriever.chain_error", error=str(exc))
            return []

        steps = result.get("intermediate_steps", [])
        cypher_query = steps[0].get("query", "") if steps else ""
        context_rows: list[dict[str, Any]] = (
            steps[1].get("context", []) if len(steps) > 1 else []
        )
        answer_text = result.get("result", "")

        logger.info(
            "cypher_retriever.executed",
            cypher=cypher_query[:200],
            rows=len(context_rows),
            answer_len=len(answer_text),
        )

        if not context_rows:
            return []

        # Convert each row into a RetrievedContext with subgraph elements
        contexts: list[RetrievedContext] = []
        entities: list[KGEntity] = []
        relations: list[KGRelation] = []

        for i, row in enumerate(context_rows):
            # Build a human-readable text snippet from the row
            text_parts = []
            entity_id = row.get("id", row.get("n.id", ""))
            entity_label = row.get("label", row.get("n.label", ""))
            entity_type = row.get("type", row.get("n.node_type", ""))
            props_raw = row.get("properties", row.get("n.properties", ""))

            # Try to parse properties JSON
            description = ""
            if isinstance(props_raw, str) and props_raw:
                try:
                    props = json.loads(props_raw)
                    description = props.get("description", "")
                except (json.JSONDecodeError, TypeError):
                    description = props_raw[:200]

            if entity_label:
                text_parts.append(f"{entity_label}")
            if entity_type:
                text_parts.append(f"[{entity_type}]")
            if description:
                text_parts.append(f"— {description}")
            if not text_parts:
                # Fallback: render the entire row as text
                text_parts.append(str(row))

            text = " ".join(text_parts)

            # Create KGEntity for subgraph
            if entity_id:
                entity = KGEntity(
                    id=entity_id,
                    label=entity_label or entity_id,
                    entity_type=entity_type or "Unknown",
                    description=description,
                    confidence=1.0,
                )
                entities.append(entity)

            ctx = RetrievedContext(
                source=RetrievalSource.GRAPH,
                text=text,
                score=1.0 - (i * 0.02),  # Rank-based score decay
                subgraph=[*entities[-1:], *relations],  # Attach this entity
                provenance=Provenance(
                    entity_ids=[entity_id] if entity_id else [],
                    retrieval_strategy="cypher",
                    retrieval_score=1.0 - (i * 0.02),
                ),
            )
            contexts.append(ctx)

        # Append the LLM's synthesised answer as an extra context piece
        if answer_text and answer_text.lower() not in ("i don't know the answer.", ""):
            summary_ctx = RetrievedContext(
                source=RetrievalSource.GRAPH,
                text=f"[Graph QA Summary] {answer_text}",
                score=1.0,
                provenance=Provenance(
                    retrieval_strategy="cypher_qa",
                    retrieval_score=1.0,
                    entity_ids=[e.id for e in entities[:10]],
                ),
            )
            contexts.insert(0, summary_ctx)

        logger.info(
            "cypher_retriever.done",
            contexts=len(contexts),
            entities=len(entities),
            cypher_preview=cypher_query[:120],
        )
        return contexts

    # -- direct Cypher execution (for advanced use) -------------------------

    async def run_cypher(self, question: str) -> dict[str, Any]:
        """Run the full chain and return raw result + intermediate steps."""
        chain = self._ensure_chain()
        try:
            result = await chain.ainvoke({"query": question})
            steps = result.get("intermediate_steps", [])
            return {
                "answer": result.get("result", ""),
                "cypher": steps[0].get("query", "") if steps else "",
                "context": steps[1].get("context", []) if len(steps) > 1 else [],
            }
        except Exception as exc:
            logger.error("cypher_retriever.run_cypher_error", error=str(exc))
            return {"answer": "", "cypher": "", "context": [], "error": str(exc)}
