"""Orchestrator (C3.4.5) — SOTA multi-step QA pipeline orchestration.

Wires together all components into a complete GraphQA agent:
- **Question parsing** + ontology expansion
- **Three-way hybrid retrieval** (vector + graph + Think-on-Graph)
- **Chain-of-Thought reasoning** (multi-hop decomposition)
- **Answer generation** with KG-grounded context
- **Answer verification** (faithfulness + entity coverage)
- **Explainer** with full reasoning DAG + subgraph visualisation

Strategies:
- ``"vector_only"``: pure vector retrieval (baseline)
- ``"graph_only"``: graph-only retrieval (baseline)
- ``"hybrid"``: two-way vector+graph fusion (standard)
- ``"hybrid_sota"``: full SOTA pipeline with ToG + CoT + verification
"""

from __future__ import annotations

import time

import structlog

from kgrag.agents.answer_generator import AnswerGenerator
from kgrag.agents.answer_verifier import AnswerVerifier
from kgrag.agents.chain_of_thought import ChainOfThoughtReasoner
from kgrag.agents.context_assembler import ContextAssembler
from kgrag.agents.explainer import Explainer
from kgrag.agents.question_parser import QuestionParser
from kgrag.adapters.agentic_reasoner_adapter import AgenticReasonerAdapter
from kgrag.connectors.fuseki import FusekiConnector
from kgrag.connectors.neo4j import Neo4jConnector
from kgrag.connectors.langchain_ollama_provider import LangChainOllamaProvider
from kgrag.connectors.qdrant import QdrantConnector
from kgrag.core.config import Settings
from kgrag.core.models import QAAnswer, RetrievedContext
from kgrag.core.protocols import Retriever
from kgrag.retrieval.entity_linker import EntityLinker
from kgrag.retrieval.graph import GraphRetriever
from kgrag.retrieval.graph_reasoning import GraphReasoner
from kgrag.retrieval.hybrid import HybridRetriever
from kgrag.retrieval.ontology import OntologyRetriever
from kgrag.retrieval.path_ranker import PathRanker
from kgrag.retrieval.reranker import CrossEncoderReranker
from kgrag.retrieval.vector import VectorRetriever

logger = structlog.get_logger(__name__)


class Orchestrator:
    """End-to-end SOTA QA pipeline.

    ``hybrid_sota`` pipeline:
    parse → expand → retrieve (3-way) → CoT reason → generate → verify → explain
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        # Connectors
        self.neo4j = Neo4jConnector(settings.neo4j)
        self.qdrant = QdrantConnector(settings.qdrant)
        self.fuseki = FusekiConnector(settings.fuseki)
        self.ollama = LangChainOllamaProvider(settings.ollama)

        # Core retrieval components
        self.entity_linker = EntityLinker(self.neo4j, self.qdrant, self.ollama)
        self.vector_retriever = VectorRetriever(
            self.qdrant, self.ollama, settings.retrieval
        )
        self.graph_retriever = GraphRetriever(
            self.neo4j, self.entity_linker, settings.retrieval
        )
        self.reranker = CrossEncoderReranker(settings.retrieval.reranker_model)
        self.ontology_retriever = OntologyRetriever(self.fuseki)

        # SOTA components
        self.graph_reasoner = GraphReasoner(
            neo4j=self.neo4j,
            ollama=self.ollama,
            entity_linker=self.entity_linker,
            config=settings.retrieval,
        )
        self.path_ranker = PathRanker()
        self.chain_of_thought = ChainOfThoughtReasoner(
            neo4j=self.neo4j,
            ollama=self.ollama,
            entity_linker=self.entity_linker,
            config=settings.retrieval,
        )
        self.answer_verifier = AnswerVerifier(
            ollama=self.ollama,
        )

        # Hybrid retrievers (standard and SOTA)
        self.hybrid_retriever = HybridRetriever(
            vector=self.vector_retriever,
            graph=self.graph_retriever,
            fuseki=self.fuseki,
            reranker=self.reranker,
            config=settings.retrieval,
        )
        self.hybrid_sota_retriever = HybridRetriever(
            vector=self.vector_retriever,
            graph=self.graph_retriever,
            fuseki=self.fuseki,
            reranker=self.reranker,
            config=settings.retrieval,
            graph_reasoner=self.graph_reasoner,
            path_ranker=self.path_ranker,
        )

        # Agents
        self.question_parser = QuestionParser(self.ollama)
        self.context_assembler = ContextAssembler()
        self.answer_generator = AnswerGenerator(self.ollama, self.context_assembler)
        self.explainer = Explainer()

        # ReAct reasoner (initialized after retrievers)
        self.react_reasoner = AgenticReasonerAdapter(
            ollama_provider=self.ollama,
            retriever=self.hybrid_sota_retriever,  # Use the full SOTA retriever for additional retrieval
            max_iterations=settings.retrieval.reasoning.react_max_iterations,
            relevance_threshold=settings.retrieval.reasoning.react_relevance_threshold,
        )

    # -- lifecycle ----------------------------------------------------------

    async def startup(self) -> None:
        """Connect to all external services."""
        await self.neo4j.connect()
        await self.qdrant.connect()
        await self.fuseki.connect()
        await self.ollama.connect()
        logger.info("orchestrator.started")

    async def shutdown(self) -> None:
        """Gracefully close all connections."""
        await self.neo4j.close()
        await self.qdrant.close()
        await self.fuseki.close()
        await self.ollama.close()
        logger.info("orchestrator.shutdown")

    # -- main pipeline ------------------------------------------------------

    async def answer(
        self,
        raw_question: str,
        *,
        strategy: str = "hybrid_sota",
    ) -> QAAnswer:
        """Full QA pipeline: question in → explained answer out.

        Args:
            raw_question: The user's question in natural language.
            strategy: Retrieval strategy — ``"vector_only"``, ``"graph_only"``,
                ``"hybrid"``, or ``"hybrid_sota"`` (default, full SOTA pipeline).
        """
        t0 = time.perf_counter()
        reasoning_cfg = self._settings.retrieval.reasoning

        # Phase 1: Parse question
        query = await self.question_parser.parse(raw_question)

        # Phase 2: Ontology expansion
        query = await self.ontology_retriever.expand_query(query)

        # Phase 3: Retrieve
        retriever = self._get_retriever(strategy)
        contexts: list[RetrievedContext] = await retriever.retrieve(query)

        # Phase 4: Reasoning (SOTA only) - Chain-of-Thought or ReAct
        reasoning_steps = []
        cot_summary = None
        react_result = None

        if strategy == "hybrid_sota":
            if reasoning_cfg.enable_react_reasoning:
                # Use ReAct reasoning with tool-calling
                logger.info("Using ReAct reasoning with tool-calling")
                react_result = await self.react_reasoner.reason_over_documents(
                    raw_question, contexts
                )
                # Convert ReAct result to answer text
                react_summary = react_result.get("reasoning_answer", "")
                # Add follow-up questions to reasoning steps for provenance
                for i, question in enumerate(react_result.get("followup_questions", [])):
                    reasoning_steps.append({
                        "step": i + 1,
                        "type": "react_followup",
                        "question": question,
                        "answer": f"Retrieved additional documents for: {question}"
                    })
            elif reasoning_cfg.enable_chain_of_thought:
                # Use traditional Chain-of-Thought reasoning
                logger.info("Using Chain-of-Thought reasoning")
                reasoning_steps = await self.chain_of_thought.reason(query, contexts)
                # Compose a CoT-grounded preamble for the answer generator
                cot_summary = self.chain_of_thought.compose_final_answer(reasoning_steps)

        # Phase 5: Generate answer (with optional reasoning context)
        if react_result and react_result.get("reasoning_answer"):
            # Use ReAct reasoning answer directly
            logger.info("Using ReAct reasoning answer directly")
            answer = await self.answer_generator.generate(
                query, contexts, preamble=react_result["reasoning_answer"]
            )
        else:
            # Use traditional answer generation with optional CoT context
            answer = await self.answer_generator.generate(
                query, contexts, cot_summary=cot_summary
            )

        # Attach reasoning steps to answer
        answer.reasoning_steps = reasoning_steps

        # Attach ReAct metadata if available
        if react_result:
            answer.react_metadata = {
                "followup_questions": react_result.get("followup_questions", []),
                "additional_chunks": len(react_result.get("additional_chunks", [])),
                "tool_calls": len(react_result.get("tool_calls", [])),
                "iterations": react_result.get("reasoning_metadata", {}).get("iterations_performed", 0)
            }

        # Attach exploration trace if available (from hybrid_sota retriever)
        if strategy == "hybrid_sota" and hasattr(self, "_last_exploration_trace"):
            answer.exploration_trace = self._last_exploration_trace

        # Phase 6: Answer verification (SOTA only)
        if strategy == "hybrid_sota" and reasoning_cfg.enable_answer_verification:
            verification = await self.answer_verifier.verify(
                answer=answer, query=query, contexts=contexts
            )
            answer.verification = verification

            # If unfaithful and we have CoT, warn in the answer
            if not verification.is_faithful and verification.contradicted_claims:
                answer.answer_text += (
                    "\n\n⚠ Note: Some claims in this answer could not be "
                    "fully verified against the knowledge graph."
                )

        # Phase 7: Explain (provenance + reasoning DAG + subgraph)
        answer = self.explainer.add_provenance(answer, contexts, query)

        answer.latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "orchestrator.answered",
            question=raw_question[:80],
            strategy=strategy,
            confidence=round(answer.confidence, 2),
            cot_steps=len(reasoning_steps),
            react_enabled=react_result is not None,
            followup_questions=len(react_result.get("followup_questions", [])) if react_result else 0,
            verified=answer.verification is not None,
            latency_ms=round(answer.latency_ms, 1),
        )
        return answer

    def _get_retriever(self, strategy: str) -> Retriever:
        """Resolve strategy name to retriever instance."""
        retrievers: dict[str, Retriever] = {
            "vector_only": self.vector_retriever,
            "graph_only": self.graph_retriever,
            "hybrid": self.hybrid_retriever,
            "hybrid_sota": self.hybrid_sota_retriever,
        }
        retriever = retrievers.get(strategy)
        if retriever is None:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from: {list(retrievers)}"
            )
        return retriever
