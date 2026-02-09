"""Multi-hop Chain-of-Thought reasoner (C3.4.6).

**SOTA technique** inspired by GCR (Luo et al. 2025), KG-GPT (Kim et al. 2023),
and the neural-symbolic NL query answering methods from Liu et al. (2024, §5).

Instead of generating a single answer from all context at once, this module:

1. Decomposes multi-hop questions into atomic reasoning steps.
2. For each step, retrieves targeted KG evidence.
3. Grounds each step's answer in specific entities and relations.
4. Composes the final answer from step-wise reasoning.

This produces a transparent, verifiable reasoning chain where each hop is
explicitly grounded in KG triples — a key differentiator for our transparency
contribution.
"""

from __future__ import annotations

import json
import time

import structlog

from kgrag.connectors.neo4j import Neo4jConnector
from kgrag.connectors.langchain_ollama_provider import LangChainOllamaProvider
from kgrag.core.config import RetrievalConfig
from kgrag.core.models import (
    KGEntity,
    KGRelation,
    QAQuery,
    QuestionType,
    ReasoningStep,
    RetrievalSource,
    RetrievedContext,
)
from kgrag.retrieval.entity_linker import EntityLinker

logger = structlog.get_logger(__name__)

_DECOMPOSE_PROMPT = """\
You are a question decomposition expert. Given a complex question, break it \
into a sequence of simpler atomic sub-questions that can each be answered by \
looking up a single fact or traversing a single relation in a knowledge graph.

Question: {question}

Detected entities: {entities}
Detected ontology types: {types}

Return a JSON array of sub-questions in order, where each sub-question builds \
on the answers of previous ones. Each element should have:
{{"sub_question": "...", "target_entity_or_relation": "...", "depends_on_step": null | step_number}}

Return ONLY valid JSON array. Max {max_steps} steps."""

_STEP_ANSWER_PROMPT = """\
Given the following evidence from a knowledge graph, answer this specific \
sub-question concisely.

Sub-question: {sub_question}
{previous_context}

Evidence:
{evidence}

Instructions:
- Answer ONLY based on the evidence provided.
- Name the specific entities and relations that support your answer.
- If the evidence is insufficient, say "insufficient evidence."

Answer:"""


class ChainOfThoughtReasoner:
    """Multi-hop KG-grounded chain-of-thought reasoning.

    Decomposes complex questions into atomic steps, retrieves evidence for
    each step from the KG, and builds a fully grounded reasoning chain.
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

    async def reason(
        self,
        query: QAQuery,
        contexts: list[RetrievedContext],
    ) -> list[ReasoningStep]:
        """Execute multi-hop chain-of-thought reasoning.

        Returns a list of :class:`ReasoningStep` objects, each grounded in
        specific KG entities and relations.
        """
        t0 = time.perf_counter()

        # 1. Decide if CoT is needed
        if not self._needs_cot(query):
            return self._single_step(query, contexts)

        # 2. Decompose question into sub-questions
        sub_questions = await self._decompose(query)
        if not sub_questions:
            return self._single_step(query, contexts)

        # 3. Answer each sub-question with targeted evidence
        steps: list[ReasoningStep] = []
        accumulated_answers: list[str] = []

        for i, sq in enumerate(sub_questions):
            sub_q_text = sq.get("sub_question", "")
            target = sq.get("target_entity_or_relation", "")

            # Retrieve evidence targeted at this sub-question's entities
            evidence_text, grounding_entities, grounding_relations = (
                await self._retrieve_for_step(sub_q_text, target, contexts)
            )

            # Build previous context from accumulated answers
            prev_ctx = ""
            if accumulated_answers:
                prev_lines = [
                    f"Step {j + 1}: {ans}"
                    for j, ans in enumerate(accumulated_answers)
                ]
                prev_ctx = f"Previous reasoning steps:\n" + "\n".join(prev_lines)

            # Ask LLM to answer this sub-question
            answer_fragment = await self._answer_step(
                sub_q_text, evidence_text, prev_ctx,
            )

            step = ReasoningStep(
                step_id=i + 1,
                sub_question=sub_q_text,
                evidence_text=evidence_text,
                answer_fragment=answer_fragment,
                grounding_entities=grounding_entities,
                grounding_relations=grounding_relations,
                confidence=self._step_confidence(grounding_entities, grounding_relations),
                source=RetrievalSource.GRAPH if grounding_relations else RetrievalSource.VECTOR,
            )
            steps.append(step)
            accumulated_answers.append(answer_fragment)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "cot_reasoner.done",
            question=query.raw_question[:80],
            steps=len(steps),
            latency_ms=round(elapsed, 1),
        )
        return steps

    @staticmethod
    def _needs_cot(query: QAQuery) -> bool:
        """Heuristic: use CoT for complex question types or multi-entity queries."""
        if query.question_type in (
            QuestionType.CAUSAL,
            QuestionType.COMPARATIVE,
        ):
            return True
        if query.sub_questions and len(query.sub_questions) > 1:
            return True
        if len(query.detected_entities) >= 2:
            return True
        return False

    async def _decompose(self, query: QAQuery) -> list[dict]:
        """Decompose question into atomic sub-questions via LLM."""
        # If QuestionParser already decomposed, use those
        if query.sub_questions:
            return [
                {"sub_question": sq, "target_entity_or_relation": "", "depends_on_step": None}
                for sq in query.sub_questions
            ]

        prompt = _DECOMPOSE_PROMPT.format(
            question=query.raw_question,
            entities=", ".join(query.detected_entities) if query.detected_entities else "none",
            types=", ".join(query.detected_types) if query.detected_types else "none",
            max_steps=self._reasoning.cot_max_steps,
        )

        try:
            response = await self._ollama.generate(
                prompt=prompt,
                temperature=0.1,
                format="json",
            )
            sub_questions = json.loads(response)
            if isinstance(sub_questions, list):
                return sub_questions[: self._reasoning.cot_max_steps]
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("cot_reasoner.decompose_failed", error=str(exc))

        return []

    async def _retrieve_for_step(
        self,
        sub_question: str,
        target: str,
        contexts: list[RetrievedContext],
    ) -> tuple[str, list[str], list[str]]:
        """Retrieve targeted evidence for a single reasoning step.

        Searches existing contexts first, then falls back to direct KG lookup.
        """
        grounding_entities: list[str] = []
        grounding_relations: list[str] = []
        evidence_parts: list[str] = []

        # Search existing contexts for relevant evidence
        target_lower = target.lower() if target else sub_question.lower()
        for ctx in contexts:
            if target_lower in ctx.text.lower() or any(
                term.lower() in ctx.text.lower()
                for term in sub_question.split()
                if len(term) > 3
            ):
                evidence_parts.append(ctx.text[:500])
                if ctx.subgraph:
                    for el in ctx.subgraph:
                        if isinstance(el, KGEntity):
                            grounding_entities.append(el.id)
                        elif isinstance(el, KGRelation):
                            grounding_relations.append(
                                f"{el.source_id}::{el.relation_type}::{el.target_id}"
                            )

        # If no evidence found in contexts, try direct KG lookup
        if not evidence_parts and target:
            try:
                linked = await self._entity_linker.link([target])
                if linked:
                    entity_ids = [e.id for e in linked]
                    entities, relations = await self._neo4j.get_neighbourhood(
                        entity_ids, max_hops=1, max_nodes=20,
                    )
                    for e in entities:
                        grounding_entities.append(e.id)
                        evidence_parts.append(f'Entity: "{e.label}" ({e.entity_type})')
                    for r in relations:
                        grounding_relations.append(
                            f"{r.source_id}::{r.relation_type}::{r.target_id}"
                        )
                        evidence_parts.append(
                            f'Relation: "{r.source_id}" --[{r.relation_type}]--> "{r.target_id}"'
                        )
            except Exception as exc:
                logger.warning("cot_reasoner.step_retrieval_failed", error=str(exc))

        evidence_text = "\n".join(evidence_parts) if evidence_parts else "(no evidence found)"
        return evidence_text, grounding_entities, grounding_relations

    async def _answer_step(
        self,
        sub_question: str,
        evidence: str,
        previous_context: str,
    ) -> str:
        """Answer a single sub-question grounded in evidence."""
        prompt = _STEP_ANSWER_PROMPT.format(
            sub_question=sub_question,
            evidence=evidence,
            previous_context=previous_context,
        )
        try:
            return await self._ollama.generate(prompt=prompt, temperature=0.1)
        except Exception as exc:
            logger.warning("cot_reasoner.step_answer_failed", error=str(exc))
            return "Could not answer this step."

    @staticmethod
    def _step_confidence(
        grounding_entities: list[str],
        grounding_relations: list[str],
    ) -> float:
        """Heuristic confidence for a reasoning step."""
        if grounding_relations:
            return min(0.5 + 0.1 * len(grounding_relations), 0.95)
        if grounding_entities:
            return min(0.3 + 0.1 * len(grounding_entities), 0.7)
        return 0.1

    @staticmethod
    def compose_final_answer(steps: list[ReasoningStep]) -> str:
        """Compose the reasoning steps into a coherent multi-hop answer."""
        if not steps:
            return ""
        if len(steps) == 1:
            return steps[0].answer_fragment

        parts: list[str] = []
        for step in steps:
            parts.append(step.answer_fragment)

        return " ".join(parts)

    def _single_step(
        self,
        query: QAQuery,
        contexts: list[RetrievedContext],
    ) -> list[ReasoningStep]:
        """Fallback for simple questions: single step using all contexts."""
        grounding_entities: list[str] = []
        grounding_relations: list[str] = []
        evidence_parts: list[str] = []

        for ctx in contexts[:5]:
            evidence_parts.append(ctx.text[:300])
            if ctx.subgraph:
                for el in ctx.subgraph:
                    if isinstance(el, KGEntity):
                        grounding_entities.append(el.id)
                    elif isinstance(el, KGRelation):
                        grounding_relations.append(el.id)

        return [
            ReasoningStep(
                step_id=1,
                sub_question=query.raw_question,
                evidence_text="\n".join(evidence_parts),
                answer_fragment="",  # will be filled by AnswerGenerator
                grounding_entities=grounding_entities,
                grounding_relations=grounding_relations,
                confidence=self._step_confidence(grounding_entities, grounding_relations),
            )
        ]
