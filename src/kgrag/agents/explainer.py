"""Explainer (C3.4.4) — Provenance, reasoning DAG, and subgraph visualisation.

**Key transparency contribution**: after an answer is generated, the Explainer
builds a full reasoning DAG that includes:
- Step-by-step reasoning chain (classic provenance)
- Chain-of-Thought sub-question traces (CoT reasoning steps)
- Think-on-Graph exploration trace (iterative graph exploration)
- Answer verification results (faithfulness + coverage)
- Cited entities/relations mapped back to the KG
- Visualisable subgraph JSON annotated with reasoning-step origins

This enables users to inspect exactly how the answer was derived, which SOTA
component contributed each piece of evidence, and whether the answer is
faithful to the underlying knowledge graph.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from kgrag.core.models import (
    GraphExplorationState,
    KGEntity,
    KGRelation,
    QAAnswer,
    QAQuery,
    ReasoningStep,
    RetrievedContext,
    VerificationResult,
)

logger = structlog.get_logger(__name__)


class Explainer:
    """Add provenance, reasoning DAG, and subgraph visualisation to answers.

    V2 enhancements over basic provenance:
    1. **Reasoning DAG**: full directed acyclic graph of reasoning steps,
       including CoT decomposition, graph exploration, and verification.
    2. **Annotated subgraph**: each node/edge is tagged with the reasoning
       step(s) that discovered or used it.
    3. **Verification summary**: faithfulness score and claim-level breakdown.
    4. **Exploration trace**: Think-on-Graph iteration details for debugging.
    """

    def add_provenance(
        self,
        answer: QAAnswer,
        contexts: list[RetrievedContext],
        query: QAQuery,
    ) -> QAAnswer:
        """Enrich *answer* with full reasoning DAG and subgraph.

        Mutates and returns the same :class:`QAAnswer` instance.
        """
        # 1. Build reasoning chain (includes CoT + exploration + verification)
        answer.reasoning_chain = self._build_reasoning_chain(query, contexts, answer)

        # 2. Collect cited entities and relations from used contexts
        cited_entities, cited_relations = self._extract_cited_elements(
            answer.answer_text, contexts
        )
        answer.cited_entities = cited_entities
        answer.cited_relations = cited_relations

        # 3. Build annotated subgraph JSON for visualisation
        answer.subgraph_json = self._build_subgraph_json(
            cited_entities,
            cited_relations,
            answer.reasoning_steps,
            answer.exploration_trace,
            answer.verification,
        )

        # 4. Compute confidence from evidence coverage + verification
        answer.confidence = self._estimate_confidence(answer, contexts)

        logger.info(
            "explainer.done",
            reasoning_steps=len(answer.reasoning_chain),
            cot_steps=len(answer.reasoning_steps) if answer.reasoning_steps else 0,
            cited_entities=len(cited_entities),
            cited_relations=len(cited_relations),
            verified=answer.verification is not None,
            confidence=round(answer.confidence, 2),
        )
        return answer

    # -- reasoning chain (DAG) -----------------------------------------------

    @staticmethod
    def _build_reasoning_chain(
        query: QAQuery,
        contexts: list[RetrievedContext],
        answer: QAAnswer,
    ) -> list[str]:
        """Build a human-readable reasoning trace (DAG linearised).

        Includes Chain-of-Thought steps, exploration trace, and verification
        results when available.
        """
        chain: list[str] = []
        chain.append(f'Received question: "{query.raw_question}"')

        if query.question_type:
            chain.append(f"Classified as: {query.question_type.value}")

        if query.detected_entities:
            chain.append(f"Detected entities: {', '.join(query.detected_entities)}")

        if query.detected_types:
            chain.append(f"Detected ontology types: {', '.join(query.detected_types)}")

        if query.sub_questions:
            chain.append(f"Decomposed into {len(query.sub_questions)} sub-questions")

        # Summarise retrieval
        sources = set()
        for ctx in contexts:
            sources.add(ctx.source.value)
        chain.append(
            f"Retrieved {len(contexts)} context pieces from: "
            f"{', '.join(sorted(sources))}"
        )

        # Note graph evidence
        graph_contexts = [c for c in contexts if c.subgraph]
        if graph_contexts:
            total_elements = sum(
                len(c.subgraph) for c in graph_contexts if c.subgraph
            )
            chain.append(
                f"Graph evidence: {total_elements} entities/relations from KG subgraphs"
            )

        # -- CoT reasoning steps -------------------------------------------
        if answer.reasoning_steps:
            chain.append(
                f"--- Chain-of-Thought ({len(answer.reasoning_steps)} steps) ---"
            )
            for step in answer.reasoning_steps:
                conf_str = f" (confidence={step.confidence:.2f})" if step.confidence else ""
                grounding = ""
                if step.grounding_entities:
                    grounding = f" [grounded: {', '.join(step.grounding_entities[:3])}]"
                chain.append(
                    f"  Step {step.step_id}: {step.sub_question}"
                    f" → {step.answer_fragment or '(no answer)'}"
                    f"{conf_str}{grounding}"
                )

        # -- Think-on-Graph exploration trace --------------------------------
        if answer.exploration_trace:
            trace = answer.exploration_trace
            chain.append(
                f"--- Graph Exploration ({trace.iterations} iterations) ---"
            )
            chain.append(
                f"  Visited {len(trace.visited_entity_ids)} entities, "
                f"collected {len(trace.collected_entities)} entities + "
                f"{len(trace.collected_relations)} relations"
            )
            if trace.exploration_path:
                for path_entry in trace.exploration_path[:10]:  # cap for readability
                    chain.append(f"  → {path_entry}")
            if trace.sufficient_evidence:
                chain.append("  ✓ Sufficient evidence found")
            else:
                chain.append("  ✗ Exploration exhausted without full evidence")

        # -- Verification results -------------------------------------------
        if answer.verification:
            v = answer.verification
            chain.append(
                f"--- Answer Verification "
                f"(faithfulness={v.faithfulness_score:.2f}) ---"
            )
            if v.supported_claims:
                chain.append(
                    f"  Supported ({len(v.supported_claims)}): "
                    + "; ".join(v.supported_claims[:3])
                )
            if v.unsupported_claims:
                chain.append(
                    f"  Unsupported ({len(v.unsupported_claims)}): "
                    + "; ".join(v.unsupported_claims[:3])
                )
            if v.contradicted_claims:
                chain.append(
                    f"  Contradicted ({len(v.contradicted_claims)}): "
                    + "; ".join(v.contradicted_claims[:3])
                )
            chain.append(
                f"  Entity coverage: {v.entity_coverage:.2f}"
            )

        chain.append(f"Generated answer ({len(answer.answer_text)} chars)")
        return chain

    # -- citation extraction ------------------------------------------------

    @staticmethod
    def _extract_cited_elements(
        answer_text: str,
        contexts: list[RetrievedContext],
    ) -> tuple[list[KGEntity], list[KGRelation]]:
        """Extract KG entities and relations that contributed to the answer."""
        entities: dict[str, KGEntity] = {}
        relations: dict[str, KGRelation] = {}

        # Find [Source:N] citations in the answer
        cited_indices: set[int] = set()
        for match in re.finditer(r"\[Source:(\d+)\]", answer_text):
            cited_indices.add(int(match.group(1)))

        # If no explicit citations, use all contexts with subgraphs
        use_all = len(cited_indices) == 0

        for i, ctx in enumerate(contexts, start=1):
            if not use_all and i not in cited_indices:
                continue
            if ctx.subgraph:
                for element in ctx.subgraph:
                    if isinstance(element, KGEntity):
                        entities[element.id] = element
                    elif isinstance(element, KGRelation):
                        key = f"{element.source_id}::{element.relation_type}::{element.target_id}"
                        relations[key] = element

        return list(entities.values()), list(relations.values())

    # -- subgraph visualisation (annotated) ----------------------------------

    @staticmethod
    def _build_subgraph_json(
        entities: list[KGEntity],
        relations: list[KGRelation],
        reasoning_steps: list[ReasoningStep] | None = None,
        exploration_trace: GraphExplorationState | None = None,
        verification: VerificationResult | None = None,
    ) -> dict[str, Any]:
        """Build a JSON-serialisable annotated graph for frontend visualisation.

        V2 enhancements:
        - Each node/edge is annotated with which reasoning step(s) use it.
        - Exploration trace adds a ``"discovery"`` layer showing iteration #.
        - Verification results add ``"verified"`` flags on nodes.

        Format compatible with common graph visualisation libraries
        (e.g. vis.js, D3, Cytoscape).
        """
        # Build entity→step mapping
        entity_step_map: dict[str, list[int]] = {}
        relation_step_map: dict[str, list[int]] = {}
        if reasoning_steps:
            for step in reasoning_steps:
                for eid in step.grounding_entities or []:
                    entity_step_map.setdefault(eid, []).append(step.step_id)
                for rid in step.grounding_relations or []:
                    relation_step_map.setdefault(rid, []).append(step.step_id)

        # Build exploration discovery iteration map
        discovery_iter: dict[str, int] = {}
        if exploration_trace and exploration_trace.exploration_path:
            for i, path_entry in enumerate(exploration_trace.exploration_path):
                # path_entry format: "iter_N: entity_label (entity_id) --rel--> ..."
                # Extract entity IDs where possible
                for eid in exploration_trace.visited_entity_ids:
                    if eid in path_entry:
                        discovery_iter.setdefault(eid, i)

        # Build node list
        nodes = []
        for e in entities:
            node: dict[str, Any] = {
                "id": e.id,
                "label": e.label,
                "type": e.entity_type,
                "confidence": e.confidence,
            }
            if e.id in entity_step_map:
                node["reasoning_steps"] = entity_step_map[e.id]
            if e.id in discovery_iter:
                node["discovery_iteration"] = discovery_iter[e.id]
            nodes.append(node)

        # Build edge list
        edges = []
        for r in relations:
            rel_key = f"{r.source_id}::{r.relation_type}::{r.target_id}"
            edge: dict[str, Any] = {
                "source": r.source_id,
                "target": r.target_id,
                "label": r.relation_type,
                "confidence": r.confidence,
            }
            if rel_key in relation_step_map:
                edge["reasoning_steps"] = relation_step_map[rel_key]
            edges.append(edge)

        # Build result
        result: dict[str, Any] = {"nodes": nodes, "edges": edges}

        # Add reasoning DAG summary
        if reasoning_steps:
            result["reasoning_dag"] = [
                {
                    "step_id": s.step_id,
                    "sub_question": s.sub_question,
                    "answer_fragment": s.answer_fragment,
                    "confidence": s.confidence,
                    "grounding_entities": s.grounding_entities,
                    "grounding_relations": s.grounding_relations,
                }
                for s in reasoning_steps
            ]

        # Add exploration summary
        if exploration_trace:
            result["exploration"] = {
                "iterations": exploration_trace.iterations,
                "visited_count": len(exploration_trace.visited_entity_ids),
                "collected_entities": len(exploration_trace.collected_entities),
                "collected_relations": len(exploration_trace.collected_relations),
                "sufficient_evidence": exploration_trace.sufficient_evidence,
            }

        # Add verification summary
        if verification:
            result["verification"] = {
                "is_faithful": verification.is_faithful,
                "faithfulness_score": verification.faithfulness_score,
                "entity_coverage": verification.entity_coverage,
                "supported_claims": len(verification.supported_claims),
                "unsupported_claims": len(verification.unsupported_claims),
                "contradicted_claims": len(verification.contradicted_claims),
            }

        return result

    # -- confidence estimation ----------------------------------------------

    @staticmethod
    def _estimate_confidence(
        answer: QAAnswer,
        contexts: list[RetrievedContext],
    ) -> float:
        """Heuristic confidence based on evidence quality + verification.

        V2 factors:
        - Average retrieval score of used contexts.
        - Whether graph evidence was available (structured > unstructured).
        - Number of contexts used.
        - Verification faithfulness score (if available).
        - CoT reasoning step confidence (if available).
        - Entity coverage from verification.
        """
        if not contexts:
            return 0.0

        avg_score = sum(c.score for c in contexts) / len(contexts)
        has_graph = any(c.subgraph for c in contexts)
        context_bonus = min(len(contexts) / 5.0, 1.0) * 0.1

        # Base confidence from retrieval quality
        base = avg_score * 0.5 + (0.1 if has_graph else 0.0) + context_bonus

        # Verification bonus/penalty
        if answer.verification:
            v = answer.verification
            # Faithfulness score: high → boost, low → penalise
            base += (v.faithfulness_score - 0.5) * 0.2
            # Entity coverage: partial credit
            base += v.entity_coverage * 0.1
            # Contradictions: penalise
            if v.contradicted_claims:
                base -= len(v.contradicted_claims) * 0.05

        # CoT average step confidence
        if answer.reasoning_steps:
            step_confs = [
                s.confidence for s in answer.reasoning_steps if s.confidence
            ]
            if step_confs:
                avg_step = sum(step_confs) / len(step_confs)
                base += avg_step * 0.1

        return min(max(base, 0.0), 1.0)
