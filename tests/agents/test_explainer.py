"""Tests for Explainer (reasoning DAG + annotated subgraph)."""

from __future__ import annotations

import pytest

from kgrag.agents.explainer import Explainer
from kgrag.core.models import (
    GraphExplorationState,
    KGEntity,
    KGRelation,
    QAAnswer,
    QAQuery,
    QuestionType,
    ReasoningStep,
    RetrievalSource,
    RetrievedContext,
    VerificationResult,
)


@pytest.fixture
def explainer() -> Explainer:
    return Explainer()


@pytest.fixture
def query() -> QAQuery:
    return QAQuery(
        raw_question="Why was Reactor A decommissioned?",
        detected_entities=["Reactor A"],
        detected_types=["NuclearFacility"],
        question_type=QuestionType.CAUSAL,
    )


@pytest.fixture
def contexts() -> list[RetrievedContext]:
    return [
        RetrievedContext(
            source=RetrievalSource.GRAPH,
            text="Reactor A was decommissioned in 2005.",
            score=0.9,
            subgraph=[
                KGEntity(
                    id="e1",
                    label="Reactor A",
                    entity_type="NuclearFacility",
                    confidence=0.95,
                ),
                KGRelation(
                    source_id="e1",
                    target_id="e2",
                    relation_type="decommissionedIn",
                    confidence=0.88,
                ),
            ],
        ),
    ]


@pytest.fixture
def answer_basic() -> QAAnswer:
    return QAAnswer(
        question="Why was Reactor A decommissioned?",
        answer_text="Reactor A was decommissioned in 2005 due to safety concerns.",
    )


@pytest.fixture
def answer_with_sota() -> QAAnswer:
    """Answer with CoT steps, exploration trace, and verification."""
    return QAAnswer(
        question="Why was Reactor A decommissioned?",
        answer_text="Reactor A was decommissioned in 2005 due to safety concerns. [Source:1]",
        reasoning_steps=[
            ReasoningStep(
                step_id=1,
                sub_question="When was Reactor A decommissioned?",
                answer_fragment="In 2005.",
                confidence=0.92,
                grounding_entities=["e1"],
                grounding_relations=["e1::decommissionedIn::e2"],
            ),
            ReasoningStep(
                step_id=2,
                sub_question="Why was it decommissioned?",
                answer_fragment="Due to safety concerns.",
                confidence=0.78,
                grounding_entities=["e1"],
            ),
        ],
        exploration_trace=GraphExplorationState(
            visited_entity_ids={"e1", "e2"},
            collected_entities=["Reactor A", "Year 2005"],
            collected_relations=["decommissionedIn"],
            exploration_path=[
                "iter_0: Reactor A (e1) --decommissionedIn--> Year 2005 (e2)"
            ],
            iterations=1,
            sufficient_evidence=True,
        ),
        verification=VerificationResult(
            is_faithful=True,
            supported_claims=["Reactor A was decommissioned in 2005"],
            unsupported_claims=["safety concerns"],
            contradicted_claims=[],
            faithfulness_score=0.5,
            entity_coverage=0.9,
        ),
    )


def test_add_provenance_basic(explainer, answer_basic, contexts, query):
    """Basic provenance should produce reasoning chain and subgraph."""
    result = explainer.add_provenance(answer_basic, contexts, query)

    assert len(result.reasoning_chain) > 0
    assert any("causal" in s.lower() for s in result.reasoning_chain)
    assert result.subgraph_json is not None
    assert "nodes" in result.subgraph_json
    assert "edges" in result.subgraph_json
    assert result.confidence > 0.0


def test_add_provenance_with_sota(explainer, answer_with_sota, contexts, query):
    """SOTA provenance should include CoT, exploration, and verification in chain."""
    result = explainer.add_provenance(answer_with_sota, contexts, query)

    chain_text = "\n".join(result.reasoning_chain)

    # Should mention Chain-of-Thought
    assert "Chain-of-Thought" in chain_text
    # Should mention Graph Exploration
    assert "Graph Exploration" in chain_text
    # Should mention Verification
    assert "Verification" in chain_text
    # Should mention specific steps
    assert "Step 1" in chain_text
    assert "Step 2" in chain_text


def test_subgraph_json_annotated_with_reasoning_steps(
    explainer, answer_with_sota, contexts, query
):
    """Subgraph JSON should include reasoning_dag, exploration, and verification."""
    result = explainer.add_provenance(answer_with_sota, contexts, query)

    sg = result.subgraph_json
    assert "reasoning_dag" in sg
    assert len(sg["reasoning_dag"]) == 2
    assert sg["reasoning_dag"][0]["step_id"] == 1

    assert "exploration" in sg
    assert sg["exploration"]["iterations"] == 1
    assert sg["exploration"]["sufficient_evidence"] is True

    assert "verification" in sg
    assert sg["verification"]["is_faithful"] is True
    assert sg["verification"]["faithfulness_score"] == 0.5


def test_node_annotations(explainer, answer_with_sota, contexts, query):
    """Nodes should carry reasoning_steps annotation."""
    result = explainer.add_provenance(answer_with_sota, contexts, query)

    nodes = result.subgraph_json["nodes"]
    # e1 is grounded by both step 1 and step 2
    e1_nodes = [n for n in nodes if n["id"] == "e1"]
    if e1_nodes:
        assert "reasoning_steps" in e1_nodes[0]
        assert 1 in e1_nodes[0]["reasoning_steps"]


def test_confidence_boosted_by_verification(
    explainer, answer_with_sota, contexts, query
):
    """Confidence should incorporate verification and CoT step scores."""
    result = explainer.add_provenance(answer_with_sota, contexts, query)

    # With verification (faithfulness=0.5, entity_coverage=0.9) and
    # CoT steps (0.92, 0.78 avg=0.85), confidence should differ from
    # a basic answer
    basic_answer = QAAnswer(
        question="Test?",
        answer_text="Reactor A was decommissioned in 2005.",
    )
    basic_result = explainer.add_provenance(basic_answer, contexts, query)

    # SOTA answer has verification/CoT → different confidence
    assert result.confidence != basic_result.confidence


def test_cited_elements_from_explicit_citations(explainer, answer_with_sota, contexts, query):
    """[Source:1] citations should correctly resolve to entities/relations."""
    result = explainer.add_provenance(answer_with_sota, contexts, query)

    assert len(result.cited_entities) >= 1
    assert result.cited_entities[0].label == "Reactor A"
    assert len(result.cited_relations) >= 1
