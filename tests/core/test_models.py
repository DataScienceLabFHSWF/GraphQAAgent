"""Tests for core data models."""

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


def test_kg_entity_creation():
    entity = KGEntity(id="e1", label="Test", entity_type="TestType")
    assert entity.id == "e1"
    assert entity.confidence == 0.0
    assert entity.properties == {}


def test_qa_query_defaults():
    query = QAQuery(raw_question="What is X?")
    assert query.language == "de"
    assert query.question_type is None
    assert query.detected_entities == []


def test_question_type_enum():
    assert QuestionType.FACTOID.value == "factoid"
    assert QuestionType("list") == QuestionType.LIST


def test_retrieval_source_enum():
    assert RetrievalSource.HYBRID.value == "hybrid"


def test_qa_answer_creation():
    answer = QAAnswer(question="Test?", answer_text="Answer.")
    assert answer.confidence == 0.0
    assert answer.reasoning_chain == []
    assert answer.subgraph_json is None


# -- SOTA model tests -------------------------------------------------------


def test_reasoning_step_creation():
    step = ReasoningStep(
        step_id=1,
        sub_question="What is X?",
        answer_fragment="X is a thing.",
        confidence=0.9,
        grounding_entities=["e1", "e2"],
        grounding_relations=["r1"],
    )
    assert step.step_id == 1
    assert step.confidence == 0.9
    assert len(step.grounding_entities) == 2


def test_reasoning_step_defaults():
    step = ReasoningStep(step_id=1, sub_question="Why?")
    assert step.evidence_text == ""
    assert step.answer_fragment == ""
    assert step.confidence == 0.0
    assert step.grounding_entities == []
    assert step.grounding_relations == []


def test_verification_result_faithful():
    result = VerificationResult(
        is_faithful=True,
        supported_claims=["Claim A is correct"],
        unsupported_claims=[],
        contradicted_claims=[],
        faithfulness_score=1.0,
        entity_coverage=0.9,
    )
    assert result.is_faithful
    assert result.faithfulness_score == 1.0


def test_verification_result_unfaithful():
    result = VerificationResult(
        is_faithful=False,
        supported_claims=["Claim A"],
        unsupported_claims=["Claim B"],
        contradicted_claims=["Claim C"],
        faithfulness_score=0.33,
        entity_coverage=0.5,
    )
    assert not result.is_faithful
    assert len(result.contradicted_claims) == 1


def test_graph_exploration_state_defaults():
    state = GraphExplorationState()
    assert state.visited_entity_ids == set()
    assert state.frontier_entity_ids == set()
    assert state.collected_entities == []
    assert state.collected_relations == []
    assert state.exploration_path == []
    assert state.iterations == 0
    assert not state.sufficient_evidence


def test_graph_exploration_state_with_data():
    state = GraphExplorationState(
        visited_entity_ids={"e1", "e2"},
        collected_entities=["Reactor A", "Method X"],
        collected_relations=["usesMethod"],
        iterations=3,
        sufficient_evidence=True,
    )
    assert len(state.visited_entity_ids) == 2
    assert state.iterations == 3
    assert state.sufficient_evidence


def test_qa_answer_with_sota_fields():
    step = ReasoningStep(step_id=1, sub_question="Why?", answer_fragment="Because.")
    verification = VerificationResult(
        is_faithful=True,
        supported_claims=["Because"],
        unsupported_claims=[],
        contradicted_claims=[],
        faithfulness_score=1.0,
        entity_coverage=1.0,
    )
    trace = GraphExplorationState(iterations=2, sufficient_evidence=True)

    answer = QAAnswer(
        question="Why?",
        answer_text="Because X.",
        reasoning_steps=[step],
        verification=verification,
        exploration_trace=trace,
    )

    assert len(answer.reasoning_steps) == 1
    assert answer.verification.is_faithful
    assert answer.exploration_trace.iterations == 2
