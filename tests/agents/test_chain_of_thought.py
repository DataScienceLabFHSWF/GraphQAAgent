"""Tests for ChainOfThoughtReasoner (multi-hop CoT decomposition)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kgrag.agents.chain_of_thought import ChainOfThoughtReasoner
from kgrag.core.config import RetrievalConfig
from kgrag.core.models import (
    QAQuery,
    QuestionType,
    ReasoningStep,
    RetrievalSource,
    RetrievedContext,
)


@pytest.fixture
def cot_reasoner(mock_neo4j, mock_ollama) -> ChainOfThoughtReasoner:
    entity_linker = AsyncMock()
    entity_linker.link = AsyncMock(return_value=[])
    config = RetrievalConfig()
    return ChainOfThoughtReasoner(
        neo4j=mock_neo4j,
        ollama=mock_ollama,
        entity_linker=entity_linker,
        config=config,
    )


@pytest.fixture
def causal_query() -> QAQuery:
    return QAQuery(
        raw_question="Why was Reactor A decommissioned using Method X?",
        detected_entities=["Reactor A", "Method X"],
        question_type=QuestionType.CAUSAL,
        sub_questions=[
            "When was Reactor A decommissioned?",
            "What is Method X?",
        ],
    )


@pytest.fixture
def simple_query() -> QAQuery:
    return QAQuery(
        raw_question="What is Reactor A?",
        detected_entities=["Reactor A"],
        question_type=QuestionType.FACTOID,
    )


@pytest.fixture
def sample_contexts() -> list[RetrievedContext]:
    return [
        RetrievedContext(
            source=RetrievalSource.GRAPH,
            text="Reactor A was decommissioned in 2005.",
            score=0.9,
        ),
        RetrievedContext(
            source=RetrievalSource.GRAPH,
            text="Method X is a thermal cutting technique.",
            score=0.85,
        ),
    ]


@pytest.mark.asyncio
async def test_reason_causal_triggers_cot(cot_reasoner, causal_query, sample_contexts):
    """CAUSAL query with sub-questions should trigger CoT decomposition."""
    # Mock LLM to return decomposition then step answers
    cot_reasoner._ollama.generate = AsyncMock(
        side_effect=[
            # decomposition
            '[{"sub_question": "When was Reactor A decommissioned?", "depends_on": []}, '
            '{"sub_question": "What is Method X?", "depends_on": []}]',
            # step 1 answer
            "Reactor A was decommissioned in 2005.",
            # step 2 answer
            "Method X is a thermal cutting technique.",
        ]
    )

    steps = await cot_reasoner.reason(causal_query, sample_contexts)

    assert len(steps) >= 1
    assert all(isinstance(s, ReasoningStep) for s in steps)


@pytest.mark.asyncio
async def test_reason_simple_returns_single_step(
    cot_reasoner, simple_query, sample_contexts
):
    """Simple FACTOID with one entity should return a single reasoning step."""
    steps = await cot_reasoner.reason(simple_query, sample_contexts)

    # Should produce 1 step (passthrough, no decomposition)
    assert len(steps) == 1
    assert steps[0].step_id == 1


@pytest.mark.asyncio
async def test_compose_final_answer():
    """compose_final_answer should combine step fragments."""
    steps = [
        ReasoningStep(
            step_id=1,
            sub_question="When was it decommissioned?",
            answer_fragment="In 2005.",
            confidence=0.9,
        ),
        ReasoningStep(
            step_id=2,
            sub_question="What method was used?",
            answer_fragment="Method X (thermal cutting).",
            confidence=0.85,
        ),
    ]

    result = ChainOfThoughtReasoner.compose_final_answer(steps)

    assert "2005" in result
    assert "Method X" in result


@pytest.mark.asyncio
async def test_reason_handles_llm_failure_gracefully(
    cot_reasoner, causal_query, sample_contexts
):
    """If LLM decomposition fails, should fall back to single-step reasoning."""
    cot_reasoner._ollama.generate = AsyncMock(
        side_effect=[
            "not valid json at all",  # decomposition fails
        ]
    )

    steps = await cot_reasoner.reason(causal_query, sample_contexts)

    # Should still return at least 1 step (fallback)
    assert len(steps) >= 1
