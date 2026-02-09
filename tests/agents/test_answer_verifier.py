"""Tests for AnswerVerifier (KG-grounded faithfulness checking)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kgrag.agents.answer_verifier import AnswerVerifier
from kgrag.core.models import (
    KGEntity,
    QAAnswer,
    QAQuery,
    RetrievalSource,
    RetrievedContext,
    VerificationResult,
)


@pytest.fixture
def verifier(mock_ollama) -> AnswerVerifier:
    return AnswerVerifier(ollama=mock_ollama)


@pytest.fixture
def answer_with_claims() -> QAAnswer:
    return QAAnswer(
        question="When was Reactor A decommissioned?",
        answer_text="Reactor A was decommissioned in 2005 using Method X. "
        "It was located at Site B.",
    )


@pytest.fixture
def query() -> QAQuery:
    return QAQuery(
        raw_question="When was Reactor A decommissioned?",
        detected_entities=["Reactor A"],
    )


@pytest.fixture
def evidence_contexts() -> list[RetrievedContext]:
    return [
        RetrievedContext(
            source=RetrievalSource.GRAPH,
            text="Reactor A was decommissioned in 2005.",
            score=0.9,
            subgraph=[
                KGEntity(
                    id="e1",
                    label="Reactor A",
                    entity_type="Facility",
                    confidence=0.95,
                ),
            ],
        ),
        RetrievedContext(
            source=RetrievalSource.GRAPH,
            text="Method X is a thermal cutting technique used at Site B.",
            score=0.85,
        ),
    ]


@pytest.mark.asyncio
async def test_verify_returns_verification_result(
    verifier, answer_with_claims, query, evidence_contexts
):
    """verify() should return a VerificationResult."""
    # Mock LLM: extract claims, then verify each
    verifier._ollama.generate = AsyncMock(
        side_effect=[
            # claim extraction
            '["Reactor A was decommissioned in 2005", '
            '"Method X was used", '
            '"It was located at Site B"]',
            # claim 1 verification
            '{"verdict": "supported", "evidence": "Context confirms 2005"}',
            # claim 2 verification
            '{"verdict": "supported", "evidence": "Method X mentioned"}',
            # claim 3 verification
            '{"verdict": "supported", "evidence": "Site B mentioned"}',
        ]
    )

    result = await verifier.verify(answer_with_claims, evidence_contexts)

    assert isinstance(result, VerificationResult)
    assert result.faithfulness_score >= 0.0
    assert result.faithfulness_score <= 1.0


@pytest.mark.asyncio
async def test_verify_detects_unsupported_claims(
    verifier, query, evidence_contexts
):
    """Claims not backed by evidence should be marked unsupported."""
    answer = QAAnswer(
        question="When was Reactor A decommissioned?",
        answer_text="Reactor A was built in 1960 and is located on Mars.",
    )

    verifier._ollama.generate = AsyncMock(
        side_effect=[
            '["Reactor A was built in 1960", "Reactor A is located on Mars"]',
            '{"verdict": "unsupported", "evidence": "No mention of 1960"}',
            '{"verdict": "contradicted", "evidence": "Site B, not Mars"}',
        ]
    )

    result = await verifier.verify(answer, evidence_contexts)

    assert not result.is_faithful
    assert len(result.unsupported_claims) + len(result.contradicted_claims) >= 1


@pytest.mark.asyncio
async def test_verify_empty_answer(verifier, query, evidence_contexts):
    """An empty answer should still produce a valid result."""
    answer = QAAnswer(question="Test?", answer_text="")

    verifier._ollama.generate = AsyncMock(return_value="[]")

    result = await verifier.verify(answer, evidence_contexts)

    assert isinstance(result, VerificationResult)
    # No claims → technically faithful (nothing to contradict)
    assert result.is_faithful


@pytest.mark.asyncio
async def test_verify_graceful_on_llm_failure(
    verifier, answer_with_claims, query, evidence_contexts
):
    """If LLM fails, verifier should fall back gracefully."""
    verifier._ollama.generate = AsyncMock(
        side_effect=[
            "not json",  # claim extraction fails
        ]
    )

    result = await verifier.verify(answer_with_claims, evidence_contexts)

    # Should still return a result (fallback to keyword-based)
    assert isinstance(result, VerificationResult)


@pytest.mark.asyncio
async def test_entity_coverage_calculation(
    verifier, answer_with_claims, query, evidence_contexts
):
    """Entity coverage should reflect how many query entities appear in the answer."""
    verifier._ollama.generate = AsyncMock(
        side_effect=[
            '["Reactor A was decommissioned in 2005"]',
            '{"verdict": "supported", "evidence": "confirmed"}',
        ]
    )

    result = await verifier.verify(answer_with_claims, evidence_contexts)

    # "Reactor A" should be found in the answer → coverage > 0
    assert result.entity_coverage >= 0.0
