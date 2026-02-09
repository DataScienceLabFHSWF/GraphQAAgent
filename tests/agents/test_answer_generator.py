"""Tests for AnswerGenerator."""

from unittest.mock import AsyncMock

import pytest

from kgrag.agents.answer_generator import AnswerGenerator
from kgrag.core.models import QAQuery, RetrievalSource, RetrievedContext


@pytest.fixture
def answer_generator(mock_ollama: AsyncMock) -> AnswerGenerator:
    return AnswerGenerator(ollama=mock_ollama)


@pytest.mark.asyncio
async def test_generate_returns_qa_answer(
    answer_generator: AnswerGenerator,
    mock_ollama: AsyncMock,
):
    mock_ollama.generate.return_value = "Reaktor A ist eine Kernanlage."

    query = QAQuery(raw_question="Was ist Reaktor A?")
    contexts = [
        RetrievedContext(source=RetrievalSource.VECTOR, text="Test context", score=0.8)
    ]

    answer = await answer_generator.generate(query, contexts)
    assert answer.answer_text == "Reaktor A ist eine Kernanlage."
    assert answer.question == "Was ist Reaktor A?"
    assert answer.latency_ms > 0
