"""Tests for QuestionParser."""

import json
from unittest.mock import AsyncMock

import pytest

from kgrag.agents.question_parser import QuestionParser
from kgrag.core.models import QuestionType


@pytest.fixture
def question_parser(mock_ollama: AsyncMock) -> QuestionParser:
    return QuestionParser(ollama=mock_ollama)


@pytest.mark.asyncio
async def test_parse_returns_qa_query(
    question_parser: QuestionParser,
    mock_ollama: AsyncMock,
):
    mock_ollama.generate.return_value = json.dumps({
        "question_type": "factoid",
        "detected_entities": ["Reaktor A"],
        "detected_types": ["NuclearFacility"],
        "sub_questions": [],
        "language": "de",
    })

    query = await question_parser.parse("Was ist Reaktor A?")
    assert query.question_type == QuestionType.FACTOID
    assert "Reaktor A" in query.detected_entities


@pytest.mark.asyncio
async def test_parse_handles_invalid_json(
    question_parser: QuestionParser,
    mock_ollama: AsyncMock,
):
    mock_ollama.generate.return_value = "not valid json"
    query = await question_parser.parse("Test question")
    # Should fallback gracefully
    assert query.raw_question == "Test question"
    assert query.question_type is None
