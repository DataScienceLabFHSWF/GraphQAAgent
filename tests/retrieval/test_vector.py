"""Tests for VectorRetriever."""

from unittest.mock import AsyncMock

import pytest

from kgrag.core.config import RetrievalConfig
from kgrag.core.models import DocumentChunk, QAQuery, RetrievalSource
from kgrag.retrieval.vector import VectorRetriever


@pytest.fixture
def vector_retriever(mock_qdrant: AsyncMock, mock_ollama: AsyncMock) -> VectorRetriever:
    return VectorRetriever(
        qdrant=mock_qdrant,
        ollama=mock_ollama,
        config=RetrievalConfig(),
    )


@pytest.mark.asyncio
async def test_retrieve_returns_vector_contexts(
    vector_retriever: VectorRetriever,
    mock_qdrant: AsyncMock,
    mock_ollama: AsyncMock,
):
    chunk = DocumentChunk(id="c1", doc_id="d1", content="Test chunk")
    mock_qdrant.search.return_value = [(chunk, 0.9)]

    query = QAQuery(raw_question="What is test?")
    results = await vector_retriever.retrieve(query)

    assert len(results) == 1
    assert results[0].source == RetrievalSource.VECTOR
    assert results[0].score == 0.9
    mock_ollama.embed.assert_called_once()


@pytest.mark.asyncio
async def test_retrieve_empty_results(
    vector_retriever: VectorRetriever,
    mock_qdrant: AsyncMock,
):
    mock_qdrant.search.return_value = []
    query = QAQuery(raw_question="Unknown question")
    results = await vector_retriever.retrieve(query)
    assert results == []
