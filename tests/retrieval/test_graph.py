"""Tests for GraphRetriever."""

from unittest.mock import AsyncMock

import pytest

from kgrag.core.config import RetrievalConfig
from kgrag.core.models import KGEntity, KGRelation, QAQuery, QuestionType, RetrievalSource
from kgrag.retrieval.entity_linker import EntityLinker
from kgrag.retrieval.graph import GraphMode, GraphRetriever


@pytest.fixture
def mock_entity_linker(mock_neo4j: AsyncMock) -> EntityLinker:
    linker = EntityLinker(neo4j=mock_neo4j)
    return linker


@pytest.fixture
def graph_retriever(
    mock_neo4j: AsyncMock,
    mock_entity_linker: EntityLinker,
) -> GraphRetriever:
    return GraphRetriever(
        neo4j=mock_neo4j,
        entity_linker=mock_entity_linker,
        config=RetrievalConfig(),
    )


@pytest.mark.asyncio
async def test_auto_selects_path_for_causal(graph_retriever: GraphRetriever):
    query = QAQuery(
        raw_question="Why was Reaktor A shut down?",
        question_type=QuestionType.CAUSAL,
    )
    mode = graph_retriever._auto_select_mode(query)
    assert mode == GraphMode.PATH


@pytest.mark.asyncio
async def test_auto_selects_entity_centric_for_list(graph_retriever: GraphRetriever):
    query = QAQuery(
        raw_question="Which methods are used?",
        question_type=QuestionType.LIST,
    )
    mode = graph_retriever._auto_select_mode(query)
    assert mode == GraphMode.ENTITY_CENTRIC


def test_serialise_subgraph(graph_retriever: GraphRetriever):
    entities = [KGEntity(id="e1", label="Reaktor A", entity_type="Facility", confidence=0.9)]
    relations = [KGRelation(source_id="e1", target_id="e2",
                            relation_type="usesMethod", confidence=0.8)]
    text = graph_retriever._serialise_subgraph(entities, relations)
    assert "Reaktor A" in text
    assert "usesMethod" in text
