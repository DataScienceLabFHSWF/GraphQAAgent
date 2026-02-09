"""Shared test fixtures — mock connectors for Neo4j, Qdrant, Fuseki, Ollama."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kgrag.core.config import (
    EvaluationConfig,
    FusekiConfig,
    Neo4jConfig,
    OllamaConfig,
    QdrantConfig,
    RetrievalConfig,
    Settings,
)
from kgrag.core.models import (
    DocumentChunk,
    KGEntity,
    KGRelation,
    RetrievalSource,
    RetrievedContext,
)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_entity() -> KGEntity:
    return KGEntity(
        id="e1",
        label="Reaktor A",
        entity_type="NuclearFacility",
        confidence=0.92,
        description="Ein Kernkraftwerk.",
    )


@pytest.fixture
def sample_relation() -> KGRelation:
    return KGRelation(
        source_id="e1",
        target_id="e2",
        relation_type="usesMethod",
        confidence=0.85,
    )


@pytest.fixture
def sample_chunk() -> DocumentChunk:
    return DocumentChunk(
        id="c1",
        doc_id="doc1",
        content="Reaktor A ist eine Kernanlage, die stillgelegt wurde.",
    )


@pytest.fixture
def sample_context(sample_chunk: DocumentChunk) -> RetrievedContext:
    return RetrievedContext(
        source=RetrievalSource.VECTOR,
        text=sample_chunk.content,
        score=0.85,
        chunk=sample_chunk,
    )


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings() -> Settings:
    return Settings(
        neo4j=Neo4jConfig(uri="bolt://localhost:7687"),
        qdrant=QdrantConfig(url="http://localhost:6333"),
        fuseki=FusekiConfig(url="http://localhost:3030"),
        ollama=OllamaConfig(base_url="http://localhost:11434"),
        retrieval=RetrievalConfig(),
        evaluation=EvaluationConfig(),
    )


# ---------------------------------------------------------------------------
# Mock connectors
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_neo4j() -> AsyncMock:
    mock = AsyncMock()
    mock.find_entities_by_label = AsyncMock(return_value=[])
    mock.find_entities_by_ids = AsyncMock(return_value=[])
    mock.get_neighbourhood = AsyncMock(return_value=([], []))
    mock.find_shortest_paths = AsyncMock(return_value=[])
    mock.compute_ppr = AsyncMock(return_value=[])
    mock.get_entity_neighbours = AsyncMock(return_value=([], []))
    mock.get_subgraph_between = AsyncMock(return_value=([], []))
    return mock


@pytest.fixture
def mock_qdrant() -> AsyncMock:
    mock = AsyncMock()
    mock.search = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def mock_fuseki() -> AsyncMock:
    mock = AsyncMock()
    mock.query = AsyncMock(return_value=[])
    mock.get_subclasses = AsyncMock(return_value=[])
    mock.get_synonyms = AsyncMock(return_value=[])
    mock.get_class_properties = AsyncMock(return_value=[])
    mock.get_class_by_label = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def mock_ollama() -> AsyncMock:
    mock = AsyncMock()
    mock.embed = AsyncMock(return_value=[0.1] * 384)
    mock.embed_batch = AsyncMock(return_value=[[0.1] * 384])
    mock.generate = AsyncMock(return_value="Test answer")
    mock.chat = AsyncMock(return_value="Test chat response")
    return mock
