"""Tests for GraphReasoner (Think-on-Graph iterative exploration)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from kgrag.core.config import RetrievalConfig
from kgrag.core.models import (
    KGEntity,
    KGRelation,
    QAQuery,
    QuestionType,
    RetrievalSource,
)
from kgrag.retrieval.graph_reasoning import GraphReasoner


@pytest.fixture
def graph_reasoner(mock_neo4j, mock_ollama) -> GraphReasoner:
    entity_linker = AsyncMock()
    entity_linker.link = AsyncMock(return_value=["e1", "e2"])
    config = RetrievalConfig()
    return GraphReasoner(
        neo4j=mock_neo4j,
        ollama=mock_ollama,
        entity_linker=entity_linker,
        config=config,
    )


@pytest.fixture
def multi_entity_query() -> QAQuery:
    return QAQuery(
        raw_question="How does Reactor A relate to decommissioning method X?",
        detected_entities=["Reactor A", "Method X"],
        question_type=QuestionType.CAUSAL,
    )


@pytest.fixture
def sample_entities() -> list[KGEntity]:
    return [
        KGEntity(id="e1", label="Reactor A", entity_type="Facility", confidence=0.9),
        KGEntity(id="e2", label="Method X", entity_type="Method", confidence=0.85),
    ]


@pytest.fixture
def sample_relations() -> list[KGRelation]:
    return [
        KGRelation(
            source_id="e1",
            target_id="e2",
            relation_type="usesMethod",
            confidence=0.88,
        ),
    ]


@pytest.mark.asyncio
async def test_explore_returns_contexts_and_state(
    graph_reasoner, multi_entity_query, sample_entities, sample_relations, mock_neo4j
):
    """Exploration should return contexts and a valid exploration state."""
    mock_neo4j.compute_ppr = AsyncMock(return_value=[(e, 0.9) for e in sample_entities])
    mock_neo4j.get_subgraph_between = AsyncMock(
        return_value=(sample_entities, sample_relations)
    )
    # Make the LLM say "stop" immediately
    graph_reasoner._ollama.generate = AsyncMock(
        return_value='{"action": "stop", "reason": "enough evidence"}'
    )

    contexts, state = await graph_reasoner.explore(multi_entity_query)

    assert state is not None
    assert isinstance(state.visited_entity_ids, set)
    assert state.iterations >= 0


@pytest.mark.asyncio
async def test_explore_no_entities_returns_empty(graph_reasoner):
    """If the query has no entities, exploration returns empty results."""
    query = QAQuery(raw_question="What is physics?", detected_entities=[])
    graph_reasoner._entity_linker.link = AsyncMock(return_value=[])

    contexts, state = await graph_reasoner.explore(query)

    assert contexts == []
    assert state.iterations == 0


@pytest.mark.asyncio
async def test_explore_iterates_on_explore_action(
    graph_reasoner, multi_entity_query, sample_entities, sample_relations, mock_neo4j
):
    """When LLM says 'explore', the reasoner should iterate."""
    mock_neo4j.compute_ppr = AsyncMock(
        return_value=[(e, 0.9) for e in sample_entities]
    )
    mock_neo4j.get_subgraph_between = AsyncMock(
        return_value=(sample_entities, sample_relations)
    )
    mock_neo4j.get_entity_neighbours = AsyncMock(
        return_value=[
            (
                KGEntity(id="e3", label="Site B", entity_type="Site", confidence=0.7),
                KGRelation(
                    source_id="e1",
                    target_id="e3",
                    relation_type="locatedAt",
                    confidence=0.8,
                ),
            )
        ]
    )

    # First call: explore, second call: stop
    graph_reasoner._ollama.generate = AsyncMock(
        side_effect=[
            '{"action": "explore", "entity_ids": ["e3"], "reason": "need more"}',
            '{"action": "stop", "reason": "sufficient"}',
        ]
    )

    contexts, state = await graph_reasoner.explore(multi_entity_query)

    assert state.iterations >= 1


@pytest.mark.asyncio
async def test_explore_respects_max_iterations(graph_reasoner, multi_entity_query, mock_neo4j, sample_entities, sample_relations):
    """Exploration should stop at max iterations even if LLM keeps saying explore."""
    mock_neo4j.compute_ppr = AsyncMock(
        return_value=[(e, 0.9) for e in sample_entities]
    )
    mock_neo4j.get_subgraph_between = AsyncMock(
        return_value=(sample_entities, sample_relations)
    )
    mock_neo4j.get_entity_neighbours = AsyncMock(
        return_value=[
            (entity, sample_relations[i % len(sample_relations)])
            for i, entity in enumerate(sample_entities)
        ]
    )
    # Always say explore
    graph_reasoner._ollama.generate = AsyncMock(
        return_value='{"action": "explore", "entity_ids": ["e1"], "reason": "more"}'
    )

    contexts, state = await graph_reasoner.explore(multi_entity_query)

    assert state.iterations <= graph_reasoner._config.reasoning.max_exploration_iterations
