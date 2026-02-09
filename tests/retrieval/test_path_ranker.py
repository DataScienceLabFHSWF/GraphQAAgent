"""Tests for PathRanker (relation-aware path scoring)."""

from __future__ import annotations

import pytest

from kgrag.core.models import (
    KGRelation,
    QAQuery,
    QuestionType,
    RetrievalSource,
    RetrievedContext,
)
from kgrag.retrieval.path_ranker import PathRanker


@pytest.fixture
def path_ranker() -> PathRanker:
    return PathRanker(alpha=0.4, beta=0.45, gamma=0.15)


@pytest.fixture
def query_with_expected_relations() -> QAQuery:
    return QAQuery(
        raw_question="How does Reactor A relate to decommissioning?",
        expected_relations=["usesMethod", "decommissionedBy", "hasProcess"],
        question_type=QuestionType.CAUSAL,
    )


@pytest.fixture
def graph_contexts() -> list[RetrievedContext]:
    """Contexts with subgraph relations of varying relevance."""
    return [
        RetrievedContext(
            source=RetrievalSource.GRAPH,
            text="Reactor A uses Method X for decommissioning.",
            score=0.8,
            subgraph=[
                KGRelation(
                    source_id="e1",
                    target_id="e2",
                    relation_type="usesMethod",
                    confidence=0.9,
                ),
            ],
        ),
        RetrievedContext(
            source=RetrievalSource.GRAPH,
            text="Reactor A is located at Site B.",
            score=0.7,
            subgraph=[
                KGRelation(
                    source_id="e1",
                    target_id="e3",
                    relation_type="locatedAt",
                    confidence=0.85,
                ),
            ],
        ),
        RetrievedContext(
            source=RetrievalSource.GRAPH,
            text="Method X is part of Process Y.",
            score=0.6,
            subgraph=[
                KGRelation(
                    source_id="e2",
                    target_id="e4",
                    relation_type="hasProcess",
                    confidence=0.75,
                ),
            ],
        ),
    ]


def test_rank_paths_boosts_matching_relations(
    path_ranker, query_with_expected_relations, graph_contexts
):
    """Contexts with ontology-matching relations should rank higher."""
    ranked = path_ranker.rank_paths(graph_contexts, query_with_expected_relations)

    assert len(ranked) == 3
    # usesMethod and hasProcess match expected relations → should be ranked above locatedAt
    labels = [
        r.relation_type
        for ctx in ranked
        for r in (ctx.subgraph or [])
        if isinstance(r, KGRelation)
    ]
    # locatedAt doesn't match any expected relation → should be last
    assert labels[-1] == "locatedAt"


def test_rank_paths_empty_contexts(path_ranker, query_with_expected_relations):
    """Empty context list returns empty."""
    assert path_ranker.rank_paths([], query_with_expected_relations) == []


def test_rank_paths_no_expected_relations(path_ranker, graph_contexts):
    """Without expected relations, ranking should still work (by confidence only)."""
    query = QAQuery(raw_question="What is this?", expected_relations=[])
    ranked = path_ranker.rank_paths(graph_contexts, query)
    assert len(ranked) == 3


def test_explain_score(path_ranker, query_with_expected_relations, graph_contexts):
    """explain_score should return a readable breakdown."""
    ctx = graph_contexts[0]
    relations = [el for el in (ctx.subgraph or []) if isinstance(el, KGRelation)]
    expected = set(r.lower() for r in query_with_expected_relations.expected_relations)
    explanation = path_ranker.explain_score(relations, expected)
    assert "avg_confidence" in explanation or "relation_relevance" in explanation


def test_score_path_formula(path_ranker):
    """Verify the scoring formula produces expected values."""
    # Relations with high confidence + matching ontology should score high
    rel_match = KGRelation(
        source_id="e1", target_id="e2",
        relation_type="usesMethod", confidence=0.95,
    )
    rel_no_match = KGRelation(
        source_id="e1", target_id="e3",
        relation_type="locatedAt", confidence=0.95,
    )

    ctx_match = RetrievedContext(
        source=RetrievalSource.GRAPH, text="match", score=0.5,
        subgraph=[rel_match],
    )
    ctx_no_match = RetrievedContext(
        source=RetrievalSource.GRAPH, text="no match", score=0.5,
        subgraph=[rel_no_match],
    )

    expected = {"usesmethod"}  # lowercased to match _score_path logic
    score_match = path_ranker._score_path([rel_match], expected)
    score_no_match = path_ranker._score_path([rel_no_match], expected)

    # Matching relation should score higher
    assert score_match > score_no_match
