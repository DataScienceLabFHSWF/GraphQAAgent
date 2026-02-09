"""Tests for HybridRetriever and RRF."""

import pytest

from kgrag.core.models import RetrievalSource, RetrievedContext
from kgrag.retrieval.hybrid import reciprocal_rank_fusion


def _ctx(text: str, score: float, source: RetrievalSource = RetrievalSource.VECTOR) -> RetrievedContext:
    return RetrievedContext(source=source, text=text, score=score)


def test_rrf_basic_merge():
    vector = [_ctx("doc_a", 0.9), _ctx("doc_b", 0.7)]
    graph = [_ctx("doc_b other", 0.8), _ctx("doc_c", 0.6, RetrievalSource.GRAPH)]

    weights = {"vector": 0.5, "graph": 0.5}
    result = reciprocal_rank_fusion({"vector": vector, "graph": graph}, weights)

    assert len(result) >= 2
    # All should have scores > 0
    assert all(ctx.score > 0 for ctx in result)


def test_rrf_empty_inputs():
    result = reciprocal_rank_fusion({"vector": [], "graph": []}, {"vector": 0.5, "graph": 0.5})
    assert result == []


def test_rrf_single_source():
    vector = [_ctx("doc_a", 0.9), _ctx("doc_b", 0.7)]
    result = reciprocal_rank_fusion({"vector": vector}, {"vector": 1.0})
    assert len(result) == 2
