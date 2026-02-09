"""Tests for evaluation metrics."""

from kgrag.core.models import RetrievalSource, RetrievedContext
from kgrag.evaluation.metrics import (
    compute_context_relevance,
    compute_exact_match,
    compute_faithfulness,
    compute_token_f1,
)


def test_token_f1_exact():
    assert compute_token_f1("hello world", "hello world") == 1.0


def test_token_f1_partial():
    score = compute_token_f1("hello world foo", "hello world bar")
    assert 0.0 < score < 1.0


def test_token_f1_no_overlap():
    assert compute_token_f1("alpha beta", "gamma delta") == 0.0


def test_exact_match_normalised():
    assert compute_exact_match("Hello World!", "hello world") is True
    assert compute_exact_match("foo", "bar") is False


def test_faithfulness():
    contexts = [
        RetrievedContext(source=RetrievalSource.VECTOR, text="hello world context", score=0.9)
    ]
    score = compute_faithfulness("hello world answer", contexts)
    assert score > 0.0


def test_context_relevance():
    contexts = [
        RetrievedContext(source=RetrievalSource.VECTOR, text="what is test entity", score=0.9)
    ]
    score = compute_context_relevance("what is test", contexts)
    assert score > 0.0
