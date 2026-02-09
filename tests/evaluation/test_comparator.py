"""Tests for StrategyComparator."""

from kgrag.core.models import QAEvalResult, StrategyComparison
from kgrag.evaluation.comparator import StrategyComparator


def _result(strategy: str, f1: float, em: bool = False) -> QAEvalResult:
    return QAEvalResult(
        question_id="q1",
        predicted_answer="pred",
        expected_answer="exp",
        f1_score=f1,
        exact_match=em,
        retrieval_strategy=strategy,
    )


def test_compare_groups_by_strategy():
    results = [
        _result("vector", 0.8, True),
        _result("vector", 0.6),
        _result("hybrid", 0.9, True),
        _result("hybrid", 0.7),
    ]
    comparator = StrategyComparator()
    comparisons = comparator.compare(results)

    assert len(comparisons) == 2
    # Hybrid should be first (higher avg F1)
    assert comparisons[0].strategy_name == "hybrid"
    assert comparisons[0].avg_f1 == pytest.approx(0.8)
    assert comparisons[1].strategy_name == "vector"
    assert comparisons[1].avg_f1 == pytest.approx(0.7)


def test_compare_empty():
    comparator = StrategyComparator()
    assert comparator.compare([]) == []


# Need this import for pytest.approx
import pytest  # noqa: E402
