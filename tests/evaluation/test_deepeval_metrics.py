"""Tests for the DeepEval integration module and v3 benchmark loading."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Benchmark v3 loading tests
# ---------------------------------------------------------------------------


class TestBenchmarkV3Loading:
    """Verify the 50-question 5-tier benchmark loads correctly."""

    @pytest.fixture()
    def v3_path(self) -> Path:
        return Path(__file__).resolve().parents[2] / "data" / "qa_benchmarks" / "benchmark_v3.json"

    def test_v3_file_exists(self, v3_path: Path) -> None:
        assert v3_path.exists(), f"benchmark_v3.json not found at {v3_path}"

    def test_v3_has_50_questions(self, v3_path: Path) -> None:
        data = json.loads(v3_path.read_text())
        assert len(data) == 50

    def test_v3_difficulty_distribution(self, v3_path: Path) -> None:
        """10 questions per difficulty tier (1-5)."""
        data = json.loads(v3_path.read_text())
        from collections import Counter
        dist = Counter(item["difficulty"] for item in data)
        for tier in (1, 2, 3, 4, 5):
            assert dist[tier] == 10, f"Tier {tier}: expected 10, got {dist.get(tier, 0)}"

    def test_v3_unique_ids(self, v3_path: Path) -> None:
        data = json.loads(v3_path.read_text())
        ids = [item["question_id"] for item in data]
        assert len(ids) == len(set(ids)), "Duplicate question IDs found"

    def test_v3_tier5_empty_expected(self, v3_path: Path) -> None:
        """Tier 5 (open-ended) questions should have empty expected_answer."""
        data = json.loads(v3_path.read_text())
        tier5 = [item for item in data if item["difficulty"] == 5]
        for item in tier5:
            assert item["expected_answer"] == "", (
                f"{item['question_id']} should have empty expected_answer for tier 5"
            )
            assert item["expected_entities"] == [], (
                f"{item['question_id']} should have empty expected_entities for tier 5"
            )

    def test_v3_has_retrieval_complexity(self, v3_path: Path) -> None:
        """All questions should have a retrieval_complexity field."""
        data = json.loads(v3_path.read_text())
        for item in data:
            assert "retrieval_complexity" in item, f"{item['question_id']} missing retrieval_complexity"
            assert item["retrieval_complexity"] != "", f"{item['question_id']} has empty retrieval_complexity"

    def test_v3_loads_via_qa_dataset(self, v3_path: Path) -> None:
        """The dataset loader should handle the new fields."""
        from kgrag.evaluation.qa_dataset import QADataset
        ds = QADataset.load(v3_path)
        assert len(ds) == 50
        # Check that extra fields are loaded
        first = list(ds)[0]
        assert first.category != ""
        assert first.retrieval_complexity != ""


# ---------------------------------------------------------------------------
# DeepEvalResult tests
# ---------------------------------------------------------------------------


class TestDeepEvalResult:
    """Unit tests for the DeepEvalResult dataclass."""

    def test_to_dict(self) -> None:
        from kgrag.evaluation.deepeval_metrics import DeepEvalResult

        result = DeepEvalResult(
            answer_relevancy=0.85,
            faithfulness=0.92,
            correctness=0.78,
        )
        d = result.to_dict()
        assert d["answer_relevancy"] == 0.85
        assert d["faithfulness"] == 0.92
        assert d["correctness"] == 0.78
        assert d["contextual_relevancy"] is None
        assert d["errors"] is None

    def test_has_errors(self) -> None:
        from kgrag.evaluation.deepeval_metrics import DeepEvalResult

        ok = DeepEvalResult(answer_relevancy=0.9)
        assert not ok.has_errors

        bad = DeepEvalResult(errors={"faithfulness": "timeout"})
        assert bad.has_errors

    def test_default_none(self) -> None:
        from kgrag.evaluation.deepeval_metrics import DeepEvalResult

        r = DeepEvalResult()
        assert r.answer_relevancy is None
        assert r.faithfulness is None
        assert r.correctness is None


# ---------------------------------------------------------------------------
# DeepEvalEvaluator construction tests (mocked OllamaModel)
# ---------------------------------------------------------------------------


class TestDeepEvalEvaluator:
    """Test evaluator construction and metric selection."""

    @pytest.fixture(autouse=True)
    def _mock_deepeval(self) -> None:
        """Skip tests if deepeval is not installed."""
        try:
            import deepeval  # noqa: F401
        except ImportError:
            pytest.skip("deepeval not installed")

    def test_init_all_metrics(self) -> None:
        from kgrag.evaluation.deepeval_metrics import DeepEvalEvaluator

        ev = DeepEvalEvaluator(model_name="deepseek-r1:1.5b")
        assert len(ev._metric_objects) == 6  # noqa: SLF001

    def test_init_subset(self) -> None:
        from kgrag.evaluation.deepeval_metrics import DeepEvalEvaluator

        ev = DeepEvalEvaluator(
            model_name="deepseek-r1:1.5b",
            metrics=["faithfulness", "answer_relevancy"],
        )
        assert set(ev._metric_objects.keys()) == {"faithfulness", "answer_relevancy"}  # noqa: SLF001

    def test_init_unknown_metric(self) -> None:
        from kgrag.evaluation.deepeval_metrics import DeepEvalEvaluator

        with pytest.raises(ValueError, match="Unknown metrics"):
            DeepEvalEvaluator(
                model_name="deepseek-r1:1.5b",
                metrics=["nonexistent_metric"],
            )

    def test_build_test_case(self) -> None:
        from kgrag.evaluation.deepeval_metrics import DeepEvalEvaluator

        tc = DeepEvalEvaluator._build_test_case(  # noqa: SLF001
            question="What is AtG?",
            answer="The Atomgesetz.",
            expected_answer="Atomgesetz regulates nuclear energy.",
            retrieval_contexts=["The AtG is..."],
        )
        assert tc.input == "What is AtG?"
        assert tc.actual_output == "The Atomgesetz."
        assert len(tc.retrieval_context) == 1

    def test_skip_metrics_without_expected(self) -> None:
        """Correctness and recall should be skipped when no expected answer."""
        from kgrag.evaluation.deepeval_metrics import DeepEvalEvaluator

        ev = DeepEvalEvaluator(
            model_name="deepseek-r1:1.5b",
            metrics=["correctness"],
        )
        # Run sync with no expected_answer → correctness should be skipped
        result = ev.evaluate_sync(
            question="test?",
            answer="test answer",
            expected_answer=None,
            retrieval_contexts=["some context"],
        )
        assert result.correctness is None

    def test_skip_metrics_without_contexts(self) -> None:
        """Context-dependent metrics should be skipped when no contexts."""
        from kgrag.evaluation.deepeval_metrics import DeepEvalEvaluator

        ev = DeepEvalEvaluator(
            model_name="deepseek-r1:1.5b",
            metrics=["faithfulness"],
        )
        result = ev.evaluate_sync(
            question="test?",
            answer="test answer",
            retrieval_contexts=None,
        )
        assert result.faithfulness is None


# ---------------------------------------------------------------------------
# QAEvalResult and QABenchmarkItem model tests
# ---------------------------------------------------------------------------


class TestModelsDeepEvalFields:
    """Verify the new DeepEval fields on the core models."""

    def test_qa_eval_result_deepeval_fields(self) -> None:
        from kgrag.core.models import QAEvalResult

        r = QAEvalResult(
            question_id="q01",
            predicted_answer="test",
            expected_answer="test",
            deepeval_answer_relevancy=0.9,
            deepeval_faithfulness=0.85,
            deepeval_correctness=0.78,
        )
        assert r.deepeval_answer_relevancy == 0.9
        assert r.deepeval_faithfulness == 0.85
        assert r.deepeval_correctness == 0.78
        assert r.deepeval_contextual_relevancy is None

    def test_qa_eval_result_defaults_none(self) -> None:
        from kgrag.core.models import QAEvalResult

        r = QAEvalResult(question_id="q01", predicted_answer="a", expected_answer="b")
        assert r.deepeval_answer_relevancy is None
        assert r.deepeval_faithfulness is None
        assert r.deepeval_correctness is None

    def test_qa_benchmark_item_int_difficulty(self) -> None:
        from kgrag.core.models import QABenchmarkItem

        item = QABenchmarkItem(
            question_id="q01",
            question="test",
            expected_answer="answer",
            difficulty=3,
            category="legal_framework",
            retrieval_complexity="multi_hop_path_traversal",
        )
        assert item.difficulty == 3
        assert item.category == "legal_framework"
        assert item.retrieval_complexity == "multi_hop_path_traversal"

    def test_qa_benchmark_item_str_difficulty_backward_compat(self) -> None:
        from kgrag.core.models import QABenchmarkItem

        item = QABenchmarkItem(
            question_id="q01",
            question="test",
            expected_answer="answer",
            difficulty="medium",
        )
        assert item.difficulty == "medium"

    def test_strategy_comparison_deepeval_fields(self) -> None:
        from kgrag.core.models import StrategyComparison

        sc = StrategyComparison(
            strategy_name="hybrid_sota",
            avg_f1=0.8,
            avg_deepeval_answer_relevancy=0.85,
            avg_deepeval_faithfulness=0.9,
            avg_deepeval_correctness=0.75,
        )
        assert sc.avg_deepeval_answer_relevancy == 0.85
        assert sc.avg_deepeval_faithfulness == 0.9
        assert sc.avg_deepeval_correctness == 0.75
