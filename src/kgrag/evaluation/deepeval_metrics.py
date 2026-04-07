"""DeepEval integration — LLM-as-a-judge metrics via local Ollama model.

Wraps DeepEval's metric classes (AnswerRelevancy, Faithfulness,
ContextualRelevancy, GEval) using ``OllamaModel`` so all evaluation
runs fully locally — no external API calls.

Usage::

    from kgrag.evaluation.deepeval_metrics import DeepEvalEvaluator

    evaluator = DeepEvalEvaluator(model_name="deepseek-r1:1.5b")
    result = await evaluator.evaluate(
        question="What is the AtG?",
        answer="The Atomgesetz regulates nuclear energy in Germany.",
        expected_answer="The AtG is the German atomic energy act.",
        retrieval_contexts=["The Atomgesetz (AtG) regulates ..."],
    )
    print(result)
    # DeepEvalResult(answer_relevancy=0.85, faithfulness=0.92, ...)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — DeepEval is an optional heavy dependency
# ---------------------------------------------------------------------------

_DEEPEVAL_AVAILABLE = False
try:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        GEval,
    )
    from deepeval.models import OllamaModel
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    _DEEPEVAL_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class DeepEvalResult:
    """Container for all DeepEval metric scores from a single evaluation."""

    answer_relevancy: float | None = None
    faithfulness: float | None = None
    contextual_relevancy: float | None = None
    contextual_precision: float | None = None
    contextual_recall: float | None = None
    correctness: float | None = None  # GEval custom metric

    # Per-metric reasoning / explanations from the LLM judge
    reasons: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    raw_scores: dict[str, Any] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    def to_dict(self) -> dict[str, Any]:
        """Flat dictionary for tabular export."""
        return {
            "answer_relevancy": self.answer_relevancy,
            "faithfulness": self.faithfulness,
            "contextual_relevancy": self.contextual_relevancy,
            "contextual_precision": self.contextual_precision,
            "contextual_recall": self.contextual_recall,
            "correctness": self.correctness,
            "errors": self.errors if self.errors else None,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class DeepEvalEvaluator:
    """LLM-as-a-judge evaluator backed by a local Ollama model.

    Configurable metric selection — by default runs all metrics. Pass
    ``metrics=["faithfulness", "answer_relevancy"]`` to run a subset.

    Available metrics:
        - ``answer_relevancy``   — is the answer relevant to the question?
        - ``faithfulness``       — is the answer grounded in the context?
        - ``contextual_relevancy`` — are the retrieved contexts relevant?
        - ``contextual_precision`` — is relevant context ranked higher?
        - ``contextual_recall``  — does the context cover the expected answer?
        - ``correctness``        — GEval: is the answer correct vs. expected?

    Args:
        model_name: Ollama model name (e.g. ``"deepseek-r1:1.5b"``, ``"gemma4:e4b"``).
        base_url: Ollama API URL.
        temperature: Sampling temperature for the judge model.
        metrics: Which metrics to compute, or ``None`` for all.
        threshold: Default passing threshold for boolean pass/fail.
    """

    ALL_METRICS = frozenset({
        "answer_relevancy",
        "faithfulness",
        "contextual_relevancy",
        "contextual_precision",
        "contextual_recall",
        "correctness",
    })

    def __init__(
        self,
        *,
        model_name: str = "deepseek-r1:1.5b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        metrics: list[str] | None = None,
        threshold: float = 0.5,
    ) -> None:
        if not _DEEPEVAL_AVAILABLE:
            raise ImportError(
                "deepeval is required for LLM-as-a-judge metrics. "
                "Install with: pip install deepeval"
            )

        self.model = OllamaModel(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
        )
        self.threshold = threshold
        self._active_metrics = (
            set(metrics) if metrics else set(self.ALL_METRICS)
        )
        unknown = self._active_metrics - self.ALL_METRICS
        if unknown:
            raise ValueError(f"Unknown metrics: {unknown}")

        self._metric_objects = self._build_metrics()
        logger.info(
            "deepeval.evaluator_init",
            model=model_name,
            metrics=sorted(self._active_metrics),
        )

    # -----------------------------------------------------------------------
    # Metric construction
    # -----------------------------------------------------------------------

    def _build_metrics(self) -> dict[str, Any]:
        """Instantiate DeepEval metric objects for the active set."""
        built: dict[str, Any] = {}

        if "answer_relevancy" in self._active_metrics:
            built["answer_relevancy"] = AnswerRelevancyMetric(
                model=self.model,
                threshold=self.threshold,
            )

        if "faithfulness" in self._active_metrics:
            built["faithfulness"] = FaithfulnessMetric(
                model=self.model,
                threshold=self.threshold,
            )

        if "contextual_relevancy" in self._active_metrics:
            built["contextual_relevancy"] = ContextualRelevancyMetric(
                model=self.model,
                threshold=self.threshold,
            )

        if "contextual_precision" in self._active_metrics:
            built["contextual_precision"] = ContextualPrecisionMetric(
                model=self.model,
                threshold=self.threshold,
            )

        if "contextual_recall" in self._active_metrics:
            built["contextual_recall"] = ContextualRecallMetric(
                model=self.model,
                threshold=self.threshold,
            )

        if "correctness" in self._active_metrics:
            built["correctness"] = GEval(
                name="Correctness",
                model=self.model,
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                evaluation_steps=[
                    "Determine whether the actual output is factually correct "
                    "compared to the expected output.",
                    "Penalise omissions and additions of key facts.",
                    "Score 0.0 if the answer is completely wrong, 1.0 if it "
                    "captures all key facts from the expected output.",
                ],
                threshold=self.threshold,
            )

        return built

    # -----------------------------------------------------------------------
    # Build test case
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_test_case(
        question: str,
        answer: str,
        expected_answer: str | None,
        retrieval_contexts: list[str] | None,
    ) -> "LLMTestCase":
        """Create a DeepEval ``LLMTestCase`` from our QA data."""
        return LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected_answer or "",
            retrieval_context=retrieval_contexts or [],
        )

    # -----------------------------------------------------------------------
    # Measure one metric (sync + async)
    # -----------------------------------------------------------------------

    def _measure_sync(
        self,
        metric_name: str,
        test_case: "LLMTestCase",
    ) -> tuple[float | None, str | None, str | None]:
        """Run one metric synchronously. Returns (score, reason, error)."""
        metric = self._metric_objects.get(metric_name)
        if metric is None:
            return None, None, f"metric '{metric_name}' not active"

        try:
            metric.measure(test_case)
            score = metric.score
            reason = getattr(metric, "reason", None)
            return score, reason, None
        except Exception as exc:
            logger.warning("deepeval.metric_failed", metric=metric_name, error=str(exc))
            return None, None, str(exc)

    async def _measure_async(
        self,
        metric_name: str,
        test_case: "LLMTestCase",
    ) -> tuple[float | None, str | None, str | None]:
        """Run one metric asynchronously."""
        metric = self._metric_objects.get(metric_name)
        if metric is None:
            return None, None, f"metric '{metric_name}' not active"

        try:
            await metric.a_measure(test_case)
            score = metric.score
            reason = getattr(metric, "reason", None)
            return score, reason, None
        except Exception as exc:
            logger.warning("deepeval.metric_failed", metric=metric_name, error=str(exc))
            return None, None, str(exc)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def evaluate_sync(
        self,
        question: str,
        answer: str,
        *,
        expected_answer: str | None = None,
        retrieval_contexts: list[str] | None = None,
    ) -> DeepEvalResult:
        """Run all active metrics synchronously.

        Use this when you're not inside an async context.
        """
        test_case = self._build_test_case(
            question, answer, expected_answer, retrieval_contexts
        )
        result = DeepEvalResult()

        for metric_name in sorted(self._active_metrics):
            # Skip metrics that need expected_output if none provided
            if metric_name in ("correctness", "contextual_precision", "contextual_recall") and not expected_answer:
                continue
            # Skip context-dependent metrics if no contexts
            if metric_name in ("faithfulness", "contextual_relevancy", "contextual_precision", "contextual_recall") and not retrieval_contexts:
                continue

            score, reason, error = self._measure_sync(metric_name, test_case)
            if error:
                result.errors[metric_name] = error
            else:
                setattr(result, metric_name, score)
                result.raw_scores[metric_name] = score
                if reason:
                    result.reasons[metric_name] = reason

        return result

    async def evaluate(
        self,
        question: str,
        answer: str,
        *,
        expected_answer: str | None = None,
        retrieval_contexts: list[str] | None = None,
    ) -> DeepEvalResult:
        """Run all active metrics asynchronously.

        Metrics are evaluated sequentially to avoid overwhelming the
        local Ollama instance.
        """
        test_case = self._build_test_case(
            question, answer, expected_answer, retrieval_contexts
        )
        result = DeepEvalResult()

        for metric_name in sorted(self._active_metrics):
            # Skip metrics that need expected_output if none provided
            if metric_name in ("correctness", "contextual_precision", "contextual_recall") and not expected_answer:
                continue
            # Skip context-dependent metrics if no contexts
            if metric_name in ("faithfulness", "contextual_relevancy", "contextual_precision", "contextual_recall") and not retrieval_contexts:
                continue

            score, reason, error = await self._measure_async(metric_name, test_case)
            if error:
                result.errors[metric_name] = error
            else:
                setattr(result, metric_name, score)
                result.raw_scores[metric_name] = score
                if reason:
                    result.reasons[metric_name] = reason

        return result

    async def evaluate_batch(
        self,
        items: list[dict[str, Any]],
    ) -> list[DeepEvalResult]:
        """Evaluate a batch of QA items.

        Each item should have keys: ``question``, ``answer``,
        and optionally ``expected_answer``, ``retrieval_contexts``.

        Returns one ``DeepEvalResult`` per item.
        """
        results = []
        for i, item in enumerate(items):
            logger.info("deepeval.batch_progress", current=i + 1, total=len(items))
            r = await self.evaluate(
                question=item["question"],
                answer=item["answer"],
                expected_answer=item.get("expected_answer"),
                retrieval_contexts=item.get("retrieval_contexts"),
            )
            results.append(r)
        return results
