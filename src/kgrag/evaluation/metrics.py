"""QA metrics (C3.5.2) — accuracy, F1, faithfulness, relevance.

All metrics operate on normalised token sequences for language-agnostic
comparison (works for both German and English).
"""

from __future__ import annotations

import re
import string

import structlog

from kgrag.core.models import RetrievedContext

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenise(text: str) -> list[str]:
    """Split normalised text into tokens."""
    return _normalise(text).split()


# ---------------------------------------------------------------------------
# Token-level metrics
# ---------------------------------------------------------------------------


def compute_token_f1(predicted: str, expected: str) -> float:
    """Compute token-level F1 between predicted and expected answers.

    This is the standard SQuAD-style F1 metric.
    """
    pred_tokens = _tokenise(predicted)
    exp_tokens = _tokenise(expected)

    if not pred_tokens or not exp_tokens:
        return 1.0 if pred_tokens == exp_tokens else 0.0

    common = set(pred_tokens) & set(exp_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(exp_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(predicted: str, expected: str) -> bool:
    """Normalised exact match."""
    return _normalise(predicted) == _normalise(expected)


# ---------------------------------------------------------------------------
# Faithfulness & relevance (heuristic — can be replaced with LLM-based)
# ---------------------------------------------------------------------------


def compute_faithfulness(
    answer: str,
    contexts: list[RetrievedContext],
) -> float:
    """Estimate fraction of answer tokens that appear in the context.

    A simple proxy for faithfulness / groundedness.  Higher = less hallucination.
    """
    answer_tokens = set(_tokenise(answer))
    if not answer_tokens:
        return 1.0

    context_text = " ".join(ctx.text for ctx in contexts)
    context_tokens = set(_tokenise(context_text))

    grounded = answer_tokens & context_tokens
    return len(grounded) / len(answer_tokens)


def compute_context_relevance(
    question: str,
    contexts: list[RetrievedContext],
) -> float:
    """Average token overlap between question and each context.

    Simple proxy for context relevance.
    """
    if not contexts:
        return 0.0

    q_tokens = set(_tokenise(question))
    if not q_tokens:
        return 0.0

    scores = []
    for ctx in contexts:
        ctx_tokens = set(_tokenise(ctx.text))
        overlap = q_tokens & ctx_tokens
        scores.append(len(overlap) / len(q_tokens))

    return sum(scores) / len(scores)
