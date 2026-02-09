"""QA benchmark dataset loader (C3.5.1).

Loads gold-standard QA datasets for evaluation from JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from kgrag.core.models import QABenchmarkItem

logger = structlog.get_logger(__name__)


class QADataset:
    """Gold-standard QA benchmark dataset."""

    def __init__(self, items: list[QABenchmarkItem]) -> None:
        self._items = items

    @classmethod
    def load(cls, path: str | Path) -> QADataset:
        """Load benchmark from a JSON file.

        Expected format::

            [
                {
                    "question_id": "q01",
                    "question": "...",
                    "expected_answer": "...",
                    "expected_entities": ["Entity1"],
                    "difficulty": "medium",
                    "question_type": "factoid"
                },
                ...
            ]
        """
        with open(path) as f:
            data = json.load(f)

        items = [
            QABenchmarkItem(
                question_id=item["question_id"],
                question=item["question"],
                expected_answer=item["expected_answer"],
                expected_entities=item.get("expected_entities", []),
                difficulty=item.get("difficulty", "medium"),
                question_type=item.get("question_type", "factoid"),
                competency_question_id=item.get("competency_question_id"),
            )
            for item in data
        ]
        logger.info("qa_dataset.loaded", path=str(path), items=len(items))
        return cls(items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):  # noqa: ANN204
        return iter(self._items)

    def filter_by_type(self, question_type: str) -> QADataset:
        """Return a filtered dataset containing only the given question type."""
        return QADataset([i for i in self._items if i.question_type == question_type])

    def filter_by_difficulty(self, difficulty: str) -> QADataset:
        """Return a filtered dataset containing only the given difficulty."""
        return QADataset([i for i in self._items if i.difficulty == difficulty])
