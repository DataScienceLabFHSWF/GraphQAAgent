"""Gap detection — identify missing knowledge from QA interactions.

When the QA pipeline returns low-confidence or empty answers, this module
captures those signals and converts them into actionable items:

- **Low-confidence answers** → potential ABox gaps (missing entities/relations)
- **Unanswerable questions** → potential TBox gaps (missing ontology classes)
- **Repeated failures** → prioritised research items for domain experts

Delegated implementation tasks
------------------------------
* TODO: Implement ``detect_gaps`` — analyse recent QA sessions for patterns.
* TODO: Auto-create ``ChangeProposal`` items from detected gaps.
* TODO: Integrate with the OntologyExtender repo for TBox-level fixes.
* TODO: Add a Streamlit page showing the gap detection dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DetectedGap:
    """A knowledge gap detected from QA interactions."""

    id: str = ""
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    gap_type: str = "unknown"  # "abox_missing_entity", "abox_missing_relation", "tbox_missing_class"
    trigger_question: str = ""
    confidence: float = 0.0  # confidence of the answer that triggered detection
    frequency: int = 1  # how many times similar questions were asked
    suggested_action: str = ""  # "add_entity", "add_relation", "extend_ontology"
    metadata: dict[str, Any] = field(default_factory=dict)


class GapDetector:
    """Detect knowledge gaps from QA interaction history.

    All methods are stubs.
    """

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        self.confidence_threshold = confidence_threshold
        self._detected_gaps: list[DetectedGap] = []

    async def analyse_answer(
        self,
        question: str,
        answer_text: str,
        confidence: float,
        *,
        reasoning_chain: list[str] | None = None,
        evidence_count: int = 0,
    ) -> DetectedGap | None:
        """Analyse a single QA answer for knowledge gaps.

        Returns a ``DetectedGap`` if the answer signals missing knowledge,
        or None if the answer seems adequate.

        Heuristics:
        - ``confidence < threshold`` → likely gap
        - ``evidence_count == 0`` → definitely missing knowledge
        - Hedging language in answer → possible gap
        - Empty reasoning chain → retrieval failure

        TODO (delegate): Implement heuristics and gap classification.
        """
        if confidence >= self.confidence_threshold and evidence_count > 0:
            return None

        gap = DetectedGap(
            trigger_question=question,
            confidence=confidence,
            gap_type=(
                "abox_missing_entity"
                if evidence_count == 0
                else "abox_weak_evidence"
            ),
            suggested_action="investigate",
        )
        self._detected_gaps.append(gap)
        logger.info(
            "hitl.gap_detected",
            gap_type=gap.gap_type,
            question=question[:80],
            confidence=confidence,
        )
        return gap

    def get_gaps(self, *, min_frequency: int = 1) -> list[DetectedGap]:
        """Return detected gaps, optionally filtered by frequency."""
        return [g for g in self._detected_gaps if g.frequency >= min_frequency]

    def clear_gaps(self) -> None:
        """Reset detected gaps."""
        self._detected_gaps.clear()
