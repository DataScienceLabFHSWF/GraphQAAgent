"""Tests for HITL gap detection."""

from __future__ import annotations

import pytest

from kgrag.hitl.gap_detection import GapDetector


class TestGapDetector:
    """Validate gap detection heuristics."""

    @pytest.fixture
    def detector(self) -> GapDetector:
        return GapDetector(confidence_threshold=0.5)

    @pytest.mark.asyncio
    async def test_no_gap_above_threshold(self, detector: GapDetector) -> None:
        gap = await detector.analyse_answer(
            question="Q", answer_text="A", confidence=0.8, evidence_count=3
        )
        assert gap is None

    @pytest.mark.asyncio
    async def test_gap_below_threshold(self, detector: GapDetector) -> None:
        gap = await detector.analyse_answer(
            question="Q", answer_text="A", confidence=0.2, evidence_count=1
        )
        assert gap is not None
        assert gap.gap_type == "abox_weak_evidence"

    @pytest.mark.asyncio
    async def test_gap_no_evidence(self, detector: GapDetector) -> None:
        gap = await detector.analyse_answer(
            question="Q", answer_text="A", confidence=0.1, evidence_count=0
        )
        assert gap is not None
        assert gap.gap_type == "abox_missing_entity"

    def test_get_gaps(self, detector: GapDetector) -> None:
        assert detector.get_gaps() == []
