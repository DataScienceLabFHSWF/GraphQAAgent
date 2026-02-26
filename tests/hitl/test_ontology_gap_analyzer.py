"""Tests for the OntologyGapAnalyzer (structural + QA-driven gap detection)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kgrag.core.models import KGEntity, OntologyClass
from kgrag.hitl.ontology_gap_analyzer import (
    GapCandidate,
    GapReport,
    OntologyGapAnalyzer,
)


@pytest.fixture
def mock_neo4j() -> MagicMock:
    neo4j = MagicMock()
    neo4j._lbl = ""
    neo4j._config = MagicMock()
    neo4j._config.database = "neo4j"
    neo4j.driver = MagicMock()
    neo4j.get_entity_neighbours = AsyncMock(return_value=[])
    return neo4j


@pytest.fixture
def mock_fuseki() -> MagicMock:
    fuseki = MagicMock()
    fuseki.query = AsyncMock(return_value=[])
    return fuseki


@pytest.fixture
def mock_ollama() -> MagicMock:
    ollama = MagicMock()
    # Return a simple embedding vector
    ollama.embed = AsyncMock(return_value=[0.1] * 384)
    return ollama


@pytest.fixture
def analyzer(
    mock_neo4j: MagicMock,
    mock_fuseki: MagicMock,
    mock_ollama: MagicMock,
) -> OntologyGapAnalyzer:
    return OntologyGapAnalyzer(
        neo4j=mock_neo4j,
        fuseki=mock_fuseki,
        ollama=mock_ollama,
        min_frequency=1,
        similarity_threshold=0.9,  # high threshold so most things are "uncovered"
    )


class TestGapCandidate:
    def test_fields(self) -> None:
        gc = GapCandidate(
            entity_type="Reactor",
            representative_label="Reactor A",
            examples=["Reactor A", "Reactor B"],
            frequency=5,
            avg_confidence=0.7,
            closest_seed_class="Facility",
            semantic_distance=0.3,
        )
        assert gc.entity_type == "Reactor"
        assert gc.frequency == 5


class TestGapReport:
    def test_total_gaps(self) -> None:
        report = GapReport(
            total_abox_entities=100,
            covered_entities=80,
            uncovered_entities=20,
            coverage_pct=0.8,
            gap_candidates=[
                GapCandidate(entity_type="A", representative_label="A1", frequency=3),
                GapCandidate(entity_type="B", representative_label="B1", frequency=2),
            ],
        )
        assert report.total_gaps == 2


class TestClassifyEntities:
    @pytest.mark.asyncio
    async def test_exact_match_covered(self, analyzer: OntologyGapAnalyzer) -> None:
        entities = [
            KGEntity(id="e1", label="Facility A", entity_type="Facility"),
        ]
        tbox_labels = ["Facility", "Regulation", "Action"]

        covered, uncovered = await analyzer._classify_entities(entities, tbox_labels)
        assert len(covered) == 1
        assert len(uncovered) == 0

    @pytest.mark.asyncio
    async def test_no_match_uncovered(self, analyzer: OntologyGapAnalyzer) -> None:
        entities = [
            KGEntity(id="e1", label="Xyz123", entity_type="Xyz123Type"),
        ]
        tbox_labels = ["Facility", "Regulation"]

        # Return different embeddings for entity vs TBox labels
        call_count = 0

        async def varying_embed(text: str) -> list[float]:
            nonlocal call_count
            call_count += 1
            if "Xyz123" in text:
                return [0.0, 1.0, 0.0] * 128  # 384-d
            return [1.0, 0.0, 0.0] * 128  # orthogonal

        analyzer._ollama.embed = varying_embed

        covered, uncovered = await analyzer._classify_entities(entities, tbox_labels)
        assert len(uncovered) == 1


class TestBuildGapCandidates:
    @pytest.mark.asyncio
    async def test_groups_by_type(self, analyzer: OntologyGapAnalyzer) -> None:
        uncovered = [
            KGEntity(id="e1", label="A1", entity_type="TypeA", confidence=0.8),
            KGEntity(id="e2", label="A2", entity_type="TypeA", confidence=0.6),
            KGEntity(id="e3", label="B1", entity_type="TypeB", confidence=0.5),
        ]
        candidates = await analyzer._build_gap_candidates(uncovered, [])
        types = {c.entity_type for c in candidates}
        assert "TypeA" in types
        assert "TypeB" in types

    @pytest.mark.asyncio
    async def test_min_frequency_filter(self, analyzer: OntologyGapAnalyzer) -> None:
        analyzer._min_frequency = 3
        uncovered = [
            KGEntity(id="e1", label="A1", entity_type="TypeA"),
            KGEntity(id="e2", label="A2", entity_type="TypeA"),
        ]
        candidates = await analyzer._build_gap_candidates(uncovered, [])
        assert len(candidates) == 0  # TypeA has freq=2 < 3


class TestCosine:
    def test_cosine_identical(self) -> None:
        vec = [1.0, 0.0, 1.0]
        assert OntologyGapAnalyzer._cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_cosine_orthogonal(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert OntologyGapAnalyzer._cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_empty(self) -> None:
        assert OntologyGapAnalyzer._cosine_similarity([], []) == 0.0


class TestExportForExtender:
    def test_export_format(self, analyzer: OntologyGapAnalyzer) -> None:
        report = GapReport(
            total_abox_entities=50,
            covered_entities=40,
            uncovered_entities=10,
            coverage_pct=0.8,
            gap_candidates=[
                GapCandidate(
                    entity_type="Reactor",
                    representative_label="Reactor A",
                    examples=["A"],
                    frequency=5,
                    avg_confidence=0.7,
                    closest_seed_class="Facility",
                    semantic_distance=0.3,
                )
            ],
        )
        export = analyzer.export_for_ontology_extender(report)
        assert export["total_extracted_entities"] == 50
        assert export["coverage_pct"] == 0.8
        assert len(export["gap_candidates"]) == 1
        assert export["gap_candidates"][0]["entity_type"] == "Reactor"


class TestEscalateToHitl:
    def test_creates_proposals(self, analyzer: OntologyGapAnalyzer) -> None:
        from kgrag.hitl.change_proposals import ChangeProposalService
        from unittest.mock import MagicMock

        versioning = MagicMock()
        service = ChangeProposalService(versioning)

        report = GapReport(
            total_abox_entities=50,
            covered_entities=40,
            uncovered_entities=10,
            coverage_pct=0.8,
            gap_candidates=[
                GapCandidate(
                    entity_type="Reactor",
                    representative_label="Reactor A",
                    examples=["A"],
                    frequency=5,
                    avg_confidence=0.7,
                    closest_seed_class="Facility",
                    semantic_distance=0.3,
                )
            ],
        )
        proposals = analyzer.escalate_to_hitl(report, service)
        assert len(proposals) == 1
        assert proposals[0].proposal_type.value == "add_entity"
        assert "structural_analysis" in proposals[0].proposed_data["gap_source"]
