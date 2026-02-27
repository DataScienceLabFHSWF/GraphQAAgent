"""Tests for KG-constrained answer validation."""

from __future__ import annotations
import pytest

from kgrag.retrieval.kg_constrained import (
    KGConstrainedValidator,
    ValidationResult,
)
from kgrag.core.models import KGEntity, QAAnswer, RetrievedContext, RetrievalSource


class TestValidationResult:
    def test_defaults(self):
        r = ValidationResult()
        assert r.is_valid is True
        assert r.entity_mentions == []
        assert r.unknown_entities == []
        assert r.corrections == {}

    def test_with_data(self):
        r = ValidationResult(
            is_valid=False,
            unknown_entities=["FooBar"],
            corrections={"FooBar": "FooBaz"},
        )
        assert r.is_valid is False
        assert r.corrections["FooBar"] == "FooBaz"


class TestKGConstrainedValidator:
    def _make_answer(self, text: str, entities: list[KGEntity] | None = None, evidence_texts: list[str] | None = None) -> QAAnswer:
        evidence = [
            RetrievedContext(source=RetrievalSource.VECTOR, text=t, score=0.8)
            for t in (evidence_texts or [])
        ]
        return QAAnswer(
            question="test?",
            answer_text=text,
            cited_entities=entities or [],
            evidence=evidence,
        )

    def test_extract_entity_mentions_from_cited(self):
        ent = KGEntity(id="1", label="AtG", entity_type="Law")
        answer = self._make_answer("The AtG regulates nuclear safety.", entities=[ent])

        # Use a real Neo4j connector mock
        class FakeNeo4j:
            async def query(self, q, p):
                return []

        validator = KGConstrainedValidator(FakeNeo4j())
        mentions = validator._extract_entity_mentions(answer)
        assert "AtG" in mentions

    def test_apply_corrections(self):
        answer = self._make_answer("The Atomgezetz is important.")
        result = ValidationResult(
            corrections={"Atomgezetz": "Atomgesetz"},
        )

        class FakeNeo4j:
            async def query(self, q, p):
                return []

        validator = KGConstrainedValidator(FakeNeo4j())
        corrected = validator.apply_corrections(answer, result)
        assert "Atomgesetz" in corrected
        assert "Atomgezetz" not in corrected

    def test_evidence_consistency_supported(self):
        evidence_texts = [
            "Nuclear decommissioning involves dismantling of nuclear facilities and waste management."
        ]
        answer = self._make_answer(
            "Nuclear decommissioning involves dismantling of nuclear facilities.",
            evidence_texts=evidence_texts,
        )

        class FakeNeo4j:
            async def query(self, q, p):
                return []

        validator = KGConstrainedValidator(FakeNeo4j())
        unsupported = validator._check_evidence_consistency(answer)
        # The answer sentence has high overlap with evidence
        assert len(unsupported) == 0

    def test_evidence_consistency_unsupported(self):
        answer = self._make_answer(
            "The quantum flux capacitor reverberates through spacetime continuum dynamics.",
            evidence_texts=["Nuclear safety regulations apply to power plants."],
        )

        class FakeNeo4j:
            async def query(self, q, p):
                return []

        validator = KGConstrainedValidator(FakeNeo4j())
        unsupported = validator._check_evidence_consistency(answer)
        assert len(unsupported) >= 1

    @pytest.mark.asyncio
    async def test_validate_known_entity(self):
        """Entity found in Neo4j → verified."""
        ent = KGEntity(id="1", label="AtG", entity_type="Law")
        answer = self._make_answer("The AtG is valid.", entities=[ent])

        class FakeNeo4j:
            async def query(self, q, p):
                if "label" in q:
                    return [{"n": {"id": "1", "type": "Law", "label": "AtG"}}]
                return []

        validator = KGConstrainedValidator(FakeNeo4j())
        result = await validator.validate(answer)
        assert "AtG" in result.verified_entities
        assert len(result.unknown_entities) == 0

    @pytest.mark.asyncio
    async def test_validate_unknown_entity(self):
        """Entity not found → unknown + fuzzy match attempted."""
        ent = KGEntity(id="1", label="AtomGezetz", entity_type="Law")
        answer = self._make_answer("The AtomGezetz is important.", entities=[ent])

        class FakeNeo4j:
            async def query(self, q, p):
                if "n.label = $label" in q:
                    return []  # exact match fails
                if "CONTAINS" in q:
                    return [{"label": "Atomgesetz", "name": None}]
                return []

        validator = KGConstrainedValidator(FakeNeo4j())
        result = await validator.validate(answer)
        assert "AtomGezetz" in result.unknown_entities
        assert result.corrections.get("AtomGezetz") == "Atomgesetz"
