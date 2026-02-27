"""Tests for active ontology learning (gap detection & proposal generation)."""

from __future__ import annotations
import pytest

from kgrag.retrieval.active_ontology import (
    OntologyGap,
    OntologyGapDetector,
    OntologyProposal,
    OntologyProposalGenerator,
)


# ---------------------------------------------------------------------------
# Fake OntologyContext for testing
# ---------------------------------------------------------------------------


class FakeOntologyContext:
    """Minimal stand-in for OntologyContext."""

    @property
    def schema_summary(self) -> str:
        return (
            "Class: NuclearFacility\n"
            "Class: DecommissioningProcess\n"
            "Class: WasteType\n"
            "Property: hasStatus\n"
            "Relation: involves\n"
        )


# ---------------------------------------------------------------------------
# OntologyGap
# ---------------------------------------------------------------------------


class TestOntologyGap:
    def test_creation(self):
        gap = OntologyGap(
            gap_type="missing_class",
            query_context="FuelRod",
            confidence=0.6,
        )
        assert gap.gap_type == "missing_class"
        assert gap.query_context == "FuelRod"
        assert gap.confidence == 0.6
        assert gap.entity_labels == []


# ---------------------------------------------------------------------------
# OntologyGapDetector
# ---------------------------------------------------------------------------


class TestOntologyGapDetector:
    def setup_method(self):
        self.detector = OntologyGapDetector(FakeOntologyContext())

    def test_detect_from_successful_lookup(self):
        """No gap when lookup returns positive results."""
        result = self.detector.detect_from_failed_lookup(
            "NuclearFacility",
            "Found class NuclearFacility with 5 properties",
        )
        assert result is None

    def test_detect_from_failed_lookup_class(self):
        """Gap detected when lookup returns 'not found'."""
        result = self.detector.detect_from_failed_lookup(
            "class FuelRod",
            "No matching classes found for FuelRod",
        )
        assert result is not None
        assert result.gap_type == "missing_class"
        assert result.confidence > 0

    def test_detect_from_failed_lookup_relation(self):
        """Gap type inferred from query keywords."""
        result = self.detector.detect_from_failed_lookup(
            "relation connects_to",
            "Unknown relation: connects_to",
        )
        assert result is not None
        assert result.gap_type == "missing_relation"

    def test_detect_from_failed_lookup_property(self):
        result = self.detector.detect_from_failed_lookup(
            "property halfLife",
            "No results for halfLife",
        )
        assert result is not None
        assert result.gap_type == "missing_property"

    def test_detect_from_low_evidence_sufficient(self):
        """No gap when evidence is sufficient."""
        result = self.detector.detect_from_low_evidence(
            "What is X?", ["EntityA"], evidence_count=3,
        )
        assert result is None

    def test_detect_from_low_evidence_insufficient(self):
        """Gap detected when evidence is sparse."""
        result = self.detector.detect_from_low_evidence(
            "What is X?", ["EntityA", "EntityB"], evidence_count=0,
        )
        assert result is not None
        assert result.gap_type == "weak_hierarchy"
        assert "EntityA" in result.entity_labels

    def test_detect_from_low_evidence_no_entities(self):
        """No gap when there are no entities to connect."""
        result = self.detector.detect_from_low_evidence(
            "What is X?", [], evidence_count=0,
        )
        assert result is None


# ---------------------------------------------------------------------------
# OntologyProposalGenerator
# ---------------------------------------------------------------------------


class TestOntologyProposalGenerator:
    def setup_method(self):
        self.generator = OntologyProposalGenerator(FakeOntologyContext())

    def test_propose_class(self):
        gap = OntologyGap(
            gap_type="missing_class",
            query_context="FuelRod",
            description="No class FuelRod found",
        )
        proposal = self.generator.generate_proposal(gap)
        assert proposal.proposal_type == "add_class"
        assert proposal.label == "FuelRod"
        assert "owl:Class" in proposal.turtle_fragment
        assert proposal.source_gap is gap

    def test_propose_property(self):
        gap = OntologyGap(
            gap_type="missing_property",
            query_context="halfLife",
            entity_labels=["FuelRod"],
            description="No property halfLife",
        )
        proposal = self.generator.generate_proposal(gap)
        assert proposal.proposal_type == "add_property"
        assert "DatatypeProperty" in proposal.turtle_fragment
        assert proposal.domain_class == "FuelRod"

    def test_propose_relation(self):
        gap = OntologyGap(
            gap_type="missing_relation",
            query_context="connects_to",
            entity_labels=["FacilityA", "FacilityB"],
            description="Missing relation",
        )
        proposal = self.generator.generate_proposal(gap)
        assert proposal.proposal_type == "add_relation"
        assert "ObjectProperty" in proposal.turtle_fragment
        assert proposal.domain_class == "FacilityA"
        assert proposal.range_class == "FacilityB"

    def test_propose_hierarchy_extension(self):
        gap = OntologyGap(
            gap_type="weak_hierarchy",
            query_context="nuclear linkage",
            entity_labels=["A", "B"],
            description="Weak hierarchy",
        )
        proposal = self.generator.generate_proposal(gap)
        assert proposal.proposal_type == "extend_hierarchy"
        assert "A" in proposal.label

    def test_unknown_gap_type_defaults_to_class(self):
        gap = OntologyGap(
            gap_type="unknown_type",
            query_context="something",
        )
        proposal = self.generator.generate_proposal(gap)
        assert proposal.proposal_type == "add_class"
