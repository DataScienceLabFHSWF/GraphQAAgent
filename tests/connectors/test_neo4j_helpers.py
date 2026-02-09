"""Unit tests for Neo4jConnector record helpers — KGB compatibility."""

from __future__ import annotations

import json
import pytest

from kgrag.connectors.neo4j import Neo4jConnector
from kgrag.core.config import Neo4jConfig


@pytest.fixture()
def connector() -> Neo4jConnector:
    """Create a connector without actually connecting."""
    return Neo4jConnector(Neo4jConfig())


# ── _record_to_entity ─────────────────────────────────────────────────────


class TestRecordToEntity:
    """Test the static entity conversion helper."""

    def test_ideal_schema(self, connector: Neo4jConnector) -> None:
        """Node already has flat properties matching KG-RAG spec."""
        node = {
            "id": "e-1",
            "label": "Reactor A",
            "entity_type": "NuclearFacility",
            "confidence": 0.92,
            "description": "A pressurised water reactor",
        }
        ent = connector._record_to_entity(node)
        assert ent.id == "e-1"
        assert ent.label == "Reactor A"
        assert ent.entity_type == "NuclearFacility"
        assert ent.confidence == 0.92
        assert ent.description == "A pressurised water reactor"

    def test_kgb_json_properties(self, connector: Neo4jConnector) -> None:
        """KGB Neo4jGraphStore stores props as JSON in `properties` field."""
        node = {
            "id": "e-2",
            "label": "Method X",
            "node_type": "DecommissionMethod",
            "properties": json.dumps({"confidence": 0.8, "description": "A method"}),
        }
        ent = connector._record_to_entity(node)
        assert ent.id == "e-2"
        assert ent.label == "Method X"
        # entity_type derived from node_type
        assert ent.entity_type == "DecommissionMethod"
        # confidence unpacked from JSON
        assert ent.confidence == 0.8
        # description unpacked from JSON
        assert ent.description == "A method"

    def test_entity_type_from_neo4j_labels(self, connector: Neo4jConnector) -> None:
        """Derive entity_type from Neo4j labels when property is absent."""
        node = {"id": "e-3", "label": "Turbine B", "confidence": 0.7}
        ent = connector._record_to_entity(
            node, neo4j_labels=["Entity", "Component"],
        )
        # Should pick "Component" (not the generic "Entity")
        assert ent.entity_type == "Component"

    def test_entity_type_fallback_single_label(self, connector: Neo4jConnector) -> None:
        """If only the generic label exists, use it."""
        node = {"id": "e-4", "label": "Unknown"}
        ent = connector._record_to_entity(node, neo4j_labels=["Entity"])
        assert ent.entity_type == "Entity"

    def test_description_defaults_to_empty(self, connector: Neo4jConnector) -> None:
        """When no description is present, default to empty string."""
        node = {"id": "e-5", "label": "X"}
        ent = connector._record_to_entity(node)
        assert ent.description == ""

    def test_missing_everything_graceful(self, connector: Neo4jConnector) -> None:
        """Empty dict should not crash — all fields get defaults."""
        ent = connector._record_to_entity({})
        assert ent.id == ""
        assert ent.label == ""
        assert ent.entity_type == ""
        assert ent.confidence == 0.0
        assert ent.description == ""


# ── _record_to_relation ───────────────────────────────────────────────────


class TestRecordToRelation:
    """Test the static relation conversion helper.

    KGRelation no longer has id, evidence_text, or properties.
    Identification is by (source_id, relation_type, target_id) tuple.
    """

    def test_ideal_schema(self, connector: Neo4jConnector) -> None:
        """Relationship has all expected flat properties."""
        rel = {
            "source_id": "e-1",
            "target_id": "e-2",
            "type": "USES_METHOD",
            "confidence": 0.85,
        }
        r = connector._record_to_relation(rel)
        assert r.source_id == "e-1"
        assert r.target_id == "e-2"
        assert r.relation_type == "USES_METHOD"
        assert r.confidence == 0.85

    def test_kgb_overrides(self, connector: Neo4jConnector) -> None:
        """KGB rel has no source_id/target_id/type — use Cypher overrides."""
        rel = {"confidence": 0.7}
        r = connector._record_to_relation(
            rel, rel_type="INVOLVES", src_id="e-10", tgt_id="e-20",
        )
        assert r.source_id == "e-10"
        assert r.target_id == "e-20"
        assert r.relation_type == "INVOLVES"

    def test_kgb_json_properties(self, connector: Neo4jConnector) -> None:
        """KGB Neo4jGraphStore serializes rel properties as JSON."""
        rel = {
            "properties": json.dumps({
                "confidence": 0.6,
            }),
        }
        r = connector._record_to_relation(rel, rel_type="DESCRIBES", src_id="a", tgt_id="b")
        assert r.confidence == 0.6
        assert r.relation_type == "DESCRIBES"

    def test_predicate_field(self, connector: Neo4jConnector) -> None:
        """KGB uses 'predicate' field — should map to relation_type."""
        rel = {"predicate": "requiresPermit", "confidence": 0.75}
        r = connector._record_to_relation(rel, src_id="a", tgt_id="b")
        assert r.relation_type == "requiresPermit"

    def test_property_priority(self, connector: Neo4jConnector) -> None:
        """Flat properties take priority over Cypher overrides."""
        rel = {
            "source_id": "flat-src",
            "target_id": "flat-tgt",
            "type": "FLAT_TYPE",
        }
        r = connector._record_to_relation(
            rel, rel_type="OVERRIDE", src_id="override-src", tgt_id="override-tgt",
        )
        assert r.source_id == "flat-src"
        assert r.target_id == "flat-tgt"
        assert r.relation_type == "FLAT_TYPE"

    def test_empty_dict_graceful(self, connector: Neo4jConnector) -> None:
        """Empty dict should not crash."""
        r = connector._record_to_relation({})
        assert r.source_id == ""
        assert r.target_id == ""
        assert r.relation_type == ""


# ── Config ────────────────────────────────────────────────────────────────


class TestNodeLabelConfig:
    """Verify node_label config plumbing."""

    def test_default_label(self) -> None:
        c = Neo4jConnector(Neo4jConfig())
        assert c._label == "Entity"

    def test_custom_label(self) -> None:
        c = Neo4jConnector(Neo4jConfig(node_label="Node"))
        assert c._label == "Node"
