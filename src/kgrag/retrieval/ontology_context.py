"""OntologyContext — shared TBox knowledge loaded once at startup.

Provides a structured summary of the ontology (classes, hierarchy, properties,
domain/range constraints) that every retrieval component can use:
- **CypherRetriever**: injects class/property catalog into Cypher generation prompt
- **EntityLinker**: maps detected types to ontology classes for better linking
- **GraphRetriever**: knows which relations to follow for a given entity type
- **VectorRetriever**: expands queries with synonyms/subclass names
- **HybridRetriever**: adaptive weights informed by ontology richness
- **AgenticGraphRAG**: full schema awareness for tool-calling agent

The ontology is loaded from Fuseki (SPARQL) and cached in memory.
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass, field
from typing import Any

from kgrag.connectors.fuseki import FusekiConnector

logger = structlog.get_logger(__name__)


@dataclass
class OntologyClass:
    """A class from the TBox with its position in the hierarchy."""

    name: str  # Local name (e.g. "Facility")
    uri: str
    parent: str | None = None  # Parent class local name
    children: list[str] = field(default_factory=list)
    properties_as_domain: list[str] = field(default_factory=list)  # Outgoing relations
    properties_as_range: list[str] = field(default_factory=list)  # Incoming relations


@dataclass
class OntologyProperty:
    """An object or datatype property from the TBox."""

    name: str  # Local name (e.g. "hasAction")
    uri: str
    prop_type: str  # "object" or "data"
    domain: str  # Class local name
    range: str  # Class or XSD type local name


class OntologyContext:
    """Shared ontology (TBox) knowledge loaded once from Fuseki.

    Call ``await load()`` at startup. Then use the cached data:
    - ``classes``: dict of class name → OntologyClass
    - ``properties``: dict of property name → OntologyProperty
    - ``schema_summary``: pre-formatted text for LLM prompts
    - ``neo4j_to_ontology``: mapping from Neo4j labels to ontology class names
    """

    def __init__(self, fuseki: FusekiConnector) -> None:
        self._fuseki = fuseki
        self.classes: dict[str, OntologyClass] = {}
        self.properties: dict[str, OntologyProperty] = {}
        self.schema_summary: str = ""
        self.neo4j_to_ontology: dict[str, str] = {}
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        """Load the full ontology from Fuseki and build cached structures."""
        try:
            await self._load_classes()
            await self._load_properties()
            self._build_hierarchy()
            self._build_neo4j_mapping()
            self.schema_summary = self._build_schema_summary()
            self._loaded = True
            logger.info(
                "ontology_context.loaded",
                classes=len(self.classes),
                properties=len(self.properties),
                summary_chars=len(self.schema_summary),
            )
        except Exception as exc:
            logger.warning("ontology_context.load_failed", error=str(exc))
            # Build a minimal fallback summary from common knowledge
            self.schema_summary = self._fallback_summary()
            self._loaded = True  # Mark as loaded even with fallback

    async def _load_classes(self) -> None:
        """Load all owl:Class instances with their hierarchy."""
        rows = await self._fuseki.query("""
        SELECT ?cls ?parent WHERE {
          ?cls a owl:Class .
          OPTIONAL { ?cls rdfs:subClassOf ?parent . ?parent a owl:Class }
        }
        """)
        for row in rows:
            uri = row.get("cls", "")
            name = self._local_name(uri)
            parent_uri = row.get("parent", "")
            parent_name = self._local_name(parent_uri) if parent_uri else None

            if name not in self.classes:
                self.classes[name] = OntologyClass(name=name, uri=uri, parent=parent_name)
            elif parent_name:
                self.classes[name].parent = parent_name

    async def _load_properties(self) -> None:
        """Load all object and datatype properties with domain/range."""
        rows = await self._fuseki.query("""
        SELECT ?prop ?type ?domain ?range WHERE {
          {
            ?prop a owl:ObjectProperty .
            BIND("object" AS ?type)
          } UNION {
            ?prop a owl:DatatypeProperty .
            BIND("data" AS ?type)
          }
          OPTIONAL { ?prop rdfs:domain ?domain }
          OPTIONAL { ?prop rdfs:range ?range }
        }
        """)
        for row in rows:
            uri = row.get("prop", "")
            name = self._local_name(uri)
            prop_type = row.get("type", "object")
            domain = self._local_name(row.get("domain", ""))
            range_ = self._local_name(row.get("range", ""))

            self.properties[name] = OntologyProperty(
                name=name,
                uri=uri,
                prop_type=prop_type,
                domain=domain,
                range=range_,
            )

    def _build_hierarchy(self) -> None:
        """Wire parent → children links and property associations."""
        for cls in self.classes.values():
            if cls.parent and cls.parent in self.classes:
                parent = self.classes[cls.parent]
                if cls.name not in parent.children:
                    parent.children.append(cls.name)

        for prop in self.properties.values():
            if prop.domain in self.classes:
                self.classes[prop.domain].properties_as_domain.append(prop.name)
            if prop.range in self.classes:
                self.classes[prop.range].properties_as_range.append(prop.name)

    def _build_neo4j_mapping(self) -> None:
        """Map Neo4j node labels to ontology class names.

        Direct match by name works for most types (Facility, Action, State, etc.)
        Special cases for the law graph labels that differ.
        """
        # Direct 1:1 where Neo4j label == ontology class name
        for name in self.classes:
            self.neo4j_to_ontology[name] = name

        # Known special mappings (law graph uses German labels)
        self.neo4j_to_ontology.update({
            "Gesetzbuch": "Regulation",
            "Paragraf": "Regulation",
            "Abschnitt": "Regulation",
            "Regulation": "Regulation",
            "Person": "Organization",
            "Location": "Facility",
            "Concept": "ProblemObject",
            "DomainObject": "ProblemObject",
            "Document": "Documentation",
            "DocumentSection": "Documentation",
            "FacilityComponent": "Component",
            "Effect": "ActionEffect",
        })

    def _build_schema_summary(self) -> str:
        """Build a structured text summary for LLM prompts."""
        lines = ["ONTOLOGY SCHEMA (TBox):"]
        lines.append("")

        # Class hierarchy
        lines.append("Classes (with hierarchy):")
        roots = [c for c in self.classes.values() if c.parent is None]
        for root in sorted(roots, key=lambda c: c.name):
            self._render_class_tree(root, lines, indent=1)

        lines.append("")

        # Properties grouped by domain
        lines.append("Properties (domain → range):")
        by_domain: dict[str, list[OntologyProperty]] = {}
        for prop in self.properties.values():
            by_domain.setdefault(prop.domain, []).append(prop)

        for domain in sorted(by_domain):
            props = by_domain[domain]
            for p in sorted(props, key=lambda x: x.name):
                arrow = "→" if p.prop_type == "object" else "⇒"
                lines.append(f"  {p.domain}.{p.name} {arrow} {p.range}")

        lines.append("")

        # Relationship patterns for Cypher
        lines.append("Key Cypher patterns:")
        lines.append("  (PlanningDomain)-[:hasAction]->(Action)-[:hasEffect]->(ActionEffect)")
        lines.append("  (PlanningDomain)-[:hasProblem]->(PlanningProblem)-[:hasGoalState]->(GoalState)")
        lines.append("  (PlanningDomain)-[:hasConstant]->(DomainConstant)")
        lines.append("  (PlanningDomain)-[:hasRequirement]->(DomainRequirement)-[:solvedBy]->(Planner)")
        lines.append("  (entity)-[:LINKED_GOVERNED_BY]->(Paragraf)-[:teilVon]->(Gesetzbuch)")
        lines.append("  (Paragraf)-[:referenziert]->(Paragraf)")

        return "\n".join(lines)

    def _render_class_tree(
        self,
        cls: OntologyClass,
        lines: list[str],
        indent: int,
    ) -> None:
        """Recursive tree rendering for a class and its children."""
        prefix = "  " * indent
        props = ", ".join(cls.properties_as_domain[:5]) if cls.properties_as_domain else ""
        suffix = f" [{props}]" if props else ""
        lines.append(f"{prefix}- {cls.name}{suffix}")
        for child_name in sorted(cls.children):
            if child_name in self.classes:
                self._render_class_tree(self.classes[child_name], lines, indent + 1)

    def _fallback_summary(self) -> str:
        """Minimal schema summary when Fuseki is unavailable."""
        return """ONTOLOGY SCHEMA (TBox) — from domain knowledge:
Classes: PlanningDomain, PlanningProblem, Action, MacroAction, ActionEffect,
  ActionPrecondition, State, GoalState, InitialState, DomainConstant,
  DomainPredicate, DomainRequirement, ProblemObject, Facility, Component,
  Organization, Process, NuclearMaterial, WasteCategory, Transport, Permit,
  Regulation, Documentation, Activity, SafetySystem, Planner, PlannerType,
  Parameter, ParameterType, Plan
Key relationships: hasAction, hasEffect, hasPrecondition, hasParameter,
  hasProblem, hasGoalState, hasInitialState, hasObject, hasConstant,
  hasPredicate, hasRequirement, solvedBy, solvesRequirement, governedBy,
  referencesLaw, involves, produces, requires, issuedBy, isGeneratedBy
Law graph: Gesetzbuch -[:teilVon]<- Paragraf -[:referenziert]-> Paragraf
  Domain entities -[:LINKED_GOVERNED_BY]-> Paragraf"""

    def get_relations_for_type(self, entity_type: str) -> list[str]:
        """Return relation names relevant for a given entity type."""
        onto_type = self.neo4j_to_ontology.get(entity_type, entity_type)
        cls = self.classes.get(onto_type)
        if not cls:
            return []
        return cls.properties_as_domain + cls.properties_as_range

    def get_subclass_names(self, class_name: str) -> list[str]:
        """Return all subclass names (recursive) for a given class."""
        cls = self.classes.get(class_name)
        if not cls:
            return []
        result = []
        for child in cls.children:
            result.append(child)
            result.extend(self.get_subclass_names(child))
        return result

    def get_related_types(self, class_name: str) -> list[str]:
        """Return types connected to this class via any property."""
        cls = self.classes.get(class_name)
        if not cls:
            return []
        related = set()
        for prop_name in cls.properties_as_domain:
            prop = self.properties.get(prop_name)
            if prop and prop.range in self.classes:
                related.add(prop.range)
        for prop_name in cls.properties_as_range:
            prop = self.properties.get(prop_name)
            if prop and prop.domain in self.classes:
                related.add(prop.domain)
        return list(related)

    @staticmethod
    def _local_name(uri: str) -> str:
        """Extract local name from a URI (after # or last /)."""
        if not uri:
            return ""
        if "#" in uri:
            return uri.split("#")[-1]
        return uri.split("/")[-1]
