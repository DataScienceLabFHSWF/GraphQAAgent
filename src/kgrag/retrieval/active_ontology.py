"""Active ontology learning — automatic gap detection and proposal generation.

When the agentic retriever's ``lookup_ontology`` tool returns "not found" or
when evidence quality is low, this module:

1. **Detects gaps** — identifies missing classes, properties, or relations
   in the current TBox schema.
2. **Proposes additions** — generates candidate ontology additions (new
   classes, properties, or relation types) based on evidence from the
   knowledge graph and document corpus.
3. **Queues proposals** — feeds proposals into the HITL change-proposal
   pipeline for human review.

Status: **foundational implementation** — gap detection heuristics and
proposal generation are functional; integration with the HITL pipeline
exists but the actual ontology mutation (writing to Fuseki) requires
human approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from kgrag.connectors.fuseki import FusekiConnector
from kgrag.retrieval.ontology_context import OntologyContext

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class OntologyGap:
    """A detected gap in the ontology schema.

    Attributes
    ----------
    gap_type:
        One of ``missing_class``, ``missing_property``, ``missing_relation``,
        ``weak_hierarchy``.
    query_context:
        The user question or sub-question that triggered the gap.
    entity_labels:
        KG entity labels involved (if any).
    description:
        Human-readable description of what's missing.
    confidence:
        Estimated confidence that this is a genuine gap (0-1).
    """

    gap_type: str
    query_context: str = ""
    entity_labels: list[str] = field(default_factory=list)
    description: str = ""
    confidence: float = 0.0


@dataclass
class OntologyProposal:
    """A proposed ontology addition generated from a detected gap.

    Attributes
    ----------
    proposal_type:
        One of ``add_class``, ``add_property``, ``add_relation``,
        ``extend_hierarchy``.
    label:
        Label for the proposed class / property / relation.
    parent_class:
        Superclass URI (for ``add_class`` / ``extend_hierarchy``).
    domain_class:
        Domain class URI (for ``add_property`` / ``add_relation``).
    range_class:
        Range class URI (for ``add_relation``).
    description:
        Human-readable rationale.
    turtle_fragment:
        Ready-to-insert Turtle/TTL fragment (for review).
    source_gap:
        The :class:`OntologyGap` that triggered this proposal.
    """

    proposal_type: str
    label: str = ""
    parent_class: str | None = None
    domain_class: str | None = None
    range_class: str | None = None
    description: str = ""
    turtle_fragment: str = ""
    source_gap: OntologyGap | None = None


# ---------------------------------------------------------------------------
# Gap detector
# ---------------------------------------------------------------------------


class OntologyGapDetector:
    """Detect gaps in the ontology by comparing queries against the TBox.

    Parameters
    ----------
    ontology_context:
        The ``OntologyContext`` instance holding the parsed TBox schema.
    """

    def __init__(self, ontology_context: OntologyContext) -> None:
        self._ontology = ontology_context

    def detect_from_failed_lookup(
        self,
        lookup_query: str,
        lookup_result: str,
    ) -> OntologyGap | None:
        """Check if a ``lookup_ontology`` result indicates a gap.

        Parameters
        ----------
        lookup_query:
            The ontology term that was searched for.
        lookup_result:
            The raw result string from the lookup tool.

        Returns
        -------
        An ``OntologyGap`` if the lookup suggests missing schema, else ``None``.
        """
        negative_indicators = [
            "not found", "no matching", "unknown", "no results",
            "no classes", "no properties", "no ontology",
        ]
        result_lower = lookup_result.lower()
        if not any(ind in result_lower for ind in negative_indicators):
            return None

        # Determine gap type from query context
        query_lower = lookup_query.lower()
        if any(kw in query_lower for kw in ("class", "type", "category")):
            gap_type = "missing_class"
        elif any(kw in query_lower for kw in ("property", "attribute", "field")):
            gap_type = "missing_property"
        elif any(kw in query_lower for kw in ("relation", "connects", "link")):
            gap_type = "missing_relation"
        else:
            gap_type = "missing_class"  # default

        return OntologyGap(
            gap_type=gap_type,
            query_context=lookup_query,
            description=f"Ontology lookup for '{lookup_query}' returned no results",
            confidence=0.6,
        )

    def detect_from_low_evidence(
        self,
        question: str,
        entity_labels: list[str],
        evidence_count: int,
    ) -> OntologyGap | None:
        """Detect a potential gap when evidence retrieval yields poor results.

        If a question references entities that exist in the KG but
        retrieval finds very little evidence, the ontology may be missing
        the relationship type needed to connect them.
        """
        if evidence_count >= 2 or not entity_labels:
            return None

        return OntologyGap(
            gap_type="weak_hierarchy",
            query_context=question,
            entity_labels=entity_labels,
            description=(
                f"Question about {', '.join(entity_labels[:3])} yielded only "
                f"{evidence_count} evidence pieces — possible missing relation type"
            ),
            confidence=0.4,
        )


# ---------------------------------------------------------------------------
# Proposal generator
# ---------------------------------------------------------------------------


class OntologyProposalGenerator:
    """Generate ontology addition proposals from detected gaps.

    Parameters
    ----------
    ontology_context:
        The ``OntologyContext`` instance for existing schema lookup.
    fuseki:
        Optional FusekiConnector for SPARQL-based enrichment queries.
    """

    def __init__(
        self,
        ontology_context: OntologyContext,
        fuseki: FusekiConnector | None = None,
    ) -> None:
        self._ontology = ontology_context
        self._fuseki = fuseki

    def generate_proposal(self, gap: OntologyGap) -> OntologyProposal:
        """Generate an ontology proposal for the given gap.

        The proposal includes a ready-to-review Turtle fragment.
        """
        if gap.gap_type == "missing_class":
            return self._propose_class(gap)
        elif gap.gap_type == "missing_property":
            return self._propose_property(gap)
        elif gap.gap_type == "missing_relation":
            return self._propose_relation(gap)
        elif gap.gap_type == "weak_hierarchy":
            return self._propose_hierarchy_extension(gap)
        else:
            return OntologyProposal(
                proposal_type="add_class",
                label=gap.query_context,
                description=f"Unclassified gap: {gap.description}",
                source_gap=gap,
            )

    def _propose_class(self, gap: OntologyGap) -> OntologyProposal:
        """Generate an ``add_class`` proposal."""
        label = gap.query_context.strip().replace(" ", "")
        # Try to find a reasonable superclass from existing schema
        parent = self._find_parent_class(gap.query_context)

        ttl = f"""### Proposed by Active Ontology Learning
:{label} a owl:Class ;
    rdfs:label "{gap.query_context}" ;
    rdfs:comment "Auto-proposed: {gap.description}" ."""
        if parent:
            ttl = ttl.replace(
                "a owl:Class",
                f"a owl:Class ;\n    rdfs:subClassOf :{parent}",
            )

        return OntologyProposal(
            proposal_type="add_class",
            label=label,
            parent_class=parent,
            description=gap.description,
            turtle_fragment=ttl,
            source_gap=gap,
        )

    def _propose_property(self, gap: OntologyGap) -> OntologyProposal:
        """Generate an ``add_property`` proposal."""
        label = gap.query_context.strip().replace(" ", "")
        domain = gap.entity_labels[0] if gap.entity_labels else "Thing"

        ttl = f"""### Proposed by Active Ontology Learning
:{label} a owl:DatatypeProperty ;
    rdfs:label "{gap.query_context}" ;
    rdfs:domain :{domain} ;
    rdfs:range xsd:string ;
    rdfs:comment "Auto-proposed: {gap.description}" ."""

        return OntologyProposal(
            proposal_type="add_property",
            label=label,
            domain_class=domain,
            description=gap.description,
            turtle_fragment=ttl,
            source_gap=gap,
        )

    def _propose_relation(self, gap: OntologyGap) -> OntologyProposal:
        """Generate an ``add_relation`` proposal."""
        label = gap.query_context.strip().replace(" ", "")
        entities = gap.entity_labels or ["Thing", "Thing"]
        domain = entities[0] if len(entities) > 0 else "Thing"
        range_cls = entities[1] if len(entities) > 1 else "Thing"

        ttl = f"""### Proposed by Active Ontology Learning
:{label} a owl:ObjectProperty ;
    rdfs:label "{gap.query_context}" ;
    rdfs:domain :{domain} ;
    rdfs:range :{range_cls} ;
    rdfs:comment "Auto-proposed: {gap.description}" ."""

        return OntologyProposal(
            proposal_type="add_relation",
            label=label,
            domain_class=domain,
            range_class=range_cls,
            description=gap.description,
            turtle_fragment=ttl,
            source_gap=gap,
        )

    def _propose_hierarchy_extension(self, gap: OntologyGap) -> OntologyProposal:
        """Generate an ``extend_hierarchy`` proposal."""
        entities = gap.entity_labels[:2]
        label = f"{'_'.join(entities)}_relation" if entities else "new_relation"

        ttl = f"""### Proposed by Active Ontology Learning
# Hierarchy extension for: {', '.join(entities)}
# Gap: {gap.description}
# TODO: Determine the appropriate superclass and relation type"""

        return OntologyProposal(
            proposal_type="extend_hierarchy",
            label=label,
            description=gap.description,
            turtle_fragment=ttl,
            source_gap=gap,
        )

    def _find_parent_class(self, term: str) -> str | None:
        """Heuristic: find the best existing superclass for a term."""
        summary = self._ontology.schema_summary.lower()
        term_lower = term.lower()

        # Simple keyword matching against known classes
        candidates = []
        for line in summary.split("\n"):
            if "class" in line.lower():
                # Extract class name
                parts = line.strip().split()
                for part in parts:
                    clean = part.strip(":,;()")
                    if clean and len(clean) > 2:
                        if term_lower in clean.lower() or clean.lower() in term_lower:
                            candidates.append(clean)

        return candidates[0] if candidates else None
