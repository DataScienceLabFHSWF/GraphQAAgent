"""GraphRAG-powered ontology gap analyzer — live Neo4j + Fuseki for gap detection.

Adapted from OntologyExtender's ``GraphRAGGapAnalyzer`` for use within KG-RAG QA.
Instead of analyzing offline checkpoints, this module:

1. Queries Neo4j for all ABox entities (the knowledge graph instances)
2. Queries Fuseki for all TBox classes (the seed ontology schema)
3. Compares ABox entity types against TBox class labels using fuzzy + embedding matching
4. Uncovered entities become gap candidates
5. Uses neighbourhood density from the graph to rank gap importance
6. Integrates with the HITL pipeline to escalate gaps as ChangeProposals

This complements the QA-driven ``GapDetector`` (which finds gaps from low-confidence
answers) with a structural approach (which finds gaps from graph schema mismatches).

The workflow:
    GapAnalyzer.analyze()  →  GapReport  →  escalate_to_hitl()  →  ChangeProposals
                                                                        ↓
                                                           Domain experts review
                                                                        ↓
                                                           OntologyExtender extends TBox

References:
    - OntologyExtender: DataScienceLabFHSWF/OntologyExtender (discovery/gap_analyzer.py)
    - OntologyExtender: GraphRAGGapAnalyzer (retrieval/graphrag_gap_analyzer.py)
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from kgrag.connectors.fuseki import FusekiConnector
from kgrag.connectors.neo4j import Neo4jConnector
from kgrag.connectors.langchain_ollama_provider import LangChainOllamaProvider
from kgrag.core.models import KGEntity, OntologyClass
from kgrag.hitl.change_proposals import (
    ChangeProposal,
    ChangeProposalService,
    ProposalType,
)
from kgrag.hitl.gap_detection import DetectedGap, GapDetector

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class GapCandidate:
    """An entity type in the ABox not covered by any TBox class."""

    entity_type: str
    representative_label: str
    examples: list[str] = field(default_factory=list)
    frequency: int = 0
    avg_confidence: float = 0.0
    closest_seed_class: str | None = None
    semantic_distance: float = 1.0        # 0 = identical, 1 = unrelated
    structural_score: float = 0.0         # avg neighbourhood size (graph importance)


@dataclass
class GapReport:
    """Output of a graph-based ontology gap analysis."""

    total_abox_entities: int = 0
    covered_entities: int = 0
    uncovered_entities: int = 0
    coverage_pct: float = 0.0
    gap_candidates: list[GapCandidate] = field(default_factory=list)
    qa_driven_gaps: list[DetectedGap] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_gaps(self) -> int:
        return len(self.gap_candidates) + len(self.qa_driven_gaps)


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------


class OntologyGapAnalyzer:
    """Live GraphRAG-powered ontology gap analyzer.

    Combines two gap detection strategies:

    1. **Structural gap detection** — Compare Neo4j ABox entity types against
       Fuseki TBox classes to find entity types the ontology cannot represent.
    2. **QA-driven gap detection** — Low confidence / unanswerable questions
       from the QA pipeline signal knowledge gaps (via ``GapDetector``).

    Both streams feed into the HITL pipeline as ``ChangeProposal`` items
    that domain experts can review. Accepted gaps can be escalated to the
    OntologyExtender for automated TBox extension.

    Parameters
    ----------
    neo4j : Neo4jConnector
        Connected async Neo4j client.
    fuseki : FusekiConnector
        Connected async Fuseki client.
    ollama : LangChainOllamaProvider
        For embedding-based semantic matching.
    min_frequency : int
        Minimum entity mentions to qualify as a gap candidate.
    similarity_threshold : float
        Cosine similarity above which an entity type is considered "covered".
    """

    def __init__(
        self,
        neo4j: Neo4jConnector,
        fuseki: FusekiConnector,
        ollama: LangChainOllamaProvider,
        *,
        min_frequency: int = 2,
        similarity_threshold: float = 0.65,
    ) -> None:
        self._neo4j = neo4j
        self._fuseki = fuseki
        self._ollama = ollama
        self._min_frequency = min_frequency
        self._similarity_threshold = similarity_threshold

        # QA-driven gap detector (companion)
        self._qa_gap_detector = GapDetector(confidence_threshold=0.5)

        # Embedding caches
        self._embedding_cache: dict[str, list[float]] = {}

    @property
    def qa_gap_detector(self) -> GapDetector:
        """Access the QA-driven gap detector for registering low-confidence answers."""
        return self._qa_gap_detector

    # -- Public API ---------------------------------------------------------

    async def analyze(self) -> GapReport:
        """Run full gap analysis: structural (ABox vs TBox) + QA-driven.

        Returns a :class:`GapReport` with both structural and QA-driven gaps.
        """
        logger.info("gap_analysis.start")

        # 1. Fetch all ABox entities from Neo4j
        abox_entities = await self._fetch_all_entities()
        logger.info("gap_analysis.abox_fetched", count=len(abox_entities))

        # 2. Fetch all TBox classes from Fuseki
        tbox_classes = await self._fetch_tbox_classes()
        tbox_labels = [c.label for c in tbox_classes]
        logger.info("gap_analysis.tbox_fetched", count=len(tbox_classes))

        # 3. Classify entities as covered or uncovered
        covered, uncovered = await self._classify_entities(abox_entities, tbox_labels)
        logger.info(
            "gap_analysis.classified",
            covered=len(covered), uncovered=len(uncovered),
        )

        # 4. Build gap candidates from uncovered entities
        gap_candidates = await self._build_gap_candidates(uncovered, tbox_labels)

        # 5. Collect QA-driven gaps
        qa_gaps = self._qa_gap_detector.get_gaps()

        total = len(abox_entities)
        coverage_pct = len(covered) / total if total > 0 else 0.0

        report = GapReport(
            total_abox_entities=total,
            covered_entities=len(covered),
            uncovered_entities=len(uncovered),
            coverage_pct=coverage_pct,
            gap_candidates=gap_candidates,
            qa_driven_gaps=qa_gaps,
        )

        logger.info(
            "gap_analysis.complete",
            total=total,
            covered=len(covered),
            structural_gaps=len(gap_candidates),
            qa_gaps=len(qa_gaps),
            coverage_pct=f"{coverage_pct:.1%}",
        )
        return report

    # -- Neo4j: fetch all entities ------------------------------------------

    async def _fetch_all_entities(self) -> list[KGEntity]:
        """Query all ABox entities from Neo4j."""
        query = f"""
        MATCH (e{self._neo4j._lbl})
        RETURN e, labels(e) AS _labels
        ORDER BY e.id
        LIMIT 5000
        """
        async with self._neo4j.driver.session(
            database=self._neo4j._config.database,
        ) as session:
            result = await session.run(query)
            records = await result.data()

        return [
            self._neo4j._record_to_entity(r["e"], neo4j_labels=r.get("_labels"))
            for r in records
        ]

    # -- Fuseki: fetch TBox classes -----------------------------------------

    async def _fetch_tbox_classes(self) -> list[OntologyClass]:
        """Query all OWL classes from Fuseki."""
        sparql = """
        SELECT DISTINCT ?class ?label
        WHERE {
            ?class a owl:Class .
            OPTIONAL { ?class rdfs:label ?label . }
        }
        ORDER BY ?class
        """
        rows = await self._fuseki.query(sparql)

        classes: list[OntologyClass] = []
        for row in rows:
            uri = row.get("class", {}).get("value", "")
            label = row.get("label", {}).get("value", "")
            if not label:
                label = uri.split("#")[-1].split("/")[-1]
            if label:
                classes.append(OntologyClass(uri=uri, label=label))
        return classes

    # -- Embedding helpers --------------------------------------------------

    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding via Ollama, using cache."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        try:
            embedding = await self._ollama.embed(text)
            self._embedding_cache[text] = embedding
            return embedding
        except Exception as exc:
            logger.debug("gap_analysis.embedding_failed", text=text[:50], error=str(exc))
            return []

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # -- Entity classification -----------------------------------------------

    async def _classify_entities(
        self,
        entities: list[KGEntity],
        tbox_labels: list[str],
    ) -> tuple[list[KGEntity], list[KGEntity]]:
        """Split entities into covered (match TBox) and uncovered.

        Uses a 3-tier matching strategy:
        1. Exact match on entity_type vs TBox label (case-insensitive)
        2. Substring/fuzzy match
        3. Embedding cosine similarity
        """
        covered: list[KGEntity] = []
        uncovered: list[KGEntity] = []

        tbox_set = {lbl.lower() for lbl in tbox_labels}

        # Pre-compute TBox embeddings
        tbox_embeddings: dict[str, list[float]] = {}
        for lbl in tbox_labels:
            tbox_embeddings[lbl] = await self._get_embedding(lbl)

        for entity in entities:
            etype = (entity.entity_type or "").lower()
            elabel = (entity.label or "").lower()

            # Tier 1: Exact match
            if etype in tbox_set or elabel in tbox_set:
                covered.append(entity)
                continue

            # Tier 2: Substring match (entity type contained in a TBox label or vice versa)
            substring_match = False
            for tbox_lbl in tbox_labels:
                tbox_lower = tbox_lbl.lower()
                if etype and (etype in tbox_lower or tbox_lower in etype):
                    substring_match = True
                    break
            if substring_match:
                covered.append(entity)
                continue

            # Tier 3: Semantic (embedding) match
            if etype:
                entity_emb = await self._get_embedding(entity.entity_type)
                if entity_emb:
                    best_sim = max(
                        (self._cosine_similarity(entity_emb, emb) for emb in tbox_embeddings.values() if emb),
                        default=0.0,
                    )
                    if best_sim >= self._similarity_threshold:
                        covered.append(entity)
                        continue

            uncovered.append(entity)

        return covered, uncovered

    # -- Gap candidate building ---------------------------------------------

    async def _build_gap_candidates(
        self,
        uncovered: list[KGEntity],
        tbox_labels: list[str],
    ) -> list[GapCandidate]:
        """Group uncovered entities by type and rank by frequency + structural importance."""
        type_groups: dict[str, list[KGEntity]] = defaultdict(list)
        for entity in uncovered:
            key = entity.entity_type or entity.label
            type_groups[key].append(entity)

        # Pre-compute TBox embeddings
        tbox_embeddings: dict[str, list[float]] = {}
        for lbl in tbox_labels:
            tbox_embeddings[lbl] = await self._get_embedding(lbl)

        candidates: list[GapCandidate] = []
        for entity_type, group in type_groups.items():
            freq = len(group)
            if freq < self._min_frequency:
                continue

            # Find closest seed class
            closest_class = None
            semantic_distance = 1.0
            entity_emb = await self._get_embedding(entity_type)
            if entity_emb:
                best_sim, best_class = -1.0, None
                for class_name, class_emb in tbox_embeddings.items():
                    if class_emb:
                        sim = self._cosine_similarity(entity_emb, class_emb)
                        if sim > best_sim:
                            best_sim, best_class = sim, class_name
                if best_class:
                    closest_class = best_class
                    semantic_distance = 1.0 - best_sim

            # Structural importance: average neighbourhood size
            structural_score = await self._compute_structural_score(group)

            avg_conf = sum(e.confidence for e in group) / freq if freq else 0.0

            candidates.append(GapCandidate(
                entity_type=entity_type,
                representative_label=group[0].label,
                examples=[e.label for e in group[:5]],
                frequency=freq,
                avg_confidence=avg_conf,
                closest_seed_class=closest_class,
                semantic_distance=semantic_distance,
                structural_score=structural_score,
            ))

        candidates.sort(key=lambda c: c.frequency * (1 + c.structural_score), reverse=True)
        return candidates

    async def _compute_structural_score(self, entities: list[KGEntity]) -> float:
        """Estimate structural importance via neighbourhood density."""
        if not entities:
            return 0.0

        total = 0
        sample = entities[:5]
        for entity in sample:
            try:
                neighbours = await self._neo4j.get_entity_neighbours(entity.id, limit=20)
                total += len(neighbours)
            except Exception:
                pass
        return total / len(sample) if sample else 0.0

    # -- HITL escalation ----------------------------------------------------

    def escalate_to_hitl(
        self,
        report: GapReport,
        proposal_service: ChangeProposalService,
        *,
        author: str = "gap_analyzer",
    ) -> list[ChangeProposal]:
        """Convert gap candidates into HITL change proposals for expert review.

        Creates one ``ChangeProposal`` per structural gap candidate and one
        per QA-driven gap. These are submitted to the ``ChangeProposalService``
        where domain experts can accept, reject, or revise them.

        Accepted proposals can then be forwarded to the OntologyExtender
        for automated TBox extension.

        Parameters
        ----------
        report : GapReport
            Output from :meth:`analyze`.
        proposal_service : ChangeProposalService
            The HITL proposal service to submit proposals to.
        author : str
            Author name for the proposals.

        Returns
        -------
        list[ChangeProposal]
            The created proposals.
        """
        proposals: list[ChangeProposal] = []

        # Structural gaps → ADD_ENTITY proposals (suggest new ontology class)
        for gap in report.gap_candidates:
            proposal = proposal_service.create_proposal(
                proposal_type=ProposalType.ADD_ENTITY,
                proposed_data={
                    "entity_type": gap.entity_type,
                    "suggested_class_label": gap.entity_type,
                    "closest_seed_class": gap.closest_seed_class,
                    "semantic_distance": round(gap.semantic_distance, 4),
                    "structural_score": round(gap.structural_score, 2),
                    "examples": gap.examples,
                    "frequency": gap.frequency,
                    "gap_source": "structural_analysis",
                },
                author=author,
                rationale=(
                    f"Structural gap: {gap.frequency} entities of type '{gap.entity_type}' "
                    f"found in the KG but no matching TBox class. "
                    f"Closest seed class: '{gap.closest_seed_class}' "
                    f"(distance: {gap.semantic_distance:.2f}). "
                    f"Structural importance: {gap.structural_score:.1f} avg neighbours."
                ),
            )
            proposals.append(proposal)
            logger.info(
                "gap_analysis.proposal_created",
                proposal_id=proposal.id,
                gap_type="structural",
                entity_type=gap.entity_type,
            )

        # QA-driven gaps → ADD_ENTITY or ADD_RELATION proposals
        for qa_gap in report.qa_driven_gaps:
            ptype = (
                ProposalType.ADD_RELATION
                if qa_gap.gap_type == "abox_missing_relation"
                else ProposalType.ADD_ENTITY
            )
            proposal = proposal_service.create_proposal(
                proposal_type=ptype,
                proposed_data={
                    "gap_type": qa_gap.gap_type,
                    "trigger_question": qa_gap.trigger_question,
                    "confidence": qa_gap.confidence,
                    "frequency": qa_gap.frequency,
                    "gap_source": "qa_interaction",
                    "metadata": qa_gap.metadata,
                },
                author=author,
                trigger_question=qa_gap.trigger_question,
                trigger_confidence=qa_gap.confidence,
                rationale=(
                    f"QA-driven gap ({qa_gap.gap_type}): Question "
                    f"'{qa_gap.trigger_question[:80]}' could not be answered "
                    f"satisfactorily (confidence: {qa_gap.confidence:.2f}). "
                    f"Suggested action: {qa_gap.suggested_action}."
                ),
            )
            proposals.append(proposal)
            logger.info(
                "gap_analysis.proposal_created",
                proposal_id=proposal.id,
                gap_type="qa_driven",
                question=qa_gap.trigger_question[:60],
            )

        logger.info(
            "gap_analysis.escalation_complete",
            structural_proposals=len(report.gap_candidates),
            qa_proposals=len(report.qa_driven_gaps),
            total_proposals=len(proposals),
        )
        return proposals

    def export_for_ontology_extender(self, report: GapReport) -> dict[str, Any]:
        """Export gap analysis results in a format the OntologyExtender can consume.

        The OntologyExtender's ``run_gap_analysis.py`` script expects a JSON
        structure compatible with its ``GapReport`` model. This method
        produces that format so gaps discovered by KG-RAG QA can be fed
        directly into the ontology extension pipeline.

        Returns
        -------
        dict
            JSON-serializable dict matching OntologyExtender's GapReport schema.
        """
        return {
            "ontology_version": "current",
            "total_extracted_entities": report.total_abox_entities,
            "covered_entities": report.covered_entities,
            "uncovered_entities": report.uncovered_entities,
            "coverage_pct": round(report.coverage_pct, 4),
            "gap_candidates": [
                {
                    "entity_type": gc.entity_type,
                    "representative_label": gc.representative_label,
                    "examples": gc.examples,
                    "frequency": gc.frequency,
                    "avg_confidence": round(gc.avg_confidence, 4),
                    "closest_seed_class": gc.closest_seed_class,
                    "semantic_distance": round(gc.semantic_distance, 4),
                }
                for gc in report.gap_candidates
            ],
            "qa_driven_gaps": [
                {
                    "gap_type": g.gap_type,
                    "trigger_question": g.trigger_question,
                    "confidence": g.confidence,
                    "frequency": g.frequency,
                    "suggested_action": g.suggested_action,
                }
                for g in report.qa_driven_gaps
            ],
            "timestamp": report.timestamp.isoformat(),
        }
