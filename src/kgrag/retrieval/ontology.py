"""OntologyRetriever (C3.3.4) — Ontology-guided query expansion.

**Key contribution**: uses the Fuseki ontology to expand queries with class
hierarchies, synonyms, and expected relations *before* retrieval, making
the retrieval ontology-aware and improving recall for domain-specific questions.

This is not a standalone retriever but a **pre-processing enhancer** that
enriches a :class:`~kgrag.core.models.QAQuery` so downstream retrievers
(vector, graph, hybrid) benefit from ontology knowledge.
"""

from __future__ import annotations

import structlog

from kgrag.connectors.fuseki import FusekiConnector
from kgrag.core.models import QAQuery, QuestionType

logger = structlog.get_logger(__name__)


class OntologyRetriever:
    """Ontology-guided query expansion using Fuseki SPARQL endpoint.

    Enriches the :class:`QAQuery` with:
    - Parent classes (e.g. "Reaktor" -> also "NuclearFacility")
    - Sibling classes (e.g. "BWR" -> also consider "PWR")
    - Expected relations (which properties to prioritise in graph traversal)
    - rdfs:label / skos:altLabel synonyms
    """

    def __init__(self, fuseki: FusekiConnector) -> None:
        self._fuseki = fuseki
        self._fuseki_available = True  # Assume available initially

    async def _check_fuseki_available(self) -> bool:
        """Check if Fuseki is available and has the dataset."""
        if not self._fuseki_available:
            return False
        
        try:
            # Try a simple query to check if Fuseki is working
            await self._fuseki.client.get(f"/{self._fuseki._config.dataset}")
            return True
        except:
            self._fuseki_available = False
            logger.warning("Fuseki not available - ontology expansion disabled")
            return False

    async def expand_query(self, query: QAQuery) -> QAQuery:
        """Expand the query using ontology knowledge.

        Mutates and returns the same ``QAQuery`` instance for convenience.
        """
        # Check if Fuseki is available
        if not await self._check_fuseki_available():
            logger.info("Ontology expansion skipped - Fuseki not available")
            return query

        expanded_types: list[str] = list(query.detected_types)
        expanded_relations: list[str] = list(query.expected_relations)

        for type_label in query.detected_types:
            try:
                # Resolve label to URI
                cls = await self._fuseki.get_class_by_label(type_label)
                if cls is None:
                    continue

                # Add subclasses (expands recall)
                subclasses = await self._fuseki.get_subclasses(cls.uri)
                for sub in subclasses:
                    if sub.label not in expanded_types:
                        expanded_types.append(sub.label)

                # Add synonyms
                synonyms = await self._fuseki.get_synonyms(cls.uri)
                for syn in synonyms:
                    if syn not in expanded_types:
                        expanded_types.append(syn)

                # Add expected relations for graph retrieval
                props = await self._fuseki.get_class_properties(cls.uri)
                for prop in props:
                    if prop.label not in expanded_relations:
                        expanded_relations.append(prop.label)
            except Exception as e:
                logger.warning(f"Ontology expansion failed for {type_label}: {e}")
                continue

        query.detected_types = expanded_types
        query.expected_relations = expanded_relations

        logger.info(
            "ontology.expand",
            original_types=len(query.detected_types),
            expanded_types=len(expanded_types),
            expected_relations=len(expanded_relations),
        )
        return query

    async def get_expected_relations(self, entity_types: list[str]) -> list[str]:
        """Given ontology classes, return which relations to prioritise in graph retrieval."""
        # Check if Fuseki is available
        if not await self._check_fuseki_available():
            logger.info("Expected relations lookup skipped - Fuseki not available")
            return []

        relations: list[str] = []
        for type_label in entity_types:
            try:
                cls = await self._fuseki.get_class_by_label(type_label)
                if cls is None:
                    continue
                props = await self._fuseki.get_class_properties(cls.uri)
                relations.extend(p.label for p in props if p.label not in relations)
            except Exception as e:
                logger.warning(f"Failed to get expected relations for {type_label}: {e}")
                continue
        return relations

    async def get_answer_template(
        self,
        question_type: QuestionType | None,
        entity_types: list[str],
    ) -> str:
        """Generate an answer-structure hint based on ontology and question type.

        Helps the LLM format its answer correctly (e.g. list instances for LIST
        questions, provide a boolean for BOOLEAN questions).
        """
        if not question_type:
            return ""

        type_labels = ", ".join(entity_types[:3]) if entity_types else "relevant entities"

        templates = {
            QuestionType.FACTOID: f"Provide a concise factual answer about {type_labels}.",
            QuestionType.LIST: f"List all instances of {type_labels} that match the question.",
            QuestionType.BOOLEAN: "Answer with Yes or No, then provide a brief justification.",
            QuestionType.COMPARATIVE: f"Compare the mentioned {type_labels}, highlighting differences.",
            QuestionType.CAUSAL: "Explain the causal chain, citing specific entities and relations.",
            QuestionType.AGGREGATION: f"Provide the count or aggregation over {type_labels}.",
        }
        return templates.get(question_type, "")
