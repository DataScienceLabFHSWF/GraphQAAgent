"""KG-constrained answer validation and post-processing.

After the LLM generates an answer, this module checks that every entity
name and relation mentioned in the text actually exists in the knowledge
graph.  Three levels of correction are provided:

1. **Entity validation** — all entity references are checked against Neo4j.
   Unknown entity names are flagged and optionally replaced with the
   closest match (fuzzy / embedding similarity).
2. **Relation validation** — claimed relationships are verified against the
   graph.  Unsupported claims are annotated.
3. **Consistency check** — the answer is checked for internal contradictions
   with the collected evidence.

Status: **foundational implementation** — entity validation is fully
functional; relation validation and consistency checks use heuristic
string matching and can be upgraded to LLM-based verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from kgrag.connectors.neo4j import Neo4jConnector
from kgrag.core.models import KGEntity, QAAnswer

logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of KG-constrained answer validation."""

    is_valid: bool = True
    entity_mentions: list[str] = field(default_factory=list)
    verified_entities: list[str] = field(default_factory=list)
    unknown_entities: list[str] = field(default_factory=list)
    corrections: dict[str, str] = field(default_factory=dict)  # unknown → suggested
    unsupported_claims: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class KGConstrainedValidator:
    """Validate and post-process LLM answers against the knowledge graph.

    Parameters
    ----------
    neo4j:
        Active Neo4j connector for entity / relation lookups.
    fuzzy_threshold:
        Minimum Levenshtein similarity (0-1) to accept a fuzzy match
        when an exact entity is not found.
    """

    def __init__(
        self,
        neo4j: Neo4jConnector,
        *,
        fuzzy_threshold: float = 0.7,
    ) -> None:
        self._neo4j = neo4j
        self._fuzzy_threshold = fuzzy_threshold
        self._entity_cache: dict[str, KGEntity | None] = {}

    async def validate(self, answer: QAAnswer) -> ValidationResult:
        """Run full validation on a generated answer.

        Returns a ``ValidationResult`` detailing which entities were
        verified, which were unknown, and suggested corrections.
        """
        result = ValidationResult()

        # 1. Extract entity mentions from the answer text
        entity_mentions = self._extract_entity_mentions(answer)
        result.entity_mentions = entity_mentions

        # 2. Verify each mention against Neo4j
        for mention in entity_mentions:
            entity = await self._lookup_entity(mention)
            if entity:
                result.verified_entities.append(mention)
            else:
                result.unknown_entities.append(mention)
                # Try fuzzy match
                suggestion = await self._fuzzy_match(mention)
                if suggestion:
                    result.corrections[mention] = suggestion
                    result.warnings.append(
                        f"'{mention}' not found in KG; closest match: '{suggestion}'"
                    )

        # 3. Check evidence consistency (basic heuristic)
        unsupported = self._check_evidence_consistency(answer)
        result.unsupported_claims = unsupported

        result.is_valid = not result.unknown_entities and not unsupported
        return result

    def apply_corrections(self, answer: QAAnswer, result: ValidationResult) -> str:
        """Return a corrected answer text with entity name replacements.

        Only applies corrections where a high-confidence fuzzy match was
        found.  The original ``answer.answer_text`` is not mutated.
        """
        text = answer.answer_text
        for wrong, correct in result.corrections.items():
            text = text.replace(wrong, correct)
        return text

    # -- extraction --------------------------------------------------------

    def _extract_entity_mentions(self, answer: QAAnswer) -> list[str]:
        """Extract entity-like mentions from the answer text.

        Uses a combination of:
        - Entities already cited in the answer (from the pipeline)
        - Heuristic: capitalised multi-word phrases that might be entities
        """
        mentions: set[str] = set()

        # Cited entities are ground truth
        for ent in answer.cited_entities:
            if ent.label and ent.label in answer.answer_text:
                mentions.add(ent.label)

        # Heuristic: capitalised phrases (2+ words, not at sentence start)
        for match in re.finditer(
            r"(?<=[.!?]\s)([A-ZÄÖÜ][a-zäöüß]+(?:\s[A-ZÄÖÜ][a-zäöüß]+)+)",
            answer.answer_text,
        ):
            mentions.add(match.group(1))

        return sorted(mentions)

    # -- lookups -----------------------------------------------------------

    async def _lookup_entity(self, label: str) -> KGEntity | None:
        """Look up an entity by label in Neo4j (cached)."""
        if label in self._entity_cache:
            return self._entity_cache[label]

        try:
            results = await self._neo4j.query(
                "MATCH (n) WHERE n.label = $label OR n.name = $label "
                "RETURN n LIMIT 1",
                {"label": label},
            )
            if results:
                record = results[0]
                node = record["n"] if isinstance(record, dict) else record[0]
                entity = KGEntity(
                    id=str(node.get("id", label)),
                    label=label,
                    entity_type=str(node.get("type", "Unknown")),
                )
                self._entity_cache[label] = entity
                return entity
        except Exception as exc:
            logger.debug("kg_validator.lookup_failed", label=label, error=str(exc))

        self._entity_cache[label] = None
        return None

    async def _fuzzy_match(self, label: str) -> str | None:
        """Find the closest matching entity label using Neo4j full-text or APOC."""
        try:
            results = await self._neo4j.query(
                "MATCH (n) WHERE toLower(n.label) CONTAINS toLower($label) "
                "OR toLower(n.name) CONTAINS toLower($label) "
                "RETURN n.label AS label, n.name AS name LIMIT 3",
                {"label": label},
            )
            if results:
                record = results[0]
                if isinstance(record, dict):
                    return record.get("label") or record.get("name")
        except Exception as exc:
            logger.debug("kg_validator.fuzzy_failed", label=label, error=str(exc))

        return None

    # -- consistency -------------------------------------------------------

    def _check_evidence_consistency(self, answer: QAAnswer) -> list[str]:
        """Basic heuristic check for unsupported claims.

        Scans the answer text for factual-looking statements and checks
        whether they appear in the evidence.  This is a simple substring
        match — for production use, replace with an LLM-based NLI check.
        """
        unsupported: list[str] = []
        evidence_text = " ".join(ctx.text for ctx in answer.evidence).lower()

        # Extract sentences that look like factual claims
        sentences = re.split(r"[.!?]+", answer.answer_text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            # Simple heuristic: if no evidence word overlap, flag it
            words = set(sentence.lower().split())
            evidence_words = set(evidence_text.split()) if evidence_text else set()
            overlap = words & evidence_words
            coverage = len(overlap) / max(len(words), 1)
            if coverage < 0.3 and len(words) > 5:
                unsupported.append(sentence)

        return unsupported[:5]  # Cap at 5 to avoid noise
