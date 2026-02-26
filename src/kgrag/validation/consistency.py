"""Logical consistency checks (C3.2.3).

Lightweight checks for contradictions and anomalies in the KG.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from kgrag.connectors.neo4j import Neo4jConnector

logger = structlog.get_logger(__name__)


@dataclass
class ConsistencyIssue:
    """A detected consistency problem."""

    check_name: str
    entity_id: str
    message: str
    severity: str = "warning"


@dataclass
class ConsistencyReport:
    """Aggregated consistency report."""

    issues: list[ConsistencyIssue] = field(default_factory=list)
    checks_run: int = 0

    @property
    def is_consistent(self) -> bool:
        return len(self.issues) == 0


class ConsistencyChecker:
    """Run logical consistency checks against the Neo4j KG."""

    def __init__(self, neo4j: Neo4jConnector) -> None:
        self._neo4j = neo4j

    async def check_all(self) -> ConsistencyReport:
        """Run all consistency checks and return a report.

        Checks:
        1. Orphan entities  — nodes with zero relationships.
        2. Self-referencing  — relationships where source == target.
        3. Duplicate entities — same ``label`` and Neo4j labels.
        """
        report = ConsistencyReport()

        await self._check_orphan_entities(report)
        await self._check_self_references(report)
        await self._check_duplicate_entities(report)

        logger.info(
            "consistency.checked",
            checks_run=report.checks_run,
            issues=len(report.issues),
        )
        return report

    # -- individual checks --------------------------------------------------

    async def _check_orphan_entities(self, report: ConsistencyReport) -> None:
        """Flag nodes with no incoming or outgoing relationships."""
        report.checks_run += 1
        query = "MATCH (n) WHERE NOT (n)--() RETURN elementId(n) AS eid, n.label AS label LIMIT 200"
        try:
            async with self._neo4j.driver.session(
                database=self._neo4j._config.database,
            ) as session:
                result = await session.run(query)
                records = await result.data()
            for rec in records:
                report.issues.append(
                    ConsistencyIssue(
                        check_name="orphan_entity",
                        entity_id=str(rec.get("eid", "")),
                        message=f"Node '{rec.get('label', '?')}' has no relationships",
                        severity="warning",
                    )
                )
        except Exception as exc:
            logger.warning("consistency.orphan_check_failed", error=str(exc))

    async def _check_self_references(self, report: ConsistencyReport) -> None:
        """Flag relationships that point from a node back to itself."""
        report.checks_run += 1
        query = (
            "MATCH (n)-[r]->(n) "
            "RETURN elementId(n) AS eid, n.label AS label, type(r) AS rel_type LIMIT 200"
        )
        try:
            async with self._neo4j.driver.session(
                database=self._neo4j._config.database,
            ) as session:
                result = await session.run(query)
                records = await result.data()
            for rec in records:
                report.issues.append(
                    ConsistencyIssue(
                        check_name="self_reference",
                        entity_id=str(rec.get("eid", "")),
                        message=(
                            f"Node '{rec.get('label', '?')}' has a self-referencing "
                            f"relationship [{rec.get('rel_type', '?')}]"
                        ),
                        severity="warning",
                    )
                )
        except Exception as exc:
            logger.warning("consistency.self_ref_check_failed", error=str(exc))

    async def _check_duplicate_entities(self, report: ConsistencyReport) -> None:
        """Flag pairs of nodes sharing the same ``label`` and Neo4j labels."""
        report.checks_run += 1
        query = (
            "MATCH (a), (b) "
            "WHERE a.label IS NOT NULL AND a.label = b.label "
            "  AND labels(a) = labels(b) "
            "  AND elementId(a) < elementId(b) "
            "RETURN a.label AS label, elementId(a) AS eid_a, elementId(b) AS eid_b "
            "LIMIT 200"
        )
        try:
            async with self._neo4j.driver.session(
                database=self._neo4j._config.database,
            ) as session:
                result = await session.run(query)
                records = await result.data()
            for rec in records:
                report.issues.append(
                    ConsistencyIssue(
                        check_name="duplicate_entity",
                        entity_id=str(rec.get("eid_a", "")),
                        message=(
                            f"Potential duplicate: nodes {rec.get('eid_a')} and "
                            f"{rec.get('eid_b')} share label '{rec.get('label')}'"
                        ),
                        severity="info",
                    )
                )
        except Exception as exc:
            logger.warning("consistency.duplicate_check_failed", error=str(exc))
