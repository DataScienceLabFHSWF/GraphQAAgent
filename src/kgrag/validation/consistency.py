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

        Checks include:
        - Orphan entities (no relations)
        - Self-referencing relations
        - Duplicate entities (same label + type)
        """
        report = ConsistencyReport()

        # TODO: Implement concrete Cypher queries for each check
        # These will be added as the KG schema stabilises.

        logger.info(
            "consistency.checked",
            checks_run=report.checks_run,
            issues=len(report.issues),
        )
        return report
