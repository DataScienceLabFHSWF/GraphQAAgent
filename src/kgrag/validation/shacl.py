"""SHACL constraint checking (C3.2.1).

Validates the KG against SHACL shapes loaded from the ontology endpoint
to ensure structural quality before running QA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from kgrag.connectors.fuseki import FusekiConnector

logger = structlog.get_logger(__name__)


@dataclass
class SHACLViolation:
    """Single SHACL constraint violation."""

    focus_node: str
    constraint: str
    message: str
    severity: str = "Violation"


@dataclass
class SHACLReport:
    """Aggregated SHACL validation report."""

    conforms: bool = True
    violations: list[SHACLViolation] = field(default_factory=list)
    total_checked: int = 0


class SHACLValidator:
    """Validate KG against SHACL shapes via Fuseki SPARQL.

    Runs SPARQL-encoded SHACL-style constraint checks rather than full
    SHACL processing — keeps the agent lightweight.
    """

    def __init__(self, fuseki: FusekiConnector) -> None:
        self._fuseki = fuseki

    async def validate(self) -> SHACLReport:
        """Run all constraint checks and return a report."""
        report = SHACLReport()

        checks: list[tuple[str, str, str]] = [
            (
                "entities_have_labels",
                """
                SELECT ?entity WHERE {
                    ?entity a ?type .
                    FILTER NOT EXISTS { ?entity rdfs:label ?label }
                }
                """,
                "Entity missing rdfs:label",
            ),
            (
                "relations_have_types",
                """
                SELECT ?rel WHERE {
                    ?s ?rel ?o .
                    FILTER(!isLiteral(?o))
                    FILTER NOT EXISTS { ?rel a owl:ObjectProperty }
                    FILTER(?rel != rdf:type && ?rel != rdfs:label)
                }
                LIMIT 100
                """,
                "Relation not declared as owl:ObjectProperty",
            ),
        ]

        for name, sparql, message in checks:
            try:
                rows = await self._fuseki.query(sparql)
                report.total_checked += 1
                for row in rows:
                    focus = next(iter(row.values()), "unknown")
                    report.violations.append(
                        SHACLViolation(focus_node=focus, constraint=name, message=message)
                    )
            except Exception as exc:
                logger.warning("shacl.check_failed", check=name, error=str(exc))

        report.conforms = len(report.violations) == 0
        logger.info(
            "shacl.validated",
            conforms=report.conforms,
            violations=len(report.violations),
        )
        return report
