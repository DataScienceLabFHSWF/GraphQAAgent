"""CQ-based KG completeness validator (C3.2.2).

Checks whether the knowledge graph can answer each competency question,
both via the QA pipeline and (optionally) via direct SPARQL queries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from kgrag.connectors.fuseki import FusekiConnector
from kgrag.core.models import CompetencyQuestion

logger = structlog.get_logger(__name__)


@dataclass
class CQResult:
    """Result of validating a single competency question."""

    question_id: str
    text: str
    answerable: bool = False
    sparql_ok: bool | None = None
    confidence: float = 0.0
    error: str | None = None


@dataclass
class CQReport:
    """Aggregated CQ coverage report."""

    total: int = 0
    answerable: int = 0
    sparql_ok: int = 0
    results: list[CQResult] = field(default_factory=list)

    @property
    def coverage_rate(self) -> float:
        return self.answerable / self.total if self.total > 0 else 0.0


class CQValidator:
    """Validate KG completeness against competency questions."""

    def __init__(self, fuseki: FusekiConnector) -> None:
        self._fuseki = fuseki

    @staticmethod
    def load_cqs(path: str | Path) -> list[CompetencyQuestion]:
        """Load competency questions from a JSON file (KGB format)."""
        with open(path) as f:
            data = json.load(f)
        # Support both top-level list and {questions: [...]} wrapper
        items = data.get("questions", data) if isinstance(data, dict) else data
        return [
            CompetencyQuestion(
                id=item["id"],
                question=item["question"],
                expected_answers=item.get("expected_answers", []),
                query_type=item.get("query_type", "entity"),
                difficulty=item.get("difficulty", 1),
                tags=item.get("tags", []),
                metadata=item.get("metadata", {}),
            )
            for item in items
        ]

    async def validate_sparql(self, cqs: list[CompetencyQuestion]) -> CQReport:
        """Check which CQs can be answered via SPARQL (if metadata has template)."""
        report = CQReport(total=len(cqs))

        for cq in cqs:
            result = CQResult(question_id=cq.id, text=cq.question)
            sparql_template = cq.metadata.get("sparql_template", "")
            if sparql_template:
                try:
                    rows = await self._fuseki.query(sparql_template)
                    result.sparql_ok = len(rows) > 0
                    if result.sparql_ok:
                        report.sparql_ok += 1
                        result.answerable = True
                        report.answerable += 1
                except Exception as exc:
                    result.error = str(exc)
                    logger.warning("cq.sparql_failed", cq_id=cq.id, error=str(exc))

            report.results.append(result)

        logger.info(
            "cq.validated",
            total=report.total,
            answerable=report.answerable,
            coverage=f"{report.coverage_rate:.0%}",
        )
        return report
