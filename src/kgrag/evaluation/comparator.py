"""Strategy comparator (C3.5.3) — compare retrieval strategies on benchmark results."""

from __future__ import annotations

from collections import defaultdict

import structlog

from kgrag.core.models import QAEvalResult, StrategyComparison

logger = structlog.get_logger(__name__)


class StrategyComparator:
    """Compare retrieval strategies by aggregating per-question eval results."""

    def compare(
        self,
        results: list[QAEvalResult],
        *,
        group_by: str = "retrieval_strategy",
    ) -> list[StrategyComparison]:
        """Group results by strategy and compute aggregated metrics."""
        grouped: dict[str, list[QAEvalResult]] = defaultdict(list)
        for r in results:
            key = getattr(r, group_by, "unknown")
            grouped[key].append(r)

        comparisons: list[StrategyComparison] = []
        for strategy_name, strategy_results in grouped.items():
            n = len(strategy_results)
            if n == 0:
                continue

            avg_f1 = sum(r.f1_score for r in strategy_results) / n
            avg_faith = sum(r.faithfulness for r in strategy_results) / n
            avg_rel = sum(r.relevance for r in strategy_results) / n
            avg_lat = sum(r.latency_ms for r in strategy_results) / n
            em_rate = sum(1 for r in strategy_results if r.exact_match) / n

            # Per question-type breakdown
            type_groups: dict[str, list[float]] = defaultdict(list)
            for r in strategy_results:
                # question_type would need to come from the benchmark item
                type_groups["all"].append(r.f1_score)

            per_type_f1 = {
                qtype: sum(scores) / len(scores) for qtype, scores in type_groups.items()
            }

            comparisons.append(
                StrategyComparison(
                    strategy_name=strategy_name,
                    avg_f1=avg_f1,
                    avg_faithfulness=avg_faith,
                    avg_relevance=avg_rel,
                    avg_latency_ms=avg_lat,
                    exact_match_rate=em_rate,
                    num_questions=n,
                    per_type_f1=per_type_f1,
                )
            )

        # Sort by avg_f1 descending
        comparisons.sort(key=lambda c: c.avg_f1, reverse=True)

        logger.info(
            "comparator.done",
            strategies=len(comparisons),
            best=comparisons[0].strategy_name if comparisons else "none",
        )
        return comparisons
