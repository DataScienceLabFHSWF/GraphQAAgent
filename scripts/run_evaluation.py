#!/usr/bin/env python3
"""Full benchmark evaluation across retrieval strategies (C3.5).

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --benchmark data/qa_benchmarks/benchmark_v1.json
    python scripts/run_evaluation.py --strategies vector_only hybrid
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from kgrag.agents.orchestrator import Orchestrator
from kgrag.agents.question_parser import QuestionParser
from kgrag.core.config import Settings
from kgrag.core.models import QAEvalResult
from kgrag.evaluation.comparator import StrategyComparator
from kgrag.evaluation.metrics import (
    compute_context_relevance,
    compute_exact_match,
    compute_faithfulness,
    compute_token_f1,
)
from kgrag.evaluation.qa_dataset import QADataset
from kgrag.evaluation.reporter import EvaluationReporter


async def main() -> None:
    parser = argparse.ArgumentParser(description="KG-RAG Evaluation")
    parser.add_argument("--benchmark", default="data/qa_benchmarks/benchmark_v1.json")
    parser.add_argument("--strategies", nargs="+", default=["vector_only", "graph_only", "hybrid"])
    parser.add_argument("--output-dir", default="reports/evaluation/")
    parser.add_argument("--formats", nargs="+", default=["json", "markdown", "html"])
    args = parser.parse_args()

    settings = Settings()  # type: ignore[call-arg]
    orchestrator = Orchestrator(settings)
    await orchestrator.startup()

    try:
        dataset = QADataset.load(args.benchmark)
        results: list[QAEvalResult] = []
        gap_list: list[dict] = []  # record any gaps encountered

        for strategy_name in args.strategies:
            print(f"\n▶ Evaluating strategy: {strategy_name}")
            for item in dataset:
                answer = await orchestrator.answer(item.question, strategy=strategy_name)

                result = QAEvalResult(
                    question_id=item.question_id,
                    predicted_answer=answer.answer_text,
                    expected_answer=item.expected_answer,
                    exact_match=compute_exact_match(answer.answer_text, item.expected_answer),
                    f1_score=compute_token_f1(answer.answer_text, item.expected_answer),
                    faithfulness=compute_faithfulness(answer.answer_text, answer.evidence),
                    relevance=compute_context_relevance(item.question, answer.evidence),
                    latency_ms=answer.latency_ms,
                    retrieval_strategy=strategy_name,
                    context_count=len(answer.evidence),
                    gap_detected=bool(getattr(answer, "gap_detection", None)),
                    gap_type=(answer.gap_detection.gap_type
                              if getattr(answer, "gap_detection", None) else None),
                )
                results.append(result)
                if result.gap_detected:
                    gap_list.append({
                        "question_id": item.question_id,
                        "strategy": strategy_name,
                        "gap_type": result.gap_type,
                    })
                print(f"  {item.question_id}: F1={result.f1_score:.3f} "
                      f"EM={'✓' if result.exact_match else '✗'} "
                      f"Faith={result.faithfulness:.3f}" +
                      (f" GAP={result.gap_type}" if result.gap_detected else ""))

        # Compare strategies
        comparator = StrategyComparator()
        comparisons = comparator.compare(results)

        # Generate reports
        reporter = EvaluationReporter()
        files = reporter.generate(comparisons, output_dir=args.output_dir, formats=args.formats)
        print(f"\n✅ Reports generated: {[str(f) for f in files]}")

        # Write gap list if any
        if gap_list:
            gap_path = Path(args.output_dir) / "evaluation_gap_report.json"
            gap_path.write_text(json.dumps(gap_list, indent=2, ensure_ascii=False))
            print(f"✅ Gap report written: {gap_path}")

    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
