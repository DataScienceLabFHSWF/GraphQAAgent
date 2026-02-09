#!/usr/bin/env python3
"""Compare retrieval strategies head-to-head (C3.3 + C3.5).

Usage:
    python scripts/compare_strategies.py
    python scripts/compare_strategies.py --question "What methods are used for reactor A?"
"""

from __future__ import annotations

import argparse
import asyncio
import json

from kgrag.agents.orchestrator import Orchestrator
from kgrag.core.config import Settings


async def main() -> None:
    parser = argparse.ArgumentParser(description="Compare retrieval strategies")
    parser.add_argument("--question", "-q", required=True, help="Question to compare")
    parser.add_argument("--strategies", nargs="+", default=["vector_only", "graph_only", "hybrid"])
    args = parser.parse_args()

    settings = Settings()  # type: ignore[call-arg]
    orchestrator = Orchestrator(settings)
    await orchestrator.startup()

    try:
        results = {}
        for strategy in args.strategies:
            answer = await orchestrator.answer(args.question, strategy=strategy)
            results[strategy] = {
                "answer": answer.answer_text,
                "confidence": round(answer.confidence, 3),
                "latency_ms": round(answer.latency_ms, 1),
                "evidence_count": len(answer.evidence),
                "cited_entities": [e.label for e in answer.cited_entities],
                "reasoning_chain": answer.reasoning_chain,
            }

        print(json.dumps(results, indent=2, ensure_ascii=False))

    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
