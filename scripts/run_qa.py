#!/usr/bin/env python3
"""Interactive QA session (C3.4).

Usage:
    python scripts/run_qa.py                          # interactive REPL
    python scripts/run_qa.py --question "What is ...?" # single question
    python scripts/run_qa.py --strategy graph_only     # choose strategy
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from kgrag.agents.orchestrator import Orchestrator
from kgrag.core.config import Settings


async def interactive_loop(orchestrator: Orchestrator, strategy: str) -> None:
    """Run an interactive QA REPL."""
    print("\n🔍 KG-RAG QA Agent — Interactive Mode")
    print(f"   Strategy: {strategy}")
    print("   Type 'quit' to exit, 'strategy <name>' to switch.\n")

    while True:
        try:
            question = input("❓ Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() == "quit":
            break
        if question.lower().startswith("strategy "):
            strategy = question.split(maxsplit=1)[1]
            print(f"   → Switched to strategy: {strategy}")
            continue

        answer = await orchestrator.answer(question, strategy=strategy)

        print(f"\n📝 Answer (confidence: {answer.confidence:.2f}):")
        print(f"   {answer.answer_text}\n")

        if answer.reasoning_chain:
            print("🔗 Reasoning chain:")
            for step in answer.reasoning_chain:
                print(f"   • {step}")
            print()

        if answer.subgraph_json and answer.subgraph_json.get("nodes"):
            print(f"📊 Subgraph: {len(answer.subgraph_json['nodes'])} nodes, "
                  f"{len(answer.subgraph_json.get('edges', []))} edges")
            print()


async def single_question(orchestrator: Orchestrator, question: str, strategy: str) -> None:
    """Answer a single question and print the result as JSON."""
    answer = await orchestrator.answer(question, strategy=strategy)
    result = {
        "question": answer.question,
        "answer": answer.answer_text,
        "confidence": round(answer.confidence, 3),
        "reasoning_chain": answer.reasoning_chain,
        "cited_entities": [e.label for e in answer.cited_entities],
        "subgraph": answer.subgraph_json,
        "latency_ms": round(answer.latency_ms, 1),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


async def main() -> None:
    parser = argparse.ArgumentParser(description="KG-RAG QA Agent")
    parser.add_argument("--question", "-q", help="Single question (non-interactive)")
    parser.add_argument("--strategy", "-s", default="hybrid",
                        choices=["vector_only", "graph_only", "hybrid"])
    args = parser.parse_args()

    settings = Settings()  # type: ignore[call-arg]
    orchestrator = Orchestrator(settings)
    await orchestrator.startup()

    try:
        if args.question:
            await single_question(orchestrator, args.question, args.strategy)
        else:
            await interactive_loop(orchestrator, args.strategy)
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
