#!/usr/bin/env python3
"""Run DeepEval LLM-as-a-judge on previously generated benchmark results.

Reads the saved comparison_results.json (which contains all answer texts)
and evaluates them with DeepEval metrics — no need to regenerate answers.

Usage:
    python scripts/run_deepeval_on_results.py
    python scripts/run_deepeval_on_results.py --results reports/comparison/comparison_results.json
    python scripts/run_deepeval_on_results.py --model qwen3:8b --metrics answer_relevancy correctness
    python scripts/run_deepeval_on_results.py --strategies vector_only hybrid_sota
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kgrag.evaluation.deepeval_metrics import DeepEvalEvaluator


async def run_deepeval_evaluation(
    results_path: str,
    output_dir: str,
    *,
    model_name: str = "qwen3:8b",
    base_url: str = "http://localhost:18136",
    metrics: list[str] | None = None,
    strategies: list[str] | None = None,
) -> dict[str, Any]:
    """Load saved results and run DeepEval metrics on all answers.

    Returns the enriched results dict.
    """
    # Load saved results
    data = json.loads(Path(results_path).read_text())
    all_strategies = data["metadata"]["strategies"]
    target_strategies = strategies or all_strategies
    print(f"Loaded {len(data['per_question'])} questions × {len(target_strategies)} strategies")

    # Init evaluator
    evaluator = DeepEvalEvaluator(
        model_name=model_name,
        base_url=base_url,
        metrics=metrics,
    )
    active = sorted(evaluator._active_metrics)
    print(f"DeepEval model: {model_name}")
    print(f"Active metrics: {active}\n")

    # Track aggregates
    agg: dict[str, dict[str, list[float]]] = {
        s: {m: [] for m in active} for s in target_strategies
    }

    total = len(data["per_question"]) * len(target_strategies)
    done = 0

    for qi, rec in enumerate(data["per_question"]):
        qid = rec["question_id"]
        question = rec["question"]
        expected = rec.get("expected_answer", "")
        difficulty = rec.get("difficulty", "?")

        print(f"[{qi+1}/{len(data['per_question'])}] {qid} (difficulty={difficulty}): {question[:60]}...")

        for strategy in target_strategies:
            sr = rec["strategies"].get(strategy)
            if sr is None or sr.get("error"):
                done += 1
                continue

            answer_text = sr.get("answer_text", "")
            if not answer_text:
                done += 1
                continue

            # Extract retrieval contexts — prefer saved evidence texts,
            # fall back to reasoning_chain as proxy for older result files.
            evidence_list = sr.get("evidence", [])
            if evidence_list and isinstance(evidence_list, list):
                if isinstance(evidence_list[0], dict):
                    retrieval_contexts = [e["text"] for e in evidence_list if e.get("text")]
                else:
                    retrieval_contexts = [str(e) for e in evidence_list if e]
            else:
                retrieval_contexts = []
            if not retrieval_contexts:
                reasoning_chain = sr.get("reasoning_chain", [])
                retrieval_contexts = reasoning_chain if reasoning_chain else None

            print(f"  → {strategy}...", end=" ", flush=True)

            try:
                de_result = await evaluator.evaluate(
                    question=question,
                    answer=answer_text,
                    expected_answer=expected or None,
                    retrieval_contexts=retrieval_contexts,
                )

                # Store in the results
                sr["deepeval"] = de_result.to_dict()

                # Collect for aggregation
                for m in active:
                    val = getattr(de_result, m, None)
                    if val is not None:
                        agg[strategy][m].append(val)

                # Print summary
                parts = []
                for m in ["answer_relevancy", "faithfulness", "correctness"]:
                    v = getattr(de_result, m, None)
                    if v is not None:
                        parts.append(f"{m[:5]}={v:.3f}")
                print(" ".join(parts) if parts else "skipped")

                if de_result.has_errors:
                    for ek, ev in de_result.errors.items():
                        print(f"    ⚠ {ek}: {ev}")

            except Exception as exc:
                sr["deepeval"] = {"error": str(exc)}
                print(f"ERROR: {exc}")

            done += 1

    # Add DeepEval aggregates to the result
    data["deepeval_aggregate"] = {}
    for strategy in target_strategies:
        strat_agg: dict[str, Any] = {}
        for m in active:
            vals = agg[strategy][m]
            if vals:
                strat_agg[f"avg_{m}"] = round(sum(vals) / len(vals), 4)
                strat_agg[f"count_{m}"] = len(vals)
            else:
                strat_agg[f"avg_{m}"] = None
                strat_agg[f"count_{m}"] = 0
        data["deepeval_aggregate"][strategy] = strat_agg

    data["deepeval_metadata"] = {
        "model": model_name,
        "metrics": active,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategies_evaluated": target_strategies,
    }

    # Write enriched results
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    enriched_path = out / "comparison_results_deepeval.json"
    enriched_path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    print(f"\n→ {enriched_path}")

    # Write summary table
    summary_path = out / "deepeval_summary.md"
    _write_summary_md(data, target_strategies, active, summary_path)
    print(f"→ {summary_path}")

    print("\n✅ DeepEval evaluation complete.")
    return data


def _write_summary_md(
    data: dict,
    strategies: list[str],
    metrics: list[str],
    path: Path,
) -> None:
    """Write a Markdown summary table of DeepEval results."""
    lines = [
        "# DeepEval LLM-as-a-Judge Results\n",
        f"Model: `{data['deepeval_metadata']['model']}` | "
        f"{data['deepeval_metadata']['timestamp']}\n",
        "## Aggregate Scores\n",
    ]

    # Header
    header = "| Strategy |"
    sep = "|----------|"
    for m in metrics:
        short = m.replace("contextual_", "ctx_").replace("answer_", "ans_")
        header += f" {short} |"
        sep += "--------|"
    lines.append(header)
    lines.append(sep)

    for s in strategies:
        sa = data["deepeval_aggregate"].get(s, {})
        row = f"| **{s}** |"
        for m in metrics:
            v = sa.get(f"avg_{m}")
            row += f" {v:.4f} |" if v is not None else " N/A |"
        lines.append(row)

    # Per-difficulty breakdown
    lines.append("\n## Per-Difficulty Breakdown\n")
    for s in strategies:
        lines.append(f"\n### {s}\n")
        diff_agg: dict[Any, dict[str, list[float]]] = {}
        for rec in data["per_question"]:
            diff = rec.get("difficulty", "?")
            sr = rec["strategies"].get(s, {})
            de = sr.get("deepeval", {})
            if isinstance(de, dict) and "error" not in de:
                if diff not in diff_agg:
                    diff_agg[diff] = {m: [] for m in metrics}
                for m in metrics:
                    v = de.get(m)
                    if v is not None:
                        diff_agg[diff][m].append(v)

        if not diff_agg:
            lines.append("No data.\n")
            continue

        header = "| Difficulty |"
        sep = "|------------|"
        for m in metrics:
            short = m.replace("contextual_", "ctx_").replace("answer_", "ans_")
            header += f" {short} | n |"
            sep += "--------|---|"
        lines.append(header)
        lines.append(sep)

        for diff in sorted(diff_agg.keys()):
            row = f"| {diff} |"
            for m in metrics:
                vals = diff_agg[diff][m]
                if vals:
                    avg = sum(vals) / len(vals)
                    row += f" {avg:.4f} | {len(vals)} |"
                else:
                    row += " N/A | 0 |"
            lines.append(row)

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DeepEval LLM-as-a-judge on saved benchmark results"
    )
    parser.add_argument(
        "--results",
        default="reports/comparison/comparison_results.json",
        help="Path to the saved comparison results JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/comparison/",
        help="Directory for output reports",
    )
    parser.add_argument(
        "--model",
        default="qwen3:8b",
        help="Ollama model for DeepEval judge (default: qwen3:8b)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:18136",
        help="Ollama API base URL",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="DeepEval metrics to run (default: all). Options: answer_relevancy faithfulness contextual_relevancy contextual_precision contextual_recall correctness",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Strategies to evaluate (default: all from results)",
    )
    args = parser.parse_args()

    await run_deepeval_evaluation(
        args.results,
        args.output_dir,
        model_name=args.model,
        base_url=args.base_url,
        metrics=args.metrics,
        strategies=args.strategies,
    )


if __name__ == "__main__":
    asyncio.run(main())
