#!/usr/bin/env python3
"""Run all retrieval strategies against the benchmark and produce a comparison table.

Generates a per-question × per-strategy comparison matrix as JSON, Markdown,
CSV, and HTML for manual rating and strategy comparison.

Optionally runs DeepEval LLM-as-a-judge metrics (answer relevancy,
faithfulness, contextual relevancy, correctness) via a local Ollama model.

Usage:
    python scripts/run_benchmark_comparison.py
    python scripts/run_benchmark_comparison.py --benchmark data/qa_benchmarks/benchmark_v3.json
    python scripts/run_benchmark_comparison.py --strategies vector_only graph_only hybrid cypher agentic hybrid_sota
    python scripts/run_benchmark_comparison.py --output-dir reports/comparison/
    python scripts/run_benchmark_comparison.py --deepeval --deepeval-model deepseek-r1:1.5b
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kgrag.agents.orchestrator import Orchestrator
from kgrag.core.config import Settings
from kgrag.evaluation.metrics import (
    compute_context_relevance,
    compute_exact_match,
    compute_faithfulness,
    compute_token_f1,
)
from kgrag.evaluation.qa_dataset import QADataset

# Optional: DeepEval LLM-as-a-judge metrics
try:
    from kgrag.evaluation.deepeval_metrics import DeepEvalEvaluator
    _DEEPEVAL_OK = True
except ImportError:
    _DEEPEVAL_OK = False

# All strategies the orchestrator supports
ALL_STRATEGIES = [
    "vector_only",
    "graph_only",
    "hybrid",
    "cypher",
    "agentic",
    "hybrid_sota",
]

# Strategy display names and descriptions for reports
STRATEGY_INFO: dict[str, dict[str, str]] = {
    "vector_only": {
        "display_name": "Vector Search",
        "description": (
            "Pure semantic similarity retrieval from Qdrant vector embeddings. "
            "Finds document chunks whose embeddings are closest to the query embedding."
        ),
    },
    "graph_only": {
        "display_name": "Graph Traversal",
        "description": (
            "Knowledge graph traversal via Neo4j. Identifies named entities in the query "
            "and explores their neighbourhood relationships to gather structured evidence."
        ),
    },
    "hybrid": {
        "display_name": "Hybrid Fusion (RRF)",
        "description": (
            "Combines vector and graph retrieval using Reciprocal Rank Fusion. "
            "Merges semantic similarity scores with structural graph evidence."
        ),
    },
    "cypher": {
        "display_name": "Cypher Query Generation",
        "description": (
            "LLM-generated Cypher queries executed directly against Neo4j. "
            "Translates natural-language questions into structured graph database queries."
        ),
    },
    "agentic": {
        "display_name": "Agentic ReAct",
        "description": (
            "Multi-step ReAct agent with iterative Think-on-Graph exploration. "
            "Plans, retrieves, reflects, and refines across multiple reasoning steps."
        ),
    },
    "hybrid_sota": {
        "display_name": "Hybrid SOTA (Ontology-Informed)",
        "description": (
            "State-of-the-art hybrid retrieval with ontology-informed query expansion. "
            "Extends Hybrid Fusion with SPARQL-based class-hierarchy and synonym expansion."
        ),
    },
}


def _display_name(strategy: str) -> str:
    """Return the human-readable display name for a strategy."""
    return STRATEGY_INFO.get(strategy, {}).get("display_name", strategy)


def _safe_answer(answer: Any) -> dict[str, Any]:
    """Extract answer fields with full retrieval context for post-hoc evaluation."""
    # Evidence — full text + provenance
    evidence_items: list[dict[str, Any]] = []
    for e in getattr(answer, "evidence", []):
        item: dict[str, Any] = {
            "text": getattr(e, "text", ""),
            "score": round(getattr(e, "score", 0.0), 4),
            "source": (
                getattr(e, "source", "").value
                if hasattr(getattr(e, "source", None), "value")
                else str(getattr(e, "source", ""))
            ),
        }
        prov = getattr(e, "provenance", None)
        if prov:
            item["provenance"] = {
                "doc_id": getattr(prov, "doc_id", None),
                "source_id": getattr(prov, "source_id", None),
                "retrieval_strategy": getattr(prov, "retrieval_strategy", ""),
                "retrieval_score": round(getattr(prov, "retrieval_score", 0.0), 4),
            }
        evidence_items.append(item)

    # Cited entities — full detail
    cited_entities: list[dict[str, Any]] = []
    for ent in getattr(answer, "cited_entities", []):
        cited_entities.append({
            "id": getattr(ent, "id", ""),
            "label": getattr(ent, "label", ""),
            "entity_type": getattr(ent, "entity_type", ""),
            "description": getattr(ent, "description", ""),
        })

    # Cited relations
    cited_relations: list[dict[str, Any]] = []
    for rel in getattr(answer, "cited_relations", []):
        cited_relations.append({
            "source_id": getattr(rel, "source_id", ""),
            "target_id": getattr(rel, "target_id", ""),
            "relation_type": getattr(rel, "relation_type", ""),
            "confidence": round(getattr(rel, "confidence", 0.0), 4),
        })

    # Reasoning steps
    reasoning_steps: list[dict[str, Any]] = []
    for step in getattr(answer, "reasoning_steps", []):
        reasoning_steps.append({
            "step_id": getattr(step, "step_id", 0),
            "sub_question": getattr(step, "sub_question", ""),
            "evidence_text": getattr(step, "evidence_text", ""),
            "answer_fragment": getattr(step, "answer_fragment", ""),
            "confidence": round(getattr(step, "confidence", 0.0), 4),
        })

    # Verification — full detail
    verif = getattr(answer, "verification", None)
    verification: dict[str, Any] | None = None
    if verif:
        verification = {
            "is_faithful": getattr(verif, "is_faithful", None),
            "faithfulness_score": getattr(verif, "faithfulness_score", None),
            "supported_claims": getattr(verif, "supported_claims", []),
            "unsupported_claims": getattr(verif, "unsupported_claims", []),
            "contradicted_claims": getattr(verif, "contradicted_claims", []),
            "entity_coverage": getattr(verif, "entity_coverage", 0.0),
        }

    return {
        "answer_text": getattr(answer, "answer_text", ""),
        "confidence": round(getattr(answer, "confidence", 0.0), 4),
        "latency_ms": round(getattr(answer, "latency_ms", 0.0), 1),
        "evidence": evidence_items,
        "evidence_count": len(evidence_items),
        "cited_entities": cited_entities,
        "cited_relations": cited_relations,
        "reasoning_chain": getattr(answer, "reasoning_chain", []),
        "reasoning_steps": reasoning_steps,
        "subgraph_json": getattr(answer, "subgraph_json", None),
        "verification": verification,
        "react_metadata": getattr(answer, "react_metadata", None),
    }


async def run_benchmark(
    benchmark_path: str,
    strategies: list[str],
    output_dir: str,
    *,
    deepeval_enabled: bool = False,
    deepeval_model: str = "gemma4:e4b",
    deepeval_metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Execute the full benchmark comparison.

    Returns the complete results dict suitable for JSON serialization.
    """
    settings = Settings()  # type: ignore[call-arg]
    orchestrator = Orchestrator(settings)
    await orchestrator.startup()

    # DeepEval evaluator (optional)
    de_evaluator = None
    if deepeval_enabled:
        if not _DEEPEVAL_OK:
            print("⚠ deepeval not installed — skipping LLM-as-a-judge metrics")
        else:
            base_url = settings.ollama.base_url
            de_evaluator = DeepEvalEvaluator(
                model_name=deepeval_model,
                base_url=base_url,
                metrics=deepeval_metrics,
            )
            print(f"DeepEval enabled: model={deepeval_model}, metrics={deepeval_metrics or 'all'}")

    dataset = QADataset.load(benchmark_path)
    print(f"Loaded {len(dataset)} questions from {benchmark_path}")
    print(f"Strategies: {strategies}\n")

    # Results: list of per-question records
    records: list[dict[str, Any]] = []
    strategy_summaries: dict[str, dict[str, Any]] = {s: {
        "total_f1": 0.0, "total_faith": 0.0, "total_relevance": 0.0,
        "total_latency": 0.0, "exact_matches": 0, "count": 0,
        "per_type_f1": {}, "per_type_count": {},
        "per_difficulty_f1": {}, "per_difficulty_count": {},
    } for s in strategies}

    try:
        for qi, item in enumerate(dataset):
            print(f"[{qi+1}/{len(dataset)}] {item.question_id}: {item.question[:70]}...")
            record: dict[str, Any] = {
                "question_id": item.question_id,
                "question": item.question,
                "expected_answer": item.expected_answer,
                "expected_entities": item.expected_entities,
                "difficulty": item.difficulty,
                "question_type": item.question_type,
                "category": getattr(item, "category", ""),
                "strategies": {},
            }

            for strategy in strategies:
                print(f"  → {strategy}...", end=" ", flush=True)
                t0 = time.perf_counter()
                try:
                    answer = await orchestrator.answer(
                        item.question, strategy=strategy
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000

                    # Compute metrics
                    f1 = compute_token_f1(answer.answer_text, item.expected_answer)
                    em = compute_exact_match(answer.answer_text, item.expected_answer)
                    faith = compute_faithfulness(answer.answer_text, answer.evidence)
                    rel = compute_context_relevance(item.question, answer.evidence)

                    entry = {
                        **_safe_answer(answer),
                        "f1_score": round(f1, 4),
                        "exact_match": em,
                        "faithfulness": round(faith, 4),
                        "relevance": round(rel, 4),
                        "latency_ms": round(elapsed_ms, 1),
                        "error": None,
                    }

                    # DeepEval LLM-as-a-judge scoring
                    if de_evaluator is not None:
                        retrieval_ctx = [
                            e.text if hasattr(e, 'text') else str(e)
                            for e in getattr(answer, 'evidence', [])
                        ]
                        try:
                            de_result = await de_evaluator.evaluate(
                                question=item.question,
                                answer=answer.answer_text,
                                expected_answer=item.expected_answer or None,
                                retrieval_contexts=retrieval_ctx or None,
                            )
                            entry["deepeval"] = de_result.to_dict()
                            print(f"    deepeval: rel={de_result.answer_relevancy} faith={de_result.faithfulness} corr={de_result.correctness}")
                        except Exception as de_exc:
                            entry["deepeval"] = {"error": str(de_exc)}
                            print(f"    deepeval error: {de_exc}")

                    # Update summaries
                    ss = strategy_summaries[strategy]
                    ss["total_f1"] += f1
                    ss["total_faith"] += faith
                    ss["total_relevance"] += rel
                    ss["total_latency"] += elapsed_ms
                    ss["exact_matches"] += int(em)
                    ss["count"] += 1
                    qt = item.question_type
                    ss["per_type_f1"][qt] = ss["per_type_f1"].get(qt, 0.0) + f1
                    ss["per_type_count"][qt] = ss["per_type_count"].get(qt, 0) + 1
                    diff = item.difficulty
                    ss["per_difficulty_f1"][diff] = ss["per_difficulty_f1"].get(diff, 0.0) + f1
                    ss["per_difficulty_count"][diff] = ss["per_difficulty_count"].get(diff, 0) + 1

                    status = f"F1={f1:.3f} {'✓' if em else '✗'} {elapsed_ms:.0f}ms"
                    print(status)

                except Exception as exc:
                    print(f"ERROR: {exc}")
                    entry = {
                        "answer_text": "",
                        "confidence": 0.0,
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                        "evidence": [],
                        "evidence_count": 0,
                        "cited_entities": [],
                        "cited_relations": [],
                        "reasoning_chain": [],
                        "reasoning_steps": [],
                        "subgraph_json": None,
                        "verification": None,
                        "react_metadata": None,
                        "f1_score": 0.0,
                        "exact_match": False,
                        "faithfulness": 0.0,
                        "relevance": 0.0,
                        "error": str(exc),
                    }

                record["strategies"][strategy] = entry

            records.append(record)

    finally:
        await orchestrator.shutdown()

    # Build aggregate summaries
    aggregates: dict[str, Any] = {}
    for strategy, ss in strategy_summaries.items():
        n = max(ss["count"], 1)
        per_type = {
            qt: round(ss["per_type_f1"][qt] / max(ss["per_type_count"][qt], 1), 4)
            for qt in ss["per_type_f1"]
        }
        per_diff = {
            d: round(ss["per_difficulty_f1"][d] / max(ss["per_difficulty_count"][d], 1), 4)
            for d in ss["per_difficulty_f1"]
        }
        agg: dict[str, Any] = {
            "avg_f1": round(ss["total_f1"] / n, 4),
            "avg_faithfulness": round(ss["total_faith"] / n, 4),
            "avg_relevance": round(ss["total_relevance"] / n, 4),
            "avg_latency_ms": round(ss["total_latency"] / n, 1),
            "exact_match_rate": round(ss["exact_matches"] / n, 4),
            "num_questions": ss["count"],
            "per_type_f1": per_type,
            "per_difficulty_f1": per_diff,
        }

        # Aggregate DeepEval scores if present
        if de_evaluator is not None:
            de_keys = [
                "answer_relevancy", "faithfulness", "contextual_relevancy",
                "contextual_precision", "contextual_recall", "correctness",
            ]
            for dk in de_keys:
                de_field = f"deepeval_{dk}"
                vals = []
                for rec in records:
                    sr = rec["strategies"].get(strategy, {})
                    de = sr.get("deepeval", {})
                    v = de.get(dk) if isinstance(de, dict) else None
                    if v is not None:
                        vals.append(v)
                agg[f"avg_{de_field}"] = round(sum(vals) / len(vals), 4) if vals else None

        aggregates[strategy] = agg

    result = {
        "metadata": {
            "benchmark": benchmark_path,
            "strategies": strategies,
            "num_questions": len(dataset),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "aggregate": aggregates,
        "per_question": records,
    }

    # Write outputs
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _write_json(result, out / "comparison_results.json")
    _write_markdown_table(result, out / "comparison_table.md")
    _write_csv(result, out / "comparison_results.csv")
    _write_html(result, out / "comparison_table.html")

    print(f"\n✅ Comparison complete. Reports in {out}/")
    return result


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    print(f"  → {path}")


def _write_markdown_table(data: dict, path: Path) -> None:
    strategies = data["metadata"]["strategies"]
    lines: list[str] = []

    # Header
    lines.append("# Retrieval Strategy Comparison\n")
    lines.append(f"Benchmark: `{data['metadata']['benchmark']}` "
                 f"| {data['metadata']['num_questions']} questions "
                 f"| {data['metadata']['timestamp']}\n")

    # Strategy legend
    lines.append("## Strategy Descriptions\n")
    for s in strategies:
        info = STRATEGY_INFO.get(s, {})
        dn = info.get("display_name", s)
        desc = info.get("description", "")
        lines.append(f"- **{dn}** (`{s}`): {desc}")
    lines.append("")

    lines.append("## Aggregate Results\n")

    # Detect if DeepEval scores are present
    has_deepeval = any(
        "avg_deepeval_answer_relevancy" in data["aggregate"].get(s, {})
        for s in strategies
    )

    header = "| Strategy | Avg F1 | EM Rate | Avg Faithfulness | Avg Relevance | Avg Latency (ms) |"
    sep = "|----------|--------|---------|-----------------|---------------|------------------|"
    if has_deepeval:
        header += " DE Answer Rel. | DE Faithful. | DE Correct. |"
        sep += "----------------|--------------|-------------|"
    lines.append(header)
    lines.append(sep)
    for s in strategies:
        a = data["aggregate"][s]
        dn = _display_name(s)
        row = (
            f"| **{dn}** | {a['avg_f1']:.4f} | {a['exact_match_rate']:.4f} "
            f"| {a['avg_faithfulness']:.4f} | {a['avg_relevance']:.4f} "
            f"| {a['avg_latency_ms']:.0f} |"
        )
        if has_deepeval:
            de_ar = a.get("avg_deepeval_answer_relevancy")
            de_f = a.get("avg_deepeval_faithfulness")
            de_c = a.get("avg_deepeval_correctness")
            de_ar_s = f"{de_ar:.4f}" if de_ar is not None else "N/A"
            de_f_s = f"{de_f:.4f}" if de_f is not None else "N/A"
            de_c_s = f"{de_c:.4f}" if de_c is not None else "N/A"
            row += f" {de_ar_s} | {de_f_s} | {de_c_s} |"
        lines.append(row)

    # Per-question detail table
    lines.append("\n## Per-Question Comparison\n")
    q_header = "| Q-ID | Question | Difficulty | Type |"
    q_sep = "|------|----------|------------|------|"
    for s in strategies:
        dn = _display_name(s)
        q_header += f" {dn} (F1) | {dn} Answer (excerpt) |"
        q_sep += "------|----------------------|"
    lines.append(q_header)
    lines.append(q_sep)

    for rec in data["per_question"]:
        q_text = rec["question"].replace("|", "\\|").replace("\n", " ")
        row = f"| {rec['question_id']} | {q_text} | {rec['difficulty']} | {rec['question_type']} |"
        for s in strategies:
            sr = rec["strategies"].get(s, {})
            f1 = sr.get("f1_score", 0.0)
            ans = sr.get("answer_text", "")[:120].replace("|", "\\|").replace("\n", " ")
            row += f" {f1:.3f} | {ans}… |"
        lines.append(row)

    # Rating columns
    lines.append("\n## Manual Rating Template\n")
    lines.append("Use this table to manually rate answer quality (1-5 scale):\n")
    r_header = "| Q-ID |"
    r_sep = "|------|"
    for s in strategies:
        r_header += f" {_display_name(s)} Rating |"
        r_sep += "--------|"
    r_header += " Notes |"
    r_sep += "-------|"
    lines.append(r_header)
    lines.append(r_sep)
    for rec in data["per_question"]:
        row = f"| {rec['question_id']} |"
        for _ in strategies:
            row += " |"
        row += " |"
        lines.append(row)

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {path}")


def _write_csv(data: dict, path: Path) -> None:
    strategies = data["metadata"]["strategies"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        header = ["question_id", "question", "expected_answer", "difficulty", "question_type", "category"]
        for s in strategies:
            header.extend([
                f"{s}_answer", f"{s}_f1", f"{s}_em", f"{s}_faith",
                f"{s}_relevance", f"{s}_latency_ms", f"{s}_confidence",
                f"{s}_evidence_count", f"{s}_error",
            ])
        writer.writerow(header)

        # Rows
        for rec in data["per_question"]:
            row = [
                rec["question_id"], rec["question"], rec["expected_answer"],
                rec["difficulty"], rec["question_type"], rec.get("category", ""),
            ]
            for s in strategies:
                sr = rec["strategies"].get(s, {})
                row.extend([
                    sr.get("answer_text", ""),
                    sr.get("f1_score", 0.0),
                    sr.get("exact_match", False),
                    sr.get("faithfulness", 0.0),
                    sr.get("relevance", 0.0),
                    sr.get("latency_ms", 0.0),
                    sr.get("confidence", 0.0),
                    sr.get("evidence_count", 0),
                    sr.get("error", ""),
                ])
            writer.writerow(row)

    print(f"  → {path}")


def _write_html(data: dict, path: Path) -> None:
    strategies = data["metadata"]["strategies"]
    _esc = lambda t: str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    html_parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>KG-RAG Strategy Comparison</title>",
        "<style>",
        "  body { font-family: 'Segoe UI', sans-serif; margin: 2em; background: #fafafa; color: #333; }",
        "  h1, h2, h3 { color: #2c3e50; }",
        "  h3 { margin-top: 1.5em; }",
        "  table { border-collapse: collapse; width: 100%; margin: 1em 0; }",
        "  th, td { border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 0.9em; }",
        "  th { background: #34495e; color: white; position: sticky; top: 0; z-index: 1; }",
        "  tr:nth-child(even) { background: #f2f2f2; }",
        "  tr:hover { background: #e8f4f8; }",
        "  .good { background: #d4edda; }",
        "  .medium { background: #fff3cd; }",
        "  .bad { background: #f8d7da; }",
        "  .score { font-weight: bold; text-align: center; }",
        "  .answer-cell { max-width: 500px; }",
        "  details { cursor: pointer; }",
        "  details summary { white-space: normal; word-wrap: break-word; }",
        "  details[open] summary { font-weight: bold; margin-bottom: 0.3em; }",
        "  .question-text { font-size: 1.05em; margin: 0.2em 0; }",
        "  .expected { background: #eef; padding: 0.5em; border-left: 3px solid #66b; margin: 0.5em 0; white-space: pre-wrap; word-wrap: break-word; }",
        "  .legend { background: #fff; border: 1px solid #ccc; border-radius: 6px; padding: 1em 1.5em; margin: 1em 0; }",
        "  .legend h2 { margin-top: 0; }",
        "  .legend dl { margin: 0; }",
        "  .legend dt { font-weight: bold; color: #2c3e50; margin-top: 0.6em; }",
        "  .legend dd { margin-left: 1.5em; color: #555; }",
        "  .evidence-list { font-size: 0.85em; color: #555; max-height: 200px; overflow-y: auto; padding: 0.3em; }",
        "  .evidence-item { border-bottom: 1px solid #eee; padding: 3px 0; }",
        "</style>",
        "</head><body>",
        f"<h1>KG-RAG Strategy Comparison</h1>",
        f"<p>Benchmark: <code>{_esc(data['metadata']['benchmark'])}</code> | "
        f"{data['metadata']['num_questions']} questions | {data['metadata']['timestamp']}</p>",
    ]

    # ── Strategy legend ──────────────────────────────────────────────
    html_parts.append("<div class='legend'>")
    html_parts.append("<h2>Strategy Descriptions</h2>")
    html_parts.append("<dl>")
    for s in strategies:
        info = STRATEGY_INFO.get(s, {})
        dn = _esc(info.get("display_name", s))
        desc = _esc(info.get("description", ""))
        html_parts.append(f"<dt>{dn} <code>({s})</code></dt>")
        html_parts.append(f"<dd>{desc}</dd>")
    html_parts.append("</dl></div>")

    # ── Aggregate table ──────────────────────────────────────────────
    has_de = any(
        "avg_deepeval_answer_relevancy" in data["aggregate"].get(s, {})
        for s in strategies
    )
    html_parts.append("<h2>Aggregate Results</h2>")
    agg_hdr = ("<table><tr><th>Strategy</th><th>Avg F1</th><th>EM Rate</th>"
               "<th>Avg Faithfulness</th><th>Avg Relevance</th><th>Avg Latency (ms)</th>")
    if has_de:
        agg_hdr += "<th>DE Answer Rel.</th><th>DE Faithful.</th><th>DE Correct.</th>"
    agg_hdr += "</tr>"
    html_parts.append(agg_hdr)
    for s in strategies:
        a = data["aggregate"][s]
        dn = _esc(_display_name(s))
        row = (
            f"<tr><td><b>{dn}</b></td>"
            f"<td class='score'>{a['avg_f1']:.4f}</td>"
            f"<td class='score'>{a['exact_match_rate']:.4f}</td>"
            f"<td class='score'>{a['avg_faithfulness']:.4f}</td>"
            f"<td class='score'>{a['avg_relevance']:.4f}</td>"
            f"<td class='score'>{a['avg_latency_ms']:.0f}</td>"
        )
        if has_de:
            de_ar = a.get("avg_deepeval_answer_relevancy")
            de_f = a.get("avg_deepeval_faithfulness")
            de_c = a.get("avg_deepeval_correctness")
            row += (
                f"<td class='score'>{f'{de_ar:.4f}' if de_ar is not None else 'N/A'}</td>"
                f"<td class='score'>{f'{de_f:.4f}' if de_f is not None else 'N/A'}</td>"
                f"<td class='score'>{f'{de_c:.4f}' if de_c is not None else 'N/A'}</td>"
            )
        row += "</tr>"
        html_parts.append(row)
    html_parts.append("</table>")

    # ── Per-question detail ──────────────────────────────────────────
    html_parts.append("<h2>Per-Question Detail</h2>")
    for rec in data["per_question"]:
        q_id = _esc(rec['question_id'])
        q_text = _esc(rec['question'])
        expected = _esc(rec.get('expected_answer', ''))
        cat = _esc(rec.get('category', ''))
        html_parts.append(f"<h3>{q_id}</h3>")
        html_parts.append(f"<p class='question-text'>{q_text}</p>")
        html_parts.append(
            f"<p>Difficulty: <b>{rec['difficulty']}</b> | "
            f"Type: <b>{_esc(rec['question_type'])}</b>"
            + (f" | Category: <b>{cat}</b>" if cat else "")
            + "</p>"
        )
        if expected:
            html_parts.append(f"<div class='expected'><b>Expected:</b> {expected}</div>")

        html_parts.append(
            "<table><tr><th>Strategy</th><th>F1</th><th>EM</th>"
            "<th>Faith.</th><th>Rel.</th><th>Conf.</th>"
            "<th>Latency</th><th>Evidence</th><th>Answer</th></tr>"
        )
        for s in strategies:
            sr = rec["strategies"].get(s, {})
            f1 = sr.get("f1_score", 0.0)
            cls = "good" if f1 >= 0.7 else ("medium" if f1 >= 0.4 else "bad")
            ans_full = _esc(sr.get("answer_text", ""))
            error = sr.get("error", "")
            if error:
                display_ans = f"<span style='color:red'>{_esc(error)}</span>"
                summary_text = _esc(error)[:150]
            else:
                display_ans = ans_full
                summary_text = ans_full[:200] + ("…" if len(ans_full) > 200 else "")

            # Evidence preview
            ev_count = sr.get("evidence_count", 0)
            evidence_list = sr.get("evidence", [])
            if evidence_list and isinstance(evidence_list, list) and isinstance(evidence_list[0], dict):
                ev_details = "<div class='evidence-list'>"
                for ei, ev in enumerate(evidence_list[:10], 1):
                    ev_text = _esc(ev.get('text', '')[:300])
                    ev_src = _esc(ev.get('source', ''))
                    ev_score = ev.get('score', 0)
                    ev_details += (
                        f"<div class='evidence-item'>"
                        f"<b>[{ei}]</b> ({ev_src}, score={ev_score:.3f}) {ev_text}"
                        f"</div>"
                    )
                if len(evidence_list) > 10:
                    ev_details += f"<div class='evidence-item'><i>… and {len(evidence_list) - 10} more</i></div>"
                ev_details += "</div>"
                ev_cell = f"<details><summary>{ev_count}</summary>{ev_details}</details>"
            else:
                ev_cell = str(ev_count)

            dn = _esc(_display_name(s))
            html_parts.append(
                f"<tr class='{cls}'>"
                f"<td><b>{dn}</b></td>"
                f"<td class='score'>{f1:.3f}</td>"
                f"<td class='score'>{'✓' if sr.get('exact_match') else '✗'}</td>"
                f"<td class='score'>{sr.get('faithfulness', 0):.3f}</td>"
                f"<td class='score'>{sr.get('relevance', 0):.3f}</td>"
                f"<td class='score'>{sr.get('confidence', 0):.3f}</td>"
                f"<td class='score'>{sr.get('latency_ms', 0):.0f}ms</td>"
                f"<td class='score'>{ev_cell}</td>"
                f"<td class='answer-cell'><details><summary>{summary_text}</summary>"
                f"{display_ans}</details></td>"
                f"</tr>"
            )
        html_parts.append("</table>")

    html_parts.append("</body></html>")
    path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run benchmark comparison across all retrieval strategies"
    )
    parser.add_argument(
        "--benchmark",
        default="data/qa_benchmarks/benchmark_v2.json",
        help="Path to the benchmark JSON file",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=ALL_STRATEGIES,
        help=f"Strategies to evaluate (default: {ALL_STRATEGIES})",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/comparison/",
        help="Directory for output reports",
    )
    parser.add_argument(
        "--deepeval",
        action="store_true",
        help="Enable DeepEval LLM-as-a-judge metrics (requires deepeval + local Ollama model)",
    )
    parser.add_argument(
        "--deepeval-model",
        default="gemma4:e4b",
        help="Ollama model for DeepEval judge (default: gemma4:e4b)",
    )
    parser.add_argument(
        "--deepeval-metrics",
        nargs="+",
        default=None,
        help="DeepEval metrics to run (default: all). Options: answer_relevancy faithfulness contextual_relevancy contextual_precision contextual_recall correctness",
    )
    args = parser.parse_args()

    await run_benchmark(
        args.benchmark,
        args.strategies,
        args.output_dir,
        deepeval_enabled=args.deepeval,
        deepeval_model=args.deepeval_model,
        deepeval_metrics=args.deepeval_metrics,
    )


if __name__ == "__main__":
    asyncio.run(main())
