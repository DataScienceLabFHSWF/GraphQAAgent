#!/usr/bin/env python3
"""Run ontology gap analysis and escalate findings to the HITL pipeline.

Combines structural gap detection (ABox vs TBox mismatch) with
QA-driven gap detection (low-confidence answers from benchmark).

Usage:
    python scripts/run_gap_analysis.py
    python scripts/run_gap_analysis.py --benchmark data/qa_benchmarks/benchmark_v2.json
    python scripts/run_gap_analysis.py --output gap_report.json
    python scripts/run_gap_analysis.py --export-for-extender gap_report_extender.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from kgrag.agents.orchestrator import Orchestrator
from kgrag.core.config import Settings
from kgrag.evaluation.qa_dataset import QADataset
from kgrag.hitl.change_proposals import ChangeProposalService
from kgrag.hitl.kg_versioning import KGVersioningService
from kgrag.hitl.ontology_gap_analyzer import OntologyGapAnalyzer


async def main() -> None:
    parser = argparse.ArgumentParser(description="KG-RAG Ontology Gap Analysis")
    parser.add_argument(
        "--benchmark",
        default="data/qa_benchmarks/benchmark_v2.json",
        help="QA benchmark to detect QA-driven gaps",
    )
    parser.add_argument(
        "--strategy",
        default="hybrid",
        help="Retrieval strategy for QA-driven gap detection",
    )
    parser.add_argument(
        "--output",
        default="reports/gap_analysis/gap_report.json",
        help="Output path for the gap report",
    )
    parser.add_argument(
        "--export-for-extender",
        default=None,
        help="If set, also export in OntologyExtender format to this path",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum entity frequency for structural gap candidates",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.65,
        help="Cosine similarity threshold for TBox matching",
    )
    args = parser.parse_args()

    settings = Settings()  # type: ignore[call-arg]
    orchestrator = Orchestrator(settings)
    await orchestrator.startup()

    try:
        # Construct the gap analyzer
        gap_analyzer = OntologyGapAnalyzer(
            neo4j=orchestrator.neo4j,
            fuseki=orchestrator.fuseki,
            ollama=orchestrator.ollama,
            min_frequency=args.min_frequency,
            similarity_threshold=args.similarity_threshold,
        )

        # Phase 1: Run QA benchmark to detect QA-driven gaps
        print("=" * 60)
        print("Phase 1: QA-driven gap detection")
        print("=" * 60)

        if Path(args.benchmark).exists():
            dataset = QADataset.load(args.benchmark)
            print(f"Running {len(dataset)} questions with strategy '{args.strategy}'...")

            for i, item in enumerate(dataset):
                print(f"  [{i+1}/{len(dataset)}] {item.question[:60]}...", end=" ")
                try:
                    answer = await orchestrator.answer(
                        item.question, strategy=args.strategy
                    )
                    # Feed into QA gap detector
                    gap = await gap_analyzer.qa_gap_detector.analyse_answer(
                        question=item.question,
                        answer_text=answer.answer_text,
                        confidence=answer.confidence,
                        evidence_count=len(answer.evidence),
                    )
                    if gap:
                        print(f"GAP: {gap.gap_type} (conf={answer.confidence:.2f})")
                    else:
                        print(f"OK (conf={answer.confidence:.2f})")
                except Exception as exc:
                    print(f"ERROR: {exc}")
                    # Register as a gap (failure to answer)
                    await gap_analyzer.qa_gap_detector.analyse_answer(
                        question=item.question,
                        answer_text="",
                        confidence=0.0,
                        evidence_count=0,
                    )

            qa_gaps = gap_analyzer.qa_gap_detector.get_gaps()
            print(f"\nQA-driven gaps found: {len(qa_gaps)}")
        else:
            print(f"Benchmark file not found: {args.benchmark} — skipping QA phase.")

        # Phase 2: Structural gap analysis (ABox vs TBox)
        print("\n" + "=" * 60)
        print("Phase 2: Structural gap analysis (ABox vs TBox)")
        print("=" * 60)

        report = await gap_analyzer.analyze()

        print(f"\nTotal ABox entities:  {report.total_abox_entities}")
        print(f"Covered by TBox:     {report.covered_entities}")
        print(f"Uncovered:           {report.uncovered_entities}")
        print(f"Coverage:            {report.coverage_pct:.1%}")
        print(f"Structural gaps:     {len(report.gap_candidates)}")
        print(f"QA-driven gaps:      {len(report.qa_driven_gaps)}")

        if report.gap_candidates:
            print("\nTop structural gap candidates:")
            for gc in report.gap_candidates[:10]:
                print(
                    f"  • {gc.entity_type} "
                    f"(freq={gc.frequency}, closest='{gc.closest_seed_class}', "
                    f"dist={gc.semantic_distance:.2f}, struct={gc.structural_score:.1f})"
                )

        # Phase 3: Escalate to HITL
        print("\n" + "=" * 60)
        print("Phase 3: Escalate gaps to HITL pipeline")
        print("=" * 60)

        versioning = KGVersioningService(orchestrator.neo4j)
        proposal_service = ChangeProposalService(versioning, orchestrator.fuseki)
        proposals = gap_analyzer.escalate_to_hitl(report, proposal_service)

        print(f"\nCreated {len(proposals)} change proposals for expert review:")
        for p in proposals[:10]:
            print(f"  • [{p.id}] {p.proposal_type.value}: {p.rationale[:80]}...")

        # Write output
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        export_data = gap_analyzer.export_for_ontology_extender(report)
        export_data["proposals"] = [
            {
                "id": p.id,
                "type": p.proposal_type.value,
                "status": p.status.value,
                "rationale": p.rationale,
                "proposed_data": p.proposed_data,
            }
            for p in proposals
        ]
        out_path.write_text(
            json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
        )
        print(f"\n✅ Gap report saved to: {out_path}")

        # Optionally export in OntologyExtender format
        if args.export_for_extender:
            ext_path = Path(args.export_for_extender)
            ext_path.parent.mkdir(parents=True, exist_ok=True)
            ext_data = gap_analyzer.export_for_ontology_extender(report)
            ext_path.write_text(
                json.dumps(ext_data, indent=2, ensure_ascii=False, default=str)
            )
            print(f"✅ OntologyExtender-compatible export: {ext_path}")

    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
