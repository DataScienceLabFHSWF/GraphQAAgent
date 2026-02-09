"""Evaluation reporter (C3.5.4) — generate reports in Markdown, JSON, and HTML."""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from kgrag.core.models import StrategyComparison

logger = structlog.get_logger(__name__)


class EvaluationReporter:
    """Generate evaluation reports from strategy comparisons."""

    def generate(
        self,
        comparisons: list[StrategyComparison],
        *,
        output_dir: str | Path,
        formats: list[str] | None = None,
    ) -> list[Path]:
        """Generate reports in the requested formats.

        Returns list of paths to generated report files.
        """
        formats = formats or ["json", "markdown"]
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        generated: list[Path] = []

        if "json" in formats:
            p = output / "evaluation_report.json"
            self._write_json(comparisons, p)
            generated.append(p)

        if "markdown" in formats:
            p = output / "evaluation_report.md"
            self._write_markdown(comparisons, p)
            generated.append(p)

        if "html" in formats:
            p = output / "evaluation_report.html"
            self._write_html(comparisons, p)
            generated.append(p)

        logger.info("reporter.generated", files=[str(p) for p in generated])
        return generated

    # -- format writers -----------------------------------------------------

    @staticmethod
    def _write_json(comparisons: list[StrategyComparison], path: Path) -> None:
        data = []
        for c in comparisons:
            data.append({
                "strategy": c.strategy_name,
                "avg_f1": round(c.avg_f1, 4),
                "avg_faithfulness": round(c.avg_faithfulness, 4),
                "avg_relevance": round(c.avg_relevance, 4),
                "avg_latency_ms": round(c.avg_latency_ms, 1),
                "exact_match_rate": round(c.exact_match_rate, 4),
                "num_questions": c.num_questions,
                "per_type_f1": {k: round(v, 4) for k, v in c.per_type_f1.items()},
            })
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @staticmethod
    def _write_markdown(comparisons: list[StrategyComparison], path: Path) -> None:
        lines = [
            "# Evaluation Report",
            "",
            "## Strategy Comparison",
            "",
            "| Strategy | Avg F1 | Exact Match | Faithfulness | Relevance | Latency (ms) | N |",
            "|----------|--------|-------------|--------------|-----------|-------------|---|",
        ]
        for c in comparisons:
            lines.append(
                f"| {c.strategy_name} "
                f"| {c.avg_f1:.3f} "
                f"| {c.exact_match_rate:.1%} "
                f"| {c.avg_faithfulness:.3f} "
                f"| {c.avg_relevance:.3f} "
                f"| {c.avg_latency_ms:.0f} "
                f"| {c.num_questions} |"
            )
        lines.append("")
        path.write_text("\n".join(lines))

    @staticmethod
    def _write_html(comparisons: list[StrategyComparison], path: Path) -> None:
        rows = ""
        for c in comparisons:
            rows += (
                f"<tr><td>{c.strategy_name}</td>"
                f"<td>{c.avg_f1:.3f}</td>"
                f"<td>{c.exact_match_rate:.1%}</td>"
                f"<td>{c.avg_faithfulness:.3f}</td>"
                f"<td>{c.avg_relevance:.3f}</td>"
                f"<td>{c.avg_latency_ms:.0f}</td>"
                f"<td>{c.num_questions}</td></tr>\n"
            )
        html = f"""\
<!DOCTYPE html>
<html><head><title>Evaluation Report</title>
<style>
  body {{ font-family: sans-serif; margin: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
  th {{ background: #f4f4f4; }}
</style>
</head><body>
<h1>Evaluation Report</h1>
<table>
<tr><th>Strategy</th><th>Avg F1</th><th>Exact Match</th>
<th>Faithfulness</th><th>Relevance</th><th>Latency (ms)</th><th>N</th></tr>
{rows}
</table>
</body></html>"""
        path.write_text(html)
