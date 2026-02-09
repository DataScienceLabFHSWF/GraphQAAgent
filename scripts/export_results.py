#!/usr/bin/env python3
"""Export evaluation results to JSON/CSV/LaTeX for thesis (C3.5)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export evaluation results")
    parser.add_argument("--input", required=True, help="Path to evaluation_report.json")
    parser.add_argument("--output-dir", default="reports/export/")
    parser.add_argument("--formats", nargs="+", default=["csv", "latex"])
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if "csv" in args.formats:
        csv_path = output / "results.csv"
        if data:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            print(f"  CSV: {csv_path}")

    if "latex" in args.formats:
        latex_path = output / "results.tex"
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Retrieval Strategy Comparison}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Strategy & Avg F1 & EM Rate & Faithfulness & Relevance & Latency \\",
            r"\midrule",
        ]
        for row in data:
            lines.append(
                f"  {row['strategy']} & {row['avg_f1']:.3f} & "
                f"{row['exact_match_rate']:.1%} & {row['avg_faithfulness']:.3f} & "
                f"{row['avg_relevance']:.3f} & {row['avg_latency_ms']:.0f} \\\\"
            )
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        latex_path.write_text("\n".join(lines))
        print(f"  LaTeX: {latex_path}")


if __name__ == "__main__":
    main()
