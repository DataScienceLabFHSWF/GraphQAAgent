"""Guided demo runner — walks through scenarios with rich terminal output.

Runs demo scenarios against the live GraphQA API, displaying results with
Rich formatting including answer text and confidence scores.
"""

from __future__ import annotations

import asyncio

import structlog
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kgrag.demo.demo_data import DEMO_SCENARIOS

logger = structlog.get_logger(__name__)
console = Console()

API_URL = "http://localhost:8080/api/v1"


async def run_demo(
    scenario_idx: int = 0,
    *,
    api_url: str = API_URL,
    strategy: str = "hybrid_sota",
) -> None:
    """Run a guided demo scenario against the running API.

    Parameters
    ----------
    scenario_idx:
        Index into ``DEMO_SCENARIOS``.
    api_url:
        Base URL of the GraphQA API.
    strategy:
        Retrieval strategy to use.
    """
    scenario = DEMO_SCENARIOS[scenario_idx]

    console.print(
        Panel(
            f"[bold]{scenario['title']}[/bold]\n"
            f"{scenario.get('description', '')}\n\n"
            f"Questions: {len(scenario['questions'])}\n"
            f"Expected features: {', '.join(scenario['expected_features'])}",
            title="Demo Scenario",
            border_style="blue",
        )
    )

    async with httpx.AsyncClient() as client:
        for q in scenario["questions"]:
            console.print(f"\n[bold cyan]Q:[/bold cyan] {q}")
            try:
                resp = await client.post(
                    f"{api_url}/chat/send",
                    json={
                        "message": q,
                        "strategy": strategy,
                        "stream": False,
                    },
                    timeout=120,
                )
                data = resp.json()
                answer = data.get("message", {}).get("content", "")
                conf = data.get("confidence", 0.0)
                console.print(f"[green]A:[/green] {answer}")
                console.print(f"   Confidence: {conf:.0%}")
            except Exception as e:
                console.print(f"[red]Error calling API:[/red] {e}")


async def run_all_demos(*, api_url: str = API_URL) -> None:
    """Run all demo scenarios sequentially."""
    for i in range(len(DEMO_SCENARIOS)):
        await run_demo(i, api_url=api_url)
        console.print("\n" + "─" * 60 + "\n")


def main() -> None:
    """CLI entry point for the demo runner."""
    asyncio.run(run_all_demos())


if __name__ == "__main__":
    main()
