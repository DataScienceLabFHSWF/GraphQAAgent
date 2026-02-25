"""Demo session export — save a completed demo run as HTML or Markdown.

Delegated implementation tasks
------------------------------
* TODO: Implement ``export_html`` using Jinja2 template.
* TODO: Implement ``export_markdown`` for sharing in GitHub issues.
* TODO: Add subgraph SVG embedding in the HTML export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def export_html(
    session_data: list[dict[str, Any]],
    output_path: str | Path = "data/demo_export.html",
) -> Path:
    """Export a demo session as a self-contained HTML file.

    TODO (delegate): Use Jinja2 template to render questions, answers,
    reasoning chains, and subgraph visualisations.
    """
    raise NotImplementedError("demo_export.export_html")


def export_markdown(
    session_data: list[dict[str, Any]],
    output_path: str | Path = "data/demo_export.md",
) -> Path:
    """Export a demo session as Markdown.

    TODO (delegate): Format Q&A pairs, reasoning steps, and confidence
    metrics as Markdown tables and collapsible sections.
    """
    raise NotImplementedError("demo_export.export_markdown")
