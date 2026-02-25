"""Interactive subgraph visualisation using pyvis.

Renders a vis.js-based network graph inside Streamlit via the pyvis
library.  The function is synchronous but may be called from async code
since it only uses local data.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network


# Colour palette for entity types
TYPE_COLORS: dict[str, str] = {
    "Facility": "#1565C0",
    "Organization": "#2E7D32",
    "LegalProvision": "#7B1FA2",
    "Process": "#EF6C00",
    "WasteType": "#C62828",
    "Material": "#00838F",
    "Document": "#4E342E",
    "Paragraph": "#5C6BC0",
    "Gesetz": "#7B1FA2",
}


def render_subgraph(subgraph: dict[str, Any], *, height: int = 500) -> None:
    """Render a subgraph as an interactive network in Streamlit.

    Parameters
    ----------
    subgraph:
        Dict with ``nodes`` and ``edges`` keys.
    height:
        Pixel height of the iframe.
    """
    net = Network(height=f"{height}px", width="100%", notebook=False)

    for node in subgraph.get("nodes", []):
        color = TYPE_COLORS.get(node.get("type", ""), "#757575")
        net.add_node(
            node.get("id"),
            label=node.get("label", ""),
            color=color,
            title=node.get("type", ""),
        )
    for edge in subgraph.get("edges", []):
        net.add_edge(
            edge.get("source"),
            edge.get("target"),
            label=edge.get("label", ""),
        )

    html = net.generate_html()
    components.html(html, height=height)
