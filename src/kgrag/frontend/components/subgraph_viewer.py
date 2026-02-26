"""Interactive subgraph visualisation using pyvis.

Re-exports the ``render_subgraph`` function and ``TYPE_COLORS`` mapping
from the components package for convenience.
"""

from __future__ import annotations

from kgrag.frontend.components import TYPE_COLORS, render_subgraph

__all__ = ["TYPE_COLORS", "render_subgraph"]
