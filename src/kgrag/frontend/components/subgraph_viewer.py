"""Interactive subgraph visualisation using pyvis.

This is the dedicated module — re-exports from ``__init__`` for convenience.

Delegated implementation tasks
------------------------------
* TODO: Implement with pyvis (``Network.generate_html`` → ``st.components.html``).
* TODO: Add physics toggle, node grouping, zoom controls.
"""

from __future__ import annotations

from kgrag.frontend.components import TYPE_COLORS, render_subgraph

__all__ = ["TYPE_COLORS", "render_subgraph"]
