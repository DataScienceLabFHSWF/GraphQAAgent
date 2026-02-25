"""Styled chat message bubble component.

Provides a richer chat message display than the default ``st.chat_message``
— with collapsible metadata sections.

Delegated implementation tasks
------------------------------
* TODO: Implement a custom Streamlit component with HTML/CSS for richer
  styling (confidence badge, entity chips, source pills).
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_chat_message(
    role: str,
    content: str,
    *,
    confidence: float = 0.0,
    reasoning: list[str] | None = None,
    provenance: list[dict[str, Any]] | None = None,
    latency_ms: float = 0.0,
) -> None:
    """Render a single chat message with metadata sections.

    TODO (delegate): Replace with a custom Streamlit component for
    richer styling.
    """
    with st.chat_message(role):
        st.write(content)

        if confidence:
            st.caption(f"Confidence: {confidence:.0%} | Latency: {latency_ms:.0f}ms")

        if reasoning:
            with st.expander("🧠 Reasoning"):
                for i, step in enumerate(reasoning, 1):
                    st.markdown(f"**{i}.** {step}")

        if provenance:
            with st.expander(f"📚 Sources ({len(provenance)})"):
                for p in provenance:
                    st.markdown(f"- [{p.get('source', '?')}] score={p.get('score', 0):.3f}")
