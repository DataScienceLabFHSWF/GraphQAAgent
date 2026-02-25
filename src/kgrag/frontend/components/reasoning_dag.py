"""Chain-of-Thought reasoning visualisation.

Renders reasoning steps as an expandable timeline with confidence bars
and grounding entity links.

Delegated implementation tasks
------------------------------
* TODO: Render each step as a rich card with entity highlighting.
* TODO: Link grounding entities to the KG Explorer page.
* TODO: Add a DAG / flowchart view for multi-hop reasoning paths.
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st


def render_reasoning_chain(
    reasoning_steps: list[dict[str, Any]],
    verification: dict[str, Any] | None = None,
) -> None:
    """Render reasoning steps as an interactive timeline.

    Each step shows:
    * Sub-question decomposition
    * Evidence text with entity highlights
    * Confidence score (as progress bar)
    * Grounding entities (linked to KG)

    Parameters
    ----------
    reasoning_steps:
        List of step dicts with keys ``sub_question``, ``evidence_text``,
        ``answer_fragment``, ``confidence``, ``grounding_entities``.
    verification:
        Optional verification result with ``is_faithful``,
        ``faithfulness_score``, ``contradicted_claims``.
    """
    if not reasoning_steps:
        st.info("No reasoning steps available for this answer.")
        return

    for i, step in enumerate(reasoning_steps, 1):
        with st.expander(
            f"Step {i}: {step.get('sub_question', 'N/A')}",
            expanded=(i == 1),
        ):
            st.markdown(f"**Evidence:** {step.get('evidence_text', 'N/A')}")
            st.markdown(f"**Answer fragment:** {step.get('answer_fragment', 'N/A')}")
            st.progress(step.get("confidence", 0.0))

            entities = step.get("grounding_entities", [])
            if entities:
                # make each grounding entity a clickable link to the KG API search
                links = []
                for ent in entities:
                    try:
                        _api = st.secrets.get('api_url', 'http://localhost:8080/api/v1')
                    except Exception:
                        _api = os.environ.get('API_URL', 'http://localhost:8080/api/v1')
                    url = f"{_api}/explore/entities?search={ent}"
                    links.append(f"[{ent}]({url})")
                st.caption("Grounding: " + ", ".join(links))

    # Verification summary
    if verification:
        st.divider()
        faithful = verification.get("is_faithful", True)
        score = verification.get("faithfulness_score", 1.0)
        st.metric(
            "Faithfulness",
            f"{score:.0%}",
            delta="✓" if faithful else "⚠ Issues found",
        )
        if verification.get("contradicted_claims"):
            st.warning("Contradicted claims:")
            for claim in verification["contradicted_claims"]:
                st.markdown(f"- {claim}")
