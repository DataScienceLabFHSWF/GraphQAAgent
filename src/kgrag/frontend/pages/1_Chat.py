"""Chat page — conversational QA over the Knowledge Graph.

Features (stub):
* Chat-style UI with ``st.chat_message``
* Strategy / language selection
* Expandable reasoning chain + provenance
* Pre-loaded demo questions

Delegated implementation tasks
------------------------------
* TODO: Wire the actual ``/api/v1/chat/send`` call (non-streaming first,
  then SSE streaming via ``httpx`` + ``sseclient-py``).
* TODO: Display subgraph visualisation below the answer using the
  ``subgraph_viewer`` component.
* TODO: Add an "Export session" button (PDF / Markdown).
"""

from __future__ import annotations

import os
import uuid

import streamlit as st

try:
    _secret_url = st.secrets.get("api_url", None)
except Exception:
    _secret_url = None
API_URL = _secret_url or os.environ.get("API_URL", "http://localhost:8080/api/v1")

st.header("💬 Chat with the Knowledge Graph")

# -- Session state ----------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []

# -- Sidebar controls (must be declared before use) -------------------------

with st.sidebar:
    st.subheader("Chat Settings")
    strategy = st.selectbox(
        "Retrieval Strategy",
        ["hybrid_sota", "agentic", "hybrid", "cypher", "graph_only", "vector_only"],
    )
    language = st.selectbox("Language", ["de", "en"])

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.rerun()

    if st.button("📁 Export Chat"):
        import json

        data = st.session_state.messages
        st.download_button(
            "Download JSON",
            json.dumps(data, indent=2),
            file_name=f"session_{st.session_state.session_id}.json",
            mime="application/json",
        )

    st.divider()

    # Demo questions
    st.subheader("Demo Questions")
    demo_qs = [
        "Welche Gesetze regeln den Rückbau von Kernkraftwerken?",
        "What entities are involved in decommissioning?",
        "Which paragraphs govern radiation protection?",
        "What is the relationship between AtG and StrlSchG?",
        "Which waste types are in the KrWG?",
    ]
    for q in demo_qs:
        if st.button(q[:50] + "...", key=f"demo_{hash(q)}"):
            st.session_state._pending_demo_q = q
            st.rerun()

# -- Render history ---------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("confidence"):
            st.caption(
                f"Confidence: {msg['confidence']:.0%} | "
                f"Latency: {msg.get('latency_ms', 0):.0f}ms"
            )
        if msg.get("reasoning"):
            with st.expander("🧠 Reasoning Chain"):
                for i, step in enumerate(msg["reasoning"], 1):
                    st.markdown(f"**Step {i}:** {step}")
        if msg.get("provenance"):
            with st.expander(f"📚 Sources ({len(msg['provenance'])})"):
                for p in msg["provenance"]:
                    st.markdown(f"- [{p['source']}] score={p['score']:.3f}")

# -- Input ------------------------------------------------------------------

if prompt := st.chat_input("Ask a question about the Knowledge Graph..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Querying knowledge graph..."):
            import httpx

            try:
                resp = httpx.post(
                    f"{API_URL}/chat/send",
                    json={
                        "session_id": st.session_state.session_id,
                        "message": prompt,
                        "strategy": strategy,
                        "language": language,
                        "stream": False,
                    },
                    timeout=120,
                )
                data = resp.json()
                answer = data.get("message", {}).get("content", "")
                confidence = data.get("confidence", 0.0)
                reasoning = data.get("reasoning_chain", [])
                provenance = data.get("provenance", [])
                latency = data.get("latency_ms", 0.0)
                subgraph = data.get("subgraph")
            except Exception as e:
                answer = f"Error: {e}"
                confidence = 0.0
                reasoning = []
                provenance = []
                latency = 0.0
                subgraph = None

        st.write(answer)
        if confidence:
            st.caption(f"Confidence: {confidence:.0%} | Latency: {latency:.0f}ms")

        # render returned subgraph if present
        if subgraph:
            from kgrag.frontend.components.subgraph_viewer import render_subgraph

            st.divider()
            st.subheader("Evidence Subgraph")
            render_subgraph(subgraph)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "confidence": confidence,
            "reasoning": reasoning,
            "provenance": provenance,
            "latency_ms": latency,
            "subgraph": subgraph,
        }
    )

# -- Handle pending demo question (from sidebar button) --------------------

if pending_q := st.session_state.pop("_pending_demo_q", None):
    st.session_state.messages.append({"role": "user", "content": pending_q})
    with st.chat_message("user"):
        st.write(pending_q)
    with st.chat_message("assistant"):
        with st.spinner("Querying knowledge graph..."):
            import httpx as _httpx

            try:
                resp = _httpx.post(
                    f"{API_URL}/chat/send",
                    json={
                        "session_id": st.session_state.session_id,
                        "message": pending_q,
                        "strategy": strategy,
                        "language": language,
                        "stream": False,
                    },
                    timeout=120,
                )
                data = resp.json()
                answer = data.get("message", {}).get("content", "")
                confidence = data.get("confidence", 0.0)
                reasoning = data.get("reasoning_chain", [])
                provenance = data.get("provenance", [])
                latency = data.get("latency_ms", 0.0)
                subgraph = data.get("subgraph")
            except Exception as e:
                answer = f"Error: {e}"
                confidence = 0.0
                reasoning = []
                provenance = []
                latency = 0.0
                subgraph = None

        st.write(answer)
        if confidence:
            st.caption(f"Confidence: {confidence:.0%} | Latency: {latency:.0f}ms")
        if subgraph:
            from kgrag.frontend.components.subgraph_viewer import render_subgraph

            st.divider()
            st.subheader("Evidence Subgraph")
            render_subgraph(subgraph)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "confidence": confidence,
            "reasoning": reasoning,
            "provenance": provenance,
            "latency_ms": latency,
            "subgraph": subgraph,
        }
    )
