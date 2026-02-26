"""Chat page — conversational QA over the Knowledge Graph.

Features:
* Chat-style UI with ``st.chat_message``
* Strategy / language selection
* Expandable reasoning chain + provenance
* Evidence panels with source texts
* Entity / relation cards extracted from KG
* Verification badge (faithfulness check)
* Gap detection alerts (HITL)
* Subgraph visualisation below the answer
* Feedback widget (thumbs-up / correction)
* Pre-loaded demo questions
* Export session as JSON / Markdown / HTML
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
            json.dumps(data, indent=2, default=str),
            file_name=f"session_{st.session_state.session_id}.json",
            mime="application/json",
        )
        # Markdown export
        try:
            from kgrag.demo.demo_export import export_markdown
            md_path = export_markdown(data)
            with open(md_path, "r") as f:
                st.download_button(
                    "Download Markdown",
                    f.read(),
                    file_name=f"session_{st.session_state.session_id}.md",
                    mime="text/markdown",
                    key="dl_md",
                )
        except Exception:
            pass

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

# -- Render helpers ---------------------------------------------------------


def _render_enriched_message(msg: dict) -> None:
    """Render all enriched fields for an assistant message."""
    st.write(msg["content"])

    # Confidence + latency metrics bar
    confidence = msg.get("confidence", 0.0)
    latency = msg.get("latency_ms", 0.0)
    strategy_used = msg.get("strategy_used", "")
    metrics_parts = []
    if confidence:
        metrics_parts.append(f"Confidence: {confidence:.0%}")
    if latency:
        metrics_parts.append(f"Latency: {latency:.0f}ms")
    if strategy_used:
        metrics_parts.append(f"Strategy: {strategy_used}")
    if metrics_parts:
        st.caption(" | ".join(metrics_parts))

    # Verification badge
    verification = msg.get("verification")
    if verification:
        is_faithful = verification.get("is_faithful", False)
        score = verification.get("faithfulness_score", 0)
        if is_faithful:
            st.success(f"✅ Verified faithful (score: {score:.0%})")
        else:
            st.warning(f"⚠️ Faithfulness uncertain (score: {score:.0%})")

    # Gap detection alerts
    gap = msg.get("gap_detection")
    if gap and gap.get("has_gap"):
        st.error(
            f"🔍 **Knowledge Gap Detected:** {gap.get('gap_type', 'unknown')} — "
            f"{gap.get('description', '')}"
        )

    # Evidence panel
    evidence_items = msg.get("evidence") or []
    if evidence_items:
        with st.expander(f"📄 Evidence ({len(evidence_items)} sources)"):
            for ev in evidence_items:
                if isinstance(ev, dict):
                    src = ev.get("source_id", "unknown")
                    text = ev.get("text", str(ev))
                    score = ev.get("relevance_score", 0)
                    st.markdown(f"**[{src}]** (relevance: {score:.2f})")
                    st.markdown(f"> {text[:500]}")
                    st.divider()
                else:
                    st.markdown(f"- {ev}")

    # Reasoning chain
    reasoning = msg.get("reasoning") or []
    reasoning_steps = msg.get("reasoning_steps") or []
    if reasoning_steps:
        with st.expander(f"🧠 Reasoning ({len(reasoning_steps)} steps)"):
            for i, step in enumerate(reasoning_steps, 1):
                if isinstance(step, dict):
                    label = step.get("description", step.get("step", f"Step {i}"))
                    conf = step.get("confidence", 0)
                    st.markdown(f"**Step {i}:** {label}")
                    if conf:
                        st.progress(min(conf, 1.0), text=f"Confidence: {conf:.0%}")
                    ev_text = step.get("evidence_text", "")
                    if ev_text:
                        st.caption(f"Evidence: {ev_text[:200]}")
                else:
                    st.markdown(f"**Step {i}:** {step}")
    elif reasoning:
        with st.expander("🧠 Reasoning Chain"):
            for i, step in enumerate(reasoning, 1):
                st.markdown(f"**Step {i}:** {step}")

    # Entity cards
    entities = msg.get("cited_entities") or []
    if entities:
        with st.expander(f"🏷️ Entities ({len(entities)})"):
            cols = st.columns(min(len(entities), 3))
            for idx, ent in enumerate(entities):
                with cols[idx % 3]:
                    if isinstance(ent, dict):
                        st.markdown(f"**{ent.get('label', '?')}**")
                        st.caption(ent.get("entity_type", ""))
                        if ent.get("description"):
                            st.markdown(f"_{ent['description'][:100]}_")
                    else:
                        st.markdown(f"- {ent}")

    # Relation cards
    relations = msg.get("cited_relations") or []
    if relations:
        with st.expander(f"🔗 Relations ({len(relations)})"):
            for rel in relations:
                if isinstance(rel, dict):
                    st.markdown(
                        f"**{rel.get('source_label', '?')}** "
                        f"→ _{rel.get('relation_type', '?')}_ → "
                        f"**{rel.get('target_label', '?')}**"
                    )
                else:
                    st.markdown(f"- {rel}")

    # Provenance
    provenance = msg.get("provenance") or []
    if provenance:
        with st.expander(f"📚 Sources ({len(provenance)})"):
            for p in provenance:
                if isinstance(p, dict):
                    st.markdown(f"- [{p.get('source', '?')}] score={p.get('score', 0):.3f}")
                else:
                    st.markdown(f"- {p}")

    # Subgraph
    subgraph = msg.get("subgraph")
    if subgraph:
        from kgrag.frontend.components.subgraph_viewer import render_subgraph

        st.divider()
        st.subheader("Evidence Subgraph")
        render_subgraph(subgraph)


# -- Render history ---------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _render_enriched_message(msg)
        else:
            st.write(msg["content"])

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
                        "include_evidence": True,
                    },
                    timeout=120,
                )
                data = resp.json()
                answer = data.get("message", {}).get("content", "")
                msg_data = {
                    "role": "assistant",
                    "content": answer,
                    "confidence": data.get("confidence", 0.0),
                    "latency_ms": data.get("latency_ms", 0.0),
                    "strategy_used": data.get("strategy_used", ""),
                    "reasoning": data.get("reasoning_chain", []),
                    "reasoning_steps": data.get("reasoning_steps", []),
                    "evidence": data.get("evidence", []),
                    "provenance": data.get("provenance", []),
                    "cited_entities": data.get("cited_entities", []),
                    "cited_relations": data.get("cited_relations", []),
                    "verification": data.get("verification"),
                    "gap_detection": data.get("gap_detection"),
                    "subgraph": data.get("subgraph"),
                }
            except Exception as e:
                answer = f"Error: {e}"
                msg_data = {"role": "assistant", "content": answer}

        _render_enriched_message(msg_data)

    st.session_state.messages.append(msg_data)

    # Feedback widget
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("👍", key=f"fb_up_{len(st.session_state.messages)}"):
            try:
                httpx.post(
                    f"{API_URL}/chat/feedback",
                    json={
                        "session_id": st.session_state.session_id,
                        "turn_index": len(st.session_state.messages) - 1,
                        "rating": 5,
                    },
                    timeout=10,
                )
                st.toast("Thanks for the feedback!")
            except Exception:
                pass
    with col2:
        if st.button("👎", key=f"fb_down_{len(st.session_state.messages)}"):
            st.session_state._show_correction = True
    if st.session_state.get("_show_correction"):
        correction = st.text_area("Suggest a correction:", key="correction_input")
        if st.button("Submit correction", key="submit_correction"):
            try:
                httpx.post(
                    f"{API_URL}/chat/feedback",
                    json={
                        "session_id": st.session_state.session_id,
                        "turn_index": len(st.session_state.messages) - 1,
                        "rating": 1,
                        "correction": correction,
                    },
                    timeout=10,
                )
                st.toast("Correction submitted — a change proposal has been created.")
                st.session_state._show_correction = False
            except Exception as e:
                st.error(f"Failed: {e}")

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
                        "include_evidence": True,
                    },
                    timeout=120,
                )
                data = resp.json()
                answer = data.get("message", {}).get("content", "")
                msg_data = {
                    "role": "assistant",
                    "content": answer,
                    "confidence": data.get("confidence", 0.0),
                    "latency_ms": data.get("latency_ms", 0.0),
                    "strategy_used": data.get("strategy_used", ""),
                    "reasoning": data.get("reasoning_chain", []),
                    "reasoning_steps": data.get("reasoning_steps", []),
                    "evidence": data.get("evidence", []),
                    "provenance": data.get("provenance", []),
                    "cited_entities": data.get("cited_entities", []),
                    "cited_relations": data.get("cited_relations", []),
                    "verification": data.get("verification"),
                    "gap_detection": data.get("gap_detection"),
                    "subgraph": data.get("subgraph"),
                }
            except Exception as e:
                answer = f"Error: {e}"
                msg_data = {"role": "assistant", "content": answer}

        _render_enriched_message(msg_data)

    st.session_state.messages.append(msg_data)
