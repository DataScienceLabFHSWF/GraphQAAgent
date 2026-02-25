"""Reasoning visualisation page — CoT steps, verification, subgraph.

Select a past QA answer and visualise its full reasoning DAG.
"""

from __future__ import annotations

import os

import streamlit as st

try:
    _secret_url = st.secrets.get("api_url", None)
except Exception:
    _secret_url = None
API_URL = _secret_url or os.environ.get("API_URL", "http://localhost:8080/api/v1")

st.header("🧠 Reasoning Visualisation")

# -- Select a past session / answer -----------------------------------------

st.subheader("Select a session")
import httpx

sessions = []
try:
    resp = httpx.get(f"{API_URL}/chat/sessions", timeout=30)
    sessions = [s["session_id"] for s in resp.json()]
except Exception:
    pass

if sessions:
    st.session_state.selected_session = st.selectbox(
        "Session", sessions, key="select_session"
    )
else:
    st.info("No active sessions available")

st.divider()

st.subheader("Chain-of-Thought Steps")
from kgrag.frontend.components.reasoning_dag import render_reasoning_chain

turn = None
if st.session_state.get("selected_session"):
    try:
        resp = httpx.get(
            f"{API_URL}/chat/sessions/{st.session_state.selected_session}/history",
            timeout=30,
        )
        history = resp.json()
    except Exception as e:
        history = []
        st.error(f"Failed to load history: {e}")

    answers = [h for h in history if h.get("assistant")]
    if answers:
        idx = st.selectbox(
            "Answer",
            list(range(len(answers))),
            format_func=lambda i: answers[i].get("assistant", "")[:80],
            key="select_answer",
        )
        turn = answers[idx]
        reasoning_steps = turn.get("reasoning_chain", [])
        verification = turn.get("verification")
        render_reasoning_chain(reasoning_steps, verification)
    else:
        st.info("No answers in this session")

st.subheader("Evidence Subgraph")
from kgrag.frontend.components.subgraph_viewer import render_subgraph

if turn is not None:
    sub = turn.get("subgraph")
    if sub:
        render_subgraph(sub)
    else:
        st.info("No subgraph available for this answer")

st.subheader("Answer Verification")
if turn is not None and turn.get("verification"):
    ver = turn["verification"]
    st.metric(
        "Faithfulness", f"{ver.get('faithfulness_score',0):.0%}",
        delta="✓" if ver.get("is_faithful", True) else "⚠ Issues found",
    )
    if ver.get("contradicted_claims"):
        st.warning("Contradicted claims:")
        for claim in ver.get("contradicted_claims", []):
            st.markdown(f"- {claim}")
else:
    st.info("Select an answer to view verification details")
