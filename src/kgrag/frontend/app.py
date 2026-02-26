"""Streamlit main app — GraphQAAgent Interactive Demo.

Launch with::

    streamlit run src/kgrag/frontend/app.py --server.port 8501

Uses the sidebar for navigation and system status.  Pages live under
``pages/`` and are auto-discovered by Streamlit.  The sidebar includes a
live health check against the API backend.
"""

from __future__ import annotations

import streamlit as st

import os

# API base can come from Streamlit secrets (used during development) or
# fallback to environment variable (useful in Docker).
try:
    _secret_url = st.secrets.get("api_url", None)
except Exception:
    _secret_url = None
API_BASE_URL = _secret_url or os.environ.get("API_URL", "http://localhost:8080/api/v1")

st.set_page_config(
    page_title="GraphQA Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔍 GraphQA Agent — Knowledge Graph Q&A")
st.markdown(
    """
Interactive demo for querying the ontology-driven Knowledge Graph.

**Pages:**
- 💬 **Chat** — Ask questions about the knowledge graph
- 🕸️ **KG Explorer** — Browse entities and relations
- 📐 **Ontology** — Explore the TBox class hierarchy
- 🧠 **Reasoning** — Visualise Chain-of-Thought reasoning

Use the sidebar to navigate.
"""
)

# -- Sidebar: system status -------------------------------------------------
with st.sidebar:
    st.header("System Status")

    # health check
    import httpx
    try:
        r = httpx.get(f"{API_BASE_URL}/health", timeout=5)
        data = r.json()
        st.success(f"API: {data['status']} (v{data['version']})")
    except Exception as e:
        st.error(f"API: Unreachable ({e})")
    st.info("Connect to API at " + API_BASE_URL)

    st.divider()
    st.caption("GraphQA Agent v0.1.0")
