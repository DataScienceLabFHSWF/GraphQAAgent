"""Ontology browser page — TBox class hierarchy and properties.

Fetches the class hierarchy from the Explorer API, renders it as a
collapsible tree, displays properties for the selected class, and shows
the raw TTL ontology file with syntax highlighting.
"""

from __future__ import annotations

import os

import streamlit as st

try:
    _secret_url = st.secrets.get("api_url", None)
except Exception:
    _secret_url = None
API_URL = _secret_url or os.environ.get("API_URL", "http://localhost:8080/api/v1")

st.header("📐 Ontology Browser")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Class Hierarchy")
    import httpx

    try:
        if "ontology_tree" not in st.session_state:
            resp = httpx.get(f"{API_URL}/explore/ontology/tree", timeout=30)
            st.session_state.ontology_tree = resp.json()
        tree_data = st.session_state.ontology_tree

        def _render_node(uri: str, level: int = 0):
            children = tree_data.get(uri, {}).get("children", [])
            st.write("    " * level + uri)
            for c in children:
                _render_node(c, level + 1)

        # render roots (those not appearing as child)
        roots = [u for u in tree_data if not any(u in tree_data[p].get("children", []) for p in tree_data)]
        for r in roots:
            _render_node(r)
    except Exception as e:
        st.error(f"Failed to load ontology tree: {e}")

    # allow class selection via dropdown
    if "ontology_tree" in st.session_state:
        all_classes = list(st.session_state.ontology_tree.keys())
        st.session_state.selected_class = st.selectbox("Select class", all_classes)

with col2:
    st.subheader("Class Properties")
    if st.session_state.get("selected_class"):
        import httpx
        try:
            resp = httpx.get(
                f"{API_URL}/explore/ontology/classes/{st.session_state.selected_class}/properties",
                timeout=30,
            )
            props = resp.json()
            st.table(props)
        except Exception as e:
            st.error(f"Failed to load properties: {e}")

st.divider()
st.subheader("Raw Ontology (TTL)")
from pathlib import Path

ttl_path = Path("data/ontologies/ndd_ontology.ttl")
if ttl_path.exists():
    st.code(ttl_path.read_text(), language="turtle")
else:
    st.info("Ontology file not found")
