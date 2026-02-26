"""KG Explorer page — browse entities, relations, and subgraphs.

Searches entities via the Explorer API with type filtering, displays
entity details on selection, renders interactive subgraphs using pyvis,
and shows relation type distribution charts.
"""

from __future__ import annotations

import os

import streamlit as st

try:
    _secret_url = st.secrets.get("api_url", None)
except Exception:
    _secret_url = None
API_URL = _secret_url or os.environ.get("API_URL", "http://localhost:8080/api/v1")

st.header("🕸️ Knowledge Graph Explorer")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Search Entities")
    search_term = st.text_input("Search", placeholder="e.g. Kernkraftwerk")
    # Load available types from KG stats
    _type_options = ["All"]
    if "kg_stats" in st.session_state:
        _type_options += [t["type"] for t in st.session_state.get("kg_stats", {}).get("node_types", [])]
    else:
        _type_options += ["Facility", "Activity", "Component", "Regulation",
                          "Gesetzbuch", "Paragraf", "Organization", "Process",
                          "DomainConstant", "Action", "GoalState"]
    entity_type = st.selectbox("Entity type", _type_options)
    limit = st.slider("Max results", 10, 200, 50)

    if "entities" in st.session_state:
        st.subheader("Results")
        for ent in st.session_state.entities:
            if st.button(ent.get("label", ent.get("id")), key=f"ent_{ent.get('id')}"):
                st.session_state.selected_entity = ent

    if st.button("🔍 Search"):
        import httpx

        try:
            params = {"search": search_term, "limit": limit}
            if entity_type != "All":
                params["entity_type"] = entity_type
            resp = httpx.get(f"{API_URL}/explore/entities", params=params, timeout=30)
            st.session_state.entities = resp.json()
        except Exception as e:
            st.error(f"Entity search failed: {e}")

    st.divider()
    st.subheader("KG Statistics")
    import httpx

    try:
        if "kg_stats" not in st.session_state:
            resp = httpx.get(f"{API_URL}/explore/stats", timeout=30)
            st.session_state.kg_stats = resp.json()
        st.write(st.session_state.kg_stats)
    except Exception as e:
        st.error(f"Failed to fetch KG stats: {e}")

with col2:
    st.subheader("Subgraph Viewer")
    from kgrag.frontend.components.subgraph_viewer import render_subgraph

    if "selected_entity" in st.session_state:
        ent = st.session_state.selected_entity
        # show basic details
        st.markdown(f"**Label:** {ent.get('label')}  ")
        st.markdown(f"**Type:** {ent.get('entity_type', ent.get('type', '?'))}  ")
        if ent.get('description'):
            st.markdown(f"**Description:** {ent.get('description')}  ")
        eid = ent["id"]
        try:
            resp = httpx.get(f"{API_URL}/explore/entities/{eid}/subgraph", timeout=30)
            render_subgraph(resp.json())
        except Exception as e:
            st.error(f"Failed to load subgraph: {e}")

st.divider()
st.subheader("Relation Types")
import httpx, pandas as pd

try:
    if "relations" not in st.session_state:
        resp = httpx.get(f"{API_URL}/explore/relations", timeout=30)
        st.session_state.relations = resp.json()
    if st.session_state.relations:
        df = pd.DataFrame(st.session_state.relations)
        df = df.sort_values("count", ascending=False)
        st.bar_chart(df.set_index("type")["count"])
except Exception as e:
    st.error(f"Failed to fetch relations: {e}")
