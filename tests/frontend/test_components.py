"""Simple tests for Streamlit components to ensure they import and run without errors."""

from __future__ import annotations

import pytest

import streamlit as st

from kgrag.frontend.components import render_subgraph
from kgrag.frontend.components.reasoning_dag import render_reasoning_chain


def test_render_subgraph_minimal(monkeypatch):
    # calling the function should not raise; monkeypatch html component to no-op
    monkeypatch.setattr(st.components.v1, "html", lambda *args, **kwargs: None)
    data = {"nodes": [{"id": "n1", "label": "Node1", "type": "Facility"}], "edges": []}
    render_subgraph(data, height=100)


def test_render_reasoning_chain_empty():
    # nothing to display should just print an info box
    render_reasoning_chain([])


def test_render_reasoning_chain_with_steps(monkeypatch):
    # simple chain should not error
    # ensure secrets are available so `.get` doesn't raise
    monkeypatch.setattr(st, "secrets", {"api_url": "http://localhost:8080/api/v1"})
    steps = [{"sub_question": "Q1", "evidence_text": "E", "answer_fragment": "A", "confidence": 0.7, "grounding_entities": ["e1"]}]
    render_reasoning_chain(steps, verification={"is_faithful": True, "faithfulness_score": 0.9, "contradicted_claims": []})
