"""Tests for the Orchestrator (integration-level, with mocks)."""

import pytest

# Orchestrator tests will be integration-level once connectors are available.
# For now, test the strategy resolution logic.

from kgrag.agents.orchestrator import Orchestrator
from kgrag.core.config import Settings


def test_get_retriever_unknown_raises(settings: Settings):
    orch = Orchestrator(settings)
    with pytest.raises(ValueError, match="Unknown strategy"):
        orch._get_retriever("nonexistent")


def test_get_retriever_valid(settings: Settings):
    orch = Orchestrator(settings)
    # These should not raise
    assert orch._get_retriever("vector_only") is orch.vector_retriever
    assert orch._get_retriever("graph_only") is orch.graph_retriever
    assert orch._get_retriever("hybrid") is orch.hybrid_retriever
