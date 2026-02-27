"""Integration-style test showing that AgenticGraphRAG records ontology gaps."""

from __future__ import annotations
import pytest

from kgrag.retrieval.agentic_rag import AgenticGraphRAG
from kgrag.retrieval.active_ontology import OntologyGap
from kgrag.retrieval.ontology_context import OntologyContext

# Fake connectors and provider minimal stubs
class DummyNeo4j:
    pass

class DummyQdrant:
    pass

class DummyOllama:
    def get_chat_model(self):
        class DummyModel:
            def bind_tools(self, tools):
                return self
            async def ainvoke(self, messages):
                class R:
                    tool_calls = []
                return R()
        return DummyModel()


def make_retriever():
    neo4j = DummyNeo4j()
    qdrant = DummyQdrant()
    ollama = DummyOllama()
    class DummyFuseki:
        async def query(self, *args, **kwargs):
            return []
    ontology = OntologyContext(DummyFuseki())  # empty context
    return AgenticGraphRAG(
        neo4j=neo4j,
        neo4j_config=None,
        qdrant=qdrant,
        ollama=ollama,
        ontology_context=ontology,
    )


def test_parse_ontology_records_gap():
    retriever = make_retriever()
    # initially no gaps
    assert retriever._detected_gaps == []

    # simulate a failed lookup
    ctxs, ents, rels, chains = retriever._parse_ontology_result(
        "No ontology entries found for FooBar",
        lookup_query="class FooBar",
    )
    assert isinstance(retriever._detected_gaps[0], OntologyGap)
    assert retriever._detected_gaps[0].gap_type == "missing_class"

