"""Unit tests for explorer API routes using fake connectors."""

from __future__ import annotations

import pytest

from kgrag.api import explorer_routes as er
from kgrag.connectors.neo4j import Neo4jConnector
from kgrag.connectors.fuseki import FusekiConnector


class DummySession:
    def __init__(self, records=None, single=None):
        self._records = records or []
        self._single = single

    async def run(self, *args, **kwargs):
        return self

    async def data(self):
        return self._records

    async def single(self):
        return self._single

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class DummyDriver:
    def __init__(self, session):
        self._session = session

    def session(self, **kwargs):
        return self._session


class DummyNeo4j:
    def __init__(self, records=None, single=None):
        self._config = type("C", (), {"database": "neo4j"})
        self._label = "Entity"
        self.driver = DummyDriver(DummySession(records, single))

    async def get_neighbourhood(self, entity_ids, *, max_hops=1):
        # return dummy nodes and relations
        class Ent:
            def __init__(self, id_, label, t):
                self.id = id_
                self.label = label
                self.entity_type = t
        class Rel:
            def __init__(self, s, t, rt):
                self.source_id = s
                self.target_id = t
                self.relation_type = rt
        return ([Ent(entity_ids[0], "Foo", "Type")], [Rel(entity_ids[0], "bar", "rel")])


class DummyFuseki:
    def __init__(self, query_result=None):
        self._query_result = query_result or []

    async def query(self, sparql: str):
        return self._query_result

    async def get_class_properties(self, class_uri: str):
        from kgrag.core.models import OntologyProperty
        return [OntologyProperty(uri="p", label="prop", domain_uri=class_uri, range_uri="r")]


@pytest.fixture(autouse=True)
def patch_connectors(monkeypatch):
    # patch module-level variables
    # New code uses dict(node.items()) and labels(n) AS _labels
    node = {"id": "e1", "label": "E1", "node_type": "Entity"}
    neo = DummyNeo4j(records=[{"n": node, "_labels": ["Entity"]}])
    fuse = DummyFuseki(query_result=[{"class": "C1", "label": "Class1", "parent": None}])
    monkeypatch.setattr(er, "_neo4j", neo)
    monkeypatch.setattr(er, "_fuseki", fuse)
    yield


def test_list_entities_simple():
    import asyncio
    data = asyncio.get_event_loop().run_until_complete(er.list_entities())
    assert isinstance(data, list)
    assert data[0]["id"] == "e1"


def test_get_entity_not_found():
    import asyncio
    # patch driver to return no record
    er._neo4j.driver = DummyDriver(DummySession(records=[]))
    with pytest.raises(er.HTTPException):
        asyncio.get_event_loop().run_until_complete(er.get_entity("missing"))


def test_get_entity_found():
    import asyncio
    node = {"id": "e2", "label": "E2", "node_type": "Entity"}
    rec = {"n": node, "_labels": ["Entity"], "outgoing": [], "incoming": []}
    er._neo4j.driver = DummyDriver(DummySession(records=[rec], single=rec))
    result = asyncio.get_event_loop().run_until_complete(er.get_entity("e2"))
    assert result["id"] == "e2"
    assert result["outgoing"] == []


def test_list_ontology_classes():
    import asyncio
    classes = asyncio.get_event_loop().run_until_complete(er.list_ontology_classes())
    assert classes[0]["class"] == "C1"


def test_list_relations_and_stats():
    import asyncio
    # patch relation counts
    er._neo4j.driver = DummyDriver(DummySession(records=[{"type": "REL", "count": 5}]))
    rels = asyncio.get_event_loop().run_until_complete(er.list_relation_types())
    assert rels[0]["type"] == "REL"
    stats = asyncio.get_event_loop().run_until_complete(er.get_kg_stats())
    assert "relation_types" in stats


def test_get_entity_subgraph():
    import asyncio
    sub = asyncio.get_event_loop().run_until_complete(er.get_entity_subgraph("x", depth=1))
    assert "nodes" in sub and "edges" in sub

