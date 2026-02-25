"""Explorer API routes — KG browsing, law graph, and ontology endpoints.

These endpoints let the frontend (Streamlit or TypeScript) interactively
browse the knowledge graph and ontology.

All endpoints currently return ``501 Not Implemented`` — they are stubs
ready for delegation.

Delegated implementation tasks
------------------------------
* TODO: Implement each endpoint using the injected Neo4j / Fuseki
  connectors.  The Cypher / SPARQL query sketches are in the docstrings.
* TODO: Add pagination helpers (cursor-based for Neo4j, OFFSET/LIMIT for
  SPARQL).
* TODO: Return vis.js-compatible JSON from ``get_entity_subgraph`` so
  the frontend can render directly.
* TODO: Build the ontology tree from Fuseki class hierarchy and cache it.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query

from kgrag.connectors.fuseki import FusekiConnector
from kgrag.connectors.neo4j import Neo4jConnector

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/explore", tags=["explorer"])

# Injected by server.py during startup
_neo4j: Neo4jConnector | None = None
_fuseki: FusekiConnector | None = None


def _serialise_props(props: dict[str, Any]) -> dict[str, Any]:
    """Convert Neo4j-specific types (DateTime, etc.) to JSON-safe values."""
    out: dict[str, Any] = {}
    for k, v in props.items():
        if hasattr(v, "iso_format"):  # neo4j.time.DateTime / Date / Time
            out[k] = v.iso_format()
        elif hasattr(v, "isoformat"):  # Python datetime
            out[k] = v.isoformat()
        elif isinstance(v, (list, tuple)):
            out[k] = [str(i) if hasattr(i, "iso_format") else i for i in v]
        else:
            out[k] = v
    return out


def set_connectors(neo4j: Neo4jConnector, fuseki: FusekiConnector) -> None:
    """Wire connectors (called once during lifespan)."""
    global _neo4j, _fuseki  # noqa: PLW0603
    _neo4j, _fuseki = neo4j, fuseki


# ===================================================================
# KG Entity browsing
# ===================================================================


@router.get("/entities")
async def list_entities(
    entity_type: str | None = None,
    search: str | None = None,
    limit: int = Query(default=50, le=500),
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List entities in the KG, optionally filtered by type or search term."""
    if _neo4j is None:
        raise HTTPException(503, "KG connector not initialised")

    # Nodes use Neo4j labels (Facility, Activity, etc.) as their type,
    # not a property called entity_type. Search across all nodes.
    if entity_type:
        query_lines: list[str] = [f"MATCH (n:`{entity_type}`)"]
    else:
        query_lines = ["MATCH (n)"]
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    where: list[str] = []
    if search:
        where.append("(toLower(n.label) CONTAINS toLower($search) OR toLower(n.name) CONTAINS toLower($search))")
        params["search"] = search
    if where:
        query_lines.append("WHERE " + " AND ".join(where))
    query_lines.append("RETURN n, labels(n) AS _labels ORDER BY n.label SKIP $offset LIMIT $limit")
    cypher = "\n".join(query_lines)

    async with _neo4j.driver.session(database=_neo4j._config.database) as session:
        result = await session.run(cypher, **params)
        records = await result.data()
    entities = []
    for r in records:
        node = r["n"]
        lbls = r.get("_labels", [])
        props = _serialise_props(dict(node.items()) if hasattr(node, 'items') else {})
        # Use Neo4j label as entity_type (skip generic ones)
        etype = next((l for l in lbls if l not in ("Entity", "Node")), lbls[0] if lbls else "Unknown")
        props["id"] = props.get("id", "")
        props["label"] = props.get("label", props.get("name", ""))
        props["entity_type"] = etype
        props["labels"] = lbls
        entities.append(props)
    return entities


@router.get("/entities/{entity_id}")
async def get_entity(entity_id: str) -> dict[str, Any]:
    """Get a single entity with its outgoing and incoming relations."""
    if _neo4j is None:
        raise HTTPException(503, "KG connector not initialised")
    # Match by id across all labels (the graph uses domain-specific labels)
    cypher = """
    MATCH (n {id: $id})
    OPTIONAL MATCH (n)-[r]->(m)
    OPTIONAL MATCH (p)-[q]->(n)
    RETURN n,
           labels(n) AS _labels,
           collect(DISTINCT {type: type(r), target_id: m.id, target_label: m.label}) AS outgoing,
           collect(DISTINCT {type: type(q), source_id: p.id, source_label: p.label}) AS incoming
    """
    async with _neo4j.driver.session(database=_neo4j._config.database) as session:
        res = await session.run(cypher, id=entity_id)
        rec = await res.single()
    if not rec or rec["n"] is None:
        raise HTTPException(404, "Entity not found")
    props = _serialise_props(dict(rec["n"].items()) if hasattr(rec["n"], 'items') else {})
    lbls = rec.get("_labels", [])
    etype = next((l for l in lbls if l not in ("Entity", "Node")), lbls[0] if lbls else "Unknown")
    props["entity_type"] = etype
    props["labels"] = lbls
    # Filter out null entries from optional matches
    props["outgoing"] = [r for r in rec["outgoing"] if r.get("type") is not None]
    props["incoming"] = [r for r in rec["incoming"] if r.get("type") is not None]
    return props


@router.get("/entities/{entity_id}/subgraph")
async def get_entity_subgraph(
    entity_id: str,
    depth: int = Query(default=2, le=4),
) -> dict[str, Any]:
    """Get the local subgraph around an entity for visualisation."""
    if _neo4j is None:
        raise HTTPException(503, "KG connector not initialised")
    # Direct Cypher avoids reliance on connector _label
    cypher = f"""
    MATCH (center {{id: $id}})
    CALL apoc.path.subgraphAll(center, {{maxLevel: $depth}}) YIELD nodes, relationships
    RETURN nodes, relationships
    """
    try:
        async with _neo4j.driver.session(database=_neo4j._config.database) as session:
            res = await session.run(cypher, id=entity_id, depth=depth)
            rec = await res.single()
        if rec:
            nodes_list = []
            for n in rec["nodes"]:
                props = dict(n.items()) if hasattr(n, 'items') else {}
                lbls = list(n.labels) if hasattr(n, 'labels') else []
                etype = next((l for l in lbls if l not in ("Entity", "Node")), lbls[0] if lbls else "Unknown")
                nodes_list.append({"id": props.get("id", ""), "label": props.get("label", ""), "type": etype})
            edges_list = []
            for r in rec["relationships"]:
                src = r.start_node
                tgt = r.end_node
                src_props = dict(src.items()) if hasattr(src, 'items') else {}
                tgt_props = dict(tgt.items()) if hasattr(tgt, 'items') else {}
                edges_list.append({
                    "source": src_props.get("id", ""),
                    "target": tgt_props.get("id", ""),
                    "label": r.type,
                })
            return {"nodes": nodes_list, "edges": edges_list}
    except Exception:
        logger.debug("subgraph.apoc_unavailable, falling back to variable-length path")

    # Fallback: simple variable-length path (no APOC)
    cypher_fb = f"""
    MATCH path = (center {{id: $id}})-[*1..{depth}]-(other)
    UNWIND nodes(path) AS n
    UNWIND relationships(path) AS r
    WITH DISTINCT n, labels(n) AS _nlabels, r,
         type(r) AS rtype, startNode(r) AS sn, endNode(r) AS en
    RETURN collect(DISTINCT {{id: n.id, label: n.label, type: head([l IN _nlabels WHERE NOT l IN ['Entity','Node']])}}) AS nodes,
           collect(DISTINCT {{source: sn.id, target: en.id, label: rtype}}) AS edges
    """
    async with _neo4j.driver.session(database=_neo4j._config.database) as session:
        res = await session.run(cypher_fb, id=entity_id)
        rec = await res.single()
    if not rec:
        return {"nodes": [], "edges": []}
    return {"nodes": rec["nodes"], "edges": rec["edges"]}


@router.get("/relations")
async def list_relation_types() -> list[dict[str, Any]]:
    """List all relation types in the KG with counts."""
    if _neo4j is None:
        raise HTTPException(503, "KG connector not initialised")
    cypher = "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"
    async with _neo4j.driver.session(database=_neo4j._config.database) as session:
        res = await session.run(cypher)
        rows = await res.data()
    return rows


@router.get("/stats")
async def get_kg_stats() -> dict[str, Any]:
    """Aggregate KG statistics — node counts, edge counts, type distribution."""
    if _neo4j is None:
        raise HTTPException(503, "KG connector not initialised")
    stats: dict[str, Any] = {}
    # node type counts (by Neo4j label)
    cy_nodes = """
    CALL db.labels() YIELD label
    CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) AS count', {}) YIELD value
    RETURN label AS type, value.count AS count
    ORDER BY count DESC
    """
    try:
        async with _neo4j.driver.session(database=_neo4j._config.database) as session:
            res = await session.run(cy_nodes)
            stats["node_types"] = await res.data()
    except Exception:
        # Fallback without APOC
        cy_fallback = "MATCH (n) UNWIND labels(n) AS lbl RETURN lbl AS type, count(*) AS count ORDER BY count DESC"
        async with _neo4j.driver.session(database=_neo4j._config.database) as session:
            res = await session.run(cy_fallback)
            stats["node_types"] = await res.data()
    # relation counts
    cy_rels = "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count"
    async with _neo4j.driver.session(database=_neo4j._config.database) as session:
        res2 = await session.run(cy_rels)
        stats["relation_types"] = await res2.data()
    return stats


# ===================================================================
# Law graph browsing
# ===================================================================


@router.get("/laws")
async def list_laws() -> list[dict[str, Any]]:
    """List all laws (Gesetzbücher) in the law graph."""
    if _neo4j is None:
        raise HTTPException(503, "KG connector not initialised")
    cy = "MATCH (g:Gesetzbuch) RETURN g, labels(g) AS _labels ORDER BY g.label"
    async with _neo4j.driver.session(database=_neo4j._config.database) as session:
        res = await session.run(cy)
        records = await res.data()
    laws = []
    for r in records:
        node = r["g"]
        props = _serialise_props(dict(node.items()) if hasattr(node, 'items') else {})
        props["labels"] = r.get("_labels", [])
        laws.append(props)
    return laws


@router.get("/laws/{law_id}/structure")
async def get_law_structure(law_id: str) -> dict[str, Any]:
    """Hierarchical structure of a law: Gesetzbuch → Paragraf."""
    if _neo4j is None:
        raise HTTPException(503, "KG connector not initialised")
    # The graph uses Paragraf -[teilVon]-> Gesetzbuch
    cy = """
    MATCH (g:Gesetzbuch {id: $law_id})
    OPTIONAL MATCH (p:Paragraf)-[:teilVon]->(g)
    WITH g, p ORDER BY p.label
    RETURN g, collect(p) AS paragraphs
    """
    async with _neo4j.driver.session(database=_neo4j._config.database) as session:
        res = await session.run(cy, law_id=law_id)
        rec = await res.single()
    if not rec or rec["g"] is None:
        raise HTTPException(404, "Law not found")
    law_props = _serialise_props(dict(rec["g"].items()) if hasattr(rec["g"], 'items') else {})
    paragraphs = []
    for p in rec.get("paragraphs", []):
        if p is not None:
            pprops = _serialise_props(dict(p.items()) if hasattr(p, 'items') else {})
            lbls = list(p.labels) if hasattr(p, 'labels') else []
            pprops["labels"] = lbls
            paragraphs.append(pprops)
    return {"law": law_props, "paragraphs": paragraphs}


@router.get("/laws/{law_id}/linked-entities")
async def get_law_linked_entities(law_id: str) -> list[dict[str, Any]]:
    """Domain entities linked to a specific law via its paragraphs."""
    if _neo4j is None:
        raise HTTPException(503, "KG connector not initialised")
    # Find Paragraf nodes belonging to this Gesetzbuch and their outgoing links
    # (excluding teilVon and referenziert which are structural)
    cy = """
    MATCH (p:Paragraf)-[:teilVon]->(g:Gesetzbuch {id: $law_id})
    MATCH (p)-[r]->(e)
    WHERE NOT type(r) IN ['teilVon', 'referenziert']
    RETURN e, labels(e) AS _elabels, p, labels(p) AS _plabels, type(r) AS rel_type
    LIMIT 100
    """
    async with _neo4j.driver.session(database=_neo4j._config.database) as session:
        res = await session.run(cy, law_id=law_id)
        rows = await res.data()
    results = []
    for r in rows:
        enode = r["e"]
        pnode = r["p"]
        eprops = _serialise_props(dict(enode.items()) if hasattr(enode, 'items') else {})
        eprops["labels"] = r.get("_elabels", [])
        pprops = _serialise_props(dict(pnode.items()) if hasattr(pnode, 'items') else {})
        pprops["labels"] = r.get("_plabels", [])
        results.append({"entity": eprops, "paragraph": pprops, "relation": r.get("rel_type", "")})
    return results


# ===================================================================
# Ontology browsing (Fuseki / SPARQL)
# ===================================================================


@router.get("/ontology/classes")
async def list_ontology_classes() -> list[dict[str, Any]]:
    """List all ontology classes with hierarchy."""
    if _fuseki is None:
        raise HTTPException(503, "Ontology connector not initialised")
    sparql = """
    SELECT ?class ?label ?parent WHERE {
      ?class a owl:Class .
      OPTIONAL { ?class rdfs:label ?label }
      OPTIONAL { ?class rdfs:subClassOf ?parent }
    }
    """
    rows = await _fuseki.query(sparql)
    return rows


@router.get("/ontology/classes/{class_uri:path}/properties")
async def get_class_properties(class_uri: str) -> list[dict[str, Any]]:
    """Get data + object properties for an ontology class."""
    if _fuseki is None:
        raise HTTPException(503, "Ontology connector not initialised")
    # leverage FusekiConnector helper
    props = await _fuseki.get_class_properties(class_uri)
    # convert dataclass to dict
    return [p.model_dump() if hasattr(p, "model_dump") else p.__dict__ for p in props]


@router.get("/ontology/tree")
async def get_ontology_tree() -> dict[str, Any]:
    """Full class hierarchy as a tree for D3.js / Cytoscape.js rendering."""
    if _fuseki is None:
        raise HTTPException(503, "Ontology connector not initialised")
    # fetch flat list then build parent→children map
    classes = await list_ontology_classes()
    tree: dict[str, Any] = {}
    for row in classes:
        cls = row.get("class")
        parent = row.get("parent")
        if parent is None:
            tree.setdefault(cls, {"children": []})
        else:
            tree.setdefault(parent, {"children": []})
            tree.setdefault(cls, {"children": []})
            tree[parent]["children"].append(cls)
    return tree
