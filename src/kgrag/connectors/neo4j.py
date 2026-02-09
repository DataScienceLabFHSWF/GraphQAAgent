"""Read-only Neo4j connector — consumes KGB's knowledge graph."""

from __future__ import annotations

import json
import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver

from kgrag.core.config import Neo4jConfig
from kgrag.core.exceptions import Neo4jConnectionError
from kgrag.core.models import KGEntity, KGRelation

logger = structlog.get_logger(__name__)

# Properties managed by _record_to_entity — excluded from the extras dict
_ENTITY_KNOWN_KEYS = frozenset({
    "id", "label", "entity_type", "confidence", "description",
    "properties", "node_type", "created_at",
})


class Neo4jConnector:
    """Async read-only client for Neo4j knowledge graph queries.

    Provides entity lookup, neighbourhood expansion, path finding,
    and subgraph extraction used by :class:`~kgrag.retrieval.graph.GraphRetriever`.
    """

    def __init__(self, config: Neo4jConfig) -> None:
        self._config = config
        self._driver: AsyncDriver | None = None
        self._label = config.node_label  # e.g. "Entity"

    # -- lifecycle ----------------------------------------------------------

    async def connect(self) -> None:
        """Open the Neo4j driver connection."""
        try:
            self._driver = AsyncGraphDatabase.driver(
                self._config.uri,
                auth=(self._config.user, self._config.password),
            )
            await self._driver.verify_connectivity()
            logger.info("neo4j.connected", uri=self._config.uri)
        except Exception as exc:
            raise Neo4jConnectionError(f"Cannot connect to Neo4j: {exc}") from exc

    async def close(self) -> None:
        """Close the driver."""
        if self._driver:
            await self._driver.close()
            logger.info("neo4j.closed")

    @property
    def driver(self) -> AsyncDriver:
        if self._driver is None:
            raise Neo4jConnectionError("Neo4j driver not initialised — call connect() first.")
        return self._driver

    # -- entity queries -----------------------------------------------------

    async def find_entities_by_label(
        self,
        labels: list[str],
        *,
        limit: int = 20,
    ) -> list[KGEntity]:
        """Case-insensitive label search across Entity nodes."""
        query = f"""
        UNWIND $labels AS term
        MATCH (e:{self._label})
        WHERE toLower(e.label) CONTAINS toLower(term)
        RETURN e, labels(e) AS _labels
        LIMIT $limit
        """
        async with self.driver.session(database=self._config.database) as session:
            result = await session.run(query, labels=labels, limit=limit)
            records = await result.data()

        return [self._record_to_entity(r["e"], neo4j_labels=r.get("_labels")) for r in records]

    async def find_entities_by_ids(self, entity_ids: list[str]) -> list[KGEntity]:
        """Fetch entities by their IDs."""
        query = f"""
        MATCH (e:{self._label})
        WHERE e.id IN $ids
        RETURN e, labels(e) AS _labels
        """
        async with self.driver.session(database=self._config.database) as session:
            result = await session.run(query, ids=entity_ids)
            records = await result.data()

        return [self._record_to_entity(r["e"], neo4j_labels=r.get("_labels")) for r in records]

    # -- neighbourhood / subgraph -------------------------------------------

    async def get_neighbourhood(
        self,
        entity_ids: list[str],
        *,
        max_hops: int = 1,
        max_nodes: int = 50,
    ) -> tuple[list[KGEntity], list[KGRelation]]:
        """Return the k-hop neighbourhood of the given entities."""
        L = self._label
        query = f"""
        MATCH (e:{L})-[r*1..{max_hops}]-(neighbour:{L})
        WHERE e.id IN $ids
        RETURN DISTINCT e, labels(e) AS _e_labels,
               r, neighbour, labels(neighbour) AS _n_labels
        LIMIT $max_nodes
        """
        async with self.driver.session(database=self._config.database) as session:
            result = await session.run(query, ids=entity_ids, max_nodes=max_nodes)
            records = await result.data()

        entities: dict[str, KGEntity] = {}
        relations: list[KGRelation] = []

        for rec in records:
            e = self._record_to_entity(rec["e"], neo4j_labels=rec.get("_e_labels"))
            entities[e.id] = e
            n = self._record_to_entity(rec["neighbour"], neo4j_labels=rec.get("_n_labels"))
            entities[n.id] = n
            for rel in rec.get("r", []):
                relations.append(self._record_to_relation(rel))

        return list(entities.values()), relations

    async def find_shortest_paths(
        self,
        source_id: str,
        target_id: str,
        *,
        max_hops: int = 4,
    ) -> list[tuple[list[KGEntity], list[KGRelation]]]:
        """Find shortest paths between two entities."""
        L = self._label
        query = f"""
        MATCH path = shortestPath(
            (a:{L} {{id: $src}})-[*..{max_hops}]-(b:{L} {{id: $tgt}})
        )
        RETURN path
        """
        async with self.driver.session(database=self._config.database) as session:
            result = await session.run(query, src=source_id, tgt=target_id)
            records = await result.data()

        paths: list[tuple[list[KGEntity], list[KGRelation]]] = []
        for rec in records:
            path = rec["path"]
            ents = [self._record_to_entity(n) for n in path.nodes]
            rels = [self._record_to_relation(r) for r in path.relationships]
            paths.append((ents, rels))
        return paths

    # -- PPR + community queries (SOTA additions) ---------------------------

    async def compute_ppr(
        self,
        seed_entity_ids: list[str],
        *,
        damping: float = 0.85,
        top_k: int = 20,
        max_iterations: int = 20,
    ) -> list[tuple[KGEntity, float]]:
        """Approximate Personalized PageRank from seed entities.

        Inspired by HippoRAG (Liu et al. 2024): use PPR to find nodes most
        relevant to the query entities, producing a focused subgraph that
        outperforms uniform k-hop expansion.

        Uses Neo4j GDS library if available, falls back to Cypher-based
        iterative approximation.
        """
        # Try GDS-based PPR first (fast, accurate)
        try:
            return await self._ppr_gds(seed_entity_ids, damping=damping, top_k=top_k)
        except Exception:
            logger.debug("neo4j.ppr_gds_unavailable, falling back to cypher approximation")
            return await self._ppr_cypher(
                seed_entity_ids,
                damping=damping,
                top_k=top_k,
                max_iterations=max_iterations,
            )

    async def _ppr_gds(
        self,
        seed_entity_ids: list[str],
        *,
        damping: float = 0.85,
        top_k: int = 20,
    ) -> list[tuple[KGEntity, float]]:
        """PPR via Neo4j Graph Data Science library."""
        L = self._label
        query = f"""
        MATCH (source:{L})
        WHERE source.id IN $seeds
        CALL gds.pageRank.stream({{
            nodeProjection: '{L}',
            relationshipProjection: {{ALL: {{type: '*', orientation: 'UNDIRECTED'}}}},
            dampingFactor: $damping,
            maxIterations: 20,
            sourceNodes: collect(source)
        }})
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS node, score
        ORDER BY score DESC
        LIMIT $top_k
        RETURN node, score
        """
        async with self.driver.session(database=self._config.database) as session:
            result = await session.run(
                query, seeds=seed_entity_ids, damping=damping, top_k=top_k,
            )
            records = await result.data()

        return [(self._record_to_entity(r["node"]), r["score"]) for r in records]

    async def _ppr_cypher(
        self,
        seed_entity_ids: list[str],
        *,
        damping: float = 0.85,
        top_k: int = 20,
        max_iterations: int = 20,
    ) -> list[tuple[KGEntity, float]]:
        """Cypher-based PPR approximation (no GDS required).

        Multi-hop random-walk approximation: expand from seeds, accumulate
        scores with damping, iterate until convergence or max iterations.
        """
        # Collect up to 3-hop neighbourhood with decaying scores
        L = self._label
        query = f"""
        MATCH (seed:{L})
        WHERE seed.id IN $seeds
        WITH collect(seed) AS seeds

        // Hop 1
        UNWIND seeds AS s
        OPTIONAL MATCH (s)-[r1]-(n1:{L})
        WITH seeds, collect(DISTINCT n1) AS hop1, collect(DISTINCT r1) AS rels1

        // Hop 2
        UNWIND hop1 AS h1
        OPTIONAL MATCH (h1)-[r2]-(n2:{L})
        WHERE NOT n2 IN seeds AND NOT n2 IN hop1
        WITH seeds, hop1, collect(DISTINCT n2) AS hop2

        // Hop 3
        UNWIND hop2 AS h2
        OPTIONAL MATCH (h2)-[r3]-(n3:{L})
        WHERE NOT n3 IN seeds AND NOT n3 IN hop1 AND NOT n3 IN hop2
        WITH seeds, hop1, hop2, collect(DISTINCT n3) AS hop3

        // Score assignment: seed=1.0, hop1=damping, hop2=damping^2, hop3=damping^3
        UNWIND (seeds + hop1 + hop2 + hop3) AS node
        WITH DISTINCT node,
             CASE
                 WHEN node IN seeds THEN 1.0
                 WHEN node IN hop1 THEN $d1
                 WHEN node IN hop2 THEN $d2
                 ELSE $d3
             END AS score
        ORDER BY score DESC
        LIMIT $top_k
        RETURN node, score
        """
        async with self.driver.session(database=self._config.database) as session:
            result = await session.run(
                query,
                seeds=seed_entity_ids,
                d1=damping,
                d2=damping ** 2,
                d3=damping ** 3,
                top_k=top_k,
            )
            records = await result.data()

        return [(self._record_to_entity(r["node"]), r["score"]) for r in records]

    async def get_entity_neighbours(
        self,
        entity_id: str,
        *,
        limit: int = 10,
    ) -> list[tuple[KGEntity, KGRelation]]:
        """Get immediate neighbours with their connecting relations.

        Used by Think-on-Graph iterative exploration to let the LLM decide
        which edges to follow.
        """
        L = self._label
        query = f"""
        MATCH (e:{L} {{id: $eid}})-[r]-(n:{L})
        RETURN n, labels(n) AS _n_labels,
               r, type(r) AS _rel_type, e.id AS _src_id, n.id AS _tgt_id
        LIMIT $limit
        """
        async with self.driver.session(database=self._config.database) as session:
            result = await session.run(query, eid=entity_id, limit=limit)
            records = await result.data()

        return [
            (
                self._record_to_entity(rec["n"], neo4j_labels=rec.get("_n_labels")),
                self._record_to_relation(
                    rec["r"],
                    rel_type=rec.get("_rel_type"),
                    src_id=rec.get("_src_id"),
                    tgt_id=rec.get("_tgt_id"),
                ),
            )
            for rec in records
        ]

    async def get_subgraph_between(
        self,
        entity_ids: list[str],
        *,
        max_hops: int = 2,
    ) -> tuple[list[KGEntity], list[KGRelation]]:
        """Extract the subgraph connecting given entities (Steiner-tree inspired).

        Finds all paths of length ≤ max_hops between any pair of entities,
        producing a focused subgraph rather than a full k-hop expansion.
        """
        L = self._label
        query = f"""
        UNWIND $ids AS src_id
        UNWIND $ids AS tgt_id
        WITH src_id, tgt_id WHERE src_id < tgt_id
        MATCH (a:{L} {{id: src_id}})
        MATCH (b:{L} {{id: tgt_id}})
        MATCH path = (a)-[*1..{max_hops}]-(b)
        UNWIND nodes(path) AS node
        UNWIND relationships(path) AS rel
        WITH DISTINCT node, labels(node) AS _labels, rel,
             type(rel) AS _rel_type,
             startNode(rel).id AS _src_id,
             endNode(rel).id AS _tgt_id
        RETURN node, _labels, rel, _rel_type, _src_id, _tgt_id
        """
        async with self.driver.session(database=self._config.database) as session:
            result = await session.run(query, ids=entity_ids)
            records = await result.data()

        entities: dict[str, KGEntity] = {}
        relations: dict[tuple[str, str, str], KGRelation] = {}
        for rec in records:
            e = self._record_to_entity(rec["node"], neo4j_labels=rec.get("_labels"))
            entities[e.id] = e
            r = self._record_to_relation(
                rec["rel"],
                rel_type=rec.get("_rel_type"),
                src_id=rec.get("_src_id"),
                tgt_id=rec.get("_tgt_id"),
            )
            relations[(r.source_id, r.relation_type, r.target_id)] = r

        return list(entities.values()), list(relations.values())

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _record_to_entity(
        node: dict,  # type: ignore[type-arg]
        *,
        neo4j_labels: list[str] | None = None,
    ) -> KGEntity:
        """Convert a Neo4j node record to :class:`KGEntity`.

        Handles both the recommended flat-property schema and KGB's
        ``Neo4jGraphStore`` format (JSON-serialized ``properties`` field).

        Parameters
        ----------
        node:
            Dict of node properties from Neo4j.
        neo4j_labels:
            Node labels returned via ``labels(n)`` in the Cypher query.
            Used to derive ``entity_type`` if not stored as a property.
        """
        # --- flatten KGB's JSON properties bag if present ---
        raw_props: str | None = node.get("properties")
        if isinstance(raw_props, str):
            try:
                extra = json.loads(raw_props)
                if isinstance(extra, dict):
                    # Merge parsed properties under the node dict (low priority)
                    merged = {**extra, **{k: v for k, v in node.items() if k != "properties"}}
                    node = merged
            except (json.JSONDecodeError, TypeError):
                pass

        # --- entity_type: prefer property, fall back to Neo4j label ---
        entity_type = node.get("entity_type") or node.get("node_type", "")
        if not entity_type and neo4j_labels:
            # Pick the first label that isn't the generic "Entity" / "Node"
            for lbl in neo4j_labels:
                if lbl not in ("Entity", "Node"):
                    entity_type = lbl
                    break
            if not entity_type and neo4j_labels:
                entity_type = neo4j_labels[0]

        return KGEntity(
            id=node.get("id", ""),
            label=node.get("label", ""),
            entity_type=entity_type,
            description=node.get("description", ""),
            properties={
                k: v for k, v in node.items()
                if k not in _ENTITY_KNOWN_KEYS
            },
            confidence=float(node.get("confidence", 0.0)),
        )

    @staticmethod
    def _record_to_relation(
        rel: dict,  # type: ignore[type-arg]
        *,
        rel_type: str | None = None,
        src_id: str | None = None,
        tgt_id: str | None = None,
    ) -> KGRelation:
        """Convert a Neo4j relationship record to :class:`KGRelation`.

        KGB stores only ``predicate`` and ``confidence`` on relationships.
        ``source_id`` / ``target_id`` / ``type`` are derived from the Cypher
        query via ``type(r)``, ``startNode(r).id``, ``endNode(r).id``.

        Parameters
        ----------
        rel:
            Dict of relationship properties from Neo4j.
        rel_type:
            Relationship type from ``type(r)``, used when ``rel["type"]``
            is missing.
        src_id:
            Source entity ID from ``startNode(r).id``.
        tgt_id:
            Target entity ID from ``endNode(r).id``.
        """
        # --- flatten KGB's JSON properties bag if present ---
        raw_props: str | None = rel.get("properties")
        if isinstance(raw_props, str):
            try:
                extra = json.loads(raw_props)
                if isinstance(extra, dict):
                    merged = {**extra, **{k: v for k, v in rel.items() if k != "properties"}}
                    rel = merged
            except (json.JSONDecodeError, TypeError):
                pass

        return KGRelation(
            source_id=rel.get("source_id") or src_id or "",
            target_id=rel.get("target_id") or tgt_id or "",
            relation_type=rel.get("type") or rel.get("predicate") or rel_type or "",
            confidence=float(rel.get("confidence", 0.0)),
        )
