"""Neosemantics (n10s) integration helpers.

n10s is a Neo4j plugin for RDF import/export, SHACL validation, and
ontology management.  This module provides helpers for:

1. **RDF snapshot export** — export the current KG state to Turtle for
   archival and diffing between versions.
2. **SHACL validation** — validate proposed changes against the domain
   ontology's SHACL shapes before applying them.
3. **Ontology sync** — keep the Neo4j ontology labels in sync with
   the Fuseki TBox.

Prerequisites
-------------
- n10s plugin installed in Neo4j (``neosemantics-*.jar`` in plugins/).
- Graph config initialised: ``CALL n10s.graphconfig.init()``.
- Constraint created: ``CREATE CONSTRAINT n10s_unique_uri FOR (r:Resource)
  REQUIRE r.uri IS UNIQUE``.
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class N10sIntegration:
    """Helpers for neosemantics (n10s) operations on Neo4j.

    Methods degrade gracefully when the n10s plugin is not installed —
    ``check_n10s_available`` returns ``False`` and all other methods log
    a warning and return empty/default values.
    """

    def __init__(self, neo4j_connector: Any) -> None:
        """
        Parameters
        ----------
        neo4j_connector:
            :class:`kgrag.connectors.neo4j.Neo4jConnector` instance.
        """
        self._neo4j = neo4j_connector
        self._available: bool | None = None  # cached probe result

    def _db(self) -> str:
        return self._neo4j._config.database

    async def check_n10s_available(self) -> bool:
        """Check whether n10s is installed and configured.

        Runs ``SHOW PROCEDURES`` and looks for procedures starting with
        ``n10s``.  The result is cached for the lifetime of this instance.
        """
        if self._available is not None:
            return self._available

        try:
            async with self._neo4j.driver.session(database=self._db()) as session:
                result = await session.run(
                    "SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'n10s' RETURN count(*) AS cnt"
                )
                record = await result.single()
                self._available = (record["cnt"] or 0) > 0
        except Exception as exc:
            logger.debug("n10s.check_failed", error=str(exc))
            self._available = False

        logger.info("n10s.available", available=self._available)
        return self._available

    async def init_graph_config(
        self,
        *,
        handle_vocab_uris: str = "SHORTEN",
        handle_multival: str = "OVERWRITE",
        handle_rdf_types: str = "LABELS",
    ) -> None:
        """Initialise or update the n10s graph configuration."""
        if not await self.check_n10s_available():
            logger.warning("n10s.init_graph_config.skipped", reason="n10s not available")
            return

        query = (
            "CALL n10s.graphconfig.init({"
            "  handleVocabUris: $hvu, "
            "  handleMultival: $hmv, "
            "  handleRDFTypes: $hrt"
            "})"
        )
        async with self._neo4j.driver.session(database=self._db()) as session:
            await session.run(
                query,
                hvu=handle_vocab_uris,
                hmv=handle_multival,
                hrt=handle_rdf_types,
            )
        logger.info("n10s.graph_config_initialised")

    async def export_rdf_snapshot(
        self,
        *,
        cypher_query: str | None = None,
        format: str = "Turtle",
    ) -> str:
        """Export (part of) the KG as RDF.

        Parameters
        ----------
        cypher_query:
            Optional Cypher to select nodes/edges for export.
            If ``None``, exports the entire graph.
        format:
            RDF serialisation format (``Turtle``, ``JSON-LD``, ``RDF/XML``).

        Returns
        -------
        str
            Serialised RDF data.
        """
        if not await self.check_n10s_available():
            logger.warning("n10s.export_rdf.skipped", reason="n10s not available")
            return ""

        cypher = cypher_query or "MATCH (n)-[r]->(m) RETURN *"
        query = "CALL n10s.rdf.export.cypher($cypher, $fmt) YIELD rdfTriple RETURN rdfTriple"

        triples: list[str] = []
        async with self._neo4j.driver.session(database=self._db()) as session:
            result = await session.run(query, cypher=cypher, fmt=format)
            async for record in result:
                triples.append(str(record["rdfTriple"]))

        snapshot = "\n".join(triples)
        logger.info("n10s.export_rdf", triples=len(triples), format=format)
        return snapshot

    async def validate_with_shacl(
        self,
        rdf_data: str,
        *,
        format: str = "Turtle",
    ) -> list[dict[str, Any]]:
        """Validate RDF data against SHACL shapes in the graph.

        Parameters
        ----------
        rdf_data:
            RDF triples to validate (e.g. a proposed change as Turtle).
        format:
            Serialisation format of ``rdf_data``.

        Returns
        -------
        list[dict]
            Validation results: each dict has ``focusNode``, ``resultPath``,
            ``resultMessage``, ``resultSeverity``.
        """
        if not await self.check_n10s_available():
            logger.warning("n10s.shacl_validate.skipped", reason="n10s not available")
            return []

        query = (
            "CALL n10s.validation.shacl.validate($rdf, $fmt) "
            "YIELD focusNode, resultPath, resultMessage, resultSeverity "
            "RETURN focusNode, resultPath, resultMessage, resultSeverity"
        )
        results: list[dict[str, Any]] = []
        async with self._neo4j.driver.session(database=self._db()) as session:
            result = await session.run(query, rdf=rdf_data, fmt=format)
            async for record in result:
                results.append(
                    {
                        "focusNode": record["focusNode"],
                        "resultPath": record["resultPath"],
                        "resultMessage": record["resultMessage"],
                        "resultSeverity": record["resultSeverity"],
                    }
                )

        logger.info("n10s.shacl_validated", violations=len(results))
        return results

    async def import_ontology(
        self,
        source: str,
        *,
        format: str = "Turtle",
    ) -> dict[str, Any]:
        """Import an ontology into Neo4j via n10s.

        Parameters
        ----------
        source:
            URL or inline RDF string.
        format:
            RDF format.

        Returns
        -------
        dict
            Import statistics: ``triplesLoaded``, ``triplesParsed``,
            ``namespaces``, ``terminationStatus``.
        """
        if not await self.check_n10s_available():
            logger.warning("n10s.import_ontology.skipped", reason="n10s not available")
            return {"terminationStatus": "skipped", "triplesLoaded": 0}

        query = (
            "CALL n10s.onto.import.fetch($src, $fmt) "
            "YIELD terminationStatus, triplesLoaded, triplesParsed, namespaces "
            "RETURN terminationStatus, triplesLoaded, triplesParsed, namespaces"
        )
        async with self._neo4j.driver.session(database=self._db()) as session:
            result = await session.run(query, src=source, fmt=format)
            record = await result.single()

        stats: dict[str, Any] = {}
        if record:
            stats = {
                "terminationStatus": record["terminationStatus"],
                "triplesLoaded": record["triplesLoaded"],
                "triplesParsed": record["triplesParsed"],
                "namespaces": record["namespaces"],
            }
        logger.info("n10s.ontology_imported", **stats)
        return stats

    async def diff_snapshots(
        self,
        snapshot_a: str,
        snapshot_b: str,
        *,
        format: str = "Turtle",
    ) -> dict[str, Any]:
        """Compute the diff between two RDF snapshots.

        Uses ``rdflib`` to parse both snapshots and compute set differences.
        Returns added, removed triples.
        """
        try:
            from rdflib import Graph as RDFGraph
        except ImportError:
            logger.warning("n10s.diff_snapshots.skipped", reason="rdflib not installed")
            return {"added": [], "removed": [], "error": "rdflib not installed"}

        fmt_map = {"Turtle": "turtle", "JSON-LD": "json-ld", "RDF/XML": "xml"}
        rdf_fmt = fmt_map.get(format, "turtle")

        g_a = RDFGraph()
        g_a.parse(data=snapshot_a, format=rdf_fmt)

        g_b = RDFGraph()
        g_b.parse(data=snapshot_b, format=rdf_fmt)

        added = list(g_b - g_a)
        removed = list(g_a - g_b)

        logger.info("n10s.diff_computed", added=len(added), removed=len(removed))
        return {
            "added": [
                {"subject": str(s), "predicate": str(p), "object": str(o)}
                for s, p, o in added
            ],
            "removed": [
                {"subject": str(s), "predicate": str(p), "object": str(o)}
                for s, p, o in removed
            ],
        }
