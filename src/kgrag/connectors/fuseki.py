"""Read-only Fuseki connector — reads ontology via SPARQL."""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from kgrag.core.config import FusekiConfig
from kgrag.core.exceptions import FusekiConnectionError
from kgrag.core.models import OntologyClass, OntologyProperty

logger = structlog.get_logger(__name__)

# Standard SPARQL prefixes — auto-prepended to every query
_SPARQL_PREFIXES = """\
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""


class FusekiConnector:
    """Async client for the Fuseki SPARQL endpoint providing ontology access.

    Used by :class:`~kgrag.retrieval.ontology.OntologyRetriever` to expand
    queries with class hierarchies, synonyms, and expected relations.
    """

    def __init__(self, config: FusekiConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None

    # -- lifecycle ----------------------------------------------------------

    async def connect(self) -> None:
        """Open HTTP client and verify the Fuseki endpoint."""
        auth = None
        if self._config.user and self._config.password:
            auth = httpx.BasicAuth(self._config.user, self._config.password)
        self._client = httpx.AsyncClient(
            base_url=self._config.url,
            timeout=30.0,
            auth=auth,
        )
        try:
            resp = await self._client.get(f"/{self._config.dataset}")
            resp.raise_for_status()
            logger.info("fuseki.connected", url=self._config.url, dataset=self._config.dataset)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.warning(
                    "fuseki.dataset_not_found", 
                    url=self._config.url, 
                    dataset=self._config.dataset,
                    message="Dataset not found - ontology features will be limited"
                )
                # Don't raise exception for missing dataset, just warn
                return
            raise FusekiConnectionError(f"Cannot connect to Fuseki: {exc}") from exc
        except Exception as exc:
            raise FusekiConnectionError(f"Cannot connect to Fuseki: {exc}") from exc

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            logger.info("fuseki.closed")

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise FusekiConnectionError("Fuseki client not initialised — call connect() first.")
        return self._client

    # -- SPARQL queries -----------------------------------------------------

    async def query(self, sparql: str) -> list[dict[str, Any]]:
        """Execute a SPARQL SELECT query and return bindings as dicts."""
        if self._client is None:
            logger.warning("fuseki.not_connected", message="Fuseki client not available, returning empty results")
            return []
            
        try:
            # Auto-prepend standard prefixes if not already present
            if not sparql.strip().upper().startswith("PREFIX"):
                sparql = _SPARQL_PREFIXES + sparql
            resp = await self._client.post(
                f"/{self._config.dataset}/sparql",
                data={"query": sparql},
                headers={"Accept": "application/sparql-results+json"},
            )
            resp.raise_for_status()
            data = resp.json()
            bindings = data.get("results", {}).get("bindings", [])
            return [
                {k: v.get("value", "") for k, v in binding.items()}
                for binding in bindings
            ]
        except Exception as exc:
            logger.warning("fuseki.query_failed", error=str(exc), message="SPARQL query failed, returning empty results")
            return []

    async def get_subclasses(self, class_uri: str) -> list[OntologyClass]:
        """Get all subclasses (transitive) of a given ontology class."""
        sparql = f"""
        SELECT ?sub ?label WHERE {{
            ?sub rdfs:subClassOf* <{class_uri}> .
            ?sub rdfs:label ?label .
        }}
        """
        rows = await self.query(sparql)
        return [
            OntologyClass(uri=r["sub"], label=r["label"])
            for r in rows
        ]

    async def get_synonyms(self, class_uri: str) -> list[str]:
        """Get skos:altLabel synonyms for a class."""
        sparql = f"""
        SELECT ?altLabel WHERE {{
            <{class_uri}> skos:altLabel ?altLabel .
        }}
        """
        rows = await self.query(sparql)
        return [r["altLabel"] for r in rows]

    async def get_class_properties(self, class_uri: str) -> list[OntologyProperty]:
        """Get properties whose domain is the given class."""
        sparql = f"""
        SELECT ?prop ?range ?propLabel WHERE {{
            ?prop rdfs:domain <{class_uri}> .
            ?prop rdfs:range ?range .
            ?prop rdfs:label ?propLabel .
        }}
        """
        rows = await self.query(sparql)
        return [
            OntologyProperty(
                uri=r["prop"],
                label=r["propLabel"],
                domain_uri=class_uri,
                range_uri=r["range"],
            )
            for r in rows
        ]

    async def get_class_by_label(self, label: str) -> OntologyClass | None:
        """Look up an ontology class by rdfs:label (case-insensitive)."""
        sparql = f"""
        SELECT ?cls ?label WHERE {{
            ?cls a owl:Class .
            ?cls rdfs:label ?label .
            FILTER(LCASE(STR(?label)) = LCASE("{label}"))
        }}
        LIMIT 1
        """
        rows = await self.query(sparql)
        if not rows:
            return None
        return OntologyClass(uri=rows[0]["cls"], label=rows[0]["label"])
