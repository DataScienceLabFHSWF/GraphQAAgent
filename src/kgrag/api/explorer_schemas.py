"""KG exploration request/response models.

These are used by the FastAPI explorer routes to provide typed
`response_model` metadata for OpenAPI and to keep client/server contracts
clear.  The definitions mirror the plan in
`planning/GRAPHQA_AGENT_API_PLAN.md`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EntityInfo(BaseModel):
    """A Knowledge Graph entity."""

    id: str
    label: str
    entity_type: str = ""
    confidence: float = 0.0
    description: str = ""
    properties: dict[str, str] = Field(default_factory=dict)


class RelationEdge(BaseModel):
    """A single relation between two entities."""

    source_id: str
    source_label: str = ""
    target_id: str
    target_label: str = ""
    predicate: str
    confidence: float = 0.0


class EntityDetail(EntityInfo):
    """Entity with its direct neighbors."""

    neighbors: list[RelationEdge] = Field(default_factory=list)


class SubgraphResponse(BaseModel):
    """vis.js-compatible subgraph for visualization."""

    nodes: list[dict] = Field(
        default_factory=list,
        description="List of {id, label, group, ...} for vis.js",
    )
    edges: list[dict] = Field(
        default_factory=list,
        description="List of {from, to, label, ...} for vis.js",
    )


class RelationType(BaseModel):
    """A relation type with count."""

    predicate: str
    label: str = ""
    count: int = 0


class KGStats(BaseModel):
    """Knowledge Graph statistics."""

    entity_count: int = 0
    relation_count: int = 0
    entity_types: dict[str, int] = Field(default_factory=dict)
    relation_types: dict[str, int] = Field(default_factory=dict)
    law_count: int = 0


class LawInfo(BaseModel):
    """A law or regulation in the KG."""

    id: str
    title: str
    abbreviation: str = ""
    section_count: int = 0


class LawStructure(BaseModel):
    """Hierarchical structure of a law."""

    id: str
    title: str
    children: list[LawStructure] = Field(default_factory=list)


class OntologyClassInfo(BaseModel):
    """An OWL class from the ontology."""

    uri: str
    label: str
    description: str = ""
    parent_uri: str | None = None
    instance_count: int = 0


class OntologyTree(BaseModel):
    """Hierarchical ontology tree node."""

    uri: str
    label: str
    children: list[OntologyTree] = Field(default_factory=list)
    instance_count: int = 0


# fix forward refs
OntologyTree.model_rebuild()
LawStructure.model_rebuild()
EntityDetail.model_rebuild()
