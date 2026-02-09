# C3: Graph Interface Specification — KnowledgeGraphBuilder ↔ KG-RAG Agent

## 1. Purpose

This document defines the **exact Neo4j graph schema, Qdrant payload schema,
and Fuseki ontology structure** that the KG-RAG Agent expects. It also documents
the current KnowledgeGraphBuilder (KGB) output format and the **gap analysis**
between the two, with concrete recommendations for a compatible interface.

---

## 2. What KGB Currently Produces

### 2.1 Neo4j Nodes (KGB)

KGB uses the **entity_type as the Neo4j node label** (e.g., `:NuclearFacility`,
`:Action`, `:Component`).

#### `Neo4jGraphStore.add_node()` — Cypher:
```cypher
MERGE (n:{entity_type} {id: $id})
SET n.label = $label,
    n.node_type = $node_type,
    n.properties = $properties,      -- ⚠ JSON-serialized string!
    n.created_at = datetime()
```

#### `SimpleKGAssembler._create_node()` — Cypher:
```cypher
MERGE (n:{entity_type} {id: $entity_id})
SET n.label = $label,
    n.description = $description,
    n.confidence = $confidence,
    n.merged_count = $merged_count,
    n.evidence_count = $evidence_count,
    n.sources = $sources,
    n.created_at = timestamp()
```

#### Effective KGB Node Properties:
| Property         | Type           | Source                |
|-----------------|----------------|-----------------------|
| `id`            | `string`       | Entity UUID           |
| `label`         | `string`       | Human-readable name   |
| `description`   | `string`       | LLM-generated desc    |
| `confidence`    | `float`        | Extraction confidence |
| `merged_count`  | `int`          | Dedup merge count     |
| `evidence_count`| `int`          | # supporting chunks   |
| `sources`       | `list[string]` | Source document IDs   |
| `node_type`     | `string`       | (Neo4jGraphStore only)|
| `properties`    | `string(JSON)` | (Neo4jGraphStore only)|
| `created_at`    | `datetime`     | Timestamp             |


### 2.2 Neo4j Relationships (KGB)

KGB uses the **predicate/edge_type as the Neo4j relationship type**
(e.g., `-[:INVOLVES]->`, `-[:locatedIn]->`).

#### `Neo4jGraphStore.add_edge()` — Cypher:
```cypher
MATCH (s {id: $source_id}), (t {id: $target_id})
CREATE (s)-[r:{edge_type} {id: $id, properties: $properties, created_at: datetime()}]->(t)
```

#### `SimpleKGAssembler._create_relationship()` — Cypher:
```cypher
MATCH (source {id: $source_id}), (target {id: $target_id})
MERGE (source)-[r:{predicate}]->(target)
SET r.confidence = $confidence,
    r.evidence_count = $evidence_count,
    r.created_at = timestamp()
```

#### `Neo4jStore.add_edge()` (graph.py) — Cypher:
```cypher
MATCH (source {id: $source_id}), (target {id: $target_id})
MERGE (source)-[r:{relation_type}]->(target)
SET r += $properties
```

#### Effective KGB Relationship Properties:
| Property          | Type           | Source                      |
|------------------|----------------|-----------------------------|
| `id`             | `string`       | Neo4jGraphStore only        |
| `confidence`     | `float`        | Extraction confidence       |
| `evidence_count` | `int`          | SimpleKGAssembler only      |
| `properties`     | `string(JSON)` | Neo4jGraphStore only (JSON) |
| `created_at`     | `datetime`     | Timestamp                   |

### 2.3 KGB Indexes:
```cypher
CREATE INDEX id_index IF NOT EXISTS FOR (n:Node) ON (n.id)
CREATE INDEX label_index IF NOT EXISTS FOR (n:Node) ON (n.label)
CREATE INDEX confidence_index IF NOT EXISTS FOR (n:Node) ON (n.confidence)
-- OR per entity_type:
CREATE INDEX IF NOT EXISTS FOR (n:{entity_type}) ON (n.id)
```

### 2.4 KGB Data Models (Python):

```python
# kgbuilder.storage.protocol
@dataclass
class Node:
    id: str
    label: str
    node_type: str                           # e.g. "NuclearFacility"
    properties: dict[str, Any] = {}          # confidence, description, ...
    metadata: dict[str, Any] = {}            # created_at, merged_from, ...

@dataclass
class Edge:
    id: str
    source_id: str
    target_id: str
    edge_type: str                           # e.g. "INVOLVES"
    properties: dict[str, Any] = {}          # confidence, evidence_count, ...
    metadata: dict[str, Any] = {}            # created_at, ...

# kgbuilder.core.models
@dataclass
class ExtractedRelation:
    id: str
    source_entity_id: str
    target_entity_id: str
    predicate: str                           # Ontology relation URI or name
    properties: dict[str, Any] = {}
    confidence: float = 0.0
    evidence: list[Evidence] = []
```

---

## 3. What KG-RAG Agent Expects

### 3.1 Neo4j Nodes (KG-RAG Required)

**All Cypher queries assume a single node label `:Entity`.** Every node must
carry this label regardless of its ontology class.

#### Required Node Properties:
| Property         | Type           | Required | Usage                                   |
|-----------------|----------------|----------|-----------------------------------------|
| `id`            | `string`       | **Yes**  | Primary key, every WHERE/MATCH uses it  |
| `label`         | `string`       | **Yes**  | Text search (CONTAINS), fuzzy matching  |
| `entity_type`   | `string`       | Expected | Ontology class URI; maps to Fuseki types|
| `confidence`    | `float`        | Optional | Default `0.0`; used for scoring         |
| `source_doc_ids`| `list[string]` | Optional | Default `[]`; provenance tracking       |
| `*` (any)       | `Any`          | Optional | Caught by `dict(record)` sweep          |

Additional recommended properties for richer QA:
| Property         | Type           | From KGB          | Benefit                        |
|-----------------|----------------|-------------------|--------------------------------|
| `description`   | `string`       | Entity description| Better context for LLM prompts |
| `aliases`       | `list[string]` | Entity aliases    | Better entity linking recall   |
| `evidence_count`| `int`          | Evidence count    | Evidence quality heuristic     |

#### Node Mapping (KGB → KG-RAG):
```
KGB Node.node_type    →  KG-RAG Entity.entity_type  (as property, NOT label)
KGB Node.label        →  KG-RAG Entity.label
KGB Node.id           →  KG-RAG Entity.id
KGB Node.properties.confidence →  KG-RAG Entity.confidence
KGB Node.properties.sources    →  KG-RAG Entity.source_doc_ids
```

### 3.2 Neo4j Relationships (KG-RAG Required)

KG-RAG is **relationship-type-agnostic** — it uses `[r]`, `[r*1..N]` patterns.
The relationship type itself is read dynamically via `type(r)`.

#### Required Relationship Properties:
| Property         | Type     | Required | Usage                                     |
|-----------------|----------|----------|-------------------------------------------|
| `id`            | `string` | Expected | Relationship identifier; default `""`      |
| `source_id`     | `string` | Expected | **Redundant** source entity ID on the rel  |
| `target_id`     | `string` | Expected | **Redundant** target entity ID on the rel  |
| `type`          | `string` | Expected | Relationship type (should equal Neo4j type)|
| `confidence`    | `float`  | Optional | Default `0.0`; used for path ranking       |
| `evidence_text` | `string` | Optional | Default `""`; shown in reasoning chains    |

> **Critical**: `source_id` and `target_id` must be stored as **properties on
> the relationship** in addition to being Neo4j's native start/end node IDs.
> Current `_record_to_relation()` reads them from the property dict.

#### Relationship Mapping (KGB → KG-RAG):
```
KGB Edge.edge_type    →  Neo4j relationship type (already correct)
KGB Edge.edge_type    →  KG-RAG Relation.type     (as property too)
KGB Edge.source_id    →  KG-RAG Relation.source_id (as property)
KGB Edge.target_id    →  KG-RAG Relation.target_id (as property)
KGB Edge.properties.confidence →  KG-RAG Relation.confidence
KGB Edge.properties.evidence_sample → KG-RAG Relation.evidence_text
```

### 3.3 Neo4j Indexes Required:
```cypher
-- REQUIRED for performance (every query uses Entity.id):
CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id);

-- RECOMMENDED for text search (entity_linker fuzzy matching):
CREATE FULLTEXT INDEX entity_label IF NOT EXISTS
  FOR (e:Entity) ON EACH [e.label];

-- OPTIONAL (used by PPR if GDS is installed):
-- Neo4j GDS plugin with pageRank support
```

### 3.4 Qdrant Payload Schema (KG-RAG Required):
```json
{
    "document_id": "string",
    "text": "string (chunk text)",
    "entity_ids": ["string (Neo4j entity IDs referenced by this chunk)"]
}
```

> The `entity_ids` field is used by `EntityLinker._embedding_fallback()` to
> cross-reference Qdrant chunks → Neo4j entities. Without it, embedding-based
> entity linking silently skips the KG resolution step.

### 3.5 Fuseki/SPARQL Ontology Required:
| Vocabulary       | Usage                          | Required |
|-----------------|--------------------------------|----------|
| `owl:Class`      | All classes typed as owl:Class | **Yes**  |
| `rdfs:label`     | On every class and property    | **Yes**  |
| `rdfs:subClassOf`| Class hierarchy (transitive)   | **Yes**  |
| `skos:altLabel`  | Synonym labels on classes      | Expected |
| `rdfs:domain`    | Property domain class          | Expected |
| `rdfs:range`     | Property range class/datatype  | Expected |
| `owl:ObjectProperty` | Object property typing     | Optional |

---

## 4. Gap Analysis

### 4.1 Critical Mismatches

| # | Issue | KGB Current | KG-RAG Expects | Severity |
|---|-------|-------------|----------------|----------|
| 1 | **Node label** | `:NuclearFacility`, `:Action`, etc. (uses `entity_type` as label) | All nodes labeled `:Entity` | **CRITICAL** — every Cypher query fails |
| 2 | **Properties storage** | `Neo4jGraphStore` serializes properties as JSON string in a single `properties` field | Flat properties directly on node (`e.label`, `e.confidence`, etc.) | **CRITICAL** — all property reads return `None` |
| 3 | **Missing `entity_type` property** | Stored as Neo4j label, not as a node property | `e.entity_type` reads from node property | **HIGH** — ontology type mapping breaks |
| 4 | **Missing `source_doc_ids`** | `sources` (different name) | `source_doc_ids` | **MEDIUM** — provenance tracking fails |
| 5 | **Relationship: missing `source_id`/`target_id` props** | Not stored as relationship properties | Reads `rel.source_id` and `rel.target_id` from relationship properties | **CRITICAL** — all relations have empty endpoints |
| 6 | **Relationship: missing `type` property** | Relationship type is the Neo4j type, not a stored property | Reads `rel.type` from relationship properties | **HIGH** — relation_type always empty |
| 7 | **Relationship: missing `evidence_text`** | `evidence_sample` (different name) or missing | `rel.evidence_text` | **LOW** — empty strings, no reasoning evidence |
| 8 | **Qdrant `entity_ids`** | Not populated by KGB | Required for embedding-based entity linking fallback | **MEDIUM** — entity linking less effective |
| 9 | **Indexes** | Index on `:Node(id)` or per-type | Expects index on `:Entity(id)` and full-text on `:Entity(label)` | **HIGH** — performance degradation |

### 4.2 Compatibility Assessment

**KGB's SimpleKGAssembler** stores properties flat → closer to what we need.
**KGB's Neo4jGraphStore** serializes to JSON → completely incompatible.

The simplest path forward is:
1. **Double-label nodes**: KGB adds `:Entity` label alongside the entity_type label
2. **Flat properties + extras**: KGB stores required properties as flat fields
3. **Redundant relationship properties**: KGB stores `source_id`, `target_id`, `type` on relationships

---

## 5. Recommended KGB Output Schema (Target)

### 5.1 Node Creation Cypher (recommended)
```cypher
MERGE (n:Entity:{entity_type} {id: $entity_id})
SET n.label         = $label,
    n.entity_type   = $entity_type,       -- ← KG-RAG reads this
    n.description   = $description,
    n.confidence    = $confidence,
    n.source_doc_ids = $source_doc_ids,    -- ← renamed from "sources"
    n.aliases       = $aliases,
    n.evidence_count = $evidence_count,
    n.merged_count  = $merged_count,
    n.created_at    = datetime()
```

> **Key**: dual label `:Entity:{entity_type}` satisfies both systems.
> KG-RAG MATCH (e:Entity) works. KGB's type-specific queries also work.

### 5.2 Relationship Creation Cypher (recommended)
```cypher
MATCH (source:Entity {id: $source_id}), (target:Entity {id: $target_id})
MERGE (source)-[r:{predicate}]->(target)
SET r.id            = $rel_id,
    r.source_id     = $source_id,          -- ← redundant but required
    r.target_id     = $target_id,          -- ← redundant but required
    r.type          = $predicate,          -- ← redundant but required
    r.confidence    = $confidence,
    r.evidence_text = $evidence_text,      -- ← first evidence text span
    r.evidence_count = $evidence_count,
    r.created_at    = datetime()
```

### 5.3 Index Creation (recommended)
```cypher
-- Primary lookup (every KG-RAG Cypher query uses this):
CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id);

-- Text search for entity linking:
CREATE FULLTEXT INDEX entity_label IF NOT EXISTS
  FOR (e:Entity) ON EACH [e.label];

-- Type-based queries (KGB compatibility):
CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type);

-- Optional: uniqueness
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
  FOR (e:Entity) REQUIRE e.id IS UNIQUE;
```

### 5.4 Qdrant Ingestion (recommended)

When KGB stores chunks in Qdrant, include entity cross-references:
```python
qdrant_client.upsert(
    collection_name="document_chunks",
    points=[
        PointStruct(
            id=chunk.chunk_id,
            vector=embedding,
            payload={
                "document_id": chunk.document_id,
                "text": chunk.text,
                "entity_ids": [e.id for e in entities_in_chunk],  # ← cross-ref
                # any other metadata...
            },
        )
    ],
)
```

---

## 6. Cross-System Reference Chain

```
┌─────────────────────────────────────────────────────────────────┐
│                     KnowledgeGraphBuilder                       │
│                                                                 │
│  ExtractedEntity          ExtractedRelation                     │
│  ├── id ─────────────────►  source_entity_id / target_entity_id│
│  ├── label                  predicate                           │
│  ├── entity_type            confidence                          │
│  ├── description            evidence                            │
│  └── confidence                                                 │
│         │                        │                              │
│         ▼                        ▼                              │
│    Node (Neo4j)            Edge (Neo4j)                         │
│    + Chunk (Qdrant)                                             │
└────┬────────────────────────────────┬───────────────────────────┘
     │                                │
     ▼                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        KG-RAG Agent                             │
│                                                                 │
│  Qdrant payload.entity_ids ──► Neo4j Entity.id                  │
│  Neo4j  Entity.entity_type ──► Fuseki owl:Class URI             │
│  Neo4j  Relation.type      ──► Fuseki property URI              │
│  Neo4j  Relation.source_id ──► Neo4j Entity.id                  │
│  Neo4j  Relation.target_id ──► Neo4j Entity.id                  │
│                                                                 │
│  Entity.id ───► PPR seed nodes                                  │
│  Entity.label ─► fuzzy matching (entity linker)                 │
│  Entity.entity_type ─► ontology expansion (Fuseki)              │
│  Relation.type ─► path ranking (expected_relations)             │
│  Relation.confidence ─► path scoring (PathRanker)               │
│  Relation.evidence_text ─► reasoning chains (Explainer)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Implementation Recommendations

### Option A: Modify KGB Assembly (Preferred)

Create a KG-RAG-compatible assembler in KGB that produces the schema above.
This is a thin wrapper around `SimpleKGAssembler`:

```python
# In KnowledgeGraphBuilder — e.g. kgbuilder/assembly/qa_assembler.py

class QACompatibleAssembler(SimpleKGAssembler):
    """Assembler that produces KG-RAG-compatible Neo4j schema."""

    def _create_node(self, session, entity):
        query = f"""
        MERGE (n:Entity:{entity.entity_type} {{id: $entity_id}})
        SET n.label         = $label,
            n.entity_type   = $entity_type,
            n.description   = $description,
            n.confidence    = $confidence,
            n.source_doc_ids = $sources,
            n.aliases       = $aliases,
            n.evidence_count = $evidence_count,
            n.merged_count  = $merged_count,
            n.created_at    = timestamp()
        """
        session.run(query, entity_id=entity.id, label=entity.label,
                    entity_type=entity.entity_type, ...)

    def _create_relationship(self, session, relation):
        query = f"""
        MATCH (s:Entity {{id: $src}}), (t:Entity {{id: $tgt}})
        MERGE (s)-[r:{relation.predicate}]->(t)
        SET r.id            = $rel_id,
            r.source_id     = $src,
            r.target_id     = $tgt,
            r.type          = $predicate,
            r.confidence    = $confidence,
            r.evidence_text = $evidence_text,
            r.created_at    = timestamp()
        """
        session.run(query, ...)
```

### Option B: Adapt KG-RAG Connector (Alternative)

Make KG-RAG's `Neo4jConnector` flexible enough to handle KGB's output:
- Accept any node label (not just `:Entity`)
- Read `entity_type` from `labels(n)[0]` instead of `n.entity_type`
- Derive `source_id`/`target_id` from Neo4j's native start/end node IDs
- Parse JSON properties when needed

**Tradeoff**: More complex connector code; introduces KGB coupling in KG-RAG.

### Option C: Post-Processing Migration Script

A one-time or periodic script that transforms KGB's Neo4j output into KG-RAG
format:

```cypher
-- Add :Entity label to all nodes
MATCH (n) WHERE NOT n:Entity SET n:Entity;

-- Copy entity_type from label
MATCH (n:Entity)
WHERE n.entity_type IS NULL
SET n.entity_type = labels(n)[0];

-- Add redundant source/target to relationships
MATCH (s)-[r]->(t)
WHERE r.source_id IS NULL
SET r.source_id = s.id,
    r.target_id = t.id,
    r.type = type(r);
```

### Recommended: Option A + C

- **Option A** for new assemblies going forward
- **Option C** for existing graphs that need to work with KG-RAG immediately

---

## 8. Ontology Alignment

KGB extracts entity types and relation predicates from an ontology. For KG-RAG's
Fuseki connector to work, the same ontology must be loaded into Fuseki with:

1. **Classes** matching KGB `entity_type` values:
   ```turtle
   ndd:NuclearFacility a owl:Class ;
       rdfs:label "NuclearFacility" ;
       skos:altLabel "Kernanlage", "Nuclear Facility" ;
       rdfs:subClassOf ndd:Facility .
   ```

2. **Properties** matching KGB `predicate` values:
   ```turtle
   ndd:usesMethod a owl:ObjectProperty ;
       rdfs:label "usesMethod" ;
       rdfs:domain ndd:NuclearFacility ;
       rdfs:range ndd:DecommissioningMethod .
   ```

3. **Hierarchy** for ontology expansion:
   - `rdfs:subClassOf` chains enable KG-RAG to expand "Facility" →
     "NuclearFacility", "ResearchFacility", etc.

The ontology file used by KGB for extraction guidance should be deployed to
Fuseki as the KG-RAG query ontology.

---

## 9. Validation Checklist

Before connecting KGB output to KG-RAG, verify:

- [ ] Every node has `:Entity` label
- [ ] Every node has flat `id`, `label`, `entity_type`, `confidence` properties
- [ ] Every relationship has `source_id`, `target_id`, `type` as properties
- [ ] Index on `:Entity(id)` exists
- [ ] Full-text index on `:Entity(label)` exists
- [ ] Qdrant chunks have `entity_ids` in payload
- [ ] Fuseki dataset has ontology with `owl:Class`, `rdfs:label`, `rdfs:subClassOf`
- [ ] Entity types in Neo4j match class labels in Fuseki
- [ ] Relation predicates in Neo4j match property labels in Fuseki

### Quick Validation Queries:
```cypher
-- Check :Entity label exists
MATCH (n:Entity) RETURN count(n) AS entity_count;

-- Check required properties
MATCH (n:Entity)
WHERE n.entity_type IS NULL OR n.label IS NULL
RETURN count(n) AS missing_props;

-- Check relationship properties
MATCH ()-[r]->()
WHERE r.source_id IS NULL OR r.target_id IS NULL
RETURN count(r) AS missing_rel_props;

-- Check indexes
SHOW INDEXES;
```
