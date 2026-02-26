"""Domain models for the KG-RAG QA Agent.

Mirrors of KGB data (read-only) + QA-specific domain objects + evaluation models.

See planning/INTERFACE_CONTRACT.md for the canonical schema definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# 1. Mirrors of KGB data (read from Neo4j / Qdrant / Fuseki)
# ---------------------------------------------------------------------------


@dataclass
class KGEntity:
    """Entity node read from Neo4j.

    See INTERFACE_CONTRACT.md §1 for Neo4j schema.
    Note: evidence and source_doc_ids are NOT stored in Neo4j —
    cross-reference by entity.id against checkpoint JSON if needed.
    """

    id: str                     # Deterministic: ent_<sha256(label::type)[:12]>
    label: str
    entity_type: str            # Ontology class name (e.g. "Facility")
    description: str = ""       # LLM-generated description
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class KGRelation:
    """Relation edge read from Neo4j.

    See INTERFACE_CONTRACT.md §1 for Neo4j schema.
    Note: Neo4j relationships have NO stored id — identify by
    (source_id, relation_type, target_id) tuple.
    Evidence text is NOT stored in Neo4j — only in checkpoint JSON.
    """

    source_id: str              # Source entity ID
    target_id: str              # Target entity ID
    relation_type: str          # Dynamic type from predicate (e.g. "requiresPermit")
    confidence: float = 0.0


@dataclass
class DocumentChunk:
    """Chunk read from Qdrant payload.

    See INTERFACE_CONTRACT.md §2 for Qdrant schema.
    Field names match KGB payload keys exactly.
    """

    id: str                     # payload["id"] — chunk identifier
    doc_id: str                 # payload["doc_id"] — source document
    content: str                # payload["content"] — chunk text
    strategy: str = ""          # payload["strategy"] — chunking strategy
    embedding: list[float] | None = None  # populated on retrieval


@dataclass
class OntologyClass:
    """Ontology class read from Fuseki."""

    uri: str
    label: str
    parent_uri: str | None = None
    properties: list[OntologyProperty] = field(default_factory=list)
    description: str = ""


@dataclass
class OntologyProperty:
    """Ontology property read from Fuseki."""

    uri: str
    label: str
    domain_uri: str
    range_uri: str
    property_type: str = "object"  # "object" | "datatype"


# ---------------------------------------------------------------------------
# 2. QA Domain Models
# ---------------------------------------------------------------------------


class QuestionType(Enum):
    """Classification of user questions."""

    FACTOID = "factoid"
    LIST = "list"
    BOOLEAN = "boolean"
    COMPARATIVE = "comparative"
    CAUSAL = "causal"
    AGGREGATION = "aggregation"


class RetrievalSource(Enum):
    """Provenance label for a retrieved context piece."""

    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"
    ONTOLOGY = "ontology"


@dataclass
class QAQuery:
    """Parsed user question enriched by question parser + ontology expansion."""

    raw_question: str
    question_type: QuestionType | None = None
    detected_entities: list[str] = field(default_factory=list)
    detected_types: list[str] = field(default_factory=list)
    expected_relations: list[str] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)
    language: str = "de"


@dataclass
class Provenance:
    """Tracks where a piece of evidence came from.

    Uses KGB field names: source_id (not chunk_id), doc_id (not document_id).
    """

    doc_id: str | None = None       # Qdrant payload "doc_id"
    source_id: str | None = None    # Qdrant payload "id" (chunk ID)
    entity_ids: list[str] = field(default_factory=list)
    retrieval_strategy: str = ""
    retrieval_score: float = 0.0


@dataclass
class RetrievedContext:
    """Single piece of retrieved evidence with provenance."""

    source: RetrievalSource
    text: str
    score: float = 0.0
    chunk: DocumentChunk | None = None
    subgraph: list[KGEntity | KGRelation] | None = None
    provenance: Provenance | None = None


@dataclass
class ReasoningStep:
    """Single step in a multi-hop chain-of-thought reasoning trace."""

    step_id: int
    sub_question: str
    evidence_text: str = ""
    answer_fragment: str = ""
    grounding_entities: list[str] = field(default_factory=list)  # entity IDs
    grounding_relations: list[str] = field(default_factory=list)  # composite keys
    confidence: float = 0.0
    source: RetrievalSource = RetrievalSource.GRAPH


@dataclass
class VerificationResult:
    """Result of verifying a generated answer against KG evidence."""

    is_faithful: bool = True
    supported_claims: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    contradicted_claims: list[str] = field(default_factory=list)
    faithfulness_score: float = 1.0
    entity_coverage: float = 0.0  # fraction of answer entities found in KG


@dataclass
class GraphExplorationState:
    """State of an iterative Think-on-Graph exploration."""

    visited_entity_ids: set[str] = field(default_factory=set)
    frontier_entity_ids: set[str] = field(default_factory=set)
    collected_entities: list[KGEntity] = field(default_factory=list)
    collected_relations: list[KGRelation] = field(default_factory=list)
    exploration_path: list[str] = field(default_factory=list)  # human-readable trace
    iterations: int = 0
    sufficient_evidence: bool = False


@dataclass
class QAAnswer:
    """Final answer with full provenance and explainability artefacts."""

    question: str
    answer_text: str
    confidence: float = 0.0
    reasoning_chain: list[str] = field(default_factory=list)
    reasoning_steps: list[ReasoningStep] = field(default_factory=list)
    evidence: list[RetrievedContext] = field(default_factory=list)
    cited_entities: list[KGEntity] = field(default_factory=list)
    cited_relations: list[KGRelation] = field(default_factory=list)
    subgraph_json: dict[str, Any] | None = None  # Visualisable subgraph
    verification: VerificationResult | None = None
    exploration_trace: GraphExplorationState | None = None
    react_metadata: dict[str, Any] | None = None  # NEW: ReAct reasoning metadata
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# 3. Evaluation Models
# ---------------------------------------------------------------------------


@dataclass
class QABenchmarkItem:
    """Single item from a gold-standard QA benchmark."""

    question_id: str
    question: str
    expected_answer: str
    expected_entities: list[str] = field(default_factory=list)
    difficulty: str | int = "medium"
    question_type: str = "factoid"
    competency_question_id: str | None = None
    category: str = ""
    retrieval_complexity: str = ""  # e.g. single_entity_lookup, multi_hop_path_traversal
    notes: str = ""


@dataclass
class QAEvalResult:
    """Evaluation result for a single question."""

    question_id: str
    predicted_answer: str
    expected_answer: str
    exact_match: bool = False
    f1_score: float = 0.0
    faithfulness: float = 0.0
    relevance: float = 0.0
    latency_ms: float = 0.0
    retrieval_strategy: str = ""
    context_count: int = 0
    # DeepEval LLM-as-a-judge scores (optional — None when not computed)
    deepeval_answer_relevancy: float | None = None
    deepeval_faithfulness: float | None = None
    deepeval_contextual_relevancy: float | None = None
    deepeval_contextual_precision: float | None = None
    deepeval_contextual_recall: float | None = None
    deepeval_correctness: float | None = None


@dataclass
class StrategyComparison:
    """Aggregated comparison of a retrieval strategy across a benchmark."""

    strategy_name: str
    avg_f1: float = 0.0
    avg_faithfulness: float = 0.0
    avg_relevance: float = 0.0
    avg_latency_ms: float = 0.0
    exact_match_rate: float = 0.0
    num_questions: int = 0
    per_type_f1: dict[str, float] = field(default_factory=dict)
    # DeepEval aggregated scores
    avg_deepeval_answer_relevancy: float | None = None
    avg_deepeval_faithfulness: float | None = None
    avg_deepeval_correctness: float | None = None


# ---------------------------------------------------------------------------
# 4. Competency Question Models
# ---------------------------------------------------------------------------


@dataclass
class CompetencyQuestion:
    """CQ in canonical KGB format (see INTERFACE_CONTRACT.md §5).

    Matches QAQuestion from kgbuilder.evaluation.qa_dataset.
    """

    id: str                             # e.g. "CQ_001"
    question: str                       # Natural language question
    expected_answers: list[str] = field(default_factory=list)
    query_type: str = "entity"          # "entity" | "relation" | "count" | "boolean" | "complex"
    difficulty: int = 1                 # 1–5
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
