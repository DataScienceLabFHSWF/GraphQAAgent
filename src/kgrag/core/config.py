"""Application configuration using Pydantic Settings.

All settings are loaded from environment variables with the ``KGRAG_`` prefix.
Nested models use ``__`` as delimiter (e.g. ``KGRAG_NEO4J__URI``).
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Neo4jConfig(BaseModel):
    """Neo4j connection (read-only — consumes KGB's graph)."""

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    node_label: str = "Entity"  # Primary node label used in all queries


class QdrantConfig(BaseModel):
    """Qdrant connection (read-only — consumes KGB's vectors)."""

    url: str = "http://localhost:6333"
    collection_name: str = "kgbuilder"      # Must match KGB (see INTERFACE_CONTRACT.md §2)


class FusekiConfig(BaseModel):
    """Fuseki SPARQL endpoint (read-only — reads ontology)."""

    url: str = "http://localhost:3030"
    dataset: str = "kgbuilder"              # Must match KGB (see INTERFACE_CONTRACT.md §3)
    user: str | None = None
    password: str | None = None


class OllamaConfig(BaseModel):
    """Ollama LLM provider configuration."""

    base_url: str = "http://localhost:11434"
    generation_model: str = "qwen3:8b"
    embedding_model: str = "qwen3-embedding:latest"
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)


class ReasoningConfig(BaseModel):
    """Think-on-Graph + Chain-of-Thought reasoning parameters."""

    max_exploration_iterations: int = 3
    exploration_breadth: int = 5  # max neighbours to consider per iteration
    ppr_damping: float = Field(default=0.85, ge=0.0, le=1.0)
    ppr_top_k: int = 20  # top-k PPR-scored nodes to keep
    min_evidence_score: float = 0.3  # stop exploration when avg score exceeds this
    enable_chain_of_thought: bool = True
    enable_react_reasoning: bool = False  # NEW: Enable ReAct reasoning with tool-calling
    enable_iterative_exploration: bool = True
    enable_answer_verification: bool = True
    cot_max_steps: int = 5
    react_max_iterations: int = 3  # NEW: Max iterations for ReAct reasoning
    react_relevance_threshold: float = 0.3  # NEW: Threshold for additional retrieval
    verification_claim_threshold: float = 0.5


class RetrievalConfig(BaseModel):
    """Retrieval strategy parameters."""

    vector_top_k: int = 10
    graph_max_hops: int = 2
    graph_max_nodes: int = 50
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    fusion_weights: dict[str, float] = Field(
        default_factory=lambda: {"vector": 0.4, "graph": 0.4, "ontology": 0.2}
    )
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)


class EvaluationConfig(BaseModel):
    """Evaluation pipeline settings."""

    benchmark_path: str = "data/qa_benchmarks/benchmark_v1.json"
    strategies: list[str] = Field(
        default_factory=lambda: ["vector_only", "graph_only", "hybrid"]
    )
    num_runs: int = 3


class HitlConfig(BaseModel):
    """Configuration for human-in-the-loop / cross-service integration."""

    # base URL of the KGBuilder API; used when reporting low-confidence
    # QA results for gap detection.
    kgbuilder_api_url: str = "http://localhost:8001"
    confidence_threshold: float = 0.5  # below this value trigger report


class Settings(BaseSettings):
    """Root settings — loaded from env vars with ``KGRAG_`` prefix."""

    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    fuseki: FusekiConfig = Field(default_factory=FusekiConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    hitl: HitlConfig = Field(default_factory=HitlConfig)
    log_level: str = "INFO"

    model_config = {
        "env_prefix": "KGRAG_",
        "env_nested_delimiter": "__",
    }
