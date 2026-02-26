"""API routes — QA endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from kgrag.agents.orchestrator import Orchestrator
from kgrag.api.schemas import AnswerResponse, HealthResponse, ProvenanceResponse, QuestionRequest

router = APIRouter()

# Will be set by server.py on startup
_orchestrator: Orchestrator | None = None


def set_orchestrator(orch: Orchestrator) -> None:
    """Inject the orchestrator instance (called by server startup)."""
    global _orchestrator  # noqa: PLW0603
    _orchestrator = orch


# ---------------------------------------------------------------------------
# Strategy catalogue — available retrieval strategies with metadata
# ---------------------------------------------------------------------------

STRATEGY_INFO: dict[str, dict[str, str]] = {
    "vector_only": {
        "display_name": "Vector Search",
        "description": "Semantic similarity search against the Qdrant vector store.",
    },
    "graph_only": {
        "display_name": "Graph Traversal",
        "description": "Structured Neo4j graph traversal with pattern-based reasoning.",
    },
    "hybrid": {
        "display_name": "Hybrid Fusion (RRF)",
        "description": "Reciprocal Rank Fusion of vector and graph results with adaptive weighting.",
    },
    "cypher": {
        "display_name": "Cypher Query Generation",
        "description": "LLM-generated Cypher queries executed against the knowledge graph.",
    },
    "agentic": {
        "display_name": "Agentic ReAct",
        "description": "Multi-step ReAct agent with tool use for iterative evidence gathering.",
    },
    "hybrid_sota": {
        "display_name": "Hybrid SOTA (Ontology-Informed)",
        "description": "State-of-the-art hybrid with ontology-guided query expansion and class-hierarchy weighting.",
    },
}


class StrategyDetail(BaseModel):
    """Single strategy description."""

    id: str
    display_name: str
    description: str


class StrategiesResponse(BaseModel):
    """List of available retrieval strategies."""

    strategies: list[StrategyDetail] = Field(default_factory=list)
    default: str = "hybrid"


@router.get("/strategies", response_model=StrategiesResponse)
async def list_strategies() -> StrategiesResponse:
    """Return the available retrieval strategies with display names."""
    items = [
        StrategyDetail(id=sid, display_name=info["display_name"], description=info["description"])
        for sid, info in STRATEGY_INFO.items()
    ]
    return StrategiesResponse(strategies=items, default="hybrid")


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    from kgrag import __version__

    return HealthResponse(status="ok", version=__version__)


@router.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest) -> AnswerResponse:
    """Answer a question using the KG-RAG pipeline."""
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialised")

    try:
        answer = await _orchestrator.answer(
            request.question,
            strategy=request.strategy,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    provenance = [
        ProvenanceResponse(
            source=ctx.source.value,
            score=ctx.score,
            doc_id=ctx.provenance.doc_id if ctx.provenance else None,
            source_id=ctx.provenance.source_id if ctx.provenance else None,
            entity_ids=ctx.provenance.entity_ids if ctx.provenance else [],
        )
        for ctx in answer.evidence
    ]

    return AnswerResponse(
        question=answer.question,
        answer=answer.answer_text,
        confidence=answer.confidence,
        reasoning_chain=answer.reasoning_chain,
        provenance=provenance,
        subgraph=answer.subgraph_json,
        latency_ms=answer.latency_ms,
    )
