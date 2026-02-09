"""API routes — QA endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from kgrag.agents.orchestrator import Orchestrator
from kgrag.api.schemas import AnswerResponse, HealthResponse, ProvenanceResponse, QuestionRequest

router = APIRouter()

# Will be set by server.py on startup
_orchestrator: Orchestrator | None = None


def set_orchestrator(orch: Orchestrator) -> None:
    """Inject the orchestrator instance (called by server startup)."""
    global _orchestrator  # noqa: PLW0603
    _orchestrator = orch


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
