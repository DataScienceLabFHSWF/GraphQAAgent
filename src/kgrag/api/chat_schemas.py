"""Chat-specific request / response schemas.

Extends the core ``schemas.py`` with models for multi-turn chat, SSE
streaming, and session management.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from kgrag.api.schemas import ProvenanceResponse


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    timestamp: str | None = None
    metadata: dict[str, Any] | None = None  # confidence, reasoning, provenance


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Request body for ``POST /chat/send``."""

    session_id: str | None = None  # None → create new session
    message: str = Field(..., min_length=1)
    strategy: str = "hybrid_sota"
    language: str = "de"
    stream: bool = True  # False → return full JSON like /ask
    include_reasoning: bool = True
    include_subgraph: bool = True


class ChatResponse(BaseModel):
    """Full (non-streaming) chat response."""

    session_id: str
    message: ChatMessage
    confidence: float = 0.0
    reasoning_chain: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceResponse] = Field(default_factory=list)
    subgraph: dict[str, Any] | None = None
    latency_ms: float = 0.0


class ChatStreamEvent(BaseModel):
    """Payload structure for a single SSE event.

    ``event`` values: ``token``, ``reasoning_step``, ``provenance``,
    ``subgraph``, ``done``, ``error``.
    """

    event: str
    data: str  # JSON-encoded payload


# ---------------------------------------------------------------------------
# HITL Feedback (stub)
# ---------------------------------------------------------------------------


class FeedbackRequest(BaseModel):
    """User feedback / correction on a QA answer.

    TODO (delegate): Implement feedback ingestion.  Route corrections to
    KGBuilder (ABox fix) or OntologyExtender (TBox gap) via the HITL
    pipeline.
    """

    session_id: str | None = None
    question: str
    original_answer: str
    corrected_answer: str | None = None
    feedback_type: str = Field(
        default="correction",
        pattern="^(correction|flag|rating)$",
    )
    rating: int | None = Field(default=None, ge=1, le=5)
    comment: str | None = None
