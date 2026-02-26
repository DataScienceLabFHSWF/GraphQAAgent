"""Chat-specific request / response schemas.

Extends the core ``schemas.py`` with models for multi-turn chat, SSE
streaming, session management, and rich answer context for the
TypeScript frontend (gaia-tt).

The schemas are designed so that:
- ``npx openapi-typescript`` can generate matching TS types
- The gaia-tt Chat.tsx component can render entity cards, evidence
  panels, subgraph visualisations, and verification badges
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
# Rich context models (for TypeScript frontend rendering)
# ---------------------------------------------------------------------------


class EntityResponse(BaseModel):
    """Entity card data — rendered as a clickable card in the frontend."""

    id: str
    label: str
    entity_type: str
    description: str = ""
    properties: dict[str, Any] = Field(default_factory=dict)


class RelationResponse(BaseModel):
    """Relation edge for subgraph visualisation."""

    source_id: str
    target_id: str
    relation_type: str
    confidence: float = 0.0


class EvidenceResponse(BaseModel):
    """A single piece of retrieved evidence with full provenance."""

    text: str
    score: float = 0.0
    source: str = ""  # "vector" | "graph" | "hybrid" | "ontology"
    doc_id: str | None = None
    source_id: str | None = None


class ReasoningStepResponse(BaseModel):
    """Structured reasoning step (CoT or ReAct)."""

    step_id: int
    sub_question: str = ""
    evidence_text: str = ""
    answer_fragment: str = ""
    confidence: float = 0.0


class VerificationResponse(BaseModel):
    """Answer verification / faithfulness result."""

    is_faithful: bool = True
    faithfulness_score: float = 1.0
    supported_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    contradicted_claims: list[str] = Field(default_factory=list)
    entity_coverage: float = 0.0


class GapDetectionResponse(BaseModel):
    """HITL gap detection alert — indicates missing knowledge."""

    gap_type: str  # "tbox_missing_class", "abox_weak_evidence", etc.
    description: str = ""
    affected_entities: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Request body for ``POST /chat/send``.

    When ``stream=True`` (default) the response is ``text/event-stream``.
    When ``stream=False`` the response is a full JSON ``ChatResponse``.
    """

    session_id: str | None = None  # None → create new session
    message: str = Field(..., min_length=1)
    strategy: str = "hybrid_sota"
    language: str = "de"
    stream: bool = True  # False → return full JSON like /ask
    include_reasoning: bool = True
    include_subgraph: bool = True
    include_evidence: bool = True


class ChatResponse(BaseModel):
    """Full (non-streaming) chat response with rich context.

    This is the main payload consumed by the gaia-tt frontend when
    ``stream=False``, and also the data model behind the SSE events
    when ``stream=True``.
    """

    session_id: str
    message: ChatMessage

    # Core metrics
    confidence: float = 0.0
    latency_ms: float = 0.0
    strategy_used: str = ""

    # Reasoning
    reasoning_chain: list[str] = Field(default_factory=list)
    reasoning_steps: list[ReasoningStepResponse] = Field(default_factory=list)

    # Evidence & provenance
    evidence: list[EvidenceResponse] = Field(default_factory=list)
    provenance: list[ProvenanceResponse] = Field(default_factory=list)

    # KG context
    cited_entities: list[EntityResponse] = Field(default_factory=list)
    cited_relations: list[RelationResponse] = Field(default_factory=list)
    subgraph: dict[str, Any] | None = None

    # Verification & gaps
    verification: VerificationResponse | None = None
    gap_detection: GapDetectionResponse | None = None


class ChatStreamEvent(BaseModel):
    """Payload structure for a single SSE event.

    ``event`` values: ``session``, ``reasoning_step``, ``token``,
    ``evidence``, ``entities``, ``relations``, ``provenance``,
    ``subgraph``, ``verification``, ``gap_alert``, ``done``, ``error``.
    """

    event: str
    data: str  # JSON-encoded payload


# ---------------------------------------------------------------------------
# HITL Feedback
# ---------------------------------------------------------------------------


class FeedbackRequest(BaseModel):
    """User feedback / correction on a QA answer.

    Routes corrections to KGBuilder (ABox fix) or OntologyExtender
    (TBox gap) via the HITL pipeline.
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
