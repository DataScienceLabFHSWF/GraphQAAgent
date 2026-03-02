"""Schema models for HITL endpoints and cross-service reports.

These are used by the ``/api/v1/hitl`` routes as well as the
low-confidence auto-reporting logic baked into ``ChatSessionManager``.

The models mirror the plan in ``planning/GRAPHQA_AGENT_API_PLAN.md`` but
coexist with the richer chat schema types.  Keeping them separate keeps
cross-service payloads lightweight.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QAResult(BaseModel):
    """A single QA result to report."""

    question: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    session_id: str = ""
    entity_types_mentioned: list[str] = Field(default_factory=list)


class LowConfidenceReport(BaseModel):
    """Batch of low-confidence QA results to send to KGBuilder."""

    qa_results: list[QAResult]
    threshold: float = Field(
        default=0.5,
        description="Confidence threshold used for filtering",
    )


class ReportResponse(BaseModel):
    """Response from KGBuilder after receiving low-confidence report."""

    status: str  # "received" | "gaps_detected" | "error"
    gaps_detected: int = 0
    suggested_classes: list[str] = Field(default_factory=list)
    message: str = ""


class FeedbackRequest(BaseModel):
    """Expert correction on an answer.  Placed at ``/api/v1/feedback``.

    This endpoint is primarily used by KGBuilder when an expert submits a
    correction via their HITL interface and flags the target service as the
    QA agent.
    """

    session_id: str
    turn_index: int
    correction: str
    feedback_type: str = "correction"  # "correction" | "approval" | "rejection"
    reviewer_id: str = "anonymous"


class FeedbackResponse(BaseModel):
    """Acknowledgment of feedback."""

    status: str  # "accepted" | "error"
    feedback_id: str = ""
