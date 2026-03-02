"""HITL feedback endpoints — low confidence reporting and expert corrections.

The ``/hitl/report-low-confidence`` endpoint accepts a batch of QA
results, filters out anything above the configured threshold, and forwards
the remaining items to KGBuilder's gap detector.  This is the primary
cross-service trigger for ontology / data updates.

The ``/feedback`` endpoint is a lightweight acknowledgement route that
can be called by KGBuilder or external tools when a human reviewer
submits a correction or rating; the payload is currently only logged and
a UUID is returned.
"""

from __future__ import annotations

import uuid
import structlog
from fastapi import APIRouter, HTTPException

from kgrag.api.hitl_schemas import (
    FeedbackRequest,
    FeedbackResponse,
    LowConfidenceReport,
    ReportResponse,
)
from kgrag.api.dependencies import get_settings

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/hitl/report-low-confidence", response_model=ReportResponse)
async def report_low_confidence(report: LowConfidenceReport) -> ReportResponse:
    """Report low-confidence QA results to KGBuilder for gap detection.

    Only items below the configured threshold are forwarded; the rest are
    ignored so callers (e.g. chat sessions) can submit whole batches without
    having to filter themselves.
    """
    settings = get_settings()
    threshold = float(settings.hitl.confidence_threshold)

    low_conf = [r.model_dump() for r in report.qa_results if r.confidence < threshold]
    if not low_conf:
        return ReportResponse(
            status="received",
            message=f"No results below threshold {threshold}",
        )

    logger.info(
        "reporting_low_confidence",
        count=len(low_conf),
        threshold=threshold,
    )

    # import here to allow tests to monkeypatch the symbol before
    # the function is executed (avoids stale binding from module import)
    from kgrag.api.dependencies import forward_to_kgbuilder

    result = forward_to_kgbuilder(low_conf)
    return ReportResponse(
        status=result.get("status", "error"),
        gaps_detected=result.get("gaps_detected", 0),
        suggested_classes=result.get("suggested_new_classes", []),
        message=result.get("message", "Forwarded to KGBuilder"),
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Receive expert feedback forwarded by KGBuilder HITL workflow.

    Currently the feedback is simply logged and a synthetic ID returned.  In
    the future this could persist to a database or route into the local
    change-proposal pipeline.
    """
    logger.info(
        "hitl.feedback_received",
        session_id=request.session_id,
        feedback_type=request.feedback_type,
    )

    # TODO: persist or forward
    feedback_id = f"fb_{uuid.uuid4().hex[:12]}"
    return FeedbackResponse(status="accepted", feedback_id=feedback_id)
