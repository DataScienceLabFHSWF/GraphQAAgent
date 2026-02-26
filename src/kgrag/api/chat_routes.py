"""Chat API routes with SSE streaming support.

Endpoints
---------
* ``POST /chat/send`` — send a message (SSE stream or JSON)
* ``GET  /chat/sessions`` — list all active sessions
* ``GET  /chat/sessions/{id}/history`` — conversation turns
* ``DELETE /chat/sessions/{id}`` — discard a session
* ``POST /chat/feedback`` — submit user corrections (HITL pipeline)

Implementation notes
--------------------
* Token-level streaming: Deferred — requires LLM callback infrastructure.
* WebSocket variant: Deferred — SSE covers gaia-tt requirements.
* Rate limiting: In-memory sliding-window per session and per IP.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from kgrag.api.chat_schemas import ChatRequest, ChatResponse, FeedbackRequest
from kgrag.chat.session import ChatSessionManager
from kgrag.chat.streaming import stream_chat_response

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# Injected by server.py during startup
_session_manager: ChatSessionManager | None = None


def set_session_manager(mgr: ChatSessionManager) -> None:
    """Wire the session manager (called once during lifespan)."""
    global _session_manager  # noqa: PLW0603
    _session_manager = mgr


# ---------------------------------------------------------------------------
# Rate limiting — sliding-window per session and per IP
# ---------------------------------------------------------------------------

_RATE_LIMIT_WINDOW = 60  # seconds
_RATE_LIMIT_MAX_REQUESTS = 30  # max requests per window

# {key: [timestamp, ...]}
_rate_buckets: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(key: str) -> None:
    """Raise 429 if the rate limit for *key* is exceeded."""
    now = time.time()
    bucket = _rate_buckets[key]
    # Prune timestamps outside window
    _rate_buckets[key] = bucket = [t for t in bucket if now - t < _RATE_LIMIT_WINDOW]
    if len(bucket) >= _RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            429,
            detail=f"Rate limit exceeded ({_RATE_LIMIT_MAX_REQUESTS} req/{_RATE_LIMIT_WINDOW}s)",
        )
    bucket.append(now)


# ---------------------------------------------------------------------------
# Chat endpoints
# ---------------------------------------------------------------------------


@router.post("/send", response_model=None)
async def chat_send(
    request: ChatRequest,
    raw_request: Request,
) -> ChatResponse | StreamingResponse:
    """Send a message — returns an SSE stream (default) or full JSON.

    If ``stream=True`` the response is ``text/event-stream`` with event
    types documented in :mod:`kgrag.chat.streaming`.
    """
    if _session_manager is None:
        raise HTTPException(503, "Chat not initialised")

    session_id = request.session_id or uuid.uuid4().hex[:12]

    # Rate limit by session and by client IP
    _check_rate_limit(f"session:{session_id}")
    if raw_request.client:
        _check_rate_limit(f"ip:{raw_request.client.host}")

    if request.stream:
        return StreamingResponse(
            stream_chat_response(session_id, request, _session_manager),
            media_type="text/event-stream",
        )

    # Non-streaming: full JSON response
    result = await _session_manager.process_message(session_id, request)
    return result


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


@router.get("/sessions")
async def list_sessions() -> list[dict]:
    """List all active chat sessions."""
    if _session_manager is None:
        raise HTTPException(503, "Chat not initialised")
    return _session_manager.list_sessions()


@router.get("/sessions/{session_id}/history")
async def get_history(session_id: str) -> list[dict]:
    """Get the conversation history for a session."""
    if _session_manager is None:
        raise HTTPException(503, "Chat not initialised")
    return await _session_manager.get_history(session_id)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    """Delete a chat session and its history."""
    if _session_manager is None:
        raise HTTPException(503, "Chat not initialised")
    _session_manager.delete_session(session_id)
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Feedback / HITL
# ---------------------------------------------------------------------------


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest) -> dict:
    """Submit feedback or corrections on a QA answer.

    Routing logic:
    * ``correction`` → persisted and queued for the HITL change-proposal
      pipeline.  If the session manager has an orchestrator with a
      ``ChangeProposalService`` attached, a ``ChangeProposal`` is created
      automatically.
    * ``flag`` → logged for manual review.
    * ``rating`` → logged for quality tracking.
    """
    logger.info(
        "chat.feedback_received",
        feedback_type=request.feedback_type,
        session_id=request.session_id,
        rating=request.rating,
    )

    # Persist the feedback in the history store
    if _session_manager is not None and _session_manager._history_store is not None:
        try:
            await _session_manager._history_store.save_turn(
                request.session_id or "__feedback__",
                request.question,
                request.original_answer,
                {
                    "feedback_type": request.feedback_type,
                    "corrected_answer": request.corrected_answer,
                    "rating": request.rating,
                    "comment": request.comment,
                    "is_feedback": True,
                },
            )
        except Exception:
            logger.warning("chat.feedback_persist_failed")

    # Route corrections to the HITL change-proposal pipeline
    if request.feedback_type == "correction" and request.corrected_answer:
        try:
            from kgrag.hitl.change_proposals import ChangeProposalService, ProposalType

            # Try to get the service from orchestrator (if wired)
            orch = getattr(_session_manager, "_orchestrator", None)
            proposal_svc = getattr(orch, "_change_proposal_service", None)
            if proposal_svc and isinstance(proposal_svc, ChangeProposalService):
                proposal_svc.create_proposal(
                    proposal_type=ProposalType.UPDATE_ENTITY,
                    proposed_data={
                        "original_answer": request.original_answer,
                        "corrected_answer": request.corrected_answer,
                        "question": request.question,
                    },
                    trigger_question=request.question,
                    rationale=request.comment or "User correction via chat feedback",
                )
                logger.info("chat.feedback_proposal_created", question=request.question[:80])
                return {"status": "accepted", "message": "Correction recorded and change proposal created"}
        except Exception:
            logger.warning("chat.feedback_proposal_failed")

    return {"status": "accepted", "message": "Feedback recorded"}
