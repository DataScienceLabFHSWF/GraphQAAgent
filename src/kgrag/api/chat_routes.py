"""Chat API routes with SSE streaming support.

Endpoints
---------
* ``POST /chat/send`` — send a message (SSE stream or JSON)
* ``GET  /chat/sessions`` — list all active sessions
* ``GET  /chat/sessions/{id}/history`` — conversation turns
* ``DELETE /chat/sessions/{id}`` — discard a session
* ``POST /chat/feedback`` — submit user corrections (HITL stub)

Delegated implementation tasks
------------------------------
* TODO: True token-level streaming via LLM callbacks.
* TODO: WebSocket variant for bi-directional communication.
* TODO: Rate limiting per session / IP.
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, HTTPException
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
# Chat endpoints
# ---------------------------------------------------------------------------


@router.post("/send", response_model=None)
async def chat_send(request: ChatRequest) -> ChatResponse | StreamingResponse:
    """Send a message — returns an SSE stream (default) or full JSON.

    If ``stream=True`` the response is ``text/event-stream`` with event
    types documented in :mod:`kgrag.chat.streaming`.
    """
    if _session_manager is None:
        raise HTTPException(503, "Chat not initialised")

    session_id = request.session_id or uuid.uuid4().hex[:12]

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
# Feedback / HITL (stub)
# ---------------------------------------------------------------------------


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest) -> dict:
    """Submit feedback or corrections on a QA answer.

    This feeds into the HITL pipeline.  Low-confidence answers and
    corrections are routed to KGBuilder (ABox) or OntologyExtender (TBox).

    TODO (delegate):
    * Persist feedback in a store (SQLite / JSON).
    * Route ``correction`` type to the KG versioning module
      (``kgrag.hitl.kg_versioning``) to create a ``ChangeProposal``.
    * Route ``flag`` type to a review queue.
    * Emit a change event for the HITL coordinator.
    """
    logger.info(
        "chat.feedback_received",
        feedback_type=request.feedback_type,
        session_id=request.session_id,
    )
    return {"status": "accepted", "message": "Feedback recorded (stub)"}
