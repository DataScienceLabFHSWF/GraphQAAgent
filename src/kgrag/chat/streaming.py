"""SSE (Server-Sent Events) streaming helpers.

Provides utilities for formatting SSE events and generating streaming
responses from the QA pipeline.

Delegated implementation tasks
------------------------------
* TODO: Hook into the LLM's token-level callback (LangChain
  ``BaseCallbackHandler.on_llm_new_token``) for true token streaming
  instead of the current word-level simulation.
* TODO: Add heartbeat events to keep the connection alive on slow queries.
* TODO: Add backpressure handling for slow clients.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import structlog

from kgrag.api.chat_schemas import ChatRequest
from kgrag.chat.session import ChatSessionManager

logger = structlog.get_logger(__name__)


def sse_event(event: str, data: object) -> str:
    """Format a single Server-Sent Event.

    Parameters
    ----------
    event:
        Event type (``token``, ``reasoning_step``, ``provenance``,
        ``subgraph``, ``done``, ``error``).
    data:
        JSON-serialisable payload.
    """
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def stream_chat_response(
    session_id: str,
    request: ChatRequest,
    session_manager: ChatSessionManager,
) -> AsyncGenerator[str, None]:
    """Generate SSE events for a streaming chat response.

    Event sequence:
    1. ``session`` — session ID for the client to track
    2. ``reasoning_step`` (0-N) — each CoT step
    3. ``token`` (1-N) — answer tokens (currently simulated)
    4. ``provenance`` — list of evidence sources
    5. ``subgraph`` — vis.js-formatted graph data
    6. ``done`` — final metadata (confidence, latency)

    On error an ``error`` event is emitted instead.

    TODO (delegate):
    * Replace word-level token simulation with real LLM callback streaming.
    * Emit ``thinking`` events during long retrieval phases.
    """
    # 1. Session ID
    yield sse_event("session", {"session_id": session_id})

    try:
        # Run full pipeline (non-streaming internally)
        result = await session_manager.process_message(session_id, request)

        # 2. Reasoning steps
        if request.include_reasoning and result.reasoning_chain:
            for i, step in enumerate(result.reasoning_chain):
                yield sse_event("reasoning_step", {"step": i + 1, "text": step})

        # 3. Answer tokens (simulated word-level streaming)
        words = result.message.content.split()
        for word in words:
            yield sse_event("token", {"text": word + " "})

        # 4. Provenance
        if result.provenance:
            yield sse_event(
                "provenance",
                [p.model_dump() for p in result.provenance],
            )

        # 5. Subgraph
        if request.include_subgraph and result.subgraph:
            yield sse_event("subgraph", result.subgraph)

        # 6. Done
        yield sse_event(
            "done",
            {
                "confidence": result.confidence,
                "latency_ms": result.latency_ms,
            },
        )

    except Exception as exc:
        logger.exception("chat.stream_error", session_id=session_id)
        yield sse_event("error", {"message": str(exc)})
