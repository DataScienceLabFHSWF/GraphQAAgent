"""SSE (Server-Sent Events) streaming helpers.

Provides utilities for formatting SSE events and generating streaming
responses from the QA pipeline.  Emits enriched events so the gaia-tt
TypeScript frontend can render reasoning steps, entity cards, evidence
panels, subgraph visualisations, and verification badges live.

SSE event sequence
------------------
1. ``session``          — session id
2. ``reasoning_step``   — 0-N structured reasoning steps
3. ``token``            — 1-N answer tokens (word-level until LLM callbacks)
4. ``evidence``         — retrieved evidence texts with provenance
5. ``entities``         — cited KG entities
6. ``relations``        — cited KG relations
7. ``provenance``       — evidence source metadata
8. ``subgraph``         — vis.js / Cytoscape-compatible graph JSON
9. ``verification``     — faithfulness check result
10. ``gap_alert``       — HITL gap detection (if triggered)
11. ``fact_chains``     — grounded KG fact chains (agentic)
12. ``tool_trace``      — agent tool call trace (agentic)
13. ``done``            — final metadata (confidence, latency, strategy)

On error an ``error`` event is emitted instead.
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
        Event type (see module docstring for the full list).
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

    Runs the full pipeline synchronously, then streams the result in
    discrete events so the frontend can progressively render each piece.

    Future improvements:
    * Replace word-level token simulation with real LLM callback streaming.
    * Emit ``thinking`` events during long retrieval phases.
    """
    # 1. Session ID
    yield sse_event("session", {"session_id": session_id})

    try:
        # Run full pipeline (non-streaming internally)
        result = await session_manager.process_message(session_id, request)

        # 2. Reasoning steps (structured)
        if request.include_reasoning:
            # Structured steps (CoT / ReAct)
            if result.reasoning_steps:
                for step in result.reasoning_steps:
                    yield sse_event("reasoning_step", step.model_dump())
            # Fallback: plain reasoning chain strings
            elif result.reasoning_chain:
                for i, text in enumerate(result.reasoning_chain):
                    yield sse_event("reasoning_step", {"step": i + 1, "text": text})

        # 3. Answer tokens (simulated word-level streaming)
        words = result.message.content.split()
        for word in words:
            yield sse_event("token", {"text": word + " "})

        # 4. Evidence (full texts with provenance)
        if request.include_evidence and result.evidence:
            yield sse_event(
                "evidence",
                [e.model_dump() for e in result.evidence],
            )

        # 5. Cited entities
        if result.cited_entities:
            yield sse_event(
                "entities",
                [e.model_dump() for e in result.cited_entities],
            )

        # 6. Cited relations
        if result.cited_relations:
            yield sse_event(
                "relations",
                [r.model_dump() for r in result.cited_relations],
            )

        # 7. Provenance (source metadata)
        if result.provenance:
            yield sse_event(
                "provenance",
                [p.model_dump() for p in result.provenance],
            )

        # 8. Subgraph (vis.js-compatible JSON)
        if request.include_subgraph and result.subgraph:
            yield sse_event("subgraph", result.subgraph)

        # 9. Verification
        if result.verification:
            yield sse_event("verification", result.verification.model_dump())

        # 10. Gap detection alert
        if result.gap_detection:
            yield sse_event("gap_alert", result.gap_detection.model_dump())

        # 11. Fact chains (agentic transparency)
        if result.fact_chains:
            yield sse_event(
                "fact_chains",
                [fc.model_dump() for fc in result.fact_chains],
            )

        # 12. Tool trace (agentic transparency)
        if result.tool_trace:
            yield sse_event(
                "tool_trace",
                [tt.model_dump() for tt in result.tool_trace],
            )

        # 13. Done
        yield sse_event(
            "done",
            {
                "confidence": result.confidence,
                "latency_ms": result.latency_ms,
                "strategy": result.strategy_used,
                "evidence_count": len(result.evidence),
                "entity_count": len(result.cited_entities),
            },
        )

    except Exception as exc:
        logger.exception("chat.stream_error", session_id=session_id)
        yield sse_event("error", {"message": str(exc)})
