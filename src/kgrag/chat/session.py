"""Chat session management with conversation history.

Each ``ChatSession`` tracks a multi-turn conversation.  The
``ChatSessionManager`` wires sessions to the :class:`Orchestrator` so every
incoming message is routed through the full QA pipeline with conversation
context.

Delegated implementation tasks
------------------------------
* TODO: Add persistent storage backend (SQLite / JSON file) so sessions
  survive server restarts.
* TODO: Honour ``ChatRequest.language`` when building the context prompt.
* TODO: Implement token-level streaming via LLM callbacks instead of
  word-level simulation.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import structlog

from kgrag.agents.orchestrator import Orchestrator
from kgrag.api.chat_schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EntityResponse,
    EvidenceResponse,
    GapDetectionResponse,
    RelationResponse,
    ReasoningStepResponse,
    VerificationResponse,
)
from kgrag.api.schemas import ProvenanceResponse

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ConversationTurn:
    """A single user ⇄ assistant exchange.

    Additional metadata (reasoning steps, provenance, subgraph, latency)
    is stored so that the history API can later render visualisations.
    """

    user_message: str
    assistant_message: str
    confidence: float = 0.0
    reasoning_chain: list[str] = field(default_factory=list)
    provenance: list[dict] = field(default_factory=list)
    subgraph: dict[str, Any] | None = None
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Single session
# ---------------------------------------------------------------------------


class ChatSession:
    """Manages one multi-turn conversation."""

    def __init__(self, session_id: str, *, max_history: int = 20) -> None:
        self.session_id = session_id
        self.turns: list[ConversationTurn] = []
        self.max_history = max_history
        self.created_at = time.time()

    # -- history bookkeeping ------------------------------------------------

    def add_turn(
        self,
        user_msg: str,
        assistant_msg: str,
        confidence: float = 0.0,
        reasoning_chain: list[str] | None = None,
        provenance: list[dict] | None = None,
        subgraph: dict[str, Any] | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        self.turns.append(
            ConversationTurn(
                user_message=user_msg,
                assistant_message=assistant_msg,
                confidence=confidence,
                reasoning_chain=reasoning_chain or [],
                provenance=provenance or [],
                subgraph=subgraph,
                latency_ms=latency_ms,
            )
        )
        if len(self.turns) > self.max_history:
            self.turns = self.turns[-self.max_history :]

    def get_context_prompt(self) -> str:
        """Build a conversation-context prefix for the LLM.

        Returns the last 5 turns so the model can maintain coherence in
        follow-up questions.
        """
        if not self.turns:
            return ""

        lines = ["Previous conversation:"]
        for turn in self.turns[-5:]:
            lines.append(f"User: {turn.user_message}")
            lines.append(f"Assistant: {turn.assistant_message}")
        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session manager (multiple conversations)
# ---------------------------------------------------------------------------


class ChatSessionManager:
    """Manage multiple chat sessions backed by the QA pipeline.

    Sessions are kept in memory but can optionally mirror their history to a
    ``HistoryStore`` backend (JSON file, SQLite, etc.).  This allows sessions
    to survive process restarts and provides an audit trail for demo exports.
    """

    def __init__(self, orchestrator: Orchestrator, history_store: Any | None = None) -> None:
        self._orchestrator = orchestrator
        self._sessions: dict[str, ChatSession] = {}
        self._history_store = history_store

    async def get_or_create_session(self, session_id: str) -> ChatSession:
        """Return existing session or create a new one.

        If a ``history_store`` is configured the session is seeded with past
        turns loaded from storage.  This method is now asynchronous so that
        history loading can ``await`` the store implementation.
        """
        if session_id not in self._sessions:
            session = ChatSession(session_id)
            self._sessions[session_id] = session
            logger.info("chat.session_created", session_id=session_id)
            # load persisted history if available
            if self._history_store is not None:
                try:
                    past = await self._history_store.load_session(session_id)
                except Exception:
                    past = []
                for turn in past:
                    session.turns.append(
                        ConversationTurn(
                            user_message=turn.get("user", ""),
                            assistant_message=turn.get("assistant", ""),
                            confidence=turn.get("confidence", 0.0),
                            timestamp=turn.get("timestamp", 0),
                        )
                    )
        return self._sessions[session_id]

    # -- core processing ----------------------------------------------------

    async def process_message(
        self,
        session_id: str,
        request: ChatRequest,
    ) -> ChatResponse:
        """Run a user message through the QA pipeline and return a response.

        The conversation history is prepended so the orchestrator sees the
        full multi-turn context.

        TODO (delegate):
        * Stream tokens from the LLM callback instead of returning the full
          answer in one shot.
        * Store the turn persistently (see ``history.py``).
        """
        session = await self.get_or_create_session(session_id)

        # Build contextual question with history
        context = session.get_context_prompt()
        full_question = (
            f"{context}Current question: {request.message}"
            if context
            else request.message
        )

        # Call the orchestrator
        answer = await self._orchestrator.answer(
            full_question,
            strategy=request.strategy,
        )

        # Record the turn
        session.add_turn(
            request.message,
            answer.answer_text,
            answer.confidence,
            reasoning_chain=answer.reasoning_chain,
            provenance=[
                {"source": ctx.source.value, "score": ctx.score}
                for ctx in answer.evidence
            ],
            subgraph=answer.subgraph_json,
            latency_ms=answer.latency_ms,
        )
        # persist to history store if configured
        if self._history_store is not None:
            try:
                await self._history_store.save_turn(
                    session_id,
                    request.message,
                    answer.answer_text,
                    {
                        "confidence": answer.confidence,
                        "reasoning_chain": answer.reasoning_chain,
                        "provenance": [
                            {"source": ctx.source.value, "score": ctx.score}
                            for ctx in answer.evidence
                        ],
                        "subgraph": answer.subgraph_json,
                        "latency_ms": answer.latency_ms,
                    },
                )
            except Exception:
                logger.warning("chat.history_save_failed", session_id=session_id)

        # Build API response — full rich context for the frontend
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

        evidence = [
            EvidenceResponse(
                text=ctx.text,
                score=ctx.score,
                source=ctx.source.value,
                doc_id=ctx.provenance.doc_id if ctx.provenance else None,
                source_id=ctx.provenance.source_id if ctx.provenance else None,
            )
            for ctx in answer.evidence
        ]

        cited_entities = [
            EntityResponse(
                id=ent.id,
                label=ent.label,
                entity_type=ent.entity_type,
                description=ent.description,
                properties=ent.properties,
            )
            for ent in answer.cited_entities
        ]

        cited_relations = [
            RelationResponse(
                source_id=rel.source_id,
                target_id=rel.target_id,
                relation_type=rel.relation_type,
                confidence=rel.confidence,
            )
            for rel in answer.cited_relations
        ]

        reasoning_steps = [
            ReasoningStepResponse(
                step_id=step.step_id,
                sub_question=step.sub_question,
                evidence_text=step.evidence_text,
                answer_fragment=step.answer_fragment,
                confidence=step.confidence,
            )
            for step in answer.reasoning_steps
        ]

        verification = None
        if answer.verification:
            verification = VerificationResponse(
                is_faithful=answer.verification.is_faithful,
                faithfulness_score=answer.verification.faithfulness_score,
                supported_claims=answer.verification.supported_claims,
                unsupported_claims=answer.verification.unsupported_claims,
                contradicted_claims=answer.verification.contradicted_claims,
                entity_coverage=answer.verification.entity_coverage,
            )

        # Gap detection (populated by Orchestrator Phase 8)
        gap_detection = None
        gap_info = getattr(answer, "_gap_info", None)
        if gap_info:
            gap_detection = GapDetectionResponse(
                gap_type=gap_info.get("gap_type", ""),
                description=gap_info.get("description", ""),
                affected_entities=gap_info.get("affected_entities", []),
            )

        return ChatResponse(
            session_id=session_id,
            message=ChatMessage(role="assistant", content=answer.answer_text),
            confidence=answer.confidence,
            latency_ms=answer.latency_ms,
            strategy_used=request.strategy,
            reasoning_chain=answer.reasoning_chain,
            reasoning_steps=reasoning_steps,
            evidence=evidence,
            provenance=provenance,
            cited_entities=cited_entities,
            cited_relations=cited_relations,
            subgraph=answer.subgraph_json,
            verification=verification,
            gap_detection=gap_detection,
        )

    # -- session queries ----------------------------------------------------

    async def get_history(self, session_id: str) -> list[dict]:
        """Return conversation history for a session.

        Preference is given to the in-memory session; if missing, the
        configured history store is consulted.
        """
        session = self._sessions.get(session_id)
        if session:
            return [
                {
                    "user": t.user_message,
                    "assistant": t.assistant_message,
                    "confidence": t.confidence,
                    "reasoning_chain": t.reasoning_chain,
                    "provenance": t.provenance,
                    "subgraph": t.subgraph,
                    "latency_ms": t.latency_ms,
                    "timestamp": t.timestamp,
                }
                for t in session.turns
            ]
        if self._history_store is not None:
            try:
                return await self._history_store.load_session(session_id)
            except Exception:
                return []
        return []

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        logger.info("chat.session_deleted", session_id=session_id)

    def list_sessions(self) -> list[dict]:
        """Return metadata for all active sessions."""
        return [
            {
                "session_id": sid,
                "turns": len(s.turns),
                "created_at": s.created_at,
            }
            for sid, s in self._sessions.items()
        ]
