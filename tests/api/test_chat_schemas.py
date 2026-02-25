"""Tests for API chat schemas."""

from __future__ import annotations

from kgrag.api.chat_schemas import ChatMessage, ChatRequest, ChatResponse, FeedbackRequest


class TestChatSchemas:
    """Validate Pydantic models for the chat API."""

    def test_chat_message(self) -> None:
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_request_defaults(self) -> None:
        req = ChatRequest(message="What is X?")
        assert req.strategy == "hybrid_sota"
        assert req.language == "de"
        assert req.stream is True
        assert req.session_id is None

    def test_chat_response(self) -> None:
        resp = ChatResponse(
            session_id="abc",
            message=ChatMessage(role="assistant", content="X is Y."),
            confidence=0.8,
        )
        assert resp.session_id == "abc"
        assert resp.confidence == 0.8

    def test_feedback_request(self) -> None:
        fb = FeedbackRequest(
            question="Q",
            original_answer="A",
            corrected_answer="B",
            feedback_type="correction",
        )
        assert fb.feedback_type == "correction"
