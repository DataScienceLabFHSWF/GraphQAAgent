"""Tests for the chat session module."""

from __future__ import annotations

import pytest

from kgrag.chat.session import ChatSession, ConversationTurn


class TestChatSession:
    """Unit tests for ChatSession."""

    def test_create_session(self) -> None:
        session = ChatSession("test-001")
        assert session.session_id == "test-001"
        assert session.turns == []
        assert session.created_at > 0

    def test_add_turn(self) -> None:
        session = ChatSession("test-002")
        session.add_turn("Hello", "Hi there", confidence=0.9)
        assert len(session.turns) == 1
        assert session.turns[0].user_message == "Hello"
        assert session.turns[0].assistant_message == "Hi there"
        assert session.turns[0].confidence == 0.9

    def test_max_history_truncation(self) -> None:
        session = ChatSession("test-003", max_history=3)
        for i in range(5):
            session.add_turn(f"Q{i}", f"A{i}")
        assert len(session.turns) == 3
        assert session.turns[0].user_message == "Q2"

    def test_context_prompt_empty(self) -> None:
        session = ChatSession("test-004")
        assert session.get_context_prompt() == ""

    def test_context_prompt_with_history(self) -> None:
        session = ChatSession("test-005")
        session.add_turn("What is X?", "X is Y.")
        prompt = session.get_context_prompt()
        assert "User: What is X?" in prompt
        assert "Assistant: X is Y." in prompt
