"""Tests for ChatSessionManager and integration with history stores."""

from __future__ import annotations

import asyncio

import pytest

from kgrag.chat.session import ChatSessionManager, ChatSession
from kgrag.chat.history import InMemoryHistoryStore, JsonFileHistoryStore
from kgrag.agents.orchestrator import Orchestrator


class DummyOrchestrator:
    async def answer(self, question: str, strategy: str = ""):
        # simple echo answer object mimic
        class A:
            answer_text = "ok"
            confidence = 0.5
            reasoning_chain = []
            evidence = []
            subgraph_json = None
            latency_ms = 1
        return A()


@pytest.fixture
def manager(tmp_path):
    # Default uses JSON history store for demo
    hist = JsonFileHistoryStore(base_dir=str(tmp_path))
    return ChatSessionManager(DummyOrchestrator(), history_store=hist)


def test_session_creation_and_history(manager):
    sess = asyncio.get_event_loop().run_until_complete(manager.get_or_create_session("s123"))
    assert isinstance(sess, ChatSession)
    # add a message via process_message
    req = type("R", (), {"message": "hello", "strategy": "", "session_id": None})
    resp = asyncio.get_event_loop().run_until_complete(manager.process_message("s123", req))
    assert resp.message.content == "ok"
    # history should now contain one turn with metadata
    hist = asyncio.get_event_loop().run_until_complete(manager.get_history("s123"))
    assert len(hist) == 1
    turn = hist[0]
    assert "confidence" in turn
    assert turn.get("reasoning_chain", []) == []
    assert turn.get("provenance", []) == []
    assert turn.get("subgraph") is None


def test_persistent_reload(tmp_path):
    hist = JsonFileHistoryStore(base_dir=str(tmp_path))
    mgr1 = ChatSessionManager(DummyOrchestrator(), history_store=hist)
    # add a turn
    req = type("R", (), {"message": "hi", "strategy": "", "session_id": None})
    asyncio.get_event_loop().run_until_complete(mgr1.process_message("sX", req))
    # create new manager and load history; metadata should survive
    mgr2 = ChatSessionManager(DummyOrchestrator(), history_store=hist)
    sess = asyncio.get_event_loop().run_until_complete(mgr2.get_or_create_session("sX"))
    assert len(sess.turns) == 1
    assert sess.turns[0].reasoning_chain == []
    assert sess.turns[0].subgraph is None
