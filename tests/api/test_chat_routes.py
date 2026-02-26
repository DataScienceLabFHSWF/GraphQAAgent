"""Unit tests for chat API routes using a dummy orchestrator."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from kgrag.api import chat_routes as cr
from kgrag.api.chat_schemas import ChatRequest
from kgrag.chat.session import ChatSessionManager


class DummyOrchestrator:
    async def answer(self, question: str, strategy: str = ""):
        # simple answer object mimic
        class A:
            answer_text = "hello"
            confidence = 0.42
            reasoning_chain = ["step1"]
            evidence = []
            cited_entities = []
            cited_relations = []
            reasoning_steps = []
            verification = None
            _gap_info = None
            subgraph_json = {"nodes": [], "edges": []}
            latency_ms = 10

        return A()


def _fake_request() -> MagicMock:
    """Create a minimal mock for ``starlette.requests.Request``."""
    req = MagicMock()
    req.client = MagicMock()
    req.client.host = "127.0.0.1"
    return req


@pytest.fixture(autouse=True)
def patch_session_manager(monkeypatch):
    mgr = ChatSessionManager(DummyOrchestrator())
    monkeypatch.setattr(cr, "_session_manager", mgr)
    # Reset rate-limit buckets between tests
    cr._rate_buckets.clear()
    yield mgr


def test_chat_send_basic():
    req = ChatRequest(message="hi", strategy="", language="de", stream=False)
    resp = asyncio.get_event_loop().run_until_complete(cr.chat_send(req, _fake_request()))
    assert resp.message.content == "hello"
    assert resp.confidence == pytest.approx(0.42)
    assert resp.subgraph == {"nodes": [], "edges": []}


def test_sessions_and_history():
    req = ChatRequest(session_id="s1", message="hi", strategy="", language="de", stream=False)
    asyncio.get_event_loop().run_until_complete(cr.chat_send(req, _fake_request()))
    sessions = asyncio.get_event_loop().run_until_complete(cr.list_sessions())
    assert any(s["session_id"] == "s1" for s in sessions)
    history = asyncio.get_event_loop().run_until_complete(cr.get_history("s1"))
    assert history[0]["user"] == "hi"
    assert history[0].get("reasoning_chain") == ["step1"]
    assert history[0].get("subgraph") == {"nodes": [], "edges": []}
