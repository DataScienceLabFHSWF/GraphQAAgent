"""Unit tests for the GraphQA API chat route with telemetry."""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from services.graphqa_api.routes.chat import router as chat_router
from kgrag.telemetry import langsmith


@pytest.fixture(autouse=True)
def fake_tracing_context(monkeypatch):
    """Replace the actual tracing_context with a dummy so we can detect usage."""
    class Dummy:
        def __init__(self):
            self.entered = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    dummy = Dummy()

    def _fake(*args, **kwargs):
        return dummy

    monkeypatch.setattr(langsmith, "tracing_context", _fake)
    yield dummy


def test_chat_route_uses_tracing(fake_tracing_context):
    app = TestClient(chat_router)
    sid = uuid.uuid4().hex[:12]
    resp = app.post("/chat", json={"session_id": sid, "message": "hi", "stream": False})
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == sid
    # context should have been entered
    assert fake_tracing_context.entered
