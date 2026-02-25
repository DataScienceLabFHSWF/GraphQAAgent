"""Tests for chat streaming helpers."""

from __future__ import annotations

import json

from kgrag.chat.streaming import sse_event


class TestSSEFormatting:
    """Unit tests for SSE event formatting."""

    def test_basic_event(self) -> None:
        result = sse_event("token", {"text": "hello "})
        assert result.startswith("event: token\n")
        assert "data: " in result
        data = json.loads(result.split("data: ")[1].strip())
        assert data["text"] == "hello "

    def test_done_event(self) -> None:
        result = sse_event("done", {"confidence": 0.85, "latency_ms": 123.4})
        assert "event: done\n" in result
        data = json.loads(result.split("data: ")[1].strip())
        assert data["confidence"] == 0.85

    def test_error_event(self) -> None:
        result = sse_event("error", {"message": "something failed"})
        assert "event: error\n" in result
