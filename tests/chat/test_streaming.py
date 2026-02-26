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

    def test_evidence_event(self) -> None:
        payload = [{"text": "Evidence 1", "score": 0.9, "source": "vector"}]
        result = sse_event("evidence", payload)
        assert "event: evidence\n" in result
        data = json.loads(result.split("data: ")[1].strip())
        assert isinstance(data, list)
        assert data[0]["score"] == 0.9

    def test_entities_event(self) -> None:
        payload = [{"id": "e1", "label": "Reactor", "entity_type": "Facility"}]
        result = sse_event("entities", payload)
        assert "event: entities\n" in result
        data = json.loads(result.split("data: ")[1].strip())
        assert data[0]["id"] == "e1"

    def test_relations_event(self) -> None:
        payload = [{"source_id": "a", "target_id": "b", "relation_type": "PART_OF"}]
        result = sse_event("relations", payload)
        assert "event: relations\n" in result
        data = json.loads(result.split("data: ")[1].strip())
        assert data[0]["relation_type"] == "PART_OF"

    def test_verification_event(self) -> None:
        payload = {"is_faithful": True, "faithfulness_score": 0.95}
        result = sse_event("verification", payload)
        assert "event: verification\n" in result
        data = json.loads(result.split("data: ")[1].strip())
        assert data["is_faithful"] is True

    def test_gap_alert_event(self) -> None:
        payload = {"gap_type": "abox_weak_evidence", "description": "Low confidence"}
        result = sse_event("gap_alert", payload)
        assert "event: gap_alert\n" in result
        data = json.loads(result.split("data: ")[1].strip())
        assert data["gap_type"] == "abox_weak_evidence"

    def test_done_event_with_enriched_fields(self) -> None:
        payload = {
            "confidence": 0.9,
            "latency_ms": 500.0,
            "strategy": "hybrid_sota",
            "evidence_count": 3,
            "entity_count": 5,
        }
        result = sse_event("done", payload)
        data = json.loads(result.split("data: ")[1].strip())
        assert data["strategy"] == "hybrid_sota"
        assert data["evidence_count"] == 3
        assert data["entity_count"] == 5

    def test_event_ends_with_double_newline(self) -> None:
        """SSE spec requires events to end with \\n\\n."""
        result = sse_event("session", {"session_id": "s1"})
        assert result.endswith("\n\n")
