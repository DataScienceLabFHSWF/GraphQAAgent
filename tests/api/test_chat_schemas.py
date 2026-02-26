"""Tests for API chat schemas."""

from __future__ import annotations

from kgrag.api.chat_schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EntityResponse,
    EvidenceResponse,
    FeedbackRequest,
    GapDetectionResponse,
    RelationResponse,
    ReasoningStepResponse,
    VerificationResponse,
)


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
        assert req.include_evidence is True
        assert req.include_reasoning is True
        assert req.include_subgraph is True

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


class TestEntityResponse:
    """Validate EntityResponse model."""

    def test_minimal(self) -> None:
        e = EntityResponse(id="e1", label="Reactor", entity_type="Facility")
        assert e.id == "e1"
        assert e.description == ""
        assert e.properties == {}

    def test_full(self) -> None:
        e = EntityResponse(
            id="e2",
            label="Sellafield",
            entity_type="Site",
            description="Nuclear reprocessing site",
            properties={"country": "UK", "status": "decommissioning"},
        )
        assert e.properties["country"] == "UK"
        d = e.model_dump()
        assert "description" in d


class TestRelationResponse:
    """Validate RelationResponse model."""

    def test_defaults(self) -> None:
        r = RelationResponse(source_id="a", target_id="b", relation_type="LOCATED_AT")
        assert r.confidence == 0.0

    def test_with_confidence(self) -> None:
        r = RelationResponse(source_id="a", target_id="b", relation_type="PART_OF", confidence=0.95)
        assert r.confidence == 0.95


class TestEvidenceResponse:
    """Validate EvidenceResponse model."""

    def test_minimal(self) -> None:
        e = EvidenceResponse(text="Some evidence text")
        assert e.score == 0.0
        assert e.source == ""

    def test_full(self) -> None:
        e = EvidenceResponse(
            text="Evidence from vector search",
            score=0.92,
            source="vector",
            doc_id="doc-123",
            source_id="src-456",
        )
        assert e.doc_id == "doc-123"


class TestReasoningStepResponse:
    """Validate ReasoningStepResponse model."""

    def test_defaults(self) -> None:
        s = ReasoningStepResponse(step_id=1)
        assert s.sub_question == ""
        assert s.confidence == 0.0

    def test_full(self) -> None:
        s = ReasoningStepResponse(
            step_id=2,
            sub_question="What waste types?",
            evidence_text="HLW and ILW",
            answer_fragment="High-level and intermediate-level waste",
            confidence=0.85,
        )
        assert s.step_id == 2
        assert "HLW" in s.evidence_text


class TestVerificationResponse:
    """Validate VerificationResponse model."""

    def test_defaults(self) -> None:
        v = VerificationResponse()
        assert v.is_faithful is True
        assert v.faithfulness_score == 1.0
        assert v.entity_coverage == 0.0

    def test_with_claims(self) -> None:
        v = VerificationResponse(
            is_faithful=False,
            faithfulness_score=0.4,
            supported_claims=["claim1"],
            unsupported_claims=["claim2"],
            contradicted_claims=["claim3"],
            entity_coverage=0.75,
        )
        assert not v.is_faithful
        assert len(v.unsupported_claims) == 1


class TestGapDetectionResponse:
    """Validate GapDetectionResponse model."""

    def test_minimal(self) -> None:
        g = GapDetectionResponse(gap_type="abox_weak_evidence")
        assert g.description == ""
        assert g.affected_entities == []

    def test_full(self) -> None:
        g = GapDetectionResponse(
            gap_type="tbox_missing_class",
            description="No class for 'Vitrification' process",
            affected_entities=["Vitrification", "HLW"],
        )
        assert len(g.affected_entities) == 2


class TestChatResponseEnriched:
    """Validate the enriched ChatResponse with all context fields."""

    def test_full_response(self) -> None:
        resp = ChatResponse(
            session_id="s1",
            message=ChatMessage(role="assistant", content="Answer text"),
            confidence=0.85,
            latency_ms=1234.5,
            strategy_used="hybrid_sota",
            reasoning_chain=["step 1", "step 2"],
            reasoning_steps=[
                ReasoningStepResponse(step_id=1, sub_question="Q1"),
            ],
            evidence=[
                EvidenceResponse(text="Evidence 1", score=0.9, source="vector"),
            ],
            cited_entities=[
                EntityResponse(id="e1", label="Reactor", entity_type="Facility"),
            ],
            cited_relations=[
                RelationResponse(source_id="e1", target_id="e2", relation_type="PART_OF"),
            ],
            subgraph={"nodes": [], "edges": []},
            verification=VerificationResponse(is_faithful=True, faithfulness_score=0.95),
            gap_detection=GapDetectionResponse(gap_type="abox_weak_evidence"),
        )
        assert resp.strategy_used == "hybrid_sota"
        assert len(resp.evidence) == 1
        assert len(resp.cited_entities) == 1
        assert resp.verification is not None
        assert resp.verification.is_faithful
        assert resp.gap_detection is not None

        # Round-trip JSON serialization
        d = resp.model_dump()
        roundtrip = ChatResponse.model_validate(d)
        assert roundtrip.session_id == "s1"
        assert roundtrip.verification.faithfulness_score == 0.95
