"""API request/response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Incoming QA request."""

    question: str = Field(..., min_length=1, description="Natural-language question")
    strategy: str = Field(default="hybrid", description="Retrieval strategy")
    language: str = Field(default="de", description="Response language (de/en)")


class ProvenanceResponse(BaseModel):
    """Provenance info for a single evidence piece."""

    source: str
    score: float
    doc_id: str | None = None
    source_id: str | None = None
    entity_ids: list[str] = Field(default_factory=list)


class FactChainResponse(BaseModel):
    """A verified fact chain from KG path traversal."""

    source: str
    target: str
    chain_text: str = ""
    edges: list[dict[str, str]] = Field(default_factory=list)
    node_labels: dict[str, str] = Field(default_factory=dict)


class ToolTraceResponse(BaseModel):
    """A single tool call made by the agentic retriever."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    result_summary: str = ""
    iteration: int = 0


class AnswerResponse(BaseModel):
    """Full QA response with provenance."""

    question: str
    answer: str
    confidence: float
    reasoning_chain: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceResponse] = Field(default_factory=list)
    fact_chains: list[FactChainResponse] = Field(default_factory=list)
    tool_trace: list[ToolTraceResponse] = Field(default_factory=list)
    subgraph: dict[str, Any] | None = None
    latency_ms: float = 0.0


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = ""
