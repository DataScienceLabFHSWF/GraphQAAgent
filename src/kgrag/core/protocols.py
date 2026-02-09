"""Protocol definitions for pluggable retrieval, expansion, reranking, and reasoning."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kgrag.core.models import (
    GraphExplorationState,
    QAAnswer,
    QAQuery,
    ReasoningStep,
    RetrievedContext,
    VerificationResult,
)


@runtime_checkable
class Retriever(Protocol):
    """All retrieval strategies implement this interface."""

    async def retrieve(self, query: QAQuery) -> list[RetrievedContext]: ...


@runtime_checkable
class QueryExpander(Protocol):
    """Ontology-based query expansion."""

    def expand_query(self, query: QAQuery) -> QAQuery: ...


@runtime_checkable
class Reranker(Protocol):
    """Context reranking after retrieval."""

    def rerank(
        self,
        query: str,
        contexts: list[RetrievedContext],
        top_k: int = 5,
    ) -> list[RetrievedContext]: ...


@runtime_checkable
class GraphExplorer(Protocol):
    """Iterative graph exploration (Think-on-Graph style)."""

    async def explore(
        self,
        query: QAQuery,
        seed_entity_ids: list[str],
    ) -> tuple[list[RetrievedContext], GraphExplorationState]: ...


@runtime_checkable
class StepReasoner(Protocol):
    """Multi-hop chain-of-thought reasoning."""

    async def reason(
        self,
        query: QAQuery,
        contexts: list[RetrievedContext],
    ) -> list[ReasoningStep]: ...


@runtime_checkable
class AnswerVerifier(Protocol):
    """Post-generation answer faithfulness verification."""

    async def verify(
        self,
        answer: QAAnswer,
        contexts: list[RetrievedContext],
    ) -> VerificationResult: ...
