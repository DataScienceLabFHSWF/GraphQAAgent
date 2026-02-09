"""Custom exceptions for the KG-RAG Agent."""

from __future__ import annotations


class KGRAGError(Exception):
    """Base exception for all kgrag errors."""


class ConnectorError(KGRAGError):
    """Raised when an external service connector fails."""


class Neo4jConnectionError(ConnectorError):
    """Failed to connect to or query Neo4j."""


class QdrantConnectionError(ConnectorError):
    """Failed to connect to or query Qdrant."""


class FusekiConnectionError(ConnectorError):
    """Failed to connect to or query Fuseki."""


class OllamaConnectionError(ConnectorError):
    """Failed to connect to or query Ollama."""


class RetrievalError(KGRAGError):
    """Raised when a retrieval strategy fails."""


class EntityLinkingError(RetrievalError):
    """Failed to link question terms to KG entities."""


class QuestionParsingError(KGRAGError):
    """Failed to parse or classify a user question."""


class AnswerGenerationError(KGRAGError):
    """Failed to generate an answer from LLM."""


class ValidationError(KGRAGError):
    """Raised during KG validation (SHACL / CQ)."""


class EvaluationError(KGRAGError):
    """Raised during evaluation pipeline."""
