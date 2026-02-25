"""QuestionParser (C3.4.1) — Decompose and classify user questions.

Uses an LLM call with structured output to classify question type, extract
entity mentions, and decompose complex questions into sub-questions.
"""

from __future__ import annotations

import json

import structlog

from kgrag.connectors.langchain_ollama_provider import LangChainOllamaProvider
from kgrag.core.domain import DomainConfig
from kgrag.core.exceptions import QuestionParsingError
from kgrag.core.models import QAQuery, QuestionType

logger = structlog.get_logger(__name__)

# Default prompt used when no DomainConfig is provided
_DEFAULT_SYSTEM_PROMPT = """\
You are a question analysis expert for a knowledge graph.
Given a user question, produce a JSON object with EXACTLY these fields:
{
  "question_type": one of "factoid", "list", "boolean", "comparative", "causal", "aggregation",
  "detected_entities": list of entity names mentioned in the question,
  "detected_types": list of ontology class names (e.g. "Entity", "Concept"),
  "sub_questions": list of simpler sub-questions (empty if the question is already simple),
  "language": "de" or "en"
}
Return ONLY valid JSON, no explanation."""


class QuestionParser:
    """Parse, classify, and decompose user questions using an LLM."""

    def __init__(
        self,
        ollama: LangChainOllamaProvider,
        domain_config: DomainConfig | None = None,
    ) -> None:
        self._ollama = ollama
        self._domain = domain_config

    @property
    def _system_prompt(self) -> str:
        if self._domain:
            rendered = self._domain.render_prompt("question_parser")
            if rendered:
                return rendered
        return _DEFAULT_SYSTEM_PROMPT

    async def parse(self, raw_question: str) -> QAQuery:
        """Analyse *raw_question* and return a structured :class:`QAQuery`."""
        try:
            response = await self._ollama.generate(
                prompt=f"Analyse this question:\n\n{raw_question}",
                system=self._system_prompt,
                temperature=0.1,
                format="json",
            )
            data = json.loads(response)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("question_parser.llm_parse_failed", error=str(exc))
            # Fallback: return a minimal QAQuery
            return QAQuery(raw_question=raw_question)
        except Exception as exc:
            raise QuestionParsingError(f"Question parsing failed: {exc}") from exc

        # Map to QuestionType enum
        qt_str = data.get("question_type", "")
        try:
            question_type = QuestionType(qt_str)
        except ValueError:
            question_type = None

        return QAQuery(
            raw_question=raw_question,
            question_type=question_type,
            detected_entities=data.get("detected_entities", []),
            detected_types=data.get("detected_types", []),
            sub_questions=data.get("sub_questions", []),
            language=data.get("language", "de"),
        )
