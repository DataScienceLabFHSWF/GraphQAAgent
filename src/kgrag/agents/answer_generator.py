"""AnswerGenerator (C3.4.3) — LLM-based answer generation with grounded citations.

Generates answers using ONLY the provided context, with citation markers
so the Explainer can trace each claim back to its source.
"""

from __future__ import annotations

import time

import structlog

from kgrag.connectors.langchain_ollama_provider import LangChainOllamaProvider
from kgrag.core.domain import DomainConfig
from kgrag.core.exceptions import AnswerGenerationError
from kgrag.core.models import QAAnswer, QAQuery, RetrievedContext
from kgrag.agents.context_assembler import ContextAssembler

logger = structlog.get_logger(__name__)

# Default prompt used when no DomainConfig is provided
_DEFAULT_SYSTEM_PROMPT = """\
You are a QA agent for domain knowledge.
Answer the question using ONLY the provided context.
Cite your sources using [Source:N] markers (where N is the context number).
If the context does not contain enough information, say so explicitly.
Be precise and concise."""


class AnswerGenerator:
    """Generate grounded answers from retrieved context via Ollama LLM."""

    def __init__(
        self,
        ollama: LangChainOllamaProvider,
        context_assembler: ContextAssembler | None = None,
        domain_config: DomainConfig | None = None,
    ) -> None:
        self._ollama = ollama
        self._assembler = context_assembler or ContextAssembler()
        self._domain = domain_config

    @property
    def _system_prompt(self) -> str:
        if self._domain:
            rendered = self._domain.render_prompt("answer_generator")
            if rendered:
                return rendered
        return _DEFAULT_SYSTEM_PROMPT

    async def generate(
        self,
        query: QAQuery,
        contexts: list[RetrievedContext],
        *,
        cot_summary: str | None = None,
    ) -> QAAnswer:
        """Generate an answer grounded in *contexts*.

        Args:
            query: Parsed question.
            contexts: Retrieved evidence.
            cot_summary: Optional Chain-of-Thought reasoning summary to
                prepend as additional context for the LLM.

        Returns a :class:`QAAnswer` with ``answer_text`` and ``latency_ms``
        populated.  The ``evidence`` field carries the contexts used.
        """
        t0 = time.perf_counter()

        assembled_context = self._assembler.assemble(contexts)

        # Prepend CoT reasoning if available
        cot_block = ""
        if cot_summary:
            cot_block = (
                f"Step-by-step reasoning:\n{cot_summary}\n\n"
            )

        prompt = (
            f"Context:\n{assembled_context}\n\n"
            f"{cot_block}"
            f"Question: {query.raw_question}\n\n"
            f"Answer (in {query.language}):"
        )

        try:
            answer_text = await self._ollama.generate(
                prompt=prompt,
                system=self._system_prompt,
            )
        except Exception as exc:
            raise AnswerGenerationError(f"LLM generation failed: {exc}") from exc

        elapsed = (time.perf_counter() - t0) * 1000

        return QAAnswer(
            question=query.raw_question,
            answer_text=answer_text.strip(),
            evidence=contexts,
            latency_ms=elapsed,
        )
