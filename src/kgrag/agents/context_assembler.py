"""ContextAssembler (C3.4.2) — Merge retrieval contexts into a unified prompt section.

Takes contexts from multiple retrievers, deduplicates, and produces a
serialised context block with source labels for the LLM prompt.
"""

from __future__ import annotations

import structlog

from kgrag.core.models import RetrievedContext

logger = structlog.get_logger(__name__)


class ContextAssembler:
    """Merge and serialise retrieved contexts for LLM consumption.

    Deduplicates overlapping chunks, attaches source labels, and enforces
    a maximum token budget.
    """

    def __init__(self, *, max_context_chars: int = 12_000) -> None:
        self._max_chars = max_context_chars

    def assemble(self, contexts: list[RetrievedContext]) -> str:
        """Return a single string ready to be injected into the LLM prompt.

        Each context piece is prefixed with its source label and score so the
        LLM (and any downstream explainer) can attribute answers.
        """
        seen_texts: set[str] = set()
        blocks: list[str] = []
        total_chars = 0

        for i, ctx in enumerate(contexts, start=1):
            fingerprint = ctx.text[:200]
            if fingerprint in seen_texts:
                continue
            seen_texts.add(fingerprint)

            source_label = ctx.source.value.upper()
            provenance_tag = ""
            if ctx.provenance:
                parts: list[str] = []
                if ctx.provenance.source_id:
                    parts.append(f"Doc:{ctx.provenance.source_id}")
                if ctx.provenance.entity_ids:
                    parts.append(f"Entities:{','.join(ctx.provenance.entity_ids[:3])}")
                provenance_tag = f" [{', '.join(parts)}]" if parts else ""

            header = f"[{i}] ({source_label}, score={ctx.score:.3f}){provenance_tag}"
            block = f"{header}\n{ctx.text}"

            if total_chars + len(block) > self._max_chars:
                break

            blocks.append(block)
            total_chars += len(block)

        assembled = "\n\n".join(blocks)
        logger.info(
            "context_assembler.done",
            total_contexts=len(contexts),
            included=len(blocks),
            chars=total_chars,
        )
        return assembled
