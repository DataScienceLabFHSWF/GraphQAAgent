"""Answer verifier — KG-grounded faithfulness checking (C3.4.7).

**SOTA technique** inspired by Graph-Constrained Reasoning (Luo et al. 2025,
ICML).  GCR integrates graph structure into LLM decoding to eliminate
hallucinations.  Our post-generation verifier achieves a similar effect by:

1. Extracting factual claims from the generated answer.
2. Checking each claim against the retrieved KG subgraph.
3. Flagging unsupported or contradicted claims.
4. Computing a grounded faithfulness score.

This is a lighter-weight alternative to KG-Trie constrained decoding that
works with any LLM (including via API) and provides more explainable feedback.
"""

from __future__ import annotations

import json
import re

import structlog

from kgrag.connectors.langchain_ollama_provider import LangChainOllamaProvider
from kgrag.core.models import (
    KGEntity,
    KGRelation,
    QAAnswer,
    RetrievedContext,
    VerificationResult,
)

logger = structlog.get_logger(__name__)

_CLAIM_EXTRACTION_PROMPT = """\
Extract factual claims from the following answer. Each claim should be an \
atomic statement that can be verified against a knowledge graph.

Answer: {answer_text}

Return a JSON array of strings, each being one factual claim.
Return ONLY valid JSON array."""

_CLAIM_VERIFICATION_PROMPT = """\
Verify whether the following claim is supported by the knowledge graph evidence.

Claim: {claim}

Knowledge graph evidence:
{evidence}

Respond with ONLY valid JSON:
{{"verdict": "supported" | "unsupported" | "contradicted", "reason": "..."}}"""


class AnswerVerifier:
    """Post-generation answer verification against KG evidence.

    Implements a lightweight version of GCR's faithfulness guarantee:
    instead of constraining decoding, we verify post-hoc and provide
    detailed feedback on which claims are / aren't grounded.
    """

    def __init__(
        self,
        ollama: LangChainOllamaProvider,
        *,
        claim_threshold: float = 0.5,
    ) -> None:
        self._ollama = ollama
        self._claim_threshold = claim_threshold

    async def verify(
        self,
        answer: QAAnswer,
        contexts: list[RetrievedContext],
    ) -> VerificationResult:
        """Verify the answer's faithfulness against retrieved KG evidence.

        Returns a :class:`VerificationResult` with supported/unsupported claims
        and an overall faithfulness score.
        """
        # 1. Extract claims from the answer
        claims = await self._extract_claims(answer.answer_text)
        if not claims:
            return VerificationResult(
                is_faithful=True,
                faithfulness_score=1.0,
                entity_coverage=self._compute_entity_coverage(answer, contexts),
            )

        # 2. Build evidence text from contexts
        evidence_text = self._build_evidence_text(contexts)

        # 3. Verify each claim
        supported: list[str] = []
        unsupported: list[str] = []
        contradicted: list[str] = []

        for claim in claims:
            verdict = await self._verify_claim(claim, evidence_text)
            if verdict == "supported":
                supported.append(claim)
            elif verdict == "contradicted":
                contradicted.append(claim)
            else:
                unsupported.append(claim)

        # 4. Compute faithfulness score
        total = len(claims)
        faithfulness = len(supported) / total if total > 0 else 1.0

        # 5. Entity coverage: fraction of answer-mentioned entities found in KG
        entity_coverage = self._compute_entity_coverage(answer, contexts)

        result = VerificationResult(
            is_faithful=faithfulness >= self._claim_threshold and not contradicted,
            supported_claims=supported,
            unsupported_claims=unsupported,
            contradicted_claims=contradicted,
            faithfulness_score=faithfulness,
            entity_coverage=entity_coverage,
        )

        logger.info(
            "verifier.done",
            total_claims=total,
            supported=len(supported),
            unsupported=len(unsupported),
            contradicted=len(contradicted),
            faithfulness=round(faithfulness, 2),
            entity_coverage=round(entity_coverage, 2),
        )
        return result

    async def _extract_claims(self, answer_text: str) -> list[str]:
        """Extract atomic factual claims from the answer via LLM."""
        # Strip citation markers before claim extraction
        clean_text = re.sub(r"\[Source:\d+\]", "", answer_text).strip()
        if not clean_text:
            return []

        prompt = _CLAIM_EXTRACTION_PROMPT.format(answer_text=clean_text)
        try:
            response = await self._ollama.generate(
                prompt=prompt,
                temperature=0.1,
                format="json",
            )
            claims = json.loads(response)
            if isinstance(claims, list):
                return [str(c) for c in claims if c]
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("verifier.claim_extraction_failed", error=str(exc))

        # Fallback: split answer into sentences as claims
        sentences = [
            s.strip() for s in re.split(r"[.!?]+", clean_text)
            if len(s.strip()) > 10
        ]
        return sentences

    async def _verify_claim(self, claim: str, evidence: str) -> str:
        """Verify a single claim against evidence. Returns verdict string."""
        prompt = _CLAIM_VERIFICATION_PROMPT.format(claim=claim, evidence=evidence)
        try:
            response = await self._ollama.generate(
                prompt=prompt,
                temperature=0.1,
                format="json",
            )
            result = json.loads(response)
            verdict = result.get("verdict", "unsupported")
            if verdict in ("supported", "unsupported", "contradicted"):
                return verdict
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("verifier.claim_check_failed", claim=claim[:50], error=str(exc))

        # Fallback: simple keyword overlap check
        return self._keyword_verify(claim, evidence)

    @staticmethod
    def _keyword_verify(claim: str, evidence: str) -> str:
        """Fast keyword-based claim verification fallback."""
        claim_words = set(claim.lower().split())
        evidence_lower = evidence.lower()

        # Remove common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "of", "for", "and", "or"}
        claim_words -= stopwords

        if not claim_words:
            return "unsupported"

        overlap = sum(1 for w in claim_words if w in evidence_lower)
        ratio = overlap / len(claim_words)

        if ratio >= 0.6:
            return "supported"
        return "unsupported"

    @staticmethod
    def _build_evidence_text(contexts: list[RetrievedContext]) -> str:
        """Build a combined evidence text from all contexts."""
        parts: list[str] = []
        for ctx in contexts:
            parts.append(ctx.text[:500])
            if ctx.subgraph:
                for el in ctx.subgraph:
                    if isinstance(el, KGEntity):
                        parts.append(f'Entity: "{el.label}" ({el.entity_type})')
                    elif isinstance(el, KGRelation):
                        parts.append(
                            f'Relation: "{el.source_id}" --[{el.relation_type}]--> "{el.target_id}"'
                        )
        return "\n".join(parts)

    @staticmethod
    def _compute_entity_coverage(
        answer: QAAnswer,
        contexts: list[RetrievedContext],
    ) -> float:
        """Compute what fraction of entities mentioned in the answer exist in KG evidence."""
        # Collect all entity labels from contexts
        kg_labels: set[str] = set()
        for ctx in contexts:
            if ctx.subgraph:
                for el in ctx.subgraph:
                    if isinstance(el, KGEntity):
                        kg_labels.add(el.label.lower())

        if not kg_labels:
            return 0.0

        # Check how many cited entity labels appear in KG evidence
        cited_labels = {e.label.lower() for e in answer.cited_entities}
        if not cited_labels:
            return 0.0

        covered = cited_labels & kg_labels
        return len(covered) / len(cited_labels)
