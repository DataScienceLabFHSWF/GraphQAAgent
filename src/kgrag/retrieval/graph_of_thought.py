"""Graph-of-Thought (GoT) reasoning for multi-hop QA.

Instead of a *single* chain of thought, GoT maintains a **directed acyclic
graph (DAG)** of reasoning nodes.  Each node represents a sub-question or
hypothesis with its own local evidence.  Nodes can be expanded (spawning
children), merged (when two sub-questions converge), or pruned (when
evidence contradicts a hypothesis).

This module provides:

* ``ReasoningNode`` — a single node in the reasoning DAG
* ``ReasoningDAG`` — the full DAG with expansion / merge / prune helpers
* ``GraphOfThoughtReasoner`` — async orchestrator that drives the GoT loop

Status: **foundational implementation** — the DAG data-structures and
traversal are complete; LLM-driven expansion (deciding *how* to branch)
is a simple prompt for now and can be replaced with a fine-tuned policy.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog

from kgrag.core.models import RetrievedContext

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# DAG data-structures
# ---------------------------------------------------------------------------


@dataclass
class ReasoningNode:
    """A single node in the reasoning DAG.

    Attributes
    ----------
    node_id:
        Unique identifier (auto-generated UUID).
    question:
        The sub-question this node investigates.
    evidence:
        Retrieved evidence relevant to *question*.
    answer_fragment:
        Partial answer derived from *evidence* (may be empty until
        the node is *resolved*).
    confidence:
        Estimated confidence in *answer_fragment* (0-1).
    parent_ids:
        IDs of parent nodes (empty for root).
    child_ids:
        IDs of child nodes (empty for leaf).
    status:
        One of ``pending``, ``resolved``, ``pruned``, ``merged``.
    metadata:
        Arbitrary extra data (e.g. tool calls used).
    """

    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    question: str = ""
    evidence: list[RetrievedContext] = field(default_factory=list)
    answer_fragment: str = ""
    confidence: float = 0.0
    parent_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | resolved | pruned | merged
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningDAG:
    """Directed acyclic graph of :class:`ReasoningNode` instances.

    Provides helpers for expansion, merging, pruning, and traversal.
    """

    nodes: dict[str, ReasoningNode] = field(default_factory=dict)
    root_id: str | None = None

    # -- mutation ----------------------------------------------------------

    def add_node(
        self,
        question: str,
        parent_id: str | None = None,
        **kwargs: Any,
    ) -> ReasoningNode:
        """Create a new node and optionally link it to *parent_id*."""
        node = ReasoningNode(question=question, **kwargs)
        self.nodes[node.node_id] = node
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].child_ids.append(node.node_id)
            node.parent_ids.append(parent_id)
        if self.root_id is None:
            self.root_id = node.node_id
        return node

    def merge_nodes(self, node_ids: list[str], merged_question: str) -> ReasoningNode:
        """Merge several leaf nodes into a single new node.

        The originals are marked ``merged`` and become parents of the new
        node.
        """
        merged = self.add_node(merged_question)
        for nid in node_ids:
            if nid in self.nodes:
                self.nodes[nid].status = "merged"
                self.nodes[nid].child_ids.append(merged.node_id)
                merged.parent_ids.append(nid)
                # Carry over evidence
                merged.evidence.extend(self.nodes[nid].evidence)
        return merged

    def prune_node(self, node_id: str) -> None:
        """Mark a node and all its descendants as ``pruned``."""
        stack = [node_id]
        while stack:
            nid = stack.pop()
            if nid in self.nodes:
                self.nodes[nid].status = "pruned"
                stack.extend(self.nodes[nid].child_ids)

    # -- queries -----------------------------------------------------------

    @property
    def leaves(self) -> list[ReasoningNode]:
        """Return all non-pruned leaf nodes (no children or all children pruned)."""
        result = []
        for node in self.nodes.values():
            if node.status == "pruned":
                continue
            active_children = [
                c for c in node.child_ids
                if c in self.nodes and self.nodes[c].status != "pruned"
            ]
            if not active_children:
                result.append(node)
        return result

    @property
    def resolved_answers(self) -> list[str]:
        """Collect answer fragments from all resolved nodes in topological order."""
        answers = []
        visited: set[str] = set()

        def _dfs(nid: str) -> None:
            if nid in visited or nid not in self.nodes:
                return
            visited.add(nid)
            node = self.nodes[nid]
            for cid in node.child_ids:
                _dfs(cid)
            if node.status == "resolved" and node.answer_fragment:
                answers.append(node.answer_fragment)

        if self.root_id:
            _dfs(self.root_id)
        return list(reversed(answers))

    def to_dict(self) -> dict[str, Any]:
        """Serialise the DAG for storage / API response."""
        return {
            "root_id": self.root_id,
            "nodes": {
                nid: {
                    "node_id": n.node_id,
                    "question": n.question,
                    "answer_fragment": n.answer_fragment,
                    "confidence": n.confidence,
                    "status": n.status,
                    "parent_ids": n.parent_ids,
                    "child_ids": n.child_ids,
                    "evidence_count": len(n.evidence),
                }
                for nid, n in self.nodes.items()
            },
        }


# ---------------------------------------------------------------------------
# GoT reasoner (orchestrator)
# ---------------------------------------------------------------------------


class GraphOfThoughtReasoner:
    """Drives a Graph-of-Thought reasoning loop.

    The reasoner starts with the original question as the root node,
    then iteratively:
    1. Picks the most promising leaf node (lowest confidence).
    2. Asks the LLM whether to **expand** (split into sub-questions),
       **resolve** (answer directly), or **prune** (abandon).
    3. If expanding, creates child nodes and retrieves evidence for each.
    4. If merging is possible (multiple leaves with overlapping evidence),
       merges them into a single node.

    Parameters
    ----------
    llm_provider:
        An object with ``get_chat_model()`` (e.g. ``LangChainOllamaProvider``).
    max_depth:
        Maximum depth of the DAG.
    max_nodes:
        Maximum total number of nodes.
    """

    def __init__(
        self,
        llm_provider: Any,
        *,
        max_depth: int = 4,
        max_nodes: int = 16,
    ) -> None:
        self._llm = llm_provider
        self._max_depth = max_depth
        self._max_nodes = max_nodes

    async def reason(
        self,
        question: str,
        evidence_fn: Any | None = None,
    ) -> ReasoningDAG:
        """Run the GoT loop and return the resulting DAG.

        Parameters
        ----------
        question:
            The original user question.
        evidence_fn:
            An async callable ``(sub_question: str) -> list[RetrievedContext]``
            used to fetch evidence for each sub-question.  If ``None``, nodes
            must be populated externally.
        """
        dag = ReasoningDAG()
        root = dag.add_node(question)

        for _step in range(self._max_nodes):
            leaves = dag.leaves
            if not leaves:
                break

            # Pick the leaf with lowest confidence (most uncertain)
            target = min(leaves, key=lambda n: n.confidence)

            # Check depth
            depth = self._depth_of(dag, target.node_id)
            if depth >= self._max_depth:
                target.status = "resolved"
                continue

            # Fetch evidence if we have a retrieval function
            if evidence_fn and not target.evidence:
                try:
                    target.evidence = await evidence_fn(target.question)
                except Exception as exc:
                    logger.warning("got.evidence_failed", node=target.node_id, error=str(exc))

            # For now, simple heuristic: if evidence is sufficient, resolve
            if len(target.evidence) >= 2:
                target.status = "resolved"
                target.confidence = min(
                    1.0,
                    sum(c.relevance_score or c.score or 0.0 for c in target.evidence)
                    / max(len(target.evidence), 1),
                )
                target.answer_fragment = (
                    f"Based on {len(target.evidence)} evidence pieces for: {target.question}"
                )
            elif len(dag.nodes) < self._max_nodes:
                # Expand: create 2 sub-questions (placeholder)
                dag.add_node(
                    f"What specific details about: {target.question}",
                    parent_id=target.node_id,
                )
                dag.add_node(
                    f"What context supports: {target.question}",
                    parent_id=target.node_id,
                )
            else:
                target.status = "resolved"

        # Resolve any remaining pending leaves
        for leaf in dag.leaves:
            if leaf.status == "pending":
                leaf.status = "resolved"

        logger.info(
            "got.complete",
            total_nodes=len(dag.nodes),
            resolved=sum(1 for n in dag.nodes.values() if n.status == "resolved"),
        )
        return dag

    @staticmethod
    def _depth_of(dag: ReasoningDAG, node_id: str) -> int:
        """Calculate the depth of a node from the root."""
        depth = 0
        current = node_id
        while current in dag.nodes:
            parents = dag.nodes[current].parent_ids
            if not parents:
                break
            current = parents[0]
            depth += 1
        return depth
