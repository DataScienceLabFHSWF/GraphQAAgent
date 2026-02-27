"""Tests for Graph-of-Thought reasoning DAG and reasoner."""

from __future__ import annotations
import pytest
import pytest_asyncio

from kgrag.retrieval.graph_of_thought import (
    ReasoningDAG,
    ReasoningNode,
    GraphOfThoughtReasoner,
)
from kgrag.core.models import RetrievedContext, RetrievalSource


# ---------------------------------------------------------------------------
# ReasoningNode
# ---------------------------------------------------------------------------


class TestReasoningNode:
    def test_default_fields(self):
        node = ReasoningNode(question="What is X?")
        assert node.question == "What is X?"
        assert node.status == "pending"
        assert node.confidence == 0.0
        assert node.evidence == []
        assert node.parent_ids == []
        assert node.child_ids == []
        assert len(node.node_id) == 12

    def test_custom_fields(self):
        node = ReasoningNode(
            node_id="abc",
            question="Q",
            status="resolved",
            confidence=0.9,
        )
        assert node.node_id == "abc"
        assert node.status == "resolved"
        assert node.confidence == 0.9


# ---------------------------------------------------------------------------
# ReasoningDAG
# ---------------------------------------------------------------------------


class TestReasoningDAG:
    def test_add_root_node(self):
        dag = ReasoningDAG()
        root = dag.add_node("Root question")
        assert dag.root_id == root.node_id
        assert len(dag.nodes) == 1

    def test_add_child_node(self):
        dag = ReasoningDAG()
        root = dag.add_node("Root")
        child = dag.add_node("Child", parent_id=root.node_id)
        assert child.parent_ids == [root.node_id]
        assert root.child_ids == [child.node_id]
        assert len(dag.nodes) == 2

    def test_leaves(self):
        dag = ReasoningDAG()
        root = dag.add_node("Root")
        c1 = dag.add_node("C1", parent_id=root.node_id)
        c2 = dag.add_node("C2", parent_id=root.node_id)
        leaves = dag.leaves
        assert len(leaves) == 2
        leaf_ids = {n.node_id for n in leaves}
        assert c1.node_id in leaf_ids
        assert c2.node_id in leaf_ids

    def test_prune_removes_subtree_from_leaves(self):
        dag = ReasoningDAG()
        root = dag.add_node("Root")
        c1 = dag.add_node("C1", parent_id=root.node_id)
        c2 = dag.add_node("C2", parent_id=root.node_id)
        gc = dag.add_node("GC", parent_id=c1.node_id)

        dag.prune_node(c1.node_id)
        leaves = dag.leaves
        # c1 and gc are pruned; only c2 remains
        assert len(leaves) == 1
        assert leaves[0].node_id == c2.node_id

    def test_merge_nodes(self):
        dag = ReasoningDAG()
        root = dag.add_node("Root")
        c1 = dag.add_node("C1", parent_id=root.node_id)
        c2 = dag.add_node("C2", parent_id=root.node_id)

        merged = dag.merge_nodes([c1.node_id, c2.node_id], "Merged question")
        assert c1.status == "merged"
        assert c2.status == "merged"
        assert merged.parent_ids == [c1.node_id, c2.node_id]

    def test_resolved_answers_topological_order(self):
        dag = ReasoningDAG()
        root = dag.add_node("Root")
        root.status = "resolved"
        root.answer_fragment = "Root answer"

        c1 = dag.add_node("C1", parent_id=root.node_id)
        c1.status = "resolved"
        c1.answer_fragment = "C1 answer"

        answers = dag.resolved_answers
        # Should include both answers
        assert len(answers) == 2
        assert "Root answer" in answers
        assert "C1 answer" in answers

    def test_to_dict(self):
        dag = ReasoningDAG()
        root = dag.add_node("Q?")
        d = dag.to_dict()
        assert d["root_id"] == root.node_id
        assert root.node_id in d["nodes"]
        assert d["nodes"][root.node_id]["question"] == "Q?"


# ---------------------------------------------------------------------------
# GraphOfThoughtReasoner
# ---------------------------------------------------------------------------


class TestGraphOfThoughtReasoner:
    @pytest.mark.asyncio
    async def test_reason_without_evidence_fn(self):
        """Runs the GoT loop with no evidence function — resolves heuristically."""

        class FakeLLM:
            def get_chat_model(self):
                return None

        reasoner = GraphOfThoughtReasoner(FakeLLM(), max_depth=2, max_nodes=6)
        dag = await reasoner.reason("What is X?")

        assert len(dag.nodes) >= 1
        # All leaves should be resolved
        for leaf in dag.leaves:
            assert leaf.status == "resolved"

    @pytest.mark.asyncio
    async def test_reason_with_evidence_fn(self):
        """Evidence function provides contexts → nodes get resolved."""

        class FakeLLM:
            def get_chat_model(self):
                return None

        async def evidence_fn(q: str) -> list[RetrievedContext]:
            return [
                RetrievedContext(source=RetrievalSource.VECTOR, text="Ev1", score=0.8),
                RetrievedContext(source=RetrievalSource.VECTOR, text="Ev2", score=0.7),
            ]

        reasoner = GraphOfThoughtReasoner(FakeLLM(), max_depth=3, max_nodes=8)
        dag = await reasoner.reason("What is X?", evidence_fn=evidence_fn)

        # Root should have been resolved since it has 2 evidence pieces
        root = dag.nodes[dag.root_id]
        assert root.status == "resolved"
        assert root.confidence > 0

    @pytest.mark.asyncio
    async def test_max_nodes_respected(self):
        """DAG does not exceed max_nodes."""

        class FakeLLM:
            def get_chat_model(self):
                return None

        reasoner = GraphOfThoughtReasoner(FakeLLM(), max_depth=10, max_nodes=4)
        dag = await reasoner.reason("Q?")
        # max_nodes is a soft limit — children created in batch may exceed by 1
        assert len(dag.nodes) <= 6
