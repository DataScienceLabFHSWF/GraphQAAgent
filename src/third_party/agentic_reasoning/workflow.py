"""
workflow.py
Simplified workflow definition for KG-RAG integration

Adapted from agentic-reasoning-framework - only includes reasoning agent functionality
"""

import logging
import os
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class SimplifiedRAGWorkflow:
    """Simplified workflow for KG-RAG with reasoning agent integration"""

    def __init__(
        self,
        reasoning_agent,  # Our adapted reasoning agent
        chroma_dir: str,
        processed_dir: str,
        max_iterations: int = 3,
        relevance_threshold: float = 0.3
    ):
        self.reasoning_agent = reasoning_agent
        self.chroma_dir = chroma_dir
        self.processed_dir = processed_dir
        self.max_iterations = max_iterations
        self.relevance_threshold = relevance_threshold

        logger.info("Simplified RAG workflow initialized with reasoning agent")

    def execute_reasoning(self, query: str, retrieved_docs: List[Any]) -> Dict[str, Any]:
        """Execute the reasoning workflow with retrieved documents"""

        # Create initial state similar to ChatState
        initial_state = {
            "query": query,
            "retrieved_docs": retrieved_docs,
            "tool_calls": [],
            "follow_up_questions": [],
            "additional_context": [],
            "workflow_metadata": {}
        }

        print("\n" + "="*80)
        print("🚀 STARTING REASONING WORKFLOW")
        print("="*80)
        print(f"📝 User Query: {query}")
        print(f"📚 Initial Documents: {len(retrieved_docs)}")
        print("-" * 80)

        # Execute reasoning
        result = self.reasoning_agent.reason_over_documents(initial_state)

        print("\n" + "="*80)
        print("✅ REASONING WORKFLOW COMPLETED")
        print("="*80)

        # Show results summary
        if result.get("reasoning_answer"):
            print(f"   🧠 Reasoning Answer: {len(result.get('reasoning_answer', ''))} characters")
        if result.get("followup_questions"):
            print(f"   🤔 Follow-up Questions: {len(result.get('followup_questions', []))}")
        if result.get("additional_context"):
            print(f"   📚 Additional Context: {len(result.get('additional_context', []))} documents")

        print("="*80)

        return result

    def get_workflow_info(self) -> Dict[str, Any]:
        """Get workflow metadata"""
        return {
            "workflow_type": "simplified_rag_with_reasoning",
            "max_iterations": self.max_iterations,
            "relevance_threshold": self.relevance_threshold,
            "has_reasoning_agent": True
        }