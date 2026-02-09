"""
agentic_reasoner_adapter.py
Adapter to integrate upstream agentic reasoning components with KG-RAG

This adapter bridges the gap between our KG-RAG system and the vendored
ReAct reasoning agent from agentic-reasoning-framework.
"""

import logging
from typing import Dict, Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel

from third_party.agentic_reasoning import ReasoningAgent, RetrieverTool
from ..core.models import DocumentChunk
from ..core.protocols import Retriever

logger = logging.getLogger(__name__)


class AgenticReasonerAdapter:
    """
    Adapter for integrating upstream ReAct reasoning agent with KG-RAG

    This adapter:
    1. Converts KG-RAG DocumentChunk objects to LangChain Document format
    2. Integrates with KG-RAG's retrieval system for additional document fetching
    3. Provides ReAct reasoning capabilities to KG-RAG agents
    """

    def __init__(
        self,
        ollama_provider: Any,  # LangChainOllamaProvider
        retriever: Retriever,
        max_iterations: int = 3,
        relevance_threshold: float = 0.3
    ):
        """
        Initialize the adapter

        Args:
            ollama_provider: KG-RAG LangChainOllamaProvider instance
            retriever: KG-RAG retriever for additional document fetching
            max_iterations: Maximum ReAct iterations
            relevance_threshold: Threshold for document relevance
        """
        self.ollama_provider = ollama_provider
        self.retriever = retriever
        self.max_iterations = max_iterations
        self.relevance_threshold = relevance_threshold

        # Get the LangChain chat model directly
        self.llm = ollama_provider.get_chat_model()

        # Initialize the adapted retriever tool
        self.retriever_tool = KGRAGRetrieverTool(
            retriever=retriever,
            relevance_threshold=relevance_threshold
        )

        # Initialize the reasoning agent with our custom retriever tool
        self.reasoning_agent = ReasoningAgent(
            llm=self.llm,
            chroma_dir="",  # Not used in our integration
            processed_dir="",  # Not used in our integration
            max_iterations=max_iterations,
            relevance_threshold=relevance_threshold
        )

        # Override the retriever tool with our KG-RAG integrated version
        self.reasoning_agent.retriever_tool = self.retriever_tool

        # Try to setup tools - if it fails, we'll fall back to CoT reasoning
        try:
            self.reasoning_agent.llm_with_tools = self.reasoning_agent._setup_tools()
            self.tool_binding_available = self.reasoning_agent.llm_with_tools is not None
        except Exception as e:
            logger.warning(f"Tool binding failed, falling back to CoT reasoning: {e}")
            self.tool_binding_available = False
            self.reasoning_agent.llm_with_tools = None

        logger.info(f"AgenticReasonerAdapter initialized with KG-RAG integration (tool binding: {self.tool_binding_available})")

    async def reason_over_documents(
        self,
        query: str,
        initial_chunks: List[DocumentChunk]
    ) -> Dict[str, Any]:
        """
        Perform ReAct reasoning over retrieved document chunks

        Args:
            query: The user's question
            initial_chunks: Initial retrieved document chunks from KG-RAG

        Returns:
            Dict containing reasoning results with KG-RAG compatible format
        """
        logger.info(f"Starting ReAct reasoning for query: '{query[:50]}...'")

        # Check if tool binding is available
        if not self.tool_binding_available:
            logger.info("Tool binding not available, falling back to CoT reasoning")
            return await self._fallback_cot_reasoning(query, initial_chunks)

        # Convert KG-RAG DocumentChunks to LangChain Documents
        initial_docs = self._convert_chunks_to_documents(initial_chunks)

        # Create state dict compatible with reasoning agent
        state = {
            "query": query,
            "retrieved_docs": initial_docs,
            "tool_calls": [],
            "follow_up_questions": [],
            "additional_context": [],
            "workflow_metadata": {}
        }

        # Execute reasoning
        result = self.reasoning_agent.reason_over_documents(state)

        # Convert back to KG-RAG compatible format
        kgrag_result = self._convert_result_to_kgrag_format(result, initial_chunks)

        logger.info(f"ReAct reasoning completed: {len(result.get('followup_questions', []))} follow-ups, "
                   f"{result.get('additional_retrieved_context', 0)} additional docs")

        return kgrag_result

    async def _fallback_cot_reasoning(
        self,
        query: str,
        initial_chunks: List[DocumentChunk]
    ) -> Dict[str, Any]:
        """
        Fallback to Chain-of-Thought reasoning when tool binding is not available

        Args:
            query: The user's question
            initial_chunks: Initial retrieved document chunks

        Returns:
            Dict containing reasoning results in KG-RAG format
        """
        logger.info("Using CoT fallback reasoning")

        # Format documents for context
        context = "\n\n".join([f"Document {i+1}: {chunk.content}" for i, chunk in enumerate(initial_chunks)])

        # Simple CoT prompt
        cot_prompt = f"""Please reason step-by-step to answer the following question based on the provided documents.

Question: {query}

Documents:
{context}

Reasoning steps:
1. Analyze the question and identify key requirements
2. Review relevant information from the documents
3. Synthesize the information to form a coherent answer
4. Provide evidence from the documents to support the answer

Final Answer:"""

        # Use the LLM directly (without tools)
        try:
            from langchain_core.messages import HumanMessage
            response = await self.llm.invoke([HumanMessage(content=cot_prompt)])
            reasoning_text = response.content
        except Exception as e:
            logger.error(f"Failed to get LLM response: {e}")
            reasoning_text = "Unable to perform reasoning due to LLM error."

        # Extract final answer (simple heuristic)
        final_answer = reasoning_text
        if "Final Answer:" in reasoning_text:
            final_answer = reasoning_text.split("Final Answer:")[-1].strip()

        # Return in KG-RAG format
        return {
            "reasoning": reasoning_text,
            "answer": final_answer,
            "confidence": 0.7,  # Default confidence for fallback
            "additional_chunks": [],  # No additional retrieval in fallback
            "followup_questions": [],
            "tool_calls": [],
            "reasoning_strategy": "cot_fallback",
            "metadata": {
                "fallback_reason": "tool_binding_unavailable",
                "tool_binding_available": False
            }
        }

    def _convert_chunks_to_documents(self, chunks: List[DocumentChunk]) -> List[Any]:
        """
        Convert KG-RAG DocumentChunk objects to LangChain Document format

        Args:
            chunks: KG-RAG document chunks

        Returns:
            List of LangChain Document objects
        """
        from langchain_core.documents import Document

        documents = []
        for chunk in chunks:
            # Create metadata similar to what the upstream system expects
            metadata = {
                'filename': f"doc_{chunk.doc_id}",
                'score': getattr(chunk, 'score', 1.0),  # Default score if not present
                'doc_id': chunk.doc_id,
                'chunk_id': chunk.id
            }

            # Create LangChain Document
            doc = Document(
                page_content=chunk.content,
                metadata=metadata
            )
            documents.append(doc)

        return documents

    def _convert_result_to_kgrag_format(
        self,
        result: Dict[str, Any],
        initial_chunks: List[DocumentChunk]
    ) -> Dict[str, Any]:
        """
        Convert reasoning agent result to KG-RAG compatible format

        Args:
            result: Result from reasoning agent
            initial_chunks: Original KG-RAG chunks

        Returns:
            KG-RAG compatible result dict
        """
        # Extract additional chunks retrieved during reasoning
        additional_chunks = []
        if result.get("additional_context"):
            for additional_doc in result["additional_context"]:
                # Convert back to DocumentChunk format
                # Note: This is a simplified conversion - in practice you'd need
                # to map back to actual DocumentChunk objects from your retrieval system
                chunk = DocumentChunk(
                    id=f"additional_{len(additional_chunks)}",
                    doc_id=additional_doc.get("source", "unknown"),
                    content=additional_doc.get("content", ""),
                    strategy="reasoning_retrieval"
                )
                additional_chunks.append(chunk)

        return {
            "reasoning_answer": result.get("reasoning_answer", ""),
            "followup_questions": result.get("followup_questions", []),
            "additional_chunks": additional_chunks,
            "tool_calls": result.get("tool_calls", []),
            "total_documents_used": len(result.get("retrieved_docs", [])),
            "reasoning_metadata": {
                "iterations_performed": result.get("workflow_metadata", {}).get("total_follow_up_questions", 0),
                "additional_context_retrieved": result.get("additional_retrieved_context", 0)
            }
        }


class KGRAGRetrieverTool(RetrieverTool):
    """
    KG-RAG integrated retriever tool

    This replaces the original RetrieverTool to work with KG-RAG's retrieval system
    instead of the upstream ChromaDB-based system.
    """

    def __init__(self, retriever: Retriever, relevance_threshold: float = 0.3):
        # Don't call super().__init__() since we don't need ChromaDB paths
        self.retriever = retriever
        self.relevance_threshold = relevance_threshold
        self.k = 3  # Default number of documents to retrieve

    def retrieve(self, query: str) -> List[Any]:
        """
        Retrieve documents using KG-RAG retrieval system

        Args:
            query: Search query

        Returns:
            List of LangChain Document objects
        """
        try:
            logger.info(f"KG-RAG RetrieverTool retrieving for: '{query[:50]}...'")

            # Use KG-RAG retrieval system
            # This assumes the retriever has a retrieve method that takes a query
            # and returns DocumentChunk objects
            retrieved_chunks = self.retriever.retrieve(query, k=self.k)

            # Convert to LangChain Document format
            from langchain_core.documents import Document

            documents = []
            for chunk in retrieved_chunks:
                metadata = {
                    'filename': f"doc_{chunk.doc_id}",
                    'score': getattr(chunk, 'score', 1.0),
                    'doc_id': chunk.doc_id,
                    'chunk_id': chunk.id
                }

                doc = Document(
                    page_content=chunk.content,
                    metadata=metadata
                )
                documents.append(doc)

            # Filter by relevance threshold (if scores are available)
            relevant_docs = []
            for doc in documents:
                score = doc.metadata.get('score', 1.0)
                if score >= self.relevance_threshold:
                    relevant_docs.append(doc)

            logger.info(f"KG-RAG RetrieverTool found {len(relevant_docs)} relevant documents")
            return relevant_docs

        except Exception as e:
            logger.error(f"KG-RAG RetrieverTool error: {e}")
            return []