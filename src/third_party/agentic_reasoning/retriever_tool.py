"""
retriever_tool.py
Tool wrapper for the retrieval functionality to be used by ReAct reasoning agent

Adapted for KG-RAG integration - original from agentic-reasoning-framework
"""

import logging
from typing import List, Any, Dict
from langchain_core.tools import Tool
from langchain_core.documents import Document
import sys
import os

logger = logging.getLogger(__name__)


class RetrieverTool:
    """Tool wrapper for document retrieval functionality"""

    def __init__(
        self,
        chroma_dir: str,
        processed_dir: str,
        k: int = 2,
        relevance_threshold: float = 0.3
    ):
        self.chroma_dir = chroma_dir
        self.processed_dir = processed_dir
        self.k = k
        self.relevance_threshold = relevance_threshold

        logger.info(f"RetrieverTool initialized with k={k}, threshold={relevance_threshold}")

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for the given query

        Args:
            query: The search query

        Returns:
            List of relevant documents that meet the threshold
        """
        try:
            logger.info(f"RetrieverTool retrieving for: '{query[:50]}...'")

            # NOTE: This is a placeholder - in the original implementation,
            # this would call hybrid_retrieve from rag_utils.retriever
            # For KG-RAG integration, this will be replaced with our retrieval system

            # Placeholder implementation - return empty list for now
            # This will be replaced in the adapter
            retrieved_docs = []

            # Filter by relevance threshold
            relevant_docs = []
            for doc in retrieved_docs:
                score = getattr(doc, 'metadata', {}).get('score', 0.0)
                if score >= self.relevance_threshold:
                    relevant_docs.append(doc)

            logger.info(f"RetrieverTool found {len(relevant_docs)} relevant documents (threshold: {self.relevance_threshold})")
            return relevant_docs

        except Exception as e:
            logger.error(f"RetrieverTool error: {e}")
            return []

    def as_langchain_tool(self) -> Tool:
        """Convert to LangChain Tool for use with tool-calling models"""
        def _retrieve_wrapper(query: str) -> str:
            """Wrapper function for LangChain Tool"""
            docs = self.retrieve(query)

            if not docs:
                return "Keine relevanten Dokumente gefunden."

            # Format documents for tool response
            result = f"Gefunden: {len(docs)} relevante Dokumente\n\n"
            for i, doc in enumerate(docs, 1):
                source = getattr(doc, 'metadata', {}).get('filename', f'Dokument {i}')
                content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                score = getattr(doc, 'metadata', {}).get('score', 0.0)
                result += f"Dokument {i} ({source}, Score: {score:.3f}):\n{content}\n\n"

            return result

        return Tool(
            name="retrieve_documents",
            description="Suche nach relevanten Dokumenten in der deutschen Nuklear-Wissensdatenbank. "
                       "Verwende spezifische Begriffe wie Kraftwerksnamen, technische Begriffe oder Genehmigungsverfahren.",
            func=_retrieve_wrapper
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Get tool metadata for debugging/monitoring"""
        return {
            "chroma_dir": self.chroma_dir,
            "processed_dir": self.processed_dir,
            "k": self.k,
            "relevance_threshold": self.relevance_threshold
        }