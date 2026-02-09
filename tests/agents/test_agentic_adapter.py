"""
Tests for the agentic reasoner adapter
"""

import pytest
from unittest.mock import Mock, MagicMock
from langchain_core.language_models.chat_models import BaseChatModel

from kgrag.adapters.agentic_reasoner_adapter import AgenticReasonerAdapter, KGRAGRetrieverTool
from kgrag.core.models import DocumentChunk
from kgrag.core.protocols import Retriever


class TestAgenticReasonerAdapter:

    @pytest.fixture
    def mock_llm(self):
        """Mock language model"""
        llm = Mock(spec=BaseChatModel)
        llm.bind_tools = Mock(return_value=llm)
        return llm

    @pytest.fixture
    def mock_retriever(self):
        """Mock KG-RAG retriever"""
        retriever = Mock(spec=Retriever)
        return retriever

    @pytest.fixture
    def sample_chunks(self):
        """Sample document chunks for testing"""
        return [
            DocumentChunk(
                id="chunk_1",
                doc_id="doc_1",
                content="This is content from document 1 about nuclear power.",
                strategy="vector_search"
            ),
            DocumentChunk(
                id="chunk_2",
                doc_id="doc_2",
                content="This is content from document 2 about safety regulations.",
                strategy="vector_search"
            )
        ]

    @pytest.fixture
    def adapter(self, mock_llm, mock_retriever):
        """Create adapter instance"""
        return AgenticReasonerAdapter(
            llm=mock_llm,
            retriever=mock_retriever,
            max_iterations=2,
            relevance_threshold=0.3
        )

    def test_adapter_initialization(self, adapter, mock_llm, mock_retriever):
        """Test that adapter initializes correctly"""
        assert adapter.llm == mock_llm
        assert adapter.retriever == mock_retriever
        assert adapter.max_iterations == 2
        assert adapter.relevance_threshold == 0.3
        assert hasattr(adapter, 'reasoning_agent')
        assert hasattr(adapter, 'retriever_tool')

    def test_convert_chunks_to_documents(self, adapter, sample_chunks):
        """Test conversion from DocumentChunk to LangChain Document"""
        documents = adapter._convert_chunks_to_documents(sample_chunks)

        assert len(documents) == 2
        assert documents[0].page_content == "This is content from document 1 about nuclear power."
        assert documents[0].metadata['doc_id'] == "doc_1"
        assert documents[0].metadata['chunk_id'] == "chunk_1"
        assert documents[1].page_content == "This is content from document 2 about safety regulations."

    async def test_reason_over_documents_basic(self, adapter, sample_chunks, mock_llm):
        """Test basic reasoning over documents"""
        # Mock the reasoning agent's response
        mock_result = {
            "reasoning_answer": "This is a reasoned answer about nuclear power.",
            "followup_questions": ["What about safety protocols?"],
            "retrieved_docs": [],  # Would contain LangChain documents
            "tool_calls": [],
            "additional_retrieved_context": 0,
            "workflow_metadata": {"total_follow_up_questions": 1}
        }

        adapter.reasoning_agent.reason_over_documents = Mock(return_value=mock_result)

        result = await adapter.reason_over_documents("What is nuclear power?", sample_chunks)

        assert result["reasoning_answer"] == "This is a reasoned answer about nuclear power."
        assert result["followup_questions"] == ["What about safety protocols?"]
        assert "additional_chunks" in result
        assert "tool_calls" in result
        assert "reasoning_metadata" in result


class TestKGRAGRetrieverTool:

    @pytest.fixture
    def mock_retriever(self):
        """Mock KG-RAG retriever that returns DocumentChunks"""
        retriever = Mock(spec=Retriever)
        retriever.retrieve = Mock(return_value=[
            DocumentChunk(
                id="retrieved_1",
                doc_id="doc_3",
                content="Retrieved content about nuclear safety.",
                strategy="reasoning_retrieval"
            )
        ])
        return retriever

    @pytest.fixture
    def retriever_tool(self, mock_retriever):
        """Create KGRAGRetrieverTool instance"""
        return KGRAGRetrieverTool(
            retriever=mock_retriever,
            relevance_threshold=0.3
        )

    def test_retriever_tool_initialization(self, retriever_tool, mock_retriever):
        """Test retriever tool initializes correctly"""
        assert retriever_tool.retriever == mock_retriever
        assert retriever_tool.relevance_threshold == 0.3
        assert retriever_tool.k == 3

    def test_retrieve_documents(self, retriever_tool, mock_retriever):
        """Test document retrieval"""
        documents = retriever_tool.retrieve("nuclear safety query")

        # Verify retriever was called
        mock_retriever.retrieve.assert_called_once_with("nuclear safety query", k=3)

        # Verify conversion to LangChain documents
        assert len(documents) == 1
        assert documents[0].page_content == "Retrieved content about nuclear safety."
        assert documents[0].metadata['doc_id'] == "doc_3"
        assert documents[0].metadata['chunk_id'] == "retrieved_1"

    def test_as_langchain_tool(self, retriever_tool):
        """Test conversion to LangChain tool"""
        tool = retriever_tool.as_langchain_tool()

        assert tool.name == "retrieve_documents"
        assert "Nuklear-Wissensdatenbank" in tool.description
        assert callable(tool.func)