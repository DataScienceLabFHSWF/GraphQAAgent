# Third-Party Components

This document describes third-party components that have been vendored into the KG-RAG codebase.

## Vendored Components

### Agentic Reasoning Framework

**Source Repository:** [DataScienceLabFHSWF/agentic-reasoning-framework](https://github.com/DataScienceLabFHSWF/agentic-reasoning-framework)

**License:** MIT License

**Components Vendored:**
- `src/third_party/agentic_reasoning/prompts.py` - Prompt templates for ReAct reasoning
- `src/third_party/agentic_reasoning/reasoning_agent.py` - ReAct reasoning agent implementation
- `src/third_party/agentic_reasoning/retriever_tool.py` - Document retrieval tool wrapper
- `src/third_party/agentic_reasoning/workflow.py` - Simplified workflow orchestration

**Integration:**
- Adapted for KG-RAG in `src/kgrag/adapters/agentic_reasoner_adapter.py`
- Integrated with KG-RAG's document retrieval system
- Maintains ReAct loop capabilities for iterative reasoning with document retrieval

**Modifications Made:**
- Updated imports to work with KG-RAG structure
- Replaced ChromaDB-specific retrieval with KG-RAG retrieval interface
- Added conversion between KG-RAG `DocumentChunk` and LangChain `Document` formats
- Simplified workflow to focus on reasoning agent functionality

**Original Functionality Preserved:**
- ReAct reasoning loop with tool-calling
- Iterative document retrieval based on reasoning needs
- German nuclear domain expertise prompts
- Tool call tracking and metadata collection

## License Compliance

All vendored components maintain their original licenses. The MIT License allows for reuse, modification, and distribution with attribution.

## Attribution

We acknowledge and thank the Data Science Lab at FHSWF for their work on the agentic-reasoning-framework, which provided valuable ReAct reasoning capabilities that enhance our KG-RAG system's ability to perform deep, iterative analysis of nuclear domain questions.