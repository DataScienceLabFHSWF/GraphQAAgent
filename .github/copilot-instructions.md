# Copilot Instructions for KG-RAG Agent

## Project Overview
This is the **KG-RAG QA Agent** — an ontology-informed GraphRAG system that
answers questions by combining vector retrieval (Qdrant), knowledge graph
traversal (Neo4j), and ontology-guided expansion (Fuseki/SPARQL).

## Key Contributions
1. **HybridGraph Retrieval** — `src/kgrag/retrieval/hybrid.py`: FusionRAG with
   adaptive ontology-informed weighting and Reciprocal Rank Fusion.
2. **Ontology-Informed Retrieval** — `src/kgrag/retrieval/ontology.py`: query
   expansion via class hierarchies, synonyms, and expected relations.
3. **Transparent Reasoning** — `src/kgrag/agents/explainer.py`: full provenance
   chains, subgraph visualisation, and step-by-step reasoning traces.

## Architecture
- **Connectors** (`connectors/`): Read-only access to Neo4j, Qdrant, Fuseki, Ollama
- **Retrieval** (`retrieval/`): Vector, Graph, Hybrid, Ontology strategies
- **Agents** (`agents/`): Question parsing, answer generation, explanation
- **Evaluation** (`evaluation/`): Metrics, strategy comparison, reporting

## Conventions
- Python 3.11+, async throughout, Pydantic v2 for config
- All retrieval strategies implement the `Retriever` protocol
- Structured logging via `structlog`
- Tests use `pytest-asyncio` with mock connectors
