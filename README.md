# KG-RAG QA Agent

**Ontology-informed GraphRAG Question Answering** over Knowledge Graphs built by [KnowledgeGraphBuilder](https://github.com/fneubuerger/KnowledgeGraphBuilder).

## Key Contributions

| Contribution | Module | Description |
|-------------|--------|-------------|
| **HybridGraph Retrieval** | `retrieval/hybrid.py` | FusionRAG — vector + graph + adaptive ontology-informed weighting with RRF |
| **Ontology-Informed Retrieval** | `retrieval/ontology.py` | Query expansion via class hierarchies, synonyms, and expected relations |
| **Transparent Reasoning** | `agents/explainer.py` | Full provenance, subgraph visualisation, step-by-step reasoning chains |

## Architecture

```
User Question
    │
    ▼
QuestionParser ──► OntologyRetriever (expand) ──► HybridRetriever
    │                                                    │
    │                              ┌─────────────────────┤
    │                              │                     │
    │                     VectorRetriever          GraphRetriever
    │                     (Qdrant chunks)        (Neo4j subgraphs)
    │                              │                     │
    │                              └──► RRF ◄────────────┘
    │                                    │
    │                          CrossEncoder Rerank
    │                                    │
    ▼                                    ▼
ContextAssembler ──► AnswerGenerator ──► Explainer ──► QA Response
                                                       (with provenance
                                                        + subgraph JSON)
```

## Quick Start

### Option 1: KG-RAG Dedicated Stack (Recommended)

```bash
# Clone and install
git clone https://github.com/fneubuerger/GraphQAAgent.git
cd GraphQAAgent
pip install -e ".[dev]"

# Copy and edit environment config
cp .env.example .env
# Edit .env with your settings (Neo4j, Qdrant, Fuseki credentials)

# Setup KG-RAG's dedicated Ollama instance with required models
./setup-ollama.sh

# Start the complete KG-RAG stack (Ollama + QA Agent)
docker-compose -f docker-compose.kgrag.yml up -d

# The API will be available at http://localhost:8080
```

### Option 2: Use Existing Infrastructure

If you already have Neo4j, Qdrant, Fuseki, and Ollama running:

```bash
# Install dependencies
pip install -e ".[dev]"

# Edit .env to point to your existing services
# KGRAG_NEO4J__URI=bolt://your-neo4j:7687
# KGRAG_QDRANT__URL=http://your-qdrant:6333
# etc.

# Start only the QA agent
docker-compose -f docker-compose.yml up -d qa-agent
```

### Interactive Usage

```bash
# Interactive QA
python scripts/run_qa.py

# Single question
python scripts/run_qa.py -q "Welche Rückbauverfahren werden bei Reaktor A eingesetzt?"

# Compare retrieval strategies
python scripts/compare_strategies.py -q "What methods are used for Reaktor A?"

# Run evaluation benchmark
python scripts/run_evaluation.py

# Start REST API (if not using Docker)
uvicorn kgrag.api.server:app --host 0.0.0.0 --port 8080
```

## Current Development Status

**🚨 Active Refactoring:** Comprehensive LangChain integration in progress

### Recent Changes
- ✅ **LangChain Migration:** Replaced custom OllamaConnector with LangChainOllamaProvider (ChatOllama + OllamaEmbeddings)
- ✅ **Qdrant Compatibility:** Fixed AsyncQdrantClient API issues (updated to use `search()` method)
- ✅ **Infrastructure:** Docker KG-RAG stack operational (Neo4j confirmed running)
- ✅ **Tool Binding:** Implemented prompt-based ReAct reasoning with LangChain components

### Known Issues
- **API Functionality:** Qdrant search methods updated but untested
- **Orchestration:** Still using custom patterns, not full LangChain agents
- **Testing:** Needs validation of end-to-end QA pipeline

See [BACKLOG.md](BACKLOG.md) for detailed current issues and development priorities.

## Project Structure

```
src/kgrag/
├── core/           # Config, models, protocols, exceptions
├── connectors/     # Neo4j, Qdrant, Fuseki, Ollama (read-only)
├── retrieval/      # Vector, Graph, Hybrid, Ontology, Reranker, EntityLinker
├── agents/         # QuestionParser, ContextAssembler, AnswerGenerator, Explainer, Orchestrator
├── validation/     # SHACL, CQ validator, Consistency checks
├── evaluation/     # QA dataset, Metrics, Comparator, Reporter
└── api/            # FastAPI server, routes, schemas
```

## Retrieval Strategies

| Strategy | Source | Key Idea |
|----------|--------|----------|
| **VectorOnly** | Qdrant | Classic RAG baseline — embed & search |
| **GraphOnly** | Neo4j | Entity-centric, subgraph, or path retrieval |
| **Hybrid (FusionRAG)** | Both | Adaptive RRF + cross-encoder reranking |
| **Ontology-Expanded** | Fuseki + Both | Class hierarchy & synonym expansion |

## Evaluation

```bash
# Full benchmark across strategies
python scripts/run_evaluation.py --strategies vector_only graph_only hybrid

# Export results for thesis
python scripts/export_results.py --input reports/evaluation/evaluation_report.json
```

Metrics: Token F1, Exact Match, Faithfulness, Context Relevance, Latency.

## Docker Setup

The project provides two Docker configurations:

### KG-RAG Dedicated Stack (`docker-compose.kgrag.yml`)
- **Dedicated Ollama** on port 18136 with GPU support
- **QA Agent API** on port 8080
- **Isolated environment** for KG-RAG development
- **Automatic model setup** via `setup-ollama.sh`

```bash
# Setup and start everything
./setup-ollama.sh
docker-compose -f docker-compose.kgrag.yml up -d
```

### Shared Infrastructure Stack (`docker-compose.yml`)
- **Shared Ollama** on port 11434 (with KGB/other projects)
- **Full infrastructure**: Neo4j, Qdrant, Fuseki, Ollama
- **Multi-project development** environment

```bash
# Start all services
docker-compose up -d
```

## Theoretical Foundations

The architecture is grounded in recent knowledge graph reasoning and GraphRAG research:

| Concept | Source | How We Apply It |
|---------|--------|-----------------|
| **Neural-symbolic fusion** | Liu et al. 2024; Zhang et al. 2021 | Symbolic graph traversal (Neo4j, Fuseki) + neural embeddings (Qdrant, Ollama) fused via adaptive RRF |
| **Operator-centric GraphRAG** | Yu 2025 | Retrieval operators (entity linking, RRF, cross-encoder reranking) matter more than graph schema richness |
| **Faithful KG-grounded reasoning** | Luo et al. 2025 (GCR, ICML) | Post-hoc citation verification in Explainer; future: constrained decoding via KG-Trie |
| **Adaptive strategy selection** | Yu 2025 | Specific QA → entity-centric; abstract QA → subgraph/path; fusion weights adjusted per question type |
| **Subgraph reasoning** | Teru et al. 2020 (GraIL) | Subgraph extraction + scoring for inductive reasoning over KG neighbourhoods |
| **Query expansion via ontology** | OWL/SPARQL standards | Class hierarchies, synonyms, and expected relations from Fuseki enrich queries before retrieval |

See [planning/C3_LITERATURE.md](planning/C3_LITERATURE.md) for the full literature synthesis.

### Key References

1. Liu, L., Wang, Z., & Tong, H. (2024). *Neural-Symbolic Reasoning over Knowledge Graphs: A Survey from a Query Perspective*. [arXiv:2412.10390](https://arxiv.org/abs/2412.10390)
2. Zhang, J., Chen, B., et al. (2021). *Neural, Symbolic and Neural-Symbolic Reasoning on Knowledge Graphs*. AI Open. [arXiv:2010.05446](https://arxiv.org/abs/2010.05446)
3. Luo, L., Zhao, Z., et al. (2025). *Graph-constrained Reasoning: Faithful Reasoning on KGs with LLMs*. ICML 2025. [arXiv:2410.13080](https://arxiv.org/abs/2410.13080)
4. Yu, F. (2025). *What Really Matters to Better GraphRAG Implementation?* [Medium](https://medium.com/@yu-joshua/what-really-matters-to-better-graphrag-implementation-part-1-e02fff773c48)
5. Edge, D. et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*. [arXiv:2404.16130](https://arxiv.org/abs/2404.16130)

## Development

```bash
# Lint
ruff check src/ tests/

# Type check
mypy src/

# Test
pytest

# Test with coverage
pytest --cov=kgrag --cov-report=html
```

## License

MIT
