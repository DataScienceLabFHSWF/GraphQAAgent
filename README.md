# KG-RAG QA Agent

**Ontology-informed GraphRAG Question Answering** over Knowledge Graphs built by [KnowledgeGraphBuilder](https://github.com/fneubuerger/KnowledgeGraphBuilder).

## Key Contributions

| Contribution | Module | Description |
|-------------|--------|-------------|
| **HybridGraph Retrieval** | `retrieval/hybrid.py` | FusionRAG — vector + graph + adaptive ontology-informed weighting with RRF |
| **Ontology-Informed Retrieval** | `retrieval/ontology.py` | Query expansion via class hierarchies, synonyms, and expected relations |
| **Agentic GraphRAG** | `retrieval/agentic_rag.py` | LangGraph-style ReAct agent that dynamically combines all retrieval strategies via tool-calling |
| **Chain-of-Thought Reasoning** | `agents/chain_of_thought.py` | Multi-hop KG-grounded CoT with per-step evidence retrieval, inspired by GCR (Luo 2025) |
| **Answer Verification** | `agents/answer_verifier.py` | Post-generation faithfulness checking — extracts claims, verifies against KG, flags unsupported assertions |
| **Transparent Reasoning** | `agents/explainer.py` | Full provenance, subgraph visualisation, step-by-step reasoning chains |
| **Human-in-the-Loop Curation** | `hitl/` | Change proposals, gap detection, KG versioning, and n10s/SHACL integration |
| **Conversational QA** | `chat/` | Multi-turn sessions with streaming SSE, persistent history, and session management |
| **Interactive Frontend** | `frontend/` | Streamlit UI with Chat, KG Explorer, Ontology Browser, and Reasoning Visualisation pages |

## Architecture

```
User Question (Chat UI / API / CLI)
    │
    ▼
ChatSession ──► QuestionParser ──► OntologyRetriever (expand) ──► Strategy Router
    │                                                                   │
    │                 ┌────────────────┬────────────────┬───────────────┤
    │                 │                │                │               │
    │          VectorRetriever   GraphRetriever   CypherRetriever  AgenticRAG
    │          (Qdrant chunks)  (Neo4j subgraphs) (LLM→Cypher)   (ReAct tools)
    │                 │                │                │               │
    │                 └──► RRF + CrossEncoder Rerank ◄─┘               │
    │                              │                                   │
    │                    GraphReasoner (Think-on-Graph + PPR)           │
    │                              │                                   │
    │                     PathRanker (ontology-aware scoring)           │
    │                              │                                   │
    ▼                              ▼                                   │
ContextAssembler ◄─────────────────┴───────────────────────────────────┘
    │
    ▼
ChainOfThoughtReasoner ──► AnswerGenerator ──► AnswerVerifier ──► Explainer
    │                                                                  │
    ▼                                                                  ▼
ChatSession (history + streaming) ◄──────────────────────── QA Response
                                                           (with provenance,
                                                            subgraph JSON,
                                                            faithfulness score)
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

### Demo helper script

A convenience shell script `scripts/start_demo.sh` will kick off the containerised
backends, the API server and the Streamlit frontend in one go.  It is
smart about existing services:

* If Neo4j (`7474`/`7687`) or Fuseki (`3030`) are already bound by other
  projects the script detects the port usage and **skips** launching those
  containers, assuming the existing instances host your data.
* The API and frontend normally run on `8080` and `8501`, but if those ports
  are in use the script will probe upwards for a free port and report the
  actual values chosen.  Local processes listening on the selected ports are
  killed so the new services can bind cleanly.

The Streamlit container is configured with `API_URL=http://qa-agent:8080/api/v1`
by default; override via environment variable or Streamlit secret if needed.

Example:

```bash
chmod +x scripts/start_demo.sh
./scripts/start_demo.sh               # start backend + ui (skips neo4j/fuseki if occupied)
# or
./scripts/start_demo.sh --no-frontend   # just services and API
./scripts/start_demo.sh --no-docker     # API + frontend only
./scripts/start_demo.sh --external-neo4j --external-fuseki   # always assume external DB services
```

Once the demo is running you can expose it publicly with ngrok:

```bash
ngrok http 8501   # tunnel the Streamlit UI
ngrok http 8080   # tunnel the REST API
```

Make sure you stop any other Streamlit/Docker services that use the same ports;
`start_demo.sh` will attempt to kill them automatically.


# Interactive Usage

```bash
# Interactive QA
python scripts/run_qa.py

# Single question
python scripts/run_qa.py -q "Welche Rückbauverfahren werden bei Reaktor A eingesetzt?"

# Compare retrieval strategies (including new: cypher, agentic, hybrid_sota)
python scripts/compare_strategies.py -q "What methods are used for Reaktor A?"

# Run evaluation benchmark
python scripts/run_evaluation.py

# Start REST API (if not using Docker)
uvicorn kgrag.api.server:app --host 0.0.0.0 --port 8080

# Start Streamlit frontend (if not using Docker)
streamlit run src/kgrag/frontend/app.py
```

## Project Structure

```
src/kgrag/
├── core/           # Config, models, protocols, exceptions, domain config (YAML-driven)
├── connectors/     # Neo4j, Qdrant, Fuseki, Ollama, LangChain Ollama provider
├── retrieval/      # Vector, Graph, Hybrid, Ontology, Cypher, AgenticRAG, GraphReasoner,
│                   #   PathRanker, OntologyContext, Reranker, EntityLinker
├── agents/         # QuestionParser, ContextAssembler, AnswerGenerator, Explainer,
│                   #   Orchestrator, ChainOfThoughtReasoner, AnswerVerifier
├── chat/           # Multi-turn sessions, history store, SSE streaming
├── frontend/       # Streamlit UI (Chat, KG Explorer, Ontology Browser, Reasoning Viz)
├── hitl/           # Human-in-the-Loop: change proposals, gap detection, KG versioning, n10s
├── demo/           # Guided demo runner, scenarios, HTML/Markdown export
├── adapters/       # Third-party integration (agentic reasoner adapter)
├── validation/     # SHACL, CQ validator, consistency checks
├── evaluation/     # QA dataset, Metrics, Comparator, Reporter
├── api/            # FastAPI server, QA routes, chat routes, KG explorer routes
└── cli.py          # CLI entry point

src/third_party/
└── agentic_reasoning/  # Vendored ReAct reasoning framework (LangChain tool-calling agent)
```

## Retrieval Strategies

| Strategy | Source | Key Idea |
|----------|--------|----------|
| **VectorOnly** | Qdrant | Classic RAG baseline — embed & search |
| **GraphOnly** | Neo4j | Entity-centric, subgraph, or path retrieval |
| **Hybrid (FusionRAG)** | Both | Adaptive RRF + cross-encoder reranking |
| **Ontology-Expanded** | Fuseki + Both | Class hierarchy & synonym expansion |
| **Cypher** | Neo4j + LLM | LLM→Cypher→Neo4j with ontology-enhanced prompts and domain-specific templates |
| **Agentic** | All | ReAct tool-calling agent dynamically combining vector, graph, entity, path, and ontology tools |
| **Hybrid SOTA** | All | Full pipeline: parse → expand → 3-way retrieve (vector + graph + Think-on-Graph) → CoT reasoning → generate → verify → explain |

### Advanced Retrieval Components

| Component | Module | Description |
|-----------|--------|-------------|
| **GraphReasoner** | `retrieval/graph_reasoning.py` | Think-on-Graph iterative exploration with Personalized PageRank and LLM-guided edge selection |
| **PathRanker** | `retrieval/path_ranker.py` | Relation-aware path scoring using ontology expected relations (`α·confidence + β·relation_relevance + γ·length_penalty`) |
| **OntologyContext** | `retrieval/ontology_context.py` | Shared TBox knowledge loaded once from Fuseki — class hierarchy, property map, Neo4j↔ontology label mapping |

## Conversational QA & Chat

Multi-turn conversational question answering with session management and real-time streaming:

- **Session management** — Create, list, and delete chat sessions with persistent history
- **SSE streaming** — Server-Sent Events for real-time reasoning steps, token-by-token generation, provenance, and subgraph events
- **History context** — Previous turns are injected into the QA pipeline for follow-up questions and coreference resolution

**API endpoints:**
- `POST /api/v1/chat/send` — Send a message (SSE stream or JSON response)
- `GET /api/v1/chat/sessions` — List sessions
- `GET /api/v1/chat/sessions/{id}/history` — Retrieve session history
- `DELETE /api/v1/chat/sessions/{id}` — Delete a session
- `POST /api/v1/chat/feedback` — Submit HITL feedback

## Streamlit Frontend

Interactive web UI with four pages:

| Page | Description |
|------|-------------|
| **Chat** | Conversational QA with strategy/language selection, demo questions, and session export |
| **KG Explorer** | Entity search with type filtering, subgraph rendering around selected entities |
| **Ontology Browser** | TBox class hierarchy tree, class property viewer, raw TTL display |
| **Reasoning Visualisation** | Inspect past sessions — CoT step timeline, verification results, subgraph viewer |

Components include rich chat bubbles (collapsible confidence/provenance), a Chain-of-Thought DAG renderer, and an interactive pyvis subgraph viewer.

## KG Explorer API

Browse and search the knowledge graph directly:

- `GET /api/v1/explore/entities` — List entities with type filtering and pagination
- `GET /api/v1/explore/entities/{id}` — Entity details with properties and relations
- `GET /api/v1/explore/entities/{id}/subgraph` — Subgraph around an entity (configurable depth)
- `GET /api/v1/explore/relations` — List relation types
- `GET /api/v1/explore/stats` — KG statistics (node/edge counts, label distribution)
- `GET /api/v1/explore/ontology/tree` — TBox class hierarchy
- `GET /api/v1/explore/ontology/classes/{uri}/properties` — Properties for a class
- `GET /api/v1/explore/search` — Full-text entity search

## Human-in-the-Loop (HITL) KG Curation

Feedback-driven knowledge graph maintenance:

| Module | Description |
|--------|-------------|
| **Change Proposals** | Full lifecycle workflow (`proposed → validated → accepted → applied`) for entity/relation/property additions, updates, and deletions |
| **Gap Detection** | Analyses QA answers to detect ABox gaps (low confidence) and TBox gaps (unanswerable questions), prioritises research items from repeated failures |
| **KG Versioning** | Temporal versioning via `ChangeEvent` nodes in Neo4j — supports query-as-of, rollback, and entity history |
| **n10s Integration** | Neosemantics helpers for RDF snapshot export, SHACL validation of proposed changes, and ontology sync from Fuseki to Neo4j |

## Chain-of-Thought & Verification

Two new agent components enable grounded, verifiable reasoning:

- **ChainOfThoughtReasoner** (`agents/chain_of_thought.py`) — Decomposes complex questions into atomic reasoning steps, retrieves targeted evidence per step, and builds a fully KG-grounded reasoning chain. Inspired by GCR (Luo 2025, ICML) and KG-GPT (Kim 2023).
- **AnswerVerifier** (`agents/answer_verifier.py`) — Post-generation faithfulness checking. Extracts factual claims, verifies each against the KG subgraph, flags unsupported/contradicted claims, and computes a grounded faithfulness score. Inspired by GCR (Luo 2025, ICML).

## Domain Configuration

All domain-specific text is externalised into `config/domain.yaml`:
- System prompts and vocabulary hints
- Neo4j label mappings and Cypher query templates
- Example entity types and demo questions
- Enables domain-neutral Python code — swap domains by editing one YAML file

## Guided Demo System

Pre-packaged demo scenarios (`demo/`) for showcasing the system:

- **Scenarios**: Nuclear Decommissioning Legal Framework, Waste Management Chain, Facility-Specific Queries, Ontology Exploration
- **Demo runner**: Rich terminal output with panels and tables via the `rich` library
- **Export**: Sessions exportable as HTML (Jinja2) or Markdown

```bash
# Run guided demo
python -m kgrag.demo.demo_runner
```

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
| **Faithful KG-grounded reasoning** | Luo et al. 2025 (GCR, ICML) | ChainOfThoughtReasoner for multi-hop grounding; AnswerVerifier for post-hoc claim verification |
| **Adaptive strategy selection** | Yu 2025 | Specific QA → entity-centric; abstract QA → subgraph/path; fusion weights adjusted per question type |
| **Subgraph reasoning** | Teru et al. 2020 (GraIL) | Subgraph extraction + scoring for inductive reasoning over KG neighbourhoods |
| **Query expansion via ontology** | OWL/SPARQL standards | Class hierarchies, synonyms, and expected relations from Fuseki enrich queries before retrieval |
| **Think-on-Graph** | Sun et al. 2023 | GraphReasoner performs iterative LLM-guided graph exploration with PPR beam search |
| **Relation-aware path scoring** | Wang et al. 2021 (PathCon) | PathRanker scores multi-hop paths using ontology-expected relation relevance |
| **Agentic RAG** | LangGraph / ReAct | AgenticGraphRAG uses tool-calling to dynamically compose retrieval strategies per query |

See [planning/C3_LITERATURE.md](planning/C3_LITERATURE.md) for the full literature synthesis.

### Key References

1. Liu, L., Wang, Z., & Tong, H. (2024). *Neural-Symbolic Reasoning over Knowledge Graphs: A Survey from a Query Perspective*. [arXiv:2412.10390](https://arxiv.org/abs/2412.10390)
2. Zhang, J., Chen, B., et al. (2021). *Neural, Symbolic and Neural-Symbolic Reasoning on Knowledge Graphs*. AI Open. [arXiv:2010.05446](https://arxiv.org/abs/2010.05446)
3. Luo, L., Zhao, Z., et al. (2025). *Graph-constrained Reasoning: Faithful Reasoning on KGs with LLMs*. ICML 2025. [arXiv:2410.13080](https://arxiv.org/abs/2410.13080)
4. Yu, F. (2025). *What Really Matters to Better GraphRAG Implementation?* [Medium](https://medium.com/@yu-joshua/what-really-matters-to-better-graphrag-implementation-part-1-e02fff773c48)
5. Edge, D. et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*. [arXiv:2404.16130](https://arxiv.org/abs/2404.16130)
6. Sun, J. et al. (2023). *Think-on-Graph: Deep and Responsible Reasoning of LLMs on Knowledge Graphs*. [arXiv:2307.07697](https://arxiv.org/abs/2307.07697)
7. Kim, J. et al. (2023). *KG-GPT: Large Language Models with Knowledge Graphs for Reasoning*. [arXiv:2310.11220](https://arxiv.org/abs/2310.11220)
8. Wang, H. et al. (2021). *Relational Message Passing for Knowledge Graph Completion* (PathCon). KDD 2021.

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
