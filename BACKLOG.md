# KG-RAG Development Backlog

> Last updated after the Phase 1-8 improvement sprint.

## ✅ Completed (current state)

### Infrastructure
- Docker stack: Neo4j 5.26.0 + APOC (bolt://localhost:7687), Qdrant (6333),
  Fuseki (3030), Ollama (18136), API (8080), Streamlit frontend (8504)
- Neo4j: ~1 400 nodes, ~5 100 edges, fully indexed
- Qdrant: collection `kgbuilder`, 3 004 vectors
- Ollama: qwen3:8b (generation), qwen3-embedding:latest (embeddings)
- DomainConfig YAML system (`config/domain.yaml`) — all domain prompts
  externalised

### Retrieval & Agent
- **10-tool AgenticGraphRAG** (`retrieval/agentic_rag.py`):
  `search_vectors`, `query_graph`, `explore_entity`, `find_connections`,
  `lookup_ontology`, `aggregate_subgraph`, `semantic_search_entities`,
  `compare_entities`, `count_and_aggregate`, `collect_evidence`
- **Tool result caching** — identical calls memo-ised within the agent loop
- **Adaptive iteration control** — `_ITERATIONS_BY_TYPE` adjusts limits per
  question type (factoid:4, boolean:3, list:5, comparative:7, causal:7,
  aggregation:5)
- **Self-reflection** — after 3 iterations the agent evaluates sufficiency
- **ReWOO-style retrieval planning** — generates a numbered plan before
  tool execution
- **Evidence quality scoring** — embedding cosine similarity filters
  low-relevance contexts (`_EVIDENCE_QUALITY_THRESHOLD = 0.15`)
- **Full tool trace** — every tool call recorded with args, result summary,
  iteration number
- **Fact chains** — shortest-path fact chains grounded in KG entity IDs
- **Entity linker** — fuzzy + embedding entity linking to Neo4j
- **CypherRetriever** — LLM-generated Cypher with mutation guards
- **HybridGraph / FusionRAG** — adaptive ontology-informed weighting + RRF
- **Ontology-informed retrieval** — class hierarchies, synonyms, expected
  relations for query expansion
- **Path ranker** — multi-hypothesis path ranking for graph reasoning
- **Reranker** — cross-encoder reranking of retrieved contexts
- **Graph reasoning** — multi-hop ThinkOnGraph with exploration state

### Agents
- **Orchestrator** — full QA pipeline: parse → retrieve → CoT → generate →
  verify → explain, forwards fact_chains + tool_trace
- **AnswerGenerator** — `preamble` parameter (priority over `cot_summary`)
- **AnswerVerifier** — faithfulness checking with supported/unsupported claims
- **ChainOfThought** — structured reasoning with sub-questions and grounding
- **QuestionParser** — question type classification and entity extraction
- **Explainer** — provenance chains, subgraph visualisation, step-by-step
  reasoning traces

### API
- `/ask` endpoint — full QA with fact_chains + tool_trace in response
- `/chat/send` — SSE streaming with 13 event types (session, reasoning_step,
  token, evidence, entities, relations, provenance, subgraph, verification,
  gap_alert, fact_chains, tool_trace, done)
- Chat session management with conversation history
- HITL feedback / correction pipeline with change proposals
- Rate limiting (30 req/60s per session and IP)

### Advanced Modules
- **Graph-of-Thought** (`retrieval/graph_of_thought.py`): DAG-based reasoning
  with `ReasoningNode`, `ReasoningDAG`, `GraphOfThoughtReasoner`
- **KG-constrained validation** (`retrieval/kg_constrained.py`): entity
  verification, fuzzy matching, evidence consistency checking
- **Active ontology learning** (`retrieval/active_ontology.py`): gap
  detection on failed lookups, Turtle proposal generation, integrated into
  agentic RAG `lookup_ontology` parser

### Models & Schemas
- `QAAnswer`: includes `fact_chains`, `tool_trace`, `verification`,
  `exploration_trace`, `reasoning_steps`
- `ChatResponse`: includes `fact_chains`, `tool_trace`, gap_detection,
  verification, subgraph
- `FactChainResponse`, `ToolTraceResponse` Pydantic models

### Tests
- 185+ tests passing (`pytest tests/ -q`)
- Coverage: agents, connectors, core, evaluation, retrieval

---

## 🔮 Future Work

### High-Priority (next sprint)
- **Real token streaming** — replace word-level simulation with LLM callback
  streaming (LangChain `BaseCallbackHandler.on_llm_new_token`)
- **LLM-driven GoT expansion** — replace heuristic sub-question generation
  with a fine-tuned policy prompt
- **Relation validation** — upgrade KG-constrained validation from substring
  matching to LLM-based NLI
- **Ontology proposal approval** — wire Fuseki writes behind human approval
  in the HITL pipeline

### Medium-Priority
- **Learned retrieval policy** — train a lightweight model to predict tool
  selection order from question features
- **Multi-agent collaboration** — specialist sub-agents (entity expert,
  relation expert, temporal reasoner) coordinated by the orchestrator
- **Evaluation harness expansion** — DeepEval integration for LLM-as-judge
  metrics, benchmark v2 with multi-hop and temporal questions
- **WebSocket support** — alternative to SSE for bidirectional communication
- **Frontend improvements** — GoT DAG visualisation, tool trace timeline,
  fact chain cards

### Low-Priority / Research
- **KG-constrained decoding** — logit biasing during LLM generation to
  prefer entity names that exist in the graph
- **Cross-lingual QA** — extend question parser and answer generator to
  handle EN/DE seamlessly
- **Active learning loop** — use ontology proposals to retrain embeddings
  and improve retrieval over time
- Prompt-based tool calling (alternative to native tool binding)

**Architecture Decisions:**
- Maintain custom orchestration layer for now (familiarity with KnowledgeGraphBuilder patterns)
- Use LangChain components internally for LLM and embedding operations
- Keep existing agent protocols and interfaces for backward compatibility

**Testing Strategy:**
- Manual testing of API endpoints post-refactoring
- Validation of Qdrant search functionality
- End-to-end QA pipeline verification
- Performance comparison with previous custom connector approach</content>
<parameter name="filePath">/home/fneubuerger/GraphQAAgent/BACKLOG.md