# C3. KG-RAG QA Agent — Literature & Theoretical Foundations

Research synthesis from key papers and analyses that inform the design of the
KG-RAG QA Agent. Each source maps to specific architectural decisions.

---

## Source 1: Neural-Symbolic Reasoning over KGs — A Survey (Liu et al., 2024)

> *arXiv 2412.10390* — Comprehensive survey of KG reasoning from a query perspective.

### Key Takeaways for Our Architecture

| Survey Insight | How We Apply It |
|---------------|-----------------|
| **Neural-symbolic integration** is the state of the art — combining interpretable symbolic methods with robust neural methods | Our hybrid pipeline: symbolic (Neo4j graph traversal + Fuseki SPARQL) fused with neural (embeddings, LLM generation) |
| **Single-hop queries** benefit from entity embedding + neighbourhood lookup | `GraphRetriever.ENTITY_CENTRIC` mode — entity link → 1-hop neighbourhood |
| **Complex logical queries** require multi-step traversal with constraint satisfaction | `GraphRetriever.SUBGRAPH` + `PATH` modes for multi-hop questions |
| **GraIL** — subgraph extraction + GNN scoring for inductive reasoning | Inspires our subgraph extraction and scoring in `_subgraph_score()` |
| **Path Ranking Algorithm (PRA)** — random walks as relational features | Influences our path-based retrieval mode |
| **QA-GNN** — joint LM + KG reasoning via working graph | Parallels our Context Assembler merging vector + graph evidence for LLM |
| **GraphRAG** (Edge et al.) — entity graph → community summaries → partial answers → final synthesis | Validates our two-stage approach (retrieve subgraphs → synthesise answer) |
| **Think-on-Graph** — LLM agent traversing KG neighbourhoods iteratively | Future direction: could extend Orchestrator with iterative graph exploration |
| **Deductive + inductive + abductive reasoning** all have roles | We use deductive (ontology constraints), inductive (embedding similarity), and abductive (LLM generation) |

### Implications for Transparency (Our Contribution #3)

The survey highlights a persistent gap: neural methods are robust but lack
interpretability; symbolic methods are interpretable but brittle. Our
**Explainer** module addresses this by:

- Providing **symbolic provenance** (which KG entities/relations were used)
- Showing the **subgraph structure** (vis.js/D3 compatible JSON)
- Building **step-by-step reasoning chains** that trace the retrieval path
- Estimating **confidence** from both graph structure and text similarity

This positions our work in the neural-symbolic integration paradigm, using
symbolic graph structure for explainability while relying on neural methods for
robustness.

---

## Source 2: Neural, Symbolic and Neural-Symbolic Reasoning on KGs (Zhang et al., 2021)

> *arXiv 2010.05446* — Earlier survey (AI Open 2021) covering KG completion + KGQA.

### Key Takeaways

| Insight | Application |
|---------|-------------|
| **KG completion and QA are unified** under the same reasoning framework | Our system can leverage both: incomplete KG edges supplemented by vector retrieval |
| **Symbolic reasoning** = intolerant of noise but highly interpretable | Validates having `GraphRetriever` as one arm rather than the sole strategy |
| **Neural reasoning** = robust to noise but lacks interpretability | Validates having `VectorRetriever` complement graph retrieval |
| **Hybrid reasoning** = the principled approach for real-world KGs | Core thesis behind `HybridRetriever / FusionRAG` |
| **Entity linking** is the critical bridge between NL and KG | Justifies our `EntityLinker` with 3 fallback strategies (exact → fuzzy → embedding) |

### Design Validation

This survey confirms our architectural split:
1. **Symbolic arm**: Neo4j traversal + Fuseki ontology constraints
2. **Neural arm**: Qdrant embeddings + LLM generation
3. **Fusion**: RRF with adaptive weighting

The survey's finding that "knowledge graphs are often incomplete" directly
motivates our hybrid approach — graph retrieval handles structured knowledge,
vector retrieval fills in the gaps with text-based evidence.

---

## Source 3: Graph-Constrained Reasoning (GCR) (Luo et al., 2025)

> *arXiv 2410.13080* — ICML 2025. Faithful KG-grounded reasoning via KG-Trie decoding.

### Key Innovation: KG-Trie Constrained Decoding

GCR introduces a **trie-based index (KG-Trie)** that encodes valid KG reasoning
paths and constrains the LLM decoding process. This eliminates reasoning
hallucinations by ensuring the LLM can only generate paths that exist in the KG.

### Takeaways for Our System

| GCR Technique | Our Adaptation |
|--------------|----------------|
| **KG-Trie** constrains LLM to valid graph paths | Our `Explainer` validates cited entities/relations exist in retrieved subgraph (post-hoc verification) |
| **Specialized KG-LLM + General LLM** dual-model architecture | Our architecture: Ollama (domain LLM) for generation + ontology constraints for grounding |
| **Zero-shot generalisation** to unseen KGs | Our ontology-driven expansion generalises — new classes/properties auto-discovered via SPARQL |
| **Faithful reasoning** = no hallucinated entities | Our `faithfulness` metric measures this; `Explainer` checks citation grounding |
| **Graph structure integrated into decoding** | Future direction: could integrate KG-Trie-style constraints into our Ollama generation |

### Implications for Faithful Reasoning

GCR's core insight — **integrating graph structure into generation, not just
retrieval** — is a natural evolution of our architecture. Currently we:

1. Retrieve subgraphs (graph structure in retrieval)
2. Serialize subgraphs to NL context (graph → text bridge)
3. Generate answer with citations (soft grounding)
4. Post-hoc verify citations (explainer validation)

A future enhancement inspired by GCR would add **constrained decoding** where
the LLM can only reference entities/relations present in the retrieved subgraph.

---

## Source 4: What Really Matters for GraphRAG (Yu, 2025)

> *Medium analysis* — Data-driven comparison of 12 GraphRAG methods across 6 datasets.

### Five Key Findings and Our Response

#### Finding 1: Richer Graphs ≠ Better Performance

> *"Relevant and discriminative structure is what matters."*

Rich Knowledge Graphs (RKG) with entity descriptions, relationship keywords,
and edge weights do not always outperform simpler KG or Passage Graph structures.
RAPTOR (tree) and HippoRAG (KG + PPR) often match or beat complex RKG systems.

**Our response**: Our ontology-informed approach focuses on **relevance** not
richness — the `OntologyRetriever` expands queries with class hierarchies and
expected relations to retrieve *discriminative* subgraphs, not exhaustive ones.

#### Finding 2: Operators Matter More Than Graph Type

> *"GraphRAG is not just about building a graph — it's about operationalizing
> the graph with intelligent retrieval operators."*

Methods combining graph topology + statistical ranking (e.g., PPR) consistently
outperform. Operator design is a stronger predictor of quality than graph schema.

**Our response**: This validates our operator-centric design:
- **Entity linking** (multi-strategy) → node selection
- **k-hop expansion** → subgraph extraction
- **RRF fusion** → statistical ranking across sources
- **Cross-encoder reranking** → final precision filtering
- **Adaptive weighting** → query-aware operator tuning

#### Finding 3: Cost vs. Accuracy Has a Clear Pareto Frontier

Systems like HippoRAG and RAPTOR sit on the Pareto frontier (high accuracy,
reasonable cost), while ToG and KGP are expensive and underperforming.

**Our response**: Our architecture targets the Pareto frontier:
- Local Ollama models (no API costs)
- Read-only KG access (no graph construction overhead)
- Qdrant for fast vector search
- Cross-encoder reranking only on top-N candidates (not exhaustive)

#### Finding 4: Dataset Size and Structure Shape Performance

Mid-sized datasets (1–3M tokens) are the sweet spot for GraphRAG. At scale,
graph density increases and multi-hop chains are harder to locate.

**Our response**: Nuclear decommissioning KGs are typically mid-sized and
domain-specific — an ideal regime for GraphRAG. Our ontology constraints help
**isolate relevant subgraphs** even as the KG grows, addressing the scalability
concern.

#### Finding 5: Abstract vs. Specific QA Require Different Architectures

- **Specific QA** (fact retrieval) → strong node indexing, edge weighting, 1-hop
- **Abstract QA** (cross-page reasoning) → communities, relationship-rich graphs

**Our response**: This directly validates our **adaptive strategy selection**:
- `FACTOID` / `BOOLEAN` → entity-centric retrieval (specific QA)
- `CAUSAL` / `COMPARATIVE` → subgraph + path retrieval (abstract QA)
- `LIST` → hybrid with ontology class expansion
- Adaptive fusion weights shift between vector and graph based on question type

---

## Synthesis: How Literature Shapes Our Contributions

### Contribution 1: HybridGraph Retrieval (`hybrid.py`)

Grounded in:
- **Neural-symbolic fusion** (Liu 2024, Zhang 2021): combining symbolic graph
  traversal with neural embeddings and generation
- **Operator-centric design** (Yu 2025): RRF, adaptive weighting, and
  cross-encoder reranking are more predictive of quality than graph schema
- **Adaptive architecture** (Yu 2025, Finding 5): question-type-aware weight
  adjustment addresses the specific-vs-abstract QA gap

### Contribution 2: Ontology-Informed Retrieval (`ontology.py`)

Grounded in:
- **Query expansion via class hierarchies** (Liu 2024, Section 5): semantic
  parsing + structured knowledge for NL query understanding
- **OWL/SPARQL for structured constraints** (Liu 2024, Section 3.1.1): using
  ontology as formal rule-set for guiding retrieval
- **Discriminative structure** (Yu 2025, Finding 1): ontology focuses retrieval
  on *relevant* relations rather than exhaustive graph traversal
- **Entity linking as critical bridge** (Zhang 2021): our multi-strategy linker
  with ontology synonym fallback

### Contribution 3: Transparent Reasoning (`explainer.py`)

Grounded in:
- **Interpretability gap** (Liu 2024, Zhang 2021): neural-symbolic methods
  should combine neural robustness with symbolic explainability
- **Faithful reasoning** (Luo 2025, GCR): answer grounding via KG structure,
  citation verification, provenance chains
- **Subgraph visualisation** (Liu 2024, GraIL): extracted subgraph as evidence
  for the reasoning process
- **Cost of transparency** (Yu 2025): our approach adds minimal overhead since
  provenance is a byproduct of the retrieval pipeline

---

## References

1. Liu, L., Wang, Z., & Tong, H. (2024). *Neural-Symbolic Reasoning over
   Knowledge Graphs: A Survey from a Query Perspective*. arXiv:2412.10390.

2. Zhang, J., Chen, B., Zhang, L., Ke, X., & Ding, H. (2021). *Neural,
   Symbolic and Neural-Symbolic Reasoning on Knowledge Graphs*. AI Open, 2021.
   arXiv:2010.05446.

3. Luo, L., Zhao, Z., Haffari, G., Li, Y.-F., Gong, C., & Pan, S. (2025).
   *Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with
   Large Language Models*. ICML 2025. arXiv:2410.13080.

4. Yu, F. (2025). *What Really Matters to Better GraphRAG Implementation? —
   Part 1*. Medium.
   https://medium.com/@yu-joshua/what-really-matters-to-better-graphrag-implementation-part-1-e02fff773c48

### Additional References from Surveys

- Edge, D. et al. (2024). *From Local to Global: A Graph RAG Approach to
  Query-Focused Summarization*. arXiv:2404.16130.
- Yasunaga, M. et al. (2021). *QA-GNN: Reasoning with Language Models and
  Knowledge Graphs for Question Answering*. arXiv:2104.06378.
- Sun, J. et al. (2023). *Think-on-Graph: Deep and Responsible Reasoning of
  Large Language Model with Knowledge Graph*. arXiv:2307.07697.
- Teru, K., Denis, E., & Hamilton, W. (2020). *Inductive Relation Prediction
  by Subgraph Reasoning*. ICML 2020.
- Lao, N., Mitchell, T., & Cohen, W. (2011). *Random Walk Inference and
  Learning in A Large Scale Knowledge Base*. EMNLP 2011.
