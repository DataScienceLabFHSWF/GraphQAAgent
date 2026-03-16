# Evaluation Plan — Paper C (Graph QA Agent)

**Goal**: Produce quantitative results for the conference paper on graph-based question answering.  
**Owner**: Ferdinand (GraphRAG evaluation), Ole (UI + frontend metrics)  
**Benchmark queue position**: Step 3 (runs last, ~Apr 3–10)  
**Resource constraint**: Benchmarks run sequentially — only this suite should be running during its window.  
**Paper status**: STRETCH goal for April — only pursue if Papers A + B are on track.

---

## Prerequisites

### 1. Services (Docker Compose)

This repo has the most services — 6 containers:

```bash
docker compose up -d
```

| Service | Port | Check |
|---------|------|-------|
| Neo4j | 7474 (browser), 7687 (bolt) | `curl http://localhost:7474` |
| Qdrant | 6333 | `curl http://localhost:6333/health` |
| Fuseki | 3030 | `curl http://localhost:3030/$/ping` |
| Ollama | 11434 | `curl http://localhost:11434/api/tags` |
| qa-agent (backend) | 8080 | `curl http://localhost:8080/health` |
| frontend (Streamlit) | 8501 | Open in browser |

### 2. Models

```bash
docker exec -it ollama ollama pull qwen3:8b
docker exec -it ollama ollama pull llama3.2:3b
```

### 3. Knowledge Graph Prerequisite

**Critical**: The QA agent needs a populated knowledge graph to answer questions against. You need a KG from the KnowledgeGraphBuilder pipeline loaded into Neo4j before running evaluations.

```bash
# Verify KG is populated
docker exec -it neo4j cypher-shell -u neo4j -p changeme \
  "MATCH (n) RETURN count(n) AS nodes"
# Should return > 0 nodes
```

### 4. Benchmark Dataset

Three benchmark versions exist:
- `benchmarks/v1/` — 1 placeholder question (do NOT use)
- `benchmarks/v2/` — 27KB, usable
- `benchmarks/v3/` — 38KB, most comprehensive (recommended)

Benchmark format (JSON):
```json
{
  "question_id": "q001",
  "question": "What materials are used in...",
  "expected_answer": "...",
  "expected_entities": ["Entity1", "Entity2"],
  "difficulty": "medium",
  "question_type": "factual",
  "competency_question_id": "cq_003"
}
```

---

## Phase 1: Benchmark Dataset Review & Extension

**Who**: Ferdinand  
**Time**: ~4 hours  
**Priority**: Do this FIRST — evaluation quality depends on benchmark quality.

### Review existing benchmarks

```bash
# Count questions per version
python -c "
import json, pathlib
for v in ['v2', 'v3']:
    p = pathlib.Path(f'benchmarks/{v}')
    for f in p.glob('*.json'):
        data = json.load(open(f))
        qs = data if isinstance(data, list) else data.get('questions', [])
        print(f'{v}/{f.name}: {len(qs)} questions')
"
```

### Extend if needed

For a strong evaluation, aim for:
- **≥50 questions** total
- Mix of difficulty levels (easy/medium/hard)
- Mix of question types (factual, comparative, multi-hop, aggregation)
- Coverage of different KG domains/entity types

If v3 is insufficient, create `benchmarks/v4/` with additional questions derived from the KG's competency questions.

---

## Phase 2: Strategy Evaluation (Core Evaluation)

**Who**: Ferdinand  
**Time**: ~2–3 days  
**Script**: `scripts/run_evaluation.py`

### Run full evaluation across all strategies

```bash
python scripts/run_evaluation.py \
  --benchmark benchmarks/v3/ \
  --strategies all \
  --output-dir results/full_evaluation/ \
  --formats markdown json
```

### 7 QA Strategies to evaluate:

1. **Direct Cypher** — LLM generates Cypher query directly
2. **Entity-first** — extract entities → lookup → answer
3. **GraphRAG** — graph-based retrieval augmented generation (Ferdinand's focus)
4. **Hybrid search** — combine vector + graph retrieval
5. **Multi-hop** — iterative graph traversal for complex questions
6. **SPARQL fallback** — federated query via Fuseki when Cypher insufficient
7. **Ensemble** — combine multiple strategies and vote

### Expected output per strategy:
- Accuracy (exact match)
- F1 score (partial match on expected entities)
- Latency (mean, p50, p95, p99)
- Token usage (input + output)
- Error rate and error categories

---

## Phase 3: Head-to-Head Strategy Comparison

**Who**: Ferdinand  
**Time**: ~2 hours  
**Script**: `scripts/compare_strategies.py`

Run targeted comparisons on specific challenging questions:

```bash
# Compare all strategies on a single question
python scripts/compare_strategies.py \
  --question "What decommissioning methods were used at facility X?" \
  --strategies direct_cypher entity_first graphrag hybrid

# Compare on multi-hop questions
python scripts/compare_strategies.py \
  --question "Which facilities share materials with those decommissioned before 2010?" \
  --strategies multi_hop graphrag ensemble
```

This produces side-by-side output showing each strategy's answer, reasoning trace, and latency. Useful for qualitative analysis and cherry-picking examples for the paper.

---

## Phase 4: Difficulty-Stratified Analysis

**Who**: Ole  
**Time**: ~4 hours

After Phase 2 completes, analyze results by difficulty and question type:

```python
import json
import pandas as pd

results = json.load(open("results/full_evaluation/results.json"))
df = pd.DataFrame(results)

# Accuracy by difficulty
print(df.groupby(["strategy", "difficulty"])["correct"].mean())

# Accuracy by question type
print(df.groupby(["strategy", "question_type"])["correct"].mean())

# Latency by strategy
print(df.groupby("strategy")["latency_ms"].describe())
```

### Charts to produce (for paper):
1. **Bar chart**: Accuracy per strategy (grouped by difficulty)
2. **Box plot**: Latency distribution per strategy
3. **Heatmap**: Strategy × question_type accuracy matrix
4. **Line chart**: Accuracy vs. latency trade-off frontier

---

## Phase 5: Model Comparison (If Time Allows)

**Who**: Ferdinand  
**Time**: ~1 day per model

Run the full evaluation with different LLM backends:

```bash
# Baseline (small)
OLLAMA_MODEL=llama3.2:3b python scripts/run_evaluation.py \
  --benchmark benchmarks/v3/ --strategies all \
  --output-dir results/eval_llama3.2/

# Primary
OLLAMA_MODEL=qwen3:8b python scripts/run_evaluation.py \
  --benchmark benchmarks/v3/ --strategies all \
  --output-dir results/eval_qwen3/
```

Compare across models to show model-agnosticism of the architecture.

---

## Output Checklist (What Goes Into the Paper)

| Result | Script | Paper Section |
|--------|--------|---------------|
| Strategy comparison table (7 strategies) | `run_evaluation.py` | 4.1 Strategy Comparison |
| Accuracy by difficulty level | Post-processing of results | 4.2 Difficulty Analysis |
| Accuracy by question type | Post-processing of results | 4.2 Difficulty Analysis |
| Latency benchmarks (p50/p95/p99) | `run_evaluation.py` | 4.3 Performance |
| Head-to-head examples (qualitative) | `compare_strategies.py` | 4.4 Qualitative Analysis |
| Model comparison (if done) | `run_evaluation.py` × models | 4.1 Strategy Comparison |
| Error analysis (failure categories) | Manual review of failures | 5 Discussion |
| Accuracy vs latency trade-off chart | Post-processing | 4.3 Performance |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "No nodes in graph" | Load KG from KnowledgeGraphBuilder first — this is a prerequisite |
| qa-agent 500 errors | Check logs: `docker compose logs qa-agent` |
| Cypher generation fails | Model may need few-shot examples — check prompt templates |
| Benchmark questions return empty | Questions may reference entities not in the KG — update benchmark or KG |
| Streamlit frontend not loading | `docker compose restart frontend` |
| Out of memory during evaluation | Run one strategy at a time: `--strategies graphrag` |
| Very slow responses | Check Ollama isn't swapping — verify GPU is being used: `nvidia-smi` |

---

*Created: 2026-03-16*
