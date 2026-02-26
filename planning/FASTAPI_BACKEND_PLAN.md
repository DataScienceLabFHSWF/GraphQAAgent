# FastAPI Backend Plan — TypeScript Frontend Integration

## Current State (What Already Exists)

The API layer at `src/kgrag/api/` already has a working FastAPI app:

| File | What it does | Status |
|------|-------------|--------|
| `server.py` | FastAPI app factory, lifespan (Orchestrator startup/shutdown), CORS | ✅ Working |
| `routes.py` | `POST /api/v1/ask` (single-shot QA), `GET /api/v1/health` | ✅ Working |
| `schemas.py` | `QuestionRequest`, `AnswerResponse`, `ProvenanceResponse` | ✅ Working |
| `chat_routes.py` | `POST /chat/send` (SSE stream + JSON), session CRUD | ✅ Working |
| `chat_schemas.py` | `ChatRequest`, `ChatResponse`, `ChatStreamEvent`, `FeedbackRequest` | ✅ Working |
| `explorer_routes.py` | KG entity browsing, subgraph viz, law graph, ontology tree | ✅ Working |
| `chat/session.py` | Multi-turn `ChatSessionManager`, conversation context | ✅ Working |
| `chat/streaming.py` | SSE streaming (reasoning steps → tokens → provenance → subgraph → done) | ✅ Working |
| `chat/history.py` | Pluggable persistence backend (JSON file store) | ✅ Working |

**Run command:** `uvicorn kgrag.api.server:app --host 0.0.0.0 --port 8000`

---

## What the TypeScript Frontend Needs

A chat-based frontend for the Agentic GraphRAG pipeline needs these capabilities:

### 1. Chat Interface
- Send a message, get back a streamed answer with reasoning
- Multi-turn context (follow-up questions understand prior context)
- Session management (create, resume, delete conversations)

### 2. Rich Answer Display
- **Answer text** (streamed token by token)
- **Reasoning trace** (step-by-step CoT or ReAct steps, shown live as they arrive)
- **Source provenance** (which documents/chunks/entities were used, with scores)
- **Cited entities** (entity cards with type, description, links to KG explorer)
- **Cited relations** (relationship edges grounding the answer)
- **Subgraph visualization** (vis.js / Cytoscape-compatible JSON for interactive graph)
- **Confidence indicator** (0-1 score + verification faithfulness)
- **Gap detection alerts** (when the system detects missing knowledge)

### 3. KG Explorer Sidebar
- Browse entities by type, search by name
- Click an entity → see its detail card + local subgraph
- Browse the law graph (Gesetzbuch → Paragraf hierarchy)
- View ontology class tree

### 4. Settings / Strategy Picker
- Pick retrieval strategy (or use default `hybrid_sota`)
- Toggle reasoning display, subgraph display
- Language selector (de/en)

---

## Gap Analysis — What Needs to Change

### Phase 1: Enrich Chat Response Payloads ⬅ Priority

The current `ChatResponse` returns answer text, confidence, reasoning chain, basic provenance, and subgraph. The TypeScript frontend needs **richer data** to render entity cards, source panels, and verification badges.

**Changes needed in `chat_schemas.py`:**

```python
# New schemas for rich answer context

class EntityResponse(BaseModel):
    """Entity card data for the frontend."""
    id: str
    label: str
    entity_type: str
    description: str = ""
    properties: dict[str, Any] = {}

class RelationResponse(BaseModel):
    """Relation edge for graph visualization."""
    source_id: str
    target_id: str
    relation_type: str
    confidence: float = 0.0

class EvidenceResponse(BaseModel):
    """A single piece of retrieved evidence with full provenance."""
    text: str
    score: float = 0.0
    source: str  # "vector" | "graph" | "hybrid" | "ontology"
    doc_id: str | None = None
    source_id: str | None = None

class VerificationResponse(BaseModel):
    """Answer verification / faithfulness result."""
    is_faithful: bool
    faithfulness_score: float
    supported_claims: list[str] = []
    unsupported_claims: list[str] = []

class ReasoningStepResponse(BaseModel):
    """Structured reasoning step (CoT or ReAct)."""
    step_id: int
    sub_question: str
    evidence_text: str = ""
    answer_fragment: str = ""
    confidence: float = 0.0

class GapDetectionResponse(BaseModel):
    """HITL gap detection alert."""
    gap_type: str  # "tbox_missing_class", "abox_weak_evidence", etc.
    description: str
    affected_entities: list[str] = []

# Enriched ChatResponse
class ChatResponse(BaseModel):
    session_id: str
    message: ChatMessage
    confidence: float = 0.0
    reasoning_chain: list[str] = []
    reasoning_steps: list[ReasoningStepResponse] = []   # NEW
    provenance: list[ProvenanceResponse] = []
    evidence: list[EvidenceResponse] = []                # NEW
    cited_entities: list[EntityResponse] = []            # NEW
    cited_relations: list[RelationResponse] = []         # NEW
    subgraph: dict[str, Any] | None = None
    verification: VerificationResponse | None = None     # NEW
    gap_detection: GapDetectionResponse | None = None    # NEW
    strategy_used: str = ""                              # NEW
    latency_ms: float = 0.0
```

**Changes needed in `chat/session.py`:**
- `process_message()` must extract and pass through the new fields from `QAAnswer`

**Changes needed in `chat/streaming.py`:**
- Add new SSE event types: `evidence`, `entities`, `verification`, `gap_alert`
- Emit them between `provenance` and `done`

### Phase 2: WebSocket Support (Optional)

SSE is unidirectional (server→client). For features like:
- Typing indicators
- Client-side cancellation of in-progress queries
- Bidirectional HITL feedback during answer generation

A WebSocket endpoint would be better:

```
WS /api/v1/chat/ws/{session_id}

Client → Server:  { "type": "message", "text": "...", "strategy": "hybrid_sota" }
Client → Server:  { "type": "cancel" }
Client → Server:  { "type": "feedback", ... }

Server → Client:  { "type": "reasoning_step", "data": {...} }
Server → Client:  { "type": "token", "data": {"text": "..."} }
Server → Client:  { "type": "evidence", "data": [...] }
Server → Client:  { "type": "entities", "data": [...] }
Server → Client:  { "type": "subgraph", "data": {...} }
Server → Client:  { "type": "verification", "data": {...} }
Server → Client:  { "type": "done", "data": {"confidence": 0.85, "latency_ms": 2340} }
Server → Client:  { "type": "error", "data": {"message": "..."} }
```

### Phase 3: CORS & Security Configuration

Current CORS allows `localhost:3000` (Next.js). Needs:
- Configurable `KGRAG_CORS_ORIGINS` env var for production deployments
- Optional API key / JWT auth middleware (if needed)

### Phase 4: OpenAPI Polish

FastAPI auto-generates `/docs` (Swagger) and `/openapi.json`. Should ensure:
- All response models are correctly typed for TypeScript codegen
- Endpoint descriptions are clear
- The TS frontend can run `npx openapi-typescript http://localhost:8000/openapi.json -o api-types.ts` to generate type-safe client code

---

## Implementation Sequence

```
Phase 1a: Enrich schemas (EntityResponse, EvidenceResponse, etc.)     ~30 min
Phase 1b: Update session.py to extract full context from QAAnswer     ~20 min
Phase 1c: Update streaming.py with new SSE events                     ~20 min
Phase 1d: Update chat_routes.py non-streaming path                    ~10 min
Phase 1e: Add tests for enriched responses                            ~30 min

Phase 2a: WebSocket endpoint in chat_routes.py                        ~45 min
Phase 2b: Client cancel / heartbeat support                           ~20 min
Phase 2c: WebSocket tests                                             ~20 min

Phase 3:  Configurable CORS from env + optional auth middleware       ~20 min

Phase 4:  OpenAPI cleanup + TypeScript type generation docs           ~15 min
```

---

## Endpoint Summary (After Phases 1-4)

### Chat
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/chat/send` | Send message (SSE stream or JSON) |
| `WS` | `/api/v1/chat/ws/{session_id}` | WebSocket chat (Phase 2) |
| `GET` | `/api/v1/chat/sessions` | List active sessions |
| `GET` | `/api/v1/chat/sessions/{id}/history` | Conversation history |
| `DELETE` | `/api/v1/chat/sessions/{id}` | Delete session |
| `POST` | `/api/v1/chat/feedback` | HITL feedback |

### QA (Single-shot)
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/ask` | Single question → full answer |
| `GET` | `/api/v1/health` | Health check |

### KG Explorer
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/explore/entities` | List/search entities |
| `GET` | `/api/v1/explore/entities/{id}` | Entity detail + relations |
| `GET` | `/api/v1/explore/entities/{id}/subgraph` | Vis.js subgraph JSON |
| `GET` | `/api/v1/explore/relations` | Relation type counts |
| `GET` | `/api/v1/explore/stats` | KG statistics |
| `GET` | `/api/v1/explore/laws` | Law list |
| `GET` | `/api/v1/explore/laws/{id}/structure` | Law → Paragraf tree |
| `GET` | `/api/v1/explore/laws/{id}/linked-entities` | Domain entities per law |
| `GET` | `/api/v1/explore/ontology/classes` | Ontology class list |
| `GET` | `/api/v1/explore/ontology/classes/{uri}/properties` | Class properties |
| `GET` | `/api/v1/explore/ontology/tree` | Ontology hierarchy tree |

### Strategies
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/strategies` | List available strategies with descriptions |

---

## SSE Event Sequence (Enriched)

```
1. event: session       → { session_id: "abc123" }
2. event: reasoning_step → { step: 1, sub_question: "...", evidence: "...", confidence: 0.8 }
   ...                     (0-N reasoning steps)
3. event: token          → { text: "The " }
   ...                     (1-N tokens — answer text streamed word-by-word)
4. event: evidence       → [{ text: "...", score: 0.92, source: "vector", doc_id: "..." }, ...]
5. event: entities       → [{ id: "ent_abc", label: "KKW Obrigheim", type: "Facility", ... }, ...]
6. event: relations      → [{ source_id: "...", target_id: "...", relation_type: "..." }, ...]
7. event: provenance     → [{ source: "vector", score: 0.92, doc_id: "..." }, ...]
8. event: subgraph       → { nodes: [...], edges: [...] }
9. event: verification   → { is_faithful: true, faithfulness_score: 0.85, ... }
10. event: gap_alert     → { gap_type: "abox_weak_evidence", description: "..." }  (if detected)
11. event: done          → { confidence: 0.85, latency_ms: 2340, strategy: "hybrid_sota" }
```

---

## TypeScript Client Usage (Example)

```typescript
// Auto-generated types from OpenAPI
import type { ChatResponse, EntityResponse, EvidenceResponse } from './api-types';

// SSE streaming
const eventSource = new EventSource('/api/v1/chat/send', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    session_id: 'abc123',
    message: 'Was ist das Atomgesetz?',
    strategy: 'hybrid_sota',
    stream: true,
  }),
});

eventSource.addEventListener('token', (e) => {
  appendToAnswer(JSON.parse(e.data).text);
});
eventSource.addEventListener('reasoning_step', (e) => {
  addReasoningStep(JSON.parse(e.data));
});
eventSource.addEventListener('entities', (e) => {
  renderEntityCards(JSON.parse(e.data));
});
eventSource.addEventListener('subgraph', (e) => {
  renderGraph(JSON.parse(e.data));  // vis.js / Cytoscape
});
eventSource.addEventListener('done', (e) => {
  finalize(JSON.parse(e.data));
});
```

---

## Notes for the TS Frontend Developer

1. **CORS**: Backend allows `localhost:3000` by default. Set `KGRAG_CORS_ORIGINS` for other origins.
2. **Type generation**: Run `npx openapi-typescript http://localhost:8000/openapi.json -o src/api-types.ts` to keep types in sync.
3. **Subgraph JSON format**: `{ nodes: [{id, label, type}], edges: [{source, target, label}] }` — compatible with vis.js Network and Cytoscape.js.
4. **Strategy recommendation**: Default to `hybrid_sota` for best results. Show strategy picker as advanced option.
5. **Language**: Pass `language: "de"` or `language: "en"` in the chat request. The KG content is predominantly German.
6. **Session lifecycle**: Sessions are in-memory. Create on first message, resume by passing the same `session_id`. Call `DELETE` to clean up.
