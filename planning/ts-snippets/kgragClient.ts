// lib/kgragClient.ts — Drop-in replacement for GeminiChatClient in gaia-tt

const API_BASE =
  process.env.NEXT_PUBLIC_KGRAG_API_URL ?? "http://localhost:8000/api/v1";

// ── Types matching our FastAPI schemas ──────────────────────────────

export interface KGRAGChatRequest {
  session_id?: string | null;
  message: string;
  strategy?: string; // "hybrid_sota" | "vector_only" | "graph_only" | ...
  language?: string; // "de" | "en"
  stream?: boolean;
  include_reasoning?: boolean;
  include_subgraph?: boolean;
  include_evidence?: boolean;
}

export interface EntityResponse {
  id: string;
  label: string;
  entity_type: string;
  description: string;
  properties: Record<string, unknown>;
}

export interface RelationResponse {
  source_id: string;
  target_id: string;
  relation_type: string;
  confidence: number;
}

export interface EvidenceResponse {
  text: string;
  score: number;
  source: string;
  doc_id?: string | null;
  source_id?: string | null;
}

export interface ReasoningStepResponse {
  step_id: number;
  sub_question: string;
  evidence_text: string;
  answer_fragment: string;
  confidence: number;
}

export interface VerificationResponse {
  is_faithful: boolean;
  faithfulness_score: number;
  supported_claims: string[];
  unsupported_claims: string[];
  contradicted_claims: string[];
  entity_coverage: number;
}

export interface GapDetectionResponse {
  gap_type: string;
  description: string;
  affected_entities: string[];
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp?: string | null;
  metadata?: Record<string, unknown> | null;
}

export interface KGRAGChatResponse {
  session_id: string;
  message: ChatMessage;
  confidence: number;
  latency_ms: number;
  strategy_used: string;
  reasoning_chain: string[];
  reasoning_steps: ReasoningStepResponse[];
  evidence: EvidenceResponse[];
  cited_entities: EntityResponse[];
  cited_relations: RelationResponse[];
  subgraph: { nodes: unknown[]; edges: unknown[] } | null;
  verification: VerificationResponse | null;
  gap_detection: GapDetectionResponse | null;
}

export interface StrategyInfo {
  id: string;
  display_name: string;
  description: string;
}

// ── Non-streaming: full JSON response ───────────────────────────────

export async function sendMessage(
  message: string,
  sessionId?: string,
  strategy = "hybrid_sota",
): Promise<KGRAGChatResponse> {
  const res = await fetch(`${API_BASE}/chat/send`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      session_id: sessionId ?? null,
      strategy,
      language: "de",
      stream: false,
      include_reasoning: true,
      include_subgraph: true,
      include_evidence: true,
    } satisfies KGRAGChatRequest),
  });

  if (!res.ok) {
    throw new Error(`KG-RAG API error: ${res.status} ${await res.text()}`);
  }
  return res.json();
}

// ── SSE streaming ───────────────────────────────────────────────────

export type SSEEvent =
  | { event: "session"; data: { session_id: string } }
  | {
      event: "reasoning_step";
      data: ReasoningStepResponse | { step: number; text: string };
    }
  | { event: "token"; data: { text: string } }
  | { event: "evidence"; data: EvidenceResponse[] }
  | { event: "entities"; data: EntityResponse[] }
  | { event: "relations"; data: RelationResponse[] }
  | { event: "provenance"; data: unknown[] }
  | { event: "subgraph"; data: { nodes: unknown[]; edges: unknown[] } }
  | { event: "verification"; data: VerificationResponse }
  | { event: "gap_alert"; data: GapDetectionResponse }
  | {
      event: "done";
      data: {
        confidence: number;
        latency_ms: number;
        strategy: string;
        evidence_count: number;
        entity_count: number;
      };
    }
  | { event: "error"; data: { message: string } };

export async function streamMessage(
  message: string,
  onEvent: (evt: SSEEvent) => void,
  sessionId?: string,
  strategy = "hybrid_sota",
): Promise<void> {
  const res = await fetch(`${API_BASE}/chat/send`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      session_id: sessionId ?? null,
      strategy,
      language: "de",
      stream: true,
      include_reasoning: true,
      include_subgraph: true,
      include_evidence: true,
    } satisfies KGRAGChatRequest),
  });

  if (!res.ok) {
    throw new Error(`KG-RAG API error: ${res.status} ${await res.text()}`);
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";

    for (const part of parts) {
      const eventMatch = part.match(/^event: (.+)$/m);
      const dataMatch = part.match(/^data: (.+)$/m);
      if (eventMatch && dataMatch) {
        onEvent({
          event: eventMatch[1] as SSEEvent["event"],
          data: JSON.parse(dataMatch[1]),
        } as SSEEvent);
      }
    }
  }
}

// ── Strategies list ─────────────────────────────────────────────────

export async function fetchStrategies(): Promise<{
  strategies: StrategyInfo[];
  default: string;
}> {
  const res = await fetch(`${API_BASE}/strategies`);
  if (!res.ok) throw new Error(`Failed to fetch strategies: ${res.status}`);
  return res.json();
}
