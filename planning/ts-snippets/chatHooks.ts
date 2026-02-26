// Example Chat.tsx integration — shows both non-streaming and streaming usage
// Replace the existing Gemini handleSend in gaia-tt's Chat.tsx

import { useState, useCallback } from "react";
import {
  sendMessage,
  streamMessage,
  fetchStrategies,
  type SSEEvent,
  type KGRAGChatResponse,
  type EvidenceResponse,
  type EntityResponse,
  type VerificationResponse,
  type GapDetectionResponse,
} from "@/lib/kgragClient";

// ── State hooks (add these to your Chat component) ──────────────────

/*
const [sessionId, setSessionId] = useState<string | undefined>();
const [selectedStrategy, setSelectedStrategy] = useState("hybrid_sota");
const [loading, setLoading] = useState(false);
const [streamingText, setStreamingText] = useState("");

// Rich context panels
const [evidence, setEvidence] = useState<EvidenceResponse[]>([]);
const [entities, setEntities] = useState<EntityResponse[]>([]);
const [subgraph, setSubgraph] = useState<{ nodes: unknown[]; edges: unknown[] } | null>(null);
const [verification, setVerification] = useState<VerificationResponse | null>(null);
const [gapAlert, setGapAlert] = useState<GapDetectionResponse | null>(null);
const [confidence, setConfidence] = useState(0);
*/

// ── Non-streaming (simplest drop-in for current Gemini flow) ────────

export function useNonStreamingChat() {
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<
    { id: string; type: "user" | "assistant"; content: string; timestamp: number }[]
  >([]);
  const [lastResponse, setLastResponse] = useState<KGRAGChatResponse | null>(null);

  const handleSend = useCallback(
    async (input: string, strategy = "hybrid_sota") => {
      const userMsg = {
        id: crypto.randomUUID(),
        type: "user" as const,
        content: input,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setLoading(true);

      try {
        const resp = await sendMessage(input, sessionId, strategy);
        setSessionId(resp.session_id);
        setLastResponse(resp);

        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            type: "assistant",
            content: resp.message.content,
            timestamp: Date.now(),
          },
        ]);
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            type: "assistant",
            content: `Error: ${err instanceof Error ? err.message : String(err)}`,
            timestamp: Date.now(),
          },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [sessionId],
  );

  return { messages, loading, handleSend, sessionId, lastResponse };
}

// ── Streaming (progressive token rendering + context panels) ────────

export function useStreamingChat() {
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [loading, setLoading] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [messages, setMessages] = useState<
    { id: string; type: "user" | "assistant"; content: string; timestamp: number }[]
  >([]);

  // Rich context
  const [evidence, setEvidence] = useState<EvidenceResponse[]>([]);
  const [entities, setEntities] = useState<EntityResponse[]>([]);
  const [subgraph, setSubgraph] = useState<{ nodes: unknown[]; edges: unknown[] } | null>(null);
  const [verification, setVerification] = useState<VerificationResponse | null>(null);
  const [gapAlert, setGapAlert] = useState<GapDetectionResponse | null>(null);
  const [confidence, setConfidence] = useState(0);

  const handleSend = useCallback(
    async (input: string, strategy = "hybrid_sota") => {
      const userMsg = {
        id: crypto.randomUUID(),
        type: "user" as const,
        content: input,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setLoading(true);
      setStreamingText("");
      setEvidence([]);
      setEntities([]);
      setSubgraph(null);
      setVerification(null);
      setGapAlert(null);

      let tokens = "";

      try {
        await streamMessage(
          input,
          (evt: SSEEvent) => {
            switch (evt.event) {
              case "session":
                setSessionId(evt.data.session_id);
                break;
              case "token":
                tokens += evt.data.text;
                setStreamingText(tokens);
                break;
              case "evidence":
                setEvidence(evt.data);
                break;
              case "entities":
                setEntities(evt.data);
                break;
              case "subgraph":
                setSubgraph(evt.data);
                break;
              case "verification":
                setVerification(evt.data);
                break;
              case "gap_alert":
                setGapAlert(evt.data);
                break;
              case "done":
                setConfidence(evt.data.confidence);
                break;
              case "error":
                tokens += `\n\n⚠️ ${evt.data.message}`;
                setStreamingText(tokens);
                break;
            }
          },
          sessionId,
          strategy,
        );

        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            type: "assistant",
            content: tokens,
            timestamp: Date.now(),
          },
        ]);
        setStreamingText("");
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            type: "assistant",
            content: `Error: ${err instanceof Error ? err.message : String(err)}`,
            timestamp: Date.now(),
          },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [sessionId],
  );

  return {
    messages,
    loading,
    handleSend,
    sessionId,
    streamingText,
    evidence,
    entities,
    subgraph,
    verification,
    gapAlert,
    confidence,
  };
}
