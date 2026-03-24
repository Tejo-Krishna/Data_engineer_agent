"use client";
import { useEffect, useRef, useState } from "react";
import type { PipelineEvent } from "@/lib/types";
import type { AgentKey } from "@/lib/constants";

type AgentState = "idle" | "running" | "waiting_hitl" | "complete" | "failed";

export function useRunEvents(runId: string | null) {
  const [events, setEvents] = useState<PipelineEvent[]>([]);
  const [agentStates, setAgentStates] = useState<Record<AgentKey, AgentState>>({
    profiler: "idle", domain: "idle", transformer: "idle", quality: "idle", catalogue: "idle",
  });
  const [hitlRequired, setHitlRequired] = useState<"code" | "drift" | null>(null);
  const [runStatus, setRunStatus] = useState<string>("running");
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!runId) return;

    const es = new EventSource(`/api/runs/${runId}/events`);
    esRef.current = es;

    es.onmessage = (e) => {
      const event: PipelineEvent = JSON.parse(e.data);
      setEvents((prev) => [...prev, event]);

      if (event.type === "agent_started") {
        setAgentStates((s) => ({ ...s, [event.agent as AgentKey]: "running" }));
      }
      if (event.type === "agent_complete") {
        setAgentStates((s) => ({ ...s, [event.agent as AgentKey]: "complete" }));
      }
      if (event.type === "agent_failed") {
        setAgentStates((s) => ({ ...s, [event.agent as AgentKey]: "failed" }));
      }
      if (event.type === "hitl_required") {
        const agent = event.agent as AgentKey;
        setAgentStates((s) => ({ ...s, [agent]: "waiting_hitl" }));
        setHitlRequired(event.agent === "transformer" ? "code" : "drift");
      }
      if (event.type === "hitl_resolved") {
        setHitlRequired(null);
      }
      if (event.type === "run_complete" || event.type === "run_failed") {
        setRunStatus(event.status ?? (event.type === "run_complete" ? "success" : "failed"));
        es.close();
      }
    };

    es.onerror = () => es.close();

    return () => { es.close(); esRef.current = null; };
  }, [runId]);

  return { events, agentStates, hitlRequired, runStatus };
}
