"use client";
import { AGENTS, type AgentKey } from "@/lib/constants";
import { cn } from "@/lib/utils";

type AgentState = "idle" | "running" | "waiting_hitl" | "complete" | "failed";

const stateStyle: Record<AgentState, string> = {
  idle:         "border-gray-200 bg-white text-gray-400",
  running:      "border-blue-300 bg-blue-50 text-blue-700",
  waiting_hitl: "border-yellow-300 bg-yellow-50 text-yellow-700",
  complete:     "border-green-300 bg-green-50 text-green-700",
  failed:       "border-red-300 bg-red-50 text-red-700",
};

const stateIcon: Record<AgentState, string> = {
  idle:         "○",
  running:      "⟳",
  waiting_hitl: "⏸",
  complete:     "✓",
  failed:       "✗",
};

interface Props {
  agentStates: Record<AgentKey, AgentState>;
}

export default function AgentTimeline({ agentStates }: Props) {
  return (
    <div className="flex items-start gap-2 flex-wrap">
      {AGENTS.map((agent, i) => {
        const state = agentStates[agent.key];
        return (
          <div key={agent.key} className="flex items-center gap-2">
            <div className={cn("rounded-lg border px-4 py-3 min-w-[120px]", stateStyle[state])}>
              <div className="flex items-center gap-1.5">
                {state === "complete" ? (
                  <span className="relative flex h-2.5 w-2.5 flex-shrink-0">
                    <span className="absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-60 animate-ping" />
                    <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-green-500" />
                  </span>
                ) : (
                  <span className={cn("text-base", state === "running" && "animate-spin inline-block")}>
                    {stateIcon[state]}
                  </span>
                )}
                <span className="text-sm font-medium">{agent.label}</span>
              </div>
              <p className="text-xs mt-0.5 opacity-70">{agent.description}</p>
            </div>
            {i < AGENTS.length - 1 && (
              <span className="text-gray-300 text-lg">→</span>
            )}
          </div>
        );
      })}
    </div>
  );
}
