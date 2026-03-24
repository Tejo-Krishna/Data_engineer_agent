"use client";
import type { PipelineEvent } from "@/lib/types";

export default function ToolCallLog({ events }: { events: PipelineEvent[] }) {
  const toolEvents = events.filter((e) => e.type === "tool_called");

  if (!toolEvents.length) {
    return <p className="text-sm text-gray-400 py-4">Waiting for tool calls…</p>;
  }

  return (
    <div className="space-y-1 max-h-64 overflow-y-auto text-sm font-mono">
      {toolEvents.map((e, i) => (
        <div key={i} className="flex items-center gap-3 py-1 border-b last:border-0">
          <span className="text-gray-400 text-xs w-16 shrink-0">{e.agent}</span>
          <span className="text-black">{e.tool}</span>
        </div>
      ))}
    </div>
  );
}
