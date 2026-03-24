"use client";
import { useEffect, useRef, useState } from "react";
import { getLineage } from "@/lib/api";

export default function LineageTab({ runId }: { runId: string }) {
  const [diagram, setDiagram] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getLineage(runId).then((d) => setDiagram(d.mermaid_diagram)).catch((e) => setError(e.message));
  }, [runId]);

  useEffect(() => {
    if (!diagram || !containerRef.current) return;
    import("mermaid").then((m) => {
      m.default.initialize({ startOnLoad: false, theme: "neutral" });
      m.default.render("lineage-diagram", diagram).then(({ svg }) => {
        if (containerRef.current) containerRef.current.innerHTML = svg;
      }).catch(() => setError("Could not render diagram"));
    });
  }, [diagram]);

  if (error) return (
    <div className="space-y-2">
      <p className="text-sm text-red-500">{error}</p>
      {diagram && <pre className="text-xs font-mono bg-gray-50 rounded p-4 overflow-auto">{diagram}</pre>}
    </div>
  );

  if (!diagram) return <p className="text-sm text-gray-400">Lineage not available yet</p>;

  return (
    <div className="space-y-3">
      <div ref={containerRef} className="border rounded-lg p-4 overflow-auto min-h-[200px]" />
      <details className="text-xs">
        <summary className="cursor-pointer text-gray-400">View Mermaid source</summary>
        <pre className="mt-2 bg-gray-50 rounded p-3 overflow-auto">{diagram}</pre>
      </details>
    </div>
  );
}
