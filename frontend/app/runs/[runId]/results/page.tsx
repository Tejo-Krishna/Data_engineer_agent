"use client";
import { useParams } from "next/navigation";
import Link from "next/link";
import { shortId } from "@/lib/utils";
import QualityTab from "@/components/results/QualityTab";
import LineageTab from "@/components/results/LineageTab";
import DbtTab from "@/components/results/DbtTab";
import DownloadsTab from "@/components/results/DownloadsTab";
import { useState } from "react";
import { cn } from "@/lib/utils";

const TABS = [
  { key: "quality",   label: "Quality Report" },
  { key: "lineage",   label: "Lineage" },
  { key: "dbt",       label: "dbt Models" },
  { key: "downloads", label: "Downloads" },
];

export default function ResultsPage() {
  const { runId } = useParams<{ runId: string }>();
  const [tab, setTab] = useState("quality");

  return (
    <div className="max-w-5xl mx-auto py-10 px-4 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <Link href={`/runs/${runId}`} className="text-xs text-gray-400 hover:underline">← Back to run</Link>
          <h1 className="text-xl font-bold mt-1">Results <span className="font-mono text-sm text-gray-400">{shortId(runId)}</span></h1>
        </div>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 border-b">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={cn(
              "px-4 py-2 text-sm -mb-px border-b-2 transition-colors",
              tab === t.key ? "border-black font-medium" : "border-transparent text-gray-500 hover:text-black"
            )}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div>
        {tab === "quality"   && <QualityTab runId={runId} />}
        {tab === "lineage"   && <LineageTab runId={runId} />}
        {tab === "dbt"       && <DbtTab runId={runId} />}
        {tab === "downloads" && <DownloadsTab runId={runId} />}
      </div>
    </div>
  );
}
