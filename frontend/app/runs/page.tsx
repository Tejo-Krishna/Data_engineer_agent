"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { listRuns } from "@/lib/api";
import type { RunSummary } from "@/lib/types";
import RunStatusBadge from "@/components/shared/RunStatusBadge";
import { formatDuration, shortId } from "@/lib/utils";

export default function RunsPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listRuns().then(setRuns).finally(() => setLoading(false));
    const interval = setInterval(() => listRuns().then(setRuns), 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="max-w-4xl mx-auto py-10 px-4">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-bold">Pipeline Runs</h1>
        <Link href="/wizard" className="px-4 py-2 bg-black text-white text-sm rounded">+ New Run</Link>
      </div>

      {loading ? (
        <p className="text-sm text-gray-400">Loading…</p>
      ) : runs.length === 0 ? (
        <div className="text-center py-20 text-gray-400">
          <p className="text-4xl mb-3">○</p>
          <p>No runs yet. <Link href="/wizard" className="underline">Start your first pipeline.</Link></p>
        </div>
      ) : (
        <div className="border rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 border-b">
              <tr>
                <th className="text-left px-4 py-3 font-medium text-gray-500">Run</th>
                <th className="text-left px-4 py-3 font-medium text-gray-500">Source</th>
                <th className="text-left px-4 py-3 font-medium text-gray-500">Status</th>
                <th className="text-left px-4 py-3 font-medium text-gray-500">Rows</th>
                <th className="text-left px-4 py-3 font-medium text-gray-500">Duration</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.id} className="border-b last:border-0 hover:bg-gray-50">
                  <td className="px-4 py-3">
                    <Link href={r.status === "success" ? `/runs/${r.id}/results` : `/runs/${r.id}`} className="font-mono text-xs hover:underline">
                      {shortId(r.id)}
                    </Link>
                  </td>
                  <td className="px-4 py-3 text-gray-600 max-w-[200px] truncate">{r.source_path.split("/").pop()}</td>
                  <td className="px-4 py-3"><RunStatusBadge status={r.status} /></td>
                  <td className="px-4 py-3 text-gray-500">{r.rows_input != null ? `${r.rows_input} → ${r.rows_output}` : "—"}</td>
                  <td className="px-4 py-3 text-gray-500">{formatDuration(r.started_at, r.completed_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
