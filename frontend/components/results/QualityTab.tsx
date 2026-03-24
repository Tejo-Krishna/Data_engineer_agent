"use client";
import { useEffect, useState } from "react";
import { getQualityReport } from "@/lib/api";
import type { QualityReport } from "@/lib/types";
import { cn } from "@/lib/utils";

export default function QualityTab({ runId }: { runId: string }) {
  const [report, setReport] = useState<QualityReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getQualityReport(runId).then(setReport).catch((e) => setError(e.message));
  }, [runId]);

  if (error) return <p className="text-sm text-red-500">{error}</p>;
  if (!report) return <p className="text-sm text-gray-400">Loading quality report…</p>;

  return (
    <div className="space-y-6">
      <div className={cn("rounded-lg border p-4", report.overall_passed ? "bg-green-50 border-green-200" : "bg-red-50 border-red-200")}>
        <p className={cn("font-semibold", report.overall_passed ? "text-green-700" : "text-red-700")}>
          {report.overall_passed ? "✓ All quality checks passed" : "✗ Quality checks failed"}
        </p>
      </div>

      <div>
        <h3 className="text-sm font-semibold mb-2">Checks</h3>
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="border-b text-left text-gray-500">
              <th className="py-2 pr-4">Check</th>
              <th className="py-2 pr-4">Column</th>
              <th className="py-2 pr-4">Expected</th>
              <th className="py-2 pr-4">Actual</th>
              <th className="py-2">Result</th>
            </tr>
          </thead>
          <tbody>
            {report.checks.map((c, i) => (
              <tr key={i} className="border-b last:border-0">
                <td className="py-2 pr-4 font-mono text-xs">{c.check_name}</td>
                <td className="py-2 pr-4">{c.column}</td>
                <td className="py-2 pr-4 text-gray-500">{c.expected ?? "—"}</td>
                <td className="py-2 pr-4 text-gray-500">{c.actual ?? "—"}</td>
                <td className="py-2">
                  <span className={cn("px-2 py-0.5 rounded-full text-xs", c.passed ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700")}>
                    {c.passed ? "pass" : "fail"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {report.anomaly_explanations?.length ? (
        <div>
          <h3 className="text-sm font-semibold mb-2">Anomaly explanations</h3>
          <div className="space-y-2">
            {report.anomaly_explanations.map((a, i) => (
              <div key={i} className="border rounded-lg p-3 bg-yellow-50 border-yellow-200">
                <p className="text-xs font-medium text-yellow-700 mb-1">{a.column}</p>
                <p className="text-sm text-gray-700">{a.explanation}</p>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
