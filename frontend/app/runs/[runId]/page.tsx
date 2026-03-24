"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { getRun } from "@/lib/api";
import type { RunDetail } from "@/lib/types";
import { useRunEvents } from "@/hooks/useRunEvents";
import AgentTimeline from "@/components/control-room/AgentTimeline";
import ToolCallLog from "@/components/control-room/ToolCallLog";
import HitlCodePanel from "@/components/control-room/HitlCodePanel";
import RunStatusBadge from "@/components/shared/RunStatusBadge";
import { formatDuration, shortId } from "@/lib/utils";

export default function ControlRoomPage() {
  const { runId } = useParams<{ runId: string }>();
  const [run, setRun] = useState<RunDetail | null>(null);
  const { events, agentStates, hitlRequired } = useRunEvents(runId);

  useEffect(() => {
    const fetchRun = () => getRun(runId).then(setRun).catch(() => {});
    fetchRun();
    const interval = setInterval(fetchRun, 3000);
    return () => clearInterval(interval);
  }, [runId]);

  const isDone = run?.status === "success" || run?.status === "failed";

  // Show HITL panel if SSE fired OR if polling sees a pending state in Redis
  const hitlCodeState = run?.hitl_code?.state;
  const showHitl =
    hitlRequired === "code" ||
    hitlCodeState === "pending" ||
    hitlCodeState === "awaiting_confirm" ||
    hitlCodeState === "approved_with_instruction";

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Top bar */}
      <div className="bg-white border-b px-6 py-4">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono text-gray-400">RUN</span>
                <span className="text-sm font-mono font-semibold">{shortId(runId)}</span>
                {run && <RunStatusBadge status={run.status} />}
              </div>
              {run?.user_goal && (
                <p className="text-gray-600 text-sm mt-0.5 max-w-lg truncate">{run.user_goal}</p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-3">
            {run && (
              <span className="text-xs text-gray-400">{formatDuration(run.started_at, run.completed_at)}</span>
            )}
            {isDone && run?.status === "success" && (
              <Link
                href={`/runs/${runId}/results`}
                className="px-4 py-2 bg-black text-white text-sm rounded-lg font-medium hover:bg-gray-800 transition-colors"
              >
                View Results →
              </Link>
            )}
          </div>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-6 py-6 space-y-6">

        {/* HITL panel — shown prominently at the top when waiting */}
        {showHitl && (
          <div className="rounded-xl border-2 border-yellow-400 bg-yellow-50 overflow-hidden shadow-sm">
            <div className="bg-yellow-400 px-5 py-3 flex items-center gap-2">
              <span className="text-lg">⏸</span>
              <div>
                <p className="font-bold text-yellow-900 text-sm">Pipeline paused — your approval is needed</p>
                <p className="text-yellow-800 text-xs">Review the generated code below before it executes</p>
              </div>
            </div>
            <div className="p-4">
              <HitlCodePanel runId={runId} />
            </div>
          </div>
        )}

        {/* Agent timeline */}
        <div className="bg-white rounded-xl border p-5 shadow-sm">
          <h2 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">Pipeline Progress</h2>
          <AgentTimeline agentStates={agentStates} />
        </div>

        {/* Stats row */}
        {run && (run.rows_input != null || run.domain || run.failure_reason) && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {run.rows_input != null && (
              <>
                <div className="bg-white rounded-xl border p-4 shadow-sm">
                  <p className="text-xs text-gray-400 mb-1">Rows in</p>
                  <p className="text-2xl font-bold">{run.rows_input.toLocaleString()}</p>
                </div>
                <div className="bg-white rounded-xl border p-4 shadow-sm">
                  <p className="text-xs text-gray-400 mb-1">Rows out</p>
                  <p className="text-2xl font-bold">{run.rows_output?.toLocaleString() ?? "—"}</p>
                </div>
              </>
            )}
            {run.domain && (
              <div className="bg-white rounded-xl border p-4 shadow-sm">
                <p className="text-xs text-gray-400 mb-1">Domain</p>
                <p className="text-lg font-bold capitalize">{run.domain}</p>
              </div>
            )}
            {run.retry_count != null && run.retry_count > 0 && (
              <div className="bg-white rounded-xl border p-4 shadow-sm">
                <p className="text-xs text-gray-400 mb-1">Retries</p>
                <p className="text-2xl font-bold text-yellow-600">{run.retry_count}</p>
              </div>
            )}
            {run.failure_reason && (
              <div className="col-span-2 sm:col-span-4 bg-red-50 rounded-xl border border-red-200 p-4">
                <p className="text-xs font-bold text-red-400 mb-1">FAILURE REASON</p>
                <p className="text-sm text-red-700">{run.failure_reason}</p>
              </div>
            )}
          </div>
        )}

        {/* Tool call log */}
        <div className="bg-white rounded-xl border p-5 shadow-sm">
          <h2 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">Tool Calls</h2>
          <ToolCallLog events={events} />
        </div>

      </div>
    </div>
  );
}
