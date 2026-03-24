"use client";
import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { approveCode, rejectCode, refineCode, confirmCode, getHitlCode } from "@/lib/api";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

interface Props { runId: string; }

export default function HitlCodePanel({ runId }: Props) {
  const [hitl, setHitl] = useState<{ state: string; code?: string; diff_summary?: string; revised_code?: string } | null>(null);
  const [instruction, setInstruction] = useState("");
  const [loading, setLoading] = useState(false);
  const [showInstructionInput, setShowInstructionInput] = useState(false);

  const refresh = async () => {
    try {
      const data = await getHitlCode(runId);
      setHitl(data);
    } catch { /* HITL not active yet */ }
  };

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 2000);
    return () => clearInterval(interval);
  }, [runId]);

  if (!hitl || hitl.state === "confirmed" || hitl.state === "rejected") {
    return null;
  }

  const act = async (action: () => Promise<unknown>) => {
    setLoading(true);
    try { await action(); await refresh(); }
    finally { setLoading(false); }
  };

  return (
    <div className="border rounded-lg overflow-hidden">
      <div className="bg-yellow-50 border-b px-4 py-3 flex items-center justify-between">
        <div>
          <p className="font-medium text-sm">Human Review Required</p>
          <p className="text-xs text-gray-500">Review the generated transformation code before it runs</p>
        </div>
        <span className="text-xs bg-yellow-100 text-yellow-700 px-2 py-0.5 rounded-full">{hitl.state}</span>
      </div>

      {/* Pending — show code for review */}
      {hitl.state === "pending" && hitl.code && (
        <div className="space-y-3 p-4">
          <MonacoEditor
            height="300px"
            language="python"
            value={hitl.code}
            options={{ readOnly: true, minimap: { enabled: false }, fontSize: 12 }}
          />
          {showInstructionInput ? (
            <div className="space-y-2">
              <textarea
                className="w-full border rounded p-2 text-sm resize-none"
                rows={3}
                placeholder="Describe what to change, e.g. 'also remove rows where amount is negative'"
                value={instruction}
                onChange={(e) => setInstruction(e.target.value)}
              />
              <div className="flex gap-2">
                <button onClick={() => act(() => refineCode(runId, instruction))} disabled={!instruction || loading} className="px-4 py-1.5 text-sm bg-black text-white rounded disabled:opacity-40">
                  Send instruction
                </button>
                <button onClick={() => setShowInstructionInput(false)} className="px-4 py-1.5 text-sm border rounded">Cancel</button>
              </div>
            </div>
          ) : (
            <div className="flex gap-2">
              <button onClick={() => act(() => approveCode(runId))} disabled={loading} className="px-4 py-1.5 text-sm bg-green-600 text-white rounded">Approve ✓</button>
              <button onClick={() => setShowInstructionInput(true)} className="px-4 py-1.5 text-sm border rounded">Edit & Refine ✎</button>
              <button onClick={() => act(() => rejectCode(runId))} disabled={loading} className="px-4 py-1.5 text-sm bg-red-50 text-red-600 border border-red-200 rounded">Reject ✗</button>
            </div>
          )}
        </div>
      )}

      {/* Awaiting confirm — show diff */}
      {hitl.state === "awaiting_confirm" && (
        <div className="space-y-3 p-4">
          <p className="text-sm font-medium">Revised code — confirm or request another change</p>
          {hitl.diff_summary && (
            <div className="bg-gray-50 rounded p-3 text-sm text-gray-700 border">{hitl.diff_summary}</div>
          )}
          <MonacoEditor
            height="260px"
            language="python"
            value={hitl.revised_code || ""}
            options={{ readOnly: true, minimap: { enabled: false }, fontSize: 12 }}
          />
          <div className="flex gap-2">
            <button onClick={() => act(() => confirmCode(runId, true))} disabled={loading} className="px-4 py-1.5 text-sm bg-green-600 text-white rounded">Confirm ✓</button>
            <button onClick={() => act(() => confirmCode(runId, false))} disabled={loading} className="px-4 py-1.5 text-sm border rounded">Request another revision</button>
          </div>
        </div>
      )}

      {(hitl.state === "approved_with_instruction") && (
        <div className="p-4 text-sm text-gray-500">Applying your instruction, generating revised code…</div>
      )}
    </div>
  );
}
