import type { RunSummary, RunDetail, QualityReport } from "./types";

const BASE = "";  // rewrites proxy /api/* → localhost:8000/api/*

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

// --- Runs ---
export const startRun = (body: {
  source_path: string;
  source_type: string;
  user_goal: string;
  incremental_mode: boolean;
  source_table?: string;
}) =>
  req<{ run_id: string }>(`${BASE}/api/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

export const listRuns = () => req<RunSummary[]>(`${BASE}/api/runs`);

export const getRun = (runId: string) =>
  req<RunDetail>(`${BASE}/api/runs/${runId}`);

// --- Upload ---
export const uploadFile = async (
  file: File,
  onProgress?: (pct: number) => void
): Promise<{ saved_path: string; filename: string }> => {
  const form = new FormData();
  form.append("file", file);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}/api/upload`);
    if (onProgress) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
      };
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        reject(new Error(`Upload failed: ${xhr.responseText}`));
      }
    };
    xhr.onerror = () => reject(new Error("Upload network error"));
    xhr.send(form);
  });
};

// --- Results ---
export const getQualityReport = (runId: string) =>
  req<QualityReport>(`${BASE}/api/runs/${runId}/quality`);

export const getLineage = (runId: string) =>
  req<{ mermaid_diagram: string }>(`${BASE}/api/runs/${runId}/lineage`);

export const getDbtFile = (runId: string, filename: string) =>
  req<{ filename: string; content: string }>(
    `${BASE}/api/runs/${runId}/dbt/${filename}`
  );

export const downloadUrl = (runId: string, fileType: string) =>
  `${BASE}/api/runs/${runId}/download/${fileType}`;

// --- HITL ---
export const getHitlCode = (runId: string) =>
  req<{ state: string; code?: string; diff_summary?: string; revised_code?: string }>(
    `/hitl/${runId}/code/review`
  );

export const approveCode = (runId: string) =>
  req(`/hitl/${runId}/code/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ approved: true }),
  });

export const rejectCode = (runId: string) =>
  req(`/hitl/${runId}/code/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ approved: false }),
  });

export const refineCode = (runId: string, instruction: string) =>
  req(`/hitl/${runId}/code/instruct`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ instruction }),
  });

export const confirmCode = (runId: string, confirmed: boolean) =>
  req(`/hitl/${runId}/code/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirmed }),
  });

export const approveDrift = (runId: string, approved: boolean) =>
  req(`/hitl/${runId}/drift/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ approved }),
  });
