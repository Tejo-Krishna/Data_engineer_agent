export type RunStatus = "running" | "success" | "failed" | "retrying";

export interface RunSummary {
  id: string;
  source_path: string;
  source_type: string;
  status: RunStatus;
  started_at: string;
  completed_at: string | null;
  rows_input: number | null;
  rows_output: number | null;
  quality_passed: boolean | null;
}

export interface HitlCodeState {
  state: "pending" | "approved" | "approved_with_instruction" | "awaiting_confirm" | "confirmed" | "rejected";
  code?: string;
  diff_summary?: string;
  revised_code?: string;
  nlp_instruction?: string;
}

export interface RunDetail extends RunSummary {
  user_goal?: string;
  domain?: string;
  domain_confidence?: number;
  failure_reason?: string;
  output_path?: string;
  pipeline_script?: string;
  quality_report_path?: string;
  dbt_model_path?: string;
  mermaid_diagram?: string;
  retry_count?: number;
  hitl_code: HitlCodeState | null;
  hitl_drift: { state: string; schema_drift?: object } | null;
}

export interface QualityCheck {
  check_name: string;
  column: string;
  passed: boolean;
  expected?: string | number;
  actual?: string | number;
  severity?: string;
}

export interface QualityReport {
  run_id: string;
  overall_passed: boolean;
  checks: QualityCheck[];
  anomaly_summary?: { anomaly_count: number; anomaly_rate: number };
  anomaly_explanations?: { column: string; explanation: string }[];
}

export interface PipelineEvent {
  type: string;
  ts: number;
  run_id?: string;
  agent?: string;
  tool?: string;
  status?: string;
  error?: string;
}
