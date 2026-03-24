export const AGENTS = [
  { key: "profiler",     label: "Profiler",    description: "Profile & schema detection" },
  { key: "domain",       label: "Domain",      description: "Detect business domain" },
  { key: "transformer",  label: "Transformer", description: "Generate & execute code" },
  { key: "quality",      label: "Quality",     description: "Validate output data" },
  { key: "catalogue",    label: "Catalogue",   description: "dbt models & lineage" },
] as const;

export type AgentKey = typeof AGENTS[number]["key"];

export const STATUS_COLORS: Record<string, string> = {
  running:   "text-blue-500",
  success:   "text-green-500",
  failed:    "text-red-500",
  retrying:  "text-yellow-500",
};

export const STATUS_BG: Record<string, string> = {
  running:   "bg-blue-50 border-blue-200",
  success:   "bg-green-50 border-green-200",
  failed:    "bg-red-50 border-red-200",
  retrying:  "bg-yellow-50 border-yellow-200",
};
