import { downloadUrl } from "@/lib/api";

const FILES = [
  { key: "parquet",         label: "Cleaned Parquet",   icon: "📦" },
  { key: "pipeline_script", label: "Pipeline Script",   icon: "🐍" },
  { key: "quality_json",    label: "Quality Report (JSON)", icon: "📊" },
  { key: "quality_md",      label: "Quality Report (MD)",   icon: "📝" },
  { key: "dbt_model",       label: "dbt SQL Model",     icon: "🏗" },
  { key: "dbt_schema",      label: "dbt schema.yml",    icon: "📋" },
  { key: "dbt_tests",       label: "dbt tests.yml",     icon: "✅" },
];

export default function DownloadsTab({ runId }: { runId: string }) {
  return (
    <div className="grid grid-cols-2 gap-3">
      {FILES.map((f) => (
        <a
          key={f.key}
          href={downloadUrl(runId, f.key)}
          download
          className="flex items-center gap-3 border rounded-lg px-4 py-3 hover:bg-gray-50 transition-colors"
        >
          <span className="text-xl">{f.icon}</span>
          <span className="text-sm font-medium">{f.label}</span>
        </a>
      ))}
    </div>
  );
}
