"use client";
import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import FileUploadZone from "@/components/shared/FileUploadZone";

type Mode = "file" | "url" | "database";

interface Props {
  wizard: { sourcePath: string; sourceType: string; sourceTable: string; incremental: boolean };
  update: (patch: Partial<{ sourcePath: string; sourceType: string; sourceTable: string; incremental: boolean }>) => void;
}

export default function Step1Source({ wizard, update }: Props) {
  const [mode, setMode] = useState<Mode>("file");

  const tabs: { key: Mode; label: string }[] = [
    { key: "file",     label: "Upload File" },
    { key: "url",      label: "URL / API" },
    { key: "database", label: "Database" },
  ];

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold">Where is your data?</h2>

      {/* Mode tabs */}
      <div className="flex gap-1 p-1 bg-gray-100 rounded-lg w-fit">
        {tabs.map((t) => (
          <button
            key={t.key}
            onClick={() => { setMode(t.key); update({ sourcePath: "", sourceType: t.key === "database" ? "postgres" : "csv" }); }}
            className={`px-4 py-1.5 rounded-md text-sm transition-colors ${mode === t.key ? "bg-white shadow font-medium" : "text-gray-500 hover:text-black"}`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* File upload */}
      {mode === "file" && (
        <div className="space-y-3">
          <FileUploadZone onFileSelected={(path) => { update({ sourcePath: path, sourceType: "csv" }); }} />
          <p className="text-xs text-gray-400">Or type a local path directly:</p>
          <Input
            placeholder="/Users/you/data/myfile.csv"
            value={wizard.sourcePath}
            onChange={(e) => update({ sourcePath: e.target.value })}
          />
        </div>
      )}

      {/* URL */}
      {mode === "url" && (
        <div className="space-y-3">
          <Label>URL</Label>
          <Input
            placeholder="https://example.com/data.csv"
            value={wizard.sourcePath}
            onChange={(e) => update({ sourcePath: e.target.value })}
          />
          <div className="flex gap-2 mt-1">
            {["csv", "parquet", "api"].map((t) => (
              <label key={t} className="flex items-center gap-1 text-sm cursor-pointer">
                <input
                  type="radio"
                  name="sourceType"
                  checked={wizard.sourceType === t}
                  onChange={() => update({ sourceType: t })}
                />
                {t}
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Database */}
      {mode === "database" && (
        <div className="space-y-3">
          <Label>Connection string</Label>
          <Input
            placeholder="postgresql://user:pass@localhost:5432/mydb"
            value={wizard.sourcePath}
            onChange={(e) => update({ sourcePath: e.target.value, sourceType: "postgres" })}
          />
          <Label>Table name</Label>
          <Input
            placeholder="orders"
            value={wizard.sourceTable}
            onChange={(e) => update({ sourceTable: e.target.value })}
          />
        </div>
      )}

      {/* Incremental toggle */}
      <label className="flex items-center gap-2 text-sm cursor-pointer mt-2">
        <input
          type="checkbox"
          checked={wizard.incremental}
          onChange={(e) => update({ incremental: e.target.checked })}
        />
        <span>Incremental mode — process only new rows since last run</span>
      </label>
    </div>
  );
}
