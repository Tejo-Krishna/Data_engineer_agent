"use client";
import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { getDbtFile } from "@/lib/api";
import { shortId } from "@/lib/utils";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

export default function DbtTab({ runId }: { runId: string }) {
  const modelFile = `pipeline_${shortId(runId).replace(/-/g, "_")}.sql`;
  const files = [
    { key: modelFile,         label: "SQL Model",       lang: "sql" },
    { key: "schema.yml",      label: "schema.yml",      lang: "yaml" },
    { key: "schema_tests.yml",label: "schema_tests.yml",lang: "yaml" },
  ];
  const [active, setActive] = useState(0);
  const [contents, setContents] = useState<Record<string, string>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    files.forEach(({ key }) => {
      getDbtFile(runId, key)
        .then((d) => setContents((c) => ({ ...c, [key]: d.content })))
        .catch(() => setErrors((e) => ({ ...e, [key]: "File not generated yet" })));
    });
  }, [runId]);

  const current = files[active];

  return (
    <div className="space-y-3">
      <div className="flex gap-1">
        {files.map((f, i) => (
          <button
            key={f.key}
            onClick={() => setActive(i)}
            className={`px-3 py-1.5 text-sm rounded ${active === i ? "bg-black text-white" : "border hover:bg-gray-50"}`}
          >
            {f.label}
          </button>
        ))}
      </div>
      {errors[current.key] ? (
        <p className="text-sm text-gray-400">{errors[current.key]}</p>
      ) : (
        <MonacoEditor
          height="400px"
          language={current.lang}
          value={contents[current.key] ?? "Loading…"}
          options={{ readOnly: true, minimap: { enabled: false }, fontSize: 12 }}
        />
      )}
    </div>
  );
}
