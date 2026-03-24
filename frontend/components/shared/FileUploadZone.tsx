"use client";
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { cn } from "@/lib/utils";

interface Props {
  onFileSelected: (path: string, filename: string) => void;
}

export default function FileUploadZone({ onFileSelected }: Props) {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [uploaded, setUploaded] = useState<string | null>(null);

  const onDrop = useCallback(
    async (files: File[]) => {
      if (!files[0]) return;
      setUploading(true);
      setError(null);
      setProgress(0);

      const form = new FormData();
      form.append("file", files[0]);

      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/upload");
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) setProgress(Math.round((e.loaded / e.total) * 100));
      };
      xhr.onload = () => {
        setUploading(false);
        if (xhr.status >= 200 && xhr.status < 300) {
          const data = JSON.parse(xhr.responseText);
          setUploaded(data.filename);
          onFileSelected(data.saved_path, data.filename);
        } else {
          setError("Upload failed. Please try again.");
        }
      };
      xhr.onerror = () => {
        setUploading(false);
        setError("Network error during upload.");
      };
      xhr.send(form);
    },
    [onFileSelected]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"], "application/octet-stream": [".parquet"] },
    maxFiles: 1,
  });

  return (
    <div
      {...getRootProps()}
      className={cn(
        "border-2 border-dashed rounded-lg p-10 text-center cursor-pointer transition-colors",
        isDragActive ? "border-blue-400 bg-blue-50" : "border-gray-200 hover:border-gray-400",
        uploaded && "border-green-400 bg-green-50"
      )}
    >
      <input {...getInputProps()} />
      {uploading ? (
        <div className="space-y-2">
          <p className="text-sm text-gray-600">Uploading… {progress}%</p>
          <div className="w-full bg-gray-200 rounded-full h-1.5">
            <div className="bg-blue-500 h-1.5 rounded-full transition-all" style={{ width: `${progress}%` }} />
          </div>
        </div>
      ) : uploaded ? (
        <p className="text-sm text-green-600 font-medium">✓ {uploaded} uploaded</p>
      ) : (
        <div className="space-y-1">
          <p className="text-sm font-medium text-gray-700">
            {isDragActive ? "Drop it here" : "Drop a CSV or Parquet file, or click to browse"}
          </p>
          <p className="text-xs text-gray-400">.csv and .parquet supported</p>
        </div>
      )}
      {error && <p className="mt-2 text-xs text-red-500">{error}</p>}
    </div>
  );
}
