"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { cn } from "@/lib/utils";
import Step1Source from "./Step1Source";
import Step2Goal from "./Step2Goal";
import { startRun } from "@/lib/api";

interface WizardState {
  sourcePath: string;
  sourceType: string;
  sourceTable: string;
  userGoal: string;
  incremental: boolean;
}

const STEPS = ["Data Source", "Your Goal", "Launch"];

export default function WizardShell() {
  const router = useRouter();
  const [step, setStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [wizard, setWizard] = useState<WizardState>({
    sourcePath: "",
    sourceType: "csv",
    sourceTable: "",
    userGoal: "",
    incremental: false,
  });

  const update = (patch: Partial<WizardState>) =>
    setWizard((w) => ({ ...w, ...patch }));

  const canProceed = () => {
    if (step === 0) return wizard.sourcePath.trim() !== "";
    if (step === 1) return wizard.userGoal.trim() !== "";
    return true;
  };

  const handleLaunch = async () => {
    setLoading(true);
    setError(null);
    try {
      const { run_id } = await startRun({
        source_path: wizard.sourcePath,
        source_type: wizard.sourceType,
        user_goal: wizard.userGoal,
        incremental_mode: wizard.incremental,
        source_table: wizard.sourceTable || undefined,
      });
      router.push(`/runs/${run_id}`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to start run");
      setLoading(false);
    }
  };

  return (
    <div className="max-w-xl mx-auto py-12 px-4">
      {/* Step indicator */}
      <div className="flex items-center gap-2 mb-10">
        {STEPS.map((label, i) => (
          <div key={i} className="flex items-center gap-2">
            <div
              className={cn(
                "w-7 h-7 rounded-full flex items-center justify-center text-xs font-medium",
                i < step ? "bg-black text-white" :
                i === step ? "border-2 border-black text-black" :
                "border border-gray-300 text-gray-400"
              )}
            >
              {i < step ? "✓" : i + 1}
            </div>
            <span className={cn("text-sm", i === step ? "font-medium" : "text-gray-400")}>
              {label}
            </span>
            {i < STEPS.length - 1 && <div className="flex-1 h-px w-8 bg-gray-200 mx-1" />}
          </div>
        ))}
      </div>

      {/* Step content */}
      {step === 0 && <Step1Source wizard={wizard} update={update} />}
      {step === 1 && <Step2Goal wizard={wizard} update={update} />}
      {step === 2 && (
        <div className="space-y-4">
          <h2 className="text-lg font-semibold">Ready to launch</h2>
          <div className="rounded-lg border p-4 space-y-2 text-sm">
            <div className="flex justify-between"><span className="text-gray-500">Source</span><span className="font-mono text-xs truncate max-w-[260px]">{wizard.sourcePath}</span></div>
            <div className="flex justify-between"><span className="text-gray-500">Type</span><span>{wizard.sourceType}</span></div>
            {wizard.sourceTable && <div className="flex justify-between"><span className="text-gray-500">Table</span><span>{wizard.sourceTable}</span></div>}
            <div className="flex justify-between"><span className="text-gray-500">Goal</span><span className="text-right max-w-[260px]">{wizard.userGoal}</span></div>
            <div className="flex justify-between"><span className="text-gray-500">Mode</span><span>{wizard.incremental ? "Incremental" : "Full"}</span></div>
          </div>
          {error && <p className="text-sm text-red-500">{error}</p>}
        </div>
      )}

      {/* Navigation */}
      <div className="flex justify-between mt-8">
        <button
          onClick={() => setStep((s) => s - 1)}
          disabled={step === 0}
          className="px-4 py-2 text-sm rounded border disabled:opacity-30"
        >
          Back
        </button>
        {step < 2 ? (
          <button
            onClick={() => setStep((s) => s + 1)}
            disabled={!canProceed()}
            className="px-5 py-2 text-sm rounded bg-black text-white disabled:opacity-30"
          >
            Next
          </button>
        ) : (
          <button
            onClick={handleLaunch}
            disabled={loading}
            className="px-5 py-2 text-sm rounded bg-black text-white disabled:opacity-50"
          >
            {loading ? "Launching…" : "Launch Pipeline"}
          </button>
        )}
      </div>
    </div>
  );
}
