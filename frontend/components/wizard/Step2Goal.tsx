"use client";
import { Textarea } from "@/components/ui/textarea";

const EXAMPLES = [
  "Clean date formats, remove duplicates, convert GBP to USD",
  "Normalise phone numbers to E.164 format and title-case names",
  "Remove PII, standardise country codes, fill missing postcodes",
  "Deduplicate on order_id, cast amounts to float, filter cancelled rows",
];

interface Props {
  wizard: { userGoal: string };
  update: (patch: { userGoal: string }) => void;
}

export default function Step2Goal({ wizard, update }: Props) {
  return (
    <div className="space-y-5">
      <h2 className="text-lg font-semibold">What should the pipeline do?</h2>
      <Textarea
        rows={4}
        placeholder="Describe your goal in plain English…"
        value={wizard.userGoal}
        onChange={(e) => update({ userGoal: e.target.value })}
        className="resize-none"
      />
      <div className="space-y-2">
        <p className="text-xs text-gray-400 uppercase tracking-wide">Examples</p>
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            onClick={() => update({ userGoal: ex })}
            className="block w-full text-left text-sm text-gray-600 hover:text-black border rounded-lg px-3 py-2 hover:border-gray-400 transition-colors"
          >
            {ex}
          </button>
        ))}
      </div>
    </div>
  );
}
