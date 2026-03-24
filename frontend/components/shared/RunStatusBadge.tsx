import { cn } from "@/lib/utils";
import type { RunStatus } from "@/lib/types";

const styles: Record<RunStatus, string> = {
  running:  "bg-blue-100 text-blue-700",
  success:  "bg-green-100 text-green-700",
  failed:   "bg-red-100 text-red-700",
  retrying: "bg-yellow-100 text-yellow-700",
};

export default function RunStatusBadge({ status }: { status: RunStatus }) {
  return (
    <span className={cn("px-2 py-0.5 rounded-full text-xs font-medium", styles[status])}>
      {status}
    </span>
  );
}
