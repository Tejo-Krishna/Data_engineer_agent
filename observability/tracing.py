"""
Observability — dual-backend tracing.

If LANGFUSE_PUBLIC_KEY is set in .env, traces are sent to the self-hosted
Langfuse instance at LANGFUSE_HOST (default http://localhost:3000).
Otherwise, a RunLogger writes a JSON file per run to logs/{run_id}.json.

Public API (identical for both backends):
    logger = start_trace(run_id, user_goal, source_path)
    span   = logger.agent_start("profiler")
    logger.tool_call(span, "profiler", "sample_data", {...}, {...})
    logger.agent_end(span)
    finish_trace(logger, "success", final_state)
"""

import os
import json
import time
from pathlib import Path

LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))

if LANGFUSE_ENABLED:
    from langfuse import Langfuse

    _lf = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
    )


# ---------------------------------------------------------------------------
# RunLogger — file-based fallback (no external dependencies)
# ---------------------------------------------------------------------------


class RunLogger:
    def __init__(self, run_id: str, user_goal: str, source_path: str) -> None:
        self.run_id = run_id
        self.log: dict = {
            "run_id": run_id,
            "user_goal": user_goal,
            "source_path": source_path,
            "started_at": time.time(),
            "completed_at": None,
            "status": "running",
            "agents": {},
            "outputs": {},
            "error": None,
        }
        self._path = Path(os.getenv("LOG_DIR", "./logs")) / f"{run_id}.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._flush()

    # -- span lifecycle -------------------------------------------------------

    def agent_start(self, name: str) -> str:
        """Open a new agent span. Returns the span handle (agent name string)."""
        self.log["agents"][name] = {
            "started_at": time.time(),
            "completed_at": None,
            "tool_calls": [],
        }
        self._flush()
        return name  # span is just the agent name in RunLogger

    def tool_call(
        self,
        span: str,
        agent: str,
        tool: str,
        input_summary: dict,
        output_summary: dict,
    ) -> None:
        """Append a tool call record under the active agent span."""
        self.log["agents"].setdefault(
            agent,
            {"started_at": time.time(), "completed_at": None, "tool_calls": []},
        )
        self.log["agents"][agent]["tool_calls"].append(
            {
                "tool": tool,
                "at": time.time(),
                "input": input_summary,
                "output": output_summary,
            }
        )
        self._flush()
        try:
            from api.event_bus import publish
            publish(self.run_id, "tool_call", {
                "agent": agent, "tool": tool,
                "input": input_summary, "output": output_summary,
            })
        except Exception:
            pass

    def agent_end(self, span: str) -> None:
        """Close the agent span identified by span (= agent name)."""
        agent = span
        self.log["agents"].setdefault(
            agent,
            {"started_at": time.time(), "completed_at": None, "tool_calls": []},
        )
        self.log["agents"][agent]["completed_at"] = time.time()
        self._flush()

    def finish(self, status: str, outputs: dict, error: str | None = None) -> None:
        """Mark the run complete and flush final state."""
        self.log["status"] = status
        self.log["completed_at"] = time.time()
        self.log["outputs"] = outputs
        self.log["error"] = error
        self._flush()

    def _flush(self) -> None:
        self._path.write_text(json.dumps(self.log, indent=2, default=str))


# ---------------------------------------------------------------------------
# LangfuseLogger — thin wrapper giving RunLogger-compatible API
# ---------------------------------------------------------------------------


class LangfuseLogger:
    def __init__(self, trace) -> None:
        self._trace = trace
        self._run_id = getattr(trace, "id", None)

    def agent_start(self, name: str):
        """Open a Langfuse span for this agent. Returns the span object."""
        span = self._trace.span(name=name)
        return span

    def tool_call(
        self,
        span,
        agent: str,
        tool: str,
        input_summary: dict,
        output_summary: dict,
    ) -> None:
        """Log a tool call event inside the current agent span."""
        span.event(
            name=tool,
            input=input_summary,
            output=output_summary,
            metadata={"agent": agent},
        )
        try:
            from api.event_bus import publish
            if self._run_id:
                publish(self._run_id, "tool_call", {
                    "agent": agent, "tool": tool,
                    "input": input_summary, "output": output_summary,
                })
        except Exception:
            pass

    def agent_end(self, span) -> None:
        """Close the Langfuse span."""
        span.end()

    def finish(self, status: str, outputs: dict, error: str | None = None) -> None:
        """Delegate to finish_trace — kept for API symmetry."""
        pass  # finish_trace() handles the root trace update directly


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def start_trace(run_id: str, user_goal: str, source_path: str):
    """
    Create a new trace for a pipeline run.

    Returns a RunLogger (offline) or LangfuseLogger (when LANGFUSE_PUBLIC_KEY
    is set). Both expose the same agent_start / tool_call / agent_end / finish
    methods — agents never need to know which backend is active.
    """
    if LANGFUSE_ENABLED:
        trace = _lf.trace(
            id=run_id,
            name="pipeline_run",
            input={"goal": user_goal, "source": source_path},
            metadata={"version": "1.0"},
        )
        return LangfuseLogger(trace)
    else:
        return RunLogger(run_id, user_goal, source_path)


def finish_trace(
    logger,
    status: str,
    final_state: dict,
    error: str | None = None,
) -> None:
    """
    Finalise the trace with status, output paths, and quality scores.

    For Langfuse: updates root trace and attaches numeric scores for
    dashboard monitoring. For RunLogger: delegates to logger.finish().
    """
    outputs = {
        "parquet": final_state.get("output_path"),
        "pipeline_script": final_state.get("pipeline_script"),
        "quality_report": final_state.get("quality_report_path"),
        "dbt_model": final_state.get("dbt_model_path"),
        "dbt_schema": final_state.get("dbt_schema_path"),
        "dbt_tests": final_state.get("dbt_tests_path"),
    }

    if LANGFUSE_ENABLED:
        run_id = final_state.get("run_id", "")
        logger._trace.update(
            output=outputs,
            metadata={"status": status, "error": error},
        )

        # Numeric scores for Langfuse dashboard monitoring
        if final_state.get("quality_passed") is not None:
            _lf.score(
                trace_id=run_id,
                name="quality_passed",
                value=1.0 if final_state["quality_passed"] else 0.0,
            )
        if final_state.get("retry_count") is not None:
            _lf.score(
                trace_id=run_id,
                name="retry_count",
                value=float(final_state["retry_count"]),
            )
        if final_state.get("rows_output") is not None:
            _lf.score(
                trace_id=run_id,
                name="rows_processed",
                value=float(final_state["rows_output"]),
            )
        if final_state.get("domain_confidence") is not None:
            _lf.score(
                trace_id=run_id,
                name="domain_confidence",
                value=float(final_state["domain_confidence"]),
            )
        anomaly_count = (
            final_state.get("anomaly_summary", {}) or {}
        ).get("anomaly_count")
        if anomaly_count is not None:
            _lf.score(
                trace_id=run_id,
                name="anomaly_count",
                value=float(anomaly_count),
            )
    else:
        logger.finish(status, outputs, error=error)


# ---------------------------------------------------------------------------
# Langfuse-only event helpers (no-ops when RunLogger is active)
# ---------------------------------------------------------------------------


def log_schema_drift(run_id: str, source_path: str, drift: dict) -> None:
    """Emit a queryable schema_drift_detected event on Langfuse traces."""
    if not LANGFUSE_ENABLED:
        return
    _lf.event(
        trace_id=run_id,
        name="schema_drift_detected",
        input={"source": source_path},
        output=drift,
        level="WARNING" if drift.get("drift_severity") == "critical" else "DEFAULT",
    )


def log_domain_low_confidence(run_id: str, domain: str, confidence: float) -> None:
    """Emit a domain_low_confidence event when confidence < 0.60."""
    if not LANGFUSE_ENABLED:
        return
    _lf.event(
        trace_id=run_id,
        name="domain_low_confidence",
        input={"domain": domain, "confidence": confidence},
        level="WARNING",
    )


def log_hitl_nlp_instruction(run_id: str, instruction: str) -> None:
    """Emit a hitl_nlp_instruction event when the human provides a refinement."""
    if not LANGFUSE_ENABLED:
        return
    _lf.event(
        trace_id=run_id,
        name="hitl_nlp_instruction",
        input={"instruction": instruction[:500]},
    )
