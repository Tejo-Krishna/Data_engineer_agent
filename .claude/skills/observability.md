# Skill: Observability

Read this before working on observability/tracing.py or adding any
logging or tracing to agents.

---

## Two-phase observability

The project uses a two-phase approach:

**Phase 1 (prototype):** `RunLogger` — writes a JSON file per run to
`./logs/{run_id}.json`. No external dependency. Works offline.

**Phase 2 (current target):** Langfuse — self-hosted in Docker at
`localhost:3000`. Full trace timeline, token costs, quality scores,
prompt inspection for debugging.

The `observability/tracing.py` module provides both. Which one runs
depends on whether `LANGFUSE_PUBLIC_KEY` is set in `.env`. If the key
is present, use Langfuse. If not, fall back to RunLogger.

```python
# observability/tracing.py

import os, json, time
from pathlib import Path

LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))

if LANGFUSE_ENABLED:
    from langfuse import Langfuse
    _lf = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    )
```

---

## Trace lifecycle

One root trace per pipeline run. Created in `main.py`. The trace ID is
the pipeline `run_id` (UUID). Every agent span is a child of this trace.

```python
# observability/tracing.py

def start_trace(run_id: str, user_goal: str, source_path: str):
    if LANGFUSE_ENABLED:
        return _lf.trace(
            id=run_id,
            name="pipeline_run",
            input={"goal": user_goal, "source": source_path},
            metadata={"version": "1.0"}
        )
    else:
        return RunLogger(run_id, user_goal, source_path)

def finish_trace(trace, status: str, final_state: dict, error: str = None):
    outputs = {
        "parquet":           final_state.get("output_path"),
        "quality_report":    final_state.get("quality_report_path"),
        "dbt_model":         final_state.get("dbt_model_path"),
        "dbt_schema":        final_state.get("dbt_schema_path"),
        "dbt_tests":         final_state.get("dbt_tests_path"),
    }
    if LANGFUSE_ENABLED:
        trace.update(output=outputs, metadata={"status": status})
        # Attach quality scores to the root trace
        if final_state.get("quality_passed") is not None:
            _lf.score(trace_id=trace.id, name="quality_passed",
                      value=1.0 if final_state["quality_passed"] else 0.0)
            _lf.score(trace_id=trace.id, name="retry_count",
                      value=float(final_state.get("retry_count", 0)))
            _lf.score(trace_id=trace.id, name="rows_processed",
                      value=float(final_state.get("rows_output", 0)))
        if final_state.get("domain_confidence"):
            _lf.score(trace_id=trace.id, name="domain_confidence",
                      value=float(final_state["domain_confidence"]))
    else:
        trace.finish(status, outputs, error=error)
```

---

## Agent span pattern

Every agent opens a child span, logs tool calls as events, and closes
the span. The pattern is identical whether using Langfuse or RunLogger
— the API is unified in `observability/tracing.py`.

```python
# In every agent — unified pattern works with both Langfuse and RunLogger

async def run_profiler_agent(state, mcp, logger):
    span = logger.agent_start("profiler")

    result = await mcp.call("sample_data", {...})

    logger.tool_call(
        span=span,
        agent="profiler",
        tool="sample_data",
        input_summary={"source": state["source_path"]},
        output_summary={"rows": len(result.get("sample", []))}
    )

    logger.agent_end(span)
    return {**state, ...}
```

The `logger` object passed to each agent is the trace/RunLogger object
created in `main.py`. Agents do not create their own loggers.

---

## What gets logged for each new feature

**Schema drift detection:**
```python
logger.tool_call(span, "profiler", "compare_schemas",
    input_summary={"source": state["source_path"]},
    output_summary={
        "has_drift": drift.get("has_drift"),
        "severity":  drift.get("drift_severity"),
        "dropped":   len(drift.get("dropped_columns", [])),
        "added":     len(drift.get("added_columns", [])),
        "type_changes": len(drift.get("type_changes", []))
    }
)
# Also log as a Langfuse event so drift history is queryable over time
if LANGFUSE_ENABLED and drift.get("has_drift"):
    _lf.event(
        trace_id=state["run_id"],
        name="schema_drift_detected",
        input={"source": state["source_path"]},
        output=drift,
        level="WARNING" if drift["drift_severity"] == "critical" else "DEFAULT"
    )
```

**Incremental processing:**
```python
logger.tool_call(span, "transformer", "detect_new_rows",
    input_summary={"watermark_col": watermark_col,
                   "previous_watermark": prev_watermark},
    output_summary={
        "new_row_count":       result["new_row_count"],
        "previous_watermark":  result["previous_watermark"],
        "current_watermark":   result["watermark_value"],
        "pct_of_total":        round(result["new_row_count"] /
                                     state["source_metadata"]["row_count"] * 100, 1)
    }
)
```

**Anomaly explanations:**
```python
logger.tool_call(span, "quality", "explain_anomalies",
    input_summary={"anomaly_count": anomalies["anomaly_count"],
                   "domain": state["domain"]},
    output_summary={"columns_explained": len(explanations["explanations"]),
                    "overall_summary": explanations["overall_summary"][:120]}
)
# Langfuse annotation for searchability
if LANGFUSE_ENABLED:
    _lf.score(trace_id=state["run_id"], name="anomaly_count",
              value=float(anomalies["anomaly_count"]))
```

**Domain detection:**
```python
logger.tool_call(span, "domain", "detect_domain",
    input_summary={"column_count": len(state["schema"])},
    output_summary={"domain": result["domain"],
                    "confidence": result["confidence"],
                    "method": result.get("method", "llm")}
)
if LANGFUSE_ENABLED:
    _lf.score(trace_id=state["run_id"], name="domain_confidence",
              value=float(result["confidence"]))
```

---

## RunLogger — file-based fallback

```python
# observability/tracing.py — RunLogger class
class RunLogger:
    def __init__(self, run_id, user_goal, source_path):
        self.run_id = run_id
        self.log = {
            "run_id": run_id, "user_goal": user_goal,
            "source_path": source_path,
            "started_at": time.time(), "completed_at": None,
            "status": "running", "agents": {}, "outputs": {}, "error": None
        }
        self._path = Path(os.getenv("LOG_DIR","./logs")) / f"{run_id}.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._flush()

    def agent_start(self, name):
        self.log["agents"][name] = {
            "started_at": time.time(), "completed_at": None, "tool_calls": []
        }
        self._flush()
        return name  # returns "span" for unified API

    def tool_call(self, span, agent, tool, input_summary, output_summary):
        self.log["agents"][agent]["tool_calls"].append({
            "tool": tool, "at": time.time(),
            "input": input_summary, "output": output_summary
        })
        self._flush()

    def agent_end(self, span):
        agent = span  # span is just the agent name in RunLogger
        self.log["agents"][agent]["completed_at"] = time.time()
        self._flush()

    def finish(self, status, outputs, error=None):
        self.log["status"] = status
        self.log["completed_at"] = time.time()
        self.log["outputs"] = outputs
        self.log["error"] = error
        self._flush()

    def _flush(self):
        self._path.write_text(json.dumps(self.log, indent=2, default=str))
```

---

## Langfuse scores summary

These scores are attached to every root trace after pipeline completion.
They are the primary metrics for monitoring pipeline health over time.

| Score name | Type | What it measures |
|------------|------|-----------------|
| quality_passed | 0.0 or 1.0 | Did all quality checks pass? Trend toward 1.0 |
| retry_count | 0–3 | How many retries were needed? Trend toward 0 |
| rows_processed | integer | Output row count |
| domain_confidence | 0.0–1.0 | How confident was domain detection? |
| anomaly_count | integer | How many anomalous rows were found? |

Query these in the Langfuse dashboard to see:
- Is the retry rate decreasing as the library fills up?
- Are certain source types consistently triggering retries?
- Is domain detection confidence low for a particular data type?
- Are anomaly counts trending up (upstream data quality degrading)?

---

## Adding Langfuse to Docker Compose

Add this service to docker-compose.yml:

```yaml
langfuse:
  image: langfuse/langfuse:latest
  ports: ["3000:3000"]
  environment:
    DATABASE_URL: postgresql://user:pass@postgres:5432/dataeng
    NEXTAUTH_SECRET: any-random-string-change-in-production
    SALT: any-other-string-change-in-production
  depends_on: [postgres]
```

Add to .env:
```
LANGFUSE_PUBLIC_KEY=pk-lf-...      # from localhost:3000 after first start
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

After starting Langfuse for the first time, navigate to localhost:3000,
create an account, create a project, and copy the API keys to .env.
