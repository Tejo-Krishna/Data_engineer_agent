"""
Profiler agent — fixed 5-call sequence, no LLM, no ReAct.

Call order (mandatory, never reorder):
  1. connect_{source_type}
  2. sample_data
  3. compute_profile
  4. detect_schema
  5. compare_schemas   ← drift detection uses full inferred schema, runs on EVERY execution

Writes to state: source_metadata, sample, schema_drift, profile, schema.
"""

from langgraph.types import RunnableConfig

from observability.tracing import log_schema_drift
from orchestrator.state import PipelineState


async def run_profiler_agent(
    state: PipelineState,
    config: RunnableConfig,
) -> PipelineState:
    mcp = config["configurable"]["mcp"]
    logger = config["configurable"]["logger"]
    span = logger.agent_start("profiler")

    run_id = state["run_id"]
    source_path = state["source_path"]
    source_type = state["source_type"]

    # ------------------------------------------------------------------
    # Tool call 1 — connect to source
    # ------------------------------------------------------------------
    connect_tool, connect_args = _build_connect_args(state)
    print(f"  [profiler] calling {connect_tool} ...", flush=True)
    metadata = await mcp.call(connect_tool, connect_args)
    print(f"  [profiler] {connect_tool} done", flush=True)
    logger.tool_call(
        span, "profiler", connect_tool,
        {"source": source_path},
        {"columns": len(metadata.get("column_names", [])),
         "rows": metadata.get("row_count", 0)},
    )

    # ------------------------------------------------------------------
    # Tool call 2 — sample 2000 rows
    # ------------------------------------------------------------------
    print("  [profiler] calling sample_data ...", flush=True)
    sample_result = await mcp.call("sample_data", {
        "source_path": source_path,
        "source_type": source_type,
        "sample_size": 2000,
    })
    print("  [profiler] sample_data done", flush=True)
    sample: list = sample_result.get("sample", [])
    logger.tool_call(
        span, "profiler", "sample_data",
        {"source": source_path},
        {"rows": len(sample)},
    )

    # ------------------------------------------------------------------
    # Tool call 3 — compute profile
    # ------------------------------------------------------------------
    print("  [profiler] calling compute_profile ...", flush=True)
    profile = await mcp.call("compute_profile", {"sample": sample})
    print("  [profiler] compute_profile done", flush=True)
    logger.tool_call(
        span, "profiler", "compute_profile",
        {},
        {"columns": len(profile)},
    )

    # ------------------------------------------------------------------
    # Tool call 4 — detect schema
    # ------------------------------------------------------------------
    print("  [profiler] calling detect_schema ...", flush=True)
    schema = await mcp.call("detect_schema", {
        "sample": sample,
        "profile": profile,
    })
    print("  [profiler] detect_schema done", flush=True)
    logger.tool_call(
        span, "profiler", "detect_schema",
        {},
        {"columns": len(schema), "needs_cast": sum(
            1 for v in schema.values() if v.get("needs_cast")
        )},
    )

    # ------------------------------------------------------------------
    # Tool call 5 — compare_schemas (drift detection, always runs)
    # Uses the full inferred schema so type comparison is apples-to-apples
    # with the schema stored in the catalogue from the prior run.
    # ------------------------------------------------------------------
    print("  [profiler] calling compare_schemas ...", flush=True)
    drift = await mcp.call("compare_schemas", {
        "source_path": source_path,
        "current_schema": schema,
    })
    print("  [profiler] compare_schemas done", flush=True)
    logger.tool_call(
        span, "profiler", "compare_schemas",
        {"source": source_path},
        {
            "has_drift": drift.get("has_drift"),
            "severity": drift.get("drift_severity"),
            "dropped": len(drift.get("dropped_columns", [])),
            "added": len(drift.get("added_columns", [])),
            "type_changes": len(drift.get("type_changes", [])),
        },
    )

    # Emit Langfuse event if drift detected
    if drift.get("has_drift"):
        log_schema_drift(run_id, source_path, drift)

    logger.agent_end(span)
    print("  [profiler] done", flush=True)

    return {
        **state,
        "source_metadata": metadata,
        "sample": sample[:10],  # keep only 10 rows in state; full sample used above
        "schema_drift": drift,
        "profile": profile,
        "schema": schema,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_connect_args(state: PipelineState) -> tuple[str, dict]:
    """Return (tool_name, args_dict) for the connect call."""
    source_type = state["source_type"]
    source_path = state["source_path"]

    if source_type in ("csv", "parquet"):
        return "connect_csv", {"file_path": source_path}

    if source_type == "postgres":
        # source_path may embed table as "postgresql://...::table_name"
        table = state.get("source_table") or "data"
        if "::" in source_path:
            conn_str, table = source_path.rsplit("::", 1)
        else:
            conn_str = source_path
        return "connect_postgres", {
            "connection_string": conn_str,
            "table_name": table,
        }

    if source_type == "api":
        return "connect_api", {
            "url": source_path,
            "headers": {},
            "params": {},
        }

    raise ValueError(f"Unknown source_type: {source_type!r}")
