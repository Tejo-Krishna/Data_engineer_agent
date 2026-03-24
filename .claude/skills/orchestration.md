# Skill: Orchestration

Read this before working on orchestrator/graph.py, orchestrator/state.py,
orchestrator/router.py, or main.py.

---

## LangGraph graph structure

The pipeline is a directed state graph. Nodes are agents or checkpoints.
Edges are either fixed (always go to the same next node) or conditional
(go to different nodes based on state).

```
START
  ↓
profiler                    → fixed node, 5 tool calls
  ↓
drift_checkpoint            → conditional pause: skip if no drift,
  ↓                           pause if drift_severity="critical"
domain_detection            → fixed node, 2 tool calls
  ↓
transformer                 → ReAct node, max 6 tool calls
  ↓
quality                     → ReAct node, max 6 tool calls
  ↓ (conditional)
  ├── success  → catalogue  → fixed node, 5 tool calls
  │               ↓
  │             END (success)
  ├── retrying → transformer (loop back, max 3 times)
  └── failed   → END (failed)
```

---

## Graph implementation

```python
# orchestrator/graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

def build_graph():
    g = StateGraph(PipelineState)

    # Register all nodes
    g.add_node("profiler",          run_profiler_agent)
    g.add_node("drift_checkpoint",  run_drift_checkpoint)
    g.add_node("domain_detection",  run_domain_agent)
    g.add_node("transformer",       run_transformer_agent)
    g.add_node("quality",           run_quality_agent)
    g.add_node("catalogue",         run_catalogue_agent)

    # Entry point
    g.set_entry_point("profiler")

    # Fixed edges
    g.add_edge("profiler",       "drift_checkpoint")
    g.add_edge("drift_checkpoint", "domain_detection")
    g.add_edge("domain_detection", "transformer")
    g.add_edge("transformer",    "quality")
    g.add_edge("catalogue",      END)

    # Conditional edge from quality
    g.add_conditional_edges(
        "quality",
        route_after_quality,
        {
            "catalogue": "catalogue",
            "retry":     "transformer",
            "failed":    END,
        }
    )

    return g.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["transformer"]  # HITL pause point
    )
```

---

## Router function

```python
# orchestrator/router.py
def route_after_quality(state: PipelineState) -> str:
    max_retries = int(os.getenv("MAX_QUALITY_RETRIES", "3"))

    if state["status"] == "success":
        return "catalogue"

    if state["status"] == "retrying":
        if state["retry_count"] >= max_retries:
            # Exceeded max retries — force to failed
            return "failed"
        return "retry"

    return "failed"
```

Note: the router modifies no state. It only reads `state["status"]`
and `state["retry_count"]`. The quality agent is responsible for setting
these correctly before returning.

---

## Drift checkpoint node

This is a lightweight conditional node, not a full agent. It does not
call any MCP tools. It reads `state["schema_drift"]` and decides whether
to pause.

```python
# orchestrator/graph.py
async def run_drift_checkpoint(state: PipelineState) -> PipelineState:
    drift = state.get("schema_drift", {})
    severity = drift.get("drift_severity", "none")

    if severity == "none":
        # No drift — pass through silently
        return {**state, "drift_checkpoint_approved": True}

    if severity == "warning":
        # Log warning, continue without pausing
        # (Langfuse logs this automatically via the span)
        return {**state, "drift_checkpoint_approved": True}

    if severity == "critical":
        # Pause — surface to HITL
        approval = await drift_hitl_checkpoint(
            run_id=state["run_id"],
            drift_report=drift
        )
        return {**state, "drift_checkpoint_approved": approval["approved"]}

    return {**state, "drift_checkpoint_approved": True}
```

The drift HITL is simpler than the code review HITL — it just asks
"schema changes detected, do you want to continue?" without code review.

---

## PipelineState — rules for adding fields

Never remove existing fields. Never change existing field types.
Only add new Optional fields when a new agent or feature genuinely
requires persistent state.

New fields added in this version:
```python
schema_drift:             Optional[dict]   # profiler → compare_schemas output
drift_checkpoint_approved: Optional[bool]  # set by drift_checkpoint node
domain:                   Optional[str]    # domain_agent output
domain_confidence:        Optional[float]  # domain_agent output
domain_context:           Optional[dict]   # loaded YAML rules
watermark_value:          Optional[str]    # transformer → detect_new_rows
anomaly_explanations:     Optional[list]   # quality → explain_anomalies
dbt_tests_path:           Optional[str]    # catalogue → generate_dbt_tests
```

---

## Incremental mode

Incremental mode is set by the user at invocation time:

```bash
python main.py --source orders.csv --type csv \
               --goal "clean and dedup" \
               --incremental
```

The `incremental_mode: bool` field in PipelineState is set in `main.py`
based on the `--incremental` flag. It is read by the transformer agent
to decide whether to call `detect_new_rows` first.

The watermark high-water mark is stored in and read from the
`pipeline_runs` Postgres table. It is never stored only in memory.

```python
# Read watermark at start of incremental run
async def get_previous_watermark(source_path: str, domain_context: dict) -> str | None:
    watermark_col = domain_context.get("default_watermark_column")
    if not watermark_col:
        return None
    row = await db.fetchrow(
        "SELECT watermark_value FROM pipeline_runs "
        "WHERE source_path = $1 AND status = 'success' "
        "ORDER BY completed_at DESC LIMIT 1",
        source_path
    )
    return row["watermark_value"] if row else None

# Write watermark after successful run
async def save_watermark(run_id: str, watermark_value: str):
    await db.execute(
        "UPDATE pipeline_runs SET watermark_value = $1 WHERE id = $2",
        watermark_value, run_id
    )
```

---

## Main entry point

```python
# main.py
import asyncio, uuid, argparse
from dotenv import load_dotenv
from orchestrator.graph import build_graph
from orchestrator.state import PipelineState
from observability.tracing import start_trace, finish_trace

load_dotenv()

async def run_pipeline(source_path, source_type, user_goal, incremental):
    run_id = str(uuid.uuid4())
    trace = start_trace(run_id, user_goal, source_path)

    initial_state: PipelineState = {
        "run_id":          run_id,
        "user_goal":       user_goal,
        "source_path":     source_path,
        "source_type":     source_type,
        "incremental_mode": incremental,
        "retry_count":     0,
        "status":          "running",
    }

    graph = build_graph()
    config = {"configurable": {"thread_id": run_id}}

    try:
        final = await graph.ainvoke(initial_state, config)
        finish_trace(trace, final["status"], final)
        print(f"\nPipeline complete — {final['status']}")
        print_outputs(final)
    except Exception as e:
        finish_trace(trace, "failed", {}, error=str(e))
        raise

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source",      required=True)
    p.add_argument("--type",        default="csv",
                   choices=["csv","parquet","postgres","api"])
    p.add_argument("--goal",        required=True)
    p.add_argument("--incremental", action="store_true", default=False)
    args = p.parse_args()
    asyncio.run(run_pipeline(args.source, args.type, args.goal, args.incremental))
```

---

## Docker Compose services

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    ports: ["5432:5432"]
    environment:
      POSTGRES_DB: dataeng
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  langfuse:
    image: langfuse/langfuse:latest
    ports: ["3000:3000"]
    environment:
      DATABASE_URL: postgresql://user:pass@postgres:5432/dataeng
      NEXTAUTH_SECRET: devsecret
      SALT: devsalt
    depends_on: [postgres]

  mcp_server:
    build: ./mcp_server
    ports: ["8000:8000"]
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
    depends_on: [postgres, redis]
```
