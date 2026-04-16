"""
Catalogue agent — fixed 5-call sequence, no ReAct.

Raises ValueError immediately if state["status"] != "success".
Only catalogue successful pipeline runs.

Call order:
  1. write_catalogue_entry    — embeds dataset, stores in catalogue table
  2. generate_lineage_graph   — maps source→output columns, writes Mermaid diagram
  3. generate_dbt_model       — LLM writes SQL SELECT replicating transforms
  4. generate_dbt_schema_yml  — writes schema.yml (from library_tools)
  5. generate_dbt_tests       — rule-based schema_tests.yml

Writes to state: catalogue_id, lineage_graph, mermaid_diagram,
dbt_model_path, dbt_schema_path, dbt_tests_path.
"""

import json
import os
from pathlib import Path

from langgraph.types import RunnableConfig

from orchestrator.state import PipelineState, require_quality_output


async def run_catalogue_agent(
    state: PipelineState,
    config: RunnableConfig,
) -> PipelineState:
    if state.get("status") != "success":
        raise ValueError(
            f"Catalogue agent must only run after a successful pipeline. "
            f"Current status: {state.get('status')!r}"
        )
    require_quality_output(state)

    mcp = config["configurable"]["mcp"]
    logger = config["configurable"]["logger"]
    span = logger.agent_start("catalogue")

    run_id = state["run_id"]
    schema: dict = state.get("schema") or {}
    transformations_applied: list = state.get("transformations_applied") or []

    if not schema:
        raise ValueError("Catalogue agent: state['schema'] is empty — cannot generate dbt artefacts")
    if not state.get("output_path"):
        raise ValueError("Catalogue agent: state['output_path'] is empty — transformer did not write a dataset")

    # model_name derived from run_id prefix (dbt model names must be valid identifiers)
    model_name = f"pipeline_{run_id[:8].replace('-', '_')}"

    # ------------------------------------------------------------------
    # Tool call 1 — write catalogue entry
    # ------------------------------------------------------------------
    entry = await mcp.call("write_catalogue_entry", {
        "pipeline_run_id": run_id,
        "dataset_name": f"output_{run_id[:8]}",
        "output_path": state.get("output_path", ""),
        "schema": schema,
        "row_count": state.get("rows_output", 0),
        "source_type": state["source_type"],
        "source_path": state["source_path"],
        "transformations_applied": transformations_applied,
    })
    catalogue_id: str = entry.get("catalogue_id", "")
    column_descriptions: dict = entry.get("column_descriptions", {})

    logger.tool_call(
        span, "catalogue", "write_catalogue_entry",
        {"dataset": f"output_{run_id[:8]}"},
        {"catalogue_id": catalogue_id,
         "columns_described": len(column_descriptions)},
    )

    # ------------------------------------------------------------------
    # Tool call 2 — generate lineage graph
    # ------------------------------------------------------------------
    lineage = await mcp.call("generate_lineage_graph", {
        "pipeline_run_id": run_id,
        "source_schema": schema,
        "output_schema": schema,   # best approximation without separate output schema
        "transformations_applied": transformations_applied,
    })
    lineage_graph: dict = lineage.get("lineage_graph", {})
    mermaid_diagram: str = lineage.get("mermaid_diagram", "")

    # Persist lineage to file so the API can serve it without a DB column
    _save_lineage(run_id, mermaid_diagram, lineage_graph)

    logger.tool_call(
        span, "catalogue", "generate_lineage_graph",
        {},
        {"edges": len(lineage_graph.get("edges", []))},
    )

    # ------------------------------------------------------------------
    # Tool call 3 — generate dbt SQL model
    # ------------------------------------------------------------------
    dbt_model = await mcp.call("generate_dbt_model", {
        "pipeline_run_id": run_id,
        "model_name": model_name,
        "transformations_applied": transformations_applied,
        "source_schema": schema,
        "output_schema": schema,
        "run_id": run_id,
    })
    dbt_model_path: str = dbt_model.get("file_path", "")

    logger.tool_call(
        span, "catalogue", "generate_dbt_model",
        {"model_name": model_name},
        {"file_path": dbt_model_path},
    )

    # ------------------------------------------------------------------
    # Tool call 4 — generate dbt schema.yml
    # ------------------------------------------------------------------
    dbt_schema = await mcp.call("generate_dbt_schema_yml", {
        "model_name": model_name,
        "output_schema": {
            col: v.get("inferred_type", "VARCHAR")
            for col, v in schema.items()
        },
        "column_descriptions": column_descriptions,
        "run_id": run_id,
    })
    dbt_schema_path: str = dbt_schema.get("file_path", "")

    logger.tool_call(
        span, "catalogue", "generate_dbt_schema_yml",
        {"model_name": model_name},
        {"file_path": dbt_schema_path},
    )

    # ------------------------------------------------------------------
    # Tool call 5 — generate dbt schema_tests.yml
    # ------------------------------------------------------------------
    dbt_tests = await mcp.call("generate_dbt_tests", {
        "pipeline_run_id": run_id,
        "model_name": model_name,
        "quality_checks": state.get("quality_checks") or [],
        "output_schema": {
            col: v.get("inferred_type", "VARCHAR")
            for col, v in schema.items()
        },
        "run_id": run_id,
    })
    dbt_tests_path: str = dbt_tests.get("file_path", "")

    logger.tool_call(
        span, "catalogue", "generate_dbt_tests",
        {"model_name": model_name},
        {"file_path": dbt_tests_path},
    )

    logger.agent_end(span)

    return {
        **state,
        "catalogue_id": catalogue_id,
        "lineage_graph": lineage_graph,
        "mermaid_diagram": mermaid_diagram,
        "dbt_model_path": dbt_model_path,
        "dbt_schema_path": dbt_schema_path,
        "dbt_tests_path": dbt_tests_path,
    }


def _save_lineage(run_id: str, mermaid_diagram: str, lineage_graph: dict) -> None:
    output_dir = Path(os.getenv("OUTPUT_DIR", "outputs")) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "lineage.json", "w", encoding="utf-8") as f:
        json.dump({"mermaid_diagram": mermaid_diagram, "lineage_graph": lineage_graph}, f, indent=2)
