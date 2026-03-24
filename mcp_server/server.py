"""
MCP server — FastAPI + SSE transport.

Registers all 24 tools across 7 domains and exposes them at /sse.
A health endpoint is available at GET /health.
The HITL router is mounted when hitl/checkpoint.py is available (Phase 4).
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
import json

# ---------------------------------------------------------------------------
# Import all tool functions
# ---------------------------------------------------------------------------

from mcp_server.tools.source_tools import (
    connect_csv,
    connect_postgres,
    connect_api,
    detect_new_rows,
)
from mcp_server.tools.profiling_tools import (
    sample_data,
    compute_profile,
    detect_schema,
    compare_schemas,
)
from mcp_server.tools.domain_tools import (
    detect_domain,
    load_domain_rules,
)
from mcp_server.tools.transform_tools import (
    generate_transform_code,
    refine_transform_code,
    execute_code,
    write_dataset,
)
from mcp_server.tools.quality_tools import (
    run_quality_checks,
    detect_anomalies,
    explain_anomalies,
    write_quality_report,
)
from mcp_server.tools.library_tools import (
    search_transform_library,
    save_to_library,
    generate_dbt_schema_yml,
)
from mcp_server.tools.catalogue_tools import (
    write_catalogue_entry,
    generate_lineage_graph,
    generate_dbt_model,
    read_catalogue,
    generate_dbt_tests,
)

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    # Source tools
    Tool(
        name="connect_csv",
        description=(
            "Open a CSV or Parquet file and return its metadata. "
            "Use when: source_type is csv or parquet and you need shape/column info before sampling. "
            "Do NOT use when: connecting to a database or API. "
            "Returns: file_path, detected_delimiter, row_count, file_size_mb, column_names, first_5_rows."
        ),
        inputSchema={
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    ),
    Tool(
        name="connect_postgres",
        description=(
            "Connect to a Postgres table and return its schema and metadata. "
            "Use when: source_type is postgres. "
            "Do NOT use when: reading a file — use connect_csv instead. "
            "Returns: table_name, row_count, column_names, column_types, first_5_rows."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "connection_string": {"type": "string"},
                "table_name": {"type": "string"},
            },
            "required": ["connection_string", "table_name"],
        },
    ),
    Tool(
        name="connect_api",
        description=(
            "Fetch data from a REST API endpoint and return its metadata. "
            "Use when: source_type is api. "
            "Do NOT use when: reading a file or database. "
            "Returns: url, detected_format, row_count, field_names, first_5_rows, pagination_info."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "headers": {"type": "object"},
                "params": {"type": "object"},
            },
            "required": ["url"],
        },
    ),
    Tool(
        name="detect_new_rows",
        description=(
            "Filter a dataset to only rows newer than the previous watermark. "
            "Use when: incremental_mode is True and you need only new rows. "
            "Do NOT use when: running in full mode. "
            "Returns: filtered_path, new_row_count, watermark_column, watermark_value, previous_watermark."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "source_path": {"type": "string"},
                "source_type": {"type": "string"},
                "watermark_column": {"type": "string"},
                "previous_watermark": {"type": ["string", "null"]},
            },
            "required": ["source_path", "source_type", "watermark_column"],
        },
    ),
    # Profiling tools
    Tool(
        name="sample_data",
        description=(
            "Draw a representative sample from the source dataset. "
            "Use when: you need rows to pass to compute_profile or detect_schema. "
            "Do NOT use when: you need full-dataset statistics — compute_profile handles that. "
            "Returns: sample (list[dict]), actual_sample_size."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "source_path": {"type": "string"},
                "source_type": {"type": "string"},
                "sample_size": {"type": "integer", "default": 2000},
            },
            "required": ["source_path", "source_type"],
        },
    ),
    Tool(
        name="compute_profile",
        description=(
            "Compute per-column statistics from a sample. "
            "Use when: you have a sample from sample_data and need statistics for schema detection. "
            "Do NOT use when: you need schema inference — call detect_schema after this. "
            "Returns: profile dict with total_rows, duplicate_row_count, and per-column stats."
        ),
        inputSchema={
            "type": "object",
            "properties": {"sample": {"type": "array"}},
            "required": ["sample"],
        },
    ),
    Tool(
        name="detect_schema",
        description=(
            "Infer semantic types and cast requirements for each column. "
            "Use when: you have compute_profile output and need type inference for the transformer. "
            "Do NOT use when: you only need raw DuckDB types — those are in the profile already. "
            "Returns: per-column dict with inferred_type, needs_cast, suggested_cast, detected_formats."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "sample": {"type": "array"},
                "profile": {"type": "object"},
            },
            "required": ["sample", "profile"],
        },
    ),
    Tool(
        name="compare_schemas",
        description=(
            "Compare the current dataset schema against the most recent catalogue entry. "
            "Use when: you need to detect drift (added/dropped columns, type changes) before running. "
            "Do NOT use when: this is skippable — it runs on every pipeline execution without exception. "
            "Returns: has_drift, no_prior_run, added_columns, dropped_columns, type_changes, drift_severity."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "source_path": {"type": "string"},
                "current_schema": {"type": "object"},
            },
            "required": ["source_path", "current_schema"],
        },
    ),
    # Domain tools
    Tool(
        name="detect_domain",
        description=(
            "Identify which business domain the dataset belongs to. "
            "Use when: you need to load domain rules before generating transformation code. "
            "Do NOT use when: domain is already known — pass it directly to load_domain_rules. "
            "Returns: domain, confidence (0-1), method (keyword_heuristic | llm_classification)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "column_names": {"type": "array", "items": {"type": "string"}},
                "sample_values": {"type": "object"},
                "user_goal": {"type": "string"},
            },
            "required": ["column_names", "sample_values", "user_goal"],
        },
    ),
    Tool(
        name="load_domain_rules",
        description=(
            "Load the YAML rule file for the detected domain. "
            "Use when: detect_domain has returned and you need rules for generate_transform_code. "
            "Do NOT use when: domain is unknown and you intend to skip rules — this handles that gracefully. "
            "Returns: domain, sensitive_columns, required_transforms, forbidden_transforms, validation_rules, default_watermark_column."
        ),
        inputSchema={
            "type": "object",
            "properties": {"domain": {"type": "string"}},
            "required": ["domain"],
        },
    ),
    # Transform tools
    Tool(
        name="generate_transform_code",
        description=(
            "Generate a Python transformation script for the dataset and goal. "
            "Use when: profile, schema, and domain_context are ready and you need executable code. "
            "Do NOT use when: only minor NLP adjustments are needed — use refine_transform_code instead. "
            "Returns: code, transformations_applied, output_columns."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "user_goal": {"type": "string"},
                "profile": {"type": "object"},
                "schema": {"type": "object"},
                "domain_context": {"type": "object"},
                "library_snippets": {"type": ["array", "null"]},
                "failure_reason": {"type": ["string", "null"]},
            },
            "required": ["user_goal", "profile", "schema", "domain_context"],
        },
    ),
    Tool(
        name="refine_transform_code",
        description=(
            "Apply a plain-English edit instruction to an existing transformation script. "
            "Use when: HITL returned an NLP instruction (Path B) and the code needs a targeted revision. "
            "Do NOT use when: the code has a runtime error — use generate_transform_code with failure_reason. "
            "Returns: revised_code, changes_summary."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "original_code": {"type": "string"},
                "nlp_instruction": {"type": "string"},
                "profile": {"type": "object"},
                "domain_context": {"type": "object"},
            },
            "required": ["original_code", "nlp_instruction", "profile", "domain_context"],
        },
    ),
    Tool(
        name="execute_code",
        description=(
            "Run a transformation script in a sandboxed subprocess. "
            "Use when: the HITL checkpoint has approved the code. "
            "Do NOT call before the HITL checkpoint has approved — this is a hard rule violation. "
            "Returns: success, stdout, stderr, rows_input, rows_output, execution_time_ms."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "input_path": {"type": "string"},
                "output_path": {"type": "string"},
                "timeout_seconds": {"type": "integer", "default": 30},
            },
            "required": ["code", "input_path", "output_path"],
        },
    ),
    Tool(
        name="write_dataset",
        description=(
            "Move the sandbox output file to the permanent per-run output directory. "
            "Use when: execute_code succeeded and you need to persist the Parquet to outputs/{run_id}/. "
            "Do NOT use when: execute_code failed — always check success first. "
            "Returns: output_path, row_count, file_size_mb, columns, schema."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "data_path": {"type": "string"},
                "output_name": {"type": "string"},
                "run_id": {"type": "string"},
            },
            "required": ["data_path", "output_name", "run_id"],
        },
    ),
    # Quality tools
    Tool(
        name="run_quality_checks",
        description=(
            "Run a targeted suite of quality checks on the transformed dataset. "
            "Use when: execute_code succeeded and you need to validate the output. "
            "Do NOT use when: the dataset has not been transformed yet. "
            "Returns: checks list, overall_passed, critical_failures."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "output_path": {"type": "string"},
                "original_profile": {"type": "object"},
                "schema": {"type": "object"},
                "transformations_applied": {"type": "array"},
                "domain_context": {"type": ["object", "null"]},
            },
            "required": ["output_path", "original_profile", "schema", "transformations_applied"],
        },
    ),
    Tool(
        name="detect_anomalies",
        description=(
            "Detect statistical outliers and pattern violations in the output dataset. "
            "Use when: quality checks completed and you need specific anomalous rows for the report. "
            "Do NOT use when: all checks passed cleanly and critical_failures is empty. "
            "Returns: anomaly_count, anomalous_rows, anomaly_rate."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "output_path": {"type": "string"},
                "schema": {"type": "object"},
                "target_columns": {"type": ["array", "null"]},
            },
            "required": ["output_path", "schema"],
        },
    ),
    Tool(
        name="explain_anomalies",
        description=(
            "Generate plain-English explanations for detected anomalies using LLM. "
            "Use when: detect_anomalies returned anomaly_count > 0. "
            "Do NOT use when: anomaly_count is 0 — skip directly to write_quality_report. "
            "Returns: explanations list, overall_summary."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "anomalous_rows": {"type": "array"},
                "column_profile": {"type": "object"},
                "domain": {"type": "string"},
                "domain_context": {"type": "object"},
            },
            "required": ["anomalous_rows", "column_profile", "domain", "domain_context"],
        },
    ),
    Tool(
        name="write_quality_report",
        description=(
            "Write a quality report as both JSON and Markdown to the per-run output directory. "
            "Use when: quality checks and anomaly explanations (if any) are complete. "
            "Do NOT use when: explain_anomalies has not been called yet and anomaly_count > 0. "
            "Returns: json_path, markdown_path, overall_status."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "pipeline_run_id": {"type": "string"},
                "checks": {"type": "array"},
                "anomaly_summary": {"type": "object"},
                "anomaly_explanations": {"type": "array"},
                "output_dir": {"type": "string"},
            },
            "required": ["pipeline_run_id", "checks", "anomaly_summary", "anomaly_explanations", "output_dir"],
        },
    ),
    # Library tools
    Tool(
        name="search_transform_library",
        description=(
            "Search the transform library for reusable code snippets by semantic similarity. "
            "Use when: about to generate transformation code and want to check for existing patterns. "
            "Do NOT use when: you already have code and are saving it — use save_to_library instead. "
            "Returns: snippets list with name, description, code, tags, use_count, similarity_score."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "tags": {"type": ["array", "null"]},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="save_to_library",
        description=(
            "Persist a reusable transformation pattern to the library with its embedding. "
            "Use when: quality_passed is True AND the transformation is generic and reusable. "
            "Do NOT call when quality_passed is False — this would corrupt the library. "
            "Returns: library_id, saved (bool)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "code": {"type": "string"},
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "tags": {"type": "array"},
            },
            "required": ["name", "description", "code", "input_schema", "output_schema", "tags"],
        },
    ),
    Tool(
        name="generate_dbt_schema_yml",
        description=(
            "Generate a dbt schema.yml file for the transformed dataset columns. "
            "Use when: the catalogue agent needs column-level YAML alongside the SQL model. "
            "Do NOT use when: you need dbt tests — use generate_dbt_tests for that. "
            "Returns: yml_content, file_path."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "model_name": {"type": "string"},
                "output_schema": {"type": "object"},
                "column_descriptions": {"type": "object"},
                "run_id": {"type": "string"},
            },
            "required": ["model_name", "output_schema", "column_descriptions", "run_id"],
        },
    ),
    # Catalogue tools
    Tool(
        name="write_catalogue_entry",
        description=(
            "Register the transformed dataset in the catalogue with embeddings. "
            "Use when: the pipeline succeeded and you need the output discoverable for future runs. "
            "Do NOT use when: the pipeline failed — only catalogue successful outputs. "
            "Returns: catalogue_id, column_descriptions."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "pipeline_run_id": {"type": "string"},
                "dataset_name": {"type": "string"},
                "output_path": {"type": "string"},
                "schema": {"type": "object"},
                "row_count": {"type": "integer"},
                "source_type": {"type": "string"},
                "source_path": {"type": "string"},
                "transformations_applied": {"type": "array"},
            },
            "required": [
                "pipeline_run_id", "dataset_name", "output_path",
                "schema", "row_count", "source_type", "source_path", "transformations_applied",
            ],
        },
    ),
    Tool(
        name="generate_lineage_graph",
        description=(
            "Map source columns to output columns and render a Mermaid lineage diagram. "
            "Use when: write_catalogue_entry completed and you need to document column derivation. "
            "Do NOT use when: skippable — always document lineage even for passthrough columns. "
            "Returns: lineage_graph (dict with edges), mermaid_diagram."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "pipeline_run_id": {"type": "string"},
                "source_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "transformations_applied": {"type": "array"},
            },
            "required": ["pipeline_run_id", "source_schema", "output_schema", "transformations_applied"],
        },
    ),
    Tool(
        name="generate_dbt_model",
        description=(
            "Generate a dbt SQL SELECT model replicating the pipeline transformations. "
            "Use when: lineage is generated and you need a schedulable dbt SQL model. "
            "Do NOT use when: skippable — generate even if the SQL is a best-effort approximation. "
            "Returns: sql_content, file_path."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "pipeline_run_id": {"type": "string"},
                "model_name": {"type": "string"},
                "transformations_applied": {"type": "array"},
                "source_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "run_id": {"type": "string"},
            },
            "required": [
                "pipeline_run_id", "model_name", "transformations_applied",
                "source_schema", "output_schema", "run_id",
            ],
        },
    ),
    Tool(
        name="read_catalogue",
        description=(
            "Search the catalogue for datasets semantically similar to a query. "
            "Use when: you need to find previously processed datasets related to a topic. "
            "Do NOT use when: you need the current run's entry — read catalogue_id from state. "
            "Returns: entries list with catalogue_id, dataset_name, schema, similarity_score."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="generate_dbt_tests",
        description=(
            "Generate a dbt schema_tests.yml based on confirmed quality check results. "
            "Use when: generate_dbt_schema_yml completed and quality_passed is True. "
            "Do NOT use when: quality_passed is False — only test datasets that passed checks. "
            "Returns: tests_yml_content, file_path."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "pipeline_run_id": {"type": "string"},
                "model_name": {"type": "string"},
                "quality_checks": {"type": "array"},
                "output_schema": {"type": "object"},
                "run_id": {"type": "string"},
            },
            "required": ["pipeline_run_id", "model_name", "quality_checks", "output_schema", "run_id"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Tool dispatch map
# ---------------------------------------------------------------------------

TOOL_HANDLERS = {
    "connect_csv": connect_csv,
    "connect_postgres": connect_postgres,
    "connect_api": connect_api,
    "detect_new_rows": detect_new_rows,
    "sample_data": sample_data,
    "compute_profile": compute_profile,
    "detect_schema": detect_schema,
    "compare_schemas": compare_schemas,
    "detect_domain": detect_domain,
    "load_domain_rules": load_domain_rules,
    "generate_transform_code": generate_transform_code,
    "refine_transform_code": refine_transform_code,
    "execute_code": execute_code,
    "write_dataset": write_dataset,
    "run_quality_checks": run_quality_checks,
    "detect_anomalies": detect_anomalies,
    "explain_anomalies": explain_anomalies,
    "write_quality_report": write_quality_report,
    "search_transform_library": search_transform_library,
    "save_to_library": save_to_library,
    "generate_dbt_schema_yml": generate_dbt_schema_yml,
    "write_catalogue_entry": write_catalogue_entry,
    "generate_lineage_graph": generate_lineage_graph,
    "generate_dbt_model": generate_dbt_model,
    "read_catalogue": read_catalogue,
    "generate_dbt_tests": generate_dbt_tests,
}


# ---------------------------------------------------------------------------
# MCP server setup
# ---------------------------------------------------------------------------

mcp_server = Server("data-engineer-agent")


@mcp_server.list_tools()
async def list_tools():
    return TOOL_DEFINITIONS


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict):
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name}")
    result = await handler(**arguments)
    return [TextContent(type="text", text=json.dumps(result, default=str))]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Data Engineer Agent MCP Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "tools": len(TOOL_DEFINITIONS)}


# Mount MCP SSE transport
sse_transport = SseServerTransport("/messages/")

# Mount the POST handler for client→server messages as a proper ASGI app
app.mount("/messages", sse_transport.handle_post_message)


@app.get("/sse")
async def sse_endpoint(request: Request):
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await mcp_server.run(
            streams[0],
            streams[1],
            mcp_server.create_initialization_options(),
        )


# HITL router
try:
    from hitl.checkpoint import hitl_router
    app.include_router(hitl_router, prefix="/hitl")
except ImportError:
    pass

# Frontend API routers
try:
    from api.pipeline_router import router as pipeline_router
    from api.upload_router import router as upload_router
    app.include_router(pipeline_router, prefix="/api")
    app.include_router(upload_router, prefix="/api")
except ImportError as e:
    print(f"Warning: could not mount API routers: {e}")
