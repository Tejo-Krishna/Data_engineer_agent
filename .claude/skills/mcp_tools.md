# Skill: MCP tools

Read this before writing or editing any MCP tool.

---

## Tool description format — mandatory

Every tool must have a description that follows this exact structure.
The LLM reads this description to decide when and how to call the tool.
A vague description causes the wrong tool to be called or the right tool
to be missed entirely.

```python
@mcp.tool(
    name="tool_name",
    description="""One sentence stating what this tool does.

Use when: [exact user intent or agent reasoning pattern that should
trigger this tool. Be specific — "when the user asks about X" not
"when needed".]

Do NOT use when: [what this tool is commonly confused with. Name the
alternative tool explicitly.]

Returns: [list every field the caller will receive, with a one-word
type hint. This is what the agent uses to plan its next step.]"""
)
async def tool_name(param: type) -> dict:
    ...
```

Example of a good description:

```python
description="""Fetch a statistical profile of a dataset sample.

Use when: you have a sample from sample_data and need null rates,
value distributions, type signatures, and duplicate counts before
generating any transformation code.

Do NOT use when: you need the schema — use detect_schema after
compute_profile. Do NOT use for quality checks on transformed
output — use run_quality_checks instead.

Returns: dict with per-column stats (dtype, null_count, null_rate,
unique_count, min, max, mean, top_5_values, sample_values) plus
total_rows (int) and duplicate_row_count (int)."""
```

---

## Tool organisation

Tools are grouped by domain in separate files. Each file owns one domain.
Never put tools from different domains in the same file.

```
source_tools.py    — connect_csv, connect_postgres, connect_api,
                     detect_new_rows
profiling_tools.py — sample_data, compute_profile, detect_schema,
                     compare_schemas
transform_tools.py — generate_transform_code, refine_transform_code,
                     execute_code, write_dataset
quality_tools.py   — run_quality_checks, detect_anomalies,
                     explain_anomalies, write_quality_report
catalogue_tools.py — write_catalogue_entry, generate_lineage_graph,
                     generate_dbt_model, read_catalogue
library_tools.py   — search_transform_library, save_to_library,
                     generate_dbt_schema_yml
domain_tools.py    — detect_domain, load_domain_rules
```

---

## Tool implementation rules

**Return types:** Every tool returns a dict. Never return a raw string,
list, or primitive — wrap it in a dict with a named key. The agent needs
named fields to reason about the result.

**Error handling:** Every tool must catch exceptions and return a dict
with `{"success": False, "error": "..."}` rather than raising. The agent
handles error cases in its ReAct loop — it should not crash because a
tool threw an exception.

**LLM calls inside tools:** Only these tools make LLM calls:
- `generate_transform_code` — generates Python transformation script
- `refine_transform_code` — revises code based on NLP instruction
- `detect_domain` — classifies dataset domain when heuristics are
  ambiguous
- `explain_anomalies` — generates plain English anomaly explanation
- `write_catalogue_entry` — generates column descriptions
- `generate_lineage_graph` — maps output columns to source columns
- `generate_dbt_model` — translates transformations to SQL

No other tool makes LLM calls. If you find yourself adding an LLM call
to a profiling or quality tool, stop — that logic belongs in the agent's
ReAct reasoning, not in the tool.

**DuckDB for data operations:** Use DuckDB for all data reading, sampling,
profiling, and Parquet I/O. Do not use pandas inside MCP tools. Pandas
is used only in generated transformation scripts that run in the sandbox.

```python
import duckdb

# Reading a CSV sample
conn = duckdb.connect()
result = conn.execute(
    "SELECT * FROM read_csv_auto(?) LIMIT ?",
    [file_path, sample_size]
).fetchall()
```

**Async throughout:** All tool functions must be `async def`. Use
`asyncpg` for Postgres, `redis.asyncio` for Redis, `httpx.AsyncClient`
for HTTP calls.

---

## The compare_schemas tool — schema drift detection

This is the most important new tool in the profiling domain. It runs on
every pipeline execution as the third step in the profiler agent.

```python
@mcp.tool(
    name="compare_schemas",
    description="""Compare the incoming dataset schema against the last
known schema for this source path stored in the catalogue.

Use when: you have just computed detect_schema and want to check if the
source has changed since the last successful pipeline run. Call this on
EVERY run — not just incremental runs.

Do NOT use when: there is no prior catalogue entry for this source
(the tool handles this gracefully and returns no_prior_run: true).

Returns: has_drift (bool), no_prior_run (bool), added_columns (list),
dropped_columns (list), renamed_columns (list of {from, to}),
type_changes (list of {column, old_type, new_type}),
drift_severity ("none"|"warning"|"critical")."""
)
async def compare_schemas(
    source_path: str,
    current_schema: dict
) -> dict:
    # Query catalogue for last entry with this source_path
    # Compare column names and types
    # Return structured drift report
    # drift_severity = "critical" if columns dropped or types changed
    # drift_severity = "warning" if columns added
    # drift_severity = "none" if no changes
    ...
```

The drift checkpoint in LangGraph reads `state["schema_drift"]["drift_severity"]`:
- `"none"` — continue silently
- `"warning"` — log to Langfuse, continue without pausing
- `"critical"` — pause at HITL checkpoint before proceeding

---

## The detect_new_rows tool — incremental processing

```python
@mcp.tool(
    name="detect_new_rows",
    description="""Filter a dataset to only rows newer than the last
successful pipeline run's watermark.

Use when: incremental_mode is True in PipelineState. Call this as the
FIRST step in the transformer agent's ReAct loop before any other tool.

Do NOT use when: incremental_mode is False. Do NOT call if no prior
run watermark exists — fall back to full processing instead.

Returns: filtered_path (str — path to temp filtered dataset),
new_row_count (int), watermark_column (str), watermark_value (str),
previous_watermark (str)."""
)
async def detect_new_rows(
    source_path: str,
    source_type: str,
    watermark_column: str,
    previous_watermark: str
) -> dict:
    # Read source, filter rows where watermark_column > previous_watermark
    # Write filtered rows to a temp Parquet file
    # Return path to temp file as filtered_path
    # The transformer uses filtered_path as its INPUT_PATH
    ...
```

The watermark value is read from the `pipeline_runs` Postgres table at
the start of each incremental run and written back after success. It is
never stored only in PipelineState — it must persist to the database so
it survives process restarts.

---

## The explain_anomalies tool

```python
@mcp.tool(
    name="explain_anomalies",
    description="""Generate a plain English explanation of anomalous rows
detected by detect_anomalies, contextualised by the dataset domain.

Use when: detect_anomalies has returned anomalous rows and you need to
write a human-readable explanation for the quality report.

Do NOT use when: anomaly_count is 0 — skip this call entirely.

Returns: explanations (list of {column, anomaly_count, explanation,
likely_cause, recommended_action}), overall_summary (str)."""
)
async def explain_anomalies(
    anomalous_rows: list,
    column_profile: dict,
    domain: str,
    domain_context: dict
) -> dict:
    # LLM call with anomaly data + domain context
    # Prompt: "Given this domain and these anomalous values, explain
    # in plain English what is likely wrong and what should be done"
    # Return structured explanations per column
    ...
```

The explanation must reference the domain. "In financial transaction data,
a revenue value of $4.2M is 6 standard deviations above the mean and
likely represents a bulk order or a data entry error. Recommend manual
review before including in revenue aggregations."

---

## Registering tools in server.py

```python
# mcp_server/server.py
from mcp.server import Server
from mcp.server.sse import SseServerTransport

from tools.source_tools import connect_csv, connect_postgres, connect_api, detect_new_rows
from tools.profiling_tools import sample_data, compute_profile, detect_schema, compare_schemas
from tools.transform_tools import generate_transform_code, refine_transform_code, execute_code, write_dataset
from tools.quality_tools import run_quality_checks, detect_anomalies, explain_anomalies, write_quality_report
from tools.catalogue_tools import write_catalogue_entry, generate_lineage_graph, generate_dbt_model, read_catalogue
from tools.library_tools import search_transform_library, save_to_library, generate_dbt_schema_yml
from tools.domain_tools import detect_domain, load_domain_rules

# Server must respond to tool/list with all 23 tools on connect
# Transport: SSE on port 8000
```
