# Data Engineering Multi-Agent System — Master Project Prompt

## Read this first, every session

You are building a data engineering automation system. Before writing any code,
read the relevant skill file for the component you are working on. Skill files
are in `.claude/skills/`. They contain the exact conventions, patterns, and
constraints this project uses. Deviating from them creates inconsistency that
is expensive to fix later.

Skill files to read before working on each area:

| You are working on | Read this skill file first |
|--------------------|---------------------------|
| MCP server or any tool | `.claude/skills/mcp_tools.md` |
| Any agent | `.claude/skills/agents.md` |
| Domain detection or YAML rules | `.claude/skills/domain.md` |
| LangGraph graph, state, routing | `.claude/skills/orchestration.md` |
| Langfuse, RunLogger, observability | `.claude/skills/observability.md` |

---

## What this project is

A multi-agent data engineering pipeline where a user describes a data
transformation goal in plain English and four AI agents collaborate to produce
five validated, documented outputs. The human's role is to describe intent and
approve plans. The agents handle all mechanical work.

**The user provides:**
- A data source: CSV/Parquet file path, Postgres/SQLite connection string,
  or REST API URL
- A processing mode: `full` (reprocess everything) or `incremental`
  (process only new rows since last run)
- A natural language goal: "clean dates, dedup, convert GBP to USD"

**The system produces per successful run:**
1. Cleaned and transformed Parquet file
2. Rerunnable Python pipeline script (schedulable as a cron job)
3. Quality report — JSON + Markdown with anomaly explanations
4. Catalogue entry with column-level lineage stored in pgvector
5. dbt SQL model + schema.yml + schema_tests.yml

---

## Architecture overview

### Five agents in sequence

```
Profiler → [drift checkpoint] → Domain detection → Transformer → Quality → Catalogue
                                                        ↑             |
                                                        └── retry ────┘ (max 3)
```

**Profiler agent** — fixed 5-call sequence, no LLM
- connect to source → sample 2000 rows → compare_schemas (drift detection)
  → compute_profile → detect_schema
- Writes: source_metadata, sample, schema_drift, profile, schema to state

**Domain detection agent** — 2 LLM-assisted calls
- detect_domain (keyword heuristics first, LLM if ambiguous)
  → load_domain_rules (reads YAML)
- Writes: domain, domain_confidence, domain_context to state
- If domain_confidence < 0.60 → pause at HITL domain checkpoint

**Transformer agent** — ReAct loop, max 6 tool calls
- If incremental: detect_new_rows (watermark filter) first
- search_transform_library → generate_transform_code (LLM)
  → HITL checkpoint (approve / NLP instruction / reject)
  → [refine_transform_code if NLP instruction given]
  → execute_code (sandbox) → write_dataset
- Writes: generated_code, transformations_applied, output_path,
  rows_input, rows_output, pipeline_script to state

**Quality agent** — ReAct loop, adaptive tool selection
- run_quality_checks (targeted at high-risk columns from profile)
  → detect_anomalies (scoped to failing columns only)
  → explain_anomalies (LLM → plain English explanation)
  → write_quality_report
  → save_to_library (only if all checks pass AND code is reusable)
- Writes: quality_checks, anomaly_summary, anomaly_explanations,
  quality_passed, quality_report_path, status to state

**Catalogue agent** — fixed 5-call sequence
- write_catalogue_entry → generate_lineage_graph → generate_dbt_model
  → generate_dbt_schema_yml → generate_dbt_tests
- Writes: catalogue_id, lineage_graph, dbt_model_path,
  dbt_schema_path, dbt_tests_path to state

### MCP server — 23 tools across 7 domains

| Domain | Tools |
|--------|-------|
| source (4) | connect_csv, connect_postgres, connect_api, detect_new_rows |
| profiling (4) | sample_data, compute_profile, detect_schema, compare_schemas |
| transform (4) | generate_transform_code, refine_transform_code, execute_code, write_dataset |
| quality (4) | run_quality_checks, detect_anomalies, explain_anomalies, write_quality_report |
| catalogue (4) | write_catalogue_entry, generate_lineage_graph, generate_dbt_model, read_catalogue |
| library (3) | search_transform_library, save_to_library, generate_dbt_schema_yml |
| domain (2) | detect_domain, load_domain_rules |

**One additional tool in next phase:** generate_dbt_tests (currently in catalogue
agent's 5th call — implement after the above 23 are working)

### Infrastructure (all local Docker)

- PostgreSQL 16 + pgvector — catalogue, transform library, pipeline runs,
  quality results, lineage
- Redis — HITL checkpoint state, watermark high-water marks
- Langfuse (self-hosted) — observability at localhost:3000
- MCP server — FastAPI + MCP SDK, port 8000
- DuckDB — in-process, for profiling and Parquet I/O

Only outbound network traffic: Anthropic API calls. No data leaves the machine.

---

## Repository structure

```
data-engineer-agent/
├── .claude/
│   ├── CLAUDE.md                    ← this file
│   └── skills/
│       ├── mcp_tools.md
│       ├── agents.md
│       ├── domain.md
│       ├── orchestration.md
│       └── observability.md
├── mcp_server/
│   ├── server.py
│   └── tools/
│       ├── source_tools.py
│       ├── profiling_tools.py
│       ├── transform_tools.py
│       ├── quality_tools.py
│       ├── catalogue_tools.py
│       ├── library_tools.py
│       └── domain_tools.py
├── agents/
│   ├── profiler_agent.py
│   ├── domain_agent.py
│   ├── transformer_agent.py
│   ├── quality_agent.py
│   └── catalogue_agent.py
├── orchestrator/
│   ├── graph.py
│   ├── state.py
│   └── router.py
├── domain_rules/
│   ├── medical.yaml
│   ├── automotive.yaml
│   ├── employment.yaml
│   ├── financial.yaml
│   └── retail.yaml
├── sandbox/
│   └── executor.py
├── hitl/
│   └── checkpoint.py
├── observability/
│   └── tracing.py
├── memory/
│   ├── transform_library.py
│   └── catalogue_store.py
├── scripts/
│   ├── migrate.py
│   └── seed_library.py
├── tests/
├── sample_data/
│   └── sales_raw.csv
├── outputs/
├── logs/
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## PipelineState — complete schema

```python
class PipelineState(TypedDict):
    # Input
    run_id:                   str
    user_goal:                str
    source_path:              str
    source_type:              Literal["csv","parquet","postgres","api"]
    incremental_mode:         bool

    # Profiler output
    source_metadata:          Optional[dict]
    sample:                   Optional[list]
    profile:                  Optional[dict]
    schema:                   Optional[dict]
    schema_drift:             Optional[dict]   # NEW — from compare_schemas
    drift_checkpoint_approved: Optional[bool]  # NEW

    # Domain detection output
    domain:                   Optional[str]    # NEW
    domain_confidence:        Optional[float]  # NEW
    domain_context:           Optional[dict]   # NEW — loaded YAML rules

    # Transformer output
    library_snippets:         Optional[list]
    watermark_value:          Optional[str]    # NEW — incremental high-water mark
    generated_code:           Optional[str]
    transformations_applied:  Optional[list[str]]
    output_columns:           Optional[list[str]]
    output_path:              Optional[str]
    rows_input:               Optional[int]
    rows_output:              Optional[int]
    pipeline_script:          Optional[str]

    # Quality output
    quality_checks:           Optional[list]
    anomaly_summary:          Optional[dict]
    anomaly_explanations:     Optional[list]   # NEW — from explain_anomalies
    quality_passed:           Optional[bool]
    quality_report_path:      Optional[str]
    retry_count:              int
    failure_reason:           Optional[str]

    # Catalogue output
    catalogue_id:             Optional[str]
    lineage_graph:            Optional[dict]
    mermaid_diagram:          Optional[str]
    dbt_model_path:           Optional[str]
    dbt_schema_path:          Optional[str]
    dbt_tests_path:           Optional[str]    # NEW

    # HITL
    hitl_approved:            Optional[bool]
    hitl_edits:               Optional[str]

    # Final
    status: Literal["running","success","failed","retrying"]
```

---

## Build order — follow this exactly

Do not skip ahead. Each step depends on the previous ones being complete and
tested before proceeding.

```
Phase 1 — Infrastructure
  1.  docker-compose.yml (Postgres, Redis, Langfuse, MCP server)
  2.  scripts/migrate.py (all DB tables)
  3.  sample_data/sales_raw.csv (500 rows, intentional quality issues)
  4.  db.py (DuckDB + Postgres + Redis helpers)

Phase 2 — MCP server foundation
  5.  mcp_server/tools/source_tools.py (connect_csv, connect_postgres,
      connect_api, detect_new_rows)
  6.  mcp_server/tools/profiling_tools.py (sample_data, compute_profile,
      detect_schema, compare_schemas)
  7.  sandbox/executor.py (sandboxed subprocess runner)
  8.  mcp_server/tools/transform_tools.py (all 4 tools)
  9.  mcp_server/tools/quality_tools.py (all 4 tools including explain_anomalies)
  10. mcp_server/tools/library_tools.py (all 3 tools)
  11. mcp_server/tools/catalogue_tools.py (all 4 tools)
  12. mcp_server/tools/domain_tools.py (detect_domain, load_domain_rules)
  13. domain_rules/ YAML files (medical, automotive, employment,
      financial, retail)
  14. scripts/seed_library.py (5+ common transform patterns)
  15. mcp_server/server.py (register all 23 tools)

Phase 3 — Observability
  16. observability/tracing.py (Langfuse wrappers)

Phase 4 — HITL
  17. hitl/checkpoint.py + FastAPI endpoints

Phase 5 — Agents
  18. agents/profiler_agent.py
  19. agents/domain_agent.py
  20. agents/transformer_agent.py
  21. agents/quality_agent.py
  22. agents/catalogue_agent.py

Phase 6 — Orchestration
  23. orchestrator/state.py
  24. orchestrator/graph.py + router.py
  25. main.py

Phase 7 — Tests + validation
  26. tests/ (unit tests per tool and agent)
  27. End-to-end smoke test with sales_raw.csv
```

---

## Critical rules — never violate these

1. Every MCP tool description must follow the "Use when / Do NOT use when /
   Returns" format. Read `.claude/skills/mcp_tools.md` for the exact pattern.

2. Every agent must call RunLogger (or Langfuse span when added) at agent_start,
   after every tool call, and at agent_end. No silent agent execution.

3. `execute_code` must never be called before the HITL checkpoint has approved.
   This is enforced in the tool description itself.

4. `save_to_library` must never be called when quality_passed is False.
   The quality agent's ReAct loop must check this before calling.

5. Domain rules take precedence over user instructions. If the domain context
   forbids dropping null rows, the transformer must not generate code that
   drops null rows even if the user asked for it.

6. On retry runs (retry_count > 0), the HITL checkpoint does not fire.
   The transformer reads failure_reason from state and regenerates directly.

7. The sandbox executor must never be given environment variables beyond
   INPUT_PATH and OUTPUT_PATH. No database credentials, no API keys,
   no home directory access.

8. Generated code must always print "ROWS_IN: N" before processing and
   "ROWS_OUT: N" after. The sandbox executor parses these to populate
   rows_input and rows_output in state.

9. Schema drift detection runs on every pipeline execution. It is not optional
   and cannot be skipped even when incremental_mode is False.

10. Watermark values must be persisted to the pipeline_runs table in Postgres
    after every successful incremental run. Read from there at the start of
    the next incremental run, not from in-memory state.
