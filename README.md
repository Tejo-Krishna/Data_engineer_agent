# Data Engineering Multi-Agent System

A multi-agent ETL pipeline where you describe a data transformation goal in plain English and five AI agents collaborate to produce cleaned data, a quality report, column lineage, and dbt models — all locally, with no data leaving your machine.

## What it does

You provide:
- A data source (CSV/Parquet file, Postgres table, or REST API)
- A processing mode (`full` or `incremental`)
- A natural language goal: `"clean dates, dedup, convert GBP to USD"`

You get back (per successful run):
1. Cleaned Parquet file at `outputs/{run_id}/transformed.parquet`
2. Rerunnable `pipeline.py` script (schedulable as a cron job)
3. Quality report — `quality_report.json` + `quality_report.md`
4. Catalogue entry with column-level lineage stored in pgvector
5. dbt SQL model + `schema.yml` + `schema_tests.yml`

## Architecture

```
Profiler → [drift checkpoint] → Domain → Transformer → Quality → Catalogue
                                              ↑              |
                                              └── retry ─────┘ (max 3)
```

### Five agents

| Agent | Type | What it does |
|-------|------|--------------|
| Profiler | Fixed 5-call sequence | Connects, samples, detects drift, profiles, infers schema |
| Domain | Fixed 2-call sequence | Classifies domain (retail/medical/financial/…), loads YAML rules |
| Transformer | ReAct loop (max 6 calls) | Generates Python transformation code, runs HITL checkpoint, executes in sandbox, verifies intent |
| Quality | ReAct loop (max 6 calls) | Runs quality checks, detects anomalies, explains them, writes report |
| Catalogue | Fixed 5-call sequence | Writes catalogue entry, generates lineage graph, produces dbt artefacts |

### MCP server — 25 tools

All agent-to-tool communication goes through a local MCP server (FastAPI + SSE, port 8000).

| Domain | Tools |
|--------|-------|
| source (4) | connect_csv, connect_postgres, connect_api, detect_new_rows |
| profiling (4) | sample_data, compute_profile, detect_schema, compare_schemas |
| transform (5) | generate_transform_code, refine_transform_code, execute_code, write_dataset, verify_transform_intent |
| quality (4) | run_quality_checks, detect_anomalies, explain_anomalies, write_quality_report |
| catalogue (5) | write_catalogue_entry, generate_lineage_graph, generate_dbt_model, read_catalogue, generate_dbt_tests |
| library (3) | search_transform_library, save_to_library, generate_dbt_schema_yml |
| domain (2) | detect_domain, load_domain_rules |

### Infrastructure (all local Docker)

- **PostgreSQL 16 + pgvector** — catalogue, transform library, pipeline runs, quality results, lineage
- **Redis** — HITL checkpoint state, incremental watermark high-water marks
- **Langfuse** (self-hosted at `localhost:3000`) — every agent start/end and tool call logged
- **DuckDB** — in-process profiling and Parquet I/O (no network calls)

Only outbound traffic: Anthropic API calls. Nothing else leaves your machine.

## Quick start

```bash
# 1. Clone and set up Python environment
git clone https://github.com/Tejo-Krishna/Data_engineer_agent
cd data_agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY at minimum

# 3. Start infrastructure
docker-compose up -d postgres redis langfuse

# 4. Run database migrations
python scripts/migrate.py

# 5. Seed the transform library with common patterns
python scripts/seed_library.py

# 6. Start the MCP server
uvicorn mcp_server.server:app --port 8000

# 7. Run the pipeline
python main.py \
  --source sample_data/sales_raw.csv \
  --goal "clean dates and remove duplicates" \
  --mode full
```

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-5` | Model for code generation tasks |
| `DOMAIN_CLASSIFY_MODEL` | No | `claude-haiku-4-5-20251001` | Model for domain classification |
| `DOCKER_SANDBOX_IMAGE` | No | (unset) | Docker image for isolated sandbox execution. Falls back to subprocess when unset. |
| `DATABASE_URL` | Yes | — | Postgres connection string |
| `REDIS_URL` | No | `redis://localhost:6379` | Redis connection string |
| `OUTPUT_DIR` | No | `outputs/` | Base directory for run outputs |
| `DOMAIN_RULES_DIR` | No | `domain_rules/` | Path to domain YAML rule files |

## Running tests

```bash
# All tests (77 total, ~3 seconds, no API calls required)
pytest tests/ -v

# Just the architectural improvement tests
pytest tests/test_improvements.py -v

# Tool-level unit tests
pytest tests/test_tools.py -v

# Agent integration tests (mocked LLM)
pytest tests/test_agents.py -v
```

## Repository layout

```
data_agent/
├── .claude/             — CLAUDE.md master prompt + skill files for AI-assisted dev
├── agents/              — Five agent implementations
├── mcp_server/
│   ├── server.py        — FastAPI + MCP SSE server, TOOL_HANDLERS registry
│   └── tools/           — 25 tools in 7 domain-scoped modules
├── orchestrator/
│   ├── state.py         — PipelineState TypedDict + boundary validators
│   ├── graph.py         — LangGraph pipeline graph
│   └── router.py        — Status-based routing logic
├── sandbox/
│   └── executor.py      — Sandboxed runner (Docker mode + subprocess fallback)
├── domain_rules/        — YAML constraint files per domain
├── hitl/                — Human-in-the-loop checkpoint (Redis-backed polling)
├── observability/       — Langfuse tracing wrappers
├── scripts/             — migrate.py, seed_library.py
├── tests/               — 77 unit + integration tests, no live API needed
├── sample_data/         — sales_raw.csv for smoke testing
└── outputs/             — Per-run output directory (gitignored)
```

## Design rationale

See [DESIGN.md](DESIGN.md) for the rationale behind each architectural decision.
