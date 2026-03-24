# Data Engineering Multi-Agent System

A fully autonomous data engineering pipeline powered by five AI agents. Describe your data transformation goal in plain English — the system profiles your data, detects its domain, transforms it, validates quality, and delivers five production-ready outputs.

## What it does

You provide:
- A **data source** — CSV/Parquet file, PostgreSQL connection string, or REST API URL
- A **processing mode** — `full` (reprocess everything) or `incremental` (only new rows since last run)
- A **natural language goal** — e.g. `"clean dates, dedup, convert GBP to USD"`

The system produces, per successful run:

| Output | Description |
|--------|-------------|
| Cleaned Parquet file | Transformed dataset ready for downstream use |
| Pipeline script | Rerunnable Python script, schedulable as a cron job |
| Quality report | JSON + Markdown report with anomaly explanations |
| Catalogue entry | Column-level lineage stored in pgvector |
| dbt artefacts | SQL model + `schema.yml` + `schema_tests.yml` |

---

## Architecture

Five agents run in sequence, connected by a LangGraph orchestration layer:

```
Profiler → [drift checkpoint] → Domain → Transformer → Quality → Catalogue
                                              ↑              |
                                              └─── retry ────┘  (max 3)
```

| Agent | Role |
|-------|------|
| **Profiler** | Connects to source, samples 2 000 rows, detects schema drift, computes statistical profile |
| **Domain** | Classifies data domain (medical, financial, retail, automotive, employment) using keyword heuristics + LLM fallback; loads domain-specific validation rules |
| **Transformer** | ReAct loop — searches transform library, generates Python code via LLM, awaits human-in-the-loop approval, executes in sandbox |
| **Quality** | Runs targeted quality checks, detects anomalies, produces plain-English explanations via LLM |
| **Catalogue** | Writes catalogue entry, generates Mermaid lineage graph, produces dbt model + tests |

### MCP Server — 23 tools across 7 domains

The agents communicate exclusively through an MCP (Model Context Protocol) server running locally on port 8000. Tools are grouped into:

`source` · `profiling` · `transform` · `quality` · `catalogue` · `library` · `domain`

### Infrastructure (all local Docker)

| Service | Purpose | Port |
|---------|---------|------|
| PostgreSQL 16 + pgvector | Catalogue, transform library, pipeline runs, lineage | 5432 |
| Redis | HITL checkpoint state, watermark high-water marks | 6379 |
| Langfuse (self-hosted) | Full observability dashboard | 3000 |
| MCP Server (FastAPI) | Tool execution layer for agents | 8000 |

Only outbound traffic: Anthropic API calls. No data leaves your machine.

---

## Quick start

### 1. Prerequisites

- Docker and Docker Compose
- Python 3.11+
- An Anthropic API key
- An OpenAI API key (embeddings only)

### 2. Clone and configure

```bash
git clone https://github.com/<your-username>/data-agent.git
cd data-agent
cp .env.example .env
# Edit .env and fill in ANTHROPIC_API_KEY and OPENAI_API_KEY
```

### 3. Start infrastructure

```bash
docker compose up -d
```

Wait ~30 seconds for all services to become healthy, then visit [http://localhost:3000](http://localhost:3000) to set up Langfuse. Create an account, create a project, and copy the public/secret keys into `.env` as `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`.

### 4. Set up the database and seed the transform library

```bash
pip install -r requirements.txt
python scripts/migrate.py
python scripts/seed_library.py
```

### 5. Run the pipeline

```bash
# Full run on the included sample CSV
python main.py \
  --source sample_data/sales_raw.csv \
  --type csv \
  --goal "clean dates, deduplicate rows, convert GBP to USD"

# Incremental run — only processes rows added since the last run
python main.py \
  --source sample_data/sales_raw.csv \
  --type csv \
  --goal "clean dates, deduplicate rows, convert GBP to USD" \
  --incremental

# PostgreSQL source
python main.py \
  --source "postgresql://user:pass@localhost:5432/mydb::orders" \
  --type postgres \
  --goal "normalise phone numbers and title-case names"

# REST API source
python main.py \
  --source "https://api.example.com/sales" \
  --type api \
  --goal "filter active records and standardise country codes"
```

### Human-in-the-loop (HITL)

At two points the pipeline pauses and waits for your input (default timeout: 10 minutes):

1. **Schema drift checkpoint** — if the incoming schema has changed since the last run
2. **Transform approval** — review and approve (or edit) the generated transformation code before it executes

You can approve, provide a natural-language edit instruction, or reject. On retry runs the checkpoint is skipped automatically.

---

## Project structure

```
data-agent/
├── agents/              # Five agent implementations
├── mcp_server/
│   ├── server.py        # FastAPI + MCP server; registers all 23 tools
│   └── tools/           # Tool implementations (7 modules)
├── orchestrator/
│   ├── graph.py         # LangGraph graph definition
│   ├── router.py        # Routing logic (retry, drift, HITL)
│   └── state.py         # PipelineState TypedDict
├── domain_rules/        # YAML validation rules per domain
├── sandbox/             # Sandboxed Python code executor
├── hitl/                # HITL checkpoint server + Redis state
├── observability/       # Langfuse tracing wrappers
├── memory/              # Transform library + catalogue store
├── scripts/             # DB migration and library seeding
├── tests/               # Unit and integration tests
├── sample_data/         # sales_raw.csv (500 rows, with quality issues)
├── outputs/             # Pipeline-generated files (gitignored)
├── docker-compose.yml
├── requirements.txt
└── main.py              # CLI entry point
```

---

## Domain rules

The system ships with validation rules for five domains. Rules take precedence over user instructions — e.g. if `medical.yaml` forbids dropping null rows, the transformer will not generate code that drops them even if asked.

| Domain | File |
|--------|------|
| Medical | `domain_rules/medical.yaml` |
| Financial | `domain_rules/financial.yaml` |
| Retail | `domain_rules/retail.yaml` |
| Automotive | `domain_rules/automotive.yaml` |
| Employment | `domain_rules/employment.yaml` |

---

## Observability

Every agent call, every tool invocation, and every retry is traced to Langfuse. Open [http://localhost:3000](http://localhost:3000) after a run to see the full execution trace, token usage, and latency breakdown.

---

## Configuration

All configuration lives in `.env`. Key variables:

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Used for all LLM reasoning calls |
| `OPENAI_API_KEY` | Used for embeddings only (`text-embedding-3-small`) |
| `ANTHROPIC_MODEL` | Model ID (default: `claude-sonnet-4-5`) |
| `MAX_QUALITY_RETRIES` | Max transformer retries on quality failure (default: 3) |
| `SANDBOX_TIMEOUT_SECONDS` | Timeout for sandboxed code execution (default: 30) |
| `SAMPLE_SIZE` | Rows sampled for profiling (default: 2000) |
| `HITL_TIMEOUT_SECONDS` | Time to wait for human approval (default: 600) |

---

## Running tests

```bash
pytest tests/ -v
```

---

## License

MIT
