"""
Data Engineering Agent — entry point.

Usage:
    python main.py --source sample_data/sales_raw.csv --type csv \
                   --goal "clean dates, dedup, convert GBP to USD"

    python main.py --source postgresql://user:pass@localhost:5432/db::orders \
                   --type postgres \
                   --goal "normalise phone numbers and title-case names" \
                   --incremental
"""

import argparse
import asyncio
import hashlib
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_LARGE_FILE_THRESHOLD_MB: float = float(os.getenv("LARGE_FILE_THRESHOLD_MB", "500"))

from direct_client import DirectClient
from db import get_postgres_pool, close_postgres_pool
from observability.tracing import start_trace, finish_trace
from orchestrator.graph import build_graph
from orchestrator.state import PipelineState

MCP_HOST = os.getenv("MCP_SERVER_HOST", "localhost")
MCP_PORT = os.getenv("MCP_SERVER_PORT", "8000")


# ---------------------------------------------------------------------------
# Idempotency helpers (Gap 2)
# ---------------------------------------------------------------------------


def _make_run_key(source_path: str, user_goal: str) -> str:
    """
    Compute a deterministic content-address for (source file, goal).

    For large files, hashes only first + last 1 MB to stay fast.
    For non-file sources (postgres, api), hashes the path string only.
    """
    try:
        p = Path(source_path)
        if p.exists() and p.is_file():
            size = p.stat().st_size
            one_mb = 1024 * 1024
            with open(p, "rb") as f:
                if size <= 2 * one_mb:
                    content = f.read()
                else:
                    head = f.read(one_mb)
                    f.seek(-one_mb, 2)
                    tail = f.read(one_mb)
                    content = head + tail
            file_hash = hashlib.sha256(content).hexdigest()[:16]
        else:
            file_hash = hashlib.sha256(source_path.encode()).hexdigest()[:16]
    except OSError:
        file_hash = hashlib.sha256(source_path.encode()).hexdigest()[:16]

    goal_hash = hashlib.sha256(user_goal.encode()).hexdigest()[:8]
    return f"{file_hash}-{goal_hash}"


async def _check_prior_run(run_key: str) -> dict | None:
    """
    Return the most recent successful run for this run_key, or None.
    Only called when --force is not set.
    """
    try:
        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, output_path, quality_passed, rows_input, rows_output
                FROM pipeline_runs
                WHERE run_key = $1
                  AND status   = 'success'
                ORDER BY completed_at DESC
                LIMIT 1
                """,
                run_key,
            )
        if row:
            return dict(row)
    except Exception:
        pass
    return None


def _check_large_file(source_path: str, source_type: str) -> bool:
    """Return True if the source file exceeds LARGE_FILE_THRESHOLD_MB."""
    if source_type not in ("csv", "parquet"):
        return False
    try:
        size_mb = Path(source_path).stat().st_size / (1024 * 1024)
        return size_mb > _LARGE_FILE_THRESHOLD_MB
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Rollback helper — finalize staging→final (Gap 3)
# ---------------------------------------------------------------------------


def _finalize_run(run_id: str, state: PipelineState) -> PipelineState:
    """
    Rename outputs/{run_id}/staging → outputs/{run_id}/final.

    This is an atomic filesystem rename (same-device). It is called only
    after status=="success", giving a rollback guarantee: if any earlier
    agent fails, staging is never promoted and final/ never exists.

    Also rewrites all staging path references in state to final paths.
    """
    output_base = Path(os.getenv("OUTPUT_DIR", "outputs"))
    staging = output_base / run_id / "staging"
    final = output_base / run_id / "final"

    if not staging.exists():
        return state

    staging.rename(final)

    # Rewrite path fields so state is consistent after finalization
    staging_str = str(staging)
    final_str = str(final)
    updates: dict = {}
    for key in (
        "output_path", "quality_report_path", "pipeline_script",
        "dbt_model_path", "dbt_schema_path", "dbt_tests_path",
    ):
        val = state.get(key)
        if isinstance(val, str) and staging_str in val:
            updates[key] = val.replace(staging_str, final_str)

    return {**state, **updates} if updates else state


# ---------------------------------------------------------------------------
# Watermark helpers (incremental mode)
# ---------------------------------------------------------------------------


async def _get_previous_watermark(source_path: str) -> str | None:
    """Read the most recent successful watermark for this source from Postgres."""
    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT watermark_value
            FROM pipeline_runs
            WHERE source_path = $1
              AND status = 'success'
              AND watermark_value IS NOT NULL
            ORDER BY completed_at DESC
            LIMIT 1
            """,
            source_path,
        )
    return row["watermark_value"] if row else None


async def _save_watermark(run_id: str, watermark_value: str) -> None:
    """Persist the new watermark after a successful incremental run."""
    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE pipeline_runs SET watermark_value = $1 WHERE id = $2",
            watermark_value,
            run_id,
        )


async def _insert_pipeline_run(
    run_id: str, source_path: str, source_type: str, run_key: str | None = None
) -> None:
    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO pipeline_runs (id, run_key, source_path, source_type, status)
            VALUES ($1, $2, $3, $4, 'running')
            ON CONFLICT (id) DO NOTHING
            """,
            run_id,
            run_key,
            source_path,
            source_type,
        )


async def _update_pipeline_run(run_id: str, final: PipelineState) -> None:
    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE pipeline_runs
            SET status         = $1,
                output_path    = $2,
                quality_passed = $3,
                rows_input     = $4,
                rows_output    = $5,
                completed_at   = NOW()
            WHERE id = $6
            """,
            final.get("status"),
            final.get("output_path"),
            final.get("quality_passed"),
            final.get("rows_input"),
            final.get("rows_output"),
            run_id,
        )


# ---------------------------------------------------------------------------
# Core pipeline runner
# ---------------------------------------------------------------------------


async def run_pipeline(
    source_path: str,
    source_type: str,
    user_goal: str,
    incremental: bool,
    source_table: str | None = None,
    run_id: str | None = None,
    force: bool = False,
) -> PipelineState:
    is_url = source_path.startswith("http://") or source_path.startswith("https://")
    if source_type in ("csv", "parquet") and not is_url and not Path(source_path).exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    # --- Idempotency check (Gap 2) ---
    run_key: str | None = None
    if not incremental:
        run_key = _make_run_key(source_path, user_goal)
        if not force:
            prior = await _check_prior_run(run_key)
            if prior:
                print(f"\n✓ Returning cached result from prior run {prior['id']}")
                print(f"  Output: {prior.get('output_path')}")
                print(f"  Use --force to reprocess.\n")
                # Build a minimal state that looks like a completed run
                return {  # type: ignore[return-value]
                    "run_id": prior["id"],
                    "run_key": run_key,
                    "user_goal": user_goal,
                    "source_path": source_path,
                    "source_type": source_type,  # type: ignore[typeddict-item]
                    "source_table": source_table,
                    "incremental_mode": incremental,
                    "large_file": None,
                    "source_metadata": None, "sample": None, "profile": None,
                    "schema": None, "schema_drift": None, "drift_checkpoint_approved": None,
                    "domain": None, "domain_confidence": None, "domain_context": None,
                    "library_snippets": None, "watermark_value": None,
                    "generated_code": None, "transformations_applied": None,
                    "output_columns": None,
                    "output_path": prior.get("output_path"),
                    "rows_input": prior.get("rows_input"),
                    "rows_output": prior.get("rows_output"),
                    "pipeline_script": None, "quality_checks": None,
                    "anomaly_summary": None, "anomaly_explanations": None,
                    "quality_passed": prior.get("quality_passed"),
                    "quality_report_path": None, "retry_count": 0,
                    "failure_reason": None, "catalogue_id": None,
                    "lineage_graph": None, "mermaid_diagram": None,
                    "dbt_model_path": None, "dbt_schema_path": None, "dbt_tests_path": None,
                    "hitl_approved": None, "hitl_edits": None,
                    "status": "success",
                }

    # --- Large file detection (Gap 1) ---
    large_file = _check_large_file(source_path, source_type)
    if large_file:
        print(f"  ⚠  Large file detected (>{_LARGE_FILE_THRESHOLD_MB} MB). "
              "Transformation will use DuckDB streaming mode.")

    _run_id_was_provided = run_id is not None
    if run_id is None:
        run_id = str(uuid.uuid4())
    trace = start_trace(run_id, user_goal, source_path)

    print(f"\n{'='*60}")
    print(f"Run ID  : {run_id}")
    print(f"Source  : {source_path}  ({source_type})")
    print(f"Goal    : {user_goal}")
    print(f"Mode    : {'incremental' if incremental else 'full'}")
    print(f"{'='*60}\n")

    # Build initial state
    initial_state: PipelineState = {
        "run_id": run_id,
        "run_key": run_key,
        "user_goal": user_goal,
        "source_path": source_path,
        "source_type": source_type,
        "source_table": source_table,
        "incremental_mode": incremental,
        # Profiler
        "source_metadata": None,
        "sample": None,
        "profile": None,
        "schema": None,
        "schema_drift": None,
        "drift_checkpoint_approved": None,
        # Domain
        "domain": None,
        "domain_confidence": None,
        "domain_context": None,
        # Transformer
        "large_file": large_file or None,
        "library_snippets": None,
        "watermark_value": None,
        "generated_code": None,
        "transformations_applied": None,
        "output_columns": None,
        "output_path": None,
        "rows_input": None,
        "rows_output": None,
        "pipeline_script": None,
        # Quality
        "quality_checks": None,
        "anomaly_summary": None,
        "anomaly_explanations": None,
        "quality_passed": None,
        "quality_report_path": None,
        "retry_count": 0,
        "failure_reason": None,
        # Catalogue
        "catalogue_id": None,
        "lineage_graph": None,
        "mermaid_diagram": None,
        "dbt_model_path": None,
        "dbt_schema_path": None,
        "dbt_tests_path": None,
        # HITL
        "hitl_approved": None,
        "hitl_edits": None,
        # Final
        "status": "running",
    }

    # Persist run record (skip if already inserted by pipeline_router.py)
    if not _run_id_was_provided:
        await _insert_pipeline_run(run_id, source_path, source_type, run_key)

    graph = build_graph()
    final: PipelineState = initial_state

    async with DirectClient() as mcp:
        config = {
            "configurable": {
                "mcp": mcp,
                "logger": trace,
            }
        }

        try:
            final = await graph.ainvoke(initial_state, config)
        except Exception as exc:
            final = {**final, "status": "failed", "failure_reason": str(exc)}
            finish_trace(trace, "failed", final, error=str(exc))
            await _update_pipeline_run(run_id, final)
            raise

    # Finalize staging → final on success (Gap 3 rollback guarantee)
    if final.get("status") == "success":
        final = _finalize_run(run_id, final)

    # Persist watermark on successful incremental run
    if incremental and final.get("status") == "success" and final.get("watermark_value"):
        await _save_watermark(run_id, final["watermark_value"])

    await _update_pipeline_run(run_id, final)
    finish_trace(trace, final.get("status", "failed"), final)

    _print_outputs(final)
    return final


# ---------------------------------------------------------------------------
# Output summary
# ---------------------------------------------------------------------------


def _print_outputs(state: PipelineState) -> None:
    status = state.get("status", "unknown")
    symbol = "✓" if status == "success" else "✗"

    print(f"\n{symbol} Pipeline {status.upper()}")
    print(f"  Run ID         : {state.get('run_id')}")
    print(f"  Domain         : {state.get('domain', 'N/A')} "
          f"(confidence: {state.get('domain_confidence', 0):.2f})")
    print(f"  Rows in/out    : {state.get('rows_input', 0)} → {state.get('rows_output', 0)}")
    print(f"  Retries        : {state.get('retry_count', 0)}")

    if status == "failed":
        print(f"  Failure reason : {state.get('failure_reason', 'N/A')}")
        return

    print("\n  Output files:")
    outputs = {
        "Parquet":        state.get("output_path"),
        "Pipeline script":state.get("pipeline_script"),
        "Quality report": state.get("quality_report_path"),
        "dbt model":      state.get("dbt_model_path"),
        "dbt schema.yml": state.get("dbt_schema_path"),
        "dbt tests.yml":  state.get("dbt_tests_path"),
    }
    for label, path in outputs.items():
        marker = "  ✓" if path else "  -"
        print(f"  {marker}  {label:20s}: {path or 'N/A'}")

    if state.get("mermaid_diagram"):
        print("\n  Lineage diagram (Mermaid):")
        for line in (state["mermaid_diagram"] or "").splitlines():
            print(f"    {line}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Data Engineering Agent — transforms any data source end-to-end"
    )
    p.add_argument(
        "--source", required=True,
        help="File path, connection string, or API URL",
    )
    p.add_argument(
        "--type", dest="source_type", default="csv",
        choices=["csv", "parquet", "postgres", "api"],
        help="Source type (default: csv)",
    )
    p.add_argument(
        "--goal", required=True,
        help="Natural language transformation goal",
    )
    p.add_argument(
        "--incremental", action="store_true", default=False,
        help="Process only new rows since the last successful run",
    )
    p.add_argument(
        "--table", dest="source_table", default=None,
        help="Table name (postgres only, if not embedded in --source via ::)",
    )
    p.add_argument(
        "--force", action="store_true", default=False,
        help="Bypass idempotency cache and force a fresh run even if prior result exists",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(
        run_pipeline(
            source_path=args.source,
            source_type=args.source_type,
            user_goal=args.goal,
            incremental=args.incremental,
            source_table=args.source_table,
            force=args.force,
        )
    )
