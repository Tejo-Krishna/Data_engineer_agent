"""
End-to-end smoke test for the full pipeline graph.

Runs the complete 5-agent pipeline against sample_data/sales_raw.csv and
auto-approves the HITL code checkpoint via Redis so no human input is needed.

Prerequisites (must be running):
    docker-compose up -d redis postgres

    ANTHROPIC_API_KEY must be set in .env

Run with:
    pytest tests/test_graph.py -v -s

This test is slower than unit tests (~2-3 minutes) due to LLM calls.
Mark with -m e2e to separate from fast tests:
    pytest tests/test_graph.py -m e2e -v -s
"""

import asyncio
import json
import time
from pathlib import Path

import pytest

from conftest import SAMPLE_CSV, PROJECT_ROOT


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# HITL auto-approver — runs concurrently with the pipeline
# ---------------------------------------------------------------------------


async def _auto_approve_hitl(run_id: str, timeout: float = 300.0) -> None:
    """
    Poll Redis for a pending code checkpoint and approve it automatically.
    Also handles the drift checkpoint if one appears.
    """
    from db import get_redis_client

    redis = await get_redis_client()
    code_key = f"hitl:{run_id}:code"
    drift_key = f"hitl:{run_id}:drift"
    deadline = time.time() + timeout

    approved_code = False
    approved_drift = False

    while time.time() < deadline:
        # Approve drift if pending
        if not approved_drift:
            raw = await redis.get(drift_key)
            if raw:
                data = json.loads(raw)
                if data.get("state") == "pending":
                    data["state"] = "confirmed"
                    await redis.setex(drift_key, 3600, json.dumps(data))
                    approved_drift = True

        # Approve code if pending
        if not approved_code:
            raw = await redis.get(code_key)
            if raw:
                data = json.loads(raw)
                if data.get("state") == "pending":
                    data["state"] = "confirmed"
                    await redis.setex(code_key, 3600, json.dumps(data))
                    approved_code = True

        if approved_code:
            return

        await asyncio.sleep(2)

    raise TimeoutError(f"HITL auto-approver timed out after {timeout}s for run {run_id}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


@pytest.mark.e2e
async def test_full_pipeline_success():
    """
    Full pipeline run against sales_raw.csv.

    Validates:
    - pipeline completes without exception
    - status is 'success'
    - rows_output is between 400 and 500
    - quality_passed is True
    - all expected output files exist on disk
    - output Parquet is readable and has the expected columns
    """
    import pyarrow.parquet as pq

    from dotenv import load_dotenv
    load_dotenv()

    from main import run_pipeline

    # Run pipeline and auto-approve HITL concurrently
    # We need the run_id before starting — get it by hooking into state creation.
    # Simpler: run pipeline in a task and start the approver once run_id is known.

    run_ids: list[str] = []

    async def _pipeline_wrapper():
        result = await run_pipeline(
            source_path=str(SAMPLE_CSV),
            source_type="csv",
            user_goal="clean dates, dedup, convert GBP to USD",
            incremental=False,
        )
        return result

    pipeline_task = asyncio.create_task(_pipeline_wrapper())

    # Wait briefly for the run_id to appear in the DB, then poll Redis
    # Easier: use asyncio to race the approver alongside the pipeline task
    # The approver polls run_ids from the DB but we don't have access to run_id yet.
    # Instead: approve any pending HITL across all recent run_ids.

    async def _approver_loop():
        """Poll ALL recent hitl:*:code keys and approve any in pending state."""
        from db import get_redis_client
        redis = await get_redis_client()
        deadline = time.time() + 400
        approved = False
        while time.time() < deadline and not pipeline_task.done():
            # scan for any hitl:*:code key in pending state
            async for key in redis.scan_iter("hitl:*:code"):
                raw = await redis.get(key)
                if raw:
                    data = json.loads(raw)
                    if data.get("state") == "pending":
                        data["state"] = "confirmed"
                        await redis.setex(key, 3600, json.dumps(data))
                        approved = True
            async for key in redis.scan_iter("hitl:*:drift"):
                raw = await redis.get(key)
                if raw:
                    data = json.loads(raw)
                    if data.get("state") == "pending":
                        data["state"] = "confirmed"
                        await redis.setex(key, 3600, json.dumps(data))
            await asyncio.sleep(2)

    approver_task = asyncio.create_task(_approver_loop())

    try:
        final = await pipeline_task
    finally:
        approver_task.cancel()
        try:
            await approver_task
        except asyncio.CancelledError:
            pass

    # --- Assertions ---

    assert final["status"] == "success", (
        f"Pipeline failed with status={final['status']!r}, "
        f"reason={final.get('failure_reason')}"
    )

    assert final["quality_passed"] is True, "Quality checks did not pass"

    rows_out = final.get("rows_output", 0)
    assert 400 <= rows_out <= 500, f"Unexpected rows_output: {rows_out}"

    run_id = final["run_id"]
    output_dir = PROJECT_ROOT / "outputs" / run_id

    # All expected output files must exist
    expected_files = {
        "Parquet":         final.get("output_path"),
        "pipeline_script": final.get("pipeline_script"),
        "quality_report":  final.get("quality_report_path"),
        "dbt_model":       final.get("dbt_model_path"),
        "dbt_schema":      final.get("dbt_schema_path"),
        "dbt_tests":       final.get("dbt_tests_path"),
    }

    for label, path in expected_files.items():
        assert path is not None, f"{label} path is None in final state"
        assert Path(path).exists(), f"{label} file does not exist: {path}"

    # Output Parquet must be readable and have transformed columns
    parquet_path = final["output_path"]
    table = pq.read_table(parquet_path)
    col_names = table.schema.names

    assert "transaction_id" in col_names, "transaction_id missing from output"
    assert table.num_rows == rows_out, "Parquet row count does not match state"


@pytest.mark.e2e
async def test_pipeline_domain_detection():
    """Verify domain is detected as 'retail' with high confidence for sales data."""
    from dotenv import load_dotenv
    load_dotenv()

    from main import run_pipeline

    pipeline_task = asyncio.create_task(
        run_pipeline(
            source_path=str(SAMPLE_CSV),
            source_type="csv",
            user_goal="clean and normalise sales data",
            incremental=False,
        )
    )

    async def _approver():
        from db import get_redis_client
        redis = await get_redis_client()
        deadline = time.time() + 400
        while time.time() < deadline and not pipeline_task.done():
            async for key in redis.scan_iter("hitl:*:code"):
                raw = await redis.get(key)
                if raw:
                    data = json.loads(raw)
                    if data.get("state") == "pending":
                        data["state"] = "confirmed"
                        await redis.setex(key, 3600, json.dumps(data))
            await asyncio.sleep(2)

    approver = asyncio.create_task(_approver())
    try:
        final = await pipeline_task
    finally:
        approver.cancel()
        try:
            await approver
        except asyncio.CancelledError:
            pass

    assert final.get("domain") == "retail"
    assert final.get("domain_confidence", 0) >= 0.75, (
        f"Expected high confidence for obvious retail data, got {final.get('domain_confidence')}"
    )


@pytest.mark.e2e
async def test_pipeline_schema_drift_none_on_second_run():
    """
    After one successful run, a second run on the same source should
    report no drift (same schema stored and compared).
    """
    from dotenv import load_dotenv
    load_dotenv()

    from main import run_pipeline

    async def _run_once():
        task = asyncio.create_task(
            run_pipeline(
                source_path=str(SAMPLE_CSV),
                source_type="csv",
                user_goal="clean dates and dedup",
                incremental=False,
            )
        )

        async def _approve():
            from db import get_redis_client
            redis = await get_redis_client()
            deadline = time.time() + 400
            while time.time() < deadline and not task.done():
                async for key in redis.scan_iter("hitl:*:code"):
                    raw = await redis.get(key)
                    if raw:
                        data = json.loads(raw)
                        if data.get("state") == "pending":
                            data["state"] = "confirmed"
                            await redis.setex(key, 3600, json.dumps(data))
                await asyncio.sleep(2)

        approver = asyncio.create_task(_approve())
        try:
            result = await task
        finally:
            approver.cancel()
            try:
                await approver
            except asyncio.CancelledError:
                pass
        return result

    first = await _run_once()
    assert first["status"] == "success"

    second = await _run_once()

    drift = second.get("schema_drift", {})
    assert drift.get("drift_severity") in ("none", "warning"), (
        f"Expected no critical drift on second run, got: {drift}"
    )
