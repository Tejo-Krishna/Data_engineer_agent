"""
Pipeline API router — all /api/* endpoints consumed by the frontend.

Endpoints:
  POST /api/runs                       start a new pipeline run
  GET  /api/runs                       list recent runs
  GET  /api/runs/{run_id}              run detail + HITL state
  GET  /api/runs/{run_id}/events       SSE stream of live events
  GET  /api/runs/{run_id}/quality      quality report JSON
  GET  /api/runs/{run_id}/lineage      mermaid diagram
  GET  /api/runs/{run_id}/dbt/{file}   dbt artefact file content
  GET  /api/runs/{run_id}/download/{type}  file download
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from api.event_bus import publish, subscribe, unsubscribe
from db import get_postgres_pool

router = APIRouter()

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class StartRunRequest(BaseModel):
    source_path: str
    source_type: str = "csv"
    user_goal: str
    incremental_mode: bool = False
    source_table: Optional[str] = None


# ---------------------------------------------------------------------------
# Background pipeline runner (wraps main.run_pipeline and publishes events)
# ---------------------------------------------------------------------------

async def _run_pipeline_bg(run_id: str, req: StartRunRequest) -> None:
    """Run the pipeline in the background and publish events to the bus."""
    from main import run_pipeline  # import here to avoid circular imports at module load

    publish(run_id, "run_started", {"run_id": run_id})

    try:
        final = await run_pipeline(
            source_path=req.source_path,
            source_type=req.source_type,
            user_goal=req.user_goal,
            incremental=req.incremental_mode,
            source_table=req.source_table,
            run_id=run_id,
        )
        publish(run_id, "run_complete", {
            "run_id": run_id,
            "status": final.get("status", "unknown"),
            "rows_input": final.get("rows_input"),
            "rows_output": final.get("rows_output"),
            "quality_passed": final.get("quality_passed"),
            "output_path": final.get("output_path"),
        })
    except Exception as exc:
        publish(run_id, "run_failed", {"run_id": run_id, "error": str(exc)})


# ---------------------------------------------------------------------------
# POST /api/runs — start a run
# ---------------------------------------------------------------------------

@router.post("/runs")
async def start_run(req: StartRunRequest, background_tasks: BackgroundTasks):
    import uuid
    run_id = str(uuid.uuid4())

    # Insert a placeholder row so GET /api/runs shows it immediately
    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO pipeline_runs (id, source_path, source_type, status)
            VALUES ($1, $2, $3, 'running')
            ON CONFLICT (id) DO NOTHING
            """,
            run_id, req.source_path, req.source_type,
        )

    background_tasks.add_task(_run_pipeline_bg, run_id, req)
    return {"run_id": run_id}


# ---------------------------------------------------------------------------
# GET /api/runs — list recent runs
# ---------------------------------------------------------------------------

@router.get("/runs")
async def list_runs():
    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, source_path, source_type, status,
                   started_at, completed_at,
                   rows_input, rows_output, quality_passed, watermark_value
            FROM pipeline_runs
            ORDER BY started_at DESC
            LIMIT 30
            """
        )
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# GET /api/runs/{run_id} — run detail
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM pipeline_runs WHERE id = $1", run_id
        )
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")

    result = dict(row)

    # Attach HITL state from Redis if run is still active
    try:
        from hitl.checkpoint import _get, _code_key, _drift_key
        code_state = await _get(_code_key(run_id))
        drift_state = await _get(_drift_key(run_id))
        result["hitl_code"] = code_state
        result["hitl_drift"] = drift_state
    except Exception:
        result["hitl_code"] = None
        result["hitl_drift"] = None

    return result


# ---------------------------------------------------------------------------
# GET /api/runs/{run_id}/events — SSE stream
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}/events")
async def run_events(run_id: str):
    q = subscribe(run_id)

    async def event_stream():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=25)
                    yield f"data: {json.dumps(event, default=str)}\n\n"
                    if event.get("type") in ("run_complete", "run_failed"):
                        break
                except asyncio.TimeoutError:
                    # Send a keepalive comment so the connection stays open
                    yield ": keepalive\n\n"
        finally:
            unsubscribe(run_id, q)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# GET /api/runs/{run_id}/quality — quality report JSON
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}/quality")
async def get_quality(run_id: str):
    path = OUTPUT_DIR / run_id / "quality_report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Quality report not found")
    async with aiofiles.open(path) as f:
        return json.loads(await f.read())


# ---------------------------------------------------------------------------
# GET /api/runs/{run_id}/lineage — mermaid diagram
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}/lineage")
async def get_lineage(run_id: str):
    path = OUTPUT_DIR / run_id / "lineage.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Lineage not available yet")
    async with aiofiles.open(path) as f:
        return json.loads(await f.read())


# ---------------------------------------------------------------------------
# GET /api/runs/{run_id}/dbt/{filename} — dbt file content
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}/dbt/{filename}")
async def get_dbt_file(run_id: str, filename: str):
    # Prevent path traversal
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    path = OUTPUT_DIR / run_id / "dbt" / "models" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not found")

    async with aiofiles.open(path) as f:
        content = await f.read()

    return {"filename": filename, "content": content}


# ---------------------------------------------------------------------------
# GET /api/runs/{run_id}/download/{file_type} — file downloads
# ---------------------------------------------------------------------------

_DOWNLOAD_MAP = {
    "parquet":         lambda r: OUTPUT_DIR / r / "output.parquet",
    "pipeline_script": lambda r: OUTPUT_DIR / r / "pipeline.py",
    "quality_json":    lambda r: OUTPUT_DIR / r / "quality_report.json",
    "quality_md":      lambda r: OUTPUT_DIR / r / "quality_report.md",
    "dbt_model":       lambda r: OUTPUT_DIR / r / "dbt" / "models" / f"pipeline_{r[:8].replace('-','_')}.sql",
    "dbt_schema":      lambda r: OUTPUT_DIR / r / "dbt" / "models" / "schema.yml",
    "dbt_tests":       lambda r: OUTPUT_DIR / r / "dbt" / "models" / "schema_tests.yml",
}


@router.get("/runs/{run_id}/download/{file_type}")
async def download_file(run_id: str, file_type: str):
    resolver = _DOWNLOAD_MAP.get(file_type)
    if not resolver:
        raise HTTPException(status_code=400, detail=f"Unknown file_type: {file_type}")

    path = resolver(run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{file_type} not available for this run")

    return FileResponse(path=str(path), filename=path.name)
