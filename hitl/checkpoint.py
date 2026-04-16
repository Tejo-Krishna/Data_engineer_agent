"""
HITL (Human-In-The-Loop) checkpoint module.

Redis state machine for code review and schema drift approval.
FastAPI router mounted at /hitl on the MCP server.

Redis key schema:
    hitl:{run_id}:code   — JSON, TTL 3600 s
    hitl:{run_id}:drift  — JSON, TTL 3600 s

Code review state machine:
    pending
      → confirmed          (approve, no instruction → ready to execute_code)
      → approved_with_instruction
          → awaiting_confirm
              → confirmed  (human confirms revised code)
              → pending    (loop back — human wants another revision)
      → rejected           (pipeline status = "failed")

Drift review state machine (simpler):
    pending → confirmed | rejected
"""

import asyncio
import json
import os
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from db import get_redis_client

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def _positive_float(env_var: str, default: str) -> float:
    try:
        v = float(os.getenv(env_var, default))
    except ValueError:
        raise ValueError(f"{env_var} must be a number, got: {os.getenv(env_var)!r}")
    if v <= 0:
        raise ValueError(f"{env_var} must be positive, got: {v}")
    return v

HITL_TTL: int = int(_positive_float("HITL_TTL_SECONDS", "3600"))
HITL_POLL_INTERVAL: float = _positive_float("HITL_POLL_INTERVAL_SECONDS", "3.0")
HITL_TIMEOUT: float = _positive_float("HITL_TIMEOUT_SECONDS", "1800.0")


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------


_WAITING_KEY = "hitl:waiting"   # Redis set — run_ids currently at HITL checkpoint


def _code_key(run_id: str) -> str:
    return f"hitl:{run_id}:code"


def _drift_key(run_id: str) -> str:
    return f"hitl:{run_id}:drift"


async def _set(key: str, data: dict) -> None:
    redis = await get_redis_client()
    await redis.setex(key, HITL_TTL, json.dumps(data))


async def _get(key: str) -> dict | None:
    redis = await get_redis_client()
    raw = await redis.get(key)
    if raw is None:
        return None
    return json.loads(raw)


async def _mark_waiting(run_id: str) -> None:
    """Register run_id as waiting at a HITL checkpoint."""
    try:
        redis = await get_redis_client()
        await redis.sadd(_WAITING_KEY, run_id)
        # TTL on the set: if the server crashes mid-wait, the entry ages out
        await redis.expire(_WAITING_KEY, HITL_TTL)
    except Exception:
        pass


async def _clear_waiting(run_id: str) -> None:
    """Remove run_id from the waiting set after decision or timeout."""
    try:
        redis = await get_redis_client()
        await redis.srem(_WAITING_KEY, run_id)
    except Exception:
        pass


async def list_waiting_runs() -> list[str]:
    """
    Return all run_ids currently waiting at a HITL checkpoint.

    Used by hitl_approve.py and the /hitl/pending API endpoint to show
    operators which runs need attention.
    """
    try:
        redis = await get_redis_client()
        members = await redis.smembers(_WAITING_KEY)
        return [m.decode() if isinstance(m, bytes) else m for m in members]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class CodeSubmitRequest(BaseModel):
    code: str
    transformations: list[str]
    source_path: str
    domain: str
    domain_context: dict


class CodeApproveRequest(BaseModel):
    approved: bool
    nlp_instruction: str | None = None


class CodeDiffRequest(BaseModel):
    diff_summary: str
    revised_code: str


class CodeConfirmRequest(BaseModel):
    confirmed: bool


class DriftSubmitRequest(BaseModel):
    drift_report: dict


class DriftApproveRequest(BaseModel):
    approved: bool


# ---------------------------------------------------------------------------
# FastAPI router — 9 endpoints, all prefixed /hitl in server.py
# ---------------------------------------------------------------------------

hitl_router = APIRouter()


# -- Pending runs endpoint ---------------------------------------------------


@hitl_router.get("/pending")
async def list_pending() -> dict:
    """
    List all run_ids currently waiting at a HITL checkpoint.

    Use this endpoint (or `python scripts/hitl_approve.py --list`) to see
    which runs need human attention before the pipeline can proceed.
    """
    waiting = await list_waiting_runs()
    return {"waiting_runs": waiting, "count": len(waiting)}


# -- Code review endpoints ---------------------------------------------------


@hitl_router.post("/{run_id}/code/submit")
async def code_submit(run_id: str, req: CodeSubmitRequest) -> dict:
    """
    Transformer posts generated code for human review.

    Transitions state to 'pending'. Overwrites any existing pending entry
    for this run_id (useful when retrying after rejection).
    """
    await _set(
        _code_key(run_id),
        {
            "state": "pending",
            "submitted_at": time.time(),
            "code": req.code,
            "transformations": req.transformations,
            "source_path": req.source_path,
            "domain": req.domain,
            "domain_context": req.domain_context,
            "nlp_instruction": None,
            "diff_summary": None,
            "revised_code": None,
        },
    )
    return {"status": "submitted", "run_id": run_id}


@hitl_router.get("/{run_id}/code/review")
async def code_review(run_id: str) -> dict:
    """
    Human fetches the pending code payload for review.

    Returns the full payload including code, transformations, domain context.
    Returns 404 if the run_id does not exist or has expired.
    """
    data = await _get(_code_key(run_id))
    if data is None:
        raise HTTPException(status_code=404, detail=f"No code checkpoint for run {run_id}")
    return data


@hitl_router.post("/{run_id}/code/approve")
async def code_approve(run_id: str, req: CodeApproveRequest) -> dict:
    """
    Human approves or rejects the generated code.

    approved=True, nlp_instruction=None  → state: confirmed (execute immediately)
    approved=True, nlp_instruction=<str> → state: approved_with_instruction
    approved=False                        → state: rejected
    """
    data = await _get(_code_key(run_id))
    if data is None:
        raise HTTPException(status_code=404, detail=f"No code checkpoint for run {run_id}")
    if data["state"] != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot approve from state '{data['state']}'",
        )

    if not req.approved:
        data["state"] = "rejected"
    elif req.nlp_instruction:
        data["state"] = "approved_with_instruction"
        data["nlp_instruction"] = req.nlp_instruction
    else:
        data["state"] = "confirmed"

    await _set(_code_key(run_id), data)
    return {"status": data["state"], "run_id": run_id}


@hitl_router.post("/{run_id}/code/diff")
async def code_diff_submit(run_id: str, req: CodeDiffRequest) -> dict:
    """
    Transformer posts the revised code diff after refine_transform_code.

    Transitions from approved_with_instruction → awaiting_confirm.
    Human then reads the diff via GET and confirms or loops back.
    """
    data = await _get(_code_key(run_id))
    if data is None:
        raise HTTPException(status_code=404, detail=f"No code checkpoint for run {run_id}")
    if data["state"] != "approved_with_instruction":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot post diff from state '{data['state']}'",
        )

    data["state"] = "awaiting_confirm"
    data["diff_summary"] = req.diff_summary
    data["revised_code"] = req.revised_code
    await _set(_code_key(run_id), data)
    return {"status": "awaiting_confirm", "run_id": run_id}


@hitl_router.get("/{run_id}/code/diff")
async def code_diff_review(run_id: str) -> dict:
    """
    Human reads the diff summary and revised code before confirming.

    Returns the full payload. Returns 404 if checkpoint expired.
    """
    data = await _get(_code_key(run_id))
    if data is None:
        raise HTTPException(status_code=404, detail=f"No code checkpoint for run {run_id}")
    return {
        "state": data["state"],
        "diff_summary": data.get("diff_summary"),
        "revised_code": data.get("revised_code"),
        "original_code": data.get("code"),
        "nlp_instruction": data.get("nlp_instruction"),
    }


@hitl_router.post("/{run_id}/code/confirm")
async def code_confirm(run_id: str, req: CodeConfirmRequest) -> dict:
    """
    Human confirms (or rejects) the revised code after diff review.

    confirmed=True  → state: confirmed
    confirmed=False → state: pending (loop back — triggers another revision)
    """
    data = await _get(_code_key(run_id))
    if data is None:
        raise HTTPException(status_code=404, detail=f"No code checkpoint for run {run_id}")
    if data["state"] != "awaiting_confirm":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot confirm from state '{data['state']}'",
        )

    if req.confirmed:
        # Promote revised_code to the active code
        if data.get("revised_code"):
            data["code"] = data["revised_code"]
        data["state"] = "confirmed"
    else:
        # Loop back — human wants another revision attempt
        data["state"] = "pending"
        data["revised_code"] = None
        data["diff_summary"] = None

    await _set(_code_key(run_id), data)
    return {"status": data["state"], "run_id": run_id}


# -- Drift review endpoints --------------------------------------------------


@hitl_router.post("/{run_id}/drift/submit")
async def drift_submit(run_id: str, req: DriftSubmitRequest) -> dict:
    """Profiler agent posts the drift report for human review."""
    await _set(
        _drift_key(run_id),
        {
            "state": "pending",
            "submitted_at": time.time(),
            "drift_report": req.drift_report,
        },
    )
    return {"status": "submitted", "run_id": run_id}


@hitl_router.get("/{run_id}/drift/review")
async def drift_review(run_id: str) -> dict:
    """Human fetches the drift report payload."""
    data = await _get(_drift_key(run_id))
    if data is None:
        raise HTTPException(status_code=404, detail=f"No drift checkpoint for run {run_id}")
    return data


@hitl_router.post("/{run_id}/drift/approve")
async def drift_approve(run_id: str, req: DriftApproveRequest) -> dict:
    """Human approves or rejects continuation after schema drift."""
    data = await _get(_drift_key(run_id))
    if data is None:
        raise HTTPException(status_code=404, detail=f"No drift checkpoint for run {run_id}")
    if data["state"] != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot approve drift from state '{data['state']}'",
        )

    data["state"] = "confirmed" if req.approved else "rejected"
    await _set(_drift_key(run_id), data)
    return {"status": data["state"], "run_id": run_id}


# ---------------------------------------------------------------------------
# Polling helpers — called from agents, not from HTTP endpoints
# ---------------------------------------------------------------------------


async def hitl_code_checkpoint(
    run_id: str,
    code: str,
    transformations: list[str],
    source_path: str,
    domain: str,
    domain_context: dict,
) -> dict[str, Any]:
    """
    Submit generated code and poll until a terminal state is reached.

    Handles the full state machine including the refinement loop.
    Returns:
        {
            "approved": bool,
            "code": str,              # original or revised
            "nlp_instruction": str,   # None if no instruction
            "needs_refinement": bool,
        }
    Raises TimeoutError if HITL_TIMEOUT seconds elapse without resolution.
    Raises ValueError if the checkpoint is rejected.
    """
    # Submit to Redis
    await _set(
        _code_key(run_id),
        {
            "state": "pending",
            "submitted_at": time.time(),
            "code": code,
            "transformations": transformations,
            "source_path": source_path,
            "domain": domain,
            "domain_context": domain_context,
            "nlp_instruction": None,
            "diff_summary": None,
            "revised_code": None,
        },
    )

    # Register as waiting so hitl_approve.py / /hitl/pending can list this run
    await _mark_waiting(run_id)

    deadline = time.time() + HITL_TIMEOUT

    try:
        while time.time() < deadline:
            await asyncio.sleep(HITL_POLL_INTERVAL)
            data = await _get(_code_key(run_id))
            if data is None:
                raise TimeoutError(f"HITL checkpoint expired for run {run_id}")

            state = data["state"]

            if state == "confirmed":
                return {
                    "approved": True,
                    "code": data.get("revised_code") or data["code"],
                    "nlp_instruction": data.get("nlp_instruction"),
                    "needs_refinement": False,
                }

            if state == "rejected":
                raise ValueError(f"HITL code checkpoint rejected for run {run_id}")

            if state == "approved_with_instruction":
                # Return to agent so it can call refine_transform_code
                return {
                    "approved": True,
                    "code": data["code"],
                    "nlp_instruction": data["nlp_instruction"],
                    "needs_refinement": True,
                }

            # Still pending / awaiting_confirm — keep polling
            continue

        raise TimeoutError(
            f"HITL timeout after {HITL_TIMEOUT}s for run {run_id}"
        )
    finally:
        # Always deregister — whether resolved, timed out, or rejected
        await _clear_waiting(run_id)


async def hitl_confirm_after_refinement(run_id: str) -> dict[str, Any]:
    """
    Poll after transformer has posted the diff (awaiting_confirm state).

    Returns:
        {"confirmed": bool, "code": str}
        confirmed=True  → proceed to execute_code with revised code
        confirmed=False → loop back (agent should call hitl_code_checkpoint again)
    Raises TimeoutError if HITL_TIMEOUT elapses.
    """
    deadline = time.time() + HITL_TIMEOUT

    while time.time() < deadline:
        await asyncio.sleep(HITL_POLL_INTERVAL)
        data = await _get(_code_key(run_id))
        if data is None:
            raise TimeoutError(f"HITL checkpoint expired during confirm for run {run_id}")

        state = data["state"]

        if state == "confirmed":
            return {"confirmed": True, "code": data["code"]}

        if state == "pending":
            # Human looped back — signal the transformer to regenerate
            return {"confirmed": False, "code": data["code"]}

        if state == "rejected":
            raise ValueError(f"HITL code checkpoint rejected during confirm for run {run_id}")

        # Still awaiting_confirm — keep polling
        continue

    raise TimeoutError(
        f"HITL confirm timeout after {HITL_TIMEOUT}s for run {run_id}"
    )


async def hitl_post_diff(
    run_id: str, diff_summary: str, revised_code: str
) -> None:
    """
    Called by the transformer agent after refine_transform_code to push
    the diff into Redis so the human can review it via GET /hitl/{run_id}/code/diff.

    Transitions state: approved_with_instruction → awaiting_confirm.
    No-op if the checkpoint is not in approved_with_instruction state.
    """
    data = await _get(_code_key(run_id))
    if data and data["state"] == "approved_with_instruction":
        data["state"] = "awaiting_confirm"
        data["diff_summary"] = diff_summary
        data["revised_code"] = revised_code
        await _set(_code_key(run_id), data)


async def hitl_drift_checkpoint(run_id: str, drift_report: dict) -> bool:
    """
    Submit a drift report and poll until the human approves or rejects.

    Returns True if approved, False if rejected.
    Raises TimeoutError if HITL_TIMEOUT elapses.
    """
    await _set(
        _drift_key(run_id),
        {
            "state": "pending",
            "submitted_at": time.time(),
            "drift_report": drift_report,
        },
    )

    deadline = time.time() + HITL_TIMEOUT

    while time.time() < deadline:
        await asyncio.sleep(HITL_POLL_INTERVAL)
        data = await _get(_drift_key(run_id))
        if data is None:
            raise TimeoutError(f"Drift HITL checkpoint expired for run {run_id}")

        state = data["state"]
        if state == "confirmed":
            return True
        if state == "rejected":
            return False
        # Still pending — keep polling
        continue

    raise TimeoutError(
        f"Drift HITL timeout after {HITL_TIMEOUT}s for run {run_id}"
    )
