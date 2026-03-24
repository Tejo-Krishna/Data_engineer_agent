"""
LangGraph conditional edge router.

route_after_quality — reads state["status"] and state["retry_count"]
after the quality agent exits and returns one of three route keys:
  "catalogue" — quality passed, proceed to catalogue agent
  "retry"     — quality failed, retry count below max, loop back to transformer
  "failed"    — quality failed, max retries exhausted, end pipeline

The router never modifies state — it only reads.
"""

import os

from orchestrator.state import PipelineState


def route_after_quality(state: PipelineState) -> str:
    max_retries = int(os.getenv("MAX_QUALITY_RETRIES", "3"))

    if state.get("status") == "success":
        return "catalogue"

    if state.get("status") == "retrying":
        if state.get("retry_count", 0) >= max_retries:
            return "failed"
        return "retry"

    # status == "failed" or any unexpected value
    return "failed"
