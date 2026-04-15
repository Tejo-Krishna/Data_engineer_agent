"""
Quality agent — ReAct loop, max 6 tool calls.

Expected tool call sequence (adaptive, not fixed):
  1. run_quality_checks       (always first)
  2. detect_anomalies         (only on failing/borderline columns)
  3. explain_anomalies        (ALWAYS if anomaly_count > 0)
  4. write_quality_report
  5. save_to_library          (ONLY if quality_passed=True AND reusable)

Status logic (set by agent, read by router):
  "success"  — all critical checks passed
  "retrying" — critical checks failed; retry_count incremented
  "failed"   — should not happen here; router handles max retries

Writes to state: quality_checks, anomaly_summary, anomaly_explanations,
quality_passed, quality_report_path, status, retry_count, failure_reason.
"""

import json
import os
from pathlib import Path

from langgraph.types import RunnableConfig

from agents.utils import (
    call_llm_with_tools,
    build_tool_results_message,
    extract_tool_calls,
)
from orchestrator.state import PipelineState


QUALITY_SYSTEM_PROMPT = """You are a data quality analyst.

You have access to tools for checking quality rules, detecting anomalies,
explaining anomalies in plain English, writing quality reports, and saving
successful transformations to the library.

Reason about which checks are most important for THIS dataset before calling
any tools. Look at the profile — columns with high null rates or wide value
ranges deserve targeted attention.

Rules:
- Call run_quality_checks first, always
- Call detect_anomalies only on columns that failed or are borderline
  (pass target_columns to scope the check)
- If anomaly_count > 0, you MUST call explain_anomalies
- Call write_quality_report after all checks and explanations are done
- Call save_to_library ONLY if quality_passed is True AND the transformation
  is generic enough to be reusable in other pipelines
- Do not call save_to_library if quality_passed is False under any circumstance"""

MAX_CALLS = 6


async def run_quality_agent(
    state: PipelineState,
    config: RunnableConfig,
) -> PipelineState:
    mcp = config["configurable"]["mcp"]
    logger = config["configurable"]["logger"]
    span = logger.agent_start("quality")

    # If the transformer did not produce an output file, quality checks cannot
    # run. Return immediately without incrementing retry_count — the transformer
    # already incremented it when execute_code failed.
    if not state.get("output_path"):
        logger.agent_end(span)
        return {
            **state,
            "quality_checks": [],
            "quality_passed": False,
            "status": "retrying",
            "failure_reason": state.get("failure_reason") or "Transformer did not produce output",
        }

    mcp_tools = await mcp.list_tools()
    available = [
        "run_quality_checks",
        "detect_anomalies",
        "explain_anomalies",
        "write_quality_report",
        "save_to_library",
    ]

    messages = [
        {"role": "user", "content": _build_context(state)},
    ]

    # Track key outputs as the loop runs
    quality_checks: list = []
    anomaly_summary: dict = {}
    anomaly_explanations: list = []
    quality_report_path: str | None = None
    quality_passed: bool = False

    tool_call_count = 0

    while tool_call_count < MAX_CALLS:
        response = await call_llm_with_tools(messages, available, mcp_tools)

        if response.stop_reason == "end_turn":
            break

        tool_calls = extract_tool_calls(response)
        if not tool_calls:
            break

        messages.append({"role": "assistant", "content": response.content})

        results_for_turn: list[tuple[str, dict]] = []

        for tc in tool_calls:
            # Guard: never let LLM call save_to_library when quality failed
            if tc["name"] == "save_to_library" and not quality_passed:
                fake_result = {
                    "saved": False,
                    "library_id": None,
                    "reason": (
                        "save_to_library is blocked because quality_passed is False. "
                        "Do NOT retry this call. Proceed to write_quality_report instead."
                    ),
                }
                logger.tool_call(
                    span, "quality", "save_to_library_blocked",
                    {},
                    fake_result,
                )
                results_for_turn.append((tc["id"], fake_result))
                tool_call_count += 1
                continue

            # Inject authoritative values the LLM tends to get wrong.
            if tc["name"] == "write_quality_report":
                tc["input"]["pipeline_run_id"] = state["run_id"]
                tc["input"]["output_dir"] = str(
                    Path(os.getenv("OUTPUT_DIR", "outputs")) / state["run_id"]
                )
            if tc["name"] == "run_quality_checks":
                tc["input"]["output_path"] = state.get("output_path") or tc["input"].get("output_path", "")
                tc["input"]["original_profile"] = state.get("profile") or {}
                tc["input"]["schema"] = state.get("schema") or {}
                tc["input"]["transformations_applied"] = state.get("transformations_applied") or []
                if state.get("domain_context"):
                    tc["input"]["domain_context"] = state["domain_context"]
            if tc["name"] == "detect_anomalies":
                tc["input"]["output_path"] = state.get("output_path") or tc["input"].get("output_path", "")
                tc["input"]["schema"] = state.get("schema") or {}
            result = await mcp.call(tc["name"], tc["input"])
            logger.tool_call(
                span, "quality", tc["name"],
                _summarise_input(tc["name"], tc["input"]),
                _summarise_output(tc["name"], result),
            )

            if tc["name"] == "run_quality_checks":
                quality_checks = result.get("checks", [])
                quality_passed = result.get("overall_passed", False)

            if tc["name"] == "detect_anomalies":
                anomaly_summary = {
                    "anomaly_count": result.get("anomaly_count", 0),
                    "anomaly_rate": result.get("anomaly_rate", 0.0),
                    "anomalous_rows": result.get("anomalous_rows", [])[:20],
                }

            if tc["name"] == "explain_anomalies":
                anomaly_explanations = result.get("explanations", [])

            if tc["name"] == "write_quality_report":
                quality_report_path = result.get("json_path")

            results_for_turn.append((tc["id"], result))
            tool_call_count += 1

        messages.append(build_tool_results_message(results_for_turn))

    # ------------------------------------------------------------------
    # Set final status
    # ------------------------------------------------------------------
    if quality_passed:
        status = "success"
        failure_reason = None
        new_retry_count = state["retry_count"]
    else:
        status = "retrying"
        new_retry_count = state["retry_count"] + 1
        failing = [
            c for c in quality_checks
            if not c.get("passed") and c.get("severity") == "critical"
        ]
        failure_reason = "; ".join(
            f"{c['check_name']} ({c.get('column_name', 'table')})"
            for c in failing[:3]
        ) or "Quality checks failed"

    logger.tool_call(
        span, "quality", "quality_result",
        {},
        {
            "quality_passed": quality_passed,
            "status": status,
            "retry_count": new_retry_count,
            "failing_checks": len([c for c in quality_checks if not c.get("passed")]),
        },
    )

    logger.agent_end(span)

    return {
        **state,
        "quality_checks": quality_checks,
        "anomaly_summary": anomaly_summary,
        "anomaly_explanations": anomaly_explanations,
        "quality_passed": quality_passed,
        "quality_report_path": quality_report_path,
        "status": status,
        "retry_count": new_retry_count,
        "failure_reason": failure_reason,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_range(max_val, min_val) -> float:
    try:
        return float(max_val) - float(min_val)
    except (TypeError, ValueError):
        return 0.0


def _build_context(state: PipelineState) -> str:
    profile: dict = state.get("profile") or {}
    schema: dict = state.get("schema") or {}

    # Identify high-risk columns from profile
    high_null = [
        col for col, stats in profile.items()
        if isinstance(stats, dict) and (stats.get("null_rate") or 0) > 0.10
    ]
    wide_range = [
        col for col, stats in profile.items()
        if isinstance(stats, dict)
        and stats.get("mean") is not None and stats.get("max") is not None
        and stats.get("min") is not None
        and _safe_range(stats["max"], stats["min"]) > 1000
    ]

    parts = [
        f"Output dataset path: {state.get('output_path', 'N/A')}",
        f"Rows input: {state.get('rows_input', 0)}",
        f"Rows output: {state.get('rows_output', 0)}",
        f"Transformations applied: {json.dumps(state.get('transformations_applied', []))}",
        f"Domain: {state.get('domain', 'unknown')}",
        f"Schema ({min(len(schema), 40)} of {len(schema)} columns): "
        f"{json.dumps({col: v.get('inferred_type') for col, v in list(schema.items())[:40]}, indent=2)}",
    ]

    if high_null:
        parts.append(f"High-null columns (>10% nulls): {high_null}")
    if wide_range:
        parts.append(f"Wide-range numeric columns: {wide_range}")

    domain_ctx = state.get("domain_context") or {}
    if domain_ctx.get("validation_rules"):
        parts.append(
            f"Domain validation rules: {json.dumps(domain_ctx['validation_rules'], indent=2)}"
        )

    parts.append(
        "\nInstructions:\n"
        "1. Call run_quality_checks first\n"
        "2. If any column fails, call detect_anomalies with those column names\n"
        "3. If anomaly_count > 0, call explain_anomalies\n"
        "4. Call write_quality_report\n"
        "5. If quality_passed=True AND the code is generic/reusable, call save_to_library"
    )

    return "\n\n".join(parts)


def _summarise_input(tool_name: str, args: dict) -> dict:
    if tool_name == "run_quality_checks":
        return {"output_path": str(args.get("output_path", ""))[-40:]}
    if tool_name == "detect_anomalies":
        return {"target_columns": args.get("target_columns", [])}
    return {k: str(v)[:60] for k, v in list(args.items())[:2]}


def _summarise_output(tool_name: str, result: dict) -> dict:
    if tool_name == "run_quality_checks":
        checks = result.get("checks", [])
        return {
            "overall_passed": result.get("overall_passed"),
            "total": len(checks),
            "failed": len([c for c in checks if not c.get("passed")]),
        }
    if tool_name == "detect_anomalies":
        return {
            "anomaly_count": result.get("anomaly_count", 0),
            "anomaly_rate": result.get("anomaly_rate", 0.0),
        }
    if tool_name == "explain_anomalies":
        return {"columns_explained": len(result.get("explanations", []))}
    if tool_name == "write_quality_report":
        return {"json_path": result.get("json_path"), "status": result.get("overall_status")}
    return {k: str(v)[:60] for k, v in list(result.items())[:2]}
