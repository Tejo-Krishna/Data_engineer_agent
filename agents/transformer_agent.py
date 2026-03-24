"""
Transformer agent — ReAct loop, max 6 tool calls.

Expected tool call sequence:
  1. detect_new_rows          (only when incremental_mode=True)
  2. search_transform_library
  3. generate_transform_code
     └─ HITL checkpoint fires here (first run only, outside the LLM loop)
        ├─ approved, no instruction → continue to execute_code
        ├─ needs_refinement → refine_transform_code → post diff → confirm
        └─ rejected → status="failed"
  4. execute_code
     └─ if fails → status="retrying", failure_reason set
  5. write_dataset
  6. (save pipeline.py to outputs/{run_id}/pipeline.py)

Domain rules (domain_context) are HARD CONSTRAINTS fed into code generation.
On retry (retry_count > 0): HITL does NOT fire; failure_reason injected into prompt.

Writes to state: library_snippets, watermark_value, generated_code,
transformations_applied, output_columns, output_path, rows_input,
rows_output, pipeline_script, hitl_approved, status, failure_reason, retry_count.
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
from hitl.checkpoint import (
    hitl_code_checkpoint,
    hitl_post_diff,
    hitl_confirm_after_refinement,
)
from observability.tracing import log_hitl_nlp_instruction
from orchestrator.state import PipelineState


TRANSFORMER_SYSTEM_PROMPT = """You are a data engineering code generator.

You have access to tools for searching reusable code, generating Python
transformation scripts, refining code based on instructions, executing
code in a sandbox, and writing the output dataset.

Domain context for this dataset will be provided. Domain rules are
HARD CONSTRAINTS — they override the user's goal if they conflict.
Never generate code that violates domain rules even if the user asks.

Loop:
1. If incremental_mode is true, call detect_new_rows first
2. Search the transformation library for reusable code
3. Generate the transformation script — use library snippets if relevant
4. After the HITL approval result is injected into context, call execute_code
   using the approved code path provided in the HITL result
5. If execution fails, reason about the error and call generate_transform_code again
   with the failure context (do not call execute_code again before regenerating)
6. Call write_dataset when execution succeeds

Generated code must:
- Read from INPUT_PATH env var, write to OUTPUT_PATH env var
- Print ROWS_IN: N before processing
- Print ROWS_OUT: N after processing
- Handle all null values gracefully — never crash on nulls
- Be a complete self-contained script with all imports at the top"""

MAX_CALLS = 6


async def run_transformer_agent(
    state: PipelineState,
    config: RunnableConfig,
) -> PipelineState:
    mcp = config["configurable"]["mcp"]
    logger = config["configurable"]["logger"]
    span = logger.agent_start("transformer")

    run_id = state["run_id"]
    mcp_tools = await mcp.list_tools()

    # Available tools for this agent
    available = [
        "search_transform_library",
        "generate_transform_code",
        "refine_transform_code",
        "execute_code",
        "write_dataset",
    ]
    if state.get("incremental_mode"):
        available.insert(0, "detect_new_rows")

    # Build initial context message
    messages = [
        {"role": "user", "content": _build_context(state)},
    ]

    # Mutable state we track as the loop runs
    generated_code: str = state.get("generated_code", "") or ""
    transformations_applied: list = state.get("transformations_applied", []) or []
    output_columns: list = state.get("output_columns", []) or []
    watermark_value: str | None = state.get("watermark_value")
    library_snippets: list = state.get("library_snippets", []) or []
    rows_input: int = 0
    rows_output: int = 0
    output_path: str = ""
    hitl_approved: bool = False

    tool_call_count = 0

    while tool_call_count < MAX_CALLS:
        response = await call_llm_with_tools(messages, available, mcp_tools)

        if response.stop_reason == "end_turn":
            break

        tool_calls = extract_tool_calls(response)
        if not tool_calls:
            break

        # Append assistant message (contains tool_use blocks)
        messages.append({"role": "assistant", "content": response.content})

        # Collect all tool results for this turn
        results_for_turn: list[tuple[str, dict]] = []
        hitl_injection: str | None = None

        for tc in tool_calls:
            print(f"  [transformer] calling {tc['name']} ...", flush=True)
            # Inject structured args directly from state so the LLM never
            # reconstructs nested dicts/lists from context with the wrong shape.
            if tc["name"] == "generate_transform_code":
                if library_snippets:
                    tc["input"]["library_snippets"] = library_snippets
                tc["input"]["schema"] = state["schema"]
                tc["input"]["profile"] = state["profile"]
            if tc["name"] == "write_dataset":
                tc["input"]["run_id"] = run_id
            result = await mcp.call(tc["name"], tc["input"])
            print(f"  [transformer] {tc['name']} done", flush=True)
            logger.tool_call(
                span, "transformer", tc["name"],
                _summarise_input(tc["name"], tc["input"]),
                _summarise_output(tc["name"], result),
            )

            # ----------------------------------------------------------
            # Track tool-specific outputs
            # ----------------------------------------------------------
            if tc["name"] == "detect_new_rows":
                watermark_value = result.get("watermark_value")
                logger.tool_call(
                    span, "transformer", "detect_new_rows_watermark",
                    {"previous": tc["input"].get("previous_watermark")},
                    {"new_rows": result.get("new_row_count", 0),
                     "watermark": watermark_value},
                )

            if tc["name"] == "search_transform_library":
                library_snippets = result.get("snippets", [])

            if tc["name"] == "generate_transform_code":
                generated_code = result.get("code", "")
                transformations_applied = result.get("transformations_applied", [])
                output_columns = result.get("output_columns", [])

                # --------------------------------------------------
                # HITL checkpoint — fires only on first run
                # --------------------------------------------------
                if state["retry_count"] == 0:
                    print(f"  [transformer] HITL checkpoint — run: python scripts/hitl_approve.py {run_id}", flush=True)
                    hitl_result = await _run_hitl(
                        run_id, state, result, mcp, logger, span
                    )

                    if hitl_result is None:
                        # Rejected
                        logger.agent_end(span)
                        return {
                            **state,
                            "status": "failed",
                            "failure_reason": "HITL checkpoint rejected the generated code",
                            "generated_code": generated_code,
                            "transformations_applied": transformations_applied,
                            "library_snippets": library_snippets,
                        }

                    hitl_approved = hitl_result["approved"]
                    # Use refined code if refinement occurred
                    if hitl_result.get("revised_code"):
                        generated_code = hitl_result["revised_code"]
                        result = {**result, "code": generated_code}

                    hitl_injection = (
                        f"HITL checkpoint result: {json.dumps(hitl_result)}\n\n"
                        "The code has been approved. Call execute_code now with "
                        f"the approved code. Input path: {_get_input_path(state)}"
                    )
                else:
                    # Retry run — no HITL, inject failure context already in prompt
                    hitl_approved = True
                    hitl_injection = (
                        "Retry run — HITL skipped. Call execute_code with the "
                        f"regenerated code. Input path: {_get_input_path(state)}"
                    )

            if tc["name"] == "execute_code":
                if not result.get("success"):
                    # Execution failed — signal retrying
                    results_for_turn.append((tc["id"], result))
                    messages.append(build_tool_results_message(results_for_turn))
                    logger.agent_end(span)
                    return {
                        **state,
                        "status": "retrying",
                        "failure_reason": f"Execution error: {result.get('stderr', '')[:500]}",
                        "retry_count": state["retry_count"] + 1,
                        "generated_code": generated_code,
                        "transformations_applied": transformations_applied,
                        "output_columns": output_columns,
                        "library_snippets": library_snippets,
                        "watermark_value": watermark_value,
                        "hitl_approved": hitl_approved,
                    }
                rows_input = result.get("rows_input", 0)
                rows_output = result.get("rows_output", 0)

            if tc["name"] == "write_dataset":
                output_path = result.get("output_path", "")
                if not output_path:
                    logger.agent_end(span)
                    return {
                        **state,
                        "status": "retrying",
                        "failure_reason": "write_dataset returned empty output_path",
                        "retry_count": state["retry_count"] + 1,
                    }

            results_for_turn.append((tc["id"], result))
            tool_call_count += 1

        # Append all tool results as a single user message
        messages.append(build_tool_results_message(results_for_turn))

        # Append HITL injection as a separate user message if needed
        if hitl_injection:
            messages.append({"role": "user", "content": hitl_injection})
            hitl_injection = None

    # ------------------------------------------------------------------
    # Save pipeline script
    # ------------------------------------------------------------------
    pipeline_script: str | None = None
    if generated_code and run_id:
        pipeline_script = _save_pipeline_script(run_id, generated_code)

    logger.agent_end(span)

    return {
        **state,
        "library_snippets": library_snippets,
        "watermark_value": watermark_value,
        "generated_code": generated_code,
        "transformations_applied": transformations_applied,
        "output_columns": output_columns,
        "output_path": output_path,
        "rows_input": rows_input,
        "rows_output": rows_output,
        "pipeline_script": pipeline_script,
        "hitl_approved": hitl_approved,
        "status": "running",  # quality agent will set final status
    }


# ---------------------------------------------------------------------------
# HITL helper
# ---------------------------------------------------------------------------


async def _run_hitl(
    run_id: str,
    state: PipelineState,
    codegen_result: dict,
    mcp,
    logger,
    span,
) -> dict | None:
    """
    Run the full HITL code review flow.

    Returns a dict with approved=True and optional revised_code on success.
    Returns None if rejected.
    """
    try:
        hitl_result = await hitl_code_checkpoint(
            run_id=run_id,
            code=codegen_result["code"],
            transformations=codegen_result.get("transformations_applied", []),
            source_path=state["source_path"],
            domain=state.get("domain", "unknown"),
            domain_context=state.get("domain_context", {}),
        )
    except ValueError:
        return None

    logger.tool_call(
        span, "transformer", "hitl_code_checkpoint",
        {},
        {
            "approved": hitl_result["approved"],
            "needs_refinement": hitl_result.get("needs_refinement", False),
        },
    )

    if not hitl_result["approved"]:
        return None

    if hitl_result.get("needs_refinement"):
        instruction = hitl_result["nlp_instruction"]
        log_hitl_nlp_instruction(run_id, instruction)

        # Agent (not LLM) calls refine directly
        refined = await mcp.call("refine_transform_code", {
            "original_code": codegen_result["code"],
            "nlp_instruction": instruction,
            "profile": state.get("profile", {}),
            "domain_context": state.get("domain_context", {}),
        })

        logger.tool_call(
            span, "transformer", "refine_transform_code",
            {"instruction": instruction[:120]},
            {"changes": len(refined.get("changes_summary", []))},
        )

        # Post diff to Redis so human can review it
        diff_summary = "\n".join(refined.get("changes_summary", []))
        await hitl_post_diff(run_id, diff_summary, refined["revised_code"])

        # Poll for human confirmation
        confirm = await hitl_confirm_after_refinement(run_id)

        logger.tool_call(
            span, "transformer", "hitl_confirm_after_refinement",
            {},
            {"confirmed": confirm["confirmed"]},
        )

        if confirm["confirmed"]:
            return {
                "approved": True,
                "revised_code": confirm["code"],
                "nlp_instruction": instruction,
                "needs_refinement": False,
            }
        else:
            # Human looped back — treat as failed for this attempt
            return None

    # Simple approve — no refinement
    return {
        "approved": True,
        "revised_code": None,
        "nlp_instruction": None,
        "needs_refinement": False,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_context(state: PipelineState) -> str:
    goal = state["user_goal"]
    if state["retry_count"] > 0 and state.get("failure_reason"):
        goal += (
            f"\n\nPREVIOUS ATTEMPT FAILED:\n{state['failure_reason']}\n"
            "Fix this specific issue. Do not change parts of the code that worked."
        )

    schema_summary = {
        col: v.get("inferred_type", "unknown")
        for col, v in (state.get("schema") or {}).items()
    }

    parts = [
        f"User goal: {goal}",
        f"Source path: {state['source_path']}",
        f"Source type: {state['source_type']}",
        f"Incremental mode: {state.get('incremental_mode', False)}",
        f"Schema: {json.dumps(schema_summary, indent=2)}",
    ]

    domain_ctx = state.get("domain_context") or {}
    if domain_ctx:
        parts.append(f"Domain: {state.get('domain', 'unknown')}")
        parts.append(
            f"Domain rules (HARD CONSTRAINTS):\n"
            f"  required_transforms: {domain_ctx.get('required_transforms', [])}\n"
            f"  forbidden_transforms: {domain_ctx.get('forbidden_transforms', [])}\n"
            f"  sensitive_columns: {domain_ctx.get('sensitive_columns', [])}"
        )

    snippets = state.get("library_snippets") or []
    if snippets:
        parts.append(
            f"Reusable library snippets ({len(snippets)} found — incorporate if relevant):"
        )
        for s in snippets[:3]:
            parts.append(f"  - {s.get('name')}: {s.get('description')}")

    if state.get("incremental_mode"):
        parts.append(
            "IMPORTANT: incremental_mode is True. "
            "Your FIRST tool call MUST be detect_new_rows to filter the source "
            "to only new rows. Use the filtered_path it returns as the input for execute_code."
        )
        if state.get("watermark_value"):
            parts.append(f"Previous watermark: {state['watermark_value']}")

    return "\n\n".join(parts)


def _get_input_path(state: PipelineState) -> str:
    """Return the absolute data path to pass to execute_code."""
    path = state.get("source_path", "")
    return str(Path(path).resolve())


def _save_pipeline_script(run_id: str, code: str) -> str:
    """Save generated code as a rerunnable pipeline.py."""
    output_dir = Path(os.getenv("OUTPUT_DIR", "outputs")) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / "pipeline.py"
    script_path.write_text(code, encoding="utf-8")
    return str(script_path)


def _summarise_input(tool_name: str, args: dict) -> dict:
    """Return a compact summary of tool input for logging."""
    if tool_name == "generate_transform_code":
        return {"goal_len": len(str(args.get("user_goal", "")))}
    if tool_name == "execute_code":
        return {"code_lines": len(str(args.get("code", "")).splitlines())}
    return {k: str(v)[:80] for k, v in list(args.items())[:3]}


def _summarise_output(tool_name: str, result: dict) -> dict:
    """Return a compact summary of tool output for logging."""
    if tool_name == "generate_transform_code":
        return {
            "code_lines": len(result.get("code", "").splitlines()),
            "transforms": len(result.get("transformations_applied", [])),
        }
    if tool_name == "execute_code":
        return {
            "success": result.get("success"),
            "rows_in": result.get("rows_input"),
            "rows_out": result.get("rows_output"),
        }
    if tool_name == "write_dataset":
        return {
            "output_path": result.get("output_path"),
            "rows": result.get("row_count"),
        }
    return {k: str(v)[:80] for k, v in list(result.items())[:3]}
