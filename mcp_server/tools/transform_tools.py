"""
Transformation tools.

Generate, refine, execute, and persist transformation code.
execute_code delegates to sandbox/executor.py — never call it
before the HITL checkpoint has approved.
"""

import asyncio
import json
import os
from pathlib import Path

import anthropic
import duckdb

from db import get_duckdb_conn
from sandbox.executor import run_sandboxed


# Sonnet only for code generation — failures here trigger expensive retries
_CODEGEN_MODEL = lambda: os.getenv("CODEGEN_MODEL", "claude-sonnet-4-5")
# Haiku for targeted edits and semantic verification
_MODEL = lambda: os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
_VERIFY_MODEL = "claude-haiku-4-5-20251001"
# Files larger than this threshold get DuckDB SQL mode instead of pandas
_LARGE_FILE_THRESHOLD_MB: float = float(os.getenv("LARGE_FILE_THRESHOLD_MB", "500"))


def _check_file_size(path: str) -> float:
    """Return file size in MB, or 0.0 if the file cannot be stat'd."""
    try:
        return Path(path).stat().st_size / (1024 * 1024)
    except OSError:
        return 0.0

# ---------------------------------------------------------------------------
# Tool schemas — force structured output via tool_choice, no text parsing
# ---------------------------------------------------------------------------

_GENERATE_TRANSFORM_TOOL = {
    "name": "submit_transform_code",
    "description": "Submit the generated Python transformation script and metadata.",
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Complete, self-contained Python script that reads INPUT_PATH and writes to OUTPUT_PATH.",
            },
            "transformations_applied": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Short names of each transformation applied, e.g. ['dedup', 'parse_dates', 'gbp_to_usd'].",
            },
            "output_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Column names present in the output Parquet file.",
            },
        },
        "required": ["code", "transformations_applied", "output_columns"],
    },
}

_REFINE_TRANSFORM_TOOL = {
    "name": "submit_refined_code",
    "description": "Submit the revised Python transformation script after applying the requested change.",
    "input_schema": {
        "type": "object",
        "properties": {
            "revised_code": {
                "type": "string",
                "description": "Updated Python script with only the requested change applied.",
            },
            "changes_summary": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Brief description of each change made.",
            },
        },
        "required": ["revised_code", "changes_summary"],
    },
}


# ---------------------------------------------------------------------------
# generate_transform_code
# ---------------------------------------------------------------------------

async def generate_transform_code(
    user_goal: str,
    profile: dict,
    schema: dict,
    domain_context: dict,
    library_snippets: list | None = None,
    failure_reason: str | None = None,
    large_file: bool = False,
) -> dict:
    """
    Generate a Python transformation script for the given dataset and goal.

    Use when: you have a profile, schema, and domain_context ready and need
    executable code to clean or transform the dataset.
    Do NOT use when: the code only needs minor adjustments based on user NLP
    feedback — use refine_transform_code instead.
    Returns: code (str), transformations_applied (list[str]),
             output_columns (list[str]).
    """
    client = anthropic.AsyncAnthropic()

    forbidden = domain_context.get("forbidden_transforms", [])
    required = domain_context.get("required_transforms", [])
    validation_rules = domain_context.get("validation_rules", {})

    snippet_block = ""
    if library_snippets:
        snippet_block = "\n\nRelevant library snippets (use if applicable):\n"
        for s in library_snippets:
            if isinstance(s, str):
                try:
                    s = json.loads(s)
                except (json.JSONDecodeError, TypeError):
                    continue
            snippet_block += f"\n--- {s['name']} ---\n{s['code']}\n"

    failure_block = ""
    if failure_reason:
        failure_block = f"\n\nPREVIOUS ATTEMPT FAILED. Fix ONLY this issue:\n{failure_reason}\n"

    # Summarise schema for prompt
    schema_summary = {
        col: {
            "inferred_type": v.get("inferred_type"),
            "needs_cast": v.get("needs_cast"),
            "suggested_cast": v.get("suggested_cast"),
            "detected_formats": v.get("detected_formats"),
        }
        for col, v in schema.items()
    }

    # Summarise profile (keep it concise)
    profile_summary = {
        col: {
            "dtype": v.get("dtype"),
            "null_rate": v.get("null_rate"),
            "top_5_values": v.get("top_5_values", [])[:3],
        }
        for col, v in profile.items()
        if col not in ("total_rows", "duplicate_row_count")
    }

    if large_file:
        requirements_block = """REQUIREMENTS FOR THE SCRIPT (LARGE FILE — DuckDB streaming mode):
1. Use DuckDB in-process — do NOT use pandas. The dataset is too large to fit in memory.
2. Read from os.environ["INPUT_PATH"] using DuckDB: conn.execute("SELECT * FROM read_parquet(?)", [input_path]) or read_csv_auto depending on extension.
3. Write output to os.environ["OUTPUT_PATH"] using: conn.execute("COPY (SELECT ...) TO ? (FORMAT PARQUET)", [output_path])
4. Print exactly "ROWS_IN: N" before any processing — get count with conn.execute("SELECT COUNT(*) FROM read_parquet(?) / read_csv_auto(?)", [input_path]).fetchone()[0]
5. Print exactly "ROWS_OUT: N" after writing — get count from output file the same way
6. Import only: os, duckdb — nothing else
7. Handle nulls gracefully — use COALESCE / NULLIF in SQL, or TRY_CAST for type conversions
8. Do NOT apply any forbidden transforms listed above
9. DO apply all required transforms listed above
10. DuckDB SQL equivalents: dedup → SELECT DISTINCT; date parse → TRY_CAST(col AS DATE); currency strip → REGEXP_REPLACE(col, '[^0-9.]', '')"""
    else:
        requirements_block = """REQUIREMENTS FOR THE SCRIPT:
1. Read input from os.environ["INPUT_PATH"] — use pandas to read CSV or Parquet by file extension
2. Write output to os.environ["OUTPUT_PATH"] as Parquet
3. Print exactly "ROWS_IN: N" before any processing (N = len of input dataframe)
4. Print exactly "ROWS_OUT: N" after writing (N = len of output dataframe)
5. Import only: os, pandas, pyarrow, datetime, re, numpy — nothing else
6. Handle nulls gracefully — do not crash on missing values
7. Do NOT apply any forbidden transforms listed above
8. DO apply all required transforms listed above"""

    mode_note = "\n⚠️  LARGE FILE MODE: Use DuckDB SQL — do NOT use pandas." if large_file else ""

    prompt = f"""You are a data engineering expert. Write a Python script to transform the dataset.{mode_note}

USER GOAL: {user_goal}

SCHEMA (per-column inferred types and cast requirements):
{json.dumps(schema_summary, indent=2)}

PROFILE SUMMARY (per-column statistics):
{json.dumps(profile_summary, indent=2)}

DOMAIN: {domain_context.get('domain', 'unknown')}
REQUIRED TRANSFORMS (you MUST apply all of these): {required}
FORBIDDEN TRANSFORMS (you MUST NOT apply any of these): {forbidden}
VALIDATION RULES: {json.dumps(validation_rules, indent=2)}
{snippet_block}
{failure_block}

{requirements_block}

Use the submit_transform_code tool to return the script and metadata."""

    message = await client.messages.create(
        model=_CODEGEN_MODEL(),
        max_tokens=4096,
        tools=[_GENERATE_TRANSFORM_TOOL],
        tool_choice={"type": "tool", "name": "submit_transform_code"},
        messages=[{"role": "user", "content": prompt}],
        timeout=90.0,
    )

    result = message.content[0].input
    return {
        "code": result["code"],
        "transformations_applied": result.get("transformations_applied", []),
        "output_columns": result.get("output_columns", []),
    }


# ---------------------------------------------------------------------------
# refine_transform_code
# ---------------------------------------------------------------------------

async def refine_transform_code(
    original_code: str,
    nlp_instruction: str,
    profile: dict,
    domain_context: dict,
) -> dict:
    """
    Apply a plain-English edit instruction to an existing transformation script.

    Use when: the HITL checkpoint returned an NLP instruction (Path B) and
    you need to revise the code without starting from scratch.
    Do NOT use when: the code has a runtime error — use generate_transform_code
    with failure_reason set instead.
    Returns: revised_code (str), changes_summary (list[str]).
    """
    client = anthropic.AsyncAnthropic()

    forbidden = domain_context.get("forbidden_transforms", [])

    prompt = f"""You are a data engineering expert. Revise the Python script below based on the instruction.

INSTRUCTION FROM HUMAN REVIEWER: {nlp_instruction}

FORBIDDEN TRANSFORMS (must not be introduced): {forbidden}

ORIGINAL CODE:
```python
{original_code}
```

Apply ONLY the requested change. Do not refactor unrelated code.
Use the submit_refined_code tool to return the updated script."""

    message = await client.messages.create(
        model=_MODEL(),
        max_tokens=4096,
        tools=[_REFINE_TRANSFORM_TOOL],
        tool_choice={"type": "tool", "name": "submit_refined_code"},
        messages=[{"role": "user", "content": prompt}],
    )

    result = message.content[0].input
    return {
        "revised_code": result["revised_code"],
        "changes_summary": result.get("changes_summary", []),
    }


# ---------------------------------------------------------------------------
# execute_code
# ---------------------------------------------------------------------------

async def execute_code(
    code: str,
    input_path: str,
    output_path: str,
    timeout_seconds: int = 30,
) -> dict:
    """
    Run a transformation script in a sandboxed subprocess.

    Use when: the HITL checkpoint has approved the code and you are ready to
    execute it against the real input data.
    Do NOT call before the HITL checkpoint has approved — executing unapproved
    code is a hard rule violation.
    Returns: success (bool), stdout (str), stderr (str),
             rows_input (int | None), rows_output (int | None),
             execution_time_ms (int).
    """
    return await asyncio.to_thread(
        run_sandboxed,
        code=code,
        input_path=input_path,
        output_path=output_path,
        timeout=timeout_seconds,
    )


# ---------------------------------------------------------------------------
# write_dataset
# ---------------------------------------------------------------------------

async def write_dataset(
    data_path: str,
    output_name: str,
    run_id: str,
) -> dict:
    """
    Move the sandbox output file to the permanent per-run output directory.

    Use when: execute_code has succeeded and you need to persist the output
    Parquet file to outputs/{run_id}/.
    Do NOT use when: execute_code failed — check success before calling this.
    Returns: output_path (str), row_count (int), file_size_mb (float),
             columns (list[str]), schema (dict[str, str]).
    """
    # Write to staging/ — main.py calls finalize_run() to rename staging→final
    # after the catalogue agent completes successfully.
    output_dir = Path(os.getenv("OUTPUT_DIR", "outputs")) / run_id / "staging"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"{output_name}.parquet"

    conn = get_duckdb_conn()
    try:
        conn.execute(
            f"COPY (SELECT * FROM read_parquet('{data_path}')) TO '{final_path}' (FORMAT PARQUET)"
        )
        row_count = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{final_path}')").fetchone()[0]
        col_info = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{final_path}')").fetchall()
        columns = [c[0] for c in col_info]
        schema = {c[0]: c[1] for c in col_info}
        file_size_mb = round(Path(final_path).stat().st_size / (1024 * 1024), 4)
    finally:
        conn.close()

    return {
        "output_path": str(final_path),
        "row_count": row_count,
        "file_size_mb": file_size_mb,
        "columns": columns,
        "schema": schema,
    }


# ---------------------------------------------------------------------------
# verify_transform_intent
# ---------------------------------------------------------------------------

_VERIFY_INTENT_TOOL = {
    "name": "submit_intent_verification",
    "description": "Submit the semantic intent verification result.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intent_matched": {
                "type": "boolean",
                "description": "True if the output data matches the user's stated goal.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence in the verdict (0-1).",
            },
            "issues": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specific mismatches found, empty if intent_matched=True.",
            },
        },
        "required": ["intent_matched", "confidence", "issues"],
    },
}


async def verify_transform_intent(
    user_goal: str,
    input_path: str,
    output_path: str,
    transformations_applied: list[str],
) -> dict:
    """
    Verify that the executed transformation semantically matches the user's goal.

    Use when: execute_code succeeded and you need lightweight semantic confirmation
    before writing the dataset to permanent storage.
    Do NOT use when: execute_code failed — only verify successful outputs.
    Returns: intent_matched (bool), confidence (float 0-1), issues (list[str]).
    """
    # Sample 5 rows from input and output — keeps the prompt tiny for Haiku
    conn = get_duckdb_conn()
    try:
        ext_in = Path(input_path).suffix.lower()
        if ext_in in (".parquet", ".pq"):
            in_rows = conn.execute(
                f"SELECT * FROM read_parquet('{input_path}') LIMIT 5"
            ).df().to_dict(orient="records")
        else:
            in_rows = conn.execute(
                f"SELECT * FROM read_csv_auto('{input_path}') LIMIT 5"
            ).df().to_dict(orient="records")

        out_rows = conn.execute(
            f"SELECT * FROM read_parquet('{output_path}') LIMIT 5"
        ).df().to_dict(orient="records")
    finally:
        conn.close()

    # Truncate values to keep token count low
    def _trim(rows: list[dict], max_val: int = 40) -> list[dict]:
        return [{k: str(v)[:max_val] for k, v in row.items()} for row in rows]

    prompt = f"""You are verifying a data transformation matches the user's intent.

USER GOAL: {user_goal}
TRANSFORMATIONS APPLIED: {transformations_applied}

INPUT SAMPLE (5 rows):
{json.dumps(_trim(in_rows), indent=2)}

OUTPUT SAMPLE (5 rows):
{json.dumps(_trim(out_rows), indent=2)}

Does the output match the goal? Use submit_intent_verification to respond."""

    client = anthropic.AsyncAnthropic()
    message = await client.messages.create(
        model=_VERIFY_MODEL,
        max_tokens=256,
        tools=[_VERIFY_INTENT_TOOL],
        tool_choice={"type": "tool", "name": "submit_intent_verification"},
        messages=[{"role": "user", "content": prompt}],
    )

    result = message.content[0].input
    return {
        "intent_matched": bool(result.get("intent_matched", True)),
        "confidence": round(float(result.get("confidence", 0.5)), 3),
        "issues": result.get("issues", []),
    }


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS: dict = {
    "generate_transform_code": generate_transform_code,
    "refine_transform_code": refine_transform_code,
    "execute_code": execute_code,
    "write_dataset": write_dataset,
    "verify_transform_intent": verify_transform_intent,
}
