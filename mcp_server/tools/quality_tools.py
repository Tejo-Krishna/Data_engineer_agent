"""
Data quality tools.

Runs targeted quality checks, detects statistical anomalies, generates
plain-English explanations via LLM, and writes JSON + Markdown reports.
All data reads use DuckDB — no pandas in these tools.
"""

import json
import os
import statistics
from pathlib import Path

import anthropic

from db import get_duckdb_conn, get_postgres_pool


_MODEL = lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")


# ---------------------------------------------------------------------------
# run_quality_checks
# ---------------------------------------------------------------------------

async def run_quality_checks(
    output_path: str,
    original_profile: dict,
    schema: dict,
    transformations_applied: list,
    domain_context: dict | None = None,
) -> dict:
    """
    Run a targeted suite of quality checks on the transformed dataset.

    Use when: execute_code has succeeded and you need to validate the output
    before deciding whether the run passes or needs a retry.
    Do NOT use when: the dataset has not been transformed yet — always run
    quality checks on the output, not the input.
    Returns: checks (list[dict] each with check_name, passed, severity,
             detail, column_name), overall_passed (bool),
             critical_failures (list[str]).
    """
    conn = get_duckdb_conn()
    checks: list[dict] = []

    try:
        conn.execute(f"CREATE OR REPLACE VIEW output AS SELECT * FROM read_parquet('{output_path}')")
        output_total = conn.execute("SELECT COUNT(*) FROM output").fetchone()[0]
        input_total = original_profile.get("total_rows", 0)

        # --- Check 1: row count ---
        deduped = any("dedup" in t.lower() for t in transformations_applied)
        min_expected = 0.30 if deduped else 0.50
        row_ratio = output_total / input_total if input_total else 1.0
        checks.append({
            "check_name": "row_count",
            "check_type": "volume",
            "column_name": None,
            "passed": row_ratio >= min_expected,
            "severity": "critical",
            "detail": (
                f"Output has {output_total} rows ({row_ratio:.1%} of input {input_total}). "
                f"Minimum expected: {min_expected:.0%}."
            ),
        })

        col_info = conn.execute("DESCRIBE output").fetchall()
        output_col_names = [c[0] for c in col_info]
        output_col_types = {c[0]: c[1] for c in col_info}

        for col in output_col_names:
            col_profile = original_profile.get(col, {})
            if not col_profile:
                continue

            # --- Check 2: null rate did not increase significantly ---
            original_null_rate = col_profile.get("null_rate", 0.0)
            current_null_count = conn.execute(
                f'SELECT COUNT(*) FROM output WHERE "{col}" IS NULL'
            ).fetchone()[0]
            current_null_rate = current_null_count / output_total if output_total else 0.0
            null_increase = current_null_rate - original_null_rate
            checks.append({
                "check_name": "null_rate",
                "check_type": "completeness",
                "column_name": col,
                "passed": null_increase <= 0.05,
                "severity": "error",
                "detail": (
                    f"Null rate: {original_null_rate:.1%} → {current_null_rate:.1%} "
                    f"(increase: {null_increase:+.1%})"
                ),
            })

            # --- Check 3: type conformance ---
            expected_type = schema.get(col, {}).get("inferred_type", "")
            actual_type = output_col_types.get(col, "")
            if expected_type:
                type_ok = _types_compatible(expected_type, actual_type)
                checks.append({
                    "check_name": "type_conformance",
                    "check_type": "schema",
                    "column_name": col,
                    "passed": type_ok,
                    "severity": "error",
                    "detail": f"Expected type '{expected_type}', actual '{actual_type}'.",
                })

            # --- Check 4: value range (numeric columns only) ---
            is_numeric = any(
                t in actual_type.upper()
                for t in ("INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "BIGINT")
            )
            if is_numeric and col_profile.get("mean") is not None:
                original_mean = float(col_profile["mean"])
                original_std_estimate = _estimate_std_from_profile(col_profile)
                row = conn.execute(
                    f'SELECT AVG("{col}") FROM output WHERE "{col}" IS NOT NULL'
                ).fetchone()
                current_mean = float(row[0]) if row[0] is not None else None
                if current_mean is not None and original_std_estimate > 0:
                    z = abs(current_mean - original_mean) / original_std_estimate
                    checks.append({
                        "check_name": "value_range",
                        "check_type": "distribution",
                        "column_name": col,
                        "passed": z <= 3.0,
                        "severity": "warning",
                        "detail": (
                            f"Mean shifted from {original_mean:.4f} to {current_mean:.4f} "
                            f"({z:.1f} std devs from original)."
                        ),
                    })

        # Domain-specific validation
        if domain_context:
            domain_checks = _run_domain_checks(conn, domain_context, output_col_names)
            checks.extend(domain_checks)

    finally:
        conn.close()

    overall_passed = all(c["passed"] for c in checks if c["severity"] == "critical")
    critical_failures = [
        c["check_name"] + (f"({c['column_name']})" if c["column_name"] else "")
        for c in checks
        if not c["passed"] and c["severity"] == "critical"
    ]

    return {
        "checks": checks,
        "overall_passed": overall_passed,
        "critical_failures": critical_failures,
    }


def _estimate_std_from_profile(col_profile: dict) -> float:
    """Rough std estimate from min/max when true std is unavailable."""
    try:
        mn = float(col_profile.get("min") or 0)
        mx = float(col_profile.get("max") or 0)
        return (mx - mn) / 4  # approximate 4-sigma range
    except (TypeError, ValueError):
        return 0.0


def _types_compatible(expected: str, actual: str) -> bool:
    expected_up = expected.upper()
    actual_up = actual.upper()
    numeric = {"INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "BIGINT", "HUGEINT"}
    if any(t in expected_up for t in numeric) and any(t in actual_up for t in numeric):
        return True
    if "DATE" in expected_up and "DATE" in actual_up:
        return True
    if "BOOL" in expected_up and "BOOL" in actual_up:
        return True
    return expected_up in actual_up or actual_up in expected_up


def _run_domain_checks(conn, domain_context: dict, col_names: list) -> list:
    """Run simple domain-specific checks from validation_rules."""
    checks = []
    rules = domain_context.get("validation_rules", {})

    # Non-negative price/amount check
    if "unit_price_non_negative" in rules or "amount_non_negative" in rules:
        for col in col_names:
            if any(kw in col.lower() for kw in ("price", "amount", "total", "fee")):
                try:
                    neg_count = conn.execute(
                        f'SELECT COUNT(*) FROM output WHERE "{col}" < 0'
                    ).fetchone()[0]
                    checks.append({
                        "check_name": "non_negative_values",
                        "check_type": "domain",
                        "column_name": col,
                        "passed": neg_count == 0,
                        "severity": "error",
                        "detail": f"Found {neg_count} negative values in '{col}'.",
                    })
                except Exception:
                    pass

    return checks


# ---------------------------------------------------------------------------
# detect_anomalies
# ---------------------------------------------------------------------------

async def detect_anomalies(
    output_path: str,
    schema: dict,
    target_columns: list | None = None,
) -> dict:
    """
    Detect statistical outliers and pattern violations in the output dataset.

    Use when: run_quality_checks has completed and you need to identify
    specific anomalous rows for the quality report.
    Do NOT use when: all quality checks passed cleanly — skip anomaly
    detection if overall_passed is True and critical_failures is empty.
    Returns: anomaly_count (int), anomalous_rows (list[dict] each with
             row_index, column, value, reason), anomaly_rate (float).
    """
    conn = get_duckdb_conn()
    anomalous_rows: list[dict] = []

    try:
        conn.execute(f"CREATE OR REPLACE VIEW output AS SELECT * FROM read_parquet('{output_path}')")
        total = conn.execute("SELECT COUNT(*) FROM output").fetchone()[0]
        if total == 0:
            return {"anomaly_count": 0, "anomalous_rows": [], "anomaly_rate": 0.0}

        col_info = conn.execute("DESCRIBE output").fetchall()
        all_cols = [c[0] for c in col_info]
        col_types = {c[0]: c[1] for c in col_info}

        cols_to_check = target_columns if target_columns else all_cols

        for col in cols_to_check:
            if col not in col_types:
                continue
            dtype = col_types[col].upper()
            is_numeric = any(t in dtype for t in ("INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "BIGINT"))

            if is_numeric:
                # IQR outlier detection
                stats = conn.execute(
                    f"""
                    SELECT
                        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY "{col}") AS q1,
                        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY "{col}") AS q3
                    FROM output
                    WHERE "{col}" IS NOT NULL
                    """
                ).fetchone()
                if stats and stats[0] is not None and stats[1] is not None:
                    q1, q3 = float(stats[0]), float(stats[1])
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outliers = conn.execute(
                        f"""
                        SELECT ROW_NUMBER() OVER () - 1 AS row_idx, "{col}"
                        FROM output
                        WHERE "{col}" IS NOT NULL AND ("{col}" < {lower} OR "{col}" > {upper})
                        LIMIT 50
                        """
                    ).fetchall()
                    for row_idx, val in outliers:
                        anomalous_rows.append({
                            "row_index": int(row_idx),
                            "column": col,
                            "value": str(val),
                            "reason": f"IQR outlier: value {val} outside [{lower:.2f}, {upper:.2f}]",
                        })

            elif "VARCHAR" in dtype or "TEXT" in dtype:
                # Pattern violation: find values deviating from the most common pattern
                top_patterns = conn.execute(
                    f"""
                    SELECT REGEXP_REPLACE("{col}", '[a-zA-Z]', 'A', 'g') AS pat,
                           COUNT(*) AS cnt
                    FROM output WHERE "{col}" IS NOT NULL
                    GROUP BY pat ORDER BY cnt DESC LIMIT 3
                    """
                ).fetchall()
                if top_patterns:
                    common_patterns = {r[0] for r in top_patterns}
                    rare = conn.execute(
                        f"""
                        SELECT ROW_NUMBER() OVER () - 1 AS row_idx, "{col}"
                        FROM output
                        WHERE "{col}" IS NOT NULL
                          AND REGEXP_REPLACE("{col}", '[a-zA-Z]', 'A', 'g') NOT IN ({','.join(f"'{p}'" for p in common_patterns)})
                        LIMIT 20
                        """
                    ).fetchall()
                    for row_idx, val in rare:
                        anomalous_rows.append({
                            "row_index": int(row_idx),
                            "column": col,
                            "value": str(val),
                            "reason": f"Unusual pattern for column '{col}'",
                        })

    finally:
        conn.close()

    anomaly_count = len(anomalous_rows)
    anomaly_rate = round(anomaly_count / total, 4) if total else 0.0

    return {
        "anomaly_count": anomaly_count,
        "anomalous_rows": anomalous_rows[:100],  # cap at 100 for prompt safety
        "anomaly_rate": anomaly_rate,
    }


# ---------------------------------------------------------------------------
# explain_anomalies
# ---------------------------------------------------------------------------

async def explain_anomalies(
    anomalous_rows: list,
    column_profile: dict,
    domain: str,
    domain_context: dict,
) -> dict:
    """
    Generate plain-English explanations for detected anomalies using LLM.

    Use when: detect_anomalies returned anomaly_count > 0. Always call this
    before write_quality_report when anomalies exist.
    Do NOT use when: anomaly_count is 0 — skip and go directly to
    write_quality_report.
    Returns: explanations (list[dict] each with column, anomaly_count,
             explanation, likely_cause, recommended_action),
             overall_summary (str).
    """
    if not anomalous_rows:
        return {"explanations": [], "overall_summary": "No anomalies detected."}

    client = anthropic.AsyncAnthropic()

    # Group anomalies by column
    by_col: dict[str, list] = {}
    for row in anomalous_rows:
        col = row["column"]
        by_col.setdefault(col, []).append(row)

    # Summarise profile for relevant columns only
    relevant_profile = {
        col: column_profile.get(col, {}) for col in by_col
    }

    prompt = f"""You are a data quality analyst. Explain the anomalies found in this dataset.

DOMAIN: {domain}
ANOMALIES BY COLUMN:
{json.dumps(by_col, indent=2)}

COLUMN PROFILES:
{json.dumps(relevant_profile, indent=2)}

DOMAIN CONTEXT:
{json.dumps(domain_context.get('validation_rules', {}), indent=2)}

For each column with anomalies, provide a concise explanation.

Return a JSON object:
{{
  "explanations": [
    {{
      "column": "<column name>",
      "anomaly_count": <int>,
      "explanation": "<what is wrong in plain English>",
      "likely_cause": "<most probable data quality root cause>",
      "recommended_action": "<what should be done to fix it>"
    }}
  ],
  "overall_summary": "<1-2 sentence summary of data quality health>"
}}"""

    message = await client.messages.create(
        model=_MODEL(),
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    text = message.content[0].text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
    result = json.loads(text.strip())

    return {
        "explanations": result.get("explanations", []),
        "overall_summary": result.get("overall_summary", ""),
    }


# ---------------------------------------------------------------------------
# write_quality_report
# ---------------------------------------------------------------------------

async def write_quality_report(
    pipeline_run_id: str,
    checks: list,
    anomaly_summary: dict,
    anomaly_explanations: list,
    output_dir: str,
) -> dict:
    """
    Write a quality report as both JSON and Markdown to the per-run output directory.

    Use when: run_quality_checks and (if applicable) explain_anomalies have
    completed and you are ready to persist the report.
    Do NOT use when: explain_anomalies has not yet been called and
    anomaly_count > 0 — always explain first.
    Returns: json_path (str), markdown_path (str),
             overall_status (str: pass | fail).
    """
    pool = await get_postgres_pool()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    overall_passed = all(c["passed"] for c in checks if c["severity"] == "critical")
    overall_status = "pass" if overall_passed else "fail"

    # --- Persist quality results to Postgres ---
    async with pool.acquire() as conn:
        for check in checks:
            await conn.execute(
                """
                INSERT INTO quality_results
                    (pipeline_run_id, check_name, check_type, column_name,
                     expected, actual, passed, severity)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                pipeline_run_id,
                check["check_name"],
                check.get("check_type", ""),
                check.get("column_name"),
                json.dumps({}),
                json.dumps({"detail": check.get("detail", "")}),
                check["passed"],
                check.get("severity", "error"),
            )

    # --- JSON report ---
    report_data = {
        "pipeline_run_id": pipeline_run_id,
        "overall_status": overall_status,
        "overall_passed": overall_passed,
        "checks": checks,
        "anomaly_summary": anomaly_summary,
        "anomaly_explanations": anomaly_explanations,
    }
    json_path = out / "quality_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, default=str)

    # --- Markdown report ---
    md_lines = [
        f"# Quality Report",
        f"",
        f"**Run ID:** `{pipeline_run_id}`  ",
        f"**Status:** {'✅ PASS' if overall_passed else '❌ FAIL'}",
        f"",
        f"## Checks",
        f"",
        f"| Check | Column | Passed | Severity | Detail |",
        f"|-------|--------|--------|----------|--------|",
    ]
    for c in checks:
        icon = "✅" if c["passed"] else "❌"
        col = c.get("column_name") or "—"
        md_lines.append(
            f"| {c['check_name']} | {col} | {icon} | {c.get('severity','')} | {c.get('detail','')} |"
        )

    if anomaly_explanations:
        md_lines += ["", "## Anomaly Explanations", ""]
        for exp in anomaly_explanations:
            md_lines += [
                f"### `{exp['column']}` — {exp['anomaly_count']} anomalies",
                f"",
                f"**Explanation:** {exp['explanation']}",
                f"",
                f"**Likely cause:** {exp['likely_cause']}",
                f"",
                f"**Recommended action:** {exp['recommended_action']}",
                f"",
            ]

    if anomaly_summary.get("overall_summary"):
        md_lines += [
            "## Overall Summary",
            "",
            anomaly_summary["overall_summary"],
            "",
        ]

    md_path = out / "quality_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "overall_status": overall_status,
    }
