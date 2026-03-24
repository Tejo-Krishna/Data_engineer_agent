"""
Data profiling tools.

Samples data, computes per-column statistics, infers schema, and
detects drift against the most recent catalogue entry.
All computation uses DuckDB — no pandas inside these tools.
"""

import json
import re
from pathlib import Path

from db import get_duckdb_conn, get_postgres_pool


# ---------------------------------------------------------------------------
# sample_data
# ---------------------------------------------------------------------------

async def sample_data(
    source_path: str,
    source_type: str,
    sample_size: int = 2000,
) -> dict:
    """
    Draw a representative sample from the source dataset.

    Use when: you need rows to pass to compute_profile or detect_schema.
    Do NOT use when: you need full-dataset statistics — use compute_profile
    directly on the source path instead.
    Returns: sample (list[dict]), actual_sample_size (int).
    """
    conn = get_duckdb_conn()
    try:
        _register_source(conn, source_path, source_type)
        total = conn.execute("SELECT COUNT(*) FROM src").fetchone()[0]
        limit = min(sample_size, total)
        rows = conn.execute(
            f"SELECT * FROM src USING SAMPLE {limit} ROWS"
        ).df().to_dict(orient="records")
        # Convert any non-serialisable types to strings
        rows = _serialise_rows(rows)
        return {"sample": rows, "actual_sample_size": len(rows)}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# compute_profile
# ---------------------------------------------------------------------------

async def compute_profile(sample: list[dict]) -> dict:
    """
    Compute per-column descriptive statistics from a sample.

    Use when: you have a sample from sample_data and need statistics for
    detect_schema and generate_transform_code.
    Do NOT use when: you need schema inference — call detect_schema after
    compute_profile, not instead of it.
    Returns: profile dict with keys total_rows, duplicate_row_count, and
             one entry per column containing dtype, null_count, null_rate,
             unique_count, min, max, mean, top_5_values, sample_values.
    """
    print(f"    [compute_profile] start: {len(sample)} rows", flush=True)
    if not sample:
        return {"total_rows": 0, "duplicate_row_count": 0}

    conn = get_duckdb_conn()
    try:
        conn.execute("CREATE TABLE prof AS SELECT * FROM (VALUES ?) t", [
            [tuple(row.values()) for row in sample]
        ])
    except Exception:
        # Fallback: register via relation API
        import pandas as pd
        df = pd.DataFrame(sample)
        conn.register("prof", df)

    try:
        total_rows = conn.execute("SELECT COUNT(*) FROM prof").fetchone()[0]
        col_info = conn.execute("DESCRIBE prof").fetchall()
        col_names = [c[0] for c in col_info]
        col_dtypes = {c[0]: c[1] for c in col_info}

        # Duplicate rows
        dup_count = total_rows - conn.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT * FROM prof)"
        ).fetchone()[0]

        profile: dict = {
            "total_rows": total_rows,
            "duplicate_row_count": dup_count,
        }

        for col in col_names:
            dtype = col_dtypes[col]
            null_count = conn.execute(
                f'SELECT COUNT(*) FROM prof WHERE "{col}" IS NULL'
            ).fetchone()[0]
            null_rate = round(null_count / total_rows, 4) if total_rows else 0.0
            unique_count = conn.execute(
                f'SELECT COUNT(DISTINCT "{col}") FROM prof'
            ).fetchone()[0]

            # Numeric stats
            is_numeric = any(t in dtype.upper() for t in ("INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "BIGINT", "HUGEINT"))
            if is_numeric:
                stats = conn.execute(
                    f'SELECT MIN("{col}"), MAX("{col}"), AVG("{col}") FROM prof'
                ).fetchone()
                col_min, col_max, col_mean = stats
                col_mean = round(float(col_mean), 4) if col_mean is not None else None
            else:
                col_min = col_max = col_mean = None

            # Top 5 values by frequency
            top_5_rows = conn.execute(
                f'SELECT "{col}", COUNT(*) AS cnt FROM prof '
                f'WHERE "{col}" IS NOT NULL '
                f'GROUP BY "{col}" ORDER BY cnt DESC LIMIT 5'
            ).fetchall()
            top_5_values = [str(r[0]) for r in top_5_rows]

            # Sample values (up to 5 non-null)
            sample_vals = conn.execute(
                f'SELECT "{col}" FROM prof WHERE "{col}" IS NOT NULL LIMIT 5'
            ).fetchall()
            sample_values = [str(r[0]) for r in sample_vals]

            profile[col] = {
                "dtype": dtype,
                "null_count": null_count,
                "null_rate": null_rate,
                "unique_count": unique_count,
                "min": _safe_str(col_min),
                "max": _safe_str(col_max),
                "mean": col_mean,
                "top_5_values": top_5_values,
                "sample_values": sample_values,
            }

        print(f"    [compute_profile] done: {len(profile)} columns profiled", flush=True)
        return profile
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# detect_schema
# ---------------------------------------------------------------------------

async def detect_schema(sample: list[dict], profile: dict) -> dict:
    """
    Infer the semantic type and cast requirements for each column.

    Use when: you have compute_profile output and need to tell the transformer
    what type each column really is (date, numeric, boolean, etc.).
    Do NOT use when: you only need raw DuckDB dtypes — those are already in
    the profile from compute_profile.
    Returns: dict keyed by column name, each with inferred_type,
             original_type, needs_cast (bool), suggested_cast,
             nullable (bool), and optional detected_formats for date columns.
    """
    print(f"    [detect_schema] start: {len(sample)} rows, {len(profile)} profile keys", flush=True)
    schema: dict = {}

    # Build a set of all non-null string values per column for heuristics
    col_values: dict[str, list[str]] = {}
    for row in sample:
        for col, val in row.items():
            if val is not None and str(val).strip() != "":
                col_values.setdefault(col, []).append(str(val).strip())
    print(f"    [detect_schema] col_values built for {len(col_values)} columns", flush=True)

    for col, col_profile in profile.items():
        if col in ("total_rows", "duplicate_row_count"):
            continue
        if not isinstance(col_profile, dict):
            continue
        print(f"    [detect_schema] processing column: {col}", flush=True)
        original_type = col_profile.get("dtype", "VARCHAR")
        nullable = col_profile.get("null_rate", 0.0) > 0
        values = col_values.get(col, [])

        inferred_type = original_type
        needs_cast = False
        suggested_cast = None
        detected_formats: list[str] | None = None

        if not values:
            schema[col] = {
                "inferred_type": "VARCHAR",
                "original_type": original_type,
                "needs_cast": False,
                "suggested_cast": None,
                "nullable": nullable,
            }
            continue

        # --- Numeric heuristic (currency prefix) ---
        stripped = [v.lstrip("£$€¥ ") for v in values]
        numeric_hits = sum(1 for v in stripped if _is_float(v))
        if numeric_hits / len(values) >= 0.90 and not _all_native_numeric(original_type):
            inferred_type = "DOUBLE"
            needs_cast = True
            suggested_cast = "strip_currency_prefix_and_cast_float"
            schema[col] = {
                "inferred_type": inferred_type,
                "original_type": original_type,
                "needs_cast": needs_cast,
                "suggested_cast": suggested_cast,
                "nullable": nullable,
            }
            continue

        # --- Date heuristic ---
        date_formats = _detect_date_formats(values)
        if date_formats:
            inferred_type = "DATE"
            needs_cast = True
            suggested_cast = "parse_mixed_dates"
            schema[col] = {
                "inferred_type": inferred_type,
                "original_type": original_type,
                "needs_cast": needs_cast,
                "suggested_cast": suggested_cast,
                "nullable": nullable,
                "detected_formats": date_formats,
            }
            continue

        # --- Boolean heuristic ---
        bool_set = {"0", "1", "true", "false", "yes", "no", "t", "f"}
        unique_lower = {v.lower() for v in values}
        if unique_lower and unique_lower.issubset(bool_set):
            inferred_type = "BOOLEAN"
            needs_cast = "BOOLEAN" not in original_type.upper()
            suggested_cast = "normalise_boolean" if needs_cast else None
            schema[col] = {
                "inferred_type": inferred_type,
                "original_type": original_type,
                "needs_cast": needs_cast,
                "suggested_cast": suggested_cast,
                "nullable": nullable,
            }
            continue

        # Default — keep as-is
        schema[col] = {
            "inferred_type": inferred_type,
            "original_type": original_type,
            "needs_cast": needs_cast,
            "suggested_cast": suggested_cast,
            "nullable": nullable,
        }

    return schema


# ---------------------------------------------------------------------------
# compare_schemas
# ---------------------------------------------------------------------------

async def compare_schemas(source_path: str, current_schema: dict) -> dict:
    """
    Compare the current dataset schema against the most recent catalogue entry.

    Use when: you need to detect column additions, drops, renames, or type
    changes before running the pipeline.
    Do NOT use when: there is no prior run — the tool handles that gracefully
    by returning has_drift=False and no_prior_run=True.
    Returns: has_drift (bool), no_prior_run (bool), added_columns (list),
             dropped_columns (list), renamed_columns (list),
             type_changes (list[dict]), drift_severity (none|warning|critical).
    """
    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT schema FROM catalogue
            WHERE source_path = $1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            source_path,
        )

    if row is None:
        return {
            "has_drift": False,
            "no_prior_run": True,
            "added_columns": [],
            "dropped_columns": [],
            "renamed_columns": [],
            "type_changes": [],
            "drift_severity": "none",
        }

    prior_schema: dict = json.loads(row["schema"]) if isinstance(row["schema"], str) else row["schema"]

    prior_cols = set(prior_schema.keys())
    current_cols = set(current_schema.keys())

    added = sorted(current_cols - prior_cols)
    dropped = sorted(prior_cols - current_cols)

    type_changes = []
    for col in prior_cols & current_cols:
        prior_type = prior_schema[col].get("inferred_type", prior_schema[col].get("dtype", ""))
        current_type = current_schema[col].get("inferred_type", "")
        if prior_type != current_type:
            type_changes.append({
                "column": col,
                "prior_type": prior_type,
                "current_type": current_type,
            })

    has_drift = bool(added or dropped or type_changes)

    if dropped or type_changes:
        drift_severity = "critical"
    elif added:
        drift_severity = "warning"
    else:
        drift_severity = "none"

    return {
        "has_drift": has_drift,
        "no_prior_run": False,
        "added_columns": added,
        "dropped_columns": dropped,
        "renamed_columns": [],  # rename detection requires NLP — left for future
        "type_changes": type_changes,
        "drift_severity": drift_severity,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _register_source(conn: object, source_path: str, source_type: str) -> None:
    ext = Path(source_path).suffix.lower()
    if source_type == "parquet" or ext in (".parquet", ".pq"):
        conn.execute(f"CREATE OR REPLACE VIEW src AS SELECT * FROM read_parquet('{source_path}')")
    else:
        conn.execute(f"CREATE OR REPLACE VIEW src AS SELECT * FROM read_csv_auto('{source_path}')")


def _serialise_rows(rows: list[dict]) -> list[dict]:
    """Convert non-JSON-serialisable values to strings."""
    import datetime
    result = []
    for row in rows:
        clean = {}
        for k, v in row.items():
            if isinstance(v, (datetime.date, datetime.datetime)):
                clean[k] = v.isoformat()
            elif v is None or isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                clean[k] = str(v)
        result.append(clean)
    return result


def _is_float(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def _all_native_numeric(dtype: str) -> bool:
    return any(t in dtype.upper() for t in ("INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "BIGINT"))


_DATE_PATTERNS = [
    (r"^\d{4}-\d{2}-\d{2}$", "YYYY-MM-DD"),
    (r"^\d{2}/\d{2}/\d{4}$", "DD/MM/YYYY"),
    (r"^\d{2}-\d{2}-\d{4}$", "DD-MM-YYYY"),
    (r"^\d{8}$", "DDMMYYYY"),
    (r"^\d{10}$", "unix_epoch"),
]


def _detect_date_formats(values: list[str]) -> list[str] | None:
    """Return detected date format strings if >70% of values match date patterns."""
    if not values:
        return None
    matched_formats: dict[str, int] = {}
    for v in values:
        for pattern, fmt in _DATE_PATTERNS:
            if re.match(pattern, v.strip()):
                matched_formats[fmt] = matched_formats.get(fmt, 0) + 1
                break
    total_matched = sum(matched_formats.values())
    if total_matched / len(values) >= 0.70:
        return list(matched_formats.keys())
    return None


def _safe_str(val) -> str | None:
    if val is None:
        return None
    return str(val)
