"""
Source connection tools.

Connects to CSV/Parquet files, Postgres tables, and REST APIs.
Also handles incremental watermark filtering via detect_new_rows.
"""

import uuid
from pathlib import Path

import duckdb
import httpx

from db import get_postgres_pool, get_duckdb_conn


# ---------------------------------------------------------------------------
# connect_csv
# ---------------------------------------------------------------------------

async def connect_csv(file_path: str) -> dict:
    """
    Open a CSV or Parquet file and return basic metadata.

    Use when: the source_type is 'csv' or 'parquet' and you need to confirm
    the file is readable and retrieve its shape before sampling.
    Do NOT use when: connecting to a database or API — use connect_postgres
    or connect_api instead.
    Returns: file_path (str), detected_delimiter (str), row_count (int),
             file_size_mb (float), column_names (list[str]),
             first_5_rows (list[dict]).
    """
    is_url = file_path.startswith("http://") or file_path.startswith("https://")

    if not is_url:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        file_size_mb = round(path.stat().st_size / (1024 * 1024), 4)
    else:
        path = None
        file_size_mb = None  # unknown for remote files

    ext = Path(file_path).suffix.lower().split("?")[0]  # strip query params from URL
    conn = get_duckdb_conn()

    try:
        if ext in (".parquet", ".pq"):
            conn.execute(f"CREATE OR REPLACE VIEW src AS SELECT * FROM read_parquet('{file_path}')")
            delimiter = "N/A (Parquet)"
        else:
            conn.execute(
                f"CREATE OR REPLACE VIEW src AS SELECT * FROM read_csv_auto('{file_path}')"
            )
            delimiter = _sniff_delimiter(file_path) if not is_url else "unknown (remote)"

        row_count = conn.execute("SELECT COUNT(*) FROM src").fetchone()[0]
        col_names = [d[0] for d in conn.execute("DESCRIBE src").fetchall()]
        first_5 = conn.execute("SELECT * FROM src LIMIT 5").df().to_dict(orient="records")

        return {
            "file_path": file_path,
            "detected_delimiter": delimiter,
            "row_count": row_count,
            "file_size_mb": file_size_mb,
            "column_names": col_names,
            "first_5_rows": first_5,
        }
    finally:
        conn.close()


def _sniff_delimiter(file_path: str) -> str:
    """Read the first line and guess the delimiter."""
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
        for delim in (",", "\t", ";", "|"):
            if delim in first_line:
                return delim
        return ","
    except Exception:
        return ","


# ---------------------------------------------------------------------------
# connect_postgres
# ---------------------------------------------------------------------------

async def connect_postgres(connection_string: str, table_name: str) -> dict:
    """
    Connect to a Postgres table and return its metadata.

    Use when: source_type is 'postgres' and you need the table schema and
    a row count before sampling.
    Do NOT use when: reading a file — use connect_csv instead.
    Returns: table_name (str), row_count (int), column_names (list[str]),
             column_types (dict[str, str]), first_5_rows (list[dict]).
    """
    import asyncpg

    conn = await asyncpg.connect(connection_string)
    try:
        row_count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")

        columns = await conn.fetch(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = $1
            ORDER BY ordinal_position
            """,
            table_name,
        )
        col_names = [r["column_name"] for r in columns]
        col_types = {r["column_name"]: r["data_type"] for r in columns}

        rows = await conn.fetch(f"SELECT * FROM {table_name} LIMIT 5")
        first_5 = [dict(r) for r in rows]

        return {
            "table_name": table_name,
            "row_count": row_count,
            "column_names": col_names,
            "column_types": col_types,
            "first_5_rows": first_5,
        }
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# connect_api
# ---------------------------------------------------------------------------

async def connect_api(
    url: str,
    headers: dict | None = None,
    params: dict | None = None,
) -> dict:
    """
    Fetch data from a REST API endpoint and return metadata.

    Use when: source_type is 'api' and you need to inspect what the endpoint
    returns before sampling.
    Do NOT use when: reading a file or database — use connect_csv or
    connect_postgres instead.
    Returns: url (str), detected_format (str), row_count (int),
             field_names (list[str]), first_5_rows (list[dict]),
             pagination_info (dict).
    """
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, headers=headers or {}, params=params or {})
        response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "json" in content_type:
        detected_format = "json"
        data = response.json()
        # Unwrap common envelope patterns
        if isinstance(data, dict):
            for key in ("data", "results", "items", "records", "rows"):
                if key in data and isinstance(data[key], list):
                    pagination_info = {k: v for k, v in data.items() if k != key}
                    data = data[key]
                    break
            else:
                pagination_info = {}
                data = [data] if data else []
        else:
            pagination_info = {}
    else:
        detected_format = "unknown"
        data = []
        pagination_info = {}

    if not data:
        return {
            "url": url,
            "detected_format": detected_format,
            "row_count": 0,
            "field_names": [],
            "first_5_rows": [],
            "pagination_info": pagination_info,
        }

    field_names = list(data[0].keys()) if isinstance(data[0], dict) else []
    first_5 = data[:5] if isinstance(data[0], dict) else []

    return {
        "url": url,
        "detected_format": detected_format,
        "row_count": len(data),
        "field_names": field_names,
        "first_5_rows": first_5,
        "pagination_info": pagination_info,
    }


# ---------------------------------------------------------------------------
# detect_new_rows
# ---------------------------------------------------------------------------

async def detect_new_rows(
    source_path: str,
    source_type: str,
    watermark_column: str,
    previous_watermark: str | None,
) -> dict:
    """
    Filter a dataset to rows newer than the previous watermark value.

    Use when: incremental_mode is True and you need only the rows added
    since the last successful pipeline run.
    Do NOT use when: running in full mode — process the whole source directly.
    Returns: filtered_path (str), new_row_count (int),
             watermark_column (str), watermark_value (str),
             previous_watermark (str | None).
    """
    temp_path = f"/tmp/incremental_{uuid.uuid4().hex}.parquet"
    conn = get_duckdb_conn()

    try:
        # Load source into DuckDB
        if source_type in ("csv", "parquet"):
            ext = Path(source_path).suffix.lower()
            if ext in (".parquet", ".pq"):
                conn.execute(f"CREATE OR REPLACE VIEW src AS SELECT * FROM read_parquet('{source_path}')")
            else:
                conn.execute(f"CREATE OR REPLACE VIEW src AS SELECT * FROM read_csv_auto('{source_path}')")
        else:
            raise ValueError(f"detect_new_rows only supports csv/parquet sources, got: {source_type}")

        # If no previous watermark, return everything
        if previous_watermark is None:
            conn.execute(
                f"COPY (SELECT * FROM src) TO '{temp_path}' (FORMAT PARQUET)"
            )
            new_max = conn.execute(
                f"SELECT MAX(CAST({watermark_column} AS VARCHAR)) FROM src"
            ).fetchone()[0]
            new_row_count = conn.execute("SELECT COUNT(*) FROM src").fetchone()[0]
        else:
            # Filter rows where watermark_column > previous_watermark
            # Cast both sides to VARCHAR for safe comparison across types
            conn.execute(
                f"""
                COPY (
                    SELECT * FROM src
                    WHERE CAST({watermark_column} AS VARCHAR) > '{previous_watermark}'
                ) TO '{temp_path}' (FORMAT PARQUET)
                """
            )
            new_row_count = conn.execute(
                f"""
                SELECT COUNT(*) FROM src
                WHERE CAST({watermark_column} AS VARCHAR) > '{previous_watermark}'
                """
            ).fetchone()[0]
            if new_row_count > 0:
                new_max = conn.execute(
                    f"""
                    SELECT MAX(CAST({watermark_column} AS VARCHAR)) FROM src
                    WHERE CAST({watermark_column} AS VARCHAR) > '{previous_watermark}'
                    """
                ).fetchone()[0]
            else:
                new_max = previous_watermark

        return {
            "filtered_path": temp_path,
            "new_row_count": new_row_count,
            "watermark_column": watermark_column,
            "watermark_value": str(new_max) if new_max is not None else previous_watermark,
            "previous_watermark": previous_watermark,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool registry — imported by server.py to build TOOL_HANDLERS
# ---------------------------------------------------------------------------

TOOLS: dict = {
    "connect_csv": connect_csv,
    "connect_postgres": connect_postgres,
    "connect_api": connect_api,
    "detect_new_rows": detect_new_rows,
}
