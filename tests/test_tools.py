"""
Unit tests for MCP tool functions.

Each test exercises one tool in isolation.
- Profiling tools use real DuckDB (in-memory) — no mocking needed.
- compare_schemas mocks the Postgres pool (no Docker required).
- Domain tools use real YAML files + keyword heuristic path (no LLM needed).
- Quality tools use a real temp Parquet file written by fixtures.
- Library/catalogue tools are not tested here — covered in test_agents.py
  via the MockMCP integration.

Run with:
    pytest tests/test_tools.py -v
"""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conftest import SAMPLE_CSV

# ---------------------------------------------------------------------------
# Source tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_csv_returns_metadata(sales_csv_path):
    from mcp_server.tools.source_tools import connect_csv

    result = await connect_csv(file_path=sales_csv_path)

    assert result["row_count"] == 500
    assert "transaction_id" in result["column_names"]
    assert "unit_price_gbp" in result["column_names"]
    assert result["file_size_mb"] > 0


@pytest.mark.asyncio
async def test_connect_csv_missing_file():
    from mcp_server.tools.source_tools import connect_csv

    with pytest.raises(Exception):
        await connect_csv(file_path="/nonexistent/path.csv")


# ---------------------------------------------------------------------------
# Profiling tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sample_data_respects_sample_size(sales_csv_path):
    from mcp_server.tools.profiling_tools import sample_data

    result = await sample_data(
        source_path=sales_csv_path,
        source_type="csv",
        sample_size=50,
    )

    assert result["actual_sample_size"] == 50
    assert len(result["sample"]) == 50
    assert "transaction_id" in result["sample"][0]


@pytest.mark.asyncio
async def test_sample_data_caps_at_total_rows(sales_csv_path):
    from mcp_server.tools.profiling_tools import sample_data

    result = await sample_data(
        source_path=sales_csv_path,
        source_type="csv",
        sample_size=99999,
    )

    assert result["actual_sample_size"] == 500


@pytest.mark.asyncio
async def test_compute_profile_null_rate(sample_rows):
    from mcp_server.tools.profiling_tools import compute_profile

    # Introduce nulls into quantity
    rows_with_nulls = [dict(r) for r in sample_rows]
    for i in [0, 1, 2, 3, 4]:  # 5 out of 20
        rows_with_nulls[i]["quantity"] = None

    result = await compute_profile(sample=rows_with_nulls)

    assert result["total_rows"] == 20
    assert abs(result["quantity"]["null_rate"] - 0.25) < 0.01


@pytest.mark.asyncio
async def test_compute_profile_duplicate_count(sample_rows):
    from mcp_server.tools.profiling_tools import compute_profile

    # Add 3 duplicate rows
    duped = sample_rows + sample_rows[:3]

    result = await compute_profile(sample=duped)

    assert result["duplicate_row_count"] == 3


@pytest.mark.asyncio
async def test_compute_profile_numeric_stats():
    from mcp_server.tools.profiling_tools import compute_profile

    rows = [{"price": float(i), "label": f"x{i}"} for i in range(1, 11)]
    result = await compute_profile(sample=rows)

    assert result["price"]["mean"] == pytest.approx(5.5, abs=0.1)
    assert result["price"]["min"] == "1.0"
    assert result["price"]["max"] == "10.0"


@pytest.mark.asyncio
async def test_detect_schema_currency_column(sample_rows, basic_profile):
    from mcp_server.tools.profiling_tools import detect_schema

    result = await detect_schema(sample=sample_rows, profile=basic_profile)

    assert "unit_price_gbp" in result
    assert result["unit_price_gbp"]["needs_cast"] is True
    assert result["unit_price_gbp"]["inferred_type"] == "DOUBLE"


@pytest.mark.asyncio
async def test_detect_schema_date_column(sample_rows, basic_profile):
    from mcp_server.tools.profiling_tools import detect_schema

    result = await detect_schema(sample=sample_rows, profile=basic_profile)

    assert "transaction_date" in result
    assert result["transaction_date"]["inferred_type"] == "DATE"
    assert result["transaction_date"]["needs_cast"] is True


@pytest.mark.asyncio
async def test_detect_schema_boolean_column(sample_rows, basic_profile):
    from mcp_server.tools.profiling_tools import detect_schema

    result = await detect_schema(sample=sample_rows, profile=basic_profile)

    assert "is_returned" in result
    assert result["is_returned"]["inferred_type"] == "BOOLEAN"


@pytest.mark.asyncio
async def test_detect_schema_skips_non_dict_profile_values(sample_rows):
    from mcp_server.tools.profiling_tools import detect_schema

    profile = {
        "total_rows": 20,           # int — must be skipped
        "duplicate_row_count": 0,   # int — must be skipped
        "quantity": {
            "dtype": "INTEGER", "null_count": 0, "null_rate": 0.0,
            "unique_count": 20, "min": "1", "max": "20", "mean": 10.5,
            "top_5_values": [], "sample_values": [],
        },
    }

    result = await detect_schema(sample=sample_rows, profile=profile)

    # Should not raise; should only have 'quantity'
    assert "quantity" in result
    assert "total_rows" not in result


@pytest.mark.asyncio
async def test_compare_schemas_no_prior_run():
    """When no catalogue entry exists, tool returns no_prior_run=True."""
    from mcp_server.tools.profiling_tools import compare_schemas

    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value=None)

    mock_pool = MagicMock()
    mock_pool.acquire = MagicMock(
        return_value=MagicMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=False),
        )
    )

    with patch("mcp_server.tools.profiling_tools.get_postgres_pool", AsyncMock(return_value=mock_pool)):
        result = await compare_schemas(
            source_path="sample_data/sales_raw.csv",
            current_schema={"col_a": {"inferred_type": "string"}},
        )

    assert result["no_prior_run"] is True
    assert result["has_drift"] is False
    assert result["drift_severity"] == "none"


@pytest.mark.asyncio
async def test_compare_schemas_detects_dropped_column():
    from mcp_server.tools.profiling_tools import compare_schemas

    prior = {"col_a": {"inferred_type": "string"}, "col_b": {"inferred_type": "integer"}}
    current = {"col_a": {"inferred_type": "string"}}  # col_b dropped

    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value={"schema": json.dumps(prior)})

    mock_pool = MagicMock()
    mock_pool.acquire = MagicMock(
        return_value=MagicMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=False),
        )
    )

    with patch("mcp_server.tools.profiling_tools.get_postgres_pool", AsyncMock(return_value=mock_pool)):
        result = await compare_schemas(
            source_path="sample_data/sales_raw.csv",
            current_schema=current,
        )

    assert result["has_drift"] is True
    assert "col_b" in result["dropped_columns"]
    assert result["drift_severity"] == "critical"


@pytest.mark.asyncio
async def test_compare_schemas_detects_added_column():
    from mcp_server.tools.profiling_tools import compare_schemas

    prior = {"col_a": {"inferred_type": "string"}}
    current = {"col_a": {"inferred_type": "string"}, "col_new": {"inferred_type": "integer"}}

    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value={"schema": json.dumps(prior)})

    mock_pool = MagicMock()
    mock_pool.acquire = MagicMock(
        return_value=MagicMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=False),
        )
    )

    with patch("mcp_server.tools.profiling_tools.get_postgres_pool", AsyncMock(return_value=mock_pool)):
        result = await compare_schemas(
            source_path="sample_data/sales_raw.csv",
            current_schema=current,
        )

    assert result["has_drift"] is True
    assert "col_new" in result["added_columns"]
    assert result["drift_severity"] == "warning"


# ---------------------------------------------------------------------------
# Domain tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_domain_retail_by_keyword():
    from mcp_server.tools.domain_tools import detect_domain

    result = await detect_domain(
        column_names=["customer_id", "product_sku", "category", "quantity", "unit_price", "order_id"],
        sample_values={},
        user_goal="clean sales data",
    )

    assert result["domain"] == "retail"
    assert result["confidence"] >= 0.75, (
        f"Expected high confidence for obvious retail columns, got {result['confidence']}"
    )
    assert result["method"] == "keyword_heuristic"


@pytest.mark.asyncio
async def test_detect_domain_medical_by_keyword():
    from mcp_server.tools.domain_tools import detect_domain

    result = await detect_domain(
        column_names=["patient_id", "diagnosis_code", "medication", "dosage_mg"],
        sample_values={},
        user_goal="clean patient records",
    )

    assert result["domain"] == "medical"
    assert result["confidence"] >= 0.60


@pytest.mark.asyncio
async def test_load_domain_rules_retail():
    from mcp_server.tools.domain_tools import load_domain_rules

    result = await load_domain_rules(domain="retail")

    assert isinstance(result.get("required_transforms"), list)
    assert isinstance(result.get("forbidden_transforms"), list)
    assert isinstance(result.get("sensitive_columns"), list)


@pytest.mark.asyncio
async def test_load_domain_rules_unknown_domain():
    from mcp_server.tools.domain_tools import load_domain_rules

    result = await load_domain_rules(domain="nonexistent_domain_xyz")

    # Should return empty rules gracefully, not raise
    assert isinstance(result.get("required_transforms"), list)
    assert isinstance(result.get("forbidden_transforms"), list)


# ---------------------------------------------------------------------------
# Quality tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_quality_checks_passes_on_clean_data(clean_parquet, basic_profile):
    from mcp_server.tools.quality_tools import run_quality_checks

    schema = {
        "quantity": {"inferred_type": "INTEGER"},
        "unit_price_usd": {"inferred_type": "DOUBLE"},
    }
    profile = {
        **basic_profile,
        "total_rows": 100,
        "quantity": {
            **basic_profile.get("quantity", {}),
            "null_rate": 0.0,
            "mean": 50.5,
            "min": "1",
            "max": "100",
        },
        "unit_price_usd": {
            "dtype": "DOUBLE", "null_rate": 0.0, "null_count": 0,
            "unique_count": 100, "mean": 35.0, "min": "10.0", "max": "60.0",
            "top_5_values": [], "sample_values": [],
        },
    }

    result = await run_quality_checks(
        output_path=clean_parquet,
        original_profile=profile,
        schema=schema,
        transformations_applied=["dedup"],
    )

    assert result["overall_passed"] is True
    assert isinstance(result["checks"], list)
    assert len(result["checks"]) > 0


@pytest.mark.asyncio
async def test_run_quality_checks_fails_on_dirty_data(dirty_parquet):
    from mcp_server.tools.quality_tools import run_quality_checks

    profile = {
        "total_rows": 500,          # original had 500; dirty has only 10 → fails row count
        "duplicate_row_count": 0,
        "quantity": {
            "dtype": "DOUBLE", "null_rate": 0.0, "null_count": 0,
            "unique_count": 10, "mean": 5.0, "min": "1", "max": "10",
            "top_5_values": [], "sample_values": [],
        },
    }

    result = await run_quality_checks(
        output_path=dirty_parquet,
        original_profile=profile,
        schema={},
        transformations_applied=[],
    )

    assert result["overall_passed"] is False
    row_check = next(c for c in result["checks"] if c["check_name"] == "row_count")
    assert row_check["passed"] is False


@pytest.mark.asyncio
async def test_detect_anomalies_finds_outliers(tmp_path):
    from mcp_server.tools.quality_tools import detect_anomalies
    import pandas as pd

    # 9 normal rows + 1 extreme outlier
    df = pd.DataFrame({
        "price": [10.0, 11.0, 10.5, 9.8, 10.2, 10.7, 9.9, 10.3, 10.1, 9999.0],
        "label": [f"item-{i}" for i in range(10)],
    })
    path = tmp_path / "outlier.parquet"
    df.to_parquet(path, index=False)

    result = await detect_anomalies(
        output_path=str(path),
        schema={"price": {"inferred_type": "DOUBLE"}},
        target_columns=["price"],
    )

    assert result["anomaly_count"] > 0
    assert result["anomaly_rate"] > 0
