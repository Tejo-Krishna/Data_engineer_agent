"""
Shared pytest fixtures and configuration for all test modules.
"""

import asyncio
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# pytest-asyncio mode
# ---------------------------------------------------------------------------

pytest_plugins = ["pytest_asyncio"]


# ---------------------------------------------------------------------------
# DB singleton reset — runs before/after every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
async def reset_db_singletons():
    """
    Reset db.py module-level singletons before each test so that each test
    gets fresh connections bound to its own event loop.

    Without this, the Redis client and asyncpg pool created in test N are
    bound to test N's event loop. When pytest-asyncio creates a new loop for
    test N+1, using those cached clients raises 'Event loop is closed'.
    """
    import asyncio
    import db

    # Reset before the test — connections will be created fresh in this loop
    db._pg_pool = None
    db._redis = None
    db._pg_lock = asyncio.Lock()

    yield

    # Cleanup after the test
    try:
        await db.close_postgres_pool()
    except Exception:
        db._pg_pool = None
    try:
        await db.close_redis_client()
    except Exception:
        db._redis = None


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
SAMPLE_CSV = PROJECT_ROOT / "sample_data" / "sales_raw.csv"


# ---------------------------------------------------------------------------
# Shared data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sales_csv_path() -> str:
    return str(SAMPLE_CSV)


@pytest.fixture
def sample_rows() -> list[dict]:
    """20 clean rows that cover all column types in sales_raw.csv."""
    return [
        {
            "transaction_id": f"txn-{i:04d}",
            "customer_id": f"CUST-{1000 + i}",
            "product_name": "Wireless Headphones",
            "quantity": i + 1,
            "unit_price_gbp": f"£{10.50 + i:.2f}",
            "transaction_date": f"2024-0{(i % 9) + 1}-{(i % 28) + 1:02d}",
            "country": "United Kingdom",
            "payment_method": "credit_card",
            "discount_pct": round(i * 1.5, 1),
            "is_returned": "False" if i % 3 else "True",
        }
        for i in range(20)
    ]


@pytest.fixture
def numeric_sample_rows() -> list[dict]:
    """Rows with pure numeric unit_price (no currency prefix) for testing."""
    return [
        {
            "id": i,
            "price": float(10 + i),
            "label": f"item-{i}",
            "active": "true" if i % 2 == 0 else "false",
            "created_at": f"2024-01-{i + 1:02d}",
        }
        for i in range(10)
    ]


@pytest.fixture
def basic_profile() -> dict:
    """Minimal profile dict matching sample_rows columns."""
    return {
        "total_rows": 20,
        "duplicate_row_count": 0,
        "transaction_id": {
            "dtype": "VARCHAR", "null_count": 0, "null_rate": 0.0,
            "unique_count": 20, "min": None, "max": None, "mean": None,
            "top_5_values": ["txn-0001"], "sample_values": ["txn-0001"],
        },
        "quantity": {
            "dtype": "INTEGER", "null_count": 0, "null_rate": 0.0,
            "unique_count": 20, "min": "1", "max": "20", "mean": 10.5,
            "top_5_values": ["1"], "sample_values": ["1"],
        },
        "unit_price_gbp": {
            "dtype": "VARCHAR", "null_count": 0, "null_rate": 0.0,
            "unique_count": 20, "min": None, "max": None, "mean": None,
            "top_5_values": ["£10.50"], "sample_values": ["£10.50"],
        },
        "transaction_date": {
            "dtype": "VARCHAR", "null_count": 0, "null_rate": 0.0,
            "unique_count": 10, "min": None, "max": None, "mean": None,
            "top_5_values": ["2024-01-01"], "sample_values": ["2024-01-01"],
        },
        "is_returned": {
            "dtype": "VARCHAR", "null_count": 0, "null_rate": 0.0,
            "unique_count": 2, "min": None, "max": None, "mean": None,
            "top_5_values": ["False", "True"], "sample_values": ["False"],
        },
    }


@pytest.fixture
def clean_parquet(tmp_path) -> str:
    """Write a clean Parquet file and return its path."""
    df = pd.DataFrame({
        "transaction_id": [f"txn-{i}" for i in range(100)],
        "quantity": list(range(1, 101)),
        "unit_price_usd": [round(10.0 + i * 0.5, 2) for i in range(100)],
        "transaction_date": pd.to_datetime(["2024-01-01"] * 100),
    })
    path = tmp_path / "clean_output.parquet"
    df.to_parquet(path, index=False)
    return str(path)


@pytest.fixture
def dirty_parquet(tmp_path) -> str:
    """Write a Parquet file with quality issues (many nulls, few rows)."""
    df = pd.DataFrame({
        "transaction_id": [f"txn-{i}" if i < 5 else None for i in range(10)],
        "quantity": [None] * 10,
        "unit_price_usd": [None] * 10,
    })
    path = tmp_path / "dirty_output.parquet"
    df.to_parquet(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Mock logger
# ---------------------------------------------------------------------------

class MockLogger:
    """Drop-in for RunLogger / LangfuseLogger — records calls for assertions."""

    def __init__(self):
        self.spans: dict = {}
        self.tool_calls: list = []

    def agent_start(self, name: str) -> str:
        self.spans[name] = {"tool_calls": []}
        return name

    def tool_call(self, span, agent, tool, input_summary, output_summary):
        self.tool_calls.append({
            "agent": agent, "tool": tool,
            "input": input_summary, "output": output_summary,
        })

    def agent_end(self, span) -> None:
        pass

    def finish(self, status, outputs, error=None):
        pass


@pytest.fixture
def mock_logger():
    return MockLogger()


# ---------------------------------------------------------------------------
# Mock MCP client
# ---------------------------------------------------------------------------

class MockMCP:
    """
    Fake MCP client for agent unit tests.

    Pass tool_responses as a dict mapping tool_name → response_dict.
    Callable values are called with the tool args dict.
    Tracks every call in self.calls for assertion.
    """

    def __init__(self, tool_responses: dict):
        self.tool_responses = tool_responses
        self.calls: list[tuple[str, dict]] = []

    async def call(self, tool_name: str, args: dict) -> dict:
        self.calls.append((tool_name, args))
        response = self.tool_responses.get(tool_name, {})
        return response(args) if callable(response) else response

    async def list_tools(self) -> list[dict]:
        return [
            {"name": name, "description": "", "inputSchema": {}}
            for name in self.tool_responses
        ]

    def called_tools(self) -> list[str]:
        return [name for name, _ in self.calls]


@pytest.fixture
def MockMCPClass():
    return MockMCP
