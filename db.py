"""
Shared database connection helpers.

Every MCP tool, agent, and script imports from here.
Connections are lazy-initialised on first call and reused across calls.
Never instantiate asyncpg, Redis, or DuckDB connections directly elsewhere.
"""

import asyncio
import os
from pathlib import Path

import asyncpg
import duckdb
from redis.asyncio import from_url as redis_from_url

# ---------------------------------------------------------------------------
# Module-level singletons — None until first call
# ---------------------------------------------------------------------------

_pg_pool: asyncpg.Pool | None = None
_redis = None
_pg_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------

async def get_postgres_pool() -> asyncpg.Pool:
    """
    Return the shared asyncpg connection pool, creating it on first call.
    Thread-safe via asyncio.Lock — safe when multiple coroutines start up
    concurrently during MCP server initialisation.
    """
    global _pg_pool
    async with _pg_lock:
        if _pg_pool is None:
            url = os.getenv("POSTGRES_URL")
            if not url:
                raise RuntimeError("POSTGRES_URL environment variable is not set")
            _pg_pool = await asyncpg.create_pool(url, min_size=2, max_size=10)
    return _pg_pool


async def close_postgres_pool() -> None:
    """Call on clean shutdown to release all Postgres connections."""
    global _pg_pool
    if _pg_pool is not None:
        await _pg_pool.close()
        _pg_pool = None


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------

async def get_redis_client():
    """
    Return the shared Redis async client, creating it on first call.
    redis.asyncio clients are connection-pooled internally — one instance
    is sufficient for the whole process.
    """
    global _redis
    if _redis is None:
        url = os.getenv("REDIS_URL", "redis://localhost:6379")
        _redis = redis_from_url(url, decode_responses=True)
    return _redis


async def close_redis_client() -> None:
    """Call on clean shutdown."""
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


# ---------------------------------------------------------------------------
# DuckDB
# ---------------------------------------------------------------------------

def get_duckdb_conn() -> duckdb.DuckDBPyConnection:
    """
    Return a new DuckDB connection for the caller's use.

    DuckDB is in-process and uses a file-based WAL; a new connection per
    tool call is intentional — it avoids cross-coroutine contention on the
    same connection object.  The data/ directory is created if absent.
    """
    path = os.getenv("DUCKDB_PATH", ":memory:")
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(path)
