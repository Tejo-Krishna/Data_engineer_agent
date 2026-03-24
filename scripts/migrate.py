"""
Database migration script.

Creates all tables required by the data_agent pipeline.
Safe to run multiple times — uses CREATE TABLE IF NOT EXISTS throughout.

Usage:
    python scripts/migrate.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Allow imports from project root (db.py, etc.)
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import asyncpg
from db import get_postgres_pool, close_postgres_pool


# ---------------------------------------------------------------------------
# DDL statements
# ---------------------------------------------------------------------------

ENABLE_PGVECTOR = "CREATE EXTENSION IF NOT EXISTS vector;"

TABLES = {
    "transform_library": """
        CREATE TABLE IF NOT EXISTS transform_library (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name            TEXT NOT NULL,
            description     TEXT NOT NULL,
            code            TEXT NOT NULL,
            input_schema    JSONB,
            output_schema   JSONB,
            tags            TEXT[],
            embedding       vector(1536),
            use_count       INTEGER NOT NULL DEFAULT 0,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """,

    "catalogue": """
        CREATE TABLE IF NOT EXISTS catalogue (
            id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            dataset_name        TEXT NOT NULL,
            source_path         TEXT NOT NULL,
            source_type         TEXT NOT NULL,
            schema              JSONB,
            column_descriptions JSONB,
            row_count           BIGINT,
            embedding           vector(1536),
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            pipeline_run_id     UUID
        );
    """,

    "lineage": """
        CREATE TABLE IF NOT EXISTS lineage (
            id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            pipeline_run_id  UUID NOT NULL,
            source_dataset   TEXT NOT NULL,
            source_column    TEXT NOT NULL,
            target_dataset   TEXT NOT NULL,
            target_column    TEXT NOT NULL,
            transformation   TEXT
        );
    """,

    "pipeline_runs": """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_goal         TEXT,
            source_path       TEXT,
            source_type       TEXT,
            status            TEXT NOT NULL DEFAULT 'running',
            retry_count       INTEGER NOT NULL DEFAULT 0,
            rows_input        BIGINT,
            rows_output       BIGINT,
            quality_passed    BOOLEAN,
            output_path       TEXT,
            pipeline_script   TEXT,
            watermark_column  TEXT,
            watermark_value   TEXT,
            started_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at      TIMESTAMPTZ
        );
    """,

    "quality_results": """
        CREATE TABLE IF NOT EXISTS quality_results (
            id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            pipeline_run_id  UUID REFERENCES pipeline_runs(id) ON DELETE CASCADE,
            check_name       TEXT NOT NULL,
            check_type       TEXT NOT NULL,
            column_name      TEXT,
            expected         JSONB,
            actual           JSONB,
            passed           BOOLEAN NOT NULL,
            severity         TEXT NOT NULL DEFAULT 'error',
            created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """,

    # ------------------------------------------------------------------
    # Mock data tables — used for Postgres source connection testing
    # ------------------------------------------------------------------

    "users": """
        CREATE TABLE IF NOT EXISTS users (
            id         TEXT PRIMARY KEY,
            name       TEXT,
            email      TEXT,
            tier       TEXT NOT NULL DEFAULT 'standard',
            joined_at  TIMESTAMPTZ
        );
    """,

    "orders": """
        CREATE TABLE IF NOT EXISTS orders (
            id         TEXT PRIMARY KEY,
            user_id    TEXT,
            status     TEXT,
            items      JSONB,
            total      NUMERIC,
            placed_at  TIMESTAMPTZ
        );
    """,

    "shipments": """
        CREATE TABLE IF NOT EXISTS shipments (
            order_id       TEXT PRIMARY KEY REFERENCES orders(id),
            carrier        TEXT,
            tracking       TEXT,
            dispatched     BOOLEAN NOT NULL DEFAULT FALSE,
            editable       BOOLEAN NOT NULL DEFAULT TRUE,
            est_delivery   DATE,
            last_location  TEXT
        );
    """,
}

INDEXES = {
    "transform_library_embedding_idx": """
        CREATE INDEX IF NOT EXISTS transform_library_embedding_idx
        ON transform_library
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """,

    "catalogue_embedding_idx": """
        CREATE INDEX IF NOT EXISTS catalogue_embedding_idx
        ON catalogue
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """,
}


# ---------------------------------------------------------------------------
# Migration runner
# ---------------------------------------------------------------------------

async def run_migrations() -> None:
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        # Enable pgvector extension first — required for vector column type
        await conn.execute(ENABLE_PGVECTOR)
        print("✓ pgvector extension enabled")

        # Create tables
        for table_name, ddl in TABLES.items():
            await conn.execute(ddl)
            print(f"✓ table '{table_name}' ready")

        # Create indexes — must come after tables exist
        for index_name, ddl in INDEXES.items():
            await conn.execute(ddl)
            print(f"✓ index '{index_name}' ready")

    await close_postgres_pool()
    print("\nMigration complete.")


if __name__ == "__main__":
    asyncio.run(run_migrations())
