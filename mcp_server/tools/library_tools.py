"""
Transform library tools.

Semantic search over stored transformation patterns, save new patterns,
and generate dbt schema YAML.
"""

import json
import os
import uuid
from pathlib import Path

from db import get_postgres_pool
from memory.embeddings import embed


def _vec_literal(embedding: list) -> str:
    """Convert an embedding to a safe SQL vector literal (floats only — no injection)."""
    return "[" + ",".join(str(float(v)) for v in embedding) + "]"


# ---------------------------------------------------------------------------
# search_transform_library
# ---------------------------------------------------------------------------

async def search_transform_library(
    query: str,
    tags: list | None = None,
    top_k: int = 5,
) -> dict:
    """
    Search the transform library for reusable code snippets by semantic similarity.

    Use when: you are about to generate transformation code and want to check
    whether a similar pattern already exists in the library.
    Do NOT use when: you already have the code and are about to save it —
    use save_to_library instead.
    Returns: snippets (list[dict] each with name, description, code, tags,
             use_count, similarity_score).
    """
    query_embedding = await embed(query)
    embedding_str = _vec_literal(query_embedding)

    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        if tags:
            # Filter by tags AND rank by embedding similarity
            rows = await conn.fetch(
                f"""
                SELECT id, name, description, code, tags, use_count,
                       1 - (embedding <=> '{embedding_str}'::vector) AS similarity
                FROM transform_library
                WHERE tags && $1::text[]
                ORDER BY similarity DESC
                LIMIT $2
                """,
                tags,
                top_k,
            )
        else:
            rows = await conn.fetch(
                f"""
                SELECT id, name, description, code, tags, use_count,
                       1 - (embedding <=> '{embedding_str}'::vector) AS similarity
                FROM transform_library
                ORDER BY similarity DESC
                LIMIT $1
                """,
                top_k,
            )

        # Increment use_count for returned rows
        if rows:
            ids = [r["id"] for r in rows]
            await conn.execute(
                "UPDATE transform_library SET use_count = use_count + 1 WHERE id = ANY($1)",
                ids,
            )

    snippets = [
        {
            "name": r["name"],
            "description": r["description"],
            "code": r["code"],
            "tags": list(r["tags"]) if r["tags"] else [],
            "use_count": r["use_count"],
            "similarity_score": round(float(r["similarity"]), 4),
        }
        for r in rows
    ]

    return {"snippets": snippets}


# ---------------------------------------------------------------------------
# save_to_library
# ---------------------------------------------------------------------------

async def save_to_library(
    name: str,
    description: str,
    code: str,
    input_schema: dict,
    output_schema: dict,
    tags: list,
) -> dict:
    """
    Persist a reusable transformation pattern to the library with its embedding.

    Use when: quality_passed is True AND the transformation is generic enough
    to be reused on other datasets.
    Do NOT call when quality_passed is False — saving a failing transform
    corrupts the library. Check quality_passed before calling this tool.
    Returns: library_id (str), saved (bool).
    """
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        # Check for existing entry with the same name
        existing = await conn.fetchval(
            "SELECT id FROM transform_library WHERE name = $1", name
        )
        if existing:
            return {"library_id": str(existing), "saved": False}

    # Embed name + description + tags for similarity search
    embed_text = f"{name}. {description}. Tags: {', '.join(tags)}"
    embedding = await embed(embed_text)
    embedding_str = _vec_literal(embedding)

    async with pool.acquire() as conn:
        library_id = await conn.fetchval(
            f"""
            INSERT INTO transform_library
                (name, description, code, input_schema, output_schema, tags, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, '{embedding_str}'::vector)
            RETURNING id
            """,
            name,
            description,
            code,
            json.dumps(input_schema),
            json.dumps(output_schema),
            tags,
        )

    return {"library_id": str(library_id), "saved": True}


# ---------------------------------------------------------------------------
# generate_dbt_schema_yml
# ---------------------------------------------------------------------------

async def generate_dbt_schema_yml(
    model_name: str,
    output_schema: dict,
    column_descriptions: dict,
    run_id: str,
) -> dict:
    """
    Generate a dbt schema.yml file for the transformed dataset.

    Use when: the catalogue agent is generating dbt artefacts and needs the
    column-level schema YAML alongside the SQL model.
    Do NOT use when: you need dbt tests — use generate_dbt_tests for that.
    Returns: yml_content (str), file_path (str).
    """
    output_dir = Path(os.getenv("OUTPUT_DIR", "outputs")) / run_id / "dbt" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "version: 2",
        "",
        "models:",
        f"  - name: {model_name}",
        f"    description: >",
        f"      Transformed dataset produced by the data_agent pipeline.",
        "    columns:",
    ]

    for col, type_str in output_schema.items():
        desc = column_descriptions.get(col, f"Column {col}")
        lines += [
            f"      - name: {col}",
            f"        description: \"{desc}\"",
        ]

    yml_content = "\n".join(lines) + "\n"
    file_path = output_dir / "schema.yml"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(yml_content)

    return {"yml_content": yml_content, "file_path": str(file_path)}
