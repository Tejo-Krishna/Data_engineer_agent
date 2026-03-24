"""
Catalogue tools.

Writes dataset catalogue entries, generates column lineage graphs,
produces dbt SQL models and YAML artefacts.
All LLM calls use AsyncAnthropic. Postgres writes use asyncpg via db.py.
"""

import ast
import json
import os
import re
from pathlib import Path

import anthropic

from db import get_postgres_pool


def _vec_literal(embedding: list) -> str:
    """Convert an embedding to a safe SQL vector literal (floats only — no injection)."""
    return "[" + ",".join(str(float(v)) for v in embedding) + "]"


def _parse_llm_json(text: str) -> dict:
    """
    Parse LLM-returned JSON robustly.
    Strips markdown fences, tries json.loads, then ast.literal_eval.
    Falls back to an empty dict rather than crashing the catalogue run.
    """
    # Strip ```json ... ``` or ``` ... ``` fences
    text = text.strip()
    if text.startswith("```"):
        # Take the content between the first and last fence
        inner = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        inner = re.sub(r"\n?```$", "", inner)
        text = inner.strip()

    # Primary: standard JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: Python dict literal (handles single-quoted keys/values)
    try:
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass

    # Last resort: return empty dict so the rest of the catalogue run can continue
    return {}


from memory.embeddings import embed


_MODEL = lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")


# ---------------------------------------------------------------------------
# write_catalogue_entry
# ---------------------------------------------------------------------------

async def write_catalogue_entry(
    pipeline_run_id: str,
    dataset_name: str,
    output_path: str,
    schema: dict,
    row_count: int,
    source_type: str,
    source_path: str,
    transformations_applied: list,
) -> dict:
    """
    Register the transformed dataset in the catalogue with embeddings.

    Use when: the pipeline run has succeeded and you need to make the output
    dataset discoverable for future search and lineage queries.
    Do NOT use when: the pipeline failed — only catalogue successful outputs.
    Returns: catalogue_id (str), column_descriptions (dict[str, str]).
    """
    client = anthropic.AsyncAnthropic()

    # Generate LLM column descriptions
    prompt = f"""You are a data catalogue assistant. Write a brief (one sentence) description
for each column in this dataset.

Dataset name: {dataset_name}
Transformations applied: {transformations_applied}
Columns and types: {json.dumps(schema, indent=2)}

Return a JSON object mapping column name to description:
{{"column_name": "one-sentence description", ...}}"""

    message = await client.messages.create(
        model=_MODEL(),
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = message.content[0].text.strip()
    column_descriptions: dict = _parse_llm_json(text)

    # Embed dataset name + column names for similarity search
    embed_text = f"{dataset_name}. Columns: {', '.join(schema.keys())}"
    embedding = await embed(embed_text)
    embedding_str = _vec_literal(embedding)

    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        catalogue_id = await conn.fetchval(
            f"""
            INSERT INTO catalogue
                (dataset_name, source_path, source_type, schema,
                 column_descriptions, row_count, embedding, pipeline_run_id)
            VALUES ($1, $2, $3, $4, $5, $6, '{embedding_str}'::vector, $7)
            RETURNING id
            """,
            dataset_name,
            source_path,
            source_type,
            json.dumps(schema),
            json.dumps(column_descriptions),
            row_count,
            pipeline_run_id,
        )

    return {
        "catalogue_id": str(catalogue_id),
        "column_descriptions": column_descriptions,
    }


# ---------------------------------------------------------------------------
# generate_lineage_graph
# ---------------------------------------------------------------------------

async def generate_lineage_graph(
    pipeline_run_id: str,
    source_schema: dict,
    output_schema: dict,
    transformations_applied: list,
) -> dict:
    """
    Map source columns to output columns and render a Mermaid lineage diagram.

    Use when: write_catalogue_entry has completed and you need to document
    how each output column was derived from source columns.
    Do NOT use when: source_schema and output_schema are identical — a simple
    identity mapping is still worth documenting, so always call this.
    Returns: lineage_graph (dict with edges list), mermaid_diagram (str).
    """
    client = anthropic.AsyncAnthropic()

    prompt = f"""You are a data lineage expert. Map each output column to its source column(s).

Source columns: {list(source_schema.keys())}
Output columns: {list(output_schema.keys())}
Transformations applied: {transformations_applied}

Return a JSON object:
{{
  "edges": [
    {{
      "source_column": "<source col name>",
      "target_column": "<output col name>",
      "transformation": "<transformation applied, or 'passthrough' if unchanged>"
    }}
  ]
}}"""

    message = await client.messages.create(
        model=_MODEL(),
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    text = message.content[0].text.strip()
    lineage_data: dict = _parse_llm_json(text)
    edges = lineage_data.get("edges", [])

    # Persist lineage edges to Postgres
    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        for edge in edges:
            await conn.execute(
                """
                INSERT INTO lineage
                    (pipeline_run_id, source_dataset, source_column,
                     target_dataset, target_column, transformation)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                pipeline_run_id,
                "source",
                edge.get("source_column", ""),
                "output",
                edge.get("target_column", ""),
                edge.get("transformation", "passthrough"),
            )

    # Render Mermaid flowchart
    mermaid_lines = ["flowchart LR"]
    for edge in edges:
        src = edge.get("source_column", "?").replace(" ", "_")
        tgt = edge.get("target_column", "?").replace(" ", "_")
        tx = edge.get("transformation", "")
        if tx and tx != "passthrough":
            mermaid_lines.append(f'    {src} -->|"{tx}"| {tgt}')
        else:
            mermaid_lines.append(f"    {src} --> {tgt}")

    mermaid_diagram = "\n".join(mermaid_lines)

    return {
        "lineage_graph": {"edges": edges},
        "mermaid_diagram": mermaid_diagram,
    }


# ---------------------------------------------------------------------------
# generate_dbt_model
# ---------------------------------------------------------------------------

async def generate_dbt_model(
    pipeline_run_id: str,
    model_name: str,
    transformations_applied: list,
    source_schema: dict,
    output_schema: dict,
    run_id: str,
) -> dict:
    """
    Generate a dbt SQL SELECT model that replicates the pipeline transformations.

    Use when: lineage has been generated and you need a schedulable dbt model
    for the transformed dataset.
    Do NOT use when: the transformation involved complex Python logic that
    cannot be expressed in SQL — generate the model anyway as a best-effort
    approximation.
    Returns: sql_content (str), file_path (str).
    """
    client = anthropic.AsyncAnthropic()

    prompt = f"""You are a dbt SQL model author. Write a dbt SQL SELECT statement that
replicates the following transformations.

Source columns: {json.dumps(source_schema, indent=2)}
Output columns: {json.dumps(output_schema, indent=2)}
Transformations applied: {transformations_applied}

Rules:
- Use {{ ref('source') }} as the FROM clause
- Include all output columns in the SELECT
- Add inline comments explaining each non-trivial transformation
- Return only the SQL, no explanation

SQL:"""

    message = await client.messages.create(
        model=_MODEL(),
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    sql_content = re.sub(r"^```[a-zA-Z]*\n?", "", message.content[0].text.strip())
    sql_content = re.sub(r"\n?```$", "", sql_content).strip()

    output_dir = Path(os.getenv("OUTPUT_DIR", "outputs")) / run_id / "dbt" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{model_name}.sql"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(sql_content)

    return {"sql_content": sql_content, "file_path": str(file_path)}


# ---------------------------------------------------------------------------
# read_catalogue
# ---------------------------------------------------------------------------

async def read_catalogue(query: str, top_k: int = 5) -> dict:
    """
    Search the catalogue for datasets semantically similar to a query.

    Use when: you need to find previously processed datasets related to a
    given topic or column set.
    Do NOT use when: you need the current run's catalogue entry — read
    catalogue_id from state instead.
    Returns: entries (list[dict] each with catalogue_id, dataset_name,
             source_path, schema, column_descriptions, row_count,
             similarity_score).
    """
    query_embedding = await embed(query)
    embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, dataset_name, source_path, source_type,
                   schema, column_descriptions, row_count,
                   1 - (embedding <=> '{embedding_str}'::vector) AS similarity
            FROM catalogue
            ORDER BY similarity DESC
            LIMIT $1
            """,
            top_k,
        )

    entries = [
        {
            "catalogue_id": str(r["id"]),
            "dataset_name": r["dataset_name"],
            "source_path": r["source_path"],
            "source_type": r["source_type"],
            "schema": r["schema"] if isinstance(r["schema"], dict) else json.loads(r["schema"]),
            "column_descriptions": (
                r["column_descriptions"]
                if isinstance(r["column_descriptions"], dict)
                else json.loads(r["column_descriptions"])
            ),
            "row_count": r["row_count"],
            "similarity_score": round(float(r["similarity"]), 4),
        }
        for r in rows
    ]

    return {"entries": entries}


# ---------------------------------------------------------------------------
# generate_dbt_tests
# ---------------------------------------------------------------------------

async def generate_dbt_tests(
    pipeline_run_id: str,
    model_name: str,
    quality_checks: list,
    output_schema: dict,
    run_id: str,
) -> dict:
    """
    Generate a dbt schema_tests.yml file based on confirmed quality check results.

    Use when: generate_dbt_schema_yml has completed and you need to add
    automated dbt tests for the model.
    Do NOT use when: quality_passed is False — only generate tests for
    datasets that passed quality checks.
    Returns: tests_yml_content (str), file_path (str).
    """
    # Rule-based — no LLM needed
    tests_by_col: dict[str, list[str]] = {}

    passed_checks = [c for c in quality_checks if c.get("passed")]

    for check in passed_checks:
        col = check.get("column_name")
        if not col:
            continue
        tests_by_col.setdefault(col, [])

    # Uniqueness: flag if type_conformance passed for an ID-like column
    for col in output_schema:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ("_id", "id_", "transaction", "order", "uuid")):
            tests_by_col.setdefault(col, [])
            if "unique" not in tests_by_col[col]:
                tests_by_col[col].append("unique")
            if "not_null" not in tests_by_col[col]:
                tests_by_col[col].append("not_null")

    # not_null: add for columns where null_rate check passed with null_rate ~0
    for check in passed_checks:
        col = check.get("column_name")
        if check.get("check_name") == "null_rate" and col:
            detail = check.get("detail", "")
            if "→ 0.0%" in detail or "null_rate: 0.0%" in detail:
                tests_by_col.setdefault(col, [])
                if "not_null" not in tests_by_col[col]:
                    tests_by_col[col].append("not_null")

    # accepted_values: boolean columns
    for col, col_type in output_schema.items():
        if "BOOL" in col_type.upper():
            tests_by_col.setdefault(col, [])
            tests_by_col[col].append("accepted_values: [true, false]")

    # Build YAML
    lines = [
        "version: 2",
        "",
        "models:",
        f"  - name: {model_name}",
        "    columns:",
    ]
    for col, col_tests in tests_by_col.items():
        if col_tests:
            lines.append(f"      - name: {col}")
            lines.append("        tests:")
            for t in col_tests:
                if t.startswith("accepted_values"):
                    vals = t.split(": ")[1]
                    lines += [
                        "          - accepted_values:",
                        f"              values: {vals}",
                    ]
                else:
                    lines.append(f"          - {t}")

    tests_yml_content = "\n".join(lines) + "\n"

    output_dir = Path(os.getenv("OUTPUT_DIR", "outputs")) / run_id / "dbt" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "schema_tests.yml"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(tests_yml_content)

    return {"tests_yml_content": tests_yml_content, "file_path": str(file_path)}
