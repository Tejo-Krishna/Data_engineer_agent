"""
Seed the transform library with common reusable transformation patterns.

Safe to run multiple times — skips entries that already exist by name.

Usage:
    python scripts/seed_library.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from db import get_postgres_pool, close_postgres_pool
from memory.embeddings import embed


LIBRARY_ENTRIES = [
    {
        "name": "gbp_to_usd_conversion",
        "description": "Strip £ prefix from price columns and convert GBP values to USD using a static FX rate.",
        "tags": ["currency", "conversion", "financial"],
        "input_schema": {"unit_price_gbp": "VARCHAR"},
        "output_schema": {"unit_price_usd": "DOUBLE"},
        "code": """\
import os, pandas as pd

df = pd.read_csv(os.environ["INPUT_PATH"]) if os.environ["INPUT_PATH"].endswith(".csv") else pd.read_parquet(os.environ["INPUT_PATH"])
print(f"ROWS_IN: {len(df)}")

FX_RATE = float(os.environ.get("FX_STATIC_RATE", "1.27"))

for col in df.columns:
    if "price" in col.lower() or "amount" in col.lower():
        df[col] = df[col].astype(str).str.lstrip("£$€¥ ").str.replace(",", "")
        df[col] = pd.to_numeric(df[col], errors="coerce") * FX_RATE

df.to_parquet(os.environ["OUTPUT_PATH"], index=False)
print(f"ROWS_OUT: {len(df)}")
""",
    },
    {
        "name": "mixed_date_standardisation",
        "description": "Detect and normalise mixed date formats (YYYY-MM-DD and DD/MM/YYYY) to ISO 8601.",
        "tags": ["date", "normalisation", "parsing"],
        "input_schema": {"transaction_date": "VARCHAR"},
        "output_schema": {"transaction_date": "DATE"},
        "code": """\
import os, pandas as pd
from datetime import datetime

df = pd.read_csv(os.environ["INPUT_PATH"]) if os.environ["INPUT_PATH"].endswith(".csv") else pd.read_parquet(os.environ["INPUT_PATH"])
print(f"ROWS_IN: {len(df)}")

def parse_date(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None

date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
for col in date_cols:
    df[col] = df[col].apply(parse_date)

df.to_parquet(os.environ["OUTPUT_PATH"], index=False)
print(f"ROWS_OUT: {len(df)}")
""",
    },
    {
        "name": "dedup_by_id_column",
        "description": "Remove duplicate rows based on an ID column, keeping the first occurrence.",
        "tags": ["dedup", "id", "cleaning"],
        "input_schema": {"transaction_id": "VARCHAR"},
        "output_schema": {"transaction_id": "VARCHAR"},
        "code": """\
import os, pandas as pd

df = pd.read_csv(os.environ["INPUT_PATH"]) if os.environ["INPUT_PATH"].endswith(".csv") else pd.read_parquet(os.environ["INPUT_PATH"])
print(f"ROWS_IN: {len(df)}")

id_cols = [c for c in df.columns if c.lower().endswith("_id") or c.lower() == "id"]
if id_cols:
    df = df.drop_duplicates(subset=[id_cols[0]], keep="first")

df.to_parquet(os.environ["OUTPUT_PATH"], index=False)
print(f"ROWS_OUT: {len(df)}")
""",
    },
    {
        "name": "negative_quantity_flag",
        "description": "Flag rows with negative quantity as returns by adding an is_return boolean column.",
        "tags": ["retail", "quantity", "flag"],
        "input_schema": {"quantity": "INTEGER"},
        "output_schema": {"quantity": "INTEGER", "is_return": "BOOLEAN"},
        "code": """\
import os, pandas as pd

df = pd.read_csv(os.environ["INPUT_PATH"]) if os.environ["INPUT_PATH"].endswith(".csv") else pd.read_parquet(os.environ["INPUT_PATH"])
print(f"ROWS_IN: {len(df)}")

qty_cols = [c for c in df.columns if "qty" in c.lower() or "quantity" in c.lower()]
if qty_cols:
    col = qty_cols[0]
    df["is_return"] = df[col].apply(lambda x: bool(pd.notna(x) and float(x) < 0))

df.to_parquet(os.environ["OUTPUT_PATH"], index=False)
print(f"ROWS_OUT: {len(df)}")
""",
    },
    {
        "name": "currency_symbol_stripping",
        "description": "Strip leading currency symbols (£ $ € ¥) from all price or amount columns and cast to float.",
        "tags": ["currency", "cleaning", "numeric"],
        "input_schema": {"unit_price": "VARCHAR"},
        "output_schema": {"unit_price": "DOUBLE"},
        "code": """\
import os, pandas as pd

df = pd.read_csv(os.environ["INPUT_PATH"]) if os.environ["INPUT_PATH"].endswith(".csv") else pd.read_parquet(os.environ["INPUT_PATH"])
print(f"ROWS_IN: {len(df)}")

for col in df.columns:
    if any(kw in col.lower() for kw in ("price", "amount", "cost", "fee", "total")):
        df[col] = df[col].astype(str).str.lstrip("£$€¥ ").str.replace(",", "")
        df[col] = pd.to_numeric(df[col], errors="coerce")

df.to_parquet(os.environ["OUTPUT_PATH"], index=False)
print(f"ROWS_OUT: {len(df)}")
""",
    },
    {
        "name": "null_fill_unknown_strings",
        "description": "Fill null values in string columns with the string 'unknown' to preserve row completeness.",
        "tags": ["medical", "null", "filling"],
        "input_schema": {"any_string_col": "VARCHAR"},
        "output_schema": {"any_string_col": "VARCHAR"},
        "code": """\
import os, pandas as pd

df = pd.read_csv(os.environ["INPUT_PATH"]) if os.environ["INPUT_PATH"].endswith(".csv") else pd.read_parquet(os.environ["INPUT_PATH"])
print(f"ROWS_IN: {len(df)}")

str_cols = df.select_dtypes(include=["object"]).columns
df[str_cols] = df[str_cols].fillna("unknown")

df.to_parquet(os.environ["OUTPUT_PATH"], index=False)
print(f"ROWS_OUT: {len(df)}")
""",
    },
    {
        "name": "phone_number_normalisation",
        "description": "Strip non-numeric characters from phone columns and format to E.164 (UK +44 default).",
        "tags": ["pii", "phone", "normalisation"],
        "input_schema": {"phone": "VARCHAR"},
        "output_schema": {"phone": "VARCHAR"},
        "code": """\
import os, re, pandas as pd

df = pd.read_csv(os.environ["INPUT_PATH"]) if os.environ["INPUT_PATH"].endswith(".csv") else pd.read_parquet(os.environ["INPUT_PATH"])
print(f"ROWS_IN: {len(df)}")

def normalise_phone(val):
    if pd.isna(val):
        return None
    digits = re.sub(r"\\D", "", str(val))
    if digits.startswith("0") and len(digits) == 11:
        return "+44" + digits[1:]
    if digits.startswith("44") and len(digits) == 12:
        return "+" + digits
    return digits if digits else None

phone_cols = [c for c in df.columns if "phone" in c.lower() or "mobile" in c.lower() or "tel" in c.lower()]
for col in phone_cols:
    df[col] = df[col].apply(normalise_phone)

df.to_parquet(os.environ["OUTPUT_PATH"], index=False)
print(f"ROWS_OUT: {len(df)}")
""",
    },
    {
        "name": "title_case_name_columns",
        "description": "Convert job_title, name, and department columns to Title Case for consistent formatting.",
        "tags": ["employment", "string", "casing"],
        "input_schema": {"job_title": "VARCHAR", "full_name": "VARCHAR"},
        "output_schema": {"job_title": "VARCHAR", "full_name": "VARCHAR"},
        "code": """\
import os, pandas as pd

df = pd.read_csv(os.environ["INPUT_PATH"]) if os.environ["INPUT_PATH"].endswith(".csv") else pd.read_parquet(os.environ["INPUT_PATH"])
print(f"ROWS_IN: {len(df)}")

title_case_cols = [
    c for c in df.columns
    if any(kw in c.lower() for kw in ("name", "title", "department", "role", "position"))
]
for col in title_case_cols:
    df[col] = df[col].apply(lambda x: str(x).title() if pd.notna(x) else x)

df.to_parquet(os.environ["OUTPUT_PATH"], index=False)
print(f"ROWS_OUT: {len(df)}")
""",
    },
]


async def seed() -> None:
    pool = await get_postgres_pool()

    seeded = 0
    skipped = 0

    for entry in LIBRARY_ENTRIES:
        async with pool.acquire() as conn:
            existing = await conn.fetchval(
                "SELECT id FROM transform_library WHERE name = $1", entry["name"]
            )
            if existing:
                print(f"  skip  '{entry['name']}' (already exists)")
                skipped += 1
                continue

        embed_text = f"{entry['name']}. {entry['description']}. Tags: {', '.join(entry['tags'])}"
        embedding = await embed(embed_text)
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO transform_library
                    (name, description, code, input_schema, output_schema, tags, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, '{embedding_str}'::vector)
                """,
                entry["name"],
                entry["description"],
                entry["code"],
                json.dumps(entry["input_schema"]),
                json.dumps(entry["output_schema"]),
                entry["tags"],
            )

        print(f"  ✓ seeded '{entry['name']}'")
        seeded += 1

    await close_postgres_pool()
    print(f"\nDone. Seeded: {seeded}, Skipped: {skipped}")


if __name__ == "__main__":
    asyncio.run(seed())
