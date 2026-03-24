"""
Generate sample_data/sales_raw.csv with ~500 rows and intentional quality issues.

The quality issues baked into this file exist to exercise every agent:
  1. ~3% duplicate transaction_id values
  2. Two rows with negative quantity
  3. ~10% of unit_price_gbp values have a '£' prefix
  4. transaction_date in mixed formats: YYYY-MM-DD and DD/MM/YYYY
  5. ~5% null values in discount_pct
  6. is_returned values as a mix of True, False, 1, 0
  7. country values in mixed casing (uk, UK, United Kingdom)

Usage:
    python scripts/generate_sample_data.py
"""

import csv
import os
import random
import sys
from datetime import date, timedelta
from pathlib import Path
from uuid import uuid4

# Reproducible output
random.seed(42)

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

PRODUCTS = [
    "Wireless Headphones", "USB-C Hub", "Mechanical Keyboard", "Gaming Mouse",
    "Monitor Stand", "Webcam HD", "Laptop Stand", "SSD 1TB", "RAM 16GB",
    "Ethernet Cable", "HDMI Adapter", "Desk Lamp", "Cable Organiser",
    "Mouse Pad XL", "Screen Cleaner Kit",
]

COUNTRIES_CLEAN = ["UK", "Germany", "France", "Netherlands", "Sweden"]

# Intentional casing variants for country (issue #7)
COUNTRY_VARIANTS = {
    "UK": ["UK", "uk", "United Kingdom", "united kingdom", "U.K."],
    "Germany": ["Germany", "germany", "GERMANY", "Deutschland"],
    "France": ["France", "france", "FRANCE"],
    "Netherlands": ["Netherlands", "netherlands", "The Netherlands"],
    "Sweden": ["Sweden", "sweden", "SWEDEN"],
}

PAYMENT_METHODS = ["credit_card", "debit_card", "paypal", "bank_transfer", "voucher"]

BASE_DATE = date(2024, 1, 1)


def random_date() -> date:
    return BASE_DATE + timedelta(days=random.randint(0, 364))


def format_date(d: date, fmt: str) -> str:
    if fmt == "iso":
        return d.strftime("%Y-%m-%d")
    else:
        return d.strftime("%d/%m/%Y")


def random_country_variant(country: str) -> str:
    return random.choice(COUNTRY_VARIANTS[country])


# ---------------------------------------------------------------------------
# Row generation
# ---------------------------------------------------------------------------

def generate_rows(n: int = 500) -> list[dict]:
    rows = []
    used_ids: list[str] = []

    for i in range(n):
        txn_date = random_date()

        # Issue #4: mixed date formats — ~50/50 split
        date_fmt = "iso" if random.random() < 0.5 else "dmy"
        date_str = format_date(txn_date, date_fmt)

        # Issue #3: ~10% have £ prefix on price
        price = round(random.uniform(4.99, 299.99), 2)
        price_str = f"£{price:.2f}" if random.random() < 0.10 else f"{price:.2f}"

        # Issue #5: ~5% null in discount_pct
        discount = round(random.uniform(0, 30), 1) if random.random() > 0.05 else ""

        # Issue #6: mixed boolean representation for is_returned
        is_returned_val = random.random() < 0.08  # ~8% returns
        is_returned_repr = random.choice(
            [str(is_returned_val), "1" if is_returned_val else "0"]
        )

        country = random.choice(COUNTRIES_CLEAN)

        # Issue #2: two rows get negative quantity
        if i in (47, 203):
            qty = random.randint(-5, -1)
        else:
            qty = random.randint(1, 10)

        txn_id = str(uuid4())
        used_ids.append(txn_id)

        rows.append({
            "transaction_id": txn_id,
            "customer_id": f"CUST-{random.randint(1000, 9999)}",
            "product_name": random.choice(PRODUCTS),
            "quantity": qty,
            "unit_price_gbp": price_str,
            "transaction_date": date_str,
            "country": random_country_variant(country),
            "payment_method": random.choice(PAYMENT_METHODS),
            "discount_pct": discount,
            "is_returned": is_returned_repr,
        })

    # Issue #1: ~3% duplicate transaction_ids — overwrite transaction_id on
    # ~15 rows with IDs already used earlier in the list
    n_dupes = max(1, int(n * 0.03))
    dupe_source_ids = random.sample(used_ids[:n // 2], n_dupes)
    dupe_target_indices = random.sample(range(n // 2, n), n_dupes)

    for idx, src_id in zip(dupe_target_indices, dupe_source_ids):
        rows[idx]["transaction_id"] = src_id

    return rows


# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

def main() -> None:
    out_path = Path(__file__).parent.parent / "sample_data" / "sales_raw.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = generate_rows(500)
    fieldnames = list(rows[0].keys())

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Verify duplicate count
    ids = [r["transaction_id"] for r in rows]
    dupes = len(ids) - len(set(ids))

    print(f"✓ Written {len(rows)} rows to {out_path}")
    print(f"  └─ duplicate transaction_ids: {dupes}")
    print(f"  └─ negative quantities: {sum(1 for r in rows if str(r['quantity']).startswith('-'))}")
    print(f"  └─ prices with £ prefix: {sum(1 for r in rows if str(r['unit_price_gbp']).startswith('£'))}")
    print(f"  └─ null discount_pct: {sum(1 for r in rows if r['discount_pct'] == '')}")
    print(f"  └─ mixed date formats (DD/MM/YYYY): {sum(1 for r in rows if '/' in str(r['transaction_date']))}")


if __name__ == "__main__":
    main()
