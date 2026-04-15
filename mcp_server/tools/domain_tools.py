"""
Domain detection and rules loading tools.

Two-stage domain detection: keyword heuristics first (no LLM),
Claude only when heuristics are ambiguous. Rules are loaded from
YAML files in domain_rules/ and fed into the transformer as hard constraints.
"""

import asyncio
import os
from pathlib import Path

import anthropic
import yaml

_DOMAIN_RULES_DIR = Path(os.getenv("DOMAIN_RULES_DIR", "domain_rules"))

# Keyword sets per domain — used in stage-1 heuristic detection
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "financial": [
        "price", "amount", "revenue", "transaction", "payment", "invoice",
        "balance", "account", "tax", "currency", "gbp", "usd", "eur",
        "fee", "discount", "refund", "credit", "debit", "ledger",
        "profit", "loss", "margin",
    ],
    "medical": [
        "patient", "diagnosis", "icd", "prescription", "medication",
        "dosage", "clinical", "hospital", "admission", "discharge",
        "procedure", "nhs", "lab", "test", "result", "symptom",
        "treatment", "physician", "doctor", "nurse", "vital",
        "blood", "glucose", "blood_pressure", "heart_rate",
        "patient_id", "diagnosis_code", "dosage_mg", "ward",
        "medical_record", "ehr", "fhir", "mrn",
    ],
    "automotive": [
        "vehicle", "vin", "make", "model", "mileage", "odometer",
        "engine", "fuel", "transmission", "registration", "plate",
        "dealer", "service", "mot", "chassis", "manufacturer",
        "horsepower", "torque", "ev", "hybrid", "emission", "co2",
    ],
    "employment": [
        "employee", "staff", "salary", "wage", "payroll", "department",
        "job_title", "hire_date", "termination", "contract", "hours",
        "overtime", "leave", "absence", "appraisal", "manager",
        "headcount", "fte", "grade", "band", "pension",
        "ni_number", "national_insurance",
    ],
    "retail": [
        "product", "sku", "category", "inventory", "stock", "order",
        "customer", "purchase", "basket", "checkout", "shipping",
        "delivery", "return", "refund", "coupon", "promotion",
        "loyalty", "store", "channel", "quantity", "unit_price",
        "total_price",
    ],
}


# ---------------------------------------------------------------------------
# detect_domain
# ---------------------------------------------------------------------------

async def detect_domain(
    column_names: list[str],
    sample_values: dict[str, list[str]],
    user_goal: str,
) -> dict:
    """
    Identify which business domain the dataset belongs to.

    Use when: you need to load the correct domain rules before generating
    transformation code.
    Do NOT use when: the domain is already known from a previous run — pass
    domain directly to load_domain_rules instead.
    Returns: domain (str), confidence (float 0-1), method (str:
             keyword_heuristic | llm_classification).
    """
    # Collect tokens from column names and sample values
    tokens = _extract_tokens(column_names, sample_values, user_goal)

    # Stage 1 — keyword heuristic
    scores = _score_domains(tokens)
    best_domain, best_score = max(scores.items(), key=lambda x: x[1])
    sorted_scores = sorted(scores.values(), reverse=True)
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    gap = best_score - second_score

    best_hits = round(best_score * len(_DOMAIN_KEYWORDS[best_domain]))
    if (best_score >= 0.30 or best_hits >= 5) and gap >= 0.15:
        confidence = min(0.95, best_score * 2 + gap)
        return {
            "domain": best_domain,
            "confidence": round(confidence, 3),
            "method": "keyword_heuristic",
        }

    # Stage 2 — LLM classification
    return await _llm_classify(column_names, sample_values, user_goal)


def _extract_tokens(
    column_names: list[str],
    sample_values: dict[str, list[str]],
    user_goal: str,
) -> set[str]:
    tokens: set[str] = set()
    for col in column_names:
        tokens.update(col.lower().split("_"))
        tokens.add(col.lower())
    for vals in sample_values.values():
        for v in vals:
            tokens.add(str(v).lower().strip())
    tokens.update(user_goal.lower().split())
    return tokens


def _score_domains(tokens: set[str]) -> dict[str, float]:
    scores = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in tokens)
        scores[domain] = hits / len(keywords) if keywords else 0.0
    return scores


async def _llm_classify(
    column_names: list[str],
    sample_values: dict[str, list[str]],
    user_goal: str,
) -> dict:
    client = anthropic.AsyncAnthropic()
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    # Format sample values for the prompt (up to 3 per column)
    sample_preview = {
        col: vals[:3] for col, vals in list(sample_values.items())[:10]
    }

    prompt = f"""You are a data domain classifier. Classify the dataset below into one of these domains:
financial, medical, automotive, employment, retail, unknown

Column names: {column_names}
Sample values (up to 3 per column): {sample_preview}
User goal: {user_goal}

Respond with a JSON object only, no explanation:
{{"domain": "<domain>", "confidence": <0.0-1.0>}}"""

    for attempt in range(5):
        try:
            message = await client.messages.create(
                model=model,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except anthropic.RateLimitError:
            if attempt == 4:
                raise
            await asyncio.sleep(2 ** attempt)  # 1s, 2s, 4s, 8s

    import json
    text = message.content[0].text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    result = json.loads(text.strip())

    return {
        "domain": result.get("domain", "unknown"),
        "confidence": round(float(result.get("confidence", 0.5)), 3),
        "method": "llm_classification",
    }


# ---------------------------------------------------------------------------
# load_domain_rules
# ---------------------------------------------------------------------------

async def load_domain_rules(domain: str) -> dict:
    """
    Load the YAML rule file for the given domain.

    Use when: detect_domain has returned and you need the rules to pass as
    domain_context into generate_transform_code.
    Do NOT use when: domain is 'unknown' and you only want to skip rules —
    this tool handles unknown gracefully by returning empty rule sets.
    Returns: domain (str), sensitive_columns (list), required_transforms (list),
             forbidden_transforms (list), validation_rules (dict),
             default_watermark_column (str | None).
    """
    if not domain or domain == "unknown":
        return _empty_rules("unknown")

    yaml_path = _DOMAIN_RULES_DIR / f"{domain}.yaml"
    if not yaml_path.exists():
        return _empty_rules(domain)

    with open(yaml_path, encoding="utf-8") as f:
        rules = yaml.safe_load(f)

    return {
        "domain": rules.get("domain", domain),
        "sensitive_columns": rules.get("sensitive_columns", []),
        "required_transforms": rules.get("required_transforms", []),
        "forbidden_transforms": rules.get("forbidden_transforms", []),
        "validation_rules": rules.get("validation_rules", {}),
        "default_watermark_column": rules.get("default_watermark_column"),
        "detection_keywords": rules.get("detection_keywords", []),
    }


def _empty_rules(domain: str) -> dict:
    return {
        "domain": domain,
        "sensitive_columns": [],
        "required_transforms": [],
        "forbidden_transforms": [],
        "validation_rules": {},
        "default_watermark_column": None,
        "detection_keywords": [],
    }
