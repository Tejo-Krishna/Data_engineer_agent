# Skill: Domain

Read this before working on domain detection, domain YAML files,
or any code that reads domain_context from state.

---

## What domain context does

Domain context is a hard constraint layer that sits above the user's
goal in the transformer agent's instruction hierarchy.

Priority order (highest to lowest):
1. Domain rules (forbidden_transforms, required_transforms)
2. User's natural language goal
3. Profile and schema heuristics

If a domain rule says "never drop null diagnosis rows" and the user says
"clean up nulls", the transformer generates code that fills diagnosis
nulls with "unknown" rather than dropping the rows. The user's goal is
interpreted within the boundaries the domain rules define.

---

## Domain YAML file format

Every domain has one YAML file in `domain_rules/`. The structure is fixed.

```yaml
# domain_rules/financial.yaml
domain: financial

# Columns that contain sensitive data — trigger PII handling
sensitive_columns:
  - account_number
  - ssn
  - card_number
  - routing_number
  - iban

# Transformations that must always be applied regardless of user goal
required_transforms:
  - mask_or_hash_sensitive_columns
  - deduplicate_transaction_id
  - validate_amount_is_numeric
  - standardise_dates_to_utc

# Transformations that must never be applied regardless of user goal
forbidden_transforms:
  - drop_rows_with_null_amount
  - divide_or_multiply_amount_columns
  - convert_negative_amounts_to_positive

# Column-specific validation rules
validation_rules:
  amount:
    type: float
    allow_negative: false
    description: "transaction amount — negative values indicate returns, flag do not remove"
  transaction_id:
    type: string
    unique: true
    description: "must be unique — duplicates are data errors"
  transaction_date:
    type: datetime
    timezone: utc
    description: "always normalise to UTC before storing"

# Watermark column for incremental processing
default_watermark_column: transaction_date

# Keywords used for domain detection heuristics
detection_keywords:
  - transaction_id
  - account_number
  - debit
  - credit
  - ledger
  - invoice
  - payment
  - revenue
  - settlement
  - clearance
```

```yaml
# domain_rules/medical.yaml
domain: medical

sensitive_columns:
  - patient_id
  - dob
  - date_of_birth
  - ssn
  - mrn
  - diagnosis
  - medication

required_transforms:
  - mask_patient_id
  - validate_icd_codes_format
  - standardise_dates_to_iso8601
  - flag_missing_diagnosis_as_unknown

forbidden_transforms:
  - drop_rows_with_null_diagnosis
  - drop_rows_with_null_patient_id
  - infer_diagnosis_from_other_columns

validation_rules:
  diagnosis:
    type: string
    allow_null: false
    null_treatment: flag_as_unknown
    description: "missing diagnosis must be flagged as unknown, never dropped"
  icd_code:
    format_regex: "^[A-Z][0-9]{2}\\.?[0-9A-Z]{0,4}$"
    description: "must conform to ICD-10 format"
  patient_id:
    unique: true
    mask_in_output: true
    description: "PII — must be hashed or masked before writing output"

default_watermark_column: admission_date

detection_keywords:
  - patient_id
  - diagnosis
  - icd_code
  - mrn
  - medication
  - dosage
  - prescription
  - clinical
  - admission
  - discharge
```

```yaml
# domain_rules/automotive.yaml
domain: automotive

sensitive_columns: []

required_transforms:
  - validate_vin_format
  - validate_year_range
  - normalise_mileage_to_integer
  - standardise_make_model_casing

forbidden_transforms:
  - drop_rows_with_invalid_vin

validation_rules:
  vin:
    type: string
    length: 17
    checksum_validate: true
    description: "Vehicle Identification Number — 17 chars, must pass checksum"
  year:
    type: integer
    min: 1886
    max_dynamic: current_year
    description: "model year — cannot be before 1886 or after current year"
  mileage:
    type: integer
    min: 0
    description: "odometer reading — always non-negative integer"
  price:
    type: float
    currency_normalise: true
    target_currency: usd
    description: "normalise to USD"

default_watermark_column: listing_date

detection_keywords:
  - vin
  - make
  - model
  - mileage
  - odometer
  - engine
  - transmission
  - dealership
  - carfax
```

```yaml
# domain_rules/employment.yaml
domain: employment

sensitive_columns:
  - salary
  - ssn
  - bank_account
  - performance_rating

required_transforms:
  - standardise_job_title_casing
  - validate_hire_date_before_end_date
  - calculate_tenure_from_dates
  - normalise_salary_to_numeric

forbidden_transforms:
  - divide_salary_columns
  - multiply_salary_columns
  - infer_salary_from_job_title

validation_rules:
  salary:
    type: float
    currency_strip: true
    allow_null: true
    description: "strip currency symbols, cast to float, nulls allowed"
  hire_date:
    type: date
    must_be_before: end_date
    description: "hire date must precede end date if both present"
  job_title:
    type: string
    casing: title_case
    description: "standardise to Title Case"

default_watermark_column: hire_date

detection_keywords:
  - employee_id
  - hire_date
  - salary
  - job_title
  - department
  - performance
  - termination
  - payroll
  - headcount
```

```yaml
# domain_rules/retail.yaml
domain: retail

sensitive_columns:
  - customer_email
  - payment_method_details

required_transforms:
  - validate_sku_or_product_id_present
  - deduplicate_order_line_items
  - normalise_price_to_float
  - flag_return_rows

forbidden_transforms:
  - drop_rows_with_null_quantity

validation_rules:
  quantity:
    type: integer
    min: 0
    negative_treatment: flag_as_return
    description: "negative quantity = return — flag do not remove"
  price:
    type: float
    currency_strip: true
    min: 0
    description: "strip currency symbols, must be non-negative"
  sku:
    type: string
    description: "product identifier — required field"
    allow_null: false

default_watermark_column: order_date

detection_keywords:
  - sku
  - product_id
  - order_id
  - quantity
  - cart
  - checkout
  - refund
  - return
  - inventory
  - discount
```

---

## detect_domain tool implementation

The detection uses two stages. Run stage 1 first — only call the LLM
in stage 2 if stage 1 is ambiguous.

```python
# Stage 1: keyword heuristics
def _keyword_score(column_names: list[str], domain: str) -> float:
    keywords = DOMAIN_KEYWORDS[domain]  # from YAML detection_keywords
    matches = sum(1 for col in column_names
                  if any(kw in col.lower() for kw in keywords))
    return matches / len(keywords) if keywords else 0.0

scores = {domain: _keyword_score(column_names, domain)
          for domain in SUPPORTED_DOMAINS}
best_domain = max(scores, key=scores.get)
best_score = scores[best_domain]

# If one domain scores clearly above others, return it without LLM
if best_score >= 0.3 and (best_score - sorted(scores.values())[-2]) >= 0.15:
    return {"domain": best_domain, "confidence": min(0.95, best_score * 2),
            "method": "keyword_heuristic"}

# Stage 2: LLM call for ambiguous cases
# Pass column names + sample values to Claude and ask it to classify
```

Confidence thresholds and their effect on pipeline behaviour:
- >= 0.85 — proceed silently
- 0.60 – 0.85 — log warning to Langfuse, proceed
- < 0.60 — pause at HITL domain checkpoint, ask human to confirm

---

## How domain_context is used in the transformer

The transformer system prompt includes the full domain context as a
constraint section. The key fields the transformer reads are:

```python
def build_transformer_context(state: PipelineState) -> str:
    ctx = state.get("domain_context", {})
    domain = state.get("domain", "unknown")

    domain_section = f"""
DOMAIN: {domain}
DOMAIN CONFIDENCE: {state.get("domain_confidence", 0)}

REQUIRED TRANSFORMATIONS (must apply regardless of user goal):
{json.dumps(ctx.get("required_transforms", []), indent=2)}

FORBIDDEN TRANSFORMATIONS (must never apply regardless of user goal):
{json.dumps(ctx.get("forbidden_transforms", []), indent=2)}

COLUMN RULES:
{json.dumps(ctx.get("validation_rules", {}), indent=2)}

SENSITIVE COLUMNS (handle with PII masking):
{json.dumps(ctx.get("sensitive_columns", []), indent=2)}
"""
    return domain_section + "\nUSER GOAL:\n" + state["user_goal"]
```

The transformer LLM reads this and generates code that satisfies the
user goal within the domain constraints. If the user goal conflicts with
a forbidden transform, the LLM must acknowledge the conflict in its
reasoning and apply the domain-compliant alternative.

---

## Adding a new domain

1. Create `domain_rules/your_domain.yaml` following the structure above
2. Add the domain name to `SUPPORTED_DOMAINS` list in `domain_tools.py`
3. Add `detection_keywords` to the YAML — these drive stage 1 heuristics
4. No other code changes needed — the tool reads YAML dynamically
