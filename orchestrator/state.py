"""
PipelineState — the single shared state dict threaded through every node.

All fields are Optional except the fixed inputs and retry_count/status.
Agents always return {**state, "new_field": value} — never a partial dict.
"""

from typing import Optional, Literal
from typing_extensions import TypedDict


class PipelineState(TypedDict):
    # ------------------------------------------------------------------
    # Input — set by main.py before the graph is invoked
    # ------------------------------------------------------------------
    run_id: str
    user_goal: str
    source_path: str
    source_type: Literal["csv", "parquet", "postgres", "api"]
    source_table: Optional[str]       # postgres only — table name
    incremental_mode: bool

    # ------------------------------------------------------------------
    # Profiler output
    # ------------------------------------------------------------------
    source_metadata: Optional[dict]
    sample: Optional[list]
    profile: Optional[dict]
    schema: Optional[dict]
    schema_drift: Optional[dict]          # from compare_schemas
    drift_checkpoint_approved: Optional[bool]

    # ------------------------------------------------------------------
    # Domain detection output
    # ------------------------------------------------------------------
    domain: Optional[str]
    domain_confidence: Optional[float]
    domain_context: Optional[dict]        # loaded YAML rules

    # ------------------------------------------------------------------
    # Transformer output
    # ------------------------------------------------------------------
    library_snippets: Optional[list]
    watermark_value: Optional[str]        # incremental high-water mark
    generated_code: Optional[str]
    transformations_applied: Optional[list]
    output_columns: Optional[list]
    output_path: Optional[str]
    rows_input: Optional[int]
    rows_output: Optional[int]
    pipeline_script: Optional[str]        # path to outputs/{run_id}/pipeline.py

    # ------------------------------------------------------------------
    # Quality output
    # ------------------------------------------------------------------
    quality_checks: Optional[list]
    anomaly_summary: Optional[dict]
    anomaly_explanations: Optional[list]
    quality_passed: Optional[bool]
    quality_report_path: Optional[str]
    retry_count: int                      # starts at 0, never None
    failure_reason: Optional[str]

    # ------------------------------------------------------------------
    # Catalogue output
    # ------------------------------------------------------------------
    catalogue_id: Optional[str]
    lineage_graph: Optional[dict]
    mermaid_diagram: Optional[str]
    dbt_model_path: Optional[str]
    dbt_schema_path: Optional[str]
    dbt_tests_path: Optional[str]

    # ------------------------------------------------------------------
    # HITL
    # ------------------------------------------------------------------
    hitl_approved: Optional[bool]
    hitl_edits: Optional[str]

    # ------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------
    status: Literal["running", "success", "failed", "retrying"]
