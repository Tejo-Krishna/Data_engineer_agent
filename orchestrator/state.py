"""
PipelineState — the single shared state dict threaded through every node.

All fields are Optional except the fixed inputs and retry_count/status.
Agents always return {**state, "new_field": value} — never a partial dict.
"""

from typing import Optional, Literal
from typing_extensions import TypedDict


class StateBoundaryError(ValueError):
    """Raised when an agent is entered without required upstream fields."""


def require_profiler_output(state: "PipelineState") -> None:
    """Call at the start of domain agent — fails fast if profiler fields missing.
    Checks only profile and schema — the two fields domain agent actually reads.
    sample is produced by the profiler but is not consumed by domain detection.
    """
    missing = [f for f in ("profile", "schema") if not state.get(f)]
    if missing:
        raise StateBoundaryError(
            f"Domain agent requires profiler output. Missing fields: {missing}"
        )


def require_domain_output(state: "PipelineState") -> None:
    """Call at the start of transformer agent — fails fast if domain fields missing."""
    if not state.get("domain_context"):
        raise StateBoundaryError(
            "Transformer agent requires domain_context. Run domain agent first."
        )


def require_transformer_output(state: "PipelineState") -> None:
    """Call at the start of quality agent — fails fast if transformer fields missing.
    Checks output_path only — what quality agent actually reads.
    generated_code is always set alongside output_path in real runs but is not
    read by the quality agent itself.
    """
    if not state.get("output_path"):
        raise StateBoundaryError(
            "Quality agent requires output_path from transformer. "
            "Run transformer agent first."
        )


def require_quality_output(state: "PipelineState") -> None:
    """Call at the start of catalogue agent — fails fast if quality fields missing."""
    if state.get("quality_passed") is None:
        raise StateBoundaryError(
            "Catalogue agent requires quality_passed from quality agent."
        )


class PipelineState(TypedDict):
    # ------------------------------------------------------------------
    # Input — set by main.py before the graph is invoked
    # ------------------------------------------------------------------
    run_id: str
    run_key: Optional[str]             # content-hash for idempotency checks
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
    large_file: Optional[bool]             # True when source > LARGE_FILE_THRESHOLD_MB
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
    status: Literal["running", "success", "failed", "retrying", "catalogue_pending"]
