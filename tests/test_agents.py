"""
Unit tests for agent functions using MockMCP.

No real MCP tools or LLM calls are made — agents are tested purely for
correct state assembly, tool call ordering, and guard logic.

Transformer HITL is bypassed by patching hitl_code_checkpoint to return
an approved result immediately.

Run with:
    pytest tests/test_agents.py -v
"""

from unittest.mock import AsyncMock, patch

import pytest

from conftest import MockMCP, MockLogger


# ---------------------------------------------------------------------------
# Fake Anthropic response helpers — make ReAct agent tests deterministic
# without real LLM calls (the real call_llm_with_tools is patched out).
# ---------------------------------------------------------------------------

class _FakeBlock:
    """Minimal tool_use content block matching what extract_tool_calls expects."""
    type = "tool_use"

    def __init__(self, name: str, inp: dict, bid: str | None = None):
        self.id = bid or f"tc_{name}"
        self.name = name
        self.input = inp


class _FakeResp:
    """Minimal Anthropic Message response for testing."""

    def __init__(self, tools: list[tuple[str, dict]] | None = None):
        if tools:
            self.stop_reason = "tool_use"
            self.content = [_FakeBlock(name, inp) for name, inp in tools]
        else:
            self.stop_reason = "end_turn"
            self.content = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> dict:
    """Minimal valid PipelineState for agent tests."""
    state = {
        "run_id": "test-run-001",
        "user_goal": "clean dates and dedup",
        "source_path": "sample_data/sales_raw.csv",
        "source_type": "csv",
        "source_table": None,
        "incremental_mode": False,
        "source_metadata": None,
        "sample": None,
        "profile": None,
        "schema": None,
        "schema_drift": None,
        "drift_checkpoint_approved": None,
        "domain": None,
        "domain_confidence": None,
        "domain_context": None,
        "library_snippets": None,
        "watermark_value": None,
        "generated_code": None,
        "transformations_applied": None,
        "output_columns": None,
        "output_path": None,
        "rows_input": None,
        "rows_output": None,
        "pipeline_script": None,
        "quality_checks": None,
        "anomaly_summary": None,
        "anomaly_explanations": None,
        "quality_passed": None,
        "quality_report_path": None,
        "retry_count": 0,
        "failure_reason": None,
        "catalogue_id": None,
        "lineage_graph": None,
        "mermaid_diagram": None,
        "dbt_model_path": None,
        "dbt_schema_path": None,
        "dbt_tests_path": None,
        "hitl_approved": None,
        "hitl_edits": None,
        "status": "running",
    }
    state.update(overrides)
    return state


def _make_config(mcp, logger=None):
    return {"configurable": {"mcp": mcp, "logger": logger or MockLogger()}}


# ---------------------------------------------------------------------------
# Profiler agent
# ---------------------------------------------------------------------------


FAKE_METADATA = {
    "file_path": "sample_data/sales_raw.csv",
    "row_count": 500,
    "column_names": ["transaction_id", "quantity", "unit_price_gbp"],
    "file_size_mb": 0.05,
}

FAKE_SAMPLE = [
    {"transaction_id": "txn-001", "quantity": 3, "unit_price_gbp": "£10.50"}
]

FAKE_PROFILE = {
    "total_rows": 500,
    "duplicate_row_count": 5,
    "transaction_id": {
        "dtype": "VARCHAR", "null_rate": 0.0, "null_count": 0,
        "unique_count": 500, "min": None, "max": None, "mean": None,
        "top_5_values": [], "sample_values": [],
    },
    "quantity": {
        "dtype": "INTEGER", "null_rate": 0.0, "null_count": 0,
        "unique_count": 10, "min": "1", "max": "10", "mean": 5.0,
        "top_5_values": [], "sample_values": [],
    },
    "unit_price_gbp": {
        "dtype": "VARCHAR", "null_rate": 0.0, "null_count": 0,
        "unique_count": 500, "min": None, "max": None, "mean": None,
        "top_5_values": [], "sample_values": [],
    },
}

FAKE_SCHEMA = {
    "transaction_id": {"inferred_type": "VARCHAR", "needs_cast": False, "suggested_cast": None, "nullable": False},
    "quantity": {"inferred_type": "INTEGER", "needs_cast": False, "suggested_cast": None, "nullable": False},
    "unit_price_gbp": {"inferred_type": "DOUBLE", "needs_cast": True, "suggested_cast": "strip_currency_prefix_and_cast_float", "nullable": False},
}

FAKE_DRIFT = {
    "has_drift": False,
    "no_prior_run": True,
    "added_columns": [],
    "dropped_columns": [],
    "type_changes": [],
    "drift_severity": "none",
}


@pytest.mark.asyncio
async def test_profiler_agent_writes_all_state_fields():
    from agents.profiler_agent import run_profiler_agent

    mcp = MockMCP({
        "connect_csv": FAKE_METADATA,
        "sample_data": {"sample": FAKE_SAMPLE, "actual_sample_size": 1},
        "compute_profile": FAKE_PROFILE,
        "detect_schema": FAKE_SCHEMA,
        "compare_schemas": FAKE_DRIFT,
    })

    result = await run_profiler_agent(_base_state(), _make_config(mcp))

    assert result["source_metadata"] == FAKE_METADATA
    assert result["profile"] == FAKE_PROFILE
    assert result["schema"] == FAKE_SCHEMA
    assert result["schema_drift"] == FAKE_DRIFT
    assert isinstance(result["sample"], list)


@pytest.mark.asyncio
async def test_profiler_agent_tool_call_order():
    from agents.profiler_agent import run_profiler_agent

    mcp = MockMCP({
        "connect_csv": FAKE_METADATA,
        "sample_data": {"sample": FAKE_SAMPLE, "actual_sample_size": 1},
        "compute_profile": FAKE_PROFILE,
        "detect_schema": FAKE_SCHEMA,
        "compare_schemas": FAKE_DRIFT,
    })

    await run_profiler_agent(_base_state(), _make_config(mcp))

    assert mcp.called_tools() == [
        "connect_csv", "sample_data", "compute_profile", "detect_schema", "compare_schemas"
    ]


@pytest.mark.asyncio
async def test_profiler_agent_caps_sample_in_state():
    from agents.profiler_agent import run_profiler_agent

    big_sample = [{"id": i} for i in range(500)]
    mcp = MockMCP({
        "connect_csv": FAKE_METADATA,
        "sample_data": {"sample": big_sample, "actual_sample_size": 500},
        "compute_profile": FAKE_PROFILE,
        "detect_schema": FAKE_SCHEMA,
        "compare_schemas": FAKE_DRIFT,
    })

    result = await run_profiler_agent(_base_state(), _make_config(mcp))

    assert len(result["sample"]) <= 10


# ---------------------------------------------------------------------------
# Domain agent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_domain_agent_writes_domain_fields():
    from agents.domain_agent import run_domain_agent

    mcp = MockMCP({
        "detect_domain": {"domain": "retail", "confidence": 0.95, "method": "keyword_heuristic"},
        "load_domain_rules": {"rules": {
            "required_transforms": ["title_case_country"],
            "forbidden_transforms": [],
            "sensitive_columns": [],
            "validation_rules": {},
        }},
    })

    state = _base_state(
        schema=FAKE_SCHEMA,
        profile=FAKE_PROFILE,
    )

    result = await run_domain_agent(state, _make_config(mcp))

    assert result["domain"] == "retail"
    assert result["domain_confidence"] == pytest.approx(0.95)
    assert "required_transforms" in result["domain_context"]


@pytest.mark.asyncio
async def test_domain_agent_calls_both_tools():
    from agents.domain_agent import run_domain_agent

    mcp = MockMCP({
        "detect_domain": {"domain": "retail", "confidence": 0.9, "method": "keyword_heuristic"},
        "load_domain_rules": {"rules": {}},
    })

    await run_domain_agent(
        _base_state(schema=FAKE_SCHEMA, profile=FAKE_PROFILE),
        _make_config(mcp),
    )

    assert mcp.called_tools() == ["detect_domain", "load_domain_rules"]


# ---------------------------------------------------------------------------
# Transformer agent
# ---------------------------------------------------------------------------

FAKE_GENERATED_CODE = "import os\nprint('ROWS_IN: 500')\nprint('ROWS_OUT: 485')"

FAKE_HITL_RESULT = {
    "approved": True,
    "code": FAKE_GENERATED_CODE,
    "nlp_instruction": None,
    "needs_refinement": False,
    "revised_code": None,
}

FAKE_EXECUTE_RESULT = {
    "success": True,
    "stdout": "ROWS_IN: 500\nROWS_OUT: 485",
    "stderr": "",
    "rows_input": 500,
    "rows_output": 485,
    "execution_time_ms": 1200,
}

FAKE_WRITE_RESULT = {
    "output_path": "outputs/test-run-001/sales_cleaned.parquet",
    "row_count": 485,
    "file_size_mb": 0.05,
    "columns": ["transaction_id", "quantity"],
    "schema": {"transaction_id": "VARCHAR", "quantity": "INTEGER"},
}


@pytest.mark.asyncio
async def test_transformer_agent_success_path():
    from agents.transformer_agent import run_transformer_agent

    mcp = MockMCP({
        "search_transform_library": {"snippets": []},
        "generate_transform_code": {
            "code": FAKE_GENERATED_CODE,
            "transformations_applied": ["dedup", "parse_dates"],
            "output_columns": ["transaction_id", "quantity"],
        },
        "execute_code": FAKE_EXECUTE_RESULT,
        "write_dataset": FAKE_WRITE_RESULT,
    })

    state = _base_state(
        schema=FAKE_SCHEMA,
        profile=FAKE_PROFILE,
        domain="retail",
        domain_context={"required_transforms": [], "forbidden_transforms": [], "sensitive_columns": []},
    )

    # Deterministic LLM sequence: library → generate → execute → write → stop
    _llm = AsyncMock(side_effect=[
        _FakeResp([("search_transform_library", {"user_goal": "clean dates and dedup"})]),
        _FakeResp([("generate_transform_code", {"user_goal": "clean dates and dedup"})]),
        _FakeResp([("execute_code", {"code": FAKE_GENERATED_CODE, "input_path": "/data/in.csv", "output_path": "/tmp/out.parquet"})]),
        _FakeResp([("write_dataset", {"data_path": "/tmp/out.parquet", "output_name": "output"})]),
        _FakeResp(),
    ])

    with patch("agents.transformer_agent.call_llm_with_tools", _llm):
        with patch("agents.transformer_agent.hitl_code_checkpoint", AsyncMock(return_value=FAKE_HITL_RESULT)):
            result = await run_transformer_agent(state, _make_config(mcp))

    assert result["output_path"] == "outputs/test-run-001/sales_cleaned.parquet"
    assert result["rows_input"] == 500
    assert result["rows_output"] == 485
    assert result["hitl_approved"] is True


@pytest.mark.asyncio
async def test_transformer_agent_skips_hitl_on_retry():
    from agents.transformer_agent import run_transformer_agent

    mcp = MockMCP({
        "search_transform_library": {"snippets": []},
        "generate_transform_code": {
            "code": FAKE_GENERATED_CODE,
            "transformations_applied": ["dedup"],
            "output_columns": ["transaction_id"],
        },
        "execute_code": FAKE_EXECUTE_RESULT,
        "write_dataset": FAKE_WRITE_RESULT,
    })

    state = _base_state(
        retry_count=1,
        failure_reason="Execution error: missing column",
        schema=FAKE_SCHEMA,
        profile=FAKE_PROFILE,
        domain="retail",
        domain_context={"required_transforms": [], "forbidden_transforms": [], "sensitive_columns": []},
    )

    hitl_mock = AsyncMock(return_value=FAKE_HITL_RESULT)
    _llm = AsyncMock(side_effect=[
        _FakeResp([("search_transform_library", {})]),
        _FakeResp([("generate_transform_code", {})]),
        _FakeResp([("execute_code", {"code": FAKE_GENERATED_CODE, "input_path": "/data/in.csv", "output_path": "/tmp/out.parquet"})]),
        _FakeResp([("write_dataset", {"data_path": "/tmp/out.parquet", "output_name": "output"})]),
        _FakeResp(),
    ])

    with patch("agents.transformer_agent.call_llm_with_tools", _llm):
        with patch("agents.transformer_agent.hitl_code_checkpoint", hitl_mock):
            await run_transformer_agent(state, _make_config(mcp))

    # HITL must NOT be called on retry runs
    hitl_mock.assert_not_called()


@pytest.mark.asyncio
async def test_transformer_agent_rejected_hitl_returns_failed():
    from agents.transformer_agent import run_transformer_agent

    mcp = MockMCP({
        "search_transform_library": {"snippets": []},
        "generate_transform_code": {
            "code": FAKE_GENERATED_CODE,
            "transformations_applied": [],
            "output_columns": [],
        },
        "execute_code": FAKE_EXECUTE_RESULT,
        "write_dataset": FAKE_WRITE_RESULT,
    })

    state = _base_state(
        schema=FAKE_SCHEMA,
        profile=FAKE_PROFILE,
        domain="retail",
        domain_context={"required_transforms": [], "forbidden_transforms": [], "sensitive_columns": []},
    )

    rejected_hitl = AsyncMock(side_effect=ValueError("HITL code checkpoint rejected"))
    _llm = AsyncMock(side_effect=[
        _FakeResp([("search_transform_library", {})]),
        _FakeResp([("generate_transform_code", {})]),
        _FakeResp(),
    ])

    with patch("agents.transformer_agent.call_llm_with_tools", _llm):
        with patch("agents.transformer_agent.hitl_code_checkpoint", rejected_hitl):
            result = await run_transformer_agent(state, _make_config(mcp))

    assert result["status"] == "failed"


# ---------------------------------------------------------------------------
# Quality agent
# ---------------------------------------------------------------------------

FAKE_QUALITY_CHECKS = [
    {"check_name": "row_count", "check_type": "volume", "column_name": None,
     "passed": True, "severity": "critical", "detail": "500 rows OK"},
    {"check_name": "null_rate", "check_type": "completeness", "column_name": "quantity",
     "passed": True, "severity": "error", "detail": "0% nulls"},
]

FAKE_ANOMALY_RESULT = {
    "anomaly_count": 2,
    "anomaly_rate": 0.004,
    "anomalous_rows": [{"transaction_id": "txn-bad", "quantity": 9999}],
}

FAKE_EXPLANATIONS = {
    "explanations": [{"column": "quantity", "explanation": "Extreme outlier detected."}],
    "overall_summary": "2 anomalies found.",
}

FAKE_QUALITY_REPORT = {
    "json_path": "outputs/test-run-001/quality_report.json",
    "markdown_path": "outputs/test-run-001/quality_report.md",
    "overall_status": "passed",
}


@pytest.mark.asyncio
async def test_quality_agent_success_sets_quality_passed():
    from agents.quality_agent import run_quality_agent

    mcp = MockMCP({
        "run_quality_checks": {
            "checks": FAKE_QUALITY_CHECKS,
            "overall_passed": True,
            "critical_failures": [],
        },
        "detect_anomalies": FAKE_ANOMALY_RESULT,
        "explain_anomalies": FAKE_EXPLANATIONS,
        "write_quality_report": FAKE_QUALITY_REPORT,
    })

    state = _base_state(
        output_path="outputs/test-run-001/sales_cleaned.parquet",
        profile=FAKE_PROFILE,
        schema=FAKE_SCHEMA,
        transformations_applied=["dedup"],
        domain="retail",
        domain_context={"validation_rules": {}},
    )

    _llm = AsyncMock(side_effect=[
        _FakeResp([("run_quality_checks", {"output_path": "outputs/test-run-001/sales_cleaned.parquet"})]),
        _FakeResp([("detect_anomalies", {"target_columns": ["quantity"]})]),
        _FakeResp([("explain_anomalies", {"anomaly_summary": {}, "profile": {}})]),
        _FakeResp([("write_quality_report", {"pipeline_run_id": "test-run-001"})]),
        _FakeResp(),
    ])

    with patch("agents.quality_agent.call_llm_with_tools", _llm):
        result = await run_quality_agent(state, _make_config(mcp))

    assert result["quality_passed"] is True
    assert result["status"] == "success"
    assert result["quality_report_path"] == FAKE_QUALITY_REPORT["json_path"]


@pytest.mark.asyncio
async def test_quality_agent_failed_checks_set_retrying():
    from agents.quality_agent import run_quality_agent

    failing_checks = [
        {"check_name": "row_count", "check_type": "volume", "column_name": None,
         "passed": False, "severity": "critical", "detail": "Too few rows"},
    ]

    mcp = MockMCP({
        "run_quality_checks": {
            "checks": failing_checks,
            "overall_passed": False,
            "critical_failures": ["row_count"],
        },
        "detect_anomalies": {"anomaly_count": 0, "anomaly_rate": 0.0, "anomalous_rows": []},
        "write_quality_report": FAKE_QUALITY_REPORT,
    })

    state = _base_state(
        output_path="outputs/test-run-001/sales_cleaned.parquet",
        profile=FAKE_PROFILE,
        schema=FAKE_SCHEMA,
        transformations_applied=[],
        domain="retail",
        domain_context={"validation_rules": {}},
    )

    _llm = AsyncMock(side_effect=[
        _FakeResp([("run_quality_checks", {"output_path": "outputs/test-run-001/sales_cleaned.parquet"})]),
        _FakeResp([("write_quality_report", {"pipeline_run_id": "test-run-001"})]),
        _FakeResp(),
    ])

    with patch("agents.quality_agent.call_llm_with_tools", _llm):
        result = await run_quality_agent(state, _make_config(mcp))

    assert result["quality_passed"] is False
    assert result["status"] == "retrying"
    assert result["retry_count"] == 1
    assert result["failure_reason"] is not None


@pytest.mark.asyncio
async def test_quality_agent_blocks_save_to_library_when_failed():
    from agents.quality_agent import run_quality_agent

    mcp = MockMCP({
        "run_quality_checks": {
            "checks": [],
            "overall_passed": False,
            "critical_failures": ["row_count"],
        },
        "save_to_library": {"saved": True, "library_id": "lib-001"},
        "detect_anomalies": {"anomaly_count": 0, "anomaly_rate": 0.0, "anomalous_rows": []},
        "write_quality_report": FAKE_QUALITY_REPORT,
    })

    state = _base_state(
        output_path="outputs/test-run-001/sales_cleaned.parquet",
        profile=FAKE_PROFILE,
        schema=FAKE_SCHEMA,
        transformations_applied=[],
        domain="retail",
        domain_context={"validation_rules": {}},
    )

    # LLM incorrectly tries save_to_library — agent guard must block it
    _llm = AsyncMock(side_effect=[
        _FakeResp([("run_quality_checks", {"output_path": "outputs/test-run-001/sales_cleaned.parquet"})]),
        _FakeResp([("save_to_library", {"code": "...", "user_goal": "clean"})]),
        _FakeResp([("write_quality_report", {"pipeline_run_id": "test-run-001"})]),
        _FakeResp(),
    ])

    with patch("agents.quality_agent.call_llm_with_tools", _llm):
        await run_quality_agent(state, _make_config(mcp))

    # The guard intercepts save_to_library and returns a fake result without
    # calling mcp.call — so it must NOT appear in mcp.calls
    assert "save_to_library" not in mcp.called_tools()


# ---------------------------------------------------------------------------
# Catalogue agent
# ---------------------------------------------------------------------------

FAKE_CATALOGUE_ENTRY = {
    "catalogue_id": "cat-001",
    "column_descriptions": {"transaction_id": "Unique transaction identifier."},
}

FAKE_LINEAGE = {
    "lineage_graph": {"edges": [{"source": "transaction_id", "target": "transaction_id"}]},
    "mermaid_diagram": "flowchart LR\n  transaction_id --> transaction_id",
}

FAKE_DBT_MODEL = {"sql_content": "SELECT * FROM source", "file_path": "outputs/test-run-001/dbt/models/pipeline_test.sql"}
FAKE_DBT_SCHEMA = {"yml_content": "models:\n  - name: test", "file_path": "outputs/test-run-001/dbt/models/schema.yml"}
FAKE_DBT_TESTS = {"tests_yml_content": "tests:\n  - unique", "file_path": "outputs/test-run-001/dbt/models/schema_tests.yml"}


@pytest.mark.asyncio
async def test_catalogue_agent_writes_all_state_fields():
    from agents.catalogue_agent import run_catalogue_agent

    mcp = MockMCP({
        "write_catalogue_entry": FAKE_CATALOGUE_ENTRY,
        "generate_lineage_graph": FAKE_LINEAGE,
        "generate_dbt_model": FAKE_DBT_MODEL,
        "generate_dbt_schema_yml": FAKE_DBT_SCHEMA,
        "generate_dbt_tests": FAKE_DBT_TESTS,
    })

    state = _base_state(
        status="success",
        schema=FAKE_SCHEMA,
        output_path="outputs/test-run-001/sales_cleaned.parquet",
        rows_output=485,
        transformations_applied=["dedup"],
        quality_checks=FAKE_QUALITY_CHECKS,
    )

    result = await run_catalogue_agent(state, _make_config(mcp))

    assert result["catalogue_id"] == "cat-001"
    assert result["mermaid_diagram"] is not None
    assert result["dbt_model_path"] == FAKE_DBT_MODEL["file_path"]
    assert result["dbt_schema_path"] == FAKE_DBT_SCHEMA["file_path"]
    assert result["dbt_tests_path"] == FAKE_DBT_TESTS["file_path"]


@pytest.mark.asyncio
async def test_catalogue_agent_raises_if_not_success():
    from agents.catalogue_agent import run_catalogue_agent

    mcp = MockMCP({})
    state = _base_state(status="retrying")

    with pytest.raises(ValueError, match="success"):
        await run_catalogue_agent(state, _make_config(mcp))


@pytest.mark.asyncio
async def test_catalogue_agent_raises_if_no_schema():
    from agents.catalogue_agent import run_catalogue_agent

    mcp = MockMCP({})
    state = _base_state(status="success", schema={}, output_path="some/path.parquet")

    with pytest.raises(ValueError, match="schema"):
        await run_catalogue_agent(state, _make_config(mcp))
