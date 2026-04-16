"""
Tests for architectural improvements 3.1 – 3.8.

Each test is self-contained and does NOT require Docker, Postgres, or
Anthropic API credits. All LLM calls are mocked with structured tool_use
responses matching the real API shape.

Run with:
    pytest tests/test_improvements.py -v
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest


# ===========================================================================
# 3.6 — TOOLS registry: each module exports a TOOLS dict; server TOOL_HANDLERS
#       is built by merging them.
# ===========================================================================

class TestToolsRegistry:
    def test_source_tools_exports_tools_dict(self):
        from mcp_server.tools.source_tools import TOOLS
        expected = {"connect_csv", "connect_postgres", "connect_api", "detect_new_rows"}
        assert set(TOOLS.keys()) == expected
        for name, fn in TOOLS.items():
            assert callable(fn), f"{name} must be callable"

    def test_profiling_tools_exports_tools_dict(self):
        from mcp_server.tools.profiling_tools import TOOLS
        expected = {"sample_data", "compute_profile", "detect_schema", "compare_schemas"}
        assert set(TOOLS.keys()) == expected

    def test_domain_tools_exports_tools_dict(self):
        from mcp_server.tools.domain_tools import TOOLS
        expected = {"detect_domain", "load_domain_rules"}
        assert set(TOOLS.keys()) == expected

    def test_transform_tools_exports_tools_dict(self):
        from mcp_server.tools.transform_tools import TOOLS
        expected = {
            "generate_transform_code", "refine_transform_code",
            "execute_code", "write_dataset", "verify_transform_intent",
        }
        assert set(TOOLS.keys()) == expected

    def test_quality_tools_exports_tools_dict(self):
        from mcp_server.tools.quality_tools import TOOLS
        expected = {
            "run_quality_checks", "detect_anomalies",
            "explain_anomalies", "write_quality_report",
        }
        assert set(TOOLS.keys()) == expected

    def test_library_tools_exports_tools_dict(self):
        from mcp_server.tools.library_tools import TOOLS
        expected = {"search_transform_library", "save_to_library", "generate_dbt_schema_yml"}
        assert set(TOOLS.keys()) == expected

    def test_catalogue_tools_exports_tools_dict(self):
        from mcp_server.tools.catalogue_tools import TOOLS
        expected = {
            "write_catalogue_entry", "generate_lineage_graph",
            "generate_dbt_model", "read_catalogue", "generate_dbt_tests",
        }
        assert set(TOOLS.keys()) == expected

    def test_server_tool_handlers_built_from_modules(self):
        from mcp_server.server import TOOL_HANDLERS
        # All 25 tools (including verify_transform_intent) must be present
        assert "connect_csv" in TOOL_HANDLERS
        assert "verify_transform_intent" in TOOL_HANDLERS
        assert "generate_dbt_tests" in TOOL_HANDLERS
        assert len(TOOL_HANDLERS) >= 25

    def test_server_tool_handlers_no_duplicate_names(self):
        from mcp_server.server import TOOL_HANDLERS, TOOL_DEFINITIONS
        handler_names = set(TOOL_HANDLERS.keys())
        definition_names = {t.name for t in TOOL_DEFINITIONS}
        # Every TOOL_DEFINITION must have a matching handler
        for name in definition_names:
            assert name in handler_names, f"Missing handler for tool: {name}"


# ===========================================================================
# 3.8 — State boundary validators: raise StateBoundaryError on missing fields
# ===========================================================================

class TestStateBoundaryValidators:
    def _base_state(self):
        return {
            "run_id": "test-run",
            "user_goal": "clean data",
            "source_path": "/tmp/data.csv",
            "source_type": "csv",
            "incremental_mode": False,
            "retry_count": 0,
            "status": "running",
        }

    def test_require_profiler_output_passes_when_fields_present(self):
        from orchestrator.state import require_profiler_output
        state = {**self._base_state(),
                 "profile": {"col_a": {}}, "schema": {"col_a": {}}, "sample": [{"col_a": 1}]}
        require_profiler_output(state)  # must not raise

    def test_require_profiler_output_raises_when_profile_missing(self):
        from orchestrator.state import require_profiler_output, StateBoundaryError
        state = {**self._base_state(), "schema": {"col_a": {}}, "sample": [{}]}
        with pytest.raises(StateBoundaryError, match="profile"):
            require_profiler_output(state)

    def test_require_profiler_output_raises_when_schema_missing(self):
        from orchestrator.state import require_profiler_output, StateBoundaryError
        state = {**self._base_state(), "profile": {"col_a": {}}, "sample": [{}]}
        with pytest.raises(StateBoundaryError, match="schema"):
            require_profiler_output(state)

    def test_require_domain_output_passes_when_domain_context_present(self):
        from orchestrator.state import require_domain_output
        state = {**self._base_state(), "domain_context": {"domain": "retail"}}
        require_domain_output(state)  # must not raise

    def test_require_domain_output_raises_when_missing(self):
        from orchestrator.state import require_domain_output, StateBoundaryError
        with pytest.raises(StateBoundaryError, match="domain_context"):
            require_domain_output(self._base_state())

    def test_require_transformer_output_passes_when_fields_present(self):
        from orchestrator.state import require_transformer_output
        state = {**self._base_state(),
                 "output_path": "/tmp/out.parquet", "generated_code": "import os"}
        require_transformer_output(state)

    def test_require_transformer_output_raises_when_output_path_missing(self):
        from orchestrator.state import require_transformer_output, StateBoundaryError
        state = {**self._base_state(), "generated_code": "import os"}
        with pytest.raises(StateBoundaryError, match="output_path"):
            require_transformer_output(state)

    def test_require_quality_output_passes_when_quality_passed_set(self):
        from orchestrator.state import require_quality_output
        for val in (True, False):
            state = {**self._base_state(), "quality_passed": val}
            require_quality_output(state)

    def test_require_quality_output_raises_when_quality_passed_none(self):
        from orchestrator.state import require_quality_output, StateBoundaryError
        state = {**self._base_state(), "quality_passed": None}
        with pytest.raises(StateBoundaryError, match="quality_passed"):
            require_quality_output(state)


# ===========================================================================
# 3.3 — Targeted repair: _try_targeted_repair returns result on success,
#        None on refine/execute failure.
# ===========================================================================

class TestTargetedRepair:
    def _base_state(self):
        return {
            "run_id": "repair-run",
            "user_goal": "clean dates",
            "source_path": "/tmp/data.csv",
            "source_type": "csv",
            "retry_count": 1,
            "status": "retrying",
            "failure_reason": "NameError: name 'pd' is not defined",
            "profile": {},
            "domain_context": {},
            "schema": {},
        }

    @pytest.mark.asyncio
    async def test_targeted_repair_succeeds_and_returns_result(self, tmp_path):
        from agents.transformer_agent import _try_targeted_repair

        # Write a tiny parquet so write_dataset has something to read
        out_parquet = tmp_path / "out.parquet"
        pd.DataFrame({"a": [1, 2]}).to_parquet(out_parquet)

        mock_mcp = AsyncMock()
        mock_mcp.call = AsyncMock(side_effect=[
            # refine_transform_code
            {"revised_code": "import os\nprint('ROWS_IN: 2')\nprint('ROWS_OUT: 2')", "changes_summary": ["fixed import"]},
            # execute_code
            {"success": True, "stdout": "ROWS_IN: 2\nROWS_OUT: 2", "stderr": "", "rows_input": 2, "rows_output": 2, "execution_time_ms": 100},
            # write_dataset
            {"output_path": str(tmp_path / "final.parquet"), "row_count": 2},
        ])
        mock_logger = MagicMock()
        mock_logger.tool_call = MagicMock()
        mock_span = MagicMock()

        result = await _try_targeted_repair(
            generated_code="import pandas as pd\nprint('ROWS_IN: 2')",
            failure_reason="NameError: name 'pd' is not defined",
            state=self._base_state(),
            mcp=mock_mcp,
            logger=mock_logger,
            span=mock_span,
        )

        assert result is not None
        assert result["output_path"] == str(tmp_path / "final.parquet")
        assert result["rows_input"] == 2

    @pytest.mark.asyncio
    async def test_targeted_repair_returns_none_when_execute_fails(self):
        from agents.transformer_agent import _try_targeted_repair

        mock_mcp = AsyncMock()
        mock_mcp.call = AsyncMock(side_effect=[
            {"revised_code": "import os", "changes_summary": []},
            {"success": False, "stdout": "", "stderr": "SyntaxError", "rows_input": None, "rows_output": None, "execution_time_ms": 0},
        ])
        mock_logger = MagicMock()
        mock_logger.tool_call = MagicMock()

        result = await _try_targeted_repair(
            generated_code="bad code",
            failure_reason="SyntaxError",
            state=self._base_state(),
            mcp=mock_mcp,
            logger=mock_logger,
            span=MagicMock(),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_targeted_repair_returns_none_when_refine_raises(self):
        from agents.transformer_agent import _try_targeted_repair

        mock_mcp = AsyncMock()
        mock_mcp.call = AsyncMock(side_effect=RuntimeError("API error"))
        mock_logger = MagicMock()
        mock_logger.tool_call = MagicMock()

        result = await _try_targeted_repair(
            generated_code="some code",
            failure_reason="error",
            state=self._base_state(),
            mcp=mock_mcp,
            logger=mock_logger,
            span=MagicMock(),
        )
        assert result is None


# ===========================================================================
# 3.5 — Semantic verification: verify_transform_intent uses Haiku and returns
#        intent_matched, confidence, issues.
# ===========================================================================

class TestSemanticVerification:
    def _make_fake_haiku_response(self, intent_matched: bool, confidence: float, issues: list):
        """Build a mock Anthropic response with tool_use block."""
        tool_use_block = MagicMock()
        tool_use_block.type = "tool_use"
        tool_use_block.input = {
            "intent_matched": intent_matched,
            "confidence": confidence,
            "issues": issues,
        }
        resp = MagicMock()
        resp.content = [tool_use_block]
        return resp

    @pytest.mark.asyncio
    async def test_verify_intent_matched_returns_true(self, tmp_path):
        from mcp_server.tools.transform_tools import verify_transform_intent

        # Write tiny input and output parquets
        input_path = tmp_path / "input.parquet"
        output_path = tmp_path / "output.parquet"
        pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "price": ["£10", "£20"]}).to_parquet(input_path)
        pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "price": [10.0, 20.0]}).to_parquet(output_path)

        fake_resp = self._make_fake_haiku_response(True, 0.95, [])
        with patch("anthropic.AsyncAnthropic") as MockClient:
            MockClient.return_value.messages.create = AsyncMock(return_value=fake_resp)
            result = await verify_transform_intent(
                user_goal="parse dates and convert GBP to float",
                input_path=str(input_path),
                output_path=str(output_path),
                transformations_applied=["parse_dates", "strip_currency"],
            )

        assert result["intent_matched"] is True
        assert result["confidence"] == 0.95
        assert result["issues"] == []

    @pytest.mark.asyncio
    async def test_verify_intent_not_matched_returns_issues(self, tmp_path):
        from mcp_server.tools.transform_tools import verify_transform_intent

        input_path = tmp_path / "input.parquet"
        output_path = tmp_path / "output.parquet"
        pd.DataFrame({"col": [1, 2]}).to_parquet(input_path)
        pd.DataFrame({"col": [1, 2]}).to_parquet(output_path)

        fake_resp = self._make_fake_haiku_response(False, 0.8, ["dates were not parsed"])
        with patch("anthropic.AsyncAnthropic") as MockClient:
            MockClient.return_value.messages.create = AsyncMock(return_value=fake_resp)
            result = await verify_transform_intent(
                user_goal="parse dates",
                input_path=str(input_path),
                output_path=str(output_path),
                transformations_applied=[],
            )

        assert result["intent_matched"] is False
        assert "dates were not parsed" in result["issues"]

    @pytest.mark.asyncio
    async def test_verify_uses_haiku_model(self, tmp_path):
        """Ensure Haiku (not Sonnet) is used for the verification call."""
        from mcp_server.tools.transform_tools import verify_transform_intent, _VERIFY_MODEL

        assert "haiku" in _VERIFY_MODEL.lower(), (
            f"verify_transform_intent must use Haiku, got: {_VERIFY_MODEL}"
        )

        input_path = tmp_path / "input.parquet"
        output_path = tmp_path / "output.parquet"
        pd.DataFrame({"x": [1]}).to_parquet(input_path)
        pd.DataFrame({"x": [1]}).to_parquet(output_path)

        captured_model = {}
        fake_resp = self._make_fake_haiku_response(True, 0.9, [])

        async def _mock_create(**kwargs):
            captured_model["model"] = kwargs.get("model")
            return fake_resp

        with patch("anthropic.AsyncAnthropic") as MockClient:
            MockClient.return_value.messages.create = AsyncMock(side_effect=_mock_create)
            await verify_transform_intent("goal", str(input_path), str(output_path), [])

        assert captured_model["model"] == _VERIFY_MODEL


# ===========================================================================
# 3.1 — Docker sandbox: run_sandboxed uses subprocess when DOCKER_SANDBOX_IMAGE
#        is unset; switches to Docker when set + docker is available.
# ===========================================================================

class TestDockerSandbox:
    def test_subprocess_path_executes_valid_code(self, tmp_path):
        from sandbox.executor import run_sandboxed

        input_csv = tmp_path / "input.csv"
        input_csv.write_text("a,b\n1,2\n3,4\n")
        output_path = tmp_path / "out.parquet"

        code = (
            "import os, pandas as pd\n"
            "df = pd.read_csv(os.environ['INPUT_PATH'])\n"
            "print(f'ROWS_IN: {len(df)}')\n"
            "df.to_parquet(os.environ['OUTPUT_PATH'], index=False)\n"
            "print(f'ROWS_OUT: {len(df)}')\n"
        )

        result = run_sandboxed(code, str(input_csv), str(output_path), timeout=30)
        assert result["success"] is True
        assert result["rows_input"] == 2
        assert result["rows_output"] == 2
        assert result["execution_time_ms"] > 0

    def test_subprocess_path_captures_stderr_on_failure(self, tmp_path):
        from sandbox.executor import run_sandboxed

        code = "raise ValueError('intentional test failure')"
        result = run_sandboxed(code, "/dev/null", str(tmp_path / "out.parquet"))
        assert result["success"] is False
        assert "intentional test failure" in result["stderr"]

    def test_subprocess_path_handles_timeout(self, tmp_path):
        from sandbox.executor import run_sandboxed

        code = "import time; time.sleep(60)"
        result = run_sandboxed(code, "/dev/null", str(tmp_path / "out.parquet"), timeout=1)
        assert result["success"] is False
        assert "timed out" in result["stderr"].lower()

    def test_docker_env_var_unset_uses_subprocess(self, tmp_path, monkeypatch):
        """Without DOCKER_SANDBOX_IMAGE, run_sandboxed must never call docker."""
        monkeypatch.delenv("DOCKER_SANDBOX_IMAGE", raising=False)
        from sandbox import executor

        with patch("sandbox.executor._run_docker") as mock_docker, \
             patch("sandbox.executor._run_subprocess", wraps=executor._run_subprocess) as mock_sub:
            executor.run_sandboxed(
                "import os; print('ROWS_IN: 0'); print('ROWS_OUT: 0')",
                "/dev/null", str(tmp_path / "out.parquet"),
            )
            mock_docker.assert_not_called()
            mock_sub.assert_called_once()

    def test_docker_env_var_set_but_docker_unavailable_falls_back(self, tmp_path, monkeypatch):
        """If Docker binary is missing, fall back to subprocess silently."""
        monkeypatch.setenv("DOCKER_SANDBOX_IMAGE", "data-agent-sandbox:latest")
        from sandbox import executor

        with patch("shutil.which", return_value=None), \
             patch("sandbox.executor._run_subprocess", wraps=executor._run_subprocess) as mock_sub:
            result = executor.run_sandboxed(
                "print('ROWS_IN: 0'); print('ROWS_OUT: 0')",
                "/dev/null", str(tmp_path / "out.parquet"),
            )
            # Should have fallen back to subprocess (no crash)
            assert "success" in result


# ===========================================================================
# 3.4 — Stratified sampling: datasets > 50K rows use head+tail+random buckets
# ===========================================================================

class TestStratifiedSampling:
    @pytest.mark.asyncio
    async def test_small_dataset_uses_random_sample(self, tmp_path):
        """Datasets <= 50K rows should use plain random SAMPLE."""
        from mcp_server.tools.profiling_tools import sample_data, _STRATIFIED_THRESHOLD

        csv = tmp_path / "small.csv"
        n = 100  # well below threshold
        pd.DataFrame({"x": range(n), "y": ["a"] * n}).to_csv(csv, index=False)

        with patch("mcp_server.tools.profiling_tools._stratified_sample") as mock_strat:
            result = await sample_data(str(csv), "csv", sample_size=50)
            mock_strat.assert_not_called()

        assert result["actual_sample_size"] == 50

    @pytest.mark.asyncio
    async def test_large_dataset_calls_stratified_sample(self, tmp_path):
        """Datasets > 50K rows must call _stratified_sample, not plain SAMPLE."""
        from mcp_server.tools import profiling_tools

        csv = tmp_path / "large.csv"
        n = 60_000
        pd.DataFrame({"x": range(n)}).to_csv(csv, index=False)

        sentinel = [{"x": i} for i in range(2000)]
        with patch.object(profiling_tools, "_stratified_sample", return_value=sentinel) as mock_strat:
            result = await profiling_tools.sample_data(str(csv), "csv", sample_size=2000)
            mock_strat.assert_called_once()

        assert result["actual_sample_size"] == 2000

    def test_stratified_sample_returns_three_buckets_deduplicated(self, tmp_path):
        """_stratified_sample must return head + tail + random without duplicates."""
        from mcp_server.tools.profiling_tools import _stratified_sample
        import duckdb

        csv = tmp_path / "big.csv"
        n = 100
        pd.DataFrame({"id": range(n), "val": [f"v{i}" for i in range(n)]}).to_csv(csv, index=False)

        conn = duckdb.connect()
        conn.execute(f"CREATE VIEW src AS SELECT * FROM read_csv_auto('{csv}')")

        rows = _stratified_sample(conn, total=n, sample_size=30)
        conn.close()

        # Should have rows from head (0-9), tail (90-99), and random
        ids = {r["id"] for r in rows}
        # Head rows must be present
        assert 0 in ids, "Head bucket missing from stratified sample"
        # Tail rows must be present
        assert 99 in ids, "Tail bucket missing from stratified sample"
        # No duplicates
        assert len(rows) == len(ids), "Stratified sample contains duplicate rows"

    def test_stratified_threshold_is_50k(self):
        from mcp_server.tools.profiling_tools import _STRATIFIED_THRESHOLD
        assert _STRATIFIED_THRESHOLD == 50_000


# ===========================================================================
# Token efficiency — trim_messages clips long ReAct histories
# ===========================================================================

class TestTrimMessages:
    def test_short_list_returned_unchanged(self):
        from agents.utils import trim_messages
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(5)]
        result = trim_messages(msgs, keep_first=1, keep_last=10)
        assert result == msgs

    def test_long_list_keeps_first_and_last(self):
        from agents.utils import trim_messages
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        result = trim_messages(msgs, keep_first=1, keep_last=5)
        assert len(result) == 6
        assert result[0] == msgs[0]           # first preserved
        assert result[-1] == msgs[-1]         # last preserved
        assert result[1] == msgs[15]          # correct tail start

    def test_exactly_at_boundary_not_trimmed(self):
        from agents.utils import trim_messages
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(11)]
        result = trim_messages(msgs, keep_first=1, keep_last=10)
        assert result == msgs  # 1 + 10 == 11, no trim

    def test_middle_messages_dropped(self):
        from agents.utils import trim_messages
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(30)]
        result = trim_messages(msgs, keep_first=2, keep_last=4)
        assert len(result) == 6
        # Middle messages must NOT appear
        middle_contents = {m["content"] for m in result}
        for i in range(2, 26):
            assert f"msg {i}" not in middle_contents

    def test_domain_classify_uses_haiku(self):
        """_llm_classify must use Haiku, not Sonnet."""
        from mcp_server.tools.domain_tools import _llm_classify
        import inspect
        source = inspect.getsource(_llm_classify)
        assert "haiku" in source.lower(), (
            "_llm_classify must use Haiku model to reduce token cost"
        )

    def test_trim_messages_called_in_quality_agent_loop(self):
        """trim_messages must be called inside the quality agent ReAct loop."""
        import inspect
        import agents.quality_agent as qa
        source = inspect.getsource(qa.run_quality_agent)
        assert "trim_messages" in source, (
            "quality agent ReAct loop must call trim_messages each iteration"
        )

    def test_trim_messages_called_in_transformer_agent_loop(self):
        """trim_messages must be called inside the transformer agent ReAct loop."""
        import inspect
        import agents.transformer_agent as ta
        source = inspect.getsource(ta.run_transformer_agent)
        assert "trim_messages" in source, (
            "transformer agent ReAct loop must call trim_messages each iteration"
        )


# ===========================================================================
# Gap fixes — domain_context bug + boundary validator integration
# ===========================================================================

class TestDomainAgentContextFix:
    @pytest.mark.asyncio
    async def test_domain_context_is_full_rules_dict_not_empty(self):
        """
        domain_agent must set domain_context to the full rules dict.

        Pre-fix bug: rules_result.get('rules', {}) always returned {} because
        load_domain_rules returns a flat dict, not {'rules': {...}}.
        require_domain_output treats {} as falsy and would raise on every run.
        """
        from agents.domain_agent import run_domain_agent

        fake_rules = {
            "domain": "retail",
            "sensitive_columns": [],
            "required_transforms": ["dedup"],
            "forbidden_transforms": [],
            "validation_rules": {},
            "default_watermark_column": None,
            "detection_keywords": [],
        }

        mock_mcp = AsyncMock()
        mock_mcp.list_tools = AsyncMock(return_value=[])
        mock_mcp.call = AsyncMock(side_effect=[
            # detect_domain
            {"domain": "retail", "confidence": 0.85, "method": "keyword_heuristic"},
            # load_domain_rules — returns flat dict, NOT {"rules": {...}}
            fake_rules,
        ])
        mock_logger = MagicMock()
        mock_logger.agent_start = MagicMock(return_value=MagicMock())
        mock_logger.tool_call = MagicMock()
        mock_logger.agent_end = MagicMock()

        state = {
            "run_id": "test",
            "user_goal": "clean data",
            "source_path": "/tmp/x.csv",
            "source_type": "csv",
            "incremental_mode": False,
            "retry_count": 0,
            "status": "running",
            "profile": {"col": {"sample_values": ["item"]}},
            "schema": {"col": {"inferred_type": "VARCHAR"}},
            "sample": [{"col": "item"}],
        }

        result = await run_domain_agent(state, {"configurable": {"mcp": mock_mcp, "logger": mock_logger}})

        # domain_context must be the full rules dict, not {}
        assert result["domain_context"] == fake_rules
        assert result["domain_context"]["required_transforms"] == ["dedup"]
        assert result["domain"] == "retail"

    @pytest.mark.asyncio
    async def test_require_domain_output_passes_after_domain_agent_runs(self):
        """
        require_domain_output must not raise when domain_context is a non-empty
        dict (as returned by a correctly fixed domain_agent).
        """
        from orchestrator.state import require_domain_output, StateBoundaryError

        good_state = {
            "run_id": "t",
            "user_goal": "g",
            "source_path": "/tmp/x.csv",
            "source_type": "csv",
            "incremental_mode": False,
            "retry_count": 0,
            "status": "running",
            "domain_context": {
                "domain": "retail",
                "required_transforms": [],
                "forbidden_transforms": [],
            },
        }
        require_domain_output(good_state)  # must not raise

        empty_ctx_state = {**good_state, "domain_context": {}}
        with pytest.raises(StateBoundaryError):
            require_domain_output(empty_ctx_state)
