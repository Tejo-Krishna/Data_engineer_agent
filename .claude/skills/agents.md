# Skill: Agents

Read this before writing or editing any agent file.

---

## Agent function signature — mandatory pattern

Every agent function has this exact signature. No exceptions.

```python
async def run_[agent_name]_agent(
    state: PipelineState,
    mcp: MCPClient,
    logger: RunLogger          # or Langfuse span when observability added
) -> PipelineState:
```

The function always returns a complete updated PipelineState using the
`{**state, "new_field": value}` spread pattern. Never mutate state in
place. Never return a partial state — always spread the full prior state
first.

```python
# Correct
return {**state, "profile": profile, "schema": schema}

# Wrong — loses all other state fields
return {"profile": profile, "schema": schema}
```

---

## Logging pattern — mandatory in every agent

Call the logger at the start of the agent, after every tool call, and
at the end of the agent. No silent execution.

```python
async def run_profiler_agent(state, mcp, logger):
    logger.agent_start("profiler")

    result = await mcp.call("sample_data", {"source_path": state["source_path"]})
    logger.tool_call(
        agent="profiler",
        tool="sample_data",
        input_summary={"source": state["source_path"]},
        output_summary={"rows": len(result.get("sample", []))}
    )

    # ... more tool calls ...

    logger.agent_end("profiler")
    return {**state, ...}
```

Keep input_summary and output_summary small. Log counts and keys, not
raw data. A summary like `{"rows": 487, "columns": 12}` is correct.
A summary that dumps the entire sample is wrong.

---

## Fixed-sequence agents (profiler, catalogue)

These agents do not use ReAct. They call tools in a predetermined order
because their work is deterministic — the right next step does not depend
on the result of the previous step.

**Profiler agent — exactly 5 tool calls in this order:**
1. `connect_csv` / `connect_postgres` / `connect_api` (based on source_type)
2. `sample_data`
3. `compare_schemas` — schema drift detection, runs on EVERY execution
4. `compute_profile`
5. `detect_schema`

```python
async def run_profiler_agent(state, mcp, logger):
    logger.agent_start("profiler")

    # Tool call 1: connect
    connect_tool = {
        "csv": "connect_csv", "parquet": "connect_csv",
        "postgres": "connect_postgres", "api": "connect_api"
    }[state["source_type"]]
    metadata = await mcp.call(connect_tool, {"file_path": state["source_path"]})
    logger.tool_call("profiler", connect_tool, {"source": state["source_path"]},
                     {"columns": len(metadata.get("column_names", []))})

    # Tool call 2: sample
    sample = await mcp.call("sample_data", {
        "source_path": state["source_path"],
        "source_type": state["source_type"],
        "sample_size": 2000
    })
    logger.tool_call("profiler", "sample_data", {}, {"rows": len(sample["sample"])})

    # Tool call 3: schema drift — ALWAYS runs
    drift = await mcp.call("compare_schemas", {
        "source_path": state["source_path"],
        "current_schema": sample["sample"][:5]  # pass sample for inference
    })
    logger.tool_call("profiler", "compare_schemas", {},
                     {"has_drift": drift.get("has_drift"), "severity": drift.get("drift_severity")})

    # Tool call 4: profile
    profile = await mcp.call("compute_profile", {"sample": sample["sample"]})
    logger.tool_call("profiler", "compute_profile", {}, {"columns": len(profile)})

    # Tool call 5: schema
    schema = await mcp.call("detect_schema", {
        "sample": sample["sample"], "profile": profile
    })
    logger.tool_call("profiler", "detect_schema", {}, {"columns": len(schema)})

    logger.agent_end("profiler")
    return {
        **state,
        "source_metadata": metadata,
        "sample": sample["sample"],
        "schema_drift": drift,
        "profile": profile,
        "schema": schema
    }
```

**Catalogue agent — exactly 5 tool calls in this order:**
1. `write_catalogue_entry`
2. `generate_lineage_graph`
3. `generate_dbt_model`
4. `generate_dbt_schema_yml`
5. `generate_dbt_tests` (from quality_checks results)

---

## ReAct agents (transformer, quality)

These agents use the ReAct (Reasoning and Acting) loop. The LLM reasons
about what to do next based on the current state and tool results, rather
than following a fixed sequence.

**Maximum tool calls per ReAct agent:**
- Transformer: 6 (including HITL checkpoint as one step)
- Quality: 6

The loop terminates when the agent either reaches its conclusion or hits
the maximum. Never let a ReAct loop run without a bound.

**ReAct loop pattern:**

```python
async def run_transformer_agent(state, mcp, logger):
    logger.agent_start("transformer")

    messages = [
        {"role": "system", "content": TRANSFORMER_SYSTEM_PROMPT},
        {"role": "user", "content": build_transformer_context(state)}
    ]

    tool_call_count = 0
    MAX_CALLS = 6

    while tool_call_count < MAX_CALLS:
        response = await call_llm_with_tools(messages, available_tools=[
            "detect_new_rows",      # only if incremental_mode
            "search_transform_library",
            "generate_transform_code",
            "refine_transform_code",
            "execute_code",
            "write_dataset"
        ])

        if response.stop_reason == "end_turn":
            break

        for tool_call in response.tool_calls:
            result = await mcp.call(tool_call.name, tool_call.input)
            logger.tool_call("transformer", tool_call.name,
                           _summarise(tool_call.input),
                           _summarise(result))
            messages.append(tool_result_message(tool_call.id, result))
            tool_call_count += 1

            # HITL fires after generate_transform_code on first run
            if tool_call.name == "generate_transform_code" and state["retry_count"] == 0:
                approval = await hitl_checkpoint(state["run_id"],
                                                result["code"],
                                                result["transformations_applied"])
                logger.tool_call("transformer", "hitl_checkpoint", {},
                               {"approved": approval["approved"],
                                "has_instruction": bool(approval.get("nlp_instruction"))})
                # inject approval into messages so LLM knows what to do next
                messages.append({"role": "user",
                               "content": f"HITL result: {json.dumps(approval)}"})

    logger.agent_end("transformer")
    return {**state, ...}
```

---

## System prompts for ReAct agents

**Transformer system prompt:**

```python
TRANSFORMER_SYSTEM_PROMPT = """You are a data engineering code generator.

You have access to tools for searching reusable code, generating Python
transformation scripts, refining code based on instructions, executing
code in a sandbox, and writing the output dataset.

Domain context for this dataset will be provided. Domain rules are
HARD CONSTRAINTS — they override the user's goal if they conflict.
Never generate code that violates domain rules even if the user asks.

Loop:
1. If incremental_mode is true, call detect_new_rows first
2. Search the transformation library for reusable code
3. Generate the transformation script — use library snippets if relevant
4. After HITL approval (provided in context), execute the code
5. If execution fails, reason about the error and fix the code
6. Write the dataset when execution succeeds

Generated code must:
- Read from INPUT_PATH env var, write to OUTPUT_PATH env var
- Print ROWS_IN: N before processing
- Print ROWS_OUT: N after processing
- Handle all null values gracefully — never crash on nulls
- Be a complete self-contained script"""
```

**Quality system prompt:**

```python
QUALITY_SYSTEM_PROMPT = """You are a data quality analyst.

You have access to tools for checking quality rules, detecting anomalies,
explaining anomalies in plain English, writing quality reports, and saving
successful transformations to the library.

Reason about which checks are most important for THIS dataset before
calling any tools. Do not always run the same checks in the same order.
Look at the profile — columns with high null rates or wide value ranges
deserve targeted attention.

Rules:
- Run run_quality_checks first
- Run detect_anomalies only on columns that failed or are borderline
- If anomaly_count > 0, ALWAYS call explain_anomalies
- Call save_to_library ONLY if quality_passed is True AND the
  transformation is generic enough to be reusable
- Set state["status"] to "success", "retrying", or "failed" before
  returning. This is how the LangGraph router makes its decision."""
```

---

## Domain detection agent — pattern

This agent is small. It has a predictable two-step flow.

```python
async def run_domain_agent(state, mcp, logger):
    logger.agent_start("domain")

    # Step 1: detect domain
    domain_result = await mcp.call("detect_domain", {
        "column_names": list(state["schema"].keys()),
        "sample_values": {k: v["sample_values"]
                         for k, v in state["profile"].items()},
        "user_goal": state["user_goal"]
    })
    logger.tool_call("domain", "detect_domain", {},
                     {"domain": domain_result["domain"],
                      "confidence": domain_result["confidence"]})

    # Step 2: load rules
    rules = await mcp.call("load_domain_rules", {
        "domain": domain_result["domain"]
    })
    logger.tool_call("domain", "load_domain_rules",
                     {"domain": domain_result["domain"]},
                     {"rules_loaded": bool(rules.get("rules"))})

    logger.agent_end("domain")
    return {
        **state,
        "domain": domain_result["domain"],
        "domain_confidence": domain_result["confidence"],
        "domain_context": rules.get("rules", {})
    }
```

---

## Error and retry handling

The transformer and quality agents communicate failure through state,
not through exceptions. Never raise an exception from an agent to signal
a retryable failure.

```python
# Correct — communicate through state
if not exec_result["success"]:
    return {
        **state,
        "status": "retrying",
        "failure_reason": f"Execution error: {exec_result['stderr'][:500]}",
        "retry_count": state["retry_count"] + 1
    }

# Wrong — never do this
raise RuntimeError("Execution failed")
```

The LangGraph router reads `state["status"]` after the quality agent
exits and routes accordingly. The retry count is checked in the router,
not in the agent.

On retry, the transformer receives `state["failure_reason"]`. It must
append this to the code generation prompt:

```python
goal = state["user_goal"]
if state["retry_count"] > 0 and state.get("failure_reason"):
    goal += f"\n\nPREVIOUS ATTEMPT FAILED:\n{state['failure_reason']}\n"
    goal += "Fix this specific issue. Do not change parts of the code that worked."
```
