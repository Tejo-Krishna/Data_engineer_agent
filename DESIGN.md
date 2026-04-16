# Design decisions

This document explains the *why* behind each architectural choice. When a decision might look strange or over-engineered, the answer is here.

---

## 1. Five separate agents instead of one large ReAct loop

**Decision:** Split profiling, domain detection, transformation, quality, and cataloguing into five separate agent functions with explicit state handoffs.

**Why:** A single monolithic ReAct agent with all 25 tools available would spend LLM budget reasoning about which tool to call next at every step. Profiling and cataloguing are fully deterministic — they always call the same tools in the same order. Only the transformer and quality agents need reasoning. Separating them means:
- Profiler and catalogue agents never burn tokens on planning
- The transformer's ReAct loop has a short, focused tool list (6 tools)
- Failures in one agent don't corrupt state from earlier agents
- Each agent can be tested and replaced independently

---

## 2. MCP server as the tool interface

**Decision:** All tools are exposed via a local MCP (Model Context Protocol) server. Agents call tools via `mcp.call("tool_name", args)` rather than calling Python functions directly.

**Why:** This gives a consistent interface regardless of where the tool runs. The same agent code works whether the MCP server is in-process (DirectClient, used in tests and local runs) or over SSE (used when the server is a separate container). It also makes the tool contract explicit — every tool has a typed schema that the LLM sees, which improves tool selection accuracy.

---

## 3. TOOLS registry pattern — modules own their tool list

**Decision:** Each tool module exports a `TOOLS: dict` at the bottom. `server.py` builds `TOOL_HANDLERS` by merging all seven dicts rather than listing every function individually.

**Why:** With 25 tools across 7 files, manually keeping `TOOL_HANDLERS` in sync with function definitions was error-prone. The registry pattern means adding a tool requires one change only: add it to the module's `TOOLS` dict. `server.py` automatically picks it up. Tests verify this by checking `TOOL_HANDLERS` covers all `TOOL_DEFINITIONS`.

---

## 4. State boundary validators

**Decision:** `orchestrator/state.py` defines `require_profiler_output`, `require_domain_output`, `require_transformer_output`, `require_quality_output`. Each agent calls its validator as the first line.

**Why:** `PipelineState` is a flat TypedDict with 35 optional fields. Without guards, an agent can silently proceed with `None` upstream fields and produce confusing downstream errors ("NoneType has no attribute 'keys'" inside the quality report, three agents after the actual failure). Validators give a `StateBoundaryError` with a clear message at the boundary crossing, making root cause obvious immediately.

**Design constraint:** Validators check only what the calling agent actually reads — not every field the upstream agent wrote. Example: `require_profiler_output` checks `profile` and `schema` but not `sample`, because the domain agent doesn't read `sample`. This avoids false failures on edge cases like zero-row datasets where `sample` is `[]`.

---

## 5. Targeted repair before full code regeneration (3.3)

**Decision:** On retry, the transformer tries `refine_transform_code(failure_reason)` → `execute_code` → `write_dataset` directly before spinning up the full Sonnet ReAct loop.

**Why:** Most retry causes are narrow: a missing import, a null dereference, a type mismatch. These require a surgical one-line fix, not a fresh code generation from scratch. Full regeneration costs 3–4 Sonnet calls (planning turn + generate_transform_code with 4096 max tokens). Targeted repair costs one `refine_transform_code` call (much smaller output). The repair path is a fast-path that returns `None` on failure, so the full loop is always available as a fallback.

---

## 6. Semantic verification with Haiku (3.5)

**Decision:** After every successful `execute_code`, call `verify_transform_intent` which samples 5 rows from input and output parquets and asks a Haiku model "does this match the user's goal?"

**Why:** Execution success (`returncode == 0`, no stderr) doesn't guarantee semantic correctness. Code can run without errors but produce wrong output — dates parsed with the wrong format, currency columns not stripped, wrong column dropped. Semantic verification catches this class of bug before the output is written to permanent storage. Using Haiku (not Sonnet) with only 5 rows keeps the cost at ~$0.0001 per check.

**Trade-off:** A false negative (Haiku incorrectly says goal not matched) triggers an unnecessary retry. The prompt is conservative — it only fails on clear mismatches — and the issue list in the retry context helps the regeneration fix the right thing.

---

## 7. Docker sandbox with subprocess fallback (3.1)

**Decision:** `executor.py` uses `docker run --rm --network none` when `DOCKER_SANDBOX_IMAGE` is set, falling back to a clean subprocess when Docker is unavailable.

**Why:** The subprocess fallback (original implementation) wipes the environment but still runs as the same user with full filesystem access. Generated code could read `~/.ssh/id_rsa` or write outside `/tmp`. Docker with `--network none` provides genuine isolation: no network, no filesystem access outside the mounted input/output volumes. The fallback exists so the pipeline runs in CI and development environments that don't have Docker configured.

**Activation:** Set `DOCKER_SANDBOX_IMAGE=data-agent-sandbox:latest` in `.env`. The image must have `pandas` and `pyarrow` installed. When unset, subprocess is used and behaviour is identical to the pre-Docker implementation.

---

## 8. Stratified sampling for large datasets (3.4)

**Decision:** `sample_data` uses random `USING SAMPLE N ROWS` for datasets ≤ 50K rows. For larger datasets it takes first N/3 + last N/3 + random N/3.

**Why:** Pure random sampling on a 10M-row dataset will miss patterns that are concentrated at the boundaries — e.g., date formats that changed in the most recent data, or trailing nulls from a partial load. Head + tail + random ensures:
- Head: representative of historical data and header rows
- Tail: representative of recent/current data (most likely to have new quality issues)
- Random: unbiased coverage of the middle

The sample size stays constant at 2000 rows regardless of dataset size, so downstream token cost is identical.

---

## 9. Keyword heuristics before LLM for domain classification

**Decision:** `detect_domain` runs keyword matching first. Only if the best domain score is below threshold or the gap between first and second is too small does it call Claude Haiku.

**Why:** Domain classification is a 6-way classification over column names. For datasets with obvious signals ("patient_id", "diagnosis_code" → medical; "sku", "unit_price" → retail) this is trivially resolvable without an LLM. Calling Haiku for these wastes ~50ms and ~$0.00001 per call. The keyword path handles ~80% of cases in testing.

**Thresholds:** Score ≥ 0.30 OR ≥ 5 keyword hits, with a gap ≥ 0.15 over the second-best domain, triggers the heuristic path. These were tuned against the test suite.

---

## 10. Domain rules as hard constraints, not suggestions

**Decision:** Domain YAML files define `required_transforms`, `forbidden_transforms`, and `validation_rules`. These are injected into the code generation prompt as `HARD CONSTRAINTS` and are enforced in the tool's prompt engineering, not in post-generation validation.

**Why:** Post-generation enforcement (parse the code, check if it drops rows) is fragile and easily fooled. Encoding constraints into the generation prompt means the LLM refuses to generate violating code in the first place. The trade-off is that a sufficiently creative LLM could still violate the rules, but in practice the HARD CONSTRAINTS framing is reliable for the kinds of rules in these YAML files.

---

## 11. Context trimming in ReAct loops

**Decision:** Both transformer and quality agents call `trim_messages(keep_first=1, keep_last=10)` at the top of each ReAct loop iteration.

**Why:** A ReAct loop that runs for 6 iterations accumulates 12+ messages (initial context + alternating assistant/tool_result turns). For the transformer, the initial context message contains the full schema (potentially very large). Without trimming, by iteration 5 the prompt includes megabytes of redundant tool results from earlier iterations. Trimming keeps the context at a fixed size: initial goal + last 5 tool-call rounds.

**Trade-off:** The LLM loses visibility into early tool results once they fall outside the window. In practice this is fine because by iteration 3+ the LLM only needs to know what it most recently did, not what happened in iteration 1.

---

## 12. HITL checkpoint via Redis polling

**Decision:** The HITL checkpoint writes the generated code to Redis and polls for a human decision. Humans interact via `python scripts/hitl_approve.py {run_id}`.

**Why:** The pipeline runs asynchronously and the human may not be at the terminal when the checkpoint fires. Redis provides persistence across process restarts and a simple key-value interface that doesn't require a running web server. The polling timeout (configurable) makes the pipeline fail gracefully if the human never responds rather than hanging indefinitely.

---

## 13. Structured tool_use output, not text parsing

**Decision:** All LLM calls that need structured output (generate_transform_code, refine_transform_code, verify_transform_intent, detect_domain, explain_anomalies) use `tool_choice={"type": "tool", "name": "submit_..."}` to force structured JSON output. No fence-stripping or `json.loads` on free text.

**Why:** Text parsing is fragile. Markdown fences change, models sometimes add preamble, JSON keys get quoted differently. Tool use forces a typed schema on the output and Anthropic validates it server-side before returning the response. The result is `message.content[0].input` — a dict, always, with the keys specified in the schema.
